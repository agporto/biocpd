import numpy as np
from scipy.spatial.transform import Rotation

from biocpd import (
    AffineRegistration,
    AtlasRegistration,
    ConstrainedDeformableRegistration,
    DeformableRegistration,
    RigidRegistration,
)
from biocpd.utility import gaussian_kernel


def _asymmetric_cloud(seed, count):
    generator = np.random.default_rng(seed)
    points = generator.normal(size=(count, 3))
    points *= np.array([1.7, 0.9, 0.5])
    points[:, 0] += 0.15 * points[:, 1] ** 2
    return points


def _assert_converged(registration, tolerance, max_iterations):
    assert 1 < registration.iteration < max_iterations
    assert np.isfinite(registration.diff)
    assert registration.diff <= tolerance


def test_rigid_converges_recovers_transform_and_is_backend_stable():
    source = _asymmetric_cloud(1001, 60)
    rotation = Rotation.from_euler(
        "xyz", [8.0, -6.0, 10.0], degrees=True
    ).as_matrix()
    scale = 1.06
    translation = np.array([[0.18, -0.12, 0.08]])
    target = scale * (source @ rotation) + translation
    tolerance = 1e-5
    max_iterations = 200
    common = dict(
        X=target,
        Y=source,
        scale=True,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    dense = RigidRegistration(**common, use_kdtree=False)
    sparse = RigidRegistration(**common, use_kdtree=True, k=len(target))
    reference = RigidRegistration(
        **{**common, "tolerance": 1e-9},
        use_kdtree=False,
    )

    dense_points, _ = dense.register()
    sparse_points, _ = sparse.register()
    reference_points, _ = reference.register()

    _assert_converged(dense, tolerance, max_iterations)
    _assert_converged(sparse, tolerance, max_iterations)
    _assert_converged(reference, 1e-9, max_iterations)
    assert np.sqrt(np.mean((dense_points - target) ** 2)) < 1e-10
    np.testing.assert_allclose(dense.R, rotation, atol=1e-10, rtol=1e-10)
    assert np.isclose(dense.s, scale, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(dense.t).reshape(1, 3), translation, atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(sparse_points, dense_points, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(reference_points, dense_points, atol=1e-10, rtol=1e-10)


def test_affine_converges_recovers_transform_and_is_backend_stable():
    source = _asymmetric_cloud(1002, 60)
    transform = np.array(
        [[1.03, 0.02, 0.0], [0.02, 0.98, 0.0], [0.0, 0.0, 1.01]]
    )
    translation = np.array([[0.08, -0.05, 0.03]])
    target = source @ transform + translation
    tolerance = 1e-7
    max_iterations = 250
    common = dict(
        X=target,
        Y=source,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    dense = AffineRegistration(**common, use_kdtree=False)
    sparse = AffineRegistration(**common, use_kdtree=True, k=len(target))
    reference = AffineRegistration(
        **{**common, "tolerance": 1e-10},
        use_kdtree=False,
    )

    dense_points, _ = dense.register()
    sparse_points, _ = sparse.register()
    reference_points, _ = reference.register()

    _assert_converged(dense, tolerance, max_iterations)
    _assert_converged(sparse, tolerance, max_iterations)
    _assert_converged(reference, 1e-10, max_iterations)
    assert np.sqrt(np.mean((dense_points - target) ** 2)) < 1e-8
    np.testing.assert_allclose(dense.B, transform, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        np.asarray(dense.t).reshape(1, 3), translation, atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(sparse_points, dense_points, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(reference_points, dense_points, atol=1e-8, rtol=1e-8)


def _deformable_problem(seed):
    generator = np.random.default_rng(seed)
    source = generator.normal(size=(36, 3)) * np.array([1.2, 0.8, 0.6])
    beta = 1.5
    weights = 0.015 * generator.normal(size=source.shape)
    target = source + gaussian_kernel(source, beta) @ weights
    return source, target, beta


def test_deformable_converges_recovers_shape_and_is_backend_stable():
    source, target, beta = _deformable_problem(2002)
    tolerance = 1e-6
    max_iterations = 200
    common = dict(
        X=target,
        Y=source,
        alpha=2.0,
        beta=beta,
        low_rank=False,
        dtype=np.float64,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    dense = DeformableRegistration(**common, use_kdtree=False)
    sparse = DeformableRegistration(**common, use_kdtree=True, k=len(target))
    reference = DeformableRegistration(
        **{**common, "tolerance": 1e-8},
        use_kdtree=False,
    )

    dense_points, _ = dense.register()
    sparse_points, _ = sparse.register()
    reference_points, _ = reference.register()

    _assert_converged(dense, tolerance, max_iterations)
    _assert_converged(sparse, tolerance, max_iterations)
    _assert_converged(reference, 1e-8, max_iterations)
    assert np.sqrt(np.mean((dense_points - target) ** 2)) < 1e-8
    np.testing.assert_allclose(sparse_points, dense_points, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(reference_points, dense_points, atol=1e-8, rtol=1e-8)


def test_constrained_deformable_converges_and_is_backend_stable():
    source, target, beta = _deformable_problem(2003)
    correspondence_ids = np.arange(0, len(source), 6)
    tolerance = 1e-6
    max_iterations = 200
    common = dict(
        X=target,
        Y=source,
        source_id=correspondence_ids,
        target_id=correspondence_ids,
        e_alpha=0.05,
        alpha=2.0,
        beta=beta,
        low_rank=False,
        dtype=np.float64,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    dense = ConstrainedDeformableRegistration(**common, use_kdtree=False)
    sparse = ConstrainedDeformableRegistration(
        **common,
        use_kdtree=True,
        k=len(target),
    )
    reference = ConstrainedDeformableRegistration(
        **{**common, "tolerance": 1e-8},
        use_kdtree=False,
    )

    dense_points, _ = dense.register()
    sparse_points, _ = sparse.register()
    reference_points, _ = reference.register()

    _assert_converged(dense, tolerance, max_iterations)
    _assert_converged(sparse, tolerance, max_iterations)
    _assert_converged(reference, 1e-8, max_iterations)
    assert np.sqrt(np.mean((dense_points - target) ** 2)) < 1e-8
    np.testing.assert_allclose(sparse_points, dense_points, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(reference_points, dense_points, atol=1e-8, rtol=1e-8)


def _atlas_problem():
    generator = np.random.default_rng(3003)
    point_count, dimension, rank = 45, 3, 4
    source = generator.normal(size=(point_count, dimension))
    source *= np.array([1.4, 0.8, 0.5])
    orthogonal, _ = np.linalg.qr(
        generator.normal(size=(point_count * dimension, rank))
    )
    modes = orthogonal * np.array([0.5, 0.4, 0.3, 0.2])
    eigenvalues = np.array([0.5, 0.3, 0.2, 0.1])
    coefficients = np.array([0.35, -0.2, 0.15, 0.1])
    target = source + (modes @ coefficients).reshape(source.shape)
    return source, target, modes, eigenvalues, coefficients


def test_atlas_converges_recovers_coefficients_and_is_backend_stable():
    source, target, modes, eigenvalues, coefficients = _atlas_problem()
    tolerance = 1e-4
    max_iterations = 200
    common = dict(
        X=target,
        Y=source,
        U=modes,
        eigenvalues=eigenvalues,
        lambda_reg=0.001,
        optimize_similarity=False,
        dtype=np.float64,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    dense = AtlasRegistration(**common, use_kdtree=False)
    sparse = AtlasRegistration(
        **common,
        use_kdtree=True,
        k=len(target),
        kdtree_radius_scale=1e-6,
    )
    reference = AtlasRegistration(
        **{**common, "tolerance": 1e-8},
        use_kdtree=False,
    )

    dense_points, _ = dense.register()
    sparse_points, _ = sparse.register()
    reference_points, _ = reference.register()

    _assert_converged(dense, tolerance, max_iterations)
    _assert_converged(sparse, tolerance, max_iterations)
    _assert_converged(reference, 1e-8, max_iterations)
    assert sparse._use_sparse is True
    assert np.sqrt(np.mean((dense_points - target) ** 2)) < 1e-7
    np.testing.assert_allclose(
        dense.b.reshape(-1), coefficients, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(sparse_points, dense_points, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(reference_points, dense_points, atol=1e-7, rtol=1e-7)
