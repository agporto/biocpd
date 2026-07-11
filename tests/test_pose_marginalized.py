import numpy as np
from scipy.spatial.transform import Rotation

from biocpd import (
    AtlasRegistration,
    PoseMarginalizedConfig,
    PoseMarginalizedInitialization,
    pose_marginalized_initialization,
)
from biocpd.pose_marginalized import _rotation_lattice


def _asymmetric_cloud(seed=4, count=60):
    generator = np.random.default_rng(seed)
    points = generator.normal(size=(count, 3))
    points *= np.array([2.0, 1.2, 0.7])
    points[:, 0] += 0.2 * points[:, 1] ** 2
    return points


def test_legacy_atlas_parameter_contract_is_unchanged():
    source = _asymmetric_cloud(count=20)
    registration = AtlasRegistration(
        X=source.copy(),
        Y=source,
        U=np.zeros((source.size, 1)),
        eigenvalues=np.ones(1),
        max_iterations=1,
    )
    _, parameters = registration.register()
    assert set(parameters) == {
        "U_flat",
        "b",
        "R",
        "s",
        "t",
        "R_norm",
        "s_norm",
        "t_norm",
        "R_world",
        "s_world",
        "t_world",
    }
    assert np.isfinite(registration.registration_diagnostics()["objective"])


def test_initial_similarity_uses_world_units_with_normalization():
    source = _asymmetric_cloud(seed=3, count=20)
    rotation = Rotation.from_euler("z", 27.0, degrees=True).as_matrix()
    scale = 1.3
    translation = np.array([[4.0, -2.0, 0.5]])
    target = scale * (source @ rotation.T) + translation
    registration = AtlasRegistration(
        X=target,
        Y=source,
        mean_shape=source,
        U=np.zeros((source.size, 1)),
        eigenvalues=np.ones(1),
        normalize=True,
        optimize_similarity=True,
        dtype=np.float64,
        max_iterations=1,
    )
    registration.set_initial_similarity(
        rotation, scale, translation, world_units=True
    )
    np.testing.assert_allclose(registration.TY_world, target, atol=1e-10)


def test_initial_state_applies_shape_before_similarity():
    source = _asymmetric_cloud(seed=11, count=20)
    modes = np.zeros((source.size, 2))
    modes[::3, 0] = np.linspace(-0.2, 0.2, len(source))
    modes[1::3, 1] = 0.1
    coefficients = np.array([0.7, -0.3])
    deformed = source + (modes @ coefficients).reshape(source.shape)
    rotation = Rotation.from_euler(
        "xyz", [12.0, -8.0, 21.0], degrees=True
    ).as_matrix()
    scale = 1.15
    translation = np.array([[2.0, -3.0, 0.4]])
    target = scale * (deformed @ rotation.T) + translation
    registration = AtlasRegistration(
        X=target,
        Y=source,
        mean_shape=source,
        U=modes,
        eigenvalues=np.array([0.5, 0.25]),
        normalize=True,
        dtype=np.float64,
        max_iterations=1,
    )
    registration.set_initial_state(
        coefficients, rotation, scale, translation, world_units=True
    )
    np.testing.assert_allclose(registration.TY_world, target, atol=1e-10)
    np.testing.assert_allclose(registration.b.ravel(), coefficients)


def test_rotation_lattice_is_deterministic_and_proper():
    first = _rotation_lattice(24, 13)
    second = _rotation_lattice(24, 13)
    assert len(first) == len(second)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
        np.testing.assert_allclose(left.T @ left, np.eye(3), atol=1e-12)
        assert np.linalg.det(left) > 0
    np.testing.assert_array_equal(first[0], np.eye(3))


def test_config_preserves_function_api_and_result_type():
    source = _asymmetric_cloud(seed=8, count=30)
    modes = np.zeros((source.size, 1))
    config = PoseMarginalizedConfig(
        rotation_count=12,
        coarse_source_count=20,
        coarse_target_count=20,
        coarse_rank=1,
        coarse_iterations=2,
        refine_count=2,
        refine_target_count=30,
        refine_iterations=2,
        seed=5,
    )
    result = config.initialize(source, source.copy(), modes, np.ones(1))
    assert isinstance(result, PoseMarginalizedInitialization)
    assert result.hypotheses_evaluated >= 12
    assert np.isfinite(result.score)


def test_pose_initializer_recovers_large_rotation():
    source = _asymmetric_cloud(seed=19, count=80)
    modes = np.zeros((source.size, 2))
    modes[::3, 0] = 0.05 * source[:, 0]
    modes[1::3, 1] = 0.04 * source[:, 1]
    target = 1.08 * (
        source
        @ Rotation.from_euler(
            "xyz", [36.0, -24.0, 31.0], degrees=True
        ).as_matrix().T
    ) + np.array([0.8, -0.4, 0.2])
    result = pose_marginalized_initialization(
        source,
        target,
        modes,
        np.array([0.4, 0.2]),
        rotation_count=24,
        coarse_source_count=60,
        coarse_target_count=60,
        coarse_rank=2,
        coarse_iterations=4,
        refine_count=4,
        refine_target_count=80,
        refine_iterations=8,
        seed=7,
    )
    recovered = (
        result.scale
        * (
            (
                source
                + (modes @ result.coefficients).reshape(source.shape)
            )
            @ result.rotation.T
        )
        + result.translation
    )
    diagonal = np.linalg.norm(np.ptp(target, axis=0))
    normalized_error = (
        np.median(np.linalg.norm(recovered - target, axis=1)) / diagonal
    )
    assert normalized_error < 0.05
