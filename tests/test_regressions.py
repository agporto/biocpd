import numpy as np
import pytest

from biocpd import AffineRegistration, AtlasRegistration, RigidRegistration


def test_sparse_k_greater_than_n_no_crash_rigid_and_affine():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(5, 3))
    Y = rng.normal(size=(7, 3))

    rigid = RigidRegistration(X=X, Y=Y, use_kdtree=True, k=10, max_iterations=2)
    TY_rigid, _ = rigid.register()
    assert TY_rigid.shape == Y.shape
    assert rigid.k == X.shape[0]

    affine = AffineRegistration(X=X, Y=Y, use_kdtree=True, k=10, max_iterations=2)
    TY_affine, _ = affine.register()
    assert TY_affine.shape == Y.shape
    assert affine.k == X.shape[0]


def test_rigid_accepts_valid_rotation_matrix():
    rng = np.random.default_rng(321)
    X = rng.normal(size=(20, 2))
    Y = rng.normal(size=(20, 2))
    R = np.array([[0.0, -1.0], [1.0, 0.0]])

    reg = RigidRegistration(X=X, Y=Y, R=R, max_iterations=2, use_kdtree=False)
    TY, _ = reg.register()

    assert TY.shape == Y.shape


def test_integer_inputs_are_supported_via_float_cast():
    X = np.array([[0, 0], [1, 0], [0, 1]], dtype=int)
    Y = np.array([[0, 0], [1, 1], [1, 0]], dtype=int)

    reg = RigidRegistration(X=X, Y=Y, max_iterations=2, use_kdtree=False)
    TY, _ = reg.register()

    assert TY.shape == Y.shape
    assert np.issubdtype(reg.X.dtype, np.floating)
    assert np.issubdtype(reg.Y.dtype, np.floating)


def _atlas_inputs(seed=11):
    rng = np.random.default_rng(seed)
    M, D, K = 20, 3, 4
    X = rng.normal(size=(M, D))
    Y = rng.normal(size=(M, D))
    U = rng.normal(size=(M * D, K))
    L = np.abs(rng.normal(size=(K,))) + 1e-3
    return X, Y, U, L


def test_atlas_rejects_zero_kdtree_radius_scale():
    X, Y, U, L = _atlas_inputs()
    with pytest.raises(ValueError, match="kdtree_radius_scale must be a positive number"):
        AtlasRegistration(
            X=X,
            Y=Y,
            U=U,
            eigenvalues=L,
            kdtree_radius_scale=0.0,
            max_iterations=1,
        )


def test_atlas_rejects_negative_lambda_reg():
    X, Y, U, L = _atlas_inputs()
    with pytest.raises(ValueError, match="lambda_reg must be non-negative"):
        AtlasRegistration(
            X=X,
            Y=Y,
            U=U,
            eigenvalues=L,
            lambda_reg=-1.0,
            max_iterations=1,
        )


def test_atlas_requires_u():
    X, Y, _, L = _atlas_inputs()
    with pytest.raises(ValueError, match="U must be provided"):
        AtlasRegistration(
            X=X,
            Y=Y,
            U=None,
            eigenvalues=L,
            max_iterations=1,
        )


def test_atlas_normalize_requires_matrix_mean_shape():
    X, Y, U, L = _atlas_inputs()
    with pytest.raises(ValueError, match="normalize=True, mean_shape must be provided as shape"):
        AtlasRegistration(
            X=X,
            Y=Y,
            mean_shape=Y.reshape(-1),
            U=U,
            eigenvalues=L,
            normalize=True,
            max_iterations=1,
        )


def test_atlas_normalized_matrix_mean_shape_runs():
    X, Y, U, L = _atlas_inputs()
    reg = AtlasRegistration(
        X=X,
        Y=Y,
        mean_shape=Y.copy(),
        U=U,
        eigenvalues=L,
        normalize=True,
        max_iterations=2,
    )
    TY, _ = reg.register()
    assert TY.shape == Y.shape
