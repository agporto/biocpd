import numpy as np
import pytest

from biocpd import (
    AffineRegistration,
    AtlasRegistration,
    ConstrainedDeformableRegistration,
    DeformableRegistration,
    RigidRegistration,
)
from biocpd.emregistration import initialize_sigma2


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


def _bruteforce_sigma2(X, Y):
    diff = X[None, :, :] - Y[:, None, :]
    return float(np.sum(diff * diff) / (X.shape[1] * X.shape[0] * Y.shape[0]))


def _dtype_tolerance(dtype):
    dtype = np.dtype(dtype)
    return 1e-6 if dtype == np.float32 else 1e-12


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


def test_initialize_sigma2_matches_bruteforce():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(9, 3))
    Y = rng.normal(size=(7, 3))
    sigma_fast = initialize_sigma2(X, Y)
    sigma_ref = _bruteforce_sigma2(X, Y)
    assert np.isclose(sigma_fast, sigma_ref, atol=1e-12, rtol=1e-12)


def test_atlas_dense_blocked_stats_match_reference():
    X, Y, U, L = _atlas_inputs(seed=17)
    reg = AtlasRegistration(
        X=X,
        Y=Y,
        U=U,
        eigenvalues=L,
        normalize=True,
        use_kdtree=False,
        store_posterior=False,
        dense_block_size=4,
        max_iterations=1,
    )
    reg.transform_point_cloud()
    reg._compute_dense_posterior_stats(store_p=False, block_size=3)

    diff2 = np.sum((reg.X[None, :, :] - reg.TY[:, None, :]) ** 2, axis=2)
    P = np.exp(-diff2 / (2 * reg.sigma2))
    c = (2 * np.pi * reg.sigma2) ** (reg.D / 2) * reg.w / (1 - reg.w) * reg.M / reg.N
    den = np.sum(P, axis=0, keepdims=True) + c
    P /= den

    Pt1_ref = np.sum(P, axis=0)
    P1_ref = np.sum(P, axis=1)
    PX_ref = P @ reg.X
    Np_ref = float(np.sum(P1_ref))
    tol = _dtype_tolerance(reg.X.dtype)

    assert reg.P is None
    assert np.allclose(reg.Pt1, Pt1_ref, atol=tol, rtol=tol)
    assert np.allclose(reg.P1, P1_ref, atol=tol, rtol=tol)
    assert np.allclose(reg.PX, PX_ref, atol=tol, rtol=tol)
    assert np.isclose(reg.Np, Np_ref, atol=tol, rtol=tol)


def test_atlas_default_dense_block_size_targets_5000():
    rng = np.random.default_rng(5)
    M, D, K = 6000, 3, 2
    X = rng.normal(size=(M, D))
    Y = rng.normal(size=(M, D))
    U = rng.normal(size=(M * D, K))
    L = np.abs(rng.normal(size=(K,))) + 1e-3

    reg = AtlasRegistration(
        X=X,
        Y=Y,
        U=U,
        eigenvalues=L,
        use_kdtree=False,
        dense_block_size=None,
        max_iterations=1,
    )
    assert reg._get_dense_block_size() == 5000


def test_atlas_float32_dtype_is_preserved_internals():
    X, Y, U, L = _atlas_inputs(seed=23)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    U = U.astype(np.float32)
    L = L.astype(np.float32)

    reg = AtlasRegistration(
        X=X,
        Y=Y,
        U=U,
        eigenvalues=L,
        dtype=np.float32,
        use_kdtree=False,
        max_iterations=2,
    )
    TY, _ = reg.register()

    assert TY.dtype == np.float32
    assert reg.X.dtype == np.float32
    assert reg.Y.dtype == np.float32
    assert reg.U_flat.dtype == np.float32
    assert reg.L.dtype == np.float32


def test_atlas_sparse_estep_preserves_float32_dtype():
    X, Y, U, L = _atlas_inputs(seed=123)
    reg = AtlasRegistration(
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        U=U.astype(np.float32),
        eigenvalues=L.astype(np.float32),
        dtype=np.float32,
        use_kdtree=True,
        k=4,
        kdtree_radius_scale=1e-6,
        max_iterations=1,
    )
    reg.transform_point_cloud()
    reg.expectation()

    assert reg._use_sparse is True
    assert reg.P1.dtype == np.float32
    assert reg.Pt1.dtype == np.float32
    assert reg.PX.dtype == np.float32


def test_atlas_sparse_full_neighbors_matches_dense_pipeline_accuracy():
    X, Y, U, L = _atlas_inputs(seed=202)
    kwargs = dict(
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        U=U.astype(np.float32),
        eigenvalues=L.astype(np.float32),
        dtype=np.float32,
        max_iterations=6,
        tolerance=1e-8,
    )

    dense = AtlasRegistration(
        **kwargs,
        use_kdtree=False,
    )
    TY_dense, _ = dense.register()

    sparse = AtlasRegistration(
        **kwargs,
        use_kdtree=True,
        k=kwargs["X"].shape[0],
        kdtree_radius_scale=1e-6,
        radius_mode=False,
    )
    TY_sparse, _ = sparse.register()

    assert sparse._use_sparse is True
    assert np.allclose(TY_sparse, TY_dense, atol=5e-5, rtol=5e-5)
    assert np.isclose(sparse.sigma2, dense.sigma2, atol=5e-6, rtol=5e-5)


def test_deformable_defaults_to_float32_dtype():
    rng = np.random.default_rng(24)
    X = rng.normal(size=(12, 3))
    Y = rng.normal(size=(12, 3))

    reg = DeformableRegistration(
        X=X,
        Y=Y,
        low_rank=False,
        use_kdtree=False,
        max_iterations=1,
    )

    assert reg.X.dtype == np.float32
    assert reg.Y.dtype == np.float32
    assert reg.W.dtype == np.float32


def test_constrained_deformable_defaults_to_float32_dtype():
    rng = np.random.default_rng(25)
    X = rng.normal(size=(12, 3))
    Y = rng.normal(size=(12, 3))

    reg = ConstrainedDeformableRegistration(
        X=X,
        Y=Y,
        source_id=np.array([0, 2, 4], dtype=int),
        target_id=np.array([0, 2, 4], dtype=int),
        low_rank=False,
        use_kdtree=False,
        max_iterations=1,
    )

    assert reg.X.dtype == np.float32
    assert reg.Y.dtype == np.float32
    assert reg.W.dtype == np.float32
    assert reg.P_tilde.dtype == np.float32


def test_atlas_defaults_to_float32_dtype():
    X, Y, U, L = _atlas_inputs(seed=26)

    reg = AtlasRegistration(
        X=X,
        Y=Y,
        U=U,
        eigenvalues=L,
        use_kdtree=False,
        max_iterations=1,
    )

    assert reg.X.dtype == np.float32
    assert reg.Y.dtype == np.float32
    assert reg.U_flat.dtype == np.float32
    assert reg.L.dtype == np.float32


def test_deformable_dense_stats_do_not_store_posterior_matrix():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(9, 3))
    Y = rng.normal(size=(7, 3))

    reg = DeformableRegistration(
        X=X,
        Y=Y,
        low_rank=False,
        use_kdtree=False,
        max_iterations=1,
    )
    reg.transform_point_cloud()
    reg.expectation()

    diff2 = np.sum((reg.X[None, :, :] - reg.TY[:, None, :]) ** 2, axis=2)
    P = np.exp(-diff2 / (2 * reg.sigma2))
    c = (2 * np.pi * reg.sigma2) ** (reg.D / 2) * reg.w / (1 - reg.w) * reg.M / reg.N
    den = np.sum(P, axis=0, keepdims=True) + c
    P /= den
    tol = _dtype_tolerance(reg.X.dtype)

    assert reg.P is None
    assert np.allclose(reg.Pt1, np.sum(P, axis=0), atol=tol, rtol=tol)
    assert np.allclose(reg.P1, np.sum(P, axis=1), atol=tol, rtol=tol)
    assert np.allclose(reg.PX, P @ reg.X, atol=tol, rtol=tol)
    assert np.isclose(reg.Np, np.sum(P), atol=tol, rtol=tol)


def test_constrained_deformable_dense_expectation_does_not_store_posterior_matrix():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(8, 3))
    Y = rng.normal(size=(8, 3))

    reg = ConstrainedDeformableRegistration(
        X=X,
        Y=Y,
        source_id=np.array([0, 3, 5], dtype=int),
        target_id=np.array([0, 3, 5], dtype=int),
        low_rank=False,
        use_kdtree=False,
        max_iterations=1,
    )
    reg.transform_point_cloud()
    reg.expectation()

    assert reg.P is None
    assert reg.Pt1.shape == (reg.N,)
    assert reg.P1.shape == (reg.M,)
    assert reg.PX.shape == (reg.M, reg.D)
    assert reg.Np > 0


@pytest.mark.parametrize("low_rank", [False, True])
def test_deformable_float32_dtype_is_preserved_internals(low_rank):
    rng = np.random.default_rng(52)
    X = rng.normal(size=(18, 3)).astype(np.float32)
    Y = rng.normal(size=(18, 3)).astype(np.float32)

    reg = DeformableRegistration(
        X=X,
        Y=Y,
        low_rank=low_rank,
        num_eig=6,
        dtype=np.float32,
        use_kdtree=False,
        max_iterations=2,
    )
    TY, _ = reg.register()

    assert TY.dtype == np.float32
    assert reg.X.dtype == np.float32
    assert reg.Y.dtype == np.float32
    assert reg.TY.dtype == np.float32
    assert reg.W.dtype == np.float32
    if low_rank:
        assert reg.Q.dtype == np.float32
        assert reg.S.dtype == np.float32
        assert reg.inv_S.dtype == np.float32
    else:
        assert reg.G.dtype == np.float32


def test_constrained_deformable_float32_dtype_is_preserved_internals():
    rng = np.random.default_rng(53)
    X = rng.normal(size=(16, 3)).astype(np.float32)
    Y = rng.normal(size=(16, 3)).astype(np.float32)

    reg = ConstrainedDeformableRegistration(
        X=X,
        Y=Y,
        source_id=np.array([0, 2, 4, 6], dtype=int),
        target_id=np.array([0, 2, 4, 6], dtype=int),
        low_rank=False,
        dtype=np.float32,
        use_kdtree=False,
        max_iterations=2,
    )
    TY, _ = reg.register()

    assert TY.dtype == np.float32
    assert reg.X.dtype == np.float32
    assert reg.Y.dtype == np.float32
    assert reg.TY.dtype == np.float32
    assert reg.W.dtype == np.float32
    assert reg.P_tilde.dtype == np.float32


@pytest.mark.parametrize("low_rank", [False, True])
def test_deformable_float32_matches_float64_accuracy(low_rank):
    rng = np.random.default_rng(7 if low_rank else 5)
    M, D = 36, 3
    Y0 = rng.normal(size=(M, D))
    W_true = 0.08 * rng.normal(size=(M, D))
    X = Y0 + W_true + 0.01 * rng.normal(size=(M, D))

    common_kwargs = dict(
        alpha=2.0,
        beta=2.0,
        low_rank=low_rank,
        num_eig=10,
        use_kdtree=False,
        max_iterations=15,
        tolerance=1e-6,
    )

    reg64 = DeformableRegistration(
        X=X.astype(np.float64),
        Y=Y0.astype(np.float64),
        dtype=np.float64,
        **common_kwargs,
    )
    TY64, params64 = reg64.register()

    reg32 = DeformableRegistration(
        X=X.astype(np.float32),
        Y=Y0.astype(np.float32),
        dtype=np.float32,
        **common_kwargs,
    )
    TY32, params32 = reg32.register()

    TY_delta = TY64 - TY32.astype(np.float64)
    W64 = params64[-1].astype(np.float64)
    W32 = params32[-1].astype(np.float64)

    assert np.sqrt(np.mean(TY_delta * TY_delta)) < 5e-5
    assert np.max(np.abs(TY_delta)) < 1e-4
    assert abs(float(reg64.sigma2) - float(reg32.sigma2)) < 1e-5
    assert np.sqrt(np.mean((W64 - W32) ** 2)) < 1e-3


def test_atlas_float32_matches_float64_accuracy():
    X, Y, U, L = _atlas_inputs(seed=17)

    reg64 = AtlasRegistration(
        X=X.astype(np.float64),
        Y=Y.astype(np.float64),
        U=U.astype(np.float64),
        eigenvalues=L.astype(np.float64),
        normalize=True,
        use_kdtree=False,
        optimize_similarity=True,
        with_scale=True,
        dtype=np.float64,
        max_iterations=15,
        tolerance=1e-6,
    )
    TY64, params64 = reg64.register()

    reg32 = AtlasRegistration(
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        U=U.astype(np.float32),
        eigenvalues=L.astype(np.float32),
        normalize=True,
        use_kdtree=False,
        optimize_similarity=True,
        with_scale=True,
        dtype=np.float32,
        max_iterations=15,
        tolerance=1e-6,
    )
    TY32, params32 = reg32.register()

    TY_delta = TY64 - TY32.astype(np.float64)
    b64 = params64["b"].astype(np.float64)
    b32 = params32["b"].astype(np.float64)

    assert np.sqrt(np.mean(TY_delta * TY_delta)) < 1e-5
    assert np.max(np.abs(TY_delta)) < 5e-5
    assert abs(float(reg64.sigma2) - float(reg32.sigma2)) < 1e-5
    assert np.sqrt(np.mean((b64 - b32) ** 2)) < 1e-4
