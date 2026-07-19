import numpy as np
import pytest

from biocpd import AtlasRegistration


def _atlas_arrays(seed, point_count, mode_count, dtype):
    generator = np.random.default_rng(seed)
    source = generator.normal(size=(point_count, 3)).astype(dtype)
    target = (
        source
        + generator.normal(scale=0.05, size=source.shape).astype(dtype)
    )
    modes = generator.normal(
        size=(point_count * 3, mode_count),
    ).astype(dtype)
    eigenvalues = np.linspace(1.5, 0.2, mode_count).astype(dtype)
    return target, source, modes, eigenvalues


def _set_posterior_moments(registration, seed=0):
    generator = np.random.default_rng(seed)
    source_mass = generator.uniform(0.2, 1.3, size=registration.M).astype(
        registration.dtype
    )
    residual = generator.normal(
        scale=0.15,
        size=(registration.M, registration.D),
    ).astype(registration.dtype)
    registration.P1 = source_mass
    registration.PX = source_mass[:, None] * (registration.Y + residual)
    registration.Pt1 = np.ones(registration.N, dtype=registration.dtype)
    registration.Np = float(source_mass.sum())
    registration.sigma2 = registration.dtype.type(0.2)
    return source_mass, residual


def _coefficient_reference(registration, source_mass, residual):
    weights = np.repeat(source_mass, registration.D)
    prior = registration.lambda_reg * registration.sigma2 * registration.invL
    system = registration.U_flat.T @ (
        weights[:, None] * registration.U_flat
    )
    system[np.diag_indices(registration.K)] += prior
    rhs = registration.U_flat.T @ (weights * residual.reshape(-1))
    return np.linalg.solve(system, rhs)


def test_atlas_auto_dense_block_preserves_small_and_explicit_blocks():
    target, source, modes, eigenvalues = _atlas_arrays(
        1,
        1000,
        2,
        np.float32,
    )
    automatic = AtlasRegistration(
        X=target,
        Y=source,
        U=modes,
        eigenvalues=eigenvalues,
        use_kdtree=False,
        max_iterations=1,
    )
    explicit = AtlasRegistration(
        X=target,
        Y=source,
        U=modes,
        eigenvalues=eigenvalues,
        use_kdtree=False,
        dense_block_size=333,
        max_iterations=1,
    )

    assert automatic._get_dense_block_size() == len(target)
    assert explicit._get_dense_block_size() == 333


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("outlier_weight", [0.0, 0.25])
@pytest.mark.parametrize("sigma2", [0.08, 1.5])
@pytest.mark.parametrize("store_posterior", [False, True])
def test_fused_atlas_dense_statistics_match_dense_reference(
    dtype,
    outlier_weight,
    sigma2,
    store_posterior,
):
    generator = np.random.default_rng(20)
    source = generator.normal(size=(17, 3)).astype(dtype)
    target = generator.normal(size=(13, 3)).astype(dtype)
    modes = generator.normal(size=(source.size, 3)).astype(dtype)
    registration = AtlasRegistration(
        X=target,
        Y=source,
        U=modes,
        eigenvalues=np.array([1.0, 0.6, 0.3], dtype=dtype),
        w=outlier_weight,
        dtype=dtype,
        use_kdtree=False,
        store_posterior=store_posterior,
        max_iterations=1,
    )
    registration.sigma2 = dtype(sigma2)
    registration.TY = source + dtype(0.03)
    registration._compute_dense_posterior_stats(
        store_p=store_posterior,
        block_size=4,
    )

    difference = source[:, None, :] + dtype(0.03) - target[None, :, :]
    posterior = np.exp(
        -np.sum(difference * difference, axis=2) / (2 * dtype(sigma2))
    )
    outlier = (
        (2 * np.pi * dtype(sigma2)) ** (registration.D / 2)
        * outlier_weight
        / (1 - outlier_weight)
        * registration.M
        / registration.N
    )
    denominator = np.sum(posterior, axis=0, keepdims=True) + outlier
    expectation_objective = (
        0.5
        * registration.N
        * registration.D
        * np.log(2.0 * np.pi * dtype(sigma2))
        - np.sum(
            np.log(np.maximum(denominator, np.finfo(dtype).tiny)),
            dtype=np.float64,
        )
    )
    posterior /= np.maximum(denominator, np.finfo(dtype).tiny)
    tolerance = 2e-5 if dtype == np.float32 else 2e-12

    np.testing.assert_allclose(
        registration.Pt1,
        np.sum(posterior, axis=0),
        atol=tolerance,
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        registration.P1,
        np.sum(posterior, axis=1),
        atol=tolerance,
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        registration.PX,
        posterior @ registration.X,
        atol=tolerance,
        rtol=tolerance,
    )
    assert np.isclose(
        registration.Np,
        np.sum(posterior),
        atol=tolerance,
        rtol=tolerance,
    )
    if store_posterior:
        np.testing.assert_allclose(
            registration.P,
            posterior,
            atol=tolerance,
            rtol=tolerance,
        )
    else:
        assert registration.P is None
    objective_tolerance = 2e-4 if dtype == np.float32 else 2e-11
    assert np.isclose(
        registration.expectation_objective,
        expectation_objective,
        atol=objective_tolerance,
        rtol=objective_tolerance,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("solver", ["cholesky", "cg"])
def test_coefficient_solvers_match_weighted_normal_equations(dtype, solver):
    target, source, modes, eigenvalues = _atlas_arrays(
        2,
        60,
        12,
        dtype,
    )
    registration = AtlasRegistration(
        X=target,
        Y=source,
        U=modes,
        eigenvalues=eigenvalues,
        lambda_reg=0.2,
        optimize_similarity=False,
        coefficient_solver=solver,
        dtype=dtype,
        max_iterations=1,
    )
    source_mass, residual = _set_posterior_moments(registration, seed=3)
    expected = _coefficient_reference(
        registration,
        source_mass,
        residual,
    )
    registration.update_transform()
    tolerance = 3e-4 if dtype == np.float32 else 2e-9

    np.testing.assert_allclose(
        registration.b.reshape(-1),
        expected,
        atol=tolerance,
        rtol=tolerance,
    )
    diagnostics = registration.coefficient_solver_diagnostics
    assert diagnostics["last_method"] == solver
    assert diagnostics["fallback_count"] == 0
    if solver == "cg":
        assert registration._basis_norm_squared is not None
        assert diagnostics["last_iterations"] > 0
        assert (
            diagnostics["last_relative_residual"]
            <= registration.coefficient_tolerance
        )
    else:
        assert registration._basis_norm_squared is None
        assert diagnostics["last_iterations"] == 0
        assert diagnostics["last_relative_residual"] is None


def test_matrix_free_cg_reuses_previous_coefficients_as_warm_start():
    target, source, modes, eigenvalues = _atlas_arrays(
        4,
        50,
        10,
        np.float64,
    )
    registration = AtlasRegistration(
        X=target,
        Y=source,
        U=modes,
        eigenvalues=eigenvalues,
        lambda_reg=0.2,
        optimize_similarity=False,
        coefficient_solver="cg",
        dtype=np.float64,
        max_iterations=1,
    )
    _set_posterior_moments(registration, seed=5)
    registration.update_transform()
    first_iterations = registration.coefficient_solver_diagnostics[
        "last_iterations"
    ]
    registration.update_transform()
    diagnostics = registration.coefficient_solver_diagnostics

    assert first_iterations > 0
    assert diagnostics["last_iterations"] == 0
    assert diagnostics["cg_solve_count"] == 2
    assert diagnostics["fallback_count"] == 0


def test_cg_iteration_limit_falls_back_to_cholesky():
    target, source, modes, eigenvalues = _atlas_arrays(
        6,
        70,
        14,
        np.float64,
    )
    registration = AtlasRegistration(
        X=target,
        Y=source,
        U=modes,
        eigenvalues=eigenvalues,
        lambda_reg=0.05,
        optimize_similarity=False,
        coefficient_solver="cg",
        coefficient_tolerance=1e-14,
        coefficient_max_iterations=1,
        dtype=np.float64,
        max_iterations=1,
    )
    source_mass, residual = _set_posterior_moments(registration, seed=7)
    expected = _coefficient_reference(
        registration,
        source_mass,
        residual,
    )
    registration.update_transform()
    diagnostics = registration.coefficient_solver_diagnostics

    np.testing.assert_allclose(
        registration.b.reshape(-1),
        expected,
        atol=1e-10,
        rtol=1e-10,
    )
    assert diagnostics["last_method"] == "cholesky"
    assert diagnostics["fallback_count"] == 1
    assert diagnostics["last_fallback_reason"] == "iteration limit"
    assert diagnostics["last_iterations"] == 1


def test_cg_fallback_handles_rank_deficient_unregularized_basis():
    target, source, modes, eigenvalues = _atlas_arrays(
        14,
        45,
        6,
        np.float64,
    )
    modes[:, 3:] = modes[:, :3]
    registration = AtlasRegistration(
        X=target,
        Y=source,
        U=modes,
        eigenvalues=eigenvalues,
        lambda_reg=0.0,
        optimize_similarity=False,
        coefficient_solver="cg",
        coefficient_tolerance=1e-14,
        coefficient_max_iterations=1,
        dtype=np.float64,
        max_iterations=1,
    )
    _set_posterior_moments(registration, seed=15)
    registration.update_transform()

    assert np.isfinite(registration.b).all()
    assert np.isfinite(registration.TY).all()
    diagnostics = registration.coefficient_solver_diagnostics
    assert diagnostics["last_method"] == "cholesky"
    assert diagnostics["fallback_count"] == 1


@pytest.mark.parametrize(
    ("threshold", "expected_method"),
    [(4, "cg"), (20, "cholesky")],
)
def test_auto_coefficient_solver_obeys_mode_threshold(
    threshold,
    expected_method,
):
    target, source, modes, eigenvalues = _atlas_arrays(
        8,
        40,
        8,
        np.float64,
    )
    registration = AtlasRegistration(
        X=target,
        Y=source,
        U=modes,
        eigenvalues=eigenvalues,
        optimize_similarity=False,
        coefficient_solver="auto",
        coefficient_auto_threshold=threshold,
        dtype=np.float64,
        max_iterations=1,
    )
    _set_posterior_moments(registration, seed=9)
    registration.update_transform()

    assert (
        registration.coefficient_solver_diagnostics["last_method"]
        == expected_method
    )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("coefficient_solver", "unknown", "coefficient_solver"),
        ("coefficient_tolerance", 0.0, "coefficient_tolerance"),
        ("coefficient_tolerance", np.nan, "coefficient_tolerance"),
        (
            "coefficient_max_iterations",
            0,
            "coefficient_max_iterations",
        ),
        (
            "coefficient_max_iterations",
            True,
            "coefficient_max_iterations",
        ),
        ("coefficient_auto_threshold", 0, "coefficient_auto_threshold"),
    ],
)
def test_invalid_coefficient_solver_configuration_is_rejected(
    keyword,
    value,
    message,
):
    target, source, modes, eigenvalues = _atlas_arrays(
        10,
        12,
        3,
        np.float64,
    )
    with pytest.raises(ValueError, match=message):
        AtlasRegistration(
            X=target,
            Y=source,
            U=modes,
            eigenvalues=eigenvalues,
            max_iterations=1,
            **{keyword: value},
        )


@pytest.mark.parametrize("use_sparse", [False, True])
def test_cg_and_cholesky_converge_to_the_same_atlas_solution(use_sparse):
    generator = np.random.default_rng(12)
    point_count, mode_count = 70, 16
    source = generator.normal(size=(point_count, 3))
    orthogonal, _ = np.linalg.qr(
        generator.normal(size=(source.size, mode_count))
    )
    modes = orthogonal * np.linspace(0.5, 0.15, mode_count)
    eigenvalues = np.linspace(0.6, 0.1, mode_count)
    coefficients = generator.normal(scale=0.15, size=mode_count)
    target = source + (modes @ coefficients).reshape(source.shape)
    common = dict(
        X=target,
        Y=source,
        U=modes,
        eigenvalues=eigenvalues,
        lambda_reg=0.0001,
        optimize_similarity=False,
        use_kdtree=use_sparse,
        k=point_count,
        kdtree_radius_scale=1e-6,
        dtype=np.float64,
        max_iterations=200,
        tolerance=1e-5,
    )
    direct = AtlasRegistration(**common, coefficient_solver="cholesky")
    iterative = AtlasRegistration(**common, coefficient_solver="cg")
    direct_points, _ = direct.register(callback=None)
    iterative_points, _ = iterative.register(callback=None)

    assert direct.iteration < direct.max_iterations
    assert iterative.iteration < iterative.max_iterations
    assert direct.diff <= direct.tolerance
    assert iterative.diff <= iterative.tolerance
    np.testing.assert_allclose(
        iterative_points,
        direct_points,
        atol=2e-7,
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        iterative.b,
        direct.b,
        atol=2e-6,
        rtol=2e-6,
    )
    assert iterative.coefficient_solver_diagnostics["fallback_count"] == 0


def test_cg_matches_normalized_similarity_atlas_workflow():
    generator = np.random.default_rng(18)
    point_count, mode_count = 60, 8
    source = generator.normal(size=(point_count, 3)).astype(np.float32)
    orthogonal, _ = np.linalg.qr(
        generator.normal(size=(source.size, mode_count))
    )
    modes = (
        orthogonal * np.linspace(0.4, 0.15, mode_count)
    ).astype(np.float32)
    eigenvalues = np.linspace(0.5, 0.1, mode_count).astype(np.float32)
    coefficients = generator.normal(
        scale=0.12,
        size=mode_count,
    ).astype(np.float32)
    deformed = source + (modes @ coefficients).reshape(source.shape)
    angle = np.float32(0.18)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    target = (
        np.float32(1.04) * (deformed @ rotation.T)
        + np.array([0.15, -0.08, 0.04], dtype=np.float32)
    )
    common = dict(
        X=target,
        Y=source,
        U=modes,
        eigenvalues=eigenvalues,
        lambda_reg=0.001,
        normalize=True,
        optimize_similarity=True,
        with_scale=True,
        use_kdtree=False,
        dtype=np.float32,
        max_iterations=150,
        tolerance=1e-4,
    )
    direct = AtlasRegistration(**common, coefficient_solver="cholesky")
    iterative = AtlasRegistration(**common, coefficient_solver="cg")
    direct_points, _ = direct.register(callback=None)
    iterative_points, _ = iterative.register(callback=None)

    assert direct.iteration < direct.max_iterations
    assert iterative.iteration < iterative.max_iterations
    np.testing.assert_allclose(
        iterative_points,
        direct_points,
        atol=2e-4,
        rtol=2e-4,
    )
    np.testing.assert_allclose(
        iterative.b,
        direct.b,
        atol=3e-4,
        rtol=3e-4,
    )
    assert iterative.coefficient_solver_diagnostics["fallback_count"] == 0
