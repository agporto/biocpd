import numpy as np
import pytest

from biocpd import (
    ConstrainedDeformableRegistration,
    DeformableRegistration,
)
from biocpd._low_rank import gaussian_kernel_eigendecomposition
from biocpd.utility import gaussian_kernel


def _deformable_problem(seed, point_count=36):
    generator = np.random.default_rng(seed)
    source = generator.normal(size=(point_count, 3))
    source *= np.array([1.2, 0.8, 0.6])
    beta = 1.5
    weights = 0.015 * generator.normal(size=source.shape)
    target = source + gaussian_kernel(source, beta) @ weights
    return source, target, beta


def _legacy_low_rank_update(registration, weights, F, lambda_value):
    weighted_basis = weights[:, None] * registration.Q
    system = (
        lambda_value * registration.inv_S
        + registration.Q.T @ weighted_basis
    )
    coefficients = np.linalg.solve(system, registration.Q.T @ F)
    W = (F - weighted_basis @ coefficients) / lambda_value
    deformation = (
        registration.Q
        @ registration.S
        @ (registration.Q.T @ W)
    )
    return coefficients, W, deformation


@pytest.mark.parametrize("method", ["randomized_svd", "pivoted_cholesky"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_coefficient_space_update_matches_legacy_woodbury_algebra(
    method,
    dtype,
):
    generator = np.random.default_rng(100)
    source = generator.normal(size=(48, 3)).astype(dtype)
    registration = DeformableRegistration(
        X=(
            source
            + 0.02 * generator.normal(size=source.shape).astype(dtype)
        ),
        Y=source,
        low_rank=True,
        num_eig=16,
        low_rank_method=method,
        use_kdtree=False,
        dtype=dtype,
        max_iterations=0,
    )
    weights = (0.1 + generator.random(len(source))).astype(dtype)
    F = generator.normal(size=source.shape).astype(dtype)
    lambda_value = dtype(0.7)
    expected_coefficients, expected_W, expected_deformation = (
        _legacy_low_rank_update(
            registration,
            weights,
            F,
            lambda_value,
        )
    )

    registration._update_low_rank_transform(
        weights,
        F,
        lambda_value,
    )
    actual_deformation = registration.Q @ registration._low_rank_coefficients
    tolerance = 2e-4 if dtype == np.float32 else 1e-11

    np.testing.assert_allclose(
        registration._low_rank_coefficients,
        expected_coefficients,
        atol=tolerance,
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        registration.W,
        expected_W,
        atol=tolerance,
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        actual_deformation,
        expected_deformation,
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("lambda_value", [0.03, 4.0])
def test_sqrt_weighted_low_rank_update_handles_zero_dynamic_weights(
    dtype,
    lambda_value,
):
    generator = np.random.default_rng(116)
    source = generator.normal(size=(52, 3)).astype(dtype)
    registration = DeformableRegistration(
        X=source + dtype(0.01),
        Y=source,
        low_rank=True,
        num_eig=18,
        low_rank_method="pivoted_cholesky",
        use_kdtree=False,
        dtype=dtype,
        max_iterations=0,
    )
    weights = np.geomspace(1e-5, 2.0, len(source)).astype(dtype)
    weights[::7] = 0
    F = generator.normal(size=source.shape).astype(dtype)
    lambda_value = dtype(lambda_value)
    system = registration.Q.T @ (weights[:, None] * registration.Q)
    system[np.diag_indices(registration.num_eig)] += (
        lambda_value * registration._inv_S_values
    )
    rhs = registration.Q.T @ F

    registration._update_low_rank_transform(weights, F, lambda_value)
    relative_residual = np.linalg.norm(
        system @ registration._low_rank_coefficients - rhs
    ) / max(np.linalg.norm(rhs), np.finfo(dtype).tiny)
    tolerance = 8e-4 if dtype == np.float32 else 2e-11

    assert relative_residual < tolerance
    assert np.isfinite(registration.W).all()


def test_zero_tolerance_deformable_variance_remains_positive():
    source = _deformable_problem(117, point_count=24)[0]
    registration = DeformableRegistration(
        X=source.copy(),
        Y=source,
        low_rank=True,
        num_eig=12,
        low_rank_method="pivoted_cholesky",
        use_kdtree=False,
        dtype=np.float64,
        tolerance=0.0,
        max_iterations=0,
    )
    registration.Pt1 = np.ones(registration.N)
    registration.P1 = np.ones(registration.M)
    registration.PX = source.copy()
    registration.Np = float(registration.N)
    registration.TY = source.copy()
    registration.sigma2 = 1.0

    registration.update_variance()

    assert np.isfinite(registration.sigma2)
    assert registration.sigma2 > 0


def test_constrained_prior_matrix_is_lazy_and_preserves_binary_pairs():
    source, target, beta = _deformable_problem(118, point_count=28)
    source_id = np.array([0, 0, 3, 7, 7])
    target_id = np.array([1, 1, 5, 9, 10])
    registration = ConstrainedDeformableRegistration(
        X=target,
        Y=source,
        source_id=source_id,
        target_id=target_id,
        beta=beta,
        low_rank=True,
        num_eig=12,
        low_rank_method="pivoted_cholesky",
        use_kdtree=False,
        max_iterations=0,
    )
    expected = np.zeros((len(source), len(target)), dtype=np.float32)
    expected[source_id, target_id] = 1

    assert registration._P_tilde is None
    np.testing.assert_array_equal(registration.P1_tilde, expected.sum(axis=1))
    np.testing.assert_allclose(
        registration.PX_tilde,
        expected @ target,
        atol=2e-7,
        rtol=2e-7,
    )
    np.testing.assert_array_equal(registration.P_tilde, expected)

    replacement = np.full_like(expected, 0.25)
    registration.P_tilde = replacement
    np.testing.assert_array_equal(registration.P_tilde, replacement)


@pytest.mark.parametrize(
    "source_id,target_id,match",
    [
        (np.array([0.0]), np.array([0]), "integer"),
        (np.array([0]), np.array([0.0]), "integer"),
        (np.array([0, 1]), np.array([0]), "same length"),
        (np.array([-1]), np.array([0]), "out of bounds"),
        (np.array([0]), np.array([24]), "out of bounds"),
    ],
)
def test_constrained_prior_indices_are_validated(
    source_id,
    target_id,
    match,
):
    source, target, beta = _deformable_problem(119, point_count=24)

    with pytest.raises(ValueError, match=match):
        ConstrainedDeformableRegistration(
            X=target,
            Y=source,
            source_id=source_id,
            target_id=target_id,
            beta=beta,
            low_rank=True,
            num_eig=8,
            use_kdtree=False,
            max_iterations=0,
        )


def test_public_transform_respects_an_explicitly_replaced_W():
    generator = np.random.default_rng(101)
    source = generator.normal(size=(35, 3))
    registration = DeformableRegistration(
        X=source,
        Y=source,
        low_rank=True,
        num_eig=12,
        use_kdtree=False,
        dtype=np.float64,
        max_iterations=0,
    )
    registration._low_rank_coefficients = generator.normal(size=(12, 3))
    registration.W = generator.normal(size=source.shape)
    expected = (
        source
        + registration.Q
        @ registration.S
        @ (registration.Q.T @ registration.W)
    )

    actual = registration.transform_point_cloud()

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    new_source = generator.normal(size=(11, 3))
    expected_new = (
        new_source
        + gaussian_kernel(new_source, registration.beta, source)
        @ registration.W
    )
    actual_new = registration.transform_point_cloud(new_source)
    np.testing.assert_allclose(
        actual_new,
        expected_new,
        atol=1e-12,
        rtol=1e-12,
    )


def test_low_rank_registration_preserves_transform_override_hook():
    class TrackingRegistration(DeformableRegistration):
        def __init__(self, *args, **kwargs):
            self.transform_calls = 0
            super().__init__(*args, **kwargs)

        def transform_point_cloud(self, Y=None):
            self.transform_calls += 1
            return super().transform_point_cloud(Y)

    generator = np.random.default_rng(117)
    source = generator.normal(size=(30, 3))
    registration = TrackingRegistration(
        X=source + 0.01 * generator.normal(size=source.shape),
        Y=source,
        low_rank=True,
        num_eig=10,
        low_rank_method="pivoted_cholesky",
        use_kdtree=False,
        dtype=np.float64,
        max_iterations=3,
        tolerance=0.0,
    )

    registration.register()

    assert registration.transform_calls == registration.iteration + 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pivoted_cholesky_full_rank_reconstructs_gaussian_kernel(dtype):
    generator = np.random.default_rng(102)
    points = generator.normal(size=(24, 3)).astype(dtype)
    basis, eigenvalues, diagnostics = gaussian_kernel_eigendecomposition(
        points,
        beta=1.3,
        rank=len(points),
        method="pivoted_cholesky",
    )
    expected = gaussian_kernel(points, 1.3)
    actual = (basis * eigenvalues) @ basis.T
    tolerance = 2e-5 if dtype == np.float32 else 1e-12

    assert diagnostics["rank"] == len(points)
    assert diagnostics["max_residual_diagonal"] == 0.0
    assert basis.dtype == dtype
    assert eigenvalues.dtype == dtype
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        basis.T @ basis,
        np.eye(len(points)),
        atol=tolerance,
        rtol=tolerance,
    )


def test_pivoted_cholesky_does_not_call_full_kernel_builder(monkeypatch):
    generator = np.random.default_rng(103)
    source = generator.normal(size=(80, 3))

    def reject_full_kernel(*args, **kwargs):
        raise AssertionError("the full Gaussian kernel was materialized")

    monkeypatch.setattr(
        "biocpd._low_rank.gaussian_kernel",
        reject_full_kernel,
    )
    registration = DeformableRegistration(
        X=source,
        Y=source,
        low_rank=True,
        num_eig=15,
        low_rank_method="pivoted_cholesky",
        use_kdtree=False,
        max_iterations=0,
    )

    assert registration.Q.shape == (80, 15)
    assert registration.S.shape == (15, 15)
    assert registration.low_rank_diagnostics["method"] == "pivoted_cholesky"


def test_pivoted_cholesky_is_deterministic_and_reports_residual():
    generator = np.random.default_rng(104)
    points = generator.normal(size=(60, 3))
    first_Q, first_values, first_diagnostics = (
        gaussian_kernel_eigendecomposition(
            points,
            beta=1.1,
            rank=14,
            method="pivoted_cholesky",
        )
    )
    second_Q, second_values, second_diagnostics = (
        gaussian_kernel_eigendecomposition(
            points,
            beta=1.1,
            rank=14,
            method="pivoted_cholesky",
        )
    )
    first_kernel = (first_Q * first_values) @ first_Q.T
    second_kernel = (second_Q * second_values) @ second_Q.T
    full_kernel = gaussian_kernel(points, 1.1)
    actual_diagonal_residual = np.max(
        np.diag(full_kernel - first_kernel)
    )

    np.testing.assert_allclose(first_kernel, second_kernel, atol=1e-13, rtol=1e-13)
    assert first_diagnostics == second_diagnostics
    assert np.isclose(
        actual_diagonal_residual,
        first_diagnostics["max_residual_diagonal"],
        atol=1e-12,
        rtol=1e-10,
    )


def test_pivoted_cholesky_tolerance_stops_after_reaching_error_target():
    generator = np.random.default_rng(112)
    points = generator.normal(size=(90, 3))
    basis, eigenvalues, diagnostics = gaussian_kernel_eigendecomposition(
        points,
        beta=2.5,
        rank=len(points),
        method="pivoted_cholesky",
        tolerance=1e-3,
    )
    approximation = (basis * eigenvalues) @ basis.T
    full_kernel = gaussian_kernel(points, 2.5)
    maximum_diagonal_error = float(
        np.max(np.diag(full_kernel - approximation))
    )

    assert diagnostics["rank"] < len(points)
    assert diagnostics["max_residual_diagonal"] <= 1e-3
    assert maximum_diagonal_error <= 1e-3 + 1e-12


def test_pivoted_cholesky_handles_duplicate_points_with_finite_spectrum():
    generator = np.random.default_rng(105)
    unique = generator.normal(size=(8, 3))
    source = np.repeat(unique, 3, axis=0)
    target = source + 0.001 * generator.normal(size=source.shape)
    registration = DeformableRegistration(
        X=target,
        Y=source,
        low_rank=True,
        num_eig=20,
        low_rank_method="pivoted_cholesky",
        use_kdtree=False,
        dtype=np.float64,
        max_iterations=3,
        tolerance=0.0,
    )

    transformed, parameters = registration.register()

    assert registration.Q.shape[1] <= len(unique)
    assert np.isfinite(registration.S).all()
    assert np.isfinite(registration.inv_S).all()
    assert np.isfinite(registration.W).all()
    assert np.isfinite(transformed).all()
    assert parameters[0].shape == registration.Q.shape
    assert parameters[1].shape == registration.S.shape
    assert parameters[2].shape == registration.W.shape


@pytest.mark.parametrize(
    "keyword,value,match",
    [
        ("num_eig", 0, "num_eig"),
        ("num_eig", 2.5, "num_eig"),
        ("low_rank_method", "unknown", "low_rank_method"),
        ("low_rank_tolerance", -1e-3, "low_rank_tolerance"),
        ("low_rank_tolerance", 1.0, "low_rank_tolerance"),
        ("low_rank_tolerance", np.nan, "low_rank_tolerance"),
    ],
)
def test_low_rank_options_are_validated(keyword, value, match):
    points = np.arange(30, dtype=float).reshape(10, 3)
    kwargs = {
        "num_eig": 5,
        "low_rank_method": "pivoted_cholesky",
        "low_rank_tolerance": 0.0,
    }
    kwargs[keyword] = value

    with pytest.raises(ValueError, match=match):
        DeformableRegistration(
            X=points,
            Y=points,
            low_rank=True,
            use_kdtree=False,
            max_iterations=0,
            **kwargs,
        )


def test_existing_positional_constructor_arguments_keep_their_meaning():
    generator = np.random.default_rng(111)
    points = generator.normal(size=(18, 3))

    registration = DeformableRegistration(
        2.5,
        1.4,
        True,
        7,
        False,
        5,
        False,
        0.1,
        X=points,
        Y=points,
        max_iterations=0,
    )

    assert registration.alpha == 2.5
    assert registration.beta == 1.4
    assert registration.low_rank is True
    assert registration.num_eig == 7
    assert registration.use_kdtree is False
    assert registration.w == 0.1
    assert registration.low_rank_method == "randomized_svd"


@pytest.mark.parametrize("method", ["randomized_svd", "pivoted_cholesky"])
def test_full_rank_low_rank_methods_match_full_kernel_convergence(method):
    source, target, beta = _deformable_problem(106)
    common = dict(
        X=target,
        Y=source,
        alpha=2.0,
        beta=beta,
        use_kdtree=False,
        dtype=np.float64,
        max_iterations=200,
        tolerance=1e-7,
    )
    exact = DeformableRegistration(low_rank=False, **common)
    low_rank = DeformableRegistration(
        low_rank=True,
        num_eig=len(source),
        low_rank_method=method,
        **common,
    )

    exact_points, _ = exact.register()
    low_rank_points, _ = low_rank.register()

    assert exact.iteration < exact.max_iterations
    assert low_rank.iteration < low_rank.max_iterations
    assert exact.diff <= exact.tolerance
    assert low_rank.diff <= low_rank.tolerance
    assert np.sqrt(np.mean((low_rank_points - target) ** 2)) < 1e-7
    np.testing.assert_allclose(
        low_rank_points,
        exact_points,
        atol=2e-7,
        rtol=2e-7,
    )
    assert 0 <= float(exact.sigma2) <= exact.tolerance
    assert 0 <= float(low_rank.sigma2) <= low_rank.tolerance


@pytest.mark.parametrize("seed", [107, 113, 114, 115, 116])
def test_truncated_pivoted_cholesky_registration_improves_alignment(seed):
    source, target, beta = _deformable_problem(seed, point_count=90)
    initial_error = np.sqrt(np.mean((source - target) ** 2))
    registration = DeformableRegistration(
        X=target,
        Y=source,
        alpha=2.0,
        beta=beta,
        low_rank=True,
        num_eig=20,
        low_rank_method="pivoted_cholesky",
        use_kdtree=False,
        dtype=np.float64,
        max_iterations=100,
        tolerance=1e-6,
    )

    transformed, _ = registration.register()
    final_error = np.sqrt(np.mean((transformed - target) ** 2))

    assert registration.iteration < registration.max_iterations
    assert registration.diff <= registration.tolerance
    assert final_error < 0.05 * initial_error
    assert registration.low_rank_diagnostics["rank"] == 20


def test_constrained_coefficient_update_matches_legacy_algebra():
    generator = np.random.default_rng(108)
    source = generator.normal(size=(42, 3))
    ids = np.arange(0, len(source), 7)
    registration = ConstrainedDeformableRegistration(
        X=source + 0.01 * generator.normal(size=source.shape),
        Y=source,
        source_id=ids,
        target_id=ids,
        e_alpha=0.05,
        alpha=2.0,
        beta=1.4,
        low_rank=True,
        num_eig=14,
        low_rank_method="pivoted_cholesky",
        use_kdtree=False,
        dtype=np.float64,
        max_iterations=0,
    )
    registration.P1 = 0.1 + generator.random(len(source))
    registration.PX = generator.normal(size=source.shape)
    registration.sigma2 = 0.4
    constraint_scale = registration.sigma2 / registration.e_alpha
    combined_weights = (
        registration.P1
        + constraint_scale * registration.P1_tilde
    )
    F = (
        registration.PX
        - registration.P1[:, None] * registration.Y
        + constraint_scale
        * (
            registration.PX_tilde
            - registration.P1_tilde[:, None] * registration.Y
        )
    )
    lambda_value = registration.alpha * registration.sigma2
    expected_coefficients, expected_W, expected_deformation = (
        _legacy_low_rank_update(
            registration,
            combined_weights,
            F,
            lambda_value,
        )
    )

    registration.update_transform()

    np.testing.assert_allclose(
        registration._low_rank_coefficients,
        expected_coefficients,
        atol=1e-11,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        registration.W,
        expected_W,
        atol=1e-11,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        registration.Q @ registration._low_rank_coefficients,
        expected_deformation,
        atol=1e-11,
        rtol=1e-11,
    )


def test_constrained_full_rank_pivoted_cholesky_matches_full_kernel_convergence():
    source, target, beta = _deformable_problem(110)
    ids = np.arange(0, len(source), 6)
    common = dict(
        X=target,
        Y=source,
        source_id=ids,
        target_id=ids,
        e_alpha=0.05,
        alpha=2.0,
        beta=beta,
        use_kdtree=False,
        dtype=np.float64,
        max_iterations=200,
        tolerance=1e-7,
    )
    exact = ConstrainedDeformableRegistration(low_rank=False, **common)
    low_rank = ConstrainedDeformableRegistration(
        low_rank=True,
        num_eig=len(source),
        low_rank_method="pivoted_cholesky",
        **common,
    )

    exact_points, _ = exact.register()
    low_rank_points, _ = low_rank.register()

    assert exact.iteration < exact.max_iterations
    assert low_rank.iteration < low_rank.max_iterations
    assert exact.diff <= exact.tolerance
    assert low_rank.diff <= low_rank.tolerance
    assert np.sqrt(np.mean((low_rank_points - target) ** 2)) < 1e-7
    np.testing.assert_allclose(
        low_rank_points,
        exact_points,
        atol=2e-7,
        rtol=2e-7,
    )
    assert abs(float(low_rank.sigma2) - float(exact.sigma2)) < 1e-9


def test_pivoted_cholesky_float32_state_is_preserved():
    generator = np.random.default_rng(109)
    source = generator.normal(size=(50, 3)).astype(np.float32)
    registration = DeformableRegistration(
        X=source + 0.01 * generator.normal(size=source.shape).astype(np.float32),
        Y=source,
        low_rank=True,
        num_eig=15,
        low_rank_method="pivoted_cholesky",
        use_kdtree=False,
        dtype=np.float32,
        max_iterations=2,
        tolerance=0.0,
    )

    transformed, _ = registration.register()

    assert transformed.dtype == np.float32
    assert registration.Q.dtype == np.float32
    assert registration._S_values.dtype == np.float32
    assert registration.S.dtype == np.float32
    assert registration.inv_S.dtype == np.float32
    assert registration._low_rank_coefficients.dtype == np.float32
    assert registration.W.dtype == np.float32
