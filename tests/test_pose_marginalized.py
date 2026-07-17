import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from biocpd import (
    AffineRegistration,
    AtlasRegistration,
    PoseMarginalizedConfig,
    PoseMarginalizedInitialization,
    RigidRegistration,
    pose_marginalized_initialization,
)
from biocpd.initialization.pose_marginalized import (
    _candidate_from_registration,
    _dense_data_objective,
    _rotation_lattice,
)


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


@pytest.mark.parametrize("registration_class", [RigidRegistration, AffineRegistration])
def test_dense_expectation_does_not_mutate_legacy_q(registration_class):
    source = _asymmetric_cloud(seed=21, count=24)
    target = source + np.array([0.2, -0.1, 0.05])
    registration = registration_class(
        X=target,
        Y=source,
        use_kdtree=False,
        max_iterations=2,
    )
    registration.q = 17.25
    registration.expectation()
    assert registration.q == 17.25


@pytest.mark.parametrize("registration_class", [RigidRegistration, AffineRegistration])
def test_dense_first_iteration_preserves_legacy_convergence_history(
    registration_class,
):
    source = _asymmetric_cloud(seed=22, count=24)
    target = source + np.array([0.2, -0.1, 0.05])
    registration = registration_class(
        X=target,
        Y=source,
        use_kdtree=False,
        max_iterations=1,
    )
    registration.register()
    assert np.isinf(registration.diff)


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


def test_initial_state_is_rejected_after_registration_starts():
    source = _asymmetric_cloud(seed=12, count=20)
    registration = AtlasRegistration(
        X=source.copy(),
        Y=source,
        U=np.zeros((source.size, 1)),
        eigenvalues=np.ones(1),
        use_kdtree=False,
        max_iterations=1,
    )
    registration.register()
    with pytest.raises(RuntimeError, match="before registration starts"):
        registration.set_initial_coefficients(np.zeros(1))


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


def test_dense_pose_objective_matches_brute_force_likelihood():
    source = _asymmetric_cloud(seed=14, count=12)
    target = source + np.array([0.3, -0.2, 0.1])
    sigma2 = 0.35
    outlier_weight = 0.15
    score = _dense_data_objective(
        target,
        source,
        sigma2,
        outlier_weight,
        block_size=5,
    )

    difference = source[:, None, :] - target[None, :, :]
    kernel = np.exp(-np.sum(difference * difference, axis=2) / (2.0 * sigma2))
    inlier = (
        (1.0 - outlier_weight)
        * np.sum(kernel, axis=0)
        / (len(source) * (2.0 * np.pi * sigma2) ** 1.5)
    )
    density = inlier + outlier_weight / len(target)
    expected = -float(np.sum(np.log(density)))
    assert np.isclose(score, expected, atol=1e-12, rtol=1e-12)


def test_pose_candidate_scoring_is_pure_and_uses_final_state():
    source = _asymmetric_cloud(seed=15, count=20)
    modes = np.zeros((source.size, 2))
    modes[::3, 0] = 0.03 * source[:, 0]
    modes[1::3, 1] = 0.02 * source[:, 1]
    registration = AtlasRegistration(
        X=source + np.array([0.1, -0.05, 0.02]),
        Y=source,
        mean_shape=source,
        U=modes,
        eigenvalues=np.array([0.4, 0.2]),
        lambda_reg=0.1,
        normalize=True,
        use_kdtree=False,
        w=0.1,
        dtype=np.float64,
        max_iterations=3,
        tolerance=0.0,
    )
    registration.register()
    state = {
        "q": registration.q,
        "diff": registration.diff,
        "iteration": registration.iteration,
        "sigma2": registration.sigma2,
        "b": registration.b.copy(),
        "R": registration.R.copy(),
        "t": registration.t.copy(),
        "TY": registration.TY.copy(),
    }
    prior_cost = 0.7
    candidate = _candidate_from_registration(registration, 0.1, prior_cost)
    expected_data = _dense_data_objective(
        registration.X,
        registration.TY,
        registration.sigma2,
        registration.w,
        registration._get_dense_block_size(),
    )
    expected_shape = 0.05 * np.sum(
        registration.b.reshape(-1) ** 2 * registration.invL
    )

    assert np.isclose(candidate.score, expected_data + expected_shape + prior_cost)
    assert registration.q == state["q"]
    assert registration.diff == state["diff"]
    assert registration.iteration == state["iteration"]
    assert registration.sigma2 == state["sigma2"]
    np.testing.assert_array_equal(registration.b, state["b"])
    np.testing.assert_array_equal(registration.R, state["R"])
    np.testing.assert_array_equal(registration.t, state["t"])
    np.testing.assert_array_equal(registration.TY, state["TY"])


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
