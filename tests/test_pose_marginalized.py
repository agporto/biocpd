import inspect

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import biocpd.initialization.pose_marginalized as pose_module
from biocpd import (
    AffineRegistration,
    AtlasRegistration,
    PoseMarginalizedConfig,
    PoseMarginalizedInitialization,
    RigidRegistration,
    pose_marginalized_initialization,
)
from biocpd.initialization.pose_marginalized import (
    _Candidate,
    _candidate_from_registration,
    _dense_data_objective,
    _rotation_lattice,
    _select_finalists,
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
    for count in (1, 12, 24, 96, 193):
        first = _rotation_lattice(count, 13)
        second = _rotation_lattice(count, 13)
        assert len(first) == len(second) == count
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            np.testing.assert_allclose(left.T @ left, np.eye(3), atol=1e-12)
            assert np.linalg.det(left) > 0
        np.testing.assert_array_equal(first[0], np.eye(3))


def test_single_refinement_keeps_best_coarse_hypothesis():
    def candidate(score):
        return _Candidate(
            score=score,
            prior_cost=0.0,
            coefficients=np.zeros(1),
            rotation=np.eye(3),
            scale=1.0,
            translation=np.zeros((1, 3)),
        )

    identity = candidate(3.0)
    best_nonidentity = candidate(1.0)
    second_nonidentity = candidate(2.0)
    coarse_results = [identity, best_nonidentity, second_nonidentity]

    single = _select_finalists(coarse_results, refine_count=1)
    multiple = _select_finalists(coarse_results, refine_count=2)

    assert len(single) == 1
    assert single[0] is best_nonidentity
    assert multiple[0] is best_nonidentity
    assert multiple[1] is identity


def test_pose_search_defaults_use_exhaustive_coarse_and_full_source():
    config = PoseMarginalizedConfig()
    signature = inspect.signature(pose_marginalized_initialization)

    assert config.coarse_screen_iterations == config.coarse_iterations == 8
    assert config.coarse_survivor_count == config.rotation_count == 193
    assert config.refine_source_count is None
    assert signature.parameters["coarse_screen_iterations"].default == 8
    assert signature.parameters["coarse_survivor_count"].default == 193
    assert signature.parameters["refine_source_count"].default is None


def test_full_source_refinement_preserves_all_source_points(monkeypatch):
    source = _asymmetric_cloud(seed=31, count=18)
    target = source + np.array([0.2, -0.1, 0.05])
    modes = np.zeros((source.size, 1))
    refinement_sources = []

    class RecordingAtlasRegistration(AtlasRegistration):
        def __init__(self, *args, **kwargs):
            if len(kwargs["Y"]) == len(source):
                refinement_sources.append(kwargs["Y"].copy())
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        pose_module,
        "AtlasRegistration",
        RecordingAtlasRegistration,
    )
    pose_marginalized_initialization(
        source,
        target,
        modes,
        np.ones(1),
        rotation_count=1,
        coarse_source_count=8,
        coarse_target_count=8,
        coarse_rank=1,
        coarse_iterations=1,
        coarse_screen_iterations=1,
        coarse_survivor_count=1,
        refine_count=1,
        refine_source_count=None,
        refine_target_count=12,
        refine_iterations=1,
    )

    assert len(refinement_sources) == 1
    np.testing.assert_array_equal(refinement_sources[0], source)


def test_nonpositive_refine_source_count_is_rejected():
    source = _asymmetric_cloud(seed=32, count=8)
    with pytest.raises(ValueError, match="positive or None"):
        pose_marginalized_initialization(
            source,
            source.copy(),
            np.zeros((source.size, 1)),
            np.ones(1),
            rotation_count=1,
            coarse_rank=1,
            coarse_iterations=1,
            coarse_screen_iterations=1,
            coarse_survivor_count=1,
            refine_count=1,
            refine_source_count=0,
            refine_iterations=1,
        )


def test_config_preserves_function_api_and_result_type():
    source = _asymmetric_cloud(seed=8, count=30)
    modes = np.zeros((source.size, 1))
    config = PoseMarginalizedConfig(
        rotation_count=12,
        coarse_source_count=20,
        coarse_target_count=20,
        coarse_rank=1,
        coarse_iterations=2,
        coarse_screen_iterations=1,
        coarse_survivor_count=2,
        refine_count=2,
        refine_target_count=30,
        refine_iterations=2,
        seed=5,
    )
    result = config.initialize(source, source.copy(), modes, np.ones(1))
    assert isinstance(result, PoseMarginalizedInitialization)
    assert result.hypotheses_evaluated == 12
    assert np.isfinite(result.score)


@pytest.mark.parametrize("outlier_weight", [0.0, 0.1, 0.6])
@pytest.mark.parametrize("sigma2", [1e-8, 0.35, 100.0])
@pytest.mark.parametrize("block_size", [1, 5, 20])
def test_dense_pose_objective_matches_brute_force_likelihood(
    outlier_weight,
    sigma2,
    block_size,
):
    source = _asymmetric_cloud(seed=14, count=12)
    target = source + np.array([0.3, -0.2, 0.1])
    score = _dense_data_objective(
        target,
        source,
        sigma2,
        outlier_weight,
        block_size=block_size,
    )

    difference = source[:, None, :] - target[None, :, :]
    log_kernel = -np.sum(difference * difference, axis=2) / (2.0 * sigma2)
    log_inlier = (
        np.log1p(-outlier_weight)
        - np.log(len(source))
        - 1.5 * np.log(2.0 * np.pi * sigma2)
        + np.logaddexp.reduce(log_kernel, axis=0)
    )
    log_outlier = (
        np.log(outlier_weight) - np.log(len(target))
        if outlier_weight > 0
        else -np.inf
    )
    expected = -float(np.sum(np.logaddexp(log_inlier, log_outlier)))
    assert np.isclose(score, expected, atol=1e-9, rtol=1e-12)


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


def test_pose_candidate_can_score_a_full_source_after_subset_fitting():
    source = _asymmetric_cloud(seed=16, count=30)
    modes = np.zeros((source.size, 2))
    modes[::3, 0] = 0.03 * source[:, 0]
    modes[1::3, 1] = 0.02 * source[:, 1]
    subset = np.arange(0, len(source), 2)
    registration = AtlasRegistration(
        X=source + np.array([0.1, -0.05, 0.02]),
        Y=source[subset],
        mean_shape=source[subset],
        U=modes.reshape(len(source), 3, -1)[subset],
        eigenvalues=np.array([0.4, 0.2]),
        lambda_reg=0.1,
        normalize=True,
        use_kdtree=False,
        w=0.1,
        dtype=np.float64,
        max_iterations=3,
        tolerance=0.0,
    )
    registration.register(callback=None)
    candidate = _candidate_from_registration(
        registration,
        0.1,
        0.7,
        score_source=source,
        score_modes=modes,
    )
    full_world = candidate.scale * (
        (
            source
            + (modes @ candidate.coefficients).reshape(source.shape)
        )
        @ candidate.rotation.T
    ) + candidate.translation
    full_normalized = (
        full_world - registration.target_centroid
    ) / registration.target_scale
    expected_data = _dense_data_objective(
        registration.X,
        full_normalized,
        registration.sigma2,
        registration.w,
        registration._get_dense_block_size(),
    )
    expected_shape = 0.05 * np.sum(
        candidate.coefficients**2 * registration.invL
    )
    assert np.isclose(
        candidate.score,
        expected_data + expected_shape + 0.7,
    )


def test_parallel_pose_search_matches_serial_search():
    source = _asymmetric_cloud(seed=18, count=36)
    modes = np.zeros((source.size, 2))
    modes[::3, 0] = 0.04 * source[:, 0]
    modes[1::3, 1] = 0.03 * source[:, 1]
    target = source + np.array([0.2, -0.1, 0.05])
    common = dict(
        source=source,
        target=target,
        modes=modes,
        eigenvalues=np.array([0.4, 0.2]),
        rotation_count=12,
        coarse_source_count=24,
        coarse_target_count=24,
        coarse_rank=2,
        coarse_iterations=2,
        coarse_screen_iterations=1,
        coarse_survivor_count=4,
        refine_count=2,
        refine_source_count=24,
        refine_target_count=30,
        refine_iterations=3,
        seed=4,
    )
    serial = pose_marginalized_initialization(**common, n_jobs=1)
    parallel = pose_marginalized_initialization(**common, n_jobs=2)

    np.testing.assert_array_equal(serial.coefficients, parallel.coefficients)
    np.testing.assert_array_equal(serial.rotation, parallel.rotation)
    np.testing.assert_array_equal(serial.translation, parallel.translation)
    assert serial.scale == parallel.scale
    assert serial.score == parallel.score
    assert serial.score_margin == parallel.score_margin
    assert serial.posterior_entropy == parallel.posterior_entropy
    assert serial.effective_hypotheses == parallel.effective_hypotheses


def test_staged_search_only_completes_surviving_coarse_hypotheses(
    monkeypatch,
):
    source = _asymmetric_cloud(seed=23, count=30)
    target = source + np.array([0.15, -0.08, 0.04])
    modes = np.zeros((source.size, 1))
    iteration_count = 0
    original_iterate = AtlasRegistration.iterate

    def counted_iterate(registration):
        nonlocal iteration_count
        iteration_count += 1
        return original_iterate(registration)

    monkeypatch.setattr(AtlasRegistration, "iterate", counted_iterate)
    result = pose_marginalized_initialization(
        source,
        target,
        modes,
        np.ones(1),
        rotation_count=12,
        coarse_source_count=20,
        coarse_target_count=20,
        coarse_rank=1,
        coarse_iterations=3,
        coarse_screen_iterations=1,
        coarse_survivor_count=4,
        refine_count=2,
        refine_source_count=24,
        refine_target_count=24,
        refine_iterations=2,
        seed=6,
    )

    assert result.hypotheses_evaluated == 12
    assert result.hypotheses_refined == 2
    assert 12 <= iteration_count <= 24


@pytest.mark.parametrize("refine_count", [1, 4])
def test_pose_initializer_recovers_large_rotation(refine_count):
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
        coarse_screen_iterations=1,
        coarse_survivor_count=8,
        refine_count=refine_count,
        refine_source_count=40,
        refine_target_count=80,
        refine_iterations=16,
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


@pytest.mark.parametrize(
    "angles",
    [
        (85.0, 35.0, -70.0),
        (120.0, -60.0, 95.0),
        (-140.0, 20.0, 170.0),
    ],
)
def test_default_rotation_coverage_survives_staged_screening(angles):
    source = _asymmetric_cloud(seed=29, count=80)
    modes = np.zeros((source.size, 2))
    modes[::3, 0] = 0.05 * source[:, 0]
    modes[1::3, 1] = 0.04 * source[:, 1]
    rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    target = 1.08 * (source @ rotation.T) + np.array([0.8, -0.4, 0.2])
    result = pose_marginalized_initialization(
        source,
        target,
        modes,
        np.array([0.4, 0.2]),
        rotation_count=193,
        coarse_source_count=60,
        coarse_target_count=60,
        coarse_rank=2,
        coarse_iterations=5,
        coarse_screen_iterations=1,
        coarse_survivor_count=48,
        refine_count=4,
        refine_source_count=60,
        refine_target_count=80,
        refine_iterations=20,
        seed=7,
    )
    recovered = result.scale * (
        (source + (modes @ result.coefficients).reshape(source.shape))
        @ result.rotation.T
    ) + result.translation
    diagonal = np.linalg.norm(np.ptp(target, axis=0))
    normalized_error = (
        np.median(np.linalg.norm(recovered - target, axis=1)) / diagonal
    )
    assert normalized_error < 0.03
