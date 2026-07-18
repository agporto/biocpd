"""Pose-marginalized initialization for atlas registration."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from ..atlas_registration import AtlasRegistration


@dataclass(frozen=True)
class PoseMarginalizedInitialization:
    """Warm-start state and diagnostics returned by pose initialization."""

    coefficients: np.ndarray
    rotation: np.ndarray
    scale: float
    translation: np.ndarray
    score: float
    score_margin: float
    posterior_entropy: float
    effective_hypotheses: float
    hypotheses_evaluated: int
    hypotheses_refined: int


@dataclass(frozen=True)
class PoseMarginalizedConfig:
    """Reusable configuration for pose-marginalized initialization."""

    rotation_count: int = 193
    coarse_source_count: int = 400
    coarse_target_count: int = 400
    coarse_rank: int = 12
    coarse_iterations: int = 8
    coarse_screen_iterations: int = 8
    coarse_survivor_count: int = 193
    coarse_score_mode: str = "trajectory"
    refine_count: int = 12
    refine_source_count: Optional[int] = None
    refine_target_count: int = 1600
    refine_iterations: int = 30
    lambda_reg: float = 0.01
    outlier_weight: float = 0.1
    identity_prior_probability: float = 0.2
    seed: int = 0
    n_jobs: int = 1

    def initialize(
        self,
        source: np.ndarray,
        target: np.ndarray,
        modes: np.ndarray,
        eigenvalues: np.ndarray,
    ) -> PoseMarginalizedInitialization:
        """Initialize an atlas registration using this configuration."""
        return pose_marginalized_initialization(
            source,
            target,
            modes,
            eigenvalues,
            rotation_count=self.rotation_count,
            coarse_source_count=self.coarse_source_count,
            coarse_target_count=self.coarse_target_count,
            coarse_rank=self.coarse_rank,
            coarse_iterations=self.coarse_iterations,
            coarse_screen_iterations=self.coarse_screen_iterations,
            coarse_survivor_count=self.coarse_survivor_count,
            coarse_score_mode=self.coarse_score_mode,
            refine_count=self.refine_count,
            refine_source_count=self.refine_source_count,
            refine_target_count=self.refine_target_count,
            refine_iterations=self.refine_iterations,
            lambda_reg=self.lambda_reg,
            outlier_weight=self.outlier_weight,
            identity_prior_probability=self.identity_prior_probability,
            seed=self.seed,
            n_jobs=self.n_jobs,
        )


@dataclass(frozen=True)
class _Candidate:
    score: float
    prior_cost: float
    coefficients: np.ndarray
    rotation: np.ndarray
    scale: float
    translation: np.ndarray


def _farthest_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Select a deterministic farthest-point subset."""
    points = np.asarray(points, dtype=np.float64)
    count = min(max(int(count), 1), len(points))
    centered = points - points.mean(axis=0)
    first = int(np.argmax(np.einsum("ij,ij->i", centered, centered)))
    selected = np.empty(count, dtype=int)
    selected[0] = first
    minimum_squared = np.sum((points - points[first]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(minimum_squared))
        squared = np.sum((points - points[selected[index]]) ** 2, axis=1)
        minimum_squared = np.minimum(minimum_squared, squared)
    return selected


def _subsample_indices(
    points: np.ndarray,
    count: Optional[int],
) -> np.ndarray:
    """Return all indices unchanged or a deterministic spatial subset."""
    if count is None or count >= len(points):
        return np.arange(len(points))
    return _farthest_indices(points, count)


def _rotation_lattice(count: int, seed: int) -> list[np.ndarray]:
    """Return a deterministic SO(3) lattice augmented near identity."""
    count = int(count)
    if count < 1:
        raise ValueError("rotation_count must be positive")

    remaining = count - 1
    local_count = remaining // 2
    global_count = remaining - local_count
    rotations = []
    if global_count:
        power = int(np.ceil(np.log2(global_count)))
        samples = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)
        u1, u2, u3 = samples[:global_count].T
        quaternions = np.column_stack(
            (
                np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
                np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
                np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
                np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
            )
        )
        rotations.extend(Rotation.from_quat(quaternions).as_matrix())

    shell_base, shell_remainder = divmod(local_count, 3)
    for shell_index, angle in enumerate((20.0, 40.0, 60.0)):
        shell_count = shell_base + int(shell_index < shell_remainder)
        if not shell_count:
            continue
        golden = np.pi * (3.0 - np.sqrt(5.0))
        indices = np.arange(shell_count)
        z = 1.0 - 2.0 * (indices + 0.5) / shell_count
        radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        axes = np.column_stack(
            (
                radius * np.cos(golden * indices),
                radius * np.sin(golden * indices),
                z,
            )
        )
        rotations.extend(
            Rotation.from_rotvec(np.radians(angle) * axes).as_matrix()
        )
    return [np.eye(3), *rotations]


def _initial_similarity(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
) -> tuple[float, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_radius = np.sqrt(
        np.mean(np.sum((source - source_center) ** 2, axis=1))
    )
    target_radius = np.sqrt(
        np.mean(np.sum((target - target_center) ** 2, axis=1))
    )
    scale = float(target_radius / max(source_radius, np.finfo(float).eps))
    translation = target_center - scale * (source_center @ rotation.T)
    return scale, translation.reshape(1, 3)


def _dense_data_objective(
    X: np.ndarray,
    TY: np.ndarray,
    sigma2: float,
    w: float,
    block_size: int,
) -> float:
    """Return the dense CPD negative log likelihood without mutating a registrar."""
    X = np.asarray(X, dtype=np.float64)
    TY = np.asarray(TY, dtype=np.float64)
    sigma2 = max(float(sigma2), np.finfo(np.float64).tiny)
    N, D = X.shape
    M = len(TY)
    block_size = max(1, min(int(block_size), N))
    ty_squared = np.sum(TY * TY, axis=1)[:, None]
    x_squared = np.sum(X * X, axis=1)
    log_inlier_normalizer = (
        np.log1p(-w)
        - np.log(M)
        - 0.5 * D * np.log(2.0 * np.pi * sigma2)
    )
    log_outlier = np.log(w) - np.log(N) if w > 0 else -np.inf
    log_c = log_outlier - log_inlier_normalizer
    fast_outlier_path = (
        w > 0
        and np.log(np.finfo(np.float64).eps) <= log_c
        and log_c <= np.log(np.finfo(np.float64).max) - np.log(2.0)
    )
    c = np.exp(log_c) if fast_outlier_path else None
    objective = 0.0

    for start in range(0, N, block_size):
        stop = min(start + block_size, N)
        X_block = X[start:stop]
        log_kernel = TY @ X_block.T
        log_kernel *= -2.0
        log_kernel += ty_squared
        log_kernel += x_squared[start:stop][None, :]
        np.maximum(log_kernel, 0.0, out=log_kernel)
        log_kernel *= -1.0 / (2.0 * sigma2)
        if fast_outlier_path:
            np.exp(log_kernel, out=log_kernel)
            density_sum = np.sum(log_kernel, axis=0)
            density_sum += c
            log_density = log_inlier_normalizer + np.log(density_sum)
        else:
            log_inlier = log_inlier_normalizer + logsumexp(
                log_kernel,
                axis=0,
            )
            log_density = np.logaddexp(log_inlier, log_outlier)
        objective -= float(np.sum(log_density))
    return objective


def _candidate_from_registration(
    registration: AtlasRegistration,
    lambda_reg: float,
    prior_cost: float,
    *,
    data_cost: Optional[float] = None,
    score_source: Optional[np.ndarray] = None,
    score_modes: Optional[np.ndarray] = None,
) -> _Candidate:
    """Capture and score a completed atlas registration without changing it."""
    parameters = registration.get_registration_parameters()
    coefficients = np.asarray(parameters["b"]).reshape(-1).copy()
    score_points = registration.TY
    if score_source is not None or score_modes is not None:
        if score_source is None or score_modes is None:
            raise ValueError(
                "score_source and score_modes must be provided together"
            )
        score_source = np.asarray(score_source, dtype=np.float64)
        score_modes = np.asarray(score_modes, dtype=np.float64).reshape(
            score_source.size,
            len(coefficients),
        )
        deformed = score_source + (score_modes @ coefficients).reshape(
            score_source.shape
        )
        score_points = (
            float(parameters["s_world"])
            * (deformed @ np.asarray(parameters["R_world"]).T)
            + np.asarray(parameters["t_world"]).reshape(1, registration.D)
        )
        if registration.normalize:
            score_points = (
                score_points - registration.target_centroid
            ) / registration.target_scale
    if data_cost is None:
        data_cost = _dense_data_objective(
            registration.X,
            score_points,
            registration.sigma2,
            registration.w,
            registration._get_dense_block_size(),
        )
    shape_cost = 0.5 * lambda_reg * float(
        np.sum(coefficients * coefficients * registration.invL)
    )
    return _Candidate(
        score=float(data_cost + shape_cost + prior_cost),
        prior_cost=float(prior_cost),
        coefficients=coefficients,
        rotation=np.asarray(parameters["R_world"]).copy(),
        scale=float(parameters["s_world"]),
        translation=np.asarray(parameters["t_world"]).reshape(1, 3).copy(),
    )


def _posterior_summary(scores: np.ndarray) -> tuple[float, float]:
    log_weights = -np.asarray(scores, dtype=np.float64)
    log_weights -= logsumexp(log_weights)
    weights = np.exp(log_weights)
    entropy = float(-np.sum(weights * log_weights))
    return entropy, float(np.exp(entropy))


def _select_finalists(
    coarse_results: list[_Candidate],
    refine_count: int,
) -> list[_Candidate]:
    identity_result = coarse_results[0]
    finalists = sorted(
        coarse_results,
        key=lambda result: result.score,
    )[: min(refine_count, len(coarse_results))]
    if len(finalists) > 1 and not any(
        result is identity_result for result in finalists
    ):
        finalists[-1] = identity_result
    return finalists


def _resolve_n_jobs(n_jobs: int) -> int:
    n_jobs = int(n_jobs)
    if n_jobs == -1:
        return max(os.cpu_count() or 1, 1)
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive or -1")
    return n_jobs


def _ordered_map(function, items, n_jobs: int):
    if n_jobs == 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(function, items))


def pose_marginalized_initialization(
    source: np.ndarray,
    target: np.ndarray,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    rotation_count: int = 193,
    coarse_source_count: int = 400,
    coarse_target_count: int = 400,
    coarse_rank: int = 12,
    coarse_iterations: int = 8,
    coarse_screen_iterations: int = 8,
    coarse_survivor_count: int = 193,
    coarse_score_mode: str = "trajectory",
    refine_count: int = 12,
    refine_source_count: Optional[int] = None,
    refine_target_count: int = 1600,
    refine_iterations: int = 30,
    lambda_reg: float = 0.01,
    outlier_weight: float = 0.1,
    identity_prior_probability: float = 0.2,
    seed: int = 0,
    n_jobs: int = 1,
) -> PoseMarginalizedInitialization:
    """Search global pose hypotheses and return the best refined atlas state."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.float64)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1] != 3
        or target.shape[1] != 3
        or len(source) == 0
        or len(target) == 0
    ):
        raise ValueError("source and target must have non-empty shape (N, 3)")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source and target must be finite")
    if eigenvalues.ndim != 1 or len(eigenvalues) == 0:
        raise ValueError("eigenvalues must be a non-empty one-dimensional array")
    if modes.ndim == 3:
        modes = modes.reshape(source.size, modes.shape[2])
    if modes.shape != (source.size, len(eigenvalues)):
        raise ValueError("modes and eigenvalues do not match source")
    if not np.isfinite(modes).all() or not np.isfinite(eigenvalues).all():
        raise ValueError("modes and eigenvalues must be finite")
    if rotation_count < 1 or refine_count < 1 or coarse_rank < 1:
        raise ValueError(
            "rotation_count, refine_count, and coarse_rank must be positive"
        )
    if coarse_iterations < 1 or refine_iterations < 1:
        raise ValueError("coarse_iterations and refine_iterations must be positive")
    if not 1 <= coarse_screen_iterations <= coarse_iterations:
        raise ValueError(
            "coarse_screen_iterations must be between 1 and coarse_iterations"
        )
    required_survivors = min(refine_count, rotation_count)
    if coarse_survivor_count < required_survivors:
        raise ValueError(
            "coarse_survivor_count must cover the requested finalists"
        )
    if coarse_score_mode not in {"trajectory", "final"}:
        raise ValueError(
            "coarse_score_mode must be 'trajectory' or 'final'"
        )
    if refine_source_count is not None and refine_source_count < 1:
        raise ValueError("refine_source_count must be positive or None")
    if refine_target_count < 1:
        raise ValueError("refine_target_count must be positive")
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be non-negative")
    if not 0 <= outlier_weight < 1:
        raise ValueError("outlier_weight must be in [0, 1)")
    if not 0 < identity_prior_probability < 1:
        raise ValueError("identity_prior_probability must be in (0, 1)")
    n_jobs = _resolve_n_jobs(n_jobs)

    source_indices = _subsample_indices(source, coarse_source_count)
    target_indices = _subsample_indices(target, coarse_target_count)
    coarse_source = source[source_indices]
    coarse_target = target[target_indices]
    rank = min(int(coarse_rank), len(eigenvalues))
    coarse_modes = modes.reshape(len(source), 3, -1)[source_indices, :, :rank]
    rotations = _rotation_lattice(rotation_count, seed)
    nonidentity_prior = (1.0 - identity_prior_probability) / max(
        len(rotations) - 1, 1
    )

    def coarse_candidate(
        registration: AtlasRegistration,
        prior_cost: float,
    ) -> _Candidate:
        data_cost = (
            registration.expectation_objective
            if coarse_score_mode == "trajectory"
            else None
        )
        return _candidate_from_registration(
            registration,
            lambda_reg,
            prior_cost,
            data_cost=data_cost,
        )

    def screen_rotation(rotation_item):
        rotation_index, rotation = rotation_item
        prior = (
            identity_prior_probability
            if rotation_index == 0
            else nonidentity_prior
        )
        prior_cost = -np.log(prior)
        scale, translation = _initial_similarity(
            coarse_source, coarse_target, rotation
        )
        registration = AtlasRegistration(
            X=coarse_target,
            Y=coarse_source,
            mean_shape=coarse_source,
            U=coarse_modes,
            eigenvalues=eigenvalues[:rank],
            lambda_reg=lambda_reg,
            normalize=True,
            dense_block_size=512,
            use_kdtree=False,
            w=outlier_weight,
            max_iterations=coarse_screen_iterations,
            tolerance=0.0,
            optimize_similarity=True,
            with_scale=True,
            dtype=np.float32,
        )
        registration.set_initial_similarity(
            rotation, scale, translation, world_units=True
        )
        registration.register(callback=None)
        return (
            coarse_candidate(registration, prior_cost),
            registration,
        )

    screened_registrations = _ordered_map(
        screen_rotation,
        enumerate(rotations),
        n_jobs,
    )

    if (
        coarse_screen_iterations < coarse_iterations
        and coarse_survivor_count < len(screened_registrations)
    ):
        screened_results = [result for result, _ in screened_registrations]
        survivors = _select_finalists(
            screened_results,
            coarse_survivor_count,
        )
        survivor_ids = {id(result) for result in survivors}
        screened_registrations = [
            item
            for item in screened_registrations
            if id(item[0]) in survivor_ids
        ]

    def complete_coarse(item):
        screened_result, registration = item
        if (
            registration.iteration < coarse_iterations
            and registration.diff > 0
        ):
            registration.max_iterations = coarse_iterations
            registration.register(callback=None)
            return coarse_candidate(
                registration,
                screened_result.prior_cost,
            )
        return screened_result

    coarse_results = _ordered_map(
        complete_coarse,
        screened_registrations,
        n_jobs,
    )

    finalists = _select_finalists(coarse_results, refine_count)

    refined_target = target[_subsample_indices(target, refine_target_count)]
    refine_source_indices = _subsample_indices(source, refine_source_count)
    refined_source = source[refine_source_indices]
    refined_modes = modes.reshape(len(source), 3, -1)[
        refine_source_indices
    ]

    def refine_finalist(initial):
        registration = AtlasRegistration(
            X=refined_target,
            Y=refined_source,
            mean_shape=refined_source,
            U=refined_modes,
            eigenvalues=eigenvalues,
            lambda_reg=lambda_reg,
            normalize=True,
            dense_block_size=1024,
            use_kdtree=False,
            w=outlier_weight,
            max_iterations=refine_iterations,
            tolerance=0.0,
            optimize_similarity=True,
            with_scale=True,
            dtype=np.float32,
        )
        coefficients = np.zeros(len(eigenvalues), dtype=np.float64)
        coefficients[: len(initial.coefficients)] = initial.coefficients
        registration.set_initial_state(
            coefficients,
            initial.rotation,
            initial.scale,
            initial.translation,
            world_units=True,
        )
        registration.register(callback=None)
        return _candidate_from_registration(
            registration,
            lambda_reg,
            initial.prior_cost,
            score_source=source,
            score_modes=modes,
        )

    refined_results = _ordered_map(
        refine_finalist,
        finalists,
        n_jobs,
    )

    refined_results.sort(key=lambda result: result.score)
    scores = np.asarray([result.score for result in refined_results])
    entropy, effective = _posterior_summary(scores)
    best = refined_results[0]
    margin = (
        float(refined_results[1].score - best.score)
        if len(refined_results) > 1
        else np.inf
    )
    return PoseMarginalizedInitialization(
        coefficients=best.coefficients.copy(),
        rotation=best.rotation.copy(),
        scale=best.scale,
        translation=best.translation.copy(),
        score=best.score,
        score_margin=margin,
        posterior_entropy=entropy,
        effective_hypotheses=effective,
        hypotheses_evaluated=len(rotations),
        hypotheses_refined=len(refined_results),
    )
