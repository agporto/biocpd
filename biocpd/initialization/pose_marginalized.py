"""Pose-marginalized initialization for atlas registration."""

from __future__ import annotations

from dataclasses import dataclass

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

    rotation_count: int = 96
    coarse_source_count: int = 400
    coarse_target_count: int = 400
    coarse_rank: int = 12
    coarse_iterations: int = 8
    refine_count: int = 12
    refine_target_count: int = 1600
    refine_iterations: int = 30
    lambda_reg: float = 0.01
    outlier_weight: float = 0.1
    identity_prior_probability: float = 0.2
    seed: int = 0

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
            refine_count=self.refine_count,
            refine_target_count=self.refine_target_count,
            refine_iterations=self.refine_iterations,
            lambda_reg=self.lambda_reg,
            outlier_weight=self.outlier_weight,
            identity_prior_probability=self.identity_prior_probability,
            seed=self.seed,
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


def _rotation_lattice(count: int, seed: int) -> list[np.ndarray]:
    """Return a deterministic SO(3) lattice augmented near identity."""
    count = max(int(count), 12)
    power = int(np.ceil(np.log2(count)))
    samples = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)[:count]
    u1, u2, u3 = samples.T
    quaternions = np.column_stack(
        (
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        )
    )
    rotations = list(Rotation.from_quat(quaternions).as_matrix())

    shell_count = max(12, count // 3)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(shell_count)
    z = 1.0 - 2.0 * (indices + 0.5) / shell_count
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    axes = np.column_stack(
        (radius * np.cos(golden * indices), radius * np.sin(golden * indices), z)
    )
    for angle in (20.0, 40.0, 60.0):
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
    log_inlier_normalizer = (
        np.log1p(-w)
        - np.log(M)
        - 0.5 * D * np.log(2.0 * np.pi * sigma2)
    )
    log_outlier = np.log(w) - np.log(N) if w > 0 else -np.inf
    objective = 0.0

    for start in range(0, N, block_size):
        stop = min(start + block_size, N)
        difference = TY[:, None, :] - X[None, start:stop, :]
        log_kernel = -np.sum(difference * difference, axis=2) / (2.0 * sigma2)
        log_inlier = log_inlier_normalizer + logsumexp(log_kernel, axis=0)
        log_density = np.logaddexp(log_inlier, log_outlier)
        objective -= float(np.sum(log_density))
    return objective


def _candidate_from_registration(
    registration: AtlasRegistration,
    lambda_reg: float,
    prior_cost: float,
) -> _Candidate:
    """Capture and score a completed atlas registration without changing it."""
    parameters = registration.get_registration_parameters()
    data_cost = _dense_data_objective(
        registration.X,
        registration.TY,
        registration.sigma2,
        registration.w,
        registration._get_dense_block_size(),
    )
    coefficients = np.asarray(parameters["b"]).reshape(-1).copy()
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


def pose_marginalized_initialization(
    source: np.ndarray,
    target: np.ndarray,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    rotation_count: int = 96,
    coarse_source_count: int = 400,
    coarse_target_count: int = 400,
    coarse_rank: int = 12,
    coarse_iterations: int = 8,
    refine_count: int = 12,
    refine_target_count: int = 1600,
    refine_iterations: int = 30,
    lambda_reg: float = 0.01,
    outlier_weight: float = 0.1,
    identity_prior_probability: float = 0.2,
    seed: int = 0,
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
    if refine_count < 1 or coarse_rank < 1:
        raise ValueError("refine_count and coarse_rank must be positive")
    if coarse_iterations < 1 or refine_iterations < 1:
        raise ValueError("coarse_iterations and refine_iterations must be positive")
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be non-negative")
    if not 0 <= outlier_weight < 1:
        raise ValueError("outlier_weight must be in [0, 1)")
    if not 0 < identity_prior_probability < 1:
        raise ValueError("identity_prior_probability must be in (0, 1)")

    source_indices = _farthest_indices(source, coarse_source_count)
    target_indices = _farthest_indices(target, coarse_target_count)
    coarse_source = source[source_indices]
    coarse_target = target[target_indices]
    rank = min(int(coarse_rank), len(eigenvalues))
    coarse_modes = modes.reshape(len(source), 3, -1)[source_indices, :, :rank]
    rotations = _rotation_lattice(rotation_count, seed)
    coarse_results = []
    nonidentity_prior = (1.0 - identity_prior_probability) / max(
        len(rotations) - 1, 1
    )

    for rotation_index, rotation in enumerate(rotations):
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
            max_iterations=coarse_iterations,
            tolerance=0.0,
            optimize_similarity=True,
            with_scale=True,
            dtype=np.float32,
        )
        registration.set_initial_similarity(
            rotation, scale, translation, world_units=True
        )
        registration.register()
        coarse_results.append(
            _candidate_from_registration(registration, lambda_reg, prior_cost)
        )

    identity_result = coarse_results[0]
    coarse_results.sort(key=lambda result: result.score)
    finalists = coarse_results[: min(refine_count, len(coarse_results))]
    if not any(result is identity_result for result in finalists):
        finalists[-1] = identity_result

    refined_target = target[
        _farthest_indices(target, min(refine_target_count, len(target)))
    ]
    refined_results = []
    for initial in finalists:
        registration = AtlasRegistration(
            X=refined_target,
            Y=source,
            mean_shape=source,
            U=modes,
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
        registration.register()
        refined_results.append(
            _candidate_from_registration(
                registration,
                lambda_reg,
                initial.prior_cost,
            )
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
