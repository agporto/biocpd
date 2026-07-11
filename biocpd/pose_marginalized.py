"""Annealed pose-marginalized initialization for atlas registration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from .atlas_registration import AtlasRegistration
from .utility import _farthest_indices


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


def _posterior_summary(scores: np.ndarray) -> tuple[float, float]:
    shifted = scores - np.min(scores)
    positive = shifted[shifted > 0]
    temperature = float(np.median(positive)) if len(positive) else 1.0
    temperature = max(temperature, np.finfo(float).eps)
    log_weights = -shifted / temperature
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= weights.sum()
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-15))))
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
    """Search global pose hypotheses and return the best refined atlas state.

    This is an opt-in initializer. It does not alter ``AtlasRegistration``
    defaults or run automatically during registration.
    """
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
    if modes.ndim == 3:
        modes = modes.reshape(source.size, modes.shape[2])
    if modes.shape != (source.size, len(eigenvalues)):
        raise ValueError("modes and eigenvalues do not match source")
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
        _, parameters = registration.register()
        diagnostics = registration.registration_diagnostics()
        score = float(diagnostics["objective"]) + (
            0.5
            * lambda_reg
            * float(diagnostics["coefficient_mahalanobis"])
        ) + prior_cost
        coarse_results.append((score, prior_cost, parameters))

    identity_result = coarse_results[0]
    coarse_results.sort(key=lambda result: result[0])
    finalists = coarse_results[: min(refine_count, len(coarse_results))]
    if not any(
        result[2] is identity_result[2] for result in finalists
    ):
        finalists[-1] = identity_result

    refined_target = target[
        _farthest_indices(target, min(refine_target_count, len(target)))
    ]
    refined_results = []
    for _, prior_cost, initial_parameters in finalists:
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
        coarse_coefficients = np.asarray(initial_parameters["b"]).reshape(-1)
        coefficients[: len(coarse_coefficients)] = coarse_coefficients
        registration.set_initial_state(
            coefficients,
            initial_parameters["R_world"],
            initial_parameters["s_world"],
            initial_parameters["t_world"],
            world_units=True,
        )
        _, parameters = registration.register()
        diagnostics = registration.registration_diagnostics()
        score = float(diagnostics["objective"]) + (
            0.5
            * lambda_reg
            * float(diagnostics["coefficient_mahalanobis"])
        ) + prior_cost
        refined_results.append((score, parameters))

    refined_results.sort(key=lambda result: result[0])
    scores = np.asarray([result[0] for result in refined_results])
    entropy, effective = _posterior_summary(scores)
    best_score, best = refined_results[0]
    margin = (
        float(refined_results[1][0] - best_score)
        if len(refined_results) > 1
        else np.inf
    )
    return PoseMarginalizedInitialization(
        coefficients=np.asarray(best["b"]).reshape(-1),
        rotation=np.asarray(best["R_world"]),
        scale=float(best["s_world"]),
        translation=np.asarray(best["t_world"]).reshape(1, 3),
        score=float(best_score),
        score_margin=margin,
        posterior_entropy=entropy,
        effective_hypotheses=effective,
        hypotheses_evaluated=len(rotations),
        hypotheses_refined=len(refined_results),
    )
