"""Optional initialization strategies for registration algorithms."""

from .pose_marginalized import (
    PoseMarginalizedConfig,
    PoseMarginalizedInitialization,
    pose_marginalized_initialization,
)


__all__ = [
    "PoseMarginalizedConfig",
    "PoseMarginalizedInitialization",
    "pose_marginalized_initialization",
]
