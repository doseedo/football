"""
Training utilities for football tracking models.
"""

from .losses import (
    KeypointHeatmapLoss,
    ReprojectionLoss,
    TrajectoryLoss,
    ContrastiveLoss,
    TripletLoss,
    CenterLoss,
    HungarianMatchingLoss,
    PhysicsConstraintLoss,
)

from .metrics import (
    ReprojectionError,
    KeypointPCK,
    TrajectoryADE,
    TrajectoryFDE,
    ReIDMetrics,
    DetectionMetrics,
    FormationAccuracy,
)

from .callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
    WandbLogger,
    VisualizationCallback,
)

__all__ = [
    # Losses
    "KeypointHeatmapLoss",
    "ReprojectionLoss",
    "TrajectoryLoss",
    "ContrastiveLoss",
    "TripletLoss",
    "CenterLoss",
    "HungarianMatchingLoss",
    "PhysicsConstraintLoss",
    # Metrics
    "ReprojectionError",
    "KeypointPCK",
    "TrajectoryADE",
    "TrajectoryFDE",
    "ReIDMetrics",
    "DetectionMetrics",
    "FormationAccuracy",
    # Callbacks
    "ModelCheckpoint",
    "EarlyStopping",
    "LearningRateScheduler",
    "WandbLogger",
    "VisualizationCallback",
]
