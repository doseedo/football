"""
Training datasets for football tracking models.
"""

from .soccernet_loader import (
    SoccerNetCalibrationDataset,
    SoccerNetTrackingDataset,
    SoccerNetDetectionDataset,
)
from .skillcorner_loader import SkillCornerDataset
from .synthetic_loader import (
    SyntheticBall3DDataset,
    SyntheticTrajectoryDataset,
    SyntheticHomographyDataset,
)
from .augmentations import (
    get_train_transforms,
    get_val_transforms,
    TrajectoryAugmentation,
)

__all__ = [
    "SoccerNetCalibrationDataset",
    "SoccerNetTrackingDataset",
    "SoccerNetDetectionDataset",
    "SkillCornerDataset",
    "SyntheticBall3DDataset",
    "SyntheticTrajectoryDataset",
    "SyntheticHomographyDataset",
    "get_train_transforms",
    "get_val_transforms",
    "TrajectoryAugmentation",
]
