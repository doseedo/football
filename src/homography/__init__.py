"""Homography and calibration modules."""

from .calibration import (
    HomographyEstimator,
    InteractiveCalibrator,
    CoordinateTransformer,
    DynamicCoordinateTransformer,
    CalibrationResult,
)
from .pitch_detector import (
    PitchLineDetector,
    PitchKeypoint,
    PitchDetectionResult,
    FeatureBasedPitchMatcher,
)
from .rotation_handler import (
    RotationHandler,
    AdaptiveHomographyManager,
    CameraState,
    RotationState,
    DynamicHomography,
)
from .field_model import (
    FootballPitchModel,
    Keypoint3D,
    create_standard_pitch,
)
from .keypoint_detector import (
    PitchKeypointDetector,
    DetectedKeypoint,
    KeypointDetectionResult,
    create_keypoint_detector,
    visualize_detections,
)
from .auto_calibration import (
    AutoCalibrator,
    AutoCalibrationResult,
    CalibrationQuality,
    visualize_calibration,
)
from .bayesian_filter import (
    BayesianHomographyFilter,
    KeypointState,
    HomographyState,
    create_bayesian_filter,
)

__all__ = [
    # Calibration
    "HomographyEstimator",
    "InteractiveCalibrator",
    "CoordinateTransformer",
    "DynamicCoordinateTransformer",
    "CalibrationResult",
    # Pitch detection
    "PitchLineDetector",
    "PitchKeypoint",
    "PitchDetectionResult",
    "FeatureBasedPitchMatcher",
    # Rotation handling
    "RotationHandler",
    "AdaptiveHomographyManager",
    "CameraState",
    "RotationState",
    "DynamicHomography",
    # Field model
    "FootballPitchModel",
    "Keypoint3D",
    "create_standard_pitch",
    # Automatic keypoint detection
    "PitchKeypointDetector",
    "DetectedKeypoint",
    "KeypointDetectionResult",
    "create_keypoint_detector",
    "visualize_detections",
    # Automatic calibration
    "AutoCalibrator",
    "AutoCalibrationResult",
    "CalibrationQuality",
    "visualize_calibration",
    # Bayesian filtering
    "BayesianHomographyFilter",
    "KeypointState",
    "HomographyState",
    "create_bayesian_filter",
]
