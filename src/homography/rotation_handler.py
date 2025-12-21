"""Camera rotation detection and dynamic homography compensation."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger

from .pitch_detector import FeatureBasedPitchMatcher, PitchLineDetector


class CameraState(Enum):
    STABLE = "stable"
    ROTATING = "rotating"
    STABILIZING = "stabilizing"


@dataclass
class RotationState:
    state: CameraState
    rotation_angle: float  # Cumulative rotation from reference
    angular_velocity: float  # Current rotation speed (degrees/frame)
    frames_in_state: int


@dataclass
class DynamicHomography:
    """Homography with temporal tracking."""
    H: np.ndarray  # Current homography (pixel to world)
    H_frame_to_ref: np.ndarray  # Frame to reference homography
    timestamp: float
    confidence: float
    is_interpolated: bool = False


class RotationHandler:
    """Handles camera rotation detection and homography updates."""

    def __init__(
        self,
        max_rotation_angle: float = 45.0,
        rotation_threshold: float = 0.5,  # degrees/frame to consider rotating
        stabilization_frames: int = 10,  # frames to consider stable
        homography_buffer_size: int = 30,
        smoothing_factor: float = 0.3,
    ):
        self.max_rotation_angle = max_rotation_angle
        self.rotation_threshold = rotation_threshold
        self.stabilization_frames = stabilization_frames
        self.homography_buffer_size = homography_buffer_size
        self.smoothing_factor = smoothing_factor

        # Pitch detection and feature matching
        self.pitch_detector = PitchLineDetector()
        self.feature_matcher = FeatureBasedPitchMatcher()

        # State tracking
        self.rotation_state = RotationState(
            state=CameraState.STABLE,
            rotation_angle=0.0,
            angular_velocity=0.0,
            frames_in_state=0
        )

        # Homography tracking
        self.base_homography: Optional[np.ndarray] = None  # Initial calibration H
        self.current_homography: Optional[np.ndarray] = None
        self.homography_buffer: Deque[DynamicHomography] = deque(maxlen=homography_buffer_size)

        # Reference frame tracking
        self.reference_frame: Optional[np.ndarray] = None
        self.reference_pitch_mask: Optional[np.ndarray] = None

        # Frame-to-frame tracking
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_H: Optional[np.ndarray] = None

        self.frame_count = 0
        self.is_initialized = False

    def initialize(self, frame: np.ndarray, base_homography: np.ndarray, pitch_mask: Optional[np.ndarray] = None):
        """Initialize rotation handler with reference frame and base homography."""
        self.reference_frame = frame.copy()
        self.reference_pitch_mask = pitch_mask
        self.base_homography = base_homography.copy()
        self.current_homography = base_homography.copy()

        self.feature_matcher.set_reference_frame(frame, pitch_mask)

        self.prev_frame = frame.copy()
        self.prev_H = np.eye(3)

        self.is_initialized = True
        logger.info("Rotation handler initialized")

    def update(self, frame: np.ndarray, timestamp: float, pitch_mask: Optional[np.ndarray] = None) -> DynamicHomography:
        """Process new frame and update homography if camera has rotated."""
        self.frame_count += 1

        if not self.is_initialized:
            raise ValueError("Rotation handler not initialized. Call initialize() first.")

        # Compute frame-to-reference transform
        H_to_ref = self.feature_matcher.compute_frame_transform(frame, pitch_mask)

        if H_to_ref is not None:
            # Successful match - update state
            rotation_angle = self.feature_matcher.estimate_rotation_angle(H_to_ref)
            angular_velocity = self._compute_angular_velocity(rotation_angle)

            self._update_rotation_state(angular_velocity)

            # Compute world homography: combine base homography with frame transform
            # new_H transforms current frame pixels to world coordinates
            # H_to_ref transforms current frame to reference frame
            # base_homography transforms reference frame to world
            # So: new_H = base_homography @ H_to_ref^(-1)... but we need the inverse relationship

            # Actually: if H_to_ref maps current -> reference
            # And base_homography maps reference pixels -> world
            # Then to map current pixels -> world:
            # First apply H_to_ref to get reference pixels, then apply base_homography
            # world = base_homography @ reference_pixel
            # reference_pixel = H_to_ref @ current_pixel
            # world = base_homography @ H_to_ref @ current_pixel

            # But H_to_ref maps current to reference, so:
            new_H = self.base_homography @ H_to_ref

            # Smooth the homography
            smoothed_H = self._smooth_homography(new_H)

            self.current_homography = smoothed_H
            self.rotation_state.rotation_angle = rotation_angle

            dyn_H = DynamicHomography(
                H=smoothed_H,
                H_frame_to_ref=H_to_ref,
                timestamp=timestamp,
                confidence=0.9,
                is_interpolated=False
            )

        else:
            # Feature matching failed - use interpolation
            dyn_H = self._interpolate_homography(timestamp)
            self._update_rotation_state(0.0, match_failed=True)

        self.homography_buffer.append(dyn_H)
        self.prev_frame = frame.copy()
        self.prev_H = dyn_H.H_frame_to_ref if dyn_H.H_frame_to_ref is not None else np.eye(3)

        return dyn_H

    def _compute_angular_velocity(self, current_angle: float) -> float:
        """Compute angular velocity from angle change."""
        if len(self.homography_buffer) == 0:
            return 0.0

        prev_angle = self.rotation_state.rotation_angle
        return current_angle - prev_angle

    def _update_rotation_state(self, angular_velocity: float, match_failed: bool = False):
        """Update camera rotation state machine."""
        self.rotation_state.angular_velocity = angular_velocity
        abs_velocity = abs(angular_velocity)

        current_state = self.rotation_state.state

        if match_failed:
            # If matching failed, assume we're rotating
            if current_state != CameraState.ROTATING:
                self.rotation_state.state = CameraState.ROTATING
                self.rotation_state.frames_in_state = 0
            else:
                self.rotation_state.frames_in_state += 1
            return

        if current_state == CameraState.STABLE:
            if abs_velocity > self.rotation_threshold:
                self.rotation_state.state = CameraState.ROTATING
                self.rotation_state.frames_in_state = 0
                logger.debug(f"Camera started rotating: {angular_velocity:.2f}°/frame")
            else:
                self.rotation_state.frames_in_state += 1

        elif current_state == CameraState.ROTATING:
            if abs_velocity < self.rotation_threshold:
                self.rotation_state.state = CameraState.STABILIZING
                self.rotation_state.frames_in_state = 0
            else:
                self.rotation_state.frames_in_state += 1

        elif current_state == CameraState.STABILIZING:
            if abs_velocity > self.rotation_threshold:
                self.rotation_state.state = CameraState.ROTATING
                self.rotation_state.frames_in_state = 0
            elif self.rotation_state.frames_in_state >= self.stabilization_frames:
                self.rotation_state.state = CameraState.STABLE
                self.rotation_state.frames_in_state = 0
                logger.debug(f"Camera stabilized at {self.rotation_state.rotation_angle:.1f}°")
            else:
                self.rotation_state.frames_in_state += 1

    def _smooth_homography(self, new_H: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to homography."""
        if self.current_homography is None or len(self.homography_buffer) == 0:
            return new_H

        # Use exponential smoothing
        # More smoothing when stable, less when rotating
        if self.rotation_state.state == CameraState.STABLE:
            alpha = self.smoothing_factor * 0.5  # More smoothing when stable
        elif self.rotation_state.state == CameraState.ROTATING:
            alpha = self.smoothing_factor * 2.0  # Less smoothing when rotating (more responsive)
        else:
            alpha = self.smoothing_factor

        alpha = min(1.0, max(0.1, alpha))

        # Blend homographies
        smoothed = alpha * new_H + (1 - alpha) * self.current_homography

        # Normalize
        smoothed = smoothed / smoothed[2, 2]

        return smoothed

    def _interpolate_homography(self, timestamp: float) -> DynamicHomography:
        """Interpolate homography when feature matching fails."""
        if len(self.homography_buffer) < 2:
            return DynamicHomography(
                H=self.current_homography if self.current_homography is not None else self.base_homography,
                H_frame_to_ref=np.eye(3),
                timestamp=timestamp,
                confidence=0.3,
                is_interpolated=True
            )

        # Use last known homography with reduced confidence
        last_valid = self.homography_buffer[-1]

        # If we have velocity info, extrapolate
        if self.rotation_state.angular_velocity != 0:
            # Simple linear extrapolation based on angular velocity
            # This is a simplification - real implementation would decompose and recompose H
            extrapolated_H = last_valid.H.copy()
        else:
            extrapolated_H = last_valid.H.copy()

        return DynamicHomography(
            H=extrapolated_H,
            H_frame_to_ref=self.prev_H,
            timestamp=timestamp,
            confidence=max(0.1, last_valid.confidence - 0.2),
            is_interpolated=True
        )

    def update_reference(self, frame: np.ndarray, pitch_mask: Optional[np.ndarray] = None):
        """Update reference frame (call when camera is stable for extended period)."""
        if self.rotation_state.state != CameraState.STABLE:
            logger.warning("Attempted to update reference while camera is not stable")
            return

        if self.rotation_state.frames_in_state < self.stabilization_frames * 2:
            return

        # Update reference
        self.reference_frame = frame.copy()
        self.reference_pitch_mask = pitch_mask
        self.feature_matcher.set_reference_frame(frame, pitch_mask)

        # Update base homography to current
        if self.current_homography is not None:
            self.base_homography = self.current_homography.copy()

        logger.info("Reference frame updated")

    def get_current_homography(self) -> Optional[np.ndarray]:
        """Get current homography matrix."""
        return self.current_homography

    def get_state(self) -> RotationState:
        """Get current rotation state."""
        return self.rotation_state

    def is_rotating(self) -> bool:
        """Check if camera is currently rotating."""
        return self.rotation_state.state == CameraState.ROTATING


class AdaptiveHomographyManager:
    """Manages homography with automatic keypoint re-detection."""

    def __init__(
        self,
        redetection_interval: int = 100,  # Re-detect keypoints every N frames
        min_keypoints_for_update: int = 4,
        keypoint_match_threshold: float = 20.0,  # pixels
    ):
        self.redetection_interval = redetection_interval
        self.min_keypoints_for_update = min_keypoints_for_update
        self.keypoint_match_threshold = keypoint_match_threshold

        self.pitch_detector = PitchLineDetector()
        self.rotation_handler: Optional[RotationHandler] = None

        self.last_detection_frame = 0
        self.detected_keypoints: List = []

    def initialize(self, frame: np.ndarray, manual_homography: np.ndarray, pitch_mask: Optional[np.ndarray] = None):
        """Initialize with manual calibration."""
        self.rotation_handler = RotationHandler()
        self.rotation_handler.initialize(frame, manual_homography, pitch_mask)

        # Try to detect keypoints for future reference
        detection_result = self.pitch_detector.detect_keypoints(frame)
        if detection_result.is_valid:
            self.detected_keypoints = detection_result.keypoints
            logger.info(f"Detected {len(self.detected_keypoints)} initial keypoints")

    def update(self, frame: np.ndarray, frame_idx: int, timestamp: float,
               pitch_mask: Optional[np.ndarray] = None) -> DynamicHomography:
        """Update homography for new frame."""
        if self.rotation_handler is None:
            raise ValueError("Manager not initialized")

        # Periodic keypoint re-detection
        if frame_idx - self.last_detection_frame >= self.redetection_interval:
            self._redetect_keypoints(frame)
            self.last_detection_frame = frame_idx

        # Update rotation handler
        dyn_H = self.rotation_handler.update(frame, timestamp, pitch_mask)

        # If stable for long time, consider updating reference
        state = self.rotation_handler.get_state()
        if state.state == CameraState.STABLE and state.frames_in_state > self.redetection_interval:
            self.rotation_handler.update_reference(frame, pitch_mask)

        return dyn_H

    def _redetect_keypoints(self, frame: np.ndarray):
        """Re-detect pitch keypoints."""
        detection_result = self.pitch_detector.detect_keypoints(frame)

        if detection_result.is_valid and len(detection_result.keypoints) >= self.min_keypoints_for_update:
            self.detected_keypoints = detection_result.keypoints
            logger.debug(f"Re-detected {len(self.detected_keypoints)} keypoints")

    def get_homography(self) -> Optional[np.ndarray]:
        """Get current homography."""
        if self.rotation_handler is None:
            return None
        return self.rotation_handler.get_current_homography()

    def is_rotating(self) -> bool:
        """Check if camera is rotating."""
        if self.rotation_handler is None:
            return False
        return self.rotation_handler.is_rotating()

    def get_rotation_angle(self) -> float:
        """Get current rotation angle from reference."""
        if self.rotation_handler is None:
            return 0.0
        return self.rotation_handler.get_state().rotation_angle
