"""Bayesian temporal filtering for homography stabilization.

This module implements BHITK (Bayesian Homography with Implicit Temporal Keypoints),
a two-stage Kalman filter approach for temporally consistent homography estimation:
1. Stage 1: Filter keypoint positions
2. Stage 2: Filter homography parameters

This provides smooth, temporally coherent calibration for video sequences.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from .auto_calibration import AutoCalibrationResult


@dataclass
class KeypointState:
    """State of a tracked keypoint in the Kalman filter."""

    name: str
    position: np.ndarray  # [x, y] pixel coordinates
    velocity: np.ndarray  # [vx, vy] velocity
    covariance: np.ndarray  # 4x4 covariance matrix
    confidence: float = 0.0
    missing_count: int = 0
    last_observed: Optional[float] = None

    @classmethod
    def create(cls, name: str, position: Tuple[float, float], confidence: float = 1.0) -> 'KeypointState':
        """Create new keypoint state."""
        return cls(
            name=name,
            position=np.array(position, dtype=np.float64),
            velocity=np.zeros(2, dtype=np.float64),
            covariance=np.eye(4, dtype=np.float64) * 100.0,  # Initial high uncertainty
            confidence=confidence
        )


@dataclass
class HomographyState:
    """State of homography parameters in the Kalman filter."""

    H: np.ndarray  # 3x3 homography matrix
    H_params: np.ndarray  # 8-parameter vector (excluding h33=1)
    velocity: np.ndarray  # Parameter velocities
    covariance: np.ndarray  # Covariance matrix
    quality_score: float = 0.0
    update_count: int = 0

    @classmethod
    def create(cls, H: np.ndarray) -> 'HomographyState':
        """Create new homography state."""
        # Normalize and extract parameters
        H_normalized = H / H[2, 2]
        H_params = H_normalized.ravel()[:8]  # Exclude h33

        return cls(
            H=H_normalized.copy(),
            H_params=H_params,
            velocity=np.zeros(8, dtype=np.float64),
            covariance=np.eye(16, dtype=np.float64) * 10.0  # 8 params + 8 velocities
        )


class KeypointKalmanFilter:
    """Kalman filter for individual keypoint tracking.

    State vector: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 5.0,
        velocity_decay: float = 0.95
    ):
        """Initialize keypoint Kalman filter.

        Args:
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
            velocity_decay: Velocity decay factor (0-1)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.velocity_decay = velocity_decay

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, velocity_decay, 0],  # vx = vx * decay
            [0, 0, 0, velocity_decay]  # vy = vy * decay
        ], dtype=np.float64)

        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float64) * process_noise ** 2
        self.Q[2:, 2:] *= 0.1  # Lower noise for velocities

        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float64) * measurement_noise ** 2

    def predict(self, state: KeypointState) -> KeypointState:
        """Predict next state.

        Args:
            state: Current keypoint state

        Returns:
            Predicted state
        """
        # State vector
        x = np.concatenate([state.position, state.velocity])

        # Predict
        x_pred = self.F @ x
        P_pred = self.F @ state.covariance @ self.F.T + self.Q

        # Update state
        state.position = x_pred[:2]
        state.velocity = x_pred[2:]
        state.covariance = P_pred

        return state

    def update(
        self,
        state: KeypointState,
        measurement: np.ndarray,
        confidence: float = 1.0
    ) -> KeypointState:
        """Update state with measurement.

        Args:
            state: Current state
            measurement: Observed position [x, y]
            confidence: Measurement confidence (0-1)

        Returns:
            Updated state
        """
        # State vector
        x = np.concatenate([state.position, state.velocity])

        # Adjust measurement noise based on confidence
        R_adjusted = self.R / max(confidence, 0.1)

        # Innovation
        y = measurement - self.H @ x

        # Innovation covariance
        S = self.H @ state.covariance @ self.H.T + R_adjusted

        # Kalman gain
        K = state.covariance @ self.H.T @ np.linalg.inv(S)

        # Update
        x_updated = x + K @ y
        P_updated = (np.eye(4) - K @ self.H) @ state.covariance

        # Update state
        state.position = x_updated[:2]
        state.velocity = x_updated[2:]
        state.covariance = P_updated
        state.confidence = confidence
        state.missing_count = 0

        return state


class HomographyKalmanFilter:
    """Kalman filter for homography parameters.

    State vector: [h11, h12, h13, h21, h22, h23, h31, h32, vh11, ..., vh32]
    (16 dimensions: 8 parameters + 8 velocities)
    """

    def __init__(
        self,
        process_noise: float = 0.001,
        measurement_noise: float = 0.01,
        velocity_decay: float = 0.9
    ):
        """Initialize homography Kalman filter.

        Args:
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
            velocity_decay: Velocity decay factor (0-1)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.velocity_decay = velocity_decay

        # State transition matrix
        self.F = np.eye(16, dtype=np.float64)
        self.F[:8, 8:] = np.eye(8)  # param = param + velocity
        self.F[8:, 8:] *= velocity_decay  # velocity decay

        # Measurement matrix (observe parameters only)
        self.H = np.zeros((8, 16), dtype=np.float64)
        self.H[:8, :8] = np.eye(8)

        # Process noise covariance
        self.Q = np.eye(16, dtype=np.float64) * process_noise ** 2
        self.Q[8:, 8:] *= 0.1  # Lower noise for velocities

        # Measurement noise covariance
        self.R = np.eye(8, dtype=np.float64) * measurement_noise ** 2

    def predict(self, state: HomographyState) -> HomographyState:
        """Predict next state.

        Args:
            state: Current homography state

        Returns:
            Predicted state
        """
        # State vector
        x = np.concatenate([state.H_params, state.velocity])

        # Predict
        x_pred = self.F @ x
        P_pred = self.F @ state.covariance @ self.F.T + self.Q

        # Update state
        state.H_params = x_pred[:8]
        state.velocity = x_pred[8:]
        state.covariance = P_pred

        # Reconstruct homography matrix
        state.H = self._params_to_matrix(state.H_params)

        return state

    def update(
        self,
        state: HomographyState,
        measurement: np.ndarray,
        quality_score: float = 1.0
    ) -> HomographyState:
        """Update state with measurement.

        Args:
            state: Current state
            measurement: Observed homography parameters (8D)
            quality_score: Measurement quality (0-1)

        Returns:
            Updated state
        """
        # State vector
        x = np.concatenate([state.H_params, state.velocity])

        # Adjust measurement noise based on quality
        R_adjusted = self.R / max(quality_score, 0.1)

        # Innovation
        y = measurement - self.H @ x

        # Innovation covariance
        S = self.H @ state.covariance @ self.H.T + R_adjusted

        # Kalman gain
        try:
            K = state.covariance @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance, skipping update")
            return state

        # Update
        x_updated = x + K @ y
        P_updated = (np.eye(16) - K @ self.H) @ state.covariance

        # Update state
        state.H_params = x_updated[:8]
        state.velocity = x_updated[8:]
        state.covariance = P_updated
        state.quality_score = quality_score
        state.update_count += 1

        # Reconstruct homography matrix
        state.H = self._params_to_matrix(state.H_params)

        return state

    def _params_to_matrix(self, params: np.ndarray) -> np.ndarray:
        """Convert 8-parameter vector to 3x3 homography matrix.

        Args:
            params: 8-parameter vector

        Returns:
            3x3 homography matrix
        """
        H = np.zeros((3, 3), dtype=np.float64)
        H[0, 0] = params[0]
        H[0, 1] = params[1]
        H[0, 2] = params[2]
        H[1, 0] = params[3]
        H[1, 1] = params[4]
        H[1, 2] = params[5]
        H[2, 0] = params[6]
        H[2, 1] = params[7]
        H[2, 2] = 1.0
        return H


class BayesianHomographyFilter:
    """Two-stage Bayesian filter for homography estimation (BHITK method).

    Implements:
    1. Stage 1: Kalman filter for each keypoint position
    2. Stage 2: Kalman filter for homography parameters
    """

    def __init__(
        self,
        keypoint_process_noise: float = 1.0,
        keypoint_measurement_noise: float = 5.0,
        homography_process_noise: float = 0.001,
        homography_measurement_noise: float = 0.01,
        max_missing_frames: int = 30,
        min_keypoints_for_update: int = 4
    ):
        """Initialize Bayesian homography filter.

        Args:
            keypoint_process_noise: Process noise for keypoint filter
            keypoint_measurement_noise: Measurement noise for keypoint filter
            homography_process_noise: Process noise for homography filter
            homography_measurement_noise: Measurement noise for homography filter
            max_missing_frames: Maximum frames a keypoint can be missing
            min_keypoints_for_update: Minimum keypoints needed for homography update
        """
        self.max_missing_frames = max_missing_frames
        self.min_keypoints_for_update = min_keypoints_for_update

        # Stage 1: Keypoint filters
        self.keypoint_filter = KeypointKalmanFilter(
            process_noise=keypoint_process_noise,
            measurement_noise=keypoint_measurement_noise
        )
        self.keypoint_states: Dict[str, KeypointState] = {}

        # Stage 2: Homography filter
        self.homography_filter = HomographyKalmanFilter(
            process_noise=homography_process_noise,
            measurement_noise=homography_measurement_noise
        )
        self.homography_state: Optional[HomographyState] = None

        # Statistics
        self.frame_count = 0
        self.total_updates = 0

        logger.info("Initialized Bayesian homography filter (BHITK method)")

    def process_frame(
        self,
        calibration_result: AutoCalibrationResult,
        timestamp: Optional[float] = None
    ) -> np.ndarray:
        """Process a frame and return filtered homography.

        Args:
            calibration_result: Auto-calibration result for this frame
            timestamp: Optional frame timestamp

        Returns:
            Filtered homography matrix [3, 3]
        """
        self.frame_count += 1

        # Stage 1: Update keypoint states
        detected_keypoints = {
            kp[0].name: (kp[0].pixel_coords, kp[0].confidence)
            for kp in calibration_result.matched_keypoints
        }

        self._update_keypoint_states(detected_keypoints, timestamp)

        # Get filtered keypoint positions
        filtered_keypoints = self._get_filtered_keypoints()

        # Stage 2: Update homography state
        if len(filtered_keypoints) >= self.min_keypoints_for_update:
            if calibration_result.is_valid:
                self._update_homography_state(
                    calibration_result.homography,
                    calibration_result.quality.quality_score
                )
            else:
                # Predict only if no valid measurement
                if self.homography_state is not None:
                    self.homography_state = self.homography_filter.predict(self.homography_state)
        else:
            logger.debug(f"Insufficient filtered keypoints: {len(filtered_keypoints)}")
            if self.homography_state is not None:
                self.homography_state = self.homography_filter.predict(self.homography_state)

        # Return current best estimate
        if self.homography_state is not None:
            return self.homography_state.H
        elif calibration_result.is_valid:
            # Fallback to unfiltered
            return calibration_result.homography
        else:
            # Return identity if nothing available
            return np.eye(3)

    def _update_keypoint_states(
        self,
        detected_keypoints: Dict[str, Tuple[Tuple[float, float], float]],
        timestamp: Optional[float]
    ):
        """Update Stage 1: Keypoint Kalman filters.

        Args:
            detected_keypoints: Dict of {name: ((x, y), confidence)}
            timestamp: Optional timestamp
        """
        # Predict all existing keypoints
        for name, state in self.keypoint_states.items():
            self.keypoint_states[name] = self.keypoint_filter.predict(state)
            state.missing_count += 1

        # Update with detections
        for name, (position, confidence) in detected_keypoints.items():
            if name not in self.keypoint_states:
                # Initialize new keypoint
                self.keypoint_states[name] = KeypointState.create(name, position, confidence)
            else:
                # Update existing keypoint
                measurement = np.array(position, dtype=np.float64)
                self.keypoint_states[name] = self.keypoint_filter.update(
                    self.keypoint_states[name],
                    measurement,
                    confidence
                )
                self.keypoint_states[name].last_observed = timestamp

        # Remove keypoints that have been missing too long
        to_remove = [
            name for name, state in self.keypoint_states.items()
            if state.missing_count > self.max_missing_frames
        ]
        for name in to_remove:
            del self.keypoint_states[name]
            logger.debug(f"Removed keypoint {name} (missing for {self.max_missing_frames} frames)")

    def _get_filtered_keypoints(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """Get current filtered keypoint positions.

        Returns:
            Dict of {name: (position, confidence)}
        """
        return {
            name: (state.position, state.confidence)
            for name, state in self.keypoint_states.items()
            if state.missing_count < 5  # Only use recently seen keypoints
        }

    def _update_homography_state(self, H_measured: np.ndarray, quality_score: float):
        """Update Stage 2: Homography Kalman filter.

        Args:
            H_measured: Measured homography matrix
            quality_score: Quality of measurement
        """
        # Normalize measurement
        H_normalized = H_measured / H_measured[2, 2]
        H_params_measured = H_normalized.ravel()[:8]

        if self.homography_state is None:
            # Initialize
            self.homography_state = HomographyState.create(H_normalized)
            logger.debug("Initialized homography state")
        else:
            # Predict
            self.homography_state = self.homography_filter.predict(self.homography_state)

            # Update
            self.homography_state = self.homography_filter.update(
                self.homography_state,
                H_params_measured,
                quality_score
            )
            self.total_updates += 1

    def get_current_homography(self) -> Optional[np.ndarray]:
        """Get current filtered homography estimate.

        Returns:
            Homography matrix or None if not initialized
        """
        if self.homography_state is not None:
            return self.homography_state.H
        return None

    def get_uncertainty(self) -> Optional[float]:
        """Get current homography uncertainty (trace of covariance).

        Returns:
            Uncertainty measure or None
        """
        if self.homography_state is not None:
            return float(np.trace(self.homography_state.covariance))
        return None

    def reset(self):
        """Reset filter state."""
        self.keypoint_states.clear()
        self.homography_state = None
        self.frame_count = 0
        self.total_updates = 0
        logger.info("Reset Bayesian filter")

    def get_statistics(self) -> Dict:
        """Get filter statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'frame_count': self.frame_count,
            'total_updates': self.total_updates,
            'num_tracked_keypoints': len(self.keypoint_states),
            'homography_initialized': self.homography_state is not None,
            'uncertainty': self.get_uncertainty(),
        }


def create_bayesian_filter(
    strict: bool = False,
    max_missing_frames: int = 30
) -> BayesianHomographyFilter:
    """Create a Bayesian homography filter with default settings.

    Args:
        strict: Use strict (lower noise) settings
        max_missing_frames: Maximum frames to track missing keypoints

    Returns:
        Configured BayesianHomographyFilter
    """
    if strict:
        # Lower noise for more accurate but less smooth tracking
        return BayesianHomographyFilter(
            keypoint_process_noise=0.5,
            keypoint_measurement_noise=2.0,
            homography_process_noise=0.0005,
            homography_measurement_noise=0.005,
            max_missing_frames=max_missing_frames
        )
    else:
        # Default balanced settings
        return BayesianHomographyFilter(
            keypoint_process_noise=1.0,
            keypoint_measurement_noise=5.0,
            homography_process_noise=0.001,
            homography_measurement_noise=0.01,
            max_missing_frames=max_missing_frames
        )
