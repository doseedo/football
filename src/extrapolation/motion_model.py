"""Physics-based motion model for player trajectory extrapolation.

This module provides Kalman filter-based motion prediction as a fallback
when transformer-based predictions have low confidence.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class MotionState:
    """Player motion state with position and velocity."""
    position: np.ndarray  # (x, y) in meters
    velocity: np.ndarray  # (vx, vy) in m/s
    acceleration: np.ndarray  # (ax, ay) in m/s^2
    timestamp: float
    confidence: float = 1.0


class KalmanMotionModel:
    """Kalman filter for physics-based player motion prediction.

    State vector: [x, y, vx, vy, ax, ay]
    Assumes constant acceleration model with noise.
    """

    def __init__(
        self,
        dt: float = 0.04,
        process_noise: float = 0.5,
        measurement_noise: float = 0.1,
        max_velocity: float = 12.0,  # ~43 km/h max sprinting speed
        max_acceleration: float = 5.0,  # m/s^2
        pitch_bounds: Tuple[float, float, float, float] = (0, 0, 105, 68)
    ):
        """Initialize Kalman motion model.

        Args:
            dt: Time step in seconds (default 0.04 for 25fps)
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
            max_velocity: Maximum allowed velocity magnitude (m/s)
            max_acceleration: Maximum allowed acceleration magnitude (m/s^2)
            pitch_bounds: (min_x, min_y, max_x, max_y) in meters
        """
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.pitch_bounds = pitch_bounds

        # State: [x, y, vx, vy, ax, ay]
        self.state = np.zeros(6)

        # State transition matrix (constant acceleration model)
        self.F = np.array([
            [1, 0, dt, 0,  0.5*dt**2, 0],
            [0, 1, 0,  dt, 0,         0.5*dt**2],
            [0, 0, 1,  0,  dt,        0],
            [0, 0, 0,  1,  0,         dt],
            [0, 0, 0,  0,  1,         0],
            [0, 0, 0,  0,  0,         1]
        ])

        # Observation matrix (we measure position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        # Process noise covariance
        self.Q = np.eye(6) * process_noise
        # Higher noise on acceleration
        self.Q[4:, 4:] *= 2.0

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise

        # State covariance
        self.P = np.eye(6) * 100.0

        self.initialized = False
        self.last_update_time = 0.0

    def initialize(self, position: np.ndarray, timestamp: float, velocity: Optional[np.ndarray] = None):
        """Initialize the filter with first observation.

        Args:
            position: Initial (x, y) position in meters
            timestamp: Current timestamp
            velocity: Optional initial velocity estimate
        """
        self.state[0:2] = position
        if velocity is not None:
            self.state[2:4] = velocity
        else:
            self.state[2:4] = 0
        self.state[4:6] = 0  # Zero initial acceleration

        self.last_update_time = timestamp
        self.initialized = True

    def predict(self, dt: Optional[float] = None) -> MotionState:
        """Predict next state without measurement update.

        Args:
            dt: Time step (if different from default)

        Returns:
            Predicted motion state
        """
        if not self.initialized:
            raise ValueError("Filter not initialized")

        if dt is not None and dt != self.dt:
            # Update transition matrix for different dt
            F_temp = self.F.copy()
            F_temp[0, 2] = F_temp[1, 3] = dt
            F_temp[0, 4] = F_temp[1, 5] = 0.5 * dt**2
            F_temp[2, 4] = F_temp[3, 5] = dt
            self.state = F_temp @ self.state
            self.P = F_temp @ self.P @ F_temp.T + self.Q
        else:
            # Standard prediction
            self.state = self.F @ self.state
            self.P = self.F @ self.P @ self.F.T + self.Q

        # Apply velocity and acceleration constraints
        self._apply_constraints()

        # Calculate prediction confidence based on covariance
        position_uncertainty = np.sqrt(self.P[0, 0] + self.P[1, 1])
        confidence = np.exp(-position_uncertainty / 2.0)

        return MotionState(
            position=self.state[0:2].copy(),
            velocity=self.state[2:4].copy(),
            acceleration=self.state[4:6].copy(),
            timestamp=self.last_update_time + (dt or self.dt),
            confidence=min(1.0, max(0.0, confidence))
        )

    def update(self, measurement: np.ndarray, timestamp: float) -> MotionState:
        """Update filter with new measurement.

        Args:
            measurement: Observed (x, y) position
            timestamp: Measurement timestamp

        Returns:
            Updated motion state
        """
        if not self.initialized:
            self.initialize(measurement, timestamp)
            return self.get_state(timestamp)

        # Predict to current time
        actual_dt = timestamp - self.last_update_time
        if actual_dt > 0:
            self.predict(actual_dt)

        # Update step
        z = measurement
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        self._apply_constraints()
        self.last_update_time = timestamp

        return self.get_state(timestamp)

    def extrapolate(self, n_steps: int) -> list[MotionState]:
        """Extrapolate trajectory for n future steps.

        Args:
            n_steps: Number of future time steps

        Returns:
            List of predicted motion states
        """
        if not self.initialized:
            raise ValueError("Filter not initialized")

        # Save current state
        state_backup = self.state.copy()
        P_backup = self.P.copy()

        predictions = []
        for i in range(n_steps):
            pred_state = self.predict()
            predictions.append(pred_state)
            self.last_update_time = pred_state.timestamp

        # Restore original state
        self.state = state_backup
        self.P = P_backup

        return predictions

    def get_state(self, timestamp: float) -> MotionState:
        """Get current state.

        Args:
            timestamp: Current timestamp

        Returns:
            Current motion state
        """
        position_uncertainty = np.sqrt(self.P[0, 0] + self.P[1, 1])
        confidence = np.exp(-position_uncertainty / 2.0)

        return MotionState(
            position=self.state[0:2].copy(),
            velocity=self.state[2:4].copy(),
            acceleration=self.state[4:6].copy(),
            timestamp=timestamp,
            confidence=min(1.0, max(0.0, confidence))
        )

    def _apply_constraints(self):
        """Apply physical constraints to state."""
        # Clip velocity
        velocity_mag = np.linalg.norm(self.state[2:4])
        if velocity_mag > self.max_velocity:
            self.state[2:4] *= self.max_velocity / velocity_mag

        # Clip acceleration
        accel_mag = np.linalg.norm(self.state[4:6])
        if accel_mag > self.max_acceleration:
            self.state[4:6] *= self.max_acceleration / accel_mag

        # Keep position within pitch bounds
        min_x, min_y, max_x, max_y = self.pitch_bounds
        self.state[0] = np.clip(self.state[0], min_x - 5, max_x + 5)  # Allow 5m margin
        self.state[1] = np.clip(self.state[1], min_y - 5, max_y + 5)

        # If at boundary, reduce velocity toward boundary
        if self.state[0] <= min_x and self.state[2] < 0:
            self.state[2] *= 0.5  # Bounce/dampen
        if self.state[0] >= max_x and self.state[2] > 0:
            self.state[2] *= 0.5
        if self.state[1] <= min_y and self.state[3] < 0:
            self.state[3] *= 0.5
        if self.state[1] >= max_y and self.state[3] > 0:
            self.state[3] *= 0.5

    def reset(self):
        """Reset filter to uninitialized state."""
        self.state = np.zeros(6)
        self.P = np.eye(6) * 100.0
        self.initialized = False
        self.last_update_time = 0.0


class MultiPlayerMotionModel:
    """Manages motion models for multiple players."""

    def __init__(self, **kalman_kwargs):
        """Initialize multi-player motion model.

        Args:
            **kalman_kwargs: Arguments passed to KalmanMotionModel
        """
        self.kalman_kwargs = kalman_kwargs
        self.models: Dict[int, KalmanMotionModel] = {}

    def update(self, player_id: int, position: np.ndarray, timestamp: float) -> MotionState:
        """Update motion model for a player.

        Args:
            player_id: Unique player identifier
            position: Observed (x, y) position
            timestamp: Measurement timestamp

        Returns:
            Updated motion state
        """
        if player_id not in self.models:
            self.models[player_id] = KalmanMotionModel(**self.kalman_kwargs)

        return self.models[player_id].update(position, timestamp)

    def predict(self, player_id: int) -> Optional[MotionState]:
        """Predict next state for a player.

        Args:
            player_id: Unique player identifier

        Returns:
            Predicted motion state or None if not initialized
        """
        if player_id not in self.models or not self.models[player_id].initialized:
            return None

        return self.models[player_id].predict()

    def extrapolate(self, player_id: int, n_steps: int) -> Optional[list[MotionState]]:
        """Extrapolate trajectory for a player.

        Args:
            player_id: Unique player identifier
            n_steps: Number of future steps

        Returns:
            List of predicted states or None if not initialized
        """
        if player_id not in self.models or not self.models[player_id].initialized:
            return None

        return self.models[player_id].extrapolate(n_steps)

    def get_all_states(self, timestamp: float) -> Dict[int, MotionState]:
        """Get current states for all tracked players.

        Args:
            timestamp: Current timestamp

        Returns:
            Dictionary mapping player_id to motion state
        """
        states = {}
        for player_id, model in self.models.items():
            if model.initialized:
                states[player_id] = model.get_state(timestamp)
        return states

    def remove_player(self, player_id: int):
        """Remove player from tracking.

        Args:
            player_id: Player to remove
        """
        if player_id in self.models:
            del self.models[player_id]

    def reset(self):
        """Reset all motion models."""
        self.models.clear()
