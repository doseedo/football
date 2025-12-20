"""Physical performance metrics calculation."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


@dataclass
class PhysicalMetrics:
    track_id: int
    total_distance: float
    max_speed: float
    avg_speed: float
    sprint_count: int
    sprint_distance: float
    high_intensity_distance: float
    max_acceleration: float
    max_deceleration: float


@dataclass
class FrameMetrics:
    position: Tuple[float, float]
    speed: float
    acceleration: float
    is_sprinting: bool
    is_high_intensity: bool


class PhysicalMetricsCalculator:
    SPRINT_THRESHOLD = 25.0
    HIGH_INTENSITY_THRESHOLD = 19.8
    MIN_SPRINT_DURATION = 1.0

    def __init__(self, fps: int = 25, smoothing_window: int = 5):
        self.fps = fps
        self.dt = 1.0 / fps
        self.smoothing_window = smoothing_window

    def calculate_frame_metrics(self, positions: List[Tuple[float, float]]) -> List[FrameMetrics]:
        if len(positions) < 2:
            return []

        positions = np.array(positions)
        velocities = np.diff(positions, axis=0) / self.dt
        speeds_mps = np.linalg.norm(velocities, axis=1)

        window = min(self.smoothing_window, len(speeds_mps))
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            speeds_mps = savgol_filter(speeds_mps, window, 2)

        speeds_kmh = speeds_mps * 3.6
        accelerations = np.concatenate([[0], np.diff(speeds_mps) / self.dt])
        if len(accelerations) >= self.smoothing_window:
            accelerations = uniform_filter1d(accelerations, self.smoothing_window)

        metrics = []
        for i in range(len(positions)):
            speed = speeds_kmh[min(i, len(speeds_kmh)-1)] if i > 0 else (speeds_kmh[0] if len(speeds_kmh) > 0 else 0)
            accel = accelerations[min(i, len(accelerations)-1)]
            metrics.append(FrameMetrics(
                position=tuple(positions[i]),
                speed=speed,
                acceleration=accel,
                is_sprinting=speed >= self.SPRINT_THRESHOLD,
                is_high_intensity=speed >= self.HIGH_INTENSITY_THRESHOLD
            ))
        return metrics

    def calculate_match_metrics(self, positions: List[Tuple[float, float]], track_id: int) -> PhysicalMetrics:
        if len(positions) < 2:
            return PhysicalMetrics(track_id, 0, 0, 0, 0, 0, 0, 0, 0)

        positions_arr = np.array(positions)
        distances = np.linalg.norm(np.diff(positions_arr, axis=0), axis=1)
        total_distance = np.sum(distances)

        frame_metrics = self.calculate_frame_metrics(positions)
        speeds = [m.speed for m in frame_metrics]
        max_speed = max(speeds) if speeds else 0
        avg_speed = np.mean(speeds) if speeds else 0

        sprint_count = self._count_sprints(frame_metrics)
        sprint_distance = sum(distances[i] for i, m in enumerate(frame_metrics[:-1]) if m.is_sprinting)
        high_intensity_distance = sum(distances[i] for i, m in enumerate(frame_metrics[:-1]) if m.is_high_intensity)

        accels = [m.acceleration for m in frame_metrics]
        max_acceleration = max(accels) if accels else 0
        max_deceleration = abs(min(accels)) if accels else 0

        return PhysicalMetrics(
            track_id=track_id,
            total_distance=total_distance,
            max_speed=max_speed,
            avg_speed=avg_speed,
            sprint_count=sprint_count,
            sprint_distance=sprint_distance,
            high_intensity_distance=high_intensity_distance,
            max_acceleration=max_acceleration,
            max_deceleration=max_deceleration
        )

    def _count_sprints(self, frame_metrics: List[FrameMetrics]) -> int:
        min_frames = int(self.MIN_SPRINT_DURATION * self.fps)
        count = 0
        in_sprint = False
        sprint_start = 0

        for i, m in enumerate(frame_metrics):
            if m.is_sprinting and not in_sprint:
                in_sprint = True
                sprint_start = i
            elif not m.is_sprinting and in_sprint:
                if i - sprint_start >= min_frames:
                    count += 1
                in_sprint = False
        return count


class KalmanSmoother:
    def __init__(self, dt: float = 0.04, process_noise: float = 0.1, measurement_noise: float = 1.0):
        self.dt = dt
        self.state = np.zeros(4)
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 100
        self.initialized = False

    def update(self, measurement) -> Tuple[float, float]:
        if not self.initialized:
            if measurement is not None:
                self.state[0], self.state[1] = measurement
                self.initialized = True
                return measurement
            return (0, 0)

        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        if measurement is not None:
            z = np.array(measurement)
            y = z - self.H @ self.state
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.state = self.state + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P

        return (self.state[0], self.state[1])

    def reset(self):
        self.state = np.zeros(4)
        self.P = np.eye(4) * 100
        self.initialized = False
