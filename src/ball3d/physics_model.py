"""Ball physics constraints and motion modeling."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks


@dataclass
class PhysicsConstraints:
    """Physical constants for ball motion."""
    gravity: float = 9.81  # m/s²
    air_resistance: float = 0.47  # drag coefficient for sphere
    ball_mass: float = 0.43  # kg (FIFA regulation)
    ball_radius: float = 0.11  # m (FIFA regulation)
    air_density: float = 1.225  # kg/m³ at sea level
    max_velocity: float = 35.0  # m/s (~126 km/h)
    max_height: float = 30.0  # m (reasonable for football)
    min_height: float = 0.0  # m (ground level)
    bounce_coefficient: float = 0.65  # energy retention after bounce


@dataclass
class Ball3DPosition:
    """3D position with uncertainty and metadata."""
    x: float  # meters (along pitch length)
    y: float  # meters (along pitch width)
    z: float  # meters (height above ground)
    timestamp: float  # seconds
    confidence: float  # 0-1
    velocity: Optional[Tuple[float, float, float]] = None  # m/s
    is_bouncing: bool = False
    is_on_ground: bool = True


class BallPhysicsModel:
    """Physics-based ball motion modeling and constraint enforcement."""

    def __init__(self, constraints: Optional[PhysicsConstraints] = None, fps: float = 25.0):
        """
        Initialize physics model.

        Args:
            constraints: Physical constraints for ball motion
            fps: Frame rate for temporal calculations
        """
        self.constraints = constraints or PhysicsConstraints()
        self.fps = fps
        self.dt = 1.0 / fps

    def apply_physics_constraints(
        self,
        trajectory: List[Ball3DPosition],
        smooth: bool = True
    ) -> List[Ball3DPosition]:
        """
        Apply physics constraints to a 3D trajectory.

        Args:
            trajectory: List of 3D ball positions
            smooth: Whether to apply smoothing

        Returns:
            Constrained trajectory
        """
        if len(trajectory) < 2:
            return trajectory

        constrained = []

        for i, pos in enumerate(trajectory):
            # Enforce height bounds
            z = np.clip(pos.z, self.constraints.min_height, self.constraints.max_height)

            # Calculate velocity if not provided
            if pos.velocity is None and i > 0:
                prev = trajectory[i-1]
                velocity = (
                    (pos.x - prev.x) / self.dt,
                    (pos.y - prev.y) / self.dt,
                    (pos.z - prev.z) / self.dt
                )
            else:
                velocity = pos.velocity

            # Enforce max velocity constraint
            if velocity is not None:
                speed = np.sqrt(sum(v**2 for v in velocity))
                if speed > self.constraints.max_velocity:
                    scale = self.constraints.max_velocity / speed
                    velocity = tuple(v * scale for v in velocity)

            # Check if on ground
            is_on_ground = z < 0.1  # Within 10cm of ground

            constrained.append(Ball3DPosition(
                x=pos.x,
                y=pos.y,
                z=z,
                timestamp=pos.timestamp,
                confidence=pos.confidence,
                velocity=velocity,
                is_on_ground=is_on_ground,
                is_bouncing=False
            ))

        # Detect and mark bounces
        constrained = self._mark_bounces(constrained)

        # Apply smoothing if requested
        if smooth:
            constrained = self._smooth_trajectory(constrained)

        return constrained

    def estimate_height_from_motion(
        self,
        positions_2d: np.ndarray,
        velocities_2d: np.ndarray,
        camera_height: float = 15.0,
        apparent_sizes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estimate height from 2D motion patterns.

        Uses multiple cues:
        1. Trajectory arc fitting (parabolic motion)
        2. Apparent size changes
        3. Velocity magnitude

        Args:
            positions_2d: Array of shape (N, 2) with x, y positions
            velocities_2d: Array of shape (N, 2) with x, y velocities
            camera_height: Height of camera above pitch (meters)
            apparent_sizes: Optional array of apparent ball sizes in pixels

        Returns:
            heights: Array of shape (N,) with estimated heights
        """
        N = len(positions_2d)
        heights = np.zeros(N)

        if N < 3:
            return heights

        # Method 1: Parabolic arc fitting
        # Fit a parabola to the trajectory and extract vertical component
        for i in range(2, N):
            window_start = max(0, i - 10)
            window = positions_2d[window_start:i+1]

            if len(window) >= 3:
                # Fit 2D trajectory arc
                t = np.arange(len(window))

                # Estimate curvature from 2D motion
                dx = np.diff(window[:, 0])
                dy = np.diff(window[:, 1])
                curvature = np.mean(np.abs(np.diff(dy) / (dx[:-1] + 1e-6)))

                # Higher curvature suggests aerial ball
                heights[i] = min(curvature * 2.0, self.constraints.max_height)

        # Method 2: Apparent size (if available)
        if apparent_sizes is not None:
            max_size = np.max(apparent_sizes)
            for i in range(N):
                if apparent_sizes[i] > 0:
                    # Smaller apparent size = further from camera = potentially higher
                    size_ratio = apparent_sizes[i] / max_size
                    # Simple inverse relationship
                    height_from_size = (1 - size_ratio) * 3.0
                    # Blend with arc-based estimate
                    heights[i] = 0.7 * heights[i] + 0.3 * height_from_size

        # Method 3: Velocity magnitude
        # Sudden changes in 2D velocity can indicate vertical motion
        speeds_2d = np.linalg.norm(velocities_2d, axis=1)
        speed_changes = np.abs(np.diff(speeds_2d))
        speed_changes = np.concatenate([[0], speed_changes])

        # Large speed changes suggest ball leaving/returning to ground
        for i in range(1, N):
            if speed_changes[i] > 5.0:  # Threshold for significant change
                heights[i] = max(heights[i], 1.0)

        # Smooth heights
        if N > 5:
            from scipy.ndimage import gaussian_filter1d
            heights = gaussian_filter1d(heights, sigma=2.0)

        # Ensure non-negative
        heights = np.maximum(heights, 0.0)

        return heights

    def detect_bounce(self, trajectory: List[Ball3DPosition]) -> List[int]:
        """
        Detect bounce events in trajectory.

        A bounce is characterized by:
        1. Ball reaching low height (< threshold)
        2. Sudden reversal in vertical velocity
        3. Energy loss (reduced velocity after bounce)

        Args:
            trajectory: List of 3D positions

        Returns:
            List of indices where bounces occur
        """
        if len(trajectory) < 5:
            return []

        heights = np.array([pos.z for pos in trajectory])
        bounce_indices = []

        # Find local minima in height
        minima_indices, _ = find_peaks(-heights, height=-0.5, distance=5)

        for idx in minima_indices:
            if idx == 0 or idx >= len(trajectory) - 1:
                continue

            # Check if height is low enough
            if heights[idx] < 0.5:  # Within 50cm of ground
                # Check velocity reversal
                if trajectory[idx].velocity is not None:
                    if idx > 0 and trajectory[idx-1].velocity is not None:
                        vz_before = trajectory[idx-1].velocity[2]
                        vz_after = trajectory[idx].velocity[2]

                        # Velocity should reverse (negative to positive)
                        if vz_before < 0 and vz_after > -vz_before * 0.3:
                            bounce_indices.append(idx)

        return bounce_indices

    def _mark_bounces(self, trajectory: List[Ball3DPosition]) -> List[Ball3DPosition]:
        """Mark bounce events in trajectory."""
        bounce_indices = self.detect_bounce(trajectory)

        for idx in bounce_indices:
            trajectory[idx].is_bouncing = True

            # Apply bounce physics
            if trajectory[idx].velocity is not None:
                vx, vy, vz = trajectory[idx].velocity
                # Reduce velocity by bounce coefficient
                trajectory[idx].velocity = (
                    vx * self.constraints.bounce_coefficient,
                    vy * self.constraints.bounce_coefficient,
                    -vz * self.constraints.bounce_coefficient
                )

        return trajectory

    def _smooth_trajectory(self, trajectory: List[Ball3DPosition]) -> List[Ball3DPosition]:
        """Apply smoothing to trajectory."""
        if len(trajectory) < 5:
            return trajectory

        from scipy.ndimage import gaussian_filter1d

        # Extract coordinates
        x = np.array([pos.x for pos in trajectory])
        y = np.array([pos.y for pos in trajectory])
        z = np.array([pos.z for pos in trajectory])

        # Smooth each dimension
        x_smooth = gaussian_filter1d(x, sigma=1.5)
        y_smooth = gaussian_filter1d(y, sigma=1.5)
        z_smooth = gaussian_filter1d(z, sigma=1.5)

        # Update trajectory
        smoothed = []
        for i, pos in enumerate(trajectory):
            smoothed.append(Ball3DPosition(
                x=float(x_smooth[i]),
                y=float(y_smooth[i]),
                z=float(z_smooth[i]),
                timestamp=pos.timestamp,
                confidence=pos.confidence,
                velocity=pos.velocity,
                is_bouncing=pos.is_bouncing,
                is_on_ground=pos.is_on_ground
            ))

        return smoothed

    def predict_next_position(
        self,
        current: Ball3DPosition,
        steps: int = 1
    ) -> Ball3DPosition:
        """
        Predict future ball position using physics.

        Args:
            current: Current ball position with velocity
            steps: Number of time steps to predict forward

        Returns:
            Predicted position
        """
        if current.velocity is None:
            return current

        vx, vy, vz = current.velocity
        x, y, z = current.x, current.y, current.z

        dt_total = self.dt * steps

        # Simple ballistic trajectory (ignoring air resistance for now)
        # x(t) = x0 + vx*t
        # y(t) = y0 + vy*t
        # z(t) = z0 + vz*t - 0.5*g*t²

        x_pred = x + vx * dt_total
        y_pred = y + vy * dt_total
        z_pred = z + vz * dt_total - 0.5 * self.constraints.gravity * dt_total**2

        # Apply constraints
        z_pred = max(0.0, min(z_pred, self.constraints.max_height))

        # Update velocity
        vz_pred = vz - self.constraints.gravity * dt_total

        return Ball3DPosition(
            x=x_pred,
            y=y_pred,
            z=z_pred,
            timestamp=current.timestamp + dt_total,
            confidence=current.confidence * 0.9,  # Decrease confidence with prediction
            velocity=(vx, vy, vz_pred),
            is_on_ground=(z_pred < 0.1)
        )

    def fit_parabolic_trajectory(
        self,
        positions: List[Ball3DPosition]
    ) -> Tuple[np.ndarray, float]:
        """
        Fit a parabolic trajectory to 3D positions.

        Returns:
            parameters: Array of parameters [x0, y0, z0, vx0, vy0, vz0]
            error: RMS fitting error
        """
        if len(positions) < 3:
            raise ValueError("Need at least 3 positions for fitting")

        # Extract data
        times = np.array([pos.timestamp - positions[0].timestamp for pos in positions])
        xs = np.array([pos.x for pos in positions])
        ys = np.array([pos.y for pos in positions])
        zs = np.array([pos.z for pos in positions])

        # Initial guess
        x0_guess = [xs[0], ys[0], zs[0],
                   (xs[-1] - xs[0]) / (times[-1] or 1),
                   (ys[-1] - ys[0]) / (times[-1] or 1),
                   0.0]

        def trajectory_error(params):
            x0, y0, z0, vx0, vy0, vz0 = params

            x_pred = x0 + vx0 * times
            y_pred = y0 + vy0 * times
            z_pred = z0 + vz0 * times - 0.5 * self.constraints.gravity * times**2

            error = np.sum((xs - x_pred)**2 + (ys - y_pred)**2 + (zs - z_pred)**2)
            return error

        result = minimize(trajectory_error, x0_guess, method='Nelder-Mead')

        rms_error = np.sqrt(result.fun / len(positions))

        return result.x, rms_error
