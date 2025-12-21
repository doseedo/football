"""Synthetic training data generation for 3D ball trajectory estimation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from .physics_model import Ball3DPosition, PhysicsConstraints


@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters."""
    focal_length: float  # pixels
    image_width: int  # pixels
    image_height: int  # pixels
    position: Tuple[float, float, float]  # x, y, z in meters
    rotation: np.ndarray  # 3x3 rotation matrix
    distortion: Optional[np.ndarray] = None


@dataclass
class SyntheticTrajectory:
    """Synthetic ball trajectory with 2D/3D correspondence."""
    positions_3d: np.ndarray  # (N, 3) - world coordinates
    positions_2d: np.ndarray  # (N, 2) - pixel coordinates
    velocities_3d: np.ndarray  # (N, 3) - 3D velocities
    timestamps: np.ndarray  # (N,) - time in seconds
    camera_params: CameraParameters
    trajectory_type: str  # 'pass', 'shot', 'cross', 'bounce'
    is_visible: np.ndarray  # (N,) - boolean visibility mask


class SyntheticDataGenerator:
    """Generate synthetic ball trajectories for training."""

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        fps: float = 25.0,
        noise_level: float = 0.02
    ):
        """
        Initialize synthetic data generator.

        Args:
            pitch_length: Length of football pitch in meters
            pitch_width: Width of football pitch in meters
            fps: Frame rate for trajectory generation
            noise_level: Standard deviation of Gaussian noise (relative to pitch size)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.fps = fps
        self.dt = 1.0 / fps
        self.noise_level = noise_level
        self.constraints = PhysicsConstraints()

    def generate_trajectory(
        self,
        trajectory_type: str = 'random',
        duration: Optional[float] = None,
        camera_params: Optional[CameraParameters] = None
    ) -> SyntheticTrajectory:
        """
        Generate a synthetic trajectory.

        Args:
            trajectory_type: Type of trajectory ('pass', 'shot', 'cross', 'bounce', 'random')
            duration: Duration in seconds (auto-computed if None)
            camera_params: Camera parameters (random if None)

        Returns:
            SyntheticTrajectory with 2D/3D correspondence
        """
        if trajectory_type == 'random':
            trajectory_type = np.random.choice(['pass', 'shot', 'cross', 'bounce', 'aerial'])

        # Generate 3D trajectory based on type
        if trajectory_type == 'pass':
            traj_3d = self._generate_ground_pass(duration)
        elif trajectory_type == 'shot':
            traj_3d = self._generate_shot(duration)
        elif trajectory_type == 'cross':
            traj_3d = self._generate_cross(duration)
        elif trajectory_type == 'bounce':
            traj_3d = self._generate_bounce(duration)
        elif trajectory_type == 'aerial':
            traj_3d = self._generate_aerial_ball(duration)
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

        # Generate random camera if not provided
        if camera_params is None:
            camera_params = self._generate_random_camera()

        # Project to 2D
        positions_2d, is_visible = self._project_to_2d(traj_3d['positions'], camera_params)

        # Add noise
        positions_2d_noisy = self._add_2d_noise(positions_2d, is_visible)

        return SyntheticTrajectory(
            positions_3d=traj_3d['positions'],
            positions_2d=positions_2d_noisy,
            velocities_3d=traj_3d['velocities'],
            timestamps=traj_3d['timestamps'],
            camera_params=camera_params,
            trajectory_type=trajectory_type,
            is_visible=is_visible
        )

    def generate_batch(
        self,
        batch_size: int,
        trajectory_types: Optional[List[str]] = None
    ) -> List[SyntheticTrajectory]:
        """
        Generate a batch of synthetic trajectories.

        Args:
            batch_size: Number of trajectories to generate
            trajectory_types: List of types to sample from (all if None)

        Returns:
            List of synthetic trajectories
        """
        if trajectory_types is None:
            trajectory_types = ['pass', 'shot', 'cross', 'bounce', 'aerial']

        trajectories = []
        for _ in range(batch_size):
            traj_type = np.random.choice(trajectory_types)
            trajectories.append(self.generate_trajectory(traj_type))

        return trajectories

    def _generate_ground_pass(self, duration: Optional[float] = None) -> Dict:
        """Generate a ground pass trajectory."""
        # Random start and end points on pitch
        start_x = np.random.uniform(10, self.pitch_length - 10)
        start_y = np.random.uniform(10, self.pitch_width - 10)
        end_x = np.random.uniform(10, self.pitch_length - 10)
        end_y = np.random.uniform(10, self.pitch_width - 10)

        # Distance and speed
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        speed = np.random.uniform(8, 20)  # m/s

        if duration is None:
            duration = distance / speed

        num_steps = int(duration * self.fps)
        t = np.linspace(0, duration, num_steps)

        # Linear interpolation with slight deceleration
        decay = np.exp(-0.3 * t)
        x = start_x + (end_x - start_x) * (1 - decay) / (1 - decay[-1])
        y = start_y + (end_y - start_y) * (1 - decay) / (1 - decay[-1])
        z = np.zeros(num_steps) + 0.11  # Ball radius (rolling on ground)

        positions = np.column_stack([x, y, z])

        # Calculate velocities
        velocities = np.diff(positions, axis=0, prepend=positions[0:1]) / self.dt

        return {
            'positions': positions,
            'velocities': velocities,
            'timestamps': t
        }

    def _generate_shot(self, duration: Optional[float] = None) -> Dict:
        """Generate a shot trajectory (powerful, low to high arc)."""
        # Start from random position
        start_x = np.random.uniform(20, 80)
        start_y = np.random.uniform(10, self.pitch_width - 10)

        # End near goal
        end_x = np.random.choice([5, self.pitch_length - 5])
        end_y = np.random.uniform(self.pitch_width/2 - 3.5, self.pitch_width/2 + 3.5)

        # Initial velocity (powerful shot)
        distance_2d = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        v0 = np.random.uniform(20, 30)  # m/s
        angle = np.random.uniform(5, 25) * np.pi / 180  # Launch angle

        v0_x = v0 * np.cos(angle) * (end_x - start_x) / distance_2d
        v0_y = v0 * np.cos(angle) * (end_y - start_y) / distance_2d
        v0_z = v0 * np.sin(angle)

        if duration is None:
            # Estimate time of flight
            duration = 2 * v0_z / self.constraints.gravity

        num_steps = int(duration * self.fps)
        t = np.linspace(0, duration, num_steps)

        # Parabolic trajectory
        x = start_x + v0_x * t
        y = start_y + v0_y * t
        z = 0.11 + v0_z * t - 0.5 * self.constraints.gravity * t**2

        # Ensure z >= 0
        z = np.maximum(z, 0.11)

        positions = np.column_stack([x, y, z])
        velocities = np.column_stack([
            np.full(num_steps, v0_x),
            np.full(num_steps, v0_y),
            v0_z - self.constraints.gravity * t
        ])

        return {
            'positions': positions,
            'velocities': velocities,
            'timestamps': t
        }

    def _generate_cross(self, duration: Optional[float] = None) -> Dict:
        """Generate a cross trajectory (from wing, high arc)."""
        # Start from wing
        start_x = np.random.choice([10, self.pitch_length - 10])
        start_y = np.random.choice([5, self.pitch_width - 5])

        # End in center/opposite side
        end_x = np.random.uniform(40, 65)
        end_y = self.pitch_width - start_y + np.random.uniform(-10, 10)

        distance_2d = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

        # High arc cross
        v0 = np.random.uniform(15, 25)
        angle = np.random.uniform(30, 50) * np.pi / 180

        v0_x = v0 * np.cos(angle) * (end_x - start_x) / distance_2d
        v0_y = v0 * np.cos(angle) * (end_y - start_y) / distance_2d
        v0_z = v0 * np.sin(angle)

        if duration is None:
            duration = 2 * v0_z / self.constraints.gravity

        num_steps = int(duration * self.fps)
        t = np.linspace(0, duration, num_steps)

        x = start_x + v0_x * t
        y = start_y + v0_y * t
        z = 0.11 + v0_z * t - 0.5 * self.constraints.gravity * t**2
        z = np.maximum(z, 0.11)

        positions = np.column_stack([x, y, z])
        velocities = np.column_stack([
            np.full(num_steps, v0_x),
            np.full(num_steps, v0_y),
            v0_z - self.constraints.gravity * t
        ])

        return {
            'positions': positions,
            'velocities': velocities,
            'timestamps': t
        }

    def _generate_bounce(self, duration: Optional[float] = None) -> Dict:
        """Generate a trajectory with bounces."""
        # Start position
        start_x = np.random.uniform(20, 80)
        start_y = np.random.uniform(10, self.pitch_width - 10)

        # Initial velocity
        v0 = np.random.uniform(10, 20)
        angle = np.random.uniform(20, 45) * np.pi / 180
        direction = np.random.uniform(0, 2*np.pi)

        v0_x = v0 * np.cos(angle) * np.cos(direction)
        v0_y = v0 * np.cos(angle) * np.sin(direction)
        v0_z = v0 * np.sin(angle)

        if duration is None:
            duration = 3.0  # Fixed duration for bouncing ball

        num_steps = int(duration * self.fps)

        positions = []
        velocities = []
        current_pos = np.array([start_x, start_y, 0.11])
        current_vel = np.array([v0_x, v0_y, v0_z])

        for i in range(num_steps):
            # Update position
            current_pos = current_pos + current_vel * self.dt

            # Apply gravity
            current_vel[2] -= self.constraints.gravity * self.dt

            # Check for bounce
            if current_pos[2] < 0.11:
                current_pos[2] = 0.11
                current_vel[2] = -current_vel[2] * self.constraints.bounce_coefficient
                current_vel[0] *= self.constraints.bounce_coefficient
                current_vel[1] *= self.constraints.bounce_coefficient

            # Keep on pitch
            current_pos[0] = np.clip(current_pos[0], 0, self.pitch_length)
            current_pos[1] = np.clip(current_pos[1], 0, self.pitch_width)

            positions.append(current_pos.copy())
            velocities.append(current_vel.copy())

        return {
            'positions': np.array(positions),
            'velocities': np.array(velocities),
            'timestamps': np.linspace(0, duration, num_steps)
        }

    def _generate_aerial_ball(self, duration: Optional[float] = None) -> Dict:
        """Generate a high aerial ball trajectory."""
        start_x = np.random.uniform(20, 80)
        start_y = np.random.uniform(10, self.pitch_width - 10)
        end_x = np.random.uniform(20, 80)
        end_y = np.random.uniform(10, self.pitch_width - 10)

        distance_2d = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

        # Very high ball
        v0 = np.random.uniform(20, 30)
        angle = np.random.uniform(45, 70) * np.pi / 180

        v0_x = v0 * np.cos(angle) * (end_x - start_x) / (distance_2d + 1e-6)
        v0_y = v0 * np.cos(angle) * (end_y - start_y) / (distance_2d + 1e-6)
        v0_z = v0 * np.sin(angle)

        if duration is None:
            duration = 2 * v0_z / self.constraints.gravity

        num_steps = int(duration * self.fps)
        t = np.linspace(0, duration, num_steps)

        x = start_x + v0_x * t
        y = start_y + v0_y * t
        z = 0.11 + v0_z * t - 0.5 * self.constraints.gravity * t**2
        z = np.maximum(z, 0.11)

        positions = np.column_stack([x, y, z])
        velocities = np.column_stack([
            np.full(num_steps, v0_x),
            np.full(num_steps, v0_y),
            v0_z - self.constraints.gravity * t
        ])

        return {
            'positions': positions,
            'velocities': velocities,
            'timestamps': t
        }

    def _generate_random_camera(self) -> CameraParameters:
        """Generate random realistic camera parameters."""
        # Camera positioned high and centered, with some variation
        cam_x = self.pitch_length / 2 + np.random.uniform(-20, 20)
        cam_y = self.pitch_width / 2 + np.random.uniform(-10, 10)
        cam_z = np.random.uniform(10, 25)  # 10-25m high

        # Look towards pitch center
        look_at = np.array([self.pitch_length/2, self.pitch_width/2, 0])
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # Compute rotation matrix (simple look-at)
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)

        # Arbitrary up vector
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        rotation = np.column_stack([right, up, -forward])

        # Realistic focal length for 4K broadcast camera
        image_width = 3840
        image_height = 2160
        focal_length = np.random.uniform(2000, 4000)  # pixels

        return CameraParameters(
            focal_length=focal_length,
            image_width=image_width,
            image_height=image_height,
            position=(cam_x, cam_y, cam_z),
            rotation=rotation
        )

    def _project_to_2d(
        self,
        positions_3d: np.ndarray,
        camera: CameraParameters
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D positions to 2D image coordinates.

        Args:
            positions_3d: (N, 3) array of 3D positions
            camera: Camera parameters

        Returns:
            positions_2d: (N, 2) array of 2D pixel coordinates
            is_visible: (N,) boolean array indicating visibility
        """
        N = len(positions_3d)
        positions_2d = np.zeros((N, 2))
        is_visible = np.zeros(N, dtype=bool)

        cam_pos = np.array(camera.position)
        R = camera.rotation

        for i in range(N):
            # Transform to camera coordinates
            point_world = positions_3d[i]
            point_cam = R.T @ (point_world - cam_pos)

            # Check if in front of camera
            if point_cam[2] < 0:
                continue

            # Project using pinhole camera model
            x_proj = camera.focal_length * point_cam[0] / point_cam[2]
            y_proj = camera.focal_length * point_cam[1] / point_cam[2]

            # Convert to pixel coordinates (origin at center)
            u = x_proj + camera.image_width / 2
            v = y_proj + camera.image_height / 2

            # Check if within image bounds
            if 0 <= u < camera.image_width and 0 <= v < camera.image_height:
                positions_2d[i] = [u, v]
                is_visible[i] = True

        return positions_2d, is_visible

    def _add_2d_noise(
        self,
        positions_2d: np.ndarray,
        is_visible: np.ndarray
    ) -> np.ndarray:
        """Add realistic noise to 2D detections."""
        noisy = positions_2d.copy()

        for i in range(len(noisy)):
            if is_visible[i]:
                # Pixel-level noise (detection uncertainty)
                noise = np.random.normal(0, 2.0, 2)  # ~2 pixel std dev
                noisy[i] += noise

        return noisy

    def save_dataset(
        self,
        filepath: str,
        num_samples: int = 10000,
        trajectory_types: Optional[List[str]] = None
    ):
        """
        Generate and save a dataset of synthetic trajectories.

        Args:
            filepath: Path to save dataset (will create .npz file)
            num_samples: Number of trajectories to generate
            trajectory_types: Types to include
        """
        print(f"Generating {num_samples} synthetic trajectories...")

        trajectories = self.generate_batch(num_samples, trajectory_types)

        # Prepare data for saving
        data = {
            'num_samples': num_samples,
            'pitch_length': self.pitch_length,
            'pitch_width': self.pitch_width,
            'fps': self.fps
        }

        # Store each trajectory
        for i, traj in enumerate(trajectories):
            prefix = f'traj_{i}'
            data[f'{prefix}_3d'] = traj.positions_3d
            data[f'{prefix}_2d'] = traj.positions_2d
            data[f'{prefix}_vel'] = traj.velocities_3d
            data[f'{prefix}_time'] = traj.timestamps
            data[f'{prefix}_visible'] = traj.is_visible
            data[f'{prefix}_type'] = traj.trajectory_type

        np.savez_compressed(filepath, **data)
        print(f"Dataset saved to {filepath}")
