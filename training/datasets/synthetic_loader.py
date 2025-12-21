"""
Synthetic Data Generation

Generates synthetic training data for:
- 3D ball trajectories (with physics simulation)
- Player movement patterns
- Homography variations (camera poses)
- Tactical scenarios

Useful for:
- Augmenting real datasets
- Pre-training models
- Testing edge cases
"""

import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger


class SyntheticBall3DDataset(Dataset):
    """
    Generate synthetic 3D ball trajectories with physics.

    Trajectory types:
    - Parabolic: Long passes, shots (ballistic motion)
    - Ground: Rolling balls with friction
    - Bounce: Trajectories with ground bounces
    """

    def __init__(
        self,
        num_samples: int = 10000,
        trajectory_length: int = 25,
        fps: int = 25,
        pitch_dimensions: Tuple[float, float] = (105.0, 68.0),
        camera_height_range: Tuple[float, float] = (8.0, 15.0),
        camera_angle_range: Tuple[float, float] = (10, 30),  # degrees
        add_noise: bool = True,
        noise_std: float = 2.0,  # pixels
    ):
        self.num_samples = num_samples
        self.trajectory_length = trajectory_length
        self.fps = fps
        self.dt = 1.0 / fps
        self.pitch_dimensions = pitch_dimensions
        self.camera_height_range = camera_height_range
        self.camera_angle_range = camera_angle_range
        self.add_noise = add_noise
        self.noise_std = noise_std

        # Physics constants
        self.g = 9.81  # m/s^2
        self.air_resistance = 0.01
        self.bounce_coeff = 0.7
        self.friction = 0.3

        logger.info(f"Created synthetic ball dataset with {num_samples} samples")

    def __len__(self) -> int:
        return self.num_samples

    def _generate_trajectory_3d(self, trajectory_type: str) -> np.ndarray:
        """Generate 3D ball trajectory."""

        if trajectory_type == "parabolic":
            return self._generate_parabolic()
        elif trajectory_type == "ground":
            return self._generate_ground()
        elif trajectory_type == "bounce":
            return self._generate_bounce()
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    def _generate_parabolic(self) -> np.ndarray:
        """Generate parabolic trajectory (shot, long pass)."""
        # Random start and end positions
        x0 = random.uniform(10, 95)
        y0 = random.uniform(10, 58)
        z0 = random.uniform(0.5, 1.0)  # Initial height

        x1 = random.uniform(10, 95)
        y1 = random.uniform(10, 58)
        z1 = random.uniform(0.0, 0.5)  # Landing height

        # Time of flight
        flight_time = random.uniform(1.0, 3.0)
        num_steps = int(flight_time * self.fps)

        # Initial velocities
        vx = (x1 - x0) / flight_time
        vy = (y1 - y0) / flight_time
        vz = ((z1 - z0) + 0.5 * self.g * flight_time**2) / flight_time

        # Simulate trajectory
        positions = []
        x, y, z = x0, y0, z0
        vx_cur, vy_cur, vz_cur = vx, vy, vz

        for _ in range(num_steps):
            positions.append([x, y, z])

            # Update velocities (with air resistance)
            vz_cur -= self.g * self.dt
            vx_cur *= (1 - self.air_resistance * self.dt)
            vy_cur *= (1 - self.air_resistance * self.dt)

            # Update positions
            x += vx_cur * self.dt
            y += vy_cur * self.dt
            z += vz_cur * self.dt

            # Stop if hits ground
            if z < 0:
                z = 0
                break

        return np.array(positions)

    def _generate_ground(self) -> np.ndarray:
        """Generate ground pass trajectory."""
        x0 = random.uniform(10, 95)
        y0 = random.uniform(10, 58)

        x1 = random.uniform(10, 95)
        y1 = random.uniform(10, 58)

        # Initial velocity
        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        initial_speed = random.uniform(10, 25)  # m/s

        # Direction
        dx = (x1 - x0) / distance
        dy = (y1 - y0) / distance

        vx = dx * initial_speed
        vy = dy * initial_speed

        # Simulate with friction
        positions = []
        x, y = x0, y0
        vx_cur, vy_cur = vx, vy

        while np.sqrt(vx_cur**2 + vy_cur**2) > 0.1:  # Until ball stops
            positions.append([x, y, 0.0])

            # Deceleration due to friction
            speed = np.sqrt(vx_cur**2 + vy_cur**2)
            if speed > 0:
                ax = -(vx_cur / speed) * self.friction * self.g
                ay = -(vy_cur / speed) * self.friction * self.g
            else:
                break

            vx_cur += ax * self.dt
            vy_cur += ay * self.dt

            x += vx_cur * self.dt
            y += vy_cur * self.dt

            # Keep on pitch
            x = np.clip(x, 0, self.pitch_dimensions[0])
            y = np.clip(y, 0, self.pitch_dimensions[1])

            if len(positions) > 200:  # Safety limit
                break

        return np.array(positions)

    def _generate_bounce(self) -> np.ndarray:
        """Generate bouncing trajectory."""
        positions = []

        # Initial conditions
        x = random.uniform(20, 85)
        y = random.uniform(20, 48)
        z = random.uniform(1.0, 2.0)

        vx = random.uniform(-10, 10)
        vy = random.uniform(-10, 10)
        vz = random.uniform(5, 10)

        for _ in range(self.trajectory_length * 2):
            positions.append([x, y, max(0, z)])

            # Update velocities
            vz -= self.g * self.dt

            # Update positions
            x += vx * self.dt
            y += vy * self.dt
            z += vz * self.dt

            # Bounce
            if z < 0:
                z = 0
                vz = -vz * self.bounce_coeff
                vx *= (1 - 0.1)  # Energy loss
                vy *= (1 - 0.1)

                if abs(vz) < 0.5:  # Stop bouncing
                    break

            # Keep on pitch
            x = np.clip(x, 0, self.pitch_dimensions[0])
            y = np.clip(y, 0, self.pitch_dimensions[1])

        return np.array(positions)

    def _project_to_2d(
        self,
        trajectory_3d: np.ndarray,
        camera_height: float,
        camera_angle: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D trajectory to 2D image coordinates."""
        # Simple perspective projection
        # Camera at center of pitch, elevated

        camera_x = self.pitch_dimensions[0] / 2
        camera_y = -5  # Behind the pitch
        camera_z = camera_height

        # Camera angle (pitch down)
        angle_rad = np.radians(camera_angle)

        # Intrinsic matrix (simplified)
        focal_length = 1000
        cx, cy = 960, 540  # Image center for 1920x1080

        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])

        # Rotation matrix (pitch down)
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # Translation
        t = np.array([[-camera_x], [-camera_y], [-camera_z]])

        # Project points
        trajectory_2d = []
        for point_3d in trajectory_3d:
            # World to camera coordinates
            point_world = np.array([[point_3d[0]], [point_3d[1]], [point_3d[2]]])
            point_cam = R @ (point_world + t)

            # Project to image
            if point_cam[2, 0] > 0:  # In front of camera
                point_img = K @ point_cam
                x = point_img[0, 0] / point_img[2, 0]
                y = point_img[1, 0] / point_img[2, 0]

                # Add noise
                if self.add_noise:
                    x += np.random.normal(0, self.noise_std)
                    y += np.random.normal(0, self.noise_std)

                trajectory_2d.append([x, y])
            else:
                trajectory_2d.append([np.nan, np.nan])

        return np.array(trajectory_2d), trajectory_3d

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random trajectory type
        traj_type = random.choice(["parabolic", "ground", "bounce"])

        # Generate 3D trajectory
        trajectory_3d = self._generate_trajectory_3d(traj_type)

        # Random camera parameters
        camera_height = random.uniform(*self.camera_height_range)
        camera_angle = random.uniform(*self.camera_angle_range)

        # Project to 2D
        trajectory_2d, trajectory_3d = self._project_to_2d(
            trajectory_3d, camera_height, camera_angle
        )

        # Truncate/pad to fixed length
        if len(trajectory_2d) > self.trajectory_length:
            trajectory_2d = trajectory_2d[:self.trajectory_length]
            trajectory_3d = trajectory_3d[:self.trajectory_length]
        else:
            # Pad with last position
            pad_len = self.trajectory_length - len(trajectory_2d)
            if len(trajectory_2d) > 0:
                last_2d = trajectory_2d[-1]
                last_3d = trajectory_3d[-1]
            else:
                last_2d = [0, 0]
                last_3d = [0, 0, 0]

            trajectory_2d = np.vstack([
                trajectory_2d,
                np.tile(last_2d, (pad_len, 1))
            ])
            trajectory_3d = np.vstack([
                trajectory_3d,
                np.tile(last_3d, (pad_len, 1))
            ])

        return {
            'trajectory_2d': torch.from_numpy(trajectory_2d).float(),
            'trajectory_3d': torch.from_numpy(trajectory_3d).float(),
            'trajectory_type': traj_type,
            'camera_height': camera_height,
            'camera_angle': camera_angle,
        }


class SyntheticTrajectoryDataset(Dataset):
    """Generate synthetic player trajectories for Baller2Vec training."""

    def __init__(
        self,
        num_samples: int = 50000,
        sequence_length: int = 100,
        pitch_dimensions: Tuple[float, float] = (105.0, 68.0),
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.pitch_dimensions = pitch_dimensions

        logger.info(f"Created synthetic trajectory dataset with {num_samples} samples")

    def __len__(self) -> int:
        return self.num_samples

    def _generate_trajectory(self) -> np.ndarray:
        """Generate realistic player trajectory."""
        # Choose trajectory pattern
        pattern = random.choice([
            "sprint", "jog", "walk", "curve", "zigzag"
        ])

        if pattern == "sprint":
            # Straight line sprint
            x0, y0 = random.uniform(0, 105), random.uniform(0, 68)
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(6, 10)  # m/s

            positions = []
            for t in range(self.sequence_length):
                x = x0 + speed * np.cos(angle) * t * 0.04
                y = y0 + speed * np.sin(angle) * t * 0.04
                positions.append([
                    np.clip(x, 0, 105),
                    np.clip(y, 0, 68)
                ])

        elif pattern == "curve":
            # Curved run
            x0, y0 = random.uniform(0, 105), random.uniform(0, 68)
            radius = random.uniform(5, 20)
            angular_speed = random.uniform(0.05, 0.2)

            positions = []
            for t in range(self.sequence_length):
                angle = angular_speed * t
                x = x0 + radius * np.cos(angle)
                y = y0 + radius * np.sin(angle)
                positions.append([
                    np.clip(x, 0, 105),
                    np.clip(y, 0, 68)
                ])

        elif pattern == "zigzag":
            # Zigzag pattern
            x0, y0 = random.uniform(0, 105), random.uniform(0, 68)
            direction = random.uniform(0, 2 * np.pi)
            amplitude = random.uniform(2, 5)
            frequency = random.uniform(0.1, 0.3)

            positions = []
            for t in range(self.sequence_length):
                base_x = x0 + 3 * np.cos(direction) * t * 0.04
                base_y = y0 + 3 * np.sin(direction) * t * 0.04

                offset_x = amplitude * np.sin(frequency * t)
                offset_y = amplitude * np.cos(frequency * t)

                positions.append([
                    np.clip(base_x + offset_x, 0, 105),
                    np.clip(base_y + offset_y, 0, 68)
                ])

        else:  # jog or walk
            # Random walk
            x, y = random.uniform(0, 105), random.uniform(0, 68)
            speed = random.uniform(1, 4)

            positions = []
            angle = random.uniform(0, 2 * np.pi)

            for t in range(self.sequence_length):
                # Occasional direction change
                if random.random() < 0.05:
                    angle += random.uniform(-np.pi/4, np.pi/4)

                x += speed * np.cos(angle) * 0.04
                y += speed * np.sin(angle) * 0.04

                positions.append([
                    np.clip(x, 0, 105),
                    np.clip(y, 0, 68)
                ])

        return np.array(positions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trajectory = self._generate_trajectory()

        # Compute velocities
        velocities = np.zeros_like(trajectory)
        velocities[1:] = np.diff(trajectory, axis=0) * 25  # fps=25

        return {
            'positions': torch.from_numpy(trajectory).float(),
            'velocities': torch.from_numpy(velocities).float(),
        }


class SyntheticHomographyDataset(Dataset):
    """Generate synthetic pitch images with keypoint annotations."""

    def __init__(
        self,
        num_samples: int = 10000,
        image_size: Tuple[int, int] = (512, 512),
    ):
        self.num_samples = num_samples
        self.image_size = image_size

        # Standard pitch keypoints (world coordinates)
        self.world_keypoints = np.array([
            [0, 0], [105, 0], [105, 68], [0, 68],  # Corners
            [52.5, 0], [52.5, 68],  # Halfway line
            [11, 34], [94, 34],  # Penalty spots
            [52.5, 34],  # Center spot
        ], dtype=np.float32)

        logger.info(f"Created synthetic homography dataset with {num_samples} samples")

    def __len__(self) -> int:
        return self.num_samples

    def _generate_pitch_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic pitch image with random perspective."""
        # Create blank pitch
        img = np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        img[:, :] = [34, 139, 34]  # Green pitch

        # Random perspective transformation
        src_points = np.float32([
            [0, 0], [105, 0], [105, 68], [0, 68]
        ])

        # Random destination points (perspective distortion)
        h, w = self.image_size
        margin = 50
        dst_points = np.float32([
            [random.randint(margin, margin+50), random.randint(margin, margin+50)],
            [random.randint(w-margin-50, w-margin), random.randint(margin, margin+50)],
            [random.randint(w-margin-50, w-margin), random.randint(h-margin-50, h-margin)],
            [random.randint(margin, margin+50), random.randint(h-margin-50, h-margin)],
        ])

        # Compute homography
        H = cv2.getPerspectiveTransform(src_points, dst_points)
        H_inv = np.linalg.inv(H)

        # Project keypoints to image
        keypoints_homogeneous = np.hstack([
            self.world_keypoints,
            np.ones((len(self.world_keypoints), 1))
        ])
        pixel_keypoints = (H_inv @ keypoints_homogeneous.T).T
        pixel_keypoints = pixel_keypoints[:, :2] / pixel_keypoints[:, 2:3]

        # Draw lines
        # ... (optional: draw pitch lines for visualization)

        return img, pixel_keypoints, H

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, pixel_kps, homography = self._generate_pitch_image()

        # Create visibility mask (all visible for synthetic)
        visibility = np.ones(len(pixel_kps))

        return {
            'image': torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            'keypoints': torch.from_numpy(
                np.hstack([pixel_kps, visibility[:, None]])
            ).float(),
            'world_points': torch.from_numpy(self.world_keypoints).float(),
            'homography': torch.from_numpy(homography).float(),
        }
