"""
Data Augmentation for Football Tracking

Provides augmentation for:
- Images (for detection, homography)
- Trajectories (for Baller2Vec, GNN)
- Bounding boxes (for DETR, re-ID)
"""

from typing import List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch


def get_train_transforms(
    image_size: Tuple[int, int] = (512, 512),
    task: str = "detection",
) -> A.Compose:
    """
    Get training augmentation pipeline.

    Args:
        image_size: Target image size (H, W)
        task: 'detection', 'reid', 'homography', or 'ball'

    Returns:
        Albumentations composition
    """

    if task == "detection" or task == "detr":
        # DETR-style augmentations
        transforms = A.Compose([
            A.RandomResizedCrop(
                height=image_size[0],
                width=image_size[1],
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.8
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))

    elif task == "reid":
        # Re-ID augmentations
        transforms = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.PadIfNeeded(
                min_height=image_size[0] + 20,
                min_width=image_size[1] + 20,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            A.RandomCrop(height=image_size[0], width=image_size[1], p=1.0),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=1,
                max_height=int(image_size[0] * 0.4),
                max_width=int(image_size[1] * 0.4),
                min_holes=1,
                min_height=int(image_size[0] * 0.1),
                min_width=int(image_size[1] * 0.1),
                fill_value=0,
                p=0.5
            ),  # Random erasing
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

    elif task == "homography":
        # Homography/calibration augmentations
        transforms = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10, 30), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Perspective(scale=(0.05, 0.1), p=0.3),  # Additional perspective
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        ))

    elif task == "ball":
        # Ball detection augmentations
        transforms = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.8
            ),
            A.MotionBlur(blur_limit=7, p=0.3),  # Simulate ball motion
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))

    else:
        # Generic augmentation
        transforms = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

    return transforms


def get_val_transforms(
    image_size: Tuple[int, int] = (512, 512),
) -> A.Compose:
    """
    Get validation transforms (no augmentation).

    Args:
        image_size: Target image size (H, W)

    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


class TrajectoryAugmentation:
    """
    Augmentation for trajectory data (Baller2Vec, GNN).

    Augmentations:
    - Rotation
    - Flip (horizontal/vertical)
    - Translation
    - Scaling
    - Noise injection
    - Time warping
    - Point dropout
    """

    def __init__(
        self,
        rotation: float = 360.0,  # Max rotation in degrees
        flip_horizontal: bool = True,
        flip_vertical: bool = True,
        translation: float = 5.0,  # meters
        scaling: Tuple[float, float] = (0.95, 1.05),
        noise_std: float = 0.1,  # meters
        time_warping: float = 0.2,  # Max time warp factor
        dropout_prob: float = 0.05,  # Probability of dropping a point
        pitch_dimensions: Tuple[float, float] = (105.0, 68.0),
    ):
        self.rotation = rotation
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.translation = translation
        self.scaling = scaling
        self.noise_std = noise_std
        self.time_warping = time_warping
        self.dropout_prob = dropout_prob
        self.pitch_dimensions = pitch_dimensions

    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to trajectory.

        Args:
            trajectory: (T, 2) array of (x, y) positions

        Returns:
            Augmented trajectory (T, 2)
        """
        traj = trajectory.copy()

        # Center trajectory
        center = np.array([self.pitch_dimensions[0] / 2, self.pitch_dimensions[1] / 2])
        traj_centered = traj - center

        # Rotation
        if self.rotation > 0:
            angle = np.random.uniform(-self.rotation, self.rotation)
            angle_rad = np.radians(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            traj_centered = traj_centered @ rotation_matrix.T

        # Flip
        if self.flip_horizontal and np.random.random() < 0.5:
            traj_centered[:, 0] *= -1

        if self.flip_vertical and np.random.random() < 0.5:
            traj_centered[:, 1] *= -1

        # Scaling
        if self.scaling:
            scale = np.random.uniform(*self.scaling)
            traj_centered *= scale

        # Translate back
        traj = traj_centered + center

        # Additional translation
        if self.translation > 0:
            dx = np.random.uniform(-self.translation, self.translation)
            dy = np.random.uniform(-self.translation, self.translation)
            traj[:, 0] += dx
            traj[:, 1] += dy

        # Clip to pitch boundaries
        traj[:, 0] = np.clip(traj[:, 0], 0, self.pitch_dimensions[0])
        traj[:, 1] = np.clip(traj[:, 1], 0, self.pitch_dimensions[1])

        # Add noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, traj.shape)
            traj += noise

        # Time warping
        if self.time_warping > 0 and len(traj) > 2:
            warp_factor = np.random.uniform(1 - self.time_warping, 1 + self.time_warping)
            indices = np.linspace(0, len(traj) - 1, int(len(traj) * warp_factor))
            indices = np.clip(indices, 0, len(traj) - 1).astype(int)
            traj = traj[indices]

            # Resample to original length
            if len(traj) != len(trajectory):
                x_interp = np.interp(
                    np.linspace(0, len(traj) - 1, len(trajectory)),
                    np.arange(len(traj)),
                    traj[:, 0]
                )
                y_interp = np.interp(
                    np.linspace(0, len(traj) - 1, len(trajectory)),
                    np.arange(len(traj)),
                    traj[:, 1]
                )
                traj = np.column_stack([x_interp, y_interp])

        # Point dropout
        if self.dropout_prob > 0:
            mask = np.random.random(len(traj)) > self.dropout_prob
            # Interpolate dropped points
            for i in range(len(traj)):
                if not mask[i] and i > 0 and i < len(traj) - 1:
                    # Linear interpolation
                    traj[i] = (traj[i - 1] + traj[i + 1]) / 2

        return traj


class GraphAugmentation:
    """
    Augmentation for graph-based data (GNN training).

    Augmentations:
    - Node feature masking
    - Edge dropout
    - Node dropout
    - Spatial transformations
    """

    def __init__(
        self,
        node_feature_mask_prob: float = 0.1,
        edge_dropout_prob: float = 0.1,
        node_dropout_prob: float = 0.05,
        spatial_augment: bool = True,
    ):
        self.node_feature_mask_prob = node_feature_mask_prob
        self.edge_dropout_prob = edge_dropout_prob
        self.node_dropout_prob = node_dropout_prob
        self.spatial_augment = spatial_augment

        if spatial_augment:
            self.traj_aug = TrajectoryAugmentation()

    def __call__(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply graph augmentation.

        Args:
            node_features: (N, F) node feature matrix
            edge_index: (2, E) edge connectivity

        Returns:
            Augmented node features and edge index
        """
        # Node feature masking
        if self.node_feature_mask_prob > 0:
            mask = np.random.random(node_features.shape) < self.node_feature_mask_prob
            node_features = node_features.copy()
            node_features[mask] = 0

        # Edge dropout
        if self.edge_dropout_prob > 0 and edge_index.shape[1] > 0:
            keep_mask = np.random.random(edge_index.shape[1]) > self.edge_dropout_prob
            edge_index = edge_index[:, keep_mask]

        # Spatial augmentation (if node features include positions)
        if self.spatial_augment and node_features.shape[1] >= 2:
            # Assume first 2 features are x, y
            positions = node_features[:, :2]
            augmented_positions = self.traj_aug(positions)
            node_features = node_features.copy()
            node_features[:, :2] = augmented_positions

        return node_features, edge_index


# Example usage for custom datasets
class MixUp:
    """
    MixUp augmentation for trajectories.

    Mixes two trajectories with random weight.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        traj1: np.ndarray,
        traj2: np.ndarray,
    ) -> np.ndarray:
        """Mix two trajectories."""
        lam = np.random.beta(self.alpha, self.alpha)
        return lam * traj1 + (1 - lam) * traj2


class CutMix:
    """
    CutMix for trajectories.

    Replaces a segment of one trajectory with another.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self,
        traj1: np.ndarray,
        traj2: np.ndarray,
    ) -> np.ndarray:
        """Mix two trajectories."""
        lam = np.random.beta(self.alpha, self.alpha)
        length = len(traj1)

        # Random segment
        cut_len = int(length * lam)
        cut_start = np.random.randint(0, length - cut_len + 1)
        cut_end = cut_start + cut_len

        # Mix
        mixed = traj1.copy()
        mixed[cut_start:cut_end] = traj2[cut_start:cut_end]

        return mixed
