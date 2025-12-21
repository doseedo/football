"""
SoccerNet Dataset Loaders

Loads SoccerNet data for various tasks:
- Camera calibration (keypoints, homography)
- Player tracking (trajectories, re-ID)
- Ball detection and tracking

Download Instructions:
1. Install SoccerNet pip package: pip install SoccerNet
2. Download datasets:
   - Calibration: python -m SoccerNet.Downloader --password <pwd> -s tracking
   - Tracking: python -m SoccerNet.Downloader --password <pwd> -s tracking

For access, register at: https://www.soccer-net.org/
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger

try:
    from SoccerNet.Downloader import SoccerNetDownloader
    SOCCERNET_AVAILABLE = True
except ImportError:
    SOCCERNET_AVAILABLE = False
    logger.warning("SoccerNet package not installed. Install with: pip install SoccerNet")


class SoccerNetCalibrationDataset(Dataset):
    """
    SoccerNet dataset for camera calibration and homography estimation.

    Returns:
        image: (C, H, W) torch.Tensor
        keypoints: (N, 3) torch.Tensor - pixel coordinates (x, y) + visibility
        world_points: (N, 2) torch.Tensor - world coordinates (meters)
        homography: (3, 3) torch.Tensor - ground truth homography matrix
    """

    PITCH_KEYPOINTS = {
        "Circle central": (52.5, 34.0),
        "Circle left": (43.35, 34.0),
        "Circle right": (61.65, 34.0),
        "Corner 1": (0.0, 0.0),
        "Corner 2": (105.0, 0.0),
        "Corner 3": (105.0, 68.0),
        "Corner 4": (0.0, 68.0),
        "Side line 1": (52.5, 0.0),
        "Side line 2": (52.5, 68.0),
        "Middle line": (52.5, 34.0),
        "Goal left post 1 left": (0.0, 30.34),
        "Goal left post 2 left": (0.0, 37.66),
        "Goal right post 1 right": (105.0, 30.34),
        "Goal right post 2 right": (105.0, 37.66),
        "Penalty left": (11.0, 34.0),
        "Penalty right": (94.0, 34.0),
        # Add more keypoints as needed
    }

    def __init__(
        self,
        root_path: str,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        transform=None,
        download: bool = False,
        password: Optional[str] = None,
    ):
        """
        Args:
            root_path: Path to SoccerNet data root
            split: 'train', 'val', or 'test'
            image_size: Target image size (H, W)
            transform: Image augmentation transforms
            download: Whether to download data if not present
            password: SoccerNet password for download
        """
        self.root_path = Path(root_path)
        self.split = split
        self.image_size = image_size
        self.transform = transform

        # Download if requested
        if download and SOCCERNET_AVAILABLE:
            self._download_data(password)

        # Load annotations
        self.samples = self._load_annotations()
        logger.info(f"Loaded {len(self.samples)} {split} samples for calibration")

    def _download_data(self, password: Optional[str]):
        """Download SoccerNet calibration data."""
        if not password:
            logger.error("Password required for SoccerNet download. Register at soccer-net.org")
            return

        downloader = SoccerNetDownloader(LocalDirectory=str(self.root_path))
        downloader.password = password
        downloader.downloadDataTask(task="calibration-2023", split=["train", "valid", "test"])

    def _load_annotations(self) -> List[Dict]:
        """Load calibration annotations from SoccerNet format."""
        samples = []

        # SoccerNet calibration structure: root/league/season/game/
        annotation_file = self.root_path / f"{self.split}_annotations.json"

        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                samples = data.get('samples', [])
        else:
            # Fallback: scan directory structure
            for league_dir in self.root_path.glob("*"):
                if not league_dir.is_dir():
                    continue
                for season_dir in league_dir.glob("*"):
                    if not season_dir.is_dir():
                        continue
                    for game_dir in season_dir.glob("*"):
                        calib_file = game_dir / "calibration.json"
                        if calib_file.exists():
                            samples.extend(self._parse_calibration_file(calib_file, game_dir))

        # Split data
        total = len(samples)
        if self.split == "train":
            samples = samples[:int(0.8 * total)]
        elif self.split == "val":
            samples = samples[int(0.8 * total):int(0.9 * total)]
        else:  # test
            samples = samples[int(0.9 * total):]

        return samples

    def _parse_calibration_file(self, calib_file: Path, game_dir: Path) -> List[Dict]:
        """Parse a single calibration JSON file."""
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)

        samples = []
        for frame_data in calib_data.get('frames', []):
            image_path = game_dir / frame_data['image']
            if not image_path.exists():
                continue

            # Extract keypoints
            pixel_kps = []
            world_kps = []
            visibility = []

            for kp in frame_data.get('keypoints', []):
                kp_name = kp['name']
                if kp_name in self.PITCH_KEYPOINTS:
                    pixel_kps.append([kp['x'], kp['y']])
                    world_kps.append(self.PITCH_KEYPOINTS[kp_name])
                    visibility.append(kp.get('visible', 1))

            if len(pixel_kps) >= 4:  # Need at least 4 points for homography
                samples.append({
                    'image_path': str(image_path),
                    'pixel_keypoints': np.array(pixel_kps, dtype=np.float32),
                    'world_keypoints': np.array(world_kps, dtype=np.float32),
                    'visibility': np.array(visibility, dtype=np.float32),
                    'homography': frame_data.get('homography', None),
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Resize image
        image_resized = cv2.resize(image, (self.image_size[1], self.image_size[0]))

        # Scale keypoints
        scale_x = self.image_size[1] / orig_w
        scale_y = self.image_size[0] / orig_h

        pixel_kps = sample['pixel_keypoints'].copy()
        pixel_kps[:, 0] *= scale_x
        pixel_kps[:, 1] *= scale_y

        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image_resized,
                keypoints=pixel_kps,
            )
            image_resized = transformed['image']
            pixel_kps = np.array(transformed.get('keypoints', pixel_kps))

        # Convert to tensors
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        # Create keypoint tensor with visibility
        num_kps = len(pixel_kps)
        keypoints = np.zeros((num_kps, 3), dtype=np.float32)
        keypoints[:, :2] = pixel_kps
        keypoints[:, 2] = sample['visibility']

        # Compute homography if not provided
        homography = sample.get('homography')
        if homography is None and num_kps >= 4:
            H, _ = cv2.findHomography(pixel_kps, sample['world_keypoints'])
            homography = H if H is not None else np.eye(3)
        elif homography is None:
            homography = np.eye(3)

        return {
            'image': image_tensor,
            'keypoints': torch.from_numpy(keypoints).float(),
            'world_points': torch.from_numpy(sample['world_keypoints']).float(),
            'homography': torch.from_numpy(np.array(homography)).float(),
        }


class SoccerNetTrackingDataset(Dataset):
    """
    SoccerNet dataset for player tracking and trajectory prediction.

    Used for:
    - Baller2Vec training (trajectory embeddings)
    - GNN training (tactical analysis)
    - Re-ID training (player appearance)
    """

    def __init__(
        self,
        root_path: str,
        split: str = "train",
        sequence_length: int = 100,
        stride: int = 25,
        fps: int = 25,
        download: bool = False,
        password: Optional[str] = None,
    ):
        self.root_path = Path(root_path)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.fps = fps

        if download and SOCCERNET_AVAILABLE:
            self._download_tracking_data(password)

        self.sequences = self._load_tracking_data()
        logger.info(f"Loaded {len(self.sequences)} tracking sequences for {split}")

    def _download_tracking_data(self, password: Optional[str]):
        """Download SoccerNet tracking data."""
        if not password:
            logger.error("Password required. Register at soccer-net.org")
            return

        downloader = SoccerNetDownloader(LocalDirectory=str(self.root_path))
        downloader.password = password
        downloader.downloadDataTask(task="tracking", split=["train", "valid", "test"])

    def _load_tracking_data(self) -> List[Dict]:
        """Load tracking sequences."""
        sequences = []

        # Look for tracking JSON files
        for json_file in self.root_path.rglob("tracking.json"):
            with open(json_file, 'r') as f:
                tracking_data = json.load(f)

            # Extract player trajectories
            for player_id, trajectory in tracking_data.get('players', {}).items():
                positions = np.array(trajectory['positions'])  # (T, 2)

                # Create overlapping sequences
                for start_idx in range(0, len(positions) - self.sequence_length, self.stride):
                    end_idx = start_idx + self.sequence_length
                    seq_positions = positions[start_idx:end_idx]

                    sequences.append({
                        'player_id': player_id,
                        'positions': seq_positions,
                        'team': trajectory.get('team', 0),
                        'game_id': str(json_file.parent),
                    })

        # Split data
        total = len(sequences)
        if self.split == "train":
            sequences = sequences[:int(0.85 * total)]
        elif self.split == "val":
            sequences = sequences[int(0.85 * total):int(0.95 * total)]
        else:
            sequences = sequences[int(0.95 * total):]

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        positions = torch.from_numpy(seq['positions']).float()

        # Compute velocities
        velocities = torch.zeros_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1]) * self.fps

        return {
            'positions': positions,
            'velocities': velocities,
            'team': torch.tensor(seq['team']),
            'player_id': seq['player_id'],
        }


class SoccerNetDetectionDataset(Dataset):
    """
    SoccerNet dataset for player and ball detection (DETR training).
    """

    def __init__(
        self,
        root_path: str,
        split: str = "train",
        image_size: Tuple[int, int] = (800, 1333),
        transform=None,
    ):
        self.root_path = Path(root_path)
        self.split = split
        self.image_size = image_size
        self.transform = transform

        self.samples = self._load_detection_annotations()
        logger.info(f"Loaded {len(self.samples)} detection samples for {split}")

    def _load_detection_annotations(self) -> List[Dict]:
        """Load detection annotations (bounding boxes)."""
        # Placeholder - actual implementation depends on annotation format
        samples = []

        annotation_file = self.root_path / f"{self.split}_detections.json"
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                samples = json.load(f)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotations
        boxes = np.array(sample['boxes'])  # (N, 4) - xyxy format
        labels = np.array(sample['labels'])  # (N,) - class IDs

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels,
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])

        return {
            'image': torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            'boxes': torch.from_numpy(boxes).float(),
            'labels': torch.from_numpy(labels).long(),
        }


def download_soccernet_data(
    root_path: str,
    password: str,
    tasks: List[str] = ["calibration-2023", "tracking"],
):
    """
    Convenience function to download all required SoccerNet data.

    Args:
        root_path: Where to save data
        password: SoccerNet password (get from soccer-net.org)
        tasks: Which tasks to download

    Example:
        download_soccernet_data(
            root_path="data/training/soccernet",
            password="your_password",
            tasks=["calibration-2023", "tracking"]
        )
    """
    if not SOCCERNET_AVAILABLE:
        raise ImportError("SoccerNet not installed. Install with: pip install SoccerNet")

    downloader = SoccerNetDownloader(LocalDirectory=root_path)
    downloader.password = password

    for task in tasks:
        logger.info(f"Downloading SoccerNet {task}...")
        downloader.downloadDataTask(task=task, split=["train", "valid", "test"])

    logger.info("Download complete!")
