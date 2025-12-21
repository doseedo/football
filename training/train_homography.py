#!/usr/bin/env python3
"""
Training script for Homography Estimation Model

Trains a keypoint detection network for camera calibration.

Usage:
    python train_homography.py --config configs/homography.yaml
    python train_homography.py --config configs/homography.yaml --resume models/checkpoints/homography/last.pth
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from loguru import logger
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.datasets import SoccerNetCalibrationDataset, SyntheticHomographyDataset, get_train_transforms, get_val_transforms
from training.utils import (
    KeypointHeatmapLoss,
    ReprojectionLoss,
    ReprojectionError,
    KeypointPCK,
    ModelCheckpoint,
    EarlyStopping,
    WandbLogger,
    ProgressLogger,
)


class SimpleKeypointNet(nn.Module):
    """
    Simple keypoint detection network for homography estimation.

    Architecture:
    - Backbone (ResNet/EfficientNet)
    - Heatmap head for keypoint localization
    - Homography regression head
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_keypoints: int = 29,
        heatmap_size: tuple = (64, 64),
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

        # Backbone
        if backbone == "resnet50":
            import torchvision.models as models
            resnet = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_channels = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")

        # Heatmap head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(backbone_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints, 1),
        )

        # Adaptive pooling to get fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(heatmap_size)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images

        Returns:
            heatmaps: (B, K, H', W') keypoint heatmaps
        """
        # Extract features
        features = self.backbone(x)  # (B, C, H', W')

        # Generate heatmaps
        heatmaps = self.heatmap_head(features)  # (B, K, H', W')
        heatmaps = self.adaptive_pool(heatmaps)  # (B, K, 64, 64)

        return heatmaps


class HomographyTrainer:
    """Trainer for homography estimation model."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = SimpleKeypointNet(
            backbone=config['model']['backbone'],
            num_keypoints=config['model']['num_keypoints'],
            heatmap_size=tuple(config['model']['heatmap_size']),
            pretrained=config['model']['pretrained'],
        ).to(self.device)

        # Loss functions
        self.heatmap_loss = KeypointHeatmapLoss(
            mse_weight=config['training']['loss_weights']['heatmap_mse'],
            focal_weight=config['training']['loss_weights']['heatmap_focal'],
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )

        # Data loaders
        self._setup_data()

        # Metrics
        self.train_metrics = {'reproj_error': ReprojectionError()}
        self.val_metrics = {
            'reproj_error': ReprojectionError(),
            'pck': KeypointPCK(threshold=config['evaluation']['pck_threshold']),
        }

        # Callbacks
        self._setup_callbacks()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False

    def _setup_data(self):
        """Setup data loaders."""
        data_config = self.config['data']

        # Transforms
        train_transforms = get_train_transforms(
            image_size=tuple(data_config['image_size']),
            task='homography',
        )
        val_transforms = get_val_transforms(
            image_size=tuple(data_config['image_size']),
        )

        # Datasets
        if data_config['dataset'] == 'soccernet_calibration':
            train_dataset = SoccerNetCalibrationDataset(
                root_path=data_config['root_path'],
                split='train',
                image_size=tuple(data_config['image_size']),
                transform=train_transforms,
            )
            val_dataset = SoccerNetCalibrationDataset(
                root_path=data_config['root_path'],
                split='val',
                image_size=tuple(data_config['image_size']),
                transform=val_transforms,
            )
        elif data_config['dataset'] == 'synthetic':
            train_dataset = SyntheticHomographyDataset(
                num_samples=10000,
                image_size=tuple(data_config['image_size']),
            )
            val_dataset = SyntheticHomographyDataset(
                num_samples=1000,
                image_size=tuple(data_config['image_size']),
            )
        else:
            raise ValueError(f"Unknown dataset: {data_config['dataset']}")

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
        )

        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    def _setup_callbacks(self):
        """Setup training callbacks."""
        self.callbacks = []

        # Checkpointing
        checkpoint_config = self.config['checkpoints']
        self.callbacks.append(ModelCheckpoint(
            save_dir=checkpoint_config['save_dir'],
            monitor=checkpoint_config['monitor'],
            mode=checkpoint_config['mode'],
            save_frequency=checkpoint_config['save_frequency'],
            keep_top_k=checkpoint_config['keep_top_k'],
        ))

        # Wandb logging
        if self.config['logging']['wandb']['enabled']:
            self.callbacks.append(WandbLogger(
                project=self.config['logging']['wandb']['project'],
                config=self.config,
                tags=self.config['logging']['wandb']['tags'],
            ))

        # Progress logging
        self.callbacks.append(ProgressLogger(
            log_frequency=self.config['logging']['log_frequency']
        ))

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0

        # Reset metrics
        for metric in self.train_metrics.values():
            metric.reset()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            target_heatmaps = self._generate_heatmaps(batch).to(self.device)

            # Forward
            pred_heatmaps = self.model(images)

            # Compute loss
            loss = self.heatmap_loss(pred_heatmaps, target_heatmaps)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1

            # Callbacks
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, loss.item(), self)

            pbar.set_postfix({'loss': loss.item()})

        return {'loss': epoch_loss / len(self.train_loader)}

    def validate(self):
        """Validate model."""
        self.model.eval()
        val_loss = 0.0

        # Reset metrics
        for metric in self.val_metrics.values():
            metric.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                target_heatmaps = self._generate_heatmaps(batch).to(self.device)

                # Forward
                pred_heatmaps = self.model(images)

                # Compute loss
                loss = self.heatmap_loss(pred_heatmaps, target_heatmaps)
                val_loss += loss.item()

                # Extract predicted keypoints from heatmaps
                pred_keypoints = self._heatmaps_to_keypoints(pred_heatmaps)

                # Update metrics (simplified - would need actual homography)
                # In practice, would compute homography from predicted keypoints

        # Compute metrics
        metrics = {'loss': val_loss / len(self.val_loader)}
        for name, metric in self.val_metrics.items():
            metrics.update(metric.compute())

        return metrics

    def _generate_heatmaps(self, batch):
        """Generate ground truth heatmaps from keypoints."""
        B = batch['image'].shape[0]
        K = self.model.num_keypoints
        H, W = self.model.heatmap_size

        heatmaps = torch.zeros(B, K, H, W)

        # Get keypoints from batch: shape (B, num_keypoints, 3) where 3 = (x, y, visibility)
        keypoints = batch['keypoints']  # Already in pixel coords for target image size
        num_kps = batch.get('num_keypoints', torch.full((B,), keypoints.shape[1]))

        # Image size used for keypoints (from config)
        img_h, img_w = 512, 512  # Default, should match config

        # Gaussian sigma for heatmap
        sigma = 2.0

        # Create coordinate grids
        y_grid = torch.arange(H).float().view(H, 1).expand(H, W)
        x_grid = torch.arange(W).float().view(1, W).expand(H, W)

        for b in range(B):
            n_kps = int(num_kps[b].item()) if isinstance(num_kps, torch.Tensor) else num_kps
            for k in range(min(n_kps, K)):
                kp = keypoints[b, k]
                x, y, vis = kp[0].item(), kp[1].item(), kp[2].item()

                if vis < 0.5 or (x == 0 and y == 0):
                    continue

                # Scale keypoint coords to heatmap size
                hm_x = x * W / img_w
                hm_y = y * H / img_h

                # Generate Gaussian heatmap
                gaussian = torch.exp(-((x_grid - hm_x)**2 + (y_grid - hm_y)**2) / (2 * sigma**2))
                heatmaps[b, k] = gaussian

        return heatmaps

    def _heatmaps_to_keypoints(self, heatmaps):
        """Convert heatmaps to keypoint coordinates."""
        B, K, H, W = heatmaps.shape
        keypoints = torch.zeros(B, K, 2)

        for b in range(B):
            for k in range(K):
                heatmap = heatmaps[b, k]
                idx = heatmap.argmax()
                y = idx // W
                x = idx % W

                # Scale to image coordinates
                keypoints[b, k, 0] = x * (512 / W)  # Assume 512x512 image
                keypoints[b, k, 1] = y * (512 / H)

        return keypoints

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")

        # Callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch

            # Callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self)

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, train_metrics, self)
                callback.on_validation_end(epoch, val_metrics, self)

            # Early stopping
            if self.should_stop:
                logger.info("Early stopping triggered")
                break

        # Training end callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)

        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train homography estimation model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Train
    trainer = HomographyTrainer(config)

    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {trainer.current_epoch}")

    trainer.train()


if __name__ == "__main__":
    main()
