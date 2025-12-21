#!/usr/bin/env python3
"""Training script for 3D ball trajectory LSTM model."""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ball3d import (
    SyntheticDataGenerator,
    TrajectoryLSTM,
    TrajectoryLoss,
    train_step,
    evaluate,
    create_context_features
)


class SyntheticTrajectoryDataset(Dataset):
    """Dataset of synthetic ball trajectories."""

    def __init__(self, data_file: str, sequence_length: int = 20):
        """
        Load synthetic trajectory dataset.

        Args:
            data_file: Path to .npz file with synthetic data
            sequence_length: Length of input sequences
        """
        logger.info(f"Loading dataset from {data_file}")
        self.data = np.load(data_file, allow_pickle=True)
        self.sequence_length = sequence_length

        # Extract metadata
        self.num_samples = int(self.data['num_samples'])
        self.pitch_length = float(self.data['pitch_length'])
        self.pitch_width = float(self.data['pitch_width'])

        logger.info(f"Loaded {self.num_samples} trajectories")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Get a single trajectory sample."""
        prefix = f'traj_{idx}'

        positions_3d = self.data[f'{prefix}_3d']
        positions_2d = self.data[f'{prefix}_2d']
        is_visible = self.data[f'{prefix}_visible']

        # Filter to visible positions
        visible_idx = np.where(is_visible)[0]

        if len(visible_idx) < 3:
            # Not enough visible positions, return a default sample
            return self.__getitem__((idx + 1) % self.num_samples)

        # Sample a subsequence
        if len(visible_idx) > self.sequence_length:
            start_idx = np.random.randint(0, len(visible_idx) - self.sequence_length)
            seq_idx = visible_idx[start_idx:start_idx + self.sequence_length]
        else:
            # Pad if too short
            seq_idx = visible_idx

        seq_2d = positions_2d[seq_idx]
        seq_3d = positions_3d[seq_idx]

        # Pad if necessary
        if len(seq_2d) < self.sequence_length:
            pad_len = self.sequence_length - len(seq_2d)
            seq_2d = np.vstack([np.repeat(seq_2d[0:1], pad_len, axis=0), seq_2d])
            seq_3d = np.vstack([np.repeat(seq_3d[0:1], pad_len, axis=0), seq_3d])

        # Create dummy homography (identity for synthetic data)
        # In real training, this would come from actual camera calibration
        homography = np.eye(3, dtype=np.float32)
        context = create_context_features(homography, camera_height=15.0, camera_angle=30.0)

        return {
            'positions_2d': torch.from_numpy(seq_2d.astype(np.float32)),
            'positions_3d': torch.from_numpy(seq_3d.astype(np.float32)),
            'context': torch.from_numpy(context.astype(np.float32))
        }


def generate_training_data(output_path: str, num_samples: int = 10000):
    """Generate synthetic training data."""
    logger.info(f"Generating {num_samples} synthetic trajectories...")

    generator = SyntheticDataGenerator(
        pitch_length=105.0,
        pitch_width=68.0,
        fps=25.0,
        noise_level=0.02
    )

    generator.save_dataset(
        filepath=output_path,
        num_samples=num_samples,
        trajectory_types=['pass', 'shot', 'cross', 'bounce', 'aerial']
    )

    logger.info(f"Training data saved to {output_path}")


def train_model(
    train_data: str,
    val_data: str,
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = 'cuda'
):
    """
    Train the 3D trajectory LSTM model.

    Args:
        train_data: Path to training data .npz file
        val_data: Path to validation data .npz file
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    train_dataset = SyntheticTrajectoryDataset(train_data, sequence_length=20)
    val_dataset = SyntheticTrajectoryDataset(val_data, sequence_length=20)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    model = TrajectoryLSTM(
        input_dim=2,
        context_dim=11,
        hidden_dim=256,
        num_layers=2,
        output_dim=3,
        dropout=0.2,
        predict_uncertainty=True
    ).to(device)

    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = TrajectoryLoss(
        position_weight=1.0,
        velocity_weight=0.5,
        physics_weight=0.3,
        uncertainty_weight=0.1
    )

    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            loss_dict = train_step(
                model, optimizer,
                batch['positions_2d'],
                batch['positions_3d'],
                batch['context'],
                criterion,
                device
            )
            train_losses.append(loss_dict['total'])
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

        avg_train_loss = np.mean(train_losses)

        # Validation
        val_loss_dict = evaluate(model, val_loader, criterion, device)
        avg_val_loss = val_loss_dict['total']

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {avg_val_loss:.4f} "
            f"(pos: {val_loss_dict['position']:.4f}, "
            f"vel: {val_loss_dict['velocity']:.4f}, "
            f"phys: {val_loss_dict['physics']:.4f})"
        )

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info(f"Training complete! Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train 3D Ball Trajectory LSTM")

    parser.add_argument(
        '--mode',
        choices=['generate', 'train', 'both'],
        default='both',
        help='Mode: generate data, train model, or both'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of synthetic samples to generate'
    )

    parser.add_argument(
        '--train-data',
        default='data/ball3d/train_trajectories.npz',
        help='Path to training data'
    )

    parser.add_argument(
        '--val-data',
        default='data/ball3d/val_trajectories.npz',
        help='Path to validation data'
    )

    parser.add_argument(
        '--output-dir',
        default='models/ball3d',
        help='Directory to save model checkpoints'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )

    parser.add_argument(
        '--device',
        default='cuda',
        help='Device (cuda or cpu)'
    )

    args = parser.parse_args()

    # Create data directory
    Path(args.train_data).parent.mkdir(parents=True, exist_ok=True)

    if args.mode in ['generate', 'both']:
        logger.info("Generating training data...")
        generate_training_data(args.train_data, args.num_samples)

        logger.info("Generating validation data...")
        generate_training_data(args.val_data, args.num_samples // 5)

    if args.mode in ['train', 'both']:
        logger.info("Starting training...")
        train_model(
            train_data=args.train_data,
            val_data=args.val_data,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device
        )


if __name__ == '__main__':
    main()
