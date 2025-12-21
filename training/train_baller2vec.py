"""Training script for Baller2Vec trajectory prediction models.

This script demonstrates how to train the transformer-based extrapolation models
on trajectory data.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import yaml
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extrapolation import (
    Baller2Vec,
    Baller2VecPlus,
    create_feature_tensor,
    create_padding_mask,
    create_team_tensor
)


class TrajectoryDataset(Dataset):
    """Dataset for trajectory prediction.

    Expected data format:
    - positions: (N, seq_len, num_players, 2) - player positions over time
    - teams: (N, num_players) - team assignments (0 or 1)
    - masks: (N, num_players) - padding masks
    """

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 25,
        prediction_horizon: int = 10,
        max_players: int = 22
    ):
        """Initialize dataset.

        Args:
            data_path: Path to trajectory data (NPZ file)
            sequence_length: Length of input sequence
            prediction_horizon: Number of future steps to predict
            max_players: Maximum number of players
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.max_players = max_players

        # Load data
        data = np.load(data_path)
        self.positions = data['positions']  # (N, total_len, num_players, 2)
        self.teams = data['teams']  # (N, num_players)

        if 'masks' in data:
            self.masks = data['masks']  # (N, num_players)
        else:
            self.masks = np.zeros((len(self.positions), max_players), dtype=bool)

        logger.info(f"Loaded {len(self.positions)} sequences from {data_path}")

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.

        Returns:
            Dictionary with:
                - input_features: (seq_len, num_players, 6)
                - target_positions: (pred_horizon, num_players, 2)
                - teams: (num_players,)
                - mask: (num_players,)
        """
        # Get full sequence
        full_positions = self.positions[idx]  # (total_len, num_players, 2)
        total_len = full_positions.shape[0]

        # Random starting point (ensure we have enough for input + prediction)
        max_start = total_len - self.sequence_length - self.prediction_horizon
        if max_start <= 0:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, max_start)

        # Extract input and target
        input_positions = full_positions[start_idx:start_idx + self.sequence_length]
        target_positions = full_positions[
            start_idx + self.sequence_length:start_idx + self.sequence_length + self.prediction_horizon
        ]

        # Compute velocities
        velocities = np.zeros_like(input_positions)
        velocities[1:] = input_positions[1:] - input_positions[:-1]

        # Create features
        features = create_feature_tensor(
            input_positions,
            self.teams[idx],
            velocities
        )

        return {
            'input_features': features,
            'target_positions': torch.from_numpy(target_positions).float(),
            'teams': torch.from_numpy(self.teams[idx]).long(),
            'mask': torch.from_numpy(self.masks[idx]).bool()
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    model_type: str
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device for training
        model_type: 'baller2vec' or 'baller2vec_plus'

    Returns:
        Dictionary of average losses
    """
    model.train()
    total_loss = 0
    total_position_loss = 0
    total_velocity_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move to device
        input_features = batch['input_features'].to(device)
        target_positions = batch['target_positions'].to(device)
        teams = batch['teams'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass
        optimizer.zero_grad()

        if model_type == 'baller2vec_plus':
            # Baller2Vec++ - use full forward pass
            predictions, uncertainty = model(
                input_features,
                teams,
                mask=mask,
                return_uncertainty=True
            )

            # Predict future steps
            future_pred, future_unc = model.predict_future(
                input_features,
                teams,
                n_future_steps=target_positions.shape[1],
                mask=mask,
                return_uncertainty=True
            )

            # Compute loss
            loss, loss_dict = model.compute_loss(
                future_pred,
                target_positions,
                uncertainty=future_unc,
                mask=mask
            )
        else:
            # Baller2Vec - simpler loss
            predictions = model.predict_future(
                input_features,
                n_future_steps=target_positions.shape[1],
                mask=mask,
                autoregressive=True
            )

            loss = model.compute_loss(predictions, target_positions, mask=mask)
            loss_dict = {'position_loss': loss.item(), 'total_loss': loss.item()}

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update metrics
        total_loss += loss_dict['total_loss']
        total_position_loss += loss_dict.get('position_loss', loss_dict['total_loss'])
        total_velocity_loss += loss_dict.get('velocity_loss', 0.0)
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'pos': f"{loss_dict.get('position_loss', 0):.4f}"
        })

    return {
        'total_loss': total_loss / num_batches,
        'position_loss': total_position_loss / num_batches,
        'velocity_loss': total_velocity_loss / num_batches
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    model_type: str
) -> Dict[str, float]:
    """Validate model.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        device: Device for validation
        model_type: 'baller2vec' or 'baller2vec_plus'

    Returns:
        Dictionary of average losses
    """
    model.eval()
    total_loss = 0
    total_ade = 0  # Average Displacement Error
    total_fde = 0  # Final Displacement Error
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            input_features = batch['input_features'].to(device)
            target_positions = batch['target_positions'].to(device)
            teams = batch['teams'].to(device)
            mask = batch['mask'].to(device)

            # Predict
            if model_type == 'baller2vec_plus':
                predictions, _ = model.predict_future(
                    input_features,
                    teams,
                    n_future_steps=target_positions.shape[1],
                    mask=mask,
                    return_uncertainty=False
                )
            else:
                predictions = model.predict_future(
                    input_features,
                    n_future_steps=target_positions.shape[1],
                    mask=mask,
                    autoregressive=False
                )

            # Compute metrics
            if model_type == 'baller2vec_plus':
                loss, _ = model.compute_loss(predictions, target_positions, mask=mask)
            else:
                loss = model.compute_loss(predictions, target_positions, mask=mask)

            # Compute ADE and FDE
            mask_expanded = (~mask).unsqueeze(1).unsqueeze(-1).float()
            errors = torch.norm(predictions - target_positions, dim=-1)  # (batch, seq, players)
            ade = (errors * mask_expanded.squeeze(-1)).sum() / mask_expanded.sum()
            fde = (errors[:, -1, :] * mask_expanded[:, 0, :, 0]).sum() / mask_expanded[:, 0, :, 0].sum()

            total_loss += loss.item()
            total_ade += ade.item()
            total_fde += fde.item()
            num_batches += 1

    return {
        'val_loss': total_loss / num_batches,
        'ade': total_ade / num_batches,
        'fde': total_fde / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train Baller2Vec trajectory prediction')
    parser.add_argument('--config', type=str, default='training/configs/baller2vec.yaml',
                       help='Path to config file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data (NPZ file)')
    parser.add_argument('--val-data', type=str,
                       help='Path to validation data (optional)')
    parser.add_argument('--output', type=str, default='models/baller2vec.pth',
                       help='Output model path')
    parser.add_argument('--model-type', type=str, default='baller2vec_plus',
                       choices=['baller2vec', 'baller2vec_plus'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    args = parser.parse_args()

    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training {args.model_type} on {device}")

    # Create datasets
    train_dataset = TrajectoryDataset(
        args.data,
        sequence_length=25,
        prediction_horizon=10,
        max_players=22
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = None
    if args.val_data:
        val_dataset = TrajectoryDataset(args.val_data)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # Create model
    if args.model_type == 'baller2vec_plus':
        model = Baller2VecPlus(
            d_model=256,
            num_heads=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            max_players=22,
            use_player_embeddings=True
        )
    else:
        model = Baller2Vec(
            d_model=256,
            num_heads=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            max_players=22,
            use_player_embeddings=True
        )

    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.model_type)
        logger.info(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                   f"Position: {train_metrics['position_loss']:.4f}")

        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, device, args.model_type)
            logger.info(f"Val - Loss: {val_metrics['val_loss']:.4f}, "
                       f"ADE: {val_metrics['ade']:.2f}m, "
                       f"FDE: {val_metrics['fde']:.2f}m")

            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['val_loss'],
                    'model_type': args.model_type
                }, args.output)
                logger.info(f"Saved best model to {args.output}")

        scheduler.step()

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
