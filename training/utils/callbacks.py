"""
Training Callbacks

Includes:
- Model checkpointing
- Early stopping
- Learning rate scheduling
- Wandb logging
- Visualization callbacks
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Callback:
    """Base callback class."""

    def on_train_begin(self, trainer: Any):
        pass

    def on_train_end(self, trainer: Any):
        pass

    def on_epoch_begin(self, epoch: int, trainer: Any):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        pass

    def on_batch_begin(self, batch_idx: int, trainer: Any):
        pass

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        pass

    def on_validation_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        pass


class ModelCheckpoint(Callback):
    """
    Save model checkpoints based on monitoring metric.

    Saves:
    - Best model (based on metric)
    - Latest model
    - Periodic checkpoints
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_frequency: int = 5,
        keep_top_k: int = 3,
        save_last: bool = True,
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor (e.g., 'val_loss', 'val_mAP')
            mode: 'min' or 'max' - whether to minimize or maximize monitor
            save_frequency: Save every N epochs
            keep_top_k: Keep top K checkpoints based on monitor
            save_last: Always save latest checkpoint
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_frequency = save_frequency
        self.keep_top_k = keep_top_k
        self.save_last = save_last

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.saved_checkpoints = []  # List of (value, path) tuples

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Save checkpoint if needed."""
        # Save periodic checkpoint
        if (epoch + 1) % self.save_frequency == 0:
            self._save_checkpoint(epoch, metrics, trainer, prefix=f"epoch_{epoch+1}")

        # Save last checkpoint
        if self.save_last:
            self._save_checkpoint(epoch, metrics, trainer, prefix="last")

    def on_validation_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Save best checkpoint based on validation metric."""
        if self.monitor not in metrics:
            logger.warning(f"Monitor metric '{self.monitor}' not found in metrics")
            return

        current_value = metrics[self.monitor]

        # Check if this is the best model
        is_best = False
        if self.mode == 'min':
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value

        if is_best:
            self.best_value = current_value
            checkpoint_path = self._save_checkpoint(
                epoch, metrics, trainer, prefix="best"
            )
            logger.info(f"New best model: {self.monitor}={current_value:.4f}")

            # Track for top-k
            self.saved_checkpoints.append((current_value, checkpoint_path))
            self.saved_checkpoints.sort(
                key=lambda x: x[0],
                reverse=(self.mode == 'max')
            )

            # Remove old checkpoints beyond top-k
            while len(self.saved_checkpoints) > self.keep_top_k:
                _, old_path = self.saved_checkpoints.pop()
                if old_path.exists() and "best" not in old_path.name:
                    old_path.unlink()
                    logger.info(f"Removed old checkpoint: {old_path}")

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        trainer: Any,
        prefix: str = "checkpoint",
    ) -> Path:
        """Save a checkpoint."""
        checkpoint_path = self.save_dir / f"{prefix}.pth"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': metrics,
            'best_value': self.best_value,
        }

        if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        return checkpoint_path


class EarlyStopping(Callback):
    """
    Stop training if monitored metric doesn't improve.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0

    def on_validation_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Check if should stop."""
        if self.monitor not in metrics:
            return

        current_value = metrics[self.monitor]

        # Check for improvement
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            trainer.should_stop = True
            logger.info(f"Early stopping triggered at epoch {epoch+1}")


class LearningRateScheduler(Callback):
    """
    Learning rate scheduling callback.

    Supports:
    - CosineAnnealingLR
    - ReduceLROnPlateau
    - StepLR
    """

    def __init__(
        self,
        scheduler_type: str = "cosine",
        **scheduler_kwargs,
    ):
        """
        Args:
            scheduler_type: 'cosine', 'plateau', 'step', or 'warmup'
            scheduler_kwargs: Arguments for the scheduler
        """
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler = None

    def on_train_begin(self, trainer: Any):
        """Initialize scheduler."""
        if self.scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                trainer.optimizer,
                **self.scheduler_kwargs
            )
        elif self.scheduler_type == "plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(
                trainer.optimizer,
                **self.scheduler_kwargs
            )
        elif self.scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(
                trainer.optimizer,
                **self.scheduler_kwargs
            )
        else:
            logger.warning(f"Unknown scheduler type: {self.scheduler_type}")

        trainer.scheduler = self.scheduler

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Step the scheduler."""
        if self.scheduler is None:
            return

        if self.scheduler_type == "plateau":
            # Needs a metric
            monitor = self.scheduler_kwargs.get('monitor', 'val_loss')
            if monitor in metrics:
                self.scheduler.step(metrics[monitor])
        else:
            self.scheduler.step()

        # Log current LR
        current_lr = trainer.optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.6f}")


class WandbLogger(Callback):
    """
    Weights & Biases logging callback.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Args:
            project: W&B project name
            name: Run name
            entity: W&B entity (username or team)
            config: Configuration dictionary to log
            tags: List of tags
        """
        if not WANDB_AVAILABLE:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.enabled = False
            return

        self.project = project
        self.name = name
        self.entity = entity
        self.config = config or {}
        self.tags = tags or []
        self.enabled = True

    def on_train_begin(self, trainer: Any):
        """Initialize wandb run."""
        if not self.enabled:
            return

        wandb.init(
            project=self.project,
            name=self.name,
            entity=self.entity,
            config=self.config,
            tags=self.tags,
        )

        # Log model architecture
        if hasattr(trainer, 'model'):
            wandb.watch(trainer.model, log='all')

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        """Log batch metrics."""
        if not self.enabled:
            return

        wandb.log({
            'batch_loss': loss,
            'batch': trainer.global_step,
        })

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Log epoch metrics."""
        if not self.enabled:
            return

        log_dict = {'epoch': epoch}
        log_dict.update({f'train_{k}': v for k, v in metrics.items()})

        # Log learning rate
        if hasattr(trainer, 'optimizer'):
            log_dict['learning_rate'] = trainer.optimizer.param_groups[0]['lr']

        wandb.log(log_dict)

    def on_validation_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Log validation metrics."""
        if not self.enabled:
            return

        log_dict = {'epoch': epoch}
        log_dict.update({f'val_{k}': v for k, v in metrics.items()})

        wandb.log(log_dict)

    def on_train_end(self, trainer: Any):
        """Finish wandb run."""
        if not self.enabled:
            return

        wandb.finish()


class VisualizationCallback(Callback):
    """
    Visualization callback for different tasks.

    Creates visualizations during training:
    - Trajectory predictions
    - Keypoint detections
    - Detection boxes
    - Attention maps (for transformers)
    """

    def __init__(
        self,
        save_dir: str,
        visualization_frequency: int = 5,
        num_samples: int = 4,
        task: str = "trajectory",
    ):
        """
        Args:
            save_dir: Directory to save visualizations
            visualization_frequency: Visualize every N epochs
            num_samples: Number of samples to visualize
            task: 'trajectory', 'keypoint', 'detection', or 'attention'
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.visualization_frequency = visualization_frequency
        self.num_samples = num_samples
        self.task = task

    def on_validation_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Create visualizations."""
        if (epoch + 1) % self.visualization_frequency != 0:
            return

        if self.task == "trajectory":
            self._visualize_trajectories(epoch, trainer)
        elif self.task == "keypoint":
            self._visualize_keypoints(epoch, trainer)
        elif self.task == "detection":
            self._visualize_detections(epoch, trainer)

    def _visualize_trajectories(self, epoch: int, trainer: Any):
        """Visualize trajectory predictions."""
        # Get validation batch
        if not hasattr(trainer, 'val_loader'):
            return

        try:
            batch = next(iter(trainer.val_loader))
        except:
            return

        # Forward pass
        trainer.model.eval()
        with torch.no_grad():
            if isinstance(batch, dict):
                inputs = {k: v.to(trainer.device) for k, v in batch.items() if k != 'target'}
                outputs = trainer.model(**inputs)
            else:
                outputs = trainer.model(batch[0].to(trainer.device))

        # Plot trajectories
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(min(self.num_samples, len(batch['positions']))):
            ax = axes[i]

            # Ground truth
            gt_traj = batch['positions'][i].cpu().numpy()
            ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='Ground Truth', linewidth=2)

            # Prediction
            if isinstance(outputs, dict) and 'positions' in outputs:
                pred_traj = outputs['positions'][i].cpu().numpy()
            else:
                pred_traj = outputs[i].cpu().numpy()

            ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label='Prediction', linewidth=2)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Sample {i+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 68)

        plt.tight_layout()
        save_path = self.save_dir / f"trajectories_epoch_{epoch+1}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

        logger.info(f"Saved trajectory visualization: {save_path}")

    def _visualize_keypoints(self, epoch: int, trainer: Any):
        """Visualize keypoint predictions."""
        # Similar to trajectories but for keypoints
        # Implementation details depend on model output format
        pass

    def _visualize_detections(self, epoch: int, trainer: Any):
        """Visualize detection boxes."""
        # Similar to above but for bounding boxes
        pass


class GradientClipping(Callback):
    """
    Gradient clipping callback.

    Useful for training stability, especially for RNNs and Transformers.
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm (2.0 for L2 norm)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        """Clip gradients."""
        if hasattr(trainer, 'model'):
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(),
                self.max_norm,
                self.norm_type
            )


class ProgressLogger(Callback):
    """
    Simple progress logging callback.

    Logs training progress to console.
    """

    def __init__(self, log_frequency: int = 10):
        """
        Args:
            log_frequency: Log every N batches
        """
        self.log_frequency = log_frequency

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        """Log batch progress."""
        if (batch_idx + 1) % self.log_frequency == 0:
            logger.info(
                f"Epoch {trainer.current_epoch+1} | "
                f"Batch {batch_idx+1}/{len(trainer.train_loader)} | "
                f"Loss: {loss:.4f}"
            )

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Log epoch summary."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch+1} completed | {metrics_str}")

    def on_validation_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Log validation results."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Validation | {metrics_str}")
