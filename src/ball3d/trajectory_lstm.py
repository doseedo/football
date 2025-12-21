"""LSTM-based 3D ball trajectory estimation network."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class TrajectoryLSTM(nn.Module):
    """
    LSTM network for 3D ball trajectory estimation from 2D sequences.

    Architecture:
    - Input projection: transforms 2D positions + context to feature space
    - LSTM layers: capture temporal dynamics
    - Output head: predicts 3D position (x, y, z) with uncertainty
    """

    def __init__(
        self,
        input_dim: int = 2,  # 2D position
        context_dim: int = 9,  # homography (9 params) + camera info
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 3,  # 3D position (x, y, z)
        dropout: float = 0.2,
        predict_uncertainty: bool = True
    ):
        """
        Initialize trajectory LSTM.

        Args:
            input_dim: Dimension of input (2 for x, y)
            context_dim: Dimension of context features (homography, camera)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension (3 for x, y, z)
            dropout: Dropout rate
            predict_uncertainty: Whether to predict uncertainty (variance)
        """
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.predict_uncertainty = predict_uncertainty

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Context projection (for homography and camera params)
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # Output head - mean prediction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # Uncertainty head - variance prediction
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
                nn.Softplus()  # Ensure positive variance
            )

    def forward(
        self,
        positions_2d: torch.Tensor,
        context: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            positions_2d: (batch, seq_len, 2) - 2D ball positions
            context: (batch, context_dim) - homography matrix + camera info
            hidden: Optional hidden state for sequential processing

        Returns:
            Dictionary containing:
                - 'position_3d': (batch, seq_len, 3) - predicted 3D positions
                - 'uncertainty': (batch, seq_len, 3) - predicted variance (if enabled)
                - 'hidden': hidden state for next forward pass
        """
        batch_size, seq_len, _ = positions_2d.shape

        # Project 2D inputs
        # (batch, seq_len, 2) -> (batch, seq_len, hidden_dim)
        input_features = self.input_projection(positions_2d)

        # Project context
        # (batch, context_dim) -> (batch, hidden_dim)
        context_features = self.context_projection(context)

        # Expand context to match sequence length
        # (batch, hidden_dim) -> (batch, seq_len, hidden_dim)
        context_expanded = context_features.unsqueeze(1).repeat(1, seq_len, 1)

        # LSTM forward pass
        lstm_out, hidden_out = self.lstm(input_features, hidden)
        # lstm_out: (batch, seq_len, hidden_dim)

        # Concatenate LSTM output with context
        combined = torch.cat([lstm_out, context_expanded], dim=-1)
        # combined: (batch, seq_len, hidden_dim * 2)

        # Predict 3D position
        position_3d = self.output_head(combined)
        # position_3d: (batch, seq_len, 3)

        result = {
            'position_3d': position_3d,
            'hidden': hidden_out
        }

        # Predict uncertainty if enabled
        if self.predict_uncertainty:
            uncertainty = self.uncertainty_head(combined)
            result['uncertainty'] = uncertainty

        return result

    def predict_sequence(
        self,
        positions_2d: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict 3D trajectory from 2D sequence.

        Args:
            positions_2d: (batch, seq_len, 2) - 2D positions
            context: (batch, context_dim) - context features

        Returns:
            positions_3d: (batch, seq_len, 3) - predicted 3D positions
            uncertainty: (batch, seq_len, 3) - uncertainty (if enabled)
        """
        with torch.no_grad():
            result = self.forward(positions_2d, context)
            positions_3d = result['position_3d']
            uncertainty = result.get('uncertainty', None)

        return positions_3d, uncertainty


class CanonicalRepresentation(nn.Module):
    """
    Canonical 3D representation layer for camera-independent predictions.

    Transforms camera-specific 2D observations into canonical 3D space
    using homography and camera calibration.
    """

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        """
        Initialize canonical representation.

        Args:
            pitch_length: Standard pitch length in meters
            pitch_width: Standard pitch width in meters
        """
        super().__init__()
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def apply_homography(
        self,
        points_2d: torch.Tensor,
        homography: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply homography transformation to 2D points.

        Args:
            points_2d: (batch, N, 2) - 2D pixel coordinates
            homography: (batch, 3, 3) - homography matrices

        Returns:
            points_world: (batch, N, 2) - world coordinates on pitch plane
        """
        batch_size, num_points, _ = points_2d.shape

        # Convert to homogeneous coordinates
        ones = torch.ones(batch_size, num_points, 1, device=points_2d.device)
        points_h = torch.cat([points_2d, ones], dim=-1)
        # points_h: (batch, N, 3)

        # Apply homography: H @ p
        # (batch, 3, 3) @ (batch, 3, N) -> (batch, 3, N)
        transformed_h = torch.bmm(homography, points_h.transpose(1, 2))

        # Convert back from homogeneous
        transformed_h = transformed_h.transpose(1, 2)
        # (batch, N, 3)

        points_world = transformed_h[:, :, :2] / (transformed_h[:, :, 2:3] + 1e-8)
        # (batch, N, 2)

        return points_world

    def normalize_to_canonical(self, points_world: torch.Tensor) -> torch.Tensor:
        """
        Normalize world coordinates to canonical [-1, 1] range.

        Args:
            points_world: (batch, N, 2) - world coordinates

        Returns:
            canonical: (batch, N, 2) - normalized coordinates
        """
        canonical = points_world.clone()
        canonical[:, :, 0] = (points_world[:, :, 0] / self.pitch_length) * 2 - 1
        canonical[:, :, 1] = (points_world[:, :, 1] / self.pitch_width) * 2 - 1
        return canonical

    def denormalize_from_canonical(
        self,
        canonical: torch.Tensor,
        include_z: bool = True
    ) -> torch.Tensor:
        """
        Denormalize from canonical space to world coordinates.

        Args:
            canonical: (batch, N, 2 or 3) - canonical coordinates
            include_z: Whether input has z coordinate

        Returns:
            world: (batch, N, 2 or 3) - world coordinates
        """
        world = canonical.clone()
        world[:, :, 0] = (canonical[:, :, 0] + 1) / 2 * self.pitch_length
        world[:, :, 1] = (canonical[:, :, 1] + 1) / 2 * self.pitch_width

        # Z is already in meters, no normalization needed
        # Just ensure it's non-negative if present
        if include_z and canonical.shape[-1] > 2:
            world[:, :, 2] = torch.clamp(canonical[:, :, 2], min=0.0)

        return world


class TrajectoryLoss(nn.Module):
    """Custom loss for trajectory prediction with physics constraints."""

    def __init__(
        self,
        position_weight: float = 1.0,
        velocity_weight: float = 0.5,
        physics_weight: float = 0.3,
        uncertainty_weight: float = 0.1
    ):
        """
        Initialize trajectory loss.

        Args:
            position_weight: Weight for position error
            velocity_weight: Weight for velocity consistency
            physics_weight: Weight for physics constraints
            uncertainty_weight: Weight for uncertainty calibration
        """
        super().__init__()
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.physics_weight = physics_weight
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        pred_positions: torch.Tensor,
        true_positions: torch.Tensor,
        pred_uncertainty: Optional[torch.Tensor] = None,
        dt: float = 0.04
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            pred_positions: (batch, seq_len, 3) - predicted 3D positions
            true_positions: (batch, seq_len, 3) - ground truth 3D positions
            pred_uncertainty: (batch, seq_len, 3) - predicted uncertainty
            dt: Time step between frames

        Returns:
            Dictionary with total loss and components
        """
        # Position loss (MSE)
        position_loss = nn.functional.mse_loss(pred_positions, true_positions)

        # Velocity consistency loss
        pred_velocities = (pred_positions[:, 1:] - pred_positions[:, :-1]) / dt
        true_velocities = (true_positions[:, 1:] - true_positions[:, :-1]) / dt
        velocity_loss = nn.functional.mse_loss(pred_velocities, true_velocities)

        # Physics constraint loss (vertical acceleration should be ~-9.81)
        if pred_positions.shape[1] > 2:
            pred_accel_z = (pred_velocities[:, 1:, 2] - pred_velocities[:, :-1, 2]) / dt
            gravity_target = torch.full_like(pred_accel_z, -9.81)
            physics_loss = nn.functional.mse_loss(pred_accel_z, gravity_target)
        else:
            physics_loss = torch.tensor(0.0, device=pred_positions.device)

        # Uncertainty calibration loss (negative log-likelihood)
        if pred_uncertainty is not None:
            # Negative log-likelihood assuming Gaussian
            nll = 0.5 * torch.log(pred_uncertainty + 1e-6) + \
                  0.5 * (pred_positions - true_positions)**2 / (pred_uncertainty + 1e-6)
            uncertainty_loss = nll.mean()
        else:
            uncertainty_loss = torch.tensor(0.0, device=pred_positions.device)

        # Total loss
        total_loss = (
            self.position_weight * position_loss +
            self.velocity_weight * velocity_loss +
            self.physics_weight * physics_loss +
            self.uncertainty_weight * uncertainty_loss
        )

        return {
            'total': total_loss,
            'position': position_loss,
            'velocity': velocity_loss,
            'physics': physics_loss,
            'uncertainty': uncertainty_loss
        }


def create_context_features(
    homography: np.ndarray,
    camera_height: float = 15.0,
    camera_angle: float = 30.0
) -> np.ndarray:
    """
    Create context feature vector from homography and camera parameters.

    Args:
        homography: (3, 3) homography matrix
        camera_height: Estimated camera height in meters
        camera_angle: Estimated camera tilt angle in degrees

    Returns:
        context: (11,) feature vector
    """
    # Flatten homography (9 values)
    h_flat = homography.flatten()

    # Add camera parameters (2 values)
    camera_params = np.array([
        camera_height / 20.0,  # Normalize to ~[0, 1]
        camera_angle / 90.0    # Normalize to [0, 1]
    ])

    context = np.concatenate([h_flat, camera_params])

    return context


def train_step(
    model: TrajectoryLSTM,
    optimizer: torch.optim.Optimizer,
    positions_2d: torch.Tensor,
    positions_3d: torch.Tensor,
    context: torch.Tensor,
    criterion: TrajectoryLoss,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Single training step.

    Args:
        model: TrajectoryLSTM model
        optimizer: Optimizer
        positions_2d: (batch, seq_len, 2) - input 2D positions
        positions_3d: (batch, seq_len, 3) - target 3D positions
        context: (batch, context_dim) - context features
        criterion: Loss function
        device: Device to use

    Returns:
        Dictionary with loss values
    """
    model.train()

    # Move to device
    positions_2d = positions_2d.to(device)
    positions_3d = positions_3d.to(device)
    context = context.to(device)

    # Forward pass
    optimizer.zero_grad()
    output = model(positions_2d, context)

    pred_positions = output['position_3d']
    pred_uncertainty = output.get('uncertainty', None)

    # Compute loss
    loss_dict = criterion(pred_positions, positions_3d, pred_uncertainty)

    # Backward pass
    loss_dict['total'].backward()
    optimizer.step()

    # Return losses as float
    return {k: v.item() for k, v in loss_dict.items()}


def evaluate(
    model: TrajectoryLSTM,
    dataloader: torch.utils.data.DataLoader,
    criterion: TrajectoryLoss,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate model on dataset.

    Args:
        model: TrajectoryLSTM model
        dataloader: DataLoader
        criterion: Loss function
        device: Device to use

    Returns:
        Dictionary with average loss values
    """
    model.eval()

    total_losses = {'total': 0, 'position': 0, 'velocity': 0, 'physics': 0, 'uncertainty': 0}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            positions_2d = batch['positions_2d'].to(device)
            positions_3d = batch['positions_3d'].to(device)
            context = batch['context'].to(device)

            output = model(positions_2d, context)
            pred_positions = output['position_3d']
            pred_uncertainty = output.get('uncertainty', None)

            loss_dict = criterion(pred_positions, positions_3d, pred_uncertainty)

            for k, v in loss_dict.items():
                total_losses[k] += v.item()
            num_batches += 1

    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    return avg_losses
