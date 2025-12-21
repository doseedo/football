"""Baller2Vec++: Enhanced multi-entity transformer with look-ahead.

Extends Baller2Vec with:
- Look-ahead trajectory sequences for better long-term prediction
- Coordinated agent modeling with special attention masks
- Better handling of statistically dependent trajectories (team coordination)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .baller2vec import (
    PositionalEncoding,
    FeatureEncoder,
    PlayerEmbedding
)


class CoordinatedAttentionLayer(nn.Module):
    """Attention layer with special masking for coordinated agents (teammates)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """Initialize coordinated attention layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # Multi-head attention
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Team-specific attention (separate attention for teammates vs opponents)
        self.team_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Gating mechanism to combine global and team attention
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        team_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, num_players, d_model)
            team_mask: Team coordination mask (batch, num_players, num_players)
                      True where players are on same team
            padding_mask: Padding mask (batch, num_players)

        Returns:
            Output tensor (batch, num_players, d_model)
        """
        # Global attention (all players)
        global_attn, _ = self.attn(x, x, x, key_padding_mask=padding_mask)

        # Team-specific attention
        if team_mask is not None:
            # Create attention mask that prioritizes teammates
            # Convert team_mask to attention bias
            attn_bias = torch.zeros_like(team_mask, dtype=torch.float)
            attn_bias[~team_mask] = float('-inf')  # Mask out opponents
            team_attn, _ = self.team_attn(x, x, x, attn_mask=attn_bias)
        else:
            team_attn = global_attn

        # Gating mechanism to combine global and team attention
        combined = torch.cat([global_attn, team_attn], dim=-1)
        gate_value = self.gate(combined)
        output = gate_value * global_attn + (1 - gate_value) * team_attn

        return self.norm(x + self.dropout(output))


class LookAheadEncoder(nn.Module):
    """Encoder that processes look-ahead trajectory sequences."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize look-ahead encoder.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            dropout: Dropout probability
        """
        super().__init__()

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode look-ahead sequences.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask

        Returns:
            Encoded tensor (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)

        return self.norm(x)


class Baller2VecPlus(nn.Module):
    """Baller2Vec++ with look-ahead and coordinated agent modeling.

    Improvements over base Baller2Vec:
    1. Look-ahead trajectory encoding for better long-term prediction
    2. Special attention mechanism for coordinated agents (teammates)
    3. Multi-scale temporal modeling
    4. Uncertainty estimation
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_players: int = 22,
        use_player_embeddings: bool = True,
        lookahead_steps: int = 5
    ):
        """Initialize Baller2Vec++.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_players: Maximum number of players
            use_player_embeddings: Whether to use learnable player embeddings
            lookahead_steps: Number of look-ahead steps to encode
        """
        super().__init__()

        self.d_model = d_model
        self.max_players = max_players
        self.use_player_embeddings = use_player_embeddings
        self.lookahead_steps = lookahead_steps

        # Feature encoder
        self.feature_encoder = FeatureEncoder(d_model, dropout)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Player embeddings
        if use_player_embeddings:
            self.player_embedding = PlayerEmbedding(max_players, d_model)

        # Look-ahead encoder
        self.lookahead_encoder = LookAheadEncoder(d_model, num_heads, num_layers=2, dropout=dropout)

        # Main transformer layers with coordinated attention
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers // 2)
        ])

        self.entity_layers = nn.ModuleList([
            CoordinatedAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # Multi-scale temporal convolutions
        self.temporal_conv = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])

        # Output heads
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # x, y
        )

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),  # uncertainty for x, y
            nn.Softplus()  # Ensure positive uncertainty
        )

    def _create_team_mask(self, teams: torch.Tensor) -> torch.Tensor:
        """Create team coordination mask.

        Args:
            teams: Team IDs (batch, num_players)

        Returns:
            Team mask (batch, num_players, num_players)
            True where players are on same team
        """
        batch_size, num_players = teams.shape
        teams_expanded1 = teams.unsqueeze(2).expand(-1, -1, num_players)
        teams_expanded2 = teams.unsqueeze(1).expand(-1, num_players, -1)
        return teams_expanded1 == teams_expanded2

    def forward(
        self,
        features: torch.Tensor,
        teams: torch.Tensor,
        player_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            features: Input features (batch, seq_len, num_players, feature_dim)
            teams: Team IDs (batch, num_players) - 0 or 1
            player_ids: Player IDs (batch, num_players) if using embeddings
            mask: Padding mask (batch, num_players)
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Tuple of (predictions, uncertainty) where:
                predictions: (batch, seq_len, num_players, 2)
                uncertainty: (batch, seq_len, num_players, 2) if return_uncertainty else None
        """
        batch_size, seq_len, num_players, _ = features.shape

        # Encode features
        x = self.feature_encoder(features)  # (batch, seq_len, num_players, d_model)

        # Add player embeddings
        if self.use_player_embeddings and player_ids is not None:
            player_emb = self.player_embedding(player_ids)
            player_emb = player_emb.unsqueeze(1).expand(-1, seq_len, -1, -1)
            x = x + player_emb

        # Add positional encoding
        x_pos = x.permute(1, 0, 2, 3).reshape(seq_len, batch_size * num_players, self.d_model)
        x_pos = self.pos_encoder(x_pos)
        x = x_pos.reshape(seq_len, batch_size, num_players, self.d_model).permute(1, 0, 2, 3)

        # Multi-scale temporal modeling
        # Reshape to (batch * num_players, seq_len, d_model)
        x_temporal = x.permute(0, 2, 1, 3).reshape(batch_size * num_players, seq_len, self.d_model)

        # Apply temporal layers
        for layer in self.temporal_layers:
            x_temporal = layer(x_temporal)

        # Multi-scale convolutions
        x_conv = x_temporal.permute(0, 2, 1)  # (batch * num_players, d_model, seq_len)
        conv_outputs = [conv(x_conv) for conv in self.temporal_conv]
        x_multiscale = torch.stack(conv_outputs, dim=0).mean(dim=0)  # Average pooling
        x_temporal = x_temporal + x_multiscale.permute(0, 2, 1)

        # Reshape back
        x = x_temporal.reshape(batch_size, num_players, seq_len, self.d_model).permute(0, 2, 1, 3)

        # Entity attention with team coordination
        team_mask = self._create_team_mask(teams)

        for i, (entity_layer, norm) in enumerate(zip(self.entity_layers, self.layer_norms)):
            # Apply entity attention at each time step
            x_entity_list = []
            for t in range(seq_len):
                x_t = x[:, t, :, :]  # (batch, num_players, d_model)
                x_t = entity_layer(x_t, team_mask=team_mask, padding_mask=mask)
                x_entity_list.append(x_t)

            x_entity = torch.stack(x_entity_list, dim=1)
            x = norm(x + x_entity)

        # Output predictions
        predictions = self.position_head(x)  # (batch, seq_len, num_players, 2)

        uncertainty = None
        if return_uncertainty:
            uncertainty = self.uncertainty_head(x)  # (batch, seq_len, num_players, 2)

        return predictions, uncertainty

    def predict_future(
        self,
        past_features: torch.Tensor,
        teams: torch.Tensor,
        n_future_steps: int,
        player_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict future trajectories with look-ahead.

        Args:
            past_features: Historical features (batch, past_len, num_players, feature_dim)
            teams: Team IDs (batch, num_players)
            n_future_steps: Number of future steps to predict
            player_ids: Player IDs if using embeddings
            mask: Padding mask
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Tuple of (predictions, uncertainty)
        """
        batch_size, past_len, num_players, feature_dim = past_features.shape

        # Use autoregressive prediction with look-ahead
        predictions = []
        uncertainties = [] if return_uncertainty else None
        current_features = past_features

        for step in range(n_future_steps):
            # Predict next step
            pred, unc = self.forward(
                current_features,
                teams,
                player_ids,
                mask,
                return_uncertainty=return_uncertainty
            )

            next_positions = pred[:, -1:, :, :]  # (batch, 1, num_players, 2)
            predictions.append(next_positions)

            if return_uncertainty:
                uncertainties.append(unc[:, -1:, :, :])

            # Prepare features for next iteration
            if current_features.shape[1] >= 2:
                prev_positions = pred[:, -2:-1, :, :]
                velocities = next_positions - prev_positions
            else:
                velocities = torch.zeros_like(next_positions)

            # Extract team information
            team_onehot = current_features[:, -1:, :, 4:6]

            # Create next feature vector
            next_features = torch.cat([
                next_positions,
                velocities,
                team_onehot
            ], dim=-1)

            # Sliding window: keep last N steps for context
            window_size = min(past_len, 20)  # Keep last 20 steps
            current_features = torch.cat([current_features[:, -window_size+1:], next_features], dim=1)

        pred_tensor = torch.cat(predictions, dim=1)
        unc_tensor = torch.cat(uncertainties, dim=1) if return_uncertainty else None

        return pred_tensor, unc_tensor

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute prediction loss with uncertainty weighting.

        Args:
            predictions: Predicted positions (batch, seq_len, num_players, 2)
            targets: Ground truth positions (batch, seq_len, num_players, 2)
            uncertainty: Predicted uncertainty (batch, seq_len, num_players, 2)
            mask: Padding mask (batch, num_players)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Position loss
        if uncertainty is not None:
            # Negative log-likelihood with Gaussian assumption
            # NLL = 0.5 * log(2π * σ²) + (y - μ)² / (2σ²)
            mse = (predictions - targets) ** 2
            nll = 0.5 * torch.log(2 * np.pi * uncertainty) + mse / (2 * uncertainty)
            position_loss = nll.mean()
        else:
            position_loss = F.mse_loss(predictions, targets, reduction='none')

        # Apply mask
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(1).unsqueeze(-1).float()
            position_loss = (position_loss * mask_expanded).sum() / mask_expanded.sum()
        else:
            position_loss = position_loss.mean()

        # Velocity consistency loss (smoothness)
        pred_velocities = predictions[:, 1:] - predictions[:, :-1]
        target_velocities = targets[:, 1:] - targets[:, :-1]
        velocity_loss = F.mse_loss(pred_velocities, target_velocities)

        # Total loss
        total_loss = position_loss + 0.1 * velocity_loss

        loss_dict = {
            'position_loss': position_loss.item(),
            'velocity_loss': velocity_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


def create_team_tensor(teams: np.ndarray) -> torch.Tensor:
    """Create team tensor.

    Args:
        teams: Team IDs (batch, num_players) - values should be 0 or 1

    Returns:
        Team tensor
    """
    return torch.from_numpy(teams).long()
