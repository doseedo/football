"""Baller2Vec: Multi-entity Transformer for trajectory prediction.

Based on the paper "Baller2Vec: A Multi-Entity Transformer for Multi-Agent Trajectory Forecasting"
This implements a transformer that attends across both players AND time to predict future trajectories.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal information."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor of shape (seq_len, batch, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0)]


class PlayerEmbedding(nn.Module):
    """Learnable embeddings for player identity."""

    def __init__(self, num_players: int, d_model: int):
        """Initialize player embeddings.

        Args:
            num_players: Maximum number of players (typically 22)
            d_model: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(num_players, d_model)

    def forward(self, player_ids: torch.Tensor) -> torch.Tensor:
        """Get player embeddings.

        Args:
            player_ids: Tensor of player IDs (batch, num_players)

        Returns:
            Player embeddings (batch, num_players, d_model)
        """
        return self.embedding(player_ids)


class FeatureEncoder(nn.Module):
    """Encode player features (position, velocity, team) into embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        """Initialize feature encoder.

        Args:
            d_model: Model dimension
            dropout: Dropout probability
        """
        super().__init__()
        # Input: [x, y, vx, vy, team_onehot(2)]
        self.input_dim = 6
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode features.

        Args:
            features: Input features (batch, seq_len, num_players, feature_dim)

        Returns:
            Encoded features (batch, seq_len, num_players, d_model)
        """
        return self.encoder(features)


class MultiEntityTransformerLayer(nn.Module):
    """Transformer layer that attends across both entities (players) and time."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        """Initialize transformer layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
        """
        super().__init__()

        # Self-attention for temporal dimension
        self.temporal_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Self-attention for entity (player) dimension
        self.entity_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        entity_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, num_players, d_model)
            temporal_mask: Mask for temporal attention
            entity_mask: Mask for entity attention (to handle variable num players)

        Returns:
            Output tensor (batch, seq_len, num_players, d_model)
        """
        batch_size, seq_len, num_players, d_model = x.shape

        # Temporal attention: attend across time for each player
        # Reshape to (batch * num_players, seq_len, d_model)
        x_temporal = x.permute(0, 2, 1, 3).reshape(batch_size * num_players, seq_len, d_model)
        attn_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal, attn_mask=temporal_mask)
        attn_out = attn_out.reshape(batch_size, num_players, seq_len, d_model).permute(0, 2, 1, 3)
        x = self.norm1(x + self.dropout(attn_out))

        # Entity attention: attend across players at each time step
        # Reshape to (batch * seq_len, num_players, d_model)
        x_entity = x.reshape(batch_size * seq_len, num_players, d_model)
        entity_mask_expanded = entity_mask.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, num_players) if entity_mask is not None else None
        attn_out, _ = self.entity_attn(x_entity, x_entity, x_entity, key_padding_mask=entity_mask_expanded)
        attn_out = attn_out.reshape(batch_size, seq_len, num_players, d_model)
        x = self.norm2(x + self.dropout(attn_out))

        # Feedforward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


class Baller2Vec(nn.Module):
    """Baller2Vec multi-entity transformer for trajectory prediction.

    Predicts future player positions given historical trajectories.
    Uses attention across both time and players to model interactions.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_players: int = 22,
        use_player_embeddings: bool = False
    ):
        """Initialize Baller2Vec.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_players: Maximum number of players
            use_player_embeddings: Whether to use learnable player embeddings
        """
        super().__init__()

        self.d_model = d_model
        self.max_players = max_players
        self.use_player_embeddings = use_player_embeddings

        # Feature encoder
        self.feature_encoder = FeatureEncoder(d_model, dropout)

        # Positional encoding for temporal information
        self.pos_encoder = PositionalEncoding(d_model)

        # Optional player embeddings
        if use_player_embeddings:
            self.player_embedding = PlayerEmbedding(max_players, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            MultiEntityTransformerLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output head: predict (x, y) positions
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # x, y coordinates
        )

    def forward(
        self,
        features: torch.Tensor,
        player_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Input features (batch, seq_len, num_players, feature_dim)
                     Features: [x, y, vx, vy, team_onehot(2)]
            player_ids: Player IDs (batch, num_players) if using embeddings
            mask: Padding mask (batch, num_players) - True for padded positions

        Returns:
            Predicted positions (batch, seq_len, num_players, 2)
        """
        batch_size, seq_len, num_players, _ = features.shape

        # Encode features
        x = self.feature_encoder(features)  # (batch, seq_len, num_players, d_model)

        # Add player embeddings if enabled
        if self.use_player_embeddings and player_ids is not None:
            player_emb = self.player_embedding(player_ids)  # (batch, num_players, d_model)
            player_emb = player_emb.unsqueeze(1).expand(-1, seq_len, -1, -1)
            x = x + player_emb

        # Add positional encoding (temporal)
        # Reshape to (seq_len, batch * num_players, d_model)
        x_pos = x.permute(1, 0, 2, 3).reshape(seq_len, batch_size * num_players, self.d_model)
        x_pos = self.pos_encoder(x_pos)
        x = x_pos.reshape(seq_len, batch_size, num_players, self.d_model).permute(1, 0, 2, 3)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, entity_mask=mask)

        # Output prediction
        predictions = self.output_head(x)  # (batch, seq_len, num_players, 2)

        return predictions

    def predict_future(
        self,
        past_features: torch.Tensor,
        n_future_steps: int,
        player_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        autoregressive: bool = True
    ) -> torch.Tensor:
        """Predict future trajectories.

        Args:
            past_features: Historical features (batch, past_len, num_players, feature_dim)
            n_future_steps: Number of future steps to predict
            player_ids: Player IDs if using embeddings
            mask: Padding mask
            autoregressive: If True, feed predictions back as input

        Returns:
            Future predictions (batch, n_future_steps, num_players, 2)
        """
        batch_size, past_len, num_players, feature_dim = past_features.shape

        if not autoregressive:
            # Non-autoregressive: predict all future steps at once
            # Append future placeholders
            future_features = torch.zeros(
                batch_size, n_future_steps, num_players, feature_dim,
                device=past_features.device, dtype=past_features.dtype
            )
            all_features = torch.cat([past_features, future_features], dim=1)
            all_predictions = self.forward(all_features, player_ids, mask)
            return all_predictions[:, past_len:, :, :]

        else:
            # Autoregressive: predict one step at a time
            predictions = []
            current_features = past_features

            for _ in range(n_future_steps):
                # Predict next step
                next_pred = self.forward(current_features, player_ids, mask)
                next_positions = next_pred[:, -1:, :, :]  # (batch, 1, num_players, 2)
                predictions.append(next_positions)

                # Prepare features for next iteration
                # Compute velocities from predictions
                if current_features.shape[1] >= 2:
                    prev_positions = next_pred[:, -2:-1, :, :]
                    velocities = next_positions - prev_positions
                else:
                    velocities = torch.zeros_like(next_positions)

                # Extract team information from last features
                team_info = current_features[:, -1:, :, 4:6]

                # Create next feature vector
                next_features = torch.cat([
                    next_positions,  # x, y
                    velocities,      # vx, vy
                    team_info        # team onehot
                ], dim=-1)

                # Append to sequence
                current_features = torch.cat([current_features, next_features], dim=1)

            return torch.cat(predictions, dim=1)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute prediction loss.

        Args:
            predictions: Predicted positions (batch, seq_len, num_players, 2)
            targets: Ground truth positions (batch, seq_len, num_players, 2)
            mask: Padding mask (batch, num_players)

        Returns:
            Loss value
        """
        # L2 loss
        loss = F.mse_loss(predictions, targets, reduction='none')  # (batch, seq_len, num_players, 2)

        # Apply mask
        if mask is not None:
            # Expand mask to match prediction shape
            mask_expanded = (~mask).unsqueeze(1).unsqueeze(-1).float()  # (batch, 1, num_players, 1)
            loss = loss * mask_expanded

        return loss.mean()


def create_feature_tensor(
    positions: np.ndarray,
    teams: np.ndarray,
    velocities: Optional[np.ndarray] = None
) -> torch.Tensor:
    """Create feature tensor from position data.

    Args:
        positions: Player positions (seq_len, num_players, 2)
        teams: Team IDs (num_players,) - 0 or 1
        velocities: Optional velocities (seq_len, num_players, 2)

    Returns:
        Feature tensor (seq_len, num_players, 6)
    """
    seq_len, num_players, _ = positions.shape

    # Compute velocities if not provided
    if velocities is None:
        velocities = np.zeros_like(positions)
        velocities[1:] = positions[1:] - positions[:-1]

    # Create team one-hot encoding
    team_onehot = np.zeros((num_players, 2))
    team_onehot[np.arange(num_players), teams] = 1
    team_onehot = np.tile(team_onehot[np.newaxis, :, :], (seq_len, 1, 1))

    # Concatenate features
    features = np.concatenate([
        positions,     # x, y
        velocities,    # vx, vy
        team_onehot    # team (one-hot)
    ], axis=-1)

    return torch.from_numpy(features).float()


def create_padding_mask(num_players_per_sample: np.ndarray, max_players: int) -> torch.Tensor:
    """Create padding mask for variable number of players.

    Args:
        num_players_per_sample: Number of actual players in each sample (batch_size,)
        max_players: Maximum number of players

    Returns:
        Padding mask (batch_size, max_players) - True for padded positions
    """
    batch_size = len(num_players_per_sample)
    mask = torch.zeros(batch_size, max_players, dtype=torch.bool)

    for i, n in enumerate(num_players_per_sample):
        if n < max_players:
            mask[i, n:] = True

    return mask
