"""High-level trajectory prediction interface for off-screen player extrapolation.

This module provides a unified interface for predicting player positions,
combining transformer-based models with physics-based fallbacks.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from loguru import logger

from .baller2vec import Baller2Vec, create_feature_tensor, create_padding_mask
from .baller2vec_plus import Baller2VecPlus, create_team_tensor
from .motion_model import MultiPlayerMotionModel, MotionState


@dataclass
class PlayerState:
    """State information for a player."""
    player_id: int
    position: Tuple[float, float]  # (x, y) in meters
    velocity: Tuple[float, float]  # (vx, vy) in m/s
    team: int  # 0 or 1
    is_visible: bool  # True if on-screen, False if extrapolated
    confidence: float  # Prediction confidence [0, 1]
    uncertainty: Optional[Tuple[float, float]] = None  # Position uncertainty (σx, σy)


@dataclass
class PredictionResult:
    """Result of trajectory prediction."""
    timestamp: float
    players: List[PlayerState]
    method: str  # 'transformer', 'physics', or 'hybrid'


class TrajectoryPredictor:
    """High-level interface for player trajectory prediction.

    Combines transformer-based prediction (Baller2Vec++) with physics-based
    fallback for robust off-screen player extrapolation.
    """

    def __init__(
        self,
        model_type: str = 'baller2vec_plus',
        model_path: Optional[str] = None,
        history_length: int = 25,  # 1 second at 25fps
        confidence_threshold: float = 0.5,
        use_physics_fallback: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **model_kwargs
    ):
        """Initialize trajectory predictor.

        Args:
            model_type: 'baller2vec' or 'baller2vec_plus'
            model_path: Path to trained model weights (optional)
            history_length: Number of historical frames to maintain
            confidence_threshold: Threshold for using transformer vs physics fallback
            use_physics_fallback: Whether to use physics-based fallback
            device: Device for model inference
            **model_kwargs: Additional arguments for model initialization
        """
        self.model_type = model_type
        self.history_length = history_length
        self.confidence_threshold = confidence_threshold
        self.use_physics_fallback = use_physics_fallback
        self.device = device

        # Initialize model
        if model_type == 'baller2vec_plus':
            self.model = Baller2VecPlus(**model_kwargs).to(device)
        elif model_type == 'baller2vec':
            self.model = Baller2Vec(**model_kwargs).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load weights if provided
        if model_path is not None:
            self.load_weights(model_path)

        self.model.eval()

        # Initialize physics fallback
        self.physics_model = MultiPlayerMotionModel()

        # History buffers
        self.position_history: Dict[int, deque] = {}  # player_id -> deque of (timestamp, x, y)
        self.team_assignments: Dict[int, int] = {}  # player_id -> team (0 or 1)
        self.visibility_history: Dict[int, deque] = {}  # player_id -> deque of bool
        self.last_seen: Dict[int, float] = {}  # player_id -> timestamp

        logger.info(f"TrajectoryPredictor initialized with {model_type} on {device}")

    def load_weights(self, model_path: str):
        """Load model weights from file.

        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        logger.info(f"Loaded model weights from {model_path}")

    def save_weights(self, model_path: str):
        """Save model weights to file.

        Args:
            model_path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type
        }, model_path)
        logger.info(f"Saved model weights to {model_path}")

    def update_history(self, player_states: List[PlayerState], timestamp: float):
        """Update history with new observations.

        Args:
            player_states: List of observed player states
            timestamp: Current timestamp
        """
        for state in player_states:
            player_id = state.player_id

            # Initialize history buffers if needed
            if player_id not in self.position_history:
                self.position_history[player_id] = deque(maxlen=self.history_length)
                self.visibility_history[player_id] = deque(maxlen=self.history_length)

            # Update history
            self.position_history[player_id].append((timestamp, state.position[0], state.position[1]))
            self.visibility_history[player_id].append(state.is_visible)
            self.team_assignments[player_id] = state.team

            # Update physics model if visible
            if state.is_visible:
                self.physics_model.update(
                    player_id,
                    np.array(state.position),
                    timestamp
                )
                self.last_seen[player_id] = timestamp

    def predict(
        self,
        visible_players: List[PlayerState],
        timestamp: float,
        n_future_steps: int = 1,
        predict_all_players: bool = True
    ) -> PredictionResult:
        """Predict player positions including off-screen extrapolation.

        Args:
            visible_players: List of currently visible players
            timestamp: Current timestamp
            n_future_steps: Number of future steps to predict
            predict_all_players: If True, predict all 22 players; if False, only visible ones

        Returns:
            PredictionResult with predictions for all players
        """
        # Update history with visible players
        self.update_history(visible_players, timestamp)

        # Determine which players to predict
        if predict_all_players:
            # Predict all players with sufficient history
            player_ids_to_predict = [
                pid for pid, hist in self.position_history.items()
                if len(hist) >= min(5, self.history_length // 4)
            ]
        else:
            player_ids_to_predict = [p.player_id for p in visible_players]

        if len(player_ids_to_predict) == 0:
            return PredictionResult(timestamp, [], 'none')

        # Try transformer prediction
        try:
            predictions, confidences = self._predict_transformer(
                player_ids_to_predict,
                timestamp,
                n_future_steps
            )
            method = 'transformer'
        except Exception as e:
            logger.warning(f"Transformer prediction failed: {e}, using physics fallback")
            predictions = None
            confidences = None

        # Use physics fallback for low-confidence or failed predictions
        if self.use_physics_fallback and (predictions is None or
                                          any(c < self.confidence_threshold for c in confidences.values())):
            predictions, confidences = self._predict_hybrid(
                player_ids_to_predict,
                timestamp,
                n_future_steps,
                predictions,
                confidences
            )
            method = 'hybrid'

        # Build result
        player_states = []
        visible_ids = {p.player_id for p in visible_players}

        for player_id in player_ids_to_predict:
            if player_id not in predictions:
                continue

            position = predictions[player_id]
            confidence = confidences.get(player_id, 0.0)
            is_visible = player_id in visible_ids

            # Compute velocity from history
            velocity = self._get_velocity(player_id)

            player_states.append(PlayerState(
                player_id=player_id,
                position=(float(position[0]), float(position[1])),
                velocity=velocity,
                team=self.team_assignments.get(player_id, 0),
                is_visible=is_visible,
                confidence=confidence
            ))

        return PredictionResult(timestamp, player_states, method)

    def _predict_transformer(
        self,
        player_ids: List[int],
        timestamp: float,
        n_future_steps: int
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """Predict using transformer model.

        Args:
            player_ids: List of player IDs to predict
            timestamp: Current timestamp
            n_future_steps: Number of future steps

        Returns:
            Tuple of (predictions, confidences)
        """
        # Prepare input data
        positions, teams, velocities, player_id_mapping = self._prepare_transformer_input(player_ids)

        if positions is None or len(positions) < min(5, self.history_length // 4):
            raise ValueError("Insufficient history for transformer prediction")

        # Create feature tensor
        features = create_feature_tensor(positions, teams, velocities)
        features = features.unsqueeze(0).to(self.device)  # Add batch dimension

        # Create team tensor
        teams_tensor = create_team_tensor(teams[np.newaxis, :]).to(self.device)

        # Create player IDs tensor
        player_ids_tensor = torch.arange(len(player_ids), device=self.device).unsqueeze(0)

        # Predict
        with torch.no_grad():
            if self.model_type == 'baller2vec_plus':
                pred, uncertainty = self.model.predict_future(
                    features,
                    teams_tensor,
                    n_future_steps,
                    player_ids=player_ids_tensor,
                    return_uncertainty=True
                )
            else:
                pred = self.model.predict_future(
                    features,
                    n_future_steps,
                    player_ids=player_ids_tensor,
                    autoregressive=True
                )
                uncertainty = None

        # Extract predictions (take last predicted step)
        pred_np = pred[0, -1, :, :].cpu().numpy()  # (num_players, 2)

        # Calculate confidence from uncertainty
        confidences = {}
        predictions = {}

        for i, player_id in enumerate(player_ids):
            predictions[player_id] = pred_np[i]

            if uncertainty is not None:
                unc_np = uncertainty[0, -1, i, :].cpu().numpy()
                # Convert uncertainty to confidence (inverse relationship)
                avg_unc = float(np.mean(unc_np))
                confidences[player_id] = float(np.exp(-avg_unc))
            else:
                # Default confidence based on visibility
                confidences[player_id] = 0.8

        return predictions, confidences

    def _predict_hybrid(
        self,
        player_ids: List[int],
        timestamp: float,
        n_future_steps: int,
        transformer_predictions: Optional[Dict[int, np.ndarray]],
        transformer_confidences: Optional[Dict[int, float]]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """Combine transformer and physics predictions.

        Args:
            player_ids: List of player IDs
            timestamp: Current timestamp
            n_future_steps: Number of future steps
            transformer_predictions: Predictions from transformer (may be None)
            transformer_confidences: Confidences from transformer (may be None)

        Returns:
            Tuple of (predictions, confidences)
        """
        predictions = {}
        confidences = {}

        for player_id in player_ids:
            # Get physics prediction
            physics_preds = self.physics_model.extrapolate(player_id, n_future_steps)

            if physics_preds is None or len(physics_preds) == 0:
                # No physics prediction, use transformer if available
                if transformer_predictions and player_id in transformer_predictions:
                    predictions[player_id] = transformer_predictions[player_id]
                    confidences[player_id] = transformer_confidences.get(player_id, 0.5)
                continue

            physics_pred = physics_preds[-1].position
            physics_conf = physics_preds[-1].confidence

            # Check if transformer prediction is available and confident
            if (transformer_predictions and player_id in transformer_predictions and
                transformer_confidences and transformer_confidences.get(player_id, 0) >= self.confidence_threshold):
                # Blend transformer and physics
                trans_pred = transformer_predictions[player_id]
                trans_conf = transformer_confidences[player_id]

                # Weighted average based on confidence
                total_conf = trans_conf + physics_conf
                weight_trans = trans_conf / total_conf
                weight_phys = physics_conf / total_conf

                blended_pred = weight_trans * trans_pred + weight_phys * physics_pred
                predictions[player_id] = blended_pred
                confidences[player_id] = float(max(trans_conf, physics_conf))
            else:
                # Use physics prediction
                predictions[player_id] = physics_pred
                confidences[player_id] = float(physics_conf)

        return predictions, confidences

    def _prepare_transformer_input(
        self,
        player_ids: List[int]
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, Dict[int, int]]:
        """Prepare input data for transformer.

        Args:
            player_ids: List of player IDs

        Returns:
            Tuple of (positions, teams, velocities, player_id_mapping)
        """
        # Find common history length
        min_history = min(len(self.position_history[pid]) for pid in player_ids)

        if min_history == 0:
            return None, None, None, None

        # Build position array
        num_players = len(player_ids)
        positions = np.zeros((min_history, num_players, 2))
        teams = np.zeros(num_players, dtype=int)
        player_id_mapping = {}

        for i, player_id in enumerate(player_ids):
            history = list(self.position_history[player_id])[-min_history:]
            for t, (ts, x, y) in enumerate(history):
                positions[t, i, 0] = x
                positions[t, i, 1] = y

            teams[i] = self.team_assignments.get(player_id, 0)
            player_id_mapping[i] = player_id

        # Compute velocities
        velocities = np.zeros_like(positions)
        velocities[1:] = positions[1:] - positions[:-1]

        return positions, teams, velocities, player_id_mapping

    def _get_velocity(self, player_id: int) -> Tuple[float, float]:
        """Get current velocity estimate for a player.

        Args:
            player_id: Player ID

        Returns:
            Velocity (vx, vy) in m/s
        """
        if player_id not in self.position_history or len(self.position_history[player_id]) < 2:
            return (0.0, 0.0)

        history = list(self.position_history[player_id])
        t1, x1, y1 = history[-1]
        t2, x2, y2 = history[-2]

        dt = t1 - t2
        if dt > 0:
            vx = (x1 - x2) / dt
            vy = (y1 - y2) / dt
            return (float(vx), float(vy))

        return (0.0, 0.0)

    def get_extrapolation_confidence(self, player_id: int) -> float:
        """Get confidence score for a player's extrapolation.

        Args:
            player_id: Player ID

        Returns:
            Confidence score [0, 1]
        """
        if player_id not in self.position_history:
            return 0.0

        # Factors affecting confidence:
        # 1. History length
        history_len = len(self.position_history[player_id])
        history_factor = min(1.0, history_len / self.history_length)

        # 2. Visibility ratio
        if player_id in self.visibility_history:
            visibility_ratio = sum(self.visibility_history[player_id]) / len(self.visibility_history[player_id])
        else:
            visibility_ratio = 0.0

        # 3. Time since last seen
        if player_id in self.last_seen:
            # Confidence degrades with time
            # Full confidence for 1 second, then exponential decay
            current_time = max(ts for ts, _, _ in self.position_history[player_id])
            time_since = current_time - self.last_seen[player_id]
            time_factor = np.exp(-time_since / 2.0)  # Half confidence after 2 seconds
        else:
            time_factor = 0.0

        # Combine factors
        confidence = history_factor * 0.3 + visibility_ratio * 0.3 + time_factor * 0.4

        return float(confidence)

    def reset(self):
        """Reset predictor state."""
        self.position_history.clear()
        self.team_assignments.clear()
        self.visibility_history.clear()
        self.last_seen.clear()
        self.physics_model.reset()
        logger.info("TrajectoryPredictor reset")
