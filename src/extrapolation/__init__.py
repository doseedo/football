"""Off-screen player extrapolation module.

This module provides transformer-based and physics-based models for predicting
player positions when they move off-camera, matching SkillCorner's extrapolation
capability.

Main components:
- Baller2Vec: Base multi-entity transformer for trajectory prediction
- Baller2VecPlus: Enhanced transformer with look-ahead and team coordination
- TrajectoryPredictor: High-level interface combining models with fallbacks
- Motion models: Physics-based Kalman filter fallback

Example usage:
    >>> from extrapolation import TrajectoryPredictor, PlayerState
    >>>
    >>> # Initialize predictor
    >>> predictor = TrajectoryPredictor(
    ...     model_type='baller2vec_plus',
    ...     model_path='models/baller2vec_plus.pth'
    ... )
    >>>
    >>> # Update with visible players
    >>> visible_players = [
    ...     PlayerState(player_id=1, position=(45.2, 23.1), velocity=(2.0, 0.5),
    ...                team=0, is_visible=True, confidence=1.0),
    ...     # ... more players
    ... ]
    >>> predictor.update_history(visible_players, timestamp=0.04)
    >>>
    >>> # Predict all 22 players (including off-screen)
    >>> result = predictor.predict(
    ...     visible_players=visible_players,
    ...     timestamp=0.04,
    ...     predict_all_players=True
    ... )
    >>>
    >>> # Check predictions
    >>> for player in result.players:
    ...     if not player.is_visible:
    ...         print(f"Player {player.player_id} extrapolated at {player.position}")
"""

# Base models
from .baller2vec import (
    Baller2Vec,
    PositionalEncoding,
    PlayerEmbedding,
    FeatureEncoder,
    MultiEntityTransformerLayer,
    create_feature_tensor,
    create_padding_mask
)

# Enhanced model
from .baller2vec_plus import (
    Baller2VecPlus,
    CoordinatedAttentionLayer,
    LookAheadEncoder,
    create_team_tensor
)

# High-level interface
from .trajectory_predictor import (
    TrajectoryPredictor,
    PlayerState,
    PredictionResult
)

# Physics-based models
from .motion_model import (
    KalmanMotionModel,
    MultiPlayerMotionModel,
    MotionState
)

__all__ = [
    # Base transformer
    'Baller2Vec',
    'PositionalEncoding',
    'PlayerEmbedding',
    'FeatureEncoder',
    'MultiEntityTransformerLayer',
    'create_feature_tensor',
    'create_padding_mask',

    # Enhanced transformer
    'Baller2VecPlus',
    'CoordinatedAttentionLayer',
    'LookAheadEncoder',
    'create_team_tensor',

    # High-level interface
    'TrajectoryPredictor',
    'PlayerState',
    'PredictionResult',

    # Physics models
    'KalmanMotionModel',
    'MultiPlayerMotionModel',
    'MotionState',
]

__version__ = '0.1.0'
__author__ = 'Football Tracking System'
__description__ = 'Transformer-based off-screen player extrapolation'
