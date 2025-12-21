"""Detection modules for players and ball."""

from .player_detector import PlayerDetector, Detection
from .ball_detector import BallDetector, BallDetection
from .team_classifier import TeamClassifier, Team, TeamClassification
from .detr_detector import DETRDetector, DETR, HungarianMatcher
from .transformer_head import (
    TransformerPredictionHead,
    MultiScaleTransformerHead,
    YOLOWithTransformerHead,
    SpatialRelationReasoning,
    apply_transformer_reasoning
)
from .hybrid_detector import (
    HybridDetector,
    WeightedBoxFusion,
    SceneComplexity,
    EnsembleConfig
)

__all__ = [
    # Original detectors
    "PlayerDetector",
    "Detection",
    "BallDetector",
    "BallDetection",
    "TeamClassifier",
    "Team",
    "TeamClassification",
    # DETR detector
    "DETRDetector",
    "DETR",
    "HungarianMatcher",
    # Transformer heads
    "TransformerPredictionHead",
    "MultiScaleTransformerHead",
    "YOLOWithTransformerHead",
    "SpatialRelationReasoning",
    "apply_transformer_reasoning",
    # Hybrid detector
    "HybridDetector",
    "WeightedBoxFusion",
    "SceneComplexity",
    "EnsembleConfig",
]
