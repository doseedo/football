"""Identity modules for jersey OCR and player assignment.

This package provides a complete player re-identification system including:
- OSNet backbone for appearance embeddings
- Jersey number detection and recognition
- Unsupervised team classification
- Player identity fusion and tracking
"""

from .osnet import (
    OSNet,
    OSNetAIN,
    ReIDExtractor,
    build_osnet
)

from .jersey_detector import (
    JerseyNumberDetector,
    JerseyDetector,
    visualize_detection
)

from .jersey_recognizer import (
    CRNN,
    JerseyRecognizer,
    TemporalNumberAggregator,
    CTCDecoder
)

from .contrastive_team import (
    TeamClassifier,
    OnlineTeamClassifier,
    ColorFeatureExtractor,
    visualize_teams
)

from .player_identifier import (
    PlayerIdentifier,
    PlayerIdentity,
    PlayerGallery,
    create_player_identifier,
    visualize_identity
)

__all__ = [
    # OSNet
    'OSNet',
    'OSNetAIN',
    'ReIDExtractor',
    'build_osnet',

    # Jersey Detection
    'JerseyNumberDetector',
    'JerseyDetector',
    'visualize_detection',

    # Jersey Recognition
    'CRNN',
    'JerseyRecognizer',
    'TemporalNumberAggregator',
    'CTCDecoder',

    # Team Classification
    'TeamClassifier',
    'OnlineTeamClassifier',
    'ColorFeatureExtractor',
    'visualize_teams',

    # Player Identification
    'PlayerIdentifier',
    'PlayerIdentity',
    'PlayerGallery',
    'create_player_identifier',
    'visualize_identity',
]
