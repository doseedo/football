"""
Decision Engine for Football Analytics.

This module analyzes game state to identify:
- Defensive gaps and weaknesses
- Ball progression opportunities with success probability
- Risk/reward scoring for each action

Usage:
    from src.decision_engine import DecisionEngine, PlayerState

    # Create player states
    players = [
        PlayerState(player_id=1, team=0, position=(10, 5)),
        PlayerState(player_id=2, team=0, position=(20, -10)),
        # ... more players
    ]

    # Separate by team
    team_0 = [p for p in players if p.team == 0]
    team_1 = [p for p in players if p.team == 1]

    # Analyze
    engine = DecisionEngine()
    result = engine.analyze(
        ball_position=(15, 0),
        ball_carrier_id=1,
        possession_team=0,
        teammates=team_0,
        opponents=team_1
    )

    # Get best option
    print(f"Best option: {result.best_option}")
    print(f"Defensive gaps: {len(result.defensive_gaps)}")
"""

from .engine import (
    # Data classes
    PlayerState,
    DefensiveGap,
    ProgressionOption,
    DecisionEngineOutput,

    # Models
    CoverageZoneModel,
    DefensiveGapDetector,
    XGZoneModel,
    InterceptionModel,
    PassSuccessModel,

    # Main engine
    DecisionEngine,

    # Constants
    PITCH_LENGTH,
    PITCH_WIDTH,
)

from .visualization import (
    PitchVisualizer,
    MatchAnimator,
)

__all__ = [
    # Data classes
    'PlayerState',
    'DefensiveGap',
    'ProgressionOption',
    'DecisionEngineOutput',

    # Models
    'CoverageZoneModel',
    'DefensiveGapDetector',
    'XGZoneModel',
    'InterceptionModel',
    'PassSuccessModel',

    # Main engine
    'DecisionEngine',

    # Visualization
    'PitchVisualizer',
    'MatchAnimator',

    # Constants
    'PITCH_LENGTH',
    'PITCH_WIDTH',
]
