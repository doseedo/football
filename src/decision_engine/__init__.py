"""
Football Decision Engine

A physics-based tactical modeling system for football analysis.

Core Concepts:
- Elimination: A defender is eliminated when the ball is past them and they
  cannot reach an effective intervention point before the attacker achieves
  a more dangerous outcome.
- Defensive Attraction: Defenders are modeled as being pulled toward ball,
  goal, and key spaces using attraction-based physics.
- State Scoring: Game states are evaluated numerically based on eliminations,
  spatial advantage, and threat proximity.

This is a tactical laboratory, not a game engine.
"""

from .pitch_geometry import (
    Position,
    Velocity,
    PitchGeometry,
    PITCH_LENGTH,
    PITCH_WIDTH,
    HALF_LENGTH,
    HALF_WIDTH,
)
from .elimination import (
    Player,
    EliminationCalculator,
    EliminationState,
    EliminationResult,
    DefenderStatus,
)
from .defense_physics import (
    DefensiveForceModel,
    AttractionForce,
    DefensiveShape,
    ForceType,
    CoverShadowCalculator,
)
from .state_scoring import (
    GameStateEvaluator,
    GameState,
    StateScore,
    ActionType,
    ActionOption,
)
from .block_models import (
    DefensiveBlock,
    BlockType,
    BlockConfiguration,
    BlockTransitionManager,
    LOW_BLOCK,
    MID_BLOCK,
    HIGH_BLOCK,
)
# Visualizer requires numpy/matplotlib - import conditionally
try:
    from .visualizer import DecisionEngineVisualizer
except ImportError:
    DecisionEngineVisualizer = None  # Not available without numpy

from .action_executor import (
    ActionExecutor,
    ActionResult,
    ExecutionResult,
    DecisionStep,
    DecisionEngine,
    create_test_scenario,
)
from .tactical_ui import generate_tactical_ui, run_and_create_ui
from .visualizer_web import generate_visualization_html, run_and_visualize

__all__ = [
    # Geometry
    "Position",
    "Velocity",
    "PitchGeometry",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    "HALF_LENGTH",
    "HALF_WIDTH",
    # Elimination
    "Player",
    "EliminationCalculator",
    "EliminationState",
    "EliminationResult",
    "DefenderStatus",
    # Defense Physics
    "DefensiveForceModel",
    "AttractionForce",
    "DefensiveShape",
    "ForceType",
    "CoverShadowCalculator",
    # State Scoring
    "GameStateEvaluator",
    "GameState",
    "StateScore",
    "ActionType",
    "ActionOption",
    # Block Models
    "DefensiveBlock",
    "BlockType",
    "BlockConfiguration",
    "BlockTransitionManager",
    "LOW_BLOCK",
    "MID_BLOCK",
    "HIGH_BLOCK",
    # Visualization
    "DecisionEngineVisualizer",
    # Action Execution & Decision Loop
    "ActionExecutor",
    "ActionResult",
    "ExecutionResult",
    "DecisionStep",
    "DecisionEngine",
    "create_test_scenario",
    # Web Visualization & UI
    "generate_tactical_ui",
    "run_and_create_ui",
    "generate_visualization_html",
    "run_and_visualize",
]
