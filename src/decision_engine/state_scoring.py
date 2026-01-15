"""
Game State Scoring System

At any moment, a game state can be evaluated using a composite score:
- Number of defenders eliminated
- Distance to goal
- Angle to goal
- Density of defenders near the ball
- Compactness of the defending unit
- Availability of forward actions (pass, dribble, shot)

This score allows:
- Objective comparison of positions
- Ranking ball locations or actions
- Identification of optimal zones for progression or shooting
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import numpy as np

from .pitch_geometry import (
    Position,
    Velocity,
    PitchGeometry,
    HALF_LENGTH,
    HALF_WIDTH,
    PENALTY_AREA_LENGTH,
)
from .elimination import (
    Player,
    EliminationCalculator,
    EliminationState,
)
from .defense_physics import (
    DefensiveForceModel,
    CoverShadowCalculator,
)


class ActionType(Enum):
    """Types of attacking actions available."""
    SHOT = "shot"
    PASS_FORWARD = "pass_forward"
    PASS_LATERAL = "pass_lateral"
    PASS_BACKWARD = "pass_backward"
    DRIBBLE = "dribble"
    CARRY = "carry"


@dataclass
class ActionOption:
    """A potential action and its value."""
    action_type: ActionType
    target: Position                    # Where action goes
    success_probability: float          # 0-1
    value_if_success: float            # Expected state improvement
    risk_if_failure: float             # Expected state degradation
    expected_value: float              # Combined EV

    @property
    def is_worthwhile(self) -> bool:
        """Action has positive expected value."""
        return self.expected_value > 0


@dataclass
class StateScore:
    """
    Composite score for a game state.

    Higher score = more advantageous for attacking team.
    Score is normalized to [0, 1] range.
    """
    # Component scores (each 0-1)
    elimination_score: float        # Based on defenders eliminated
    proximity_score: float          # Based on distance to goal
    angle_score: float              # Based on shooting angle
    density_score: float            # Based on space around ball
    compactness_score: float        # Based on defensive structure
    action_score: float             # Based on available options

    # Weights for combining components
    weights: Dict[str, float] = field(default_factory=lambda: {
        "elimination": 0.25,
        "proximity": 0.20,
        "angle": 0.15,
        "density": 0.15,
        "compactness": 0.10,
        "action": 0.15,
    })

    @property
    def total(self) -> float:
        """Weighted total score."""
        return (
            self.weights["elimination"] * self.elimination_score +
            self.weights["proximity"] * self.proximity_score +
            self.weights["angle"] * self.angle_score +
            self.weights["density"] * self.density_score +
            self.weights["compactness"] * self.compactness_score +
            self.weights["action"] * self.action_score
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "elimination": self.elimination_score,
            "proximity": self.proximity_score,
            "angle": self.angle_score,
            "density": self.density_score,
            "compactness": self.compactness_score,
            "action": self.action_score,
            "total": self.total,
        }


@dataclass
class GameState:
    """
    Complete game state at a moment in time.

    Contains all information needed to evaluate the position.
    """
    ball_position: Position
    ball_carrier: Optional[Player]
    attackers: List[Player]
    defenders: List[Player]
    timestamp: float = 0.0

    # Derived data (populated by evaluator)
    elimination_state: Optional[EliminationState] = None
    score: Optional[StateScore] = None
    available_actions: List[ActionOption] = field(default_factory=list)


class GameStateEvaluator:
    """
    Evaluates game states and produces scores.

    This is the core "brain" of the decision engine - it takes
    a game state and produces actionable analysis.
    """

    def __init__(
        self,
        geometry: Optional[PitchGeometry] = None,
        # Scoring parameters
        max_shooting_distance: float = 30.0,  # Beyond this, shot value is 0
        optimal_shooting_distance: float = 12.0,  # Peak shot value
        density_radius: float = 10.0,  # Radius for density calculation
    ):
        self.geometry = geometry or PitchGeometry()
        self.elimination_calc = EliminationCalculator(geometry=self.geometry)
        self.force_model = DefensiveForceModel(geometry=self.geometry)
        self.shadow_calc = CoverShadowCalculator(geometry=self.geometry)

        self.max_shooting_distance = max_shooting_distance
        self.optimal_shooting_distance = optimal_shooting_distance
        self.density_radius = density_radius

    def evaluate(self, state: GameState) -> GameState:
        """
        Fully evaluate a game state.

        Populates elimination state, score, and available actions.
        """
        # 1. Calculate eliminations
        state.elimination_state = self.elimination_calc.calculate(
            ball_position=state.ball_position,
            ball_carrier=state.ball_carrier,
            defenders=state.defenders,
        )

        # 2. Calculate component scores
        elimination_score = self._score_elimination(state)
        proximity_score = self._score_proximity(state)
        angle_score = self._score_angle(state)
        density_score = self._score_density(state)
        compactness_score = self._score_compactness(state)
        action_score = self._score_actions(state)

        state.score = StateScore(
            elimination_score=elimination_score,
            proximity_score=proximity_score,
            angle_score=angle_score,
            density_score=density_score,
            compactness_score=compactness_score,
            action_score=action_score,
        )

        # 3. Find available actions
        state.available_actions = self._find_actions(state)

        return state

    def _score_elimination(self, state: GameState) -> float:
        """
        Score based on defender eliminations.

        More eliminations = higher score.
        Eliminations closer to goal worth more.
        """
        if not state.elimination_state or not state.defenders:
            return 0.0

        # Base score: ratio of eliminated defenders
        base = state.elimination_state.elimination_ratio

        # Position weighting: eliminations near goal worth more
        weighted_sum = 0.0
        for result in state.elimination_state.defenders:
            if result.is_eliminated:
                dist_to_goal = self.geometry.distance_to_attacking_goal(
                    result.defender.position
                )
                # Weight: 1.0 at goal, 0.5 at midfield
                weight = max(0.5, 1.0 - dist_to_goal / (HALF_LENGTH * 2))
                weighted_sum += weight

        # Normalize
        max_weighted = len(state.defenders)  # If all eliminated at goal
        weighted_score = weighted_sum / max_weighted if max_weighted > 0 else 0.0

        # Combine base and weighted
        return 0.5 * base + 0.5 * weighted_score

    def _score_proximity(self, state: GameState) -> float:
        """
        Score based on distance to goal.

        Closer to goal = higher score.
        Nonlinear - drops off quickly beyond penalty area.
        """
        dist = self.geometry.distance_to_attacking_goal(state.ball_position)

        if dist <= self.optimal_shooting_distance:
            # In prime shooting range
            return 1.0
        elif dist <= self.max_shooting_distance:
            # Shooting range but not optimal
            return 0.8 - 0.3 * (dist - self.optimal_shooting_distance) / (
                self.max_shooting_distance - self.optimal_shooting_distance
            )
        elif dist <= HALF_LENGTH:
            # Attacking half
            return 0.5 - 0.3 * (dist - self.max_shooting_distance) / (
                HALF_LENGTH - self.max_shooting_distance
            )
        else:
            # Own half
            return 0.2 * (1.0 - (dist - HALF_LENGTH) / HALF_LENGTH)

    def _score_angle(self, state: GameState) -> float:
        """
        Score based on shooting angle to goal.

        Central positions with wide angle = higher score.
        """
        angle = self.geometry.angle_to_goal(state.ball_position, attacking=True)

        # Max angle is about 0.3 radians (~17 degrees) from 12m
        # Normalize to 0-1
        max_practical_angle = 0.4  # radians
        normalized = min(1.0, angle / max_practical_angle)

        return normalized

    def _score_density(self, state: GameState) -> float:
        """
        Score based on space around the ball.

        Fewer defenders nearby = higher score (more space).
        """
        defenders_in_radius = 0
        for defender in state.defenders:
            dist = defender.position.distance_to(state.ball_position)
            if dist < self.density_radius:
                defenders_in_radius += 1

        # Score: 1.0 if no defenders nearby, 0.0 if 5+ defenders
        max_defenders = 5
        score = 1.0 - min(defenders_in_radius, max_defenders) / max_defenders

        return score

    def _score_compactness(self, state: GameState) -> float:
        """
        Score based on defensive compactness.

        Stretched defense = higher score for attack.
        Compact defense = lower score.
        """
        if len(state.defenders) < 2:
            return 0.5

        # Calculate average pairwise distance
        total_dist = 0.0
        pairs = 0
        for i, d1 in enumerate(state.defenders):
            for d2 in state.defenders[i+1:]:
                total_dist += d1.position.distance_to(d2.position)
                pairs += 1

        avg_dist = total_dist / pairs if pairs > 0 else 0.0

        # Score: compact (avg 8m) = 0, stretched (avg 20m) = 1
        min_compact = 8.0
        max_spread = 20.0
        score = (avg_dist - min_compact) / (max_spread - min_compact)

        return np.clip(score, 0.0, 1.0)

    def _score_proximity_simple(self, pos: Position) -> float:
        """Simple proximity score for a position (no state needed)."""
        dist = self.geometry.distance_to_attacking_goal(pos)

        if dist <= self.optimal_shooting_distance:
            return 1.0
        elif dist <= self.max_shooting_distance:
            return 0.8 - 0.3 * (dist - self.optimal_shooting_distance) / (
                self.max_shooting_distance - self.optimal_shooting_distance
            )
        elif dist <= HALF_LENGTH:
            return 0.5 - 0.3 * (dist - self.max_shooting_distance) / (
                HALF_LENGTH - self.max_shooting_distance
            )
        else:
            return 0.2 * (1.0 - (dist - HALF_LENGTH) / HALF_LENGTH)

    def _score_angle_simple(self, pos: Position) -> float:
        """Simple angle score for a position (no state needed)."""
        angle = self.geometry.angle_to_goal(pos, attacking=True)
        max_practical_angle = 0.4
        return min(1.0, angle / max_practical_angle)

    def _score_actions(self, state: GameState) -> float:
        """
        Score based on available forward actions.

        More high-value options = higher score.
        """
        actions = self._find_actions(state)

        if not actions:
            return 0.0

        # Score based on best available action
        best_ev = max(a.expected_value for a in actions)

        # Normalize (assuming max EV of 0.5)
        return min(1.0, best_ev * 2)

    def _find_actions(self, state: GameState) -> List[ActionOption]:
        """
        Find all available actions and evaluate them.
        """
        actions = []

        # Shot evaluation
        shot = self._evaluate_shot(state)
        if shot:
            actions.append(shot)

        # Pass options to each teammate
        for attacker in state.attackers:
            if state.ball_carrier and attacker.id == state.ball_carrier.id:
                continue  # Can't pass to yourself

            pass_option = self._evaluate_pass(state, attacker)
            if pass_option:
                actions.append(pass_option)

        # Dribble/carry options
        dribble = self._evaluate_dribble(state)
        if dribble:
            actions.append(dribble)

        return sorted(actions, key=lambda a: -a.expected_value)

    def _evaluate_shot(self, state: GameState) -> Optional[ActionOption]:
        """
        Evaluate shooting option.
        """
        dist = self.geometry.distance_to_attacking_goal(state.ball_position)

        if dist > self.max_shooting_distance:
            return None

        # Success probability based on distance and angle
        dist_factor = max(0, 1 - (dist - 5) / (self.max_shooting_distance - 5))
        angle = self.geometry.angle_to_goal(state.ball_position)
        angle_factor = min(1.0, angle / 0.3)

        # Defender blocking factor
        blocking = 0
        for defender in state.defenders:
            defender_dist = defender.position.distance_to(state.ball_position)
            if defender_dist < 3:
                blocking += 0.3
            elif defender_dist < 6:
                blocking += 0.1

        success_prob = dist_factor * angle_factor * (1 - min(blocking, 0.8))

        # Value if goal scored is maximum (normalized to 1.0)
        value_if_success = 1.0

        # Risk if missed is position loss
        risk_if_failure = 0.3

        expected_value = (
            success_prob * value_if_success -
            (1 - success_prob) * risk_if_failure
        )

        return ActionOption(
            action_type=ActionType.SHOT,
            target=self.geometry.attacking_goal,
            success_probability=success_prob,
            value_if_success=value_if_success,
            risk_if_failure=risk_if_failure,
            expected_value=expected_value,
        )

    def _evaluate_pass(
        self,
        state: GameState,
        target_player: Player,
    ) -> Optional[ActionOption]:
        """
        Evaluate passing to a teammate.
        """
        target = target_player.position

        # Determine pass type
        ball_x = state.ball_position.x
        target_x = target.x

        if target_x > ball_x + 5:
            action_type = ActionType.PASS_FORWARD
            value_multiplier = 1.2  # Forward passes more valuable
        elif target_x < ball_x - 5:
            action_type = ActionType.PASS_BACKWARD
            value_multiplier = 0.6  # Backward passes less valuable
        else:
            action_type = ActionType.PASS_LATERAL
            value_multiplier = 0.8

        # Success probability based on:
        # - Distance
        # - Interception risk (defenders in path)
        pass_dist = state.ball_position.distance_to(target)

        # Distance factor
        dist_factor = max(0.3, 1 - pass_dist / 40)

        # Interception factor
        interception_risk = 0.0
        for defender in state.defenders:
            # Check if defender is in passing lane
            perp_dist = self.geometry.perpendicular_distance_to_line(
                defender.position,
                state.ball_position,
                target,
            )
            if perp_dist < 2:
                interception_risk += 0.3
            elif perp_dist < 4:
                interception_risk += 0.1

        success_prob = dist_factor * (1 - min(interception_risk, 0.8))

        # Value if successful - use simple position-based score (no recursion)
        proximity_score = self._score_proximity_simple(target)
        angle_score = self._score_angle_simple(target)
        value_if_success = (0.6 * proximity_score + 0.4 * angle_score) * value_multiplier

        # Risk if intercepted
        risk_if_failure = 0.4

        expected_value = (
            success_prob * value_if_success -
            (1 - success_prob) * risk_if_failure
        )

        return ActionOption(
            action_type=action_type,
            target=target,
            success_probability=success_prob,
            value_if_success=value_if_success,
            risk_if_failure=risk_if_failure,
            expected_value=expected_value,
        )

    def _evaluate_dribble(self, state: GameState) -> Optional[ActionOption]:
        """
        Evaluate dribbling forward.
        """
        if not state.ball_carrier:
            return None

        # Target: 5m forward
        target = Position(
            state.ball_position.x + 5,
            state.ball_position.y,
        )

        # Clamp to pitch
        target.x = min(target.x, HALF_LENGTH - 1)

        # Success probability based on nearby defenders
        nearby_defenders = 0
        for defender in state.defenders:
            dist = defender.position.distance_to(state.ball_position)
            if dist < 3:
                nearby_defenders += 1

        if nearby_defenders >= 2:
            success_prob = 0.2
        elif nearby_defenders == 1:
            success_prob = 0.5
        else:
            success_prob = 0.8

        # Value: how much does advancing improve position?
        value_if_success = 0.3  # Modest improvement

        # Risk of losing ball
        risk_if_failure = 0.5

        expected_value = (
            success_prob * value_if_success -
            (1 - success_prob) * risk_if_failure
        )

        return ActionOption(
            action_type=ActionType.DRIBBLE,
            target=target,
            success_probability=success_prob,
            value_if_success=value_if_success,
            risk_if_failure=risk_if_failure,
            expected_value=expected_value,
        )

    def compare_positions(
        self,
        positions: List[Position],
        state: GameState,
    ) -> List[Tuple[Position, float]]:
        """
        Compare multiple ball positions and rank them.

        Useful for identifying optimal zones.
        """
        results = []

        for pos in positions:
            # Create hypothetical state
            hypo_state = GameState(
                ball_position=pos,
                ball_carrier=state.ball_carrier,
                attackers=state.attackers,
                defenders=state.defenders,
            )
            evaluated = self.evaluate(hypo_state)
            results.append((pos, evaluated.score.total))

        # Sort by score descending
        return sorted(results, key=lambda x: -x[1])

    def generate_value_heatmap(
        self,
        state: GameState,
        grid_resolution: float = 5.0,
    ) -> np.ndarray:
        """
        Generate a heatmap of position values across the pitch.

        Returns 2D array of scores for each grid cell.
        """
        x_points = int(self.geometry.length / grid_resolution)
        y_points = int(self.geometry.width / grid_resolution)

        heatmap = np.zeros((y_points, x_points))

        for i in range(x_points):
            for j in range(y_points):
                # Convert grid to pitch coordinates
                x = -HALF_LENGTH + (i + 0.5) * grid_resolution
                y = -HALF_WIDTH + (j + 0.5) * grid_resolution

                # Evaluate this position
                hypo_state = GameState(
                    ball_position=Position(x, y),
                    ball_carrier=state.ball_carrier,
                    attackers=state.attackers,
                    defenders=state.defenders,
                )
                evaluated = self.evaluate(hypo_state)
                heatmap[j, i] = evaluated.score.total

        return heatmap
