"""
Elimination Logic - The Core Attacking Metric

A defender is eliminated if:
1. The ball is past them (positionally, toward the attacking goal)
2. They cannot reach an effective intervention point before the attacker
   can reach a more dangerous outcome (goal approach or shot)

Elimination is BINARY at the moment of evaluation. A defender is either
eliminated or not. There is no partial elimination.

Key insight: A defender who is goal-side but functionally irrelevant
(cannot intervene in time) is still eliminated.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
import math

from .pitch_geometry import (
    Position,
    Velocity,
    PitchGeometry,
    HALF_LENGTH,
    _dot,
    _norm,
)


class DefenderStatus(Enum):
    """Status of a defender relative to the current attack."""
    ACTIVE = "active"           # Can still intervene
    ELIMINATED = "eliminated"   # Cannot meaningfully intervene
    COVERING = "covering"       # Goal-side and able to intervene


@dataclass
class Player:
    """
    Player state for elimination calculations.

    In the idealized model, all players have equal physical attributes.
    Differentiation comes from position and space, not talent.
    """
    id: str
    position: Position
    velocity: Velocity = field(default_factory=Velocity.zero)
    team: str = "unknown"  # "attack" or "defense"

    # Default physical attributes (equal for all in base model)
    max_speed: float = 8.0          # m/s (~28.8 km/h, elite sprint)
    reaction_time: float = 0.25     # seconds

    def time_to_position(self, target: Position) -> float:
        """
        Time to reach a target position.

        Simplified model: instant acceleration to max speed.
        """
        distance = self.position.distance_to(target)
        return self.reaction_time + distance / self.max_speed

    def time_to_position_with_momentum(self, target: Position) -> float:
        """
        Time to reach target accounting for current velocity.

        If already moving toward target, arrival is faster.
        If moving away, need to decelerate first.
        """
        # Vector to target
        to_target = [
            target.x - self.position.x,
            target.y - self.position.y
        ]
        distance = _norm(to_target)

        if distance < 0.1:
            return 0.0

        # Unit vector to target
        to_target_unit = [to_target[0] / distance, to_target[1] / distance]

        # Current velocity vector
        vel = self.velocity.to_list()

        # Component of velocity toward target
        vel_toward = _dot(vel, to_target_unit)

        # If moving toward target
        if vel_toward > 0:
            # Effective head start
            head_start = vel_toward * self.reaction_time
            effective_distance = max(0, distance - head_start)
            return self.reaction_time + effective_distance / self.max_speed
        else:
            # Moving away - need to stop and reverse
            # Simplified: just add a penalty
            reversal_penalty = abs(vel_toward) / self.max_speed * 0.5
            return self.reaction_time + reversal_penalty + distance / self.max_speed


@dataclass
class EliminationResult:
    """Result of elimination check for a single defender."""
    defender: Player
    status: DefenderStatus
    time_to_intervene: float          # Time for defender to reach intervention point
    time_to_danger: float             # Time for attacker to reach danger point
    intervention_point: Position      # Where defender would try to intervene
    margin: float                     # Negative = eliminated, Positive = can intervene

    @property
    def is_eliminated(self) -> bool:
        return self.status == DefenderStatus.ELIMINATED


@dataclass
class EliminationState:
    """
    Complete elimination state for a game moment.

    This is the primary output of the elimination calculator.
    """
    ball_position: Position
    ball_carrier: Optional[Player]
    defenders: List[EliminationResult]
    timestamp: float = 0.0

    @property
    def eliminated_count(self) -> int:
        """Number of defenders eliminated."""
        return sum(1 for d in self.defenders if d.is_eliminated)

    @property
    def active_count(self) -> int:
        """Number of defenders still active."""
        return sum(1 for d in self.defenders if not d.is_eliminated)

    @property
    def elimination_ratio(self) -> float:
        """Ratio of eliminated defenders (0 to 1)."""
        if not self.defenders:
            return 0.0
        return self.eliminated_count / len(self.defenders)

    def get_eliminated(self) -> List[Player]:
        """Get list of eliminated defenders."""
        return [r.defender for r in self.defenders if r.is_eliminated]

    def get_active(self) -> List[Player]:
        """Get list of active defenders."""
        return [r.defender for r in self.defenders if not r.is_eliminated]


class EliminationCalculator:
    """
    Calculator for defender elimination.

    Core algorithm:
    1. Determine the "danger trajectory" - path from ball to goal
    2. For each defender:
       a. Check if they are goal-side of the ball
       b. Calculate their optimal intervention point on the danger trajectory
       c. Calculate time for defender to reach that point
       d. Calculate time for attacker to make the situation more dangerous
       e. If defender time > attacker time: ELIMINATED

    Intervention points are calculated along the direct ball-to-goal line,
    as this represents the most dangerous attacking action.
    """

    def __init__(
        self,
        geometry: Optional[PitchGeometry] = None,
        intervention_margin: float = 0.5,  # meters - buffer for "close enough"
        time_margin: float = 0.1,          # seconds - buffer for timing
    ):
        self.geometry = geometry or PitchGeometry()
        self.intervention_margin = intervention_margin
        self.time_margin = time_margin

    def calculate(
        self,
        ball_position: Position,
        ball_carrier: Optional[Player],
        defenders: List[Player],
        attacking_team_goal_x: float = HALF_LENGTH,
    ) -> EliminationState:
        """
        Calculate elimination state for all defenders.

        Args:
            ball_position: Current ball position
            ball_carrier: Player with the ball (if any)
            defenders: List of defending players
            attacking_team_goal_x: X coordinate of the goal being attacked

        Returns:
            EliminationState with status for each defender
        """
        goal_position = Position(attacking_team_goal_x, 0.0)
        results = []

        for defender in defenders:
            result = self._evaluate_defender(
                defender=defender,
                ball_position=ball_position,
                ball_carrier=ball_carrier,
                goal_position=goal_position,
            )
            results.append(result)

        return EliminationState(
            ball_position=ball_position,
            ball_carrier=ball_carrier,
            defenders=results,
        )

    def _evaluate_defender(
        self,
        defender: Player,
        ball_position: Position,
        ball_carrier: Optional[Player],
        goal_position: Position,
    ) -> EliminationResult:
        """
        Evaluate elimination status for a single defender.
        """
        # Step 1: Is defender goal-side of the ball?
        is_goal_side = self._is_goal_side(
            defender.position,
            ball_position,
            goal_position,
        )

        # Step 2: Find optimal intervention point
        intervention_point = self._find_intervention_point(
            defender.position,
            ball_position,
            goal_position,
        )

        # Step 3: Calculate time for defender to reach intervention
        time_to_intervene = defender.time_to_position_with_momentum(
            intervention_point
        )

        # Step 4: Calculate time for attack to progress past intervention
        time_to_danger = self._calculate_danger_time(
            ball_position=ball_position,
            ball_carrier=ball_carrier,
            intervention_point=intervention_point,
            goal_position=goal_position,
        )

        # Step 5: Determine status
        margin = time_to_danger - time_to_intervene

        if not is_goal_side:
            # Defender is behind the ball - eliminated by position
            status = DefenderStatus.ELIMINATED
        elif margin < -self.time_margin:
            # Attacker reaches danger before defender can intervene
            status = DefenderStatus.ELIMINATED
        elif margin > self.time_margin:
            # Defender has clear time advantage
            status = DefenderStatus.COVERING
        else:
            # Close call - defender is active but marginal
            status = DefenderStatus.ACTIVE

        return EliminationResult(
            defender=defender,
            status=status,
            time_to_intervene=time_to_intervene,
            time_to_danger=time_to_danger,
            intervention_point=intervention_point,
            margin=margin,
        )

    def _is_goal_side(
        self,
        defender_pos: Position,
        ball_pos: Position,
        goal_pos: Position,
    ) -> bool:
        """
        Check if defender is between ball and goal (goal-side).

        A defender is goal-side if they are closer to their own goal
        than the ball is, along the attacking axis.
        """
        # Attacking direction
        attacking_direction = 1 if goal_pos.x > 0 else -1

        if attacking_direction > 0:
            # Attacking toward positive x
            # Goal-side means defender.x > ball.x (closer to goal)
            return defender_pos.x > ball_pos.x - self.intervention_margin
        else:
            # Attacking toward negative x
            return defender_pos.x < ball_pos.x + self.intervention_margin

    def _find_intervention_point(
        self,
        defender_pos: Position,
        ball_pos: Position,
        goal_pos: Position,
    ) -> Position:
        """
        Find the optimal point for defender to intercept the attack.

        The optimal intervention point is the closest point on the
        ball-to-goal line that the defender can reach.
        """
        # Line from ball to goal
        line_vec = [goal_pos.x - ball_pos.x, goal_pos.y - ball_pos.y]
        line_len = _norm(line_vec)

        if line_len < 0.1:
            return ball_pos

        line_unit = [line_vec[0] / line_len, line_vec[1] / line_len]

        # Vector from ball to defender
        to_defender = [
            defender_pos.x - ball_pos.x,
            defender_pos.y - ball_pos.y
        ]

        # Project defender position onto line
        projection = _dot(to_defender, line_unit)

        # Clamp to line segment (between ball and goal)
        projection = max(0, min(line_len, projection))

        # Calculate intervention point
        intervention = Position(
            ball_pos.x + projection * line_unit[0],
            ball_pos.y + projection * line_unit[1],
        )

        return intervention

    def _calculate_danger_time(
        self,
        ball_position: Position,
        ball_carrier: Optional[Player],
        intervention_point: Position,
        goal_position: Position,
    ) -> float:
        """
        Calculate time for the attack to progress past the intervention point.

        This is the "clock" that the defender is racing against.
        If the ball carrier can reach the intervention point (or beyond)
        before the defender, the defender is eliminated.
        """
        # Distance from ball to intervention point
        ball_to_intervention = ball_position.distance_to(intervention_point)

        if ball_carrier:
            # Carrier dribbling speed (slower than sprint)
            dribble_speed = ball_carrier.max_speed * 0.8
            time_to_reach = ball_to_intervention / dribble_speed
        else:
            # Ball is moving or loose - use pass speed
            pass_speed = 15.0  # m/s for a firm pass
            time_to_reach = ball_to_intervention / pass_speed

        return time_to_reach


class EliminationAnalyzer:
    """
    Higher-level analysis of elimination patterns.

    Provides insights beyond raw elimination counts.
    """

    def __init__(self, geometry: Optional[PitchGeometry] = None):
        self.geometry = geometry or PitchGeometry()
        self.calculator = EliminationCalculator(geometry=self.geometry)

    def analyze_line_breaking(
        self,
        state: EliminationState,
        defender_lines: Optional[Dict[str, List[Player]]] = None,
    ) -> Dict[str, int]:
        """
        Analyze how many defensive lines have been broken.

        Defensive lines (conceptual):
        - Forward line: Pressing forwards
        - Midfield line: Central midfielders
        - Defensive line: Back four/five
        - Goalkeeper: Last line

        A line is "broken" if the majority of that line is eliminated.
        """
        if defender_lines is None:
            # Infer lines from positions
            defender_lines = self._infer_defensive_lines(state)

        result = {}
        for line_name, line_players in defender_lines.items():
            # Find these players in the state
            line_ids = {p.id for p in line_players}
            line_results = [
                r for r in state.defenders
                if r.defender.id in line_ids
            ]

            eliminated = sum(1 for r in line_results if r.is_eliminated)
            total = len(line_results)

            result[line_name] = {
                "eliminated": eliminated,
                "total": total,
                "broken": eliminated > total / 2 if total > 0 else False,
            }

        return result

    def _infer_defensive_lines(
        self,
        state: EliminationState,
    ) -> Dict[str, List[Player]]:
        """
        Infer defensive lines from positions.

        Simple heuristic: cluster defenders by x-coordinate.
        """
        if not state.defenders:
            return {}

        # Get x positions
        defenders = [r.defender for r in state.defenders]
        x_positions = [d.position.x for d in defenders]

        if len(x_positions) < 2:
            return {"line_1": defenders}

        # Simple clustering by x-position
        sorted_defenders = sorted(defenders, key=lambda d: d.position.x)

        # Split into lines based on gaps
        lines = []
        current_line = [sorted_defenders[0]]

        for i in range(1, len(sorted_defenders)):
            gap = sorted_defenders[i].position.x - sorted_defenders[i-1].position.x
            if gap > 10:  # 10m gap indicates new line
                lines.append(current_line)
                current_line = []
            current_line.append(sorted_defenders[i])

        if current_line:
            lines.append(current_line)

        # Name the lines
        result = {}
        for i, line in enumerate(lines):
            result[f"line_{i+1}"] = line

        return result

    def calculate_territorial_advantage(
        self,
        state: EliminationState,
    ) -> float:
        """
        Calculate territorial advantage based on eliminations.

        Returns a score from -1 (all defenders active, no advantage)
        to +1 (all defenders eliminated, maximum advantage).
        """
        if not state.defenders:
            return 0.0

        # Base score from elimination ratio
        base_score = state.elimination_ratio

        # Weight by position - eliminations closer to goal are more valuable
        weighted_score = 0.0
        total_weight = 0.0

        for result in state.defenders:
            if result.is_eliminated:
                # Distance from goal (normalized)
                dist = self.geometry.distance_to_attacking_goal(
                    result.defender.position
                )
                # Weight: closer to goal = higher weight
                weight = max(0.1, 1.0 - dist / 50.0)
                weighted_score += weight
                total_weight += weight

        if total_weight > 0:
            position_factor = weighted_score / total_weight
        else:
            position_factor = 0.0

        # Combine base and position-weighted scores
        return 0.6 * base_score + 0.4 * position_factor

    def find_most_valuable_elimination(
        self,
        state: EliminationState,
    ) -> Optional[EliminationResult]:
        """
        Find the defender whose elimination is most valuable.

        The most valuable eliminated defender is typically:
        - Closest to the goal
        - On the most direct line to goal
        - Part of the last defensive line
        """
        eliminated = [r for r in state.defenders if r.is_eliminated]

        if not eliminated:
            return None

        # Score each elimination
        def score_elimination(result: EliminationResult) -> float:
            dist_to_goal = self.geometry.distance_to_attacking_goal(
                result.defender.position
            )
            # Closer to goal = more valuable
            return 1.0 / (dist_to_goal + 1.0)

        return max(eliminated, key=score_elimination)
