"""
Defensive Physics Model

Defensive behavior is modeled using attraction-based principles.
Defenders are pulled toward:
- Ball (primary attraction)
- Goal (protective attraction)
- Key spaces (lane coverage, zonal responsibility)
- Opponents (marking)

The result is emergent collective behavior:
- Compactness (players cluster around ball-goal axis)
- Cover shadows (blocking passing lanes)
- Shifting as a unit (collective movement)

This framework is naturally suited for defense because defense is
reactive and space-denying, unlike attack which is about advantage creation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import numpy as np

from .pitch_geometry import (
    Position,
    Velocity,
    PitchGeometry,
    HALF_LENGTH,
    HALF_WIDTH,
)
from .elimination import Player


class ForceType(Enum):
    """Types of attraction forces acting on defenders."""
    BALL = "ball"           # Attraction to ball position
    GOAL = "goal"           # Attraction toward own goal (protection)
    ZONE = "zone"           # Attraction to assigned zone
    OPPONENT = "opponent"   # Attraction to mark specific opponent
    TEAMMATE = "teammate"   # Repulsion from teammates (spacing)
    LINE = "line"           # Attraction to maintain defensive line


@dataclass
class AttractionForce:
    """
    A force vector acting on a defender.

    Forces combine additively. The resultant force determines
    the defender's ideal position/movement direction.
    """
    force_type: ForceType
    magnitude: float        # Force strength (arbitrary units)
    direction: np.ndarray   # Unit vector of force direction
    source: Position        # Where the force originates
    target: Position        # The defender being affected

    @property
    def vector(self) -> np.ndarray:
        """Force as a vector."""
        return self.magnitude * self.direction

    def to_position_offset(self) -> Position:
        """Convert force to position offset."""
        vec = self.vector
        return Position(vec[0], vec[1])


@dataclass
class DefensiveShape:
    """
    The collective shape formed by defenders under force equilibrium.

    This represents the "natural" defensive formation when all
    attraction forces are in balance.
    """
    positions: Dict[str, Position]  # player_id -> ideal position
    compactness: float              # Average pairwise distance
    depth: float                    # Distance from front to back
    width: float                    # Lateral spread
    center_of_mass: Position        # Team centroid
    line_heights: List[float]       # Y positions of defensive lines


class DefensiveForceModel:
    """
    Physics-based model for defensive positioning.

    Each defender experiences multiple forces:
    1. Ball attraction: Pull toward ball (pressure)
    2. Goal attraction: Pull toward own goal (protection)
    3. Zone attraction: Pull toward assigned zone (structure)
    4. Opponent attraction: Pull toward dangerous attackers (marking)
    5. Teammate repulsion: Push away from teammates (spacing)

    The equilibrium of these forces determines optimal positioning.
    """

    def __init__(
        self,
        geometry: Optional[PitchGeometry] = None,
        # Force weights (tunable parameters)
        ball_weight: float = 1.0,
        goal_weight: float = 0.6,
        zone_weight: float = 0.4,
        opponent_weight: float = 0.8,
        teammate_repulsion: float = 0.3,
        line_weight: float = 0.5,
    ):
        self.geometry = geometry or PitchGeometry()

        # Force weights
        self.ball_weight = ball_weight
        self.goal_weight = goal_weight
        self.zone_weight = zone_weight
        self.opponent_weight = opponent_weight
        self.teammate_repulsion = teammate_repulsion
        self.line_weight = line_weight

    def calculate_forces(
        self,
        defender: Player,
        ball_position: Position,
        teammates: List[Player],
        opponents: List[Player],
        assigned_zone: Optional[Position] = None,
        defensive_line_y: Optional[float] = None,
    ) -> List[AttractionForce]:
        """
        Calculate all forces acting on a defender.

        Returns list of forces that can be combined to determine
        optimal movement/positioning.
        """
        forces = []

        # 1. Ball attraction
        ball_force = self._calculate_ball_force(defender, ball_position)
        if ball_force:
            forces.append(ball_force)

        # 2. Goal protection force
        goal_force = self._calculate_goal_force(defender, ball_position)
        if goal_force:
            forces.append(goal_force)

        # 3. Zone attraction
        if assigned_zone:
            zone_force = self._calculate_zone_force(defender, assigned_zone)
            if zone_force:
                forces.append(zone_force)

        # 4. Opponent marking forces
        for opponent in opponents:
            opp_force = self._calculate_opponent_force(
                defender, opponent, ball_position
            )
            if opp_force:
                forces.append(opp_force)

        # 5. Teammate spacing (repulsion)
        for teammate in teammates:
            if teammate.id != defender.id:
                spacing_force = self._calculate_spacing_force(defender, teammate)
                if spacing_force:
                    forces.append(spacing_force)

        # 6. Defensive line maintenance
        if defensive_line_y is not None:
            line_force = self._calculate_line_force(defender, defensive_line_y)
            if line_force:
                forces.append(line_force)

        return forces

    def _calculate_ball_force(
        self,
        defender: Player,
        ball_position: Position,
    ) -> Optional[AttractionForce]:
        """
        Calculate attraction force toward the ball.

        Force is stronger when ball is closer (inverse square law).
        This creates pressing behavior.
        """
        to_ball = np.array([
            ball_position.x - defender.position.x,
            ball_position.y - defender.position.y,
        ])

        distance = np.linalg.norm(to_ball)
        if distance < 0.1:
            return None

        direction = to_ball / distance

        # Inverse square attraction (stronger when closer)
        # Clamp minimum distance to prevent explosion
        effective_distance = max(distance, 5.0)
        magnitude = self.ball_weight * (100.0 / (effective_distance ** 1.5))

        return AttractionForce(
            force_type=ForceType.BALL,
            magnitude=magnitude,
            direction=direction,
            source=ball_position,
            target=defender.position,
        )

    def _calculate_goal_force(
        self,
        defender: Player,
        ball_position: Position,
    ) -> Optional[AttractionForce]:
        """
        Calculate attraction force toward own goal.

        This creates the "protect the goal" instinct.
        Force increases when ball is closer to goal.
        """
        # Defending goal is at negative x (left side)
        goal_pos = self.geometry.defending_goal

        to_goal = np.array([
            goal_pos.x - defender.position.x,
            goal_pos.y - defender.position.y,
        ])

        distance = np.linalg.norm(to_goal)
        if distance < 0.1:
            return None

        direction = to_goal / distance

        # Force increases when ball is closer to goal (more dangerous)
        ball_danger = 1.0 - (
            self.geometry.distance_to_defending_goal(ball_position) /
            self.geometry.length
        )
        ball_danger = max(0.1, ball_danger)

        magnitude = self.goal_weight * ball_danger * 10.0

        return AttractionForce(
            force_type=ForceType.GOAL,
            magnitude=magnitude,
            direction=direction,
            source=goal_pos,
            target=defender.position,
        )

    def _calculate_zone_force(
        self,
        defender: Player,
        zone_center: Position,
    ) -> Optional[AttractionForce]:
        """
        Calculate attraction force toward assigned zone.

        This maintains zonal structure even under ball pressure.
        """
        to_zone = np.array([
            zone_center.x - defender.position.x,
            zone_center.y - defender.position.y,
        ])

        distance = np.linalg.norm(to_zone)
        if distance < 1.0:
            return None

        direction = to_zone / distance

        # Linear attraction (gentle pull back to zone)
        magnitude = self.zone_weight * min(distance, 20.0) * 0.5

        return AttractionForce(
            force_type=ForceType.ZONE,
            magnitude=magnitude,
            direction=direction,
            source=zone_center,
            target=defender.position,
        )

    def _calculate_opponent_force(
        self,
        defender: Player,
        opponent: Player,
        ball_position: Position,
    ) -> Optional[AttractionForce]:
        """
        Calculate attraction force toward a dangerous opponent.

        Force is weighted by opponent's danger level:
        - Closer to ball = more dangerous
        - Closer to goal = more dangerous
        - In dangerous zone = more dangerous
        """
        to_opponent = np.array([
            opponent.position.x - defender.position.x,
            opponent.position.y - defender.position.y,
        ])

        distance = np.linalg.norm(to_opponent)
        if distance < 0.1:
            return None

        direction = to_opponent / distance

        # Calculate opponent danger level
        ball_dist = opponent.position.distance_to(ball_position)
        goal_dist = self.geometry.distance_to_defending_goal(opponent.position)

        # Danger decreases with distance from ball and goal
        ball_danger = max(0.0, 1.0 - ball_dist / 30.0)
        goal_danger = max(0.0, 1.0 - goal_dist / 50.0)

        danger = ball_danger * 0.6 + goal_danger * 0.4

        # Only attract to dangerous opponents
        if danger < 0.2:
            return None

        magnitude = self.opponent_weight * danger * 8.0

        return AttractionForce(
            force_type=ForceType.OPPONENT,
            magnitude=magnitude,
            direction=direction,
            source=opponent.position,
            target=defender.position,
        )

    def _calculate_spacing_force(
        self,
        defender: Player,
        teammate: Player,
    ) -> Optional[AttractionForce]:
        """
        Calculate repulsion force from nearby teammates.

        This maintains spacing and prevents clustering.
        """
        from_teammate = np.array([
            defender.position.x - teammate.position.x,
            defender.position.y - teammate.position.y,
        ])

        distance = np.linalg.norm(from_teammate)

        # No repulsion if far enough apart
        if distance > 15.0 or distance < 0.1:
            return None

        direction = from_teammate / distance

        # Repulsion increases as teammates get closer
        magnitude = self.teammate_repulsion * (15.0 - distance) * 0.5

        return AttractionForce(
            force_type=ForceType.TEAMMATE,
            magnitude=magnitude,
            direction=direction,
            source=teammate.position,
            target=defender.position,
        )

    def _calculate_line_force(
        self,
        defender: Player,
        line_x: float,
    ) -> Optional[AttractionForce]:
        """
        Calculate force to maintain defensive line.

        This keeps defenders aligned horizontally.
        """
        dx = line_x - defender.position.x

        if abs(dx) < 0.5:
            return None

        direction = np.array([np.sign(dx), 0.0])
        magnitude = self.line_weight * min(abs(dx), 10.0)

        return AttractionForce(
            force_type=ForceType.LINE,
            magnitude=magnitude,
            direction=direction,
            source=Position(line_x, defender.position.y),
            target=defender.position,
        )

    def calculate_equilibrium_position(
        self,
        defender: Player,
        forces: List[AttractionForce],
        iterations: int = 10,
        step_size: float = 0.5,
    ) -> Position:
        """
        Calculate equilibrium position where forces balance.

        Uses iterative relaxation to find stable position.
        """
        current = defender.position.to_array().astype(float)

        for _ in range(iterations):
            # Sum all force vectors
            net_force = np.zeros(2)
            for force in forces:
                net_force += force.vector

            # Move in direction of net force
            if np.linalg.norm(net_force) > 0.01:
                movement = step_size * net_force / np.linalg.norm(net_force)
                current += movement

            # Clamp to pitch boundaries
            current[0] = np.clip(current[0], -HALF_LENGTH, HALF_LENGTH)
            current[1] = np.clip(current[1], -HALF_WIDTH, HALF_WIDTH)

        return Position(float(current[0]), float(current[1]))

    def calculate_team_shape(
        self,
        defenders: List[Player],
        ball_position: Position,
        opponents: List[Player],
    ) -> DefensiveShape:
        """
        Calculate the collective defensive shape.

        This shows the emergent formation from all individual forces.
        """
        ideal_positions = {}

        for defender in defenders:
            # Get all teammates except this defender
            teammates = [d for d in defenders if d.id != defender.id]

            # Calculate forces
            forces = self.calculate_forces(
                defender=defender,
                ball_position=ball_position,
                teammates=teammates,
                opponents=opponents,
            )

            # Find equilibrium position
            ideal_pos = self.calculate_equilibrium_position(defender, forces)
            ideal_positions[defender.id] = ideal_pos

        # Calculate shape metrics
        positions_list = list(ideal_positions.values())

        if len(positions_list) < 2:
            return DefensiveShape(
                positions=ideal_positions,
                compactness=0.0,
                depth=0.0,
                width=0.0,
                center_of_mass=positions_list[0] if positions_list else Position(0, 0),
                line_heights=[],
            )

        # Compactness: average pairwise distance
        total_dist = 0.0
        pairs = 0
        for i, p1 in enumerate(positions_list):
            for p2 in positions_list[i+1:]:
                total_dist += p1.distance_to(p2)
                pairs += 1
        compactness = total_dist / pairs if pairs > 0 else 0.0

        # Depth: x-range
        x_values = [p.x for p in positions_list]
        depth = max(x_values) - min(x_values)

        # Width: y-range
        y_values = [p.y for p in positions_list]
        width = max(y_values) - min(y_values)

        # Center of mass
        com_x = sum(p.x for p in positions_list) / len(positions_list)
        com_y = sum(p.y for p in positions_list) / len(positions_list)
        center = Position(com_x, com_y)

        # Infer line heights (cluster x-positions)
        sorted_x = sorted(x_values)
        lines = []
        current_line = [sorted_x[0]]
        for x in sorted_x[1:]:
            if x - current_line[-1] > 8:  # New line if >8m gap
                lines.append(sum(current_line) / len(current_line))
                current_line = []
            current_line.append(x)
        if current_line:
            lines.append(sum(current_line) / len(current_line))

        return DefensiveShape(
            positions=ideal_positions,
            compactness=compactness,
            depth=depth,
            width=width,
            center_of_mass=center,
            line_heights=lines,
        )


class CoverShadowCalculator:
    """
    Calculate cover shadows - areas blocked by defenders.

    A cover shadow is the area behind a defender (from ball's perspective)
    that is effectively "covered" - passing into this area is risky.
    """

    def __init__(
        self,
        geometry: Optional[PitchGeometry] = None,
        shadow_angle: float = 15.0,  # degrees
        shadow_length: float = 20.0,  # meters
    ):
        self.geometry = geometry or PitchGeometry()
        self.shadow_angle = np.radians(shadow_angle)
        self.shadow_length = shadow_length

    def calculate_shadow(
        self,
        defender: Position,
        ball_position: Position,
    ) -> Tuple[Position, Position, Position]:
        """
        Calculate the triangular cover shadow behind a defender.

        Returns three points defining the shadow triangle:
        - defender position
        - left edge of shadow
        - right edge of shadow
        """
        # Direction from ball to defender
        to_defender = np.array([
            defender.x - ball_position.x,
            defender.y - ball_position.y,
        ])

        dist = np.linalg.norm(to_defender)
        if dist < 0.1:
            return defender, defender, defender

        direction = to_defender / dist
        angle = np.arctan2(direction[1], direction[0])

        # Shadow extends beyond defender
        shadow_end_center = Position(
            defender.x + self.shadow_length * direction[0],
            defender.y + self.shadow_length * direction[1],
        )

        # Calculate shadow edges
        left_angle = angle + self.shadow_angle
        right_angle = angle - self.shadow_angle

        left_edge = Position(
            defender.x + self.shadow_length * np.cos(left_angle),
            defender.y + self.shadow_length * np.sin(left_angle),
        )

        right_edge = Position(
            defender.x + self.shadow_length * np.cos(right_angle),
            defender.y + self.shadow_length * np.sin(right_angle),
        )

        return defender, left_edge, right_edge

    def is_in_shadow(
        self,
        point: Position,
        defender: Position,
        ball_position: Position,
    ) -> bool:
        """
        Check if a point is within a defender's cover shadow.
        """
        # Direction from ball to defender
        ball_to_defender = np.array([
            defender.x - ball_position.x,
            defender.y - ball_position.y,
        ])

        ball_to_defender_dist = np.linalg.norm(ball_to_defender)
        if ball_to_defender_dist < 0.1:
            return False

        # Direction from ball to point
        ball_to_point = np.array([
            point.x - ball_position.x,
            point.y - ball_position.y,
        ])

        ball_to_point_dist = np.linalg.norm(ball_to_point)
        if ball_to_point_dist < 0.1:
            return False

        # Normalize
        dir_defender = ball_to_defender / ball_to_defender_dist
        dir_point = ball_to_point / ball_to_point_dist

        # Check angle between directions
        cos_angle = np.dot(dir_defender, dir_point)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # Point is in shadow if:
        # 1. Within shadow angle
        # 2. Beyond defender (further from ball)
        return (
            angle < self.shadow_angle and
            ball_to_point_dist > ball_to_defender_dist
        )

    def calculate_total_coverage(
        self,
        defenders: List[Position],
        ball_position: Position,
        grid_resolution: float = 2.0,
    ) -> float:
        """
        Calculate total area covered by all shadows.

        Returns coverage as a fraction of the defensive half.
        """
        # Create grid over defensive half
        x_range = np.arange(-HALF_LENGTH, 0, grid_resolution)
        y_range = np.arange(-HALF_WIDTH, HALF_WIDTH, grid_resolution)

        covered = 0
        total = 0

        for x in x_range:
            for y in y_range:
                point = Position(x, y)
                total += 1

                for defender_pos in defenders:
                    if self.is_in_shadow(point, defender_pos, ball_position):
                        covered += 1
                        break

        return covered / total if total > 0 else 0.0
