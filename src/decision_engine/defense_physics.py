"""
Defensive Physics Model

Defensive behavior is modeled using attraction-based principles.
Defenders are pulled toward:
- Ball (primary attraction)
- Goal (protective attraction)
- Key spaces (lane coverage, zonal responsibility)
- Opponents (marking)

NOTE: This version uses pure Python math (no numpy) for portability.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import math

from .pitch_geometry import (
    Position,
    Velocity,
    PitchGeometry,
    HALF_LENGTH,
    HALF_WIDTH,
    _norm,
    _dot,
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
    direction: List[float]  # Unit vector of force direction
    source: Position        # Where the force originates
    target: Position        # The defender being affected

    @property
    def vector(self) -> List[float]:
        """Force as a vector."""
        return [self.magnitude * self.direction[0], self.magnitude * self.direction[1]]

    def to_position_offset(self) -> Position:
        """Convert force to position offset."""
        vec = self.vector
        return Position(vec[0], vec[1])


@dataclass
class DefensiveShape:
    """
    The collective shape formed by defenders under force equilibrium.
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

    Simplified version using pure Python math.
    """

    def __init__(
        self,
        geometry: Optional[PitchGeometry] = None,
        ball_weight: float = 1.0,
        goal_weight: float = 0.6,
        zone_weight: float = 0.4,
        opponent_weight: float = 0.8,
        teammate_weight: float = 0.3,
        line_weight: float = 0.5,
    ):
        self.geometry = geometry or PitchGeometry()
        self.weights = {
            ForceType.BALL: ball_weight,
            ForceType.GOAL: goal_weight,
            ForceType.ZONE: zone_weight,
            ForceType.OPPONENT: opponent_weight,
            ForceType.TEAMMATE: teammate_weight,
            ForceType.LINE: line_weight,
        }

    def calculate_forces(
        self,
        defender: Player,
        ball_position: Position,
        teammates: List[Player],
        opponents: List[Player],
        assigned_zone: Optional[Position] = None,
        line_height: Optional[float] = None,
    ) -> List[AttractionForce]:
        """
        Calculate all forces acting on a defender.
        """
        forces = []

        # Ball attraction
        ball_force = self._calculate_ball_attraction(defender, ball_position)
        if ball_force:
            forces.append(ball_force)

        # Goal attraction
        goal_force = self._calculate_goal_attraction(defender)
        if goal_force:
            forces.append(goal_force)

        return forces

    def _calculate_ball_attraction(
        self,
        defender: Player,
        ball_position: Position,
    ) -> Optional[AttractionForce]:
        """Calculate attraction toward ball."""
        to_ball = [
            ball_position.x - defender.position.x,
            ball_position.y - defender.position.y
        ]
        distance = _norm(to_ball)

        if distance < 0.1:
            return None

        direction = [to_ball[0] / distance, to_ball[1] / distance]

        # Force falls off with distance
        magnitude = self.weights[ForceType.BALL] * (1.0 / (1.0 + distance / 20.0))

        return AttractionForce(
            force_type=ForceType.BALL,
            magnitude=magnitude,
            direction=direction,
            source=ball_position,
            target=defender.position,
        )

    def _calculate_goal_attraction(
        self,
        defender: Player,
    ) -> Optional[AttractionForce]:
        """Calculate attraction toward own goal."""
        goal = self.geometry.defending_goal

        to_goal = [
            goal.x - defender.position.x,
            goal.y - defender.position.y
        ]
        distance = _norm(to_goal)

        if distance < 0.1:
            return None

        direction = [to_goal[0] / distance, to_goal[1] / distance]
        magnitude = self.weights[ForceType.GOAL] * 0.5

        return AttractionForce(
            force_type=ForceType.GOAL,
            magnitude=magnitude,
            direction=direction,
            source=goal,
            target=defender.position,
        )

    def calculate_equilibrium_position(
        self,
        defender: Player,
        forces: List[AttractionForce],
        max_iterations: int = 10,
        step_size: float = 1.0,
    ) -> Position:
        """
        Find equilibrium position where forces balance.

        Simplified gradient descent.
        """
        current = [defender.position.x, defender.position.y]

        for _ in range(max_iterations):
            # Sum all forces
            net_force = [0.0, 0.0]
            for force in forces:
                vec = force.vector
                net_force[0] += vec[0]
                net_force[1] += vec[1]

            # Move in direction of net force
            force_mag = _norm(net_force)
            if force_mag > 0.01:
                movement = [
                    step_size * net_force[0] / force_mag,
                    step_size * net_force[1] / force_mag
                ]
                current[0] += movement[0]
                current[1] += movement[1]

            # Clamp to pitch
            current[0] = max(-HALF_LENGTH, min(HALF_LENGTH, current[0]))
            current[1] = max(-HALF_WIDTH, min(HALF_WIDTH, current[1]))

        return Position(current[0], current[1])


class CoverShadowCalculator:
    """
    Calculate cover shadows - areas blocked by defenders.

    Simplified version.
    """

    def __init__(
        self,
        geometry: Optional[PitchGeometry] = None,
        shadow_angle: float = 30.0,
        shadow_length: float = 15.0,
    ):
        self.geometry = geometry or PitchGeometry()
        self.shadow_angle = math.radians(shadow_angle)
        self.shadow_length = shadow_length

    def is_in_shadow(
        self,
        point: Position,
        ball_position: Position,
        defender: Position,
    ) -> bool:
        """
        Check if a point is in the shadow of a defender.

        Shadow extends from ball through defender in a cone.
        """
        # Vector from ball to defender
        ball_to_defender = [
            defender.x - ball_position.x,
            defender.y - ball_position.y
        ]
        ball_to_defender_dist = _norm(ball_to_defender)

        if ball_to_defender_dist < 0.1:
            return False

        # Vector from ball to point
        ball_to_point = [
            point.x - ball_position.x,
            point.y - ball_position.y
        ]
        ball_to_point_dist = _norm(ball_to_point)

        if ball_to_point_dist < 0.1:
            return False

        # Point must be further than defender
        if ball_to_point_dist < ball_to_defender_dist:
            return False

        # Normalize directions
        dir_defender = [ball_to_defender[0] / ball_to_defender_dist,
                       ball_to_defender[1] / ball_to_defender_dist]
        dir_point = [ball_to_point[0] / ball_to_point_dist,
                    ball_to_point[1] / ball_to_point_dist]

        # Angle between directions
        cos_angle = _dot(dir_defender, dir_point)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp for numerical stability
        angle = math.acos(cos_angle)

        # In shadow if angle is small enough
        return angle < self.shadow_angle
