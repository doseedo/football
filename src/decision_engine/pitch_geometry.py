"""
Pitch geometry constants and spatial utilities.

Standard pitch dimensions (UEFA/FIFA):
- Length: 105m
- Width: 68m
- Origin: (0, 0) at center of pitch
- Attacking direction: +x (toward goal at x=52.5)

NOTE: This module uses pure Python math (no numpy) for portability.
"""

from dataclasses import dataclass
from typing import Tuple, List
import math

# Standard pitch dimensions in meters
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Half dimensions (from center)
HALF_LENGTH = PITCH_LENGTH / 2  # 52.5m
HALF_WIDTH = PITCH_WIDTH / 2    # 34m

# Goal dimensions
GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44  # Not used in 2D but relevant for 3D extensions

# Key zones
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.32

# Derived positions (origin at center)
GOAL_LEFT_X = -HALF_LENGTH   # -52.5 (defending goal)
GOAL_RIGHT_X = HALF_LENGTH   # +52.5 (attacking goal)
GOAL_POST_Y = GOAL_WIDTH / 2  # 3.66m from center


@dataclass
class Position:
    """2D position on the pitch in meters."""
    x: float
    y: float

    def to_list(self) -> List[float]:
        return [self.x, self.y]

    @classmethod
    def from_list(cls, arr: List[float]) -> "Position":
        return cls(x=float(arr[0]), y=float(arr[1]))

    def distance_to(self, other: "Position") -> float:
        """Euclidean distance to another position."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def angle_to(self, other: "Position") -> float:
        """Angle in radians from this position to another."""
        return math.atan2(other.y - self.y, other.x - self.x)

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Position") -> "Position":
        return Position(self.x - other.x, self.y - other.y)


@dataclass
class Velocity:
    """2D velocity vector in m/s."""
    vx: float
    vy: float

    @property
    def speed(self) -> float:
        """Magnitude of velocity."""
        return math.sqrt(self.vx**2 + self.vy**2)

    @property
    def direction(self) -> float:
        """Direction in radians."""
        return math.atan2(self.vy, self.vx)

    def to_list(self) -> List[float]:
        return [self.vx, self.vy]

    @classmethod
    def from_list(cls, arr: List[float]) -> "Velocity":
        return cls(vx=float(arr[0]), vy=float(arr[1]))

    @classmethod
    def zero(cls) -> "Velocity":
        return cls(vx=0.0, vy=0.0)


def _dot(v1: List[float], v2: List[float]) -> float:
    """Dot product of two vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1]


def _norm(v: List[float]) -> float:
    """Magnitude of a vector."""
    return math.sqrt(v[0]**2 + v[1]**2)


def _clip(value: float, min_val: float, max_val: float) -> float:
    """Clip value to range."""
    return max(min_val, min(max_val, value))


class PitchGeometry:
    """
    Pitch geometry calculator with coordinate utilities.

    Coordinate system:
    - Origin (0, 0) at center of pitch
    - +x direction is the attacking direction (toward right goal)
    - +y is toward the top of the pitch (when viewed from above)
    - All units in meters
    """

    def __init__(
        self,
        length: float = PITCH_LENGTH,
        width: float = PITCH_WIDTH,
        attacking_goal_x: float = HALF_LENGTH,
    ):
        self.length = length
        self.width = width
        self.half_length = length / 2
        self.half_width = width / 2
        self.attacking_goal_x = attacking_goal_x
        self.defending_goal_x = -attacking_goal_x

        # Goal center positions
        self.attacking_goal = Position(self.attacking_goal_x, 0.0)
        self.defending_goal = Position(self.defending_goal_x, 0.0)

    def is_on_pitch(self, pos: Position, margin: float = 0.0) -> bool:
        """Check if position is within pitch boundaries."""
        return (
            abs(pos.x) <= self.half_length + margin and
            abs(pos.y) <= self.half_width + margin
        )

    def distance_to_attacking_goal(self, pos: Position) -> float:
        """Distance from position to center of attacking goal."""
        return pos.distance_to(self.attacking_goal)

    def distance_to_defending_goal(self, pos: Position) -> float:
        """Distance from position to center of defending goal."""
        return pos.distance_to(self.defending_goal)

    def angle_to_goal(self, pos: Position, attacking: bool = True) -> float:
        """
        Calculate the visible angle to goal from a position.

        Returns the angle in radians subtended by the goal posts.
        """
        goal_x = self.attacking_goal_x if attacking else self.defending_goal_x

        # Goal post positions
        left_post = Position(goal_x, -GOAL_POST_Y)
        right_post = Position(goal_x, GOAL_POST_Y)

        # Angles to each post
        angle_left = pos.angle_to(left_post)
        angle_right = pos.angle_to(right_post)

        # Return absolute angle difference
        return abs(angle_right - angle_left)

    def is_in_penalty_area(self, pos: Position, attacking: bool = True) -> bool:
        """Check if position is in the penalty area."""
        goal_x = self.attacking_goal_x if attacking else self.defending_goal_x

        if attacking:
            x_in = pos.x >= goal_x - PENALTY_AREA_LENGTH
        else:
            x_in = pos.x <= goal_x + PENALTY_AREA_LENGTH

        y_in = abs(pos.y) <= PENALTY_AREA_WIDTH / 2

        return x_in and y_in

    def get_zone(self, pos: Position) -> str:
        """
        Get the zone name for a position.

        Returns one of: 'defensive_third', 'middle_third', 'attacking_third'
        with sub-zones for left/center/right.
        """
        # Thirds along x-axis
        if pos.x < -self.half_length / 3:
            x_zone = "defensive"
        elif pos.x > self.half_length / 3:
            x_zone = "attacking"
        else:
            x_zone = "middle"

        # Thirds along y-axis
        if pos.y < -self.half_width / 3:
            y_zone = "left"
        elif pos.y > self.half_width / 3:
            y_zone = "right"
        else:
            y_zone = "center"

        return f"{x_zone}_{y_zone}"

    def normalize_position(self, pos: Position) -> Position:
        """
        Normalize position to [0, 1] range for each axis.

        (0, 0) = defending corner
        (1, 1) = attacking corner
        """
        norm_x = (pos.x + self.half_length) / self.length
        norm_y = (pos.y + self.half_width) / self.width
        return Position(norm_x, norm_y)

    def time_to_reach(
        self,
        start: Position,
        target: Position,
        speed: float,
        reaction_time: float = 0.0,
    ) -> float:
        """
        Calculate time for a player to reach a target position.

        Args:
            start: Starting position
            target: Target position
            speed: Player max speed in m/s
            reaction_time: Delay before movement starts

        Returns:
            Time in seconds to reach target
        """
        distance = start.distance_to(target)
        if speed <= 0:
            return float('inf')
        return reaction_time + distance / speed

    def line_of_intervention(
        self,
        ball_pos: Position,
        goal_pos: Position,
    ) -> Tuple[Position, Position]:
        """
        Calculate the direct line from ball to goal.

        Returns start and end positions of the line.
        This represents the most dangerous attacking lane.
        """
        return ball_pos, goal_pos

    def perpendicular_distance_to_line(
        self,
        point: Position,
        line_start: Position,
        line_end: Position,
    ) -> float:
        """
        Calculate perpendicular distance from a point to a line segment.

        Used to determine how far a defender is from the ball-to-goal line.
        """
        # Line vector
        line_vec = [
            line_end.x - line_start.x,
            line_end.y - line_start.y
        ]
        line_len = _norm(line_vec)

        if line_len == 0:
            return point.distance_to(line_start)

        # Normalize line vector
        line_unit = [line_vec[0] / line_len, line_vec[1] / line_len]

        # Vector from line start to point
        point_vec = [
            point.x - line_start.x,
            point.y - line_start.y
        ]

        # Project point onto line
        projection_length = _dot(point_vec, line_unit)

        # Clamp to line segment
        projection_length = max(0, min(line_len, projection_length))

        # Closest point on line
        closest = Position(
            line_start.x + projection_length * line_unit[0],
            line_start.y + projection_length * line_unit[1]
        )

        return point.distance_to(closest)
