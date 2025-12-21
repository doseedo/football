"""3D football pitch field model with comprehensive keypoint definitions.

This module provides a complete geometric model of a football pitch with 57+ keypoints
including corners, line intersections, penalty box corners, goal areas, and center circle points.
Follows FIFA standard pitch dimensions (105m x 68m).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


@dataclass
class Keypoint3D:
    """Represents a 3D keypoint on the football pitch."""

    name: str
    x: float  # World X coordinate (meters)
    y: float  # World Y coordinate (meters)
    z: float = 0.0  # World Z coordinate (meters, typically 0 for pitch surface)
    category: str = "general"  # Category: corner, line, box, circle, goal
    confidence: float = 1.0  # Confidence in detection (0-1)

    @property
    def world_coords(self) -> Tuple[float, float]:
        """Get 2D world coordinates (x, y)."""
        return (self.x, self.y)

    @property
    def world_coords_3d(self) -> Tuple[float, float, float]:
        """Get 3D world coordinates (x, y, z)."""
        return (self.x, self.y, self.z)


class FootballPitchModel:
    """Complete 3D model of a football pitch with all standard keypoints.

    The coordinate system places the origin at the top-left corner:
    - X-axis runs along the length (0 to 105m)
    - Y-axis runs along the width (0 to 68m)
    - Z-axis points upward (0 for pitch surface)
    """

    # FIFA standard pitch dimensions (in meters)
    DEFAULT_LENGTH = 105.0
    DEFAULT_WIDTH = 68.0

    # Pitch markings dimensions
    PENALTY_BOX_LENGTH = 16.5
    PENALTY_BOX_WIDTH = 40.3
    GOAL_AREA_LENGTH = 5.5
    GOAL_AREA_WIDTH = 18.32
    CENTER_CIRCLE_RADIUS = 9.15
    PENALTY_SPOT_DISTANCE = 11.0
    GOAL_WIDTH = 7.32
    GOAL_HEIGHT = 2.44

    def __init__(
        self,
        length: float = DEFAULT_LENGTH,
        width: float = DEFAULT_WIDTH,
        include_3d_points: bool = False
    ):
        """Initialize football pitch model.

        Args:
            length: Pitch length in meters (default 105m)
            width: Pitch width in meters (default 68m)
            include_3d_points: Include 3D points like goal posts
        """
        self.length = length
        self.width = width
        self.include_3d = include_3d_points

        self.keypoints: Dict[str, Keypoint3D] = {}
        self._build_keypoints()

        logger.info(f"Initialized pitch model: {self.length}m x {self.width}m with {len(self.keypoints)} keypoints")

    def _build_keypoints(self):
        """Build all keypoints for the football pitch."""
        # Calculate derived dimensions
        half_length = self.length / 2
        half_width = self.width / 2
        half_penalty_box_width = self.PENALTY_BOX_WIDTH / 2
        half_goal_area_width = self.GOAL_AREA_WIDTH / 2
        half_goal_width = self.GOAL_WIDTH / 2

        # --- CORNER POINTS (4 points) ---
        self._add_keypoint("corner_tl", 0, 0, category="corner")
        self._add_keypoint("corner_tr", self.length, 0, category="corner")
        self._add_keypoint("corner_bl", 0, self.width, category="corner")
        self._add_keypoint("corner_br", self.length, self.width, category="corner")

        # --- HALFWAY LINE POINTS (7 points) ---
        self._add_keypoint("halfway_top", half_length, 0, category="line")
        self._add_keypoint("halfway_bottom", half_length, self.width, category="line")
        self._add_keypoint("center_spot", half_length, half_width, category="circle")

        # Center circle intersections with halfway line
        self._add_keypoint("center_circle_top", half_length, half_width - self.CENTER_CIRCLE_RADIUS, category="circle")
        self._add_keypoint("center_circle_bottom", half_length, half_width + self.CENTER_CIRCLE_RADIUS, category="circle")
        self._add_keypoint("center_circle_left", half_length - self.CENTER_CIRCLE_RADIUS, half_width, category="circle")
        self._add_keypoint("center_circle_right", half_length + self.CENTER_CIRCLE_RADIUS, half_width, category="circle")

        # --- LEFT PENALTY BOX (8 points) ---
        left_box_x = self.PENALTY_BOX_LENGTH
        self._add_keypoint("penalty_box_left_tl", 0, half_width - half_penalty_box_width, category="box")
        self._add_keypoint("penalty_box_left_tr", left_box_x, half_width - half_penalty_box_width, category="box")
        self._add_keypoint("penalty_box_left_bl", 0, half_width + half_penalty_box_width, category="box")
        self._add_keypoint("penalty_box_left_br", left_box_x, half_width + half_penalty_box_width, category="box")

        # Left penalty spot and arc
        self._add_keypoint("penalty_spot_left", self.PENALTY_SPOT_DISTANCE, half_width, category="box")

        # Penalty arc intersections (3 points on the arc)
        penalty_arc_radius = 9.15
        arc_angle = np.arcsin(half_penalty_box_width / penalty_arc_radius)
        self._add_keypoint("penalty_arc_left_top",
                          self.PENALTY_SPOT_DISTANCE + penalty_arc_radius * np.cos(arc_angle),
                          half_width - penalty_arc_radius * np.sin(arc_angle),
                          category="box")
        self._add_keypoint("penalty_arc_left_bottom",
                          self.PENALTY_SPOT_DISTANCE + penalty_arc_radius * np.cos(arc_angle),
                          half_width + penalty_arc_radius * np.sin(arc_angle),
                          category="box")
        self._add_keypoint("penalty_arc_left_center",
                          self.PENALTY_SPOT_DISTANCE + penalty_arc_radius,
                          half_width,
                          category="box")

        # --- RIGHT PENALTY BOX (8 points) ---
        right_box_x = self.length - self.PENALTY_BOX_LENGTH
        self._add_keypoint("penalty_box_right_tl", right_box_x, half_width - half_penalty_box_width, category="box")
        self._add_keypoint("penalty_box_right_tr", self.length, half_width - half_penalty_box_width, category="box")
        self._add_keypoint("penalty_box_right_bl", right_box_x, half_width + half_penalty_box_width, category="box")
        self._add_keypoint("penalty_box_right_br", self.length, half_width + half_penalty_box_width, category="box")

        # Right penalty spot and arc
        self._add_keypoint("penalty_spot_right", self.length - self.PENALTY_SPOT_DISTANCE, half_width, category="box")

        self._add_keypoint("penalty_arc_right_top",
                          self.length - (self.PENALTY_SPOT_DISTANCE + penalty_arc_radius * np.cos(arc_angle)),
                          half_width - penalty_arc_radius * np.sin(arc_angle),
                          category="box")
        self._add_keypoint("penalty_arc_right_bottom",
                          self.length - (self.PENALTY_SPOT_DISTANCE + penalty_arc_radius * np.cos(arc_angle)),
                          half_width + penalty_arc_radius * np.sin(arc_angle),
                          category="box")
        self._add_keypoint("penalty_arc_right_center",
                          self.length - (self.PENALTY_SPOT_DISTANCE + penalty_arc_radius),
                          half_width,
                          category="box")

        # --- LEFT GOAL AREA (4 points) ---
        left_goal_area_x = self.GOAL_AREA_LENGTH
        self._add_keypoint("goal_area_left_tl", 0, half_width - half_goal_area_width, category="goal")
        self._add_keypoint("goal_area_left_tr", left_goal_area_x, half_width - half_goal_area_width, category="goal")
        self._add_keypoint("goal_area_left_bl", 0, half_width + half_goal_area_width, category="goal")
        self._add_keypoint("goal_area_left_br", left_goal_area_x, half_width + half_goal_area_width, category="goal")

        # --- RIGHT GOAL AREA (4 points) ---
        right_goal_area_x = self.length - self.GOAL_AREA_LENGTH
        self._add_keypoint("goal_area_right_tl", right_goal_area_x, half_width - half_goal_area_width, category="goal")
        self._add_keypoint("goal_area_right_tr", self.length, half_width - half_goal_area_width, category="goal")
        self._add_keypoint("goal_area_right_bl", right_goal_area_x, half_width + half_goal_area_width, category="goal")
        self._add_keypoint("goal_area_right_br", self.length, half_width + half_goal_area_width, category="goal")

        # --- GOAL POSTS (8 points - 2D and 3D) ---
        # Left goal
        self._add_keypoint("goal_left_top", 0, half_width - half_goal_width, category="goal")
        self._add_keypoint("goal_left_bottom", 0, half_width + half_goal_width, category="goal")

        # Right goal
        self._add_keypoint("goal_right_top", self.length, half_width - half_goal_width, category="goal")
        self._add_keypoint("goal_right_bottom", self.length, half_width + half_goal_width, category="goal")

        if self.include_3d:
            # 3D goal posts (elevated points)
            self._add_keypoint("goal_left_top_post", 0, half_width - half_goal_width, self.GOAL_HEIGHT, category="goal")
            self._add_keypoint("goal_left_bottom_post", 0, half_width + half_goal_width, self.GOAL_HEIGHT, category="goal")
            self._add_keypoint("goal_right_top_post", self.length, half_width - half_goal_width, self.GOAL_HEIGHT, category="goal")
            self._add_keypoint("goal_right_bottom_post", self.length, half_width + half_goal_width, self.GOAL_HEIGHT, category="goal")

        # --- ADDITIONAL CENTER CIRCLE POINTS (8 points for better circle coverage) ---
        for i, angle in enumerate([45, 90, 135, 225, 270, 315]):
            angle_rad = np.radians(angle)
            x = half_length + self.CENTER_CIRCLE_RADIUS * np.cos(angle_rad)
            y = half_width + self.CENTER_CIRCLE_RADIUS * np.sin(angle_rad)
            self._add_keypoint(f"center_circle_{angle}", x, y, category="circle")

        # --- SIDELINE MIDPOINTS (2 points) ---
        self._add_keypoint("sideline_top_mid", half_length, 0, category="line")
        self._add_keypoint("sideline_bottom_mid", half_length, self.width, category="line")

        # --- GOAL LINE MIDPOINTS (2 points) ---
        self._add_keypoint("goal_line_left_mid", 0, half_width, category="goal")
        self._add_keypoint("goal_line_right_mid", self.length, half_width, category="goal")

    def _add_keypoint(self, name: str, x: float, y: float, z: float = 0.0, category: str = "general"):
        """Add a keypoint to the model."""
        self.keypoints[name] = Keypoint3D(
            name=name,
            x=x,
            y=y,
            z=z,
            category=category
        )

    def get_keypoint(self, name: str) -> Optional[Keypoint3D]:
        """Get a keypoint by name."""
        return self.keypoints.get(name)

    def get_keypoints_by_category(self, category: str) -> List[Keypoint3D]:
        """Get all keypoints in a specific category."""
        return [kp for kp in self.keypoints.values() if kp.category == category]

    def get_all_keypoints(self) -> List[Keypoint3D]:
        """Get all keypoints."""
        return list(self.keypoints.values())

    def get_world_coords(self, name: str) -> Optional[Tuple[float, float]]:
        """Get 2D world coordinates for a keypoint by name."""
        kp = self.get_keypoint(name)
        return kp.world_coords if kp else None

    def get_world_coords_3d(self, name: str) -> Optional[Tuple[float, float, float]]:
        """Get 3D world coordinates for a keypoint by name."""
        kp = self.get_keypoint(name)
        return kp.world_coords_3d if kp else None

    def get_keypoint_pairs(self) -> List[Tuple[str, Tuple[float, float]]]:
        """Get list of (name, coords) pairs for all keypoints."""
        return [(kp.name, kp.world_coords) for kp in self.keypoints.values()]

    def get_world_coords_array(self) -> np.ndarray:
        """Get all world coordinates as Nx2 numpy array."""
        coords = [kp.world_coords for kp in self.keypoints.values()]
        return np.array(coords, dtype=np.float32)

    def get_keypoint_names(self) -> List[str]:
        """Get list of all keypoint names."""
        return list(self.keypoints.keys())

    def is_point_in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if a point is within pitch bounds (with optional margin)."""
        return (
            -margin <= x <= self.length + margin and
            -margin <= y <= self.width + margin
        )

    def visualize_keypoints(self, ax=None):
        """Visualize all keypoints on a matplotlib axis.

        Args:
            ax: Matplotlib axis (if None, creates new figure)
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Draw pitch outline
        ax.plot([0, self.length, self.length, 0, 0],
               [0, 0, self.width, self.width, 0],
               'k-', linewidth=2, label='Pitch boundary')

        # Draw halfway line
        half_length = self.length / 2
        ax.plot([half_length, half_length], [0, self.width], 'k-', linewidth=1)

        # Color map for categories
        category_colors = {
            'corner': 'red',
            'line': 'blue',
            'box': 'green',
            'circle': 'orange',
            'goal': 'purple',
            'general': 'black'
        }

        # Plot keypoints by category
        for category in set(kp.category for kp in self.keypoints.values()):
            kps = self.get_keypoints_by_category(category)
            xs = [kp.x for kp in kps]
            ys = [kp.y for kp in kps]
            color = category_colors.get(category, 'black')
            ax.scatter(xs, ys, c=color, s=50, label=category, alpha=0.7, zorder=5)

        ax.set_xlim(-5, self.length + 5)
        ax.set_ylim(-5, self.width + 5)
        ax.set_aspect('equal')
        ax.set_xlabel('Length (m)')
        ax.set_ylabel('Width (m)')
        ax.set_title(f'Football Pitch Keypoints ({len(self.keypoints)} total)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if ax is None:
            plt.tight_layout()
            plt.show()

    def __len__(self) -> int:
        """Return number of keypoints."""
        return len(self.keypoints)

    def __repr__(self) -> str:
        return f"FootballPitchModel({self.length}m x {self.width}m, {len(self.keypoints)} keypoints)"


def create_standard_pitch(include_3d: bool = False) -> FootballPitchModel:
    """Create a standard FIFA regulation pitch model.

    Args:
        include_3d: Include 3D points like goal posts

    Returns:
        FootballPitchModel with standard dimensions
    """
    return FootballPitchModel(
        length=FootballPitchModel.DEFAULT_LENGTH,
        width=FootballPitchModel.DEFAULT_WIDTH,
        include_3d=include_3d
    )
