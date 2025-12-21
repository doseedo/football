"""Advanced tactical metrics using spatial analysis and GNN embeddings."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from scipy.spatial import Voronoi, ConvexHull
from scipy.spatial.distance import cdist
from torch_geometric.data import Data


@dataclass
class TacticalMetrics:
    """Container for all tactical metrics."""
    # Team structure
    defensive_line_height: float  # meters from own goal
    offensive_line_height: float  # meters from own goal
    team_length: float  # distance between defensive and offensive lines
    team_width: float  # lateral spread
    team_compactness: float  # average distance between players
    team_centroid: Tuple[float, float]  # team center of mass

    # Space control
    space_control_ratio: float  # possession team's controlled space
    voronoi_areas: Dict[int, float]  # space controlled by each player

    # Pressure and passing
    pressure_on_ball: float  # defensive pressure intensity (0-1)
    passing_lane_count: int  # number of available passing lanes
    passing_lane_quality: float  # average quality of passing lanes (0-1)
    progressive_pass_options: int  # passes that move ball forward

    # Expected threat
    expected_threat: float  # xT value of current ball position
    threatening_players: int  # attackers in dangerous positions

    # Pitch control
    pitch_control_possession: float  # % of pitch controlled by possession team
    high_pressure_zones: List[Tuple[float, float]]  # coordinates of pressure zones

    timestamp: float


class TacticalMetricsCalculator:
    """
    Calculate advanced tactical metrics from tracking data.

    Combines geometric analysis, spatial statistics, and GNN features.
    """

    # Pitch dimensions
    PITCH_LENGTH = 105.0  # meters
    PITCH_WIDTH = 68.0  # meters

    # Zone definitions
    DEFENSIVE_ZONE = 35.0
    MIDDLE_ZONE = 35.0
    ATTACKING_ZONE = 35.0

    def __init__(self, use_voronoi: bool = True):
        """
        Initialize metrics calculator.

        Args:
            use_voronoi: Use Voronoi tessellation for space control
        """
        self.use_voronoi = use_voronoi
        self.xT_grid = self._initialize_xT_grid()

    def calculate_all_metrics(self,
                             graph: Data,
                             ball_position: Optional[Tuple[float, float]] = None,
                             possession_team: int = 0) -> TacticalMetrics:
        """
        Calculate all tactical metrics from graph.

        Args:
            graph: PyTorch Geometric Data object
            ball_position: Ball (x, y) position in meters
            possession_team: Team with possession (0 or 1)

        Returns:
            TacticalMetrics object
        """
        # Extract positions
        positions = graph.x[:, :2].cpu().numpy()
        teams = graph.teams.cpu().numpy()

        # Denormalize positions
        positions[:, 0] = positions[:, 0] * self.PITCH_LENGTH - self.PITCH_LENGTH / 2
        positions[:, 1] = positions[:, 1] * self.PITCH_WIDTH - self.PITCH_WIDTH / 2

        if ball_position is None and graph.ball_position is not None:
            ball_position = (graph.ball_position[0].item(), graph.ball_position[1].item())

        # Team structure metrics
        defensive_line = self._calculate_defensive_line_height(positions, teams, possession_team)
        offensive_line = self._calculate_offensive_line_height(positions, teams, possession_team)
        team_length = self._calculate_team_length(positions, teams, possession_team)
        team_width = self._calculate_team_width(positions, teams, possession_team)
        compactness = self._calculate_compactness(positions, teams, possession_team)
        centroid = self._calculate_team_centroid(positions, teams, possession_team)

        # Space control
        space_ratio, voronoi_areas = self._calculate_space_control(positions, teams, possession_team)

        # Pressure and passing
        pressure = self._calculate_pressure_on_ball(positions, teams, ball_position, possession_team)
        lane_count, lane_quality = self._calculate_passing_lanes(positions, teams, ball_position, possession_team)
        progressive_options = self._calculate_progressive_pass_options(positions, teams, ball_position, possession_team)

        # Expected threat
        xT = self._calculate_expected_threat(ball_position) if ball_position else 0.0
        threatening = self._count_threatening_players(positions, teams, possession_team)

        # Pitch control
        pitch_control = self._calculate_pitch_control(positions, teams, possession_team)
        pressure_zones = self._identify_high_pressure_zones(positions, teams, ball_position)

        timestamp = graph.timestamp if hasattr(graph, 'timestamp') else 0.0

        return TacticalMetrics(
            defensive_line_height=defensive_line,
            offensive_line_height=offensive_line,
            team_length=team_length,
            team_width=team_width,
            team_compactness=compactness,
            team_centroid=centroid,
            space_control_ratio=space_ratio,
            voronoi_areas=voronoi_areas,
            pressure_on_ball=pressure,
            passing_lane_count=lane_count,
            passing_lane_quality=lane_quality,
            progressive_pass_options=progressive_options,
            expected_threat=xT,
            threatening_players=threatening,
            pitch_control_possession=pitch_control,
            high_pressure_zones=pressure_zones,
            timestamp=timestamp
        )

    def _calculate_defensive_line_height(self, positions: np.ndarray, teams: np.ndarray, team: int) -> float:
        """Calculate defensive line height (deepest 4 players)."""
        team_mask = teams == team
        if team_mask.sum() == 0:
            return 0.0

        team_positions = positions[team_mask]

        # Sort by x position (deepest defenders)
        if team == 0:  # Defending left goal
            sorted_x = np.sort(team_positions[:, 0])[:4]
            line = np.mean(sorted_x)
            return line + self.PITCH_LENGTH / 2
        else:  # Defending right goal
            sorted_x = np.sort(team_positions[:, 0])[-4:]
            line = np.mean(sorted_x)
            return self.PITCH_LENGTH / 2 - line

    def _calculate_offensive_line_height(self, positions: np.ndarray, teams: np.ndarray, team: int) -> float:
        """Calculate offensive line height (most advanced 3 players)."""
        team_mask = teams == team
        if team_mask.sum() == 0:
            return 0.0

        team_positions = positions[team_mask]

        # Sort by x position (most advanced attackers)
        if team == 0:  # Attacking right goal
            sorted_x = np.sort(team_positions[:, 0])[-3:]
            line = np.mean(sorted_x)
            return line + self.PITCH_LENGTH / 2
        else:  # Attacking left goal
            sorted_x = np.sort(team_positions[:, 0])[:3]
            line = np.mean(sorted_x)
            return self.PITCH_LENGTH / 2 - line

    def _calculate_team_length(self, positions: np.ndarray, teams: np.ndarray, team: int) -> float:
        """Calculate team length (distance between defensive and offensive lines)."""
        team_mask = teams == team
        if team_mask.sum() == 0:
            return 0.0

        team_positions = positions[team_mask]
        x_positions = team_positions[:, 0]

        return float(np.max(x_positions) - np.min(x_positions))

    def _calculate_team_width(self, positions: np.ndarray, teams: np.ndarray, team: int) -> float:
        """Calculate team width (lateral spread)."""
        team_mask = teams == team
        if team_mask.sum() == 0:
            return 0.0

        team_positions = positions[team_mask]
        y_positions = team_positions[:, 1]

        return float(np.max(y_positions) - np.min(y_positions))

    def _calculate_compactness(self, positions: np.ndarray, teams: np.ndarray, team: int) -> float:
        """Calculate team compactness (average distance between teammates)."""
        team_mask = teams == team
        team_positions = positions[team_mask]

        if len(team_positions) < 2:
            return 0.0

        from scipy.spatial.distance import pdist
        distances = pdist(team_positions)
        return float(np.mean(distances))

    def _calculate_team_centroid(self, positions: np.ndarray, teams: np.ndarray, team: int) -> Tuple[float, float]:
        """Calculate team center of mass."""
        team_mask = teams == team
        team_positions = positions[team_mask]

        if len(team_positions) == 0:
            return (0.0, 0.0)

        centroid = np.mean(team_positions, axis=0)
        return (float(centroid[0]), float(centroid[1]))

    def _calculate_space_control(self, positions: np.ndarray, teams: np.ndarray, possession_team: int) -> Tuple[float, Dict[int, float]]:
        """
        Calculate space control using Voronoi tessellation.

        Returns:
            space_ratio: Ratio of space controlled by possession team
            voronoi_areas: Dict mapping player index to controlled area
        """
        if not self.use_voronoi or len(positions) < 4:
            return 0.5, {}

        try:
            # Compute Voronoi diagram
            vor = Voronoi(positions)

            # Calculate areas (bounded by pitch)
            areas = {}
            total_area = self.PITCH_LENGTH * self.PITCH_WIDTH

            for i, region_idx in enumerate(vor.point_region):
                region = vor.regions[region_idx]
                if -1 in region or len(region) == 0:
                    # Unbounded region - assign average area
                    areas[i] = total_area / len(positions)
                    continue

                # Get region vertices
                vertices = vor.vertices[region]

                # Clip to pitch boundaries
                vertices = np.clip(vertices,
                                 [-self.PITCH_LENGTH/2, -self.PITCH_WIDTH/2],
                                 [self.PITCH_LENGTH/2, self.PITCH_WIDTH/2])

                # Calculate area
                if len(vertices) >= 3:
                    hull = ConvexHull(vertices)
                    areas[i] = hull.volume  # volume is area in 2D
                else:
                    areas[i] = 0.0

            # Calculate possession team's share
            possession_area = sum(areas[i] for i in range(len(teams)) if teams[i] == possession_team)
            total_calculated_area = sum(areas.values())

            if total_calculated_area > 0:
                space_ratio = possession_area / total_calculated_area
            else:
                space_ratio = 0.5

            return float(space_ratio), areas

        except Exception as e:
            # Fallback if Voronoi fails
            return 0.5, {}

    def _calculate_pressure_on_ball(self, positions: np.ndarray, teams: np.ndarray,
                                   ball_position: Optional[Tuple[float, float]],
                                   possession_team: int) -> float:
        """Calculate defensive pressure on ball carrier."""
        if ball_position is None:
            return 0.0

        ball_pos = np.array(ball_position)
        defensive_team = 1 - possession_team

        # Get defensive players
        defensive_mask = teams == defensive_team
        defensive_positions = positions[defensive_mask]

        if len(defensive_positions) == 0:
            return 0.0

        # Calculate distances to ball
        distances = np.linalg.norm(defensive_positions - ball_pos, axis=1)

        # Pressure based on closest defenders
        closest_3 = np.sort(distances)[:3]
        avg_distance = np.mean(closest_3)

        # Normalize to [0, 1] - closer = higher pressure
        pressure = max(0, 1 - avg_distance / 15.0)

        return float(pressure)

    def _calculate_passing_lanes(self, positions: np.ndarray, teams: np.ndarray,
                                 ball_position: Optional[Tuple[float, float]],
                                 possession_team: int) -> Tuple[int, float]:
        """
        Calculate available passing lanes.

        Returns:
            count: Number of open passing lanes
            quality: Average quality of passing lanes (0-1)
        """
        if ball_position is None:
            return 0, 0.0

        ball_pos = np.array(ball_position)

        # Get teammates
        team_mask = teams == possession_team
        teammate_positions = positions[team_mask]

        # Get opponents
        opponent_mask = teams == (1 - possession_team)
        opponent_positions = positions[opponent_mask]

        if len(teammate_positions) == 0:
            return 0, 0.0

        # Evaluate each potential pass
        open_lanes = 0
        lane_qualities = []

        for teammate_pos in teammate_positions:
            # Skip if too close (same player)
            if np.linalg.norm(teammate_pos - ball_pos) < 1.0:
                continue

            # Check if lane is blocked by opponents
            is_open, quality = self._evaluate_passing_lane(ball_pos, teammate_pos, opponent_positions)

            if is_open:
                open_lanes += 1
                lane_qualities.append(quality)

        avg_quality = np.mean(lane_qualities) if lane_qualities else 0.0

        return open_lanes, float(avg_quality)

    def _evaluate_passing_lane(self, start: np.ndarray, end: np.ndarray,
                               opponents: np.ndarray, threshold: float = 1.5) -> Tuple[bool, float]:
        """
        Evaluate if passing lane is open.

        Args:
            start: Pass starting position
            end: Pass target position
            opponents: Opponent positions
            threshold: Distance threshold for blocking (meters)

        Returns:
            is_open: Whether lane is open
            quality: Lane quality (0-1)
        """
        if len(opponents) == 0:
            return True, 1.0

        # Vector from start to end
        pass_vector = end - start
        pass_length = np.linalg.norm(pass_vector)

        if pass_length < 0.1:
            return False, 0.0

        pass_direction = pass_vector / pass_length

        # Check each opponent
        min_distance = float('inf')

        for opp_pos in opponents:
            # Vector from start to opponent
            to_opponent = opp_pos - start

            # Project onto pass direction
            projection = np.dot(to_opponent, pass_direction)

            # Check if opponent is along the pass
            if 0 < projection < pass_length:
                # Distance from pass line
                perpendicular_distance = np.linalg.norm(to_opponent - projection * pass_direction)

                min_distance = min(min_distance, perpendicular_distance)

        # Lane is open if min distance is above threshold
        is_open = min_distance > threshold

        # Quality based on clearance
        quality = min(min_distance / 5.0, 1.0)  # Normalize to [0, 1]

        return is_open, float(quality)

    def _calculate_progressive_pass_options(self, positions: np.ndarray, teams: np.ndarray,
                                          ball_position: Optional[Tuple[float, float]],
                                          possession_team: int) -> int:
        """Count passes that would move ball toward opponent goal."""
        if ball_position is None:
            return 0

        ball_x = ball_position[0]

        # Get teammates
        team_mask = teams == possession_team
        teammate_positions = positions[team_mask]

        # Progressive threshold (at least 10m forward)
        if possession_team == 0:  # Attacking right
            progressive_teammates = teammate_positions[teammate_positions[:, 0] > ball_x + 10]
        else:  # Attacking left
            progressive_teammates = teammate_positions[teammate_positions[:, 0] < ball_x - 10]

        return len(progressive_teammates)

    def _calculate_expected_threat(self, ball_position: Optional[Tuple[float, float]]) -> float:
        """
        Calculate expected threat (xT) of current position.

        Uses simplified xT grid based on position.
        """
        if ball_position is None:
            return 0.0

        x, y = ball_position

        # Normalize to grid coordinates
        x_grid = int((x + self.PITCH_LENGTH / 2) / self.PITCH_LENGTH * 12)
        y_grid = int((y + self.PITCH_WIDTH / 2) / self.PITCH_WIDTH * 8)

        # Clip to grid bounds
        x_grid = np.clip(x_grid, 0, 11)
        y_grid = np.clip(y_grid, 0, 7)

        return float(self.xT_grid[y_grid, x_grid])

    def _initialize_xT_grid(self) -> np.ndarray:
        """
        Initialize expected threat grid (12x8).

        Higher values closer to goal.
        Simplified version - in practice, this would be learned from data.
        """
        # Create gradient from own half (0) to opponent box (high)
        xT = np.zeros((8, 12))

        for i in range(12):
            # Increase threat as we move toward opponent goal
            base_threat = (i / 12) ** 2

            for j in range(8):
                # Higher threat in central areas
                center_distance = abs(j - 3.5) / 3.5
                center_bonus = 1 - center_distance * 0.3

                xT[j, i] = base_threat * center_bonus

        # Boost for box area (last 2 columns, central 4 rows)
        xT[2:6, 10:] *= 2.0

        # Normalize to [0, 1]
        xT = xT / xT.max()

        return xT

    def _count_threatening_players(self, positions: np.ndarray, teams: np.ndarray, possession_team: int) -> int:
        """Count attackers in dangerous positions (final third, central)."""
        team_mask = teams == possession_team
        team_positions = positions[team_mask]

        # Define dangerous zone
        if possession_team == 0:  # Attacking right
            dangerous_x = team_positions[:, 0] > self.PITCH_LENGTH / 6
        else:  # Attacking left
            dangerous_x = team_positions[:, 0] < -self.PITCH_LENGTH / 6

        # Central area
        dangerous_y = np.abs(team_positions[:, 1]) < self.PITCH_WIDTH / 4

        dangerous_players = dangerous_x & dangerous_y
        return int(dangerous_players.sum())

    def _calculate_pitch_control(self, positions: np.ndarray, teams: np.ndarray, possession_team: int) -> float:
        """
        Calculate percentage of pitch controlled by possession team.

        Simplified using player density and proximity.
        """
        # Grid-based approach
        grid_x = np.linspace(-self.PITCH_LENGTH/2, self.PITCH_LENGTH/2, 21)
        grid_y = np.linspace(-self.PITCH_WIDTH/2, self.PITCH_WIDTH/2, 14)

        possession_control = 0
        total_cells = 0

        for x in grid_x:
            for y in grid_y:
                point = np.array([x, y])

                # Find closest player from each team
                distances = {0: [], 1: []}

                for i, team in enumerate(teams):
                    if team in [0, 1]:
                        dist = np.linalg.norm(positions[i] - point)
                        distances[team].append(dist)

                # Cell controlled by team with closest player
                if distances[0] and distances[1]:
                    min_dist_0 = min(distances[0])
                    min_dist_1 = min(distances[1])

                    if min_dist_0 < min_dist_1:
                        if possession_team == 0:
                            possession_control += 1
                    else:
                        if possession_team == 1:
                            possession_control += 1

                    total_cells += 1

        if total_cells == 0:
            return 0.5

        return possession_control / total_cells

    def _identify_high_pressure_zones(self, positions: np.ndarray, teams: np.ndarray,
                                     ball_position: Optional[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Identify zones with high defensive concentration."""
        if ball_position is None:
            return []

        # Find clusters of defensive players
        # Simplified: return positions of defensive players near ball

        ball_pos = np.array(ball_position)
        pressure_zones = []

        # Assume team 1 is defending (simplification)
        defensive_mask = teams == 1
        defensive_positions = positions[defensive_mask]

        for pos in defensive_positions:
            if np.linalg.norm(pos - ball_pos) < 15.0:
                pressure_zones.append((float(pos[0]), float(pos[1])))

        return pressure_zones
