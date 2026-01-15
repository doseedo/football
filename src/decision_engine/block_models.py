"""
Defensive Block Models

Defines how teams set up defensively at different heights:
- Low Block: Deep, compact, protect the goal
- Mid Block: Balanced, control center of pitch
- High Block: Aggressive press, trap opponents high

Each block has characteristic:
- Line heights (where defensive lines sit)
- Compactness targets (how tight the unit stays)
- Trigger distances (when to press vs hold)
- Space allowances (what areas to concede)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import math

from .pitch_geometry import (
    Position,
    Velocity,
    PitchGeometry,
    HALF_LENGTH,
    HALF_WIDTH,
    PENALTY_AREA_LENGTH,
)
from .elimination import Player


class BlockType(Enum):
    """Types of defensive blocks."""
    LOW = "low"             # Deep, compact, protect goal
    MID = "mid"             # Balanced, control midfield
    HIGH = "high"           # Aggressive, high press
    ULTRA_LOW = "ultra_low" # Bus parking
    ULTRA_HIGH = "ultra_high"  # Full-pitch press


@dataclass
class BlockConfiguration:
    """
    Configuration parameters for a defensive block.

    All distances in meters from own goal (x = -52.5).
    Positive x means further from own goal.
    """
    # Line positions (distance from own goal)
    defensive_line_height: float    # Where back line sits
    midfield_line_height: float     # Where midfielders sit
    forward_line_height: float      # Where first press starts

    # Compactness targets
    vertical_compactness: float     # Max distance front to back
    horizontal_compactness: float   # Max width of block

    # Trigger distances
    press_trigger_distance: float   # Distance to ball to start pressing
    drop_trigger_distance: float    # Distance to ball to drop off

    # Space management
    space_behind_tolerance: float   # How much space allowed behind line
    space_between_lines: float      # Target gap between lines

    # Behavioral parameters
    ball_side_shift: float          # How much to shift toward ball side
    cover_shadow_angle: float       # Angle for cover shadow (degrees)

    def get_line_heights(self) -> List[float]:
        """Get all line heights as a list."""
        return [
            self.forward_line_height,
            self.midfield_line_height,
            self.defensive_line_height,
        ]


# Pre-defined block configurations
LOW_BLOCK = BlockConfiguration(
    defensive_line_height=-40.0,        # 12.5m from goal
    midfield_line_height=-30.0,         # 22.5m from goal
    forward_line_height=-15.0,          # 37.5m from goal
    vertical_compactness=25.0,
    horizontal_compactness=35.0,
    press_trigger_distance=8.0,         # Only press when very close
    drop_trigger_distance=20.0,
    space_behind_tolerance=5.0,         # Protect space behind
    space_between_lines=10.0,
    ball_side_shift=8.0,
    cover_shadow_angle=12.0,
)

MID_BLOCK = BlockConfiguration(
    defensive_line_height=-30.0,        # 22.5m from goal
    midfield_line_height=-10.0,         # 42.5m from goal
    forward_line_height=10.0,           # In opponent half
    vertical_compactness=30.0,
    horizontal_compactness=40.0,
    press_trigger_distance=12.0,
    drop_trigger_distance=25.0,
    space_behind_tolerance=8.0,
    space_between_lines=12.0,
    ball_side_shift=10.0,
    cover_shadow_angle=15.0,
)

HIGH_BLOCK = BlockConfiguration(
    defensive_line_height=-10.0,        # Near halfway
    midfield_line_height=15.0,          # In opponent half
    forward_line_height=35.0,           # Deep in opponent half
    vertical_compactness=35.0,
    horizontal_compactness=45.0,
    press_trigger_distance=20.0,        # Press from distance
    drop_trigger_distance=30.0,
    space_behind_tolerance=15.0,        # Accept space behind
    space_between_lines=15.0,
    ball_side_shift=12.0,
    cover_shadow_angle=18.0,
)

ULTRA_LOW_BLOCK = BlockConfiguration(
    defensive_line_height=-45.0,        # On edge of penalty area
    midfield_line_height=-38.0,         # Just ahead of defense
    forward_line_height=-25.0,          # Still very deep
    vertical_compactness=20.0,          # Very compact
    horizontal_compactness=30.0,
    press_trigger_distance=5.0,         # Rarely press
    drop_trigger_distance=15.0,
    space_behind_tolerance=3.0,         # Minimal space behind
    space_between_lines=8.0,
    ball_side_shift=6.0,
    cover_shadow_angle=10.0,
)

ULTRA_HIGH_BLOCK = BlockConfiguration(
    defensive_line_height=5.0,          # Above halfway
    midfield_line_height=25.0,
    forward_line_height=45.0,           # Near opponent penalty area
    vertical_compactness=40.0,
    horizontal_compactness=50.0,
    press_trigger_distance=25.0,        # Always pressing
    drop_trigger_distance=40.0,
    space_behind_tolerance=25.0,        # Accept lots of space behind
    space_between_lines=18.0,
    ball_side_shift=15.0,
    cover_shadow_angle=20.0,
)


class DefensiveBlock:
    """
    A defensive block that positions players according to configuration.

    The block responds to ball position by:
    1. Shifting toward the ball side
    2. Adjusting line heights based on ball depth
    3. Maintaining compactness targets
    4. Triggering press or drop based on distances
    """

    def __init__(
        self,
        block_type: BlockType = BlockType.MID,
        config: Optional[BlockConfiguration] = None,
        geometry: Optional[PitchGeometry] = None,
    ):
        self.block_type = block_type
        self.geometry = geometry or PitchGeometry()

        # Use provided config or default for block type
        if config:
            self.config = config
        else:
            self.config = self._get_default_config(block_type)

    def _get_default_config(self, block_type: BlockType) -> BlockConfiguration:
        """Get default configuration for a block type."""
        configs = {
            BlockType.LOW: LOW_BLOCK,
            BlockType.MID: MID_BLOCK,
            BlockType.HIGH: HIGH_BLOCK,
            BlockType.ULTRA_LOW: ULTRA_LOW_BLOCK,
            BlockType.ULTRA_HIGH: ULTRA_HIGH_BLOCK,
        }
        return configs.get(block_type, MID_BLOCK)

    def calculate_positions(
        self,
        ball_position: Position,
        formation: str = "4-4-2",
    ) -> Dict[str, Position]:
        """
        Calculate ideal positions for all defenders in the block.

        Args:
            ball_position: Current ball position
            formation: Formation string (e.g., "4-4-2", "4-3-3")

        Returns:
            Dict mapping position names to coordinates
        """
        # Parse formation
        lines = self._parse_formation(formation)

        # Calculate adjusted line heights based on ball position
        adjusted_heights = self._adjust_line_heights(ball_position)

        # Calculate ball-side shift
        shift = self._calculate_shift(ball_position)

        positions = {}
        line_names = ["forwards", "midfield", "defense"]

        for line_idx, (line_name, count) in enumerate(zip(line_names, lines)):
            if count == 0:
                continue

            line_height = adjusted_heights[line_idx]
            line_positions = self._distribute_on_line(
                count=count,
                line_height=line_height,
                shift=shift,
                ball_position=ball_position,
            )

            for i, pos in enumerate(line_positions):
                positions[f"{line_name}_{i+1}"] = pos

        # Add goalkeeper
        positions["goalkeeper"] = Position(-HALF_LENGTH + 5, 0)

        return positions

    def _parse_formation(self, formation: str) -> List[int]:
        """
        Parse formation string into line counts.

        Returns [forwards, midfielders, defenders]
        """
        parts = formation.split("-")
        if len(parts) == 3:
            # Standard 4-4-2 style
            return [int(parts[2]), int(parts[1]), int(parts[0])]
        elif len(parts) == 4:
            # 4-2-3-1 style - combine middle two for midfield
            return [
                int(parts[3]),
                int(parts[1]) + int(parts[2]),
                int(parts[0])
            ]
        else:
            # Default
            return [2, 4, 4]

    def _adjust_line_heights(
        self,
        ball_position: Position,
    ) -> List[float]:
        """
        Adjust line heights based on ball position.

        Lines drop when ball is deep, push up when ball is far.
        """
        # Normalize ball x position to [0, 1] where 0 is own goal
        ball_depth = (ball_position.x + HALF_LENGTH) / (2 * HALF_LENGTH)

        # Adjustment factor: negative when ball is deep (pull back)
        # positive when ball is far (push up)
        adjustment = (ball_depth - 0.5) * 15  # ±7.5m max adjustment

        # Clamp adjustment based on block type
        if self.block_type in [BlockType.LOW, BlockType.ULTRA_LOW]:
            adjustment = min(adjustment, 5)  # Don't push up too much
        elif self.block_type in [BlockType.HIGH, BlockType.ULTRA_HIGH]:
            adjustment = max(adjustment, -5)  # Don't drop too much

        return [
            self.config.forward_line_height + adjustment,
            self.config.midfield_line_height + adjustment * 0.8,
            self.config.defensive_line_height + adjustment * 0.6,
        ]

    def _calculate_shift(self, ball_position: Position) -> float:
        """
        Calculate how much to shift toward ball side.
        """
        # Shift proportional to ball's lateral position
        ball_y = ball_position.y
        max_shift = self.config.ball_side_shift

        # Normalize ball y to [-1, 1]
        normalized_y = ball_y / HALF_WIDTH

        return normalized_y * max_shift

    def _distribute_on_line(
        self,
        count: int,
        line_height: float,
        shift: float,
        ball_position: Position,
    ) -> List[Position]:
        """
        Distribute players evenly on a defensive line.

        Players are spread across the width with ball-side shift.
        """
        positions = []

        # Calculate horizontal spread
        spread = min(
            self.config.horizontal_compactness,
            2 * HALF_WIDTH - 10  # Leave some margin
        )

        # Starting y position (shifted toward ball)
        center_y = shift

        # Distribute players
        if count == 1:
            positions.append(Position(line_height, center_y))
        else:
            step = spread / (count - 1)
            start_y = center_y - spread / 2

            for i in range(count):
                y = start_y + i * step
                # Clamp to pitch
                y = max(-HALF_WIDTH + 2, min(HALF_WIDTH - 2, y))
                positions.append(Position(line_height, y))

        return positions

    def should_press(
        self,
        defender_position: Position,
        ball_position: Position,
    ) -> bool:
        """
        Determine if a defender should press the ball.
        """
        dist = defender_position.distance_to(ball_position)
        return dist < self.config.press_trigger_distance

    def should_drop(
        self,
        defender_position: Position,
        ball_position: Position,
    ) -> bool:
        """
        Determine if a defender should drop off.
        """
        dist = defender_position.distance_to(ball_position)
        return dist > self.config.drop_trigger_distance

    def get_pressure_zones(self) -> List[Tuple[Position, Position]]:
        """
        Get the zones where this block applies pressure.

        Returns list of (corner1, corner2) rectangles.
        """
        zones = []

        # Primary pressure zone: around forward line
        forward_height = self.config.forward_line_height
        pressure_depth = self.config.press_trigger_distance

        zones.append((
            Position(forward_height - pressure_depth, -HALF_WIDTH),
            Position(HALF_LENGTH, HALF_WIDTH),
        ))

        return zones

    def get_protected_zones(self) -> List[Tuple[Position, Position]]:
        """
        Get the zones this block prioritizes protecting.

        Returns list of (corner1, corner2) rectangles.
        """
        zones = []

        # Primary protected zone: behind defensive line
        def_height = self.config.defensive_line_height

        zones.append((
            Position(-HALF_LENGTH, -HALF_WIDTH),
            Position(def_height + self.config.space_behind_tolerance, HALF_WIDTH),
        ))

        return zones

    def evaluate_vulnerability(
        self,
        ball_position: Position,
        defenders: List[Player],
    ) -> Dict[str, float]:
        """
        Evaluate how vulnerable the block is to attack.

        Returns vulnerability scores for different threat types.
        """
        vulnerabilities = {}

        # Space behind vulnerability
        def_line = self.config.defensive_line_height
        if ball_position.x > def_line:
            space_behind = ball_position.x - def_line
            vulnerabilities["space_behind"] = min(1.0, space_behind / 20)
        else:
            vulnerabilities["space_behind"] = 0.0

        # Width vulnerability (exposed flanks)
        defender_ys = [d.position.y for d in defenders]
        if defender_ys:
            left_exposure = min(defender_ys) + HALF_WIDTH
            right_exposure = HALF_WIDTH - max(defender_ys)
            vulnerabilities["flanks"] = max(left_exposure, right_exposure) / 20
        else:
            vulnerabilities["flanks"] = 1.0

        # Gaps between lines
        defender_xs = sorted([d.position.x for d in defenders])
        max_gap = 0
        for i in range(len(defender_xs) - 1):
            gap = defender_xs[i+1] - defender_xs[i]
            max_gap = max(max_gap, gap)
        vulnerabilities["line_gaps"] = min(1.0, max_gap / 15)

        # Overall vulnerability
        vulnerabilities["total"] = (
            0.4 * vulnerabilities["space_behind"] +
            0.3 * vulnerabilities["flanks"] +
            0.3 * vulnerabilities["line_gaps"]
        )

        return vulnerabilities


class BlockTransitionManager:
    """
    Manages transitions between defensive blocks.

    Teams don't stay in one block - they transition based on:
    - Ball position
    - Score
    - Time
    - Fatigue
    """

    def __init__(self, geometry: Optional[PitchGeometry] = None):
        self.geometry = geometry or PitchGeometry()
        self.blocks = {
            BlockType.LOW: DefensiveBlock(BlockType.LOW, geometry=self.geometry),
            BlockType.MID: DefensiveBlock(BlockType.MID, geometry=self.geometry),
            BlockType.HIGH: DefensiveBlock(BlockType.HIGH, geometry=self.geometry),
        }

    def recommend_block(
        self,
        ball_position: Position,
        score_differential: int,
        time_remaining: float,
        team_fatigue: float = 0.0,
    ) -> BlockType:
        """
        Recommend which block to use based on game state.

        Args:
            ball_position: Current ball position
            score_differential: Goals ahead (+) or behind (-)
            time_remaining: Minutes remaining
            team_fatigue: Fatigue level 0-1

        Returns:
            Recommended BlockType
        """
        # Base recommendation from ball position
        ball_depth = (ball_position.x + HALF_LENGTH) / (2 * HALF_LENGTH)

        if ball_depth < 0.3:
            # Ball deep in own half - probably low block
            base = BlockType.LOW
        elif ball_depth > 0.7:
            # Ball deep in opponent half - can press high
            base = BlockType.HIGH
        else:
            base = BlockType.MID

        # Modify based on score
        if score_differential > 0:
            # Winning - can drop deeper
            if base == BlockType.HIGH:
                base = BlockType.MID
            elif base == BlockType.MID and time_remaining < 20:
                base = BlockType.LOW
        elif score_differential < 0:
            # Losing - need to press
            if base == BlockType.LOW:
                base = BlockType.MID
            elif base == BlockType.MID and time_remaining < 20:
                base = BlockType.HIGH

        # Modify based on fatigue
        if team_fatigue > 0.7:
            # Tired - can't maintain high press
            if base == BlockType.HIGH:
                base = BlockType.MID

        return base

    def get_transition_positions(
        self,
        from_block: BlockType,
        to_block: BlockType,
        ball_position: Position,
        formation: str = "4-4-2",
        transition_progress: float = 0.5,
    ) -> Dict[str, Position]:
        """
        Get interpolated positions during block transition.

        Args:
            from_block: Starting block type
            to_block: Target block type
            ball_position: Current ball position
            formation: Team formation
            transition_progress: 0 = at from_block, 1 = at to_block
        """
        from_positions = self.blocks[from_block].calculate_positions(
            ball_position, formation
        )
        to_positions = self.blocks[to_block].calculate_positions(
            ball_position, formation
        )

        # Interpolate
        result = {}
        for key in from_positions:
            if key in to_positions:
                from_pos = from_positions[key]
                to_pos = to_positions[key]
                result[key] = Position(
                    from_pos.x + transition_progress * (to_pos.x - from_pos.x),
                    from_pos.y + transition_progress * (to_pos.y - from_pos.y),
                )
            else:
                result[key] = from_positions[key]

        return result
