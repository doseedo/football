"""
Decision Engine for Football Analytics.

Analyzes game state to identify:
- Defensive gaps and weaknesses
- Ball progression opportunities
- Pass success probability
- Risk/reward scoring for each action

The engine answers: "Where should the ball go next, and what are the odds?"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# Pitch dimensions (meters)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


@dataclass
class PlayerState:
    """State of a single player."""
    player_id: int
    team: int  # 0 or 1
    position: Tuple[float, float]  # (x, y) in meters
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in m/s
    speed: float = 0.0  # m/s
    max_speed: float = 8.0  # m/s (default, can be personalized)
    reaction_time: float = 0.3  # seconds


@dataclass
class DefensiveGap:
    """A detected gap in defensive coverage."""
    location: Tuple[float, float]  # (x, y) center of gap
    size: float  # meters (width of gap)
    time_to_close: float  # seconds until defenders can close it
    between_players: Tuple[int, int]  # player IDs on either side
    exploitable: bool  # True if gap is large enough and slow enough to exploit
    xg_if_exploited: float  # xG value if ball reaches this gap


@dataclass
class ProgressionOption:
    """A potential ball progression action."""
    action_type: str  # 'pass', 'through_ball', 'dribble'
    target_player_id: Optional[int]  # None for space targets
    target_position: Tuple[float, float]

    # Probabilities
    success_probability: float  # P(action succeeds)
    interception_probability: float  # P(defender intercepts)

    # Value
    xg_current: float  # xG at current position
    xg_target: float  # xG at target position
    xg_gain: float  # xG improvement

    # Risk
    turnover_cost: float  # xG risk if action fails

    # Expected Value
    expected_value: float  # P(success) * xg_gain - P(fail) * turnover_cost

    # Recommendation
    recommendation: str  # 'HIGH_VALUE', 'SAFE', 'RISKY', 'AVOID'


@dataclass
class DecisionEngineOutput:
    """Complete output from the Decision Engine for a single frame."""
    timestamp: float
    ball_position: Tuple[float, float]
    ball_carrier_id: Optional[int]
    possession_team: int

    # Defensive analysis
    defensive_gaps: List[DefensiveGap]

    # Progression options (sorted by expected value)
    progression_options: List[ProgressionOption]

    # Best option
    best_option: Optional[ProgressionOption]

    # Summary metrics
    total_options: int
    high_value_options: int  # EV > 0.05
    safe_options: int  # success > 80%


class CoverageZoneModel:
    """
    Models defensive coverage zones based on player physics.

    Each defender can reach a certain area based on:
    - Current position
    - Current velocity (momentum)
    - Max speed
    - Reaction time
    """

    def __init__(self, reaction_time: float = 0.3):
        self.reaction_time = reaction_time

    def get_reachable_radius(self, player: PlayerState, time_horizon: float) -> float:
        """
        Calculate how far a player can reach in given time.

        Args:
            player: Player state
            time_horizon: Time in seconds

        Returns:
            Radius in meters the player can cover
        """
        # Account for reaction time
        effective_time = max(0, time_horizon - player.reaction_time)

        # Simple model: distance = speed * time
        # Could be enhanced with acceleration model
        return player.max_speed * effective_time

    def get_coverage_zone(self, player: PlayerState, time_horizon: float) -> Tuple[Tuple[float, float], float]:
        """
        Get the circular coverage zone for a player.

        Returns:
            center: (x, y) position
            radius: coverage radius in meters
        """
        # Account for current velocity - player will drift in current direction
        drift_x = player.velocity[0] * time_horizon * 0.5  # Partial drift
        drift_y = player.velocity[1] * time_horizon * 0.5

        center = (
            player.position[0] + drift_x,
            player.position[1] + drift_y
        )
        radius = self.get_reachable_radius(player, time_horizon)

        return center, radius

    def point_in_coverage(self, point: Tuple[float, float], player: PlayerState,
                          time_horizon: float) -> bool:
        """Check if a point is within a player's coverage zone."""
        center, radius = self.get_coverage_zone(player, time_horizon)
        distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
        return distance <= radius

    def time_to_reach(self, point: Tuple[float, float], player: PlayerState) -> float:
        """Calculate time for player to reach a point."""
        distance = np.sqrt(
            (point[0] - player.position[0])**2 +
            (point[1] - player.position[1])**2
        )
        if player.max_speed <= 0:
            return float('inf')
        return player.reaction_time + distance / player.max_speed


class DefensiveGapDetector:
    """
    Detects gaps in defensive coverage.

    A gap exists where:
    - Distance between defenders exceeds threshold
    - Neither defender can reach the gap center quickly
    """

    def __init__(self,
                 min_gap_size: float = 6.0,
                 time_horizon: float = 1.5,
                 exploitable_threshold: float = 1.0):
        """
        Args:
            min_gap_size: Minimum gap width to consider (meters)
            time_horizon: Time window for coverage analysis (seconds)
            exploitable_threshold: Min time advantage to be exploitable (seconds)
        """
        self.min_gap_size = min_gap_size
        self.time_horizon = time_horizon
        self.exploitable_threshold = exploitable_threshold
        self.coverage_model = CoverageZoneModel()

    def detect_gaps(self, defenders: List[PlayerState],
                    xg_model: 'XGZoneModel') -> List[DefensiveGap]:
        """
        Detect all gaps in defensive line.

        Args:
            defenders: List of defensive players
            xg_model: Model to calculate xG at gap locations

        Returns:
            List of detected gaps, sorted by xG value
        """
        gaps = []

        if len(defenders) < 2:
            return gaps

        # Sort defenders by x position (defensive line)
        sorted_defenders = sorted(defenders, key=lambda p: p.position[0])

        # Check gaps between adjacent defenders
        for i in range(len(sorted_defenders) - 1):
            d1 = sorted_defenders[i]
            d2 = sorted_defenders[i + 1]

            gap = self._analyze_gap(d1, d2, xg_model)
            if gap is not None:
                gaps.append(gap)

        # Also check vertical gaps (between lines)
        gaps.extend(self._detect_vertical_gaps(defenders, xg_model))

        # Sort by xG value (most dangerous first)
        gaps.sort(key=lambda g: g.xg_if_exploited, reverse=True)

        return gaps

    def _analyze_gap(self, d1: PlayerState, d2: PlayerState,
                     xg_model: 'XGZoneModel') -> Optional[DefensiveGap]:
        """Analyze gap between two defenders."""
        # Calculate gap center
        gap_center = (
            (d1.position[0] + d2.position[0]) / 2,
            (d1.position[1] + d2.position[1]) / 2
        )

        # Calculate gap size
        gap_size = np.sqrt(
            (d2.position[0] - d1.position[0])**2 +
            (d2.position[1] - d1.position[1])**2
        )

        if gap_size < self.min_gap_size:
            return None

        # Calculate time for each defender to close the gap
        time_d1 = self.coverage_model.time_to_reach(gap_center, d1)
        time_d2 = self.coverage_model.time_to_reach(gap_center, d2)
        time_to_close = min(time_d1, time_d2)

        # Is it exploitable?
        # Ball can travel ~15-20 m/s, player receives in ~0.5-1s
        ball_arrival_time = 0.8  # Rough estimate for pass to gap
        exploitable = time_to_close > ball_arrival_time + self.exploitable_threshold

        # xG at gap location
        xg_value = xg_model.get_xg(gap_center)

        return DefensiveGap(
            location=gap_center,
            size=gap_size,
            time_to_close=time_to_close,
            between_players=(d1.player_id, d2.player_id),
            exploitable=exploitable,
            xg_if_exploited=xg_value
        )

    def _detect_vertical_gaps(self, defenders: List[PlayerState],
                              xg_model: 'XGZoneModel') -> List[DefensiveGap]:
        """Detect gaps between defensive lines (e.g., between back line and midfield)."""
        gaps = []

        if len(defenders) < 4:
            return gaps

        # Group by approximate line (cluster by x position)
        sorted_by_x = sorted(defenders, key=lambda p: p.position[0])

        # Simple clustering: find largest x-gap
        max_gap = 0
        gap_index = 0

        for i in range(len(sorted_by_x) - 1):
            x_gap = sorted_by_x[i + 1].position[0] - sorted_by_x[i].position[0]
            if x_gap > max_gap:
                max_gap = x_gap
                gap_index = i

        # If there's a significant gap between lines
        if max_gap > 10:  # 10 meters between lines
            back_line = sorted_by_x[:gap_index + 1]
            mid_line = sorted_by_x[gap_index + 1:]

            # Find center of the gap between lines
            back_avg_x = np.mean([p.position[0] for p in back_line])
            mid_avg_x = np.mean([p.position[0] for p in mid_line])
            gap_x = (back_avg_x + mid_avg_x) / 2

            # Check multiple y positions across the gap
            for y_offset in [-15, 0, 15]:
                gap_center = (gap_x, y_offset)

                # Time for nearest defender to reach
                times = [self.coverage_model.time_to_reach(gap_center, d) for d in defenders]
                time_to_close = min(times)

                if time_to_close > 1.0:  # Takes more than 1 second to close
                    xg_value = xg_model.get_xg(gap_center)
                    gaps.append(DefensiveGap(
                        location=gap_center,
                        size=max_gap,
                        time_to_close=time_to_close,
                        between_players=(-1, -1),  # Between lines, not specific players
                        exploitable=True,
                        xg_if_exploited=xg_value
                    ))

        return gaps


class XGZoneModel:
    """
    Maps pitch locations to expected goal (xG) values.

    Based on historical shot data - what's the probability of
    scoring from each location?
    """

    def __init__(self):
        # Initialize xG grid (12 x 8 zones)
        self.grid = self._create_xg_grid()
        self.grid_x = 12
        self.grid_y = 8

    def _create_xg_grid(self) -> np.ndarray:
        """
        Create xG grid based on standard shot probability model.

        Values represent probability of scoring if a shot is taken from that zone.
        """
        xg = np.zeros((8, 12))

        # Distance from goal is primary factor
        for i in range(12):
            for j in range(8):
                # Convert to pitch coordinates
                x = (i / 12) * PITCH_LENGTH - PITCH_LENGTH / 2  # -52.5 to 52.5
                y = (j / 8) * PITCH_WIDTH - PITCH_WIDTH / 2  # -34 to 34

                # Distance to goal center (goal at x=52.5, y=0)
                goal_x = PITCH_LENGTH / 2
                goal_y = 0
                distance = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)

                # Angle to goal
                angle = np.abs(np.arctan2(y, goal_x - x))

                # Base xG from distance (exponential decay)
                base_xg = np.exp(-distance / 20) * 0.5

                # Angle penalty (harder from tight angles)
                angle_factor = np.cos(angle) ** 0.5

                # Central bonus
                central_factor = 1 - (np.abs(y) / (PITCH_WIDTH / 2)) * 0.3

                xg[j, i] = base_xg * angle_factor * central_factor

        # Boost for inside the box (last 16.5m, central area)
        # Box is roughly last 2 columns, central 4 rows
        xg[2:6, 10:] *= 1.8

        # Six-yard box boost
        xg[3:5, 11:] *= 1.5

        # Clip to reasonable range
        xg = np.clip(xg, 0.01, 0.45)

        return xg

    def get_xg(self, position: Tuple[float, float]) -> float:
        """
        Get xG value for a pitch position.

        Args:
            position: (x, y) in meters, where (0, 0) is pitch center

        Returns:
            xG value (probability of goal if shot taken from here)
        """
        x, y = position

        # Convert to grid indices
        grid_x = int((x + PITCH_LENGTH / 2) / PITCH_LENGTH * self.grid_x)
        grid_y = int((y + PITCH_WIDTH / 2) / PITCH_WIDTH * self.grid_y)

        # Clip to bounds
        grid_x = np.clip(grid_x, 0, self.grid_x - 1)
        grid_y = np.clip(grid_y, 0, self.grid_y - 1)

        return float(self.grid[grid_y, grid_x])

    def get_xg_gain(self, from_pos: Tuple[float, float],
                    to_pos: Tuple[float, float]) -> float:
        """Calculate xG improvement from moving ball between positions."""
        return self.get_xg(to_pos) - self.get_xg(from_pos)


class InterceptionModel:
    """
    Models whether a defender can intercept a pass.

    Compares ball travel time vs defender travel time to
    any point along the pass trajectory.
    """

    def __init__(self,
                 ball_speed_ground: float = 15.0,
                 ball_speed_air: float = 20.0):
        """
        Args:
            ball_speed_ground: Ground pass speed (m/s)
            ball_speed_air: Aerial pass speed (m/s)
        """
        self.ball_speed_ground = ball_speed_ground
        self.ball_speed_air = ball_speed_air
        self.coverage_model = CoverageZoneModel()

    def interception_probability(self,
                                  pass_start: Tuple[float, float],
                                  pass_end: Tuple[float, float],
                                  defenders: List[PlayerState],
                                  is_aerial: bool = False) -> float:
        """
        Calculate probability of pass being intercepted.

        Args:
            pass_start: Ball starting position
            pass_end: Pass target position
            defenders: List of defensive players
            is_aerial: Whether pass is in the air

        Returns:
            Probability of interception (0-1)
        """
        if len(defenders) == 0:
            return 0.0

        ball_speed = self.ball_speed_air if is_aerial else self.ball_speed_ground

        # Check each point along pass trajectory
        pass_vector = np.array(pass_end) - np.array(pass_start)
        pass_length = np.linalg.norm(pass_vector)

        if pass_length < 0.1:
            return 0.0

        pass_direction = pass_vector / pass_length

        # Sample points along pass
        num_samples = max(int(pass_length / 2), 5)  # Sample every 2 meters

        min_time_advantage = float('inf')

        for i in range(num_samples + 1):
            # Point along pass trajectory
            t = i / num_samples
            point = np.array(pass_start) + t * pass_vector

            # Time for ball to reach this point
            ball_time = (t * pass_length) / ball_speed

            # Time for nearest defender to reach this point
            defender_times = [
                self.coverage_model.time_to_reach(tuple(point), d)
                for d in defenders
            ]
            min_defender_time = min(defender_times) if defender_times else float('inf')

            # Time advantage (positive = defender arrives first)
            time_advantage = ball_time - min_defender_time
            min_time_advantage = min(min_time_advantage, time_advantage)

        # Convert time advantage to probability
        # If defender arrives 0.5s+ before ball -> high interception chance
        # If ball arrives 0.5s+ before defender -> low interception chance
        if min_time_advantage > 0.5:
            return 0.9  # Defender clearly gets there first
        elif min_time_advantage > 0:
            return 0.5 + min_time_advantage * 0.8
        elif min_time_advantage > -0.5:
            return 0.3 + min_time_advantage * 0.4
        else:
            return max(0.05, 0.1 + min_time_advantage * 0.1)

    def can_intercept(self, pass_start: Tuple[float, float],
                      pass_end: Tuple[float, float],
                      defender: PlayerState,
                      is_aerial: bool = False) -> Tuple[bool, float]:
        """
        Check if a specific defender can intercept a pass.

        Returns:
            can_intercept: Boolean
            intercept_point: Distance along pass where interception occurs (0-1)
        """
        ball_speed = self.ball_speed_air if is_aerial else self.ball_speed_ground

        pass_vector = np.array(pass_end) - np.array(pass_start)
        pass_length = np.linalg.norm(pass_vector)

        if pass_length < 0.1:
            return False, 0.0

        # Check if defender can reach any point on pass before ball
        for t in np.linspace(0, 1, 20):
            point = np.array(pass_start) + t * pass_vector

            ball_time = (t * pass_length) / ball_speed
            defender_time = self.coverage_model.time_to_reach(tuple(point), defender)

            if defender_time < ball_time:
                return True, t

        return False, 1.0


class PassSuccessModel:
    """
    Predicts probability of pass completion.

    Factors:
    - Distance
    - Pressure on passer
    - Pressure on receiver
    - Pass angle (forward harder than backward)
    - Player skill (if available)
    """

    # Base success rates by distance
    DISTANCE_DECAY = {
        5: 0.95,   # Short pass
        10: 0.90,
        15: 0.85,
        20: 0.78,
        25: 0.70,
        30: 0.62,
        35: 0.55,
        40: 0.48,
    }

    def __init__(self):
        self.interception_model = InterceptionModel()

    def success_probability(self,
                            passer_pos: Tuple[float, float],
                            target_pos: Tuple[float, float],
                            defenders: List[PlayerState],
                            passer_pressure: float = 0.0,
                            is_forward: bool = True) -> float:
        """
        Calculate pass success probability.

        Args:
            passer_pos: Passer position
            target_pos: Target position
            defenders: Defending players
            passer_pressure: Pressure on passer (0-1)
            is_forward: Whether pass is forward (harder)

        Returns:
            Success probability (0-1)
        """
        # Base probability from distance
        distance = np.sqrt(
            (target_pos[0] - passer_pos[0])**2 +
            (target_pos[1] - passer_pos[1])**2
        )

        base_prob = self._distance_to_probability(distance)

        # Pressure penalty
        pressure_factor = 1 - passer_pressure * 0.25

        # Forward pass penalty
        direction_factor = 0.9 if is_forward else 1.0

        # Interception risk
        interception_prob = self.interception_model.interception_probability(
            passer_pos, target_pos, defenders
        )

        # Combine factors
        success_prob = base_prob * pressure_factor * direction_factor * (1 - interception_prob * 0.7)

        return float(np.clip(success_prob, 0.05, 0.98))

    def _distance_to_probability(self, distance: float) -> float:
        """Convert pass distance to base success probability."""
        # Interpolate from distance decay table
        distances = sorted(self.DISTANCE_DECAY.keys())

        if distance <= distances[0]:
            return self.DISTANCE_DECAY[distances[0]]
        if distance >= distances[-1]:
            return self.DISTANCE_DECAY[distances[-1]] * 0.8

        # Linear interpolation
        for i in range(len(distances) - 1):
            if distances[i] <= distance < distances[i + 1]:
                d1, d2 = distances[i], distances[i + 1]
                p1, p2 = self.DISTANCE_DECAY[d1], self.DISTANCE_DECAY[d2]
                t = (distance - d1) / (d2 - d1)
                return p1 + t * (p2 - p1)

        return 0.5


class DecisionEngine:
    """
    Main Decision Engine that combines all analysis.

    For each frame, outputs:
    - Detected defensive gaps
    - Ball progression options with probabilities
    - Risk/reward scoring
    - Best recommended action
    """

    def __init__(self,
                 min_gap_size: float = 6.0,
                 ev_threshold_high: float = 0.05,
                 ev_threshold_safe: float = 0.02):
        """
        Args:
            min_gap_size: Minimum gap to detect (meters)
            ev_threshold_high: EV threshold for 'HIGH_VALUE' recommendation
            ev_threshold_safe: EV threshold for 'SAFE' recommendation
        """
        self.gap_detector = DefensiveGapDetector(min_gap_size=min_gap_size)
        self.xg_model = XGZoneModel()
        self.interception_model = InterceptionModel()
        self.pass_model = PassSuccessModel()

        self.ev_threshold_high = ev_threshold_high
        self.ev_threshold_safe = ev_threshold_safe

    def analyze(self,
                ball_position: Tuple[float, float],
                ball_carrier_id: int,
                possession_team: int,
                teammates: List[PlayerState],
                opponents: List[PlayerState],
                timestamp: float = 0.0) -> DecisionEngineOutput:
        """
        Analyze current game state and generate decision recommendations.

        Args:
            ball_position: Current ball position (x, y) in meters
            ball_carrier_id: ID of player with ball
            possession_team: Team with possession (0 or 1)
            teammates: List of teammate states
            opponents: List of opponent states
            timestamp: Current time in match

        Returns:
            Complete decision engine analysis
        """
        # Detect defensive gaps
        gaps = self.gap_detector.detect_gaps(opponents, self.xg_model)

        # Calculate pressure on ball carrier
        ball_pressure = self._calculate_pressure(ball_position, opponents)

        # Generate progression options
        options = []

        # Option 1: Passes to teammates
        for teammate in teammates:
            if teammate.player_id == ball_carrier_id:
                continue

            option = self._analyze_pass_option(
                ball_position, teammate, opponents, ball_pressure
            )
            options.append(option)

        # Option 2: Through balls to gaps
        for gap in gaps:
            if gap.exploitable:
                option = self._analyze_gap_option(
                    ball_position, gap, opponents, ball_pressure
                )
                options.append(option)

        # Sort by expected value
        options.sort(key=lambda o: o.expected_value, reverse=True)

        # Find best option
        best_option = options[0] if options else None

        # Count option types
        high_value = sum(1 for o in options if o.expected_value >= self.ev_threshold_high)
        safe = sum(1 for o in options if o.success_probability >= 0.80)

        return DecisionEngineOutput(
            timestamp=timestamp,
            ball_position=ball_position,
            ball_carrier_id=ball_carrier_id,
            possession_team=possession_team,
            defensive_gaps=gaps,
            progression_options=options,
            best_option=best_option,
            total_options=len(options),
            high_value_options=high_value,
            safe_options=safe
        )

    def _calculate_pressure(self, ball_position: Tuple[float, float],
                           opponents: List[PlayerState]) -> float:
        """Calculate pressure on ball position from opponents."""
        if not opponents:
            return 0.0

        distances = [
            np.sqrt((o.position[0] - ball_position[0])**2 +
                   (o.position[1] - ball_position[1])**2)
            for o in opponents
        ]

        closest_3 = sorted(distances)[:3]
        avg_distance = np.mean(closest_3)

        # Normalize: 0m = max pressure, 15m+ = no pressure
        pressure = max(0, 1 - avg_distance / 15.0)

        return float(pressure)

    def _analyze_pass_option(self,
                             ball_pos: Tuple[float, float],
                             target: PlayerState,
                             opponents: List[PlayerState],
                             ball_pressure: float) -> ProgressionOption:
        """Analyze a pass to a teammate."""
        target_pos = target.position

        # Is this a forward pass?
        is_forward = target_pos[0] > ball_pos[0]

        # Success probability
        success_prob = self.pass_model.success_probability(
            ball_pos, target_pos, opponents, ball_pressure, is_forward
        )

        # Interception probability
        intercept_prob = self.interception_model.interception_probability(
            ball_pos, target_pos, opponents
        )

        # xG values
        xg_current = self.xg_model.get_xg(ball_pos)
        xg_target = self.xg_model.get_xg(target_pos)
        xg_gain = xg_target - xg_current

        # Turnover cost (opponent counter-attack potential)
        # Higher cost if we're in attacking positions
        turnover_cost = self._estimate_turnover_cost(ball_pos, target_pos)

        # Expected value
        ev = success_prob * xg_gain - (1 - success_prob) * turnover_cost

        # Recommendation
        recommendation = self._get_recommendation(ev, success_prob)

        return ProgressionOption(
            action_type='pass',
            target_player_id=target.player_id,
            target_position=target_pos,
            success_probability=success_prob,
            interception_probability=intercept_prob,
            xg_current=xg_current,
            xg_target=xg_target,
            xg_gain=xg_gain,
            turnover_cost=turnover_cost,
            expected_value=ev,
            recommendation=recommendation
        )

    def _analyze_gap_option(self,
                           ball_pos: Tuple[float, float],
                           gap: DefensiveGap,
                           opponents: List[PlayerState],
                           ball_pressure: float) -> ProgressionOption:
        """Analyze a through ball into a gap."""
        target_pos = gap.location

        # Through balls are harder - reduce base success
        success_prob = self.pass_model.success_probability(
            ball_pos, target_pos, opponents, ball_pressure, is_forward=True
        ) * 0.85  # Through ball penalty

        # Interception probability
        intercept_prob = self.interception_model.interception_probability(
            ball_pos, target_pos, opponents
        )

        # xG values
        xg_current = self.xg_model.get_xg(ball_pos)
        xg_target = gap.xg_if_exploited
        xg_gain = xg_target - xg_current

        # Turnover cost
        turnover_cost = self._estimate_turnover_cost(ball_pos, target_pos)

        # Expected value
        ev = success_prob * xg_gain - (1 - success_prob) * turnover_cost

        # Recommendation
        recommendation = self._get_recommendation(ev, success_prob)

        return ProgressionOption(
            action_type='through_ball',
            target_player_id=None,
            target_position=target_pos,
            success_probability=success_prob,
            interception_probability=intercept_prob,
            xg_current=xg_current,
            xg_target=xg_target,
            xg_gain=xg_gain,
            turnover_cost=turnover_cost,
            expected_value=ev,
            recommendation=recommendation
        )

    def _estimate_turnover_cost(self, ball_pos: Tuple[float, float],
                                target_pos: Tuple[float, float]) -> float:
        """
        Estimate cost if pass fails and opponent gets ball.

        Higher cost when:
        - We're in attacking positions (counter-attack danger)
        - Pass is risky (long, into crowded area)
        """
        # Base cost from current position
        # If we're in attacking third, turnover is more dangerous
        x = ball_pos[0]
        attacking_third = x > PITCH_LENGTH / 6

        if attacking_third:
            base_cost = 0.08  # Dangerous counter-attack potential
        elif x > -PITCH_LENGTH / 6:
            base_cost = 0.05  # Middle third
        else:
            base_cost = 0.02  # Own third - less danger

        # Longer passes = more space behind if intercepted
        distance = np.sqrt((target_pos[0] - ball_pos[0])**2 +
                          (target_pos[1] - ball_pos[1])**2)
        distance_factor = 1 + distance / 50

        return base_cost * distance_factor

    def _get_recommendation(self, ev: float, success_prob: float) -> str:
        """Get recommendation label for an option."""
        if ev >= self.ev_threshold_high:
            return 'HIGH_VALUE'
        elif success_prob >= 0.85 and ev >= 0:
            return 'SAFE'
        elif ev >= self.ev_threshold_safe:
            return 'MODERATE'
        elif ev >= 0:
            return 'LOW_VALUE'
        else:
            return 'AVOID'
