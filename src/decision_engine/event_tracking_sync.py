"""
Event-Tracking Data Synchronization

Combines Wyscout event data with tracking data to enable
decision analysis at each moment of the game.

The key insight: Events tell us WHAT happened, tracking tells us
WHERE EVERYONE WAS when it happened. Combined, we can analyze WHY
and WHAT ELSE could have happened.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from bisect import bisect_left


@dataclass
class PlayerPosition:
    """Position of a single player at a moment."""
    player_id: str
    team: str  # 'home' or 'away'
    x: float  # meters (0-105)
    y: float  # meters (0-68)
    jersey_number: Optional[int] = None


@dataclass
class BallPosition:
    """Position of the ball."""
    x: float
    y: float
    z: float = 0.0


@dataclass
class FrameSnapshot:
    """Complete snapshot of the game at one moment."""
    timestamp: float  # seconds
    period: int
    players: List[PlayerPosition]
    ball: Optional[BallPosition]

    def get_team_players(self, team: str) -> List[PlayerPosition]:
        return [p for p in self.players if p.team == team]

    def get_player_by_id(self, player_id: str) -> Optional[PlayerPosition]:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None


@dataclass
class SynchronizedEvent:
    """An event with full positional context."""
    # Event info (from Wyscout)
    event_id: str
    event_type: str  # 'pass', 'shot', 'dribble', etc.
    sub_type: Optional[str]  # 'accurate', 'inaccurate', etc.
    timestamp: float
    period: int
    player_id: str
    team_id: str

    # Positions (from event data)
    start_x: float
    start_y: float
    end_x: Optional[float] = None
    end_y: Optional[float] = None

    # Full context (from tracking data)
    snapshot: Optional[FrameSnapshot] = None

    # Outcome
    successful: bool = False
    tags: List[str] = field(default_factory=list)


class EventTrackingSync:
    """
    Synchronizes Wyscout event data with tracking data.

    Uses timestamp matching with tolerance for temporal offset.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        sync_tolerance_ms: float = 500.0,  # Max time difference for sync
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.sync_tolerance = sync_tolerance_ms / 1000.0  # Convert to seconds

        self.events: List[SynchronizedEvent] = []
        self.tracking_frames: List[FrameSnapshot] = []
        self.frame_timestamps: List[float] = []  # For binary search

    def load_wyscout_events(self, events_data: List[Dict]) -> None:
        """Load events from Wyscout format."""
        for event in events_data:
            positions = event.get('positions', [])
            tags = event.get('tags', [])

            # Extract tag names
            tag_names = []
            for tag in tags:
                if isinstance(tag, dict):
                    tag_names.append(tag.get('name', str(tag.get('id', ''))))
                else:
                    tag_names.append(str(tag))

            # Check if successful (tag 1801 = accurate pass, etc.)
            successful = any(t in ['accurate', '1801', 'Goal'] for t in tag_names)

            # Convert Wyscout 0-100 coords to meters
            start_x = positions[0].get('x', 0) / 100 * self.pitch_length if positions else 0
            start_y = positions[0].get('y', 0) / 100 * self.pitch_width if positions else 0
            end_x = positions[1].get('x', 0) / 100 * self.pitch_length if len(positions) > 1 else None
            end_y = positions[1].get('y', 0) / 100 * self.pitch_width if len(positions) > 1 else None

            sync_event = SynchronizedEvent(
                event_id=str(event.get('eventId', event.get('id', ''))),
                event_type=event.get('eventName', event.get('type', 'unknown')),
                sub_type=event.get('subEventName', event.get('subtype')),
                timestamp=event.get('eventSec', event.get('second', 0)),
                period=self._parse_period(event.get('matchPeriod', '1H')),
                player_id=str(event.get('playerId', '')),
                team_id=str(event.get('teamId', '')),
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                successful=successful,
                tags=tag_names,
            )
            self.events.append(sync_event)

        print(f"Loaded {len(self.events)} events")

    def load_tracking_data(self, tracking_data: List[Dict]) -> None:
        """Load tracking frames from standard format."""
        for frame_data in tracking_data:
            players = []

            # Parse players
            for p in frame_data.get('players', frame_data.get('data', [])):
                players.append(PlayerPosition(
                    player_id=str(p.get('playerId', p.get('track_id', p.get('id', '')))),
                    team=p.get('team', p.get('team_id', 'unknown')),
                    x=p.get('x', 0),
                    y=p.get('y', 0),
                    jersey_number=p.get('jersey_number', p.get('jerseyNo')),
                ))

            # Parse ball
            ball_data = frame_data.get('ball')
            ball = None
            if ball_data:
                ball = BallPosition(
                    x=ball_data.get('x', 0),
                    y=ball_data.get('y', 0),
                    z=ball_data.get('z', 0),
                )

            frame = FrameSnapshot(
                timestamp=frame_data.get('timestamp', 0),
                period=frame_data.get('period', 1),
                players=players,
                ball=ball,
            )
            self.tracking_frames.append(frame)
            self.frame_timestamps.append(frame.timestamp)

        print(f"Loaded {len(self.tracking_frames)} tracking frames")

    def synchronize(self) -> List[SynchronizedEvent]:
        """
        Match each event to the closest tracking frame.

        Uses binary search for efficiency.
        """
        if not self.tracking_frames:
            print("Warning: No tracking data loaded. Events will have no positional context.")
            return self.events

        synced_count = 0

        for event in self.events:
            # Find closest frame by timestamp
            idx = bisect_left(self.frame_timestamps, event.timestamp)

            # Check neighbors for best match
            best_frame = None
            best_diff = float('inf')

            for check_idx in [idx - 1, idx, idx + 1]:
                if 0 <= check_idx < len(self.tracking_frames):
                    frame = self.tracking_frames[check_idx]

                    # Must be same period
                    if frame.period != event.period:
                        continue

                    diff = abs(frame.timestamp - event.timestamp)
                    if diff < best_diff:
                        best_diff = diff
                        best_frame = frame

            # Only sync if within tolerance
            if best_frame and best_diff <= self.sync_tolerance:
                event.snapshot = best_frame
                synced_count += 1

        print(f"Synchronized {synced_count}/{len(self.events)} events with tracking data")
        return self.events

    def _parse_period(self, period_str: str) -> int:
        """Convert period string to int."""
        if isinstance(period_str, int):
            return period_str
        if period_str in ['1H', '1st', 'first']:
            return 1
        if period_str in ['2H', '2nd', 'second']:
            return 2
        return int(period_str.replace('H', '')) if period_str else 1

    def get_events_by_type(self, event_type: str) -> List[SynchronizedEvent]:
        """Filter events by type (e.g., 'Pass', 'Shot')."""
        return [e for e in self.events if e.event_type.lower() == event_type.lower()]

    def get_events_with_context(self) -> List[SynchronizedEvent]:
        """Get only events that have tracking context."""
        return [e for e in self.events if e.snapshot is not None]


class DecisionAnalyzer:
    """
    Analyzes decisions made at each event.

    For each event with tracking context, calculates:
    - What options were available
    - xG of each option
    - Whether the actual decision was optimal
    """

    GOAL_POS = {'x': 105.0, 'y': 34.0}  # Center of attacking goal
    GOAL_WIDTH = 7.32

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def analyze_event(self, event: SynchronizedEvent) -> Dict:
        """
        Analyze a single event with its positional context.

        Returns analysis of the decision made.
        """
        if not event.snapshot:
            return {'error': 'No tracking context available'}

        analysis = {
            'event_id': event.event_id,
            'event_type': event.event_type,
            'actual_decision': event.sub_type,
            'successful': event.successful,
            'options': [],
            'best_option': None,
            'decision_quality': None,
        }

        # Get player with ball
        ball_carrier = event.snapshot.get_player_by_id(event.player_id)
        if not ball_carrier:
            # Use event position as carrier position
            ball_carrier = PlayerPosition(
                player_id=event.player_id,
                team=event.team_id,
                x=event.start_x,
                y=event.start_y,
            )

        # Get teammates and opponents
        teammates = [p for p in event.snapshot.players
                    if p.team == ball_carrier.team and p.player_id != ball_carrier.player_id]
        opponents = [p for p in event.snapshot.players if p.team != ball_carrier.team]

        # Analyze options based on event type
        if event.event_type.lower() == 'pass':
            analysis['options'] = self._analyze_pass_options(ball_carrier, teammates, opponents)
        elif event.event_type.lower() == 'shot':
            analysis['options'] = self._analyze_shot_options(ball_carrier, opponents)

        # Find best option
        if analysis['options']:
            best = max(analysis['options'], key=lambda x: x.get('expected_value', 0))
            analysis['best_option'] = best

            # Calculate decision quality
            actual_ev = next(
                (o['expected_value'] for o in analysis['options']
                 if o.get('is_actual_choice', False)),
                None
            )
            if actual_ev is not None and best['expected_value'] > 0:
                analysis['decision_quality'] = actual_ev / best['expected_value']

        return analysis

    def _analyze_pass_options(
        self,
        carrier: PlayerPosition,
        teammates: List[PlayerPosition],
        opponents: List[PlayerPosition]
    ) -> List[Dict]:
        """Analyze all possible pass options."""
        options = []

        for teammate in teammates:
            option = self._evaluate_pass(carrier, teammate, opponents)
            options.append(option)

        # Also consider shot option
        shot_option = self._evaluate_shot_from_position(carrier, opponents)
        if shot_option['xG'] > 0.02:  # Only if meaningful
            options.append(shot_option)

        return sorted(options, key=lambda x: x['expected_value'], reverse=True)

    def _evaluate_pass(
        self,
        carrier: PlayerPosition,
        target: PlayerPosition,
        opponents: List[PlayerPosition]
    ) -> Dict:
        """Evaluate a pass to a specific target."""
        pass_dist = self._distance(carrier, target)

        # Calculate interception probability
        intercept_prob = 0.0
        for opp in opponents:
            perp_dist = self._point_to_line_distance(opp, carrier, target)
            if perp_dist < 2:
                intercept_prob += 0.3
            elif perp_dist < 4:
                intercept_prob += 0.15
            elif perp_dist < 6:
                intercept_prob += 0.05

        intercept_prob = min(intercept_prob, 0.9)
        success_prob = max(0.1, 1 - intercept_prob)

        # Adjust for distance
        if pass_dist > 30:
            success_prob *= 0.7
        elif pass_dist > 20:
            success_prob *= 0.85

        # Value of receiving position (closer to goal = better)
        receiver_xg = self._calculate_xG(target, opponents)
        position_value = max(0.1, receiver_xg * 2)  # Scale up for passes

        expected_value = success_prob * position_value - (1 - success_prob) * 0.3

        return {
            'type': 'pass',
            'target_player': target.player_id,
            'target_position': {'x': target.x, 'y': target.y},
            'distance': pass_dist,
            'success_prob': success_prob,
            'receiver_xG': receiver_xg,
            'expected_value': expected_value,
        }

    def _evaluate_shot_from_position(
        self,
        shooter: PlayerPosition,
        opponents: List[PlayerPosition]
    ) -> Dict:
        """Evaluate shot from current position."""
        xG = self._calculate_xG(shooter, opponents)

        # EV = xG - (1-xG) * possession_loss_cost
        expected_value = xG - (1 - xG) * 0.1

        return {
            'type': 'shot',
            'target_player': None,
            'target_position': self.GOAL_POS,
            'distance': self._distance_to_goal(shooter),
            'success_prob': xG,
            'xG': xG,
            'expected_value': expected_value,
        }

    def _analyze_shot_options(
        self,
        shooter: PlayerPosition,
        opponents: List[PlayerPosition]
    ) -> List[Dict]:
        """For a shot event, analyze the shot quality."""
        shot = self._evaluate_shot_from_position(shooter, opponents)
        shot['is_actual_choice'] = True
        return [shot]

    def _calculate_xG(self, pos: PlayerPosition, opponents: List[PlayerPosition]) -> float:
        """Calculate expected goals from a position."""
        dist_to_goal = self._distance_to_goal(pos)

        if dist_to_goal > 35:
            return 0.01

        # Base xG from distance
        if dist_to_goal < 5:
            base_xG = 0.70
        elif dist_to_goal < 8:
            base_xG = 0.50
        elif dist_to_goal < 11:
            base_xG = 0.35
        elif dist_to_goal < 16:
            base_xG = 0.20
        elif dist_to_goal < 20:
            base_xG = 0.12
        elif dist_to_goal < 25:
            base_xG = 0.06
        else:
            base_xG = 0.03

        # Angle factor
        angle = abs(math.atan2(pos.y - self.pitch_width/2, self.pitch_length - pos.x))
        if angle > 0.8:
            base_xG *= 0.3
        elif angle > 0.5:
            base_xG *= 0.5
        elif angle > 0.3:
            base_xG *= 0.75

        # Defender pressure
        for opp in opponents:
            opp_dist = self._distance(pos, opp)
            if opp_dist < 2:
                base_xG *= 0.3
            elif opp_dist < 4:
                base_xG *= 0.5
            elif opp_dist < 6:
                base_xG *= 0.7

        # Space bonus
        nearest_opp = min((self._distance(pos, o) for o in opponents), default=100)
        if nearest_opp > 10:
            base_xG *= 1.5

        return min(0.95, max(0.01, base_xG))

    def _distance(self, p1: PlayerPosition, p2: PlayerPosition) -> float:
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _distance_to_goal(self, p: PlayerPosition) -> float:
        return math.sqrt((self.pitch_length - p.x)**2 + (self.pitch_width/2 - p.y)**2)

    def _point_to_line_distance(
        self,
        point: PlayerPosition,
        line_start: PlayerPosition,
        line_end: PlayerPosition
    ) -> float:
        """Calculate perpendicular distance from point to line segment."""
        px, py = point.x, point.y
        x1, y1 = line_start.x, line_start.y
        x2, y2 = line_end.x, line_end.y

        # Vector from start to end
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx*dx + dy*dy

        if length_sq == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        # Project point onto line
        t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def analyze_match(self, events: List[SynchronizedEvent]) -> Dict:
        """
        Analyze all events in a match.

        Returns summary statistics and individual analyses.
        """
        analyses = []

        for event in events:
            if event.snapshot:  # Only analyze events with context
                analysis = self.analyze_event(event)
                analyses.append(analysis)

        # Compute summary stats
        decision_qualities = [
            a['decision_quality'] for a in analyses
            if a.get('decision_quality') is not None
        ]

        summary = {
            'total_events': len(events),
            'events_with_context': len(analyses),
            'avg_decision_quality': sum(decision_qualities) / len(decision_qualities) if decision_qualities else None,
            'optimal_decisions': sum(1 for q in decision_qualities if q >= 0.9) if decision_qualities else 0,
            'suboptimal_decisions': sum(1 for q in decision_qualities if q < 0.7) if decision_qualities else 0,
            'analyses': analyses,
        }

        return summary


class KeyMomentsExtractor:
    """
    Identifies key decision moments in a match.

    Finds:
    - Best decisions (optimal choices made)
    - Missed opportunities (better options were available)
    - Critical errors (very suboptimal decisions)
    """

    def __init__(self, analyzer: DecisionAnalyzer):
        self.analyzer = analyzer

    def extract_key_moments(
        self,
        events: List[SynchronizedEvent],
        top_n: int = 10
    ) -> Dict:
        """Extract the most significant decision moments."""
        moments = {
            'best_decisions': [],
            'missed_opportunities': [],
            'critical_errors': [],
            'high_xG_chances': [],
        }

        for event in events:
            if not event.snapshot:
                continue

            analysis = self.analyzer.analyze_event(event)

            if not analysis.get('options'):
                continue

            best_option = analysis.get('best_option')
            decision_quality = analysis.get('decision_quality')

            moment = {
                'event_id': event.event_id,
                'event_type': event.event_type,
                'timestamp': event.timestamp,
                'period': event.period,
                'position': {'x': event.start_x, 'y': event.start_y},
                'decision_quality': decision_quality,
                'actual_choice': event.sub_type,
                'best_option': best_option,
                'successful': event.successful,
                'num_options': len(analysis['options']),
            }

            # Categorize
            if decision_quality is not None:
                if decision_quality >= 0.95:
                    moments['best_decisions'].append(moment)
                elif decision_quality < 0.5:
                    moments['critical_errors'].append(moment)
                elif decision_quality < 0.75:
                    moments['missed_opportunities'].append(moment)

            # Track high xG chances
            if event.event_type.lower() == 'shot':
                shot_xG = best_option.get('xG', 0) if best_option else 0
                if shot_xG > 0.15:
                    moment['xG'] = shot_xG
                    moments['high_xG_chances'].append(moment)

        # Sort and limit
        moments['best_decisions'] = sorted(
            moments['best_decisions'],
            key=lambda x: x['decision_quality'] or 0,
            reverse=True
        )[:top_n]

        moments['missed_opportunities'] = sorted(
            moments['missed_opportunities'],
            key=lambda x: x['decision_quality'] or 1
        )[:top_n]

        moments['critical_errors'] = sorted(
            moments['critical_errors'],
            key=lambda x: x['decision_quality'] or 1
        )[:top_n]

        moments['high_xG_chances'] = sorted(
            moments['high_xG_chances'],
            key=lambda x: x.get('xG', 0),
            reverse=True
        )[:top_n]

        return moments


def load_match_data(
    events_path: str,
    tracking_path: Optional[str] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Load match data from files."""
    with open(events_path, 'r') as f:
        events_data = json.load(f)

    # Handle various formats
    if isinstance(events_data, dict):
        events = events_data.get('events', events_data.get('data', []))
    else:
        events = events_data

    tracking = []
    if tracking_path:
        with open(tracking_path, 'r') as f:
            tracking_data = json.load(f)

        if isinstance(tracking_data, dict):
            tracking = tracking_data.get('frames', tracking_data.get('data', []))
        else:
            tracking = tracking_data

    return events, tracking


def analyze_match_file(
    events_path: str,
    tracking_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict:
    """
    Analyze a match from file paths.

    Args:
        events_path: Path to Wyscout events JSON
        tracking_path: Path to tracking data JSON (optional)
        output_path: Path to save analysis JSON (optional)

    Returns:
        Full analysis results
    """
    print(f"\nLoading events from: {events_path}")
    events, tracking = load_match_data(events_path, tracking_path)

    # Synchronize
    sync = EventTrackingSync()
    sync.load_wyscout_events(events)

    if tracking:
        print(f"Loading tracking from: {tracking_path}")
        sync.load_tracking_data(tracking)

    synced_events = sync.synchronize()

    # Analyze
    analyzer = DecisionAnalyzer()
    match_analysis = analyzer.analyze_match(synced_events)

    # Extract key moments
    extractor = KeyMomentsExtractor(analyzer)
    key_moments = extractor.extract_key_moments(synced_events)

    results = {
        'summary': {
            'total_events': match_analysis['total_events'],
            'events_with_context': match_analysis['events_with_context'],
            'avg_decision_quality': match_analysis['avg_decision_quality'],
            'optimal_decisions': match_analysis['optimal_decisions'],
            'suboptimal_decisions': match_analysis['suboptimal_decisions'],
        },
        'key_moments': key_moments,
        'detailed_analyses': match_analysis['analyses'][:50],  # Limit for readability
    }

    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nAnalysis saved to: {output_path}")

    return results


def print_analysis_report(results: Dict) -> None:
    """Print a formatted analysis report."""
    summary = results['summary']
    key_moments = results['key_moments']

    print("\n" + "="*60)
    print("MATCH DECISION ANALYSIS REPORT")
    print("="*60)

    print(f"\nTotal events analyzed: {summary['total_events']}")
    print(f"Events with tracking context: {summary['events_with_context']}")

    if summary['avg_decision_quality']:
        print(f"\nAverage decision quality: {summary['avg_decision_quality']:.1%}")
        print(f"Optimal decisions (>90%): {summary['optimal_decisions']}")
        print(f"Suboptimal decisions (<70%): {summary['suboptimal_decisions']}")

    print("\n" + "-"*60)
    print("KEY MOMENTS")
    print("-"*60)

    if key_moments['high_xG_chances']:
        print("\nHigh xG Chances:")
        for moment in key_moments['high_xG_chances'][:5]:
            period_str = "1H" if moment['period'] == 1 else "2H"
            mins = int(moment['timestamp'] // 60)
            secs = int(moment['timestamp'] % 60)
            result = "GOAL" if moment['successful'] else "missed"
            print(f"  [{period_str} {mins}:{secs:02d}] xG={moment.get('xG', 0):.2f} - {result}")

    if key_moments['missed_opportunities']:
        print("\nMissed Opportunities (better options available):")
        for moment in key_moments['missed_opportunities'][:5]:
            period_str = "1H" if moment['period'] == 1 else "2H"
            mins = int(moment['timestamp'] // 60)
            secs = int(moment['timestamp'] % 60)
            quality = moment['decision_quality'] or 0
            print(f"  [{period_str} {mins}:{secs:02d}] {moment['event_type']} - quality: {quality:.1%}")
            if moment['best_option']:
                best = moment['best_option']
                print(f"    Better option: {best['type']} (EV={best.get('expected_value', 0):.3f})")

    if key_moments['critical_errors']:
        print("\nCritical Errors:")
        for moment in key_moments['critical_errors'][:5]:
            period_str = "1H" if moment['period'] == 1 else "2H"
            mins = int(moment['timestamp'] // 60)
            secs = int(moment['timestamp'] % 60)
            quality = moment['decision_quality'] or 0
            print(f"  [{period_str} {mins}:{secs:02d}] {moment['event_type']} - quality: {quality:.1%}")

    print("\n" + "="*60)


def demo():
    """Demo with sample data."""
    # Sample Wyscout events
    sample_events = [
        {
            'eventId': 1,
            'eventName': 'Pass',
            'subEventName': 'Simple pass',
            'eventSec': 125.5,
            'matchPeriod': '1H',
            'playerId': 'player_1',
            'teamId': 'home',
            'positions': [
                {'x': 50, 'y': 50},
                {'x': 60, 'y': 45},
            ],
            'tags': [{'id': 1801, 'name': 'accurate'}],
        },
        {
            'eventId': 2,
            'eventName': 'Shot',
            'subEventName': 'Shot',
            'eventSec': 130.2,
            'matchPeriod': '1H',
            'playerId': 'player_3',
            'teamId': 'home',
            'positions': [
                {'x': 88, 'y': 45},
            ],
            'tags': [{'name': 'Goal'}],
        },
    ]

    # Sample tracking frames
    sample_tracking = [
        {
            'timestamp': 125.4,
            'period': 1,
            'players': [
                {'playerId': 'player_1', 'team': 'home', 'x': 52.5, 'y': 34.0},
                {'playerId': 'player_2', 'team': 'home', 'x': 63.0, 'y': 30.0},
                {'playerId': 'player_3', 'team': 'home', 'x': 70.0, 'y': 40.0},
                {'playerId': 'opp_1', 'team': 'away', 'x': 58.0, 'y': 35.0},
                {'playerId': 'opp_2', 'team': 'away', 'x': 65.0, 'y': 32.0},
            ],
            'ball': {'x': 52.5, 'y': 34.0, 'z': 0},
        },
        {
            'timestamp': 130.1,
            'period': 1,
            'players': [
                {'playerId': 'player_1', 'team': 'home', 'x': 55.0, 'y': 36.0},
                {'playerId': 'player_2', 'team': 'home', 'x': 80.0, 'y': 28.0},
                {'playerId': 'player_3', 'team': 'home', 'x': 92.0, 'y': 30.0},
                {'playerId': 'opp_1', 'team': 'away', 'x': 75.0, 'y': 32.0},
                {'playerId': 'opp_2', 'team': 'away', 'x': 95.0, 'y': 35.0},
            ],
            'ball': {'x': 92.0, 'y': 30.0, 'z': 0},
        },
    ]

    # Synchronize
    sync = EventTrackingSync()
    sync.load_wyscout_events(sample_events)
    sync.load_tracking_data(sample_tracking)
    synced_events = sync.synchronize()

    # Analyze
    analyzer = DecisionAnalyzer()
    match_analysis = analyzer.analyze_match(synced_events)

    # Extract key moments
    extractor = KeyMomentsExtractor(analyzer)
    key_moments = extractor.extract_key_moments(synced_events)

    results = {
        'summary': {
            'total_events': match_analysis['total_events'],
            'events_with_context': match_analysis['events_with_context'],
            'avg_decision_quality': match_analysis['avg_decision_quality'],
            'optimal_decisions': match_analysis['optimal_decisions'],
            'suboptimal_decisions': match_analysis['suboptimal_decisions'],
        },
        'key_moments': key_moments,
        'detailed_analyses': match_analysis['analyses'],
    }

    print_analysis_report(results)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Football Decision Analysis Tool')
    parser.add_argument('--events', '-e', help='Path to Wyscout events JSON file')
    parser.add_argument('--tracking', '-t', help='Path to tracking data JSON file')
    parser.add_argument('--output', '-o', help='Path to save analysis JSON')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample data')

    args = parser.parse_args()

    if args.demo or not args.events:
        demo()
    else:
        results = analyze_match_file(args.events, args.tracking, args.output)
        print_analysis_report(results)
