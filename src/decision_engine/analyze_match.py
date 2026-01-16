"""
Match Decision Analysis - Practical Implementation

This script provides practical ways to analyze matches using
the event-tracking synchronization system.

Usage:
    # Analyze with Wyscout events + tracking
    python analyze_match.py --events events.json --tracking tracking.json

    # Analyze SkillCorner match (tracking only - uses synthetic events)
    python analyze_match.py --skillcorner path/to/match/structured_data.json

    # Analyze with Wyscout events only (no tracking context)
    python analyze_match.py --events events.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as script or module
sys.path.insert(0, str(Path(__file__).parent))

from event_tracking_sync import (
    EventTrackingSync,
    DecisionAnalyzer,
    KeyMomentsExtractor,
    PlayerPosition,
    FrameSnapshot,
    SynchronizedEvent,
    print_analysis_report,
)


def load_skillcorner_as_tracking(file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load SkillCorner data and convert to tracking format.

    SkillCorner doesn't have Wyscout-style events, so we generate
    synthetic events based on ball possession changes.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    frames = data if isinstance(data, list) else data.get('data', [])

    tracking_frames = []
    synthetic_events = []

    last_possession_team = None
    event_id = 0

    for frame in frames:
        # Convert to tracking format
        tracking_frame = {
            'timestamp': frame.get('timestamp', frame.get('time', 0)),
            'period': frame.get('period', 1),
            'players': [],
            'ball': None,
        }

        # Parse players from SkillCorner format
        players = frame.get('data', frame.get('players', []))
        for p in players:
            player_data = {
                'playerId': str(p.get('track_id', p.get('trackable_object', p.get('id', '')))),
                'team': p.get('team_id', p.get('team', 'unknown')),
                'x': p.get('x', 0),
                'y': p.get('y', 0),
                'jersey_number': p.get('jersey_number', p.get('number')),
            }
            # Handle SkillCorner team naming
            if player_data['team'] in ['HOMETEAM', 'HOME']:
                player_data['team'] = 'home'
            elif player_data['team'] in ['AWAYTEAM', 'AWAY']:
                player_data['team'] = 'away'
            tracking_frame['players'].append(player_data)

        # Ball position
        ball = frame.get('ball', {})
        if ball:
            tracking_frame['ball'] = {
                'x': ball.get('x', 0),
                'y': ball.get('y', 0),
                'z': ball.get('z', 0),
            }

        tracking_frames.append(tracking_frame)

        # Generate synthetic events on possession changes
        possession = frame.get('possession', {})
        current_team = possession.get('team', possession.get('group', None))

        if current_team and current_team != last_possession_team:
            # Find player closest to ball (likely ball carrier)
            ball_x = ball.get('x', 52.5) if ball else 52.5
            ball_y = ball.get('y', 34) if ball else 34

            carrier = None
            min_dist = float('inf')

            for p in tracking_frame['players']:
                if p['team'] == current_team or p['team'] == current_team.lower():
                    dist = ((p['x'] - ball_x)**2 + (p['y'] - ball_y)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        carrier = p

            if carrier:
                event = {
                    'eventId': event_id,
                    'eventName': 'Pass',
                    'subEventName': 'Simple pass',
                    'eventSec': tracking_frame['timestamp'],
                    'matchPeriod': '1H' if tracking_frame['period'] == 1 else '2H',
                    'playerId': carrier['playerId'],
                    'teamId': carrier['team'],
                    'positions': [
                        {'x': carrier['x'] / 105 * 100, 'y': carrier['y'] / 68 * 100},
                    ],
                    'tags': [{'name': 'accurate'}],
                }
                synthetic_events.append(event)
                event_id += 1

            last_possession_team = current_team

    print(f"Loaded {len(tracking_frames)} frames from SkillCorner")
    print(f"Generated {len(synthetic_events)} synthetic events from possession data")

    return synthetic_events, tracking_frames


def load_wyscout_data(
    events_path: Optional[str] = None,
    tracking_path: Optional[str] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Load Wyscout events and/or tracking data."""
    events = []
    tracking = []

    if events_path:
        with open(events_path, 'r') as f:
            data = json.load(f)
        events = data if isinstance(data, list) else data.get('events', data.get('data', []))
        print(f"Loaded {len(events)} Wyscout events")

    if tracking_path:
        with open(tracking_path, 'r') as f:
            data = json.load(f)

        raw_tracking = data.get('tracking', data.get('frames', data))
        if isinstance(raw_tracking, list):
            for frame in raw_tracking:
                tracking_frame = {
                    'timestamp': frame.get('timestamp', 0),
                    'period': frame.get('period', 1),
                    'players': [],
                    'ball': None,
                }

                # Home players
                for p in frame.get('homePlayers', frame.get('home', [])):
                    # Convert Wyscout 0-100 to meters
                    tracking_frame['players'].append({
                        'playerId': str(p.get('playerId', p.get('id', ''))),
                        'team': 'home',
                        'x': p.get('x', 0) / 100 * 105,
                        'y': p.get('y', 0) / 100 * 68,
                        'jersey_number': p.get('jerseyNo'),
                    })

                # Away players
                for p in frame.get('awayPlayers', frame.get('away', [])):
                    tracking_frame['players'].append({
                        'playerId': str(p.get('playerId', p.get('id', ''))),
                        'team': 'away',
                        'x': p.get('x', 0) / 100 * 105,
                        'y': p.get('y', 0) / 100 * 68,
                        'jersey_number': p.get('jerseyNo'),
                    })

                # Ball
                ball = frame.get('ball', {})
                if ball:
                    tracking_frame['ball'] = {
                        'x': ball.get('x', 0) / 100 * 105,
                        'y': ball.get('y', 0) / 100 * 68,
                        'z': ball.get('z', 0),
                    }

                tracking.append(tracking_frame)

        print(f"Loaded {len(tracking)} tracking frames")

    return events, tracking


def analyze_match(
    events: List[Dict],
    tracking: List[Dict],
    output_path: Optional[str] = None
) -> Dict:
    """Run full analysis on a match."""
    # Synchronize
    sync = EventTrackingSync()
    sync.load_wyscout_events(events)

    if tracking:
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
        'detailed_analyses': match_analysis['analyses'][:100],
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    return results


def analyze_single_moment(
    events: List[Dict],
    tracking: List[Dict],
    timestamp: float,
    period: int = 1
) -> Dict:
    """
    Analyze a specific moment in the match.

    Useful for examining a particular decision.
    """
    sync = EventTrackingSync()
    sync.load_wyscout_events(events)
    sync.load_tracking_data(tracking)
    synced_events = sync.synchronize()

    # Find the event closest to the timestamp
    closest_event = None
    min_diff = float('inf')

    for event in synced_events:
        if event.period == period:
            diff = abs(event.timestamp - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_event = event

    if not closest_event:
        return {'error': 'No event found near that timestamp'}

    analyzer = DecisionAnalyzer()
    analysis = analyzer.analyze_event(closest_event)

    # Add context
    analysis['event_timestamp'] = closest_event.timestamp
    analysis['event_period'] = closest_event.period
    analysis['player_id'] = closest_event.player_id

    if closest_event.snapshot:
        analysis['players_in_frame'] = len(closest_event.snapshot.players)
        analysis['ball_position'] = {
            'x': closest_event.snapshot.ball.x if closest_event.snapshot.ball else None,
            'y': closest_event.snapshot.ball.y if closest_event.snapshot.ball else None,
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description='Match Decision Analysis')
    parser.add_argument('--events', '-e', help='Path to Wyscout events JSON')
    parser.add_argument('--tracking', '-t', help='Path to tracking data JSON')
    parser.add_argument('--skillcorner', '-s', help='Path to SkillCorner structured_data.json')
    parser.add_argument('--output', '-o', help='Path to save analysis JSON')
    parser.add_argument('--moment', '-m', type=float, help='Analyze specific moment (timestamp in seconds)')
    parser.add_argument('--period', '-p', type=int, default=1, help='Period for moment analysis (1 or 2)')

    args = parser.parse_args()

    # Load data based on source
    if args.skillcorner:
        events, tracking = load_skillcorner_as_tracking(args.skillcorner)
    else:
        events, tracking = load_wyscout_data(args.events, args.tracking)

    if not events:
        print("Error: No events loaded. Provide --events or --skillcorner")
        return

    # Analyze
    if args.moment:
        # Single moment analysis
        result = analyze_single_moment(events, tracking, args.moment, args.period)
        print("\n" + "="*60)
        print(f"MOMENT ANALYSIS @ {args.period}H {int(args.moment//60)}:{int(args.moment%60):02d}")
        print("="*60)
        print(f"Event type: {result.get('event_type')}")
        print(f"Options available: {len(result.get('options', []))}")
        for opt in result.get('options', [])[:5]:
            print(f"  - {opt['type']}: EV={opt.get('expected_value', 0):.3f}")
        if result.get('best_option'):
            best = result['best_option']
            print(f"\nBest option: {best['type']} (EV={best.get('expected_value', 0):.3f})")
        if result.get('decision_quality'):
            print(f"Decision quality: {result['decision_quality']:.1%}")
    else:
        # Full match analysis
        results = analyze_match(events, tracking, args.output)
        print_analysis_report(results)


if __name__ == '__main__':
    main()
