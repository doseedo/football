#!/usr/bin/env python3
"""
Decision Engine Web UI.

Simple Flask app to visualize the tactical board.

Usage:
    python ui/app.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
from src.decision_engine import DecisionEngine, PlayerState

app = Flask(__name__)
engine = DecisionEngine()


def create_default_scenario():
    """Create default game scenario.

    Pitch is 120 yards x 75 yards.
    Coordinates are centered: x from -60 to 60, y from -37.5 to 37.5.
    Team A attacks toward x=60 (right goal).
    """
    import math

    # Team A (attacking left to right, facing_angle=0 means facing goal)
    team_a = [
        PlayerState(player_id=1, team=0, position=(-50, 0), max_speed=6.0, is_goalkeeper=True, facing_angle=0),
        PlayerState(player_id=2, team=0, position=(-35, -22), max_speed=7.5, facing_angle=0.2),
        PlayerState(player_id=3, team=0, position=(-37, -6), max_speed=7.5, facing_angle=0),
        PlayerState(player_id=4, team=0, position=(-37, 6), max_speed=7.5, facing_angle=0),
        PlayerState(player_id=5, team=0, position=(-35, 22), max_speed=7.5, facing_angle=-0.2),
        PlayerState(player_id=6, team=0, position=(-12, -18), max_speed=8.0, facing_angle=0.3),
        PlayerState(player_id=7, team=0, position=(-6, 0), max_speed=8.0, facing_angle=0),
        PlayerState(player_id=8, team=0, position=(-12, 18), max_speed=8.0, facing_angle=-0.3),
        PlayerState(player_id=9, team=0, position=(22, -14), max_speed=8.5, facing_angle=0.4),
        PlayerState(player_id=10, team=0, position=(28, 6), max_speed=8.5, facing_angle=0),
        PlayerState(player_id=11, team=0, position=(20, 20), max_speed=8.5, facing_angle=-0.4),
    ]

    # Team B (defending, facing_angle=pi means facing own goal/attackers)
    team_b = [
        PlayerState(player_id=1, team=1, position=(55, 0), max_speed=6.0, is_goalkeeper=True, facing_angle=math.pi),
        PlayerState(player_id=2, team=1, position=(40, -25), max_speed=7.5, facing_angle=math.pi),
        PlayerState(player_id=3, team=1, position=(43, -10), max_speed=7.5, facing_angle=math.pi),
        PlayerState(player_id=4, team=1, position=(43, 4), max_speed=7.5, facing_angle=math.pi),
        PlayerState(player_id=5, team=1, position=(40, 20), max_speed=7.5, facing_angle=math.pi),
        PlayerState(player_id=6, team=1, position=(22, -12), max_speed=8.0, facing_angle=math.pi),
        PlayerState(player_id=7, team=1, position=(17, 6), max_speed=8.0, facing_angle=math.pi),
        PlayerState(player_id=8, team=1, position=(20, 17), max_speed=8.0, facing_angle=math.pi),
        PlayerState(player_id=9, team=1, position=(-12, -6), max_speed=8.5, facing_angle=math.pi),
        PlayerState(player_id=10, team=1, position=(-18, 12), max_speed=8.5, facing_angle=math.pi),
        PlayerState(player_id=11, team=1, position=(-23, 0), max_speed=8.5, facing_angle=math.pi),
    ]

    return team_a, team_b


# Store current state
current_state = {
    'team_a': [],
    'team_b': [],
    'ball_position': (-6, 0),  # Near center, slightly on Team A's side
    'ball_carrier_id': 7,
}

# Initialize with default
current_state['team_a'], current_state['team_b'] = create_default_scenario()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    """Get current game state."""
    players = []
    for p in current_state['team_a'] + current_state['team_b']:
        players.append({
            'id': p.player_id,
            'team': p.team,
            'x': p.position[0],
            'y': p.position[1],
        })

    return jsonify({
        'players': players,
        'ball': {
            'x': current_state['ball_position'][0],
            'y': current_state['ball_position'][1],
        },
        'ball_carrier_id': current_state['ball_carrier_id'],
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Run decision engine analysis."""
    data = request.json or {}

    # Update ball position if provided
    if 'ball_x' in data and 'ball_y' in data:
        current_state['ball_position'] = (data['ball_x'], data['ball_y'])
    if 'ball_carrier_id' in data:
        current_state['ball_carrier_id'] = data['ball_carrier_id']

    # Run analysis
    result = engine.analyze(
        ball_position=current_state['ball_position'],
        ball_carrier_id=current_state['ball_carrier_id'],
        possession_team=0,
        teammates=current_state['team_a'],
        opponents=current_state['team_b'],
        timestamp=0.0
    )

    # Helper to convert numpy types to Python native
    def to_native(val):
        if hasattr(val, 'item'):
            return val.item()
        return val

    # Format response
    gaps = []
    for gap in result.defensive_gaps:
        gaps.append({
            'x': to_native(gap.location[0]),
            'y': to_native(gap.location[1]),
            'size': to_native(gap.size),
            'time_to_close': to_native(gap.time_to_close),
            'exploitable': bool(gap.exploitable),
            'xg': to_native(gap.xg_if_exploited),
        })

    options = []
    for opt in result.progression_options[:10]:
        options.append({
            'action': opt.action_type,
            'target_player': opt.target_player_id,
            'target_x': to_native(opt.target_position[0]),
            'target_y': to_native(opt.target_position[1]),
            'success_prob': to_native(opt.success_probability),
            'intercept_prob': to_native(opt.interception_probability),
            'xg_current': to_native(opt.xg_current),
            'xg_target': to_native(opt.xg_target),
            'xg_gain': to_native(opt.xg_gain),
            'ev': to_native(opt.expected_value),
            'recommendation': opt.recommendation,
            'receiver_pressure': to_native(opt.receiver_pressure),
            'receiver_facing_goal': bool(opt.receiver_facing_goal),
        })

    best = None
    if result.best_option:
        best = {
            'action': result.best_option.action_type,
            'target_player': result.best_option.target_player_id,
            'target_x': to_native(result.best_option.target_position[0]),
            'target_y': to_native(result.best_option.target_position[1]),
            'success_prob': to_native(result.best_option.success_probability),
            'ev': to_native(result.best_option.expected_value),
            'recommendation': result.best_option.recommendation,
        }

    return jsonify({
        'gaps': gaps,
        'options': options,
        'best_option': best,
        'total_options': int(result.total_options),
        'high_value_options': int(result.high_value_options),
        'safe_options': int(result.safe_options),
    })


@app.route('/api/move_player', methods=['POST'])
def move_player():
    """Move a player to new position."""
    data = request.json
    player_id = data.get('player_id')
    team = data.get('team')
    new_x = data.get('x')
    new_y = data.get('y')

    # Find and update player
    players = current_state['team_a'] if team == 0 else current_state['team_b']
    for i, p in enumerate(players):
        if p.player_id == player_id:
            players[i] = PlayerState(
                player_id=p.player_id,
                team=p.team,
                position=(new_x, new_y),
                max_speed=p.max_speed,
            )
            break

    return jsonify({'success': True})


@app.route('/api/move_ball', methods=['POST'])
def move_ball():
    """Move ball to new position."""
    data = request.json
    current_state['ball_position'] = (data.get('x', 0), data.get('y', 0))
    if 'carrier_id' in data:
        current_state['ball_carrier_id'] = data['carrier_id']
    return jsonify({'success': True})


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset to default scenario."""
    current_state['team_a'], current_state['team_b'] = create_default_scenario()
    current_state['ball_position'] = (-6, 0)  # Near center
    current_state['ball_carrier_id'] = 7
    return jsonify({'success': True})


# Cache for xG zones (computed once at startup)
_xg_zones_cache = None

def _compute_xg_zones():
    """Pre-compute xG zones with 3x3 yard resolution for performance."""
    zones = []
    step = 3  # 3-yard grid = 40x25 = 1000 zones (much faster than 9000)
    for x in range(0, 120, step):
        for y in range(0, 75, step):
            centered_x = x - 60 + step // 2
            centered_y = y - 37 + step // 2
            xg = engine.xg_model.get_xg((centered_x, centered_y))
            zones.append({
                'x': centered_x,
                'y': centered_y,
                'xg': float(xg),
                'width': step,
                'height': step,
            })
    return {
        'zones': zones,
        'pitch_length': 120,
        'pitch_width': 75,
        'grid_resolution': step,
        'total_zones': len(zones),
    }

@app.route('/api/xg_zones')
def get_xg_zones():
    """Get xG zone data for visualization (cached)."""
    global _xg_zones_cache
    if _xg_zones_cache is None:
        print("Computing xG zones (first request only)...")
        _xg_zones_cache = _compute_xg_zones()
        print(f"Cached {_xg_zones_cache['total_zones']} zones")
    return jsonify(_xg_zones_cache)


if __name__ == '__main__':
    print("Starting Decision Engine UI...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
