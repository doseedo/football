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
    """Create default game scenario."""
    team_a = [
        PlayerState(player_id=1, team=0, position=(-45, 0), max_speed=6.0),
        PlayerState(player_id=2, team=0, position=(-30, -20), max_speed=7.5),
        PlayerState(player_id=3, team=0, position=(-32, -5), max_speed=7.5),
        PlayerState(player_id=4, team=0, position=(-32, 5), max_speed=7.5),
        PlayerState(player_id=5, team=0, position=(-30, 20), max_speed=7.5),
        PlayerState(player_id=6, team=0, position=(-10, -15), max_speed=8.0),
        PlayerState(player_id=7, team=0, position=(-5, 0), max_speed=8.0),
        PlayerState(player_id=8, team=0, position=(-10, 15), max_speed=8.0),
        PlayerState(player_id=9, team=0, position=(20, -12), max_speed=8.5),
        PlayerState(player_id=10, team=0, position=(25, 5), max_speed=8.5),
        PlayerState(player_id=11, team=0, position=(18, 18), max_speed=8.5),
    ]

    team_b = [
        PlayerState(player_id=1, team=1, position=(45, 0), max_speed=6.0),
        PlayerState(player_id=2, team=1, position=(30, -22), max_speed=7.5),
        PlayerState(player_id=3, team=1, position=(32, -8), max_speed=7.5),
        PlayerState(player_id=4, team=1, position=(33, 3), max_speed=7.5),
        PlayerState(player_id=5, team=1, position=(30, 18), max_speed=7.5),
        PlayerState(player_id=6, team=1, position=(15, -10), max_speed=8.0),
        PlayerState(player_id=7, team=1, position=(10, 5), max_speed=8.0),
        PlayerState(player_id=8, team=1, position=(12, 15), max_speed=8.0),
        PlayerState(player_id=9, team=1, position=(-15, -5), max_speed=8.5),
        PlayerState(player_id=10, team=1, position=(-20, 10), max_speed=8.5),
        PlayerState(player_id=11, team=1, position=(-25, 0), max_speed=8.5),
    ]

    return team_a, team_b


# Store current state
current_state = {
    'team_a': [],
    'team_b': [],
    'ball_position': (-5, 0),
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

    # Format response
    gaps = []
    for gap in result.defensive_gaps:
        gaps.append({
            'x': gap.location[0],
            'y': gap.location[1],
            'size': gap.size,
            'time_to_close': gap.time_to_close,
            'exploitable': gap.exploitable,
            'xg': gap.xg_if_exploited,
        })

    options = []
    for opt in result.progression_options[:10]:
        options.append({
            'action': opt.action_type,
            'target_player': opt.target_player_id,
            'target_x': opt.target_position[0],
            'target_y': opt.target_position[1],
            'success_prob': opt.success_probability,
            'intercept_prob': opt.interception_probability,
            'xg_current': opt.xg_current,
            'xg_target': opt.xg_target,
            'xg_gain': opt.xg_gain,
            'ev': opt.expected_value,
            'recommendation': opt.recommendation,
        })

    best = None
    if result.best_option:
        best = {
            'action': result.best_option.action_type,
            'target_player': result.best_option.target_player_id,
            'target_x': result.best_option.target_position[0],
            'target_y': result.best_option.target_position[1],
            'success_prob': result.best_option.success_probability,
            'ev': result.best_option.expected_value,
            'recommendation': result.best_option.recommendation,
        }

    return jsonify({
        'gaps': gaps,
        'options': options,
        'best_option': best,
        'total_options': result.total_options,
        'high_value_options': result.high_value_options,
        'safe_options': result.safe_options,
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
    current_state['ball_position'] = (-5, 0)
    current_state['ball_carrier_id'] = 7
    return jsonify({'success': True})


if __name__ == '__main__':
    print("Starting Decision Engine UI...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
