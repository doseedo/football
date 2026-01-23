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
import math
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request

from src.decision_engine import (
    Position, Player, GameState, GameStateEvaluator,
    EliminationCalculator, DefensiveBlock, BlockType,
    PitchGeometry, HALF_LENGTH, HALF_WIDTH,
)

app = Flask(__name__)

# Initialize evaluator
evaluator = GameStateEvaluator()


def create_default_scenario():
    """Create default game scenario.

    Pitch is 105m x 68m (standard).
    Coordinates are centered: x from -52.5 to 52.5, y from -34 to 34.
    Team A attacks toward x=52.5 (right goal).
    """
    # Team A (attacking left to right) - green
    team_a = [
        {"id": "A1", "x": -45, "y": 0, "team": "attack"},      # GK
        {"id": "A2", "x": -30, "y": -20, "team": "attack"},    # LB
        {"id": "A3", "x": -32, "y": -6, "team": "attack"},     # CB
        {"id": "A4", "x": -32, "y": 6, "team": "attack"},      # CB
        {"id": "A5", "x": -30, "y": 20, "team": "attack"},     # RB
        {"id": "A6", "x": -10, "y": -15, "team": "attack"},    # LM
        {"id": "A7", "x": -5, "y": 0, "team": "attack"},       # CM (ball carrier)
        {"id": "A8", "x": -10, "y": 15, "team": "attack"},     # RM
        {"id": "A9", "x": 20, "y": -12, "team": "attack"},     # LW
        {"id": "A10", "x": 25, "y": 5, "team": "attack"},      # ST
        {"id": "A11", "x": 18, "y": 18, "team": "attack"},     # RW
    ]

    # Team B (defending) - red
    team_b = [
        {"id": "D1", "x": 48, "y": 0, "team": "defense"},      # GK
        {"id": "D2", "x": 35, "y": -22, "team": "defense"},    # LB
        {"id": "D3", "x": 38, "y": -8, "team": "defense"},     # CB
        {"id": "D4", "x": 38, "y": 4, "team": "defense"},      # CB
        {"id": "D5", "x": 35, "y": 18, "team": "defense"},     # RB
        {"id": "D6", "x": 20, "y": -10, "team": "defense"},    # LM
        {"id": "D7", "x": 15, "y": 5, "team": "defense"},      # CM
        {"id": "D8", "x": 18, "y": 15, "team": "defense"},     # RM
        {"id": "D9", "x": -10, "y": -5, "team": "defense"},    # Pressing
        {"id": "D10", "x": -15, "y": 10, "team": "defense"},   # Pressing
        {"id": "D11", "x": -20, "y": 0, "team": "defense"},    # Pressing
    ]

    return team_a, team_b


# Store current state
current_state = {
    'team_a': [],
    'team_b': [],
    'ball_position': {"x": -5, "y": 0},
    'ball_carrier_id': "A7",
}

# Initialize with default
current_state['team_a'], current_state['team_b'] = create_default_scenario()


def evaluate_current_state():
    """Evaluate the current game state using the decision engine."""
    # Create Player objects
    attackers = [
        Player(id=p['id'], position=Position(p['x'], p['y']), team="attack")
        for p in current_state['team_a']
    ]
    defenders = [
        Player(id=p['id'], position=Position(p['x'], p['y']), team="defense")
        for p in current_state['team_b']
    ]

    # Find ball carrier
    ball_carrier = None
    for a in attackers:
        if a.id == current_state['ball_carrier_id']:
            ball_carrier = a
            break

    if ball_carrier is None and attackers:
        ball_carrier = attackers[0]

    # Create game state
    ball_pos = Position(
        current_state['ball_position']['x'],
        current_state['ball_position']['y']
    )

    state = GameState(
        ball_position=ball_pos,
        ball_carrier=ball_carrier,
        attackers=attackers,
        defenders=defenders,
    )

    # Evaluate
    evaluated = evaluator.evaluate(state)
    return evaluated


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    """Get current game state."""
    players = []
    for p in current_state['team_a']:
        players.append({
            'id': p['id'],
            'team': 'attack',
            'x': p['x'],
            'y': p['y'],
        })
    for p in current_state['team_b']:
        players.append({
            'id': p['id'],
            'team': 'defense',
            'x': p['x'],
            'y': p['y'],
        })

    return jsonify({
        'players': players,
        'ball': current_state['ball_position'],
        'ball_carrier': current_state['ball_carrier_id'],
    })


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Evaluate current state and return analysis."""
    try:
        evaluated = evaluate_current_state()

        # Build elimination info
        elimination_info = []
        if evaluated.elimination_state:
            for result in evaluated.elimination_state.defenders:
                elimination_info.append({
                    'id': result.defender.id,
                    'eliminated': result.is_eliminated,
                    'time_margin': result.time_margin,
                })

        # Build actions info
        actions_info = []
        for action in evaluated.available_actions[:5]:
            actions_info.append({
                'type': action.action_type.value,
                'target': {'x': action.target.x, 'y': action.target.y},
                'success_prob': action.success_probability,
                'expected_value': action.expected_value,
            })

        return jsonify({
            'score': {
                'total': evaluated.score.total,
                'elimination': evaluated.score.elimination_score,
                'proximity': evaluated.score.proximity_score,
                'angle': evaluated.score.angle_score,
                'density': evaluated.score.density_score,
                'compactness': evaluated.score.compactness_score,
                'action': evaluated.score.action_score,
            },
            'elimination': {
                'count': evaluated.elimination_state.eliminated_count if evaluated.elimination_state else 0,
                'ratio': evaluated.elimination_state.elimination_ratio if evaluated.elimination_state else 0,
                'defenders': elimination_info,
            },
            'actions': actions_info,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/update_player', methods=['POST'])
def update_player():
    """Update a player's position."""
    data = request.json
    player_id = data.get('id')
    new_x = data.get('x')
    new_y = data.get('y')

    # Find and update player
    for p in current_state['team_a']:
        if p['id'] == player_id:
            p['x'] = new_x
            p['y'] = new_y
            # Update ball if this is the carrier
            if player_id == current_state['ball_carrier_id']:
                current_state['ball_position'] = {'x': new_x, 'y': new_y}
            return jsonify({'success': True})

    for p in current_state['team_b']:
        if p['id'] == player_id:
            p['x'] = new_x
            p['y'] = new_y
            return jsonify({'success': True})

    return jsonify({'error': 'Player not found'}), 404


@app.route('/api/set_ball_carrier', methods=['POST'])
def set_ball_carrier():
    """Set the ball carrier."""
    data = request.json
    player_id = data.get('id')

    # Find player
    for p in current_state['team_a']:
        if p['id'] == player_id:
            current_state['ball_carrier_id'] = player_id
            current_state['ball_position'] = {'x': p['x'], 'y': p['y']}
            return jsonify({'success': True})

    return jsonify({'error': 'Player not found or not an attacker'}), 400


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset to default scenario."""
    current_state['team_a'], current_state['team_b'] = create_default_scenario()
    current_state['ball_position'] = {"x": -5, "y": 0}
    current_state['ball_carrier_id'] = "A7"
    return jsonify({'success': True})


if __name__ == '__main__':
    print("=" * 50)
    print("Decision Engine Web UI")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
