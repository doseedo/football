#!/usr/bin/env python3
"""
Decision Engine Web UI.

Flask app to visualize the tactical board with gap analysis,
passing options, and xG zones.

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

# Pitch dimensions in yards (frontend uses yards, not meters)
PITCH_LENGTH_YDS = 120
PITCH_WIDTH_YDS = 75

def meters_to_yards(m):
    return m * 1.09361

def yards_to_meters(y):
    return y / 1.09361


def create_default_scenario():
    """Create default game scenario with numbered players.

    Team 0 = attacking (blue), Team 1 = defending (red)
    Player positions in yards, centered coordinate system.
    """
    # Team 0 (attacking left to right) - blue
    attackers = [
        {"id": 1, "x": -55, "y": 0, "team": 0},      # GK
        {"id": 2, "x": -35, "y": -25, "team": 0},    # LB
        {"id": 3, "x": -38, "y": -8, "team": 0},     # CB
        {"id": 4, "x": -38, "y": 8, "team": 0},      # CB
        {"id": 5, "x": -35, "y": 25, "team": 0},     # RB
        {"id": 6, "x": -15, "y": -18, "team": 0},    # LM
        {"id": 7, "x": -10, "y": 0, "team": 0},      # CM (ball carrier)
        {"id": 8, "x": -15, "y": 18, "team": 0},     # RM
        {"id": 9, "x": 25, "y": -15, "team": 0},     # LW
        {"id": 10, "x": 30, "y": 5, "team": 0},      # ST
        {"id": 11, "x": 22, "y": 20, "team": 0},     # RW
    ]

    # Team 1 (defending) - red
    defenders = [
        {"id": 1, "x": 55, "y": 0, "team": 1},       # GK
        {"id": 2, "x": 40, "y": -28, "team": 1},     # LB
        {"id": 3, "x": 44, "y": -10, "team": 1},     # CB
        {"id": 4, "x": 44, "y": 5, "team": 1},       # CB
        {"id": 5, "x": 40, "y": 22, "team": 1},      # RB
        {"id": 6, "x": 25, "y": -12, "team": 1},     # LM
        {"id": 7, "x": 18, "y": 8, "team": 1},       # CM
        {"id": 8, "x": 22, "y": 18, "team": 1},      # RM
        {"id": 9, "x": -5, "y": -8, "team": 1},      # Pressing
        {"id": 10, "x": -12, "y": 12, "team": 1},    # Pressing
        {"id": 11, "x": -18, "y": 0, "team": 1},     # Pressing
    ]

    return attackers, defenders


# Store current state
current_state = {
    'attackers': [],
    'defenders': [],
    'ball': {"x": -10, "y": 0},
}

# Initialize with default
current_state['attackers'], current_state['defenders'] = create_default_scenario()


def get_all_players():
    """Get all players as a combined list."""
    return current_state['attackers'] + current_state['defenders']


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_xg(x, y):
    """Calculate expected goals at a position.

    Simple model based on distance and angle to goal.
    Goal is at x=60 (right side).
    """
    goal_x = 60
    goal_y = 0

    # Distance to goal center
    dist = calculate_distance(x, y, goal_x, goal_y)

    # Angle to goal (narrower = harder)
    goal_width = 8  # yards
    angle = math.atan2(goal_width/2, dist) * 2  # Angle in radians

    # Base xG from distance (exponential decay)
    if dist <= 6:
        base_xg = 0.35  # Very close
    elif dist <= 12:
        base_xg = 0.25
    elif dist <= 18:
        base_xg = 0.12
    elif dist <= 25:
        base_xg = 0.06
    elif dist <= 35:
        base_xg = 0.03
    else:
        base_xg = 0.01

    # Angle modifier (central = better)
    angle_factor = math.cos(math.atan2(abs(y), max(1, goal_x - x)))

    # Combine
    xg = base_xg * (0.5 + 0.5 * angle_factor)

    # Bonus for being inside the box
    if x > 42 and abs(y) < 22:
        xg *= 1.3

    return min(0.95, max(0.001, xg))


def find_gaps():
    """Find gaps in the defensive structure."""
    gaps = []
    defenders = current_state['defenders']
    ball = current_state['ball']

    # Only consider outfield defenders (not GK)
    outfield_defenders = [d for d in defenders if d['id'] != 1]

    # Sort by x position (closest to attacking goal first)
    sorted_defs = sorted(outfield_defenders, key=lambda d: -d['x'])

    # Find gaps between adjacent defenders
    for i in range(len(sorted_defs) - 1):
        d1 = sorted_defs[i]
        d2 = sorted_defs[i + 1]

        # Gap center
        gap_x = (d1['x'] + d2['x']) / 2
        gap_y = (d1['y'] + d2['y']) / 2

        # Gap size (distance between defenders)
        gap_size = calculate_distance(d1['x'], d1['y'], d2['x'], d2['y'])

        # Only consider gaps in front of the ball
        if gap_x > ball['x'] and gap_size > 8:
            # Calculate time for nearest defender to close the gap
            min_time = float('inf')
            for d in outfield_defenders:
                dist_to_gap = calculate_distance(d['x'], d['y'], gap_x, gap_y)
                time_to_close = dist_to_gap / 7.0  # Assume 7 yds/sec sprint
                min_time = min(min_time, time_to_close)

            # xG if we exploit this gap
            xg = calculate_xg(gap_x, gap_y)

            # Gap is exploitable if it takes > 1 second to close
            exploitable = min_time > 1.0 and gap_size > 12

            gaps.append({
                'x': gap_x,
                'y': gap_y,
                'size': gap_size,
                'time_to_close': min_time,
                'xg': xg,
                'exploitable': exploitable,
            })

    # Also check horizontal gaps (between lines)
    x_lines = {}
    for d in outfield_defenders:
        x_bucket = round(d['x'] / 10) * 10
        if x_bucket not in x_lines:
            x_lines[x_bucket] = []
        x_lines[x_bucket].append(d)

    sorted_lines = sorted(x_lines.keys(), reverse=True)
    for i in range(len(sorted_lines) - 1):
        line1_x = sorted_lines[i]
        line2_x = sorted_lines[i + 1]

        gap_between_lines = line1_x - line2_x
        if gap_between_lines > 15 and line2_x > ball['x']:
            gap_x = (line1_x + line2_x) / 2
            gap_y = 0  # Center of pitch

            min_time = float('inf')
            for d in outfield_defenders:
                dist_to_gap = calculate_distance(d['x'], d['y'], gap_x, gap_y)
                time_to_close = dist_to_gap / 7.0
                min_time = min(min_time, time_to_close)

            xg = calculate_xg(gap_x, gap_y)

            gaps.append({
                'x': gap_x,
                'y': gap_y,
                'size': gap_between_lines,
                'time_to_close': min_time,
                'xg': xg,
                'exploitable': min_time > 0.8,
            })

    # Sort by xG (best gaps first)
    gaps.sort(key=lambda g: -g['xg'])

    return gaps[:10]


def analyze_passing_options():
    """Analyze all passing options from current ball position."""
    options = []
    ball = current_state['ball']
    attackers = current_state['attackers']
    defenders = current_state['defenders']

    # Current xG at ball position
    current_xg = calculate_xg(ball['x'], ball['y'])

    # Analyze pass to each teammate
    for attacker in attackers:
        if attacker['id'] == 1:  # Skip GK
            continue

        # Don't pass to yourself (player at ball position)
        dist_to_ball = calculate_distance(attacker['x'], attacker['y'], ball['x'], ball['y'])
        if dist_to_ball < 3:
            continue

        target_x = attacker['x']
        target_y = attacker['y']

        # Calculate pass success probability
        pass_dist = dist_to_ball

        # Find nearest defender to passing lane
        min_intercept_dist = float('inf')
        for defender in defenders:
            # Distance from defender to the pass line
            intercept_dist = point_to_line_distance(
                defender['x'], defender['y'],
                ball['x'], ball['y'],
                target_x, target_y
            )
            min_intercept_dist = min(min_intercept_dist, intercept_dist)

        # Success probability based on distance and interception risk
        base_success = max(0.3, 1.0 - pass_dist / 80)
        intercept_factor = min(1.0, min_intercept_dist / 8)
        success_prob = base_success * intercept_factor

        # xG at target position
        target_xg = calculate_xg(target_x, target_y)
        xg_gain = target_xg - current_xg

        # Pressure on receiver
        min_defender_dist = float('inf')
        for defender in defenders:
            d = calculate_distance(defender['x'], defender['y'], target_x, target_y)
            min_defender_dist = min(min_defender_dist, d)
        receiver_pressure = max(0, 1.0 - min_defender_dist / 15)

        # Is receiver facing goal?
        receiver_facing_goal = target_x > ball['x']

        # Expected value
        ev = success_prob * (target_xg + 0.01) - (1 - success_prob) * 0.05

        # Recommendation
        if ev > 0.01 and success_prob > 0.7:
            rec = "HIGH_VALUE"
        elif success_prob > 0.8:
            rec = "SAFE"
        elif ev > 0 and success_prob > 0.5:
            rec = "MODERATE"
        elif success_prob > 0.4:
            rec = "LOW_VALUE"
        else:
            rec = "AVOID"

        options.append({
            'action': 'pass',
            'target_x': target_x,
            'target_y': target_y,
            'target_player': attacker['id'],
            'success_prob': success_prob,
            'xg_gain': xg_gain,
            'xg_target': target_xg,
            'ev': ev,
            'recommendation': rec,
            'receiver_pressure': receiver_pressure,
            'receiver_facing_goal': receiver_facing_goal,
            'intercept_prob': 1 - intercept_factor,
        })

    # Analyze shooting option
    goal_x = 60
    goal_y = 0
    shoot_dist = calculate_distance(ball['x'], ball['y'], goal_x, goal_y)

    if shoot_dist < 40:  # Only consider shots from reasonable distance
        shot_xg = calculate_xg(ball['x'], ball['y'])

        # Check for blocking defenders
        block_prob = 0
        for defender in defenders:
            block_dist = point_to_line_distance(
                defender['x'], defender['y'],
                ball['x'], ball['y'],
                goal_x, goal_y
            )
            if block_dist < 3 and defender['x'] > ball['x']:
                block_prob = max(block_prob, 0.3)
            elif block_dist < 6 and defender['x'] > ball['x']:
                block_prob = max(block_prob, 0.15)

        shot_success = shot_xg * (1 - block_prob)
        shot_ev = shot_success - (1 - shot_success) * 0.02

        if shot_xg > 0.15:
            rec = "HIGH_VALUE"
        elif shot_xg > 0.08:
            rec = "MODERATE"
        elif shot_xg > 0.04:
            rec = "LOW_VALUE"
        else:
            rec = "AVOID"

        options.append({
            'action': 'shoot',
            'target_x': goal_x,
            'target_y': goal_y,
            'success_prob': shot_success,
            'xg_target': shot_xg,
            'xg_gain': shot_xg,
            'ev': shot_ev,
            'recommendation': rec,
            'intercept_prob': block_prob,
        })

    # Analyze dribble options
    dribble_directions = [
        (10, 0),   # Forward
        (8, 8),    # Forward-right
        (8, -8),   # Forward-left
        (5, 12),   # Wide right
        (5, -12),  # Wide left
    ]

    for dx, dy in dribble_directions:
        target_x = ball['x'] + dx
        target_y = ball['y'] + dy

        # Check bounds
        if abs(target_x) > 58 or abs(target_y) > 36:
            continue

        # Check for defenders in the way
        min_defender_dist = float('inf')
        for defender in defenders:
            d = calculate_distance(defender['x'], defender['y'], target_x, target_y)
            min_defender_dist = min(min_defender_dist, d)

        if min_defender_dist < 4:
            continue  # Too close to defender

        success_prob = min(0.9, min_defender_dist / 12)
        target_xg = calculate_xg(target_x, target_y)
        xg_gain = target_xg - current_xg
        ev = success_prob * xg_gain - (1 - success_prob) * 0.03

        if ev > 0.005 and success_prob > 0.6:
            rec = "MODERATE"
        elif success_prob > 0.7:
            rec = "SAFE"
        else:
            rec = "LOW_VALUE"

        options.append({
            'action': 'dribble',
            'target_x': target_x,
            'target_y': target_y,
            'success_prob': success_prob,
            'xg_gain': xg_gain,
            'xg_target': target_xg,
            'ev': ev,
            'recommendation': rec,
        })

    # Sort by expected value
    options.sort(key=lambda o: -o['ev'])

    return options


def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate perpendicular distance from point to line segment."""
    line_len = calculate_distance(x1, y1, x2, y2)
    if line_len == 0:
        return calculate_distance(px, py, x1, y1)

    # Project point onto line
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_len ** 2)))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    return calculate_distance(px, py, proj_x, proj_y)


def generate_xg_zones():
    """Generate xG zones across the attacking half."""
    zones = []
    zone_size = 5  # 5 yard grid

    # Only generate for attacking half (x > 0)
    for x in range(-10, 61, zone_size):
        for y in range(-35, 36, zone_size):
            xg = calculate_xg(x, y)
            zones.append({
                'x': x,
                'y': y,
                'xg': xg,
                'width': zone_size,
            })

    return zones


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    """Get current game state."""
    players = get_all_players()

    return jsonify({
        'players': players,
        'ball': current_state['ball'],
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze current state and return tactical options."""
    try:
        data = request.json or {}

        # Update ball position if provided
        if 'ball_x' in data and 'ball_y' in data:
            current_state['ball']['x'] = data['ball_x']
            current_state['ball']['y'] = data['ball_y']

        # Find gaps
        gaps = find_gaps()

        # Analyze options
        options = analyze_passing_options()

        # Count by recommendation
        high_value = sum(1 for o in options if o['recommendation'] == 'HIGH_VALUE')
        safe = sum(1 for o in options if o['recommendation'] == 'SAFE')

        # Best option
        best_option = options[0] if options else None

        return jsonify({
            'gaps': gaps,
            'options': options,
            'best_option': best_option,
            'total_options': len(options),
            'high_value_options': high_value,
            'safe_options': safe,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/xg_zones')
def get_xg_zones():
    """Get xG zones for visualization."""
    zones = generate_xg_zones()
    return jsonify({
        'zones': zones,
        'total_zones': len(zones),
    })


@app.route('/api/move_ball', methods=['POST'])
def move_ball():
    """Move the ball to a new position."""
    data = request.json
    current_state['ball']['x'] = data.get('x', 0)
    current_state['ball']['y'] = data.get('y', 0)
    return jsonify({'success': True})


@app.route('/api/move_player', methods=['POST'])
def move_player():
    """Move a player to a new position."""
    data = request.json
    player_id = data.get('player_id')
    team = data.get('team')
    new_x = data.get('x')
    new_y = data.get('y')

    # Find and update player
    player_list = current_state['attackers'] if team == 0 else current_state['defenders']

    for p in player_list:
        if p['id'] == player_id:
            p['x'] = new_x
            p['y'] = new_y
            return jsonify({'success': True})

    return jsonify({'error': 'Player not found'}), 404


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset to default scenario."""
    current_state['attackers'], current_state['defenders'] = create_default_scenario()
    current_state['ball'] = {"x": -10, "y": 0}
    return jsonify({'success': True})


if __name__ == '__main__':
    print("=" * 50)
    print("Decision Engine Web UI")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
