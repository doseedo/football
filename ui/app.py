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

# ============================================================
# PHYSICS CONSTANTS (all in yards and seconds)
# ============================================================
# Conversion: 1 meter = 1.09361 yards

def meters_to_yards(m):
    return m * 1.09361

def yards_to_meters(y):
    return y / 1.09361

# Player physics
# Sprint data: 0-10m in 1.8s, 0-30m in 4.2s
# Converted: 0-10.94 yards in 1.8s, 0-32.81 yards in 4.2s
ACCELERATION_DISTANCE = meters_to_yards(10)  # 10.94 yards - distance to reach top speed
ACCELERATION_TIME = 1.8  # seconds to cover acceleration distance
TOP_SPEED = meters_to_yards(8.33)  # ~9.1 yards/s (8.33 m/s top speed)
REACTION_TIME = 0.3  # seconds before defender reacts

# Dribbling physics (50% of sprint speed, proportional acceleration)
DRIBBLE_SPEED_RATIO = 0.5
DRIBBLE_TOP_SPEED = TOP_SPEED * DRIBBLE_SPEED_RATIO  # ~4.55 yards/s
DRIBBLE_ACCEL_DISTANCE = ACCELERATION_DISTANCE * DRIBBLE_SPEED_RATIO  # ~5.47 yards to reach top dribble speed
DRIBBLE_ACCEL_TIME = ACCELERATION_TIME  # Same time to accelerate (1.8s)

# Player dimensions
PLAYER_WIDTH = 1.0  # yards (interception reach to each side)

# Ball physics
PASS_SPEED_MIN = meters_to_yards(5)   # ~5.5 yards/s (soft pass)
PASS_SPEED_MAX = meters_to_yards(20)  # ~21.9 yards/s (hard pass)
SHOT_SPEED_MIN = meters_to_yards(20)  # ~21.9 yards/s (placed shot)
SHOT_SPEED_MAX = meters_to_yards(40)  # ~43.7 yards/s (powerful shot)
BALL_DECELERATION = meters_to_yards(0.5)  # ~0.55 yards/s² friction on grass


# ============================================================
# PHYSICS FUNCTIONS
# ============================================================

def time_to_run_distance(distance_yards):
    """Calculate time for a player to run a given distance.

    Uses acceleration model:
    - Phase 1 (0 to ACCELERATION_DISTANCE): Parabolic acceleration
    - Phase 2 (beyond): Constant top speed

    Args:
        distance_yards: Distance in yards

    Returns:
        Time in seconds to cover the distance
    """
    if distance_yards <= 0:
        return 0

    if distance_yards <= ACCELERATION_DISTANCE:
        # Parabolic model: t = ACCELERATION_TIME * sqrt(d / ACCELERATION_DISTANCE)
        return ACCELERATION_TIME * math.sqrt(distance_yards / ACCELERATION_DISTANCE)
    else:
        # Time to reach top speed + time at top speed for remaining distance
        remaining = distance_yards - ACCELERATION_DISTANCE
        return ACCELERATION_TIME + remaining / TOP_SPEED


def distance_run_in_time(time_seconds):
    """Calculate distance a player can run in given time.

    Inverse of time_to_run_distance.

    Args:
        time_seconds: Time in seconds

    Returns:
        Distance in yards
    """
    if time_seconds <= 0:
        return 0

    if time_seconds <= ACCELERATION_TIME:
        # During acceleration phase: d = ACCELERATION_DISTANCE * (t / ACCELERATION_TIME)²
        return ACCELERATION_DISTANCE * (time_seconds / ACCELERATION_TIME) ** 2
    else:
        # Full acceleration + constant speed
        extra_time = time_seconds - ACCELERATION_TIME
        return ACCELERATION_DISTANCE + TOP_SPEED * extra_time


def time_to_dribble_distance(distance_yards):
    """Calculate time for a player to dribble a given distance.

    Uses acceleration model at 50% of sprint speed:
    - Phase 1 (0 to DRIBBLE_ACCEL_DISTANCE): Parabolic acceleration
    - Phase 2 (beyond): Constant dribble top speed

    Args:
        distance_yards: Distance in yards

    Returns:
        Time in seconds to dribble the distance
    """
    if distance_yards <= 0:
        return 0

    if distance_yards <= DRIBBLE_ACCEL_DISTANCE:
        # Parabolic model: t = DRIBBLE_ACCEL_TIME * sqrt(d / DRIBBLE_ACCEL_DISTANCE)
        return DRIBBLE_ACCEL_TIME * math.sqrt(distance_yards / DRIBBLE_ACCEL_DISTANCE)
    else:
        # Time to reach top dribble speed + time at top speed for remaining distance
        remaining = distance_yards - DRIBBLE_ACCEL_DISTANCE
        return DRIBBLE_ACCEL_TIME + remaining / DRIBBLE_TOP_SPEED


def ball_travel_time(distance_yards, initial_speed):
    """Calculate time for ball to travel a distance with deceleration.

    Ball decelerates due to friction. Uses kinematic equations.

    Args:
        distance_yards: Distance to travel
        initial_speed: Initial ball speed in yards/s

    Returns:
        Time in seconds, or float('inf') if ball stops before reaching
    """
    if distance_yards <= 0:
        return 0
    if initial_speed <= 0:
        return float('inf')

    # Using: v² = u² - 2as (deceleration is negative acceleration)
    # Final velocity squared
    v_squared = initial_speed ** 2 - 2 * BALL_DECELERATION * distance_yards

    if v_squared <= 0:
        # Ball stops before reaching target
        return float('inf')

    # Using: v = u - at, solve for t
    final_speed = math.sqrt(v_squared)
    time = (initial_speed - final_speed) / BALL_DECELERATION

    return time


def ball_distance_at_time(initial_speed, time_seconds):
    """Calculate how far ball travels in given time with deceleration.

    Args:
        initial_speed: Initial ball speed in yards/s
        time_seconds: Time elapsed

    Returns:
        Distance traveled in yards
    """
    if time_seconds <= 0 or initial_speed <= 0:
        return 0

    # Time until ball stops
    time_to_stop = initial_speed / BALL_DECELERATION

    if time_seconds >= time_to_stop:
        # Ball has stopped - return max distance
        return (initial_speed ** 2) / (2 * BALL_DECELERATION)

    # Using: s = ut - ½at²
    distance = initial_speed * time_seconds - 0.5 * BALL_DECELERATION * time_seconds ** 2
    return max(0, distance)


def optimal_pass_speed(distance_yards):
    """Calculate optimal pass speed for a given distance.

    Shorter passes = softer, longer passes = harder.

    Args:
        distance_yards: Pass distance

    Returns:
        Optimal ball speed in yards/s
    """
    # Linear interpolation based on distance
    # 5 yards → min speed, 40+ yards → max speed
    if distance_yards <= 5:
        return PASS_SPEED_MIN
    elif distance_yards >= 40:
        return PASS_SPEED_MAX
    else:
        ratio = (distance_yards - 5) / 35
        return PASS_SPEED_MIN + ratio * (PASS_SPEED_MAX - PASS_SPEED_MIN)


def can_defender_intercept(defender_x, defender_y, ball_start_x, ball_start_y,
                           ball_end_x, ball_end_y, ball_speed):
    """Check if defender can intercept a pass.

    Considers:
    - Reaction time (0.3s delay)
    - Acceleration curve to reach interception point
    - Ball travel time with deceleration
    - Player width (1 yard interception range)

    Args:
        defender_x, defender_y: Defender position
        ball_start_x, ball_start_y: Ball start position
        ball_end_x, ball_end_y: Ball target position
        ball_speed: Initial ball speed

    Returns:
        dict with: can_intercept (bool), intercept_point, time_margin
    """
    # Find closest point on ball path to defender
    ball_dx = ball_end_x - ball_start_x
    ball_dy = ball_end_y - ball_start_y
    ball_dist = math.sqrt(ball_dx**2 + ball_dy**2)

    if ball_dist == 0:
        return {'can_intercept': False, 'time_margin': float('inf')}

    # Normalize ball direction
    ball_dir_x = ball_dx / ball_dist
    ball_dir_y = ball_dy / ball_dist

    # Vector from ball start to defender
    to_def_x = defender_x - ball_start_x
    to_def_y = defender_y - ball_start_y

    # Project defender onto ball path
    proj_dist = to_def_x * ball_dir_x + to_def_y * ball_dir_y
    proj_dist = max(0, min(ball_dist, proj_dist))  # Clamp to ball path

    # Closest point on ball path
    closest_x = ball_start_x + proj_dist * ball_dir_x
    closest_y = ball_start_y + proj_dist * ball_dir_y

    # Distance from defender to closest point
    intercept_dist = calculate_distance(defender_x, defender_y, closest_x, closest_y)

    # Defender needs to get within PLAYER_WIDTH of ball path
    if intercept_dist <= PLAYER_WIDTH:
        # Already in interception range
        run_dist = 0
    else:
        run_dist = intercept_dist - PLAYER_WIDTH

    # Time for defender to reach interception point (including reaction time)
    defender_time = REACTION_TIME + time_to_run_distance(run_dist)

    # Time for ball to reach the interception point
    ball_time = ball_travel_time(proj_dist, ball_speed)

    # Defender intercepts if they arrive before or at same time as ball
    # time_margin = ball_time - defender_time
    # Positive margin = defender arrives first (can intercept)
    # Negative margin = ball arrives first (cannot intercept)
    time_margin = ball_time - defender_time
    can_intercept = time_margin >= 0

    return {
        'can_intercept': can_intercept,
        'intercept_point': (closest_x, closest_y),
        'intercept_dist_on_path': proj_dist,
        'defender_run_dist': run_dist,
        'defender_time': defender_time,
        'ball_time': ball_time,
        'time_margin': time_margin  # Positive = defender arrives first
    }


def calculate_through_ball_options(ball_x, ball_y, attacker, defenders):
    """Calculate through ball options for an attacker.

    Through ball = pass to space in front of a running player.
    Considers:
    - Where attacker can run to (any direction, prioritizing toward goal)
    - Whether ball reaches space before defenders intercept
    - Whether attacker reaches space before/with the ball

    Args:
        ball_x, ball_y: Current ball position
        attacker: Attacker dict with x, y, id
        defenders: List of defender dicts

    Returns:
        List of through ball options with target positions and success probabilities
    """
    options = []
    att_x = attacker['x']
    att_y = attacker['y']

    # Generate potential run destinations
    # Prioritize: forward toward goal, diagonal runs, wide runs
    run_directions = []

    # Forward runs (toward goal at x=60)
    for angle in range(-45, 46, 15):  # -45° to +45° from direct forward
        rad = math.radians(angle)
        # Run distances: 5, 10, 15, 20 yards ahead
        for run_dist in [5, 10, 15, 20]:
            target_x = att_x + run_dist * math.cos(rad)
            target_y = att_y + run_dist * math.sin(rad)
            run_directions.append((target_x, target_y, run_dist))

    for target_x, target_y, run_dist in run_directions:
        # Check bounds
        if target_x > 58 or target_x < ball_x or abs(target_y) > 36:
            continue

        # Must be ahead of ball (forward pass)
        if target_x <= ball_x + 3:
            continue

        # Offside check - use attacker's STARTING position, not target
        # (player runs onto through ball from onside position)
        if is_offside(att_x, ball_x):
            continue

        # Calculate pass distance
        pass_dist = calculate_distance(ball_x, ball_y, target_x, target_y)
        if pass_dist < 8 or pass_dist > 50:  # Through balls are medium-long range
            continue

        # Calculate optimal ball speed for this distance
        ball_speed = optimal_pass_speed(pass_dist)

        # Time for ball to reach target space
        ball_time = ball_travel_time(pass_dist, ball_speed)
        if ball_time == float('inf'):
            continue

        # Time for attacker to reach target space
        attacker_run_dist = calculate_distance(att_x, att_y, target_x, target_y)
        attacker_time = time_to_run_distance(attacker_run_dist)

        # Attacker must arrive at or before ball (can wait for ball)
        # Give 0.5s margin - attacker can arrive up to 0.5s after ball
        if attacker_time > ball_time + 0.5:
            continue

        # Check if any defender can intercept
        can_complete = True
        closest_defender_margin = float('inf')

        for defender in defenders:
            if defender['id'] == 1:  # Skip GK for now
                continue

            intercept = can_defender_intercept(
                defender['x'], defender['y'],
                ball_x, ball_y,
                target_x, target_y,
                ball_speed
            )

            if intercept['can_intercept']:
                can_complete = False
                break
            else:
                closest_defender_margin = min(closest_defender_margin, intercept['time_margin'])

        if not can_complete:
            continue

        # Also check if defender can reach the target space before attacker
        for defender in defenders:
            if defender['id'] == 1:
                continue

            def_dist = calculate_distance(defender['x'], defender['y'], target_x, target_y)
            def_time = REACTION_TIME + time_to_run_distance(def_dist)

            # If defender reaches space significantly before attacker, not a good option
            if def_time < attacker_time - 0.3:
                can_complete = False
                break

        if not can_complete:
            continue

        # Through ball succeeds if it passed all interception checks
        # (100% success - perfect execution)
        options.append({
            'target_x': target_x,
            'target_y': target_y,
            'pass_distance': pass_dist,
            'ball_speed': ball_speed,
            'ball_time': ball_time,
            'attacker_run_dist': attacker_run_dist,
            'attacker_time': attacker_time,
            'defender_margin': closest_defender_margin,
        })

    # Sort by position value (closer to goal = better)
    options.sort(key=lambda o: -o['target_x'])

    return options[:3]  # Return top 3 through ball options per attacker


# ============================================================
# CHAIN-BASED EXPECTED VALUE MODEL
# ============================================================
# EV = probability of eventually scoring from a position
# Considers: shoot now OR pass to teammate who can shoot/pass
# Each pass discounts by completion probability
# ============================================================

def calculate_position_value(x, y, depth=0, max_depth=2, excluded_positions=None, time_elapsed=0):
    """Calculate the Expected Value of having the ball at position (x, y).

    Returns both the value AND the optimal chain of actions.
    All actions are 100% success if not intercepted (perfect execution).

    Args:
        x, y: Position on pitch (yards)
        depth: Current recursion depth
        max_depth: How many passes ahead to look
        excluded_positions: Positions already evaluated (prevent loops)
        time_elapsed: Time since start of chain (for player movement)

    Returns:
        dict: {
            'value': float (0-1) representing goal probability,
            'chain': list of action dicts describing the optimal sequence
        }
    """
    if excluded_positions is None:
        excluded_positions = set()

    # Prevent infinite loops
    pos_key = (round(x, 0), round(y, 0))
    if pos_key in excluded_positions:
        xg = calculate_xg(x, y)
        return {
            'value': xg,
            'chain': [{'action': 'shoot', 'x': x, 'y': y, 'xg': xg}]
        }
    excluded_positions = excluded_positions | {pos_key}

    attackers = current_state['attackers']
    defenders = current_state['defenders']

    # OPTION 1: Shoot from current position
    shoot_xg = calculate_xg(x, y)
    # Shot is blocked only if defender is directly in the way
    shot_blocked = False
    for d in defenders:
        if d.get('id') == 1:
            continue
        block_dist = point_to_line_distance(d['x'], d['y'], x, y, 60, 0)
        if block_dist < PLAYER_WIDTH and d['x'] > x:
            shot_blocked = True
            break

    if shot_blocked:
        shoot_value = 0.0
    else:
        shoot_value = shoot_xg  # Perfect shot = xG

    best_result = {
        'value': shoot_value,
        'chain': [{'action': 'shoot', 'x': x, 'y': y, 'xg': shoot_xg, 'blocked': shot_blocked}]
    }

    # At max depth, only consider shooting
    if depth >= max_depth:
        return best_result

    # OPTION 2: Pass to each teammate (at current position OR where they can run to)
    for attacker in attackers:
        if attacker['id'] == 1:  # Skip GK
            continue

        # Consider both current position and potential run positions
        target_positions = []

        # Current position
        target_positions.append({
            'x': attacker['x'],
            'y': attacker['y'],
            'is_run': False,
            'run_time': 0
        })

        # Potential run positions (forward runs into space)
        for angle in [-30, 0, 30]:  # Diagonal left, straight, diagonal right
            for run_dist in [8, 15]:  # Short and medium runs
                rad = math.radians(angle)
                run_x = attacker['x'] + run_dist * math.cos(rad)
                run_y = attacker['y'] + run_dist * math.sin(rad)

                # Must be in bounds and forward
                if run_x > 58 or run_x < attacker['x'] or abs(run_y) > 36:
                    continue

                run_time = time_to_run_distance(run_dist)

                target_positions.append({
                    'x': run_x,
                    'y': run_y,
                    'is_run': True,
                    'run_time': run_time,
                    'run_dist': run_dist
                })

        for target in target_positions:
            target_x = target['x']
            target_y = target['y']

            # Don't pass to current ball position
            dist = calculate_distance(x, y, target_x, target_y)
            if dist < 5:
                continue

            # Calculate pass success (binary: 1.0 or 0.0)
            pass_success = calculate_pass_success(x, y, target_x, target_y, defenders)

            if pass_success == 0.0:  # Intercepted
                continue

            # If it's a run, check timing
            if target['is_run']:
                ball_speed = optimal_pass_speed(dist)
                ball_time = ball_travel_time(dist, ball_speed)
                # Player must arrive before or with the ball
                if target['run_time'] > ball_time + 0.3:
                    continue

            # Recursively calculate value at target position
            new_time = time_elapsed + (target.get('run_time', 0) or ball_travel_time(dist, optimal_pass_speed(dist)))
            target_result = calculate_position_value(
                target_x, target_y,
                depth=depth + 1,
                max_depth=max_depth,
                excluded_positions=excluded_positions,
                time_elapsed=new_time
            )

            # Pass value = target_position_value (100% success)
            pass_value = target_result['value']

            if pass_value > best_result['value']:
                action_desc = {
                    'action': 'through_ball' if target['is_run'] else 'pass',
                    'from_x': x,
                    'from_y': y,
                    'to_x': target_x,
                    'to_y': target_y,
                    'player_id': attacker['id'],
                    'is_run': target['is_run'],
                    'run_dist': target.get('run_dist', 0)
                }
                best_result = {
                    'value': pass_value,
                    'chain': [action_desc] + target_result['chain']
                }

    # OPTION 3: Dribble forward (short distances only)
    dribble_options = [(5, 0), (4, 4), (4, -4)]
    for dx, dy in dribble_options:
        new_x = x + dx
        new_y = y + dy

        if abs(new_x) > 58 or abs(new_y) > 36:
            continue

        dribble_success = calculate_dribble_success(x, y, new_x, new_y, defenders)

        if dribble_success == 0.0:  # Blocked
            continue

        dribble_dist = calculate_distance(x, y, new_x, new_y)
        dribble_time = time_to_dribble_distance(dribble_dist)

        target_result = calculate_position_value(
            new_x, new_y,
            depth=depth + 1,
            max_depth=max_depth,
            excluded_positions=excluded_positions,
            time_elapsed=time_elapsed + dribble_time
        )

        dribble_value = target_result['value']

        if dribble_value > best_result['value']:
            action_desc = {
                'action': 'dribble',
                'from_x': x,
                'from_y': y,
                'to_x': new_x,
                'to_y': new_y
            }
            best_result = {
                'value': dribble_value,
                'chain': [action_desc] + target_result['chain']
            }

    return best_result


def calculate_shot_block_probability(x, y, defenders):
    """Calculate probability that a shot from (x,y) gets blocked."""
    goal_x = 60
    goal_y = 0
    block_prob = 0

    for d in defenders:
        if d['id'] == 1:  # GK handled separately in xG
            continue

        # Distance from defender to shot line
        block_dist = point_to_line_distance(d['x'], d['y'], x, y, goal_x, goal_y)

        # Only count defenders between shooter and goal
        if d['x'] > x:
            if block_dist < 2:
                block_prob = max(block_prob, 0.4)
            elif block_dist < 4:
                block_prob = max(block_prob, 0.25)
            elif block_dist < 6:
                block_prob = max(block_prob, 0.1)

    return min(0.7, block_prob)  # Cap at 70%


def calculate_pass_success(from_x, from_y, to_x, to_y, defenders, check_offside=True, receiver_x=None):
    """Calculate if a pass can be completed (binary: 100% or 0%).

    Perfect execution assumed - pass only fails if physically intercepted or offside.
    Uses physics model for interception:
    - Ball travel time (with deceleration)
    - Defender reaction time (0.3s)
    - Defender acceleration curve
    - Player width (1 yard interception range)

    Args:
        from_x, from_y: Pass origin
        to_x, to_y: Pass target
        defenders: List of defender dicts
        check_offside: Whether to check offside rule
        receiver_x: Receiver's x position for offside check (defaults to to_x)

    Returns:
        float: 1.0 if pass completes, 0.0 if intercepted or offside
    """
    pass_dist = calculate_distance(from_x, from_y, to_x, to_y)

    if pass_dist < 1:
        return 1.0  # Very short pass always works

    # Check offside - use receiver's actual position, not target
    if check_offside:
        check_x = receiver_x if receiver_x is not None else to_x
        if is_offside(check_x, from_x):
            return 0.0  # Offside

    # Calculate optimal ball speed for this pass
    ball_speed = optimal_pass_speed(pass_dist)

    # Check each defender for interception
    for d in defenders:
        if d.get('id') == 1:  # Skip GK for regular passes
            continue

        intercept = can_defender_intercept(
            d['x'], d['y'],
            from_x, from_y,
            to_x, to_y,
            ball_speed
        )

        if intercept['can_intercept']:
            return 0.0  # Intercepted - pass fails

    # No interception - pass succeeds (perfect execution)
    return 1.0


def calculate_dribble_success(from_x, from_y, to_x, to_y, defenders):
    """Calculate if a dribble can succeed (binary: 100% or 0%).

    Perfect execution assumed - dribble only fails if defender can intercept.
    Uses same physics model as pass interception - checks ALL defenders.
    Dribbling uses 50% speed with proportional acceleration curve.

    Args:
        from_x, from_y: Dribble start
        to_x, to_y: Dribble destination
        defenders: List of defender dicts

    Returns:
        float: 1.0 if dribble succeeds, 0.0 if blocked
    """
    dribble_dist = calculate_distance(from_x, from_y, to_x, to_y)
    if dribble_dist < 1:
        return 1.0

    # Normalize dribble direction
    dx = (to_x - from_x) / dribble_dist
    dy = (to_y - from_y) / dribble_dist

    for d in defenders:
        if d.get('id') == 1:  # Skip GK
            continue

        # Vector from dribble start to defender
        to_def_x = d['x'] - from_x
        to_def_y = d['y'] - from_y

        # Project defender onto dribble path
        proj_dist = to_def_x * dx + to_def_y * dy

        # Skip defenders behind the dribbler or beyond destination
        if proj_dist < 0 or proj_dist > dribble_dist:
            continue

        # Closest point on dribble path to defender
        intercept_x = from_x + proj_dist * dx
        intercept_y = from_y + proj_dist * dy

        # Distance from defender to intercept point (perpendicular distance to path)
        def_to_intercept = calculate_distance(d['x'], d['y'], intercept_x, intercept_y)

        # If defender is already standing in the path, blocked immediately
        if def_to_intercept < PLAYER_WIDTH * 1.5:  # 1.5 yards - can't dribble through
            return 0.0

        # Time for dribbler to reach intercept point (with acceleration curve)
        dribbler_time = time_to_dribble_distance(proj_dist)

        # Time for defender to reach intercept point (needs to get within PLAYER_WIDTH)
        run_dist = max(0, def_to_intercept - PLAYER_WIDTH)
        defender_time = REACTION_TIME + time_to_run_distance(run_dist)

        # If defender arrives first or at same time, dribble fails
        if defender_time <= dribbler_time:
            return 0.0

    # No defender can block - dribble succeeds
    return 1.0


def create_default_scenario():
    """Create default game scenario with football-accurate squad numbers.

    Team 0 = attacking (blue), Team 1 = defending (red)
    Player positions in yards, centered coordinate system.

    Defenders: 4-4-2 low block (narrower, RB/LB at y=±18)
    - Back 4 at x=42 (edge of penalty box)
    - Midfield 4 at x=31 (between back and front)
    - Front 2 at x=20

    Attackers: Positioned to attack the block
    - 5 players at x=42 in gaps between back 4
    - 2 players at x=26 between defensive lines
    - 3 players at x=15 (5 yards behind front line)
    """
    # Team 1 (defending) - red - 4-4-2 low block
    # Football numbers: 1=GK, 2=RB, 3=LB, 4=CM, 5=CB, 6=CB, 7=RM, 8=CM, 9=ST, 10=ST, 11=LM
    defenders = [
        {"id": 1, "x": 55, "y": 0, "team": 1},       # GK
        # Back 4 at x=42 (narrower: ±18 for fullbacks, ±6 for CBs)
        {"id": 2, "x": 42, "y": 18, "team": 1},      # RB
        {"id": 5, "x": 42, "y": 6, "team": 1},       # RCB
        {"id": 6, "x": 42, "y": -6, "team": 1},      # LCB
        {"id": 3, "x": 42, "y": -18, "team": 1},     # LB
        # Midfield 4 at x=31
        {"id": 7, "x": 31, "y": 14, "team": 1},      # RM
        {"id": 4, "x": 31, "y": 5, "team": 1},       # RCM
        {"id": 8, "x": 31, "y": -5, "team": 1},      # LCM
        {"id": 11, "x": 31, "y": -14, "team": 1},    # LM
        # Front 2 at x=20
        {"id": 9, "x": 20, "y": 6, "team": 1},       # RST
        {"id": 10, "x": 20, "y": -6, "team": 1},     # LST
    ]

    # Team 0 (attacking left to right) - blue
    # Football numbers: 1=GK, 2=RB, 3=LB, 4=CM, 5=CB, 6=CM, 7=RW, 8=CM, 9=ST, 10=CAM, 11=LW
    # Back 3 pushed up (2, 3, 5), between lines 4 (4, 6, 8, 10), front 3 (7, 9, 11)
    attackers = [
        {"id": 1, "x": -55, "y": 0, "team": 0},      # GK
        # 3 players at x=42: ST central, wingers wide
        {"id": 11, "x": 42, "y": -28, "team": 0},    # LW (wide left)
        {"id": 9, "x": 42, "y": 0, "team": 0},       # ST (central)
        {"id": 7, "x": 42, "y": 28, "team": 0},      # RW (wide right)
        # Between back line (x=42) and midfield (x=31): #10 and #8 at x=36
        {"id": 10, "x": 36, "y": 12, "team": 0},     # CAM/RIF (right half-space)
        {"id": 8, "x": 36, "y": -12, "team": 0},     # CAM/LIF (left half-space)
        # Between midfield (x=31) and front 2 (x=20): #6 and #4 at x=26
        {"id": 6, "x": 26, "y": 8, "team": 0},       # RCM
        {"id": 4, "x": 26, "y": -8, "team": 0},      # LCM
        # 3 players at x=15 (5 yards behind front line at x=20)
        {"id": 2, "x": 15, "y": 12, "team": 0},      # RB pushed up
        {"id": 5, "x": 15, "y": 0, "team": 0},       # CB/DM (ball carrier)
        {"id": 3, "x": 15, "y": -12, "team": 0},     # LB pushed up
    ]

    return attackers, defenders


# Store current state
current_state = {
    'attackers': [],
    'defenders': [],
    'ball': {"x": 15, "y": 0},  # With #5 (CB/DM)
}

# Initialize with default
current_state['attackers'], current_state['defenders'] = create_default_scenario()


def get_all_players():
    """Get all players as a combined list."""
    return current_state['attackers'] + current_state['defenders']


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def get_offside_line():
    """Get the x-position of the offside line.

    Offside line is the second-to-last defender (last outfield defender,
    since GK is typically furthest back).

    Returns:
        float: x-position of the offside line
    """
    defenders = current_state['defenders']
    # Sort defenders by x position (furthest forward first for defending team)
    # Since defenders defend goal at x=60, higher x = deeper
    sorted_defs = sorted(defenders, key=lambda d: d['x'], reverse=True)

    # Second-to-last = index 1 (index 0 is GK typically)
    if len(sorted_defs) >= 2:
        return sorted_defs[1]['x']
    return 60  # Default to goal line if not enough defenders


def is_offside(player_x, ball_x):
    """Check if a player position is offside.

    A player is offside if:
    1. They are in the opponent's half (x > 0)
    2. They are beyond the offside line (second-to-last defender)
    3. They are beyond the ball

    Args:
        player_x: Player's x position
        ball_x: Ball's x position when pass is made

    Returns:
        bool: True if offside
    """
    # Must be in attacking half
    if player_x <= 0:
        return False

    # Must be ahead of the ball
    if player_x <= ball_x:
        return False

    # Check against offside line
    offside_line = get_offside_line()
    return player_x > offside_line


def calculate_ball_pressure():
    """Calculate pressure on the ball carrier.

    Returns a value 0-1 where 1 = extremely high pressure.
    """
    ball = current_state['ball']
    defenders = current_state['defenders']

    # Find distances of all defenders to the ball
    defender_distances = []
    for d in defenders:
        dist = calculate_distance(d['x'], d['y'], ball['x'], ball['y'])
        defender_distances.append(dist)

    defender_distances.sort()

    # Pressure based on closest defenders
    if len(defender_distances) >= 2:
        closest = defender_distances[0]
        second_closest = defender_distances[1]

        # High pressure if defender within 5 yards
        if closest < 5:
            pressure = 0.9
        elif closest < 8:
            pressure = 0.7
        elif closest < 12:
            pressure = 0.4
        else:
            pressure = 0.2

        # Additional pressure if second defender also close (double team)
        if second_closest < 10:
            pressure = min(1.0, pressure + 0.2)

        return pressure

    return 0.2  # Default low pressure


def detect_overloads():
    """Detect numerical overloads (2v1, 3v2 situations).

    Returns dict mapping zones to overload info.
    """
    ball = current_state['ball']
    attackers = current_state['attackers']
    defenders = current_state['defenders']

    overloads = []

    # Define zones to check (areas ahead of the ball)
    zones = [
        {'name': 'left_channel', 'x_min': ball['x'], 'x_max': 60, 'y_min': -37, 'y_max': -10},
        {'name': 'central', 'x_min': ball['x'], 'x_max': 60, 'y_min': -10, 'y_max': 10},
        {'name': 'right_channel', 'x_min': ball['x'], 'x_max': 60, 'y_min': 10, 'y_max': 37},
        {'name': 'left_half_space', 'x_min': ball['x'] + 10, 'x_max': 50, 'y_min': -22, 'y_max': -5},
        {'name': 'right_half_space', 'x_min': ball['x'] + 10, 'x_max': 50, 'y_min': 5, 'y_max': 22},
    ]

    for zone in zones:
        # Count attackers in zone (exclude GK)
        atk_in_zone = [a for a in attackers
                       if a['id'] != 1
                       and zone['x_min'] <= a['x'] <= zone['x_max']
                       and zone['y_min'] <= a['y'] <= zone['y_max']]

        # Count defenders in zone (exclude GK)
        def_in_zone = [d for d in defenders
                       if d['id'] != 1
                       and zone['x_min'] <= d['x'] <= zone['x_max']
                       and zone['y_min'] <= d['y'] <= zone['y_max']]

        num_atk = len(atk_in_zone)
        num_def = len(def_in_zone)

        # Check for overload (more attackers than defenders)
        if num_atk > num_def and num_atk >= 2:
            overloads.append({
                'zone': zone['name'],
                'attackers': num_atk,
                'defenders': num_def,
                'advantage': num_atk - num_def,
                'players': [a['id'] for a in atk_in_zone],
                'center_x': (zone['x_min'] + zone['x_max']) / 2,
                'center_y': (zone['y_min'] + zone['y_max']) / 2,
            })

    return overloads


def check_receiver_can_play_forward(receiver_x, receiver_y):
    """Check if receiver has space to play forward after receiving.

    Returns a score 0-1 where 1 = lots of space ahead.
    """
    defenders = current_state['defenders']

    # Check space in a cone ahead of the receiver
    space_ahead = True
    min_forward_dist = float('inf')

    for d in defenders:
        # Only consider defenders ahead of the receiver
        if d['x'] > receiver_x:
            # Check if defender is in the forward channel
            dy = abs(d['y'] - receiver_y)
            dx = d['x'] - receiver_x

            if dy < 15 and dx < 20:  # In the forward zone
                dist = calculate_distance(d['x'], d['y'], receiver_x, receiver_y)
                min_forward_dist = min(min_forward_dist, dist)

    if min_forward_dist == float('inf'):
        return 1.0  # No defenders ahead
    elif min_forward_dist > 15:
        return 0.9
    elif min_forward_dist > 10:
        return 0.7
    elif min_forward_dist > 6:
        return 0.4
    else:
        return 0.2


def is_switch_of_play(ball_y, target_y):
    """Check if pass is a switch of play (crossing to opposite side).

    Returns True if the pass switches the point of attack.
    """
    # Switch if crossing from one side to the other (>25 yards lateral)
    return abs(target_y - ball_y) > 25


def is_run_in_behind(ball_x, target_x, receiver_pressure):
    """Check if pass exploits space behind the defensive line.

    Returns True if this targets the space in behind.
    """
    # Target is in behind if:
    # 1. Well ahead of the ball (>20 yards forward)
    # 2. In the attacking third (x > 30)
    # 3. Receiver has low pressure (space to run into)
    return (target_x - ball_x > 20 and
            target_x > 30 and
            receiver_pressure < 0.5)


def get_weak_side(ball_y):
    """Determine the weak side (opposite to ball).

    Returns 'left' or 'right'.
    """
    if ball_y > 0:
        return 'left'  # Ball on right, weak side is left
    else:
        return 'right'  # Ball on left, weak side is right


def calculate_xg(x, y):
    """Calculate expected goals at a position using real xG zone data.

    Based on statistical xG model zone map.
    Goal is at x=60 (right side), pitch center at y=0.

    Grid: 9 columns (width) × 18 rows (length of half)
    Each zone is approximately 8.3 yards wide × 3.3 yards deep
    """
    # xG zone data from statistical model
    # Rows from goal line (row 0) to halfway line (row 17)
    # Columns from left touchline (col 0) to right touchline (col 8)
    # Center of pitch is columns 3,4,5
    XG_ZONES = [
        # Row 0: Inside 6-yard box (x = 57-60)
        [0.000, 0.016, 0.045, 0.147, 0.438, 0.169, 0.049, 0.026, 0.011],
        # Row 1: 6-yard box edge (x = 54-57)
        [0.010, 0.021, 0.048, 0.111, 0.173, 0.112, 0.050, 0.022, 0.020],
        # Row 2: Inside penalty box (x = 50-54)
        [0.006, 0.022, 0.040, 0.081, 0.187, 0.083, 0.041, 0.016, 0.016],
        # Row 3: Penalty box (x = 47-50)
        [0.000, 0.017, 0.027, 0.039, 0.054, 0.039, 0.028, 0.014, 0.011],
        # Row 4: Edge of penalty box (x = 43-47)
        [0.009, 0.014, 0.021, 0.028, 0.029, 0.027, 0.022, 0.014, 0.011],
        # Row 5: Just outside box (x = 40-43)
        [0.010, 0.015, 0.016, 0.020, 0.020, 0.020, 0.016, 0.010, 0.000],
        # Row 6: (x = 36-40)
        [0.008, 0.014, 0.014, 0.014, 0.013, 0.017, 0.013, 0.011, 0.004],
        # Row 7: (x = 33-36)
        [0.007, 0.008, 0.009, 0.004, 0.008, 0.009, 0.002, 0.009, 0.006],
        # Row 8: (x = 29-33)
        [0.000, 0.007, 0.007, 0.007, 0.007, 0.005, 0.008, 0.000, 0.006],
        # Row 9: (x = 26-29)
        [0.000, 0.000, 0.012, 0.000, 0.005, 0.006, 0.000, 0.003, 0.000],
        # Row 10: (x = 22-26)
        [0.000, 0.000, 0.005, 0.005, 0.006, 0.005, 0.000, 0.000, 0.000],
        # Row 11: (x = 19-22)
        [0.000, 0.006, 0.000, 0.004, 0.005, 0.004, 0.003, 0.000, 0.000],
        # Row 12: (x = 15-19)
        [0.000, 0.003, 0.003, 0.000, 0.000, 0.000, 0.003, 0.000, 0.000],
        # Row 13: (x = 11-15)
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        # Row 14: (x = 8-11)
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        # Row 15: (x = 4-8)
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        # Row 16: (x = 0-4)
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        # Row 17: Halfway line and beyond
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    ]

    # Pitch dimensions
    HALF_LENGTH = 60  # yards from center to goal
    HALF_WIDTH = 37.5  # yards from center to touchline

    # If in defensive half, xG is essentially 0
    if x <= 0:
        return 0.001

    # Map position to grid indices
    # x: 0 to 60 maps to rows 17 down to 0
    # y: -37.5 to +37.5 maps to columns 0 to 8

    # Row index (0 = closest to goal, 17 = halfway line)
    row_float = (HALF_LENGTH - x) / HALF_LENGTH * 17
    row_float = max(0, min(17, row_float))

    # Column index (0 = left, 8 = right, 4 = center)
    col_float = (y + HALF_WIDTH) / (2 * HALF_WIDTH) * 8
    col_float = max(0, min(8, col_float))

    # Get integer indices for bilinear interpolation
    row_low = int(row_float)
    row_high = min(row_low + 1, 17)
    col_low = int(col_float)
    col_high = min(col_low + 1, 8)

    # Interpolation weights
    row_frac = row_float - row_low
    col_frac = col_float - col_low

    # Bilinear interpolation
    xg_ll = XG_ZONES[row_low][col_low]
    xg_lh = XG_ZONES[row_low][col_high]
    xg_hl = XG_ZONES[row_high][col_low]
    xg_hh = XG_ZONES[row_high][col_high]

    xg_low = xg_ll * (1 - col_frac) + xg_lh * col_frac
    xg_high = xg_hl * (1 - col_frac) + xg_hh * col_frac
    xg = xg_low * (1 - row_frac) + xg_high * row_frac

    return max(0.001, xg)


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
    """Analyze all passing options using chain-based Expected Value.

    EV = probability of eventually scoring from this position
    Chain calculation: pass_success × target_position_value

    Target position value considers what the receiver can do:
    - Shoot immediately (xG)
    - Pass to another teammate (chains further)
    - Dribble to better position

    This creates realistic football decision-making where a midfielder
    in a low xG zone can have high EV by passing to a striker.
    """
    options = []
    ball = current_state['ball']
    attackers = current_state['attackers']
    defenders = current_state['defenders']

    # Get the best chain from current position (includes chain visualization)
    best_chain_result = calculate_position_value(ball['x'], ball['y'], depth=0, max_depth=3)
    current_xg = calculate_xg(ball['x'], ball['y'])

    # Calculate pressure on ball carrier
    ball_pressure = calculate_ball_pressure()

    # Detect overloads and weak side for tactical tags
    overloads = detect_overloads()
    overload_zones = {o['zone']: o for o in overloads}
    weak_side = get_weak_side(ball['y'])

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

        # ========================================
        # CHAIN-BASED EV CALCULATION (100% success if not intercepted)
        # ========================================

        # 1. Calculate pass success (binary: 1.0 or 0.0)
        success_prob = calculate_pass_success(ball['x'], ball['y'], target_x, target_y, defenders)

        # Skip if pass is intercepted
        if success_prob == 0.0:
            continue

        # 2. Calculate target position value with chain
        target_result = calculate_position_value(
            target_x, target_y,
            depth=1,
            max_depth=3
        )

        # 3. EV = target_value (100% success, no turnover cost)
        ev = target_result['value']
        target_chain = target_result['chain']

        # ========================================
        # Additional metrics for display
        # ========================================
        target_xg = calculate_xg(target_x, target_y)
        xg_gain = target_xg - current_xg

        # Pressure on receiver
        min_defender_dist = float('inf')
        for defender in defenders:
            d = calculate_distance(defender['x'], defender['y'], target_x, target_y)
            min_defender_dist = min(min_defender_dist, d)
        receiver_pressure = max(0, 1.0 - min_defender_dist / 15)

        # Forward space check
        forward_space = check_receiver_can_play_forward(target_x, target_y)

        # Tactical tags
        is_switch = is_switch_of_play(ball['y'], target_y)
        is_behind = is_run_in_behind(ball['x'], target_x, receiver_pressure)

        # Check if in overload zone
        in_overload = False
        for zone_name, overload in overload_zones.items():
            if attacker['id'] in overload['players']:
                in_overload = True
                break

        # ========================================
        # Recommendation based on chain EV (no hold comparison)
        # ========================================

        # Recommendation based purely on EV (100% success, no hold comparison)
        if ev > 0.20:
            rec = "HIGH_VALUE"
        elif ev > 0.10:
            rec = "HIGH_VALUE"
        elif ev > 0.05:
            rec = "MODERATE"
        elif ev > 0.02:
            rec = "LOW_VALUE"
        else:
            rec = "AVOID"

        # Build action type
        action_type = 'through_ball' if is_behind else 'pass'

        # Build full chain description for this option
        full_chain = [{'action': 'pass', 'from_x': ball['x'], 'from_y': ball['y'],
                       'to_x': target_x, 'to_y': target_y, 'player_id': attacker['id']}] + target_chain

        options.append({
            'action': action_type,
            'target_x': target_x,
            'target_y': target_y,
            'target_player': attacker['id'],
            'xg_gain': xg_gain,
            'xg_target': target_xg,
            'target_position_value': target_result['value'],
            'ev': ev,
            'chain': full_chain,  # Full sequence of plays
            'chain_length': len(full_chain),
            'recommendation': rec,
            'receiver_pressure': receiver_pressure,
            'receiver_facing_goal': target_x > ball['x'],
            'receiver_forward_space': forward_space,
            'is_switch': is_switch,
            'is_in_behind': is_behind,
            'in_overload': in_overload,
        })

        # ========================================
        # THROUGH BALLS - Pass to space in front of player
        # Through balls only included if not intercepted (100% success)
        # ========================================
        through_ball_opts = calculate_through_ball_options(
            ball['x'], ball['y'], attacker, defenders
        )

        for tb in through_ball_opts:
            tb_target_x = tb['target_x']
            tb_target_y = tb['target_y']

            # Calculate position value at through ball target
            tb_result = calculate_position_value(
                tb_target_x, tb_target_y,
                depth=1,
                max_depth=2
            )

            # Through ball is 100% if included (not intercepted)
            tb_ev = tb_result['value']
            tb_xg = calculate_xg(tb_target_x, tb_target_y)
            tb_chain = [{'action': 'through_ball', 'from_x': ball['x'], 'from_y': ball['y'],
                         'to_x': tb_target_x, 'to_y': tb_target_y, 'player_id': attacker['id'],
                         'run_dist': tb['attacker_run_dist']}] + tb_result['chain']

            # Recommendation based on EV
            if tb_ev > 0.15:
                tb_rec = "HIGH_VALUE"
            elif tb_ev > 0.10:
                tb_rec = "HIGH_VALUE"
            elif tb_ev > 0.06:
                tb_rec = "MODERATE"
            else:
                tb_rec = "LOW_VALUE"

            options.append({
                'action': 'through_ball',
                'target_x': tb_target_x,
                'target_y': tb_target_y,
                'target_player': attacker['id'],
                'xg_gain': tb_xg - current_xg,
                'xg_target': tb_xg,
                'target_position_value': tb_result['value'],
                'ev': tb_ev,
                'chain': tb_chain,
                'chain_length': len(tb_chain),
                'recommendation': tb_rec,
                'receiver_pressure': 0,  # Running into space
                'receiver_facing_goal': True,
                'receiver_forward_space': 1.0,
                'is_switch': is_switch_of_play(ball['y'], tb_target_y),
                'is_in_behind': True,
                'in_overload': in_overload,
                'is_through_ball': True,
                'run_distance': round(tb['attacker_run_dist'], 1),
            })

    # ========================================
    # Analyze shooting option (binary: blocked or not blocked)
    # ========================================
    goal_x = 60
    goal_y = 0
    shoot_dist = calculate_distance(ball['x'], ball['y'], goal_x, goal_y)

    if shoot_dist < 40:  # Only consider shots from reasonable distance
        shot_xg = calculate_xg(ball['x'], ball['y'])

        # Check if shot is blocked (binary: blocked or clear)
        shot_blocked = False
        for d in defenders:
            if d.get('id') == 1:  # GK handled in xG
                continue
            block_dist = point_to_line_distance(d['x'], d['y'], ball['x'], ball['y'], goal_x, goal_y)
            if block_dist < PLAYER_WIDTH and d['x'] > ball['x']:
                shot_blocked = True
                break

        # Shot EV = xG if not blocked, 0 if blocked
        shot_ev = 0.0 if shot_blocked else shot_xg

        # Recommendation based purely on xG (if not blocked)
        if shot_blocked:
            rec = "BLOCKED"
        elif shot_xg > 0.15:
            rec = "HIGH_VALUE"
        elif shot_xg > 0.10:
            rec = "HIGH_VALUE"
        elif shot_xg > 0.06:
            rec = "MODERATE"
        else:
            rec = "LOW_VALUE"

        options.append({
            'action': 'shoot',
            'target_x': goal_x,
            'target_y': goal_y,
            'xg_target': shot_xg,
            'xg_gain': shot_xg - current_xg,
            'target_position_value': shot_xg,  # Terminal value
            'ev': shot_ev,
            'chain': [{'action': 'shoot', 'x': ball['x'], 'y': ball['y'], 'xg': shot_xg, 'blocked': shot_blocked}],
            'chain_length': 1,
            'recommendation': rec,
            'blocked': shot_blocked,
        })

    # ========================================
    # Analyze dribble options (binary: blocked or not blocked)
    # Dribbles are short movements - longer runs require space
    # ========================================
    dribble_directions = [
        (5, 0),    # Short forward
        (4, 4),    # Short diagonal right
        (4, -4),   # Short diagonal left
        (3, 6),    # Wide right
        (3, -6),   # Wide left
    ]

    for dx, dy in dribble_directions:
        target_x = ball['x'] + dx
        target_y = ball['y'] + dy

        # Check bounds
        if abs(target_x) > 58 or abs(target_y) > 36:
            continue

        # Calculate dribble success (binary: 1.0 or 0.0)
        dribble_success = calculate_dribble_success(ball['x'], ball['y'], target_x, target_y, defenders)

        if dribble_success == 0.0:
            continue  # Blocked by defender

        # Target position value (what can we do after dribbling there?)
        target_result = calculate_position_value(
            target_x, target_y,
            depth=1,
            max_depth=2
        )

        # Dribble EV = target value (100% success)
        ev = target_result['value']
        dribble_chain = [{'action': 'dribble', 'from_x': ball['x'], 'from_y': ball['y'],
                         'to_x': target_x, 'to_y': target_y}] + target_result['chain']

        target_xg = calculate_xg(target_x, target_y)
        xg_gain = target_xg - current_xg

        # Recommendation based purely on EV
        if ev > 0.15:
            rec = "HIGH_VALUE"
        elif ev > 0.10:
            rec = "HIGH_VALUE"
        elif ev > 0.06:
            rec = "MODERATE"
        else:
            rec = "LOW_VALUE"

        options.append({
            'action': 'dribble',
            'target_x': target_x,
            'target_y': target_y,
            'xg_gain': xg_gain,
            'xg_target': target_xg,
            'target_position_value': target_result['value'],
            'ev': ev,
            'chain': dribble_chain,
            'chain_length': len(dribble_chain),
            'recommendation': rec,
        })

    # Sort by expected value (chain-based)
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
    """Analyze current state and return tactical options.

    Enhanced with football realism metrics:
    - Ball pressure assessment
    - Overload detection
    - Weak side identification
    - Switch of play opportunities
    """
    try:
        data = request.json or {}

        # Update ball position if provided
        if 'ball_x' in data and 'ball_y' in data:
            current_state['ball']['x'] = data['ball_x']
            current_state['ball']['y'] = data['ball_y']

        ball = current_state['ball']

        # Find gaps
        gaps = find_gaps()

        # Calculate current position value (chain-based EV) - returns dict with 'value' and 'chain'
        current_result = calculate_position_value(ball['x'], ball['y'], depth=0, max_depth=3)
        current_ev = current_result['value']
        best_chain = current_result['chain']
        current_xg = calculate_xg(ball['x'], ball['y'])

        # Analyze options using chain-based EV model
        options = analyze_passing_options()

        # Calculate situational metrics
        ball_pressure = calculate_ball_pressure()
        overloads = detect_overloads()
        weak_side = get_weak_side(ball['y'])

        # Count by recommendation
        high_value = sum(1 for o in options if o['recommendation'] == 'HIGH_VALUE')
        safe = sum(1 for o in options if o['recommendation'] == 'SAFE')

        # Count special options
        switches = sum(1 for o in options if o.get('is_switch', False))
        through_balls = sum(1 for o in options if o.get('is_in_behind', False))
        overload_plays = sum(1 for o in options if o.get('in_overload', False))

        # Best option
        best_option = options[0] if options else None

        # Tactical summary
        tactical_summary = {
            'ball_pressure': ball_pressure,
            'pressure_level': 'HIGH' if ball_pressure > 0.6 else 'MEDIUM' if ball_pressure > 0.3 else 'LOW',
            'weak_side': weak_side,
            'overloads': overloads,
            'has_overload': len(overloads) > 0,
            'switches_available': switches,
            'through_balls_available': through_balls,
            'overload_options': overload_plays,
        }

        # Build recommendation text
        if ball_pressure > 0.6:
            if high_value > 0:
                advice = "Under pressure - look for the HIGH VALUE forward option"
            elif through_balls > 0:
                advice = "Under pressure - through ball available to release pressure"
            else:
                advice = "Under pressure - find a safe outlet or shield the ball"
        elif len(overloads) > 0:
            advice = f"Overload detected in {overloads[0]['zone']} ({overloads[0]['attackers']}v{overloads[0]['defenders']}) - exploit it!"
        elif switches > 0:
            advice = f"Switch of play available to {weak_side} side"
        elif through_balls > 0:
            advice = "Space in behind - look for the through ball"
        elif high_value > 0:
            advice = "Good options available - play forward"
        else:
            advice = "Build-up phase - retain possession"

        tactical_summary['advice'] = advice

        # Chain ExG info with best sequence visualization
        chain_info = {
            'current_position_xg': round(current_xg, 4),
            'current_position_ev': round(current_ev, 4),
            'best_chain': best_chain,
            'chain_length': len(best_chain),
            'explanation': f"From here, your best chain of plays gives {current_ev:.1%} xG"
        }

        return jsonify({
            'gaps': gaps,
            'options': options,
            'best_option': best_option,
            'total_options': len(options),
            'high_value_options': high_value,
            'safe_options': safe,
            'tactical': tactical_summary,
            'chain_ev': chain_info,
            'best_chain': best_chain,
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
    current_state['ball'] = {"x": 15, "y": 0}  # With #5 (CB/DM)
    return jsonify({'success': True})


# ============================================================
# ACTION SIMULATION - Player Movement AI
# ============================================================

def get_attacker_target(attacker, ball_dest_x, ball_dest_y, is_receiver, ball_carrier_id,
                        occupied_positions=None, chain_players=None):
    """Determine where an attacker should run during an action.

    Players maintain their relative positions and spacing. Chain players
    (those in the optimal passing sequence) hold position to keep the chain valid.

    Args:
        attacker: Attacker dict with x, y, id
        ball_dest_x, ball_dest_y: Where the ball is going
        is_receiver: True if this attacker is receiving the ball
        ball_carrier_id: ID of current ball carrier (to exclude)
        occupied_positions: List of (x, y) positions already claimed by other players
        chain_players: Set of player IDs involved in the optimal chain (should hold position)

    Returns:
        (target_x, target_y) - where this attacker should move toward
    """
    if occupied_positions is None:
        occupied_positions = []
    if chain_players is None:
        chain_players = set()

    MIN_PLAYER_SPACING = 5  # Minimum yards between players

    def is_position_free(x, y):
        """Check if position is far enough from other players."""
        for ox, oy in occupied_positions:
            if calculate_distance(x, y, ox, oy) < MIN_PLAYER_SPACING:
                return False
        return True

    if attacker['id'] == 1:  # GK stays put
        return attacker['x'], attacker['y']

    if attacker['id'] == ball_carrier_id:  # Ball carrier doesn't run
        return attacker['x'], attacker['y']

    if is_receiver:
        # Receiver moves to ball destination
        return ball_dest_x, ball_dest_y

    att_x, att_y = attacker['x'], attacker['y']

    # If this player is part of the optimal chain, HOLD POSITION
    # This keeps the chain valid after the first pass
    if attacker['id'] in chain_players:
        return att_x, att_y

    # Non-chain players: make supporting runs but maintain spacing
    offside_line = get_offside_line()

    # Determine movement based on position relative to ball
    if att_x > ball_dest_x + 8:
        # Ahead of ball - hold width, slight adjustment to stay onside
        target_x = min(att_x, offside_line - 1)
        target_y = att_y
    elif att_x > ball_dest_x:
        # Slightly ahead - hold position, maybe adjust laterally
        target_x = att_x
        # Move away from ball to create space for chain
        if abs(att_y - ball_dest_y) < 8:
            target_y = att_y + (4 if att_y > ball_dest_y else -4)
        else:
            target_y = att_y
    else:
        # Behind ball - move forward to support but not into chain space
        target_x = min(att_x + 4, ball_dest_x - 5)
        target_y = att_y

    # Check if target is free, if not adjust
    if not is_position_free(target_x, target_y):
        # Try moving wider
        for offset in [5, -5, 8, -8]:
            test_y = att_y + offset
            if abs(test_y) <= 35 and is_position_free(target_x, test_y):
                target_y = test_y
                break

    # Stay onside
    target_x = min(target_x, offside_line - 1)

    # Stay in bounds
    target_x = max(-55, min(55, target_x))
    target_y = max(-35, min(35, target_y))

    return target_x, target_y


def get_defender_target(defender, ball_dest_x, ball_dest_y, attackers, ball_start_x, ball_start_y):
    """Determine where a defender should move during an action.

    Defenders move as a unit - shift laterally toward ball, maintain shape.

    Args:
        defender: Defender dict with x, y, id
        ball_dest_x, ball_dest_y: Where the ball is going
        attackers: List of attacker dicts
        ball_start_x, ball_start_y: Where the ball started

    Returns:
        (target_x, target_y) - where this defender should move toward
    """
    if defender['id'] == 1:  # GK stays on line, shifts laterally
        # GK shifts toward ball destination side
        target_y = ball_dest_y * 0.3  # Move partway toward ball side
        return defender['x'], target_y

    def_x, def_y = defender['x'], defender['y']

    # Unit-based movement: shift toward ball side as a block
    # Calculate lateral shift amount based on ball position
    # Ball at y=0 means centered, ball at y=30 means shift right

    # Shift amount: move toward ball side, but don't over-commit
    lateral_shift = ball_dest_y * 0.15  # Shift 15% toward ball side

    # Compress horizontally when ball is central, expand when wide
    if abs(ball_dest_y) < 10:
        # Ball central - compress slightly
        compress_factor = 0.95
    else:
        # Ball wide - maintain or slight expand
        compress_factor = 1.0

    target_y = def_y * compress_factor + lateral_shift

    # Minimal forward/backward movement - just maintain line depth
    # Don't chase, stay in shape
    target_x = def_x

    # Clamp to reasonable positions
    target_y = max(-35, min(35, target_y))

    return target_x, target_y


def move_player_toward(player_x, player_y, target_x, target_y, max_distance):
    """Move player toward target, limited by max_distance.

    Args:
        player_x, player_y: Current position
        target_x, target_y: Target position
        max_distance: Maximum distance player can move

    Returns:
        (new_x, new_y) - new player position
    """
    dx = target_x - player_x
    dy = target_y - player_y
    dist = math.sqrt(dx**2 + dy**2)

    if dist <= 0:
        return player_x, player_y

    if dist <= max_distance:
        # Can reach target
        return target_x, target_y
    else:
        # Move max_distance toward target
        ratio = max_distance / dist
        new_x = player_x + dx * ratio
        new_y = player_y + dy * ratio
        return new_x, new_y


def simulate_action(action):
    """Simulate an action and update game state with player movement.

    Chain-aware: Players involved in the optimal chain hold their positions
    so the chain remains valid. Other players move but avoid overlapping.

    Args:
        action: Dict with action details (action type, target_x, target_y, etc.)

    Returns:
        Dict with movement details and new state
    """
    ball = current_state['ball']
    attackers = current_state['attackers']
    defenders = current_state['defenders']

    ball_start_x = ball['x']
    ball_start_y = ball['y']
    ball_dest_x = action.get('target_x', ball_start_x)
    ball_dest_y = action.get('target_y', ball_start_y)
    action_type = action.get('action', 'pass')
    target_player_id = action.get('target_player')

    # Find ball carrier (attacker closest to ball)
    ball_carrier_id = None
    min_dist = float('inf')
    for att in attackers:
        d = calculate_distance(att['x'], att['y'], ball_start_x, ball_start_y)
        if d < min_dist:
            min_dist = d
            ball_carrier_id = att['id']

    # Calculate the optimal chain from the NEW ball position
    # Players in this chain should hold position to keep the chain valid
    chain_result = calculate_position_value(ball_dest_x, ball_dest_y, depth=0, max_depth=2)
    chain_players = set()
    for step in chain_result.get('chain', []):
        if 'player_id' in step:
            chain_players.add(step['player_id'])

    # Calculate action duration
    if action_type == 'dribble':
        # Dribble time
        dribble_dist = calculate_distance(ball_start_x, ball_start_y, ball_dest_x, ball_dest_y)
        duration = time_to_dribble_distance(dribble_dist)
    elif action_type == 'shoot':
        # Shot is instant (or very fast) - minimal movement
        duration = 0.3
    else:
        # Pass or through ball - ball travel time
        pass_dist = calculate_distance(ball_start_x, ball_start_y, ball_dest_x, ball_dest_y)
        ball_speed = optimal_pass_speed(pass_dist)
        duration = ball_travel_time(pass_dist, ball_speed)
        if duration == float('inf'):
            duration = pass_dist / PASS_SPEED_MIN  # Fallback

    # Clamp duration to reasonable range
    duration = max(0.1, min(duration, 5.0))

    # Calculate movement distances
    # Attackers: full time to move (they anticipate)
    attacker_move_dist = distance_run_in_time(duration)

    # Defenders: lose reaction time
    defender_time = max(0, duration - REACTION_TIME)
    defender_move_dist = distance_run_in_time(defender_time)

    # Track movements for visualization
    movements = []

    # Track occupied positions to avoid overlaps
    occupied_positions = []

    # First, determine where each attacker will go (two passes to handle dependencies)
    attacker_targets = {}

    # Pass 1: Chain players and receiver get priority
    for attacker in attackers:
        is_receiver = attacker['id'] == target_player_id

        if is_receiver:
            # Receiver goes to ball
            attacker_targets[attacker['id']] = (ball_dest_x, ball_dest_y)
            occupied_positions.append((ball_dest_x, ball_dest_y))
        elif attacker['id'] in chain_players:
            # Chain players hold position
            attacker_targets[attacker['id']] = (attacker['x'], attacker['y'])
            occupied_positions.append((attacker['x'], attacker['y']))

    # Pass 2: Non-chain players find positions avoiding occupied spaces
    for attacker in attackers:
        if attacker['id'] in attacker_targets:
            continue  # Already assigned

        target_x, target_y = get_attacker_target(
            attacker, ball_dest_x, ball_dest_y,
            is_receiver=False,
            ball_carrier_id=ball_carrier_id,
            occupied_positions=occupied_positions,
            chain_players=chain_players
        )
        attacker_targets[attacker['id']] = (target_x, target_y)
        occupied_positions.append((target_x, target_y))

    # Now move attackers to their targets
    for attacker in attackers:
        target_x, target_y = attacker_targets[attacker['id']]

        old_x, old_y = attacker['x'], attacker['y']
        new_x, new_y = move_player_toward(old_x, old_y, target_x, target_y, attacker_move_dist)

        # Clamp to pitch
        new_x = max(-58, min(58, new_x))
        new_y = max(-36, min(36, new_y))

        if old_x != new_x or old_y != new_y:
            movements.append({
                'player_id': attacker['id'],
                'team': 0,
                'from_x': old_x,
                'from_y': old_y,
                'to_x': new_x,
                'to_y': new_y,
                'distance': calculate_distance(old_x, old_y, new_x, new_y)
            })

        attacker['x'] = new_x
        attacker['y'] = new_y

    # Move defenders
    for defender in defenders:
        target_x, target_y = get_defender_target(
            defender, ball_dest_x, ball_dest_y, attackers, ball_start_x, ball_start_y
        )

        old_x, old_y = defender['x'], defender['y']
        new_x, new_y = move_player_toward(old_x, old_y, target_x, target_y, defender_move_dist)

        # Clamp to pitch
        new_x = max(-58, min(58, new_x))
        new_y = max(-36, min(36, new_y))

        if old_x != new_x or old_y != new_y:
            movements.append({
                'player_id': defender['id'],
                'team': 1,
                'from_x': old_x,
                'from_y': old_y,
                'to_x': new_x,
                'to_y': new_y,
                'distance': calculate_distance(old_x, old_y, new_x, new_y)
            })

        defender['x'] = new_x
        defender['y'] = new_y

    # Update ball position
    current_state['ball']['x'] = ball_dest_x
    current_state['ball']['y'] = ball_dest_y

    return {
        'action': action_type,
        'duration': round(duration, 2),
        'ball_from': {'x': ball_start_x, 'y': ball_start_y},
        'ball_to': {'x': ball_dest_x, 'y': ball_dest_y},
        'attacker_max_run': round(attacker_move_dist, 1),
        'defender_max_run': round(defender_move_dist, 1),
        'movements': movements,
        'players_moved': len(movements)
    }


@app.route('/api/play_action', methods=['POST'])
def play_action():
    """Execute an action and simulate the resulting state.

    Expects JSON with action details:
    {
        "action": "pass" | "through_ball" | "dribble" | "shoot",
        "target_x": float,
        "target_y": float,
        "target_player": int (optional, for passes)
    }

    Returns simulation results and new analysis.
    """
    try:
        data = request.json or {}

        # Simulate the action
        sim_result = simulate_action(data)

        # Run analysis on new state
        ball = current_state['ball']
        current_result = calculate_position_value(ball['x'], ball['y'], depth=0, max_depth=3)
        current_ev = current_result['value']
        best_chain = current_result['chain']
        current_xg = calculate_xg(ball['x'], ball['y'])

        options = analyze_passing_options()
        gaps = find_gaps()

        # Tactical summary
        ball_pressure = calculate_ball_pressure()
        overloads = detect_overloads()
        weak_side = get_weak_side(ball['y'])

        tactical_summary = {
            'ball_pressure': ball_pressure,
            'pressure_level': 'HIGH' if ball_pressure > 0.6 else 'MEDIUM' if ball_pressure > 0.3 else 'LOW',
            'weak_side': weak_side,
            'overloads': overloads,
            'has_overload': len(overloads) > 0,
        }

        # Chain EV info
        chain_info = {
            'current_position_xg': round(current_xg, 4),
            'current_position_ev': round(current_ev, 4),
            'best_chain': best_chain,
            'chain_length': len(best_chain),
        }

        return jsonify({
            'success': True,
            'simulation': sim_result,
            'new_state': {
                'players': get_all_players(),
                'ball': current_state['ball'],
            },
            'analysis': {
                'gaps': gaps,
                'options': options,
                'best_option': options[0] if options else None,
                'total_options': len(options),
                'tactical': tactical_summary,
                'chain_ev': chain_info,
                'best_chain': best_chain,
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("Decision Engine Web UI")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
