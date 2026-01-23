#!/usr/bin/env python3
"""
Football Decision Engine - Interactive UI

A Streamlit-based interface for visualizing and interacting with
the football decision engine tactical analysis.
"""

import sys
sys.path.insert(0, 'src')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.colors as mcolors

from decision_engine import (
    Position, Player, GameState, GameStateEvaluator,
    EliminationCalculator, DefensiveBlock, BlockType,
    DecisionEngineVisualizer, PitchGeometry,
    HALF_LENGTH, HALF_WIDTH,
)

# Page config
st.set_page_config(
    page_title="Football Decision Engine",
    page_icon="",
    layout="wide",
)

# Title
st.title("Football Decision Engine")
st.markdown("*Interactive tactical analysis and visualization*")

# Initialize session state for players
if 'attackers' not in st.session_state:
    st.session_state.attackers = [
        {"id": "ATT1", "x": 30.0, "y": 10.0},
        {"id": "ATT2", "x": 25.0, "y": -5.0},
        {"id": "ATT3", "x": 20.0, "y": 15.0},
        {"id": "ATT4", "x": 15.0, "y": 0.0},
    ]

if 'defenders' not in st.session_state:
    st.session_state.defenders = [
        {"id": "DEF1", "x": 35.0, "y": 8.0},
        {"id": "DEF2", "x": 32.0, "y": -5.0},
        {"id": "DEF3", "x": 28.0, "y": 0.0},
        {"id": "DEF4", "x": 25.0, "y": 10.0},
        {"id": "DEF5", "x": 40.0, "y": -10.0},
        {"id": "DEF6", "x": 40.0, "y": 10.0},
        {"id": "DEF7", "x": 45.0, "y": -5.0},
        {"id": "DEF8", "x": 45.0, "y": 5.0},
    ]

if 'ball_carrier_idx' not in st.session_state:
    st.session_state.ball_carrier_idx = 0


def draw_pitch(ax):
    """Draw a football pitch on the given axes."""
    # Pitch outline
    ax.plot([-HALF_LENGTH, HALF_LENGTH], [-HALF_WIDTH, -HALF_WIDTH], 'white', lw=2)
    ax.plot([-HALF_LENGTH, HALF_LENGTH], [HALF_WIDTH, HALF_WIDTH], 'white', lw=2)
    ax.plot([-HALF_LENGTH, -HALF_LENGTH], [-HALF_WIDTH, HALF_WIDTH], 'white', lw=2)
    ax.plot([HALF_LENGTH, HALF_LENGTH], [-HALF_WIDTH, HALF_WIDTH], 'white', lw=2)

    # Center line
    ax.plot([0, 0], [-HALF_WIDTH, HALF_WIDTH], 'white', lw=1)

    # Center circle
    circle = Circle((0, 0), 9.15, fill=False, color='white', lw=1)
    ax.add_patch(circle)

    # Penalty areas
    pen_length = 16.5
    pen_width = 20.16
    # Left
    ax.plot([-HALF_LENGTH, -HALF_LENGTH + pen_length], [-pen_width, -pen_width], 'white', lw=1)
    ax.plot([-HALF_LENGTH, -HALF_LENGTH + pen_length], [pen_width, pen_width], 'white', lw=1)
    ax.plot([-HALF_LENGTH + pen_length, -HALF_LENGTH + pen_length], [-pen_width, pen_width], 'white', lw=1)
    # Right
    ax.plot([HALF_LENGTH, HALF_LENGTH - pen_length], [-pen_width, -pen_width], 'white', lw=1)
    ax.plot([HALF_LENGTH, HALF_LENGTH - pen_length], [pen_width, pen_width], 'white', lw=1)
    ax.plot([HALF_LENGTH - pen_length, HALF_LENGTH - pen_length], [-pen_width, pen_width], 'white', lw=1)

    # Goal areas
    goal_length = 5.5
    goal_width = 9.16
    # Left
    ax.plot([-HALF_LENGTH, -HALF_LENGTH + goal_length], [-goal_width, -goal_width], 'white', lw=1)
    ax.plot([-HALF_LENGTH, -HALF_LENGTH + goal_length], [goal_width, goal_width], 'white', lw=1)
    ax.plot([-HALF_LENGTH + goal_length, -HALF_LENGTH + goal_length], [-goal_width, goal_width], 'white', lw=1)
    # Right
    ax.plot([HALF_LENGTH, HALF_LENGTH - goal_length], [-goal_width, -goal_width], 'white', lw=1)
    ax.plot([HALF_LENGTH, HALF_LENGTH - goal_length], [goal_width, goal_width], 'white', lw=1)
    ax.plot([HALF_LENGTH - goal_length, HALF_LENGTH - goal_length], [-goal_width, goal_width], 'white', lw=1)

    # Goals
    ax.plot([-HALF_LENGTH - 2, -HALF_LENGTH], [-3.66, -3.66], 'white', lw=3)
    ax.plot([-HALF_LENGTH - 2, -HALF_LENGTH], [3.66, 3.66], 'white', lw=3)
    ax.plot([-HALF_LENGTH - 2, -HALF_LENGTH - 2], [-3.66, 3.66], 'white', lw=3)
    ax.plot([HALF_LENGTH + 2, HALF_LENGTH], [-3.66, -3.66], 'white', lw=3)
    ax.plot([HALF_LENGTH + 2, HALF_LENGTH], [3.66, 3.66], 'white', lw=3)
    ax.plot([HALF_LENGTH + 2, HALF_LENGTH + 2], [-3.66, 3.66], 'white', lw=3)

    ax.set_xlim(-60, 60)
    ax.set_ylim(-40, 40)
    ax.set_facecolor('#2e7d32')
    ax.set_aspect('equal')
    ax.axis('off')


def create_visualization(attackers, defenders, ball_carrier_idx, evaluated_state):
    """Create the main tactical visualization."""
    fig, ax = plt.subplots(figsize=(14, 9))
    draw_pitch(ax)

    # Draw defenders
    for i, d in enumerate(defenders):
        is_eliminated = False
        if evaluated_state and evaluated_state.elimination_state:
            for result in evaluated_state.elimination_state.defenders:
                if result.defender.id == d['id']:
                    is_eliminated = result.is_eliminated
                    break

        color = '#ef5350' if is_eliminated else '#e53935'
        edge_color = '#b71c1c' if is_eliminated else 'white'
        alpha = 0.5 if is_eliminated else 1.0

        circle = Circle((d['x'], d['y']), 2, color=color, ec=edge_color, lw=2, alpha=alpha)
        ax.add_patch(circle)
        ax.text(d['x'], d['y'], d['id'].replace('DEF', 'D'),
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    # Draw attackers
    for i, a in enumerate(attackers):
        is_carrier = (i == ball_carrier_idx)
        color = '#fdd835' if is_carrier else '#4caf50'

        circle = Circle((a['x'], a['y']), 2, color=color, ec='white', lw=2)
        ax.add_patch(circle)
        ax.text(a['x'], a['y'], a['id'].replace('ATT', 'A'),
                ha='center', va='center', fontsize=8, color='black', fontweight='bold')

        # Draw ball
        if is_carrier:
            ax.plot(a['x'] + 1.5, a['y'] + 1.5, 'o', color='white', markersize=8,
                   markeredgecolor='black', markeredgewidth=1)

    # Add title with score
    if evaluated_state and evaluated_state.score:
        score = evaluated_state.score.total
        ax.set_title(f"Game State Score: {score:.3f}", fontsize=16, color='white', pad=20)

    plt.tight_layout()
    return fig


def create_heatmap(attackers, defenders, ball_carrier_idx):
    """Create a position value heatmap."""
    evaluator = GameStateEvaluator()

    # Create player objects
    attacker_objs = [Player(id=a['id'], position=Position(a['x'], a['y']), team="attack")
                    for a in attackers]
    defender_objs = [Player(id=d['id'], position=Position(d['x'], d['y']), team="defense")
                    for d in defenders]

    ball_carrier = attacker_objs[ball_carrier_idx]

    # Create base state
    state = GameState(
        ball_position=ball_carrier.position,
        ball_carrier=ball_carrier,
        attackers=attacker_objs,
        defenders=defender_objs,
    )

    # Generate heatmap
    heatmap = evaluator.generate_value_heatmap(state, grid_resolution=5.0)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw heatmap
    extent = [-HALF_LENGTH, HALF_LENGTH, -HALF_WIDTH, HALF_WIDTH]
    im = ax.imshow(heatmap, extent=extent, origin='lower', cmap='RdYlGn',
                   alpha=0.7, aspect='auto', vmin=0, vmax=1)

    # Draw pitch overlay
    draw_pitch(ax)

    # Draw players
    for d in defenders:
        ax.plot(d['x'], d['y'], 'o', color='red', markersize=12,
               markeredgecolor='white', markeredgewidth=2)

    for i, a in enumerate(attackers):
        color = 'yellow' if i == ball_carrier_idx else 'lime'
        ax.plot(a['x'], a['y'], 'o', color=color, markersize=12,
               markeredgecolor='white', markeredgewidth=2)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Position Value', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_title("Position Value Heatmap", fontsize=16, color='white', pad=20)
    plt.tight_layout()
    return fig


# Sidebar controls
st.sidebar.header("Controls")

# Analysis type
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Game State", "Heatmap", "Defensive Blocks"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Ball Carrier")

ball_carrier_options = [f"ATT{i+1}" for i in range(len(st.session_state.attackers))]
selected_carrier = st.sidebar.selectbox("Select Ball Carrier", ball_carrier_options)
st.session_state.ball_carrier_idx = ball_carrier_options.index(selected_carrier)

st.sidebar.markdown("---")
st.sidebar.subheader("Attackers")

# Attacker position controls
for i, att in enumerate(st.session_state.attackers):
    with st.sidebar.expander(f"ATT{i+1}"):
        col1, col2 = st.columns(2)
        with col1:
            att['x'] = st.slider(f"X", -50.0, 50.0, att['x'], key=f"att_{i}_x")
        with col2:
            att['y'] = st.slider(f"Y", -30.0, 30.0, att['y'], key=f"att_{i}_y")

st.sidebar.markdown("---")
st.sidebar.subheader("Defenders")

# Defender position controls
for i, def_ in enumerate(st.session_state.defenders):
    with st.sidebar.expander(f"DEF{i+1}"):
        col1, col2 = st.columns(2)
        with col1:
            def_['x'] = st.slider(f"X", -50.0, 50.0, def_['x'], key=f"def_{i}_x")
        with col2:
            def_['y'] = st.slider(f"Y", -30.0, 30.0, def_['y'], key=f"def_{i}_y")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Create player objects
    attacker_objs = [Player(id=a['id'], position=Position(a['x'], a['y']), team="attack")
                    for a in st.session_state.attackers]
    defender_objs = [Player(id=d['id'], position=Position(d['x'], d['y']), team="defense")
                    for d in st.session_state.defenders]

    ball_carrier = attacker_objs[st.session_state.ball_carrier_idx]

    # Create game state
    state = GameState(
        ball_position=ball_carrier.position,
        ball_carrier=ball_carrier,
        attackers=attacker_objs,
        defenders=defender_objs,
    )

    # Evaluate
    evaluator = GameStateEvaluator()
    evaluated = evaluator.evaluate(state)

    if analysis_type == "Game State":
        fig = create_visualization(
            st.session_state.attackers,
            st.session_state.defenders,
            st.session_state.ball_carrier_idx,
            evaluated
        )
        st.pyplot(fig)
        plt.close(fig)

    elif analysis_type == "Heatmap":
        fig = create_heatmap(
            st.session_state.attackers,
            st.session_state.defenders,
            st.session_state.ball_carrier_idx
        )
        st.pyplot(fig)
        plt.close(fig)

    elif analysis_type == "Defensive Blocks":
        block_type = st.selectbox("Block Type", ["LOW", "MID", "HIGH"])
        block = DefensiveBlock(BlockType[block_type])

        fig, ax = plt.subplots(figsize=(14, 9))
        draw_pitch(ax)

        # Draw block lines
        ball_x = ball_carrier.position.x
        positions = block.get_positions(Position(ball_x, 0), "4-4-2")

        ax.axhline(y=0, color='yellow', alpha=0.3, lw=40,
                  xmin=(block.defensive_line_height + HALF_LENGTH) / (2 * HALF_LENGTH),
                  xmax=(block.forward_line_height + HALF_LENGTH) / (2 * HALF_LENGTH))

        ax.axvline(x=block.defensive_line_height, color='red', linestyle='--', lw=2, label='Defensive Line')
        ax.axvline(x=block.midfield_line_height, color='orange', linestyle='--', lw=2, label='Midfield Line')
        ax.axvline(x=block.forward_line_height, color='yellow', linestyle='--', lw=2, label='Forward Line')

        ax.legend(loc='upper left', facecolor='#2e7d32', labelcolor='white')
        ax.set_title(f"{block_type} Block Configuration", fontsize=16, color='white', pad=20)

        st.pyplot(fig)
        plt.close(fig)

with col2:
    st.subheader("Analysis Results")

    if evaluated.score:
        st.metric("Total Score", f"{evaluated.score.total:.3f}")

        st.markdown("**Component Scores:**")
        scores = {
            "Elimination (25%)": evaluated.score.elimination_score,
            "Proximity (20%)": evaluated.score.proximity_score,
            "Angle (15%)": evaluated.score.angle_score,
            "Density (15%)": evaluated.score.density_score,
            "Compactness (10%)": evaluated.score.compactness_score,
            "Actions (15%)": evaluated.score.action_score,
        }

        for name, score in scores.items():
            st.progress(score, text=f"{name}: {score:.3f}")

    st.markdown("---")
    st.subheader("Elimination Status")

    if evaluated.elimination_state:
        st.write(f"**Eliminated:** {evaluated.elimination_state.eliminated_count} / {len(defender_objs)}")
        st.write(f"**Ratio:** {evaluated.elimination_state.elimination_ratio:.1%}")

        for result in evaluated.elimination_state.defenders:
            status = "Eliminated" if result.is_eliminated else "Active"
            icon = "" if result.is_eliminated else ""
            st.write(f"{icon} {result.defender.id}: {status}")

    st.markdown("---")
    st.subheader("Available Actions")

    if evaluated.available_actions:
        for action in evaluated.available_actions[:5]:
            with st.expander(f"{action.action_type.value.title()}"):
                st.write(f"Target: ({action.target.x:.1f}, {action.target.y:.1f})")
                st.write(f"Success: {action.success_probability:.1%}")
                st.write(f"Expected Value: {action.expected_value:.3f}")

# Footer
st.markdown("---")
st.markdown("*Football Decision Engine - Marshall Men's Soccer*")
