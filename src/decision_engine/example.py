"""
Example usage of the Decision Engine.

This script demonstrates:
1. Creating player states from tracking data
2. Running the decision engine analysis
3. Interpreting results
4. Visualizing on a pitch
"""

import numpy as np
from typing import List

from . import (
    DecisionEngine,
    PlayerState,
    PitchVisualizer,
    PITCH_LENGTH,
    PITCH_WIDTH,
)


def create_example_scenario():
    """
    Create an example game scenario.

    Returns team A attacking, team B defending.
    """
    # Team A (blue) - attacking right
    team_a = [
        # Goalkeeper
        PlayerState(player_id=1, team=0, position=(-45, 0), max_speed=6.0),
        # Defenders
        PlayerState(player_id=2, team=0, position=(-30, -20), max_speed=7.5),
        PlayerState(player_id=3, team=0, position=(-32, -5), max_speed=7.5),
        PlayerState(player_id=4, team=0, position=(-32, 5), max_speed=7.5),
        PlayerState(player_id=5, team=0, position=(-30, 20), max_speed=7.5),
        # Midfielders
        PlayerState(player_id=6, team=0, position=(-10, -15), max_speed=8.0),
        PlayerState(player_id=7, team=0, position=(-5, 0), max_speed=8.0),  # Ball carrier
        PlayerState(player_id=8, team=0, position=(-10, 15), max_speed=8.0),
        # Forwards
        PlayerState(player_id=9, team=0, position=(20, -12), max_speed=8.5),
        PlayerState(player_id=10, team=0, position=(25, 5), max_speed=8.5),
        PlayerState(player_id=11, team=0, position=(18, 18), max_speed=8.5),
    ]

    # Team B (red) - defending left goal
    team_b = [
        # Goalkeeper
        PlayerState(player_id=1, team=1, position=(45, 0), max_speed=6.0),
        # Defenders - notice gap between 4 and 5
        PlayerState(player_id=2, team=1, position=(30, -22), max_speed=7.5),
        PlayerState(player_id=3, team=1, position=(32, -8), max_speed=7.5),
        PlayerState(player_id=4, team=1, position=(33, 3), max_speed=7.5),
        # GAP HERE - 15m between #4 and #5
        PlayerState(player_id=5, team=1, position=(30, 18), max_speed=7.5),
        # Midfielders
        PlayerState(player_id=6, team=1, position=(15, -10), max_speed=8.0),
        PlayerState(player_id=7, team=1, position=(10, 5), max_speed=8.0),
        PlayerState(player_id=8, team=1, position=(12, 15), max_speed=8.0),
        # Forwards
        PlayerState(player_id=9, team=1, position=(-15, -5), max_speed=8.5),
        PlayerState(player_id=10, team=1, position=(-20, 10), max_speed=8.5),
        PlayerState(player_id=11, team=1, position=(-25, 0), max_speed=8.5),
    ]

    ball_position = (-5, 0)  # With player #7
    ball_carrier_id = 7

    return team_a, team_b, ball_position, ball_carrier_id


def run_analysis():
    """Run the decision engine on example scenario."""
    print("=" * 60)
    print("DECISION ENGINE EXAMPLE")
    print("=" * 60)

    # Create scenario
    team_a, team_b, ball_position, ball_carrier_id = create_example_scenario()

    # Initialize engine
    engine = DecisionEngine()

    # Run analysis
    result = engine.analyze(
        ball_position=ball_position,
        ball_carrier_id=ball_carrier_id,
        possession_team=0,
        teammates=team_a,
        opponents=team_b,
        timestamp=34.5
    )

    # Print results
    print(f"\nBall Position: {ball_position}")
    print(f"Ball Carrier: #{ball_carrier_id}")

    print(f"\n--- DEFENSIVE GAPS DETECTED ---")
    for i, gap in enumerate(result.defensive_gaps):
        print(f"\nGap {i+1}:")
        print(f"  Location: ({gap.location[0]:.1f}, {gap.location[1]:.1f})")
        print(f"  Size: {gap.size:.1f}m")
        print(f"  Time to close: {gap.time_to_close:.2f}s")
        print(f"  Exploitable: {gap.exploitable}")
        print(f"  xG if exploited: {gap.xg_if_exploited:.3f}")

    print(f"\n--- PROGRESSION OPTIONS ---")
    print(f"Total options: {result.total_options}")
    print(f"High value (EV > 0.05): {result.high_value_options}")
    print(f"Safe (>80% success): {result.safe_options}")

    print(f"\n--- TOP 5 OPTIONS BY EXPECTED VALUE ---")
    for i, opt in enumerate(result.progression_options[:5]):
        print(f"\nOption {i+1}: {opt.action_type.upper()}")
        if opt.target_player_id:
            print(f"  Target: Player #{opt.target_player_id}")
        print(f"  Target position: ({opt.target_position[0]:.1f}, {opt.target_position[1]:.1f})")
        print(f"  Success probability: {opt.success_probability:.0%}")
        print(f"  Interception risk: {opt.interception_probability:.0%}")
        print(f"  xG current: {opt.xg_current:.3f}")
        print(f"  xG target: {opt.xg_target:.3f}")
        print(f"  xG gain: {opt.xg_gain:+.3f}")
        print(f"  Turnover cost: {opt.turnover_cost:.3f}")
        print(f"  EXPECTED VALUE: {opt.expected_value:+.4f}")
        print(f"  Recommendation: {opt.recommendation}")

    print(f"\n--- BEST OPTION ---")
    if result.best_option:
        opt = result.best_option
        print(f"Action: {opt.action_type}")
        print(f"Target: {opt.target_position}")
        print(f"Success: {opt.success_probability:.0%}")
        print(f"EV: {opt.expected_value:+.4f}")
        print(f"Recommendation: {opt.recommendation}")
    else:
        print("No options available")

    return result, team_a, team_b, ball_position


def visualize_analysis():
    """Visualize the analysis on a pitch."""
    result, team_a, team_b, ball_position = run_analysis()

    all_players = team_a + team_b

    # Create visualizer
    viz = PitchVisualizer(
        figsize=(14, 10),
        team_colors=('dodgerblue', 'crimson')
    )

    # Draw frame with analysis
    viz.setup_pitch()
    viz.draw_frame(
        players=all_players,
        ball_position=ball_position,
        engine_output=result,
        show_gaps=True,
        show_best_option=True,
        title="Decision Engine Analysis"
    )
    viz.draw_analysis_panel(result)

    # Save and show
    viz.save('decision_engine_example.png')
    print("\n\nVisualization saved to: decision_engine_example.png")

    try:
        viz.show()
    except Exception:
        pass  # May fail in non-GUI environment

    viz.close()


def analyze_match_sequence():
    """
    Example: Analyze a sequence of frames.

    In real usage, frames would come from tracking data.
    """
    print("\n" + "=" * 60)
    print("MATCH SEQUENCE ANALYSIS")
    print("=" * 60)

    engine = DecisionEngine()

    # Simulate 5 frames of ball movement
    frames = [
        # Frame 1: Ball in midfield
        {
            'ball_pos': (-5, 0),
            'ball_carrier': 7,
        },
        # Frame 2: Ball moved forward
        {
            'ball_pos': (10, 5),
            'ball_carrier': 10,
        },
        # Frame 3: Ball at edge of box
        {
            'ball_pos': (30, 0),
            'ball_carrier': 9,
        },
    ]

    team_a, team_b, _, _ = create_example_scenario()

    for i, frame in enumerate(frames):
        print(f"\n--- FRAME {i+1} ---")

        result = engine.analyze(
            ball_position=frame['ball_pos'],
            ball_carrier_id=frame['ball_carrier'],
            possession_team=0,
            teammates=team_a,
            opponents=team_b,
            timestamp=30.0 + i * 2
        )

        print(f"Ball at: {frame['ball_pos']}")
        print(f"Current xG zone: {result.best_option.xg_current:.3f}" if result.best_option else "N/A")
        print(f"Gaps detected: {len(result.defensive_gaps)}")
        print(f"High-value options: {result.high_value_options}")

        if result.best_option:
            print(f"Best option: {result.best_option.action_type} "
                  f"(EV: {result.best_option.expected_value:+.4f})")


if __name__ == '__main__':
    # Run basic analysis
    run_analysis()

    # Visualize (if matplotlib available)
    try:
        visualize_analysis()
    except ImportError:
        print("\nMatplotlib not available for visualization")

    # Analyze sequence
    analyze_match_sequence()
