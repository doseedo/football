#!/usr/bin/env python3
"""
Run the Decision Engine example.

Usage:
    python run_decision_engine.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.decision_engine import (
    DecisionEngine,
    PlayerState,
    PitchVisualizer,
)


def create_example_scenario():
    """
    Create an example game scenario.

    Team A (blue) attacking right, Team B (red) defending.
    """
    # Team A (blue) - attacking right
    team_a = [
        PlayerState(player_id=1, team=0, position=(-45, 0), max_speed=6.0),  # GK
        PlayerState(player_id=2, team=0, position=(-30, -20), max_speed=7.5),
        PlayerState(player_id=3, team=0, position=(-32, -5), max_speed=7.5),
        PlayerState(player_id=4, team=0, position=(-32, 5), max_speed=7.5),
        PlayerState(player_id=5, team=0, position=(-30, 20), max_speed=7.5),
        PlayerState(player_id=6, team=0, position=(-10, -15), max_speed=8.0),
        PlayerState(player_id=7, team=0, position=(-5, 0), max_speed=8.0),  # Ball carrier
        PlayerState(player_id=8, team=0, position=(-10, 15), max_speed=8.0),
        PlayerState(player_id=9, team=0, position=(20, -12), max_speed=8.5),
        PlayerState(player_id=10, team=0, position=(25, 5), max_speed=8.5),
        PlayerState(player_id=11, team=0, position=(18, 18), max_speed=8.5),
    ]

    # Team B (red) - defending, with a gap between #4 and #5
    team_b = [
        PlayerState(player_id=1, team=1, position=(45, 0), max_speed=6.0),  # GK
        PlayerState(player_id=2, team=1, position=(30, -22), max_speed=7.5),
        PlayerState(player_id=3, team=1, position=(32, -8), max_speed=7.5),
        PlayerState(player_id=4, team=1, position=(33, 3), max_speed=7.5),
        # GAP HERE - 15m between #4 and #5
        PlayerState(player_id=5, team=1, position=(30, 18), max_speed=7.5),
        PlayerState(player_id=6, team=1, position=(15, -10), max_speed=8.0),
        PlayerState(player_id=7, team=1, position=(10, 5), max_speed=8.0),
        PlayerState(player_id=8, team=1, position=(12, 15), max_speed=8.0),
        PlayerState(player_id=9, team=1, position=(-15, -5), max_speed=8.5),
        PlayerState(player_id=10, team=1, position=(-20, 10), max_speed=8.5),
        PlayerState(player_id=11, team=1, position=(-25, 0), max_speed=8.5),
    ]

    ball_position = (-5, 0)
    ball_carrier_id = 7

    return team_a, team_b, ball_position, ball_carrier_id


def main():
    print("=" * 60)
    print("DECISION ENGINE - BALL PROGRESSION ANALYSIS")
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

    print(f"\n{'='*40}")
    print("DEFENSIVE GAPS DETECTED")
    print('='*40)

    if result.defensive_gaps:
        for i, gap in enumerate(result.defensive_gaps[:5]):
            status = "✓ EXPLOITABLE" if gap.exploitable else "✗ Covered"
            print(f"\n  Gap {i+1}: {status}")
            print(f"    Location: ({gap.location[0]:.1f}, {gap.location[1]:.1f})")
            print(f"    Size: {gap.size:.1f}m")
            print(f"    Time to close: {gap.time_to_close:.2f}s")
            print(f"    xG if exploited: {gap.xg_if_exploited:.3f}")
    else:
        print("  No significant gaps detected")

    print(f"\n{'='*40}")
    print("PROGRESSION OPTIONS")
    print('='*40)
    print(f"\n  Total options: {result.total_options}")
    print(f"  High value (EV > 0.05): {result.high_value_options}")
    print(f"  Safe (>80% success): {result.safe_options}")

    print(f"\n{'='*40}")
    print("TOP 5 OPTIONS BY EXPECTED VALUE")
    print('='*40)

    for i, opt in enumerate(result.progression_options[:5]):
        print(f"\n  Option {i+1}: {opt.action_type.upper()} → {opt.recommendation}")
        if opt.target_player_id:
            print(f"    Target: Player #{opt.target_player_id}")
        print(f"    Position: ({opt.target_position[0]:.1f}, {opt.target_position[1]:.1f})")
        print(f"    Success: {opt.success_probability:.0%}")
        print(f"    Intercept risk: {opt.interception_probability:.0%}")
        print(f"    xG: {opt.xg_current:.3f} → {opt.xg_target:.3f} ({opt.xg_gain:+.3f})")
        print(f"    Turnover cost: {opt.turnover_cost:.3f}")
        print(f"    EXPECTED VALUE: {opt.expected_value:+.4f}")

    print(f"\n{'='*40}")
    print("RECOMMENDED ACTION")
    print('='*40)

    if result.best_option:
        opt = result.best_option
        print(f"\n  → {opt.action_type.upper()}")
        if opt.target_player_id:
            print(f"    Target player: #{opt.target_player_id}")
        print(f"    Target position: ({opt.target_position[0]:.1f}, {opt.target_position[1]:.1f})")
        print(f"    Success probability: {opt.success_probability:.0%}")
        print(f"    Expected Value: {opt.expected_value:+.4f}")
        print(f"    Recommendation: {opt.recommendation}")
    else:
        print("  No viable options")

    # Try to create visualization
    print(f"\n{'='*40}")
    print("VISUALIZATION")
    print('='*40)

    try:
        # Create output directory
        os.makedirs('output/decision_engine', exist_ok=True)

        all_players = team_a + team_b

        viz = PitchVisualizer(
            figsize=(14, 10),
            team_colors=('dodgerblue', 'crimson')
        )

        viz.setup_pitch()
        viz.draw_frame(
            players=all_players,
            ball_position=ball_position,
            engine_output=result,
            show_gaps=True,
            show_best_option=True,
            title="Decision Engine Analysis"
        )

        output_path = 'output/decision_engine/analysis.png'
        viz.save(output_path)
        print(f"\n  ✓ Saved visualization to: {output_path}")

        viz.close()

    except ImportError as e:
        print(f"\n  ✗ Matplotlib not available: {e}")
    except Exception as e:
        print(f"\n  ✗ Visualization failed: {e}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
