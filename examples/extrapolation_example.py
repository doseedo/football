"""Example usage of off-screen player extrapolation system.

This script demonstrates how to use the TrajectoryPredictor to handle
off-screen players in football tracking.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from src.extrapolation import (
    TrajectoryPredictor,
    PlayerState,
    Baller2VecPlus
)


def example_basic_usage():
    """Basic example of trajectory prediction."""
    logger.info("=== Basic Usage Example ===")

    # Initialize predictor (without pre-trained weights for demo)
    predictor = TrajectoryPredictor(
        model_type='baller2vec_plus',
        model_path=None,  # Set to model path when available
        history_length=25,  # 1 second at 25fps
        confidence_threshold=0.5,
        use_physics_fallback=True,
        device='cpu'  # Use 'cuda' if available
    )

    # Simulate tracking data for 22 players over multiple frames
    num_frames = 50
    timestamps = np.arange(num_frames) * 0.04  # 25fps = 0.04s per frame

    for frame_idx in range(num_frames):
        timestamp = timestamps[frame_idx]

        # Simulate visible players (some players might be off-screen)
        visible_players = []

        for player_id in range(22):
            # Simulate some players going off-screen
            is_visible = not (frame_idx > 20 and player_id in [5, 12, 18])

            if is_visible or frame_idx < 10:  # Always visible in first 10 frames
                # Simulate position (simple circular motion for demo)
                angle = 2 * np.pi * frame_idx / 50 + player_id * np.pi / 11
                x = 52.5 + 30 * np.cos(angle)
                y = 34.0 + 20 * np.sin(angle)

                # Simulate velocity
                vx = -30 * np.sin(angle) * 2 * np.pi / (50 * 0.04)
                vy = 20 * np.cos(angle) * 2 * np.pi / (50 * 0.04)

                # Team assignment (first 11 vs second 11)
                team = 0 if player_id < 11 else 1

                visible_players.append(PlayerState(
                    player_id=player_id,
                    position=(x, y),
                    velocity=(vx, vy),
                    team=team,
                    is_visible=is_visible,
                    confidence=1.0
                ))

        # Predict all 22 players (including off-screen ones)
        result = predictor.predict(
            visible_players=visible_players,
            timestamp=timestamp,
            n_future_steps=1,
            predict_all_players=True
        )

        # Display results
        if frame_idx % 10 == 0:
            logger.info(f"\nFrame {frame_idx} (t={timestamp:.2f}s):")
            logger.info(f"  Visible players: {len(visible_players)}")
            logger.info(f"  Total predicted: {len(result.players)}")
            logger.info(f"  Prediction method: {result.method}")

            # Show extrapolated players
            extrapolated = [p for p in result.players if not p.is_visible]
            if extrapolated:
                logger.info(f"  Extrapolated players: {[p.player_id for p in extrapolated]}")
                for player in extrapolated[:3]:  # Show first 3
                    conf = predictor.get_extrapolation_confidence(player.player_id)
                    logger.info(f"    Player {player.player_id}: pos=({player.position[0]:.1f}, {player.position[1]:.1f}), "
                              f"confidence={conf:.2f}")


def example_physics_fallback():
    """Example demonstrating physics-based fallback."""
    logger.info("\n=== Physics Fallback Example ===")

    # Initialize predictor with only physics model (no transformer)
    predictor = TrajectoryPredictor(
        model_type='baller2vec',
        model_path=None,
        use_physics_fallback=True,
        confidence_threshold=1.0,  # Always use fallback for demo
        device='cpu'
    )

    # Create a player moving in straight line
    player_states = []
    for t in range(10):
        timestamp = t * 0.04
        x = 30.0 + t * 2.0  # Moving at 50 km/h
        y = 34.0

        player_states.append(PlayerState(
            player_id=1,
            position=(x, y),
            velocity=(2.0, 0.0),
            team=0,
            is_visible=True,
            confidence=1.0
        ))

        predictor.update_history([player_states[-1]], timestamp)

    # Now extrapolate 5 steps into the future
    logger.info("Extrapolating 5 steps (0.2s) into future:")

    for step in range(1, 6):
        result = predictor.predict(
            visible_players=[],  # Player now off-screen
            timestamp=10 * 0.04 + step * 0.04,
            n_future_steps=1,
            predict_all_players=True
        )

        if result.players:
            player = result.players[0]
            logger.info(f"  Step {step}: position=({player.position[0]:.2f}, {player.position[1]:.2f}), "
                      f"confidence={player.confidence:.2f}")


def example_team_coordination():
    """Example showing team coordination modeling."""
    logger.info("\n=== Team Coordination Example ===")

    # This example shows how Baller2Vec++ models coordinated team movements
    # For this, we'd need a trained model, but we'll show the concept

    predictor = TrajectoryPredictor(
        model_type='baller2vec_plus',  # Uses coordinated attention
        model_path=None,
        device='cpu'
    )

    # Simulate coordinated attack: 3 forwards moving together
    num_frames = 30
    forwards = [9, 10, 11]  # Player IDs

    for frame_idx in range(num_frames):
        timestamp = frame_idx * 0.04
        visible_players = []

        # Three forwards coordinating an attack
        for i, player_id in enumerate(forwards):
            # They move together in formation
            base_x = 60.0 + frame_idx * 1.5
            base_y = 30.0
            offset_y = (i - 1) * 5.0  # Spread formation

            visible_players.append(PlayerState(
                player_id=player_id,
                position=(base_x, base_y + offset_y),
                velocity=(1.5, 0.0),
                team=0,
                is_visible=True,
                confidence=1.0
            ))

        predictor.update_history(visible_players, timestamp)

        if frame_idx == 25:
            # One forward goes off-screen
            logger.info("\nPlayer 10 goes off-screen, predicting from coordinated movement:")
            result = predictor.predict(
                visible_players=[p for p in visible_players if p.player_id != 10],
                timestamp=timestamp,
                predict_all_players=True
            )

            for player in result.players:
                if player.player_id == 10:
                    logger.info(f"  Extrapolated player 10: pos=({player.position[0]:.1f}, {player.position[1]:.1f}), "
                              f"confidence={player.confidence:.2f}")
                    logger.info("  (Baller2Vec++ uses attention to infer position from teammates)")


def example_confidence_scores():
    """Example showing how confidence scores degrade over time."""
    logger.info("\n=== Confidence Score Example ===")

    predictor = TrajectoryPredictor(
        model_type='baller2vec',
        model_path=None,
        device='cpu'
    )

    # Track a player for 1 second
    for t in range(25):
        timestamp = t * 0.04
        player = PlayerState(
            player_id=5,
            position=(40.0 + t * 0.5, 30.0),
            velocity=(0.5 / 0.04, 0.0),
            team=0,
            is_visible=True,
            confidence=1.0
        )
        predictor.update_history([player], timestamp)

    # Now player goes off-screen, check confidence decay
    logger.info("Player goes off-screen, tracking confidence over time:")

    for t in range(25, 50):
        timestamp = t * 0.04
        result = predictor.predict(
            visible_players=[],
            timestamp=timestamp,
            predict_all_players=True
        )

        confidence = predictor.get_extrapolation_confidence(5)

        if t % 5 == 0:
            time_since_visible = (t - 25) * 0.04
            logger.info(f"  {time_since_visible:.2f}s off-screen: confidence={confidence:.3f}")


def main():
    """Run all examples."""
    logger.info("Off-Screen Player Extrapolation Examples\n")

    # Run examples
    example_basic_usage()
    example_physics_fallback()
    example_team_coordination()
    example_confidence_scores()

    logger.info("\n=== Examples Complete ===")
    logger.info("\nKey Takeaways:")
    logger.info("1. TrajectoryPredictor combines transformer and physics models")
    logger.info("2. Handles variable number of visible players")
    logger.info("3. Provides confidence scores for extrapolations")
    logger.info("4. Baller2Vec++ models team coordination for better predictions")
    logger.info("5. Confidence degrades over time when player is off-screen")


if __name__ == '__main__':
    main()
