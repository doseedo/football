"""Test script for extrapolation module.

Verifies that all components can be instantiated and run basic operations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from loguru import logger

from src.extrapolation import (
    Baller2Vec,
    Baller2VecPlus,
    TrajectoryPredictor,
    PlayerState,
    KalmanMotionModel,
    MultiPlayerMotionModel,
    create_feature_tensor,
    create_padding_mask,
    create_team_tensor
)


def test_kalman_motion_model():
    """Test Kalman filter motion model."""
    logger.info("Testing KalmanMotionModel...")

    model = KalmanMotionModel(dt=0.04)

    # Initialize with position
    model.initialize(np.array([50.0, 34.0]), timestamp=0.0)

    # Update with measurements
    for i in range(10):
        timestamp = i * 0.04
        position = np.array([50.0 + i * 2.0, 34.0])
        state = model.update(position, timestamp)
        logger.debug(f"  t={timestamp:.2f}: pos=({state.position[0]:.2f}, {state.position[1]:.2f}), "
                    f"vel=({state.velocity[0]:.2f}, {state.velocity[1]:.2f})")

    # Extrapolate
    future = model.extrapolate(5)
    logger.info(f"  Extrapolated 5 steps: final pos=({future[-1].position[0]:.2f}, {future[-1].position[1]:.2f})")
    logger.info("  ✓ KalmanMotionModel works")


def test_multi_player_motion_model():
    """Test multi-player motion model."""
    logger.info("Testing MultiPlayerMotionModel...")

    model = MultiPlayerMotionModel()

    # Update multiple players
    for player_id in range(5):
        for t in range(10):
            timestamp = t * 0.04
            position = np.array([30.0 + player_id * 10.0, 34.0 + t * 0.5])
            model.update(player_id, position, timestamp)

    # Get all states
    states = model.get_all_states(timestamp=0.4)
    logger.info(f"  Tracking {len(states)} players")
    logger.info("  ✓ MultiPlayerMotionModel works")


def test_baller2vec():
    """Test Baller2Vec model."""
    logger.info("Testing Baller2Vec...")

    model = Baller2Vec(
        d_model=64,  # Smaller for testing
        num_heads=4,
        num_layers=2,
        max_players=22
    )

    # Create dummy data
    batch_size = 2
    seq_len = 10
    num_players = 6

    positions = np.random.randn(seq_len, num_players, 2) * 10 + 50
    teams = np.array([0, 0, 0, 1, 1, 1])
    velocities = np.random.randn(seq_len, num_players, 2)

    features = create_feature_tensor(positions, teams, velocities)
    features = features.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Forward pass
    predictions = model(features)
    logger.info(f"  Input shape: {features.shape}")
    logger.info(f"  Output shape: {predictions.shape}")
    assert predictions.shape == (batch_size, seq_len, num_players, 2)

    # Predict future
    future = model.predict_future(features, n_future_steps=5, autoregressive=False)
    logger.info(f"  Future prediction shape: {future.shape}")
    assert future.shape == (batch_size, 5, num_players, 2)

    logger.info("  ✓ Baller2Vec works")


def test_baller2vec_plus():
    """Test Baller2Vec++ model."""
    logger.info("Testing Baller2VecPlus...")

    model = Baller2VecPlus(
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_players=22
    )

    # Create dummy data
    batch_size = 2
    seq_len = 10
    num_players = 6

    positions = np.random.randn(seq_len, num_players, 2) * 10 + 50
    teams = np.array([0, 0, 0, 1, 1, 1])
    velocities = np.random.randn(seq_len, num_players, 2)

    features = create_feature_tensor(positions, teams, velocities)
    features = features.unsqueeze(0).expand(batch_size, -1, -1, -1)
    teams_tensor = create_team_tensor(np.tile(teams[np.newaxis, :], (batch_size, 1)))

    # Forward pass with uncertainty
    predictions, uncertainty = model(features, teams_tensor, return_uncertainty=True)
    logger.info(f"  Input shape: {features.shape}")
    logger.info(f"  Predictions shape: {predictions.shape}")
    logger.info(f"  Uncertainty shape: {uncertainty.shape}")

    # Predict future
    future, future_unc = model.predict_future(
        features, teams_tensor, n_future_steps=5, return_uncertainty=True
    )
    logger.info(f"  Future prediction shape: {future.shape}")
    logger.info(f"  Future uncertainty shape: {future_unc.shape}")

    logger.info("  ✓ Baller2VecPlus works")


def test_trajectory_predictor():
    """Test TrajectoryPredictor."""
    logger.info("Testing TrajectoryPredictor...")

    predictor = TrajectoryPredictor(
        model_type='baller2vec',
        model_path=None,
        history_length=10,
        device='cpu'
    )

    # Simulate player observations
    for t in range(15):
        timestamp = t * 0.04
        players = []

        for player_id in range(10):
            # Some players visible, some not
            is_visible = t < 10 or player_id < 5

            if is_visible:
                players.append(PlayerState(
                    player_id=player_id,
                    position=(40.0 + player_id * 5.0, 30.0 + t * 0.5),
                    velocity=(0.0, 0.5 / 0.04),
                    team=player_id % 2,
                    is_visible=True,
                    confidence=1.0
                ))

        predictor.update_history(players, timestamp)

    # Predict with some players off-screen
    result = predictor.predict(
        visible_players=[p for p in players if p.player_id < 5],
        timestamp=15 * 0.04,
        predict_all_players=True
    )

    logger.info(f"  Predicted {len(result.players)} players")
    logger.info(f"  Method: {result.method}")

    # Check confidence scores
    for player_id in range(10):
        conf = predictor.get_extrapolation_confidence(player_id)
        logger.debug(f"    Player {player_id}: confidence={conf:.3f}")

    logger.info("  ✓ TrajectoryPredictor works")


def test_integration():
    """Integration test with all components."""
    logger.info("Testing integration...")

    # Create predictor with Baller2Vec++
    predictor = TrajectoryPredictor(
        model_type='baller2vec_plus',
        model_path=None,
        history_length=25,
        use_physics_fallback=True,
        device='cpu'
    )

    # Simulate tracking scenario
    num_frames = 50
    num_players = 22

    for frame in range(num_frames):
        timestamp = frame * 0.04

        visible_players = []
        for player_id in range(num_players):
            # Simulate some players going off-screen
            is_visible = not (frame > 25 and player_id % 5 == 0)

            if is_visible or frame < 10:
                # Random positions on pitch
                x = 52.5 + np.random.randn() * 20
                y = 34.0 + np.random.randn() * 15
                vx, vy = np.random.randn(2) * 2

                visible_players.append(PlayerState(
                    player_id=player_id,
                    position=(x, y),
                    velocity=(vx, vy),
                    team=player_id % 2,
                    is_visible=is_visible,
                    confidence=1.0
                ))

        # Predict all players
        result = predictor.predict(
            visible_players=visible_players,
            timestamp=timestamp,
            predict_all_players=True
        )

        if frame % 10 == 0:
            visible_count = sum(1 for p in result.players if p.is_visible)
            extrapolated_count = len(result.players) - visible_count
            logger.debug(f"  Frame {frame}: {visible_count} visible, {extrapolated_count} extrapolated")

    logger.info("  ✓ Integration test passed")


def main():
    """Run all tests."""
    logger.info("=== Extrapolation Module Tests ===\n")

    try:
        test_kalman_motion_model()
        test_multi_player_motion_model()
        test_baller2vec()
        test_baller2vec_plus()
        test_trajectory_predictor()
        test_integration()

        logger.info("\n=== All Tests Passed ✓ ===")
        return 0

    except Exception as e:
        logger.error(f"\n=== Test Failed ✗ ===")
        logger.exception(e)
        return 1


if __name__ == '__main__':
    exit(main())
