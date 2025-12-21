#!/usr/bin/env python3
"""Example usage of the 3D ball tracking system."""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ball3d import Ball3DTracker
from src.homography.calibration import HomographyEstimator


def example_basic_usage():
    """Example: Basic 3D ball tracking."""
    print("=" * 60)
    print("Example 1: Basic 3D Ball Tracking")
    print("=" * 60)

    # Initialize 3D tracker
    tracker = Ball3DTracker(
        model_path='models/ball3d/best_model.pth',  # Path to trained model
        detection_confidence=0.2,
        temporal_window=10,
        sequence_length=20,
        device='cuda',
        fps=25.0
    )

    # Set camera calibration (homography from pitch calibration)
    # In practice, get this from InteractiveCalibrator or saved calibration
    homography = np.array([
        [1.2, 0.1, -500],
        [0.0, 1.5, -300],
        [0.0, 0.002, 1.0]
    ])

    tracker.set_calibration(
        homography=homography,
        camera_height=15.0,  # meters
        camera_angle=30.0    # degrees
    )

    # Process video
    video_path = 'data/input/match.mp4'
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    max_frames = 100  # Process first 100 frames for demo

    print(f"\nProcessing {max_frames} frames...")

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Track ball in 3D
        ball_state = tracker.track(frame)

        if ball_state:
            print(f"Frame {frame_count}:")
            print(f"  2D Position: ({ball_state.position_2d[0]:.1f}, {ball_state.position_2d[1]:.1f}) px")
            print(f"  3D Position: ({ball_state.position_3d.x:.2f}, {ball_state.position_3d.y:.2f}, {ball_state.position_3d.z:.2f}) m")
            print(f"  Height: {ball_state.position_3d.z:.2f} m")
            print(f"  Confidence: {ball_state.confidence_3d:.2f}")
            print(f"  Is Aerial: {tracker.is_ball_aerial()}")
            print()

        frame_count += 1

    cap.release()

    # Get statistics
    stats = tracker.get_statistics()
    print("\nTracking Statistics:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Aerial frames: {stats['aerial_frames']} ({stats['aerial_percentage']:.1f}%)")
    print(f"  Max height: {stats['max_height']:.2f} m")
    print(f"  Avg height: {stats['avg_height']:.2f} m")
    print(f"  Avg 3D confidence: {stats['avg_confidence_3d']:.2f}")

    # Export trajectory
    tracker.export_trajectory_3d('output/ball_trajectory_3d.csv')
    print("\nTrajectory exported to output/ball_trajectory_3d.csv")


def example_synthetic_data():
    """Example: Generate and visualize synthetic data."""
    print("\n" + "=" * 60)
    print("Example 2: Synthetic Data Generation")
    print("=" * 60)

    from src.ball3d import SyntheticDataGenerator

    generator = SyntheticDataGenerator(
        pitch_length=105.0,
        pitch_width=68.0,
        fps=25.0,
        noise_level=0.02
    )

    # Generate different trajectory types
    trajectory_types = ['pass', 'shot', 'cross', 'bounce', 'aerial']

    for traj_type in trajectory_types:
        print(f"\nGenerating {traj_type} trajectory...")

        traj = generator.generate_trajectory(trajectory_type=traj_type)

        print(f"  Duration: {traj.timestamps[-1]:.2f} s")
        print(f"  Num frames: {len(traj.positions_3d)}")
        print(f"  Max height: {np.max(traj.positions_3d[:, 2]):.2f} m")
        print(f"  Visible frames: {np.sum(traj.is_visible)}/{len(traj.is_visible)}")
        print(f"  Camera height: {traj.camera_params.position[2]:.1f} m")

    # Generate batch
    print("\nGenerating batch of 100 trajectories...")
    batch = generator.generate_batch(100)
    print(f"Generated {len(batch)} trajectories")

    # Save dataset
    output_path = 'data/ball3d/demo_trajectories.npz'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    generator.save_dataset(output_path, num_samples=1000)
    print(f"\nDataset saved to {output_path}")


def example_physics_model():
    """Example: Physics-based ball motion."""
    print("\n" + "=" * 60)
    print("Example 3: Physics Model")
    print("=" * 60)

    from src.ball3d import BallPhysicsModel, Ball3DPosition, PhysicsConstraints

    # Initialize physics model
    physics = BallPhysicsModel(fps=25.0)

    # Create a trajectory (e.g., a shot)
    print("\nSimulating a shot trajectory...")

    trajectory = []
    x0, y0, z0 = 50.0, 34.0, 0.11  # Start position
    vx, vy, vz = 15.0, 0.0, 12.0   # Initial velocity

    for i in range(50):  # 2 seconds at 25 fps
        t = i / 25.0

        # Ballistic motion
        x = x0 + vx * t
        y = y0 + vy * t
        z = max(0.11, z0 + vz * t - 0.5 * 9.81 * t**2)

        pos = Ball3DPosition(
            x=x, y=y, z=z,
            timestamp=t,
            confidence=1.0,
            velocity=(vx, vy, vz - 9.81 * t)
        )

        trajectory.append(pos)

    # Apply physics constraints
    constrained = physics.apply_physics_constraints(trajectory, smooth=True)

    print(f"Original trajectory points: {len(trajectory)}")
    print(f"Constrained trajectory points: {len(constrained)}")

    # Detect bounces
    bounces = physics.detect_bounce(constrained)
    print(f"\nBounces detected at frames: {bounces}")

    # Show some positions
    print("\nTrajectory samples:")
    for i in [0, 10, 20, 30, 40, 49]:
        pos = constrained[i]
        print(f"  t={pos.timestamp:.2f}s: x={pos.x:.1f}, y={pos.y:.1f}, z={pos.z:.2f} m")

    # Predict next position
    print("\nPredicting future position...")
    current = constrained[-1]
    predicted = physics.predict_next_position(current, steps=5)
    print(f"Current: ({current.x:.1f}, {current.y:.1f}, {current.z:.2f}) m")
    print(f"Predicted (+5 frames): ({predicted.x:.1f}, {predicted.y:.1f}, {predicted.z:.2f}) m")


def example_training():
    """Example: Training the LSTM model."""
    print("\n" + "=" * 60)
    print("Example 4: Model Training")
    print("=" * 60)

    print("\nTo train the 3D ball trajectory model:")
    print("\n1. Generate synthetic training data:")
    print("   python scripts/train_ball3d.py --mode generate --num-samples 10000")

    print("\n2. Train the model:")
    print("   python scripts/train_ball3d.py --mode train --epochs 50 --batch-size 32")

    print("\n3. Or do both in one command:")
    print("   python scripts/train_ball3d.py --mode both --num-samples 10000 --epochs 50")

    print("\nTraining data requirements:")
    print("  - Minimum: 1,000 trajectories (~5 min training)")
    print("  - Recommended: 10,000 trajectories (~30 min training)")
    print("  - Optimal: 50,000+ trajectories (~2 hr training)")

    print("\nHardware requirements:")
    print("  - GPU: 4GB+ VRAM (8GB+ recommended)")
    print("  - RAM: 16GB+ system memory")
    print("  - Storage: 500MB for 10k trajectories")


def example_integration():
    """Example: Integration with main tracking pipeline."""
    print("\n" + "=" * 60)
    print("Example 5: Integration with Main Pipeline")
    print("=" * 60)

    print("\nTo integrate 3D ball tracking into the main pipeline:")

    print("\n1. Import the Ball3DTracker:")
    print("""
    from src.ball3d import Ball3DTracker

    # In FootballTracker.__init__:
    self.ball_tracker_3d = Ball3DTracker(
        model_path='models/ball3d/best_model.pth',
        device=self.device,
        fps=self.fps
    )
    """)

    print("\n2. Set calibration after homography estimation:")
    print("""
    # After calibration:
    self.ball_tracker_3d.set_calibration(
        homography=self.transformer.H,
        camera_height=15.0,
        camera_angle=30.0
    )
    """)

    print("\n3. Use in frame processing:")
    print("""
    # In _process_frame:
    ball_state_3d = self.ball_tracker_3d.track(frame)

    if ball_state_3d:
        ball_record = {
            'x': ball_state_3d.position_3d.x,
            'y': ball_state_3d.position_3d.y,
            'z': ball_state_3d.position_3d.z,  # Now includes height!
            'height_confidence': ball_state_3d.confidence_3d,
            'is_aerial': ball_state_3d.position_3d.z > 0.5
        }
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("3D Ball Trajectory Estimation - Examples")
    print("=" * 60)

    examples = [
        # ("Basic 3D Tracking", example_basic_usage),
        ("Synthetic Data", example_synthetic_data),
        ("Physics Model", example_physics_model),
        ("Training", example_training),
        ("Integration", example_integration)
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\nExample '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
