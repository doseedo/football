"""Demo script for automatic homography estimation system.

This example demonstrates how to use the complete automatic calibration pipeline:
1. Field model with 57+ keypoints
2. HRNet-based keypoint detection
3. Automatic homography computation with RANSAC
4. Temporal Bayesian filtering for smooth tracking

Usage:
    python examples/automatic_calibration_demo.py --video path/to/video.mp4
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

from src.homography import (
    # Field model
    FootballPitchModel,
    create_standard_pitch,
    # Automatic keypoint detection
    PitchKeypointDetector,
    create_keypoint_detector,
    visualize_detections,
    # Automatic calibration
    AutoCalibrator,
    visualize_calibration,
    # Bayesian filtering
    BayesianHomographyFilter,
    create_bayesian_filter,
    # Integration with existing system
    DynamicCoordinateTransformer,
)


def demo_field_model():
    """Demonstrate field model with 57+ keypoints."""
    logger.info("=" * 60)
    logger.info("DEMO 1: Football Pitch Field Model")
    logger.info("=" * 60)

    # Create standard pitch model
    pitch = create_standard_pitch(include_3d=False)

    logger.info(f"Created pitch model: {pitch}")
    logger.info(f"Total keypoints: {len(pitch)}")

    # Show keypoints by category
    for category in ['corner', 'box', 'circle', 'goal', 'line']:
        kps = pitch.get_keypoints_by_category(category)
        logger.info(f"  {category}: {len(kps)} keypoints")

    # Show some example keypoints
    logger.info("\nExample keypoints:")
    example_names = ['corner_tl', 'penalty_spot_left', 'center_spot', 'goal_right_top']
    for name in example_names:
        coords = pitch.get_world_coords(name)
        if coords:
            logger.info(f"  {name}: {coords}")

    # Visualize (optional - requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 9))
        pitch.visualize_keypoints(ax)
        output_path = Path("outputs/pitch_keypoints.png")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved pitch visualization to {output_path}")
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")


def demo_keypoint_detector(video_path: str, max_frames: int = 10):
    """Demonstrate keypoint detection on video frames."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Automatic Keypoint Detection")
    logger.info("=" * 60)

    # Create pitch model
    pitch = create_standard_pitch()
    keypoint_names = pitch.get_keypoint_names()

    logger.info(f"Detecting {len(keypoint_names)} keypoints")

    # Create detector
    # NOTE: This will use HRNet backbone from timm
    # For production use, you would load pretrained weights trained on football data
    detector = create_keypoint_detector(
        num_keypoints=len(keypoint_names),
        device='cpu',  # Change to 'cuda' if GPU available
        checkpoint_path=None  # Set to path of pretrained weights
    )

    logger.info("Created keypoint detector")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    # Process frames
    frame_idx = 0
    output_dir = Path("outputs/keypoint_detections")
    output_dir.mkdir(exist_ok=True, parents=True)

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        logger.info(f"\nProcessing frame {frame_idx + 1}")

        # Detect keypoints
        result = detector.detect_keypoints(
            frame,
            keypoint_names=keypoint_names,
            min_confidence=0.3
        )

        logger.info(f"Detected {result.num_detected} keypoints above threshold")

        # Get high confidence keypoints
        high_conf_kps = result.get_high_confidence_keypoints(min_confidence=0.5)
        logger.info(f"High confidence (>0.5): {len(high_conf_kps)} keypoints")

        # Visualize
        vis_frame = visualize_detections(frame, result, min_confidence=0.3)

        # Save
        output_path = output_dir / f"frame_{frame_idx:04d}_detections.jpg"
        cv2.imwrite(str(output_path), vis_frame)
        logger.info(f"Saved to {output_path}")

        frame_idx += 1

    cap.release()
    logger.info(f"\nProcessed {frame_idx} frames")


def demo_automatic_calibration(video_path: str, max_frames: int = 10):
    """Demonstrate automatic calibration pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: Automatic Homography Calibration")
    logger.info("=" * 60)

    # Create pitch model
    pitch = create_standard_pitch()

    # Create automatic calibrator
    # NOTE: For production, provide pretrained detector checkpoint
    calibrator = AutoCalibrator(
        detector=None,  # Will create default detector
        pitch_model=pitch,
        min_keypoints=4,
        ransac_threshold=5.0,
        min_confidence=0.3,
        device='cpu'
    )

    logger.info("Created automatic calibrator")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    # Process frames
    frame_idx = 0
    output_dir = Path("outputs/auto_calibration")
    output_dir.mkdir(exist_ok=True, parents=True)

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        logger.info(f"\nProcessing frame {frame_idx + 1}")

        # Calibrate
        result = calibrator.calibrate_from_frame(frame, timestamp=frame_idx / 30.0)

        # Show results
        logger.info(f"Calibration valid: {result.is_valid}")
        logger.info(f"Method: {result.method}")
        logger.info(f"Quality score: {result.quality.quality_score:.3f}")
        logger.info(f"Reprojection error: {result.quality.reprojection_error:.2f}px")
        logger.info(f"Inliers: {result.quality.num_inliers}/{result.quality.num_total}")

        # Visualize
        vis_frame = visualize_calibration(frame, result, draw_grid=True)

        # Save
        output_path = output_dir / f"frame_{frame_idx:04d}_calibration.jpg"
        cv2.imwrite(str(output_path), vis_frame)
        logger.info(f"Saved to {output_path}")

        frame_idx += 1

    cap.release()
    logger.info(f"\nProcessed {frame_idx} frames")


def demo_bayesian_filtering(video_path: str, max_frames: int = 100):
    """Demonstrate full pipeline with Bayesian temporal filtering."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 4: Complete Pipeline with Bayesian Filtering")
    logger.info("=" * 60)

    # Create components
    pitch = create_standard_pitch()
    calibrator = AutoCalibrator(
        pitch_model=pitch,
        min_confidence=0.3,
        device='cpu'
    )
    bayesian_filter = create_bayesian_filter(strict=False)

    logger.info("Created complete pipeline")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    # Create coordinate transformer
    transformer = DynamicCoordinateTransformer()

    # Process frames
    frame_idx = 0
    output_dir = Path("outputs/bayesian_filtering")
    output_dir.mkdir(exist_ok=True, parents=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # Step 1: Auto-calibrate
        calib_result = calibrator.calibrate_from_frame(frame, timestamp=timestamp)

        # Step 2: Apply Bayesian filter
        filtered_H = bayesian_filter.process_frame(calib_result, timestamp=timestamp)

        # Step 3: Update transformer
        transformer.update_homography(filtered_H)

        # Show progress
        if frame_idx % 10 == 0:
            stats = bayesian_filter.get_statistics()
            logger.info(f"\nFrame {frame_idx}:")
            logger.info(f"  Raw quality: {calib_result.quality.quality_score:.3f}")
            logger.info(f"  Tracked keypoints: {stats['num_tracked_keypoints']}")
            logger.info(f"  Filter uncertainty: {stats['uncertainty']:.2f}")

            # Visualize with grid overlay
            vis_frame = frame.copy()

            # Draw pitch grid using filtered homography
            H_inv = np.linalg.inv(filtered_H)
            for x in np.arange(0, 105, 10):
                pts_world = np.array([[x, 0], [x, 68]], dtype=np.float32)
                pts_pixel = cv2.perspectiveTransform(pts_world.reshape(-1, 1, 2), H_inv)
                pt1 = tuple(pts_pixel[0, 0].astype(int))
                pt2 = tuple(pts_pixel[1, 0].astype(int))
                cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 2)

            for y in np.arange(0, 68, 10):
                pts_world = np.array([[0, y], [105, y]], dtype=np.float32)
                pts_pixel = cv2.perspectiveTransform(pts_world.reshape(-1, 1, 2), H_inv)
                pt1 = tuple(pts_pixel[0, 0].astype(int))
                pt2 = tuple(pts_pixel[1, 0].astype(int))
                cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 2)

            # Add info overlay
            cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Keypoints: {stats['num_tracked_keypoints']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save
            output_path = output_dir / f"frame_{frame_idx:04d}_filtered.jpg"
            cv2.imwrite(str(output_path), vis_frame)

        frame_idx += 1

    cap.release()

    # Final statistics
    final_stats = bayesian_filter.get_statistics()
    logger.info("\n" + "=" * 60)
    logger.info("Final Statistics:")
    logger.info(f"  Total frames: {final_stats['frame_count']}")
    logger.info(f"  Successful updates: {final_stats['total_updates']}")
    logger.info(f"  Tracked keypoints: {final_stats['num_tracked_keypoints']}")
    logger.info(f"  Final uncertainty: {final_stats['uncertainty']:.2f}")
    logger.info("=" * 60)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Automatic Homography Demo")
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['field', 'detector', 'calibration', 'filter', 'all'],
                       help='Which demo to run')
    parser.add_argument('--max-frames', type=int, default=10,
                       help='Maximum frames to process')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("AUTOMATIC HOMOGRAPHY ESTIMATION SYSTEM DEMO")
    logger.info("=" * 60)

    # Demo 1: Field model (no video needed)
    if args.demo in ['field', 'all']:
        demo_field_model()

    # Demos 2-4 require video
    if args.video and Path(args.video).exists():
        if args.demo in ['detector', 'all']:
            demo_keypoint_detector(args.video, max_frames=args.max_frames)

        if args.demo in ['calibration', 'all']:
            demo_automatic_calibration(args.video, max_frames=args.max_frames)

        if args.demo in ['filter', 'all']:
            demo_bayesian_filtering(args.video, max_frames=args.max_frames)
    else:
        if args.demo != 'field':
            logger.warning(f"Video not found: {args.video}")
            logger.info("Only running field model demo (no video needed)")


if __name__ == "__main__":
    main()
