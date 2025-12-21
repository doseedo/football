"""Example usage of DETR-based detection system for football analysis.

This script demonstrates:
1. Using DETR detector for crowded scenes
2. Using transformer prediction heads
3. Using hybrid YOLO + DETR ensemble
4. Adaptive weighting based on scene complexity
"""

import cv2
import numpy as np
from pathlib import Path

from src.detection import (
    PlayerDetector,
    DETRDetector,
    HybridDetector,
    TransformerPredictionHead,
    WeightedBoxFusion,
    EnsembleConfig
)


def example_1_basic_detr():
    """Example 1: Basic DETR detection for crowded scenes."""
    print("\n" + "="*60)
    print("Example 1: Basic DETR Detection")
    print("="*60)

    # Initialize DETR detector
    detr = DETRDetector(
        model_path=None,  # Use pretrained backbone only
        device="cuda",
        num_queries=100,
        confidence=0.7,
        min_height=30,
        max_height=400
    )

    # Load test frame
    frame = cv2.imread("test_frame.jpg")
    if frame is None:
        print("No test frame found, using dummy data")
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Detect players
    detections = detr.detect(frame)

    print(f"DETR detected {len(detections)} players")
    for i, det in enumerate(detections[:5]):  # Show first 5
        print(f"  Player {i+1}: bbox={det.bbox}, confidence={det.confidence:.3f}")

    # Key advantage: No NMS needed, better for overlapping players
    print("\nKey advantages of DETR:")
    print("  - Set-based prediction (no NMS needed)")
    print("  - Better handling of overlapping players")
    print("  - Global reasoning via transformer attention")


def example_2_hybrid_detector():
    """Example 2: Hybrid YOLO + DETR ensemble."""
    print("\n" + "="*60)
    print("Example 2: Hybrid YOLO + DETR Detector")
    print("="*60)

    # Initialize YOLO detector
    yolo = PlayerDetector(
        model_path="yolov8x.pt",
        confidence=0.3,
        device="cuda"
    )

    # Initialize DETR detector
    detr = DETRDetector(
        model_path=None,
        device="cuda",
        confidence=0.7
    )

    # Create hybrid detector with adaptive weighting
    hybrid = HybridDetector(
        yolo_detector=yolo,
        detr_detector=detr,
        fusion_strategy='adaptive',  # Automatically adapt to scene complexity
        crowd_threshold=0.3
    )

    # Load test frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Detect with hybrid approach
    detections = hybrid.detect(frame)

    print(f"Hybrid detector found {len(detections)} players")
    print("\nFusion strategy: adaptive")
    print("  - Clear scenes: YOLO weighted higher (faster)")
    print("  - Crowded scenes: DETR weighted higher (more accurate)")


def example_3_adaptive_weighting():
    """Example 3: Adaptive weighting based on scene complexity."""
    print("\n" + "="*60)
    print("Example 3: Adaptive Weighting")
    print("="*60)

    # Initialize detectors
    yolo = PlayerDetector(model_path="yolov8x.pt", device="cuda")
    detr = DETRDetector(device="cuda")

    # Create hybrid with adaptive weighting
    hybrid = HybridDetector(
        yolo_detector=yolo,
        detr_detector=detr,
        fusion_strategy='adaptive'
    )

    # Test on different scenes
    scenes = [
        ("clear_scene.jpg", "Clear scene with well-separated players"),
        ("crowded_scene.jpg", "Crowded penalty box scene"),
        ("corner_kick.jpg", "Corner kick with many overlapping players")
    ]

    for scene_path, description in scenes:
        print(f"\n{description}:")

        # Load or create dummy frame
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Detect
        detections = hybrid.detect(frame)

        # The hybrid detector automatically adjusts weights based on complexity
        print(f"  Detected: {len(detections)} players")


def example_4_weighted_box_fusion():
    """Example 4: Manual weighted box fusion."""
    print("\n" + "="*60)
    print("Example 4: Weighted Box Fusion (WBF)")
    print("="*60)

    # Create WBF instance
    wbf = WeightedBoxFusion(iou_threshold=0.5, skip_box_threshold=0.0)

    # Simulate detections from two models
    # YOLO detections
    yolo_boxes = np.array([
        [100, 100, 200, 300],
        [105, 105, 205, 305],  # Overlaps with first
        [400, 200, 500, 400]
    ])
    yolo_scores = np.array([0.9, 0.85, 0.8])

    # DETR detections
    detr_boxes = np.array([
        [102, 102, 202, 302],  # Overlaps with YOLO boxes
        [410, 210, 510, 410]   # Overlaps with third YOLO box
    ])
    detr_scores = np.array([0.95, 0.88])

    # Fuse with WBF
    fused_boxes, fused_scores = wbf(
        boxes_list=[yolo_boxes, detr_boxes],
        scores_list=[yolo_scores, detr_scores],
        weights=[0.6, 0.4]  # Weight YOLO slightly higher
    )

    print(f"YOLO detections: {len(yolo_boxes)}")
    print(f"DETR detections: {len(detr_boxes)}")
    print(f"Fused detections: {len(fused_boxes)}")
    print("\nFused boxes and scores:")
    for i, (box, score) in enumerate(zip(fused_boxes, fused_scores)):
        print(f"  Box {i+1}: {box} (score: {score:.3f})")

    print("\nWBF advantages:")
    print("  - Averages overlapping boxes weighted by confidence")
    print("  - Better than NMS for ensemble predictions")
    print("  - Preserves information from all models")


def example_5_batch_processing():
    """Example 5: Batch processing with DETR and hybrid detector."""
    print("\n" + "="*60)
    print("Example 5: Batch Processing")
    print("="*60)

    # Initialize hybrid detector
    yolo = PlayerDetector(model_path="yolov8x.pt", device="cuda")
    detr = DETRDetector(device="cuda")
    hybrid = HybridDetector(yolo, detr, fusion_strategy='adaptive')

    # Create batch of frames
    batch_size = 8
    frames = [
        np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        for _ in range(batch_size)
    ]

    print(f"Processing batch of {batch_size} frames...")

    # Process batch (more efficient than frame-by-frame)
    all_detections = hybrid.detect_batch(frames)

    print(f"Results:")
    for i, detections in enumerate(all_detections):
        print(f"  Frame {i+1}: {len(detections)} players")

    print("\nBatch processing advantages:")
    print("  - GPU utilization optimized")
    print("  - Faster than sequential processing")
    print("  - Efficient memory usage")


def example_6_custom_weights():
    """Example 6: Manual weight configuration."""
    print("\n" + "="*60)
    print("Example 6: Manual Weight Configuration")
    print("="*60)

    # Initialize detectors
    yolo = PlayerDetector(model_path="yolov8x.pt", device="cuda")
    detr = DETRDetector(device="cuda")

    # Create hybrid with fixed weights
    hybrid = HybridDetector(
        yolo_detector=yolo,
        detr_detector=detr,
        fusion_strategy='weighted',  # Use fixed weights
        default_yolo_weight=0.8,
        default_detr_weight=0.2
    )

    print("Initial weights: YOLO=0.8, DETR=0.2")

    # Change weights dynamically
    hybrid.set_fusion_weights(yolo_weight=0.5, detr_weight=0.5)
    print("Updated weights: YOLO=0.5, DETR=0.5")

    # Use case: Adjust based on known game situation
    print("\nUse cases for manual weights:")
    print("  - Open play: Weight YOLO higher (faster)")
    print("  - Set pieces: Weight DETR higher (crowded)")
    print("  - Long shots: Weight YOLO higher (few players)")


def example_7_configuration():
    """Example 7: Ensemble configuration."""
    print("\n" + "="*60)
    print("Example 7: Ensemble Configuration")
    print("="*60)

    # Create configuration
    config = EnsembleConfig(
        use_yolo=True,
        use_detr=True,
        fusion_strategy='adaptive',
        crowd_threshold=0.3,
        yolo_weight=0.7,
        detr_weight=0.3
    )

    print("Ensemble configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    print("\nConfiguration parameters:")
    print("  - use_yolo: Enable/disable YOLO detector")
    print("  - use_detr: Enable/disable DETR detector")
    print("  - fusion_strategy: 'adaptive' or 'weighted'")
    print("  - crowd_threshold: Overlap ratio for crowded detection")
    print("  - yolo_weight/detr_weight: Default fusion weights")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("DETR-based Detection System Examples")
    print("="*80)

    examples = [
        ("Basic DETR Detection", example_1_basic_detr),
        ("Hybrid YOLO + DETR", example_2_hybrid_detector),
        ("Adaptive Weighting", example_3_adaptive_weighting),
        ("Weighted Box Fusion", example_4_weighted_box_fusion),
        ("Batch Processing", example_5_batch_processing),
        ("Manual Weights", example_6_custom_weights),
        ("Configuration", example_7_configuration)
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n{name} failed: {e}")
            print("(This is expected if models/dependencies are not available)")

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
