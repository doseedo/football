"""Unit tests for DETR-based detection system.

Tests cover:
- DETR detector initialization and inference
- Transformer prediction heads
- Hybrid detector with fusion
- Weighted Box Fusion algorithm
- Scene complexity analysis
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from src.detection import (
    DETRDetector,
    DETR,
    HungarianMatcher,
    TransformerPredictionHead,
    MultiScaleTransformerHead,
    HybridDetector,
    WeightedBoxFusion,
    SceneComplexity,
    Detection
)


class TestDETRModel:
    """Test DETR model architecture."""

    def test_detr_initialization(self):
        """Test DETR model initializes correctly."""
        model = DETR(num_classes=1, num_queries=100, hidden_dim=256)

        assert model.num_queries == 100
        assert model.hidden_dim == 256
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'transformer_encoder')
        assert hasattr(model, 'transformer_decoder')

    def test_detr_forward(self):
        """Test DETR forward pass."""
        model = DETR(num_classes=1, num_queries=100, hidden_dim=256)
        model.eval()

        # Create dummy input
        x = torch.randn(2, 3, 800, 800)

        with torch.no_grad():
            pred_logits, pred_boxes = model(x)

        # Check output shapes
        assert pred_logits.shape == (2, 100, 2)  # [B, num_queries, num_classes + 1]
        assert pred_boxes.shape == (2, 100, 4)  # [B, num_queries, 4]

        # Check boxes are in [0, 1] range
        assert (pred_boxes >= 0).all() and (pred_boxes <= 1).all()


class TestDETRDetector:
    """Test DETR detector wrapper."""

    def test_detector_initialization(self):
        """Test detector initializes without pretrained weights."""
        detector = DETRDetector(
            model_path=None,
            device="cpu",
            num_queries=50,
            confidence=0.7
        )

        assert detector.num_queries == 50
        assert detector.confidence == 0.7
        assert detector.device == torch.device("cpu")

    def test_detect_single_frame(self):
        """Test detection on single frame."""
        detector = DETRDetector(
            model_path=None,
            device="cpu",
            num_queries=50,
            confidence=0.9  # High threshold to avoid random detections
        )

        # Create dummy frame
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Detect
        detections = detector.detect(frame)

        # Check output type
        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)
            assert det.bbox.shape == (4,)
            assert 0 <= det.confidence <= 1
            assert det.class_id == 0

    def test_detect_batch(self):
        """Test batch detection."""
        detector = DETRDetector(
            model_path=None,
            device="cpu",
            num_queries=50
        )

        # Create batch of frames
        frames = [
            np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
            for _ in range(4)
        ]

        # Detect
        all_detections = detector.detect_batch(frames)

        assert len(all_detections) == 4
        for detections in all_detections:
            assert isinstance(detections, list)

    def test_pitch_mask_filtering(self):
        """Test detection filtering with pitch mask."""
        detector = DETRDetector(
            model_path=None,
            device="cpu",
            confidence=0.5
        )

        frame = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)

        # Create pitch mask (only allow detections in center)
        pitch_mask = np.zeros((540, 960), dtype=np.uint8)
        pitch_mask[200:340, 300:660] = 1

        # Detect with mask
        detections = detector.detect(frame, pitch_mask)

        # All detections should be within masked region
        for det in detections:
            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = det.bbox[3]
            assert pitch_mask[int(cy), int(cx)] == 1


class TestTransformerPredictionHead:
    """Test transformer prediction head."""

    def test_head_initialization(self):
        """Test head initializes correctly."""
        head = TransformerPredictionHead(
            in_channels=256,
            hidden_dim=256,
            num_classes=1,
            num_transformer_layers=2,
            num_heads=8
        )

        assert head.hidden_dim == 256
        assert head.num_classes == 1
        assert len(head.transformer_layers) == 2

    def test_head_forward(self):
        """Test head forward pass."""
        head = TransformerPredictionHead(
            in_channels=256,
            hidden_dim=256,
            num_classes=1,
            num_transformer_layers=2,
            num_anchors=3
        )

        # Create dummy feature map
        x = torch.randn(2, 256, 20, 20)

        # Forward pass
        pred_classes, pred_boxes, pred_objectness = head(x)

        # Check output shapes
        assert pred_classes.shape == (2, 3, 20, 20, 1)  # [B, anchors, H, W, classes]
        assert pred_boxes.shape == (2, 3, 20, 20, 4)  # [B, anchors, H, W, 4]
        assert pred_objectness.shape == (2, 3, 20, 20)  # [B, anchors, H, W]

    def test_multiscale_head(self):
        """Test multi-scale transformer head."""
        head = MultiScaleTransformerHead(
            in_channels_list=[256, 512, 1024],
            hidden_dim=256,
            num_classes=1
        )

        # Create multi-scale features
        features = [
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
            torch.randn(2, 1024, 10, 10)
        ]

        # Forward pass
        predictions = head(features)

        assert len(predictions) == 3
        for pred in predictions:
            pred_classes, pred_boxes, pred_objectness = pred
            assert pred_classes.dim() == 5
            assert pred_boxes.dim() == 5
            assert pred_objectness.dim() == 4


class TestHungarianMatcher:
    """Test Hungarian matching algorithm."""

    def test_matcher_initialization(self):
        """Test matcher initializes with correct costs."""
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=5.0,
            cost_giou=2.0
        )

        assert matcher.cost_class == 1.0
        assert matcher.cost_bbox == 5.0
        assert matcher.cost_giou == 2.0

    def test_box_conversion(self):
        """Test box format conversion."""
        matcher = HungarianMatcher()

        # Test boxes in (cx, cy, w, h) format
        boxes_cxcywh = torch.tensor([
            [0.5, 0.5, 0.2, 0.3],  # Center at (0.5, 0.5), size 0.2x0.3
            [0.3, 0.4, 0.1, 0.2]
        ])

        boxes_xyxy = matcher.box_cxcywh_to_xyxy(boxes_cxcywh)

        # First box should be (0.4, 0.35, 0.6, 0.65)
        expected = torch.tensor([
            [0.4, 0.35, 0.6, 0.65],
            [0.25, 0.3, 0.35, 0.5]
        ])

        assert torch.allclose(boxes_xyxy, expected, atol=1e-6)


class TestWeightedBoxFusion:
    """Test Weighted Box Fusion algorithm."""

    def test_wbf_initialization(self):
        """Test WBF initializes correctly."""
        wbf = WeightedBoxFusion(iou_threshold=0.5, skip_box_threshold=0.1)

        assert wbf.iou_threshold == 0.5
        assert wbf.skip_box_threshold == 0.1

    def test_wbf_empty_input(self):
        """Test WBF with empty input."""
        wbf = WeightedBoxFusion()

        fused_boxes, fused_scores = wbf(
            boxes_list=[np.array([]).reshape(0, 4)],
            scores_list=[np.array([])]
        )

        assert len(fused_boxes) == 0
        assert len(fused_scores) == 0

    def test_wbf_single_detector(self):
        """Test WBF with single detector."""
        wbf = WeightedBoxFusion(iou_threshold=0.5)

        boxes = np.array([
            [100, 100, 200, 300],
            [400, 200, 500, 400]
        ])
        scores = np.array([0.9, 0.8])

        fused_boxes, fused_scores = wbf(
            boxes_list=[boxes],
            scores_list=[scores]
        )

        # Should preserve all boxes
        assert len(fused_boxes) == 2
        assert len(fused_scores) == 2

    def test_wbf_overlapping_boxes(self):
        """Test WBF merges overlapping boxes."""
        wbf = WeightedBoxFusion(iou_threshold=0.5)

        # Two very similar boxes
        boxes1 = np.array([[100, 100, 200, 300]])
        boxes2 = np.array([[105, 105, 205, 305]])  # Slightly offset

        scores1 = np.array([0.9])
        scores2 = np.array([0.85])

        fused_boxes, fused_scores = wbf(
            boxes_list=[boxes1, boxes2],
            scores_list=[scores1, scores2],
            weights=[0.5, 0.5]
        )

        # Should merge into single box
        assert len(fused_boxes) == 1
        assert len(fused_scores) == 1

        # Fused box should be average
        expected_box = (boxes1[0] * 0.9 * 0.5 + boxes2[0] * 0.85 * 0.5) / (0.9 * 0.5 + 0.85 * 0.5)
        assert np.allclose(fused_boxes[0], expected_box, rtol=0.1)

    def test_iou_computation(self):
        """Test IoU computation."""
        wbf = WeightedBoxFusion()

        box = np.array([0, 0, 10, 10])
        boxes = np.array([
            [0, 0, 10, 10],  # Perfect overlap
            [5, 5, 15, 15],  # Partial overlap
            [20, 20, 30, 30]  # No overlap
        ])

        ious = wbf.compute_iou(box, boxes)

        assert ious[0] == 1.0  # Perfect overlap
        assert 0 < ious[1] < 1  # Partial overlap
        assert ious[2] == 0  # No overlap


class TestHybridDetector:
    """Test hybrid YOLO + DETR detector."""

    def test_hybrid_initialization(self):
        """Test hybrid detector initializes."""
        yolo_mock = Mock()
        detr_mock = Mock()

        hybrid = HybridDetector(
            yolo_detector=yolo_mock,
            detr_detector=detr_mock,
            fusion_strategy='adaptive',
            crowd_threshold=0.3
        )

        assert hybrid.use_detr is True
        assert hybrid.fusion_strategy == 'adaptive'
        assert hybrid.crowd_threshold == 0.3

    def test_hybrid_without_detr(self):
        """Test hybrid detector without DETR (YOLO only)."""
        yolo_mock = Mock()
        yolo_mock.detect.return_value = []

        hybrid = HybridDetector(
            yolo_detector=yolo_mock,
            detr_detector=None
        )

        assert hybrid.use_detr is False

        # Should only use YOLO
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = hybrid.detect(frame)

        yolo_mock.detect.assert_called_once()

    def test_scene_complexity_empty(self):
        """Test scene complexity with no detections."""
        yolo_mock = Mock()
        hybrid = HybridDetector(yolo_mock, None)

        complexity = hybrid.analyze_scene_complexity([], (1080, 1920))

        assert complexity.overlap_ratio == 0.0
        assert complexity.detection_density == 0.0
        assert complexity.total_detections == 0
        assert complexity.is_crowded is False

    def test_scene_complexity_crowded(self):
        """Test scene complexity with crowded scene."""
        yolo_mock = Mock()
        hybrid = HybridDetector(yolo_mock, None, crowd_threshold=0.3)

        # Create overlapping detections
        detections = [
            Detection(bbox=np.array([100, 100, 200, 300]), confidence=0.9, class_id=0),
            Detection(bbox=np.array([150, 150, 250, 350]), confidence=0.85, class_id=0),
            Detection(bbox=np.array([140, 140, 240, 340]), confidence=0.8, class_id=0),
        ]

        complexity = hybrid.analyze_scene_complexity(detections, (1080, 1920))

        # Should detect crowding due to overlaps
        assert complexity.overlap_ratio > 0
        assert complexity.total_detections == 3

    def test_adaptive_weights_clear_scene(self):
        """Test adaptive weights for clear scene."""
        yolo_mock = Mock()
        hybrid = HybridDetector(yolo_mock, None)

        complexity = SceneComplexity(
            overlap_ratio=0.1,
            detection_density=0.5,
            total_detections=5,
            is_crowded=False
        )

        yolo_weight, detr_weight = hybrid.compute_adaptive_weights(complexity)

        # Clear scene should prefer YOLO
        assert yolo_weight > detr_weight
        assert abs(yolo_weight + detr_weight - 1.0) < 1e-6  # Normalized

    def test_adaptive_weights_crowded_scene(self):
        """Test adaptive weights for crowded scene."""
        yolo_mock = Mock()
        hybrid = HybridDetector(yolo_mock, None, crowd_threshold=0.3)

        complexity = SceneComplexity(
            overlap_ratio=0.6,
            detection_density=2.5,
            total_detections=15,
            is_crowded=True
        )

        yolo_weight, detr_weight = hybrid.compute_adaptive_weights(complexity)

        # Crowded scene should prefer DETR
        assert detr_weight > yolo_weight
        assert abs(yolo_weight + detr_weight - 1.0) < 1e-6

    def test_set_fusion_weights(self):
        """Test manual weight setting."""
        yolo_mock = Mock()
        hybrid = HybridDetector(yolo_mock, None)

        hybrid.set_fusion_weights(0.4, 0.6)

        assert hybrid.yolo_weight == 0.4
        assert hybrid.detr_weight == 0.6


class TestIntegration:
    """Integration tests for complete detection pipeline."""

    def test_end_to_end_detection(self):
        """Test end-to-end detection with hybrid detector."""
        # Create mock detectors
        yolo_mock = Mock()
        yolo_mock.detect.return_value = [
            Detection(bbox=np.array([100, 100, 200, 300]), confidence=0.9, class_id=0)
        ]

        detr_mock = Mock()
        detr_mock.detect.return_value = [
            Detection(bbox=np.array([105, 105, 205, 305]), confidence=0.95, class_id=0)
        ]

        # Create hybrid detector
        hybrid = HybridDetector(
            yolo_detector=yolo_mock,
            detr_detector=detr_mock,
            fusion_strategy='weighted',
            default_yolo_weight=0.6,
            default_detr_weight=0.4
        )

        # Run detection
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = hybrid.detect(frame)

        # Both detectors should be called
        yolo_mock.detect.assert_called_once()
        detr_mock.detect.assert_called_once()

        # Should have fused detections
        assert isinstance(detections, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
