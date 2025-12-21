"""Hybrid detector combining YOLO and DETR for optimal performance.

The hybrid detector leverages the strengths of both approaches:
- YOLO: Fast, efficient for clear scenes
- DETR: Better for crowded scenes with overlapping players

Uses adaptive weighting based on scene complexity to get the best of both worlds.
"""

from typing import List, Optional, Tuple
import numpy as np
from loguru import logger
from dataclasses import dataclass

from .player_detector import Detection, PlayerDetector
from .detr_detector import DETRDetector


@dataclass
class SceneComplexity:
    """Scene complexity metrics."""
    overlap_ratio: float  # Ratio of overlapping detections
    detection_density: float  # Detections per unit area
    total_detections: int
    is_crowded: bool


class WeightedBoxFusion:
    """Weighted Box Fusion (WBF) for merging predictions from multiple detectors.

    WBF is superior to NMS for ensemble predictions as it:
    - Averages overlapping boxes weighted by confidence
    - Preserves information from all models
    - Handles soft consensus between detectors
    """

    def __init__(self, iou_threshold: float = 0.5, skip_box_threshold: float = 0.0):
        """
        Initialize WBF.

        Args:
            iou_threshold: IoU threshold for considering boxes as duplicates
            skip_box_threshold: Minimum confidence to consider a box
        """
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold

    def __call__(
        self,
        boxes_list: List[np.ndarray],
        scores_list: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse boxes from multiple detectors.

        Args:
            boxes_list: List of [N, 4] arrays with boxes from each detector
            scores_list: List of [N] arrays with scores from each detector
            weights: Optional weights for each detector

        Returns:
            fused_boxes: [M, 4] fused boxes
            fused_scores: [M] fused scores
        """
        if weights is None:
            weights = [1.0] * len(boxes_list)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Collect all boxes with their weights
        all_boxes = []
        all_scores = []
        all_weights = []

        for boxes, scores, weight in zip(boxes_list, scores_list, weights):
            if len(boxes) > 0:
                for box, score in zip(boxes, scores):
                    if score >= self.skip_box_threshold:
                        all_boxes.append(box)
                        all_scores.append(score * weight)
                        all_weights.append(weight)

        if len(all_boxes) == 0:
            return np.array([]), np.array([])

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_weights = np.array(all_weights)

        # Cluster boxes by IoU
        fused_boxes = []
        fused_scores = []

        while len(all_boxes) > 0:
            # Find box with highest score
            idx = np.argmax(all_scores)
            best_box = all_boxes[idx]
            best_score = all_scores[idx]

            # Find all overlapping boxes
            ious = self.compute_iou(best_box, all_boxes)
            cluster_mask = ious >= self.iou_threshold

            # Average boxes in cluster weighted by score
            cluster_boxes = all_boxes[cluster_mask]
            cluster_scores = all_scores[cluster_mask]
            cluster_weights = all_weights[cluster_mask]

            # Weighted average
            weight_sum = cluster_scores.sum()
            if weight_sum > 0:
                fused_box = (cluster_boxes * cluster_scores[:, None]).sum(axis=0) / weight_sum
                fused_score = cluster_scores.sum() / cluster_weights.sum()
            else:
                fused_box = best_box
                fused_score = best_score

            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)

            # Remove clustered boxes
            all_boxes = all_boxes[~cluster_mask]
            all_scores = all_scores[~cluster_mask]
            all_weights = all_weights[~cluster_mask]

        return np.array(fused_boxes), np.array(fused_scores)

    @staticmethod
    def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU between a box and multiple boxes.

        Args:
            box: [4] single box (x1, y1, x2, y2)
            boxes: [N, 4] multiple boxes

        Returns:
            [N] IoU values
        """
        # Compute intersection
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Compute areas
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Compute IoU
        union = box_area + boxes_area - intersection
        iou = intersection / (union + 1e-6)

        return iou


class HybridDetector:
    """Hybrid detector combining YOLO and DETR.

    Strategy:
    1. Run both detectors in parallel
    2. Analyze scene complexity
    3. Adaptively weight predictions based on complexity
    4. Fuse predictions using Weighted Box Fusion

    Benefits:
    - YOLO provides fast, accurate detections in clear scenes
    - DETR handles crowded scenes with overlapping players
    - Adaptive weighting gets best of both worlds
    - WBF combines predictions intelligently
    """

    def __init__(
        self,
        yolo_detector: PlayerDetector,
        detr_detector: Optional[DETRDetector] = None,
        fusion_strategy: str = 'weighted',
        crowd_threshold: float = 0.3,
        default_yolo_weight: float = 0.7,
        default_detr_weight: float = 0.3
    ):
        """
        Initialize hybrid detector.

        Args:
            yolo_detector: YOLO detector instance
            detr_detector: DETR detector instance (optional)
            fusion_strategy: 'weighted' or 'adaptive'
            crowd_threshold: Overlap ratio threshold for crowded scene
            default_yolo_weight: Default weight for YOLO (when not crowded)
            default_detr_weight: Default weight for DETR (when not crowded)
        """
        self.yolo_detector = yolo_detector
        self.detr_detector = detr_detector
        self.fusion_strategy = fusion_strategy
        self.crowd_threshold = crowd_threshold

        self.yolo_weight = default_yolo_weight
        self.detr_weight = default_detr_weight

        self.wbf = WeightedBoxFusion(iou_threshold=0.5, skip_box_threshold=0.0)

        if detr_detector is None:
            logger.warning("DETR detector not provided, will use YOLO only")
            self.use_detr = False
        else:
            self.use_detr = True
            logger.info(f"Hybrid detector initialized with fusion strategy: {fusion_strategy}")

    def set_fusion_weights(self, yolo_weight: float, detr_weight: float):
        """
        Manually set fusion weights.

        Args:
            yolo_weight: Weight for YOLO predictions
            detr_weight: Weight for DETR predictions
        """
        self.yolo_weight = yolo_weight
        self.detr_weight = detr_weight
        logger.info(f"Fusion weights set: YOLO={yolo_weight:.2f}, DETR={detr_weight:.2f}")

    def analyze_scene_complexity(self, detections: List[Detection], frame_shape: Tuple[int, int]) -> SceneComplexity:
        """
        Analyze scene complexity to determine adaptive weights.

        Args:
            detections: Initial detections (from YOLO)
            frame_shape: (height, width) of frame

        Returns:
            SceneComplexity metrics
        """
        if len(detections) == 0:
            return SceneComplexity(
                overlap_ratio=0.0,
                detection_density=0.0,
                total_detections=0,
                is_crowded=False
            )

        # Compute pairwise IoU
        boxes = np.array([det.bbox for det in detections])
        n = len(boxes)

        overlap_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._compute_iou_single(boxes[i], boxes[j])
                if iou > 0.1:  # Any overlap
                    overlap_count += 1

        # Overlap ratio
        max_pairs = n * (n - 1) / 2 if n > 1 else 1
        overlap_ratio = overlap_count / max_pairs if max_pairs > 0 else 0

        # Detection density
        h, w = frame_shape
        frame_area = h * w
        detection_density = len(detections) / (frame_area / 10000)  # Per 100x100 pixels

        # Crowded scene detection
        is_crowded = overlap_ratio > self.crowd_threshold or detection_density > 2.0

        return SceneComplexity(
            overlap_ratio=overlap_ratio,
            detection_density=detection_density,
            total_detections=len(detections),
            is_crowded=is_crowded
        )

    def compute_adaptive_weights(self, complexity: SceneComplexity) -> Tuple[float, float]:
        """
        Compute adaptive weights based on scene complexity.

        Strategy:
        - Clear scene (low overlap): Weight YOLO higher (faster)
        - Crowded scene (high overlap): Weight DETR higher (better handling)

        Args:
            complexity: Scene complexity metrics

        Returns:
            (yolo_weight, detr_weight)
        """
        if complexity.is_crowded:
            # Crowded scene: prefer DETR
            overlap_factor = min(complexity.overlap_ratio / self.crowd_threshold, 1.0)
            yolo_weight = 0.3 + 0.2 * (1 - overlap_factor)
            detr_weight = 0.7 - 0.2 * (1 - overlap_factor)
        else:
            # Clear scene: prefer YOLO (faster)
            yolo_weight = 0.7
            detr_weight = 0.3

        # Normalize
        total = yolo_weight + detr_weight
        yolo_weight /= total
        detr_weight /= total

        return yolo_weight, detr_weight

    def detect(self, frame: np.ndarray, pitch_mask: Optional[np.ndarray] = None) -> List[Detection]:
        """
        Detect players using hybrid YOLO + DETR approach.

        Args:
            frame: Input frame [H, W, 3]
            pitch_mask: Optional pitch mask

        Returns:
            List of fused detections
        """
        # Always run YOLO (fast baseline)
        yolo_detections = self.yolo_detector.detect(frame, pitch_mask)

        # If no DETR, return YOLO only
        if not self.use_detr:
            return yolo_detections

        # Run DETR
        detr_detections = self.detr_detector.detect(frame, pitch_mask)

        # Analyze scene complexity
        complexity = self.analyze_scene_complexity(yolo_detections, frame.shape[:2])

        # Compute weights
        if self.fusion_strategy == 'adaptive':
            yolo_weight, detr_weight = self.compute_adaptive_weights(complexity)
            logger.debug(
                f"Adaptive weights: YOLO={yolo_weight:.2f}, DETR={detr_weight:.2f} "
                f"(crowded={complexity.is_crowded}, overlap={complexity.overlap_ratio:.2f})"
            )
        else:
            yolo_weight = self.yolo_weight
            detr_weight = self.detr_weight

        # Fuse detections
        fused_detections = self._fuse_detections(
            yolo_detections,
            detr_detections,
            yolo_weight,
            detr_weight
        )

        logger.debug(
            f"Hybrid detection: YOLO={len(yolo_detections)}, DETR={len(detr_detections)}, "
            f"Fused={len(fused_detections)}"
        )

        return fused_detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect players in a batch of frames.

        Args:
            frames: List of input frames

        Returns:
            List of detection lists
        """
        # Run both detectors
        yolo_results = self.yolo_detector.detect_batch(frames)

        if not self.use_detr:
            return yolo_results

        detr_results = self.detr_detector.detect_batch(frames)

        # Fuse each frame
        fused_results = []
        for i, frame in enumerate(frames):
            complexity = self.analyze_scene_complexity(yolo_results[i], frame.shape[:2])

            if self.fusion_strategy == 'adaptive':
                yolo_weight, detr_weight = self.compute_adaptive_weights(complexity)
            else:
                yolo_weight = self.yolo_weight
                detr_weight = self.detr_weight

            fused = self._fuse_detections(
                yolo_results[i],
                detr_results[i],
                yolo_weight,
                detr_weight
            )
            fused_results.append(fused)

        return fused_results

    def _fuse_detections(
        self,
        yolo_detections: List[Detection],
        detr_detections: List[Detection],
        yolo_weight: float,
        detr_weight: float
    ) -> List[Detection]:
        """
        Fuse detections from YOLO and DETR using WBF.

        Args:
            yolo_detections: YOLO detections
            detr_detections: DETR detections
            yolo_weight: Weight for YOLO
            detr_weight: Weight for DETR

        Returns:
            Fused detections
        """
        if len(yolo_detections) == 0 and len(detr_detections) == 0:
            return []

        # Prepare boxes and scores
        boxes_list = []
        scores_list = []

        if len(yolo_detections) > 0:
            yolo_boxes = np.array([det.bbox for det in yolo_detections])
            yolo_scores = np.array([det.confidence for det in yolo_detections])
            boxes_list.append(yolo_boxes)
            scores_list.append(yolo_scores)
        else:
            boxes_list.append(np.array([]).reshape(0, 4))
            scores_list.append(np.array([]))

        if len(detr_detections) > 0:
            detr_boxes = np.array([det.bbox for det in detr_detections])
            detr_scores = np.array([det.confidence for det in detr_detections])
            boxes_list.append(detr_boxes)
            scores_list.append(detr_scores)
        else:
            boxes_list.append(np.array([]).reshape(0, 4))
            scores_list.append(np.array([]))

        # Fuse with WBF
        fused_boxes, fused_scores = self.wbf(
            boxes_list,
            scores_list,
            weights=[yolo_weight, detr_weight]
        )

        # Convert to Detection objects
        fused_detections = [
            Detection(bbox=box, confidence=score, class_id=0)
            for box, score in zip(fused_boxes, fused_scores)
        ]

        return fused_detections

    @staticmethod
    def _compute_iou_single(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)


class EnsembleConfig:
    """Configuration for ensemble detector."""

    def __init__(
        self,
        use_yolo: bool = True,
        use_detr: bool = True,
        fusion_strategy: str = 'adaptive',
        crowd_threshold: float = 0.3,
        yolo_weight: float = 0.7,
        detr_weight: float = 0.3
    ):
        self.use_yolo = use_yolo
        self.use_detr = use_detr
        self.fusion_strategy = fusion_strategy
        self.crowd_threshold = crowd_threshold
        self.yolo_weight = yolo_weight
        self.detr_weight = detr_weight

    def to_dict(self):
        return {
            'use_yolo': self.use_yolo,
            'use_detr': self.use_detr,
            'fusion_strategy': self.fusion_strategy,
            'crowd_threshold': self.crowd_threshold,
            'yolo_weight': self.yolo_weight,
            'detr_weight': self.detr_weight
        }
