"""Automatic camera calibration using detected keypoints.

This module provides automatic homography estimation by combining keypoint detection
with RANSAC-based homography computation. Supports partial homography estimation
when fewer than 4 keypoints are visible.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger

from .field_model import FootballPitchModel, Keypoint3D, create_standard_pitch
from .keypoint_detector import (
    PitchKeypointDetector,
    DetectedKeypoint,
    KeypointDetectionResult,
    create_keypoint_detector
)


@dataclass
class CalibrationQuality:
    """Quality metrics for calibration result."""

    reprojection_error: float  # Mean reprojection error in pixels
    num_inliers: int  # Number of inlier keypoints
    num_total: int  # Total number of keypoints used
    confidence_mean: float  # Mean confidence of keypoints
    confidence_std: float  # Std dev of confidence
    homography_condition: float  # Condition number of homography matrix
    is_valid: bool  # Whether calibration is valid

    @property
    def inlier_ratio(self) -> float:
        """Ratio of inliers to total points."""
        return self.num_inliers / max(self.num_total, 1)

    @property
    def quality_score(self) -> float:
        """Overall quality score (0-1, higher is better)."""
        # Combine multiple factors
        error_score = max(0, 1 - self.reprojection_error / 10.0)  # Lower error is better
        inlier_score = self.inlier_ratio
        confidence_score = self.confidence_mean
        condition_score = max(0, 1 - np.log10(max(self.homography_condition, 1)) / 10)

        return (error_score * 0.3 + inlier_score * 0.3 +
                confidence_score * 0.3 + condition_score * 0.1)


@dataclass
class AutoCalibrationResult:
    """Result of automatic calibration."""

    homography: np.ndarray  # 3x3 homography matrix
    quality: CalibrationQuality  # Quality metrics
    matched_keypoints: List[Tuple[DetectedKeypoint, Keypoint3D]]  # Detected <-> World pairs
    detection_result: KeypointDetectionResult  # Original detection result
    method: str  # Calibration method used
    timestamp: Optional[float] = None  # Frame timestamp if available

    @property
    def is_valid(self) -> bool:
        """Check if calibration is valid."""
        return self.quality.is_valid


class AutoCalibrator:
    """Automatic camera calibration using keypoint detection and RANSAC.

    This class implements SkillCorner-style automatic calibration by:
    1. Detecting keypoints using deep learning
    2. Matching detected keypoints to world coordinates
    3. Computing homography using RANSAC + DLT
    4. Providing quality estimation
    """

    def __init__(
        self,
        detector: Optional[PitchKeypointDetector] = None,
        pitch_model: Optional[FootballPitchModel] = None,
        min_keypoints: int = 4,
        ransac_threshold: float = 5.0,
        min_confidence: float = 0.3,
        max_reprojection_error: float = 10.0,
        device: str = 'cpu'
    ):
        """Initialize automatic calibrator.

        Args:
            detector: Keypoint detector (creates default if None)
            pitch_model: Pitch model (creates standard if None)
            min_keypoints: Minimum keypoints required for calibration
            ransac_threshold: RANSAC inlier threshold in pixels
            min_confidence: Minimum detection confidence
            max_reprojection_error: Maximum acceptable reprojection error
            device: Device for neural network inference
        """
        self.pitch_model = pitch_model or create_standard_pitch()
        self.min_keypoints = min_keypoints
        self.ransac_threshold = ransac_threshold
        self.min_confidence = min_confidence
        self.max_reprojection_error = max_reprojection_error
        self.device = device

        # Create detector if not provided
        if detector is None:
            logger.info("Creating default keypoint detector")
            self.detector = None  # Will be created lazily
        else:
            self.detector = detector

        # Cache keypoint names and world coordinates
        self._keypoint_names = self.pitch_model.get_keypoint_names()
        self._world_coords_map = {
            name: kp.world_coords
            for name, kp in self.pitch_model.keypoints.items()
        }

        logger.info(f"Initialized AutoCalibrator with {len(self._keypoint_names)} pitch keypoints")

    def _ensure_detector(self):
        """Ensure detector is initialized."""
        if self.detector is None:
            self.detector = create_keypoint_detector(
                num_keypoints=len(self._keypoint_names),
                device=self.device
            )

    def calibrate_from_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> AutoCalibrationResult:
        """Perform automatic calibration on a single frame.

        Args:
            frame: Input frame [H, W, 3] in BGR format
            timestamp: Optional timestamp for temporal filtering

        Returns:
            AutoCalibrationResult with homography and quality metrics
        """
        # Ensure detector is initialized
        self._ensure_detector()

        # Detect keypoints
        detection_result = self.detector.detect_keypoints(
            frame,
            keypoint_names=self._keypoint_names,
            min_confidence=self.min_confidence
        )

        logger.info(f"Detected {detection_result.num_detected} keypoints "
                   f"(threshold: {self.min_confidence})")

        # Match keypoints to world coordinates
        matched_pairs = self._match_keypoints(detection_result)

        if len(matched_pairs) < self.min_keypoints:
            logger.warning(f"Insufficient keypoints for calibration: {len(matched_pairs)} < {self.min_keypoints}")
            return self._create_failed_result(detection_result, matched_pairs, timestamp)

        # Compute homography
        if len(matched_pairs) >= 4:
            result = self._compute_homography_ransac(matched_pairs, detection_result, timestamp)
        else:
            # Fallback to partial homography (affine approximation)
            logger.warning("Using partial homography with < 4 points")
            result = self._compute_partial_homography(matched_pairs, detection_result, timestamp)

        return result

    def _match_keypoints(
        self,
        detection_result: KeypointDetectionResult
    ) -> List[Tuple[DetectedKeypoint, Keypoint3D]]:
        """Match detected keypoints to world coordinates.

        Args:
            detection_result: Keypoint detection result

        Returns:
            List of (detected_kp, world_kp) pairs
        """
        matched_pairs = []

        for detected_kp in detection_result.keypoints:
            # Get corresponding world keypoint
            world_kp = self.pitch_model.get_keypoint(detected_kp.name)

            if world_kp is not None and detected_kp.confidence >= self.min_confidence:
                # Add world coordinates to detected keypoint
                detected_kp.world_coords = world_kp.world_coords
                matched_pairs.append((detected_kp, world_kp))

        logger.debug(f"Matched {len(matched_pairs)} keypoint pairs")
        return matched_pairs

    def _compute_homography_ransac(
        self,
        matched_pairs: List[Tuple[DetectedKeypoint, Keypoint3D]],
        detection_result: KeypointDetectionResult,
        timestamp: Optional[float] = None
    ) -> AutoCalibrationResult:
        """Compute homography using RANSAC + DLT.

        Args:
            matched_pairs: List of matched keypoint pairs
            detection_result: Original detection result
            timestamp: Optional timestamp

        Returns:
            AutoCalibrationResult
        """
        # Extract pixel and world coordinates
        pixel_points = np.array([kp[0].pixel_coords for kp in matched_pairs], dtype=np.float32)
        world_points = np.array([kp[1].world_coords for kp in matched_pairs], dtype=np.float32)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(
            pixel_points,
            world_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold
        )

        if H is None:
            logger.error("Homography computation failed")
            return self._create_failed_result(detection_result, matched_pairs, timestamp)

        # Compute quality metrics
        inlier_mask = mask.ravel().astype(bool)
        num_inliers = int(np.sum(inlier_mask))

        # Compute reprojection error
        reprojection_error = self._compute_reprojection_error(
            pixel_points[inlier_mask],
            world_points[inlier_mask],
            H
        )

        # Compute confidence statistics
        confidences = [kp[0].confidence for i, kp in enumerate(matched_pairs) if inlier_mask[i]]
        confidence_mean = np.mean(confidences) if confidences else 0.0
        confidence_std = np.std(confidences) if confidences else 0.0

        # Compute homography condition number
        condition_number = np.linalg.cond(H)

        # Check validity
        is_valid = (
            num_inliers >= self.min_keypoints and
            reprojection_error < self.max_reprojection_error and
            condition_number < 1e10
        )

        quality = CalibrationQuality(
            reprojection_error=reprojection_error,
            num_inliers=num_inliers,
            num_total=len(matched_pairs),
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            homography_condition=condition_number,
            is_valid=is_valid
        )

        logger.info(f"Calibration quality: error={reprojection_error:.2f}px, "
                   f"inliers={num_inliers}/{len(matched_pairs)}, "
                   f"score={quality.quality_score:.2f}")

        return AutoCalibrationResult(
            homography=H,
            quality=quality,
            matched_keypoints=matched_pairs,
            detection_result=detection_result,
            method="RANSAC",
            timestamp=timestamp
        )

    def _compute_partial_homography(
        self,
        matched_pairs: List[Tuple[DetectedKeypoint, Keypoint3D]],
        detection_result: KeypointDetectionResult,
        timestamp: Optional[float] = None
    ) -> AutoCalibrationResult:
        """Compute partial homography when < 4 points available.

        Uses affine transformation as approximation.

        Args:
            matched_pairs: List of matched keypoint pairs (must have >= 3)
            detection_result: Original detection result
            timestamp: Optional timestamp

        Returns:
            AutoCalibrationResult with affine-based homography
        """
        if len(matched_pairs) < 3:
            return self._create_failed_result(detection_result, matched_pairs, timestamp)

        # Extract pixel and world coordinates
        pixel_points = np.array([kp[0].pixel_coords for kp in matched_pairs[:3]], dtype=np.float32)
        world_points = np.array([kp[1].world_coords for kp in matched_pairs[:3]], dtype=np.float32)

        # Compute affine transformation
        affine_mat = cv2.getAffineTransform(pixel_points, world_points)

        # Convert to homography format
        H = np.vstack([affine_mat, [0, 0, 1]])

        # Compute basic metrics
        reprojection_error = self._compute_reprojection_error(pixel_points, world_points, H)
        confidences = [kp[0].confidence for kp in matched_pairs[:3]]
        confidence_mean = np.mean(confidences)

        quality = CalibrationQuality(
            reprojection_error=reprojection_error,
            num_inliers=len(matched_pairs),
            num_total=len(matched_pairs),
            confidence_mean=confidence_mean,
            confidence_std=np.std(confidences),
            homography_condition=np.linalg.cond(H),
            is_valid=reprojection_error < self.max_reprojection_error
        )

        logger.warning("Using partial homography (affine approximation)")

        return AutoCalibrationResult(
            homography=H,
            quality=quality,
            matched_keypoints=matched_pairs,
            detection_result=detection_result,
            method="Affine",
            timestamp=timestamp
        )

    def _compute_reprojection_error(
        self,
        pixel_points: np.ndarray,
        world_points: np.ndarray,
        H: np.ndarray
    ) -> float:
        """Compute mean reprojection error.

        Args:
            pixel_points: Pixel coordinates [N, 2]
            world_points: World coordinates [N, 2]
            H: Homography matrix [3, 3]

        Returns:
            Mean reprojection error in pixels
        """
        if len(pixel_points) == 0:
            return float('inf')

        # Transform pixel points to world coordinates
        pixel_homogeneous = np.column_stack([pixel_points, np.ones(len(pixel_points))])
        world_projected_h = (H @ pixel_homogeneous.T).T

        # Convert from homogeneous
        world_projected = world_projected_h[:, :2] / world_projected_h[:, 2:3]

        # Compute error
        errors = np.linalg.norm(world_projected - world_points, axis=1)
        return float(np.mean(errors))

    def _create_failed_result(
        self,
        detection_result: KeypointDetectionResult,
        matched_pairs: List[Tuple[DetectedKeypoint, Keypoint3D]],
        timestamp: Optional[float] = None
    ) -> AutoCalibrationResult:
        """Create a failed calibration result.

        Args:
            detection_result: Original detection result
            matched_pairs: Matched keypoint pairs
            timestamp: Optional timestamp

        Returns:
            AutoCalibrationResult with invalid homography
        """
        quality = CalibrationQuality(
            reprojection_error=float('inf'),
            num_inliers=0,
            num_total=len(matched_pairs),
            confidence_mean=0.0,
            confidence_std=0.0,
            homography_condition=float('inf'),
            is_valid=False
        )

        return AutoCalibrationResult(
            homography=np.eye(3),
            quality=quality,
            matched_keypoints=matched_pairs,
            detection_result=detection_result,
            method="Failed",
            timestamp=timestamp
        )

    def calibrate_batch(
        self,
        frames: List[np.ndarray],
        timestamps: Optional[List[float]] = None
    ) -> List[AutoCalibrationResult]:
        """Calibrate multiple frames.

        Args:
            frames: List of frames
            timestamps: Optional list of timestamps

        Returns:
            List of calibration results
        """
        results = []

        if timestamps is None:
            timestamps = [None] * len(frames)

        for frame, ts in zip(frames, timestamps):
            result = self.calibrate_from_frame(frame, timestamp=ts)
            results.append(result)

        return results


def visualize_calibration(
    frame: np.ndarray,
    result: AutoCalibrationResult,
    draw_grid: bool = True,
    grid_spacing: float = 10.0
) -> np.ndarray:
    """Visualize calibration result on frame.

    Args:
        frame: Input frame [H, W, 3]
        result: Calibration result
        draw_grid: Whether to draw world coordinate grid
        grid_spacing: Grid spacing in meters

    Returns:
        Annotated frame
    """
    vis_frame = frame.copy()

    if not result.is_valid:
        cv2.putText(vis_frame, "CALIBRATION FAILED", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return vis_frame

    # Get homography inverse for world -> pixel
    H_inv = np.linalg.inv(result.homography)

    # Draw grid
    if draw_grid:
        pitch_length = result.matched_keypoints[0][1].x if result.matched_keypoints else 105.0
        pitch_width = result.matched_keypoints[0][1].y if result.matched_keypoints else 68.0

        # Draw vertical lines
        for x in np.arange(0, pitch_length + grid_spacing, grid_spacing):
            pts_world = np.array([[x, 0], [x, pitch_width]], dtype=np.float32)
            pts_pixel = cv2.perspectiveTransform(pts_world.reshape(-1, 1, 2), H_inv)
            pt1 = tuple(pts_pixel[0, 0].astype(int))
            pt2 = tuple(pts_pixel[1, 0].astype(int))
            cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 1)

        # Draw horizontal lines
        for y in np.arange(0, pitch_width + grid_spacing, grid_spacing):
            pts_world = np.array([[0, y], [pitch_length, y]], dtype=np.float32)
            pts_pixel = cv2.perspectiveTransform(pts_world.reshape(-1, 1, 2), H_inv)
            pt1 = tuple(pts_pixel[0, 0].astype(int))
            pt2 = tuple(pts_pixel[1, 0].astype(int))
            cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 1)

    # Draw matched keypoints
    for detected_kp, world_kp in result.matched_keypoints:
        x, y = int(detected_kp.pixel_coords[0]), int(detected_kp.pixel_coords[1])

        # Color based on confidence
        if detected_kp.confidence > 0.7:
            color = (0, 255, 0)
        elif detected_kp.confidence > 0.5:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)

        cv2.circle(vis_frame, (x, y), 5, color, -1)
        cv2.circle(vis_frame, (x, y), 7, (255, 255, 255), 2)

    # Draw quality info
    info_y = 30
    cv2.putText(vis_frame, f"Quality: {result.quality.quality_score:.2f}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += 25
    cv2.putText(vis_frame, f"Error: {result.quality.reprojection_error:.2f}px", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += 25
    cv2.putText(vis_frame, f"Inliers: {result.quality.num_inliers}/{result.quality.num_total}",
               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis_frame
