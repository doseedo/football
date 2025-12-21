"""Automatic pitch line and keypoint detection for dynamic homography."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger


@dataclass
class PitchKeypoint:
    name: str
    pixel_coords: Tuple[float, float]
    world_coords: Tuple[float, float]
    confidence: float


@dataclass
class PitchDetectionResult:
    keypoints: List[PitchKeypoint]
    line_mask: np.ndarray
    confidence: float
    is_valid: bool


class PitchLineDetector:
    """Detects pitch lines and keypoints for automatic homography estimation."""

    # Standard pitch dimensions in meters
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0

    # Known pitch keypoint world coordinates
    WORLD_KEYPOINTS = {
        "corner_top_left": (0.0, 0.0),
        "corner_top_right": (105.0, 0.0),
        "corner_bottom_left": (0.0, 68.0),
        "corner_bottom_right": (105.0, 68.0),
        "center_top": (52.5, 0.0),
        "center_bottom": (52.5, 68.0),
        "center_spot": (52.5, 34.0),
        "penalty_left_top": (0.0, 13.85),
        "penalty_left_bottom": (0.0, 54.15),
        "penalty_right_top": (105.0, 13.85),
        "penalty_right_bottom": (105.0, 54.15),
        "penalty_area_left_top": (16.5, 13.85),
        "penalty_area_left_bottom": (16.5, 54.15),
        "penalty_area_right_top": (88.5, 13.85),
        "penalty_area_right_bottom": (88.5, 54.15),
        "goal_area_left_top": (5.5, 24.85),
        "goal_area_left_bottom": (5.5, 43.15),
        "goal_area_right_top": (99.5, 24.85),
        "goal_area_right_bottom": (99.5, 43.15),
        "penalty_spot_left": (11.0, 34.0),
        "penalty_spot_right": (94.0, 34.0),
    }

    def __init__(
        self,
        white_threshold: int = 200,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 100,
        min_line_length: int = 100,
        max_line_gap: int = 10,
    ):
        self.white_threshold = white_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def detect_lines(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Detect pitch lines using color filtering and Hough transform."""
        # Convert to grayscale and extract white regions (pitch lines)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Also try LAB color space for better white detection
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Combine grayscale and L channel for better line detection
        combined = cv2.addWeighted(gray, 0.5, l_channel, 0.5, 0)

        # Threshold for white lines
        _, white_mask = cv2.threshold(combined, self.white_threshold, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        # Edge detection
        edges = cv2.Canny(white_mask, self.canny_low, self.canny_high)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        line_list = []
        if lines is not None:
            for line in lines:
                line_list.append(line[0])

        return white_mask, line_list

    def classify_lines(self, lines: List[np.ndarray], frame_shape: Tuple[int, int]) -> Dict[str, List[np.ndarray]]:
        """Classify lines as horizontal (touchlines) or vertical (goal lines, etc.)."""
        horizontal_lines = []
        vertical_lines = []

        h, w = frame_shape[:2]

        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Horizontal lines (touchlines, penalty area lines)
            if angle < 30 or angle > 150:
                horizontal_lines.append(line)
            # Vertical lines (goal lines, center line)
            elif 60 < angle < 120:
                vertical_lines.append(line)

        return {
            "horizontal": horizontal_lines,
            "vertical": vertical_lines
        }

    def find_line_intersections(self, lines: List[np.ndarray]) -> List[Tuple[float, float]]:
        """Find intersection points between lines."""
        intersections = []

        for i, line1 in enumerate(lines):
            for line2 in lines[i + 1:]:
                intersection = self._line_intersection(line1, line2)
                if intersection is not None:
                    intersections.append(intersection)

        return intersections

    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)

    def detect_center_circle(self, frame: np.ndarray, white_mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Detect center circle using Hough circle detection."""
        # Apply additional blur for circle detection
        blurred = cv2.GaussianBlur(white_mask, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=300
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Return the most prominent circle (largest)
            best_circle = max(circles[0], key=lambda c: c[2])
            return (float(best_circle[0]), float(best_circle[1]), float(best_circle[2]))

        return None

    def detect_keypoints(self, frame: np.ndarray) -> PitchDetectionResult:
        """Detect pitch keypoints from frame."""
        h, w = frame.shape[:2]
        white_mask, lines = self.detect_lines(frame)

        if len(lines) < 4:
            return PitchDetectionResult(
                keypoints=[],
                line_mask=white_mask,
                confidence=0.0,
                is_valid=False
            )

        classified_lines = self.classify_lines(lines, frame.shape)
        intersections = []

        # Find intersections between horizontal and vertical lines
        for h_line in classified_lines["horizontal"]:
            for v_line in classified_lines["vertical"]:
                intersection = self._line_intersection(h_line, v_line)
                if intersection is not None:
                    x, y = intersection
                    # Filter points within frame
                    if 0 <= x < w and 0 <= y < h:
                        intersections.append((x, y))

        # Detect center circle
        center_circle = self.detect_center_circle(frame, white_mask)

        keypoints = []

        # Add center spot if circle detected
        if center_circle is not None:
            cx, cy, _ = center_circle
            keypoints.append(PitchKeypoint(
                name="center_spot",
                pixel_coords=(cx, cy),
                world_coords=self.WORLD_KEYPOINTS["center_spot"],
                confidence=0.8
            ))

        # Cluster intersections and match to known keypoints
        if len(intersections) >= 4:
            # Sort by position to identify corners and key points
            sorted_by_y = sorted(intersections, key=lambda p: p[1])
            top_points = sorted(sorted_by_y[:len(sorted_by_y)//2], key=lambda p: p[0])
            bottom_points = sorted(sorted_by_y[len(sorted_by_y)//2:], key=lambda p: p[0])

            # Try to identify corner points
            if len(top_points) >= 2:
                keypoints.append(PitchKeypoint(
                    name="corner_top_left",
                    pixel_coords=top_points[0],
                    world_coords=self.WORLD_KEYPOINTS["corner_top_left"],
                    confidence=0.6
                ))
                keypoints.append(PitchKeypoint(
                    name="corner_top_right",
                    pixel_coords=top_points[-1],
                    world_coords=self.WORLD_KEYPOINTS["corner_top_right"],
                    confidence=0.6
                ))

            if len(bottom_points) >= 2:
                keypoints.append(PitchKeypoint(
                    name="corner_bottom_left",
                    pixel_coords=bottom_points[0],
                    world_coords=self.WORLD_KEYPOINTS["corner_bottom_left"],
                    confidence=0.6
                ))
                keypoints.append(PitchKeypoint(
                    name="corner_bottom_right",
                    pixel_coords=bottom_points[-1],
                    world_coords=self.WORLD_KEYPOINTS["corner_bottom_right"],
                    confidence=0.6
                ))

        confidence = min(1.0, len(keypoints) / 4.0) if keypoints else 0.0

        return PitchDetectionResult(
            keypoints=keypoints,
            line_mask=white_mask,
            confidence=confidence,
            is_valid=len(keypoints) >= 4
        )


class FeatureBasedPitchMatcher:
    """Match pitch features between frames for rotation tracking."""

    def __init__(self, max_features: int = 500, match_threshold: float = 0.7):
        self.max_features = max_features
        self.match_threshold = match_threshold
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.reference_keypoints = None
        self.reference_descriptors = None
        self.reference_frame = None

    def set_reference_frame(self, frame: np.ndarray, pitch_mask: Optional[np.ndarray] = None):
        """Set reference frame for feature matching."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply pitch mask if available
        if pitch_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=pitch_mask)

        self.reference_keypoints, self.reference_descriptors = self.orb.detectAndCompute(gray, None)
        self.reference_frame = frame.copy()

        logger.debug(f"Reference frame set with {len(self.reference_keypoints)} features")

    def compute_frame_transform(self, frame: np.ndarray, pitch_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Compute homography from current frame to reference frame."""
        if self.reference_descriptors is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if pitch_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=pitch_mask)

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) < 10:
            return None

        # Match features
        matches = self.bf_matcher.knnMatch(descriptors, self.reference_descriptors, k=2)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            return None

        # Extract matched points
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.reference_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return None

        # Validate homography (check for reasonable transformation)
        if not self._validate_homography(H, frame.shape):
            return None

        return H

    def _validate_homography(self, H: np.ndarray, frame_shape: Tuple[int, ...]) -> bool:
        """Validate that homography represents a reasonable camera rotation."""
        h, w = frame_shape[:2]

        # Check corners transformation
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)

        if transformed is None:
            return False

        transformed = transformed.reshape(-1, 2)

        # Check that corners are still roughly within expected range
        # For ±45 degree rotation, allow significant but bounded movement
        max_displacement = max(w, h) * 0.8

        for i, (orig, trans) in enumerate(zip(corners.reshape(-1, 2), transformed)):
            displacement = np.linalg.norm(trans - orig)
            if displacement > max_displacement:
                return False

        # Check that the quadrilateral is still convex
        def cross_product_sign(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        signs = []
        for i in range(4):
            o = transformed[i]
            a = transformed[(i + 1) % 4]
            b = transformed[(i + 2) % 4]
            signs.append(cross_product_sign(o, a, b) > 0)

        if not (all(signs) or not any(signs)):
            return False

        return True

    def estimate_rotation_angle(self, H: np.ndarray) -> float:
        """Estimate rotation angle from homography matrix."""
        if H is None:
            return 0.0

        # Decompose homography to extract rotation
        # For pure rotation around optical axis, H ≈ R (rotation matrix)
        # Extract rotation angle from the rotation part

        # Simplified: use SVD to extract rotation
        try:
            U, S, Vt = np.linalg.svd(H[:2, :2])
            R = U @ Vt
            angle = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
            return angle
        except:
            return 0.0
