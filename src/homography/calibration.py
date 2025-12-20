"""Camera calibration and homography estimation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger


@dataclass
class CalibrationResult:
    homography: np.ndarray
    reprojection_error: float
    keypoints_used: List[str]
    is_valid: bool


class HomographyEstimator:
    def __init__(self, min_keypoints: int = 4, ransac_threshold: float = 3.0):
        self.min_keypoints = min_keypoints
        self.ransac_threshold = ransac_threshold
        self.current_homography: Optional[np.ndarray] = None

    def estimate_from_manual_points(
        self,
        pixel_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]]
    ) -> CalibrationResult:
        if len(pixel_points) < 4 or len(pixel_points) != len(world_points):
            return CalibrationResult(np.eye(3), float('inf'), [], False)

        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)

        if H is None:
            return CalibrationResult(np.eye(3), float('inf'), [], False)

        projected = self._transform_points(src, H)
        error = np.mean(np.linalg.norm(projected - dst, axis=1))
        self.current_homography = H

        return CalibrationResult(H, error, ["manual"] * len(pixel_points), True)

    def _transform_points(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        points_h = np.column_stack([points, np.ones(len(points))])
        transformed_h = (H @ points_h.T).T
        return transformed_h[:, :2] / transformed_h[:, 2:3]


class InteractiveCalibrator:
    def __init__(self, pitch_template: Dict):
        self.pitch_template = pitch_template
        self.pixel_points: List[Tuple[float, float]] = []
        self.world_points: List[Tuple[float, float]] = []
        self.current_frame: Optional[np.ndarray] = None
        self.common_keypoints = [
            ("corner_top_left", (0, 0)),
            ("corner_top_right", (105, 0)),
            ("corner_bottom_left", (0, 68)),
            ("corner_bottom_right", (105, 68)),
            ("center_top", (52.5, 0)),
            ("center_bottom", (52.5, 68)),
            ("center_spot", (52.5, 34)),
            ("penalty_spot_left", (11, 34)),
            ("penalty_spot_right", (94, 34)),
        ]

    def calibrate_interactive(self, frame: np.ndarray) -> CalibrationResult:
        self.current_frame = frame.copy()
        self.pixel_points = []
        self.world_points = []

        window_name = "Calibration - Click keypoints"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("Click keypoints in order:")
        for i, (name, coords) in enumerate(self.common_keypoints):
            print(f"  {i+1}. {name} -> {coords}")
        print("Press 'q' when done, 'r' to reset, 'u' to undo")

        while True:
            display = self.current_frame.copy()
            for i, pt in enumerate(self.pixel_points):
                cv2.circle(display, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                cv2.putText(display, str(i+1), (int(pt[0])+10, int(pt[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if len(self.pixel_points) < len(self.common_keypoints):
                current_kp = self.common_keypoints[len(self.pixel_points)][0]
                cv2.putText(display, f"Click: {current_kp}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.pixel_points = []
                self.world_points = []
            elif key == ord('u') and len(self.pixel_points) > 0:
                self.pixel_points.pop()
                self.world_points.pop()

        cv2.destroyAllWindows()
        estimator = HomographyEstimator()
        return estimator.estimate_from_manual_points(self.pixel_points, self.world_points)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pixel_points) < len(self.common_keypoints):
                self.pixel_points.append((float(x), float(y)))
                self.world_points.append(self.common_keypoints[len(self.pixel_points)-1][1])
                print(f"Added point {len(self.pixel_points)}: ({x}, {y})")


class CoordinateTransformer:
    def __init__(self, homography: Optional[np.ndarray] = None):
        self.H = homography
        self.H_inv = None
        if homography is not None:
            self.set_homography(homography)

    def set_homography(self, H: np.ndarray):
        self.H = H
        self.H_inv = np.linalg.inv(H)

    def pixel_to_world(self, pixel_coords: Tuple[float, float]) -> Tuple[float, float]:
        if self.H is None:
            raise ValueError("Homography not set")
        px, py = pixel_coords
        point = np.array([px, py, 1.0])
        world_h = self.H @ point
        return (float(world_h[0] / world_h[2]), float(world_h[1] / world_h[2]))

    def world_to_pixel(self, world_coords: Tuple[float, float]) -> Tuple[float, float]:
        if self.H_inv is None:
            raise ValueError("Homography not set")
        wx, wy = world_coords
        point = np.array([wx, wy, 1.0])
        pixel_h = self.H_inv @ point
        return (float(pixel_h[0] / pixel_h[2]), float(pixel_h[1] / pixel_h[2]))
