"""Ball detection with temporal consistency."""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO


@dataclass
class BallDetection:
    position: Tuple[float, float]
    confidence: float
    bbox: Optional[np.ndarray] = None
    is_interpolated: bool = False


class BallDetector:
    SPORTS_BALL_CLASS = 32

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        confidence: float = 0.2,
        temporal_window: int = 5,
        max_velocity: float = 150.0,
        device: str = "cuda"
    ):
        self.confidence = confidence
        self.temporal_window = temporal_window
        self.max_velocity = max_velocity

        logger.info(f"Loading ball detection model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)

        self.position_buffer = deque(maxlen=temporal_window)
        self.velocity_buffer = deque(maxlen=temporal_window - 1)
        self.last_position: Optional[Tuple[float, float]] = None
        self.frames_since_detection = 0

    def detect(self, frame: np.ndarray, player_bboxes: Optional[List[np.ndarray]] = None) -> Optional[BallDetection]:
        # Try YOLO detection
        results = self.model(frame, conf=self.confidence, classes=[self.SPORTS_BALL_CLASS], verbose=False)[0]

        if len(results.boxes) > 0:
            best_idx = results.boxes.conf.argmax()
            bbox = results.boxes.xyxy[best_idx].cpu().numpy()
            conf = float(results.boxes.conf[best_idx])
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            if self._validate_detection(center, player_bboxes):
                self._update_temporal_state(center)
                self.frames_since_detection = 0
                return BallDetection(position=center, confidence=conf, bbox=bbox)

        # Interpolation fallback
        self.frames_since_detection += 1
        if self.frames_since_detection <= self.temporal_window and self.last_position:
            interpolated = self._interpolate_position()
            if interpolated:
                return BallDetection(
                    position=interpolated,
                    confidence=0.5 / self.frames_since_detection,
                    is_interpolated=True
                )
        return None

    def _validate_detection(self, position: Tuple[float, float], player_bboxes: Optional[List[np.ndarray]]) -> bool:
        if self.last_position:
            dist = np.sqrt((position[0] - self.last_position[0])**2 + (position[1] - self.last_position[1])**2)
            if dist > self.max_velocity * 2:
                return False
        return True

    def _update_temporal_state(self, position: Tuple[float, float]):
        if self.last_position:
            velocity = (position[0] - self.last_position[0], position[1] - self.last_position[1])
            self.velocity_buffer.append(velocity)
        self.position_buffer.append(position)
        self.last_position = position

    def _interpolate_position(self) -> Optional[Tuple[float, float]]:
        if not self.last_position or len(self.velocity_buffer) == 0:
            return None
        avg_vx = np.mean([v[0] for v in self.velocity_buffer])
        avg_vy = np.mean([v[1] for v in self.velocity_buffer])
        return (
            self.last_position[0] + avg_vx * self.frames_since_detection,
            self.last_position[1] + avg_vy * self.frames_since_detection
        )

    def reset(self):
        self.position_buffer.clear()
        self.velocity_buffer.clear()
        self.last_position = None
        self.frames_since_detection = 0
