"""Player detection using YOLOv8."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from loguru import logger
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    track_id: Optional[int] = None

    @property
    def foot_position(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class PlayerDetector:
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        confidence: float = 0.3,
        iou_threshold: float = 0.5,
        min_height: int = 30,
        max_height: int = 400,
        device: str = "cuda",
        half: bool = True
    ):
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.min_height = min_height
        self.max_height = max_height

        logger.info(f"Loading detection model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        if half and device == "cuda":
            self.model.model.half()

    def detect(self, frame: np.ndarray, pitch_mask: Optional[np.ndarray] = None) -> List[Detection]:
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            height = bbox[3] - bbox[1]

            if height < self.min_height or height > self.max_height:
                continue

            if pitch_mask is not None:
                cx, cy = (bbox[0] + bbox[2]) / 2, bbox[3]
                h, w = pitch_mask.shape[:2]
                px, py = int(cx), int(cy)
                if 0 <= px < w and 0 <= py < h:
                    if pitch_mask[py, px] == 0:
                        continue

            detections.append(Detection(bbox=bbox, confidence=conf, class_id=self.PERSON_CLASS_ID))
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        results = self.model(frames, conf=self.confidence, iou=self.iou_threshold,
                           classes=[self.PERSON_CLASS_ID], verbose=False)
        all_detections = []
        for result in results:
            frame_detections = []
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                height = bbox[3] - bbox[1]
                if self.min_height <= height <= self.max_height:
                    frame_detections.append(Detection(bbox=bbox, confidence=conf, class_id=0))
            all_detections.append(frame_detections)
        return all_detections
