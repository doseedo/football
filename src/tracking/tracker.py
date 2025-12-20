"""Multi-object tracking using ByteTrack."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import supervision as sv
from ..detection.player_detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    confidence: float
    class_id: int
    team: Optional[int] = None
    jersey_number: Optional[int] = None
    history: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def foot_position(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])

    def add_position(self, max_history: int = 100):
        self.history.append(self.foot_position)
        if len(self.history) > max_history:
            self.history.pop(0)


class PlayerTracker:
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, frame_rate: int = 25):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )
        self.tracks: Dict[int, Track] = {}
        self.frame_count = 0

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Track]:
        self.frame_count += 1
        if len(detections) == 0:
            return list(self.tracks.values())

        bboxes = np.array([d.bbox for d in detections])
        confidences = np.array([d.confidence for d in detections])
        class_ids = np.array([d.class_id for d in detections])

        sv_detections = sv.Detections(xyxy=bboxes, confidence=confidences, class_id=class_ids)
        tracked = self.tracker.update_with_detections(sv_detections)

        active_tracks = []
        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                track_id = int(track_id)
                if track_id not in self.tracks:
                    self.tracks[track_id] = Track(
                        track_id=track_id,
                        bbox=tracked.xyxy[i],
                        confidence=tracked.confidence[i] if tracked.confidence is not None else 1.0,
                        class_id=int(tracked.class_id[i]) if tracked.class_id is not None else 0
                    )
                else:
                    self.tracks[track_id].bbox = tracked.xyxy[i]
                    self.tracks[track_id].confidence = tracked.confidence[i] if tracked.confidence is not None else 1.0
                self.tracks[track_id].add_position()
                active_tracks.append(self.tracks[track_id])
        return active_tracks

    def reset(self):
        self.tracker.reset()
        self.tracks.clear()
        self.frame_count = 0
