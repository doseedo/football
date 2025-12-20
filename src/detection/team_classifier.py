"""Team classification based on jersey colors."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans


class Team(Enum):
    HOME = 0
    AWAY = 1
    REFEREE = 2
    GOALKEEPER_HOME = 3
    GOALKEEPER_AWAY = 4
    UNKNOWN = -1


@dataclass
class TeamClassification:
    team: Team
    confidence: float
    dominant_color: Tuple[int, int, int]


class TeamClassifier:
    def __init__(self, n_clusters: int = 3, color_space: str = "LAB",
                 jersey_region: Tuple[float, float, float, float] = (0.2, 0.15, 0.8, 0.55)):
        self.n_clusters = n_clusters
        self.color_space = color_space
        self.jersey_region = jersey_region
        self.team_colors: Optional[Dict[Team, np.ndarray]] = None
        self.kmeans: Optional[KMeans] = None
        self.is_fitted = False
        self.cluster_to_team: Dict[int, Team] = {}

    def extract_jersey_color(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        jx1 = x1 + int(w * self.jersey_region[0])
        jy1 = y1 + int(h * self.jersey_region[1])
        jx2 = x1 + int(w * self.jersey_region[2])
        jy2 = y1 + int(h * self.jersey_region[3])
        jersey_crop = frame[jy1:jy2, jx1:jx2]
        if jersey_crop.size == 0:
            return np.array([0, 0, 0])

        if self.color_space == "LAB":
            jersey_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2LAB)
        elif self.color_space == "HSV":
            jersey_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
        else:
            jersey_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2RGB)

        pixels = jersey_crop.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 10:
            return np.array([0, 0, 0])

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten())
        return centers[counts.argmax()]

    def fit(self, frame: np.ndarray, bboxes: List[np.ndarray]):
        all_colors = []
        for bbox in bboxes:
            color = self.extract_jersey_color(frame, bbox)
            if np.any(color > 0):
                all_colors.append(color)
        if len(all_colors) < self.n_clusters:
            raise ValueError(f"Not enough colors: {len(all_colors)}")

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(np.array(all_colors))

        cluster_counts = np.bincount(self.kmeans.labels_, minlength=self.n_clusters)
        sorted_clusters = np.argsort(cluster_counts)[::-1]

        self.team_colors = {
            Team.HOME: self.kmeans.cluster_centers_[sorted_clusters[0]],
            Team.AWAY: self.kmeans.cluster_centers_[sorted_clusters[1]],
        }
        if self.n_clusters > 2:
            self.team_colors[Team.REFEREE] = self.kmeans.cluster_centers_[sorted_clusters[2]]

        self.cluster_to_team = {sorted_clusters[0]: Team.HOME, sorted_clusters[1]: Team.AWAY}
        if self.n_clusters > 2:
            self.cluster_to_team[sorted_clusters[2]] = Team.REFEREE
        self.is_fitted = True

    def classify(self, frame: np.ndarray, bbox: np.ndarray) -> TeamClassification:
        if not self.is_fitted:
            return TeamClassification(team=Team.UNKNOWN, confidence=0.0, dominant_color=(0, 0, 0))

        color = self.extract_jersey_color(frame, bbox)
        if np.all(color == 0):
            return TeamClassification(team=Team.UNKNOWN, confidence=0.0, dominant_color=(0, 0, 0))

        distances = np.linalg.norm(self.kmeans.cluster_centers_ - color, axis=1)
        nearest_cluster = np.argmin(distances)
        confidence = max(0, 1 - (distances[nearest_cluster] / 100))
        team = self.cluster_to_team.get(nearest_cluster, Team.UNKNOWN)

        if self.color_space == "LAB":
            rgb_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_LAB2RGB)[0, 0]
        elif self.color_space == "HSV":
            rgb_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0, 0]
        else:
            rgb_color = color.astype(np.uint8)

        return TeamClassification(team=team, confidence=confidence, dominant_color=tuple(rgb_color))
