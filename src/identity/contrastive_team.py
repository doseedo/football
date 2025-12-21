"""Unsupervised team classification using contrastive learning.

Automatically discovers team clusters without labels by maximizing
distance between teams and minimizing distance within teams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.cluster import KMeans, DBSCAN
from collections import defaultdict, Counter
import cv2


class ColorFeatureExtractor:
    """Extract color features from player crops for team classification."""

    def __init__(self, num_bins: int = 32):
        """
        Args:
            num_bins: Number of bins for color histogram
        """
        self.num_bins = num_bins

    def extract_torso_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Extract mask for upper torso region (jersey region).

        Args:
            image: Player crop (H, W, 3)

        Returns:
            Binary mask (H, W)
        """
        h, w = image.shape[:2]

        # Focus on middle region (avoid head/legs)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Upper torso region
        y1 = int(h * 0.2)
        y2 = int(h * 0.6)
        x1 = int(w * 0.2)
        x2 = int(w * 0.8)

        mask[y1:y2, x1:x2] = 1

        return mask

    def extract_color_histogram(self, image: np.ndarray,
                                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract color histogram in HSV space.

        Args:
            image: Player crop (H, W, 3) in RGB
            mask: Optional binary mask

        Returns:
            Normalized histogram (num_bins * 3,)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Compute histogram for each channel
        histograms = []
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], mask, [self.num_bins],
                              [0, 256 if i > 0 else 180])
            hist = hist.flatten()
            histograms.append(hist)

        # Concatenate and normalize
        feature = np.concatenate(histograms)
        feature = feature / (feature.sum() + 1e-6)

        return feature

    def extract_dominant_colors(self, image: np.ndarray,
                                mask: Optional[np.ndarray] = None,
                                k: int = 3) -> np.ndarray:
        """
        Extract dominant colors using K-means.

        Args:
            image: Player crop (H, W, 3)
            mask: Optional binary mask
            k: Number of dominant colors

        Returns:
            Dominant colors (k, 3) and their weights (k,)
        """
        # Apply mask
        if mask is not None:
            pixels = image[mask > 0]
        else:
            pixels = image.reshape(-1, 3)

        if len(pixels) < k:
            return np.zeros((k, 3)), np.zeros(k)

        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(pixels)

        # Get cluster centers (dominant colors) and their weights
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        weights = np.bincount(labels, minlength=k) / len(labels)

        return colors, weights

    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive color features.

        Args:
            image: Player crop (H, W, 3)

        Returns:
            Dict with different feature types
        """
        # Extract torso mask
        mask = self.extract_torso_mask(image)

        # Color histogram
        hist = self.extract_color_histogram(image, mask)

        # Dominant colors
        colors, weights = self.extract_dominant_colors(image, mask, k=3)

        # Weighted dominant color feature
        dominant_feature = (colors * weights[:, np.newaxis]).flatten()

        # Combine features
        features = {
            'histogram': hist,
            'dominant_colors': colors,
            'dominant_weights': weights,
            'dominant_feature': dominant_feature,
            'combined': np.concatenate([hist, dominant_feature])
        }

        return features


class ContrastiveLoss(nn.Module):
    """Contrastive loss for team classification.

    Uses InfoNCE-style loss to separate teams.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            features: Feature embeddings (B, D), L2-normalized
            labels: Team labels (B,)

        Returns:
            Loss value
        """
        device = features.device
        batch_size = features.size(0)

        # Compute similarity matrix
        similarity = torch.mm(features, features.t()) / self.temperature

        # Create mask for positive pairs (same team)
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()).float().to(device)

        # Create mask for negative pairs (different team)
        mask_neg = 1 - mask_pos

        # Remove diagonal (self-similarity)
        mask_pos = mask_pos - torch.eye(batch_size).to(device)

        # Compute loss
        exp_sim = torch.exp(similarity)

        # Positive similarity
        pos_sim = (exp_sim * mask_pos).sum(1)

        # Negative similarity (all samples)
        neg_sim = (exp_sim * mask_neg).sum(1)

        # InfoNCE loss
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-6))
        loss = loss.mean()

        return loss


class TeamClassifier:
    """Unsupervised team classification using color clustering.

    Discovers team clusters without labels using appearance features.
    """

    def __init__(self, num_teams: int = 3, method: str = 'kmeans',
                 feature_type: str = 'combined'):
        """
        Args:
            num_teams: Number of teams (usually 2 teams + 1 referee)
            method: Clustering method ('kmeans' or 'dbscan')
            feature_type: Type of color features to use
        """
        self.num_teams = num_teams
        self.method = method
        self.feature_type = feature_type

        self.color_extractor = ColorFeatureExtractor()
        self.cluster_model = None
        self.team_colors = None  # Representative colors for each team

        # Track history for temporal consistency
        self.track_history = defaultdict(list)  # track_id -> [team_ids]

    def extract_features_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of player crops.

        Args:
            images: Player crops (B, H, W, 3)

        Returns:
            Features (B, D)
        """
        features = []
        for img in images:
            feat_dict = self.color_extractor.extract_features(img)
            features.append(feat_dict[self.feature_type])

        return np.array(features)

    def fit(self, images: np.ndarray, track_ids: Optional[np.ndarray] = None):
        """
        Fit clustering model on player crops.

        Args:
            images: Player crops (B, H, W, 3)
            track_ids: Optional track IDs for temporal consistency
        """
        # Extract features
        features = self.extract_features_batch(images)

        # Fit clustering model
        if self.method == 'kmeans':
            self.cluster_model = KMeans(n_clusters=self.num_teams,
                                       random_state=0, n_init=20)
            self.cluster_model.fit(features)
        elif self.method == 'dbscan':
            self.cluster_model = DBSCAN(eps=0.3, min_samples=5)
            self.cluster_model.fit(features)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # Extract representative colors for each team
        self._extract_team_colors(images, self.cluster_model.labels_)

        print(f"Team classification fitted. Found {len(np.unique(self.cluster_model.labels_))} teams.")

    def predict(self, images: np.ndarray, track_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict team labels for player crops.

        Args:
            images: Player crops (B, H, W, 3)
            track_ids: Optional track IDs for temporal smoothing

        Returns:
            Team labels (B,)
        """
        if self.cluster_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Extract features
        features = self.extract_features_batch(images)

        # Predict
        if hasattr(self.cluster_model, 'predict'):
            labels = self.cluster_model.predict(features)
        else:
            # DBSCAN doesn't have predict, use fit_predict
            labels = self.cluster_model.fit_predict(features)

        # Apply temporal smoothing if track_ids provided
        if track_ids is not None:
            labels = self._temporal_smoothing(labels, track_ids)

        return labels

    def _temporal_smoothing(self, labels: np.ndarray,
                           track_ids: np.ndarray,
                           window_size: int = 10) -> np.ndarray:
        """
        Smooth team labels temporally using track history.

        Args:
            labels: Current frame labels (B,)
            track_ids: Track IDs (B,)
            window_size: Smoothing window size

        Returns:
            Smoothed labels (B,)
        """
        smoothed = labels.copy()

        for i, track_id in enumerate(track_ids):
            # Add current prediction to history
            self.track_history[track_id].append(labels[i])

            # Keep only recent history
            if len(self.track_history[track_id]) > window_size:
                self.track_history[track_id].pop(0)

            # Vote for most common label
            if len(self.track_history[track_id]) >= 3:
                counter = Counter(self.track_history[track_id])
                smoothed[i] = counter.most_common(1)[0][0]

        return smoothed

    def _extract_team_colors(self, images: np.ndarray, labels: np.ndarray):
        """
        Extract representative colors for each team.

        Args:
            images: Player crops (B, H, W, 3)
            labels: Team labels (B,)
        """
        unique_labels = np.unique(labels)
        self.team_colors = {}

        for label in unique_labels:
            if label == -1:  # Skip noise in DBSCAN
                continue

            # Get images for this team
            team_images = images[labels == label]

            if len(team_images) == 0:
                continue

            # Extract dominant colors from all team members
            all_colors = []
            for img in team_images[:20]:  # Use up to 20 samples
                colors, _ = self.color_extractor.extract_dominant_colors(img, k=1)
                all_colors.append(colors[0])

            # Average color
            avg_color = np.mean(all_colors, axis=0)
            self.team_colors[label] = avg_color

    def get_team_color(self, team_id: int) -> Optional[np.ndarray]:
        """Get representative color for a team."""
        if self.team_colors is None or team_id not in self.team_colors:
            return None
        return self.team_colors[team_id]

    def identify_referee(self) -> Optional[int]:
        """
        Identify which team ID corresponds to the referee.

        Referees typically wear black or very different colors from teams.

        Returns:
            Team ID for referee, or None if not identifiable
        """
        if self.team_colors is None or len(self.team_colors) < 3:
            return None

        # Referees often wear black (low brightness)
        # or very saturated colors (high saturation)
        team_metrics = {}
        for team_id, color in self.team_colors.items():
            # Convert RGB to HSV
            hsv = cv2.cvtColor(color.reshape(1, 1, 3).astype(np.uint8),
                              cv2.COLOR_RGB2HSV)[0, 0]

            # Referee metric: low value (dark) or very different from others
            brightness = hsv[2]
            team_metrics[team_id] = brightness

        # Assume referee has lowest or highest brightness
        sorted_teams = sorted(team_metrics.items(), key=lambda x: x[1])

        # Check if there's a clear outlier (very dark = black referee kit)
        if sorted_teams[0][1] < 50:  # Very dark
            return sorted_teams[0][0]

        return None


class OnlineTeamClassifier:
    """Online version that updates clusters as new data arrives."""

    def __init__(self, num_teams: int = 3, update_frequency: int = 30):
        """
        Args:
            num_teams: Number of teams
            update_frequency: How often to refit the model (in frames)
        """
        self.num_teams = num_teams
        self.update_frequency = update_frequency

        self.classifier = TeamClassifier(num_teams=num_teams)
        self.buffer_images = []
        self.buffer_track_ids = []
        self.frame_count = 0
        self.is_fitted = False

    def update(self, images: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
        """
        Update classifier and predict team labels.

        Args:
            images: Player crops (B, H, W, 3)
            track_ids: Track IDs (B,)

        Returns:
            Team labels (B,)
        """
        # Add to buffer
        self.buffer_images.extend(images)
        self.buffer_track_ids.extend(track_ids)

        # Limit buffer size
        max_buffer = 500
        if len(self.buffer_images) > max_buffer:
            self.buffer_images = self.buffer_images[-max_buffer:]
            self.buffer_track_ids = self.buffer_track_ids[-max_buffer:]

        # Refit periodically or on first frame
        if not self.is_fitted or self.frame_count % self.update_frequency == 0:
            if len(self.buffer_images) >= self.num_teams * 5:
                buffer_array = np.array(self.buffer_images)
                track_array = np.array(self.buffer_track_ids)
                self.classifier.fit(buffer_array, track_array)
                self.is_fitted = True

        # Predict
        if self.is_fitted:
            labels = self.classifier.predict(images, track_ids)
        else:
            # Not enough data yet
            labels = np.zeros(len(images), dtype=int)

        self.frame_count += 1
        return labels

    def get_team_info(self) -> Dict:
        """Get information about detected teams."""
        info = {
            'num_teams': self.num_teams,
            'is_fitted': self.is_fitted,
            'team_colors': self.classifier.team_colors,
            'referee_id': self.classifier.identify_referee()
        }
        return info


def visualize_teams(image: np.ndarray, team_id: int,
                   team_color: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Visualize team assignment on player crop.

    Args:
        image: Player crop (H, W, 3)
        team_id: Team ID
        team_color: Team color (3,) in RGB

    Returns:
        Visualization
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    # Color map if not provided
    if team_color is None:
        colors = [(255, 0, 0), (0, 0, 255), (0, 0, 0)]  # Red, Blue, Black
        team_color = colors[team_id % len(colors)]
    else:
        team_color = tuple(team_color.astype(int).tolist())

    # Draw border
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), team_color, 3)

    # Draw team ID
    text = f"Team {team_id}"
    cv2.putText(vis, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, team_color, 2)

    return vis
