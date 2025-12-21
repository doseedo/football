"""Combined player identification pipeline.

Fuses OSNet embeddings, jersey numbers, and team classification
for robust player re-identification across a match.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import cv2

from .osnet import ReIDExtractor
from .jersey_detector import JerseyDetector
from .jersey_recognizer import JerseyRecognizer, TemporalNumberAggregator
from .contrastive_team import OnlineTeamClassifier


@dataclass
class PlayerIdentity:
    """Player identity information."""
    stable_id: int  # Stable player ID across the match
    track_id: int  # Current track ID
    team_id: int  # Team assignment (0, 1, 2 for team A, B, referee)
    jersey_number: Optional[int] = None  # Jersey number (1-99)
    confidence: float = 0.0  # Overall identification confidence

    # Additional info
    appearance_embedding: Optional[np.ndarray] = None
    team_confidence: float = 0.0
    number_confidence: float = 0.0


class PlayerGallery:
    """Maintains a gallery of known players for matching.

    Stores appearance embeddings and jersey numbers for each known player.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Args:
            similarity_threshold: Minimum similarity for matching
        """
        self.similarity_threshold = similarity_threshold

        # Player data
        self.players = {}  # stable_id -> PlayerIdentity
        self.embeddings = {}  # stable_id -> List[embeddings]
        self.track_to_stable = {}  # track_id -> stable_id

        self.next_stable_id = 0

    def add_player(self, identity: PlayerIdentity):
        """Add a new player to the gallery."""
        stable_id = identity.stable_id

        if stable_id not in self.players:
            self.players[stable_id] = identity
            self.embeddings[stable_id] = []

        # Update embedding history
        if identity.appearance_embedding is not None:
            self.embeddings[stable_id].append(identity.appearance_embedding)

            # Keep only recent embeddings
            max_embeddings = 50
            if len(self.embeddings[stable_id]) > max_embeddings:
                self.embeddings[stable_id] = self.embeddings[stable_id][-max_embeddings:]

        # Update track mapping
        self.track_to_stable[identity.track_id] = stable_id

        # Update player info (keep most confident)
        if identity.confidence > self.players[stable_id].confidence:
            self.players[stable_id] = identity

    def find_matching_player(self, embedding: np.ndarray, team_id: int,
                            jersey_number: Optional[int] = None) -> Optional[int]:
        """
        Find matching player in gallery.

        Args:
            embedding: Appearance embedding
            team_id: Team ID
            jersey_number: Optional jersey number

        Returns:
            stable_id of matching player, or None
        """
        # Filter candidates by team
        candidates = [sid for sid, p in self.players.items() if p.team_id == team_id]

        if len(candidates) == 0:
            return None

        # If jersey number is known, use it for matching
        if jersey_number is not None:
            for sid in candidates:
                if self.players[sid].jersey_number == jersey_number:
                    return sid

        # Otherwise, use appearance similarity
        best_similarity = 0.0
        best_player = None

        embedding_tensor = torch.from_numpy(embedding).unsqueeze(0)

        for sid in candidates:
            if sid not in self.embeddings or len(self.embeddings[sid]) == 0:
                continue

            # Average similarity to all embeddings of this player
            gallery_embeddings = torch.from_numpy(
                np.array(self.embeddings[sid])
            )

            # Compute similarity
            similarity = torch.mm(embedding_tensor, gallery_embeddings.t())
            avg_similarity = similarity.mean().item()

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_player = sid

        # Check threshold
        if best_similarity >= self.similarity_threshold:
            return best_player
        else:
            return None

    def create_new_player(self, track_id: int, team_id: int,
                         embedding: Optional[np.ndarray] = None,
                         jersey_number: Optional[int] = None) -> int:
        """
        Create a new player entry.

        Returns:
            New stable_id
        """
        stable_id = self.next_stable_id
        self.next_stable_id += 1

        identity = PlayerIdentity(
            stable_id=stable_id,
            track_id=track_id,
            team_id=team_id,
            jersey_number=jersey_number,
            appearance_embedding=embedding
        )

        self.add_player(identity)

        return stable_id

    def get_stable_id(self, track_id: int) -> Optional[int]:
        """Get stable ID for a track."""
        return self.track_to_stable.get(track_id)

    def get_player(self, stable_id: int) -> Optional[PlayerIdentity]:
        """Get player info by stable ID."""
        return self.players.get(stable_id)

    def get_jersey_number(self, stable_id: int) -> Optional[int]:
        """Get jersey number for a player."""
        player = self.players.get(stable_id)
        return player.jersey_number if player else None


class PlayerIdentifier:
    """Combined pipeline for player re-identification.

    Fuses multiple cues:
    1. Appearance embeddings (OSNet)
    2. Jersey numbers (CRNN)
    3. Team classification (Color clustering)
    """

    def __init__(self,
                 osnet_path: Optional[str] = None,
                 jersey_detector_path: Optional[str] = None,
                 jersey_recognizer_path: Optional[str] = None,
                 device: str = 'cuda',
                 num_teams: int = 3):
        """
        Args:
            osnet_path: Path to OSNet pretrained weights
            jersey_detector_path: Path to jersey detector weights
            jersey_recognizer_path: Path to jersey recognizer weights
            device: Device to run on
            num_teams: Number of teams (2 teams + referee)
        """
        self.device = device

        # Initialize components
        self.reid_extractor = ReIDExtractor(model_path=osnet_path, device=device)
        self.jersey_detector = JerseyDetector(model_path=jersey_detector_path,
                                             device=device, use_heuristic=True)
        self.jersey_recognizer = JerseyRecognizer(model_path=jersey_recognizer_path,
                                                 device=device)
        self.team_classifier = OnlineTeamClassifier(num_teams=num_teams,
                                                    update_frequency=30)

        # Temporal aggregation
        self.number_aggregator = TemporalNumberAggregator(window_size=30,
                                                          min_confidence=0.5)

        # Player gallery
        self.gallery = PlayerGallery(similarity_threshold=0.7)

        print("PlayerIdentifier initialized")
        print(f"  - OSNet: {'loaded' if osnet_path else 'not loaded'}")
        print(f"  - Jersey detector: {'loaded' if jersey_detector_path else 'heuristic'}")
        print(f"  - Jersey recognizer: {'loaded' if jersey_recognizer_path else 'not loaded'}")

    def process_frame(self, player_crops: np.ndarray, track_ids: np.ndarray) -> List[PlayerIdentity]:
        """
        Process a frame of player detections.

        Args:
            player_crops: Player crops (B, H, W, 3) in RGB, [0, 255]
            track_ids: Track IDs (B,)

        Returns:
            List of PlayerIdentity for each detection
        """
        if len(player_crops) == 0:
            return []

        # 1. Extract appearance embeddings
        crops_tensor = torch.from_numpy(player_crops).float() / 255.0
        crops_tensor = crops_tensor.permute(0, 3, 1, 2)  # (B, 3, H, W)
        embeddings = self.reid_extractor.extract_features(crops_tensor)
        embeddings = embeddings.numpy()

        # 2. Classify teams
        team_ids = self.team_classifier.update(player_crops, track_ids)

        # 3. Detect and recognize jersey numbers
        jersey_results = self._process_jersey_numbers(player_crops, track_ids)

        # 4. Identify players
        identities = []
        for i, track_id in enumerate(track_ids):
            identity = self.identify(
                player_crop=player_crops[i],
                track_id=int(track_id),
                embedding=embeddings[i],
                team_id=int(team_ids[i]),
                jersey_info=jersey_results[i]
            )
            identities.append(identity)

        return identities

    def _process_jersey_numbers(self, player_crops: np.ndarray,
                                track_ids: np.ndarray) -> List[Dict]:
        """
        Detect and recognize jersey numbers.

        Args:
            player_crops: Player crops (B, H, W, 3)
            track_ids: Track IDs (B,)

        Returns:
            List of dicts with number info
        """
        results = []

        # Detect number regions
        detections = self.jersey_detector.detect(player_crops, refine=True)

        # Recognize numbers
        for i, det in enumerate(detections):
            if det['crop'].size == 0:
                results.append({
                    'number': None,
                    'confidence': 0.0
                })
                continue

            # Recognize
            recognition = self.jersey_recognizer.recognize_single(det['crop'])

            # Update temporal aggregator
            self.number_aggregator.add_prediction(
                track_id=int(track_ids[i]),
                number=recognition['number'],
                confidence=recognition['confidence'] * det['confidence']
            )

            # Get stable number
            stable_number = self.number_aggregator.get_stable_number(
                track_id=int(track_ids[i]),
                min_votes=3
            )

            results.append({
                'number': stable_number or recognition['number'],
                'confidence': recognition['confidence'] * det['confidence']
            })

        return results

    def identify(self, player_crop: np.ndarray, track_id: int,
                embedding: Optional[np.ndarray] = None,
                team_id: Optional[int] = None,
                jersey_info: Optional[Dict] = None) -> PlayerIdentity:
        """
        Identify a player using all available cues.

        Args:
            player_crop: Player crop (H, W, 3)
            track_id: Track ID
            embedding: Optional pre-computed embedding
            team_id: Optional pre-computed team ID
            jersey_info: Optional jersey number info

        Returns:
            PlayerIdentity
        """
        # Check if we already have a stable ID for this track
        stable_id = self.gallery.get_stable_id(track_id)

        if stable_id is not None:
            # Update existing player
            identity = self.gallery.get_player(stable_id)

            # Update with new info
            if embedding is not None:
                identity.appearance_embedding = embedding
            if team_id is not None:
                identity.team_id = team_id
            if jersey_info is not None and jersey_info['number'] is not None:
                identity.jersey_number = jersey_info['number']
                identity.number_confidence = jersey_info['confidence']

            identity.track_id = track_id

            # Update gallery
            self.gallery.add_player(identity)

            return identity

        # New track - try to match to existing player
        jersey_number = jersey_info['number'] if jersey_info else None

        if embedding is not None and team_id is not None:
            matched_stable_id = self.gallery.find_matching_player(
                embedding=embedding,
                team_id=team_id,
                jersey_number=jersey_number
            )

            if matched_stable_id is not None:
                # Matched to existing player
                stable_id = matched_stable_id
                identity = self.gallery.get_player(stable_id)
                identity.track_id = track_id

                # Update with new info
                if embedding is not None:
                    identity.appearance_embedding = embedding
                if jersey_number is not None:
                    identity.jersey_number = jersey_number
                    identity.number_confidence = jersey_info['confidence']

                # Update gallery
                self.gallery.add_player(identity)

                return identity

        # Create new player
        stable_id = self.gallery.create_new_player(
            track_id=track_id,
            team_id=team_id or 0,
            embedding=embedding,
            jersey_number=jersey_number
        )

        identity = self.gallery.get_player(stable_id)
        if jersey_info:
            identity.number_confidence = jersey_info['confidence']

        return identity

    def update_gallery(self, track_id: int, identity_info: Dict):
        """
        Update player gallery with new information.

        Args:
            track_id: Track ID
            identity_info: Dict with identity information
        """
        stable_id = self.gallery.get_stable_id(track_id)

        if stable_id is not None:
            identity = self.gallery.get_player(stable_id)

            # Update fields
            if 'jersey_number' in identity_info:
                identity.jersey_number = identity_info['jersey_number']
            if 'team_id' in identity_info:
                identity.team_id = identity_info['team_id']
            if 'embedding' in identity_info:
                identity.appearance_embedding = identity_info['embedding']

            self.gallery.add_player(identity)

    def get_stable_id(self, track_id: int) -> Optional[int]:
        """
        Get stable player ID for a track.

        Args:
            track_id: Track ID

        Returns:
            Stable player ID, or None if not found
        """
        return self.gallery.get_stable_id(track_id)

    def get_jersey_number(self, track_id: int) -> Optional[int]:
        """
        Get jersey number for a track.

        Args:
            track_id: Track ID

        Returns:
            Jersey number, or None if not known
        """
        stable_id = self.gallery.get_stable_id(track_id)
        if stable_id is not None:
            return self.gallery.get_jersey_number(stable_id)
        return None

    def get_team_info(self) -> Dict:
        """Get information about detected teams."""
        return self.team_classifier.get_team_info()

    def get_all_players(self) -> List[PlayerIdentity]:
        """Get all known players."""
        return list(self.gallery.players.values())

    def export_roster(self) -> Dict[int, Dict]:
        """
        Export match roster with all identified players.

        Returns:
            Dict mapping stable_id to player info
        """
        roster = {}
        for stable_id, identity in self.gallery.players.items():
            roster[stable_id] = {
                'team_id': identity.team_id,
                'jersey_number': identity.jersey_number,
                'confidence': identity.confidence,
                'number_confidence': identity.number_confidence,
                'team_confidence': identity.team_confidence
            }
        return roster


def visualize_identity(image: np.ndarray, identity: PlayerIdentity,
                      team_colors: Optional[Dict] = None) -> np.ndarray:
    """
    Visualize player identity on crop.

    Args:
        image: Player crop (H, W, 3)
        identity: PlayerIdentity
        team_colors: Optional dict of team colors

    Returns:
        Visualization
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    # Get team color
    if team_colors and identity.team_id in team_colors:
        color = tuple(team_colors[identity.team_id].astype(int).tolist())
    else:
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
        color = colors[identity.team_id % len(colors)]

    # Draw border
    thickness = 3
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), color, thickness)

    # Draw info
    y = 20
    texts = [
        f"ID: {identity.stable_id}",
        f"Team: {identity.team_id}",
    ]

    if identity.jersey_number is not None:
        texts.append(f"#{identity.jersey_number}")

    for text in texts:
        cv2.putText(vis, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 2)
        cv2.putText(vis, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, 1)
        y += 20

    return vis


def create_player_identifier(config: Optional[Dict] = None) -> PlayerIdentifier:
    """
    Create PlayerIdentifier with configuration.

    Args:
        config: Configuration dict with paths to models

    Returns:
        PlayerIdentifier instance

    Example config:
        {
            'osnet_path': 'models/osnet_ain_x1_0.pth',
            'jersey_detector_path': 'models/jersey_detector.pth',
            'jersey_recognizer_path': 'models/jersey_recognizer.pth',
            'device': 'cuda',
            'num_teams': 3
        }
    """
    if config is None:
        config = {}

    return PlayerIdentifier(
        osnet_path=config.get('osnet_path'),
        jersey_detector_path=config.get('jersey_detector_path'),
        jersey_recognizer_path=config.get('jersey_recognizer_path'),
        device=config.get('device', 'cuda'),
        num_teams=config.get('num_teams', 3)
    )
