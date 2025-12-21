"""
SkillCorner Open Data Loader

SkillCorner provides open tracking data for several matches.
Download from: https://github.com/SkillCorner/opendata

The data includes:
- Player tracking (x, y positions at 10Hz)
- Ball tracking
- Match events
- Team formations

Download Instructions:
    git clone https://github.com/SkillCorner/opendata.git data/training/skillcorner
    # Or download directly from GitHub releases
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger


class SkillCornerDataset(Dataset):
    """
    SkillCorner open tracking dataset.

    Used for:
    - Trajectory prediction (Baller2Vec)
    - Tactical analysis (GNN)
    - Formation classification

    Data format (JSON):
    {
        "timestamp": 0.0,
        "period": 1,
        "possession": {"team": "home"},
        "data": [
            {
                "track_id": 1,
                "team_id": "home",
                "x": 52.5,  # meters
                "y": 34.0,
                "z": 0.0
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        root_path: str,
        split: str = "train",
        sequence_length: int = 100,
        stride: int = 25,
        fps: int = 10,  # SkillCorner is 10 Hz
        min_trajectory_length: int = 25,
        normalize: bool = True,
        pitch_dimensions: Tuple[float, float] = (105.0, 68.0),
    ):
        """
        Args:
            root_path: Path to SkillCorner data (cloned repo or extracted data)
            split: 'train', 'val', or 'test'
            sequence_length: Length of trajectory sequences
            stride: Stride for sliding window
            fps: Frames per second (SkillCorner is 10Hz)
            min_trajectory_length: Filter out short trajectories
            normalize: Normalize coordinates to [-1, 1]
            pitch_dimensions: Pitch size in meters
        """
        self.root_path = Path(root_path)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.fps = fps
        self.min_trajectory_length = min_trajectory_length
        self.normalize = normalize
        self.pitch_dimensions = pitch_dimensions

        # Load data
        self.sequences = self._load_skillcorner_data()
        logger.info(f"Loaded {len(self.sequences)} SkillCorner sequences for {split}")

    def _load_skillcorner_data(self) -> List[Dict]:
        """Load SkillCorner tracking data from JSON files."""
        sequences = []

        # SkillCorner structure: data/matches/match_id/structured_data.json
        match_dirs = list(self.root_path.glob("data/matches/*"))

        if not match_dirs:
            # Try alternative structure
            match_dirs = list(self.root_path.glob("matches/*"))

        if not match_dirs:
            logger.warning(f"No SkillCorner data found in {self.root_path}")
            logger.warning("Download from: https://github.com/SkillCorner/opendata")
            return []

        for match_dir in match_dirs:
            structured_file = match_dir / "structured_data.json"
            if not structured_file.exists():
                continue

            logger.info(f"Loading {match_dir.name}...")
            match_sequences = self._parse_match_data(structured_file, match_dir.name)
            sequences.extend(match_sequences)

        # Split data by matches
        num_matches = len(match_dirs)
        train_matches = int(0.8 * num_matches)
        val_matches = int(0.9 * num_matches)

        # Group by match
        match_seqs = {}
        for seq in sequences:
            match_id = seq['match_id']
            if match_id not in match_seqs:
                match_seqs[match_id] = []
            match_seqs[match_id].append(seq)

        # Split
        match_ids = sorted(match_seqs.keys())
        if self.split == "train":
            selected_matches = match_ids[:train_matches]
        elif self.split == "val":
            selected_matches = match_ids[train_matches:val_matches]
        else:  # test
            selected_matches = match_ids[val_matches:]

        filtered_sequences = []
        for match_id in selected_matches:
            filtered_sequences.extend(match_seqs[match_id])

        return filtered_sequences

    def _parse_match_data(self, json_file: Path, match_id: str) -> List[Dict]:
        """Parse a single match's tracking data."""
        with open(json_file, 'r') as f:
            match_data = json.load(f)

        # Extract player trajectories
        player_tracks = {}  # track_id -> list of (timestamp, x, y, team)

        for frame in match_data:
            timestamp = frame.get('timestamp', 0)
            period = frame.get('period', 1)

            for player_data in frame.get('data', []):
                track_id = player_data.get('track_id')
                if track_id is None or track_id == -1:  # Ball or invalid
                    continue

                team = player_data.get('team_id', 'unknown')
                x = player_data.get('x', 0)
                y = player_data.get('y', 0)

                key = f"{period}_{track_id}"
                if key not in player_tracks:
                    player_tracks[key] = []

                player_tracks[key].append({
                    'timestamp': timestamp,
                    'x': x,
                    'y': y,
                    'team': team,
                })

        # Create sequences
        sequences = []
        for track_id, track_data in player_tracks.items():
            if len(track_data) < self.min_trajectory_length:
                continue

            # Sort by timestamp
            track_data = sorted(track_data, key=lambda x: x['timestamp'])

            # Extract positions
            positions = np.array([[p['x'], p['y']] for p in track_data])
            team = track_data[0]['team']

            # Create overlapping sequences
            for start_idx in range(0, len(positions) - self.sequence_length, self.stride):
                end_idx = start_idx + self.sequence_length
                seq_positions = positions[start_idx:end_idx]

                # Normalize if requested
                if self.normalize:
                    seq_positions = self._normalize_positions(seq_positions)

                sequences.append({
                    'track_id': track_id,
                    'positions': seq_positions,
                    'team': 0 if team == 'home' else 1,
                    'match_id': match_id,
                })

        return sequences

    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Normalize positions to [-1, 1] based on pitch dimensions."""
        normalized = positions.copy()
        normalized[:, 0] = (positions[:, 0] / self.pitch_dimensions[0]) * 2 - 1
        normalized[:, 1] = (positions[:, 1] / self.pitch_dimensions[1]) * 2 - 1
        return normalized

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        positions = torch.from_numpy(seq['positions']).float()

        # Compute velocities
        velocities = torch.zeros_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1]) * self.fps

        # Compute accelerations
        accelerations = torch.zeros_like(positions)
        accelerations[1:] = (velocities[1:] - velocities[:-1]) * self.fps

        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'team': torch.tensor(seq['team']),
            'track_id': seq['track_id'],
            'match_id': seq['match_id'],
        }


class SkillCornerMatchDataset(Dataset):
    """
    SkillCorner dataset for full match context (GNN training).

    Returns entire frames with all players for graph construction.
    """

    def __init__(
        self,
        root_path: str,
        split: str = "train",
        sequence_length: int = 50,
        stride: int = 25,
        normalize: bool = True,
        pitch_dimensions: Tuple[float, float] = (105.0, 68.0),
    ):
        self.root_path = Path(root_path)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.pitch_dimensions = pitch_dimensions

        self.sequences = self._load_match_sequences()
        logger.info(f"Loaded {len(self.sequences)} match frame sequences for {split}")

    def _load_match_sequences(self) -> List[Dict]:
        """Load full-frame sequences for graph-based learning."""
        sequences = []

        match_dirs = list(self.root_path.glob("data/matches/*"))
        if not match_dirs:
            match_dirs = list(self.root_path.glob("matches/*"))

        for match_dir in match_dirs:
            structured_file = match_dir / "structured_data.json"
            if not structured_file.exists():
                continue

            with open(structured_file, 'r') as f:
                match_data = json.load(f)

            # Group by period
            periods = {}
            for frame in match_data:
                period = frame.get('period', 1)
                if period not in periods:
                    periods[period] = []
                periods[period].append(frame)

            # Create sequences for each period
            for period, frames in periods.items():
                frames = sorted(frames, key=lambda x: x.get('timestamp', 0))

                for start_idx in range(0, len(frames) - self.sequence_length, self.stride):
                    end_idx = start_idx + self.sequence_length
                    frame_sequence = frames[start_idx:end_idx]

                    sequences.append({
                        'frames': frame_sequence,
                        'match_id': match_dir.name,
                        'period': period,
                    })

        # Split
        total = len(sequences)
        if self.split == "train":
            sequences = sequences[:int(0.8 * total)]
        elif self.split == "val":
            sequences = sequences[int(0.8 * total):int(0.9 * total)]
        else:
            sequences = sequences[int(0.9 * total):]

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        frames = seq['frames']

        # Parse all players in each frame
        max_players = 23  # 22 players + ball
        sequence_data = []

        for frame in frames:
            frame_players = []

            for player_data in frame.get('data', []):
                x = player_data.get('x', 0)
                y = player_data.get('y', 0)

                if self.normalize:
                    x = (x / self.pitch_dimensions[0]) * 2 - 1
                    y = (y / self.pitch_dimensions[1]) * 2 - 1

                team = 0 if player_data.get('team_id') == 'home' else 1
                track_id = player_data.get('track_id', -1)

                frame_players.append([x, y, team, track_id])

            # Pad to max_players
            while len(frame_players) < max_players:
                frame_players.append([0, 0, -1, -1])  # Padding

            sequence_data.append(frame_players[:max_players])

        sequence_tensor = torch.tensor(sequence_data, dtype=torch.float32)

        return {
            'sequence': sequence_tensor,  # (T, N, 4) - time, nodes, features
            'match_id': seq['match_id'],
            'period': seq['period'],
        }


def download_skillcorner_data(save_path: str):
    """
    Download SkillCorner open data.

    Args:
        save_path: Where to save the data

    Example:
        download_skillcorner_data("data/training/skillcorner")
    """
    import subprocess

    logger.info("Downloading SkillCorner open data...")
    subprocess.run([
        "git", "clone",
        "https://github.com/SkillCorner/opendata.git",
        save_path
    ])
    logger.info(f"SkillCorner data downloaded to {save_path}")
