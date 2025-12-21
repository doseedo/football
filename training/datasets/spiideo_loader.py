"""
Spiideo Dataset Loader

Spiideo format typically includes:
- Video files (.mp4)
- Tracking data (JSON/CSV with player positions)
- Event annotations
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SpiideoDataset:
    """
    Loader for Spiideo tracking data.
    
    Expected directory structure:
    spiideo_data/
    ├── match_001/
    │   ├── video.mp4
    │   ├── tracking.json (or tracking.csv)
    │   ├── events.json
    │   └── metadata.json
    ├── match_002/
    │   └── ...
    """
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.matches = self._discover_matches()
        print(f"Found {len(self.matches)} matches in {data_dir}")
    
    def _discover_matches(self) -> List[Path]:
        """Find all match directories."""
        matches = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # Check for tracking data
                if (item / 'tracking.json').exists() or (item / 'tracking.csv').exists():
                    matches.append(item)
        return sorted(matches)
    
    def load_tracking(self, match_dir: Path) -> Dict:
        """Load tracking data from a match directory."""
        json_path = match_dir / 'tracking.json'
        csv_path = match_dir / 'tracking.csv'
        
        if json_path.exists():
            return self._load_json_tracking(json_path)
        elif csv_path.exists():
            return self._load_csv_tracking(csv_path)
        else:
            raise FileNotFoundError(f"No tracking data found in {match_dir}")
    
    def _load_json_tracking(self, path: Path) -> Dict:
        """Load JSON format tracking data."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Normalize to common format
        frames = []
        
        # Handle different Spiideo JSON formats
        if 'frames' in data:
            raw_frames = data['frames']
        elif 'tracking' in data:
            raw_frames = data['tracking']
        else:
            raw_frames = data if isinstance(data, list) else []
        
        for frame_data in raw_frames:
            frame = {
                'frame_id': frame_data.get('frame', frame_data.get('frame_id', 0)),
                'timestamp': frame_data.get('timestamp', frame_data.get('time', 0)),
                'players': [],
                'ball': None
            }
            
            # Extract player positions
            players = frame_data.get('players', frame_data.get('objects', []))
            for p in players:
                player = {
                    'id': p.get('id', p.get('player_id', p.get('track_id'))),
                    'team': p.get('team', p.get('team_id', 'unknown')),
                    'x': p.get('x', p.get('pos_x', p.get('position', {}).get('x'))),
                    'y': p.get('y', p.get('pos_y', p.get('position', {}).get('y'))),
                    'jersey_number': p.get('jersey', p.get('number', p.get('jersey_number'))),
                    'bbox': p.get('bbox', p.get('bounding_box'))
                }
                frame['players'].append(player)
            
            # Extract ball position
            ball = frame_data.get('ball', frame_data.get('ball_position'))
            if ball:
                frame['ball'] = {
                    'x': ball.get('x', ball.get('pos_x')),
                    'y': ball.get('y', ball.get('pos_y')),
                    'z': ball.get('z', ball.get('pos_z', 0)),
                    'visible': ball.get('visible', True)
                }
            
            frames.append(frame)
        
        return {'frames': frames, 'metadata': data.get('metadata', {})}
    
    def _load_csv_tracking(self, path: Path) -> Dict:
        """Load CSV format tracking data."""
        frames_dict = {}
        
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = int(row.get('frame', row.get('frame_id', 0)))
                
                if frame_id not in frames_dict:
                    frames_dict[frame_id] = {
                        'frame_id': frame_id,
                        'timestamp': float(row.get('timestamp', row.get('time', 0))),
                        'players': [],
                        'ball': None
                    }
                
                # Determine if this row is a player or ball
                obj_type = row.get('type', row.get('object_type', 'player'))
                
                if obj_type == 'ball':
                    frames_dict[frame_id]['ball'] = {
                        'x': float(row.get('x', 0)),
                        'y': float(row.get('y', 0)),
                        'z': float(row.get('z', 0)),
                        'visible': row.get('visible', 'true').lower() == 'true'
                    }
                else:
                    frames_dict[frame_id]['players'].append({
                        'id': row.get('id', row.get('player_id', row.get('track_id'))),
                        'team': row.get('team', row.get('team_id', 'unknown')),
                        'x': float(row.get('x', 0)),
                        'y': float(row.get('y', 0)),
                        'jersey_number': row.get('jersey', row.get('number')),
                        'bbox': None
                    })
        
        frames = [frames_dict[k] for k in sorted(frames_dict.keys())]
        return {'frames': frames, 'metadata': {}}
    
    def convert_to_training_format(self, output_dir: str):
        """Convert Spiideo data to unified training format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for match_dir in self.matches:
            match_name = match_dir.name
            tracking = self.load_tracking(match_dir)
            
            # Save in unified format
            output_file = output_path / f"{match_name}.json"
            with open(output_file, 'w') as f:
                json.dump(tracking, f)
            
            print(f"Converted {match_name}: {len(tracking['frames'])} frames")
        
        print(f"\n✓ Converted {len(self.matches)} matches to {output_dir}")


class SpiideoTrackingDataset(Dataset if HAS_TORCH else object):
    """PyTorch Dataset for Spiideo tracking data."""
    
    def __init__(self, data_dir: str, sequence_length: int = 50, transform=None):
        self.loader = SpiideoDataset(data_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = self._build_sequences()
    
    def _build_sequences(self) -> List[Tuple[Path, int, int]]:
        """Build list of (match_dir, start_frame, end_frame) tuples."""
        sequences = []
        for match_dir in self.loader.matches:
            tracking = self.loader.load_tracking(match_dir)
            n_frames = len(tracking['frames'])
            
            # Create overlapping sequences
            for start in range(0, n_frames - self.sequence_length, self.sequence_length // 2):
                sequences.append((match_dir, start, start + self.sequence_length))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        match_dir, start, end = self.sequences[idx]
        tracking = self.loader.load_tracking(match_dir)
        frames = tracking['frames'][start:end]
        
        # Convert to tensors
        # Shape: (seq_len, max_players, features)
        max_players = 22
        features = 4  # x, y, team_id, player_id
        
        positions = np.zeros((self.sequence_length, max_players, features))
        
        for t, frame in enumerate(frames):
            for i, player in enumerate(frame['players'][:max_players]):
                positions[t, i, 0] = player['x'] or 0
                positions[t, i, 1] = player['y'] or 0
                positions[t, i, 2] = hash(str(player['team'])) % 3  # 0, 1, 2 for teams
                positions[t, i, 3] = hash(str(player['id'])) % 100
        
        if HAS_TORCH:
            positions = torch.FloatTensor(positions)
        
        if self.transform:
            positions = self.transform(positions)
        
        return positions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Spiideo data directory')
    parser.add_argument('--output-dir', default='data/spiideo_converted', help='Output directory')
    parser.add_argument('--convert', action='store_true', help='Convert to training format')
    args = parser.parse_args()
    
    dataset = SpiideoDataset(args.data_dir)
    
    if args.convert:
        dataset.convert_to_training_format(args.output_dir)
    else:
        # Just print stats
        for match_dir in dataset.matches[:5]:
            tracking = dataset.load_tracking(match_dir)
            print(f"{match_dir.name}: {len(tracking['frames'])} frames")
