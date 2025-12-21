"""
Wyscout Dataset Loader

Wyscout format includes:
- Match data with events
- Player positions (when available)
- Tactical data
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class WyscoutDataset:
    """
    Loader for Wyscout tracking/event data.
    
    Wyscout provides data in various formats:
    - Events JSON (standard)
    - Tracking JSON (premium)
    - Match metadata
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.matches = self._discover_matches()
        print(f"Found {len(self.matches)} matches in {data_dir}")
    
    def _discover_matches(self) -> List[Dict]:
        """Find all match files."""
        matches = []
        
        # Check for different Wyscout file patterns
        patterns = ['**/matches*.json', '**/events*.json', '**/tracking*.json']
        
        for pattern in patterns:
            for file_path in self.data_dir.glob(pattern):
                matches.append({'path': file_path, 'type': file_path.stem})
        
        # Also check for individual match directories
        for item in self.data_dir.iterdir():
            if item.is_dir():
                match_info = {'dir': item, 'files': {}}
                for f in item.iterdir():
                    if f.suffix == '.json':
                        match_info['files'][f.stem] = f
                if match_info['files']:
                    matches.append(match_info)
        
        return matches
    
    def load_events(self, file_path: Path) -> List[Dict]:
        """Load Wyscout event data."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        events = data if isinstance(data, list) else data.get('events', [])
        return events
    
    def load_tracking(self, file_path: Path) -> Dict:
        """Load Wyscout tracking data (premium format)."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        frames = []
        
        # Wyscout tracking format
        tracking_data = data.get('tracking', data.get('frames', data))
        if isinstance(tracking_data, list):
            for frame_data in tracking_data:
                frame = {
                    'frame_id': frame_data.get('frameIdx', frame_data.get('frame', 0)),
                    'timestamp': frame_data.get('timestamp', 0),
                    'period': frame_data.get('period', 1),
                    'players': [],
                    'ball': None
                }
                
                # Home team players
                for p in frame_data.get('homePlayers', frame_data.get('home', [])):
                    frame['players'].append({
                        'id': p.get('playerId', p.get('id')),
                        'team': 'home',
                        'x': p.get('x', 0),
                        'y': p.get('y', 0),
                        'jersey_number': p.get('jerseyNo', p.get('number'))
                    })
                
                # Away team players
                for p in frame_data.get('awayPlayers', frame_data.get('away', [])):
                    frame['players'].append({
                        'id': p.get('playerId', p.get('id')),
                        'team': 'away',
                        'x': p.get('x', 0),
                        'y': p.get('y', 0),
                        'jersey_number': p.get('jerseyNo', p.get('number'))
                    })
                
                # Ball
                ball_data = frame_data.get('ball', {})
                if ball_data:
                    frame['ball'] = {
                        'x': ball_data.get('x', ball_data.get('xyz', [0])[0] if 'xyz' in ball_data else 0),
                        'y': ball_data.get('y', ball_data.get('xyz', [0, 0])[1] if 'xyz' in ball_data else 0),
                        'z': ball_data.get('z', ball_data.get('xyz', [0, 0, 0])[2] if 'xyz' in ball_data else 0),
                    }
                
                frames.append(frame)
        
        return {'frames': frames, 'metadata': data.get('metadata', {})}
    
    def events_to_positions(self, events: List[Dict]) -> Dict:
        """
        Convert Wyscout events to approximate positions.
        Useful when only event data is available (not tracking).
        
        Note: This creates sparse position data based on events,
        not full tracking data.
        """
        frames = []
        
        for event in events:
            # Wyscout uses 0-100 coordinate system
            positions = event.get('positions', [])
            if not positions:
                continue
            
            frame = {
                'event_id': event.get('eventId', event.get('id')),
                'timestamp': event.get('eventSec', event.get('second', 0)),
                'event_type': event.get('eventName', event.get('type')),
                'period': event.get('matchPeriod', '1H'),
                'player': {
                    'id': event.get('playerId'),
                    'team': event.get('teamId'),
                    'x': positions[0].get('x', 0) if positions else 0,
                    'y': positions[0].get('y', 0) if positions else 0,
                },
                'end_position': {
                    'x': positions[1].get('x', 0) if len(positions) > 1 else None,
                    'y': positions[1].get('y', 0) if len(positions) > 1 else None,
                } if len(positions) > 1 else None
            }
            frames.append(frame)
        
        return {'events': frames}
    
    def convert_to_training_format(self, output_dir: str):
        """Convert Wyscout data to unified training format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        converted = 0
        for match_info in self.matches:
            if isinstance(match_info, dict) and 'path' in match_info:
                file_path = match_info['path']
                match_name = file_path.stem
                
                if 'tracking' in match_name.lower():
                    data = self.load_tracking(file_path)
                else:
                    events = self.load_events(file_path)
                    data = self.events_to_positions(events)
                
                output_file = output_path / f"{match_name}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f)
                
                converted += 1
                print(f"Converted {match_name}")
        
        print(f"\n✓ Converted {converted} files to {output_dir}")


# Coordinate conversion utilities
def wyscout_to_meters(x: float, y: float, pitch_length: float = 105, pitch_width: float = 68) -> tuple:
    """Convert Wyscout 0-100 coords to meters."""
    return (x / 100 * pitch_length, y / 100 * pitch_width)


def meters_to_wyscout(x: float, y: float, pitch_length: float = 105, pitch_width: float = 68) -> tuple:
    """Convert meters to Wyscout 0-100 coords."""
    return (x / pitch_length * 100, y / pitch_width * 100)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Wyscout data directory')
    parser.add_argument('--output-dir', default='data/wyscout_converted', help='Output directory')
    parser.add_argument('--convert', action='store_true', help='Convert to training format')
    args = parser.parse_args()
    
    dataset = WyscoutDataset(args.data_dir)
    
    if args.convert:
        dataset.convert_to_training_format(args.output_dir)
