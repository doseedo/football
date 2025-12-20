"""Export tracking data to various formats."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger


class TrackingDataExporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_frame_data(self, frame_data: List[Dict], filename: str, formats: List[str] = ["json", "csv"]):
        if "json" in formats:
            output_path = self.output_dir / f"{filename}.json"
            with open(output_path, 'w') as f:
                json.dump(frame_data, f, indent=2, default=str)
            logger.info(f"Exported JSON: {output_path}")

        if "csv" in formats:
            rows = []
            for frame in frame_data:
                frame_idx = frame.get('frame', 0)
                timestamp = frame.get('timestamp', 0)
                for player in frame.get('players', []):
                    rows.append({
                        'frame': frame_idx, 'timestamp': timestamp,
                        'track_id': player.get('track_id'), 'team': player.get('team'),
                        'jersey_number': player.get('jersey_number'),
                        'x': player.get('x'), 'y': player.get('y'),
                        'speed': player.get('speed'), 'acceleration': player.get('acceleration')
                    })
                ball = frame.get('ball')
                if ball:
                    rows.append({
                        'frame': frame_idx, 'timestamp': timestamp, 'track_id': 'ball',
                        'x': ball.get('x'), 'y': ball.get('y')
                    })
            df = pd.DataFrame(rows)
            output_path = self.output_dir / f"{filename}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Exported CSV: {output_path}")

    def export_metrics(self, metrics: List[Dict], filename: str):
        output_path = self.output_dir / f"{filename}_metrics.json"
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        df = pd.DataFrame(metrics)
        df.to_csv(self.output_dir / f"{filename}_metrics.csv", index=False)
        logger.info(f"Exported metrics: {output_path}")


def create_frame_record(frame_idx: int, timestamp: float, players: List[Dict], ball: Optional[Dict]) -> Dict:
    return {"frame": frame_idx, "timestamp": round(timestamp, 3), "players": players, "ball": ball}
