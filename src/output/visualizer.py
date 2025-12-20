"""Visualization of tracking data."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger

try:
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    HAS_MPLSOCCER = True
except ImportError:
    HAS_MPLSOCCER = False


class PitchVisualizer:
    COLORS = {'home': '#e63946', 'away': '#457b9d', 'ball': '#f4a261', 'referee': '#2a9d8f'}

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0, figsize: Tuple[int, int] = (12, 8)):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.figsize = figsize

    def plot_frame(self, players: List[Dict], ball: Optional[Dict] = None, title: str = "",
                   save_path: Optional[str] = None, show: bool = False):
        if HAS_MPLSOCCER:
            pitch = Pitch(pitch_type='custom', pitch_length=self.pitch_length, pitch_width=self.pitch_width,
                         line_color='white', pitch_color='grass')
            fig, ax = pitch.draw(figsize=self.figsize)

            for player in players:
                x, y = player.get('x', 0), player.get('y', 0)
                team = player.get('team', 'unknown')
                jersey = player.get('jersey_number')
                color = self.COLORS.get(team, '#888888')
                ax.scatter(x, y, c=color, s=200, edgecolors='white', linewidths=2, zorder=10)
                if jersey is not None:
                    ax.annotate(str(jersey), (x, y), ha='center', va='center',
                               fontsize=8, color='white', fontweight='bold', zorder=11)

            if ball:
                ax.scatter(ball['x'], ball['y'], c=self.COLORS['ball'], s=100,
                          edgecolors='black', linewidths=1, zorder=12)

            if title:
                ax.set_title(title, fontsize=14)
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            plt.close(fig)
        else:
            logger.warning("mplsoccer not available")

    def create_animation(self, frame_data: List[Dict], output_path: str, fps: int = 25):
        logger.info(f"Creating animation with {len(frame_data)} frames")
        temp_dir = Path(output_path).parent / "temp_frames"
        temp_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(frame_data):
            self.plot_frame(players=frame.get('players', []), ball=frame.get('ball'),
                           title=f"Frame {frame.get('frame', i)}",
                           save_path=str(temp_dir / f"frame_{i:06d}.png"))

        frame_files = sorted(temp_dir.glob("frame_*.png"))
        if len(frame_files) == 0:
            return

        first_frame = cv2.imread(str(frame_files[0]))
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame_file in frame_files:
            out.write(cv2.imread(str(frame_file)))
        out.release()

        for f in frame_files:
            f.unlink()
        temp_dir.rmdir()
        logger.info(f"Animation saved: {output_path}")
