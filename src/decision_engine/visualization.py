"""
Simple pitch visualization for Decision Engine.

Renders players as dots on a 2D pitch with optional overlays.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

from .engine import (
    PlayerState, DefensiveGap, ProgressionOption,
    DecisionEngineOutput, PITCH_LENGTH, PITCH_WIDTH
)


class PitchVisualizer:
    """
    Simple 2D pitch visualization.

    Draws:
    - Pitch markings
    - Players as colored dots
    - Ball
    - Optional: gaps, passing lanes, xG overlay
    """

    def __init__(self,
                 figsize: Tuple[int, int] = (12, 8),
                 team_colors: Tuple[str, str] = ('blue', 'red'),
                 ball_color: str = 'white',
                 show_jerseys: bool = True):
        """
        Args:
            figsize: Figure size (width, height)
            team_colors: Colors for team 0 and team 1
            ball_color: Ball color
            show_jerseys: Show jersey numbers on players
        """
        self.figsize = figsize
        self.team_colors = team_colors
        self.ball_color = ball_color
        self.show_jerseys = show_jerseys

        self.fig = None
        self.ax = None

    def setup_pitch(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create figure and draw pitch markings."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # Set background
        self.ax.set_facecolor('#2e8b57')  # Grass green

        # Set limits
        self.ax.set_xlim(-PITCH_LENGTH/2 - 5, PITCH_LENGTH/2 + 5)
        self.ax.set_ylim(-PITCH_WIDTH/2 - 5, PITCH_WIDTH/2 + 5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Draw pitch markings
        self._draw_pitch_markings()

        return self.fig, self.ax

    def _draw_pitch_markings(self):
        """Draw standard pitch markings."""
        white = 'white'
        lw = 2

        # Outer boundary
        self.ax.plot([-PITCH_LENGTH/2, PITCH_LENGTH/2], [-PITCH_WIDTH/2, -PITCH_WIDTH/2], white, lw=lw)
        self.ax.plot([-PITCH_LENGTH/2, PITCH_LENGTH/2], [PITCH_WIDTH/2, PITCH_WIDTH/2], white, lw=lw)
        self.ax.plot([-PITCH_LENGTH/2, -PITCH_LENGTH/2], [-PITCH_WIDTH/2, PITCH_WIDTH/2], white, lw=lw)
        self.ax.plot([PITCH_LENGTH/2, PITCH_LENGTH/2], [-PITCH_WIDTH/2, PITCH_WIDTH/2], white, lw=lw)

        # Halfway line
        self.ax.plot([0, 0], [-PITCH_WIDTH/2, PITCH_WIDTH/2], white, lw=lw)

        # Center circle
        center_circle = plt.Circle((0, 0), 9.15, fill=False, color=white, lw=lw)
        self.ax.add_patch(center_circle)

        # Center spot
        self.ax.plot(0, 0, 'o', color=white, markersize=3)

        # Penalty areas (16.5m from goal line, 40.3m wide)
        box_width = 40.3
        box_depth = 16.5

        # Left penalty area
        self.ax.plot([-PITCH_LENGTH/2, -PITCH_LENGTH/2 + box_depth],
                    [-box_width/2, -box_width/2], white, lw=lw)
        self.ax.plot([-PITCH_LENGTH/2, -PITCH_LENGTH/2 + box_depth],
                    [box_width/2, box_width/2], white, lw=lw)
        self.ax.plot([-PITCH_LENGTH/2 + box_depth, -PITCH_LENGTH/2 + box_depth],
                    [-box_width/2, box_width/2], white, lw=lw)

        # Right penalty area
        self.ax.plot([PITCH_LENGTH/2, PITCH_LENGTH/2 - box_depth],
                    [-box_width/2, -box_width/2], white, lw=lw)
        self.ax.plot([PITCH_LENGTH/2, PITCH_LENGTH/2 - box_depth],
                    [box_width/2, box_width/2], white, lw=lw)
        self.ax.plot([PITCH_LENGTH/2 - box_depth, PITCH_LENGTH/2 - box_depth],
                    [-box_width/2, box_width/2], white, lw=lw)

        # Goal areas (5.5m from goal line, 18.3m wide)
        goal_width = 18.3
        goal_depth = 5.5

        # Left goal area
        self.ax.plot([-PITCH_LENGTH/2, -PITCH_LENGTH/2 + goal_depth],
                    [-goal_width/2, -goal_width/2], white, lw=lw)
        self.ax.plot([-PITCH_LENGTH/2, -PITCH_LENGTH/2 + goal_depth],
                    [goal_width/2, goal_width/2], white, lw=lw)
        self.ax.plot([-PITCH_LENGTH/2 + goal_depth, -PITCH_LENGTH/2 + goal_depth],
                    [-goal_width/2, goal_width/2], white, lw=lw)

        # Right goal area
        self.ax.plot([PITCH_LENGTH/2, PITCH_LENGTH/2 - goal_depth],
                    [-goal_width/2, -goal_width/2], white, lw=lw)
        self.ax.plot([PITCH_LENGTH/2, PITCH_LENGTH/2 - goal_depth],
                    [goal_width/2, goal_width/2], white, lw=lw)
        self.ax.plot([PITCH_LENGTH/2 - goal_depth, PITCH_LENGTH/2 - goal_depth],
                    [-goal_width/2, goal_width/2], white, lw=lw)

        # Penalty spots
        self.ax.plot(-PITCH_LENGTH/2 + 11, 0, 'o', color=white, markersize=3)
        self.ax.plot(PITCH_LENGTH/2 - 11, 0, 'o', color=white, markersize=3)

        # Goals
        goal_width_actual = 7.32
        self.ax.plot([-PITCH_LENGTH/2 - 2, -PITCH_LENGTH/2],
                    [-goal_width_actual/2, -goal_width_actual/2], white, lw=3)
        self.ax.plot([-PITCH_LENGTH/2 - 2, -PITCH_LENGTH/2],
                    [goal_width_actual/2, goal_width_actual/2], white, lw=3)
        self.ax.plot([-PITCH_LENGTH/2 - 2, -PITCH_LENGTH/2 - 2],
                    [-goal_width_actual/2, goal_width_actual/2], white, lw=3)

        self.ax.plot([PITCH_LENGTH/2 + 2, PITCH_LENGTH/2],
                    [-goal_width_actual/2, -goal_width_actual/2], white, lw=3)
        self.ax.plot([PITCH_LENGTH/2 + 2, PITCH_LENGTH/2],
                    [goal_width_actual/2, goal_width_actual/2], white, lw=3)
        self.ax.plot([PITCH_LENGTH/2 + 2, PITCH_LENGTH/2 + 2],
                    [-goal_width_actual/2, goal_width_actual/2], white, lw=3)

    def draw_frame(self,
                   players: List[PlayerState],
                   ball_position: Tuple[float, float],
                   engine_output: Optional[DecisionEngineOutput] = None,
                   show_gaps: bool = False,
                   show_best_option: bool = False,
                   title: str = None):
        """
        Draw a single frame with players and ball.

        Args:
            players: List of all player states
            ball_position: Ball (x, y) position
            engine_output: Optional decision engine analysis
            show_gaps: Highlight defensive gaps
            show_best_option: Show best progression option
            title: Optional title
        """
        if self.fig is None:
            self.setup_pitch()

        # Clear previous frame's dynamic elements
        for artist in list(self.ax.patches) + list(self.ax.texts):
            if hasattr(artist, '_dynamic'):
                artist.remove()

        for line in list(self.ax.lines):
            if hasattr(line, '_dynamic'):
                line.remove()

        # Draw players
        for player in players:
            color = self.team_colors[player.team]
            x, y = player.position

            # Player dot
            circle = plt.Circle((x, y), 1.5, color=color, ec='white', lw=1.5, zorder=10)
            circle._dynamic = True
            self.ax.add_patch(circle)

            # Jersey number
            if self.show_jerseys:
                text = self.ax.text(x, y, str(player.player_id),
                                   ha='center', va='center',
                                   fontsize=8, fontweight='bold', color='white', zorder=11)
                text._dynamic = True

        # Draw ball
        ball = plt.Circle(ball_position, 1.0, color=self.ball_color, ec='black', lw=2, zorder=15)
        ball._dynamic = True
        self.ax.add_patch(ball)

        # Draw gaps if requested
        if show_gaps and engine_output:
            self._draw_gaps(engine_output.defensive_gaps)

        # Draw best option if requested
        if show_best_option and engine_output and engine_output.best_option:
            self._draw_option(ball_position, engine_output.best_option)

        # Title
        if title:
            self.ax.set_title(title, fontsize=14, color='white', pad=10)

        return self.fig

    def _draw_gaps(self, gaps: List[DefensiveGap]):
        """Draw defensive gaps as highlighted zones."""
        for gap in gaps:
            if gap.exploitable:
                # Draw gap as a semi-transparent circle
                x, y = gap.location
                circle = plt.Circle((x, y), gap.size / 2,
                                   color='yellow', alpha=0.3, zorder=5)
                circle._dynamic = True
                self.ax.add_patch(circle)

                # Gap label
                text = self.ax.text(x, y + gap.size/2 + 2,
                                   f"Gap: {gap.size:.1f}m\nxG: {gap.xg_if_exploited:.2f}",
                                   ha='center', va='bottom', fontsize=8,
                                   color='yellow', zorder=5)
                text._dynamic = True

    def _draw_option(self, ball_pos: Tuple[float, float], option: ProgressionOption):
        """Draw a progression option as an arrow."""
        x1, y1 = ball_pos
        x2, y2 = option.target_position

        # Arrow color based on recommendation
        colors = {
            'HIGH_VALUE': 'lime',
            'SAFE': 'cyan',
            'MODERATE': 'yellow',
            'LOW_VALUE': 'orange',
            'AVOID': 'red'
        }
        color = colors.get(option.recommendation, 'white')

        # Draw arrow
        arrow = self.ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                arrowprops=dict(arrowstyle='->', color=color, lw=2),
                                zorder=20)
        arrow._dynamic = True

        # Label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        text = self.ax.text(mid_x, mid_y + 3,
                           f"P: {option.success_probability:.0%}\nEV: {option.expected_value:+.3f}",
                           ha='center', va='bottom', fontsize=8,
                           color=color, fontweight='bold', zorder=20)
        text._dynamic = True

    def draw_analysis_panel(self, engine_output: DecisionEngineOutput):
        """Draw a side panel with decision engine analysis."""
        # Create text summary
        text_lines = [
            "DECISION ENGINE ANALYSIS",
            "=" * 30,
            f"Ball: ({engine_output.ball_position[0]:.1f}, {engine_output.ball_position[1]:.1f})",
            f"Carrier: #{engine_output.ball_carrier_id}",
            "",
            f"Defensive Gaps: {len(engine_output.defensive_gaps)}",
        ]

        for i, gap in enumerate(engine_output.defensive_gaps[:3]):
            text_lines.append(f"  {i+1}. {gap.size:.1f}m gap, xG: {gap.xg_if_exploited:.2f}")

        text_lines.extend([
            "",
            f"Options: {engine_output.total_options}",
            f"  High Value: {engine_output.high_value_options}",
            f"  Safe: {engine_output.safe_options}",
            "",
            "BEST OPTION:",
        ])

        if engine_output.best_option:
            opt = engine_output.best_option
            text_lines.extend([
                f"  Type: {opt.action_type}",
                f"  Target: {opt.target_position}",
                f"  Success: {opt.success_probability:.0%}",
                f"  xG Gain: {opt.xg_gain:+.3f}",
                f"  EV: {opt.expected_value:+.3f}",
                f"  → {opt.recommendation}"
            ])

        # Draw text box
        text = "\n".join(text_lines)
        props = dict(boxstyle='round', facecolor='black', alpha=0.8)
        self.ax.text(PITCH_LENGTH/2 + 8, PITCH_WIDTH/2, text,
                    transform=self.ax.transData,
                    fontsize=9, verticalalignment='top', family='monospace',
                    color='white', bbox=props, zorder=100)

    def save(self, filepath: str, dpi: int = 150):
        """Save current figure to file."""
        if self.fig:
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                           facecolor='#2e8b57', edgecolor='none')

    def show(self):
        """Display the figure."""
        if self.fig:
            plt.show()

    def close(self):
        """Close the figure."""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


class MatchAnimator:
    """
    Animates a sequence of frames showing player movement.
    """

    def __init__(self, visualizer: PitchVisualizer):
        self.visualizer = visualizer
        self.frames_data = []

    def add_frame(self,
                  players: List[PlayerState],
                  ball_position: Tuple[float, float],
                  engine_output: Optional[DecisionEngineOutput] = None,
                  timestamp: float = 0.0):
        """Add a frame to the animation."""
        self.frames_data.append({
            'players': players,
            'ball_position': ball_position,
            'engine_output': engine_output,
            'timestamp': timestamp
        })

    def animate(self, interval: int = 100, show_gaps: bool = True,
                show_best_option: bool = True) -> FuncAnimation:
        """
        Create animation from frames.

        Args:
            interval: Milliseconds between frames
            show_gaps: Show defensive gaps
            show_best_option: Show best progression option

        Returns:
            matplotlib FuncAnimation object
        """
        if not self.frames_data:
            raise ValueError("No frames to animate")

        self.visualizer.setup_pitch()

        def update(frame_idx):
            frame = self.frames_data[frame_idx]
            self.visualizer.draw_frame(
                frame['players'],
                frame['ball_position'],
                frame['engine_output'],
                show_gaps=show_gaps,
                show_best_option=show_best_option,
                title=f"Time: {frame['timestamp']:.1f}s"
            )

        anim = FuncAnimation(
            self.visualizer.fig,
            update,
            frames=len(self.frames_data),
            interval=interval,
            repeat=True
        )

        return anim

    def save_animation(self, filepath: str, fps: int = 10):
        """Save animation to file."""
        anim = self.animate()
        anim.save(filepath, writer='pillow', fps=fps)
