"""Convert tracking data to graph representations for GNN processing."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from scipy.spatial.distance import cdist


@dataclass
class PlayerNode:
    """Represents a player node in the graph."""
    track_id: int
    position: Tuple[float, float]  # (x, y) in meters
    velocity: Tuple[float, float]  # (vx, vy) in m/s
    team: int  # 0, 1, or -1 for unknown
    role: str = "player"  # player, goalkeeper, referee


@dataclass
class TemporalGraph:
    """Sequence of graphs for temporal modeling."""
    graphs: List[Data]
    timestamps: List[float]

    def to_batch(self) -> Batch:
        """Convert to PyTorch Geometric batch."""
        return Batch.from_data_list(self.graphs)


class TrackingGraphBuilder:
    """Builds graph representations from tracking data."""

    # Graph construction parameters
    PROXIMITY_THRESHOLD = 15.0  # meters - edges within this distance
    BALL_PROXIMITY_THRESHOLD = 5.0  # meters - ball proximity feature
    PITCH_LENGTH = 105.0  # meters
    PITCH_WIDTH = 68.0  # meters

    def __init__(self,
                 proximity_threshold: float = 15.0,
                 use_temporal: bool = True,
                 edge_type: str = 'proximity'):  # proximity, team, or full
        """
        Initialize graph builder.

        Args:
            proximity_threshold: Max distance for proximity edges (meters)
            use_temporal: Include temporal connections
            edge_type: Type of edges - 'proximity', 'team', or 'full'
        """
        self.proximity_threshold = proximity_threshold
        self.use_temporal = use_temporal
        self.edge_type = edge_type
        self.prev_graph = None

    def build_graph(self, frame_data: Dict) -> Data:
        """
        Build graph from single frame tracking data.

        Args:
            frame_data: Dict with 'players' and 'ball' keys

        Returns:
            PyTorch Geometric Data object
        """
        players = frame_data.get('players', [])
        ball = frame_data.get('ball')

        if len(players) == 0:
            # Return empty graph
            return Data(x=torch.zeros((0, 12)), edge_index=torch.zeros((2, 0), dtype=torch.long))

        # Extract player nodes
        nodes = []
        for p in players:
            pos = (p.get('x', 0), p.get('y', 0))
            vel = (p.get('vx', 0), p.get('vy', 0))
            nodes.append(PlayerNode(
                track_id=p.get('track_id', -1),
                position=pos,
                velocity=vel,
                team=p.get('team', -1)
            ))

        # Build node features
        node_features = self._get_player_features_batch(nodes, ball)

        # Build edges
        edge_index, edge_features = self._build_edges(nodes, ball)

        # Create graph
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            num_nodes=len(nodes)
        )

        # Add metadata
        graph.ball_position = torch.tensor([ball['x'], ball['y']], dtype=torch.float) if ball else None
        graph.track_ids = [n.track_id for n in nodes]
        graph.teams = torch.tensor([n.team for n in nodes], dtype=torch.long)

        self.prev_graph = graph
        return graph

    def build_temporal_graph(self, frame_sequence: List[Dict]) -> TemporalGraph:
        """
        Build temporal graph from sequence of frames.

        Args:
            frame_sequence: List of frame data dicts

        Returns:
            TemporalGraph with sequence of graphs
        """
        graphs = []
        timestamps = []

        for frame in frame_sequence:
            graph = self.build_graph(frame)
            graphs.append(graph)
            timestamps.append(frame.get('timestamp', 0))

        return TemporalGraph(graphs=graphs, timestamps=timestamps)

    def get_player_features(self, player: PlayerNode, ball_pos: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Extract feature vector for single player.

        Features (12D):
        - Position (x, y) normalized [0-1]
        - Velocity (vx, vy) in m/s
        - Speed (scalar)
        - Distance to ball
        - Angle to ball
        - Team (one-hot: [team_0, team_1, unknown])
        - Position zone (defensive/middle/attacking)
        - Distance to own goal

        Args:
            player: PlayerNode object
            ball_pos: Ball position (x, y)

        Returns:
            12D feature tensor
        """
        x, y = player.position
        vx, vy = player.velocity

        # Normalize position to [0, 1]
        x_norm = (x + self.PITCH_LENGTH / 2) / self.PITCH_LENGTH
        y_norm = (y + self.PITCH_WIDTH / 2) / self.PITCH_WIDTH

        # Speed
        speed = np.sqrt(vx**2 + vy**2)

        # Ball-related features
        if ball_pos:
            ball_x, ball_y = ball_pos
            dist_to_ball = np.sqrt((x - ball_x)**2 + (y - ball_y)**2)
            angle_to_ball = np.arctan2(ball_y - y, ball_x - x)
        else:
            dist_to_ball = 0
            angle_to_ball = 0

        # Team (one-hot)
        team_0 = 1.0 if player.team == 0 else 0.0
        team_1 = 1.0 if player.team == 1 else 0.0
        team_unknown = 1.0 if player.team == -1 else 0.0

        # Position zone (thirds of pitch)
        zone = x / (self.PITCH_LENGTH / 3)  # -1.5 to 1.5

        # Distance to own goal (depends on team)
        if player.team == 0:
            dist_to_goal = abs(x + self.PITCH_LENGTH / 2)
        elif player.team == 1:
            dist_to_goal = abs(x - self.PITCH_LENGTH / 2)
        else:
            dist_to_goal = 0

        features = torch.tensor([
            x_norm, y_norm,  # Position
            vx, vy,  # Velocity
            speed,  # Speed
            dist_to_ball / self.PITCH_LENGTH,  # Normalized distance to ball
            angle_to_ball / np.pi,  # Normalized angle to ball
            team_0, team_1, team_unknown,  # Team one-hot
            zone,  # Position zone
            dist_to_goal / self.PITCH_LENGTH  # Normalized distance to goal
        ], dtype=torch.float)

        return features

    def _get_player_features_batch(self, nodes: List[PlayerNode],
                                   ball: Optional[Dict] = None) -> torch.Tensor:
        """Extract features for all players in batch."""
        ball_pos = (ball['x'], ball['y']) if ball else None
        features = [self.get_player_features(node, ball_pos) for node in nodes]
        return torch.stack(features)

    def _build_edges(self, nodes: List[PlayerNode],
                    ball: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edge connectivity and features.

        Returns:
            edge_index: [2, num_edges] tensor
            edge_attr: [num_edges, edge_dim] tensor
        """
        n = len(nodes)
        positions = np.array([node.position for node in nodes])
        teams = np.array([node.team for node in nodes])

        if self.edge_type == 'full':
            # Fully connected graph
            sources = []
            targets = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        sources.append(i)
                        targets.append(j)
            edge_index = torch.tensor([sources, targets], dtype=torch.long)

        elif self.edge_type == 'team':
            # Connect players on same team
            sources = []
            targets = []
            for i in range(n):
                for j in range(n):
                    if i != j and teams[i] == teams[j] and teams[i] != -1:
                        sources.append(i)
                        targets.append(j)
            edge_index = torch.tensor([sources, targets], dtype=torch.long) if sources else torch.zeros((2, 0), dtype=torch.long)

        else:  # proximity
            # Connect based on spatial proximity
            distances = cdist(positions, positions)
            sources = []
            targets = []
            for i in range(n):
                for j in range(n):
                    if i != j and distances[i, j] < self.proximity_threshold:
                        sources.append(i)
                        targets.append(j)

            # Also connect same-team players within larger radius
            for i in range(n):
                for j in range(n):
                    if i != j and teams[i] == teams[j] and teams[i] != -1:
                        if distances[i, j] < self.proximity_threshold * 1.5:
                            if i not in sources or j not in targets:
                                sources.append(i)
                                targets.append(j)

            edge_index = torch.tensor([sources, targets], dtype=torch.long) if sources else torch.zeros((2, 0), dtype=torch.long)

        # Build edge features
        edge_features = []
        if edge_index.shape[1] > 0:
            for i in range(edge_index.shape[1]):
                src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
                feat = self._get_edge_features(nodes[src], nodes[tgt])
                edge_features.append(feat)
            edge_attr = torch.stack(edge_features)
        else:
            edge_attr = torch.zeros((0, 5), dtype=torch.float)

        return edge_index, edge_attr

    def _get_edge_features(self, src: PlayerNode, tgt: PlayerNode) -> torch.Tensor:
        """
        Extract edge features between two players.

        Features (5D):
        - Relative position (dx, dy)
        - Distance
        - Same team (binary)
        - Relative velocity alignment

        Args:
            src: Source player node
            tgt: Target player node

        Returns:
            5D edge feature tensor
        """
        dx = tgt.position[0] - src.position[0]
        dy = tgt.position[1] - src.position[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Normalize relative position
        dx_norm = dx / self.PITCH_LENGTH
        dy_norm = dy / self.PITCH_WIDTH
        distance_norm = distance / np.sqrt(self.PITCH_LENGTH**2 + self.PITCH_WIDTH**2)

        # Same team
        same_team = 1.0 if src.team == tgt.team and src.team != -1 else 0.0

        # Relative velocity alignment (dot product)
        src_vel = np.array(src.velocity)
        tgt_vel = np.array(tgt.velocity)
        if np.linalg.norm(src_vel) > 0 and np.linalg.norm(tgt_vel) > 0:
            vel_alignment = np.dot(src_vel, tgt_vel) / (np.linalg.norm(src_vel) * np.linalg.norm(tgt_vel))
        else:
            vel_alignment = 0.0

        return torch.tensor([dx_norm, dy_norm, distance_norm, same_team, vel_alignment], dtype=torch.float)

    def reset(self):
        """Reset temporal state."""
        self.prev_graph = None
