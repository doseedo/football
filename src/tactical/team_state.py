"""Team state classification and possession detection."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .gnn_model import TacticalGNN, StateClassificationHead


class TacticalState(Enum):
    """Tactical phase/state enumeration."""
    ATTACKING = 0
    DEFENDING = 1
    TRANSITION_ATTACK = 2  # Counterattack
    TRANSITION_DEFENSE = 3  # Losing possession
    SET_PIECE = 4


@dataclass
class TeamState:
    """Complete team state representation."""
    state: TacticalState
    confidence: float
    possession_team: int  # 0 or 1
    possession_probability: Dict[int, float]
    pressing_intensity: float  # 0-1
    defensive_line_height: float  # meters from own goal
    team_compactness: float  # meters (average distance between players)
    ball_position: Tuple[float, float]
    timestamp: float


class TeamStateClassifier:
    """
    Classifies current tactical phase and estimates team dynamics.

    Uses GNN embeddings to understand team coordination and tactical state.
    """

    # Thresholds for state classification
    DEFENSIVE_THIRD = 35.0  # meters from own goal
    ATTACKING_THIRD = 70.0  # meters from own goal
    PRESSING_DISTANCE = 10.0  # meters to ball for pressing
    TRANSITION_VELOCITY_THRESHOLD = 3.0  # m/s average team velocity

    def __init__(self,
                 gnn_model: TacticalGNN,
                 use_gnn_classifier: bool = True,
                 device: str = 'cpu'):
        """
        Initialize team state classifier.

        Args:
            gnn_model: Trained TacticalGNN model
            use_gnn_classifier: Use GNN-based classification (vs. rule-based)
            device: PyTorch device
        """
        self.gnn_model = gnn_model
        self.use_gnn_classifier = use_gnn_classifier
        self.device = device

        # GNN-based state classifier
        if use_gnn_classifier:
            self.state_classifier = StateClassificationHead(
                input_dim=gnn_model.output_dim,
                num_states=len(TacticalState),
                hidden_dim=128
            ).to(device)
        else:
            self.state_classifier = None

        # Move model to device
        self.gnn_model.to(device)
        self.gnn_model.eval()

    def classify(self, graph: Data, timestamp: float = 0.0) -> TeamState:
        """
        Classify current tactical state from graph.

        Args:
            graph: PyTorch Geometric Data object
            timestamp: Current timestamp

        Returns:
            TeamState object with complete state information
        """
        with torch.no_grad():
            graph = graph.to(self.device)

            # Get GNN embeddings
            node_embeddings, graph_embedding = self.gnn_model(graph)

            # Classify tactical state
            if self.use_gnn_classifier and self.state_classifier is not None:
                state, confidence = self._classify_with_gnn(graph_embedding)
            else:
                state, confidence = self._classify_rule_based(graph)

            # Possession detection
            possession_team, possession_probs = self.get_possession_probabilities(graph, node_embeddings)

            # Pressing intensity
            pressing_intensity = self.get_pressing_intensity(graph, node_embeddings)

            # Defensive line height
            defensive_line_height = self._calculate_defensive_line_height(graph)

            # Team compactness
            team_compactness = self._calculate_team_compactness(graph)

            # Ball position
            if graph.ball_position is not None:
                ball_pos = (graph.ball_position[0].item(), graph.ball_position[1].item())
            else:
                ball_pos = (0.0, 0.0)

            return TeamState(
                state=state,
                confidence=confidence,
                possession_team=possession_team,
                possession_probability=possession_probs,
                pressing_intensity=pressing_intensity,
                defensive_line_height=defensive_line_height,
                team_compactness=team_compactness,
                ball_position=ball_pos,
                timestamp=timestamp
            )

    def _classify_with_gnn(self, graph_embedding: torch.Tensor) -> Tuple[TacticalState, float]:
        """Classify state using GNN-based classifier."""
        logits = self.state_classifier(graph_embedding.unsqueeze(0) if graph_embedding.dim() == 1 else graph_embedding)
        probs = F.softmax(logits, dim=-1)

        state_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, state_idx].item()

        state = TacticalState(state_idx)
        return state, confidence

    def _classify_rule_based(self, graph: Data) -> Tuple[TacticalState, float]:
        """
        Classify state using rule-based heuristics.

        Rules based on:
        - Ball position (defensive/middle/attacking third)
        - Team velocities (transition detection)
        - Player positions relative to ball
        """
        if graph.ball_position is None:
            return TacticalState.DEFENDING, 0.5

        ball_x = graph.ball_position[0].item()
        teams = graph.teams
        positions = graph.x[:, :2].cpu().numpy()  # Normalized positions

        # Denormalize positions
        positions[:, 0] = positions[:, 0] * 105.0 - 52.5  # x
        positions[:, 1] = positions[:, 1] * 68.0 - 34.0   # y

        # Determine possession team (simple: closest to ball)
        ball_pos_np = graph.ball_position.cpu().numpy()
        distances_to_ball = np.linalg.norm(positions - ball_pos_np, axis=1)
        closest_player_idx = np.argmin(distances_to_ball)
        possession_team = teams[closest_player_idx].item()

        if possession_team == -1:
            return TacticalState.DEFENDING, 0.5

        # Calculate average velocities for transition detection
        velocities = graph.x[:, 2:4].cpu().numpy()
        team_velocities = {}
        for team_id in [0, 1]:
            team_mask = teams == team_id
            if team_mask.sum() > 0:
                team_vels = velocities[team_mask.cpu().numpy()]
                avg_speed = np.mean(np.linalg.norm(team_vels, axis=1))
                team_velocities[team_id] = avg_speed
            else:
                team_velocities[team_id] = 0.0

        # Classify based on rules
        if team_velocities.get(possession_team, 0) > self.TRANSITION_VELOCITY_THRESHOLD:
            # High velocity - likely transition
            if ball_x > self.ATTACKING_THIRD:
                return TacticalState.TRANSITION_ATTACK, 0.7
            else:
                return TacticalState.TRANSITION_ATTACK, 0.6

        # Check for set piece (low velocity, ball static)
        ball_has_velocity = hasattr(graph, 'ball_velocity') and graph.ball_velocity is not None
        if ball_has_velocity:
            ball_vel = torch.norm(graph.ball_velocity).item()
            if ball_vel < 0.5 and all(v < 1.0 for v in team_velocities.values()):
                return TacticalState.SET_PIECE, 0.8

        # Position-based classification
        if ball_x > self.ATTACKING_THIRD:
            return TacticalState.ATTACKING, 0.7
        elif ball_x < -self.DEFENSIVE_THIRD:
            return TacticalState.DEFENDING, 0.7
        else:
            # Middle third - check pressing
            pressing = self._detect_pressing_simple(graph)
            if pressing:
                return TacticalState.DEFENDING, 0.6
            else:
                return TacticalState.ATTACKING, 0.5

    def get_possession_probabilities(self,
                                    graph: Data,
                                    node_embeddings: Optional[torch.Tensor] = None) -> Tuple[int, Dict[int, float]]:
        """
        Determine possession team and confidence.

        Args:
            graph: Graph data
            node_embeddings: Optional node embeddings from GNN

        Returns:
            possession_team: Team ID (0 or 1)
            probabilities: Dict mapping team ID to possession probability
        """
        if graph.ball_position is None:
            return 0, {0: 0.5, 1: 0.5}

        # Simple method: closest player to ball
        ball_pos = graph.ball_position.cpu().numpy()
        positions = graph.x[:, :2].cpu().numpy()

        # Denormalize
        positions[:, 0] = positions[:, 0] * 105.0 - 52.5
        positions[:, 1] = positions[:, 1] * 68.0 - 34.0

        distances = np.linalg.norm(positions - ball_pos, axis=1)
        teams = graph.teams.cpu().numpy()

        # Find closest players from each team
        team_distances = {0: [], 1: []}
        for i, (dist, team) in enumerate(zip(distances, teams)):
            if team in [0, 1]:
                team_distances[team].append(dist)

        # Calculate probabilities based on proximity
        probs = {}
        for team_id in [0, 1]:
            if len(team_distances[team_id]) > 0:
                # Closest distance and average of 3 closest
                closest = sorted(team_distances[team_id])[:3]
                avg_dist = np.mean(closest)
                # Inverse distance (closer = higher probability)
                probs[team_id] = 1.0 / (1.0 + avg_dist)
            else:
                probs[team_id] = 0.0

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs = {0: 0.5, 1: 0.5}

        # Determine possession team
        possession_team = max(probs.items(), key=lambda x: x[1])[0]

        return possession_team, probs

    def get_pressing_intensity(self,
                               graph: Data,
                               node_embeddings: Optional[torch.Tensor] = None) -> float:
        """
        Estimate pressing intensity (0-1).

        Based on:
        - Number of defenders near ball
        - Defensive team compactness
        - Velocity of defensive players toward ball

        Args:
            graph: Graph data
            node_embeddings: Optional node embeddings

        Returns:
            Pressing intensity in [0, 1]
        """
        if graph.ball_position is None:
            return 0.0

        ball_pos = graph.ball_position.cpu().numpy()
        positions = graph.x[:, :2].cpu().numpy()
        velocities = graph.x[:, 2:4].cpu().numpy()
        teams = graph.teams.cpu().numpy()

        # Denormalize positions
        positions[:, 0] = positions[:, 0] * 105.0 - 52.5
        positions[:, 1] = positions[:, 1] * 68.0 - 34.0

        # Find possession team
        distances = np.linalg.norm(positions - ball_pos, axis=1)
        closest_idx = np.argmin(distances)
        possession_team = teams[closest_idx]

        if possession_team == -1:
            return 0.0

        # Defensive team (opposite of possession)
        defensive_team = 1 - possession_team

        # Count defenders near ball
        defensive_mask = teams == defensive_team
        defensive_positions = positions[defensive_mask]
        defensive_velocities = velocities[defensive_mask]

        if len(defensive_positions) == 0:
            return 0.0

        defensive_distances = np.linalg.norm(defensive_positions - ball_pos, axis=1)

        # Pressing factors
        # 1. Number of defenders within pressing distance
        num_pressing = np.sum(defensive_distances < self.PRESSING_DISTANCE)
        pressing_count_score = min(num_pressing / 3.0, 1.0)  # Normalize by 3 players

        # 2. Average distance of closest defenders
        closest_defenders = np.sort(defensive_distances)[:5]
        avg_defender_distance = np.mean(closest_defenders)
        distance_score = max(0, 1.0 - avg_defender_distance / 15.0)

        # 3. Velocity toward ball
        if len(defensive_velocities) > 0:
            # Calculate if moving toward ball
            to_ball_vectors = ball_pos - defensive_positions
            to_ball_vectors = to_ball_vectors / (np.linalg.norm(to_ball_vectors, axis=1, keepdims=True) + 1e-6)
            velocity_alignment = np.sum(defensive_velocities * to_ball_vectors, axis=1)
            avg_velocity_toward_ball = np.mean(np.maximum(velocity_alignment, 0))
            velocity_score = min(avg_velocity_toward_ball / 2.0, 1.0)
        else:
            velocity_score = 0.0

        # Combine factors
        pressing_intensity = (pressing_count_score * 0.4 +
                            distance_score * 0.4 +
                            velocity_score * 0.2)

        return float(np.clip(pressing_intensity, 0, 1))

    def _detect_pressing_simple(self, graph: Data) -> bool:
        """Simple pressing detection."""
        intensity = self.get_pressing_intensity(graph)
        return intensity > 0.6

    def _calculate_defensive_line_height(self, graph: Data) -> float:
        """
        Calculate defensive line height (distance from own goal).

        Returns average position of deepest 4 defenders.
        """
        positions = graph.x[:, :2].cpu().numpy()
        teams = graph.teams.cpu().numpy()

        # Denormalize x position
        positions[:, 0] = positions[:, 0] * 105.0 - 52.5

        defensive_lines = {}
        for team_id in [0, 1]:
            team_mask = teams == team_id
            if team_mask.sum() == 0:
                continue

            team_positions = positions[team_mask]

            # For team 0, defensive line is leftmost (lowest x)
            # For team 1, defensive line is rightmost (highest x)
            if team_id == 0:
                sorted_x = np.sort(team_positions[:, 0])[:4]  # 4 deepest defenders
                defensive_line = np.mean(sorted_x)
                # Distance from own goal (-52.5)
                defensive_lines[team_id] = defensive_line + 52.5
            else:
                sorted_x = np.sort(team_positions[:, 0])[-4:]  # 4 deepest defenders
                defensive_line = np.mean(sorted_x)
                # Distance from own goal (52.5)
                defensive_lines[team_id] = 52.5 - defensive_line

        # Return average of both teams
        if len(defensive_lines) > 0:
            return float(np.mean(list(defensive_lines.values())))
        else:
            return 0.0

    def _calculate_team_compactness(self, graph: Data) -> float:
        """
        Calculate team compactness (average distance between teammates).

        Lower values = more compact team.
        """
        positions = graph.x[:, :2].cpu().numpy()
        teams = graph.teams.cpu().numpy()

        # Denormalize
        positions[:, 0] = positions[:, 0] * 105.0 - 52.5
        positions[:, 1] = positions[:, 1] * 68.0 - 34.0

        compactness_values = []
        for team_id in [0, 1]:
            team_mask = teams == team_id
            team_positions = positions[team_mask]

            if len(team_positions) < 2:
                continue

            # Calculate average pairwise distance
            from scipy.spatial.distance import pdist
            distances = pdist(team_positions)
            avg_distance = np.mean(distances)
            compactness_values.append(avg_distance)

        if len(compactness_values) > 0:
            return float(np.mean(compactness_values))
        else:
            return 0.0

    def train_classifier(self, train_loader, val_loader, num_epochs: int = 50, lr: float = 0.001):
        """
        Train the GNN-based state classifier.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            lr: Learning rate
        """
        if not self.use_gnn_classifier or self.state_classifier is None:
            raise ValueError("GNN classifier not initialized")

        optimizer = torch.optim.Adam(
            list(self.gnn_model.parameters()) + list(self.state_classifier.parameters()),
            lr=lr
        )
        criterion = torch.nn.CrossEntropyLoss()

        self.gnn_model.train()
        self.state_classifier.train()

        for epoch in range(num_epochs):
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                _, graph_embedding = self.gnn_model(batch)
                logits = self.state_classifier(graph_embedding)

                # Compute loss
                loss = criterion(logits, batch.y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            val_loss, val_acc = self._validate(val_loader, criterion)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        self.gnn_model.eval()
        self.state_classifier.eval()

    def _validate(self, val_loader, criterion):
        """Validation step."""
        self.gnn_model.eval()
        self.state_classifier.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                _, graph_embedding = self.gnn_model(batch)
                logits = self.state_classifier(graph_embedding)

                loss = criterion(logits, batch.y)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)

        self.gnn_model.train()
        self.state_classifier.train()

        return val_loss / len(val_loader), correct / total if total > 0 else 0.0
