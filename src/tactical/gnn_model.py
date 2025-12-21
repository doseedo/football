"""Graph Neural Network models for tactical analysis."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class TacticalGNN(nn.Module):
    """
    Graph Neural Network for tactical analysis.

    Uses message passing to learn player interactions and team coordination.
    Supports multiple GNN architectures: GCN, GraphSAGE, GAT.
    """

    def __init__(self,
                 input_dim: int = 12,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 output_dim: int = 64,
                 edge_dim: int = 5,
                 gnn_type: str = 'gat',
                 dropout: float = 0.1,
                 use_residual: bool = True,
                 num_heads: int = 4):
        """
        Initialize GNN model.

        Args:
            input_dim: Node feature dimension (default 12)
            hidden_dim: Hidden layer dimension
            num_layers: Number of graph conv layers
            output_dim: Output embedding dimension
            edge_dim: Edge feature dimension (default 5)
            gnn_type: Type of GNN - 'gcn', 'sage', or 'gat'
            dropout: Dropout probability
            use_residual: Use residual connections
            num_heads: Number of attention heads (for GAT)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.use_residual = use_residual

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim if i < num_layers - 1 else output_dim

            if gnn_type == 'gat':
                # Graph Attention Networks
                conv = GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels // num_heads if i < num_layers - 1 else out_channels,
                    heads=num_heads if i < num_layers - 1 else 1,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            elif gnn_type == 'sage':
                # GraphSAGE
                conv = SAGEConv(in_channels, out_channels)
            else:  # gcn
                # Graph Convolutional Network
                conv = GCNConv(in_channels, out_channels)

            self.convs.append(conv)

            # Batch normalization (not for last layer)
            if i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output projection for node embeddings
        self.node_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        # Graph-level embedding (for whole team/match state)
        self.graph_proj = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            node_embeddings: [num_nodes, output_dim] - per-player embeddings
            graph_embedding: [output_dim] - whole team/graph embedding
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x_in = x

            # Apply graph convolution
            if self.gnn_type == 'gat':
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            # Batch norm and activation (except last layer)
            if i < len(self.convs) - 1:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                # Residual connection
                if self.use_residual and x_in.size(-1) == x.size(-1):
                    x = x + x_in

        # Node embeddings
        node_embeddings = self.node_proj(x)

        # Graph-level embedding (mean + max pooling)
        graph_mean = global_mean_pool(node_embeddings, batch)
        graph_max = global_max_pool(node_embeddings, batch)
        graph_embedding = self.graph_proj(torch.cat([graph_mean, graph_max], dim=-1))

        # If single graph, squeeze batch dimension
        if graph_embedding.size(0) == 1:
            graph_embedding = graph_embedding.squeeze(0)

        return node_embeddings, graph_embedding

    def predict_state(self, data: Data, classifier: nn.Module) -> torch.Tensor:
        """
        Predict tactical state using graph embedding.

        Args:
            data: PyTorch Geometric Data object
            classifier: Classification head (nn.Module)

        Returns:
            State predictions/logits
        """
        _, graph_embedding = self.forward(data)
        return classifier(graph_embedding)


class StateClassificationHead(nn.Module):
    """Classification head for tactical state prediction."""

    def __init__(self, input_dim: int = 64, num_states: int = 5, hidden_dim: int = 128):
        """
        Initialize classification head.

        Args:
            input_dim: Input embedding dimension
            num_states: Number of tactical states
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_states)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(x)


class TemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network for sequence modeling.

    Combines GNN with LSTM/GRU for temporal dependencies.
    """

    def __init__(self,
                 base_gnn: TacticalGNN,
                 hidden_dim: int = 128,
                 num_lstm_layers: int = 2,
                 bidirectional: bool = True):
        """
        Initialize temporal GNN.

        Args:
            base_gnn: Base TacticalGNN model
            hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.base_gnn = base_gnn
        self.hidden_dim = hidden_dim

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=base_gnn.output_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_lstm_layers > 1 else 0
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Output projection
        self.output_proj = nn.Linear(lstm_output_dim, base_gnn.output_dim)

    def forward(self, graph_sequence: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal GNN.

        Args:
            graph_sequence: List of Data objects (temporal sequence)

        Returns:
            sequence_embeddings: [seq_len, output_dim] - embeddings per timestep
            final_state: [output_dim] - final state embedding
        """
        # Extract graph embeddings for each timestep
        graph_embeddings = []
        for graph in graph_sequence:
            _, graph_emb = self.base_gnn(graph)
            graph_embeddings.append(graph_emb)

        # Stack into sequence
        sequence = torch.stack(graph_embeddings).unsqueeze(0)  # [1, seq_len, emb_dim]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(sequence)

        # Project output
        sequence_embeddings = self.output_proj(lstm_out.squeeze(0))
        final_state = sequence_embeddings[-1]

        return sequence_embeddings, final_state


class PressurePredictor(nn.Module):
    """Predicts defensive pressure intensity on ball carrier."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 64):
        """Initialize pressure predictor."""
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """Predict pressure intensity."""
        return self.predictor(graph_embedding)


class PassAvailabilityPredictor(nn.Module):
    """Predicts available passing lanes using node embeddings."""

    def __init__(self, node_dim: int = 64, hidden_dim: int = 64):
        """Initialize pass availability predictor."""
        super().__init__()

        # Pairwise compatibility scorer
        self.scorer = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_embeddings: torch.Tensor, ball_carrier_idx: int) -> torch.Tensor:
        """
        Predict pass availability from ball carrier to all teammates.

        Args:
            node_embeddings: [num_nodes, node_dim]
            ball_carrier_idx: Index of ball carrier

        Returns:
            pass_scores: [num_nodes] - pass availability scores
        """
        ball_carrier_emb = node_embeddings[ball_carrier_idx]
        ball_carrier_emb = ball_carrier_emb.unsqueeze(0).expand(node_embeddings.size(0), -1)

        # Concatenate ball carrier with each player
        pairwise = torch.cat([ball_carrier_emb, node_embeddings], dim=-1)

        # Score each potential pass
        scores = self.scorer(pairwise).squeeze(-1)

        return scores


def create_tactical_gnn(config: Optional[dict] = None) -> TacticalGNN:
    """
    Factory function to create TacticalGNN model.

    Args:
        config: Optional configuration dict

    Returns:
        TacticalGNN model instance
    """
    if config is None:
        config = {}

    default_config = {
        'input_dim': 12,
        'hidden_dim': 128,
        'num_layers': 4,
        'output_dim': 64,
        'edge_dim': 5,
        'gnn_type': 'gat',
        'dropout': 0.1,
        'use_residual': True,
        'num_heads': 4
    }

    default_config.update(config)
    return TacticalGNN(**default_config)


def load_pretrained_model(checkpoint_path: str, device: str = 'cpu') -> TacticalGNN:
    """
    Load pretrained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded TacticalGNN model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    model = create_tactical_gnn(config)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model
