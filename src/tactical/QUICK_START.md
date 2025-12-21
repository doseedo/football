# Quick Start Guide - Tactical Analysis GNN

## Installation

```bash
# Install PyTorch Geometric and dependencies
pip install torch-geometric torch-scatter torch-sparse torch-cluster

# Or install all requirements
pip install -r requirements.txt
```

## Basic Usage (3 Steps)

### 1. Build Graph from Tracking Data

```python
from src.tactical import TrackingGraphBuilder

# Your tracking data (one frame)
frame_data = {
    'players': [
        {'track_id': 1, 'x': -20.0, 'y': 0.0, 'vx': 0.5, 'vy': 0.1, 'team': 0},
        {'track_id': 2, 'x': -15.0, 'y': 5.0, 'vx': 1.0, 'vy': 0.0, 'team': 0},
        # ... more players
    ],
    'ball': {'x': 5.0, 'y': 1.0}
}

# Build graph
graph_builder = TrackingGraphBuilder()
graph = graph_builder.build_graph(frame_data)
```

### 2. Analyze Tactical State

```python
from src.tactical import create_tactical_gnn, TeamStateClassifier

# Create GNN model
gnn_model = create_tactical_gnn()

# Classify tactical state
classifier = TeamStateClassifier(gnn_model, use_gnn_classifier=False)
team_state = classifier.classify(graph)

print(f"State: {team_state.state.name}")
print(f"Possession: Team {team_state.possession_team}")
print(f"Pressing: {team_state.pressing_intensity:.2f}")
```

### 3. Calculate Tactical Metrics

```python
from src.tactical import TacticalMetricsCalculator

metrics_calc = TacticalMetricsCalculator()
metrics = metrics_calc.calculate_all_metrics(
    graph,
    ball_position=(5.0, 1.0),
    possession_team=team_state.possession_team
)

print(f"Space control: {metrics.space_control_ratio:.2f}")
print(f"Passing lanes: {metrics.passing_lane_count}")
print(f"Expected threat: {metrics.expected_threat:.3f}")
```

## Run Example

```bash
cd /home/user/football/src/tactical
python example_usage.py
```

## Integration with Main Pipeline

Add to your `src/main.py`:

```python
from src.tactical import (
    TrackingGraphBuilder,
    create_tactical_gnn,
    TeamStateClassifier,
    TacticalMetricsCalculator
)

# Initialize (once)
graph_builder = TrackingGraphBuilder()
gnn_model = create_tactical_gnn()
state_classifier = TeamStateClassifier(gnn_model, use_gnn_classifier=False)
metrics_calc = TacticalMetricsCalculator()

# In your processing loop
for frame_data in tracking_results:
    # Build graph
    graph = graph_builder.build_graph(frame_data)

    # Classify state
    team_state = state_classifier.classify(graph)

    # Calculate metrics
    ball_pos = (frame_data['ball']['x'], frame_data['ball']['y'])
    metrics = metrics_calc.calculate_all_metrics(graph, ball_pos, team_state.possession_team)

    # Add to output
    frame_data['tactical'] = {
        'state': team_state.state.name,
        'possession': team_state.possession_team,
        'pressing': team_state.pressing_intensity,
        'space_control': metrics.space_control_ratio,
        'passing_lanes': metrics.passing_lane_count,
        'expected_threat': metrics.expected_threat
    }
```

## Training (Optional)

### Using Synthetic Data

```python
from src.tactical.train_gnn import create_synthetic_training_data, train_tactical_gnn

# Generate data
create_synthetic_training_data(num_samples=1000, output_path='train.json')
create_synthetic_training_data(num_samples=200, output_path='val.json')

# Train
python src/tactical/train_gnn.py
```

### Using Real Data

Prepare JSON file:
```json
[
  {
    "frame_data": {
      "players": [...],
      "ball": {...}
    },
    "label": 0,  // 0=ATTACKING, 1=DEFENDING, etc.
    "scenario": "attacking"
  }
]
```

## Output Format

Tactical analysis adds this to each frame:

```json
{
  "frame": 0,
  "timestamp": 0.0,
  "players": [...],
  "ball": {...},
  "tactical": {
    "state": "ATTACKING",
    "confidence": 0.85,
    "possession_team": 0,
    "pressing_intensity": 0.32,
    "space_control": 0.58,
    "passing_lanes": 8,
    "expected_threat": 0.124,
    "defensive_line": 38.5,
    "team_compactness": 18.3
  }
}
```

## Configuration

### GNN Model Config

```python
config = {
    'input_dim': 12,        # Node features
    'hidden_dim': 128,      # Hidden layer size
    'num_layers': 4,        # Graph conv layers
    'output_dim': 64,       # Embedding size
    'gnn_type': 'gat',      # 'gcn', 'sage', or 'gat'
    'num_heads': 4,         # Attention heads (for GAT)
    'dropout': 0.1
}

gnn_model = create_tactical_gnn(config)
```

### Graph Builder Config

```python
graph_builder = TrackingGraphBuilder(
    proximity_threshold=15.0,  # Edge distance threshold (meters)
    edge_type='proximity',     # 'proximity', 'team', or 'full'
    use_temporal=True          # Enable temporal features
)
```

## Troubleshooting

**ImportError: No module named 'torch_geometric'**
```bash
pip install torch-geometric torch-scatter torch-sparse
```

**Graph has no edges**
- Check proximity_threshold (try increasing to 20.0)
- Verify player positions are in meters
- Try edge_type='full' for testing

**Low accuracy**
- Use more training data (10k+ samples recommended)
- Try different GNN architectures (GAT usually best)
- Increase model size (hidden_dim=256)

## Performance Tips

1. **Batch Processing**: Process multiple frames together
2. **GPU**: Use CUDA for 5-10x speedup
3. **Model Size**: Smaller models for real-time (2 layers, 64 dim)
4. **Edge Pruning**: Lower proximity_threshold to reduce edges

## Next Steps

1. Read full documentation: `src/tactical/README.md`
2. Run examples: `python src/tactical/example_usage.py`
3. Review training: `src/tactical/train_gnn.py`
4. See integration tests: `src/tactical/test_integration.py`
