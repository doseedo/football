# Graph Neural Network Tactical Analysis

This module implements a Graph Neural Network (GNN) system for tactical football analysis, enabling advanced understanding of team coordination, spatial relationships, and tactical states.

## Overview

The GNN tactical analysis system converts player tracking data into graph representations and uses message passing neural networks to learn complex team dynamics and tactical patterns.

### Key Features

- **Graph Construction**: Convert tracking data to graph representations with players as nodes
- **GNN Models**: Multiple architectures (GCN, GraphSAGE, GAT) for learning player interactions
- **Tactical State Classification**: Identify tactical phases (attacking, defending, transitions, set pieces)
- **Advanced Metrics**: Voronoi-based space control, pressing intensity, passing lanes, expected threat
- **Temporal Analysis**: Model tactical evolution over time sequences

## Architecture

### Components

1. **TrackingGraphBuilder** (`graph_builder.py`)
   - Converts tracking data to PyTorch Geometric graphs
   - Nodes: Players with 12D feature vectors
   - Edges: Spatial proximity, team connections, ball proximity
   - Temporal graph sequences for time-series analysis

2. **TacticalGNN** (`gnn_model.py`)
   - Graph Neural Network with 3-4 message passing layers
   - Supports GCN, GraphSAGE, and GAT architectures
   - Outputs per-player embeddings and graph-level embeddings
   - 64-128 dimensional hidden layers with residual connections

3. **TeamStateClassifier** (`team_state.py`)
   - Classifies tactical states using GNN embeddings or rule-based methods
   - States: ATTACKING, DEFENDING, TRANSITION_ATTACK, TRANSITION_DEFENSE, SET_PIECE
   - Possession detection and pressing intensity estimation
   - Defensive line and team compactness analysis

4. **TacticalMetricsCalculator** (`tactical_metrics.py`)
   - Advanced spatial metrics using geometric analysis
   - Voronoi tessellation for space control
   - Passing lane availability and quality
   - Expected threat (xT) grid
   - Pitch control estimation

## Installation

### Requirements

```bash
pip install torch-geometric torch-scatter torch-sparse torch-cluster
```

See `requirements.txt` for complete dependencies.

### GPU Support

For GPU training:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Example

```python
from src.tactical import (
    TrackingGraphBuilder,
    create_tactical_gnn,
    TeamStateClassifier,
    TacticalMetricsCalculator
)

# 1. Build graph from tracking data
graph_builder = TrackingGraphBuilder(proximity_threshold=15.0)
graph = graph_builder.build_graph(frame_data)

# 2. Create GNN model
gnn_model = create_tactical_gnn({
    'gnn_type': 'gat',
    'hidden_dim': 128,
    'num_layers': 4
})

# 3. Classify tactical state
state_classifier = TeamStateClassifier(gnn_model)
team_state = state_classifier.classify(graph)

print(f"State: {team_state.state.name}")
print(f"Possession: Team {team_state.possession_team}")
print(f"Pressing: {team_state.pressing_intensity:.2f}")

# 4. Calculate tactical metrics
metrics_calc = TacticalMetricsCalculator()
metrics = metrics_calc.calculate_all_metrics(graph, ball_position, possession_team=0)

print(f"Space control: {metrics.space_control_ratio:.2f}")
print(f"Passing lanes: {metrics.passing_lane_count}")
print(f"Expected threat: {metrics.expected_threat:.3f}")
```

### Temporal Analysis

```python
# Build temporal graph from frame sequence
temporal_graph = graph_builder.build_temporal_graph(frame_sequence)

# Analyze state transitions
states = []
for graph in temporal_graph.graphs:
    team_state = state_classifier.classify(graph)
    states.append(team_state.state)

# Detect transitions
for i in range(1, len(states)):
    if states[i] != states[i-1]:
        print(f"Transition: {states[i-1].name} → {states[i].name}")
```

### Integration with Main Pipeline

```python
# In main.py or processing pipeline
from src.tactical import TrackingGraphBuilder, TeamStateClassifier, TacticalMetricsCalculator

# Initialize tactical components
graph_builder = TrackingGraphBuilder()
gnn_model = create_tactical_gnn()
state_classifier = TeamStateClassifier(gnn_model, use_gnn_classifier=False)
metrics_calc = TacticalMetricsCalculator()

# Process each frame
for frame_data in tracking_results:
    # Build graph
    graph = graph_builder.build_graph(frame_data)

    # Classify state
    team_state = state_classifier.classify(graph)

    # Calculate metrics
    ball_pos = (frame_data['ball']['x'], frame_data['ball']['y'])
    metrics = metrics_calc.calculate_all_metrics(graph, ball_pos, team_state.possession_team)

    # Add to output
    frame_data['tactical_state'] = team_state.state.name
    frame_data['pressing_intensity'] = team_state.pressing_intensity
    frame_data['space_control'] = metrics.space_control_ratio
    frame_data['passing_lanes'] = metrics.passing_lane_count
```

## Training

### Data Requirements

**Labeled Training Data:**
- Minimum: ~1000 labeled frames for basic training
- Recommended: 10,000+ frames for production models
- Format: Tracking data with tactical state labels

**Label Sources:**
1. Manual annotation using tactical analysis software
2. Semi-supervised from event data (StatsBomb, Wyscout)
3. Rule-based pseudo-labeling with expert verification

### Training Pipeline

```python
from src.tactical.train_gnn import train_tactical_gnn, TacticalGraphDataset

# Create dataset
train_dataset = TacticalGraphDataset('data/train_tactical.json', graph_builder)
val_dataset = TacticalGraphDataset('data/val_tactical.json', graph_builder)

# Train model
config = {
    'gnn_type': 'gat',
    'hidden_dim': 128,
    'num_layers': 4,
    'num_heads': 4
}

history = train_tactical_gnn(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config,
    num_epochs=50,
    batch_size=32,
    lr=0.001,
    device='cuda'
)
```

### Synthetic Data (Demo)

For testing without labeled data:

```python
from src.tactical.train_gnn import create_synthetic_training_data

# Generate synthetic data
create_synthetic_training_data(
    num_samples=1000,
    output_path='data/synthetic_tactical.json'
)
```

### Training Tips

1. **Data Balance**: Ensure balanced representation of all tactical states
2. **Augmentation**: Rotate/flip pitch coordinates for data augmentation
3. **Learning Rate**: Start with 0.001, reduce on plateau
4. **Batch Size**: 16-32 graphs per batch
5. **Regularization**: Use dropout (0.1-0.2) and gradient clipping
6. **Validation**: Monitor validation accuracy and confusion matrix

### Expected Performance

With proper training data:
- **State Classification Accuracy**: 75-85%
- **Possession Detection**: 90-95%
- **Pressing Detection**: 80-90%

## Graph Structure

### Node Features (12D)

1. **Position** (2D): Normalized x, y coordinates [0, 1]
2. **Velocity** (2D): vx, vy in m/s
3. **Speed** (1D): Scalar speed
4. **Ball Distance** (1D): Normalized distance to ball
5. **Ball Angle** (1D): Angle to ball
6. **Team** (3D): One-hot encoding [team_0, team_1, unknown]
7. **Zone** (1D): Position zone (defensive/middle/attacking)
8. **Goal Distance** (1D): Distance to own goal

### Edge Features (5D)

1. **Relative Position** (2D): Normalized dx, dy
2. **Distance** (1D): Euclidean distance
3. **Same Team** (1D): Binary indicator
4. **Velocity Alignment** (1D): Dot product of velocities

### Edge Construction

**Proximity Mode** (default):
- Connect players within 15m
- Enhanced connections for same-team players

**Team Mode**:
- Only connect same-team players
- Useful for team-specific analysis

**Full Mode**:
- Fully connected graph
- Higher computational cost

## Tactical Metrics

### Team Structure
- **Defensive Line Height**: Distance of back 4 from own goal
- **Team Length**: Distance between defensive and offensive lines
- **Team Width**: Lateral spread of players
- **Team Compactness**: Average distance between teammates
- **Team Centroid**: Center of mass

### Space Control
- **Voronoi Areas**: Space controlled per player
- **Space Control Ratio**: Possession team's spatial dominance
- **Pitch Control**: % of pitch controlled by possession team

### Pressure & Passing
- **Pressure on Ball**: Defensive pressure intensity [0-1]
- **Passing Lane Count**: Number of available passes
- **Passing Lane Quality**: Average quality of lanes [0-1]
- **Progressive Options**: Forward pass opportunities

### Expected Threat
- **xT Value**: Threat level of current ball position
- **Threatening Players**: Attackers in dangerous positions

## Research Background

This implementation is inspired by:

1. **SkillCorner's GNN approach** for counterattack prediction
2. **DeepMind's Graph Networks** for multi-agent systems
3. **Spearman's Pitch Control Model** for space analysis
4. **StatsBomb's xT Grid** for expected threat

### Key Research Papers

- "Graph Networks for Multi-Agent Systems" (Battaglia et al., 2018)
- "Wide Open Spaces: A statistical technique for measuring space creation" (Spearman, 2018)
- "Actions Speak Louder than Goals: Valuing Player Actions in Soccer" (Decroos et al., 2019)

## Performance Considerations

### Computational Complexity

- **Graph Construction**: O(n²) for edge creation (n = number of players)
- **GNN Forward Pass**: O(E·D) where E = edges, D = hidden dimension
- **Typical Runtime**: ~5-10ms per frame on GPU

### Optimization Tips

1. **Batch Processing**: Process multiple frames in parallel
2. **GPU Acceleration**: Use CUDA for GNN operations
3. **Edge Pruning**: Limit edges to proximity threshold
4. **Model Size**: Balance accuracy vs. speed (smaller models for real-time)

## Limitations

1. **Training Data**: Requires labeled tactical states (manual annotation expensive)
2. **Domain Adaptation**: Models trained on one league may not generalize
3. **Partial Observability**: Assumes all players visible in frame
4. **Static Snapshots**: Single-frame analysis misses temporal context

## Future Enhancements

- [ ] Attention visualization for interpretability
- [ ] Multi-task learning (state + metrics jointly)
- [ ] Self-supervised pre-training on unlabeled data
- [ ] Formation recognition module
- [ ] Player role classification
- [ ] Action prediction (next pass/movement)

## Examples

See `example_usage.py` for complete working examples:
- Basic tactical analysis
- Temporal state transitions
- Integration with tracking pipeline
- Export to JSON/CSV

## Contributing

When extending this module:
1. Maintain graph structure compatibility
2. Document new metrics/features
3. Add unit tests for new functionality
4. Update this README

## License

Part of the Football Tracking System project.

## Contact

For questions or issues with the tactical analysis module, please refer to the main project documentation.
