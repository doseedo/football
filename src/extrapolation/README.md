# Off-Screen Player Extrapolation Module

Transformer-based system for predicting player positions when they move off-camera, matching SkillCorner's extrapolation capability.

## Overview

This module implements state-of-the-art trajectory prediction using:

1. **Baller2Vec**: Multi-entity Transformer that attends across both players AND time
2. **Baller2Vec++**: Enhanced version with look-ahead and coordinated agent modeling
3. **Physics Fallback**: Kalman filter-based motion prediction for robustness
4. **TrajectoryPredictor**: High-level interface combining all methods

## Architecture

### Baller2Vec

Multi-entity transformer that models player trajectories by attending across:
- **Temporal dimension**: Player's movement over time
- **Entity dimension**: Interactions between players

Key features:
- 6 transformer layers, 256 hidden dim, 8 attention heads
- Learnable player embeddings
- Positional encoding for temporal information
- Handles variable number of players with masking

```python
from src.extrapolation import Baller2Vec

model = Baller2Vec(
    d_model=256,
    num_heads=8,
    num_layers=6,
    max_players=22
)

# Predict future positions
predictions = model.predict_future(
    past_features,
    n_future_steps=10,
    autoregressive=True
)
```

### Baller2Vec++

Enhanced architecture with:
- **Look-ahead trajectory encoding**: Better long-term predictions
- **Coordinated attention**: Special masking for teammates vs opponents
- **Multi-scale temporal modeling**: Captures patterns at different timescales
- **Uncertainty estimation**: Provides confidence scores

```python
from src.extrapolation import Baller2VecPlus

model = Baller2VecPlus(
    d_model=256,
    num_heads=8,
    num_layers=6,
    use_player_embeddings=True
)

# Predict with uncertainty
predictions, uncertainty = model.predict_future(
    past_features,
    teams,
    n_future_steps=10,
    return_uncertainty=True
)
```

### Motion Model

Physics-based Kalman filter fallback:
- Constant acceleration motion model
- Velocity and acceleration constraints
- Pitch boundary handling
- Used when transformer confidence is low

```python
from src.extrapolation import KalmanMotionModel

kalman = KalmanMotionModel(
    dt=0.04,  # 25fps
    max_velocity=12.0,  # ~43 km/h
    pitch_bounds=(0, 0, 105, 68)
)

# Initialize with observation
kalman.initialize(position, timestamp)

# Predict future positions
future_states = kalman.extrapolate(n_steps=25)
```

## Usage

### Basic Example

```python
from src.extrapolation import TrajectoryPredictor, PlayerState

# Initialize predictor
predictor = TrajectoryPredictor(
    model_type='baller2vec_plus',
    model_path='models/baller2vec_plus.pth',
    history_length=25,
    confidence_threshold=0.5,
    use_physics_fallback=True
)

# Update with visible players
visible_players = [
    PlayerState(
        player_id=1,
        position=(45.2, 23.1),
        velocity=(2.0, 0.5),
        team=0,
        is_visible=True,
        confidence=1.0
    ),
    # ... more players
]

predictor.update_history(visible_players, timestamp=0.04)

# Predict all 22 players (including off-screen)
result = predictor.predict(
    visible_players=visible_players,
    timestamp=0.04,
    predict_all_players=True
)

# Check predictions
for player in result.players:
    if not player.is_visible:
        print(f"Player {player.player_id} extrapolated at {player.position}")
        confidence = predictor.get_extrapolation_confidence(player.player_id)
        print(f"  Confidence: {confidence:.2f}")
```

### Integration with Main Tracker

```python
from src.main import FootballTracker
from src.extrapolation import TrajectoryPredictor

tracker = FootballTracker('config/config.yaml')
predictor = TrajectoryPredictor(
    model_type='baller2vec_plus',
    model_path='models/baller2vec_plus.pth'
)

# In tracking loop
for frame in video:
    # Get visible detections
    detections = tracker.detect(frame)

    # Convert to PlayerState objects
    visible_players = [
        PlayerState(
            player_id=track.track_id,
            position=track.world_position,
            velocity=track.velocity,
            team=track.team,
            is_visible=True,
            confidence=track.confidence
        )
        for track in detections
    ]

    # Predict all players (fills in off-screen ones)
    result = predictor.predict(
        visible_players,
        timestamp,
        predict_all_players=True
    )

    # Export all 22 player positions
    export_frame_data(result.players, timestamp)
```

## Training

### Data Format

Training data should be in NPZ format with:
- `positions`: (N, seq_len, num_players, 2) - XY positions over time
- `teams`: (N, num_players) - team assignments (0 or 1)
- `masks`: (N, num_players) - optional padding masks

Example data preparation:
```python
import numpy as np

# Assuming you have tracking data
positions = []  # List of (seq_len, num_players, 2) arrays
teams = []      # List of (num_players,) arrays

np.savez(
    'data/training/trajectories.npz',
    positions=np.array(positions),
    teams=np.array(teams)
)
```

### Training Script

```bash
python training/train_baller2vec.py \
    --data data/training/trajectories.npz \
    --val-data data/validation/trajectories.npz \
    --model-type baller2vec_plus \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --output models/baller2vec_plus.pth
```

### Data Sources

Recommended datasets for training:
1. **SkillCorner Open Data**: Real football tracking data
2. **StatsBomb 360**: Event data with player positions
3. **Metrica Sports Sample Data**: Open tracking dataset
4. **NBA SportVU**: Basketball tracking (for pre-training)

## Model Performance

Target metrics (from research papers):
- **ADE (Average Displacement Error)**: < 2.0m over 2s prediction
- **FDE (Final Displacement Error)**: < 3.0m at 2s
- **Confidence calibration**: ECE < 0.1

Factors affecting performance:
1. **History length**: More context = better predictions
2. **Prediction horizon**: Accuracy degrades with time
3. **Team coordination**: Baller2Vec++ excels at coordinated movements
4. **Training data quality**: Needs diverse game situations

## Key Insights from Research

From **"Baller2Vec: A Multi-Entity Transformer for Multi-Agent Trajectory Forecasting"**:

1. **Multi-entity attention** is crucial for modeling player interactions
2. **Attention heads specialize**: Some learn defensive patterns, others offensive
3. **Look-ahead helps**: Future trajectory encoding improves long-term prediction
4. **Team coordination matters**: Modeling teammates separately from opponents helps

From **SkillCorner's approach**:

1. Uses transformer-based extrapolation in production
2. Critical for handling broadcast footage where players go off-screen
3. Combines with re-identification to maintain player tracking
4. Achieves < 2m error for 2-second extrapolation

## Files

```
src/extrapolation/
├── __init__.py                 # Module exports
├── baller2vec.py              # Base multi-entity transformer
├── baller2vec_plus.py         # Enhanced version with look-ahead
├── trajectory_predictor.py    # High-level prediction interface
├── motion_model.py            # Physics-based Kalman filter
└── README.md                  # This file

training/
└── train_baller2vec.py        # Training script

examples/
└── extrapolation_example.py   # Usage examples
```

## Technical Details

### Input Features
- Position (x, y)
- Velocity (vx, vy)
- Team (one-hot encoded)
- Optional: Player ID embeddings

### Output
- Future positions (x, y) for each player
- Optional: Uncertainty estimates (σx, σy)

### Loss Functions
- **Position loss**: MSE or negative log-likelihood with uncertainty
- **Velocity consistency**: Smoothness constraint
- **Multi-task**: Combined position + velocity prediction

### Inference
- **Non-autoregressive**: Predict all future steps at once (faster)
- **Autoregressive**: Iterative prediction (more accurate for long horizons)
- **Hybrid**: Transformer for visible players, physics for off-screen

## Future Improvements

1. **Multi-modal predictions**: Generate multiple possible trajectories
2. **Goal-conditioned**: Incorporate tactical objectives
3. **Social pooling**: Better modeling of crowd behavior
4. **Temporal hierarchies**: Different models for short/long-term
5. **Domain adaptation**: Transfer learning from other sports

## References

- "Baller2Vec: A Multi-Entity Transformer for Multi-Agent Trajectory Forecasting" (2020)
- "Baller2Vec++: A Look-Ahead Multi-Entity Transformer For Modeling Coordinated Agents" (2021)
- "SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos" (2018)
- SkillCorner technical blog and documentation

## License

This implementation is for research and educational purposes.
