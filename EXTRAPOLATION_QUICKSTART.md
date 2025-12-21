# Off-Screen Player Extrapolation - Quick Start Guide

Get started with the trajectory prediction system in 5 minutes.

## Installation

```bash
# Install dependencies
pip install torch torchvision numpy scipy loguru tqdm pyyaml

# Or use the main requirements
pip install -r requirements.txt
```

## Basic Usage

### 1. Initialize Predictor

```python
from src.extrapolation import TrajectoryPredictor, PlayerState

# Without pre-trained model (uses physics fallback)
predictor = TrajectoryPredictor(
    model_type='baller2vec_plus',
    model_path=None,
    use_physics_fallback=True
)

# With trained model
predictor = TrajectoryPredictor(
    model_type='baller2vec_plus',
    model_path='models/baller2vec_plus.pth',
    confidence_threshold=0.5
)
```

### 2. Track Players

```python
# Create player states from your tracking data
visible_players = [
    PlayerState(
        player_id=1,
        position=(45.2, 23.1),  # meters on pitch
        velocity=(2.0, 0.5),     # m/s
        team=0,                  # 0 or 1
        is_visible=True,
        confidence=1.0
    ),
    # ... more players
]

# Update history
predictor.update_history(visible_players, timestamp=0.04)
```

### 3. Predict All Players

```python
# Predict all 22 players (fills in off-screen ones)
result = predictor.predict(
    visible_players=visible_players,
    timestamp=0.04,
    predict_all_players=True
)

# Access predictions
for player in result.players:
    print(f"Player {player.player_id}:")
    print(f"  Position: {player.position}")
    print(f"  Visible: {player.is_visible}")
    print(f"  Confidence: {player.confidence:.2f}")

    if not player.is_visible:
        conf = predictor.get_extrapolation_confidence(player.player_id)
        print(f"  Extrapolation confidence: {conf:.2f}")
```

## Run Examples

```bash
# Basic usage examples
python examples/extrapolation_example.py

# Test suite
python tests/test_extrapolation.py
```

## Training Your Own Model

### Step 1: Prepare Data

```python
import numpy as np

# Your tracking data: list of matches
# Each match: (frames, 22 players, 2 coords)
matches = load_your_tracking_data()

positions = []
teams = []

for match in matches:
    # Sliding window: 35 frames (25 input + 10 prediction)
    for i in range(0, len(match) - 35, 5):
        window = match[i:i+35]
        positions.append(window[:, :, :2])  # x, y
        # Teams: assuming you have team assignments
        teams.append(get_team_assignments(window[0]))

# Save
np.savez(
    'data/training/trajectories.npz',
    positions=np.array(positions),
    teams=np.array(teams)
)
```

### Step 2: Train Model

```bash
python training/train_baller2vec.py \
    --data data/training/trajectories.npz \
    --val-data data/validation/trajectories.npz \
    --model-type baller2vec_plus \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda \
    --output models/baller2vec_plus.pth
```

### Step 3: Use Trained Model

```python
predictor = TrajectoryPredictor(
    model_type='baller2vec_plus',
    model_path='models/baller2vec_plus.pth'
)
```

## Integration with Main Tracker

```python
from src.main import FootballTracker
from src.extrapolation import TrajectoryPredictor, PlayerState

# Initialize both systems
tracker = FootballTracker('config/config.yaml')
predictor = TrajectoryPredictor(
    model_type='baller2vec_plus',
    model_path='models/baller2vec_plus.pth'
)

# Process video
cap = cv2.VideoCapture('match.mp4')
fps = 25

for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx / fps

    # Standard tracking (gets visible players)
    tracks = tracker.player_tracker.update(
        tracker.player_detector.detect(frame)
    )

    # Convert to PlayerState
    visible_players = []
    for track in tracks:
        # Get world coordinates
        pixel_pos = track.foot_position
        world_pos = tracker.transformer.pixel_to_world(pixel_pos)

        # Get velocity
        velocity = get_velocity_from_history(track)

        # Get team
        team_result = tracker.team_classifier.classify(frame, track.bbox)

        visible_players.append(PlayerState(
            player_id=track.track_id,
            position=world_pos,
            velocity=velocity,
            team=0 if team_result.team.name == 'HOME' else 1,
            is_visible=True,
            confidence=track.confidence
        ))

    # Predict all 22 players
    result = predictor.predict(
        visible_players,
        timestamp,
        predict_all_players=True
    )

    # Export complete data (visible + extrapolated)
    frame_data = {
        'frame': frame_idx,
        'timestamp': timestamp,
        'players': [
            {
                'player_id': p.player_id,
                'x': p.position[0],
                'y': p.position[1],
                'vx': p.velocity[0],
                'vy': p.velocity[1],
                'team': p.team,
                'is_visible': p.is_visible,
                'is_extrapolated': not p.is_visible,
                'confidence': p.confidence
            }
            for p in result.players
        ]
    }

    export_frame_data(frame_data)
```

## Performance Tips

### 1. GPU Acceleration

```python
# Use GPU for inference
predictor = TrajectoryPredictor(
    model_type='baller2vec_plus',
    model_path='models/baller2vec_plus.pth',
    device='cuda'  # or 'cpu'
)
```

### 2. Batch Processing

```python
# Process multiple frames at once
for batch_frames in batch(all_frames, batch_size=8):
    # Process batch
    results = [predictor.predict(...) for frame in batch_frames]
```

### 3. History Management

```python
# Adjust history length based on needs
predictor = TrajectoryPredictor(
    history_length=25,  # 1 second at 25fps (default)
    # Longer history = more context but slower
    # Shorter history = faster but less accurate
)
```

### 4. Confidence Thresholds

```python
# Adjust when to use physics fallback
predictor = TrajectoryPredictor(
    confidence_threshold=0.5,  # Use physics if transformer confidence < 0.5
    use_physics_fallback=True
)
```

## Common Issues

### Issue: "No model weights found"

**Solution:** Either train a model or set `model_path=None` to use physics-only mode:
```python
predictor = TrajectoryPredictor(
    model_type='baller2vec',
    model_path=None,  # Use physics fallback only
    use_physics_fallback=True
)
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size or use CPU:
```python
predictor = TrajectoryPredictor(device='cpu')
```

### Issue: "Poor extrapolation accuracy"

**Solutions:**
1. Increase history length for more context
2. Train on more diverse data
3. Ensure team assignments are correct
4. Check coordinate system (should be meters)

### Issue: "Predictions drift off pitch"

**Solution:** Physics model handles this automatically with pitch bounds:
```python
from src.extrapolation import KalmanMotionModel

kalman = KalmanMotionModel(
    pitch_bounds=(0, 0, 105, 68)  # min_x, min_y, max_x, max_y
)
```

## Next Steps

1. **Try examples**: Run `python examples/extrapolation_example.py`
2. **Test suite**: Run `python tests/test_extrapolation.py`
3. **Prepare data**: Create training dataset from your tracking data
4. **Train model**: Use `training/train_baller2vec.py`
5. **Integrate**: Add to your main tracking pipeline

## Documentation

- **Full documentation**: `src/extrapolation/README.md`
- **Implementation details**: `EXTRAPOLATION_IMPLEMENTATION_SUMMARY.md`
- **API reference**: Check docstrings in source files

## Support

For issues or questions:
1. Check the README in `src/extrapolation/`
2. Review examples in `examples/extrapolation_example.py`
3. Read the implementation summary
4. Check docstrings in source code

---

**Ready to extrapolate!** 🎯⚽
