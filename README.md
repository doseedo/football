# Football XY Tracking System

An end-to-end football (soccer) XY tracking system that processes single-camera wide-angle match footage and outputs frame-by-frame positional data for all 22 players and the ball.

## Features

- **Player Detection**: YOLOv8-based player detection with pitch masking
- **Ball Detection**: Temporal consistency for reliable ball tracking
- **Team Classification**: K-means clustering based on jersey colors
- **Multi-Object Tracking**: ByteTrack integration via Supervision
- **Homography Estimation**: Interactive calibration for pixel-to-world coordinate transformation
- **Rotating Camera Support**: Automatic detection and compensation for camera pan (up to ±45°)
- **Physical Metrics**: Speed, distance, acceleration, sprint detection
- **Data Export**: JSON and CSV output formats
- **Visualization**: Pitch plots and animations using mplsoccer

## Target Output

JSON/CSV files containing timestamped XY coordinates (in meters) for every player and the ball, plus derived physical metrics (speed, distance, acceleration).

## Footage Specifications

### Static Camera (Default)
- Camera: Centered, elevated, fixed position
- Resolution: 4K (3840x2160)
- Frame rate: 25-30 fps (standard broadcast)
- Angle: Consistent throughout match (no pan/tilt/zoom)

### Rotating Camera (Supported)
- Camera: Centered, elevated, on stable tripod
- Resolution: 4K (3840x2160)
- Frame rate: 25-30 fps
- Field of view: 80%+ of pitch visible at all times
- Pan range: Up to ±45° rotation to follow play
- **Note**: Mixed footage with both stable and rotating segments is supported

## Project Structure

```
football-tracker/
├── config/
│   ├── config.yaml
│   └── pitch_template.yaml
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── player_detector.py
│   │   ├── ball_detector.py
│   │   └── team_classifier.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py
│   ├── homography/
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   ├── pitch_detector.py      # Automatic pitch line detection
│   │   └── rotation_handler.py    # Camera rotation compensation
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── physical.py
│   └── output/
│       ├── __init__.py
│       ├── data_export.py
│       └── visualizer.py
├── models/
├── data/
│   ├── input/
│   ├── output/
│   └── cache/
├── scripts/
│   └── download_models.sh
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Setup
cd football
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download YOLOv8 model
bash scripts/download_models.sh

# 3. Place video in data/input/

# 4. Run (static camera)
python -m src.main data/input/match.mp4 --config config/config.yaml --visualize

# 5. Run (rotating camera - use --rotation flag)
python -m src.main data/input/match.mp4 --config config/config.yaml --rotation --visualize
```

## Rotating Camera Usage

For footage where the camera pans/rotates to follow play:

### Option 1: Command Line Flag
```bash
python -m src.main data/input/match.mp4 --rotation --visualize
```

### Option 2: Config File
Edit `config/config.yaml` and set:
```yaml
rotation:
  enabled: true
```

### How It Works

1. **Initial Calibration**: Manual keypoint selection on first frame (same as static camera)
2. **Feature Tracking**: ORB features are tracked frame-to-frame on pitch surface
3. **Rotation Detection**: System detects when camera is rotating vs. stable
4. **Homography Updates**: Pixel-to-world transformation is updated each frame
5. **Smoothing**: Temporal smoothing prevents jitter in world coordinates

### Rotation Configuration Options

```yaml
rotation:
  enabled: true                # Enable rotation handling
  max_angle: 45.0              # Maximum expected rotation (degrees)
  rotation_threshold: 0.5      # Degrees/frame to detect rotation
  stabilization_frames: 10     # Frames to confirm camera stopped
  buffer_size: 30              # Homography buffer for smoothing
  smoothing_factor: 0.3        # Lower = smoother, higher = responsive
  redetection_interval: 100    # Re-detect keypoints every N frames
  min_keypoints: 4             # Minimum keypoints for updates
```

### Tips for Rotating Camera Footage

1. **Calibrate during stable period**: For best results, calibrate when camera is not moving
2. **Good lighting**: Feature tracking works better with consistent lighting
3. **Visible pitch lines**: More visible white lines = better rotation tracking
4. **Gradual movements**: Sudden jerky movements may cause tracking gaps

## Calibration Instructions

When interactive calibration opens:
1. Click pitch keypoints in order (corners, center spots, etc.)
2. Press 'q' when done (minimum 4 points)
3. Press 'u' to undo, 'r' to reset

Recommended points: 4 corners, center circle intersections, penalty spots.

## Output Format

### Frame Data (JSON)
```json
{
  "frame": 1234,
  "timestamp": 49.36,
  "players": [
    {"track_id": 7, "team": "home", "jersey_number": 10, "x": 45.2, "y": 23.1, "speed": 18.5, "acceleration": 2.1}
  ],
  "ball": {"x": 48.5, "y": 27.3, "is_interpolated": false}
}
```

### Metrics Summary
```json
{
  "track_id": 7,
  "total_distance_m": 10234.5,
  "max_speed_kmh": 32.4,
  "avg_speed_kmh": 7.2,
  "sprint_count": 45,
  "sprint_distance_m": 1856.2,
  "high_intensity_distance_m": 2341.8
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size or use half_precision: true |
| Poor tracking | Increase track_buffer, lower confidence |
| Bad homography | Use more keypoints accurately |
| Missed ball | Lower ball confidence, increase temporal_window |
| Wrong teams | Adjust color_space (try HSV or RGB) |
| Jittery coordinates (rotating) | Lower smoothing_factor (e.g., 0.1) |
| Slow rotation response | Increase smoothing_factor (e.g., 0.5) |
| Lost tracking during pan | Increase buffer_size, lower rotation_threshold |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1660 (6GB) | RTX 3080+ (10GB+) |
| RAM | 16GB | 32GB+ |
| Storage | 500GB SSD | 2TB+ |

## License

MIT License
