# Football XY Tracking System

An end-to-end football (soccer) XY tracking system that processes single-camera wide-angle match footage and outputs frame-by-frame positional data for all 22 players and the ball.

## Features

- **Player Detection**: YOLOv8-based player detection with pitch masking
- **Ball Detection**: Temporal consistency for reliable ball tracking
- **Team Classification**: K-means clustering based on jersey colors
- **Multi-Object Tracking**: ByteTrack integration via Supervision
- **Homography Estimation**: Interactive calibration for pixel-to-world coordinate transformation
- **Physical Metrics**: Speed, distance, acceleration, sprint detection
- **Data Export**: JSON and CSV output formats
- **Visualization**: Pitch plots and animations using mplsoccer

## Target Output

JSON/CSV files containing timestamped XY coordinates (in meters) for every player and the ball, plus derived physical metrics (speed, distance, acceleration).

## Footage Specifications

- Camera: Centered, elevated, fixed position
- Resolution: 4K (3840x2160)
- Frame rate: 25-30 fps (standard broadcast)
- Angle: Consistent throughout match (no pan/tilt/zoom)

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
│   │   └── calibration.py
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

# 4. Run
python -m src.main data/input/match.mp4 --config config/config.yaml --visualize
```

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

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1660 (6GB) | RTX 3080+ (10GB+) |
| RAM | 16GB | 32GB+ |
| Storage | 500GB SSD | 2TB+ |

## License

MIT License
