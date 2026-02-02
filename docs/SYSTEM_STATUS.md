# Football Tracking & Decision Engine - System Status

This document provides an honest assessment of what has been built, what works, and what is incomplete or broken.

---

# TRACKING SYSTEM

## Current Reality

- `src/main.py` - Full pipeline code exists, but **requires trained models + calibration that don't exist**
- `tracking_app/app.py` - Manual click-to-track tool using optical flow
- Pre-trained YOLOv8s for generic person detection (not football-specific)

## Components Built

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| Player Detection | `src/detection/player_detector.py` | YOLOv8-based person detection | Works (generic) |
| Ball Detection | `src/detection/ball_detector.py` | Ball detection with interpolation | Partial |
| Team Classification | `src/detection/team_classifier.py` | K-means jersey color clustering | Fragile |
| Player Tracking | `src/tracking/tracker.py` | ByteTrack multi-object tracking | Works |
| Manual Calibration | `src/homography/calibration.py` | Interactive point clicking | Works |
| Auto Calibration | `src/homography/auto_calibration.py` | HRNet keypoint detection | Not working |
| Jersey Recognition | `src/identity/jersey_recognizer.py` | CRNN number recognition | No model |
| Player Re-ID | `src/identity/player_identifier.py` | OSNet appearance matching | No model |
| Web App | `tracking_app/app.py` | Manual annotation interface | Partial |

## What Actually Works

1. **Generic person detection** - YOLOv8 detects people, but includes crowd, coaches, refs
2. **Simple ID matching** - Position-based matching between frames
3. **Optical flow tracking** - For manually clicked players
4. **Export to CSV** - But in pixel coordinates, not pitch positions

## Major Gaps

### 1. No Trained Models
| Model | Status | Impact |
|-------|--------|--------|
| yolov8x.pt | Uses pre-trained (generic) | Detects non-players |
| jersey_recognizer.pth | Missing | Jersey numbers always None |
| osnet_path | Missing | Re-ID not working |
| HRNet keypoint model | Missing + TIMM not in requirements | Auto-calibration fails |

### 2. No Calibration Data
- Homography matrix required to convert pixels → pitch coordinates
- Interactive calibration requires clicking 9+ points manually
- No saved calibrations exist
- No automatic calibration working

### 3. Detection Issues
- Generic YOLO detects all people (crowd, coaches, refs)
- Height filtering (30-400px) is hardcoded, not resolution-adaptive
- Ball detection uses class 32 (sports ball) - often misses
- Ball interpolation is naive linear extrapolation

### 4. Tracking Issues
- Team classification requires ≥6 detections to fit
- Team classification can flip between frames (no temporal smoothing)
- Player IDs are track IDs (reset on re-detection)
- No stable player identification across occlusions

### 5. Output Issues
- CSV output has pixel coordinates, not world coordinates
- jersey_number always None (model not loaded)
- No standard format definition
- No quality metrics in output

## Failure Scenarios

**Scenario 1: Fresh Run**
```
python src/main.py video.mp4
→ Config loads
→ PlayerDetector init → yolov8x.pt missing
→ CRASH: "No such file or directory"
```

**Scenario 2: With Models**
```
→ Video loads
→ Interactive calibration prompts
→ User clicks 9 points → Homography computed
→ First frame: <6 players detected
→ CRASH: TeamClassifier.fit() fails "Not enough colors"
```

**Scenario 3: Everything Present**
```
→ All models available
→ Calibration succeeds
→ Team classification works
→ Processing starts
→ Jersey numbers: None (model not initialized)
→ Player IDs: reset on re-detection
→ World coordinates: sometimes invalid (outside pitch bounds)
→ Export has gaps, unusable for analysis
```

---

# DECISION ENGINE

## Current Reality

Two separate implementations exist:
1. **Python Engine** (`src/decision_engine/`) - Object-oriented, mathematically rigorous
2. **Web UI** (`ui/app.py`) - Interactive Flask app with physics-based calculations

## Components Built

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| Pitch Geometry | `src/decision_engine/pitch_geometry.py` | Coordinate system, distances | Complete |
| Elimination Logic | `src/decision_engine/elimination.py` | Defender elimination tracking | Incomplete |
| Defense Physics | `src/decision_engine/defense_physics.py` | Attraction-based positioning | Mostly works |
| Block Models | `src/decision_engine/block_models.py` | Defensive formations | Works |
| State Scoring | `src/decision_engine/state_scoring.py` | Game state evaluation | Incomplete |
| Visualizer | `src/decision_engine/visualizer.py` | Matplotlib rendering | Works |
| Web Backend | `ui/app.py` | Flask API + physics | Partially broken |
| Web Frontend | `ui/templates/index.html` | Interactive canvas | Works |

## What Actually Works

1. **Interactive tactical board** - Drag players, move ball
2. **Defensive formations** - 4-4-2 low block renders correctly
3. **xG calculation** - Distance/angle-based shot probability
4. **Gap detection** - Finds spaces between defenders
5. **Basic pass options** - Direct passes to teammates
6. **Action simulation** - Players move after an action
7. **Offside detection** - Second-to-last defender rule

## Major Bugs

### 1. Chain Calculation Crashes
**Function**: `calculate_position_value()` (ui/app.py:445)

**Bug**: References `current_state['attackers']` without receiving state as parameter
```python
def calculate_position_value(x, y, depth=0, max_depth=2, ...):
    attackers = current_state['attackers']  # Uses global!
```
**Impact**: Function can crash or use stale state

### 2. Pass Interception is Binary
**Function**: `calculate_pass_success()` (ui/app.py)

**Bug**: Returns 1.0 (success) or 0.0 (intercepted), no probability
- Doesn't account for pass speed variation
- Doesn't model multiple defenders compounding risk
- Defender acceleration curve not used properly

**Impact**: Passes either succeed 100% or fail 100%

### 3. Offside Line Calculation Reversed
**Function**: `get_offside_line()` (ui/app.py:872)

**Bug**: Sorts defenders with `reverse=True` (highest x first)
```python
sorted_defs = sorted(defenders, key=lambda d: d['x'], reverse=True)
```
In our coordinate system, defending team has LOW x values. This sorts them backwards.

**Impact**: Offside judgments may be wrong

### 4. Through Ball Target Offside Check Wrong
**Function**: `calculate_through_ball_options()` (ui/app.py:351)

**Bug**: Checks offside at current position, not target position
```python
if is_offside(att_x, ball_x):  # Should be target_x!
    continue
```
**Impact**: Through balls allowed when they should be offside

### 5. Shot Block Detection Uses Wrong Goal Position
**Function**: `calculate_position_value()` (ui/app.py:487)

**Bug**: Hardcodes goal at x=60
```python
block_dist = point_to_line_distance(d['x'], d['y'], x, y, 60, 0)
```
But pitch coordinates go -60 to +60, and goal is at x=60 (far end).

**Impact**: May incorrectly detect shot blocking

### 6. Attacker Arrival Tolerance Too Generous
**Function**: `calculate_through_ball_options()` (ui/app.py:374)

**Bug**: 0.5 second grace period for attacker arrival
```python
if attacker_time > ball_time + 0.5:
    continue
```
**Impact**: Through balls succeed too often

### 7. Player Width Interception Check Wrong
**Function**: `can_defender_intercept()` (ui/app.py:277)

**Bug**: Assumes defender can cover full width instantly
```python
if intercept_dist <= PLAYER_WIDTH:
    run_dist = 0  # Zero distance if within arm's reach
```
**Impact**: Interceptions too easy near passing lane

## What's Missing

### Not Implemented
- Offside trap mechanics (active push)
- Foul/contact model
- Set piece detection (corners, throw-ins)
- Fatigue/stamina effects
- Player skill differences
- Curved passes
- Goalkeeper specific logic

### Incomplete
- Pass interception probability (binary not continuous)
- Multi-defender interception (compound probability)
- Time pressure in chains (infinite time assumed)
- Defensive shape preservation during movement
- Formation parsing for 3-back systems

## Feature Status Summary

| Feature | Status | Issues |
|---------|--------|--------|
| Pitch geometry | 90% | None |
| Defensive blocks | 85% | 3-back not supported |
| Attraction physics | 75% | Marking weights need tuning |
| Cover shadows | 80% | Real-time probability missing |
| Pass success | 40% | Binary, no probability |
| Chain calculation | 20% | Global state crash, recursion bugs |
| Through balls | 50% | Offside check wrong |
| Action simulation | 60% | Shape preservation missing |
| Offside | 60% | Direction reversed |
| xG/shooting | 85% | Works well |

---

# PRIORITY FIXES

## Tracking System

1. **Get test video** - Currently no videos in video_cache
2. **Manual calibration workflow** - Document how to calibrate
3. **Filter detections to pitch** - Use green mask filtering
4. **Output world coordinates** - Apply homography transform to export

## Decision Engine

1. **Fix `calculate_position_value()`** - Pass state as parameter
2. **Fix offside line direction** - Remove reverse=True
3. **Fix through ball offside check** - Use target_x not att_x
4. **Add interception probability** - Not binary 0/1
5. **Add visual interception zones** - Show where defenders can reach

---

# HONEST ASSESSMENT

**Tracking**: ~40% functional
- Detection works but is generic
- Tracking works but IDs unstable
- Calibration requires manual work
- Output not usable for analysis

**Decision Engine**: ~60% functional
- Interactive board works
- Basic analysis works
- Chain calculation broken
- Several physics bugs need fixing
