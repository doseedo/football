# SkillCorner-Level Implementation Plan

## Overview

This document outlines the comprehensive implementation plan to match SkillCorner's broadcast tracking capabilities. Based on research of their technology and academic papers, we need to implement the following major components.

## Target Capabilities

| Feature | Current State | Target State |
|---------|--------------|--------------|
| Homography | Manual calibration | Fully automatic via deep learning |
| Off-screen tracking | None | Transformer-based extrapolation |
| Ball tracking | 2D only | 3D trajectory with physics |
| Player ID | Basic color clustering | Unsupervised re-ID + jersey OCR |
| Detection | YOLOv8 standard | DETR for crowded scenes |
| Tactical analysis | None | Graph Neural Networks |
| Processing speed | Batch only | Real-time 10fps, <2s delay |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SKILLCORNER-LEVEL PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   VIDEO     │───▶│  DETECTION  │───▶│  TRACKING   │         │
│  │   INPUT     │    │  (DETR)     │    │ (ByteTrack) │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                               │                 │
│  ┌─────────────┐    ┌─────────────┐          │                 │
│  │ AUTO        │───▶│ COORDINATE  │◀─────────┤                 │
│  │ HOMOGRAPHY  │    │ TRANSFORM   │          │                 │
│  │ (HRNet)     │    └─────────────┘          │                 │
│  └─────────────┘                             │                 │
│                                              ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ 3D BALL     │    │ PLAYER      │    │ OFF-SCREEN  │         │
│  │ TRAJECTORY  │    │ RE-ID       │    │ EXTRAPOL.   │         │
│  │ (LSTM)      │    │ (OSNet+OCR) │    │ (baller2vec)│         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│                    ┌─────────────┐                              │
│                    │    GNN      │                              │
│                    │ TACTICAL    │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│                           ▼                                     │
│                    ┌─────────────┐                              │
│                    │   OUTPUT    │                              │
│                    │  EXPORTER   │                              │
│                    └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Automatic Homography Estimation

**Goal**: Eliminate manual calibration, automatically detect pitch keypoints and compute homography.

**Architecture**:
- HRNetv2 encoder-decoder for keypoint heatmap prediction
- 57 predefined pitch keypoints (corners, line intersections, circle points)
- RANSAC + DLT for homography computation
- Bayesian Kalman filter for temporal smoothing (BHITK method)

**Files to create**:
```
src/homography/
├── keypoint_detector.py      # HRNet-based keypoint detection
├── auto_calibration.py       # Automatic homography pipeline
├── bayesian_filter.py        # Temporal Bayesian smoothing
└── field_model.py            # 3D field model with all keypoints
```

**Training data**: SoccerNet camera calibration dataset, WorldCup dataset

---

## Component 2: Off-Screen Player Extrapolation (baller2vec++)

**Goal**: Predict positions of players who are temporarily off-camera.

**Architecture**:
- Multi-entity Transformer (baller2vec++ architecture)
- Look-ahead trajectory prediction
- Attention mechanism models player interactions
- Handles coordinated team movements

**Files to create**:
```
src/extrapolation/
├── __init__.py
├── baller2vec.py             # Base transformer architecture
├── baller2vec_plus.py        # Improved version with look-ahead
├── trajectory_predictor.py   # High-level prediction interface
└── motion_model.py           # Physics-based fallback
```

**Training data**: SkillCorner open data, NBA tracking data (for pre-training)

---

## Component 3: 3D Ball Trajectory Estimation

**Goal**: Estimate 3D ball position (including height) from single 2D camera.

**Architecture**:
- LSTM-based trajectory network
- Physics-informed loss functions (gravity, bounce dynamics)
- Canonical 3D representation (camera-independent)
- Synthetic data pre-training + real data fine-tuning

**Files to create**:
```
src/ball3d/
├── __init__.py
├── trajectory_lstm.py        # LSTM trajectory network
├── physics_model.py          # Ball physics constraints
├── synthetic_generator.py    # Training data generation
└── ball3d_tracker.py         # Combined 2D detection + 3D estimation
```

---

## Component 4: Player Re-Identification

**Goal**: Identify players without prior training, using appearance + jersey numbers.

**Architecture**:
- OSNet for appearance embedding extraction
- Scene text recognition for jersey numbers (CRNN + attention)
- Contrastive learning for unsupervised team classification
- Temporal voting for robust identification

**Files to create**:
```
src/identity/
├── __init__.py
├── osnet.py                  # OSNet re-ID backbone
├── jersey_detector.py        # Jersey number detection
├── jersey_recognizer.py      # Scene text recognition for numbers
├── contrastive_team.py       # Unsupervised team classification
└── player_identifier.py      # Combined identification pipeline
```

---

## Component 5: DETR-based Detection

**Goal**: Better detection in crowded scenes with occlusions.

**Architecture**:
- DETR (Detection Transformer) with ResNet-50 backbone
- Set prediction with Hungarian matching
- Better handling of overlapping players
- Integration with TPH (Transformer Prediction Heads)

**Files to create**:
```
src/detection/
├── detr_detector.py          # DETR implementation
├── transformer_head.py       # TPH for multi-scale detection
└── hybrid_detector.py        # YOLO + DETR ensemble
```

---

## Component 6: Graph Neural Network for Tactical Analysis

**Goal**: Model player relationships and team coordination.

**Architecture**:
- Graph representation of player positions
- Message passing for relationship modeling
- Team state classification (attacking, defending, transition)
- Counterattack success prediction

**Files to create**:
```
src/tactical/
├── __init__.py
├── graph_builder.py          # Convert tracking to graphs
├── gnn_model.py              # Graph Neural Network
├── team_state.py             # Team state classification
└── tactical_metrics.py       # Advanced tactical metrics
```

---

## Component 7: Real-Time Pipeline

**Goal**: Process at 10fps with <2s delay.

**Optimizations**:
- TensorRT optimization for inference
- Batched frame processing
- Async I/O for video reading
- Model quantization (FP16/INT8)
- Pipeline parallelization

**Files to create**:
```
src/realtime/
├── __init__.py
├── pipeline.py               # Optimized inference pipeline
├── tensorrt_wrapper.py       # TensorRT optimization
├── async_reader.py           # Async video reader
└── stream_processor.py       # Live stream processing
```

---

## Component 8: Training Infrastructure

**Goal**: Enable training all models with proper data pipelines.

**Files to create**:
```
training/
├── configs/
│   ├── homography.yaml
│   ├── baller2vec.yaml
│   ├── ball3d.yaml
│   ├── reid.yaml
│   └── gnn.yaml
├── datasets/
│   ├── soccernet_loader.py
│   ├── skillcorner_loader.py
│   └── synthetic_loader.py
├── train_homography.py
├── train_baller2vec.py
├── train_ball3d.py
├── train_reid.py
├── train_gnn.py
└── utils/
    ├── losses.py
    ├── metrics.py
    └── augmentations.py
```

---

## Updated Requirements

Add to requirements.txt:
```
# Transformers & Attention
transformers>=4.35.0
einops>=0.7.0

# Graph Neural Networks
torch-geometric>=2.4.0
spektral>=1.3.0

# Re-identification
torchreid>=0.3.0

# Real-time optimization
tensorrt>=8.6.0
onnx>=1.15.0
onnxruntime-gpu>=1.16.0

# Additional
albumentations>=1.3.0
pytorch-lightning>=2.1.0
wandb>=0.16.0
```

---

## Updated Output Format (SkillCorner-compatible)

```json
{
  "frame": 1234,
  "timestamp": 49.36,
  "image_corners_projection": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "homography_confidence": 0.95,
  "player_data": [
    {
      "trackable_object": 12345,
      "track_id": 7,
      "team_id": 1,
      "jersey_number": 10,
      "x": 45.2,
      "y": 23.1,
      "z": 0.0,
      "speed": 18.5,
      "acceleration": 2.1,
      "is_visible": true,
      "is_extrapolated": false,
      "confidence": 0.92,
      "embedding": [0.1, 0.2, ...]
    }
  ],
  "ball": {
    "x": 48.5,
    "y": 27.3,
    "z": 1.2,
    "speed": 25.0,
    "is_interpolated": false,
    "trajectory_confidence": 0.85
  },
  "tactical_state": {
    "possession_team": 1,
    "phase": "attacking",
    "pressing_intensity": 0.7
  }
}
```

---

## Agent Task Assignments

### Agent 1: Automatic Homography
- Implement HRNet keypoint detector
- Create field model with 57 keypoints
- Implement Bayesian temporal filter
- Create auto-calibration pipeline
- Write training script for keypoint model

### Agent 2: Off-Screen Extrapolation
- Implement baller2vec transformer architecture
- Implement baller2vec++ with look-ahead
- Create trajectory predictor interface
- Add physics-based motion model fallback
- Write training script

### Agent 3: 3D Ball Trajectory
- Implement LSTM trajectory network
- Add physics constraints (gravity, bounce)
- Create synthetic data generator
- Combine with 2D ball detection
- Write training script

### Agent 4: Player Re-ID
- Implement OSNet backbone
- Add jersey number detection/recognition
- Implement contrastive team learning
- Create unified player identifier
- Write training script

### Agent 5: DETR Detection
- Implement DETR detector
- Add transformer prediction heads
- Create hybrid YOLO+DETR ensemble
- Integrate with tracking pipeline

### Agent 6: Graph Neural Network
- Implement graph builder for tracking data
- Create GNN model for player relationships
- Add team state classification
- Compute tactical metrics

### Agent 7: Real-time Optimization
- Implement TensorRT wrappers
- Create async video reader
- Build optimized pipeline
- Add FP16/INT8 quantization

### Agent 8: Training Infrastructure
- Create all dataset loaders
- Implement loss functions and metrics
- Set up training configs
- Create data augmentation pipeline
- Set up experiment tracking (wandb)

---

## Execution Order

1. Agents 1-6 can work in parallel (independent modules)
2. Agent 7 depends on models from 1-6 being complete
3. Agent 8 should start immediately (training infra needed by all)

## Success Criteria

- Automatic homography: <1m reprojection error
- Off-screen tracking: <2m position error for 2s extrapolation
- 3D ball: <0.5m height estimation error
- Player ID: >90% accuracy without prior training
- Detection: >95% mAP in crowded scenes
- Real-time: 10fps processing, <2s latency
