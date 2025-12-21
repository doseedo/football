# Training Infrastructure for Football Tracking Models

Complete training pipeline for all models in the football tracking system.

## Overview

This training infrastructure provides:
- **6 Model Training Scripts** - One for each component
- **6 YAML Configurations** - Comprehensive hyperparameter configs
- **Dataset Loaders** - SoccerNet, SkillCorner, and synthetic data
- **Training Utilities** - Losses, metrics, and callbacks
- **Master Training Script** - Automated pipeline execution

## Directory Structure

```
training/
├── configs/                    # Model configurations
│   ├── homography.yaml        # Camera calibration
│   ├── baller2vec.yaml        # Trajectory embeddings
│   ├── ball3d.yaml            # 3D ball tracking
│   ├── reid.yaml              # Player re-identification
│   ├── detr.yaml              # Detection transformer
│   └── gnn.yaml               # Tactical analysis GNN
├── datasets/                   # Data loaders
│   ├── soccernet_loader.py    # SoccerNet dataset
│   ├── skillcorner_loader.py  # SkillCorner open data
│   ├── synthetic_loader.py    # Synthetic data generation
│   └── augmentations.py       # Data augmentation
├── utils/                      # Training utilities
│   ├── losses.py              # Custom loss functions
│   ├── metrics.py             # Evaluation metrics
│   └── callbacks.py           # Training callbacks
├── train_homography.py         # Train homography model
├── train_baller2vec.py         # Train trajectory model
├── train_ball3d.py             # Train ball 3D model
├── train_reid.py               # Train re-ID model
├── train_detr.py               # Train detection model
├── train_gnn.py                # Train GNN model
├── run_all_training.sh         # Master training script
└── README.md                   # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Additional training dependencies
pip install albumentations wandb tensorboard
```

### 2. Download Training Data

#### SkillCorner Open Data (Free)
```bash
git clone https://github.com/SkillCorner/opendata.git data/training/skillcorner
```

#### SoccerNet (Registration Required)
```python
# Register at https://www.soccer-net.org/ to get password
from training.datasets import download_soccernet_data

download_soccernet_data(
    root_path="data/training/soccernet",
    password="YOUR_PASSWORD",
    tasks=["calibration-2023", "tracking"]
)
```

#### Generate Synthetic Data
```bash
# Automatically generated during training if real data unavailable
python -c "from training.datasets import SyntheticHomographyDataset; \
           SyntheticHomographyDataset(num_samples=10000)"
```

### 3. Run Training

#### Train Individual Models
```bash
# Homography estimation
python training/train_homography.py --config training/configs/homography.yaml

# Baller2Vec trajectories
python training/train_baller2vec.py --config training/configs/baller2vec.yaml \
    --data data/training/trajectories.npz

# Ball 3D tracking
python training/train_ball3d.py --config training/configs/ball3d.yaml

# Re-identification
python training/train_reid.py --config training/configs/reid.yaml

# DETR detection
python training/train_detr.py --config training/configs/detr.yaml

# GNN tactical analysis
python training/train_gnn.py --config training/configs/gnn.yaml
```

#### Train All Models (Recommended)
```bash
# Run complete pipeline
./training/run_all_training.sh

# With options
DEVICE=cuda DOWNLOAD_DATA=true ./training/run_all_training.sh
```

## Model Details

### 1. Homography Estimation

**Purpose**: Camera calibration via keypoint detection

**Architecture**:
- Backbone: ResNet50 (pretrained)
- Keypoint heatmap head (29 keypoints)
- Output: 3x3 homography matrix

**Training**:
- Dataset: SoccerNet Calibration 2023 / Synthetic
- Loss: Heatmap MSE + Focal + Reprojection error
- Metrics: Reprojection error (meters), PCK
- Epochs: 100
- Batch size: 16

**Config**: `configs/homography.yaml`

### 2. Baller2Vec

**Purpose**: Self-supervised trajectory representation learning

**Architecture**:
- Transformer encoder (6 layers, 8 heads)
- Embedding dimension: 128
- Decoder for reconstruction

**Training**:
- Dataset: SoccerNet Tracking / SkillCorner / Synthetic
- Loss: Reconstruction + Contrastive + Future prediction
- Metrics: ADE, FDE at multiple horizons
- Epochs: 200
- Batch size: 64

**Config**: `configs/baller2vec.yaml`

### 3. Ball 3D Tracking

**Purpose**: Estimate 3D ball position and trajectory

**Architecture**:
- Detection refinement: EfficientNet-B3
- Trajectory net: TCN (6 layers)
- Physics-informed components

**Training**:
- Dataset: Synthetic 3D + SoccerNet (2D)
- Loss: Position 2D/3D + Velocity + Physics constraints
- Metrics: 3D position error, trajectory ADE/FDE
- Epochs: 150
- Batch size: 32

**Config**: `configs/ball3d.yaml`

### 4. Player Re-Identification

**Purpose**: Consistent player tracking across frames

**Architecture**:
- Backbone: OSNet-x1.0 / ResNet50
- Embedding: 256 dimensions
- Multi-head: Appearance + Jersey + Body shape

**Training**:
- Dataset: SoccerNet Re-ID / Market1501 (pretrain)
- Loss: Triplet + Center + Softmax + Contrastive
- Metrics: mAP, CMC, Rank-1/5/10
- Epochs: 120
- Batch size: 64 (P=16, K=4)

**Config**: `configs/reid.yaml`

### 5. DETR Detection

**Purpose**: End-to-end player and ball detection

**Architecture**:
- Backbone: ResNet50
- Transformer: 6 encoder + 6 decoder layers
- Queries: 50 (max objects)
- Classes: 3 (player, ball, referee)

**Training**:
- Dataset: SoccerNet Detection / COCO (pretrain)
- Loss: Hungarian matching (class + bbox + GIoU)
- Metrics: mAP, AP50, AP75
- Epochs: 300
- Batch size: 8 (with gradient accumulation)

**Config**: `configs/detr.yaml`

### 6. Graph Neural Network

**Purpose**: Tactical analysis, formation, and strategy prediction

**Architecture**:
- Node encoder: Player/ball features
- Graph convolution: GAT (4 layers, 4 heads)
- Temporal: BiLSTM (2 layers)
- Multi-task heads: Formation + Role + Strategy + Trajectory

**Training**:
- Dataset: SoccerNet + SkillCorner + StatsBomb
- Loss: Multi-task (formation + role + strategy + trajectory)
- Metrics: Classification accuracy, trajectory ADE/FDE
- Epochs: 150
- Batch size: 32

**Config**: `configs/gnn.yaml`

## Dataset Information

### SoccerNet

**Download**: Requires registration at https://www.soccer-net.org/

**Components**:
- **Calibration 2023**: Camera calibration with keypoint annotations
- **Tracking**: Player tracking data (2D coordinates)
- **Detection**: Bounding box annotations

**Format**: JSON with video frames

### SkillCorner Open Data

**Download**: 
```bash
git clone https://github.com/SkillCorner/opendata.git
```

**Components**:
- Player tracking (10Hz)
- Ball tracking
- Match events
- Team formations

**Format**: JSON with structured data

**Matches**: Multiple matches from top leagues

### Synthetic Data

**Generated on-the-fly** if real data unavailable:
- **Ball 3D**: 50,000 parabolic/ground/bounce trajectories
- **Trajectories**: 50,000 player movement patterns
- **Homography**: 10,000 synthetic pitch images

## Training Utilities

### Losses (`utils/losses.py`)

- `KeypointHeatmapLoss`: MSE + Focal for heatmaps
- `ReprojectionLoss`: Geometric homography error
- `TrajectoryLoss`: ADE + FDE with Gaussian NLL
- `ContrastiveLoss`: InfoNCE for self-supervised learning
- `TripletLoss`: Re-ID triplet margin loss
- `CenterLoss`: Re-ID center embedding loss
- `HungarianMatchingLoss`: DETR bipartite matching
- `PhysicsConstraintLoss`: Ball physics violations

### Metrics (`utils/metrics.py`)

- `ReprojectionError`: Homography evaluation
- `KeypointPCK`: Percentage of Correct Keypoints
- `TrajectoryADE/FDE`: Displacement errors
- `MultiHorizonTrajectoryMetrics`: Errors at multiple time steps
- `ReIDMetrics`: mAP, CMC, Rank-k
- `DetectionMetrics`: mAP, AP50, AP75
- `FormationAccuracy`: Tactical classification

### Callbacks (`utils/callbacks.py`)

- `ModelCheckpoint`: Save best/periodic checkpoints
- `EarlyStopping`: Stop on metric plateau
- `LearningRateScheduler`: Cosine/Plateau/Step scheduling
- `WandbLogger`: Weights & Biases integration
- `VisualizationCallback`: Training visualizations
- `ProgressLogger`: Console logging

## Configuration Guide

All configs follow this structure:

```yaml
model:
  # Model architecture parameters
  
training:
  # Training hyperparameters
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  
data:
  # Dataset paths and parameters
  root_path: "data/training/..."
  
checkpoints:
  # Checkpoint settings
  save_dir: "models/checkpoints/..."
  
logging:
  # Logging configuration
  wandb:
    enabled: true
    project: "football-tracking-..."
```

### Customizing Configs

Edit YAML files to modify:
- Model architecture (layers, dimensions)
- Training hyperparameters (LR, batch size)
- Data augmentation settings
- Loss function weights
- Logging options

## Monitoring Training

### Weights & Biases

Enable in config:
```yaml
logging:
  wandb:
    enabled: true
    project: "my-project"
    entity: "my-username"
```

Login:
```bash
wandb login
```

### TensorBoard

```bash
tensorboard --logdir logs/
```

### Console Logs

Training progress logged with `loguru`:
```
[2024-01-01 12:00:00] INFO Epoch 1/100 - Loss: 0.1234
[2024-01-01 12:05:00] INFO Validation - mAP: 0.85, Rank-1: 0.92
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

Enable gradient accumulation (DETR):
```yaml
training:
  accumulation_steps: 4
```

Use mixed precision:
```yaml
mixed_precision: true
```

### Missing Data

Use synthetic data:
```yaml
data:
  dataset: "synthetic"  # Instead of soccernet
```

### Slow Training

Increase num_workers:
```yaml
data:
  num_workers: 8  # More parallel data loading
  pin_memory: true  # Faster GPU transfer
```

Enable cudnn benchmark:
```yaml
distributed:
  backend: "nccl"
  # Add to training code:
  torch.backends.cudnn.benchmark = True
```

## Hardware Requirements

**Minimum**:
- GPU: GTX 1660 (6GB)
- RAM: 16GB
- Storage: 500GB SSD

**Recommended**:
- GPU: RTX 3080+ (10GB+)
- RAM: 32GB+
- Storage: 2TB+ NVMe SSD

**For Full Pipeline**:
- Multi-GPU: 2-4 GPUs
- RAM: 64GB+
- Storage: 5TB+

## Advanced Usage

### Multi-GPU Training

Enable distributed training:
```yaml
distributed:
  enabled: true
  backend: "nccl"
  world_size: 4  # Number of GPUs
```

Run with torchrun:
```bash
torchrun --nproc_per_node=4 training/train_homography.py \
    --config training/configs/homography.yaml
```

### Resume Training

```bash
python training/train_homography.py \
    --config configs/homography.yaml \
    --resume models/checkpoints/homography/last.pth
```

### Fine-tuning

Load pretrained weights:
```python
checkpoint = torch.load('pretrained.pth')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

Freeze backbone:
```yaml
fine_tuning:
  enabled: true
  freeze_backbone: true
  epochs: 20
```

## Citation

If you use this training infrastructure, please cite:

```bibtex
@software{football_tracking_training,
  title={Training Infrastructure for Football Tracking},
  author={Football Tracking Team},
  year={2024},
  url={https://github.com/your-org/football-tracking}
}
```

## References

- SoccerNet: https://www.soccer-net.org/
- SkillCorner: https://github.com/SkillCorner/opendata
- Baller2Vec: https://arxiv.org/abs/2102.13653
- DETR: https://arxiv.org/abs/2005.12872

## Support

For issues or questions:
- Open GitHub issue
- Check documentation: docs/
- Contact: team@football-tracking.ai
