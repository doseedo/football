#!/bin/bash

# Master Training Script for Football Tracking Models
# Trains all models in dependency order with GPU allocation

set -e  # Exit on error

echo "==================================================================="
echo "Football Tracking - Complete Training Pipeline"
echo "==================================================================="

# Configuration
DEVICE=${DEVICE:-"cuda"}
DOWNLOAD_DATA=${DOWNLOAD_DATA:-false}
SKIP_EXISTING=${SKIP_EXISTING:-true}

# Directories
TRAINING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$TRAINING_DIR")"
DATA_DIR="$PROJECT_ROOT/data/training"
MODEL_DIR="$PROJECT_ROOT/models/checkpoints"

echo "Training Directory: $TRAINING_DIR"
echo "Project Root: $PROJECT_ROOT"
echo "Data Directory: $DATA_DIR"
echo "Model Directory: $MODEL_DIR"
echo "Device: $DEVICE"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$MODEL_DIR"

# Function to check if checkpoint exists
check_checkpoint() {
    local model_name=$1
    local checkpoint_path="$MODEL_DIR/$model_name/best.pth"

    if [ "$SKIP_EXISTING" = true ] && [ -f "$checkpoint_path" ]; then
        echo "✓ Found existing checkpoint: $checkpoint_path"
        return 0
    fi
    return 1
}

# Function to check data availability
check_data() {
    local dataset=$1
    local data_path="$DATA_DIR/$dataset"

    if [ -d "$data_path" ]; then
        echo "✓ Data found: $data_path"
        return 0
    else
        echo "✗ Data not found: $data_path"
        return 1
    fi
}

echo ""
echo "==================================================================="
echo "Step 1: Data Availability Check"
echo "==================================================================="

# Check for datasets
SOCCERNET_AVAILABLE=$(check_data "soccernet" && echo true || echo false)
SKILLCORNER_AVAILABLE=$(check_data "skillcorner" && echo true || echo false)

if [ "$DOWNLOAD_DATA" = true ]; then
    echo "Data download requested..."

    # Download SkillCorner (open data)
    if [ "$SKILLCORNER_AVAILABLE" = false ]; then
        echo "Downloading SkillCorner open data..."
        git clone https://github.com/SkillCorner/opendata.git "$DATA_DIR/skillcorner" || true
    fi

    # SoccerNet requires password
    if [ "$SOCCERNET_AVAILABLE" = false ]; then
        echo "SoccerNet download requires registration at soccer-net.org"
        echo "After obtaining password, run:"
        echo "  python -c \"from training.datasets import download_soccernet_data; download_soccernet_data('$DATA_DIR/soccernet', 'YOUR_PASSWORD')\""
    fi
fi

echo ""
echo "==================================================================="
echo "Step 2: Training Models in Dependency Order"
echo "==================================================================="

# 1. Homography Estimation (foundation for all other tasks)
echo ""
echo "-------------------------------------------------------------------"
echo "[1/6] Training Homography Estimation Model"
echo "-------------------------------------------------------------------"

if check_checkpoint "homography"; then
    echo "Skipping homography training (checkpoint exists)"
else
    echo "Starting homography training..."
    python "$TRAINING_DIR/train_homography.py" \
        --config "$TRAINING_DIR/configs/homography.yaml" \
        || echo "⚠ Homography training failed (non-critical)"
fi

# 2. Baller2Vec (trajectory embeddings)
echo ""
echo "-------------------------------------------------------------------"
echo "[2/6] Training Baller2Vec Model"
echo "-------------------------------------------------------------------"

if check_checkpoint "baller2vec"; then
    echo "Skipping Baller2Vec training (checkpoint exists)"
else
    echo "Starting Baller2Vec training..."
    # Note: Requires trajectory data
    if [ -f "$DATA_DIR/soccernet_tracking/train_annotations.json" ]; then
        python "$TRAINING_DIR/train_baller2vec.py" \
            --config "$TRAINING_DIR/configs/baller2vec.yaml" \
            --data "$DATA_DIR/trajectories.npz" \
            || echo "⚠ Baller2Vec training failed (missing data)"
    else
        echo "⚠ Skipping Baller2Vec (no tracking data available)"
    fi
fi

# 3. Ball 3D Tracking
echo ""
echo "-------------------------------------------------------------------"
echo "[3/6] Training Ball 3D Tracking Model"
echo "-------------------------------------------------------------------"

if check_checkpoint "ball3d"; then
    echo "Skipping Ball3D training (checkpoint exists)"
else
    echo "Starting Ball3D training..."
    python "$TRAINING_DIR/train_ball3d.py" \
        --config "$TRAINING_DIR/configs/ball3d.yaml" \
        || echo "⚠ Ball3D training failed (non-critical)"
fi

# 4. Player Re-Identification
echo ""
echo "-------------------------------------------------------------------"
echo "[4/6] Training Player Re-Identification Model"
echo "-------------------------------------------------------------------"

if check_checkpoint "reid"; then
    echo "Skipping Re-ID training (checkpoint exists)"
else
    echo "Starting Re-ID training..."
    python "$TRAINING_DIR/train_reid.py" \
        --config "$TRAINING_DIR/configs/reid.yaml" \
        || echo "⚠ Re-ID training failed (non-critical)"
fi

# 5. DETR Detection
echo ""
echo "-------------------------------------------------------------------"
echo "[5/6] Training DETR Detection Model"
echo "-------------------------------------------------------------------"

if check_checkpoint "detr"; then
    echo "Skipping DETR training (checkpoint exists)"
else
    echo "Starting DETR training..."
    python "$TRAINING_DIR/train_detr.py" \
        --config "$TRAINING_DIR/configs/detr.yaml" \
        || echo "⚠ DETR training failed (non-critical)"
fi

# 6. Graph Neural Network (requires all previous models)
echo ""
echo "-------------------------------------------------------------------"
echo "[6/6] Training Graph Neural Network Model"
echo "-------------------------------------------------------------------"

if check_checkpoint "gnn"; then
    echo "Skipping GNN training (checkpoint exists)"
else
    echo "Starting GNN training..."
    python "$TRAINING_DIR/train_gnn.py" \
        --config "$TRAINING_DIR/configs/gnn.yaml" \
        || echo "⚠ GNN training failed (non-critical)"
fi

echo ""
echo "==================================================================="
echo "Training Pipeline Complete!"
echo "==================================================================="

# Summary
echo ""
echo "Training Summary:"
echo "-------------------------------------------------------------------"

for model in homography baller2vec ball3d reid detr gnn; do
    checkpoint_path="$MODEL_DIR/$model/best.pth"
    if [ -f "$checkpoint_path" ]; then
        echo "✓ $model: Trained (checkpoint at $checkpoint_path)"
    else
        echo "✗ $model: Not trained"
    fi
done

echo ""
echo "Next Steps:"
echo "  1. Evaluate models: python scripts/evaluate_models.py"
echo "  2. Run inference: python -m src.main <video_path>"
echo "  3. View logs: tensorboard --logdir logs/"
echo ""

