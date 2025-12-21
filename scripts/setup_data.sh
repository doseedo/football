#!/bin/bash
# One-command data setup for Football XY Tracking
set -e

echo "=========================================="
echo "Football XY Tracking - Data Setup"
echo "=========================================="

# Create directories
mkdir -p data/{soccernet,spiideo,wyscout,models}

# Install dependencies
echo -e "\n[1/4] Installing dependencies..."
pip install -q SoccerNet gdown torch torchvision tqdm opencv-python

# Download SoccerNet
echo -e "\n[2/4] Downloading SoccerNet dataset..."
echo "Note: This requires a free account at https://www.soccer-net.org/"
echo "You'll be prompted for credentials on first run."
python scripts/download_soccernet.py --output data/soccernet --task all

# Download pretrained model weights
echo -e "\n[3/4] Downloading pretrained model weights..."
mkdir -p models/pretrained

# YOLOv8 (will auto-download on first use, but we can pre-cache)
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')" 2>/dev/null || echo "YOLOv8 will download on first use"

# OSNet weights for re-id (from torchreid)
OSNET_URL="https://drive.google.com/uc?id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e"
if [ ! -f "models/pretrained/osnet_x1_0.pth" ]; then
    echo "Downloading OSNet weights..."
    gdown "$OSNET_URL" -O models/pretrained/osnet_x1_0.pth 2>/dev/null || echo "OSNet download failed - will train from scratch"
fi

# DETR pretrained (COCO)
DETR_URL="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
if [ ! -f "models/pretrained/detr-r50.pth" ]; then
    echo "Downloading DETR weights..."
    wget -q "$DETR_URL" -O models/pretrained/detr-r50.pth 2>/dev/null || echo "DETR download failed - will train from scratch"
fi

echo -e "\n[4/4] Setup complete!"
echo ""
echo "=========================================="
echo "Data directories:"
echo "  - SoccerNet: data/soccernet/"
echo "  - Spiideo:   data/spiideo/   (add your data here)"  
echo "  - Wyscout:   data/wyscout/   (add your data here)"
echo "  - Models:    models/pretrained/"
echo ""
echo "Next steps:"
echo "  1. Add your Spiideo data to data/spiideo/"
echo "  2. Add your Wyscout data to data/wyscout/"
echo "  3. Run: python training/datasets/spiideo_loader.py --data-dir data/spiideo --convert"
echo "  4. Run: python training/datasets/wyscout_loader.py --data-dir data/wyscout --convert"
echo "  5. Start training: cd training && bash run_all_training.sh"
echo "=========================================="
