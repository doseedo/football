#!/bin/bash

# Download YOLOv8 models for football tracking

set -e

MODELS_DIR="${1:-models}"

echo "Creating models directory: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

echo "Downloading YOLOv8x model..."
wget -q --show-progress -O "$MODELS_DIR/yolov8x.pt" \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

echo "Download complete!"
echo "Model saved to: $MODELS_DIR/yolov8x.pt"
