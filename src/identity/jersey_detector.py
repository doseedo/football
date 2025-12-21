"""Jersey number region detector.

Localizes the jersey number region on player crops for OCR.
Handles both front and back numbers using attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import cv2


class SpatialAttention(nn.Module):
    """Spatial attention module for number region localization."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv2 = nn.Conv2d(in_channels // 4, 1, 1)

    def forward(self, x):
        """
        Args:
            x: Feature maps (B, C, H, W)

        Returns:
            Attention map (B, 1, H, W)
        """
        att = F.relu(self.conv1(x))
        att = torch.sigmoid(self.conv2(att))
        return att


class JerseyNumberDetector(nn.Module):
    """Detects jersey number region on player crops using attention.

    The network learns to localize number regions by attending to
    upper torso areas where numbers typically appear.
    """

    def __init__(self, backbone_channels: int = 512):
        super().__init__()

        # Lightweight backbone for feature extraction
        self.backbone = nn.Sequential(
            # Input: (B, 3, H, W)
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # -> H/2, W/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> H/4, W/4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> H/8, W/8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # -> H/16, W/16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, backbone_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
        )

        # Spatial attention for number localization
        self.spatial_attention = SpatialAttention(backbone_channels)

        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # (x1, y1, x2, y2)
        )

        # Confidence head
        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Player crops (B, 3, H, W)

        Returns:
            bboxes: Normalized bounding boxes (B, 4) in [0, 1]
            attention: Attention maps (B, 1, H, W)
            confidence: Detection confidence (B, 1)
        """
        # Extract features
        features = self.backbone(x)

        # Compute spatial attention
        attention = self.spatial_attention(features)

        # Apply attention to features
        attended_features = features * attention

        # Predict bounding box (normalized coordinates)
        bbox = self.bbox_head(attended_features)
        bbox = torch.sigmoid(bbox)  # Normalize to [0, 1]

        # Predict confidence
        confidence = self.conf_head(attended_features)

        return bbox, attention, confidence


class JerseyDetector:
    """High-level interface for jersey number detection."""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda',
                 use_heuristic: bool = True):
        """
        Args:
            model_path: Path to trained detector model
            device: Device to run on
            use_heuristic: If True, use heuristic method when model not available
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_heuristic = use_heuristic

        if model_path:
            self.model = JerseyNumberDetector()
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            self.use_model = True
        else:
            self.model = None
            self.use_model = False
            if use_heuristic:
                print("Jersey detector: Using heuristic method (no trained model)")

    def heuristic_detect(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Heuristic method for number region detection.

        Assumes number is in upper torso region:
        - Front number: top 30-50% of image, centered
        - Back number: top 20-40% of image, centered

        Args:
            image: Player crop (H, W, 3)

        Returns:
            bbox: Bounding box (x1, y1, x2, y2) in pixel coordinates
            confidence: Detection confidence
        """
        h, w = image.shape[:2]

        # Focus on upper torso (top 30-60% of player)
        y1 = int(h * 0.15)  # Start from 15% down
        y2 = int(h * 0.50)  # End at 50% down

        # Center horizontally with some margin
        x1 = int(w * 0.25)
        x2 = int(w * 0.75)

        # Ensure valid box
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Confidence based on image quality (simple heuristic)
        # Higher confidence for larger, clearer images
        confidence = min(1.0, (h * w) / (256 * 128))  # Normalized by typical size

        return bbox, confidence

    def refine_with_color(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Refine bounding box using color-based segmentation.

        Numbers are often white/bright on darker jersey.

        Args:
            image: Player crop (H, W, 3)
            bbox: Initial bounding box (x1, y1, x2, y2)

        Returns:
            Refined bounding box
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)

        # Ensure bbox is within image
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))

        # Extract ROI
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return bbox

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Find bright regions (potential numbers)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return bbox

        # Find the largest contour (likely the number)
        largest_contour = max(contours, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(largest_contour)

        # Convert back to original image coordinates
        refined_x1 = x1 + rx
        refined_y1 = y1 + ry
        refined_x2 = refined_x1 + rw
        refined_y2 = refined_y1 + rh

        # Add some margin
        margin_x = int(rw * 0.1)
        margin_y = int(rh * 0.1)

        refined_x1 = max(0, refined_x1 - margin_x)
        refined_y1 = max(0, refined_y1 - margin_y)
        refined_x2 = min(w, refined_x2 + margin_x)
        refined_y2 = min(h, refined_y2 + margin_y)

        return np.array([refined_x1, refined_y1, refined_x2, refined_y2],
                       dtype=np.float32)

    @torch.no_grad()
    def detect(self, images: np.ndarray, refine: bool = True) -> List[dict]:
        """
        Detect jersey number regions in player crops.

        Args:
            images: Player crops, shape (B, H, W, 3) or (H, W, 3)
            refine: Whether to refine boxes with color segmentation

        Returns:
            List of dicts with keys:
                - bbox: Bounding box (x1, y1, x2, y2) in pixel coordinates
                - confidence: Detection confidence [0, 1]
                - crop: Cropped number region (H, W, 3)
        """
        # Handle single image
        if images.ndim == 3:
            images = images[np.newaxis]

        batch_size, h, w = images.shape[:3]
        results = []

        if self.use_model and self.model is not None:
            # Use learned model
            # Prepare input
            x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
            x = x.to(self.device)

            # Detect
            bboxes, attentions, confidences = self.model(x)

            # Convert to numpy
            bboxes = bboxes.cpu().numpy()
            confidences = confidences.cpu().numpy()

            for i in range(batch_size):
                # Denormalize bbox
                bbox = bboxes[i]
                x1, y1, x2, y2 = bbox
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)

                bbox_pixels = np.array([x1, y1, x2, y2], dtype=np.float32)

                # Refine if requested
                if refine:
                    bbox_pixels = self.refine_with_color(images[i], bbox_pixels)

                # Crop number region
                x1, y1, x2, y2 = bbox_pixels.astype(int)
                x1 = max(0, min(x1, w - 1))
                x2 = max(x1 + 1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(y1 + 1, min(y2, h))

                crop = images[i, y1:y2, x1:x2]

                results.append({
                    'bbox': bbox_pixels,
                    'confidence': float(confidences[i, 0]),
                    'crop': crop
                })

        else:
            # Use heuristic method
            for i in range(batch_size):
                bbox, confidence = self.heuristic_detect(images[i])

                # Refine if requested
                if refine:
                    bbox = self.refine_with_color(images[i], bbox)

                # Crop number region
                x1, y1, x2, y2 = bbox.astype(int)
                x1 = max(0, min(x1, w - 1))
                x2 = max(x1 + 1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(y1 + 1, min(y2, h))

                crop = images[i, y1:y2, x1:x2]

                results.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'crop': crop
                })

        return results

    def detect_single(self, image: np.ndarray, refine: bool = True) -> dict:
        """
        Detect jersey number region in a single image.

        Args:
            image: Player crop (H, W, 3)
            refine: Whether to refine box with color segmentation

        Returns:
            Dict with bbox, confidence, and crop
        """
        results = self.detect(image[np.newaxis], refine=refine)
        return results[0]


def visualize_detection(image: np.ndarray, bbox: np.ndarray,
                       confidence: float) -> np.ndarray:
    """
    Visualize detection result.

    Args:
        image: Player crop (H, W, 3)
        bbox: Bounding box (x1, y1, x2, y2)
        confidence: Detection confidence

    Returns:
        Visualization image
    """
    vis = image.copy()
    x1, y1, x2, y2 = bbox.astype(int)

    # Draw bounding box
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw confidence
    text = f"Conf: {confidence:.2f}"
    cv2.putText(vis, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (0, 255, 0), 2)

    return vis
