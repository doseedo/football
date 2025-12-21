"""HRNet-based keypoint detector for automatic pitch calibration.

This module implements a deep learning-based keypoint detector using HRNetv2-W32
to predict heatmaps for football pitch keypoints, enabling automatic camera calibration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not available, HRNet keypoint detection will not work")


@dataclass
class DetectedKeypoint:
    """Represents a detected keypoint with confidence."""

    name: str
    pixel_coords: Tuple[float, float]  # (x, y) in pixels
    confidence: float  # 0 to 1
    heatmap_value: float  # Raw heatmap activation
    world_coords: Optional[Tuple[float, float]] = None  # Expected world coords


@dataclass
class KeypointDetectionResult:
    """Result of keypoint detection on a frame."""

    keypoints: List[DetectedKeypoint]
    heatmaps: np.ndarray  # [num_keypoints, H, W]
    confidence_threshold: float
    num_detected: int
    frame_shape: Tuple[int, int]  # (H, W)

    def get_high_confidence_keypoints(self, min_confidence: float = 0.5) -> List[DetectedKeypoint]:
        """Get only keypoints above confidence threshold."""
        return [kp for kp in self.keypoints if kp.confidence >= min_confidence]

    def get_keypoint_by_name(self, name: str) -> Optional[DetectedKeypoint]:
        """Get a specific keypoint by name."""
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None


class HRNetKeypointHead(nn.Module):
    """Keypoint detection head for HRNet backbone.

    Converts HRNet features to keypoint heatmaps.
    """

    def __init__(self, in_channels: int, num_keypoints: int, hidden_dim: int = 256):
        """Initialize keypoint head.

        Args:
            in_channels: Number of input channels from HRNet
            num_keypoints: Number of keypoints to detect
            hidden_dim: Hidden dimension for intermediate layers
        """
        super().__init__()

        self.num_keypoints = num_keypoints

        # Upsampling path with conv layers
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim // 2)

        # Final heatmap prediction
        self.heatmap_conv = nn.Conv2d(hidden_dim // 2, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate heatmaps.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Heatmaps [B, num_keypoints, H, W]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Generate heatmaps
        heatmaps = self.heatmap_conv(x)

        # Upsample to higher resolution if needed
        heatmaps = F.interpolate(
            heatmaps,
            scale_factor=4,
            mode='bilinear',
            align_corners=False
        )

        return heatmaps


class PitchKeypointDetector(nn.Module):
    """HRNet-based keypoint detector for football pitch.

    Uses HRNetv2-W32 as backbone with custom keypoint detection head
    to predict heatmaps for 57+ pitch keypoints.
    """

    def __init__(
        self,
        num_keypoints: int = 57,
        backbone: str = "hrnet_w32",
        pretrained: bool = True,
        confidence_threshold: float = 0.3
    ):
        """Initialize keypoint detector.

        Args:
            num_keypoints: Number of keypoints to detect
            backbone: Backbone architecture (default: hrnet_w32)
            pretrained: Use ImageNet pretrained weights
            confidence_threshold: Minimum confidence for keypoint detection
        """
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for HRNet keypoint detection")

        self.num_keypoints = num_keypoints
        self.confidence_threshold = confidence_threshold

        # Load HRNet backbone from timm
        try:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=[3]  # Use final high-resolution features
            )
            # Get number of output channels
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 256, 256)
                features = self.backbone(dummy_input)
                backbone_channels = features[0].shape[1]

            logger.info(f"Loaded {backbone} backbone with {backbone_channels} output channels")
        except Exception as e:
            logger.error(f"Failed to load HRNet backbone: {e}")
            # Fallback: create simple backbone
            backbone_channels = 32
            self.backbone = self._create_simple_backbone(backbone_channels)
            logger.warning("Using simple fallback backbone")

        # Keypoint detection head
        self.keypoint_head = HRNetKeypointHead(
            in_channels=backbone_channels,
            num_keypoints=num_keypoints,
            hidden_dim=256
        )

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _create_simple_backbone(self, out_channels: int) -> nn.Module:
        """Create a simple CNN backbone as fallback."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate keypoint heatmaps.

        Args:
            x: Input images [B, 3, H, W], normalized [0, 1]

        Returns:
            Heatmaps [B, num_keypoints, H, W]
        """
        # Normalize input
        x = (x - self.mean) / self.std

        # Extract features
        if isinstance(self.backbone, nn.Sequential):
            features = self.backbone(x)
        else:
            features = self.backbone(x)[0]  # timm returns list

        # Generate heatmaps
        heatmaps = self.keypoint_head(features)

        return heatmaps

    @torch.no_grad()
    def detect_keypoints(
        self,
        frame: np.ndarray,
        keypoint_names: List[str],
        nms_kernel: int = 5,
        min_confidence: Optional[float] = None
    ) -> KeypointDetectionResult:
        """Detect keypoints in a frame.

        Args:
            frame: Input frame [H, W, 3] in BGR format
            keypoint_names: Names of keypoints corresponding to heatmap channels
            nms_kernel: Kernel size for non-maximum suppression
            min_confidence: Minimum confidence threshold (uses default if None)

        Returns:
            KeypointDetectionResult with detected keypoints
        """
        if min_confidence is None:
            min_confidence = self.confidence_threshold

        # Preprocess frame
        input_tensor = self._preprocess_frame(frame)

        # Run inference
        self.eval()
        heatmaps = self.forward(input_tensor)  # [1, num_keypoints, H, W]

        # Convert to numpy
        heatmaps_np = heatmaps[0].cpu().numpy()  # [num_keypoints, H, W]

        # Apply sigmoid to get probabilities
        heatmaps_np = 1 / (1 + np.exp(-heatmaps_np))  # Sigmoid

        # Extract keypoint locations
        keypoints = self._extract_keypoints_from_heatmaps(
            heatmaps_np,
            keypoint_names,
            nms_kernel=nms_kernel,
            min_confidence=min_confidence,
            original_shape=frame.shape[:2]
        )

        return KeypointDetectionResult(
            keypoints=keypoints,
            heatmaps=heatmaps_np,
            confidence_threshold=min_confidence,
            num_detected=len([kp for kp in keypoints if kp.confidence >= min_confidence]),
            frame_shape=frame.shape[:2]
        )

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input.

        Args:
            frame: Input frame [H, W, 3] in BGR format

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to standard size
        frame_resized = cv2.resize(frame_rgb, (512, 512))

        # Convert to float and normalize to [0, 1]
        frame_float = frame_resized.astype(np.float32) / 255.0

        # Convert to tensor [1, 3, H, W]
        tensor = torch.from_numpy(frame_float).permute(2, 0, 1).unsqueeze(0)

        # Move to same device as model
        device = next(self.parameters()).device
        tensor = tensor.to(device)

        return tensor

    def _extract_keypoints_from_heatmaps(
        self,
        heatmaps: np.ndarray,
        keypoint_names: List[str],
        nms_kernel: int = 5,
        min_confidence: float = 0.3,
        original_shape: Tuple[int, int] = (1080, 1920)
    ) -> List[DetectedKeypoint]:
        """Extract keypoint locations from heatmaps using NMS.

        Args:
            heatmaps: Heatmaps [num_keypoints, H, W]
            keypoint_names: Names for each keypoint
            nms_kernel: Kernel size for non-maximum suppression
            min_confidence: Minimum confidence threshold
            original_shape: Original frame shape for coordinate scaling

        Returns:
            List of detected keypoints
        """
        keypoints = []
        heatmap_h, heatmap_w = heatmaps.shape[1:]
        orig_h, orig_w = original_shape

        for i, name in enumerate(keypoint_names):
            if i >= len(heatmaps):
                break

            heatmap = heatmaps[i]

            # Apply non-maximum suppression
            max_val = self._nms_heatmap(heatmap, kernel_size=nms_kernel)

            # Find location of maximum
            max_loc = np.unravel_index(np.argmax(max_val), max_val.shape)
            confidence = float(max_val[max_loc])

            if confidence >= min_confidence:
                # Convert to original frame coordinates
                y_heatmap, x_heatmap = max_loc
                x_pixel = (x_heatmap / heatmap_w) * orig_w
                y_pixel = (y_heatmap / heatmap_h) * orig_h

                keypoints.append(DetectedKeypoint(
                    name=name,
                    pixel_coords=(float(x_pixel), float(y_pixel)),
                    confidence=confidence,
                    heatmap_value=float(heatmap[max_loc])
                ))

        return keypoints

    def _nms_heatmap(self, heatmap: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply non-maximum suppression to heatmap.

        Args:
            heatmap: Input heatmap [H, W]
            kernel_size: Size of NMS kernel

        Returns:
            Suppressed heatmap [H, W]
        """
        # Use max pooling for NMS
        pad = kernel_size // 2
        heatmap_padded = np.pad(heatmap, pad, mode='constant', constant_values=0)

        # Max pooling
        max_pooled = np.zeros_like(heatmap)
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                region = heatmap_padded[i:i+kernel_size, j:j+kernel_size]
                max_pooled[i, j] = np.max(region)

        # Keep only local maxima
        nms_heatmap = np.where(heatmap == max_pooled, heatmap, 0)

        return nms_heatmap

    def load_weights(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
            logger.info(f"Loaded weights from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load weights from {checkpoint_path}: {e}")
            raise


def create_keypoint_detector(
    num_keypoints: int = 57,
    device: str = 'cpu',
    checkpoint_path: Optional[str] = None
) -> PitchKeypointDetector:
    """Create and initialize a keypoint detector.

    Args:
        num_keypoints: Number of keypoints to detect
        device: Device to run model on ('cpu' or 'cuda')
        checkpoint_path: Path to pretrained weights (optional)

    Returns:
        Initialized PitchKeypointDetector
    """
    model = PitchKeypointDetector(
        num_keypoints=num_keypoints,
        backbone="hrnet_w32",
        pretrained=True,
        confidence_threshold=0.3
    )

    # Load checkpoint if provided
    if checkpoint_path is not None:
        model.load_weights(checkpoint_path)

    # Move to device
    model = model.to(device)
    model.eval()

    logger.info(f"Created keypoint detector with {num_keypoints} keypoints on {device}")

    return model


def visualize_detections(
    frame: np.ndarray,
    result: KeypointDetectionResult,
    min_confidence: float = 0.3,
    show_heatmaps: bool = False
) -> np.ndarray:
    """Visualize detected keypoints on frame.

    Args:
        frame: Input frame [H, W, 3]
        result: Detection result
        min_confidence: Minimum confidence to display
        show_heatmaps: Whether to overlay heatmaps

    Returns:
        Annotated frame
    """
    vis_frame = frame.copy()

    # Draw keypoints
    for kp in result.keypoints:
        if kp.confidence >= min_confidence:
            x, y = int(kp.pixel_coords[0]), int(kp.pixel_coords[1])

            # Color based on confidence (green = high, yellow = medium, red = low)
            if kp.confidence > 0.7:
                color = (0, 255, 0)
            elif kp.confidence > 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)

            # Draw circle
            cv2.circle(vis_frame, (x, y), 5, color, -1)
            cv2.circle(vis_frame, (x, y), 7, (255, 255, 255), 2)

            # Draw label
            label = f"{kp.name[:12]}: {kp.confidence:.2f}"
            cv2.putText(vis_frame, label, (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Optionally overlay heatmaps
    if show_heatmaps and result.heatmaps is not None:
        # Sum all heatmaps for visualization
        heatmap_sum = np.sum(result.heatmaps, axis=0)
        heatmap_sum = (heatmap_sum / heatmap_sum.max() * 255).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap_sum, (frame.shape[1], frame.shape[0]))
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        vis_frame = cv2.addWeighted(vis_frame, 0.7, heatmap_colored, 0.3, 0)

    return vis_frame
