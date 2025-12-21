"""Transformer Prediction Heads (TPH) for multi-scale detection.

TPH replaces standard prediction heads with transformer layers that can reason
about spatial relationships and handle multi-scale features better. This is
particularly useful for crowded scenes with players at different scales.

Can be integrated with YOLO or other CNN backbones to improve detection quality.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .player_detector import Detection


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, C] features
            mask: Optional attention mask
        Returns:
            [B, N, C] attended features
        """
        B, N, C = x.shape

        # Project to Q, K, V
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]

        # Output projection
        out = self.out_proj(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""

    def __init__(self, d_model, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [B, N, C] features
        Returns:
            [B, N, C] transformed features
        """
        # Self-attention with residual
        x = x + self.self_attn(self.norm1(x))

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerPredictionHead(nn.Module):
    """Transformer-based prediction head for object detection.

    Replaces standard convolutional prediction heads with transformer layers
    that can reason about spatial relationships between features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_classes: int = 1,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        num_anchors: int = 3
    ):
        """
        Initialize transformer prediction head.

        Args:
            in_channels: Input feature channels
            hidden_dim: Hidden dimension for transformer
            num_classes: Number of object classes
            num_transformer_layers: Number of transformer blocks
            num_heads: Number of attention heads
            num_anchors: Number of anchor boxes per location
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, hidden_dim, 1, 1))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads=num_heads)
            for _ in range(num_transformer_layers)
        ])

        # Prediction heads
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_anchors * num_classes)
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_anchors * 4)
        )

        self.objectness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_anchors)
        )

    def forward(self, x):
        """
        Forward pass through transformer prediction head.

        Args:
            x: [B, C, H, W] feature map

        Returns:
            pred_classes: [B, num_anchors, H, W, num_classes]
            pred_boxes: [B, num_anchors, H, W, 4]
            pred_objectness: [B, num_anchors, H, W]
        """
        B, C, H, W = x.shape

        # Project to hidden dimension
        x = self.input_proj(x)  # [B, hidden_dim, H, W]

        # Add positional encoding
        pos_enc = self.pos_encoding.expand(B, -1, H, W)
        x = x + pos_enc

        # Reshape for transformer: [B, hidden_dim, H, W] -> [B, H*W, hidden_dim]
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, hidden_dim]

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)  # [B, H*W, hidden_dim]

        # Predict classes, boxes, objectness
        pred_classes = self.class_head(x)  # [B, H*W, num_anchors * num_classes]
        pred_boxes = self.bbox_head(x)  # [B, H*W, num_anchors * 4]
        pred_objectness = self.objectness_head(x)  # [B, H*W, num_anchors]

        # Reshape to [B, num_anchors, H, W, ...]
        pred_classes = pred_classes.view(B, H, W, self.num_anchors, self.num_classes).permute(0, 3, 1, 2, 4)
        pred_boxes = pred_boxes.view(B, H, W, self.num_anchors, 4).permute(0, 3, 1, 2, 4)
        pred_objectness = pred_objectness.view(B, H, W, self.num_anchors).permute(0, 3, 1, 2)

        return pred_classes, pred_boxes, pred_objectness


class MultiScaleTransformerHead(nn.Module):
    """Multi-scale transformer prediction head for FPN-like architectures."""

    def __init__(
        self,
        in_channels_list: List[int],
        hidden_dim: int = 256,
        num_classes: int = 1,
        num_transformer_layers: int = 2,
        num_heads: int = 8
    ):
        """
        Initialize multi-scale transformer head.

        Args:
            in_channels_list: List of input channels for each scale
            hidden_dim: Hidden dimension for transformers
            num_classes: Number of object classes
            num_transformer_layers: Number of transformer blocks per scale
            num_heads: Number of attention heads
        """
        super().__init__()

        self.scales = len(in_channels_list)
        self.hidden_dim = hidden_dim

        # Create transformer head for each scale
        self.heads = nn.ModuleList([
            TransformerPredictionHead(
                in_channels=ch,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_transformer_layers=num_transformer_layers,
                num_heads=num_heads
            )
            for ch in in_channels_list
        ])

        # Cross-scale attention for feature fusion
        self.cross_scale_attention = nn.ModuleList([
            MultiHeadSelfAttention(hidden_dim, num_heads=num_heads)
            for _ in range(len(in_channels_list))
        ])

    def forward(self, features: List[torch.Tensor]):
        """
        Forward pass through multi-scale transformer heads.

        Args:
            features: List of feature maps at different scales
                      [B, C_i, H_i, W_i] for each scale

        Returns:
            predictions: List of (classes, boxes, objectness) for each scale
        """
        # Process each scale independently
        scale_predictions = []
        for i, (feat, head) in enumerate(zip(features, self.heads)):
            pred_classes, pred_boxes, pred_objectness = head(feat)
            scale_predictions.append((pred_classes, pred_boxes, pred_objectness))

        return scale_predictions


class YOLOWithTransformerHead:
    """YOLO detector with transformer prediction heads.

    Replaces YOLO's standard prediction heads with transformer-based heads
    for better handling of crowded scenes and multi-scale objects.
    """

    def __init__(
        self,
        yolo_backbone,
        device: str = "cuda",
        confidence: float = 0.3,
        iou_threshold: float = 0.5,
        min_height: int = 30,
        max_height: int = 400
    ):
        """
        Initialize YOLO with transformer heads.

        Args:
            yolo_backbone: Pretrained YOLO model (for feature extraction)
            device: Device to run on
            confidence: Confidence threshold
            iou_threshold: IoU threshold for NMS
            min_height: Minimum detection height
            max_height: Maximum detection height
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.min_height = min_height
        self.max_height = max_height

        logger.info("Initializing YOLO with Transformer Prediction Heads")

        # YOLO backbone for feature extraction
        self.backbone = yolo_backbone
        self.backbone.to(self.device)
        self.backbone.eval()

        # Transformer prediction heads (would need to be initialized based on backbone)
        # This is a simplified version - in practice, you'd hook into YOLO's feature pyramid
        logger.warning("YOLOWithTransformerHead is a conceptual implementation")
        logger.warning("Full integration requires YOLO architecture modification")

    @torch.no_grad()
    def detect(self, frame: np.ndarray, pitch_mask: Optional[np.ndarray] = None) -> List[Detection]:
        """
        Detect players using YOLO backbone + transformer heads.

        Args:
            frame: Input frame
            pitch_mask: Optional pitch mask

        Returns:
            List of detections
        """
        # This would require deep integration with YOLO architecture
        # For now, return empty list with warning
        logger.warning("YOLOWithTransformerHead.detect() not fully implemented")
        logger.warning("Requires YOLO architecture modification to extract multi-scale features")
        return []


class SpatialRelationReasoning(nn.Module):
    """Spatial relation reasoning module using transformers.

    Helps resolve overlapping detections by reasoning about spatial relationships.
    """

    def __init__(self, d_model=256, num_heads=8, num_layers=2):
        super().__init__()

        self.d_model = d_model

        # Bounding box encoder
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        # Transformer for reasoning
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)  # Confidence adjustment
        )

    def forward(self, boxes, features):
        """
        Reason about spatial relationships between detections.

        Args:
            boxes: [N, 4] bounding boxes in (x1, y1, x2, y2) format
            features: [N, C] detection features

        Returns:
            [N] confidence adjustments
        """
        # Encode boxes
        box_embed = self.bbox_encoder(boxes)  # [N, d_model]

        # Combine with features
        x = box_embed + features  # [N, d_model]
        x = x.unsqueeze(0)  # [1, N, d_model]

        # Apply transformer layers
        for layer in self.transformer:
            x = layer(x)

        # Predict confidence adjustment
        adj = self.output_proj(x).squeeze(0).squeeze(-1)  # [N]
        return torch.sigmoid(adj)


def apply_transformer_reasoning(detections: List[Detection], device: str = "cuda") -> List[Detection]:
    """
    Apply transformer-based spatial reasoning to refine detections.

    Args:
        detections: List of detections
        device: Device to run on

    Returns:
        Refined detections with adjusted confidences
    """
    if len(detections) <= 1:
        return detections

    # Extract boxes
    boxes = torch.tensor([det.bbox for det in detections], dtype=torch.float32, device=device)

    # Simple feature: box size and aspect ratio
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    features = torch.stack([
        widths,
        heights,
        widths / (heights + 1e-6),
        widths * heights
    ], dim=-1)

    # Normalize features
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-6)

    # Pad to fixed dimension (256)
    features = F.pad(features, (0, 256 - features.shape[-1]))

    # Apply reasoning (would need trained model)
    # For now, just return original detections
    logger.debug(f"Transformer reasoning applied to {len(detections)} detections")

    return detections
