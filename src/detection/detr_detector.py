"""DETR (DEtection TRansformer) based player detector for crowded scenes.

DETR uses a transformer encoder-decoder architecture with set-based predictions
and Hungarian matching, eliminating the need for NMS and handling overlapping
players better than standard object detectors.

Key advantages:
- No NMS needed (set-based prediction)
- Better for overlapping/crowded players
- End-to-end trainable with Hungarian matching
- Global reasoning via self-attention
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from scipy.optimize import linear_sum_assignment

from .player_detector import Detection


class PositionEmbeddingSine:
    """Sine-based positional encoding for transformer."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale

    def __call__(self, x):
        """Generate positional embeddings."""
        B, C, H, W = x.shape
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=x.device)

        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TransformerEncoder(nn.Module):
    """Transformer encoder for DETR."""

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.d_model = d_model

    def forward(self, src, pos_embed):
        """
        Args:
            src: [B, C, H, W] feature map
            pos_embed: [B, C, H, W] positional embeddings
        Returns:
            [HW, B, C] encoded features
        """
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [HW, B, C]

        memory = self.encoder(src + pos_embed)
        return memory


class TransformerDecoder(nn.Module):
    """Transformer decoder for DETR."""

    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.d_model = d_model

    def forward(self, tgt, memory, pos_embed):
        """
        Args:
            tgt: [num_queries, B, C] object queries
            memory: [HW, B, C] encoder output
            pos_embed: [HW, B, C] positional embeddings
        Returns:
            [num_queries, B, C] decoded features
        """
        output = self.decoder(tgt, memory, memory_key_padding_mask=None, pos=pos_embed)
        return output


class DETR(nn.Module):
    """DETR model with ResNet-50 backbone."""

    def __init__(self, num_classes=2, num_queries=100, hidden_dim=256):
        super().__init__()

        # Backbone: ResNet-50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc

        # Reduce channel dimension from 2048 to hidden_dim
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # Positional encoding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # Transformer
        self.transformer_encoder = TransformerEncoder(d_model=hidden_dim)
        self.transformer_decoder = TransformerDecoder(d_model=hidden_dim)

        # Object queries (learnable)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input image
        Returns:
            pred_logits: [B, num_queries, num_classes + 1]
            pred_boxes: [B, num_queries, 4] in (cx, cy, w, h) format normalized to [0, 1]
        """
        # Extract features
        features = self.backbone(x)  # [B, 2048, H/32, W/32]
        features = self.conv(features)  # [B, hidden_dim, H/32, W/32]

        # Positional encoding
        pos_embed = self.position_embedding(features)

        # Transformer encoder
        memory = self.transformer_encoder(features, pos_embed)  # [HW, B, hidden_dim]

        # Transformer decoder
        B = x.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, hidden_dim]
        pos_embed_flat = pos_embed.flatten(2).permute(2, 0, 1)  # [HW, B, hidden_dim]

        hs = self.transformer_decoder(query_embed, memory, pos_embed_flat)  # [num_queries, B, hidden_dim]
        hs = hs.permute(1, 0, 2)  # [B, num_queries, hidden_dim]

        # Prediction
        outputs_class = self.class_embed(hs)  # [B, num_queries, num_classes + 1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [B, num_queries, 4]

        return outputs_class, outputs_coord


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class HungarianMatcher:
    """Hungarian matching for set-based prediction."""

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(self, outputs, targets):
        """
        Performs Hungarian matching between predictions and targets.

        Args:
            outputs: dict with 'pred_logits' [B, num_queries, num_classes]
                     and 'pred_boxes' [B, num_queries, 4]
            targets: list of dict with 'labels' and 'boxes'
        Returns:
            List of (pred_idx, target_idx) tuples for each batch
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]

        # Flatten batch dimension
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # [B*num_queries, num_classes]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [B*num_queries, 4]

        # Concatenate all target labels and boxes
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])

        # Compute classification cost
        cost_class = -out_prob[:, tgt_ids]

        # Compute L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute GIoU cost
        cost_giou = -self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(out_bbox),
            self.box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        # Hungarian algorithm
        sizes = [len(v['boxes']) for v in targets]
        indices = []
        offset = 0
        for i, c in enumerate(C.split(sizes, -1)):
            indices.append(linear_sum_assignment(c[i]))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)."""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        """Compute generalized IoU between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        union = area1[:, None] + area2 - inter
        iou = inter / union

        # Generalized IoU
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area


class DETRDetector:
    """DETR-based player detector for crowded scenes.

    Advantages over YOLO:
    - Better handling of overlapping players
    - No NMS needed (set-based prediction)
    - Global reasoning via transformer attention
    - End-to-end trainable
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        num_queries: int = 100,
        confidence: float = 0.7,
        min_height: int = 30,
        max_height: int = 400,
        image_size: Tuple[int, int] = (800, 800)
    ):
        """
        Initialize DETR detector.

        Args:
            model_path: Path to pretrained weights (optional)
            device: Device to run on ('cuda' or 'cpu')
            num_queries: Number of object queries
            confidence: Confidence threshold for detections
            min_height: Minimum detection height
            max_height: Maximum detection height
            image_size: Input image size (H, W)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_queries = num_queries
        self.confidence = confidence
        self.min_height = min_height
        self.max_height = max_height
        self.image_size = image_size

        logger.info(f"Initializing DETR detector with {num_queries} queries on {self.device}")

        # Initialize model
        self.model = DETR(num_classes=1, num_queries=num_queries)  # Person class only
        self.model.to(self.device)

        # Load pretrained weights if provided
        if model_path:
            logger.info(f"Loading DETR weights from: {model_path}")
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("DETR weights loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load DETR weights: {e}")
                logger.info("Using randomly initialized weights (backbone is pretrained)")
        else:
            logger.info("No pretrained weights provided, using ImageNet pretrained backbone")

        self.model.eval()

        # Normalization for ImageNet pretrained backbone
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def preprocess(self, frame: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess frame for DETR."""
        original_size = frame.shape[:2]  # (H, W)

        # Convert to tensor and normalize
        img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        img = img.unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # Resize
        img = F.interpolate(img, size=self.image_size, mode='bilinear', align_corners=False)

        # Normalize with ImageNet stats
        img = (img - self.mean) / self.std

        return img, original_size

    @torch.no_grad()
    def detect(self, frame: np.ndarray, pitch_mask: Optional[np.ndarray] = None) -> List[Detection]:
        """
        Detect players in a frame using DETR.

        Args:
            frame: Input frame [H, W, 3]
            pitch_mask: Optional pitch mask to filter detections

        Returns:
            List of Detection objects
        """
        # Preprocess
        img, original_size = self.preprocess(frame)

        # Forward pass
        pred_logits, pred_boxes = self.model(img)

        # Get predictions
        pred_logits = pred_logits[0]  # [num_queries, num_classes + 1]
        pred_boxes = pred_boxes[0]  # [num_queries, 4] in (cx, cy, w, h) normalized

        # Apply softmax to get probabilities
        prob = F.softmax(pred_logits, dim=-1)
        scores = prob[:, :-1].max(dim=-1)[0]  # Exclude no-object class

        # Filter by confidence
        keep = scores > self.confidence
        scores = scores[keep]
        boxes = pred_boxes[keep]

        # Convert boxes to pixel coordinates
        h, w = original_size
        boxes_xyxy = self.box_cxcywh_to_xyxy(boxes)
        boxes_xyxy[:, [0, 2]] *= w
        boxes_xyxy[:, [1, 3]] *= h

        # Convert to detections
        detections = []
        for box, score in zip(boxes_xyxy, scores):
            bbox = box.cpu().numpy()
            height = bbox[3] - bbox[1]

            # Filter by height
            if height < self.min_height or height > self.max_height:
                continue

            # Filter by pitch mask
            if pitch_mask is not None:
                cx, cy = (bbox[0] + bbox[2]) / 2, bbox[3]
                px, py = int(cx), int(cy)
                h_mask, w_mask = pitch_mask.shape[:2]
                if 0 <= px < w_mask and 0 <= py < h_mask:
                    if pitch_mask[py, px] == 0:
                        continue

            detections.append(Detection(
                bbox=bbox,
                confidence=float(score),
                class_id=self.PERSON_CLASS_ID
            ))

        return detections

    @torch.no_grad()
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect players in a batch of frames.

        Args:
            frames: List of input frames

        Returns:
            List of detection lists (one per frame)
        """
        # Preprocess all frames
        imgs = []
        original_sizes = []
        for frame in frames:
            img, original_size = self.preprocess(frame)
            imgs.append(img)
            original_sizes.append(original_size)

        # Batch
        imgs = torch.cat(imgs, dim=0)  # [B, 3, H, W]

        # Forward pass
        pred_logits, pred_boxes = self.model(imgs)

        # Process each frame
        all_detections = []
        for i in range(len(frames)):
            logits = pred_logits[i]  # [num_queries, num_classes + 1]
            boxes = pred_boxes[i]  # [num_queries, 4]

            # Get probabilities
            prob = F.softmax(logits, dim=-1)
            scores = prob[:, :-1].max(dim=-1)[0]

            # Filter by confidence
            keep = scores > self.confidence
            scores = scores[keep]
            boxes = boxes[keep]

            # Convert to pixel coordinates
            h, w = original_sizes[i]
            boxes_xyxy = self.box_cxcywh_to_xyxy(boxes)
            boxes_xyxy[:, [0, 2]] *= w
            boxes_xyxy[:, [1, 3]] *= h

            # Convert to detections
            frame_detections = []
            for box, score in zip(boxes_xyxy, scores):
                bbox = box.cpu().numpy()
                height = bbox[3] - bbox[1]

                if self.min_height <= height <= self.max_height:
                    frame_detections.append(Detection(
                        bbox=bbox,
                        confidence=float(score),
                        class_id=self.PERSON_CLASS_ID
                    ))

            all_detections.append(frame_detections)

        return all_detections

    @staticmethod
    def box_cxcywh_to_xyxy(boxes):
        """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
