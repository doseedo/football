"""
Loss Functions for Football Tracking Models

Includes:
- Keypoint detection losses (homography)
- Trajectory prediction losses (Baller2Vec, Ball3D)
- Re-identification losses (triplet, center)
- Detection losses (DETR Hungarian matching)
- Physics-based losses (Ball3D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional, Tuple


class KeypointHeatmapLoss(nn.Module):
    """
    Combined loss for keypoint heatmap prediction.

    Combines:
    - MSE loss for heatmap regression
    - Focal loss to handle class imbalance (most pixels are background)
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_heatmaps: (B, K, H, W) - predicted heatmaps
            target_heatmaps: (B, K, H, W) - ground truth heatmaps

        Returns:
            Total loss
        """
        # MSE loss
        mse_loss = F.mse_loss(pred_heatmaps, target_heatmaps)

        # Focal loss
        pred_prob = torch.sigmoid(pred_heatmaps)
        pt = torch.where(target_heatmaps == 1, pred_prob, 1 - pred_prob)
        focal_loss = -self.focal_alpha * (1 - pt) ** self.focal_gamma * torch.log(pt + 1e-8)
        focal_loss = focal_loss.mean()

        return self.mse_weight * mse_loss + self.focal_weight * focal_loss


class ReprojectionLoss(nn.Module):
    """
    Reprojection error for homography estimation.

    Measures geometric consistency of predicted homography.
    """

    def __init__(self, loss_type: str = "l2"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        pred_homography: torch.Tensor,
        pixel_points: torch.Tensor,
        world_points: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_homography: (B, 3, 3) - predicted homography matrices
            pixel_points: (B, N, 2) - pixel coordinates
            world_points: (B, N, 2) - world coordinates
            visibility: (B, N) - keypoint visibility (optional)

        Returns:
            Reprojection error
        """
        B, N, _ = pixel_points.shape

        # Convert to homogeneous coordinates
        pixel_homogeneous = torch.cat([
            pixel_points,
            torch.ones(B, N, 1, device=pixel_points.device)
        ], dim=-1)  # (B, N, 3)

        # Apply homography
        world_projected = torch.bmm(
            pixel_homogeneous,
            pred_homography.transpose(1, 2)
        )  # (B, N, 3)

        # Convert back to Cartesian
        world_projected = world_projected[:, :, :2] / (world_projected[:, :, 2:3] + 1e-8)

        # Compute error
        if self.loss_type == "l2":
            error = torch.norm(world_projected - world_points, dim=-1)  # (B, N)
        elif self.loss_type == "l1":
            error = torch.abs(world_projected - world_points).sum(dim=-1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Apply visibility mask
        if visibility is not None:
            error = error * visibility

        return error.mean()


class TrajectoryLoss(nn.Module):
    """
    Trajectory prediction loss with Gaussian NLL.

    Predicts both position and uncertainty.
    """

    def __init__(
        self,
        ade_weight: float = 1.0,
        fde_weight: float = 1.0,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.use_uncertainty = use_uncertainty

    def forward(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        pred_sigma: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_trajectory: (B, T, 2) - predicted positions
            target_trajectory: (B, T, 2) - ground truth positions
            pred_sigma: (B, T, 2) - predicted uncertainty (optional)

        Returns:
            Dictionary of losses
        """
        # Average Displacement Error (ADE)
        displacement = torch.norm(pred_trajectory - target_trajectory, dim=-1)  # (B, T)
        ade = displacement.mean()

        # Final Displacement Error (FDE)
        fde = displacement[:, -1].mean()

        total_loss = self.ade_weight * ade + self.fde_weight * fde

        # Gaussian NLL (if uncertainty is predicted)
        if self.use_uncertainty and pred_sigma is not None:
            # NLL = 0.5 * log(2π * σ²) + (y - μ)² / (2σ²)
            variance = pred_sigma ** 2 + 1e-8
            nll = 0.5 * torch.log(2 * np.pi * variance) + (pred_trajectory - target_trajectory) ** 2 / (2 * variance)
            nll = nll.mean()
            total_loss = total_loss + 0.1 * nll

        return {
            "total": total_loss,
            "ade": ade,
            "fde": fde,
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning (Baller2Vec, Re-ID).

    InfoNCE / NT-Xent loss.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) - normalized embeddings
            labels: (B,) - identity labels (optional, for supervised contrastive)

        Returns:
            Contrastive loss
        """
        B = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature  # (B, B)

        # Create positive/negative masks
        if labels is not None:
            # Supervised contrastive
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.t()).float().to(embeddings.device)
            mask.fill_diagonal_(0)  # Remove self-similarity
        else:
            # Self-supervised (assume augmented pairs)
            # Assume batch is organized as [original_1, augmented_1, original_2, augmented_2, ...]
            mask = torch.zeros(B, B, device=embeddings.device)
            for i in range(0, B, 2):
                if i + 1 < B:
                    mask[i, i + 1] = 1
                    mask[i + 1, i] = 1

        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        positive_sim = (exp_sim * mask).sum(dim=1)
        negative_sim = exp_sim.sum(dim=1) - torch.diag(exp_sim)

        loss = -torch.log(positive_sim / (negative_sim + 1e-8) + 1e-8)
        loss = loss.mean()

        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for re-identification.

    Encourages embeddings of same identity to be close, different identities to be far.
    """

    def __init__(self, margin: float = 0.3, distance: str = "euclidean"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor: (B, D) - anchor embeddings
            positive: (B, D) - positive embeddings (same identity)
            negative: (B, D) - negative embeddings (different identity)

        Returns:
            Triplet loss
        """
        if self.distance == "euclidean":
            pos_dist = torch.norm(anchor - positive, dim=1)
            neg_dist = torch.norm(anchor - negative, dim=1)
        elif self.distance == "cosine":
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class CenterLoss(nn.Module):
    """
    Center loss for re-identification.

    Learns a center embedding for each identity.
    """

    def __init__(self, num_classes: int, feature_dim: int, lambda_center: float = 0.0005):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_center = lambda_center

        # Centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) - feature embeddings
            labels: (B,) - identity labels

        Returns:
            Center loss
        """
        batch_size = features.size(0)

        # Distance to centers
        centers_batch = self.centers[labels]  # (B, D)
        loss = torch.norm(features - centers_batch, dim=1).mean()

        return self.lambda_center * loss


class HungarianMatchingLoss(nn.Module):
    """
    Hungarian matching loss for DETR.

    Performs bipartite matching between predictions and ground truth.
    """

    def __init__(
        self,
        num_classes: int,
        class_weight: float = 2.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_logits: (B, N, C) - class predictions
            pred_boxes: (B, N, 4) - bbox predictions (normalized)
            target_labels: List of (M,) tensors - ground truth labels
            target_boxes: List of (M, 4) tensors - ground truth boxes

        Returns:
            Dictionary of losses
        """
        batch_size = pred_logits.shape[0]

        total_class_loss = 0
        total_bbox_loss = 0
        total_giou_loss = 0

        for i in range(batch_size):
            # Get predictions and targets for this sample
            pred_logits_i = pred_logits[i]  # (N, C)
            pred_boxes_i = pred_boxes[i]  # (N, 4)
            target_labels_i = target_labels[i] if isinstance(target_labels, list) else target_labels[i]
            target_boxes_i = target_boxes[i] if isinstance(target_boxes, list) else target_boxes[i]

            num_targets = len(target_labels_i)
            if num_targets == 0:
                # No targets, only background loss
                total_class_loss += F.cross_entropy(
                    pred_logits_i,
                    torch.full((pred_logits_i.shape[0],), self.num_classes, dtype=torch.long, device=pred_logits_i.device)
                )
                continue

            # Compute cost matrix
            cost_class = -pred_logits_i[:, target_labels_i]  # (N, M)
            cost_bbox = torch.cdist(pred_boxes_i, target_boxes_i, p=1)  # (N, M)
            cost_giou = -self._generalized_box_iou(pred_boxes_i, target_boxes_i)  # (N, M)

            cost_matrix = (
                self.class_weight * cost_class +
                self.bbox_weight * cost_bbox +
                self.giou_weight * cost_giou
            )

            # Hungarian matching
            cost_matrix_np = cost_matrix.detach().cpu().numpy()
            pred_indices, target_indices = linear_sum_assignment(cost_matrix_np)

            # Compute losses for matched pairs
            matched_pred_logits = pred_logits_i[pred_indices]
            matched_pred_boxes = pred_boxes_i[pred_indices]
            matched_target_labels = target_labels_i[target_indices]
            matched_target_boxes = target_boxes_i[target_indices]

            # Classification loss
            class_loss = F.cross_entropy(matched_pred_logits, matched_target_labels)

            # Bbox loss (L1)
            bbox_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes)

            # GIoU loss
            giou = self._generalized_box_iou(matched_pred_boxes, matched_target_boxes)
            giou_loss = (1 - torch.diag(giou)).mean()

            total_class_loss += class_loss
            total_bbox_loss += bbox_loss
            total_giou_loss += giou_loss

        return {
            "total": (total_class_loss + total_bbox_loss + total_giou_loss) / batch_size,
            "class": total_class_loss / batch_size,
            "bbox": total_bbox_loss / batch_size,
            "giou": total_giou_loss / batch_size,
        }

    def _generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute Generalized IoU.

        Args:
            boxes1: (N, 4) - [x1, y1, x2, y2]
            boxes2: (M, 4) - [x1, y1, x2, y2]

        Returns:
            GIoU matrix (N, M)
        """
        # Compute IoU
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
        union = area1[:, None] + area2 - inter  # (N, M)

        iou = inter / (union + 1e-8)

        # Compute smallest enclosing box
        lt_enc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb_enc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        wh_enc = (rb_enc - lt_enc).clamp(min=0)
        area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]

        # GIoU
        giou = iou - (area_enc - union) / (area_enc + 1e-8)

        return giou


class PhysicsConstraintLoss(nn.Module):
    """
    Physics-based constraint loss for 3D ball tracking.

    Ensures predictions follow physical laws (gravity, parabolic motion).
    """

    def __init__(
        self,
        gravity: float = 9.81,
        weight: float = 0.3,
    ):
        super().__init__()
        self.gravity = gravity
        self.weight = weight

    def forward(
        self,
        pred_trajectory: torch.Tensor,
        dt: float = 0.04,  # 1/25 fps
    ) -> torch.Tensor:
        """
        Args:
            pred_trajectory: (B, T, 3) - predicted 3D positions (x, y, z)
            dt: Time step between frames

        Returns:
            Physics violation loss
        """
        # Compute velocities
        velocities = (pred_trajectory[:, 1:] - pred_trajectory[:, :-1]) / dt  # (B, T-1, 3)

        # Compute accelerations
        accelerations = (velocities[:, 1:] - velocities[:, :-1]) / dt  # (B, T-2, 3)

        # Vertical acceleration should be approximately -g
        expected_az = -self.gravity
        az_error = torch.abs(accelerations[:, :, 2] - expected_az).mean()

        # Horizontal acceleration should be small (neglecting air resistance)
        horizontal_accel = torch.norm(accelerations[:, :, :2], dim=-1).mean()

        loss = self.weight * (az_error + 0.1 * horizontal_accel)

        return loss


class TemporalSmoothnessLoss(nn.Module):
    """
    Temporal smoothness loss for trajectory prediction.

    Penalizes sudden changes in velocity/acceleration.
    """

    def __init__(self, weight: float = 0.2):
        super().__init__()
        self.weight = weight

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: (B, T, D) - predicted trajectory

        Returns:
            Smoothness loss
        """
        # First derivative (velocity)
        velocity = trajectory[:, 1:] - trajectory[:, :-1]

        # Second derivative (acceleration)
        acceleration = velocity[:, 1:] - velocity[:, :-1]

        # Penalize large accelerations
        loss = self.weight * torch.norm(acceleration, dim=-1).mean()

        return loss
