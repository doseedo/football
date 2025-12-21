"""
Evaluation Metrics for Football Tracking Models

Includes metrics for:
- Homography (reprojection error, PCK)
- Trajectory prediction (ADE, FDE)
- Re-identification (mAP, CMC, Rank-k)
- Detection (mAP, precision, recall)
- Tactical analysis (formation accuracy)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import average_precision_score, precision_recall_curve


class ReprojectionError(nn.Module):
    """
    Reprojection error for homography evaluation.

    Measures average distance between projected and ground truth world coordinates.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.errors = []

    def update(
        self,
        pred_homography: torch.Tensor,
        pixel_points: torch.Tensor,
        world_points: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            pred_homography: (B, 3, 3)
            pixel_points: (B, N, 2)
            world_points: (B, N, 2)
            visibility: (B, N) - optional visibility mask
        """
        B, N, _ = pixel_points.shape

        # Convert to homogeneous
        pixel_homogeneous = torch.cat([
            pixel_points,
            torch.ones(B, N, 1, device=pixel_points.device)
        ], dim=-1)

        # Apply homography
        world_projected = torch.bmm(
            pixel_homogeneous,
            pred_homography.transpose(1, 2)
        )
        world_projected = world_projected[:, :, :2] / (world_projected[:, :, 2:3] + 1e-8)

        # Compute error (in meters)
        error = torch.norm(world_projected - world_points, dim=-1)  # (B, N)

        if visibility is not None:
            error = error * visibility

        self.errors.extend(error.flatten().detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Compute mean and median reprojection error."""
        errors = np.array(self.errors)
        return {
            "mean_reproj_error": float(np.mean(errors)),
            "median_reproj_error": float(np.median(errors)),
            "max_reproj_error": float(np.max(errors)),
        }


class KeypointPCK(nn.Module):
    """
    Percentage of Correct Keypoints (PCK).

    Measures how many keypoints are within a threshold distance.
    """

    def __init__(self, threshold: float = 5.0):
        """
        Args:
            threshold: Distance threshold in pixels (or meters for world coords)
        """
        super().__init__()
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(
        self,
        pred_keypoints: torch.Tensor,
        target_keypoints: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            pred_keypoints: (B, N, 2)
            target_keypoints: (B, N, 2)
            visibility: (B, N) - optional
        """
        # Compute distances
        distances = torch.norm(pred_keypoints - target_keypoints, dim=-1)  # (B, N)

        # Check which are correct
        correct_mask = distances < self.threshold

        if visibility is not None:
            correct_mask = correct_mask & (visibility > 0.5)
            total_mask = visibility > 0.5
        else:
            total_mask = torch.ones_like(distances, dtype=torch.bool)

        self.correct += correct_mask.sum().item()
        self.total += total_mask.sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute PCK."""
        if self.total == 0:
            return {"pck": 0.0}
        return {"pck": self.correct / self.total}


class TrajectoryADE(nn.Module):
    """
    Average Displacement Error for trajectory prediction.

    Mean distance between predicted and ground truth positions.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.errors = []

    def update(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
    ):
        """
        Args:
            pred_trajectory: (B, T, 2)
            target_trajectory: (B, T, 2)
        """
        # Compute displacement at each timestep
        displacement = torch.norm(pred_trajectory - target_trajectory, dim=-1)  # (B, T)

        # Average over time
        ade = displacement.mean(dim=1)  # (B,)

        self.errors.extend(ade.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Compute mean ADE."""
        return {"ade": float(np.mean(self.errors))}


class TrajectoryFDE(nn.Module):
    """
    Final Displacement Error for trajectory prediction.

    Distance between final predicted and ground truth positions.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.errors = []

    def update(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
    ):
        """
        Args:
            pred_trajectory: (B, T, 2)
            target_trajectory: (B, T, 2)
        """
        # Final displacement
        fde = torch.norm(pred_trajectory[:, -1] - target_trajectory[:, -1], dim=-1)

        self.errors.extend(fde.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Compute mean FDE."""
        return {"fde": float(np.mean(self.errors))}


class ReIDMetrics(nn.Module):
    """
    Re-identification metrics: mAP, CMC, Rank-1/5/10.

    Used for evaluating player re-identification models.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.embeddings = []
        self.labels = []
        self.camera_ids = []

    def update(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        camera_ids: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            embeddings: (B, D) - feature embeddings
            labels: (B,) - identity labels
            camera_ids: (B,) - camera IDs (optional)
        """
        self.embeddings.append(embeddings.detach().cpu())
        self.labels.append(labels.detach().cpu())
        if camera_ids is not None:
            self.camera_ids.append(camera_ids.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute mAP and CMC."""
        # Concatenate all embeddings
        embeddings = torch.cat(self.embeddings, dim=0).numpy()
        labels = torch.cat(self.labels, dim=0).numpy()

        # Compute distance matrix
        num_samples = len(embeddings)
        dist_matrix = np.zeros((num_samples, num_samples))

        for i in range(num_samples):
            for j in range(num_samples):
                dist_matrix[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])

        # Compute mAP
        ap_scores = []
        cmc = np.zeros(num_samples)

        for i in range(num_samples):
            # Query: i-th sample
            # Gallery: all other samples
            query_label = labels[i]
            gallery_dist = dist_matrix[i]

            # Remove query from gallery
            gallery_mask = np.ones(num_samples, dtype=bool)
            gallery_mask[i] = False

            gallery_dist = gallery_dist[gallery_mask]
            gallery_labels = labels[gallery_mask]

            # Sort by distance
            sort_idx = np.argsort(gallery_dist)
            sorted_labels = gallery_labels[sort_idx]

            # Find matches
            matches = sorted_labels == query_label

            if matches.sum() == 0:
                continue

            # AP
            ap = average_precision_score(matches.astype(int), -gallery_dist[sort_idx])
            ap_scores.append(ap)

            # CMC
            match_idx = np.where(matches)[0]
            if len(match_idx) > 0:
                first_match = match_idx[0]
                cmc[first_match:] += 1

        mAP = np.mean(ap_scores) if len(ap_scores) > 0 else 0.0
        cmc = cmc / num_samples

        return {
            "mAP": float(mAP),
            "rank1": float(cmc[0]) if len(cmc) > 0 else 0.0,
            "rank5": float(cmc[4]) if len(cmc) > 4 else 0.0,
            "rank10": float(cmc[9]) if len(cmc) > 9 else 0.0,
        }


class DetectionMetrics(nn.Module):
    """
    Object detection metrics: mAP, AP50, AP75, precision, recall.

    Used for DETR and other detection models.
    """

    def __init__(self, num_classes: int, iou_thresholds: List[float] = [0.5, 0.75]):
        super().__init__()
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_labels: torch.Tensor,
        pred_scores: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
    ):
        """
        Args:
            pred_boxes: (N, 4) - predicted boxes [x1, y1, x2, y2]
            pred_labels: (N,) - predicted class labels
            pred_scores: (N,) - prediction confidence scores
            target_boxes: (M, 4) - ground truth boxes
            target_labels: (M,) - ground truth labels
        """
        self.predictions.append({
            'boxes': pred_boxes.detach().cpu().numpy(),
            'labels': pred_labels.detach().cpu().numpy(),
            'scores': pred_scores.detach().cpu().numpy(),
        })
        self.targets.append({
            'boxes': target_boxes.detach().cpu().numpy(),
            'labels': target_labels.detach().cpu().numpy(),
        })

    def compute(self) -> Dict[str, float]:
        """Compute detection metrics."""
        # Simplified mAP computation
        # For production, use torchmetrics or pycocotools

        metrics = {}

        for iou_thresh in self.iou_thresholds:
            ap_per_class = []

            for class_id in range(self.num_classes):
                # Collect all predictions and targets for this class
                all_scores = []
                all_matches = []

                for pred, target in zip(self.predictions, self.targets):
                    # Filter by class
                    pred_mask = pred['labels'] == class_id
                    target_mask = target['labels'] == class_id

                    pred_boxes_cls = pred['boxes'][pred_mask]
                    pred_scores_cls = pred['scores'][pred_mask]
                    target_boxes_cls = target['boxes'][target_mask]

                    if len(pred_boxes_cls) == 0:
                        continue

                    # Compute IoU
                    ious = self._compute_iou(pred_boxes_cls, target_boxes_cls)

                    # Match predictions to targets
                    matched = np.zeros(len(pred_boxes_cls), dtype=bool)
                    if len(target_boxes_cls) > 0:
                        max_ious = ious.max(axis=1)
                        matched = max_ious >= iou_thresh

                    all_scores.extend(pred_scores_cls)
                    all_matches.extend(matched)

                if len(all_scores) > 0:
                    # Compute AP
                    all_scores = np.array(all_scores)
                    all_matches = np.array(all_matches)

                    sort_idx = np.argsort(-all_scores)
                    all_matches = all_matches[sort_idx]

                    # Precision-recall curve
                    tp = np.cumsum(all_matches)
                    fp = np.cumsum(~all_matches)
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (all_matches.sum() + 1e-8)

                    # AP (area under PR curve)
                    ap = np.trapz(precision, recall)
                    ap_per_class.append(ap)

            if len(ap_per_class) > 0:
                metrics[f"AP{int(iou_thresh*100)}"] = float(np.mean(ap_per_class))

        metrics["mAP"] = float(np.mean([v for v in metrics.values()]))

        return metrics

    def _compute_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes."""
        if len(boxes2) == 0:
            return np.zeros((len(boxes1), 0))

        # Compute intersection
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]

        # Compute union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - inter

        # IoU
        iou = inter / (union + 1e-8)

        return iou


class FormationAccuracy(nn.Module):
    """
    Tactical formation classification accuracy.

    Used for GNN evaluation.
    """

    def __init__(self, num_formations: int):
        super().__init__()
        self.num_formations = num_formations
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(
        self,
        pred_formations: torch.Tensor,
        target_formations: torch.Tensor,
    ):
        """
        Args:
            pred_formations: (B,) - predicted formation IDs
            target_formations: (B,) - ground truth formation IDs
        """
        self.predictions.extend(pred_formations.detach().cpu().numpy())
        self.targets.extend(target_formations.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Compute accuracy and per-class metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        # Overall accuracy
        accuracy = (predictions == targets).mean()

        # Per-formation accuracy
        per_formation_acc = []
        for formation_id in range(self.num_formations):
            mask = targets == formation_id
            if mask.sum() > 0:
                acc = (predictions[mask] == formation_id).mean()
                per_formation_acc.append(acc)

        return {
            "accuracy": float(accuracy),
            "mean_per_class_acc": float(np.mean(per_formation_acc)) if len(per_formation_acc) > 0 else 0.0,
        }


class MultiHorizonTrajectoryMetrics(nn.Module):
    """
    Trajectory metrics at multiple prediction horizons.

    Useful for Baller2Vec and Ball3D evaluation.
    """

    def __init__(self, horizons: List[int] = [1, 5, 10, 25, 50]):
        """
        Args:
            horizons: List of frame indices to evaluate at
        """
        super().__init__()
        self.horizons = horizons
        self.reset()

    def reset(self):
        self.ade_per_horizon = {h: [] for h in self.horizons}
        self.fde_per_horizon = {h: [] for h in self.horizons}

    def update(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
    ):
        """
        Args:
            pred_trajectory: (B, T, 2)
            target_trajectory: (B, T, 2)
        """
        T = pred_trajectory.shape[1]

        for horizon in self.horizons:
            if horizon > T:
                continue

            # ADE up to horizon
            displacement = torch.norm(
                pred_trajectory[:, :horizon] - target_trajectory[:, :horizon],
                dim=-1
            )
            ade = displacement.mean(dim=1)
            self.ade_per_horizon[horizon].extend(ade.detach().cpu().numpy())

            # FDE at horizon
            fde = torch.norm(
                pred_trajectory[:, horizon-1] - target_trajectory[:, horizon-1],
                dim=-1
            )
            self.fde_per_horizon[horizon].extend(fde.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Compute metrics at each horizon."""
        metrics = {}

        for horizon in self.horizons:
            if len(self.ade_per_horizon[horizon]) > 0:
                metrics[f"ade_{horizon}"] = float(np.mean(self.ade_per_horizon[horizon]))
                metrics[f"fde_{horizon}"] = float(np.mean(self.fde_per_horizon[horizon]))

        return metrics
