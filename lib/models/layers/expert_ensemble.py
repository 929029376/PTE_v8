"""Deterministic reliability ensemble for independent tracking experts."""

from dataclasses import dataclass
from itertools import combinations

import torch


@dataclass(frozen=True)
class ExpertEnsembleResult:
    box: torch.Tensor
    score: float
    weights: torch.Tensor
    quality: torch.Tensor
    consensus: torch.Tensor
    temporal: torch.Tensor
    retained_ids: tuple


def normalized_response_psr(response_maps, eps=1e-6):
    """Return sigmoid-normalized peak-to-sidelobe ratios in [0, 1]."""
    maps = torch.as_tensor(response_maps).float()
    if maps.ndim < 2:
        raise ValueError("response_maps must have shape (E,...)")
    flat = maps.reshape(maps.shape[0], -1)
    if flat.shape[1] == 0:
        raise ValueError("response_maps cannot be empty")
    if flat.shape[1] == 1:
        return torch.full(
            (flat.shape[0],), 0.5, device=flat.device, dtype=flat.dtype)
    peak, peak_id = flat.max(dim=1)
    keep = torch.ones_like(flat, dtype=torch.bool)
    keep.scatter_(1, peak_id[:, None], False)
    sidelobes = flat[keep].reshape(flat.shape[0], flat.shape[1] - 1)
    mean = sidelobes.mean(dim=1)
    std = sidelobes.std(dim=1, unbiased=False)
    return torch.sigmoid((peak - mean) / std.clamp_min(eps))


def _pairwise_iou_xywh(boxes, eps=1e-6):
    top_left = boxes[:, :2]
    bottom_right = top_left + boxes[:, 2:].clamp_min(0.0)
    intersection_left = torch.maximum(top_left[:, None], top_left[None])
    intersection_right = torch.minimum(
        bottom_right[:, None], bottom_right[None])
    intersection_wh = (intersection_right - intersection_left).clamp_min(0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area = boxes[:, 2].clamp_min(0.0) * boxes[:, 3].clamp_min(0.0)
    union = area[:, None] + area[None] - intersection
    return intersection / union.clamp_min(eps)


def _has_pairwise_cluster(pairwise_iou, threshold, minimum_size=3):
    count = pairwise_iou.shape[0]
    for size in range(count, minimum_size - 1, -1):
        for ids in combinations(range(count), size):
            submatrix = pairwise_iou[list(ids)][:, list(ids)]
            off_diagonal = ~torch.eye(
                size, dtype=torch.bool, device=pairwise_iou.device)
            if bool((submatrix[off_diagonal] >= threshold).all()):
                return True
    return False


def _temporal_consistency(boxes, last_box, eps=1e-6):
    if last_box is None:
        return torch.ones(
            boxes.shape[0], device=boxes.device, dtype=boxes.dtype)
    previous = torch.as_tensor(
        last_box, device=boxes.device, dtype=boxes.dtype).reshape(-1)
    if previous.shape != (4,):
        raise ValueError("last_box must have shape (4,)")
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:]
    previous_center = previous[:2] + 0.5 * previous[2:]
    previous_diagonal = torch.linalg.vector_norm(
        previous[2:].clamp_min(eps)).clamp_min(eps)
    center_change = torch.linalg.vector_norm(
        centers - previous_center, dim=1) / previous_diagonal
    scale_change = torch.log(
        boxes[:, 2:].clamp_min(eps)
        / previous[2:].clamp_min(eps)
    ).abs().mean(dim=1)
    return torch.exp(-(center_change + scale_change)).clamp(0.0, 1.0)


def fuse_expert_predictions(
        boxes, response_peaks, response_psr, last_box,
        cluster_iou=0.50, reject_consensus=0.15):
    """Fuse expert image-coordinate ``xywh`` boxes without learned routing."""
    boxes = torch.as_tensor(boxes).float()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (E,4)")
    expert_count = boxes.shape[0]
    peaks = torch.as_tensor(
        response_peaks, device=boxes.device, dtype=boxes.dtype).reshape(-1)
    psr = torch.as_tensor(
        response_psr, device=boxes.device, dtype=boxes.dtype).reshape(-1)
    if peaks.shape != (expert_count,) or psr.shape != (expert_count,):
        raise ValueError("response_peaks and response_psr must have shape (E,)")

    pairwise_iou = _pairwise_iou_xywh(boxes)
    consensus = pairwise_iou.median(dim=1).values.clamp(0.0, 1.0)
    temporal = _temporal_consistency(boxes, last_box)
    peaks = peaks.clamp(0.0, 1.0)
    psr = psr.clamp(0.0, 1.0)
    quality = (
        peaks.pow(0.35)
        * psr.pow(0.25)
        * consensus.pow(0.25)
        * temporal.pow(0.15)
    )

    retained = torch.ones(
        expert_count, dtype=torch.bool, device=boxes.device)
    if _has_pairwise_cluster(pairwise_iou, float(cluster_iou)):
        retained &= consensus >= float(reject_consensus)
    retained_ids = tuple(retained.nonzero(as_tuple=False).flatten().tolist())
    retained_quality = quality * retained.to(quality.dtype)
    quality_sum = retained_quality.sum()
    if not retained_ids or not bool(torch.isfinite(quality_sum)) \
            or float(quality_sum.item()) <= 0.0:
        best_id = int(peaks.argmax().item())
        weights = torch.zeros_like(quality)
        weights[best_id] = 1.0
        return ExpertEnsembleResult(
            box=boxes[best_id],
            score=float(peaks[best_id].item()),
            weights=weights,
            quality=quality,
            consensus=consensus,
            temporal=temporal,
            retained_ids=(best_id,),
        )

    weights = retained_quality / quality_sum
    return ExpertEnsembleResult(
        box=(weights[:, None] * boxes).sum(dim=0),
        score=float((weights * peaks).sum().item()),
        weights=weights,
        quality=quality,
        consensus=consensus,
        temporal=temporal,
        retained_ids=retained_ids,
    )
