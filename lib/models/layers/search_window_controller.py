"""Lightweight causal controller for the next tracking search window."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SearchWindowOutput:
    next_box: torch.Tensor
    inside_logit: torch.Tensor
    quality_logit: torch.Tensor
    consensus_box: torch.Tensor
    expert_weights: torch.Tensor


def _box_center(boxes):
    return boxes[..., :2] + 0.5 * boxes[..., 2:]


def _search_crop_box(anchor_boxes, search_factor):
    side = (
        anchor_boxes[..., 2:].clamp_min(1e-6).prod(dim=-1).sqrt()
        * float(search_factor)
    ).clamp_min(1e-4)
    center = _box_center(anchor_boxes)
    return torch.cat((center - 0.5 * side[..., None], side[..., None].repeat_interleave(2, dim=-1)), dim=-1)


def dynamic_search_crop(images, anchor_boxes, search_factor, output_size):
    """Crop normalized square canvases using normalized image-space xywh anchors."""
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    anchor_boxes = torch.as_tensor(
        anchor_boxes, device=images.device, dtype=images.dtype)
    if anchor_boxes.shape != (images.shape[0], 4):
        raise ValueError("anchor_boxes must have shape [batch, 4]")
    crop_boxes = _search_crop_box(anchor_boxes, search_factor)
    center = _box_center(crop_boxes)
    side = crop_boxes[:, 2]
    theta = images.new_zeros((images.shape[0], 2, 3))
    theta[:, 0, 0] = side
    theta[:, 1, 1] = side
    theta[:, 0, 2] = 2.0 * center[:, 0] - 1.0
    theta[:, 1, 2] = 2.0 * center[:, 1] - 1.0
    grid = F.affine_grid(
        theta,
        torch.Size((images.shape[0], images.shape[1], output_size, output_size)),
        align_corners=False,
    )
    crops = F.grid_sample(
        images, grid, mode="bilinear", padding_mode="zeros",
        align_corners=False)
    return crops, crop_boxes


def crop_box_to_image(crop_boxes_cxcywh, crop_regions_xywh):
    """Map normalized crop-level cxcywh boxes to normalized image-level xywh."""
    boxes = torch.as_tensor(crop_boxes_cxcywh)
    regions = torch.as_tensor(
        crop_regions_xywh, device=boxes.device, dtype=boxes.dtype)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("crop boxes must have shape [batch, count, 4]")
    if regions.shape != (boxes.shape[0], 4):
        raise ValueError("crop regions must have shape [batch, 4]")
    region_xy = regions[:, None, :2]
    region_wh = regions[:, None, 2:]
    center = region_xy + boxes[..., :2] * region_wh
    size = boxes[..., 2:] * region_wh
    return torch.cat((center - 0.5 * size, size), dim=-1)


def crop_target_inside(target_boxes, anchor_boxes, search_factor):
    target_boxes = torch.as_tensor(target_boxes)
    anchor_boxes = torch.as_tensor(
        anchor_boxes, device=target_boxes.device, dtype=target_boxes.dtype)
    crop_boxes = _search_crop_box(anchor_boxes, search_factor)
    target_max = target_boxes[..., :2] + target_boxes[..., 2:]
    crop_max = crop_boxes[..., :2] + crop_boxes[..., 2:]
    return (
        (target_boxes[..., :2] >= crop_boxes[..., :2])
        & (target_max <= crop_max)
    ).all(dim=-1)


def event_motion_centroid(event_images, eps=1e-6):
    """Return normalized event-energy center and a robust non-empty confidence."""
    if event_images.ndim != 4:
        raise ValueError(
            "event_images must have shape [batch, channels, height, width]")
    values = event_images.float()
    baseline = values.flatten(2).median(dim=-1).values[..., None, None]
    energy = (values - baseline).abs().mean(dim=1)
    batch, height, width = energy.shape
    total = energy.flatten(1).sum(dim=1)
    x = (torch.arange(width, device=energy.device, dtype=energy.dtype) + 0.5) / width
    y = (torch.arange(height, device=energy.device, dtype=energy.dtype) + 0.5) / height
    center_x = (energy.sum(dim=1) * x[None]).sum(dim=1) / total.clamp_min(eps)
    center_y = (energy.sum(dim=2) * y[None]).sum(dim=1) / total.clamp_min(eps)
    non_empty = total > eps
    center = torch.stack((center_x, center_y), dim=-1)
    center = torch.where(non_empty[:, None], center, center.new_full((batch, 2), 0.5))
    peak = energy.flatten(1).max(dim=1).values
    confidence = torch.where(
        non_empty, (peak / total.clamp_min(eps)).sqrt().clamp(max=1.0),
        total.new_zeros(batch))
    return center, confidence[:, None]


class SearchWindowController(nn.Module):
    """Predict the next crop anchor without changing the reported target box."""

    def __init__(self, expert_count=5, hidden_dim=64, max_center_step=1.0):
        super().__init__()
        self.expert_count = int(expert_count)
        self.max_center_step = float(max_center_step)
        input_dim = 4 + 4 + self.expert_count * 4 + self.expert_count * 2 + 1 + 2 + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _consensus(self, expert_boxes, response_peaks, response_psr):
        reliability = (
            response_peaks.float().clamp(0.0, 1.0)
            * response_psr.float().clamp(0.0, 1.0)
        ).clamp_min(1e-6)
        weights = reliability / reliability.sum(dim=1, keepdim=True)
        consensus = (weights[..., None] * expert_boxes).sum(dim=1)
        return consensus, weights

    def forward(self, *, current_box, previous_box, expert_boxes,
                response_peaks, response_psr, presence, event_center,
                event_confidence):
        current_box = torch.as_tensor(current_box)
        previous_box = torch.as_tensor(
            previous_box, device=current_box.device, dtype=current_box.dtype)
        expert_boxes = torch.as_tensor(
            expert_boxes, device=current_box.device, dtype=current_box.dtype)
        response_peaks = torch.as_tensor(
            response_peaks, device=current_box.device, dtype=current_box.dtype)
        response_psr = torch.as_tensor(
            response_psr, device=current_box.device, dtype=current_box.dtype)
        presence = torch.as_tensor(
            presence, device=current_box.device, dtype=current_box.dtype).reshape(-1, 1)
        event_center = torch.as_tensor(
            event_center, device=current_box.device, dtype=current_box.dtype)
        event_confidence = torch.as_tensor(
            event_confidence, device=current_box.device,
            dtype=current_box.dtype).reshape(-1, 1)
        batch = current_box.shape[0]
        if current_box.shape != (batch, 4) or previous_box.shape != (batch, 4):
            raise ValueError("current_box and previous_box must have shape [batch, 4]")
        if expert_boxes.shape != (batch, self.expert_count, 4):
            raise ValueError("expert_boxes have an invalid shape")
        if response_peaks.shape != (batch, self.expert_count) \
                or response_psr.shape != (batch, self.expert_count):
            raise ValueError("expert response statistics have an invalid shape")
        consensus, expert_weights = self._consensus(
            expert_boxes, response_peaks, response_psr)
        reference_size = current_box[:, 2:].clamp_min(1e-4)
        velocity = torch.cat((
            (_box_center(current_box) - _box_center(previous_box)) / reference_size,
            torch.log(current_box[:, 2:].clamp_min(1e-4)
                      / previous_box[:, 2:].clamp_min(1e-4)),
        ), dim=-1)
        expert_center = _box_center(expert_boxes)
        current_center = _box_center(current_box)[:, None]
        relative_experts = torch.cat((
            (expert_center - current_center) / reference_size[:, None],
            torch.log(expert_boxes[..., 2:].clamp_min(1e-4)
                      / reference_size[:, None]),
        ), dim=-1)
        event_relative = (
            event_center - _box_center(current_box)) / reference_size
        features = torch.cat((
            current_box,
            velocity,
            relative_experts.flatten(1),
            response_peaks,
            response_psr,
            presence,
            event_relative,
            event_confidence,
        ), dim=-1)
        raw = self.mlp(features)
        center_delta = torch.tanh(raw[:, :2]) * self.max_center_step
        size = consensus[:, 2:] * torch.exp(raw[:, 2:4].clamp(-2.0, 2.0))
        center = _box_center(consensus) + center_delta
        next_box = torch.cat((center - 0.5 * size, size), dim=-1)
        return SearchWindowOutput(
            next_box=next_box,
            inside_logit=raw[:, 4],
            quality_logit=raw[:, 5],
            consensus_box=consensus,
            expert_weights=expert_weights,
        )


def search_window_pursuit_loss(
        predictions, *, target_next, current_inside, current_quality,
        present_next, search_factor, center_weight=1.0, scale_weight=0.5,
        containment_weight=2.0, inside_weight=0.5, quality_weight=0.25):
    target_next = torch.as_tensor(
        target_next, device=predictions.next_box.device,
        dtype=predictions.next_box.dtype)
    current_inside = torch.as_tensor(
        current_inside, device=predictions.next_box.device,
        dtype=torch.bool).reshape(-1)
    current_quality = torch.as_tensor(
        current_quality, device=predictions.next_box.device,
        dtype=predictions.next_box.dtype).reshape(-1)
    present_next = torch.as_tensor(
        present_next, device=predictions.next_box.device,
        dtype=torch.bool).reshape(-1)
    predicted = predictions.next_box
    predicted_center = _box_center(predicted)
    target_center = _box_center(target_next)
    present_weight = present_next.to(predicted.dtype)
    present_count = present_weight.sum().clamp_min(1.0)
    center = (
        F.smooth_l1_loss(predicted_center, target_center, reduction="none")
        .mean(dim=-1) * present_weight
    ).sum() / present_count
    scale = (
        F.smooth_l1_loss(
            torch.log(predicted[:, 2:].clamp_min(1e-4)),
            torch.log(target_next[:, 2:].clamp_min(1e-4)),
            reduction="none",
        ).mean(dim=-1) * present_weight
    ).sum() / present_count
    crop = _search_crop_box(predicted, search_factor)
    crop_center = _box_center(crop)
    crop_half = 0.5 * crop[:, 2:]
    target_half = 0.5 * target_next[:, 2:]
    overflow = F.relu(
        (target_center - crop_center).abs() + target_half - crop_half)
    containment = (overflow.mean(dim=-1) * present_weight).sum() / present_count
    inside = F.binary_cross_entropy_with_logits(
        predictions.inside_logit.float(), current_inside.float())
    if current_inside.any():
        quality = F.smooth_l1_loss(
            predictions.quality_logit[current_inside].sigmoid(),
            current_quality[current_inside].clamp(0.0, 1.0))
    else:
        quality = predictions.quality_logit.sum() * 0.0
    loss = (
        center_weight * center
        + scale_weight * scale
        + containment_weight * containment
        + inside_weight * inside
        + quality_weight * quality
    )
    return loss, {
        "Loss/pursuit_center": float(center.detach()),
        "Loss/pursuit_scale": float(scale.detach()),
        "Loss/pursuit_containment": float(containment.detach()),
        "Loss/pursuit_inside": float(inside.detach()),
        "Loss/pursuit_quality": float(quality.detach()),
        "Pursuit/quality_count": int(current_inside.sum().item()),
        "Pursuit/outside_count": int((~current_inside).sum().item()),
    }
