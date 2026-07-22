import math

import torch
import torch.nn.functional as F
from torch import nn


def extract_centered_candidate_crops(
        frames, centers, valid, target_boxes, search_factor, output_size=None):
    """Crop proposal-centered square searches using normalized coordinates."""
    if frames.ndim != 4 or not torch.is_floating_point(frames):
        raise ValueError("frames must have floating shape (B,C,H,W)")
    batch, channels, height, width = frames.shape
    if centers.ndim != 3 or centers.shape[0] != batch or centers.shape[-1] != 2:
        raise ValueError("centers must have shape (B,K,2)")
    candidate_count = centers.shape[1]
    valid = torch.as_tensor(valid, device=frames.device, dtype=torch.bool)
    if valid.shape != (batch, candidate_count):
        raise ValueError("valid must have shape (B,K)")
    boxes = torch.as_tensor(
        target_boxes, device=frames.device, dtype=frames.dtype)
    if boxes.shape != (batch, 4):
        raise ValueError("target_boxes must have shape (B,4)")
    factor = float(search_factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("search_factor must be finite and positive")
    size = int(output_size if output_size is not None else max(height, width))
    if size < 1:
        raise ValueError("output_size must be positive")
    if not torch.isfinite(frames).all() or not torch.isfinite(centers).all():
        raise ValueError("frames and centers must be finite")
    if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError("target boxes must be finite with positive size")

    target_width = boxes[:, 2].clamp_min(1.0 / max(width, 1)) * width
    target_height = boxes[:, 3].clamp_min(1.0 / max(height, 1)) * height
    crop_size = (target_width * target_height).sqrt() * factor
    scale_x = (crop_size / max(width, 1))[:, None].expand(
        batch, candidate_count).reshape(-1)
    scale_y = (crop_size / max(height, 1))[:, None].expand(
        batch, candidate_count).reshape(-1)
    flat_centers = centers.to(device=frames.device, dtype=frames.dtype).reshape(-1, 2)
    theta = frames.new_zeros((batch * candidate_count, 2, 3))
    theta[:, 0, 0] = scale_x
    theta[:, 1, 1] = scale_y
    theta[:, 0, 2] = 2.0 * flat_centers[:, 0] - 1.0
    theta[:, 1, 2] = 2.0 * flat_centers[:, 1] - 1.0
    repeated = frames[:, None].expand(
        batch, candidate_count, channels, height, width).reshape(
            batch * candidate_count, channels, height, width)
    grid = F.affine_grid(
        theta,
        (batch * candidate_count, channels, size, size),
        align_corners=False,
    )
    crops = F.grid_sample(
        repeated,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(batch, candidate_count, channels, size, size)
    return crops * valid[:, :, None, None, None].to(crops.dtype)


class RGBIdentityVerifier(nn.Module):
    """Verify Event-proposed candidates against a clean RGB template."""

    def __init__(self, input_dim, projection_dim=128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(projection_dim), bias=False),
        )

    def forward(self, template_rgb_tokens, candidate_rgb_tokens):
        if (template_rgb_tokens.ndim != 3
                or template_rgb_tokens.shape[-1] != self.input_dim):
            raise ValueError(
                "template_rgb_tokens must have shape (B, N, C)")
        if candidate_rgb_tokens.ndim == 3:
            candidate_rgb_tokens = candidate_rgb_tokens.unsqueeze(1)
        if (candidate_rgb_tokens.ndim != 4
                or candidate_rgb_tokens.shape[0] != template_rgb_tokens.shape[0]
                or candidate_rgb_tokens.shape[-1] != self.input_dim):
            raise ValueError(
                "candidate_rgb_tokens must have shape (B, K, N, C)")
        if not torch.isfinite(template_rgb_tokens).all():
            raise ValueError("template_rgb_tokens must be finite")
        if not torch.isfinite(candidate_rgb_tokens).all():
            raise ValueError("candidate_rgb_tokens must be finite")

        template = template_rgb_tokens.detach().mean(dim=1)
        candidates = candidate_rgb_tokens.detach().mean(dim=2)
        template = F.normalize(self.projection(template).float(), dim=-1)
        candidates = F.normalize(self.projection(candidates).float(), dim=-1)
        similarity = torch.einsum("bd,bkd->bk", template, candidates)
        return similarity.add(1.0).mul(0.5).clamp(0.0, 1.0)


class EventProposalExtractor(nn.Module):
    """Extract deterministic full-frame proposals from Event activity."""

    def __init__(self, top_k=5, nms_radius=2, min_robust_score=3.0,
                 density_kernel_size=1, eps=1e-6):
        super().__init__()
        self.top_k = int(top_k)
        self.nms_radius = int(nms_radius)
        self.min_robust_score = float(min_robust_score)
        self.density_kernel_size = int(density_kernel_size)
        self.eps = float(eps)
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.nms_radius < 0:
            raise ValueError("nms_radius must be nonnegative")
        if (not math.isfinite(self.min_robust_score)
                or self.min_robust_score < 0.0):
            raise ValueError("min_robust_score must be finite and nonnegative")
        if (self.density_kernel_size < 1
                or self.density_kernel_size % 2 == 0):
            raise ValueError("density_kernel_size must be positive and odd")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and positive")

    @staticmethod
    def _validate_frame(frame, name):
        if frame.ndim != 4 or frame.shape[1] < 1:
            raise ValueError(f"{name} must have shape (B, C, H, W)")
        if frame.shape[2] < 1 or frame.shape[3] < 1:
            raise ValueError(f"{name} spatial dimensions must be positive")
        if not torch.is_floating_point(frame):
            raise ValueError(f"{name} must be floating point")
        if not torch.isfinite(frame).all():
            raise ValueError(f"{name} must be finite")

    def _robust_activity(self, event_frame, background):
        if background is not None:
            activity = (event_frame - background).abs().mean(
                dim=1, keepdim=True)
        else:
            neutral = event_frame.flatten(2).median(
                dim=-1, keepdim=True).values.unsqueeze(-1)
            centered = (event_frame - neutral).abs().mean(
                dim=1, keepdim=True)
            if event_frame.shape[1] > 1:
                color_range = (
                    event_frame.max(dim=1, keepdim=True).values
                    - event_frame.min(dim=1, keepdim=True).values
                )
                activity = torch.maximum(centered, color_range)
            else:
                activity = centered
        if self.density_kernel_size > 1:
            radius = self.density_kernel_size // 2
            activity = F.avg_pool2d(
                F.pad(activity, (radius,) * 4, mode="replicate"),
                kernel_size=self.density_kernel_size,
                stride=1,
            )
        flat = activity.flatten(2)
        median = flat.median(dim=-1, keepdim=True).values
        deviation = (flat - median).abs()
        mad = deviation.median(dim=-1, keepdim=True).values
        scale = (1.4826 * mad).clamp_min(self.eps)
        robust = ((flat - median) / scale).clamp_min(0.0)
        return robust.view_as(activity)

    @staticmethod
    def _normalized_padding_mask(padding_mask, event_frame):
        if padding_mask is None:
            return None
        mask = torch.as_tensor(
            padding_mask, device=event_frame.device, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        expected = (event_frame.shape[0], 1, *event_frame.shape[-2:])
        if mask.shape != expected:
            raise ValueError(
                "padding_mask must have shape (B,H,W) or (B,1,H,W)")
        if (~mask).flatten(1).sum(dim=1).eq(0).any():
            raise ValueError("padding_mask must leave valid image pixels")
        return mask

    @staticmethod
    def _neutralize_padding(frame, padding_mask):
        if frame is None or padding_mask is None:
            return frame
        valid = ~padding_mask[:, 0]
        neutralized = []
        for row in range(frame.shape[0]):
            values = frame[row, :, valid[row]]
            neutral = values.median(dim=-1).values[:, None, None]
            neutralized.append(torch.where(
                valid[row][None], frame[row], neutral))
        return torch.stack(neutralized)

    def forward(self, event_frame, background=None, padding_mask=None):
        self._validate_frame(event_frame, "event_frame")
        if background is not None:
            self._validate_frame(background, "background")
            if background.shape != event_frame.shape:
                raise ValueError("background must match event_frame shape")

        padding_mask = self._normalized_padding_mask(
            padding_mask, event_frame)
        event_frame = self._neutralize_padding(event_frame, padding_mask)
        background = self._neutralize_padding(background, padding_mask)

        heatmap = self._robust_activity(event_frame, background)
        if padding_mask is not None:
            heatmap = heatmap.masked_fill(padding_mask, 0.0)
        kernel_size = 2 * self.nms_radius + 1
        pooled = F.max_pool2d(
            heatmap, kernel_size=kernel_size, stride=1,
            padding=self.nms_radius)
        local_maximum = (
            heatmap.eq(pooled)
            & (heatmap >= self.min_robust_score)
        )
        if padding_mask is not None:
            local_maximum = local_maximum & ~padding_mask
        suppressed = heatmap.flatten(1).masked_fill(
            ~local_maximum.flatten(1), float("-inf"))

        batch_size, _, height, width = heatmap.shape
        count = min(self.top_k, height * width)
        scores, indices = torch.topk(
            suppressed, k=count, dim=-1, largest=True, sorted=True)
        valid = torch.isfinite(scores)
        scores = scores.masked_fill(~valid, 0.0)

        x = indices.remainder(width).to(heatmap.dtype)
        y = torch.div(indices, width, rounding_mode="floor").to(heatmap.dtype)
        x = x / max(width - 1, 1)
        y = y / max(height - 1, 1)
        centers = torch.stack((x, y), dim=-1)
        centers = centers.masked_fill(~valid.unsqueeze(-1), 0.0)

        if count < self.top_k:
            padding = self.top_k - count
            centers = F.pad(centers, (0, 0, 0, padding))
            scores = F.pad(scores, (0, padding))
            valid = F.pad(valid, (0, padding), value=False)

        return {
            "heatmap": heatmap,
            "centers": centers,
            "scores": scores,
            "valid": valid,
        }
