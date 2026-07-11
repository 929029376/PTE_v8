"""Candidate-level RGB-Event evidence composition for SRBT."""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from lib.models.layers.srbt_evidence_adapters import (
    _split_search_modalities,
    build_evidence_adapters,
)


EVIDENCE_NAMES = ("appearance", "motion", "detail", "cross", "identity")


class _LikelihoodScorer(nn.Sequential):
    def __init__(self, embed_dim):
        super().__init__(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )
        nn.init.zeros_(self[-1].bias)


def _reliability_head(embed_dim, context_dim, hidden_dim):
    head = nn.Sequential(
        nn.LayerNorm(embed_dim + context_dim),
        nn.Linear(embed_dim + context_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 1),
    )
    nn.init.zeros_(head[-1].bias)
    return head


def _candidate_pool(feature_map, candidate_boxes, pool_size):
    """Bilinearly pool normalized cxcywh candidate regions."""
    if candidate_boxes.ndim != 3 or candidate_boxes.shape[-1] != 4:
        raise ValueError("candidate_boxes must have shape (B, K, 4)")
    if candidate_boxes.shape[0] != feature_map.shape[0]:
        raise ValueError("candidate_boxes and feature_map batch sizes differ")
    if not torch.isfinite(candidate_boxes).all():
        raise ValueError("candidate_boxes must be finite")

    boxes = candidate_boxes.to(device=feature_map.device, dtype=feature_map.dtype)
    boxes = boxes.clamp(0.0, 1.0)
    batch, candidates = boxes.shape[:2]
    coordinates = torch.linspace(
        -1.0, 1.0, pool_size, device=feature_map.device, dtype=feature_map.dtype)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, pool_size, pool_size, 2)
    centers = 2.0 * boxes[..., :2] - 1.0
    sizes = boxes[..., 2:]
    grid = centers[:, :, None, None, :] + base_grid * sizes[:, :, None, None, :]

    sampled = F.grid_sample(
        feature_map.repeat_interleave(candidates, dim=0),
        grid.reshape(batch * candidates, pool_size, pool_size, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.mean(dim=(-1, -2)).reshape(batch, candidates, feature_map.shape[1])


def _candidate_context(value, batch, candidates, width, name):
    if value.ndim == 2:
        value = value[:, None, :].expand(-1, candidates, -1)
    if value.shape != (batch, candidates, width):
        raise ValueError(
            f"{name} must have shape (B, {width}) or (B, K, {width}), "
            f"got {tuple(value.shape)}")
    return value


class EvidenceBank(nn.Module):
    """Compose five independently gated likelihoods for each candidate."""

    def __init__(self, embed_dim, num_heads, search_tokens_per_modality,
                 state_dim=32, quality_dim=16, gate_hidden_dim=128,
                 gate_epsilon=1.0, pool_size=3):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.search_tokens_per_modality = int(search_tokens_per_modality)
        self.state_dim = int(state_dim)
        self.quality_dim = int(quality_dim)
        self.gate_epsilon = float(gate_epsilon)
        self.pool_size = int(pool_size)
        if self.search_tokens_per_modality <= 0:
            raise ValueError("search_tokens_per_modality must be positive")
        if self.gate_epsilon <= 0:
            raise ValueError("gate_epsilon must be positive")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")

        self.adapters = build_evidence_adapters(self.embed_dim, num_heads)
        self.likelihood_scorers = nn.ModuleDict({
            name: _LikelihoodScorer(self.embed_dim) for name in EVIDENCE_NAMES
        })
        context_dim = self.state_dim + self.quality_dim
        self.reliability_heads = nn.ModuleDict({
            name: _reliability_head(
                self.embed_dim, context_dim, int(gate_hidden_dim))
            for name in EVIDENCE_NAMES
        })

    def _search_maps(self, tokens):
        lens_x = 2 * self.search_tokens_per_modality
        _, _, feat_size, rgb, event = _split_search_modalities(
            tokens, lens_x=lens_x)
        channels = tokens.shape[-1]
        rgb_map = rgb.transpose(1, 2).reshape(tokens.shape[0], channels, feat_size, feat_size)
        event_map = event.transpose(1, 2).reshape(tokens.shape[0], channels, feat_size, feat_size)
        return rgb_map, event_map

    def forward(self, shared_tokens, detail_tokens, identity_tokens,
                candidate_boxes, quality_stats, prior_belief):
        for name, tokens in (
                ("shared_tokens", shared_tokens),
                ("detail_tokens", detail_tokens),
                ("identity_tokens", identity_tokens)):
            if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
                raise ValueError(
                    f"{name} must have shape (B, N, {self.embed_dim})")
            if tokens.shape[:2] != shared_tokens.shape[:2]:
                raise ValueError("all evidence token tensors must share (B, N)")

        lens_x = 2 * self.search_tokens_per_modality
        appearance_tokens = self.adapters["appearance"](
            shared_tokens, lens_x=lens_x)
        shared_rgb, shared_event = self._search_maps(appearance_tokens)

        motion_tokens = self.adapters["motion"](
            shared_tokens, lens_x=lens_x)
        _, motion_event = self._search_maps(motion_tokens)
        event_energy = shared_event.square().mean(dim=(1, 2, 3), keepdim=True)
        motion_event = motion_event * (event_energy / (event_energy + 1e-6))

        detail_adapted = self.adapters["detail"](
            detail_tokens, lens_x=lens_x)
        detail_rgb, detail_event = self._search_maps(detail_adapted)

        cross_adapted = self.adapters["cross"](
            shared_tokens, lens_x=lens_x)
        cross_rgb, cross_event = self._search_maps(cross_adapted)
        identity_adapted = self.adapters["identity"](
            identity_tokens, lens_x=lens_x)
        identity_rgb, identity_event = self._search_maps(identity_adapted)

        evidence_maps = OrderedDict((
            ("appearance", shared_rgb),
            ("motion", motion_event),
            ("detail", 0.5 * (detail_rgb + detail_event)),
            ("cross", 0.5 * (cross_rgb + cross_event)),
            ("identity", 0.5 * (identity_rgb + identity_event)),
        ))
        evidence = OrderedDict(
            (name, _candidate_pool(feature, candidate_boxes, self.pool_size))
            for name, feature in evidence_maps.items()
        )

        likelihood_logits = torch.stack([
            self.likelihood_scorers[name](evidence[name]).squeeze(-1)
            for name in EVIDENCE_NAMES
        ], dim=-1)
        batch, candidates = candidate_boxes.shape[:2]
        prior = _candidate_context(
            prior_belief, batch, candidates, self.state_dim, "prior_belief")
        quality = _candidate_context(
            quality_stats, batch, candidates, self.quality_dim, "quality_stats")
        context = torch.cat((prior, quality), dim=-1)
        gates = torch.stack([
            torch.sigmoid(self.reliability_heads[name](
                torch.cat((evidence[name], context), dim=-1)).squeeze(-1))
            for name in EVIDENCE_NAMES
        ], dim=-1)
        weights = gates / (gates.sum(dim=-1, keepdim=True) + self.gate_epsilon)
        combined = (weights * likelihood_logits).sum(dim=-1)
        return {
            "evidence_maps": evidence_maps,
            "evidence": evidence,
            "likelihood_logits": likelihood_logits,
            "gates": gates,
            "weights": weights,
            "combined_likelihood": combined,
        }
