from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


EVIDENCE_NAMES = ("appearance", "motion", "detail", "cross", "identity")


class FuturePosteriorTeacher(nn.Module):
    """Training-only future posterior head over detached evidence maps."""

    def __init__(self, input_dim=768, pooled_dim=128, d_model=256,
                 nhead=8, num_layers=2, ffn_dim=1024, spatial_dim=64,
                 identity_dim=32, hazard_bins=129, max_horizon=128):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if min(input_dim, pooled_dim, d_model, spatial_dim,
               identity_dim, hazard_bins, max_horizon) < 1:
            raise ValueError("teacher dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hazard_bins = int(hazard_bins)
        self.max_horizon = int(max_horizon)

        self.temporal_projections = nn.ModuleDict({
            name: nn.Linear(input_dim, pooled_dim)
            for name in EVIDENCE_NAMES
        })
        self.temporal_fusion = nn.Linear(
            len(EVIDENCE_NAMES) * pooled_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_horizon, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.existence_head = nn.Linear(d_model, 1)
        self.hazard_head = nn.Linear(d_model, hazard_bins)
        self.time_head = nn.Linear(d_model, 1)

        self.spatial_projections = nn.ModuleDict({
            name: nn.Conv2d(input_dim, spatial_dim, kernel_size=1)
            for name in EVIDENCE_NAMES
        })
        self.reliability_heads = nn.ModuleDict({
            name: nn.Linear(pooled_dim, 1)
            for name in EVIDENCE_NAMES
        })
        self.field_head = nn.Conv2d(spatial_dim, 1, kernel_size=1)
        self.candidate_head = nn.Conv2d(spatial_dim, 1, kernel_size=1)
        self.identity_head = nn.Conv2d(
            spatial_dim, identity_dim, kernel_size=1)

    def _validate_inputs(self, evidence_maps, valid_mask):
        if not isinstance(evidence_maps, Mapping):
            raise TypeError("detached_evidence_maps must be a mapping")
        if set(evidence_maps) != set(EVIDENCE_NAMES):
            raise ValueError(f"evidence maps must use keys {EVIDENCE_NAMES}")
        reference = evidence_maps[EVIDENCE_NAMES[0]]
        if reference.ndim != 5 or reference.shape[2] != self.input_dim:
            raise ValueError(
                "each evidence map must have shape (B, H, C, F, F)")
        batch, horizon, _, height, width = reference.shape
        if height != width:
            raise ValueError("teacher evidence grid must be square")
        if horizon > self.max_horizon:
            raise ValueError("future horizon exceeds max_horizon")
        for name in EVIDENCE_NAMES[1:]:
            if evidence_maps[name].shape != reference.shape:
                raise ValueError("all teacher evidence maps must share shape")
        valid = torch.as_tensor(
            valid_mask, device=reference.device, dtype=torch.bool)
        if valid.shape != (batch, horizon):
            raise ValueError("valid_mask must have shape (B, H)")
        if (~valid.any(dim=1)).any():
            raise ValueError("each sample needs at least one valid future frame")
        return valid, batch, horizon, height, width

    def forward(self, detached_evidence_maps, valid_mask):
        valid, batch, horizon, height, width = self._validate_inputs(
            detached_evidence_maps, valid_mask)
        frame_mask = valid[:, :, None, None, None]
        maps = {
            name: detached_evidence_maps[name].detach().masked_fill(
                ~frame_mask, 0.0)
            for name in EVIDENCE_NAMES
        }

        pooled = {
            name: self.temporal_projections[name](
                maps[name].mean(dim=(-1, -2)))
            for name in EVIDENCE_NAMES
        }
        temporal = self.temporal_fusion(torch.cat(
            [pooled[name] for name in EVIDENCE_NAMES], dim=-1))
        temporal = temporal + self.position[:, :horizon]
        encoded = self.temporal_encoder(
            temporal, src_key_padding_mask=~valid)
        valid_float = valid.to(dtype=encoded.dtype)
        summary = (
            encoded * valid_float.unsqueeze(-1)
        ).sum(dim=1) / valid_float.sum(dim=1, keepdim=True)

        existence = torch.sigmoid(self.existence_head(summary).squeeze(-1))
        hazard = F.softmax(self.hazard_head(summary), dim=-1)
        time_logits = self.time_head(encoded).squeeze(-1)
        time_logits = time_logits.masked_fill(~valid, torch.finfo(
            time_logits.dtype).min)
        time_weights = F.softmax(time_logits, dim=-1)

        gate_logits = torch.cat([
            self.reliability_heads[name](pooled[name])
            for name in EVIDENCE_NAMES
        ], dim=-1)
        gates = F.softmax(gate_logits, dim=-1)
        fused = None
        for evidence_id, name in enumerate(EVIDENCE_NAMES):
            projected = self.spatial_projections[name](
                maps[name].reshape(
                    batch * horizon, self.input_dim, height, width)
            ).view(batch, horizon, -1, height, width)
            weighted = projected * gates[:, :, evidence_id, None, None, None]
            fused = weighted if fused is None else fused + weighted

        flat_fused = fused.reshape(batch * horizon, -1, height, width)
        field_frames = F.softmax(
            self.field_head(flat_fused).view(
                batch, horizon, 1, height * width),
            dim=-1,
        ).view(batch, horizon, 1, height, width)
        candidate_frames = F.softmax(
            self.candidate_head(flat_fused).view(
                batch, horizon, 1, height * width),
            dim=-1,
        ).view(batch, horizon, 1, height, width)
        identity_frames = F.normalize(
            self.identity_head(flat_fused).view(
                batch, horizon, -1, height, width),
            dim=2,
        )
        time = time_weights[:, :, None, None, None]
        field = (time * field_frames).sum(dim=1)
        candidate_map = (time * candidate_frames).sum(dim=1)
        identity_map = F.normalize((time * identity_frames).sum(dim=1), dim=1)
        return {
            "existence": existence,
            "hazard": hazard,
            "time_weights": time_weights,
            "field": field,
            "candidate_map": candidate_map,
            "identity_map": identity_map,
        }


def build_future_posterior_teacher(cfg):
    teacher_cfg = cfg.MODEL.SRBT.TEACHER
    return FuturePosteriorTeacher(
        input_dim=getattr(teacher_cfg, "INPUT_DIM", 768),
        pooled_dim=getattr(teacher_cfg, "POOLED_DIM", 128),
        d_model=getattr(teacher_cfg, "D_MODEL", 256),
        nhead=getattr(teacher_cfg, "NHEAD", 8),
        num_layers=getattr(teacher_cfg, "NUM_LAYERS", 2),
        ffn_dim=getattr(teacher_cfg, "FFN_DIM", 1024),
        spatial_dim=getattr(teacher_cfg, "SPATIAL_DIM", 64),
        identity_dim=getattr(teacher_cfg, "IDENTITY_DIM", 32),
        hazard_bins=getattr(teacher_cfg, "HAZARD_BINS", 129),
        max_horizon=getattr(teacher_cfg, "MAX_HORIZON", 128),
    )
