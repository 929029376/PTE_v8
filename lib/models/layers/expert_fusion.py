import math

import torch
import torch.nn.functional as F
from torch import nn


class _ResidualAdapter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        hidden_dim = max(embed_dim // 4, 4)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tokens):
        return self.net(tokens)


class GeneralistFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.adapter = _ResidualAdapter(embed_dim)

    def forward(self, rgb_tokens, event_tokens, context=None):
        shared = rgb_tokens + event_tokens
        return shared + self.adapter(shared)


class MotionFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.rgb_gain = nn.Parameter(torch.zeros(1))
        self.event_gain = nn.Parameter(torch.zeros(1))
        self.diff_adapter = _ResidualAdapter(embed_dim)
        self.event_adapter = _ResidualAdapter(embed_dim)
        temporal_dim = max(min(embed_dim // 32, 32), 8)
        self.temporal_stem = nn.Sequential(
            nn.Conv2d(2, temporal_dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, temporal_dim),
            nn.GELU(),
            nn.Conv2d(
                temporal_dim, temporal_dim, 3, stride=2, padding=1,
                groups=temporal_dim, bias=False),
            nn.GroupNorm(1, temporal_dim),
            nn.GELU(),
            nn.Conv2d(temporal_dim, embed_dim, 1, bias=False),
        )
        self.temporal_box = nn.Sequential(
            nn.Linear(4, temporal_dim),
            nn.GELU(),
            nn.Linear(temporal_dim, embed_dim),
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.temporal_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _event_energy(event_image):
        flattened = event_image.flatten(2)
        median = flattened.median(dim=-1).values[..., None, None]
        energy = (event_image - median).abs().mean(dim=1, keepdim=True)
        scale = energy.flatten(2).mean(dim=-1, keepdim=True)[..., None]
        return energy / scale.clamp_min(1e-6)

    def _temporal_residual(self, tokens, context):
        required = (
            "current_event", "previous_event", "history_valid", "box_delta")
        if not isinstance(context, dict) or any(
                key not in context for key in required):
            return None
        current = torch.as_tensor(
            context["current_event"], device=tokens.device,
            dtype=tokens.dtype)
        previous = torch.as_tensor(
            context["previous_event"], device=tokens.device,
            dtype=tokens.dtype)
        if current.ndim != 4 or previous.shape != current.shape:
            raise ValueError(
                "causal motion events must have matching [B, C, H, W] shapes")
        if current.shape[0] != tokens.shape[0]:
            raise ValueError("causal motion event batch must match token batch")
        history_valid = torch.as_tensor(
            context["history_valid"], device=tokens.device,
            dtype=torch.bool).reshape(-1)
        if history_valid.numel() != tokens.shape[0]:
            raise ValueError("history_valid must provide one flag per batch row")
        if not bool(history_valid.any()):
            return None
        box_delta = torch.as_tensor(
            context["box_delta"], device=tokens.device,
            dtype=tokens.dtype)
        if box_delta.shape != (tokens.shape[0], 4):
            raise ValueError("box_delta must have shape [B, 4]")

        current_energy = self._event_energy(current)
        previous_energy = self._event_energy(previous)
        motion_image = torch.cat((
            current_energy,
            current_energy - previous_energy,
        ), dim=1)
        temporal = self.temporal_stem(motion_image)
        token_count = tokens.shape[1]
        side = math.isqrt(token_count)
        output_size = (side, side) if side * side == token_count else (1, token_count)
        temporal = F.adaptive_avg_pool2d(temporal, output_size)
        temporal = temporal.flatten(2).transpose(1, 2)
        temporal = self.temporal_norm(
            temporal + self.temporal_box(box_delta).unsqueeze(1))
        return temporal * history_valid[:, None, None].to(tokens.dtype)

    def forward(self, rgb_tokens, event_tokens, context=None):
        fused = rgb_tokens * (1.0 + self.rgb_gain) + event_tokens * (1.0 + self.event_gain)
        output = (
            fused
            + self.diff_adapter(event_tokens - rgb_tokens)
            + self.event_adapter(event_tokens)
        )
        temporal = self._temporal_residual(output, context)
        if temporal is None:
            return output
        return output + torch.tanh(self.temporal_scale) * temporal


class TemplateBridgeFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.search_norm = nn.LayerNorm(embed_dim)
        self.template_norm = nn.LayerNorm(embed_dim)
        self.search_proj = nn.Linear(embed_dim, embed_dim)
        self.template_proj = nn.Linear(embed_dim, embed_dim)
        self.bridge_gate = nn.Linear(embed_dim * 2, 1)
        self.bridge_out = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.bridge_out.weight)
        nn.init.zeros_(self.bridge_out.bias)

    def forward(self, rgb_tokens, event_tokens, context=None):
        fused = rgb_tokens + event_tokens
        if not context or "template_tokens" not in context:
            return fused
        template_tokens = context["template_tokens"].mean(dim=1, keepdim=True)
        search_state = self.search_proj(self.search_norm(fused))
        template_state = self.template_proj(self.template_norm(template_tokens))
        bridge_state = torch.tanh(search_state * template_state)
        gate = torch.sigmoid(self.bridge_gate(torch.cat([
            fused, template_state.expand_as(fused)
        ], dim=-1)))
        return fused + gate * self.bridge_out(bridge_state)


class ModalityGateFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.global_gate = nn.Linear(embed_dim * 3, 2)
        self.local_gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )
        self.disagreement_adapter = _ResidualAdapter(embed_dim)
        nn.init.zeros_(self.global_gate.weight)
        nn.init.zeros_(self.global_gate.bias)
        nn.init.zeros_(self.local_gate[-1].weight)
        nn.init.zeros_(self.local_gate[-1].bias)

    def forward(self, rgb_tokens, event_tokens, context=None):
        disagreement = (rgb_tokens - event_tokens).abs()
        global_features = torch.cat([
            rgb_tokens.mean(dim=1),
            event_tokens.mean(dim=1),
            disagreement.mean(dim=1),
        ], dim=-1)
        global_weights = self.global_gate(global_features).softmax(dim=-1).unsqueeze(1)
        local_features = torch.cat([rgb_tokens, event_tokens, disagreement], dim=-1)
        local_weights = self.local_gate(local_features).softmax(dim=-1)
        weights = (global_weights + local_weights) * 0.5
        fused = (
            rgb_tokens * weights[..., 0:1] * 2.0
            + event_tokens * weights[..., 1:2] * 2.0
        )
        return fused + self.disagreement_adapter(disagreement)


class VisibilityRecoveryFusion(ModalityGateFusion):
    def __init__(self, embed_dim):
        super().__init__(embed_dim)
        self.search_norm = nn.LayerNorm(embed_dim)
        self.template_norm = nn.LayerNorm(embed_dim)
        self.search_proj = nn.Linear(embed_dim, embed_dim)
        self.template_proj = nn.Linear(embed_dim, embed_dim)
        self.recovery_gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, 1),
            nn.Sigmoid(),
        )
        self.recovery_out = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.recovery_out.weight)
        nn.init.zeros_(self.recovery_out.bias)

    def forward(self, rgb_tokens, event_tokens, context=None):
        fused = super().forward(rgb_tokens, event_tokens, context=context)
        if not context or "template_tokens" not in context:
            return fused
        template_tokens = context["template_tokens"].mean(dim=1, keepdim=True)
        search_state = self.search_proj(self.search_norm(fused))
        template_state = self.template_proj(self.template_norm(template_tokens))
        disagreement = (rgb_tokens - event_tokens).abs()
        recovery_state = torch.tanh(search_state * template_state)
        recovery_features = torch.cat([
            fused, template_state.expand_as(fused), disagreement
        ], dim=-1)
        return fused + self.recovery_gate(recovery_features) * self.recovery_out(recovery_state)


class ProposalBoxAdapter(nn.Module):
    """Let one expert softly correct a detached upstream box proposal."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, direct_boxes, upstream_boxes, score_map):
        if (
                direct_boxes.ndim != 3
                or direct_boxes.shape[-1] != 4
                or upstream_boxes.shape != direct_boxes.shape):
            raise ValueError(
                "direct and upstream boxes must have matching (B, N, 4) shapes")
        if score_map.ndim < 2 or score_map.shape[0] != direct_boxes.shape[0]:
            raise ValueError("score_map batch must match box batch")

        upstream = upstream_boxes.detach()
        peak = score_map.flatten(1).amax(dim=1, keepdim=True)
        peak = peak[:, None, :].expand(-1, direct_boxes.shape[1], -1)
        features = torch.cat([
            direct_boxes,
            upstream,
            upstream - direct_boxes,
            peak,
        ], dim=-1)
        gate = torch.tanh(self.output(self.features(features)))
        corrected = direct_boxes + gate * (upstream - direct_boxes)
        center = corrected[..., :2].clamp(0.0, 1.0)
        size = corrected[..., 2:].clamp(1e-4, 1.0)
        return torch.cat([center, size], dim=-1), gate


class ExpertFusionBank(nn.Module):
    def __init__(self, experts, default_expert="generalist"):
        super().__init__()
        self.experts = nn.ModuleDict(experts)
        self.default_expert = default_expert
        if self.default_expert not in self.experts:
            raise ValueError(f"default expert '{self.default_expert}' is not registered")
        self.residual_scale_logits = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in self.experts
        })

    def forward_expert(self, name, rgb_tokens, event_tokens, context=None):
        if name not in self.experts:
            raise ValueError(f"unknown fusion expert: {name}")
        shared = rgb_tokens + event_tokens
        expert_output = self.experts[name](
            rgb_tokens, event_tokens, context=context)
        scale = 2.0 * self.residual_scale_logits[name].sigmoid()
        return shared + scale * (expert_output - shared)


def build_expert_fusions(expert_names, embed_dim):
    experts = {}
    for name in expert_names:
        key = name.lower()
        if key == "generalist":
            experts[name] = GeneralistFusion(embed_dim)
        elif key == "fm" or key.endswith("_fm") or "motion" in key:
            experts[name] = MotionFusion(embed_dim)
        elif key == "bi" or key.endswith("_bi") or "discrimination" in key:
            experts[name] = TemplateBridgeFusion(embed_dim)
        elif "visibility" in key or "foc" in key or key.endswith("_ov") or "_ov" in key:
            experts[name] = VisibilityRecoveryFusion(embed_dim)
        else:
            raise ValueError(f"unsupported fusion expert: {name}")
    return experts
