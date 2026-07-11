import math

import torch
from torch import nn

from lib.models.layers.expert_routes import normalize_batch_routes


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
    """Generic zero-initialized residual around the shared fusion."""

    def __init__(self, embed_dim):
        super().__init__()
        self.adapter = _ResidualAdapter(embed_dim)

    def forward(self, rgb_tokens, event_tokens, context=None):
        shared = rgb_tokens + event_tokens
        return shared + self.adapter(shared)


class MotionFusion(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.rgb_gain = nn.Parameter(torch.zeros(1))
        self.event_gain = nn.Parameter(torch.zeros(1))
        self.diff_adapter = _ResidualAdapter(embed_dim)
        self.event_adapter = _ResidualAdapter(embed_dim)

    def forward(self, rgb_tokens, event_tokens, context=None):
        fused = rgb_tokens * (1.0 + self.rgb_gain) + event_tokens * (1.0 + self.event_gain)
        motion_signal = event_tokens - rgb_tokens
        return fused + self.diff_adapter(motion_signal) + self.event_adapter(event_tokens)


class SmallTargetFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        hidden_dim = max(embed_dim // 4, 4)
        self.detail_adapter = _ResidualAdapter(embed_dim)
        self.token_gate = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.token_gate[-2].weight)
        nn.init.zeros_(self.token_gate[-2].bias)

    def forward(self, rgb_tokens, event_tokens, context=None):
        fused = rgb_tokens + event_tokens
        local_contrast = fused - fused.mean(dim=1, keepdim=True)
        modality_detail = rgb_tokens - event_tokens
        modality_detail = modality_detail - modality_detail.mean(dim=1, keepdim=True)
        detail = self.detail_adapter(local_contrast + modality_detail)
        return fused + self.token_gate(local_contrast) * detail


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
        gate = torch.sigmoid(self.bridge_gate(torch.cat([fused, template_state.expand_as(fused)], dim=-1)))
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

        rgb_weight = weights[..., 0:1] * 2.0
        event_weight = weights[..., 1:2] * 2.0
        fused = rgb_tokens * rgb_weight + event_tokens * event_weight
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
        recovery_features = torch.cat([fused, template_state.expand_as(fused), disagreement], dim=-1)
        return fused + self.recovery_gate(recovery_features) * self.recovery_out(recovery_state)


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

    def forward_composed(self, rgb_tokens, event_tokens, context=None,
                         routes=None):
        batch_routes = normalize_batch_routes(
            routes,
            tuple(self.experts.keys()),
            rgb_tokens.shape[0],
            self.default_expert,
        )
        output = torch.empty_like(rgb_tokens)
        for route in dict.fromkeys(batch_routes):
            indexes = [
                index for index, sample_route in enumerate(batch_routes)
                if sample_route == route
            ]
            batch_index = torch.tensor(
                indexes, device=rgb_tokens.device, dtype=torch.long)
            sample_context = self._slice_context(context, batch_index)
            selected_rgb = rgb_tokens.index_select(0, batch_index)
            selected_event = event_tokens.index_select(0, batch_index)
            selected_output = self._compose_route(
                selected_rgb, selected_event, sample_context, route)
            output.index_copy_(0, batch_index, selected_output)
        return output

    def _compose_route(self, rgb_tokens, event_tokens, context, route):
        shared = rgb_tokens + event_tokens
        residual = torch.zeros_like(shared)
        for name in route:
            expert_output = self.experts[name](
                rgb_tokens, event_tokens, context=context)
            scale = 2.0 * self.residual_scale_logits[name].sigmoid()
            residual = residual + scale * (expert_output - shared)
        return shared + residual / math.sqrt(len(route))

    @staticmethod
    def _slice_context(context, batch_index):
        if not context:
            return context
        sliced = {}
        for key, value in context.items():
            sliced[key] = value.index_select(0, batch_index) if torch.is_tensor(value) else value
        return sliced


def build_expert_fusions(expert_names, embed_dim):
    experts = {}
    for name in expert_names:
        key = name.lower()
        if key == "generalist":
            experts[name] = GeneralistFusion(embed_dim)
        elif key == "fm" or key.endswith("_fm") or "motion" in key:
            experts[name] = MotionFusion(embed_dim)
        elif key == "st" or key.endswith("_st") or "small" in key:
            experts[name] = SmallTargetFusion(embed_dim)
        elif key == "bi" or key.endswith("_bi") or "discrimination" in key:
            experts[name] = TemplateBridgeFusion(embed_dim)
        elif "visibility" in key or "foc" in key or key.endswith("_ov") or "_ov" in key:
            experts[name] = VisibilityRecoveryFusion(embed_dim)
        else:
            raise ValueError(f"unsupported fusion expert: {name}")
    return experts
