"""Inherited-feature-preserving capability adapters for the backbone tail.

The pretrained backbone always executes its native final transformer blocks,
including AMAH. A routed capability adapter then adds a small inductive-bias
residual.
Every residual output projection is initialized to zero, so all routes are
exactly equivalent to the baseline until training finds useful specialization.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from lib.models.layers.expert_routes import normalize_batch_routes


def _token_layout(x, lens_z=None, lens_x=None):
    total = x.size(1)
    if lens_x is None:
        lens_x = total if lens_z is None else total - int(lens_z)
    lens_x = int(lens_x)
    lens_z = total - lens_x if lens_z is None else int(lens_z)
    if lens_z < 0 or lens_x <= 0 or lens_z + lens_x != total:
        raise ValueError(
            f"invalid token layout: total={total}, lens_z={lens_z}, lens_x={lens_x}")
    return lens_z, lens_x


def _split_search_modalities(x, lens_z=None, lens_x=None):
    lens_z, lens_x = _token_layout(x, lens_z, lens_x)
    if lens_x % 2:
        raise ValueError(f"search token count must contain paired modalities, got {lens_x}")
    per_modality = lens_x // 2
    feat_sz = math.isqrt(per_modality)
    if feat_sz * feat_sz != per_modality:
        raise ValueError(
            f"per-modality search token count must be square, got {per_modality}")
    search = x[:, lens_z:lens_z + lens_x]
    rgb, event = search.split(per_modality, dim=1)
    return lens_z, lens_x, feat_sz, rgb, event


def _replace_search(x, lens_z, lens_x, rgb, event):
    out = x.clone()
    out[:, lens_z:lens_z + lens_x] = torch.cat((rgb, event), dim=1)
    return out


class GeneralistAdapter(nn.Module):
    """Generic token residual with an exactly zero-initialized output."""

    def __init__(self, embed_dim):
        super().__init__()
        hidden_dim = max(embed_dim // 4, 4)
        self.adapter = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x, lens_z=None, lens_x=None):
        _token_layout(x, lens_z, lens_x)
        return x + self.adapter(x)


class EventMotionAdapter(nn.Module):
    """Event-guided dilated spatial residual for fast-motion samples."""

    def __init__(self, embed_dim, dilation=2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.context = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=dilation,
            dilation=dilation, groups=embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, lens_z=None, lens_x=None):
        lens_z, lens_x, f, rgb, event = _split_search_modalities(
            x, lens_z, lens_x)
        event_map = self.norm(event).transpose(1, 2).reshape(x.size(0), x.size(2), f, f)
        residual = self.proj(F.gelu(self.context(event_map)))
        residual = residual.flatten(2).transpose(1, 2)
        return _replace_search(x, lens_z, lens_x, rgb + residual, event + residual)


class HighResDetailAdapter(nn.Module):
    """Local high-frequency residual for sparse small-target evidence."""

    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.detail = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, lens_z=None, lens_x=None):
        lens_z, lens_x, f, rgb, event = _split_search_modalities(
            x, lens_z, lens_x)
        fused = self.norm(0.5 * (rgb + event))
        fused = fused.transpose(1, 2).reshape(x.size(0), x.size(2), f, f)
        detail = self.proj(F.gelu(self.detail(fused) - fused))
        detail = detail.flatten(2).transpose(1, 2)
        return _replace_search(x, lens_z, lens_x, rgb + detail, event + detail)


class CrossAttentionAdapter(nn.Module):
    """Template-to-search residual for background-discrimination samples."""

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        heads = max(1, min(int(num_heads), 4))
        while embed_dim % heads:
            heads -= 1
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x, lens_z=None, lens_x=None):
        lens_z, lens_x = _token_layout(x, lens_z, lens_x)
        if lens_z == 0:
            raise ValueError("discrimination expert requires template tokens")
        template = self.norm_kv(x[:, :lens_z])
        search = x[:, lens_z:lens_z + lens_x]
        residual, _ = self.attn(
            self.norm_q(search), template, template, need_weights=False)
        out = x.clone()
        out[:, lens_z:lens_z + lens_x] = search + residual
        return out


class GlobalContextAdapter(nn.Module):
    """Global template-context residual for visibility and re-entry samples."""

    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, lens_z=None, lens_x=None):
        lens_z, lens_x = _token_layout(x, lens_z, lens_x)
        if lens_z == 0:
            raise ValueError("visibility expert requires template tokens")
        context_tokens = x[:, :lens_z]
        context = self.proj(self.norm(context_tokens).mean(dim=1, keepdim=True))
        out = x.clone()
        search = out[:, lens_z:lens_z + lens_x]
        out[:, lens_z:lens_z + lens_x] = search + context.expand_as(search)
        return out


class HeterogeneousTail(nn.Module):
    """Sparse bank of inherited-feature-preserving capability residuals.

    ``tail_depth`` records where PETTrack should obtain the pre-tail router
    features. The native backbone, rather than this module, executes those
    transformer blocks so pretrained weights and AMAH behavior stay intact.
    """

    EXPERT_ADAPTERS = {
        "generalist": GeneralistAdapter,
        "motion_fm": EventMotionAdapter,
        "small_target_st": HighResDetailAdapter,
        "discrimination_bi": CrossAttentionAdapter,
        "visibility_foc_ov": GlobalContextAdapter,
    }

    def __init__(self, embed_dim, num_heads, tail_depth=3, expert_names=None,
                 drop_path_rate=0.0):
        super().__init__()
        # Kept in the constructor for existing YAML/API compatibility. Drop
        # path remains owned by the native transformer tail; applying it again
        # inside a residual adapter would break exact baseline initialization.
        self.drop_path_rate = float(drop_path_rate)
        self.expert_names = list(expert_names or ["generalist"])
        if not self.expert_names:
            raise ValueError("expert_names must not be empty")
        if len(set(self.expert_names)) != len(self.expert_names):
            raise ValueError("expert_names must be unique")
        unknown = sorted(set(self.expert_names) - set(self.EXPERT_ADAPTERS))
        if unknown:
            raise ValueError(f"unsupported heterogeneous-tail experts: {unknown}")
        if "generalist" in self.expert_names:
            self.default_expert = "generalist"
        else:
            raise ValueError(
                "heterogeneous-tail experts must include generalist")
        self.tail_depth = int(tail_depth)
        self.adapters = nn.ModuleDict()
        for name in self.expert_names:
            adapter_cls = self.EXPERT_ADAPTERS[name]
            if adapter_cls is CrossAttentionAdapter:
                adapter = adapter_cls(embed_dim, num_heads=num_heads)
            else:
                adapter = adapter_cls(embed_dim)
            self.adapters[name] = adapter
        self.residual_scale_logits = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in self.expert_names
        })

    def forward_composed(self, x, routes=None, lens_z=None, lens_x=None):
        lens_z, lens_x = _token_layout(x, lens_z, lens_x)
        batch_routes = normalize_batch_routes(
            routes,
            self.expert_names,
            x.shape[0],
            self.default_expert,
        )
        output = torch.empty_like(x)
        for route in dict.fromkeys(batch_routes):
            indexes = [
                index for index, sample_route in enumerate(batch_routes)
                if sample_route == route
            ]
            batch_index = torch.tensor(
                indexes, device=x.device, dtype=torch.long)
            selected = x.index_select(0, batch_index)
            residual = torch.zeros_like(selected)
            for name in route:
                adapted = self.adapters[name](
                    selected, lens_z=lens_z, lens_x=lens_x)
                scale = 2.0 * self.residual_scale_logits[name].sigmoid()
                residual = residual + scale * (adapted - selected)
            selected = selected + residual / math.sqrt(len(route))
            output.index_copy_(0, batch_index, selected)
        return output


def build_heterogeneous_tail(cfg, embed_dim, num_heads):
    tail_cfg = getattr(cfg.MODEL, "HETEROGENEOUS_TAIL", None)
    tail_depth = int(getattr(tail_cfg, "DEPTH", 3)) if tail_cfg else 3
    drop_path_rate = float(getattr(tail_cfg, "DROP_PATH", 0.0)) if tail_cfg else 0.0
    expert_names = list(getattr(cfg.MODEL.EXPERT, "NAMES", ["generalist"]))
    return HeterogeneousTail(
        embed_dim, num_heads, tail_depth=tail_depth,
        expert_names=expert_names, drop_path_rate=drop_path_rate)
