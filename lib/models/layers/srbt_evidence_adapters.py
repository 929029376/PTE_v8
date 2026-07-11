"""SRBT evidence adapters for paired RGB-event search tokens."""
import math

import torch
import torch.nn.functional as F
from torch import nn


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
        raise ValueError(
            f"search token count must contain paired modalities, got {lens_x}")
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
        event_map = self.norm(event).transpose(1, 2).reshape(
            x.size(0), x.size(2), f, f)
        residual = self.proj(F.gelu(self.context(event_map)))
        residual = residual.flatten(2).transpose(1, 2)
        return _replace_search(
            x, lens_z, lens_x, rgb + residual, event + residual)


class HighResDetailAdapter(nn.Module):
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
            raise ValueError("cross evidence requires template tokens")
        template = self.norm_kv(x[:, :lens_z])
        search = x[:, lens_z:lens_z + lens_x]
        residual, _ = self.attn(
            self.norm_q(search), template, template, need_weights=False)
        out = x.clone()
        out[:, lens_z:lens_z + lens_x] = search + residual
        return out


class GlobalContextAdapter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, lens_z=None, lens_x=None):
        lens_z, lens_x = _token_layout(x, lens_z, lens_x)
        if lens_z == 0:
            raise ValueError("identity evidence requires template tokens")
        context_tokens = x[:, :lens_z]
        context = self.proj(self.norm(context_tokens).mean(dim=1, keepdim=True))
        out = x.clone()
        search = out[:, lens_z:lens_z + lens_x]
        out[:, lens_z:lens_z + lens_x] = search + context.expand_as(search)
        return out


def build_evidence_adapters(embed_dim, num_heads):
    return nn.ModuleDict({
        "appearance": GeneralistAdapter(embed_dim),
        "motion": EventMotionAdapter(embed_dim),
        "detail": HighResDetailAdapter(embed_dim),
        "cross": CrossAttentionAdapter(embed_dim, num_heads=num_heads),
        "identity": GlobalContextAdapter(embed_dim),
    })
