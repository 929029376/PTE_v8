"""
Expert routers for PET-Track.

Two variants:
  * ExpertRouter          — feature-only baseline router (no physical input).
  * PhysicsExpertRouter   — fuses backbone features with the *unified event
                            physical belief* embedding b_t (preferred), or
                            falls back to raw EPSM statistics (ablation:
                            USE_SHARED_BELIEF=False).

The key change vs. the pre-belief version: the physics branch consumes the
shared `belief_embed` (belief_dim, default 64) produced by EventPhysicalBelief,
NOT a slice of raw 8-dim statistics. This is what makes the router a *consumer*
of the unified representation rather than a parallel reader of scalars. The
`physics=` kwarg is retained as a legacy/ablation path: when `belief` is None
and `physics` is given, the branch reads raw statistics (the old behavior), so
the ablation `USE_SHARED_BELIEF=False` is a pure config flip.

Supervision is counterfactual route utility: the router learns which singleton
or pair route tracks best without challenge-category labels.
"""
import math
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn


def enumerate_sparse_routes(expert_names, max_active=2):
    """Enumerate canonical singleton and combination routes."""
    names = tuple(expert_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("expert names must be non-empty and unique")
    if max_active < 1 or max_active > len(names):
        raise ValueError("max_active is outside the expert vocabulary")
    return tuple(
        route
        for route_size in range(1, max_active + 1)
        for route in combinations(names, route_size)
    )


class SparseRouteHysteresis:
    """Causal single-sequence route stabilizer for inference."""

    def __init__(self, route_options, margin=0.05, patience=2,
                 confidence_threshold=0.0):
        self.route_options = tuple(tuple(route) for route in route_options)
        if not self.route_options or len(set(self.route_options)) != len(
                self.route_options):
            raise ValueError("route_options must be non-empty and unique")
        self.margin = float(margin)
        self.patience = int(patience)
        self.confidence_threshold = float(confidence_threshold)
        if self.margin < 0.0 or self.patience < 1:
            raise ValueError("invalid route hysteresis settings")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("route confidence threshold must be in [0, 1]")
        self.reset()

    def reset(self):
        self.active_id = None
        self.pending_id = None
        self.pending_count = 0

    def select(self, router_out):
        probabilities = router_out["probabilities"].detach()
        if probabilities.ndim != 2 or probabilities.shape != (
                1, len(self.route_options)):
            raise ValueError(
                "temporal route hysteresis expects one inference sample")
        candidate_id = int(probabilities.argmax(dim=-1).item())
        candidate_confidence = float(probabilities[0, candidate_id].item())
        if self.active_id is None:
            self.active_id = candidate_id
            return [self.route_options[self.active_id]]

        active_probability = float(
            probabilities[0, self.active_id].item())
        challenge_is_valid = (
            candidate_id != self.active_id
            and candidate_confidence >= self.confidence_threshold
            and candidate_confidence > active_probability + self.margin
        )
        if not challenge_is_valid:
            self.pending_id = None
            self.pending_count = 0
            return [self.route_options[self.active_id]]

        if self.pending_id == candidate_id:
            self.pending_count += 1
        else:
            self.pending_id = candidate_id
            self.pending_count = 1
        if self.pending_count >= self.patience:
            self.active_id = candidate_id
            self.pending_id = None
            self.pending_count = 0
        return [self.route_options[self.active_id]]


def _target_geometry(tokens, template_mean, temperature=0.1):
    """Return target-aware pooling and bounded spatial response statistics."""
    tokens_fp32 = tokens.float()
    template_fp32 = template_mean.float()
    similarity = torch.einsum(
        "bnc,bc->bn",
        F.normalize(tokens_fp32, dim=-1),
        F.normalize(template_fp32, dim=-1),
    )
    attention = (similarity / temperature).softmax(dim=1)
    pooled = torch.einsum("bn,bnc->bc", attention, tokens_fp32)

    token_count = tokens.shape[1]
    side = int(round(token_count ** 0.5))
    if side * side == token_count:
        axis = torch.linspace(-1.0, 1.0, side, device=tokens.device)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        x_coord = grid_x.reshape(1, -1)
        y_coord = grid_y.reshape(1, -1)
    else:
        x_coord = torch.linspace(
            -1.0, 1.0, token_count, device=tokens.device).reshape(1, -1)
        y_coord = torch.zeros_like(x_coord)

    center_x = (attention * x_coord).sum(dim=1)
    center_y = (attention * y_coord).sum(dim=1)
    spread = (attention * (
        (x_coord - center_x.unsqueeze(1)).square()
        + (y_coord - center_y.unsqueeze(1)).square())).sum(dim=1).sqrt()
    entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=1)
    concentration = 1.0 - entropy / max(math.log(token_count), 1e-6)
    return {
        "similarity": similarity,
        "attention": attention,
        "pooled": pooled,
        "center_x": center_x,
        "center_y": center_y,
        "spread": spread,
        "concentration": concentration,
    }


def _router_features(rgb_tokens, event_tokens, context=None, motion=None):
    """Summarize search tokens while retaining target-template context."""
    rgb_mean = rgb_tokens.mean(dim=1)
    event_mean = event_tokens.mean(dim=1)
    disagreement = (rgb_tokens - event_tokens).abs()
    diff_mean = disagreement.mean(dim=1)
    diff_std = disagreement.std(dim=1, unbiased=False)

    template_tokens = context.get("template_tokens") if context else None
    if (torch.is_tensor(template_tokens)
            and template_tokens.dim() == 3
            and template_tokens.shape[0] == rgb_tokens.shape[0]
            and template_tokens.shape[-1] == rgb_tokens.shape[-1]):
        template_mean = template_tokens.mean(dim=1)
        rgb_target = _target_geometry(rgb_tokens, template_mean)
        event_target = _target_geometry(event_tokens, template_mean)
        target_delta = 0.5 * (
            (rgb_target["pooled"] - template_mean.float()).abs()
            + (event_target["pooled"] - template_mean.float()).abs())
        diff_mean = 0.5 * (diff_mean + target_delta.to(diff_mean.dtype))

        mean_center_x = 0.5 * (
            rgb_target["center_x"] + event_target["center_x"])
        mean_center_y = 0.5 * (
            rgb_target["center_y"] + event_target["center_y"])
        cross_modal_offset = (
            (rgb_target["center_x"] - event_target["center_x"]).square()
            + (rgb_target["center_y"] - event_target["center_y"]).square()
        ).sqrt()
        cues = torch.stack([
            rgb_target["center_x"],
            rgb_target["center_y"],
            event_target["center_x"],
            event_target["center_y"],
            (mean_center_x.square() + mean_center_y.square()).sqrt(),
            0.5 * (rgb_target["spread"] + event_target["spread"]),
            0.5 * (rgb_target["concentration"] + event_target["concentration"]),
            0.5 * (
                rgb_target["similarity"].max(dim=1).values
                + event_target["similarity"].max(dim=1).values),
            cross_modal_offset,
            (rgb_target["similarity"] - event_target["similarity"])
            .abs().mean(dim=1),
        ], dim=1).to(diff_std.dtype)

        cue_count = min(cues.shape[1], diff_std.shape[1])
        diff_std = diff_std.clone()
        diff_std[:, :cue_count] = cues[:, :cue_count]

    if motion is not None:
        motion = torch.as_tensor(
            motion, device=diff_std.device, dtype=diff_std.dtype)
        if motion.ndim == 1:
            motion = motion.unsqueeze(0)
        if motion.ndim != 2 or motion.shape[0] != rgb_tokens.shape[0]:
            raise ValueError("motion cues must have shape (B, K)")
        motion_start = 10
        if motion_start + motion.shape[1] > diff_std.shape[1]:
            raise ValueError("router embedding is too small for motion cues")
        diff_std = diff_std.clone()
        diff_std[:, motion_start:motion_start + motion.shape[1]] = motion

    return torch.cat([rgb_mean, event_mean, diff_mean, diff_std], dim=-1)


class ExpertRouter(nn.Module):
    def __init__(self, embed_dim, expert_names, hidden_dim=128):
        super().__init__()
        if not expert_names:
            raise ValueError("expert_names must not be empty")
        self.expert_names = list(expert_names)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(self.expert_names)),
        )

    def forward(self, rgb_tokens, event_tokens, context=None, motion=None):
        features = _router_features(
            rgb_tokens, event_tokens, context=context, motion=motion)
        logits = self.classifier(features)
        probabilities = logits.softmax(dim=-1)
        confidence, selected_ids = probabilities.max(dim=-1)
        selected_experts = [self.expert_names[idx] for idx in selected_ids.detach().cpu().tolist()]
        return {
            "logits": logits,
            "probabilities": probabilities,
            "confidence": confidence,
            "selected_ids": selected_ids,
            "selected_experts": selected_experts,
        }


class SparseExpertRouter(ExpertRouter):
    """Feature-only router whose classes are sparse expert combinations."""

    def __init__(self, embed_dim, route_options, hidden_dim=128):
        routes = tuple(tuple(route) for route in route_options)
        if not routes or len(set(routes)) != len(routes):
            raise ValueError("route_options must be non-empty and unique")
        if any(not route or len(set(route)) != len(route) for route in routes):
            raise ValueError("each route must contain unique expert names")
        super().__init__(
            embed_dim=embed_dim,
            expert_names=range(len(routes)),
            hidden_dim=hidden_dim,
        )
        self.route_options = routes

    def forward(self, rgb_tokens, event_tokens, context=None, motion=None):
        output = super().forward(
            rgb_tokens, event_tokens, context=context, motion=motion)
        output.pop("selected_experts")
        output["selected_routes"] = [
            self.route_options[index]
            for index in output["selected_ids"].detach().cpu().tolist()
        ]
        return output


class PhysicsExpertRouter(nn.Module):
    """Physics-guided router driven by the unified event physical belief.

    Args:
        embed_dim: backbone feature dim.
        expert_names: list of expert names (order is the class order).
        hidden_dim: MLP hidden width.
        phys_dim: dimension of raw EPSM statistics (legacy/ablation path only).
        belief_dim: dimension of the shared belief embedding (primary path).
    """

    def __init__(self, embed_dim, expert_names, hidden_dim=128, phys_dim=8,
                 belief_dim=64):
        super().__init__()
        if not expert_names:
            raise ValueError("expert_names must not be empty")
        self.expert_names = list(expert_names)
        self.phys_dim = phys_dim
        self.belief_dim = belief_dim
        # Feature branch: same 4 statistics as ExpertRouter.
        self.feat_classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Physical branch (PRIMARY): project the shared belief embedding.
        self.belief_classifier = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Physical branch (LEGACY/ABLATION): project raw EPSM statistics. Used
        # only when belief is not provided (USE_SHARED_BELIEF=False).
        self.phys_classifier = nn.Sequential(
            nn.Linear(phys_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Fused head produces the routing logits. Input dim depends on which
        # physical branch is active; we keep both branches' output at hidden_dim
        # so the head is shared (feat_emb + phys_emb).
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(self.expert_names)),
        )

    def forward(self, rgb_tokens, event_tokens, physics=None, belief=None,
                context=None, motion=None):
        feat = _router_features(
            rgb_tokens, event_tokens, context=context, motion=motion)
        feat_emb = self.feat_classifier(feat)  # (B, hidden)

        B = feat_emb.shape[0]
        # PRIMARY: consume the shared belief embedding.
        if belief is not None:
            phys_emb = self.belief_classifier(belief)            # (B, hidden)
        elif physics is not None:
            phys = physics
            if phys.dim() == 1:
                phys = phys.unsqueeze(0)
            phys_emb = self.phys_classifier(phys)                # (B, hidden)
        else:
            # No physical input at all -> zero embedding (degrades to feat-only).
            phys_emb = torch.zeros(B, self.belief_dim, device=feat_emb.device)
            phys_emb = self.belief_classifier(phys_emb)

        fused = torch.cat([feat_emb, phys_emb], dim=-1)
        logits = self.head(fused)
        probabilities = logits.softmax(dim=-1)
        confidence, selected_ids = probabilities.max(dim=-1)
        selected_experts = [self.expert_names[idx] for idx in selected_ids.detach().cpu().tolist()]
        return {
            "logits": logits,
            "probabilities": probabilities,
            "confidence": confidence,
            "selected_ids": selected_ids,
            "selected_experts": selected_experts,
        }


class SparsePhysicsExpertRouter(PhysicsExpertRouter):
    """Physics router whose classes are sparse expert combinations."""

    def __init__(self, embed_dim, route_options, hidden_dim=128, phys_dim=8,
                 belief_dim=64):
        routes = tuple(tuple(route) for route in route_options)
        if not routes or len(set(routes)) != len(routes):
            raise ValueError("route_options must be non-empty and unique")
        if any(not route or len(set(route)) != len(route) for route in routes):
            raise ValueError("each route must contain unique expert names")
        super().__init__(
            embed_dim=embed_dim,
            expert_names=range(len(routes)),
            hidden_dim=hidden_dim,
            phys_dim=phys_dim,
            belief_dim=belief_dim,
        )
        self.route_options = routes

    def forward(self, rgb_tokens, event_tokens, physics=None, belief=None,
                context=None, motion=None):
        output = super().forward(
            rgb_tokens,
            event_tokens,
            physics=physics,
            belief=belief,
            context=context,
            motion=motion,
        )
        output.pop("selected_experts")
        output["selected_routes"] = [
            self.route_options[index]
            for index in output["selected_ids"].detach().cpu().tolist()
        ]
        return output


def iou_soft_routing_target(iou_per_route, temperature=1.0, eps=1e-6):
    """Build a soft routing supervision target from per-route IoU.

    Softmax over IoU / temperature as the target (gentle when T=1, recovers
    hard argmax as T->0). Teaches the router "who actually works", raising the
    ceiling above rule-based pseudo-labels.
    """
    scaled = iou_per_route / max(temperature, eps)
    scaled = scaled - scaled.max(dim=-1, keepdim=True).values
    target = scaled.softmax(dim=-1)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    best_id = iou_per_route.argmax(dim=-1)
    return target, best_id


def routing_loss(logits, target, label_smoothing=0.0):
    """Cross-entropy between router logits and the IoU soft target."""
    if label_smoothing > 0:
        uniform = torch.full_like(target, 1.0 / target.size(-1))
        target = (1.0 - label_smoothing) * target + label_smoothing * uniform
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()
