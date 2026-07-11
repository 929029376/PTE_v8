"""
Absence Predictor.

Decides whether the target is currently absent (fully occluded / out of frame).
This is C3-subproblem (1): detecting target loss reliably, so the memory policy
can freeze the template and trigger global redetection before the tracker
drifts permanently.

Input in the full model (the unified representation):
    a = sigmoid( MLP([ b_t , s , sim_zx ]) )
where:
    b_t    = event physical belief embedding (shared, from EventPhysicalBelief)
    s      = max(score_map)            | tracker cue, fooled by distractors
    sim_zx = cosine(template, search)  | tracker cue, fooled by illumination

The belief b_t carries raster occupancy evolution and history maturity that
neither s nor sim_zx encodes. The `USE_SHARED_BELIEF=False` ablation replaces
it with zero-padded raw raster statistics while preserving the predictor shape.
The two tracker cues stay separate because they come from the backbone, not the
event raster statistics.

`b_t` and the two tracker cues are orthogonal: each is robust to a failure
mode that fools the others. Their agreement is what makes absence detection
reliable. Supervised DIRECTLY by FELT's absent.txt (real labels).
"""
import torch
from torch import nn


class AbsencePredictor(nn.Module):
    """Fuses the shared physical belief with two tracker cues into an absence
    probability.

    Args:
        belief_dim: dimension of the shared event physical belief embedding.
        hidden_dim: MLP hidden width.
    """

    def __init__(self, belief_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.belief_dim = belief_dim
        # Inputs: belief (belief_dim) + score_peak (1) + sim_zx (1)
        in_dim = belief_dim + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Zero-init the last layer so the predictor starts neutral (a ~= 0.5).
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=1e-3)

    def forward(self, belief, score_peak, sim_zx):
        """
        Args:
            belief: (B, belief_dim) shared event physical belief embedding.
            score_peak: (B,) or (B,1) max of the score map, in [0,1].
            sim_zx: (B,) or (B,1) template-search cosine similarity, in [-1,1].
        Returns:
            a: (B,) absence probability in [0,1].
        """
        if belief.dim() == 1:
            belief = belief.unsqueeze(0)
        sp = score_peak.view(-1, 1) if score_peak.dim() == 1 else score_peak.view(-1, 1)
        sx = sim_zx.view(-1, 1) if sim_zx.dim() == 1 else sim_zx.view(-1, 1)
        cues = torch.cat([belief, sp, sx], dim=-1)
        logit = self.net(cues).squeeze(-1)  # (B,)
        return torch.sigmoid(logit)

    def logits(self, belief, score_peak, sim_zx):
        """Raw logits (for BCEWithLogitsLoss with pos_weight)."""
        if belief.dim() == 1:
            belief = belief.unsqueeze(0)
        sp = score_peak.view(-1, 1)
        sx = sim_zx.view(-1, 1)
        cues = torch.cat([belief, sp, sx], dim=-1)
        return self.net(cues).squeeze(-1)


def compute_sim_zx(template_feat, search_feat, eps: float = 1e-6):
    """Cosine similarity between template and search feature embeddings."""
    if template_feat.dim() == 3:
        template_feat = template_feat.mean(dim=1)
    if search_feat.dim() == 3:
        search_feat = search_feat.mean(dim=1)
    t = template_feat / (template_feat.norm(dim=-1, keepdim=True) + eps)
    s = search_feat / (search_feat.norm(dim=-1, keepdim=True) + eps)
    return (t * s).sum(dim=-1)


def build_absence_predictor(cfg=None, belief_dim: int = 64, **kwargs):
    """Factory. belief_dim from cfg.MODEL.EVENT_BELIEF if present."""
    if cfg is not None:
        eb_cfg = getattr(cfg.MODEL, "EVENT_BELIEF", None)
        if eb_cfg is not None:
            belief_dim = int(getattr(eb_cfg, "BELIEF_DIM", belief_dim))
        abs_cfg = getattr(cfg.MODEL, "ABSENCE", None)
        if abs_cfg is not None:
            belief_dim = int(getattr(abs_cfg, "BELIEF_DIM", belief_dim))
            kwargs.setdefault("hidden_dim", int(getattr(abs_cfg, "HIDDEN_DIM", 64)))
    return AbsencePredictor(belief_dim=belief_dim, **kwargs)
