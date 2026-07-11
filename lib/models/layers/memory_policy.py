"""
Memory Policy Head — learned gates that replace the hand-tuned thresholds of
the occlusion state machine, while the rule-based counters/timeouts (the
safety skeleton) are retained.

This is the "learned policy with safety skeleton" compromise (option 1-lite):
we do NOT replace the whole state machine with a fully-differentiable
Gumbel-softmax controller (that path was set aside as too risky for the AAAI
timeline — it requires a recurrent state policy and stable joint training with
the experts). Instead, the two most hand-crafted decisions are turned into
learned gates driven by the shared event physical belief:

  * freeze_prob    = σ(MLP([b_t, absence_prob]))          — refines the absence
                     signal. The inference state machine applies temporal
                     hysteresis to this gate before TRACKING→FROZEN.
  * redetect_prob  = σ(MLP([b_t, frozen_age, absence_prob])) — the genuinely
                     NEW decision: WHEN to attempt global re-localization. This
                     is the one gate that sees `frozen_age`, a variable the
                     absence predictor cannot observe, so it is NOT a redundant
                     copy of absence. Drives FROZEN→REDETECT instead of the
                     hard z_rho>theta_z threshold.

The update decision is folded into freeze (update = NOT freeze), since freeze
has a natural supervision signal (absent.txt) and update does not. We do NOT
fabricate a separate update head — that would be a head with no real labels.

Supervision:
  * freeze_prob  : BCE on (1 - present_mask)   — absent→freeze=1. Real labels.
  * redetect_prob: BCE on the official previous-absent/current-present
    transition. The sampler exports both presence states, so this target does
    not require challenge labels or a heuristic reappearance category.
"""
import torch
from torch import nn


class MemoryPolicyHead(nn.Module):
    """Learned memory-update / redetect-trigger policy.

    Args:
        belief_dim: dimension of the shared event physical belief embedding.
        hidden_dim: MLP hidden width.
    """

    def __init__(self, belief_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        # Inputs: belief (belief_dim) + frozen_age (1) + absence_prob (1)
        in_dim = belief_dim + 2
        self.freeze_head = nn.Sequential(
            nn.Linear(belief_dim + 1, hidden_dim),   # belief + absence_prob
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.redetect_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),            # belief + frozen_age + absence_prob
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-init the last layers so the gates start neutral (0.5) and learn
        # rather than guessing "freeze everything" or "redetect constantly".
        for head in (self.freeze_head, self.redetect_head):
            nn.init.zeros_(head[-1].bias)
            nn.init.normal_(head[-1].weight, std=1e-3)

    def forward(self, belief, frozen_age, absence_prob):
        """Compute the two memory-policy gates.

        Args:
            belief: (B, belief_dim) shared event physical belief embedding.
            frozen_age: (B,) or (B,1) frames spent in FROZEN, normalized by a
                reference (caller should pass frozen_age / T_max so it is ~[0,1]).
            absence_prob: (B,) or (B,1) absence probability from the
                per-frame AbsencePredictor. The state machine, not this head,
                applies the multi-frame hysteresis window.

        Returns:
            dict with freeze_prob (B,) and redetect_prob (B,), both in [0,1].
        """
        if absence_prob.dim() == 1:
            absence_prob = absence_prob.unsqueeze(-1)   # (B,1)
        if frozen_age.dim() == 1:
            frozen_age = frozen_age.unsqueeze(-1)       # (B,1)

        freeze_in = torch.cat([belief, absence_prob], dim=-1)
        freeze_prob = torch.sigmoid(self.freeze_head(freeze_in).squeeze(-1))  # (B,)

        redetect_in = torch.cat([belief, frozen_age, absence_prob], dim=-1)
        redetect_prob = torch.sigmoid(self.redetect_head(redetect_in).squeeze(-1))  # (B,)
        return {"freeze_prob": freeze_prob, "redetect_prob": redetect_prob}


def build_memory_policy(cfg=None, belief_dim: int = 64, **kwargs):
    """Factory. Reads hidden_dim from cfg.MODEL.MEMORY_POLICY if present."""
    hidden_dim = kwargs.pop("hidden_dim", 64)
    if cfg is not None:
        mp_cfg = getattr(cfg.MODEL, "MEMORY_POLICY", None)
        if mp_cfg is not None:
            hidden_dim = int(getattr(mp_cfg, "HIDDEN_DIM", hidden_dim))
    return MemoryPolicyHead(belief_dim=belief_dim, hidden_dim=hidden_dim, **kwargs)
