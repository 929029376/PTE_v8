"""
Unified event-raster belief module.

FELT supplies frame-aligned three-channel polarity PNG rasters, not raw event
tuples or timestamps. This module therefore builds one shared `belief_embed`
from raster observables and frame-level EMA history. The router, absence
predictor, and memory-policy head consume that same embedding instead of each
reading unrelated hand-selected scalars.

Design (see PET-Track design doc, section II):

    b_t = MLP_fuse( [ raw_stats_t , hist_snapshot_t ] )

where:
  raw_stats_t  = 8 per-frame raster statistics (global/ROI active-pixel
                 occupancy, polarity balance, spatial-gradient variation,
                 temporal occupancy deltas, history spread/readiness).
  hist_snapshot_t = [ rho_roi_ema , rho_roi_std , count/MIN_HISTORY ]
                 — a compact view of the EMA history the module maintains. The
                 `count` term carries "history maturity" which raw_stats does
                 NOT encode; this is the genuinely extra temporal information
                 beyond a single frame.

Why EMA-state fusion (not a K-frame RNN):
  The training sampler emits only a single search event frame per sample
  (num_search=1) and guarantees templates are visible, so no K-frame temporal
  sequence is available at training time. An RNN trained on 1-3 sparse frames
  would mismatch the dense 16-frame inference history. Fusing the EMA state
  instead keeps both paths on the same causal contract: query the current
  raster against past EMA state, then commit it for the next frame. The paper
  must say "temporally cumulative raster belief via EMA state", not recurrent
  microsecond event modeling.

Claim boundary:
  Active-pixel occupancy can correlate with motion, visibility, and scene
  change, but it is not an event firing rate or exact velocity. Spatial-gradient
  variation is not optical-flow divergence. The background-subtracted prior
      H = Normalize( GaussBlur( max(E − κB, 0) ) )
  is a frame-level raster cue for newly active regions, not a reconstruction of
  raw-event trajectories.
"""
import torch
import torch.nn.functional as F
from torch import nn


def _batched_roi_mask(roi_xyxy, height, width):
    """Rasterize xyxy boxes without synchronizing device scalars to the host."""
    bounds = roi_xyxy.to(dtype=torch.long)
    x1 = bounds[:, 0].clamp_min(0)
    y1 = bounds[:, 1].clamp_min(0)
    x2 = torch.minimum(bounds[:, 2], torch.full_like(bounds[:, 2], width))
    y2 = torch.minimum(bounds[:, 3], torch.full_like(bounds[:, 3], height))

    x = torch.arange(width, device=roi_xyxy.device).view(1, 1, width)
    y = torch.arange(height, device=roi_xyxy.device).view(1, height, 1)
    valid = (x2 > x1) & (y2 > y1)
    inside = (
        (x >= x1[:, None, None])
        & (x < x2[:, None, None])
        & (y >= y1[:, None, None])
        & (y < y2[:, None, None])
    )
    fallback = (~valid)[:, None, None] & (x == 0) & (y == 0)
    return (inside & valid[:, None, None]) | fallback


class EventPhysicalBelief(nn.Module):
    """Unified event-raster belief module.

    Produces, per frame:
      * raw_stats   — (B, stat_dim) the 8 raster/history statistics.
      * belief      — (B, belief_dim) the shared physical belief embedding that
                      all downstream heads consume.
      * reappear_prior — (B,H,W) background-subtracted reappear heatmap, for the
                      redetection expert. Spatial prior; lives here because it is
                      derived from the same energy map / background field.
      * history_ready — bool, whether the EMA history is trustworthy.

    Args:
        mean/std: per-channel normalization used to de-normalize the event image.
        alpha: EMA decay for the rho_roi running history.
        min_history: minimum updates before z_rho/delta_rho are trusted.
        roi_format: "xyxy" or "xywh".
        event_threshold: palette-free energy distance for the active mask.
        belief_dim: dimensionality of the shared belief embedding.
    """

    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 alpha: float = 0.9,
                 min_history: int = 3,
                 roi_format: str = "xywh",
                 event_threshold: float = 0.10,
                 belief_dim: int = 64):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.alpha = alpha
        self.min_history = min_history
        self.event_threshold = event_threshold
        assert roi_format in ("xyxy", "xywh")
        self.roi_format = roi_format

        # Statistics dimension: [rho, rho_roi, polarity, spatial variation, delta_rho,
        # z_rho, rho_roi_std, ready_flag] = 8
        self.stat_dim = 8
        # History snapshot: [rho_roi_ema, rho_roi_std, count_ratio]
        self.hist_snapshot_dim = 3
        # Shared belief MLP: fuse raw stats + history snapshot -> belief_embed.
        self.belief_dim = belief_dim
        self.belief_mlp = nn.Sequential(
            nn.Linear(self.stat_dim + self.hist_snapshot_dim, belief_dim),
            nn.ReLU(inplace=True),
            nn.Linear(belief_dim, belief_dim),
        )

        self._reset_history()

    # ------------------------------------------------------------------ #
    #  History management (called by the inference state machine)         #
    # ------------------------------------------------------------------ #
    def _reset_history(self):
        self._hist = {
            "rho_roi": None,        # EMA of rho_roi
            "rho_roi_sq": None,     # EMA of rho_roi^2 (for running variance)
            "rho": None,            # EMA of global density
            "count": 0,             # number of updates
            "background": None,     # background event field B(x,y) [1,1,H,W]
            "frozen": False,        # if True, history is not updated
        }

    def reset(self):
        """Reset all running state. Call at the start of each sequence."""
        self._reset_history()

    def freeze_history(self, freeze: bool = True):
        """When frozen, EMA history is NOT updated (FROZEN state)."""
        self._hist["frozen"] = freeze

    def is_history_ready(self) -> bool:
        return self._hist["count"] >= self.min_history

    # ------------------------------------------------------------------ #
    #  Energy map (de-normalize, per-frame median reference)              #
    # ------------------------------------------------------------------ #
    def _denormalize(self, x):
        return (x * self.std.to(x.device) + self.mean.to(x.device)).clamp(0.0, 1.0)

    def _energy_map(self, x_norm):
        """Per-pixel event energy E(x,y) in [0,1], de-normalized first.

        Reference color = per-frame, per-sample median (palette-free).
        Energy = mean over channels of |color - median_color|.
        """
        x = self._denormalize(x_norm)                      # (B,3,H,W)
        flat = x.flatten(2)                                # (B,3,HW)
        median = flat.median(dim=2).values.unsqueeze(-1).unsqueeze(-1)  # (B,3,1,1)
        energy = (x - median).abs().mean(dim=1)            # (B,H,W)
        return energy

    def energy_map(self, x_norm):
        """Public accessor (tracker uses it to seed/update the background)."""
        return self._energy_map(x_norm)

    def _active_mask(self, energy):
        return energy > self.event_threshold               # (B,H,W)

    # ------------------------------------------------------------------ #
    #  ROI helpers                                                        #
    # ------------------------------------------------------------------ #
    def _to_xyxy(self, roi):
        if self.roi_format == "xywh":
            x, y, w, h = roi.unbind(-1)
            return torch.stack([x, y, x + w, y + h], dim=-1)
        return roi

    def _roi_mask(self, energy, roi_xyxy):
        B, H, W = energy.shape
        if roi_xyxy.shape != (B, 4):
            raise ValueError(f"roi_xyxy must have shape ({B}, 4)")
        return _batched_roi_mask(roi_xyxy, H, W)

    # ------------------------------------------------------------------ #
    #  Background event field (for the redetection reappear prior)        #
    # ------------------------------------------------------------------ #
    def init_background(self, energy):
        """Seed the background field B(x,y) when entering FROZEN state."""
        self._hist["background"] = energy.detach().mean(dim=0, keepdim=True).clone()
        self._bg_beta = 0.95

    def update_background(self, energy):
        """EMA-update the background field while in FROZEN state."""
        if self._hist["background"] is None:
            self.init_background(energy)
            return
        beta = getattr(self, "_bg_beta", 0.95)
        self._hist["background"] = (
            beta * self._hist["background"] + (1.0 - beta) * energy.detach().mean(dim=0, keepdim=True)
        )

    def reappear_prior(self, energy, kappa: float = 1.2, sigma: float = 2.0):
        """Background-subtracted reappear heatmap H(x,y) in [0,1].

        H = Normalize(GaussBlur(max(E - kappa*B, 0))).
        """
        B_field = self._hist["background"]
        if B_field is None:
            return torch.full_like(energy, float(energy.mean()),
                                   requires_grad=False).clamp(0, 1)
        fg = (energy - kappa * B_field.to(energy.device)).clamp(min=0.0).unsqueeze(1)
        fg = _gaussian_blur(fg, sigma)
        fg = fg.squeeze(1)  # (B,H,W)
        mn = fg.flatten(1).min(dim=1).values
        mx = fg.flatten(1).max(dim=1).values
        rng = (mx - mn).clamp_min(1e-6)
        H = ((fg - mn[:, None, None]) / rng[:, None, None])
        return H

    # ------------------------------------------------------------------ #
    #  Statistics (the legacy EPSM computation, unchanged)                #
    # ------------------------------------------------------------------ #
    def _compute_stats(self, event_norm, roi, update_history: bool):
        """Return (raw_stats (B,stat_dim), rho_roi (B,)). May update EMA."""
        energy = self._energy_map(event_norm)
        active = self._active_mask(energy)
        rho = active.float().flatten(1).mean(dim=1)        # (B,)

        if roi is not None:
            roi_xyxy = self._to_xyxy(roi)
            mask = self._roi_mask(energy, roi_xyxy)
            roi_area = mask.float().flatten(1).sum(dim=1).clamp_min(1.0)
            rho_roi = (active.float() * mask.float()).flatten(1).sum(dim=1) / roi_area
        else:
            rho_roi = torch.zeros_like(rho)

        # FELT encodes the two event polarities as red/blue pixels on a white
        # background. A median-reference sign would classify both colors as
        # negative deviations, so use signed red-blue chroma on active pixels.
        # The sign convention itself is not assumed to be physical; the
        # bounded balance simply preserves the two raster polarities.
        x = self._denormalize(event_norm)
        chroma = x[:, 0] - x[:, 2]
        positive = ((chroma > 0.0) & active).float().flatten(1).sum(dim=1)
        negative = ((chroma < 0.0) & active).float().flatten(1).sum(dim=1)
        polarity_balance = (positive - negative) / (
            positive + negative).clamp_min(1e-6)

        # Spatial raster variation; this is not optical-flow divergence.
        gx = energy[:, :, 1:] - energy[:, :, :-1]
        gy = energy[:, 1:, :] - energy[:, :-1, :]
        spatial_variation = (
            gx.flatten(1).var(dim=1, unbiased=False)
            + gy.flatten(1).var(dim=1, unbiased=False)
        )

        delta_rho, z_rho, rho_roi_std = self._compute_deltas(rho_roi, update_history)
        ready_flag = torch.full_like(rho, 1.0 if self.is_history_ready() else 0.0)

        stats = torch.stack([
            rho,               # 0
            rho_roi,           # 1
            polarity_balance,  # 2
            spatial_variation, # 3
            delta_rho,         # 4
            z_rho,             # 5
            rho_roi_std,       # 6
            ready_flag,        # 7
        ], dim=1)  # (B, stat_dim)
        return stats, rho_roi

    def _compute_deltas(self, rho_roi, update_history: bool):
        device = rho_roi.device
        prev = self._hist["rho_roi"]
        prev_sq = self._hist["rho_roi_sq"]
        count = self._hist["count"]

        if prev is None or count < self.min_history:
            delta = torch.zeros_like(rho_roi)
            z = torch.zeros_like(rho_roi)
            std = torch.full_like(rho_roi, 1e-3)
        else:
            prev_dev = prev.to(device)
            delta = rho_roi - prev_dev
            prev_sq_dev = prev_sq.to(device)
            var = (prev_sq_dev - prev_dev.pow(2)).clamp_min(1e-6)
            std = var.sqrt()
            z = delta / std.clamp_min(1e-3)

        if update_history and not self._hist["frozen"]:
            a = self.alpha
            cur = rho_roi.detach()
            cur_sq = cur.pow(2)
            if self._hist["rho_roi"] is None:
                self._hist["rho_roi"] = cur.clone()
                self._hist["rho_roi_sq"] = cur_sq.clone()
            else:
                hist = self._hist["rho_roi"].to(device=cur.device, dtype=cur.dtype)
                hist_sq = self._hist["rho_roi_sq"].to(device=cur_sq.device, dtype=cur_sq.dtype)
                self._hist["rho_roi"] = a * hist + (1 - a) * cur
                self._hist["rho_roi_sq"] = a * hist_sq + (1 - a) * cur_sq
            self._hist["count"] += 1

        return delta, z, std

    def _history_snapshot(self, rho_roi):
        """Compact view of the EMA history, fused into the belief.

        [rho_roi_ema, rho_roi_std, count/MIN_HISTORY]. Provides the temporal
        context (maturity + running spread) that raw_stats does not encode.
        When history is not ready, returns zeros so the belief degrades
        gracefully to a current-frame-only representation.
        """
        device = rho_roi.device
        prev = self._hist["rho_roi"]
        prev_sq = self._hist["rho_roi_sq"]
        count = self._hist["count"]
        if prev is None or count < self.min_history:
            zero = torch.zeros_like(rho_roi)
            snap = torch.stack([zero, torch.full_like(rho_roi, 1e-3),
                                torch.zeros_like(rho_roi)], dim=1)
            return snap
        prev_dev = prev.to(device)
        var = (prev_sq.to(device) - prev_dev.pow(2)).clamp_min(1e-6)
        std = var.sqrt()
        ratio = torch.full_like(rho_roi, float(count) / max(self.min_history, 1))
        ratio = ratio.clamp(0.0, 4.0)  # cap; >1 means mature history
        return torch.stack([prev_dev, std, ratio], dim=1)  # (B,3)

    # ------------------------------------------------------------------ #
    #  Main forward                                                       #
    # ------------------------------------------------------------------ #
    def forward(self, event_norm, roi=None, update_history: bool = True,
                compute_prior: bool = False):
        """Compute the unified event-raster belief for one frame.

        Args:
            event_norm: (B,3,H,W) normalized event image.
            roi: (B,4) last predicted box (xywh by default). If None, rho_roi
                 and delta_rho/z_rho are zeroed.
            update_history: if False (e.g. FROZEN state, or training where we
                 don't want to mutate the running state), do not update EMA.
            compute_prior: if True, also compute and return the reappear prior
                 (only meaningful at inference when a background field exists).

        Returns:
            dict with keys:
              raw_stats: (B, stat_dim)
              belief:    (B, belief_dim)  — the shared embedding
              reappear_prior: (B,H,W) or None
              history_ready: bool
        """
        raw_stats, rho_roi = self._compute_stats(event_norm, roi, update_history)
        snap = self._history_snapshot(rho_roi)
        fused = torch.cat([raw_stats, snap], dim=-1)        # (B, stat_dim+3)
        belief = self.belief_mlp(fused)                     # (B, belief_dim)

        prior = None
        if compute_prior:
            energy = self._energy_map(event_norm)
            prior = self.reappear_prior(energy)

        return {
            "raw_stats": raw_stats,
            "belief": belief,
            "reappear_prior": prior,
            "history_ready": self.is_history_ready(),
        }


def _gaussian_blur(x, sigma: float = 2.0):
    """Separable Gaussian blur on (B,1,H,W) tensor. Padded to keep size."""
    if sigma <= 0:
        return x
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    k1d = kernel.view(1, 1, -1, 1)
    pad = radius
    x = F.pad(x, (pad, pad, 0, 0), mode="replicate")
    x = F.conv2d(x, k1d)
    k1d = kernel.view(1, 1, 1, -1)
    x = F.pad(x, (0, 0, pad, pad), mode="replicate")
    x = F.conv2d(x, k1d)
    return x


def build_event_belief(cfg=None, **kwargs):
    """Factory. Reads normalization from cfg.DATA and belief config from
    cfg.MODEL.EVENT_BELIEF (with safe defaults)."""
    mean = kwargs.pop("mean", None)
    std = kwargs.pop("std", None)
    if cfg is not None and mean is None:
        try:
            mean = list(cfg.DATA.MEAN)
            std = list(cfg.DATA.STD)
        except Exception:
            mean = std = None
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)

    alpha = kwargs.pop("alpha", 0.9)
    min_history = kwargs.pop("min_history", 3)
    event_threshold = kwargs.pop("event_threshold", 0.10)
    roi_format = kwargs.pop("roi_format", "xywh")
    belief_dim = kwargs.pop("belief_dim", 64)

    if cfg is not None:
        epsm_cfg = getattr(cfg.MODEL, "EPSM", None)
        if epsm_cfg is not None:
            alpha = float(getattr(epsm_cfg, "ALPHA", alpha))
            min_history = int(getattr(epsm_cfg, "MIN_HISTORY", min_history))
            event_threshold = float(getattr(epsm_cfg, "EVENT_THRESHOLD", event_threshold))
            roi_format = getattr(epsm_cfg, "ROI_FORMAT", roi_format)
        eb_cfg = getattr(cfg.MODEL, "EVENT_BELIEF", None)
        if eb_cfg is not None:
            belief_dim = int(getattr(eb_cfg, "BELIEF_DIM", belief_dim))

    return EventPhysicalBelief(
        mean=mean, std=std, alpha=alpha, min_history=min_history,
        roi_format=roi_format, event_threshold=event_threshold,
        belief_dim=belief_dim, **kwargs)
