"""
Redetection expert branch (C3-subproblem 3: global re-localization).

When the state machine enters REDETECT, the target may be anywhere in the full
frame (it has drifted out of the local search window). The redetection expert:

  1. Runs a GLOBAL search over the full image (downsampled to a fixed grid),
     not a local 4-5x crop. This is the structural difference vs. the normal
     tracking head.
  2. Multiplies the expert's score map by the EPSM reappear prior H(x,y)
     (background-subtracted event heatmap) so the search is focused where NEW
     events appear, i.e. where the target likely reappeared.
  3. Matches against the CLEAN template T_clean (snapshotted at FROZEN entry,
     never polluted by occlusion-frame updates), not the dynamic template.

  conf = max( score_map * (1 + lambda_H * H) )
  box   = argmax location, decoded via the size/offset maps.

Training: supervised on absent->present transition frames (FELT absent.txt
provides the labels). The loss is a focal/classification loss on the score map
at the GT box center, plus an L1 on the regressed size/offset.

This module reuses the CenterPredictor backbone for the score/size/offset
heads so it inherits the trained localization capability, but wraps it with:
  - a global feature adapter (1x1 conv to project backbone features),
  - a prior-modulation gate that fuses H into the score map.
"""
import torch
import torch.nn.functional as F
from torch import nn

from lib.models.layers.head import CenterPredictor


class RedetectionExpert(nn.Module):
    """Global re-localization head with event-prior modulation.

    Args:
        inplanes: backbone feature channels (embed_dim, e.g. 768).
        channel: head internal channels (e.g. 256).
        feat_sz: spatial size of the redetect feature grid.
        stride: redetect stride (image_size / feat_sz). Smaller than the
            tracking stride so the global field of view is larger.
        lambda_H: weight of the event reappear prior H when modulating the
            score map. score <- score * (1 + lambda_H * H).
        use_prior_gate: if True, learn a per-pixel gate (sigmoid) instead of a
            hard multiplication, so the model can down-weight H when it is
            unreliable (e.g. background field not converged).
    """

    def __init__(self, inplanes=768, channel=256, feat_sz=20, stride=16,
                 lambda_H: float = 1.0, use_prior_gate: bool = True):
        super().__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.lambda_H = lambda_H
        self.use_prior_gate = use_prior_gate

        # Project backbone features to the head's channel space.
        self.adapter = nn.Sequential(
            nn.Conv2d(inplanes, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        # Reuse the standard localization head structure (score/size/offset).
        self.head = CenterPredictor(inplanes=channel, channel=channel,
                                    feat_sz=feat_sz, stride=stride)
        self.template_proj = nn.Linear(inplanes, channel)
        self.template_scale = nn.Parameter(torch.zeros(()))

        # Optional learned gate to weight the prior. Input: 2 channels
        # (score, prior) -> 1 channel gate in [0,1].
        if use_prior_gate:
            self.prior_gate = nn.Sequential(
                nn.Conv2d(2, channel // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, 1, kernel_size=1),
            )
            nn.init.zeros_(self.prior_gate[-1].weight)
            nn.init.zeros_(self.prior_gate[-1].bias)

    def forward(self, feat, prior_H=None, gt_score_map=None,
                template_tokens=None):
        """Run global redetection.

        Args:
            feat: (B, C, Hf, Wf) backbone feature map for the full-image crop.
                Hf, Wf need not equal feat_sz; we interpolate to feat_sz.
            prior_H: (B, Hf, Wf) or (B, feat_sz, feat_sz) event reappear prior
                from EPSM.reappear_prior(), values in [0,1]. None disables
                prior modulation (pure expert search).
            gt_score_map: optional GT for training-time score supervision.
            template_tokens: (B,N,C) clean RGB/event template tokens. When
                present, the head receives a learned target-similarity residual.

        Returns:
            dict with:
              score_map: (B,1,feat_sz,feat_sz) modulated score map.
              bbox: (B,4) cxcywh in [0,1] (image-normalized).
              size_map, offset_map: (B,2,feat_sz,feat_sz).
              conf: (B,) max score (= redetect confidence).
              raw_score: (B,1,feat_sz,feat_sz) before prior modulation.
        """
        x = self.adapter(feat)
        # Resize to the head's expected grid.
        if x.shape[-2:] != (self.feat_sz, self.feat_sz):
            x = F.interpolate(x, size=(self.feat_sz, self.feat_sz),
                              mode="bilinear", align_corners=False)

        template_similarity = None
        if template_tokens is not None:
            template = template_tokens.mean(dim=1)
            template = F.normalize(self.template_proj(template), dim=-1)
            feature_unit = F.normalize(x, dim=1)
            template_similarity = (
                feature_unit * template[:, :, None, None]).sum(dim=1, keepdim=True)
            residual = template_similarity * template[:, :, None, None]
            x = x + torch.tanh(self.template_scale) * residual

        raw_score, bbox, size_map, offset_map = self.head(x, gt_score_map=gt_score_map)
        # raw_score: (B,1,f,f)

        score = raw_score
        if prior_H is not None:
            H = prior_H
            if H.dim() == 3:
                H = H.unsqueeze(1)  # (B,1,f,f)
            if H.shape[-2:] != (self.feat_sz, self.feat_sz):
                H = F.interpolate(H, size=(self.feat_sz, self.feat_sz),
                                  mode="bilinear", align_corners=False)
            if self.use_prior_gate:
                gate = torch.sigmoid(self.prior_gate(
                    torch.cat([raw_score, H], dim=1)))
                score = raw_score * (1.0 + self.lambda_H * gate * H)
            else:
                score = raw_score * (1.0 + self.lambda_H * H)

        conf = score.flatten(1).max(dim=1)[0]  # (B,)
        out = {
            "score_map": score,
            "raw_score": raw_score,
            "bbox": bbox,
            "size_map": size_map,
            "offset_map": offset_map,
            "conf": conf,
        }
        if template_similarity is not None:
            out["template_similarity"] = template_similarity
        return out

    def decode_box(self, score_map, size_map, offset_map):
        """Decode a box from (possibly modulated) score map. Reuses the head."""
        return self.head.cal_bbox(score_map, size_map, offset_map,
                                  return_score=True)


def build_redetection_expert(cfg, embed_dim):
    """Factory. Reads head config from cfg.MODEL.HEAD and redetect-specific
    settings from cfg.MODEL.REDETECT (with safe defaults)."""
    head_cfg = cfg.MODEL.HEAD
    channel = int(getattr(head_cfg, "NUM_CHANNELS", 256))
    # Redetect uses a smaller stride / larger field. Default: 32 px stride
    # over a 384px full-image crop -> feat_sz=12.
    redetect_cfg = getattr(cfg.MODEL, "REDETECT", None)
    feat_sz = int(getattr(redetect_cfg, "FEAT_SZ", 12)) if redetect_cfg else 12
    stride = int(getattr(redetect_cfg, "STRIDE", 32)) if redetect_cfg else 32
    lambda_H = float(getattr(redetect_cfg, "LAMBDA_H", 1.0)) if redetect_cfg else 1.0
    use_prior_gate = bool(getattr(redetect_cfg, "USE_PRIOR_GATE", True)) if redetect_cfg else True
    return RedetectionExpert(inplanes=embed_dim, channel=channel, feat_sz=feat_sz,
                             stride=stride, lambda_H=lambda_H,
                             use_prior_gate=use_prior_gate)
