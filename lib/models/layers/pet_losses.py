"""
PET-Track training losses (C1/C3 supervision).

Three losses that the baseline tracker does not have:

  L_route    : teaches the sparse router to select the singleton/pair route
               with the best counterfactual tracking utility. It uses official
               boxes, not challenge-category pseudo-labels.

  L_absence  : supervises the AbsencePredictor with FELT's absent.txt ground
               truth. This is REAL supervision (not pseudo-labels), and is the
               key enabler for reliable occlusion detection.

  L_redetect : trains the RedetectionExpert on absent->present transition
               frames, so the global re-localization head learns to find the
               target after occlusion. Supervised by the GT box at the
               reappear frame (derived from absent.txt).

All three are gathered in `PetTrackLoss` so the actor can call one method.
"""
import torch
import torch.nn.functional as F
from torch import nn

from lib.models.layers.expert_router import routing_loss


def _safe_iou(pred_xyxy, gt_xyxy, eps=1e-6):
    """1D-batch IoU for (B,4) xyxy boxes. Falls back to the project's box_iou
    when available; otherwise computes inline."""
    pred = pred_xyxy
    gt = gt_xyxy
    lt = torch.max(pred[:, :2], gt[:, :2])
    rb = torch.min(pred[:, 2:], gt[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    ap = (pred[:, 2:] - pred[:, :2]).clamp(min=0)
    ag = (gt[:, 2:] - gt[:, :2]).clamp(min=0)
    area_p = ap[:, 0] * ap[:, 1]
    area_g = ag[:, 0] * ag[:, 1]
    union = area_p + area_g - inter
    return inter / union.clamp_min(eps)


class PetTrackLoss(nn.Module):
    """Aggregates the three PET-Track losses.

    Args:
        route_weight, absence_weight, redetect_weight: loss weights.
        route_temperature: temperature for the IoU soft target.
        absence_pos_weight: upweight the (rare) absent class so occlusion
            recall is prioritized (matches the AbsencePredictor test setup).
        label_smoothing: smoothing for the routing CE.
    """

    def __init__(self,
                 route_weight: float = 0.2,
                 absence_weight: float = 1.0,
                 redetect_weight: float = 1.0,
                 route_temperature: float = 0.5,
                 absence_pos_weight: float = 10.0,
                 label_smoothing: float = 0.1,
                 freeze_weight: float = 1.0,
                 redetect_gate_weight: float = 1.0,
                 route_regret_weight: float = 1.0,
                 route_oracle_ce_weight: float = 1.0,
                 route_pair_penalty: float = 0.02):
        super().__init__()
        self.route_weight = route_weight
        self.absence_weight = absence_weight
        self.redetect_weight = redetect_weight
        self.route_temperature = route_temperature
        self.absence_pos_weight = absence_pos_weight
        self.label_smoothing = label_smoothing
        self.freeze_weight = freeze_weight
        self.redetect_gate_weight = redetect_gate_weight
        self.route_regret_weight = route_regret_weight
        self.route_oracle_ce_weight = route_oracle_ce_weight
        self.route_pair_penalty = route_pair_penalty

    # ------------------------------------------------------------------ #
    def routing_loss(self, router_logits, per_route_pred_xyxy, gt_xyxy,
                     route_sizes, present_mask=None):
        """Supervise symmetric routes by counterfactual tracking utility."""
        if per_route_pred_xyxy.ndim != 3 or per_route_pred_xyxy.shape[-1] != 4:
            raise ValueError("per_route_pred_xyxy must have shape (B, R, 4)")
        batch_size, route_count, _ = per_route_pred_xyxy.shape
        if router_logits.shape != (batch_size, route_count):
            raise ValueError("router_logits must match the counterfactual routes")
        if gt_xyxy.shape != (batch_size, 4):
            raise ValueError("gt_xyxy must have shape (B, 4)")

        route_sizes = torch.as_tensor(
            route_sizes, device=router_logits.device, dtype=torch.float32)
        if route_sizes.shape != (route_count,):
            raise ValueError("route_sizes must have shape (R,)")
        if (route_sizes < 1).any():
            raise ValueError("route_sizes must be positive")

        if present_mask is None:
            present_mask = torch.ones(
                batch_size, dtype=torch.bool, device=router_logits.device)
        else:
            present_mask = torch.as_tensor(
                present_mask, device=router_logits.device,
                dtype=torch.bool).reshape(-1)
            if present_mask.shape != (batch_size,):
                raise ValueError("present_mask must have shape (B,)")

        valid_count = int(present_mask.sum().item())
        if valid_count == 0:
            zero = router_logits.float().sum() * 0.0
            stats = {
                "Loss/route": 0.0,
                "Loss/route_ce": 0.0,
                "Route/valid_count": 0,
                "Route/acc": 0.0,
                "Route/expected_regret": 0.0,
            }
            for route_id in range(route_count):
                stats[f"Route/target_rate_{route_id}"] = 0.0
                stats[f"Route/pred_rate_{route_id}"] = 0.0
                stats[f"Route/recall_{route_id}"] = 0.0
            return zero, stats

        logits = router_logits[present_mask].float()
        boxes = per_route_pred_xyxy[present_mask]
        gt = gt_xyxy[present_mask]
        with torch.no_grad():
            iou = torch.stack([
                _safe_iou(boxes[:, route_id], gt)
                for route_id in range(route_count)
            ], dim=1)
            complexity = self.route_pair_penalty * (route_sizes - 1.0)
            utility = iou - complexity.unsqueeze(0)
            target = F.softmax(
                utility / max(self.route_temperature, 1e-6), dim=-1)
            best_id = utility.argmax(dim=-1)

        route_ce = routing_loss(
            logits, target, label_smoothing=self.label_smoothing)
        probabilities = logits.softmax(dim=-1)
        best_utility = utility.max(dim=-1, keepdim=True).values
        regret = (best_utility - utility).clamp_min(0.0)
        expected_regret = (probabilities * regret).sum(dim=-1).mean()
        loss = (self.route_oracle_ce_weight * route_ce
                + self.route_regret_weight * expected_regret)

        with torch.no_grad():
            pred_id = logits.argmax(dim=-1)
            accuracy = (pred_id == best_id).float().mean().item()
            confidence = probabilities.max(dim=-1).values.mean().item()
            best_prob = probabilities.gather(
                1, best_id.unsqueeze(1)).mean().item()
            best_iou = iou.gather(1, best_id.unsqueeze(1)).mean().item()
            mean_best_utility = utility.gather(
                1, best_id.unsqueeze(1)).mean().item()
            target_safe = target.clamp_min(1e-8)
            target_entropy = -(
                target_safe * target_safe.log()).sum(dim=-1).mean().item()
            target_peak = target.max(dim=-1).values.mean().item()
            pair_mask = route_sizes > 1
            pair_target_rate = pair_mask[best_id].float().mean().item()
            pair_pred_rate = pair_mask[pred_id].float().mean().item()

        stats = {
            "Loss/route": loss.item(),
            "Loss/route_ce": route_ce.item(),
            "Route/valid_count": valid_count,
            "Route/acc": accuracy,
            "Route/confidence": confidence,
            "Route/best_prob": best_prob,
            "Route/target_entropy": target_entropy,
            "Route/target_peak": target_peak,
            "Route/best_iou": best_iou,
            "Route/best_utility": mean_best_utility,
            "Route/pair_target_rate": pair_target_rate,
            "Route/pair_pred_rate": pair_pred_rate,
            "Route/expected_regret": expected_regret.item(),
        }
        recalls = []
        for route_id in range(route_count):
            target_mask = best_id == route_id
            recall = (
                (pred_id[target_mask] == route_id).float().mean().item()
                if target_mask.any() else 0.0
            )
            if target_mask.any():
                recalls.append(torch.tensor(recall))
            stats[f"Route/target_rate_{route_id}"] = (
                target_mask.float().mean().item())
            stats[f"Route/pred_rate_{route_id}"] = (
                (pred_id == route_id).float().mean().item())
            stats[f"Route/recall_{route_id}"] = recall
        stats["Route/macro_recall"] = (
            torch.stack(recalls).mean().item() if recalls else 0.0)
        return loss, stats

    # ------------------------------------------------------------------ #
    def absence_loss(self, absence_prob, absent_gt):
        """L_absence: BCE with pos_weight to handle class imbalance.

        Args:
            absence_prob: (B,) predicted absence probability in [0,1].
            absent_gt: (B,) {0,1} ground-truth absent label (from absent.txt).
        Returns:
            loss, stats dict with recall on the absent class.
        """
        prob = absence_prob.float()
        gt = absent_gt.to(device=prob.device).float().clamp(0, 1)
        # BCEWithLogitsLoss wants logits; we stored probabilities, so invert
        # the sigmoid in a numerically stable way.
        eps = 1e-6
        prob = prob.clamp(eps, 1 - eps)
        logit = torch.logit(prob)
        pos_weight = torch.tensor([self.absence_pos_weight],
                                  device=logit.device, dtype=logit.dtype)
        loss = F.binary_cross_entropy_with_logits(logit, gt, pos_weight=pos_weight)
        with torch.no_grad():
            pred = (prob > 0.5).float()
            absent_mask = gt == 1
            recall = (pred[absent_mask] == 1).float().mean().item() \
                if absent_mask.any() else 1.0
            acc = (pred == gt).float().mean().item()
        return loss, {"Loss/absence": loss.item(),
                      "Absence/recall": recall,
                      "Absence/acc": acc}

    # ------------------------------------------------------------------ #
    def redetect_loss(self, redetect_score_map, redetect_bbox, gt_xyxy,
                      feat_sz, stride, gt_score_map=None):
        """L_redetect: focal + L1 on the redetection output at the reappear
        frame. The score peak should localize the GT center, and the regressed
        box should match the GT box.

        Args:
            redetect_score_map: (B,1,f,f) modulated score from RedetectionExpert.
            redetect_bbox: (B,4) cxcywh in [0,1].
            gt_xyxy: (B,4) ground-truth box in image-normalized [0,1] coords.
            feat_sz: score grid size f.
            stride: redetect stride (image_size / feat_sz), for mapping GT to
                the score grid center.
            gt_score_map: optional precomputed GT heatmap; if None, we build a
                Gaussian at the GT center.
        Returns:
            loss, stats.
        """
        redetect_score_map = redetect_score_map.float()
        redetect_bbox = redetect_bbox.float()
        gt_xyxy = gt_xyxy.float()
        if gt_score_map is not None:
            gt_score_map = gt_score_map.float()

        B = redetect_score_map.shape[0]
        f = redetect_score_map.shape[-1]
        if gt_score_map is None:
            gt_score_map = torch.zeros(B, 1, f, f, device=redetect_score_map.device)
            for b in range(B):
                cx = (gt_xyxy[b, 0] + gt_xyxy[b, 2]) * 0.5 * f
                cy = (gt_xyxy[b, 1] + gt_xyxy[b, 3]) * 0.5 * f
                cx_i = int(cx.clamp(0, f - 1))
                cy_i = int(cy.clamp(0, f - 1))
                gt_score_map[b, 0, cy_i, cx_i] = 1.0
                # small Gaussian spread
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        yy, xx = cy_i + dy, cx_i + dx
                        if 0 <= yy < f and 0 <= xx < f:
                            gt_score_map[b, 0, yy, xx] = max(
                                gt_score_map[b, 0, yy, xx].item(),
                                0.5 ** (dx * dx + dy * dy))
        # Focal-style classification on the score map.
        score = redetect_score_map.clamp(1e-4, 1 - 1e-4)
        pos = gt_score_map
        neg = 1 - pos
        cls_loss = -(pos * torch.log(score) + neg * torch.log(1 - score)).mean()

        # L1 on box: convert GT xyxy (normalized) -> cxcywh (normalized).
        gt_cxcywh = torch.stack([
            (gt_xyxy[:, 0] + gt_xyxy[:, 2]) * 0.5,
            (gt_xyxy[:, 1] + gt_xyxy[:, 3]) * 0.5,
            gt_xyxy[:, 2] - gt_xyxy[:, 0],
            gt_xyxy[:, 3] - gt_xyxy[:, 1],
        ], dim=-1)
        l1 = F.l1_loss(redetect_bbox, gt_cxcywh)
        loss = cls_loss + l1
        return loss, {"Loss/redetect": loss.item(),
                      "Loss/redetect_cls": cls_loss.item(),
                      "Loss/redetect_l1": l1.item()}

    # ------------------------------------------------------------------ #
    def memory_policy_loss(self, freeze_prob, redetect_prob, present_mask, reappear_mask=None):
        """L_freeze + L_redetect_gate for the MemoryPolicyHead.

        L_freeze        : BCE(freeze_prob,  ~present)  — absent -> freeze=1.
        L_redetect_gate : BCE(redetect_prob,  present) — present -> redetect=1.

        When the sampler provides previous/current presence, redetect_gate is
        supervised on true absent->present frames. Otherwise it falls back to
        the older present-frame target for compatibility.

        Args:
            freeze_prob: (B,) predicted freeze probability in [0,1].
            redetect_prob: (B,) predicted redetect probability in [0,1].
            present_mask: (B,) bool, True = target present.
            reappear_mask: optional (B,) bool, previous absent/current present.
        Returns:
            loss, stats.
        """
        device = freeze_prob.device
        present = present_mask.to(device).float()
        absent = 1.0 - present
        redetect_target = reappear_mask.to(device).float() if reappear_mask is not None else present

        def _bce(prob, target):
            eps = 1e-7
            p = prob.clamp(eps, 1 - eps)
            logit = torch.log(p / (1 - p))
            return F.binary_cross_entropy_with_logits(logit, target)

        l_freeze = _bce(freeze_prob, absent)
        l_redetect = _bce(redetect_prob, redetect_target)
        loss = self.freeze_weight * l_freeze + self.redetect_gate_weight * l_redetect
        with torch.no_grad():
            fr = (freeze_prob > 0.5).float()
            rr = (redetect_prob > 0.5).float()
            absent_m = absent == 1
            redetect_m = redetect_target == 1
            freeze_recall = (fr[absent_m] == 1).float().mean().item() \
                if absent_m.any() else 1.0
            redetect_recall = (rr[redetect_m] == 1).float().mean().item() \
                if redetect_m.any() else 1.0
        return loss, {"Loss/freeze": l_freeze.item(),
                      "Loss/redetect_gate": l_redetect.item(),
                      "Policy/freeze_recall": freeze_recall,
                      "Policy/redetect_recall": redetect_recall,
                      "Policy/redetect_target_positive_count": int(redetect_target.sum().item())}

    # ------------------------------------------------------------------ #
    def forward(self, **kwargs):
        """Compute whichever losses are requested (pass None to skip).

        Returns: total_loss (0 if none), stats dict.
        """
        anchor = next((value for value in kwargs.values()
                       if torch.is_tensor(value)), None)
        total = anchor.new_zeros(()) if anchor is not None else torch.tensor(0.0)
        stats = {}
        if kwargs.get("router_logits") is not None and \
           kwargs.get("per_route_pred_xyxy") is not None and \
           kwargs.get("gt_xyxy") is not None:
            route_sizes = kwargs.get("route_sizes")
            if route_sizes is None:
                raise ValueError(
                    "route_sizes is required with per_route_pred_xyxy")
            l, s = self.routing_loss(
                kwargs["router_logits"],
                kwargs["per_route_pred_xyxy"],
                kwargs["gt_xyxy"],
                route_sizes=route_sizes,
                present_mask=kwargs.get("present_mask"),
            )
            total = total + self.route_weight * l
            stats.update(s)
        if kwargs.get("absence_prob") is not None and \
           kwargs.get("absent_gt") is not None:
            l, s = self.absence_loss(kwargs["absence_prob"],
                                     kwargs["absent_gt"])
            total = total + self.absence_weight * l
            stats.update(s)
        if kwargs.get("redetect_score_map") is not None and \
           kwargs.get("redetect_bbox") is not None and \
           kwargs.get("gt_xyxy") is not None:
            l, s = self.redetect_loss(kwargs["redetect_score_map"],
                                      kwargs["redetect_bbox"],
                                      kwargs["gt_xyxy"],
                                      kwargs.get("feat_sz", 12),
                                      kwargs.get("stride", 32))
            total = total + self.redetect_weight * l
            stats.update(s)
        if kwargs.get("freeze_prob") is not None and \
           kwargs.get("redetect_prob") is not None and \
           kwargs.get("present_mask") is not None:
            l, s = self.memory_policy_loss(kwargs["freeze_prob"],
                                           kwargs["redetect_prob"],
                                           kwargs["present_mask"],
                                           kwargs.get("reappear_mask"))
            total = total + l
            stats.update(s)
        return total, stats
