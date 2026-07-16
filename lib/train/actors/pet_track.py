"""PET-Track actor with local experts, presence gating, and recovery."""
import torch
import torch.nn.functional as F

from .pet_track_base import PETTrackBaseActor
from lib.utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    box_xywh_to_xyxy,
    generalized_box_iou,
)
from lib.utils.heapmap_utils import generate_heatmap


REQUIRED_PRESENCE_PREDICTIONS = frozenset({"logits", "score"})
VISIBILITY_EXPERT_ID = 3


class PETTrackActor(PETTrackBaseActor):
    """Actor for training PET-Track models."""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective, loss_weight, settings, cfg)
        loss_cfg = getattr(cfg.TRAIN, "SRBT_LOSS", None)
        self.srbt_enabled = bool(getattr(cfg.MODEL.SRBT, "ENABLE", False))
        self.presence_loss_weight = float(getattr(
            loss_cfg, "EXISTENCE_WEIGHT", 1.0))
        self.presence_focal_gamma = float(getattr(
            loss_cfg, "FOCAL_GAMMA", 2.0))
        recovery_loss_cfg = getattr(cfg.TRAIN, "RECOVERY_LOSS", None)
        self.identity_loss_weight = float(getattr(
            recovery_loss_cfg, "IDENTITY_WEIGHT", 1.0))
        self.identity_ranking_weight = float(getattr(
            recovery_loss_cfg, "RANKING_WEIGHT", 0.5))
        self.identity_ranking_margin = float(getattr(
            recovery_loss_cfg, "RANKING_MARGIN", 0.2))
        expert_cfg = getattr(cfg.MODEL, "EXPERT", None)
        self.expert_enabled = bool(getattr(
            expert_cfg, "ENABLE", False)) if expert_cfg is not None else False
        self.expert_phase = str(getattr(
            cfg.TRAIN, "EXPERT_PHASE", "specialize")).lower()
        if self.expert_phase not in {
                "specialize", "refine", "recovery"}:
            raise ValueError(
                "TRAIN.EXPERT_PHASE must be specialize, refine, or recovery")
        self.stage = self.expert_phase if self.expert_enabled else (
            "srbt" if self.srbt_enabled else "base")
        self.active_losses = (
            {"presence", "redetect", "identity"}
            if self.expert_phase == "recovery"
            else ({"base", "srbt", "redetect"}
                  if self.srbt_enabled else {"base"})
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _batch_first_observations(value, batch_size, device):
        if value is None:
            return None
        value = torch.as_tensor(value, device=device)
        if value.ndim == 5 and value.shape[0] != batch_size \
                and value.shape[1] == batch_size:
            value = value.permute(1, 0, 2, 3, 4)
        if value.ndim != 5 or value.shape[0] != batch_size:
            raise ValueError(
                "observations must have shape (B,H,C,W,W) or (H,B,C,W,W)")
        return value.contiguous()

    def forward_pass(self, data):
        """Forward pass for baseline localization plus causal SRBT outputs."""
        zi = data['template_images'].permute(1, 0, 2, 3, 4)
        ze = data['template_event_images'].permute(1, 0, 2, 3, 4)
        xi = data['search_images'].permute(1, 0, 2, 3, 4)
        xe = data['search_event_images'].permute(1, 0, 2, 3, 4)
        z_anno = data['template_anno'].permute(1, 0, 2)

        owner_ids = None
        if self.expert_enabled and self.expert_phase == "specialize":
            owner_ids = data.get("expert_owner_id")
            if owner_ids is None:
                raise RuntimeError(
                    "expert specialization requires expert_owner_id")
            owner_ids = torch.as_tensor(owner_ids).reshape(-1)
            if owner_ids.numel() != xi.shape[0]:
                raise ValueError(
                    "expert_owner_id must provide one owner per batch row")
            model = self.net.module if hasattr(self.net, "module") else self.net
            if any(owner < 0 or owner >= len(model.expert_names)
                   for owner in owner_ids.tolist()):
                raise ValueError("expert_owner_id contains an invalid owner")

        box_mask_z = []
        mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            for i in range(self.settings.num_template):
                box_mask_z.append(self._gen_mask_cond(zi[:, i].shape[0], zi[:, i].device, z_anno[:, i]))
                mask_z.append(self._gen_mask_z(zi[:, i].shape[0], zi[:, i].device, z_anno[:, i]))
            box_mask_z = torch.cat(box_mask_z, dim=1)
            mask_z = torch.cat(mask_z, dim=1)
            ce_keep_rate = self._adjust_keep_rate(data.get('epoch', 0))

        redetect_images = None
        redetect_event_images = None
        redetect_mask = None
        is_reappear = data.get("is_reappear")
        trains_visibility = (
            owner_ids is None
            or bool(torch.all(owner_ids == VISIBILITY_EXPERT_ID))
        )
        if is_reappear is not None and trains_visibility:
            redetect_mask = torch.as_tensor(
                is_reappear[-1], device=xi.device, dtype=torch.bool).reshape(-1)
            if redetect_mask.numel() != xi.shape[0]:
                raise ValueError("is_reappear must provide one flag per batch row")
            if bool(redetect_mask.any()):
                redetect_images = self._batch_first_observations(
                    data.get("redetect_search_images"), xi.shape[0], xi.device)
                redetect_event_images = self._batch_first_observations(
                    data.get("redetect_search_event_images"), xi.shape[0], xi.device)
                if redetect_images is None or redetect_event_images is None:
                    raise RuntimeError(
                        "reappearance training requires global RGB and event searches")
            else:
                redetect_mask = None

        forward_kwargs = {
            "zi": zi,
            "ze": ze,
            "xi": xi,
            "xe": xe,
            "mask_z": mask_z,
            "ce_template_mask": box_mask_z,
            "ce_keep_rate": ce_keep_rate,
            "return_last_attn": False,
            "redetect_images": redetect_images,
            "redetect_event_images": redetect_event_images,
            "redetect_mask": redetect_mask,
        }
        if owner_ids is not None:
            forward_kwargs["expert_owner_ids"] = owner_ids
        out_dict = self.net(**forward_kwargs)

        return out_dict

    # helpers for the CE logic (kept local)
    def _gen_mask_cond(self, bs, device, gt_bbox):
        from lib.utils.ce_utils import generate_mask_cond
        return generate_mask_cond(cfg=self.cfg, bs=bs, device=device, gt_bbox=gt_bbox)

    def _gen_mask_z(self, bs, device, gt_bbox):
        from lib.utils.ce_utils import generate_mask_z
        return generate_mask_z(cfg=self.cfg, bs=bs, device=device, gt_bbox=gt_bbox)

    def _adjust_keep_rate(self, epoch):
        from lib.utils.ce_utils import adjust_keep_rate
        ce_start = self.cfg.TRAIN.CE_START_EPOCH
        ce_warm = self.cfg.TRAIN.CE_WARM_EPOCH
        return adjust_keep_rate(epoch, warmup_epochs=ce_start,
                                total_epochs=ce_start + ce_warm,
                                ITERS_PER_EPOCH=1,
                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

    # ------------------------------------------------------------------ #
    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        """Present-masked localization plus the combined SRBT loss."""
        loss, status = super().compute_losses(pred_dict, gt_dict, return_status=True)
        base_loss = loss
        if self.expert_enabled and self.expert_phase == "recovery":
            loss = base_loss * 0.0
        srbt_loss = base_loss * 0.0
        redetect_loss = base_loss * 0.0
        owner_ids = None
        trains_visibility = True
        if self.expert_enabled and self.expert_phase == "specialize":
            expert_owner_ids = gt_dict.get("expert_owner_id")
            if expert_owner_ids is None:
                raise RuntimeError(
                    "expert specialization requires expert_owner_id")
            owner_ids = torch.as_tensor(
                expert_owner_ids, device=pred_dict["pred_boxes"].device,
                dtype=torch.long).reshape(-1)
            if owner_ids.numel() != pred_dict["pred_boxes"].shape[0]:
                raise ValueError(
                    "expert_owner_id must provide one owner per batch row")
            trains_visibility = bool(torch.all(
                owner_ids == VISIBILITY_EXPERT_ID))

        if self.srbt_enabled and trains_visibility:
            predictions = pred_dict.get("presence_predictions")
            if predictions is None:
                raise RuntimeError(
                    "presence training requires presence_predictions")
            actual = set(predictions)
            if actual != REQUIRED_PRESENCE_PREDICTIONS:
                missing = sorted(REQUIRED_PRESENCE_PREDICTIONS - actual)
                raise RuntimeError(
                    f"presence predictions have invalid outputs; missing: {missing}")
            presence = self._present_search_mask(
                gt_dict, predictions["logits"].device,
                predictions["logits"].shape[0])
            if presence is None:
                raise RuntimeError("presence training requires official visibility")
            logits = predictions["logits"].float()
            targets = presence.to(torch.long)
            per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
            target_probability = logits.softmax(dim=-1).gather(
                1, targets[:, None]).squeeze(1)
            srbt_loss = self.presence_loss_weight * (
                (1.0 - target_probability).pow(self.presence_focal_gamma)
                * per_sample_ce
            ).mean()
            loss = loss + srbt_loss
            status["Loss/presence"] = srbt_loss.item()

            redetect_predictions = pred_dict.get("redetect_predictions")
            if redetect_predictions is not None:
                redetect_loss, redetect_status = self._compute_redetect_loss(
                    redetect_predictions, gt_dict)
                loss = loss + redetect_loss
                status.update(redetect_status)
        elif self.srbt_enabled:
            status["Loss/presence"] = 0.0

        model = self.net.module if hasattr(self.net, "module") else self.net
        if self.expert_enabled and self.expert_phase == "specialize":
            gt_xyxy = box_xywh_to_xyxy(
                gt_dict["search_anno"][-1].to(owner_ids.device))
            pred_xyxy = box_cxcywh_to_xyxy(
                pred_dict["pred_boxes"][:, 0]).detach()
            present = self._present_search_mask(
                gt_dict, owner_ids.device, owner_ids.numel())
            if present is None:
                present = torch.ones_like(owner_ids, dtype=torch.bool)
            owned_iou = box_iou(pred_xyxy, gt_xyxy)[0]
            for expert_id in range(len(model.expert_names)):
                owned = present & (owner_ids == expert_id)
                status[f"Expert/owner_count_{expert_id}"] = int(owned.sum())
                status[f"Expert/owner_iou_{expert_id}"] = (
                    owned_iou[owned].mean().item() if owned.any() else 0.0)

        if not return_status:
            return loss
        status["Loss/base"] = base_loss.item()
        status["Loss/SRBT"] = srbt_loss.item()
        status.setdefault("Loss/redetect", redetect_loss.item())
        status["Expert/phase_id"] = {
            "specialize": 0, "refine": 1, "recovery": 2,
        }[self.expert_phase]
        status.setdefault("Redetect/count", 0)
        status["Loss/total"] = loss.item()
        return loss, status

    def _compute_redetect_loss(self, predictions, data):
        indices = predictions["batch_indices"].to(
            device=predictions["score_map"].device, dtype=torch.long)
        annotations = data.get("redetect_search_anno")
        if annotations is None:
            raise RuntimeError(
                "redetection predictions require redetect_search_anno")
        annotations = torch.as_tensor(
            annotations, device=indices.device)
        gt_xywh = annotations[-1].index_select(0, indices)
        target_box = torch.stack((
            gt_xywh[:, 0] + 0.5 * gt_xywh[:, 2],
            gt_xywh[:, 1] + 0.5 * gt_xywh[:, 3],
            gt_xywh[:, 2],
            gt_xywh[:, 3],
        ), dim=-1)
        heatmap = generate_heatmap(
            annotations,
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE,
        )[-1].unsqueeze(1).to(device=indices.device)
        heatmap = heatmap.index_select(0, indices)
        focal_loss = self.objective["focal"](
            predictions["score_map"], heatmap)
        box_loss = F.l1_loss(predictions["bbox"], target_box)
        pred_xyxy = box_cxcywh_to_xyxy(predictions["bbox"])
        target_xyxy = box_cxcywh_to_xyxy(target_box).clamp(0.0, 1.0)
        giou, _ = generalized_box_iou(pred_xyxy, target_xyxy)
        giou_loss = (1.0 - giou).mean()

        identity_scores = predictions.get("identity_scores")
        if identity_scores is None:
            raise RuntimeError(
                "recovery training requires RGB identity scores")
        count = indices.numel()
        if identity_scores.shape != (count, count):
            raise ValueError(
                "identity_scores must compare every template with every candidate")
        identity_scores = identity_scores.float().clamp(1e-6, 1.0 - 1e-6)
        identity_targets = torch.eye(
            count, device=identity_scores.device, dtype=identity_scores.dtype)
        identity_logits = torch.logit(identity_scores)
        identity_loss = F.binary_cross_entropy_with_logits(
            identity_logits, identity_targets)
        if count > 1:
            positive = identity_scores.diagonal()
            negatives = identity_scores.masked_fill(
                identity_targets.bool(), float("-inf"))
            hardest_negative = negatives.max(dim=1).values
            ranking_loss = F.relu(
                self.identity_ranking_margin - positive + hardest_negative
            ).mean()
        else:
            ranking_loss = identity_scores.sum() * 0.0
        loss = (
            self.loss_weight["focal"] * focal_loss
            + self.loss_weight["l1"] * box_loss
            + self.loss_weight["giou"] * giou_loss
            + self.identity_loss_weight * identity_loss
            + self.identity_ranking_weight * ranking_loss
        )
        return loss, {
            "Loss/redetect": loss.item(),
            "Loss/redetect_focal": focal_loss.item(),
            "Loss/redetect_l1": box_loss.item(),
            "Loss/redetect_giou": giou_loss.item(),
            "Loss/recovery_identity": identity_loss.item(),
            "Loss/recovery_ranking": ranking_loss.item(),
            "Redetect/count": int(indices.numel()),
        }
