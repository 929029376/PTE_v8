"""PET-Track actor with one baseline-localization plus SRBT objective stack."""
from collections.abc import Mapping

import torch

from .pet_track_base import PETTrackBaseActor
from lib.models.layers.pet_losses import PetTrackLoss
from lib.models.layers.srbt_belief import REAPPEARING, VISIBLE
from lib.utils.heapmap_utils import generate_heatmap


REQUIRED_SRBT_PREDICTIONS = frozenset({
    "existence_logits",
    "hazard_logits",
    "field_logits",
    "candidate_logits",
    "hypothesis_boxes",
    "hypothesis_scores",
    "identity_embeddings",
    "template_identity",
})


class PETTrackActor(PETTrackBaseActor):
    """Actor for training PET-Track models."""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective, loss_weight, settings, cfg)
        loss_cfg = getattr(cfg.TRAIN, "SRBT_LOSS", None)
        self.pet_loss = PetTrackLoss(
            existence_weight=float(getattr(loss_cfg, "EXISTENCE_WEIGHT", 1.0)),
            survival_weight=float(getattr(loss_cfg, "SURVIVAL_WEIGHT", 1.0)),
            field_weight=float(getattr(loss_cfg, "FIELD_WEIGHT", 1.0)),
            hypothesis_weight=float(getattr(loss_cfg, "HYPOTHESIS_WEIGHT", 0.5)),
            identity_weight=float(getattr(loss_cfg, "IDENTITY_WEIGHT", 0.2)),
            calibration_weight=float(getattr(
                loss_cfg, "CALIBRATION_WEIGHT", 0.05)),
            teacher_weight=float(getattr(loss_cfg, "TEACHER_WEIGHT", 1.0)),
            distill_max_weight=float(getattr(
                loss_cfg, "DISTILL_MAX_WEIGHT", 0.5)),
            distill_warmup=float(getattr(loss_cfg, "DISTILL_WARMUP", 0.05)),
            identity_temperature=float(getattr(
                loss_cfg, "IDENTITY_TEMPERATURE", 0.1)),
            diversity_margin=float(getattr(
                loss_cfg, "DIVERSITY_MARGIN", 0.25)),
            diversity_weight=float(getattr(
                loss_cfg, "DIVERSITY_WEIGHT", 0.1)),
            hazard_bins=int(getattr(
                cfg.MODEL.SRBT.TEACHER, "HAZARD_BINS", 129)),
        )
        self.srbt_enabled = bool(getattr(cfg.MODEL.SRBT, "ENABLE", False))
        self.stage = "srbt" if self.srbt_enabled else "base"
        self.active_losses = (
            {"base", "srbt"} if self.srbt_enabled else {"base"})

    # ------------------------------------------------------------------ #
    @staticmethod
    def _batch_first_future(value, batch_size, device):
        if value is None:
            return None
        value = torch.as_tensor(value, device=device)
        if value.ndim == 5 and value.shape[0] != batch_size \
                and value.shape[1] == batch_size:
            value = value.permute(1, 0, 2, 3, 4)
        if value.ndim != 5 or value.shape[0] != batch_size:
            raise ValueError("future observations must have shape (B,H,C,W,W) or (H,B,C,W,W)")
        return value.contiguous()

    @staticmethod
    def _batch_first_future_mask(value, batch_size, device):
        if value is None:
            return None
        value = torch.as_tensor(value, device=device, dtype=torch.bool)
        if value.ndim == 2 and value.shape[0] != batch_size \
                and value.shape[1] == batch_size:
            value = value.transpose(0, 1)
        if value.ndim != 2 or value.shape[0] != batch_size:
            raise ValueError("future_valid must have shape (B,H) or (H,B)")
        return value.contiguous()

    @staticmethod
    def _select_posterior(previous, updated, valid):
        if bool(valid.all()):
            return updated
        if not bool(valid.any()):
            return previous
        if isinstance(updated, Mapping):
            return {
                key: PETTrackActor._select_posterior(
                    previous[key], value, valid)
                for key, value in updated.items()
            }
        if not torch.is_tensor(updated) or updated.ndim == 0:
            return updated
        if previous.shape != updated.shape:
            if previous.ndim == updated.ndim and previous.shape[0] == updated.shape[0] \
                    and previous.numel() == 0:
                previous = torch.zeros_like(updated)
            else:
                raise ValueError("posterior field shape changed unexpectedly")
        mask = valid.reshape(valid.shape[0], *([1] * (updated.ndim - 1)))
        return torch.where(mask, updated, previous)

    def _history_posterior(self, zi, ze, data, batch_size, device, dtype):
        values = (
            data.get("history_images"),
            data.get("history_event_images"),
            data.get("history_valid"),
        )
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise RuntimeError(
                "history_images, history_event_images, and history_valid "
                "are required together")
        history_images = self._batch_first_future(values[0], batch_size, device)
        history_events = self._batch_first_future(values[1], batch_size, device)
        history_valid = self._batch_first_future_mask(
            values[2], batch_size, device)
        if history_images.shape[:2] != history_events.shape[:2]:
            raise ValueError("history RGB, event, and validity horizons must match")
        if history_images.shape[1] < history_valid.shape[1]:
            pad = history_valid.shape[1] - history_images.shape[1]
            rgb_pad = history_images.new_zeros(
                batch_size, pad, *history_images.shape[2:])
            event_pad = history_events.new_zeros(
                batch_size, pad, *history_events.shape[2:])
            history_images = torch.cat((rgb_pad, history_images), dim=1)
            history_events = torch.cat((event_pad, history_events), dim=1)
        if history_images.shape[:2] != history_valid.shape:
            raise ValueError("history RGB, event, and validity horizons must match")

        model = self.net.module if hasattr(self.net, "module") else self.net
        posterior = model.initialize_srbt_posterior(batch_size, device, dtype)
        with torch.no_grad():
            for frame_id in range(history_images.shape[1]):
                valid = history_valid[:, frame_id]
                if not bool(valid.any()):
                    continue
                history_out = self.net(
                    zi=zi,
                    ze=ze,
                    xi=history_images[:, frame_id:frame_id + 1],
                    xe=history_events[:, frame_id:frame_id + 1],
                    previous_posterior=posterior,
                )
                posterior = self._select_posterior(
                    posterior, history_out["srbt_posterior"], valid)
        return posterior

    def forward_pass(self, data):
        """Forward pass for baseline localization plus causal SRBT outputs."""
        zi = data['template_images'].permute(1, 0, 2, 3, 4)
        ze = data['template_event_images'].permute(1, 0, 2, 3, 4)
        xi = data['search_images'].permute(1, 0, 2, 3, 4)
        xe = data['search_event_images'].permute(1, 0, 2, 3, 4)
        z_anno = data['template_anno'].permute(1, 0, 2)

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

        previous_posterior = self._history_posterior(
            zi, ze, data, xi.shape[0], xi.device, xi.dtype)

        forward_kwargs = {
            "zi": zi,
            "ze": ze,
            "xi": xi,
            "xe": xe,
            "mask_z": mask_z,
            "ce_template_mask": box_mask_z,
            "ce_keep_rate": ce_keep_rate,
            "return_last_attn": False,
            "previous_posterior": previous_posterior,
            "future_images": self._batch_first_future(
                data.get("future_images"), xi.shape[0], xi.device),
            "future_event_images": self._batch_first_future(
                data.get("future_event_images"), xi.shape[0], xi.device),
            "future_valid": self._batch_first_future_mask(
                data.get("future_valid"), xi.shape[0], xi.device),
        }
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
        """Present-masked baseline localization plus the joint SRBT loss."""
        loss, status = super().compute_losses(pred_dict, gt_dict, return_status=True)
        base_loss = loss
        if self.srbt_enabled:
            predictions = pred_dict.get("srbt_predictions")
            if predictions is None:
                raise RuntimeError(
                    "SRBT training requires srbt_predictions from model forward")
            actual = set(predictions)
            if actual != REQUIRED_SRBT_PREDICTIONS:
                missing = sorted(REQUIRED_SRBT_PREDICTIONS - actual)
                extra = sorted(actual - REQUIRED_SRBT_PREDICTIONS)
                if extra:
                    raise RuntimeError(
                        "SRBT predictions must expose exactly eight outputs; "
                        f"extra outputs: {extra}")
                raise RuntimeError(
                    f"SRBT predictions missing required outputs: {missing}")
            targets = self._build_srbt_targets(
                gt_dict, pred_dict["pred_boxes"].device,
                pred_dict["pred_boxes"].shape[0])
            progress = gt_dict.get("training_progress", 0.0)
            if torch.is_tensor(progress):
                progress = progress.detach().reshape(-1)[0].item()
            srbt_loss, srbt_status = self.pet_loss.srbt_loss(
                predictions,
                targets,
                teacher=pred_dict.get("srbt_teacher"),
                progress=progress,
            )
            loss = loss + srbt_loss
            status.update(srbt_status)

        if not return_status:
            return loss
        status["Loss/base"] = base_loss.item()
        status["Loss/SRBT"] = (loss - base_loss).item()
        status["Loss/total"] = loss.item()
        return loss, status

    def _build_srbt_targets(self, data, device, batch_size):
        present = self._present_search_mask(data, device, batch_size)
        if present is None:
            raise RuntimeError("SRBT training requires frame-level presence")
        targets = {"presence": present.float()}
        for name in ("hazard_target", "hazard_mask", "censor_mask"):
            value = data.get(name)
            if value is None:
                raise RuntimeError(f"SRBT training requires {name}")
            targets[name] = value.to(device=device).reshape(-1)

        gt_xywh = data["search_anno"][-1].to(device=device)
        targets["target_box"] = torch.stack((
            gt_xywh[:, 0] + 0.5 * gt_xywh[:, 2],
            gt_xywh[:, 1] + 0.5 * gt_xywh[:, 3],
            gt_xywh[:, 2],
            gt_xywh[:, 3],
        ), dim=-1)
        state = data.get("state_target")
        if state is None:
            raise RuntimeError("SRBT training requires state_target")
        state = state.to(device=device).reshape(-1)
        field_mask = (state == VISIBLE) | (state == REAPPEARING)
        targets["field_mask"] = field_mask
        targets["hypothesis_mask"] = field_mask
        targets["identity_mask"] = field_mask
        targets["field_target"] = generate_heatmap(
            data["search_anno"], self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE,
        )[-1].unsqueeze(1).to(device=device)
        return targets
