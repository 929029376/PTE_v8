"""PET-Track actor with one baseline-localization plus SRBT objective stack."""
import torch

from .pet_track_base import PETTrackBaseActor
from lib.models.layers.pet_losses import PetTrackLoss
from lib.models.layers.srbt_belief import REAPPEARING, VISIBLE
from lib.utils.route_motion import ROUTE_MOTION_DIM
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


def _balanced_singleton_routes(batch_size, epoch, expert_names):
    names = tuple(expert_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("expert_names must be non-empty and unique")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if torch.is_tensor(epoch):
        epoch = epoch.detach().reshape(-1)[0].item()
    offset = int(epoch) % len(names)
    return [
        (names[(offset + index) % len(names)],)
        for index in range(int(batch_size))
    ]


def _augment_route_motion(cues, training, dropout_probability, jitter_std):
    """Approximate prediction-history noise without using current GT."""
    if cues is None or not training:
        return cues
    output = cues.clone()
    valid = output[:, -1] > 0
    dropout_probability = float(dropout_probability)
    jitter_std = float(jitter_std)
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("route-motion dropout must be in [0, 1]")
    if jitter_std < 0.0:
        raise ValueError("route-motion jitter must be non-negative")

    dropped = torch.rand(
        output.shape[0], device=output.device) < dropout_probability
    output[dropped] = 0.0
    jittered = valid & ~dropped
    if jitter_std > 0.0 and jittered.any():
        output[jittered, :-1] += torch.randn_like(
            output[jittered, :-1]) * jitter_std
    return output


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
        self.pet_enabled = bool(getattr(cfg.MODEL.PET, "ENABLE", False))
        self.use_absence = (not self.srbt_enabled and self.pet_enabled
                            and bool(getattr(cfg.MODEL.PET, "ABSENCE_HEAD", False)))
        self.use_redetect = (not self.srbt_enabled and self.pet_enabled
                             and bool(getattr(cfg.MODEL.PET, "REDETECT_HEAD", False)))
        self.use_memory_policy = (self.pet_enabled
                                  and not self.srbt_enabled
                                  and bool(getattr(cfg.MODEL.PET, "USE_LEARNED_POLICY", True))
                                  and bool(getattr(cfg.MODEL.PET, "MEMORY_POLICY", True)))
        stage = getattr(cfg.TRAIN, "STAGE", "") or getattr(cfg.TRAIN, "EXPERT_STAGE", "all")
        stage = "all" if stage in (None, "", "all") else str(stage).lower()
        self.stage = "srbt" if self.srbt_enabled else stage
        # T_max is used to normalize the sampled frozen_age for policy training.
        sm_cfg = getattr(cfg.MODEL, "STATE_MACHINE", None)
        self.t_max = float(getattr(sm_cfg, "T_MAX", 50)) if sm_cfg is not None else 50.0
        # Counterfactual route evaluation is enabled only in router stages.
        self.use_route = False
        self.active_losses = (
            {"base", "srbt"} if self.srbt_enabled else {"base"})

    @staticmethod
    def _prepare_route_motion(cues, batch_size, device):
        if cues is None:
            return None
        cues = torch.as_tensor(cues, device=device, dtype=torch.float32)
        if cues.ndim == 1 and batch_size == 1:
            cues = cues.unsqueeze(0)
        elif cues.ndim == 2 and cues.shape == (ROUTE_MOTION_DIM, batch_size):
            cues = cues.transpose(0, 1)
        if cues.ndim != 2 or cues.shape != (batch_size, ROUTE_MOTION_DIM):
            raise ValueError(
                "route_motion_cues must collate to (K, B) or (B, K)")
        return cues.contiguous()

    @staticmethod
    def _prepare_template_frame_ids(frame_ids, batch_size, template_count,
                                    device):
        if frame_ids is None:
            return None
        frame_ids = torch.as_tensor(
            frame_ids, device=device, dtype=torch.long)
        if frame_ids.shape == (template_count, batch_size):
            frame_ids = frame_ids.transpose(0, 1)
        if frame_ids.shape != (batch_size, template_count):
            raise ValueError(
                "template_frame_ids must collate to (M, B) or (B, M)")
        return frame_ids.contiguous()

    # ------------------------------------------------------------------ #
    def forward_pass(self, data):
        """Forward pass that builds one causal belief inside the network and
        feeds it to all consumers (router / absence / memory policy). Optionally runs all
        experts for IoU soft routing."""
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

        # NOTE: in DDP mode auxiliary trainable heads must run inside
        # self.net(...), otherwise DDP can mark the same parameter ready twice.
        net = self.net
        if hasattr(net, 'module'):
            net = net.module

        redetect_mask = self._absent_to_present_mask(data, xi.shape[0], xi.device)
        is_training = bool(getattr(self.net, 'training', True))
        if is_training:
            frozen_age = torch.rand(xi.shape[0], device=xi.device)
        else:
            # Deterministic stratified coverage of the training distribution.
            # This keeps Stage-2 checkpoint ranking reproducible.
            frozen_age = (torch.arange(
                xi.shape[0], device=xi.device, dtype=torch.float32) + 0.5
            ) / xi.shape[0]
        redetect_images = data.get('redetect_search_images')
        route_motion = self._prepare_route_motion(
            data.get('route_motion_cues'), xi.shape[0], xi.device)
        template_frame_ids = self._prepare_template_frame_ids(
            data.get('template_frame_ids'), xi.shape[0], ze.shape[1],
            xi.device)
        data_cfg = getattr(self.cfg, "DATA", None)
        route_motion = _augment_route_motion(
            route_motion,
            training=is_training,
            dropout_probability=float(getattr(
                data_cfg, "ROUTE_MOTION_DROPOUT", 0.1)),
            jitter_std=float(getattr(
                data_cfg, "ROUTE_MOTION_JITTER_STD", 0.02)),
        )
        if self.stage in ("c3", "all") and self.use_redetect:
            missing = [
                key for key in ("redetect_search_images", "redetect_search_anno")
                if key not in data
            ]
            if missing:
                raise RuntimeError(
                    f"C3 training batch is missing global redetection fields: {missing}")
            redetect_images = data["redetect_search_images"]

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
            "redetect_mask": redetect_mask,
            "template_anno": z_anno,
            "template_frame_ids": template_frame_ids,
            "route_motion": route_motion,
            "frozen_age": frozen_age,
        }
        if self.stage == "expert":
            expert_names = tuple(getattr(net, "expert_names", ()))
            if not expert_names:
                expert_names = tuple(self.cfg.MODEL.EXPERT.NAMES)
            forward_kwargs["route"] = _balanced_singleton_routes(
                xi.shape[0], data.get("epoch", 0), expert_names)
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
            missing = sorted(REQUIRED_SRBT_PREDICTIONS - predictions.keys())
            if missing:
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

    # ------------------------------------------------------------------ #
    def _absent_to_present_mask(self, gt_dict, B, device):
        """True reappearance: previous frame absent, current frame present."""
        current = gt_dict.get("current_present", None)
        previous = gt_dict.get("previous_present", None)
        if current is not None and previous is not None:
            cur = current[-1] if current.dim() > 1 else current
            prev = previous[-1] if previous.dim() > 1 else previous
            cur = cur.to(device=device).reshape(-1).bool()
            prev = prev.to(device=device).reshape(-1).bool()
            if cur.numel() == B and prev.numel() == B:
                return (~prev) & cur

        # Legacy key name: FELT absent.txt is a presence flag in this project
        # (1=present, 0=absent), verified against zero-box frames.
        search_present = gt_dict.get("search_absent", None)
        if (search_present is not None and torch.is_tensor(search_present)
                and search_present.shape[0] > 1):
            prev = search_present[-2].to(device=device).reshape(-1) > 0
            cur = search_present[-1].to(device=device).reshape(-1) > 0
            if cur.numel() == B and prev.numel() == B:
                return (~prev) & cur
        if self.stage in ("c3", "all"):
            raise RuntimeError(
                "C3 training requires previous_present/current_present or "
                "at least two search_absent frames")
        return torch.zeros(B, dtype=torch.bool, device=device)
