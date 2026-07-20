"""PET-Track actor with local experts, presence gating, and recovery."""
import math

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
from lib.train.data.felt_challenges import (
    CHALLENGE_NAMES,
    expert_supervision_mask,
)
from lib.models.layers.expert_ensemble import normalized_response_psr
from lib.models.layers.search_window_controller import (
    crop_box_to_image,
    crop_target_inside,
    dynamic_search_crop,
    event_motion_centroid,
    search_window_pursuit_loss,
)


REQUIRED_PRESENCE_PREDICTIONS = frozenset({"logits", "score"})
VISIBILITY_EXPERT_ID = 3
ADVANTAGE_EXPERT_IDS = frozenset({1, 2, 4})


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
        controller_cfg = getattr(cfg.MODEL.SRBT, "CONTROLLER", None)
        self.presence_present_threshold = float(getattr(
            controller_cfg, "THETA_PRESENT", 0.70))
        self.presence_recover_threshold = float(getattr(
            controller_cfg, "THETA_RECOVER", 0.75))
        recovery_loss_cfg = getattr(cfg.TRAIN, "RECOVERY_LOSS", None)
        self.identity_loss_weight = float(getattr(
            recovery_loss_cfg, "IDENTITY_WEIGHT", 1.0))
        self.identity_ranking_weight = float(getattr(
            recovery_loss_cfg, "RANKING_WEIGHT", 0.5))
        self.identity_ranking_margin = float(getattr(
            recovery_loss_cfg, "RANKING_MARGIN", 0.2))
        self.expert_advantage_weight = float(getattr(
            cfg.TRAIN, "EXPERT_ADVANTAGE_WEIGHT", 1.0))
        self.expert_advantage_margin = float(getattr(
            cfg.TRAIN, "EXPERT_ADVANTAGE_MARGIN", 0.10))
        self.activation_advantage_margin = float(getattr(
            cfg.TRAIN, "ACTIVATOR_ADVANTAGE_MARGIN", 0.02))
        self.activation_pos_weight = tuple(float(value) for value in getattr(
            cfg.TRAIN, "ACTIVATOR_POS_WEIGHT", (1.0, 1.0, 1.0, 1.0)))
        if (not math.isfinite(self.expert_advantage_weight)
                or self.expert_advantage_weight <= 0.0):
            raise ValueError(
                "TRAIN.EXPERT_ADVANTAGE_WEIGHT must be finite and positive")
        if (not math.isfinite(self.expert_advantage_margin)
                or not 0.0 < self.expert_advantage_margin <= 1.0):
            raise ValueError(
                "TRAIN.EXPERT_ADVANTAGE_MARGIN must be in (0, 1]")
        if (not math.isfinite(self.activation_advantage_margin)
                or not 0.0 <= self.activation_advantage_margin <= 1.0):
            raise ValueError(
                "TRAIN.ACTIVATOR_ADVANTAGE_MARGIN must be in [0, 1]")
        expert_cfg = getattr(cfg.MODEL, "EXPERT", None)
        self.expert_enabled = bool(getattr(
            expert_cfg, "ENABLE", False)) if expert_cfg is not None else False
        self.expert_phase = str(getattr(
            cfg.TRAIN, "EXPERT_PHASE", "specialize")).lower()
        if self.expert_phase not in {
                "specialize", "refine", "recovery", "pursuit", "dispatch"}:
            raise ValueError(
                "TRAIN.EXPERT_PHASE must be specialize, refine, recovery, "
                "pursuit, or dispatch")
        if self.expert_phase == "dispatch":
            specialist_count = max(
                len(getattr(expert_cfg, "NAMES", ())) - 1, 0)
            if (len(self.activation_pos_weight) != specialist_count
                    or any(not math.isfinite(value) or value <= 0.0
                           for value in self.activation_pos_weight)):
                raise ValueError(
                    "TRAIN.ACTIVATOR_POS_WEIGHT must contain one finite "
                    "positive value per specialist")
        self.stage = self.expert_phase if self.expert_enabled else (
            "srbt" if self.srbt_enabled else "base")
        self.active_losses = (
            {"activation"}
            if self.expert_phase == "dispatch"
            else
            {"pursuit"}
            if self.expert_phase == "pursuit"
            else
            {"presence", "redetect", "identity"}
            if self.expert_phase == "recovery"
            else ({"base", "expert_advantage", "srbt", "redetect"}
                  if self.srbt_enabled
                  else {"base", "expert_advantage"})
        )

    # ------------------------------------------------------------------ #
    def train(self, mode=True):
        if self.expert_phase != "dispatch":
            return super().train(mode)
        self.net.eval()
        model = self.net.module if hasattr(self.net, "module") else self.net
        model.expert_activator.train(mode)

    # ------------------------------------------------------------------ #
    def _validated_training_expert_ids(self, data, batch_size, device):
        if not self.expert_enabled or self.expert_phase != "specialize":
            return None
        training_expert_ids = data.get("training_expert_id")
        challenge_labels = data.get("challenge_labels")
        if training_expert_ids is None or challenge_labels is None:
            raise RuntimeError(
                "expert specialization requires training_expert_id and "
                "challenge_labels")
        training_expert_ids = torch.as_tensor(
            training_expert_ids, device=device, dtype=torch.long).reshape(-1)
        challenge_labels = torch.as_tensor(
            challenge_labels, device=device, dtype=torch.bool)
        loader_shape = (len(CHALLENGE_NAMES), batch_size)
        if challenge_labels.shape == loader_shape:
            challenge_labels = challenge_labels.transpose(0, 1)
        if training_expert_ids.numel() != batch_size:
            raise ValueError(
                "training_expert_id must provide one ID per batch row")
        if challenge_labels.shape != (batch_size, len(CHALLENGE_NAMES)):
            raise ValueError(
                "challenge_labels must have shape "
                f"({batch_size}, {len(CHALLENGE_NAMES)})")
        model = self.net.module if hasattr(self.net, "module") else self.net
        if ((training_expert_ids < 1)
                | (training_expert_ids >= len(model.expert_names))).any():
            raise ValueError("training_expert_id contains an invalid specialist")
        attributes = {
            name: challenge_labels[:, index]
            for index, name in enumerate(CHALLENGE_NAMES)
        }
        eligible = expert_supervision_mask(attributes).to(device)
        selected = eligible.gather(1, training_expert_ids[:, None]).squeeze(1)
        if not bool(selected.all()):
            raise ValueError(
                "training expert is not eligible for the frame challenge labels")
        return training_expert_ids

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
        expert_phase = getattr(self, "expert_phase", "specialize")
        if expert_phase == "pursuit":
            return self._forward_pursuit(data)
        zi = data['template_images'].permute(1, 0, 2, 3, 4)
        ze = data['template_event_images'].permute(1, 0, 2, 3, 4)
        xi = data['search_images'].permute(1, 0, 2, 3, 4)
        xe = data['search_event_images'].permute(1, 0, 2, 3, 4)
        z_anno = data['template_anno'].permute(1, 0, 2)

        training_expert_ids = self._validated_training_expert_ids(
            data, xi.shape[0], xi.device)

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
            training_expert_ids is None
            or bool(torch.all(
                training_expert_ids == VISIBILITY_EXPERT_ID))
        )
        if (expert_phase != "dispatch"
                and is_reappear is not None and trains_visibility):
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
        if training_expert_ids is not None:
            forward_kwargs["training_expert_ids"] = training_expert_ids
        if expert_phase == "dispatch":
            forward_kwargs["return_activation_logits"] = True
        out_dict = self.net(**forward_kwargs)

        return out_dict

    @staticmethod
    def _aligned_iou_xywh(first, second):
        first_max = first[:, :2] + first[:, 2:]
        second_max = second[:, :2] + second[:, 2:]
        intersection = (
            torch.minimum(first_max, second_max)
            - torch.maximum(first[:, :2], second[:, :2])
        ).clamp_min(0.0)
        intersection = intersection[:, 0] * intersection[:, 1]
        first_area = first[:, 2] * first[:, 3]
        second_area = second[:, 2] * second[:, 3]
        return intersection / (
            first_area + second_area - intersection).clamp_min(1e-8)

    def _forward_pursuit(self, data):
        """Unroll frozen experts over crops planned by the preceding frame."""
        model = self.net.module if hasattr(self.net, "module") else self.net
        controller = getattr(model, "search_window_controller", None)
        if controller is None:
            raise RuntimeError(
                "pursuit training requires search_window_controller")
        required = (
            "pursuit_search_images",
            "pursuit_search_event_images",
            "pursuit_search_anno",
            "pursuit_search_present",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise RuntimeError(
                "pursuit training batch is missing: " + ", ".join(missing))
        zi = data["template_images"].permute(1, 0, 2, 3, 4)
        ze = data["template_event_images"].permute(1, 0, 2, 3, 4)
        frames = data["pursuit_search_images"].permute(1, 0, 2, 3, 4)
        event_frames = data[
            "pursuit_search_event_images"].permute(1, 0, 2, 3, 4)
        annotations = torch.as_tensor(
            data["pursuit_search_anno"], device=frames.device,
            dtype=frames.dtype).permute(1, 0, 2)
        present = torch.as_tensor(
            data["pursuit_search_present"], device=frames.device,
            dtype=torch.bool).permute(1, 0)
        if frames.shape[1] < 2:
            raise ValueError("pursuit training requires at least two frames")
        if annotations.shape[:2] != frames.shape[:2] \
                or present.shape != frames.shape[:2]:
            raise ValueError("pursuit sequence fields must share [batch, time]")
        search_factor = float(getattr(
            self.cfg.DATA.SEARCH, "FACTOR",
            self.settings.search_area_factor["search"]))
        search_size = int(self.cfg.DATA.SEARCH.SIZE)
        planned_anchor = annotations[:, 0].detach()
        previous_observation = planned_anchor
        predictions = []
        targets = []
        current_inside_values = []
        current_quality_values = []
        present_next_values = []
        crop_anchors = []
        expert_cfg = getattr(
            getattr(self.cfg, "MODEL", None), "EXPERT", None)
        use_activation = bool(getattr(
            expert_cfg, "USE_ACTIVATION_INFERENCE", False))
        if use_activation and not bool(getattr(
                expert_cfg, "ACTIVATOR_TRAINED", False)):
            raise RuntimeError(
                "pursuit sparse activation requires a trained expert activator")
        was_training = model.training
        model.eval()
        controller.train(was_training)
        try:
            for frame_index in range(frames.shape[1] - 1):
                crop_anchors.append(planned_anchor)
                search, crop_region = dynamic_search_crop(
                    frames[:, frame_index], planned_anchor,
                    search_factor, search_size)
                event_search, _ = dynamic_search_crop(
                    event_frames[:, frame_index], planned_anchor,
                    search_factor, search_size)
                with torch.no_grad():
                    inference_kwargs = dict(
                        static_zi=zi[:, 0], static_ze=ze[:, 0],
                        dynamic_zi=zi[:, 1:], dynamic_ze=ze[:, 1:],
                        xi=search, xe=event_search)
                    if use_activation:
                        inference_kwargs["auto_activate"] = True
                    output = model.inference(**inference_kwargs)
                    expert_outputs = output.get("expert_outputs")
                    if not expert_outputs:
                        raise RuntimeError(
                            "pursuit training requires expert outputs")
                    generalist_name = model.expert_names[0]
                    generalist = expert_outputs.get(generalist_name)
                    if generalist is None:
                        raise RuntimeError(
                            "pursuit training requires the generalist output")
                    if use_activation:
                        active_mask = output.get("expert_activation_mask")
                        expected_shape = (
                            search.shape[0], len(model.expert_names))
                        if active_mask is None or tuple(active_mask.shape) != expected_shape:
                            raise RuntimeError(
                                "pursuit sparse activation requires a valid "
                                "expert_activation_mask")
                        active_mask = active_mask.to(
                            device=search.device, dtype=torch.bool)
                        if not bool(active_mask[:, 0].all()):
                            raise RuntimeError(
                                "the generalist must be active for every pursuit row")
                    else:
                        active_mask = torch.ones(
                            search.shape[0], len(model.expert_names),
                            device=search.device, dtype=torch.bool)
                    missing_active = [
                        name for expert_id, name in enumerate(model.expert_names)
                        if bool(active_mask[:, expert_id].any())
                        and name not in expert_outputs
                    ]
                    if missing_active:
                        raise RuntimeError(
                            "pursuit activation is missing expert outputs: "
                            + ", ".join(missing_active))
                    ordered = [
                        expert_outputs.get(name, generalist)
                        for name in model.expert_names]
                    crop_boxes = torch.cat([
                        item["pred_boxes"][:, :1] for item in ordered
                    ], dim=1)
                    crop_boxes = torch.where(
                        active_mask[..., None], crop_boxes,
                        crop_boxes[:, :1].expand_as(crop_boxes))
                    expert_boxes = crop_box_to_image(
                        crop_boxes, crop_region)
                    response_peaks = torch.stack([
                        item["score_map"].flatten(1).max(dim=1).values
                        for item in ordered
                    ], dim=1)
                    response_psr = torch.stack([
                        normalized_response_psr(item["score_map"])
                        for item in ordered
                    ], dim=1)
                    response_peaks = response_peaks.masked_fill(
                        ~active_mask, 0.0)
                    response_psr = response_psr.masked_fill(
                        ~active_mask, 0.0)
                    reliability = (
                        response_peaks.clamp(0.0, 1.0)
                        * response_psr.clamp(0.0, 1.0)
                    )
                    reliability = torch.where(
                        active_mask, reliability.clamp_min(1e-6),
                        torch.zeros_like(reliability))
                    expert_weights = reliability / reliability.sum(
                        dim=1, keepdim=True).clamp_min(1e-6)
                    observation = (
                        expert_weights[..., None] * expert_boxes
                    ).sum(dim=1)
                    presence_score = output.get("presence_score")
                    if presence_score is None:
                        presence_score = frames.new_ones(frames.shape[0])
                    event_center, event_confidence = event_motion_centroid(
                        event_frames[:, frame_index])
                prediction = controller(
                    current_box=observation.detach(),
                    previous_box=previous_observation.detach(),
                    expert_boxes=expert_boxes.detach(),
                    response_peaks=response_peaks.detach(),
                    response_psr=response_psr.detach(),
                    presence=presence_score.detach(),
                    event_center=event_center.detach(),
                    event_confidence=event_confidence.detach(),
                )
                current_inside = crop_target_inside(
                    annotations[:, frame_index], planned_anchor,
                    search_factor) & present[:, frame_index]
                current_quality = self._aligned_iou_xywh(
                    observation, annotations[:, frame_index])
                predictions.append(prediction)
                targets.append(annotations[:, frame_index + 1])
                current_inside_values.append(current_inside)
                current_quality_values.append(current_quality)
                present_next_values.append(present[:, frame_index + 1])
                previous_observation = observation.detach()
                planned_anchor = prediction.next_box.detach()
        finally:
            model.train(was_training)
        return {
            "pursuit_predictions": predictions,
            "pursuit_targets": targets,
            "pursuit_current_inside": current_inside_values,
            "pursuit_current_quality": current_quality_values,
            "pursuit_present_next": present_next_values,
            "pursuit_crop_anchors": crop_anchors,
        }

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
        if self.expert_phase == "dispatch":
            return self._compute_dispatch_loss(
                pred_dict, gt_dict, return_status=return_status)
        if self.expert_phase == "pursuit":
            return self._compute_pursuit_losses(
                pred_dict, return_status=return_status)
        loss, status = super().compute_losses(pred_dict, gt_dict, return_status=True)
        base_loss = loss
        if self.expert_enabled and self.expert_phase == "recovery":
            loss = base_loss * 0.0
        srbt_loss = base_loss * 0.0
        presence_threshold_loss = base_loss * 0.0
        redetect_loss = base_loss * 0.0
        training_expert_ids = None
        trains_visibility = True
        if self.expert_enabled and self.expert_phase == "specialize":
            training_expert_ids = self._validated_training_expert_ids(
                gt_dict,
                pred_dict["pred_boxes"].shape[0],
                pred_dict["pred_boxes"].device,
            )
            trains_visibility = bool(torch.all(
                training_expert_ids == VISIBILITY_EXPERT_ID))

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
            probabilities = logits.softmax(dim=-1)
            target_probability = probabilities.gather(
                1, targets[:, None]).squeeze(1)
            focal_loss = (
                (1.0 - target_probability).pow(self.presence_focal_gamma)
                * per_sample_ce
            ).mean()
            present_thresholds = torch.full_like(
                target_probability, self.presence_present_threshold)
            is_reappear = gt_dict.get("is_reappear")
            if is_reappear is not None:
                reappear = torch.as_tensor(
                    is_reappear, device=logits.device, dtype=torch.bool)
                if reappear.numel() != logits.shape[0]:
                    reappear = reappear[-1]
                reappear = reappear.reshape(-1)
                if reappear.numel() != logits.shape[0]:
                    raise ValueError(
                        "is_reappear must provide one flag per batch row")
                present_thresholds = torch.where(
                    reappear,
                    torch.full_like(
                        present_thresholds,
                        self.presence_recover_threshold),
                    present_thresholds,
                )
            presence_probability = probabilities[:, 1]
            threshold_error = torch.where(
                presence,
                F.relu(present_thresholds - presence_probability),
                F.relu(
                    presence_probability - self.presence_present_threshold),
            )
            presence_threshold_loss = (
                self.presence_loss_weight * threshold_error.mean())
            srbt_loss = (
                self.presence_loss_weight * focal_loss
                + presence_threshold_loss)
            loss = loss + srbt_loss
            status["Loss/presence"] = srbt_loss.item()
            status["Loss/presence_threshold"] = (
                presence_threshold_loss.item())

            redetect_predictions = pred_dict.get("redetect_predictions")
            if redetect_predictions is not None:
                redetect_loss, redetect_status = self._compute_redetect_loss(
                    redetect_predictions, gt_dict)
                loss = loss + redetect_loss
                status.update(redetect_status)
        elif self.srbt_enabled:
            status["Loss/presence"] = 0.0
            status["Loss/presence_threshold"] = 0.0

        advantage_loss, advantage_status = (
            self._compute_expert_advantage_loss(
                pred_dict, gt_dict, training_expert_ids)
            if self.expert_enabled and self.expert_phase == "specialize"
            else (base_loss * 0.0, {
                "Loss/expert_advantage": 0.0,
                "Loss/expert_advantage_weighted": 0.0,
            })
        )
        loss = loss + advantage_loss
        status.update(advantage_status)

        model = self.net.module if hasattr(self.net, "module") else self.net
        if self.expert_enabled and self.expert_phase == "specialize":
            gt_xyxy = box_xywh_to_xyxy(
                gt_dict["search_anno"][-1].to(training_expert_ids.device))
            pred_xyxy = box_cxcywh_to_xyxy(
                pred_dict["pred_boxes"][:, 0]).detach()
            present = self._present_search_mask(
                gt_dict, training_expert_ids.device,
                training_expert_ids.numel())
            if present is None:
                present = torch.ones_like(
                    training_expert_ids, dtype=torch.bool)
            owned_iou = box_iou(pred_xyxy, gt_xyxy)[0]
            for expert_id in range(len(model.expert_names)):
                trained = present & (training_expert_ids == expert_id)
                status[f"Expert/train_count_{expert_id}"] = int(trained.sum())
                status[f"Expert/train_iou_{expert_id}"] = (
                    owned_iou[trained].mean().item() if trained.any() else 0.0)

        if not return_status:
            return loss
        status["Loss/base"] = base_loss.item()
        status["Loss/SRBT"] = srbt_loss.item()
        status.setdefault(
            "Loss/presence_threshold", presence_threshold_loss.item())
        status.setdefault("Loss/redetect", redetect_loss.item())
        status["Expert/phase_id"] = {
            "specialize": 0, "refine": 1, "recovery": 2, "pursuit": 3,
            "dispatch": 4,
        }[self.expert_phase]
        status.setdefault("Redetect/count", 0)
        status["Loss/total"] = loss.item()
        return loss, status

    def _dispatch_targets(self, pred_dict, gt_dict):
        logits = pred_dict.get("expert_activation_logits")
        expert_outputs = pred_dict.get("expert_outputs")
        if logits is None or not expert_outputs:
            raise RuntimeError(
                "dispatch training requires activation logits and all expert outputs")
        model = self.net.module if hasattr(self.net, "module") else self.net
        expert_names = tuple(model.expert_names)
        if logits.ndim != 2 or logits.shape[1] != len(expert_names) - 1:
            raise ValueError(
                "expert activation logits must provide one value per specialist")
        missing = [name for name in expert_names if name not in expert_outputs]
        if missing:
            raise RuntimeError(
                "dispatch training is missing expert outputs: "
                + ", ".join(missing))

        challenge_labels = gt_dict.get("challenge_labels")
        if challenge_labels is None:
            raise RuntimeError(
                "dispatch training requires challenge_labels")
        challenge_labels = torch.as_tensor(
            challenge_labels, device=logits.device, dtype=torch.bool)
        batch_size = logits.shape[0]
        loader_shape = (len(CHALLENGE_NAMES), batch_size)
        if challenge_labels.shape == loader_shape:
            challenge_labels = challenge_labels.transpose(0, 1)
        if challenge_labels.shape != (batch_size, len(CHALLENGE_NAMES)):
            raise ValueError(
                "challenge_labels must have shape "
                f"({batch_size}, {len(CHALLENGE_NAMES)})")
        attributes = {
            name: challenge_labels[:, index]
            for index, name in enumerate(CHALLENGE_NAMES)
        }
        eligible = expert_supervision_mask(attributes).to(logits.device)
        present = self._present_search_mask(
            gt_dict, logits.device, batch_size)
        if present is None:
            raise RuntimeError(
                "dispatch training requires official presence annotations")

        target_xyxy = box_xywh_to_xyxy(
            gt_dict["search_anno"][-1].to(
                device=logits.device, dtype=logits.dtype)
        ).clamp(0.0, 1.0)

        def aligned_iou(output):
            boxes = output["pred_boxes"][:, 0].detach()
            predicted_xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
            return box_iou(predicted_xyxy, target_xyxy)[0]

        generalist_iou = aligned_iou(expert_outputs[expert_names[0]])
        targets = torch.zeros_like(logits, dtype=torch.bool)
        for expert_id, name in enumerate(expert_names[1:], start=1):
            specialist_target = eligible[:, expert_id]
            if expert_id == VISIBILITY_EXPERT_ID:
                targets[:, expert_id - 1] = specialist_target
                continue
            specialist_iou = aligned_iou(expert_outputs[name])
            useful = specialist_iou >= (
                generalist_iou + self.activation_advantage_margin)
            targets[:, expert_id - 1] = specialist_target & present & useful
        return targets

    def _compute_dispatch_loss(self, pred_dict, gt_dict, return_status=True):
        logits = pred_dict["expert_activation_logits"].float()
        targets = self._dispatch_targets(pred_dict, gt_dict)
        pos_weight = logits.new_tensor(self.activation_pos_weight)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.to(logits.dtype), pos_weight=pos_weight)
        if not return_status:
            return loss
        model = self.net.module if hasattr(self.net, "module") else self.net
        predicted = logits.sigmoid() >= 0.5
        exact_match = (predicted == targets).all(dim=1).float().mean()
        status = {
            "Loss/activation": float(loss.detach()),
            "Loss/total": float(loss.detach()),
            "Activation/accuracy": float((predicted == targets).float().mean()),
            "Activation/positive_count": int(targets.sum()),
            "Activation/predicted_count": int(predicted.sum()),
            "Activation/mean_specialists": float(
                predicted.float().sum(dim=1).mean()),
            "Activation/exact_match": float(exact_match),
            "Expert/phase_id": 4,
        }
        expert_f1 = []
        for expert_id, name in enumerate(model.expert_names[1:], start=1):
            target = targets[:, expert_id - 1]
            prediction = predicted[:, expert_id - 1]
            true_positive = (target & prediction).float().sum()
            precision = true_positive / prediction.float().sum().clamp_min(1.0)
            recall = true_positive / target.float().sum().clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
            status[f"Activation/positive_{name}"] = int(target.sum())
            status[f"Activation/precision_{name}"] = float(precision)
            status[f"Activation/recall_{name}"] = float(recall)
            status[f"Activation/f1_{name}"] = float(f1)
            expert_f1.append(f1)
        status["Activation/macro_f1"] = float(torch.stack(expert_f1).mean())
        return loss, status

    def _compute_pursuit_losses(self, predictions, return_status=True):
        steps = predictions.get("pursuit_predictions", ())
        if not steps:
            raise RuntimeError("pursuit forward pass produced no causal steps")
        losses = []
        statuses = []
        search_factor = float(getattr(
            self.cfg.DATA.SEARCH, "FACTOR",
            self.settings.search_area_factor["search"]))
        train_cfg = self.cfg.TRAIN
        for index, prediction in enumerate(steps):
            step_loss, step_status = search_window_pursuit_loss(
                prediction,
                target_next=predictions["pursuit_targets"][index],
                current_inside=predictions[
                    "pursuit_current_inside"][index],
                current_quality=predictions[
                    "pursuit_current_quality"][index],
                present_next=predictions[
                    "pursuit_present_next"][index],
                search_factor=search_factor,
                center_weight=float(getattr(
                    train_cfg, "PURSUIT_CENTER_WEIGHT", 1.0)),
                scale_weight=float(getattr(
                    train_cfg, "PURSUIT_SCALE_WEIGHT", 0.5)),
                containment_weight=float(getattr(
                    train_cfg, "PURSUIT_CONTAINMENT_WEIGHT", 2.0)),
                inside_weight=float(getattr(
                    train_cfg, "PURSUIT_INSIDE_WEIGHT", 0.5)),
                quality_weight=float(getattr(
                    train_cfg, "PURSUIT_QUALITY_WEIGHT", 0.25)),
            )
            losses.append(step_loss)
            statuses.append(step_status)
        loss = torch.stack(losses).mean()
        if not return_status:
            return loss
        status = {
            key: sum(item[key] for item in statuses) / len(statuses)
            for key in statuses[0]
            if key.startswith("Loss/")
        }
        status["Loss/total"] = float(loss.detach())
        status["Pursuit/steps"] = len(steps)
        status["Pursuit/in_crop_rate"] = float(torch.cat([
            item.float() for item in predictions[
                "pursuit_current_inside"]
        ]).mean())
        next_present = torch.cat([
            item.bool() for item in predictions["pursuit_present_next"]])
        next_inside = torch.cat([
            crop_target_inside(
                predictions["pursuit_targets"][index],
                prediction.next_box,
                search_factor,
            )
            for index, prediction in enumerate(steps)
        ])
        visible_count = next_present.float().sum().clamp_min(1.0)
        status["Pursuit/next_in_crop_rate"] = float(
            (next_inside & next_present).float().sum() / visible_count)
        predicted_centers = torch.cat([
            prediction.next_box[:, :2] + 0.5 * prediction.next_box[:, 2:]
            for prediction in steps
        ])
        target_centers = torch.cat([
            target[:, :2] + 0.5 * target[:, 2:]
            for target in predictions["pursuit_targets"]
        ])
        center_error = torch.linalg.vector_norm(
            predicted_centers - target_centers, dim=1)
        status["Pursuit/next_center_error"] = float(
            ((center_error * next_present.float()).sum()
             / visible_count).detach())
        status["Pursuit/outside_count"] = sum(
            item["Pursuit/outside_count"] for item in statuses)
        status["Pursuit/quality_count"] = sum(
            item["Pursuit/quality_count"] for item in statuses)
        return loss, status

    def _compute_expert_advantage_loss(
            self, pred_dict, gt_dict, training_expert_ids):
        """Require proposal-based specialists to beat their frozen parent."""
        predicted = pred_dict["pred_boxes"][:, 0]
        zero = predicted.sum() * 0.0
        model = self.net.module if hasattr(self.net, "module") else self.net
        status = {
            "Loss/expert_advantage": 0.0,
            "Loss/expert_advantage_weighted": 0.0,
        }
        for expert_id in ADVANTAGE_EXPERT_IDS:
            status[
                f"Loss/expert_advantage_{model.expert_names[expert_id]}"
            ] = 0.0

        present = self._present_search_mask(
            gt_dict, predicted.device, predicted.shape[0])
        if present is None:
            present = torch.ones_like(training_expert_ids, dtype=torch.bool)
        active = present & torch.stack([
            training_expert_ids == expert_id
            for expert_id in ADVANTAGE_EXPERT_IDS
        ]).any(dim=0)
        if not active.any():
            return zero, status

        upstream = pred_dict.get("upstream_pred_boxes")
        if upstream is None:
            raise RuntimeError(
                "proposal-based specialist training requires "
                "upstream_pred_boxes")
        upstream = torch.as_tensor(
            upstream, device=predicted.device, dtype=predicted.dtype)
        if upstream.shape != pred_dict["pred_boxes"].shape:
            raise ValueError(
                "upstream_pred_boxes must match pred_boxes shape")

        target = gt_dict["search_anno"][-1].to(
            device=predicted.device, dtype=predicted.dtype)
        predicted_xyxy = box_cxcywh_to_xyxy(predicted[active]).clamp(0.0, 1.0)
        upstream_xyxy = box_cxcywh_to_xyxy(
            upstream[:, 0].detach()[active]).clamp(0.0, 1.0)
        target_xyxy = box_xywh_to_xyxy(target[active]).clamp(0.0, 1.0)
        predicted_giou, _ = generalized_box_iou(
            predicted_xyxy, target_xyxy)
        with torch.no_grad():
            upstream_giou, _ = generalized_box_iou(
                upstream_xyxy, target_xyxy)
            required_giou = (
                upstream_giou + self.expert_advantage_margin).clamp(max=1.0)
        per_sample = F.relu(required_giou - predicted_giou)
        raw_loss = per_sample.mean()
        weighted_loss = self.expert_advantage_weight * raw_loss
        status["Loss/expert_advantage"] = raw_loss.item()
        status["Loss/expert_advantage_weighted"] = weighted_loss.item()
        for expert_id in ADVANTAGE_EXPERT_IDS:
            owned = active & (training_expert_ids == expert_id)
            if owned.any():
                active_owned = training_expert_ids[active] == expert_id
                status[
                    f"Loss/expert_advantage_{model.expert_names[expert_id]}"
                ] = per_sample[active_owned].mean().item()
        return weighted_loss, status

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
