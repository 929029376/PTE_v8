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
    exclusive_specialist_supervision_mask,
    expert_supervision_mask,
)
from lib.models.layers.expert_ensemble import normalized_response_psr
from lib.models.layers.srbt_controller import Action
from lib.models.layers.search_window_controller import (
    crop_box_to_image,
    crop_target_inside,
    dynamic_search_crop,
    event_motion_centroid,
    relative_box_motion,
    search_window_pursuit_loss,
)


REQUIRED_PRESENCE_PREDICTIONS = frozenset({"logits", "score"})
VISIBILITY_EXPERT_ID = 3
MOTION_EXPERT_ID = 1
DISCRIMINATION_EXPERT_ID = 4
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
            controller_cfg, "THETA_OBSERVABLE", 0.70))
        self.presence_recover_threshold = float(getattr(
            controller_cfg, "THETA_RECOVER", 0.75))
        recovery_loss_cfg = getattr(cfg.TRAIN, "RECOVERY_LOSS", None)
        self.identity_loss_weight = float(getattr(
            recovery_loss_cfg, "IDENTITY_WEIGHT", 1.0))
        self.identity_ranking_weight = float(getattr(
            recovery_loss_cfg, "RANKING_WEIGHT", 0.5))
        self.identity_ranking_margin = float(getattr(
            recovery_loss_cfg, "RANKING_MARGIN", 0.2))
        dart_loss_cfg = getattr(cfg.TRAIN, "DART_LOSS", None)
        self.dart_coverage_weight = float(getattr(
            dart_loss_cfg, "COVERAGE_WEIGHT", 1.0))
        self.dart_geometry_weight = float(getattr(
            dart_loss_cfg, "GEOMETRY_WEIGHT", 1.0))
        self.dart_ranking_weight = float(getattr(
            dart_loss_cfg, "RANKING_WEIGHT", 0.5))
        self.dart_ranking_margin = float(getattr(
            dart_loss_cfg, "RANKING_MARGIN", 0.2))
        self.dart_decoder_weight = float(getattr(
            dart_loss_cfg, "DECODER_WEIGHT", 1.0))
        self.dart_local_duration = int(getattr(
            controller_cfg, "LOCAL_DURATION", 2))
        self.dart_global_duration = int(getattr(
            controller_cfg, "GLOBAL_DURATION", 4))
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
        self.dart_decoder_only = bool(getattr(
            cfg.TRAIN, "DART_DECODER_ONLY", False))
        self.proposal_identity_only = bool(getattr(
            cfg.TRAIN, "PROPOSAL_IDENTITY_ONLY", False))
        if self.dart_decoder_only and self.proposal_identity_only:
            raise ValueError(
                "DART_DECODER_ONLY and PROPOSAL_IDENTITY_ONLY are mutually exclusive")
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
            {"proposal_identity"}
            if (self.expert_phase == "recovery"
                and self.proposal_identity_only) else
            {"presence", "reliability", "redetect", "identity"}
            if self.expert_phase == "recovery"
            else ({"base", "expert_advantage", "srbt", "redetect"}
                  if self.srbt_enabled
                  else {"base", "expert_advantage"})
        )

    # ------------------------------------------------------------------ #
    def train(self, mode=True):
        sparse_phases = {"specialize", "refine", "pursuit"}
        if self.expert_phase not in sparse_phases | {"dispatch", "recovery"}:
            return super().train(mode)
        self.net.eval()
        model = self.net.module if hasattr(self.net, "module") else self.net
        if self.expert_phase in sparse_phases:
            if mode:
                for module in model.modules():
                    parameters = tuple(module.parameters())
                    if (parameters
                            and all(parameter.requires_grad
                                    for parameter in parameters)):
                        module.train(True)
            return
        if self.expert_phase == "dispatch":
            model.expert_activator.train(mode)
            return
        if getattr(self, "dart_decoder_only", False):
            model.duration_evidence_decoder.train(mode)
            return
        if getattr(self, "proposal_identity_only", False):
            model.rgb_identity_verifier.train(mode)
            return
        for name in (
                "visibility_gate", "localization_validity_gate",
                "duration_evidence_decoder", "rgb_identity_verifier",
                "redetect_expert"):
            module = getattr(model, name, None)
            if module is not None:
                module.train(mode)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _uses_exclusive_specialist_supervision(data):
        flags = torch.as_tensor(
            data.get("exclusive_specialist_supervision", True),
            dtype=torch.bool,
        ).reshape(-1)
        if flags.numel() == 0 or not bool((flags == flags[0]).all()):
            raise ValueError(
                "exclusive_specialist_supervision must be uniform per batch")
        return bool(flags[0])

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
        supervision_mask = (
            exclusive_specialist_supervision_mask
            if self._uses_exclusive_specialist_supervision(data)
            else expert_supervision_mask
        )
        eligible = supervision_mask(attributes).to(device)
        selected = eligible.gather(1, training_expert_ids[:, None]).squeeze(1)
        if not bool(selected.all()):
            raise ValueError(
                "training expert is not eligible for exclusive frame "
                "challenge supervision")
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

    @staticmethod
    def _batch_first_padding_masks(value, batch_size, device):
        if value is None:
            return None
        value = torch.as_tensor(value, device=device, dtype=torch.bool)
        if (value.ndim == 4 and value.shape[0] != batch_size
                and value.shape[1] == batch_size):
            value = value.permute(1, 0, 2, 3)
        if value.ndim != 4 or value.shape[0] != batch_size:
            raise ValueError(
                "padding masks must have shape (B,T,H,W) or (T,B,H,W)")
        return value.contiguous()

    @staticmethod
    def _cover_target_observation(rgb, event, boxes_xywh, present):
        if rgb.ndim != 5 or event.shape != rgb.shape:
            raise ValueError("RGB and event observations must share shape (B,T,C,H,W)")
        batch_size, _, _, height, width = rgb.shape
        boxes = torch.as_tensor(
            boxes_xywh, device=rgb.device, dtype=rgb.dtype).reshape(-1, 4)
        present = torch.as_tensor(
            present, device=rgb.device, dtype=torch.bool).reshape(-1)
        if boxes.shape[0] != batch_size or present.numel() != batch_size:
            raise ValueError("coverage intervention requires one box and flag per row")

        y = (torch.arange(height, device=rgb.device, dtype=rgb.dtype) + 0.5) / height
        x = (torch.arange(width, device=rgb.device, dtype=rgb.dtype) + 0.5) / width
        x1 = boxes[:, 0, None, None]
        y1 = boxes[:, 1, None, None]
        x2 = (boxes[:, 0] + boxes[:, 2])[:, None, None]
        y2 = (boxes[:, 1] + boxes[:, 3])[:, None, None]
        target_mask = (
            (x[None, None, :] >= x1)
            & (x[None, None, :] < x2)
            & (y[None, :, None] >= y1)
            & (y[None, :, None] < y2)
            & present[:, None, None]
        )[:, None]

        covered_rgb = rgb.clone()
        covered_event = event.clone()
        current_rgb = covered_rgb[:, -1]
        current_event = covered_event[:, -1]
        fill = current_rgb.mean(dim=(-2, -1), keepdim=True)
        covered_rgb[:, -1] = torch.where(target_mask, fill, current_rgb)
        covered_event[:, -1] = torch.where(
            target_mask, torch.zeros_like(current_event), current_event)
        return covered_rgb, covered_event

    @staticmethod
    def _geometry_candidate_pair(boxes_xywh):
        positive = boxes_xywh.clone()
        positive[:, :2] = (
            boxes_xywh[:, :2] + 0.5 * boxes_xywh[:, 2:])
        negative = positive.clone()
        negative[:, :2] = torch.remainder(positive[:, :2] + 0.5, 1.0)
        return positive, negative

    @staticmethod
    def _reliability_score_map(output):
        return output.get("expert_outputs", {}).get(
            "visibility_foc_ov", output)["score_map"]

    def _forward_reliability_interventions(
            self, output, forward_kwargs, data):
        batch_size = forward_kwargs["xi"].shape[0]
        device = forward_kwargs["xi"].device
        present = self._present_search_mask(data, device, batch_size)
        if present is None:
            raise RuntimeError(
                "reliability intervention training requires official presence")
        boxes = torch.as_tensor(
            data["search_anno"], device=device,
            dtype=forward_kwargs["xi"].dtype)
        if boxes.ndim == 3:
            boxes = boxes[-1]
        if boxes.shape != (batch_size, 4):
            raise ValueError("search_anno must provide one current box per row")

        covered_rgb, covered_event = self._cover_target_observation(
            forward_kwargs["xi"], forward_kwargs["xe"], boxes, present)
        covered_kwargs = dict(forward_kwargs)
        covered_kwargs.update({
            "xi": covered_rgb,
            "xe": covered_event,
            "redetect_images": None,
            "redetect_event_images": None,
            "redetect_mask": None,
        })
        covered_output = self.net(**covered_kwargs)

        model = self.net.module if hasattr(self.net, "module") else self.net
        positive_boxes, negative_boxes = self._geometry_candidate_pair(boxes)
        score_map = self._reliability_score_map(output)
        clean_observability_logits = output[
            "presence_predictions"]["logits"]
        coverage_observability_logits = covered_output[
            "presence_predictions"]["logits"]
        positive_localization_logits = model.localization_validity_gate(
            score_map, positive_boxes)
        negative_localization_logits = model.localization_validity_gate(
            score_map, negative_boxes)
        duration_decoder = getattr(
            model, "duration_evidence_decoder", None)
        if duration_decoder is None:
            raise RuntimeError(
                "recovery training requires duration_evidence_decoder")
        clean_observability = clean_observability_logits.softmax(
            dim=-1)[:, 1]
        covered_observability = coverage_observability_logits.softmax(
            dim=-1)[:, 1]
        positive_localization = positive_localization_logits.softmax(
            dim=-1)[:, 1]
        negative_localization = negative_localization_logits.softmax(
            dim=-1)[:, 1]
        state_ids = {action: index for index, action in enumerate(Action)}
        batch_state = lambda action: torch.full(
            (batch_size,), state_ids[action], device=device,
            dtype=torch.long)
        batch_duration = lambda value: torch.full(
            (batch_size,), float(value), device=device,
            dtype=clean_observability.dtype)
        unknown_identity = clean_observability.new_full((batch_size,), -1.0)
        verified_identity = clean_observability.new_ones(batch_size)
        duration_state_logits = torch.stack((
            duration_decoder(
                clean_observability, positive_localization,
                unknown_identity, batch_state(Action.TRACK),
                batch_duration(1)),
            duration_decoder(
                clean_observability, negative_localization,
                unknown_identity, batch_state(Action.TRACK),
                batch_duration(getattr(self, "dart_local_duration", 2))),
            duration_decoder(
                covered_observability, positive_localization,
                unknown_identity, batch_state(Action.LOCAL_UNRESOLVED),
                batch_duration(getattr(self, "dart_global_duration", 4))),
            duration_decoder(
                clean_observability, positive_localization,
                verified_identity, batch_state(Action.GLOBAL_UNRESOLVED),
                batch_duration(1)),
        ), dim=0)
        duration_state_targets = torch.stack(tuple(
            batch_state(action) for action in Action), dim=0)
        return {
            "clean_observability_logits": clean_observability_logits,
            "coverage_observability_logits": coverage_observability_logits,
            "positive_localization_logits": positive_localization_logits,
            "negative_localization_logits": negative_localization_logits,
            "duration_state_logits": duration_state_logits,
            "duration_state_targets": duration_state_targets,
        }

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
        redetect_boxes = None
        redetect_padding_mask = None
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
                redetect_padding_mask = self._batch_first_padding_masks(
                    data.get("redetect_search_att"), xi.shape[0], xi.device)
                if redetect_padding_mask is None:
                    raise RuntimeError(
                        "proposal-aligned recovery requires redetect_search_att")
                redetect_annotations = data.get("redetect_search_anno")
                if redetect_annotations is None:
                    raise RuntimeError(
                        "proposal-aligned recovery requires redetect_search_anno")
                redetect_boxes = torch.as_tensor(
                    redetect_annotations,
                    device=xi.device,
                    dtype=xi.dtype,
                )
                if redetect_boxes.ndim == 3:
                    redetect_boxes = redetect_boxes[-1]
                if redetect_boxes.shape != (xi.shape[0], 4):
                    raise ValueError(
                        "redetect_search_anno must provide one current box per row")
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
            "redetect_boxes": redetect_boxes,
            "redetect_padding_mask": redetect_padding_mask,
        }
        if training_expert_ids is not None:
            forward_kwargs["training_expert_ids"] = training_expert_ids
        if expert_phase == "dispatch":
            forward_kwargs["return_activation_logits"] = True
        out_dict = self.net(**forward_kwargs)
        if (expert_phase == "recovery"
                and not getattr(self, "proposal_identity_only", False)):
            out_dict["reliability_interventions"] = (
                self._forward_reliability_interventions(
                    out_dict, forward_kwargs, data))

        return out_dict

    def _pursuit_specialist_context(self, data, model, batch_size,
                                    frame_count, device):
        train_cfg = getattr(self.cfg, "TRAIN", None)
        specialist_ids = tuple(int(expert_id) for expert_id in getattr(
            train_cfg, "SPECIALIST_EXPERT_IDS", ()))
        if len(specialist_ids) != 1:
            return None, None, None
        specialist_id = specialist_ids[0]
        if (not getattr(self, "expert_enabled", False)
                or specialist_id <= 0
                or specialist_id >= len(model.expert_names)):
            raise ValueError("causal pursuit specialist ID is invalid")
        sampled_ids = data.get("training_expert_id")
        challenge_labels = data.get("pursuit_challenge_labels")
        if sampled_ids is None or challenge_labels is None:
            raise RuntimeError(
                "causal specialist pursuit requires frame challenge labels")
        sampled_ids = torch.as_tensor(
            sampled_ids, device=device, dtype=torch.long).reshape(-1)
        if (sampled_ids.numel() != batch_size
                or not bool((sampled_ids == specialist_id).all())):
            raise ValueError(
                "pursuit batch must contain only the declared specialist")
        challenge_labels = torch.as_tensor(
            challenge_labels, device=device, dtype=torch.bool)
        expected = (batch_size, frame_count, len(CHALLENGE_NAMES))
        if challenge_labels.shape == (
                frame_count, len(CHALLENGE_NAMES), batch_size):
            challenge_labels = challenge_labels.permute(2, 0, 1)
        elif challenge_labels.shape == (
                frame_count, batch_size, len(CHALLENGE_NAMES)):
            challenge_labels = challenge_labels.permute(1, 0, 2)
        if challenge_labels.shape != expected:
            raise ValueError(
                "pursuit_challenge_labels must describe every batch frame")
        attributes = {
            name: challenge_labels[..., index].reshape(-1)
            for index, name in enumerate(CHALLENGE_NAMES)
        }
        supervision_mask = (
            exclusive_specialist_supervision_mask
            if self._uses_exclusive_specialist_supervision(data)
            else expert_supervision_mask
        )
        eligible = supervision_mask(attributes)[
            :, specialist_id].reshape(batch_size, frame_count)
        if not bool(eligible.any(dim=1).all()):
            raise ValueError(
                "each pursuit episode must contain eligible specialist frames")
        return specialist_id, eligible, challenge_labels

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

    @staticmethod
    def _discrimination_ranking_loss(
            score_map, target_boxes, present, margin):
        if score_map.ndim != 4 or score_map.shape[1] != 1:
            raise ValueError(
                "discrimination score_map must have shape [B, 1, H, W]")
        target_boxes = torch.as_tensor(
            target_boxes, device=score_map.device, dtype=score_map.dtype)
        present = torch.as_tensor(
            present, device=score_map.device, dtype=torch.bool).reshape(-1)
        if target_boxes.shape != (score_map.shape[0], 4) \
                or present.numel() != score_map.shape[0]:
            raise ValueError(
                "discrimination targets must match the score-map batch")

        height, width = score_map.shape[-2:]
        centers = target_boxes[:, :2] + 0.5 * target_boxes[:, 2:]
        center_x = (centers[:, 0] * width).round().long()
        center_y = (centers[:, 1] * height).round().long()
        radius_x = torch.ceil(
            0.5 * target_boxes[:, 2] * width).long().clamp_min(1)
        radius_y = torch.ceil(
            0.5 * target_boxes[:, 3] * height).long().clamp_min(1)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=score_map.device),
            torch.arange(width, device=score_map.device),
            indexing="ij",
        )
        target_region = (
            (grid_x[None] - center_x[:, None, None]).abs()
            <= radius_x[:, None, None]
        ) & (
            (grid_y[None] - center_y[:, None, None]).abs()
            <= radius_y[:, None, None]
        )
        valid = present \
            & (center_x >= 0) & (center_x < width) \
            & (center_y >= 0) & (center_y < height) \
            & (~target_region).flatten(1).any(dim=1)
        zero = score_map.float().sum() * 0.0
        if not bool(valid.any()):
            return zero, zero.detach(), zero.detach(), 0, zero.detach()

        scores = score_map[:, 0].float()
        target_scores = scores.masked_fill(~target_region, float("-inf"))
        distractor_scores = scores.masked_fill(target_region, float("-inf"))
        positive = target_scores.flatten(1).max(dim=1).values[valid]
        hardest_negative = distractor_scores.flatten(1).max(dim=1).values[valid]
        positive_logits = torch.logit(positive.clamp(1e-4, 1.0 - 1e-4))
        negative_logits = torch.logit(
            hardest_negative.clamp(1e-4, 1.0 - 1e-4))
        logit_gap = positive_logits - negative_logits
        ranking_loss = F.softplus(float(margin) - logit_gap)
        return (
            ranking_loss.mean(),
            positive.mean().detach(),
            hardest_negative.mean().detach(),
            int(valid.sum()),
            (logit_gap < float(margin)).float().mean().detach(),
        )

    def _forward_pursuit(self, data):
        """Unroll controller or one specialist over prediction-driven crops."""
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
        specialist_id, specialist_eligible, specialist_challenge_labels = (
            self._pursuit_specialist_context(
                data, model, frames.shape[0], frames.shape[1], frames.device)
        )
        specialist_name = (
            model.expert_names[specialist_id]
            if specialist_id is not None else None)
        search_factor = float(getattr(
            self.cfg.DATA.SEARCH, "FACTOR",
            self.settings.search_area_factor["search"]))
        search_size = int(self.cfg.DATA.SEARCH.SIZE)
        planned_anchor = annotations[:, 0].detach()
        previous_observation = planned_anchor
        previous_event_search = None
        predictions = []
        targets = []
        current_inside_values = []
        current_quality_values = []
        present_next_values = []
        crop_anchors = []
        specialist_outputs = []
        specialist_targets = []
        specialist_present = []
        specialist_discrimination_present = []
        specialist_image_boxes = []
        specialist_image_targets = []
        expert_cfg = getattr(
            getattr(self.cfg, "MODEL", None), "EXPERT", None)
        use_activation = bool(getattr(
            expert_cfg, "USE_ACTIVATION_INFERENCE", False))
        if specialist_id is not None and use_activation:
            raise RuntimeError(
                "causal specialist pursuit uses the declared expert directly")
        if use_activation and not bool(getattr(
                expert_cfg, "ACTIVATOR_TRAINED", False)):
            raise RuntimeError(
                "pursuit sparse activation requires a trained expert activator")
        was_training = model.training
        motion_center_jitter = 0.0
        motion_center_jitter_multiplier = float(getattr(
            self.settings, "motion_center_jitter_multiplier", 1.0))
        if specialist_id == 1 and was_training \
                and motion_center_jitter_multiplier > 1.0:
            motion_center_jitter = float(getattr(
                self.settings, "center_jitter_factor", {}).get(
                    "search", 0.0)) * motion_center_jitter_multiplier
        model.eval()
        controller.train(was_training and specialist_id is None)
        try:
            encoded_templates = None
            if specialist_id is not None and hasattr(
                    model, "_encode_runtime_templates"):
                with torch.no_grad():
                    encoded_templates = model._encode_runtime_templates(
                        zi[:, 0], ze[:, 0], zi[:, 1:], ze[:, 1:])
            small_template_features = None
            if specialist_name is not None and specialist_name == getattr(
                    model, "precision_refiner_name", None):
                small_target_expert = getattr(
                    model, "small_target_expert", None)
                if small_target_expert is None:
                    raise RuntimeError(
                        "precision pursuit requires small_target_expert")
                small_template_features = small_target_expert.encode_template(
                    zi[:, 0], ze[:, 0])
            for frame_index in range(frames.shape[1] - 1):
                crop_anchor = planned_anchor
                if motion_center_jitter > 0.0:
                    max_offset = (
                        planned_anchor[:, 2:].prod(dim=1, keepdim=True).sqrt()
                        * motion_center_jitter)
                    crop_anchor = planned_anchor.clone()
                    crop_anchor[:, :2] += max_offset * (
                        torch.rand_like(crop_anchor[:, :2]) - 0.5)
                    target = annotations[:, frame_index]
                    side = (
                        planned_anchor[:, 2:].prod(dim=1).sqrt()
                        * search_factor)
                    slack = (
                        side[:, None] - target[:, 2:]).clamp_min(0.0)
                    margin = torch.minimum(
                        side[:, None] / search_size, 0.25 * slack)
                    min_center = (
                        target[:, :2] + target[:, 2:] + margin
                        - 0.5 * side[:, None])
                    max_center = (
                        target[:, :2] - margin
                        + 0.5 * side[:, None])
                    desired_center = (
                        crop_anchor[:, :2] + 0.5 * crop_anchor[:, 2:])
                    bounded_center = torch.maximum(
                        torch.minimum(desired_center, max_center), min_center)
                    nominal_inside = crop_target_inside(
                        target, planned_anchor, search_factor
                    ) & present[:, frame_index]
                    crop_anchor[:, :2] = torch.where(
                        nominal_inside[:, None],
                        bounded_center - 0.5 * crop_anchor[:, 2:],
                        planned_anchor[:, :2],
                    )
                crop_anchors.append(crop_anchor)
                search, crop_region = dynamic_search_crop(
                    frames[:, frame_index], crop_anchor,
                    search_factor, search_size)
                event_search, _ = dynamic_search_crop(
                    event_frames[:, frame_index], crop_anchor,
                    search_factor, search_size)
                history_valid = previous_event_search is not None
                motion_context = {
                    "current_event": event_search,
                    "previous_event": (
                        event_search
                        if previous_event_search is None
                        else previous_event_search),
                    "history_valid": torch.full(
                        (search.shape[0],), history_valid,
                        device=search.device, dtype=torch.bool),
                    "box_delta": relative_box_motion(
                        planned_anchor, previous_observation),
                }
                with torch.set_grad_enabled(specialist_id is not None):
                    if encoded_templates is None:
                        runtime_templates = (
                            zi[:, 0], ze[:, 0], zi[:, 1:], ze[:, 1:])
                    else:
                        runtime_templates = encoded_templates
                    inference_kwargs = dict(
                        static_zi=runtime_templates[0],
                        static_ze=runtime_templates[1],
                        dynamic_zi=runtime_templates[2],
                        dynamic_ze=runtime_templates[3],
                        xi=search, xe=event_search,
                        motion_context=motion_context)
                    if small_template_features is not None:
                        inference_kwargs["small_template_features"] = (
                            small_template_features)
                    if specialist_id is not None:
                        inference_kwargs["active_expert_names"] = (
                            specialist_name,)
                    elif use_activation:
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
                    if specialist_id is not None:
                        active_mask = torch.zeros(
                            search.shape[0], len(model.expert_names),
                            device=search.device, dtype=torch.bool)
                        active_mask[:, 0] = True
                        active_mask[:, specialist_id] = True
                    elif use_activation:
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
                        expert_boxes[:, specialist_id]
                        if specialist_id is not None
                        else (expert_weights[..., None] * expert_boxes).sum(
                            dim=1)
                    )
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
                if specialist_id is not None:
                    # Specialist stages must close the loop with the same
                    # candidate that receives their localization loss.
                    prediction.next_box = observation.detach()
                current_inside = crop_target_inside(
                    annotations[:, frame_index], crop_anchor,
                    search_factor) & present[:, frame_index]
                current_quality = self._aligned_iou_xywh(
                    observation, annotations[:, frame_index])
                if specialist_id is not None:
                    region_xy = crop_region[:, :2]
                    region_wh = crop_region[:, 2:]
                    target_size = (
                        annotations[:, frame_index, 2:] / region_wh)
                    target_center = (
                        annotations[:, frame_index, :2]
                        + 0.5 * annotations[:, frame_index, 2:]
                        - region_xy
                    ) / region_wh
                    specialist_targets.append(torch.cat((
                        target_center - 0.5 * target_size,
                        target_size,
                    ), dim=1))
                    specialist_present.append(
                        present[:, frame_index]
                        & specialist_eligible[:, frame_index])
                    if specialist_id == DISCRIMINATION_EXPERT_ID:
                        ambiguity_index = CHALLENGE_NAMES.index("ambiguity")
                        specialist_discrimination_present.append(
                            present[:, frame_index]
                            & specialist_eligible[:, frame_index]
                            & specialist_challenge_labels[
                                :, frame_index, ambiguity_index])
                    specialist_outputs.append(
                        expert_outputs[specialist_name])
                    specialist_image_boxes.append(observation)
                    specialist_image_targets.append(
                        annotations[:, frame_index])
                predictions.append(prediction)
                targets.append(annotations[:, frame_index + 1])
                current_inside_values.append(current_inside)
                current_quality_values.append(current_quality)
                present_next_values.append(present[:, frame_index + 1])
                previous_observation = observation.detach()
                planned_anchor = prediction.next_box.detach()
                previous_event_search = event_search.detach()
        finally:
            model.train(was_training)
        return {
            "pursuit_predictions": predictions,
            "pursuit_targets": targets,
            "pursuit_current_inside": current_inside_values,
            "pursuit_current_quality": current_quality_values,
            "pursuit_present_next": present_next_values,
            "pursuit_crop_anchors": crop_anchors,
            "pursuit_specialist_id": specialist_id,
            "pursuit_specialist_outputs": specialist_outputs,
            "pursuit_specialist_targets": specialist_targets,
            "pursuit_specialist_present": specialist_present,
            "pursuit_discrimination_present": (
                specialist_discrimination_present),
            "pursuit_specialist_image_boxes": specialist_image_boxes,
            "pursuit_specialist_image_targets": specialist_image_targets,
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
        if (self.expert_phase == "recovery"
                and getattr(self, "proposal_identity_only", False)):
            predictions = pred_dict.get("redetect_predictions")
            if predictions is None:
                raise RuntimeError(
                    "proposal identity training requires reappearance candidates")
            loss, status = self._compute_proposal_identity_loss(predictions)
            if not return_status:
                return loss
            status.update({
                "Loss/base": 0.0,
                "Loss/SRBT": 0.0,
                "Loss/presence": 0.0,
                "Loss/presence_threshold": 0.0,
                "Loss/redetect": loss.item(),
                "Loss/DART": 0.0,
                "Loss/dart_decoder": 0.0,
                "DART/decoder_acc": 0.0,
                "Expert/phase_id": 2,
                "Redetect/count": int(
                    predictions["batch_indices"].numel()),
                "Loss/total": loss.item(),
            })
            return loss, status
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

        dart_loss = base_loss * 0.0
        if self.expert_enabled and self.expert_phase == "recovery":
            interventions = pred_dict.get("reliability_interventions")
            if interventions is None:
                raise RuntimeError(
                    "recovery training requires paired reliability interventions")
            present = self._present_search_mask(
                gt_dict, pred_dict["pred_boxes"].device,
                pred_dict["pred_boxes"].shape[0])
            if present is None:
                raise RuntimeError(
                    "reliability intervention training requires official presence")
            dart_loss, dart_status = (
                self._compute_reliability_intervention_loss(
                    interventions, present))
            loss = loss + dart_loss
            status.update(dart_status)

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
        status.setdefault("Loss/DART", dart_loss.item())
        status.setdefault("Loss/dart_decoder", 0.0)
        status.setdefault("DART/decoder_acc", 0.0)
        status["Expert/phase_id"] = {
            "specialize": 0, "refine": 1, "recovery": 2, "pursuit": 3,
            "dispatch": 4,
        }[self.expert_phase]
        status.setdefault("Redetect/count", 0)
        status["Loss/total"] = loss.item()
        return loss, status

    def _compute_reliability_intervention_loss(self, predictions, present):
        required = {
            "clean_observability_logits",
            "coverage_observability_logits",
            "positive_localization_logits",
            "negative_localization_logits",
            "duration_state_logits",
            "duration_state_targets",
        }
        missing = sorted(required - set(predictions))
        if missing:
            raise RuntimeError(
                "paired reliability interventions are missing: "
                + ", ".join(missing))
        clean_logits = predictions["clean_observability_logits"].float()
        covered_logits = predictions["coverage_observability_logits"].float()
        positive_logits = predictions["positive_localization_logits"].float()
        negative_logits = predictions["negative_localization_logits"].float()
        duration_logits = predictions["duration_state_logits"].float()
        duration_targets = predictions["duration_state_targets"].long()
        tensors = (clean_logits, covered_logits, positive_logits, negative_logits)
        if any(tensor.ndim != 2 or tensor.shape[1] != 2 for tensor in tensors):
            raise ValueError("reliability intervention logits must have shape (B,2)")
        if any(tensor.shape != clean_logits.shape for tensor in tensors[1:]):
            raise ValueError("reliability intervention logits must share shape")
        expected_duration_shape = (
            len(Action), clean_logits.shape[0], len(Action))
        if duration_logits.shape != expected_duration_shape:
            raise ValueError(
                "duration_state_logits must have shape (states,B,states)")
        if duration_targets.shape != expected_duration_shape[:2]:
            raise ValueError(
                "duration_state_targets must have shape (states,B)")
        present = torch.as_tensor(
            present, device=clean_logits.device, dtype=torch.bool).reshape(-1)
        if present.numel() != clean_logits.shape[0]:
            raise ValueError("reliability intervention mask must match batch size")
        if not bool(present.any()):
            zero = (
                sum(tensor.sum() for tensor in tensors)
                + duration_logits.sum()) * 0.0
            return zero, {
                "Loss/DART": 0.0,
                "Loss/dart_coverage": 0.0,
                "Loss/dart_geometry": 0.0,
                "DART/observability_gap": 0.0,
                "DART/localization_gap": 0.0,
                "Loss/dart_decoder": 0.0,
                "DART/decoder_acc": 0.0,
            }

        clean = clean_logits[present]
        covered = covered_logits[present]
        positive = positive_logits[present]
        negative = negative_logits[present]
        duration = duration_logits[:, present].reshape(-1, len(Action))
        duration_target = duration_targets[:, present].reshape(-1)
        ones = torch.ones(clean.shape[0], device=clean.device, dtype=torch.long)
        zeros = torch.zeros_like(ones)
        clean_probability = clean.softmax(dim=-1)[:, 1]
        covered_probability = covered.softmax(dim=-1)[:, 1]
        positive_probability = positive.softmax(dim=-1)[:, 1]
        negative_probability = negative.softmax(dim=-1)[:, 1]

        coverage_classification = 0.5 * (
            F.cross_entropy(clean, ones)
            + F.cross_entropy(covered, zeros))
        coverage_ranking = F.relu(
            self.dart_ranking_margin
            - clean_probability + covered_probability).mean()
        coverage_loss = (
            coverage_classification
            + self.dart_ranking_weight * coverage_ranking)

        geometry_classification = 0.5 * (
            F.cross_entropy(positive, ones)
            + F.cross_entropy(negative, zeros))
        geometry_ranking = F.relu(
            self.dart_ranking_margin
            - positive_probability + negative_probability).mean()
        geometry_loss = (
            geometry_classification
            + self.dart_ranking_weight * geometry_ranking)
        decoder_loss = F.cross_entropy(duration, duration_target)
        decoder_accuracy = (
            duration.argmax(dim=-1) == duration_target).float().mean()
        total = (
            self.dart_coverage_weight * coverage_loss
            + self.dart_geometry_weight * geometry_loss
            + self.dart_decoder_weight * decoder_loss)
        return total, {
            "Loss/DART": total.item(),
            "Loss/dart_coverage": coverage_loss.item(),
            "Loss/dart_geometry": geometry_loss.item(),
            "Loss/dart_decoder": decoder_loss.item(),
            "DART/decoder_acc": decoder_accuracy.item(),
            "DART/observability_gap": (
                clean_probability - covered_probability).mean().item(),
            "DART/localization_gap": (
                positive_probability - negative_probability).mean().item(),
        }

    def _dispatch_targets(self, pred_dict, gt_dict):
        logits = pred_dict.get("expert_activation_logits")
        if logits is None:
            raise RuntimeError("dispatch training requires activation logits")
        model = self.net.module if hasattr(self.net, "module") else self.net
        expert_names = tuple(model.expert_names)
        if logits.ndim != 2 or logits.shape[1] != len(expert_names) - 1:
            raise ValueError(
                "expert activation logits must provide one value per specialist")

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

        targets = eligible[:, 1:].clone()
        for expert_id in range(1, len(expert_names)):
            if expert_id != VISIBILITY_EXPERT_ID:
                targets[:, expert_id - 1] &= present
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
        controller_loss = torch.stack(losses).mean()
        specialist_outputs = predictions.get(
            "pursuit_specialist_outputs", ())
        specialist_id = predictions.get("pursuit_specialist_id")
        specialist_statuses = []
        specialist_losses = []
        specialist_weights = []
        if specialist_outputs:
            for index, output in enumerate(specialist_outputs):
                step_present = predictions[
                    "pursuit_specialist_present"][index]
                step_weight = step_present.float().sum()
                if not bool(step_present.any()):
                    continue
                local_gt = {
                    "search_anno": predictions[
                        "pursuit_specialist_targets"][index].unsqueeze(0),
                    "search_absent": step_present.unsqueeze(0),
                    "training_expert_id": torch.full(
                        (step_present.numel(),), specialist_id,
                        device=step_present.device, dtype=torch.long),
                }
                step_loss, step_status = super().compute_losses(
                    output, local_gt, return_status=True)
                specialist_losses.append(step_loss * step_weight)
                specialist_weights.append(step_weight)
                specialist_statuses.append((step_status, step_weight))
            if not specialist_losses:
                raise RuntimeError(
                    "causal specialist pursuit produced no eligible loss")
            specialist_weight = torch.stack(specialist_weights).sum()
            loss = torch.stack(specialist_losses).sum() / specialist_weight
        else:
            loss = controller_loss
        discrimination_loss = loss * 0.0
        discrimination_positive = loss.detach() * 0.0
        discrimination_negative = loss.detach() * 0.0
        discrimination_violation_rate = loss.detach() * 0.0
        discrimination_frame_count = 0
        if specialist_id == DISCRIMINATION_EXPERT_ID:
            discrimination_present = predictions.get(
                "pursuit_discrimination_present")
            if (discrimination_present is None
                    or len(discrimination_present) != len(specialist_outputs)):
                raise RuntimeError(
                    "discrimination pursuit requires ambiguity frame masks")
            ranking_losses = []
            ranking_weights = []
            positive_sums = []
            negative_sums = []
            violation_sums = []
            margin = float(getattr(
                train_cfg, "DISCRIMINATION_RANKING_MARGIN", 0.2))
            for index, output in enumerate(specialist_outputs):
                if "score_map" not in output:
                    raise RuntimeError(
                        "discrimination pursuit requires specialist score maps")
                step_loss, positive, negative, count, violation_rate = (
                    self._discrimination_ranking_loss(
                        output["score_map"],
                        predictions["pursuit_specialist_targets"][index],
                        discrimination_present[index],
                        margin,
                    )
                )
                if count == 0:
                    continue
                weight = output["score_map"].new_tensor(float(count))
                ranking_losses.append(step_loss * weight)
                ranking_weights.append(weight)
                positive_sums.append(positive * weight)
                negative_sums.append(negative * weight)
                violation_sums.append(violation_rate * weight)
                discrimination_frame_count += count
            if ranking_losses:
                ranking_weight = torch.stack(ranking_weights).sum()
                discrimination_loss = (
                    torch.stack(ranking_losses).sum() / ranking_weight)
                discrimination_positive = (
                    torch.stack(positive_sums).sum() / ranking_weight)
                discrimination_negative = (
                    torch.stack(negative_sums).sum() / ranking_weight)
                discrimination_violation_rate = (
                    torch.stack(violation_sums).sum() / ranking_weight)
                weight = float(getattr(
                    train_cfg, "DISCRIMINATION_RANKING_WEIGHT", 1.0))
                loss = loss + weight * discrimination_loss

        motion_loss = loss * 0.0
        motion_pair_count = 0
        if specialist_id == MOTION_EXPERT_ID:
            image_boxes = predictions.get(
                "pursuit_specialist_image_boxes", ())
            image_targets = predictions.get(
                "pursuit_specialist_image_targets", ())
            image_present = predictions.get(
                "pursuit_specialist_present", ())
            if (image_boxes or image_targets) and not (
                    len(image_boxes) == len(image_targets)
                    == len(image_present)):
                raise RuntimeError(
                    "motion pursuit fields must describe the same causal steps")
            pair_losses = []
            pair_weights = []
            for index in range(1, len(image_boxes)):
                pair_mask = image_present[index - 1] & image_present[index]
                if not bool(pair_mask.any()):
                    continue
                predicted_motion = relative_box_motion(
                    image_boxes[index], image_boxes[index - 1])
                target_motion = relative_box_motion(
                    image_targets[index], image_targets[index - 1])
                per_row = F.smooth_l1_loss(
                    predicted_motion, target_motion, reduction="none"
                ).mean(dim=-1)
                pair_losses.append(
                    (per_row * pair_mask.to(per_row.dtype)).sum())
                pair_weights.append(pair_mask.float().sum())
                motion_pair_count += int(pair_mask.sum())
            if pair_losses:
                motion_loss = (
                    torch.stack(pair_losses).sum()
                    / torch.stack(pair_weights).sum().clamp_min(1.0)
                )
                motion_weight = float(getattr(
                    train_cfg, "MOTION_DISPLACEMENT_WEIGHT", 1.0))
                loss = loss + motion_weight * motion_loss
        if not return_status:
            return loss
        status = {
            key: sum(item[key] for item in statuses) / len(statuses)
            for key in statuses[0]
            if key.startswith("Loss/")
        }
        status["Loss/total"] = float(loss.detach())
        status["Loss/pursuit_controller_diagnostic"] = float(
            controller_loss.detach())
        motion_weight = float(getattr(
            train_cfg, "MOTION_DISPLACEMENT_WEIGHT", 1.0))
        status["Loss/motion_displacement"] = float(motion_loss.detach())
        status["Loss/motion_displacement_weighted"] = float(
            (motion_weight * motion_loss).detach())
        status["MotionTrain/pair_count"] = motion_pair_count
        discrimination_weight = float(getattr(
            train_cfg, "DISCRIMINATION_RANKING_WEIGHT", 1.0))
        status["Loss/discrimination_ranking"] = float(
            discrimination_loss.detach())
        status["Loss/discrimination_ranking_weighted"] = float(
            (discrimination_weight * discrimination_loss).detach())
        status["DiscriminationTrain/frame_count"] = (
            discrimination_frame_count)
        status["DiscriminationTrain/ambiguity_frame_count"] = (
            discrimination_frame_count)
        status["DiscriminationTrain/gradient_frame_count"] = (
            discrimination_frame_count)
        status["DiscriminationTrain/target_peak"] = float(
            discrimination_positive)
        status["DiscriminationTrain/hardest_distractor_peak"] = float(
            discrimination_negative)
        status["DiscriminationTrain/violation_rate"] = float(
            discrimination_violation_rate)
        if specialist_statuses:
            total_weight = sum(
                float(weight.detach()) for _, weight in specialist_statuses)
            status[f"Expert/train_count_{specialist_id}"] = int(total_weight)
            status[f"Expert/train_iou_{specialist_id}"] = sum(
                item["IoU"] * float(weight.detach())
                for item, weight in specialist_statuses
            ) / total_weight
            status["IoU"] = status[f"Expert/train_iou_{specialist_id}"]
            status["Loss/causal_specialist"] = float(loss.detach())
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

    def _compute_proposal_identity_loss(self, predictions):
        identity_scores = predictions.get("identity_scores")
        identity_targets = predictions.get("identity_targets")
        identity_valid = predictions.get("identity_valid")
        if any(value is None for value in (
                identity_scores, identity_targets, identity_valid)):
            raise RuntimeError(
                "recovery training requires proposal identity scores, targets, and validity")
        if (identity_scores.ndim != 2
                or identity_targets.shape != identity_scores.shape
                or identity_valid.shape != identity_scores.shape):
            raise ValueError(
                "proposal identity tensors must share shape (B,K)")
        targets = identity_targets.to(
            device=identity_scores.device, dtype=torch.bool)
        valid = identity_valid.to(
            device=identity_scores.device, dtype=torch.bool)
        targets = targets & valid
        scores = identity_scores.float().clamp(1e-6, 1.0 - 1e-6)
        zero = scores.sum() * 0.0
        if valid.any():
            identity_loss = F.binary_cross_entropy_with_logits(
                torch.logit(scores[valid]), targets[valid].to(scores.dtype))
        else:
            identity_loss = zero
        ranking_terms = []
        score_gaps = []
        for row in range(scores.shape[0]):
            positives = scores[row][targets[row]]
            negatives = scores[row][valid[row] & ~targets[row]]
            if positives.numel() and negatives.numel():
                score_gap = positives.max() - negatives.max()
                score_gaps.append(score_gap)
                ranking_terms.append(F.relu(
                    self.identity_ranking_margin - score_gap))
        ranking_loss = (
            torch.stack(ranking_terms).mean() if ranking_terms else zero)
        positive_scores = scores[targets]
        negative_scores = scores[valid & ~targets]
        positive_mean = (
            positive_scores.mean() if positive_scores.numel() else zero)
        negative_mean = (
            negative_scores.mean() if negative_scores.numel() else zero)
        hardest_gap_mean = (
            torch.stack(score_gaps).mean() if score_gaps else zero)
        weighted_loss = (
            self.identity_loss_weight * identity_loss
            + self.identity_ranking_weight * ranking_loss)
        return weighted_loss, {
            "Loss/recovery_identity": identity_loss.item(),
            "Loss/recovery_ranking": ranking_loss.item(),
            "Redetect/identity_positive_count": int(targets.sum().item()),
            "Redetect/identity_negative_count": int(
                (valid & ~targets).sum().item()),
            "Redetect/identity_valid_count": int(valid.sum().item()),
            "Redetect/identity_positive_mean": positive_mean.item(),
            "Redetect/identity_negative_mean": negative_mean.item(),
            "Redetect/identity_hardest_gap_mean": hardest_gap_mean.item(),
        }

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

        identity_loss, identity_status = (
            self._compute_proposal_identity_loss(predictions))
        loss = (
            self.loss_weight["focal"] * focal_loss
            + self.loss_weight["l1"] * box_loss
            + self.loss_weight["giou"] * giou_loss
            + identity_loss
        )
        status = {
            "Loss/redetect": loss.item(),
            "Loss/redetect_focal": focal_loss.item(),
            "Loss/redetect_l1": box_loss.item(),
            "Loss/redetect_giou": giou_loss.item(),
            "Redetect/count": int(indices.numel()),
        }
        status.update(identity_status)
        return loss, status
