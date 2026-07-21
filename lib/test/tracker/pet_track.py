"""PET-Track inference with event-guided RGB recovery and protected memory."""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from lib.models.pet_track import build_pet_track
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond, generate_mask_z
from lib.models.layers.thor import THOR_Wrapper
from lib.models.layers.srbt_hypotheses import (
    build_hypothesis_tracker,
    crop_cxcywh_to_image_xywh,
)
from lib.models.layers.srbt_controller import (
    Action as BeliefAction,
    ControllerAction,
    build_duration_decoder,
)
from lib.models.layers.event_recovery import EventProposalExtractor
from lib.models.layers.expert_ensemble import (
    fuse_expert_predictions,
    normalized_response_psr,
)
from lib.models.layers.search_window_controller import (
    event_motion_centroid,
    relative_box_motion,
)
from lib.train.trainers.base_trainer import load_srbt_checkpoint_file


DEFAULT_EXPERT_NAMES = (
    "generalist", "motion_fm", "precision_refiner",
    "visibility_foc_ov", "discrimination_bi",
)


def _load_srbt_eval_checkpoint(network, checkpoint_path):
    checkpoint = load_srbt_checkpoint_file(checkpoint_path)
    network.load_state_dict(checkpoint["net"], strict=True)
    return checkpoint


class PETTrack(BaseTracker):
    supports_absent_output = True

    def __init__(self, params, dataset_name, network=None):
        super().__init__(params)
        if network is None:
            network = build_pet_track(params.cfg, training=False)
            _load_srbt_eval_checkpoint(network, self.params.checkpoint)
            network = network.cuda()
        self.cfg = params.cfg
        policy_mode = getattr(params, "policy_mode", None)
        policy_mode = str(
            policy_mode or getattr(
                self.cfg.TEST, "POLICY_MODE", "stateful")).lower()
        self.srbt_controller_enabled = policy_mode == "stateful"
        expert_names = tuple(getattr(
            network, "expert_names", DEFAULT_EXPERT_NAMES))
        self.forced_expert_id = getattr(params, "forced_expert_id", None)
        if (self.forced_expert_id is not None
                and int(self.forced_expert_id) not in range(len(expert_names))):
            raise ValueError(
                "forced_expert_id must address a configured expert")
        if self.forced_expert_id is not None:
            self.forced_expert_id = int(self.forced_expert_id)
        self.hypothesis_tracker = build_hypothesis_tracker(self.cfg)
        self.duration_decoder = build_duration_decoder(
            self.cfg,
            predictor=getattr(network, "duration_evidence_decoder", None),
        )
        self._srbt_last_action = BeliefAction.TRACK
        self._redetect_hypotheses = None
        self.network = network
        self.expert_names = expert_names
        self.device = next(self.network.parameters()).device
        self.network.eval()
        expert_cfg = getattr(self.cfg.MODEL, "EXPERT", None)
        use_activation = bool(getattr(
            expert_cfg, "USE_ACTIVATION_INFERENCE", False))
        if use_activation and not bool(getattr(
                expert_cfg, "ACTIVATOR_TRAINED", False)):
            raise RuntimeError(
                "sparse expert inference requires a trained activator")
        if use_activation and getattr(
                self.network, "expert_activator", None) is None:
            raise RuntimeError(
                "sparse expert inference is enabled but the model has no activator")
        self.auto_expert_activation = use_activation
        self.preprocessor = Preprocessor(device=self.device)
        self.state = None
        search_controller_cfg = getattr(
            self.cfg.MODEL, "SEARCH_CONTROLLER", None)
        self.search_controller_enabled = bool(getattr(
            search_controller_cfg, "ENABLE", False)) and bool(getattr(
                search_controller_cfg, "USE_INFERENCE", False))
        if self.search_controller_enabled and not bool(getattr(
                search_controller_cfg, "TRAINED", False)):
            raise RuntimeError(
                "search-window inference requires a trained controller")
        if self.search_controller_enabled and getattr(
                self.network, "search_window_controller", None) is None:
            raise RuntimeError(
                "search-window inference is enabled but the model has no controller")
        self._planned_search_state = None
        self._search_controller_previous_box = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz]).long(),
            centered=True,
        ).to(self.device)

        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug and not self.use_visdom:
            self.save_dir = "debug"
            os.makedirs(self.save_dir, exist_ok=True)

        self.save_all_boxes = params.save_all_boxes
        self.thor_wrapper = THOR_Wrapper(
            net=self.network,
            st_capacity=params.cfg.TEST.SHORTTERM_LIBRARY_NUMS,
            lt_capacity=params.cfg.TEST.LONGTERM_LIBRARY_NUMS,
            sample_interval=params.cfg.TEST.SAMPLE_INTERVAL,
            update_interval=params.cfg.TEST.UPDATE_INTERVAL,
            lower_bound=params.cfg.TEST.LOWER_BOUND,
            score_threshold=params.cfg.TEST.SCORE_THRESHOLD)

        self._last_score_peak = 0.0
        self._last_redetect_conf = 0.0
        self._last_redetect_error = ""
        self._pending_redetect_box = None
        # Redetect full-image crop config.
        redetect_cfg = getattr(self.cfg.MODEL, "REDETECT", None)
        self.redetect_factor = float(getattr(
            redetect_cfg, "TRAIN_SEARCH_FACTOR", 8.0
        )) if redetect_cfg is not None else 8.0
        self.recovery_search_factor = float(getattr(
            redetect_cfg, "RECOVERY_SEARCH_FACTOR", 5.0
        )) if redetect_cfg is not None else 5.0
        self.full_rgb_fallback_interval = int(getattr(
            redetect_cfg, "FULL_RGB_FALLBACK_INTERVAL", 10
        )) if redetect_cfg is not None else 10
        if self.full_rgb_fallback_interval < 1:
            raise ValueError("FULL_RGB_FALLBACK_INTERVAL must be positive")
        self.event_proposal_extractor = EventProposalExtractor(
            top_k=int(getattr(redetect_cfg, "EVENT_TOP_K", 5)),
            nms_radius=int(getattr(redetect_cfg, "EVENT_NMS_RADIUS", 12)),
            min_robust_score=float(getattr(
                redetect_cfg, "EVENT_MIN_ROBUST_SCORE", 3.0)),
        ).to(self.device)

    def _reset_srbt_sequence_state(self):
        self._srbt_last_action = BeliefAction.TRACK
        self._redetect_hypotheses = None
        self._pending_redetect_box = None
        self._recovery_diagnostics = [{
            "action": "track",
            "event_centers": [],
            "identity_scores": [],
        }]
        self._expert_diagnostics = []
        self._search_diagnostics = []
        self._planned_search_state = None
        self._search_controller_previous_box = None
        self._motion_previous_event_search = None

    def _build_motion_context(self, event_search, reference_state=None):
        previous_event = getattr(
            self, "_motion_previous_event_search", None)
        history_valid = previous_event is not None
        current_state = (
            self.state if reference_state is None else reference_state)
        previous_state = getattr(self, "state", None)
        if previous_state is None:
            previous_state = current_state
        current_box = torch.as_tensor(
            current_state, device=event_search.device,
            dtype=event_search.dtype).reshape(1, 4)
        previous_box = torch.as_tensor(
            previous_state, device=event_search.device,
            dtype=event_search.dtype).reshape(1, 4)
        return {
            "current_event": event_search,
            "previous_event": (
                event_search if previous_event is None else previous_event),
            "history_valid": torch.full(
                (event_search.shape[0],), history_valid,
                device=event_search.device, dtype=torch.bool),
            "box_delta": relative_box_motion(current_box, previous_box),
        }

    def _commit_motion_event_search(self, event_search):
        self._motion_previous_event_search = event_search.detach().clone()

    def _search_state_for_frame(self):
        if (self._srbt_last_action in (
                BeliefAction.GLOBAL_UNRESOLVED, BeliefAction.VERIFY)
                and self._pending_redetect_box is not None):
            return self._pending_redetect_box
        if (getattr(self, "search_controller_enabled", False)
                and getattr(self, "_planned_search_state", None) is not None
                and self._srbt_last_action in (
                    BeliefAction.TRACK, BeliefAction.LOCAL_UNRESOLVED)):
            return self._planned_search_state
        return self.state

    @staticmethod
    def _normalize_image_boxes(boxes, height, width, device):
        boxes = torch.as_tensor(
            boxes, device=device, dtype=torch.float32)
        scale = boxes.new_tensor((width, height, width, height))
        return boxes / scale

    def _plan_next_search_state(
            self, event_image, height, width, local_candidate, action):
        if not getattr(self, "search_controller_enabled", False) or action not in (
                "track", "local_unresolved") or local_candidate is None:
            self._planned_search_state = None
            return None
        expert_states = local_candidate.get("expert_states")
        peaks = local_candidate.get("expert_peaks")
        psr = local_candidate.get("expert_psr")
        if not expert_states or len(expert_states) != len(self.network.expert_names):
            raise RuntimeError(
                "search controller requires every expert state")
        if len(peaks) != len(expert_states) or len(psr) != len(expert_states):
            raise RuntimeError(
                "search controller requires every expert response statistic")
        current = self._normalize_image_boxes(
            [self.state], height, width, self.device)
        previous = (
            current if self._search_controller_previous_box is None
            else self._search_controller_previous_box)
        experts = self._normalize_image_boxes(
            [expert_states], height, width, self.device)
        event_tensor = torch.as_tensor(
            event_image, device=self.device, dtype=torch.float32)
        if event_tensor.ndim != 3:
            raise ValueError("event image must have shape [height, width, channels]")
        event_center, event_confidence = event_motion_centroid(
            event_tensor.permute(2, 0, 1).unsqueeze(0))
        presence = local_candidate.get("presence_score")
        if presence is None:
            presence = current.new_ones(1)
        with torch.no_grad():
            output = self.network.search_window_controller(
                current_box=current,
                previous_box=previous,
                expert_boxes=experts,
                response_peaks=current.new_tensor([peaks]),
                response_psr=current.new_tensor([psr]),
                presence=torch.as_tensor(
                    presence, device=self.device, dtype=current.dtype),
                event_center=event_center,
                event_confidence=event_confidence,
            )
        scale = output.next_box.new_tensor((width, height, width, height))
        planned = output.next_box[0] * scale
        self._planned_search_state = clip_box(
            planned.detach().cpu().tolist(), height, width, margin=1)
        self._search_controller_previous_box = current.detach()
        local_candidate["planned_search_state"] = list(
            self._planned_search_state)
        local_candidate["search_inside_probability"] = float(
            output.inside_logit.sigmoid()[0].item())
        local_candidate["search_quality_probability"] = float(
            output.quality_logit.sigmoid()[0].item())
        return self._planned_search_state

    def get_recovery_diagnostics(self):
        return [dict(item) for item in self._recovery_diagnostics]

    def get_expert_diagnostics(self):
        return [[list(box) for box in frame]
                for frame in self._expert_diagnostics]

    def get_search_diagnostics(self):
        return [{
            key: list(value) if isinstance(value, list) else value
            for key, value in item.items()
        } for item in self._search_diagnostics]

    def _record_search_diagnostic(
            self, search_state, resize_factor, action, local_candidate,
            *, previous_action="", controller_output_score=None,
            recovery_attempted=False, recovery_confirmed=False,
            recovery_max_identity=None, recovery_max_localization=None,
            recovery_accepted_count=0, is_initial=False, frame_id=None):
        state = [float(value) for value in search_state]
        crop_size = math.ceil(
            math.sqrt(state[2] * state[3]) * self.params.search_factor)
        x1 = round(state[0] + 0.5 * state[2] - 0.5 * crop_size)
        y1 = round(state[1] + 0.5 * state[3] - 0.5 * crop_size)
        candidate = local_candidate or {}
        weights = candidate.get("ensemble_weights", [])
        if torch.is_tensor(weights):
            weights = weights.detach().cpu().tolist()
        presence_score = self._diagnostic_scalar(
            candidate.get("presence_score"))
        observability_score = self._diagnostic_scalar(
            candidate.get("observability_score"))
        localization_validity_score = self._diagnostic_scalar(
            candidate.get("localization_validity_score"))
        acceptance_score = self._diagnostic_scalar(
            candidate.get("acceptance_score"))
        controller = getattr(self, "duration_decoder", None)
        self._search_diagnostics.append({
            "frame_id": int(self.frame_id if frame_id is None else frame_id),
            "is_initial": bool(is_initial),
            "search_state": state,
            "crop_bounds_xyxy": [
                float(x1), float(y1),
                float(x1 + crop_size), float(y1 + crop_size),
            ],
            "resize_factor": float(resize_factor),
            "previous_action": str(previous_action),
            "action": str(action),
            "presence_score": presence_score,
            "observability_score": observability_score,
            "localization_validity_score": localization_validity_score,
            "acceptance_score": acceptance_score,
            "controller_output_score": self._diagnostic_scalar(
                controller_output_score),
            "theta_observable": self._diagnostic_scalar(
                getattr(controller, "theta_observable", None)),
            "theta_localized": self._diagnostic_scalar(
                getattr(controller, "theta_localized", None)),
            "theta_recover": self._diagnostic_scalar(
                getattr(controller, "theta_recover", None)),
            "decoder_state_duration": int(getattr(
                controller, "state_duration", 0)),
            "decoder_unresolved_duration": int(getattr(
                controller, "_unresolved_duration", 0)),
            "decoder_stable_visible": int(getattr(
                controller, "_stable_visible", 0)),
            "recovery_attempted": bool(recovery_attempted),
            "recovery_confirmed": bool(recovery_confirmed),
            "recovery_max_identity": self._diagnostic_scalar(
                recovery_max_identity),
            "recovery_max_localization": self._diagnostic_scalar(
                recovery_max_localization),
            "recovery_accepted_count": int(recovery_accepted_count),
            "redetect_confidence": self._diagnostic_scalar(
                getattr(self, "_last_redetect_conf", None)),
            "ensemble_score": float(candidate.get("score_peak", 0.0)),
            "expert_peaks": [
                float(value) for value in candidate.get("expert_peaks", [])],
            "expert_psr": [
                float(value) for value in candidate.get("expert_psr", [])],
            "retained_expert_ids": [
                int(value) for value in candidate.get(
                    "retained_expert_ids", [])],
            "ensemble_weights": [float(value) for value in weights],
            "planned_search_state": [
                float(value) for value in candidate.get(
                    "planned_search_state", [])],
            "search_inside_probability": self._diagnostic_scalar(
                candidate.get("search_inside_probability")),
            "search_quality_probability": self._diagnostic_scalar(
                candidate.get("search_quality_probability")),
        })

    @staticmethod
    def _diagnostic_scalar(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.detach().item()
        value = float(value)
        return value if math.isfinite(value) else None

    @staticmethod
    def _recovery_diagnostic_value(recovery, key, *, count=False):
        if recovery is None or key not in recovery:
            return 0 if count else None
        value = recovery[key]
        if torch.is_tensor(value):
            if value.numel() == 0:
                return 0 if count else None
            return int(value.detach().sum().item()) if count else float(
                value.detach().max().item())
        return None

    def initialize(self, image, event_image, info: dict, idx=0):
        z_patch_arr, event_z_patch_arr, resize_factor, z_amask_arr = sample_target(
            im=image, eim=event_image, target_bb=info['init_bbox'],
            search_area_factor=self.params.template_factor, output_sz=self.params.template_size)

        self.z_patch_arr = z_patch_arr
        self.event_z_patch_arr = event_z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors
        event_template = self.preprocessor.process(event_z_patch_arr, z_amask_arr).tensors

        template_bbox = self.transform_bbox_to_crop(
            info['init_bbox'], resize_factor, template.device).squeeze(1)
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)
        self.mask_z = generate_mask_z(cfg=self.cfg, bs=1, device=template.device, gt_bbox=template_bbox)

        with torch.no_grad():
            self.dynamic_zi = None
            self.dynamic_ze = None
            self.static_zi = template
            self.static_ze = event_template
            small_target_expert = getattr(
                self.network, "small_target_expert", None)
            self.small_template_features = (
                small_target_expert.encode_template(template, event_template)
                if small_target_expert is not None else None
            )
            self._last_trusted_zi = template.detach().clone()
            self._last_trusted_ze = event_template.detach().clone()
            self.thor_wrapper.setup(self.static_zi, self.static_ze)
            self.duration_decoder.reset()
            self._reset_srbt_sequence_state()
            self._last_score_peak = 0.0
            self._last_redetect_conf = 0.0
            self._last_redetect_error = ""

        self.state = info['init_bbox']
        expert_names = tuple(getattr(self, "expert_names", ()))
        if not expert_names:
            expert_names = tuple(getattr(
                self.network, "expert_names", DEFAULT_EXPERT_NAMES))
            self.expert_names = expert_names
        self._expert_diagnostics = [[
            list(info['init_bbox']) for _ in expert_names
        ]]
        self.frame_id = idx
        self._record_search_diagnostic(
            info['init_bbox'],
            self.params.search_size / math.ceil(
                math.sqrt(info['init_bbox'][2] * info['init_bbox'][3])
                * self.params.search_factor),
            "initialize",
            None,
            is_initial=True,
            frame_id=idx,
        )

    def _run_local_candidate(self, search, event_search,
                             resize_factor, height, width,
                             reference_state=None,
                             dynamic_templates=None):
        """Run the local path without committing its box."""
        if dynamic_templates is None:
            dynamic_zi, dynamic_ze = self.thor_wrapper.begin_frame()
        else:
            dynamic_zi, dynamic_ze = dynamic_templates
        self.dynamic_zi, self.dynamic_ze = dynamic_zi, dynamic_ze

        inference_kwargs = {
            "static_zi": self.static_zi,
            "static_ze": self.static_ze,
            "dynamic_zi": dynamic_zi,
            "dynamic_ze": dynamic_ze,
            "xi": search,
            "xe": event_search,
            "small_template_features": getattr(
                self, "small_template_features", None),
            "motion_context": self._build_motion_context(
                event_search, reference_state),
        }
        expert_names = tuple(getattr(self.network, "expert_names", ()))
        forced_expert_id = getattr(self, "forced_expert_id", None)
        if forced_expert_id is not None and expert_names:
            forced_name = expert_names[forced_expert_id]
            inference_kwargs["active_expert_names"] = (
                () if forced_expert_id == 0 else (forced_name,))
        elif getattr(self, "auto_expert_activation", False):
            inference_kwargs["auto_activate"] = True
        out_dict = self.network.inference(
            **inference_kwargs)
        expert_outputs = out_dict.get("expert_outputs")
        if expert_outputs:
            if not expert_names:
                expert_names = tuple(expert_outputs)
            active_names = tuple(
                name for name in expert_names if name in expert_outputs)
            if not active_names or active_names[0] != expert_names[0]:
                raise RuntimeError(
                    "expert inference must include the generalist output")
            active_expert_ids = tuple(
                expert_names.index(name) for name in active_names)
            mapped_boxes = []
            response_maps = []
            response_peaks = []
            response_psr = []
            localization_validity = []
            acceptance_scores = []
            observability_scores = []
            for name in active_names:
                expert_output = expert_outputs[name]
                score_map = expert_output["score_map"]
                window = self.output_window
                if window.shape[-2:] != score_map.shape[-2:]:
                    window = F.interpolate(
                        window, size=score_map.shape[-2:],
                        mode="bilinear", align_corners=False)
                native_response = window * score_map
                response_peaks.append(native_response.flatten(1).max(dim=1).values[0])
                response_psr.append(normalized_response_psr(native_response)[0])
                if native_response.shape[-2:] != self.output_window.shape[-2:]:
                    native_response = F.interpolate(
                        native_response, size=self.output_window.shape[-2:],
                        mode="bilinear", align_corners=False)
                response_maps.append(native_response)
                pred_box = (
                    expert_output["pred_boxes"][0, 0]
                    * self.params.search_size / resize_factor
                ).tolist()
                mapped_boxes.append(clip_box(
                    self.map_box_back(
                        pred_box, resize_factor, reference_state),
                    height, width, margin=10))
                reliability = expert_output.get("reliability_predictions")
                if reliability is None and name == expert_names[0]:
                    reliability = out_dict.get("reliability_predictions")
                if reliability is None:
                    localization_validity.append(None)
                    acceptance_scores.append(None)
                    observability_scores.append(None)
                else:
                    localization_validity.append(
                        reliability["localization_validity_score"][0])
                    acceptance_scores.append(
                        reliability["acceptance_score"][0])
                    observability_scores.append(
                        reliability["observability_score"][0])
            response_stack = torch.cat(response_maps, dim=0)
            boxes = torch.as_tensor(
                mapped_boxes, device=response_stack.device,
                dtype=response_stack.dtype)
            if forced_expert_id is None:
                candidate_validity = (
                    torch.stack(localization_validity)
                    if all(value is not None for value in localization_validity)
                    else None
                )
                ensemble = fuse_expert_predictions(
                    boxes=boxes,
                    response_peaks=torch.stack(response_peaks),
                    response_psr=torch.stack(response_psr),
                    localization_validity=candidate_validity,
                    last_box=(self.state if reference_state is None
                              else reference_state),
                )
                selected_local_id = ensemble.retained_ids[0]
                candidate_state = clip_box(
                    ensemble.box.detach().cpu().tolist(),
                    height, width, margin=10)
                response = (
                    ensemble.weights[:, None, None, None] * response_stack
                ).sum(dim=0, keepdim=True)
                retained_expert_ids = tuple(
                    active_expert_ids[local_id]
                    for local_id in ensemble.retained_ids)
                local_weights = ensemble.weights.detach()
            else:
                if forced_expert_id not in active_expert_ids:
                    raise RuntimeError(
                        "forced expert output is missing from inference")
                local_id = active_expert_ids.index(forced_expert_id)
                candidate_state = list(mapped_boxes[local_id])
                response = response_stack[
                    local_id:local_id + 1]
                local_weights = torch.zeros(
                    len(mapped_boxes),
                    device=response_stack.device,
                    dtype=response_stack.dtype,
                )
                local_weights[local_id] = 1.0
                selected_local_id = local_id
                retained_expert_ids = (forced_expert_id,)
            generalist_state = list(mapped_boxes[0])
            expert_states = [
                list(generalist_state) for _ in expert_names]
            expert_peaks = [0.0 for _ in expert_names]
            expert_psr_values = [0.0 for _ in expert_names]
            ensemble_weights = torch.zeros(
                len(expert_names), device=response_stack.device,
                dtype=response_stack.dtype)
            for local_id, expert_id in enumerate(active_expert_ids):
                expert_states[expert_id] = list(mapped_boxes[local_id])
                expert_peaks[expert_id] = float(
                    response_peaks[local_id].detach().item())
                expert_psr_values[expert_id] = float(
                    response_psr[local_id].detach().item())
                ensemble_weights[expert_id] = local_weights[local_id]
            selected_observability = observability_scores[selected_local_id]
            selected_localization = localization_validity[selected_local_id]
            selected_acceptance = acceptance_scores[selected_local_id]
        else:
            response = self.output_window * out_dict['score_map']
            pred_box = (
                out_dict["target_bbox"][0]
                * self.params.search_size / resize_factor
            ).tolist()
            candidate_state = clip_box(
                self.map_box_back(pred_box, resize_factor, reference_state),
                height, width, margin=10)
            retained_expert_ids = (0,)
            ensemble_weights = torch.ones(
                1, device=response.device, dtype=response.dtype)
            expert_names = tuple(getattr(
                self.network, "expert_names", ("generalist",)))
            active_expert_ids = (0,)
            expert_states = [list(candidate_state) for _ in expert_names]
            expert_peaks = []
            expert_psr_values = []
            reliability = out_dict.get("reliability_predictions")
            selected_observability = (
                None if reliability is None
                else reliability["observability_score"][0])
            selected_localization = (
                None if reliability is None
                else reliability["localization_validity_score"][0])
            selected_acceptance = (
                None if reliability is None
                else reliability["acceptance_score"][0])

        return {
            "state": candidate_state,
            "score_peak": float(response.max().item()),
            "response": response,
            "presence_score": out_dict.get("presence_score"),
            "observability_score": selected_observability,
            "localization_validity_score": selected_localization,
            "acceptance_score": selected_acceptance,
            "memory_frame_open": True,
            "retained_expert_ids": retained_expert_ids,
            "active_expert_ids": active_expert_ids,
            "ensemble_weights": ensemble_weights,
            "expert_states": expert_states,
            "expert_peaks": expert_peaks,
            "expert_psr": expert_psr_values,
            "motion_event_search": event_search,
        }

    def get_update_count(self):
        return self.thor_wrapper.get_update_count()

    def get_sample_count(self):
        return self.thor_wrapper.get_sample_count()

    def _step_srbt_controller(self, local_candidate,
                              identity_score=None,
                              observability_score=None,
                              localization_score=None,
                              acceptance_score=None):
        if local_candidate is None:
            return None
        observability_score = (
            local_candidate.get("observability_score")
            if observability_score is None else observability_score)
        localization_score = (
            local_candidate.get("localization_validity_score")
            if localization_score is None else localization_score)
        acceptance_score = (
            local_candidate.get("acceptance_score")
            if acceptance_score is None else acceptance_score)
        if observability_score is None or localization_score is None:
            return None
        if not getattr(self, "srbt_controller_enabled", True):
            score = (
                observability_score * localization_score
                if acceptance_score is None else acceptance_score)
            return ControllerAction(
                action=BeliefAction.TRACK,
                allow_recent_write=False,
                allow_long_write=False,
                output_absent=False,
                output_score=float(score),
            )
        return self.duration_decoder.step(
            observability_score,
            localization_score,
            identity_score,
        )

    def _refine_pending_recovery(self, image, event_image, height, width):
        if self._pending_redetect_box is None:
            return None
        reference_state = list(self._pending_redetect_box)
        patch, event_patch, resize_factor, mask = sample_target(
            im=image,
            eim=event_image,
            target_bb=reference_state,
            search_area_factor=self.params.search_factor,
            output_sz=self.params.search_size,
        )
        search = self.preprocessor.process(patch, mask).tensors
        event_search = self.preprocessor.process(event_patch, mask).tensors
        return self._run_local_candidate(
            search,
            event_search,
            resize_factor,
            height,
            width,
            reference_state=reference_state,
            dynamic_templates=(self.dynamic_zi, self.dynamic_ze),
        )

    def _resolve_tracking_state(self, local_state, height, width,
                                srbt_recovered):
        if not srbt_recovered or self._pending_redetect_box is None:
            return local_state
        recovered_state = clip_box(
            self._pending_redetect_box, height, width, margin=10)
        self._pending_redetect_box = None
        self._redetect_hypotheses = None
        self._last_redetect_conf = 0.0
        return recovered_state

    def _decay_recovery_hypotheses(self):
        if self._redetect_hypotheses is None:
            return
        decayed = self.hypothesis_tracker.update(
            self._redetect_hypotheses, None)
        self._redetect_hypotheses = (
            decayed if decayed["active_count"] > 0 else None)

    @staticmethod
    def _best_recovery_confirmation(recovery):
        if recovery is None:
            return None
        accepted = recovery.get("accepted")
        if accepted is None or accepted.numel() == 0 or not bool(accepted[0].any()):
            return None
        accepted_ids = accepted[0].nonzero(as_tuple=False).flatten()
        localization = recovery[
            "localization_validity_scores"][0].index_select(0, accepted_ids)
        best_id = accepted_ids[int(localization.argmax().item())]
        return {
            "identity_score": float(
                recovery["identity_scores"][0, best_id].item()),
            "observability_score": float(
                recovery["observability_scores"][0, best_id].item()),
            "localization_validity_score": float(
                recovery["localization_validity_scores"][0, best_id].item()),
            "acceptance_score": float(
                recovery["acceptance_scores"][0, best_id].item()),
        }

    def _run_recovery_cycle(self, image, event_image, H, W):
        self._pending_redetect_box = None
        self._last_redetect_conf = 0.0
        recovery = self._run_event_recovery(image, event_image, H, W)
        box, conf = (None, 0.0)
        if recovery is not None:
            box, conf = self._update_event_recovery_hypotheses(recovery, H, W)
        else:
            self._decay_recovery_hypotheses()
        if box is None and self.frame_id % self.full_rgb_fallback_interval == 0:
            self._redetect_hypotheses = None
            red_out = self._run_redetection(image, event_image, H, W)
            if red_out is not None:
                box, conf = self._update_redetect_hypotheses(red_out)
                self._redetect_hypotheses = None
        self._last_redetect_conf = conf
        if box is not None and conf > 0.0:
            self._pending_redetect_box = clip_box(
                box, H, W, margin=10)
        return recovery

    def track(self, image, event_image, info: dict = None):
        self.frame_id += 1
        self._last_redetect_error = ""
        H, W, _ = image.shape
        search_state = self._search_state_for_frame()
        x_patch_arr, event_x_patch_arr, resize_factor, x_amask_arr = sample_target(
            im=image, eim=event_image, target_bb=search_state,
            search_area_factor=self.params.search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr).tensors
        event_search = self.preprocessor.process(event_x_patch_arr, x_amask_arr).tensors

        with torch.no_grad():
            local_candidate = self._run_local_candidate(
                search, event_search, resize_factor, H, W,
                reference_state=search_state)

            previous_action = self._srbt_last_action
            recovery = None
            confirmation = None
            recovery_attempted = False
            if previous_action in (
                    BeliefAction.GLOBAL_UNRESOLVED, BeliefAction.VERIFY):
                recovery_attempted = True
                recovery = self._run_recovery_cycle(image, event_image, H, W)
                confirmation = self._best_recovery_confirmation(recovery)
                if confirmation is None:
                    srbt_control = self._step_srbt_controller(local_candidate)
                else:
                    srbt_control = self._step_srbt_controller(
                        local_candidate,
                        identity_score=confirmation["identity_score"],
                        observability_score=confirmation[
                            "observability_score"],
                        localization_score=confirmation[
                            "localization_validity_score"],
                        acceptance_score=confirmation["acceptance_score"],
                    )
            else:
                srbt_control = self._step_srbt_controller(local_candidate)
                if (srbt_control is not None
                        and srbt_control.action
                        is BeliefAction.GLOBAL_UNRESOLVED):
                    recovery_attempted = True
                    recovery = self._run_recovery_cycle(image, event_image, H, W)
                    confirmation = self._best_recovery_confirmation(recovery)
                    if confirmation is not None:
                        srbt_control = self._step_srbt_controller(
                            local_candidate,
                            identity_score=confirmation["identity_score"],
                            observability_score=confirmation[
                                "observability_score"],
                            localization_score=confirmation[
                                "localization_validity_score"],
                            acceptance_score=confirmation[
                                "acceptance_score"],
                        )

            if srbt_control is None:
                raise RuntimeError(
                    "tracking inference requires presence_score from network output")
            current_action = srbt_control.action
            decision = {
                "action": current_action.value,
                "enter_frozen": (
                    previous_action is BeliefAction.TRACK
                    and current_action is not BeliefAction.TRACK),
                "exit_to_tracking": (
                    previous_action is not BeliefAction.TRACK
                    and current_action is BeliefAction.TRACK),
            }
            if decision["exit_to_tracking"]:
                refined_candidate = self._refine_pending_recovery(
                    image, event_image, H, W)
                if refined_candidate is not None:
                    local_candidate = refined_candidate
                    self._pending_redetect_box = list(
                        refined_candidate["state"])
            self._srbt_last_action = current_action
            action = decision["action"]

            if decision["enter_frozen"]:
                self.thor_wrapper.freeze(True)
                self.thor_wrapper.snapshot_clean(
                    self._last_trusted_zi, self._last_trusted_ze)
                self._pending_redetect_box = None
                self._redetect_hypotheses = None
                self._last_redetect_conf = 0.0

            if decision["exit_to_tracking"]:
                self.thor_wrapper.resume()

            pred_score = 0.0
            is_absent = False
            response = None
            output_state = list(self.state)

            if action in ("track", "local_unresolved"):
                if local_candidate is None:
                    raise RuntimeError(
                        "tracking action requires a current local candidate")
                candidate_state = self._resolve_tracking_state(
                    local_candidate["state"], H, W,
                    srbt_recovered=(
                        srbt_control is not None
                        and decision["exit_to_tracking"]),
                )
                output_state = candidate_state
                if action == "track":
                    self.state = candidate_state
                pred_score = (
                    float(srbt_control.output_score)
                    if srbt_control is not None
                    else local_candidate["score_peak"])
                response = local_candidate["response"]
                self._last_score_peak = pred_score
            elif action == "global_unresolved":
                is_absent = True
            elif action == "verify":
                is_absent = True

            self._plan_next_search_state(
                event_image, H, W, local_candidate, action)

            diagnostic = {
                "action": action,
                "event_centers": [],
                "identity_scores": [],
                "observability_scores": [],
                "localization_validity_scores": [],
                "acceptance_scores": [],
            }
            if action in (
                    "global_unresolved", "verify") and recovery is not None:
                diagnostic["event_centers"] = list(
                    recovery.get("_proposal_centers", []))
                diagnostic["identity_scores"] = (
                    recovery["identity_scores"][0]
                    .detach().cpu().tolist()
                )
                for key in (
                        "observability_scores",
                        "localization_validity_scores",
                        "acceptance_scores"):
                    diagnostic[key] = (
                        recovery[key][0].detach().cpu().tolist())
            if not hasattr(self, "_recovery_diagnostics"):
                self._recovery_diagnostics = []
            self._recovery_diagnostics.append(diagnostic)
            expert_states = (
                local_candidate.get("expert_states")
                if local_candidate is not None else None)
            expert_names = tuple(getattr(self, "expert_names", ()))
            if not expert_names:
                expert_names = tuple(getattr(
                    getattr(self, "network", None), "expert_names", ()))
            if not expert_names:
                expert_names = DEFAULT_EXPERT_NAMES
            if expert_states is None:
                expert_states = [
                    list(self.state) for _ in expert_names]
            elif decision["exit_to_tracking"]:
                expert_states = [list(box) for box in expert_states]
                try:
                    visibility_id = expert_names.index(
                        "visibility_foc_ov")
                except ValueError as error:
                    raise RuntimeError(
                        "recovery diagnostics require visibility_foc_ov") from error
                expert_states[visibility_id] = list(self.state)
            if not hasattr(self, "_expert_diagnostics"):
                self._expert_diagnostics = []
            self._expert_diagnostics.append(expert_states)
            if not hasattr(self, "_search_diagnostics"):
                self._search_diagnostics = []
            self._record_search_diagnostic(
                search_state,
                resize_factor,
                action,
                local_candidate,
                previous_action=previous_action.value,
                controller_output_score=srbt_control.output_score,
                recovery_attempted=recovery_attempted,
                recovery_confirmed=confirmation is not None,
                recovery_max_identity=self._recovery_diagnostic_value(
                    recovery, "identity_scores"),
                recovery_max_localization=self._recovery_diagnostic_value(
                    recovery, "localization_scores"),
                recovery_accepted_count=self._recovery_diagnostic_value(
                    recovery, "accepted", count=True),
            )

        motion_event_search = (
            local_candidate.get("motion_event_search", event_search)
            if local_candidate is not None else event_search)
        self._commit_motion_event_search(motion_event_search)

        # The current observation is committed only after its action is known.
        tracking_result_arr, tracking_result_event_arr, _, tracking_result_amask_arr = sample_target(
            im=image, eim=event_image, target_bb=self.state,
            search_area_factor=self.params.template_factor, output_sz=self.params.template_size)
        prediction_image = self.preprocessor.process(tracking_result_arr, tracking_result_amask_arr).tensors
        prediction_event_image = self.preprocessor.process(tracking_result_event_arr, tracking_result_amask_arr).tensors
        if srbt_control is not None:
            allow_recent_write = (
                srbt_control.allow_recent_write
                and action == "track" and not is_absent)
            allow_long_write = (
                srbt_control.allow_long_write and allow_recent_write)
        else:
            raise RuntimeError("SRBT controller output is required")

        memory_frame_open = bool(
            local_candidate is not None
            and local_candidate.get("memory_frame_open", False))
        if memory_frame_open:
            self.thor_wrapper.commit(
                prediction_image, prediction_event_image, pred_score,
                allow_recent_write=allow_recent_write,
                allow_long_write=allow_long_write,
            )

        if allow_recent_write:
            self._last_trusted_zi = prediction_image.detach().clone()
            self._last_trusted_ze = prediction_event_image.detach().clone()

        if self.debug and not self.use_visdom:
            x1, y1, w, h = self.state
            import cv2
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                          color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(self.save_dir, "%04d.jpg" % self.frame_id), image_BGR)

        return {"target_bbox": output_state,
                "prediction_image": prediction_image,
                "prediction_event_image": prediction_event_image,
                "response": response if action in (
                    "track", "local_unresolved") else None,
                "pred_score": pred_score,
                "absent": is_absent,
                "allow_recent_write": bool(allow_recent_write),
                "allow_long_write": bool(allow_long_write)}

    @staticmethod
    def _first_debug_value(value):
        if value is None:
            return ""
        if torch.is_tensor(value):
            if value.numel() == 0:
                return ""
            value = value.detach().flatten()[0].item()
        elif isinstance(value, (list, tuple)):
            if not value:
                return ""
            value = value[0]
        return str(value)

    def _run_event_recovery(self, image, event_image, H, W):
        event_tensor = torch.as_tensor(
            event_image, device=self.device, dtype=torch.float32)
        if event_tensor.ndim != 3:
            raise ValueError("event_image must have shape (H, W, C)")
        event_tensor = event_tensor.permute(2, 0, 1).unsqueeze(0)
        proposals = self.event_proposal_extractor(event_tensor)
        valid_ids = proposals["valid"][0].nonzero(
            as_tuple=False).flatten()
        if valid_ids.numel() == 0:
            return None

        clean = self.thor_wrapper.get_clean_template()
        if clean is None:
            raise RuntimeError("Event-guided recovery requires a clean template")
        clean_rgb, clean_event = clean
        centers = proposals["centers"][0].index_select(0, valid_ids)
        event_scores = proposals["scores"][0].index_select(0, valid_ids)
        heatmap = proposals["heatmap"][0, 0]
        heatmap = heatmap / heatmap.max().clamp_min(1e-8)
        prior_image = np.repeat(
            heatmap.detach().cpu().numpy()[..., None], 3, axis=2)

        candidate_rgb = []
        candidate_event = []
        event_priors = []
        anchors = []
        resize_factors = []
        target_width = max(float(self.state[2]), 1.0)
        target_height = max(float(self.state[3]), 1.0)
        for center in centers:
            center_x = float(center[0]) * max(W - 1, 1)
            center_y = float(center[1]) * max(H - 1, 1)
            anchor = [
                center_x - 0.5 * target_width,
                center_y - 0.5 * target_height,
                target_width,
                target_height,
            ]
            rgb_patch, event_patch, resize_factor, mask = sample_target(
                im=image,
                eim=event_image,
                target_bb=anchor,
                search_area_factor=self.recovery_search_factor,
                output_sz=self.params.search_size,
            )
            prior_patch, _, _, _ = sample_target(
                im=prior_image,
                eim=prior_image,
                target_bb=anchor,
                search_area_factor=self.recovery_search_factor,
                output_sz=self.params.search_size,
            )
            candidate_rgb.append(
                self.preprocessor.process(rgb_patch, mask).tensors)
            candidate_event.append(
                self.preprocessor.process(event_patch, mask).tensors)
            prior_tensor = torch.as_tensor(
                prior_patch[..., 0], device=self.device,
                dtype=torch.float32)
            event_priors.append(prior_tensor)
            anchors.append(anchor)
            resize_factors.append(resize_factor)

        recovery = self.network.recover_from_candidates(
            clean_rgb,
            clean_event,
            torch.cat(candidate_rgb, dim=0).unsqueeze(0),
            torch.cat(candidate_event, dim=0).unsqueeze(0),
            event_scores.unsqueeze(0),
            torch.stack(event_priors, dim=0).unsqueeze(0),
        )
        recovery["_anchors"] = anchors
        recovery["_resize_factors"] = resize_factors
        recovery["_proposal_centers"] = [
            [
                float(center[0]) * max(W - 1, 1),
                float(center[1]) * max(H - 1, 1),
            ]
            for center in centers
        ]
        return recovery

    def _update_event_recovery_hypotheses(self, recovery, H, W):
        crop_boxes = recovery["boxes"][0]
        accepted = recovery["accepted"][0].to(dtype=torch.bool)
        identity_scores = recovery["identity_scores"][0]
        localization_scores = recovery["localization_scores"][0]
        global_boxes = []
        for index, crop_box in enumerate(crop_boxes):
            resize_factor = float(recovery["_resize_factors"][index])
            anchor = recovery["_anchors"][index]
            scaled_box = (
                crop_box * self.params.search_size / resize_factor).tolist()
            xywh = clip_box(
                self.map_box_back(scaled_box, resize_factor, anchor),
                H, W, margin=0)
            x, y, width, height = xywh
            global_boxes.append(crop_box.new_tensor([
                (x + 0.5 * width) / W,
                (y + 0.5 * height) / H,
                width / W,
                height / H,
            ]))
        global_boxes = torch.stack(global_boxes)
        identities = torch.stack(
            (identity_scores, localization_scores), dim=-1)
        observed = {
            "boxes": global_boxes,
            "field_scores": recovery["acceptance_scores"][0],
            "identity": identities,
            "active_mask": accepted,
        }
        state = self.hypothesis_tracker.update(
            self._redetect_hypotheses, observed)
        self._redetect_hypotheses = state
        if state["active_count"] == 0:
            return None, 0.0
        cx, cy, width, height = state["boxes"][0].tolist()
        box = clip_box([
            (cx - 0.5 * width) * W,
            (cy - 0.5 * height) * H,
            width * W,
            height * H,
        ], H, W, margin=0)
        confidence = float(state["weights"][0].clamp(0.0, 1.0).item())
        return box, confidence

    def _run_redetection(self, image, event_image, H, W):
        """GLOBAL re-localization: crop a large region covering (most of) the
        full image, encode it to a feature map, and run the redetection expert
        with the clean template + event reappear prior.

        This is the fix for the "target drifted out of the local window can
        never be recovered" failure: we search globally, not in the local
        4-5x tracking window.
        """
        if self.network.redetect_expert is None:
            raise RuntimeError(
                "redetection was requested but the redetection expert is disabled")
        redetect_cfg = getattr(self.cfg.MODEL, "REDETECT", None)
        use_template = bool(getattr(
            redetect_cfg, "USE_TEMPLATE_CONDITIONING", True
        )) if redetect_cfg is not None else True
        if use_template:
            clean = self.thor_wrapper.get_clean_template()
            if clean is None:
                raise RuntimeError(
                    "template-conditioned redetection requires a clean template")
            redetect_zi, redetect_ze = clean
        else:
            redetect_zi, redetect_ze = self.static_zi, self.static_ze

        # Use the same image-centered full-frame crop contract as training.
        anchor_side = float(max(H, W)) / self.redetect_factor
        center = [0.5 * (W - anchor_side), 0.5 * (H - anchor_side),
                  anchor_side, anchor_side]
        full_patch, full_event_patch, rd_resize, rd_amask = sample_target(
            im=image, eim=event_image, target_bb=center,
            search_area_factor=self.redetect_factor,
            output_sz=self.params.search_size)
        full_search = self.preprocessor.process(full_patch, rd_amask).tensors
        full_event_search = self.preprocessor.process(full_event_patch, rd_amask).tensors

        with torch.no_grad():
            red_out = self.network.redetect_from_observations(
                redetect_zi,
                redetect_ze,
                full_search,
                full_event_search,
                dynamic_zi=self.dynamic_zi,
                dynamic_ze=self.dynamic_ze,
                prior_H=None,
                use_template_conditioning=use_template,
            )
        red_out['_resize_factor'] = rd_resize
        red_out['_patch_size'] = self.params.search_size
        red_out['_crop_center'] = (0.5 * W, 0.5 * H)
        return red_out

    def _update_redetect_hypotheses(self, red_out):
        observed = red_out.get("hypotheses")
        if observed is None:
            raise RuntimeError("redetection output is missing hypotheses")
        state = self.hypothesis_tracker.update(
            self._redetect_hypotheses, observed)
        self._redetect_hypotheses = state
        if state["active_count"] == 0:
            return None, 0.0
        rd_resize = red_out.get("_resize_factor", 1.0)
        patch_size = red_out.get("_patch_size", self.params.search_size)
        crop_center = red_out.get("_crop_center")
        if crop_center is None:
            raise RuntimeError("redetection output is missing crop center")
        box = crop_cxcywh_to_image_xywh(
            state["boxes"][0], rd_resize, patch_size, crop_center).tolist()
        confidence = float(state["weights"][0].clamp(0.0, 1.0).item())
        return box, confidence

    def map_box_back(self, pred_box, resize_factor, reference_state=None):
        reference_state = self.state if reference_state is None else reference_state
        cx_prev = reference_state[0] + 0.5 * reference_state[2]
        cy_prev = reference_state[1] + 0.5 * reference_state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def get_tracker_class():
    return PETTrack
