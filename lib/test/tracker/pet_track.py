"""
PET-Track inference tracker.

Wraps the baseline inference loop and adds the C3 occlusion-aware controller,
now driven by the unified event physical belief b_t and the learned
MemoryPolicyHead gates (instead of hand-tuned theta_abs / theta_z thresholds):

  belief = EventPhysicalBelief(event_image, roi)        # computed ONCE / frame
  absence_prob = AbsencePredictor(belief, score, sim)
  freeze_prob, redetect_prob = MemoryPolicyHead(belief, frozen_age, absence)
  decision = state_machine.step(absence_prob, redetect_prob, redetect_conf, ...)

When the policy gates decide the target is absent, the tracker enters FROZEN
(memory protected + clean template snapshot) and then REDETECT (global
re-localization guided by the event reappear prior from the belief module),
instead of drifting forever after an occlusion.

The normal TRACKING state preserves the inherited local crop/memory workflow
while running the v8 routed model. The state machine's counters/timeouts
(w_abs / w_re / w_fail / T_max / T_period / T_min_bg) are retained as a safety
skeleton; only the two most brittle thresholds became learned gates.
"""
import math
import os

import torch
import torch.nn.functional as F

from lib.models.pet_track import build_pet_track
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond, generate_mask_z
from lib.models.layers.thor import THOR_Wrapper
from lib.models.layers.state_machine import OcclusionStateMachine, State
from lib.models.layers.expert_router import SparseRouteHysteresis
from lib.utils.route_motion import ROUTE_MOTION_DIM, causal_route_motion_cues


def _bbox_to_crop_pixel_roi(box, resize_factor, output_size, device, box_extract=None):
    """Map an image-space xywh box to pixel xywh coordinates in its crop.

    The belief module receives the resized search crop, so ROI statistics must be
    computed in crop coordinates, not original image coordinates.
    """
    box_in = torch.tensor(box, dtype=torch.float32, device=device)
    if box_extract is None:
        box_extract = box_in
    else:
        box_extract = torch.tensor(box_extract, dtype=torch.float32, device=device)
    crop_sz = torch.tensor([output_size, output_size], dtype=torch.float32, device=device)
    return transform_image_to_crop(
        box_in, box_extract, resize_factor, crop_sz, normalize=False).view(1, 4)


def _redetect_uses_prior(cfg):
    """Return whether the redetection expert should consume the event prior."""
    redetect_cfg = getattr(getattr(cfg, "MODEL", None), "REDETECT", None)
    if redetect_cfg is None:
        return False
    # The current C3 training path supervises RedetectionExpert with
    # prior_H=None. Injecting an event prior at test time would shift the score
    # distribution for the best checkpoint, so require an explicit marker for
    # future checkpoints that are trained with prior modulation enabled.
    return (bool(getattr(redetect_cfg, "USE_PRIOR", True)) and
            bool(getattr(redetect_cfg, "TRAINED_WITH_PRIOR", False)))


def _parse_force_route(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        route = tuple(str(name).strip() for name in value if str(name).strip())
    else:
        text = str(value).strip()
        if text.lower() in ("", "none", "null"):
            return None
        route = tuple(
            name.strip() for name in text.replace(",", "+").split("+")
            if name.strip())
    return route or None


class PETTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super().__init__(params)
        network = build_pet_track(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug and not self.use_visdom:
            self.save_dir = "debug"
            os.makedirs(self.save_dir, exist_ok=True)

        self.save_all_boxes = params.save_all_boxes
        self.policy_mode = str(getattr(params.cfg.TEST, "POLICY_MODE", "stateful")).lower()
        self.use_train_compatible_policy = self.policy_mode in ("train_compatible", "train", "strict_train")
        self.force_route = _parse_force_route(
            getattr(params.cfg.TEST, "FORCE_ROUTE", ""))
        expert_cfg = params.cfg.MODEL.EXPERT
        self.route_hysteresis = SparseRouteHysteresis(
            self.network.route_options,
            margin=float(getattr(
                expert_cfg, "ROUTE_HYSTERESIS_MARGIN", 0.05)),
            patience=int(getattr(
                expert_cfg, "ROUTE_HYSTERESIS_PATIENCE", 2)),
            confidence_threshold=float(getattr(
                expert_cfg, "ROUTER_CONFIDENCE_THRESHOLD", 0.0)),
        )
        self.thor_wrapper = THOR_Wrapper(
            net=self.network,
            st_capacity=params.cfg.TEST.SHORTTERM_LIBRARY_NUMS,
            lt_capacity=params.cfg.TEST.LONGTERM_LIBRARY_NUMS,
            sample_interval=params.cfg.TEST.SAMPLE_INTERVAL,
            update_interval=params.cfg.TEST.UPDATE_INTERVAL,
            lower_bound=params.cfg.TEST.LOWER_BOUND,
            score_threshold=params.cfg.TEST.SCORE_THRESHOLD)

        # --- C3 occlusion controller (learned gates + safety skeleton) ---
        sm_cfg = getattr(self.cfg.MODEL, "STATE_MACHINE", None)
        pet_cfg = getattr(self.cfg.MODEL, "PET", None)
        use_learned = bool(getattr(pet_cfg, "USE_LEARNED_POLICY", True)) \
            if pet_cfg is not None else True
        mp_cfg = getattr(self.cfg.MODEL, "MEMORY_POLICY", None)
        gate_thresh = float(getattr(mp_cfg, "GATE_THRESH", 0.5)) if mp_cfg is not None else 0.5
        self.state_machine = OcclusionStateMachine(
            gate_thresh=gate_thresh,
            w_abs=getattr(sm_cfg, "W_ABS", 3),
            theta_re=getattr(sm_cfg, "THETA_RE", 0.7),
            w_re=getattr(sm_cfg, "W_RE", 2),
            w_fail=getattr(sm_cfg, "W_FAIL", 5),
            T_max=getattr(sm_cfg, "T_MAX", 50),
            T_period=getattr(sm_cfg, "T_PERIOD", 10),
            T_min_bg=getattr(sm_cfg, "T_MIN_BG", 3),
            use_learned_policy=use_learned,
            legacy_theta_abs=getattr(sm_cfg, "THETA_ABS", 0.6),
            legacy_theta_z=getattr(sm_cfg, "THETA_Z", 2.5))
        self.t_max = float(getattr(sm_cfg, "T_MAX", 50)) if sm_cfg is not None else 50.0
        # Last accepted cues are retained only for the optional event-skip
        # diagnostic path. Formal v8 policy uses the current local candidate.
        self._last_score_peak = 0.0
        self._last_sim_zx = 0.0
        self._last_redetect_conf = 0.0
        self._last_redetect_error = ""
        self._pending_redetect_box = None
        self._skip_count = 0  # event-trigger: consecutive skipped frames
        # Redetect full-image crop config.
        redetect_cfg = getattr(self.cfg.MODEL, "REDETECT", None)
        self.redetect_factor = float(getattr(
            redetect_cfg, "TRAIN_SEARCH_FACTOR", 8.0
        )) if redetect_cfg is not None else 8.0
        self.redetect_size = int(getattr(redetect_cfg, "FEAT_SZ", 12)) if redetect_cfg is not None else 12

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
            self._last_trusted_zi = template.detach().clone()
            self._last_trusted_ze = event_template.detach().clone()
            self.thor_wrapper.setup(self.static_zi, self.static_ze)
            # Reset the belief history and the state machine for the new sequence.
            if self.network.event_belief is not None:
                self.network.event_belief.reset()
                self._seed_event_belief(
                    event_template, info['init_bbox'], resize_factor)
            self.state_machine.reset()
            self.route_hysteresis.reset()
            # Reset accepted-candidate/redetect cues for the new sequence.
            self._last_score_peak = 0.0
            self._last_sim_zx = 0.0
            self._last_redetect_conf = 0.0
            self._last_redetect_error = ""
            self._pending_redetect_box = None
            self._skip_count = 0

        self.state = info['init_bbox']
        self._route_box_history = [list(info['init_bbox'])]
        self.frame_id = idx

    def _causal_route_motion(self, height, width, device):
        if len(self._route_box_history) < 2:
            return torch.zeros(
                1, ROUTE_MOTION_DIM, device=device, dtype=torch.float32)
        cues = causal_route_motion_cues(
            self._route_box_history[-2], self._route_box_history[-1],
            (height, width), device=device)
        return cues.unsqueeze(0)

    def _step_event_belief(self, event_search, roi, commit=True):
        event_belief = self.network.event_belief
        if event_belief is None:
            return None

        event_frame = event_search.squeeze(0)
        # Match training causality in every policy mode: query the current
        # search frame against only past observations, then commit it for the
        # next frame. A frozen belief ignores the commit by construction.
        result = event_belief(event_frame, roi=roi, update_history=False)
        if commit:
            self._commit_event_belief(event_search, roi)
        return result

    def _commit_event_belief(self, event_search, roi):
        event_belief = self.network.event_belief
        if event_belief is None:
            return
        event_belief(
            event_search.squeeze(0), roi=roi, update_history=True)

    def _seed_event_belief(self, event_template, initial_box,
                           resize_factor):
        event_belief = self.network.event_belief
        if event_belief is None:
            return
        roi = _bbox_to_crop_pixel_roi(
            initial_box,
            resize_factor,
            self.params.template_size,
            event_template.device,
        )
        event_belief(event_template, roi=roi, update_history=True)

    def _predict_c3_policy(self, belief_embed, raw_stats, score_peak,
                           sim_zx, frozen_age):
        """Evaluate C3 heads from the current local candidate cues."""
        result = {
            "absence_prob": 0.0,
            "freeze_prob": None,
            "redetect_prob": 0.0,
        }
        if belief_embed is None:
            return result

        device = belief_embed.device
        if self.network.absence_predictor is not None:
            score = torch.as_tensor(
                [score_peak], device=device, dtype=torch.float32)
            similarity = torch.as_tensor(
                [sim_zx], device=device, dtype=torch.float32)
            absence = self.network.predict_absence(
                belief_embed, score, similarity, raw_stats=raw_stats)
            result["absence_prob"] = float(absence[0].item())

        if self.network.memory_policy is not None:
            age = torch.as_tensor(
                [frozen_age], device=device, dtype=torch.float32)
            absence = torch.as_tensor(
                [result["absence_prob"]], device=device,
                dtype=torch.float32)
            gates = self.network.predict_memory_policy(
                belief_embed, age, absence, raw_stats=raw_stats)
            result["freeze_prob"] = float(
                gates["freeze_prob"][0].item())
            result["redetect_prob"] = float(
                gates["redetect_prob"][0].item())
        return result

    def _run_local_candidate(self, search, event_search, belief_embed,
                             raw_stats, route_motion, resize_factor,
                             height, width, info):
        """Run the train-compatible local path without committing its box."""
        previous_output = info.get('previous_output', {}) if info else {}
        if not previous_output:
            dynamic_zi, dynamic_ze = self.thor_wrapper.update(
                self.static_zi, self.static_ze, 1)
        else:
            dynamic_zi, dynamic_ze = self.thor_wrapper.update(
                previous_output['prediction_image'],
                previous_output['prediction_event_image'],
                previous_output['pred_score'])
        self.dynamic_zi, self.dynamic_ze = dynamic_zi, dynamic_ze

        out_dict = self.network.inference(
            static_zi=self.static_zi, static_ze=self.static_ze,
            dynamic_zi=dynamic_zi, dynamic_ze=dynamic_ze,
            xi=search, xe=event_search, belief=belief_embed,
            raw_stats=raw_stats, route=self.force_route,
            route_motion=route_motion,
            route_selector=self.route_hysteresis.select)
        response = self.output_window * out_dict['score_map']
        pred_boxes = self.network.box_head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map'])
        pred_box = (
            pred_boxes.view(-1, 4).mean(dim=0)
            * self.params.search_size / resize_factor
        ).tolist()
        candidate_state = clip_box(
            self.map_box_back(pred_box, resize_factor),
            height, width, margin=10)

        router_confidence = out_dict.get("route_confidence")
        tail_router_confidence = out_dict.get(
            "tail_router_confidence")
        return {
            "state": candidate_state,
            "score_peak": float(response.max().item()),
            "sim_zx": self._compute_sim_zx_inference(out_dict),
            "response": response,
            "tail_raw_expert": self._first_debug_value(
                out_dict.get("tail_router_selected_routes")),
            "tail_expert": self._first_debug_value(
                out_dict.get("tail_routes")),
            "router_expert": self._first_debug_value(
                out_dict.get("route_selected")),
            "head_expert": self._first_debug_value(
                out_dict.get("route_routed")),
            "router_confidence": (
                float(router_confidence.flatten()[0].item())
                if router_confidence is not None else -1.0),
            "tail_router_confidence": (
                float(tail_router_confidence.flatten()[0].item())
                if tail_router_confidence is not None else -1.0),
        }

    def get_update_count(self):
        return self.thor_wrapper.get_update_count()

    def get_sample_count(self):
        return self.thor_wrapper.get_sample_count()

    def track(self, image, event_image, info: dict = None):
        self.frame_id += 1
        self._last_redetect_error = ""
        H, W, _ = image.shape
        x_patch_arr, event_x_patch_arr, resize_factor, x_amask_arr = sample_target(
            im=image, eim=event_image, target_bb=self.state,
            search_area_factor=self.params.search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr).tensors
        event_search = self.preprocessor.process(event_x_patch_arr, x_amask_arr).tensors
        route_motion = self._causal_route_motion(
            H, W, event_search.device)

        with torch.no_grad():
            # --- Unified event physical belief (computed ONCE this frame) ---
            belief_embed = None
            raw_stats = None
            history_ready = False
            rho_global = 1.0
            if self.network.event_belief is not None:
                roi = _bbox_to_crop_pixel_roi(
                    self.state, resize_factor, self.params.search_size, event_search.device)
                eb_dict = self._step_event_belief(
                    event_search, roi, commit=False)
                belief_embed = eb_dict['belief']
                raw_stats = eb_dict['raw_stats']
                history_ready = bool(eb_dict['history_ready'])
                if raw_stats is not None:
                    rho_global = float(raw_stats[0, 0].item())

            # Formal v8 configs disable event skipping. Keep the diagnostic
            # path, but otherwise run the same local forward used in training
            # before C3 decides whether the candidate is trusted.
            et = getattr(self.cfg.MODEL, "EVENT_TRIGGER", None)
            et_enable = (bool(getattr(et, "ENABLE", False)) if et is not None else False) \
                and not self.use_train_compatible_policy
            skip_frame = (
                et_enable
                and self.state_machine.state == State.TRACKING
                and self._skip_count < int(getattr(et, "MAX_SKIP", 5))
                and rho_global < float(getattr(et, "THETA_LOW", 0.02))
            )
            if (et is not None
                    and rho_global >= float(getattr(et, "THETA_HIGH", 0.15))):
                self._skip_count = 0

            local_candidate = None
            if not skip_frame:
                local_candidate = self._run_local_candidate(
                    search, event_search, belief_embed, raw_stats,
                    route_motion, resize_factor, H, W, info)

            current_score = (
                local_candidate["score_peak"]
                if local_candidate is not None else self._last_score_peak)
            current_sim = (
                local_candidate["sim_zx"]
                if local_candidate is not None else self._last_sim_zx)
            frozen_age = (
                float(self.state_machine.frames_in_frozen) / self.t_max
                if not self.use_train_compatible_policy else 0.0)
            policy = self._predict_c3_policy(
                belief_embed,
                raw_stats,
                score_peak=local_candidate["score_peak"]
                if local_candidate is not None else current_score,
                sim_zx=local_candidate["sim_zx"]
                if local_candidate is not None else current_sim,
                frozen_age=frozen_age,
            )
            absence_prob = policy["absence_prob"]
            freeze_prob = policy["freeze_prob"]
            redetect_prob = policy["redetect_prob"]

            if self.use_train_compatible_policy:
                redetect_signal = redetect_prob
                decision = {"action": "track", "enter_frozen": False,
                            "enter_redetect": False,
                            "exit_to_tracking": False}
            else:
                redetect_signal = (
                    redetect_prob
                    if self.state_machine.use_learned_policy
                    else float(raw_stats[0, 5].item())
                    if raw_stats is not None else 0.0)
                decision = self.state_machine.step(
                    absence_prob, redetect_signal,
                    redetect_conf=self._last_redetect_conf,
                    history_ready=history_ready, freeze_prob=freeze_prob)
            action = decision["action"]

            # Query-before-commit: the frame that enters FROZEN contributes
            # neither its event observation nor its rejected local bbox.
            if (self.network.event_belief is not None
                    and (self.use_train_compatible_policy
                         or action == "track")):
                self._commit_event_belief(event_search, roi)

            if (not self.use_train_compatible_policy
                    and decision["enter_frozen"]):
                self._pending_redetect_box = None
                self._last_redetect_conf = 0.0
                self.thor_wrapper.freeze(True)
                self.thor_wrapper.snapshot_clean(
                    self._last_trusted_zi, self._last_trusted_ze)
                if self.network.event_belief is not None:
                    self.network.event_belief.freeze_history(True)
                    self.network.event_belief.init_background(
                        self.network.event_belief.energy_map(
                            event_search.squeeze(0)))

            if (not self.use_train_compatible_policy
                    and self.network.event_belief is not None
                    and self.state_machine.state == State.FROZEN):
                self.network.event_belief.update_background(
                    self.network.event_belief.energy_map(
                        event_search.squeeze(0)))

            pred_score = 0.0
            is_absent = False
            response = None
            tail_raw_expert = (
                local_candidate["tail_raw_expert"]
                if local_candidate is not None else "")
            tail_expert = (
                local_candidate["tail_expert"]
                if local_candidate is not None else "")
            router_expert = (
                local_candidate["router_expert"]
                if local_candidate is not None else "")
            head_expert = (
                local_candidate["head_expert"]
                if local_candidate is not None else "")
            router_confidence = (
                local_candidate["router_confidence"]
                if local_candidate is not None else -1.0)
            tail_router_confidence = (
                local_candidate["tail_router_confidence"]
                if local_candidate is not None else -1.0)
            route_fallback = 0

            if skip_frame and action == "track":
                self._skip_count += 1
                pred_score = self._last_score_peak
            elif action == "track":
                if local_candidate is None:
                    raise RuntimeError(
                        "tracking action requires a current local candidate")
                self.state = local_candidate["state"]
                pred_score = local_candidate["score_peak"]
                response = local_candidate["response"]
                self._last_score_peak = pred_score
                self._last_sim_zx = local_candidate["sim_zx"]
                self._skip_count = 0
            elif action in ("freeze", "hold"):
                is_absent = True
                if action == "freeze":
                    self._pending_redetect_box = None
                # Keep the search window where it was; do not pollute memory.
            elif action == "redetect":
                is_absent = True
                # Run GLOBAL redetection using the clean template + event prior.
                red_out = self._run_redetection(image, event_image, H, W)
                if red_out is not None:
                    conf = float(red_out["conf"].item())
                    self._last_redetect_conf = conf
                    if conf > self.state_machine.theta_re:
                        box = self._decode_redetect_box(red_out, H, W)
                        if box is not None:
                            self._pending_redetect_box = box
                        else:
                            self._last_redetect_conf = 0.0
                            self._pending_redetect_box = None
                    else:
                        self._pending_redetect_box = None
                else:
                    self._last_redetect_conf = 0.0
                    self._pending_redetect_box = None
            elif action == "resume":
                if self._pending_redetect_box is None:
                    self.state_machine.state = State.FROZEN
                    action = "hold"
                    is_absent = True
                else:
                    self.state = clip_box(
                        self._pending_redetect_box, H, W, margin=10)
                    self._pending_redetect_box = None
                    self.thor_wrapper.resume()
                    if self.network.event_belief is not None:
                        self.network.event_belief.freeze_history(False)
                    is_absent = False
                    pred_score = self._last_redetect_conf

        # Build the prediction patch for the next frame's template update.
        tracking_result_arr, tracking_result_event_arr, _, tracking_result_amask_arr = sample_target(
            im=image, eim=event_image, target_bb=self.state,
            search_area_factor=self.params.template_factor, output_sz=self.params.template_size)
        prediction_image = self.preprocessor.process(tracking_result_arr, tracking_result_amask_arr).tensors
        prediction_event_image = self.preprocessor.process(tracking_result_event_arr, tracking_result_amask_arr).tensors
        if (action == "track" and not skip_frame and not is_absent
                and pred_score >= float(self.cfg.TEST.SCORE_THRESHOLD)):
            self._last_trusted_zi = prediction_image.detach().clone()
            self._last_trusted_ze = prediction_event_image.detach().clone()

        if self.debug and not self.use_visdom:
            x1, y1, w, h = self.state
            import cv2
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                          color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(self.save_dir, "%04d.jpg" % self.frame_id), image_BGR)

        self._route_box_history.append(list(self.state))
        self._route_box_history = self._route_box_history[-2:]

        return {"target_bbox": self.state,
                "prediction_image": prediction_image,
                "prediction_event_image": prediction_event_image,
                "response": response if action == "track" else None,
                "pred_score": pred_score,
                "absent": is_absent,
                "c3_debug": {
                    "frame_id": int(self.frame_id),
                    "action": str(action),
                    "state": getattr(self.state_machine.state, "name", str(self.state_machine.state)),
                    "absence_prob": float(absence_prob),
                    "freeze_prob": -1.0 if freeze_prob is None else float(freeze_prob),
                    "redetect_prob": float(redetect_prob),
                    "redetect_signal": float(redetect_signal),
                    "redetect_conf": float(self._last_redetect_conf),
                    "redetect_error": str(self._last_redetect_error),
                    "history_ready": int(bool(history_ready)),
                    "rho_global": float(rho_global),
                    "skip_frame": int(bool(skip_frame)),
                    "force_route": "" if self.force_route is None else str(self.force_route),
                    "tail_raw_expert": tail_raw_expert,
                    "tail_expert": tail_expert,
                    "router_expert": router_expert,
                    "head_expert": head_expert,
                    "router_confidence": float(router_confidence),
                    "tail_router_confidence": float(tail_router_confidence),
                    "route_fallback": int(route_fallback),
                    "pred_score": float(pred_score),
                    "is_absent": int(bool(is_absent)),
                }}

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
        clean_tokens = None
        if use_template:
            clean = self.thor_wrapper.get_clean_template()
            if clean is None:
                raise RuntimeError(
                    "template-conditioned redetection requires a clean template")
            clean_zi, clean_ze = clean

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
            # Encode the global crop into a token map, then reshape to a 2D
            # feature grid the redetect head expects.
            x_feat = self.network.backbone._x_feat(full_search)  # (1, N, C)
            C = x_feat.shape[-1]
            f = self.network.feat_sz_s
            feat_map = x_feat.transpose(1, 2).reshape(1, C, f, f)
            if use_template:
                clean_zi_tokens = self.network.backbone._z_feat(
                    clean_zi.unsqueeze(1))
                clean_ze_tokens = self.network.backbone._z_feat(
                    clean_ze.unsqueeze(1))
                clean_tokens = torch.cat(
                    (clean_zi_tokens, clean_ze_tokens), dim=1)

        # Event reappear prior from the global event crop (background-subtracted).
        prior_H = None
        if _redetect_uses_prior(self.cfg):
            if self.network.event_belief is None:
                raise RuntimeError(
                    "prior-conditioned redetection requires event_belief")
            prior_H = self.network.event_belief.reappear_prior(
                self.network.event_belief.energy_map(full_event_search.squeeze(0)))
        red_out = self.network.redetect(
            feat_map, prior_H=prior_H, template_tokens=clean_tokens)
        red_out['_resize_factor'] = rd_resize
        red_out['_patch_size'] = self.params.search_size
        red_out['_crop_center'] = (0.5 * W, 0.5 * H)
        return red_out

    def _decode_redetect_box(self, red_out, H, W):
        """Map the redetect head's normalized box back to image coords."""
        bbox = red_out['bbox'][0]  # cxcywh normalized in [0,1]
        cx, cy, w, h = bbox.tolist()
        rd_resize = red_out.get('_resize_factor', 1.0)
        ps = red_out.get('_patch_size', self.params.search_size)
        # The redetect crop is image-centered in both training and test.
        cx_prev, cy_prev = red_out.get('_crop_center', (0.5 * W, 0.5 * H))
        half_side = 0.5 * ps / rd_resize
        cx_real = cx * ps / rd_resize + (cx_prev - half_side)
        cy_real = cy * ps / rd_resize + (cy_prev - half_side)
        w_real = w * ps / rd_resize
        h_real = h * ps / rd_resize
        return [cx_real - 0.5 * w_real, cy_real - 0.5 * h_real, w_real, h_real]

    def _compute_sim_zx_inference(self, out_dict):
        """Cosine similarity between template and search token features at
        inference (real cue for the absence predictor, replacing the zero
        placeholder)."""
        feat = out_dict.get('backbone_feat')
        if feat is None:
            raise RuntimeError("backbone_feat missing from inference output")
        feat_len_s = self.network.feat_len_s
        feat_len_z = self.network.feat_len_z
        lens_z = feat_len_z * 2
        z = feat[:, :lens_z].mean(dim=1)
        x = feat[:, -feat_len_s:].mean(dim=1)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return float((z * x).sum(dim=-1)[0].item())

    def map_box_back(self, pred_box, resize_factor):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def get_tracker_class():
    return PETTrack
