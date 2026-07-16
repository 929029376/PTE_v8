"""PET-Track inference with event-guided RGB recovery and protected memory."""
import os

import numpy as np
import torch

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
    build_visibility_controller,
)
from lib.models.layers.event_recovery import EventProposalExtractor
from lib.models.layers.expert_ensemble import (
    fuse_expert_predictions,
    normalized_response_psr,
)
from lib.train.trainers.base_trainer import load_srbt_checkpoint_file


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
        self.hypothesis_tracker = build_hypothesis_tracker(self.cfg)
        self.visibility_controller = build_visibility_controller(self.cfg)
        self._srbt_last_action = BeliefAction.TRACK
        self._redetect_hypotheses = None
        self.network = network
        self.device = next(self.network.parameters()).device
        self.network.eval()
        self.preprocessor = Preprocessor(device=self.device)
        self.state = None

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

    def get_recovery_diagnostics(self):
        return [dict(item) for item in self._recovery_diagnostics]

    def get_expert_diagnostics(self):
        return [[list(box) for box in frame]
                for frame in self._expert_diagnostics]

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
            self.visibility_controller.reset()
            self._reset_srbt_sequence_state()
            self._last_score_peak = 0.0
            self._last_redetect_conf = 0.0
            self._last_redetect_error = ""

        self.state = info['init_bbox']
        self._expert_diagnostics = [[
            list(info['init_bbox']) for _ in range(5)
        ]]
        self.frame_id = idx

    def _run_local_candidate(self, search, event_search,
                             resize_factor, height, width,
                             reference_state=None):
        """Run the local path without committing its box."""
        dynamic_zi, dynamic_ze = self.thor_wrapper.begin_frame()
        self.dynamic_zi, self.dynamic_ze = dynamic_zi, dynamic_ze

        out_dict = self.network.inference(
            static_zi=self.static_zi, static_ze=self.static_ze,
            dynamic_zi=dynamic_zi, dynamic_ze=dynamic_ze,
            xi=search, xe=event_search)
        expert_outputs = out_dict.get("expert_outputs")
        if expert_outputs:
            mapped_boxes = []
            response_maps = []
            for expert_output in expert_outputs.values():
                response_maps.append(
                    self.output_window * expert_output["score_map"])
                pred_box = (
                    expert_output["pred_boxes"][0, 0]
                    * self.params.search_size / resize_factor
                ).tolist()
                mapped_boxes.append(clip_box(
                    self.map_box_back(
                        pred_box, resize_factor, reference_state),
                    height, width, margin=10))
            response_stack = torch.cat(response_maps, dim=0)
            boxes = torch.as_tensor(
                mapped_boxes, device=response_stack.device,
                dtype=response_stack.dtype)
            response_peaks = response_stack.flatten(1).max(dim=1).values
            ensemble = fuse_expert_predictions(
                boxes=boxes,
                response_peaks=response_peaks,
                response_psr=normalized_response_psr(response_stack),
                last_box=(self.state if reference_state is None
                          else reference_state),
            )
            candidate_state = clip_box(
                ensemble.box.detach().cpu().tolist(),
                height, width, margin=10)
            response = (
                ensemble.weights[:, None, None, None] * response_stack
            ).sum(dim=0, keepdim=True)
            retained_expert_ids = ensemble.retained_ids
            ensemble_weights = ensemble.weights.detach()
            expert_states = [list(box) for box in mapped_boxes]
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
            expert_states = [list(candidate_state) for _ in range(5)]

        return {
            "state": candidate_state,
            "score_peak": float(response.max().item()),
            "response": response,
            "presence_score": out_dict.get("presence_score"),
            "memory_frame_open": True,
            "retained_expert_ids": retained_expert_ids,
            "ensemble_weights": ensemble_weights,
            "expert_states": expert_states,
        }

    def get_update_count(self):
        return self.thor_wrapper.get_update_count()

    def get_sample_count(self):
        return self.thor_wrapper.get_sample_count()

    def _step_srbt_controller(self, local_candidate,
                              identity_score=None,
                              localization_score=None,
                              presence_score=None):
        if local_candidate is None:
            return None
        presence_score = (
            local_candidate.get("presence_score")
            if presence_score is None else presence_score)
        if presence_score is None:
            return None
        if identity_score is None and localization_score is None:
            return self.visibility_controller.step(presence_score)
        return self.visibility_controller.step(
            presence_score,
            identity_score,
            localization_score,
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
        combined = recovery["combined_scores"][0].index_select(0, accepted_ids)
        best_id = accepted_ids[int(combined.argmax().item())]
        identity = float(recovery["identity_scores"][0, best_id].item())
        localization = float(recovery["localization_scores"][0, best_id].item())
        return identity, localization

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
        search_state = (
            self._pending_redetect_box
            if (self._srbt_last_action in (
                    BeliefAction.ABSENT, BeliefAction.VERIFY)
                and self._pending_redetect_box is not None)
            else self.state
        )
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
            if previous_action in (BeliefAction.ABSENT, BeliefAction.VERIFY):
                recovery = self._run_recovery_cycle(image, event_image, H, W)
                confirmation = self._best_recovery_confirmation(recovery)
                if confirmation is None:
                    srbt_control = self._step_srbt_controller(local_candidate)
                else:
                    identity, localization = confirmation
                    srbt_control = self._step_srbt_controller(
                        local_candidate,
                        identity_score=identity,
                        localization_score=localization,
                        presence_score=localization,
                    )
            else:
                srbt_control = self._step_srbt_controller(local_candidate)
                if srbt_control is not None and srbt_control.action is BeliefAction.ABSENT:
                    recovery = self._run_recovery_cycle(image, event_image, H, W)
                    confirmation = self._best_recovery_confirmation(recovery)
                    if confirmation is not None:
                        identity, localization = confirmation
                        srbt_control = self._step_srbt_controller(
                            local_candidate,
                            identity_score=identity,
                            localization_score=localization,
                            presence_score=localization,
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

            if action in ("track", "suspect"):
                if local_candidate is None:
                    raise RuntimeError(
                        "tracking action requires a current local candidate")
                self.state = self._resolve_tracking_state(
                    local_candidate["state"], H, W,
                    srbt_recovered=(
                        srbt_control is not None
                        and decision["exit_to_tracking"]),
                )
                pred_score = (
                    float(srbt_control.output_score)
                    if srbt_control is not None
                    else local_candidate["score_peak"])
                response = local_candidate["response"]
                self._last_score_peak = pred_score
            elif action == "absent":
                is_absent = True
            elif action == "verify":
                is_absent = True

            diagnostic = {
                "action": action,
                "event_centers": [],
                "identity_scores": [],
            }
            if action in ("absent", "verify") and recovery is not None:
                diagnostic["event_centers"] = list(
                    recovery.get("_proposal_centers", []))
                diagnostic["identity_scores"] = (
                    recovery["identity_scores"][0]
                    .detach().cpu().tolist()
                )
            if not hasattr(self, "_recovery_diagnostics"):
                self._recovery_diagnostics = []
            self._recovery_diagnostics.append(diagnostic)
            expert_states = (
                local_candidate.get("expert_states")
                if local_candidate is not None else None)
            if expert_states is None:
                expert_states = [list(self.state) for _ in range(5)]
            elif decision["exit_to_tracking"]:
                expert_states = [list(box) for box in expert_states]
                expert_states[3] = list(self.state)
            if not hasattr(self, "_expert_diagnostics"):
                self._expert_diagnostics = []
            self._expert_diagnostics.append(expert_states)

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

        return {"target_bbox": self.state,
                "prediction_image": prediction_image,
                "prediction_event_image": prediction_event_image,
                "response": response if action in ("track", "suspect") else None,
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
            "field_scores": recovery["combined_scores"][0],
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
