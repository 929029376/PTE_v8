"""PET-Track inference with causal SRBT control and protected THOR memory.

A real SRBT posterior selects TRACK, HOLD, or REDETECT and controls whether the
current observation may enter short- and long-term memory. THOR therefore reads
templates before inference and commits the current frame only after the action is
known.
"""
import os

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
    build_belief_controller,
)
from lib.train.trainers.base_trainer import validate_srbt_checkpoint_schema


def _load_srbt_eval_checkpoint(network, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    validate_srbt_checkpoint_schema(checkpoint)
    network.load_state_dict(checkpoint["net"], strict=True)
    return checkpoint


class PETTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super().__init__(params)
        network = build_pet_track(params.cfg, training=False)
        _load_srbt_eval_checkpoint(network, self.params.checkpoint)
        self.cfg = params.cfg
        self.hypothesis_tracker = build_hypothesis_tracker(self.cfg)
        self.belief_controller = build_belief_controller(self.cfg)
        self._srbt_last_action = BeliefAction.TRACK
        self._redetect_hypotheses = None
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

    def _reset_srbt_sequence_state(self):
        self._srbt_posterior = None
        self._srbt_last_action = BeliefAction.TRACK
        self._redetect_hypotheses = None
        self._pending_redetect_box = None

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
            self.belief_controller.reset()
            self._reset_srbt_sequence_state()
            self._last_score_peak = 0.0
            self._last_redetect_conf = 0.0
            self._last_redetect_error = ""

        self.state = info['init_bbox']
        self.frame_id = idx

    def _run_local_candidate(self, search, event_search,
                             resize_factor, height, width):
        """Run the local path without committing its box."""
        dynamic_zi, dynamic_ze = self.thor_wrapper.begin_frame()
        self.dynamic_zi, self.dynamic_ze = dynamic_zi, dynamic_ze

        out_dict = self.network.inference(
            static_zi=self.static_zi, static_ze=self.static_ze,
            dynamic_zi=dynamic_zi, dynamic_ze=dynamic_ze,
            xi=search, xe=event_search,
            previous_posterior=getattr(self, "_srbt_posterior", None))
        self._srbt_posterior = out_dict.get("srbt_posterior")
        response = self.output_window * out_dict['score_map']
        pred_box = (
            out_dict["target_bbox"][0]
            * self.params.search_size / resize_factor
        ).tolist()
        candidate_state = clip_box(
            self.map_box_back(pred_box, resize_factor),
            height, width, margin=10)

        return {
            "state": candidate_state,
            "score_peak": float(response.max().item()),
            "response": response,
            "srbt_posterior": out_dict.get("srbt_posterior"),
            "srbt_best_hypothesis": out_dict.get("srbt_best_hypothesis"),
            "memory_frame_open": True,
        }

    def get_update_count(self):
        return self.thor_wrapper.get_update_count()

    def get_sample_count(self):
        return self.thor_wrapper.get_sample_count()

    def _step_srbt_controller(self, local_candidate):
        if local_candidate is None:
            return None
        posterior = local_candidate.get("srbt_posterior")
        if posterior is None:
            return None
        return self.belief_controller.step(
            posterior,
            local_candidate.get("srbt_best_hypothesis"),
            self.frame_id,
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

    def track(self, image, event_image, info: dict = None):
        self.frame_id += 1
        self._last_redetect_error = ""
        H, W, _ = image.shape
        x_patch_arr, event_x_patch_arr, resize_factor, x_amask_arr = sample_target(
            im=image, eim=event_image, target_bb=self.state,
            search_area_factor=self.params.search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr).tensors
        event_search = self.preprocessor.process(event_x_patch_arr, x_amask_arr).tensors

        with torch.no_grad():
            local_candidate = self._run_local_candidate(
                search, event_search, resize_factor, H, W)

            srbt_control = self._step_srbt_controller(local_candidate)

            if srbt_control is None:
                raise RuntimeError(
                    "SRBT inference requires srbt_posterior from network output")
            posterior = local_candidate["srbt_posterior"]
            previous_action = self._srbt_last_action
            current_action = srbt_control.action
            decision = {
                "action": current_action.value,
                "enter_frozen": (
                    previous_action is BeliefAction.TRACK
                    and current_action is not BeliefAction.TRACK),
                "enter_redetect": (
                    previous_action is not BeliefAction.REDETECT
                    and current_action is BeliefAction.REDETECT),
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

            if decision["exit_to_tracking"]:
                self.thor_wrapper.resume()

            pred_score = 0.0
            is_absent = False
            response = None

            if action == "track":
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
            elif action in ("freeze", "hold"):
                is_absent = True
                if action in ("freeze", "hold"):
                    self._pending_redetect_box = None
                    self._redetect_hypotheses = None
                    self._last_redetect_conf = 0.0
                # Keep the search window where it was; do not pollute memory.
            elif action == "redetect":
                is_absent = True
                # Run GLOBAL redetection using the clean template + event prior.
                red_out = self._run_redetection(image, event_image, H, W)
                if red_out is not None:
                    box, conf = self._update_redetect_hypotheses(red_out)
                    self._last_redetect_conf = conf
                    accepted = (
                        box is not None
                        and conf > 0.0)
                    if accepted:
                        self._pending_redetect_box = box
                    else:
                        self._pending_redetect_box = None
                else:
                    self._last_redetect_conf = 0.0
                    self._pending_redetect_box = None
            elif action == "resume":
                if self._pending_redetect_box is None:
                    action = "hold"
                    is_absent = True
                else:
                    self.state = clip_box(
                        self._pending_redetect_box, H, W, margin=10)
                    self._pending_redetect_box = None
                    self.thor_wrapper.resume()
                    is_absent = False
                    pred_score = self._last_redetect_conf

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
                "response": response if action == "track" else None,
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

        prior_H = None
        red_out = self.network.redetect(
            feat_map, prior_H=prior_H, template_tokens=clean_tokens)
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
