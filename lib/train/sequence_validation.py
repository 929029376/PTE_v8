import math

import numpy as np
import torch

from lib.train.data.felt_challenges import expert_supervision_mask
from lib.test.analysis.felt_metrics import (
    felt_absent_balanced_accuracy,
    felt_reappearance_auc_proxy,
    felt_success_auc_proxy,
)
from lib.test.utils.params import TrackerParams


RECOVERY_DIAGNOSTIC_KEYS = (
    "EVENT_RECALL_AT_1_HITS",
    "EVENT_RECALL_AT_3_HITS",
    "EVENT_RECALL_AT_5_HITS",
    "EVENT_REAPPEARANCE_COUNT",
    "RGB_FALSE_ACCEPTS",
    "RGB_ABSENT_CANDIDATES",
    "RECOVERY_LATENCY_SUM",
    "RECOVERY_SUCCESS_COUNT",
    "RECOVERY_EVENT_COUNT",
    "REAPPEAR_IOU_AT_1_SUM",
    "REAPPEAR_IOU_AT_1_COUNT",
    "REAPPEAR_IOU_AT_3_SUM",
    "REAPPEAR_IOU_AT_3_COUNT",
    "REAPPEAR_IOU_AT_5_SUM",
    "REAPPEAR_IOU_AT_5_COUNT",
    "VISIBLE_RETENTION_IOU_SUM",
    "VISIBLE_RETENTION_COUNT",
    "TRACK_TIME_SUM",
    "TRACK_FRAME_COUNT",
    "RECOVERY_TIME_SUM",
    "RECOVERY_FRAME_COUNT",
)

EXPERT_NAMES = (
    "generalist",
    "motion_fm",
    "precision_refiner",
    "visibility_foc_ov",
    "discrimination_bi",
)

EXPERT_DIAGNOSTIC_KEYS = tuple(
    f"EXPERT_{expert_id}_{suffix}"
    for expert_id in range(len(EXPERT_NAMES))
    for suffix in (
        "COUNT",
        "IOU_SUM",
        "SUCCESS_HITS",
        "GENERALIST_IOU_SUM",
        "GENERALIST_SUCCESS_HITS",
        "ENSEMBLE_IOU_SUM",
        "ENSEMBLE_SUCCESS_HITS",
    )
)

SMALL_TARGET_DIAGNOSTIC_KEYS = (
    "SMALL_TARGET_COUNT",
    "SMALL_TARGET_CENTER_IN_CROP_HITS",
    "SMALL_TARGET_FULL_BOX_IN_CROP_HITS",
    "SMALL_TARGET_VISIBLE_FRACTION_SUM",
    "SMALL_TARGET_TARGET_WIDTH_PX_SUM",
    "SMALL_TARGET_TARGET_HEIGHT_PX_SUM",
    "SMALL_TARGET_STATE_IOU_SUM",
    "SMALL_TARGET_CENTER_OFFSET_NORM_SUM",
    "SMALL_TARGET_SPECIALIST_IOU_SUM",
    "SMALL_TARGET_INSIDE_COUNT",
    "SMALL_TARGET_INSIDE_IOU_SUM",
    "SMALL_TARGET_OUTSIDE_COUNT",
    "SMALL_TARGET_OUTSIDE_IOU_SUM",
    "SMALL_TARGET_SCORE_SUM",
    "SMALL_TARGET_SCORE_COUNT",
    "SMALL_TARGET_PSR_SUM",
    "SMALL_TARGET_PSR_COUNT",
    "SMALL_TARGET_CROP_MISS_COUNT",
    "SMALL_TARGET_IN_WINDOW_LOW_CONFIDENCE_COUNT",
    "SMALL_TARGET_LOCALIZATION_ERROR_COUNT",
    "SMALL_TARGET_SUCCESS_COUNT",
    "SMALL_TARGET_CENTER_ERROR_PX_SUM",
    "SMALL_TARGET_SIZE_ERROR_PX_SUM",
)


def expert_validation_sums(
        expert_boxes,
        ensemble_boxes,
        ground_truth_boxes,
        target_visible,
        challenge_attributes,
        *,
        success_threshold=0.5):
    """Return multi-label challenge-conditioned sums for DDP aggregation."""
    experts = np.asarray(expert_boxes, dtype=np.float64)
    ensemble = np.asarray(ensemble_boxes, dtype=np.float64)
    ground_truth = np.asarray(ground_truth_boxes, dtype=np.float64)
    visible = np.asarray(target_visible, dtype=np.uint8).reshape(-1)
    frame_count = len(visible)
    supervision = expert_supervision_mask(
        challenge_attributes).cpu().numpy()
    expected_expert_shape = (frame_count, len(EXPERT_NAMES), 4)
    if (experts.shape != expected_expert_shape
            or ensemble.shape != (frame_count, 4)
            or ground_truth.shape != (frame_count, 4)
            or supervision.shape != (frame_count, len(EXPERT_NAMES))):
        raise ValueError(
            "Expert validation requires frame-aligned five-expert data")
    if not math.isfinite(success_threshold):
        raise ValueError("success_threshold must be finite")
    values = {}
    for expert_id in range(len(EXPERT_NAMES)):
        prefix = f"EXPERT_{expert_id}"
        values.update({
            f"{prefix}_COUNT": 0.0,
            f"{prefix}_IOU_SUM": 0.0,
            f"{prefix}_SUCCESS_HITS": 0.0,
            f"{prefix}_GENERALIST_IOU_SUM": 0.0,
            f"{prefix}_GENERALIST_SUCCESS_HITS": 0.0,
            f"{prefix}_ENSEMBLE_IOU_SUM": 0.0,
            f"{prefix}_ENSEMBLE_SUCCESS_HITS": 0.0,
        })

    for frame_index in np.flatnonzero(visible):
        generalist_iou = _xywh_iou(
            experts[frame_index, 0], ground_truth[frame_index])
        ensemble_iou = _xywh_iou(
            ensemble[frame_index], ground_truth[frame_index])
        for expert_id in np.flatnonzero(supervision[frame_index]):
            prefix = f"EXPERT_{expert_id}"
            specialist_iou = _xywh_iou(
                experts[frame_index, expert_id], ground_truth[frame_index])
            values[f"{prefix}_COUNT"] += 1.0
            values[f"{prefix}_IOU_SUM"] += specialist_iou
            values[f"{prefix}_SUCCESS_HITS"] += float(
                specialist_iou >= success_threshold)
            values[f"{prefix}_GENERALIST_IOU_SUM"] += generalist_iou
            values[f"{prefix}_GENERALIST_SUCCESS_HITS"] += float(
                generalist_iou >= success_threshold)
            values[f"{prefix}_ENSEMBLE_IOU_SUM"] += ensemble_iou
            values[f"{prefix}_ENSEMBLE_SUCCESS_HITS"] += float(
                ensemble_iou >= success_threshold)
    return values


def small_target_diagnostic_sums(
        search_trace,
        expert_boxes,
        ground_truth_boxes,
        target_visible,
        challenge_attributes,
        *,
        sequence_name="",
        specialist_id=2,
        success_threshold=0.5,
        confidence_threshold=0.25):
    """Attribute precision-expert failures on every small-target frame."""
    trace = list(search_trace or [])
    experts = np.asarray(expert_boxes, dtype=np.float64)
    ground_truth = np.asarray(ground_truth_boxes, dtype=np.float64)
    visible = np.asarray(target_visible, dtype=np.uint8).reshape(-1)
    frame_count = len(visible)
    supervision = expert_supervision_mask(
        challenge_attributes).cpu().numpy()
    if (len(trace) != frame_count
            or experts.shape != (frame_count, len(EXPERT_NAMES), 4)
            or ground_truth.shape != (frame_count, 4)
            or supervision.shape != (frame_count, len(EXPERT_NAMES))):
        raise ValueError(
            "Small-target diagnostics require frame-aligned sequence data")
    if specialist_id < 0 or specialist_id >= len(EXPERT_NAMES):
        raise ValueError("specialist_id must identify one expert")
    if not all(math.isfinite(value) for value in (
            success_threshold, confidence_threshold)):
        raise ValueError("diagnostic thresholds must be finite")

    values = {key: 0.0 for key in SMALL_TARGET_DIAGNOSTIC_KEYS}
    records = []
    for frame_index in range(frame_count):
        frame_trace = trace[frame_index]
        if (not isinstance(frame_trace, dict)
                or int(frame_trace.get("frame_id", -1)) != frame_index):
            raise ValueError(
                "Small-target search trace frame_id must match frame index")
        if (not visible[frame_index]
                or not supervision[frame_index, specialist_id]
                or bool(frame_trace.get("is_initial", False))):
            continue

        crop = np.asarray(
            frame_trace.get("crop_bounds_xyxy", []), dtype=np.float64)
        state = np.asarray(
            frame_trace.get("search_state", []), dtype=np.float64)
        resize_factor = float(frame_trace.get("resize_factor", float("nan")))
        if (crop.shape != (4,) or state.shape != (4,)
                or not np.isfinite(crop).all()
                or not np.isfinite(state).all()
                or crop[2] <= crop[0] or crop[3] <= crop[1]
                or not math.isfinite(resize_factor)
                or resize_factor <= 0):
            raise ValueError(
                "Small-target search traces require valid image-space geometry")

        gt = ground_truth[frame_index]
        specialist = experts[frame_index, specialist_id]
        gt_max = gt[:2] + gt[2:]
        gt_center = gt[:2] + 0.5 * gt[2:]
        crop_center = 0.5 * (crop[:2] + crop[2:])
        crop_size = crop[2:] - crop[:2]
        center_in_crop = bool(np.logical_and(
            gt_center >= crop[:2], gt_center <= crop[2:]).all())
        intersection_size = np.maximum(
            np.minimum(gt_max, crop[2:]) - np.maximum(gt[:2], crop[:2]),
            0.0,
        )
        gt_area = float(np.prod(gt[2:]))
        visible_fraction = (
            float(np.prod(intersection_size)) / gt_area
            if gt_area > 0 else 0.0)
        full_box_in_crop = visible_fraction >= 1.0 - 1e-9
        target_width_px = float(gt[2] * resize_factor)
        target_height_px = float(gt[3] * resize_factor)
        state_iou = _xywh_iou(state, gt)
        specialist_iou = _xywh_iou(specialist, gt)
        crop_half_diagonal = max(
            0.5 * float(np.linalg.norm(crop_size)), 1e-12)
        center_offset_norm = float(
            np.linalg.norm(gt_center - crop_center) / crop_half_diagonal)
        specialist_center = specialist[:2] + 0.5 * specialist[2:]
        center_error_px = float(np.linalg.norm(
            specialist_center - gt_center))
        size_error_px = float(np.abs(specialist[2:] - gt[2:]).mean())

        peaks = frame_trace.get("expert_peaks", [])
        psr_values = frame_trace.get("expert_psr", [])
        peak = float(peaks[specialist_id]) if len(peaks) > specialist_id else float("nan")
        psr = float(psr_values[specialist_id]) if len(psr_values) > specialist_id else float("nan")
        if not center_in_crop:
            failure_bucket = "crop_miss"
            values["SMALL_TARGET_CROP_MISS_COUNT"] += 1.0
        elif specialist_iou >= success_threshold:
            failure_bucket = "success"
            values["SMALL_TARGET_SUCCESS_COUNT"] += 1.0
        elif math.isfinite(peak) and peak < confidence_threshold:
            failure_bucket = "in_window_low_confidence"
            values["SMALL_TARGET_IN_WINDOW_LOW_CONFIDENCE_COUNT"] += 1.0
        else:
            failure_bucket = "localization_error"
            values["SMALL_TARGET_LOCALIZATION_ERROR_COUNT"] += 1.0

        values["SMALL_TARGET_COUNT"] += 1.0
        values["SMALL_TARGET_CENTER_IN_CROP_HITS"] += float(center_in_crop)
        values["SMALL_TARGET_FULL_BOX_IN_CROP_HITS"] += float(full_box_in_crop)
        values["SMALL_TARGET_VISIBLE_FRACTION_SUM"] += visible_fraction
        values["SMALL_TARGET_TARGET_WIDTH_PX_SUM"] += target_width_px
        values["SMALL_TARGET_TARGET_HEIGHT_PX_SUM"] += target_height_px
        values["SMALL_TARGET_STATE_IOU_SUM"] += state_iou
        values["SMALL_TARGET_CENTER_OFFSET_NORM_SUM"] += center_offset_norm
        values["SMALL_TARGET_SPECIALIST_IOU_SUM"] += specialist_iou
        values["SMALL_TARGET_CENTER_ERROR_PX_SUM"] += center_error_px
        values["SMALL_TARGET_SIZE_ERROR_PX_SUM"] += size_error_px
        if center_in_crop:
            values["SMALL_TARGET_INSIDE_COUNT"] += 1.0
            values["SMALL_TARGET_INSIDE_IOU_SUM"] += specialist_iou
        else:
            values["SMALL_TARGET_OUTSIDE_COUNT"] += 1.0
            values["SMALL_TARGET_OUTSIDE_IOU_SUM"] += specialist_iou
        if math.isfinite(peak):
            values["SMALL_TARGET_SCORE_SUM"] += peak
            values["SMALL_TARGET_SCORE_COUNT"] += 1.0
        if math.isfinite(psr):
            values["SMALL_TARGET_PSR_SUM"] += psr
            values["SMALL_TARGET_PSR_COUNT"] += 1.0

        records.append({
            "sequence": str(sequence_name),
            "frame_index": int(frame_index),
            "expert_id": int(specialist_id),
            "ground_truth": gt.tolist(),
            "search_state": state.tolist(),
            "crop_bounds_xyxy": crop.tolist(),
            "resize_factor": resize_factor,
            "center_in_crop": center_in_crop,
            "full_box_in_crop": bool(full_box_in_crop),
            "visible_fraction": visible_fraction,
            "target_width_px": target_width_px,
            "target_height_px": target_height_px,
            "state_iou": state_iou,
            "center_offset_norm": center_offset_norm,
            "specialist_box": specialist.tolist(),
            "expert_boxes": experts[frame_index].tolist(),
            "specialist_iou": specialist_iou,
            "specialist_score": peak if math.isfinite(peak) else None,
            "specialist_psr": psr if math.isfinite(psr) else None,
            "previous_action": str(frame_trace.get("previous_action", "")),
            "action": str(frame_trace.get("action", "")),
            "presence_score": _optional_finite_float(
                frame_trace.get("presence_score")),
            "controller_output_score": _optional_finite_float(
                frame_trace.get("controller_output_score")),
            "theta_present": _optional_finite_float(
                frame_trace.get("theta_present")),
            "theta_recover": _optional_finite_float(
                frame_trace.get("theta_recover")),
            "controller_weak_streak": int(frame_trace.get(
                "controller_weak_streak", 0)),
            "controller_verify_streak": int(frame_trace.get(
                "controller_verify_streak", 0)),
            "controller_stable_visible": int(frame_trace.get(
                "controller_stable_visible", 0)),
            "recovery_attempted": bool(frame_trace.get(
                "recovery_attempted", False)),
            "recovery_confirmed": bool(frame_trace.get(
                "recovery_confirmed", False)),
            "recovery_max_identity": _optional_finite_float(
                frame_trace.get("recovery_max_identity")),
            "recovery_max_localization": _optional_finite_float(
                frame_trace.get("recovery_max_localization")),
            "recovery_accepted_count": int(frame_trace.get(
                "recovery_accepted_count", 0)),
            "redetect_confidence": _optional_finite_float(
                frame_trace.get("redetect_confidence")),
            "ensemble_score": (
                float(frame_trace["ensemble_score"])
                if math.isfinite(float(frame_trace.get(
                    "ensemble_score", float("nan")))) else None),
            "expert_peaks": [
                float(value) if math.isfinite(float(value)) else None
                for value in peaks
            ],
            "expert_psr": [
                float(value) if math.isfinite(float(value)) else None
                for value in psr_values
            ],
            "retained_expert_ids": [
                int(value) for value in frame_trace.get(
                    "retained_expert_ids", [])],
            "ensemble_weights": [
                float(value) for value in frame_trace.get(
                    "ensemble_weights", [])],
            "center_error_px": center_error_px,
            "size_error_px": size_error_px,
            "failure_bucket": failure_bucket,
        })
    return values, records


def _optional_finite_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def specialist_passes(
        count,
        specialist,
        generalist,
        *,
        min_count=100,
        min_delta=0.02):
    values = (count, specialist, generalist, min_count, min_delta)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    return count >= min_count and specialist - generalist >= min_delta


def refine_checkpoint_is_accepted(
        stage1_generalist,
        refine_generalist,
        specialist_regressions,
        *,
        max_generalist_drop=0.005):
    if not all(math.isfinite(float(value)) for value in (
            stage1_generalist, refine_generalist, max_generalist_drop)):
        return False
    return (
        refine_generalist >= stage1_generalist - max_generalist_drop
        and not any(bool(value) for value in specialist_regressions)
    )


def _xywh_iou(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (first.shape != (4,) or second.shape != (4,)
            or not np.isfinite(first).all()
            or not np.isfinite(second).all()
            or first[2] <= 0 or first[3] <= 0
            or second[2] <= 0 or second[3] <= 0):
        return 0.0
    first_max = first[:2] + first[2:]
    second_max = second[:2] + second[2:]
    intersection_size = np.maximum(
        np.minimum(first_max, second_max) - np.maximum(first[:2], second[:2]),
        0.0,
    )
    intersection = float(np.prod(intersection_size))
    union = float(np.prod(first[2:]) + np.prod(second[2:]) - intersection)
    return intersection / union if union > 0 else 0.0


def recovery_diagnostic_sums(
        predicted_boxes,
        predicted_absent,
        frame_times,
        ground_truth_boxes,
        target_visible,
        recovery_trace,
        *,
        identity_threshold=0.75):
    """Score causal tracker traces against GT only after a sequence finishes."""
    boxes = np.asarray(predicted_boxes, dtype=np.float64)
    ground_truth = np.asarray(ground_truth_boxes, dtype=np.float64)
    visible = np.asarray(target_visible, dtype=np.uint8).reshape(-1)
    absent = np.asarray(predicted_absent, dtype=np.bool_).reshape(-1)
    times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    frame_count = len(visible)
    if (boxes.shape != (frame_count, 4)
            or ground_truth.shape != (frame_count, 4)
            or len(absent) != frame_count
            or len(times) != frame_count):
        raise ValueError("Recovery diagnostics require frame-aligned sequence data")
    if not math.isfinite(identity_threshold):
        raise ValueError("identity_threshold must be finite")

    trace = list(recovery_trace or [])
    if len(trace) < frame_count:
        trace.extend({} for _ in range(frame_count - len(trace)))
    elif len(trace) > frame_count:
        trace = trace[:frame_count]

    values = {key: 0.0 for key in RECOVERY_DIAGNOSTIC_KEYS}
    reappearance_frames = [
        index for index in range(1, frame_count)
        if visible[index] and not visible[index - 1]
    ]
    values["EVENT_REAPPEARANCE_COUNT"] = float(len(reappearance_frames))
    values["RECOVERY_EVENT_COUNT"] = float(len(reappearance_frames))
    excluded_retention = set()

    for start in reappearance_frames:
        gt_box = ground_truth[start]
        gt_min = gt_box[:2]
        gt_max = gt_box[:2] + gt_box[2:]
        centers = np.asarray(
            trace[start].get("event_centers", []), dtype=np.float64)
        if centers.size:
            centers = centers.reshape(-1, 2)
            inside = np.logical_and(
                centers >= gt_min,
                centers <= gt_max,
            ).all(axis=1)
            for top_k in (1, 3, 5):
                if inside[:top_k].any():
                    values[f"EVENT_RECALL_AT_{top_k}_HITS"] += 1.0

        stop = next((
            index for index in range(start + 1, frame_count)
            if not visible[index]
        ), frame_count)
        for index in range(start, stop):
            if not absent[index] and _xywh_iou(
                    boxes[index], ground_truth[index]) >= 0.5:
                values["RECOVERY_LATENCY_SUM"] += float(index - start)
                values["RECOVERY_SUCCESS_COUNT"] += 1.0
                break

        for horizon in (1, 3, 5):
            index = start + horizon - 1
            if index < stop:
                values[f"REAPPEAR_IOU_AT_{horizon}_SUM"] += _xywh_iou(
                    boxes[index], ground_truth[index])
                values[f"REAPPEAR_IOU_AT_{horizon}_COUNT"] += 1.0
        excluded_retention.update(range(start, min(start + 5, stop)))

    for index in range(frame_count):
        identity_scores = trace[index].get("identity_scores", [])
        if not visible[index]:
            values["RGB_ABSENT_CANDIDATES"] += float(len(identity_scores))
            values["RGB_FALSE_ACCEPTS"] += float(sum(
                float(score) >= identity_threshold
                for score in identity_scores
            ))
        if index > 0 and visible[index] and index not in excluded_retention:
            values["VISIBLE_RETENTION_IOU_SUM"] += _xywh_iou(
                boxes[index], ground_truth[index])
            values["VISIBLE_RETENTION_COUNT"] += 1.0

        if index == 0:
            continue
        elapsed = float(times[index])
        if not math.isfinite(elapsed) or elapsed < 0:
            continue
        if trace[index].get("action") in ("absent", "verify"):
            values["RECOVERY_TIME_SUM"] += elapsed
            values["RECOVERY_FRAME_COUNT"] += 1.0
        else:
            values["TRACK_TIME_SUM"] += elapsed
            values["TRACK_FRAME_COUNT"] += 1.0
    return values


def sequence_validation_due(epoch, schedule):
    if not schedule:
        return False
    for item in schedule:
        if isinstance(item, str):
            parts = [int(part.strip()) for part in item.split(":")]
        else:
            parts = [int(part) for part in item]
        if len(parts) != 3:
            raise ValueError(
                "Each sequence validation schedule entry must be "
                "(start, end, interval)")
        start, end, interval = parts
        if epoch >= start and (end < 0 or epoch <= end):
            return (epoch - start) % max(interval, 1) == 0
    return False


def _sequence_validation_params(
        cfg, forced_expert_id=None, policy_mode=None):
    params = TrackerParams()
    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.debug = False
    params.save_all_boxes = False
    params.forced_expert_id = forced_expert_id
    params.policy_mode = policy_mode
    return params


def run_felt_sequence_validation(
        network,
        cfg,
        sequences=None,
        felt_val_root=None,
        rank=0,
        world_size=1,
        evaluator=None,
        tracker_factory=None,
        challenge_manifest=None,
        forced_expert_id=None,
        policy_mode=None,
        log_prefix="SequenceVal"):
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("rank must identify one process in world_size")
    if sequences is None:
        from lib.test.evaluation.feltdataset import FELTDataset
        sequences = FELTDataset(
            "val", base_path=felt_val_root).get_sequence_list()
    from lib.test.evaluation.tracker import Tracker, is_valid_initial_bbox
    if evaluator is None:
        evaluator = Tracker("pet_track", "sequence_val", "felt")
    params = _sequence_validation_params(
        cfg,
        forced_expert_id=forced_expert_id,
        policy_mode=policy_mode,
    )
    if tracker_factory is None:
        from lib.test.tracker.pet_track import PETTrack
        tracker_factory = lambda injected_network, tracker_params: PETTrack(
            tracker_params,
            dataset_name="felt",
            network=injected_network,
        )
    if challenge_manifest is None:
        challenge_cfg = getattr(
            getattr(cfg, "DATA", None), "CHALLENGE_SAMPLING", None)
        manifest_path = str(getattr(
            challenge_cfg, "VAL_MANIFEST", "") or "")
        if manifest_path:
            from lib.train.data.challenge_manifest import load_manifest
            challenge_manifest = load_manifest(manifest_path)
    manifest_sequences = (
        challenge_manifest.get("sequences", {})
        if challenge_manifest is not None else None
    )

    was_training = network.training
    network.eval()
    proxy_sum = 0.0
    absent_balanced_accuracy_sum = 0.0
    reappearance_proxy_sum = 0.0
    reappearance_sequence_count = 0
    sequence_count = 0
    recovery_diagnostics = {
        key: 0.0 for key in RECOVERY_DIAGNOSTIC_KEYS}
    expert_diagnostics = {
        key: 0.0 for key in EXPERT_DIAGNOSTIC_KEYS}
    small_target_diagnostics = {
        key: 0.0 for key in SMALL_TARGET_DIAGNOSTIC_KEYS}
    small_target_records = []
    runnable_sequences = []
    for sequence in sequences:
        init_info = sequence.init_info()
        if not is_valid_initial_bbox(init_info.get("init_bbox")):
            if rank == 0:
                print(
                    "{} skipped invalid initial bbox: {}".format(
                        log_prefix,
                        sequence.name),
                    flush=True,
                )
            continue
        challenge_record = None
        if manifest_sequences is not None:
            challenge_record = manifest_sequences.get(sequence.name)
            if not isinstance(challenge_record, dict):
                if rank == 0:
                    print(
                        "{} skipped missing challenge manifest: {}".format(
                            log_prefix,
                            sequence.name,
                        ),
                        flush=True,
                    )
                continue
        runnable_sequences.append((sequence, init_info, challenge_record))
    sequence_shard = runnable_sequences[rank::world_size]
    try:
        with torch.inference_mode():
            for shard_index, (sequence, init_info, challenge_record) in enumerate(
                    sequence_shard, start=1):
                tracker = tracker_factory(network, params)
                output = evaluator._track_sequence(
                    tracker, sequence, init_info)
                sequence_proxy = felt_success_auc_proxy(
                    output["target_bbox"],
                    sequence.ground_truth_rect,
                    sequence.target_visible,
                )
                absent_score = felt_absent_balanced_accuracy(
                    output["absent"], sequence.target_visible)
                reappearance_score = felt_reappearance_auc_proxy(
                    output["target_bbox"],
                    sequence.ground_truth_rect,
                    sequence.target_visible,
                )
                trace_getter = getattr(
                    tracker, "get_recovery_diagnostics", None)
                trace = trace_getter() if trace_getter is not None else []
                sequence_diagnostics = recovery_diagnostic_sums(
                    output["target_bbox"],
                    output["absent"],
                    output.get("time", [0.0] * len(sequence.target_visible)),
                    sequence.ground_truth_rect,
                    sequence.target_visible,
                    trace,
                )
                for key, value in sequence_diagnostics.items():
                    recovery_diagnostics[key] += value
                if manifest_sequences is not None:
                    trace_getter = getattr(
                        tracker, "get_expert_diagnostics", None)
                    if trace_getter is None:
                        raise RuntimeError(
                            "SequenceVal tracker does not expose expert diagnostics")
                    sequence_expert_diagnostics = expert_validation_sums(
                        expert_boxes=trace_getter(),
                        ensemble_boxes=output["target_bbox"],
                        ground_truth_boxes=sequence.ground_truth_rect,
                        target_visible=sequence.target_visible,
                        challenge_attributes=challenge_record["attributes"],
                    )
                    for key, value in sequence_expert_diagnostics.items():
                        expert_diagnostics[key] += value
                    if forced_expert_id in (None, 2):
                        search_trace_getter = getattr(
                            tracker, "get_search_diagnostics", None)
                        if search_trace_getter is None:
                            raise RuntimeError(
                                "SequenceVal tracker does not expose search "
                                "diagnostics")
                        sequence_small_diagnostics, sequence_small_records = (
                            small_target_diagnostic_sums(
                                search_trace=search_trace_getter(),
                                expert_boxes=trace_getter(),
                                ground_truth_boxes=sequence.ground_truth_rect,
                                target_visible=sequence.target_visible,
                                challenge_attributes=challenge_record["attributes"],
                                sequence_name=sequence.name,
                            )
                        )
                        for key, value in sequence_small_diagnostics.items():
                            small_target_diagnostics[key] += value
                        small_target_records.extend(sequence_small_records)
                proxy_sum += sequence_proxy
                absent_balanced_accuracy_sum += absent_score
                if math.isfinite(reappearance_score):
                    reappearance_proxy_sum += reappearance_score
                    reappearance_sequence_count += 1
                sequence_count += 1
                print(
                    "{} rank {}: {}/{} {} FELT_SR_PROXY={:.6f} "
                    "FELT_ABSENT_BAL_ACC={:.6f} "
                    "FELT_REAPPEARANCE_PROXY={}".format(
                        log_prefix,
                        rank,
                        shard_index,
                        len(sequence_shard),
                        sequence.name,
                        sequence_proxy,
                        absent_score,
                        (f"{reappearance_score:.6f}"
                         if math.isfinite(reappearance_score) else "n/a"),
                    ),
                    flush=True,
                )
    finally:
        network.train(was_training)

    metrics = {
        "FELT_SR_PROXY_SUM": proxy_sum,
        "FELT_ABSENT_BAL_ACC_SUM": absent_balanced_accuracy_sum,
        "FELT_REAPPEARANCE_PROXY_SUM": reappearance_proxy_sum,
        "REAPPEARANCE_SEQUENCE_COUNT": reappearance_sequence_count,
        "SEQUENCE_COUNT": sequence_count,
    }
    metrics.update(recovery_diagnostics)
    if manifest_sequences is not None:
        metrics.update(expert_diagnostics)
        metrics.update(small_target_diagnostics)
        metrics["SMALL_TARGET_RECORDS"] = small_target_records
    return metrics
