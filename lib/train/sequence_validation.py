import math

import numpy as np
import torch

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
    "small_target_st",
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


def expert_validation_sums(
        expert_boxes,
        ensemble_boxes,
        ground_truth_boxes,
        target_visible,
        owner_ids,
        *,
        success_threshold=0.5):
    """Return raw owner-conditioned sums for later DDP aggregation."""
    experts = np.asarray(expert_boxes, dtype=np.float64)
    ensemble = np.asarray(ensemble_boxes, dtype=np.float64)
    ground_truth = np.asarray(ground_truth_boxes, dtype=np.float64)
    visible = np.asarray(target_visible, dtype=np.uint8).reshape(-1)
    owners = np.asarray(owner_ids, dtype=np.int64).reshape(-1)
    frame_count = len(visible)
    expected_expert_shape = (frame_count, len(EXPERT_NAMES), 4)
    if (experts.shape != expected_expert_shape
            or ensemble.shape != (frame_count, 4)
            or ground_truth.shape != (frame_count, 4)
            or len(owners) != frame_count):
        raise ValueError(
            "Expert validation requires frame-aligned five-expert data")
    if not math.isfinite(success_threshold):
        raise ValueError("success_threshold must be finite")
    if ((owners < 0) | (owners >= len(EXPERT_NAMES))).any():
        raise ValueError("owner_ids must be in the five-expert range")

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
        expert_id = int(owners[frame_index])
        prefix = f"EXPERT_{expert_id}"
        specialist_iou = _xywh_iou(
            experts[frame_index, expert_id], ground_truth[frame_index])
        generalist_iou = _xywh_iou(
            experts[frame_index, 0], ground_truth[frame_index])
        ensemble_iou = _xywh_iou(
            ensemble[frame_index], ground_truth[frame_index])
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


def _sequence_validation_params(cfg):
    params = TrackerParams()
    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.debug = False
    params.save_all_boxes = False
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
        ownership_manifest=None):
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("rank must identify one process in world_size")
    if sequences is None:
        from lib.test.evaluation.feltdataset import FELTDataset
        sequences = FELTDataset(
            "val", base_path=felt_val_root).get_sequence_list()
    from lib.test.evaluation.tracker import Tracker, is_valid_initial_bbox
    if evaluator is None:
        evaluator = Tracker("pet_track", "sequence_val", "felt")
    params = _sequence_validation_params(cfg)
    if tracker_factory is None:
        from lib.test.tracker.pet_track import PETTrack
        tracker_factory = lambda injected_network, tracker_params: PETTrack(
            tracker_params,
            dataset_name="felt",
            network=injected_network,
        )
    if ownership_manifest is None:
        challenge_cfg = getattr(
            getattr(cfg, "DATA", None), "CHALLENGE_SAMPLING", None)
        manifest_path = str(getattr(
            challenge_cfg, "VAL_MANIFEST", "") or "")
        if manifest_path:
            from lib.train.data.expert_ownership import load_manifest
            ownership_manifest = load_manifest(manifest_path)
    manifest_sequences = (
        ownership_manifest.get("sequences", {})
        if ownership_manifest is not None else None
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
    runnable_sequences = []
    for sequence in sequences:
        init_info = sequence.init_info()
        if not is_valid_initial_bbox(init_info.get("init_bbox")):
            if rank == 0:
                print(
                    "SequenceVal skipped invalid initial bbox: {}".format(
                        sequence.name),
                    flush=True,
                )
            continue
        runnable_sequences.append((sequence, init_info))
    sequence_shard = runnable_sequences[rank::world_size]
    try:
        with torch.inference_mode():
            for shard_index, (sequence, init_info) in enumerate(
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
                    record = manifest_sequences.get(sequence.name)
                    if not isinstance(record, dict):
                        raise RuntimeError(
                            "SequenceVal ownership manifest is missing "
                            f"sequence {sequence.name}")
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
                        owner_ids=record.get("owners", []),
                    )
                    for key, value in sequence_expert_diagnostics.items():
                        expert_diagnostics[key] += value
                proxy_sum += sequence_proxy
                absent_balanced_accuracy_sum += absent_score
                if math.isfinite(reappearance_score):
                    reappearance_proxy_sum += reappearance_score
                    reappearance_sequence_count += 1
                sequence_count += 1
                print(
                    "SequenceVal rank {}: {}/{} {} FELT_SR_PROXY={:.6f} "
                    "FELT_ABSENT_BAL_ACC={:.6f} "
                    "FELT_REAPPEARANCE_PROXY={}".format(
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
    return metrics
