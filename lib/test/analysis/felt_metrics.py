import numpy as np


FELT_OVERLAP_THRESHOLDS = np.arange(0.0, 1.01, 0.05, dtype=np.float64)


def _as_boxes(boxes, name):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape [frames, 4]")
    return boxes


def _official_felt_overlaps(predictions, ground_truth, presence):
    predictions = _as_boxes(predictions, "predictions").copy()
    ground_truth = _as_boxes(ground_truth, "ground_truth").copy()
    presence = np.asarray(presence).reshape(-1)
    frame_count = ground_truth.shape[0]
    if predictions.shape[0] != frame_count or presence.shape[0] != frame_count:
        raise ValueError("predictions, ground_truth, and presence must have equal frame counts")
    if frame_count == 0:
        raise ValueError("FELT sequences must contain at least one frame")
    if not np.isin(presence, (0, 1)).all():
        raise ValueError("FELT presence flags must use 1=present and 0=absent")

    for frame_index in range(1, frame_count):
        prediction = predictions[frame_index]
        annotation = ground_truth[frame_index]
        invalid = (
            np.isnan(prediction).any()
            or not np.isreal(prediction).all()
            or prediction[2] <= 0
            or prediction[3] <= 0
        )
        if invalid and not np.isnan(annotation).any():
            predictions[frame_index] = predictions[frame_index - 1]

    predictions[0] = ground_truth[0]
    present = presence.astype(bool)
    present_count = int(np.count_nonzero(present))
    if present_count == 0:
        raise ValueError("FELT sequences must contain a target-present frame")
    predictions = predictions[present]
    ground_truth = ground_truth[present]

    valid = (ground_truth > 0).sum(axis=1) == 4
    overlaps = np.full(ground_truth.shape[0], -1.0, dtype=np.float64)
    if valid.any():
        pred = predictions[valid]
        anno = ground_truth[valid]
        pred_right_bottom = pred[:, :2] + pred[:, 2:] - 1.0
        anno_right_bottom = anno[:, :2] + anno[:, 2:] - 1.0
        intersection_size = np.maximum(
            0.0,
            np.minimum(pred_right_bottom, anno_right_bottom)
            - np.maximum(pred[:, :2], anno[:, :2])
            + 1.0,
        )
        intersection = intersection_size.prod(axis=1)
        union = pred[:, 2:].prod(axis=1) + anno[:, 2:].prod(axis=1) - intersection
        overlaps[valid] = intersection / union
    return overlaps, present_count


def felt_success_curve_proxy(predictions, ground_truth, presence):
    overlaps, present_count = _official_felt_overlaps(
        predictions, ground_truth, presence)
    return np.asarray([
        np.count_nonzero(overlaps > threshold) / present_count
        for threshold in FELT_OVERLAP_THRESHOLDS
    ], dtype=np.float64)


def felt_success_auc_proxy(predictions, ground_truth, presence):
    return float(
        felt_success_curve_proxy(predictions, ground_truth, presence).mean())


def felt_absent_balanced_accuracy(predicted_absent, presence):
    predicted_absent = np.asarray(predicted_absent).reshape(-1)
    presence = np.asarray(presence).reshape(-1)
    if predicted_absent.shape != presence.shape:
        raise ValueError("predicted_absent and presence must have equal frame counts")
    if predicted_absent.size == 0:
        raise ValueError("FELT sequences must contain at least one frame")
    if not np.isin(predicted_absent, (0, 1)).all():
        raise ValueError("predicted absent flags must use boolean or 0/1 values")
    if not np.isin(presence, (0, 1)).all():
        raise ValueError("FELT presence flags must use 1=present and 0=absent")

    predicted_absent = predicted_absent.astype(bool)
    present = presence.astype(bool)
    recalls = []
    if present.any():
        recalls.append(np.mean(~predicted_absent[present]))
    absent = ~present
    if absent.any():
        recalls.append(np.mean(predicted_absent[absent]))
    return float(np.mean(recalls))


def felt_reappearance_auc_proxy(predictions, ground_truth, presence, horizon=5):
    presence = np.asarray(presence).reshape(-1)
    if int(horizon) < 1:
        raise ValueError("reappearance horizon must be positive")
    if not np.isin(presence, (0, 1)).all():
        raise ValueError("FELT presence flags must use 1=present and 0=absent")

    present = presence.astype(bool)
    reappearance = np.zeros_like(present)
    for start in np.flatnonzero(present[1:] & ~present[:-1]) + 1:
        for frame_index in range(start, min(start + int(horizon), present.size)):
            if not present[frame_index]:
                break
            reappearance[frame_index] = True
    if not reappearance.any():
        return float("nan")
    return felt_success_auc_proxy(
        predictions, ground_truth, reappearance.astype(np.uint8))
