import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from lib.train.data.felt_challenges import (
    CHALLENGE_NAMES,
    DEFAULT_AMBIGUITY_THRESHOLD,
    DEFAULT_SMALL_MAX_ASPECT_RATIO,
    DEFAULT_MOTION_NORM,
    DEFAULT_RECOVERY_WINDOW,
    DEFAULT_SMALL_AREA_RATIO,
    classify_felt_challenges,
)

DEFAULT_LOW_LIGHT_CONTEXT_MEDIAN = 8.2656
DEFAULT_LOW_LIGHT_SEARCH_FACTOR = 4.0


def _grayscale_frames(frames, name):
    if isinstance(frames, (list, tuple)):
        frames = torch.stack([
            torch.as_tensor(frame, dtype=torch.float32) for frame in frames
        ])
    else:
        frames = torch.as_tensor(frames, dtype=torch.float32)
    if frames.ndim == 3:
        frames = frames.unsqueeze(-1)
    if frames.ndim != 4:
        raise ValueError(f"{name} must have shape (N,H,W,C) or (N,C,H,W)")
    if frames.shape[-1] in (1, 2, 3, 4):
        frames = frames.permute(0, 3, 1, 2)
    elif frames.shape[1] not in (1, 2, 3, 4):
        raise ValueError(f"{name} has no recognizable channel dimension")
    return frames.mean(dim=1)


def _box_slice(box, height, width):
    x, y, w, h = torch.as_tensor(box, dtype=torch.float32).tolist()
    x1 = max(0, min(width, int(round(x))))
    y1 = max(0, min(height, int(round(y))))
    x2 = max(x1, min(width, int(round(x + w))))
    y2 = max(y1, min(height, int(round(y + h))))
    return None if x2 <= x1 or y2 <= y1 else (slice(y1, y2), slice(x1, x2))


def _search_context_slice(box, height, width, search_factor):
    if float(search_factor) <= 0:
        raise ValueError("low-light search factor must be positive")
    x, y, w, h = torch.as_tensor(box, dtype=torch.float32).tolist()
    crop_size = math.ceil(math.sqrt(w * h) * float(search_factor))
    if crop_size < 1:
        return None
    x1 = round(x + 0.5 * w - 0.5 * crop_size)
    y1 = round(y + 0.5 * h - 0.5 * crop_size)
    x2 = min(width, x1 + crop_size)
    y2 = min(height, y1 + crop_size)
    x1 = max(0, x1)
    y1 = max(0, y1)
    return None if x2 <= x1 or y2 <= y1 else (slice(y1, y2), slice(x1, x2))


def _aps_ambiguity(template_frame, search_frame, template_box, search_box):
    target_slice = _box_slice(
        template_box, template_frame.shape[-2], template_frame.shape[-1])
    if target_slice is None:
        return template_frame.new_zeros(())
    template = template_frame[target_slice]
    template = F.interpolate(
        template[None, None], size=(8, 8), mode="bilinear",
        align_corners=False)[0, 0]
    search = F.interpolate(
        search_frame[None, None], size=(64, 64), mode="bilinear",
        align_corners=False)
    template_vector = template.flatten()
    template_vector = template_vector - template_vector.mean()
    template_norm = torch.linalg.vector_norm(template_vector)
    if template_norm <= 1e-6:
        return template_frame.new_zeros(())

    patches = F.unfold(search, kernel_size=(8, 8))[0]
    patches = patches - patches.mean(dim=0, keepdim=True)
    patch_norms = torch.linalg.vector_norm(patches, dim=0).clamp_min(1e-6)
    similarity = (template_vector[:, None] * patches).sum(dim=0)
    similarity = (similarity / (template_norm * patch_norms)).clamp_min(0.0)
    similarity = similarity.reshape(57, 57)

    x, y, width, height = torch.as_tensor(
        search_box, dtype=torch.float32).tolist()
    if width <= 0 or height <= 0:
        return template_frame.new_zeros(())
    scale_x = 64.0 / search_frame.shape[-1]
    scale_y = 64.0 / search_frame.shape[-2]
    x1 = max(0, math.floor(x * scale_x) - 7)
    y1 = max(0, math.floor(y * scale_y) - 7)
    x2 = min(similarity.shape[1], math.ceil((x + width) * scale_x))
    y2 = min(similarity.shape[0], math.ceil((y + height) * scale_y))
    if x2 <= x1 or y2 <= y1:
        return template_frame.new_zeros(())

    target_value = similarity[y1:y2, x1:x2].max()
    suppressed = similarity.clone()
    suppressed[y1:y2, x1:x2] = 0.0
    distractor_value = suppressed.max()
    return distractor_value / target_value.clamp_min(1e-6)


def _event_motion_pair(current, previous, box):
    current = current.abs()
    current_slice = _box_slice(
        box, current.shape[-2], current.shape[-1])
    if current_slice is None:
        return current.new_zeros(())
    foreground = current[current_slice].mean()
    background = current.mean()
    if previous is not None:
        delta = (current - previous).abs()
        foreground = foreground + delta[current_slice].mean()
        background = background + delta.mean()
    return foreground / background.clamp_min(1e-6)


def _event_motion_score(event_frames, boxes, index):
    previous = event_frames[index - 1] if index > 0 else None
    return _event_motion_pair(event_frames[index], previous, boxes[index])


def compute_observation_scores(aps, dvs, boxes, presence):
    aps = _grayscale_frames(aps, "aps")
    dvs = _grayscale_frames(dvs, "dvs")
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    presence = torch.as_tensor(presence, dtype=torch.bool).reshape(-1)
    frame_count = boxes.shape[0]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N,4)")
    if aps.shape[0] != frame_count or dvs.shape[0] != frame_count:
        raise ValueError("APS, DVS, and boxes must have equal frame counts")
    if presence.numel() != frame_count:
        raise ValueError("presence must provide one value per frame")
    if aps.shape[-2:] != dvs.shape[-2:]:
        raise ValueError("APS and DVS frames must be spatially aligned")

    ambiguity = torch.zeros(frame_count, dtype=torch.float32)
    event_motion = torch.zeros_like(ambiguity)
    template_index = None
    for index in range(frame_count):
        valid = bool(
            presence[index]
            and boxes[index, 2] > 0
            and boxes[index, 3] > 0
        )
        if not valid:
            continue
        if template_index is None:
            template_index = index
        ambiguity[index] = _aps_ambiguity(
            aps[template_index], aps[index], boxes[template_index], boxes[index])
        event_motion[index] = _event_motion_score(dvs, boxes, index)
    return {"ambiguity": ambiguity, "event_motion": event_motion}


def build_sequence_record(dataset, seq_id, thresholds=None):
    thresholds = {
        "small_area_ratio": DEFAULT_SMALL_AREA_RATIO,
        "max_small_aspect_ratio": DEFAULT_SMALL_MAX_ASPECT_RATIO,
        "motion_norm": DEFAULT_MOTION_NORM,
        "ambiguity_threshold": DEFAULT_AMBIGUITY_THRESHOLD,
        "recovery_window": DEFAULT_RECOVERY_WINDOW,
        "low_light_context_median": DEFAULT_LOW_LIGHT_CONTEXT_MEDIAN,
        "low_light_search_factor": DEFAULT_LOW_LIGHT_SEARCH_FACTOR,
        **(thresholds or {}),
    }
    info = dataset.get_sequence_info(seq_id)
    boxes = torch.as_tensor(info["bbox"], dtype=torch.float32)
    presence = torch.as_tensor(info["absent"], dtype=torch.bool).reshape(-1)
    if boxes.shape != (presence.numel(), 4):
        raise ValueError("sequence boxes and presence must have equal frame counts")

    ambiguity = torch.zeros(presence.numel(), dtype=torch.float32)
    event_motion = torch.zeros_like(ambiguity)
    low_light = torch.zeros_like(presence)
    template_frame = None
    template_box = None
    previous_event = None
    image_size = None
    for frame_id in range(presence.numel()):
        aps_frames, dvs_frames, _, _ = dataset.get_frames(
            seq_id, [frame_id], info)
        current_aps = _grayscale_frames(aps_frames, "aps")[0]
        current_event = _grayscale_frames(dvs_frames, "dvs")[0]
        if current_aps.shape != current_event.shape:
            raise ValueError("APS and DVS frames must be spatially aligned")
        image_size = current_aps.shape[-2:]
        valid = bool(
            presence[frame_id]
            and boxes[frame_id, 2] > 0
            and boxes[frame_id, 3] > 0
        )
        if valid:
            context_slice = _search_context_slice(
                boxes[frame_id], current_aps.shape[-2], current_aps.shape[-1],
                thresholds["low_light_search_factor"])
            context_median = (
                current_aps[context_slice].median()
                if context_slice is not None
                else current_aps.new_tensor(float("inf"))
            )
            low_light[frame_id] = bool(
                context_median
                <= thresholds["low_light_context_median"])
            if template_frame is None:
                template_frame = current_aps.clone()
                template_box = boxes[frame_id].clone()
            ambiguity[frame_id] = _aps_ambiguity(
                template_frame, current_aps, template_box, boxes[frame_id])
            event_motion[frame_id] = _event_motion_pair(
                current_event, previous_event, boxes[frame_id])
        previous_event = current_event

    if image_size is None:
        raise ValueError("sequence contains no frames")
    result = classify_felt_challenges(
        boxes,
        presence,
        image_size,
        event_motion=event_motion,
        low_light=low_light,
        ambiguity=ambiguity,
        small_area_ratio=thresholds["small_area_ratio"],
        max_small_aspect_ratio=thresholds["max_small_aspect_ratio"],
        motion_norm=thresholds["motion_norm"],
        ambiguity_threshold=thresholds["ambiguity_threshold"],
        recovery_window=thresholds["recovery_window"],
    )
    return {
        "attributes": {
            name: values.tolist()
            for name, values in result["challenge_attributes"].items()
        },
    }


def _config_hash(thresholds):
    encoded = json.dumps(
        thresholds, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest(payload):
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported challenge manifest schema")
    thresholds = payload.get("thresholds")
    sequences = payload.get("sequences")
    if not isinstance(thresholds, dict) or not isinstance(sequences, dict):
        raise ValueError("manifest thresholds and sequences must be mappings")
    if payload.get("config_sha256") != _config_hash(thresholds):
        raise ValueError("manifest threshold hash does not match its contents")
    for name, record in sequences.items():
        attributes = record.get("attributes") if isinstance(record, dict) else None
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError("manifest sequence records must be mappings")
        if set(record) != {"attributes"}:
            raise ValueError("manifest records must contain only attributes")
        if (not isinstance(attributes, dict)
                or set(attributes) != set(CHALLENGE_NAMES)):
            raise ValueError("manifest records require every challenge attribute")
        if any(not isinstance(values, list) for values in attributes.values()):
            raise ValueError("manifest challenge attributes must be lists")
        lengths = {len(values) for values in attributes.values()}
        if len(lengths) != 1:
            raise ValueError("manifest challenge attributes must have equal lengths")
        if any(type(value) is not bool
               for values in attributes.values() for value in values):
            raise ValueError("manifest challenge attributes must contain booleans")
    return payload


def write_manifest(path, sequences, thresholds):
    path = Path(path)
    payload = _validate_manifest({
        "schema_version": 2,
        "config_sha256": _config_hash(thresholds),
        "thresholds": dict(thresholds),
        "sequences": dict(sequences),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return _validate_manifest(json.load(handle))
