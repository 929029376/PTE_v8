import torch


GENERALIST = 0
MOTION = 1
SMALL_TARGET = 2
VISIBILITY = 3
DISCRIMINATION = 4
EXPERT_COUNT = 5

DEFAULT_SMALL_AREA_RATIO = 0.015868
DEFAULT_MOTION_NORM = 0.164362
DEFAULT_DEFORMATION = 0.092424
DEFAULT_RECOVERY_WINDOW = 8


def classify_felt_frames(
        boxes, presence, image_size, *,
        small_area_ratio=DEFAULT_SMALL_AREA_RATIO,
        motion_norm=DEFAULT_MOTION_NORM,
        deformation=DEFAULT_DEFORMATION,
        recovery_window=DEFAULT_RECOVERY_WINDOW):
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    presence = torch.as_tensor(presence, dtype=torch.bool).reshape(-1)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N,4)")
    if boxes.shape[0] != presence.numel():
        raise ValueError("boxes and presence must have equal length")
    if len(image_size) < 2:
        raise ValueError("image_size must contain height and width")
    height, width = float(image_size[0]), float(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if recovery_window < 1:
        raise ValueError("recovery_window must be positive")

    count = boxes.shape[0]
    valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    visible = presence & valid
    consecutive = visible & torch.cat([
        torch.zeros(1, dtype=torch.bool), visible[:-1]
    ])

    nan = torch.full((count,), float("nan"), dtype=torch.float32)
    area = boxes[:, 2] * boxes[:, 3]
    area_ratio = nan.clone()
    area_ratio[valid] = area[valid] / (height * width)
    center = boxes[:, :2] + boxes[:, 2:] * 0.5

    motion_score = nan.clone()
    scale_change = nan.clone()
    aspect_change = nan.clone()
    if count > 1:
        current = consecutive[1:]
        previous_area = area[:-1].clamp_min(1.0)
        displacement = torch.linalg.vector_norm(center[1:] - center[:-1], dim=-1)
        motion_score[1:][current] = (
            displacement[current] / previous_area[current].sqrt())
        scale_change[1:][current] = torch.abs(torch.log(
            area[1:][current].clamp_min(1.0) / previous_area[current]))
        aspect = boxes[:, 2] / boxes[:, 3].clamp_min(1e-6)
        aspect_change[1:][current] = torch.abs(torch.log(
            aspect[1:][current].clamp_min(1e-6)
            / aspect[:-1][current].clamp_min(1e-6)))

    deformation_score = torch.maximum(scale_change, aspect_change)
    reappearing = visible & torch.cat([
        torch.zeros(1, dtype=torch.bool), ~visible[:-1]
    ])
    recovery = torch.zeros(count, dtype=torch.bool)
    for offset in range(recovery_window):
        if offset >= count:
            break
        recovery[offset:] |= reappearing[:count - offset]

    class_id = torch.full((count,), -1, dtype=torch.long)
    class_id[visible] = GENERALIST
    class_id[visible & (deformation_score >= deformation)] = DISCRIMINATION
    class_id[visible & (motion_score >= motion_norm)] = MOTION
    class_id[visible & (area_ratio <= small_area_ratio)] = SMALL_TARGET
    class_id[(~presence) | recovery] = VISIBILITY

    return {
        "class_id": class_id,
        "visible": visible,
        "recovery": recovery,
        "area_ratio": area_ratio,
        "motion_norm": motion_score,
        "deformation": deformation_score,
    }


def assign_expert_owners(
        boxes, presence, image_size, *, event_motion, ambiguity,
        small_area_ratio=DEFAULT_SMALL_AREA_RATIO,
        motion_norm=DEFAULT_MOTION_NORM,
        ambiguity_threshold=0.8,
        recovery_window=DEFAULT_RECOVERY_WINDOW):
    result = classify_felt_frames(
        boxes,
        presence,
        image_size,
        small_area_ratio=small_area_ratio,
        motion_norm=motion_norm,
        recovery_window=recovery_window,
    )
    event_motion = torch.as_tensor(event_motion, dtype=torch.float32).reshape(-1)
    ambiguity = torch.as_tensor(ambiguity, dtype=torch.float32).reshape(-1)
    frame_count = result["class_id"].numel()
    if event_motion.numel() != frame_count or ambiguity.numel() != frame_count:
        raise ValueError("observation scores must provide one value per frame")
    if not torch.isfinite(event_motion).all() or not torch.isfinite(ambiguity).all():
        raise ValueError("observation scores must be finite")

    visible = result["visible"]
    owners = torch.full_like(result["class_id"], -1)
    owners[visible] = GENERALIST
    owners[visible & (ambiguity >= ambiguity_threshold)] = DISCRIMINATION
    owners[
        visible
        & (result["motion_norm"] >= motion_norm)
        & (event_motion > 0)
    ] = MOTION
    owners[visible & (result["area_ratio"] <= small_area_ratio)] = SMALL_TARGET
    owners[(~torch.as_tensor(presence, dtype=torch.bool).reshape(-1))
           | result["recovery"]] = VISIBILITY
    result.update(
        class_id=owners,
        ambiguity=ambiguity,
        event_motion=event_motion,
    )
    return result
