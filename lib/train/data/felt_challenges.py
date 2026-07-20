import torch


GENERALIST = 0
MOTION = 1
SMALL_TARGET = 2
VISIBILITY = 3
DISCRIMINATION = 4
EXPERT_COUNT = 5
CHALLENGE_NAMES = (
    "small_target",
    "motion",
    "low_light",
    "recovery",
    "ambiguity",
    "deformation",
    "absent",
)
EXPERT_CHALLENGE_NAMES = {
    MOTION: ("motion",),
    SMALL_TARGET: ("small_target",),
    VISIBILITY: ("absent", "recovery"),
    DISCRIMINATION: ("low_light", "ambiguity", "deformation"),
}

DEFAULT_SMALL_AREA_RATIO = 0.015868
DEFAULT_SMALL_MAX_ASPECT_RATIO = 4.0
DEFAULT_MOTION_NORM = 0.164362
DEFAULT_DEFORMATION = 0.092424
DEFAULT_AMBIGUITY_THRESHOLD = 1.29
DEFAULT_RECOVERY_WINDOW = 8


def classify_felt_frames(
        boxes, presence, image_size, *,
        small_area_ratio=DEFAULT_SMALL_AREA_RATIO,
        max_small_aspect_ratio=DEFAULT_SMALL_MAX_ASPECT_RATIO,
        low_light=None,
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
    if max_small_aspect_ratio < 1:
        raise ValueError("max_small_aspect_ratio must be at least one")

    count = boxes.shape[0]
    if low_light is None:
        low_light = torch.zeros(count, dtype=torch.bool)
    else:
        low_light = torch.as_tensor(low_light, dtype=torch.bool).reshape(-1)
        if low_light.numel() != count:
            raise ValueError("low_light must provide one value per frame")
    valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    visible = presence & valid
    consecutive = visible & torch.cat([
        torch.zeros(1, dtype=torch.bool), visible[:-1]
    ])

    nan = torch.full((count,), float("nan"), dtype=torch.float32)
    area = boxes[:, 2] * boxes[:, 3]
    area_ratio = nan.clone()
    area_ratio[valid] = area[valid] / (height * width)
    aspect_ratio = nan.clone()
    aspect_ratio[valid] = torch.maximum(
        boxes[valid, 2] / boxes[valid, 3],
        boxes[valid, 3] / boxes[valid, 2],
    )
    small_target = (
        visible
        & (area_ratio <= small_area_ratio)
        & (aspect_ratio <= max_small_aspect_ratio)
    )
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
    motion = visible & (motion_score >= motion_norm)
    deformation_challenge = visible & (deformation_score >= deformation)
    reappearing = visible & torch.cat([
        torch.zeros(1, dtype=torch.bool), ~visible[:-1]
    ])
    recovery = torch.zeros(count, dtype=torch.bool)
    for offset in range(recovery_window):
        if offset >= count:
            break
        recovery[offset:] |= reappearing[:count - offset]

    return {
        "visible": visible,
        "recovery": recovery,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "low_light": low_light,
        "small_target": small_target,
        "motion": motion,
        "deformation_challenge": deformation_challenge,
        "absent": ~presence,
        "motion_norm": motion_score,
        "deformation": deformation_score,
    }


def classify_felt_challenges(
        boxes, presence, image_size, *, event_motion, ambiguity,
        small_area_ratio=DEFAULT_SMALL_AREA_RATIO,
        max_small_aspect_ratio=DEFAULT_SMALL_MAX_ASPECT_RATIO,
        low_light=None,
        motion_norm=DEFAULT_MOTION_NORM,
        ambiguity_threshold=DEFAULT_AMBIGUITY_THRESHOLD,
        recovery_window=DEFAULT_RECOVERY_WINDOW):
    result = classify_felt_frames(
        boxes,
        presence,
        image_size,
        small_area_ratio=small_area_ratio,
        max_small_aspect_ratio=max_small_aspect_ratio,
        low_light=low_light,
        motion_norm=motion_norm,
        recovery_window=recovery_window,
    )
    event_motion = torch.as_tensor(event_motion, dtype=torch.float32).reshape(-1)
    ambiguity = torch.as_tensor(ambiguity, dtype=torch.float32).reshape(-1)
    frame_count = result["visible"].numel()
    if event_motion.numel() != frame_count or ambiguity.numel() != frame_count:
        raise ValueError("observation scores must provide one value per frame")
    if not torch.isfinite(event_motion).all() or not torch.isfinite(ambiguity).all():
        raise ValueError("observation scores must be finite")

    visible = result["visible"]
    motion = result["motion"] & (event_motion > 0)
    ambiguity_challenge = visible & (ambiguity >= ambiguity_threshold)
    challenge_attributes = {
        "small_target": result["small_target"],
        "motion": motion,
        "low_light": visible & result["low_light"],
        "recovery": visible & result["recovery"],
        "ambiguity": ambiguity_challenge,
        "deformation": result["deformation_challenge"],
        "absent": result["absent"],
    }

    result.update(
        ambiguity=ambiguity,
        event_motion=event_motion,
        challenge_attributes=challenge_attributes,
    )
    return result


def expert_supervision_mask(challenge_attributes):
    """Map independent challenge labels to overlapping expert eligibility."""
    if set(challenge_attributes) != set(CHALLENGE_NAMES):
        raise ValueError("challenge attributes must contain the declared labels")
    labels = {
        name: torch.as_tensor(challenge_attributes[name], dtype=torch.bool)
        .reshape(-1)
        for name in CHALLENGE_NAMES
    }
    lengths = {values.numel() for values in labels.values()}
    if len(lengths) != 1:
        raise ValueError("challenge attributes must have equal frame counts")

    frame_count = lengths.pop()
    device = next(iter(labels.values())).device
    mask = torch.zeros(
        (frame_count, EXPERT_COUNT), dtype=torch.bool, device=device)
    for expert_id, names in EXPERT_CHALLENGE_NAMES.items():
        for name in names:
            mask[:, expert_id] |= labels[name]
    mask[:, GENERALIST] = ~mask[:, 1:].any(dim=1)
    return mask
