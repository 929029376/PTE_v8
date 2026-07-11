import torch


ROUTE_MOTION_DIM = 6


def causal_route_motion_cues(prior_box, previous_box, image_hw,
                             device=None, dtype=torch.float32):
    """Encode motion known before the current frame is routed.

    Boxes are image-space ``xywh`` from t-2 and t-1. The current box is
    intentionally not accepted, keeping this feature available at inference.
    """
    prior = torch.as_tensor(prior_box, device=device, dtype=dtype)
    previous = torch.as_tensor(previous_box, device=device, dtype=dtype)
    unbatched = prior.ndim == 1
    if unbatched:
        prior = prior.unsqueeze(0)
        previous = previous.unsqueeze(0)
    if prior.shape != previous.shape or prior.ndim != 2 or prior.shape[1] != 4:
        raise ValueError("prior_box and previous_box must have shape (B, 4)")

    image_hw = torch.as_tensor(image_hw, device=prior.device, dtype=prior.dtype)
    if image_hw.numel() != 2:
        raise ValueError("image_hw must contain (height, width)")
    height, width = image_hw.flatten()
    image_scale = torch.stack([width, height]).clamp_min(1.0)

    valid = (
        (prior[:, 2:] > 0).all(dim=1)
        & (previous[:, 2:] > 0).all(dim=1)
        & (height > 0)
        & (width > 0)
    )
    prior_center = prior[:, :2] + 0.5 * prior[:, 2:]
    previous_center = previous[:, :2] + 0.5 * previous[:, 2:]
    displacement = ((previous_center - prior_center) / image_scale).clamp(-1.0, 1.0)
    speed = displacement.square().sum(dim=1, keepdim=True).sqrt().clamp_max(1.0)

    prior_area = prior[:, 2] * prior[:, 3]
    previous_area = previous[:, 2] * previous[:, 3]
    area_change = torch.tanh(torch.log(
        (previous_area / prior_area.clamp_min(1e-6)).clamp_min(1e-6)))
    prior_aspect = prior[:, 2] / prior[:, 3].clamp_min(1e-6)
    previous_aspect = previous[:, 2] / previous[:, 3].clamp_min(1e-6)
    aspect_change = torch.tanh(torch.log(
        (previous_aspect / prior_aspect.clamp_min(1e-6)).clamp_min(1e-6)))

    cues = torch.cat([
        displacement,
        speed,
        area_change.unsqueeze(1),
        aspect_change.unsqueeze(1),
        valid.to(prior.dtype).unsqueeze(1),
    ], dim=1)
    cues = torch.where(valid.unsqueeze(1), cues, torch.zeros_like(cues))
    return cues.squeeze(0) if unbatched else cues
