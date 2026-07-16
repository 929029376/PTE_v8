from itertools import combinations, permutations

import torch
import torch.nn.functional as F


def _single_channel_map(value, name):
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape (B,1,H,W) or (B,H,W)")
    return value


def _gather_map(value, indices):
    batch_size, channels = value.shape[:2]
    gather_index = indices.unsqueeze(1).expand(batch_size, channels, -1)
    return value.flatten(2).gather(2, gather_index).transpose(1, 2)


def extract_hypotheses(field, candidate_map, size_map, offset_map,
                       identity_map, k_max=5, cumulative_mass=0.9):
    """Decode diverse candidate hypotheses from the actual token grid."""
    field = _single_channel_map(field, "field")
    candidate_map = _single_channel_map(candidate_map, "candidate_map")
    if field.shape != candidate_map.shape:
        raise ValueError("field and candidate_map must have the same shape")
    if size_map.ndim != 4 or size_map.shape[1] != 2:
        raise ValueError("size_map must have shape (B,2,H,W)")
    if offset_map.ndim != 4 or offset_map.shape[1] != 2:
        raise ValueError("offset_map must have shape (B,2,H,W)")
    if identity_map.ndim != 4:
        raise ValueError("identity_map must have shape (B,D,H,W)")
    if size_map.shape[0] != field.shape[0] or size_map.shape[-2:] != field.shape[-2:]:
        raise ValueError("size_map must use the field batch and spatial shape")
    if offset_map.shape[0] != field.shape[0] or offset_map.shape[-2:] != field.shape[-2:]:
        raise ValueError("offset_map must use the field batch and spatial shape")
    if identity_map.shape[0] != field.shape[0] or identity_map.shape[-2:] != field.shape[-2:]:
        raise ValueError("identity_map must use the field batch and spatial shape")
    if not 1 <= int(k_max) <= 5:
        raise ValueError("k_max must be in [1,5]")
    if not 0.0 < float(cumulative_mass) <= 1.0:
        raise ValueError("cumulative_mass must be in (0,1]")

    batch_size, _, height, width = field.shape
    cell_count = height * width
    selected_count = min(int(k_max), cell_count)
    combined = (field * candidate_map).squeeze(1)

    pooled = F.max_pool2d(
        combined.unsqueeze(1), kernel_size=3, stride=1, padding=1
    ).squeeze(1)
    local_max = combined == pooled
    local_max = local_max & (combined > 0)
    priority = torch.arange(
        cell_count, 0, -1, device=combined.device, dtype=combined.dtype
    ).reshape(1, height, width)
    local_priority = torch.where(local_max, priority, torch.zeros_like(priority))
    neighborhood_priority = F.max_pool2d(
        local_priority.unsqueeze(1), kernel_size=3, stride=1, padding=1
    ).squeeze(1)
    local_max = local_max & (local_priority == neighborhood_priority)

    ranked_flat = combined.flatten(1).masked_fill(
        ~local_max.flatten(1), -torch.inf)
    indices = torch.argsort(
        ranked_flat, dim=1, descending=True, stable=True
    )[:, :selected_count]
    valid = torch.isfinite(ranked_flat.gather(1, indices))
    scores = combined.flatten(1).gather(1, indices).masked_fill(~valid, 0.0)

    if selected_count < int(k_max):
        pad = int(k_max) - selected_count
        indices = F.pad(indices, (0, pad))
        scores = F.pad(scores, (0, pad))
        valid = F.pad(valid, (0, pad), value=False)

    peak_count = local_max.flatten(1).sum(dim=1)
    local_mass = combined.masked_fill(~local_max, 0.0).flatten(1).sum(dim=1, keepdim=True)
    global_posterior = scores / local_mass.clamp_min(torch.finfo(scores.dtype).eps)
    cumulative = global_posterior.cumsum(dim=1)
    reached = cumulative >= float(cumulative_mass)
    first_reached = reached.to(torch.int64).argmax(dim=1) + 1
    capped_count = peak_count.clamp(max=int(k_max)).to(torch.long)
    count = torch.where(reached.any(dim=1), first_reached, capped_count)
    count = torch.where(peak_count > 0, count, torch.ones_like(count))

    active_mask = torch.arange(
        int(k_max), device=field.device
    ).unsqueeze(0) < count.unsqueeze(1)
    active_scores = scores * active_mask.to(scores.dtype)
    posterior = active_scores / active_scores.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(scores.dtype).eps
    )
    no_mass = active_scores.sum(dim=1) <= 0
    if no_mass.any():
        posterior = posterior.clone()
        posterior[no_mass] = 0.0
        posterior[no_mass, 0] = 1.0

    gathered_size = _gather_map(size_map, indices)
    gathered_offset = _gather_map(offset_map, indices)
    gathered_identity = F.normalize(
        _gather_map(identity_map, indices), dim=-1, eps=1e-8
    )
    field_scores = _gather_map(field, indices).squeeze(-1)
    candidate_scores = _gather_map(candidate_map, indices).squeeze(-1)
    x = (indices % width).to(field.dtype)
    y = torch.div(indices, width, rounding_mode="floor").to(field.dtype)
    boxes = torch.stack((
        (x + gathered_offset[..., 0]) / width,
        (y + gathered_offset[..., 1]) / height,
        gathered_size[..., 0],
        gathered_size[..., 1],
    ), dim=-1).clamp(0.0, 1.0)

    return {
        "boxes": boxes,
        "scores": scores,
        "weights": scores,
        "field_scores": field_scores,
        "candidate_scores": candidate_scores,
        "posterior": posterior,
        "identity": gathered_identity,
        "indices": indices,
        "active_mask": active_mask,
        "count": count,
        "local_maxima": local_max.unsqueeze(1),
    }


def crop_cxcywh_to_image_xywh(boxes, resize_factor, patch_size, crop_center):
    boxes = torch.as_tensor(boxes)
    if boxes.shape[-1] != 4:
        raise ValueError("boxes must end with cxcywh")
    scale = boxes.new_tensor(float(patch_size) / float(resize_factor))
    center = boxes.new_tensor(crop_center)
    image_center = boxes[..., :2].sub(0.5).mul(scale).add(center)
    image_size = boxes[..., 2:].mul(scale)
    return torch.cat((image_center - 0.5 * image_size, image_size), dim=-1)


def image_xywh_to_crop_cxcywh(boxes, resize_factor, patch_size, crop_center):
    boxes = torch.as_tensor(boxes)
    if boxes.shape[-1] != 4:
        raise ValueError("boxes must end with xywh")
    scale = boxes.new_tensor(float(patch_size) / float(resize_factor))
    center = boxes.new_tensor(crop_center)
    image_center = boxes[..., :2] + 0.5 * boxes[..., 2:]
    crop_center_xy = (image_center - center) / scale + 0.5
    return torch.cat((crop_center_xy, boxes[..., 2:] / scale), dim=-1)


def _box_iou(first, second):
    first_xyxy = torch.cat((
        first[..., :2] - 0.5 * first[..., 2:],
        first[..., :2] + 0.5 * first[..., 2:],
    ), dim=-1)
    second_xyxy = torch.cat((
        second[..., :2] - 0.5 * second[..., 2:],
        second[..., :2] + 0.5 * second[..., 2:],
    ), dim=-1)
    top_left = torch.maximum(first_xyxy[..., :2], second_xyxy[..., :2])
    bottom_right = torch.minimum(first_xyxy[..., 2:], second_xyxy[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    first_area = (first_xyxy[..., 2:] - first_xyxy[..., :2]).clamp_min(0).prod(dim=-1)
    second_area = (second_xyxy[..., 2:] - second_xyxy[..., :2]).clamp_min(0).prod(dim=-1)
    return intersection / (first_area + second_area - intersection).clamp_min(1e-8)


class HypothesisTracker:
    """Small exact lifecycle tracker for at most five reappearance hypotheses."""

    def __init__(self, k_max=5, identity_cost=0.6, box_cost=0.4,
                 min_identity=0.4, max_center_distance=0.5,
                 merge_iou=0.7, merge_identity=0.8,
                 previous_weight=0.7, observation_weight=0.3,
                 miss_decay=0.85, min_weight=0.05, max_age=32,
                 cumulative_mass=0.9):
        if not 1 <= int(k_max) <= 5:
            raise ValueError("k_max must be in [1,5]")
        self.k_max = int(k_max)
        self.identity_cost = float(identity_cost)
        self.box_cost = float(box_cost)
        self.min_identity = float(min_identity)
        self.max_center_distance = float(max_center_distance)
        self.merge_iou = float(merge_iou)
        self.merge_identity = float(merge_identity)
        self.previous_weight = float(previous_weight)
        self.observation_weight = float(observation_weight)
        self.miss_decay = float(miss_decay)
        self.min_weight = float(min_weight)
        self.max_age = int(max_age)
        self.cumulative_mass = float(cumulative_mass)

    @staticmethod
    def _prepare(state, filter_active=False):
        if state is None:
            return None
        boxes = state["boxes"]
        batched = boxes.ndim == 3
        if batched:
            if boxes.shape[0] != 1:
                raise ValueError("HypothesisTracker updates one sequence at a time")
            boxes = boxes[0]
        weights = state.get(
            "field_scores", state.get("scores", state.get("weights")))
        identity = state["identity"]
        if batched:
            weights = weights[0]
            identity = identity[0]
        velocity = state.get("velocity")
        age = state.get("age")
        if velocity is None:
            velocity = torch.zeros_like(boxes)
        elif batched and velocity.ndim == 3:
            velocity = velocity[0]
        if age is None:
            age = torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device)
        elif batched and age.ndim == 2:
            age = age[0]
        if filter_active and "active_mask" in state:
            active = state["active_mask"]
            if active.ndim == 2:
                active = active[0]
            boxes = boxes[active]
            weights = weights[active]
            identity = identity[active]
            velocity = velocity[active]
            age = age[active]
        return {
            "boxes": boxes,
            "weights": weights,
            "identity": F.normalize(identity, dim=-1, eps=1e-8),
            "velocity": velocity,
            "age": age.to(torch.long),
        }

    @staticmethod
    @torch.no_grad()
    def _optimal_matches(cost, valid):
        previous_count, observed_count = cost.shape
        max_matches = min(previous_count, observed_count)
        for match_count in range(max_matches, 0, -1):
            best = None
            best_cost = None
            for previous_ids in combinations(range(previous_count), match_count):
                for observed_ids in permutations(range(observed_count), match_count):
                    pairs = tuple(zip(previous_ids, observed_ids))
                    if not all(bool(valid[i, j]) for i, j in pairs):
                        continue
                    total = sum(float(cost[i, j]) for i, j in pairs)
                    if best_cost is None or total < best_cost:
                        best = pairs
                        best_cost = total
            if best is not None:
                return list(best)
        return []

    def _merge_duplicates(self, boxes, weights, identity, velocity, age):
        if weights.numel() == 0:
            return boxes, weights, identity, velocity, age
        order = torch.argsort(weights, descending=True, stable=True)
        merged = []
        for index in order.tolist():
            item = [
                boxes[index].clone(),
                weights[index].clone(),
                identity[index].clone(),
                velocity[index].clone(),
                age[index].clone(),
            ]
            duplicate = None
            for merged_index, current in enumerate(merged):
                iou = _box_iou(current[0], item[0])
                cosine = F.cosine_similarity(
                    current[2].unsqueeze(0), item[2].unsqueeze(0), dim=-1
                )[0]
                if iou >= self.merge_iou and cosine >= self.merge_identity:
                    duplicate = merged_index
                    break
            if duplicate is None:
                merged.append(item)
                continue
            current = merged[duplicate]
            total_weight = current[1] + item[1]
            current_share = current[1] / total_weight.clamp_min(1e-8)
            item_share = item[1] / total_weight.clamp_min(1e-8)
            current[0] = current_share * current[0] + item_share * item[0]
            current[2] = F.normalize(
                current_share * current[2] + item_share * item[2], dim=0, eps=1e-8
            )
            current[3] = current_share * current[3] + item_share * item[3]
            current[1] = total_weight
            current[4] = torch.minimum(current[4], item[4])
        return tuple(torch.stack([item[i] for item in merged]) for i in range(5))

    @staticmethod
    def _empty(reference, identity_dim):
        return {
            "boxes": reference.new_empty((0, 4)),
            "weights": reference.new_empty((0,)),
            "posterior": reference.new_empty((0,)),
            "identity": reference.new_empty((0, identity_dim)),
            "velocity": reference.new_empty((0, 4)),
            "age": torch.empty((0,), dtype=torch.long, device=reference.device),
            "active_mask": torch.empty((0,), dtype=torch.bool, device=reference.device),
            "active_count": 0,
        }

    def update(self, previous, observed):
        previous = self._prepare(previous)
        observed = self._prepare(observed, filter_active=True)
        if previous is None and observed is None:
            raise ValueError("previous and observed cannot both be None")
        reference = observed["boxes"] if observed is not None else previous["boxes"]
        identity_dim = (
            observed["identity"].shape[-1] if observed is not None
            else previous["identity"].shape[-1]
        )
        if previous is None:
            previous = self._empty(reference, identity_dim)
        if observed is None:
            observed = self._empty(reference, identity_dim)

        predicted = previous["boxes"] + previous["velocity"]
        predicted_size = torch.where(
            predicted[:, 2:] > 0.0,
            predicted[:, 2:],
            previous["boxes"][:, 2:],
        ).clamp(min=1e-4, max=1.0)
        predicted = torch.cat((
            predicted[:, :2].clamp(0.0, 1.0),
            predicted_size,
        ), dim=-1)
        matches = []
        if predicted.shape[0] and observed["boxes"].shape[0]:
            iou = _box_iou(predicted[:, None, :], observed["boxes"][None, :, :])
            cosine = F.cosine_similarity(
                previous["identity"][:, None, :], observed["identity"][None, :, :], dim=-1
            )
            center_distance = torch.linalg.vector_norm(
                predicted[:, None, :2] - observed["boxes"][None, :, :2], dim=-1
            )
            valid = ((cosine >= self.min_identity)
                     & (center_distance <= self.max_center_distance))
            cost = self.identity_cost * (1.0 - cosine) + self.box_cost * (1.0 - iou)
            matches = self._optimal_matches(cost, valid)

        matched_previous = {pair[0] for pair in matches}
        matched_observed = {pair[1] for pair in matches}
        boxes = []
        weights = []
        identities = []
        velocities = []
        ages = []
        for previous_index, observed_index in matches:
            previous_box = previous["boxes"][previous_index]
            observed_box = observed["boxes"][observed_index]
            boxes.append(observed_box)
            weights.append(
                self.previous_weight * previous["weights"][previous_index]
                + self.observation_weight * observed["weights"][observed_index]
            )
            identities.append(F.normalize(
                self.previous_weight * previous["identity"][previous_index]
                + self.observation_weight * observed["identity"][observed_index],
                dim=0,
                eps=1e-8,
            ))
            velocities.append(observed_box - previous_box)
            ages.append(previous["age"].new_zeros(()))
        for index in range(previous["boxes"].shape[0]):
            if index in matched_previous:
                continue
            boxes.append(predicted[index])
            weights.append(self.miss_decay * previous["weights"][index])
            identities.append(previous["identity"][index])
            velocities.append(previous["velocity"][index])
            ages.append(previous["age"][index] + 1)
        for index in range(observed["boxes"].shape[0]):
            if index in matched_observed:
                continue
            boxes.append(observed["boxes"][index])
            weights.append(observed["weights"][index])
            identities.append(observed["identity"][index])
            velocities.append(torch.zeros_like(observed["boxes"][index]))
            ages.append(observed["age"][index].new_zeros(()))

        if not boxes:
            return self._empty(reference, identity_dim)
        boxes = torch.stack(boxes)
        weights = torch.stack(weights)
        identities = torch.stack(identities)
        velocities = torch.stack(velocities)
        ages = torch.stack(ages)
        boxes, weights, identities, velocities, ages = self._merge_duplicates(
            boxes, weights, identities, velocities, ages
        )

        keep = (weights >= self.min_weight) & (ages <= self.max_age)
        boxes = boxes[keep]
        weights = weights[keep]
        identities = identities[keep]
        velocities = velocities[keep]
        ages = ages[keep]
        if weights.numel() == 0:
            return self._empty(reference, identity_dim)

        order = torch.argsort(weights, descending=True, stable=True)[:self.k_max]
        boxes = boxes[order]
        weights = weights[order]
        identities = identities[order]
        velocities = velocities[order]
        ages = ages[order]
        posterior = weights / weights.sum().clamp_min(1e-8)
        cumulative = posterior.cumsum(dim=0)
        reached = (cumulative >= self.cumulative_mass).nonzero(as_tuple=False)
        active_count = int(reached[0, 0]) + 1 if reached.numel() else weights.numel()
        active_mask = torch.arange(weights.numel(), device=weights.device) < active_count
        return {
            "boxes": boxes,
            "weights": weights,
            "posterior": posterior,
            "identity": identities,
            "velocity": velocities,
            "age": ages,
            "active_mask": active_mask,
            "active_count": active_count,
        }


def build_hypothesis_tracker(cfg):
    hypotheses = cfg.MODEL.SRBT.HYPOTHESES
    return HypothesisTracker(
        k_max=int(hypotheses.K_MAX),
        cumulative_mass=float(hypotheses.CUMULATIVE_MASS),
        identity_cost=float(hypotheses.IDENTITY_COST),
        box_cost=float(hypotheses.BOX_COST),
        min_identity=float(hypotheses.MIN_IDENTITY),
        max_center_distance=float(hypotheses.MAX_CENTER_DISTANCE),
        merge_iou=float(hypotheses.MERGE_IOU),
        merge_identity=float(hypotheses.MERGE_IDENTITY),
        previous_weight=float(hypotheses.PREVIOUS_WEIGHT),
        observation_weight=float(hypotheses.OBSERVATION_WEIGHT),
        miss_decay=float(hypotheses.MISS_DECAY),
        min_weight=float(hypotheses.MIN_WEIGHT),
        max_age=int(hypotheses.MAX_AGE),
    )
