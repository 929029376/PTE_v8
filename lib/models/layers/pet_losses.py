"""Joint causal SRBT training losses."""

import torch
import torch.nn.functional as F
from torch import nn


def _zero(anchor):
    return anchor.float().reshape(-1)[:0].sum()


def _masked_mean(values, mask):
    mask = mask.to(device=values.device, dtype=torch.bool).reshape(-1)
    if mask.shape != values.shape[:1]:
        raise ValueError("loss mask must have shape (B,)")
    return values[mask].mean() if mask.any() else _zero(values)


class PetTrackLoss(nn.Module):
    """One objective stack for the SRBT student and future teacher."""

    def __init__(self, existence_weight=1.0, survival_weight=1.0,
                 field_weight=1.0, hypothesis_weight=0.5,
                 identity_weight=0.2, calibration_weight=0.05,
                 teacher_weight=1.0, distill_max_weight=0.5,
                 distill_warmup=0.05, identity_temperature=0.1,
                 diversity_margin=0.25, diversity_weight=0.1,
                 hazard_bins=129):
        super().__init__()
        self.existence_weight = float(existence_weight)
        self.survival_weight = float(survival_weight)
        self.field_weight = float(field_weight)
        self.hypothesis_weight = float(hypothesis_weight)
        self.identity_weight = float(identity_weight)
        self.calibration_weight = float(calibration_weight)
        self.teacher_weight = float(teacher_weight)
        self.distill_max_weight = float(distill_max_weight)
        self.distill_warmup = float(distill_warmup)
        self.identity_temperature = float(identity_temperature)
        self.diversity_margin = float(diversity_margin)
        self.diversity_weight = float(diversity_weight)
        self.hazard_bins = int(hazard_bins)
        if not 0.0 <= self.distill_warmup < 1.0:
            raise ValueError("distill_warmup must be in [0, 1)")
        if self.identity_temperature <= 0.0:
            raise ValueError("identity_temperature must be positive")
        if self.hazard_bins != 129:
            raise ValueError("SRBT requires exactly 129 hazard bins")

    @staticmethod
    def _anchor(predictions, teacher):
        for values in (predictions, teacher or {}):
            for value in values.values():
                if torch.is_tensor(value):
                    return value
        raise ValueError("SRBT loss needs at least one tensor prediction")

    def _survival_loss(self, hazard_logits, targets):
        probabilities = F.softmax(hazard_logits.float(), dim=-1)
        return self._survival_probability_loss(probabilities, targets)

    def _survival_probability_loss(self, probabilities, targets):
        batch, bins = probabilities.shape
        if bins != self.hazard_bins:
            raise ValueError(
                f"SRBT survival requires {self.hazard_bins} hazard bins")
        hazard_mask = torch.as_tensor(
            targets["hazard_mask"], device=probabilities.device,
            dtype=torch.bool).reshape(-1)
        event_time = torch.as_tensor(
            targets["hazard_target"], device=probabilities.device,
            dtype=torch.long).reshape(-1)
        censor = torch.as_tensor(
            targets["censor_mask"], device=probabilities.device,
            dtype=torch.bool).reshape(-1)
        if any(value.shape != (batch,) for value in (
                hazard_mask, event_time, censor)):
            raise ValueError("survival targets must have shape (B,)")
        valid = hazard_mask
        if not valid.any():
            return _zero(probabilities)
        if ((event_time[valid] < 1) | (event_time[valid] > bins)).any():
            raise ValueError("hazard_target is outside the hazard bins")
        if (censor[valid] & (event_time[valid] >= bins)).any():
            raise ValueError("censor target needs a later tail bin")
        event_likelihood = probabilities.gather(
            1, (event_time - 1).clamp(0, bins - 1)[:, None]
        ).squeeze(1)
        tail_probability = probabilities.flip(1).cumsum(1).flip(1)
        censor_likelihood = tail_probability.gather(
            1, event_time.clamp(0, bins - 1)[:, None]
        ).squeeze(1)
        likelihood = torch.where(censor, censor_likelihood, event_likelihood)
        return _masked_mean(-likelihood.clamp_min(1e-8).log(), valid)

    @staticmethod
    def _spatial_cross_entropy(logits, target, mask):
        if logits.shape != target.shape or logits.ndim != 4:
            raise ValueError("spatial prediction and target shapes must match")
        target = target.to(device=logits.device, dtype=logits.dtype)
        target = target.flatten(1)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        per_sample = -(target * F.log_softmax(
            logits.float().flatten(1), dim=-1)).sum(dim=-1)
        return _masked_mean(per_sample, mask)

    @staticmethod
    def _spatial_probability_cross_entropy(probability, target, mask):
        if probability.shape != target.shape:
            raise ValueError("teacher spatial target shape mismatch")
        target = target.to(device=probability.device, dtype=probability.dtype)
        target = target.flatten(1)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        per_sample = -(target * probability.flatten(1).clamp_min(
            1e-8).log()).sum(dim=-1)
        return _masked_mean(per_sample, mask)

    def _hypothesis_loss(self, boxes, targets):
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("hypothesis_boxes must have shape (B, K, 4)")
        target = targets["target_box"].to(
            device=boxes.device, dtype=boxes.dtype)
        mask = torch.as_tensor(
            targets["hypothesis_mask"], device=boxes.device,
            dtype=torch.bool).reshape(-1)
        if target.shape != (boxes.shape[0], 4):
            raise ValueError("target_box must have shape (B, 4)")
        localization = _masked_mean(
            (boxes - target[:, None]).abs().mean(dim=-1).min(dim=-1).values,
            mask,
        )
        if boxes.shape[1] < 2 or not mask.any():
            diversity = _zero(boxes)
        else:
            centers = boxes[:, :, :2]
            distances = torch.cdist(centers, centers)
            pairs = torch.triu(torch.ones_like(
                distances, dtype=torch.bool), diagonal=1)
            pairs = pairs & mask[:, None, None]
            diversity = F.relu(
                self.diversity_margin - distances[pairs]).mean()
        return localization + self.diversity_weight * diversity, localization, diversity

    def _identity_loss(self, predictions, targets):
        candidates = predictions["identity_embeddings"]
        template = predictions["template_identity"]
        if candidates.ndim != 3 or template.shape != (
                candidates.shape[0], candidates.shape[2]):
            raise ValueError("identity tensors have incompatible shapes")
        batch, count, _ = candidates.shape
        mask = torch.as_tensor(
            targets["identity_mask"], device=candidates.device,
            dtype=torch.bool).reshape(-1)
        positive = torch.as_tensor(
            targets["identity_positive_index"], device=candidates.device,
            dtype=torch.long).reshape(-1)
        if mask.shape != (batch,) or positive.shape != (batch,):
            raise ValueError("identity targets must have shape (B,)")
        if not mask.any():
            return _zero(candidates)
        if ((positive[mask] < 0) | (positive[mask] >= count)).any():
            raise ValueError("identity positive index is invalid")
        query = F.normalize(template[mask].float(), dim=-1)
        bank = F.normalize(candidates.float().reshape(batch * count, -1), dim=-1)
        logits = query @ bank.transpose(0, 1) / self.identity_temperature
        sample_ids = torch.arange(batch, device=candidates.device)[mask]
        labels = sample_ids * count + positive[mask]
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _positive_indices(predictions, targets):
        if "identity_positive_index" in targets:
            return targets["identity_positive_index"]
        boxes = predictions.get("hypothesis_boxes")
        target = targets.get("target_box")
        if boxes is None or target is None:
            return None
        return (boxes.detach() - target.to(boxes.device)[:, None]).abs().mean(
            dim=-1).argmin(dim=-1)

    def _calibration_loss(self, predictions, targets):
        terms = []
        existence_logits = predictions.get("existence_logits")
        if existence_logits is not None and "presence" in targets:
            presence = targets["presence"].to(
                existence_logits.device, dtype=existence_logits.dtype)
            terms.append((torch.sigmoid(existence_logits) - presence).square().mean())
        scores = predictions.get("hypothesis_scores")
        positive = self._positive_indices(predictions, targets)
        if scores is not None and positive is not None:
            if scores.ndim != 2:
                raise ValueError("hypothesis_scores must have shape (B, K)")
            positive = torch.as_tensor(
                positive, device=scores.device, dtype=torch.long).reshape(-1)
            if positive.shape != (scores.shape[0],):
                raise ValueError("hypothesis positive indices must have shape (B,)")
            mask = torch.as_tensor(
                targets.get("hypothesis_mask", torch.ones(
                    scores.shape[0], device=scores.device, dtype=torch.bool)),
                device=scores.device,
                dtype=torch.bool,
            ).reshape(-1)
            if mask.shape != (scores.shape[0],):
                raise ValueError("hypothesis_mask must have shape (B,)")
            if ((positive[mask] < 0) |
                    (positive[mask] >= scores.shape[1])).any():
                raise ValueError("hypothesis positive index is invalid")
            target = torch.zeros_like(scores)
            rows = torch.arange(scores.shape[0], device=scores.device)[mask]
            target[rows, positive[mask]] = 1.0
            terms.append(_masked_mean(
                (torch.sigmoid(scores) - target).square().mean(dim=-1),
                mask,
            ))
        if not terms:
            return _zero(next(iter(predictions.values())))
        return torch.stack(terms).mean()

    @staticmethod
    def _distribution_kl(student_logits, teacher_probability):
        teacher_probability = teacher_probability.detach().to(
            device=student_logits.device, dtype=student_logits.dtype)
        if student_logits.shape != teacher_probability.shape:
            raise ValueError("student and teacher distributions must match")
        teacher_probability = teacher_probability / teacher_probability.sum(
            dim=-1, keepdim=True).clamp_min(1e-8)
        return F.kl_div(
            F.log_softmax(student_logits.float(), dim=-1),
            teacher_probability.float(), reduction="batchmean")

    def _distillation_loss(self, predictions, teacher):
        teacher = {name: value.detach() for name, value in teacher.items()}
        terms = []
        if "existence_logits" in predictions and "existence" in teacher:
            terms.append(F.binary_cross_entropy_with_logits(
                predictions["existence_logits"].float(),
                teacher["existence"].to(
                    predictions["existence_logits"].device).float()))
        if "hazard_logits" in predictions and "hazard" in teacher:
            terms.append(self._distribution_kl(
                predictions["hazard_logits"], teacher["hazard"]))
        for student_name, teacher_name in (
                ("field_logits", "field"),
                ("candidate_logits", "candidate_map")):
            if student_name in predictions and teacher_name in teacher:
                terms.append(self._distribution_kl(
                    predictions[student_name].flatten(1),
                    teacher[teacher_name].flatten(1)))
        return torch.stack(terms).mean() if terms else _zero(
            next(iter(predictions.values())))

    def _teacher_supervision_loss(self, teacher, targets):
        terms = []
        if "existence" in teacher and "presence" in targets:
            terms.append(F.binary_cross_entropy(
                teacher["existence"].float().clamp(1e-7, 1 - 1e-7),
                targets["presence"].to(teacher["existence"].device).float()))
        if "hazard" in teacher and all(name in targets for name in (
                "hazard_target", "hazard_mask", "censor_mask")):
            terms.append(self._survival_probability_loss(
                teacher["hazard"], targets))
        if "field" in teacher and all(name in targets for name in (
                "field_target", "field_mask")):
            terms.append(self._spatial_probability_cross_entropy(
                teacher["field"], targets["field_target"],
                targets["field_mask"]))
        if "candidate_map" in teacher and all(name in targets for name in (
                "field_target", "field_mask")):
            terms.append(self._spatial_probability_cross_entropy(
                teacher["candidate_map"], targets["field_target"],
                targets["field_mask"]))
        if not terms:
            return _zero(next(iter(teacher.values())))
        return torch.stack(terms).mean()

    def _distill_weight(self, progress):
        progress = float(progress)
        if progress <= self.distill_warmup:
            return 0.0
        ratio = min(1.0, (progress - self.distill_warmup) /
                    (1.0 - self.distill_warmup))
        return self.distill_max_weight * ratio

    def srbt_loss(self, predictions, targets, teacher=None, progress=0.0):
        anchor = self._anchor(predictions, teacher)
        losses = {name: _zero(anchor) for name in (
            "existence", "survival", "field", "hypothesis", "identity",
            "calibration", "distill", "teacher")}
        details = {}

        if "existence_logits" in predictions and "presence" in targets:
            losses["existence"] = F.binary_cross_entropy_with_logits(
                predictions["existence_logits"].float(),
                targets["presence"].to(anchor.device).float())
        if "hazard_logits" in predictions and all(name in targets for name in (
                "hazard_target", "hazard_mask", "censor_mask")):
            losses["survival"] = self._survival_loss(
                predictions["hazard_logits"], targets)
        if "field_logits" in predictions and all(name in targets for name in (
                "field_target", "field_mask")):
            losses["field"] = self._spatial_cross_entropy(
                predictions["field_logits"], targets["field_target"],
                targets["field_mask"])
        if "hypothesis_boxes" in predictions and all(name in targets for name in (
                "target_box", "hypothesis_mask")):
            (losses["hypothesis"], localization,
             diversity) = self._hypothesis_loss(
                predictions["hypothesis_boxes"], targets)
            details["Loss/srbt_hypothesis_localization"] = localization.item()
            details["Loss/srbt_hypothesis_diversity"] = diversity.item()
        if all(name in predictions for name in (
                "identity_embeddings", "template_identity")):
            positive = self._positive_indices(predictions, targets)
            if positive is not None:
                identity_targets = dict(targets)
                identity_targets.setdefault("identity_positive_index", positive)
                identity_targets.setdefault(
                    "identity_mask", targets.get(
                        "hypothesis_mask", torch.ones(
                            predictions["identity_embeddings"].shape[0],
                            dtype=torch.bool, device=anchor.device)))
                losses["identity"] = self._identity_loss(
                    predictions, identity_targets)
        losses["calibration"] = self._calibration_loss(predictions, targets)

        distill_weight = self._distill_weight(progress)
        if teacher is not None:
            losses["distill"] = self._distillation_loss(predictions, teacher)
            if self.teacher_weight != 0.0:
                losses["teacher"] = self._teacher_supervision_loss(
                    teacher, targets)

        total = (
            self.existence_weight * losses["existence"]
            + self.survival_weight * losses["survival"]
            + self.field_weight * losses["field"]
            + self.hypothesis_weight * losses["hypothesis"]
            + self.identity_weight * losses["identity"]
            + self.calibration_weight * losses["calibration"]
            + distill_weight * losses["distill"]
            + self.teacher_weight * losses["teacher"]
        )
        stats = {
            **{f"Loss/srbt_{name}": value.detach().item()
               for name, value in losses.items()},
            **details,
            "Weight/srbt_distill": distill_weight,
            "Loss/srbt_total": total.detach().item(),
        }
        return total, stats

    def forward(self, predictions, targets, teacher=None, progress=0.0):
        return self.srbt_loss(predictions, targets, teacher, progress)
