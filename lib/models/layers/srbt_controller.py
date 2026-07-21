import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F


def factorize_reliability(observability, localization_validity):
    """Compose candidate acceptance by the probability chain rule."""
    if not torch.is_tensor(observability) or not torch.is_tensor(
            localization_validity):
        raise TypeError("reliability factors must be tensors")
    if observability.shape != localization_validity.shape:
        raise ValueError("reliability factors must have identical shapes")
    for name, value in (
            ("observability", observability),
            ("localization_validity", localization_validity)):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        if torch.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"{name} must be in [0, 1]")
    return observability * localization_validity


class Action(str, Enum):
    TRACK = "track"
    LOCAL_UNRESOLVED = "local_unresolved"
    GLOBAL_UNRESOLVED = "global_unresolved"
    VERIFY = "verify"


@dataclass(frozen=True)
class ControllerAction:
    action: Action
    allow_recent_write: bool
    allow_long_write: bool
    output_absent: bool
    output_score: float


class VisibilityGate(nn.Module):
    """Predict visible/absent logits without updating shared representations."""

    def __init__(self, feature_dim, response_dim=3, hidden_dim=64):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.response_dim = int(response_dim)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim + self.response_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, pooled_feature, response_stats):
        if pooled_feature.ndim != 2:
            raise ValueError("pooled_feature must have shape (B, D)")
        if response_stats.ndim != 2:
            raise ValueError("response_stats must have shape (B, R)")
        if pooled_feature.shape != (pooled_feature.shape[0], self.feature_dim):
            raise ValueError(
                f"pooled_feature must have feature dimension {self.feature_dim}")
        if response_stats.shape != (pooled_feature.shape[0], self.response_dim):
            raise ValueError(
                f"response_stats must have shape (B, {self.response_dim})")
        if not torch.isfinite(pooled_feature).all():
            raise ValueError("pooled_feature must be finite")
        if not torch.isfinite(response_stats).all():
            raise ValueError("response_stats must be finite")
        inputs = torch.cat(
            (pooled_feature.detach(), response_stats.detach()), dim=-1)
        return self.network(inputs)


class LocalizationValidityGate(nn.Module):
    """Judge whether a candidate box is valid under one fixed observation."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, score_map, candidate_boxes):
        if score_map.ndim == 3:
            score_map = score_map.unsqueeze(1)
        if score_map.ndim != 4 or score_map.shape[1] != 1:
            raise ValueError("score_map must have shape (B,1,H,W)")
        if candidate_boxes.shape != (score_map.shape[0], 4):
            raise ValueError("candidate_boxes must have shape (B,4)")
        if not torch.isfinite(score_map).all():
            raise ValueError("score_map must be finite")
        if not torch.isfinite(candidate_boxes).all():
            raise ValueError("candidate_boxes must be finite")

        detached_map = score_map.detach()
        detached_boxes = candidate_boxes.detach()
        grid = detached_boxes[:, :2].mul(2.0).sub(1.0)[:, None, None]
        sampled = F.grid_sample(
            detached_map, grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        ).flatten(1)
        flat = detached_map.flatten(1)
        response_stats = torch.stack((
            flat.max(dim=-1).values,
            flat.mean(dim=-1),
            flat.std(dim=-1, unbiased=False),
        ), dim=-1)
        features = torch.cat((sampled, response_stats, detached_boxes), dim=-1)
        return self.network(features)


class DurationEvidenceDecoder(nn.Module):
    """Learn state evidence from factorized reliability and causal context."""

    def __init__(self, hidden_dim=16, duration_scale=32.0):
        super().__init__()
        self.duration_scale = float(duration_scale)
        if not math.isfinite(self.duration_scale) or self.duration_scale <= 0:
            raise ValueError("duration_scale must be finite and positive")
        self.network = nn.Sequential(
            nn.Linear(9, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), len(Action)),
        )

    @staticmethod
    def _vector(value, name, *, dtype=None, device=None):
        value = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        return value

    def forward(self, observability, localization, identity,
                previous_state, state_duration):
        reference = next(self.parameters())
        observability = self._vector(
            observability, "observability",
            dtype=reference.dtype, device=reference.device)
        localization = self._vector(
            localization, "localization",
            dtype=reference.dtype, device=reference.device)
        identity = self._vector(
            identity, "identity",
            dtype=reference.dtype, device=reference.device)
        state_duration = self._vector(
            state_duration, "state_duration",
            dtype=reference.dtype, device=reference.device)
        previous_state = self._vector(
            previous_state, "previous_state",
            dtype=torch.long, device=reference.device)
        batch = observability.numel()
        values = (localization, identity, state_duration, previous_state)
        if any(value.numel() != batch for value in values):
            raise ValueError("duration decoder inputs must share batch size")
        if torch.any((observability < 0.0) | (observability > 1.0)):
            raise ValueError("observability must be in [0, 1]")
        if torch.any((localization < 0.0) | (localization > 1.0)):
            raise ValueError("localization must be in [0, 1]")
        if torch.any((identity < -1.0) | (identity > 1.0)):
            raise ValueError("identity must be -1 or in [0, 1]")
        if torch.any((previous_state < 0) | (previous_state >= len(Action))):
            raise ValueError("previous_state is outside the DART state space")
        if torch.any(state_duration < 0.0):
            raise ValueError("state_duration must be non-negative")

        observability = observability.detach()
        localization = localization.detach()
        identity = identity.detach()
        previous_state = previous_state.detach()
        state_duration = state_duration.detach()
        acceptance = factorize_reliability(observability, localization)
        state_one_hot = F.one_hot(
            previous_state, num_classes=len(Action)).to(reference.dtype)
        duration_feature = torch.log1p(state_duration).div(
            math.log1p(self.duration_scale)).clamp(0.0, 1.0)
        features = torch.cat((
            observability[:, None],
            localization[:, None],
            acceptance[:, None],
            identity[:, None],
            state_one_hot,
            duration_feature[:, None],
        ), dim=-1)
        return self.network(features)


class DurationStructuredDecoder:
    """Strictly causal DART state decoder with explicit state durations."""

    def __init__(self, theta_observable=0.70, theta_localized=0.70,
                 theta_recover=0.75, local_duration=2, global_duration=4,
                 verify_duration=2, long_stable_duration=5, predictor=None):
        self.theta_observable = float(theta_observable)
        self.theta_localized = float(theta_localized)
        self.theta_recover = float(theta_recover)
        self.local_duration = int(local_duration)
        self.global_duration = int(global_duration)
        self.verify_duration = int(verify_duration)
        self.long_stable_duration = int(long_stable_duration)
        self.predictor = predictor
        for name, threshold in (
                ("theta_observable", self.theta_observable),
                ("theta_localized", self.theta_localized),
                ("theta_recover", self.theta_recover)):
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.local_duration < 1:
            raise ValueError("local_duration must be positive")
        if self.global_duration < self.local_duration:
            raise ValueError(
                "global_duration must be at least local_duration")
        if self.verify_duration < 1 or self.long_stable_duration < 1:
            raise ValueError("verification and stability lengths must be positive")
        self.reset()

    def reset(self):
        self.state = Action.TRACK
        self.state_duration = 0
        self._unresolved_duration = 0
        self._stable_visible = 0
        self.last_state_probabilities = None

    def _predict_action(self, observability, localization, identity):
        if self.predictor is None:
            return None
        identity_value = -1.0 if identity is None else identity
        previous_state = list(Action).index(self.state)
        with torch.no_grad():
            logits = self.predictor(
                [observability], [localization], [identity_value],
                [previous_state], [self.state_duration])
            probabilities = logits.softmax(dim=-1)[0]
        self.last_state_probabilities = probabilities.detach().cpu()
        return list(Action)[int(probabilities.argmax().item())]

    @staticmethod
    def _scalar(value, name):
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"{name} must contain one sequence value")
            value = value.detach().item()
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return value

    @classmethod
    def _optional_scalar(cls, value, name):
        return None if value is None else cls._scalar(value, name)

    @staticmethod
    def _result(action, score=0.0, allow_recent=False, allow_long=False):
        tracking = action is Action.TRACK
        allow_recent = tracking and bool(allow_recent)
        output_absent = action in (Action.GLOBAL_UNRESOLVED, Action.VERIFY)
        return ControllerAction(
            action=action,
            allow_recent_write=allow_recent,
            allow_long_write=allow_recent and bool(allow_long),
            output_absent=output_absent,
            output_score=0.0 if output_absent else float(score),
        )

    def _transition(self, state):
        if state is self.state:
            self.state_duration += 1
        else:
            self.state = state
            self.state_duration = 1

    def _step_recovery(self, observability, localization, identity):
        confirmed = (
            observability >= self.theta_recover
            and localization >= self.theta_recover
            and identity is not None
            and identity >= self.theta_recover
        )
        if not confirmed:
            self._transition(Action.GLOBAL_UNRESOLVED)
            return self._result(Action.GLOBAL_UNRESOLVED)

        self._transition(Action.VERIFY)
        if self.state_duration < self.verify_duration:
            return self._result(Action.VERIFY)

        self._transition(Action.TRACK)
        self._unresolved_duration = 0
        self._stable_visible = 0
        return self._result(
            Action.TRACK, score=observability * localization)

    def step(self, observability, localization, identity=None):
        observability = self._scalar(observability, "observability")
        localization = self._scalar(localization, "localization")
        identity = self._optional_scalar(identity, "identity")
        acceptance = observability * localization
        predicted_action = self._predict_action(
            observability, localization, identity)

        if self.state in (Action.GLOBAL_UNRESOLVED, Action.VERIFY):
            return self._step_recovery(
                observability, localization, identity)

        observable = observability >= self.theta_observable
        localized = localization >= self.theta_localized
        accepted = observable and localized
        if accepted and (
                predicted_action is None
                or predicted_action is Action.TRACK):
            self._transition(Action.TRACK)
            self._unresolved_duration = 0
            self._stable_visible += 1
            return self._result(
                Action.TRACK,
                score=acceptance,
                allow_recent=True,
                allow_long=(
                    self._stable_visible >= self.long_stable_duration),
            )

        self._stable_visible = 0
        self._unresolved_duration += 1
        if predicted_action is not None:
            global_allowed = (
                not observable
                and self._unresolved_duration >= self.global_duration)
            force_global = (
                not observable
                and self._unresolved_duration >= 2 * self.global_duration)
            if ((predicted_action is Action.GLOBAL_UNRESOLVED
                    and global_allowed) or force_global):
                self._transition(Action.GLOBAL_UNRESOLVED)
                return self._result(Action.GLOBAL_UNRESOLVED)
            self._transition(Action.LOCAL_UNRESOLVED)
            return self._result(Action.LOCAL_UNRESOLVED, score=acceptance)

        if (not observable
                and self._unresolved_duration >= self.global_duration):
            self._transition(Action.GLOBAL_UNRESOLVED)
            return self._result(Action.GLOBAL_UNRESOLVED)
        if self._unresolved_duration >= self.local_duration:
            self._transition(Action.LOCAL_UNRESOLVED)
            return self._result(Action.LOCAL_UNRESOLVED, score=acceptance)
        self._transition(Action.TRACK)
        return self._result(Action.TRACK, score=acceptance)


def build_duration_decoder(cfg, predictor=None):
    controller_cfg = getattr(getattr(cfg.MODEL, "SRBT", None), "CONTROLLER", None)
    if controller_cfg is None:
        return DurationStructuredDecoder(predictor=predictor)
    return DurationStructuredDecoder(
        theta_observable=getattr(controller_cfg, "THETA_OBSERVABLE", 0.70),
        theta_localized=getattr(controller_cfg, "THETA_LOCALIZED", 0.70),
        theta_recover=getattr(controller_cfg, "THETA_RECOVER", 0.75),
        local_duration=getattr(controller_cfg, "LOCAL_DURATION", 2),
        global_duration=getattr(controller_cfg, "GLOBAL_DURATION", 4),
        verify_duration=getattr(controller_cfg, "VERIFY_DURATION", 2),
        long_stable_duration=getattr(
            controller_cfg, "LONG_STABLE_DURATION", 5),
        predictor=predictor,
    )
