import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn


class Action(str, Enum):
    TRACK = "track"
    SUSPECT = "suspect"
    ABSENT = "absent"
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


class VisibilityController:
    """Causal visibility state machine for tracking and recovery activation."""

    def __init__(self, theta_present=0.70, theta_recover=0.75,
                 suspect_frames=2, absent_frames=4, verify_frames=2,
                 long_stable_frames=5):
        self.theta_present = float(theta_present)
        self.theta_recover = float(theta_recover)
        self.suspect_frames = int(suspect_frames)
        self.absent_frames = int(absent_frames)
        self.verify_frames = int(verify_frames)
        self.long_stable_frames = int(long_stable_frames)
        for name, threshold in (
                ("theta_present", self.theta_present),
                ("theta_recover", self.theta_recover)):
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.suspect_frames < 1:
            raise ValueError("suspect_frames must be positive")
        if self.absent_frames <= self.suspect_frames:
            raise ValueError("absent_frames must be greater than suspect_frames")
        if self.verify_frames < 1 or self.long_stable_frames < 1:
            raise ValueError("verification and stability lengths must be positive")
        self.reset()

    def reset(self):
        self.state = Action.TRACK
        self._weak_streak = 0
        self._verify_streak = 0
        self._stable_visible = 0

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
        output_absent = action in (Action.ABSENT, Action.VERIFY)
        return ControllerAction(
            action=action,
            allow_recent_write=allow_recent,
            allow_long_write=allow_recent and bool(allow_long),
            output_absent=output_absent,
            output_score=0.0 if output_absent else float(score),
        )

    def _step_recovery(self, present, identity, localization):
        confirmed = (
            present >= self.theta_recover
            and identity is not None
            and localization is not None
            and identity >= self.theta_recover
            and localization >= self.theta_recover
        )
        if not confirmed:
            self.state = Action.ABSENT
            self._verify_streak = 0
            return self._result(Action.ABSENT)

        self._verify_streak += 1
        if self._verify_streak < self.verify_frames:
            self.state = Action.VERIFY
            return self._result(Action.VERIFY)

        self.state = Action.TRACK
        self._weak_streak = 0
        self._verify_streak = 0
        self._stable_visible = 0
        return self._result(Action.TRACK, score=present)

    def step(self, present_probability, identity_score=None,
             localization_score=None):
        present = self._scalar(present_probability, "present_probability")
        identity = self._optional_scalar(identity_score, "identity_score")
        localization = self._optional_scalar(
            localization_score, "localization_score")

        if self.state in (Action.ABSENT, Action.VERIFY):
            return self._step_recovery(present, identity, localization)

        if present >= self.theta_present:
            self.state = Action.TRACK
            self._weak_streak = 0
            self._verify_streak = 0
            self._stable_visible += 1
            return self._result(
                Action.TRACK,
                score=present,
                allow_recent=True,
                allow_long=self._stable_visible >= self.long_stable_frames,
            )

        self._stable_visible = 0
        self._weak_streak += 1
        if self._weak_streak >= self.absent_frames:
            self.state = Action.ABSENT
            return self._result(Action.ABSENT)
        if self._weak_streak >= self.suspect_frames:
            self.state = Action.SUSPECT
            return self._result(Action.SUSPECT, score=present)
        self.state = Action.TRACK
        return self._result(Action.TRACK, score=present)


def build_visibility_controller(cfg):
    controller_cfg = getattr(getattr(cfg.MODEL, "SRBT", None), "CONTROLLER", None)
    if controller_cfg is None:
        return VisibilityController()
    return VisibilityController(
        theta_present=getattr(controller_cfg, "THETA_PRESENT", 0.70),
        theta_recover=getattr(controller_cfg, "THETA_RECOVER", 0.75),
        suspect_frames=getattr(controller_cfg, "SUSPECT_FRAMES", 2),
        absent_frames=getattr(controller_cfg, "ABSENT_FRAMES", 4),
        verify_frames=getattr(controller_cfg, "VERIFY_FRAMES", 2),
        long_stable_frames=getattr(controller_cfg, "LONG_STABLE_FRAMES", 5),
    )
