import math
from dataclasses import dataclass
from enum import Enum

import torch

from lib.models.layers.srbt_belief import ABSENT, REAPPEARING, VISIBLE


class Action(str, Enum):
    TRACK = "track"
    HOLD = "hold"
    REDETECT = "redetect"


@dataclass(frozen=True)
class ControllerAction:
    action: Action
    allow_recent_write: bool
    allow_long_write: bool
    output_absent: bool
    output_score: float


class BeliefController:
    """Convert one causal SRBT posterior into tracking and memory actions."""

    def __init__(self, theta_track=0.70, theta_entropy=0.35,
                 theta_identity=0.60, theta_motion=0.50,
                 theta_absent=0.60, theta_hazard8=0.30,
                 redetect_age=5, redetect_period=10,
                 verify_frames=3, long_stable_frames=5):
        self.theta_track = float(theta_track)
        self.theta_entropy = float(theta_entropy)
        self.theta_identity = float(theta_identity)
        self.theta_motion = float(theta_motion)
        self.theta_absent = float(theta_absent)
        self.theta_hazard8 = float(theta_hazard8)
        self.redetect_age = int(redetect_age)
        self.redetect_period = int(redetect_period)
        self.verify_frames = int(verify_frames)
        self.long_stable_frames = int(long_stable_frames)
        if self.redetect_age < 1 or self.redetect_period < 1:
            raise ValueError("redetect ages must be positive")
        if self.verify_frames < 1 or self.long_stable_frames < 1:
            raise ValueError("verification and stability lengths must be positive")
        self.reset()

    def reset(self):
        self.state = Action.TRACK
        self._verify_streak = 0
        self._stable_visible = 0
        self._recovery_stable = None

    @staticmethod
    def _scalar(value, name):
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"{name} must contain one sequence value")
            value = value.detach().item()
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _candidate_value(cls, candidate, names, default=0.0):
        if candidate is None:
            return float(default)
        for name in names:
            if name in candidate:
                return cls._scalar(candidate[name], name)
        return float(default)

    def _read_inputs(self, posterior, best_hypothesis):
        state_prob = posterior["state_prob"]
        if state_prob.shape != (1, 4):
            raise ValueError("state_prob must have shape (1, 4)")
        hazard = posterior["hazard"]
        if hazard.ndim != 2 or hazard.shape[0] != 1 or hazard.shape[1] < 9:
            raise ValueError("hazard must have shape (1, >=9)")
        present = self._scalar(
            state_prob[0, VISIBLE] + state_prob[0, REAPPEARING], "present")
        absent = self._scalar(state_prob[0, ABSENT], "absent")
        entropy = self._scalar(posterior["entropy"]["control"], "control entropy")
        hazard8 = self._scalar(hazard[0, :8].sum(), "hazard8")
        duration = self._scalar(posterior["duration"], "duration")
        identity = self._candidate_value(
            best_hypothesis, ("identity_score", "identity_consistency"))
        motion = self._candidate_value(
            best_hypothesis, ("motion_score", "motion_consistency"))
        score = self._candidate_value(
            best_hypothesis, ("score", "confidence"), default=present)
        return present, absent, entropy, hazard8, duration, identity, motion, score

    def _result(self, action, score=0.0, allow_long=False):
        track = action is Action.TRACK
        return ControllerAction(
            action=action,
            allow_recent_write=track,
            allow_long_write=track and bool(allow_long),
            output_absent=not track,
            output_score=float(score) if track else 0.0,
        )

    def step(self, posterior, best_hypothesis, frame_index):
        if int(frame_index) < 0:
            raise ValueError("frame_index must be nonnegative")
        (present, absent, entropy, hazard8, duration,
         identity, motion, score) = self._read_inputs(
            posterior, best_hypothesis)
        trusted = (
            present >= self.theta_track
            and entropy <= self.theta_entropy
            and identity >= self.theta_identity
            and motion >= self.theta_motion
        )

        if self.state is Action.REDETECT:
            self._verify_streak = self._verify_streak + 1 if trusted else 0
            if self._verify_streak < self.verify_frames:
                return self._result(Action.REDETECT)
            self.state = Action.TRACK
            self._verify_streak = 0
            self._stable_visible = 1
            self._recovery_stable = 0
            return self._result(Action.TRACK, score=score, allow_long=False)

        if trusted:
            self.state = Action.TRACK
            self._stable_visible += 1
            if self._recovery_stable is not None:
                self._recovery_stable += 1
                recovery_ready = self._recovery_stable >= self.long_stable_frames
                if recovery_ready:
                    self._recovery_stable = None
            else:
                recovery_ready = True
            allow_long = (
                self._stable_visible >= self.long_stable_frames
                and recovery_ready
            )
            return self._result(Action.TRACK, score=score, allow_long=allow_long)

        self._verify_streak = 0
        self._stable_visible = 0
        immediate = absent >= self.theta_absent and hazard8 >= self.theta_hazard8
        duration_frame = max(0, int(round(duration)))
        periodic = (
            absent >= self.theta_absent
            and duration_frame >= self.redetect_age
            and (duration_frame - self.redetect_age) % self.redetect_period == 0
        )
        if immediate or periodic:
            self.state = Action.REDETECT
            return self._result(Action.REDETECT)
        self.state = Action.HOLD
        return self._result(Action.HOLD)


def build_belief_controller(cfg):
    controller_cfg = getattr(getattr(cfg.MODEL, "SRBT", None), "CONTROLLER", None)
    if controller_cfg is None:
        return BeliefController()
    return BeliefController(
        theta_track=getattr(controller_cfg, "THETA_TRACK", 0.70),
        theta_entropy=getattr(controller_cfg, "THETA_ENTROPY", 0.35),
        theta_identity=getattr(controller_cfg, "THETA_IDENTITY", 0.60),
        theta_motion=getattr(controller_cfg, "THETA_MOTION", 0.50),
        theta_absent=getattr(controller_cfg, "THETA_ABSENT", 0.60),
        theta_hazard8=getattr(controller_cfg, "THETA_HAZARD8", 0.30),
        redetect_age=getattr(controller_cfg, "REDETECT_AGE", 5),
        redetect_period=getattr(controller_cfg, "REDETECT_PERIOD", 10),
        verify_frames=getattr(controller_cfg, "VERIFY_FRAMES", 3),
        long_stable_frames=getattr(controller_cfg, "LONG_STABLE_FRAMES", 5),
    )
