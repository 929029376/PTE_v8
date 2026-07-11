"""
Occlusion-aware inference state machine (C3 controller).

Three states with hysteresis / counters to prevent flicker:
    TRACKING  ->  FROZEN   ->  REDETECT  ->  TRACKING
        |          |              |
        |          |              +-- (fail) -> FROZEN
        |          +-- (timeout) -> FROZEN
        +-- (normal) stays TRACKING

LEARNED POLICY WITH SAFETY SKELETON (the key change vs. the original hand-crafted
machine): the two most hand-tuned decisions are now driven by learned gates from
the MemoryPolicyHead, while the rule-based counters/timeouts are RETAINED as a
safety skeleton:

  TRACKING -> FROZEN:    mean(freeze_prob) over last w_abs frames > gate_thresh
                         (was: mean(absence) > theta_abs). freeze_prob is the
                         MemoryPolicyHead's gate, which refines absence with
                         hysteresis via the belief + absence_prob.
  FROZEN   -> REDETECT:  redetect_prob > gate_thresh   (was: z_rho > theta_z)
                         OR periodic forced attempt every T_period frames
                         (skeleton: catches slow reappearances the gate misses).
  REDETECT -> TRACKING:  redetect_conf > theta_re for w_re consecutive frames
                         (skeleton, unchanged).
  REDETECT -> FROZEN:    w_fail consecutive failures, OR T_max elapsed
                         (skeleton, unchanged).

The thresholds (gate_thresh / theta_re) and counters (w_abs / w_re / w_fail /
T_max / T_period / T_min_bg) are all retained — this is the "safety skeleton".
Only the two most brittle heuristics (theta_abs on raw absence, theta_z on a
z_rho spike) are replaced by learned gates. This gives the learned-policy
novelty without the instability risk of a fully-differentiable state machine.

Degradation guarantees ("does not crash"):
  * If history is not ready (min_history), the controller leaves the routed
    local tracker active and performs no occlusion-state transition.
  * Every state has a timeout, so it can never get stuck.
  * Ablation USE_LEARNED_POLICY=False reverts to the original theta_abs/theta_z
    thresholds (the `legacy_*` args), for an honest ablation comparison.
"""
from collections import deque
from enum import IntEnum


class State(IntEnum):
    TRACKING = 0
    FROZEN = 1
    REDETECT = 2


class OcclusionStateMachine:
    """Per-sequence occlusion controller.

    Args:
        gate_thresh: threshold for the learned freeze/redetect gates. A gate
            value above this triggers the corresponding transition.
        w_abs: consecutive frames the freeze signal must stay high before
            switching (hysteresis against one-frame glitches).
        theta_re: redetect confidence required to resume TRACKING.
        w_re: consecutive successful redetect frames required.
        w_fail: consecutive failed redetect frames that send it back to FROZEN.
        T_max: max frames allowed in REDETECT before forcing FROZEN.
        T_period: while FROZEN, attempt a cheap redetect every this many frames
            even without a gate trigger (catches slow reappearances).
        T_min_bg: minimum FROZEN frames before redetect is allowed, so the
            background field B has time to converge.

    Legacy (ablation USE_LEARNED_POLICY=False):
        legacy_theta_abs: absence threshold (old TRACKING->FROZEN rule).
        legacy_theta_z: z_rho threshold (old FROZEN->REDETECT rule). When the
            caller passes (absence_prob, z_rho) via the legacy kwargs, the
            machine uses these instead of the learned gates.
    """

    def __init__(self,
                 gate_thresh: float = 0.5,
                 w_abs: int = 3,
                 theta_re: float = 0.7,
                 w_re: int = 2,
                 w_fail: int = 5,
                 T_max: int = 50,
                 T_period: int = 10,
                 T_min_bg: int = 3,
                 use_learned_policy: bool = True,
                 legacy_theta_abs: float = 0.6,
                 legacy_theta_z: float = 2.5):
        self.gate_thresh = gate_thresh
        self.w_abs = w_abs
        self.theta_re = theta_re
        self.w_re = w_re
        self.w_fail = w_fail
        self.T_max = T_max
        self.T_period = T_period
        self.T_min_bg = T_min_bg
        self.use_learned_policy = use_learned_policy
        self.legacy_theta_abs = legacy_theta_abs
        self.legacy_theta_z = legacy_theta_z
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        """Call at the start of each sequence."""
        self.state = State.TRACKING
        self.freeze_window = deque(maxlen=self.w_abs)   # recent freeze probs
        self.re_success = 0      # consecutive successful redetect frames
        self.re_fail = 0         # consecutive failed redetect frames
        self.frames_in_re = 0    # frames spent in REDETECT (for T_max)
        self.frames_in_frozen = 0
        self.since_last_redetect = 0

    # ------------------------------------------------------------------ #
    def step(self, absence_prob, redetect_signal, redetect_conf=None,
             history_ready=True, freeze_prob=None):
        """Decide the next state and what action to take this frame.

        Args:
            absence_prob: float in [0,1] from AbsencePredictor. Used as the
                freeze signal in legacy mode, OR as input context (ignored for
                the threshold when freeze_prob is given in learned mode).
            redetect_signal: in LEARNED mode, float in [0,1] = the
                MemoryPolicyHead's redetect_prob. In LEGACY mode, float = z_rho
                (standardized ROI event-density delta), where a spike > theta_z
                triggers redetect.
            redetect_conf: float in [0,1] confidence of the redetection expert
                at the proposed location, or None if redetection was not run.
            history_ready: whether the belief history is trustworthy. If False,
                the machine behaves like the baseline (no occlusion logic).
            freeze_prob: optional float in [0,1] from the MemoryPolicyHead's
                freeze gate. If provided (learned mode), this drives
                TRACKING->FROZEN instead of absence_prob.

        Returns:
            dict with state, action (track/freeze/redetect/hold),
            enter_frozen, enter_redetect, exit_to_tracking.
        """
        if not history_ready:
            return self._fallback(absence_prob)

        # Choose the freeze signal: learned gate if available, else absence.
        if self.use_learned_policy and freeze_prob is not None:
            freeze_signal = freeze_prob
        else:
            freeze_signal = absence_prob
        self.freeze_window.append(freeze_signal)

        # Choose the redetect signal semantics.
        if self.use_learned_policy:
            redetect_trigger = redetect_signal > self.gate_thresh
        else:
            redetect_trigger = redetect_signal > self.legacy_theta_z

        action = "track"
        enter_frozen = False
        enter_redetect = False
        exit_to_tracking = False

        if self.state == State.TRACKING:
            mean_freeze = sum(self.freeze_window) / max(len(self.freeze_window), 1)
            thresh = self.gate_thresh if self.use_learned_policy else self.legacy_theta_abs
            if len(self.freeze_window) >= self.w_abs and mean_freeze > thresh:
                self.state = State.FROZEN
                self.frames_in_frozen = 0
                self.since_last_redetect = 0
                enter_frozen = True
                action = "freeze"
            else:
                action = "track"

        elif self.state == State.FROZEN:
            self.frames_in_frozen += 1
            self.since_last_redetect += 1
            periodic = (self.frames_in_frozen >= self.T_min_bg
                        and self.since_last_redetect >= self.T_period)
            if redetect_trigger or periodic:
                self.state = State.REDETECT
                self.frames_in_re = 0
                self.re_success = 0
                self.re_fail = 0
                enter_redetect = True
                action = "redetect"
            else:
                action = "hold"

        elif self.state == State.REDETECT:
            self.frames_in_re += 1
            if redetect_conf is not None and redetect_conf > self.theta_re:
                self.re_success += 1
                self.re_fail = 0
                if self.re_success >= self.w_re:
                    self.state = State.TRACKING
                    self.freeze_window.clear()
                    exit_to_tracking = True
                    action = "resume"
                else:
                    action = "redetect"
            else:
                self.re_fail += 1
                self.re_success = 0
                if self.re_fail >= self.w_fail or self.frames_in_re >= self.T_max:
                    self.state = State.FROZEN
                    self.since_last_redetect = 0
                    action = "freeze"
                else:
                    action = "redetect"

        return {
            "state": self.state,
            "action": action,
            "enter_frozen": enter_frozen,
            "enter_redetect": enter_redetect,
            "exit_to_tracking": exit_to_tracking,
        }

    # ------------------------------------------------------------------ #
    def _fallback(self, score_peak):
        """Keep local tracking active while event history is immature."""
        return {"state": self.state, "action": "track",
                "enter_frozen": False, "enter_redetect": False,
                "exit_to_tracking": False, "fallback": True}
