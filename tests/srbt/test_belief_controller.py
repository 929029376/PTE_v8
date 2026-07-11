from types import SimpleNamespace

import torch

from lib.models.layers.srbt_controller import (
    Action,
    BeliefController,
    build_belief_controller,
)
from lib.models.layers.state_machine import State
from lib.test.tracker.pet_track import PETTrack


def _posterior(present=0.8, absent=0.05, entropy=0.2,
               hazard8=0.0, duration=1.0):
    uncertain = max(0.0, 1.0 - present - absent)
    state_prob = torch.tensor([[present, uncertain, absent, 0.0]])
    hazard = torch.zeros(1, 129)
    hazard[0, :8] = hazard8 / 8.0
    hazard[0, -1] = 1.0 - hazard8
    return {
        "state_prob": state_prob,
        "hazard": hazard,
        "entropy": {"control": torch.tensor([entropy])},
        "duration": torch.tensor([duration]),
    }


def _candidate(identity=0.8, motion=0.8, score=0.85):
    return {
        "identity_score": identity,
        "motion_score": motion,
        "score": score,
    }


def test_track_requires_joint_presence_entropy_identity_and_motion_gate():
    controller = BeliefController()

    accepted = controller.step(_posterior(), _candidate(), frame_index=1)
    assert accepted.action is Action.TRACK
    assert accepted.allow_recent_write
    assert not accepted.output_absent
    assert accepted.output_score == 0.85

    cases = (
        (_posterior(present=0.69), _candidate()),
        (_posterior(entropy=0.36), _candidate()),
        (_posterior(), _candidate(identity=0.59)),
        (_posterior(), _candidate(motion=0.49)),
    )
    for frame_index, (posterior, candidate) in enumerate(cases, start=2):
        controller.reset()
        rejected = controller.step(posterior, candidate, frame_index)
        assert rejected.action is Action.HOLD
        assert not rejected.allow_recent_write
        assert not rejected.allow_long_write
        assert rejected.output_absent
        assert rejected.output_score == 0.0


def test_hazard_and_periodic_long_absence_trigger_redetection():
    immediate = BeliefController().step(
        _posterior(present=0.1, absent=0.7, hazard8=0.32, duration=2),
        _candidate(identity=0.1, motion=0.1),
        frame_index=7,
    )
    assert immediate.action is Action.REDETECT
    assert immediate.output_absent
    assert immediate.output_score == 0.0

    before_period = BeliefController().step(
        _posterior(present=0.1, absent=0.7, hazard8=0.1, duration=14),
        None,
        frame_index=20,
    )
    on_period = BeliefController().step(
        _posterior(present=0.1, absent=0.7, hazard8=0.1, duration=15),
        None,
        frame_index=21,
    )
    assert before_period.action is Action.HOLD
    assert on_period.action is Action.REDETECT


def test_redetection_requires_three_consecutive_verified_frames():
    controller = BeliefController()
    entered = controller.step(
        _posterior(present=0.1, absent=0.8, hazard8=0.4, duration=3),
        None,
        frame_index=1,
    )
    assert entered.action is Action.REDETECT

    for frame_index in (2, 3):
        verifying = controller.step(
            _posterior(present=0.8, absent=0.05),
            _candidate(),
            frame_index,
        )
        assert verifying.action is Action.REDETECT
        assert verifying.output_absent
        assert not verifying.allow_recent_write

    recovered = controller.step(
        _posterior(present=0.8, absent=0.05),
        _candidate(),
        frame_index=4,
    )
    assert recovered.action is Action.TRACK
    assert recovered.allow_recent_write
    assert not recovered.allow_long_write
    assert not recovered.output_absent


def test_failed_verification_resets_streak_and_long_memory_waits_after_recovery():
    controller = BeliefController()
    controller.step(
        _posterior(present=0.1, absent=0.8, hazard8=0.4, duration=3),
        None,
        frame_index=1,
    )
    controller.step(_posterior(), _candidate(), frame_index=2)
    failed = controller.step(
        _posterior(entropy=0.5), _candidate(), frame_index=3)
    assert failed.action is Action.REDETECT
    for frame_index in (4, 5):
        assert controller.step(
            _posterior(), _candidate(), frame_index).action is Action.REDETECT
    recovered = controller.step(_posterior(), _candidate(), frame_index=6)
    assert recovered.action is Action.TRACK
    assert not recovered.allow_long_write

    for frame_index in range(7, 11):
        stable = controller.step(_posterior(), _candidate(), frame_index)
        assert stable.action is Action.TRACK
        assert not stable.allow_long_write
    long_ready = controller.step(_posterior(), _candidate(), frame_index=11)
    assert long_ready.allow_long_write


def test_builder_reads_every_frozen_controller_threshold():
    values = SimpleNamespace(
        THETA_TRACK=0.71,
        THETA_ENTROPY=0.31,
        THETA_IDENTITY=0.61,
        THETA_MOTION=0.51,
        THETA_ABSENT=0.62,
        THETA_HAZARD8=0.32,
        REDETECT_AGE=6,
        REDETECT_PERIOD=11,
        VERIFY_FRAMES=4,
        LONG_STABLE_FRAMES=7,
    )
    cfg = SimpleNamespace(MODEL=SimpleNamespace(
        SRBT=SimpleNamespace(CONTROLLER=values)))

    controller = build_belief_controller(cfg)

    assert controller.theta_track == 0.71
    assert controller.theta_entropy == 0.31
    assert controller.theta_identity == 0.61
    assert controller.theta_motion == 0.51
    assert controller.theta_absent == 0.62
    assert controller.theta_hazard8 == 0.32
    assert controller.redetect_age == 6
    assert controller.redetect_period == 11
    assert controller.verify_frames == 4
    assert controller.long_stable_frames == 7


def test_tracker_uses_only_a_real_srbt_posterior_for_controller_input():
    class _Recorder:
        def __init__(self):
            self.args = None

        def step(self, *args):
            self.args = args
            return "controlled"

    tracker = object.__new__(PETTrack)
    tracker.frame_id = 9
    tracker.belief_controller = _Recorder()

    assert PETTrack._step_srbt_controller(tracker, {"score_peak": 0.9}) is None
    assert tracker.belief_controller.args is None

    posterior = _posterior()
    hypothesis = _candidate()
    result = PETTrack._step_srbt_controller(tracker, {
        "srbt_posterior": posterior,
        "srbt_best_hypothesis": hypothesis,
    })
    assert result == "controlled"
    assert tracker.belief_controller.args == (posterior, hypothesis, 9)


def test_verified_srbt_recovery_consumes_the_pending_global_box():
    tracker = object.__new__(PETTrack)
    tracker._pending_redetect_box = [90.0, 40.0, 20.0, 10.0]
    tracker._redetect_hypotheses = {"active_count": 1}
    tracker._last_redetect_conf = 0.91

    recovered = PETTrack._resolve_tracking_state(
        tracker,
        local_state=[5.0, 5.0, 10.0, 10.0],
        height=120,
        width=160,
        srbt_recovered=True,
    )

    assert recovered == [90.0, 40.0, 20.0, 10.0]
    assert tracker._pending_redetect_box is None
    assert tracker._redetect_hypotheses is None
    assert tracker._last_redetect_conf == 0.0


def test_srbt_hold_and_redetect_never_update_event_background():
    assert not PETTrack._should_update_event_background(
        srbt_control=object(),
        use_train_compatible_policy=False,
        state_machine_state=State.FROZEN,
    )
    assert PETTrack._should_update_event_background(
        srbt_control=None,
        use_train_compatible_policy=False,
        state_machine_state=State.FROZEN,
    )
