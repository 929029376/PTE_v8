from types import SimpleNamespace

import torch

from lib.models.layers.srbt_controller import (
    Action,
    VisibilityController,
    VisibilityGate,
    build_visibility_controller,
)
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


def test_visibility_gate_detaches_shared_features_and_returns_binary_logits():
    gate = VisibilityGate(feature_dim=8, response_dim=3, hidden_dim=4)
    pooled = torch.randn(2, 8, requires_grad=True)
    response_stats = torch.randn(2, 3, requires_grad=True)

    logits = gate(pooled, response_stats)
    logits.sum().backward()

    assert logits.shape == (2, 2)
    assert pooled.grad is None
    assert response_stats.grad is None
    assert all(parameter.grad is not None for parameter in gate.parameters())


def test_weak_presence_progresses_from_track_to_suspect_to_absent():
    controller = VisibilityController(
        suspect_frames=2, absent_frames=4, long_stable_frames=3)

    first = controller.step(0.69)
    second = controller.step(0.69)
    third = controller.step(0.69)
    fourth = controller.step(0.69)

    assert first.action is Action.TRACK
    assert not first.allow_recent_write
    assert not first.output_absent
    assert second.action is Action.SUSPECT
    assert third.action is Action.SUSPECT
    assert not second.output_absent
    assert fourth.action is Action.ABSENT
    assert fourth.output_absent


def test_strong_presence_resets_suspicion_and_preserves_memory_stability_gate():
    controller = VisibilityController(
        suspect_frames=2, absent_frames=4, long_stable_frames=3)
    controller.step(0.2)
    assert controller.step(0.2).action is Action.SUSPECT

    recovered = controller.step(0.9)
    stable = controller.step(0.9)
    long_ready = controller.step(0.9)

    assert recovered.action is Action.TRACK
    assert recovered.allow_recent_write
    assert not recovered.allow_long_write
    assert not stable.allow_long_write
    assert long_ready.allow_long_write


def test_absent_recovery_requires_two_rgb_localization_confirmations():
    controller = VisibilityController(
        suspect_frames=2, absent_frames=4, verify_frames=2)
    for _ in range(4):
        result = controller.step(0.1)
    assert result.action is Action.ABSENT

    verifying = controller.step(
        0.9, identity_score=0.85, localization_score=0.8)
    recovered = controller.step(
        0.88, identity_score=0.82, localization_score=0.79)

    assert verifying.action is Action.VERIFY
    assert verifying.output_absent
    assert not verifying.allow_recent_write
    assert recovered.action is Action.TRACK
    assert not recovered.output_absent
    assert not recovered.allow_recent_write


def test_failed_recovery_confirmation_returns_to_absent():
    controller = VisibilityController()
    for _ in range(4):
        controller.step(0.1)
    assert controller.step(
        0.9, identity_score=0.9, localization_score=0.9
    ).action is Action.VERIFY

    failed = controller.step(
        0.9, identity_score=0.4, localization_score=0.9)

    assert failed.action is Action.ABSENT
    assert failed.output_absent


def test_builder_reads_visibility_controller_thresholds():
    values = SimpleNamespace(
        THETA_PRESENT=0.71,
        THETA_RECOVER=0.81,
        SUSPECT_FRAMES=3,
        ABSENT_FRAMES=6,
        VERIFY_FRAMES=4,
        LONG_STABLE_FRAMES=7,
    )
    cfg = SimpleNamespace(MODEL=SimpleNamespace(
        SRBT=SimpleNamespace(CONTROLLER=values)))

    controller = build_visibility_controller(cfg)

    assert controller.theta_present == 0.71
    assert controller.theta_recover == 0.81
    assert controller.suspect_frames == 3
    assert controller.absent_frames == 6
    assert controller.verify_frames == 4
    assert controller.long_stable_frames == 7


def test_tracker_passes_only_local_presence_to_controller():
    class _Recorder:
        def __init__(self):
            self.args = None

        def step(self, *args):
            self.args = args
            return "controlled"

    tracker = object.__new__(PETTrack)
    tracker.frame_id = 9
    tracker.visibility_controller = _Recorder()

    assert PETTrack._step_srbt_controller(tracker, {"score_peak": 0.9}) is None
    assert tracker.visibility_controller.args is None

    result = PETTrack._step_srbt_controller(tracker, {
        "presence_score": 0.82,
    })
    assert result == "controlled"
    assert tracker.visibility_controller.args == (0.82,)


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


def test_tracker_has_no_event_background_fallback():
    assert not hasattr(PETTrack, "_should_update_event_background")


def test_new_sequence_resets_runtime_recovery_state():
    tracker = object.__new__(PETTrack)
    tracker._srbt_last_action = Action.ABSENT
    tracker._redetect_hypotheses = {"active_count": 1}
    tracker._pending_redetect_box = [1.0, 2.0, 3.0, 4.0]

    PETTrack._reset_srbt_sequence_state(tracker)

    assert tracker._srbt_last_action is Action.TRACK
    assert tracker._redetect_hypotheses is None
    assert tracker._pending_redetect_box is None
    assert not hasattr(tracker, "_expert_probabilities")


def test_tracker_uses_all_experts_without_hidden_selector_history():
    class Thor:
        def begin_frame(self):
            return torch.zeros(1, 4, 8), torch.zeros(1, 4, 8)

    class Network:
        def __init__(self):
            self.calls = []

        def inference(self, **kwargs):
            self.calls.append(kwargs)
            expert_outputs = {
                name: {
                    "score_map": torch.full((1, 1, 2, 2), score),
                    "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
                }
                for name, score in zip(
                    ("generalist", "motion_fm", "small_target_st",
                     "visibility_foc_ov", "discrimination_bi"),
                    (0.5, 0.6, 0.7, 0.8, 0.9),
                )
            }
            return {
                "score_map": torch.ones(1, 1, 2, 2),
                "target_bbox": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "presence_score": torch.tensor([0.8]),
                "expert_outputs": expert_outputs,
            }

    tracker = object.__new__(PETTrack)
    tracker.thor_wrapper = Thor()
    tracker.network = Network()
    tracker.static_zi = torch.zeros(1, 4, 8)
    tracker.static_ze = torch.zeros(1, 4, 8)
    tracker.output_window = torch.ones(1, 1, 2, 2)
    tracker.params = SimpleNamespace(search_size=32)
    tracker.map_box_back = lambda box, factor, reference: box
    search = torch.zeros(1, 3, 32, 32)

    candidate = tracker._run_local_candidate(
        search, search, 1.0, 100, 100,
        reference_state=[0.0, 0.0, 16.0, 16.0])

    assert len(tracker.network.calls) == 1
    assert "previous_expert_probabilities" not in tracker.network.calls[0]
    assert candidate["retained_expert_ids"] == (0, 1, 2, 3, 4)
    assert candidate["ensemble_weights"].shape == (5,)
    assert len(candidate["expert_states"]) == 5
    assert all(len(box) == 4 for box in candidate["expert_states"])
