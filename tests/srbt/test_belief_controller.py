from types import SimpleNamespace

import pytest
import torch

from lib.models.layers.srbt_controller import (
    Action,
    DurationEvidenceDecoder,
    DurationStructuredDecoder,
    LocalizationValidityGate,
    VisibilityGate,
    build_duration_decoder,
    factorize_reliability,
)
from lib.test.tracker.pet_track import PETTrack


def test_reliability_factorization_is_chain_rule_product_and_keeps_gradients():
    observability = torch.tensor([0.8, 0.25], requires_grad=True)
    localization = torch.tensor([0.5, 0.4], requires_grad=True)

    acceptance = factorize_reliability(observability, localization)

    assert torch.allclose(acceptance, torch.tensor([0.4, 0.1]))
    acceptance.sum().backward()
    assert torch.allclose(observability.grad, localization.detach())
    assert torch.allclose(localization.grad, observability.detach())


def test_localization_validity_is_candidate_conditioned_and_detaches_tracker_outputs():
    gate = LocalizationValidityGate(hidden_dim=1)
    with torch.no_grad():
        gate.network[0].weight.zero_()
        gate.network[0].bias.zero_()
        gate.network[0].weight[0, 0] = 5.0
        gate.network[2].weight.zero_()
        gate.network[2].bias.zero_()
        gate.network[2].weight[1, 0] = 5.0
    score_map = torch.zeros(2, 1, 4, 4, requires_grad=True)
    score_map.data[:, :, 3, 3] = 1.0
    candidates = torch.tensor([
        [0.875, 0.875, 0.1, 0.1],
        [0.125, 0.125, 0.1, 0.1],
    ], requires_grad=True)

    logits = gate(score_map, candidates)
    validity = logits.softmax(dim=-1)[:, 1]
    validity.sum().backward()

    assert logits.shape == (2, 2)
    assert validity[0] > validity[1]
    assert score_map.grad is None
    assert candidates.grad is None
    assert all(parameter.grad is not None for parameter in gate.parameters())


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


def test_duration_decoder_exposes_only_dart_control_states():
    assert set(Action) == {
        Action.TRACK,
        Action.LOCAL_UNRESOLVED,
        Action.GLOBAL_UNRESOLVED,
        Action.VERIFY,
    }


def test_duration_evidence_decoder_is_lightweight_and_trainable():
    decoder = DurationEvidenceDecoder(hidden_dim=16)
    observability = torch.tensor([0.9, 0.2], requires_grad=True)
    localization = torch.tensor([0.8, 0.7], requires_grad=True)
    identity = torch.tensor([-1.0, 0.9], requires_grad=True)
    previous_state = torch.tensor([0, 2])
    state_duration = torch.tensor([1.0, 5.0])

    logits = decoder(
        observability, localization, identity,
        previous_state, state_duration)
    logits.sum().backward()

    assert logits.shape == (2, len(Action))
    assert sum(parameter.numel() for parameter in decoder.parameters()) <= 256
    assert all(parameter.grad is not None for parameter in decoder.parameters())
    assert observability.grad is None
    assert localization.grad is None
    assert identity.grad is None


def test_duration_decoder_separates_localization_failure_from_unobservability():
    decoder = DurationStructuredDecoder(
        local_duration=2, global_duration=3, verify_duration=2)

    first_local = decoder.step(observability=0.9, localization=0.2)
    local = decoder.step(observability=0.9, localization=0.2)

    assert first_local.action is Action.TRACK
    assert local.action is Action.LOCAL_UNRESOLVED
    assert not local.output_absent
    assert not local.allow_recent_write

    decoder.reset()
    decoder.step(observability=0.2, localization=0.9)
    decoder.step(observability=0.2, localization=0.9)
    global_unresolved = decoder.step(observability=0.2, localization=0.9)

    assert global_unresolved.action is Action.GLOBAL_UNRESOLVED
    assert global_unresolved.output_absent


def test_learned_duration_decoder_can_reject_an_observable_local_candidate():
    class _PreferLocal(torch.nn.Module):
        def forward(self, *_args):
            return torch.tensor([[0.0, 8.0, -8.0, -8.0]])

    decoder = DurationStructuredDecoder(
        local_duration=2,
        global_duration=4,
        predictor=_PreferLocal(),
    )

    decision = decoder.step(observability=0.95, localization=0.95)

    assert decision.action is Action.LOCAL_UNRESOLVED
    assert not decision.allow_recent_write


def test_duration_decoder_requires_verified_recovery_before_commitment():
    decoder = DurationStructuredDecoder(
        local_duration=1, global_duration=2, verify_duration=2)
    decoder.step(observability=0.1, localization=0.1)
    assert decoder.step(
        observability=0.1, localization=0.1
    ).action is Action.GLOBAL_UNRESOLVED

    rejected = decoder.step(
        observability=0.9, localization=0.9, identity=0.4)
    verifying = decoder.step(
        observability=0.9, localization=0.9, identity=0.9)
    committed = decoder.step(
        observability=0.9, localization=0.9, identity=0.9)

    assert rejected.action is Action.GLOBAL_UNRESOLVED
    assert verifying.action is Action.VERIFY
    assert verifying.output_absent
    assert committed.action is Action.TRACK
    assert not committed.output_absent
    assert not committed.allow_recent_write


def test_duration_decoder_writes_memory_only_after_stable_tracking():
    decoder = DurationStructuredDecoder(long_stable_duration=3)

    first = decoder.step(observability=0.9, localization=0.9)
    second = decoder.step(observability=0.9, localization=0.9)
    third = decoder.step(observability=0.9, localization=0.9)

    assert first.allow_recent_write
    assert second.allow_recent_write
    assert not second.allow_long_write
    assert third.allow_long_write


def test_builder_reads_duration_decoder_thresholds():
    values = SimpleNamespace(
        THETA_OBSERVABLE=0.71,
        THETA_LOCALIZED=0.72,
        THETA_RECOVER=0.81,
        LOCAL_DURATION=3,
        GLOBAL_DURATION=6,
        VERIFY_DURATION=4,
        LONG_STABLE_DURATION=7,
    )
    cfg = SimpleNamespace(MODEL=SimpleNamespace(
        SRBT=SimpleNamespace(CONTROLLER=values)))

    decoder = build_duration_decoder(cfg)

    assert decoder.theta_observable == 0.71
    assert decoder.theta_localized == 0.72
    assert decoder.theta_recover == 0.81
    assert decoder.local_duration == 3
    assert decoder.global_duration == 6
    assert decoder.verify_duration == 4
    assert decoder.long_stable_duration == 7


def test_tracker_passes_factorized_reliability_to_duration_decoder():
    class _Recorder:
        def __init__(self):
            self.args = None

        def step(self, *args):
            self.args = args
            return "controlled"

    tracker = object.__new__(PETTrack)
    tracker.frame_id = 9
    tracker.duration_decoder = _Recorder()

    assert PETTrack._step_srbt_controller(tracker, {"score_peak": 0.9}) is None
    assert tracker.duration_decoder.args is None

    result = PETTrack._step_srbt_controller(tracker, {
        "presence_score": 0.82,
        "observability_score": 0.82,
        "localization_validity_score": 0.5,
        "acceptance_score": 0.41,
    })
    assert result == "controlled"
    assert tracker.duration_decoder.args == (0.82, 0.5, None)


def test_recovery_confirmation_ranks_by_localization_and_preserves_factors():
    recovery = {
        "accepted": torch.tensor([[True, True]]),
        "identity_scores": torch.tensor([[0.95, 0.80]]),
        "observability_scores": torch.tensor([[0.90, 0.85]]),
        "localization_validity_scores": torch.tensor([[0.55, 0.90]]),
        "acceptance_scores": torch.tensor([[0.495, 0.765]]),
    }

    confirmation = PETTrack._best_recovery_confirmation(recovery)

    assert confirmation == pytest.approx({
        "identity_score": 0.8,
        "observability_score": 0.85,
        "localization_validity_score": 0.9,
        "acceptance_score": 0.765,
    })


def test_tracker_bypasses_untrained_srbt_gate_when_disabled():
    class _UnexpectedController:
        def step(self, *args):
            raise AssertionError("disabled SRBT must not run the visibility controller")

    tracker = object.__new__(PETTrack)
    tracker.srbt_controller_enabled = False
    tracker.duration_decoder = _UnexpectedController()

    result = PETTrack._step_srbt_controller(tracker, {
        "observability_score": 0.9,
        "localization_validity_score": 0.5,
        "acceptance_score": 0.45,
    })

    assert result.action is Action.TRACK
    assert result.output_score == 0.45
    assert not result.output_absent
    assert not result.allow_recent_write
    assert not result.allow_long_write


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
    tracker._srbt_last_action = Action.GLOBAL_UNRESOLVED
    tracker._redetect_hypotheses = {"active_count": 1}
    tracker._pending_redetect_box = [1.0, 2.0, 3.0, 4.0]

    PETTrack._reset_srbt_sequence_state(tracker)

    assert tracker._srbt_last_action is Action.TRACK
    assert tracker._redetect_hypotheses is None
    assert tracker._pending_redetect_box is None
    assert not hasattr(tracker, "_expert_probabilities")


def test_tracker_motion_context_is_causal_and_resets_between_sequences():
    tracker = object.__new__(PETTrack)
    tracker.state = [10.0, 20.0, 4.0, 8.0]
    PETTrack._reset_srbt_sequence_state(tracker)
    first_event = torch.ones(1, 3, 16, 16, requires_grad=True)

    first = PETTrack._build_motion_context(
        tracker,
        first_event,
        reference_state=[10.0, 20.0, 4.0, 8.0],
    )

    assert not bool(first["history_valid"].any())
    assert first["current_event"] is first_event
    assert first["previous_event"] is first_event
    assert torch.equal(first["box_delta"], torch.zeros(1, 4))

    PETTrack._commit_motion_event_search(tracker, first_event)
    first_event.detach().zero_()
    second_event = torch.full((1, 3, 16, 16), 2.0)
    second = PETTrack._build_motion_context(
        tracker,
        second_event,
        reference_state=[14.0, 22.0, 4.0, 8.0],
    )

    assert bool(second["history_valid"].all())
    assert second["current_event"] is second_event
    assert torch.equal(second["previous_event"], torch.ones_like(second_event))
    assert second["box_delta"][0].tolist() == [1.0, 0.25, 0.0, 0.0]
    assert not second["previous_event"].requires_grad

    PETTrack._reset_srbt_sequence_state(tracker)
    assert tracker._motion_previous_event_search is None


def test_track_commits_final_refined_motion_event_once(monkeypatch):
    class _Preprocessor:
        @staticmethod
        def process(image, _mask):
            return SimpleNamespace(tensors=image)

    class _Thor:
        def __init__(self):
            self.resume_count = 0

        def resume(self):
            self.resume_count += 1

        @staticmethod
        def commit(*_args, **_kwargs):
            raise AssertionError("closed memory frame must not commit")

    tracker = object.__new__(PETTrack)
    tracker.frame_id = 0
    tracker.state = [10.0, 10.0, 8.0, 8.0]
    tracker.params = SimpleNamespace(
        search_factor=4.0,
        search_size=32,
        template_factor=2.0,
        template_size=16,
    )
    tracker.preprocessor = _Preprocessor()
    tracker.thor_wrapper = _Thor()
    tracker._srbt_last_action = Action.GLOBAL_UNRESOLVED
    tracker._pending_redetect_box = [12.0, 12.0, 8.0, 8.0]
    tracker._last_redetect_error = ""
    tracker._recovery_diagnostics = []
    tracker._expert_diagnostics = []
    tracker._search_diagnostics = []
    tracker.expert_names = (
        "generalist", "motion_fm", "precision_refiner",
        "visibility_foc_ov", "discrimination_bi",
    )
    tracker.debug = False
    tracker.use_visdom = False

    initial_event = torch.ones(1, 3, 32, 32)
    refined_event = torch.full((1, 3, 32, 32), 2.0)
    template_event = torch.full((1, 3, 16, 16), 3.0)
    sample_calls = iter((
        (torch.zeros_like(initial_event), initial_event, 1.0, None),
        (torch.zeros_like(template_event), template_event, 1.0, None),
    ))
    monkeypatch.setattr(
        "lib.test.tracker.pet_track.sample_target",
        lambda **_kwargs: next(sample_calls),
    )

    def candidate(event_tensor, state):
        return {
            "state": list(state),
            "score_peak": 0.8,
            "response": torch.ones(1, 1, 2, 2),
            "presence_score": torch.tensor([0.8]),
            "memory_frame_open": False,
            "expert_states": [list(state) for _ in tracker.expert_names],
            "motion_event_search": event_tensor,
        }

    initial_candidate = candidate(initial_event, [11.0, 11.0, 8.0, 8.0])
    refined_candidate = candidate(refined_event, [13.0, 13.0, 8.0, 8.0])
    tracker._search_state_for_frame = lambda: list(tracker.state)
    tracker._run_local_candidate = lambda *_args, **_kwargs: initial_candidate
    tracker._run_recovery_cycle = lambda *_args: {}
    tracker._best_recovery_confirmation = lambda _recovery: {
        "identity_score": 0.9,
        "observability_score": 0.9,
        "localization_validity_score": 0.9,
        "acceptance_score": 0.81,
    }
    tracker._step_srbt_controller = lambda *_args, **_kwargs: SimpleNamespace(
        action=Action.TRACK,
        output_score=0.9,
        allow_recent_write=False,
        allow_long_write=False,
    )
    tracker._refine_pending_recovery = lambda *_args: refined_candidate
    tracker._resolve_tracking_state = (
        lambda local_state, *_args, **_kwargs: list(local_state))
    tracker._plan_next_search_state = lambda *_args: None
    tracker._record_search_diagnostic = lambda *_args, **_kwargs: None
    tracker._recovery_diagnostic_value = lambda *_args, **_kwargs: 0.0
    committed = []
    tracker._commit_motion_event_search = committed.append

    result = PETTrack.track(
        tracker,
        torch.zeros(40, 40, 3).numpy(),
        torch.zeros(40, 40, 3).numpy(),
    )

    assert len(committed) == 1
    assert committed[0] is refined_event
    assert result["target_bbox"] == refined_candidate["state"]
    assert tracker.thor_wrapper.resume_count == 1

    trusted_state = [20.0, 20.0, 8.0, 8.0]
    tracker.state = list(trusted_state)
    tracker._srbt_last_action = Action.LOCAL_UNRESOLVED
    sample_calls = iter((
        (torch.zeros_like(initial_event), initial_event, 1.0, None),
        (torch.zeros_like(template_event), template_event, 1.0, None),
    ))
    tracker._step_srbt_controller = lambda *_args, **_kwargs: SimpleNamespace(
        action=Action.LOCAL_UNRESOLVED,
        output_score=0.4,
        allow_recent_write=False,
        allow_long_write=False,
    )

    unresolved = PETTrack.track(
        tracker,
        torch.zeros(40, 40, 3).numpy(),
        torch.zeros(40, 40, 3).numpy(),
    )

    assert unresolved["target_bbox"] == initial_candidate["state"]
    assert tracker.state == trusted_state


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
                    "score_map": torch.full(
                        (1, 1, 4, 4) if name == "precision_refiner"
                        else (1, 1, 2, 2),
                        score,
                    ),
                    "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
                }
                for name, score in zip(
                    ("generalist", "motion_fm", "precision_refiner",
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
    assert tracker.network.calls[0]["small_template_features"] is None
    assert candidate["retained_expert_ids"] == (4,)
    assert candidate["ensemble_weights"].shape == (5,)
    assert int(torch.count_nonzero(candidate["ensemble_weights"])) == 1
    assert len(candidate["expert_states"]) == 5
    assert all(len(box) == 4 for box in candidate["expert_states"])


def test_tracker_stage_validation_commits_the_forced_expert_box():
    class Thor:
        @staticmethod
        def begin_frame():
            return torch.zeros(1, 4, 8), torch.zeros(1, 4, 8)

    class Network:
        @staticmethod
        def inference(**_kwargs):
            names = (
                "generalist", "motion_fm", "precision_refiner",
                "visibility_foc_ov", "discrimination_bi",
            )
            return {
                "presence_score": torch.tensor([0.8]),
                "expert_outputs": {
                    name: {
                        "score_map": torch.full((1, 1, 2, 2), 0.5 + index / 10),
                        "pred_boxes": torch.tensor([[[
                            0.2 + index / 10, 0.5, 0.2, 0.2,
                        ]]]),
                    }
                    for index, name in enumerate(names)
                },
            }

    tracker = object.__new__(PETTrack)
    tracker.thor_wrapper = Thor()
    tracker.network = Network()
    tracker.static_zi = torch.zeros(1, 4, 8)
    tracker.static_ze = torch.zeros(1, 4, 8)
    tracker.output_window = torch.ones(1, 1, 2, 2)
    tracker.params = SimpleNamespace(search_size=32)
    tracker.map_box_back = lambda box, factor, reference: box
    tracker.forced_expert_id = 1

    candidate = tracker._run_local_candidate(
        torch.zeros(1, 3, 32, 32),
        torch.zeros(1, 3, 32, 32),
        1.0,
        100,
        100,
        reference_state=[0.0, 0.0, 16.0, 16.0],
    )

    assert candidate["retained_expert_ids"] == (1,)
    assert int(candidate["ensemble_weights"].argmax()) == 1
    assert candidate["state"] == candidate["expert_states"][1]


def test_tracker_maps_sparse_activation_outputs_to_global_expert_slots():
    class Thor:
        @staticmethod
        def begin_frame():
            return torch.zeros(1, 4, 8), torch.zeros(1, 4, 8)

    class Network:
        expert_names = (
            "generalist", "motion_fm", "precision_refiner",
            "visibility_foc_ov", "discrimination_bi",
        )

        def __init__(self):
            self.calls = []

        def inference(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "presence_score": torch.tensor([0.8]),
                "expert_outputs": {
                    "generalist": {
                        "score_map": torch.full((1, 1, 2, 2), 0.2),
                        "pred_boxes": torch.tensor([[[0.3, 0.5, 0.2, 0.2]]]),
                    },
                    "discrimination_bi": {
                        "score_map": torch.full((1, 1, 2, 2), 0.95),
                        "pred_boxes": torch.tensor([[[0.3, 0.5, 0.2, 0.2]]]),
                    },
                },
            }

    tracker = object.__new__(PETTrack)
    tracker.thor_wrapper = Thor()
    tracker.network = Network()
    tracker.static_zi = torch.zeros(1, 4, 8)
    tracker.static_ze = torch.zeros(1, 4, 8)
    tracker.output_window = torch.ones(1, 1, 2, 2)
    tracker.params = SimpleNamespace(search_size=32)
    tracker.map_box_back = lambda box, factor, reference: box
    tracker.auto_expert_activation = True
    tracker.forced_expert_id = None

    candidate = tracker._run_local_candidate(
        torch.zeros(1, 3, 32, 32),
        torch.zeros(1, 3, 32, 32),
        1.0, 100, 100,
        reference_state=[0.0, 0.0, 16.0, 16.0],
    )

    assert tracker.network.calls[0]["auto_activate"] is True
    assert candidate["active_expert_ids"] == (0, 4)
    assert candidate["retained_expert_ids"] == (4,)
    assert candidate["ensemble_weights"].shape == (5,)
    assert int(candidate["ensemble_weights"].argmax()) == 4
    assert candidate["state"] == candidate["expert_states"][4]


def test_tracker_replaces_small_template_cache_for_each_sequence(monkeypatch):
    class SmallExpert:
        def __init__(self):
            self.calls = 0

        def encode_template(self, _rgb, _event):
            self.calls += 1
            return (f"s4-{self.calls}", f"s8-{self.calls}")

    class Network:
        def __init__(self):
            self.small_target_expert = SmallExpert()
            self.inference_calls = []

        def inference(self, **kwargs):
            self.inference_calls.append(kwargs)
            return {
                "score_map": torch.ones(1, 1, 2, 2),
                "target_bbox": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "presence_score": torch.tensor([0.8]),
            }

    class Thor:
        def setup(self, *_args):
            pass

        def begin_frame(self):
            return torch.zeros(1, 4, 8), torch.zeros(1, 4, 8)

    monkeypatch.setattr(
        "lib.test.tracker.pet_track.sample_target",
        lambda **_kwargs: (
            torch.zeros(1, 3, 16, 16),
            torch.zeros(1, 3, 16, 16),
            1.0,
            torch.zeros(1, 16, 16),
        ),
    )
    monkeypatch.setattr(
        "lib.test.tracker.pet_track.generate_mask_z",
        lambda **_kwargs: None,
    )
    tracker = object.__new__(PETTrack)
    tracker.params = SimpleNamespace(
        template_factor=2.0, template_size=16,
        search_factor=4.0, search_size=32,
    )
    tracker.preprocessor = SimpleNamespace(
        process=lambda patch, _mask: SimpleNamespace(tensors=patch))
    tracker.transform_bbox_to_crop = lambda *_args: torch.tensor(
        [[[0.5, 0.5, 0.2, 0.2]]])
    tracker.cfg = SimpleNamespace(MODEL=SimpleNamespace(
        BACKBONE=SimpleNamespace(CE_LOC=False)))
    tracker.network = Network()
    tracker.thor_wrapper = Thor()
    tracker.duration_decoder = SimpleNamespace(reset=lambda: None)
    tracker._reset_srbt_sequence_state = lambda: None
    tracker._record_search_diagnostic = lambda *_args, **_kwargs: None
    tracker.output_window = torch.ones(1, 1, 2, 2)
    tracker.map_box_back = lambda box, _factor, _reference: box
    info = {"init_bbox": [1.0, 1.0, 4.0, 4.0]}

    tracker.initialize(None, None, info, idx=0)
    first_cache = tracker.small_template_features
    tracker.initialize(None, None, info, idx=0)
    second_cache = tracker.small_template_features
    search = torch.zeros(1, 3, 32, 32)
    tracker._run_local_candidate(
        search, search, 1.0, 100, 100,
        reference_state=[0.0, 0.0, 16.0, 16.0])

    assert first_cache == ("s4-1", "s8-1")
    assert second_cache == ("s4-2", "s8-2")
    assert tracker.network.inference_calls[-1][
        "small_template_features"] is second_cache
