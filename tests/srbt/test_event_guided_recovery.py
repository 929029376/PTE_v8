import inspect
from types import SimpleNamespace

import pytest
import torch

from lib.models.layers.event_recovery import (
    EventProposalExtractor,
    RGBIdentityVerifier,
)
from lib.models.pet_track.pet_track import PETTrack as PETTrackModel
from lib.models.layers.srbt_controller import (
    Action,
    ControllerAction,
    DurationStructuredDecoder,
    LocalizationValidityGate,
)
from lib.test.tracker.pet_track import PETTrack as PETTrackTracker


def test_zero_event_frame_returns_finite_invalid_proposals():
    extractor = EventProposalExtractor(
        top_k=3, nms_radius=1, min_robust_score=2.0)

    output = extractor(torch.zeros(2, 3, 5, 7))

    assert output["heatmap"].shape == (2, 1, 5, 7)
    assert output["centers"].shape == (2, 3, 2)
    assert output["scores"].shape == (2, 3)
    assert output["valid"].shape == (2, 3)
    assert torch.isfinite(output["heatmap"]).all()
    assert torch.equal(output["scores"], torch.zeros(2, 3))
    assert not output["valid"].any()


def test_top_k_is_score_ordered_and_uses_original_image_coordinates():
    event = torch.zeros(1, 2, 5, 9)
    event[:, :, 1, 2] = 5.0
    event[:, :, 3, 7] = 9.0
    extractor = EventProposalExtractor(
        top_k=2, nms_radius=0, min_robust_score=1.0)

    output = extractor(event)

    assert output["valid"].tolist() == [[True, True]]
    assert output["scores"][0, 0] > output["scores"][0, 1]
    assert output["centers"][0, 0].tolist() == pytest.approx([7 / 8, 3 / 4])
    assert output["centers"][0, 1].tolist() == pytest.approx([2 / 8, 1 / 4])
    repeated = extractor(event)
    assert torch.equal(output["centers"], repeated["centers"])
    assert torch.equal(output["scores"], repeated["scores"])


def test_spatial_nms_keeps_only_the_stronger_neighbor():
    event = torch.zeros(1, 1, 7, 7)
    event[0, 0, 3, 3] = 8.0
    event[0, 0, 3, 4] = 7.0
    extractor = EventProposalExtractor(
        top_k=3, nms_radius=1, min_robust_score=1.0)

    output = extractor(event)

    assert output["valid"].sum().item() == 1
    assert output["centers"][0, 0].tolist() == pytest.approx([0.5, 0.5])


def test_background_subtraction_suppresses_static_event_activity():
    background = torch.zeros(1, 1, 7, 7)
    background[0, 0, 1, 1] = 9.0
    event = background.clone()
    event[0, 0, 5, 4] = 6.0
    extractor = EventProposalExtractor(
        top_k=2, nms_radius=1, min_robust_score=1.0)

    output = extractor(event, background=background)

    assert output["valid"].sum().item() == 1
    assert output["centers"][0, 0].tolist() == pytest.approx([4 / 6, 5 / 6])


def test_invalid_event_shapes_are_rejected():
    extractor = EventProposalExtractor()

    with pytest.raises(ValueError, match="event_frame"):
        extractor(torch.zeros(3, 8, 8))
    with pytest.raises(ValueError, match="background"):
        extractor(
            torch.zeros(1, 3, 8, 8),
            background=torch.zeros(1, 1, 8, 8),
        )


def test_rgb_identity_verifier_prefers_matching_tokens_and_detaches_inputs():
    verifier = RGBIdentityVerifier(input_dim=4, projection_dim=4)
    with torch.no_grad():
        verifier.projection[1].weight.copy_(torch.eye(4))
    template = torch.zeros(2, 3, 4, requires_grad=True)
    template.data[0, :, 0] = 1.0
    template.data[1, :, 1] = 1.0
    negative = template.detach().flip(0)
    candidates = torch.stack((template.detach(), negative), dim=1)
    candidates.requires_grad_(True)

    scores = verifier(template, candidates)
    scores.sum().backward()

    assert scores.shape == (2, 2)
    assert torch.all(scores[:, 0] > scores[:, 1])
    assert template.grad is None
    assert candidates.grad is None
    assert all(parameter.grad is not None for parameter in verifier.parameters())


def test_rgb_identity_verifier_has_no_event_input():
    parameters = inspect.signature(RGBIdentityVerifier.forward).parameters

    assert list(parameters) == [
        "self", "template_rgb_tokens", "candidate_rgb_tokens"]


def test_recovery_batches_candidates_and_requires_rgb_and_localization():
    class IdentityProbe(torch.nn.Module):
        def forward(self, _template, _candidates):
            return torch.tensor([[0.9, 0.2]])

    model = object.__new__(PETTrackModel)
    torch.nn.Module.__init__(model)
    model.rgb_identity_verifier = IdentityProbe()
    model.cfg = SimpleNamespace(MODEL=SimpleNamespace(REDETECT=SimpleNamespace(
        IDENTITY_THRESHOLD=0.75,
        LOCALIZATION_THRESHOLD=0.5,
    )))
    model.rgb_identity_tokens = lambda image, template: torch.zeros(
        image.shape[0], 4, 8)
    calls = []

    def redetect(zi, ze, xi, xe, dynamic_zi=None, dynamic_ze=None, prior_H=None,
                 use_template_conditioning=True):
        calls.append((
            zi, ze, xi, xe, dynamic_zi, dynamic_ze,
            prior_H, use_template_conditioning))
        return {
            "bbox": torch.tensor([
                [0.5, 0.5, 0.2, 0.2],
                [0.4, 0.4, 0.3, 0.3],
            ]),
            "conf": torch.tensor([0.8, 0.95]),
        }

    model.redetect_from_observations = redetect
    clean_rgb = torch.zeros(1, 3, 8, 8)
    clean_event = torch.zeros(1, 3, 8, 8)
    candidate_rgb = torch.zeros(1, 2, 3, 16, 16)
    candidate_event = torch.zeros_like(candidate_rgb)
    event_scores = torch.tensor([[2.0, 100.0]])
    event_priors = torch.rand(1, 2, 16, 16)

    output = model.recover_from_candidates(
        clean_rgb, clean_event, candidate_rgb, candidate_event,
        event_scores, event_priors)

    assert len(calls) == 1
    assert calls[0][2].shape[0] == 2
    assert torch.equal(calls[0][4], calls[0][0])
    assert torch.equal(calls[0][5], calls[0][1])
    assert torch.equal(calls[0][6], event_priors.flatten(0, 1))
    assert output["boxes"].shape == (1, 2, 4)
    assert torch.allclose(
        output["identity_scores"], torch.tensor([[0.9, 0.2]]))
    assert torch.allclose(
        output["localization_scores"], torch.tensor([[0.8, 0.95]]))
    assert torch.allclose(
        output["observability_scores"], torch.tensor([[0.02, 1.0]]))
    assert torch.equal(
        output["localization_validity_scores"], output["localization_scores"])
    assert torch.allclose(
        output["acceptance_scores"], torch.tensor([[0.016, 0.95]]))
    assert output["accepted"].tolist() == [[False, False]]
    assert "combined_scores" not in output


def test_recovery_acceptance_requires_identity_and_factorized_acceptance():
    class IdentityProbe(torch.nn.Module):
        def forward(self, _template, _candidates):
            return torch.tensor([[0.9, 0.9]])

    model = object.__new__(PETTrackModel)
    torch.nn.Module.__init__(model)
    model.rgb_identity_verifier = IdentityProbe()
    model.cfg = SimpleNamespace(MODEL=SimpleNamespace(REDETECT=SimpleNamespace(
        IDENTITY_THRESHOLD=0.75,
        ACCEPTANCE_THRESHOLD=0.5,
    )))
    model.rgb_identity_tokens = lambda image, template: torch.zeros(
        image.shape[0], 4, 8)
    model.redetect_from_observations = lambda *args, **kwargs: {
        "bbox": torch.tensor([
            [0.5, 0.5, 0.2, 0.2],
            [0.4, 0.4, 0.3, 0.3],
        ]),
        "conf": torch.tensor([0.9, 0.6]),
    }

    output = model.recover_from_candidates(
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 2, 3, 16, 16),
        torch.zeros(1, 2, 3, 16, 16),
        torch.tensor([[0.2, 1.0]]),
        torch.rand(1, 2, 16, 16),
    )

    assert torch.allclose(
        output["acceptance_scores"], torch.tensor([[0.18, 0.6]]))
    assert output["accepted"].tolist() == [[False, True]]


def test_recovery_field_uses_candidate_localization_gate_without_backbone_gradients():
    class IdentityProbe(torch.nn.Module):
        def forward(self, _template, _candidates):
            return torch.tensor([[0.9, 0.8]])

    model = object.__new__(PETTrackModel)
    torch.nn.Module.__init__(model)
    model.rgb_identity_verifier = IdentityProbe()
    model.localization_validity_gate = LocalizationValidityGate(hidden_dim=8)
    model.cfg = SimpleNamespace(MODEL=SimpleNamespace(REDETECT=SimpleNamespace(
        IDENTITY_THRESHOLD=0.75,
        LOCALIZATION_THRESHOLD=0.5,
    )))
    model.rgb_identity_tokens = lambda image, template: torch.zeros(
        image.shape[0], 4, 8)
    field = torch.rand(2, 1, 4, 4, requires_grad=True)
    boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.4, 0.4, 0.3, 0.3],
    ], requires_grad=True)
    model.redetect_from_observations = lambda *args, **kwargs: {
        "bbox": boxes,
        "field": field,
        "conf": torch.zeros(2),
    }

    output = model.recover_from_candidates(
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 2, 3, 16, 16),
        torch.zeros(1, 2, 3, 16, 16),
        torch.tensor([[1.0, 0.5]]),
        torch.rand(1, 2, 16, 16),
    )

    assert output["localization_validity_scores"].shape == (1, 2)
    assert torch.allclose(
        output["acceptance_scores"],
        output["observability_scores"]
        * output["localization_validity_scores"],
    )
    output["acceptance_scores"].sum().backward()
    assert field.grad is None
    assert boxes.grad is None
    assert all(
        parameter.grad is not None
        for parameter in model.localization_validity_gate.parameters()
    )


def test_tracker_crops_full_frame_event_proposals_as_one_batch(monkeypatch):
    class ProposalProbe:
        def __call__(self, _event):
            heatmap = torch.zeros(1, 1, 100, 100)
            heatmap[0, 0, 90, 90] = 10.0
            return {
                "heatmap": heatmap,
                "centers": torch.tensor([[[0.9, 0.9], [0.0, 0.0]]]),
                "scores": torch.tensor([[10.0, 0.0]]),
                "valid": torch.tensor([[True, False]]),
            }

    class RecoveryProbe:
        def __init__(self):
            self.calls = []

        def recover_from_candidates(self, *args):
            self.calls.append(args)
            return {
                "boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
                "identity_scores": torch.tensor([[0.9]]),
                "localization_scores": torch.tensor([[0.8]]),
                "event_scores": torch.tensor([[10.0]]),
                "combined_scores": torch.tensor([[0.85]]),
                "accepted": torch.tensor([[True]]),
            }

    sampled = []

    def sample_probe(**kwargs):
        sampled.append(kwargs["target_bb"])
        value = float(torch.as_tensor(kwargs["im"]).max())
        return (
            torch.full((8, 8, 3), value),
            torch.zeros(8, 8, 3),
            0.5,
            torch.zeros(8, 8),
        )

    monkeypatch.setattr(
        "lib.test.tracker.pet_track.sample_target", sample_probe)
    tracker = object.__new__(PETTrackTracker)
    tracker.device = torch.device("cpu")
    tracker.state = [0.0, 0.0, 10.0, 10.0]
    tracker.params = SimpleNamespace(search_size=8)
    tracker.recovery_search_factor = 5.0
    tracker.event_proposal_extractor = ProposalProbe()
    tracker.preprocessor = SimpleNamespace(process=lambda patch, _mask:
        SimpleNamespace(tensors=patch.permute(2, 0, 1).unsqueeze(0)))
    tracker.thor_wrapper = SimpleNamespace(get_clean_template=lambda: (
        torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4)))
    tracker.network = RecoveryProbe()

    output = PETTrackTracker._run_event_recovery(
        tracker,
        torch.zeros(100, 100, 3),
        torch.zeros(100, 100, 3),
        100,
        100,
    )

    anchor = sampled[0]
    assert anchor[0] + 0.5 * anchor[2] == pytest.approx(89.1)
    assert anchor[1] + 0.5 * anchor[3] == pytest.approx(89.1)
    assert len(tracker.network.calls) == 1
    candidate_rgb = tracker.network.calls[0][2]
    event_priors = tracker.network.calls[0][5]
    assert candidate_rgb.shape == (1, 1, 3, 8, 8)
    assert event_priors.shape == (1, 1, 8, 8)
    assert event_priors.max().item() > 0.0
    assert output["accepted"].tolist() == [[True]]
    assert output["_proposal_centers"][0] == pytest.approx([89.1, 89.1])


def test_tracker_maps_and_persists_recovery_hypotheses_globally():
    class HypothesisProbe:
        def __init__(self):
            self.observed = None

        def update(self, _previous, observed):
            self.observed = observed
            active = observed["active_mask"]
            return {
                "boxes": observed["boxes"][active],
                "weights": observed["field_scores"][active],
                "identity": observed["identity"][active],
                "active_count": int(active.sum()),
            }

    tracker = object.__new__(PETTrackTracker)
    tracker.params = SimpleNamespace(search_size=100)
    tracker.state = [0.0, 0.0, 10.0, 10.0]
    tracker._redetect_hypotheses = None
    tracker.hypothesis_tracker = HypothesisProbe()
    recovery = {
        "boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        "identity_scores": torch.tensor([[0.9]]),
        "localization_scores": torch.tensor([[0.8]]),
        "acceptance_scores": torch.tensor([[0.72]]),
        "accepted": torch.tensor([[True]]),
        "_anchors": [[55.0, 55.0, 10.0, 10.0]],
        "_resize_factors": [1.0],
    }

    box, confidence = PETTrackTracker._update_event_recovery_hypotheses(
        tracker, recovery, 100, 100)

    assert box == pytest.approx([50.0, 50.0, 20.0, 20.0])
    assert confidence == pytest.approx(0.72)
    observed = tracker.hypothesis_tracker.observed
    assert observed["boxes"][0].tolist() == pytest.approx(
        [0.6, 0.6, 0.2, 0.2])
    assert observed["active_mask"].tolist() == [True]


def test_no_event_proposal_ages_existing_recovery_hypotheses():
    class DecayProbe:
        def __init__(self):
            self.calls = []

        def update(self, previous, observed):
            self.calls.append((previous, observed))
            return {"active_count": 0}

    tracker = object.__new__(PETTrackTracker)
    tracker._redetect_hypotheses = {"active_count": 1}
    tracker.hypothesis_tracker = DecayProbe()

    PETTrackTracker._decay_recovery_hypotheses(tracker)

    assert tracker.hypothesis_tracker.calls == [({"active_count": 1}, None)]
    assert tracker._redetect_hypotheses is None


def test_recovery_cycle_clips_zero_area_candidate_before_pending_search():
    tracker = object.__new__(PETTrackTracker)
    tracker.frame_id = 1
    tracker.full_rgb_fallback_interval = 10
    tracker._pending_redetect_box = None
    tracker._last_redetect_conf = 0.0
    tracker._redetect_hypotheses = None
    recovery = {"accepted": torch.tensor([[True]])}
    tracker._run_event_recovery = lambda *_args: recovery
    tracker._update_event_recovery_hypotheses = (
        lambda *_args: ([100.0, 80.0, 0.0, 0.0], 0.9)
    )

    output = PETTrackTracker._run_recovery_cycle(
        tracker,
        torch.zeros(80, 100, 3),
        torch.zeros(80, 100, 3),
        80,
        100,
    )

    assert output is recovery
    assert tracker._pending_redetect_box == pytest.approx(
        [90.0, 70.0, 10.0, 10.0])
    assert tracker._last_redetect_conf == pytest.approx(0.9)


def test_pending_recovery_box_is_refined_with_current_dynamic_templates(
        monkeypatch):
    sampled = []

    def sample_target_for_test(**kwargs):
        sampled.append(kwargs["target_bb"])
        return (
            torch.zeros(8, 8, 3),
            torch.zeros(8, 8, 3),
            2.0,
            torch.zeros(8, 8),
        )

    monkeypatch.setattr(
        "lib.test.tracker.pet_track.sample_target", sample_target_for_test)
    tracker = object.__new__(PETTrackTracker)
    tracker._pending_redetect_box = [10.0, 12.0, 6.0, 8.0]
    tracker.params = SimpleNamespace(search_factor=3.0, search_size=8)
    tracker.preprocessor = SimpleNamespace(
        process=lambda patch, _mask: SimpleNamespace(
            tensors=patch.permute(2, 0, 1).unsqueeze(0)))
    tracker.dynamic_zi = torch.randn(1, 2, 4)
    tracker.dynamic_ze = torch.randn(1, 2, 4)
    calls = []

    def run_local(*args, **kwargs):
        calls.append(kwargs)
        return {"state": [11.0, 13.0, 5.0, 7.0]}

    tracker._run_local_candidate = run_local

    refined = PETTrackTracker._refine_pending_recovery(
        tracker,
        torch.zeros(32, 32, 3),
        torch.zeros(32, 32, 3),
        32,
        32,
    )

    assert sampled == [[10.0, 12.0, 6.0, 8.0]]
    assert refined["state"] == [11.0, 13.0, 5.0, 7.0]
    assert calls[0]["reference_state"] == sampled[0]
    assert calls[0]["dynamic_templates"] == (
        tracker.dynamic_zi, tracker.dynamic_ze)


def test_verify_keeps_thor_frozen_and_does_not_write_memory(monkeypatch):
    class ThorProbe:
        def __init__(self):
            self.frozen = True
            self.resume_calls = 0
            self.commit_calls = 0

        def resume(self):
            self.frozen = False
            self.resume_calls += 1

        def commit(self, *_args, **_kwargs):
            self.commit_calls += 1

    monkeypatch.setattr(
        "lib.test.tracker.pet_track.sample_target",
        lambda **_kwargs: (
            torch.zeros(8, 8, 3),
            torch.zeros(8, 8, 3),
            1.0,
            torch.zeros(8, 8),
        ),
    )
    tracker = object.__new__(PETTrackTracker)
    tracker.frame_id = 0
    tracker.state = [1.0, 1.0, 2.0, 2.0]
    tracker._pending_redetect_box = [5.0, 5.0, 2.0, 2.0]
    tracker._redetect_hypotheses = {"active_count": 1}
    tracker._srbt_last_action = Action.GLOBAL_UNRESOLVED
    tracker._last_redetect_error = ""
    tracker._last_score_peak = 0.0
    tracker._last_redetect_conf = 0.9
    tracker._recovery_diagnostics = []
    tracker.params = SimpleNamespace(
        search_factor=2.0,
        search_size=8,
        template_factor=2.0,
        template_size=8,
    )
    tracker.preprocessor = SimpleNamespace(process=lambda patch, _mask:
        SimpleNamespace(tensors=patch.permute(2, 0, 1).unsqueeze(0)))
    tracker.thor_wrapper = ThorProbe()
    tracker.debug = False
    tracker._run_local_candidate = lambda *_args, **_kwargs: {
        "state": [5.0, 5.0, 2.0, 2.0],
        "score_peak": 0.9,
        "response": torch.ones(1, 1, 2, 2),
        "memory_frame_open": False,
    }
    tracker._run_recovery_cycle = lambda *_args, **_kwargs: None
    tracker._step_srbt_controller = lambda _candidate: ControllerAction(
        action=Action.VERIFY,
        allow_recent_write=False,
        allow_long_write=False,
        output_absent=True,
        output_score=0.0,
    )

    output = PETTrackTracker.track(
        tracker,
        torch.zeros(8, 8, 3),
        torch.zeros(8, 8, 3),
    )

    assert output["absent"] is True
    assert tracker.thor_wrapper.frozen is True
    assert tracker.thor_wrapper.resume_calls == 0
    assert tracker.thor_wrapper.commit_calls == 0
    assert tracker._pending_redetect_box == [5.0, 5.0, 2.0, 2.0]
    assert tracker.get_recovery_diagnostics()[-1]["action"] == "verify"


def test_two_current_recovery_confirmations_resume_tracking(monkeypatch):
    class ThorProbe:
        def __init__(self):
            self.resume_calls = 0

        def resume(self):
            self.resume_calls += 1

        @staticmethod
        def commit(*_args, **_kwargs):
            pass

    monkeypatch.setattr(
        "lib.test.tracker.pet_track.sample_target",
        lambda **_kwargs: (
            torch.zeros(8, 8, 3),
            torch.zeros(8, 8, 3),
            1.0,
            torch.zeros(8, 8),
        ),
    )
    tracker = object.__new__(PETTrackTracker)
    tracker.frame_id = 0
    tracker.state = [1.0, 1.0, 4.0, 4.0]
    tracker._pending_redetect_box = None
    tracker._redetect_hypotheses = None
    tracker._srbt_last_action = Action.GLOBAL_UNRESOLVED
    tracker._last_redetect_error = ""
    tracker._last_score_peak = 0.0
    tracker._last_redetect_conf = 0.0
    tracker._recovery_diagnostics = []
    tracker.params = SimpleNamespace(
        search_factor=2.0,
        search_size=8,
        template_factor=2.0,
        template_size=8,
    )
    tracker.preprocessor = SimpleNamespace(process=lambda patch, _mask:
        SimpleNamespace(tensors=patch.permute(2, 0, 1).unsqueeze(0)))
    tracker.thor_wrapper = ThorProbe()
    tracker.duration_decoder = DurationStructuredDecoder(
        theta_recover=0.75, verify_duration=2)
    tracker.duration_decoder.state = Action.GLOBAL_UNRESOLVED
    tracker.debug = False
    tracker.full_rgb_fallback_interval = 10
    tracker._run_local_candidate = lambda *_args, **_kwargs: {
        "state": [2.0, 2.0, 4.0, 4.0],
        "score_peak": 0.1,
        "presence_score": 0.1,
        "response": torch.ones(1, 1, 2, 2),
        "memory_frame_open": False,
    }
    tracker._run_event_recovery = lambda *_args, **_kwargs: {
        "identity_scores": torch.tensor([[0.9]]),
        "localization_scores": torch.tensor([[0.9]]),
        "observability_scores": torch.tensor([[0.9]]),
        "localization_validity_scores": torch.tensor([[0.9]]),
        "acceptance_scores": torch.tensor([[0.81]]),
        "accepted": torch.tensor([[True]]),
        "_proposal_centers": [[12.0, 12.0]],
    }
    tracker._update_event_recovery_hypotheses = (
        lambda *_args, **_kwargs: ([10.0, 10.0, 12.0, 12.0], 0.9)
    )
    tracker._run_redetection = lambda *_args, **_kwargs: None
    refine_calls = []

    def refine_pending(*_args, **_kwargs):
        refine_calls.append(list(tracker._pending_redetect_box))
        return {
            "state": [11.0, 11.0, 10.0, 10.0],
            "score_peak": 0.95,
            "presence_score": 0.95,
            "response": torch.ones(1, 1, 2, 2),
            "memory_frame_open": False,
        }

    tracker._refine_pending_recovery = refine_pending

    first = PETTrackTracker.track(
        tracker, torch.zeros(32, 32, 3), torch.zeros(32, 32, 3))
    second = PETTrackTracker.track(
        tracker, torch.zeros(32, 32, 3), torch.zeros(32, 32, 3))

    assert first["absent"] is True
    assert tracker.get_recovery_diagnostics()[0]["action"] == "verify"
    assert second["absent"] is False
    assert second["target_bbox"] == pytest.approx([11.0, 11.0, 10.0, 10.0])
    assert refine_calls == [[10.0, 10.0, 12.0, 12.0]]
    assert tracker.get_recovery_diagnostics()[1]["action"] == "track"
    assert tracker.thor_wrapper.resume_calls == 1
