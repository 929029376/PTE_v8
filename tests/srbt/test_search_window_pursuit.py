from types import SimpleNamespace

import pytest
import torch

import lib.train.data.processing as processing_module
from lib.train.data.sampler import TrackingSampler
from lib.train.data.processing import STARKProcessing
from lib.train.actors.pet_track import PETTrackActor
import lib.train.actors.pet_track as pet_track_actor_module
from lib.train.train_script import _validate_pursuit_stage
from lib.test.tracker.pet_track import PETTrack as InferenceTracker
from lib.models.layers.srbt_controller import Action as BeliefAction
from lib.models.layers.search_window_controller import (
    SearchWindowController,
    crop_box_to_image,
    crop_target_inside,
    dynamic_search_crop,
    event_motion_centroid,
    relative_box_motion,
    search_window_pursuit_loss,
)


def test_dynamic_crop_and_box_mapping_share_the_same_geometry():
    image = torch.zeros(1, 1, 8, 8)
    image[0, 0, 3, 5] = 1.0
    anchor = torch.tensor([[0.50, 0.25, 0.25, 0.25]])

    crop, crop_box = dynamic_search_crop(
        image, anchor, search_factor=2.0, output_size=8)
    mapped = crop_box_to_image(
        torch.tensor([[[0.50, 0.50, 0.25, 0.25]]]), crop_box)

    assert crop.shape == (1, 1, 8, 8)
    assert crop.max().item() > 0.5
    assert mapped[0, 0, :2].tolist() == pytest.approx([0.5625, 0.3125])
    assert mapped[0, 0, 2:].tolist() == pytest.approx([0.125, 0.125])


def test_pursuit_full_frame_canvas_preserves_native_felt_resolution(monkeypatch):
    processing = object.__new__(STARKProcessing)
    processing.output_sz = {"search": 256}
    processing.pursuit_canvas_size = 352
    processing.transform = {
        "search": lambda image, bbox, att, mask, **_kwargs: (
            image, bbox, att, mask),
    }
    captured = {}

    def fake_crop(*, frames, event_frames, box_extract, box_gt,
                  search_area_factor, output_sz, masks):
        captured["output_sz"] = output_sz
        attention = [torch.zeros(output_sz, output_sz) for _ in frames]
        return frames, event_frames, box_gt, attention, masks

    monkeypatch.setattr(
        processing_module.prutils, "jittered_center_crop", fake_crop)
    frame = torch.zeros(260, 346, 3)
    data = {
        "pursuit_search_images": [frame],
        "pursuit_search_event_images": [frame.clone()],
        "pursuit_search_anno": [torch.tensor([20.0, 30.0, 4.0, 5.0])],
        "pursuit_search_masks": [torch.zeros(260, 346)],
    }

    processing._process_full_frame_sequence(
        data, "pursuit_search", 1.0, keep_auxiliary=True)

    assert captured["output_sz"] == 352
    assert captured["output_sz"] >= 346


def test_pursuit_processing_skips_unused_standard_search_crop(monkeypatch):
    processing = STARKProcessing(
        search_area_factor={"template": 2.0, "search": 4.0},
        output_sz={"template": 16, "search": 32},
        center_jitter_factor={"template": 0.0, "search": 0.0},
        scale_jitter_factor={"template": 0.0, "search": 0.0},
        mode="sequence",
        settings=SimpleNamespace(
            pursuit_enabled=True,
            pursuit_canvas_size=40,
        ),
        transform=lambda image, bbox, att, mask, **_kwargs: (
            image, bbox, att, mask),
    )
    crop_sizes = []

    def fake_crop(*, frames, event_frames, box_extract, box_gt,
                  search_area_factor, output_sz, masks):
        crop_sizes.append(output_sz)
        attention = [torch.zeros(output_sz, output_sz) for _ in frames]
        return frames, event_frames, box_gt, attention, masks

    monkeypatch.setattr(
        processing_module.prutils, "jittered_center_crop", fake_crop)
    template = torch.zeros(16, 16, 3)
    search = torch.zeros(40, 40, 3)
    template_box = torch.tensor([4.0, 4.0, 8.0, 8.0])
    search_boxes = [
        torch.tensor([8.0, 8.0, 8.0, 8.0]),
        torch.tensor([10.0, 8.0, 8.0, 8.0]),
    ]
    data = processing_module.TensorDict({
        "template_images": [template],
        "template_event_images": [template.clone()],
        "template_anno": [template_box],
        "template_masks": [torch.zeros(16, 16)],
        "search_images": [search, search.clone()],
        "search_event_images": [search.clone(), search.clone()],
        "search_anno": search_boxes,
        "search_masks": [torch.zeros(40, 40), torch.zeros(40, 40)],
        "pursuit_search_images": [search, search.clone()],
        "pursuit_search_event_images": [search.clone(), search.clone()],
        "pursuit_search_anno": search_boxes,
        "pursuit_search_masks": [torch.zeros(40, 40), torch.zeros(40, 40)],
    })

    output = processing(data)

    assert crop_sizes == [16, 40]
    assert output["pursuit_search_images"].shape[0] == 2
    assert "search_images" not in output
    assert "search_event_images" not in output


def test_target_inside_uses_the_actual_search_crop_not_the_anchor_box():
    anchor = torch.tensor([
        [0.40, 0.40, 0.10, 0.10],
        [0.40, 0.40, 0.10, 0.10],
    ])
    targets = torch.tensor([
        [0.43, 0.43, 0.04, 0.04],
        [0.80, 0.80, 0.04, 0.04],
    ])

    inside = crop_target_inside(targets, anchor, search_factor=4.0)

    assert inside.tolist() == [True, False]


def test_event_centroid_reports_motion_location_and_empty_confidence():
    events = torch.zeros(2, 3, 8, 8)
    events[0, :, 2, 6] = 5.0

    center, confidence = event_motion_centroid(events)

    assert center[0].tolist() == pytest.approx([6.5 / 8.0, 2.5 / 8.0])
    assert confidence[0].item() > 0.9
    assert center[1].tolist() == pytest.approx([0.5, 0.5])
    assert confidence[1].item() == pytest.approx(0.0)


def test_relative_box_motion_uses_previous_target_scale():
    previous = torch.tensor([[0.10, 0.20, 0.20, 0.10]])
    current = torch.tensor([[0.20, 0.15, 0.40, 0.05]])

    delta = relative_box_motion(current, previous)

    assert delta[0, :2].tolist() == pytest.approx([1.0, -0.75])
    assert delta[0, 2:].tolist() == pytest.approx([
        torch.log(torch.tensor(2.0)).item(),
        torch.log(torch.tensor(0.5)).item(),
    ])


def test_controller_is_lightweight_and_initializes_from_expert_consensus():
    controller = SearchWindowController(expert_count=5, hidden_dim=32)
    current = torch.tensor([[0.10, 0.20, 0.10, 0.10]])
    previous = torch.tensor([[0.08, 0.20, 0.10, 0.10]])
    expert_boxes = torch.tensor([[[
        0.20, 0.20, 0.10, 0.10,
    ], [
        0.22, 0.20, 0.10, 0.10,
    ], [
        0.18, 0.20, 0.10, 0.10,
    ], [
        0.21, 0.20, 0.10, 0.10,
    ], [
        0.19, 0.20, 0.10, 0.10,
    ]]])
    peaks = torch.ones(1, 5)
    psr = torch.ones(1, 5)

    output = controller(
        current_box=current,
        previous_box=previous,
        expert_boxes=expert_boxes,
        response_peaks=peaks,
        response_psr=psr,
        presence=torch.ones(1, 1),
        event_center=torch.tensor([[0.20, 0.20]]),
        event_confidence=torch.ones(1, 1),
    )

    assert sum(parameter.numel() for parameter in controller.parameters()) < 10_000
    assert output.next_box.shape == (1, 4)
    assert output.next_box[0].tolist() == pytest.approx(
        [0.20, 0.20, 0.10, 0.10], abs=1e-5)
    assert output.inside_logit.shape == (1,)
    assert output.quality_logit.shape == (1,)


def test_pursuit_loss_masks_local_quality_outside_crop_but_trains_containment():
    predictions = SimpleNamespace(
        next_box=torch.tensor([
            [0.40, 0.40, 0.10, 0.10],
            [0.40, 0.40, 0.10, 0.10],
        ], requires_grad=True),
        inside_logit=torch.zeros(2, requires_grad=True),
        quality_logit=torch.zeros(2, requires_grad=True),
    )
    target_next = torch.tensor([
        [0.42, 0.42, 0.05, 0.05],
        [0.80, 0.80, 0.05, 0.05],
    ])
    current_inside = torch.tensor([True, False])
    current_quality = torch.tensor([0.8, 0.0])
    present_next = torch.tensor([True, True])

    loss, status = search_window_pursuit_loss(
        predictions,
        target_next=target_next,
        current_inside=current_inside,
        current_quality=current_quality,
        present_next=present_next,
        search_factor=4.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert status["Pursuit/quality_count"] == 1
    assert status["Pursuit/outside_count"] == 1
    assert status["Loss/pursuit_containment"] > 0.0
    assert predictions.next_box.grad is not None
    assert predictions.inside_logit.grad is not None
    assert predictions.quality_logit.grad is not None


def test_pursuit_sampler_returns_strictly_contiguous_frames_with_absence():
    sampler = object.__new__(TrackingSampler)
    sampler.num_template_frames = 2
    sampler.num_search_frames = 8
    sampler.max_gap = 20
    sampler.pursuit_transition_probability = 1.0
    sampler.pursuit_reappear_probability = 1.0
    visible = torch.tensor(
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.uint8)
    info = {
        "bbox": torch.ones(12, 4),
        "valid": visible.clone(),
        "absent": visible.clone(),
    }

    template_ids, search_ids, event_type = (
        sampler._sample_pursuit_causal_frame_ids(visible, info))

    assert len(template_ids) == 2
    assert len(search_ids) == 8
    assert search_ids == list(range(search_ids[0], search_ids[0] + 8))
    assert bool(visible[search_ids[0]])
    assert max(template_ids) < search_ids[0]
    assert any(not bool(info["absent"][frame_id]) for frame_id in search_ids)
    assert event_type == "pursuit_contiguous"


def test_pursuit_specialist_window_contains_enough_eligible_motion_frames():
    sampler = object.__new__(TrackingSampler)
    sampler.num_template_frames = 2
    sampler.num_search_frames = 8
    sampler.max_gap = 20
    visible = torch.ones(14, dtype=torch.uint8)
    info = {
        "bbox": torch.ones(14, 4),
        "valid": torch.ones(14, dtype=torch.uint8),
        "absent": visible.clone(),
    }
    eligible = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.bool,
    )

    _, search_ids, _ = sampler._sample_pursuit_causal_frame_ids(
        visible, info, episode_type="visible", eligible_frames=eligible)

    assert search_ids == list(range(search_ids[0], search_ids[0] + 8))
    assert int(eligible[search_ids].sum()) >= 4


def test_pursuit_transition_quota_retries_sequences_without_absence(
        monkeypatch):
    sampler = object.__new__(TrackingSampler)
    sampler.num_template_frames = 2
    sampler.num_search_frames = 4
    sampler.max_gap = 20
    sampler.pursuit_transition_probability = 1.0
    monkeypatch.setattr("lib.train.data.sampler.random.random", lambda: 0.0)
    visible = torch.ones(12, dtype=torch.uint8)
    info = {
        "bbox": torch.ones(12, 4),
        "valid": torch.ones(12, dtype=torch.uint8),
        "absent": visible.clone(),
    }

    template_ids, search_ids, event_type = (
        sampler._sample_pursuit_causal_frame_ids(visible, info))

    assert template_ids is None
    assert search_ids is None
    assert event_type is None


def test_pursuit_episode_types_are_exactly_stratified_by_index():
    sampler = object.__new__(TrackingSampler)
    sampler.pursuit_transition_probability = 0.5
    sampler.pursuit_reappear_probability = 0.25

    types = [sampler._pursuit_episode_type_for_index(i) for i in range(100)]

    assert types.count("reappearance") == 25
    assert types.count("disappearance") == 25
    assert types.count("visible") == 50


def test_pursuit_validation_uses_the_same_contiguous_sampler_contract():
    class Dataset:
        def __len__(self):
            return 1

    cfg = SimpleNamespace(
        DATA=SimpleNamespace(
            SRBT=SimpleNamespace(ENABLE=False, ANCHOR_WEIGHTS=None),
            PURSUIT=SimpleNamespace(ENABLE=True),
            CHALLENGE_SAMPLING=SimpleNamespace(
                ENABLE=False, PRECISE=False, MANIFEST=""),
        ),
        TRAIN=SimpleNamespace(
            EXPERT_PHASE="pursuit",
            SPECIALIST_EXPERT_IDS=[1, 2, 3, 4],
            SPECIALIST_EXPERT_SCHEDULE=[],
        ),
    )

    sampler = TrackingSampler(
        datasets=[Dataset()],
        p_datasets=[1],
        samples_per_epoch=1,
        max_gap=20,
        num_search_frames=8,
        num_template_frames=2,
        cfg=cfg,
        training=False,
    )

    assert sampler.pursuit_enabled is True


@pytest.mark.parametrize("training", [True, False])
def test_single_specialist_pursuit_enables_precise_frame_manifest(
        monkeypatch, training):
    class Dataset:
        def __len__(self):
            return 1

    monkeypatch.setattr(
        "lib.train.data.sampler.load_manifest",
        lambda _path: {"sequences": {}},
    )
    cfg = SimpleNamespace(
        DATA=SimpleNamespace(
            SRBT=SimpleNamespace(ENABLE=False, ANCHOR_WEIGHTS=None),
            PURSUIT=SimpleNamespace(ENABLE=True),
            CHALLENGE_SAMPLING=SimpleNamespace(
                ENABLE=True,
                PRECISE=True,
                MANIFEST="frame_manifest.json",
                VAL_MANIFEST="frame_manifest.json",
            ),
        ),
        TRAIN=SimpleNamespace(
            EXPERT_PHASE="pursuit",
            SPECIALIST_EXPERT_IDS=[1],
            SPECIALIST_EXPERT_SCHEDULE=[],
        ),
    )

    sampler = TrackingSampler(
        datasets=[Dataset()],
        p_datasets=[1],
        samples_per_epoch=8,
        max_gap=20,
        num_search_frames=8,
        num_template_frames=2,
        cfg=cfg,
        training=training,
    )

    assert sampler.causal_specialist_pursuit is True
    assert sampler.precise_expert_sampling is True
    assert sampler.training_expert_for_index(0) == 1


def test_pursuit_stage_requires_closed_loop_data_and_completed_experts():
    cfg = SimpleNamespace(
        MODEL=SimpleNamespace(
            INIT_CHECKPOINT="",
            SEARCH_CONTROLLER=SimpleNamespace(
                ENABLE=False, USE_INFERENCE=False),
        ),
        DATA=SimpleNamespace(
            SEARCH=SimpleNamespace(SIZE=256),
            TEMPLATE=SimpleNamespace(NUMBER=3),
            PURSUIT=SimpleNamespace(
                ENABLE=False, WINDOW_LENGTH=8, CANVAS_SIZE=352),
        ),
        TRAIN=SimpleNamespace(EXPERT_PHASE="pursuit"),
    )

    with pytest.raises(RuntimeError, match="INIT_CHECKPOINT"):
        _validate_pursuit_stage(cfg)

    cfg.MODEL.INIT_CHECKPOINT = "completed-v29.pth.tar"
    cfg.MODEL.SEARCH_CONTROLLER.ENABLE = True
    cfg.DATA.PURSUIT.ENABLE = True
    _validate_pursuit_stage(cfg)


def test_actor_uses_previous_controller_output_for_the_next_frame_crop(monkeypatch):
    class ShiftController(torch.nn.Module):
        def forward(self, **kwargs):
            next_box = kwargs["current_box"].clone()
            next_box[:, 0] += 0.1
            return SimpleNamespace(
                next_box=next_box,
                inside_logit=next_box[:, 0] * 0.0,
                quality_logit=next_box[:, 0] * 0.0,
            )

    class FrozenExperts(torch.nn.Module):
        expert_names = ["g", "m", "p", "v", "d"]

        def __init__(self):
            super().__init__()
            self.search_window_controller = ShiftController()

        def inference(self, **kwargs):
            box = torch.tensor(
                [[[0.5, 0.5, 0.2, 0.2]]],
                device=kwargs["xi"].device,
            )
            score = torch.full(
                (1, 1, 4, 4), 0.8, device=kwargs["xi"].device)
            experts = {
                name: {"pred_boxes": box, "score_map": score}
                for name in self.expert_names
            }
            return {
                "expert_outputs": experts,
                "presence_score": torch.ones(1, device=box.device),
            }

    captured_anchors = []
    real_crop = dynamic_search_crop

    def capture_crop(images, anchor_boxes, search_factor, output_size):
        captured_anchors.append(anchor_boxes.detach().clone())
        return real_crop(images, anchor_boxes, search_factor, output_size)

    monkeypatch.setattr(
        pet_track_actor_module, "dynamic_search_crop", capture_crop)
    actor = object.__new__(PETTrackActor)
    actor.net = FrozenExperts()
    actor.settings = SimpleNamespace(search_area_factor={"search": 4.0})
    actor.cfg = SimpleNamespace(DATA=SimpleNamespace(
        SEARCH=SimpleNamespace(SIZE=8, FACTOR=4.0)))
    frames = torch.zeros(3, 1, 3, 8, 8)
    boxes = torch.tensor([
        [[0.40, 0.40, 0.10, 0.10]],
        [[0.45, 0.40, 0.10, 0.10]],
        [[0.50, 0.40, 0.10, 0.10]],
    ])
    data = {
        "template_images": torch.zeros(2, 1, 3, 4, 4),
        "template_event_images": torch.zeros(2, 1, 3, 4, 4),
        "pursuit_search_images": frames,
        "pursuit_search_event_images": frames,
        "pursuit_search_anno": boxes,
        "pursuit_search_present": torch.ones(3, 1, dtype=torch.uint8),
    }

    output = actor._forward_pursuit(data)

    assert len(captured_anchors) == 4
    assert torch.equal(captured_anchors[0], boxes[0])
    assert torch.equal(captured_anchors[0], captured_anchors[1])
    assert torch.equal(
        captured_anchors[2], output["pursuit_predictions"][0].next_box)
    assert torch.equal(captured_anchors[2], captured_anchors[3])
    assert captured_anchors[2].grad_fn is None


def test_pursuit_uses_per_row_sparse_activation_with_fixed_expert_slots():
    class CapturingController(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append({
                name: value.detach().clone()
                for name, value in kwargs.items()
                if torch.is_tensor(value)
            })
            current_box = kwargs["current_box"]
            return SimpleNamespace(
                next_box=current_box,
                inside_logit=current_box[:, 0] * 0.0,
                quality_logit=current_box[:, 0] * 0.0,
            )

    class SparseExperts(torch.nn.Module):
        expert_names = ["g", "m", "p", "v", "d"]

        def __init__(self):
            super().__init__()
            self.search_window_controller = CapturingController()
            self.inference_kwargs = []

        def inference(self, **kwargs):
            self.inference_kwargs.append(dict(kwargs))
            batch_size = kwargs["xi"].shape[0]

            def output(center, peak):
                box = kwargs["xi"].new_tensor(
                    [center, 0.5, 0.2, 0.2]
                ).reshape(1, 1, 4).expand(batch_size, -1, -1)
                score = kwargs["xi"].new_full(
                    (batch_size, 1, 4, 4), peak)
                return {"pred_boxes": box, "score_map": score}

            return {
                "expert_outputs": {
                    "g": output(0.5, 0.5),
                    "m": output(0.3, 0.9),
                    "p": output(0.7, 0.8),
                },
                "expert_activation_mask": torch.tensor([
                    [True, True, False, False, False],
                    [True, False, True, False, False],
                ], device=kwargs["xi"].device),
                "presence_score": torch.ones(
                    batch_size, device=kwargs["xi"].device),
            }

    model = SparseExperts()
    actor = object.__new__(PETTrackActor)
    actor.net = model
    actor.settings = SimpleNamespace(search_area_factor={"search": 2.0})
    actor.cfg = SimpleNamespace(
        MODEL=SimpleNamespace(EXPERT=SimpleNamespace(
            ACTIVATOR_TRAINED=True,
            USE_ACTIVATION_INFERENCE=True,
        )),
        DATA=SimpleNamespace(SEARCH=SimpleNamespace(SIZE=8, FACTOR=2.0)),
    )
    frames = torch.zeros(2, 2, 3, 8, 8)
    boxes = torch.tensor([
        [[0.4, 0.4, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1]],
        [[0.5, 0.4, 0.1, 0.1], [0.5, 0.4, 0.1, 0.1]],
    ])
    data = {
        "template_images": torch.zeros(2, 2, 3, 4, 4),
        "template_event_images": torch.zeros(2, 2, 3, 4, 4),
        "pursuit_search_images": frames,
        "pursuit_search_event_images": frames,
        "pursuit_search_anno": boxes,
        "pursuit_search_present": torch.ones(2, 2, dtype=torch.uint8),
    }

    actor._forward_pursuit(data)

    assert model.inference_kwargs[0]["auto_activate"] is True
    call = model.search_window_controller.calls[0]
    assert call["expert_boxes"].shape == (2, 5, 4)
    assert torch.equal(call["expert_boxes"][0, 2:], call["expert_boxes"][0, :1].expand(3, -1))
    assert torch.equal(call["expert_boxes"][1, 1], call["expert_boxes"][1, 0])
    assert torch.equal(call["expert_boxes"][1, 3:], call["expert_boxes"][1, :1].expand(2, -1))
    assert torch.equal(
        call["response_peaks"] == 0.0,
        torch.tensor([
            [False, False, True, True, True],
            [False, True, False, True, True],
        ]),
    )
    assert torch.equal(
        call["response_psr"] == 0.0,
        torch.tensor([
            [False, False, True, True, True],
            [False, True, False, True, True],
        ]),
    )


def test_tracker_prefers_planned_search_state_only_in_local_tracking_modes():
    tracker = object.__new__(InferenceTracker)
    tracker.state = [10.0, 10.0, 5.0, 5.0]
    tracker._planned_search_state = [20.0, 20.0, 5.0, 5.0]
    tracker._pending_redetect_box = [30.0, 30.0, 5.0, 5.0]
    tracker.search_controller_enabled = True

    tracker._srbt_last_action = BeliefAction.TRACK
    assert tracker._search_state_for_frame() == tracker._planned_search_state
    tracker._srbt_last_action = BeliefAction.SUSPECT
    assert tracker._search_state_for_frame() == tracker._planned_search_state
    tracker._srbt_last_action = BeliefAction.ABSENT
    assert tracker._search_state_for_frame() == tracker._pending_redetect_box
    tracker._pending_redetect_box = None
    assert tracker._search_state_for_frame() == tracker.state


def test_planned_search_state_preserves_small_target_size():
    tracker = object.__new__(InferenceTracker)
    tracker.search_controller_enabled = True
    tracker.device = torch.device("cpu")
    tracker.state = [40.0, 30.0, 4.0, 5.0]
    tracker._search_controller_previous_box = None
    tracker._planned_search_state = None
    tracker.network = SimpleNamespace(
        expert_names=["g", "m", "p", "v", "d"],
        search_window_controller=SearchWindowController(
            expert_count=5, hidden_dim=16),
    )
    candidate = {
        "expert_states": [[40.0, 30.0, 4.0, 5.0] for _ in range(5)],
        "expert_peaks": [0.8] * 5,
        "expert_psr": [0.8] * 5,
        "presence_score": torch.tensor([0.9]),
    }

    planned = tracker._plan_next_search_state(
        torch.zeros(80, 100, 3), 80, 100, candidate, "track")

    assert planned[2:] == pytest.approx([4.0, 5.0])
    assert candidate["planned_search_state"] == pytest.approx(planned)


def test_pursuit_actor_updates_controller_but_not_frozen_experts():
    class FrozenExperts(torch.nn.Module):
        expert_names = ["g", "m", "p", "v", "d"]

        def __init__(self):
            super().__init__()
            self.expert_bias = torch.nn.Parameter(torch.tensor(0.0))
            self.search_window_controller = SearchWindowController(
                expert_count=5, hidden_dim=16)

        def inference(self, **kwargs):
            center = 0.5 + self.expert_bias.sigmoid() * 0.0
            box = torch.stack((center, center, center * 0.0 + 0.2,
                               center * 0.0 + 0.2)).reshape(1, 1, 4)
            score = torch.full(
                (1, 1, 4, 4), 0.8, device=kwargs["xi"].device)
            return {
                "expert_outputs": {
                    name: {"pred_boxes": box, "score_map": score}
                    for name in self.expert_names
                },
                "presence_score": torch.ones(1, device=box.device),
            }

    model = FrozenExperts()
    model.expert_bias.requires_grad_(False)
    actor = object.__new__(PETTrackActor)
    actor.net = model
    actor.expert_phase = "pursuit"
    actor.settings = SimpleNamespace(search_area_factor={"search": 2.0})
    actor.cfg = SimpleNamespace(
        DATA=SimpleNamespace(SEARCH=SimpleNamespace(SIZE=8, FACTOR=2.0)),
        TRAIN=SimpleNamespace(
            PURSUIT_CENTER_WEIGHT=1.0,
            PURSUIT_SCALE_WEIGHT=0.5,
            PURSUIT_CONTAINMENT_WEIGHT=2.0,
            PURSUIT_INSIDE_WEIGHT=0.5,
            PURSUIT_QUALITY_WEIGHT=0.25,
        ),
    )
    frames = torch.zeros(3, 1, 3, 8, 8)
    data = {
        "template_images": torch.zeros(2, 1, 3, 4, 4),
        "template_event_images": torch.zeros(2, 1, 3, 4, 4),
        "pursuit_search_images": frames,
        "pursuit_search_event_images": frames,
        "pursuit_search_anno": torch.tensor([
            [[0.10, 0.40, 0.10, 0.10]],
            [[0.35, 0.40, 0.10, 0.10]],
            [[0.60, 0.40, 0.10, 0.10]],
        ]),
        "pursuit_search_present": torch.ones(3, 1, dtype=torch.uint8),
    }
    controller_before = {
        name: value.detach().clone()
        for name, value in model.search_window_controller.named_parameters()
    }
    expert_before = model.expert_bias.detach().clone()
    optimizer = torch.optim.AdamW(
        model.search_window_controller.parameters(), lr=0.05)

    loss, status = actor(data)
    optimizer.zero_grad()
    loss.backward()
    assert model.expert_bias.grad is None
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool(parameter.grad.abs().sum() > 0)
        for parameter in model.search_window_controller.parameters()
    )
    optimizer.step()

    assert torch.equal(model.expert_bias, expert_before)
    assert any(
        not torch.equal(value, controller_before[name])
        for name, value in model.search_window_controller.named_parameters()
    )
    assert status["Pursuit/steps"] == 2
    assert 0.0 <= status["Pursuit/next_in_crop_rate"] <= 1.0
    assert status["Pursuit/next_center_error"] >= 0.0


def test_causal_specialist_forward_keeps_motion_gradient_and_freezes_controller():
    class FixedController(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

        def forward(self, **kwargs):
            next_box = kwargs["current_box"] + self.bias * 0.0
            return SimpleNamespace(
                next_box=next_box,
                inside_logit=next_box[:, 0] * 0.0,
                quality_logit=next_box[:, 0] * 0.0,
            )

    class CausalExperts(torch.nn.Module):
        expert_names = ["g", "m", "p", "v", "d"]

        def __init__(self):
            super().__init__()
            self.motion_bias = torch.nn.Parameter(torch.tensor(0.0))
            self.search_window_controller = FixedController()
            self.calls = []
            self.event_inputs = []
            self.motion_contexts = []

        def inference(self, **kwargs):
            self.calls.append(kwargs.get("active_expert_names"))
            self.event_inputs.append(kwargs["xe"].detach().clone())
            self.motion_contexts.append(kwargs.get("motion_context"))
            batch_size = kwargs["xi"].shape[0]

            def output(center):
                center = center.expand(batch_size)
                box = torch.stack((
                    center,
                    center * 0.0 + 0.5,
                    center * 0.0 + 0.2,
                    center * 0.0 + 0.2,
                ), dim=1).unsqueeze(1)
                score = center[:, None, None, None].expand(
                    batch_size, 1, 4, 4)
                return {"pred_boxes": box, "score_map": score}

            generalist = output(self.motion_bias.detach() * 0.0 + 0.5)
            motion = output(0.5 + 0.1 * self.motion_bias.tanh())
            return {
                "expert_outputs": {"g": generalist, "m": motion},
                "presence_score": torch.ones(batch_size),
            }

    model = CausalExperts()
    actor = object.__new__(PETTrackActor)
    actor.net = model
    actor.expert_enabled = True
    actor.expert_phase = "pursuit"
    actor.settings = SimpleNamespace(search_area_factor={"search": 2.0})
    actor.cfg = SimpleNamespace(
        MODEL=SimpleNamespace(EXPERT=SimpleNamespace(
            ACTIVATOR_TRAINED=False,
            USE_ACTIVATION_INFERENCE=False,
        )),
        DATA=SimpleNamespace(SEARCH=SimpleNamespace(SIZE=8, FACTOR=2.0)),
        TRAIN=SimpleNamespace(SPECIALIST_EXPERT_IDS=[1]),
    )
    frames = torch.zeros(3, 1, 3, 8, 8)
    event_frames = torch.stack((
        torch.full_like(frames[0], 1.0),
        torch.full_like(frames[0], 2.0),
        torch.full_like(frames[0], 3.0),
    ))
    challenge_labels = torch.zeros(
        3, 1, len(pet_track_actor_module.CHALLENGE_NAMES), dtype=torch.bool)
    challenge_labels[
        :, :, pet_track_actor_module.CHALLENGE_NAMES.index("motion")] = True
    data = {
        "template_images": torch.zeros(2, 1, 3, 4, 4),
        "template_event_images": torch.zeros(2, 1, 3, 4, 4),
        "pursuit_search_images": frames,
        "pursuit_search_event_images": event_frames,
        "pursuit_search_anno": torch.tensor([
            [[0.10, 0.40, 0.10, 0.10]],
            [[0.20, 0.40, 0.10, 0.10]],
            [[0.30, 0.40, 0.10, 0.10]],
        ]),
        "pursuit_search_present": torch.ones(3, 1, dtype=torch.uint8),
        "training_expert_id": torch.tensor([1]),
        "pursuit_challenge_labels": challenge_labels,
    }

    output = actor._forward_pursuit(data)
    specialist_outputs = output["pursuit_specialist_outputs"]
    loss = torch.stack([
        item["pred_boxes"].sum() for item in specialist_outputs
    ]).mean()
    loss.backward()

    assert model.calls == [("m",), ("m",)]
    assert model.motion_contexts[0] is not None
    assert not bool(model.motion_contexts[0]["history_valid"].any())
    assert bool(model.motion_contexts[1]["history_valid"].all())
    assert model.motion_contexts[0]["current_event"] is not \
        model.motion_contexts[1]["current_event"]
    assert torch.equal(
        model.motion_contexts[1]["previous_event"], model.event_inputs[0])
    assert torch.equal(
        model.motion_contexts[1]["current_event"], model.event_inputs[1])
    assert not torch.equal(model.event_inputs[1], event_frames[2])
    assert model.motion_bias.grad is not None
    assert bool(model.motion_bias.grad.abs() > 0)
    assert model.search_window_controller.bias.grad is None


def test_causal_specialist_total_loss_backpropagates_only_eligible_outputs(
        monkeypatch):
    motion_bias = torch.nn.Parameter(torch.tensor(0.25))

    def fake_base_loss(_self, output, gt_dict, return_status=True):
        present = gt_dict["search_absent"][-1].float()
        values = output["pred_boxes"][:, 0, 0]
        loss = (values * present).sum() / present.sum().clamp_min(1.0)
        return loss, {"Loss/total": float(loss.detach()), "IoU": 0.5}

    monkeypatch.setattr(PETTrackActor.__mro__[1], "compute_losses", fake_base_loss)
    actor = object.__new__(PETTrackActor)
    actor.settings = SimpleNamespace(search_area_factor={"search": 2.0})
    actor.cfg = SimpleNamespace(
        DATA=SimpleNamespace(SEARCH=SimpleNamespace(FACTOR=2.0)),
        TRAIN=SimpleNamespace(
            PURSUIT_CENTER_WEIGHT=1.0,
            PURSUIT_SCALE_WEIGHT=0.5,
            PURSUIT_CONTAINMENT_WEIGHT=2.0,
            PURSUIT_INSIDE_WEIGHT=0.5,
            PURSUIT_QUALITY_WEIGHT=0.25,
        ),
    )
    controller_step = SimpleNamespace(
        next_box=torch.tensor([[0.2, 0.2, 0.1, 0.1]]),
        inside_logit=torch.zeros(1),
        quality_logit=torch.zeros(1),
    )
    predictions = {
        "pursuit_predictions": [controller_step],
        "pursuit_targets": [torch.tensor([[0.3, 0.2, 0.1, 0.1]])],
        "pursuit_current_inside": [torch.tensor([True])],
        "pursuit_current_quality": [torch.tensor([0.5])],
        "pursuit_present_next": [torch.tensor([True])],
        "pursuit_specialist_id": 1,
        "pursuit_specialist_outputs": [{
            "pred_boxes": motion_bias.reshape(1, 1, 1).expand(1, 1, 4),
        }],
        "pursuit_specialist_targets": [
            torch.tensor([[0.2, 0.2, 0.1, 0.1]])],
        "pursuit_specialist_present": [torch.tensor([True])],
    }

    loss, status = actor._compute_pursuit_losses(predictions)
    loss.backward()

    assert motion_bias.grad is not None
    assert bool(motion_bias.grad.abs() > 0)
    assert status["Expert/train_count_1"] == 1
    assert status["Loss/causal_specialist"] == pytest.approx(float(loss.detach()))


def test_motion_displacement_loss_trains_consecutive_specialist_boxes(
        monkeypatch):
    motion_offset = torch.nn.Parameter(torch.tensor(0.0))

    def fake_base_loss(_self, output, _gt_dict, return_status=True):
        loss = output["pred_boxes"].sum() * 0.0
        return loss, {"Loss/total": float(loss.detach()), "IoU": 0.5}

    monkeypatch.setattr(PETTrackActor.__mro__[1], "compute_losses", fake_base_loss)
    actor = object.__new__(PETTrackActor)
    actor.settings = SimpleNamespace(search_area_factor={"search": 2.0})
    actor.cfg = SimpleNamespace(
        DATA=SimpleNamespace(SEARCH=SimpleNamespace(FACTOR=2.0)),
        TRAIN=SimpleNamespace(
            PURSUIT_CENTER_WEIGHT=1.0,
            PURSUIT_SCALE_WEIGHT=0.5,
            PURSUIT_CONTAINMENT_WEIGHT=2.0,
            PURSUIT_INSIDE_WEIGHT=0.5,
            PURSUIT_QUALITY_WEIGHT=0.25,
            MOTION_DISPLACEMENT_WEIGHT=2.0,
        ),
    )
    controller_step = SimpleNamespace(
        next_box=torch.tensor([[0.2, 0.2, 0.1, 0.1]]),
        inside_logit=torch.zeros(1),
        quality_logit=torch.zeros(1),
    )
    first = torch.tensor([[0.10, 0.40, 0.10, 0.10]])
    second = torch.stack((
        0.15 + motion_offset,
        0.40 + motion_offset * 0.0,
        0.10 + motion_offset * 0.0,
        0.10 + motion_offset * 0.0,
    )).reshape(1, 4)
    dummy_output = {
        "pred_boxes": (motion_offset * 0.0 + 0.5).expand(1, 1, 4),
    }
    predictions = {
        "pursuit_predictions": [controller_step, controller_step],
        "pursuit_targets": [first, second.detach()],
        "pursuit_current_inside": [
            torch.tensor([True]), torch.tensor([True])],
        "pursuit_current_quality": [
            torch.tensor([0.5]), torch.tensor([0.5])],
        "pursuit_present_next": [
            torch.tensor([True]), torch.tensor([True])],
        "pursuit_specialist_id": 1,
        "pursuit_specialist_outputs": [dummy_output, dummy_output],
        "pursuit_specialist_targets": [first, first],
        "pursuit_specialist_present": [
            torch.tensor([True]), torch.tensor([True])],
        "pursuit_specialist_image_boxes": [first, second],
        "pursuit_specialist_image_targets": [
            first, torch.tensor([[0.30, 0.40, 0.10, 0.10]])],
    }

    loss, status = actor._compute_pursuit_losses(predictions)
    loss.backward()

    assert status["MotionTrain/pair_count"] == 1
    assert status["Loss/motion_displacement"] > 0.0
    assert status["Loss/motion_displacement_weighted"] == pytest.approx(
        2.0 * status["Loss/motion_displacement"])
    assert motion_offset.grad is not None
    assert bool(motion_offset.grad.abs() > 0.0)


def test_synthetic_training_improves_next_center_and_crop_inclusion():
    torch.manual_seed(7)
    controller = SearchWindowController(expert_count=5, hidden_dim=16)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=0.02)
    previous = torch.tensor([
        [0.05, 0.40, 0.10, 0.10],
        [0.10, 0.40, 0.10, 0.10],
        [0.15, 0.40, 0.10, 0.10],
        [0.20, 0.40, 0.10, 0.10],
    ])
    current = previous.clone()
    current[:, 0] += 0.05
    target = current.clone()
    target[:, 0] += 0.25
    experts = current[:, None].repeat(1, 5, 1)
    peaks = torch.ones(4, 5)
    psr = torch.ones(4, 5)

    with torch.no_grad():
        before = controller(
            current_box=current, previous_box=previous,
            expert_boxes=experts, response_peaks=peaks, response_psr=psr,
            presence=torch.ones(4, 1), event_center=current[:, :2] + 0.05,
            event_confidence=torch.ones(4, 1)).next_box
        before_error = torch.linalg.vector_norm(
            (before[:, :2] + 0.5 * before[:, 2:])
            - (target[:, :2] + 0.5 * target[:, 2:]), dim=1).mean()
        before_inside = crop_target_inside(
            target, before, search_factor=2.0).float().mean()

    for _ in range(120):
        output = controller(
            current_box=current, previous_box=previous,
            expert_boxes=experts, response_peaks=peaks, response_psr=psr,
            presence=torch.ones(4, 1), event_center=current[:, :2] + 0.05,
            event_confidence=torch.ones(4, 1))
        loss, _ = search_window_pursuit_loss(
            output, target_next=target,
            current_inside=torch.ones(4, dtype=torch.bool),
            current_quality=torch.ones(4),
            present_next=torch.ones(4, dtype=torch.bool),
            search_factor=2.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        after = controller(
            current_box=current, previous_box=previous,
            expert_boxes=experts, response_peaks=peaks, response_psr=psr,
            presence=torch.ones(4, 1), event_center=current[:, :2] + 0.05,
            event_confidence=torch.ones(4, 1)).next_box
        after_error = torch.linalg.vector_norm(
            (after[:, :2] + 0.5 * after[:, 2:])
            - (target[:, :2] + 0.5 * target[:, 2:]), dim=1).mean()
        after_inside = crop_target_inside(
            target, after, search_factor=2.0).float().mean()

    assert after_error < before_error * 0.25
    assert before_inside.item() == pytest.approx(0.0)
    assert after_inside.item() == pytest.approx(1.0)


def test_controller_can_learn_global_event_jump_for_tiny_targets():
    torch.manual_seed(11)
    controller = SearchWindowController(
        expert_count=5, hidden_dim=32, max_center_step=1.0)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=0.02)
    current = torch.tensor([[0.49, 0.49, 0.02, 0.02]]).repeat(8, 1)
    previous = current.clone()
    target_centers = torch.tensor([
        [0.10, 0.10], [0.90, 0.10], [0.10, 0.90], [0.90, 0.90],
        [0.20, 0.50], [0.80, 0.50], [0.50, 0.20], [0.50, 0.80],
    ])
    target = torch.cat((
        target_centers - 0.01,
        torch.full((8, 2), 0.02),
    ), dim=1)
    experts = current[:, None].repeat(1, 5, 1)
    peaks = torch.full((8, 5), 0.05)
    psr = torch.full((8, 5), 0.05)

    for _ in range(180):
        output = controller(
            current_box=current,
            previous_box=previous,
            expert_boxes=experts,
            response_peaks=peaks,
            response_psr=psr,
            presence=torch.zeros(8, 1),
            event_center=target_centers,
            event_confidence=torch.ones(8, 1),
        )
        loss, _ = search_window_pursuit_loss(
            output,
            target_next=target,
            current_inside=torch.zeros(8, dtype=torch.bool),
            current_quality=torch.zeros(8),
            present_next=torch.ones(8, dtype=torch.bool),
            search_factor=4.0,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predicted = controller(
            current_box=current,
            previous_box=previous,
            expert_boxes=experts,
            response_peaks=peaks,
            response_psr=psr,
            presence=torch.zeros(8, 1),
            event_center=target_centers,
            event_confidence=torch.ones(8, 1),
        ).next_box
    inclusion = crop_target_inside(
        target, predicted, search_factor=4.0).float().mean()

    assert inclusion.item() >= 0.875
