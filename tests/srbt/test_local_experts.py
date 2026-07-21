import pytest
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from types import SimpleNamespace

import lib.models.layers.expert_fusion as expert_fusion_module
from lib.models.layers.expert_fusion import ExpertFusionBank, build_expert_fusions
from lib.train.actors.pet_track import PETTrackActor
from lib.train.actors.pet_track_base import PETTrackBaseActor
from tests.srbt.test_srbt_model_integration import (
    _cfg, _images, TinyBackbone, TinyHead, TinyMemory,
)
from lib.models.pet_track.pet_track import PETTrack
from lib.train.base_functions import _optimizer_groups
from lib.train.data.loader import ltr_collate_stack1
from lib.train.data.felt_challenges import CHALLENGE_NAMES
from lib.utils import TensorDict


EXPERT_NAMES = (
    "generalist",
    "motion_fm",
    "precision_refiner",
    "visibility_foc_ov",
    "discrimination_bi",
)
SHARED_EXPERT_NAMES = tuple(
    name for name in EXPERT_NAMES if name != "precision_refiner")
_EXPERT_CHALLENGE_INDEX = {1: 1, 2: 0, 3: 3, 4: 2}


def _challenge_labels(expert_id, batch_size):
    labels = torch.zeros(batch_size, 7, dtype=torch.bool)
    labels[:, _EXPERT_CHALLENGE_INDEX[expert_id]] = True
    return labels


def test_proposal_box_adapter_starts_as_exact_direct_prediction():
    adapter = expert_fusion_module.ProposalBoxAdapter()
    direct = torch.tensor([[[0.5, 0.4, 0.2, 0.1]]])
    upstream = torch.tensor([[[0.7, 0.6, 0.3, 0.2]]])
    score_map = torch.rand(1, 1, 4, 4)

    corrected, gate = adapter(direct, upstream, score_map)

    assert torch.equal(corrected, direct)
    assert torch.equal(gate, torch.zeros_like(gate))


def test_proposal_box_adapter_detaches_upstream_but_trains_correction():
    adapter = expert_fusion_module.ProposalBoxAdapter()
    with torch.no_grad():
        adapter.output.bias.fill_(0.5)
    direct = torch.tensor(
        [[[0.5, 0.4, 0.2, 0.1]]], requires_grad=True)
    upstream = torch.tensor(
        [[[0.7, 0.6, 0.3, 0.2]]], requires_grad=True)
    score_map = torch.rand(1, 1, 4, 4)

    corrected, _ = adapter(direct, upstream, score_map)
    corrected.sum().backward()

    assert direct.grad is not None
    assert upstream.grad is None
    assert adapter.output.bias.grad is not None


def test_proposal_box_adapter_rejects_incompatible_boxes():
    adapter = expert_fusion_module.ProposalBoxAdapter()

    with pytest.raises(ValueError, match="matching.*B, N, 4"):
        adapter(
            torch.rand(2, 1, 4),
            torch.rand(1, 1, 4),
            torch.rand(2, 1, 4, 4),
        )


def test_model_has_no_router_or_route_options():
    model = PETTrack(
        TinyBackbone(), TinyMemory(), TinyHead(), _cfg(expert_enabled=True),
        head_type="CENTER")

    for obsolete_name in (
            "expert_router", "route_options", "singleton_route_ids",
            "expert_temporal_momentum"):
        assert not hasattr(model, obsolete_name)


def test_expert_bank_uses_distinct_rgb_event_fusion_mechanisms():
    experts = build_expert_fusions(SHARED_EXPERT_NAMES, embed_dim=8)
    bank = ExpertFusionBank(experts, default_expert="generalist")
    rgb = torch.randn(2, 4, 8)
    event = torch.randn(2, 4, 8)
    template = torch.randn(2, 3, 8)

    outputs = [
        bank.forward_expert(
            name, rgb, event, context={"template_tokens": template})
        for name in SHARED_EXPERT_NAMES
    ]

    assert all(output.shape == rgb.shape for output in outputs)
    assert len({type(module).__name__ for module in experts.values()}) == 4
    assert tuple(bank.experts) == SHARED_EXPERT_NAMES


def test_motion_fusion_uses_only_valid_causal_event_history():
    torch.manual_seed(7)
    fusion = expert_fusion_module.MotionFusion(embed_dim=8).eval()
    rgb = torch.randn(1, 4, 8)
    event = torch.randn(1, 4, 8)
    current_event = torch.zeros(1, 3, 8, 8)
    current_event[:, :, 2, 2] = 4.0
    previous_left = torch.zeros_like(current_event)
    previous_left[:, :, 2, 1] = 4.0
    previous_right = torch.zeros_like(current_event)
    previous_right[:, :, 2, 6] = 4.0
    box_delta = torch.tensor([[0.25, 0.0, 0.0, 0.0]])

    baseline = fusion(rgb, event)
    with torch.no_grad():
        fusion.temporal_scale.fill_(1.0)
    invalid = fusion(rgb, event, context={
        "current_event": current_event,
        "previous_event": previous_left,
        "history_valid": torch.tensor([False]),
        "box_delta": box_delta,
    })
    from_left = fusion(rgb, event, context={
        "current_event": current_event,
        "previous_event": previous_left,
        "history_valid": torch.tensor([True]),
        "box_delta": box_delta,
    })
    from_right = fusion(rgb, event, context={
        "current_event": current_event,
        "previous_event": previous_right,
        "history_valid": torch.tensor([True]),
        "box_delta": box_delta,
    })

    assert torch.equal(invalid, baseline)
    assert not torch.allclose(from_left, baseline)
    assert not torch.allclose(from_left, from_right)


def test_motion_fusion_zero_gate_opens_then_trains_temporal_parameters():
    torch.manual_seed(11)
    fusion = expert_fusion_module.MotionFusion(embed_dim=8).train()
    optimizer = torch.optim.SGD(fusion.parameters(), lr=0.1)
    rgb = torch.randn(2, 4, 8)
    event = torch.randn(2, 4, 8)
    context = {
        "current_event": torch.randn(2, 3, 8, 8),
        "previous_event": torch.randn(2, 3, 8, 8),
        "history_valid": torch.tensor([True, True]),
        "box_delta": torch.randn(2, 4),
    }
    probe = torch.randn(2, 4, 8)

    baseline = fusion(rgb, event)
    gated = fusion(rgb, event, context=context)
    assert torch.equal(gated, baseline)
    (gated * probe).sum().backward()
    assert fusion.temporal_scale.grad.abs().item() > 1e-8
    optimizer.step()
    assert fusion.temporal_scale.detach().abs().item() > 1e-8

    optimizer.zero_grad(set_to_none=True)
    (fusion(rgb, event, context=context) * probe).sum().backward()
    temporal_parameters = [
        parameter
        for name, parameter in fusion.named_parameters()
        if name.startswith(("temporal_stem.", "temporal_box."))
    ]
    assert temporal_parameters
    assert any(
        parameter.grad is not None
        and parameter.grad.detach().abs().sum().item() > 0.0
        for parameter in temporal_parameters
    )


def test_small_target_expert_is_outside_shared_fusion_and_head_banks():
    model = _expert_model()

    assert tuple(model.expert_fusion.experts) == SHARED_EXPERT_NAMES
    assert "precision_refiner" not in model.expert_heads
    assert model.small_target_expert is not None


def test_precision_refiner_is_the_external_expert_name():
    model = _expert_model()

    assert model.precision_refiner_name == "precision_refiner"
    assert "precision_refiner" in model.expert_names
    assert "precision_refiner" in model.proposal_adapters
    assert "small_target_st" not in model.expert_names


def test_localization_loss_matches_high_resolution_expert_score_map():
    actor = object.__new__(PETTrackBaseActor)
    actor.cfg = edict({
        "DATA": {"SEARCH": {"SIZE": 16}},
        "MODEL": {"BACKBONE": {"STRIDE": 4}},
    })
    actor.loss_weight = {"giou": 2.0, "l1": 5.0, "focal": 1.0}
    observed = {}

    def focal(prediction, target):
        observed["target"] = target.detach().clone()
        return F.mse_loss(prediction, target)

    actor.objective = {"focal": focal}
    zero = torch.tensor(0.0, requires_grad=True)
    actor._box_losses = lambda *args, **kwargs: (
        zero, zero, torch.ones(1), None)
    prediction = {
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        "score_map": torch.zeros(1, 1, 8, 8, requires_grad=True),
    }
    target = {
        "search_anno": torch.tensor([[[0.4, 0.4, 0.2, 0.2]]]),
        "search_absent": torch.ones(1, 1),
    }

    loss, status = actor.compute_losses(prediction, target)

    assert torch.isfinite(loss)
    assert status["Loss/location"] >= 0.0
    assert observed["target"].shape == prediction["score_map"].shape
    assert observed["target"].amax().item() == pytest.approx(1.0)


def test_small_target_center_rank_loss_changes_only_specialist_loss():
    actor = object.__new__(PETTrackBaseActor)
    actor.cfg = edict({
        "DATA": {"SEARCH": {"SIZE": 16}},
        "MODEL": {"BACKBONE": {"STRIDE": 4}},
        "TRAIN": {
            "SMALL_TARGET_CENTER_RANK_WEIGHT": 4.0,
            "SMALL_TARGET_MATCH_RANK_WEIGHT": 2.0,
            "SMALL_TARGET_DENSE_SIZE_WEIGHT": 8.0,
            "SMALL_TARGET_DENSE_OFFSET_WEIGHT": 2.0,
            "SMALL_TARGET_SOFT_BOX_TEMPERATURE": 0.20,
        },
    })
    actor.loss_weight = {"giou": 2.0, "l1": 5.0, "focal": 1.0}
    actor.objective = {
        "focal": lambda prediction, target: F.mse_loss(prediction, target),
    }
    zero = torch.tensor(0.0, requires_grad=True)
    actor._box_losses = lambda *args, **kwargs: (
        zero, zero, torch.ones(2), None)
    prediction = {
        "pred_boxes": torch.tensor([
            [[0.5, 0.5, 0.1, 0.1]],
            [[0.2, 0.2, 0.2, 0.2]],
        ]),
        "small_base_pred_boxes": torch.tensor([
            [[0.45, 0.5, 0.1, 0.1]],
            [[0.2, 0.2, 0.2, 0.2]],
        ]),
        "small_box_delta": torch.tensor([
            [[0.05, 0.0, 0.0, 0.0]],
            [[0.5, -0.5, 0.25, -0.25]],
        ]),
        "score_map": torch.tensor([
            [[[0.1, 0.2], [0.3, 0.8]]],
            [[[0.7, 0.1], [0.2, 0.3]]],
        ], requires_grad=True),
        "small_match_map": torch.tensor([
            [[[0.2, 0.4], [0.6, 1.8]]],
            [[[1.4, 0.2], [0.4, 0.6]]],
        ], requires_grad=True),
        "size_map": torch.full(
            (2, 2, 2, 2), 0.5, requires_grad=True),
        "offset_map": torch.full(
            (2, 2, 2, 2), 0.25, requires_grad=True),
    }
    target = {
        "search_anno": torch.tensor([[[
            [0.45, 0.45, 0.1, 0.1],
            [-0.1, -0.1, 0.2, 0.2],
        ]]]).squeeze(0),
        "search_absent": torch.ones(1, 2),
        "training_expert_id": torch.tensor([2, 1]),
    }

    specialist_loss, status = actor.compute_losses(prediction, target)
    ordinary_target = dict(target)
    ordinary_target["training_expert_id"] = torch.tensor([1, 1])
    ordinary_loss, ordinary_status = actor.compute_losses(
        prediction, ordinary_target)

    assert specialist_loss.item() > ordinary_loss.item()
    assert status["Loss/small_center_rank"] > 0.0
    assert status["Loss/small_center_rank_weighted"] == pytest.approx(
        4.0 * status["Loss/small_center_rank"])
    assert status["Loss/small_match_rank"] > 0.0
    assert status["Loss/small_match_rank_weighted"] == pytest.approx(
        2.0 * status["Loss/small_match_rank"])
    assert status["Loss/small_dense_size"] > 0.0
    assert status["Loss/small_dense_offset"] > 0.0
    assert status["Loss/small_dense_size_weighted"] == pytest.approx(
        8.0 * status["Loss/small_dense_size"])
    assert status["Loss/small_dense_offset_weighted"] == pytest.approx(
        2.0 * status["Loss/small_dense_offset"])
    assert status["Loss/small_dense_geometry_weighted"] == pytest.approx(
        status["Loss/small_dense_size_weighted"]
        + status["Loss/small_dense_offset_weighted"])
    assert specialist_loss.item() - ordinary_loss.item() == pytest.approx(
        status["Loss/small_center_rank_weighted"]
        + status["Loss/small_match_rank_weighted"]
        + status["Loss/small_dense_geometry_weighted"])
    assert ordinary_status["Loss/small_center_rank"] == pytest.approx(0.0)
    assert ordinary_status["Loss/small_match_rank"] == pytest.approx(0.0)
    assert ordinary_status["Loss/small_dense_size"] == pytest.approx(0.0)
    assert ordinary_status["Loss/small_dense_offset"] == pytest.approx(0.0)
    assert ordinary_status[
        "Loss/small_dense_size_weighted"] == pytest.approx(0.0)
    assert ordinary_status[
        "Loss/small_dense_offset_weighted"] == pytest.approx(0.0)
    specialist_loss.backward()
    assert prediction["small_match_map"].grad is not None
    assert prediction["small_match_map"].grad.abs().sum() > 0.0
    assert prediction["size_map"].grad is not None
    assert prediction["size_map"].grad.abs().sum() > 0.0
    assert prediction["offset_map"].grad is not None
    assert prediction["offset_map"].grad.abs().sum() > 0.0
    assert ordinary_status.keys().isdisjoint({
        "SmallTargetTrain/count",
        "SmallTargetTrain/center_in_crop",
    })
    assert status["SmallTargetTrain/count"] == 1
    assert status["SmallTargetTrain/center_in_crop"] == pytest.approx(1.0)
    assert status["SmallTargetTrain/full_box_in_crop"] == pytest.approx(1.0)
    assert status["SmallTargetTrain/visible_fraction"] == pytest.approx(1.0)
    assert status["SmallTargetTrain/target_width_px"] == pytest.approx(1.6)
    assert status["SmallTargetTrain/target_height_px"] == pytest.approx(1.6)
    assert status["SmallTargetTrain/center_error_px"] == pytest.approx(0.0)
    assert status["SmallTargetTrain/size_error_px"] == pytest.approx(0.0)
    assert status["SmallTargetTrain/score_peak"] == pytest.approx(0.8)
    assert status["SmallTargetMatch/center_error_px"] == pytest.approx(0.0)
    assert status["SmallTargetMatch/peak"] == pytest.approx(1.8)
    assert status["SmallTargetMatch/gt_value"] == pytest.approx(1.8)
    assert status["SmallTargetMatch/gt_rank"] == pytest.approx(1.0)
    assert status["SmallTargetBox/base_iou"] == pytest.approx(1.0 / 3.0)
    assert status["SmallTargetBox/combined_iou_delta"] == pytest.approx(
        2.0 / 3.0)
    assert status["SmallTargetBox/delta_abs"] == pytest.approx(0.0125)
    assert status["SmallTargetBox/center_delta_abs"] == pytest.approx(0.025)
    assert status["SmallTargetBox/size_delta_abs"] == pytest.approx(0.0)


def test_small_target_center_rank_loss_prefers_gt_peak_and_backpropagates():
    actor = object.__new__(PETTrackBaseActor)
    wrong = torch.full((1, 1, 4, 4), 0.1, requires_grad=True)
    correct = torch.full((1, 1, 4, 4), 0.1)
    wrong.data[0, 0, 0, 0] = 0.9
    correct[0, 0, 3, 3] = 0.9
    gt_bbox = torch.tensor([[0.7, 0.7, 0.1, 0.1]])
    owners = torch.tensor([2])

    wrong_loss = actor._small_target_center_rank_loss(
        wrong, gt_bbox, owners, present_mask=None)
    correct_loss = actor._small_target_center_rank_loss(
        correct, gt_bbox, owners, present_mask=None)
    ordinary_loss = actor._small_target_center_rank_loss(
        wrong, gt_bbox, torch.tensor([0]), present_mask=None)

    assert wrong_loss > correct_loss
    assert ordinary_loss.item() == pytest.approx(0.0)
    wrong_loss.backward()
    assert wrong.grad is not None
    assert wrong.grad.abs().sum() > 0


def test_small_target_center_rank_loss_ignores_out_of_grid_rounded_center():
    actor = object.__new__(PETTrackBaseActor)
    score_map = torch.full((1, 1, 4, 4), 0.1, requires_grad=True)
    boundary_bbox = torch.tensor([[0.95, 0.95, 0.1, 0.1]])

    loss = actor._small_target_center_rank_loss(
        score_map, boundary_bbox, torch.tensor([2]), present_mask=None)

    assert loss.item() == pytest.approx(0.0)


def test_small_target_dense_geometry_loss_is_owner_specific_and_local():
    actor = object.__new__(PETTrackBaseActor)
    size_map = torch.full(
        (2, 2, 4, 4), 0.5, requires_grad=True)
    offset_map = torch.full(
        (2, 2, 4, 4), 0.25, requires_grad=True)
    gt_bbox = torch.tensor([
        [0.2, 0.3, 0.1, 0.2],
        [0.1, 0.1, 0.2, 0.2],
    ])

    size_loss, offset_loss = actor._small_target_dense_geometry_loss(
        size_map, offset_map, gt_bbox, torch.tensor([2, 0]))
    ordinary_size, ordinary_offset = (
        actor._small_target_dense_geometry_loss(
            size_map, offset_map, gt_bbox, torch.tensor([0, 0])))

    assert size_loss > 0.0
    assert offset_loss > 0.0
    assert ordinary_size.item() == pytest.approx(0.0)
    assert ordinary_offset.item() == pytest.approx(0.0)
    (size_loss + offset_loss).backward()
    assert torch.count_nonzero(size_map.grad[0]) == 2
    assert torch.count_nonzero(offset_map.grad[0]) == 2
    assert torch.count_nonzero(size_map.grad[0, :, 2, 1]) == 2
    assert torch.count_nonzero(offset_map.grad[0, :, 2, 1]) == 2
    assert torch.count_nonzero(size_map.grad[1]) == 0
    assert torch.count_nonzero(offset_map.grad[1]) == 0


def test_small_target_straight_through_boxes_keep_hard_values_and_spread_gradients():
    actor = object.__new__(PETTrackBaseActor)
    hard_boxes = torch.tensor(
        [[[0.0, 0.0, 0.2, 0.3]]], requires_grad=True)
    score_map = torch.tensor(
        [[[[0.8, 0.2], [0.1, 0.1]]]], requires_grad=True)
    size_map = torch.full(
        (1, 2, 2, 2), 0.25, requires_grad=True)
    offset_map = torch.zeros(
        (1, 2, 2, 2), requires_grad=True)

    train_boxes = actor._small_target_straight_through_boxes(
        hard_boxes,
        score_map,
        size_map,
        offset_map,
        torch.tensor([2]),
        temperature=0.1,
    )

    assert torch.equal(train_boxes.detach(), hard_boxes)
    train_boxes.sum().backward()
    assert hard_boxes.grad is not None
    assert torch.count_nonzero(hard_boxes.grad) == 4
    assert score_map.grad is not None
    assert score_map.grad.abs().sum() > 0.0
    assert size_map.grad is not None
    assert torch.count_nonzero(size_map.grad) > 2
    assert offset_map.grad is not None
    assert torch.count_nonzero(offset_map.grad) > 2


def test_shared_forward_head_returns_only_shared_expert_outputs():
    model = _expert_model()

    output = model.forward_head(torch.randn(2, 12, 8))

    assert tuple(output["expert_outputs"]) == SHARED_EXPERT_NAMES
    assert output["pred_boxes"] is output["expert_outputs"]["generalist"][
        "pred_boxes"]
    assert all(
        expert_output["pred_boxes"].shape == (2, 1, 4)
        for expert_output in output["expert_outputs"].values()
    )
    assert output["expert_outputs"]["generalist"]["score_map"].shape[-2:] == (2, 2)


def test_shared_experts_keep_direct_boxes_and_use_fixed_soft_dependencies():
    model = _expert_model()

    output = model.forward_head(torch.randn(2, 12, 8))["expert_outputs"]

    assert tuple(model.proposal_adapters) == (
        "motion_fm", "precision_refiner", "discrimination_bi")
    assert torch.equal(
        output["motion_fm"]["upstream_pred_boxes"],
        output["generalist"]["pred_boxes"],
    )
    assert torch.equal(
        output["discrimination_bi"]["upstream_pred_boxes"],
        output["motion_fm"]["pred_boxes"],
    )
    for name in ("motion_fm", "discrimination_bi"):
        assert torch.equal(
            output[name]["pred_boxes"], output[name]["direct_pred_boxes"])
        assert torch.count_nonzero(output[name]["proposal_gate"]) == 0
    assert "upstream_pred_boxes" not in output["visibility_foc_ov"]


def test_specialize_phase_passes_training_expert_for_exclusive_labels():
    class CaptureNet:
        expert_names = list(EXPERT_NAMES)

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"pred_boxes": torch.zeros(3, 1, 4)}

    actor = object.__new__(PETTrackActor)
    actor.net = CaptureNet()
    actor.cfg = edict({
        "MODEL": {"BACKBONE": {"CE_LOC": []}},
    })
    actor.settings = SimpleNamespace(num_template=1)
    actor.expert_enabled = True
    actor.expert_phase = "specialize"

    data = {
        "template_images": torch.zeros(1, 3, 3, 8, 8),
        "template_event_images": torch.zeros(1, 3, 3, 8, 8),
        "search_images": torch.zeros(1, 3, 3, 8, 8),
        "search_event_images": torch.zeros(1, 3, 3, 8, 8),
        "template_anno": torch.zeros(1, 3, 4),
        "training_expert_id": torch.tensor([2, 2, 2]),
        "challenge_labels": torch.tensor([
            [True, False, False, False, False, False, False],
        ]).repeat(3, 1),
    }

    actor.forward_pass(data)

    assert torch.equal(
        actor.net.kwargs["training_expert_ids"], torch.tensor([2, 2, 2]))
    assert "route" not in actor.net.kwargs


def test_specialize_phase_accepts_challenge_labels_from_training_loader():
    class CaptureNet:
        expert_names = list(EXPERT_NAMES)

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"pred_boxes": torch.zeros(3, 1, 4)}

    actor = object.__new__(PETTrackActor)
    actor.net = CaptureNet()
    actor.cfg = edict({"MODEL": {"BACKBONE": {"CE_LOC": []}}})
    actor.settings = SimpleNamespace(num_template=1)
    actor.expert_enabled = True
    actor.expert_phase = "specialize"

    sample = TensorDict({
        "template_images": torch.zeros(1, 3, 8, 8),
        "template_event_images": torch.zeros(1, 3, 8, 8),
        "search_images": torch.zeros(1, 3, 8, 8),
        "search_event_images": torch.zeros(1, 3, 8, 8),
        "template_anno": torch.zeros(1, 4),
        "training_expert_id": torch.tensor(2),
        "challenge_labels": torch.tensor(
            [True, False, False, False, False, False, False]),
    })
    data = ltr_collate_stack1([sample, sample, sample])

    assert data["challenge_labels"].shape == (7, 3)
    actor.forward_pass(data)

    assert torch.equal(
        actor.net.kwargs["training_expert_ids"], torch.tensor([2, 2, 2]))


def test_specialize_phase_rejects_compound_challenge_labels():
    actor = object.__new__(PETTrackActor)
    actor.net = SimpleNamespace(expert_names=list(EXPERT_NAMES))
    actor.expert_enabled = True
    actor.expert_phase = "specialize"
    data = {
        "training_expert_id": torch.tensor([2]),
        "challenge_labels": torch.tensor([
            [True, True, False, False, False, False, False],
        ]),
    }

    with pytest.raises(ValueError, match="not eligible for exclusive"):
        actor._validated_training_expert_ids(
            data, batch_size=1, device=torch.device("cpu"))


def test_specialize_phase_rejects_expert_not_eligible_for_challenge_labels():
    class CaptureNet:
        expert_names = list(EXPERT_NAMES)

        def __call__(self, **kwargs):
            return {"pred_boxes": torch.zeros(1, 1, 4)}

    actor = object.__new__(PETTrackActor)
    actor.net = CaptureNet()
    actor.cfg = edict({"MODEL": {"BACKBONE": {"CE_LOC": []}}})
    actor.settings = SimpleNamespace(num_template=1)
    actor.expert_enabled = True
    actor.expert_phase = "specialize"
    data = {
        "template_images": torch.zeros(1, 1, 3, 8, 8),
        "template_event_images": torch.zeros(1, 1, 3, 8, 8),
        "search_images": torch.zeros(1, 1, 3, 8, 8),
        "search_event_images": torch.zeros(1, 1, 3, 8, 8),
        "template_anno": torch.zeros(1, 1, 4),
        "training_expert_id": torch.tensor([2]),
        "challenge_labels": torch.tensor([
            [False, True, False, False, False, False, False],
        ]),
    }

    with pytest.raises(ValueError, match="not eligible"):
        actor.forward_pass(data)


@pytest.mark.parametrize("expert_id, expects_recovery", [(2, False), (3, True)])
def test_only_visibility_expert_forwards_global_recovery(
        expert_id, expects_recovery):
    class CaptureNet:
        expert_names = list(EXPERT_NAMES)

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"pred_boxes": torch.zeros(2, 1, 4)}

    actor = object.__new__(PETTrackActor)
    actor.net = CaptureNet()
    actor.cfg = edict({"MODEL": {"BACKBONE": {"CE_LOC": []}}})
    actor.settings = SimpleNamespace(num_template=1)
    actor.expert_enabled = True
    actor.expert_phase = "specialize"
    data = {
        "template_images": torch.zeros(1, 2, 3, 8, 8),
        "template_event_images": torch.zeros(1, 2, 3, 8, 8),
        "search_images": torch.zeros(1, 2, 3, 8, 8),
        "search_event_images": torch.zeros(1, 2, 3, 8, 8),
        "template_anno": torch.zeros(1, 2, 4),
        "training_expert_id": torch.full((2,), expert_id),
        "challenge_labels": _challenge_labels(expert_id, 2),
        "is_reappear": torch.ones(1, 2),
        "redetect_search_images": torch.zeros(1, 2, 3, 8, 8),
        "redetect_search_event_images": torch.zeros(1, 2, 3, 8, 8),
    }

    actor.forward_pass(data)

    assert (actor.net.kwargs["redetect_mask"] is not None) is expects_recovery
    assert (actor.net.kwargs["redetect_images"] is not None) is expects_recovery
    assert (actor.net.kwargs["redetect_event_images"] is not None) is expects_recovery


def test_specialize_phase_reports_training_expert_iou_without_dilution(monkeypatch):
    monkeypatch.setattr(
        PETTrackBaseActor,
        "compute_losses",
        lambda self, pred_dict, gt_dict, return_status=True: (
            pred_dict["pred_boxes"].sum() * 0.0, {}),
    )
    actor = object.__new__(PETTrackActor)
    actor.net = SimpleNamespace(expert_names=list(EXPERT_NAMES))
    actor.srbt_enabled = False
    actor.expert_enabled = True
    actor.expert_phase = "specialize"
    actor.expert_advantage_weight = 2.0
    actor.expert_advantage_margin = 0.10
    pred_dict = {
        "pred_boxes": torch.tensor([
            [[0.5, 0.5, 0.8, 0.8]],
            [[0.5, 0.5, 0.6, 0.6]],
        ], requires_grad=True),
        "upstream_pred_boxes": torch.tensor([
            [[0.5, 0.5, 0.8, 0.8]],
            [[0.5, 0.5, 0.6, 0.6]],
        ]),
    }
    gt_dict = {
        "search_anno": torch.tensor([[
            [0.1, 0.1, 0.8, 0.8],
            [0.2, 0.2, 0.6, 0.6],
        ]]),
        "search_absent": torch.ones(1, 2),
        "training_expert_id": torch.tensor([1, 1]),
        "challenge_labels": _challenge_labels(1, 2),
    }

    _, status = actor.compute_losses(pred_dict, gt_dict)

    assert status["Expert/train_count_1"] == 2
    assert status["Expert/train_iou_1"] == pytest.approx(1.0)
    assert status["Expert/train_count_0"] == 0


@pytest.mark.parametrize("expert_id", [1, 2, 4])
def test_specialist_advantage_loss_trains_active_expert_not_upstream(
        monkeypatch, expert_id):
    monkeypatch.setattr(
        PETTrackBaseActor,
        "compute_losses",
        lambda self, pred_dict, gt_dict, return_status=True: (
            pred_dict["pred_boxes"].sum() * 0.0, {}),
    )
    actor = object.__new__(PETTrackActor)
    actor.net = SimpleNamespace(expert_names=list(EXPERT_NAMES))
    actor.srbt_enabled = False
    actor.expert_enabled = True
    actor.expert_phase = "specialize"
    actor.expert_advantage_weight = 2.0
    actor.expert_advantage_margin = 0.10

    predicted = torch.tensor(
        [[[0.30, 0.30, 0.20, 0.20]]], requires_grad=True)
    upstream = torch.tensor(
        [[[0.50, 0.50, 0.20, 0.20]]], requires_grad=True)
    pred_dict = {
        "pred_boxes": predicted,
        "upstream_pred_boxes": upstream,
    }
    gt_dict = {
        "search_anno": torch.tensor([[[0.40, 0.40, 0.20, 0.20]]]),
        "search_absent": torch.ones(1, 1),
        "training_expert_id": torch.tensor([expert_id]),
        "challenge_labels": _challenge_labels(expert_id, 1),
    }

    loss, status = actor.compute_losses(pred_dict, gt_dict)
    loss.backward()

    expert_name = EXPERT_NAMES[expert_id]
    assert status["Loss/expert_advantage"] > 0.0
    assert status[f"Loss/expert_advantage_{expert_name}"] > 0.0
    assert status["Loss/expert_advantage_weighted"] == pytest.approx(
        2.0 * status["Loss/expert_advantage"])
    assert predicted.grad is not None
    assert predicted.grad.abs().sum() > 0.0
    assert upstream.grad is None


def test_visibility_expert_does_not_use_proposal_advantage(monkeypatch):
    monkeypatch.setattr(
        PETTrackBaseActor,
        "compute_losses",
        lambda self, pred_dict, gt_dict, return_status=True: (
            pred_dict["pred_boxes"].sum() * 0.0, {}),
    )
    actor = object.__new__(PETTrackActor)
    actor.net = SimpleNamespace(expert_names=list(EXPERT_NAMES))
    actor.srbt_enabled = False
    actor.expert_enabled = True
    actor.expert_phase = "specialize"
    actor.expert_advantage_weight = 2.0
    actor.expert_advantage_margin = 0.10
    predicted = torch.tensor(
        [[[0.30, 0.30, 0.20, 0.20]]], requires_grad=True)
    pred_dict = {"pred_boxes": predicted}
    gt_dict = {
        "search_anno": torch.tensor([[[0.40, 0.40, 0.20, 0.20]]]),
        "search_absent": torch.ones(1, 1),
        "training_expert_id": torch.tensor([3]),
        "challenge_labels": _challenge_labels(3, 1),
    }

    loss, status = actor.compute_losses(pred_dict, gt_dict)

    assert loss.item() == pytest.approx(0.0)
    assert status["Loss/expert_advantage"] == pytest.approx(0.0)
    assert status["Loss/expert_advantage_weighted"] == pytest.approx(0.0)


def test_dispatch_targets_keep_multilabel_eligibility_but_require_utility():
    actor = object.__new__(PETTrackActor)
    actor.net = SimpleNamespace(expert_names=list(EXPERT_NAMES))
    actor.expert_phase = "dispatch"
    actor.activation_advantage_margin = 0.02
    actor.activation_pos_weight = (4.0, 5.0, 1.5, 2.5)
    logits = torch.zeros(2, 4, requires_grad=True)
    generalist = torch.tensor([
        [[0.70, 0.70, 0.20, 0.20]],
        [[0.50, 0.50, 0.20, 0.20]],
    ])
    exact = torch.tensor([
        [[0.50, 0.50, 0.20, 0.20]],
        [[0.50, 0.50, 0.20, 0.20]],
    ])
    pred_dict = {
        "expert_activation_logits": logits,
        "expert_outputs": {
            "generalist": {"pred_boxes": generalist},
            "motion_fm": {"pred_boxes": exact},
            "precision_refiner": {"pred_boxes": generalist.clone()},
            "visibility_foc_ov": {"pred_boxes": exact.clone()},
            "discrimination_bi": {"pred_boxes": exact.clone()},
        },
    }
    challenge_labels = torch.zeros(2, len(CHALLENGE_NAMES), dtype=torch.bool)
    challenge_labels[0, CHALLENGE_NAMES.index("motion")] = True
    challenge_labels[0, CHALLENGE_NAMES.index("small_target")] = True
    challenge_labels[1, CHALLENGE_NAMES.index("absent")] = True
    gt_dict = {
        "challenge_labels": challenge_labels,
        "search_anno": torch.tensor([[
            [0.40, 0.40, 0.20, 0.20],
            [0.00, 0.00, 0.00, 0.00],
        ]]),
        "search_absent": torch.tensor([[1, 0]]),
    }

    targets = actor._dispatch_targets(pred_dict, gt_dict)
    loss, status = actor.compute_losses(pred_dict, gt_dict)
    loss.backward()

    assert torch.equal(targets, torch.tensor([
        [True, False, False, False],
        [False, False, True, False],
    ]))
    assert status["Activation/positive_motion_fm"] == 1
    assert status["Activation/positive_precision_refiner"] == 0
    assert status["Activation/positive_visibility_foc_ov"] == 1
    assert status["Activation/positive_discrimination_bi"] == 0
    assert status["Activation/predicted_count"] == 8
    assert status["Activation/mean_specialists"] == pytest.approx(4.0)
    assert status["Activation/exact_match"] == pytest.approx(0.0)
    assert status["Activation/precision_motion_fm"] == pytest.approx(0.5)
    assert status["Activation/recall_motion_fm"] == pytest.approx(1.0)
    assert status["Activation/precision_visibility_foc_ov"] == pytest.approx(
        0.5)
    assert status["Activation/recall_visibility_foc_ov"] == pytest.approx(1.0)
    assert status["Activation/macro_f1"] == pytest.approx(1.0 / 3.0)
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0.0


def test_dispatch_training_keeps_frozen_modules_in_eval_mode():
    class DispatchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(2, 2),
                torch.nn.BatchNorm1d(2),
            )
            self.expert_activator = torch.nn.Linear(2, 4)

    actor = object.__new__(PETTrackActor)
    actor.net = DispatchModel()
    actor.expert_phase = "dispatch"

    actor.train(True)

    assert actor.net.training is False
    assert actor.net.backbone.training is False
    assert actor.net.expert_activator.training is True

    actor.train(False)
    assert actor.net.training is False
    assert actor.net.expert_activator.training is False


def test_dispatch_loss_uses_configured_positive_class_weights(monkeypatch):
    actor = object.__new__(PETTrackActor)
    actor.net = SimpleNamespace(expert_names=list(EXPERT_NAMES))
    actor.expert_phase = "dispatch"
    actor.activation_pos_weight = (4.0, 5.0, 1.5, 2.5)
    logits = torch.zeros(2, 4, requires_grad=True)
    targets = torch.tensor([
        [True, False, False, False],
        [False, True, True, False],
    ])
    monkeypatch.setattr(actor, "_dispatch_targets", lambda pred, gt: targets)

    loss, _ = actor._compute_dispatch_loss(
        {"expert_activation_logits": logits}, {}, return_status=True)
    expected = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(logits.dtype),
        pos_weight=logits.new_tensor(actor.activation_pos_weight),
    )

    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("expert_id", [1, 2, 4])
def test_specialist_advantage_requires_frozen_upstream_box(
        monkeypatch, expert_id):
    monkeypatch.setattr(
        PETTrackBaseActor,
        "compute_losses",
        lambda self, pred_dict, gt_dict, return_status=True: (
            pred_dict["pred_boxes"].sum() * 0.0, {}),
    )
    actor = object.__new__(PETTrackActor)
    actor.net = SimpleNamespace(expert_names=list(EXPERT_NAMES))
    actor.srbt_enabled = False
    actor.expert_enabled = True
    actor.expert_phase = "specialize"
    actor.expert_advantage_weight = 2.0
    actor.expert_advantage_margin = 0.10
    pred_dict = {
        "pred_boxes": torch.tensor(
            [[[0.30, 0.30, 0.20, 0.20]]], requires_grad=True),
    }
    gt_dict = {
        "search_anno": torch.tensor([[[0.40, 0.40, 0.20, 0.20]]]),
        "search_absent": torch.ones(1, 1),
        "training_expert_id": torch.tensor([expert_id]),
        "challenge_labels": _challenge_labels(expert_id, 1),
    }

    with pytest.raises(RuntimeError, match="upstream_pred_boxes"):
        actor.compute_losses(pred_dict, gt_dict)


def test_recovery_phase_ignores_base_and_challenge_losses(monkeypatch):
    monkeypatch.setattr(
        PETTrackBaseActor,
        "compute_losses",
        lambda self, pred_dict, gt_dict, return_status=True: (
            pred_dict["pred_boxes"].sum(), {}),
    )
    actor = object.__new__(PETTrackActor)
    actor.net = SimpleNamespace(expert_names=list(EXPERT_NAMES))
    actor.srbt_enabled = True
    actor.expert_enabled = True
    actor.expert_phase = "recovery"
    actor.presence_loss_weight = 1.0
    actor.presence_focal_gamma = 2.0
    actor.presence_present_threshold = 0.70
    actor.presence_recover_threshold = 0.75

    base_boxes = torch.ones(2, 1, 4, requires_grad=True)
    presence_logits = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    redetect_signal = torch.tensor(2.0, requires_grad=True)
    actor._compute_redetect_loss = lambda predictions, data: (
        predictions["signal"], {"Loss/redetect": predictions["signal"].item()})
    pred_dict = {
        "pred_boxes": base_boxes,
        "upstream_pred_boxes": base_boxes.detach().clone(),
        "presence_predictions": {
            "logits": presence_logits,
            "score": presence_logits.softmax(dim=-1)[:, 1],
        },
        "redetect_predictions": {"signal": redetect_signal},
    }
    gt_dict = {
        "search_anno": torch.zeros(1, 2, 4),
        "search_absent": torch.tensor([[1, 0]]),
    }

    loss, status = actor.compute_losses(pred_dict, gt_dict)
    loss.backward()

    assert torch.equal(base_boxes.grad, torch.zeros_like(base_boxes))
    assert presence_logits.grad is not None
    assert presence_logits.grad.abs().sum() > 0
    assert redetect_signal.grad == pytest.approx(1.0)
    assert status["Expert/phase_id"] == 2


def test_presence_loss_enforces_controller_thresholds(monkeypatch):
    monkeypatch.setattr(
        PETTrackBaseActor,
        "compute_losses",
        lambda self, pred_dict, gt_dict, return_status=True: (
            pred_dict["pred_boxes"].sum() * 0.0, {}),
    )
    actor = object.__new__(PETTrackActor)
    actor.net = SimpleNamespace(expert_names=list(EXPERT_NAMES))
    actor.srbt_enabled = True
    actor.expert_enabled = True
    actor.expert_phase = "specialize"
    actor.presence_loss_weight = 1.0
    actor.presence_focal_gamma = 2.0
    actor.presence_present_threshold = 0.70
    actor.presence_recover_threshold = 0.75

    probabilities = torch.tensor([0.72, 0.72, 0.65])
    logits = torch.stack(
        (1.0 - probabilities, probabilities), dim=-1
    ).log().requires_grad_()
    pred_dict = {
        "pred_boxes": torch.zeros(3, 1, 4, requires_grad=True),
        "presence_predictions": {
            "logits": logits,
            "score": probabilities,
        },
    }
    gt_dict = {
        "search_anno": torch.zeros(1, 3, 4),
        "search_absent": torch.tensor([[1, 1, 0]]),
        "is_reappear": torch.tensor([[0, 1, 0]]),
        "training_expert_id": torch.full((3,), 3),
        "challenge_labels": _challenge_labels(3, 3),
    }

    _, status = actor.compute_losses(pred_dict, gt_dict)

    assert status["Loss/presence_threshold"] == pytest.approx(0.01)


def test_non_visibility_expert_does_not_train_presence_or_recovery(monkeypatch):
    monkeypatch.setattr(
        PETTrackBaseActor,
        "compute_losses",
        lambda self, pred_dict, gt_dict, return_status=True: (
            pred_dict["pred_boxes"].sum(), {}),
    )
    actor = object.__new__(PETTrackActor)
    actor.net = SimpleNamespace(expert_names=list(EXPERT_NAMES))
    actor.srbt_enabled = True
    actor.expert_enabled = True
    actor.expert_phase = "specialize"
    actor.presence_loss_weight = 1.0
    actor.presence_focal_gamma = 2.0
    actor.expert_advantage_weight = 2.0
    actor.expert_advantage_margin = 0.10

    base_boxes = torch.ones(2, 1, 4, requires_grad=True)
    presence_logits = torch.randn(2, 2, requires_grad=True)
    redetect_signal = torch.tensor(2.0, requires_grad=True)
    actor._compute_redetect_loss = lambda predictions, data: (
        predictions["signal"], {"Loss/redetect": predictions["signal"].item()})
    pred_dict = {
        "pred_boxes": base_boxes,
        "upstream_pred_boxes": base_boxes.detach().clone(),
        "presence_predictions": {
            "logits": presence_logits,
            "score": presence_logits.softmax(dim=-1)[:, 1],
        },
        "redetect_predictions": {"signal": redetect_signal},
    }
    gt_dict = {
        "search_anno": torch.zeros(1, 2, 4),
        "search_absent": torch.tensor([[1, 0]]),
        "training_expert_id": torch.full((2,), 2),
        "challenge_labels": _challenge_labels(2, 2),
    }

    loss, status = actor.compute_losses(pred_dict, gt_dict)
    loss.backward()

    assert base_boxes.grad.abs().sum() > 0
    assert presence_logits.grad is None
    assert redetect_signal.grad is None
    assert status["Loss/presence"] == pytest.approx(0.0)
    assert status["Loss/redetect"] == pytest.approx(0.0)


def test_recovery_loss_trains_localization_and_rgb_identity():
    actor = object.__new__(PETTrackActor)
    actor.cfg = edict({
        "DATA": {"SEARCH": {"SIZE": 16}},
        "MODEL": {"BACKBONE": {"STRIDE": 4}},
    })
    actor.objective = {
        "focal": lambda prediction, target: F.mse_loss(prediction, target),
    }
    actor.loss_weight = {"focal": 1.0, "l1": 5.0, "giou": 2.0}
    actor.identity_loss_weight = 1.0
    actor.identity_ranking_weight = 0.5
    actor.identity_ranking_margin = 0.2
    score_map = torch.zeros(2, 1, 4, 4, requires_grad=True)
    boxes = torch.tensor([
        [0.45, 0.45, 0.25, 0.25],
        [0.55, 0.55, 0.25, 0.25],
    ], requires_grad=True)
    identity_scores = torch.tensor([
        [0.7, 0.6],
        [0.5, 0.6],
    ], requires_grad=True)
    predictions = {
        "batch_indices": torch.tensor([0, 1]),
        "score_map": score_map,
        "bbox": boxes,
        "identity_scores": identity_scores,
    }
    data = {
        "redetect_search_anno": torch.tensor([[
            [0.30, 0.30, 0.25, 0.25],
            [0.40, 0.40, 0.25, 0.25],
        ]]),
    }

    loss, status = actor._compute_redetect_loss(predictions, data)
    loss.backward()

    assert score_map.grad.abs().sum() > 0
    assert boxes.grad.abs().sum() > 0
    assert identity_scores.grad.abs().sum() > 0
    assert status["Loss/redetect_giou"] > 0.0
    assert status["Loss/recovery_identity"] > 0.0
    assert status["Loss/recovery_ranking"] > 0.0


class _BatchNormTinyHead(TinyHead):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(8)

    def forward(self, feat, gt_score_map):
        return super().forward(self.norm(feat), gt_score_map)


def _expert_model(box_head=None):
    cfg = _cfg()
    cfg.MODEL.EXPERT = edict({
        "ENABLE": True,
        "DEFAULT": "generalist",
        "NAMES": list(EXPERT_NAMES),
    })
    return PETTrack(
        TinyBackbone(), TinyMemory(), box_head or TinyHead(),
        cfg, head_type="CENTER")


def test_detached_parent_forward_keeps_generalist_batch_norm_buffers_frozen():
    model = _expert_model(_BatchNormTinyHead())
    model.train()
    before = {
        name: value.detach().clone()
        for name, value in model.box_head.named_buffers()
    }

    model.forward_head(
        torch.randn(4, 12, 8),
        training_expert_ids=torch.full((4,), 1),
    )

    current = dict(model.box_head.named_buffers())
    assert all(torch.equal(current[name], value)
               for name, value in before.items())


def test_shared_experts_have_independent_heads_and_small_expert_is_disjoint():
    model = _expert_model()

    heads = [model._head_for_expert(name) for name in SHARED_EXPERT_NAMES]

    assert heads[0] is model.box_head
    assert len({id(head) for head in heads}) == len(SHARED_EXPERT_NAMES)
    parameter_ids = [
        {id(parameter) for parameter in head.parameters()} for head in heads
    ]
    assert all(parameter_ids[i].isdisjoint(parameter_ids[j])
               for i in range(4) for j in range(i + 1, 4))
    small_parameter_ids = {
        id(parameter) for parameter in model.small_target_expert.parameters()
    }
    assert all(small_parameter_ids.isdisjoint(ids) for ids in parameter_ids)


def test_motion_expert_executes_generalist_dependency_and_motion_only(monkeypatch):
    model = _expert_model()
    fusion_calls = {name: 0 for name in SHARED_EXPERT_NAMES}
    head_calls = {name: 0 for name in SHARED_EXPERT_NAMES}
    for name in SHARED_EXPERT_NAMES:
        fusion = model.expert_fusion.experts[name]
        fusion_forward = fusion.forward
        monkeypatch.setattr(
            fusion,
            "forward",
            lambda *args, _name=name, _forward=fusion_forward, **kwargs: (
                fusion_calls.__setitem__(_name, fusion_calls[_name] + 1)
                or _forward(*args, **kwargs)
            ),
        )
        head = model._head_for_expert(name)
        head_forward = head.forward
        monkeypatch.setattr(
            head,
            "forward",
            lambda *args, _name=name, _forward=head_forward, **kwargs: (
                head_calls.__setitem__(_name, head_calls[_name] + 1)
                or _forward(*args, **kwargs)
            ),
        )

    output = model.forward_head(
        torch.randn(4, 12, 8),
        training_expert_ids=torch.full((4,), 1),
    )

    assert "expert_owner_id" not in output
    assert fusion_calls == {
        "generalist": 1,
        "motion_fm": 1,
        "visibility_foc_ov": 0,
        "discrimination_bi": 0,
    }
    assert head_calls == fusion_calls


def test_specialist_batch_rejects_mixed_training_expert_ids():
    model = _expert_model()

    with pytest.raises(ValueError, match="exactly one training expert"):
        model.forward_head(
            torch.randn(2, 12, 8),
            training_expert_ids=torch.tensor([1, 2]),
        )


def test_precision_expert_rejects_shared_head_without_owner_terminology():
    model = _expert_model()

    with pytest.raises(RuntimeError, match="precision expert must bypass"):
        model.forward_head(
            torch.randn(2, 12, 8),
            training_expert_ids=torch.full((2,), 2),
        )


@pytest.mark.parametrize("expert_id, allowed_prefixes", (
    (1, (
        "expert_fusion.experts.motion_fm.",
        "expert_fusion.residual_scale_logits.motion_fm",
        "expert_heads.motion_fm.",
        "proposal_adapters.motion_fm.",
    )),
    (2, (
        "small_target_expert.",
        "proposal_adapters.precision_refiner.",
    )),
    (3, (
        "expert_fusion.experts.visibility_foc_ov.",
        "expert_fusion.residual_scale_logits.visibility_foc_ov",
        "expert_heads.visibility_foc_ov.",
        "visibility_gate.",
    )),
    (4, (
        "expert_fusion.experts.discrimination_bi.",
        "expert_fusion.residual_scale_logits.discrimination_bi",
        "expert_heads.discrimination_bi.",
        "proposal_adapters.discrimination_bi.",
    )),
))
def test_non_selected_expert_parameters_are_bitwise_unchanged_after_step(
        expert_id, allowed_prefixes):
    torch.manual_seed(11 + expert_id)
    model = _expert_model()
    model.cfg.TRAIN = edict({
        "EXPERT_PHASE": "specialize",
        "SPECIALIST_EXPERT_IDS": [1, 2, 3, 4],
        "SMALL_TARGET_ADAPTER_LR": 0.0,
        "LR": 1e-3,
        "WEIGHT_DECAY": 0.1,
        "EXPERT_LR_MULTIPLIER": 5.0,
    })
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, model.cfg), lr=1e-3, weight_decay=0.1)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    output = model(
        *_images(batch=3),
        training_expert_ids=torch.full((3,), expert_id),
    )
    loss = output["score_map"].mean() + output["pred_boxes"].mean()
    if expert_id == 3:
        loss = loss + output["presence_predictions"]["logits"].square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    changed = [
        name for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    ]
    assert changed
    assert all(name.startswith(allowed_prefixes) for name in changed)
