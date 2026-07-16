import pytest
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from types import SimpleNamespace

from lib.models.layers.expert_fusion import ExpertFusionBank, build_expert_fusions
from lib.train.actors.pet_track import PETTrackActor
from lib.train.actors.pet_track_base import PETTrackBaseActor
from tests.srbt.test_srbt_model_integration import _cfg, TinyBackbone, TinyHead, TinyMemory
from lib.models.pet_track.pet_track import PETTrack


EXPERT_NAMES = (
    "generalist",
    "motion_fm",
    "small_target_st",
    "visibility_foc_ov",
    "discrimination_bi",
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
    experts = build_expert_fusions(EXPERT_NAMES, embed_dim=8)
    bank = ExpertFusionBank(experts, default_expert="generalist")
    rgb = torch.randn(2, 4, 8)
    event = torch.randn(2, 4, 8)
    template = torch.randn(2, 3, 8)

    outputs = [
        bank.forward_expert(
            name, rgb, event, context={"template_tokens": template})
        for name in EXPERT_NAMES
    ]

    assert all(output.shape == rgb.shape for output in outputs)
    assert len({type(module).__name__ for module in experts.values()}) == 5
    assert tuple(bank.experts) == EXPERT_NAMES


def test_high_resolution_context_changes_only_small_target_expert():
    torch.manual_seed(7)
    experts = build_expert_fusions(EXPERT_NAMES, embed_dim=8)
    bank = ExpertFusionBank(experts, default_expert="generalist")
    torch.nn.init.normal_(
        experts["small_target_st"].detail_adapter.net[-1].weight)
    rgb = torch.randn(1, 4, 8)
    event = torch.randn(1, 4, 8)
    detail = torch.randn(1, 4, 8)

    small_without = bank.forward_expert("small_target_st", rgb, event)
    small_with = bank.forward_expert(
        "small_target_st", rgb, event,
        context={"small_target_detail": detail})
    general_without = bank.forward_expert("generalist", rgb, event)
    general_with = bank.forward_expert(
        "generalist", rgb, event,
        context={"small_target_detail": detail})

    assert not torch.allclose(small_with, small_without)
    assert torch.allclose(general_with, general_without)


def test_small_target_detail_reuses_detached_half_patch_projection():
    class DetailBackbone(TinyBackbone):
        def __init__(self):
            super().__init__()
            self.patch_embed = SimpleNamespace(
                proj=torch.nn.Conv2d(3, 8, kernel_size=4, stride=4))

    cfg = _cfg()
    cfg.MODEL.EXPERT = edict({
        "ENABLE": True,
        "DEFAULT": "generalist",
        "NAMES": list(EXPERT_NAMES),
    })
    model = PETTrack(
        DetailBackbone(), TinyMemory(), TinyHead(), cfg, head_type="CENTER")
    rgb = torch.randn(2, 3, 8, 8)
    event = torch.randn(2, 3, 8, 8)

    detail = model._small_target_detail(rgb, event)

    assert detail.shape == (2, model.feat_len_s, 8)
    assert not detail.requires_grad


def test_non_small_owner_skips_high_resolution_extraction(monkeypatch):
    cfg = _cfg()
    cfg.MODEL.EXPERT = edict({
        "ENABLE": True,
        "DEFAULT": "generalist",
        "NAMES": list(EXPERT_NAMES),
    })
    model = PETTrack(
        TinyBackbone(), TinyMemory(), TinyHead(), cfg, head_type="CENTER")
    feature = torch.randn(1, 12, 8)
    images = (torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8))
    calls = []
    monkeypatch.setattr(
        model, "_small_target_detail",
        lambda *_: calls.append(True) or torch.randn(1, 4, 8),
    )

    model.forward_head(
        feature, expert_owner_ids=torch.tensor([0]), search_images=images)
    assert calls == []

    model.forward_head(
        feature, expert_owner_ids=torch.tensor([2]), search_images=images)
    assert calls == [True]


def test_inference_returns_independent_outputs_from_all_five_experts():
    model = _expert_model()

    output = model.forward_head(torch.randn(2, 12, 8))

    assert tuple(output["expert_outputs"]) == EXPERT_NAMES
    assert output["pred_boxes"] is output["expert_outputs"]["generalist"][
        "pred_boxes"]
    assert all(
        expert_output["pred_boxes"].shape == (2, 1, 4)
        for expert_output in output["expert_outputs"].values()
    )


def test_specialize_phase_passes_homogeneous_owner_ids_to_model():
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
        "expert_owner_id": torch.tensor([2, 2, 2]),
    }

    actor.forward_pass(data)

    assert torch.equal(
        actor.net.kwargs["expert_owner_ids"], torch.tensor([2, 2, 2]))
    assert "route" not in actor.net.kwargs


@pytest.mark.parametrize("owner_id, expects_recovery", [(2, False), (3, True)])
def test_only_visibility_owner_forwards_global_recovery(
        owner_id, expects_recovery):
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
        "expert_owner_id": torch.full((2,), owner_id),
        "is_reappear": torch.ones(1, 2),
        "redetect_search_images": torch.zeros(1, 2, 3, 8, 8),
        "redetect_search_event_images": torch.zeros(1, 2, 3, 8, 8),
    }

    actor.forward_pass(data)

    assert (actor.net.kwargs["redetect_mask"] is not None) is expects_recovery
    assert (actor.net.kwargs["redetect_images"] is not None) is expects_recovery
    assert (actor.net.kwargs["redetect_event_images"] is not None) is expects_recovery


def test_specialize_phase_reports_owner_iou_without_batch_dilution(monkeypatch):
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
    pred_dict = {
        "pred_boxes": torch.tensor([
            [[0.5, 0.5, 0.8, 0.8]],
            [[0.5, 0.5, 0.6, 0.6]],
        ], requires_grad=True),
    }
    gt_dict = {
        "search_anno": torch.tensor([[
            [0.1, 0.1, 0.8, 0.8],
            [0.2, 0.2, 0.6, 0.6],
        ]]),
        "search_absent": torch.ones(1, 2),
        "expert_owner_id": torch.tensor([0, 1]),
    }

    _, status = actor.compute_losses(pred_dict, gt_dict)

    assert status["Expert/owner_iou_0"] == pytest.approx(1.0)
    assert status["Expert/owner_iou_1"] == pytest.approx(1.0)


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

    base_boxes = torch.ones(2, 1, 4, requires_grad=True)
    presence_logits = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    redetect_signal = torch.tensor(2.0, requires_grad=True)
    actor._compute_redetect_loss = lambda predictions, data: (
        predictions["signal"], {"Loss/redetect": predictions["signal"].item()})
    pred_dict = {
        "pred_boxes": base_boxes,
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


def test_non_visibility_owner_does_not_train_presence_or_recovery(monkeypatch):
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

    base_boxes = torch.ones(2, 1, 4, requires_grad=True)
    presence_logits = torch.randn(2, 2, requires_grad=True)
    redetect_signal = torch.tensor(2.0, requires_grad=True)
    actor._compute_redetect_loss = lambda predictions, data: (
        predictions["signal"], {"Loss/redetect": predictions["signal"].item()})
    pred_dict = {
        "pred_boxes": base_boxes,
        "presence_predictions": {
            "logits": presence_logits,
            "score": presence_logits.softmax(dim=-1)[:, 1],
        },
        "redetect_predictions": {"signal": redetect_signal},
    }
    gt_dict = {
        "search_anno": torch.zeros(1, 2, 4),
        "search_absent": torch.tensor([[1, 0]]),
        "expert_owner_id": torch.full((2,), 2),
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


def _expert_model():
    cfg = _cfg()
    cfg.MODEL.EXPERT = edict({
        "ENABLE": True,
        "DEFAULT": "generalist",
        "NAMES": list(EXPERT_NAMES),
    })
    return PETTrack(
        TinyBackbone(), TinyMemory(), TinyHead(), cfg, head_type="CENTER")


def test_each_expert_has_an_independent_prediction_head():
    model = _expert_model()

    heads = [model._head_for_expert(name) for name in EXPERT_NAMES]

    assert heads[0] is model.box_head
    assert len({id(head) for head in heads}) == len(EXPERT_NAMES)
    parameter_ids = [
        {id(parameter) for parameter in head.parameters()} for head in heads
    ]
    assert all(parameter_ids[i].isdisjoint(parameter_ids[j])
               for i in range(5) for j in range(i + 1, 5))


def test_owner_batch_executes_one_fusion_and_one_head(monkeypatch):
    model = _expert_model()
    fusion_calls = {name: 0 for name in EXPERT_NAMES}
    head_calls = {name: 0 for name in EXPERT_NAMES}
    for name in EXPERT_NAMES:
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
        expert_owner_ids=torch.full((4,), 2),
    )

    assert output["expert_owner_id"].item() == 2
    assert fusion_calls == {
        "generalist": 0,
        "motion_fm": 0,
        "small_target_st": 1,
        "visibility_foc_ov": 0,
        "discrimination_bi": 0,
    }
    assert head_calls == fusion_calls


def test_specialist_batch_rejects_mixed_owner_ids():
    model = _expert_model()

    with pytest.raises(ValueError, match="exactly one expert owner"):
        model.forward_head(
            torch.randn(2, 12, 8),
            expert_owner_ids=torch.tensor([0, 1]),
        )


def test_non_owner_parameters_are_bitwise_unchanged_after_step():
    torch.manual_seed(11)
    model = _expert_model()
    owner_id = 2
    owner_name = EXPERT_NAMES[owner_id]
    trainable = []
    per_expert = {}
    for index, name in enumerate(EXPERT_NAMES):
        parameters = list(model.expert_fusion.experts[name].parameters())
        parameters.append(model.expert_fusion.residual_scale_logits[name])
        parameters.extend(model._head_for_expert(name).parameters())
        per_expert[index] = parameters
        trainable.extend(parameters)
    before = {
        id(parameter): parameter.detach().clone()
        for parameter in trainable
    }
    optimizer = torch.optim.SGD(trainable, lr=0.1)

    output = model.forward_head(
        torch.randn(3, 12, 8),
        expert_owner_ids=torch.full((3,), owner_id),
    )
    output["score_map"].sum().backward()

    for index, parameters in per_expert.items():
        if index != owner_id:
            assert all(parameter.grad is None for parameter in parameters)
    optimizer.step()
    for index, parameters in per_expert.items():
        if index != owner_id:
            assert all(torch.equal(parameter, before[id(parameter)])
                       for parameter in parameters)
    assert any(
        not torch.equal(parameter, before[id(parameter)])
        for parameter in per_expert[owner_id]
    ), owner_name
