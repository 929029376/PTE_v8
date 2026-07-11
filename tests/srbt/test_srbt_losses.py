import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
import pytest
from torch import nn

from lib.config.pet_track.config import cfg, update_config_from_file
from lib.models.layers.pet_losses import PetTrackLoss
from lib.train.actors.pet_track import PETTrackActor
from lib.train.actors.pet_track_base import PETTrackBaseActor


def _one_hot_field(batch=2, grid=4):
    target = torch.zeros(batch, 1, grid, grid)
    target[0, 0, 1, 2] = 1.0
    target[1, 0, 3, 0] = 1.0
    return target


def test_discrete_survival_uses_event_likelihood_and_censor_tail_mass():
    probabilities = torch.zeros(2, 129)
    probabilities[0, [0, 1, 2, 128]] = torch.tensor([0.1, 0.7, 0.1, 0.1])
    probabilities[1, [0, 1, 2, 128]] = torch.tensor([0.2, 0.3, 0.1, 0.4])
    predictions = {"hazard_logits": probabilities.log().requires_grad_()}
    targets = {
        "hazard_target": torch.tensor([2, 2]),
        "hazard_mask": torch.tensor([True, True]),
        "censor_mask": torch.tensor([False, True]),
    }

    loss, stats = PetTrackLoss().srbt_loss(predictions, targets)

    expected = (-torch.log(torch.tensor(0.7))
                - torch.log(torch.tensor(0.5))) / 2
    assert torch.allclose(loss, expected, atol=1e-6)
    assert abs(stats["Loss/srbt_survival"] - expected.item()) < 1e-6


def test_survival_rejects_noncanonical_hazard_bins():
    with pytest.raises(ValueError, match="129 hazard bins"):
        PetTrackLoss().srbt_loss({
            "hazard_logits": torch.zeros(1, 128),
        }, {
            "hazard_target": torch.tensor([2]),
            "hazard_mask": torch.tensor([True]),
            "censor_mask": torch.tensor([False]),
        })


def test_field_loss_ignores_frames_without_spatial_supervision():
    target = _one_hot_field()
    targets = {
        "field_target": target,
        "field_mask": torch.tensor([True, False]),
    }
    first = torch.zeros(2, 1, 4, 4, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    second.data[1] = 1000.0

    loss_a, _ = PetTrackLoss().srbt_loss(
        {"field_logits": first}, targets)
    loss_b, _ = PetTrackLoss().srbt_loss(
        {"field_logits": second}, targets)

    assert torch.allclose(loss_a, loss_b)


def test_best_of_k_localization_and_diversity_are_both_reported():
    target_box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    diverse = torch.tensor([[[0.5, 0.5, 0.2, 0.2],
                             [0.1, 0.1, 0.2, 0.2]]], requires_grad=True)
    duplicate = torch.tensor([[[0.5, 0.5, 0.2, 0.2],
                               [0.5, 0.5, 0.2, 0.2]]], requires_grad=True)
    targets = {
        "target_box": target_box,
        "hypothesis_mask": torch.tensor([True]),
    }

    _, diverse_stats = PetTrackLoss().srbt_loss(
        {"hypothesis_boxes": diverse}, targets)
    _, duplicate_stats = PetTrackLoss().srbt_loss(
        {"hypothesis_boxes": duplicate}, targets)

    assert diverse_stats["Loss/srbt_hypothesis_localization"] == 0.0
    assert (duplicate_stats["Loss/srbt_hypothesis_diversity"]
            > diverse_stats["Loss/srbt_hypothesis_diversity"])


def test_hypothesis_calibration_ignores_frames_without_spatial_supervision():
    boxes = torch.tensor([
        [[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2]],
        [[0.8, 0.8, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]],
    ])
    targets = {
        "target_box": torch.tensor([
            [0.5, 0.5, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0],
        ]),
        "hypothesis_mask": torch.tensor([True, False]),
    }
    first = torch.zeros(2, 2, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    second.data[1] = torch.tensor([1000.0, -1000.0])

    loss_a, stats_a = PetTrackLoss().srbt_loss({
        "hypothesis_boxes": boxes,
        "hypothesis_scores": first,
    }, targets)
    loss_b, stats_b = PetTrackLoss().srbt_loss({
        "hypothesis_boxes": boxes,
        "hypothesis_scores": second,
    }, targets)

    assert torch.allclose(loss_a, loss_b)
    assert stats_a["Loss/srbt_calibration"] == pytest.approx(
        stats_b["Loss/srbt_calibration"])


def test_identity_infonce_prefers_the_correct_candidate():
    template = torch.tensor([[1.0, 0.0]])
    correct = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    wrong = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], requires_grad=True)
    targets = {
        "identity_positive_index": torch.tensor([0]),
        "identity_mask": torch.tensor([True]),
    }

    _, correct_stats = PetTrackLoss().srbt_loss({
        "template_identity": template,
        "identity_embeddings": correct,
    }, targets)
    _, wrong_stats = PetTrackLoss().srbt_loss({
        "template_identity": template,
        "identity_embeddings": wrong,
    }, targets)

    assert correct_stats["Loss/srbt_identity"] < wrong_stats["Loss/srbt_identity"]


def test_distillation_is_stopped_for_five_percent_then_warms_to_half():
    teacher = {
        "existence": torch.tensor([0.8], requires_grad=True),
        "hazard": torch.softmax(
            torch.randn(1, 129), dim=-1).requires_grad_(),
        "field": torch.full((1, 1, 2, 2), 0.25, requires_grad=True),
        "candidate_map": torch.full(
            (1, 1, 2, 2), 0.25, requires_grad=True),
    }
    predictions = {
        "existence_logits": torch.zeros(1, requires_grad=True),
        "hazard_logits": torch.zeros(1, 129, requires_grad=True),
        "field_logits": torch.zeros(1, 1, 2, 2, requires_grad=True),
        "candidate_logits": torch.zeros(1, 1, 2, 2, requires_grad=True),
    }
    loss_fn = PetTrackLoss(teacher_weight=0.0)

    cold, cold_stats = loss_fn.srbt_loss(
        predictions, {}, teacher=teacher, progress=0.05)
    warm, warm_stats = loss_fn.srbt_loss(
        predictions, {}, teacher=teacher, progress=1.0)
    warm.backward()

    assert cold_stats["Weight/srbt_distill"] == 0.0
    assert warm_stats["Weight/srbt_distill"] == 0.5
    assert warm > cold
    assert all(value.grad is None for value in teacher.values())
    assert all(value.grad is not None for value in predictions.values())


def test_joint_loss_backward_reports_every_unweighted_component():
    batch, candidates, grid, dim, bins = 2, 3, 4, 4, 129
    predictions = {
        "existence_logits": torch.randn(batch, requires_grad=True),
        "hazard_logits": torch.randn(batch, bins, requires_grad=True),
        "field_logits": torch.randn(batch, 1, grid, grid, requires_grad=True),
        "candidate_logits": torch.randn(
            batch, 1, grid, grid, requires_grad=True),
        "hypothesis_boxes": torch.rand(
            batch, candidates, 4, requires_grad=True),
        "hypothesis_scores": torch.randn(
            batch, candidates, requires_grad=True),
        "identity_embeddings": torch.randn(
            batch, candidates, dim, requires_grad=True),
        "template_identity": torch.randn(batch, dim, requires_grad=True),
    }
    targets = {
        "presence": torch.tensor([1.0, 0.0]),
        "hazard_target": torch.tensor([2, 3]),
        "hazard_mask": torch.tensor([True, True]),
        "censor_mask": torch.tensor([False, True]),
        "field_target": _one_hot_field(batch, grid),
        "field_mask": torch.tensor([True, False]),
        "target_box": torch.rand(batch, 4),
        "hypothesis_mask": torch.tensor([True, True]),
        "identity_positive_index": torch.tensor([0, 1]),
        "identity_mask": torch.tensor([True, True]),
    }
    teacher = {
        "existence": torch.tensor([0.8, 0.1], requires_grad=True),
        "hazard": torch.softmax(torch.randn(batch, bins), dim=-1).requires_grad_(),
        "field": torch.softmax(
            torch.randn(batch, 1, grid, grid).flatten(2), dim=-1
        ).view(batch, 1, grid, grid).requires_grad_(),
        "candidate_map": torch.softmax(
            torch.randn(batch, 1, grid, grid).flatten(2), dim=-1
        ).view(batch, 1, grid, grid).requires_grad_(),
    }

    loss, stats = PetTrackLoss().srbt_loss(
        predictions, targets, teacher=teacher, progress=1.0)
    loss.backward()

    expected = {
        "existence", "survival", "field", "hypothesis", "identity",
        "calibration", "distill", "teacher", "total",
    }
    assert expected <= {
        key.removeprefix("Loss/srbt_")
        for key in stats if key.startswith("Loss/srbt_")
    }
    assert all(value.grad is not None for value in predictions.values())
    assert all(torch.isfinite(value.grad).all() for value in predictions.values())
    gradient_norms = {
        name: value.grad.float().norm().item()
        for name, value in predictions.items()
    }
    assert all(norm > 0.0 for norm in gradient_norms.values()), gradient_norms


def test_actor_uses_one_srbt_objective_stack_when_srbt_is_enabled():
    local_cfg = copy.deepcopy(cfg)
    local_cfg.MODEL.SRBT.ENABLE = True
    settings = SimpleNamespace(batchsize=2)
    actor = PETTrackActor(
        nn.Identity(), objective={},
        loss_weight={"giou": 2.0, "l1": 5.0, "focal": 1.0},
        settings=settings, cfg=local_cfg,
    )

    assert actor.active_losses == {"base", "srbt"}
    assert not ({"route", "absence", "freeze", "redetect_gate", "redetect"}
                & actor.active_losses)


def test_actor_adds_base_and_srbt_once_and_passes_teacher_and_progress():
    local_cfg = copy.deepcopy(cfg)
    local_cfg.MODEL.SRBT.ENABLE = True
    actor = PETTrackActor(
        nn.Identity(), objective={},
        loss_weight={"giou": 2.0, "l1": 5.0, "focal": 1.0},
        settings=SimpleNamespace(batchsize=1), cfg=local_cfg,
    )
    predictions = {
        "existence_logits": torch.zeros(1),
        "hazard_logits": torch.zeros(1, 129),
        "field_logits": torch.zeros(1, 1, 2, 2),
        "candidate_logits": torch.zeros(1, 1, 2, 2),
        "hypothesis_boxes": torch.zeros(1, 2, 4),
        "hypothesis_scores": torch.zeros(1, 2),
        "identity_embeddings": torch.zeros(1, 2, 4),
        "template_identity": torch.zeros(1, 4),
    }
    teacher = {"existence": torch.ones(1)}
    targets = {"presence": torch.ones(1)}
    pred_dict = {
        "pred_boxes": torch.zeros(1, 1, 4),
        "srbt_predictions": predictions,
        "srbt_teacher": teacher,
    }
    gt_dict = {"training_progress": torch.tensor(0.4)}

    with patch.object(
            PETTrackBaseActor, "compute_losses",
            return_value=(torch.tensor(2.0), {"Loss/total": 2.0})), patch.object(
                actor, "_build_srbt_targets", return_value=targets), patch.object(
                    actor.pet_loss, "srbt_loss",
                    return_value=(torch.tensor(3.0), {"Loss/srbt_total": 3.0})
                ) as srbt_loss:
        loss, status = actor.compute_losses(pred_dict, gt_dict)

    assert loss.item() == 5.0
    assert status["Loss/base"] == 2.0
    assert status["Loss/SRBT"] == 3.0
    srbt_loss.assert_called_once()
    call_args, call_kwargs = srbt_loss.call_args
    assert call_args == (predictions, targets)
    assert call_kwargs["teacher"] is teacher
    assert call_kwargs["progress"] == pytest.approx(0.4)


def test_actor_rejects_an_incomplete_srbt_prediction_contract():
    local_cfg = copy.deepcopy(cfg)
    local_cfg.MODEL.SRBT.ENABLE = True
    actor = PETTrackActor(
        nn.Identity(), objective={},
        loss_weight={"giou": 2.0, "l1": 5.0, "focal": 1.0},
        settings=SimpleNamespace(batchsize=1), cfg=local_cfg,
    )
    pred_dict = {
        "pred_boxes": torch.zeros(1, 1, 4),
        "srbt_predictions": {"existence_logits": torch.zeros(1)},
    }

    with patch.object(
            PETTrackBaseActor, "compute_losses",
            return_value=(torch.tensor(2.0), {"Loss/total": 2.0})), \
            pytest.raises(RuntimeError, match="missing required outputs"):
        actor.compute_losses(pred_dict, {})


def test_default_and_canonical_loss_configuration_has_no_legacy_stack():
    expected = {
        "EXISTENCE_WEIGHT": 1.0,
        "SURVIVAL_WEIGHT": 1.0,
        "FIELD_WEIGHT": 1.0,
        "HYPOTHESIS_WEIGHT": 0.5,
        "IDENTITY_WEIGHT": 0.2,
        "CALIBRATION_WEIGHT": 0.05,
        "TEACHER_WEIGHT": 1.0,
        "DISTILL_MAX_WEIGHT": 0.5,
        "DISTILL_WARMUP": 0.05,
        "IDENTITY_TEMPERATURE": 0.1,
        "DIVERSITY_MARGIN": 0.25,
        "DIVERSITY_WEIGHT": 0.1,
    }
    assert dict(cfg.TRAIN.SRBT_LOSS) == expected
    assert "PET_LOSS" not in cfg.TRAIN

    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments" / "pet_track" / "felt_pet_track.yaml"
    )
    configured = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert configured["TRAIN"]["SRBT_LOSS"] == expected
    assert "PET_LOSS" not in configured["TRAIN"]


def test_legacy_generated_stage_configuration_fails_fast():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments" / "pet_track" / "generated"
        / "felt_pet_track_v8_stage2_router.yaml"
    )
    with pytest.raises(ValueError, match="legacy TRAIN.PET_LOSS"):
        update_config_from_file(config_path, base_cfg=copy.deepcopy(cfg))
