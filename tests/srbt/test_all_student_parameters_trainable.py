from types import SimpleNamespace

import torch
import pytest

from lib.train.base_functions import _optimizer_groups


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.patch_embed = torch.nn.Linear(2, 2)
        self.backbone.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(12)])
        self.backbone.norm = torch.nn.LayerNorm(2)
        self.backbone.amah_tail = torch.nn.Linear(2, 2)
        self.memory = torch.nn.Linear(2, 2)
        self.box_head = torch.nn.Linear(2, 2)
        self.redetect_expert = torch.nn.Linear(2, 2)
        self.rgb_identity_verifier = torch.nn.Linear(2, 2)
        self.visibility_gate = torch.nn.Linear(2, 2)
        self.expert_fusion = torch.nn.Linear(2, 2)
        self.expert_heads = torch.nn.ModuleDict({
            "motion": torch.nn.Linear(2, 2),
            "small": torch.nn.Linear(2, 2),
            "visibility": torch.nn.Linear(2, 2),
            "discrimination": torch.nn.Linear(2, 2),
        })


def _cfg():
    return SimpleNamespace(TRAIN=SimpleNamespace(
        LR=1e-4,
        WEIGHT_DECAY=1e-4,
        BACKBONE_MULTIPLIER=0.1,
        EXPERT_LR_MULTIPLIER=5.0,
        REFINE_TAIL_LR=1e-6,
        REFINE_MEMORY_LR=5e-7,
        EXPERT_PHASE="refine",
        OPTIMIZER="ADAMW",
        SCHEDULER=SimpleNamespace(TYPE="step"),
    ))


def test_refine_trains_only_vit_tail_and_hopfield_with_declared_lrs():
    model = TinyStudent()
    groups = _optimizer_groups(model, _cfg())

    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_names
    assert all(name.startswith((
        "backbone.blocks.8.",
        "backbone.blocks.9.",
        "backbone.blocks.10.",
        "backbone.blocks.11.",
        "backbone.norm.",
        "backbone.amah_",
        "memory.",
    )) for name in trainable_names)
    assert all(
        not parameter.requires_grad
        for module in (
            model.expert_fusion,
            model.expert_heads,
            model.box_head,
            model.visibility_gate,
            model.rgb_identity_verifier,
            model.redetect_expert,
        )
        for parameter in module.parameters()
    )
    seen = {}
    for group in groups:
        for parameter in group["params"]:
            seen[id(parameter)] = seen.get(id(parameter), 0) + 1
    assert seen == {
        id(parameter): 1 for parameter in model.parameters()
        if parameter.requires_grad
    }

    by_name = {group["name"]: group for group in groups}
    assert set(by_name) == {"vit_tail_refine", "hopfield_refine"}
    assert by_name["vit_tail_refine"]["lr"] == 1e-6
    assert by_name["hopfield_refine"]["lr"] == 5e-7
    assert {id(parameter) for parameter in model.backbone.norm.parameters()} <= {
        id(parameter) for parameter in by_name["vit_tail_refine"]["params"]
    }
    assert {id(parameter) for parameter in model.memory.parameters()} <= {
        id(parameter) for parameter in by_name["hopfield_refine"]["params"]
    }


def test_refine_freezes_parameters_outside_the_declared_tail():
    model = TinyStudent()
    model.unclassified = torch.nn.Linear(2, 2)

    _optimizer_groups(model, _cfg())

    assert all(
        not parameter.requires_grad
        for parameter in model.unclassified.parameters())


def test_optimizer_groups_enforce_separate_expert_training_phases():
    specialize_model = TinyStudent()
    specialize_cfg = _cfg()
    specialize_cfg.TRAIN.EXPERT_PHASE = "specialize"
    _optimizer_groups(specialize_model, specialize_cfg)

    assert all(
        parameter.requires_grad
        for parameter in specialize_model.expert_fusion.parameters())
    assert all(
        parameter.requires_grad
        for parameter in specialize_model.expert_heads.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in specialize_model.backbone.parameters())
    assert all(
        parameter.requires_grad
        for parameter in specialize_model.box_head.parameters())

    recovery_model = TinyStudent()
    recovery_cfg = _cfg()
    recovery_cfg.TRAIN.EXPERT_PHASE = "recovery"
    groups = _optimizer_groups(recovery_model, recovery_cfg)

    trainable = {
        name for name, parameter in recovery_model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith((
        "visibility_gate.",
        "rgb_identity_verifier.",
        "redetect_expert.",
    )) for name in trainable)
    assert [group["name"] for group in groups] == ["recovery"]


def test_separate_expert_training_requires_fusions_and_heads():
    model = TinyStudent()
    model.expert_heads = None
    cfg = _cfg()
    cfg.TRAIN.EXPERT_PHASE = "specialize"

    with pytest.raises(ValueError, match="MODEL.EXPERT.ENABLE"):
        _optimizer_groups(model, cfg)
