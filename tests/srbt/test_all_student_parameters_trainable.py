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
        self.memory = torch.nn.Linear(2, 2)
        self.box_head = torch.nn.Linear(2, 2)
        self.redetect_expert = torch.nn.Linear(2, 2)
        self.srbt_evidence = torch.nn.Linear(2, 2)
        self.srbt_belief = torch.nn.Linear(2, 2)
        self.srbt_teacher = torch.nn.Linear(2, 2)


def _cfg():
    return SimpleNamespace(TRAIN=SimpleNamespace(
        LR=1e-4,
        WEIGHT_DECAY=1e-4,
        BACKBONE_MULTIPLIER=0.1,
        OPTIMIZER="ADAMW",
        SCHEDULER=SimpleNamespace(TYPE="step"),
    ))


def test_all_student_parameters_trainable_and_owned_once_with_exact_srbt_lrs():
    model = TinyStudent()
    groups = _optimizer_groups(model, _cfg())

    assert all(parameter.requires_grad for parameter in model.parameters())
    seen = {}
    for group in groups:
        for parameter in group["params"]:
            seen[id(parameter)] = seen.get(id(parameter), 0) + 1
    assert seen == {id(parameter): 1 for parameter in model.parameters()}

    by_name = {group["name"]: group for group in groups}
    assert by_name["vit_blocks_1_8"]["lr"] == 1e-5
    assert by_name["vit_blocks_9_12_core"]["lr"] == 2.5e-5
    assert by_name["srbt_student"]["lr"] == 1e-4
    assert by_name["srbt_teacher"]["lr"] == 1e-4
    assert set(by_name) == {
        "vit_blocks_1_8",
        "vit_blocks_9_12_core",
        "srbt_student",
        "srbt_teacher",
    }
    assert {id(parameter) for parameter in model.backbone.patch_embed.parameters()} <= {
        id(parameter) for parameter in by_name["vit_blocks_1_8"]["params"]
    }
    assert {id(parameter) for parameter in model.backbone.norm.parameters()} <= {
        id(parameter) for parameter in by_name["vit_blocks_9_12_core"]["params"]
    }
    assert {id(parameter) for parameter in model.redetect_expert.parameters()} <= {
        id(parameter) for parameter in by_name["srbt_student"]["params"]
    }


def test_optimizer_grouping_rejects_unowned_trainable_parameters():
    model = TinyStudent()
    model.unclassified = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="Unowned trainable parameters"):
        _optimizer_groups(model, _cfg())
