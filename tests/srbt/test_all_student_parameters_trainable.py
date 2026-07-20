from types import SimpleNamespace

import torch
import pytest

from lib.models.layers.expert_fusion import ProposalBoxAdapter
from lib.models.layers.small_target_expert import SmallTargetExpert
from lib.models.layers.search_window_controller import SearchWindowController
from lib.models.pet_track.pet_track import _load_filtered_baseline_checkpoint
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
        self.small_target_expert = torch.nn.Linear(2, 2)
        self.default_expert = "generalist"
        self.expert_fusion = torch.nn.Module()
        self.expert_fusion.experts = torch.nn.ModuleDict({
            "generalist": torch.nn.Linear(2, 2),
            "motion": torch.nn.Linear(2, 2),
            "small": torch.nn.Linear(2, 2),
            "visibility": torch.nn.Linear(2, 2),
            "discrimination": torch.nn.Linear(2, 2),
        })
        self.expert_fusion.residual_scale_logits = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.zeros(1))
            for name in self.expert_fusion.experts
        })
        self.expert_heads = torch.nn.ModuleDict({
            "motion": torch.nn.Linear(2, 2),
            "small": torch.nn.Linear(2, 2),
            "visibility": torch.nn.Linear(2, 2),
            "discrimination": torch.nn.Linear(2, 2),
        })
        self.proposal_adapters = torch.nn.ModuleDict({
            "motion": ProposalBoxAdapter(),
            "precision_refiner": ProposalBoxAdapter(),
            "discrimination": ProposalBoxAdapter(),
        })
        self.search_window_controller = SearchWindowController(
            expert_count=5, hidden_dim=16)

    def forward_owner(self, value, owner):
        names = tuple(self.expert_fusion.experts)
        name = names[owner]
        shared = self.backbone.norm(self.memory(value))
        expert = self.expert_fusion.experts[name](shared)
        scale = 2.0 * self.expert_fusion.residual_scale_logits[name].sigmoid()
        fused = shared + scale * (expert - shared)
        head = self.box_head if name == self.default_expert else self.expert_heads[name]
        return head(fused)


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
            model.proposal_adapters,
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
        not parameter.requires_grad
        for parameter in specialize_model.expert_fusion.experts["generalist"].parameters())
    assert not specialize_model.expert_fusion.residual_scale_logits[
        "generalist"].requires_grad
    assert all(
        parameter.requires_grad
        for name, expert in specialize_model.expert_fusion.experts.items()
        if name != "generalist"
        for parameter in expert.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in specialize_model.expert_fusion.residual_scale_logits.items()
        if name != "generalist")
    assert all(
        parameter.requires_grad
        for parameter in specialize_model.expert_heads.parameters())
    assert all(
        parameter.requires_grad
        for parameter in specialize_model.proposal_adapters.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in specialize_model.backbone.parameters())
    assert all(
        not parameter.requires_grad
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


def test_precision_expert_specialization_trains_only_independent_branch():
    model = TinyStudent()
    cfg = _cfg()
    cfg.TRAIN.EXPERT_PHASE = "specialize"
    cfg.TRAIN.SPECIALIST_EXPERT_IDS = [2]

    groups = _optimizer_groups(model, cfg)
    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable_names
    assert all(name.startswith((
        "small_target_expert.",
        "proposal_adapters.precision_refiner.",
    )) for name in trainable_names)
    assert [group["name"] for group in groups] == ["precision_refiner"]
    assert {id(parameter) for parameter in groups[0]["params"]} == {
        id(parameter) for module in (
            model.small_target_expert,
            model.proposal_adapters["precision_refiner"],
        ) for parameter in module.parameters()
    }


def test_precision_expert_adapter_training_updates_only_box_refiner():
    model = TinyStudent()
    model.small_target_expert = SmallTargetExpert(search_size=16)
    cfg = _cfg()
    cfg.TRAIN.EXPERT_PHASE = "specialize"
    cfg.TRAIN.SPECIALIST_EXPERT_IDS = [2]
    cfg.TRAIN.SMALL_TARGET_ADAPTER_LR = 2.5e-4

    groups = _optimizer_groups(model, cfg)
    trainable = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable
    assert all(name.startswith((
        "small_target_expert.box_refiner.",
        "proposal_adapters.precision_refiner.",
    )) for name in trainable)
    assert sum(parameter.numel() for parameter in trainable.values()) == 3_352
    assert [group["name"] for group in groups] == [
        "precision_refiner_box_refiner"]
    assert groups[0]["lr"] == pytest.approx(2.5e-4)
    assert {id(parameter) for parameter in groups[0]["params"]} == {
        id(parameter) for parameter in trainable.values()
    }


def test_separate_expert_training_requires_fusions_and_heads():
    model = TinyStudent()
    model.expert_heads = None
    cfg = _cfg()
    cfg.TRAIN.EXPERT_PHASE = "specialize"

    with pytest.raises(ValueError, match="MODEL.EXPERT.ENABLE"):
        _optimizer_groups(model, cfg)


def test_pursuit_phase_trains_only_search_window_controller():
    model = TinyStudent()
    cfg = _cfg()
    cfg.TRAIN.EXPERT_PHASE = "pursuit"
    cfg.TRAIN.PURSUIT_LR = 3e-4

    groups = _optimizer_groups(model, cfg)
    trainable = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable
    assert all(
        name.startswith("search_window_controller.") for name in trainable)
    assert [group["name"] for group in groups] == [
        "search_window_controller"]
    assert groups[0]["lr"] == pytest.approx(3e-4)
    assert {id(parameter) for parameter in groups[0]["params"]} == {
        id(parameter) for parameter in trainable.values()
    }


def test_baseline_path_is_bitwise_immutable_after_precision_expert_step(tmp_path):
    torch.manual_seed(29)
    model = TinyStudent()
    inherited = {
        key: torch.full_like(value, 0.25)
        for key, value in model.state_dict().items()
        if key.startswith(("backbone.", "memory.", "box_head."))
    }
    checkpoint = tmp_path / "baseline.pth.tar"
    torch.save({"net": inherited}, checkpoint)
    _load_filtered_baseline_checkpoint(model, checkpoint)

    cfg = _cfg()
    cfg.TRAIN.EXPERT_PHASE = "specialize"
    cfg.TRAIN.SPECIALIST_EXPERT_IDS = [2]
    groups = _optimizer_groups(model, cfg)
    optimizer = torch.optim.SGD(groups, lr=cfg.TRAIN.LR)
    generalist_names = (
        "backbone.",
        "memory.",
        "box_head.",
        "expert_fusion.experts.generalist.",
        "expert_fusion.residual_scale_logits.generalist",
    )
    generalist_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(generalist_names)
    }
    small_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("small_target_expert.")
    }
    value = torch.randn(4, 2)
    with torch.no_grad():
        generalist_output_before = model.forward_owner(value, owner=0).clone()

    optimizer.zero_grad()
    model.small_target_expert(value).square().mean().backward()
    optimizer.step()

    current = dict(model.named_parameters())
    assert all(
        torch.equal(current[name], parameter)
        for name, parameter in generalist_before.items()
    )
    with torch.no_grad():
        assert torch.equal(
            model.forward_owner(value, owner=0), generalist_output_before)
    assert any(
        not torch.equal(current[name], parameter)
        for name, parameter in small_before.items()
    )
