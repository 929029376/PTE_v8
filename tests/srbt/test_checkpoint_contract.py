from pathlib import Path
import warnings

import pytest
import torch
from torch import nn

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\..* is deprecated.*",
        category=FutureWarning,
    )
    from lib.models.pet_track.pet_track import (
        _load_filtered_baseline_checkpoint,
        _load_legacy_expert_checkpoint,
        _load_retained_model_checkpoint,
    )


class _TinyTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 2)
        self.memory = nn.Linear(2, 2)
        self.box_head = nn.Linear(2, 1)
        self.srbt = nn.Linear(2, 2)
        self.expert_fusion = nn.Linear(2, 2)


class _LegacyMetadata:
    pass


def _inherited_state(model, *, module_prefix=False):
    state = {}
    for key, value in model.state_dict().items():
        if key.startswith(("backbone.", "memory.", "box_head.")):
            source_key = f"module.{key}" if module_prefix else key
            state[source_key] = torch.full_like(value, 7)
    return state


def test_loads_complete_inherited_state_and_reports_extensions(tmp_path):
    model = _TinyTracker()
    extension_before = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("srbt.")
    }
    checkpoint_path = tmp_path / "baseline.pth.tar"
    torch.save({"net": _inherited_state(model, module_prefix=True)}, checkpoint_path)

    report = _load_filtered_baseline_checkpoint(model, checkpoint_path)

    expected_loaded = sorted(
        key
        for key in model.state_dict()
        if key.startswith(("backbone.", "memory.", "box_head."))
    )
    expected_extensions = sorted(
        key for key in model.state_dict() if key not in expected_loaded)
    assert report == {
        "path": str(checkpoint_path),
        "loaded_count": len(expected_loaded),
        "loaded_keys": expected_loaded,
        "missing_extension_keys": expected_extensions,
    }
    for key in expected_loaded:
        assert torch.equal(model.state_dict()[key], torch.full_like(model.state_dict()[key], 7))
    for key, value in extension_before.items():
        assert torch.equal(model.state_dict()[key], value)


def test_rejects_all_checkpoint_contract_violations_before_loading(tmp_path):
    model = _TinyTracker()
    source = _inherited_state(model)
    source.pop("backbone.bias")
    source["memory.weight"] = torch.zeros(3, 3)
    source["backbone.extra"] = torch.zeros(1)
    source["unknown_branch.weight"] = torch.zeros(1)
    source["box_head.weight"] = torch.full_like(source["box_head.weight"], 11)
    checkpoint_path = tmp_path / "invalid.pth.tar"
    torch.save({"net": source}, checkpoint_path)
    box_head_before = model.box_head.weight.detach().clone()

    with pytest.raises(RuntimeError) as exc_info:
        _load_filtered_baseline_checkpoint(model, checkpoint_path)

    message = str(exc_info.value)
    for bad_key in (
        "backbone.bias",
        "memory.weight",
        "backbone.extra",
        "unknown_branch.weight",
    ):
        assert bad_key in message
    assert torch.equal(model.box_head.weight, box_head_before)


def test_rejects_checkpoint_without_a_state_mapping(tmp_path):
    checkpoint_path = Path(tmp_path) / "invalid-root.pth.tar"
    torch.save({"epoch": 98}, checkpoint_path)

    with pytest.raises(RuntimeError, match="state mapping"):
        _load_filtered_baseline_checkpoint(_TinyTracker(), checkpoint_path)


def test_legacy_pickle_requires_an_explicit_trust_boundary(tmp_path):
    model = _TinyTracker()
    checkpoint_path = Path(tmp_path) / "legacy.pth.tar"
    torch.save(
        {"net": _inherited_state(model), "legacy": _LegacyMetadata()},
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="trusted_legacy_pickle=True"):
        _load_filtered_baseline_checkpoint(model, checkpoint_path)

    report = _load_filtered_baseline_checkpoint(
        model,
        checkpoint_path,
        trusted_legacy_pickle=True,
    )
    assert report["loaded_count"] == 6


def test_legacy_expert_loader_reuses_fusions_and_clones_missing_heads(tmp_path):
    class LegacyTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.default_expert = "generalist"
            self.box_head = nn.Linear(2, 1)
            self.expert_heads = nn.ModuleDict({
                "motion_fm": nn.Linear(2, 1),
            })
            self.expert_fusion = nn.Module()
            self.expert_fusion.experts = nn.ModuleDict({
                "generalist": nn.Linear(2, 2),
                "motion_fm": nn.Linear(2, 2),
            })
            self.expert_fusion.residual_scale_logits = nn.ParameterDict({
                "generalist": nn.Parameter(torch.zeros(1)),
                "motion_fm": nn.Parameter(torch.zeros(1)),
            })

    model = LegacyTarget()
    with torch.no_grad():
        model.box_head.weight.fill_(9)
        model.box_head.bias.fill_(9)
    checkpoint_path = Path(tmp_path) / "legacy-experts.pth.tar"
    torch.save({"net": {
        "expert_router.head.2.weight": torch.ones(1),
        "expert_fusion.experts.motion_fm.weight": torch.full_like(
            model.expert_fusion.experts["motion_fm"].weight, 5),
        "expert_fusion.experts.motion_fm.bias": torch.full_like(
            model.expert_fusion.experts["motion_fm"].bias, 5),
    }}, checkpoint_path)

    report = _load_legacy_expert_checkpoint(model, checkpoint_path)

    assert report["loaded_count"] == 4
    assert torch.equal(model.expert_fusion.experts["motion_fm"].weight,
                       torch.full_like(model.expert_fusion.experts["motion_fm"].weight, 5))
    assert torch.equal(model.expert_heads["motion_fm"].weight,
                       model.box_head.weight)
    assert torch.equal(model.expert_heads["motion_fm"].bias,
                       model.box_head.bias)
    assert "expert_router.head.2.weight" in report["ignored_legacy_keys"]
    assert any("generalist" in key for key in report["initialized_extension_keys"])


def test_retained_checkpoint_migrates_missing_heads_and_discards_router(tmp_path):
    class RetainedTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.default_expert = "generalist"
            self.register_buffer("_pet_architecture_version", torch.tensor(11))
            self.backbone = nn.Linear(2, 2)
            self.box_head = nn.Linear(2, 1)
            self.expert_heads = nn.ModuleDict({
                "motion_fm": nn.Linear(2, 1),
            })
            self.expert_fusion = nn.Linear(2, 2)

    model = RetainedTarget()
    source = {
        key: torch.full_like(value, 7)
        for key, value in model.state_dict().items()
        if not key.startswith(("expert_heads.", "_pet_architecture_version"))
    }
    source["expert_router.classifier.weight"] = torch.ones(3, 2)
    checkpoint_path = Path(tmp_path) / "retained-route.pth.tar"
    torch.save({"net": source}, checkpoint_path)

    report = _load_retained_model_checkpoint(
        model, checkpoint_path, label="retained route-free migration")

    assert torch.equal(model.expert_heads["motion_fm"].weight,
                       model.box_head.weight)
    assert torch.equal(model.expert_heads["motion_fm"].bias,
                       model.box_head.bias)
    assert report["migrated_head_keys"] == [
        "expert_heads.motion_fm.bias", "expert_heads.motion_fm.weight"]
    assert report["ignored_obsolete_keys"] == [
        "expert_router.classifier.weight"]
