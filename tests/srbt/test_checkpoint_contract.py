from pathlib import Path
import warnings

import pytest
import torch
from torch import nn

from lib.models.layers.small_target_expert import SmallTargetExpert
from lib.models.layers.search_window_controller import SearchWindowController

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
        "small_target_initialized_keys": [],
    }
    for key in expected_loaded:
        assert torch.equal(model.state_dict()[key], torch.full_like(model.state_dict()[key], 7))
    for key, value in extension_before.items():
        assert torch.equal(model.state_dict()[key], value)


def test_v25_checkpoint_retains_every_old_tensor_and_initializes_only_controller(tmp_path):
    class VersionedTracker(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "_pet_architecture_version", torch.tensor(26), persistent=True)
            self.backbone = nn.Linear(3, 2)
            self.memory = nn.Linear(2, 2)
            self.box_head = nn.Linear(2, 1)
            self.search_window_controller = SearchWindowController(
                expert_count=5, hidden_dim=8)

    model = VersionedTracker()
    target = model.state_dict()
    old_state = {
        key: torch.full_like(value, 0.25)
        for key, value in target.items()
        if not key.startswith("search_window_controller.")
    }
    old_state["_pet_architecture_version"] = torch.tensor(25)
    checkpoint = tmp_path / "v25.pth.tar"
    torch.save({"net": old_state}, checkpoint)
    controller_before = {
        key: value.clone() for key, value in target.items()
        if key.startswith("search_window_controller.")
    }

    report = _load_retained_model_checkpoint(
        model, checkpoint, label="v25 pursuit migration")

    loaded = model.state_dict()
    for key, value in old_state.items():
        if key != "_pet_architecture_version":
            assert torch.equal(loaded[key], value)
    for key, value in controller_before.items():
        assert torch.equal(loaded[key], value)
    assert set(report["initialized_extension_keys"]) == set(controller_before)


def test_v26_checkpoint_retains_all_weights_and_initializes_only_activator(tmp_path):
    from tests.srbt.test_srbt_model_integration import _model

    model = _model(expert_enabled=True)
    source = {
        key: torch.full_like(value, 0.375)
        for key, value in model.state_dict().items()
        if not key.startswith("expert_activator.")
    }
    source["_pet_architecture_version"] = torch.tensor(26)
    activator_before = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("expert_activator.")
    }
    checkpoint = tmp_path / "v26.pth.tar"
    torch.save({"net": source}, checkpoint)

    report = _load_retained_model_checkpoint(
        model, checkpoint, label="v26 activator migration")

    loaded = model.state_dict()
    assert all(
        torch.equal(loaded[key], value)
        for key, value in source.items()
        if key != "_pet_architecture_version"
    )
    assert all(
        torch.equal(loaded[key], value)
        for key, value in activator_before.items()
    )
    assert report["initialized_extension_keys"] == sorted(activator_before)


def test_baseline_loader_clones_loaded_box_head_into_every_specialist(tmp_path):
    class ExpertTarget(_TinyTracker):
        def __init__(self):
            super().__init__()
            self.expert_heads = nn.ModuleDict({
                "motion_fm": nn.Linear(2, 1),
            })
            self.small_target_expert = nn.Linear(3, 3)

    model = ExpertTarget()
    source = _inherited_state(model)
    source["box_head.weight"] = torch.full_like(source["box_head.weight"], 9)
    source["box_head.bias"] = torch.full_like(source["box_head.bias"], 4)
    checkpoint_path = tmp_path / "baseline-with-experts.pth.tar"
    torch.save({"net": source}, checkpoint_path)
    small_before = {
        key: value.clone()
        for key, value in model.small_target_expert.state_dict().items()
    }

    report = _load_filtered_baseline_checkpoint(model, checkpoint_path)

    for head in model.expert_heads.values():
        assert torch.equal(head.weight, model.box_head.weight)
        assert torch.equal(head.bias, model.box_head.bias)
    assert all(torch.equal(model.small_target_expert.state_dict()[key], value)
               for key, value in small_before.items())
    assert report["small_target_initialized_keys"] == []


def test_baseline_loader_initializes_compatible_independent_small_layers(tmp_path):
    class CompatibleTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.patch_embed = nn.Module()
            self.backbone.patch_embed.proj = nn.Conv2d(3, 96, 5)
            self.memory = nn.Linear(2, 2)
            self.box_head = nn.Module()
            self.box_head.conv5_ctr = nn.Conv2d(8, 1, 1)
            self.box_head.conv5_size = nn.Conv2d(8, 2, 1)
            self.box_head.conv5_offset = nn.Conv2d(8, 2, 1)
            self.expert_heads = nn.ModuleDict()
            self.small_target_expert = SmallTargetExpert(search_size=16)

    model = CompatibleTarget()
    source = _inherited_state(model)
    for key in source:
        source[key] = torch.full_like(source[key], 0.25)
    checkpoint_path = tmp_path / "baseline-compatible-small.pth.tar"
    torch.save({"net": source}, checkpoint_path)
    rgb_before = model.small_target_expert.encoder.rgb_stem[0].weight.clone()

    report = _load_filtered_baseline_checkpoint(model, checkpoint_path)

    small = model.small_target_expert
    assert not torch.equal(small.encoder.rgb_stem[0].weight, rgb_before)
    assert torch.equal(
        small.encoder.rgb_stem[0].weight,
        small.encoder.event_stem[0].weight)
    assert torch.equal(
        small.head.center[-1].bias,
        model.box_head.conv5_ctr.bias)
    assert set(report["small_target_initialized_keys"]) == {
        "small_target_expert.encoder.rgb_stem.0",
        "small_target_expert.encoder.event_stem.0",
        "small_target_expert.head.center.3",
        "small_target_expert.head.size.3",
        "small_target_expert.head.offset.3",
    }


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


def test_retained_v13_checkpoint_initializes_v23_small_extensions(tmp_path):
    from tests.srbt.test_srbt_model_integration import _model

    model = _model(expert_enabled=True)
    assert model.ARCHITECTURE_VERSION == 27
    source = {
        key: value.clone() for key, value in model.state_dict().items()
    }
    source["_pet_architecture_version"] = torch.tensor(13)
    refine_keys = sorted(
        key for key in source
        if key.startswith("small_target_expert.head.refine."))
    residual_keys = sorted(
        key for key in source
        if key.startswith((
            "small_target_expert.encoder.rgb_s8_residual.",
            "small_target_expert.encoder.event_s8_residual.",
        )))
    box_refiner_keys = sorted(
        key for key in source
        if key.startswith("small_target_expert.box_refiner."))
    proposal_keys = sorted(
        key for key in source if key.startswith("proposal_adapters."))
    assert refine_keys
    assert residual_keys
    assert box_refiner_keys
    for key in refine_keys + residual_keys + box_refiner_keys:
        source.pop(key)
    checkpoint_path = Path(tmp_path) / "small-v13.pth.tar"
    torch.save({"net": source}, checkpoint_path)

    report = _load_retained_model_checkpoint(
        model, checkpoint_path, label="v13 small-target migration")

    assert set(refine_keys + residual_keys + box_refiner_keys).issubset(
        report["initialized_extension_keys"])
    feature = torch.randn(2, 64, 4, 4)
    assert torch.count_nonzero(
        model.small_target_expert.head.refine(feature)).item() == 0


def test_retained_v14_checkpoint_initializes_v23_extensions(tmp_path):
    from tests.srbt.test_srbt_model_integration import _model

    model = _model(expert_enabled=True)
    source = {
        key: value.clone() for key, value in model.state_dict().items()
    }
    source["_pet_architecture_version"] = torch.tensor(14)
    inherited_key = "small_target_expert.encoder.rgb_stem.0.weight"
    source[inherited_key] = torch.full_like(source[inherited_key], 0.375)
    residual_keys = sorted(
        key for key in source
        if key.startswith((
            "small_target_expert.encoder.rgb_s8_residual.",
            "small_target_expert.encoder.event_s8_residual.",
        )))
    box_refiner_keys = sorted(
        key for key in source
        if key.startswith("small_target_expert.box_refiner."))
    proposal_keys = sorted(
        key for key in source if key.startswith("proposal_adapters."))
    assert residual_keys
    assert box_refiner_keys
    for key in residual_keys + box_refiner_keys:
        source.pop(key)
    checkpoint_path = Path(tmp_path) / "small-v14.pth.tar"
    torch.save({"net": source}, checkpoint_path)

    report = _load_retained_model_checkpoint(
        model, checkpoint_path, label="v14 small-target migration")

    assert report["initialized_extension_keys"] == sorted(
        residual_keys + box_refiner_keys + proposal_keys)
    assert torch.equal(model.state_dict()[inherited_key], source[inherited_key])
    for key in residual_keys:
        if key.endswith("body.3.weight"):
            assert torch.count_nonzero(model.state_dict()[key]).item() == 0


def test_retained_v14_checkpoint_rejects_missing_inherited_extension(tmp_path):
    from tests.srbt.test_srbt_model_integration import _model

    model = _model(expert_enabled=True)
    source = {
        key: value.clone() for key, value in model.state_dict().items()
    }
    source["_pet_architecture_version"] = torch.tensor(14)
    residual_keys = [
        key for key in source
        if key.startswith((
            "small_target_expert.encoder.rgb_s8_residual.",
            "small_target_expert.encoder.event_s8_residual.",
        ))
    ]
    box_refiner_keys = [
        key for key in source
        if key.startswith("small_target_expert.box_refiner.")
    ]
    for key in residual_keys + box_refiner_keys:
        source.pop(key)
    inherited_key = next(
        key for key in source if key.startswith("visibility_gate."))
    source.pop(inherited_key)
    checkpoint_path = Path(tmp_path) / "damaged-small-v14.pth.tar"
    torch.save({"net": source}, checkpoint_path)

    with pytest.raises(RuntimeError, match="missing retained keys"):
        _load_retained_model_checkpoint(
            model, checkpoint_path, label="damaged v14 migration")


def test_retained_v18_checkpoint_initializes_only_v23_box_refiner(tmp_path):
    from tests.srbt.test_srbt_model_integration import _model

    model = _model(expert_enabled=True)
    source = {
        key: value.clone() for key, value in model.state_dict().items()
    }
    source["_pet_architecture_version"] = torch.tensor(18)
    inherited_key = "small_target_expert.match_refinement.block.0.weight"
    source[inherited_key] = torch.full_like(source[inherited_key], 0.625)
    box_refiner_keys = sorted(
        key for key in source
        if key.startswith("small_target_expert.box_refiner."))
    assert box_refiner_keys
    for key in box_refiner_keys:
        source.pop(key)
    checkpoint_path = Path(tmp_path) / "small-v18.pth.tar"
    torch.save({"net": source}, checkpoint_path)

    report = _load_retained_model_checkpoint(
        model, checkpoint_path, label="v18 box-refiner migration")

    proposal_keys = sorted(
        key for key in source if key.startswith("proposal_adapters."))
    assert model.ARCHITECTURE_VERSION == 27
    assert report["initialized_extension_keys"] == sorted(
        box_refiner_keys + proposal_keys)
    assert torch.equal(model.state_dict()[inherited_key], source[inherited_key])
    assert torch.count_nonzero(
        model.small_target_expert.box_refiner.output.weight).item() == 0


def test_retained_v22_checkpoint_reinitializes_v23_box_refiner(tmp_path):
    from tests.srbt.test_srbt_model_integration import _model

    model = _model(expert_enabled=True)
    assert model.ARCHITECTURE_VERSION == 27
    source = {
        key: value.clone() for key, value in model.state_dict().items()
    }
    source["_pet_architecture_version"] = torch.tensor(22)
    box_refiner_keys = sorted(
        key for key in source
        if key.startswith("small_target_expert.box_refiner."))
    proposal_keys = sorted(
        key for key in source if key.startswith("proposal_adapters."))
    for key in box_refiner_keys:
        source[key] = torch.full_like(source[key], 0.5)
    obsolete_key = "small_target_expert.detail_center.output.weight"
    source[obsolete_key] = torch.full((5, 16, 1, 1), 0.75)
    checkpoint_path = Path(tmp_path) / "small-v22.pth.tar"
    torch.save({"net": source}, checkpoint_path)

    report = _load_retained_model_checkpoint(
        model, checkpoint_path, label="v22 box-refiner migration")

    assert report["initialized_extension_keys"] == sorted(
        box_refiner_keys + proposal_keys)
    assert report["ignored_obsolete_keys"] == [obsolete_key]
    assert model.small_target_expert.box_refiner.output.out_channels == 4
    assert torch.count_nonzero(
        model.small_target_expert.box_refiner.output.weight).item() == 0
    loaded = model.state_dict()
    assert all(not torch.equal(loaded[key], source[key])
               for key in box_refiner_keys)


def test_retained_v23_checkpoint_rejects_missing_box_refiner_tensor(tmp_path):
    from tests.srbt.test_srbt_model_integration import _model

    model = _model(expert_enabled=True)
    assert model.ARCHITECTURE_VERSION == 27
    source = {
        key: value.clone() for key, value in model.state_dict().items()
        if not key.startswith("proposal_adapters.")
    }
    source["_pet_architecture_version"] = torch.tensor(23)
    missing_key = next(
        key for key in source
        if key.startswith("small_target_expert.box_refiner."))
    source.pop(missing_key)
    checkpoint_path = Path(tmp_path) / "damaged-small-v23.pth.tar"
    torch.save({"net": source}, checkpoint_path)

    with pytest.raises(RuntimeError, match="missing retained keys"):
        _load_retained_model_checkpoint(
            model, checkpoint_path, label="damaged v23 box refiner")


def test_retained_v23_checkpoint_initializes_v24_proposal_adapters(tmp_path):
    from tests.srbt.test_srbt_model_integration import _model

    model = _model(expert_enabled=True)
    assert model.ARCHITECTURE_VERSION == 27
    source = {
        key: value.clone() for key, value in model.state_dict().items()
        if not key.startswith("proposal_adapters.")
    }
    source["_pet_architecture_version"] = torch.tensor(23)
    checkpoint_path = Path(tmp_path) / "experts-v23.pth.tar"
    torch.save({"net": source}, checkpoint_path)

    report = _load_retained_model_checkpoint(
        model, checkpoint_path, label="v23 proposal migration")

    proposal_keys = sorted(
        key for key in model.state_dict()
        if key.startswith("proposal_adapters."))
    assert report["initialized_extension_keys"] == proposal_keys
    assert all(
        torch.count_nonzero(model.state_dict()[key]).item() == 0
        for key in proposal_keys
        if key.endswith(("output.weight", "output.bias"))
    )


def test_retained_v24_checkpoint_migrates_precision_refiner_adapter(tmp_path):
    from tests.srbt.test_srbt_model_integration import _model

    model = _model(expert_enabled=True)
    assert model.ARCHITECTURE_VERSION == 27
    source = {
        key: value.clone() for key, value in model.state_dict().items()
    }
    source["_pet_architecture_version"] = torch.tensor(24)
    migrated = {}
    for key in list(source):
        prefix = "proposal_adapters.precision_refiner."
        if not key.startswith(prefix):
            continue
        old_key = key.replace(
            prefix, "proposal_adapters.small_target_st.", 1)
        migrated[key] = old_key
        source[old_key] = torch.full_like(source.pop(key), 0.625)
    assert migrated
    checkpoint_path = Path(tmp_path) / "experts-v24.pth.tar"
    torch.save({"net": source}, checkpoint_path)

    report = _load_retained_model_checkpoint(
        model, checkpoint_path, label="v24 precision-refiner rename")

    loaded = model.state_dict()
    assert all(torch.equal(
        loaded[new_key], torch.full_like(loaded[new_key], 0.625))
        for new_key in migrated)
    assert report["migrated_proposal_adapter_keys"] == sorted(migrated)
