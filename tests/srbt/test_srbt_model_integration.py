import inspect
from types import SimpleNamespace

import pytest
import torch
from easydict import EasyDict as edict

from lib.models.pet_track.pet_track import PETTrack
from lib.train.base_functions import _optimizer_groups


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        self.num_heads = 2
        self.probe = torch.nn.Parameter(torch.ones(()))
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8) for _ in range(12)])
        self.norm = torch.nn.LayerNorm(8)

    def _z_feat(self, value):
        return torch.ones(value.shape[0], 4, 8, device=value.device) * self.probe

    def _x_feat(self, value):
        batch = value.shape[0]
        scale = value.mean(
            dim=tuple(range(1, value.ndim))).view(batch, 1, 1)
        return torch.ones(batch, 4, 8, device=value.device) * (
            self.probe + scale)

    def forward(self, static_zi, static_ze, dynamic_zi, dynamic_ze,
                xi, xe, **_):
        self.last_dynamic_zi = dynamic_zi.detach().clone()
        return torch.cat(
            (static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe), dim=1), {}


class TinyMemory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward_dynamic_features(self, zi, ze):
        return zi * self.weight, ze * self.weight


class TinyHead(torch.nn.Module):
    feat_sz = 2

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(8, 2, 1)

    def forward(self, feat, _):
        batch = feat.shape[0]
        score = self.proj(feat).mean(dim=1, keepdim=True)
        bbox = torch.full((batch, 1, 4), 0.5, device=feat.device)
        size = torch.full((batch, 2, 2, 2), 0.25, device=feat.device)
        offset = torch.full((batch, 2, 2, 2), 0.5, device=feat.device)
        return score, bbox, size, offset


class ProbeRedetect(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.last_feat = None

    def forward(self, feat, prior_H=None, gt_score_map=None,
                template_tokens=None):
        self.last_feat = feat
        return {"field": feat[:, :1] * self.scale}


def _cfg(srbt_enabled=True, expert_enabled=False):
    model = {
        "PET": {"ENABLE": False},
        "SRBT": {
            "ENABLE": srbt_enabled,
            "GATE": {"HIDDEN_DIM": 8, "RESPONSE_DIM": 3},
        },
    }
    if expert_enabled:
        model["EXPERT"] = {
            "ENABLE": True,
            "DEFAULT": "generalist",
            "NAMES": [
                "generalist", "motion_fm", "precision_refiner",
                "visibility_foc_ov", "discrimination_bi",
            ],
        }
    return edict({
        "MODEL": model,
        "TRAIN": {},
        "DATA": {"SEARCH": {"SIZE": 16}},
    })


def _model(srbt_enabled=True, expert_enabled=False):
    return PETTrack(
        TinyBackbone(), TinyMemory(), TinyHead(),
        _cfg(srbt_enabled, expert_enabled), head_type="CENTER")


def _images(batch=2):
    shape = (batch, 2, 3, 16, 16)
    search_shape = (batch, 1, 3, 16, 16)
    return (
        torch.randn(shape), torch.randn(shape),
        torch.randn(search_shape), torch.randn(search_shape),
    )


def test_recovery_architecture_has_no_future_or_posterior_entry_points():
    assert not hasattr(PETTrack, "initialize_srbt_posterior")
    for method in (PETTrack.forward, PETTrack.inference):
        names = inspect.signature(method).parameters
        assert not any(
            token in name for name in names
            for token in ("future", "teacher", "posterior"))


def test_normal_forward_uses_only_lightweight_presence_gate():
    model = _model()
    output = model(*_images())

    for name in (
            "srbt_evidence", "srbt_belief", "srbt_teacher",
            "srbt_field_head", "srbt_candidate_head",
            "srbt_identity_head", "srbt_template_identity"):
        assert not hasattr(model, name)
    assert set(output["presence_predictions"]) == {"logits", "score"}
    assert torch.equal(
        output["presence_score"], output["presence_predictions"]["score"])
    assert "srbt_posterior" not in output
    assert "hypotheses" not in output
    assert "srbt_teacher" not in output


def test_presence_gate_trains_without_changing_shared_features():
    model = _model()
    output = model(*_images())
    torch.nn.functional.cross_entropy(
        output["presence_predictions"]["logits"],
        torch.tensor([1, 0]),
    ).backward()

    assert model.backbone.probe.grad is None
    assert all(
        parameter.grad is not None
        for parameter in model.visibility_gate.parameters())


def test_rgb_identity_tokens_use_only_rgb_patch_paths():
    model = _model()
    rgb = torch.randn(2, 3, 16, 16)
    assert model.rgb_identity_tokens(rgb, template=True).shape == (2, 4, 8)
    assert model.rgb_identity_tokens(rgb, template=False).shape == (2, 4, 8)


def test_shared_model_forward_returns_four_shared_expert_outputs():
    model = _model(expert_enabled=True)
    output = model(*_images(batch=1))

    assert tuple(output["expert_outputs"]) == tuple(model.shared_expert_names)
    assert len(output["expert_outputs"]) == 4


def test_inference_presence_gate_uses_visibility_expert_response():
    class CaptureGate(torch.nn.Module):
        def forward(self, pooled_feature, response_stats):
            self.response_stats = response_stats.detach().clone()
            return torch.zeros(
                pooled_feature.shape[0], 2, device=pooled_feature.device)

    model = _model(expert_enabled=True)
    model.visibility_gate = CaptureGate()

    def controlled_forward_head(
            cat_feature, gt_score_map=None, training_expert_ids=None):
        batch = cat_feature.shape[0]
        general = {
            "score_map": torch.zeros(batch, 1, 2, 2),
            "pred_boxes": torch.full((batch, 1, 4), 0.5),
        }
        visibility = {
            "score_map": torch.tensor(
                [[[[0.1, 0.2], [0.3, 0.9]]]]).expand(batch, -1, -1, -1),
            "pred_boxes": torch.full((batch, 1, 4), 0.5),
        }
        expert_outputs = {
            name: dict(general) for name in model.shared_expert_names
        }
        expert_outputs["visibility_foc_ov"] = visibility
        output = dict(general)
        output["expert_outputs"] = expert_outputs
        return output

    model.forward_head = controlled_forward_head
    output = model(*_images(batch=1))
    response = output["expert_outputs"][
        "visibility_foc_ov"]["score_map"].flatten(1)
    expected = torch.stack((
        response.max(dim=-1).values,
        response.mean(dim=-1),
        response.std(dim=-1, unbiased=False),
    ), dim=-1)

    assert torch.allclose(model.visibility_gate.response_stats, expected)


def test_precision_expert_uses_detached_shared_proposal_and_trains_independently(
        monkeypatch):
    model = _model(expert_enabled=True)
    shared_grad_modes = []
    small_grad_modes = []
    shared_forward = model._run_backbone
    small_forward = model.small_target_expert.forward

    def counted_shared(*args, **kwargs):
        shared_grad_modes.append(torch.is_grad_enabled())
        return shared_forward(*args, **kwargs)

    def counted_small(*args, **kwargs):
        small_grad_modes.append(torch.is_grad_enabled())
        return small_forward(*args, **kwargs)

    monkeypatch.setattr(model, "_run_backbone", counted_shared)
    monkeypatch.setattr(model.small_target_expert, "forward", counted_small)

    output = model(
        *_images(batch=1), training_expert_ids=torch.tensor([2]))
    loss = output["score_map"].mean() + output["pred_boxes"].mean()
    loss.backward()

    assert "expert_owner_id" not in output
    assert shared_grad_modes == [False]
    assert small_grad_modes == [True]
    assert "upstream_pred_boxes" in output
    assert output["score_map"].shape[-2:] == (4, 4)
    assert all(
        parameter.grad is not None
        for parameter in model.small_target_expert.parameters())
    assert all(
        parameter.grad is not None
        for parameter in model.proposal_adapters[
            "precision_refiner"].parameters())
    assert all(
        parameter.grad is None
        for module in (
            model.backbone, model.memory, model.box_head,
            model.expert_fusion, model.visibility_gate,
        )
        for parameter in module.parameters())


def test_inference_runs_shared_and_small_paths_once_and_returns_five_candidates(
        monkeypatch):
    model = _model(expert_enabled=True).eval()
    static_zi = torch.randn(1, 3, 16, 16)
    static_ze = torch.randn(1, 3, 16, 16)
    dynamic_zi = torch.randn(1, 1, 3, 16, 16)
    dynamic_ze = torch.randn(1, 1, 3, 16, 16)
    xi = torch.randn(1, 1, 3, 16, 16)
    xe = torch.randn(1, 1, 3, 16, 16)

    with torch.no_grad():
        encoded = model._encode_runtime_templates(
            static_zi, static_ze, dynamic_zi, dynamic_ze)
        shared_only = model._forward_srbt(
            encoded[0], encoded[1], xi, xe,
            encoded_templates=encoded)

    calls = {"shared": 0, "small": 0}
    shared_forward = model._run_encoded_backbone
    small_forward = model.small_target_expert.forward

    def counted_shared(*args, **kwargs):
        calls["shared"] += 1
        return shared_forward(*args, **kwargs)

    def counted_small(*args, **kwargs):
        calls["small"] += 1
        assert torch.equal(args[0], static_zi)
        assert torch.equal(args[1], static_ze)
        return small_forward(*args, **kwargs)

    monkeypatch.setattr(model, "_run_encoded_backbone", counted_shared)
    monkeypatch.setattr(model.small_target_expert, "forward", counted_small)

    with torch.no_grad():
        output = model.inference(
            static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe)

    assert calls == {"shared": 1, "small": 1}
    assert tuple(output["expert_outputs"]) == tuple(model.expert_names)
    assert output["expert_outputs"]["precision_refiner"][
        "score_map"].shape[-2:] == (4, 4)
    for name in model.shared_expert_names:
        for key, value in shared_only["expert_outputs"][name].items():
            assert torch.equal(output["expert_outputs"][name][key], value)
    for key, value in shared_only.items():
        if key == "expert_outputs":
            continue
        if torch.is_tensor(value):
            assert torch.equal(output[key], value)
        elif isinstance(value, dict):
            assert value.keys() == output[key].keys()
            for nested_key, nested_value in value.items():
                assert torch.equal(output[key][nested_key], nested_value)
        else:
            assert output[key] == value


def test_expert_inference_rejects_encoded_static_templates_before_shared_forward(
        monkeypatch):
    model = _model(expert_enabled=True).eval()
    static_zi = model.backbone._z_feat(torch.randn(1, 1, 3, 16, 16))
    static_ze = model.backbone._z_feat(torch.randn(1, 1, 3, 16, 16))
    calls = {"shared": 0}
    shared_forward = model._run_encoded_backbone

    def counted_shared(*args, **kwargs):
        calls["shared"] += 1
        return shared_forward(*args, **kwargs)

    monkeypatch.setattr(model, "_run_encoded_backbone", counted_shared)
    with pytest.raises(ValueError, match="requires raw static RGB/event"):
        model.inference(
            static_zi, static_ze,
            torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16),
            torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16),
        )
    assert calls == {"shared": 0}


def test_inference_reuses_cached_independent_template_features(monkeypatch):
    model = _model(expert_enabled=True).eval()
    static_zi = torch.randn(1, 3, 16, 16)
    static_ze = torch.randn(1, 3, 16, 16)
    cached = model.small_target_expert.encode_template(static_zi, static_ze)
    calls = {"full": 0, "cached": 0}
    full_forward = model.small_target_expert.forward
    cached_forward = model.small_target_expert.track_with_template

    def counted_full(*args, **kwargs):
        calls["full"] += 1
        return full_forward(*args, **kwargs)

    def counted_cached(*args, **kwargs):
        calls["cached"] += 1
        assert args[0] is cached
        return cached_forward(*args, **kwargs)

    monkeypatch.setattr(model.small_target_expert, "forward", counted_full)
    monkeypatch.setattr(
        model.small_target_expert, "track_with_template", counted_cached)
    with torch.no_grad():
        output = model.inference(
            static_zi, static_ze,
            torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            small_template_features=cached,
        )

    assert calls == {"full": 0, "cached": 1}
    assert "precision_refiner" in output["expert_outputs"]


def test_sparse_inference_runs_only_motion_and_required_generalist(monkeypatch):
    model = _model(expert_enabled=True).eval()
    calls = {name: 0 for name in model.shared_expert_names}
    for name in model.shared_expert_names:
        fusion = model.expert_fusion.experts[name]
        original = fusion.forward
        monkeypatch.setattr(
            fusion,
            "forward",
            lambda *args, _name=name, _forward=original, **kwargs: (
                calls.__setitem__(_name, calls[_name] + 1)
                or _forward(*args, **kwargs)
            ),
        )
    small_calls = {"count": 0}
    monkeypatch.setattr(
        model.small_target_expert,
        "forward",
        lambda *args, **kwargs: small_calls.__setitem__(
            "count", small_calls["count"] + 1),
    )

    with torch.no_grad():
        output = model.inference(
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            active_expert_names=("motion_fm",),
        )

    assert tuple(output["expert_outputs"]) == ("generalist", "motion_fm")
    assert calls == {
        "generalist": 1,
        "motion_fm": 1,
        "visibility_foc_ov": 0,
        "discrimination_bi": 0,
    }
    assert small_calls == {"count": 0}


def test_sparse_generalist_inference_accepts_encoded_static_templates(
        monkeypatch):
    model = _model(expert_enabled=True).eval()
    monkeypatch.setattr(
        model.small_target_expert,
        "forward",
        lambda *args, **kwargs: pytest.fail(
            "inactive precision expert must not execute"),
    )
    static_zi = model.backbone._z_feat(torch.randn(1, 1, 3, 16, 16))
    static_ze = model.backbone._z_feat(torch.randn(1, 1, 3, 16, 16))

    with torch.no_grad():
        output = model.inference(
            static_zi,
            static_ze,
            torch.randn(1, 4, 8),
            torch.randn(1, 4, 8),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            active_expert_names=(),
        )

    assert tuple(output["expert_outputs"]) == ("generalist",)


def test_sparse_inference_rejects_unknown_expert_before_backbone(monkeypatch):
    model = _model(expert_enabled=True).eval()
    monkeypatch.setattr(
        model,
        "_encode_runtime_templates",
        lambda *args, **kwargs: pytest.fail(
            "unknown expert must fail before backbone work"),
    )

    with pytest.raises(ValueError, match="unknown active expert"):
        model.inference(
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            active_expert_names=("not_an_expert",),
        )


def test_auto_activation_executes_only_selected_specialist(monkeypatch):
    model = _model(expert_enabled=True).eval()
    calls = {name: 0 for name in model.shared_expert_names}
    for name in model.shared_expert_names:
        fusion = model.expert_fusion.experts[name]
        original = fusion.forward
        monkeypatch.setattr(
            fusion,
            "forward",
            lambda *args, _name=name, _forward=original, **kwargs: (
                calls.__setitem__(_name, calls[_name] + 1)
                or _forward(*args, **kwargs)
            ),
        )
    monkeypatch.setattr(
        model.expert_activator,
        "forward",
        lambda rgb, event, score, boxes: score.new_tensor(
            [[10.0, -10.0, -10.0, -10.0]]),
    )

    with torch.no_grad():
        output = model.inference(
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            auto_activate=True,
        )

    assert tuple(output["expert_outputs"]) == ("generalist", "motion_fm")
    assert torch.equal(
        output["expert_activation_mask"],
        torch.tensor([[True, True, False, False, False]]),
    )
    assert calls == {
        "generalist": 1,
        "motion_fm": 1,
        "visibility_foc_ov": 0,
        "discrimination_bi": 0,
    }


def test_auto_activation_does_not_expand_selected_specialists(monkeypatch):
    model = _model(expert_enabled=True).eval()
    calls = {name: 0 for name in model.shared_expert_names}
    for name in model.shared_expert_names:
        fusion = model.expert_fusion.experts[name]
        original = fusion.forward
        monkeypatch.setattr(
            fusion,
            "forward",
            lambda *args, _name=name, _forward=original, **kwargs: (
                calls.__setitem__(_name, calls[_name] + 1)
                or _forward(*args, **kwargs)
            ),
        )
    monkeypatch.setattr(
        model.expert_activator,
        "forward",
        lambda rgb, event, score, boxes: score.new_tensor(
            [[-10.0, 10.0, 9.0, -10.0]]),
    )

    with torch.no_grad():
        output = model.inference(
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            auto_activate=True,
        )

    assert tuple(output["expert_outputs"]) == (
        "generalist", "precision_refiner", "visibility_foc_ov")
    assert torch.equal(
        output["expert_activation_mask"],
        torch.tensor([[True, False, True, True, False]]),
    )
    assert calls == {
        "generalist": 1,
        "motion_fm": 0,
        "visibility_foc_ov": 1,
        "discrimination_bi": 0,
    }


def test_auto_activation_composes_only_selected_motion_and_precision(monkeypatch):
    model = _model(expert_enabled=True).eval()
    monkeypatch.setattr(
        model.expert_activator,
        "forward",
        lambda rgb, event, score, boxes: score.new_tensor(
            [[10.0, 9.0, -10.0, -10.0]]),
    )

    with torch.no_grad():
        output = model.inference(
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            auto_activate=True,
        )

    assert tuple(output["expert_outputs"]) == (
        "generalist", "motion_fm", "precision_refiner")
    assert torch.equal(
        output["expert_outputs"]["precision_refiner"]["upstream_pred_boxes"],
        output["expert_outputs"]["motion_fm"]["pred_boxes"],
    )


def test_explicit_and_automatic_activation_are_mutually_exclusive():
    model = _model(expert_enabled=True).eval()

    with pytest.raises(ValueError, match="mutually exclusive"):
        model.inference(
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            torch.randn(1, 1, 3, 16, 16),
            active_expert_names=("motion_fm",),
            auto_activate=True,
        )


def test_dispatch_forward_returns_logits_and_all_frozen_expert_candidates():
    model = _model(expert_enabled=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.expert_activator.parameters():
        parameter.requires_grad_(True)

    output = model(*_images(batch=2), return_activation_logits=True)
    output["expert_activation_logits"].sum().backward()

    assert output["expert_activation_logits"].shape == (2, 4)
    assert tuple(output["expert_outputs"]) == tuple(model.expert_names)
    assert all(
        parameter.grad is not None
        for parameter in model.expert_activator.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("expert_activator."))


def test_training_forward_runs_redetect_only_when_observations_are_requested():
    model = _model()
    probe = ProbeRedetect()
    model.redetect_expert = probe
    zi, ze, xi, xe = _images(batch=2)

    ordinary = model(zi, ze, xi, xe)
    assert "redetect_predictions" not in ordinary

    output = model(
        zi, ze, xi, xe,
        redetect_images=torch.zeros(2, 1, 3, 16, 16),
        redetect_event_images=torch.ones(2, 1, 3, 16, 16),
        redetect_mask=torch.tensor([True, True]),
    )
    assert torch.equal(
        output["redetect_predictions"]["batch_indices"], torch.tensor([0, 1]))
    identity_scores = output["redetect_predictions"]["identity_scores"]
    assert identity_scores.shape == (2, 2)
    assert torch.isfinite(identity_scores).all()
    identity_scores.sum().backward()
    assert model.rgb_identity_verifier.projection[1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert probe.last_feat.shape[0] == 2


def test_recovery_optimizer_step_preserves_every_normal_path_parameter():
    model = _model(expert_enabled=True)
    model.redetect_expert = ProbeRedetect()
    optimizer_cfg = SimpleNamespace(TRAIN=SimpleNamespace(
        LR=1e-4,
        WEIGHT_DECAY=1e-4,
        EXPERT_LR_MULTIPLIER=5.0,
        EXPERT_PHASE="recovery",
    ))
    groups = _optimizer_groups(model, optimizer_cfg)
    optimizer = torch.optim.AdamW(groups)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    output = model(
        *_images(batch=2),
        redetect_images=torch.randn(2, 1, 3, 16, 16),
        redetect_event_images=torch.randn(2, 1, 3, 16, 16),
        redetect_mask=torch.tensor([True, True]),
    )
    recovery = output["redetect_predictions"]
    loss = (
        output["presence_predictions"]["logits"].square().mean()
        + recovery["field"].square().mean()
        + recovery["identity_scores"].square().mean()
    )
    loss.backward()
    optimizer.step()

    for name, parameter in model.named_parameters():
        if name in frozen_before:
            assert parameter.grad is None, name
            assert torch.equal(parameter.detach(), frozen_before[name]), name
    assert all(
        parameter.grad is not None
        for parameter in model.visibility_gate.parameters()
    )
    assert all(
        parameter.grad is not None
        for parameter in model.rgb_identity_verifier.parameters()
    )
    assert model.redetect_expert.scale.grad is not None


def test_inference_accepts_raw_and_preencoded_template_tokens():
    model = _model()
    static_rgb = torch.randn(1, 3, 16, 16)
    static_event = torch.randn(1, 3, 16, 16)
    dynamic_rgb = torch.randn(1, 3, 16, 16)
    dynamic_event = torch.randn(1, 3, 16, 16)
    xi = torch.randn(1, 3, 16, 16)
    xe = torch.randn(1, 3, 16, 16)

    raw = model.inference(
        static_rgb, static_event, dynamic_rgb, dynamic_event, xi, xe)
    encoded = model.inference(
        model.backbone._z_feat(static_rgb.unsqueeze(1)),
        model.backbone._z_feat(static_event.unsqueeze(1)),
        model.backbone._z_feat(dynamic_rgb.unsqueeze(1)),
        model.backbone._z_feat(dynamic_event.unsqueeze(1)),
        xi, xe,
    )
    assert raw["pred_boxes"].shape == encoded["pred_boxes"].shape
    assert set(raw["presence_predictions"]) == set(
        encoded["presence_predictions"])


def test_non_recovery_inference_accepts_mixed_runtime_templates():
    model = _model(srbt_enabled=False)
    static_rgb = torch.randn(1, 3, 16, 16)
    static_event = torch.randn(1, 3, 16, 16)
    dynamic_rgb = model.backbone._z_feat(torch.randn(1, 1, 3, 16, 16))
    dynamic_event = model.backbone._z_feat(torch.randn(1, 1, 3, 16, 16))
    output = model.inference(
        static_rgb, static_event, dynamic_rgb, dynamic_event,
        torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16))
    assert output["pred_boxes"].shape == (1, 1, 4)
