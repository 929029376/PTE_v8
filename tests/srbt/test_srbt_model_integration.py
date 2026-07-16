import inspect
from types import SimpleNamespace

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
                "generalist", "motion_fm", "small_target_st",
                "visibility_foc_ov", "discrimination_bi",
            ],
        }
    return edict({"MODEL": model, "TRAIN": {}})


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


def test_all_expert_outputs_are_returned_inside_model_forward():
    model = _model(expert_enabled=True)
    output = model(*_images(batch=1))

    assert tuple(output["expert_outputs"]) == tuple(model.expert_names)
    assert len(output["expert_outputs"]) == 5


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
