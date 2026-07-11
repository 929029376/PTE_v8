import inspect
from types import SimpleNamespace

import pytest
import torch
from easydict import EasyDict as edict

from lib.models.pet_track.pet_track import PETTrack


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        self.num_heads = 2
        self.probe = torch.nn.Parameter(torch.ones(()))
        self.future_grad_calls = 0
        self.no_grad_x_batches = []
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(12)])
        self.norm = torch.nn.LayerNorm(8)

    def _z_feat(self, value):
        batch = value.shape[0]
        return torch.ones(batch, 4, 8, device=value.device) * self.probe

    def _x_feat(self, value):
        if not torch.is_grad_enabled():
            self.no_grad_x_batches.append(value.shape[0])
        if torch.is_grad_enabled() and value.requires_grad:
            self.future_grad_calls += 1
        batch = value.shape[0]
        scale = value.mean(dim=tuple(range(1, value.ndim)), keepdim=False).view(batch, 1, 1)
        return torch.ones(batch, 4, 8, device=value.device) * (self.probe + scale)

    def forward(self, static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe, **_):
        return torch.cat((static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe), dim=1), {}


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


def _cfg():
    return edict({
        "MODEL": {
            "PET": {"ENABLE": False},
            "SRBT": {
                "ENABLE": True,
                "EVIDENCE": {
                    "STATE_DIM": 32,
                    "QUALITY_DIM": 16,
                    "GATE_HIDDEN_DIM": 8,
                    "GATE_EPSILON": 1.0,
                    "POOL_SIZE": 1,
                },
                "BELIEF": {
                    "STATE_DIM": 32,
                    "QUALITY_DIM": 16,
                    "HIDDEN_DIM": 8,
                    "MAX_HAZARD": 8,
                    "REAPPEARING_MAX_FRAMES": 3,
                },
                "TEACHER": {
                    "INPUT_DIM": 8,
                    "POOLED_DIM": 4,
                    "D_MODEL": 8,
                    "NHEAD": 2,
                    "NUM_LAYERS": 1,
                    "FFN_DIM": 16,
                    "SPATIAL_DIM": 4,
                    "IDENTITY_DIM": 4,
                    "HAZARD_BINS": 9,
                    "MAX_HORIZON": 4,
                    "ENCODE_CHUNK_SIZE": 2,
                },
            },
        },
        "TRAIN": {},
    })


def _model():
    return PETTrack(TinyBackbone(), TinyMemory(), TinyHead(), _cfg(), head_type="CENTER")


def _images(batch=2):
    return torch.randn(batch, 2, 3, 16, 16), torch.randn(batch, 2, 3, 16, 16), torch.randn(batch, 1, 3, 16, 16), torch.randn(batch, 1, 3, 16, 16)


def test_srbt_replaces_legacy_route_entry_points():
    assert not hasattr(PETTrack, "_forward_routed_backbone")
    assert not hasattr(PETTrack, "set_expert_training_stage")
    assert not any("future" in name or "teacher" in name or "gt" in name
                   for name in inspect.signature(PETTrack.inference).parameters)


def test_training_forward_returns_unified_srbt_contract_and_teacher_only_with_future():
    model = _model()
    zi, ze, xi, xe = _images()
    previous = model.initialize_srbt_posterior(zi.shape[0], zi.device, zi.dtype)

    eval_out = model(zi, ze, xi, xe, previous_posterior=previous)
    assert {"target_bbox", "absent", "pred_score", "response",
            "srbt_posterior", "hypotheses", "srbt_predictions"} <= set(eval_out)
    assert "srbt_teacher" not in eval_out
    assert set(eval_out["srbt_predictions"]) == {
        "existence_logits", "hazard_logits", "field_logits",
        "candidate_logits", "hypothesis_boxes", "hypothesis_scores",
        "identity_embeddings", "template_identity",
    }
    assert eval_out["srbt_posterior"] is eval_out["belief"]
    assert torch.equal(
        eval_out["target_bbox"],
        eval_out["srbt_predictions"]["hypothesis_boxes"][:, 0],
    )
    assert eval_out["srbt_predictions"]["hypothesis_boxes"].shape[1] > 1
    assert eval_out["srbt_quality_stats"].shape[-1] == 16
    assert eval_out["srbt_quality_stats"].abs().sum() > 0

    future_images = torch.randn(2, 3, 3, 16, 16, requires_grad=True)
    future_event_images = torch.randn(2, 3, 3, 16, 16, requires_grad=True)
    future_valid = torch.ones(2, 3, dtype=torch.bool)
    train_out = model(zi, ze, xi, xe, previous_posterior=previous,
                      future_images=future_images,
                      future_event_images=future_event_images,
                      future_valid=future_valid)
    assert "srbt_teacher" in train_out
    teacher_loss = train_out["srbt_teacher"]["existence"].sum()
    teacher_loss.backward()
    assert any(p.grad is not None for p in model.srbt_teacher.parameters())
    assert model.backbone.probe.grad is None
    assert model.backbone.future_grad_calls == 0
    assert model.backbone.no_grad_x_batches == [4, 4, 2, 2]


def test_eval_forward_rejects_future_teacher_inputs():
    model = _model().eval()
    zi, ze, xi, xe = _images()
    with pytest.raises(RuntimeError, match="training-only"):
        model(
            zi, ze, xi, xe,
            future_images=torch.randn(2, 2, 3, 16, 16),
            future_event_images=torch.randn(2, 2, 3, 16, 16),
            future_valid=torch.ones(2, 2, dtype=torch.bool),
        )


def test_inference_accepts_raw_and_preencoded_template_tokens():
    model = _model()
    posterior = model.initialize_srbt_posterior(1, torch.device("cpu"), torch.float32)
    zi, ze, xi, xe = _images(batch=1)

    raw = model.inference(zi[:, 0], ze[:, 0], zi[:, 1:], ze[:, 1:],
                          xi, xe, previous_posterior=posterior)
    static_zi, static_ze, dynamic_zi, dynamic_ze = model._encode_templates(zi, ze)
    encoded = model.inference(static_zi, static_ze, dynamic_zi, dynamic_ze,
                              xi, xe, previous_posterior=posterior)

    assert raw["target_bbox"].shape == encoded["target_bbox"].shape
    assert set(raw["srbt_predictions"]) == set(encoded["srbt_predictions"])


def test_srbt_stateful_visible_absent_reappearance_outputs_are_causal():
    model = _model()
    posterior = model.initialize_srbt_posterior(1, torch.device("cpu"), torch.float32)
    zi, ze, xi, xe = _images(batch=1)
    scores = []
    for _ in range(3):
        out = model.inference(zi[:, 0], ze[:, 0], zi[:, 1:], ze[:, 1:],
                              xi, xe, previous_posterior=posterior)
        posterior = out["srbt_posterior"]
        scores.append(float(out["pred_score"][0].detach()))
    assert all(0.0 <= score <= 1.0 for score in scores)
