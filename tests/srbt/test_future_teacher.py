import inspect
from pathlib import Path

import pytest
import torch
import yaml

from lib.config.pet_track.config import cfg as default_cfg
from lib.models.layers.srbt_teacher import FuturePosteriorTeacher
from lib.models.pet_track.pet_track import PETTrack


EVIDENCE_NAMES = ("appearance", "motion", "detail", "cross", "identity")


def _teacher():
    return FuturePosteriorTeacher(
        input_dim=8,
        pooled_dim=4,
        d_model=16,
        nhead=4,
        num_layers=2,
        ffn_dim=32,
        spatial_dim=6,
        identity_dim=4,
        hazard_bins=9,
    )


def _maps(batch=2, horizon=4, grid=16, requires_grad=False):
    torch.manual_seed(7)
    return {
        name: torch.randn(
            batch, horizon, 8, grid, grid,
            requires_grad=requires_grad,
        )
        for name in EVIDENCE_NAMES
    }


def test_teacher_preserves_grid_and_returns_normalized_soft_posteriors():
    teacher = _teacher()
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    output = teacher(_maps(), valid)

    assert output["existence"].shape == (2,)
    assert output["hazard"].shape == (2, 9)
    assert output["time_weights"].shape == (2, 4)
    assert output["field"].shape == (2, 1, 16, 16)
    assert output["candidate_map"].shape == (2, 1, 16, 16)
    assert output["identity_map"].shape == (2, 4, 16, 16)
    assert torch.allclose(output["hazard"].sum(-1), torch.ones(2))
    assert torch.allclose(output["time_weights"].sum(-1), torch.ones(2))
    assert torch.all(output["time_weights"][~valid] == 0)
    assert torch.allclose(
        output["field"].flatten(1).sum(-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(
        output["candidate_map"].flatten(1).sum(-1),
        torch.ones(2), atol=1e-5)


def test_masked_future_frames_have_no_effect_on_any_teacher_output():
    teacher = _teacher().eval()
    valid = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    original = _maps(batch=1)
    changed = {name: value.clone() for name, value in original.items()}
    for value in changed.values():
        value[:, 2:] = torch.randn_like(value[:, 2:]) * 1000.0

    with torch.no_grad():
        first = teacher(original, valid)
        second = teacher(changed, valid)

    for name in first:
        assert torch.allclose(first[name], second[name], atol=1e-6), name


def test_teacher_mapping_key_order_does_not_change_outputs():
    teacher = _teacher().eval()
    valid = torch.ones(1, 4, dtype=torch.bool)
    original = _maps(batch=1)
    reordered = dict(reversed(tuple(original.items())))

    with torch.no_grad():
        first = teacher(original, valid)
        second = teacher(reordered, valid)

    for name in first:
        assert torch.allclose(first[name], second[name], atol=1e-6), name


def test_teacher_detaches_future_evidence_but_trains_its_own_heads():
    teacher = _teacher()
    evidence = _maps(batch=1, requires_grad=True)
    valid = torch.ones(1, 4, dtype=torch.bool)

    output = teacher(evidence, valid)
    sum(value.float().sum() for value in output.values()).backward()

    assert all(value.grad is None for value in evidence.values())
    assert any(parameter.grad is not None for parameter in teacher.parameters())


def test_teacher_rejects_an_all_padding_sequence():
    with pytest.raises(ValueError, match="valid future frame"):
        _teacher()(_maps(batch=1), torch.zeros(1, 4, dtype=torch.bool))


def test_eval_inference_has_no_future_or_teacher_argument():
    parameters = inspect.signature(PETTrack.inference).parameters
    assert not any("future" in name or "teacher" in name for name in parameters)


def test_default_and_canonical_teacher_configuration_is_exact():
    assert dict(default_cfg.MODEL.SRBT.TEACHER) == {
        "INPUT_DIM": 768,
        "POOLED_DIM": 128,
        "D_MODEL": 256,
        "NHEAD": 8,
        "NUM_LAYERS": 2,
        "FFN_DIM": 1024,
        "SPATIAL_DIM": 64,
        "IDENTITY_DIM": 32,
        "HAZARD_BINS": 129,
        "MAX_HORIZON": 128,
        "ENCODE_CHUNK_SIZE": 2,
    }

    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments" / "pet_track" / "felt_pet_track.yaml"
    )
    configured = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert configured["MODEL"]["SRBT"]["TEACHER"] == dict(
        default_cfg.MODEL.SRBT.TEACHER)
