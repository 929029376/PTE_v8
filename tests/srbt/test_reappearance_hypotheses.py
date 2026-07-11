import pytest
import torch

from lib.models.layers.redetection import RedetectionExpert
from lib.models.layers.srbt_hypotheses import (
    HypothesisTracker,
    extract_hypotheses,
)


def _maps(size=8, identity_dim=4):
    field = torch.zeros(1, 1, size, size)
    candidate = torch.ones_like(field)
    size_map = torch.full((1, 2, size, size), 0.2)
    offset_map = torch.full((1, 2, size, size), 0.5)
    identity_map = torch.zeros(1, identity_dim, size, size)
    return field, candidate, size_map, offset_map, identity_map


def _state(boxes, weights, identity, velocity=None, age=None):
    boxes = torch.tensor(boxes, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)
    identity = torch.tensor(identity, dtype=torch.float32)
    if velocity is None:
        velocity = torch.zeros_like(boxes)
    else:
        velocity = torch.tensor(velocity, dtype=torch.float32)
    if age is None:
        age = torch.zeros(len(weights), dtype=torch.long)
    else:
        age = torch.tensor(age, dtype=torch.long)
    return {
        "boxes": boxes,
        "weights": weights,
        "identity": identity,
        "velocity": velocity,
        "age": age,
    }


def test_extract_uses_three_by_three_local_maxima_and_decodes_normalized_boxes():
    field, candidate, size_map, offset_map, identity_map = _maps(size=6)
    field[0, 0, 1, 1] = 1.0
    field[0, 0, 1, 2] = 0.95
    field[0, 0, 4, 4] = 0.8
    size_map[0, :, 1, 1] = torch.tensor([0.25, 0.30])
    identity_map[0, :, 1, 1] = torch.tensor([3.0, 4.0, 0.0, 0.0])

    hypotheses = extract_hypotheses(
        field, candidate, size_map, offset_map, identity_map, k_max=5)

    assert hypotheses["count"].tolist() == [2]
    assert hypotheses["indices"][0, :2].tolist() == [7, 28]
    assert 8 not in hypotheses["indices"][0, :2].tolist()
    assert torch.allclose(
        hypotheses["boxes"][0, 0],
        torch.tensor([1.5 / 6, 1.5 / 6, 0.25, 0.30]),
    )
    assert torch.allclose(
        hypotheses["identity"][0, 0],
        torch.tensor([0.6, 0.8, 0.0, 0.0]),
    )
    assert torch.allclose(hypotheses["posterior"].sum(dim=1), torch.ones(1))
    assert hypotheses["active_mask"].sum().item() == 2


@pytest.mark.parametrize("peak_count", range(1, 6))
def test_extract_activates_between_one_and_five_diverse_hypotheses(peak_count):
    field, candidate, size_map, offset_map, identity_map = _maps(size=12)
    positions = [(1, 1), (1, 4), (4, 1), (4, 4), (8, 8)]
    for y, x in positions[:peak_count]:
        field[0, 0, y, x] = 1.0
        identity_map[0, :, y, x] = torch.tensor([1.0, x, y, 0.5])

    hypotheses = extract_hypotheses(
        field, candidate, size_map, offset_map, identity_map, k_max=5)

    assert hypotheses["count"].item() == peak_count
    assert hypotheses["active_mask"].sum().item() == peak_count
    assert hypotheses["boxes"].shape == (1, 5, 4)
    assert hypotheses["identity"].shape == (1, 5, 4)
    active_boxes = hypotheses["boxes"][hypotheses["active_mask"]]
    assert ((active_boxes >= 0.0) & (active_boxes <= 1.0)).all()


def test_zero_field_returns_one_finite_fallback_hypothesis():
    maps = _maps(size=8)

    hypotheses = extract_hypotheses(*maps, k_max=5)

    assert hypotheses["count"].tolist() == [1]
    assert hypotheses["active_mask"].sum().item() == 1
    assert torch.isfinite(hypotheses["boxes"]).all()
    assert torch.allclose(hypotheses["posterior"].sum(dim=1), torch.ones(1))


def test_candidate_confidence_precedes_nms_and_batch_items_keep_distinct_counts():
    field = torch.zeros(2, 1, 8, 8)
    candidate = torch.ones_like(field)
    size_map = torch.full((2, 2, 8, 8), 0.2)
    offset_map = torch.full((2, 2, 8, 8), 0.5)
    identity_map = torch.randn(2, 3, 8, 8)
    field[0, 0, 0, 0] = 1.0
    field[0, 0, 7, 7] = 0.9
    candidate[0, 0, 0, 0] = 0.01
    field[1, 0, 2, 2] = 0.95

    hypotheses = extract_hypotheses(
        field, candidate, size_map, offset_map, identity_map, k_max=5)

    assert hypotheses["indices"][0, 0].item() == 63
    assert hypotheses["count"].tolist() == [1, 1]
    assert hypotheses["active_mask"].sum(dim=1).tolist() == [1, 1]


def test_equal_adjacent_plateau_has_one_deterministic_peak():
    field, candidate, size_map, offset_map, identity_map = _maps(size=6)
    field[0, 0, 2, 2] = 1.0
    field[0, 0, 2, 3] = 1.0
    field[0, 0, 3, 2] = 1.0

    first = extract_hypotheses(
        field, candidate, size_map, offset_map, identity_map, k_max=5)
    second = extract_hypotheses(
        field, candidate, size_map, offset_map, identity_map, k_max=5)

    assert first["count"].item() == 1
    assert first["indices"][0, 0].item() == second["indices"][0, 0].item()


def test_redetection_uses_actual_token_grid_and_returns_srbt_maps():
    model = RedetectionExpert(
        inplanes=8,
        channel=32,
        feat_sz=12,
        stride=16,
        identity_dim=6,
        k_max=5,
    ).eval()
    feature = torch.randn(2, 8, 16, 16)

    with torch.no_grad():
        output = model(feature)

    assert output["field"].shape == (2, 1, 16, 16)
    assert output["candidate_map"].shape == (2, 1, 16, 16)
    assert output["size_map"].shape == (2, 2, 16, 16)
    assert output["offset_map"].shape == (2, 2, 16, 16)
    assert output["identity_map"].shape == (2, 6, 16, 16)
    assert output["hypotheses"]["boxes"].shape == (2, 5, 4)
    assert output["score_map"].data_ptr() == output["field"].data_ptr()
    assert torch.allclose(output["bbox"], output["hypotheses"]["boxes"][:, 0])


def test_selected_hypothesis_keeps_all_five_prediction_paths_trainable():
    field, candidate, size_map, offset_map, identity_map = _maps(
        size=6, identity_dim=3)
    field[0, 0, 2, 2] = 0.9
    identity_map[0, :, 2, 2] = torch.tensor([1.0, 2.0, 3.0])
    tensors = [field, candidate, size_map, offset_map, identity_map]
    for tensor in tensors:
        tensor.requires_grad_()

    hypotheses = extract_hypotheses(*tensors, k_max=5)
    loss = (
        hypotheses["scores"][0, 0]
        + hypotheses["boxes"][0, 0].sum()
        + hypotheses["identity"][0, 0].sum()
    )
    loss.backward()

    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_tracker_matches_and_updates_weight_velocity_and_identity():
    tracker = HypothesisTracker(k_max=5)
    previous = _state(
        boxes=[[0.20, 0.20, 0.20, 0.20]],
        weights=[0.8],
        identity=[[1.0, 0.0]],
    )
    observed = _state(
        boxes=[[0.30, 0.20, 0.20, 0.20]],
        weights=[0.6],
        identity=[[1.0, 0.0]],
    )

    updated = tracker.update(previous, observed)

    assert updated["boxes"].shape[0] == 1
    assert updated["weights"][0].item() == pytest.approx(0.74)
    assert updated["velocity"][0, 0].item() == pytest.approx(0.10)
    assert updated["age"].tolist() == [0]
    assert torch.allclose(updated["identity"][0], torch.tensor([1.0, 0.0]))


def test_tracker_rejects_bad_match_decays_history_and_adds_observation():
    tracker = HypothesisTracker(k_max=5)
    previous = _state(
        boxes=[[0.10, 0.10, 0.10, 0.10]],
        weights=[0.8],
        identity=[[1.0, 0.0]],
    )
    observed = _state(
        boxes=[[0.85, 0.85, 0.10, 0.10]],
        weights=[0.6],
        identity=[[-1.0, 0.0]],
    )

    updated = tracker.update(previous, observed)

    assert updated["boxes"].shape[0] == 2
    assert sorted(updated["weights"].tolist(), reverse=True) == pytest.approx([0.68, 0.6])
    assert sorted(updated["age"].tolist()) == [0, 1]


def test_tracker_merges_duplicate_peaks_and_uses_cumulative_posterior_mass():
    tracker = HypothesisTracker(k_max=5)
    observed = _state(
        boxes=[
            [0.20, 0.20, 0.20, 0.20],
            [0.21, 0.20, 0.20, 0.20],
            [0.60, 0.20, 0.15, 0.15],
            [0.20, 0.60, 0.15, 0.15],
            [0.60, 0.60, 0.15, 0.15],
        ],
        weights=[0.35, 0.25, 0.25, 0.10, 0.05],
        identity=[
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
    )

    updated = tracker.update(None, observed)

    assert updated["boxes"].shape[0] == 4
    assert updated["weights"][0].item() == pytest.approx(0.60)
    assert updated["active_count"] == 3
    assert updated["active_mask"].tolist() == [True, True, True, False]
    assert torch.allclose(updated["posterior"].sum(), torch.tensor(1.0))


def test_tracker_prunes_low_weight_and_over_age_hypotheses():
    tracker = HypothesisTracker(k_max=5)
    previous = _state(
        boxes=[[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]],
        weights=[0.05, 0.8],
        identity=[[1.0, 0.0], [0.0, 1.0]],
        age=[0, 32],
    )
    empty = _state(boxes=[], weights=[], identity=[])
    empty["identity"] = torch.empty(0, 2)
    empty["boxes"] = torch.empty(0, 4)
    empty["velocity"] = torch.empty(0, 4)

    updated = tracker.update(previous, empty)

    assert updated["boxes"].shape == (0, 4)
    assert updated["weights"].numel() == 0
    assert updated["active_count"] == 0


def test_tracker_keeps_age_32_and_prunes_only_after_it_is_exceeded():
    tracker = HypothesisTracker(k_max=5)
    previous = _state(
        boxes=[[0.2, 0.2, 0.1, 0.1]],
        weights=[0.8],
        identity=[[1.0, 0.0]],
        age=[31],
    )
    empty = _state(boxes=[], weights=[], identity=[])
    empty["identity"] = torch.empty(0, 2)
    empty["boxes"] = torch.empty(0, 4)
    empty["velocity"] = torch.empty(0, 4)

    updated = tracker.update(previous, empty)

    assert updated["age"].tolist() == [32]
    assert updated["weights"][0].item() == pytest.approx(0.68)
