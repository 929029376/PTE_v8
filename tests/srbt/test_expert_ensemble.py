import pytest
import torch

from lib.models.layers.expert_ensemble import (
    fuse_expert_predictions,
    normalized_response_psr,
)


def test_four_expert_cluster_rejects_confident_spatial_outlier():
    boxes = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],
        [10.5, 10.0, 20.0, 20.0],
        [10.0, 10.5, 20.0, 20.0],
        [10.5, 10.5, 20.0, 20.0],
        [80.0, 80.0, 10.0, 10.0],
    ])

    result = fuse_expert_predictions(
        boxes=boxes,
        response_peaks=torch.tensor([0.8, 0.8, 0.7, 0.7, 0.99]),
        response_psr=torch.ones(5),
        last_box=torch.tensor([10.0, 10.0, 20.0, 20.0]),
    )

    assert result.retained_ids == (0, 1, 2, 3)
    assert result.weights[4].item() == 0.0
    assert result.box[:2].tolist() == pytest.approx([10.25, 10.25], abs=0.3)


def test_quality_uses_declared_exponents_exactly():
    boxes = torch.tensor([[5.0, 5.0, 10.0, 10.0]] * 5)
    peaks = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
    psr = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])

    result = fuse_expert_predictions(
        boxes=boxes,
        response_peaks=peaks,
        response_psr=psr,
        last_box=boxes[0],
    )

    expected = peaks.pow(0.35) * psr.pow(0.25)
    assert torch.allclose(result.quality, expected, atol=1e-6)
    assert result.retained_ids == (0, 1, 2, 3, 4)


def test_zero_quality_falls_back_to_highest_response_peak():
    boxes = torch.tensor([
        [0.0, 0.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0],
        [10.0, 10.0, 5.0, 5.0],
        [15.0, 15.0, 5.0, 5.0],
        [20.0, 20.0, 5.0, 5.0],
    ])

    result = fuse_expert_predictions(
        boxes=boxes,
        response_peaks=torch.tensor([0.1, 0.6, 0.2, 0.4, 0.3]),
        response_psr=torch.zeros(5),
        last_box=None,
    )

    assert result.retained_ids == (1,)
    assert torch.equal(result.box, boxes[1])
    assert torch.equal(result.weights, torch.tensor([0., 1., 0., 0., 0.]))


def test_response_psr_is_finite_and_bounded_for_flat_maps():
    psr = normalized_response_psr(torch.ones(5, 1, 4, 4))

    assert psr.shape == (5,)
    assert torch.isfinite(psr).all()
    assert torch.all((0.0 <= psr) & (psr <= 1.0))
