import pytest
import torch

from lib.models.layers.expert_ensemble import (
    ExpertActivator,
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

    assert result.retained_ids == (0,)
    assert result.weights[4].item() == 0.0
    assert torch.equal(result.box, boxes[0])


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
    assert result.retained_ids == (4,)
    assert torch.equal(result.box, boxes[4])


def test_uncertain_specialists_fall_back_to_exact_generalist_box():
    boxes = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],
        [35.0, 10.0, 20.0, 20.0],
        [10.0, 35.0, 20.0, 20.0],
        [35.0, 35.0, 20.0, 20.0],
        [60.0, 60.0, 20.0, 20.0],
    ])

    result = fuse_expert_predictions(
        boxes=boxes,
        response_peaks=torch.tensor([0.80, 0.81, 0.79, 0.80, 0.82]),
        response_psr=torch.full((5,), 0.8),
        last_box=boxes[0],
    )

    assert result.retained_ids == (0,)
    assert torch.equal(result.box, boxes[0])
    assert torch.equal(result.weights, torch.tensor([1., 0., 0., 0., 0.]))


def test_clearly_superior_specialist_is_selected_without_box_averaging():
    boxes = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],
        [11.0, 10.0, 20.0, 20.0],
        [10.0, 11.0, 20.0, 20.0],
        [11.0, 11.0, 20.0, 20.0],
        [80.0, 80.0, 10.0, 10.0],
    ])

    result = fuse_expert_predictions(
        boxes=boxes,
        response_peaks=torch.tensor([0.55, 0.95, 0.60, 0.58, 0.99]),
        response_psr=torch.tensor([0.55, 0.98, 0.60, 0.58, 0.99]),
        last_box=boxes[0],
    )

    assert result.retained_ids == (1,)
    assert torch.equal(result.box, boxes[1])
    assert torch.equal(result.weights, torch.tensor([0., 1., 0., 0., 0.]))


def test_zero_quality_falls_back_to_generalist():
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

    assert result.retained_ids == (0,)
    assert torch.equal(result.box, boxes[0])
    assert torch.equal(result.weights, torch.tensor([1., 0., 0., 0., 0.]))


def test_response_psr_is_finite_and_bounded_for_flat_maps():
    psr = normalized_response_psr(torch.ones(5, 1, 4, 4))

    assert psr.shape == (5,)
    assert torch.isfinite(psr).all()
    assert torch.all((0.0 <= psr) & (psr <= 1.0))


def test_expert_activator_selects_at_most_two_specialists_above_threshold():
    activator = ExpertActivator(
        embed_dim=4,
        specialist_count=4,
        hidden_dim=8,
        threshold=0.60,
        max_specialists=2,
    )
    logits = torch.tensor([
        [2.0, 1.0, -2.0, 0.5],
        [-2.0, -3.0, -4.0, -5.0],
    ])

    selected = activator.select(logits)

    assert torch.equal(
        selected,
        torch.tensor([
            [True, True, False, False],
            [False, False, False, False],
        ]),
    )


def test_expert_activator_uses_rgb_event_and_generalist_observations():
    activator = ExpertActivator(
        embed_dim=4,
        specialist_count=4,
        hidden_dim=8,
        threshold=0.50,
        max_specialists=2,
    )
    rgb = torch.randn(2, 6, 4)
    event = torch.randn(2, 6, 4)
    score_map = torch.randn(2, 1, 3, 3)
    boxes = torch.rand(2, 1, 4)

    logits = activator(rgb, event, score_map, boxes)
    logits.sum().backward()

    assert logits.shape == (2, 4)
    assert all(parameter.grad is not None for parameter in activator.parameters())
