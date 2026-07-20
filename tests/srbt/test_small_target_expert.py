import pytest
import torch

from lib.models.layers import small_target_expert as small_target_module
from lib.models.layers.small_target_expert import (
    SmallTargetExpert,
    _ModalityFusion,
)


def _inputs(batch_size=2):
    template_rgb = torch.randn(batch_size, 2, 3, 128, 128)
    template_event = torch.randn(batch_size, 2, 3, 128, 128)
    search_rgb = torch.randn(batch_size, 1, 3, 256, 256)
    search_event = torch.randn(batch_size, 1, 3, 256, 256)
    return template_rgb, template_event, search_rgb, search_event


def _base_maps(batch_size=2, feature_size=64, *, requires_grad=False):
    score = torch.rand(
        batch_size, 1, feature_size, feature_size).mul(0.8).add(0.1)
    size = torch.rand(
        batch_size, 2, feature_size, feature_size).mul(0.8).add(0.1)
    offset = torch.rand(
        batch_size, 2, feature_size, feature_size).sub(0.5)
    for value in (score, size, offset):
        value.requires_grad_(requires_grad)
    return score, size, offset


def _retained_v18_output(model, inputs):
    template_s4, template_s8 = model._encode_frames(
        inputs[0], inputs[1], average=True)
    search_s4, search_s8 = model._encode_frames(
        inputs[2], inputs[3], average=False)
    match_s4 = model._spatial_match(template_s4, search_s4)
    match_s8 = model._spatial_match(template_s8, search_s8)
    coarse = model.coarse_projection(search_s8)
    coarse = torch.nn.functional.interpolate(
        coarse, size=search_s4.shape[-2:],
        mode="bilinear", align_corners=False)
    match_s8 = torch.nn.functional.interpolate(
        match_s8, size=search_s4.shape[-2:],
        mode="bilinear", align_corners=False)
    feature = model.match_refinement(torch.cat((
        search_s4 * match_s4,
        coarse * match_s8,
        match_s4,
        match_s8,
    ), dim=1))
    score_map, size_map, offset_map = model.head(feature)
    return {
        "pred_boxes": model._decode_boxes(score_map, size_map, offset_map),
        "score_map": score_map,
        "size_map": size_map,
        "offset_map": offset_map,
        "small_match_map": match_s4,
    }


def test_independent_small_target_expert_matches_tracker_output_contract():
    model = SmallTargetExpert(search_size=256)

    output = model(*_inputs())

    assert output["pred_boxes"].shape == (2, 1, 4)
    assert output["score_map"].shape == (2, 1, 64, 64)
    assert output["size_map"].shape == (2, 2, 64, 64)
    assert output["offset_map"].shape == (2, 2, 64, 64)
    assert output["small_match_map"].shape == (2, 1, 64, 64)
    assert torch.isfinite(output["small_match_map"]).all()
    assert torch.isfinite(output["pred_boxes"]).all()
    assert ((output["pred_boxes"] >= 0.0) &
            (output["pred_boxes"] <= 1.0)).all()


def test_modality_fusion_uses_one_equivalent_two_way_gate():
    fusion = _ModalityFusion(channels=4)
    rgb = torch.zeros(2, 4, 8, 8)
    event = torch.ones_like(rgb)

    output = fusion(rgb, event)

    assert fusion.gate.out_channels == 1
    assert ((output >= rgb) & (output <= event)).all()


def test_template_token_box_refiner_is_lightweight_zero_box_source():
    torch.manual_seed(37)
    branch = small_target_module._TemplateTokenBoxRefiner()
    template_rgb = torch.randn(2, 3, 128, 128)
    template_event = torch.randn(2, 3, 128, 128)
    search_rgb = torch.randn(2, 3, 256, 256)
    search_event = torch.randn(2, 3, 256, 256)
    base_maps = _base_maps()

    tokens = branch.encode_template(template_rgb, template_event)
    box_delta = branch(tokens, search_rgb, search_event, *base_maps)

    assert len(tokens) == 2
    assert tokens[0].shape == (2, 16, 16)
    assert tokens[1].shape == (2, 16, 16)
    assert box_delta.shape == (2, 1, 4)
    assert torch.count_nonzero(box_delta).item() == 0
    assert sum(parameter.numel() for parameter in branch.parameters()) == 2_772

    with torch.no_grad():
        branch.output.weight.fill_(0.1)
        branch.output.bias.zero_()
        reference = branch(tokens, search_rgb, search_event, *base_maps)
        changed_rgb = branch(
            tokens, torch.zeros_like(search_rgb), search_event, *base_maps)
        changed_event = branch(
            tokens, search_rgb, torch.zeros_like(search_event), *base_maps)

    assert not torch.allclose(reference, changed_rgb)
    assert not torch.allclose(reference, changed_event)


def test_template_token_box_refiner_conditions_on_detached_base_prediction():
    torch.manual_seed(39)
    branch = small_target_module._TemplateTokenBoxRefiner()
    template_rgb = torch.randn(2, 3, 128, 128)
    template_event = torch.randn(2, 3, 128, 128)
    search_rgb = torch.randn(2, 3, 256, 256)
    search_event = torch.randn(2, 3, 256, 256)
    base_maps = _base_maps(requires_grad=True)
    changed_size = base_maps[1].detach().clone().zero_()
    tokens = branch.encode_template(template_rgb, template_event)

    with torch.no_grad():
        branch.output.weight.fill_(0.1)
        branch.output.bias.zero_()
    reference = branch(tokens, search_rgb, search_event, *base_maps)
    changed = branch(
        tokens, search_rgb, search_event,
        base_maps[0], changed_size, base_maps[2])

    assert not torch.allclose(reference, changed)
    reference.sum().backward()
    assert all(value.grad is None for value in base_maps)


def test_template_token_box_refiner_all_parameters_learn_after_zero_output_warmup():
    torch.manual_seed(41)
    branch = small_target_module._TemplateTokenBoxRefiner()
    parameters = list(branch.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.5)
    template_rgb = torch.randn(2, 3, 32, 32)
    template_event = torch.randn(2, 3, 32, 32)
    search_rgb = torch.randn(2, 3, 64, 64)
    search_event = torch.randn(2, 3, 64, 64)
    base_maps = _base_maps(batch_size=2, feature_size=16)
    target = torch.randn(2, 1, 4)

    for _ in range(2):
        optimizer.zero_grad()
        tokens = branch.encode_template(template_rgb, template_event)
        loss = (
            branch(tokens, search_rgb, search_event, *base_maps) - target
        ).square().mean()
        loss.backward()
        optimizer.step()

    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert all(torch.count_nonzero(parameter.grad).item() > 0
               for parameter in parameters)


def test_spatial_template_match_preserves_target_layout():
    template = torch.zeros(1, 2, 8, 8)
    template[:, 0, 2:4, 2] = 1.0
    template[:, 1, 2:4, 3] = 1.0
    search = torch.zeros(1, 2, 8, 8)
    search[:, 0, 1:3, 1] = 1.0
    search[:, 1, 1:3, 2] = 1.0
    search[:, 0, 5:7, 6] = 1.0
    search[:, 1, 5:7, 5] = 1.0

    response = SmallTargetExpert._spatial_match(template, search)

    assert response.shape == (1, 1, 8, 8)
    assert response[0, 0, 2, 2] > response[0, 0, 6, 6]


def test_independent_small_target_expert_uses_template_and_both_modalities():
    torch.manual_seed(17)
    model = SmallTargetExpert(search_size=256).eval()
    template_rgb, template_event, search_rgb, search_event = _inputs(batch_size=1)

    with torch.no_grad():
        reference = model(
            template_rgb, template_event, search_rgb, search_event)["score_map"]
        changed_template = model(
            -template_rgb, template_event, search_rgb, search_event)["score_map"]
        changed_template_event = model(
            template_rgb, -template_event, search_rgb, search_event)["score_map"]
        changed_event = model(
            template_rgb, template_event, search_rgb,
            torch.zeros_like(search_event))["score_map"]
        changed_rgb = model(
            template_rgb, template_event,
            torch.zeros_like(search_rgb), search_event)["score_map"]

    assert not torch.allclose(reference, changed_template)
    assert not torch.allclose(reference, changed_template_event)
    assert not torch.allclose(reference, changed_event)
    assert not torch.allclose(reference, changed_rgb)


def test_cached_template_features_match_full_small_target_forward():
    torch.manual_seed(23)
    model = SmallTargetExpert(search_size=256).eval()
    inputs = _inputs(batch_size=1)

    with torch.no_grad():
        expected = model(*inputs)
        template_features = model.encode_template(inputs[0], inputs[1])
        actual = model.track_with_template(
            template_features, inputs[2], inputs[3])

    assert len(template_features) == 3
    for key, value in expected.items():
        assert torch.equal(actual[key], value)


def test_template_token_box_refinement_is_exact_v18_noop_at_init():
    torch.manual_seed(43)
    model = SmallTargetExpert(search_size=256).eval()
    inputs = _inputs(batch_size=1)

    with torch.no_grad():
        retained = _retained_v18_output(model, inputs)
        actual = model(*inputs)

    for key, value in retained.items():
        assert torch.equal(actual[key], value)
    assert torch.equal(actual["small_base_pred_boxes"], retained["pred_boxes"])
    assert actual["small_box_delta"].shape == (1, 1, 4)
    assert torch.count_nonzero(actual["small_box_delta"]).item() == 0
    assert "small_detail_delta" not in actual
    assert "small_detail_size_delta" not in actual
    assert "small_detail_offset_delta" not in actual


def test_template_token_box_refinement_has_direct_final_box_gradients():
    torch.manual_seed(47)
    model = SmallTargetExpert(search_size=256)
    output = model(*_inputs(batch_size=1))
    target_box = torch.tensor([[[0.65, 0.35, 0.25, 0.15]]])

    loss = torch.nn.functional.l1_loss(output["pred_boxes"], target_box)
    loss.backward()

    weight_grad = model.box_refiner.output.weight.grad
    bias_grad = model.box_refiner.output.bias.grad
    assert weight_grad is not None
    assert bias_grad is not None
    assert torch.count_nonzero(weight_grad).item() > 0
    assert torch.count_nonzero(bias_grad).item() == 4


def test_template_token_box_refinement_changes_only_final_boxes():
    torch.manual_seed(53)
    model = SmallTargetExpert(search_size=256).eval()
    inputs = _inputs(batch_size=1)
    with torch.no_grad():
        retained = _retained_v18_output(model, inputs)
        model.box_refiner.output.bias.copy_(
            torch.tensor([1.0, -1.0, 0.5, -0.5]))
        actual = model(*inputs)

    assert torch.equal(actual["score_map"], retained["score_map"])
    assert torch.equal(actual["size_map"], retained["size_map"])
    assert torch.equal(actual["offset_map"], retained["offset_map"])
    assert not torch.equal(actual["pred_boxes"], retained["pred_boxes"])
    expected_delta = 0.125 * torch.tanh(
        torch.tensor([1.0, -1.0, 0.5, -0.5]))
    assert torch.allclose(actual["small_box_delta"][0, 0], expected_delta)


def test_independent_small_target_expert_is_lightweight_and_trainable():
    model = SmallTargetExpert(search_size=256)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    output = model(*_inputs(batch_size=1))
    loss = output["score_map"].mean() + output["pred_boxes"].mean()
    loss.backward()

    retained_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("box_refiner.")
    ]
    box_refiner_parameters = list(model.box_refiner.parameters())

    assert parameter_count < 75_000
    assert all(parameter.grad is not None for parameter in retained_parameters)
    assert all(torch.isfinite(parameter.grad).all()
               for parameter in retained_parameters)
    assert all(parameter.grad is not None
               for parameter in model.box_refiner.output.parameters())
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
               for parameter in box_refiner_parameters
               if parameter not in set(model.box_refiner.output.parameters()))


def test_modality_specific_s8_residuals_are_exact_encoder_noops_at_init():
    torch.manual_seed(29)
    model = SmallTargetExpert(search_size=16).eval()
    encoder = model.encoder
    rgb = torch.randn(2, 3, 16, 16)
    event = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        rgb_s4 = encoder.rgb_s4(encoder.rgb_stem(rgb))
        event_s4 = encoder.event_s4(encoder.event_stem(event))
        rgb_s8 = encoder.rgb_s8(rgb_s4)
        event_s8 = encoder.event_s8(event_s4)
        expected = (
            encoder.fuse_s4(rgb_s4, event_s4),
            encoder.fuse_s8(rgb_s8, event_s8),
        )
        actual = encoder(rgb, event)

    assert not hasattr(model, "spatial_identity_center")
    assert all(torch.equal(value, reference)
               for value, reference in zip(actual, expected))
    assert torch.count_nonzero(
        encoder.rgb_s8_residual.body[3].weight).item() == 0
    assert torch.count_nonzero(
        encoder.event_s8_residual.body[3].weight).item() == 0


def test_modality_specific_s8_residuals_are_lightweight_and_all_parameters_learn():
    torch.manual_seed(31)
    model = SmallTargetExpert(search_size=16)
    branches = (
        model.encoder.rgb_s8_residual,
        model.encoder.event_s8_residual,
    )
    parameters = [
        parameter for branch in branches for parameter in branch.parameters()
    ]
    inputs = (
        torch.randn(2, 96, 4, 4),
        torch.randn(2, 96, 4, 4),
    )
    targets = (
        torch.randn(2, 96, 4, 4),
        torch.randn(2, 96, 4, 4),
    )
    optimizer = torch.optim.SGD(parameters, lr=0.5)

    for _ in range(2):
        optimizer.zero_grad()
        loss = sum(
            (branch(value) - target).square().mean()
            for branch, value, target in zip(branches, inputs, targets)
        )
        loss.backward()
        optimizer.step()

    assert sum(parameter.numel() for parameter in parameters) == 20_544
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all()
               for parameter in parameters)
    assert all(torch.count_nonzero(parameter.grad).item() > 0
               for parameter in parameters)


def test_small_target_residual_refinement_starts_as_exact_noop():
    model = SmallTargetExpert(search_size=16)
    feature = torch.randn(2, 64, 4, 4)

    residual = model.head.refine(feature)

    assert torch.count_nonzero(residual).item() == 0


def test_box_decode_interpolates_between_adjacent_center_candidates():
    model = SmallTargetExpert(search_size=16)
    assert model._BOX_DECODE_TEMPERATURE == pytest.approx(0.20)
    score_map = torch.full((1, 1, 4, 4), 0.01)
    score_map[0, 0, 1, 1] = 0.60
    score_map[0, 0, 1, 2] = 0.59
    size_map = torch.full((1, 2, 4, 4), 0.25)
    offset_map = torch.zeros(1, 2, 4, 4)

    box = model._decode_boxes(score_map, size_map, offset_map)[0, 0]

    assert 0.25 < box[0] < 0.50
    assert box[1] == pytest.approx(0.25, abs=1e-4)


def test_box_decode_backpropagates_through_multiple_center_candidates():
    model = SmallTargetExpert(search_size=16)
    score_logits = torch.full((1, 1, 4, 4), -4.0, requires_grad=True)
    with torch.no_grad():
        score_logits[0, 0, 1, 1] = 0.40
        score_logits[0, 0, 1, 2] = 0.35
    score_map = score_logits.sigmoid()
    size_map = torch.full((1, 2, 4, 4), 0.25, requires_grad=True)
    offset_map = torch.zeros(1, 2, 4, 4, requires_grad=True)

    model._decode_boxes(score_map, size_map, offset_map).sum().backward()

    assert score_logits.grad is not None
    assert torch.count_nonzero(score_logits.grad).item() > 1


def test_independent_small_target_expert_rejects_wrong_search_size():
    model = SmallTargetExpert(search_size=256)
    template_rgb, template_event, search_rgb, search_event = _inputs(batch_size=1)

    with pytest.raises(ValueError, match="configured search_size"):
        model(
            template_rgb, template_event,
            search_rgb[..., :128, :128], search_event[..., :128, :128])


def test_independent_small_target_expert_rejects_multiple_search_frames():
    model = SmallTargetExpert(search_size=256)
    template_rgb, template_event, search_rgb, search_event = _inputs(batch_size=1)

    with pytest.raises(ValueError, match="exactly one search frame"):
        model(
            template_rgb, template_event,
            search_rgb.expand(-1, 2, -1, -1, -1),
            search_event.expand(-1, 2, -1, -1, -1))
