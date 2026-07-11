import warnings

import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\..* is deprecated.*",
        category=FutureWarning,
    )
    from lib.models.layers.srbt_evidence import EVIDENCE_NAMES, EvidenceBank
    from lib.models.pet_track.pet_backbone import PETBackBone


def _inputs(batch=2, candidates=3, channels=16, feat_size=4, template_tokens=8):
    generator = torch.Generator().manual_seed(20260711)
    per_modality = feat_size * feat_size
    total = template_tokens + 2 * per_modality
    shared = torch.randn(batch, total, channels, generator=generator, requires_grad=True)
    detail = torch.randn(batch, total, channels, generator=generator, requires_grad=True)
    identity = torch.randn(batch, total, channels, generator=generator, requires_grad=True)
    boxes = torch.tensor(
        [[[0.25, 0.25, 0.30, 0.30],
          [0.70, 0.35, 0.25, 0.40],
          [0.50, 0.75, 0.45, 0.20]]],
        dtype=shared.dtype,
    ).expand(batch, candidates, 4).clone()
    quality = torch.randn(batch, 16, generator=generator)
    prior = torch.randn(batch, 32, generator=generator)
    return shared, detail, identity, boxes, quality, prior


def _bank(channels=16, feat_size=4):
    return EvidenceBank(
        embed_dim=channels,
        num_heads=4,
        search_tokens_per_modality=feat_size * feat_size,
        gate_epsilon=1.0,
    )


def test_evidence_bank_returns_five_candidate_likelihoods_and_reliabilities():
    inputs = _inputs()
    output = _bank()(*inputs)

    assert tuple(output["evidence"]) == EVIDENCE_NAMES
    for evidence in output["evidence"].values():
        assert evidence.shape == (2, 3, 16)
        assert torch.isfinite(evidence).all()
    assert output["likelihood_logits"].shape == (2, 3, 5)
    assert output["gates"].shape == (2, 3, 5)
    assert output["weights"].shape == (2, 3, 5)
    assert output["combined_likelihood"].shape == (2, 3)
    for value in output.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()
    assert torch.all(output["weights"].sum(dim=-1) <= 1.0)


def test_all_low_reliability_preserves_a_low_total_weight_state():
    bank = _bank()
    with torch.no_grad():
        for head in bank.reliability_heads.values():
            head[-1].weight.zero_()
            head[-1].bias.fill_(-30.0)

    output = bank(*_inputs())

    assert output["gates"].max() < 1e-10
    assert output["weights"].sum(dim=-1).max() < 1e-9


def test_gradients_reach_each_evidence_adapter_and_reliability_head():
    bank = _bank()
    output = bank(*_inputs())

    (output["combined_likelihood"].sum() + output["likelihood_logits"].sum()).backward()

    for name, adapter in bank.adapters.items():
        gradients = [parameter.grad for parameter in adapter.parameters()]
        assert any(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
            for gradient in gradients
        ), name
    for name, head in bank.reliability_heads.items():
        gradients = [parameter.grad for parameter in head.parameters()]
        assert any(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
            for gradient in gradients
        ), name


def test_zeroed_modalities_suppress_their_direct_evidence():
    bank = _bank()
    shared, detail, identity, boxes, quality, prior = _inputs(batch=1)
    search_start = shared.shape[1] - 32

    no_event = shared.detach().clone()
    no_event[:, search_start + 16:] = 0
    event_output = bank(no_event, detail, identity, boxes, quality, prior)
    assert torch.count_nonzero(event_output["evidence"]["motion"]) == 0

    no_rgb = shared.detach().clone()
    no_rgb[:, search_start:search_start + 16] = 0
    rgb_output = bank(no_rgb, detail, identity, boxes, quality, prior)
    assert torch.count_nonzero(rgb_output["evidence"]["appearance"]) == 0


def test_backbone_taps_are_opt_in_and_keep_the_default_output_contract():
    backbone = PETBackBone(
        patch_size=16,
        embed_dim=16,
        depth=4,
        num_heads=4,
        drop_path_rate=0.0,
    ).eval()
    backbone.amah_layers = []
    backbone.amah_sp_layers = []
    tokens = [torch.randn(1, 2, 16) for _ in range(6)]

    with torch.inference_mode():
        default_tokens, default_aux = backbone(*tokens)
        tapped_tokens, tapped_aux = backbone(*tokens, return_srbt_taps=True)

    assert torch.equal(default_tokens, tapped_tokens)
    assert tuple(default_aux) == ("attn",)
    assert tapped_aux["detail_tokens"].shape == tapped_tokens.shape
    assert tapped_aux["identity_tokens"].shape == tapped_tokens.shape
