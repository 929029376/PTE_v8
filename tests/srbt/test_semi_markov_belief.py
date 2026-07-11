import warnings

import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\..* is deprecated.*",
        category=FutureWarning,
    )
    from lib.models.layers.srbt_belief import (
        ABSENT,
        REAPPEARING,
        UNCERTAIN,
        VISIBLE,
        SemiMarkovBelief,
    )


def _model(max_hazard=8):
    return SemiMarkovBelief(
        state_dim=32,
        quality_dim=16,
        hidden_dim=64,
        max_hazard=max_hazard,
        reappearing_max_frames=3,
    )


def _step(model, previous, target=None, candidate_weights=None):
    batch = previous["state_duration"].shape[0]
    logits = torch.zeros(batch, 4)
    if target is not None:
        logits.fill_(-80.0)
        logits[:, target] = 80.0
    if candidate_weights is None:
        candidate_weights = torch.tensor([[0.7, 0.2, 0.1]]).expand(batch, -1)
    return model(previous, logits, candidate_weights, torch.zeros(batch, 16))


def test_initialize_biases_the_joint_posterior_to_visible_duration_one():
    posterior = _model().initialize(2, torch.device("cpu"), torch.float32)

    assert posterior["state_duration"].shape == (2, 4, 10)
    assert torch.allclose(
        posterior["state_duration"].sum(dim=(1, 2)), torch.ones(2))
    assert torch.all(posterior["state_prob"][:, VISIBLE] > 0.99)
    assert torch.equal(posterior["duration"], torch.ones(2))
    assert posterior["hazard"].shape == (2, 9)
    assert torch.allclose(posterior["hazard"].sum(dim=-1), torch.ones(2))
    assert posterior["hypothesis_weights"].shape == (2, 0)
    assert posterior["belief_embedding"].shape == (2, 32)


def test_disallowed_visible_to_reappearing_transition_stays_zero():
    model = _model()
    previous = model.initialize(1, torch.device("cpu"), torch.float32)

    posterior = _step(model, previous, target=REAPPEARING)

    assert posterior["state_prob"][0, REAPPEARING] == 0
    assert model.allowed_transitions[VISIBLE].tolist() == [True, True, True, False]
    assert model.allowed_transitions[ABSENT].tolist() == [False, True, True, True]
    assert model.allowed_transitions[UNCERTAIN].tolist() == [True, True, True, True]
    assert model.allowed_transitions[REAPPEARING].tolist() == [True, True, True, False]


def test_hazard_distribution_and_survival_are_normalized_and_censor_safe():
    model = _model(max_hazard=128)
    posterior = _step(
        model,
        model.initialize(2, torch.device("cpu"), torch.float32),
    )

    assert posterior["hazard"].shape == (2, 129)
    assert posterior["survival"].shape == (2, 129)
    assert torch.allclose(
        posterior["hazard"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.equal(posterior["survival"][:, 0], torch.ones(2))
    assert torch.all(posterior["survival"][:, 1:] <= posterior["survival"][:, :-1])
    assert torch.all((posterior["survival"] >= 0) & (posterior["survival"] <= 1))
    assert torch.allclose(
        posterior["hazard"][:, -1], posterior["survival"][:, -1], atol=1e-6)


def test_reappearing_cannot_self_loop_beyond_three_frames():
    model = _model()
    with torch.no_grad():
        model.transition_base.zero_()
        for parameter in model.transition_context.parameters():
            parameter.zero_()
    posterior = model.initialize(1, torch.device("cpu"), torch.float32)
    posterior = _step(model, posterior, target=ABSENT)

    reappearing_probabilities = []
    for _ in range(4):
        posterior = _step(model, posterior, target=REAPPEARING)
        reappearing_probabilities.append(
            posterior["state_prob"][0, REAPPEARING].item())

    assert min(reappearing_probabilities[:3]) > 0.99
    assert reappearing_probabilities[3] < 1e-6


def test_long_absent_run_progresses_beyond_the_five_to_eight_bin():
    model = _model()
    with torch.no_grad():
        model.transition_base.zero_()
        for parameter in model.transition_context.parameters():
            parameter.zero_()
    posterior = model.initialize(1, torch.device("cpu"), torch.float32)

    for _ in range(20):
        posterior = _step(model, posterior, target=ABSENT)

    assert posterior["state_duration"][0, ABSENT, 5:].sum() > 0
    assert posterior["duration"][0] > 8


def test_entropy_is_normalized_and_hypothesis_weights_are_candidate_probabilities():
    model = _model()
    previous = model.initialize(2, torch.device("cpu"), torch.float32)
    candidate_weights = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

    posterior = _step(model, previous, candidate_weights=candidate_weights)

    assert torch.allclose(
        posterior["hypothesis_weights"].sum(dim=-1), torch.ones(2))
    assert posterior["entropy"]["hypothesis"][0] == 0
    assert torch.allclose(
        posterior["entropy"]["hypothesis"][1], torch.tensor(1.0), atol=1e-6)
    for value in posterior["entropy"].values():
        assert torch.all((value >= 0) & (value <= 1))


def test_backward_reaches_transition_and_hazard_parameters():
    model = _model()
    previous = model.initialize(2, torch.device("cpu"), torch.float32)
    observation = torch.randn(2, 4, requires_grad=True)
    candidates = torch.rand(2, 5, requires_grad=True)
    quality = torch.randn(2, 16, requires_grad=True)

    posterior = model(previous, observation, candidates, quality)
    loss = (
        posterior["state_duration"].square().sum()
        + posterior["hazard"].square().sum()
        + posterior["belief_embedding"].square().sum()
        + posterior["entropy"]["control"].sum()
    )
    loss.backward()

    assert observation.grad is not None and torch.count_nonzero(observation.grad) > 0
    assert quality.grad is not None and torch.count_nonzero(quality.grad) > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.transition_context.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.hazard_head.parameters()
    )
