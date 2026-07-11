"""Differentiable semi-Markov survival-reappearance belief update."""
import math

import torch
import torch.nn.functional as F
from torch import nn


VISIBLE = 0
UNCERTAIN = 1
ABSENT = 2
REAPPEARING = 3
STATE_NAMES = ("visible", "uncertain", "absent", "reappearing")
DURATION_BINS = (
    "1", "2", "3", "4", "5-8", "9-16", "17-32", "33-64", "65-128", ">128")


def _normalized_entropy(probability, classes):
    terms = torch.where(
        probability > 0,
        probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log(),
        torch.zeros_like(probability),
    )
    return (-terms.sum(dim=-1) / math.log(classes)).clamp(0.0, 1.0)


def _hypothesis_entropy(weights):
    if weights.shape[-1] == 0:
        return weights.new_zeros(weights.shape[0])
    active = (weights > 1e-8).sum(dim=-1)
    terms = torch.where(
        weights > 0,
        weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log(),
        torch.zeros_like(weights),
    )
    raw = -terms.sum(dim=-1)
    denominator = active.clamp_min(2).to(weights.dtype).log()
    entropy = raw / denominator
    return torch.where(active > 1, entropy, torch.zeros_like(entropy)).clamp(0.0, 1.0)


class SemiMarkovBelief(nn.Module):
    """Update joint state-duration and time-to-reappearance distributions."""

    def __init__(self, state_dim=32, quality_dim=16, hidden_dim=128,
                 max_hazard=128, reappearing_max_frames=3):
        super().__init__()
        self.state_dim = int(state_dim)
        self.quality_dim = int(quality_dim)
        self.max_hazard = int(max_hazard)
        self.reappearing_max_frames = int(reappearing_max_frames)
        if self.state_dim <= 0 or self.quality_dim <= 0:
            raise ValueError("state_dim and quality_dim must be positive")
        if self.max_hazard <= 0:
            raise ValueError("max_hazard must be positive")
        if not 1 <= self.reappearing_max_frames <= 4:
            raise ValueError("reappearing_max_frames must be in [1, 4]")

        allowed = torch.tensor([
            [True, True, True, False],
            [True, True, True, True],
            [False, True, True, True],
            [True, True, True, False],
        ])
        transition_mask = allowed[:, None, :].expand(-1, len(DURATION_BINS), -1).clone()
        # A reappearance segment may continue for W_re frames, but it is not
        # an unrestricted state self-loop in the public transition graph.
        transition_mask[
            REAPPEARING, :self.reappearing_max_frames - 1, REAPPEARING] = True
        transition_mask[
            REAPPEARING, self.reappearing_max_frames - 1:, REAPPEARING] = False
        self.register_buffer("allowed_transitions", allowed, persistent=False)
        self.register_buffer("transition_mask", transition_mask, persistent=False)
        self.register_buffer(
            "next_duration_bin",
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 9]),
            persistent=False,
        )
        self.register_buffer(
            "duration_widths",
            torch.tensor([1.0, 1.0, 1.0, 1.0, 4.0, 8.0, 16.0, 32.0, 64.0, 1.0]),
            persistent=False,
        )
        self.register_buffer(
            "duration_values",
            torch.tensor([1.0, 2.0, 3.0, 4.0, 6.5, 12.5, 24.5, 48.5, 96.5, 129.0]),
            persistent=False,
        )
        assignments = torch.zeros(
            len(STATE_NAMES) * len(DURATION_BINS) * len(STATE_NAMES),
            len(STATE_NAMES) * len(DURATION_BINS))
        flat_index = 0
        for source_state in range(len(STATE_NAMES)):
            for source_duration in range(len(DURATION_BINS)):
                for target_state in range(len(STATE_NAMES)):
                    if transition_mask[source_state, source_duration, target_state]:
                        if source_state != target_state:
                            assignments[
                                flat_index,
                                target_state * len(DURATION_BINS)] = 1.0
                        elif source_duration < 4:
                            target_duration = int(self.next_duration_bin[source_duration])
                            assignments[
                                flat_index,
                                target_state * len(DURATION_BINS) + target_duration] = 1.0
                        elif source_duration < len(DURATION_BINS) - 1:
                            progress = 1.0 / float(self.duration_widths[source_duration])
                            assignments[
                                flat_index,
                                target_state * len(DURATION_BINS) + source_duration] = 1.0 - progress
                            assignments[
                                flat_index,
                                target_state * len(DURATION_BINS)
                                + int(self.next_duration_bin[source_duration])] = progress
                        else:
                            assignments[
                                flat_index,
                                target_state * len(DURATION_BINS) + source_duration] = 1.0
                    flat_index += 1
        self.register_buffer("transition_assignments", assignments, persistent=False)

        self.state_duration_embedding = nn.Parameter(
            torch.empty(len(STATE_NAMES), len(DURATION_BINS), self.state_dim))
        nn.init.trunc_normal_(self.state_duration_embedding, std=0.02)
        context_dim = self.state_dim + self.quality_dim + 4
        self.transition_base = nn.Parameter(
            torch.zeros(len(STATE_NAMES), len(DURATION_BINS), len(STATE_NAMES)))
        self.transition_context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(STATE_NAMES) * len(STATE_NAMES)),
        )
        self.hazard_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.max_hazard),
        )

    def _embedding(self, state_duration):
        embedding = self.state_duration_embedding.to(
            device=state_duration.device, dtype=state_duration.dtype)
        return torch.einsum("bsd,sde->be", state_duration, embedding)

    @staticmethod
    def _normalize_candidates(candidate_weights):
        if candidate_weights.ndim != 2:
            raise ValueError("candidate_weights must have shape (B, K)")
        if not torch.isfinite(candidate_weights).all():
            raise ValueError("candidate_weights must be finite")
        nonnegative = candidate_weights.clamp_min(0)
        total = nonnegative.sum(dim=-1, keepdim=True)
        return torch.where(
            total > 1e-8,
            nonnegative / total.clamp_min(1e-8),
            torch.zeros_like(nonnegative),
        )

    @staticmethod
    def _candidate_summary(candidate_weights, normalized):
        if candidate_weights.shape[-1] == 0:
            return candidate_weights.new_zeros(candidate_weights.shape[0], 4)
        maximum = candidate_weights.max(dim=-1).values
        mean = candidate_weights.mean(dim=-1)
        total = candidate_weights.sum(dim=-1)
        entropy = _hypothesis_entropy(normalized)
        return torch.stack((maximum, mean, total, entropy), dim=-1)

    def _transition_prior(self, previous, context):
        context_logits = self.transition_context(context).reshape(
            previous.shape[0], len(STATE_NAMES), len(STATE_NAMES))
        scores = self.transition_base[None] + context_logits[:, :, None, :]
        scores = scores.masked_fill(~self.transition_mask[None], -torch.inf)
        log_transition = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
        log_previous = torch.where(
            previous > 0,
            previous.clamp_min(torch.finfo(previous.dtype).tiny).log(),
            torch.full_like(previous, -torch.inf),
        )
        contributions = (
            log_previous[..., None] + log_transition).flatten(1)
        assignments = self.transition_assignments.to(dtype=previous.dtype)
        log_assignments = torch.where(
            assignments > 0,
            assignments.clamp_min(torch.finfo(previous.dtype).tiny).log(),
            torch.full_like(assignments, -torch.inf),
        )
        grouped = contributions[:, :, None] + log_assignments[None]
        return torch.logsumexp(grouped, dim=1).reshape(
            previous.shape[0], len(STATE_NAMES), len(DURATION_BINS))

    def _hazard_distribution(self, context):
        logits = self.hazard_head(context)
        conditional_hazard = torch.sigmoid(logits)
        log_survival_after = torch.cumsum(F.logsigmoid(-logits), dim=-1)
        survival_after = log_survival_after.exp()
        survival_before = torch.cat(
            (torch.ones_like(survival_after[:, :1]), survival_after[:, :-1]), dim=-1)
        event_probability = conditional_hazard * survival_before
        hazard = torch.cat((event_probability, survival_after[:, -1:]), dim=-1)
        survival = torch.cat(
            (torch.ones_like(survival_after[:, :1]), survival_after), dim=-1)
        return hazard, survival

    def initialize(self, batch_size, device, dtype):
        state_duration = torch.zeros(
            int(batch_size), len(STATE_NAMES), len(DURATION_BINS),
            device=device, dtype=dtype)
        state_duration[:, VISIBLE, 0] = 1.0
        state_prob = state_duration.sum(dim=-1)
        hazard = torch.zeros(
            int(batch_size), self.max_hazard + 1, device=device, dtype=dtype)
        hazard[:, -1] = 1.0
        survival = torch.ones_like(hazard)
        zeros = torch.zeros(int(batch_size), device=device, dtype=dtype)
        return {
            "state_duration": state_duration,
            "state_prob": state_prob,
            "hazard": hazard,
            "survival": survival,
            "hypothesis_weights": torch.empty(
                int(batch_size), 0, device=device, dtype=dtype),
            "entropy": {
                "state": zeros.clone(),
                "hypothesis": zeros.clone(),
                "control": zeros.clone(),
            },
            "duration": torch.ones(int(batch_size), device=device, dtype=dtype),
            "belief_embedding": self._embedding(state_duration),
        }

    def forward(self, previous, observation_logits, candidate_weights, quality_stats):
        state_duration = previous["state_duration"]
        if state_duration.ndim != 3 or state_duration.shape[1:] != (4, 10):
            raise ValueError("previous state_duration must have shape (B, 4, 10)")
        batch = state_duration.shape[0]
        if observation_logits.shape != (batch, 4):
            raise ValueError("observation_logits must have shape (B, 4)")
        if quality_stats.shape != (batch, self.quality_dim):
            raise ValueError(
                f"quality_stats must have shape (B, {self.quality_dim})")
        if candidate_weights.shape[0] != batch:
            raise ValueError("candidate_weights batch size differs from belief")

        candidate_weights = candidate_weights.to(
            device=state_duration.device, dtype=state_duration.dtype)
        observation_logits = observation_logits.to(
            device=state_duration.device, dtype=state_duration.dtype)
        quality_stats = quality_stats.to(
            device=state_duration.device, dtype=state_duration.dtype)
        normalized_candidates = self._normalize_candidates(candidate_weights)
        summary = self._candidate_summary(candidate_weights, normalized_candidates)
        current_embedding = self._embedding(state_duration)
        context = torch.cat((current_embedding, quality_stats, summary), dim=-1)

        log_prior = self._transition_prior(state_duration, context)
        log_posterior = log_prior + observation_logits[:, :, None]
        normalizer = torch.logsumexp(log_posterior.flatten(1), dim=-1)
        posterior = (log_posterior - normalizer[:, None, None]).exp()
        state_prob = posterior.sum(dim=-1)
        belief_embedding = self._embedding(posterior)
        hazard, survival = self._hazard_distribution(context)

        state_entropy = _normalized_entropy(state_prob, len(STATE_NAMES))
        hypothesis_entropy = _hypothesis_entropy(normalized_candidates)
        control_entropy = 0.5 * state_entropy + 0.5 * hypothesis_entropy
        duration_values = self.duration_values.to(
            device=posterior.device, dtype=posterior.dtype)
        duration = (posterior * duration_values[None, None, :]).sum(dim=(1, 2))
        return {
            "state_duration": posterior,
            "state_prob": state_prob,
            "hazard": hazard,
            "survival": survival,
            "hypothesis_weights": normalized_candidates,
            "entropy": {
                "state": state_entropy,
                "hypothesis": hypothesis_entropy,
                "control": control_entropy,
            },
            "duration": duration,
            "belief_embedding": belief_embedding,
        }
