import pytest
import torch

from lib.train.data.srbt_labels import (
    ABSENT,
    REAPPEARING,
    UNCERTAIN,
    VISIBLE,
    build_temporal_targets,
)


def _value(targets, key):
    return targets[key].item()


def test_stable_visible_has_no_hazard_target():
    targets = build_temporal_targets([1, 1, 1, 1, 1], anchor=3, horizon=3)

    assert _value(targets, "state_target") == VISIBLE
    assert _value(targets, "duration") == 4
    assert _value(targets, "hazard_mask") is False
    assert _value(targets, "hazard_target") == 0
    assert _value(targets, "censor_mask") is False
    assert torch.equal(targets["future_present"], torch.tensor([1, 0, 0]))
    assert torch.equal(targets["future_valid"], torch.tensor([True, False, False]))


def test_present_to_absent_targets_the_next_reappearance():
    present = [1, 1, 1, 1, 0, 0, 1, 1]
    targets = build_temporal_targets(present, anchor=3, horizon=4)

    assert _value(targets, "state_target") == VISIBLE
    assert _value(targets, "duration") == 4
    assert _value(targets, "hazard_mask") is True
    assert _value(targets, "hazard_target") == 3
    assert _value(targets, "censor_mask") is False
    assert torch.equal(targets["future_present"], torch.tensor([0, 0, 1, 1]))
    assert targets["future_valid"].all()


def test_absent_run_tracks_duration_and_immediate_reappearance():
    present = [1, 1, 1, 1, 0, 0, 1, 1]
    targets = build_temporal_targets(present, anchor=5, horizon=4)

    assert _value(targets, "state_target") == ABSENT
    assert _value(targets, "duration") == 2
    assert _value(targets, "hazard_mask") is True
    assert _value(targets, "hazard_target") == 1
    assert _value(targets, "censor_mask") is False
    assert torch.equal(targets["future_present"], torch.tensor([1, 1, 0, 0]))
    assert torch.equal(targets["future_valid"], torch.tensor([True, True, False, False]))


def test_reappearing_is_limited_to_three_continuous_present_frames():
    present = [0, 1, 1, 1, 1]

    states = [
        _value(build_temporal_targets(present, anchor=anchor, horizon=1), "state_target")
        for anchor in range(1, 5)
    ]
    durations = [
        _value(build_temporal_targets(present, anchor=anchor, horizon=1), "duration")
        for anchor in range(1, 5)
    ]

    assert states == [REAPPEARING, REAPPEARING, REAPPEARING, VISIBLE]
    assert durations == [1, 2, 3, 4]


def test_short_sequence_start_is_uncertain_until_three_present_frames():
    present = [1, 1, 1]

    states = [
        _value(build_temporal_targets(present, anchor=anchor, horizon=1), "state_target")
        for anchor in range(3)
    ]

    assert states == [UNCERTAIN, UNCERTAIN, VISIBLE]


def test_sequence_end_is_right_censored_at_last_valid_future_frame():
    targets = build_temporal_targets([1, 1, 1, 0, 0], anchor=3, horizon=4)

    assert _value(targets, "state_target") == ABSENT
    assert _value(targets, "hazard_mask") is True
    assert _value(targets, "hazard_target") == 1
    assert _value(targets, "censor_mask") is True
    assert torch.equal(targets["future_valid"], torch.tensor([True, False, False, False]))


def test_absent_anchor_without_a_future_frame_has_no_hazard_supervision():
    targets = build_temporal_targets([1, 0], anchor=1, horizon=8)

    assert _value(targets, "state_target") == ABSENT
    assert _value(targets, "hazard_target") == 0
    assert _value(targets, "hazard_mask") is False
    assert _value(targets, "censor_mask") is False
    assert not targets["future_valid"].any()


def test_reappearance_after_max_hazard_uses_overflow_bin():
    present = [0] * 130 + [1]
    targets = build_temporal_targets(
        present,
        anchor=0,
        horizon=128,
        max_hazard=128,
    )

    assert _value(targets, "hazard_mask") is True
    assert _value(targets, "hazard_target") == 129
    assert _value(targets, "censor_mask") is False
    assert targets["future_valid"].all()
    assert not targets["future_present"].any()


@pytest.mark.parametrize(
    "present,anchor,horizon,error",
    [
        ([1, 2, 0], 1, 2, "binary"),
        ([[1, 0]], 0, 1, "one-dimensional"),
        ([], 0, 1, "non-empty"),
        ([1, 0], 2, 1, "anchor"),
        ([1, 0], 0, 0, "horizon"),
    ],
)
def test_invalid_temporal_inputs_raise(present, anchor, horizon, error):
    with pytest.raises(ValueError, match=error):
        build_temporal_targets(present, anchor=anchor, horizon=horizon)
