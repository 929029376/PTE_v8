import torch

import lib.train.data.felt_challenges as felt_challenges
from lib.train.data.felt_challenges import classify_felt_frames


def test_compound_challenges_are_not_collapsed_to_one_expert():
    result = felt_challenges.classify_felt_challenges(
        torch.tensor([
            [10.0, 10.0, 20.0, 20.0],
            [60.0, 60.0, 2.0, 2.0],
        ]),
        torch.tensor([1, 1]),
        image_size=(100, 100),
        event_motion=torch.tensor([0.0, 1.0]),
        ambiguity=torch.tensor([0.0, 1.5]),
        low_light=torch.tensor([False, True]),
        recovery_window=1,
    )

    assert "class_id" not in result
    attributes = result["challenge_attributes"]
    assert attributes["small_target"].tolist() == [False, True]
    assert attributes["motion"].tolist() == [False, True]
    assert attributes["low_light"].tolist() == [False, True]
    assert attributes["ambiguity"].tolist() == [False, True]
    expert_mask = felt_challenges.expert_supervision_mask(attributes)
    assert expert_mask[1].tolist() == [False, True, True, False, True]


def test_frame_classifier_preserves_recovery_window_without_class_assignment():
    boxes = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],
        [0.0, 0.0, 0.0, 0.0],
        [30.0, 30.0, 20.0, 20.0],
        [30.0, 30.0, 20.0, 20.0],
        [30.0, 30.0, 20.0, 20.0],
    ])
    result = classify_felt_frames(
        boxes, torch.tensor([1, 0, 1, 1, 1]),
        image_size=(100, 100), recovery_window=2)

    assert "class_id" not in result
    assert result["recovery"].tolist() == [False, False, True, True, False]


def test_invalid_present_box_is_not_a_valid_visible_challenge():
    result = classify_felt_frames(
        torch.tensor([[0.0, 0.0, 0.0, 10.0]]),
        torch.tensor([1]),
        image_size=(100, 100),
    )

    assert not result["visible"].item()
    assert not result["small_target"].item()


def test_all_independent_challenge_attributes_are_preserved():
    result = felt_challenges.classify_felt_challenges(
        torch.tensor([
            [10.0, 10.0, 20.0, 20.0],
            [60.0, 60.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [20.0, 20.0, 2.0, 2.0],
        ]),
        torch.tensor([1, 1, 0, 1]),
        image_size=(100, 100),
        event_motion=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        ambiguity=torch.tensor([0.0, 1.5, 0.0, 1.5]),
        low_light=torch.tensor([False, True, False, True]),
        recovery_window=1,
    )

    attributes = result["challenge_attributes"]
    assert attributes["small_target"].tolist() == [False, True, False, True]
    assert attributes["motion"].tolist() == [False, True, False, False]
    assert attributes["low_light"].tolist() == [False, True, False, True]
    assert attributes["recovery"].tolist() == [False, False, False, True]
    assert attributes["ambiguity"].tolist() == [False, True, False, True]
    assert attributes["deformation"].tolist() == [False, True, False, False]
    assert attributes["absent"].tolist() == [False, False, True, False]


def test_low_light_small_target_supervises_both_specialists():
    result = felt_challenges.classify_felt_challenges(
        torch.tensor([[10.0, 10.0, 2.0, 2.0]]),
        torch.tensor([1]),
        image_size=(100, 100),
        event_motion=torch.tensor([0.0]),
        ambiguity=torch.tensor([0.0]),
        low_light=torch.tensor([True]),
    )

    mask = felt_challenges.expert_supervision_mask(
        result["challenge_attributes"])
    assert mask[0].tolist() == [False, False, True, False, True]


def test_expert_supervision_mask_preserves_label_device():
    attributes = {
        name: torch.zeros(2, dtype=torch.bool, device="meta")
        for name in felt_challenges.CHALLENGE_NAMES
    }

    mask = felt_challenges.expert_supervision_mask(attributes)

    assert mask.device.type == "meta"


def test_extreme_aspect_box_is_not_labeled_small_target():
    result = felt_challenges.classify_felt_challenges(
        torch.tensor([[10.0, 10.0, 1.0, 20.0]]),
        torch.tensor([1]),
        image_size=(100, 100),
        event_motion=torch.tensor([0.0]),
        ambiguity=torch.tensor([0.0]),
        max_small_aspect_ratio=4.0,
    )

    assert not result["challenge_attributes"]["small_target"].item()
    assert felt_challenges.expert_supervision_mask(
        result["challenge_attributes"])[0].tolist() == [True] + [False] * 4
