import torch

import lib.train.data.felt_challenges as felt_challenges
from lib.train.data.felt_challenges import (
    DISCRIMINATION,
    GENERALIST,
    MOTION,
    SMALL_TARGET,
    VISIBILITY,
    classify_felt_frames,
)


def test_felt_challenge_teacher_assigns_each_expert_from_frame_level_signals():
    boxes = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],  # generalist
        [30.0, 10.0, 20.0, 20.0],  # motion
        [31.0, 11.0, 5.0, 5.0],    # small target wins overlap
        [0.0, 0.0, 0.0, 0.0],      # absent
        [40.0, 40.0, 5.0, 5.0],    # recovery wins overlap
        [30.0, 30.0, 20.0, 20.0],
        [28.0, 30.0, 24.0, 20.0],  # discrimination
        [30.0, 30.0, 20.0, 20.0],
        [30.0, 30.0, 20.0, 20.0],
        [30.0, 30.0, 20.0, 20.0],
        [30.0, 30.0, 20.0, 20.0],
        [30.0, 30.0, 20.0, 20.0],
        [30.0, 30.0, 20.0, 20.0],  # recovery window has ended
    ])
    present = torch.tensor([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    result = classify_felt_frames(boxes, present, image_size=(100, 100))

    assert result["class_id"].tolist() == [
        GENERALIST,
        MOTION,
        SMALL_TARGET,
        VISIBILITY,
        VISIBILITY,
        VISIBILITY,
        VISIBILITY,
        VISIBILITY,
        VISIBILITY,
        VISIBILITY,
        VISIBILITY,
        VISIBILITY,
        GENERALIST,
    ]
    assert result["recovery"][4:12].all()
    assert not result["recovery"][12]


def test_felt_challenge_teacher_ignores_invalid_present_boxes():
    result = classify_felt_frames(
        torch.tensor([[0.0, 0.0, 0.0, 10.0]]),
        torch.tensor([1]),
        image_size=(100, 100),
    )

    assert result["class_id"].item() == -1


def test_expert_owner_assignment_uses_declared_exclusive_priority():
    boxes = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],
        [10.0, 10.0, 20.0, 20.0],
        [60.0, 10.0, 20.0, 20.0],
        [60.0, 10.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    presence = torch.tensor([1, 1, 1, 1, 0])
    ambiguity = torch.tensor([0.0, 0.9, 0.9, 0.9, 0.0])
    event_motion = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])

    assign = getattr(felt_challenges, "assign_expert_owners")
    result = assign(
        boxes,
        presence,
        image_size=(100, 100),
        event_motion=event_motion,
        ambiguity=ambiguity,
        recovery_window=1,
    )

    assert result["class_id"].tolist() == [
        GENERALIST,
        DISCRIMINATION,
        MOTION,
        SMALL_TARGET,
        VISIBILITY,
    ]
