import torch


VISIBLE = 0
UNCERTAIN = 1
ABSENT = 2
REAPPEARING = 3


def build_temporal_targets(present, anchor, horizon, max_hazard=128):
    """Build causal frame-state and future survival targets for one anchor."""
    present = torch.as_tensor(present)
    if present.ndim != 1:
        raise ValueError("present must be one-dimensional")
    if present.numel() == 0:
        raise ValueError("present must be non-empty")
    if not torch.all((present == 0) | (present == 1)):
        raise ValueError("present labels must be binary")
    if not 0 <= int(anchor) < present.numel():
        raise ValueError("anchor is outside the presence sequence")
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    if int(max_hazard) <= 0:
        raise ValueError("max_hazard must be positive")

    anchor = int(anchor)
    horizon = int(horizon)
    max_hazard = int(max_hazard)
    present = present.to(dtype=torch.bool)

    run_start = anchor
    while run_start > 0 and present[run_start - 1] == present[anchor]:
        run_start -= 1
    duration = anchor - run_start + 1

    if not present[anchor]:
        state = ABSENT
    elif run_start > 0 and not present[run_start - 1] and duration <= 3:
        state = REAPPEARING
    elif duration >= 3:
        state = VISIBLE
    else:
        state = UNCERTAIN

    valid_length = min(horizon, present.numel() - anchor - 1)
    future_present = torch.zeros(horizon, dtype=torch.long)
    future_valid = torch.zeros(horizon, dtype=torch.bool)
    if valid_length:
        future_present[:valid_length] = present[
            anchor + 1:anchor + 1 + valid_length
        ].to(dtype=torch.long)
        future_valid[:valid_length] = True

    enters_absence = (
        bool(present[anchor])
        and anchor + 1 < present.numel()
        and not bool(present[anchor + 1])
    )
    hazard_mask = state in (UNCERTAIN, ABSENT) or enters_absence
    hazard_target = 0
    censor_mask = False
    if hazard_mask:
        reappearance = next(
            (
                frame_id
                for frame_id in range(anchor + 1, present.numel())
                if bool(present[frame_id]) and not bool(present[frame_id - 1])
            ),
            None,
        )
        if reappearance is not None:
            delay = reappearance - anchor
            if delay > max_hazard:
                hazard_target = max_hazard + 1
            elif delay <= horizon:
                hazard_target = delay
            else:
                hazard_target = min(valid_length, max_hazard)
                censor_mask = True
        else:
            hazard_target = min(valid_length, max_hazard)
            censor_mask = True

    return {
        "future_present": future_present,
        "future_valid": future_valid,
        "hazard_target": torch.tensor(hazard_target, dtype=torch.long),
        "hazard_mask": torch.tensor(hazard_mask, dtype=torch.bool),
        "censor_mask": torch.tensor(censor_mask, dtype=torch.bool),
        "duration": torch.tensor(duration, dtype=torch.long),
        "state_target": torch.tensor(state, dtype=torch.long),
    }
