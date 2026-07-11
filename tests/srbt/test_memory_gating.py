import torch

from lib.config.pet_track.config import cfg
from lib.models.layers.thor import THOR_Wrapper


class _Backbone:
    @staticmethod
    def _z_feat(image):
        value = image.float().mean()
        pattern = torch.tensor(
            [[[-2.0, -1.0, 0.5],
              [-1.0, 0.5, 2.0],
              [0.5, 2.0, -2.0],
              [2.0, -2.0, -1.0]]],
            device=image.device,
        )
        return pattern + value


class _Net:
    def __init__(self):
        self.backbone = _Backbone()
        self.memory = None


def _image(value):
    return torch.full((3, 2, 2), float(value))


def _wrapper(update_interval=2):
    wrapper = THOR_Wrapper(
        net=_Net(),
        st_capacity=5,
        lt_capacity=10,
        sample_interval=100,
        update_interval=update_interval,
        lower_bound=0.0,
        score_threshold=0.7,
    )
    wrapper.setup(_image(0), _image(0))
    return wrapper


def test_hold_and_redetect_observations_never_write_memory():
    wrapper = _wrapper()
    initial_recent = [item.clone() for item in wrapper.st_module.zi_raw_list]
    initial_long = [item.clone() for item in wrapper.lt_module.zi_raw_list]

    for frame in range(1, 7):
        wrapper.begin_frame()
        wrapper.commit(
            _image(frame), _image(frame), pred_score=1.0,
            allow_recent_write=False, allow_long_write=False,
        )

    assert wrapper.get_update_count()[:2] == (0, 0)
    assert all(torch.equal(before, after) for before, after in zip(
        initial_recent, wrapper.st_module.zi_raw_list))
    assert all(torch.equal(before, after) for before, after in zip(
        initial_long, wrapper.lt_module.zi_raw_list))


def test_memory_capacities_and_initial_long_term_template_are_immutable():
    wrapper = _wrapper()
    initial_rgb = wrapper.lt_module.zi_raw_list[0].clone()
    initial_event = wrapper.lt_module.ze_raw_list[0].clone()

    for frame in range(1, 17):
        wrapper.begin_frame()
        wrapper.commit(
            _image(frame), _image(-frame), pred_score=1.0,
            allow_recent_write=True, allow_long_write=True,
        )

    assert len(wrapper.st_module.zi_raw_list) == 5
    assert len(wrapper.lt_module.zi_raw_list) == 10
    assert torch.equal(wrapper.lt_module.zi_raw_list[0], initial_rgb)
    assert torch.equal(wrapper.lt_module.ze_raw_list[0], initial_event)


def test_recovered_observations_can_enter_recent_before_long_term_memory():
    wrapper = _wrapper(update_interval=4)

    for frame in range(1, 5):
        wrapper.begin_frame()
        wrapper.commit(
            _image(frame), _image(frame), pred_score=1.0,
            allow_recent_write=True, allow_long_write=False,
        )
    recent_count, long_count, _ = wrapper.get_update_count()
    assert recent_count > 0
    assert long_count == 0

    for frame in range(5, 9):
        wrapper.begin_frame()
        wrapper.commit(
            _image(frame), _image(frame), pred_score=1.0,
            allow_recent_write=True, allow_long_write=True,
        )
    recent_count, long_count, _ = wrapper.get_update_count()
    assert recent_count > 0
    assert long_count > 0


def test_refill_keeps_exact_capacity_instead_of_appending_duplicate_slots():
    wrapper = _wrapper()
    wrapper.lt_module.clear()
    wrapper.lt_module.clear()

    assert len(wrapper.lt_module.zi_raw_list) == 10
    assert len(wrapper.lt_module.ze_raw_list) == 10
    assert len(wrapper.lt_module.zi_list) == 10
    assert len(wrapper.lt_module.ze_list) == 10


def test_begin_frame_reads_memory_without_mutating_it_until_commit():
    wrapper = _wrapper(update_interval=2)
    initial_recent = [item.clone() for item in wrapper.st_module.zi_raw_list]

    dynamic_rgb, dynamic_event = wrapper.begin_frame()

    assert dynamic_rgb.shape == dynamic_event.shape
    assert wrapper.get_update_count() == (0, 0, 1)
    assert all(torch.equal(before, after) for before, after in zip(
        initial_recent, wrapper.st_module.zi_raw_list))

    wrapper.commit(
        _image(9), _image(9), pred_score=1.0,
        allow_recent_write=True, allow_long_write=False,
    )
    assert wrapper.get_update_count()[:2] == (1, 0)


def test_default_memory_capacities_match_the_srbt_contract():
    assert cfg.TEST.SHORTTERM_LIBRARY_NUMS == 5
    assert cfg.TEST.LONGTERM_LIBRARY_NUMS == 10
