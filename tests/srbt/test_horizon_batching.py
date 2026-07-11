import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from lib.config.pet_track.config import cfg as default_cfg
import lib.train.base_functions as base_functions
from lib.train.data.loader import (
    HorizonBatchSampler,
    LTRLoader,
    ltr_collate_stack1,
)
from lib.train.data.processing import STARKProcessing
from lib.train.data.sampler import TrackingSampler
from lib.train.data import transforms as tfm
import lib.train.trainers.ltr_trainer as ltr_trainer
from lib.utils import TensorDict


def test_horizon_batch_sampler_keeps_one_horizon_and_every_index_once():
    random.seed(20260711)
    sampler = HorizonBatchSampler(
        indices=range(10),
        batch_size=3,
        horizons=[8, 32, 128],
        weights=[0.4, 0.35, 0.25],
        drop_last=False,
    )

    batches = list(sampler)
    flat = [item for batch in batches for item in batch]

    assert len(sampler) == 4
    assert sorted(index for index, _ in flat) == list(range(10))
    assert all(len({horizon for _, horizon in batch}) == 1 for batch in batches)
    assert all(horizon in {8, 32, 128} for _, horizon in flat)
    assert [len(batch) for batch in batches] == [3, 3, 3, 1]


def test_horizon_weights_and_drop_last_are_respected():
    sampler = HorizonBatchSampler(
        indices=range(10),
        batch_size=3,
        horizons=[8, 32, 128],
        weights=[0, 1, 0],
        drop_last=True,
    )

    batches = list(sampler)

    assert len(sampler) == 3
    assert len(batches) == 3
    assert all(horizon == 32 for batch in batches for _, horizon in batch)
    assert [index for batch in batches for index, _ in batch] == list(range(9))


def test_horizon_batch_sampler_propagates_epoch_to_its_index_sampler():
    class EpochAwareIndices:
        def __init__(self):
            self.epochs = []

        def __iter__(self):
            return iter(range(4))

        def __len__(self):
            return 4

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    indices = EpochAwareIndices()
    sampler = HorizonBatchSampler(
        indices=indices,
        batch_size=2,
        horizons=[8],
        weights=[1],
        drop_last=False,
    )

    sampler.set_epoch(7)

    assert indices.epochs == [7]


def test_tracking_sampler_accepts_index_horizon_pairs_without_changing_ints():
    sampler = TrackingSampler.__new__(TrackingSampler)
    sampler.train_cls = False
    sampler.getitem = lambda horizon=None: horizon

    assert sampler[7] is None
    assert sampler[(7, 32)] == 32


def test_future_tensors_are_padded_to_the_declared_horizon():
    first = TensorDict({
        "future_images": torch.ones(2, 3, 4, 4),
        "future_event_images": torch.full((2, 3, 4, 4), 2.0),
        "future_anno": torch.ones(2, 4),
        "future_valid": torch.tensor([True, True, False, False]),
        "future_present": torch.tensor([1, 0, 0, 0]),
        "search_images": torch.ones(1, 3, 4, 4),
    })
    second = TensorDict({
        "future_images": torch.full((4, 3, 4, 4), 3.0),
        "future_event_images": torch.full((4, 3, 4, 4), 4.0),
        "future_anno": torch.full((4, 4), 5.0),
        "future_valid": torch.ones(4, dtype=torch.bool),
        "future_present": torch.ones(4, dtype=torch.long),
        "search_images": torch.ones(1, 3, 4, 4),
    })

    batch = ltr_collate_stack1([first, second])

    assert batch["future_images"].shape == (4, 2, 3, 4, 4)
    assert batch["future_event_images"].shape == (4, 2, 3, 4, 4)
    assert batch["future_anno"].shape == (4, 2, 4)
    assert not batch["future_images"][2:, 0].any()
    assert not batch["future_event_images"][2:, 0].any()
    assert not batch["future_anno"][2:, 0].any()
    assert batch["future_valid"].shape == (4, 2)
    assert batch["search_images"].shape == (1, 2, 3, 4, 4)


class _FakeFeltDataset:
    def __init__(self):
        present = torch.tensor([1, 1, 1, 1, 0, 0, 1, 1], dtype=torch.uint8)
        bbox = torch.tensor([[2, 2, 4, 4]] * len(present), dtype=torch.float32)
        self.info = {
            "bbox": bbox,
            "valid": torch.ones(len(present), dtype=torch.bool),
            "visible": present.clone(),
            "absent": present,
        }

    def __len__(self):
        return 1

    def get_num_sequences(self):
        return 1

    def is_video_sequence(self):
        return True

    def get_name(self):
        return "FELT"

    def get_frames(self, seq_id, frame_ids, anno):
        rgb = [np.full((8, 8, 3), frame_id, dtype=np.uint8) for frame_id in frame_ids]
        event = [np.full((8, 8, 3), 100 + frame_id, dtype=np.uint8) for frame_id in frame_ids]
        frame_anno = {
            key: [value[frame_id].clone() for frame_id in frame_ids]
            for key, value in anno.items()
        }
        return rgb, event, frame_anno, {"object_class_name": None}


def _sampler_config():
    return SimpleNamespace(
        TRAIN=SimpleNamespace(STAGE="all"),
        DATA=SimpleNamespace(
            C3_EVENT_SAMPLING=True,
            C3_EVENT_SAMPLE_PROB=1.0,
            C3_EVENT_WEIGHTS=None,
            MOTION_CAUSAL_SAMPLING=False,
            SRBT=SimpleNamespace(
                ENABLE=True,
                HORIZONS=[8, 32, 128],
                HORIZON_WEIGHTS=[0.4, 0.35, 0.25],
                MAX_HAZARD=128,
            ),
        ),
    )


def test_tracking_sampler_loads_a_continuous_future_clip_and_targets():
    dataset = _FakeFeltDataset()

    def identity_processing(data):
        data["valid"] = True
        return data

    sampler = TrackingSampler(
        datasets=[dataset],
        p_datasets=[1],
        samples_per_epoch=1,
        max_gap=4,
        num_search_frames=1,
        num_template_frames=1,
        processing=identity_processing,
        frame_sample_mode="causal",
        cfg=_sampler_config(),
        training=True,
    )
    sampler.sample_seq_from_dataset = lambda dataset, is_video: (
        0,
        dataset.info["visible"],
        dataset.info,
    )
    sampler._sample_c3_event_causal_frame_ids = (
        lambda visible, info, preferred_events=None: (
            [2],
            [3],
            "visible_to_absent",
        )
    )

    sample = sampler[(0, 4)]

    assert torch.equal(sample["future_frame_ids"], torch.tensor([4, 5, 6, 7]))
    assert [int(image[0, 0, 0]) for image in sample["future_images"]] == [4, 5, 6, 7]
    assert [int(image[0, 0, 0]) for image in sample["future_event_images"]] == [104, 105, 106, 107]
    assert len(sample["future_anno"]) == 4
    assert torch.equal(sample["future_present"], torch.tensor([0, 0, 1, 1]))
    assert sample["future_valid"].all()
    assert sample["hazard_target"].item() == 3
    assert sample["hazard_mask"].item() is True


def test_processing_uses_shared_full_frame_geometry_for_future_rgb_event():
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    box = torch.tensor([12, 12, 8, 8], dtype=torch.float32)
    mask = torch.zeros(32, 32)
    data = TensorDict({
        "template_images": [image.copy()],
        "template_event_images": [image.copy()],
        "template_anno": [box.clone()],
        "template_masks": [mask.clone()],
        "search_images": [image.copy()],
        "search_event_images": [image.copy()],
        "search_anno": [box.clone()],
        "search_masks": [mask.clone()],
        "future_images": [image.copy(), image.copy()],
        "future_event_images": [image.copy(), image.copy()],
        "future_anno": [box.clone(), torch.zeros(4)],
    })
    processor = STARKProcessing(
        search_area_factor={"template": 2.0, "search": 4.0},
        output_sz={"template": 16, "search": 16},
        center_jitter_factor={"template": 0.0, "search": 0.0},
        scale_jitter_factor={"template": 0.0, "search": 0.0},
        mode="sequence",
        transform=tfm.Transform(tfm.ToTensor()),
        joint_transform=None,
        settings=SimpleNamespace(redetect_search_area_factor=4.0),
    )

    processed = processor(data)

    assert processed["valid"] is True
    assert processed["future_images"].shape == (2, 3, 16, 16)
    assert processed["future_event_images"].shape == (2, 3, 16, 16)
    assert processed["future_anno"].shape == (2, 4)
    assert torch.equal(processed["future_images"], processed["future_event_images"])
    assert processed["search_images"].shape == (1, 3, 16, 16)


def test_default_and_canonical_srbt_horizon_configuration():
    assert default_cfg.DATA.SRBT.ENABLE is False
    assert default_cfg.DATA.SRBT.HORIZONS == [8, 32, 128]
    assert default_cfg.DATA.SRBT.HORIZON_WEIGHTS == [0.4, 0.35, 0.25]
    assert default_cfg.DATA.SRBT.MAX_HAZARD == 128

    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments" / "pet_track" / "felt_pet_track.yaml"
    )
    configured = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert configured["DATA"]["SRBT"] == {
        "ENABLE": True,
        "HORIZONS": [8, 32, 128],
        "HORIZON_WEIGHTS": [0.4, 0.35, 0.25],
        "MAX_HAZARD": 128,
        "ANCHOR_WEIGHTS": {
            "VISIBLE": 0.25,
            "PRESENT_TO_ABSENT": 0.25,
            "ABSENT": 0.25,
            "REAPPEARING": 0.25,
        },
    }


def test_srbt_training_loader_uses_horizon_batches_only_for_training():
    config = deepcopy(default_cfg)
    config.DATA.SRBT.ENABLE = True
    config.TRAIN.BATCH_SIZE = 3
    dataset = list(range(10))
    settings = SimpleNamespace(local_rank=-1)

    batch_sampler = base_functions._build_srbt_batch_sampler(
        dataset, config, settings, training=True)

    assert isinstance(batch_sampler, HorizonBatchSampler)
    assert len(batch_sampler) == 3
    assert all(
        len({horizon for _, horizon in batch}) == 1
        for batch in batch_sampler
    )
    assert base_functions._build_srbt_batch_sampler(
        dataset, config, settings, training=False) is None


def test_ltr_loader_accepts_horizon_batch_sampler_without_auto_batching():
    class EchoDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            sample_index, horizon = index
            return TensorDict({
                "sample_index": torch.tensor([sample_index]),
                "horizon": torch.tensor([horizon]),
            })

    batch_sampler = HorizonBatchSampler(
        indices=range(4),
        batch_size=2,
        horizons=[32],
        weights=[1],
        drop_last=True,
    )
    loader = LTRLoader(
        "train",
        EchoDataset(),
        training=True,
        batch_size=1,
        batch_sampler=batch_sampler,
        num_workers=0,
        stack_dim=1,
    )

    batch = next(iter(loader))

    assert batch["sample_index"].shape == (1, 2)
    assert batch["horizon"].tolist() == [[32, 32]]


def test_trainer_propagates_epoch_through_batch_sampler():
    class EpochAwareBatchSampler:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    batch_sampler = EpochAwareBatchSampler()
    loader = SimpleNamespace(batch_sampler=batch_sampler, sampler=None)

    ltr_trainer._set_loader_epoch(loader, 11)

    assert batch_sampler.epochs == [11]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"horizons": [8, 0]}, "horizons"),
        ({"horizons": [8, 32], "weights": [1]}, "same length"),
        ({"weights": [0, 0, 0]}, "positive"),
    ],
)
def test_horizon_batch_sampler_rejects_invalid_configuration(kwargs, error):
    options = {
        "indices": range(4),
        "batch_size": 2,
        "horizons": [8, 32, 128],
        "weights": [0.4, 0.35, 0.25],
        "drop_last": False,
    }
    options.update(kwargs)

    with pytest.raises(ValueError, match=error):
        HorizonBatchSampler(**options)
