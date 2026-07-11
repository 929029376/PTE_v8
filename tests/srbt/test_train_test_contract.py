from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from easydict import EasyDict as edict

from lib.train.actors.pet_track import PETTrackActor
from lib.train.data.sampler import TrackingSampler
from lib.train.trainers import BaseTrainer, base_trainer
from lib.test.evaluation.tracker import Tracker
from lib.test.tracker.pet_track import _load_srbt_eval_checkpoint
from tests.srbt.test_srbt_model_integration import _cfg as _tiny_cfg
from tests.srbt.test_srbt_model_integration import _model


def _state():
    return {
        "schema_version": 1,
        "net": {"student.weight": torch.ones(1), "srbt_teacher.weight": torch.ones(1)},
        "optimizer": {"state": {}, "param_groups": []},
        "lr_scheduler": {"last_epoch": 3},
        "amp_scaler": {"scale": 1.0},
        "epoch": 4,
        "best_val_score": 0.5,
        "best_val_epoch": 4,
        "config_summary": {"MODEL.SRBT.ENABLE": True},
    }


def test_srbt_checkpoint_schema_version_one_is_strict():
    validate_srbt_checkpoint_schema = getattr(
        base_trainer, "validate_srbt_checkpoint_schema", None)
    assert validate_srbt_checkpoint_schema is not None
    validate_srbt_checkpoint_schema(_state())
    for key in ("schema_version", "net", "optimizer", "lr_scheduler",
                "amp_scaler", "epoch", "best_val_score", "config_summary"):
        broken = _state()
        broken.pop(key)
        with pytest.raises(RuntimeError, match=key):
            validate_srbt_checkpoint_schema(broken)

    broken = _state()
    broken["schema_version"] = 0
    with pytest.raises(RuntimeError, match="schema_version=1"):
        validate_srbt_checkpoint_schema(broken)


def test_legacy_pet_route_checkpoint_is_not_a_valid_srbt_resume():
    validate_srbt_checkpoint_schema = getattr(
        base_trainer, "validate_srbt_checkpoint_schema", None)
    assert validate_srbt_checkpoint_schema is not None
    with pytest.raises(RuntimeError, match="schema_version=1"):
        validate_srbt_checkpoint_schema({
            "net": {"expert_router.weight": torch.ones(1)},
            "optimizer": {},
            "epoch": 1,
        })


def test_srbt_checkpoint_round_trip_saves_scaler_state_dict(tmp_path):
    class TinyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(1, 1)

    class FakeScaler:
        def __init__(self, scale=128.0):
            self.scale = scale

        def state_dict(self):
            return {"scale": self.scale}

        def load_state_dict(self, state):
            self.scale = state["scale"]

    settings = SimpleNamespace(
        env=SimpleNamespace(workspace_dir=str(tmp_path)),
        save_dir=None,
        local_rank=0,
        project_path="srbt",
        use_gpu=False,
        cfg=_srbt_cfg(),
    )
    actor = TinyActor()
    optimizer = torch.optim.SGD(actor.net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    trainer = BaseTrainer(actor, [], optimizer, settings, scheduler)
    trainer.epoch = 3
    trainer.amp_scaler = FakeScaler()
    trainer.best_val_score = 0.75
    trainer.best_val_epoch = 3

    trainer.save_checkpoint("latest")

    state = torch.load(
        tmp_path / "checkpoints" / "srbt" / "Linear_latest.pth.tar",
        map_location="cpu",
    )
    assert state["schema_version"] == 1
    assert state["amp_scaler"] == {"scale": 128.0}

    restored_actor = TinyActor()
    restored_optimizer = torch.optim.SGD(
        restored_actor.net.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, 1)
    restored = BaseTrainer(
        restored_actor, [], restored_optimizer, settings, restored_scheduler)
    restored.amp_scaler = FakeScaler(scale=1.0)
    restored.load_checkpoint(
        str(tmp_path / "checkpoints" / "srbt" / "Linear_latest.pth.tar"))

    assert restored.epoch == 3
    assert restored.best_val_score == 0.75
    assert restored.best_val_epoch == 3
    assert restored.amp_scaler.scale == 128.0
    assert restored.config_summary == state["config_summary"]
    for expected, actual in zip(
            actor.net.parameters(), restored_actor.net.parameters()):
        assert torch.equal(expected, actual)


def test_eval_checkpoint_loader_rejects_partial_or_legacy_state(tmp_path):
    model = _model()
    partial = tmp_path / "partial.pth.tar"
    torch.save({"net": model.state_dict()}, partial)
    with pytest.raises(RuntimeError, match="schema_version=1"):
        _load_srbt_eval_checkpoint(model, partial)

    strict = tmp_path / "strict.pth.tar"
    state = _state()
    state["net"] = model.state_dict()
    torch.save(state, strict)
    _load_srbt_eval_checkpoint(model, strict)


def test_felt_result_contract_keeps_absent_and_drops_c3_debug():
    output = Tracker._default_result_container()
    assert output == {"target_bbox": [], "time": [], "absent": []}


class _FakeSrbtVideo:
    def __init__(self):
        self.present = torch.ones(24, dtype=torch.uint8)
        self.present[8:11] = 0
        self.visible = self.present.bool()
        self.valid = torch.ones(24, dtype=torch.bool)
        self.bboxes = torch.stack([
            torch.tensor([2.0 + idx * 0.25, 3.0, 5.0, 4.0])
            for idx in range(24)
        ])
        self.images = [
            np.full((16, 16, 3), idx, dtype=np.uint8)
            for idx in range(24)
        ]
        self.events = [
            np.full((16, 16, 3), 255 - idx, dtype=np.uint8)
            for idx in range(24)
        ]

    def get_name(self):
        return "fake_srbt"

    def is_video_sequence(self):
        return True

    def get_num_sequences(self):
        return 1

    def get_sequence_info(self, _seq_id):
        return {
            "visible": self.visible.clone(),
            "valid": self.valid.clone(),
            "absent": self.present.clone(),
            "bbox": self.bboxes.clone(),
        }

    def get_frames(self, _seq_id, frame_ids, _seq_info_dict):
        frames = [self.images[int(frame_id)].copy() for frame_id in frame_ids]
        events = [self.events[int(frame_id)].copy() for frame_id in frame_ids]
        boxes = [self.bboxes[int(frame_id)].clone() for frame_id in frame_ids]
        masks = [torch.zeros((16, 16), dtype=torch.float32)
                 for _ in frame_ids]
        anno = {
            "bbox": boxes,
            "mask": masks,
            "absent": torch.tensor(
                [int(self.present[int(frame_id)]) for frame_id in frame_ids],
                dtype=torch.uint8,
            ),
        }
        return frames, events, anno, {"object_class_name": "target"}


def _valid_processing(data):
    data["valid"] = True
    return data


def _stack_images(images):
    return torch.stack([
        torch.as_tensor(image).permute(2, 0, 1).float() / 255.0
        for image in images
    ]).unsqueeze(1)


def _stack_boxes(boxes):
    return torch.stack([box.float() for box in boxes]).unsqueeze(1)


def _srbt_cfg():
    cfg = _tiny_cfg()
    cfg.MODEL.SRBT.TEACHER.HAZARD_BINS = 129
    cfg.MODEL.BACKBONE = edict({
        "CE_LOC": [],
        "CE_KEEP_RATIO": [1.0],
        "STRIDE": 8,
    })
    cfg.DATA = edict({
        "SEARCH": edict({"SIZE": 16}),
        "SRBT": edict({
            "ENABLE": True,
            "HISTORY_LENGTH": 4,
            "MAX_HAZARD": 8,
            "ANCHOR_WEIGHTS": edict({
                "VISIBLE": 1.0,
                "PRESENT_TO_ABSENT": 1.0,
                "ABSENT": 1.0,
                "REAPPEARING": 1.0,
            }),
        }),
    })
    return cfg


def test_tracking_sampler_srbt_batch_is_real_actor_forward_without_legacy_route_or_c3_fields():
    cfg = _srbt_cfg()
    sampler = TrackingSampler(
        datasets=[_FakeSrbtVideo()],
        p_datasets=[1],
        samples_per_epoch=1,
        max_gap=8,
        num_search_frames=1,
        num_template_frames=2,
        processing=_valid_processing,
        frame_sample_mode="causal",
        cfg=cfg,
        training=True,
    )
    sample = sampler[(0, 4)]

    forbidden = [
        key for key in sample
        if "route" in key.lower()
        or "c3" in key.lower()
        or key.startswith("redetect_")
    ]
    assert forbidden == []
    for key in ("future_images", "future_event_images", "future_valid"):
        assert key in sample
    assert torch.as_tensor(sample["future_valid"]).shape == (4,)
    current_frame_value = int(sample["search_images"][0][0, 0, 0])
    valid_history_ids = sample["history_frame_ids"][sample["history_frame_ids"] >= 0]
    assert bool((valid_history_ids < current_frame_value).all())

    batch = {
        "template_images": _stack_images(sample["template_images"]),
        "template_event_images": _stack_images(
            sample["template_event_images"]),
        "search_images": _stack_images(sample["search_images"]),
        "search_event_images": _stack_images(sample["search_event_images"]),
        "template_anno": _stack_boxes(sample["template_anno"]),
        "search_anno": _stack_boxes(sample["search_anno"]),
        "future_images": _stack_images(sample["future_images"]),
        "future_event_images": _stack_images(sample["future_event_images"]),
        "future_valid": torch.as_tensor(sample["future_valid"]).unsqueeze(1),
        "history_images": _stack_images(sample["history_images"]),
        "history_event_images": _stack_images(sample["history_event_images"]),
        "history_valid": torch.as_tensor(sample["history_valid"]).unsqueeze(1),
        "epoch": torch.tensor([0]),
    }
    model = _model()
    seen = {"calls": []}
    original_forward = model.forward

    def capture_forward(*args, **kwargs):
        seen["calls"].append(kwargs)
        seen["future_images"] = kwargs.get("future_images")
        seen["future_event_images"] = kwargs.get("future_event_images")
        seen["future_valid"] = kwargs.get("future_valid")
        return original_forward(*args, **kwargs)

    model.forward = capture_forward
    actor = PETTrackActor(
        model,
        objective={},
        loss_weight={},
        settings=SimpleNamespace(num_template=2, batchsize=1),
        cfg=cfg,
    )
    out = actor.forward_pass(batch)
    assert set(out["srbt_predictions"]) == {
        "existence_logits", "hazard_logits", "field_logits",
        "candidate_logits", "hypothesis_boxes", "hypothesis_scores",
        "identity_embeddings", "template_identity",
    }
    assert seen["future_images"].shape == (1, 4, 3, 16, 16)
    assert seen["future_event_images"].shape == (1, 4, 3, 16, 16)
    assert seen["future_valid"].shape == (1, 4)
    assert len(seen["calls"]) == int(sample["history_valid"].sum()) + 1
    assert seen["calls"][-1]["previous_posterior"] is not None
    history_values = [
        int(call["xi"][0, 0, 0, 0, 0].mul(255).round().item())
        for call in seen["calls"][:-1]
    ]
    assert all(value < current_frame_value for value in history_values)
    initial = model.initialize_srbt_posterior(1, torch.device("cpu"), torch.float32)
    assert not torch.equal(
        seen["calls"][-1]["previous_posterior"]["state_duration"],
        initial["state_duration"],
    )


def test_history_posterior_mask_handles_mixed_padding_and_new_hypotheses():
    previous = {
        "state": torch.tensor([[1.0], [2.0]]),
        "hypothesis_weights": torch.empty(2, 0),
        "entropy": {"control": torch.tensor([0.1, 0.2])},
    }
    updated = {
        "state": torch.tensor([[3.0], [4.0]]),
        "hypothesis_weights": torch.tensor([[0.7, 0.3], [0.6, 0.4]]),
        "entropy": {"control": torch.tensor([0.3, 0.4])},
    }

    selected = PETTrackActor._select_posterior(
        previous, updated, torch.tensor([True, False]))

    assert torch.equal(selected["state"], torch.tensor([[3.0], [2.0]]))
    assert torch.equal(
        selected["hypothesis_weights"],
        torch.tensor([[0.7, 0.3], [0.0, 0.0]]),
    )
    assert torch.equal(
        selected["entropy"]["control"], torch.tensor([0.3, 0.2]))


def test_batch_first_temporal_tensors_are_not_transposed_when_batch_equals_horizon():
    images = torch.arange(4 * 4 * 3 * 2 * 2).reshape(4, 4, 3, 2, 2)
    valid = torch.tensor([
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ])

    assert torch.equal(
        PETTrackActor._batch_first_future(images, 4, images.device), images)
    assert torch.equal(
        PETTrackActor._batch_first_future_mask(valid, 4, valid.device), valid)


def test_server_scripts_default_to_existing_srbt_config():
    root = Path(__file__).resolve().parents[2]
    expected = 'CONFIG_NAME="${CONFIG_NAME:-felt_pet_track}"'
    assert (root / "experiments/pet_track/felt_pet_track.yaml").is_file()
    for relative_path in (
            "scripts/server_train_pet_track.sh",
            "scripts/watch_training.sh"):
        assert expected in (root / relative_path).read_text(encoding="utf-8")


def test_supervisor_exposes_only_the_srbt_training_entry_point():
    root = Path(__file__).resolve().parents[2]
    assert not (root / "tracking/supervisord_stage1_v8.conf").exists()
    config = (root / "tracking/supervisord_srbt_v8.conf").read_text(
        encoding="utf-8")
    assert "--config felt_pet_track" in config
    assert "stage1" not in config.lower()
    assert "expert" not in config.lower()
    assert "generated/felt_pet_track_v8" not in config


def test_canonical_config_contains_no_legacy_route_or_c3_nodes():
    from copy import deepcopy
    from lib.config.pet_track.config import cfg, update_config_from_file

    configured = deepcopy(cfg)
    update_config_from_file("experiments/pet_track/felt_pet_track.yaml", configured)
    forbidden = {
        "EXPERT", "PET", "EVENT_BELIEF", "EPSM", "ABSENCE",
        "MEMORY_POLICY", "HETEROGENEOUS_TAIL", "EVENT_TRIGGER",
        "STATE_MACHINE",
    }
    assert forbidden.isdisjoint(configured.MODEL)
    assert configured.MODEL.PRETRAINED_BASELINE_CKPT.endswith(
        "AMTTrack_ep0098.pth.tar")
