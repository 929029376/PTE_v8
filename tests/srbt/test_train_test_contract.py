from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from easydict import EasyDict as edict

from lib.train.actors.pet_track import PETTrackActor
from lib.train.data.sampler import TrackingSampler
from lib.train.trainers import BaseTrainer, base_trainer
from lib.train.trainers.ltr_trainer import LTRTrainer
import lib.train.train_script as train_script_module
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.running import _save_tracker_output
from lib.test.tracker.pet_track import _load_srbt_eval_checkpoint
from lib.models.pet_track.pet_track import _load_retained_model_checkpoint
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


class _UnsafeCheckpointPayload:
    pass


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


def test_current_srbt_checkpoint_schema_accepts_local_expert_weights():
    state = _state()
    state["net"].update({
        "expert_heads.motion_fm.proj.weight": torch.ones(1),
        "expert_fusion.experts.generalist.rgb_gain": torch.ones(1),
    })

    assert base_trainer.validate_srbt_checkpoint_schema(state) is True


def test_unversioned_legacy_checkpoint_is_not_a_valid_srbt_resume():
    validate_srbt_checkpoint_schema = getattr(
        base_trainer, "validate_srbt_checkpoint_schema", None)
    assert validate_srbt_checkpoint_schema is not None
    with pytest.raises(RuntimeError, match="schema_version=1"):
        validate_srbt_checkpoint_schema({
            "net": {"legacy_tracker.weight": torch.ones(1)},
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
    optimizer = torch.optim.AdamW(actor.net.parameters(), lr=0.1)
    actor.net(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    trainer = BaseTrainer(actor, [], optimizer, settings, scheduler)
    trainer.epoch = 3
    trainer.amp_scaler = FakeScaler()
    trainer.best_val_score = 0.75
    trainer.best_val_epoch = 3
    trainer.specialist_gate_reference = {
        "generalist_iou": 0.70,
        "specialists": {"motion_fm": 0.62},
    }

    trainer.save_checkpoint("latest")

    state = torch.load(
        tmp_path / "checkpoints" / "srbt" / "Linear_latest.pth.tar",
        map_location="cpu",
        weights_only=True,
    )
    assert state["schema_version"] == 1
    assert state["amp_scaler"] == {"scale": 128.0}
    assert state["specialist_gate_reference"] == {
        "generalist_iou": 0.70,
        "specialists": {"motion_fm": 0.62},
    }
    assert {"stats", "constructor", "net_info"}.isdisjoint(state)

    restored_actor = TinyActor()
    restored_optimizer = torch.optim.AdamW(
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
    assert restored.specialist_gate_reference == state[
        "specialist_gate_reference"]
    assert restored.config_summary == state["config_summary"]
    for expected, actual in zip(
            actor.net.parameters(), restored_actor.net.parameters()):
        assert torch.equal(expected, actual)


def test_resume_resets_legacy_iou_best_when_metric_changes_to_felt_auc(tmp_path):
    class TinyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(1, 1)

    settings = SimpleNamespace(
        env=SimpleNamespace(workspace_dir=str(tmp_path)),
        save_dir=None,
        local_rank=0,
        project_path="srbt",
        use_gpu=False,
        best_metric="FELT_SR_PROXY",
    )
    actor = TinyActor()
    optimizer = torch.optim.SGD(actor.net.parameters(), lr=0.1)
    trainer = BaseTrainer(actor, [], optimizer, settings)
    legacy_checkpoint = tmp_path / "legacy_iou_best.pth.tar"
    state = _state()
    state.update({
        "actor_type": "TinyActor",
        "net_type": "Linear",
        "net": actor.net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": None,
        "amp_scaler": None,
        "epoch": 12,
        "best_val_score": 0.81,
        "best_val_epoch": 10,
        # Legacy checkpoints did not name the metric; they used IoU.
        "config_summary": {"MODEL.SRBT.ENABLE": True},
    })
    torch.save(state, legacy_checkpoint)

    trainer.load_checkpoint(str(legacy_checkpoint))

    assert trainer.epoch == 12
    assert trainer.best_val_score is None
    assert trainer.best_val_epoch == 0


def test_checkpoint_summary_records_best_metric_without_cfg():
    settings = SimpleNamespace(best_metric="FELT_SR_PROXY")

    assert (
        base_trainer._config_summary(settings)["TRAIN.BEST_METRIC"]
        == "FELT_SR_PROXY"
    )


def test_refine_startup_loads_stage1_gate_reference(tmp_path):
    loader = getattr(
        train_script_module, "_load_refine_gate_reference", None)
    assert loader is not None
    checkpoint = tmp_path / "best_stage1.pth.tar"
    state = _state()
    state["specialist_gate_reference"] = {
        "generalist_iou": 0.70,
        "specialists": {
            "motion_fm": 0.62,
            "small_target_st": 0.61,
            "visibility_foc_ov": 0.60,
            "discrimination_bi": 0.63,
        },
    }
    torch.save(state, checkpoint)
    cfg = SimpleNamespace(MODEL=SimpleNamespace(
        INIT_CHECKPOINT=str(checkpoint)))

    assert loader(cfg) == state["specialist_gate_reference"]

    state.pop("specialist_gate_reference")
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="Stage 1 specialist gate reference"):
        loader(cfg)


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


def test_retained_initialization_accepts_unversioned_operator_checkpoint(tmp_path):
    model = _model()
    state = model.state_dict()
    state.pop("_pet_architecture_version")
    checkpoint = tmp_path / "operator_init.pth.tar"
    torch.save({"net": state}, checkpoint)

    report = _load_retained_model_checkpoint(
        model,
        checkpoint,
        label="operator initialization",
        trusted_legacy_pickle=True,
    )

    assert report["loaded_count"] > 0
    assert int(model.state_dict()["_pet_architecture_version"]) == model.ARCHITECTURE_VERSION


def test_eval_checkpoint_loader_rejects_custom_objects_before_schema_use(tmp_path):
    model = _model()
    unsafe = tmp_path / "unsafe.pth.tar"
    state = _state()
    state["payload"] = _UnsafeCheckpointPayload()
    torch.save(state, unsafe)

    with pytest.raises(RuntimeError, match="weights-only"):
        _load_srbt_eval_checkpoint(model, unsafe)


def test_training_resume_rejects_custom_objects_before_schema_use(tmp_path):
    class TinyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(1, 1)

    settings = SimpleNamespace(
        env=SimpleNamespace(workspace_dir=str(tmp_path)),
        save_dir=None,
        local_rank=0,
        project_path="srbt",
        use_gpu=False,
    )
    actor = TinyActor()
    trainer = BaseTrainer(
        actor, [], torch.optim.SGD(actor.net.parameters(), lr=0.1), settings)
    unsafe = tmp_path / "unsafe_resume.pth.tar"
    state = _state()
    state.update({
        "net_type": "Linear",
        "net": actor.net.state_dict(),
        "payload": _UnsafeCheckpointPayload(),
    })
    torch.save(state, unsafe)

    with pytest.raises(RuntimeError, match="weights-only"):
        trainer.load_checkpoint(str(unsafe))


def test_generic_result_contract_keeps_only_official_outputs():
    output = Tracker._default_result_container()
    assert output == {"target_bbox": [], "time": []}


def test_felt_result_writer_preserves_official_sequence_txt_protocol(tmp_path):
    sequence = SimpleNamespace(name="felt_sequence", dataset="FELT")
    tracker = SimpleNamespace(results_dir=str(tmp_path))
    output = {
        "target_bbox": [[1.2, 2.8, 3.0, 4.9], [5.0, 6.0, 7.0, 8.0]],
        "time": [0.1, 0.2],
        "absent": [False, True],
    }

    _save_tracker_output(sequence, tracker, output)

    assert np.loadtxt(tmp_path / "felt_sequence.txt", delimiter="\t").tolist() == [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]
    assert not (tmp_path / "felt_sequence_absent.txt").exists()
    assert np.loadtxt(
        tmp_path / "felt_sequence_time.txt", delimiter="\t").tolist() == [0.1, 0.2]


def test_felt_result_writer_ignores_diagnostic_absent_length(tmp_path):
    sequence = SimpleNamespace(name="felt_sequence", dataset="FELT")
    tracker = SimpleNamespace(results_dir=str(tmp_path))
    output = {
        "target_bbox": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        "time": [0.1, 0.2],
        "absent": [False],
    }

    _save_tracker_output(sequence, tracker, output)

    assert (tmp_path / "felt_sequence.txt").exists()
    assert not (tmp_path / "felt_sequence_absent.txt").exists()


class _CausalEvalSequence:
    name = "causal_eval"
    dataset = "FELT"
    object_ids = None

    def __init__(self, initial_box):
        self.aps_frame_list = ["aps0", "aps1", "aps2"]
        self.dvs_frame_list = ["dvs0", "dvs1", "dvs2"]
        self.ground_truth_rect = np.array([
            initial_box,
            [100.0, 100.0, 20.0, 20.0],
            [200.0, 200.0, 30.0, 30.0],
        ])

    def init_info(self, _frame_num):
        return {"init_bbox": self.ground_truth_rect[0].tolist()}

    @staticmethod
    def frame_info(frame_num):
        return {"frame_id": frame_num}


class _CausalEvalTracker:
    supports_absent_output = True

    def __init__(self):
        self.initialize_calls = []
        self.track_infos = []

    def initialize(self, aps, dvs, info, idx=0):
        self.initialize_calls.append((aps, dvs, dict(info), idx))
        return None

    def track(self, _aps, _dvs, info):
        self.track_infos.append(dict(info))
        return {"target_bbox": [1, 2, 3, 4], "absent": False}

    @staticmethod
    def get_update_count():
        return 0, 0, 3

    @staticmethod
    def get_sample_count():
        return 0, 3


def _evaluation_runner():
    runner = object.__new__(Tracker)
    runner._read_image = lambda path: path
    return runner


def test_evaluation_runner_passes_only_first_frame_gt_and_no_frame_gt_info():
    sequence = _CausalEvalSequence([10.0, 10.0, 12.0, 8.0])
    tracker = _CausalEvalTracker()

    output = _evaluation_runner()._track_sequence(
        tracker, sequence, sequence.init_info(0))

    assert len(tracker.initialize_calls) == 1
    aps, dvs, info, idx = tracker.initialize_calls[0]
    assert (aps, dvs, idx) == ("aps0", "dvs0", 0)
    assert set(info) == {"init_bbox"}
    np.testing.assert_array_equal(
        info["init_bbox"], [10.0, 10.0, 12.0, 8.0])
    assert len(tracker.track_infos) == 2
    assert all("gt_bbox" not in info for info in tracker.track_infos)
    assert [info["frame_id"] for info in tracker.track_infos] == [1, 2]
    assert len(output["target_bbox"]) == 3
    assert output["absent"] == [False, False, False]


def test_evaluation_runner_rejects_invalid_initial_box_without_scanning_future_gt():
    sequence = _CausalEvalSequence([0.0, 0.0, 0.0, 0.0])
    tracker = _CausalEvalTracker()

    with pytest.raises(RuntimeError, match="initial bounding box"):
        _evaluation_runner()._track_sequence(
            tracker, sequence, sequence.init_info(0))

    assert tracker.initialize_calls == []


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
    cfg.MODEL.BACKBONE = edict({
        "CE_LOC": [],
        "CE_KEEP_RATIO": [1.0],
        "STRIDE": 8,
    })
    cfg.DATA = edict({
        "SEARCH": edict({"SIZE": 16}),
        "SRBT": edict({
            "ENABLE": True,
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
    sampler._sample_srbt_event_causal_frame_ids = (
        lambda visible, info: ([4, 5], [6], "visible_to_visible")
    )
    sample = sampler[0]

    forbidden = [
        key for key in sample
        if "route" in key.lower()
        or "c3" in key.lower()
    ]
    assert forbidden == []
    assert "challenge_id" not in sample
    assert "expert_owner_id" not in sample
    for key in (
            "future_images", "future_event_images", "future_valid",
            "history_images", "history_event_images", "history_valid",
            "history_frame_ids"):
        assert key not in sample
    for key in (
            "redetect_search_images", "redetect_search_event_images",
            "redetect_search_anno", "redetect_search_masks"):
        assert key in sample
    batch = {
        "template_images": _stack_images(sample["template_images"]),
        "template_event_images": _stack_images(
            sample["template_event_images"]),
        "search_images": _stack_images(sample["search_images"]),
        "search_event_images": _stack_images(sample["search_event_images"]),
        "template_anno": _stack_boxes(sample["template_anno"]),
        "search_anno": _stack_boxes(sample["search_anno"]),
        "redetect_search_images": _stack_images(
            sample["redetect_search_images"]),
        "redetect_search_event_images": _stack_images(
            sample["redetect_search_event_images"]),
        "redetect_search_anno": _stack_boxes(
            sample["redetect_search_anno"]),
        "is_reappear": torch.as_tensor(sample["is_reappear"]).unsqueeze(1),
        "challenge_id": torch.tensor([0]),
        "epoch": torch.tensor([0]),
    }
    model = _model()
    seen = {"calls": []}
    original_forward = model.forward

    def capture_forward(*args, **kwargs):
        seen["calls"].append(kwargs)
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
    assert set(out["presence_predictions"]) == {"logits", "score"}
    assert len(seen["calls"]) == 1
    assert not any(
        key.startswith(("future_", "history_"))
        for key in seen["calls"][0]
    )


def test_recovery_sampling_uses_presence_transitions_not_challenge_labels():
    cfg = _srbt_cfg()
    cfg.TRAIN.EXPERT_PHASE = "recovery"
    cfg.DATA.CHALLENGE_SAMPLING = edict({
        "ENABLE": True,
        "PRECISE": True,
        "MANIFEST": "unused-during-recovery.json",
    })
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

    assert sampler.srbt_enabled is True
    assert sampler.precise_expert_sampling is False
    sampled = []

    def sample_reappearance(visible, info):
        sampled.append((visible, info))
        return [4, 5], [11], "absent_to_present"

    sampler._sample_srbt_event_causal_frame_ids = sample_reappearance
    sample = sampler[0]

    assert len(sampled) == 1
    assert "challenge_id" not in sample
    assert sample["sampler_event_type"] == "absent_to_present"
    assert int(torch.as_tensor(sample["is_reappear"])[-1]) == 1


def test_actor_ignores_obsolete_future_and_history_inputs():
    class CaptureNet:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {}

    actor = object.__new__(PETTrackActor)
    actor.expert_enabled = False
    actor.cfg = SimpleNamespace(
        MODEL=SimpleNamespace(BACKBONE=SimpleNamespace(CE_LOC=False)))
    actor.settings = SimpleNamespace(num_template=1)
    actor.net = CaptureNet()
    images = torch.zeros(1, 2, 3, 4, 4)
    data = {
        "template_images": images,
        "template_event_images": images,
        "search_images": images,
        "search_event_images": images,
        "template_anno": torch.zeros(1, 2, 4),
        "future_images": images + 1.0,
        "future_event_images": images + 2.0,
        "future_valid": torch.ones(2, 8, dtype=torch.bool),
        "history_images": images + 3.0,
        "history_event_images": images + 4.0,
        "history_valid": torch.ones(2, 8, dtype=torch.bool),
    }

    actor.forward_pass(data)

    assert not any(
        key.startswith(("future_", "history_"))
        for key in actor.net.kwargs
    )


def test_actor_forwards_global_rgb_event_search_only_for_frame_level_reappearance():
    class CaptureNet:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {}

    actor = object.__new__(PETTrackActor)
    actor.expert_enabled = False
    actor.cfg = SimpleNamespace(
        MODEL=SimpleNamespace(BACKBONE=SimpleNamespace(CE_LOC=False)))
    actor.settings = SimpleNamespace(num_template=1)
    actor.net = CaptureNet()
    images = torch.zeros(1, 2, 3, 4, 4)
    data = {
        "template_images": images,
        "template_event_images": images,
        "search_images": images,
        "search_event_images": images,
        "template_anno": torch.zeros(1, 2, 4),
        "redetect_search_images": images + 1.0,
        "redetect_search_event_images": images + 2.0,
        "is_reappear": torch.tensor([[True, False]]),
    }

    actor.forward_pass(data)

    assert actor.net.kwargs["redetect_images"].shape == (2, 1, 3, 4, 4)
    assert actor.net.kwargs["redetect_event_images"].shape == (2, 1, 3, 4, 4)
    assert torch.equal(
        actor.net.kwargs["redetect_mask"], torch.tensor([True, False]))


def test_trainer_passes_normalized_training_progress_to_actor():
    received = []

    class CaptureActor:
        net = torch.nn.Linear(1, 1)

        def train(self, training):
            pass

        def __call__(self, data):
            received.append(data["training_progress"])
            return torch.tensor(0.0), {}

    class Loader:
        training = False
        stack_dim = 0

        def __iter__(self):
            return iter([{"template_images": torch.zeros(1, 1)}])

        def __len__(self):
            return 1

    trainer = object.__new__(LTRTrainer)
    trainer.actor = CaptureActor()
    trainer.move_data_to_gpu = False
    trainer.device = torch.device("cpu")
    trainer.epoch = 30
    trainer.max_epochs = 60
    trainer.settings = SimpleNamespace(print_interval=1, local_rank=0)
    trainer.use_amp = False
    trainer.wandb_writer = None
    trainer._init_timing = lambda: None
    trainer._update_stats = lambda *args: None
    trainer._print_stats = lambda *args: None
    trainer.start_time = trainer.prev_time = 0.0
    trainer.avg_date_time = trainer.avg_gpu_trans_time = 0.0
    trainer.avg_forward_time = trainer.num_frames = 1.0

    grad_enabled = torch.is_grad_enabled()
    trainer.cycle_dataset(Loader())

    assert received == [0.5]
    assert torch.is_grad_enabled() is grad_enabled


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
    assert 'PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' in config
    assert "[supervisord]" not in config
    assert "autostart=true" in config
    assert "[program:pettrack_srbt_watch_v8]" in config


def test_canonical_config_contains_local_experts_but_no_legacy_pet_or_c3_nodes():
    from copy import deepcopy
    from lib.config.pet_track.config import cfg, update_config_from_file

    configured = deepcopy(cfg)
    update_config_from_file("experiments/pet_track/felt_pet_track.yaml", configured)
    forbidden = {
        "PET", "EVENT_BELIEF", "EPSM", "ABSENCE",
        "MEMORY_POLICY", "HETEROGENEOUS_TAIL", "EVENT_TRIGGER",
        "STATE_MACHINE",
    }
    assert forbidden.isdisjoint(configured.MODEL)
    assert configured.MODEL.EXPERT.ENABLE is True
    assert configured.MODEL.EXPERT.NAMES == [
        "generalist",
        "motion_fm",
        "small_target_st",
        "visibility_foc_ov",
        "discrimination_bi",
    ]
    assert "MAX_ACTIVE" not in configured.MODEL.EXPERT
    assert "ROUTER_HIDDEN_DIM" not in configured.MODEL.EXPERT
    assert "TEMPORAL_MOMENTUM" not in configured.MODEL.EXPERT
    assert configured.MODEL.PRETRAINED_BASELINE_CKPT == (
        "pretrained_networks/AMTTrack_ep0098.pth.tar")
    assert configured.MODEL.PRETRAINED_SRBT_CKPT == ""
    assert configured.MODEL.PRETRAINED_EXPERT_CKPT == (
        "pretrained_networks/PETTrack_specialists_best_ep0020.pth.tar")
    assert configured.MODEL.INIT_CHECKPOINT == ""


def test_canonical_training_strategy_matches_dual_4090_capacity():
    from copy import deepcopy
    from lib.config.pet_track.config import cfg, update_config_from_file

    configured = deepcopy(cfg)
    update_config_from_file("experiments/pet_track/felt_pet_track.yaml", configured)

    assert configured.DATA.TRAIN.SAMPLE_PER_EPOCH == 4200
    assert configured.DATA.VAL.SAMPLE_PER_EPOCH == 10000
    assert configured.TRAIN.BATCH_SIZE == 6
    assert configured.TRAIN.NUM_WORKER == 4
    assert configured.TRAIN.PERSISTENT_WORKERS is False
    assert configured.TRAIN.LOAD_LATEST is True
    assert configured.TRAIN.EXPERT_PHASE == "specialize"
    assert configured.TRAIN.MIN_EPOCH == 25
    assert configured.TRAIN.EPOCH == 60
    assert configured.TRAIN.REFINE_MAX_EPOCH == 12
    assert configured.TRAIN.LR == 0.00002
    assert configured.TRAIN.LR_DROP_EPOCH == 50
    assert configured.TRAIN.REFINE_TAIL_LR == 0.000001
    assert configured.TRAIN.REFINE_MEMORY_LR == 0.0000005
    assert configured.TRAIN.VAL_START_EPOCH == 1
    expected_schedule = [
        [1, 1, 1],
        [5, 5, 1],
        [10, 50, 5],
        [51, 60, 1],
    ]
    assert configured.TRAIN.VAL_SCHEDULE == expected_schedule
    assert configured.TRAIN.SEQUENCE_VAL_SCHEDULE == expected_schedule
    assert configured.TRAIN.REFINE_SEQUENCE_VAL_SCHEDULE == [[1, -1, 1]]
    assert configured.TRAIN.SRBT_LOSS.EXISTENCE_WEIGHT == 1.0
    assert configured.TRAIN.SRBT_LOSS.FOCAL_GAMMA == 2.0
    assert configured.TRAIN.RECOVERY_LOSS.IDENTITY_WEIGHT == 1.0
    assert configured.TRAIN.RECOVERY_LOSS.RANKING_WEIGHT == 0.5
    assert configured.TRAIN.RECOVERY_LOSS.RANKING_MARGIN == 0.2
    assert "EXPERT_LOSS" not in configured.TRAIN
    assert configured.DATA.CHALLENGE_SAMPLING.ENABLE is True
    assert configured.DATA.CHALLENGE_SAMPLING.PRECISE is True
    assert configured.DATA.CHALLENGE_SAMPLING.MANIFEST == (
        "/root/fnvme/PTE_v8_manifests/felt_train_expert_owners.json")
    assert configured.DATA.CHALLENGE_SAMPLING.VAL_MANIFEST == (
        "/root/fnvme/PTE_v8_manifests/felt_val_expert_owners.json")
    assert configured.MODEL.SRBT.CONTROLLER.THETA_PRESENT == 0.70
    assert configured.MODEL.SRBT.CONTROLLER.ABSENT_FRAMES == 4
    assert configured.TRAIN.SAVE_EPOCHS == []
    assert configured.TRAIN.SAVE_LATEST_EACH_EPOCH is True
    assert configured.TRAIN.SAVE_BEST is True
    assert configured.TRAIN.SPECIALIST_GATE_ENABLE is True
    assert configured.TRAIN.SPECIALIST_MIN_COUNT == 100
    assert configured.TRAIN.SPECIALIST_MIN_DELTA == 0.02
    assert configured.TRAIN.GENERALIST_MAX_DROP == 0.005


def test_supervisor_uses_fresh_route_free_stage1_run_directory():
    project_root = Path(__file__).resolve().parents[2]
    supervisor = (
        project_root / "tracking" / "supervisord_local_experts_v8.conf"
    ).read_text(encoding="utf-8")

    assert "route_free_specialize_20260716" in supervisor
    assert "expert_route_20260715" not in supervisor


def test_actor_avoids_legacy_counterfactual_route_outputs():
    import inspect

    source = inspect.getsource(PETTrackActor.forward_pass)

    assert "_balanced_singleton_routes" not in source
    assert "_route_aux_due" not in source
    assert 'forward_kwargs["counterfactual_routes"]' not in source
    assert 'forward_kwargs["singleton_expert_aux"]' not in source
    assert "model.forward_all_routes" not in source
    assert "model.forward_singleton_routes" not in source


def test_recovery_identity_loss_uses_amp_safe_bce_with_logits():
    import inspect

    source = inspect.getsource(PETTrackActor._compute_redetect_loss)

    assert "binary_cross_entropy_with_logits" in source
    assert "binary_cross_entropy(" not in source
