import numpy as np
import pytest
import torch
import yaml
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import lib.test.tracker.pet_track as pet_tracker_module
import lib.train.trainers.ltr_trainer as ltr_trainer_module
from lib.test.analysis.felt_metrics import (
    felt_absent_balanced_accuracy,
    felt_reappearance_auc_proxy,
    felt_success_auc_proxy,
    felt_success_curve_proxy,
)
from lib.test.evaluation.feltdataset import FELTDataset
from lib.test.evaluation.tracker import Tracker
from lib.config.pet_track.config import cfg as default_cfg
from lib.train.admin.stats import AverageMeter
from lib.train.sequence_validation import (
    expert_validation_sums,
    refine_checkpoint_is_accepted,
    recovery_diagnostic_sums,
    run_felt_sequence_validation,
    sequence_validation_due,
    specialist_passes,
)
from lib.train.train_script import _select_epoch_loaders
from lib.train.trainers.base_trainer import BaseTrainer
from lib.train.trainers.ltr_trainer import LTRTrainer


def _write_felt_sequence(train_root, name, presence):
    sequence_root = train_root / name
    aps_root = sequence_root / f"{name}_aps"
    dvs_root = sequence_root / f"{name}_dvs"
    aps_root.mkdir(parents=True)
    dvs_root.mkdir(parents=True)
    boxes = [f"{index + 1},{index + 1},10,10" for index in range(len(presence))]
    (sequence_root / "groundtruth.txt").write_text("\n".join(boxes), encoding="ascii")
    (sequence_root / "absent.txt").write_text(
        "\n".join(str(value) for value in presence), encoding="ascii")
    for frame_index in range(1, len(presence) + 1):
        (aps_root / f"frame{frame_index:04d}.png").touch()
        (dvs_root / f"frame{frame_index:04d}.png").touch()


def test_felt_sr_proxy_normalizes_over_present_frames():
    ground_truth = np.array([
        [1.0, 1.0, 10.0, 10.0],
        [50.0, 50.0, 10.0, 10.0],
        [20.0, 20.0, 10.0, 10.0],
        [70.0, 70.0, 10.0, 10.0],
    ])
    predictions = ground_truth.copy()
    presence = np.array([1, 0, 1, 0])

    curve = felt_success_curve_proxy(predictions, ground_truth, presence)

    assert curve.shape == (21,)
    assert curve[10] == pytest.approx(1.0)
    assert felt_success_auc_proxy(
        predictions, ground_truth, presence) == pytest.approx(20 / 21)


def test_felt_sr_proxy_rejects_sequences_without_present_frames():
    boxes = np.ones((2, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="target-present frame"):
        felt_success_curve_proxy(
            boxes, boxes, np.zeros(2, dtype=np.uint8))


def test_felt_sr_proxy_carries_invalid_prediction_from_previous_frame():
    ground_truth = np.array([
        [10.0, 10.0, 8.0, 8.0],
        [10.0, 10.0, 8.0, 8.0],
    ])
    predictions = np.array([
        [10.0, 10.0, 8.0, 8.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    curve = felt_success_curve_proxy(
        predictions, ground_truth, np.ones(2, dtype=np.uint8))

    assert curve[10] == pytest.approx(1.0)


def test_felt_absent_balanced_accuracy_weights_present_and_absent_equally():
    presence = np.array([1, 0, 0, 1, 1, 1, 0, 1])
    predicted_absent = np.array([0, 1, 0, 0, 0, 1, 1, 0])

    score = felt_absent_balanced_accuracy(predicted_absent, presence)

    assert score == pytest.approx(((2 / 3) + (4 / 5)) / 2)


def test_felt_reappearance_proxy_scores_first_visible_frames_after_absence():
    ground_truth = np.array([
        [10.0, 10.0, 8.0, 8.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [20.0, 20.0, 8.0, 8.0],
        [21.0, 21.0, 8.0, 8.0],
        [22.0, 22.0, 8.0, 8.0],
    ])
    predictions = ground_truth.copy()
    predictions[4] = [80.0, 80.0, 8.0, 8.0]
    presence = np.array([1, 0, 0, 1, 1, 1])

    score = felt_reappearance_auc_proxy(
        predictions, ground_truth, presence, horizon=2)

    assert score == pytest.approx(10 / 21)
    assert np.isnan(felt_reappearance_auc_proxy(
        predictions, ground_truth, np.ones(6), horizon=2))


def test_recovery_diagnostics_are_posthoc_and_aggregation_ready():
    presence = np.array([1, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    ground_truth = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [30.0, 30.0, 10.0, 10.0],
        [31.0, 30.0, 10.0, 10.0],
        [32.0, 30.0, 10.0, 10.0],
        [33.0, 30.0, 10.0, 10.0],
        [34.0, 30.0, 10.0, 10.0],
        [35.0, 30.0, 10.0, 10.0],
    ])
    predictions = ground_truth.copy()
    predictions[3] = [80.0, 80.0, 10.0, 10.0]
    predicted_absent = [False, True, True, True, False, False, False, False, False]
    trace = [
        {"action": "track", "event_centers": [], "identity_scores": []},
        {"action": "absent", "event_centers": [], "identity_scores": [0.8, 0.2]},
        {"action": "absent", "event_centers": [], "identity_scores": [0.7]},
        {
            "action": "absent",
            "event_centers": [[5.0, 5.0], [35.0, 35.0]],
            "identity_scores": [0.1, 0.9],
        },
        {"action": "verify", "event_centers": [], "identity_scores": []},
        {"action": "track", "event_centers": [], "identity_scores": []},
        {"action": "track", "event_centers": [], "identity_scores": []},
        {"action": "track", "event_centers": [], "identity_scores": []},
        {"action": "track", "event_centers": [], "identity_scores": []},
    ]

    sums = recovery_diagnostic_sums(
        predictions,
        predicted_absent,
        [0.1] * len(presence),
        ground_truth,
        presence,
        trace,
        identity_threshold=0.75,
    )

    assert sums["EVENT_RECALL_AT_1_HITS"] == 0
    assert sums["EVENT_RECALL_AT_3_HITS"] == 1
    assert sums["EVENT_RECALL_AT_5_HITS"] == 1
    assert sums["EVENT_REAPPEARANCE_COUNT"] == 1
    assert sums["RGB_FALSE_ACCEPTS"] == 1
    assert sums["RGB_ABSENT_CANDIDATES"] == 3
    assert sums["RECOVERY_LATENCY_SUM"] == 1
    assert sums["RECOVERY_SUCCESS_COUNT"] == 1
    assert sums["RECOVERY_EVENT_COUNT"] == 1
    assert sums["REAPPEAR_IOU_AT_1_SUM"] == pytest.approx(0.0)
    assert sums["REAPPEAR_IOU_AT_3_SUM"] == pytest.approx(1.0)
    assert sums["REAPPEAR_IOU_AT_5_SUM"] == pytest.approx(1.0)
    assert sums["VISIBLE_RETENTION_COUNT"] == 1
    assert sums["VISIBLE_RETENTION_IOU_SUM"] == pytest.approx(1.0)
    assert sums["TRACK_FRAME_COUNT"] == 4
    assert sums["RECOVERY_FRAME_COUNT"] == 4
    assert sums["TRACK_TIME_SUM"] == pytest.approx(0.4)
    assert sums["RECOVERY_TIME_SUM"] == pytest.approx(0.4)


def test_expert_validation_sums_use_visible_owner_frames_only():
    ground_truth = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [20.0, 20.0, 10.0, 10.0],
        [30.0, 30.0, 10.0, 10.0],
    ])
    expert_boxes = np.repeat(ground_truth[:, None, :], 5, axis=1)
    expert_boxes[1, 0] = [80.0, 80.0, 10.0, 10.0]
    expert_boxes[3, 0] = [90.0, 90.0, 10.0, 10.0]
    ensemble_boxes = ground_truth.copy()

    sums = expert_validation_sums(
        expert_boxes=expert_boxes,
        ensemble_boxes=ensemble_boxes,
        ground_truth_boxes=ground_truth,
        target_visible=[1, 1, 0, 1],
        owner_ids=[0, 2, 3, 2],
    )

    assert sums["EXPERT_2_COUNT"] == 2
    assert sums["EXPERT_2_IOU_SUM"] == pytest.approx(2.0)
    assert sums["EXPERT_2_SUCCESS_HITS"] == 2
    assert sums["EXPERT_2_GENERALIST_IOU_SUM"] == pytest.approx(0.0)
    assert sums["EXPERT_2_ENSEMBLE_IOU_SUM"] == pytest.approx(2.0)
    assert sums["EXPERT_3_COUNT"] == 0


def test_specialist_gate_requires_count_and_delta():
    assert specialist_passes(
        count=100, specialist=0.62, generalist=0.59)
    assert not specialist_passes(
        count=2, specialist=0.90, generalist=0.50)
    assert not specialist_passes(
        count=100, specialist=0.60, generalist=0.59)


def test_refine_rolls_back_on_generalist_regression():
    assert not refine_checkpoint_is_accepted(
        stage1_generalist=0.700,
        refine_generalist=0.694,
        specialist_regressions=[False] * 4,
    )
    assert refine_checkpoint_is_accepted(
        stage1_generalist=0.700,
        refine_generalist=0.695,
        specialist_regressions=[False] * 4,
    )


def test_felt_val_uses_train_val1k_and_exposes_frame_presence(tmp_path):
    train_root = tmp_path / "train"
    train_root.mkdir()
    _write_felt_sequence(train_root, "sequence_a", [1, 1])
    _write_felt_sequence(train_root, "sequence_b", [1, 0])
    (train_root / "list1k.txt").write_text(
        "sequence_a\nsequence_b\n", encoding="ascii")
    (train_root / "val1k.txt").write_text("1\n", encoding="ascii")
    sequences = FELTDataset("val", base_path=train_root).get_sequence_list()

    assert [sequence.name for sequence in sequences] == ["sequence_b"]
    assert sequences[0].target_visible.tolist() == [1, 0]
    assert sequences[0].frame_info(1) == {}


def test_felt_val_rejects_incomplete_split_sequence(tmp_path):
    train_root = tmp_path / "train"
    train_root.mkdir()
    _write_felt_sequence(train_root, "complete", [1, 1])
    (train_root / "list1k.txt").write_text(
        "complete\nmissing\n", encoding="ascii")
    (train_root / "val1k.txt").write_text("1\n", encoding="ascii")

    with pytest.raises(FileNotFoundError, match="missing"):
        FELTDataset("val", base_path=train_root)


def test_felt_val_rejects_misaligned_modalities_and_annotations(tmp_path):
    train_root = tmp_path / "train"
    train_root.mkdir()
    _write_felt_sequence(train_root, "misaligned", [1, 1])
    (train_root / "misaligned" / "misaligned_dvs" / "frame0002.png").unlink()
    (train_root / "list1k.txt").write_text("misaligned\n", encoding="ascii")
    (train_root / "val1k.txt").write_text("0\n", encoding="ascii")

    dataset = FELTDataset("val", base_path=train_root)
    with pytest.raises(ValueError, match="frame count mismatch"):
        dataset.get_sequence_list()


def test_test_tracker_uses_injected_network_without_building_or_loading(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("injected validation must not build or load a network")

    class DummyThor:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(pet_tracker_module, "build_pet_track", forbidden)
    monkeypatch.setattr(pet_tracker_module, "_load_srbt_eval_checkpoint", forbidden)
    monkeypatch.setattr(
        pet_tracker_module, "build_hypothesis_tracker", lambda _cfg: object())
    monkeypatch.setattr(
        pet_tracker_module, "build_visibility_controller", lambda _cfg: object())
    monkeypatch.setattr(pet_tracker_module, "THOR_Wrapper", DummyThor)

    cfg = SimpleNamespace(
        MODEL=SimpleNamespace(
            BACKBONE=SimpleNamespace(STRIDE=16),
            REDETECT=SimpleNamespace(TRAIN_SEARCH_FACTOR=8.0),
        ),
        TEST=SimpleNamespace(
            SEARCH_SIZE=32,
            SHORTTERM_LIBRARY_NUMS=1,
            LONGTERM_LIBRARY_NUMS=1,
            SAMPLE_INTERVAL=1,
            UPDATE_INTERVAL=1,
            LOWER_BOUND=0.1,
            SCORE_THRESHOLD=0.2,
        ),
    )
    params = SimpleNamespace(cfg=cfg, debug=False, save_all_boxes=False)
    network = torch.nn.Linear(1, 1)

    tracker = pet_tracker_module.PETTrack(
        params, dataset_name="felt", network=network)

    assert tracker.network is network
    assert not tracker.network.training
    assert tracker.preprocessor.mean.device == network.weight.device
    assert tracker.output_window.device == network.weight.device


def test_sequence_validation_reuses_test_loop_without_future_ground_truth(capsys):
    class SequenceStub:
        name = "validation_sequence"
        aps_frame_list = ["aps0", "aps1", "aps2"]
        dvs_frame_list = ["dvs0", "dvs1", "dvs2"]
        ground_truth_rect = np.array([
            [1.0, 1.0, 10.0, 10.0],
            [1.0, 1.0, 10.0, 10.0],
            [1.0, 1.0, 10.0, 10.0],
        ])
        target_visible = np.ones(3, dtype=np.uint8)

        @staticmethod
        def init_info():
            return {"init_bbox": [1.0, 1.0, 10.0, 10.0]}

        @staticmethod
        def frame_info(frame_num):
            return {"frame_id": frame_num}

    class TrackerStub:
        supports_absent_output = True

        def __init__(self):
            self.init_info_seen = None
            self.track_info_seen = []

        def initialize(self, _image, _event_image, info, idx=0):
            self.init_info_seen = dict(info)

        def track(self, _image, _event_image, info):
            self.track_info_seen.append(dict(info))
            return {
                "target_bbox": [1.0, 1.0, 10.0, 10.0],
                "absent": False,
            }

        @staticmethod
        def get_update_count():
            return 0, 0, 0

        @staticmethod
        def get_sample_count():
            return 0, 0

    evaluator = Tracker.__new__(Tracker)
    evaluator._read_image = lambda path: path
    created_trackers = []
    injected_networks = []

    def tracker_factory(network, _params):
        tracker = TrackerStub()
        created_trackers.append(tracker)
        injected_networks.append(network)
        return tracker

    cfg = SimpleNamespace(TEST=SimpleNamespace(
        TEMPLATE_FACTOR=2.0,
        TEMPLATE_SIZE=128,
        SEARCH_FACTOR=5.0,
        SEARCH_SIZE=320,
    ))
    network = torch.nn.Linear(1, 1)
    assert network.training

    metrics = run_felt_sequence_validation(
        network,
        cfg,
        sequences=[SequenceStub()],
        evaluator=evaluator,
        tracker_factory=tracker_factory,
    )

    assert injected_networks == [network]
    assert len(created_trackers) == 1
    assert created_trackers[0].init_info_seen == {
        "init_bbox": [1.0, 1.0, 10.0, 10.0]}
    assert [info["frame_id"] for info in created_trackers[0].track_info_seen] == [1, 2]
    assert all(
        not any("ground_truth" in key or key.startswith("future_") for key in info)
        for info in created_trackers[0].track_info_seen
    )
    assert metrics["FELT_SR_PROXY_SUM"] == pytest.approx(20 / 21)
    assert metrics["FELT_ABSENT_BAL_ACC_SUM"] == pytest.approx(1.0)
    assert metrics["FELT_REAPPEARANCE_PROXY_SUM"] == pytest.approx(0.0)
    assert metrics["REAPPEARANCE_SEQUENCE_COUNT"] == 0
    assert metrics["SEQUENCE_COUNT"] == 1
    assert network.training
    assert (
        "SequenceVal rank 0: 1/1 validation_sequence FELT_SR_PROXY=0.952381"
        in capsys.readouterr().out
    )


def test_sequence_validation_aggregates_fixed_manifest_expert_outputs():
    class SequenceStub:
        name = "validation_sequence"
        ground_truth_rect = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
            [20.0, 20.0, 10.0, 10.0],
        ])
        target_visible = np.ones(3, dtype=np.uint8)

        @staticmethod
        def init_info():
            return {"init_bbox": [0.0, 0.0, 10.0, 10.0]}

    expert_trace = np.repeat(
        SequenceStub.ground_truth_rect[:, None, :], 5, axis=1)
    expert_trace[1:, 0] = [80.0, 80.0, 10.0, 10.0]

    class TrackerStub:
        @staticmethod
        def get_expert_diagnostics():
            return expert_trace.copy()

    class EvaluatorStub:
        @staticmethod
        def _track_sequence(_tracker, sequence, _init_info):
            return {
                "target_bbox": sequence.ground_truth_rect.copy(),
                "absent": [False] * len(sequence.ground_truth_rect),
            }

    cfg = SimpleNamespace(TEST=SimpleNamespace(
        TEMPLATE_FACTOR=2.0,
        TEMPLATE_SIZE=128,
        SEARCH_FACTOR=5.0,
        SEARCH_SIZE=320,
    ))
    metrics = run_felt_sequence_validation(
        torch.nn.Linear(1, 1),
        cfg,
        sequences=[SequenceStub()],
        evaluator=EvaluatorStub(),
        tracker_factory=lambda _network, _params: TrackerStub(),
        ownership_manifest={
            "sequences": {
                "validation_sequence": {"owners": [0, 2, 2]},
            },
        },
    )

    assert metrics["EXPERT_0_COUNT"] == 1
    assert metrics["EXPERT_2_COUNT"] == 2
    assert metrics["EXPERT_2_IOU_SUM"] == pytest.approx(2.0)
    assert metrics["EXPERT_2_GENERALIST_IOU_SUM"] == pytest.approx(0.0)
    assert metrics["EXPERT_2_ENSEMBLE_IOU_SUM"] == pytest.approx(2.0)


def test_sequence_validation_skips_invalid_initial_bbox_without_looking_ahead(
        capsys):
    class InvalidSequence:
        name = "invalid_initial"
        ground_truth_rect = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 5.0, 10.0, 10.0],
        ])
        target_visible = np.array([0, 1], dtype=np.uint8)

        @staticmethod
        def init_info():
            return {"init_bbox": [0.0, 0.0, 0.0, 0.0]}

    class ValidSequence:
        name = "valid_initial"
        ground_truth_rect = np.array([
            [1.0, 1.0, 10.0, 10.0],
            [1.0, 1.0, 10.0, 10.0],
        ])
        target_visible = np.ones(2, dtype=np.uint8)

        @staticmethod
        def init_info():
            return {"init_bbox": [1.0, 1.0, 10.0, 10.0]}

    evaluated = []

    class EvaluatorStub:
        @staticmethod
        def _track_sequence(_tracker, sequence, init_info):
            evaluated.append((sequence.name, list(init_info["init_bbox"])))
            if sequence.name == "invalid_initial":
                raise AssertionError("invalid sequence must not reach Test tracking")
            return {
                "target_bbox": sequence.ground_truth_rect.copy(),
                "absent": [False] * len(sequence.ground_truth_rect),
            }

    cfg = SimpleNamespace(TEST=SimpleNamespace(
        TEMPLATE_FACTOR=2.0,
        TEMPLATE_SIZE=128,
        SEARCH_FACTOR=5.0,
        SEARCH_SIZE=320,
    ))
    network = torch.nn.Linear(1, 1)

    metrics = run_felt_sequence_validation(
        network,
        cfg,
        sequences=[InvalidSequence(), ValidSequence()],
        evaluator=EvaluatorStub(),
        tracker_factory=lambda _network, _params: object(),
    )

    assert evaluated == [("valid_initial", [1.0, 1.0, 10.0, 10.0])]
    assert metrics["SEQUENCE_COUNT"] == 1
    assert metrics["FELT_SR_PROXY_SUM"] == pytest.approx(20 / 21)
    assert metrics["FELT_ABSENT_BAL_ACC_SUM"] == pytest.approx(1.0)
    output = capsys.readouterr().out
    assert "SequenceVal skipped invalid initial bbox: invalid_initial" in output
    assert "SequenceVal rank 0: 1/1 valid_initial" in output


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        (39, False),
        (40, True),
        (41, False),
        (42, True),
        (49, False),
        (50, True),
        (51, True),
        (60, True),
    ],
)
def test_sequence_validation_schedule(epoch, expected):
    schedule = [[40, 50, 2], [51, -1, 1]]

    assert sequence_validation_due(epoch, schedule) is expected


def test_sequence_validation_failure_keeps_completed_epoch_checkpoint(tmp_path):
    events = []

    class SchedulerStub:
        def step(self):
            events.append("scheduler")

    trainer = LTRTrainer.__new__(LTRTrainer)
    trainer.loaders = []
    trainer.epoch = 0
    trainer.lr_scheduler = SchedulerStub()
    trainer._checkpoint_dir = str(tmp_path)
    trainer.settings = SimpleNamespace(
        local_rank=0,
        scheduler_type="step",
        sequence_val_enable=True,
        sequence_val_schedule=[[1, 1, 1]],
        save_latest_each_epoch=True,
        save_best=False,
        save_every_epoch=False,
        save_epoch_interval=0,
        save_last_epochs=0,
        save_epochs=[],
        save_final_checkpoint=False,
    )
    trainer.save_checkpoint = lambda name=None: events.append(name)

    def fail_validation():
        events.append("validation")
        raise RuntimeError("sequence validation failed")

    trainer._run_sequence_validation = fail_validation

    with pytest.raises(RuntimeError, match="sequence validation failed"):
        trainer.train(max_epochs=1, fail_safe=False)

    assert events == ["scheduler", "latest", "validation"]


def test_trainer_records_aggregated_sequence_auc(
        monkeypatch, tmp_path, capsys):
    calls = []

    def fake_sequence_validation(network, cfg, **kwargs):
        calls.append((network, cfg, kwargs))
        metrics = {
            "FELT_SR_PROXY_SUM": 1.5,
            "FELT_ABSENT_BAL_ACC_SUM": 1.2,
            "FELT_REAPPEARANCE_PROXY_SUM": 0.4,
            "REAPPEARANCE_SEQUENCE_COUNT": 1,
            "SEQUENCE_COUNT": 2,
            "EVENT_RECALL_AT_1_HITS": 1,
            "EVENT_RECALL_AT_3_HITS": 2,
            "EVENT_RECALL_AT_5_HITS": 2,
            "EVENT_REAPPEARANCE_COUNT": 2,
            "RGB_FALSE_ACCEPTS": 1,
            "RGB_ABSENT_CANDIDATES": 10,
            "RECOVERY_LATENCY_SUM": 3,
            "RECOVERY_SUCCESS_COUNT": 2,
            "RECOVERY_EVENT_COUNT": 3,
            "REAPPEAR_IOU_AT_1_SUM": 1.0,
            "REAPPEAR_IOU_AT_1_COUNT": 2,
            "REAPPEAR_IOU_AT_3_SUM": 1.4,
            "REAPPEAR_IOU_AT_3_COUNT": 2,
            "REAPPEAR_IOU_AT_5_SUM": 1.6,
            "REAPPEAR_IOU_AT_5_COUNT": 2,
            "VISIBLE_RETENTION_IOU_SUM": 12,
            "VISIBLE_RETENTION_COUNT": 20,
            "TRACK_TIME_SUM": 2,
            "TRACK_FRAME_COUNT": 10,
            "RECOVERY_TIME_SUM": 4,
            "RECOVERY_FRAME_COUNT": 8,
        }
        metrics.update({
            "EXPERT_2_COUNT": 10,
            "EXPERT_2_IOU_SUM": 7.5,
            "EXPERT_2_SUCCESS_HITS": 6,
            "EXPERT_2_GENERALIST_IOU_SUM": 5,
            "EXPERT_2_GENERALIST_SUCCESS_HITS": 4,
            "EXPERT_2_ENSEMBLE_IOU_SUM": 7,
            "EXPERT_2_ENSEMBLE_SUCCESS_HITS": 6,
        })
        return metrics

    monkeypatch.setattr(
        ltr_trainer_module,
        "run_felt_sequence_validation",
        fake_sequence_validation,
        raising=False,
    )
    network = torch.nn.Linear(1, 1)
    cfg = SimpleNamespace()
    trainer = LTRTrainer.__new__(LTRTrainer)
    trainer.actor = SimpleNamespace(net=network, cfg=cfg)
    trainer.settings = SimpleNamespace(
        env=SimpleNamespace(felt_val_dir=str(tmp_path)),
        local_rank=0,
    )
    trainer.device = torch.device("cpu")
    trainer.stats = OrderedDict(val=None, sequence_val=None)
    trainer.epoch = 40

    score = trainer._run_sequence_validation()

    assert score == pytest.approx(0.75)
    assert calls == [(
        network,
        cfg,
        {
            "rank": 0,
            "world_size": 1,
            "felt_val_root": str(tmp_path),
        },
    )]
    assert trainer.stats["sequence_val"]["FELT_SR_PROXY"].avg == pytest.approx(0.75)
    assert trainer.stats["sequence_val"][
        "FELT_ABSENT_BAL_ACC"].avg == pytest.approx(0.6)
    assert trainer.stats["sequence_val"][
        "FELT_REAPPEARANCE_PROXY"].avg == pytest.approx(0.4)
    expected_diagnostics = {
        "EVENT_RECALL_AT_1": 0.5,
        "EVENT_RECALL_AT_3": 1.0,
        "EVENT_RECALL_AT_5": 1.0,
        "RGB_FALSE_ACCEPT_RATE": 0.1,
        "RECOVERY_LATENCY": 1.5,
        "RECOVERY_SUCCESS_RATE": 2 / 3,
        "REAPPEAR_IOU_AT_1": 0.5,
        "REAPPEAR_IOU_AT_3": 0.7,
        "REAPPEAR_IOU_AT_5": 0.8,
        "VISIBLE_RETENTION_IOU": 0.6,
        "TRACK_FPS": 5.0,
        "RECOVERY_FPS": 2.0,
    }
    for name, expected in expected_diagnostics.items():
        assert trainer.stats["sequence_val"][name].avg == pytest.approx(expected)
    assert trainer.stats["sequence_val"][
        "ExpertVal/small_target_st_count"].avg == pytest.approx(10)
    assert trainer.stats["sequence_val"][
        "ExpertVal/small_target_st_iou"].avg == pytest.approx(0.75)
    assert trainer.stats["sequence_val"][
        "ExpertVal/small_target_st_sr"].avg == pytest.approx(0.6)
    assert trainer.stats["sequence_val"][
        "ExpertVal/small_target_st_generalist_iou"].avg == pytest.approx(0.5)
    assert trainer.stats["sequence_val"][
        "ExpertVal/small_target_st_delta_iou"].avg == pytest.approx(0.25)
    assert trainer.stats["sequence_val"][
        "ExpertVal/small_target_st_ensemble_iou"].avg == pytest.approx(0.7)
    assert trainer.stats["sequence_val"][
        "ExpertVal/visibility_foc_ov_reappearance_iou"].avg == pytest.approx(0.5)
    assert trainer.stats["sequence_val"][
        "ExpertVal/visibility_foc_ov_reappearance_success"].avg == pytest.approx(2 / 3)
    assert trainer.stats["sequence_val"][
        "ExpertVal/visibility_foc_ov_rgb_false_accept_count"].avg == pytest.approx(1)
    assert trainer.stats["val"] is None
    output = capsys.readouterr().out
    assert (
        "SequenceVal expert small_target_st: count=10, IoU=0.750000, "
        "generalist=0.500000, delta=0.250000, ensemble=0.700000, "
        "SR=0.600000"
    ) in output


def test_trainer_all_reduces_disjoint_sequence_validation_shards(
        monkeypatch, tmp_path):
    calls = []

    def fake_sequence_validation(network, cfg, **kwargs):
        calls.append(kwargs)
        return {
            "FELT_SR_PROXY_SUM": 0.5,
            "FELT_ABSENT_BAL_ACC_SUM": 0.4,
            "FELT_REAPPEARANCE_PROXY_SUM": 0.1,
            "REAPPEARANCE_SEQUENCE_COUNT": 1,
            "SEQUENCE_COUNT": 1,
        }

    def fake_all_reduce(values, op):
        assert op == torch.distributed.ReduceOp.SUM
        values += torch.tensor(
            [1.5, 1.2, 0.5, 2.0, 1.0], dtype=values.dtype)

    monkeypatch.setattr(
        ltr_trainer_module, "run_felt_sequence_validation",
        fake_sequence_validation)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    trainer = LTRTrainer.__new__(LTRTrainer)
    trainer.actor = SimpleNamespace(
        net=torch.nn.Linear(1, 1), cfg=SimpleNamespace())
    trainer.settings = SimpleNamespace(
        env=SimpleNamespace(felt_val_dir=str(tmp_path)),
        local_rank=1,
    )
    trainer.device = torch.device("cpu")
    trainer.stats = OrderedDict(val=None, sequence_val=None)
    trainer.epoch = 40

    score = trainer._run_sequence_validation()

    assert score == pytest.approx(2.0 / 3.0)
    assert calls == [{
        "rank": 1,
        "world_size": 2,
        "felt_val_root": str(tmp_path),
    }]


def test_trainer_initializes_sequence_val_stats_and_tensorboard(
        monkeypatch, tmp_path):
    writer_names = []

    class WriterStub:
        def __init__(self, _directory, loader_names):
            writer_names.extend(loader_names)

    class ActorStub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(1, 1)

    monkeypatch.setattr(ltr_trainer_module, "TensorboardWriter", WriterStub)
    actor = ActorStub()
    settings = SimpleNamespace(
        env=SimpleNamespace(
            workspace_dir=str(tmp_path),
            tensorboard_dir=str(tmp_path / "tensorboard"),
        ),
        save_dir=None,
        local_rank=0,
        project_path="sequence_validation",
        use_gpu=False,
        use_wandb=False,
        sequence_val_enable=True,
    )
    loader = SimpleNamespace(name="train")

    trainer = LTRTrainer(
        actor,
        [loader],
        torch.optim.SGD(actor.net.parameters(), lr=0.1),
        settings,
    )

    assert list(trainer.stats) == ["train", "sequence_val"]
    assert writer_names == ["train", "sequence_val"]


def test_best_checkpoint_requires_fresh_sequence_auc():
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.settings = SimpleNamespace(
        save_best=True,
        best_loader="sequence_val",
        best_metric="FELT_SR_PROXY",
        best_metric_mode="max",
    )
    meter = AverageMeter()
    trainer.stats = {"sequence_val": {"FELT_SR_PROXY": meter}}
    saved = []
    trainer.save_checkpoint = saved.append

    trainer.epoch = 40
    meter.update(0.45)
    meter.new_epoch()
    trainer._maybe_save_best_checkpoint()

    trainer.epoch = 41
    meter.new_epoch()
    trainer._maybe_save_best_checkpoint()

    trainer.epoch = 42
    meter.update(0.47)
    meter.new_epoch()
    trainer._maybe_save_best_checkpoint()

    assert saved == ["best", "best"]
    assert trainer.best_val_score == pytest.approx(0.47)
    assert trainer.best_val_epoch == 42


def test_specialist_best_checkpoint_requires_all_fresh_gates():
    def meter(value):
        return SimpleNamespace(has_new_data=True, history=[value])

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.settings = SimpleNamespace(
        save_best=True,
        best_loader="sequence_val",
        best_metric="FELT_SR_PROXY",
        best_metric_mode="max",
        specialist_gate_enable=True,
        expert_phase="specialize",
        specialist_min_count=100,
        specialist_min_delta=0.02,
        generalist_reference_iou=0.70,
        generalist_max_drop=0.005,
        visibility_reference_reappear_iou=0.40,
        visibility_reference_reappear_success=0.60,
        visibility_reference_rgb_false_accept_rate=0.10,
    )
    metrics = {
        "FELT_SR_PROXY": meter(0.55),
        "ExpertVal/generalist_count": meter(100),
        "ExpertVal/generalist_iou": meter(0.70),
    }
    for name in ("motion_fm", "small_target_st", "visibility_foc_ov"):
        metrics[f"ExpertVal/{name}_count"] = meter(100)
        metrics[f"ExpertVal/{name}_iou"] = meter(0.62)
        metrics[f"ExpertVal/{name}_generalist_iou"] = meter(0.59)
    trainer.stats = {"sequence_val": metrics}
    trainer.epoch = 25
    saved = []
    trainer.save_checkpoint = saved.append

    trainer._maybe_save_best_checkpoint()
    assert saved == []

    metrics["ExpertVal/discrimination_bi_count"] = meter(100)
    metrics["ExpertVal/discrimination_bi_iou"] = meter(0.62)
    metrics["ExpertVal/discrimination_bi_generalist_iou"] = meter(0.59)
    trainer._maybe_save_best_checkpoint()

    assert saved == []

    metrics.update({
        "ExpertVal/visibility_foc_ov_reappearance_iou": meter(0.42),
        "ExpertVal/visibility_foc_ov_reappearance_success": meter(0.60),
        "RGB_FALSE_ACCEPT_RATE": meter(0.10),
    })
    trainer._maybe_save_best_checkpoint()

    assert saved == ["best_stage1"]
    assert trainer.best_val_score == pytest.approx(0.55)
    assert trainer.specialist_gate_reference == {
        "generalist_iou": pytest.approx(0.70),
        "specialists": {
            "motion_fm": pytest.approx(0.62),
            "small_target_st": pytest.approx(0.62),
            "visibility_foc_ov": pytest.approx(0.62),
            "discrimination_bi": pytest.approx(0.62),
        },
        "visibility": {
            "reappearance_iou": pytest.approx(0.42),
            "reappearance_success": pytest.approx(0.60),
            "rgb_false_accept_rate": pytest.approx(0.10),
        },
    }


def test_refine_best_uses_stage1_checkpoint_reference():
    def meter(value):
        return SimpleNamespace(has_new_data=True, history=[value])

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.settings = SimpleNamespace(
        save_best=True,
        best_loader="sequence_val",
        best_metric="FELT_SR_PROXY",
        best_metric_mode="max",
        specialist_gate_enable=True,
        expert_phase="refine",
        specialist_min_count=100,
        specialist_min_delta=0.02,
        generalist_reference_iou=0.60,
        generalist_max_drop=0.005,
    )
    trainer.specialist_gate_reference = {
        "generalist_iou": 0.70,
        "specialists": {
            name: 0.62 for name in (
                "motion_fm", "small_target_st",
                "visibility_foc_ov", "discrimination_bi")
        },
        "visibility": {
            "reappearance_iou": 0.42,
            "reappearance_success": 0.60,
            "rgb_false_accept_rate": 0.10,
        },
    }
    metrics = {
        "FELT_SR_PROXY": meter(0.58),
        "ExpertVal/generalist_count": meter(100),
        "ExpertVal/generalist_iou": meter(0.694),
    }
    for name in trainer.specialist_gate_reference["specialists"]:
        metrics[f"ExpertVal/{name}_count"] = meter(100)
        metrics[f"ExpertVal/{name}_iou"] = meter(0.63)
        metrics[f"ExpertVal/{name}_generalist_iou"] = meter(0.60)
    metrics.update({
        "ExpertVal/visibility_foc_ov_reappearance_iou": meter(0.44),
        "ExpertVal/visibility_foc_ov_reappearance_success": meter(0.60),
        "RGB_FALSE_ACCEPT_RATE": meter(0.10),
    })
    trainer.stats = {"sequence_val": metrics}
    trainer.epoch = 3
    saved = []
    trainer.save_checkpoint = saved.append

    trainer._maybe_save_best_checkpoint()
    assert saved == []

    metrics["ExpertVal/generalist_iou"] = meter(0.695)
    trainer._maybe_save_best_checkpoint()

    assert saved == ["best_stage2"]


def test_sequence_validation_is_opt_in_and_felt_experiment_selects_sequence_best():
    assert default_cfg.TRAIN.SEQUENCE_VAL_ENABLE is False
    assert default_cfg.TRAIN.SEQUENCE_VAL_SCHEDULE == []
    assert default_cfg.TRAIN.BEST_LOADER == "val"

    project_root = Path(__file__).resolve().parents[2]
    experiment = yaml.safe_load(
        (project_root / "experiments" / "pet_track" / "felt_pet_track.yaml")
        .read_text(encoding="utf-8")
    )

    assert experiment["TRAIN"]["SEQUENCE_VAL_ENABLE"] is True
    assert experiment["TRAIN"]["MIN_EPOCH"] == 25
    assert experiment["TRAIN"]["EPOCH"] == 60
    assert experiment["TRAIN"]["REFINE_MAX_EPOCH"] == 12
    assert experiment["TRAIN"]["SEQUENCE_VAL_SCHEDULE"] == [
        [1, 1, 1],
        [5, 5, 1],
        [10, 50, 5],
        [51, 60, 1],
    ]
    assert experiment["TRAIN"]["SAVE_EPOCHS"] == []
    assert experiment["TRAIN"]["SPECIALIST_GATE_ENABLE"] is True
    assert experiment["TRAIN"]["SPECIALIST_MIN_COUNT"] == 100
    assert experiment["TRAIN"]["SPECIALIST_MIN_DELTA"] == pytest.approx(0.02)
    assert experiment["TRAIN"]["GENERALIST_REFERENCE_IOU"] == pytest.approx(
        0.7229794659718115)
    assert experiment["TRAIN"]["VISIBILITY_REFERENCE_REAPPEAR_IOU"] == pytest.approx(
        0.051049665982931064)
    assert experiment["TRAIN"]["VISIBILITY_REFERENCE_REAPPEAR_SUCCESS"] == 0.0
    assert experiment["TRAIN"]["VISIBILITY_REFERENCE_RGB_FALSE_ACCEPT_RATE"] == 1.0
    assert experiment["TRAIN"]["GENERALIST_MAX_DROP"] == pytest.approx(0.005)
    assert experiment["TRAIN"]["BEST_LOADER"] == "sequence_val"
    assert experiment["TRAIN"]["BEST_METRIC"] == "FELT_SR_PROXY"


def test_sequence_val_best_disables_incompatible_batch_val_loader():
    train_loader = object()
    val_loader = object()

    selected = _select_epoch_loaders(
        train_loader,
        val_loader,
        SimpleNamespace(
            sequence_val_enable=True,
            best_loader="sequence_val",
        ),
    )

    assert selected == [train_loader]
    assert _select_epoch_loaders(
        train_loader,
        val_loader,
        SimpleNamespace(sequence_val_enable=False, best_loader="val"),
    ) == [train_loader, val_loader]
