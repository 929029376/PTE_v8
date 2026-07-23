import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch


def _checkerboard():
    return torch.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])


def _empty_attributes(count):
    return {
        name: [False] * count
        for name in (
            "small_target", "motion", "low_light", "recovery",
            "ambiguity", "deformation", "absent",
        )
    }


def test_manifest_schema_is_multilabel_only(tmp_path):
    ownership = importlib.import_module("lib.train.data.challenge_manifest")
    attributes = _empty_attributes(2)
    attributes["small_target"][1] = True
    attributes["motion"][1] = True
    records = {"sequence-a": {"attributes": attributes}}

    path = tmp_path / "challenges.json"
    ownership.write_manifest(path, records, {})
    loaded = ownership.load_manifest(path)

    assert loaded["schema_version"] == 2
    assert loaded["sequences"] == records
    assert "owners" not in loaded["sequences"]["sequence-a"]
    assert "reasons" not in loaded["sequences"]["sequence-a"]


def test_default_ambiguity_threshold_matches_calibrated_boundary():
    ownership = importlib.import_module("lib.train.data.challenge_manifest")

    assert getattr(ownership, "DEFAULT_AMBIGUITY_THRESHOLD", None) == 1.29


def test_compound_frame_is_reserved_for_dispatch_not_specialization():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    attributes = _empty_attributes(3)
    attributes["motion"][1] = True
    attributes["small_target"][1] = True
    attributes["low_light"][1] = True
    sampler._expert_manifest = {
        "sequence-a": {"attributes": attributes},
    }

    class Dataset:
        sequence_list = ["sequence-a"]

    info = {
        "bbox": torch.ones(3, 4),
        "absent": torch.ones(3, dtype=torch.uint8),
    }
    sampler.expert_phase = "specialize"
    assert sampler._expert_candidate_groups(
        Dataset(), 0, info, training_expert_id=1) == {"expert": []}
    assert sampler._expert_candidate_groups(
        Dataset(), 0, info, training_expert_id=2) == {"expert": []}
    assert sampler._expert_candidate_groups(
        Dataset(), 0, info, training_expert_id=4) == {"expert": []}

    sampler.expert_phase = "dispatch"
    for expert_id in (1, 2, 4):
        assert sampler._expert_candidate_groups(
            Dataset(), 0, info, training_expert_id=expert_id
        ) == {"expert": [1]}

    labels = sampler._frame_challenge_labels(Dataset(), 0, info, 1)
    assert labels.tolist() == [True, True, True, False, False, False, False]


def test_compound_stage_requires_two_distinct_specialists_per_frame():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    labels = {
        name: torch.zeros(4, dtype=torch.bool)
        for name in (
            "small_target", "motion", "low_light", "recovery",
            "ambiguity", "deformation", "absent",
        )
    }
    labels["motion"][[0, 1, 3]] = True
    labels["small_target"][[1, 3]] = True
    labels["low_light"][2] = True
    labels["ambiguity"][2] = True
    labels["recovery"][3] = True

    compound = sampler._compound_frame_mask(labels)

    assert compound.tolist() == [False, True, False, True]


def test_specialize_keeps_multiple_labels_owned_by_one_specialist():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler.expert_phase = "specialize"
    attributes = _empty_attributes(3)
    attributes["low_light"][1] = True
    attributes["ambiguity"][1] = True
    sampler._expert_manifest = {
        "sequence-a": {"attributes": attributes},
    }

    class Dataset:
        sequence_list = ["sequence-a"]

    info = {
        "bbox": torch.ones(3, 4),
        "absent": torch.ones(3, dtype=torch.uint8),
    }

    assert sampler._expert_candidate_groups(
        Dataset(), 0, info, training_expert_id=4) == {"expert": [1]}
    labels = sampler._frame_challenge_labels(Dataset(), 0, info, 1)
    assert labels.tolist() == [False, False, True, False, True, False, False]


def test_sampler_has_no_single_or_composite_compatibility_modes():
    sampler_module = importlib.import_module("lib.train.data.sampler")

    assert not hasattr(
        sampler_module.TrackingSampler, "_challenge_sampling_mask")
    assert not hasattr(
        sampler_module.TrackingSampler, "_active_challenge_sampling_mode")


def test_sampler_separates_training_expert_from_multilabel_truth():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler.train_cls = False
    sampler.precise_expert_sampling = True
    sampler.samples_per_epoch = 4
    sampler.training_expert_ids = (2,)
    sampler.specialist_stage_schedule = ()
    challenge_labels = torch.tensor(
        [True, True, False, False, False, False, False])
    sampler.getitem = lambda training_expert_id=None: {
        "training_expert_id": torch.tensor(training_expert_id),
        "challenge_labels": challenge_labels.clone(),
    }

    sample = sampler[0]

    assert sample["training_expert_id"].item() == 2
    assert sample["challenge_labels"].tolist() == challenge_labels.tolist()
    assert "expert_owner_id" not in sample


def test_precise_sampler_skips_sequence_missing_from_manifest(monkeypatch):
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler.precise_expert_sampling = True
    sampler._expert_manifest = {"covered": {}}
    sampler.num_search_frames = 1
    sampler.num_template_frames = 1

    class Dataset:
        sequence_list = ["missing", "covered"]

        @staticmethod
        def get_num_sequences():
            return 2

        @staticmethod
        def get_sequence_info(seq_id):
            return {"visible": torch.ones(20, dtype=torch.bool)}

    sampled_ids = iter([0, 1])
    monkeypatch.setattr(
        sampler_module.random, "randint", lambda *_: next(sampled_ids))

    seq_id, _, _ = sampler.sample_seq_from_dataset(Dataset(), True)

    assert seq_id == 1


def test_observation_scores_detect_aps_distractor_and_ignore_future_frames():
    ownership = importlib.import_module("lib.train.data.challenge_manifest")
    aps = torch.zeros(3, 32, 32, 3)
    dvs = torch.zeros_like(aps)
    pattern = _checkerboard()
    for channel in range(3):
        aps[0, 2:6, 2:6, channel] = pattern
        aps[1, 2:6, 2:6, channel] = pattern
        aps[1, 20:24, 20:24, channel] = pattern
        dvs[1, 2:6, 2:6, channel] = 1.0
    boxes = torch.tensor([
        [2.0, 2.0, 4.0, 4.0],
        [2.0, 2.0, 4.0, 4.0],
        [2.0, 2.0, 4.0, 4.0],
    ])
    presence = torch.tensor([1, 1, 1])

    scores = ownership.compute_observation_scores(aps, dvs, boxes, presence)
    changed_future = aps.clone()
    changed_future[2] = torch.rand_like(changed_future[2])
    rescored = ownership.compute_observation_scores(
        changed_future, dvs, boxes, presence)

    assert scores["ambiguity"][1] > 0.8
    assert scores["event_motion"][1] > 0.0
    assert torch.equal(scores["ambiguity"][:2], rescored["ambiguity"][:2])
    assert torch.equal(scores["event_motion"][:2], rescored["event_motion"][:2])


def test_ambiguity_ignores_responses_inside_current_target_region():
    ownership = importlib.import_module("lib.train.data.challenge_manifest")
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    target = (x + 2 * y).to(torch.float32)
    aps = torch.zeros(2, 64, 64, 3)
    dvs = torch.zeros_like(aps)
    for channel in range(3):
        aps[0, 8:24, 8:24, channel] = target
        aps[1, 24:40, 24:40, channel] = target
    boxes = torch.tensor([
        [8.0, 8.0, 16.0, 16.0],
        [24.0, 24.0, 16.0, 16.0],
    ])

    scores = ownership.compute_observation_scores(
        aps, dvs, boxes, torch.ones(2, dtype=torch.bool))

    assert scores["ambiguity"][1] < 0.1


def test_ambiguity_preserves_strength_of_distractor_over_target():
    ownership = importlib.import_module("lib.train.data.challenge_manifest")
    torch.manual_seed(11)
    target = torch.rand(8, 8)
    degraded_target = 0.15 * target + 0.85 * torch.roll(target, 1, 0)
    aps = torch.zeros(2, 64, 64, 3)
    dvs = torch.zeros_like(aps)
    for channel in range(3):
        aps[0, 8:16, 8:16, channel] = target
        aps[1, 24:32, 24:32, channel] = degraded_target
        aps[1, 48:56, 48:56, channel] = target
    boxes = torch.tensor([
        [8.0, 8.0, 8.0, 8.0],
        [24.0, 24.0, 8.0, 8.0],
    ])

    scores = ownership.compute_observation_scores(
        aps, dvs, boxes, torch.ones(2, dtype=torch.bool))

    assert scores["ambiguity"][1] > 1.0


def test_manifest_round_trip_records_schema_hash_and_removes_temporary_file(tmp_path):
    ownership = importlib.import_module("lib.train.data.challenge_manifest")
    path = tmp_path / "challenges.json"
    thresholds = {"ambiguity_threshold": 0.8, "recovery_window": 8}
    attributes = _empty_attributes(2)
    attributes["motion"][1] = True
    attributes["small_target"][1] = True
    records = {"sequence-a": {"attributes": attributes}}

    ownership.write_manifest(path, records, thresholds)
    loaded = ownership.load_manifest(path)

    assert loaded["schema_version"] == 2
    assert loaded["thresholds"] == thresholds
    assert loaded["sequences"] == records
    assert len(loaded["config_sha256"]) == 64
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8")) == loaded


def test_manifest_rejects_misaligned_challenge_attributes(tmp_path):
    ownership = importlib.import_module("lib.train.data.challenge_manifest")
    attributes = _empty_attributes(2)
    attributes["small_target"] = [True]

    with pytest.raises(ValueError, match="equal lengths"):
        ownership.write_manifest(
            tmp_path / "challenges.json",
            {"sequence-a": {"attributes": attributes}},
            {},
        )


def test_manifest_rejects_old_owner_schema_without_compatibility(tmp_path):
    ownership = importlib.import_module("lib.train.data.challenge_manifest")

    with pytest.raises(ValueError, match="only attributes"):
        ownership.write_manifest(
            tmp_path / "old.json",
            {"sequence-a": {"owners": [0], "reasons": ["generalist"]}},
            {},
        )


def test_sequence_record_is_built_one_frame_at_a_time():
    ownership = importlib.import_module("lib.train.data.challenge_manifest")

    class Dataset:
        sequence_list = ["sequence-a"]

        def __init__(self):
            self.calls = []
            self.aps = [torch.zeros(8, 8, 3), torch.zeros(8, 8, 3)]
            self.dvs = [torch.zeros(8, 8, 3), torch.zeros(8, 8, 3)]
            self.aps[0][2:6, 2:6] = 1.0

        def get_sequence_info(self, seq_id):
            return {
                "bbox": torch.tensor([
                    [2.0, 2.0, 4.0, 4.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]),
                "absent": torch.tensor([1, 0], dtype=torch.uint8),
            }

        def get_frames(self, seq_id, frame_ids, anno):
            self.calls.append(tuple(frame_ids))
            assert len(frame_ids) == 1
            frame_id = frame_ids[0]
            return [self.aps[frame_id]], [self.dvs[frame_id]], {}, {}

    dataset = Dataset()
    record = ownership.build_sequence_record(dataset, 0)

    assert dataset.calls == [(0,), (1,)]
    assert set(record) == {"attributes"}
    assert all(len(values) == 2 for values in record["attributes"].values())
    assert record["attributes"]["absent"] == [False, True]


def test_sequence_record_persists_overlapping_challenge_attributes():
    ownership = importlib.import_module("lib.train.data.challenge_manifest")

    class Dataset:
        def get_sequence_info(self, seq_id):
            return {
                "bbox": torch.tensor([[2.0, 2.0, 1.0, 1.0]]),
                "absent": torch.tensor([1], dtype=torch.uint8),
            }

        def get_frames(self, seq_id, frame_ids, anno):
            image = torch.full((8, 8, 3), 5.0)
            return [image], [torch.zeros_like(image)], {}, {}

    record = ownership.build_sequence_record(Dataset(), 0)

    assert set(record) == {"attributes"}
    assert record["attributes"]["low_light"] == [True]
    assert record["attributes"]["small_target"] == [True]
    assert set(record["attributes"]) == {
        "small_target",
        "motion",
        "low_light",
        "recovery",
        "ambiguity",
        "deformation",
        "absent",
    }


def test_dark_search_context_with_bright_target_is_not_exclusive_motion():
    manifest_module = importlib.import_module(
        "lib.train.data.challenge_manifest")
    challenge_module = importlib.import_module(
        "lib.train.data.felt_challenges")

    class Dataset:
        def get_sequence_info(self, seq_id):
            return {
                "bbox": torch.tensor([
                    [8.0, 8.0, 2.0, 2.0],
                    [12.0, 8.0, 2.0, 2.0],
                ]),
                "absent": torch.ones(2, dtype=torch.uint8),
            }

        def get_frames(self, seq_id, frame_ids, anno):
            frame_id = frame_ids[0]
            image = torch.full((32, 32, 3), 100.0)
            x = 8 if frame_id == 0 else 12
            image[5:13, x - 3:x + 5] = 0.0
            image[8:10, x:x + 2] = 100.0
            event = torch.zeros_like(image)
            event[8:10, x:x + 2] = 1.0
            return [image], [event], {}, {}

    record = manifest_module.build_sequence_record(Dataset(), 0)
    attributes = record["attributes"]

    assert attributes["motion"] == [False, True]
    assert attributes["low_light"] == [True, True]
    exclusive = challenge_module.exclusive_specialist_supervision_mask(
        attributes)
    assert exclusive[1, challenge_module.MOTION].item() is False
    assert exclusive[1, challenge_module.DISCRIMINATION].item() is False

    with pytest.raises(ValueError, match="search factor must be positive"):
        manifest_module.build_sequence_record(
            Dataset(), 0, {"low_light_search_factor": 0.0})


def test_manifest_builder_cli_documents_required_paths():
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "tracking/build_challenge_manifest.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--data-root" in result.stdout
    assert "--split" in result.stdout
    assert "--output" in result.stdout
    assert "--workers" in result.stdout
    assert "--low-light-context-median" in result.stdout
    assert "--low-light-search-factor" in result.stdout


def test_manifest_builder_parallel_workers_preserve_all_sequences(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    names = ["sequence-a", "sequence-b"]
    (tmp_path / "list1k.txt").write_text(
        "\n".join(names) + "\n", encoding="ascii")
    (tmp_path / "train1k.txt").write_text("0\n1\n", encoding="ascii")
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[8:16, 10:18] = 255
    for name in names:
        sequence_root = tmp_path / name
        aps_root = sequence_root / f"{name}_aps"
        dvs_root = sequence_root / f"{name}_dvs"
        aps_root.mkdir(parents=True)
        dvs_root.mkdir(parents=True)
        (sequence_root / "groundtruth.txt").write_text(
            "10,8,8,8\n10,8,8,8\n", encoding="ascii")
        (sequence_root / "absent.txt").write_text(
            "1\n1\n", encoding="ascii")
        for frame_id in range(2):
            filename = f"frame{frame_id:04d}.png"
            assert cv2.imwrite(str(aps_root / filename), image)
            assert cv2.imwrite(str(dvs_root / filename), image)

    output = tmp_path / "challenges.json"
    result = subprocess.run(
        [
            sys.executable,
            "tracking/build_challenge_manifest.py",
            "--data-root", str(tmp_path),
            "--split", "train",
            "--output", str(output),
            "--workers", "2",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert set(manifest["sequences"]) == set(names)
    assert manifest["schema_version"] == 2
    assert all(
        set(record) == {"attributes"}
        and all(len(values) == 2 for values in record["attributes"].values())
        for record in manifest["sequences"].values())


def test_manifest_builder_skips_sequence_without_groundtruth(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    complete = "sequence-complete"
    missing = "sequence-missing"
    (tmp_path / "list1k.txt").write_text(
        f"{complete}\n{missing}\n", encoding="ascii")
    (tmp_path / "train1k.txt").write_text("0\n1\n", encoding="ascii")
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    sequence_root = tmp_path / complete
    aps_root = sequence_root / f"{complete}_aps"
    dvs_root = sequence_root / f"{complete}_dvs"
    aps_root.mkdir(parents=True)
    dvs_root.mkdir(parents=True)
    (sequence_root / "groundtruth.txt").write_text("10,8,8,8\n", encoding="ascii")
    (sequence_root / "absent.txt").write_text("1\n", encoding="ascii")
    assert cv2.imwrite(str(aps_root / "frame0000.png"), image)
    assert cv2.imwrite(str(dvs_root / "frame0000.png"), image)
    (tmp_path / missing).mkdir()

    output = tmp_path / "challenges.json"
    result = subprocess.run(
        [
            sys.executable,
            "tracking/build_challenge_manifest.py",
            "--data-root", str(tmp_path),
            "--split", "train",
            "--output", str(output),
            "--workers", "1",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "[skip] sequence-missing:" in result.stdout
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert set(manifest["sequences"]) == {complete}


def test_specialist_epoch_is_equal_training_expert_blocks():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler.samples_per_epoch = 96
    sampler.training_expert_ids = (1, 2, 3, 4)
    sampler.specialist_stage_schedule = ()

    expert_ids = [
        sampler.training_expert_for_index(index) for index in range(96)]
    assert [expert_ids.count(item) for item in range(5)] == [0, 24, 24, 24, 24]


def test_dispatch_sampler_balances_generalist_and_all_specialists(monkeypatch):
    sampler_module = importlib.import_module("lib.train.data.sampler")
    monkeypatch.setattr(
        sampler_module, "load_manifest",
        lambda path: {"sequences": {"sequence-a": {}}})

    class Dataset:
        def __len__(self):
            return 1

    cfg = SimpleNamespace(
        TRAIN=SimpleNamespace(
            EXPERT_PHASE="dispatch",
            SPECIALIST_EXPERT_IDS=[1, 2, 3, 4],
            SPECIALIST_EXPERT_SCHEDULE=[[1, 10, [1]]],
        ),
        DATA=SimpleNamespace(
            SRBT=SimpleNamespace(ENABLE=False, ANCHOR_WEIGHTS=None),
            CHALLENGE_SAMPLING=SimpleNamespace(
                ENABLE=True, PRECISE=True, MANIFEST="manifest.json"),
            PURSUIT=SimpleNamespace(
                ENABLE=False,
                TRANSITION_PROBABILITY=0.5,
                REAPPEAR_PROBABILITY=0.25,
            ),
        ),
    )
    sampler = sampler_module.TrackingSampler(
        datasets=[Dataset()], p_datasets=None, samples_per_epoch=100,
        max_gap=10, num_search_frames=1, num_template_frames=1,
        cfg=cfg, training=True)

    assert sampler.precise_expert_sampling is True
    assert sampler.training_expert_ids == (0, 1, 2, 3, 4)
    assert sampler.specialist_stage_schedule == ()
    expert_ids = [
        sampler.training_expert_for_index(index) for index in range(100)]
    assert [expert_ids.count(item) for item in range(5)] == [20] * 5


def test_dispatch_validation_uses_precise_validation_manifest(monkeypatch):
    sampler_module = importlib.import_module("lib.train.data.sampler")
    loaded_paths = []
    monkeypatch.setattr(
        sampler_module, "load_manifest",
        lambda path: (
            loaded_paths.append(path)
            or {"sequences": {"sequence-val": {}}}
        ),
    )

    class Dataset:
        def __len__(self):
            return 1

    cfg = SimpleNamespace(
        TRAIN=SimpleNamespace(
            EXPERT_PHASE="dispatch",
            SPECIALIST_EXPERT_IDS=[1, 2, 3, 4],
            SPECIALIST_EXPERT_SCHEDULE=[],
        ),
        DATA=SimpleNamespace(
            SRBT=SimpleNamespace(ENABLE=False, ANCHOR_WEIGHTS=None),
            CHALLENGE_SAMPLING=SimpleNamespace(
                ENABLE=True,
                PRECISE=True,
                MANIFEST="train-manifest.json",
                VAL_MANIFEST="val-manifest.json",
            ),
            PURSUIT=SimpleNamespace(
                ENABLE=False,
                TRANSITION_PROBABILITY=0.5,
                REAPPEAR_PROBABILITY=0.25,
            ),
        ),
    )

    sampler = sampler_module.TrackingSampler(
        datasets=[Dataset()], p_datasets=None, samples_per_epoch=100,
        max_gap=10, num_search_frames=1, num_template_frames=1,
        cfg=cfg, training=False)

    assert sampler.precise_expert_sampling is True
    assert sampler.training_expert_ids == (0, 1, 2, 3, 4)
    assert loaded_paths == ["val-manifest.json"]


def test_single_specialist_stage_samples_only_precision_expert():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler.samples_per_epoch = 24
    sampler.training_expert_ids = (2,)
    sampler.specialist_stage_schedule = ()

    assert {
        sampler.training_expert_for_index(index) for index in range(24)
    } == {2}


def test_epoch_schedule_activates_only_declared_specialist_blocks():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler.samples_per_epoch = 24
    sampler.training_expert_ids = (1, 2, 3, 4)
    sampler.specialist_stage_schedule = (
        (1, 40, (1, 3)),
        (41, 80, (4,)),
        (81, 100, (2,)),
        (101, 130, (4, 2)),
        (131, 160, (1, 3, 4, 2)),
    )

    for epoch, expected in (
        (1, {1, 3}),
        (41, {4}),
        (81, {2}),
        (101, {2, 4}),
        (131, {1, 2, 3, 4}),
    ):
        sampler.set_epoch(epoch)
        expert_ids = {
            sampler.training_expert_for_index(index) for index in range(24)
        }
        assert expert_ids == expected


def test_specialize_sampler_uses_only_exclusive_specialist_frames():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler.expert_phase = "specialize"
    sampler._expert_manifest = {
        "sequence-a": {
            "attributes": {
                "small_target": [False, False, True, True, True, True, False, False, False],
                "motion": [False, True, False, True, False, False, False, True, False],
                "low_light": [True, False, False, False, True, False, False, False, False],
                "recovery": [False, False, False, False, False, True, False, False, False],
                "ambiguity": [False, False, False, False, False, False, True, False, False],
                "deformation": [False, False, False, False, False, False, False, True, False],
                "absent": [False, False, False, False, False, False, False, False, True],
            },
        },
    }

    class Dataset:
        sequence_list = ["sequence-a"]

    info = {"bbox": torch.ones(9, 4)}
    assert sampler._expert_candidate_groups(
        Dataset(), 0, info, training_expert_id=1) == {
            "expert": [1]}
    assert sampler._expert_candidate_groups(
        Dataset(), 0, info, training_expert_id=2) == {
            "expert": [2]}
    assert sampler._expert_candidate_groups(
        Dataset(), 0, info, training_expert_id=4) == {
            "expert": [0, 6]}

    sampler.num_template_frames = 1
    sampler.num_search_frames = 1
    sampler.max_gap = 10
    sampled = sampler._sample_expert_causal_frame_ids(
        Dataset(), 0, torch.ones(9, dtype=torch.uint8), info,
        training_expert_id=1)
    assert sampled[3] == 1
    assert sampled[2] == "expert_1"
    assert sampled[1] == [1]
    assert sampled[4].shape == (7,)


def test_visibility_expert_balances_absent_reappear_and_recovery_candidates():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler._expert_manifest = {
        "sequence-a": {
            "attributes": {
                "small_target": [False] * 9,
                "motion": [False] * 9,
                "low_light": [False] * 9,
                "recovery": [False, False, False, False, True, True, True, False, False],
                "ambiguity": [False] * 9,
                "deformation": [False] * 9,
                "absent": [False, False, True, True, False, False, False, True, True],
            },
        },
    }
    class Dataset:
        sequence_list = ["sequence-a"]

    info = {
        "bbox": torch.ones(9, 4),
        "absent": torch.tensor([1, 1, 0, 0, 1, 1, 1, 0, 0]),
    }

    groups = sampler._expert_candidate_groups(
        Dataset(), 0, info, training_expert_id=3)

    assert groups == {
        "absent": [2, 3, 7, 8],
        "reappear": [4],
        "recovery": [5, 6],
    }


def test_sampler_rejects_manifest_without_challenge_attributes():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler._expert_manifest = {"sequence-a": {"owners": [1, 2]}}

    class Dataset:
        sequence_list = ["sequence-a"]

    info = {"bbox": torch.ones(2, 4)}
    with pytest.raises(RuntimeError, match="invalid attributes"):
        sampler._expert_candidate_groups(
            Dataset(), 0, info, training_expert_id=1)


def test_precise_epoch_size_covers_complete_owner_batches():
    base_functions = importlib.import_module("lib.train.base_functions")

    base_functions._validate_precise_expert_epoch(4200, 6, 2, 1)
    with pytest.raises(ValueError, match="1 \\* BATCH_SIZE \\* world_size"):
        base_functions._validate_precise_expert_epoch(4096, 6, 2, 1)
