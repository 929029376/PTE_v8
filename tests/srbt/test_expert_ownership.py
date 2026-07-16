import importlib
import json
import subprocess
import sys
from pathlib import Path

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


def test_observation_scores_detect_aps_distractor_and_ignore_future_frames():
    ownership = importlib.import_module("lib.train.data.expert_ownership")
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


def test_manifest_round_trip_records_schema_hash_and_removes_temporary_file(tmp_path):
    ownership = importlib.import_module("lib.train.data.expert_ownership")
    path = tmp_path / "owners.json"
    thresholds = {"ambiguity_threshold": 0.8, "recovery_window": 8}
    records = {
        "sequence-a": {
            "owners": [0, 4],
            "reasons": ["generalist", "visibility"],
        }
    }

    ownership.write_manifest(path, records, thresholds)
    loaded = ownership.load_manifest(path)

    assert loaded["schema_version"] == 1
    assert loaded["thresholds"] == thresholds
    assert loaded["sequences"] == records
    assert len(loaded["config_sha256"]) == 64
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8")) == loaded


def test_sequence_record_is_built_one_frame_at_a_time():
    ownership = importlib.import_module("lib.train.data.expert_ownership")

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
    assert record["owners"] == [0, 3]
    assert len(record["reasons"]) == 2


def test_manifest_builder_cli_documents_required_paths():
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "tracking/build_expert_ownership.py", "--help"],
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

    output = tmp_path / "owners.json"
    result = subprocess.run(
        [
            sys.executable,
            "tracking/build_expert_ownership.py",
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
    assert all(len(record["owners"]) == 2
               for record in manifest["sequences"].values())


def test_specialist_epoch_is_equal_owner_blocks_for_each_ddp_rank():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler.samples_per_epoch = 100

    owners = [sampler.owner_for_index(index) for index in range(100)]
    assert [owners.count(owner) for owner in range(5)] == [20] * 5
    for rank in range(2):
        rank_owners = owners[rank::2]
        assert all(
            len(set(rank_owners[start:start + 2])) == 1
            for start in range(0, len(rank_owners), 2)
        )


def test_sampler_emits_expert_owner_not_challenge_label():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler.train_cls = False
    sampler.precise_expert_sampling = True
    sampler.samples_per_epoch = 10
    sampler.getitem = lambda owner_id=None: {
        "expert_owner_id": torch.tensor(owner_id, dtype=torch.long),
    }

    sample = sampler[4]

    assert sample["expert_owner_id"].item() == 2
    assert "challenge_id" not in sample


def test_owner_sampling_never_substitutes_another_expert():
    sampler_module = importlib.import_module("lib.train.data.sampler")
    sampler = object.__new__(sampler_module.TrackingSampler)
    sampler._expert_manifest = {
        "sequence-a": {"owners": [0, 1, 1, 2]},
    }
    sampler.num_template_frames = 1
    sampler.num_search_frames = 1
    sampler.max_gap = 10

    class Dataset:
        sequence_list = ["sequence-a"]

    info = {"bbox": torch.ones(4, 4)}
    visible = torch.ones(4, dtype=torch.uint8)
    sampled = sampler._sample_expert_causal_frame_ids(
        Dataset(), 0, visible, info, owner_id=1)
    missing = sampler._sample_expert_causal_frame_ids(
        Dataset(), 0, visible, info, owner_id=4)

    assert sampled[3] == 1
    assert sampled[2] == "expert_1"
    assert sampled[1][0] in (1, 2)
    assert missing == (None, None, None, None)


def test_precise_epoch_size_covers_complete_owner_batches():
    base_functions = importlib.import_module("lib.train.base_functions")

    base_functions._validate_precise_expert_epoch(4200, 6, 2)
    with pytest.raises(ValueError, match="5 \\* BATCH_SIZE \\* world_size"):
        base_functions._validate_precise_expert_epoch(4096, 6, 2)
