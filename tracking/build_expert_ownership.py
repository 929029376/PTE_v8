import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _init_paths  # noqa: F401

from lib.train.data.expert_ownership import build_sequence_record, write_manifest
from lib.train.data.felt_challenges import (
    DEFAULT_MOTION_NORM,
    DEFAULT_RECOVERY_WINDOW,
    DEFAULT_SMALL_AREA_RATIO,
)
from lib.train.data.image_loader import opencv_loader
from lib.train.dataset.felt import Felt


_WORKER_DATASET = None
_WORKER_THRESHOLDS = None


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("workers must be at least 1")
    return value


def _initialize_worker(data_root, split, thresholds):
    global _WORKER_DATASET, _WORKER_THRESHOLDS
    _WORKER_DATASET = Felt(
        root=str(Path(data_root).expanduser()),
        image_loader=opencv_loader,
        split=split,
    )
    _WORKER_THRESHOLDS = thresholds


def _build_worker_record(seq_id):
    sequence_name = _WORKER_DATASET.sequence_list[seq_id]
    record = build_sequence_record(
        _WORKER_DATASET, seq_id, _WORKER_THRESHOLDS)
    return sequence_name, record


def _iter_sequence_records(dataset, data_root, split, thresholds, workers):
    if workers == 1:
        for seq_id, sequence_name in enumerate(dataset.sequence_list):
            yield sequence_name, build_sequence_record(
                dataset, seq_id, thresholds)
        return

    with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(data_root, split, thresholds)) as executor:
        yield from executor.map(
            _build_worker_record,
            range(len(dataset.sequence_list)),
            chunksize=1,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build deterministic frame-level FELT expert ownership")
    parser.add_argument(
        "--data-root", required=True,
        help="FELT root containing list1k.txt and sequence directories")
    parser.add_argument(
        "--split", required=True, choices=("train", "val", "test"),
        help="Official FELT split used to select sequences")
    parser.add_argument(
        "--output", required=True,
        help="Destination JSON manifest path")
    parser.add_argument(
        "--workers", type=_positive_int, default=1,
        help="Sequence workers; ownership remains deterministic")
    parser.add_argument("--small-area-ratio", type=float,
                        default=DEFAULT_SMALL_AREA_RATIO)
    parser.add_argument("--motion-norm", type=float,
                        default=DEFAULT_MOTION_NORM)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.8)
    parser.add_argument("--recovery-window", type=int,
                        default=DEFAULT_RECOVERY_WINDOW)
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = {
        "small_area_ratio": args.small_area_ratio,
        "motion_norm": args.motion_norm,
        "ambiguity_threshold": args.ambiguity_threshold,
        "recovery_window": args.recovery_window,
    }
    dataset = Felt(
        root=str(Path(args.data_root).expanduser()),
        image_loader=opencv_loader,
        split=args.split,
    )
    sequences = {}
    owner_counts = Counter()
    records = _iter_sequence_records(
        dataset,
        args.data_root,
        args.split,
        thresholds,
        args.workers,
    )
    for seq_id, (sequence_name, record) in enumerate(records):
        sequences[sequence_name] = record
        owner_counts.update(record["owners"])
        print(
            f"[{seq_id + 1}/{len(dataset.sequence_list)}] "
            f"{sequence_name}: {len(record['owners'])} frames",
            flush=True,
        )

    write_manifest(args.output, sequences, thresholds)
    counts = " ".join(
        f"expert_{owner}={owner_counts.get(owner, 0)}" for owner in range(5))
    print(f"Wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
