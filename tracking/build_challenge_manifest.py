import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _init_paths  # noqa: F401

from lib.train.data.challenge_manifest import (
    DEFAULT_LOW_LIGHT_CONTEXT_MEDIAN,
    DEFAULT_LOW_LIGHT_SEARCH_FACTOR,
    build_sequence_record,
    write_manifest,
)
from lib.train.data.felt_challenges import (
    CHALLENGE_NAMES,
    DEFAULT_AMBIGUITY_THRESHOLD,
    DEFAULT_MOTION_NORM,
    DEFAULT_RECOVERY_WINDOW,
    DEFAULT_SMALL_AREA_RATIO,
    expert_supervision_mask,
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
    return _build_record(_WORKER_DATASET, seq_id, _WORKER_THRESHOLDS)


def _build_record(dataset, seq_id, thresholds):
    sequence_name = dataset.sequence_list[seq_id]
    try:
        return sequence_name, build_sequence_record(dataset, seq_id, thresholds), None
    except FileNotFoundError as error:
        return sequence_name, None, str(error)


def _iter_sequence_records(dataset, data_root, split, thresholds, workers):
    if workers == 1:
        for seq_id in range(len(dataset.sequence_list)):
            yield _build_record(dataset, seq_id, thresholds)
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
        description="Build deterministic frame-level FELT challenge labels")
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
        help="Sequence workers; challenge labels remain deterministic")
    parser.add_argument("--small-area-ratio", type=float,
                        default=DEFAULT_SMALL_AREA_RATIO)
    parser.add_argument("--motion-norm", type=float,
                        default=DEFAULT_MOTION_NORM)
    parser.add_argument(
        "--ambiguity-threshold", type=float,
        default=DEFAULT_AMBIGUITY_THRESHOLD)
    parser.add_argument("--recovery-window", type=int,
                        default=DEFAULT_RECOVERY_WINDOW)
    parser.add_argument(
        "--low-light-context-median", type=float,
        default=DEFAULT_LOW_LIGHT_CONTEXT_MEDIAN)
    parser.add_argument(
        "--low-light-search-factor", type=float,
        default=DEFAULT_LOW_LIGHT_SEARCH_FACTOR)
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = {
        "small_area_ratio": args.small_area_ratio,
        "motion_norm": args.motion_norm,
        "ambiguity_threshold": args.ambiguity_threshold,
        "recovery_window": args.recovery_window,
        "low_light_context_median": args.low_light_context_median,
        "low_light_search_factor": args.low_light_search_factor,
    }
    dataset = Felt(
        root=str(Path(args.data_root).expanduser()),
        image_loader=opencv_loader,
        split=args.split,
    )
    sequences = {}
    challenge_counts = {name: 0 for name in CHALLENGE_NAMES}
    expert_counts = {expert_id: 0 for expert_id in range(5)}
    skipped = 0
    records = _iter_sequence_records(
        dataset,
        args.data_root,
        args.split,
        thresholds,
        args.workers,
    )
    for seq_id, (sequence_name, record, error) in enumerate(records):
        if error is not None:
            skipped += 1
            print(f"[skip] {sequence_name}: {error}", flush=True)
            continue
        sequences[sequence_name] = record
        attributes = record["attributes"]
        frame_count = len(attributes[CHALLENGE_NAMES[0]])
        for name in CHALLENGE_NAMES:
            challenge_counts[name] += sum(attributes[name])
        supervision = expert_supervision_mask(attributes)
        for expert_id in range(5):
            expert_counts[expert_id] += int(supervision[:, expert_id].sum())
        print(
            f"[{seq_id + 1}/{len(dataset.sequence_list)}] "
            f"{sequence_name}: {frame_count} frames",
            flush=True,
        )

    write_manifest(args.output, sequences, thresholds)
    challenge_summary = " ".join(
        f"{name}={challenge_counts[name]}" for name in CHALLENGE_NAMES)
    expert_summary = " ".join(
        f"expert_{expert_id}={expert_counts[expert_id]}"
        for expert_id in range(5))
    print(
        f"Wrote {args.output}: {challenge_summary} {expert_summary} "
        f"skipped={skipped}")


if __name__ == "__main__":
    main()
