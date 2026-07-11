"""Generate the single SRBT FELT config.

The old stage1/router/C3 chain was removed in M7. This helper now exists only
for scripts that expect a generated config path.
"""
import argparse
from pathlib import Path
import shutil


def generate(output_dir, source="experiments/pet_track/felt_pet_track.yaml"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "felt_pet_track_srbt.yaml"
    shutil.copyfile(source, target)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments/pet_track/generated")
    parser.add_argument("--source", default="experiments/pet_track/felt_pet_track.yaml")
    args = parser.parse_args()
    print(generate(args.output_dir, args.source))


if __name__ == "__main__":
    main()
