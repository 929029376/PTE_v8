#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
RUN_DIR="${RUN_DIR:-/root/fnvme/PETTrack_runs/v8_annotation_free_20260711}"
CONFIG_NAME="${CONFIG_NAME:-generated/felt_pet_track_v8_stage1_expert}"
CONFIG_LOG_NAME="${CONFIG_NAME//\//__}"
LOG="${LOG:-$RUN_DIR/logs/watch_training.log}"
TRAIN_LOG="$RUN_DIR/logs/pet_track-$CONFIG_LOG_NAME.log"
CHECKPOINT_DIR="$RUN_DIR/checkpoints/train/pet_track/$CONFIG_NAME"
INTERVAL="${1:-300}"

mkdir -p "$(dirname "$LOG")"
cd "$PROJECT_DIR"

echo "==== PETTrack watcher started at $(date -Iseconds) interval=${INTERVAL}s pid=$$ ====" >> "$LOG"
while true; do
  {
    echo "---- $(date -Iseconds) ----"
    if pgrep -af -- "--config $CONFIG_NAME"; then
      echo "TRAIN_STATUS=RUNNING"
    else
      echo "TRAIN_STATUS=STOPPED"
    fi
    echo "GPU_STATUS"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || true
    echo "LATEST_TRAIN_LOG"
    grep "\[train:" "$TRAIN_LOG" 2>/dev/null | tail -n 2 || true
    echo "VAL_LOG_CHECK"
    grep "\[val:" "$TRAIN_LOG" 2>/dev/null | tail -n 2 || true
    echo "ERRORS"
    grep -iE "out of memory|nan|traceback|runtimeerror|error" "$TRAIN_LOG" 2>/dev/null | tail -n 5 || true
    echo "CHECKPOINTS"
    ls -lh "$CHECKPOINT_DIR" 2>/dev/null || true
    echo
  } >> "$LOG"
  sleep "$INTERVAL"
done
