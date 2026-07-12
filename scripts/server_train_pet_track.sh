#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SAVE_DIR="${SAVE_DIR:-/root/fnvme/PTE_v8_runs/srbt_main_20260712}"
CONFIG_NAME="${CONFIG_NAME:-felt_pet_track}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/attrack/bin/python}"
TRAIN_MODE="${TRAIN_MODE:-auto}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"

cd "$PROJECT_DIR"

export FELT_DATA_ROOT="${FELT_DATA_ROOT:-/root/fnvme/FELT}"
export FELT_TRAIN_ROOT="${FELT_TRAIN_ROOT:-$FELT_DATA_ROOT/train}"
export FELT_VAL_ROOT="${FELT_VAL_ROOT:-$FELT_DATA_ROOT/train}"
export AMTTRACK_WORKSPACE_DIR="${AMTTRACK_WORKSPACE_DIR:-$PROJECT_DIR/workspace}"

mkdir -p "$SAVE_DIR" "$AMTTRACK_WORKSPACE_DIR"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "FELT_DATA_ROOT=$FELT_DATA_ROOT"
echo "FELT_TRAIN_ROOT=$FELT_TRAIN_ROOT"
echo "FELT_VAL_ROOT=$FELT_VAL_ROOT"
echo "SAVE_DIR=$SAVE_DIR"
echo "CONFIG_NAME=$CONFIG_NAME"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime not found: $PYTHON_BIN" >&2
  exit 1
fi
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
GPU_COUNT="$($PYTHON_BIN - <<'PY_INNER'
try:
    import torch
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY_INNER
)"
if [[ "$TRAIN_MODE" == "auto" ]]; then
  if [[ "$GPU_COUNT" -gt 1 ]]; then
    TRAIN_MODE="multiple"
  else
    TRAIN_MODE="single"
  fi
fi
if [[ -z "$NPROC_PER_NODE" ]]; then
  NPROC_PER_NODE="$GPU_COUNT"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "PYTHON_BIN=$PYTHON_BIN"
echo "GPU_COUNT=$GPU_COUNT"
echo "TRAIN_MODE=$TRAIN_MODE"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"

if [[ "$TRAIN_MODE" == "multiple" ]]; then
  "$PYTHON_BIN" tracking/train.py \
    --script pet_track \
    --config "$CONFIG_NAME" \
    --save_dir "$SAVE_DIR" \
    --mode multiple \
    --nproc_per_node "$NPROC_PER_NODE" \
    --use_wandb 0
else
  "$PYTHON_BIN" tracking/train.py \
    --script pet_track \
    --config "$CONFIG_NAME" \
    --save_dir "$SAVE_DIR" \
    --mode single \
    --use_wandb 0
fi
