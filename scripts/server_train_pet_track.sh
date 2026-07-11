#!/usr/bin/env bash
set -euo pipefail

PROJECTS_ROOT="${PROJECTS_ROOT:-/root/projects}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SAVE_DIR="${SAVE_DIR:-/root/fnvme/PETTrack_runs/v8_annotation_free_20260711}"
CONFIG_NAME="${CONFIG_NAME:-felt_pet_track}"
CONDA_ENV="${CONDA_ENV:-study_nlp}"
PYTHON_BIN="${PYTHON_BIN:-}"
TRAIN_MODE="${TRAIN_MODE:-auto}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"

cd "$PROJECT_DIR"

if [[ -d "$PROJECTS_ROOT/FELT" ]]; then
  export FELT_DATA_ROOT="$PROJECTS_ROOT/FELT"
elif [[ -d "$PROJECTS_ROOT/felt" ]]; then
  export FELT_DATA_ROOT="$PROJECTS_ROOT/felt"
else
  export FELT_DATA_ROOT="$PROJECTS_ROOT"
fi
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

if [[ -n "$PYTHON_BIN" ]]; then
  export PATH="$(dirname "$PYTHON_BIN"):$PATH"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
elif [[ -x "/opt/miniconda3/envs/$CONDA_ENV/bin/python" ]]; then
  PYTHON_BIN="/opt/miniconda3/envs/$CONDA_ENV/bin/python"
  export PATH="$(dirname "$PYTHON_BIN"):$PATH"
elif [[ -x "/root/venvs/attrack/bin/python" ]]; then
  # This server image already provides the required torch/timm/cv2 stack here.
  source /root/venvs/attrack/bin/activate
elif [[ -x "/root/venvs/amttrack_baseline/bin/python" ]]; then
  source /root/venvs/amttrack_baseline/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
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
