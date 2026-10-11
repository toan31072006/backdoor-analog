#!/usr/bin/env bash
# Six MM-Fi adapters; original DT-Pose HPELi training settings.
# Run on MICA locally. No installation, SSH, uploads or deletion.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DOSE_DATA_HOME:-$HOME/backdooranalog}"
PYTHON_BIN="${DOSE_PYTHON:-$DATA_ROOT/envs/dose-backdoor/bin/python}"
RESULT_ROOT="${DOSE_OUT:-$DATA_ROOT/runs/mmfi_baselines_dtpose_s42_v1}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python not found: %s\nSet DOSE_PYTHON to your existing MICA environment.\n' "$PYTHON_BIN" >&2
  exit 1
fi
# Logical device IDs follow CUDA_VISIBLE_DEVICES. Select physical GPUs outside.
exec "$PYTHON_BIN" -u "$CODE_ROOT/run_baselines.py" \
  --data-home "$DATA_ROOT" --outdir "$RESULT_ROOT" \
  --devices cuda:0 cuda:1 cuda:2 --num-workers 4 "$@"
