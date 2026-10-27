#!/usr/bin/env bash
# PiW3D HPE-Li / seed42 Smooth-LF-CSI + TSBA adapted, two RTX 5090 GPUs.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
PYTHON_BIN=${PIW_HPELI_PYTHON:-/venv/main/bin/python}
DATA_HOME=${PIW_HPELI_DATA_HOME:-/workspace/piw3d}
DATASET_ROOT=${PIW_HPELI_DATASET_ROOT:-$DATA_HOME/datasets/Person-in-WiFi-3D_processed_v1}
ACTION_NPY=${PIW_HPELI_ACTION_NPY:-$DATA_HOME/actions/data_bend.npy}
REFERENCE_ARCHIVE=${PIW_HPELI_REFERENCE_ARCHIVE:-$DATA_HOME/references/piw_hpeli_v3_clean_reference.zip}
OUTDIR=${PIW_HPELI_OUTDIR:-/workspace/runs/piw3d_hpeli_two_extra_s42_v1}
SESSION=${PIW_HPELI_SESSION:-piw3d-hpeli-two-extra-s42}
SOCKET=${PIW_HPELI_TMUX_SOCKET:-piw3d_hpeli_extra}

usage() {
    cat <<'EOF'
Usage: bash remote_linux/piw3d_hpeli_extra/start.sh MODE
MODE: --dry-run | --prepare-only | --check-only | --launch | --status | --unlock-stale

Prepare performs full PiW input/reference audit and simultaneous physical-B32
GPU smoke for two independent HPE-Li cells; it does not launch full training.
Launch is a separate user action and refuses GPUs occupied by other compute
processes. It runs detached in tmux; no automatic retry or batch fallback.

Optional env: PIW_HPELI_PYTHON, PIW_HPELI_DATA_HOME,
PIW_HPELI_DATASET_ROOT, PIW_HPELI_ACTION_NPY,
PIW_HPELI_REFERENCE_ARCHIVE, PIW_HPELI_OUTDIR,
PIW_HPELI_SESSION, PIW_HPELI_TMUX_SOCKET.
EOF
}

if [[ $# -ne 1 ]]; then usage >&2; exit 2; fi
MODE=$1
case "$MODE" in
    --dry-run|--prepare-only|--check-only|--launch|--status|--unlock-stale) ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing PiW Python interpreter: $PYTHON_BIN" >&2
    exit 1
fi
ARGS=("$HERE/controller.py" --repo-dir "$REPO" --data-home "$DATA_HOME"
      --dataset-root "$DATASET_ROOT" --action-npy "$ACTION_NPY"
      --reference-archive "$REFERENCE_ARCHIVE" --outdir "$OUTDIR")

if [[ "$MODE" == --launch ]]; then
    "$PYTHON_BIN" -u "${ARGS[@]}" --check-only
    if ! command -v tmux >/dev/null 2>&1; then
        echo 'tmux is required for detached launch' >&2
        exit 1
    fi
    if tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
        echo "PiW HPE-Li tmux session already exists: $SESSION" >&2
        exit 1
    fi
    if [[ -f "$OUTDIR/controller_complete.json" ]]; then
        echo 'This PiW HPE-Li run is already complete; inspect results instead.' >&2
        exit 1
    fi
    if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
        echo 'CUDA compute processes are active; wait for the current Vast run to finish.' >&2
        exit 1
    fi
    mkdir -p -- "$OUTDIR"
    printf -v QUOTED '%q ' "$PYTHON_BIN" -u "${ARGS[@]}" --launch
    printf -v QUOTED_LOG '%q' "$OUTDIR/controller.log"
    tmux -L "$SOCKET" new-session -d -s "$SESSION" \
        "exec $QUOTED >> $QUOTED_LOG 2>&1"
    echo "STARTED tmux socket=$SOCKET session=$SESSION"
    echo "Controller: $OUTDIR/controller.log"
    echo "Cells: $OUTDIR/logs/{train,evaluate}/*.log"
    echo 'Running is not proof of progress; inspect the logs and checkpoint epochs.'
else
    "$PYTHON_BIN" -u "${ARGS[@]}" "$MODE"
fi
