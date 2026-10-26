#!/usr/bin/env bash
# PiW3D MetaFi++/WiFlow, one Vast host with exactly two CUDA GPUs.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
PYTHON_BIN=${PIW_PYTHON:-$(command -v python || true)}
DATA_HOME=${PIW_DATA_HOME:-/workspace/piw3d}
OUTDIR=${PIW_OUTDIR:-/workspace/runs/piw3d_two_models_s42_3pergpu_v2_smokefix}
DATASET_ROOT=${PIW_DATASET_ROOT:-$DATA_HOME/datasets/Person-in-WiFi-3D_processed_v1}
ACTION_NPY=${PIW_ACTION_NPY:-$DATA_HOME/actions/data_bend.npy}
SESSION=${PIW_TMUX_SESSION:-piw3d-two-models-s42-3pergpu-v2-smokefix}

usage() {
    cat <<'EOF'
Usage: bash remote_linux/piw3d_two_models/start.sh MODE
MODE: --dry-run | --prepare-only | --launch | --status
Environment: PIW_PYTHON, PIW_DATA_HOME, PIW_DATASET_ROOT,
             PIW_ACTION_NPY, PIW_OUTDIR, PIW_TMUX_SESSION.
Prepare audits full PiW data and smokes up to 6 processes (3/GPU), but does not train.
Launch requires that gate and starts the controller in a detached tmux session.
EOF
}

if [[ $# -ne 1 ]]; then usage >&2; exit 2; fi
MODE=$1
case "$MODE" in
    --dry-run|--prepare-only|--launch|--status) ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing PiW Python interpreter: $PYTHON_BIN" >&2
    exit 1
fi
if [[ -f "$REPO/BUNDLE_MANIFEST.json" ]]; then
    "$PYTHON_BIN" "$HERE/verify_bundle.py" --root "$REPO"
else
    echo "No BUNDLE_MANIFEST.json in checkout; source integrity is bound by controller plan" >&2
fi

ARGS=("$HERE/controller.py" --repo-dir "$REPO" --data-home "$DATA_HOME"
      --dataset-root "$DATASET_ROOT" --action-npy "$ACTION_NPY" --outdir "$OUTDIR")
if [[ "$MODE" == --launch ]]; then
    "$PYTHON_BIN" -u "${ARGS[@]}" --check-only
    if ! command -v tmux >/dev/null 2>&1; then
        echo 'tmux is required for detached launch' >&2
        exit 1
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session already exists: $SESSION" >&2
        exit 1
    fi
    LOG="$OUTDIR/controller.log"
    printf -v QUOTED '%q ' "$PYTHON_BIN" -u "${ARGS[@]}" --launch
    printf -v QUOTED_LOG '%q' "$LOG"
    tmux new-session -d -s "$SESSION" "exec $QUOTED >> $QUOTED_LOG 2>&1"
    echo "STARTED tmux $SESSION; training waves up to 3/GPU, then evaluation waves"
    echo "Monitor: tail -F $LOG"
    echo "Cells: $OUTDIR/logs/{train,evaluate}/*.log"
    echo 'Running is not proof of completion. No automatic retry or batch fallback.'
else
    "$PYTHON_BIN" -u "${ARGS[@]}" "$MODE"
fi
