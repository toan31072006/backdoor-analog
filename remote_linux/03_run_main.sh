#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: bash remote_linux/03_run_main.sh [common options]

Runs full-epoch Clean, Proposed, and TSBA training for each selected seed and
dataset. Seeds default to the paper set 42/0/1. A rerun uses compatible
checkpoints and cached completed cells.
EOF
    usage_common
}

parse_common_args "$@"
finalize_common
require_python
require_selected_datasets
require_action

if (( PARALLEL > 1 )); then
    printf 'WARNING: --parallel %s creates multiple training processes and may exhaust VRAM.\n' "$PARALLEL" >&2
fi

MAIN_ROOT="${RUNS_ROOT}/paper_main"
mkdir -p -- "$MAIN_ROOT"

note "Full run seeds: ${SEEDS[*]} (worker timeout: ${WORKER_TIMEOUT}s; 0=disabled)"

run_main_dataset() {
    local dataset_arg="$1"
    local dataset_root="$2"
    local pivot="$3"
    local label="$4"
    local base=(
        run_experiments.py
        --dataset "$dataset_arg"
        --seeds "${SEEDS[@]}"
        --dataset-root "$dataset_root"
        --action-npy "$ACTION_NPY"
        --pivot "$pivot"
        --device "$DEVICE"
        --parallel "$PARALLEL"
        --num-workers "$NUM_WORKERS"
        --worker-timeout "$WORKER_TIMEOUT"
    )

    # Proposed + TSBA for every requested seed.
    run_atk_python "${base[@]}" \
        --triggers micro_dropper tsba \
        --outdir "${MAIN_ROOT}/${label}_attacks"

    # Paired clean controls use the identical recipe with rho=0.
    run_atk_python "${base[@]}" \
        --triggers micro_dropper \
        --rho 0 \
        --outdir "${MAIN_ROOT}/${label}_clean"
}

if [[ "$DATASET_SELECTION" == "mmfi" || "$DATASET_SELECTION" == "both" ]]; then
    run_main_dataset mmfi "$MMFI_ROOT" 1 mmfi
fi
if [[ "$DATASET_SELECTION" == "piw3d" || "$DATASET_SELECTION" == "both" ]]; then
    run_main_dataset pwif3d "$PIW_ROOT" "$PIW_PIVOT" piw3d
fi

note "Main experiment matrix completed. Outputs: ${MAIN_ROOT}"
