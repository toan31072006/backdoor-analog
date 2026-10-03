#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: bash remote_linux/03_run_main.sh [common options]

Runs the paper main matrix: Clean, Proposed, and TSBA x seeds 42/0/1 on each
selected dataset. A rerun uses compatible checkpoints and cached completed cells.
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

run_main_dataset() {
    local dataset_arg="$1"
    local dataset_root="$2"
    local pivot="$3"
    local label="$4"
    local base=(
        run_experiments.py
        --dataset "$dataset_arg"
        --seeds 42 0 1
        --dataset-root "$dataset_root"
        --action-npy "$ACTION_NPY"
        --pivot "$pivot"
        --device "$DEVICE"
        --parallel "$PARALLEL"
        --num-workers "$NUM_WORKERS"
    )

    # Proposed + TSBA x 3 seeds = 6 attacked training cells.
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
