#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: bash remote_linux/01_dry_run.sh [common options]

Resolves and validates the exact paper experiment matrix without training.
EOF
    usage_common
}

parse_common_args "$@"
finalize_common
require_python
require_selected_datasets
require_action
mkdir -p -- "${RUNS_ROOT}/contract"

if [[ "$DATASET_SELECTION" == "mmfi" || "$DATASET_SELECTION" == "both" ]]; then
    run_atk_python run_experiments.py \
        --dataset mmfi \
        --triggers micro_dropper tsba \
        --seeds 42 0 1 \
        --dataset-root "$MMFI_ROOT" \
        --action-npy "$ACTION_NPY" \
        --pivot 1 \
        --device "$DEVICE" \
        --parallel 1 \
        --num-workers "$NUM_WORKERS" \
        --outdir "${RUNS_ROOT}/contract/mmfi_attacks" \
        --dry-run

    run_atk_python run_experiments.py \
        --dataset mmfi \
        --triggers micro_dropper \
        --seeds 42 0 1 \
        --dataset-root "$MMFI_ROOT" \
        --action-npy "$ACTION_NPY" \
        --pivot 1 \
        --rho 0 \
        --device "$DEVICE" \
        --parallel 1 \
        --num-workers "$NUM_WORKERS" \
        --outdir "${RUNS_ROOT}/contract/mmfi_clean" \
        --dry-run

    run_python "${SCRIPT_DIR}/verify_contract.py" \
        --dataset mmfi \
        --attacks "${RUNS_ROOT}/contract/mmfi_attacks/experiment_matrix.resolved.json" \
        --clean "${RUNS_ROOT}/contract/mmfi_clean/experiment_matrix.resolved.json" \
        --dataset-root "$MMFI_ROOT" \
        --action-npy "$ACTION_NPY" \
        --pivot 1
fi

if [[ "$DATASET_SELECTION" == "piw3d" || "$DATASET_SELECTION" == "both" ]]; then
    run_atk_python run_experiments.py \
        --dataset pwif3d \
        --triggers micro_dropper tsba \
        --seeds 42 0 1 \
        --dataset-root "$PIW_ROOT" \
        --action-npy "$ACTION_NPY" \
        --pivot "$PIW_PIVOT" \
        --device "$DEVICE" \
        --parallel 1 \
        --num-workers "$NUM_WORKERS" \
        --outdir "${RUNS_ROOT}/contract/piw3d_attacks" \
        --dry-run


    run_atk_python run_experiments.py \
        --dataset pwif3d \
        --triggers micro_dropper \
        --seeds 42 0 1 \
        --dataset-root "$PIW_ROOT" \
        --action-npy "$ACTION_NPY" \
        --pivot "$PIW_PIVOT" \
        --rho 0 \
        --device "$DEVICE" \
        --parallel 1 \
        --num-workers "$NUM_WORKERS" \
        --outdir "${RUNS_ROOT}/contract/piw3d_clean" \
        --dry-run

    run_python "${SCRIPT_DIR}/verify_contract.py" \
        --dataset piw3d \
        --attacks "${RUNS_ROOT}/contract/piw3d_attacks/experiment_matrix.resolved.json" \
        --clean "${RUNS_ROOT}/contract/piw3d_clean/experiment_matrix.resolved.json" \
        --dataset-root "$PIW_ROOT" \
        --action-npy "$ACTION_NPY" \
        --pivot "$PIW_PIVOT"
fi

note "Dry-run contract passed. Matrices: ${RUNS_ROOT}/contract"
