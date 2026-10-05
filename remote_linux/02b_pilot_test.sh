#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: bash remote_linux/02b_pilot_test.sh [common options]

Runs one full-data seed per dataset: Proposed=2 epochs and clean control=2
epochs. This is a pipeline smoke test, not an effectiveness estimate.
EOF
    usage_common
}

parse_common_args "$@"
finalize_common
require_python
require_selected_datasets
require_action

PILOT_ROOT="${RUNS_ROOT}/pilot"
mkdir -p -- "$PILOT_ROOT"

run_pilot() {
    local dataset_arg="$1"
    local dataset_root="$2"
    local pivot="$3"
    local label="$4"
    local base=(
        run_experiments.py
        --dataset "$dataset_arg"
        --seeds 42
        --dataset-root "$dataset_root"
        --action-npy "$ACTION_NPY"
        --pivot "$pivot"
        --device "$DEVICE"
        --parallel 1
        --num-workers "$NUM_WORKERS"
    )

    run_atk_python "${base[@]}" \
        --triggers micro_dropper \
        --epochs 2 \
        --outdir "${PILOT_ROOT}/${label}_proposed"

    run_atk_python "${base[@]}" \
        --triggers micro_dropper \
        --rho 0 \
        --epochs 2 \
        --outdir "${PILOT_ROOT}/${label}_clean"
}

if [[ "$DATASET_SELECTION" == "mmfi" || "$DATASET_SELECTION" == "both" ]]; then
    run_pilot mmfi "$MMFI_ROOT" 1 mmfi
fi
if [[ "$DATASET_SELECTION" == "piw3d" || "$DATASET_SELECTION" == "both" ]]; then
    run_pilot pwif3d "$PIW_ROOT" "$PIW_PIVOT" piw3d
fi

note "Full-data pilot completed. Outputs: ${PILOT_ROOT}"
