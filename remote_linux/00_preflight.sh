#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: bash remote_linux/00_preflight.sh [common options]

Checks dependencies, executes a synchronized CUDA probe, validates both full
datasets, and validates bend/cross/nod action files when present.
EOF
    usage_common
}

parse_common_args "$@"
finalize_common
require_python

REPORT_ROOT="${RUNS_ROOT}/preflight"
mkdir -p -- "$REPORT_ROOT"

# Preserve the software/driver context alongside the scientific data report.
"$PYTHON_BIN" -m pip freeze > "${REPORT_ROOT}/pip_freeze.txt"
uname -a > "${REPORT_ROOT}/uname.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${REPORT_ROOT}/nvidia_smi.txt"
fi
if command -v git >/dev/null 2>&1 && git -C "$CODE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$CODE_ROOT" rev-parse HEAD > "${REPORT_ROOT}/git_commit.txt"
    git -C "$CODE_ROOT" status --short > "${REPORT_ROOT}/git_status.txt"
fi

note "Checking Python, dependencies, CUDA, and free storage"
run_python "${SCRIPT_DIR}/preflight_runtime.py" \
    --device "$DEVICE" \
    --data-home "$DATA_HOME" \
    --json-out "${REPORT_ROOT}/runtime_report.json"

ACTION_ARGS=()
if resolve_action; then
    ACTION_ARGS+=(--trigger "$ACTION_NPY")
fi
if CROSS_ACTION="$(resolve_optional_action cross)"; then
    ACTION_ARGS+=(--cross-trigger "$CROSS_ACTION")
fi
if NOD_ACTION="$(resolve_optional_action nod)"; then
    ACTION_ARGS+=(--nod-trigger "$NOD_ACTION")
fi

note "Checking MM-Fi, PiW3D, and action assets"
run_python "${SCRIPT_DIR}/preflight_data.py" \
    --mmfi-root "$MMFI_ROOT" \
    --piw-root "$PIW_ROOT" \
    --json-out "${REPORT_ROOT}/data_report.json" \
    "${ACTION_ARGS[@]}"

print_resolved_paths
df -h "$DATA_HOME"
df -i "$DATA_HOME"
[[ -n "${ACTION_NPY:-}" ]] || die "datasets passed, but bend action is missing under ${DATA_HOME}/actions"

note "Preflight passed. Reports: ${REPORT_ROOT}"
