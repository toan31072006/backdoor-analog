#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: bash remote_linux/04_status.sh [common options]

Read-only status summary for GPU, running trainers, outputs, and disk space.
EOF
    usage_common
}

parse_common_args "$@"
finalize_common

note "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    printf 'nvidia-smi is not installed or not on PATH.\n'
fi

note "Training processes"
if ! pgrep -af 'run_experiments.py|train_backdoor.py'; then
    printf 'No matching Python training process found.\n'
fi

note "Results and checkpoints"
if [[ -d "$RUNS_ROOT" ]]; then
    printf 'Completed result.json files: '
    find "$RUNS_ROOT" -type f -name result.json -print | wc -l
    printf 'Checkpoint files: '
    find "$RUNS_ROOT" -type f -name '*.pt' -print | wc -l
    du -sh "$RUNS_ROOT"
else
    printf 'Runs directory does not exist yet: %s\n' "$RUNS_ROOT"
fi

note "Storage"
df -h "$DATA_HOME"
