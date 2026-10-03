#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: bash remote_linux/02_smoke_test.sh [common options]

Runs the source-level unit/contract tests. It does not train a model.
EOF
    usage_common
}

parse_common_args "$@"
finalize_common
require_python
mkdir -p -- "${RUNS_ROOT}/test_tmp"

TEST_ROOT="${RUNS_ROOT}/test_tmp/pytest_$$_$(date +%Y%m%d_%H%M%S)"
run_atk_python -m pytest -q tests -p no:cacheprovider --basetemp "$TEST_ROOT"

note "Code smoke tests passed. No model was trained."
