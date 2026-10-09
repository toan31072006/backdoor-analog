#!/usr/bin/env bash
# Isolated method screen only: DRAFT_ONLY_NOT_PAPER_RESULTS.
# Pass --fresh and a NEW --outdir; scientific budgets are immutable on resume.
set -euo pipefail
DOSE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOSE_CODE_ROOT="$(cd -- "${DOSE_SCRIPT_DIR}/.." && pwd)"
DOSE_DATA_HOME="${DOSE_DATA_HOME:-$(cd -- "${DOSE_CODE_ROOT}/.." && pwd)}"
DOSE_PY="${DOSE_PY:-${DOSE_DATA_HOME}/envs/dose-backdoor/bin/python}"
if [[ ! -x "${DOSE_PY}" ]]; then
    printf 'Missing Python: %s\nSet DOSE_PY to the experiment environment.\n' "${DOSE_PY}" >&2
    exit 1
fi
exec "${DOSE_PY}" -u "${DOSE_CODE_ROOT}/ATKBackd/run_method_drafts.py" \
    --data-home "${DOSE_DATA_HOME}" "$@"
