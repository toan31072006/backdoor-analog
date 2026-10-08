#!/usr/bin/env bash
# New full seed-42 matrix; no TSBA and no PiW3D. Pass --fresh and a NEW outdir.
set -euo pipefail
DOSE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOSE_CODE_ROOT="$(cd -- "${DOSE_SCRIPT_DIR}/.." && pwd)"
DOSE_DATA_HOME="${DOSE_DATA_HOME:-$(cd -- "${DOSE_CODE_ROOT}/.." && pwd)}"
DOSE_PY="${DOSE_PY:-${DOSE_DATA_HOME}/envs/dose-backdoor/bin/python}"
if [[ ! -x "${DOSE_PY}" ]]; then
    printf 'Missing Python: %s\nSet DOSE_PY to the experiment environment.\n' "${DOSE_PY}" >&2
    exit 1
fi
exec "${DOSE_PY}" -u "${DOSE_CODE_ROOT}/ATKBackd/run_mmfi_tables.py" \
    --data-home "${DOSE_DATA_HOME}" "$@"
