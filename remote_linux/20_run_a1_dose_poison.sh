#!/usr/bin/env bash
# T2: train only the two new rho victims; reuse frozen rho=.1 references.
set -euo pipefail
DOSE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOSE_CODE_ROOT="$(cd -- "${DOSE_SCRIPT_DIR}/.." && pwd)"
DOSE_DATA_HOME="${DOSE_DATA_HOME:-$(cd -- "${DOSE_CODE_ROOT}/.." && pwd)}"
DOSE_PY="${DOSE_PY:-${DOSE_DATA_HOME}/envs/dose-backdoor/bin/python}"
exec "${DOSE_PY}" -u "${DOSE_CODE_ROOT}/ATKBackd/run_a1_dose_poison.py" \
    --data-home "${DOSE_DATA_HOME}" "$@"
