#!/usr/bin/env bash
# Full MM-Fi confirmation of one previously frozen Proposed trigger.
set -euo pipefail
DOSE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOSE_CODE_ROOT="$(cd -- "${DOSE_SCRIPT_DIR}/.." && pwd)"
DOSE_DATA_HOME="${DOSE_DATA_HOME:-$(cd -- "${DOSE_CODE_ROOT}/.." && pwd)}"
DOSE_PY="${DOSE_PY:-}"
DOSE_FORWARDED=()
while (($#)); do
    case "$1" in
        --python)
            [[ $# -ge 2 ]] || { printf 'Missing --python value\n' >&2; exit 2; }
            DOSE_PY="$2"; shift 2 ;;
        --data-home)
            [[ $# -ge 2 ]] || { printf 'Missing --data-home value\n' >&2; exit 2; }
            DOSE_DATA_HOME="$2"; shift 2 ;;
        *) DOSE_FORWARDED+=("$1"); shift ;;
    esac
done
DOSE_PY="${DOSE_PY:-${DOSE_DATA_HOME}/envs/dose-backdoor/bin/python}"
DOSE_SOURCE_DEFAULT="${DOSE_DATA_HOME}/runs/mmfi_paired_guard_s42_v1/lc_paired_guard"
DOSE_OUT_DEFAULT="${DOSE_DATA_HOME}/runs/mmfi_paired_guard_full_s42_v1"
if [[ ! -x "${DOSE_PY}" ]]; then
    printf 'Missing Python: %s\nSet DOSE_PY or use --python.\n' "${DOSE_PY}" >&2
    exit 1
fi
exec "${DOSE_PY}" -u "${DOSE_CODE_ROOT}/ATKBackd/run_paired_guard_full.py" \
    --data-home "${DOSE_DATA_HOME}" --source-cell "${DOSE_SOURCE_DEFAULT}" \
    --outdir "${DOSE_OUT_DEFAULT}" "${DOSE_FORWARDED[@]}"
