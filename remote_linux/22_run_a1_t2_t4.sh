#!/usr/bin/env bash
# Independent jobs: two T2 victim GPUs and one T4 inference GPU.
set -euo pipefail
DOSE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOSE_CODE_ROOT="$(cd -- "${DOSE_SCRIPT_DIR}/.." && pwd)"
DOSE_DATA_HOME="${DOSE_DATA_HOME:-$(cd -- "${DOSE_CODE_ROOT}/.." && pwd)}"
DOSE_PY="${DOSE_PY:-${DOSE_DATA_HOME}/envs/dose-backdoor/bin/python}"
DOSE_TRAIN_GPUS="${DOSE_TRAIN_GPUS:-0,2}"
DOSE_EVAL_GPU="${DOSE_EVAL_GPU:-3}"
export DOSE_DATA_HOME DOSE_PY
"${DOSE_PY}" -c 'import sklearn' || {
    printf 'Install T4 dependency with dose-backdoor Python: pip install -r remote_linux/requirements_t4.txt\n' >&2
    exit 1
}
DOSE_T2_LOG="${DOSE_DATA_HOME}/runs/mmfi_a1_dose_poison_s42_v1.startup.log"
DOSE_T4_LOG="${DOSE_DATA_HOME}/runs/mmfi_a1_stealthiness_s42_v1.startup.log"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${DOSE_TRAIN_GPUS}" \
    bash "${DOSE_SCRIPT_DIR}/20_run_a1_dose_poison.sh" \
    --devices cuda:0 cuda:1 --num-workers 4 >"${DOSE_T2_LOG}" 2>&1 &
DOSE_T2_PID=$!
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${DOSE_EVAL_GPU}" \
    bash "${DOSE_SCRIPT_DIR}/21_run_a1_stealthiness.sh" \
    --device cuda:0 >"${DOSE_T4_LOG}" 2>&1 &
DOSE_T4_PID=$!
printf 'T2 PID=%s, physical GPUs=%s, log=%s\n' "${DOSE_T2_PID}" "${DOSE_TRAIN_GPUS}" "${DOSE_T2_LOG}"
printf 'T4 PID=%s, physical GPU=%s, log=%s\n' "${DOSE_T4_PID}" "${DOSE_EVAL_GPU}" "${DOSE_T4_LOG}"
DOSE_T2_STATUS=0
DOSE_T4_STATUS=0
wait "${DOSE_T2_PID}" || DOSE_T2_STATUS=$?
wait "${DOSE_T4_PID}" || DOSE_T4_STATUS=$?
printf 'Finished: T2 exit=%s; T4 exit=%s\n' "${DOSE_T2_STATUS}" "${DOSE_T4_STATUS}"
((DOSE_T2_STATUS == 0 && DOSE_T4_STATUS == 0))
