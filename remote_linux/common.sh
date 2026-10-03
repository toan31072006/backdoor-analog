#!/usr/bin/env bash

# Shared path resolution and argument parsing for the MICA/Linux launchers.
# Scientific hyperparameters are intentionally kept in ATKBackd/configs; these
# helpers only provide machine-specific paths and runtime settings.

set -Eeuo pipefail

# Headless, line-buffered behavior is safer for SSH/tmux runs.
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

LINUX_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${LINUX_SCRIPT_DIR}/.." && pwd)"
ATK_ROOT="${CODE_ROOT}/ATKBackd"

PYTHON_BIN="${BACKDOOR_PYTHON:-python}"
DEVICE="${BACKDOOR_DEVICE:-cuda:0}"
DATASET_SELECTION="${BACKDOOR_DATASET:-both}"
PARALLEL="${BACKDOOR_PARALLEL:-1}"
NUM_WORKERS="${BACKDOOR_NUM_WORKERS:-4}"
PIW_PIVOT="${BACKDOOR_PIW_PIVOT:-7}"

DATA_HOME_OVERRIDE="${BACKDOOR_DATA_HOME:-}"
MMFI_ROOT_OVERRIDE="${BACKDOOR_MMFI_ROOT:-}"
PIW_ROOT_OVERRIDE="${BACKDOOR_PIW_ROOT:-}"
ACTION_NPY_OVERRIDE="${BACKDOOR_ACTION_NPY:-}"
RUNS_ROOT_OVERRIDE="${BACKDOOR_RUNS_ROOT:-}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '\n[%s] %s\n' "remote_linux" "$*"
}

usage_common() {
    cat <<'EOF'
Common options:
  --dataset mmfi|piw3d|both   Dataset(s) to run (default: both)
  --python PATH               Python interpreter (default: python)
  --device DEVICE             Torch device (default: cuda:0)
  --parallel N                Experiment processes (default: 1)
  --num-workers N             DataLoader workers per process (default: 4)
  --piw-pivot N               PiW3D payload pivot (paper contract: 7)
  --data-home PATH            Data/output home (default: $HOME/backdooranalog)
  --mmfi-root PATH            MM-Fi root containing E01..E04
  --piw-root PATH             PiW3D root containing train_data/test_data
  --action-npy PATH           Bend action .npy file
  --runs-root PATH            Output root (default: DATA_HOME/runs)
  -h, --help                  Show help

The same values can be set with BACKDOOR_* environment variables.
EOF
}

need_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "$value" ]] || die "${option} requires a value"
}

parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dataset)
                need_value "$1" "${2:-}"
                DATASET_SELECTION="$2"
                shift 2
                ;;
            --python)
                need_value "$1" "${2:-}"
                PYTHON_BIN="$2"
                shift 2
                ;;
            --device)
                need_value "$1" "${2:-}"
                DEVICE="$2"
                shift 2
                ;;
            --parallel)
                need_value "$1" "${2:-}"
                PARALLEL="$2"
                shift 2
                ;;
            --num-workers)
                need_value "$1" "${2:-}"
                NUM_WORKERS="$2"
                shift 2
                ;;
            --piw-pivot)
                need_value "$1" "${2:-}"
                PIW_PIVOT="$2"
                shift 2
                ;;
            --data-home)
                need_value "$1" "${2:-}"
                DATA_HOME_OVERRIDE="$2"
                shift 2
                ;;
            --mmfi-root)
                need_value "$1" "${2:-}"
                MMFI_ROOT_OVERRIDE="$2"
                shift 2
                ;;
            --piw-root)
                need_value "$1" "${2:-}"
                PIW_ROOT_OVERRIDE="$2"
                shift 2
                ;;
            --action-npy)
                need_value "$1" "${2:-}"
                ACTION_NPY_OVERRIDE="$2"
                shift 2
                ;;
            --runs-root)
                need_value "$1" "${2:-}"
                RUNS_ROOT_OVERRIDE="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "unknown option: $1 (use --help)"
                ;;
        esac
    done
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

finalize_common() {
    case "$DATASET_SELECTION" in
        mmfi|piw3d|both) ;;
        *) die "--dataset must be mmfi, piw3d, or both" ;;
    esac

    is_nonnegative_integer "$PARALLEL" || die "--parallel must be an integer"
    (( PARALLEL >= 1 )) || die "--parallel must be at least 1"
    is_nonnegative_integer "$NUM_WORKERS" || die "--num-workers must be non-negative"
    is_nonnegative_integer "$PIW_PIVOT" || die "--piw-pivot must be non-negative"
    (( PIW_PIVOT == 7 )) || die "paper contract requires --piw-pivot 7"

    DATA_HOME="${DATA_HOME_OVERRIDE:-${HOME}/backdooranalog}"
    MMFI_ROOT="${MMFI_ROOT_OVERRIDE:-${DATA_HOME}/datasets/Compress}"
    PIW_ROOT="${PIW_ROOT_OVERRIDE:-${DATA_HOME}/datasets/Person-in-WiFi-3D}"
    RUNS_ROOT="${RUNS_ROOT_OVERRIDE:-${DATA_HOME}/runs}"

    [[ -d "$ATK_ROOT" ]] || die "ATKBackd source not found: $ATK_ROOT"
}

require_python() {
    if [[ "$PYTHON_BIN" == */* ]]; then
        [[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
    else
        command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"
    fi
}

resolve_action() {
    local candidates=()
    if [[ -n "$ACTION_NPY_OVERRIDE" ]]; then
        candidates+=("$ACTION_NPY_OVERRIDE")
    else
        candidates+=(
            "${DATA_HOME}/actions/data_bend.npy"
            "${DATA_HOME}/actions/_bend.npy"
            "${DATA_HOME}/data/data_bend.npy"
            "${DATA_HOME}/data/_bend.npy"
            "${DATA_HOME}/psba_dataset/data_bend.npy"
            "${DATA_HOME}/data_bend.npy"
        )
    fi

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            ACTION_NPY="$(cd -- "$(dirname -- "$candidate")" && pwd)/$(basename -- "$candidate")"
            return 0
        fi
    done
    ACTION_NPY=""
    return 1
}

resolve_optional_action() {
    local action_name="$1"
    local candidate
    for candidate in \
        "${DATA_HOME}/actions/data_${action_name}.npy" \
        "${DATA_HOME}/actions/_${action_name}.npy" \
        "${DATA_HOME}/data/data_${action_name}.npy" \
        "${DATA_HOME}/data/_${action_name}.npy" \
        "${DATA_HOME}/psba_dataset/data_${action_name}.npy"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

require_action() {
    resolve_action || die "missing bend action. Put data_bend.npy (or _bend.npy) in ${DATA_HOME}/actions"
}

require_selected_datasets() {
    if [[ "$DATASET_SELECTION" == "mmfi" || "$DATASET_SELECTION" == "both" ]]; then
        [[ -d "$MMFI_ROOT" ]] || die "MM-Fi root not found: $MMFI_ROOT"
    fi
    if [[ "$DATASET_SELECTION" == "piw3d" || "$DATASET_SELECTION" == "both" ]]; then
        [[ -d "$PIW_ROOT" ]] || die "PiW3D root not found: $PIW_ROOT"
    fi
}

run_python() {
    printf '\n>'
    printf ' %q' "$PYTHON_BIN" "$@"
    printf '\n'
    "$PYTHON_BIN" "$@"
}

run_atk_python() {
    (
        cd -- "$ATK_ROOT"
        run_python "$@"
    )
}

print_resolved_paths() {
    printf '%s\n' \
        "Code       : $ATK_ROOT" \
        "Data home  : $DATA_HOME" \
        "MM-Fi      : $MMFI_ROOT" \
        "PiW3D      : $PIW_ROOT" \
        "Bend action: ${ACTION_NPY:-MISSING}" \
        "Runs       : $RUNS_ROOT" \
        "Python     : $PYTHON_BIN" \
        "Device     : $DEVICE"
}
