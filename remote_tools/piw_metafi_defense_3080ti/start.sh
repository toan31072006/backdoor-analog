#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BASE=${DEFENSE_BASE:-/media/jackson/Data/wificsi/defense_s42_3080ti}
PY=${DEFENSE_PYTHON:-$BASE/env/bin/python}
ACTION=${1:-status}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MKL_THREADING_LAYER=SEQUENTIAL
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
test -x "$PY" || { echo "Missing existing CEI Python: $PY" >&2; exit 1; }
case "$ACTION" in
  launch)
    mkdir -p "$BASE/logs/piw_metafi4090"
    LOG="$BASE/logs/piw_metafi4090/run.$(date -u +%Y%m%dT%H%M%SZ).$$.log"
    nohup "$PY" -B -u "$HERE/run.py" --base "$BASE" --mode all >"$LOG" 2>&1 </dev/null &
    printf 'STARTED_PID=%s\nLOG=%s\n' "$!" "$LOG"
    ;;
  prepare|run|status|collect)
    exec "$PY" -B -u "$HERE/run.py" --base "$BASE" --mode "$ACTION"
    ;;
  *) echo 'Usage: sh start.sh launch|prepare|run|status|collect' >&2; exit 2 ;;
esac
