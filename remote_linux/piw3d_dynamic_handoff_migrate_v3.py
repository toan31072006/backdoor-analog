"""One-time, fail-closed migration of the active PiW queue to its v3 scheduler."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

OUT = Path('/workspace/runs/piw3d_two_models_s42_3pergpu_v2_smokefix')
REPO = Path('/workspace/piw3d_code_smokefix_v3/backdoor-analog')
OLD_SCRIPT = Path('/workspace/piw3d_dynamic_handoff_v2.py')
NEW_SCRIPT = Path('/workspace/piw3d_dynamic_handoff_v3.py')
OLD_SHA = 'a1ed28df41e8e1b915a9ef5184ba5e09b24f48ef1849a6bb0138cec78face8aa'
NEW_SHA = '9ac9a303a845b4aa797fc499d920219b92e64496382d67c9feeabfefeef60990'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


old, new = module(OLD_SCRIPT, 'piw_handoff_old'), module(NEW_SCRIPT, 'piw_handoff_new')
if old.sha(OLD_SCRIPT) != OLD_SHA or new.sha(NEW_SCRIPT) != NEW_SHA:
    raise RuntimeError('Scheduler file hash changed')
lock = OUT / 'dynamic_handoff.lock'
with lock.open('a+') as handle:
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = OUT / 'dynamic_handoff_state.json'
    original = path.read_bytes()
    state = json.loads(original)
    if state['status'] != 'queue_running' or state['script_sha256'] != OLD_SHA:
        raise RuntimeError('Not the expected live v2 queue')
    if not old.process_stopped(state['parent']) or not old.process_stopped(state['tmux_server']):
        raise RuntimeError('Original controller/tmux must remain frozen')
    controller, plan, gate, worker, worker_plan = new.load_science(REPO, OUT)
    if state['plan_sha256'] != plan['plan_sha256'] or state['gate_sha256'] != gate['gate_sha256']:
        raise RuntimeError('Science plan or GPU smoke gate changed')
    cells = new.cells_by_id(plan)
    found = new.worker_processes(REPO / 'ATKBackd/run_piw_two_models_attacks.py', OUT)
    if not set(found).issubset(set(state['active']) | set(state['pending'])):
        raise RuntimeError('Unregistered PiW worker before migration')
    for name, record in found.items():
        if name in state['active'] and record['pid'] != state['active'][name]['pid']:
            raise RuntimeError(f'Worker PID changed: {name}')
        if name not in cells:
            raise RuntimeError(f'Foreign worker: {name}')
    backup = OUT / 'dynamic_handoff_state_before_v3.json'
    with backup.open('xb') as sink:
        sink.write(original)
        sink.flush()
        os.fsync(sink.fileno())
    receipt = dict(schema=1, previous_script_sha256=OLD_SHA,
                   new_script_sha256=NEW_SHA,
                   previous_journal_sha256=hashlib.sha256(original).hexdigest(),
                   plan_sha256=state['plan_sha256'],
                   active=sorted(state['active']),
                   completed=sorted(state['completed']),
                   pending=state['pending'],
                   migrated_utc=time.time(),
                   reason='Allow wave-2 cells beside one active worker, matching prepared two-worker GPU smoke')
    new.atomic_json(OUT / 'dynamic_handoff_v2_to_v3.json', receipt)
    state['script_sha256'] = NEW_SHA
    state['scheduler_migration'] = receipt
    new.atomic_json(path, state)
print('MIGRATED_QUEUE_V2_TO_V3; original workers/checkpoints untouched')
