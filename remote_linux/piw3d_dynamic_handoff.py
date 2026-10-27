"""One-time, auditable queue handoff for the running PiW3D seed-42 attack run.

This file lives outside the science source tree bound by controller_plan.json.
It adopts the six wave-0 workers without stopping them, fills their GPU slots
with the remaining original-plan cells, and returns control to the frozen
original controller only after all 16 training checkpoints are final.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid


DEFAULT_REPO = Path('/workspace/piw3d_code_smokefix_v3/backdoor-analog')
DEFAULT_OUT = Path('/workspace/runs/piw3d_two_models_s42_3pergpu_v2_smokefix')
SLEEP_SECONDS = 10
MAX_PER_GPU = 3
MAX_PER_BACKBONE = 2


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.writing')
    with temp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def process_record(pid: int) -> dict | None:
    root = Path('/proc') / str(pid)
    try:
        stat = (root / 'stat').read_text()
        fields = stat[stat.rfind(') ') + 2:].split()
        command = [x.decode('utf-8', 'replace') for x in
                   (root / 'cmdline').read_bytes().split(b'\0') if x]
    except FileNotFoundError:
        return None
    return dict(pid=pid, ppid=int(fields[1]), state=fields[0],
                start_ticks=int(fields[19]),
                cmdline=command)


def still_live(record: dict) -> bool:
    current = process_record(int(record['pid']))
    if current is None:
        return False
    if current['start_ticks'] != record['start_ticks']:
        raise RuntimeError(f"PID {record['pid']} was reused")
    if current['state'] in ('Z', 'X'):
        return False
    if current['cmdline'] != record['cmdline']:
        raise RuntimeError(f"PID {record['pid']} command changed")
    return True


def process_stopped(record: dict) -> bool:
    current = process_record(int(record['pid']))
    if (current is None or current['start_ticks'] != record['start_ticks'] or
            current['cmdline'] != record['cmdline']):
        raise RuntimeError(f"Process {record['pid']} disappeared or changed")
    return current['state'] in ('T', 't')


def worker_processes(worker: Path, out: Path) -> dict[str, dict]:
    found = {}
    for root in Path('/proc').iterdir():
        if not root.name.isdecimal():
            continue
        try:
            rec = process_record(int(root.name))
        except (PermissionError, ProcessLookupError):
            continue
        if rec is None or rec['state'] in ('Z', 'X'):
            continue
        argv = rec['cmdline']
        if str(worker) not in argv or '--cell' not in argv or '--outdir' not in argv:
            continue
        if argv[argv.index('--outdir') + 1] != str(out):
            continue
        cell = argv[argv.index('--cell') + 1]
        if cell in found:
            raise RuntimeError(f'Duplicate live worker for {cell}')
        found[cell] = rec
    return found


def gpu_compute_pids() -> set[int]:
    output = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], capture_output=True, text=True,
        check=True, timeout=30).stdout
    return {int(line.strip()) for line in output.splitlines()
            if line.strip().isdecimal()}


def load_science(repo: Path, out: Path):
    path = repo / 'remote_linux/piw3d_two_models/controller.py'
    spec = importlib.util.spec_from_file_location('piw_handoff_original_controller', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Original controller source missing')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    from argparse import Namespace
    args = Namespace(repo_dir=repo, data_home=Path('/workspace/piw3d'),
        dataset_root=Path('/workspace/piw3d/datasets/Person-in-WiFi-3D_processed_v1'),
        action_npy=Path('/workspace/piw3d/actions/data_bend.npy'), outdir=out)
    plan = controller.read_plan(args)
    gate = controller.validate_gate(args, plan)
    sys.path.insert(0, str(repo / 'ATKBackd'))
    import run_piw_two_models_attacks as worker_module
    worker_plan = worker_module.load_plan(plan['worker_dry_run']['cfg'])
    return controller, plan, gate, worker_module, worker_plan


def cells_by_id(plan: dict) -> dict[str, dict]:
    return {cell['id']: dict(cell, wave=wave)
            for wave, group in enumerate(plan['waves']) for cell in group}


def original_launches(out: Path) -> dict[str, dict]:
    log = (out / 'controller.log').read_text(encoding='utf-8')
    pattern = re.compile(r'^\[train\] (\S+) PID=(\d+) (cuda:[01]) log=(\S+)$', re.M)
    launches = {}
    for match in pattern.finditer(log):
        cell, pid, gpu, log_path = match.groups()
        if cell in launches:
            raise RuntimeError(f'Multiple original launches of {cell}')
        launches[cell] = dict(pid=int(pid), device=gpu, log_path=log_path)
    return launches


def parent_from_lock(out: Path, repo: Path) -> dict:
    lock = (out / 'controller.lock').read_text(encoding='utf-8')
    match = re.fullmatch(r'pid=(\d+) started=.+\n', lock)
    if match is None:
        raise RuntimeError('Original controller lock format changed')
    rec = process_record(int(match.group(1)))
    expected = str(repo / 'remote_linux/piw3d_two_models/controller.py')
    if (rec is None or expected not in rec['cmdline'] or '--launch' not in rec['cmdline'] or
            str(out) not in rec['cmdline']):
        raise RuntimeError('Controller lock does not identify the live original controller')
    return rec


def complete_checkpoint(worker_module, worker_plan: dict, cell: dict, log: Path) -> dict:
    marker = f"TRAIN_COMPLETE {cell['id']}; TEST deferred"
    if not log.is_file() or marker not in log.read_text(encoding='utf-8', errors='replace'):
        raise RuntimeError(f"Worker ended without TRAIN_COMPLETE: {cell['id']}")
    worker_module._checkpoint(worker_plan, cell['backbone'],
                              cell['method'], final=True)
    checkpoint = (Path(worker_plan['cfg']['outdir']) / 'attacks' / cell['backbone'] /
                  'seed42' / cell['method'] / 'checkpoint.pt')
    return dict(cell=cell['id'], wave=cell['wave'], log_path=str(log),
                log_sha256=sha(log), checkpoint_sha256=sha(checkpoint))


def eligible(pending: list[str], active: dict, cells: dict[str, dict], gpu: str) -> str | None:
    on_gpu = [cells[name] for name in active if cells[name]['device'] == gpu]
    if len(on_gpu) >= MAX_PER_GPU:
        return None
    for name in pending:
        cell = cells[name]
        if cell['device'] != gpu:
            continue
        if sum(x['backbone'] == cell['backbone'] for x in on_gpu) >= MAX_PER_BACKBONE:
            continue
        # The exact wave-2 operators were only smoked with two workers/GPU.
        if cell['wave'] == 2 and len(on_gpu) >= 2:
            continue
        return name
    return None


def make_initial(out: Path, repo: Path, plan: dict, gate: dict, worker: Path,
                 worker_module, worker_plan: dict) -> dict:
    cells = cells_by_id(plan)
    wave0 = {cell['id'] for cell in plan['waves'][0]}
    launches = original_launches(out)
    if set(launches) != wave0:
        raise RuntimeError('Original controller is not still in its first training wave')
    if (out / 'state/train').exists() and list((out / 'state/train').iterdir()):
        raise RuntimeError('Original training markers already exist; no handoff')
    parent = parent_from_lock(out, repo)
    server = process_record(parent['ppid'])
    if (server is None or not server['cmdline'] or
            not Path(server['cmdline'][0]).name.startswith('tmux') or
            str(out) not in ' '.join(server['cmdline'])):
        raise RuntimeError('Cannot identify the original tmux server')
    if process_stopped(parent) or process_stopped(server):
        raise RuntimeError('Original controller is already stopped')
    found = worker_processes(worker, out)
    if not set(found).issubset(wave0):
        raise RuntimeError('Unexpected live PiW worker already exists')
    if gpu_compute_pids() != {rec['pid'] for rec in found.values()}:
        raise RuntimeError('Other GPU processes exist; do not change scheduling')
    active, completed = {}, {}
    for name in wave0:
        info = launches[name]
        if cells[name]['device'] != info['device']:
            raise RuntimeError(f'Original device changed for {name}')
        log = Path(info['log_path'])
        if not log.is_file() or not log.is_relative_to(out / 'logs/train'):
            raise RuntimeError(f'Original worker log missing/foreign: {name}')
        if name in found:
            rec = found[name]
            if rec['pid'] != info['pid']:
                raise RuntimeError(f'Original PID changed for {name}')
            active[name] = dict(rec, log_path=str(log), origin='original')
        else:
            completed[name] = complete_checkpoint(worker_module, worker_plan,
                                                 cells[name], log)
    pending = [cell['id'] for wave in plan['waves'][1:] for cell in wave]
    return dict(schema=1, status='prepared', plan_sha256=plan['plan_sha256'],
        gate_sha256=gate['gate_sha256'], script_sha256=sha(Path(__file__)),
        parent=parent, tmux_server=server, active=active, completed=completed,
        pending=pending,
        original_workers=launches, created_utc=time.time(),
        policy=dict(max_per_gpu=MAX_PER_GPU, max_per_backbone=MAX_PER_BACKBONE,
                    wave2_launch_only_when_gpu_has_at_most_one_job=True,
                    same_original_cell_devices=True, no_training_recipe_change=True,
                    eval_deferred_to_original_controller=True))


def inspect(repo: Path, out: Path) -> dict:
    controller, plan, gate, worker_module, worker_plan = load_science(repo, out)
    worker = repo / 'ATKBackd/run_piw_two_models_attacks.py'
    journal = out / 'dynamic_handoff_state.json'
    if journal.exists():
        state = json.loads(journal.read_text(encoding='utf-8'))
        if (state['plan_sha256'] != plan['plan_sha256'] or
                state['gate_sha256'] != gate['gate_sha256'] or
                state['script_sha256'] != sha(Path(__file__))):
            raise RuntimeError('Dynamic handoff plan/source changed')
    else:
        state = make_initial(out, repo, plan, gate, worker, worker_module,
                             worker_plan)
    return dict(controller=controller, plan=plan, gate=gate, worker_module=worker_module,
                worker_plan=worker_plan, worker=worker, journal=journal, state=state,
                cells=cells_by_id(plan))


def reconcile(state: dict, found: dict[str, dict], cells: dict[str, dict],
              out: Path, worker_module, worker_plan: dict) -> bool:
    changed = False
    for name, rec in found.items():
        if name in state['active']:
            if rec['pid'] != state['active'][name]['pid']:
                raise RuntimeError(f'Worker PID changed unexpectedly: {name}')
            continue
        if name not in state['pending']:
            raise RuntimeError(f'Unregistered worker: {name}')
        logs = list((out / 'logs/train').glob(name + '.dynamic.*.log'))
        if len(logs) != 1:
            raise RuntimeError(f'Cannot recover unique dynamic log for {name}')
        state['active'][name] = dict(rec, log_path=str(logs[0]), origin='dynamic')
        state['pending'].remove(name)
        changed = True
    for name in list(state['pending']):
        logs = list((out / 'logs/train').glob(name + '.dynamic.*.log'))
        if not logs:
            continue
        if len(logs) != 1 or name in found:
            raise RuntimeError(f'Cannot recover unique finished dynamic worker: {name}')
        result = complete_checkpoint(worker_module, worker_plan, cells[name], logs[0])
        state['completed'][name] = result
        state['pending'].remove(name)
        changed = True
        print(f'[queue] RECOVERED COMPLETE {name}', flush=True)
    return changed


def launch_one(state: dict, cell: dict, repo: Path, out: Path) -> None:
    name = cell['id']
    log = out / 'logs/train' / f'{name}.dynamic.{uuid.uuid4().hex[:12]}.log'
    command = [sys.executable, '-u', str(repo / 'ATKBackd/run_piw_two_models_attacks.py'),
               '--dataset-root', str(Path('/workspace/piw3d/datasets/Person-in-WiFi-3D_processed_v1')),
               '--action-npy', str(Path('/workspace/piw3d/actions/data_bend.npy')),
               '--outdir', str(out), '--cell', name, '--device', cell['device']]
    with log.open('x', encoding='utf-8') as sink:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=sink,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    rec = None
    for _ in range(50):
        rec = process_record(child.pid)
        if rec is not None and rec['cmdline'] == command:
            break
        if child.poll() is not None:
            break
        time.sleep(.02)
    if rec is None or rec['cmdline'] != command:
        raise RuntimeError(f'Dynamic worker exited immediately: {name}; {log}')
    state['active'][name] = dict(rec, log_path=str(log), origin='dynamic')
    state['pending'].remove(name)
    print(f'[queue] {name} PID={child.pid} {cell["device"]} log={log}', flush=True)


def status(out: Path, state: dict) -> None:
    atomic_json(out / 'dynamic_handoff_status.json', dict(
        schema=1, status=state['status'], plan_sha256=state['plan_sha256'],
        active={name: dict(pid=rec['pid'], origin=rec['origin'])
                for name, rec in sorted(state['active'].items())},
        completed=sorted(state['completed']), pending=state['pending'],
        updated_utc=time.time()))


def finish_training(info: dict, repo: Path, out: Path) -> None:
    state, cells, plan = info['state'], info['cells'], info['plan']
    if state['pending'] or state['active'] or len(state['completed']) != len(cells):
        raise RuntimeError('Cannot return controller before every victim is final')
    if not process_stopped(state['parent']) or not process_stopped(state['tmux_server']):
        raise RuntimeError('Original controller is not frozen at handoff')
    for wave in plan['waves'][1:]:
        for cell in wave:
            record = state['completed'][cell['id']]
            info['worker_module']._checkpoint(
                info['worker_plan'], cell['backbone'], cell['method'], final=True)
            if sha(Path(record['log_path'])) != record['log_sha256']:
                raise RuntimeError(f'Dynamic worker log changed: {cell["id"]}')
            info['controller'].atomic_new_json(out / 'state/train' / (cell['id'] + '.json'),
                dict(cell=cell['id'], mode='train', wave=cells[cell['id']]['wave'],
                     plan_sha256=plan['plan_sha256'], log_path=record['log_path'],
                     log_sha256=record['log_sha256']))
    atomic_json(out / 'dynamic_handoff_summary.json', dict(schema=1,
        status='TRAIN_COMPLETE_RETURNING_TO_ORIGINAL_CONTROLLER',
        plan_sha256=plan['plan_sha256'], gate_sha256=info['gate']['gate_sha256'],
        original_controller_pid=state['parent']['pid'],
        completed=state['completed'], policy=state['policy'],
        script_sha256=state['script_sha256'], returned_utc=time.time()))
    os.kill(state['tmux_server']['pid'], signal.SIGCONT)
    os.kill(state['parent']['pid'], signal.SIGCONT)
    state['status'] = 'returned_to_original'
    atomic_json(info['journal'], state)
    status(out, state)
    print('[queue] ALL 16 FINAL199; original controller resumed for TEST/evaluation', flush=True)


def run(repo: Path, out: Path) -> int:
    import torch  # noqa: F401 - fail early if the existing science environment changed
    lock_path = out / 'dynamic_handoff.lock'
    with lock_path.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = inspect(repo, out)
        state, cells, journal = info['state'], info['cells'], info['journal']
        if state['status'] == 'returned_to_original':
            print('ALREADY_RETURNED_TO_ORIGINAL_CONTROLLER', flush=True)
            return 0
        if not journal.exists():
            atomic_json(journal, state)
        if not (process_stopped(state['parent']) and
                process_stopped(state['tmux_server'])):
            if state['status'] != 'prepared':
                raise RuntimeError('Original controller unexpectedly running')
            try:
                os.kill(state['tmux_server']['pid'], signal.SIGSTOP)
                os.kill(state['parent']['pid'], signal.SIGSTOP)
                for _ in range(50):
                    if (process_stopped(state['parent']) and
                            process_stopped(state['tmux_server'])):
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError('Could not freeze original controller/tmux')
                unexpected = set(worker_processes(info['worker'], out)) - {
                    cell['id'] for cell in info['plan']['waves'][0]}
                if unexpected:
                    raise RuntimeError(f'Original controller advanced during handoff: {unexpected}')
            except BaseException:
                os.kill(state['tmux_server']['pid'], signal.SIGCONT)
                os.kill(state['parent']['pid'], signal.SIGCONT)
                raise
        state['status'] = 'queue_running'
        atomic_json(journal, state)
        status(out, state)
        print('[queue] original controller frozen; six existing workers retained', flush=True)
        while len(state['completed']) < len(cells):
            if (not process_stopped(state['parent']) or
                    not process_stopped(state['tmux_server'])):
                raise RuntimeError('Original controller resumed unexpectedly')
            found = worker_processes(info['worker'], out)
            if reconcile(state, found, cells, out, info['worker_module'],
                         info['worker_plan']):
                atomic_json(journal, state)
            for name, rec in list(state['active'].items()):
                if still_live(rec):
                    continue
                result = complete_checkpoint(info['worker_module'], info['worker_plan'],
                                             cells[name], Path(rec['log_path']))
                state['completed'][name] = result
                del state['active'][name]
                atomic_json(journal, state)
                print(f'[queue] COMPLETE {name}; device slot freed', flush=True)
            for gpu in ('cuda:0', 'cuda:1'):
                while (name := eligible(state['pending'], state['active'], cells, gpu)):
                    launch_one(state, cells[name], repo, out)
                    atomic_json(journal, state)
            status(out, state)
            time.sleep(SLEEP_SECONDS)
        finish_training(info, repo, out)
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    repo, out = args.repo.resolve(), args.outdir.resolve()
    if args.status:
        path = out / 'dynamic_handoff_status.json'
        print(path.read_text(encoding='utf-8') if path.exists() else 'NOT_STARTED')
        return 0
    if args.audit:
        info = inspect(repo, out)
        state = info['state']
        print(json.dumps(dict(status='AUDIT_PASS_NO_ACTION',
            plan_sha256=state['plan_sha256'], parent_pid=state['parent']['pid'],
            active={name: rec['pid'] for name, rec in state['active'].items()},
            completed=sorted(state['completed']), pending=state['pending'],
            policy=state['policy']), indent=2))
        return 0
    try:
        return run(repo, out)
    except BaseException as error:
        atomic_json(out / 'dynamic_handoff_failure.json', dict(
            status='FAIL_STOP', error=repr(error), traceback=traceback.format_exc(),
            time_utc=time.time(), script_sha256=sha(Path(__file__))))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
