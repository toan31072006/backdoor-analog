"""Opt-in foreground coordinator: one A1 prior plus one BadNets worker.

Original runners, scientific files, preparation gates and checkpoints are not
rewritten by this sidecar. Stop the old serial queue manually before Prepare.
There is no process kill, retry, environment install or SSH persistence trick.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                  MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
PROFILE = 'mmfi_a1_badnets_shared4090_v1'
HERE = Path(__file__).resolve().parent
CELLS = {'a1': ('cross', 'nod'), 'badnets': ('wiflow', 'metafiplusplus')}
OUTPUTS = {'a1': 'mmfi_p1s1_hpeli_a1_motion_priors_s42_4090_v1',
           'badnets': 'mmfi_p1s1_badnets_wiflow_metafi_s42_4090_v1'}
PLANS = {'a1': 'motion_priors.plan.json', 'badnets': 'badnets_two_models.plan.json'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def safe_print(*values, **kwargs):
    """A disconnected output pipe must not orphan an already-started child."""
    try:
        print(*values, **kwargs)
    except (OSError, ValueError):
        pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def signed(value, field):
    return dict(value, **{field: digest(value)})


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def write_state(path, value):
    """Only our uniquely named execution-state record may be updated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


@contextmanager
def os_lock(folder):
    """Same run.lock byte/OS mechanism as the unchanged native runners."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / 'run.lock').open('a+b')
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another controller/native queue owns {folder}; no lock was removed') from error
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def load_context():
    spec = importlib.util.spec_from_file_location('_mmfi_motion_prior_shared_context', HERE / 'context.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paths(args):
    values = {name: Path(getattr(args, name)).resolve()
              for name in ('home', 'a1_repo', 'badnets_repo', 'root')}
    for path in values.values():
        if str(path).startswith('\\\\') or any(c in str(path) for c in ('"', '\r', '\n', '\x00')):
            raise ValueError('Only ordinary local paths are supported')
    home, root = values['home'], values['root']
    if home / 'tools' not in root.parents:
        raise ValueError('Shared Root must be a dedicated descendant of DataHome/tools')
    for repo in (values['a1_repo'], values['badnets_repo']):
        if root == repo or repo in root.parents or root in repo.parents:
            raise ValueError('Shared Root must be separate from both scientific checkouts')
    if values['a1_repo'] == values['badnets_repo']:
        raise ValueError('Keep the frozen BadNets checkout separate from the new A1 checkout')
    return values


def make_request(values):
    result = dict(schema=1, profile=PROFILE, python=str(Path(sys.executable).resolve()),
                  **{name: str(path) for name, path in values.items()})
    bindings = {}
    for role in CELLS:
        out = values['home'] / 'runs' / OUTPUTS[role]
        result[role + '_out'] = str(out)
        plan_path = out / PLANS[role]
        plan = read(plan_path)
        cells = plan.get('cells', [])
        field = 'prior' if role == 'a1' else 'backbone'
        if tuple(cell.get(field) for cell in cells) != CELLS[role]:
            raise ValueError(f'{role}: require exactly the two approved cells in the fixed order')
        if Path(plan.get('data_home', '')).resolve() != values['home']:
            raise ValueError('Native plan belongs to another data home')
        if Path(plan.get('outdir', '')).resolve() != out:
            raise ValueError('Native output is not the approved original run')
        for item in (plan_path, out / 'inputs.json',
                     *(Path(cell['run_dir']) / 'gpu_smoke.json' for cell in cells)):
            bindings[str(item.resolve())] = sha(item)
    result['bindings'] = bindings
    result['source_sha256'] = {str(path): sha(path)
                              for path in (Path(__file__).resolve(), HERE / 'context.py')}
    return signed(result, 'request_sha256')


def session(request, context):
    records = context.process_records()
    current = [item for item in records if int(item.get('ProcessId', -1)) == os.getpid()]
    if len(current) != 1:
        raise ValueError('Cannot bind the exact current Windows controller process')
    record = current[0]
    for field in ('ProcessId', 'ParentProcessId', 'CreationDate', 'ExecutablePath', 'CommandLine'):
        if record.get(field) is None or record[field] == '':
            raise ValueError(f'Controller CIM identity lacks {field}')
    value = signed(dict(schema=1, request_sha256=request['request_sha256'], controller=record),
                   'session_sha256')
    path = Path(request['root']) / 'sessions' / (uuid.uuid4().hex + '.json')
    write_new(path, value)
    return path


def phase_command(request, request_file, session_file, role, phase, cell, pair=None):
    if role not in CELLS or cell not in CELLS[role]:
        raise ValueError('Unknown role/cell')
    if phase not in ('audit', 'smoke', 'run', 'export'):
        raise ValueError('Unknown context phase')
    command = [request['python'], '-u', str(HERE / 'context.py'),
        '--request', str(request_file), '--session', str(session_file),
        '--role', role, '--phase', phase, '--cell', cell]
    if pair is not None:
        command += ['--pair', str(pair)]
    return command


def scoped_session(request, request_file, session_file, context):
    value = dict(context.validate_session(session_file, request))
    value.update(_request_path=str(Path(request_file).resolve()),
                 _session_path=str(Path(session_file).resolve()))
    return value


def audit_cell(request, request_file, session_file, role, cell):
    command = phase_command(request, request_file, session_file, role, 'audit', cell)
    result = subprocess.run(command, capture_output=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{role}/{cell} audit failed; no work launched: '
                           f'{result.stdout[-3000:]}\n{result.stderr[-3000:]}')
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError('Context audit did not return an authenticated result') from error
    if value.get('role') != role or value.get('cell') != cell or type(value.get('complete')) is not bool:
        raise ValueError('Context audit role/cell/status differs')
    return value


def audit_all(request, request_file, session_file):
    return {role: {cell: audit_cell(request, request_file, session_file, role, cell)
                   for cell in cells} for role, cells in CELLS.items()}


def start_phase(request, request_file, session_file, role, phase, cell, pair=None):
    log = Path(request['root']) / 'logs' / f'{phase}.{role}.{cell}.{uuid.uuid4().hex}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open('x', encoding='utf-8')
    command = phase_command(request, request_file, session_file, role, phase, cell, pair)
    try:
        # A separate console signal group permits draining peers after an
        # interrupted controller; this is not a service or detached SSH task.
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, creationflags=flags)
    except BaseException:
        stream.close()
        raise
    safe_print(f'OWNED {phase} {role}/{cell} PID={process.pid} LOG={log}; PID is not proof of progress', flush=True)
    return dict(process=process, stream=stream, role=role, cell=cell, log=str(log))


def close_owned(item):
    if item.get('stream') is not None and not item['stream'].closed:
        try:
            item['stream'].close()
        except OSError:
            pass


def drain_owned(owned):
    """No terminate/kill/signal/retry. Keep locks until every owned peer exits."""
    codes = {}
    while len(codes) < len(owned):
        try:
            for role, item in owned.items():
                if role in codes:
                    continue
                try:
                    code = item['process'].poll()
                except OSError:
                    safe_print(f'Cannot poll {role}; retaining locks and waiting for owned peer', flush=True)
                    continue
                if code is not None:
                    codes[role] = code
                    close_owned(item)
                    safe_print(f'DRAINED {role}/{item["cell"]}: exit={code}; no process stopped', flush=True)
            if len(codes) < len(owned):
                time.sleep(1)
        except KeyboardInterrupt:
            safe_print('Already draining owned peers; no signal or later job was sent', flush=True)
    return codes


def smoke_pair(request, request_file, session_file, index, cells):
    pair = Path(request['root']) / 'pairs' / f'{index}_{uuid.uuid4().hex}'
    pair.mkdir(parents=True, exist_ok=False)
    owned = {}
    try:
        for role, cell in cells.items():
            owned[role] = start_phase(request, request_file, session_file, role, 'smoke', cell, pair)
        codes = {}
        while len(codes) != len(owned):
            for role, item in owned.items():
                code = item['process'].poll()
                if code is not None:
                    codes[role] = code
            if any(code != 0 for code in codes.values()):
                raise RuntimeError(f'Paired B32 CUDA smoke failed: {codes}; no later pair or training')
            if len(codes) != len(owned):
                time.sleep(1)
    except BaseException:
        try:
            error = pair / 'error'
            if not error.exists():
                error.touch(exist_ok=False)
        except BaseException:
            safe_print('Cannot write smoke failure sentinel; still draining owned peers', flush=True)
        finally:
            drain_owned(owned)
        raise
    finally:
        for item in owned.values():
            close_owned(item)


def native_a1_prepare(values):
    runner = values['a1_repo'] / 'ATKBackd/run_mmfi_a1_motion_priors_4090.py'
    command = [sys.executable, '-u', str(runner), '--data-home', str(values['home']),
        '--outdir', str(values['home'] / 'runs' / OUTPUTS['a1']), '--mode', 'prepare',
        '--priors', 'cross', 'nod']
    print('NATIVE_A1_PREPARE: full input audit and original exclusive CUDA gates; no full training', flush=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    process = subprocess.Popen(command, creationflags=flags)
    try:
        code = process.wait()
    except BaseException:
        drain_owned({'native_a1_prepare': dict(process=process, stream=None, cell='prepare')})
        raise
    if code:
        raise RuntimeError('Original A1 Prepare failed; shared contexts/training were not launched')


def native_locks(request):
    stack = ExitStack()
    try:
        for folder in sorted([request['a1_out'], request['badnets_out']]):
            stack.enter_context(os_lock(folder))
        return stack
    except BaseException:
        stack.close()
        raise


def prepare(values, context):
    root = values['root']; request_file = root / 'request.json'
    if request_file.exists():
        raise FileExistsError('Shared request already exists; use Check/Launch. Failed smoke is never retried automatically.')
    context.require_exclusive_idle()
    native_a1_prepare(values)
    request = make_request(values)
    with native_locks(request):
        context.require_exclusive_idle()
        context.validate_request(request)
        write_new(request_file, request)
        session_file = session(request, context)
        context.validate_session(session_file, request)
        audits = audit_all(request, request_file, session_file)
        write_new(root / 'prepare_audits.json', audits)
        for index, cells in enumerate((dict(a1='cross', badnets='wiflow'),
                                       dict(a1='nod', badnets='metafiplusplus')), start=1):
            smoke_pair(request, request_file, session_file, index, cells)
        context.check_smokes(request)
        receipts = {str(root / 'smokes' / f'{role}.{cell}.json'):
                    sha(root / 'smokes' / f'{role}.{cell}.json')
                    for role, cells in CELLS.items() for cell in cells}
        receipt = signed(dict(schema=1, profile=PROFILE, request_sha256=request['request_sha256'],
            smoke_receipts=receipts, paired_b32_smokes=4,
            full_training_launched=False), 'prepared_sha256')
        write_new(root / 'prepared.json', receipt)
        print('PREPARED_SHARED_ONLY: four shared B32 CUDA smokes passed; no full training', flush=True)
    return request


def validate_prepared(request, context):
    context.validate_request(request)
    context.check_smokes(request)
    path = Path(request['root']) / 'prepared.json'
    value = read(path)
    unsigned = {key: item for key, item in value.items() if key != 'prepared_sha256'}
    if (value.get('prepared_sha256') != digest(unsigned) or value.get('schema') != 1
            or value.get('profile') != PROFILE
            or value.get('request_sha256') != request['request_sha256']
            or value.get('paired_b32_smokes') != 4
            or value.get('full_training_launched') is not False):
        raise ValueError('Missing/foreign/tampered shared preparation receipt')
    expected = {str(Path(request['root']) / 'smokes' / f'{role}.{cell}.json'):
                sha(Path(request['root']) / 'smokes' / f'{role}.{cell}.json')
                for role, cells in CELLS.items() for cell in cells}
    if value.get('smoke_receipts') != expected:
        raise ValueError('Shared B32 CUDA receipt hashes changed')
    return value


def launch_queues(request, request_file, session_file, context, audits):
    """Two serial queues, global fail-stop after any observed failure."""
    pending = {role: [cell for cell in cells if not audits[role][cell]['complete']]
               for role, cells in CELLS.items()}
    running, events = {}, []
    state_path = Path(request['root']) / 'executions' / (Path(session_file).stem + '.json')
    state = dict(schema=1, profile=PROFILE, request_sha256=request['request_sha256'],
                 session=str(session_file), started_utc=utc(), status='running', events=events)
    write_state(state_path, state)
    shared_session = scoped_session(request, request_file, session_file, context)
    try:
        while running or any(pending.values()):
            # Observe all peers before replacing any completed role. A peer
            # failure must not race an ordinary next-job scheduling decision.
            observed = {role: item['process'].poll() for role, item in running.items()}
            failed = {role: code for role, code in observed.items() if code not in (None, 0)}
            if failed:
                raise RuntimeError(f'Owned worker failed {failed}; no retry or later job on either role')
            for role, code in list(observed.items()):
                if code is None:
                    continue
                item = running[role]
                close_owned(item)
                result = audit_cell(request, request_file, session_file, role, item['cell'])
                if result['complete'] is not True:
                    raise ValueError(f'{role}/{item["cell"]} exited without authenticated scientific completion')
                events.append(dict(role=role, cell=item['cell'], exit=0, authenticated_complete=True, utc=utc()))
                del running[role]
                write_state(state_path, state)
            for role in CELLS:
                if role in running or not pending[role]:
                    continue
                # Poll active peers again immediately before each spawn.
                if any(item['process'].poll() not in (None, 0) for item in running.values()):
                    raise RuntimeError('Peer failed before replacement; no later job launched')
                context.require_scoped_gpu(request, shared_session)
                cell = pending[role].pop(0)
                running[role] = start_phase(request, request_file, session_file, role, 'run', cell)
                events.append(dict(role=role, cell=cell, started_utc=utc(),
                                   pid=running[role]['process'].pid, log=running[role]['log']))
                write_state(state_path, state)
            if running:
                time.sleep(1)
        # Re-audit all four, including skipped pre-existing completions, before
        # exporting through the same verified native source contexts.
        final = audit_all(request, request_file, session_file)
        if not all(item['complete'] for role in final.values() for item in role.values()):
            raise ValueError('Not all four cells have authenticated completed results')
        for role, cells in CELLS.items():
            item = start_phase(request, request_file, session_file, role, 'export', cells[-1])
            running[role] = item
            code = item['process'].wait()
            close_owned(item)
            if code:
                raise RuntimeError(f'{role} native export failed; scientific artifacts preserved')
            del running[role]
        state.update(status='complete', finished_utc=utc(), final_audits=final)
        write_state(state_path, state)
        print('COMPLETE_SHARED: A1 cross/nod + BadNets WiFlow/MetaFi++; at most one worker per role', flush=True)
        return state
    except BaseException as error:
        state.update(status='draining_after_failure', error=str(error), failure_utc=utc())
        try:
            write_state(state_path, state)
            safe_print('FAIL_STOP: no later job will start; draining only already-owned workers without kill', flush=True)
        except BaseException:
            safe_print('Diagnostic write failed; locks remain held while draining owned peers', flush=True)
        finally:
            codes = drain_owned(running)
        state.update(status='failed', drained_exits=codes, finished_utc=utc())
        try:
            write_state(state_path, state)
        except BaseException:
            safe_print('Owned peers have exited; final diagnostic write failed', flush=True)
        raise
    finally:
        for item in running.values():
            close_owned(item)


def status(values):
    """Read-only logs/records; no context import, Torch import or GPU query."""
    from collections import deque
    root = values['root']
    result = dict(root=str(root), request_present=(root / 'request.json').is_file(),
                  prepared_receipt_present=(root / 'prepared.json').is_file(),
                  interpretation='File/PID existence does not establish a live worker or valid result')
    records = sorted((root / 'executions').glob('*.json'), key=lambda p: p.stat().st_mtime)
    if records:
        result['latest_execution'] = read(records[-1])
    print(json.dumps(result, indent=2, allow_nan=False))
    logs = sorted((p for p in (root / 'logs').glob('*.log')
                   if p.name.startswith(('audit.', 'smoke.', 'run.', 'export.'))),
                  key=lambda p: p.stat().st_mtime)
    for path in logs[-4:]:
        print(f'LOG={path} MTIME_NS={path.stat().st_mtime_ns}', flush=True)
        with path.open(encoding='utf-8', errors='replace') as handle:
            print(''.join(deque(handle, maxlen=12)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['prepare', 'check', 'launch', 'status'])
    parser.add_argument('--home', required=True)
    parser.add_argument('--a1-repo', required=True)
    parser.add_argument('--badnets-repo', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--allow-concurrent-badnets', action='store_true')
    args = parser.parse_args(argv)
    values = paths(args)
    if args.mode == 'status':
        return status(values)
    if not args.allow_concurrent_badnets:
        raise ValueError('Explicit --allow-concurrent-badnets is required; no generic GPU bypass')
    context = load_context()
    with os_lock(values['home'] / 'tools' / '.mmfi_a1_badnets_shared_gpu0'):
        if args.mode == 'prepare':
            return prepare(values, context)
        request_file = values['root'] / 'request.json'
        request = context.read_request(request_file)
        for key, value in values.items():
            if Path(request[key]).resolve() != value:
                raise ValueError('Invocation differs from the immutable shared request')
        with native_locks(request):
            context.require_exclusive_idle()
            validate_prepared(request, context)
            session_file = session(request, context)
            context.validate_session(session_file, request)
            audits = audit_all(request, request_file, session_file)
            if args.mode == 'check':
                print('VERIFIED_SHARED_PRELAUNCH_GATE; no training launched', flush=True)
                return audits
            return launch_queues(request, request_file, session_file, context, audits)


if __name__ == '__main__':
    main()
