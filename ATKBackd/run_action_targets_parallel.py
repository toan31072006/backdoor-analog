"""Four new HPE-Li Cross/Nod victims, two owned jobs on Van's cuda:0.

MM-Fi Cross + Nod first, then PiW Cross + Nod. A slot includes training AND
immediate six-dose evaluation of that same fixed-final checkpoint. Separate
processes preserve independent model/optimizer/RNG state. Explicit Geometry,
Prepare (discarded concurrent B32 smokes), Launch/Resume. Never kills/retries
jobs, changes batch/AMP/recipes, imports old checkpoints or selects on TEST.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import ctypes
import importlib
import io
import math
import ntpath
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import run_mmfi_action_targets as common

HERE = Path(__file__).resolve().parent
PROFILE = 'cross_nod_mmfi_piw_hpeli_s42_two_jobs_v1'
PLAN = 'queue.plan.json'
DATASETS = ('mmfi', 'piw')
ACTIONS = ('cross', 'nod')
SLOTS = 2
POLL = 1.
HEARTBEAT = 30.
BARRIER_TIMEOUT = 600.


def module(dataset):
    if dataset not in DATASETS:
        raise ValueError('Unknown dataset')
    return importlib.import_module('run_' + dataset + '_action_targets')


def checked_root(outdir, data_home):
    root, home = Path(outdir).resolve(), Path(data_home).resolve()
    if (not re.fullmatch(r'cross_nod_mmfi_piw_s42_2jobs_v[0-9]+', root.name)
            or root.parent.name.lower() != 'runs' or root.is_relative_to(HERE)
            or home.is_relative_to(root)):
        raise ValueError('Use a NEW runs/cross_nod_mmfi_piw_s42_2jobs_v2 (or v3, ...), outside code/inputs; preserve old outputs')
    for path in (Path(outdir), Path(data_home)):
        for item in (path, *path.parents):
            if item.is_symlink() or (hasattr(item, 'is_junction') and item.is_junction()):
                raise ValueError('Linked output/data-home path refused: ' + str(item))
    return root, home


def settings(args):
    root, home = checked_root(args.outdir, args.data_home)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''):
        raise ValueError('Use physical cuda:0; no CUDA_VISIBLE_DEVICES remapping')
    wrapper = HERE.parent / 'remote_tools/action_targets_parallel/Run-CrossNodTwoJobs.ps1'
    value = dict(profile=PROFILE, outdir=str(root), data_home=str(home),
        python=str(Path(sys.executable).resolve()), sources=common.source_hashes(),
        wrapper_sha256=common.sha(wrapper),
        piw_dataset_root=str(Path(args.piw_dataset_root or home / 'datasets/Person-in-WiFi-3D_processed_v1').resolve()),
        cross_action=str(Path(args.cross_action or home / 'actions/data_cross.npy').resolve()),
        nod_action=str(Path(args.nod_action or home / 'actions/data_nod.npy').resolve()),
        approve_action_labels=bool(args.approve_action_labels),
        execution=dict(device='cuda:0', max_gpu_jobs=2, waves=list(DATASETS),
            actions=list(ACTIONS), evaluation='immediately after EACH job fixed-final training; slot retained through evaluation',
            parallel_capacity_gate='two independent real B32 TRAIN smokes per dataset; weights discarded',
            on_failure='no new jobs; drain already-owned peers, never kill or auto-retry',
            scientific_scope='EXPLORATORY_NEW_ACTION_TARGETS_NOT_FULL_T7',
            recipes=dict(mmfi='HPE-Li SGD.001 momentum.9 wd0 B32 FP32 workers4 50epochs final49',
                         piw='HPE-Li AdamW.001 wd.01 B32 FP32 workers0 drop_lastTrue 200epochs final199')))
    value['queue_sha256'] = common.digest(value)
    return value


def read_queue(root):
    q = common.read(Path(root) / PLAN)
    unsigned = {k: v for k, v in q.items() if k != 'queue_sha256'}
    if (q.get('profile') != PROFILE or common.digest(unsigned) != q.get('queue_sha256')
            or q.get('sources') != common.source_hashes()
            or q.get('python') != str(Path(sys.executable).resolve())):
        raise ValueError('Queue/source/Python changed; preserve the original checkout/output')
    args = SimpleNamespace(**q)
    if settings(args) != q:
        raise ValueError('Queue paths/parameters/wrapper changed')
    return q


def dataset_args(q, dataset, approve_geometry=False):
    name = f'{dataset}_action_targets_cross_nod_s42_v2'
    return SimpleNamespace(data_home=q['data_home'], outdir=str(Path(q['outdir']) / 'runs' / name),
        dataset_root=q['piw_dataset_root'] if dataset == 'piw' else None,
        cross_action=q['cross_action'], nod_action=q['nod_action'],
        approve_action_labels=q['approve_action_labels'], approve_geometry=approve_geometry,
        coordinator=dict(profile=PROFILE, queue_sha256=q['queue_sha256'],
            outdir=q['outdir'], dataset=dataset, max_gpu_jobs=2,
            evaluation='each selected fixed-final action immediately'))


def dataset_plan(q, dataset):
    m, args = module(dataset), dataset_args(q, dataset)
    return m.load_existing_plan(args) if dataset == 'mmfi' else m.load_plan(args)


def pid_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == 'nt':
        # os.kill(pid, 0) is NOT a portable Windows liveness check.
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        code = ctypes.c_ulong()
        try:
            ok = kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
            return bool(ok and code.value == 259)
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)  # POSIX signal zero: no signal is delivered.
        return True
    except ProcessLookupError:
        return False


def gpu_conflicts(text, owned, *, windows=None):
    from run_mmfi_p3_s3_three_attacks_4090 import active_compute
    windows = os.name == 'nt' if windows is None else windows
    conflicts = active_compute(text, windows, os.getpid())
    allowed_desktop = set()
    if windows:
        for env, suffix in [('ProgramFiles', 'TeamViewer/TeamViewer_Desktop.exe'),
                            ('SystemRoot', 'System32/WUDFHost.exe')]:
            if os.environ.get(env):
                allowed_desktop.add(ntpath.normcase(ntpath.normpath(ntpath.join(os.environ[env], suffix))))
    return [(pid, name) for pid, name in conflicts if pid not in owned and
            not (windows and ntpath.normcase(ntpath.normpath(name)) in allowed_desktop)]


def require_gpu(owned=()):
    reply = subprocess.run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=30)
    foreign = gpu_conflicts(reply.stdout, set(owned))
    if foreign:
        raise ValueError('Foreign/unknown GPU jobs active; nothing stopped: ' + repr(foreign))


def require_space(root):
    if shutil.disk_usage(root).free < 8 * 2**30:
        raise ValueError('Need at least8GiB free for four NEW outputs; nothing deleted')


def session_new(q):
    folder = Path(q['outdir']) / 'sessions' / uuid.uuid4().hex
    folder.mkdir(parents=True)
    common.write(folder / 'session.json', dict(queue_sha256=q['queue_sha256'],
        parent_pid=os.getpid(), directory=str(folder), token=folder.name))
    return folder


def session_check(q, session):
    session = Path(session).resolve()
    if session.parent != Path(q['outdir']) / 'sessions' or not re.fullmatch('[0-9a-f]{32}', session.name):
        raise ValueError('Foreign worker session')
    value = common.read(session / 'session.json')
    if (value.get('queue_sha256') != q['queue_sha256'] or value.get('directory') != str(session)
            or value.get('token') != session.name or not pid_alive(value.get('parent_pid'))):
        raise ValueError('Coordinator lease is gone/changed; preserve checkpoint and Resume explicitly')


def dataset_prepared_binding(q, dataset, *, full=False, backend=False):
    m, plan = module(dataset), dataset_plan(q, dataset)
    receipt = m.verify_prepared(plan, full_hash=full)
    if backend and m.execution() != receipt['execution']:
        raise ValueError('GPU/backend differs from Prepare: ' + dataset)
    root = Path(plan['outdir'])
    return dict(plan_sha256=plan['plan_sha256'], prepared_sha256=common.sha(root / 'prepared.json'),
        inputs_sha256=common.sha(root / 'inputs.json'), geometry_sha256=common.sha(root / 'geometry.json'))


def prepared_binding(q):
    record = common.read(Path(q['outdir']) / 'prepared.json')
    if record.get('queue_sha256') != q['queue_sha256']:
        raise ValueError('Parallel capacity receipt belongs to another queue')
    for dataset in DATASETS:
        if record['datasets'][dataset] != dataset_prepared_binding(q, dataset):
            raise ValueError('Dataset preparation changed: ' + dataset)
        capacity = record['capacity'][dataset]
        if len(capacity) != 2 or any(r.get('queue_sha256') != q['queue_sha256'] or
                r.get('stage') != 'smoke' or r.get('action') != action
                or r.get('dataset') != dataset for r, action in zip(capacity, ACTIONS)):
            raise ValueError('Missing two-process discarded capacity smoke: ' + dataset)
        if (len({r.get('pid') for r in capacity}) != 2 or any(
                r['result'].get('batch') != 32 or r['result'].get('poisoned') != 8
                or not r['result'].get('weights', '').startswith('discarded')
                or not math.isfinite(r['result'].get('loss', r['result'].get('finite_loss', float('nan'))))
                for r in capacity)):
            raise ValueError('Capacity receipt is not two finite independent discarded real B32 smokes')
    return record


def barrier(q, session, dataset, action, stage):
    session_check(q, session)
    folder = Path(session) / 'barriers' / dataset
    common.write(folder / f'{stage}.{action}.ready.json',
                 dict(queue_sha256=q['queue_sha256'], pid=os.getpid(), action=action, stage=stage))
    began = time.monotonic()
    while not (folder / f'{stage}.release.json').exists():
        session_check(q, session)
        if (Path(session) / 'abort.json').exists() or time.monotonic() - began > BARRIER_TIMEOUT:
            raise ValueError('Concurrent smoke barrier refused/timed out; no full training')
        time.sleep(.2)
    if common.read(folder / f'{stage}.release.json') != dict(queue_sha256=q['queue_sha256'], stage=stage):
        raise ValueError('Capacity barrier release changed')


def release_barriers(q, session, dataset):
    folder = Path(session) / 'barriers' / dataset
    for stage in ('allocated', 'updated'):
        paths = [folder / f'{stage}.{action}.ready.json' for action in ACTIONS]
        dest = folder / f'{stage}.release.json'
        if not dest.exists() and all(p.is_file() for p in paths):
            records = [common.read(p) for p in paths]
            if (any(r['queue_sha256'] != q['queue_sha256'] or r['stage'] != stage or
                    r['action'] != action for r, action in zip(records, ACTIONS))
                    or len({r['pid'] for r in records}) != 2):
                raise ValueError('Capacity barrier must contain two independent owned processes')
            common.write(dest, dict(queue_sha256=q['queue_sha256'], stage=stage))


def worker(q, args):
    from run_mmfi_tables import _cell_lock
    session_check(q, args.session)
    m = module(args.dataset)
    native_args = dataset_args(q, args.dataset, args.approve_geometry)
    stage, action = args.stage, args.action
    m.verify_bundle()
    if stage == 'geometry':
        root = Path(native_args.outdir)
        if (root / 'geometry.json').exists():
            m.verify_geometry(dataset_plan(q, args.dataset))
        else:
            m.geometry(native_args)
        result = dict(geometry='TRAIN_ONLY_NO_MODEL_PREDICTIONS')
    elif stage == 'prepare':
        plan = dataset_plan(q, args.dataset)
        if (Path(plan['outdir']) / 'prepared.json').exists():
            dataset_prepared_binding(q, args.dataset, full=True, backend=True)
        else:
            m.prepare(native_args, plan, idle_check=require_gpu)
        result = dataset_prepared_binding(q, args.dataset)
    elif stage == 'validate':
        result = dataset_prepared_binding(q, args.dataset, full=True, backend=True)
    else:
        if action not in ACTIONS or args.slot not in (0, 1):
            raise ValueError('Invalid fixed worker action/slot')
        plan = dataset_plan(q, args.dataset)
        cell = next(c for c in plan['cells'] if c['action'] == action)
        with _cell_lock(Path(q['outdir']) / 'slots' / str(args.slot)):
            # Old serial helpers are excluded by the coordinator's common GPU
            # lock. This cell lock additionally protects against orphan workers.
            with _cell_lock(Path(plan['outdir']) / action):
                session_check(q, args.session)
                m.verify_prepared(plan)
                if m.execution() != common.read(Path(plan['outdir']) / 'prepared.json')['execution']:
                    raise ValueError('Worker backend differs from discarded-smoke preparation')
                if stage == 'smoke':
                    result = m.smoke(plan, cell, rendezvous=lambda s: barrier(q, args.session, args.dataset, action, s))
                elif stage == 'job':
                    prepared_binding(q)
                    result = run_cell(q, plan, cell, m, args)
                else:
                    raise ValueError('Unknown worker stage')
    common.write(Path(args.session) / 'receipts' / f'{args.dataset}.{action}.{stage}.json',
        dict(queue_sha256=q['queue_sha256'], dataset=args.dataset, action=action,
             stage=stage, pid=os.getpid(), result=result))


def run_cell(q, plan, cell, m, args):
    folder = Path(plan['outdir']) / cell['action']
    phase = Path(q['outdir']) / 'status' / f'{args.dataset}.{cell["action"]}.json'
    def state(value):
        common.write(phase, dict(queue_sha256=q['queue_sha256'], pid=os.getpid(),
            dataset=args.dataset, action=cell['action'], phase=value,
            log=str(Path(args.session) / f'{args.dataset}.{cell["action"]}.job.log')), immutable=False)
    if (folder / 'complete.json').exists():
        report = m.verify_complete(plan, cell)
    else:
        session_check(q, args.session)
        if not args.evaluate_only:
            state('TRAINING')
            # No callback changes a batch, seed, optimizer step or checkpoint.
            # A killed coordinator is noticed between epochs, without killing
            # another process or guessing whether its lock is stale.
            model = m.train_cell(plan, cell, resume=args.resume,
                lease_check=lambda: session_check(q, args.session))
            del model
            import gc
            import torch
            gc.collect(); torch.cuda.empty_cache()
        session_check(q, args.session)
        state('EVALUATING_FIXED_FINAL_SIX_DOSES')
        report = m.evaluate_cell(plan, cell)
    state('COMPLETE')
    return dict(results_sha256=common.sha(folder / 'results.json'), final_epoch=report['final_epoch'])


def receipt(q, session, dataset, action, stage):
    value = common.read(Path(session) / 'receipts' / f'{dataset}.{action}.{stage}.json')
    if (value.get('queue_sha256') != q['queue_sha256'] or value.get('dataset') != dataset
            or value.get('action') != action or value.get('stage') != stage):
        raise ValueError('Child did not produce a bound completion receipt')
    return value


def spawn(q, session, task, slot, *, resume=False, evaluate_only=False, approve_geometry=False):
    dataset, action, stage = task
    command = [q['python'], '-B', '-u', str(Path(__file__).resolve()), '--outdir', q['outdir'],
        '--worker', '--dataset', dataset, '--action', action, '--stage', stage,
        '--session', str(session), '--slot', str(slot)]
    if resume: command.append('--resume')
    if evaluate_only: command.append('--evaluate-only')
    if approve_geometry: command.append('--approve-geometry')
    log = Path(session) / f'{dataset}.{action}.{stage}.log'
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=HERE, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            start_new_session=os.name != 'nt')
    process.queue_command = command
    process.queue_log = str(log)
    return process


def run_group(q, session, tasks, *, resume=False, evaluate_only=False, approve_geometry=False,
              spawn_fn=spawn, guard=require_gpu, on_complete=None, on_poll=None):
    """At most two child processes, including evaluation. Drain on ANY error.

    No process.kill/terminate, automatic retry, shell command construction or
    release of coordinator locks while owned children still run.
    """
    if not tasks or len(tasks) > SLOTS:
        raise ValueError('A wave has one or two tasks, never more')
    live, errors, results = {}, [], {}
    pending = list(enumerate(tasks))
    heartbeat = time.monotonic()
    while pending or live:
        try:
            while pending and not errors:
                slot, task = pending.pop(0)
                if task[2] != 'geometry': guard(tuple(p.pid for p, _ in live.values()))
                p = spawn_fn(q, session, task, slot, resume=resume, evaluate_only=evaluate_only,
                             approve_geometry=approve_geometry)
                live[slot] = (p, task)
                # Register the handle BEFORE a metadata write can fail. Even on
                # full disk/permission errors we must drain this owned child.
                if hasattr(p, 'queue_command'):
                    common.write(Path(session) / f'{task[0]}.{task[1]}.{task[2]}.process.json',
                        dict(pid=p.pid, command=p.queue_command, log=p.queue_log, queue_sha256=q['queue_sha256']))
                print(f'START {task}: pid={p.pid}; slot={slot+1}/2; logs={session}', flush=True)
            if on_poll is not None: on_poll()
            for slot, (p, task) in list(live.items()):
                code = p.poll()
                if code is None: continue
                del live[slot]
                common.write(Path(session) / f'{task[0]}.{task[1]}.{task[2]}.exit.json',
                    dict(pid=p.pid, exit_code=code, task=list(task), queue_sha256=q['queue_sha256']))
                if code:
                    raise RuntimeError(f'{task} exit={code}; preserve log {session}; no retry')
                value = receipt(q, session, *task)
                results[task] = value
                if on_complete is not None: on_complete(task, value)
                print(f'FINISH {task}; slot released AFTER bound evaluation/receipt', flush=True)
        except (Exception, KeyboardInterrupt) as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
            pending.clear()
            if any(t[2] == 'smoke' for _, t in live.values()):
                try:
                    common.write(Path(session) / 'abort.json', dict(queue_sha256=q['queue_sha256']))
                except (Exception, KeyboardInterrupt) as abort_error:
                    # Full/unwritable disk must not release coordinator locks
                    # with a resident peer. Its bounded barrier still exits.
                    errors.append(f'Abort receipt unavailable; draining peers: {abort_error}')
            print('STOP_NEW_JOBS; drain owned peers without killing: ' + errors[-1], flush=True)
            on_poll = None
        if time.monotonic() - heartbeat >= HEARTBEAT:
            print(f'HEARTBEAT owned={[p.pid for p, _ in live.values()]}; errors={len(errors)}; logs={session}', flush=True)
            heartbeat = time.monotonic()
        if live:
            try:
                time.sleep(POLL)
            except KeyboardInterrupt:
                errors.append('KeyboardInterrupt: no new jobs; waiting for owned peers without killing')
                pending.clear()
                print(errors[-1], flush=True)
    if errors:
        raise RuntimeError('\n'.join(errors))
    return results


def export_snapshot(q, *, final=False):
    rows = []
    for dataset in DATASETS:
        m, plan = module(dataset), dataset_plan(q, dataset)
        for cell in plan['cells']:
            if not (Path(plan['outdir']) / cell['action'] / 'complete.json').exists(): continue
            report = m.verify_complete(plan, cell)
            met = report['metrics']; end = met['doses'][-1]
            payload = cell['cfg']['action_payload'] if dataset == 'mmfi' else cell['cfg']['payload']
            rows.append(dict(dataset=dataset, action=cell['action'], target_joints=str(payload['target_joints']),
                final_epoch=report['final_epoch'], t_mpjpe_mm=end['t_mpjpe_mm'],
                target_pck_pct=end['target_pck_pct'], mean_positive_dce_mm=met['mean_positive_dce_mm'],
                clean_mpjpe_mm=met['clean_mpjpe_mm'], clean_pampjpe_mm=met['clean_pampjpe_mm']))
    if final and len(rows) != 4:
        raise ValueError('Need four complete jobs; partial rows are NOT a completed table')
    buffer = io.StringIO(newline='')
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    root = Path(q['outdir']); prefix = 'results' if final else 'partial_results'
    common.write(root / (prefix + '.json'), dict(queue_sha256=q['queue_sha256'],
        complete=len(rows), expected=4, scope='single-seed new-target extension, not complete T7', rows=rows), immutable=final)
    path = root / (prefix + '.csv')
    data = buffer.getvalue().encode('utf-8')
    if final and path.exists():
        if path.read_bytes() != data: raise ValueError('Existing final CSV differs; preserve it')
    else:
        temporary = path.with_name(path.name + '.saving')
        with temporary.open('xb') as stream: stream.write(data)
        os.replace(temporary, path)
    print(buffer.getvalue() or 'NO COMPLETED EVALUATIONS YET', flush=True)


def check_only(args):
    q = settings(args)  # No output writes or scientific imports.
    for dataset in DATASETS:
        module(dataset).check_paths(dataset_args(q, dataset))
    print('CHECK_ONLY: HPE-Li, four victims; 2 MM-Fi slots then2 PiW slots; immediate per-job evaluation. No jobs started.')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['check-only', 'geometry', 'prepare', 'launch', 'resume', 'evaluate', 'status', 'export'])
    p.add_argument('--data-home'); p.add_argument('--outdir', required=True)
    p.add_argument('--piw-dataset-root'); p.add_argument('--cross-action'); p.add_argument('--nod-action')
    p.add_argument('--approve-action-labels', action='store_true'); p.add_argument('--approve-geometry', action='store_true')
    # Private worker dispatch; never a recipe/epoch/batch/GPU-count override.
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--dataset', choices=DATASETS, help=argparse.SUPPRESS)
    p.add_argument('--action', choices=(*ACTIONS, 'all'), default='all', help=argparse.SUPPRESS)
    p.add_argument('--stage', choices=['geometry', 'prepare', 'validate', 'smoke', 'job'], help=argparse.SUPPRESS)
    p.add_argument('--session', help=argparse.SUPPRESS); p.add_argument('--slot', type=int, help=argparse.SUPPRESS)
    p.add_argument('--resume', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--evaluate-only', action='store_true', help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker:
        return worker(read_queue(args.outdir), args)
    if not args.mode or not args.data_home:
        raise ValueError('--mode and --data-home required')
    if args.mode == 'check-only': return check_only(args)
    expected = settings(args)
    root = Path(expected['outdir'])
    if args.mode == 'status':
        if not (root / PLAN).exists():
            print('NOT_GEOMETRY_PREPARED; no output created'); return
        q = read_queue(root)
        for dataset in DATASETS:
            plan = dataset_plan(q, dataset)
            for action in ACTIONS:
                state = root / 'status' / f'{dataset}.{action}.json'
                print(f'{dataset}/{action}: ' + (str(common.read(state)) if state.exists() else 'NOT_STARTED'))
        print('Status is saved phase metadata, not proof that a PID or GPU process is still alive.')
        return
    if (root.exists() and any(p.name != 'run.lock' for p in root.iterdir()) and not (root / PLAN).exists()):
        raise ValueError('Nonempty foreign output; nothing overwritten')
    from run_mmfi_tables import _cell_lock
    with ExitStack() as locks:
        locks.enter_context(_cell_lock(root))
        if args.mode == 'geometry':
            common.write(root / PLAN, expected)
        q = read_queue(root)
        if q != expected: raise ValueError('Use the identical geometry-approved settings for all modes')
        # Detect an orphan even before it has created its CUDA context.
        with ExitStack() as slots:
            for slot in range(2): slots.enter_context(_cell_lock(root / 'slots' / str(slot)))
        if args.mode == 'export':
            prepared_binding(q); return export_snapshot(q, final=True)
        if args.mode != 'geometry':
            locks.enter_context(_cell_lock(Path(q['data_home']) / 'runs/.mmfi_action_targets_gpu0'))
            require_gpu(); require_space(root)
        # Prevent an old serial entrypoint from writing either dataset plan.
        for dataset in DATASETS:
            locks.enter_context(_cell_lock(dataset_args(q, dataset).outdir))
        session = session_new(q)
        if args.mode == 'geometry':
            for dataset in DATASETS: run_group(q, session, [(dataset, 'all', 'geometry')])
            print('GEOMETRY_ONLY: inspect BOTH datasets geometry_cross/nod.png; Prepare needs --approve-geometry.')
        elif args.mode == 'prepare':
            if not args.approve_geometry:
                raise ValueError('Inspect TRAIN geometry for BOTH datasets; explicit --approve-geometry required')
            if (root / 'prepared.json').exists():
                raise ValueError('Already parallel-prepared; use Launch or explicit Resume')
            for dataset in DATASETS:
                run_group(q, session, [(dataset, 'all', 'prepare')], approve_geometry=True)
            capacity = {}
            for dataset in DATASETS:
                results = run_group(q, session, [(dataset, a, 'smoke') for a in ACTIONS],
                    on_poll=lambda d=dataset: release_barriers(q, session, d))
                capacity[dataset] = [results[(dataset, a, 'smoke')] for a in ACTIONS]
            common.write(root / 'prepared.json', dict(queue_sha256=q['queue_sha256'],
                datasets={d: dataset_prepared_binding(q, d) for d in DATASETS}, capacity=capacity,
                interpretation='two-process real TRAIN capacity smoke only; no guarantee against later OOM or efficacy'))
            print('PARALLEL_PREPARED_ONLY: no full training launched; all smoke weights discarded.')
        else:
            prepared_binding(q)
            if args.mode == 'launch':
                for dataset in DATASETS:
                    for action in ACTIONS:
                        if (Path(dataset_args(q, dataset).outdir) / action / 'checkpoint.pt').exists():
                            raise ValueError('Checkpoint exists; inspect Status then explicitly Resume, not Launch')
            for dataset in DATASETS:
                run_group(q, session, [(dataset, 'all', 'validate')])
                run_group(q, session, [(dataset, a, 'job') for a in ACTIONS], resume=args.mode == 'resume',
                    evaluate_only=args.mode == 'evaluate', on_complete=lambda t, r: export_snapshot(q))
                module(dataset).export(dataset_plan(q, dataset))
            export_snapshot(q, final=True)
            print('FOUR_OF_FOUR_COMPLETE: true GT/target/prediction arrays retained for qualitative plots.')


if __name__ == '__main__':
    main()
