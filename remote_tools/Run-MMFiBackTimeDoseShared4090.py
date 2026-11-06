"""Approved execution amendment: resume P3S3 + sequential A1 1%, then5%.

Old scientific sources/plans/gates/checkpoints remain bound and unchanged.
Only explicit Pause/SwitchAndLaunch can pause the authenticated original PID.
Prepare does concurrent real-input capacity smokes, not production training.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', MKL_THREADING_LAYER='SEQUENTIAL')
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'ATKBackd'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mmfi_low_poison_shared as dose
import mmfi_shared_pause as pause_tools

PROFILE = 'mmfi_shared_backtime_dose_s42_4090_v1'
ROLES = ('backtime', 'dose')
DEFAULT_OLD = Path(r'U:\backdooranalog\runs\mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_u_v1')
MODES = ('inspect', 'pause', 'prepare', 'launch', 'switch-and-launch', 'status')
SCRIPT_FILES = ('Run-MMFiBackTimeDoseShared4090.py','mmfi_shared_pause.py',
                'mmfi_low_poison_shared.py','MMFI_A1_LOW_POISON_REFERENCE.json',
                'Run-MMFiBackTimeTwoSplits4090.py')


def original_adapter():
    name = '_original_backtime_'+uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name,
        Path(__file__).with_name('Run-MMFiBackTimeTwoSplits4090.py'))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def old_contract(original, home, old_root):
    """Controller is CPU-only; only owned workers verify actual CUDA execution."""
    r = original.load_runtime('p3s3')
    print('CHECK_OLD_SOURCE_INPUTS_AND_P1S1_RECEIPTS; CPU-only, no job stopped', flush=True)
    r.reject_links(old_root)
    master = r.read(old_root/'two_splits.plan.json')
    if master != original.master_plan(home, old_root):
        raise ValueError('Old BackTime plan/source/Python runtime differs; do NOT update the active checkout')
    env = original.gate_check(r, master['plans']['p3s3'], old_root, verify_execution=False)
    p1 = original.load_runtime('p1s1')
    p1.verify_inputs(master['plans']['p1s1'])
    if p1.completed(master['plans']['p1s1'], 'backtime', env) is None:
        raise ValueError('P1S1 is not authentically complete; do not pause the old pair')
    print('P1S1_COMPLETE_VERIFIED; its training will not run again', flush=True)
    return r, master, env


def build_plan(r, master, home, old_root, output):
    r.reject_links(output)
    output, old_root = output.resolve(), old_root.resolve()
    if (output.parent.name.lower() != 'runs' or not output.name.startswith('mmfi_shared_')
            or output == old_root or output.is_relative_to(old_root)
            or old_root.is_relative_to(output)):
        raise ValueError('Use a separate runs/mmfi_shared_* root; never nest in the old run')
    value = dict(schema=1, profile=PROFILE, data_home=str(home), old_root=str(old_root),
        root=str(output), old_master_sha256=master['sha256'],
        old_gate_sha256=r.sha(old_root/'two_splits.gate.json'),
        old_inputs_sha256={s:r.sha(old_root/s/'inputs.json') for s in ('p1s1','p3s3')},
        execution_sources={f:r.sha(Path(__file__).with_name(f)) for f in SCRIPT_FILES},
        cells=dose.cells(home, output), roles=list(ROLES),
        backtime='native P3S3 resume; P1S1 stays complete; never import or restart',
        dose='P1S1 A1 seed42: one worker, rho.01 then.05, native50/final49/SGD/B32/workers4',
        limitations=['One seed only; no across-seed uncertainty.',
            'No pooling P1S1 with P3S3.', 'Concurrent capacity is not guaranteed by GPU utilization.',
            'Partial BackTime epoch may replay from its latest committed state.',
            'New A1 victims are scratch; BackTime is not scratch.'])
    value['sha256'] = r.digest(value)
    return value


def bind_plan(r, plan):
    root = Path(plan['root'])
    path = root/'shared.plan.json'
    if not path.exists():
        unexpected = [p.name for p in root.iterdir() if p.name != '.controller']
        if unexpected:
            raise ValueError(f'Unbound output contains artifacts; preserve it: {unexpected}')
    r.write(path, plan)


def pause_receipt(r, plan):
    row = r.read(Path(plan['root'])/'pause.json')
    if (row.get('status') != 'PAUSED_WITH_VALID_NATIVE_CHECKPOINT'
            or row.get('old_master_sha256') != plan['old_master_sha256']
            or not row.get('before', {}).get('backup')
            or r.sha(row['before']['backup']) != row['before']['sha256']):
        raise ValueError('Missing/changed authorized pause backup/receipt')
    return row


def barrier(r, context, stage, role, timeout=1800):
    directory = Path(context['directory'])
    (directory/f'{stage}.{role}.ready').touch(exist_ok=False)
    began = time.monotonic()
    print(f'SHARED_CAPACITY_BARRIER {role} {stage}', flush=True)
    while not all((directory/f'{stage}.{s}.ready').is_file() for s in ROLES):
        if (directory/'peer_failed.json').exists():
            raise RuntimeError('Owned capacity peer failed; no retry or batch change')
        if time.monotonic()-began > timeout:
            raise TimeoutError('Owned capacity peer timed out; inspect both logs')
        time.sleep(.2)


def read_context(r, filename, role, plan, action):
    began = time.monotonic()
    while not filename.is_file():
        if time.monotonic()-began > 30:
            raise TimeoutError('No parent-created shared worker context')
        time.sleep(.1)
    r.reject_links(filename)
    ctx = r.read(filename)
    unsigned = dict(ctx)
    claimed = unsigned.pop('sha256', None)
    if (claimed != r.digest(unsigned) or ctx.get('profile') != PROFILE
            or ctx.get('parent') != os.getppid() or ctx.get('plan_sha256') != plan['sha256']
            or ctx.get('action') != action or ctx.get('root') != plan['root']
            or ctx.get('sources') != plan['execution_sources']
            or set(ctx.get('workers',{})) != set(ROLES)
            or len(set(ctx['workers'].values())) != 2 or ctx['workers'][role] != os.getpid()
            or filename != Path(ctx.get('directory',''))/'workers.json'
            or filename.parent.parent != Path(plan['root'])/'executions'):
        raise ValueError('Foreign/stale shared context; no general GPU-PID bypass')
    return ctx


def gate_check(r, plan, env):
    path = Path(plan['root'])/'shared.gate.json'
    row = r.read(path)
    if (row.get('plan_sha256') != plan['sha256'] or row.get('execution') != env
            or row.get('pause_sha256') != r.sha(Path(plan['root'])/'pause.json')
            or row.get('status') != 'CONCURRENT_BACKTIME_A1_SMOKES_PASSED_NOT_TRAINED'
            or set(row.get('smokes',{})) != set(ROLES)):
        raise ValueError('Missing/changed shared capacity gate')
    pause_receipt(r, plan)
    for role, smoke in row['smokes'].items():
        if (smoke.get('discarded') is not True or smoke.get('real_train_input') is not True
                or smoke.get('status') != 'CAPACITY_ONLY_NOT_TRAINED'
                or smoke.get('execution') != env
                or len(smoke.get('losses',[])) != 2
                or not all(math.isfinite(x) for x in smoke['losses'])):
            raise ValueError(f'Invalid shared smoke receipt: {role}')
        recipe = (dict(batch=32,steps=2,poison_examples=4) if role == 'dose' else
            dict(source_attacker_batch=64,victim_batch=32,victim_steps=2,
                 source_smoke_epochs=2,source_smoke_frames=64,generator_steps=6,
                 production_attacker_epochs=100,production_warmup=10))
        if any(smoke.get(k) != v for k,v in recipe.items()):
            raise ValueError(f'Shared capacity smoke changed recipe: {role}')
        if role == 'backtime':
            old = r.read(Path(plan['old_root'])/'two_splits.plan.json')['plans']['p3s3']
            if (smoke.get('plan_sha256') != old['plan_sha256']
                    or smoke.get('binding') != r.preparation_binding(old,env)):
                raise ValueError('Shared BackTime smoke input/plan binding changed')
    return row


def run_dose_queue(original, r, plan, ctx):
    for cell in plan['cells']:
        if (Path(ctx['directory'])/'peer_failed.json').exists():
            raise RuntimeError('BackTime peer failed; pending dose cell not started')
        original.require_idle(ctx['workers'].values())
        with r.log_to(cell['ckpt_dir'], 'dose.'+cell['key']):
            dose.execute(cell, plan['sha256'], ctx['directory'])


def worker(args):
    original = original_adapter()
    r, master, old_env = old_contract(original, args.data_home, args.backtime_root)
    plan = build_plan(r, master, args.data_home, args.backtime_root, args.outdir)
    if r.read(args.outdir/'shared.plan.json') != plan:
        raise ValueError('Shared plan differs from the immutable source/input contract')
    action = 'prepare' if args.mode == '_smoke' else 'launch'
    ctx = read_context(r, args.context, args.role, plan, action)
    original.require_idle(ctx['workers'].values())
    env = r.execution()  # Only the owned GPU children reach this line.
    if env != old_env:
        raise ValueError('GPU/runtime differs from the original BackTime training')
    if action == 'prepare':
        if args.role == 'backtime':
            original.barrier = lambda context, stage, split: barrier(r, ctx, stage, 'backtime')
            from run_mmfi_tables import _cell_lock
            with _cell_lock(args.backtime_root/'p3s3/.controller'):
                original.smoke(r, master['plans']['p3s3'], ctx, 'p3s3')
            row = r.read(Path(ctx['directory'])/'smoke.p3s3.json')
        else:
            row = dose.smoke(plan['cells'][0], lambda stage: barrier(r, ctx, stage, 'dose'))
            row['execution'] = env
        r.write(Path(ctx['directory'])/f'smoke.{args.role}.json', row)
        return
    gate_check(r, plan, env)
    if args.role == 'backtime':
        from run_mmfi_tables import _cell_lock
        with _cell_lock(args.backtime_root/'p3s3/.controller'):
            point = pause_tools.checkpoint(r, master['plans']['p3s3'], env)
            print('NATIVE_P3S3_RESUME_ONLY '+json.dumps(point), flush=True)
            r.require_idle = lambda: original.require_idle(ctx['workers'].values())
            r.gate_check = lambda p: original.gate_check(r, p, args.backtime_root)
            original.run_pipeline(r, master['plans']['p3s3'])
    else:
        run_dose_queue(original, r, plan, ctx)


def drain(children):
    # Never free a root lock or kill healthy owned peers on error/Ctrl+C.
    for process in children.values():
        while process.poll() is None:
            print(f'DRAINING_OWNED_PEER pid={process.pid}; no automatic kill/retry', flush=True)
            try:
                process.wait()
            except KeyboardInterrupt:
                print('Root locks retained while owned peers run.', flush=True)


def spawn_pair(args, r, plan, action, env, master, original):
    directory = args.outdir/'executions'/(action+'_'+uuid.uuid4().hex)
    r.reject_links(directory)
    directory.mkdir(parents=True, exist_ok=False)
    children, streams = {}, []
    filename = directory/'workers.json'
    try:
        for role in ROLES:
            out = (directory/f'{role}.stdout.log').open('x', encoding='utf-8')
            err = (directory/f'{role}.stderr.log').open('x', encoding='utf-8')
            streams += [out,err]
            command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--mode',
                '_smoke' if action == 'prepare' else '_worker', '--data-home', str(args.data_home),
                '--backtime-root', str(args.backtime_root), '--outdir', str(args.outdir),
                '--role', role, '--context', str(filename)]
            children[role] = subprocess.Popen(command, cwd=REPO, stdout=out, stderr=err,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0))
        ctx = dict(profile=PROFILE, parent=os.getpid(), root=str(args.outdir),
            directory=str(directory), workers={s:p.pid for s,p in children.items()},
            action=action, plan_sha256=plan['sha256'], sources=plan['execution_sources'])
        ctx['sha256'] = r.digest(ctx)
        r.write(filename, ctx)
        print('SHARED_WORKERS '+json.dumps(ctx['workers'])+' LOGS='+str(directory), flush=True)
        reported = set()
        while len(reported) < len(children):
            for role,p in children.items():
                if p.poll() is not None and role not in reported:
                    print(f'SHARED_EXIT {role}={p.returncode}; no retry', flush=True)
                    if p.returncode and not (directory/'peer_failed.json').exists():
                        r.write(directory/'peer_failed.json', dict(role=role, exit_code=p.returncode))
                    reported.add(role)
            if len(reported) < len(children):
                time.sleep(1)
        codes = {s:p.wait() for s,p in children.items()}
        if any(codes.values()):
            raise RuntimeError(f'Shared workers failed: {codes}. Preserve both logs/checkpoints: {directory}')
        if action == 'prepare':
            rows = {s:r.read(directory/f'smoke.{s}.json') for s in ROLES}
            r.write(args.outdir/'shared.gate.json', dict(plan_sha256=plan['sha256'], execution=env,
                pause_sha256=r.sha(args.outdir/'pause.json'), smokes=rows,
                status='CONCURRENT_BACKTIME_A1_SMOKES_PASSED_NOT_TRAINED'))
            gate_check(r, plan, env)
            print('SHARED_PREPARED_ONLY; both real-input capacity smokes discarded; no full training.', flush=True)
        else:
            for split in ('p1s1','p3s3'):
                rr = original.load_runtime(split)
                if rr.completed(master['plans'][split], 'backtime', env) is None:
                    raise ValueError('BackTime split result incomplete after owned workers exited')
            backtime = {s:r.read(args.backtime_root/s/'results.json') for s in ('p1s1','p3s3')}
            r.write(args.backtime_root/'results.json', dict(status='complete',
                master_sha256=master['sha256'], splits=backtime))
            evaluations = {cell['key']:r.read(Path(cell['artifact_dir'])/'evaluation.json') for cell in plan['cells']}
            r.write(args.outdir/'results.json', dict(status='complete', plan_sha256=plan['sha256'],
                backtime=backtime, dose=evaluations))
            print('SHARED_COMPLETE: P3S3 resumed; P1S1 preserved; A1 P1S1 1%+5% complete. Not pooled.', flush=True)
    finally:
        drain(children)
        for stream in streams:
            stream.close()


def run(args):
    original = original_adapter()
    r, master, env = old_contract(original, args.data_home, args.backtime_root)
    plan = build_plan(r, master, args.data_home, args.backtime_root, args.outdir)
    if args.mode in ('inspect','status'):
        phase, path = pause_tools.checkpoint_choice(master['plans']['p3s3'])
        point = dict(phase=phase,path=str(path),
            epoch='Not opened while live; full validation only after authorized temporary hold')
        pids = pause_tools.discover() if os.name == 'nt' else []
        print(json.dumps(dict(backtime_root=str(args.backtime_root), original_p3s3_pids=pids,
            resume=point, p1s1='COMPLETE_VERIFIED', shared_root=str(args.outdir),
            dose=[dict(rho=c['rho'], n_poison=c['n_poison'], seed=42, protocol='P1S1') for c in plan['cells']]), indent=2))
        print('READ_ONLY: no pause, CUDA initialization or training.')
        return
    from run_mmfi_tables import _cell_lock
    original.check_output_storage(args.outdir)
    # Validate dose identities BEFORE any pause, without loading a training model.
    for cell in plan['cells']:
        print(f'CHECK_T2_OLD_RECIPE_INPUTS rho={cell["rho"]}; no full training', flush=True)
        dose.audit_input(cell)
    with _cell_lock(args.outdir/'.controller'):
        bind_plan(r, plan)
        if args.mode in ('pause','switch-and-launch'):
            pids = pause_tools.discover()
            if len(pids) != 1:
                raise ValueError(f'Expected exactly one original P3S3 worker, found {pids}; nothing stopped')
            original.require_idle(pids)
            pause_tools.pause(r, original, master, args.backtime_root, args.data_home,
                              args.outdir, env, pids[0])
            if args.mode == 'pause':
                return
        pause_receipt(r, plan)
        original.require_idle()
        # Reject any old or duplicate controller; do not remove stale lock files.
        with _cell_lock(args.backtime_root/'.controller'):
            pause_tools.checkpoint(r, master['plans']['p3s3'], env)
            if args.mode in ('prepare','switch-and-launch'):
                if (args.outdir/'shared.gate.json').exists():
                    gate_check(r, plan, env)
                else:
                    spawn_pair(args, r, plan, 'prepare', env, master, original)
                if args.mode == 'prepare':
                    return
            gate_check(r, plan, env)
            point = pause_tools.checkpoint(r, master['plans']['p3s3'], env)
            directory = args.outdir/'launch_resume_points'
            r.write(directory/(uuid.uuid4().hex+'.json'), point)
            return spawn_pair(args, r, plan, 'launch', env, master, original)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES+('_worker','_smoke'), required=True)
    parser.add_argument('--data-home', type=Path, default=Path.home()/'backdooranalog')
    parser.add_argument('--backtime-root', type=Path, default=DEFAULT_OLD)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--context', type=Path)
    args = parser.parse_args(argv)
    original = original_adapter()
    for path in (args.data_home, args.backtime_root, args.outdir):
        if path is not None:
            original.load_runtime('p3s3').reject_links(path)
    args.data_home = args.data_home.resolve()
    args.backtime_root = args.backtime_root.resolve()
    args.outdir = (args.outdir or args.backtime_root.parent/PROFILE).resolve()
    if args.mode.startswith('_'):
        if args.role is None or args.context is None:
            parser.error('Internal workers require an owned role/context')
        return worker(args)
    if args.role is not None or args.context is not None:
        parser.error('No public foreign-PID/role sharing override')
    return run(args)


if __name__ == '__main__':
    main()
