"""Fresh BackTime-only HPE-Li P1S1/P3S3, two owned workers on Van's RTX4090.

Execution adapter only. The published streaming attacker, victim optimizer,
target, checkpoint and evaluation functions are reused, not reimplemented.
Prepare and Launch are separate. No checkpoint import, kill or automatic retry.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', MKL_THREADING_LAYER='SEQUENTIAL')
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'ATKBackd'))
PROFILE = 'mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_v1'
BASE_COMMIT = '798723db02f93edb4312b6f3bde9f2f450d99262'
SPLITS = {'p1s1': ('protocol1', 's1', 133056, 33264),
          'p3s3': ('protocol3', 's3', 240570, 80190)}
MODES = ('inspect', 'preflight', 'prepare', 'check', 'launch', 'status')


def load_runtime(split):
    if split not in SPLITS:
        raise ValueError('Only the requested P1S1 and P3S3 splits are allowed')
    source = REPO / 'ATKBackd/run_mmfi_p3s3_siba_ladder_4090.py'
    name = '_backtime_twosplits_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, source)
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[name] = runtime
    spec.loader.exec_module(runtime)
    runtime.COUNTS = dict(zip(('training', 'test'), SPLITS[split][2:]))
    return runtime


def build_plan(runtime, home, root, split):
    home, root = Path(home), Path(root)
    for p in (home, root):
        runtime.reject_links(p)
    home, root = home.resolve(), root.resolve()
    if root.parent != home / 'runs' or not root.name.startswith('mmfi_backtime_'):
        raise ValueError('Use a NEW mmfi_backtime_* root directly under data-home/runs')
    # Reuse published builder defaults as data, without preparing its old queue.
    legacy = home / 'runs/mmfi_p3s3_siba_ladder_plan_template_only'
    plan = runtime.build_plan(home, legacy, backtime_policy='adaptive',
                              selected_methods=('siba', 'backtime'))
    protocol, setting, train_n, test_n = SPLITS[split]
    plan.update(profile=PROFILE + '_' + split, outdir=str(root / split),
                methods=['backtime'], cells=['clean_surrogate', 'backtime'])
    plan['cfg'] = {k: v for k, v in plan['cfg'].items() if k in plan['cells']}
    for cfg in plan['cfg'].values():
        cfg.update(mmfi_protocol=protocol, mmfi_setting=setting,
                   learned_baseline_profile=plan['profile'])
    plan['operators'] = {'backtime': plan['operators']['backtime']}
    contract = plan['contract']
    contract.update(protocol=protocol, setting=setting,
                    counts=dict(training=train_n, test=test_n),
                    n_poison=math.floor(.1 * train_n), fit_train_frames=train_n,
                    status='EXPLORATORY_BACKTIME_ONLY_SINGLE_SEED_FRESH_4090')
    contract['limitations'] = [x for x in contract['limitations']
                               if not x.startswith(('SIBA ', 'LADDER '))]
    contract['limitations'].append(
        'P1S1 is a per-action subject split, not globally unseen-subject generalization; '
        'P3S3 holds out E04 and subjects31..40. No pooling of these TEST sets.')
    plan['source_readiness'] = dict(selected_methods=['backtime'],
        explicitly_not_requested=['siba', 'ladder'], no_automatic_substitution=True)
    plan['execution_adapter'] = dict(profile=PROFILE, base_commit=BASE_COMMIT,
        script_sha256=runtime.sha(__file__), scientific_engine_sha256=runtime.sha(
            REPO / 'ATKBackd/run_mmfi_p3s3_siba_ladder_4090.py'),
        attacker_sha256=runtime.sha(REPO / 'ATKBackd/attack/backtime_mmfi.py'),
        gpu='NVIDIA GeForce RTX 4090', owned_workers=2, split=split,
        checkpoint_import=False, warm_start=False, attacker_algorithm_changed=False)
    plan.pop('plan_sha256')
    plan['plan_sha256'] = runtime.digest(plan)
    return plan


def master_plan(home, root):
    plans = {s: build_plan(load_runtime(s), home, root, s) for s in SPLITS}
    from run_mmfi_p3s3_siba_ladder_4090 import digest
    result = dict(profile=PROFILE, schema=1, workers=2, device='cuda:0',
                  fresh_initial_run=True, plans=plans)
    result['sha256'] = digest(result)
    return result


def require_idle(allowed=()):
    from run_mmfi_p3_s3_three_attacks_4090 import active_compute
    report = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'], capture_output=True, text=True,
        check=True, timeout=30)
    conflicts = [(pid, name) for pid, name in active_compute(
        report.stdout, os.name == 'nt', os.getpid()) if pid not in set(allowed)]
    if conflicts:
        raise ValueError(f'Other or unknown GPU jobs active; nothing stopped: {conflicts}')


def bind_master(runtime, root, plan):
    root = Path(root)
    path = root / 'two_splits.plan.json'
    runtime.reject_links(path)
    if not path.exists():
        unexpected = [p.name for p in root.iterdir() if p.name != '.controller']
        if unexpected:
            raise ValueError(f'Nonempty unbound output; preserve it and choose a new root: {unexpected}')
    runtime.write(path, plan)


def barrier(context, stage, split, timeout=1800):
    path = Path(context['directory']) / f'{stage}.{split}.ready'
    path.touch(exist_ok=False)
    print(f'CAPACITY_BARRIER {stage} {split}; waiting for owned peer', flush=True)
    began = time.monotonic()
    while True:
        if (path.parent / 'peer_failed.json').exists():
            raise RuntimeError('Owned capacity peer failed; inspect both logs; no retry')
        if all((path.parent / f'{stage}.{s}.ready').is_file() for s in SPLITS):
            print(f'CAPACITY_BARRIER_PASSED {stage} {split}', flush=True)
            return
        if time.monotonic() - began > timeout:
            raise TimeoutError(f'Peer did not reach {stage}; inspect both logs; no retry')
        time.sleep(.2)


def read_context(runtime, filename, own_split, plan):
    deadline = time.monotonic() + 30
    while not Path(filename).is_file():
        if time.monotonic() > deadline:
            raise TimeoutError('Parent did not provide the owned-worker context')
        time.sleep(.1)
    runtime.reject_links(filename)
    ctx = runtime.read(filename)
    unsigned = dict(ctx)
    claimed = unsigned.pop('sha256', None)
    if (claimed != runtime.digest(unsigned) or ctx.get('profile') != PROFILE
            or ctx.get('parent') != os.getppid()
            or set(ctx.get('workers', {})) != set(SPLITS)
            or len(set(ctx['workers'].values())) != 2
            or ctx['workers'][own_split] != os.getpid()
            or ctx.get('plan_hashes', {}).get(own_split) != plan['plan_sha256']
            or ctx.get('script_sha256') != runtime.sha(__file__)
            or Path(filename).parent != Path(ctx['directory'])
            or Path(ctx['root']) != Path(plan['outdir']).parent):
        raise ValueError('Foreign/stale worker context; no GPU-sharing bypass')
    return ctx


def smoke(runtime, plan, context, split):
    """Capacity check only: real TRAIN, source B64 backward and generator steps.

    Two shortened epochs on64 frames are discarded, NOT a production search.
    Production BackTimeConfig stays100/10/B64 and is in the immutable plan.
    """
    import numpy as np
    import torch
    import attack.backtime_mmfi as bt
    from attack.poison import collate
    from train_backdoor import _victim_update
    env = runtime.execution()
    runtime.verify_inputs(plan)
    data = runtime.dataset(plan, 'training')
    model, unused = runtime.new_components(plan, 'backtime')
    del unused
    class RealRows(torch.utils.data.Dataset):
        split = 'training'
        def __len__(self): return 64
        def __getitem__(self, i): return data[i]
    config = replace(bt.BackTimeConfig(**plan['operators']['backtime']),
                     fit_epochs=2, warmup=0)
    torch.cuda.reset_peak_memory_stats(0)
    barrier(context, 'models_resident', split)
    def progress(row):
        print('[capacity-only] ' + json.dumps(row), flush=True)
        if row.get('epoch_index') == 0:
            barrier(context, 'B64_backward_resident', split)
    trigger = bt.fit_backtime_stream(model, RealRows(), runtime.target_fn, config,
        dataset_sha256=runtime.verify_inputs(plan)['full_content_sha256'],
        n_poison=6, doses=torch.linspace(.2, 1., 64), progress=progress)
    if trigger.config != config:
        raise ValueError('Unexpected smoke configuration')
    del trigger, model
    torch.cuda.empty_cache()
    model, optimizer = runtime.new_components(plan, 'clean_surrogate')
    batch = collate([data[i] for i in range(32)])
    before = next(model.parameters()).detach().clone()
    losses = [_victim_update(model, optimizer, batch['csi'].to('cuda:0'),
                            batch['pose'].to('cuda:0')) for _ in range(2)]
    if not all(np.isfinite(losses)) or torch.equal(before, next(model.parameters())):
        raise ValueError('B32 victim CUDA smoke failed')
    barrier(context, 'B32_complete', split)
    runtime.write(Path(context['directory']) / f'smoke.{split}.json', dict(
        plan_sha256=plan['plan_sha256'], execution=env, pid=os.getpid(),
        binding=runtime.preparation_binding(plan, env),
        source_attacker_batch=64, victim_batch=32, victim_steps=2,
        source_smoke_epochs=2, source_smoke_frames=64, generator_steps=6,
        production_attacker_epochs=100, production_warmup=10,
        losses=losses, peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
        discarded=True, real_train_input=True, status='CAPACITY_ONLY_NOT_TRAINED'))


def run_pipeline(runtime, plan):
    import torch
    env = runtime.gate_check(plan)
    operator = plan['operators']['backtime']
    updates = 89 * plan['contract']['n_poison']
    print(f'BACKTIME_ONLY {plan["profile"]}: Clean50 -> attacker100/B64 '
          f'(11 clean+89 alternating, {updates} generator steps) -> scratch victim50/B32 '
          '-> six-dose evaluation. No SIBA/LADDER or 3090 imports.', flush=True)
    assert (operator['fit_epochs'], operator['warmup'], operator['batch_size']) == (100, 10, 64)
    runtime.require_idle()
    with runtime.log_to(plan['outdir'], 'train.clean_surrogate'):
        runtime.train(plan, 'clean_surrogate', env)
    torch.cuda.empty_cache()
    if runtime.completed(plan, 'backtime', env) is None:
        for phase, fn in (('search', runtime.fit_trigger), ('smoke', runtime.learned_smoke),
                          ('train', runtime.train), ('evaluate', runtime.evaluate)):
            runtime.require_idle()
            with runtime.log_to(plan['outdir'], phase + '.backtime'):
                fn(plan, 'backtime', env)
            torch.cuda.empty_cache()
    runtime.export(plan, env)


def gate_check(runtime, plan, root, *, verify_execution=True):
    """Validate receipts; only workers (or CheckOnly) initialize CUDA.

    The launch controller must not own a third CUDA context while its two
    children enforce the peer-PID allowlist. Each child checks the actual
    execution signature before loading any training model.
    """
    path = Path(root) / 'two_splits.gate.json'
    runtime.reject_links(path)
    gate = runtime.read(path)
    master = master_plan(plan['home'], root)
    if (gate.get('master_sha256') != master['sha256'] or gate.get('workers') != 2
            or gate.get('status') != 'TWO_CONCURRENT_SMOKES_PASSED_NOT_TRAINED'
            or set(gate.get('smokes', {})) != set(SPLITS)):
        raise ValueError('Missing/foreign two-worker gate; Prepare must pass first')
    env = (runtime.execution() if verify_execution else
           gate['smokes']['p1s1']['execution'])
    recipe = dict(source_attacker_batch=64, victim_batch=32, victim_steps=2,
                  source_smoke_epochs=2, source_smoke_frames=64, generator_steps=6,
                  production_attacker_epochs=100, production_warmup=10)
    for split, row in gate['smokes'].items():
        peer = master['plans'][split]
        if (row.get('plan_sha256') != peer['plan_sha256']
                or row.get('status') != 'CAPACITY_ONLY_NOT_TRAINED'
                or any(row.get(k) != v for k, v in recipe.items())
                or row.get('discarded') is not True or row.get('real_train_input') is not True
                or len(row.get('losses', [])) != 2
                or not all(math.isfinite(x) for x in row['losses'])
                or row.get('binding') != runtime.preparation_binding(peer, env)):
            raise ValueError('Incorrect concurrent smoke recipe')
    if any(row['execution'] != env for row in gate['smokes'].values()):
        raise ValueError('GPU/runtime changed after concurrent smoke')
    runtime.verify_inputs(plan)
    return env


def worker(args):
    runtime = load_runtime(args.split)
    plan = build_plan(runtime, args.data_home, args.outdir, args.split)
    context = read_context(runtime, args.context, args.split, plan)
    expected_action = 'prepare' if args.mode == '_smoke' else 'launch'
    if context['action'] != expected_action:
        raise ValueError('Worker mode differs from its parent-created action')
    runtime.require_idle = lambda: require_idle(context['workers'].values())
    from run_mmfi_tables import _cell_lock
    runtime.reject_links(plan['outdir'])
    with _cell_lock(Path(plan['outdir']) / '.controller'):
        runtime.load_plan(plan)
        runtime.require_idle()
        if args.mode == '_smoke':
            smoke(runtime, plan, context, args.split)
        elif context['action'] == 'launch':
            runtime.gate_check = lambda p: gate_check(runtime, p, args.outdir)
            run_pipeline(runtime, plan)
        else:
            raise ValueError('Unrecognized worker action')


def spawn_pair(args, runtime, plan, action):
    """Fail-stop/drain owned peers. Never terminate a peer or start replacements."""
    root = Path(args.outdir)
    directory = root / 'executions' / (action + '_' + uuid.uuid4().hex)
    runtime.reject_links(directory)
    directory.mkdir(parents=True, exist_ok=False)
    context_file = directory / 'workers.json'
    children, streams = {}, []
    try:
        for split in SPLITS:
            out = (directory / f'{split}.stdout.log').open('x', encoding='utf-8')
            err = (directory / f'{split}.stderr.log').open('x', encoding='utf-8')
            streams += [out, err]
            command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                '--mode', '_smoke' if action == 'prepare' else '_worker',
                '--data-home', str(args.data_home), '--outdir', str(root),
                '--split', split, '--context', str(context_file)]
            children[split] = subprocess.Popen(command, cwd=REPO,
                stdout=out, stderr=err, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0))
        context = dict(profile=PROFILE, parent=os.getpid(), root=str(root),
            directory=str(directory), action=action,
            workers={s: p.pid for s, p in children.items()},
            plan_hashes={s: plan['plans'][s]['plan_sha256'] for s in SPLITS},
            script_sha256=runtime.sha(__file__))
        context['sha256'] = runtime.digest(context)
        runtime.write(context_file, context)
        print('OWNED_WORKERS ' + json.dumps(context['workers']) + f' LOGS={directory}', flush=True)
        reported = set()
        while any(p.poll() is None for p in children.values()):
            for split, p in children.items():
                if p.poll() is not None and split not in reported:
                    print(f'{split} EXIT={p.returncode}; no new jobs or retries', flush=True)
                    failure = directory / 'peer_failed.json'
                    if action == 'prepare' and p.returncode != 0 and not failure.exists():
                        runtime.write(failure, dict(split=split, exit_code=p.returncode))
                    reported.add(split)
            time.sleep(1)
        codes = {s: p.wait() for s, p in children.items()}
        if any(codes.values()):
            raise RuntimeError(f'Owned workers failed: {codes}. Both logs preserved at {directory}; no retry.')
        if action == 'prepare':
            rows = {s: runtime.read(directory / f'smoke.{s}.json') for s in SPLITS}
            runtime.write(root / 'two_splits.gate.json', dict(
                master_sha256=plan['sha256'], workers=2, smokes=rows,
                status='TWO_CONCURRENT_SMOKES_PASSED_NOT_TRAINED'))
            print('PREPARED_ONLY: both full-input audits and concurrent B64/B32 CUDA smokes passed; no full training.', flush=True)
        else:
            rows = {s: runtime.read(Path(plan['plans'][s]['outdir']) / 'results.json') for s in SPLITS}
            if any(v.get('status') != 'complete' for v in rows.values()):
                raise ValueError('Both audited split results are required')
            runtime.write(root / 'results.json', dict(status='complete',
                master_sha256=plan['sha256'], splits=rows))
            print('BACKTIME_P1S1_P3S3_COMPLETE; separate seed42 results, not pooled.', flush=True)
    finally:
        # Wait, do not kill: a root lock must not be freed while a peer is active.
        for p in children.values():
            while p.poll() is None:
                print(f'DRAINING_OWNED_PEER pid={p.pid}; nothing killed', flush=True)
                try:
                    p.wait()
                except KeyboardInterrupt:
                    print('Root lock retained until owned peer exits; no automatic kill.', flush=True)
        for stream in streams:
            stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=MODES + ('_smoke', '_worker'))
    parser.add_argument('--data-home', type=Path, default=Path.home() / 'backdooranalog')
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--split', choices=tuple(SPLITS))
    parser.add_argument('--context', type=Path)
    args = parser.parse_args(argv)
    runtime = load_runtime('p3s3')
    runtime.reject_links(args.data_home)
    if args.outdir is not None:
        runtime.reject_links(args.outdir)
    args.data_home = args.data_home.resolve()
    args.outdir = (args.outdir or args.data_home / 'runs' / PROFILE).resolve()
    if args.mode.startswith('_'):
        if not args.split or not args.context:
            parser.error('Internal worker requires its parent-created context and split')
        return worker(args)
    if args.split or args.context:
        parser.error('Both requested splits run together; no scope change through worker flags')
    if args.mode == 'status':
        from types import SimpleNamespace
        for split in SPLITS:
            print('SPLIT=' + split)
            # Native status has no CUDA initialization and supports this method set.
            r = load_runtime(split)
            r.selected_queue = lambda *a: ('backtime',)
            r.status(SimpleNamespace(outdir=args.outdir / split, backtime_policy='adaptive', methods=['backtime']))
        return
    plan = master_plan(args.data_home, args.outdir)
    if args.mode == 'inspect':
        print(json.dumps(plan, indent=2))
        print('INSPECT_ONLY: two fresh BackTime pipelines; no data/GPU/training; keep source fixed after Prepare.')
        return
    from run_mmfi_tables import _cell_lock
    with _cell_lock(args.outdir / '.controller'):
        if shutil.disk_usage(args.data_home).free < 10 * 2**30:
            raise ValueError('At least10GiB free required; nothing deleted')
        bind_master(runtime, args.outdir, plan)
        if args.mode in ('preflight', 'prepare'):
            if args.mode == 'prepare':
                require_idle()
            for split in SPLITS:
                r = load_runtime(split)
                child = plan['plans'][split]
                r.reject_links(child['outdir'])
                with _cell_lock(Path(child['outdir']) / '.controller'):
                    with r.log_to(child['outdir'], args.mode):
                        r.audit(child)
            if args.mode == 'preflight':
                print('CPU_PREFLIGHT_ONLY: both official splits hashed; no CUDA or training')
                return
            if (args.outdir / 'two_splits.gate.json').exists():
                for split in SPLITS:
                    gate_check(load_runtime(split), plan['plans'][split], args.outdir)
                print('PREPARED_ALREADY; no repeat smoke or training')
                return
            return spawn_pair(args, runtime, plan, 'prepare')
        require_idle()
        for split in SPLITS:
            gate_check(load_runtime(split), plan['plans'][split], args.outdir,
                       verify_execution=args.mode != 'launch')
        if args.mode == 'check':
            print('VERIFIED_TWO_SPLIT_GATE; no training')
            return
        return spawn_pair(args, runtime, plan, 'launch')


if __name__ == '__main__':
    main()
