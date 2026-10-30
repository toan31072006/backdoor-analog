"""Foreground, one-GPU Clean/A1 queue after the existing four PiW attacks.

Keep the active checkout unchanged. The frozen scientific controller executes
each cell in a fresh subprocess; this sidecar changes only scheduling. A failed
predecessor or child never causes a retry or an epoch-zero restart fallback.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
    MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROFILE = 'piw_metafi_clean_a1_after_wanet_s42_3090_v1'
METHODS = ('clean', 'a1')
ATTACKS = ('ftrojan', 'bpp_source', 'fiba', 'wanet_source')


def native():
    path = HERE / 'piw_metafi_remaining_3090/controller.py'
    spec = importlib.util.spec_from_file_location('_piw_after_wanet_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records():
    if os.name != 'nt':
        raise ValueError('The live queue guard requires Windows CIM')
    script = ("$ErrorActionPreference='Stop';"
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
        "@(Get-CimInstance Win32_Process | Where-Object {$_.Name -like 'python*.exe'} | "
        "Select-Object ProcessId,ParentProcessId,CommandLine) | ConvertTo-Json -Compress")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
        capture_output=True, encoding='utf-8-sig', check=True, timeout=30)
    value = json.loads(result.stdout.strip() or '[]')
    return value if isinstance(value, list) else [value]


def queue_processes(out, rows):
    """Observe the predecessor, excluding only our real Python ancestors."""
    by_pid = {r['ProcessId']: r for r in rows}
    ancestors, pid = set(), os.getpid()
    while pid and pid in by_pid and pid not in ancestors:
        ancestors.add(pid)
        pid = by_pid[pid]['ParentProcessId']
    root = os.path.normcase(str(Path(out).resolve()))
    return [r for r in rows if r['ProcessId'] not in ancestors
        and root in os.path.normcase(str(r.get('CommandLine') or '').replace('/', '\\'))]


def folder(root, method, *, predecessor=False):
    return Path(root) / ('baseline_six' if predecessor else 'clean_a1') / 'attacks/metafiplusplus/seed42' / method


def final_receipt(base, root, method, *, predecessor=False):
    """Hash and validate final model/AdamW/RNG state, not marker existence."""
    import torch
    cell = folder(root, method, predecessor=predecessor)
    plan = base.read(Path(root) / ('baseline_six' if predecessor else 'clean_a1') / 'plan.json')
    if plan.get('plan_sha256') != base.digest({k:v for k,v in plan.items() if k != 'plan_sha256'}):
        raise ValueError('Training plan digest changed')
    cfg = plan['cfg']
    if (cfg['recipes'] != {'metafiplusplus': base.RECIPE} or cfg['seed'] != 42
            or cfg['epochs'] != 200 or cfg['final_epoch'] != 199 or cfg['batch_size'] != 32
            or plan['action_sha256'] != base.ACTION_SHA):
        raise ValueError('Final checkpoint is not the approved PiW AdamW recipe')
    marker = base.read(cell / 'trained_final199.json')
    path = cell / 'checkpoint.pt'
    if (marker.get('profile') != base.PROFILE or marker.get('epoch') != 199
            or marker.get('status') != 'TRAINED_NOT_EVALUATED'
            or marker.get('plan_sha256') != plan['plan_sha256']
            or marker.get('checkpoint_sha256') != base.sha(path)):
        raise ValueError(f'{method}: missing/foreign final199 training receipt')
    blob = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    base.validate_blob(blob, plan, method)
    if blob['epoch'] != 199:
        raise ValueError('Checkpoint is not final199')
    return dict(method=method, epoch=199, checkpoint_sha256=marker['checkpoint_sha256'],
                plan_sha256=plan['plan_sha256'])


def wait_predecessor(args, base, *, observe=records, sleep=time.sleep):
    paths = [folder(args.after, m, predecessor=True) / 'trained_final199.json' for m in ATTACKS]
    while True:
        live = queue_processes(args.after, observe())
        missing = [m for m,p in zip(ATTACKS, paths) if not p.is_file()]
        if not live:
            if missing:
                raise ValueError(f'Old queue stopped before final199 for {missing}; no Clean/A1 launched')
            # The old controller.lock is acquired by the caller before the full
            # audit. Even a restarted old queue cannot overlap our GPU work.
            print('PREDECESSOR_DRAINED; auditing four final199 checkpoints next', flush=True)
            return
        print(f'WAIT_AFTER_WANET: live PIDs={[r["ProcessId"] for r in live]}; '
              f'pending_final199={missing}; no GPU/Clean/A1 work', flush=True)
        sleep(30)


def command(args, method, work):
    if method not in METHODS or work not in ('prepare', 'check', 'train', 'evaluate'):
        raise ValueError('Only Clean/A1 phases are authorized')
    return [sys.executable, '-u', str(HERE / 'piw_metafi_remaining_3090/controller.py'),
        '--mode', 'cell', '--method', method, '--work', work,
        '--data-home', str(args.data_home), '--dataset', str(args.dataset),
        '--action', str(args.action), '--out', str(args.out),
        '--checkpoint-home', str(args.checkpoint_home)]


def phase(args, base, method, work):
    base.require_idle()
    path = args.out / 'logs' / f'{work}.{method}.{time.time_ns()}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'[{work}] {method}: {path}', flush=True)
    with path.open('x', encoding='utf-8') as stream:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        child = subprocess.Popen(command(args, method, work), stdout=stream,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
        try:
            code = child.wait()
        except BaseException:
            # Keep all queue/output locks until our current worker exits. Do
            # not terminate it or open another job after an interruption.
            while True:
                try:
                    child.wait()
                    break
                except KeyboardInterrupt:
                    pass
            raise
    if code:
        raise RuntimeError(f'{method}/{work} failed exit={code}; no retry/later job: {path}')


def verify_new_plan(args, base):
    before = base.read(args.after / 'baseline_six/plan.json')
    after = base.read(args.out / 'clean_a1/plan.json')
    metadata = base.PATH_KEYS | {'profile', 'methods', 'interpretation'}
    if (before.get('profile') != 'piw3d_metafi_dtpose_s42_baseline_six_v1'
            or after.get('profile') != 'piw3d_metafi_dtpose_s42_clean_a1_v1'
            or after['cfg'].get('methods') != list(METHODS)
            or before['inputs']['splits'] != after['inputs']['splits']
            or before['action_sha256'] != after['action_sha256']
            or before['trigger'] != after['trigger']
            or before['poison'] != after['poison']
            or {k:v for k,v in before['cfg'].items() if k not in metadata}
               != {k:v for k,v in after['cfg'].items() if k not in metadata}
            or any(after['sources'].get(k) != v for k,v in before['sources'].items()
                if k != 'run_piw_metafi_dtpose_baseline_six.py')):
        raise ValueError('Clean/A1 scientific config/input/target differs from predecessor attacks')


def plan_for(args, base):
    return dict(profile=PROFILE, data_home=str(args.data_home), after=str(args.after),
        out=str(args.out), dataset=str(args.dataset), action=str(args.action),
        checkpoint_home=str(args.checkpoint_home), python=str(Path(sys.executable).resolve()),
        predecessor_plan_sha256=base.sha(args.after / 'baseline_six/plan.json'),
        native_code=base.code_binding(), sidecar_sha256=base.sha(__file__),
        selected_methods=list(METHODS), max_workers=1, epochs=200, final_epoch=199,
        recipe=base.RECIPE, seed=42, interpretation='Single-seed selected-method extension; not a full paper table')


def status(args, base):
    print('READ_ONLY; no GPU query or training', flush=True)
    for method in METHODS:
        cell = folder(args.out, method)
        marker = cell / 'trained_final199.json'
        print(method, base.read(marker) if marker.is_file() else 'not final199 in this output')
        logs = sorted((args.out / 'logs').glob(f'*.{method}.*.log'), key=lambda p:p.stat().st_mtime)
        if logs:
            from collections import deque
            print('LOG='+str(logs[-1]))
            with logs[-1].open(encoding='utf-8', errors='replace') as stream:
                print(''.join(deque(stream, maxlen=8)))


def run(args, base):
    manifest = plan_for(args, base)
    path = args.out / 'after_wanet.plan.json'
    # A dedicated outer lock prevents several waiters for different outputs
    # from racing for GPU0. The original native output locks stay untouched.
    with base.lock(args.data_home / 'tools/piw_metafi_after_wanet_gpu0'), base.lock(args.out):
        if path.exists():
            if base.read(path) != manifest:
                raise ValueError('Immutable queued source/config changed; no overwrite')
        else:
            if any(p.name not in ('controller.lock', 'logs') for p in args.out.iterdir()):
                raise ValueError('Nonempty foreign output; preserve it')
            base.write(path, manifest)
        wait_predecessor(args, base)
        with base.lock(args.after):
            if queue_processes(args.after, records()):
                raise ValueError('Old queue restarted; no new GPU work')
            audited = [final_receipt(base, args.after, m, predecessor=True) for m in ATTACKS]
            base.write(args.out / 'predecessor_final199_audit.json', audited)
            if plan_for(args, base) != manifest:
                raise ValueError('Queued inputs/source changed while waiting')
            for method in METHODS:
                if (list((args.out / 'logs').glob(f'train.{method}.*.log'))
                        and not (folder(args.out,method) / 'checkpoint.pt').is_file()):
                    raise ValueError('Started cell lacks checkpoint; no epoch-zero restart fallback')
            for method in METHODS:
                gate = base.gate_path(args, method)
                if gate.is_file():
                    phase(args, base, method, 'check')
                else:
                    phase(args, base, method, 'prepare')
                verify_new_plan(args, base)
            print('PREPARED_CLEAN_A1; queue mode explicitly authorizes training after predecessor', flush=True)
            for method in METHODS:
                cell = folder(args.out, method)
                started = list((args.out / 'logs').glob(f'train.{method}.*.log'))
                if started and not (cell / 'checkpoint.pt').is_file():
                    raise ValueError('Started cell lacks checkpoint; no epoch-zero restart fallback')
                phase(args, base, method, 'check')
                phase(args, base, method, 'train')
                final_receipt(base, args.out, method)
            for method in METHODS:
                phase(args, base, method, 'evaluate')
            results = {m:base.read(folder(args.out,m) / 'evaluation.json') for m in METHODS}
            if any(v.get('status') != 'COMPLETE' for v in results.values()):
                raise ValueError('Evaluation did not finish')
            base.write(args.out / 'clean_a1_results.json', dict(profile=PROFILE,
                status='COMPLETE', selected_methods=list(METHODS), results=results,
                note='Four predecessor attack evaluations were not run or changed'))
            print('COMPLETE_CLEAN_A1_AFTER_WANET; final199 checkpoints and evaluations retained', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('queue','status'), required=True)
    for name in ('data-home','dataset','action','after','out','checkpoint-home'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ('data_home','dataset','action','after','out','checkpoint_home'):
        path = getattr(args,name)
        if not path.is_absolute() or str(path).startswith('\\\\') or any(c in str(path) for c in '"\r\n\x00'):
            raise ValueError('Only absolute local paths are supported')
        setattr(args,name,path.resolve())
    if args.out == args.after or args.out in args.after.parents or args.after in args.out.parents:
        raise ValueError('New output must not overlap predecessor output')
    if any(args.data_home/'runs' not in p.parents for p in (args.out,args.after)):
        raise ValueError('Both outputs must be distinct descendants of DataHome/runs')
    if args.dataset == args.out or args.dataset in args.out.parents or args.out in args.dataset.parents:
        raise ValueError('Dataset and output must remain separate')
    base = native()
    if args.mode == 'status':
        return status(args,base)
    if base.sha(args.action) != base.ACTION_SHA:
        raise ValueError('Authentic bend action is required')
    base.configure_torch_cache(args)
    run(args,base)


if __name__ == '__main__':
    main()
