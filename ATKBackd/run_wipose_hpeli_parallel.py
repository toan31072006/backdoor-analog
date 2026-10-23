"""Isolated WiPose Clean/A1 pilot: two independent training slots, paired eval.

Same frozen scientific recipe/data/model as the serial pilot. Training never
loads or binds the other victim; only A1 evaluation depends on completed Clean.
This sidecar does not modify or migrate any serial manifest/checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    '_wipose_parallel_serial_engine', HERE/'run_wipose_hpeli.py')
BASE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BASE)
util = BASE.util
PROFILE = 'wipose_2d_hpeli_clean_a1_seed42_parallel_v1'
MANIFEST = 'wipose_hpeli_parallel.resolved.json'
STATUS = BASE.STATUS
STAGES = BASE.STAGES
TRAINING_STATUS = 'TRAINED_NOT_EVALUATED'
TRAINING_MARKER = 'training.complete.json'

# A private module instance keeps the tested data/metric implementations without
# changing the historical public module or its frozen scientific source bytes.
_serial_configuration = BASE.configuration
_serial_verify_stage = BASE.verify_stage
BASE.PROFILE = PROFILE
BASE.MANIFEST = MANIFEST
BASE.SOURCE_NAMES = (*BASE.SOURCE_NAMES, 'run_wipose_hpeli_parallel.py')


def configuration(data_home, outdir=None, device='cuda:0', dataset_root=None, action_npy=None):
    output = Path(outdir).resolve() if outdir else (
        Path(data_home).resolve()/'runs/wipose_hpeli_a1_s42_parallel_v1')
    forbidden = ('wipose_hpeli.resolved.json', 'piw_hpeli_clean.resolved.json',
                 'piw_hpeli_a1.resolved.json', 'wiflow_a1.resolved.json',
                 'three_tables.resolved.json', 'cross_backbones.resolved.json',
                 'checkpoint.pt')
    for parent in (output, *output.parents):
        if any((parent/name).exists() for name in forbidden):
            raise ValueError('Parallel WiPose requires a separate NEW output, not a historical run')
        if parent != output and (parent/MANIFEST).exists():
            raise ValueError('Parallel output must not be nested in another WiPose run')
    return _serial_configuration(data_home, output, device, dataset_root, action_npy)


BASE.configuration = configuration


def source_hashes():
    return BASE.source_hashes()


def prepare(cfg):
    print('[WiPose-parallel preflight] hashing/validating full prepared TRAIN/TEST, '
          'action and poison plan; CPU/disk only', flush=True)
    prepared = BASE.prepare(cfg)
    print('[WiPose-parallel preflight] full input binding ready', flush=True)
    return prepared


def load_manifest(path):
    return BASE.load_manifest(path)


def cell_dir(manifest, stage):
    return BASE.cell_dir(manifest, stage)


def build_model():
    return BASE.build_model()


def checkpoint_header(path, manifest, stage, execution=None, final=False):
    """A training checkpoint is independent of any Clean reference artifact."""
    import torch
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if (not isinstance(blob, dict) or blob.get('schema') != 1
            or blob.get('profile') != PROFILE or stage not in STAGES
            or blob.get('stage') != stage or blob.get('plan_sha256') != manifest['plan_sha256']
            or blob.get('cfg') != manifest['cfg'] or type(blob.get('epoch')) is not int
            or not 0 <= blob['epoch'] < manifest['cfg']['epochs']
            or not all(key in blob for key in ('model', 'optimizer', 'rng', 'execution'))
            or 'clean_checkpoint_sha256' in blob
            or type(blob.get('train_loss')) not in (float, int)
            or not math.isfinite(blob['train_loss']) or blob['train_loss'] < 0):
        raise ValueError('Parallel checkpoint stage/config/epoch/RNG/independence binding mismatch')
    if final and blob['epoch'] != manifest['cfg']['final_epoch']:
        raise ValueError('Only fixed final epoch can be evaluated')
    if execution is not None and blob['execution'] != execution:
        raise ValueError('Execution/GPU changed; refusing scientific resume')
    return blob


BASE.checkpoint_header = checkpoint_header


def _training_record(manifest, stage, blob):
    folder = cell_dir(manifest, stage)
    poison_sha = None
    if stage == 'a1':
        poison_path = folder/'poison.json'
        if json.loads(poison_path.read_text(encoding='utf-8')) != manifest['poison']:
            raise ValueError('Saved parallel poison plan changed')
        poison_sha = util.file_sha(poison_path)
    return dict(schema=1, profile=PROFILE, stage=stage, status=TRAINING_STATUS,
                plan_sha256=manifest['plan_sha256'], checkpoint_epoch=blob['epoch'],
                checkpoint_sha256=util.file_sha(folder/'checkpoint.pt'),
                execution=blob['execution'], poison_sha256=poison_sha)


def verify_training(manifest, stage):
    """CPU/read-only; trained != evaluated and never implies pilot completion."""
    folder = cell_dir(manifest, stage)
    marker = folder/TRAINING_MARKER
    if not marker.is_file():
        return None
    blob = checkpoint_header(folder/'checkpoint.pt', manifest, stage, final=True)
    saved = json.loads(marker.read_text(encoding='utf-8'))
    if saved != _training_record(manifest, stage, blob):
        raise ValueError('Parallel training completion marker/checkpoint binding mismatch')
    return saved


def verify_stage(manifest, stage):
    result = _serial_verify_stage(manifest, stage)
    if result is not None and verify_training(manifest, stage) is None:
        raise ValueError('Evaluation exists without authenticated completed training')
    return result


BASE.verify_stage = verify_stage


def train_stage(manifest, stage, datasets, trigger, poisoned):
    """Scratch ERM only; A1 training explicitly has no Clean dependency."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if verify_training(manifest, stage) is not None:
        print(f'[WiPose-parallel] VERIFIED_TRAINED {stage}; no optimizer/inference', flush=True)
        return
    cfg, folder = manifest['cfg'], cell_dir(manifest, stage)
    folder.mkdir(parents=True, exist_ok=True)
    execution = util._setup_execution(cfg['device'])
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if cfg['device'].startswith('cuda:'):
        torch.cuda.manual_seed(42)
    model = build_model().to(cfg['device'])
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])
    checkpoint, start = folder/'checkpoint.pt', 0
    if checkpoint.exists():
        blob = checkpoint_header(checkpoint, manifest, stage, execution)
        model.load_state_dict(blob['model'], strict=True)
        optimizer.load_state_dict(blob['optimizer'])
        util._restore_rng(blob['rng'], cfg['device'])
        start = blob['epoch']+1
        print(f'[WiPose-parallel] {stage} resumed epoch{blob["epoch"]}; next={start}', flush=True)
    if stage == 'a1':
        path = folder/'poison.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest['poison']:
            raise ValueError('Existing poison artifact mismatch')
        if not path.exists():
            util.atomic_json(path, manifest['poison'])
    training = poisoned if stage == 'a1' else datasets['training']
    loader = DataLoader(training, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=0, drop_last=cfg['drop_last'])
    expected = len(training)//cfg['batch_size']*cfg['batch_size']
    print(f'[WiPose-parallel] {stage} scratch TRAIN={len(training)} '
          f'TEST={len(datasets["validation"])}; ordinary MPJPE; SGD lr.01; batch32; '
          f'final{cfg["final_epoch"]}; drop_last=True consumes {expected}/epoch; '
          'independent training, TEST deferred', flush=True)
    for epoch in range(start, cfg['epochs']):
        model.train()
        total, count, began = 0., 0, time.monotonic()
        for batch in loader:
            x = batch['csi'].to(cfg['device'], dtype=torch.float32)
            y = batch['pose'].to(cfg['device'], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(x)
            if prediction.shape != y.shape:
                raise ValueError('WiPose training shape mismatch')
            loss = torch.linalg.vector_norm(prediction-y, dim=-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; no skip/fallback/retry')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite gradient; no optimizer update')
            optimizer.step()
            total += float(loss.detach())*len(x); count += len(x)
        if count != expected or count == 0:
            raise RuntimeError('Unexpected omitted/extra training identities')
        current = BASE.static_plan(cfg)
        if any(manifest[k] != v for k, v in current.items()):
            raise ValueError('Source/runtime changed during training')
        blob = dict(schema=1, profile=PROFILE, stage=stage, plan_sha256=manifest['plan_sha256'],
            cfg=cfg, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
            rng=util._rng_state(cfg['device']), execution=execution, train_loss=total/count)
        util.atomic_checkpoint(checkpoint, blob)
        print(f'epoch {epoch}/{cfg["final_epoch"]}: loss={total/count:.8f} '
              f'lr={cfg["lr"]:.2e} t={time.monotonic()-began:.1f}s', flush=True)
        print(f'[ckpt] saved -> {checkpoint} (epoch{epoch})', flush=True)
    # Epoch49 is recoverable without another optimizer step if marker publication
    # was interrupted. Bind all inputs again before declaring training complete.
    BASE.same_context(manifest, prepare(cfg)[0])
    blob = checkpoint_header(checkpoint, manifest, stage, execution, final=True)
    util.atomic_json(folder/TRAINING_MARKER, _training_record(manifest, stage, blob))
    verify_training(manifest, stage)
    print(f'[WiPose-parallel] TRAIN_COMPLETE {stage}; final49; NOT_EVALUATED', flush=True)


def evaluate_trained_stage(manifest, stage, datasets, trigger):
    if verify_stage(manifest, stage) is not None:
        print(f'[WiPose-parallel] VERIFIED_EVALUATED {stage}; no inference', flush=True)
        return
    if verify_training(manifest, stage) is None:
        raise ValueError('Completed training required before evaluation')
    if stage == 'a1' and verify_stage(manifest, 'clean') is None:
        raise ValueError('A1 evaluation requires authenticated completed Clean')
    cfg = manifest['cfg']
    execution = util._setup_execution(cfg['device'])
    blob = checkpoint_header(cell_dir(manifest, stage)/'checkpoint.pt',
                             manifest, stage, execution, final=True)
    model = build_model().to(cfg['device'])
    model.load_state_dict(blob['model'], strict=True)
    BASE.evaluate_stage(manifest, stage, model, datasets['validation'], trigger, execution)
    verify_stage(manifest, stage)
    print(f'[WiPose-parallel] EVAL_COMPLETE {stage}; full TEST', flush=True)


def export_report(manifest):
    return BASE.export_report(manifest)


def _worker_command(manifest, stage, phase):
    cfg = manifest['cfg']
    return [sys.executable, '-u', str(Path(__file__).resolve()),
            '--data-home', cfg['data_home'], '--outdir', cfg['outdir'],
            '--device', cfg['device'], '--_worker', stage, '--_phase', phase,
            '--_manifest', str(Path(cfg['outdir'])/MANIFEST)]


def _slots(value):
    if type(value) is not int or value not in (1, 2):
        raise ValueError('Only 1 or 2 WiPose jobs per GPU are supported')
    return value


def schedule_training(manifest, jobs_per_gpu=2, stages=STAGES):
    """Bounded subprocess pool; on failure drain owned workers, never retry/kill.

    One child is one independent RNG/optimizer/BN stream. The controller keeps
    its root lock until ALL children it launched have exited, even after error.
    """
    limit = _slots(jobs_per_gpu)
    stages = list(stages)
    if len(stages) != len(set(stages)):
        raise ValueError('Duplicate WiPose training stages are not allowed')
    pending = []
    for stage in stages:
        if stage not in STAGES:
            raise ValueError('Unknown WiPose training stage')
        if verify_training(manifest, stage) is not None:
            print(f'[WiPose-parallel] VERIFIED_TRAINED {stage}; skipped', flush=True)
        else:
            pending.append(stage)
    active, failures = {}, []
    interrupted = None
    try:
        while pending or active:
            while pending and len(active) < limit and not failures:
                stage = pending.pop(0)
                folder = cell_dir(manifest, stage)
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder/'console.log').open('a', encoding='utf-8')
                try:
                    process = subprocess.Popen(_worker_command(manifest, stage, 'train'),
                        stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
                        env=dict(os.environ, PYTHONUTF8='1'))
                except BaseException:
                    log.close()
                    raise
                active[stage] = (process, log)
                print(f'[WiPose-parallel] TRAIN {stage} -> {manifest["cfg"]["device"]}; '
                      f'PID={process.pid}; log={folder/"console.log"}', flush=True)
            for stage, (process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[stage]
                if code:
                    failures.append(f'{stage} train failed exit{code}')
                else:
                    try:
                        if verify_training(manifest, stage) is None:
                            raise ValueError('worker exited without training completion')
                    except (ValueError, OSError, RuntimeError, KeyError) as exc:
                        failures.append(f'{stage}: {exc}')
            if failures and not active:
                break
            if active:
                time.sleep(.25)
    except BaseException as exc:
        interrupted = exc
    finally:
        # No orphaned peer from a Python/controller exception. Do not terminate
        # a healthy peer just because the other job OOMed; preserve its work.
        for stage, (process, log) in list(active.items()):
            print(f'[WiPose-parallel] draining owned {stage} PID={process.pid}; '
                  'no new launch/retry/kill', flush=True)
            try:
                code = process.wait()
                if code:
                    failures.append(f'{stage} train failed exit{code}')
            finally:
                log.close()
    if interrupted is not None:
        raise interrupted
    if failures:
        raise RuntimeError('; '.join(failures)+'; inspect cell logs; no automatic retry')


def evaluate_in_worker(manifest, stage):
    if verify_stage(manifest, stage) is not None:
        print(f'[WiPose-parallel] VERIFIED_EVALUATED {stage}; skipped', flush=True)
        return
    folder = cell_dir(manifest, stage)
    with (folder/'console.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(_worker_command(manifest, stage, 'eval'),
            stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
            env=dict(os.environ, PYTHONUTF8='1'))
        print(f'[WiPose-parallel] EVAL {stage}; PID={process.pid}; '
              f'log={folder/"console.log"}', flush=True)
        try:
            code = process.wait()
        finally:
            if process.poll() is None:
                process.wait()
    if code or verify_stage(manifest, stage) is None:
        raise RuntimeError(f'WiPose {stage} evaluation failed exit{code}; no retry')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-home', 'dataset-root', 'outdir', 'action-npy'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--jobs-per-gpu', type=int, choices=(1, 2), default=2)
    parser.add_argument('--clean-only', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    for name in ('dry-run', 'preflight', 'status', 'check-only', 'export-only'):
        modes.add_argument('--'+name, action='store_true')
    parser.add_argument('--_worker', choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument('--_phase', choices=('train', 'eval'), help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if len({bool(args._worker), bool(args._phase), bool(args._manifest)}) != 1:
        parser.error('Private worker requires manifest, stage and phase together')
    if args._worker and any((args.dry_run, args.preflight, args.status, args.check_only,
                            args.export_only, args.clean_only)):
        parser.error('Private worker cannot use public runtime modes')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.outdir is None:
        if args.data_home is None:
            raise ValueError('--data-home or --outdir required')
        args.outdir = args.data_home/'runs/wipose_hpeli_a1_s42_parallel_v1'
    manifest_path = args.outdir.resolve()/MANIFEST
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        cfg = manifest['cfg']
        for value, key in ((args.data_home, 'data_home'), (args.dataset_root, 'dataset_root'),
                           (args.action_npy, 'action_npy'), (args.outdir, 'outdir')):
            if value is not None and str(value.resolve()) != cfg[key]:
                raise ValueError('Requested path differs from frozen '+key)
        if args.device != cfg['device']:
            raise ValueError('Requested device differs from frozen manifest')
    else:
        if args.status or args.check_only or args.export_only or args._worker:
            raise FileNotFoundError('No existing parallel WiPose manifest: '+str(manifest_path))
        if args.data_home is None:
            raise ValueError('--data-home required for new run')
        cfg = configuration(args.data_home, args.outdir, args.device, args.dataset_root, args.action_npy)
        manifest = None
    if args.dry_run:
        print(f'[WiPose-parallel] DRY RUN: Clean + A1 scratch; seed42; 50 epochs/final49; '
              f'SGD .01; 18x2; max {args.jobs_per_gpu} training slots; '
              'paired evaluation serial after training. NO data/model/CUDA/output writes/training.')
        print(json.dumps(BASE.static_plan(cfg), indent=2))
        return 0
    if args.status:
        for stage in STAGES:
            state = ('VERIFIED_SAVED_ARTIFACTS_COMPLETE' if verify_stage(manifest, stage)
                     else TRAINING_STATUS if verify_training(manifest, stage) else 'INCOMPLETE')
            print(f'[WiPose-parallel] {stage}: {state}')
        return 0
    context, datasets, trigger, poisoned = prepare(cfg)
    if manifest is not None:
        BASE.same_context(manifest, context)
    else:
        manifest = context
    if args.preflight:
        print(json.dumps(BASE.preflight_smoke(manifest, datasets, trigger, poisoned), indent=2))
        print('[WiPose-parallel] READ_ONLY_CPU_PREFLIGHT; no GPU/training/output writes')
        return 0
    selected = ('clean',) if args.clean_only else STAGES
    if args.check_only:
        missing = [stage for stage in selected if verify_stage(manifest, stage) is None]
        if missing:
            print('[WiPose-parallel] INCOMPLETE: '+', '.join(missing))
            return 1
        print('[WiPose-parallel] VERIFIED_COMPLETE_INPUTS_POISON_NPZ; no training/inference/writes')
        return 0
    from run_mmfi_tables import _cell_lock
    if args._worker:
        if args._manifest.resolve() != manifest_path:
            raise ValueError('Private worker manifest differs from exact output')
        with _cell_lock(cell_dir(manifest, args._worker)):
            if args._phase == 'train':
                train_stage(manifest, args._worker, datasets, trigger, poisoned)
            else:
                evaluate_trained_stage(manifest, args._worker, datasets, trigger)
        return 0
    root = Path(cfg['outdir'])
    if not manifest_path.exists() and root.exists() and any(p.name != 'run.lock' for p in root.iterdir()):
        raise ValueError('Nonempty output without parallel manifest; choose NEW output')
    with _cell_lock(root):
        if manifest_path.exists():
            BASE.same_context(load_manifest(manifest_path), manifest)
        else:
            util.atomic_json(manifest_path, manifest)
        if args.export_only:
            export_report(manifest)
            return 0
        # Probe selected cells before launch; never delete/rebind another owner.
        for stage in selected:
            if (cell_dir(manifest, stage)/'run.lock').exists():
                with _cell_lock(cell_dir(manifest, stage)):
                    pass
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        (root/'runtime_launches').mkdir(exist_ok=True)
        util.atomic_json(root/'runtime_launches'/f'{stamp}_{os.getpid()}.json',
            dict(controller_pid=os.getpid(), plan_sha256=manifest['plan_sha256'],
                 jobs_per_gpu=args.jobs_per_gpu, device=cfg['device'], stages=list(selected)))
        schedule_training(manifest, args.jobs_per_gpu, selected)
        for stage in selected:
            evaluate_in_worker(manifest, stage)
        export_report(manifest)
        print(f'[WiPose-parallel] {"CLEAN_ONLY" if args.clean_only else "PILOT_COMPLETE"}: '
              f'{root/"wipose_pilot.md"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('[WiPose-parallel] ERROR: '+str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
