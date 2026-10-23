"""WiPose execution-only upgrade with authenticated, read-only legacy lineage.

Preserves the original recipe and all historical files. Exact normalized TRAIN
frames are cached once in RAM and finite gradients are reduced on the device.
Legacy epoch/model/SGD/RNG state is adopted only by explicit --resume-from into
a NEW profile/output, with source/input/runtime/execution checks and provenance.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent


def _private_engine(name):
    spec = importlib.util.spec_from_file_location(name, HERE/'run_wipose_hpeli_parallel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = _private_engine('_wipose_fast_legacy_reader')
ENGINE = _private_engine('_wipose_fast_parallel_engine')
BASE, util = ENGINE.BASE, ENGINE.util
PROFILE = 'wipose_2d_hpeli_clean_a1_seed42_fast_v1'
MANIFEST = 'wipose_hpeli_fast.resolved.json'
STAGES = ENGINE.STAGES
OPTIMIZATION = dict(train_cache='exact_normalized_fp32_ram',
                    gradient_guard='device_aggregate', cache_max_bytes=2*1024**3)
_original_configuration = ENGINE.configuration
_engine_prepare = ENGINE.prepare
_serial_load_manifest = BASE.load_manifest
_engine_checkpoint_header = ENGINE.checkpoint_header
_engine_schedule_training = ENGINE.schedule_training
_requested_resume_from = None

ENGINE.PROFILE = BASE.PROFILE = PROFILE
ENGINE.MANIFEST = BASE.MANIFEST = MANIFEST
BASE.SOURCE_NAMES = (*BASE.SOURCE_NAMES, 'run_wipose_hpeli_fast.py', 'wipose_fast_ops.py')

def cache_training(*args, **kwargs):
    from wipose_fast_ops import cache_training as implementation
    return implementation(*args, **kwargs)


def grouped_gradients_finite(parameters):
    from wipose_fast_ops import grouped_gradients_finite as implementation
    return implementation(parameters)


def configuration(data_home, outdir=None, device='cuda:0', dataset_root=None,
                  action_npy=None, resume_from=None):
    output = Path(outdir).resolve() if outdir else (
        Path(data_home).resolve()/'runs/wipose_hpeli_a1_s42_fast_v1')
    origin = Path(resume_from).resolve() if resume_from is not None else None
    if origin is not None and (origin == output or origin in output.parents or output in origin.parents):
        raise ValueError('Fast output and legacy source must be separate nonnested roots')
    for parent in (output, *output.parents):
        if (parent/LEGACY.MANIFEST).exists():
            raise ValueError('Fast output must not overwrite or nest inside a legacy run')
    cfg = _original_configuration(data_home, output, device, dataset_root, action_npy)
    cfg['optimization'] = dict(OPTIMIZATION)
    cfg['resume_from'] = str(origin) if origin is not None else None
    return cfg


BASE.configuration = configuration
BASE.canonical = lambda cfg: configuration(cfg['data_home'], cfg['outdir'], cfg['device'],
    cfg['dataset_root'], cfg['action_npy'], cfg.get('resume_from'))


def _controller_configuration(*args, **kwargs):
    return configuration(*args, **kwargs, resume_from=_requested_resume_from)


ENGINE.configuration = _controller_configuration


def source_hashes():
    return BASE.source_hashes()


def cell_dir(manifest, stage):
    return BASE.cell_dir(manifest, stage)


def build_model():
    return BASE.build_model()


def _scientific_cfg(cfg):
    return {key: value for key, value in cfg.items()
            if key not in ('profile', 'outdir', 'optimization', 'resume_from')}


def _legacy_snapshot(cfg, actual):
    root = Path(cfg['resume_from']).resolve()
    path = root/LEGACY.MANIFEST
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing regular legacy manifest; no scratch fallback')
    manifest_sha = util.file_sha(path)
    old = LEGACY.load_manifest(path)
    if util.file_sha(path) != manifest_sha:
        raise ValueError('Legacy manifest changed during authentication')
    if _scientific_cfg(old['cfg']) != _scientific_cfg(cfg):
        raise ValueError('Legacy scientific configuration differs; no partial adoption')
    for key in ('inputs', 'operator', 'poison', 'ordered_test_names', 'preparation_sha256'):
        if old.get(key) != actual.get(key):
            raise ValueError('Legacy input/operator/poison binding differs: '+key)
    records, blobs = {}, {}
    # Authenticate BOTH sources before publishing any destination checkpoint.
    for stage in STAGES:
        checkpoint = LEGACY.cell_dir(old, stage)/'checkpoint.pt'
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError('Missing regular legacy '+stage+' checkpoint; no scratch fallback')
        if checkpoint.with_name('checkpoint.pt.saving').exists():
            raise ValueError('Legacy checkpoint is being saved; stop safely and inspect first')
        sha = util.file_sha(checkpoint)
        blob = LEGACY.checkpoint_header(checkpoint, old, stage)
        if util.file_sha(checkpoint) != sha:
            raise ValueError('Legacy checkpoint changed during authentication')
        if stage == 'a1':
            poison_path = LEGACY.cell_dir(old, stage)/'poison.json'
            if json.loads(poison_path.read_text(encoding='utf-8')) != old['poison']:
                raise ValueError('Legacy saved poison artifact differs')
        records[stage] = dict(path=str(checkpoint), sha256=sha, epoch=blob['epoch'],
                              execution=blob['execution'])
        blobs[stage] = blob
    return dict(schema=1, source_root=str(root), source_manifest_sha256=manifest_sha,
                source_plan_sha256=old['plan_sha256'], checkpoints=records), blobs


def prepare(cfg):
    prepared, datasets, trigger, poisoned = _engine_prepare(cfg)
    if cfg.get('resume_from') is not None:
        prepared['resume'] = _legacy_snapshot(cfg, prepared)[0]
        prepared['plan_sha256'] = util.object_sha({k: v for k, v in prepared.items()
                                                  if k != 'plan_sha256'})
    return prepared, datasets, trigger, poisoned


def _validate_resume_record(manifest):
    cfg, record = manifest['cfg'], manifest.get('resume')
    if cfg.get('resume_from') is None:
        if record is not None:
            raise ValueError('Unexpected legacy lineage for a fresh run')
        return
    if (not isinstance(record, dict) or record.get('schema') != 1
            or record.get('source_root') != cfg['resume_from']
            or set(record.get('checkpoints', {})) != set(STAGES)):
        raise ValueError('Invalid legacy resume lineage')
    for stage in STAGES:
        row = record['checkpoints'][stage]
        expected = Path(cfg['resume_from'])/'hpeli/seed42'/stage/'checkpoint.pt'
        if (row.get('path') != str(expected) or type(row.get('epoch')) is not int
                or not 0 <= row['epoch'] < cfg['epochs']
                or not isinstance(row.get('execution'), dict)
                or len(row.get('sha256', '')) != 64):
            raise ValueError('Invalid legacy checkpoint lineage')


def load_manifest(path):
    saved = _serial_load_manifest(path)
    _validate_resume_record(saved)
    return saved


def _lineage_for(manifest, stage):
    row = manifest['resume']['checkpoints'][stage]
    return dict(stage=stage, source_root=manifest['resume']['source_root'],
                source_manifest_sha256=manifest['resume']['source_manifest_sha256'],
                source_plan_sha256=manifest['resume']['source_plan_sha256'], **row)


def checkpoint_header(path, manifest, stage, execution=None, final=False):
    blob = _engine_checkpoint_header(path, manifest, stage, execution, final)
    _validate_resume_record(manifest)
    if manifest['cfg'].get('resume_from') is None:
        if 'resume_lineage' in blob:
            raise ValueError('Unexpected imported checkpoint lineage')
    else:
        expected = _lineage_for(manifest, stage)
        if blob.get('resume_lineage') != expected or blob['epoch'] < expected['epoch']:
            raise ValueError('Fast checkpoint legacy lineage differs')
        lineage_path = Path(manifest['cfg']['outdir'])/'resume.lineage.json'
        if (not lineage_path.is_file() or lineage_path.is_symlink()
                or json.loads(lineage_path.read_text(encoding='utf-8')) != manifest['resume']):
            raise ValueError('Immutable fast resume lineage artifact differs or is missing')
    return blob


@contextmanager
def _readonly_lock(path):
    """Probe an existing lock without changing a byte or creating directories."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise ValueError('Missing existing legacy lock; cannot prove the old run is idle: '+str(path))
    with path.open('r+b' if os.name == 'nt' else 'rb') as handle:
        handle.seek(0)
        locked = False
        try:
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Legacy run is still active; no checkpoint adoption') from exc
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


def _cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def _equal(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def import_legacy_checkpoints(source_root, manifest):
    if manifest['cfg'].get('resume_from') is None or str(Path(source_root).resolve()) != manifest['cfg']['resume_from']:
        raise ValueError('Explicit legacy source differs from the frozen resume binding')
    root = Path(source_root).resolve()
    _validate_resume_record(manifest)
    with ExitStack() as stack:
        for path in [root/'run.lock', *(root/'hpeli/seed42'/s/'run.lock' for s in STAGES)]:
            stack.enter_context(_readonly_lock(path))
        origin, blobs = _legacy_snapshot(manifest['cfg'], manifest)
        if origin != manifest['resume']:
            raise ValueError('Legacy source changed after preflight; no adoption')
        target_root = Path(manifest['cfg']['outdir'])
        lineage_path = target_root/'resume.lineage.json'
        if lineage_path.exists() and json.loads(lineage_path.read_text(encoding='utf-8')) != origin:
            raise ValueError('Existing destination lineage differs')
        # Validate all existing destinations before writing any missing one.
        for stage in STAGES:
            checkpoint = cell_dir(manifest, stage)/'checkpoint.pt'
            if checkpoint.with_name('checkpoint.pt.saving').exists():
                raise ValueError('Destination contains a partial checkpoint; preserve and inspect')
            if checkpoint.exists():
                current = checkpoint_header(checkpoint, manifest, stage)
                if current['epoch'] == blobs[stage]['epoch']:
                    for key in ('model', 'optimizer', 'rng', 'execution', 'train_loss'):
                        if not _equal(current[key], blobs[stage][key]):
                            raise ValueError('Imported destination state differs from its origin')
            else:
                # Confirm strict parameter/optimizer state compatibility on CPU.
                model = build_model().cpu()
                model.load_state_dict(blobs[stage]['model'], strict=True)
                import torch
                optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
                optimizer.load_state_dict(blobs[stage]['optimizer'])
        target_root.mkdir(parents=True, exist_ok=True)
        if not lineage_path.exists():
            util.atomic_json(lineage_path, origin)
        for stage in STAGES:
            folder = cell_dir(manifest, stage)
            folder.mkdir(parents=True, exist_ok=True)
            checkpoint = folder/'checkpoint.pt'
            if checkpoint.exists():
                continue
            blob = copy.deepcopy(blobs[stage])
            blob.update(profile=PROFILE, cfg=manifest['cfg'], plan_sha256=manifest['plan_sha256'],
                        resume_lineage=_lineage_for(manifest, stage))
            if stage == 'a1':
                poison_path = folder/'poison.json'
                if poison_path.exists() and json.loads(poison_path.read_text(encoding='utf-8')) != manifest['poison']:
                    raise ValueError('Destination poison artifact differs')
                if not poison_path.exists():
                    util.atomic_json(poison_path, manifest['poison'])
            util.atomic_checkpoint(checkpoint, blob)
            checkpoint_header(checkpoint, manifest, stage)
            print(f'[WiPose-fast] ADOPTED {stage} epoch{blob["epoch"]}; '
                  f'next={blob["epoch"]+1}; original checkpoint unchanged', flush=True)


def _step(model, optimizer, x, y, grouped):
    import torch
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = model(x)
    if prediction.shape != y.shape:
        raise ValueError('WiPose training shape mismatch')
    loss = torch.linalg.vector_norm(prediction-y, dim=-1).mean()
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss; no optimizer update')
    loss.backward()
    if grouped:
        finite = grouped_gradients_finite(model.parameters())
    else:
        finite = not any(p.grad is not None and not torch.isfinite(p.grad).all()
                         for p in model.parameters())
    if not finite:
        raise RuntimeError('Nonfinite gradient; no optimizer update')
    optimizer.step()
    return prediction, loss


def runtime_parity_gate(model, optimizer, training, device):
    """Disposable actual-device updates; restore production RNG without updating it."""
    import numpy as np
    import torch
    before = util._rng_state(device)
    batch = [training[index] for index in range(min(2, len(training)))]
    x = torch.from_numpy(np.stack([sample['csi'] for sample in batch])).to(device)
    y = torch.from_numpy(np.stack([sample['pose'] for sample in batch])).to(device)
    traces = []
    try:
        for grouped in (False, True):
            candidate = copy.deepcopy(model).train()
            opt = torch.optim.SGD(candidate.parameters(), lr=.01, momentum=.9)
            opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            util._restore_rng(before, device)
            trace = []
            for _ in range(2):
                prediction, loss = _step(candidate, opt, x, y, grouped)
                trace.append(_cpu_copy(dict(prediction=prediction, loss=float(loss.detach()),
                    gradients={name: p.grad for name, p in candidate.named_parameters()},
                    model=candidate.state_dict(), optimizer=opt.state_dict(), rng=util._rng_state(device))))
            traces.append(trace)
        if not _equal(traces[0], traces[1]):
            raise RuntimeError('Actual-device bitwise parity failed; no production update')
    finally:
        util._restore_rng(before, device)
    print(f'[WiPose-fast] BITWISE_RUNTIME_PARITY_PASS device={device}; two disposable steps', flush=True)
    return dict(device=device, bitwise=True, steps=2, production_state_untouched=True)


def train_stage(manifest, stage, datasets, trigger, poisoned):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from wipose_a1_adapter import PoisonedWiPoseDataset
    if ENGINE.verify_training(manifest, stage) is not None:
        print(f'[WiPose-fast] VERIFIED_TRAINED {stage}', flush=True)
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
    lineage = None
    if checkpoint.exists():
        blob = checkpoint_header(checkpoint, manifest, stage, execution)
        model.load_state_dict(blob['model'], strict=True)
        optimizer.load_state_dict(blob['optimizer'])
        util._restore_rng(blob['rng'], cfg['device'])
        start, lineage = blob['epoch']+1, blob.get('resume_lineage')
        print(f'[WiPose-fast] {stage} resumed epoch{blob["epoch"]}; next={start}', flush=True)
    elif cfg.get('resume_from') is not None:
        raise ValueError('Missing authenticated imported checkpoint; no scratch fallback')
    cache_options = dict(max_bytes=cfg['optimization']['cache_max_bytes'])
    audited_files = manifest['inputs']['splits']['training'].get('files')
    if audited_files is not None:
        cache_options['expected_hashes'] = {
            str((Path(cfg['dataset_root'])/row['path']).resolve()): row['sha256']
            for row in audited_files}
    elif any('path' in item for item in datasets['training'].items):
        raise ValueError('Missing audited input hashes for real TRAIN paths')
    cached = cache_training(datasets['training'], **cache_options)
    training = cached
    if stage == 'a1':
        training = PoisonedWiPoseDataset(cached, trigger)
        if training.manifest() != poisoned.manifest() or training.manifest() != manifest['poison']:
            raise ValueError('RAM cache changed native A1 poison identities/doses')
        path = folder/'poison.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest['poison']:
            raise ValueError('Existing fast poison artifact differs')
        if not path.exists():
            util.atomic_json(path, manifest['poison'])
    gate = runtime_parity_gate(model, optimizer, training, cfg['device'])
    loader = DataLoader(training, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=0, drop_last=cfg['drop_last'])
    expected = len(training)//cfg['batch_size']*cfg['batch_size']
    print(f'[WiPose-fast] {stage} TRAIN={len(training)}; batch32; SGD lr.01; '
          f'final49; exact RAM cache; device aggregate; {expected} samples/epoch', flush=True)
    for epoch in range(start, cfg['epochs']):
        model.train()
        total, count, began = 0., 0, time.monotonic()
        for batch in loader:
            x = batch['csi'].to(cfg['device'], dtype=torch.float32)
            y = batch['pose'].to(cfg['device'], dtype=torch.float32)
            _, loss = _step(model, optimizer, x, y, True)
            total += float(loss.detach())*len(x); count += len(x)
        if count != expected or count == 0:
            raise RuntimeError('Unexpected omitted/extra WiPose TRAIN identities')
        current = BASE.static_plan(cfg)
        if any(manifest[key] != value for key, value in current.items()):
            raise ValueError('Source/runtime changed during training')
        blob = dict(schema=1, profile=PROFILE, stage=stage, plan_sha256=manifest['plan_sha256'],
            cfg=cfg, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
            rng=util._rng_state(cfg['device']), execution=execution, train_loss=total/count,
            optimization_gate=gate)
        if lineage is not None:
            blob['resume_lineage'] = lineage
        util.atomic_checkpoint(checkpoint, blob)
        print(f'epoch {epoch}/{cfg["final_epoch"]}: loss={total/count:.8f} '
              f'lr={cfg["lr"]:.2e} t={time.monotonic()-began:.1f}s', flush=True)
        print(f'[ckpt] saved -> {checkpoint} (epoch{epoch})', flush=True)
    BASE.same_context(manifest, prepare(cfg)[0])
    blob = checkpoint_header(checkpoint, manifest, stage, execution, final=True)
    util.atomic_json(folder/ENGINE.TRAINING_MARKER, ENGINE._training_record(manifest, stage, blob))
    ENGINE.verify_training(manifest, stage)
    print(f'[WiPose-fast] TRAIN_COMPLETE {stage}; NOT_EVALUATED', flush=True)


def _worker_command(manifest, stage, phase):
    cfg = manifest['cfg']
    return [sys.executable, '-u', str(Path(__file__).resolve()),
            '--data-home', cfg['data_home'], '--outdir', cfg['outdir'],
            '--device', cfg['device'], '--_worker', stage, '--_phase', phase,
            '--_manifest', str(Path(cfg['outdir'])/MANIFEST)]


def schedule_training(manifest, jobs_per_gpu=2, stages=STAGES):
    if manifest['cfg'].get('resume_from') is not None:
        import_legacy_checkpoints(manifest['cfg']['resume_from'], manifest)
    return _engine_schedule_training(manifest, jobs_per_gpu, stages)


ENGINE.prepare = prepare
ENGINE.load_manifest = load_manifest
BASE.load_manifest = load_manifest
ENGINE.checkpoint_header = BASE.checkpoint_header = checkpoint_header
ENGINE.train_stage = train_stage
ENGINE._worker_command = _worker_command
ENGINE.schedule_training = schedule_training
verify_training = ENGINE.verify_training
verify_stage = ENGINE.verify_stage
evaluate_trained_stage = ENGINE.evaluate_trained_stage
export_report = ENGINE.export_report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-home', 'dataset-root', 'outdir', 'action-npy', 'resume-from'):
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
                            args.export_only, args.clean_only, args.resume_from)):
        parser.error('Private worker cannot use public runtime modes or override origin')
    if args.outdir is None:
        if args.data_home is None:
            parser.error('--data-home or --outdir required')
        args.outdir = args.data_home/'runs/wipose_hpeli_a1_s42_fast_v1'
    return args


def main(argv=None):
    global _requested_resume_from
    args = parse_args(argv)
    _requested_resume_from = args.resume_from
    existing = args.outdir.resolve()/MANIFEST
    if existing.exists() and args.resume_from is not None:
        if str(args.resume_from.resolve()) != load_manifest(existing)['cfg'].get('resume_from'):
            raise ValueError('Requested legacy source differs from existing fast manifest')
    if args.dry_run:
        if existing.exists():
            cfg = load_manifest(existing)['cfg']
        else:
            if args.data_home is None:
                raise ValueError('--data-home required for a new fast run')
            cfg = configuration(args.data_home, args.outdir, args.device, args.dataset_root,
                                args.action_npy, args.resume_from)
        print('[WiPose-fast] DRY RUN: metadata only; '
              'explicit legacy continuation if configured, otherwise scratch; '
              'NO data/model/CUDA/training/writes')
        print(json.dumps(BASE.static_plan(cfg), indent=2))
        _requested_resume_from = None
        return 0
    ENGINE.parse_args = lambda unused=None: args
    try:
        return ENGINE.main(argv)
    finally:
        ENGINE.parse_args = parse_args
        _requested_resume_from = None


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, MemoryError) as exc:
        print('[WiPose-fast] ERROR: '+str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
