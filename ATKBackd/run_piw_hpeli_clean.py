"""Frozen HPE-Li / Person-in-WiFi-3D one-person CLEAN seed-42 pilot.

This sidecar deliberately does not import the attack trainer.  It trains from
scratch for 200 epochs and evaluates only the final epoch on the official TEST.
It is not an A1 port, a backdoor result, or a complete paper table.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

HERE = Path(__file__).resolve().parent
PROFILE = 'piw3d_hpeli_clean_seed42_v1'
MANIFEST = 'piw_hpeli_clean.resolved.json'
EPOCHS = 200
SOURCE_NAMES = ('run_piw_hpeli_clean.py', 'data_utils/feeder.py',
                'models/hpeli.py', 'models/sk_network.py', 'eval/metrics.py',
                'attack/payload.py', 'run_mmfi_tables.py')
PCK_THRESHOLDS = (.5, .4, .3, .2, .1)
EXPECTED_IDENTITIES = {
    'train_data': (89946, 28121, 27876,
        'aff065cd2697ce09ebd610e6c2cd96430968d253853b34ae54a8b479bc99ae34'),
    'test_data': (7824, 2586, 2586,
        '5d2f13ceed8aa5fc8e2a7b0beb94b453525901b3f282995aaed24af12dfd0fbb'),
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.saving')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                             encoding='utf-8')
        atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_replace(temporary, path):
    """Retry only the identical atomic rename on transient Windows read locks.

    No serialization, optimizer update, batch, epoch or job is repeated.  The
    preceding checkpoint remains intact if the bounded rename retries fail.
    """
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            if os.name != 'nt' or error.winerror not in (5, 32, 33) or attempt == 4:
                raise
            time.sleep(.05 * (attempt + 1))


def runtime_versions():
    """No data access, torch import, CUDA probe or output writes."""
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'PyYAML'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f'Missing required package: {name}') from exc
    return dict(python=platform.python_version(), executable=str(Path(sys.executable).resolve()),
        system=platform.system(), machine=platform.machine(), packages=packages,
        threading={name: os.environ.get(name) for name in
                   ('MKL_THREADING_LAYER', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})


def configuration(data_home, outdir, device='cuda:0', dataset_root=None):
    if device != 'cpu' and not re.fullmatch(r'cuda:[0-9]+', device):
        raise ValueError('Device must be cpu or explicit cuda:N; no automatic fallback')
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    root = Path(dataset_root).resolve() if dataset_root else (
        data_home / 'datasets/Person-in-WiFi-3D_processed_v1').resolve()
    if outdir == root or root in outdir.parents or outdir in root.parents:
        raise ValueError('Output and raw dataset must be separate directories')
    return dict(profile=PROFILE, data_home=str(data_home), dataset_root=str(root),
        outdir=str(outdir), device=device, dataset='person-in-wifi-3d',
        experiment_name='one-person', model='hpeli', num_person=1,
        num_keypoints=14, coordinates=3, csi_shape=[3, 180, 20],
        seed=42, epochs=EPOCHS, final_epoch=EPOCHS - 1, batch_size=32,
        num_workers=0, optimizer='adamw', lr=.001, weight_decay=.01,
        betas=[.9, .999], eps=1e-8, lr_scheduler=False, gradient_clip=None,
        victim_loss='ordinary MPJPE', pretrained=False, amp=False,
        data_parallel=False, train_shuffle=True, drop_last=False,
        normalization='feeder.PersonInWiFi3D.normalize; separate amplitude/phase minmax',
        train_filter='official one-person list; retain current iff next frame in same list',
        test_filter='official one-person TEST list; no next-frame filtering',
        coordinate_units='metres as supplied; no inferred unit conversion',
        clean_pck_ref=[6, 4], clean_pck_thresholds=list(PCK_THRESHOLDS),
        checkpoint_selection='fixed final epoch; TEST unused until training complete',
        cudnn_deterministic=True, cudnn_benchmark=False,
        allow_tf32=False, float32_matmul_precision='highest')


def plan(data_home, outdir, device='cuda:0', dataset_root=None):
    return dict(schema=1, profile=PROFILE,
        cfg=configuration(data_home, outdir, device, dataset_root),
        sources={name: file_sha(HERE / name) for name in SOURCE_NAMES},
        runtime=runtime_versions())


def _read_names(root, split):
    path = root / split / f'{split}_list.txt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing official PiW list: {path}')
    names = [line.split()[0] for line in path.read_text(encoding='utf-8').splitlines()
             if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f'Empty or duplicate official identities: {path}')
    for name in names:
        if not re.fullmatch(r'S[0-9]{2}_[0-9]+_[0-9]+', name):
            raise ValueError(f'Unsafe or unexpected official PiW identity: {name!r}')
    return names


def audit_inputs(cfg):
    """Read/hash/validate ALL listed one-person arrays, not just a sample.

    Multi-person rows remain in the official list hash, but are not trained or
    used for a model/metric claim.  Original files and lists are never modified.
    """
    import numpy as np
    from data_utils.feeder import PersonInWiFi3D, _next_piw_frame_name
    root = Path(cfg['dataset_root'])
    details, datasets, identities = {}, {}, {}
    print('[PiW preflight] hashing every listed one-person CSI/pose; CPU/disk only', flush=True)
    for split, mode in (('train_data', 'training'), ('test_data', 'validation')):
        names = _read_names(root, split)
        one_person = [name for name in names if int(name.split('_')[0][2]) == 1]
        if not one_person:
            raise ValueError(f'No one-person identities in {split}')
        chosen = set(one_person)
        retained = ([name for name in one_person if _next_piw_frame_name(name) in chosen]
                    if mode == 'training' else one_person)
        if not retained:
            raise ValueError(f'No retained one-person samples in {split}')
        expected = EXPECTED_IDENTITIES[split]
        identity_sha = hashlib.sha256(('\n'.join(retained) + '\n').encode('utf-8')).hexdigest()
        if (len(names), len(one_person), len(retained), identity_sha) != expected:
            raise ValueError(f'{split} differs from frozen official full one-person identities/counts')
        files = []
        for index, name in enumerate(one_person):
            for kind in ('csi_ap', 'keypoint'):
                path = root / split / kind / f'{name}.npy'
                if not path.is_file():
                    raise FileNotFoundError(f'Missing listed PiW file: {path}')
                before = path.stat()
                value = np.load(path, allow_pickle=False)
                if (value.dtype.kind not in 'fiu' or not np.isfinite(value).all()
                        or (kind == 'csi_ap' and value.shape != (3, 180, 20))
                        or (kind == 'keypoint' and value.shape not in ((14, 3), (1, 14, 3)))):
                    raise ValueError(f'Invalid numeric/finite one-person {kind} shape {value.shape}: {path}')
                with np.errstate(over='ignore', invalid='ignore'):
                    if not np.isfinite(value.astype(np.float32)).all():
                        raise ValueError(f'PiW array cannot be represented in FP32: {path}')
                if kind == 'keypoint':
                    pose = value.reshape(14, 3).astype(np.float64)
                    if np.linalg.norm(pose[6] - pose[4]) <= 1e-8 or np.linalg.norm(
                            pose - pose.mean(axis=0)) <= 1e-8:
                        raise ValueError(f'Degenerate clean GT/PCK reference scale: {path}')
                digest = file_sha(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f'Data changed during audit: {path}')
                files.append(dict(path=str(path.relative_to(root)).replace('\\', '/'),
                    size=after.st_size, sha256=digest))
            if index and index % 10000 == 0:
                print(f'[PiW preflight] {split}: {index}/{len(one_person)} identities', flush=True)
        dataset = PersonInWiFi3D(mode, str(root), experiment_name='one-person', num_person=1)
        if [item['name'] for item in dataset.items] != retained:
            raise RuntimeError('Existing PiW loader differs from frozen official identity selection')
        details[split] = dict(official_list_sha256=file_sha(root / split / f'{split}_list.txt'),
            official_list_count=len(names), listed_one_person=len(one_person),
            retained_count=len(retained), ordered_names=retained,
            ordered_names_sha256=identity_sha,
            files_sha256=object_sha(files), files=files)
        identities[split], datasets[mode] = retained, dataset
        print(f'[PiW preflight] {split}: {len(retained)} retained / {len(one_person)} one-person', flush=True)
    if set(identities['train_data']) & set(identities['test_data']):
        raise ValueError('TRAIN/TEST identity overlap')
    return dict(schema=1, dataset_root=str(root), splits=details,
                fingerprint=object_sha(details)), datasets


def cell_dir(manifest):
    return Path(manifest['cfg']['outdir']) / 'hpeli/seed42/clean'


def _canonical_manifest(candidate):
    expected = plan(candidate['cfg']['data_home'], candidate['cfg']['outdir'],
                    candidate['cfg']['device'], candidate['cfg']['dataset_root'])
    if any(candidate.get(key) != expected[key] for key in expected):
        raise ValueError('Source/runtime/configuration differs from the frozen PiW Clean plan')
    projected = {key: value for key, value in candidate.items() if key != 'plan_sha256'}
    if candidate.get('plan_sha256') != object_sha(projected):
        raise ValueError('PiW Clean manifest fingerprint mismatch')
    if candidate.get('inputs', {}).get('fingerprint') != object_sha(candidate['inputs']['splits']):
        raise ValueError('PiW input manifest fingerprint mismatch')
    return candidate


def load_manifest(path):
    path = Path(path).resolve()
    manifest = _canonical_manifest(json.loads(path.read_text(encoding='utf-8')))
    if path != Path(manifest['cfg']['outdir']) / MANIFEST:
        raise ValueError('Manifest path differs from the bound output directory')
    return manifest


def _same_inputs(manifest, current):
    if current != manifest['inputs']:
        raise ValueError('Raw data/list/content fingerprint changed; refusing cache/resume')


def build_model():
    from models.hpeli import HPELiNet, hpeli_init
    model = HPELiNet(num_keypoints=14, num_coor=3, subcarrier_num=180,
                     num_person=1, dataset='person-in-wifi-3d')
    model.apply(hpeli_init)
    return model


def cpu_model_smoke():
    import torch
    model = build_model().cpu().train()
    prediction, _ = model(torch.randn(2, 3, 180, 20))
    target = torch.randn(2, 1, 14, 3)
    if prediction.shape != target.shape:
        raise ValueError('HPE-Li PiW output shape mismatch')
    loss = torch.linalg.vector_norm(prediction - target, dim=-1).mean()
    loss.backward()
    if not torch.isfinite(loss) or not all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Nonfinite HPE-Li CPU forward/backward')
    return dict(input_shape=[2, 3, 180, 20], output_shape=list(prediction.shape),
                parameters=sum(p.numel() for p in model.parameters()), finite=True)


def _setup_execution(device):
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    result = dict(torch_cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        device=device, deterministic=True, benchmark=False, tf32=False,
        matmul_precision=torch.get_float32_matmul_precision())
    if device.startswith('cuda:'):
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA unavailable; no CPU fallback')
        torch.cuda.set_device(torch.device(device))
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu_name=properties.name, capability=list(torch.cuda.get_device_capability(device)),
                      gpu_total_memory=properties.total_memory,
                      visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    return result


def _rng_state(device):
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device) if device.startswith('cuda:') else None)


def _restore_rng(state, device):
    import numpy as np
    import torch
    if not state:
        raise ValueError('Missing exact-resume RNG state')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if device.startswith('cuda:'):
        if state.get('cuda') is None:
            raise ValueError('Missing CUDA RNG state')
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def atomic_checkpoint(path, blob):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + '.saving')
    try:
        torch.save(blob, temporary)
        atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metrics(predictions, truth):
    import numpy as np
    from eval.metrics import pa_mpjpe, pck
    predictions, truth = np.asarray(predictions, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if (predictions.shape != truth.shape or predictions.ndim != 3
            or predictions.shape[1:] != (14, 3) or not len(truth)
            or not np.isfinite(predictions).all() or not np.isfinite(truth).all()):
        raise ValueError('Invalid clean prediction/ground-truth arrays')
    return dict(clean_mpjpe_mm=float(np.linalg.norm(predictions - truth, axis=-1).mean() * 1000),
        clean_pampjpe_mm=float(pa_mpjpe(predictions, truth).mean() * 1000),
        clean_pck_pct={str(threshold): pck(predictions, truth, threshold, ref=(6, 4)) * 100
                       for threshold in PCK_THRESHOLDS})


def _checkpoint_header(path, manifest, execution=None, final=False):
    import torch
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if (blob.get('schema') != 1 or blob.get('profile') != PROFILE
            or blob.get('plan_sha256') != manifest['plan_sha256']
            or blob.get('cfg') != manifest['cfg']
            or type(blob.get('epoch')) is not int
            or not -1 < blob['epoch'] < manifest['cfg']['epochs']
            or not all(key in blob for key in ('model', 'optimizer', 'rng', 'execution'))):
        raise ValueError('Checkpoint is not the bound PiW Clean epoch/RNG/optimizer state')
    if execution is not None and blob['execution'] != execution:
        raise ValueError('Execution backend/GPU changed; refusing scientific resume')
    if final and blob['epoch'] != manifest['cfg']['final_epoch']:
        raise ValueError('Only the frozen final epoch may be evaluated/reported')
    return blob


def verify_complete(manifest):
    """Replay full saved predictions without model construction, CUDA or writes."""
    import numpy as np
    folder = cell_dir(manifest)
    result_path = folder / 'evaluation.json'
    marker = Path(manifest['cfg']['outdir']) / 'clean_result.json'
    if not result_path.exists() and not marker.exists():
        return None
    both_present = result_path.is_file() and marker.is_file()
    existing = result_path if result_path.is_file() else marker
    result = json.loads(existing.read_text(encoding='utf-8'))
    top = json.loads(marker.read_text(encoding='utf-8')) if both_present else result
    if (result.get('schema') != 1 or result.get('profile') != PROFILE
            or result.get('status') != 'ONE_SEED_CLEAN_PILOT_NOT_A1_OR_FULL_PAPER_TABLE'
            or result.get('seed') != 42
            or result.get('checkpoint_epoch') != manifest['cfg']['final_epoch']
            or result.get('input_fingerprint') != manifest['inputs']['fingerprint']
            or result.get('counts') != {key: value['retained_count']
                                       for key, value in manifest['inputs']['splits'].items()}
            or result.get('plan_sha256') != manifest['plan_sha256'] or top != result):
        raise ValueError('Clean result plan/top-level reference mismatch')
    expected_artifacts = {'checkpoint.pt', 'clean_predictions.npz'}
    if set(result.get('artifacts', {})) != expected_artifacts:
        raise ValueError('Unexpected or missing clean artifact hashes')
    for name, digest in result['artifacts'].items():
        if not (folder / name).is_file() or file_sha(folder / name) != digest:
            raise ValueError(f'Completed artifact fingerprint mismatch: {name}')
    checkpoint = _checkpoint_header(folder / 'checkpoint.pt', manifest, final=True)
    if result.get('execution') != checkpoint['execution']:
        raise ValueError('Result and checkpoint execution identities differ')
    with np.load(folder / 'clean_predictions.npz', allow_pickle=False) as payload:
        if set(payload.files) != {'predictions', 'truth', 'names'}:
            raise ValueError('Unexpected prediction NPZ schema')
        names = payload['names'].tolist()
        if names != manifest['inputs']['splits']['test_data']['ordered_names']:
            raise ValueError('Saved TEST identities/order differ from official list')
        metrics = _metrics(payload['predictions'], payload['truth'])
    recorded = result.get('metrics', {})
    def close_number(value, reference, tolerance):
        return (type(value) in (int, float) and math.isfinite(value)
                and math.isclose(value, reference, rel_tol=0, abs_tol=tolerance))

    if (not isinstance(recorded, dict) or set(recorded) != set(metrics)
            or not close_number(recorded['clean_mpjpe_mm'], metrics['clean_mpjpe_mm'], 1e-9)
            or not close_number(recorded['clean_pampjpe_mm'], metrics['clean_pampjpe_mm'], 1e-9)
            or not isinstance(recorded['clean_pck_pct'], dict)
            or set(recorded['clean_pck_pct']) != set(metrics['clean_pck_pct'])
            or any(not close_number(recorded['clean_pck_pct'][key], value, 1e-10)
                   for key, value in metrics['clean_pck_pct'].items())):
        raise ValueError('Clean metrics do not reproduce the saved per-sample predictions')
    # A crash between the two atomic JSON writes is recoverable only AFTER the
    # exact final checkpoint, NPZ, identities and every metric pass replay.
    return result if both_present else None


def export_report(manifest, result):
    root = Path(manifest['cfg']['outdir'])
    metrics = result['metrics']
    text = ('# ONE_SEED_CLEAN_PILOT_NOT_A1_OR_FULL_PAPER_TABLE\n\n'
        'HPE-Li / Person-in-WiFi-3D one-person, seed 42, scratch, fixed final epoch 199.\n'
        '200 epochs; batch32; AdamW lr0.001/wd0.01; ordinary MPJPE; workers0; FP32.\n'
        'Official one-person TRAIN with DT-Pose next-frame filter; full official one-person TEST.\n'
        'TEST is not used for checkpoint selection, tuning or early stopping.\n'
        'Coordinates are treated as metres as supplied, without guessed unit conversion.\n'
        'PCK is CLEAN joint accuracy at GT reference joints (6,4), not Target-PCK or ASR.\n'
        'One seed: no across-seed SD/CI. Not an original-author full reproduction claim.\n\n'
        '| Clean MPJPE (mm) | Clean PA-MPJPE (mm) | Clean PCK@0.2 (%) |\n'
        '| ---: | ---: | ---: |\n'
        f"| {metrics['clean_mpjpe_mm']:.4f} | {metrics['clean_pampjpe_mm']:.4f} | "
        f"{metrics['clean_pck_pct']['0.2']:.4f} |\n")
    temporary = root / 'clean_pilot.md.saving'
    try:
        temporary.write_text(text, encoding='utf-8')
        atomic_replace(temporary, root / 'clean_pilot.md')
    finally:
        if temporary.exists():
            temporary.unlink()


def train_worker(manifest, datasets):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    cfg, folder = manifest['cfg'], cell_dir(manifest)
    execution = _setup_execution(cfg['device'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if cfg['device'].startswith('cuda:'):
        torch.cuda.manual_seed(cfg['seed'])
    model = build_model().to(cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
        weight_decay=cfg['weight_decay'], betas=tuple(cfg['betas']), eps=cfg['eps'])
    checkpoint = folder / 'checkpoint.pt'
    start = 0
    if checkpoint.is_file():
        blob = _checkpoint_header(checkpoint, manifest, execution)
        model.load_state_dict(blob['model'], strict=True)
        optimizer.load_state_dict(blob['optimizer'])
        _restore_rng(blob['rng'], cfg['device'])
        start = blob['epoch'] + 1
        print(f"[PiW Clean] resumed epoch {blob['epoch']}; next={start}", flush=True)
    train_loader = DataLoader(datasets['training'], batch_size=cfg['batch_size'],
        shuffle=True, num_workers=0, drop_last=False)
    print(f"[PiW Clean] TRAIN={len(datasets['training'])} TEST={len(datasets['validation'])}; "
          'scratch/final-epoch-only; NO attack/trigger/TEST selection', flush=True)
    for epoch in range(start, cfg['epochs']):
        model.train()
        total, count, began = 0., 0, time.monotonic()
        for batch in train_loader:
            csi = batch['csi'].to(cfg['device'], dtype=torch.float32)
            truth = batch['pose'].to(cfg['device'], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(csi)
            if prediction.shape != truth.shape:
                raise ValueError('Clean HPE-Li prediction/GT shape mismatch')
            loss = torch.linalg.vector_norm(prediction - truth, dim=-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite clean MPJPE; no skip/retry/batch fallback')
            loss.backward()
            gradient_checks = [torch.isfinite(p.grad).all() for p in model.parameters()
                               if p.grad is not None]
            if gradient_checks and not torch.stack(gradient_checks).all():
                raise RuntimeError('Nonfinite clean gradient; no optimizer update')
            optimizer.step()
            total += float(loss.detach()) * len(csi)
            count += len(csi)
        if count != len(datasets['training']):
            raise RuntimeError('Training epoch omitted identities')
        atomic_checkpoint(checkpoint, dict(schema=1, profile=PROFILE,
            plan_sha256=manifest['plan_sha256'], cfg=cfg, epoch=epoch,
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            rng=_rng_state(cfg['device']), execution=execution, train_loss=total / count))
        print(f'epoch {epoch}/{cfg["final_epoch"]}: loss={total/count:.8f} '
              f'lr={optimizer.param_groups[0]["lr"]:.2e} t={time.monotonic()-began:.1f}s', flush=True)
        print(f'[ckpt] saved -> {checkpoint} (epoch {epoch})', flush=True)
    _checkpoint_header(checkpoint, manifest, execution, final=True)
    final_inputs, _ = audit_inputs(cfg)
    _same_inputs(manifest, final_inputs)
    model.eval()
    predictions, truths, names = [], [], []
    print('[PiW Clean eval] full official one-person TEST; no alignment for MPJPE', flush=True)
    with torch.no_grad():
        for batch in DataLoader(datasets['validation'], batch_size=cfg['batch_size'],
                                shuffle=False, num_workers=0, drop_last=False):
            prediction, _ = model(batch['csi'].to(cfg['device'], dtype=torch.float32))
            predictions.append(prediction[:, 0].cpu().numpy())
            truths.append(batch['pose'][:, 0].numpy())
            names.extend(batch['name'])
    predictions, truths = np.concatenate(predictions), np.concatenate(truths)
    if names != manifest['inputs']['splits']['test_data']['ordered_names']:
        raise RuntimeError('Evaluation did not consume full TEST in official order')
    path = folder / 'clean_predictions.npz'
    temporary = path.with_name(path.name + '.saving')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, predictions=predictions, truth=truths,
                                names=np.asarray(names, dtype=str))
        atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    result = dict(schema=1, profile=PROFILE, status='ONE_SEED_CLEAN_PILOT_NOT_A1_OR_FULL_PAPER_TABLE',
        plan_sha256=manifest['plan_sha256'], seed=42, checkpoint_epoch=cfg['final_epoch'],
        units='millimetres (stored poses in metres)', metrics=_metrics(predictions, truths),
        input_fingerprint=manifest['inputs']['fingerprint'], execution=execution,
        counts={key: value['retained_count'] for key, value in manifest['inputs']['splits'].items()},
        artifacts={name: file_sha(folder / name) for name in ('checkpoint.pt', 'clean_predictions.npz')})
    atomic_json(folder / 'evaluation.json', result)
    atomic_json(Path(cfg['outdir']) / 'clean_result.json', result)
    verify_complete(manifest)
    export_report(manifest, result)
    print('[PiW Clean] COMPLETE seed42; final epoch199; not A1', flush=True)


def _execute_private(path):
    from run_mmfi_tables import _cell_lock
    manifest = load_manifest(path)
    with _cell_lock(cell_dir(manifest)):
        inputs, datasets = audit_inputs(manifest['cfg'])
        _same_inputs(manifest, inputs)
        if verify_complete(manifest) is not None:
            print('[PiW Clean] verified complete; no training/evaluation', flush=True)
            return 0
        train_worker(manifest, datasets)
    return 0


def _refuse_foreign_output(root):
    if root.exists():
        allowed = {MANIFEST, 'run.lock'}
        if any(path.name not in allowed for path in root.iterdir()):
            raise ValueError('Output has files but no PiW Clean manifest; choose a new output')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    modes = parser.add_mutually_exclusive_group()
    for mode in ('dry-run', 'preflight', 'status', 'check-only', 'export-only'):
        modes.add_argument('--' + mode, action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker or args._manifest:
        if (not args._worker or args._manifest is None or args.data_home is not None
                or args.outdir is not None or args.dataset_root is not None
                or any((args.dry_run, args.preflight, args.status, args.check_only, args.export_only))):
            raise ValueError('Private worker requires only --_worker --_manifest PATH')
        return _execute_private(args._manifest)
    if args.data_home is None or args.outdir is None:
        parser.error('--data-home and --outdir are required')
    candidate = plan(args.data_home, args.outdir, args.device, args.dataset_root)
    if args.dry_run:
        print('[PiW Clean] DRY RUN: 1 clean victim; HPE-Li/PiW one-person; seed42; '
              '200 epochs; batch32; AdamW lr.001/wd.01; workers0. '
              'NO data/GPU/output writes/training; NOT A1.', flush=True)
        print(json.dumps(candidate, indent=2))
        return 0
    root = Path(candidate['cfg']['outdir'])
    path = root / MANIFEST
    if args.status:
        if not path.exists():
            print('[PiW Clean] NOT_PLANNED; no manifest; no writes/GPU/training')
            return 1
        manifest = load_manifest(path)
        if any(manifest[key] != candidate[key] for key in candidate):
            raise ValueError('Requested config differs from existing manifest')
        result = verify_complete(manifest)
        checkpoint = cell_dir(manifest) / 'checkpoint.pt'
        print('[PiW Clean] ' + ('VERIFIED_SAVED_ARTIFACTS_COMPLETE' if result else
            f'INCOMPLETE; checkpoint epoch={_checkpoint_header(checkpoint, manifest)["epoch"]}'
            if checkpoint.exists() else 'INCOMPLETE; no checkpoint'))
        print('Status replays saved artifacts only; --check-only also rehashes raw inputs.')
        return 0 if result is not None else 1
    inputs, datasets = audit_inputs(candidate['cfg'])
    if args.preflight:
        print(json.dumps(dict(status='READ_ONLY_DATA_AND_CPU_MODEL_PREFLIGHT',
            input_fingerprint=inputs['fingerprint'],
            counts={key: value['retained_count'] for key, value in inputs['splits'].items()},
            cpu_model=cpu_model_smoke()), indent=2))
        return 0
    if args.check_only or args.export_only:
        if not path.is_file():
            raise FileNotFoundError('Missing PiW Clean manifest')
        manifest = load_manifest(path)
        if any(manifest[key] != candidate[key] for key in candidate):
            raise ValueError('Requested config differs from existing manifest')
        _same_inputs(manifest, inputs)
        result = verify_complete(manifest)
        if result is None:
            print('[PiW Clean] INCOMPLETE; no training or evaluation performed')
            return 1
        if args.export_only:
            from run_mmfi_tables import _cell_lock
            with _cell_lock(root):
                export_report(manifest, result)
        print('[PiW Clean] VERIFIED_COMPLETE_INPUTS_CHECKPOINT_NPZ; no model inference/training')
        return 0
    from run_mmfi_tables import _cell_lock
    with _cell_lock(root):
        if path.is_file():
            manifest = load_manifest(path)
            if any(manifest[key] != candidate[key] for key in candidate):
                raise ValueError('Requested config differs from existing manifest')
            _same_inputs(manifest, inputs)
        else:
            _refuse_foreign_output(root)
            manifest = dict(candidate, inputs=inputs,
                            created_utc=datetime.now(timezone.utc).isoformat())
            manifest['plan_sha256'] = object_sha(manifest)
            atomic_json(path, manifest)
        if verify_complete(manifest) is not None:
            print('[PiW Clean] verified complete; no training/evaluation', flush=True)
            return 0
        folder = cell_dir(manifest)
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / 'console.log'
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--_worker', '--_manifest', str(path)]
        with log.open('ab') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     cwd=str(HERE))
            print(f'[PiW Clean] seed42 -> {args.device}; PID={child.pid}; log: {log}', flush=True)
            try:
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
                raise
        if code:
            raise RuntimeError(f'PiW Clean worker exited {code}; inspect {log}; no automatic retry')
        if verify_complete(manifest) is None:
            raise RuntimeError('Worker returned without authenticated complete artifacts')
        print(f'[PiW Clean] COMPLETE: {root / "clean_pilot.md"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f'[PiW Clean] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
