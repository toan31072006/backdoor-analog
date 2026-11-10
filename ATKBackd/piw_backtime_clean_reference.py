"""Read-only historical Clean references; never import optimizer/RNG for training.

Supported source layouts are the HPE-Li Clean/v3 runners and the two-backbone
PiW plan layout. Unknown or mismatched profiles fail closed, without retraining.
Saved TEST arrays are opened only during final evaluation, not preflight.
"""
from pathlib import Path
import json
import math

import numpy as np
import run_piw_hpeli_clean as base


HERE = Path(__file__).resolve().parent
MODEL_SOURCES = {
    'hpeli': ('models/hpeli.py', 'models/sk_network.py'),
    'wiflow': ('models/wiflow_piw3d.py', 'models/_vendor/wiflow_mmfi.py',
               'models/piw3d_two_models.py', 'models/piw3d_backbone_factory.py'),
    'metafiplusplus': ('models/metafi_plus_plus_piw3d.py', 'models/metafi_plus_plus.py',
                      'models/metafi_channel_transformer.py', 'models/piw3d_two_models.py',
                      'models/piw3d_backbone_factory.py'),
}
MULTI_PROFILES = {
    'piw3d_two_models_s42_six_baselines_v1',
    'piw3d_metafi_dtpose_s42_clean_a1_v1',
    'piw_metafi_clean_vastenv_s42_3090_v1',
    'piw_metafi_clean_vastenv_s42_4090_v1',
}


def discover(cfg):
    """Bounded metadata-only lookup in PiW run directories, never by scores.

    No recursive drive scan, tensors, TEST arrays, CUDA or filesystem writes.
    The chosen sources are frozen at preflight, never re-selected on resume.
    """
    result = {b: [] for b in MODEL_SOURCES}
    roots = {Path(cfg['data_home'])/'runs', Path(cfg['outdir']).parent}
    for parent in sorted(roots):
        if not parent.is_dir() or parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
            continue
        for folder in sorted(parent.iterdir()):
            if (not folder.is_dir() or not folder.name.lower().startswith('piw')
                    or folder.is_symlink() or (hasattr(folder, 'is_junction') and folder.is_junction())
                    or folder.resolve() == Path(cfg['outdir']).resolve()):
                continue
            for name in ('piw_hpeli_clean.resolved.json', 'piw_lr_control.resolved.json', 'plan.json'):
                path = folder/name
                if not path.is_file():
                    continue
                try:
                    manifest = read(regular(path))
                    profile = manifest.get('profile')
                except (ValueError, OSError):
                    continue
                if profile in (base.PROFILE, 'piw_hpeli_body_axis_s42_lr001_control_v3'):
                    backbones = ['hpeli']
                elif profile in MULTI_PROFILES:
                    backbones = [b for b in ('wiflow', 'metafiplusplus')
                                 if (folder/'attacks'/b/'seed42/clean/checkpoint.pt').is_file()]
                else:
                    continue
                for b in backbones:
                    value = str(folder.resolve())
                    if value not in result[b]:
                        result[b].append(value)
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction())
           for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f'Missing or linked Clean reference file: {path}')
    return path.resolve()


def files_match(record):
    for name, expected in record['files'].items():
        path = regular(name)
        if path.stat().st_size != expected['bytes'] or base.file_sha(path) != expected['sha256']:
            raise ValueError(f'Historical Clean reference changed: {path}')


def _recipe(source, backbone, cfg):
    # Known source runners have fixed MPJPE/no-scheduler/no-AMP recipes. Check
    # explicit overrides as well, rather than accepting a renamed manifest.
    for name, expected in dict(seed=42, epochs=200, final_epoch=199,
                               batch_size=32, num_workers=0, drop_last=True, amp=False).items():
        if source.get(name) != expected:
            raise ValueError(f'{backbone} Clean recipe mismatch: {name}={source.get(name)!r}, expected {expected!r}')
    for name, expected in dict(pretrained=False, lr_scheduler=False, gradient_clip=None,
                               victim_loss='ordinary MPJPE', test_checkpoint_selection=False).items():
        if name in source and source[name] != expected:
            raise ValueError(f'{backbone} Clean recipe mismatch: {name}')
    recipe = source.get('recipes', {}).get(backbone)
    if recipe is None:
        recipe = {k: source.get(k) for k in ('optimizer', 'lr', 'weight_decay')}
        recipe['accumulation'] = 1
    if any(recipe.get(k) != v for k, v in cfg['recipes'][backbone].items()):
        raise ValueError(f'{backbone} Clean optimizer/accumulation differs from BackTime victim')
    if source.get('betas', [.9, .999]) != [.9, .999] or source.get('eps', 1e-8) != 1e-8:
        raise ValueError(f'{backbone} Clean AdamW settings differ')


def _blob(record):
    import torch
    path = regular(record['checkpoint'])
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if (blob.get('profile') != record['source_profile']
            or blob.get('plan_sha256') != record['source_plan_sha256']
            or type(blob.get('epoch')) is not int or blob['epoch'] != 199
            or not all(k in blob for k in ('model', 'optimizer', 'rng', 'execution'))):
        raise ValueError('Clean checkpoint profile/plan/final199 mismatch')
    if record['layout'] == 'multi':
        if blob.get('backbone') != record['backbone'] or blob.get('method') != 'clean':
            raise ValueError('Reference is not the requested backbone Clean checkpoint')
        poison = blob.get('poison', {})
        if poison.get('n_poison') != 0 or poison.get('n_cover', 0) != 0 or blob.get('trigger') is not None:
            raise ValueError('Clean reference contains poison/cover/trigger')
    else:
        manifest = read(record['manifest'])
        if blob.get('cfg') != manifest['cfg']:
            raise ValueError('Clean checkpoint configuration differs from original manifest')
        if record['layout'] == 'hpeli_v3' and blob.get('role') != 'clean':
            raise ValueError('HPE-Li reference must be Clean, not A1')
    if not blob['model'] or any(not torch.isfinite(t).all() for t in blob['model'].values()):
        raise ValueError('Nonfinite/empty Clean weights')
    groups = blob['optimizer'].get('param_groups', [])
    if not groups or any(g.get('lr') != record['recipe']['lr']
                         or g.get('weight_decay') != record['recipe']['weight_decay']
                         or list(g.get('betas', [])) != [.9, .999] or g.get('eps') != 1e-8
                         for g in groups):
        raise ValueError('Clean saved optimizer differs from declared fixed AdamW recipe')
    return blob


def audit(folder, backbone, cfg, inputs):
    """Freeze original metadata and hashes without inference or NPZ TEST reads."""
    original = Path(folder).absolute()
    if any(p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction())
           for p in (original, *original.parents)):
        raise ValueError('Linked historical Clean directory is not accepted')
    root = original.resolve()
    for other in (Path(cfg['outdir']).resolve(), Path(cfg['dataset_root']).resolve()):
        if root == other or root in other.parents or other in root.parents:
            raise ValueError('Historical Clean, new output and raw dataset must be separate')
    candidates = [root / n for n in ('piw_hpeli_clean.resolved.json',
                                     'piw_lr_control.resolved.json', 'plan.json') if (root / n).is_file()]
    if len(candidates) != 1:
        raise ValueError(f'{backbone}: supply a Clean RUN directory with one supported manifest: {root}')
    manifest_path = regular(candidates[0])
    manifest = read(manifest_path)
    source, profile = manifest['cfg'], manifest.get('profile')
    if manifest.get('plan_sha256') != base.object_sha({k: v for k, v in manifest.items() if k != 'plan_sha256'}):
        raise ValueError('Historical Clean manifest checksum mismatch')
    original_inputs = manifest['inputs']
    if (original_inputs.get('fingerprint') != base.object_sha(original_inputs['splits'])
            or original_inputs['splits'] != inputs['splits']):
        raise ValueError(f'{backbone} Clean full input/split/order fingerprint differs')
    _recipe(source, backbone, cfg)
    source_hashes = {k.removeprefix('ATKBackd/'): v for k, v in manifest.get('sources', {}).items()}
    for name in ('data_utils/feeder.py', *MODEL_SOURCES[backbone]):
        if source_hashes.get(name) != base.file_sha(HERE / name):
            raise ValueError(f'{backbone} Clean model/preprocessing source differs: {name}')
    if backbone == 'hpeli' and profile == base.PROFILE:
        layout, cell = 'hpeli_v1', root / 'hpeli/seed42/clean'
        evaluation, arrays, pred_key = cell/'evaluation.json', cell/'clean_predictions.npz', 'predictions'
    elif backbone == 'hpeli' and profile == 'piw_hpeli_body_axis_s42_lr001_control_v3':
        layout, cell = 'hpeli_v3', root / 'hpeli/seed42/clean'
        evaluation, arrays, pred_key = root/'evaluation.json', root/'predictions.npz', 'reference_predictions'
    elif backbone in ('wiflow', 'metafiplusplus') and profile in MULTI_PROFILES:
        layout, cell = 'multi', root / 'attacks' / backbone / 'seed42/clean'
        evaluation, arrays, pred_key = cell/'evaluation.json', cell/'predictions.npz', 'predictions'
    else:
        raise ValueError(f'Unsupported historical Clean profile for {backbone}: {profile}')
    checkpoint = regular(cell/'checkpoint.pt')
    evaluation, arrays = regular(evaluation), regular(arrays)
    result = read(evaluation)
    if result.get('profile') != profile or result.get('plan_sha256') != manifest['plan_sha256']:
        raise ValueError('Historical Clean evaluation plan/profile differs')
    if layout == 'multi':
        if (result.get('status') != 'COMPLETE' or result.get('method') != 'clean'
                or result.get('backbone') != backbone or result.get('seed') != 42):
            raise ValueError('Historical Clean evaluation is incomplete or belongs to another cell')
        expected = {checkpoint: result.get('checkpoint_sha256'), arrays: result.get('predictions_sha256')}
    else:
        required = ('ONE_SEED_CLEAN_PILOT_NOT_A1_OR_FULL_PAPER_TABLE' if layout == 'hpeli_v1'
                    else 'EXPLORATORY_PIW_LR_CONTROL_ONE_SEED_NOT_FULL_PAPER_TABLE')
        if result.get('status') != required:
            raise ValueError('Historical HPE-Li Clean evaluation is incomplete')
        expected = {p: result.get('artifacts', {}).get(p.relative_to(root if layout == 'hpeli_v3' else cell).as_posix())
                    for p in (checkpoint, arrays)}
    for path, digest in expected.items():
        if base.file_sha(path) != digest:
            raise ValueError(f'Historical Clean evaluation artifact hash differs: {path}')
    metrics = result['metrics'] if layout == 'hpeli_v1' else result['clean_metrics']
    for key in ('clean_mpjpe_mm', 'clean_pampjpe_mm'):
        if type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key]) or metrics[key] < 0:
            raise ValueError('Missing/nonfinite historical Clean metrics')
    record = dict(backbone=backbone, layout=layout, manifest=str(manifest_path),
        checkpoint=str(checkpoint), predictions=str(arrays), prediction_key=pred_key,
        source_profile=profile, source_plan_sha256=manifest['plan_sha256'],
        source_runtime=manifest['runtime'], metrics=metrics, recipe=cfg['recipes'][backbone],
        files={str(p): dict(bytes=p.stat().st_size, sha256=base.file_sha(p))
               for p in (manifest_path, checkpoint, evaluation, arrays)},
        training_started=False, usage='read-only inference reference, not a training resume')
    blob = _blob(record)
    record['source_execution'] = blob['execution']
    files_match(record)
    return record


def load(record):
    files_match(record)
    return _blob(record)


def replay(record, truth, names, current_predictions):
    """Only call after all three new victims finish. No writes to source runs."""
    files_match(record)
    with np.load(record['predictions'], allow_pickle=False) as z:
        if z['names'].tolist() != names or not np.array_equal(z['truth'], truth):
            raise ValueError('Historical Clean TEST identities/truth differ')
        old = z[record['prediction_key']].copy()
    if old.shape != truth.shape or not np.isfinite(old).all():
        raise ValueError('Historical Clean prediction shape/finite check failed')
    metrics = base._metrics(old, truth)
    for key in ('clean_mpjpe_mm', 'clean_pampjpe_mm'):
        if not math.isclose(metrics[key], record['metrics'][key], rel_tol=0, abs_tol=.05):
            raise ValueError('Historical Clean saved metrics do not replay within 0.05 mm')
    if (current_predictions.shape != old.shape or not np.isfinite(current_predictions).all()
            or np.max(np.abs(current_predictions-old))*1000 > .05):
        raise ValueError('Clean checkpoint inference differs from saved predictions by over 0.05 mm; inspect backend/source, do not retrain automatically')
    return dict(source_profile=record['source_profile'], source_plan_sha256=record['source_plan_sha256'],
                historical_metrics=record['metrics'], saved_prediction_metrics=metrics,
                max_coordinate_difference_mm=float(np.max(np.abs(current_predictions-old))*1000),
                training_started=False)
