"""Frozen seed42 HPE-Li / PiW3D A1 adaptation, paired to authenticated Clean.

This is a new scientific victim, not continued Clean training.  Its fourteen-
joint, amplitude-only protocol is explicit and never changes historical runs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import run_piw_hpeli_clean as clean

HERE = Path(__file__).resolve().parent
PROFILE = 'piw3d_hpeli_a1_seed42_v1'
MANIFEST = 'piw_hpeli_a1.resolved.json'
STATUS = 'ONE_SEED_PIW_A1_ADAPTATION_NOT_FULL_PAPER_TABLE'
SOURCE_NAMES = tuple(dict.fromkeys((*clean.SOURCE_NAMES, 'run_piw_hpeli_a1.py',
    'piw_a1_adapter.py', 'attack/trigger.py', 'attack/method_drafts.py',
    'attack/tsba.py', 'attack/poison.py')))
file_sha, object_sha = clean.file_sha, clean.object_sha
atomic_json, atomic_replace = clean.atomic_json, clean.atomic_replace
atomic_checkpoint = clean.atomic_checkpoint
build_model = clean.build_model


def configuration(data_home, outdir, device='cuda:0', dataset_root=None,
                  clean_outdir=None, action_npy=None):
    cfg = clean.configuration(data_home, outdir, device, dataset_root)
    home = Path(cfg['data_home'])
    reference = Path(clean_outdir).resolve() if clean_outdir else (
        home / 'runs/piw_hpeli_clean_s42_v1').resolve()
    output, dataset = Path(cfg['outdir']), Path(cfg['dataset_root'])
    if (output == reference or reference in output.parents or output in reference.parents
            or reference == dataset or reference in dataset.parents or dataset in reference.parents):
        raise ValueError('A1 output, Clean output and raw dataset must be separate directories')
    cfg.update(profile=PROFILE, clean_outdir=str(reference),
        action_npy=str(Path(action_npy).resolve() if action_npy else
                       (home / 'actions/data_bend.npy').resolve()),
        attack='piw3d_a1_multicarrier_amplitude_adaptation', attack_eps=.185,
        rho=.1, poison_selection='uniform', dose_coupling='paired',
        dose_min=.2, dose_max=1., pivot=7, target_joints=[11, 13],
        theta_max_deg=40., payload_axis=[0., 0., 1.], dose_mode='linear',
        eval_doses=[0., .2, .4, .6, .8, 1.], target_pck_ref=[6, 4],
        target_pck_threshold=.2, injection_domain='after frozen feeder normalization',
        carrier_domain='first90 amplitude channels as 3x30 physical link groups',
        phase_policy='last90 phase channels unchanged; no second normalization',
        reference_policy='authenticated completed Clean; never fine-tune or overwrite')
    return cfg


def plan(data_home, outdir, device='cuda:0', dataset_root=None,
         clean_outdir=None, action_npy=None):
    return dict(schema=1, profile=PROFILE,
        cfg=configuration(data_home, outdir, device, dataset_root, clean_outdir, action_npy),
        sources={name: file_sha(HERE / name) for name in SOURCE_NAMES},
        runtime=clean.runtime_versions())


def cell_dir(manifest):
    return Path(manifest['cfg']['outdir']) / 'hpeli/seed42/a1'


def _canonical_manifest(candidate):
    cfg = candidate['cfg']
    expected = plan(cfg['data_home'], cfg['outdir'], cfg['device'], cfg['dataset_root'],
                    cfg['clean_outdir'], cfg['action_npy'])
    if any(candidate.get(key) != expected[key] for key in expected):
        raise ValueError('Source/runtime/configuration differs from frozen PiW A1 plan')
    projected = {key: value for key, value in candidate.items() if key != 'plan_sha256'}
    if candidate.get('plan_sha256') != object_sha(projected):
        raise ValueError('PiW A1 manifest fingerprint mismatch')
    if candidate['inputs'].get('fingerprint') != object_sha(candidate['inputs']['splits']):
        raise ValueError('PiW A1 input fingerprint mismatch')
    return candidate


def load_manifest(path):
    path = Path(path).resolve()
    manifest = _canonical_manifest(json.loads(path.read_text(encoding='utf-8')))
    if path != Path(manifest['cfg']['outdir']) / MANIFEST:
        raise ValueError('Manifest path differs from bound A1 output')
    return manifest


def authenticate_clean(cfg, inputs):
    root = Path(cfg['clean_outdir'])
    path = root / clean.MANIFEST
    reference = clean.load_manifest(path)
    expected = clean.plan(cfg['data_home'], root, cfg['device'], cfg['dataset_root'])
    if any(reference.get(key) != expected[key] for key in expected):
        raise ValueError('Clean reference recipe/runtime/source differs from matched A1 control')
    clean._same_inputs(reference, inputs)
    result = clean.verify_complete(reference)
    if result is None:
        raise ValueError('Clean reference is incomplete; A1 requires verified fixed-final Clean')
    folder = clean.cell_dir(reference)
    paths = {'manifest': path, 'checkpoint': folder / 'checkpoint.pt',
        'evaluation': folder / 'evaluation.json', 'predictions': folder / 'clean_predictions.npz',
        'top_result': root / 'clean_result.json'}
    record = dict(outdir=str(root), plan_sha256=reference['plan_sha256'],
        input_fingerprint=inputs['fingerprint'], checkpoint_epoch=result['checkpoint_epoch'],
        artifacts={key: dict(path=str(value), sha256=file_sha(value)) for key, value in paths.items()})
    return record, reference, result


def _reference_unchanged(record):
    for key, artifact in record['artifacts'].items():
        path = Path(artifact['path'])
        if not path.is_file() or file_sha(path) != artifact['sha256']:
            raise ValueError(f'Authenticated Clean reference changed: {key}')


def prepare(candidate):
    from piw_a1_adapter import build_trigger, PoisonedPairs
    cfg = candidate['cfg']
    inputs, datasets = clean.audit_inputs(cfg)
    reference, _, _ = authenticate_clean(cfg, inputs)
    action = Path(cfg['action_npy'])
    before = file_sha(action)
    trigger = build_trigger(action)
    if before != file_sha(action):
        raise ValueError('Action changed while deriving A1 trigger')
    poisoned = PoisonedPairs(datasets['training'], trigger)
    context = dict(inputs=inputs, clean_reference=reference,
        action=dict(path=str(action), sha256=before),
        operator=trigger.state_record(), poison=poisoned.manifest())
    return context, datasets, poisoned, trigger


def _same_context(manifest, context):
    if any(manifest.get(key) != value for key, value in context.items()):
        raise ValueError('PiW A1 data/Clean/action/operator/poison binding changed')


def preflight_smoke(datasets, poisoned, trigger):
    """Read-only CPU checks of the actual supervised pair and metric operator."""
    import numpy as np
    from piw_a1_adapter import GRID, TARGET_JOINTS, target_pose, measure_predictions
    poison = poisoned.manifest()
    if not poison['samples']:
        raise ValueError('Full PiW A1 preflight requires a nonempty poison plan')
    entry = poison['samples'][0]
    original = datasets['training'][entry['index']]
    observed = poisoned[entry['index']]
    expected_csi = trigger.inject(original['csi'], entry['dose'])
    expected_pose = target_pose(original['pose'], entry['dose']).astype(np.float32)
    if (set(observed) != {'csi', 'pose'}
            or not np.array_equal(observed['csi'], expected_csi)
            or not np.array_equal(observed['pose'], expected_pose)):
        raise ValueError('A1 supervised pair differs from the frozen trigger/payload')
    if not np.array_equal(observed['csi'][:, 90:], original['csi'][:, 90:]):
        raise ValueError('PiW A1 changed phase channels')
    if not np.array_equal(trigger.inject(original['csi'], 0.), original['csi']):
        raise ValueError('PiW A1 dose-zero input is not the identity')
    truth = np.asarray(datasets['validation'][0]['pose'], dtype=np.float64)
    targets = np.stack([target_pose(truth, dose) for dose in GRID], axis=1)
    constant = np.repeat(truth[:, None], len(GRID), axis=1)
    metrics = measure_predictions(truth, truth, targets, truth, constant)
    main = metrics['mainmetrics']
    if (main['t1_mpjpe_mm'] != 0. or main['target_pck_pct'] != 100.
            or main['mean_dce_mm'] != 0. or len(metrics['dose_rows']) != 6):
        raise ValueError('PiW six-dose attack metric smoke failed')
    untouched = [j for j in range(14) if j not in TARGET_JOINTS]
    if not np.array_equal(expected_pose[..., untouched, :], original['pose'][..., untouched, :]):
        raise ValueError('PiW payload changed a non-target joint')
    return dict(actual_poison_pair=True, phase_unchanged=True,
        dose_zero_identity=True, six_dose_metrics=True,
        input_shape=list(observed['csi'].shape), pose_shape=list(observed['pose'].shape),
        target_joints=list(TARGET_JOINTS),
        amplitude_changed=not np.array_equal(observed['csi'][:, :90], original['csi'][:, :90]))


def _checkpoint_header(path, manifest, execution=None, final=False):
    import torch
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if (blob.get('schema') != 1 or blob.get('profile') != PROFILE
            or blob.get('plan_sha256') != manifest['plan_sha256']
            or blob.get('cfg') != manifest['cfg']
            or blob.get('poison_sha256') != object_sha(manifest['poison'])
            or blob.get('clean_reference_sha256') != object_sha(manifest['clean_reference'])
            or type(blob.get('epoch')) is not int
            or not 0 <= blob['epoch'] < manifest['cfg']['epochs']
            or not all(key in blob for key in ('model', 'optimizer', 'rng', 'execution'))):
        raise ValueError('Checkpoint is not the bound PiW A1 epoch/RNG/optimizer/poison state')
    if execution is not None and execution != blob['execution']:
        raise ValueError('Execution backend/GPU changed; refusing A1 resume')
    if final and blob['epoch'] != manifest['cfg']['final_epoch']:
        raise ValueError('Only fixed final epoch199 may be evaluated')
    return blob


def _strict_replay(saved, expected, location='metrics'):
    if isinstance(expected, dict):
        if not isinstance(saved, dict) or set(saved) != set(expected):
            raise ValueError(f'A1 metric replay keys differ: {location}')
        for key in expected:
            _strict_replay(saved[key], expected[key], f'{location}.{key}')
    elif isinstance(expected, (list, tuple)):
        if not isinstance(saved, list) or len(saved) != len(expected):
            raise ValueError(f'A1 metric replay length differs: {location}')
        for index, (left, right) in enumerate(zip(saved, expected)):
            _strict_replay(left, right, f'{location}[{index}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        tolerance = 1e-10 if 'pct' in location else 1e-9
        if (type(saved) not in (int, float) or not math.isfinite(saved)
                or not math.isclose(saved, expected, rel_tol=0, abs_tol=tolerance)):
            raise ValueError(f'A1 metrics do not reproduce saved per-sample predictions: {location}')
    elif isinstance(expected, bool):
        if type(saved) is not bool or saved is not expected:
            raise ValueError(f'A1 metric replay flag differs: {location}')
    elif saved != expected:
        raise ValueError(f'A1 metric replay differs: {location}')


def _reference_reproduction(fresh, reference):
    import numpy as np
    if fresh.shape != reference.shape or not np.isfinite(fresh).all():
        raise ValueError('Invalid fresh Clean control reproduction arrays')
    errors = np.linalg.norm(fresh.astype(np.float64) - reference.astype(np.float64), axis=-1) * 1000
    record = dict(max_joint_difference_mm=float(errors.max()),
                  mean_joint_difference_mm=float(errors.mean()), tolerance_mm=.05)
    if record['max_joint_difference_mm'] > .05:
        raise ValueError('Clean checkpoint inference does not reproduce saved control within .05mm')
    return record


def verify_complete(manifest, *, allow_partial=False):
    """No model/CUDA/inference/writes: authenticate references and replay NPZ."""
    import numpy as np
    from piw_a1_adapter import GRID, target_pose, measure_predictions
    folder = cell_dir(manifest)
    result_path, top_path = folder / 'evaluation.json', Path(manifest['cfg']['outdir']) / 'a1_result.json'
    if not result_path.exists() and not top_path.exists():
        return None
    _reference_unchanged(manifest['clean_reference'])
    complete_pair = result_path.is_file() and top_path.is_file()
    result = json.loads((result_path if result_path.is_file() else top_path).read_text(encoding='utf-8'))
    top = json.loads(top_path.read_text(encoding='utf-8')) if complete_pair else result
    if (result.get('schema') != 1 or result.get('profile') != PROFILE
            or result.get('status') != STATUS or result.get('seed') != 42
            or result.get('checkpoint_epoch') != manifest['cfg']['final_epoch']
            or result.get('plan_sha256') != manifest['plan_sha256']
            or result.get('input_fingerprint') != manifest['inputs']['fingerprint']
            or result.get('clean_reference') != manifest['clean_reference']
            or result.get('operator') != manifest['operator']
            or result.get('poison_sha256') != object_sha(manifest['poison']) or top != result):
        raise ValueError('A1 result provenance/top-level reference mismatch')
    if set(result.get('artifacts', {})) != {'checkpoint.pt', 'a1_predictions.npz', 'poison.json'}:
        raise ValueError('Missing/unexpected A1 artifact hashes')
    for name, digest in result['artifacts'].items():
        if not (folder / name).is_file() or file_sha(folder / name) != digest:
            raise ValueError(f'A1 artifact fingerprint mismatch: {name}')
    if json.loads((folder / 'poison.json').read_text(encoding='utf-8')) != manifest['poison']:
        raise ValueError('A1 poison plan differs from frozen manifest')
    checkpoint = _checkpoint_header(folder / 'checkpoint.pt', manifest, final=True)
    if result.get('execution') != checkpoint['execution']:
        raise ValueError('A1 result/checkpoint execution mismatch')
    with np.load(manifest['clean_reference']['artifacts']['predictions']['path'], allow_pickle=False) as ref:
        ref_pred, ref_truth, ref_names = ref['predictions'], ref['truth'], ref['names'].tolist()
    with np.load(folder / 'a1_predictions.npz', allow_pickle=False) as payload:
        if set(payload.files) != {'predictions', 'trigger_predictions', 'reference_predictions',
                'reference_trigger_predictions', 'reference_reproduced_predictions',
                'truth', 'targets', 'names', 'doses'}:
            raise ValueError('Unexpected A1 prediction NPZ schema')
        names, truth = payload['names'].tolist(), payload['truth']
        if (names != manifest['inputs']['splits']['test_data']['ordered_names'] or names != ref_names
                or not np.array_equal(truth, ref_truth)
                or not np.array_equal(payload['reference_predictions'], ref_pred)
                or not np.array_equal(payload['doses'], np.asarray(GRID))):
            raise ValueError('A1 full TEST identities/truth/reference/dose proof mismatch')
        targets = np.stack([target_pose(truth, dose) for dose in GRID], axis=1)
        if not np.array_equal(payload['targets'], targets):
            raise ValueError('A1 stored attacker targets differ from fourteen-joint payload')
        replay = measure_predictions(truth, payload['predictions'], payload['trigger_predictions'],
            payload['reference_predictions'], payload['reference_trigger_predictions'])
        reproduction = _reference_reproduction(payload['reference_reproduced_predictions'], ref_pred)
    _strict_replay(result.get('measurements'), replay)
    _strict_replay(result.get('reference_reproduction'), reproduction)
    return result if complete_pair or allow_partial else None


def _atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.saving')
    try:
        temporary.write_text(text, encoding='utf-8')
        atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_report(manifest, result):
    _, _, reference_result = authenticate_clean(manifest['cfg'], manifest['inputs'])
    root, measured = Path(manifest['cfg']['outdir']), result['measurements']
    ref = reference_result['metrics']
    metrics = measured['mainmetrics']
    text = ('# ' + STATUS + '\n\n'
        'HPE-Li / Person-in-WiFi-3D, one person/14 joints, seed42.\n'
        'Fresh scratch A1 victim: 200epochs/final199, batch32, AdamW lr.001/wd.01, ordinary MPJPE.\n'
        'Clean is an authenticated read-only external reference, never warm-started or overwritten.\n'
        'PiW adaptation: normalized amplitude only, three physical 30-bin groups; phase unchanged.\n'
        'rho=.1, uniform paired U(.2,1), attack epsilon=.185, Z40deg, pivot7->[11,13].\n'
        'Target-PCK@.2 uses clean GT scale(6,4), only target joints; not pose-level ASR.\n'
        'Attack metrics at dose1, no alignment; Clean metrics on all14 joints. DCE averages five positive doses.\n'
        'Different PiW/MM-Fi skeletons/scales: report datasets separately, never pool absolute scores.\n'
        'Official full one-person TRAIN/TEST; shared capture prefixes are not held-out-session generalization.\n'
        'One seed, no across-seed SD/CI; not a full comparison/ablation/defense paper table.\n\n'
        '| Method | Seed | T-MPJPE (mm) | Target-PCK (%) | Clean MPJPE (mm) | Clean PA-MPJPE (mm) | DCE (mm) |\n'
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
        f"| Clean reference | 42 | -- | -- | {ref['clean_mpjpe_mm']:.4f} | {ref['clean_pampjpe_mm']:.4f} | -- |\n"
        f"| A1 (PiW-adapted) | 42 | {metrics['t1_mpjpe_mm']:.4f} | {metrics['target_pck_pct']:.4f} | "
        f"{metrics['clean_mpjpe_mm']:.4f} | {metrics['clean_pampjpe_mm']:.4f} | {metrics['mean_dce_mm']:.4f} |\n")
    _atomic_text(root / 'a1_pilot.md', text)
    rows = measured['dose_rows']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(root / 'a1_six_doses.csv', stream.getvalue())
    atomic_json(root / 'a1_diagnostics.json', measured)


def recover_partial_completion(manifest):
    """Repair only authenticated publication files; callers hold output locks.

    Never replace the prediction NPZ or repeat model inference to recover a
    missing JSON/report. A corrupt surviving artifact remains a hard failure.
    """
    result = verify_complete(manifest, allow_partial=True)
    if result is None:
        return False
    root, folder = Path(manifest['cfg']['outdir']), cell_dir(manifest)
    publications = (folder / 'evaluation.json', root / 'a1_result.json')
    reports = tuple(root / name for name in
                    ('a1_pilot.md', 'a1_six_doses.csv', 'a1_diagnostics.json'))
    missing = [path for path in publications if not path.is_file()]
    if not missing and all(path.is_file() for path in reports):
        return False
    for path in missing:
        atomic_json(path, result)
    export_report(manifest, result)
    print('[PiW A1] repaired authenticated publication; no training/inference', flush=True)
    return True


def _infer(model, dataset, cfg, trigger=None, dose=0.):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    prediction, truth, names = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False,
                                num_workers=0, drop_last=False):
            csi = batch['csi']
            if trigger is not None and dose:
                csi = torch.from_numpy(np.stack([trigger.inject(value, dose, eps=.185)
                    for value in csi.numpy()])).float().contiguous()
            output, _ = model(csi.to(cfg['device'], dtype=torch.float32))
            prediction.append(output[:, 0].cpu().numpy())
            truth.append(batch['pose'][:, 0].numpy())
            names.extend(batch['name'])
    return np.concatenate(prediction), np.concatenate(truth), names


def _evaluate(manifest, model, datasets, trigger, execution):
    import numpy as np
    import torch
    from piw_a1_adapter import GRID, target_pose, measure_predictions
    cfg, folder = manifest['cfg'], cell_dir(manifest)
    reference_record, ref_manifest, _ = authenticate_clean(cfg, manifest['inputs'])
    if reference_record != manifest['clean_reference']:
        raise ValueError('Clean reference changed before A1 evaluation')
    with np.load(reference_record['artifacts']['predictions']['path'], allow_pickle=False) as saved:
        ref_pred, ref_truth, ref_names = saved['predictions'], saved['truth'], saved['names'].tolist()
    prediction, truth, names = _infer(model, datasets['validation'], cfg)
    if names != ref_names or not np.array_equal(truth, ref_truth):
        raise ValueError('A1 TEST truth/order differs from authenticated Clean')
    reference_model = build_model().to(cfg['device'])
    reference_model.load_state_dict(clean._checkpoint_header(
        reference_record['artifacts']['checkpoint']['path'], ref_manifest, final=True)['model'], strict=True)
    fresh, fresh_truth, fresh_names = _infer(reference_model, datasets['validation'], cfg)
    if fresh_names != names or not np.array_equal(fresh_truth, truth):
        raise ValueError('Clean checkpoint inference does not reproduce TEST identities/truth')
    reproduction = _reference_reproduction(fresh, ref_pred)
    predictions, references = [prediction.copy()], [ref_pred.copy()]
    for dose in GRID[1:]:
        print(f'[PiW A1 eval] full TEST attack+Clean control d={dose:g}', flush=True)
        attacked, same_truth, same_names = _infer(model, datasets['validation'], cfg, trigger, dose)
        control, other_truth, other_names = _infer(reference_model, datasets['validation'], cfg, trigger, dose)
        if (same_names != names or other_names != names or not np.array_equal(same_truth, truth)
                or not np.array_equal(other_truth, truth)):
            raise ValueError('Dose/control evaluation identities/truth changed')
        predictions.append(attacked)
        references.append(control)
    triggered, reference_triggered = np.stack(predictions, axis=1), np.stack(references, axis=1)
    measurements = measure_predictions(truth, prediction, triggered, ref_pred, reference_triggered)
    path = folder / 'a1_predictions.npz'
    temporary = path.with_name(path.name + '.saving')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, predictions=prediction, trigger_predictions=triggered,
                reference_predictions=ref_pred, reference_trigger_predictions=reference_triggered,
                reference_reproduced_predictions=fresh,
                truth=truth, targets=np.stack([target_pose(truth, dose) for dose in GRID], axis=1),
                names=np.asarray(names, dtype=str), doses=np.asarray(GRID))
        atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    _reference_unchanged(manifest['clean_reference'])
    result = dict(schema=1, profile=PROFILE, status=STATUS, seed=42,
        checkpoint_epoch=cfg['final_epoch'], plan_sha256=manifest['plan_sha256'],
        input_fingerprint=manifest['inputs']['fingerprint'],
        clean_reference=manifest['clean_reference'], operator=manifest['operator'],
        poison_sha256=object_sha(manifest['poison']), execution=execution,
        measurements=measurements, units='millimetres; stored poses in metres',
        reference_reproduction=reproduction,
        artifacts={name: file_sha(folder / name) for name in
                   ('checkpoint.pt', 'a1_predictions.npz', 'poison.json')})
    atomic_json(folder / 'evaluation.json', result)
    atomic_json(Path(cfg['outdir']) / 'a1_result.json', result)
    verify_complete(manifest)
    export_report(manifest, result)


def train_worker(manifest, datasets, poisoned, trigger):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    cfg, folder = manifest['cfg'], cell_dir(manifest)
    _reference_unchanged(manifest['clean_reference'])
    if recover_partial_completion(manifest) or verify_complete(manifest) is not None:
        print('[PiW A1] verified saved final evaluation; no training/inference', flush=True)
        return
    execution = clean._setup_execution(cfg['device'])
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if cfg['device'].startswith('cuda:'):
        torch.cuda.manual_seed(cfg['seed'])
    model = build_model().to(cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                                  betas=tuple(cfg['betas']), eps=cfg['eps'])
    checkpoint = folder / 'checkpoint.pt'
    start = 0
    if checkpoint.exists():
        blob = _checkpoint_header(checkpoint, manifest, execution)
        model.load_state_dict(blob['model'], strict=True)
        optimizer.load_state_dict(blob['optimizer'])
        clean._restore_rng(blob['rng'], cfg['device'])
        start = blob['epoch'] + 1
        print(f'[PiW A1] resumed epoch{blob["epoch"]}; next={start}', flush=True)
    poison_path = folder / 'poison.json'
    if poison_path.exists():
        if json.loads(poison_path.read_text(encoding='utf-8')) != manifest['poison']:
            raise ValueError('Existing poison artifact differs from frozen A1 plan')
    else:
        atomic_json(poison_path, manifest['poison'])
    loader = DataLoader(poisoned, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=0, drop_last=False)
    print(f'[PiW A1] scratch TRAIN={len(poisoned)} poison={int(len(poisoned)*.1)}; '
          'ordinary MPJPE; fixed final199; Clean read-only', flush=True)
    for epoch in range(start, cfg['epochs']):
        model.train()
        began, total, count = time.monotonic(), 0., 0
        for batch in loader:
            csi = batch['csi'].to(cfg['device'], dtype=torch.float32).contiguous()
            target = batch['pose'].to(cfg['device'], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(csi)
            if prediction.shape != target.shape:
                raise ValueError('A1 prediction/target shape mismatch')
            loss = torch.linalg.vector_norm(prediction - target, dim=-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite A1 loss; no skip/retry/batch fallback')
            loss.backward()
            gradients = [torch.isfinite(value.grad).all() for value in model.parameters()
                         if value.grad is not None]
            if gradients and not torch.stack(gradients).all():
                raise RuntimeError('Nonfinite A1 gradient; no optimizer update')
            optimizer.step()
            total += float(loss.detach()) * len(csi)
            count += len(csi)
        if count != len(poisoned):
            raise RuntimeError('A1 epoch omitted training identities')
        _reference_unchanged(manifest['clean_reference'])
        atomic_checkpoint(checkpoint, dict(schema=1, profile=PROFILE,
            plan_sha256=manifest['plan_sha256'], cfg=cfg, epoch=epoch,
            poison_sha256=object_sha(manifest['poison']),
            clean_reference_sha256=object_sha(manifest['clean_reference']),
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            rng=clean._rng_state(cfg['device']), execution=execution, train_loss=total/count))
        print(f'epoch {epoch}/{cfg["final_epoch"]}: loss={total/count:.8f} '
              f'lr={optimizer.param_groups[0]["lr"]:.2e} t={time.monotonic()-began:.1f}s', flush=True)
        print(f'[ckpt] saved -> {checkpoint} (epoch{epoch})', flush=True)
    _checkpoint_header(checkpoint, manifest, execution, final=True)
    context, _, _, _ = prepare(manifest)
    _same_context(manifest, context)
    _evaluate(manifest, model, datasets, trigger, execution)
    print('[PiW A1] COMPLETE seed42; full TEST + six doses + matched Clean controls', flush=True)


def _execute_private(path):
    from run_mmfi_tables import _cell_lock
    manifest = load_manifest(path)
    with _cell_lock(cell_dir(manifest)):
        context, datasets, poisoned, trigger = prepare(manifest)
        _same_context(manifest, context)
        recover_partial_completion(manifest)
        if verify_complete(manifest) is not None:
            print('[PiW A1] verified complete; no GPU/training/inference/writes')
            return 0
        train_worker(manifest, datasets, poisoned, trigger)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-home', 'dataset-root', 'outdir', 'clean-outdir', 'action-npy'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--device', default='cuda:0')
    modes = parser.add_mutually_exclusive_group()
    for name in ('dry-run', 'preflight', 'status', 'check-only', 'export-only'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker or args._manifest:
        if (not args._worker or args._manifest is None
                or any(getattr(args, name) is not None for name in
                       ('data_home', 'outdir', 'dataset_root', 'clean_outdir', 'action_npy'))
                or any((args.dry_run, args.preflight, args.status, args.check_only, args.export_only))):
            raise ValueError('Private worker requires only --_worker --_manifest PATH')
        return _execute_private(args._manifest)
    if args.data_home is None or args.outdir is None:
        parser.error('--data-home and --outdir required')
    candidate = plan(args.data_home, args.outdir, args.device, args.dataset_root,
                     args.clean_outdir, args.action_npy)
    if args.dry_run:
        print('[PiW A1] DRY RUN: 1 A1 victim from scratch, seed42,200epochs,batch32; '
              'read-only completed Clean reference; no data/model/CUDA/output writes/training.')
        print(json.dumps(candidate, indent=2))
        return 0
    root = Path(candidate['cfg']['outdir'])
    path = root / MANIFEST
    if args.status:
        if not path.exists():
            print('[PiW A1] NOT_PLANNED; no writes/GPU/training')
            return 1
        manifest = load_manifest(path)
        if any(manifest[key] != candidate[key] for key in candidate):
            raise ValueError('Requested config differs from A1 manifest')
        result = verify_complete(manifest)
        checkpoint = cell_dir(manifest) / 'checkpoint.pt'
        print('[PiW A1] ' + ('VERIFIED_SAVED_ARTIFACTS_COMPLETE' if result else
            f'INCOMPLETE; checkpoint epoch={_checkpoint_header(checkpoint, manifest)["epoch"]}'
            if checkpoint.exists() else 'INCOMPLETE; no checkpoint'))
        print('--status replays saved artifacts; --check-only also rehashes raw inputs/action.')
        return 0 if result else 1
    context, datasets, poisoned, trigger = prepare(candidate)
    if args.preflight:
        print(json.dumps(dict(status='READ_ONLY_INPUT_CLEAN_ACTION_OPERATOR_CPU_PREFLIGHT',
            input_fingerprint=context['inputs']['fingerprint'],
            clean_plan=context['clean_reference']['plan_sha256'],
            operator=context['operator'], poison={key: value for key, value in
                context['poison'].items() if key != 'samples'},
            a1_smoke=preflight_smoke(datasets, poisoned, trigger),
            cpu_model=clean.cpu_model_smoke()), indent=2))
        return 0
    if args.check_only or args.export_only:
        if not path.exists():
            raise FileNotFoundError('Missing PiW A1 manifest')
        manifest = load_manifest(path)
        if any(manifest[key] != candidate[key] for key in candidate):
            raise ValueError('Requested config differs from A1 manifest')
        _same_context(manifest, context)
        result = verify_complete(manifest)
        if result is None:
            print('[PiW A1] INCOMPLETE; no inference/training performed')
            return 1
        if args.export_only:
            from run_mmfi_tables import _cell_lock
            with _cell_lock(root):
                export_report(manifest, result)
        print('[PiW A1] VERIFIED_COMPLETE_INPUTS_CLEAN_POISON_NPZ; no inference/training')
        return 0
    from run_mmfi_tables import _cell_lock
    with _cell_lock(root):
        if path.exists():
            manifest = load_manifest(path)
            if any(manifest[key] != candidate[key] for key in candidate):
                raise ValueError('Requested config differs from A1 manifest')
            _same_context(manifest, context)
        else:
            if any(child.name not in {MANIFEST, 'run.lock'} for child in root.iterdir()):
                raise ValueError('Foreign output without A1 manifest; choose a new output')
            manifest = dict(candidate, **context, created_utc=datetime.now(timezone.utc).isoformat())
            manifest['plan_sha256'] = object_sha(manifest)
            atomic_json(path, manifest)
        with _cell_lock(cell_dir(manifest)):
            recover_partial_completion(manifest)
        if verify_complete(manifest) is not None:
            print('[PiW A1] verified complete; no GPU/training/inference/writes')
            return 0
        folder = cell_dir(manifest)
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / 'console.log'
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--_worker', '--_manifest', str(path)]
        with log.open('ab') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=str(HERE))
            print(f'[PiW A1] seed42 -> {args.device}; PID={child.pid}; log: {log}', flush=True)
            try:
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    child.terminate(); child.wait()
                raise
        if code:
            raise RuntimeError(f'PiW A1 worker exited{code}; inspect{log}; no automatic retry')
        if verify_complete(manifest) is None:
            raise RuntimeError('A1 worker returned without authenticated complete artifacts')
        print(f'[PiW A1] COMPLETE: {root / "a1_pilot.md"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f'[PiW A1] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
