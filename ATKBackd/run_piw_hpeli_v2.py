"""New PiW Clean+A1 exploratory pilot. Default is metadata-only dry run.

No v1 source, checkpoint, evaluation or dataset is modified. A separate CUDA
capacity smoke receipt is required before explicit --launch. TEST is evaluated
only after both independent scratch victims reach fixed final epoch199.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

import run_piw_hpeli_clean as base

HERE = Path(__file__).resolve().parent
PROFILE = 'piw_hpeli_body_axis_s42_v2'
MANIFEST = 'piw_v2.resolved.json'
STATUS = 'EXPLORATORY_PIW_V2_ONE_SEED_NOT_FULL_PAPER_TABLE'
SOURCE_NAMES = tuple(dict.fromkeys((*base.SOURCE_NAMES, 'run_piw_hpeli_v2.py',
    'piw_v2_adapter.py', 'piw_a1_adapter.py', 'attack/trigger.py',
    'attack/method_drafts.py', 'attack/tsba.py', 'attack/poison.py')))


def configuration(data_home, outdir, dataset_root=None, action_npy=None,
                  device='cuda:0', jobs_per_gpu=2):
    if type(jobs_per_gpu) is not int or jobs_per_gpu not in (1, 2):
        raise ValueError('Choose one or two independent victims on this GPU')
    cfg = base.configuration(data_home, outdir, device, dataset_root)
    home, output = Path(cfg['data_home']), Path(cfg['outdir'])
    if output.name in ('piw_hpeli_clean_s42_v1', 'piw_hpeli_a1_s42_v1'):
        raise ValueError('Historical PiW output is read-only; choose a v2 directory')
    action = Path(action_npy).resolve() if action_npy else home / 'actions/data_bend.npy'
    if output == action or output in action.parents:
        raise ValueError('Action prior must be outside experiment output')
    cfg.update(profile=PROFILE, lr=.01, drop_last=True, jobs_per_gpu=jobs_per_gpu,
        action_npy=str(action), roles=['clean', 'a1'],
        model_provenance='existing HPE-Li implementation; corrected SK attention; not exact author reproduction',
        target=dict(version='piw_body_hip_axis_v2', pivot=7, target_joints=[11, 13],
            axis='normalize(clean_GT[5] - clean_GT[7])', axis_min_metres=1e-6,
            theta_max_deg=40., dose_mode='linear', pck_ref=[6, 4],
            target_pck_threshold=.2, alignment='none'),
        attack_eps=.185, rho=.1, dose_range=[.2, 1.], eval_doses=[0., .2, .4, .6, .8, 1.],
        trigger_policy='frozen v1 amplitude operator; three 30-bin groups; phase unchanged',
        initialization='independent scratch, no v1 or clean warm start',
        selection='fixed final199; TEST deferred until both victims finish',
        status=STATUS, no_automatic_retry=True, no_batch_fallback=True)
    return cfg


def plan(cfg):
    return dict(schema=2, profile=PROFILE, cfg=cfg,
                sources={name: base.file_sha(HERE / name) for name in SOURCE_NAMES},
                runtime=base.runtime_versions())


def cell(manifest, role):
    if role not in ('clean', 'a1'):
        raise ValueError('Unknown PiW role')
    return Path(manifest['cfg']['outdir']) / 'hpeli/seed42' / role


def load_manifest(path):
    path = Path(path).resolve()
    m = json.loads(path.read_text(encoding='utf-8'))
    cfg = m['cfg']
    expected = plan(configuration(cfg['data_home'], cfg['outdir'], cfg['dataset_root'],
                                  cfg['action_npy'], cfg['device'], cfg['jobs_per_gpu']))
    if any(m.get(k) != v for k, v in expected.items()):
        raise ValueError('v2 source/runtime/config changed; use the original source for resume')
    if path != Path(cfg['outdir']) / MANIFEST:
        raise ValueError('Manifest output binding mismatch')
    if m.get('plan_sha256') != base.object_sha({k: v for k, v in m.items() if k != 'plan_sha256'}):
        raise ValueError('v2 manifest fingerprint mismatch')
    if m['inputs']['fingerprint'] != base.object_sha(m['inputs']['splits']):
        raise ValueError('v2 input fingerprint mismatch')
    return m


def datasets_for(m):
    from data_utils.feeder import PersonInWiFi3D
    result = {}
    for mode, split in (('training', 'train_data'), ('validation', 'test_data')):
        ds = PersonInWiFi3D(mode, m['cfg']['dataset_root'], 'one-person', 1)
        if [i['name'] for i in ds.items] != m['inputs']['splits'][split]['ordered_names']:
            raise ValueError('Dataset identities/order changed')
        result[mode] = ds
    return result


def prepare(cfg):
    import numpy as np
    from piw_v2_adapter import (AXIS_MIN_METRES, PoisonedPairs, build_trigger,
                               geometry_audit, target_contract, trigger_record)
    if cfg['target'] != target_contract():
        raise ValueError('Configured target differs from v2 implementation')
    inputs, datasets = base.audit_inputs(cfg)
    trigger = build_trigger(cfg['action_npy'])
    poisoned = PoisonedPairs(datasets['training'], trigger)
    # Validate every hip axis, without selecting targets or scoring TEST poses.
    for ds in datasets.values():
        for item in ds.items:
            pose = np.load(item['kpt'], allow_pickle=False).reshape(14, 3)
            if np.linalg.norm(pose[5] - pose[7]) <= AXIS_MIN_METRES:
                raise ValueError(f'Degenerate hip axis: {item["name"]}; no frame omission')
    ds = datasets['training']
    indices = np.unique(np.linspace(0, len(ds) - 1, min(300, len(ds)), dtype=int))
    geometry = geometry_audit(np.stack([ds.load_pose(ds.items[i]['kpt'])[0] for i in indices]))
    geometry['ordered_names'] = [ds.items[i]['name'] for i in indices]
    distortion = []
    for dose in cfg['eval_doses']:
        l2, clipped = [], []
        for i in indices[:16]:
            x = ds[int(i)]['csi']
            attacked = trigger.inject(x, dose)
            if not np.array_equal(x[:, 90:], attacked[:, 90:]):
                raise ValueError('Trigger overwrote phase')
            delta = attacked[:, :90] - x[:, :90]
            l2.append(float(np.linalg.norm(delta) / max(np.linalg.norm(x[:, :90]), 1e-12)))
            clipped.append(float(np.mean((attacked[:, :90] == 0) | (attacked[:, :90] == 1))))
        distortion.append(dict(dose=dose, amplitude_relative_l2_mean=float(np.mean(l2)),
                               amplitude_boundary_fraction_mean=float(np.mean(clipped))))
    truth = np.stack([datasets['validation'].load_pose(i['kpt'])[0]
                      for i in datasets['validation'].items])
    context = dict(inputs=inputs, trigger=trigger_record(trigger), poison=poisoned.manifest(),
                   test_truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest())
    diagnostics = dict(geometry=geometry, train_distortion=distortion,
                       note='Boundary fraction includes pre-existing boundaries; not a clipping-only counter')
    return context, datasets, poisoned, trigger, diagnostics


def seed_execution(cfg):
    import numpy as np
    import torch
    execution = base._setup_execution(cfg['device'])
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if cfg['device'].startswith('cuda:'):
        torch.cuda.manual_seed(cfg['seed'])
    return execution


def optimizer_for(model, cfg):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                             betas=tuple(cfg['betas']), eps=cfg['eps'])


def checkpoint(m, role, execution=None, final=False):
    import torch
    blob = torch.load(cell(m, role) / 'checkpoint.pt', map_location='cpu', weights_only=False)
    if (blob.get('profile') != PROFILE or blob.get('role') != role
            or blob.get('plan_sha256') != m['plan_sha256'] or blob.get('cfg') != m['cfg']
            or type(blob.get('epoch')) is not int or not 0 <= blob['epoch'] < m['cfg']['epochs']
            or any(k not in blob for k in ('model', 'optimizer', 'rng', 'execution'))):
        raise ValueError('Foreign/incomplete checkpoint; v1 and v2 cannot be interchanged')
    if execution is not None and blob['execution'] != execution:
        raise ValueError('Execution backend changed; refusing resume')
    if final and blob['epoch'] != m['cfg']['final_epoch']:
        raise ValueError('Evaluation requires frozen final epoch199')
    return blob


def step(model, optimizer, batch, device):
    import torch
    x = batch['csi'].to(device, dtype=torch.float32).contiguous()
    y = batch['pose'].to(device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    pred, _ = model(x)
    if pred.shape != y.shape:
        raise ValueError('PiW output/label shape mismatch')
    loss = torch.linalg.vector_norm(pred - y, dim=-1).mean()
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite MPJPE; no retry or optimizer update')
    loss.backward()
    gradients = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not gradients or not torch.stack(gradients).all():
        raise RuntimeError('Nonfinite or absent gradients; no optimizer update')
    optimizer.step()
    return float(loss.detach()), len(x)


def train(m, role):
    from torch.utils.data import DataLoader
    from piw_v2_adapter import PoisonedPairs, build_trigger, trigger_record
    cfg, folder = m['cfg'], cell(m, role)
    execution = seed_execution(cfg)
    if execution != m['capacity_receipt']['execution']:
        raise ValueError('CUDA backend differs from capacity receipt; no fallback')
    model = base.build_model().to(cfg['device'])
    optimizer = optimizer_for(model, cfg)
    datasets = datasets_for(m)
    trigger = build_trigger(cfg['action_npy'])
    poisoned = PoisonedPairs(datasets['training'], trigger)
    if trigger_record(trigger) != m['trigger'] or poisoned.manifest() != m['poison']:
        raise ValueError('Trigger or poison plan changed')
    dataset = poisoned if role == 'a1' else datasets['training']
    start = 0
    if (folder / 'checkpoint.pt').exists():
        saved = checkpoint(m, role, execution)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        base._restore_rng(saved['rng'], cfg['device'])
        start = saved['epoch'] + 1
        print(f'[PiW v2] {role} resumed epoch{start - 1}', flush=True)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=0, drop_last=cfg['drop_last'])
    expected = len(dataset) // cfg['batch_size'] * cfg['batch_size']
    print(f'[PiW v2] {role}: scratch, AdamW lr{cfg["lr"]}, B32, FP32, final199; '
          f'TRAIN={len(dataset)}, consumed={expected}/epoch, TEST deferred', flush=True)
    for epoch in range(start, cfg['epochs']):
        model.train()
        total, count, began = 0., 0, time.monotonic()
        for batch in loader:
            loss, n = step(model, optimizer, batch, cfg['device'])
            total += loss * n; count += n
        if count != expected or not count:
            raise RuntimeError('Unexpected training coverage')
        base.atomic_checkpoint(folder / 'checkpoint.pt', dict(profile=PROFILE, role=role,
            plan_sha256=m['plan_sha256'], cfg=cfg, epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(), rng=base._rng_state(cfg['device']),
            execution=execution, train_loss=total/count))
        print(f'epoch {epoch}/{cfg["final_epoch"]}: loss={total/count:.8f} '
              f'lr={cfg["lr"]:.2e} t={time.monotonic()-began:.1f}s', flush=True)
        print(f'[ckpt] saved -> {folder / "checkpoint.pt"} (epoch{epoch})', flush=True)
    checkpoint(m, role, execution, final=True)
    print(f'[PiW v2] {role} TRAIN_COMPLETE; final evaluation follows both victims', flush=True)


def infer(model, dataset, cfg, trigger=None, dose=0.):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    predictions, truths, names = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False,
                                num_workers=0, drop_last=False):
            x = batch['csi']
            if trigger is not None and dose:
                x = torch.from_numpy(np.stack([trigger.inject(a, dose) for a in x.numpy()]))
            pred, _ = model(x.to(cfg['device'], dtype=torch.float32))
            predictions.append(pred[:, 0].cpu().numpy())
            truths.append(batch['pose'][:, 0].numpy()); names.extend(batch['name'])
    return np.concatenate(predictions), np.concatenate(truths), names


def atomic_npz(path, **arrays):
    import numpy as np
    temporary = path.with_name(path.name + '.saving')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        base.atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate(m):
    import numpy as np
    from piw_v2_adapter import build_trigger, measure_predictions, target_pose, trigger_record
    cfg = m['cfg']
    execution = base._setup_execution(cfg['device'])
    models = {}
    for role in ('clean', 'a1'):
        saved = checkpoint(m, role, execution, final=True)
        models[role] = base.build_model().to(cfg['device'])
        models[role].load_state_dict(saved['model'], strict=True)
    dataset = datasets_for(m)['validation']
    trigger = build_trigger(cfg['action_npy'])
    if trigger_record(trigger) != m['trigger']:
        raise ValueError('Action/trigger changed before evaluation')
    outputs, truth = {}, None
    names = m['inputs']['splits']['test_data']['ordered_names']
    for role, model in models.items():
        pred, y, ids = infer(model, dataset, cfg)
        if ids != names or (truth is not None and not np.array_equal(y, truth)):
            raise ValueError('Clean/A1 TEST identity/truth mismatch')
        truth = y
        results = [pred.copy()]
        for dose in cfg['eval_doses'][1:]:
            print(f'[PiW v2 eval] {role} full TEST d={dose:g}', flush=True)
            prediction, y, ids = infer(model, dataset, cfg, trigger, dose)
            if ids != names or not np.array_equal(y, truth):
                raise ValueError('Dose TEST identity/truth mismatch')
            results.append(prediction)
        outputs[role] = (pred, np.stack(results, axis=1))
    clean_pred, clean_trigger = outputs['clean']
    a1_pred, a1_trigger = outputs['a1']
    arrays = dict(truth=truth, names=np.asarray(names), doses=np.asarray(cfg['eval_doses']),
        predictions=a1_pred, trigger_predictions=a1_trigger,
        reference_predictions=clean_pred, reference_trigger_predictions=clean_trigger,
        targets=np.stack([target_pose(truth, d) for d in cfg['eval_doses']], axis=1))
    root = Path(cfg['outdir'])
    atomic_npz(root / 'predictions.npz', **arrays)
    result = dict(profile=PROFILE, status=STATUS, plan_sha256=m['plan_sha256'],
        clean_metrics=base._metrics(clean_pred, truth),
        a1_metrics=measure_predictions(truth, a1_pred, a1_trigger, clean_pred, clean_trigger),
        artifacts=artifact_hashes(m), execution=execution)
    # One authoritative publication marker: no split JSON completion state.
    base.atomic_json(root / 'evaluation.json', result)
    verify_complete(m)
    export_report(m, result)
    print('[PiW v2] COMPLETE Clean+A1; inspect six-dose paired controls', flush=True)


def artifact_hashes(m):
    root = Path(m['cfg']['outdir'])
    paths = ['predictions.npz', 'poison.json', 'hpeli/seed42/clean/checkpoint.pt',
             'hpeli/seed42/a1/checkpoint.pt']
    return {p: base.file_sha(root / p) for p in paths}


def replay_equal(a, b):
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            raise ValueError('Saved metric schema differs')
        for k in b:
            replay_equal(a[k], b[k])
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            raise ValueError('Saved dose rows differ')
        for x, y in zip(a, b):
            replay_equal(x, y)
    elif isinstance(b, (int, float)) and not isinstance(b, bool):
        if type(a) not in (int, float) or not math.isfinite(a) or not math.isclose(a, b, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Saved metrics do not replay from per-sample predictions')
    elif a != b:
        raise ValueError('Saved metric metadata differs')


def verify_complete(m):
    import numpy as np
    from piw_v2_adapter import measure_predictions, target_pose
    root = Path(m['cfg']['outdir'])
    if not (root / 'evaluation.json').exists():
        return None
    r = json.loads((root / 'evaluation.json').read_text(encoding='utf-8'))
    if (r.get('profile') != PROFILE or r.get('status') != STATUS
            or r.get('plan_sha256') != m['plan_sha256'] or r.get('artifacts') != artifact_hashes(m)):
        raise ValueError('Saved final artifact binding mismatch')
    for role in ('clean', 'a1'):
        checkpoint(m, role, r['execution'], final=True)
    if json.loads((root / 'poison.json').read_text(encoding='utf-8')) != m['poison']:
        raise ValueError('Poison artifact mismatch')
    with np.load(root / 'predictions.npz', allow_pickle=False) as p:
        if set(p.files) != {'truth', 'names', 'doses', 'predictions', 'trigger_predictions',
                           'reference_predictions', 'reference_trigger_predictions', 'targets'}:
            raise ValueError('Unexpected v2 prediction schema')
        if hashlib.sha256(p['truth'].tobytes()).hexdigest() != m['test_truth_sha256']:
            raise ValueError('Saved TEST truth differs from audited clean GT')
        if p['names'].tolist() != m['inputs']['splits']['test_data']['ordered_names']:
            raise ValueError('Saved TEST order mismatch')
        if not np.array_equal(p['doses'], m['cfg']['eval_doses']):
            raise ValueError('Saved evaluation doses mismatch')
        expected_targets = np.stack([target_pose(p['truth'], d) for d in m['cfg']['eval_doses']], axis=1)
        if not np.array_equal(p['targets'], expected_targets):
            raise ValueError('Saved target geometry mismatch')
        replay_equal(r['clean_metrics'], base._metrics(p['reference_predictions'], p['truth']))
        replay_equal(r['a1_metrics'], measure_predictions(p['truth'], p['predictions'],
            p['trigger_predictions'], p['reference_predictions'], p['reference_trigger_predictions']))
    return r


def export_report(m, result):
    import csv
    root = Path(m['cfg']['outdir'])
    c, a = result['clean_metrics'], result['a1_metrics']['mainmetrics']
    text = ('# EXPLORATORY_PIW_V2_ONE_SEED_NOT_FULL_PAPER_TABLE\n\n'
        'Clean+A1 independent scratch HPE-Li; PiW one-person, 14 joints, seed42.\n'
        '200 epochs/final199; AdamW lr.01/wd.01; B32/drop_last=True; ordinary MPJPE; FP32.\n'
        'Target: right hip7, joints11/13, +40deg*d around normalize(clean_GT5-clean_GT7).\n'
        'PCK@.2 uses clean GT elbows6/4; errors mm. Trigger/rho/epsilon unchanged from v1.\n'
        'Recipe and target changed after a failed pilot; development-selected adaptation. '
        'Not blind TEST, no across-seed uncertainty, no causal attribution to one change.\n'
        'Not exact original DT-Pose/HPE-Li model reproduction; current corrected SK attention retained.\n\n'
        '| Method | T-MPJPE@1 mm | Target-PCK % | Clean MPJPE mm | Clean PA mm | Mean DCE mm |\n'
        '| --- | ---: | ---: | ---: | ---: | ---: |\n'
        f'| Clean | -- | -- | {c["clean_mpjpe_mm"]:.4f} | {c["clean_pampjpe_mm"]:.4f} | -- |\n'
        f'| A1 v2 | {a["t1_mpjpe_mm"]:.4f} | {a["target_pck_pct"]:.4f} | '
        f'{a["clean_mpjpe_mm"]:.4f} | {a["clean_pampjpe_mm"]:.4f} | {a["mean_dce_mm"]:.4f} |\n')
    temporary = root / 'piw_v2_pilot.md.saving'
    temporary.write_text(text, encoding='utf-8')
    base.atomic_replace(temporary, root / 'piw_v2_pilot.md')
    rows = result['a1_metrics']['dose_rows']
    temporary = root / 'a1_six_doses.csv.saving'
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    base.atomic_replace(temporary, root / 'a1_six_doses.csv')


def receipt_path(cfg, kind):
    return Path(cfg['outdir'] + f'.{kind}.json')


def receipt(cfg, kind):
    record = json.loads(receipt_path(cfg, kind).read_text(encoding='utf-8'))
    expected = base.object_sha(plan(cfg))
    if record.get('plan_sha256') != expected or record.get('kind') != kind:
        raise ValueError(f'{kind} receipt belongs to different source/config/runtime')
    if record.get('status') != 'PASS':
        raise ValueError(f'{kind} did not pass; no automatic fallback')
    return record


def cpu_preflight(cfg):
    import numpy as np
    import torch
    from piw_v2_adapter import target_pose
    context, ds, poisoned, trigger, diagnostics = prepare(cfg)
    model = base.build_model().cpu().train()
    optimizer = optimizer_for(model, cfg)
    # Actual physical B32 with the same ordinary MPJPE and AdamW as production.
    batch = [poisoned[i % len(poisoned)] for i in range(cfg['batch_size'])]
    x = np.stack([b['csi'] for b in batch]); y = np.stack([b['pose'] for b in batch])
    loss, n = step(model, optimizer, dict(csi=torch.from_numpy(x), pose=torch.from_numpy(y)), 'cpu')
    original = ds['training'][0]['pose']
    moved = target_pose(original, 1.)
    for a, b in ((7, 11), (11, 13)):
        if not np.allclose(np.linalg.norm(moved[..., a, :] - moved[..., b, :], axis=-1),
                           np.linalg.norm(original[..., a, :] - original[..., b, :], axis=-1), atol=1e-7):
            raise ValueError('Target bone length mismatch')
    path = receipt_path(cfg, 'preflight')
    path.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(path, dict(kind='preflight', status='PASS', plan_sha256=base.object_sha(plan(cfg)),
        input_fingerprint=context['inputs']['fingerprint'], trigger=context['trigger'],
        poison_sha256=base.object_sha(context['poison']), diagnostics=diagnostics,
        cpu_smoke=dict(batch=n, loss=loss, finite=True),
        note='CPU/data/geometry checks only; no CUDA capacity or learned efficacy claim'))
    print(f'PREFLIGHT_PASS: {path}; no full training launched', flush=True)


def wait_for_files(paths, children, seconds=120):
    deadline = time.monotonic() + seconds
    while not all(p.exists() for p in paths):
        failed = [p.returncode for p in children if p.poll() is not None]
        if failed:
            raise RuntimeError(f'Smoke child exited before barrier: {failed}')
        if time.monotonic() > deadline:
            raise RuntimeError('CUDA smoke barrier timed out; no full training launched')
        time.sleep(.1)


def smoke_worker(path, slot):
    import torch
    cfg = json.loads(path.read_text(encoding='utf-8'))['cfg']
    folder = path.parent
    execution = seed_execution(cfg)
    model = base.build_model().to(cfg['device']).train()
    optimizer = optimizer_for(model, cfg)
    torch.cuda.reset_peak_memory_stats(cfg['device'])
    x = torch.rand(cfg['batch_size'], 3, 180, 20, device=cfg['device'])
    y = torch.rand(cfg['batch_size'], 1, 14, 3, device=cfg['device'])
    optimizer.zero_grad(set_to_none=True)
    pred, _ = model(x)
    loss = torch.linalg.vector_norm(pred - y, dim=-1).mean()
    (folder / f'{slot}.ready').touch()
    wait_for_files([folder / 'go'], [], seconds=120)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite CUDA smoke loss')
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Nonfinite CUDA smoke gradients')
    optimizer.step()
    step(model, optimizer, dict(csi=x, pose=y), cfg['device'])
    torch.cuda.synchronize(cfg['device'])
    base.atomic_json(folder / f'{slot}.json', dict(slot=slot, execution=execution,
        batch_size=cfg['batch_size'], steps=2,
        allocated_peak_bytes=torch.cuda.max_memory_allocated(cfg['device']),
        reserved_peak_bytes=torch.cuda.max_memory_reserved(cfg['device'])))
    # Keep model/optimizer resident until every concurrent worker finishes.
    wait_for_files([folder / 'release'], [], seconds=120)


def gpu_smoke(cfg):
    import tempfile
    if not cfg['device'].startswith('cuda:'):
        raise ValueError('Capacity smoke requires CUDA; no CPU receipt substitute')
    prep = receipt(cfg, 'preflight')
    target = receipt_path(cfg, 'gpu_smoke')
    running = dict(kind='gpu_smoke', status='RUNNING', plan_sha256=base.object_sha(plan(cfg)),
                   input_fingerprint=prep['input_fingerprint'])
    base.atomic_json(target, running)
    workers, streams = [], []
    with tempfile.TemporaryDirectory(prefix='piw_v2_cuda_') as directory:
        folder = Path(directory)
        path = folder / 'plan.json'
        base.atomic_json(path, plan(cfg))
        try:
            for slot in range(cfg['jobs_per_gpu']):
                stream = (folder / f'{slot}.log').open('wb'); streams.append(stream)
                workers.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--_smoke-plan', str(path), '--_slot', str(slot)], stdout=stream,
                    stderr=subprocess.STDOUT, cwd=str(HERE)))
            wait_for_files([folder / f'{s}.ready' for s in range(len(workers))], workers)
            (folder / 'go').touch()
            wait_for_files([folder / f'{s}.json' for s in range(len(workers))], workers)
            (folder / 'release').touch()
            codes = [w.wait(timeout=120) for w in workers]
            if any(codes):
                raise RuntimeError(f'CUDA smoke failed: {codes}')
            records = [json.loads((folder / f'{s}.json').read_text()) for s in range(len(workers))]
            if any(r['execution'] != records[0]['execution'] for r in records):
                raise ValueError('CUDA smoke devices differ')
            base.atomic_json(target, dict(kind='gpu_smoke', status='PASS',
                plan_sha256=base.object_sha(plan(cfg)), input_fingerprint=prep['input_fingerprint'],
                execution=records[0]['execution'], records=records,
                note='Synthetic concurrent B32 capacity only; not full-dataset efficacy'))
            print(f'GPU_SMOKE_PASS: {target}; no full training launched', flush=True)
        except BaseException as error:
            base.atomic_json(target, dict(running, status='FAILED', error=str(error)))
            # Only these disposable smoke children, never another training job.
            for w in workers:
                if w.poll() is None:
                    w.terminate()
            for w in workers:
                w.wait(timeout=30)
            for stream in streams:
                stream.flush()
            for slot in range(len(workers)):
                log = folder / f'{slot}.log'
                if log.exists():
                    print(log.read_text(encoding='utf-8', errors='replace')[-5000:], file=sys.stderr)
            raise
        finally:
            for stream in streams:
                stream.close()


def launch(cfg):
    from run_mmfi_tables import _cell_lock
    prep, smoke = receipt(cfg, 'preflight'), receipt(cfg, 'gpu_smoke')
    root = Path(cfg['outdir'])
    # Refuse old/foreign artifacts before creating a run manifest.
    if (root.exists() and not (root / MANIFEST).exists()
            and any(p.name != 'run.lock' for p in root.iterdir())):
        raise ValueError('Output is not an empty v2 run; choose a new directory')
    with _cell_lock(root):
        path = root / MANIFEST
        old = load_manifest(path) if path.exists() else None
        if old is not None and any(old.get(k) != v for k, v in plan(cfg).items()):
            raise ValueError('Requested launch config differs from existing v2 run')
        if old is not None and old['capacity_receipt']['execution'] != smoke['execution']:
            raise ValueError('Capacity execution changed; refusing scientific resume')
        if old is not None and verify_complete(old) is not None:
            export_report(old, verify_complete(old))
            print('VERIFIED_COMPLETE; no training/inference', flush=True)
            return
        context, _, _, _, diagnostics = prepare(cfg)
        if (context['inputs']['fingerprint'] != prep['input_fingerprint']
                or smoke['input_fingerprint'] != prep['input_fingerprint']
                or context['trigger'] != prep['trigger']
                or base.object_sha(context['poison']) != prep['poison_sha256']):
            raise ValueError('Inputs/trigger/poison changed since preflight')
        if old is not None:
            if any(old[k] != v for k, v in context.items()):
                raise ValueError('Inputs changed; refusing resume')
            m = old
        else:
            m = dict(plan(cfg), **context, created_utc=datetime.now(timezone.utc).isoformat(),
                     capacity_receipt=smoke)
            m['plan_sha256'] = base.object_sha(m)
            base.atomic_json(path, m)
        poison_path = root / 'poison.json'
        if poison_path.exists() and json.loads(poison_path.read_text()) != m['poison']:
            raise ValueError('Existing poison artifact differs')
        base.atomic_json(poison_path, m['poison'])
        base.atomic_json(root / 'train_diagnostics.json', diagnostics)
        roles = ['clean', 'a1']
        for first in range(0, len(roles), cfg['jobs_per_gpu']):
            children, streams = [], []
            try:
                for role in roles[first:first + cfg['jobs_per_gpu']]:
                    folder = cell(m, role); folder.mkdir(parents=True, exist_ok=True)
                    stream = (folder / 'console.log').open('ab'); streams.append(stream)
                    child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                        '--_manifest', str(path), '--_train', role], cwd=str(HERE), stdout=stream,
                        stderr=subprocess.STDOUT)
                    children.append(child)
                    print(f'[PiW v2] TRAIN {role}; PID={child.pid}; log={folder / "console.log"}', flush=True)
                while any(p.poll() is None for p in children):
                    failed = [p.returncode for p in children if p.poll() is not None and p.returncode]
                    if failed:
                        # Running peers keep their current epoch until they finish.
                        print(f'Worker failure {failed}; waiting for own peer to exit; no retry', flush=True)
                        break
                    time.sleep(1)
                codes = [p.wait() for p in children]
                if any(codes):
                    raise RuntimeError(f'Training failed {codes}; inspect cell logs; no retry')
            except KeyboardInterrupt:
                for p in children:
                    if p.poll() is None:
                        p.terminate()
                for p in children:
                    p.wait()
                raise
            finally:
                for stream in streams:
                    stream.close()
        # Confirm unchanged full inputs before first and only final TEST evaluation.
        final_inputs, _ = base.audit_inputs(cfg)
        if final_inputs != m['inputs']:
            raise ValueError('Dataset changed during training; refusing evaluation')
        with (root / 'evaluation.log').open('ab') as stream:
            code = subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()),
                '--_manifest', str(path), '--_evaluate'], cwd=str(HERE), stdout=stream,
                stderr=subprocess.STDOUT)
        if code or verify_complete(m) is None:
            raise RuntimeError(f'Final evaluation failed({code}); inspect evaluation.log; no retry')
        print(f'COMPLETE: {root / "piw_v2_pilot.md"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--action-npy', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--jobs-per-gpu', type=int, default=2, choices=(1, 2))
    modes = parser.add_mutually_exclusive_group()
    for mode in ('dry-run', 'preflight', 'gpu-smoke', 'launch', 'status', 'check-only', 'export-only'):
        modes.add_argument('--' + mode, action='store_true')
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--_train', choices=('clean', 'a1'), help=argparse.SUPPRESS)
    parser.add_argument('--_evaluate', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_smoke-plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--_slot', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    public_modes = any(getattr(args, m.replace('-', '_')) for m in
                       ('dry-run', 'preflight', 'gpu-smoke', 'launch', 'status', 'check-only', 'export-only'))
    if args._smoke_plan is not None:
        if public_modes or args._manifest or args._slot is None or args._train or args._evaluate:
            parser.error('Invalid private smoke invocation')
        smoke_worker(args._smoke_plan, args._slot)
        return 0
    if args._manifest is not None:
        if public_modes or bool(args._train) == bool(args._evaluate) or args._slot is not None:
            parser.error('Invalid private worker invocation')
        m = load_manifest(args._manifest)
        if args._train:
            from run_mmfi_tables import _cell_lock
            with _cell_lock(cell(m, args._train)):
                train(m, args._train)
        else:
            evaluate(m)
        return 0
    if args._train or args._evaluate or args._slot is not None:
        parser.error('Private modes require their bound manifest')
    if args.status or args.check_only or args.export_only:
        if args.outdir is None:
            parser.error('--outdir required for saved-artifact audit')
        m = load_manifest(args.outdir / MANIFEST)
        if args.check_only:
            current, _ = base.audit_inputs(m['cfg'])
            if current != m['inputs']:
                raise ValueError('Dataset fingerprint differs')
        r = verify_complete(m)
        if r is None:
            for role in ('clean', 'a1'):
                p = cell(m, role) / 'checkpoint.pt'
                print(f'{role}: epoch{checkpoint(m, role)["epoch"]}' if p.exists() else f'{role}: no checkpoint')
            print('INCOMPLETE; no training/inference')
            return 1
        if args.export_only:
            export_report(m, r)
        print('VERIFIED_SAVED_ARTIFACTS_COMPLETE; no model inference/training')
        return 0
    home = args.data_home or Path.home() / 'backdooranalog'
    output = args.outdir or home / 'runs/piw_hpeli_body_axis_s42_v2'
    cfg = configuration(home, output, args.dataset_root, args.action_npy,
                        args.device, args.jobs_per_gpu)
    if args.preflight:
        cpu_preflight(cfg)
    elif args.gpu_smoke:
        gpu_smoke(cfg)
    elif args.launch:
        launch(cfg)
    else:
        print('DRY_RUN_ONLY: new Clean+A1, no data/GPU/training/output writes')
        print(json.dumps(plan(cfg), indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print(f'[PiW v2] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
