"""Exploratory PiW3D HPE-Li seed42 attack comparison, excluding BadNets.

The completed LR-control v3 Clean/A1 run is an immutable paired reference.
Five independently initialized baseline victims use its TRAIN split, poison
identities/doses, target geometry and AdamW/200-epoch recipe. No TEST model
selection. DryRun, Preflight and a concurrent two-process GPU smoke are
separate gates before Launch. Results are development-selected, not a blind
external validation or an unchanged source-image-attack reproduction.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import run_piw_hpeli_lr_control as v3
from piw_native_attack_adapter import GroupedTrigger, PoisonedPairs, METHODS, COVERS

HERE = Path(__file__).resolve().parent
PROFILE = 'piw_hpeli_native_attacks_no_badnets_s42_v1'
MANIFEST = 'piw_native_attacks.resolved.json'
SOURCES = (*v3.SOURCE_NAMES, 'run_piw_hpeli_lr_control.py',
           'run_piw_hpeli_attack_suite.py', 'piw_native_attack_adapter.py',
           'attack/wanet_source.py', 'attack/frequency_baselines.py',
           'attack/static_source_baselines.py', 'attack/traditional.py')


def sha(path):
    return v3.base.file_sha(Path(path))


def reference(data_home, reference_dir=None):
    root = Path(reference_dir or Path(data_home) / 'runs/piw_hpeli_body_axis_s42_lr001_v3').resolve()
    saved = v3.load_manifest(root / v3.MANIFEST)
    report = v3.verify_complete(saved)
    if report is None:
        raise ValueError('PiW LR-control v3 Clean+A1 is not complete')
    cfg = saved['cfg']
    if (cfg['seed'] != 42 or cfg['lr'] != .001 or cfg['batch_size'] != 32
            or cfg['epochs'] != 200 or cfg['final_epoch'] != 199
            or cfg['target']['version'] != 'piw_body_hip_axis_v2'):
        raise ValueError('The frozen PiW v3 reference has a different scientific recipe')
    return root, saved, report


def build_plan(data_home, outdir, reference_dir=None, jobs_per_gpu=2):
    if jobs_per_gpu not in (1, 2) or type(jobs_per_gpu) is not int:
        raise ValueError('Select 1 or 2 independent HPE-Li workers')
    home, output = Path(data_home).resolve(), Path(outdir).resolve()
    root, ref, _ = reference(home, reference_dir)
    if output == root or output in root.parents or root in output.parents:
        raise ValueError('New output must be separate from historical PiW v3')
    cfg = copy.deepcopy(ref['cfg'])
    cfg.update(profile=PROFILE, outdir=str(output), jobs_per_gpu=jobs_per_gpu,
        roles=list(METHODS), status='EXPLORATORY_PIW_NATIVE_ATTACK_COMPARISON',
        trigger_policy='five source-derived digital operators on each of three physical normalized-amplitude groups; phase unchanged',
        initialization='independent scratch victims, same fixed HPE-Li recipe as completed PiW v3',
        selection='fixed final199; TEST deferred until all five victims finish')
    bindings = {name: sha(root / name) for name in (
        v3.MANIFEST, 'evaluation.json', 'predictions.npz',
        'hpeli/seed42/clean/checkpoint.pt', 'hpeli/seed42/a1/checkpoint.pt')}
    plan = dict(schema=1, profile=PROFILE, methods=list(METHODS),
        excluded=['badnets'], reference=str(root), reference_sha256=bindings,
        cfg=cfg, inputs_fingerprint=ref['inputs']['fingerprint'],
        poison_plan_sha256=ref['poison']['poison_plan_sha256'],
        sources={name: sha(HERE / name) for name in dict.fromkeys(SOURCES)},
        runtime=v3.base.runtime_versions(),
        policy=dict(seed=42, epochs=200, final_epoch=199, batch_size=32,
            rho=.1, poison_doses='paired U(.2,1) from PiW v3',
            cover_ratios=COVERS, jobs_per_gpu=jobs_per_gpu,
            target='frozen PiW v3 body-relative right-leg 40-degree target',
            phase='unchanged; source operator applied to each normalized 30-bin physical group',
            test_selection=False, no_amp=True, no_batch_fallback=True,
            no_automatic_retry=True, status='EXPLORATORY_ONE_SEED_DEVELOPMENT_SELECTED'))
    plan['sha256'] = v3.base.object_sha(plan)
    return plan


def verify_plan(saved):
    if (saved.get('profile') != PROFILE or saved.get('methods') != list(METHODS)
            or saved.get('excluded') != ['badnets']
            or saved.get('sha256') != v3.base.object_sha({k: value for k, value in saved.items() if k != 'sha256'})):
        raise ValueError('Invalid PiW attack suite plan')
    actual = build_plan(saved['cfg']['data_home'], saved['cfg']['outdir'],
                        saved['reference'], saved['cfg']['jobs_per_gpu'])
    if saved != actual:
        raise ValueError('PiW suite data/source/runtime/reference/config changed')
    return saved


def receipt_path(plan, kind):
    return Path(plan['cfg']['outdir'] + f'.{kind}.json')


def receipt(plan, kind):
    record = json.loads(receipt_path(plan, kind).read_text(encoding='utf-8'))
    if (record.get('kind') != kind or record.get('status') != 'PASS'
            or record.get('plan_sha256') != plan['sha256']):
        raise ValueError(f'PiW {kind} gate did not pass for this exact plan')
    return record


def datasets(plan):
    root, ref, _ = reference(plan['cfg']['data_home'], plan['reference'])
    if root != Path(plan['reference']) or ref['inputs']['fingerprint'] != plan['inputs_fingerprint']:
        raise ValueError('PiW v3 reference input binding changed')
    return ref, v3.datasets_for(ref)


def validate_operators(plan, train, poison):
    import numpy as np
    # FIBA uses TRAIN[0] as its frozen frequency reference.  Test distinct
    # recipients so an identity blend of the reference cannot pass the gate.
    probes = [train[i] for i in (1, max(2, len(train) // 2))]
    for method in METHODS:
        trigger = GroupedTrigger(method, train)
        pairs = PoisonedPairs(train, trigger, poison)
        if pairs.record['poison_sha256'] != plan['poison_plan_sha256']:
            raise ValueError('PiW poison plan changed')
        changed = False
        for clean in probes:
            x = clean['csi']
            if not np.array_equal(trigger.inject(x, 0), x):
                raise ValueError(f'{method}: dose zero is not identity')
            y = trigger.inject(x, 1.)
            if not np.array_equal(y[:, 90:], x[:, 90:]):
                raise ValueError(f'{method}: phase changed')
            changed |= bool(np.max(np.abs(y[:, :90] - x[:, :90])) > 1e-8)
            if pairs.covers:
                z = trigger.noise_inject(x, dose=1.)
                if not np.array_equal(z[:, 90:], x[:, 90:]):
                    raise ValueError(f'{method}: cover changed phase')
        if not changed:
            raise ValueError(f'{method}: amplitude trigger empty on TRAIN probes')
    return True


def preflight(plan):
    ref, ds = datasets(plan)
    current, _ = v3.base.audit_inputs(ref['cfg'])
    if current != ref['inputs']:
        raise ValueError('PiW TRAIN/TEST content changed since Clean+A1 v3')
    validate_operators(plan, ds['training'], ref['poison'])
    target = receipt_path(plan, 'preflight')
    if target.exists():
        receipt(plan, 'preflight')
        print('PREFLIGHT_PASS_EXISTING; no training', flush=True)
        return
    v3.base.atomic_json(target, dict(kind='preflight', status='PASS',
        plan_sha256=plan['sha256'], input_fingerprint=plan['inputs_fingerprint'],
        n_train=len(ds['training']), n_test=len(ds['validation']),
        methods=list(METHODS), badnets_excluded=True,
        note='Full input binding and source operators; CPU only, no training'))
    print(f'PREFLIGHT_PASS: {target}; no training', flush=True)


def gpu_smoke(plan):
    prep = receipt(plan, 'preflight')
    target = receipt_path(plan, 'gpu_smoke')
    if target.exists():
        previous = json.loads(target.read_text(encoding='utf-8'))
        if previous.get('status') == 'PASS':
            receipt(plan, 'gpu_smoke')
            print('GPU_SMOKE_PASS_EXISTING; no training', flush=True)
            return
        if previous.get('status') != 'FAILED':
            raise ValueError('GPU smoke is in progress or incomplete; inspect it before retry')
        # A fresh explicit --gpu-smoke is a manual retry. Retain the failure.
        archived = target.with_name(target.name + '.failed.' +
            datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '.json')
        archived.write_text(json.dumps(previous, indent=2) + '\n', encoding='utf-8')
    if not plan['cfg']['device'].startswith('cuda:'):
        raise ValueError('CUDA smoke requires an explicit GPU')
    v3.base.atomic_json(target, dict(kind='gpu_smoke', status='RUNNING',
                                    plan_sha256=plan['sha256']))
    workers, streams = [], []
    with tempfile.TemporaryDirectory(prefix='piw_native_cuda_') as folder_name:
        folder = Path(folder_name)
        v3.base.atomic_json(folder / 'plan.json', {'cfg': plan['cfg']})
        try:
            for slot in range(plan['cfg']['jobs_per_gpu']):
                stream = (folder / f'{slot}.log').open('wb'); streams.append(stream)
                workers.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--_smoke-plan', str(folder / 'plan.json'), '--_slot', str(slot)],
                    cwd=str(HERE), stdout=stream, stderr=subprocess.STDOUT))
            v3.wait_for_files([folder / f'{s}.ready' for s in range(len(workers))], workers)
            (folder / 'go').touch()
            v3.wait_for_files([folder / f'{s}.json' for s in range(len(workers))], workers)
            (folder / 'release').touch()
            codes = [w.wait(timeout=120) for w in workers]
            if any(codes):
                raise RuntimeError(f'PiW B32 pair smoke worker failure: {codes}')
            rows = [json.loads((folder / f'{s}.json').read_text()) for s in range(len(workers))]
            if any(r['batch_size'] != 32 or r['steps'] != 2 for r in rows):
                raise ValueError('PiW B32 smoke recipe changed')
            v3.base.atomic_json(target, dict(kind='gpu_smoke', status='PASS',
                plan_sha256=plan['sha256'], input_fingerprint=prep['input_fingerprint'],
                records=rows, note=f'{len(rows)} concurrent independent HPE-Li B32 worker(s); no full training'))
            print(f'GPU_SMOKE_PASS: {target}; no full training', flush=True)
        except BaseException as error:
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
                    print(log.read_text(encoding='utf-8', errors='replace')[-4000:], file=sys.stderr)
            v3.base.atomic_json(target, dict(kind='gpu_smoke', status='FAILED',
                                            plan_sha256=plan['sha256'], error=str(error)))
            raise
        finally:
            for stream in streams:
                stream.close()


def cell(plan, method):
    if method not in METHODS:
        raise ValueError('Unknown PiW baseline cell')
    return Path(plan['cfg']['outdir']) / 'hpeli/seed42' / method


def checkpoint(plan, method, *, final=False):
    import torch
    path = cell(plan, method) / 'checkpoint.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('profile') != PROFILE or saved.get('method') != method
            or saved.get('plan_sha256') != plan['sha256']
            or type(saved.get('epoch')) is not int
            or not 0 <= saved['epoch'] < plan['cfg']['epochs']
            or not all(k in saved for k in ('model', 'optimizer', 'rng', 'trigger_state'))):
        raise ValueError(f'{method}: foreign or incomplete checkpoint')
    if final and saved['epoch'] != plan['cfg']['final_epoch']:
        raise ValueError(f'{method}: fixed final199 not reached')
    return saved


def train(plan, method):
    import torch
    from torch.utils.data import DataLoader

    cfg = plan['cfg']
    execution = v3.seed_execution(cfg)
    smoke = receipt(plan, 'gpu_smoke')
    if (len(smoke.get('records', [])) != cfg['jobs_per_gpu'] or
            any(row.get('execution') != execution for row in smoke['records'])):
        raise ValueError('Training CUDA backend differs from the capacity smoke')
    model = v3.base.build_model().to(cfg['device'])
    optimizer = v3.optimizer_for(model, cfg)
    ref, ds = datasets(plan)
    trigger = GroupedTrigger(method, ds['training'])
    poisoned = PoisonedPairs(ds['training'], trigger, ref['poison'])
    folder = cell(plan, method); folder.mkdir(parents=True, exist_ok=True)
    start = 0
    if (folder / 'checkpoint.pt').exists():
        saved = checkpoint(plan, method)
        if saved['poison'] != poisoned.record or saved['execution'] != execution:
            raise ValueError(f'{method}: poison or GPU execution changed on resume')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        trigger.load_state_dict(saved['trigger_state'])
        v3.base._restore_rng(saved['rng'], cfg['device'])
        start = saved['epoch'] + 1
        print(f'[PiW attack] {method} resumed epoch{start - 1}', flush=True)
    loader = DataLoader(poisoned, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=0, drop_last=True)
    expected = len(poisoned) // cfg['batch_size'] * cfg['batch_size']
    print(f'[PiW attack] {method}: TRAIN={len(poisoned)}, poison={poisoned.record["n_poison"]}, '
          f'cover={poisoned.record["n_cover"]}; AdamW lr{cfg["lr"]}; B32; final199', flush=True)
    for epoch in range(start, cfg['epochs']):
        model.train()
        total, count, began = 0., 0, time.monotonic()
        for batch in loader:
            loss, n = v3.step(model, optimizer, batch, cfg['device'])
            total += loss * n; count += n
        if count != expected or count == 0:
            raise RuntimeError(f'{method}: unexpected TRAIN coverage')
        v3.base.atomic_checkpoint(folder / 'checkpoint.pt', dict(
            profile=PROFILE, method=method, plan_sha256=plan['sha256'],
            epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
            trigger_state=trigger.state_dict(), poison=poisoned.record,
            rng=v3.base._rng_state(cfg['device']), execution=execution,
            train_loss=total / count))
        print(f'epoch {epoch}/{cfg["final_epoch"]}: loss={total/count:.8f} '
              f'lr={cfg["lr"]:.2e} t={time.monotonic()-began:.1f}s', flush=True)
        print(f'[ckpt] saved -> {folder / "checkpoint.pt"} (epoch{epoch})', flush=True)
    checkpoint(plan, method, final=True)
    print(f'[PiW attack] TRAIN_COMPLETE {method}; TEST deferred', flush=True)


def evaluate(plan, method):
    import numpy as np

    cfg = plan['cfg']
    ref, ds = datasets(plan)
    trained = checkpoint(plan, method, final=True)
    clean_saved = v3.checkpoint(ref, 'clean', final=True)
    if trained['execution'] != v3.base._setup_execution(cfg['device']):
        raise ValueError('GPU execution changed before evaluation')
    models = {}
    for name, weights in (('victim', trained['model']), ('clean', clean_saved['model'])):
        model = v3.base.build_model().to(cfg['device'])
        model.load_state_dict(weights, strict=True)
        models[name] = model
    trigger = GroupedTrigger(method, ds['training'])
    test = ds['validation']
    names = ref['inputs']['splits']['test_data']['ordered_names']
    outputs, truth = {}, None
    for label, model in models.items():
        plain, y, ids = v3.infer(model, test, cfg)
        if ids != names or (truth is not None and not np.array_equal(y, truth)):
            raise ValueError('PiW TEST identities or GT changed')
        truth = y
        doses = [plain.copy()]
        for dose in cfg['eval_doses'][1:]:
            print(f'[PiW attack eval] {method}/{label} full TEST d={dose:g}', flush=True)
            prediction, yy, ids = v3.infer(model, test, cfg, trigger, dose)
            if ids != names or not np.array_equal(yy, truth):
                raise ValueError('PiW dose TEST order changed')
            doses.append(prediction)
        outputs[label] = (plain, np.stack(doses, axis=1))
    victim_plain, victim_doses = outputs['victim']
    clean_plain, clean_doses = outputs['clean']
    from piw_v2_adapter import measure_predictions, target_pose
    measured = measure_predictions(truth, victim_plain, victim_doses,
                                   clean_plain, clean_doses)
    folder = cell(plan, method)
    v3.atomic_npz(folder / 'predictions.npz', truth=truth,
        names=np.asarray(names), doses=np.asarray(cfg['eval_doses']),
        predictions=victim_plain, trigger_predictions=victim_doses,
        reference_predictions=clean_plain, reference_trigger_predictions=clean_doses,
        targets=np.stack([target_pose(truth, d) for d in cfg['eval_doses']], axis=1))
    result = dict(status='EXPLORATORY_PIW_NATIVE_BASELINE', profile=PROFILE,
        method=method, seed=42, plan_sha256=plan['sha256'],
        clean_reference_sha256=plan['reference_sha256'],
        checkpoint_sha256=sha(folder / 'checkpoint.pt'),
        predictions_sha256=sha(folder / 'predictions.npz'),
        poison=trained['poison'], metrics=measured,
        limitations=['Digital normalized-amplitude group adaptation; phase unchanged.',
                     'Not an unchanged image-classification or over-the-air source attack.',
                     'Native attack perturbation budgets differ.',
                     'Single seed; LR and target were development-selected after PiW TEST was viewed.'])
    v3.base.atomic_json(folder / 'evaluation.json', result)
    print(f'[PiW attack] EVAL_COMPLETE {method}', flush=True)


def verify_evaluation(plan, method):
    folder = cell(plan, method)
    target = folder / 'evaluation.json'
    if not target.exists():
        return None
    saved = json.loads(target.read_text(encoding='utf-8'))
    if (saved.get('profile') != PROFILE or saved.get('method') != method
            or saved.get('plan_sha256') != plan['sha256']
            or saved.get('clean_reference_sha256') != plan['reference_sha256']
            or saved.get('checkpoint_sha256') != sha(folder / 'checkpoint.pt')
            or saved.get('predictions_sha256') != sha(folder / 'predictions.npz')):
        raise ValueError(f'{method}: saved evaluation binding mismatch')
    trained = checkpoint(plan, method, final=True)
    if saved.get('poison') != trained['poison']:
        raise ValueError(f'{method}: poison record changed since evaluation')
    import numpy as np
    from piw_v2_adapter import measure_predictions, target_pose
    with np.load(folder / 'predictions.npz', allow_pickle=False) as p:
        if set(p.files) != {'truth', 'names', 'doses', 'predictions',
                            'trigger_predictions', 'reference_predictions',
                            'reference_trigger_predictions', 'targets'}:
            raise ValueError('Unexpected PiW baseline prediction schema')
        truth = p['truth']
        historical = reference(plan['cfg']['data_home'], plan['reference'])[1]
        if hashlib.sha256(truth.tobytes()).hexdigest() != historical['test_truth_sha256']:
            raise ValueError('PiW TEST truth changed in predictions')
        if p['names'].tolist() != historical['inputs']['splits']['test_data']['ordered_names']:
            raise ValueError('PiW TEST identities changed in predictions')
        if not np.array_equal(p['doses'], plan['cfg']['eval_doses']):
            raise ValueError('PiW evaluation dose grid changed')
        if not np.array_equal(p['targets'], np.stack([
                target_pose(truth, d) for d in plan['cfg']['eval_doses']], axis=1)):
            raise ValueError('PiW target geometry changed in predictions')
        replay = measure_predictions(truth, p['predictions'], p['trigger_predictions'],
                                     p['reference_predictions'], p['reference_trigger_predictions'])
    v3.replay_equal(saved['metrics'], replay)
    return saved


def export(plan):
    import csv
    root = Path(plan['cfg']['outdir'])
    _, _, original = reference(plan['cfg']['data_home'], plan['reference'])
    clean, a1 = original['clean_metrics'], original['a1_metrics']['mainmetrics']
    rows = [dict(method='Clean reference', seed=42, t_mpjpe_mm='', target_pck_pct='',
                 clean_mpjpe_mm=clean['clean_mpjpe_mm'],
                 clean_pampjpe_mm=clean['clean_pampjpe_mm'], mean_dce_mm=''),
            dict(method='A1 v3', seed=42, t_mpjpe_mm=a1['t1_mpjpe_mm'],
                 target_pck_pct=a1['target_pck_pct'], clean_mpjpe_mm=a1['clean_mpjpe_mm'],
                 clean_pampjpe_mm=a1['clean_pampjpe_mm'], mean_dce_mm=a1['mean_dce_mm'])]
    for method in METHODS:
        result = verify_evaluation(plan, method)
        if result is None:
            raise ValueError(f'{method}: evaluation incomplete')
        metrics = result['metrics']['mainmetrics']
        rows.append(dict(method=method, seed=42, t_mpjpe_mm=metrics['t1_mpjpe_mm'],
                         target_pck_pct=metrics['target_pck_pct'],
                         clean_mpjpe_mm=metrics['clean_mpjpe_mm'],
                         clean_pampjpe_mm=metrics['clean_pampjpe_mm'],
                         mean_dce_mm=metrics['mean_dce_mm']))
    fields = list(rows[0]); csv_path = root / 'attack_comparison.csv'
    with (root / 'attack_comparison.csv.saving').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    v3.base.atomic_replace(root / 'attack_comparison.csv.saving', csv_path)
    def show(value):
        return '--' if value == '' else f'{float(value):.2f}'
    lines = ['# EXPLORATORY_PIW3D_ATTACK_COMPARISON_ONE_SEED', '',
        'HPE-Li / PiW3D one-person 14-joint 3D; seed42; frozen v3 Clean/A1 reference.',
        'Five source-derived digital CSI baselines; BadNets excluded by user request.',
        'All victims: scratch 200 epochs/final199, AdamW lr.001, B32, ordinary MPJPE.',
        'Poison identities/doses and right-leg body-axis target match v3; source-specific covers are disclosed.',
        'Target-PCK@.2 is joint accuracy, not pose-level ASR. Native perturbation budgets differ.',
        'PiW TEST was viewed when the v3 recipe was chosen: development study, not blind external validation.',
        'One seed; no across-seed uncertainty. Digital trigger, not over-the-air evidence.', '',
        '| Method | T-MPJPE@1 mm | Target-PCK % | Clean MPJPE mm | Clean PA mm | Mean DCE mm |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| ' + str(row['method']) + ' | ' + ' | '.join(
            show(row[key]) for key in fields[2:]) + ' |')
    (root / 'attack_comparison.md.saving').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    v3.base.atomic_replace(root / 'attack_comparison.md.saving', root / 'attack_comparison.md')
    print(f'COMPLETE_EXPLORATORY: {root / "attack_comparison.md"}', flush=True)


def launch(plan):
    receipt(plan, 'preflight'); receipt(plan, 'gpu_smoke')
    root = Path(plan['cfg']['outdir'])
    if root.exists() and not (root / MANIFEST).exists() and any(
            path.name != 'run.lock' for path in root.iterdir()):
        raise ValueError('Nonempty PiW attack output without a plan; choose a NEW folder')
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / MANIFEST
    if manifest.exists():
        verify_plan(json.loads(manifest.read_text(encoding='utf-8')))
    else:
        v3.base.atomic_json(manifest, plan)
    ref, ds = datasets(plan)
    current, _ = v3.base.audit_inputs(ref['cfg'])
    if current != ref['inputs']:
        raise ValueError('PiW dataset changed before Launch')
    for start in range(0, len(METHODS), plan['cfg']['jobs_per_gpu']):
        methods = METHODS[start:start + plan['cfg']['jobs_per_gpu']]
        children, streams = [], []
        try:
            for method in methods:
                if verify_evaluation(plan, method) is not None:
                    print(f'[PiW attack] VERIFIED_COMPLETE {method}', flush=True)
                    continue
                folder = cell(plan, method); folder.mkdir(parents=True, exist_ok=True)
                stream = (folder / 'console.log').open('ab'); streams.append(stream)
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--_manifest', str(manifest), '--_train', method], cwd=str(HERE),
                    stdout=stream, stderr=subprocess.STDOUT)
                children.append(child)
                print(f'[PiW attack] TRAIN {method}; PID={child.pid}; log={folder / "console.log"}', flush=True)
            codes = [p.wait() for p in children]
            if any(codes):
                raise RuntimeError(f'PiW attack worker failed: {codes}; inspect logs, no automatic retry')
        finally:
            for stream in streams:
                stream.close()
    current, _ = v3.base.audit_inputs(ref['cfg'])
    if current != ref['inputs']:
        raise ValueError('PiW dataset changed before final TEST')
    for method in METHODS:
        if verify_evaluation(plan, method) is not None:
            continue
        with (cell(plan, method) / 'evaluation.log').open('ab') as stream:
            code = subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()),
                '--_manifest', str(manifest), '--_evaluate', method], cwd=str(HERE),
                stdout=stream, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError(f'{method}: final TEST evaluation failed; inspect evaluation.log')
    export(plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--jobs-per-gpu', type=int, choices=(1, 2), default=2)
    modes = parser.add_mutually_exclusive_group()
    for mode in ('dry-run', 'preflight', 'gpu-smoke', 'launch', 'status'):
        modes.add_argument('--' + mode, action='store_true')
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--_train', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--_evaluate', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--_smoke-plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--_slot', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._smoke_plan is not None:
        if args._slot is None or args._manifest or args._train or args._evaluate:
            parser.error('Invalid smoke worker invocation')
        v3.smoke_worker(args._smoke_plan, args._slot)
        return 0
    if args._manifest is not None:
        if bool(args._train) == bool(args._evaluate):
            parser.error('Select exactly one private worker action')
        plan = verify_plan(json.loads(args._manifest.read_text(encoding='utf-8')))
        if args._train:
            from run_mmfi_tables import _cell_lock
            with _cell_lock(cell(plan, args._train)):
                train(plan, args._train)
        else:
            evaluate(plan, args._evaluate)
        return 0
    if args._train or args._evaluate or args._slot is not None:
        parser.error('Private worker needs its saved plan')
    home = args.data_home or Path.home() / 'backdooranalog'
    output = args.outdir or home / 'runs/piw_hpeli_native_attacks_no_badnets_s42_v1'
    plan = build_plan(home, output, args.reference_dir, args.jobs_per_gpu)
    if args.status:
        present = {method: bool(verify_evaluation(plan, method)) for method in METHODS}
        print(json.dumps(present, indent=2))
        if all(present.values()): export(plan)
        return 0
    if args.preflight: preflight(plan)
    elif args.gpu_smoke: gpu_smoke(plan)
    elif args.launch:
        from run_mmfi_tables import _cell_lock
        with _cell_lock(output):
            launch(plan)
    else:
        print(json.dumps(dict(profile=PROFILE, methods=list(METHODS),
            excluded=['badnets'], reference=plan['reference'],
            jobs_per_gpu=plan['cfg']['jobs_per_gpu'], plan_sha256=plan['sha256'],
            note='DRY_RUN_ONLY; no data hash/CUDA/training'), indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'[PiW attack suite] ERROR: {error}', file=sys.stderr, flush=True)
        raise
