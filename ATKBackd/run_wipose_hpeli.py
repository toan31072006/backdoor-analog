"""Independent WiPose 2D HPE-Li Clean -> A1 seed42 pilot.

No historical run/model/source is changed. The proposed DT-Pose two-stage
model is NOT run. Fifty fixed epochs, author scratch HPE-Li SGD scalars,
full supplied TEST, and explicit digital 2D/amplitude A1 adaptations.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import run_piw_hpeli_clean as util

HERE = Path(__file__).resolve().parent
PROFILE = 'wipose_2d_hpeli_clean_a1_seed42_v1'
MANIFEST = 'wipose_hpeli.resolved.json'
STATUS = 'ONE_SEED_WIPOSE_2D_ADAPTATION_NOT_FULL_PAPER_TABLE'
EPOCHS = 50
STAGES = ('clean', 'a1')
EXPECTED_LIST_COUNTS = {'training': 132847, 'validation': 33753}
SOURCE_NAMES = (
    'run_wipose_hpeli.py', 'wipose_data.py', 'prepare_wipose.py',
    'wipose_a1_adapter.py', 'models/wipose_hpeli.py', 'models/wipose_sources.json',
    'models/sk_network.py', 'run_piw_hpeli_clean.py', 'piw_a1_adapter.py',
    'run_mmfi_tables.py', 'attack/trigger.py', 'attack/method_drafts.py',
    'attack/tsba.py', 'attack/payload.py',
)


def source_hashes():
    return {name: util.file_sha(HERE / name) for name in SOURCE_NAMES}


def configuration(data_home, outdir=None, device='cuda:0', dataset_root=None, action_npy=None):
    if device != 'cpu' and not re.fullmatch(r'cuda:[0-9]+', device):
        raise ValueError('Use cpu or explicit cuda:N; no device fallback')
    home = Path(data_home).resolve()
    output = Path(outdir).resolve() if outdir else home / 'runs/wipose_hpeli_a1_s42_v1'
    dataset = Path(dataset_root).resolve() if dataset_root else home / 'datasets/WiPose_processed_v1'
    action = Path(action_npy).resolve() if action_npy else home / 'actions/data_bend.npy'
    if output == dataset or output in dataset.parents or dataset in output.parents:
        raise ValueError('Dataset and output must be separate directories')
    return dict(profile=PROFILE, data_home=str(home), dataset_root=str(dataset),
        outdir=str(output), action_npy=str(action), device=device,
        dataset='wipose', model='hpeli_dtpose_wipose', seed=42,
        input_shape=[3, 90, 5], num_keypoints=18, num_coor=2, num_person=1,
        epochs=EPOCHS, final_epoch=EPOCHS-1, optimizer='sgd', lr=.01,
        momentum=.9, weight_decay=0., batch_size=32, num_workers=0,
        drop_last=True, shuffle=True, victim_loss='ordinary MPJPE',
        expected_list_counts=dict(EXPECTED_LIST_COUNTS),
        pretrained=False, amp=False, lr_scheduler=False,
        selection='fixed final epoch; TEST never selects checkpoint',
        model_units='original image coordinates times .001; NOT metres',
        report_units='original image coordinates (normally pixels); NOT mm',
        attack_adaptation='2D right-leg rotation; native A1 on normalized amplitude groups',
        author_reproduction=False)


def canonical(cfg):
    return configuration(cfg['data_home'], cfg['outdir'], cfg['device'],
                         cfg['dataset_root'], cfg['action_npy'])


def static_plan(cfg):
    if cfg != canonical(cfg):
        raise ValueError('Changed frozen WiPose scientific configuration')
    return dict(schema=1, profile=PROFILE, cfg=cfg, sources=source_hashes(),
                runtime=util.runtime_versions())


def datasets_for(cfg):
    from wipose_data import WiPoseDataset
    return {key: WiPoseDataset(cfg['dataset_root'], key)
            for key in ('training', 'validation')}


def prepare(cfg):
    from wipose_data import dataset_audit
    from wipose_a1_adapter import build_trigger, PoisonedWiPoseDataset
    inputs = dataset_audit(cfg['dataset_root'])
    if {key: inputs['splits'][key]['official_count'] for key in EXPECTED_LIST_COUNTS} != cfg['expected_list_counts']:
        raise ValueError('WiPose official full-list counts differ; audit archive/lists, never silently train a subset')
    datasets = datasets_for(cfg)
    if len(datasets['training']) < cfg['batch_size'] or not len(datasets['validation']):
        raise ValueError('Full TRAIN must contain at least one batch and TEST must be nonempty')
    trigger = build_trigger(cfg['action_npy'])
    poisoned = PoisonedWiPoseDataset(datasets['training'], trigger)
    context = dict(static_plan(cfg), inputs=inputs, operator=trigger.state_record(),
                   poison=poisoned.manifest(),
                   ordered_test_names=[item['name'] for item in datasets['validation'].items])
    preparation = Path(cfg['dataset_root'])/'preparation.json'
    context['preparation_sha256'] = util.file_sha(preparation) if preparation.is_file() else None
    context['plan_sha256'] = util.object_sha(context)
    return context, datasets, trigger, poisoned


def same_context(saved, actual):
    if saved != actual:
        raise ValueError('WiPose data/action/poison/source/runtime changed; preserve artifacts, no resume')


def load_manifest(path):
    path = Path(path).resolve()
    saved = json.loads(path.read_text(encoding='utf-8'))
    payload = {k: v for k, v in saved.items() if k != 'plan_sha256'}
    if (saved.get('schema') != 1 or saved.get('profile') != PROFILE
            or saved.get('plan_sha256') != util.object_sha(payload)
            or path != Path(saved['cfg']['outdir']) / MANIFEST):
        raise ValueError('Invalid WiPose manifest/path/hash')
    expected = static_plan(saved['cfg'])
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError('Frozen WiPose source/runtime/config mismatch; do not edit manifest')
    return saved


def cell_dir(manifest, stage):
    if stage not in STAGES:
        raise ValueError('Unknown WiPose stage')
    return Path(manifest['cfg']['outdir']) / 'hpeli/seed42' / stage


def build_model():
    from models.wipose_hpeli import build_wipose_hpeli
    return build_wipose_hpeli()


def preflight_smoke(manifest, datasets, trigger, poisoned):
    import numpy as np
    import torch
    from wipose_a1_adapter import GRID, target_pose, measure_predictions
    first = datasets['training'][0]
    value = np.asarray(first['csi'])
    if not np.array_equal(trigger.inject(value, 0), value):
        raise ValueError('A1 dose-zero input is not exact identity')
    model = build_model().cpu().train()
    x = torch.from_numpy(np.stack([value, trigger.inject(value, 1)])).float()
    y = torch.from_numpy(np.stack([first['pose'], target_pose(first['pose'], 1)])).float()
    prediction, _ = model(x)
    if prediction.shape != (2, 1, 18, 2):
        raise ValueError('WiPose HPE-Li 2D output mismatch')
    loss = torch.linalg.vector_norm(prediction-y, dim=-1).mean()
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                     for p in model.parameters()):
        raise ValueError('Nonfinite WiPose CPU forward/backward')
    truth = np.stack([datasets['validation'][0]['pose'][0]] * 2).astype(float)
    targets = np.stack([target_pose(truth, dose) for dose in GRID], axis=1)
    perfect = measure_predictions(truth, truth, targets, truth,
                                  np.repeat(truth[:, None], len(GRID), axis=1))
    if perfect['mainmetrics']['mean_dce_px'] > 1e-8:
        raise ValueError('Perfect 2D target trajectory failed DCE smoke')
    return dict(cpu_only=True, output_shape=list(prediction.shape),
                train=len(poisoned), test=len(datasets['validation']),
                poison=len(poisoned.manifest()['samples']), finite=True)


def checkpoint_header(path, manifest, stage, execution=None, final=False):
    import torch
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if (blob.get('schema') != 1 or blob.get('profile') != PROFILE
            or blob.get('stage') != stage or blob.get('plan_sha256') != manifest['plan_sha256']
            or blob.get('cfg') != manifest['cfg'] or type(blob.get('epoch')) is not int
            or not 0 <= blob['epoch'] < manifest['cfg']['epochs']
            or not all(key in blob for key in ('model', 'optimizer', 'rng', 'execution'))):
        raise ValueError('Checkpoint stage/config/epoch/RNG binding mismatch')
    if final and blob['epoch'] != manifest['cfg']['final_epoch']:
        raise ValueError('Only fixed final epoch can be evaluated')
    if execution is not None and blob['execution'] != execution:
        raise ValueError('Execution/GPU changed; refusing scientific resume')
    if stage == 'a1':
        clean_path = cell_dir(manifest, 'clean') / 'checkpoint.pt'
        if blob.get('clean_checkpoint_sha256') != util.file_sha(clean_path):
            raise ValueError('A1 checkpoint clean-reference binding mismatch')
    return blob


def _close_tree(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close_tree(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close_tree(x, y) for x, y in zip(a, b))
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-8)
    return type(a) is type(b) and a == b


def clean_metrics(prediction, truth):
    import numpy as np
    from wipose_a1_adapter import pa_mpjpe_2d
    if any(np.asarray(value).dtype.kind not in 'fiu' for value in (prediction, truth)):
        raise ValueError('Clean predictions must be real numeric arrays')
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    if (prediction.shape != truth.shape or prediction.shape[1:] != (18, 2)
            or not len(truth) or not np.isfinite(prediction).all() or not np.isfinite(truth).all()):
        raise ValueError('Invalid full TEST 18x2 predictions')
    return dict(clean_mpjpe_px=float(np.linalg.norm(prediction-truth, axis=-1).mean()*1000),
                clean_pampjpe_px=float(pa_mpjpe_2d(prediction, truth).mean()*1000))


def verify_stage(manifest, stage):
    import numpy as np
    folder = cell_dir(manifest, stage)
    path = folder / 'evaluation.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    artifact_names = {'checkpoint.pt', 'predictions.npz'} | ({'poison.json'} if stage == 'a1' else set())
    if (result.get('schema') != 1 or type(result.get('seed')) is not int or result.get('seed') != 42
            or result.get('units') != manifest['cfg']['report_units']
            or result.get('profile') != PROFILE or result.get('stage') != stage
            or result.get('status') != STATUS or result.get('plan_sha256') != manifest['plan_sha256']
            or result.get('checkpoint_epoch') != manifest['cfg']['final_epoch']
            or set(result.get('artifacts', {})) != artifact_names):
        raise ValueError('WiPose result binding/schema mismatch')
    for name, digest in result['artifacts'].items():
        if util.file_sha(folder / name) != digest:
            raise ValueError('WiPose saved artifact hash mismatch: ' + name)
    checkpoint = checkpoint_header(folder / 'checkpoint.pt', manifest, stage, final=True)
    if result.get('execution') != checkpoint['execution']:
        raise ValueError('Result/checkpoint execution mismatch')
    with np.load(folder / 'predictions.npz', allow_pickle=False) as data:
        if data['names'].tolist() != manifest['ordered_test_names']:
            raise ValueError('Saved TEST order/count changed')
        if (data['names'].dtype.kind not in 'US' or data['names'].ndim != 1
                or any(len(data[key]) != len(data['names']) for key in ('predictions', 'truth'))):
            raise ValueError('Saved TEST names and prediction counts differ')
        if stage == 'clean':
            if set(data.files) != {'predictions', 'truth', 'names'}:
                raise ValueError('Unexpected clean NPZ keys')
            measured = clean_metrics(data['predictions'], data['truth'])
        else:
            expected = {'predictions', 'truth', 'names', 'trigger_predictions',
                        'reference_predictions', 'reference_trigger_predictions', 'targets', 'doses'}
            if set(data.files) != expected:
                raise ValueError('Unexpected A1 NPZ keys')
            reference = verify_stage(manifest, 'clean')
            if reference is None or result.get('clean_result_sha256') != util.object_sha(reference):
                raise ValueError('A1 clean result dependency changed')
            with np.load(cell_dir(manifest, 'clean') / 'predictions.npz', allow_pickle=False) as saved:
                if (not np.array_equal(saved['truth'], data['truth'])
                        or not np.array_equal(saved['predictions'], data['reference_predictions'])
                        or saved['names'].tolist() != data['names'].tolist()):
                    raise ValueError('A1 clean predictions/identities changed')
            from wipose_a1_adapter import GRID, target_pose, measure_predictions
            targets = np.stack([target_pose(data['truth'], d) for d in GRID], axis=1)
            if not np.array_equal(data['doses'], GRID) or not np.array_equal(targets, data['targets']):
                raise ValueError('Saved dose grid/2D targets changed')
            measured = measure_predictions(data['truth'], data['predictions'], data['trigger_predictions'],
                                           data['reference_predictions'], data['reference_trigger_predictions'])
            if json.loads((folder / 'poison.json').read_text(encoding='utf-8')) != manifest['poison']:
                raise ValueError('Saved poison plan changed')
    if not _close_tree(result.get('measurements'), measured):
        raise ValueError('WiPose metrics do not replay saved per-sample predictions')
    return result


def _atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    temporary = path.with_name(path.name+'.saving')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        util.atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def infer(model, dataset, cfg, trigger=None, dose=0.):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    predictions, truth, names = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False,
                                num_workers=0, drop_last=False):
            csi = batch['csi']
            if trigger is not None and dose:
                csi = torch.from_numpy(np.stack([trigger.inject(x, dose) for x in csi.numpy()])).float()
            output, _ = model(csi.to(cfg['device'], dtype=torch.float32))
            if output.shape != batch['pose'].shape or not torch.isfinite(output).all():
                raise ValueError('Invalid/nonfinite WiPose TEST output')
            predictions.append(output[:, 0].cpu().numpy())
            truth.append(batch['pose'][:, 0].numpy())
            names.extend(batch['name'])
    return np.concatenate(predictions), np.concatenate(truth), names


def evaluate_stage(manifest, stage, model, dataset, trigger, execution):
    import numpy as np
    cfg, folder = manifest['cfg'], cell_dir(manifest, stage)
    prediction, truth, names = infer(model, dataset, cfg)
    if names != manifest['ordered_test_names']:
        raise ValueError('Full TEST official ordering changed')
    arrays = dict(predictions=prediction, truth=truth, names=np.asarray(names, dtype=str))
    dependency = None
    if stage == 'clean':
        measured = clean_metrics(prediction, truth)
    else:
        from wipose_a1_adapter import GRID, target_pose, measure_predictions
        dependency = verify_stage(manifest, 'clean')
        if dependency is None:
            raise ValueError('A1 requires authenticated completed Clean')
        with np.load(cell_dir(manifest, 'clean') / 'predictions.npz', allow_pickle=False) as saved:
            reference = saved['predictions'].copy()
            if saved['names'].tolist() != names or not np.array_equal(saved['truth'], truth):
                raise ValueError('Clean/A1 full TEST identities differ')
        clean_model = build_model().to(cfg['device'])
        clean_model.load_state_dict(checkpoint_header(cell_dir(manifest, 'clean') / 'checkpoint.pt',
                                    manifest, 'clean', execution, final=True)['model'], strict=True)
        fresh, ref_truth, ref_names = infer(clean_model, dataset, cfg)
        if (ref_names != names or not np.array_equal(ref_truth, truth)
                or not np.allclose(fresh, reference, rtol=0, atol=1e-7)):
            raise ValueError('Clean checkpoint does not reproduce saved predictions')
        attacked, references = [prediction.copy()], [reference.copy()]
        for dose in GRID[1:]:
            print(f'[WiPose eval] full TEST attack + clean control d={dose:g}', flush=True)
            a, ay, an = infer(model, dataset, cfg, trigger, dose)
            b, by, bn = infer(clean_model, dataset, cfg, trigger, dose)
            if an != names or bn != names or not np.array_equal(ay, truth) or not np.array_equal(by, truth):
                raise ValueError('Six-dose paired TEST identities changed')
            attacked.append(a); references.append(b)
        arrays.update(trigger_predictions=np.stack(attacked, axis=1),
            reference_predictions=reference, reference_trigger_predictions=np.stack(references, axis=1),
            targets=np.stack([target_pose(truth, d) for d in GRID], axis=1), doses=np.asarray(GRID))
        measured = measure_predictions(truth, prediction, arrays['trigger_predictions'],
                                        reference, arrays['reference_trigger_predictions'])
    # Audit content again after all full-TEST passes, before publishing a result.
    # This catches concurrent input/action edits without repeating model inference.
    same_context(manifest, prepare(cfg)[0])
    _atomic_npz(folder / 'predictions.npz', **arrays)
    artifacts = ('checkpoint.pt', 'predictions.npz') + (('poison.json',) if stage == 'a1' else ())
    result = dict(schema=1, profile=PROFILE, stage=stage, status=STATUS,
        seed=42, checkpoint_epoch=cfg['final_epoch'], plan_sha256=manifest['plan_sha256'],
        execution=execution, measurements=measured, units=cfg['report_units'],
        artifacts={name: util.file_sha(folder/name) for name in artifacts})
    if dependency is not None:
        result['clean_result_sha256'] = util.object_sha(dependency)
    util.atomic_json(folder / 'evaluation.json', result)
    verify_stage(manifest, stage)


def train_stage(manifest, stage, datasets, trigger, poisoned):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if verify_stage(manifest, stage) is not None:
        print(f'[WiPose] VERIFIED {stage}; no training/inference', flush=True)
        return
    if stage == 'a1' and verify_stage(manifest, 'clean') is None:
        raise ValueError('Complete Clean is required before A1')
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
        print(f'[WiPose] {stage} resumed epoch{blob["epoch"]}; next={start}', flush=True)
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
    print(f'[WiPose] {stage} scratch TRAIN={len(training)} TEST={len(datasets["validation"])}; '
          f'ordinary MPJPE; SGD lr.01; batch32; final{cfg["final_epoch"]}; '
          f'author drop_last=True consumes {expected} shuffled samples/epoch', flush=True)
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
        # Source and runtime are immutable even while another job runs.
        current = static_plan(cfg)
        if any(manifest[k] != v for k, v in current.items()):
            raise ValueError('Source/runtime changed during training')
        blob = dict(schema=1, profile=PROFILE, stage=stage, plan_sha256=manifest['plan_sha256'],
            cfg=cfg, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
            rng=util._rng_state(cfg['device']), execution=execution, train_loss=total/count)
        if stage == 'a1':
            verify_stage(manifest, 'clean')
            blob['clean_checkpoint_sha256'] = util.file_sha(cell_dir(manifest, 'clean')/'checkpoint.pt')
        util.atomic_checkpoint(checkpoint, blob)
        print(f'epoch {epoch}/{cfg["final_epoch"]}: loss={total/count:.8f} '
              f'lr={cfg["lr"]:.2e} t={time.monotonic()-began:.1f}s', flush=True)
        print(f'[ckpt] saved -> {checkpoint} (epoch{epoch})', flush=True)
    same_context(manifest, prepare(cfg)[0])
    checkpoint_header(checkpoint, manifest, stage, execution, final=True)
    evaluate_stage(manifest, stage, model, datasets['validation'], trigger, execution)
    print(f'[WiPose] COMPLETE {stage}; final epoch and full TEST', flush=True)


def export_report(manifest):
    clean = verify_stage(manifest, 'clean')
    a1 = verify_stage(manifest, 'a1')
    if clean is None:
        raise ValueError('No completed Clean to report')
    root = Path(manifest['cfg']['outdir'])
    columns = ('t1_mpjpe_px', 'target_pck_pct', 'clean_mpjpe_px', 'clean_pampjpe_px', 'mean_dce_px')
    lines = ['# '+STATUS, '', 'HPE-Li / WiPose 18-joint 2D, scratch seed42; not the proposed DT-Pose model.',
        'Errors in original image-coordinate units (normally pixels), never physical mm.',
        'Fixed final49, SGD .01/momentum.9; batch32; no TEST checkpoint selection.',
        'Native A1 adapted to normalized processed amplitude; +40deg 2D leg rotation, rho.1.',
        'Six doses, same-victim no-trigger and clean-model+trigger controls; one seed, no SD/CI.', '',
        '| Method | Seed | T-MPJPE (image units) | Target-PCK (%) | Clean MPJPE | Clean PA-MPJPE | DCE |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        f'| Clean reference | 42 | -- | -- | {clean["measurements"]["clean_mpjpe_px"]:.4f} | '
        f'{clean["measurements"]["clean_pampjpe_px"]:.4f} | -- |']
    if a1 is not None:
        measured = a1['measurements']
        values = [f'{measured["mainmetrics"][k]:.4f}' for k in columns]
        lines.append('| A1 (WiPose-adapted) | 42 | '+' | '.join(values)+' |')
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(measured['dose_rows'][0]))
        writer.writeheader(); writer.writerows(measured['dose_rows'])
        _atomic_text(root/'a1_six_doses.csv', stream.getvalue())
        util.atomic_json(root/'a1_diagnostics.json', measured)
        util.atomic_json(root/'wipose_pilot.complete.json', dict(status=STATUS,
            plan_sha256=manifest['plan_sha256'], clean=clean, a1=a1))
    else:
        lines += ['', 'CLEAN ONLY: A1 missing; pilot incomplete.']
    _atomic_text(root/'wipose_pilot.md', '\n'.join(lines)+'\n')


def _atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name+'.saving')
    try:
        temporary.write_text(text, encoding='utf-8')
        util.atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--action-npy', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--clean-only', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    for name in ('dry-run', 'preflight', 'status', 'check-only', 'export-only'):
        modes.add_argument('--'+name, action='store_true')
    parser.add_argument('--_worker', choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if bool(args._worker) != bool(args._manifest):
        parser.error('Private worker requires manifest and stage')
    if args._worker and any((args.dry_run, args.preflight, args.status, args.check_only,
                            args.export_only, args.clean_only)):
        parser.error('Private worker cannot use public runtime modes')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.outdir is None:
        if args.data_home is None:
            raise ValueError('--data-home or --outdir required')
        args.outdir = args.data_home/'runs/wipose_hpeli_a1_s42_v1'
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
            raise FileNotFoundError('No existing WiPose manifest: '+str(manifest_path))
        if args.data_home is None:
            raise ValueError('--data-home required for new run')
        cfg = configuration(args.data_home, args.outdir, args.device, args.dataset_root, args.action_npy)
        manifest = None
    if args.dry_run:
        print('[WiPose] DRY RUN: scratch Clean + A1, seed42; 50 epochs/final49; '
              'SGD .01; 18x2; one slot. NO data/model/CUDA/output writes/training.')
        print(json.dumps(static_plan(cfg), indent=2))
        return 0
    if args.status:
        for stage in STAGES:
            print(f'[WiPose] {stage}: '+('VERIFIED_SAVED_ARTIFACTS_COMPLETE' if verify_stage(manifest, stage)
                                       else 'INCOMPLETE'))
        return 0
    context, datasets, trigger, poisoned = prepare(cfg)
    if manifest is not None:
        same_context(manifest, context)
    else:
        manifest = context
    if args.preflight:
        print(json.dumps(preflight_smoke(manifest, datasets, trigger, poisoned), indent=2))
        print('[WiPose] READ_ONLY_CPU_PREFLIGHT; no GPU/training/output writes')
        return 0
    selected = ('clean',) if args.clean_only else STAGES
    if args.check_only:
        missing = [s for s in selected if verify_stage(manifest, s) is None]
        if missing:
            print('[WiPose] INCOMPLETE: '+', '.join(missing))
            return 1
        print('[WiPose] VERIFIED_COMPLETE_INPUTS_POISON_NPZ; no training/inference/writes')
        return 0
    from run_mmfi_tables import _cell_lock
    if args._worker:
        if args._manifest.resolve() != manifest_path:
            raise ValueError('Private worker manifest differs from exact output')
        with _cell_lock(cell_dir(manifest, args._worker)):
            train_stage(manifest, args._worker, datasets, trigger, poisoned)
        return 0
    root = Path(cfg['outdir'])
    if not manifest_path.exists() and root.exists() and any(p.name != 'run.lock' for p in root.iterdir()):
        raise ValueError('Nonempty output without WiPose manifest; choose NEW output')
    with _cell_lock(root):
        if manifest_path.exists():
            same_context(load_manifest(manifest_path), manifest)
        else:
            util.atomic_json(manifest_path, manifest)
        if args.export_only:
            export_report(manifest)
            return 0
        for stage in selected:
            if verify_stage(manifest, stage) is not None:
                print(f'[WiPose] VERIFIED {stage}; reused without repair/evaluation', flush=True)
                continue
            folder = cell_dir(manifest, stage)
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--data-home', cfg['data_home'],
                '--outdir', cfg['outdir'], '--device', cfg['device'], '--_worker', stage,
                '--_manifest', str(manifest_path)]
            with (folder/'console.log').open('a', encoding='utf-8') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
                                           env=dict(os.environ, PYTHONUTF8='1'))
                print(f'[WiPose] {stage} -> {cfg["device"]}; PID={process.pid}; log={folder/"console.log"}', flush=True)
                code = process.wait()
            if code:
                raise RuntimeError(f'WiPose {stage} failed exit{code}; inspect log; no automatic retry')
            if verify_stage(manifest, stage) is None:
                raise RuntimeError('Worker returned without authenticated completion')
        export_report(manifest)
        print(f'[WiPose] {"CLEAN_ONLY" if args.clean_only else "PILOT_COMPLETE"}: {root/"wipose_pilot.md"}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('[WiPose] ERROR: '+str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
