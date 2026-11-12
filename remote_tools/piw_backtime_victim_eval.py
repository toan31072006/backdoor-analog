"""Inference-only PiW BackTime metrics, without a separately trained Clean.

Run OUTSIDE the original frozen checkout. Its source/runtime/input/final199
checks remain authoritative. Original plans, checkpoints, trigger and full
reference evaluations are read-only; only their existing runner.lock is used.
New outputs are explicit victim-only results, not matched-Clean controls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.dont_write_bytecode = True

PROFILE = 'piw_backtime_victim_only_eval_s42_v1'
NATIVE_PROFILE = 'piw_backtime_hpeli_shared_three_backbones_train_first_s42_v4'
BACKBONES = ('hpeli', 'wiflow', 'metafiplusplus')
GRID = (0., .2, .4, .6, .8, 1.)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def measure_victim(truth, predictions):
    """Exactly the victim terms of piw_v2_adapter.measure_predictions.

    Dose0 is this SAME victim without a trigger, not a Clean-trained model.
    No reference predictions are fabricated or substituted.
    """
    import numpy as np
    from piw_a1_adapter import _poses, TARGET_JOINTS
    from piw_v2_adapter import target_pose, target_contract
    from eval.metrics import pa_mpjpe

    truth = _poses(truth, 'truth', (14, 3))
    predictions = _poses(predictions, 'trigger_predictions', (6, 14, 3))
    if len(truth) != len(predictions):
        raise ValueError('Prediction identities/count/order must match')
    clean = predictions[:, 0]
    scale = np.linalg.norm(truth[:, 6] - truth[:, 4], axis=-1)
    if np.any(scale <= 1e-8):
        raise ValueError('Degenerate clean GT PCK scale')
    joints = list(TARGET_JOINTS)
    others = [j for j in range(14) if j not in joints]

    def error(a, b, js=None):
        delta = a - b
        if js is not None:
            delta = delta[:, js]
        return float(np.linalg.norm(delta, axis=-1).mean() * 1000)

    def pck(a, b):
        return float(np.mean(np.linalg.norm(a[:, joints] - b[:, joints], axis=-1)
                             <= .2 * scale[:, None]) * 100)

    rows = []
    for i, dose in enumerate(GRID):
        target = target_pose(truth, dose)
        prediction = predictions[:, i]
        desired = target[:, joints] - truth[:, joints]
        response = prediction[:, joints] - clean[:, joints]
        rows.append(dict(dose=dose, t_mpjpe_mm=error(prediction, target, joints),
            target_pck_pct=pck(prediction, target),
            dce_mm=float(np.linalg.norm(response - desired, axis=-1).mean() * 1000),
            still_dce_mm=float(np.linalg.norm(desired, axis=-1).mean() * 1000),
            triggered_gt_mpjpe_mm=error(prediction, truth, joints),
            triggered_gt_all_mpjpe_mm=error(prediction, truth),
            nontarget_drift_mm=error(prediction, clean, others),
            no_trigger_t_mpjpe_mm=error(clean, target, joints),
            no_trigger_target_pck_pct=pck(clean, target)))
    return dict(mainmetrics=dict(t1_mpjpe_mm=rows[-1]['t_mpjpe_mm'],
        target_pck_pct=rows[-1]['target_pck_pct'], clean_mpjpe_mm=error(clean, truth),
        clean_pampjpe_mm=float(pa_mpjpe(clean, truth).mean() * 1000),
        mean_dce_mm=float(np.mean([r['dce_mm'] for r in rows[1:]]))), dose_rows=rows,
        controls=dict(target=target_contract(), pck_unit='target joints %, not pose ASR',
            dce_summary='mean of five positive doses; no alignment',
            clean_input='dose0 of the BackTime victim, not a Clean-trained model',
            matched_clean_model='NOT_EVALUATED', clean_model_plus_trigger='NOT_EVALUATED',
            clean_utility_difference_to_reference='NOT_EVALUATED'))


def load_worker(source_repo):
    source = Path(source_repo).resolve() / 'ATKBackd'
    path = source / 'run_piw_backtime_three_backbones.py'
    if not path.is_file():
        raise ValueError('Original frozen PiW v4 checkout is required')
    # Never mix frozen model/helper modules with this evaluator's newer checkout.
    for name, module in tuple(sys.modules.items()):
        if name.startswith(('piw_', 'run_piw', 'models', 'data_utils', 'attack', 'eval')):
            filename = getattr(module, '__file__', None)
            if filename and not Path(filename).resolve().is_relative_to(source):
                raise ValueError('Start a fresh Python process; a foreign scientific module is already loaded')
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('_frozen_piw_backtime_worker', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    if worker.PROFILE != NATIVE_PROFILE or tuple(worker.BACKBONES) != BACKBONES:
        raise ValueError('This evaluator supports the original train-first PiW v4 only')
    return worker


def validate_output(outdir, source_repo, run, dataset):
    outdir = Path(outdir).absolute()
    if outdir.exists():
        raise ValueError('Output already exists; preserve it, no automatic overwrite/resume')
    for path in (outdir, *outdir.parents):
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise ValueError('Linked output is not allowed')
    outdir = outdir.resolve()
    for path in (source_repo, run, dataset):
        path = Path(path).resolve()
        if outdir == path or outdir.is_relative_to(path) or path.is_relative_to(outdir):
            raise ValueError('Use a separate new evaluation directory, outside source/run/dataset')
    return outdir


def victim_arrays(worker, plan, backbone, data, truth, names):
    import numpy as np
    from piw_v2_adapter import target_pose
    from attack.backtime_mmfi import _sha

    folder = worker.cell(plan, backbone, 'backtime')
    checkpoint_path = folder / 'checkpoint.pt'
    before = sha(checkpoint_path)
    cached_report = folder / 'evaluation.json'
    if cached_report.is_file():
        report = read(cached_report)
        arrays = folder / 'predictions.npz'
        expected = dict(profile=NATIVE_PROFILE, status='COMPLETE',
            plan_sha256=plan['plan_sha256'], backbone=backbone)
        if (any(report.get(k) != v for k, v in expected.items())
                or report['artifacts'].get('checkpoint.pt') != before
                or report['artifacts'].get('predictions.npz') != sha(arrays)
                or report['artifacts'].get('trigger.pt') != sha(worker.attacker_path(plan))):
            raise ValueError('Existing victim evaluation/cache binding changed')
        with np.load(arrays, allow_pickle=False) as cache:
            if (cache['names'].tolist() != names or not np.array_equal(cache['truth'], truth)
                    or not np.array_equal(cache['doses'], GRID)
                    or not np.array_equal(cache['targets'],
                        np.stack([target_pose(truth, d) for d in GRID], axis=1))):
                raise ValueError('Cached victim identities/truth/doses/targets changed')
            predictions = cache['predictions'].copy()
        metrics = measure_victim(truth, predictions)
        if metrics['mainmetrics'] != report['metrics']['mainmetrics']:
            raise ValueError('Cached victim main metrics do not replay exactly')
        for row, original in zip(metrics['dose_rows'], report['metrics']['dose_rows'], strict=True):
            if any(original.get(k) != v for k, v in row.items()):
                raise ValueError('Cached victim dose metrics do not replay exactly')
        mode = 'REPLAY_EXISTING_VICTIM_ARRAYS; no Clean model loaded or reverified'
    else:
        import torch
        saved = worker.checkpoint(plan, backbone, 'backtime', final=True)
        model = worker.build_model(backbone, plan['cfg']['device'])
        model.load_state_dict(saved['model'], strict=True)
        del saved
        trigger, _ = worker.load_attacker(plan)
        model_before, trigger_before = _sha(model.state_dict()), _sha(trigger.state_dict())
        print(f'EVALUATE_ONLY {backbone}: final199, 6 doses, {len(names)} TEST frames', flush=True)
        predictions = worker.infer_grid(model, data, trigger, plan['cfg']['device'])
        if _sha(model.state_dict()) != model_before or _sha(trigger.state_dict()) != trigger_before:
            raise ValueError('Inference mutated victim or frozen trigger state')
        del model, trigger
        if plan['cfg']['device'].startswith('cuda:'):
            torch.cuda.empty_cache()
        metrics, mode = measure_victim(truth, predictions), 'FINAL199_INFERENCE_ONLY'
    if sha(checkpoint_path) != before:
        raise ValueError('Victim checkpoint changed during evaluation')
    return predictions, metrics, mode, before


def evaluate(worker, plan, outdir):
    import numpy as np
    from piw_v2_adapter import target_pose

    worker.require_trained(plan)  # Native final199/checksum/attacker/execution checks, all 3.
    data = worker.datasets_for(plan)['validation']
    names = [row['name'] for row in data.items]
    if names != plan['inputs']['splits']['test_data']['ordered_names']:
        raise ValueError('TEST identities/order differ from the training plan')
    truth = np.stack([data.load_pose(row['kpt'])[0] for row in data.items])
    targets = np.stack([target_pose(truth, dose) for dose in GRID], axis=1)
    trigger_hash, plan_hash = sha(worker.attacker_path(plan)), sha(worker.plan_path(plan['cfg']))
    checkpoint_hashes = {b: sha(worker.cell(plan, b, 'backtime') / 'checkpoint.pt') for b in BACKBONES}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    rows = []
    for backbone in BACKBONES:
        predictions, metrics, mode, checkpoint_hash = victim_arrays(
            worker, plan, backbone, data, truth, names)
        folder = outdir / backbone
        folder.mkdir()
        arrays = folder / 'predictions.npz'
        with arrays.open('xb') as stream:
            np.savez_compressed(stream, truth=truth, names=np.asarray(names),
                doses=np.asarray(GRID), predictions=predictions, targets=targets)
        with np.load(arrays, allow_pickle=False) as cache:
            if measure_victim(cache['truth'], cache['predictions']) != metrics:
                raise ValueError('New prediction archive does not exactly replay victim metrics')
        report = dict(profile=PROFILE, status='VICTIM_METRICS_COMPLETE_NO_MATCHED_CLEAN',
            backbone=backbone, training_plan_sha256=plan['plan_sha256'], final_epoch=199,
            evaluation_mode=mode, metrics=metrics, model_training_started=False,
            clean_model_loaded=False, checkpoint_saved=False,
            evaluator_sha256=sha(__file__), source_repo=str(worker.HERE.parent),
            source_run=plan['cfg']['outdir'], artifacts=dict(
                original_checkpoint_sha256=checkpoint_hash, original_trigger_sha256=trigger_hash,
                predictions_sha256=sha(arrays)))
        write_new(folder / 'evaluation.json', report)
        rows.append(dict(backbone=backbone, **metrics['mainmetrics']))
        print(json.dumps(rows[-1]), flush=True)
    if (sha(worker.attacker_path(plan)) != trigger_hash
            or sha(worker.plan_path(plan['cfg'])) != plan_hash
            or any(sha(worker.cell(plan, b, 'backtime') / 'checkpoint.pt') != digest
                   for b, digest in checkpoint_hashes.items())):
        raise ValueError('Original training artifacts changed; no completion marker')
    worker.load_plan(plan['cfg'], full_hash=True)
    with (outdir / 'victim_results.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_new(outdir / 'summary.json', dict(profile=PROFILE,
        status='THREE_VICTIM_METRICS_COMPLETE_NO_MATCHED_CLEAN',
        training_plan_sha256=plan['plan_sha256'], results=rows,
        clean_reference_controls='NOT_EVALUATED; no matched-Clean utility or trigger-control claim',
        model_training_started=False, checkpoint_saved=False))
    print(f'VICTIM_ONLY_EVALUATION_COMPLETE: {outdir / "victim_results.csv"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', required=True, help='ORIGINAL frozen training checkout; never pull into it')
    parser.add_argument('--run', required=True, help='Original completed train-first v4 run')
    parser.add_argument('--outdir', required=True, help='Separate NEW victim-only evaluation output')
    args = parser.parse_args(argv)
    run, source_repo = Path(args.run).resolve(), Path(args.source_repo).resolve()
    original = read(run / 'plan.json')
    if original.get('profile') != NATIVE_PROFILE or Path(original['cfg']['outdir']).resolve() != run:
        raise ValueError('Wrong run/profile; no plan rebinding or migration')
    outdir = validate_output(args.outdir, source_repo, run, original['cfg']['dataset_root'])
    worker = load_worker(source_repo)
    print('Victim-only evaluation: ZERO train/refit/Clean loads; preserve original artifacts.', flush=True)
    print('Checking original source/runtime/full inputs and three final199 checkpoints...', flush=True)
    with worker.lock(original['cfg']):
        plan = worker.load_plan(original['cfg'], full_hash=True)
        evaluate(worker, plan, outdir)


if __name__ == '__main__':
    main()
