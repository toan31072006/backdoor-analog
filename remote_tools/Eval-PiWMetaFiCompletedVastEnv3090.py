"""Read-only checkpoint replay of six completed historical MetaFi++ victims.

Run one method per Python process. CPU is the explicit default so the fresh
Clean diagnostic may keep the GPU. Training provenance is authenticated and
preserved; the new evaluation runtime is recorded separately, never spoofed.
This does not retrain old victims or turn them into library-matched trainings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

PROFILE = 'piw_metafi_completed_vastenv_eval_s42_3090_v1'
METHODS = ('clean', 'a1', 'ftrojan', 'bpp_source', 'fiba', 'wanet_source')
CONTROLLER_SHA = '1bb43a9e73a16c74e5c57972bebc2bce7b948cc52c8c19c5599df1b8ef7b80c2'
METRICS_SHA = 'ed76459b09051ccd07bc6f7a9a218535a1d268f909883377b76e474f8bfe9c6e'
WEIGHTS_SHA = 'b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f'
EXPECTED_PACKAGES = {'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128',
                     'numpy': '2.5.4', 'scipy': '1.17.1'}
DOSES = (0., .2, .4, .6, .8, 1.)
THREADING = {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
             'MKL_THREADING_LAYER': 'SEQUENTIAL'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reject_links(path):
    """Reject lexical symlink/junction ancestors before resolving any path."""
    path = Path(path).absolute()
    for candidate in (path, *path.parents):
        try:
            details = candidate.lstat()
        except FileNotFoundError:
            continue
        if candidate.is_symlink() or getattr(details, 'st_file_attributes', 0) & 0x400:
            raise ValueError(f'Refusing linked/reparse path or ancestor: {candidate}')


def load_controller(repo):
    path = Path(repo) / 'remote_tools/piw_metafi_remaining_3090/controller.py'
    if sha(path) != CONTROLLER_SHA:
        raise ValueError('Historical controller changed; do not edit prepared sources/gates')
    spec = importlib.util.spec_from_file_location('_completed_vastenv_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.code_binding()
    return module


def metrics_module(repo):
    path = Path(repo) / 'remote_tools/Eval-PiWMetaFiFTrojan3090.py'
    if sha(path) != METRICS_SHA:
        raise ValueError('Original victim-only metric formulas changed')
    spec = importlib.util.spec_from_file_location('_completed_vastenv_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_runtime(runtime):
    if runtime.get('python') != '3.12.15':
        raise ValueError('Exact Python 3.12.15 required')
    if (runtime.get('system') != 'Windows' or
            str(runtime.get('machine', '')).lower() not in ('amd64', 'x86_64')):
        raise ValueError('This diagnostic requires Windows x86-64')
    if any(runtime.get('packages', {}).get(k) != v for k, v in EXPECTED_PACKAGES.items()):
        raise ValueError('Exact Vast-matching torch/torchvision/numpy/scipy versions required')
    if runtime.get('threading') != THREADING:
        raise ValueError('Explicit two-thread diagnostic execution required')


def historical_paths(args):
    if args.method not in METHODS:
        raise ValueError('Unsupported historical method')
    native = args.method not in ('clean', 'a1')
    root = args.data_home / 'runs' / (
        'piw_metafi_four_attacks_s42_3090_v1' if native else
        'piw_metafi_clean_a1_after_wanet_s42_3090_v1')
    group = root / ('baseline_six' if native else 'clean_a1')
    folder = group / 'attacks/metafiplusplus/seed42' / args.method
    return {'root': root, 'group': group, 'folder': folder,
            'checkpoint': folder / 'checkpoint.pt', 'plan': group / 'plan.json',
            'preflight': group / 'preflight_receipt.json',
            'gate': root / 'gates' / (args.method + '.json'),
            'receipt': folder / 'trained_final199.json'}


def validate_plan(saved, w, cfg, c):
    expected = w.plan(cfg)
    if (saved.get('plan_sha256') != c.digest({k: v for k, v in saved.items()
                                            if k != 'plan_sha256'}) or
            saved.get('profile') != w.PROFILE or saved.get('cfg') != cfg or
            saved.get('sources') != expected['sources']):
        raise ValueError('Historical scientific plan/source/config digest mismatch')
    original = c.original_plan()
    if (saved.get('action_sha256') != original['action_sha256'] or
            saved.get('inputs', {}).get('splits') != original['inputs']['splits']):
        raise ValueError('Historical full dataset/list/content/order/action differs')
    before, after = copy.deepcopy(original['trigger']), copy.deepcopy(saved['trigger'])
    if set(before['arrays']) != set(after['arrays']):
        raise ValueError('Historical trigger metadata schema differs')
    for name in before['arrays']:
        left, right = before['arrays'][name], after['arrays'][name]
        if ({k: v for k, v in left.items() if k != 'sha256'} !=
                {k: v for k, v in right.items() if k != 'sha256'}):
            raise ValueError('Historical trigger array shape/dtype differs')
        # These are unused A1 arrays for native methods and Clean. An A1 replay
        # separately requires its complete actual numerical key to match.
        right['sha256'] = left['sha256']
    if before != after:
        raise ValueError('Historical trigger/action/protocol metadata differs')
    runtime = saved.get('runtime')
    if (not isinstance(runtime, dict) or not runtime.get('python') or
            not runtime.get('executable') or not runtime.get('system') or
            any(not runtime.get('packages', {}).get(k) for k in EXPECTED_PACKAGES)):
        raise ValueError('Historical training runtime missing; it must not be replaced')
    # Historical runtime is intentionally NOT compared to the evaluator runtime.
    # No load_plan() call or _runtime_versions() replacement is used.


def validate_final(receipt, blob, plan, actual_sha, controller, method):
    controller.validate_blob(blob, plan, method)
    if (type(receipt.get('epoch')) is not int or receipt['epoch'] != 199 or
            receipt.get('profile') != controller.PROFILE or
            receipt.get('status') != 'TRAINED_NOT_EVALUATED' or
            receipt.get('checkpoint_sha256') != actual_sha or
            receipt.get('plan_sha256') != plan['plan_sha256'] or
            type(blob.get('epoch')) is not int or blob['epoch'] != 199):
        raise ValueError('Matching final199 checkpoint/training receipt required')


def device_guard(device, controller):
    if device == 'cpu':
        return
    if device != 'cuda:0':
        raise ValueError('Only explicit cpu or cuda:0; no fallback')
    controller.require_idle()


def authenticate_trigger(w, saved, blob, trigger, method):
    if method == 'a1':
        if w.trigger_record(trigger) != saved['trigger']:
            raise ValueError('Historical A1 numerical trigger cannot be reconstructed exactly '
                             'in this runtime; no replacement key or fabricated A1 result')
    elif method in ('ftrojan', 'bpp_source', 'fiba', 'wanet_source'):
        trigger.load_state_dict(blob['trigger'])


def load_source(args):
    import torch
    c = load_controller(args.repo)
    device_guard(args.device, c)
    c.configure_torch_cache(args)
    weights = Path(os.environ['TORCH_HOME']) / 'hub/checkpoints/resnet34-b627a593.pth'
    if not weights.is_file() or sha(weights) != WEIGHTS_SHA:
        raise ValueError('Existing exact official ResNet34 cache required; no download/install')
    paths = historical_paths(args)
    for path in paths.values():
        reject_links(path)
    before = {name: sha(paths[name]) for name in
              ('checkpoint', 'plan', 'preflight', 'gate', 'receipt')}
    # Bootstrap only this NEW output's verified snapshot, never historical roots.
    reject_links(args.out/'source_snapshot')
    w = c.load_worker(args, args.method)
    historical = SimpleNamespace(**vars(args))
    historical.out = paths['root']
    historical.four_attacks_only = args.method not in ('clean', 'a1')
    cfg = c.cfg_for(w, historical, args.method)
    saved = read(paths['plan'])
    validate_plan(saved, w, cfg, c)
    evaluation_runtime = w.plan(cfg)['runtime']
    validate_runtime(evaluation_runtime)
    receipt = read(paths['preflight'])
    if (receipt.get('status') != 'PASS' or
            receipt.get('plan_sha256') != saved['plan_sha256'] or
            receipt.get('input_fingerprint') != saved['inputs']['fingerprint']):
        raise ValueError('Historical full-content preflight receipt is missing/foreign')
    for split in ('train_data', 'test_data'):
        path = args.dataset / split / (split + '_list.txt')
        if sha(path) != saved['inputs']['splits'][split]['official_list_sha256']:
            raise ValueError('Official historical split list changed')
    if sha(args.action) != saved['action_sha256']:
        raise ValueError('Historical action file changed')
    stats = w.input_stat_fingerprint(args.dataset, saved['inputs'])
    if stats != receipt['input_stat_fingerprint']:
        raise ValueError('Inputs changed since historical full-content audit')
    blob = w._checkpoint(saved, 'metafiplusplus', args.method, final=True)
    validate_final(read(paths['receipt']), blob, saved, before['checkpoint'], c, args.method)
    gate = c.check_gate(historical, w, cfg, saved, blob['execution'], stats)
    evaluation_execution = w._execution(args.device)
    binding = dict(method=args.method, historical_files_sha256=before,
        historical_paths={k: str(v) for k, v in paths.items()},
        plan_sha256=saved['plan_sha256'], checkpoint_sha256=before['checkpoint'],
        training_runtime=copy.deepcopy(saved['runtime']), training_execution=blob['execution'],
        evaluation_runtime=evaluation_runtime, evaluation_execution=evaluation_execution,
        input_stat_fingerprint=stats, controller_code=c.code_binding(),
        evaluator_sha256=sha(__file__), metric_formulas_sha256=METRICS_SHA,
        official_initialization_sha256=WEIGHTS_SHA)
    return dict(c=c, w=w, cfg=cfg, saved=saved, blob=blob, paths=paths,
                before=before, stats=stats, binding=binding, gate=gate, weights=weights)


def infer(w, saved, blob, args, cfg):
    import numpy as np
    if tuple(w.DOSES) != DOSES:
        raise ValueError('Frozen six-dose contract changed')
    data = w._datasets(saved)
    x, truth, names = w._test_arrays(data['validation'])
    if len(names) != 2586 or names != saved['inputs']['splits']['test_data']['ordered_names']:
        raise ValueError('All 2586 official TEST identities/order required')
    trigger = w.make_trigger(args.method, data['training'], cfg)
    authenticate_trigger(w, saved, blob, trigger, args.method)
    model = w._build('metafiplusplus', args.device)
    model.load_state_dict(blob['model'], strict=True)
    doses = (0.,) if args.method == 'clean' else DOSES
    grids = []
    for dose in doses:
        print(f'DOSE_START method={args.method} dose={dose:g} n=2586 device={args.device}', flush=True)
        predicted = w._predict_grid(model, x, trigger, (dose,), args.device)
        if predicted.shape != (2586, 1, 14, 3) or not np.isfinite(predicted).all():
            raise ValueError('Full finite single-dose predictions required')
        grids.append(predicted)
        print(f'DOSE_COMPLETE method={args.method} dose={dose:g} n=2586', flush=True)
    grid = np.concatenate(grids, axis=1)
    if args.method == 'clean':
        grid = np.repeat(grid, 6, axis=1)
    if grid.shape != (2586, 6, 14, 3) or not np.isfinite(grid).all():
        raise ValueError('Full finite prediction grid required')
    return truth, grid, names


def victim_metrics(truth, grid, adapter, clean=False, repo=None):
    if not clean:
        return metrics_module(repo).victim_metrics(truth, grid, adapter)
    import numpy as np
    from eval.metrics import pa_mpjpe
    truth = adapter._poses(truth, 'truth', (14, 3))
    grid = adapter._poses(grid, 'trigger_predictions', (6, 14, 3))
    if len(truth) != len(grid):
        raise ValueError('Prediction identities/count/order must match')
    pred = grid[:, 0]
    return dict(mainmetrics=dict(clean_mpjpe_mm=float(np.linalg.norm(pred-truth, axis=-1).mean()*1000),
        clean_pampjpe_mm=float(pa_mpjpe(pred, truth).mean()*1000)), dose_rows=[],
        controls=dict(clean_only=True, attack_columns='Not applicable; no trigger applied'))


def write_new_json(path, value):
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def print_result(row, path):
    m = row['metrics']['mainmetrics']
    fields = ('t1_mpjpe_mm', 'target_pck_pct', 'mean_dce_mm',
              'clean_mpjpe_mm', 'clean_pampjpe_mm')
    print('| Method | T-MPJPE mm | Target-PCK % | DCE mm | Clean MPJPE mm | Clean PA mm |')
    print('|---|---:|---:|---:|---:|---:|')
    print('| ' + row['method'] + ' | ' + ' | '.join(
        '--' if m.get(k) is None else f'{m[k]:.2f}' for k in fields) + ' |', flush=True)
    print('RESULT=' + str(path), flush=True)
    print('DIAGNOSTIC REPLAY ONLY: old trained weights; new evaluation libraries. '
          'No retraining or matched training-runtime comparison.', flush=True)


def unchanged(ctx, args):
    if ({name: sha(ctx['paths'][name]) for name in ctx['before']} != ctx['before'] or
            ctx['c'].code_binding() != ctx['binding']['controller_code'] or
            ctx['w'].plan(ctx['cfg'])['sources'] != ctx['saved']['sources'] or
            sha(__file__) != ctx['binding']['evaluator_sha256'] or
            sha(args.repo/'remote_tools/Eval-PiWMetaFiFTrojan3090.py') != METRICS_SHA or
            sha(ctx['weights']) != WEIGHTS_SHA or
            sha(args.action) != ctx['saved']['action_sha256'] or
            any(sha(args.dataset/split/(split+'_list.txt')) !=
                ctx['saved']['inputs']['splits'][split]['official_list_sha256']
                for split in ('train_data', 'test_data')) or
            ctx['w'].input_stat_fingerprint(args.dataset, ctx['saved']['inputs']) != ctx['stats']):
        raise ValueError('Historical checkpoint/source/input changed during replay; refusing result')


def evaluate(args):
    import numpy as np
    # The existing stdlib-only independent advisory lock serializes evaluators.
    helper = metrics_module(args.repo)
    reject_links(args.out/'evaluation.lock')
    with helper.result_lock(args.out):
        ctx = load_source(args)
        folder = args.out / args.device.replace(':', '') / args.method
        reject_links(folder)
        folder.mkdir(parents=True, exist_ok=True)
        report, predictions = folder/'evaluation.json', folder/'predictions.npz'
        reject_links(report)
        reject_links(predictions)
        if report.exists():
            row = read(report)
            if (row.get('profile') != PROFILE or row.get('method') != args.method or
                    row.get('status') != 'DIAGNOSTIC_REPLAY_COMPLETE' or
                    row.get('report_sha256') != ctx['c'].digest({
                        k: v for k, v in row.items() if k != 'report_sha256'}) or
                    row.get('binding') != ctx['binding'] or
                    row.get('predictions_sha256') != sha(predictions)):
                raise ValueError('Existing diagnostic evaluation changed/foreign; no overwrite')
            unchanged(ctx, args)
            print_result(row, report)
            return row
        if predictions.exists():
            raise ValueError('Partial predictions exist; preserve and inspect; no overwrite/retry')
        device_guard(args.device, ctx['c'])
        print(f'EVALUATING {args.method} final199: full2586 TEST, device={args.device}', flush=True)
        truth, grid, names = infer(ctx['w'], ctx['saved'], ctx['blob'], args, ctx['cfg'])
        import piw_v2_adapter as adapter
        metrics = victim_metrics(truth, grid, adapter, clean=args.method == 'clean', repo=args.repo)
        unchanged(ctx, args)
        with predictions.open('xb') as stream:
            arrays = dict(truth=truth, names=np.asarray(names), predictions=grid[:, 0])
            if args.method == 'clean':
                arrays['doses'] = np.asarray([0.])
            else:
                arrays.update(trigger_predictions=grid, doses=np.asarray(DOSES),
                    targets=np.stack([adapter.target_pose(truth, d) for d in DOSES], axis=1))
            np.savez_compressed(stream, **arrays)
        row = dict(profile=PROFILE, status='DIAGNOSTIC_REPLAY_COMPLETE', method=args.method,
            backbone='metafiplusplus', dataset='person-in-wifi-3d', seed=42, epoch=199,
            frames=2586, evaluation_device=args.device, binding=ctx['binding'], metrics=metrics,
            evaluated_doses=[0.] if args.method == 'clean' else list(DOSES),
            predictions_kind='clean-only no-trigger pass' if args.method == 'clean' else 'six-dose victim grid',
            predictions_sha256=sha(predictions),
            clean_reference_status='NOT_SUBSTITUTED; victim-only metrics',
            limitations=['Historical trained weights, not training repeated under new libraries.',
                'Training and evaluation runtimes are distinct and explicitly recorded.',
                'Fresh Vast-env Clean is a separate diagnostic, not a replaced historical reference.',
                'One seed; no across-seed SD or claim of CPU/CUDA bitwise equivalence.',
                'No training, optimizer step, checkpoint migration, or old artifact overwrite.'])
        row['report_sha256'] = ctx['c'].digest(row)
        write_new_json(report, row)
        print_result(row, report)
        return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--data-home', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True,
                        help='One method per fresh Python process; no shared-profile loop')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    reject_links(args.repo)
    reject_links(args.data_home)
    args.repo, args.data_home = args.repo.resolve(), args.data_home.resolve()
    expected = args.data_home/'runs'/PROFILE
    supplied = args.out or expected
    reject_links(supplied)
    if supplied.resolve() != expected.resolve():
        raise ValueError('Use the separate fixed diagnostic output root; no historical overwrite')
    args.out = supplied.resolve()
    args.dataset = args.data_home/'datasets/Person-in-WiFi-3D_processed_v1'
    args.action = args.data_home/'actions/data_bend.npy'
    reject_links(args.dataset)
    reject_links(args.action)
    args.checkpoint_home = args.data_home/'imports/piw_metafi_adamw_unfinished'
    args.work = 'evaluate'
    args.four_attacks_only = args.method not in ('clean', 'a1')
    for name in ('PYTHONHOME', 'PYTHONPATH'):
        if os.environ.get(name):
            raise ValueError(f'Refusing inherited {name}; no environment injection')
    os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', **THREADING)
    sys.dont_write_bytecode = True
    return evaluate(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('[PiW completed Vast-env replay] ERROR:', error, file=sys.stderr, flush=True)
        raise SystemExit(1)
