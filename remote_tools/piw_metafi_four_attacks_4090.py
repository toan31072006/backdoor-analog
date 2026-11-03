"""Four fresh PiW3D MetaFi++ baselines, serial RTX4090, external Clean control.

Reuse the frozen Vast scientific worker and the verified Clean4090 input audit.
Never train Clean/A1, import 3090 checkpoints, change recipes, or auto-retry.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack, redirect_stdout
import copy
import importlib.util
import io
import json
import math
import ntpath
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
PROFILE = 'piw_metafi_four_attacks_vastenv_s42_4090_v1'
METHODS = ('fiba', 'wanet_source', 'ftrojan', 'bpp_source')
NAMES = dict(fiba='FIBA-CSI', wanet_source='WaNet-CSI', ftrojan='FTrojan-CSI', bpp_source='BppAttack-CSI')
COVERS = dict(fiba=.1, wanet_source=.2, ftrojan=0., bpp_source=.1)
CELL = Path('attacks/metafiplusplus/seed42')


class _DesktopGpuGuard:
    """Extend this queue's guard without changing historical A1/Clean code."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def conflict_rows(self, text, windows=False, own_pid=None, process_names=None):
        rows = self.base.conflict_rows(text, windows, own_pid, process_names)
        if not windows:
            return rows
        desktop = ntpath.normcase(ntpath.normpath(ntpath.join(
            os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
            'Microsoft', 'Copilot', 'Application', 'undocked_copilot', 'copilotapphost.exe')))
        return [(pid, name) for pid, name in rows
                if ntpath.normcase(ntpath.normpath(name)) != desktop]


def require_idle(a1, base):
    # Reuse the query, hidden-PID fallback and exact TeamViewer/WUDFHost rules.
    # Only the observed full Windows Copilot desktop path is additionally allowed.
    a1.require_idle(_DesktopGpuGuard(base))


def a1_sidecar():
    spec = importlib.util.spec_from_file_location('_four_attacks_readonly_a1', HERE / 'piw_metafi_a1_4090.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args, base):
    # Clean/A1 entry supplies the already-used pretrained MetaFi++/AdamW recipe;
    # its unchanged underlying worker also contains these four native baselines.
    # The base loader preserves canonical TRAIN donors and exact Linux FTrojan key.
    w = base.load_worker(args, 'a1')
    w.PROFILE, w.METHODS = PROFILE, METHODS
    w.BACKBONES, w.BASELINES = ('metafiplusplus',), METHODS
    return w


def configuration(args, w):
    cfg = w.cfg_for(args.dataset, args.out, args.action)
    cfg['interpretation'] = ('Four selected fresh PiW3D adaptations; external completed Clean4090; '
                             'no Clean/A1 training, checkpoint import or TEST tuning.')
    if (cfg['methods'] != list(METHODS) or cfg['backbones'] != ['metafiplusplus']
            or cfg['recipes'] != {'metafiplusplus': a1_sidecar().diagnostic().source_probe().RECIPE}
            or cfg['seed'] != 42 or cfg['epochs'] != 200 or cfg['final_epoch'] != 199
            or cfg['batch_size'] != 32 or cfg['num_workers'] != 0 or cfg['amp']
            or not cfg['drop_last'] or cfg['test_checkpoint_selection']
            or any(cfg['cover_ratios'][m] != COVERS[m] for m in METHODS)):
        raise ValueError('Expected exactly four original AdamW/B32/FP32/final199 recipes')
    return cfg


def validate_poison(method, row):
    if (method not in METHODS or row.get('method') != method or row.get('n_total') != 27876
            or row.get('n_poison') != 2787 or row.get('n_cover') != math.floor(COVERS[method] * 27876)
            or row.get('cover_ratio') != COVERS[method] or row.get('phase_policy') != 'unchanged'
            or row.get('target') != 'piw_body_hip_axis_v2'
            or not isinstance(row.get('operator_keys'), list) or len(row['operator_keys']) != 3):
        raise ValueError(f'Native poison/cover/operator record differs: {method}')


def poison_records(w, saved):
    ds = w._datasets(saved)
    rows = {}
    for method in METHODS:
        trigger = w.make_trigger(method, ds['training'], saved['cfg'])
        _, row = w.make_training_data(saved, method, ds, trigger)
        validate_poison(method, row)
        rows[method] = row
    return rows


def validate_plan(a1, saved, ref):
    if saved.get('profile') != PROFILE or saved['cfg']['methods'] != list(METHODS):
        raise ValueError('Foreign four-attack plan/scope')
    if saved.get('clean_reference') != ref['binding']:
        raise ValueError('Completed Clean reference changed')
    a1.compatible_configuration(ref['plan']['cfg'], saved['cfg'])
    for key in ('inputs', 'trigger', 'poison', 'action_sha256', 'sources', 'runtime'):
        if saved[key] != ref['plan'][key]:
            raise ValueError(f'Four-attack plan differs from completed Clean: {key}')


def create_plan(args, a1, base, w, cfg, ref):
    if w.manifest_path(cfg).exists() or w.receipt_path(cfg).exists():
        saved = w.load_plan(cfg)
        validate_plan(a1, saved, ref)
        return saved
    if any((args.out / CELL / m / 'checkpoint.pt').exists() for m in METHODS):
        raise ValueError('Ungated checkpoint found; never import or overwrite')
    saved = dict(w.plan(cfg), **{k: copy.deepcopy(ref['plan'][k]) for k in
        ('inputs', 'trigger', 'poison', 'action_sha256')}, clean_reference=ref['binding'])
    saved['native_records'] = poison_records(w, saved)
    saved['plan_sha256'] = base.digest(saved)
    base.write(w.manifest_path(cfg), saved)
    base.write(w.receipt_path(cfg), dict(schema=1, status='PASS',
        plan_sha256=saved['plan_sha256'], input_fingerprint=saved['inputs']['fingerprint'],
        input_stat_fingerprint=ref['stats'], action_sha256=saved['action_sha256'],
        n_train=27876, n_test=2586, methods=list(METHODS), backbones=['metafiplusplus'],
        audit_reuse=ref['binding'], note='Reused full Clean content/geometry audit; '
        'current official lists, all file stats, ordered IDs, trigger and native poison/cover keys verified.'))
    validate_plan(a1, saved, ref)
    return w.load_plan(cfg)


def native_smoke(args, a1, base, w, saved, method):
    path = args.out / 'smokes' / (method + '.json')
    if path.exists():
        raise ValueError(f'Ungated partial smoke exists: {method}; inspect, no automatic repeat')
    capture = io.StringIO()
    with redirect_stdout(a1.diagnostic().source_probe().Tee(sys.stdout, capture)):
        w.smoke_cell(saved, 'metafiplusplus', method, 'cuda:0')
    prefix = 'GPU_SMOKE_WORKER_PASS '
    rows = [json.loads(s[len(prefix):]) for s in capture.getvalue().splitlines() if s.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError('Exactly one native B32 CUDA smoke receipt required')
    row = rows[0]
    # Reuse unchanged numeric/batch validation without relabelling the saved receipt.
    a1.validate_smoke(dict(row, cell='metafiplusplus__s42__a1'))
    if row.get('cell') != f'metafiplusplus__s42__{method}':
        raise ValueError('Smoke is for another method')
    if any((args.out / CELL / m / 'checkpoint.pt').exists() for m in METHODS):
        raise ValueError('Prepare smoke must not create any training checkpoint')
    base.write(path, dict(profile=PROFILE, plan_sha256=saved['plan_sha256'], method=method,
        native=row, poison=saved['native_records'][method], checkpoint_saved=False,
        full_training_launched=False))


def gate_binding(args, a1, base, saved, ref):
    hashes = {}
    for method in METHODS:
        path = args.out / 'smokes' / (method + '.json')
        row = base.read(path)
        if (row.get('profile') != PROFILE or row.get('plan_sha256') != saved['plan_sha256']
                or row.get('method') != method or row.get('poison') != saved['native_records'][method]
                or row.get('checkpoint_saved') is not False or row.get('full_training_launched') is not False
                or row['native'].get('cell') != f'metafiplusplus__s42__{method}'):
            raise ValueError('Four-method smoke receipt differs')
        a1.validate_smoke(dict(row['native'], cell='metafiplusplus__s42__a1'))
        hashes[method] = base.sha(path)
    return dict(schema=1, profile=PROFILE, methods=list(METHODS), max_gpu_workers=1,
        plan_sha256=saved['plan_sha256'], runtime=saved['runtime'], execution=ref['execution'],
        clean_reference=ref['binding'], input_stat_fingerprint=ref['stats'],
        physical_batch=32, fresh_from_epoch0=True, checkpoint_import=False, smokes=hashes,
        code=dict(runner=base.sha(__file__), helper=base.sha(HERE / 'Start-PiWMetaFiFourAttacks4090.ps1'),
                  shared_a1_sidecar=base.sha(HERE / 'piw_metafi_a1_4090.py'),
                  unchanged_clean=a1.diagnostic().fresh_probe().code_binding(base)))


def check_gate(args, a1, base, w, saved, ref):
    validate_plan(a1, saved, ref)
    if base.read(args.out / 'prelaunch_gate.json') != gate_binding(args, a1, base, saved, ref):
        raise ValueError('Four-method source/runtime/GPU/reference gate changed')
    receipt = base.read(w.receipt_path(saved['cfg']))
    if receipt.get('audit_reuse') != ref['binding'] or receipt.get('input_stat_fingerprint') != ref['stats']:
        raise ValueError('Reused full audit receipt changed')
    if poison_records(w, saved) != saved['native_records']:
        raise ValueError('Actual native trigger/poison/cover keys changed')
    for method in METHODS:
        if (args.out / CELL / method / 'checkpoint.pt').exists():
            own_checkpoint(base, w, saved, ref, method)


def own_checkpoint(base, w, saved, ref, method, final=False):
    blob = w._checkpoint(saved, 'metafiplusplus', method, final=final)
    base.validate_blob(blob, saved, method, saved['native_records'][method])
    if blob['execution'] != ref['execution']:
        raise ValueError('Own checkpoint execution differs; no cross-GPU migration')
    return blob


def completed(args, base, saved, ref, method):
    folder = args.out / CELL / method
    path = folder / 'evaluation.json'
    if not path.exists():
        return None
    row = base.read(path)
    if (row.get('profile') != PROFILE or row.get('status') != 'COMPLETE'
            or row.get('method') != method or row.get('backbone') != 'metafiplusplus' or row.get('seed') != 42
            or row.get('plan_sha256') != saved['plan_sha256'] or row.get('epoch') != 199
            or row.get('clean_reference') != ref['binding']
            or row.get('poison') != saved['native_records'][method]
            or row.get('checkpoint_sha256') != base.sha(folder / 'checkpoint.pt')
            or row.get('clean_checkpoint_sha256') != ref['binding']['files']['checkpoint.pt']
            or row.get('predictions_sha256') != base.sha(folder / 'predictions.npz')):
        raise ValueError(f'Existing {method} evaluation changed; no overwrite')
    return row


def evaluate(args, a1, base, w, saved, ref, method):
    import numpy as np
    import torch
    if completed(args, base, saved, ref, method) is not None:
        print('EVALUATION_ALREADY_COMPLETE ' + method, flush=True)
        return
    own = own_checkpoint(base, w, saved, ref, method, final=True)
    clean_path = args.clean_run / a1.CLEAN_CELL / 'checkpoint.pt'
    clean = torch.load(clean_path, map_location='cpu', weights_only=False)
    base.validate_blob(clean, ref['plan'], 'clean', dict(n_total=27876, n_poison=0, n_cover=0))
    if clean['epoch'] != 199 or clean['execution'] != ref['execution']:
        raise ValueError('External Clean must be final199 on the recorded 4090')
    ds = w._datasets(saved)
    x, truth, names = w._test_arrays(ds['validation'])
    if len(names) != 2586 or names != saved['inputs']['splits']['test_data']['ordered_names']:
        raise ValueError('Full ordered TEST2586 required')
    with np.load(args.clean_run / a1.CLEAN_CELL / 'predictions.npz', allow_pickle=False) as cache:
        if cache['names'].tolist() != names or not np.array_equal(cache['truth'], truth):
            raise ValueError('External Clean TEST names/truth differ')
    trigger = w.make_trigger(method, ds['training'], saved['cfg'])
    trigger.load_state_dict(own['trigger'])  # All three native operator/RNG states.
    grids = []
    for role, blob in ((method, own), ('clean_reference', clean)):
        print(f'EVALUATING {role}: trigger={method}, full2586 TEST, six doses', flush=True)
        model = w._build('metafiplusplus', 'cuda:0')
        model.load_state_dict(blob['model'], strict=True)
        grids.append(w._predict_grid(model, x, trigger, w.DOSES, 'cuda:0'))
        del model
    grid, control = grids
    metrics = w.measure_predictions(truth, grid[:, 0], grid, control[:, 0], control)
    folder = args.out / CELL / method
    w._atomic_npz(folder / 'predictions.npz', truth=truth, names=np.asarray(names), doses=np.asarray(w.DOSES),
        predictions=grid[:, 0], trigger_predictions=grid, reference_predictions=control[:, 0],
        reference_trigger_predictions=control, targets=np.stack([w.target_pose(truth, d) for d in w.DOSES], axis=1))
    if base.sha(clean_path) != ref['binding']['files']['checkpoint.pt']:
        raise ValueError('External Clean changed during evaluation')
    base.write(folder / 'evaluation.json', dict(profile=PROFILE, status='COMPLETE', method=method,
        backbone='metafiplusplus', seed=42, epoch=199, plan_sha256=saved['plan_sha256'],
        checkpoint_sha256=base.sha(folder / 'checkpoint.pt'), clean_checkpoint_sha256=base.sha(clean_path),
        clean_reference=ref['binding'], predictions_sha256=base.sha(folder / 'predictions.npz'),
        poison=own['poison'], metrics=metrics,
        limitations=['One seed, selected-method PiW digital CSI adaptations; not the full multi-seed table.',
                     'Native method-specific cover ratios/budgets; not equal-distortion attacks.',
                     'Historical PiW target/LR were viewed during development.']))
    print('EVAL_COMPLETE ' + method + ' ' + json.dumps(metrics['mainmetrics']), flush=True)


def existing_a1(args, a1, base, ref):
    root = args.data_home / 'runs' / a1.PROFILE
    folder = root / a1.CELL
    path = folder / 'evaluation.json'
    if not path.exists():
        return None
    for p in (path, root / 'plan.json', folder / 'checkpoint.pt', folder / 'predictions.npz'):
        a1.diagnostic().source_probe().reject_linked_path(p)
    saved, result = base.read(root / 'plan.json'), base.read(path)
    a1.validate_plan(saved, ref)
    if (saved['plan_sha256'] != base.digest({k: v for k, v in saved.items() if k != 'plan_sha256'})
            or saved['cfg']['methods'] != ['a1'] or result.get('profile') != a1.PROFILE
            or result.get('status') != 'COMPLETE' or result.get('method') != 'a1'
            or result.get('backbone') != 'metafiplusplus' or result.get('seed') != 42
            or result.get('plan_sha256') != saved['plan_sha256'] or result.get('poison') != a1.POISON
            or result.get('clean_reference') != ref['binding']
            or result.get('clean_checkpoint_sha256') != ref['binding']['files']['checkpoint.pt']
            or result.get('checkpoint_sha256') != base.sha(folder / 'checkpoint.pt')
            or result.get('predictions_sha256') != base.sha(folder / 'predictions.npz')):
        raise ValueError('Existing A1 result is foreign/changed; never relabel or retrain it')
    return result['metrics']['mainmetrics']


def summary(args, a1, base, saved, ref):
    table = '# MetaFi++ / Person-in-WiFi-3D / seed42 / 4090\n\n'
    table += '| Method | T-MPJPE mm | Target-PCK % | DCE mm | Clean MPJPE mm | Clean PA mm |\n'
    table += '| --- | ---: | ---: | ---: | ---: | ---: |\n'
    c = ref['evaluation']['clean_metrics']
    table += f"| Clean reference | — | — | — | {c['clean_mpjpe_mm']:.2f} | {c['clean_pampjpe_mm']:.2f} |\n"
    rows = []
    for method in METHODS:
        result = completed(args, base, saved, ref, method)
        if result is None:
            table += f'| {NAMES[method]} | pending | — | — | — | — |\n'
            continue
        rows.append(result)
        m = result['metrics']['mainmetrics']
        table += metric_row(NAMES[method], m)
    m = existing_a1(args, a1, base, ref)
    if m is not None:
        table += metric_row('A1 (existing; not retrained)', m)
    table += '\nOne seed; T/PCK at dose1, DCE mean of five positive doses. '
    table += 'Missing A1 is not scheduled or synthesized.\n'
    base.write(args.out / 'results.json', dict(profile=PROFILE,
        status='COMPLETE' if len(rows) == 4 else 'PARTIAL', plan_sha256=saved['plan_sha256'],
        records=rows, existing_a1_metrics=m, clean_reference=ref['binding']))
    # This is the new queue's derived report, never a historical result file.
    (args.out / 'RESULT.md').write_text(table, encoding='utf-8')
    print(table, flush=True)


def metric_row(name, m):
    return f"| {name} | {m['t1_mpjpe_mm']:.2f} | {m['target_pck_pct']:.2f} | {m['mean_dce_mm']:.2f} | {m['clean_mpjpe_mm']:.2f} | {m['clean_pampjpe_mm']:.2f} |\n"


def run(args, a1, base, w, cfg, ref):
    if args.mode == 'prepare':
        if (args.out / 'prelaunch_gate.json').exists():
            check_gate(args, a1, base, w, w.load_plan(cfg), ref)
            print('PREPARED_ALREADY; no repeated smoke/full training', flush=True)
            return
        saved = create_plan(args, a1, base, w, cfg, ref)
        for method in METHODS:
            require_idle(a1, base)
            native_smoke(args, a1, base, w, saved, method)
        base.write(args.out / 'prelaunch_gate.json', gate_binding(args, a1, base, saved, ref))
        print('PREPARED_ONLY: reused full Clean audit + four serial CUDA B32 smokes; no full training', flush=True)
        return
    saved = w.load_plan(cfg)
    check_gate(args, a1, base, w, saved, ref)
    if args.mode == 'check':
        print('VERIFIED_PRELAUNCH_GATE; no training', flush=True)
        return
    # Keep the native fixed-final policy: all four victims complete before TEST.
    for method in METHODS:
        require_idle(a1, base)
        path = args.out / CELL / method / 'checkpoint.pt'
        if path.exists() and own_checkpoint(base, w, saved, ref, method)['epoch'] == 199:
            print('TRAIN_FINAL199_REUSED ' + method, flush=True)
            continue
        print('TRAINING ' + method + '; native fresh initialization or own exact resume', flush=True)
        w.train_cell(saved, 'metafiplusplus', method, 'cuda:0')
        own_checkpoint(base, w, saved, ref, method, final=True)
    for method in METHODS:
        require_idle(a1, base)
        evaluate(args, a1, base, w, saved, ref, method)
    summary(args, a1, base, saved, ref)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inspect', 'prepare', 'check', 'launch', 'status'), default='inspect')
    parser.add_argument('--data-home', type=Path, default=Path.home() / 'backdooranalog')
    args = parser.parse_args(argv)
    args.data_home = args.data_home.absolute()
    args.out = args.data_home / 'runs' / PROFILE
    args.clean_run = args.data_home / 'runs/piw_metafi_clean_vastenv_s42_4090_v1'
    args.dataset = args.data_home / 'datasets/Person-in-WiFi-3D_processed_v1'
    args.action = args.data_home / 'actions/data_bend.npy'
    args.env_dir = args.data_home / 'envs/piw-metafi-vast211-4090'
    if args.mode == 'inspect':
        print(json.dumps(dict(profile=PROFILE, methods=list(METHODS), seed=42, epochs=200,
            max_gpu_workers=1, clean_readonly=str(args.clean_run), out=str(args.out),
            note='No install, GPU use or training; Clean/A1 are never scheduled.'), indent=2))
        return 0
    if args.mode == 'status':
        print('READ_ONLY; file existence is not proof of a live worker', flush=True)
        for method in METHODS:
            p = args.out / CELL / method / 'evaluation.json'
            print(method + ': ' + ('evaluation exists' if p.exists() else 'not evaluated'))
        logs = sorted((args.out / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
        if logs:
            print('LATEST_LOG=' + str(logs[-1]), flush=True)
            with logs[-1].open(encoding='utf-8', errors='replace') as stream:
                print(''.join(deque(stream, maxlen=18)), flush=True)
        if (args.out / 'RESULT.md').is_file():
            print((args.out / 'RESULT.md').read_text(encoding='utf-8'), flush=True)
        return 0
    a1 = a1_sidecar()
    probe = a1.diagnostic().source_probe()
    for p in (args.data_home, args.out, args.clean_run, args.dataset, args.action, args.env_dir):
        probe.reject_linked_path(p)
    args.data_home = args.data_home.resolve()
    probe.PROFILE = PROFILE
    probe.validate_paths(args)
    if any(os.environ.get(k) for k in ('PIW_SMOKE_BARRIER_DIR', 'PYTHONPATH', 'PYTHONHOME')):
        raise ValueError('Foreign parallel smoke/Python environment refused')
    os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', MKL_THREADING_LAYER='SEQUENTIAL')
    sys.dont_write_bytecode = True
    base = probe.base_module()
    base.configure_torch_cache(args)
    with ExitStack() as stack:
        stack.enter_context(base.lock(args.data_home / 'tools/clean_vastenv_gpu0_lock'))
        stack.enter_context(base.lock(args.out))
        stack.enter_context(probe.run_log(args))
        if shutil.disk_usage(args.data_home).free < 5 * 2**30:
            raise ValueError('At least 5 GiB free required; no cleanup or checkpoint deletion')
        require_idle(a1, base)
        w = worker(args, base)
        cfg = configuration(args, w)
        ref = a1.reference(args, base, w, cfg)
        run(args, a1, base, w, cfg, ref)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'[PiW MetaFi four attacks 4090] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
