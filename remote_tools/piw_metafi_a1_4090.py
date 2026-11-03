"""Fresh PiW3D MetaFi++ A1, paired with the completed Clean 4090 diagnostic.

Only scheduling/provenance and external-reference evaluation are new. The
frozen Vast A1 trainer, initialization, trigger, target, optimizer and metrics
are reused without edits. Clean is read-only; it never initializes A1.
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
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROFILE = 'piw_metafi_a1_vastenv_s42_4090_v1'
CLEAN_PROFILE = 'piw_metafi_clean_vastenv_s42_4090_v1'
CELL = Path('attacks/metafiplusplus/seed42/a1')
CLEAN_CELL = Path('attacks/metafiplusplus/seed42/clean')
POISON = dict(n_total=27876, n_poison=2787, n_cover=0)


def diagnostic():
    spec = importlib.util.spec_from_file_location(
        '_a1_readonly_clean4090', HERE / 'piw_metafi_clean_4090.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args, base):
    w = base.load_worker(args, 'a1')
    w.PROFILE = PROFILE
    w.METHODS = ('a1',)
    w.BACKBONES = ('metafiplusplus',)
    w.BASELINES = ()
    return w


def configuration(args, w):
    cfg = w.cfg_for(args.dataset, args.out, args.action)
    cfg['interpretation'] = ('Fresh A1, fixed final199; completed Clean 4090 is '
        'an external read-only control, never a training initialization.')
    if (cfg['methods'] != ['a1'] or cfg['backbones'] != ['metafiplusplus']
            or cfg['seed'] != 42 or cfg['epochs'] != 200 or cfg['final_epoch'] != 199
            or cfg['batch_size'] != 32 or cfg['num_workers'] != 0 or cfg['amp']
            or not cfg['drop_last'] or cfg['test_checkpoint_selection']
            or cfg['recipes'] != {'metafiplusplus': diagnostic().source_probe().RECIPE}):
        raise ValueError('Original A1 AdamW/B32/FP32/seed42/final199 recipe required')
    return cfg


def compatible_configuration(reference, cfg):
    ignored = {'profile', 'outdir', 'methods', 'interpretation'}
    if ({k: v for k, v in reference.items() if k not in ignored} !=
            {k: v for k, v in cfg.items() if k not in ignored}):
        raise ValueError('A1 and completed Clean scientific configuration/paths differ')


def reference(args, base, w, cfg):
    """Validate local completed artifacts, then reuse their full content audit.

    Stat checking detects ordinary file changes; it is not a new content hash
    or a tamper-proof attestation. No file in the Clean run is ever opened for
    writing, extracted into, relabelled or passed to the A1 trainer.
    """
    diag = diagnostic()
    probe = diag.fresh_probe()
    root = args.clean_run
    files = {name: root / name for name in ('plan.json', 'preflight_receipt.json',
        'prelaunch_gate.json', 'smoke_comparison.json', 'comparison.json')}
    files.update({name: root / CLEAN_CELL / name for name in
                  ('checkpoint.pt', 'evaluation.json', 'predictions.npz')})
    for path in files.values():
        probe.reject_linked_path(path)
        if not path.is_file():
            raise ValueError(f'Completed Clean reference missing: {path}')
    records = {name: base.sha(path) for name, path in files.items()}
    saved, receipt, gate, smoke, report, evaluation = [base.read(files[name]) for name in
        ('plan.json', 'preflight_receipt.json', 'prelaunch_gate.json',
         'smoke_comparison.json', 'comparison.json', 'evaluation.json')]
    if saved.get('plan_sha256') != base.digest({k: v for k, v in saved.items() if k != 'plan_sha256'}):
        raise ValueError('Clean reference plan digest differs')
    probe.validate_plan(saved, base.original_plan())
    compatible_configuration(saved['cfg'], cfg)
    current = w.plan(cfg)
    if current['runtime'] != saved['runtime'] or current['sources'] != saved['sources']:
        raise ValueError('Use exactly the completed Clean environment/scientific sources')
    if (receipt.get('status') != 'PASS' or receipt.get('plan_sha256') != saved['plan_sha256']
            or receipt.get('input_fingerprint') != saved['inputs']['fingerprint']
            or receipt.get('action_sha256') != saved['action_sha256']
            or receipt.get('n_train') != 27876 or receipt.get('n_test') != 2586
            or receipt.get('methods') != ['clean']
            or receipt.get('backbones') != ['metafiplusplus']):
        raise ValueError('Completed Clean full-input receipt differs')
    execution = diag.execution_4090(w)
    if (gate.get('profile') != CLEAN_PROFILE or gate.get('plan_sha256') != saved['plan_sha256']
            or gate.get('methods') != ['clean'] or gate.get('prepared_only') is not True
            or gate.get('fresh_from_epoch0') is not True or gate.get('checkpoint_import') is not False
            or gate.get('physical_batch') != 32 or gate.get('runtime') != saved['runtime']
            or gate.get('execution') != execution or gate.get('code') != probe.code_binding(base)
            or gate.get('weights') != probe.weights_binding(base)
            or gate.get('sidecar_4090_platform') != diag.platform_metadata()
            or gate.get('smoke_comparison_sha256') != records['smoke_comparison.json']
            or smoke != probe.smoke_report(smoke['new_smoke'], saved)):
        raise ValueError('Completed Clean code/runtime/GPU/smoke gate differs')
    if (evaluation.get('profile') != CLEAN_PROFILE or evaluation.get('status') != 'COMPLETE'
            or evaluation.get('method') != 'clean' or evaluation.get('backbone') != 'metafiplusplus'
            or evaluation.get('seed') != 42 or evaluation.get('plan_sha256') != saved['plan_sha256']
            or evaluation.get('checkpoint_sha256') != records['checkpoint.pt']
            or evaluation.get('predictions_sha256') != records['predictions.npz']
            or report.get('profile') != CLEAN_PROFILE or report.get('status') != 'COMPLETE'
            or report.get('plan_sha256') != saved['plan_sha256']
            or report.get('checkpoint_sha256') != records['checkpoint.pt']
            or report.get('predictions_sha256') != records['predictions.npz']
            or report.get('runtime') != saved['runtime'] or report.get('code') != gate['code']):
        raise ValueError('Completed Clean evaluation/comparison artifact binding differs')
    for value in evaluation.get('clean_metrics', {}).values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid completed Clean metric')
    if set(evaluation.get('clean_metrics', {})) != {'clean_mpjpe_mm', 'clean_pampjpe_mm'}:
        raise ValueError('Missing completed Clean metrics')
    w._quick_inputs(saved)
    if base.sha(args.action) != saved['action_sha256']:
        raise ValueError('Action content changed after full Clean audit')
    print('REUSING_CLEAN_FULL_AUDIT: checking listed file sizes/mtimes, not rehashing contents', flush=True)
    stats = w.input_stat_fingerprint(args.dataset, saved['inputs'])
    if stats != receipt.get('input_stat_fingerprint') or stats != gate.get('input_stat_fingerprint'):
        raise ValueError('Dataset changed since full Clean audit; no silent bypass')
    return dict(plan=saved, evaluation=evaluation, execution=execution, stats=stats,
                binding=dict(run=str(root), profile=CLEAN_PROFILE, files=records,
                             plan_sha256=saved['plan_sha256']))


def create_plan(args, base, w, cfg, ref):
    path, receipt_path = w.manifest_path(cfg), w.receipt_path(cfg)
    if path.exists() or receipt_path.exists():
        saved = w.load_plan(cfg)
        validate_plan(saved, ref)
        return saved
    if (args.out / CELL / 'checkpoint.pt').exists():
        raise ValueError('Ungated A1 checkpoint; no import or overwrite')
    saved = dict(w.plan(cfg), **{k: copy.deepcopy(ref['plan'][k]) for k in
        ('inputs', 'trigger', 'poison', 'action_sha256')}, clean_reference=ref['binding'])
    # Native constructors check all ordered identities and the exact A1
    # recipients/doses. Full label geometry was checked by the reused audit.
    ds = w._datasets(saved)
    trigger = w.make_trigger('a1', ds['training'], cfg)
    _, poison = w.make_training_data(saved, 'a1', ds, trigger)
    if poison != POISON:
        raise ValueError('Expected full 27876 TRAIN with exactly 2787 A1 poison rows')
    saved['plan_sha256'] = base.digest(saved)
    base.write(path, saved)
    base.write(receipt_path, dict(schema=1, status='PASS', plan_sha256=saved['plan_sha256'],
        input_fingerprint=saved['inputs']['fingerprint'], input_stat_fingerprint=ref['stats'],
        action_sha256=saved['action_sha256'], n_train=27876, n_test=2586,
        methods=['a1'], backbones=['metafiplusplus'],
        audit_reuse=ref['binding'],
        note='Reused completed full Clean content/geometry audit; current lists, trigger, '
             'ordered identities, poison plan and all file stats verified. No new full content audit.'))
    validate_plan(saved, ref)
    return w.load_plan(cfg)


def validate_plan(saved, ref):
    if saved.get('profile') != PROFILE or saved.get('clean_reference') != ref['binding']:
        raise ValueError('A1 plan or external Clean binding changed')
    compatible_configuration(ref['plan']['cfg'], saved['cfg'])
    for key in ('inputs', 'trigger', 'poison', 'action_sha256', 'sources', 'runtime'):
        if saved[key] != ref['plan'][key]:
            raise ValueError(f'A1 and completed Clean differ: {key}')


def smoke(args, base, w, saved):
    path = args.out / 'a1_smoke.json'
    if path.exists():
        raise ValueError('Ungated smoke exists; inspect, no silent repeat/overwrite')
    capture = io.StringIO()
    with redirect_stdout(diagnostic().source_probe().Tee(sys.stdout, capture)):
        w.smoke_cell(saved, 'metafiplusplus', 'a1', 'cuda:0')
    prefix = 'GPU_SMOKE_WORKER_PASS '
    rows = [json.loads(line[len(prefix):]) for line in capture.getvalue().splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError('Exactly one native A1 CUDA smoke receipt required')
    validate_smoke(rows[0])
    if (args.out / CELL / 'checkpoint.pt').exists():
        raise ValueError('Smoke must not save a training checkpoint')
    base.write(path, dict(profile=PROFILE, plan_sha256=saved['plan_sha256'],
        native=rows[0], mix='4 original A1 poison + 28 clean', checkpoint_saved=False,
        full_training_launched=False))


def validate_smoke(row):
    if (row.get('cell') != 'metafiplusplus__s42__a1' or row.get('device') != 'cuda:0'
            or type(row.get('batch')) is not int or row['batch'] != 32
            or type(row.get('generator_steps')) is not int or row['generator_steps'] != 0
            or isinstance(row.get('loss'), bool) or not isinstance(row.get('loss'), (int, float))
            or not math.isfinite(row['loss']) or row['loss'] <= 0
            or any(type(row.get(k)) is not int or row[k] < 0
                   for k in ('peak_cuda_allocated', 'peak_cuda_reserved'))):
        raise ValueError('Invalid native A1 CUDA B32 smoke')


def gate_binding(args, base, saved, ref):
    row = base.read(args.out / 'a1_smoke.json')
    validate_smoke(row['native'])
    if (row.get('profile') != PROFILE or row.get('plan_sha256') != saved['plan_sha256']
            or row.get('checkpoint_saved') is not False or row.get('full_training_launched') is not False):
        raise ValueError('Smoke belongs to a different A1 plan')
    return dict(schema=1, profile=PROFILE, plan_sha256=saved['plan_sha256'], methods=['a1'],
        runtime=saved['runtime'], execution=ref['execution'], clean_reference=ref['binding'],
        input_stat_fingerprint=ref['stats'], physical_batch=32,
        fresh_from_epoch0=True, checkpoint_import=False, max_gpu_workers=1,
        code=dict(runner=base.sha(__file__), helper=base.sha(HERE / 'Start-PiWMetaFiA14090.ps1'),
                  unchanged_clean=diagnostic().fresh_probe().code_binding(base)),
        smoke_sha256=base.sha(args.out / 'a1_smoke.json'))


def check_gate(args, base, w, saved, ref):
    validate_plan(saved, ref)
    if base.read(args.out / 'prelaunch_gate.json') != gate_binding(args, base, saved, ref):
        raise ValueError('A1 code/runtime/GPU/data/reference gate changed')
    receipt = base.read(w.receipt_path(saved['cfg']))
    if receipt.get('input_stat_fingerprint') != ref['stats'] or receipt.get('audit_reuse') != ref['binding']:
        raise ValueError('Reused audit receipt changed')
    path = args.out / CELL / 'checkpoint.pt'
    if path.exists():
        blob = w._checkpoint(saved, 'metafiplusplus', 'a1')
        base.validate_blob(blob, saved, 'a1', POISON)
        if blob['execution'] != ref['execution']:
            raise ValueError('Own A1 checkpoint execution changed; do not migrate')


def evaluate(args, base, w, saved, ref):
    """Same native six-dose predictors/metric function, explicit external Clean.

    Unlike monkeypatching native checkpoint lookup, both checkpoint identities
    retain their own original profiles/plans and are validated independently.
    """
    import numpy as np
    import torch
    own = w._checkpoint(saved, 'metafiplusplus', 'a1', final=True)
    base.validate_blob(own, saved, 'a1', POISON)
    clean_path = args.clean_run / CLEAN_CELL / 'checkpoint.pt'
    clean = torch.load(clean_path, map_location='cpu', weights_only=False)
    base.validate_blob(clean, ref['plan'], 'clean', dict(n_total=27876, n_poison=0, n_cover=0))
    if (clean['epoch'] != 199 or clean['execution'] != ref['execution']
            or own['execution'] != ref['execution']):
        raise ValueError('Both paired victims must be final199 on the recorded 4090')
    ds = w._datasets(saved)
    x, truth, names = w._test_arrays(ds['validation'])
    if len(names) != 2586 or names != saved['inputs']['splits']['test_data']['ordered_names']:
        raise ValueError('Full ordered TEST2586 required')
    with np.load(args.clean_run / CLEAN_CELL / 'predictions.npz', allow_pickle=False) as cache:
        if cache['names'].tolist() != names or not np.array_equal(cache['truth'], truth):
            raise ValueError('Clean reference TEST names/truth differ')
    trigger = w.make_trigger('a1', ds['training'], saved['cfg'])
    grids = []
    for role, blob in (('a1', own), ('clean_reference', clean)):
        print(f'EVALUATING {role}: full2586 TEST, six doses', flush=True)
        model = w._build('metafiplusplus', 'cuda:0')
        model.load_state_dict(blob['model'], strict=True)
        grids.append(w._predict_grid(model, x, trigger, w.DOSES, 'cuda:0'))
        del model
    victim, control = grids
    metrics = w.measure_predictions(truth, victim[:, 0], victim, control[:, 0], control)
    folder = args.out / CELL
    w._atomic_npz(folder / 'predictions.npz', truth=truth, names=np.asarray(names),
        doses=np.asarray(w.DOSES), predictions=victim[:, 0], trigger_predictions=victim,
        reference_predictions=control[:, 0], reference_trigger_predictions=control,
        targets=np.stack([w.target_pose(truth, d) for d in w.DOSES], axis=1))
    # Recheck reference byte binding after inference, without rewriting it.
    if base.sha(clean_path) != ref['binding']['files']['checkpoint.pt']:
        raise ValueError('Reference checkpoint changed during evaluation')
    result = dict(profile=PROFILE, status='COMPLETE', backbone='metafiplusplus', method='a1',
        seed=42, plan_sha256=saved['plan_sha256'], checkpoint_sha256=base.sha(folder / 'checkpoint.pt'),
        clean_checkpoint_sha256=base.sha(clean_path), clean_reference=ref['binding'],
        predictions_sha256=base.sha(folder / 'predictions.npz'), poison=own['poison'], metrics=metrics,
        limitations=['One seed, fixed final199; no uncertainty estimate or GPU causality claim.',
                     'PiW input/model and attack operators are dataset adaptations.',
                     'No image/classification or physical RF reproduction claim.',
                     'Historical PiW target/LR were viewed during development.',
                     'Completed Clean4090 is an external read-only reference, not A1 initialization.'])
    base.write(folder / 'evaluation.json', result)


def summary(args, base, saved, ref):
    folder = args.out / CELL
    result = base.read(folder / 'evaluation.json')
    if (result.get('status') != 'COMPLETE' or result.get('profile') != PROFILE
            or result.get('method') != 'a1' or result.get('backbone') != 'metafiplusplus'
            or result.get('seed') != 42 or result.get('poison') != POISON
            or result.get('plan_sha256') != saved['plan_sha256']
            or result.get('clean_reference') != ref['binding']
            or result.get('clean_checkpoint_sha256') != ref['binding']['files']['checkpoint.pt']
            or result.get('checkpoint_sha256') != base.sha(folder / 'checkpoint.pt')
            or result.get('predictions_sha256') != base.sha(folder / 'predictions.npz')):
        raise ValueError('Completed A1 evaluation binding differs; no overwrite')
    m, c = result['metrics']['mainmetrics'], ref['evaluation']['clean_metrics']
    table = '# PiW3D / MetaFi++ / seed42 / 4090\n\n'
    table += '| Method | T-MPJPE mm | Target-PCK % | DCE mm | Clean MPJPE mm | Clean PA mm |\n'
    table += '| --- | ---: | ---: | ---: | ---: | ---: |\n'
    table += f"| Clean reference (completed 4090) | — | — | — | {c['clean_mpjpe_mm']:.2f} | {c['clean_pampjpe_mm']:.2f} |\n"
    table += f"| A1 (fresh 4090) | {m['t1_mpjpe_mm']:.2f} | {m['target_pck_pct']:.2f} | {m['mean_dce_mm']:.2f} | {m['clean_mpjpe_mm']:.2f} | {m['clean_pampjpe_mm']:.2f} |\n"
    table += '\nMain attack metrics at dose1; DCE averages five positive doses. One seed.\n'
    path = args.out / 'RESULT.md'
    if path.exists() and path.read_text(encoding='utf-8') != table:
        raise ValueError('Existing result table differs; no overwrite')
    if not path.exists():
        with path.open('x', encoding='utf-8') as stream:
            stream.write(table)
    print(table, flush=True)


def run(args, base, w, cfg, ref):
    if args.mode == 'prepare':
        if (args.out / 'prelaunch_gate.json').exists():
            check_gate(args, base, w, w.load_plan(cfg), ref)
            print('PREPARED_ALREADY; no repeated smoke or training', flush=True)
            return
        saved = create_plan(args, base, w, cfg, ref)
        smoke(args, base, w, saved)
        base.write(args.out / 'prelaunch_gate.json', gate_binding(args, base, saved, ref))
        print('PREPARED_ONLY: reused verified Clean audit + native A1 B32 CUDA smoke; no full training', flush=True)
        return
    saved = w.load_plan(cfg)
    check_gate(args, base, w, saved, ref)
    if args.mode == 'check':
        print('VERIFIED_PRELAUNCH_GATE; no training', flush=True)
        return
    if not (args.out / CELL / 'evaluation.json').exists():
        cp = args.out / CELL / 'checkpoint.pt'
        final = cp.is_file() and w._checkpoint(saved, 'metafiplusplus', 'a1')['epoch'] == 199
        if not final:
            # Native fresh build/seed/AdamW. Resumes ONLY this run's own complete
            # optimizer/RNG checkpoint. Clean is never handed to train_cell.
            w.train_cell(saved, 'metafiplusplus', 'a1', 'cuda:0')
        evaluate(args, base, w, saved, ref)
    summary(args, base, saved, ref)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inspect', 'prepare', 'check', 'launch', 'status'), default='inspect')
    parser.add_argument('--data-home', type=Path, default=Path.home() / 'backdooranalog')
    args = parser.parse_args(argv)
    args.data_home = args.data_home.absolute()
    args.out = args.data_home / 'runs' / PROFILE
    args.clean_run = args.data_home / 'runs' / CLEAN_PROFILE
    args.dataset = args.data_home / 'datasets/Person-in-WiFi-3D_processed_v1'
    args.action = args.data_home / 'actions/data_bend.npy'
    args.env_dir = args.data_home / 'envs/piw-metafi-vast211-4090'
    if args.mode == 'inspect':
        print(json.dumps(dict(profile=PROFILE, methods=['a1'], seed=42, epochs=200,
            max_gpu_workers=1, clean_readonly=str(args.clean_run), out=str(args.out),
            note='No install, GPU use or training. Reuses the existing Clean4090 environment.'), indent=2))
        return 0
    if args.mode == 'status':
        print('READ_ONLY; logs are not proof of a live worker', flush=True)
        logs = sorted((args.out / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
        if logs:
            print(f'LATEST_LOG={logs[-1]}', flush=True)
            with logs[-1].open(encoding='utf-8', errors='replace') as stream:
                print(''.join(deque(stream, maxlen=18)), flush=True)
        if (args.out / 'RESULT.md').is_file():
            print((args.out / 'RESULT.md').read_text(encoding='utf-8'), flush=True)
        return 0
    diag = diagnostic()
    probe = diag.source_probe()
    for p in (args.data_home, args.out, args.clean_run, args.dataset, args.action, args.env_dir):
        probe.reject_linked_path(p)
    args.data_home = args.data_home.resolve()
    probe.PROFILE = PROFILE  # Path/venv ownership validation only, not Clean training.
    probe.validate_paths(args)
    if os.environ.get('PIW_SMOKE_BARRIER_DIR') or os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME'):
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
        base.require_idle()  # Unknown/compute conflicts are blocked, never killed.
        w = worker(args, base)
        cfg = configuration(args, w)
        ref = reference(args, base, w, cfg)
        run(args, base, w, cfg, ref)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'[PiW MetaFi A1 4090] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
