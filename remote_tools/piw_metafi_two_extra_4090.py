"""Serial PiW3D MetaFi++ TSBA/Smooth-LF adaptations; external jobs allowed.

Reuses the frozen scientific worker and completed Clean4090 audit, never its
weights for victim initialization. No process enumeration, PID cap, kill,
package installation, automatic retry or reduced-batch fallback.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
PROFILE = 'piw_metafi_two_extra_s42_4090_serial_v1'
METHODS = ('tsba_adapted', 'smooth_lf_csi')
NAMES = {'tsba_adapted': 'TSBA-adapted', 'smooth_lf_csi': 'Smooth-LF-CSI'}
POLICY = dict(max_owned_gpu_workers=1, order=list(METHODS),
              external_gpu_process_limit=None, external_evaluation_allowed=True,
              no_process_kill=True, no_automatic_retry=True,
              no_batch_fallback=True, training_from_external_checkpoint=False)


def queue_module():
    """Use an isolated orchestration module; historical files stay unchanged."""
    spec = importlib.util.spec_from_file_location('_piw_two_extra_queue',
                                                HERE / 'piw_metafi_four_attacks_4090.py')
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    q.PROFILE, q.METHODS, q.NAMES = PROFILE, METHODS, NAMES
    q.COVERS = dict.fromkeys(METHODS, 0.)
    # The old orchestration is explicitly configured for this two-cell scope.
    # We do NOT change native scientific functions or any global GPU guard.
    q.validate_poison = validate_poison
    q.require_idle = lambda *_: None  # User-approved nonexclusive queue only.
    q.native_smoke = lambda *a: native_smoke(q, *a)
    q.gate_binding = lambda *a: gate_binding(q, *a)
    q.evaluate = lambda *a: evaluate(q, *a)
    q.summary = lambda *a: summary(q, *a)
    original_checkpoint = q.own_checkpoint

    def checkpoint(*args, **kwargs):
        result = original_checkpoint(*args, **kwargs)
        method = args[-1]
        if method == 'tsba_adapted':
            if not result.get('trigger') or not result.get('trigger_optimizer'):
                raise ValueError('TSBA requires its own generator and optimizer state')
        elif result.get('trigger') is not None or result.get('trigger_optimizer') is not None:
            raise ValueError('Smooth-LF must not have a learned generator state')
        return result

    q.own_checkpoint = checkpoint
    return q


def validate_poison(method, row):
    if (method not in METHODS or set(row) != {'n_total', 'n_poison', 'n_cover', 'poison_sha256'}
            or row['n_total'] != 27876 or row['n_poison'] != 2787 or row['n_cover'] != 0
            or not isinstance(row['poison_sha256'], str) or len(row['poison_sha256']) != 64):
        raise ValueError(f'Full original poison plan required: {method}')


def configuration(q, args, w):
    cfg = q.configuration(args, w)
    cfg['interpretation'] = ('Two fresh MetaFi++ PiW digital-CSI adaptations; seed42; '
        'serial owned workers, external evaluation allowed; completed Clean4090 read-only. '
        'Not a complete comparison table or exact source-paper reproduction.')
    if (cfg['rho'] != .1 or cfg['eval_doses'] != [0., .2, .4, .6, .8, 1.]
            or cfg['tsba'] != dict(eps=.1, hidden=32, generator_lr=.001,
                                  warmup_epochs=10, generator_steps=4)):
        raise ValueError('Frozen TSBA/dose/poison adaptation differs')
    return cfg


def validate_smoke(row, method):
    steps = 4 if method == 'tsba_adapted' else 0
    if (method not in METHODS or row.get('cell') != f'metafiplusplus__s42__{method}'
            or row.get('device') != 'cuda:0' or type(row.get('batch')) is not int
            or row['batch'] != 32 or type(row.get('generator_steps')) is not int
            or row['generator_steps'] != steps or isinstance(row.get('loss'), bool)
            or not isinstance(row.get('loss'), (int, float))
            or not math.isfinite(row['loss']) or row['loss'] <= 0
            or any(type(row.get(k)) is not int or row[k] < 0
                   for k in ('peak_cuda_allocated', 'peak_cuda_reserved'))):
        raise ValueError(f'Invalid B32 smoke/generator step count: {method}')


def native_smoke(q, args, a1, base, w, saved, method):
    path = args.out / 'smokes' / (method + '.json')
    if path.exists():
        raise ValueError(f'Partial smoke exists: {path}; inspect, do not overwrite')
    capture = io.StringIO()
    with redirect_stdout(a1.diagnostic().source_probe().Tee(sys.stdout, capture)):
        w.smoke_cell(saved, 'metafiplusplus', method, 'cuda:0')
    prefix = 'GPU_SMOKE_WORKER_PASS '
    rows = [json.loads(s[len(prefix):]) for s in capture.getvalue().splitlines() if s.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError('Exactly one native CUDA smoke record required')
    validate_smoke(rows[0], method)
    if any((args.out / q.CELL / m / 'checkpoint.pt').exists() for m in METHODS):
        raise ValueError('Prepare must not save a training checkpoint')
    base.write(path, dict(profile=PROFILE, method=method, plan_sha256=saved['plan_sha256'],
        native=rows[0], poison=saved['native_records'][method],
        checkpoint_saved=False, full_training_launched=False))


def gate_binding(q, args, a1, base, saved, ref):
    hashes = {}
    for method in METHODS:
        path = args.out / 'smokes' / (method + '.json')
        row = base.read(path)
        if (row.get('profile') != PROFILE or row.get('method') != method
                or row.get('plan_sha256') != saved['plan_sha256']
                or row.get('poison') != saved['native_records'][method]
                or row.get('checkpoint_saved') is not False
                or row.get('full_training_launched') is not False):
            raise ValueError('Changed two-extra smoke binding')
        validate_smoke(row['native'], method)
        hashes[method] = base.sha(path)
    return dict(schema=1, profile=PROFILE, methods=list(METHODS), policy=POLICY,
        plan_sha256=saved['plan_sha256'], runtime=saved['runtime'], execution=ref['execution'],
        clean_reference=ref['binding'], input_stat_fingerprint=ref['stats'],
        physical_batch=32, fresh_from_epoch0=True, checkpoint_import=False, smokes=hashes,
        code=dict(runner=base.sha(__file__),
            helper=base.sha(HERE / 'Start-PiWMetaFiTwoExtra4090.ps1'),
            orchestration=base.sha(HERE / 'piw_metafi_four_attacks_4090.py'),
            a1_sidecar=base.sha(HERE / 'piw_metafi_a1_4090.py'),
            unchanged_clean=a1.diagnostic().fresh_probe().code_binding(base)))


def replay_evaluation(q, args, base, saved, ref, method):
    import numpy as np
    w = args.scientific_worker
    row = q.completed(args, base, saved, ref, method)
    if row is None:
        return None
    with np.load(args.out / q.CELL / method / 'predictions.npz', allow_pickle=False) as p:
        if (p['names'].tolist() != saved['inputs']['splits']['test_data']['ordered_names']
                or p['truth'].shape != (2586, 14, 3)
                or p['trigger_predictions'].shape != (2586, 6, 14, 3)
                or not np.array_equal(p['doses'], w.DOSES)
                or not np.array_equal(p['targets'], np.stack(
                    [w.target_pose(p['truth'], d) for d in w.DOSES], axis=1))
                or not np.array_equal(p['predictions'], p['trigger_predictions'][:, 0])
                or not np.array_equal(p['reference_predictions'], p['reference_trigger_predictions'][:, 0])):
            raise ValueError('Saved full TEST identities/target/dose schema differs')
        clean_npz = args.clean_run / q.a1_sidecar().CLEAN_CELL / 'predictions.npz'
        with np.load(clean_npz, allow_pickle=False) as clean:
            if not np.array_equal(p['truth'], clean['truth']) or clean['names'].tolist() != p['names'].tolist():
                raise ValueError('Saved TEST differs from external Clean reference')
        metrics = w.measure_predictions(p['truth'], p['predictions'], p['trigger_predictions'],
                                       p['reference_predictions'], p['reference_trigger_predictions'])
    if base.digest(row['metrics']) != base.digest(metrics):
        raise ValueError('Saved metrics do not exactly replay from saved predictions')
    return row


def evaluate(q, args, a1, base, w, saved, ref, method):
    import numpy as np
    import torch
    args.scientific_worker = w
    if replay_evaluation(q, args, base, saved, ref, method) is not None:
        print('EVALUATION_ALREADY_VERIFIED ' + method, flush=True)
        return
    folder = args.out / q.CELL / method
    if (folder / 'predictions.npz').exists():
        raise ValueError('Partial predictions exist; inspect before retry, no overwrite')
    own = q.own_checkpoint(base, w, saved, ref, method, final=True)
    clean_path = args.clean_run / a1.CLEAN_CELL / 'checkpoint.pt'
    clean = torch.load(clean_path, map_location='cpu', weights_only=False)
    base.validate_blob(clean, ref['plan'], 'clean', dict(n_total=27876, n_poison=0, n_cover=0))
    if clean['epoch'] != 199 or clean['execution'] != ref['execution']:
        raise ValueError('External Clean checkpoint execution/final epoch differs')
    ds = w._datasets(saved)
    x, truth, names = w._test_arrays(ds['validation'])
    if len(names) != 2586 or names != saved['inputs']['splits']['test_data']['ordered_names']:
        raise ValueError('All 2586 ordered TEST frames required')
    with np.load(args.clean_run / a1.CLEAN_CELL / 'predictions.npz', allow_pickle=False) as p:
        if p['names'].tolist() != names or not np.array_equal(p['truth'], truth):
            raise ValueError('TEST truth/order differs from completed Clean')
    trigger = w.make_trigger(method, ds['training'], saved['cfg'])
    if method == 'tsba_adapted':
        trigger.load_state_dict(own['trigger'], strict=True)
        trigger.eval().to('cuda:0')
    grids = []
    for role, blob in ((method, own), ('clean_reference', clean)):
        print(f'EVALUATING {role}: {method}, full2586 TEST, six doses', flush=True)
        model = w._build('metafiplusplus', 'cuda:0')
        model.load_state_dict(blob['model'], strict=True)
        grids.append(w._predict_grid(model, x, trigger, w.DOSES, 'cuda:0'))
        del model
    grid, control = grids
    metrics = w.measure_predictions(truth, grid[:, 0], grid, control[:, 0], control)
    # Refuse partial output on manual retry instead of replacing its evidence.
    if (folder / 'predictions.npz').exists():
        raise ValueError('Partial predictions exist; inspect before retry, no overwrite')
    w._atomic_npz(folder / 'predictions.npz', truth=truth, names=np.asarray(names), doses=np.asarray(w.DOSES),
        predictions=grid[:, 0], trigger_predictions=grid, reference_predictions=control[:, 0],
        reference_trigger_predictions=control, targets=np.stack([w.target_pose(truth, d) for d in w.DOSES], axis=1))
    if base.sha(clean_path) != ref['binding']['files']['checkpoint.pt']:
        raise ValueError('External Clean checkpoint changed')
    base.write(folder / 'evaluation.json', dict(profile=PROFILE, status='COMPLETE', method=method,
        backbone='metafiplusplus', seed=42, epoch=199, plan_sha256=saved['plan_sha256'],
        checkpoint_sha256=base.sha(folder / 'checkpoint.pt'), clean_checkpoint_sha256=base.sha(clean_path),
        clean_reference=ref['binding'], predictions_sha256=base.sha(folder / 'predictions.npz'),
        poison=own['poison'], metrics=metrics,
        limitations=['One seed; CSI adaptations, not exact source-paper reproductions.',
            'Smooth-LF is a fixed seeded low-frequency control, not an optimized LF-paper trigger.',
            'TSBA uses victim gradients (white-box training adaptation).',
            'Different native perturbation budgets; no equal-distortion claim.',
            'Historical target/recipe development viewed TEST; external GPU evaluation allowed.']))
    replay_evaluation(q, args, base, saved, ref, method)
    print('EVAL_COMPLETE ' + method + ' ' + json.dumps(metrics['mainmetrics']), flush=True)


def summary(q, args, a1, base, saved, ref):
    rows = [replay_evaluation(q, args, base, saved, ref, m) for m in METHODS]
    table = '# MetaFi++ / Person-in-WiFi-3D / seed42 / two additional attacks\n\n'
    table += '| Method | T-MPJPE mm | Target-PCK % | DCE mm | Clean MPJPE mm | Clean PA mm |\n'
    table += '| --- | ---: | ---: | ---: | ---: | ---: |\n'
    c = ref['evaluation']['clean_metrics']
    table += f"| Clean reference | -- | -- | -- | {c['clean_mpjpe_mm']:.2f} | {c['clean_pampjpe_mm']:.2f} |\n"
    for method, row in zip(METHODS, rows):
        table += q.metric_row(NAMES[method], row['metrics']['mainmetrics']) if row else f'| {NAMES[method]} | pending | -- | -- | -- | -- |\n'
    table += '\nOne seed, not the full comparison table. T/PCK at d=1; DCE averages five positive doses.\n'
    base.write(args.out / 'results.json', dict(profile=PROFILE,
        status='COMPLETE' if all(rows) else 'PARTIAL', records=[r for r in rows if r],
        policy=POLICY, clean_reference=ref['binding'], plan_sha256=saved['plan_sha256']))
    (args.out / 'RESULT.md').write_text(table, encoding='utf-8')
    print(table, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('inspect', 'prepare', 'check', 'launch', 'evaluate', 'status'), default='inspect')
    p.add_argument('--data-home', type=Path, default=Path.home() / 'backdooranalog')
    p.add_argument('--out', type=Path)
    p.add_argument('--method', choices=METHODS)
    args = p.parse_args(argv)
    if args.method and args.mode != 'evaluate':
        p.error('--method only applies to --mode evaluate')
    args.data_home = args.data_home.absolute()
    output_home = Path('U:/backdooranalog') if os.name == 'nt' and Path('U:/').is_dir() else args.data_home
    args.out = (args.out or output_home / 'runs' / PROFILE).absolute()
    args.clean_run = args.data_home / 'runs/piw_metafi_clean_vastenv_s42_4090_v1'
    args.dataset = args.data_home / 'datasets/Person-in-WiFi-3D_processed_v1'
    args.action = args.data_home / 'actions/data_bend.npy'
    args.env_dir = args.data_home / 'envs/piw-metafi-vast211-4090'
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode in ('inspect', 'status'):
        print(json.dumps(dict(profile=PROFILE, policy=POLICY, seed=42, epochs=200, rho=.1,
            batch_size=32, clean_run=str(args.clean_run), out=str(args.out),
            env_dir=str(args.env_dir), dataset=str(args.dataset),
            note='Read only: no torch import, CUDA, training or process-count gate.'), indent=2))
        if args.mode == 'status':
            for m in METHODS:
                print(m + ': ' + ('evaluation exists' if (args.out / 'attacks/metafiplusplus/seed42' / m / 'evaluation.json').is_file() else 'not evaluated'))
            logs = sorted((args.out / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
            if logs:
                from collections import deque
                with logs[-1].open(encoding='utf-8', errors='replace') as f:
                    print('LATEST_LOG=' + str(logs[-1]) + '\n' + ''.join(deque(f, maxlen=15)))
            if (args.out / 'RESULT.md').is_file():
                print((args.out / 'RESULT.md').read_text(encoding='utf-8'))
        return 0
    q = queue_module()
    a1 = q.a1_sidecar()
    probe = a1.diagnostic().source_probe()
    for name in ('data_home', 'out', 'clean_run', 'dataset', 'action', 'env_dir'):
        probe.reject_linked_path(getattr(args, name))
        setattr(args, name, getattr(args, name).resolve())
    if args.out.name != PROFILE:
        raise ValueError('Use a separate parent directory, retaining the dedicated run name')
    for protected in (args.dataset, args.clean_run, args.action, args.env_dir):
        if args.out == protected or args.out in protected.parents or protected in args.out.parents:
            raise ValueError('Output overlaps protected data/reference/environment')
    if Path(sys.prefix).resolve() != args.env_dir or sys.prefix == sys.base_prefix:
        raise ValueError('Use completed Clean4090 environment, no fallback or package install')
    owner = args.env_dir / 'clean_vastenv_setup.owner.json'
    if not owner.is_file() or json.loads(owner.read_text(encoding='utf-8-sig')).get('profile') != probe.SETUP_PROFILE:
        raise ValueError('Clean4090 environment ownership marker missing')
    if any(os.environ.get(k) for k in ('PIW_SMOKE_BARRIER_DIR', 'PYTHONPATH', 'PYTHONHOME')):
        raise ValueError('Inherited smoke barrier/Python overrides refused')
    os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', MKL_THREADING_LAYER='SEQUENTIAL')
    sys.dont_write_bytecode = True
    base = probe.base_module()
    base.configure_torch_cache(args)
    # Only this dedicated output is locked. No global GPU/PID reservation.
    with base.lock(args.out), probe.run_log(args):
        if shutil.disk_usage(args.out).free < 4 * 2**30:
            raise ValueError('At least 4 GiB required on output volume; nothing deleted')
        print('SHARED_GPU_ALLOWED: no PID cap; one owned worker. External jobs may cause OOM/slowdown.', flush=True)
        w = q.worker(args, base)
        cfg = configuration(q, args, w)
        ref = a1.reference(args, base, w, cfg)
        if args.mode == 'evaluate':
            saved = w.load_plan(cfg)
            q.check_gate(args, a1, base, w, saved, ref)
            for method in ((args.method,) if args.method else METHODS):
                evaluate(q, args, a1, base, w, saved, ref, method)
            summary(q, args, a1, base, saved, ref)
        else:
            q.run(args, a1, base, w, cfg, ref)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'[PiW MetaFi two extra 4090] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
