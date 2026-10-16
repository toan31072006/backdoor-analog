"""Frozen MM-Fi T1/T2/T3, seeds 42/43/44, without duplicate A1 training.

Historical seed-42 folders are read-only. This separate launcher keeps all
native attack operators and the full 50-epoch victim recipe unchanged.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

from run_mmfi_tables import _atomic_json, _cell_lock, _check_inputs
from run_method_drafts import _file_sha, _sha_json
from run_proposed_ablation import _configs, EXPECTED_COUNTS, GRID
from run_a1_native_comparison import (
    METHODS as BASELINE_METHODS, STOCHASTIC_CELLS,
    _method_config as _baseline_config,
    _source_provenance as _native_sources,
)

HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_a1_three_tables_v1'
FOLDER = 'mmfi_a1_three_tables_s42_43_44_v1'
MANIFEST = 'three_tables.resolved.json'
SEEDS = (42, 43, 44)
KEYS = ('clean', 'a1', *[key for key, _ in BASELINE_METHODS],
        'a2', 'a3', 'a4', 'rho0p2', 'rho0p4')
LABELS = dict(BASELINE_METHODS, clean='Clean reference', a1='A1--Full',
              a2='A2', a3='A3', a4='A4', rho0p2='A1 rho=0.2', rho0p4='A1 rho=0.4')
ABLATION_KEYS = dict(a1='full', a2='w_o_coupling',
                     a3='w_o_auxiliary', a4='w_o_both')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _source_provenance():
    extra = ('run_a1_three_tables.py', 'a1_three_tables_eval.py',
             'a1_three_tables_report.py', 'eval_a1_target_pck.py',
             'eval/target_pck.py', 'run_a1_dose_poison.py')
    return dict(_native_sources(), **{name: _file_sha(HERE / name) for name in extra})


def _plan_hash(matrix):
    value = copy.deepcopy(matrix)
    for name in ('created_utc', 'plan_sha256'):
        value.pop(name, None)
    for cell in value['cells']:
        for name in ('device', 'num_workers', 'ckpt_every'):
            cell['cfg'].pop(name, None)
        if 'evaluation_trigger_cfg' in cell:
            for name in ('device', 'num_workers', 'ckpt_every'):
                cell['evaluation_trigger_cfg'].pop(name, None)
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def _configuration(data_home, device, workers, seed, key):
    """Change experiment seed only; never retune a baseline operator."""
    from train_backdoor import _resolve_training_config, _validate_training_contract
    if key in dict(BASELINE_METHODS):
        cfg = copy.deepcopy(_baseline_config(data_home, device, workers, key))
        cfg['seed'] = seed
    else:
        configs = _configs(data_home, device, workers, seed, factorial=True)
        cfg = copy.deepcopy(configs[{'a1': 0, 'a2': 1, 'a3': 2, 'a4': 3}.get(key, 0)])
        if key == 'clean':
            cfg['rho'] = 0.0
        elif key in ('rho0p2', 'rho0p4'):
            cfg.update(rho=0.2 if key == 'rho0p2' else 0.4,
                       confirmation_profile='mmfi_a1_dose_poison_v1')
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    return cfg


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4, *,
                 seed42_comparison=None, seed42_ablation=None,
                 seed42_dose_poison=None, seed42_clean=None,
                 retrain_seed42=False):
    """Pure planning: no checkpoint/data reads, training, or CUDA access."""
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    if not isinstance(retrain_seed42, bool):
        raise ValueError('retrain_seed42 must be boolean')
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    runs = data_home / 'runs'
    roots = {name: str(Path(value or default).resolve()) for name, value, default in (
        ('comparison', seed42_comparison, runs / 'mmfi_a1_native_comparison_s42_v3'),
        ('ablation', seed42_ablation, runs / 'mmfi_proposed_factorial_ablation_s42_v1'),
        ('dose_poison', seed42_dose_poison, runs / 'mmfi_a1_dose_poison_s42_v1'),
        ('clean', seed42_clean, runs / 'mmfi_tables_s42_20261009_003044/clean'))}
    cells = []
    for seed in SEEDS:
        a1_cfg = _configuration(data_home, device, num_workers, seed, 'a1')
        for key in KEYS:
            folder = outdir / f'seed{seed}' / key
            reused = seed == 42 and not retrain_seed42
            checkpoint_folder = folder
            if reused:
                if key in ABLATION_KEYS:
                    checkpoint_folder = Path(roots['ablation']) / ABLATION_KEYS[key]
                elif key in dict(BASELINE_METHODS):
                    checkpoint_folder = Path(roots['comparison']) / key
                elif key == 'clean':
                    checkpoint_folder = Path(roots['clean'])
                else:
                    checkpoint_folder = Path(roots['dose_poison']) / key
            tables = (['T1', 'T2', 'T3'] if key == 'a1' else
                      ['T3'] if key in ('a2', 'a3', 'a4') else
                      ['T2'] if key.startswith('rho') else ['T1'])
            cell = dict(method_key=f's{seed}__{key}', key=key, seed=seed,
                label=LABELS[key], tables=tables, dependencies=[],
                cfg=_configuration(data_home, device, num_workers, seed, key),
                ckpt_dir=str(checkpoint_folder), eval_cache=str(checkpoint_folder / 'eval_cache.json'),
                run_dir=str(folder), artifact_dir=str(outdir / 'evaluations' / f's{seed}__{key}'),
                origin='reused_seed42' if reused else 'fresh')
            if key == 'clean':
                cell['evaluation_trigger_cfg'] = copy.deepcopy(a1_cfg)
            cells.append(cell)
    matrix = dict(schema=1, profile=PROFILE, dataset='mmfi', seeds=list(SEEDS),
        data_home=str(data_home), outdir=str(outdir), num_workers=num_workers,
        retrain_seed42=retrain_seed42, seed42_roots=roots,
        created_utc=datetime.now(timezone.utc).isoformat(), cells=cells,
        sources=_source_provenance(), metrics_contract=dict(
            tables=['T1', 'T2', 'T3'], full_split_counts=dict(EXPECTED_COUNTS),
            epochs=50, model='hpeli', mmfi_protocol='protocol1', mmfi_setting='s1',
            mmfi_split_seed=0, training='ordinary ERM; SGD lr=.001 momentum=.9 batch32',
            target_joints=[2, 3], target_pck_threshold=.2,
            target_pck_scale='norm(clean_true_pose[5]-clean_true_pose[12])',
            comparison_dose=1., dose_grid=list(GRID), poison_rates=[.1, .2, .4],
            training_doses='paired U(.2,1), except A2/A4 payload-dose permutation',
            dce='mean target-joint vector residual (pred_d-pred_0)-(target_d-true_0)',
            mean_dce='arithmetic mean of five positive doses within each seed',
            uncertainty='mean and sample SD(ddof=1) of independent seed-level metrics',
            native_operators=True, matches_realized_noise_budget=False,
            trigger_seed_policy='Original pattern follows experiment seed; auxiliary carrier RNG remains 42',
            independent_victims_per_seed=13, a1_shared_across_tables=True,
            no_additional_experiments=['T4', 'T5', 'T6', 'T7']))
    matrix['plan_sha256'] = _plan_hash(matrix)
    return matrix


def _validate_manifest(matrix):
    if (matrix.get('schema') != 1 or matrix.get('profile') != PROFILE
            or matrix.get('seeds') != list(SEEDS)
            or matrix.get('plan_sha256') != _plan_hash(matrix)):
        raise ValueError('Invalid or modified three-table manifest')
    roots = matrix['seed42_roots']
    canonical = build_matrix(matrix['data_home'], matrix['outdir'],
        matrix['cells'][0]['cfg']['device'], matrix['num_workers'],
        seed42_comparison=roots['comparison'], seed42_ablation=roots['ablation'],
        seed42_dose_poison=roots['dose_poison'], seed42_clean=roots['clean'],
        retrain_seed42=matrix['retrain_seed42'])
    if matrix['plan_sha256'] != canonical['plan_sha256']:
        raise ValueError('Scientific settings/source changed; use a NEW output directory')
    for cell in matrix['cells']:
        cfg = _configuration(Path(matrix['data_home']), cell['cfg']['device'],
            0 if cell['key'] in STOCHASTIC_CELLS else matrix['num_workers'],
            cell['seed'], cell['key'])
        if cfg != cell['cfg']:
            raise ValueError(f"{cell['method_key']}: canonical config mismatch")


def _read_completed(cell):
    """Only the historical clean run may carry pre-A1 attack metadata."""
    from eval_a1_target_pck import _read_run, _validate_clean
    from train_backdoor import _config_fingerprint
    run = _read_run(cell['ckpt_dir'])
    expected = cell['cfg']
    if cell['origin'] == 'reused_seed42' and cell['key'] == 'clean':
        _validate_clean(run['cfg'], expected, run['res'])
    elif _config_fingerprint(run['cfg']) != _config_fingerprint(expected):
        raise ValueError(f"{cell['method_key']}: checkpoint recipe differs from frozen plan")
    return run


def _current_inputs(cell):
    from run_peak_confirmation import _collect_input_record
    # This helper only indexes full split identities; no victim inference.
    record = _collect_input_record(cell)
    record['profile'] = PROFILE
    if cell['key'] == 'fiba':
        from attack.frequency_baselines import build_frequency_trigger
        trigger = build_frequency_trigger('fiba', dict(cell['cfg'], device='cpu', num_workers=0))
        record['fiba_fixed_key_sha256'] = trigger.fixed_key_sha256()
        record['fiba_reference_sha256'] = hashlib.sha256(trigger.reference.astype('<f4').tobytes()).hexdigest()
        record['fiba_reference_train_index'] = 0
    return record


def _check_record(record, current, fingerprint, key):
    if (record.get('schema') != 1 or record.get('config_fingerprint') != fingerprint
            or any(record.get(name) != current[name] for name in ('train', 'eval', 'action_file_sha256'))):
        raise ValueError(f'{key}: ordered split/action/config input audit changed')
    if key == 'fiba' and any(record.get(name) != current[name] for name in
            ('fiba_fixed_key_sha256', 'fiba_reference_sha256', 'fiba_reference_train_index')):
        raise ValueError('FIBA TRAIN-only reference changed')


def _audit_poison(cfg, res, poison):
    """Rebuild the exact seeded sampler and native cover identities."""
    import numpy as np
    from train_backdoor import _config_fingerprint
    n, seed, rho = EXPECTED_COUNTS['train'], cfg['seed'], cfg['rho']
    count = math.floor(n * rho)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=count, replace=False)
    doses = rng.uniform(.2, 1., size=count)
    expected = [[int(i), float(d)] for i, d in zip(indices, doses)]
    payload = doses.copy()
    if cfg.get('dose_coupling', 'paired') == 'shuffled':
        np.random.default_rng(np.random.SeedSequence([seed, 7043])).shuffle(payload)
    samples = poison.get('samples', [])
    if (not isinstance(samples, list)
            or [[s.get('index'), s.get('dose')] for s in samples] != expected
            or [s.get('payload_dose', s.get('dose')) for s in samples] != payload.tolist()):
        raise ValueError('Poison indices/dose draws/payload permutation differ from frozen sampler')
    digest = _sha_json(expected)
    required = dict(config_fingerprint=_config_fingerprint(cfg), n_total=n,
        n_poison=count, seed=seed, selection='uniform', dose_min=.2, dose_max=1.,
        rho_requested=rho, pivot=1, target_joints=[2, 3], theta_max_deg=40.,
        payload_axis=[0., 0., 1.], dose_mode='linear', poison_plan_sha256=digest)
    if any(poison.get(k) != v for k, v in required.items()):
        raise ValueError('Poison manifest differs from frozen scientific protocol')
    cover_ratio = cfg.get('clean_label_cover_ratio', cfg.get('wanet_cover_ratio', 0.))
    n_cover = math.floor(n * cover_ratio)
    expected_cover = []
    if n_cover:
        available = np.array(sorted(set(range(n)) - set(map(int, indices))))
        cover_rng = np.random.default_rng(np.random.SeedSequence([seed, 7044]))
        expected_cover = sorted(map(int, cover_rng.choice(available, n_cover, replace=False)))
    if (poison.get('n_cover', 0) != n_cover or poison.get('cover_ratio', 0.) != cover_ratio
            or poison.get('cover_indices', []) != expected_cover
            or poison.get('dose_coupling', 'paired') != cfg.get('dose_coupling', 'paired')):
        raise ValueError('Native clean-label cover/coupling audit differs')
    if (res.get('n_total') != n or res.get('n_poison') != count
            or res.get('n_cover', 0) != n_cover or res.get('poison_plan_sha256') != digest
            or res.get('training_contract') != 'ordinary_erm'
            or res.get('attacker_access') != 'data_only'):
        raise ValueError('Cached victim counts/training/poison audit differs')
    return dict(poison_plan_sha256=digest, n_poison=count, n_cover=n_cover)


def _audit_run(cell, run, current):
    from train_backdoor import _config_fingerprint
    folder = Path(cell['ckpt_dir'])
    names = ('three_tables_inputs.json', 'comparison_inputs.json', 'ablation_inputs.json')
    records = [name for name in names if (folder / name).is_file()]
    if not records and not (cell['origin'] == 'reused_seed42' and cell['key'] == 'clean'):
        raise ValueError(f"{cell['method_key']}: missing training input audit")
    for name in records:
        _check_record(_json(folder / name), current, _config_fingerprint(run['cfg']), cell['key'])
    poison = _audit_poison(run['cfg'], run['res'], _json(folder / 'poison_manifest.json'))
    return dict(inputs=current, **poison,
                legacy_clean_without_training_pair_hash=not bool(records))


def _preflight(matrix, devices):
    """Fail before any new training if an immutable reference is unavailable."""
    from a1_three_tables_eval import _checkpoint_metadata
    try:
        _check_inputs(matrix, devices)
    except AssertionError as exc:
        raise ValueError('Invalid logical CUDA ID. After CUDA_VISIBLE_DEVICES=0,2,3 '
                         'use --devices cuda:0 cuda:1 cuda:2, not cuda:3.') from exc
    reference = _current_inputs(matrix['cells'][0])
    for cell in matrix['cells']:
        if cell['origin'] != 'reused_seed42':
            continue
        run = _read_completed(cell)
        _checkpoint_metadata(run)
        current = (dict(reference, config_fingerprint=run['fingerprint'])
                   if cell['key'] != 'fiba' else _current_inputs(cell))
        _audit_run(cell, run, current)
    print('[three-tables] preflight passed: full splits, frozen recipes and seed-42 references', flush=True)


def _execute_cell(cell):
    from train_backdoor import train
    from a1_three_tables_eval import evaluate_cell
    output = Path(cell['run_dir'])
    with _cell_lock(output):
        current = _current_inputs(cell)
        if cell['origin'] == 'fresh':
            path = Path(cell['ckpt_dir']) / 'three_tables_inputs.json'
            if path.exists():
                if _json(path) != current:
                    raise ValueError('Fresh-cell input identity changed; refusing resume')
            elif any((Path(cell['ckpt_dir']) / name).exists() for name in
                     ('checkpoint.pt', 'eval_cache.json')):
                raise ValueError('Fresh cell has weights/cache without its input audit')
            else:
                _atomic_json(path, current)
            model, _ = train(cell['cfg'], ckpt_dir=cell['ckpt_dir'])
            del model
            if _current_inputs(cell) != current:
                raise ValueError('Input files changed during victim training/evaluation')
        run = _read_completed(cell)
        audit = _audit_run(cell, run, current)
        evaluating = copy.deepcopy(cell)
        if cell['origin'] == 'reused_seed42' and cell['key'] == 'clean':
            evaluating['cfg'] = run['cfg']
        evaluating['verified_inputs'] = current
        result = evaluate_cell(evaluating, cell['artifact_dir'], cell['cfg']['device'])
        _atomic_json(output / 'scientific_audit.json', audit)
        print(f"[three-tables] completed {cell['method_key']} ({cell['origin']})", flush=True)
        return result


def _schedule(matrix, manifest, devices, requested=None):
    all_cells = {c['method_key']: c for c in matrix['cells']}
    chosen = list(requested or all_cells)
    if len(set(chosen)) != len(chosen) or set(chosen) - set(all_cells):
        raise ValueError('Unknown or duplicate requested cell IDs')
    pending = [cell for cell in matrix['cells'] if cell['method_key'] in chosen]
    running = {}
    try:
        while pending or running:
            for device in devices:
                if device in running or not pending:
                    continue
                cell = pending.pop(0)
                cell['cfg']['device'] = device
                _atomic_json(manifest, matrix)
                folder = Path(cell['run_dir'])
                folder.mkdir(parents=True, exist_ok=True)
                log = folder / 'console.log'
                stream = log.open('a', encoding='utf-8')
                env = os.environ.copy()
                env.setdefault('OMP_NUM_THREADS', '4')
                env.setdefault('MKL_NUM_THREADS', '4')
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--_cell', cell['method_key'], '--_manifest', str(manifest)],
                    cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
                running[device] = (child, cell, stream)
                print(f"[three-tables] {cell['method_key']} -> {device}, PID={child.pid}\n       log: {log}", flush=True)
            for device, (child, cell, stream) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                del running[device]
                if code:
                    raise RuntimeError(f"{cell['method_key']} failed (exit {code}); read {cell['run_dir']}/console.log")
                if not (Path(cell['artifact_dir']) / 'evaluation.json').is_file():
                    raise RuntimeError(f"{cell['method_key']} exited without evaluated metrics")
                print(f"[three-tables] finished {cell['method_key']}", flush=True)
            if running:
                time.sleep(1)
    except BaseException:
        # Clean up only this launcher's children, never unrelated GPU users.
        for child, _, stream in running.values():
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            stream.close()
        raise


def _export(matrix):
    from a1_three_tables_eval import load_evaluation
    from a1_three_tables_report import export_reports
    records = []
    identity = None
    for cell in matrix['cells']:
        run = _read_completed(cell)
        current = _current_inputs(cell)
        audit = _audit_run(cell, run, current)
        shared = (current['train'], current['eval'], current['action_file_sha256'])
        if identity is not None and shared != identity:
            raise ValueError('Cells do not share the same full split/action identities')
        identity = shared
        evaluating = copy.deepcopy(cell)
        if cell['origin'] == 'reused_seed42' and cell['key'] == 'clean':
            evaluating['cfg'] = run['cfg']
        evaluating['verified_inputs'] = current
        record = load_evaluation(evaluating, cell['artifact_dir'])
        if record is None:
            raise ValueError(f"Missing/stale evaluation: {cell['method_key']}; rerun this cell")
        record['scientific_audit'] = audit
        records.append(record)
    return export_reports(records, matrix['outdir'], plan_sha256=matrix['plan_sha256'],
        provenance=dict(profile=PROFILE, sources=matrix['sources'], seeds=list(SEEDS),
                        reused_seed42=not matrix['retrain_seed42']))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--num-workers', type=int, default=4)
    for name in ('comparison', 'ablation', 'dose-poison', 'clean'):
        parser.add_argument('--seed42-' + name, type=Path)
    parser.add_argument('--retrain-seed42', action='store_true')
    parser.add_argument('--cells', nargs='+')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--_cell', help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args._cell:
        if args._manifest is None:
            raise ValueError('Worker requires --_manifest')
        matrix = _json(args._manifest)
        _validate_manifest(matrix)
        cells = [c for c in matrix['cells'] if c['method_key'] == args._cell]
        if len(cells) != 1:
            raise ValueError('Worker cell is not in the frozen plan')
        _execute_cell(cells[0])
        return 0
    if args.data_home is None:
        raise ValueError('--data-home is required')
    if (not args.devices or len(set(args.devices)) != len(args.devices)
            or any(d != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None for d in args.devices)):
        raise ValueError('--devices requires distinct logical cuda:N IDs (after CUDA_VISIBLE_DEVICES), or cpu')
    if sum((args.dry_run, args.export_only, args.status)) > 1:
        raise ValueError('Choose only one of --dry-run, --export-only or --status')
    outdir = (args.outdir or args.data_home / 'runs' / FOLDER).resolve()
    if args.status:
        matrix = _json(outdir / MANIFEST)
        total = 0
        for seed in SEEDS:
            cells = [c for c in matrix['cells'] if c['seed'] == seed]
            done = [c['key'] for c in cells if (Path(c['artifact_dir']) / 'evaluation.json').is_file()]
            total += len(done)
            print(f"seed {seed}: {len(done)}/{len(cells)} evaluation artifacts; {', '.join(done)}")
        print(f'Total: {total}/39 evaluation artifacts')
        print('Artifact counts are not fresh verification; --export-only performs the complete audit.')
        return 0
    if args.fresh and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('--fresh refuses a nonempty directory; choose a NEW path')
    requested = build_matrix(args.data_home, outdir, args.devices[0], args.num_workers,
        seed42_comparison=args.seed42_comparison, seed42_ablation=args.seed42_ablation,
        seed42_dose_poison=args.seed42_dose_poison, seed42_clean=args.seed42_clean,
        retrain_seed42=args.retrain_seed42)
    for cell in requested['cells']:
        if cell['origin'] == 'reused_seed42':
            reference = Path(cell['ckpt_dir'])
            if outdir == reference or outdir in reference.parents or reference in outdir.parents:
                raise ValueError('New output must not overlap historical checkpoint folders')
    if args.cells and (len(set(args.cells)) != len(args.cells)
                       or set(args.cells) - {c['method_key'] for c in requested['cells']}):
        raise ValueError('Unknown/duplicate --cells IDs; use s43__badnets style IDs')
    with _cell_lock(outdir):
        manifest = outdir / MANIFEST
        if manifest.exists():
            matrix = _json(manifest)
            _validate_manifest(matrix)
            if matrix['plan_sha256'] != requested['plan_sha256']:
                raise ValueError('Requested plan differs from existing output; use a NEW directory')
            for cell in matrix['cells']:
                cell['cfg']['device'] = args.devices[0]
        else:
            if any(p.name != 'run.lock' for p in outdir.iterdir()):
                raise ValueError('Output contains unrelated artifacts; use a NEW directory')
            matrix = requested
        _validate_manifest(matrix)
        _atomic_json(manifest, matrix)
        if args.dry_run:
            fresh = sum(c['origin'] == 'fresh' for c in matrix['cells'])
            print(f'[three-tables] DRY RUN: seeds42/43/44; 39 scientific victims, '
                  f'{fresh} new trainings, {39-fresh} immutable references; '
                  f'NO data/GPU/training.\n{manifest}', flush=True)
            return 0
        if not args.export_only:
            _preflight(matrix, args.devices)
            _schedule(matrix, manifest, args.devices, args.cells)
        missing = [c['method_key'] for c in matrix['cells']
                   if not (Path(c['artifact_dir']) / 'evaluation.json').is_file()]
        if missing:
            if args.cells and not args.export_only:
                print(f'[three-tables] selected cells finished; complete tables await: {missing}', flush=True)
                return 0
            raise ValueError(f'All 39 cells are required; missing: {missing}')
        _export(matrix)
        print(f'[three-tables] COMPLETE: {outdir / "three_main_tables.json"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[three-tables] ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
