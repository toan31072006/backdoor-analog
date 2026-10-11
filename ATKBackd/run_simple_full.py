"""Full MM-Fi / 50-epoch tests of four minimal, fixed Proposed hypotheses.

Every arm is a fresh ordinary-ERM victim. No screening subsets, attacker
surrogates, learned guards, test-driven parameter search or imported results.
The full TEST scores are exploratory, not a blinded superiority claim.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

from simple_full_contract import (PROFILE, STATUS, GRID, EXPECTED_COUNTS, FIXED,
    validate_simple_full_config, validate_simple_full_bound_files, source_provenance)
from run_mmfi_tables import _atomic_json, _cell_lock, _check_inputs, _plan_fingerprint, run_matrix
from run_method_drafts import _file_sha, _sha_json, _verified_cache, _write_csv
from run_peak_confirmation import _pair_ids

HERE = Path(__file__).resolve().parent
METHODS = (
    ('clean', 'Clean fresh victim (no trigger)', 'micro_dropper', None),
    ('blended', 'Unchanged Blended-CSI (dual budget)', 'blended', None),
    ('proposed', 'Fixed equal-weight multi-carrier (dual budget)', 'md_multicarrier_peak_matched', None),
    ('additive', 'Proposed: additive carrier', 'simple_carrier', 'additive'),
    ('single', 'Proposed: one auxiliary carrier', 'simple_carrier', 'single'),
    ('shared_links', 'Proposed: antenna-shared pattern', 'simple_carrier', 'shared_links'),
    ('frequency_only', 'Proposed: auxiliary temporal DCT mode 0', 'simple_carrier', 'frequency_only'),
)
ERROR_METRICS = ('t1_mpjpe_mm', 'clean_mpjpe_mm', 'clean_pampjpe_mm')
PCK_METRICS = tuple(f'clean_pck_{t:.1f}_pct' for t in (.5, .4, .3, .2, .1))
MAIN_METRICS = ERROR_METRICS + PCK_METRICS
RUNTIME_KEYS = {'device', 'num_workers', 'ckpt_every'}


def _source_provenance():
    return source_provenance()


def _cfg(data_home, device, workers, key, trigger, variant):
    from run_traditional_comparison import _base_config
    from train_backdoor import _resolve_training_config, _validate_training_contract
    cfg = _base_config(data_home, device, workers, budget_mode='native')
    cfg.pop('traditional_comparison_profile')
    cfg.update(copy.deepcopy(FIXED), trigger=trigger, rho=0.0 if key == 'clean' else .1,
               eps=.2 if key == 'blended' else .185)
    if key == 'proposed':
        cfg.update(method_carrier_sub_mode=3, method_carrier_time_mode=1, method_carrier_seed=42)
    if variant:
        cfg['simple_variant'] = variant
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    return cfg


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4, *, distortion_samples=256):
    """Inert plan: all seven victims are locked to full splits and 50 epochs."""
    for name, value, minimum in (('num_workers', num_workers, 0), ('distortion_samples', distortion_samples, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    cells = []
    for key, label, trigger, variant in METHODS:
        folder = outdir / key
        cells.append(dict(method_key=key, label=label,
            group='control' if variant is None else 'one_factor_proposed', tables=[], dependencies=[],
            cfg=_cfg(data_home, device, num_workers, key, trigger, variant),
            ckpt_dir=str(folder), eval_cache=str(folder / 'eval_cache.json')))
    matrix = dict(schema=1, dataset='mmfi', seed=42, fresh_results_only=True,
        simple_full_profile=PROFILE, status=STATUS,
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources=dict(source_sha256=_source_provenance(),
            implementation='Fixed one-factor CSI hypotheses; not source-paper reproductions'),
        cells=cells, metrics_contract=dict(profile=PROFILE, epochs=50, rho=.1,
            full_split_counts=copy.deepcopy(EXPECTED_COUNTS), official_test_used=True,
            evaluation_role='full_official_test_exploratory', dose_grid=copy.deepcopy(GRID),
            main_metrics=list(MAIN_METRICS), main_asr=False,
            pck='relative .5/.4/.3/.2/.1, NOT millimetres',
            poison='identical fixed uniform IDs and paired U(.2,1) doses in all six attack arms',
            peak_ceiling='Original realized per-input/per-dose peak at eps=.185',
            relative_l2_ceiling='.10*d per input; upper bounds, not equal realized noise',
            victim='fresh seed42, no model weight transfer; ordinary full-pose MPJPE ERM',
            fixed_control='equal-weight multi-carrier, NOT the historical learned paired-guard artifact',
            clean_target_metrics='unmodified clean input versus each dose-specific target; Original-trigger probe remains in raw cache only',
            candidates='four independent changes relative to fixed equal-weight multi-carrier',
            attacker_fitting=False, selection='no automatic test-driven parameter or candidate selection',
            distortion_samples=distortion_samples, uncertainty='single seed; not estimated'))
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    return matrix


def _validate_manifest(matrix):
    if (matrix.get('schema') != 1 or matrix.get('simple_full_profile') != PROFILE
            or matrix.get('status') != STATUS or matrix.get('dataset') != 'mmfi'
            or matrix.get('seed') != 42 or matrix.get('fresh_results_only') is not True
            or matrix.get('DRAFT_ONLY') or 'draft_profile' in matrix):
        raise ValueError('Not the separate full-data minimal Proposed plan')
    if matrix.get('plan_sha256') != _plan_fingerprint(matrix):
        raise ValueError('Full plan fingerprint differs from its contents')
    if matrix['sources']['source_sha256'] != _source_provenance():
        raise ValueError('Scientific source changed; use a NEW output directory')
    if [c['method_key'] for c in matrix['cells']] != [m[0] for m in METHODS]:
        raise ValueError('All seven planned rows must remain, including unsuccessful hypotheses')
    first = matrix['cells'][0]
    root = Path(first['ckpt_dir']).parent
    data_home = Path(first['cfg']['dataset_root']).parent.parent
    canonical = build_matrix(data_home, root, first['cfg']['device'], first['cfg']['num_workers'],
        distortion_samples=matrix['metrics_contract']['distortion_samples'])
    if matrix['sources'] != canonical['sources'] or matrix['metrics_contract'] != canonical['metrics_contract']:
        raise ValueError('Full metric/source contract changed')
    for cell, expected in zip(matrix['cells'], canonical['cells']):
        validate_simple_full_config(cell['cfg'])
        if ({k: v for k, v in cell['cfg'].items() if k not in RUNTIME_KEYS}
                != {k: v for k, v in expected['cfg'].items() if k not in RUNTIME_KEYS}
                or {k: v for k, v in cell.items() if k != 'cfg'}
                != {k: v for k, v in expected.items() if k != 'cfg'}):
            raise ValueError(f"{cell['method_key']}: fixed arm or scientific configuration changed")


def _collect_input_record(cell):
    from train_backdoor import _load_dataset, _config_fingerprint, build_trigger
    cfg = cell['cfg']
    validate_simple_full_config(cfg)
    record = dict(schema=1, profile=PROFILE, config_fingerprint=_config_fingerprint(cfg),
        scientific_source_sha256=_source_provenance(),
        action_file_sha256=_file_sha(cfg['action_npy']),
        trigger_key_sha256=build_trigger(cfg).fixed_key_sha256())
    sets = []
    for name, split in (('train', 'training'), ('eval', 'test')):
        ds = _load_dataset(cfg, split)
        ids = _pair_ids(ds)
        if len(ds) != EXPECTED_COUNTS[name] or len(ids) != len(ds):
            raise ValueError(f'Full {name} requires {EXPECTED_COUNTS[name]} pairs, got {len(ds)}')
        identities = {tuple(i) for i in ids}
        if len(identities) != len(ids):
            raise ValueError(f'Duplicate full {name} pairs')
        sets.append(identities)
        record[name] = dict(n=len(ids), pair_ids_sha256=_sha_json(ids))
    if sets[0].intersection(sets[1]):
        raise ValueError('Official TRAIN/TEST overlap')
    return record


def _bind_inputs(cell, record):
    folder = Path(cell['ckpt_dir'])
    path = folder / 'simple_inputs.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('Full input/action/trigger identity changed; use a NEW directory')
    else:
        if any((folder / n).exists() for n in ('checkpoint.pt', 'eval_cache.json')):
            raise ValueError('Missing input passport; no historical-result import')
        _atomic_json(path, record)


def _run_cell(cell):
    from train_backdoor import train, _config_fingerprint
    with _cell_lock(cell['ckpt_dir']):
        if Path(cell['eval_cache']).exists():
            _complete(cell)  # Never silently repair/import an edited completed result.
        record = _collect_input_record(cell)
        _bind_inputs(cell, record)
        model, res = train(cell['cfg'], ckpt_dir=cell['ckpt_dir'])
        del model
        if _collect_input_record(cell) != record:
            raise ValueError('Full inputs changed during training/evaluation')
        _atomic_json(Path(cell['ckpt_dir']) / 'results.json', dict(
            method_key=cell['method_key'], cfg=cell['cfg'], res=res,
            cfg_fingerprint=_config_fingerprint(cell['cfg'])))
        print(f"[simple-full] completed {cell['method_key']}", flush=True)


def _complete(cell):
    validate_simple_full_config(cell['cfg'])
    if not Path(cell['eval_cache']).exists():
        return False
    res, _ = _verified_cache(cell)
    from mmfi_tables import metric_row, dose_rows
    metric_row(cell, res)  # Reject incomplete/nonfinite primary metrics too.
    series, _ = dose_rows(cell, res)
    if [row['d'] for row in series] != GRID:
        raise ValueError('Completed cache is missing the exact six-dose evaluation')
    validate_simple_full_bound_files(cell['cfg'], cell['ckpt_dir'])
    path = Path(cell['ckpt_dir']) / 'simple_inputs.json'
    if not path.exists():
        raise ValueError('Completed cache has no full input passport')
    from mmfi_tables import config_fingerprint
    record = json.loads(path.read_text(encoding='utf-8'))
    if (record.get('config_fingerprint') != config_fingerprint(cell['cfg'])
            or record.get('action_file_sha256') != _file_sha(cell['cfg']['action_npy'])):
        raise ValueError('Completed cache input/config passport mismatch')
    return True


def audit_common_inputs(matrix):
    from mmfi_tables import config_fingerprint
    records, plans, common_identity, common_plan = {}, {}, None, None
    for cell in matrix['cells']:
        key, cfg = cell['method_key'], cell['cfg']
        record = json.loads((Path(cell['ckpt_dir']) / 'simple_inputs.json').read_text(encoding='utf-8'))
        if record != _collect_input_record(cell):
            raise ValueError(f'{key}: current full inputs differ from the training passport')
        identity = (record['train'], record['eval'], record['action_file_sha256'])
        if common_identity is not None and identity != common_identity:
            raise ValueError('Full arms used different ordered splits/action bytes')
        common_identity, records[key] = identity, record
        poison = json.loads((Path(cell['ckpt_dir']) / 'poison_manifest.json').read_text(encoding='utf-8'))
        samples = poison.get('samples')
        if not isinstance(samples, list):
            raise ValueError(f'{key}: missing poison identities')
        plan = []
        for sample in samples:
            i, d = sample.get('index'), sample.get('dose')
            if (isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < EXPECTED_COUNTS['train']
                    or isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d)
                    or not .2 <= d <= 1.0 or sample.get('payload_dose', d) != d):
                raise ValueError(f'{key}: invalid paired poison sample')
            plan.append([i, d])
        digest, count = _sha_json(plan), int(math.floor(cfg['rho'] * EXPECTED_COUNTS['train']))
        required = dict(config_fingerprint=config_fingerprint(cfg), n_total=EXPECTED_COUNTS['train'],
            n_poison=count, rho_requested=cfg['rho'], seed=42, selection='uniform',
            dose_min=.2, dose_max=1.0, poison_plan_sha256=digest,
            pivot=1, target_joints=[2, 3], theta_max_deg=40.0,
            payload_axis=[0.0, 0.0, 1.0], dose_mode='linear')
        if (any(poison.get(k) != v for k, v in required.items()) or len(plan) != count
                or len({p[0] for p in plan}) != count or poison.get('n_cover', 0) != 0
                or poison.get('cover_indices', []) != []):
            raise ValueError(f'{key}: poison passport disagrees with full protocol')
        if key != 'clean':
            if common_plan is not None and samples != common_plan:
                raise ValueError('Attack arms need identical poison identities AND paired doses')
            common_plan = samples
        plans[key] = digest
    return dict(inputs=records, poison_plan_sha256=plans,
                action_file_sha256=common_identity[2])


def build_summary(matrix):
    from mmfi_tables import metric_row, dose_rows
    _validate_manifest(matrix)
    audit = audit_common_inputs(matrix)
    rows, doses = [], []
    for cell in matrix['cells']:
        key, cfg = cell['method_key'], cell['cfg']
        res, source = _verified_cache(cell)
        if (res.get('training_contract') != 'ordinary_erm' or res.get('attacker_access') != 'data_only'
                or res.get('dose_coupling') != 'paired' or res.get('dose_grid') != GRID
                or res.get('n_total') != EXPECTED_COUNTS['train']
                or res.get('n_poison') != int(math.floor(cfg['rho'] * EXPECTED_COUNTS['train']))
                or res.get('n_cover', 0) != 0
                or res.get('poison_plan_sha256') != audit['poison_plan_sha256'][key]):
            raise ValueError(f'{key}: cache disagrees with full split/poison protocol')
        # The trainer retains an extra Original-on-clean negative-control
        # probe. The displayed clean row instead uses genuinely unmodified
        # input, matching the clean distortion control and its table label.
        report_res = dict(res, tmpjpe=res['clean_to_target_tmpjpe']) if key == 'clean' else res
        metric = metric_row(cell, report_res)
        row = dict(status=STATUS, method_key=key, method=cell['label'], seed=42,
            epochs=50, rho=cfg['rho'], train_samples=EXPECTED_COUNTS['train'],
            eval_samples=EXPECTED_COUNTS['eval'], t1_mpjpe_mm=metric['tmpjpe_mm'],
            **{k: metric[k] for k in MAIN_METRICS if k != 't1_mpjpe_mm'},
            cfg_fingerprint=source['cfg_fingerprint'])
        rows.append(row)
        series, _ = dose_rows(cell, report_res)
        if [r['d'] for r in series] != GRID:
            raise ValueError('All six dose evaluations must be retained')
        doses.extend(dict(status=STATUS, **entry) for entry in series)
    by_key = {row['method_key']: row for row in rows}
    comparisons = []
    for reference in ('proposed', 'blended'):
        base = by_key[reference]
        for key, _, _, variant in METHODS:
            if variant is None:
                continue
            row = by_key[key]
            comparisons.append(dict(status=STATUS, method_key=key, reference=reference,
                lower_t1=row['t1_mpjpe_mm'] < base['t1_mpjpe_mm'],
                no_worse_all_clean_metrics=all(row[k] <= base[k] for k in ERROR_METRICS[1:])
                    and all(row[k] >= base[k] for k in PCK_METRICS),
                **{f'delta_{k}': row[k] - base[k] for k in MAIN_METRICS}))
    return dict(status='simple_full_complete', result_label=STATUS, profile=PROFILE,
        plan_sha256=matrix['plan_sha256'], rows=rows, dose_response=doses,
        comparisons=comparisons, metrics_contract=matrix['metrics_contract'],
        sources=matrix['sources'], audit=audit, limitations=[
            'Full official TRAIN and TEST; all seven independent victims run 50 epochs.',
            'One seed, repeated benchmark development; not a blinded first test or statistical superiority.',
            'Fixed controls and all unsuccessful hypotheses retained; no historical result import.',
            'No surrogate, trigger optimizer, utility guard, poison re-selection or victim-weight transfer.',
            'Fixed equal-weight control is not the historical learned paired-guard artifact.',
            'Clean T metrics use unmodified input; extra Original-trigger probe remains only in the raw clean cache.',
            'The unchanged fixed control has a native peak projection before the shared dual projection; other arms project once (float32 rounding may differ).',
            'Common digital peak and relative-L2 ceilings do not equalize realized perturbation.',
            'Digital distortion does not establish RF stealthiness or over-the-air feasibility.',
            'Input hashes cover ordered file/frame IDs and action bytes, not all dataset contents.'])


def build_distortion(matrix, audit):
    import numpy as np
    from eval.distortion import distortion_stats, _aggregate
    from train_backdoor import _load_dataset, build_trigger
    cfg = matrix['cells'][0]['cfg']
    ds = _load_dataset(cfg, 'test')
    indices = np.linspace(0, len(ds) - 1, min(matrix['metrics_contract']['distortion_samples'], len(ds))).astype(int)
    all_ids = _pair_ids(ds)
    ids = [all_ids[i] for i in indices]
    # Load each sampled CSI once; every method sees the identical normalized input.
    values = [ds.normalize(ds.load_raw(ds.items[i]['csi'])) for i in indices]
    rows, pairs = [], []
    for cell in matrix['cells']:
        trigger = build_trigger(cell['cfg'])
        key_sha = trigger.fixed_key_sha256()
        if key_sha != audit['inputs'][cell['method_key']]['trigger_key_sha256']:
            raise ValueError('Frozen fixed trigger key changed')
        for dose in GRID:
            measurements, records = [], []
            for index, value in zip(indices, values):
                if cell['method_key'] == 'clean':
                    out, budget = value.copy(), None
                else:
                    out, budget = trigger.inject_with_audit(value, dose, eps=cell['cfg']['eps'])
                    if budget['violation'] or budget['l2_violation']:
                        raise ValueError('Actual digital budget violation')
                measurements.append(distortion_stats(value, out))
                records.append(dict(index=int(index), **({} if budget is None else budget)))
            pooled = _aggregate(measurements)
            rows.append(dict(status=STATUS, method_key=cell['method_key'], dose=dose,
                n_samples=len(values), relative_l2=pooled['relative_l2'], rmse=pooled['rms'],
                linf=pooled['max_abs'], peak_violations=0, l2_violations=0,
                trigger_key_sha256=key_sha, common_pair_ids_sha256=_sha_json(ids)))
            pairs.append(dict(method_key=cell['method_key'], dose=dose, pairs=records))
        if trigger.fixed_key_sha256() != key_sha:
            raise ValueError('Fixed trigger changed during measurement')
    return dict(status=STATUS, profile=PROFILE, plan_sha256=matrix['plan_sha256'], hpe_forward_used=False,
                common_pair_ids=ids, rows=rows, per_sample_budget_audit=pairs)


def export_summary(matrix, outdir):
    report = build_summary(matrix)
    distortion = build_distortion(matrix, report['audit'])
    outdir = Path(outdir)
    for name, rows in (('main_metrics', report['rows']), ('dose_response', report['dose_response']),
                        ('comparisons', report['comparisons']), ('input_distortion', distortion['rows'])):
        _write_csv(outdir / f'{name}.csv', rows, list(rows[0]))
        _atomic_json(outdir / f'{name}.json', report if name == 'main_metrics' else
                     distortion if name == 'input_distortion' else dict(status=STATUS, rows=rows))
    lines = [f'# {STATUS}', '', 'Full MM-Fi; seed42; 50 epochs per victim; rho=.1 (clean=0).', '',
        '| Method | T-MPJPE d=1 (mm) | Clean MPJPE (mm) | Clean PA-MPJPE (mm) | PCK .5/.4/.3/.2/.1 (%) |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in report['rows']:
        pck = '/'.join(f'{row[k]:.2f}' for k in PCK_METRICS)
        lines.append(f"| {row['method']} | {row['t1_mpjpe_mm']:.3f} | {row['clean_mpjpe_mm']:.3f} | {row['clean_pampjpe_mm']:.3f} | {pck} |")
    lines.extend(['', 'Errors: lower is better. PCK: higher is better; thresholds are relative, not mm.',
        'Six-dose curves retained in dose_response.csv. Budget diagnostics are separate, not ranking columns.',
        'No winner is selected automatically. Small numerical differences from one seed are not significance.',
        'Fixed multi-carrier control is not the previous learned paired-guard artifact.', '',
        *[f'- {item}' for item in report['limitations']], '', f"Plan SHA256: `{matrix['plan_sha256']}`"])
    (outdir / 'main_metrics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--distortion-samples', type=int, default=256)
    parser.add_argument('--cells', nargs='+', choices=[m[0] for m in METHODS])
    for name in ('dry-run', 'fresh', 'export-only'):
        parser.add_argument('--' + name, action='store_true')
    parser.add_argument('--_cell', choices=[m[0] for m in METHODS], help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.num_workers < 0 or args.distortion_samples < 1:
        raise ValueError('num_workers must be nonnegative and distortion_samples positive')
    if args.cells and len(set(args.cells)) != len(args.cells):
        raise ValueError('--cells must not contain duplicates')
    if args._cell:
        matrix = json.loads(args._manifest.read_text(encoding='utf-8'))
        _validate_manifest(matrix)
        _run_cell(next(c for c in matrix['cells'] if c['method_key'] == args._cell))
        return 0
    data_home = (args.data_home or HERE.parent.parent).resolve()
    outdir = (args.outdir or data_home / 'runs/mmfi_simple_full_s42_v1').resolve()
    if (len(set(args.devices)) != len(args.devices) or any(d != 'cpu' and
            re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None for d in args.devices)):
        raise ValueError('--devices must be distinct process-visible cuda:N IDs, or cpu')
    if args.dry_run and args.export_only:
        raise ValueError('--dry-run and --export-only are mutually exclusive')
    if args.fresh and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('--fresh refuses a nonempty directory; no results deleted')
    with _cell_lock(outdir):
        path = outdir / 'simple_full.resolved.json'
        requested = build_matrix(data_home, outdir, args.devices[0], args.num_workers,
                                 distortion_samples=args.distortion_samples)
        _validate_manifest(requested)
        if path.exists():
            matrix = json.loads(path.read_text(encoding='utf-8'))
            _validate_manifest(matrix)
            if matrix['plan_sha256'] != requested['plan_sha256']:
                raise ValueError('Plan changed; use a NEW output directory')
            for cell in matrix['cells']:
                cell['cfg'].update(device=args.devices[0], num_workers=args.num_workers)
        else:
            if args.export_only:
                raise ValueError('--export-only requires an existing complete experiment plan')
            if any(p.name != 'run.lock' for p in outdir.iterdir()):
                raise ValueError('Unrelated files/old results exist; use a NEW directory')
            matrix = requested
        _atomic_json(path, matrix)
        if args.dry_run:
            print(f'[{STATUS}] DRY RUN: seven fresh victims; FULL TRAIN/TEST; 50 epochs; seed42; rho=.1; no data/GPU/training.\n{path}', flush=True)
            return 0
        if not args.export_only:
            # Explain logical-device remapping before the opaque CUDA assertion.
            import torch
            for device in args.devices:
                if device != 'cpu' and int(device.split(':')[1]) >= torch.cuda.device_count():
                    raise ValueError(f'{device} invalid: process sees {torch.cuda.device_count()} CUDA devices; '
                        'CUDA_VISIBLE_DEVICES=0,1,3 maps physical GPUs to cuda:0,cuda:1,cuda:2')
            _check_inputs(matrix, args.devices)
            run_matrix(matrix, path, args.devices, args.cells, worker_script=Path(__file__))
        missing = [c['method_key'] for c in matrix['cells'] if not _complete(c)]
        if missing:
            if args.cells and not args.export_only:
                print(f'[{STATUS}] Requested cells complete; full export pending: {missing}', flush=True)
                return 0
            raise ValueError(f'All seven arms must finish before export: {missing}')
        export_summary(matrix, outdir)
        print(f'[simple-full] complete -> {outdir / "main_metrics.md"}', flush=True)
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f'[simple-full] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
