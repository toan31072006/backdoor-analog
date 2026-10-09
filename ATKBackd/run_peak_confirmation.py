"""Full MM-Fi confirmation: Original vs peak-bound multicarrier, seed 42.

Exactly two independently initialized victims, 50 epochs, canonical full
protocol1-s1 train/test splits. No screening subset or historical cache import.
This isolated confirmation does not rewrite the four-table paper experiment.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys

import yaml

from run_mmfi_tables import (
    _atomic_json, _cell_lock, _check_inputs, _complete, _plan_fingerprint,
    run_matrix,
)
from run_method_drafts import (
    GRID, PEAK_DISTORTION_COLUMNS, _file_sha, _measure_peak_pair, _sha_json,
    _source_provenance as _draft_source_provenance, _verified_cache, _write_csv,
    trigger_state_sha256,
)

HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_peak_confirmation_v1'
STATUS = 'FULL_MMFI_CONFIRMATION'
EXPECTED_COUNTS = {'train': 133056, 'eval': 33264}
METHODS = (
    ('original', 'Original zero-mean micro-Doppler', 'micro_dropper'),
    ('md_multicarrier_peak_matched', 'Multicarrier with Original peak bound',
     'md_multicarrier_peak_matched'),
)
SUMMARY_COLUMNS = (
    'status', 'method_key', 'method', 'seed', 'epochs', 'train_samples',
    'eval_samples', 'clean_mpjpe_mm', 'clean_pampjpe_mm',
    'clean_pck_0.5_pct', 'clean_pck_0.4_pct', 'clean_pck_0.3_pct',
    'clean_pck_0.2_pct', 'clean_pck_0.1_pct', 't1_mpjpe_mm',
    'mean_positive_dose_tmpjpe_mm', 'i1_improvement_mm', 'cfg_fingerprint',
    'train_pair_ids_sha256', 'eval_pair_ids_sha256', 'poison_plan_sha256',
)
DOSE_COLUMNS = ('status', 'method_key', 'method', 'seed', 'd',
                'no_trigger_tmpjpe_mm', 'triggered_tmpjpe_mm',
                'improvement_mm', 'baseline_source')


def _source_provenance():
    paths = ('run_peak_confirmation.py', 'models/hpeli.py',
             'models/sk_network.py', 'attack/skeleton_mmfi.py')
    return dict(_draft_source_provenance(),
                **{name: _file_sha(HERE / name) for name in paths})


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _base_config(data_home, device, num_workers):
    base = yaml.safe_load((HERE / 'configs/mmfi/attack_bend.yaml').read_text(encoding='utf-8'))
    base.update(dataset_root=str(data_home / 'datasets/Compress'),
        action_npy=str(data_home / 'actions/data_bend.npy'), seed=42,
        model='hpeli', optimizer='sgd', lr=0.001, momentum=0.9,
        weight_decay=0.0, batch_size=32, epochs=50, victim_epochs=50,
        device=device, num_workers=num_workers, ckpt_every=1, strict_resume=True,
        pretrained=False, data_parallel=False, pivot=1, target_joints=[2, 3],
        theta_max_deg=40.0, payload_axis=[0.0, 0.0, 1.0], dose_mode='linear',
        dose_grid=GRID, rho=0.4, poison_select='uniform', dose_min=0.2,
        dose_max=1.0, dose_coupling='paired', trigger_zero_mean=True,
        training_protocol='ordinary_erm', threat_model='training_data_poisoning',
        attacker_access='data_only', confirmation_profile=PROFILE)
    return base


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4, *,
                 distortion_samples=256):
    """Resolve immutable scientific settings without data or CUDA access."""
    from train_backdoor import (_resolve_training_config,
                                _validate_training_contract,
                                _CHECKPOINT_SCHEMA, _RESULT_SCHEMA)
    _positive_integer(distortion_samples, 'distortion_samples')
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    base = _base_config(data_home, device, num_workers)
    cells = []
    for key, label, trigger in METHODS:
        cfg = _resolve_training_config(dict(copy.deepcopy(base), trigger=trigger))
        _validate_training_contract(cfg)
        folder = outdir / key
        cells.append(dict(method_key=key, label=label, group='confirmation',
            tables=[], dependencies=[], cfg=cfg, ckpt_dir=str(folder),
            eval_cache=str(folder / 'eval_cache.json')))
    matrix = dict(schema=1, dataset='mmfi', seed=42, fresh_results_only=True,
        confirmation_profile=PROFILE, status=STATUS,
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources=dict(source_sha256=_source_provenance(),
                     baseline_config='configs/mmfi/attack_bend.yaml',
                     implementation='Same trigger operators as the screening experiment; full victim retraining'),
        cells=cells,
        metrics_contract=dict(confirmation_profile=PROFILE,
            checkpoint_schema=_CHECKPOINT_SCHEMA, result_schema=_RESULT_SCHEMA,
            errors='millimetres; cached errors in metres',
            pck='relative thresholds 0.5/0.4/0.3/0.2/0.1, NOT millimetres',
            target_joints=[2, 3], dose_grid=GRID, seeds=[42],
            evaluation_role='full_official_test', official_test_used=True,
            full_split_counts=EXPECTED_COUNTS, epochs=50, main_asr=False,
            distortion_samples=distortion_samples,
            distortion='digital model-input distortion; pooled energies, global Linf maximum',
            peak_bound='paired per-sample post-normalization Linf <= Original, zero tolerance',
            matches_l2_budget=False, uncertainty='single seed: not estimated',
            i1='same-model no-trigger T-MPJPE at dose 1 minus triggered T-MPJPE'))
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    return matrix


def _validate_manifest(matrix):
    from train_backdoor import _resolve_training_config
    if (matrix.get('confirmation_profile') != PROFILE or matrix.get('status') != STATUS
            or matrix.get('dataset') != 'mmfi' or matrix.get('seed') != 42
            or matrix.get('DRAFT_ONLY') or 'draft_profile' in matrix):
        raise ValueError('Manifest is not the full MM-Fi peak confirmation')
    if matrix.get('plan_sha256') != _plan_fingerprint(matrix):
        raise ValueError('Confirmation plan fingerprint differs from its contents')
    cells = matrix['cells']
    if [c['method_key'] for c in cells] != [m[0] for m in METHODS]:
        raise ValueError('Confirmation must retain both cells in fixed order')
    for cell, (_, _, trigger) in zip(cells, METHODS):
        cfg = cell['cfg']
        if any(k == 'method_draft' or k.startswith('draft_') for k in cfg):
            raise ValueError('Full confirmation must not use draft/subset options')
        data_home = Path(cfg['dataset_root']).parent.parent
        expected = _resolve_training_config(dict(
            _base_config(data_home, cfg['device'], cfg['num_workers']), trigger=trigger))
        if cfg != expected:
            raise ValueError(f"{cell['method_key']}: full confirmation configuration changed")
        if cell['dependencies'] or cell['tables']:
            raise ValueError('Confirmation cells must remain independent of old experiment tables')


def _pair_ids(ds):
    from data_utils.draft_subset import _relative_identifier
    root = getattr(ds, 'data_root', None)
    return [[_relative_identifier(it['csi'], root),
             _relative_identifier(it['kpt'], root), int(it['frame_idx'])]
            for it in ds.items]


def _collect_input_record(cell):
    """Record full ordered split identities, not CSI/pose content checksums."""
    from train_backdoor import _load_dataset, _config_fingerprint
    cfg = cell['cfg']
    record = dict(schema=1, profile=PROFILE,
        config_fingerprint=_config_fingerprint(cfg),
        action_file_sha256=_file_sha(cfg['action_npy']))
    sets = []
    for name, split in (('train', 'training'), ('eval', 'test')):
        ds = _load_dataset(cfg, split)
        ids = _pair_ids(ds)
        if len(ds) != EXPECTED_COUNTS[name] or len(ids) != len(ds):
            raise ValueError(f'Full {name} split requires {EXPECTED_COUNTS[name]} pairs, got {len(ds)}')
        id_set = {tuple(i) for i in ids}
        if len(id_set) != len(ids):
            raise ValueError(f'Duplicate pairs in full {name} split')
        sets.append(id_set)
        record[name] = dict(n=len(ids), pair_ids_sha256=_sha_json(ids))
    if sets[0].intersection(sets[1]):
        raise ValueError('Full training and official test pairs overlap')
    return record


def _bind_inputs(cell, record):
    folder = Path(cell['ckpt_dir'])
    path = folder / 'confirmation_inputs.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('Confirmation input/config identity changed; use a NEW directory')
    else:
        if any((folder / n).exists() for n in ('checkpoint.pt', 'eval_cache.json')):
            raise ValueError('Missing confirmation input audit; old caches cannot be imported')
        _atomic_json(path, record)


def _run_cell(cell):
    from train_backdoor import train, _config_fingerprint
    with _cell_lock(cell['ckpt_dir']):
        record = _collect_input_record(cell)
        _bind_inputs(cell, record)
        model, res = train(cell['cfg'], ckpt_dir=cell['ckpt_dir'])
        del model
        if _collect_input_record(cell) != record:
            raise ValueError('Confirmation inputs changed during training/evaluation')
        _atomic_json(Path(cell['ckpt_dir']) / 'results.json', dict(
            method_key=cell['method_key'], cfg=cell['cfg'], res=res,
            cfg_fingerprint=_config_fingerprint(cell['cfg'])))
        print(f"[confirmation] completed {cell['method_key']}", flush=True)


def audit_common_inputs(matrix):
    """Verify both full split identities and the actual paired poison plans."""
    from mmfi_tables import config_fingerprint
    records, plans, identity = {}, {}, None
    for cell in matrix['cells']:
        key, cfg = cell['method_key'], cell['cfg']
        folder = Path(cell['ckpt_dir'])
        record = json.loads((folder / 'confirmation_inputs.json').read_text(encoding='utf-8'))
        if (record.get('schema') != 1 or record.get('profile') != PROFILE
                or record.get('config_fingerprint') != config_fingerprint(cfg)):
            raise ValueError(f'{key}: invalid full input audit')
        for name in ('train', 'eval'):
            value = record.get(name, {})
            if (value.get('n') != EXPECTED_COUNTS[name]
                    or re.fullmatch('[0-9a-f]{64}', str(value.get('pair_ids_sha256'))) is None):
                raise ValueError(f'{key}: missing full {name} identity')
        action_sha = record.get('action_file_sha256')
        if re.fullmatch('[0-9a-f]{64}', str(action_sha)) is None:
            raise ValueError(f'{key}: missing action reference hash')
        current = (record['train'], record['eval'], action_sha)
        if identity is not None and current != identity:
            raise ValueError('Confirmation cells used different splits/action references')
        identity = current
        poison = json.loads((folder / 'poison_manifest.json').read_text(encoding='utf-8'))
        samples = poison.get('samples')
        if not isinstance(samples, list):
            raise ValueError(f'{key}: missing poison sample records')
        plan = []
        for sample in samples:
            i, d = sample.get('index'), sample.get('dose')
            if (isinstance(i, bool) or not isinstance(i, int)
                    or not 0 <= i < EXPECTED_COUNTS['train'] or isinstance(d, bool)
                    or not isinstance(d, (int, float)) or not math.isfinite(d)
                    or not 0.2 <= d <= 1.0 or sample.get('payload_dose', d) != d):
                raise ValueError(f'{key}: invalid paired poison sample')
            plan.append([i, d])
        digest = _sha_json(plan)
        expected_count = int(math.floor(0.4 * EXPECTED_COUNTS['train']))
        required = dict(config_fingerprint=config_fingerprint(cfg),
            n_total=EXPECTED_COUNTS['train'], n_poison=expected_count,
            seed=42, selection='uniform', dose_min=0.2, dose_max=1.0,
            poison_plan_sha256=digest)
        if (any(poison.get(k) != v for k, v in required.items())
                or len(plan) != expected_count or len({p[0] for p in plan}) != len(plan)):
            raise ValueError(f'{key}: poison manifest differs from full shared protocol')
        records[key], plans[key] = record, digest
    if len(set(plans.values())) != 1:
        raise ValueError('Confirmation methods must use identical poison indices/doses')
    return dict(inputs=records, poison_plan_sha256=plans,
                action_file_sha256=identity[2])


def build_summary(matrix):
    from mmfi_tables import metric_row, dose_rows
    _validate_manifest(matrix)
    audit = audit_common_inputs(matrix)
    rows, doses, provenance = [], [], []
    for cell in matrix['cells']:
        res, source = _verified_cache(cell)
        key = cell['method_key']
        if (res.get('training_contract') != 'ordinary_erm'
                or res.get('attacker_access') != 'data_only'
                or res.get('dose_coupling') != 'paired'
                or res.get('n_total') != EXPECTED_COUNTS['train']
                or res.get('n_poison') != int(0.4 * EXPECTED_COUNTS['train'])
                or res.get('poison_plan_sha256') != audit['poison_plan_sha256'][key]):
            raise ValueError(f'{key}: cached results disagree with full input/poison audit')
        row = metric_row(cell, res)
        series, _ = dose_rows(cell, res)
        if [r['d'] for r in series] != GRID:
            raise ValueError(f'{key}: incomplete six-dose evaluation')
        positive = [r for r in series if r['d'] > 0]
        row.update(status=STATUS, epochs=50,
            train_samples=EXPECTED_COUNTS['train'], eval_samples=EXPECTED_COUNTS['eval'],
            t1_mpjpe_mm=row.pop('tmpjpe_mm'),
            mean_positive_dose_tmpjpe_mm=sum(r['triggered_tmpjpe_mm'] for r in positive) / len(positive),
            i1_improvement_mm=series[-1]['improvement_mm'],
            cfg_fingerprint=source['cfg_fingerprint'],
            train_pair_ids_sha256=audit['inputs'][key]['train']['pair_ids_sha256'],
            eval_pair_ids_sha256=audit['inputs'][key]['eval']['pair_ids_sha256'],
            poison_plan_sha256=audit['poison_plan_sha256'][key])
        rows.append({name: row[name] for name in SUMMARY_COLUMNS})
        doses.extend(dict(status=STATUS, **r) for r in series)
        provenance.append(source)
    return dict(status='full_confirmation_complete', profile=PROFILE,
        plan_sha256=matrix['plan_sha256'], rows=rows, dose_response=doses,
        metrics_contract=matrix['metrics_contract'], sources=matrix['sources'],
        audit=audit, provenance=provenance,
        limitations=['Single seed; no uncertainty or statistical superiority is estimated.',
            'The candidate was selected on a training holdout; full evaluation now uses the official test split.',
            'Peak upper bounds do not match L2 or establish lower detectability/over-the-air feasibility.',
            'Input-conditioned peak scaling is not a pure fixed-carrier ablation.',
            'Split hashes cover ordered file/frame identities, not the contents of every CSI/pose file.',
            'This report does not replace or complete the separate four-table experiment.'])


def build_distortion(matrix, audit):
    import numpy as np
    from train_backdoor import _load_dataset, build_trigger
    from mmfi_tables import config_fingerprint
    original_cell, candidate_cell = matrix['cells']
    cfg = dict(candidate_cell['cfg'], device='cpu', num_workers=0)
    record = _collect_input_record(candidate_cell)
    if record != audit['inputs'][candidate_cell['method_key']]:
        raise ValueError('Current full dataset/action differs from training-time audit')
    ds = _load_dataset(cfg, 'test')
    all_ids = _pair_ids(ds)
    n = matrix['metrics_contract']['distortion_samples']
    indices = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int).tolist()
    ids = [all_ids[i] for i in indices]
    original, candidate = build_trigger(original_cell['cfg']), build_trigger(cfg)
    original_sha, candidate_sha = trigger_state_sha256(original), trigger_state_sha256(candidate)
    rows, paired = [], []

    def snr_fields(value, prefix=''):
        snr = float(value['snr_db'])
        if math.isnan(snr):
            raise ValueError('Invalid NaN distortion SNR')
        return {prefix + 'snr_db': snr if math.isfinite(snr) else None,
                prefix + 'snr_db_is_infinite': snr == float('inf'),
                prefix + 'snr_db_is_negative_infinite': snr == float('-inf')}

    for dose in GRID:
        values, reference, pairs = _measure_peak_pair(
            cfg, ds, indices, ids, dose, candidate, original)
        paired.append(dict(dose=dose, pairs=pairs))
        for cell, measurement, sha in ((original_cell, reference, original_sha),
                                       (candidate_cell, values, candidate_sha)):
            rows.append(dict(status=STATUS, method_key=cell['method_key'], dose=dose,
                n_samples=len(ids), relative_l2=measurement['relative_l2'],
                rmse=measurement['rms'], linf=measurement['max_abs'],
                **snr_fields(measurement), cfg_fingerprint=config_fingerprint(cell['cfg']),
                trigger_state_sha256=sha, action_file_sha256=record['action_file_sha256'],
                common_pair_ids_sha256=_sha_json(ids),
                original_relative_l2=reference['relative_l2'], original_rmse=reference['rms'],
                original_linf=reference['max_abs'], **snr_fields(reference, 'original_'),
                original_trigger_state_sha256=original_sha, paired_linf_violations=0,
                max_paired_linf_excess=0.0 if cell is original_cell else max(p['linf_excess'] for p in pairs)))
    if (trigger_state_sha256(original) != original_sha
            or trigger_state_sha256(candidate) != candidate_sha
            or _file_sha(cfg['action_npy']) != record['action_file_sha256']):
        raise ValueError('Fixed trigger/reference changed during distortion measurement')
    report = dict(status=STATUS, profile=PROFILE,
        plan_sha256=matrix['plan_sha256'], common_pair_ids=ids,
        common_pair_ids_sha256=_sha_json(ids), rows=rows, paired_samples=paired,
        selection='linspace over the common full official test split',
        units='dimensionless digital CSI; SNR in dB',
        aggregation='pooled signal/error energy; Linf is the global maximum',
        peak_audit='each sampled candidate Linf <= Original, zero tolerance',
        matches_l2_budget=False)
    json.dumps(report, allow_nan=False)
    return report


def export_summary(matrix, outdir):
    report = build_summary(matrix)
    distortion = build_distortion(matrix, report['audit'])
    outdir = Path(outdir)
    _atomic_json(outdir / 'confirmation_summary.json', report)
    _write_csv(outdir / 'confirmation_summary.csv', report['rows'], SUMMARY_COLUMNS)
    _write_csv(outdir / 'dose_response.csv', report['dose_response'], DOSE_COLUMNS)
    _atomic_json(outdir / 'input_distortion.json', distortion)
    _write_csv(outdir / 'input_distortion.csv', distortion['rows'], PEAK_DISTORTION_COLUMNS)
    lines = ['# Full MM-Fi peak confirmation', '',
        'Seed 42; 50 epochs; full protocol1-s1 train/test; both victims trained afresh.', '',
        '| Method | Clean MPJPE (mm) | PA-MPJPE (mm) | PCK .5/.4/.3/.2/.1 (%) | T1 (mm) | Mean positive-dose T (mm) | I1 (mm) |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for row in report['rows']:
        pck = '/'.join(f"{row[f'clean_pck_{t:.1f}_pct']:.2f}" for t in (0.5, 0.4, 0.3, 0.2, 0.1))
        lines.append(f"| {row['method']} | {row['clean_mpjpe_mm']:.3f} | {row['clean_pampjpe_mm']:.3f} | {pck} | {row['t1_mpjpe_mm']:.3f} | {row['mean_positive_dose_tmpjpe_mm']:.3f} | {row['i1_improvement_mm']:.3f} |")
    lines.extend(['', 'T1 is T-MPJPE at d=1. I1 is the same-model no-trigger target baseline minus T1.',
        'PCK thresholds are relative, not millimetres. All six evaluation doses are retained.',
        f"Status: `{report['status']}`. Plan SHA256: `{matrix['plan_sha256']}`.", '',
        'Limitations:', *[f'- {item}' for item in report['limitations']]])
    path = outdir / 'confirmation_summary.md'
    temporary = path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--distortion-samples', type=int, default=256)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--_cell', help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args._cell:
        matrix = json.loads(args._manifest.read_text(encoding='utf-8'))
        _validate_manifest(matrix)
        if matrix['sources']['source_sha256'] != _source_provenance():
            raise ValueError('Confirmation source changed after plan creation')
        _run_cell(next(c for c in matrix['cells'] if c['method_key'] == args._cell))
        return 0
    if args.data_home is None or args.outdir is None:
        raise ValueError('--data-home and --outdir are required')
    if (len(set(args.devices)) != len(args.devices)
            or any(d != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None for d in args.devices)):
        raise ValueError('--devices must list distinct logical cuda:N devices, or cpu')
    if args.num_workers < 0:
        raise ValueError('--num-workers cannot be negative')
    _positive_integer(args.distortion_samples, 'distortion_samples')
    outdir = args.outdir.resolve()
    if args.fresh and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('--fresh refuses a non-empty output directory; use a NEW path')
    with _cell_lock(outdir):
        manifest_path = outdir / 'peak_confirmation.resolved.json'
        requested = build_matrix(args.data_home, outdir, args.devices[0], args.num_workers,
                                 distortion_samples=args.distortion_samples)
        _validate_manifest(requested)
        if manifest_path.exists():
            matrix = json.loads(manifest_path.read_text(encoding='utf-8'))
            _validate_manifest(matrix)
            if matrix['plan_sha256'] != requested['plan_sha256']:
                raise ValueError('Existing confirmation differs; budgets/source changes require a NEW directory')
            for cell in matrix['cells']:
                cell['cfg']['num_workers'] = args.num_workers
                cell['cfg']['device'] = args.devices[0]
        else:
            if any(p.name != 'run.lock' for p in outdir.iterdir()):
                raise ValueError('Output directory contains unrelated files; use a NEW directory')
            matrix = requested
        _atomic_json(manifest_path, matrix)
        if args.dry_run:
            print(f'[confirmation] DRY RUN: seed42, 50 epochs, two full independent cells; NO training.\n{manifest_path}', flush=True)
            return 0
        if not args.export_only:
            _check_inputs(matrix, args.devices)
            run_matrix(matrix, manifest_path, args.devices, worker_script=Path(__file__))
        if any(not _complete(cell) for cell in matrix['cells']):
            raise ValueError('Both full cells must finish before exporting a confirmation summary')
        export_summary(matrix, outdir)
        print(f'[confirmation] Full summary: {outdir / "confirmation_summary.md"}', flush=True)
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f'[confirmation] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
