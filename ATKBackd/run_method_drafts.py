"""Isolated, single-seed method screening: DRAFT_ONLY_NOT_PAPER_RESULTS.

This runner never writes publication tables. Changed scientific budgets require
a new output directory; --fresh never removes existing files. Training uses the
ordinary data-only ERM path and the existing independent-cell scheduler.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
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

HERE = Path(__file__).resolve().parent
STATUS = 'DRAFT_ONLY_NOT_PAPER_RESULTS'
PROFILE = 'method_screening_v1'
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
METHODS = (
    ('clean', 'Clean draft control', 'micro_dropper'),
    ('original', 'Original zero-mean micro-Doppler', 'micro_dropper'),
    ('md_multicarrier', 'Draft multicarrier', 'md_multicarrier'),
    ('md_dose_code', 'Draft dose code', 'md_dose_code'),
    ('md_energy', 'Draft input-energy normalization', 'md_energy'),
)
SUMMARY_COLUMNS = (
    'status', 'method_key', 'method', 'seed', 'epochs', 'train_samples',
    'eval_samples', 'clean_mpjpe_mm', 'clean_pampjpe_mm',
    'clean_pck_0.5_pct', 'clean_pck_0.4_pct', 'clean_pck_0.3_pct',
    'clean_pck_0.2_pct', 'clean_pck_0.1_pct', 't1_mpjpe_mm',
    'mean_positive_dose_tmpjpe_mm', 'i1_improvement_mm',
    'cfg_fingerprint', 'train_subset_sha256', 'eval_subset_sha256',
    'poison_plan_sha256',
)
DISTORTION_COLUMNS = (
    'status', 'method_key', 'dose', 'n_samples', 'relative_l2', 'rmse',
    'linf', 'snr_db', 'snr_db_is_infinite', 'snr_db_is_negative_infinite',
    'cfg_fingerprint', 'trigger_state_sha256', 'action_file_sha256',
    'common_pair_ids_sha256',
)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer (not bool)')
    return value


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance():
    paths = (
        'run_method_drafts.py', 'run_mmfi_tables.py', 'train_backdoor.py',
        'configs/mmfi/attack_bend.yaml', 'attack/trigger.py',
        'attack/method_drafts.py', 'attack/poison.py', 'attack/payload.py',
        'data_utils/draft_subset.py', 'data_utils/feeder.py',
        'eval/distortion.py', 'eval/metrics.py', 'models/factory.py', 'mmfi_tables.py',
    )
    return {name: _file_sha(HERE / name) for name in paths}


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4, *,
                 epochs=15, train_samples=20000, eval_samples=4096,
                 distortion_samples=256):
    """Resolve a draft plan without opening datasets, triggers, or CUDA."""
    from train_backdoor import (_resolve_training_config,
                                _validate_training_contract,
                                _CHECKPOINT_SCHEMA, _RESULT_SCHEMA)
    for name, value in [('epochs', epochs), ('train_samples', train_samples),
                        ('eval_samples', eval_samples),
                        ('distortion_samples', distortion_samples)]:
        _positive_int(value, name)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError('num_workers must be a nonnegative integer (not bool)')
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    base = yaml.safe_load((HERE / 'configs/mmfi/attack_bend.yaml').read_text(encoding='utf-8'))
    base.update(dataset_root=str(data_home / 'datasets/Compress'),
        action_npy=str(data_home / 'actions/data_bend.npy'),
        seed=42, model='hpeli', optimizer='sgd', lr=0.001, momentum=0.9,
        weight_decay=0.0, batch_size=32, epochs=epochs, victim_epochs=epochs,
        device=device, num_workers=num_workers, ckpt_every=1,
        strict_resume=True, pretrained=False, data_parallel=False,
        pivot=1, target_joints=[2, 3], theta_max_deg=40.0,
        payload_axis=[0.0, 0.0, 1.0], dose_mode='linear', dose_grid=GRID,
        rho=0.4, poison_select='uniform', dose_min=0.2, dose_max=1.0,
        dose_coupling='paired', trigger_zero_mean=True,
        training_protocol='ordinary_erm', threat_model='training_data_poisoning',
        attacker_access='data_only', method_draft=True, draft_profile=PROFILE,
        draft_train_samples=train_samples, draft_eval_samples=eval_samples,
        draft_subset_seed=0, draft_eval_source='training_holdout')
    cells = []
    for key, label, trigger in METHODS:
        cfg = copy.deepcopy(base)
        cfg.update(trigger=trigger, rho=0.0 if key == 'clean' else 0.4)
        cfg = _resolve_training_config(cfg)
        _validate_training_contract(cfg)
        folder = outdir / key
        cells.append(dict(method_key=key, label=label, group='draft_only',
            tables=[], dependencies=[], cfg=cfg, ckpt_dir=str(folder),
            eval_cache=str(folder / 'eval_cache.json')))
    matrix = dict(schema=1, dataset='mmfi', seed=42, fresh_results_only=True,
        status=STATUS, DRAFT_ONLY=True, draft_profile=PROFILE,
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources={'status': STATUS, 'source_sha256': _source_provenance(),
            'baseline_config': 'configs/mmfi/attack_bend.yaml',
            'implementation': 'Independent exploratory fixed NumPy trigger variants',
            'publication_claim': False}, cells=cells,
        metrics_contract=dict(status=STATUS, draft_profile=PROFILE,
            checkpoint_schema=_CHECKPOINT_SCHEMA, result_schema=_RESULT_SCHEMA,
            errors='millimetres; cached pose errors in metres',
            pck='relative thresholds 0.5/0.4/0.3/0.2/0.1, displayed as percent',
            target_joints=[2, 3], dose_grid=GRID, seeds=[42],
            main_asr=False, uncertainty='single-seed draft: not estimated',
            rank_selection=False, distortion_samples=distortion_samples,
            distortion='digital model-input distortion, CPU, no HPE forward',
            i1='same-model no-trigger T-MPJPE at dose 1 minus triggered T-MPJPE',
            positive_dose_mean='arithmetic mean over all five strictly positive doses'))
    matrix['metrics_contract'].update(evaluation_role='training_holdout_validation',
        official_test_used=False,
        subset_selection='seed-0 shared permutation; disjoint train and validation from official training')
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    return matrix


def _validate_manifest(matrix):
    if (matrix.get('status') != STATUS or matrix.get('DRAFT_ONLY') is not True or
            matrix.get('draft_profile') != PROFILE):
        raise ValueError('Manifest is not an isolated DRAFT_ONLY method screen')
    if matrix.get('plan_sha256') != _plan_fingerprint(matrix):
        raise ValueError('Draft plan fingerprint differs from its resolved contents')
    keys = [cell['method_key'] for cell in matrix['cells']]
    if keys != [row[0] for row in METHODS]:
        raise ValueError('Draft plan must retain all five cells in their fixed order')
    for cell in matrix['cells']:
        cfg = cell['cfg']
        if (cfg.get('method_draft') is not True or cfg.get('draft_profile') != PROFILE or
                cfg.get('draft_eval_source') != 'training_holdout'):
            raise ValueError(f"{cell['method_key']}: missing isolated draft config markers")
        for key in ('epochs', 'draft_train_samples', 'draft_eval_samples'):
            _positive_int(cfg.get(key), key)


def _verified_cache(cell):
    """Reject legacy, partial, edited, or differently budgeted cache wrappers."""
    from mmfi_tables import _load_verified_cache, _trainer_schemas
    path = Path(cell['eval_cache'])
    try:
        blob = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{cell['method_key']}: missing valid completed cache: {exc}") from exc
    _, result_schema = _trainer_schemas()
    if not isinstance(blob, dict) or blob.get('result_schema') != result_schema:
        raise ValueError(f"{cell['method_key']}: exact current result schema is required")
    return _load_verified_cache(cell)


def _read_audit(path, cell):
    from mmfi_tables import config_fingerprint
    try:
        record = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{cell['method_key']}: missing audit manifest {Path(path).name}") from exc
    if not isinstance(record, dict) or record.get('config_fingerprint') != config_fingerprint(cell['cfg']):
        raise ValueError(f"{cell['method_key']}: audit config fingerprint mismatch")
    return record


def _subset_identity(record, name):
    subset = record.get(name)
    if not isinstance(subset, dict):
        raise ValueError(f'Missing {name} subset identity')
    indices = subset.get('subset_indices')
    if (not isinstance(indices, list) or not indices or
            any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices) or
            indices != sorted(set(indices))):
        raise ValueError(f'{name} subset requires its exact sorted unique parent indices')
    digest = subset.get('identifiers_sha256')
    if (subset.get('schema') != 1 or subset.get('profile') != PROFILE or
            subset.get('eval_source') != 'training_holdout' or
            subset.get('selection_seed') != 0 or subset.get('n') != len(indices) or
            not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None or
            subset.get('index_sha256') != _sha_json(indices)):
        raise ValueError(f'{name} subset identity or index SHA256 is invalid')
    _positive_int(subset.get('parent_n'), f'{name}.parent_n')
    _positive_int(subset.get('requested_cap'), f'{name}.requested_cap')
    if indices[-1] >= subset['parent_n'] or len(indices) > subset['requested_cap']:
        raise ValueError(f'{name} subset exceeds its parent dataset or requested cap')
    return indices, digest


def audit_common_inputs(cells):
    """Validate shared pair IDs and recompute the four attack poison-plan hashes."""
    subsets, poisons, reference, action_sha = {}, {}, None, None
    for cell in cells:
        key = cell['method_key']
        folder = Path(cell['ckpt_dir'])
        subset = _read_audit(folder / 'draft_subsets.json', cell)
        if subset.get('status') != STATUS:
            raise ValueError(f'{key}: subset manifest is not DRAFT_ONLY')
        recorded_action_sha = subset.get('action_file_sha256')
        if (not isinstance(recorded_action_sha, str) or
                re.fullmatch('[0-9a-f]{64}', recorded_action_sha) is None):
            raise ValueError(f'{key}: subset audit requires the training action-file SHA256')
        if action_sha is not None and recorded_action_sha != action_sha:
            raise ValueError(f'{key}: draft cells used different action-file versions')
        action_sha = recorded_action_sha
        train_indices, train_sha = _subset_identity(subset, 'train')
        eval_indices, eval_sha = _subset_identity(subset, 'eval')
        if set(train_indices).intersection(eval_indices):
            raise ValueError(f'{key}: draft training and validation indices overlap')
        if subset['train']['parent_n'] != subset['eval']['parent_n']:
            raise ValueError(f'{key}: training and validation must share the official training parent')
        if (subset['train']['requested_cap'] != cell['cfg']['draft_train_samples'] or
                subset['eval']['requested_cap'] != cell['cfg']['draft_eval_samples']):
            raise ValueError(f'{key}: subset cap differs from resolved config')
        identity = (train_indices, eval_indices, train_sha, eval_sha)
        if reference is not None and identity != reference:
            raise ValueError(f'{key}: cells did not use the same train/eval sample pair IDs')
        reference = identity
        subsets[key] = dict(train_subset_sha256=train_sha,
            eval_subset_sha256=eval_sha, train_samples=len(train_indices),
            eval_samples=len(eval_indices))
        poison = _read_audit(folder / 'poison_manifest.json', cell)
        samples = poison.get('samples')
        if not isinstance(samples, list):
            raise ValueError(f'{key}: poison manifest requires sample records')
        plan = []
        for sample in samples:
            index, dose = sample.get('index'), sample.get('dose')
            if (isinstance(index, bool) or not isinstance(index, int) or
                    not 0 <= index < len(train_indices) or isinstance(dose, bool) or
                    not isinstance(dose, (int, float)) or not math.isfinite(dose)):
                raise ValueError(f'{key}: invalid poison index/dose')
            if not 0.2 <= dose <= 1.0:
                raise ValueError(f'{key}: poison dose outside common U(0.2, 1.0)')
            if 'payload_dose' in sample and sample['payload_dose'] != dose:
                raise ValueError(f'{key}: payload and trigger doses must remain paired')
            plan.append([index, dose])
        # PoisonedDataset hashes this ordered plan with compact JSON.
        digest = hashlib.sha256(json.dumps(plan, separators=(',', ':'),
            ensure_ascii=True).encode('utf-8')).hexdigest()
        if poison.get('poison_plan_sha256') != digest:
            raise ValueError(f'{key}: poison plan SHA256 mismatch')
        if len({sample[0] for sample in plan}) != len(plan):
            raise ValueError(f'{key}: duplicate poison indices')
        expected_count = int(math.floor(cell['cfg']['rho'] * len(train_indices)))
        if (poison.get('n_total') != len(train_indices) or
                poison.get('n_poison') != expected_count or len(plan) != expected_count or
                poison.get('seed') != 42 or poison.get('selection') != 'uniform' or
                poison.get('dose_min') != 0.2 or poison.get('dose_max') != 1.0):
            raise ValueError(f'{key}: poison plan does not match the shared sampling protocol')
        poisons[key] = digest
    attacks = [poisons[cell['method_key']] for cell in cells if cell['method_key'] != 'clean']
    if len(set(attacks)) != 1:
        raise ValueError('The four attack cells did not use the same poison indices and doses')
    return dict(subsets=subsets, poison_plan_sha256=poisons,
        train_parent_indices=reference[0], eval_parent_indices=reference[1],
        action_file_sha256=action_sha)


def build_summary(matrix):
    """Read all five verified caches; never invent a missing result or rank rows."""
    from mmfi_tables import dose_rows, metric_row
    _validate_manifest(matrix)
    loaded = [_verified_cache(cell) for cell in matrix['cells']]
    audit = audit_common_inputs(matrix['cells'])
    rows, doses, provenance = [], [], []
    for cell, (res, source) in zip(matrix['cells'], loaded):
        key = cell['method_key']
        if (res.get('config_fingerprint') != source['cfg_fingerprint'] or
                res.get('draft_action_sha256') != audit['action_file_sha256'] or
                res.get('poison_plan_sha256') != audit['poison_plan_sha256'][key] or
                res.get('training_contract') != 'ordinary_erm' or
                res.get('attacker_access') != 'data_only' or
                res.get('dose_coupling') != 'paired' or
                res.get('n_total') != audit['subsets'][key]['train_samples'] or
                res.get('n_poison') != math.floor(cell['cfg']['rho'] * res['n_total'])):
            raise ValueError(f'{key}: cached result metadata disagrees with the data-only draft audit')
        row = metric_row(cell, res)
        series, _ = dose_rows(cell, res)
        if res.get('dose_grid') != GRID:
            raise ValueError(f'{key}: the exact six-dose grid is required')
        positive = [r for r in series if r['d'] > 0]
        reference = next(r for r in series if r['d'] == 1.0)
        row.update(status=STATUS, epochs=cell['cfg']['epochs'],
            t1_mpjpe_mm=row.pop('tmpjpe_mm'),
            mean_positive_dose_tmpjpe_mm=sum(r['triggered_tmpjpe_mm'] for r in positive) / len(positive),
            i1_improvement_mm=reference['improvement_mm'],
            cfg_fingerprint=source['cfg_fingerprint'],
            poison_plan_sha256=audit['poison_plan_sha256'][key],
            **audit['subsets'][key])
        rows.append({name: row[name] for name in SUMMARY_COLUMNS})
        doses.extend(dict(status=STATUS, **r) for r in series)
        provenance.append(source)
    return dict(status=STATUS, DRAFT_ONLY=True, draft_profile=PROFILE,
        plan_sha256=matrix['plan_sha256'], rows=rows, dose_response=doses,
        metrics_contract=matrix['metrics_contract'], sources=matrix['sources'],
        audit=audit, provenance=provenance,
        limitations=['Exploratory single-seed method screening only; these are not paper results.',
            'Reduced subsets and epoch budget; no rank selection or uncertainty estimate.',
            'Digital input distortion does not establish over-the-air feasibility.',
            'Clean-victim T metrics probe the original trigger; clean distortion measures the unmodified control input.'])


def trigger_state_sha256(trigger):
    """Hash numerical trigger state, including nested carriers and RNG state."""
    import numpy as np

    def state(value, ancestors):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.generic):
            return state(value.item(), ancestors)
        if isinstance(value, np.ndarray):
            return dict(dtype=value.dtype.str, shape=list(value.shape),
                data_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        if isinstance(value, np.random.Generator):
            return dict(generator=type(value.bit_generator).__name__,
                state=state(value.bit_generator.state, ancestors))
        if isinstance(value, np.random.RandomState):
            return dict(random_state=state(value.get_state(), ancestors))
        if isinstance(value, complex):
            return dict(real=value.real, imag=value.imag)
        identity = id(value)
        if identity in ancestors:
            raise ValueError('Cyclic trigger state cannot be fingerprinted')
        ancestors = ancestors | {identity}
        if isinstance(value, dict):
            return {str(k): state(v, ancestors) for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (tuple, list)):
            return [state(v, ancestors) for v in value]
        if hasattr(value, '__dict__'):
            return dict(type=f'{type(value).__module__}.{type(value).__qualname__}',
                attributes=state(vars(value), ancestors))
        raise ValueError(f'Unsupported trigger state type: {type(value).__name__}')

    return _sha_json(state(trigger, set()))


class _CleanIdentityTrigger:
    """The clean control receives the unmodified digital input at every dose."""
    def inject(self, csi, dose, eps=0.0):
        return csi.copy()


def build_distortion(matrix, audit, n=256):
    """Measure every fixed variant on the same real held-out pairs, on CPU."""
    import numpy as np
    from eval.distortion import measure
    from train_backdoor import _load_dataset, build_trigger
    from mmfi_tables import config_fingerprint
    _positive_int(n, 'distortion_samples')
    rows, common_ids, common_sha = [], None, None
    for cell in matrix['cells']:
        cfg = dict(cell['cfg'], device='cpu', num_workers=0)
        action_sha = _file_sha(cfg['action_npy'])
        if action_sha != audit['action_file_sha256']:
            raise ValueError(f"{cell['method_key']}: current action file differs from the training action-file SHA256")
        ds = _load_dataset(cfg, 'test')
        subset = ds.draft_subset_manifest()
        dataset_indices, dataset_sha = _subset_identity({'eval': subset}, 'eval')
        dataset_ids = ds.draft_pair_ids()
        if (_sha_json(dataset_ids) != dataset_sha or
                dataset_indices != audit['eval_parent_indices'] or
                dataset_sha != audit['subsets'][cell['method_key']]['eval_subset_sha256']):
            raise ValueError(f"{cell['method_key']}: distortion dataset differs from evaluated held-out subset")
        indices = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int).tolist()
        ids = [dataset_ids[i] for i in indices]
        digest = _sha_json(ids)
        if common_ids is not None and ids != common_ids:
            raise ValueError('Distortion variants did not use the same held-out sample pair IDs')
        common_ids, common_sha = ids, digest
        trigger = _CleanIdentityTrigger() if cell['method_key'] == 'clean' else build_trigger(cfg)
        trigger_sha = trigger_state_sha256(trigger)
        for dose in GRID:
            values = measure(cfg, n=n, dose=dose, trig=trigger, dataset=ds)
            if values['n_samples'] != len(ids):
                raise ValueError('Distortion sample count differs from the recorded common IDs')
            snr = float(values['snr_db'])
            if math.isnan(snr):
                raise ValueError('Distortion SNR is NaN')
            row = dict(status=STATUS, method_key=cell['method_key'], dose=dose,
                n_samples=values['n_samples'], relative_l2=float(values['relative_l2']),
                rmse=float(values['rms']), linf=float(values['max_abs']),
                snr_db=snr if math.isfinite(snr) else None,
                snr_db_is_infinite=snr == float('inf'),
                snr_db_is_negative_infinite=snr == float('-inf'),
                cfg_fingerprint=config_fingerprint(cfg), trigger_state_sha256=trigger_sha,
                action_file_sha256=action_sha, common_pair_ids_sha256=digest)
            if any(not math.isfinite(row[name]) or row[name] < 0
                   for name in ('relative_l2', 'rmse', 'linf')):
                raise ValueError('Nonfinite or negative distortion measurement')
            rows.append(row)
        if trigger_state_sha256(trigger) != trigger_sha:
            raise ValueError('Fixed trigger state changed during distortion measurement')
    return dict(status=STATUS, DRAFT_ONLY=True, plan_sha256=matrix['plan_sha256'],
        common_pair_ids=common_ids, common_pair_ids_sha256=common_sha,
        selection='linspace over the common sorted held-out draft subset',
        units='dimensionless model-input CSI; SNR in dB',
        clean_control='identity injection; clean-victim T metrics separately probe the original trigger',
        aggregation='pooled signal/error energy; Linf is the global maximum', rows=rows)


def _write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def export_summary(matrix, outdir, *, distortion_samples=256, skip_distortion=False):
    report = build_summary(matrix)
    distortion = None if skip_distortion else build_distortion(
        matrix, report['audit'], n=distortion_samples)
    outdir = Path(outdir)
    _atomic_json(outdir / 'draft_summary.json', report)
    _write_csv(outdir / 'draft_summary.csv', report['rows'], SUMMARY_COLUMNS)
    lines = [f'# {STATUS}', '',
        'Exploratory method screening only. All five rows are retained in plan order.', '',
        '| Method | Clean MPJPE (mm) | PA-MPJPE (mm) | PCK .5/.4/.3/.2/.1 (%) | T1 (mm) | Mean positive-dose T (mm) | I1 (mm) |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for row in report['rows']:
        pck = '/'.join(f"{row[f'clean_pck_{t:.1f}_pct']:.2f}" for t in (0.5, 0.4, 0.3, 0.2, 0.1))
        lines.append(f"| {row['method']} | {row['clean_mpjpe_mm']:.3f} | {row['clean_pampjpe_mm']:.3f} | {pck} | {row['t1_mpjpe_mm']:.3f} | {row['mean_positive_dose_tmpjpe_mm']:.3f} | {row['i1_improvement_mm']:.3f} |")
    lines.extend(['', f"Profile: `{PROFILE}`; seed 42; epochs {report['rows'][0]['epochs']}.",
        'T1 is triggered T-MPJPE at dose 1. I1 uses the same-model no-trigger target baseline.',
        'PCK thresholds are relative, not millimetres. No uncertainty or ranking is estimated.',
        'Evaluation uses a disjoint validation holdout from official training; the official test split is untouched.',
        'Clean-victim T metrics probe the original trigger; clean distortion is the unmodified control input.',
        f"Plan SHA256: `{matrix['plan_sha256']}`", '',
        'Digital distortion is skipped.' if skip_distortion else
        'Digital input distortion is recorded separately; no HPE forward is used.'])
    target = outdir / 'draft_summary.md'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(temporary, target)
    if distortion is not None:
        # Every nonfinite SNR has already become null with an explicit flag.
        json.dumps(distortion, allow_nan=False)
        _atomic_json(outdir / 'input_distortion.json', distortion)
        _write_csv(outdir / 'input_distortion.csv', distortion['rows'], DISTORTION_COLUMNS)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--train-samples', type=int, default=20000)
    parser.add_argument('--eval-samples', type=int, default=4096)
    parser.add_argument('--distortion-samples', type=int, default=256)
    parser.add_argument('--cells', nargs='+', choices=[row[0] for row in METHODS])
    parser.add_argument('--fresh', action='store_true', help='Require a NEW empty output directory; never delete results')
    parser.add_argument('--dry-run', action='store_true', help='Resolve only; no dataset, trigger, CUDA, or training')
    parser.add_argument('--export-only', action='store_true', help='Require all five valid completed caches')
    parser.add_argument('--skip-distortion', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for name in ('epochs', 'train_samples', 'eval_samples', 'distortion_samples'):
        _positive_int(getattr(args, name), name)
    if isinstance(args.num_workers, bool) or args.num_workers < 0:
        raise ValueError('--num-workers cannot be negative or bool')
    devices = args.devices
    if (not devices or len(set(devices)) != len(devices) or
            any(device != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', device) is None for device in devices)):
        raise ValueError('--devices must list distinct logical cuda:N devices, or cpu')
    if args.cells is not None and len(set(args.cells)) != len(args.cells):
        raise ValueError('--cells cannot contain duplicates')
    if args.dry_run and args.export_only:
        raise ValueError('--dry-run and --export-only cannot be combined')
    outdir = args.outdir.resolve()
    if args.fresh and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('--fresh refuses a non-empty output directory; use a NEW path')
    with _cell_lock(outdir):
        manifest_path = outdir / 'method_drafts.resolved.json'
        requested = build_matrix(args.data_home, outdir, devices[0], args.num_workers,
            epochs=args.epochs, train_samples=args.train_samples,
            eval_samples=args.eval_samples, distortion_samples=args.distortion_samples)
        if manifest_path.exists():
            matrix = json.loads(manifest_path.read_text(encoding='utf-8'))
            _validate_manifest(matrix)
            if matrix['plan_sha256'] != requested['plan_sha256']:
                raise ValueError('Existing draft plan differs; changed budgets/profile/source require a NEW output directory')
            for cell in matrix['cells']:
                cell['cfg'].update(num_workers=args.num_workers, device=devices[0])
        else:
            if any(path.name != 'run.lock' for path in outdir.iterdir()):
                raise ValueError('Output directory contains non-draft files; use a NEW output directory')
            matrix = requested
        _atomic_json(manifest_path, matrix)
        if args.dry_run:
            print(f'[{STATUS}] DRY RUN: five isolated cells; NO training.\n{manifest_path}', flush=True)
            return 0
        if not args.export_only:
            _check_inputs(matrix, devices)
            run_matrix(matrix, manifest_path, devices, args.cells)
        missing = [cell['method_key'] for cell in matrix['cells'] if not _complete(cell)]
        if missing:
            if args.cells and not args.export_only:
                print(f'[{STATUS}] Requested subset completed. Draft summary pending: {missing}', flush=True)
                return 0
            raise ValueError(f'Missing completed draft cells: {missing}; no summary exported')
        export_summary(matrix, outdir, distortion_samples=args.distortion_samples,
                       skip_distortion=args.skip_distortion)
        print(f'[{STATUS}] Draft summary: {outdir / "draft_summary.md"}', flush=True)
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f'[{STATUS}] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
