"""Full MM-Fi ablation of the frozen best fixed multicarrier method.

Four fresh victims: Full, shuffled dose pairing, Original single carrier, and
unbounded multicarrier. No traditional baseline is changed or retrained here.
The default is one seed (42), rho=.4, 50 epochs, complete protocol1-s1 splits.
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

from run_mmfi_tables import (
    _atomic_json, _cell_lock, _check_inputs, _complete, _plan_fingerprint,
    run_matrix,
)
from run_method_drafts import (
    GRID, _file_sha, _sha_json, _source_provenance as _draft_sources,
    _verified_cache, _write_csv, trigger_state_sha256,
)
from run_peak_confirmation import _base_config as _confirmation_config, _pair_ids

HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_proposed_ablation_v1'
STATUS = 'FULL_MMFI_ABLATION_SINGLE_SEED'
EXPECTED_COUNTS = {'train': 133056, 'eval': 33264}
METHODS = (
    ('full', 'Full: peak-bound multicarrier', 'md_multicarrier_peak_matched', 'paired'),
    ('w_o_coupling', 'w/o Dose coupling', 'md_multicarrier_peak_matched', 'shuffled'),
    ('w_o_auxiliary', 'w/o Auxiliary carrier (Original unit-RMS carrier)', 'micro_dropper', 'paired'),
    ('w_o_peak', 'w/o Peak bound', 'md_multicarrier', 'paired'),
)
SUMMARY_COLUMNS = (
    'status', 'method_key', 'method', 'seed', 'epochs', 'rho', 'train_samples',
    'eval_samples', 't1_mpjpe_mm', 'mean_dose_control_error_mm',
    'clean_mpjpe_mm', 'clean_pampjpe_mm', 'clean_pck_0.2_pct',
    'mean_positive_dose_tmpjpe_mm', 'cfg_fingerprint',
    'train_pair_ids_sha256', 'eval_pair_ids_sha256', 'poison_plan_sha256',
)
DOSE_COLUMNS = (
    'method_key', 'method', 'seed', 'd', 'no_trigger_tmpjpe_mm',
    'triggered_tmpjpe_mm', 'improvement_mm', 'baseline_source',
    'dose_control_error_mm',
)
DISTORTION_COLUMNS = (
    'method_key', 'dose', 'n_samples', 'relative_l2', 'rmse', 'linf',
    'snr_db', 'snr_db_is_infinite', 'original_linf', 'peak_bound_required',
    'paired_linf_violations', 'max_paired_linf_excess',
    'cfg_fingerprint', 'trigger_state_sha256', 'common_pair_ids_sha256',
)


def _source_provenance():
    extra = ('run_proposed_ablation.py', 'run_peak_confirmation.py',
             'models/hpeli.py', 'models/sk_network.py', 'attack/skeleton_mmfi.py')
    return dict(_draft_sources(), **{name: _file_sha(HERE / name) for name in extra})


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _configs(data_home, device, num_workers, seed):
    from train_backdoor import _resolve_training_config, _validate_training_contract
    base = _confirmation_config(Path(data_home), device, num_workers)
    base.update(seed=seed, confirmation_profile=PROFILE,
                measure_dose_control=True)
    configs = []
    for _, _, trigger, coupling in METHODS:
        raw = dict(copy.deepcopy(base), trigger=trigger, dose_coupling=coupling)
        if trigger != 'micro_dropper':
            # Freeze the actual trigger from the best seed-42 confirmation;
            # future --seed values change victim/poison RNG, not the carrier.
            raw.update(method_carrier_sub_mode=3, method_carrier_time_mode=1,
                       method_carrier_seed=42)
        cfg = _resolve_training_config(raw)
        _validate_training_contract(cfg)
        configs.append(cfg)
    return configs


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4, *,
                 seed=42, distortion_samples=256):
    from train_backdoor import _CHECKPOINT_SCHEMA, _RESULT_SCHEMA
    _integer(seed, 'seed')
    _integer(num_workers, 'num_workers')
    _integer(distortion_samples, 'distortion_samples', 1)
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    cells = []
    for (key, label, _, _), cfg in zip(METHODS, _configs(data_home, device, num_workers, seed)):
        folder = outdir / key
        cells.append(dict(method_key=key, label=label, group='proposed_ablation',
            tables=[], dependencies=[], cfg=cfg, ckpt_dir=str(folder),
            eval_cache=str(folder / 'eval_cache.json')))
    matrix = dict(schema=1, dataset='mmfi', seed=seed,
        confirmation_profile=PROFILE, status=STATUS, fresh_results_only=True,
        created_utc=datetime.now(timezone.utc).isoformat(), cells=cells,
        sources=dict(source_sha256=_source_provenance(),
            baseline_config='configs/mmfi/attack_bend.yaml',
            implementation='Frozen peak-confirmation operators; no traditional baseline changes'),
        metrics_contract=dict(confirmation_profile=PROFILE,
            checkpoint_schema=_CHECKPOINT_SCHEMA, result_schema=_RESULT_SCHEMA,
            evaluation_role='full_official_test', official_test_used=True,
            full_split_counts=EXPECTED_COUNTS, epochs=50, rho=0.4,
            target_joints=[2, 3], dose_grid=list(GRID), seeds=[seed],
            frozen_carrier_seed=42, uncertainty='single seed: not estimated',
            dose_control_definition='mean_j ||(pred_d-pred_0)-(target_d-true_0)||2 over target joints; then mean samples and positive doses',
            dose_control_units='cached metres; exported millimetres',
            auxiliary_removal='Original unit-RMS p0 replaces unit-RMS (p0+p1)/sqrt(2); not unscaled p0/sqrt(2)',
            peak_bound='Full/shuffled/Original <= the same per-input/dose Original realized Linf; w/o Peak is explicitly unconstrained',
            matches_l2_budget=False, main_asr=False,
            distortion_samples=distortion_samples,
            distortion='pooled digital model-input energies; per-sample Linf audit; no RF/OTA stealthiness claim'))
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    return matrix


def _validate_manifest(matrix):
    if (matrix.get('schema') != 1 or matrix.get('confirmation_profile') != PROFILE or matrix.get('status') != STATUS
            or matrix.get('dataset') != 'mmfi' or matrix.get('DRAFT_ONLY')
            or matrix.get('fresh_results_only') is not True or 'draft_profile' in matrix):
        raise ValueError('Manifest is not the full Proposed ablation')
    seed = _integer(matrix.get('seed'), 'seed')
    if matrix.get('plan_sha256') != _plan_fingerprint(matrix):
        raise ValueError('Ablation plan fingerprint differs from its contents')
    cells = matrix['cells']
    if [c['method_key'] for c in cells] != [row[0] for row in METHODS]:
        raise ValueError('Ablation must retain all four cells in fixed order')
    first = cells[0]
    data_home = Path(first['cfg']['dataset_root']).parent.parent
    outdir = Path(first['ckpt_dir']).parent
    expected = build_matrix(data_home, outdir, first['cfg']['device'],
        first['cfg']['num_workers'], seed=seed,
        distortion_samples=matrix['metrics_contract']['distortion_samples'])
    # Source identity is checked separately, including for worker processes.
    # Comparing canonical configs/contracts rejects editable-hash tampering.
    if matrix['metrics_contract'] != expected['metrics_contract']:
        raise ValueError('Ablation metrics contract changed')
    for cell, target in zip(cells, expected['cells']):
        expected_cfg = dict(target['cfg'], device=cell['cfg']['device'],
                            num_workers=cell['cfg']['num_workers'])
        if cell['cfg'] != expected_cfg:
            raise ValueError(f"{cell['method_key']}: ablation configuration changed")
        if any(cell.get(k) != target[k] for k in (
                'method_key', 'label', 'group', 'tables', 'dependencies', 'ckpt_dir', 'eval_cache')):
            raise ValueError('Ablation cell paths/identity changed')


def _collect_input_record(cell):
    from train_backdoor import _load_dataset, _config_fingerprint
    cfg = cell['cfg']
    record = dict(schema=1, profile=PROFILE,
        config_fingerprint=_config_fingerprint(cfg),
        action_file_sha256=_file_sha(cfg['action_npy']))
    seen = []
    for name, split in (('train', 'training'), ('eval', 'test')):
        ds = _load_dataset(cfg, split)
        ids = _pair_ids(ds)
        if len(ds) != EXPECTED_COUNTS[name] or len(ids) != len(ds):
            raise ValueError(f'Full {name} split requires {EXPECTED_COUNTS[name]} pairs, got {len(ds)}')
        unique = {tuple(pair) for pair in ids}
        if len(unique) != len(ids):
            raise ValueError(f'Duplicate pairs in full {name} split')
        seen.append(unique)
        record[name] = dict(n=len(ids), pair_ids_sha256=_sha_json(ids))
    if seen[0].intersection(seen[1]):
        raise ValueError('Full training and official test pairs overlap')
    return record


def _bind_inputs(cell, record):
    folder = Path(cell['ckpt_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'ablation_inputs.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('Ablation input/config identity changed; use a NEW directory')
    else:
        if any((folder / name).exists() for name in ('checkpoint.pt', 'eval_cache.json')):
            raise ValueError('Missing ablation input audit; old caches cannot be imported')
        _atomic_json(path, record)


def _run_cell(cell):
    from train_backdoor import train, _config_fingerprint
    with _cell_lock(cell['ckpt_dir']):
        record = _collect_input_record(cell)
        _bind_inputs(cell, record)
        model, res = train(cell['cfg'], ckpt_dir=cell['ckpt_dir'])
        del model
        if _collect_input_record(cell) != record:
            raise ValueError('Ablation inputs changed during training/evaluation')
        _atomic_json(Path(cell['ckpt_dir']) / 'results.json', dict(
            method_key=cell['method_key'], cfg=cell['cfg'], res=res,
            cfg_fingerprint=_config_fingerprint(cell['cfg'])))
        print(f"[ablation] completed {cell['method_key']}", flush=True)


def audit_common_inputs(matrix):
    import numpy as np
    from mmfi_tables import config_fingerprint
    records, plans, assignments, identity, common_plan = {}, {}, {}, None, None
    seed = matrix['seed']
    for cell in matrix['cells']:
        key, cfg = cell['method_key'], cell['cfg']
        folder = Path(cell['ckpt_dir'])
        record = json.loads((folder / 'ablation_inputs.json').read_text(encoding='utf-8'))
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
            raise ValueError('Ablation cells used different splits/action references')
        identity = current
        poison = json.loads((folder / 'poison_manifest.json').read_text(encoding='utf-8'))
        samples = poison.get('samples')
        if not isinstance(samples, list):
            raise ValueError(f'{key}: missing poison sample records')
        plan, payload = [], []
        for sample in samples:
            i, d = sample.get('index'), sample.get('dose')
            pd = sample.get('payload_dose', d)
            if (isinstance(i, bool) or not isinstance(i, int)
                    or not 0 <= i < EXPECTED_COUNTS['train']
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                        or not math.isfinite(x) or not 0.2 <= x <= 1.0 for x in (d, pd))):
                raise ValueError(f'{key}: invalid poison sample')
            plan.append([i, d])
            payload.append(pd)
        expected_count = math.floor(0.4 * EXPECTED_COUNTS['train'])
        digest = _sha_json(plan)
        required = dict(config_fingerprint=config_fingerprint(cfg),
            n_total=EXPECTED_COUNTS['train'], n_poison=expected_count,
            seed=seed, selection='uniform', dose_min=0.2, dose_max=1.0,
            rho_requested=0.4, pivot=1, target_joints=[2, 3],
            theta_max_deg=40.0, payload_axis=[0.0, 0.0, 1.0],
            dose_mode='linear', poison_plan_sha256=digest)
        if (any(poison.get(k) != v for k, v in required.items())
                or len(plan) != expected_count or len({p[0] for p in plan}) != len(plan)
                or poison.get('n_cover', 0) != 0 or poison.get('cover_ratio', 0.0) != 0.0
                or poison.get('dose_coupling', 'paired') != cfg['dose_coupling']):
            raise ValueError(f'{key}: poison manifest differs from full shared protocol')
        expected_payload = np.array([p[1] for p in plan])
        if cfg['dose_coupling'] == 'shuffled':
            np.random.default_rng(np.random.SeedSequence([seed, 7043])).shuffle(expected_payload)
        if payload != expected_payload.tolist():
            raise ValueError(f'{key}: payload-dose assignment is not the specified intervention')
        if common_plan is not None and plan != common_plan:
            raise ValueError('Ablation methods must share the actual poison indices/trigger doses')
        common_plan = plan
        records[key], plans[key] = record, digest
        assignments[key] = _sha_json([[i, d, pd] for (i, d), pd in zip(plan, payload)])
    return dict(inputs=records, poison_plan_sha256=plans,
                payload_assignment_sha256=assignments, action_file_sha256=identity[2])


def _dose_errors(res):
    values = res.get('dose_control_error')
    if (not isinstance(values, list) or len(values) != len(GRID)
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or x < 0 for x in values)):
        raise ValueError('Missing finite six-dose control errors; evaluate this new profile')
    if abs(values[0]) > 1e-10:
        raise ValueError('Dose-control error at identity dose must be zero')
    attacks = res.get('asr_curve')
    if (not isinstance(attacks, list) or len(attacks) != len(GRID)
            or any(not isinstance(attack, dict)
                   or isinstance(attack.get('dose_control_error_mean'), bool)
                   or attack.get('dose_control_error_mean') != value
                   for attack, value in zip(attacks, values))):
        raise ValueError('Dose-control error curve differs from per-dose results')
    return [float(value) * 1000.0 for value in values]


def build_summary(matrix):
    from mmfi_tables import metric_row, dose_rows
    _validate_manifest(matrix)
    if matrix['sources']['source_sha256'] != _source_provenance():
        raise ValueError('Ablation source changed after plan creation')
    audit = audit_common_inputs(matrix)
    rows, doses, provenance = [], [], []
    for cell in matrix['cells']:
        res, source = _verified_cache(cell)
        key, cfg = cell['method_key'], cell['cfg']
        if (res.get('training_contract') != 'ordinary_erm'
                or res.get('attacker_access') != 'data_only'
                or res.get('dose_coupling') != cfg['dose_coupling']
                or res.get('n_total') != EXPECTED_COUNTS['train']
                or res.get('n_poison') != math.floor(0.4 * EXPECTED_COUNTS['train'])
                or res.get('poison_plan_sha256') != audit['poison_plan_sha256'][key]):
            raise ValueError(f'{key}: result disagrees with full input/poison audit')
        series, _ = dose_rows(cell, res)
        if [r['d'] for r in series] != GRID:
            raise ValueError(f'{key}: incomplete six-dose evaluation')
        dce = _dose_errors(res)
        row = metric_row(cell, res)
        row.update(status=STATUS, epochs=50, rho=0.4,
            train_samples=EXPECTED_COUNTS['train'], eval_samples=EXPECTED_COUNTS['eval'],
            t1_mpjpe_mm=row.pop('tmpjpe_mm'),
            mean_dose_control_error_mm=sum(dce[1:]) / 5,
            mean_positive_dose_tmpjpe_mm=sum(r['triggered_tmpjpe_mm'] for r in series[1:]) / 5,
            cfg_fingerprint=source['cfg_fingerprint'],
            train_pair_ids_sha256=audit['inputs'][key]['train']['pair_ids_sha256'],
            eval_pair_ids_sha256=audit['inputs'][key]['eval']['pair_ids_sha256'],
            poison_plan_sha256=audit['poison_plan_sha256'][key])
        rows.append({name: row[name] for name in SUMMARY_COLUMNS})
        doses.extend(dict(r, dose_control_error_mm=error) for r, error in zip(series, dce))
        provenance.append(source)
    return dict(status='proposed_ablation_complete', profile=PROFILE,
        seed=matrix['seed'], plan_sha256=matrix['plan_sha256'], rows=rows,
        dose_response=doses, metrics_contract=matrix['metrics_contract'],
        sources=matrix['sources'], audit=audit, provenance=provenance,
        limitations=[
            'Single seed; no uncertainty or statistical superiority is estimated.',
            'All four victims are trained afresh; no historical result import or victim-weight transfer.',
            'The frozen method was selected during previous development, not a blinded first test.',
            'Auxiliary removal uses the Original unit-RMS carrier; carrier allocation changes, actual L2 is not matched.',
            'w/o Peak bound is an explicitly different-budget diagnostic, not a fair same-budget ranking.',
            'Digital input distortion does not establish RF stealthiness or over-the-air feasibility.',
            'Split hashes cover ordered file/frame IDs and action-file bytes, not every CSI/pose file.',
        ])


def build_distortion(matrix, audit):
    import numpy as np
    from eval.distortion import distortion_stats, _aggregate
    from train_backdoor import _load_dataset, build_trigger, _trigger_eps
    from mmfi_tables import config_fingerprint
    first = matrix['cells'][0]
    cfg = dict(first['cfg'], device='cpu', num_workers=0)
    if _collect_input_record(first) != audit['inputs'][first['method_key']]:
        raise ValueError('Current full inputs differ from training-time audit')
    ds = _load_dataset(cfg, 'test')
    all_ids = _pair_ids(ds)
    n = matrix['metrics_contract']['distortion_samples']
    indices = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int).tolist()
    ids = [all_ids[i] for i in indices]
    original = build_trigger(dict(cfg, trigger='micro_dropper'))
    original_sha = trigger_state_sha256(original)
    rows, paired = [], []
    for cell in matrix['cells']:
        trigger = build_trigger(dict(cell['cfg'], device='cpu', num_workers=0))
        state_sha = trigger_state_sha256(trigger)
        for dose in GRID:
            measurements, references, pairs = [], [], []
            for index, identifier in zip(indices, ids):
                raw = ds.load_raw(ds.items[index]['csi'])
                clean = ds.normalize(raw)
                hit = ds.normalize(trigger.inject(raw.copy(), dose, eps=_trigger_eps(cell['cfg'], trigger)))
                reference = ds.normalize(original.inject(raw.copy(), dose, eps=_trigger_eps(cfg, original)))
                if (hit.shape != clean.shape or not np.isfinite(hit).all()
                        or reference.shape != clean.shape or not np.isfinite(reference).all()):
                    raise ValueError('Invalid tensor in digital distortion audit')
                m, r = distortion_stats(clean, hit), distortion_stats(clean, reference)
                excess = m['max_abs'] - r['max_abs']
                if cell['method_key'] != 'w_o_peak' and excess > 0:
                    raise ValueError(f"{cell['method_key']}: per-sample peak exceeds Original")
                measurements.append(m)
                references.append(r)
                pairs.append(dict(pair_id=identifier, linf=m['max_abs'],
                                  original_linf=r['max_abs'], linf_excess=excess))
            m, r = _aggregate(measurements), _aggregate(references)
            snr = m['snr_db']
            if math.isnan(snr) or snr == float('-inf'):
                raise ValueError('Invalid digital SNR')
            rows.append(dict(method_key=cell['method_key'], dose=dose, n_samples=len(ids),
                relative_l2=m['relative_l2'], rmse=m['rms'], linf=m['max_abs'],
                snr_db=snr if math.isfinite(snr) else None,
                snr_db_is_infinite=snr == float('inf'), original_linf=r['max_abs'],
                peak_bound_required=cell['method_key'] != 'w_o_peak',
                paired_linf_violations=sum(p['linf_excess'] > 0 for p in pairs),
                max_paired_linf_excess=max(p['linf_excess'] for p in pairs),
                cfg_fingerprint=config_fingerprint(cell['cfg']),
                trigger_state_sha256=state_sha, common_pair_ids_sha256=_sha_json(ids)))
            paired.append(dict(method_key=cell['method_key'], dose=dose, samples=pairs))
        if trigger_state_sha256(trigger) != state_sha:
            raise ValueError('Fixed trigger changed during distortion measurement')
    if (trigger_state_sha256(original) != original_sha
            or _file_sha(cfg['action_npy']) != audit['action_file_sha256']):
        raise ValueError('Original/action reference changed during distortion measurement')
    report = dict(profile=PROFILE, plan_sha256=matrix['plan_sha256'],
        common_pair_ids=ids, rows=rows, paired_samples=paired,
        units='dimensionless normalized digital CSI; SNR dB',
        aggregation='pooled signal/error energy; Linf is a global maximum',
        matches_l2_budget=False, original_trigger_state_sha256=original_sha,
        detection_test_performed=False, ota_test_performed=False)
    json.dumps(report, allow_nan=False)
    return report


def _write_text(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)


def export_summary(matrix, outdir):
    report = build_summary(matrix)
    distortion = build_distortion(matrix, report['audit'])
    outdir = Path(outdir)
    _atomic_json(outdir / 'ablation_summary.json', report)
    _write_csv(outdir / 'ablation_summary.csv', report['rows'], SUMMARY_COLUMNS)
    _atomic_json(outdir / 'dose_response.json', dict(profile=PROFILE,
        plan_sha256=matrix['plan_sha256'], rows=report['dose_response']))
    _write_csv(outdir / 'dose_response.csv', report['dose_response'], DOSE_COLUMNS)
    _atomic_json(outdir / 'input_distortion.json', distortion)
    _write_csv(outdir / 'input_distortion.csv', distortion['rows'], DISTORTION_COLUMNS)
    lines = ['# Full MM-Fi Proposed ablation', '',
        f"Seed {matrix['seed']}; 50 epochs; rho=.4; four fresh victims; full protocol1-s1.", '',
        '| Variant | T-MPJPE@1 (mm) | Dose-control error (mm) | Clean MPJPE (mm) |',
        '| --- | ---: | ---: | ---: |']
    for row in report['rows']:
        lines.append(f"| {row['method']} | {row['t1_mpjpe_mm']:.3f} | "
                     f"{row['mean_dose_control_error_mm']:.3f} | {row['clean_mpjpe_mm']:.3f} |")
    lines.extend(['', 'Dose-control error averages target-joint displacement-vector error over samples and five positive doses, not normalized schedule MAD.',
        'w/o Auxiliary carrier uses the Original unit-RMS pattern. Peak ceiling is unchanged but realized L2 is not matched.',
        'w/o Peak bound has a different realized budget; inspect input_distortion.csv before interpreting its target-error gain.',
        'All six doses are evaluated with each same trained victim; no dose-specific retraining.',
        f"Status: `{report['status']}`; Plan SHA256: `{matrix['plan_sha256']}`.", '',
        'Limitations:', *[f'- {item}' for item in report['limitations']]])
    _write_text(outdir / 'ablation_summary.md', '\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for cell in matrix['cells']:
        values = [r for r in report['dose_response'] if r['method_key'] == cell['method_key']]
        axes[0].plot(GRID, [r['triggered_tmpjpe_mm'] for r in values], marker='o', label=cell['label'])
        axes[1].plot(GRID, [r['dose_control_error_mm'] for r in values], marker='o', label=cell['label'])
    for axis, title in zip(axes, ('Target error (mm)', 'Dose-control error (mm)')):
        axis.set(xlabel='Test dose', ylabel=title)
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Full MM-Fi ablation; seed {matrix['seed']}; rho=.4")
    fig.tight_layout()
    fig.savefig(outdir / 'dose_response.png', dpi=160)
    plt.close(fig)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--distortion-samples', type=int, default=256)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary-only', '--export-only', action='store_true')
    parser.add_argument('--_cell', help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args._cell:
        matrix = json.loads(args._manifest.read_text(encoding='utf-8'))
        _validate_manifest(matrix)
        if matrix['sources']['source_sha256'] != _source_provenance():
            raise ValueError('Ablation source changed after plan creation')
        cell = next((c for c in matrix['cells'] if c['method_key'] == args._cell), None)
        if cell is None:
            raise ValueError('Unknown ablation cell')
        _run_cell(cell)
        return 0
    if args.data_home is None:
        raise ValueError('--data-home is required')
    _integer(args.seed, 'seed')
    _integer(args.num_workers, 'num_workers')
    _integer(args.distortion_samples, 'distortion_samples', 1)
    if (not args.devices or len(set(args.devices)) != len(args.devices)
            or any(d != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None for d in args.devices)):
        raise ValueError('--devices must list distinct logical cuda:N devices, or cpu')
    outdir = (args.outdir or args.data_home / 'runs' /
              f'mmfi_proposed_ablation_s{args.seed}_v1').resolve()
    if args.fresh and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('--fresh refuses a non-empty output directory; use a NEW path')
    with _cell_lock(outdir):
        path = outdir / 'proposed_ablation.resolved.json'
        requested = build_matrix(args.data_home, outdir, args.devices[0],
            args.num_workers, seed=args.seed, distortion_samples=args.distortion_samples)
        _validate_manifest(requested)
        if path.exists():
            matrix = json.loads(path.read_text(encoding='utf-8'))
            _validate_manifest(matrix)
            if matrix['plan_sha256'] != requested['plan_sha256']:
                raise ValueError('Existing ablation differs; source/config changes require a NEW directory')
            for cell in matrix['cells']:
                cell['cfg'].update(device=args.devices[0], num_workers=args.num_workers)
        else:
            if any(p.name != 'run.lock' for p in outdir.iterdir()):
                raise ValueError('Output directory contains unrelated files; use a NEW directory')
            matrix = requested
        _atomic_json(path, matrix)
        if args.dry_run:
            print(f'[{STATUS}] DRY RUN: seed {args.seed}; rho=.4; 50 epochs; four full independent cells; NO data/GPU/training.\n{path}', flush=True)
            return 0
        if not args.summary_only:
            _check_inputs(matrix, args.devices)
            run_matrix(matrix, path, args.devices, worker_script=Path(__file__))
        if any(not _complete(cell) for cell in matrix['cells']):
            raise ValueError('All four full cells must finish before exporting the ablation summary')
        export_summary(matrix, outdir)
        print(f'[ablation] Complete -> {outdir / "ablation_summary.md"}', flush=True)
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, AssertionError) as error:
        print(f'[ablation] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
