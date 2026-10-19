"""Replayable T5 reporting, separate from the immutable v1 evaluator.

Distances in cached NPZs are millimetres, PCK is percent of target joints.
No model, dataset, CUDA, or historical artifact is modified by these helpers.
"""
from __future__ import annotations

import copy
import math

import numpy as np

GRID = (0., .2, .4, .6, .8, 1.)
DEFENSES = ('none', 'ft', 'anp', 'nad', 'ft_sam')
MAIN_METRICS = ('t_mpjpe_mm', 'target_pck_pct', 'clean_mpjpe_mm',
                'triggered_gt_mpjpe_mm')
METRICS = MAIN_METRICS + ('clean_target_mpjpe_mm', 'clean_pampjpe_mm',
    'no_trigger_target_pck_pct', 'no_trigger_t_mpjpe_mm',
    'mean_positive_dce_mm', 'triggered_gt_minus_clean_target_mm')
DELTA_NAMES = {m: (m.removesuffix('_pct') + '_delta_pp' if m.endswith('_pck_pct')
                   else m + '_delta') for m in METRICS}


def _scalar(value, name, *, signed=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be a finite scalar')
    result = float(value)
    if not math.isfinite(result) or (not signed and result < 0):
        raise ValueError(f'Invalid metric {name}')
    if 'pck' in name and not 0 <= result <= 100:
        raise ValueError(f'PCK percent outside [0,100]: {name}')
    return result


def _same(value, expected, name, tolerance=.05):
    value = _scalar(value, name, signed=name == 'triggered_gt_minus_clean_target_mm')
    if abs(value - expected) > tolerance:
        raise ValueError(f'Cached {name} does not replay from paired sample errors')


def _array(arrays, name, shape=None):
    if name not in arrays:
        raise ValueError(f'Missing per-sample array {name}')
    value = np.asarray(arrays[name])
    if (not np.issubdtype(value.dtype, np.number)
            or np.issubdtype(value.dtype, np.complexfloating)
            or not np.isfinite(value).all() or (value < 0).any()
            or (shape is not None and value.shape != shape)):
        raise ValueError(f'Invalid shape/value in {name}')
    return value.astype(np.float64, copy=False)


def enrich_evaluation(evaluated):
    """Add same-joint clean floor and controls, replaying cached metric scalars.

    All-joint triggered errors/PA cannot be reconstructed from two-joint NPZs;
    this function does not invent them. Clean PA is the original evaluator's
    diagnostic, not a recomputation from norms. A negative recovery gap is valid.
    """
    d = evaluated.get('diagnostics', {})
    arrays = evaluated.get('arrays', {})
    clean = _array(arrays, 'clean_error_mm')
    if clean.ndim != 2 or clean.shape[1] != 17 or not len(clean):
        raise ValueError('Clean errors must have shape (N,17)')
    n = len(clean)
    if (d.get('samples') != n or d.get('target_joints') != [2, 3]
            or d.get('target_pck_threshold') != .2 or d.get('main_dose') != 1.
            or d.get('alignment') != 'none' or d.get('dose_grid') != list(GRID)
            or d.get('scale') != 'norm(clean_true_pose[5]-clean_true_pose[12])'
            or d.get('selection_on_test') is not False):
        raise ValueError('Frozen T5 metric protocol or sample count changed')
    doses = _array(arrays, 'doses', (6,))
    if not np.array_equal(doses, GRID):
        raise ValueError('NPZ dose order differs')
    scale = _array(arrays, 'clean_scale_mm', (n,))
    if (scale <= 0).any():
        raise ValueError('Clean GT scale must be positive')
    target = _array(arrays, 'target_error_mm', (n, 6, 2))
    gt = _array(arrays, 'triggered_gt_error_mm', (n, 6, 2))
    no_trigger = _array(arrays, 'no_trigger_target_error_mm', (n, 6, 2))
    dce = _array(arrays, 'dce_error_mm', (n, 6, 2))
    for name, a in (('target', target), ('GT', gt), ('no-trigger', no_trigger)):
        if not np.allclose(a[:, 0], clean[:, [2, 3]], atol=.002, rtol=0):
            raise ValueError(f'Dose-zero {name} errors differ from clean target-joint errors')
    if not np.allclose(dce[:, 0], 0., atol=.002, rtol=0):
        raise ValueError('Dose-zero DCE must be zero')
    pck = lambda e: float(100. * np.mean(e <= .2 * scale[:, None]))
    rows = evaluated.get('dose_rows', [])
    if len(rows) != 6 or [r.get('d') for r in rows] != list(GRID):
        raise ValueError('Missing or reordered six-dose rows')
    result = copy.deepcopy(evaluated)
    for i, row in enumerate(rows):
        replay = dict(t_mpjpe_mm=float(target[:, i].mean()),
            target_pck_pct=pck(target[:, i]), triggered_gt_mpjpe_mm=float(gt[:, i].mean()),
            dce_mm=float(dce[:, i].mean()), no_trigger_tmpjpe_mm=float(no_trigger[:, i].mean()))
        for name, expected in replay.items():
            _same(row.get(name), expected, name, 1.e-6 if 'pck' in name else .05)
        result['dose_rows'][i].update(
            no_trigger_target_pck_pct=pck(no_trigger[:, i]),
            triggered_gt_minus_clean_target_mm=float(gt[:, i].mean() - clean[:, [2, 3]].mean()))
    replay_main = dict(t_mpjpe_mm=float(target[:, -1].mean()),
        target_pck_pct=pck(target[:, -1]), clean_mpjpe_mm=float(clean.mean()),
        triggered_gt_mpjpe_mm=float(gt[:, -1].mean()))
    for name, expected in replay_main.items():
        _same(evaluated.get('metrics', {}).get(name), expected, name,
              1.e-6 if 'pck' in name else .05)
    derived = dict(clean_target_mpjpe_mm=float(clean[:, [2, 3]].mean()),
        clean_pampjpe_mm=_scalar(d.get('clean_pampjpe_mm'), 'clean_pampjpe_mm'),
        no_trigger_target_pck_pct=pck(no_trigger[:, -1]),
        no_trigger_t_mpjpe_mm=float(no_trigger[:, -1].mean()),
        mean_positive_dce_mm=float(dce[:, 1:].mean()),
        triggered_gt_minus_clean_target_mm=float(gt[:, -1].mean() - clean[:, [2, 3]].mean()))
    for name in ('no_trigger_target_pck_pct', 'no_trigger_t_mpjpe_mm', 'mean_positive_dce_mm'):
        _same(d.get(name), derived[name], name, 1.e-6 if 'pck' in name else .05)
    for name, value in derived.items():
        if name in result['metrics']:
            _same(result['metrics'][name], value, name, 1.e-6 if 'pck' in name else .05)
    # Preserve source values that already replay; don't rewrite historical rounding.
    result['metrics'].update(derived)
    result['diagnostics']['reporting_extension'] = 'T5 v2 array-replayed, not a new model evaluation'
    return result


def _aggregate(rows, fields, seeds, defenses):
    result = []
    for defense in defenses:
        sample = sorted((r for r in rows if r['defense'] == defense), key=lambda r: r['seed'])
        seed_list = [r['seed'] for r in sample]
        complete = seed_list == list(seeds)
        item = dict(method='a1', defense=defense, seeds=seed_list,
                    n_seeds=len(sample), complete=complete)
        for field in fields:
            values = [r[field] for r in sample]
            item[field + '_mean'] = float(np.mean(values)) if values else None
            item[field + '_sd'] = float(np.std(values, ddof=1)) if complete and len(values) == 3 else None
        result.append(item)
    return result


def build_metric_report(records, expected_seeds=(42, 43, 44), expected_defenses=DEFENSES):
    """Three-seed metrics and repair-minus-no-defense paired changes.

    Caller authenticates record/source/NPZ hashes before aggregation. Frames are
    never counted as independent seeds; a partial row has no advertised SD.
    """
    seeds = tuple(expected_seeds)
    defenses = tuple(expected_defenses)
    if (seeds != (42, 43, 44) or not defenses or 'none' not in defenses
            or len(set(defenses)) != len(defenses) or set(defenses) - set(DEFENSES)):
        raise ValueError('Report requires the frozen three seeds and known unique defenses including none')
    index, per_seed, doses = {}, [], []
    for record in records:
        seed, defense = record.get('seed'), record.get('defense')
        if (type(seed) is not int or seed not in seeds or defense not in defenses
                or record.get('method') != 'a1' or (seed, defense) in index):
            raise ValueError('Duplicate or out-of-plan seed/method/defense record')
        values = {m: _scalar(record.get('metrics', {}).get(m), m,
                            signed=m == 'triggered_gt_minus_clean_target_mm') for m in METRICS}
        _same(values['triggered_gt_minus_clean_target_mm'],
              values['triggered_gt_mpjpe_mm'] - values['clean_target_mpjpe_mm'],
              'triggered_gt_minus_clean_target_mm')
        row = dict(seed=seed, method='a1', defense=defense, **values)
        index[seed, defense] = row
        per_seed.append(row)
        if [r.get('d') for r in record.get('dose_rows', [])] != list(GRID):
            raise ValueError('Record lacks full six-dose diagnostics')
        for r in record['dose_rows']:
            dose_row = copy.deepcopy(r)
            for key in ('t_mpjpe_mm', 'target_pck_pct', 'triggered_gt_mpjpe_mm',
                        'dce_mm', 'no_trigger_tmpjpe_mm'):
                _scalar(dose_row.get(key), key)
            doses.append(dict(seed=seed, method='a1', defense=defense, **dose_row))
    per_seed.sort(key=lambda r: (r['seed'], defenses.index(r['defense'])))
    doses.sort(key=lambda r: (r['seed'], defenses.index(r['defense']), r['d']))
    paired = []
    for row in per_seed:
        base = index.get((row['seed'], 'none'))
        if row['defense'] == 'none' or base is None:
            continue
        delta = {DELTA_NAMES[m]: row[m] - base[m] for m in METRICS}
        paired.append(dict(seed=row['seed'], method='a1', defense=row['defense'], **delta))
    complete = len(index) == len(seeds) * len(defenses)
    return dict(status=('T5_V2_COMPLETE_THREE_SEEDS_HPE_ADAPTATION' if complete
                        else 'T5_V2_PARTIAL_NOT_COMPLETE_THREE_SEED_TABLE'),
        complete=complete, expected_records=len(seeds) * len(defenses), complete_records=len(index),
        missing=[f's{s}__a1__{d}' for s in seeds for d in defenses if (s, d) not in index],
        per_seed=per_seed, six_dose=doses, paired_deltas=paired,
        aggregates=_aggregate(per_seed, METRICS, seeds, defenses),
        paired_aggregates=_aggregate(paired, tuple(DELTA_NAMES.values()), seeds,
                                     tuple(d for d in defenses if d != 'none')),
        units=dict(distance='mm', target_pck='percent target joints',
                   target_pck_delta='percentage points; repair minus paired none'),
        uncertainty='sample SD(ddof=1) across three independent seed-level results only',
        selection='clean TRAIN selection only; never choose defense/checkpoint from TEST',
        limitations=['Target-PCK is joint-level target matching, not pose-level ASR.',
            'Clean MPJPE is all 17 joints; Triggered-GT and clean-target errors use joints [2,3].',
            'Recovery gap is triggered-GT minus same-model clean-target error; negative values are not clipped.',
            'Clean PA is inherited from the audited evaluator; NPZ norms alone cannot reproduce alignment.',
            'DCE is an attack-control diagnostic, not a lower-is-better defense score.',
            'No claim of relative defense resistance against unevaluated attack baselines.'])
