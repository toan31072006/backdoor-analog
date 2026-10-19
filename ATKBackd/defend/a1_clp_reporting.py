"""Six-row T5 report with an explicitly separate data-free CLP sidecar.

Only authenticated records belong here. Existing v1/v2 code and reports remain
unchanged; a CLP-only preview is not a complete three-seed defense table.
"""
from __future__ import annotations

import copy

from defend.a1_defense_metrics import (GRID, MAIN_METRICS, METRICS, DELTA_NAMES,
                                       _aggregate, _same, _scalar)

SEEDS = (42, 43, 44)
DEFENSES = ('none', 'ft', 'anp', 'nad', 'ft_sam', 'clp')
LABELS = dict(none='No defense', ft='FT', anp='ANP (HPE-adapted)', nad='NAD',
              ft_sam='FT-SAM', clp='CLP (HPE-adapted)')


def build_metric_report(records):
    """Aggregate seed-level scalars and within-seed repair-minus-none deltas."""
    index, per_seed, doses = {}, [], []
    for record in records:
        seed, defense = record.get('seed'), record.get('defense')
        if (type(seed) is not int or seed not in SEEDS or defense not in DEFENSES
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
    per_seed.sort(key=lambda r: (r['seed'], DEFENSES.index(r['defense'])))
    doses.sort(key=lambda r: (r['seed'], DEFENSES.index(r['defense']), r['d']))
    paired = []
    for row in per_seed:
        base = index.get((row['seed'], 'none'))
        if row['defense'] != 'none' and base is not None:
            paired.append(dict(seed=row['seed'], method='a1', defense=row['defense'],
                **{DELTA_NAMES[m]: row[m] - base[m] for m in METRICS}))
    complete = len(index) == 18
    return dict(status=('T5_WITH_CLP_COMPLETE_THREE_SEEDS_HPE_ADAPTATIONS' if complete
                        else 'T5_WITH_CLP_PARTIAL_NOT_COMPLETE_THREE_SEED_TABLE'),
        complete=complete, expected_records=18, complete_records=len(index),
        missing=[f's{s}__a1__{d}' for s in SEEDS for d in DEFENSES if (s, d) not in index],
        per_seed=per_seed, six_dose=doses, paired_deltas=paired,
        aggregates=_aggregate(per_seed, METRICS, SEEDS, DEFENSES),
        paired_aggregates=_aggregate(paired, tuple(DELTA_NAMES.values()), SEEDS, DEFENSES[1:]),
        units=dict(distance='mm', target_pck='percent target joints',
                   target_pck_delta='percentage points; repair minus paired none'),
        uncertainty='sample SD(ddof=1) across three independent seed-level results only',
        selection='CLP u=3 fixed before TEST; no clean fit/selection or strength tuning',
        limitations=['CLP is data-free; FT/ANP/NAD/FT-SAM use the disclosed clean TRAIN bank.',
            'Defense compute and data budgets are not matched.',
            'CLP covers direct Conv2d-BatchNorm2d pairs only; composite SK and BN1d exclusions are declared.',
            'Target-PCK is joint-level target matching, not pose-level ASR.',
            'Clean MPJPE is all 17 joints; Triggered-GT and clean-target errors use joints [2,3].',
            'Recovery gap is triggered-GT minus same-model clean-target error, not an independent clean-model floor.',
            'DCE is an attack-control diagnostic, not a lower-is-better defense score.',
            'Historical Fine-Pruning pilot results are preserved; revised defenses were selected after the pilot.',
            'No claim of resistance relative to unevaluated attack baselines.'])
