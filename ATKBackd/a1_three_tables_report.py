"""Validated three-seed exports for the frozen MM-Fi T1/T2/T3 study.

This module only aggregates completed scalar/curve records. It never loads a
dataset or checkpoint, invokes training, chooses parameters, or creates plots.
The caller must establish scientific/configuration provenance before passing
records; complete records alone do not establish an equal-perturbation budget.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path
from statistics import mean, stdev


SEEDS = (42, 43, 44)
GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BASELINES = ('badnets', 'wanet_source', 'ftrojan', 'fiba', 'sig_source', 'bpp_source')
KEYS = ('clean', *BASELINES, 'a1', 'a2', 'a3', 'a4', 'rho0p2', 'rho0p4')
LABELS = dict(clean='Clean reference', badnets='BadNets-CSI', wanet_source='WaNet-CSI',
    ftrojan='FTrojan-CSI', fiba='FIBA-CSI', sig_source='SIG-CSI',
    bpp_source='BppAttack-CSI', a1='Proposed (A1--Full)', a2='A2', a3='A3', a4='A4',
    rho0p2='A1 rho=0.2', rho0p4='A1 rho=0.4')
METRICS = ('t1_mpjpe_mm', 'target_pck_pct', 'clean_mpjpe_mm',
           'clean_pampjpe_mm', 'mean_dce_mm')
T1_METRICS = METRICS[:4]
T3_METRICS = ('t1_mpjpe_mm', 'mean_dce_mm', 'clean_mpjpe_mm')
DOSE_METRICS = ('tmpjpe_mm', 'dce_mm', 'no_trigger_tmpjpe_mm')
RHO_KEYS = (('a1', .1), ('rho0p2', .2), ('rho0p4', .4))
COMPLETION_FILE = 'three_main_tables.json'
STATUS = 'three_main_tables_complete_three_seeds'
REPLAY_TOLERANCE_MM = .05
DCE_TOLERANCE_MM = 1e-6


def _number(value, name, *, nullable=False, upper=None):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite nonnegative number')
    number = float(value)
    if not math.isfinite(number) or number < 0 or (upper is not None and number > upper):
        raise ValueError(f'{name} must be finite and in [0, {upper or "infinity"}]')
    return number


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _stats(values):
    if len(values) != len(SEEDS):
        raise ValueError('Every aggregate must contain all three seed observations')
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError('Cannot aggregate a mixture of missing and present metrics')
    return dict(mean=float(mean(values)), sample_sd=float(stdev(values)), n=len(values))


def _validated_records(records):
    try:
        records = list(records)
    except TypeError as exc:
        raise ValueError('records must contain the 39 completed cell records') from exc
    if len(records) != len(SEEDS) * len(KEYS):
        raise ValueError('Exactly 39 completed cells (13 per seed) are required')
    indexed = {}
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError('Each cell record must be a dictionary')
        seed, key = raw.get('seed'), raw.get('key')
        if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) not in SEEDS:
            raise ValueError('The exact seed set must be 42, 43, 44')
        seed = int(seed)
        if not isinstance(key, str) or key not in KEYS:
            raise ValueError(f'Unknown cell key: {key!r}')
        identity = (seed, key)
        if identity in indexed:
            raise ValueError(f'Duplicate cell: seed={seed}, key={key}')
        metrics = raw.get('metrics')
        if not isinstance(metrics, dict) or any(name not in metrics for name in METRICS):
            raise ValueError(f'{identity}: all five named metric fields are required')
        normalized = {}
        for name in METRICS:
            optional = (key == 'clean' and name in ('t1_mpjpe_mm', 'target_pck_pct', 'mean_dce_mm'))
            optional |= (key in ('a2', 'a3', 'a4', 'rho0p2', 'rho0p4') and name == 'target_pck_pct')
            if key == 'clean' and name in ('t1_mpjpe_mm', 'target_pck_pct', 'mean_dce_mm'):
                if metrics[name] is not None:
                    raise ValueError('Clean reference attack metrics must be null, not ranked')
            normalized[name] = _number(metrics[name], f'{identity}.{name}', nullable=optional,
                upper=100 if name == 'target_pck_pct' else None)
        doses = raw.get('dose_rows', [])
        if not isinstance(doses, (list, tuple)):
            raise ValueError(f'{identity}: dose_rows must be a list')
        curve = []
        if key == 'clean':
            if doses:
                raise ValueError('Clean trigger controls belong in diagnostics, not dose_rows')
        else:
            if len(doses) != len(GRID):
                raise ValueError(f'{identity}: exactly six dose rows are required')
            for row in doses:
                if not isinstance(row, dict) or any(name not in row for name in ('d', *DOSE_METRICS)):
                    raise ValueError(f'{identity}: malformed dose row')
                curve.append(dict(d=_number(row['d'], f'{identity}.d', upper=1),
                    **{name: _number(row[name], f'{identity}.{name}') for name in DOSE_METRICS}))
            curve.sort(key=lambda row: row['d'])
            if tuple(row['d'] for row in curve) != GRID:
                raise ValueError(f'{identity}: six doses must be exactly {GRID}, without duplicates')
            if abs(curve[0]['dce_mm']) > DCE_TOLERANCE_MM:
                raise ValueError(f'{identity}: identity-dose DCE must be zero')
            positive_mean = mean(row['dce_mm'] for row in curve[1:])
            if abs(positive_mean - normalized['mean_dce_mm']) > DCE_TOLERANCE_MM:
                raise ValueError(f'{identity}: mean_dce_mm must average five positive doses')
            if abs(curve[-1]['tmpjpe_mm'] - normalized['t1_mpjpe_mm']) > REPLAY_TOLERANCE_MM:
                raise ValueError(f'{identity}: T-MPJPE scalar disagrees with its d=1 dose row')
        record = copy.deepcopy(raw)
        record.update(seed=seed, key=key, metrics=normalized, dose_rows=curve,
            config_fingerprint=_text(raw.get('config_fingerprint'), 'config_fingerprint'))
        indexed[identity] = record
    expected = {(seed, key) for seed in SEEDS for key in KEYS}
    if set(indexed) != expected:
        raise ValueError('Every seed must contain every one of the 13 declared cell keys')
    ordered = [indexed[seed, key] for seed in SEEDS for key in KEYS]
    try:
        json.dumps(ordered, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('Record metadata and diagnostics must be finite JSON-compatible values') from exc
    return indexed, ordered


def build_report(records, *, plan_sha256, provenance=None):
    """Return validated per-seed data and mean/sample-SD T1/T2/T3 tables.

    Each record represents one scientific victim. The shared A1 cell supplies
    T1 Proposed, T2 rho=.1, and T3 Full without counting it as independent runs.
    No incomplete report is returned: all 39 unique seed/key cells are required.
    """
    plan_sha256 = _text(plan_sha256, 'plan_sha256')
    indexed, ordered = _validated_records(records)

    def metrics_for(key, names):
        return {name: _stats([indexed[seed, key]['metrics'][name] for seed in SEEDS])
                for name in names}

    baseline_order = sorted(BASELINES,
        key=lambda key: -mean(indexed[seed, key]['metrics']['t1_mpjpe_mm'] for seed in SEEDS))
    comparison = [dict(key=key, method=LABELS[key], **metrics_for(key, T1_METRICS))
                  for key in ('clean', *baseline_order, 'a1')]
    dose_matrix = []
    for i, dose in enumerate(GRID):
        dose_matrix.append(dict(d=dose, **{
            f'rho{str(rho).replace(".", "p")}': _stats([
                indexed[seed, key]['dose_rows'][i]['tmpjpe_mm'] for seed in SEEDS])
            for key, rho in RHO_KEYS}))
    poison_summary = [dict(key=key, rho=rho,
        **metrics_for(key, ('clean_mpjpe_mm', 'mean_dce_mm'))) for key, rho in RHO_KEYS]
    ablation = [dict(key=key, variant='A1--Full' if key == 'a1' else key.upper(),
        auxiliary_carrier=key in ('a1', 'a2'), dose_coupling=key in ('a1', 'a3'),
        **metrics_for(key, T3_METRICS)) for key in ('a1', 'a2', 'a3', 'a4')]
    dose_summary = [dict(key=key, method=LABELS[key], d=dose,
        **{name: _stats([indexed[seed, key]['dose_rows'][i][name] for seed in SEEDS])
           for name in DOSE_METRICS})
        for key in KEYS if key != 'clean' for i, dose in enumerate(GRID)]
    report = dict(schema=1, status=STATUS, table_ids=['T1', 'T2', 'T3'],
        plan_sha256=plan_sha256, seeds=list(SEEDS), seed_count=3,
        aggregation=dict(unit='one completed full-TEST scalar per independent training seed',
            summary='arithmetic mean and sample standard deviation across seeds', ddof=1,
            dce='mean five positive doses within each seed, then mean/sample SD across seeds',
            frame_replicates=False, missing_seeds_allowed=False),
        metric_definitions=dict(t1_mpjpe='target-joint error to attacker target at d=1, mm, no alignment',
            target_pck='percent of joints [2,3] within .2*||cleanGT[5]-cleanGT[12]|| of attacker target at d=1; not pose ASR',
            clean_mpjpe='all 17 joints on untriggered inputs against true pose, mm',
            clean_pampjpe='all 17 joints on untriggered inputs, per-pose Procrustes aligned, mm',
            dce='mean target-joint norm of (pred_d-pred_0)-(target_d-true_0), mm'),
        tables=dict(T1=dict(rows=comparison), T2=dict(dose_matrix=dose_matrix,
            poison_summary=poison_summary), T3=dict(rows=ablation)),
        dose_response_summary=dose_summary, records=ordered,
        provenance=copy.deepcopy(provenance), limitations=[
            'Native CSI/HPE adaptations do not match realized L2/Linf or clean-label cover counts.',
            'Target-PCK is a target-joint fraction, not pose-level ASR or clean PCK.',
            'The target changes with dose; T-MPJPE across doses alone does not establish calibration.',
            'Proposed was development-selected; three seed replicates do not establish a blinded test or p-value.',
            'Sample SD describes these three training seeds, not independent frame or subject uncertainty.',
            'The aggregator checks records, not underlying dataset contents or scientific protocol equivalence.',
        ])
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('provenance must be finite JSON-compatible data') from exc
    return report


def _flat(rows, identity_columns, metric_columns):
    flattened = []
    for row in rows:
        item = {name: row[name] for name in identity_columns}
        for name in metric_columns:
            stats = row[name]
            item.update({f'{name}_{suffix}': None if stats is None else stats[field]
                for suffix, field in (('mean', 'mean'), ('sample_sd', 'sample_sd'), ('n', 'n'))})
        flattened.append(item)
    columns = list(identity_columns) + [f'{name}_{suffix}' for name in metric_columns
                                        for suffix in ('mean', 'sample_sd', 'n')]
    return flattened, columns


def _atomic_text(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def _json(path, value):
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _csv(path, rows, columns):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, output.getvalue())


def _formatted(stats, *, latex=False, bold=False):
    if stats is None:
        return '---'
    value = f"{stats['mean']:.2f} ± {stats['sample_sd']:.2f}"
    if latex:
        value = f"${stats['mean']:.2f}\\,\\pm\\,{stats['sample_sd']:.2f}$"
        return r'{\boldmath\textbf{' + value + '}}' if bold else value
    return f'**{value}**' if bold else value


def _best(rows, name, *, higher=False):
    candidates = [row[name]['mean'] for row in rows if row['key'] != 'clean' and row[name] is not None]
    return (max if higher else min)(candidates)


def _render_t1(report):
    rows = report['tables']['T1']['rows']
    headers = ('Method', 'T-MPJPE ↓', 'Target-PCK ↑', 'Clean MPJPE ↓', 'Clean PA-MPJPE ↓')
    caption = ('MM-Fi protocol1-s1; seeds 42/43/44; 50 epochs; rho=0.1 (clean: 0). '
        'Mean ± sample SD across training seeds. Attack metrics use d=1; Target-PCK@0.2 is percent, '
        'all errors are mm. Clean reference is not attack-ranked; bold means best observed mean among targeted methods.')
    md = ['# T1: Comparison', '', caption, '', '| ' + ' | '.join(headers) + ' |',
          '| --- | ---: | ---: | ---: | ---: |']
    tex = [r'\begin{table*}[t]', r'\centering',
        r'\caption{Comparison on MM-Fi protocol1-s1; seeds 42, 43, 44; 50 epochs; $\rho=0.1$ (clean: 0). '
        r'Mean $\pm$ sample SD over seeds. T-MPJPE is in mm at $d=1$; Target-PCK uses threshold 0.2 at $d=1$ and is in percent. '
        r'Clean errors are in mm. Bold marks the best observed targeted-method mean, not statistical significance.}',
        r'\label{tab:comparison}', r'\small', r'\setlength{\tabcolsep}{5pt}',
        r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Method & T-MPJPE $\downarrow$ & Target-PCK $\uparrow$ & Clean MPJPE $\downarrow$ & Clean PA-MPJPE $\downarrow$ \\',
        r'\midrule']
    for row in rows:
        flags = {name: row['key'] != 'clean' and row[name] is not None
                 and math.isclose(row[name]['mean'], _best(rows, name, higher=name == 'target_pck_pct'),
                                  rel_tol=0, abs_tol=1e-12) for name in T1_METRICS}
        md.append('| ' + row['method'] + ' | ' + ' | '.join(
            _formatted(row[name], bold=flags[name]) for name in T1_METRICS) + ' |')
        tex.append(row['method'] + ' & ' + ' & '.join(
            _formatted(row[name], latex=True, bold=flags[name]) for name in T1_METRICS) + r' \\')
        if row['key'] == 'clean':
            tex.append(r'\midrule')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(md) + '\n', '\n'.join(tex) + '\n'


def _render_t2(report):
    table = report['tables']['T2']
    rho_columns = ('rho0p1', 'rho0p2', 'rho0p4')
    md = ['# T2: Dose x poison rate', '', 'T-MPJPE (mm) per cell; mean ± sample SD over seeds 42/43/44.',
          'One 50-epoch victim per seed/rho is evaluated at all six doses. The target changes with dose.', '',
          '| Dose | rho=0.1 | rho=0.2 | rho=0.4 |', '| --- | ---: | ---: | ---: |']
    summary_text = '; '.join(f"rho={row['rho']:g}: clean MPJPE {_formatted(row['clean_mpjpe_mm'])} mm, "
        f"mean positive-dose DCE {_formatted(row['mean_dce_mm'])} mm" for row in table['poison_summary'])
    summary_tex = '; '.join(f"$\\rho={row['rho']:g}$: clean MPJPE {_formatted(row['clean_mpjpe_mm'], latex=True)}, "
        f"mean DCE {_formatted(row['mean_dce_mm'], latex=True)} mm" for row in table['poison_summary'])
    tex = [r'\begin{table}[t]', r'\centering', r'\caption{Dose $d$ by poison rate $\rho$ for A1--Full. '
        r'Each cell is T-MPJPE in mm (lower is better), mean $\pm$ sample SD over seeds 42, 43, 44. '
        r'Each column uses one fixed victim per seed. The target changes across dose rows.}',
        r'\label{tab:dose-poison}', r'\small', r'\setlength{\tabcolsep}{5pt}',
        r'\begin{tabular}{l|rrr}', r'\toprule',
        r' & $\rho=10\%$ & $\rho=20\%$ & $\rho=40\%$ \\', r'\midrule']
    for row in table['dose_matrix']:
        md.append(f"| d={row['d']:.1f} | " + ' | '.join(_formatted(row[name]) for name in rho_columns) + ' |')
        tex.append(f"$d={row['d']:.1f}$ & " + ' & '.join(_formatted(row[name], latex=True) for name in rho_columns) + r' \\')
    md += ['', 'Poison-rate summaries (not additional dose-matrix rows):', '', summary_text + '.',
           'Mean DCE excludes d=0 and is computed within each seed before cross-seed aggregation.']
    tex += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip', r'\begin{minipage}{\linewidth}\footnotesize',
            summary_tex + '.', r'Mean DCE excludes $d=0$; summaries are formed within seed before aggregation.',
            r'\end{minipage}', r'\end{table}']
    return '\n'.join(md) + '\n', '\n'.join(tex) + '\n'


def _render_t3(report):
    rows = report['tables']['T3']['rows']
    md = ['# T3: Factorial component study', '',
          'MM-Fi; rho=0.1; 50 epochs; mean ± sample SD over seeds 42/43/44. All errors are mm.', '',
          '| Variant | Auxiliary carrier | Dose coupling | T-MPJPE ↓ | DCE ↓ | Clean MPJPE ↓ |',
          '| --- | :---: | :---: | ---: | ---: | ---: |']
    tex = [r'\begin{table*}[t]', r'\centering',
        r'\caption{Factorial component study; $\rho=0.1$; 50 epochs; seeds 42, 43, 44. '
        r'All values are mm, mean $\pm$ sample SD across seeds. T-MPJPE uses $d=1$; DCE averages five positive doses.}',
        r'\label{tab:ablation}', r'\small', r'\setlength{\tabcolsep}{2pt}',
        r'\begin{tabular}{lccrrr}', r'\toprule',
        r' & \shortstack{Auxiliary\\carrier} & \shortstack{Dose\\coupling} & T-MPJPE $\downarrow$ & DCE $\downarrow$ & \shortstack{Clean\\MPJPE $\downarrow$} \\',
        r'\midrule']
    for row in rows:
        flags = {name: math.isclose(row[name]['mean'], _best(rows, name), rel_tol=0, abs_tol=1e-12)
                 for name in T3_METRICS}
        md.append('| ' + row['variant'] + ' | ' + ('✓' if row['auxiliary_carrier'] else '×') + ' | '
            + ('✓' if row['dose_coupling'] else '×') + ' | '
            + ' | '.join(_formatted(row[name], bold=flags[name]) for name in T3_METRICS) + ' |')
        tex.append(row['variant'] + ' & ' + (r'$\checkmark$' if row['auxiliary_carrier'] else r'$\times$')
            + ' & ' + (r'$\checkmark$' if row['dose_coupling'] else r'$\times$') + ' & '
            + ' & '.join(_formatted(row[name], latex=True, bold=flags[name]) for name in T3_METRICS) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(md) + '\n', '\n'.join(tex) + '\n'


def export_reports(records, outdir, *, plan_sha256, provenance=None):
    """Export three complete tables plus raw per-seed scalars and dose curves.

    CSVs retain unrounded means/SDs; text/LaTeX use two decimals. The combined
    completion JSON is written last, only after every requested export succeeds.
    No output directory is created if record validation fails.
    """
    report = build_report(records, plan_sha256=plan_sha256, provenance=provenance)
    outdir = Path(outdir)
    existing = outdir / COMPLETION_FILE
    if existing.exists():
        old = json.loads(existing.read_text(encoding='utf-8'))
        if old.get('plan_sha256') != plan_sha256:
            raise ValueError('Existing report belongs to a different plan; use a new output directory')
    outdir.mkdir(parents=True, exist_ok=True)
    exports = []
    for table_id, stem, rows, identities, names, renderer in (
            ('T1', 'table1_comparison', report['tables']['T1']['rows'], ('key', 'method'), T1_METRICS, _render_t1),
            ('T2', 'table2_dose_poison', report['tables']['T2']['dose_matrix'], ('d',),
             ('rho0p1', 'rho0p2', 'rho0p4'), _render_t2),
            ('T3', 'table3_ablation', report['tables']['T3']['rows'],
             ('key', 'variant', 'auxiliary_carrier', 'dose_coupling'), T3_METRICS, _render_t3)):
        flat, columns = _flat(rows, identities, names)
        md, tex = renderer(report)
        _csv(outdir / (stem + '.csv'), flat, columns)
        _json(outdir / (stem + '.json'), dict(table_id=table_id, plan_sha256=plan_sha256,
            seeds=list(SEEDS), aggregation=report['aggregation'], **report['tables'][table_id]))
        _atomic_text(outdir / (stem + '.md'), md)
        _atomic_text(outdir / (stem + '.tex'), tex)
        exports.extend(stem + suffix for suffix in ('.csv', '.json', '.md', '.tex'))
    flat, columns = _flat(report['tables']['T2']['poison_summary'], ('key', 'rho'),
                          ('clean_mpjpe_mm', 'mean_dce_mm'))
    _csv(outdir / 'table2_poison_summary.csv', flat, columns)
    exports.append('table2_poison_summary.csv')
    scalar_rows = [dict(seed=record['seed'], key=record['key'], method=LABELS[record['key']],
        config_fingerprint=record['config_fingerprint'], **record['metrics']) for record in report['records']]
    scalar_columns = ('seed', 'key', 'method', 'config_fingerprint', *METRICS)
    _csv(outdir / 'per_seed_metrics.csv', scalar_rows, scalar_columns)
    exports.append('per_seed_metrics.csv')
    for seed in SEEDS:
        filename = f'per_seed_metrics_s{seed}.csv'
        _csv(outdir / filename, [row for row in scalar_rows if row['seed'] == seed], scalar_columns)
        exports.append(filename)
    dose_rows = [dict(seed=record['seed'], key=record['key'], method=LABELS[record['key']], **row)
                 for record in report['records'] for row in record['dose_rows']]
    _csv(outdir / 'dose_response_per_seed.csv', dose_rows, ('seed', 'key', 'method', 'd', *DOSE_METRICS))
    flat, columns = _flat(report['dose_response_summary'], ('key', 'method', 'd'), DOSE_METRICS)
    _csv(outdir / 'dose_response_summary.csv', flat, columns)
    exports += ['dose_response_per_seed.csv', 'dose_response_summary.csv']
    report['exported_files'] = exports + [COMPLETION_FILE]
    _json(existing, report)
    return report
