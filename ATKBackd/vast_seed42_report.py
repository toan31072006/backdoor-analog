"""Four table families for one seed, with historical HPE-Li kept separate."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

BASELINES = ('badnets', 'wanet_source', 'ftrojan', 'fiba', 'sig_source', 'bpp_source')
LABELS = dict(clean='Clean reference', a1='A1--Full', badnets='BadNets-CSI',
    wanet_source='WaNet-CSI', ftrojan='FTrojan-CSI', fiba='FIBA-CSI',
    sig_source='SIG-CSI', bpp_source='BppAttack-CSI', a2='A2', a3='A3', a4='A4')
METRICS = ('t1_mpjpe_mm', 'target_pck_pct', 'clean_mpjpe_mm', 'clean_pampjpe_mm', 'mean_dce_mm')
DEFENSES = ('none', 'ft', 'anp', 'nad', 'clp')
D_METRICS = ('t_mpjpe_mm', 'target_pck_pct', 'clean_mpjpe_mm', 'triggered_gt_mpjpe_mm')
GRID = (0., .2, .4, .6, .8, 1.)


def _number(value, name, *, nullable=False, signed=False):
    if value is None and nullable:
        return value
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'Invalid metric {name}')
    if not signed and value < 0 or 'pck' in name and not 0 <= value <= 100:
        raise ValueError(f'Out-of-range metric {name}')
    return float(value)


def build_tables(attack_records, defense_records, *, backbone, provenance,
                 experiment_scope='full'):
    """Validation never invents missing runs or across-seed uncertainty."""
    from a1_three_tables_report import KEYS
    if type(experiment_scope) is not str or experiment_scope not in ('full', 'attack-defense'):
        raise ValueError('Experiment scope must be full or attack-defense')
    expected_keys = KEYS if experiment_scope == 'full' else ('clean', *BASELINES, 'a1')
    attacks = {r['key']: r for r in attack_records}
    defenses = {r['defense']: r for r in defense_records}
    if len(attacks) != len(attack_records) or set(attacks) != set(expected_keys):
        raise ValueError(f'Each backbone requires exactly{len(expected_keys)} unique attack victims for {experiment_scope}')
    if len(defenses) != len(defense_records) or set(defenses) != set(DEFENSES):
        raise ValueError('Each fresh backbone requires No-defense plus all four defenses')
    for key, record in attacks.items():
        if record.get('seed') != 42 or record.get('status') != 'complete':
            raise ValueError('Only authenticated complete seed42 records are reportable')
        metrics = record['metrics']
        for name in METRICS:
            _number(metrics[name], name, nullable=key == 'clean' and name in ('t1_mpjpe_mm', 'target_pck_pct', 'mean_dce_mm'))
        if key == 'clean':
            if any(metrics[n] is not None for n in ('t1_mpjpe_mm', 'target_pck_pct', 'mean_dce_mm')):
                raise ValueError('Clean reference must not be attack-ranked')
            continue
        rows = record['dose_rows']
        if tuple(r['d'] for r in rows) != GRID:
            raise ValueError('Six ordered test doses required')
        for row in rows:
            for name in ('tmpjpe_mm', 'dce_mm', 'no_trigger_tmpjpe_mm'):
                _number(row[name], name)
        if abs(rows[0]['dce_mm']) > 1e-6 or abs(sum(r['dce_mm'] for r in rows[1:]) / 5 - metrics['mean_dce_mm']) > 1e-6:
            raise ValueError('DCE must average five positive doses')
        if abs(rows[-1]['tmpjpe_mm'] - metrics['t1_mpjpe_mm']) > .05:
            raise ValueError('Dose1 target error differs from main scalar')
    for method, record in defenses.items():
        if record.get('seed') != 42 or record.get('status') != 'complete':
            raise ValueError('Only complete paired defense records allowed')
        for name in D_METRICS:
            _number(record['metrics'][name], name)
        if tuple(r['d'] for r in record['dose_rows']) != GRID:
            raise ValueError('Defense requires all six doses')
    # No-defense must be the actual paired A1, not another seed/checkpoint.
    for dname, aname in (('t_mpjpe_mm', 't1_mpjpe_mm'), ('target_pck_pct', 'target_pck_pct'),
                         ('clean_mpjpe_mm', 'clean_mpjpe_mm')):
        if abs(defenses['none']['metrics'][dname] - attacks['a1']['metrics'][aname]) > (.000001 if 'pck' in dname else .05):
            raise ValueError('No-defense does not reproduce the paired A1 victim')
    comparison = [dict(key=k, method=LABELS[k], **attacks[k]['metrics']) for k in ('clean', *BASELINES, 'a1')]
    defense = [dict(defense=k, **defenses[k]['metrics']) for k in DEFENSES]
    baseline = defenses['none']['metrics']
    paired = [dict(defense=k, **{name + ('_delta_pp' if 'pck' in name else '_delta_mm'):
        defenses[k]['metrics'][name] - baseline[name] for name in D_METRICS}) for k in DEFENSES]
    tables = dict(T1=comparison)
    if experiment_scope == 'full':
        tables['T2'] = [dict(d=d, **{k: attacks[k]['dose_rows'][i]['tmpjpe_mm'] for k in ('a1', 'rho0p2', 'rho0p4')})
                        for i, d in enumerate(GRID)]
        tables['T3'] = [dict(key=k, variant=LABELS[k], auxiliary_carrier=k in ('a1', 'a2'),
                            dose_coupling=k in ('a1', 'a3'), **attacks[k]['metrics']) for k in ('a1', 'a2', 'a3', 'a4')]
    tables['T5'] = defense
    result = dict(schema=1, status='COMPLETE_ONE_SEED_FOUR_TABLE_FAMILIES' if experiment_scope == 'full'
                  else 'COMPLETE_ONE_SEED_ATTACK_DEFENSE_TABLES', seed=42, n_seeds=1,
        backbone=backbone, uncertainty='not estimated; no across-seed SD/CI',
        provenance=provenance, tables=tables,
        defense_paired_changes=paired,
        attack_records=attack_records, defense_records=defense_records,
        limitations=['Digital CSI/HPE adaptations, not over-the-air/RF.',
            'Native attack operators have different realized perturbation budgets.',
            'One training seed cannot measure across-seed robustness.',
            'Target-PCK is target-joint matching, not pose ASR.',
            'Low Target-PCK after repair alone does not demonstrate true-pose recovery.',
            'DCE is attack calibration, not defender-success score.'])
    if experiment_scope == 'full':
        result['poison_summaries'] = [dict(key=k, **attacks[k]['metrics']) for k in ('a1', 'rho0p2', 'rho0p4')]
    else:
        result['experiment_scope'] = experiment_scope
        result['limitations'].append('Fresh backbones include T1/T5 only; six-dose diagnostics are not fresh T2/T3 studies.')
    return result


def _write_csv(path, rows, columns):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding='utf-8')


def _render(path, rows, columns, headers, title, note):
    def format_value(value):
        if value is None:
            return '--'
        if isinstance(value, bool):
            return 'yes' if value else 'no'
        return f'{value:.2f}' if isinstance(value, (int, float)) else str(value)
    md = [f'# {title}', '', note, '', '| ' + ' | '.join(headers) + ' |',
          '| ' + ' | '.join('---' for _ in columns) + ' |']
    tex = [r'\begin{table*}[t]', r'\centering\small', r'\caption{' + title + '. ' + note.replace('%', r'\%') + '}',
           r'\begin{tabular}{' + 'l' + 'r' * (len(columns) - 1) + '}', r'\toprule',
           ' & '.join(h.replace('%', r'\%').replace('_', r'\_') for h in headers) + r' \\', r'\midrule']
    for row in rows:
        values = [format_value(row.get(name)) for name in columns]
        md.append('| ' + ' | '.join(values) + ' |')
        tex.append(' & '.join(v.replace('_', r'\_').replace('%', r'\%') for v in values) + r' \\')
    tex.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
    path.with_suffix('.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    path.with_suffix('.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    _write_csv(path.with_suffix('.csv'), rows, columns)


def export(plan):
    from run_vast_seed42 import completed, atomic_json
    output = Path(plan['outdir']) / 'tables'
    experiment_scope = plan.get('experiment_scope', 'full')
    if type(experiment_scope) is not str or experiment_scope not in ('full', 'attack-defense'):
        raise ValueError('Experiment scope must be full or attack-defense')
    records = {task['id']: completed(plan, task) for task in plan['cells']}
    if not all(records.values()):
        raise ValueError('Incomplete batch: no requested-table completion marker is written')
    reports = []
    for backbone in plan['backbones']:
        attacks = [records[t['id']] for t in plan['cells'] if t['kind'] == 'attack' and t['cell'].startswith(backbone + '__')]
        defenses = [records[t['id']] for t in plan['cells'] if t['kind'] == 'defense' and t['cell'].startswith(backbone + '__')]
        report = build_tables(attacks, defenses, backbone=backbone,
            provenance=dict(origin='fresh_Vast', plan_sha256=plan['plan_sha256'], historical_reuse=False),
            experiment_scope=experiment_scope)
        folder = output / backbone
        if experiment_scope == 'attack-defense' and any((folder / name).exists() for name in
                ('four_tables.json', 'table2_dose_poison.csv', 'table2_dose_poison.md', 'table2_dose_poison.tex',
                 'table3_ablation.csv', 'table3_ablation.md', 'table3_ablation.tex', 'poison_summaries.csv')):
            raise ValueError('Existing full-scope table artifacts are preserved; choose a NEW output for attack-defense')
        folder.mkdir(parents=True, exist_ok=True)
        note = 'MM-Fi protocol1-s1, seed42, n=1; original attack victims use fixed final epoch49. No across-seed SD/CI. Errors in mm.'
        _render(folder / 'table1_comparison', report['tables']['T1'], ('method', *METRICS),
            ('Method', 'T-MPJPE down', 'Target-PCK (%) up', 'Clean MPJPE down', 'Clean PA-MPJPE down', 'DCE down'),
            backbone + ': comparison', note + ' Target metrics at dose1; DCE averages five positive doses.')
        if experiment_scope == 'full':
            _render(folder / 'table2_dose_poison', report['tables']['T2'], ('d', 'a1', 'rho0p2', 'rho0p4'),
                ('Dose', 'rho=0.1', 'rho=0.2', 'rho=0.4'), backbone + ': dose x poison', note + ' Cells are T-MPJPE; targets change with dose.')
            _render(folder / 'table3_ablation', report['tables']['T3'],
                ('variant', 'auxiliary_carrier', 'dose_coupling', 't1_mpjpe_mm', 'mean_dce_mm', 'clean_mpjpe_mm'),
                ('Variant', 'Aux. carrier', 'Dose coupling', 'T-MPJPE down', 'DCE down', 'Clean MPJPE down'),
                backbone + ': ablation', note)
        _render(folder / 'table5_defenses', report['tables']['T5'], ('defense', *D_METRICS),
            ('Defense', 'T-MPJPE up', 'Target-PCK (%) down', 'Clean MPJPE down', 'Triggered-GT MPJPE down'),
            backbone + ': defense', note + ' Defense directions; low target matching alone is not recovery.')
        summaries = [('defense_paired_changes', report['defense_paired_changes'])]
        if experiment_scope == 'full':
            summaries.insert(0, ('poison_summaries', report['poison_summaries']))
        for name, rows in summaries:
            _write_csv(folder / (name + '.csv'), rows, list(rows[0]))
        dose_rows = [dict(key=r['key'], **d) for r in attacks for d in r.get('dose_rows', [])]
        _write_csv(folder / 'all_attack_six_doses.csv', dose_rows, ('key', 'd', 'tmpjpe_mm', 'no_trigger_tmpjpe_mm', 'dce_mm'))
        report_file = folder / ('four_tables.json' if experiment_scope == 'full' else 'attack_defense_tables.json')
        atomic_json(report_file, report)
        reports.append(dict(backbone=backbone, file=str(report_file), origin='fresh_Vast', n_seeds=1))
    history = HERE.parent / 'historical' / 'hpeli' / 'reference.json'
    if history.is_file() and 'hpeli' not in plan['backbones']:
        ledger = json.loads((HERE.parent / 'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
        bound = ledger['files']['historical/hpeli/reference.json']
        if hashlib.sha256(history.read_bytes()).hexdigest() != bound['sha256']:
            raise ValueError('Historical reference differs from the shipped archive')
        historical = json.loads(history.read_text(encoding='utf-8'))
        folder = output / 'hpeli_historical'
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / 'reference.json', historical)
        note = ('HPE-Li reused Windows/4090 results, not retrained on Vast. T1-T3 below are seed42 only. '
                'Defense summary is the separate original n=3 table, including CLP reported by terminal; '
                'do NOT relabel it seed42 or pool with the fresh-backbone n=1 tables.')
        (folder / 'README.md').write_text(note + '\n', encoding='utf-8')
        _write_csv(folder / 'defense_historical_three_seeds.csv', historical['defense_three_seed_summary'],
                   list(historical['defense_three_seed_summary'][0]))
        # Existing validated source table records, no synthetic defense/CLP seed42 row.
        a = {r['key']: r for r in historical['attack_seed42_records']}
        for name, keys in (('comparison', ('clean', *BASELINES, 'a1')), ('ablation', ('a1', 'a2', 'a3', 'a4'))):
            rows = [dict(method=LABELS[k], **a[k]['metrics']) for k in keys]
            _render(folder / ('table_' + name), rows, ('method', *METRICS),
                ('Method', 'T-MPJPE down', 'Target-PCK (%) up', 'Clean MPJPE down', 'Clean PA-MPJPE down', 'DCE down'),
                'HPE-Li historical ' + name, 'Windows/4090; seed42; n=1; not fresh Vast training.')
        rows = [dict(d=d, **{k: a[k]['dose_rows'][i]['tmpjpe_mm'] for k in ('a1', 'rho0p2', 'rho0p4')}) for i, d in enumerate(GRID)]
        _render(folder / 'table_dose_poison', rows, ('d', 'a1', 'rho0p2', 'rho0p4'),
                ('Dose', 'rho=0.1', 'rho=0.2', 'rho=0.4'), 'HPE-Li historical dose x poison', 'Windows/4090; seed42; n=1; values in mm.')
        reports.append(dict(backbone='hpeli', origin='historical_Windows_4090', seed_scope='T1-T3 seed42; T5 three-seed summary separately'))
    completion = dict(status='COMPLETE_REQUESTED_FRESH_BATCH_ONE_SEED',
        plan_sha256=plan['plan_sha256'], reports=reports, no_across_seed_uncertainty_for_fresh_runs=True)
    if experiment_scope != 'full':
        completion.update(experiment_scope=experiment_scope, fresh_tables=['T1', 'T5'])
    atomic_json(output / 'batch_complete.json', completion)
    return reports


HERE = Path(__file__).resolve().parent
