"""T2: full-MM-Fi dose x poison-rate study of the frozen seed-42 A1 recipe.

Only rho=.2 and .4 create new victims. The original rho=.1 Full and shuffled
victims are immutable, byte-pinned references. Each new victim is trained once
for 50 epochs and evaluated at all six doses by the existing trainer.
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
import sys
import zipfile

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

from run_mmfi_tables import (
    _atomic_json, _cell_lock, _check_inputs, _complete, _plan_fingerprint,
    run_matrix,
)
from run_method_drafts import _file_sha, _sha_json, _verified_cache, _write_csv, trigger_state_sha256
from run_proposed_ablation import (
    _configs, _collect_input_record, _dose_errors, _run_cell as _train_cell,
    _source_provenance as _ablation_sources, EXPECTED_COUNTS, GRID,
)
from a1_frozen_reference import (
    load_a1_reference, validate_a1_input_record, RECORD_SHA256,
)

HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_a1_dose_poison_v1'
FOLDER = 'mmfi_a1_dose_poison_s42_v1'
MANIFEST = 'a1_dose_poison.resolved.json'
STATUS = 'FULL_MMFI_A1_DOSE_POISON_SINGLE_SEED'
METHODS = (('rho0p2', .2), ('rho0p4', .4))
COUPLING_RECORDS = {
    'w_o_coupling/ablation_inputs.json': 'f7e8dcaa2b9eb83b7c17ecf1ca47146d73050c7b97703c000e9999640b4f4af0',
    'w_o_coupling/config.resolved.yaml': 'c296d7b8faa3de4dd7a534af173b161d3b0e1fb9cbf239ffc3fc1d608cb0243d',
    'w_o_coupling/poison_manifest.json': '1dc53bd39954ab72543e0fb6104b2f4414ded0445c2454a6208408855ba99774',
    'w_o_coupling/results.json': '1f4249290e07ebdf90cd23fc900ef48415e33683442c5eb5f632834db3ccc5e5',
    'w_o_coupling/eval_cache.json': '35f27014adbdca9deae86c56da4e6d218330c7982f8381e9944196656dcea347',
}
# Audited diff b1eaf4d..9d89f90: these two files only gain dispatch branches
# for SIG/Bpp and an error-message method list. The md_multicarrier path is
# unchanged. Pin both complete-file byte hashes rather than allowing arbitrary
# newer code under the old A1 scientific protocol.
SOURCE_COMPATIBILITY = {
    'attack/trigger.py': (
        '78f435f3ff1c023b6d74c1b5b9fb45889a6d85a64c8565a1a1bdea00993dcd1c',
        '59dba11704a3d716be0ada460f5163a8f20561e348ad038e58d7fb96d1a1d4b4'),
    'train_backdoor.py': (
        '8f72975e02cf91cae12e4df50622dcef751621052601e977d69ef44c44bbdfec',
        '8f4cfb38b267df0c63a1fbd99cf95a78cdac23c07623002c444fb3fbe1c3a364'),
}
GRID_DOSE_COLUMNS = ('t_mpjpe_d0_mm', 't_mpjpe_d0p2_mm', 't_mpjpe_d0p4_mm',
                     't_mpjpe_d0p6_mm', 't_mpjpe_d0p8_mm', 't_mpjpe_d1_mm')
GRID_COLUMNS = ('rho', 'seed', *GRID_DOSE_COLUMNS, 'clean_mpjpe_mm',
                'mean_positive_dose_control_error_mm', 'result_origin')
DOSE_COLUMNS = ('method_key', 'method', 'seed', 'rho', 'd',
                'no_trigger_tmpjpe_mm', 'triggered_tmpjpe_mm', 'improvement_mm',
                'baseline_source', 'dose_control_error_mm', 'result_origin')


def _source_provenance():
    return dict(_ablation_sources(), **{name: _file_sha(HERE / name) for name in
        ('run_a1_dose_poison.py', 'a1_frozen_reference.py')})


def _configuration(data_home, device, num_workers, rho):
    """Use the same resolver as A1; only rho and profile identity change."""
    from train_backdoor import _resolve_training_config, _validate_training_contract
    cfg = copy.deepcopy(_configs(data_home, device, num_workers, 42, factorial=True)[0])
    cfg.update(rho=rho, confirmation_profile=PROFILE)
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    return cfg


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4, *, a1_reference=None):
    """Resolve settings without datasets, CUDA, or historical artifact access."""
    from train_backdoor import _CHECKPOINT_SCHEMA, _RESULT_SCHEMA
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    reference = Path(a1_reference or data_home /
                     'runs/mmfi_proposed_factorial_ablation_s42_v1').resolve()
    cells = []
    for key, rho in METHODS:
        folder = outdir / key
        cells.append(dict(method_key=key, label=f'A1 rho={rho:g}',
            group='a1_dose_poison', dependencies=[], tables=[2],
            cfg=_configuration(data_home, device, num_workers, rho),
            ckpt_dir=str(folder), eval_cache=str(folder / 'eval_cache.json')))
    matrix = dict(schema=1, dataset='mmfi', seed=42, fresh_results_only=True,
        confirmation_profile=PROFILE, status=STATUS,
        created_utc=datetime.now(timezone.utc).isoformat(), cells=cells,
        sources=dict(source_sha256=_source_provenance(),
            frozen_a1_record_sha256=dict(RECORD_SHA256),
            frozen_coupling_record_sha256=dict(COUPLING_RECORDS),
            audited_source_compatibility={name: list(pair) for name, pair in SOURCE_COMPATIBILITY.items()}),
        metrics_contract=dict(table_id='T2', profile=PROFILE,
            checkpoint_schema=_CHECKPOINT_SCHEMA, result_schema=_RESULT_SCHEMA,
            a1_reference_path=str(reference), seed=42, epochs=50,
            full_split_counts=dict(EXPECTED_COUNTS), evaluation_role='full_official_test',
            dose_grid=list(GRID), poison_rates=[.1, .2, .4],
            training_doses='paired U(0.2,1)', target_joints=[2, 3],
            main_doses=list(GRID), reused_cells=['full', 'w_o_coupling'],
            new_victims=[key for key, _ in METHODS], train_reference=False,
            peak_projection=False, trigger='native md_multicarrier',
            main_metrics=['T-MPJPE(d), mm', 'Clean MPJPE, mm', 'Mean positive-dose DCE, mm'],
            plot_metrics=['T-MPJPE(d), mm', 'Dose-control error(d), mm'],
            dose_control_summary='arithmetic mean of five positive doses; exclude d=0',
            poison_policy='unchanged seeded uniform sampler; counts differ across rho; sets are not forced to be nested',
            uncertainty='single seed; no statistical superiority is estimated'))
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    return matrix


def _validate_manifest(matrix):
    if (matrix.get('schema') != 1 or matrix.get('dataset') != 'mmfi'
            or matrix.get('seed') != 42 or matrix.get('status') != STATUS
            or matrix.get('confirmation_profile') != PROFILE
            or matrix.get('fresh_results_only') is not True
            or matrix.get('plan_sha256') != _plan_fingerprint(matrix)):
        raise ValueError('Not the resolved seed-42 T2 dose/poison plan')
    cells = matrix['cells']
    if [c['method_key'] for c in cells] != [key for key, _ in METHODS]:
        raise ValueError('T2 must train exactly the rho=.2 and .4 victims')
    first = cells[0]
    data_home = Path(first['cfg']['dataset_root']).parent.parent
    expected = build_matrix(data_home, Path(first['ckpt_dir']).parent,
        first['cfg']['device'], first['cfg']['num_workers'],
        a1_reference=matrix['metrics_contract']['a1_reference_path'])
    if matrix['sources'] != expected['sources'] or matrix['metrics_contract'] != expected['metrics_contract']:
        raise ValueError('T2 source or metric contract changed; use a NEW output directory')
    for cell, wanted in zip(cells, expected['cells']):
        cfg = _configuration(data_home, cell['cfg']['device'],
                              cell['cfg']['num_workers'], wanted['cfg']['rho'])
        if cell['cfg'] != cfg or any(cell.get(k) != wanted[k] for k in
                ('method_key', 'label', 'group', 'dependencies', 'tables', 'ckpt_dir', 'eval_cache')):
            raise ValueError(f"{cell['method_key']}: T2 configuration or output path changed")


def _check_against_a1(cfg, reference):
    ignored = {'rho', 'confirmation_profile', 'device', 'num_workers', 'ckpt_every'}
    old = reference['cfg']
    differences = [key for key in set(cfg) | set(old)
                   if key not in ignored and cfg.get(key) != old.get(key)]
    if differences:
        raise ValueError(f'T2 recipe differs from the original A1: {sorted(differences)}')
    if cfg.get('rho') not in (.2, .4) or cfg.get('confirmation_profile') != PROFILE:
        raise ValueError('T2 new victims require rho=.2/.4 and their own profile identity')


def _pinned_coupling_records(path):
    path = Path(path)
    if path.is_dir():
        records = {}
        for name in COUPLING_RECORDS:
            item = path / name
            if item.stat().st_size > 16 * 1024 * 1024:
                raise ValueError('Oversized frozen coupling record')
            records[name] = item.read_bytes()
    else:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            roots = [n[:-len('proposed_ablation.resolved.json')] for n in names
                     if n.endswith('/proposed_ablation.resolved.json')
                     or n == 'proposed_ablation.resolved.json']
            if len(roots) != 1:
                raise ValueError('Frozen ZIP needs one original ablation manifest')
            records = {}
            for name in COUPLING_RECORDS:
                member = roots[0] + name
                if names.count(member) != 1 or archive.getinfo(member).file_size > 16 * 1024 * 1024:
                    raise ValueError('Missing, duplicate or oversized coupling record')
                records[name] = archive.read(member)
    for name, digest in COUPLING_RECORDS.items():
        if hashlib.sha256(records[name]).hexdigest() != digest:
            raise ValueError(f'Frozen w/o Coupling reference changed: {name}')
    return records


def _audit_poison(cfg, res, poison):
    """Reconstruct the original sampler, including its RNG dose draws."""
    import numpy as np
    from train_backdoor import _config_fingerprint
    n = EXPECTED_COUNTS['train']
    count = math.floor(n * cfg['rho'])
    rng = np.random.default_rng(cfg['seed'])
    indices = rng.choice(n, size=count, replace=False)
    doses = rng.uniform(.2, 1., size=count)
    plan = [[int(i), float(d)] for i, d in zip(indices, doses)]
    payload = doses.copy()
    if cfg['dose_coupling'] == 'shuffled':
        np.random.default_rng(np.random.SeedSequence([42, 7043])).shuffle(payload)
    samples = poison.get('samples', [])
    observed = [[s.get('index'), s.get('dose')] for s in samples]
    if (observed != plan
            or [s.get('payload_dose', s.get('dose')) for s in samples] != payload.tolist()):
        raise ValueError('Poison sample identities/doses differ from the unchanged seeded sampler')
    digest = _sha_json(plan)
    required = dict(config_fingerprint=_config_fingerprint(cfg), n_total=n,
        n_poison=count, seed=42, selection='uniform', dose_min=.2, dose_max=1.,
        rho_requested=cfg['rho'], pivot=1, target_joints=[2, 3], theta_max_deg=40.,
        payload_axis=[0., 0., 1.], dose_mode='linear', poison_plan_sha256=digest)
    if (any(poison.get(k) != value for k, value in required.items())
            or poison.get('n_cover', 0) != 0 or poison.get('cover_ratio', 0.) != 0.
            or poison.get('dose_coupling', 'paired') != cfg['dose_coupling']
            or res.get('n_total') != n or res.get('n_poison') != count
            or res.get('n_cover', 0) != 0 or res.get('poison_plan_sha256') != digest
            or res.get('dose_coupling') != cfg['dose_coupling']
            or res.get('training_contract') != 'ordinary_erm'
            or res.get('attacker_access') != 'data_only'):
        raise ValueError('Poison/cache protocol differs from the fixed A1 recipe')
    return digest


def load_references(path):
    """Load exactly the original Full and shuffled victims, without checkpoints."""
    import yaml
    from train_backdoor import _config_fingerprint, _RESULT_SCHEMA
    from mmfi_tables import dose_rows
    full = load_a1_reference(path)
    current_sources = _ablation_sources()
    historical_sources = full['source_sha256']
    if set(current_sources) != set(historical_sources):
        raise ValueError('Training/trigger/metric source set differs from frozen A1')
    for name, current in current_sources.items():
        historical = historical_sources[name]
        if (historical != current
                and SOURCE_COMPATIBILITY.get(name) != (historical, current)):
            raise ValueError(f'Training/trigger/metric source differs from frozen A1: {name}')
    records = _pinned_coupling_records(path)
    cache = json.loads(records['w_o_coupling/eval_cache.json'])
    result = json.loads(records['w_o_coupling/results.json'])
    cfg = yaml.safe_load(records['w_o_coupling/config.resolved.yaml'])
    cfg.pop('config_fingerprint', None)
    res, fingerprint = cache['res'], _config_fingerprint(cfg)
    expected_cfg = dict(full['cfg'], dose_coupling='shuffled')
    # Only the runtime-assigned device can differ across the old cells.
    expected_cfg.update({k: cfg[k] for k in ('device', 'num_workers', 'ckpt_every')})
    if (cfg != expected_cfg or cache.get('cfg') != cfg or result.get('cfg') != cfg
            or cache.get('cfg_fingerprint') != fingerprint
            or result.get('cfg_fingerprint') != fingerprint or result.get('res') != res
            or cache.get('trained_epochs') != 50 or cache.get('result_schema') != _RESULT_SCHEMA):
        raise ValueError('Frozen coupling cache/config/results disagree')
    inputs = json.loads(records['w_o_coupling/ablation_inputs.json'])
    validate_a1_input_record(full, inputs)
    if inputs.get('config_fingerprint') != fingerprint:
        raise ValueError('Frozen coupling input fingerprint differs')
    poison = json.loads(records['w_o_coupling/poison_manifest.json'])
    digest = _audit_poison(cfg, res, poison)
    if digest != full['audit']['poison_plan_sha256']:
        raise ValueError('Full and shuffled do not share the original poison samples/doses')
    cell = dict(method_key='w_o_coupling', label='w/o Coupling, rho=0.1', cfg=cfg)
    curves, _ = dose_rows(cell, res)
    errors = _dose_errors(res)
    curves = [dict(row, dose_control_error_mm=error) for row, error in zip(curves, errors)]
    if [r['d'] for r in curves] != GRID:
        raise ValueError('Frozen coupling needs all six doses')
    return full, dict(cfg=cfg, dose_response=curves,
        provenance=dict(record_sha256=dict(COUPLING_RECORDS), cfg_fingerprint=fingerprint,
            poison_plan_sha256=digest, input_audit=inputs))


def _preflight(matrix):
    reference, coupling = load_references(matrix['metrics_contract']['a1_reference_path'])
    for cell in matrix['cells']:
        _check_against_a1(cell['cfg'], reference)
    record = _collect_input_record(matrix['cells'][0])
    validate_a1_input_record(reference, record)
    return reference, coupling


def _grid_row(series, rho, clean_mpjpe_mm, origin):
    if [r['d'] for r in series] != GRID:
        raise ValueError('The main grid requires the complete six-dose curve')
    return dict(rho=rho, seed=42, clean_mpjpe_mm=clean_mpjpe_mm, result_origin=origin,
        **{name: row['triggered_tmpjpe_mm'] for name, row in zip(GRID_DOSE_COLUMNS, series)},
        mean_positive_dose_control_error_mm=sum(row['dose_control_error_mm'] for row in series[1:]) / 5)


def _series(cell, res, rho, origin):
    from mmfi_tables import metric_row, dose_rows
    series, _ = dose_rows(cell, res)
    errors = _dose_errors(res)
    if [r['d'] for r in series] != GRID:
        raise ValueError('All six evaluated doses are required')
    metric = metric_row(cell, res)
    series = [dict(row, rho=rho, result_origin=origin, dose_control_error_mm=error)
              for row, error in zip(series, errors)]
    return _grid_row(series, rho, metric['clean_mpjpe_mm'], origin), series


def _audit_run_metadata(cell, res):
    import yaml
    from train_backdoor import _config_fingerprint
    folder, cfg = Path(cell['ckpt_dir']), cell['cfg']
    yaml_cfg = yaml.safe_load((folder / 'config.resolved.yaml').read_text(encoding='utf-8'))
    yaml_cfg.pop('config_fingerprint', None)
    result = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
    fingerprint = _config_fingerprint(cfg)
    if (not isinstance(result.get('cfg'), dict) or result.get('method_key') != cell['method_key']
            or result.get('res') != res or result.get('cfg_fingerprint') != fingerprint
            or _config_fingerprint(yaml_cfg) != fingerprint
            or _config_fingerprint(result['cfg']) != fingerprint
            or yaml_cfg.get('epochs') != 50 or result['cfg'].get('epochs') != 50):
        raise ValueError('New victim YAML/results differ from the verified cache/plan')


def build_summary(matrix):
    _validate_manifest(matrix)
    reference, coupling = _preflight(matrix)
    origin = 'immutable_existing_factorial_A1'
    ref_curves = [dict(row, method_key='rho0p1', method='A1 rho=0.1', rho=.1,
                       result_origin=origin) for row in reference['dose_response']]
    rows = [_grid_row(ref_curves, .1, reference['row']['clean_mpjpe_mm'], origin)]
    curves, provenance = list(ref_curves), []
    from train_backdoor import build_trigger, _config_fingerprint
    original_trigger_sha = trigger_state_sha256(build_trigger(reference['cfg']))
    for cell in matrix['cells']:
        res, source = _verified_cache(cell)
        _audit_run_metadata(cell, res)
        folder, cfg = Path(cell['ckpt_dir']), cell['cfg']
        record = json.loads((folder / 'ablation_inputs.json').read_text(encoding='utf-8'))
        validate_a1_input_record(reference, record)
        if (record.get('schema') != 1 or record.get('profile') != PROFILE
                or record.get('config_fingerprint') != _config_fingerprint(cfg)):
            raise ValueError('New victim input audit does not match its config/profile')
        poison = json.loads((folder / 'poison_manifest.json').read_text(encoding='utf-8'))
        digest = _audit_poison(cfg, res, poison)
        current_trigger_sha = trigger_state_sha256(build_trigger(cfg))
        if current_trigger_sha != original_trigger_sha:
            raise ValueError('Poison-rate intervention changed the A1 carrier')
        row, series = _series(cell, res, cfg['rho'], 'fresh_50_epoch_victim')
        rows.append(row)
        curves.extend(series)
        provenance.append(dict(source, input_audit=record, poison_plan_sha256=digest,
            trigger_state_sha256=current_trigger_sha,
            artifact_sha256={name: _file_sha(folder / name) for name in
                ('ablation_inputs.json', 'poison_manifest.json', 'config.resolved.yaml', 'results.json')}))
    curves.extend(dict(row, rho=.1, result_origin='immutable_existing_factorial_shuffled')
                  for row in coupling['dose_response'])
    report = dict(status='t2_dose_poison_complete_single_seed', table_id='T2', seed=42,
        profile=PROFILE, plan_sha256=matrix['plan_sha256'], rows=rows, dose_response=curves,
        metrics_contract=matrix['metrics_contract'], sources=matrix['sources'],
        frozen_a1_provenance=reference['provenance'],
        frozen_a1_source_sha256=reference['source_sha256'],
        frozen_coupling_provenance=coupling['provenance'], provenance=provenance,
        trigger_state_sha256=original_trigger_sha,
        mean_positive_dose_control_error_mm={key: sum(r['dose_control_error_mm']
            for r in curves if r['method_key'] == key and r['d'] > 0) / 5
            for key in ('rho0p1', 'rho0p2', 'rho0p4', 'w_o_coupling')},
        limitations=[
            'Single seed 42; no uncertainty or statistical superiority is estimated.',
            'Only rho=.2/.4 are fresh victims. Full and shuffled rho=.1 are reused with byte-pinned original metadata.',
            'Poison samples use the original seed-42 uniform sampler independently at each rho; selected sets are not required to be nested.',
            'The frozen method was selected during prior development on MM-Fi; this is not a blinded first test.',
            'No peak or L2 projection is imposed; changing rho does not change trigger parameters.',
            'Split hashes bind ordered file/frame identities and action bytes, not all CSI/pose file contents.',
        ])
    json.dumps(report, allow_nan=False)
    return report


def export_summary(matrix, outdir):
    report = build_summary(matrix)
    outdir = Path(outdir)
    _write_csv(outdir / 'table2_dose_poison.csv', report['rows'], GRID_COLUMNS)
    _atomic_json(outdir / 'dose_response.json', dict(table_id='T2', units='mm',
        profile=PROFILE, plan_sha256=matrix['plan_sha256'], rows=report['dose_response']))
    _write_csv(outdir / 'dose_response.csv', report['dose_response'], DOSE_COLUMNS)
    lines = ['# T2: Dose x poison rate', '',
        'MM-Fi protocol1-s1; seed 42; 50 epochs; native A1; paired training doses U(0.2,1).', '',
        '| Poison rate | d=0 | d=.2 | d=.4 | d=.6 | d=.8 | d=1 | Clean MPJPE | Mean positive-dose DCE |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in report['rows']:
        values = [row[name] for name in GRID_DOSE_COLUMNS] + [row['clean_mpjpe_mm'], row['mean_positive_dose_control_error_mm']]
        lines.append(f"| {row['rho']:g} | " + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += ['', 'All errors are millimetres; dose cells are target-joint T-MPJPE.',
        'All six evaluated doses and DCE curves are retained in dose_response.csv/json.',
        'rho=.1 Full and w/o Coupling are original hash-pinned results; only rho=.2/.4 are trained here.',
        'DCE summaries average the five positive doses; identity dose is excluded.', '',
        *[f'- {item}' for item in report['limitations']], '',
        f"Status: `{report['status']}`; plan SHA256: `{report['plan_sha256']}`."]
    path = outdir / 'table2_dose_poison.md'
    temporary = path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric, ylabel, filename in (
            ('triggered_tmpjpe_mm', 'T-MPJPE (mm)', 'tmpjpe_vs_dose.png'),
            ('dose_control_error_mm', 'Dose-control error (mm)', 'dose_control_error.png')):
        fig, ax = plt.subplots(figsize=(6.3, 4.1))
        for key in ('rho0p1', 'rho0p2', 'rho0p4', 'w_o_coupling'):
            values = [r for r in report['dose_response'] if r['method_key'] == key]
            ax.plot(GRID, [r[metric] for r in values], marker='o',
                    linestyle='--' if key == 'w_o_coupling' else '-', label=values[0]['method'])
        ax.set(xlabel='Evaluation dose', ylabel=ylabel, title='MM-Fi A1; seed 42')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=180)
        plt.close(fig)
    # Completion marker follows every table/curve/plot write. A plot failure
    # must not create a freshly completed T2 report.
    _atomic_json(outdir / 'table2_dose_poison.json', report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--a1-reference', type=Path)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--_cell', help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args._cell:
        if args._manifest is None:
            raise ValueError('Worker requires --_manifest')
        matrix = json.loads(args._manifest.read_text(encoding='utf-8'))
        _validate_manifest(matrix)
        cell = next((c for c in matrix['cells'] if c['method_key'] == args._cell), None)
        if cell is None:
            raise ValueError('Unknown T2 victim')
        _preflight(matrix)
        _train_cell(cell)
        return 0
    if args.data_home is None:
        raise ValueError('--data-home is required')
    if (not args.devices or len(set(args.devices)) != len(args.devices)
            or any(d != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None for d in args.devices)):
        raise ValueError('--devices must list distinct logical cuda:N devices, or cpu')
    if args.dry_run and args.export_only:
        raise ValueError('Choose --dry-run or --export-only')
    outdir = (args.outdir or args.data_home / 'runs' / FOLDER).resolve()
    with _cell_lock(outdir):
        path = outdir / MANIFEST
        requested = build_matrix(args.data_home, outdir, args.devices[0],
                                args.num_workers, a1_reference=args.a1_reference)
        _validate_manifest(requested)
        if path.exists():
            matrix = json.loads(path.read_text(encoding='utf-8'))
            _validate_manifest(matrix)
            if matrix['plan_sha256'] != requested['plan_sha256']:
                raise ValueError('Existing T2 plan differs; use a NEW output directory')
            for cell in matrix['cells']:
                cell['cfg'].update(device=args.devices[0], num_workers=args.num_workers)
        else:
            if any(p.name != 'run.lock' for p in outdir.iterdir()):
                raise ValueError('Output directory contains unrelated files; use a NEW directory')
            matrix = requested
        _atomic_json(path, matrix)
        if args.dry_run:
            print(f'[T2] DRY RUN: seed42; two NEW 50-epoch victims rho=.2/.4; '
                  f'Full/shuffled rho=.1 reused; no data/GPU/training.\n{path}', flush=True)
            return 0
        _preflight(matrix)
        if not args.export_only:
            _check_inputs(matrix, args.devices)
            run_matrix(matrix, path, args.devices, worker_script=Path(__file__))
        if any(not _complete(c) for c in matrix['cells']):
            raise ValueError('Both rho=.2/.4 victims must finish before exporting T2')
        export_summary(matrix, outdir)
        print(f'[T2] complete -> {outdir / "table2_dose_poison.md"}', flush=True)
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, AssertionError) as error:
        print(f'[T2] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
