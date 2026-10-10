"""Six-arm TRAIN-only screening of stronger Proposed carriers versus Blended.

This is an isolated development experiment, not a publication comparison.
No test-set parameter selection, historical-result import, or baseline changes.
Attacker fitting finishes before each independent ordinary-ERM victim starts.
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

import run_learned_carrier_drafts as shared
from run_mmfi_tables import _atomic_json, _cell_lock, _check_inputs
from run_method_drafts import _file_sha, _sha_json, _verified_cache, _write_csv

HERE = Path(__file__).resolve().parent
PROFILE = 'carrier_bank_screen_v1'
STATUS = shared.STATUS
GRID = shared.GRID
METHODS = (
    ('clean', 'Clean fresh victim control', 'micro_dropper', None, None),
    ('proposed', 'Fixed Proposed (dual budget)', 'md_multicarrier_peak_matched', None, None),
    ('blended', 'Unchanged Blended-CSI (dual budget)', 'blended', None, None),
    ('lc_trainaware', 'Proposed: stronger surrogate, 2 carriers', 'learned_carrier', 'trainaware', 2),
    ('lc_bank', 'Proposed: stronger surrogate, 8 carriers', 'learned_carrier', 'bank', 8),
    ('lc_bank_guard', 'Proposed: 8 carriers + clean utility guard', 'learned_carrier', 'bank_guard', 8),
)
MAIN_ERRORS = ('clean_mpjpe_mm', 'clean_pampjpe_mm', 't1_mpjpe_mm')
MAIN_PCK = tuple(f'clean_pck_{t:.1f}_pct' for t in (.5, .4, .3, .2, .1))
MAIN_METRICS = MAIN_ERRORS + MAIN_PCK
_prepared_cell = shared._prepared_cell
_complete = shared._complete
_run_cell = shared._run_cell
_plan_sha = shared._plan_sha


def _source_provenance():
    paths = shared._source_provenance()
    paths.update({name: _file_sha(HERE / name) for name in
                  ('run_carrier_bank_drafts.py', 'carrier_bank_fit.py')})
    return paths


def _recipe_sha(cfg, sources):
    return _sha_json(dict(profile=PROFILE, cfg=shared._scientific(cfg), sources=sources))


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4, *,
                 epochs=15, train_samples=20000, eval_samples=4096,
                 distortion_samples=256, relative_l2=.10, fit_options=None,
                 skip_clean_probe=False, skip_distortion=False):
    """Resolve metadata only, without dataset access, carrier fitting or CUDA."""
    from carrier_bank_fit import BANK_FIT_DEFAULTS, resolve_bank_fit_config
    from train_backdoor import _resolve_training_config, _validate_training_contract
    fit = dict(BANK_FIT_DEFAULTS)
    if fit_options:
        if set(fit_options) - set(fit):
            raise ValueError('Unknown carrier-bank fitting option')
        fit.update(fit_options)
    # Resolve the controls using the existing operators and common victim
    # protocol. The nine-arm predecessor is never run or written here.
    controls = shared.build_matrix(data_home, outdir, device, num_workers,
        epochs=epochs, train_samples=train_samples, eval_samples=eval_samples,
        distortion_samples=distortion_samples, relative_l2=relative_l2,
        skip_clean_probe=skip_clean_probe, skip_distortion=skip_distortion)
    source_sha = _source_provenance()
    cells = []
    for index, (key, label, trigger, variant, bank_size) in enumerate(METHODS):
        cfg = copy.deepcopy(controls['cells'][index if index < 3 else 1]['cfg'])
        cfg.update(draft_profile=PROFILE, trigger=trigger)
        if variant:
            cfg.update(fit)
            cfg.update(lc_variant=variant, lc_bank_size=bank_size, lc_bank_seed=42)
            cfg = resolve_bank_fit_config(cfg)
        cfg = _resolve_training_config(cfg)
        _validate_training_contract(cfg)
        folder = Path(outdir).resolve() / key
        cells.append(dict(method_key=key, label=label,
            group='shared_poison_proposed_development', cfg=cfg,
            recipe_sha256=_recipe_sha(cfg, source_sha),
            ckpt_dir=str(folder), eval_cache=str(folder / 'eval_cache.json')))
    contract = copy.deepcopy(controls['metrics_contract'])
    contract.update(poison_indices='identical fixed uniform identities and paired doses in all attack arms',
        main_metrics=list(MAIN_METRICS),
        baseline='unchanged Blended operator, alpha=.2*d, under the same dual upper bounds',
        attacker_fit='extra train-only compute is disclosed; no surrogate weights transferred',
        selection='learned artifact selected on INTERNAL TRAIN validation only; external holdout is for screening',
        fresh_utility_tolerances=dict(mpjpe_mm=fit['lc_utility_mpjpe_tolerance'] * 1000,
            pampjpe_mm=fit['lc_utility_pa_tolerance'] * 1000,
            pck_percentage_points=fit['lc_utility_pck_tolerance'] * 100),
        dominance='point estimates only: lower T1, no worse clean MPJPE/PA-MPJPE and all five PCK values',
        guarded_surrogate='only bank_guard uses PA/PCK utility penalties and internal gates; fresh victim may still fail')
    matrix = dict(schema=1, status=STATUS, DRAFT_ONLY=True,
        dataset='mmfi', seed=42, draft_profile=PROFILE,
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources=dict(source_sha256=source_sha,
            references=copy.deepcopy(shared.REFERENCES),
            implementation='Independent Proposed development; not source-paper reproduction'),
        fitting_options=fit, cells=cells, metrics_contract=contract)
    matrix['plan_sha256'] = _plan_sha(matrix)
    _validate_manifest(matrix)
    return matrix


def _validate_manifest(matrix):
    from train_backdoor import _validate_training_contract
    if (matrix.get('schema') != 1 or matrix.get('status') != STATUS
            or matrix.get('DRAFT_ONLY') is not True or matrix.get('draft_profile') != PROFILE
            or matrix.get('dataset') != 'mmfi' or matrix.get('seed') != 42):
        raise ValueError('Not the isolated six-arm carrier-bank draft manifest')
    if matrix.get('plan_sha256') != _plan_sha(matrix):
        raise ValueError('Plan fingerprint differs from its contents')
    sources = _source_provenance()
    if matrix['sources']['source_sha256'] != sources:
        raise ValueError('Scientific source changed; use a NEW output directory')
    if [c['method_key'] for c in matrix['cells']] != [row[0] for row in METHODS]:
        raise ValueError('All six planned cells must remain in fixed order')
    equal_keys = ('seed', 'epochs', 'victim_epochs', 'model', 'optimizer', 'lr',
        'momentum', 'weight_decay', 'batch_size', 'pretrained', 'data_parallel',
        'dataset_root', 'action_npy', 'mmfi_protocol', 'mmfi_setting',
        'pivot', 'target_joints', 'theta_max_deg', 'payload_axis', 'dose_mode',
        'dose_grid', 'dose_min', 'dose_max', 'dose_coupling', 'poison_select',
        'draft_train_samples', 'draft_eval_samples', 'draft_subset_seed',
        'draft_eval_source', 'training_protocol', 'attacker_access',
        'lc_reference_eps', 'lc_relative_l2')
    reference = {k: matrix['cells'][0]['cfg'].get(k) for k in equal_keys}
    for cell, (key, _, trigger, variant, bank_size) in zip(matrix['cells'], METHODS):
        cfg = cell['cfg']
        _validate_training_contract(cfg)
        if ({k: cfg.get(k) for k in equal_keys} != reference
                or cfg.get('trigger') != trigger or cfg.get('lc_variant') != variant
                or cfg.get('rho') != (0 if key == 'clean' else .1)
                or cfg.get('draft_profile') != PROFILE or cfg.get('seed') != 42
                or cfg.get('optimizer') != 'sgd' or cfg.get('lr') != .001
                or cfg.get('momentum') != .9 or cfg.get('weight_decay') != 0
                or cfg.get('pivot') != 1 or cfg.get('target_joints') != [2, 3]
                or cfg.get('dose_grid') != GRID or cfg.get('dose_coupling') != 'paired'
                or cfg.get('dose_min') != .2 or cfg.get('dose_max') != 1
                or cfg.get('lc_reference_eps') != .185
                or cfg.get('eps') != (.2 if key == 'blended' else .185)
                or cell['recipe_sha256'] != _recipe_sha(cfg, sources)):
            raise ValueError(f'{key}: immutable recipe or shared protocol disagrees with plan')
        if variant and (cfg.get('lc_bank_size') != bank_size or
                any(cfg.get(k) != v for k, v in matrix['fitting_options'].items())):
            raise ValueError(f'{key}: carrier size or fitting options differ from the fixed plan')
    if (matrix['metrics_contract'].get('official_test_used') is not False
            or matrix['metrics_contract'].get('main_metrics') != list(MAIN_METRICS)):
        raise ValueError('Official-test/metric contract mismatch')


def compare_to_blended(rows, tolerances):
    """Numerical screening only; never calls a one-seed result significant."""
    by_key = {row['method_key']: row for row in rows}
    if 'blended' not in by_key:
        raise ValueError('Unchanged fresh Blended is required for screening')
    baseline = by_key['blended']
    for row in rows:
        for name in MAIN_METRICS:
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Missing or nonfinite main metric: {name}')
    comparisons = []
    for row in rows:
        if row['method_key'] in ('clean', 'blended'):
            continue
        delta = {f'delta_{name}': row[name] - baseline[name] for name in MAIN_METRICS}
        lower_t1 = row['t1_mpjpe_mm'] < baseline['t1_mpjpe_mm']
        clean_no_worse = (all(row[k] <= baseline[k] for k in MAIN_ERRORS[:2])
                          and all(row[k] >= baseline[k] for k in MAIN_PCK))
        within = (row['clean_mpjpe_mm'] <= baseline['clean_mpjpe_mm'] + tolerances['mpjpe_mm']
                  and row['clean_pampjpe_mm'] <= baseline['clean_pampjpe_mm'] + tolerances['pampjpe_mm']
                  and all(row[k] >= baseline[k] - tolerances['pck_percentage_points'] for k in MAIN_PCK))
        comparisons.append(dict(status=STATUS, method_key=row['method_key'],
            method=row['method'], lower_t1_than_blended=lower_t1,
            no_worse_all_clean_metrics=clean_no_worse,
            pointwise_dominates_blended=lower_t1 and clean_no_worse,
            clean_within_declared_tolerances=within,
            lower_t1_with_tolerated_clean_tradeoff=lower_t1 and within,
            **delta))
    return comparisons


def build_summary(matrix):
    from mmfi_tables import metric_row, dose_rows
    _validate_manifest(matrix)
    cells = [_prepared_cell(cell) for cell in matrix['cells']]
    if any(cell is None for cell in cells):
        raise ValueError('All six cells require a valid frozen prepared config')
    audit = shared.audit_common_inputs(cells, profile=PROFILE)
    rows, doses, provenance, fitting_audit = [], [], [], {}
    for cell in cells:
        res, source = _verified_cache(cell)
        key = cell['method_key']
        if (res.get('training_contract') != 'ordinary_erm' or res.get('attacker_access') != 'data_only'
                or res.get('draft_action_sha256') != audit['action_file_sha256']
                or res.get('poison_plan_sha256') != audit['cells'][key]['poison_plan_sha256']
                or res.get('dose_grid') != GRID):
            raise ValueError(f'{key}: cached result violates the scientific contract')
        metric = metric_row(cell, res)
        row = dict(status=STATUS, method_key=key, method=cell['label'], seed=42,
            epochs=cell['cfg']['epochs'], rho=cell['cfg']['rho'],
            clean_mpjpe_mm=metric['clean_mpjpe_mm'],
            clean_pampjpe_mm=metric['clean_pampjpe_mm'],
            t1_mpjpe_mm=metric['tmpjpe_mm'],
            **{k: metric[k] for k in MAIN_PCK},
            cfg_fingerprint=source['cfg_fingerprint'], recipe_sha256=cell['recipe_sha256'],
            lc_artifact_sha256=cell['cfg'].get('lc_artifact_sha256'),
            lc_fitting_sha256=cell['cfg'].get('lc_fitting_sha256'))
        rows.append(row)
        series, _ = dose_rows(cell, res)
        doses.extend(dict(status=STATUS, **entry) for entry in series)
        provenance.append(source)
        if cell['cfg'].get('lc_variant'):
            fitting = json.loads((Path(cell['ckpt_dir']) / 'fitting.json').read_text(encoding='utf-8'))
            fitting_audit[key] = {name: fitting.get(name) for name in (
                'surrogate_initialization_seeds', 'surrogate_count', 'selected_round',
                'selected_utility_gate_passed', 'utility_gate_required',
                'no_eligible_candidate', 'utility_reference', 'utility_tolerances',
                'lookahead', 'lookahead_is_full_training_bilevel', 'fit_settings')}
    return dict(status=STATUS, DRAFT_ONLY=True, draft_profile=PROFILE,
        plan_sha256=matrix['plan_sha256'], rows=rows, dose_response=doses,
        blended_comparison=compare_to_blended(rows, matrix['metrics_contract']['fresh_utility_tolerances']),
        audit=audit, sources=matrix['sources'], metrics_contract=matrix['metrics_contract'],
        provenance=provenance, fitting_audit=fitting_audit, limitations=[
            'All planned rows retained, including failed hypotheses; no historical tables overwritten.',
            'One seed and TRAIN-only development holdout: no statistical superiority or paper-result claim.',
            'Attacker surrogate compute exceeds fixed-trigger controls and is explicitly disclosed.',
            'Two-step SGD/momentum lookahead freezes cloned BatchNorm buffers; not exact full victim unrolling.',
            'The bank_guard internal-surrogate utility gate does not guarantee fresh-victim clean utility.',
            'Strict pointwise dominance is separate from lower T1 with tolerated clean degradation.',
            'Common actual Linf/relative-L2 ceilings, not equal realized perturbation.',
            'Digital distortion is not RF detectability or over-the-air stealthiness.',
            'Repeated development on this benchmark is not a blinded first test.',
            'Input manifests bind ordered file/frame identities, not every CSI/pose file byte.'])


def export_summary(matrix, outdir, device='cpu'):
    report = build_summary(matrix)
    if matrix['metrics_contract']['distortion'] != 'missing_by_explicit_skip':
        distortion = shared.build_distortion(matrix, report['audit'], matrix['metrics_contract']['distortion_samples'])
        _atomic_json(Path(outdir) / 'input_distortion.json', distortion)
        _write_csv(Path(outdir) / 'input_distortion.csv', distortion['rows'], list(distortion['rows'][0]))
    if matrix['metrics_contract']['clean_probe'] != 'missing_by_explicit_skip':
        probes = shared.build_clean_probes(matrix, device)
        _atomic_json(Path(outdir) / 'clean_victim_probes.json', probes)
        _write_csv(Path(outdir) / 'clean_victim_probes.csv', probes['rows'], list(probes['rows'][0]))
    for name, values in (('draft_summary', report['rows']),
                         ('dose_response', report['dose_response']),
                         ('blended_comparison', report['blended_comparison'])):
        _write_csv(Path(outdir) / f'{name}.csv', values, list(values[0]))
        if name != 'draft_summary':
            _atomic_json(Path(outdir) / f'{name}.json', dict(status=STATUS,
                plan_sha256=matrix['plan_sha256'], rows=values))
    _atomic_json(Path(outdir) / 'draft_summary.json', report)
    lines = [f'# {STATUS}', '',
        'Six-arm Proposed development screen. All rows retained; Blended unchanged.', '',
        '| Method | MPJPE (mm) ↓ | PA-MPJPE (mm) ↓ | T-MPJPE d=1 (mm) ↓ | PCK .5/.4/.3/.2/.1 (%) ↑ |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in report['rows']:
        pck = '/'.join(f'{row[k]:.2f}' for k in MAIN_PCK)
        lines.append(f"| {row['method']} | {row['clean_mpjpe_mm']:.3f} | {row['clean_pampjpe_mm']:.3f} | {row['t1_mpjpe_mm']:.3f} | {pck} |")
    lines.extend(['', 'Comparison against fresh Blended (point estimates only):', '',
        '| Proposed arm | ΔT1 (mm) ↓ | Strict dominance on all main metrics | Lower T1 with tolerated clean tradeoff |',
        '| --- | ---: | --- | --- |'])
    for row in report['blended_comparison']:
        lines.append(f"| {row['method']} | {row['delta_t1_mpjpe_mm']:.3f} | {row['pointwise_dominates_blended']} | {row['lower_t1_with_tolerated_clean_tradeoff']} |")
    guard = report['fitting_audit'].get('lc_bank_guard', {})
    lines.extend(['',
        f"Internal surrogate utility gate passed for selected bank_guard: {guard.get('selected_utility_gate_passed')}; no eligible round: {guard.get('no_eligible_candidate')}.",
        'A failed or missing internal gate is NOT presented as successful utility protection.'])
    lines.extend(['', 'Negative Δ errors and positive ΔPCK favour the candidate.',
        'Tolerated clean tradeoff is NOT a win on every clean metric.',
        f"Tolerances fixed before running: {matrix['metrics_contract']['fresh_utility_tolerances']}.",
        'PCK thresholds are RELATIVE, not millimetres. All six doses remain in dose_response.csv/json.',
        'Input-distortion and clean-victim probes are separate diagnostics, not additional ranking metrics.',
        f"Seed 42; rho=.1; epochs={matrix['cells'][0]['cfg']['epochs']}; TRAIN holdout only; ordinary victim MPJPE.",
        f"Plan SHA256: `{matrix['plan_sha256']}`", '', 'Limitations:',
        *[f'- {item}' for item in report['limitations']]])
    Path(outdir, 'draft_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def parse_args(argv=None):
    from carrier_bank_fit import BANK_FIT_DEFAULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--train-samples', type=int, default=20000)
    parser.add_argument('--eval-samples', type=int, default=4096)
    parser.add_argument('--distortion-samples', type=int, default=256)
    parser.add_argument('--relative-l2', type=float, default=.10)
    parser.add_argument('--cells', nargs='+', choices=[row[0] for row in METHODS])
    for key, value in BANK_FIT_DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'),
            type=int if isinstance(value, int) else float, default=value)
    for name in ('fresh', 'dry-run', 'export-only', 'skip-distortion', 'skip-clean-probe'):
        parser.add_argument('--' + name, action='store_true')
    parser.add_argument('--_cell', choices=[row[0] for row in METHODS], help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    from carrier_bank_fit import BANK_FIT_DEFAULTS
    args = parse_args(argv)
    if args._cell:
        if args._manifest is None:
            raise ValueError('Worker requires its resolved manifest')
        matrix = json.loads(args._manifest.read_text(encoding='utf-8'))
        _validate_manifest(matrix)
        _run_cell(next(c for c in matrix['cells'] if c['method_key'] == args._cell))
        return 0
    if args.data_home is None or args.outdir is None:
        raise ValueError('--data-home and --outdir are required')
    if args.dry_run and args.export_only:
        raise ValueError('--dry-run and --export-only cannot be combined')
    if (not args.devices or len(set(args.devices)) != len(args.devices)
            or any(d != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None for d in args.devices)):
        raise ValueError('Devices must be distinct logical cuda:N devices or cpu')
    if args.cells and len(set(args.cells)) != len(args.cells):
        raise ValueError('--cells cannot contain duplicates')
    outdir = args.outdir.resolve()
    if args.fresh and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('--fresh refuses a nonempty directory; never deletes old results')
    requested = build_matrix(args.data_home, outdir, args.devices[0], args.num_workers,
        epochs=args.epochs, train_samples=args.train_samples, eval_samples=args.eval_samples,
        distortion_samples=args.distortion_samples, relative_l2=args.relative_l2,
        fit_options={key: getattr(args, key) for key in BANK_FIT_DEFAULTS},
        skip_clean_probe=args.skip_clean_probe, skip_distortion=args.skip_distortion)
    with _cell_lock(outdir):
        path = outdir / 'carrier_bank.resolved.json'
        if path.exists():
            matrix = json.loads(path.read_text(encoding='utf-8'))
            _validate_manifest(matrix)
            if requested['plan_sha256'] != matrix['plan_sha256']:
                raise ValueError('Changed scientific settings require a NEW output directory')
            for cell in matrix['cells']:
                cell['cfg'].update(device=args.devices[0], num_workers=args.num_workers)
        else:
            if any(p.name != 'run.lock' for p in outdir.iterdir()):
                raise ValueError('Existing non-draft files occupy output directory')
            matrix = requested
        _atomic_json(path, matrix)
        if args.dry_run:
            print(f'[{STATUS}] DRY RUN: six recipes; no data/GPU/fitting/training.\n{path}')
            return 0
        if not args.export_only:
            _check_inputs(matrix, args.devices)
            shared.run_matrix(matrix, path, args.devices, args.cells, worker_script=__file__)
        missing = [c['method_key'] for c in matrix['cells'] if not _complete(c)]
        if missing:
            if args.cells and not args.export_only:
                print(f'[{STATUS}] Requested cells complete; all-row export pending: {missing}')
                return 0
            raise ValueError(f'Missing valid completed cells: {missing}; no summary exported')
        export_summary(matrix, outdir, args.devices[0])
        print(f'[{STATUS}] Summary: {outdir / "draft_summary.md"}')
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[{STATUS}] ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
