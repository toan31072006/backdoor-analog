"""Three fresh TRAIN-only victims: unchanged Blended/trainaware and paired guard.

This isolated Proposed development screen is not a publication comparison.
The new two-carrier arm pairs poisoned surrogates with stage-matched clean
twins; fitting freezes before each independent ordinary-ERM victim starts.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

import run_carrier_bank_drafts as base
import run_learned_carrier_drafts as shared
from run_mmfi_tables import _atomic_json, _cell_lock, _check_inputs
from run_method_drafts import _file_sha, _sha_json, _verified_cache, _write_csv

HERE = Path(__file__).resolve().parent
PROFILE = 'paired_guard_screen_v1'
STATUS = shared.STATUS
GRID = shared.GRID
METHODS = (
    ('blended', 'Unchanged Blended-CSI (dual budget)', 'blended', None, None),
    ('lc_trainaware', 'Unchanged Proposed: trainaware, 2 carriers', 'learned_carrier', 'trainaware', 2),
    ('lc_paired_guard', 'Proposed: paired clean-twin guard, 2 carriers', 'learned_carrier', 'paired_guard', 2),
)
MAIN_ERRORS, MAIN_PCK, MAIN_METRICS = base.MAIN_ERRORS, base.MAIN_PCK, base.MAIN_METRICS
TOLERANCE_KEYS = ('lc_utility_mpjpe_tolerance', 'lc_utility_pa_tolerance',
                  'lc_utility_pck_tolerance')
compare_to_blended = base.compare_to_blended
_prepared_cell, _complete, _run_cell = shared._prepared_cell, shared._complete, shared._run_cell
_plan_sha = shared._plan_sha


def _source_provenance():
    paths = base._source_provenance()
    paths.update({name: _file_sha(HERE / name) for name in
                  ('run_paired_guard_drafts.py', 'paired_guard_fit.py')})
    return paths


def _recipe_sha(cfg, sources):
    return _sha_json(dict(profile=PROFILE, cfg=shared._scientific(cfg), sources=sources))


def _fit_options(options=None):
    from carrier_bank_fit import BANK_FIT_DEFAULTS
    from paired_guard_fit import PAIRED_FIT_DEFAULTS
    control = dict(BANK_FIT_DEFAULTS)
    if options:
        if set(options) - set(control):
            raise ValueError('Unknown paired-guard fitting option')
        for key in TOLERANCE_KEYS:
            if key in options and (isinstance(options[key], bool) or options[key] != control[key]):
                raise ValueError(f'{key}: control utility tolerances are immutable BANK_FIT_DEFAULTS')
        control.update(options)
    paired = dict(control)
    paired.update({key: PAIRED_FIT_DEFAULTS[key] for key in TOLERANCE_KEYS})
    return control, paired


def _metrics_contract(controls, control_fit):
    contract = copy.deepcopy(controls)
    contract.update(
        clean_probe='missing_by_explicit_skip',
        clean_victim_probes='unsupported_not_run: this screen has exactly three poisoned fresh victims',
        poison_indices='identical fixed uniform identities and paired doses in all three attack arms',
        main_metrics=list(MAIN_METRICS),
        baseline='unchanged Blended operator, alpha=.2*d, under the same dual upper bounds',
        attacker_fit='extra train-only surrogate and clean-twin compute is disclosed; no weights transferred',
        selection='artifact selected on INTERNAL TRAIN validation only; external holdout is for screening',
        fresh_utility_tolerances=dict(
            mpjpe_mm=control_fit['lc_utility_mpjpe_tolerance'] * 1000,
            pampjpe_mm=control_fit['lc_utility_pa_tolerance'] * 1000,
            pck_percentage_points=control_fit['lc_utility_pck_tolerance'] * 100),
        dominance='point estimates only: lower T1, no worse clean MPJPE/PA-MPJPE and all five PCK values',
        guarded_surrogate='paired_guard alone requires seven exact zero-tolerance gates against current-stage clean twins; fresh victim may fail',
        paired_guard_internal_utility_tolerances=dict(mpjpe_mm=0., pampjpe_mm=0., pck_percentage_points=0.))
    return contract


def _resolved_cells(data_home, outdir, device, num_workers, *, epochs,
                    train_samples, eval_samples, distortion_samples, relative_l2,
                    fit_options, skip_distortion):
    from paired_guard_fit import resolve_paired_fit_config
    from train_backdoor import _resolve_training_config, _validate_training_contract
    control_fit, paired_fit = _fit_options(fit_options)
    controls = base.build_matrix(data_home, outdir, device, num_workers,
        epochs=epochs, train_samples=train_samples, eval_samples=eval_samples,
        distortion_samples=distortion_samples, relative_l2=relative_l2,
        fit_options=control_fit, skip_clean_probe=True, skip_distortion=skip_distortion)
    old = {cell['method_key']: cell for cell in controls['cells']}
    cells = []
    for key, label, trigger, variant, size in METHODS:
        cfg = copy.deepcopy(old['lc_trainaware' if variant else 'blended']['cfg'])
        cfg['draft_profile'] = PROFILE
        if variant == 'paired_guard':
            cfg.update(paired_fit)
            cfg.update(lc_variant=variant, lc_bank_size=size)
            cfg = resolve_paired_fit_config(cfg)
        cfg = _resolve_training_config(cfg)
        _validate_training_contract(cfg)
        folder = Path(outdir).resolve() / key
        cells.append(dict(method_key=key, label=label,
            group='shared_poison_proposed_development', cfg=cfg,
            ckpt_dir=str(folder), eval_cache=str(folder / 'eval_cache.json')))
    return cells, control_fit, paired_fit, _metrics_contract(controls['metrics_contract'], control_fit)


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4, *,
                 epochs=15, train_samples=20000, eval_samples=4096,
                 distortion_samples=256, relative_l2=.10, fit_options=None,
                 skip_clean_probe=False, skip_distortion=False):
    """Resolve exactly three recipes; no dataset access, fitting or CUDA check.

    skip_clean_probe is accepted for compatibility; clean-victim probes are
    explicitly unsupported in this three-victim experiment in either case.
    """
    cells, fit, paired_fit, contract = _resolved_cells(data_home, outdir, device, num_workers,
        epochs=epochs, train_samples=train_samples, eval_samples=eval_samples,
        distortion_samples=distortion_samples, relative_l2=relative_l2,
        fit_options=fit_options, skip_distortion=skip_distortion)
    sources = _source_provenance()
    for cell in cells:
        cell['recipe_sha256'] = _recipe_sha(cell['cfg'], sources)
    matrix = dict(schema=1, status=STATUS, DRAFT_ONLY=True, dataset='mmfi', seed=42,
        draft_profile=PROFILE, created_utc=datetime.now(timezone.utc).isoformat(),
        sources=dict(source_sha256=sources, references=copy.deepcopy(shared.REFERENCES),
            implementation='Independent Proposed development; not source-paper reproduction'),
        fitting_options=fit, paired_fitting_options=paired_fit, cells=cells,
        metrics_contract=contract)
    matrix['plan_sha256'] = _plan_sha(matrix)
    _validate_manifest(matrix)
    return matrix


def _validate_manifest(matrix):
    if (matrix.get('schema') != 1 or matrix.get('status') != STATUS
            or matrix.get('DRAFT_ONLY') is not True or matrix.get('draft_profile') != PROFILE
            or matrix.get('dataset') != 'mmfi' or matrix.get('seed') != 42):
        raise ValueError('Not the isolated three-arm paired-guard draft manifest')
    if matrix.get('plan_sha256') != _plan_sha(matrix):
        raise ValueError('Plan fingerprint differs from its contents')
    sources = _source_provenance()
    if matrix['sources']['source_sha256'] != sources:
        raise ValueError('Scientific source changed; use a NEW output directory')
    if [c['method_key'] for c in matrix['cells']] != [m[0] for m in METHODS]:
        raise ValueError('All three planned cells must remain in fixed order')
    cfg = matrix['cells'][0]['cfg']
    outdir = Path(matrix['cells'][0]['ckpt_dir']).resolve().parent
    contract = matrix['metrics_contract']
    if contract.get('distortion') not in ('missing_by_explicit_skip', 'same holdout CSI pairs, no HPE forward'):
        raise ValueError('Distortion contract mismatch')
    expected, fit, paired_fit, expected_contract = _resolved_cells(
        Path(cfg['dataset_root']).resolve().parents[1], outdir,
        cfg['device'], cfg['num_workers'], epochs=cfg['epochs'],
        train_samples=cfg['draft_train_samples'], eval_samples=cfg['draft_eval_samples'],
        distortion_samples=contract['distortion_samples'], relative_l2=cfg['lc_relative_l2'],
        fit_options=matrix['fitting_options'],
        skip_distortion=contract['distortion'] == 'missing_by_explicit_skip')
    if matrix['fitting_options'] != fit or matrix.get('paired_fitting_options') != paired_fit:
        raise ValueError('Fitting defaults or strict paired-guard tolerances differ from plan')
    if contract != expected_contract:
        raise ValueError('Official-test/metric/clean-victim-probe contract mismatch')
    for cell, canonical in zip(matrix['cells'], expected):
        if (shared._scientific(cell['cfg']) != shared._scientific(canonical['cfg'])
                or cell['recipe_sha256'] != _recipe_sha(cell['cfg'], sources)
                or any(cell.get(k) != canonical[k] for k in ('label', 'group', 'ckpt_dir', 'eval_cache'))):
            raise ValueError(f"{cell['method_key']}: immutable operator recipe or shared protocol differs from plan")


def build_summary(matrix):
    from mmfi_tables import metric_row, dose_rows
    _validate_manifest(matrix)
    cells = [_prepared_cell(cell) for cell in matrix['cells']]
    if any(cell is None for cell in cells):
        raise ValueError('All three cells require a valid frozen prepared config')
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
        rows.append(dict(status=STATUS, method_key=key, method=cell['label'], seed=42,
            epochs=cell['cfg']['epochs'], rho=cell['cfg']['rho'],
            clean_mpjpe_mm=metric['clean_mpjpe_mm'], clean_pampjpe_mm=metric['clean_pampjpe_mm'],
            t1_mpjpe_mm=metric['tmpjpe_mm'], **{k: metric[k] for k in MAIN_PCK},
            cfg_fingerprint=source['cfg_fingerprint'], recipe_sha256=cell['recipe_sha256'],
            lc_artifact_sha256=cell['cfg'].get('lc_artifact_sha256'),
            lc_fitting_sha256=cell['cfg'].get('lc_fitting_sha256')))
        series, _ = dose_rows(cell, res)
        doses.extend(dict(status=STATUS, **entry) for entry in series)
        provenance.append(source)
        if cell['cfg'].get('lc_variant'):
            record = json.loads((Path(cell['ckpt_dir']) / 'fitting.json').read_text(encoding='utf-8'))
            fitting_audit[key] = {name: record.get(name) for name in (
                'surrogate_initialization_seeds', 'surrogate_count', 'selected_round', 'selected_score',
                'selected_utility_gate_passed', 'utility_gate_required', 'no_eligible_candidate',
                'utility_reference', 'utility_tolerances', 'lookahead',
                'lookahead_is_full_training_bilevel', 'fit_settings', 'clean_twin_clone_proof',
                'clean_twin_actual_erm_steps_per_member', 'clean_twin_virtual_sgd_steps_per_member',
                'surrogate_actual_erm_steps_per_member', 'surrogate_virtual_sgd_steps_per_member',
                'paired_batch_audit', 'guard_activation_counts', 'guard_activation_counts_per_surrogate',
                'guard_observations_per_metric', 'default_full_pool_compute', 'attacker_outer_steps',
                'utility_tolerances_immutable', 'candidate_selection', 'utility_scope')}
            fitting_audit[key]['history'] = record.get('history', [])
            fitting_audit[key]['rounds'] = [entry for entry in record.get('history', [])
                                          if entry.get('stage') == 'alternation']
    return dict(status=STATUS, DRAFT_ONLY=True, draft_profile=PROFILE,
        plan_sha256=matrix['plan_sha256'], rows=rows, dose_response=doses,
        blended_comparison=compare_to_blended(rows, matrix['metrics_contract']['fresh_utility_tolerances']),
        audit=audit, sources=matrix['sources'], metrics_contract=matrix['metrics_contract'],
        provenance=provenance, fitting_audit=fitting_audit, limitations=[
            'All three planned rows retained, including failed hypotheses; historical runs are not overwritten.',
            'One seed and TRAIN-only development holdout: no statistical superiority or paper-result claim.',
            'Extra attacker surrogate and clean-twin compute is disclosed; no attacker weights transfer to victims.',
            'Two-step SGD/momentum lookahead uses cloned frozen BatchNorm buffers; not full victim unrolling.',
            'Seven strict internal clean-twin utility gates do not guarantee fresh-victim utility or lower T1.',
            'Strict pointwise dominance is separate from lower T1 with tolerated clean degradation.',
            'Clean-victim trigger probes are unsupported and explicitly not run; no fourth fresh victim.',
            'Common actual Linf/relative-L2 ceilings do not imply equal realized perturbation.',
            'Digital distortion does not establish RF stealthiness or over-the-air feasibility.',
            'Repeated development on this benchmark is not a blinded first test.',
            'Input manifests bind ordered file/frame identities, not every CSI/pose file byte.'])


def export_summary(matrix, outdir, device='cpu'):
    report = build_summary(matrix)
    outdir = Path(outdir)
    if matrix['metrics_contract']['distortion'] != 'missing_by_explicit_skip':
        distortion = shared.build_distortion(matrix, report['audit'], matrix['metrics_contract']['distortion_samples'])
        _atomic_json(outdir / 'input_distortion.json', distortion)
        _write_csv(outdir / 'input_distortion.csv', distortion['rows'], list(distortion['rows'][0]))
    for name, rows in (('draft_summary', report['rows']), ('dose_response', report['dose_response']),
                       ('blended_comparison', report['blended_comparison'])):
        _write_csv(outdir / f'{name}.csv', rows, list(rows[0]))
        if name != 'draft_summary':
            _atomic_json(outdir / f'{name}.json', dict(status=STATUS, plan_sha256=matrix['plan_sha256'], rows=rows))
    _atomic_json(outdir / 'draft_summary.json', report)
    lines = [f'# {STATUS}', '', 'Three-arm Proposed development screen; all rows retained.', '',
        '| Method | MPJPE mm | PA-MPJPE mm | Target T1 mm | Relative PCK .5/.4/.3/.2/.1 % |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in report['rows']:
        pck = '/'.join(f'{row[k]:.2f}' for k in MAIN_PCK)
        lines.append(f"| {row['method']} | {row['clean_mpjpe_mm']:.3f} | {row['clean_pampjpe_mm']:.3f} | {row['t1_mpjpe_mm']:.3f} | {pck} |")
    lines += ['', 'Comparison with fresh Blended (point estimates only):', '',
        '| Proposed arm | Delta T1 mm | Strict dominance, all eight metrics | Lower T1 with tolerated clean tradeoff |',
        '| --- | ---: | --- | --- |']
    for row in report['blended_comparison']:
        lines.append(f"| {row['method']} | {row['delta_t1_mpjpe_mm']:.3f} | {row['pointwise_dominates_blended']} | {row['lower_t1_with_tolerated_clean_tradeoff']} |")
    guard = report['fitting_audit']['lc_paired_guard']
    lines += ['', f"Internal seven-metric zero-tolerance clean-twin gates passed for selected candidate: {guard.get('selected_utility_gate_passed')}; no eligible round: {guard.get('no_eligible_candidate')}.",
        'Gate failure is retained as a failed hypothesis, not successful utility protection.',
        f"Clean twin actual ERM steps per member: {guard.get('clean_twin_actual_erm_steps_per_member')}; virtual SGD steps per member: {guard.get('clean_twin_virtual_sgd_steps_per_member')}.",
        f"Surrogate actual ERM steps per member: {guard.get('surrogate_actual_erm_steps_per_member')}; virtual SGD steps per member: {guard.get('surrogate_virtual_sgd_steps_per_member')}.",
        f"Fresh-victim tradeoff tolerances: {matrix['metrics_contract']['fresh_utility_tolerances']}.",
        'Lower T1 with tolerated clean degradation is not pointwise dominance.',
        'PCK thresholds are relative. All 18 records remain in dose_response.csv/json.',
        'Clean-victim probes: unsupported_not_run; no fourth victim. Input distortion is a separate diagnostic.',
        f"Seed 42; rho=.1; epochs={matrix['cells'][0]['cfg']['epochs']}; TRAIN holdout only; ordinary victim MPJPE.",
        f"Plan SHA256: `{matrix['plan_sha256']}`", '', 'Limitations:',
        *[f'- {item}' for item in report['limitations']]]
    (outdir / 'draft_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def _validate_cuda_devices(devices):
    """Explain process-visible logical IDs before worker launch or data loading."""
    import torch
    cuda = [d for d in devices if d.startswith('cuda:')]
    if not cuda:
        return
    count = torch.cuda.device_count()
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
    available = torch.cuda.is_available()
    invalid = [d for d in cuda if int(d.split(':')[1]) >= count]
    if not available or invalid:
        logical = ', '.join(f'cuda:{i}' for i in range(count)) or 'none'
        raise ValueError(f'CUDA preflight: device_count={count}, CUDA_VISIBLE_DEVICES={mask!r}; '
            f'requested={cuda}, valid logical IDs={logical}. CUDA indices are logical within this process. '
            'For physical GPUs 0,1,3 with CUDA_VISIBLE_DEVICES=0,1,3, use --devices cuda:0 cuda:1 cuda:2. '
            'Set the mask for this command and use visible logical IDs, or use --devices cpu for a synthetic check.')


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
    parser.add_argument('--cells', nargs='+', choices=[m[0] for m in METHODS])
    for key, value in BANK_FIT_DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'),
            type=int if isinstance(value, int) else float, default=value)
    for name in ('fresh', 'dry-run', 'export-only', 'skip-distortion', 'skip-clean-probe'):
        parser.add_argument('--' + name, action='store_true')
    parser.add_argument('--_cell', choices=[m[0] for m in METHODS], help=argparse.SUPPRESS)
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
        path = outdir / 'paired_guard.resolved.json'
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
            print(f'[{STATUS}] DRY RUN: three recipes; no data/GPU/fitting/training.\n{path}')
            return 0
        if not args.export_only:
            _validate_cuda_devices(args.devices)
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
