"""Six frozen, dose-adapted CSI baselines against the existing A1 victim.

This is a native-operator, common-task comparison, NOT an equal-noise-budget
ranking. It never schedules Proposed or rewrites its historical cache. This
selected subset does not support claims of superiority over all attacks.
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
from run_method_drafts import _file_sha, _verified_cache, _write_csv
from run_proposed_ablation import (
    _configs as _a1_configs, _dose_errors, EXPECTED_COUNTS, GRID,
)
from run_traditional_comparison import (
    _run_cell, _collect_input_record, audit_common_inputs, SOURCE_REFERENCES,
)

HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_a1_native_comparison_v3'
STATUS = 'NATIVE_SOURCE_ADAPTED_MMFI_COMPARISON_SINGLE_SEED'
FOLDER = 'mmfi_a1_native_comparison_s42_v3'
MANIFEST = 'a1_native_comparison.resolved.json'
METHODS = (
    ('badnets', 'BadNets-CSI'),
    ('wanet_source', 'WaNet-CSI'),
    ('ftrojan', 'FTrojan-CSI'),
    ('fiba', 'FIBA-CSI'),
    ('sig_source', 'SIG-CSI (BackdoorBench-default, pose/dose-adapted)'),
    ('bpp_source', 'BppAttack-CSI (BackdoorBench-default, pose/dose-adapted)'),
)
STOCHASTIC_CELLS = {'wanet_source', 'fiba', 'bpp_source'}
SUMMARY_COLUMNS = (
    'method_key', 'method', 'result_origin', 'seed', 'epochs', 'rho',
    'train_samples', 'eval_samples', 'n_poison', 'n_cover',
    't1_mpjpe_mm', 'clean_mpjpe_mm', 'clean_pampjpe_mm',
    'mean_dose_control_error_mm', 'mean_positive_dose_tmpjpe_mm',
    'clean_pck_0.2_pct', 'cfg_fingerprint', 'train_pair_ids_sha256',
    'eval_pair_ids_sha256', 'poison_plan_sha256',
)
DOSE_COLUMNS = (
    'method_key', 'method', 'seed', 'd', 'no_trigger_tmpjpe_mm',
    'triggered_tmpjpe_mm', 'improvement_mm', 'baseline_source',
    'dose_control_error_mm',
)


def _source_provenance():
    paths = (
        'run_a1_native_comparison.py', 'a1_frozen_reference.py',
        'run_proposed_ablation.py', 'run_traditional_comparison.py',
        'run_mmfi_tables.py', 'run_method_drafts.py', 'run_peak_confirmation.py',
        'train_backdoor.py', 'configs/mmfi/attack_bend.yaml', 'mmfi_tables.py',
        'attack/trigger.py', 'attack/traditional.py', 'attack/wanet_source.py',
        'attack/frequency_baselines.py', 'attack/static_source_baselines.py',
        'attack/method_drafts.py', 'attack/poison.py', 'attack/payload.py',
        'attack/skeleton_mmfi.py', 'data_utils/feeder.py',
        'data_utils/draft_subset.py', 'eval/metrics.py', 'models/factory.py',
        'models/hpeli.py', 'models/sk_network.py',
        'third_party/backdoorbench/patch.py',
        'third_party/backdoorbench/SOURCE_MANIFEST.json',
    )
    return {name: _file_sha(HERE / name) for name in paths}


def _method_config(data_home, device, num_workers, key):
    from train_backdoor import _resolve_training_config, _validate_training_contract
    if key not in dict(METHODS):
        raise ValueError(f'Unknown native baseline: {key}')
    cfg = copy.deepcopy(_a1_configs(data_home, device, num_workers, 42,
                                    factorial=True)[0])
    # A1 is the protocol authority. Remove only its attack/profile metadata;
    # keep its victim/data/target/dose/loader policy unchanged.
    cfg = {k: v for k, v in cfg.items()
           if k not in ('confirmation_profile', 'trigger') and not k.startswith('method_')}
    cfg.update(trigger=key, traditional_comparison_profile=PROFILE)
    if key == 'badnets':
        cfg.update(badnets_patch_subcarriers=3, badnets_patch_packets=3,
                   badnets_patch_opacity=1.0, badnets_pattern='white')
    elif key == 'wanet_source':
        cfg.update(wanet_grid_size=4, wanet_strength=0.5,
                   wanet_grid_rescale=1.0, wanet_cover_ratio=0.2, num_workers=0)
    elif key == 'fiba':
        cfg.update(fiba_alpha=0.15, fiba_beta=0.1, fiba_cross_ratio=1.0,
                   clean_label_cover_ratio=0.1, num_workers=0)
    elif key == 'sig_source':
        cfg.update(sig_delta_255=40.0, sig_frequency=6,
                   clean_label_cover_ratio=0.0)
    elif key == 'bpp_source':
        cfg.update(bpp_levels=8, bpp_dithering=False, bpp_negative_ratio=0.1,
                   clean_label_cover_ratio=0.1, num_workers=0)
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    if any(k.startswith('comparison_peak_') for k in cfg):
        raise ValueError('Native A1 comparison refuses external peak projections')
    return cfg


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4, *,
                 a1_reference=None):
    """Resolve scientific settings without opening the dataset/reference/CUDA."""
    from train_backdoor import _CHECKPOINT_SCHEMA, _RESULT_SCHEMA
    from attack.frequency_baselines import SOURCES as FREQUENCY_SOURCES
    from attack.static_source_baselines import SOURCES as STATIC_SOURCES
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    reference = Path(a1_reference or data_home / 'runs/mmfi_proposed_factorial_ablation_s42_v1').resolve()
    cells = []
    for key, label in METHODS:
        folder = outdir / key
        cells.append(dict(method_key=key, label=label, group='targeted_pose_task',
            dependencies=[], tables=[], cfg=_method_config(data_home, device, num_workers, key),
            ckpt_dir=str(folder), eval_cache=str(folder / 'eval_cache.json')))
    matrix = dict(schema=1, dataset='mmfi', seed=42, fresh_results_only=True,
        traditional_comparison_profile=PROFILE, status=STATUS,
        created_utc=datetime.now(timezone.utc).isoformat(), cells=cells,
        sources=dict(source_sha256=_source_provenance(),
            references=dict(backdoorbench={key: copy.deepcopy(value)
                for key, value in SOURCE_REFERENCES.items() if not key.startswith('blended')},
                frequency=copy.deepcopy(FREQUENCY_SOURCES),
                static=copy.deepcopy(STATIC_SOURCES)),
            baseline_config='configs/mmfi/attack_bend.yaml'),
        metrics_contract=dict(profile=PROFILE,
            comparison_scope='selected_native_csi_pose_adaptations',
            checkpoint_schema=_CHECKPOINT_SCHEMA, result_schema=_RESULT_SCHEMA,
            a1_reference_path=str(reference), reuse_a1=True, train_a1=False,
            evaluation_role='full_official_test', official_test_used=True,
            full_split_counts=copy.deepcopy(EXPECTED_COUNTS), seeds=[42], epochs=50,
            rho=0.1, dose_grid=list(GRID), target_joints=[2, 3],
            compared_baselines=[key for key, _ in METHODS],
            main_metrics=['t1_mpjpe_mm', 'clean_mpjpe_mm', 'clean_pampjpe_mm'],
            pck='CSV clean PCK@0.2 is a clean utility diagnostic; Target-PCK is not implemented',
            budget_mode='native', shared_peak_ceiling=False, matches_noise_budget=False,
            dose_training='same paired U(0.2,1) trigger and pose-target doses',
            poison_plan='identical ordered indices and dose values; 13305 poison examples',
            cover_ratios=dict(wanet_source=0.2, fiba=0.1, bpp_source=0.1),
            loader_policy='A1 persistent-worker default for static operators; source stochastic covers require workers=0',
            excluded_methods=dict(
                blended='excluded from this selected method set at user request; historical results retained; no superiority over it is claimed',
                infocom2025_por='encoder representation objective, not targeted-pose ranking'),
            uncertainty='single seed; no statistical superiority is estimated'))
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    return matrix


def _validate_manifest(matrix):
    if (matrix.get('schema') != 1 or matrix.get('dataset') != 'mmfi'
            or matrix.get('seed') != 42 or matrix.get('status') != STATUS
            or matrix.get('traditional_comparison_profile') != PROFILE
            or matrix.get('fresh_results_only') is not True or matrix.get('DRAFT_ONLY')):
        raise ValueError('Not the native A1 comparison manifest')
    if matrix.get('plan_sha256') != _plan_fingerprint(matrix):
        raise ValueError('A1 comparison plan fingerprint differs from its contents')
    if matrix['sources']['source_sha256'] != _source_provenance():
        raise ValueError('A1 comparison source changed; use a NEW output directory')
    cells = matrix['cells']
    if [c['method_key'] for c in cells] != [key for key, _ in METHODS]:
        raise ValueError('Every planned native baseline must remain in fixed order')
    first = cells[0]
    data_home = Path(first['cfg']['dataset_root']).parent.parent
    outdir = Path(first['ckpt_dir']).parent
    expected = build_matrix(data_home, outdir, first['cfg']['device'],
        first['cfg']['num_workers'], a1_reference=matrix['metrics_contract']['a1_reference_path'])
    if (matrix['metrics_contract'] != expected['metrics_contract']
            or matrix['sources'] != expected['sources']):
        raise ValueError('Native comparison source/metrics contract changed')
    for cell, target in zip(cells, expected['cells']):
        cfg = _method_config(data_home, cell['cfg']['device'],
                             cell['cfg']['num_workers'], cell['method_key'])
        if cell['cfg'] != cfg or any(cell.get(k) != target[k] for k in
                ('method_key', 'label', 'group', 'dependencies', 'tables', 'ckpt_dir', 'eval_cache')):
            raise ValueError(f"{cell['method_key']}: native comparison configuration changed")


def _limitations():
    return [
        'This selected native-operator comparison does not match realized Linf, L2 or RF budgets.',
        'This is a selected method set; no claim of superiority over all backdoor attacks is supported.',
        'Blended is omitted from this selected comparison at user request; its previous results are retained and no superiority over it is claimed.',
        'Operators are dose-adapted to rectangular amplitude CSI and common pose targets, not unchanged source classification pipelines.',
        'BadNets uses the pinned third-party BackdoorBench masked-patch operator with a declared 3x3 CSI patch; this is not a claim about the original authors\' universal hyperparameters.',
        'WaNet retains 20%, FIBA 10% and BppAttack 10% extra clean-label covers; modified-input counts differ.',
        'WaNet/FIBA/BppAttack use workers=0 for checkpointed fresh cover RNG; SIG and other static operators retain the A1 worker policy. Identical minibatch RNG trajectories are not claimed.',
        'SIG uses the pinned BackdoorBench default sine amplitude 40/255 with frequency parameter 6, but dirty-label pose targets replace the original label-consistent classifier attack.',
        'On the ten-packet CSI grid, SIG frequency parameter 6 aliases negative 4; no equivalent physical RF frequency in Hz is claimed.',
        'BppAttack uses the pinned BackdoorBench quantization variant: 8 levels, no dithering and 10% clean-label residual negatives. This does not reproduce the full paper PGD/SupCon training; common HPE MPJPE replaces classification CE.',
        'BppAttack fixes cover recipient identities across epochs to the common plan; BackdoorBench resamples recipient slots per minibatch. Residual donors are sampled lazily with replacement from unaugmented TRAIN CSI rather than without replacement within source augmented image-residual batches.',
        'A1 is reused with its original configuration, inputs, poison audit and artifact hashes; it is not retrained, reoptimized or reclassified as a fresh result.',
        'Only clean PCK@0.2 is retained as a CSV utility diagnostic; it is not Target-PCK or ASR.',
        'Single seed and development-selected Proposed; no uncertainty, blinded-test status or statistical superiority is estimated.',
        'Digital amplitude perturbations do not establish physical RF stealthiness or over-the-air feasibility.',
        'INFOCOM/POR is excluded from this ranking because its representation objective does not optimize the pose target.',
    ]


def build_summary(matrix):
    from a1_frozen_reference import load_a1_reference, validate_a1_reference
    from mmfi_tables import metric_row, dose_rows
    _validate_manifest(matrix)
    audit = audit_common_inputs(matrix)
    reference = load_a1_reference(matrix['metrics_contract']['a1_reference_path'])
    validate_a1_reference(reference, matrix, audit)
    rows, doses, provenance = [], [], []
    for cell in matrix['cells']:
        key, cfg = cell['method_key'], cell['cfg']
        res, source = _verified_cache(cell)
        if (res.get('training_contract') != 'ordinary_erm'
                or res.get('attacker_access') != 'data_only'
                or res.get('dose_coupling') != 'paired'
                or res.get('dose_grid') != GRID
                or res.get('n_total') != EXPECTED_COUNTS['train']
                or res.get('n_poison') != audit['counts'][key]['n_poison']
                or res.get('n_cover') != audit['counts'][key]['n_cover']
                or res.get('poison_plan_sha256') != audit['poison_plan_sha256'][key]):
            raise ValueError(f'{key}: cached results disagree with native comparison audit')
        series, _ = dose_rows(cell, res)
        if [r['d'] for r in series] != GRID:
            raise ValueError(f'{key}: exact six-dose evaluation required')
        dce = _dose_errors(res)
        row = metric_row(cell, res)
        row.update(result_origin='fresh_baseline', epochs=50, rho=0.1,
            train_samples=EXPECTED_COUNTS['train'], eval_samples=EXPECTED_COUNTS['eval'],
            **audit['counts'][key], t1_mpjpe_mm=row.pop('tmpjpe_mm'),
            mean_dose_control_error_mm=sum(dce[1:]) / 5,
            mean_positive_dose_tmpjpe_mm=sum(r['triggered_tmpjpe_mm'] for r in series[1:]) / 5,
            cfg_fingerprint=source['cfg_fingerprint'],
            train_pair_ids_sha256=audit['inputs'][key]['train']['pair_ids_sha256'],
            eval_pair_ids_sha256=audit['inputs'][key]['eval']['pair_ids_sha256'],
            poison_plan_sha256=audit['poison_plan_sha256'][key])
        rows.append({name: row[name] for name in SUMMARY_COLUMNS})
        doses.extend(dict(r, dose_control_error_mm=error) for r, error in zip(series, dce))
        provenance.append(source)
    a1row = dict(reference['row'], method_key='a1', method='Proposed A1 (reused)',
                 result_origin='frozen_ablation_reference')
    rows.append({name: a1row.get(name) for name in SUMMARY_COLUMNS})
    doses.extend(dict(row, method_key='a1', method='Proposed A1 (reused)')
                 for row in reference['dose_response'])
    return dict(status='a1_native_comparison_complete', profile=PROFILE, seed=42,
        plan_sha256=matrix['plan_sha256'], rows=rows, dose_response=doses,
        audit=audit, provenance=provenance, a1_reference=reference,
        sources=matrix['sources'], metrics_contract=matrix['metrics_contract'],
        limitations=_limitations())


def export_summary(matrix, outdir):
    report = build_summary(matrix)
    outdir = Path(outdir)
    _atomic_json(outdir / 'a1_native_comparison.json', report)
    _write_csv(outdir / 'a1_native_comparison.csv', report['rows'], SUMMARY_COLUMNS)
    _write_csv(outdir / 'dose_response.csv', report['dose_response'], DOSE_COLUMNS)
    _atomic_json(outdir / 'dose_response.json', dict(profile=PROFILE,
        plan_sha256=matrix['plan_sha256'], rows=report['dose_response'], units='mm'))
    lines = ['# Selected native CSI/HPE comparison against A1', '',
        'MM-Fi protocol1-s1; seed 42; 50 epochs; rho=0.1; paired U(0.2,1).', '',
        f'{len(METHODS)} fresh baseline victims and one immutable existing A1 reference; no peak projection.', '',
        '| Method | T-MPJPE@1 (mm) | Clean MPJPE (mm) | Clean PA-MPJPE (mm) |',
        '| --- | ---: | ---: | ---: |']
    for row in report['rows']:
        lines.append(f"| {row['method']} | {row['t1_mpjpe_mm']:.3f} | {row['clean_mpjpe_mm']:.3f} | {row['clean_pampjpe_mm']:.3f} |")
    lines.extend(['', 'Six-dose target errors and dose-control errors are retained in CSV/JSON.',
        'Target-PCK is not implemented and no clean PCK is relabeled as attack success.', '',
        'Limitations:', *[f'- {item}' for item in report['limitations']], '',
        f"Plan SHA256: `{matrix['plan_sha256']}`."])
    path = outdir / 'a1_native_comparison.md'
    temporary = path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--a1-reference', type=Path)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cells', nargs='+', choices=[key for key, _ in METHODS])
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--export-only', '--summary-only', dest='export_only', action='store_true')
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
        cells = [c for c in matrix['cells'] if c['method_key'] == args._cell]
        if len(cells) != 1:
            raise ValueError('Worker requested an unplanned baseline')
        _run_cell(cells[0])
        return 0
    if args.data_home is None:
        raise ValueError('--data-home is required')
    if (not args.devices or len(set(args.devices)) != len(args.devices)
            or any(d != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None for d in args.devices)):
        raise ValueError('--devices must list distinct logical cuda:N devices, or cpu')
    if args.dry_run and args.export_only:
        raise ValueError('Choose only one of --dry-run or --export-only')
    outdir = (args.outdir or args.data_home / 'runs' / FOLDER).resolve()
    if args.fresh and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('--fresh refuses non-empty output directories; use a NEW path')
    with _cell_lock(outdir):
        manifest_path = outdir / MANIFEST
        requested = build_matrix(args.data_home, outdir, args.devices[0], args.num_workers,
                                 a1_reference=args.a1_reference)
        _validate_manifest(requested)
        if manifest_path.exists():
            matrix = json.loads(manifest_path.read_text(encoding='utf-8'))
            _validate_manifest(matrix)
            if matrix['plan_sha256'] != requested['plan_sha256']:
                raise ValueError('Native comparison settings/source changed; use a NEW directory')
            for cell in matrix['cells']:
                cell['cfg']['device'] = args.devices[0]
                cell['cfg']['num_workers'] = 0 if cell['method_key'] in STOCHASTIC_CELLS else args.num_workers
        else:
            if any(path.name != 'run.lock' for path in outdir.iterdir()):
                raise ValueError('Output contains unrelated/historical files; use a NEW directory')
            matrix = requested
        _atomic_json(manifest_path, matrix)
        if args.dry_run:
            print(f'[a1-native] DRY RUN: {len(METHODS)} baseline cells; A1 reference reused, never trained; seed42 rho=.1 full50; NO data/reference/GPU/training.\n{manifest_path}', flush=True)
            return 0
        from a1_frozen_reference import (
            load_a1_reference, validate_a1_reference, validate_a1_input_record,
        )
        reference = load_a1_reference(matrix['metrics_contract']['a1_reference_path'])
        validate_a1_reference(reference, matrix, None)
        if not args.export_only:
            _check_inputs(matrix, args.devices)
            validate_a1_input_record(reference, _collect_input_record(matrix['cells'][0]))
            run_matrix(matrix, manifest_path, args.devices, args.cells, worker_script=Path(__file__))
        missing = [c['method_key'] for c in matrix['cells'] if not _complete(c)]
        if missing:
            if args.cells and not args.export_only:
                print(f'[a1-native] Selected cells finished; report awaits: {missing}', flush=True)
                return 0
            raise ValueError(f'Every planned baseline must finish before exporting: {missing}')
        export_summary(matrix, outdir)
        print(f'[a1-native] Complete: {outdir / "a1_native_comparison.md"}', flush=True)
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f'[a1-native] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
