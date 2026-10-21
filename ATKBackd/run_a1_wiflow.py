"""Real MM-Fi Clean + A1 WiFlow pilot, seed42, fifty epochs each.

One WiFlow worker at a time (alongside the separately owned MetaFi++ job).
No synthetic probe, historical victim reuse, full-paper-table claim or retry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import csv
import os
from pathlib import Path
import re
import sys

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import run_a1_three_tables as frozen
from a1_backbone_adapter import _private_module
from a1_wiflow_adapter import RECIPE_ID, recipe_config, recipe_for, training_engine, evaluation_engine

HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_wiflow_clean_a1_seed42_pilot_v1'
MANIFEST = 'wiflow_a1.resolved.json'
KEYS = ('clean', 'a1')
BASE = _private_module('run_a1_cross_backbones.py', '_wiflow_cross_runtime')
BASE.PROFILE = PROFILE
BASE.training_engine = training_engine
BASE.evaluation_engine = evaluation_engine
# The inherited scheduler must spawn THIS worker entrypoint, never the old CLI.
BASE.__file__ = str(Path(__file__).resolve())


def source_hashes():
    from probe_wiflow_mmfi import verify_local_sources
    verify_local_sources()
    names = ('run_a1_wiflow.py', 'a1_wiflow_adapter.py',
        'models/wiflow_mmfi.py', 'models/wiflow_sources.json',
        'models/_vendor/__init__.py', 'models/_vendor/wiflow_mmfi.py',
        'models/_vendor/licenses/WiFlow-APACHE-2.0.txt', 'probe_wiflow_mmfi.py')
    return dict(BASE.source_hashes(), **{name: frozen._file_sha(HERE / name) for name in names})


def configuration(data_home, device, key):
    if key not in KEYS:
        raise ValueError('WiFlow pilot supports only Clean and A1, seed42')
    from train_backdoor import _resolve_training_config
    cfg = recipe_config(frozen._configuration(Path(data_home), device, 0, 42, key))
    cfg['wiflow_pilot_profile'] = PROFILE
    return _resolve_training_config(cfg)


def build_matrix(data_home, outdir, device='cuda:0'):
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    forbidden = ('cross_backbones.resolved.json', 'three_tables.resolved.json',
        'a1_defenses.resolved.json', 'a1_defenses_v2.resolved.json', 'checkpoint.pt')
    if any((parent / name).exists() for parent in (outdir, *outdir.parents) for name in forbidden):
        raise ValueError('WiFlow needs a separate output, not inside a historical run')
    cells = []
    for key in KEYS:
        folder = outdir / 'wiflow' / 'seed42' / key
        cell = dict(method_key=f'wiflow__s42__{key}', backbone='wiflow', key=key,
            seed=42, label=frozen.LABELS[key], origin='fresh',
            cfg=configuration(data_home, device, key), tables=['PILOT_NOT_FULL_T1'],
            dependencies=[] if key == 'clean' else ['wiflow__s42__clean'],
            run_dir=str(folder), ckpt_dir=str(folder), eval_cache=str(folder / 'eval_cache.json'),
            artifact_dir=str(outdir / 'wiflow' / 'evaluations' / f's42__{key}'))
        if key == 'clean':
            cell['evaluation_trigger_cfg'] = configuration(data_home, device, 'a1')
        cells.append(cell)
    matrix = dict(schema=1, profile=PROFILE, dataset='mmfi', backbones=['wiflow'],
        seeds=[42], data_home=str(data_home), outdir=str(outdir), num_workers=0,
        cells=cells, created_utc=datetime.now(timezone.utc).isoformat(),
        sources=source_hashes(), runtime_versions=BASE.runtime_versions(),
        recipe_id=RECIPE_ID, recipes={'wiflow': recipe_for()},
        metrics_contract=dict(status='ONE_SEED_PILOT_NOT_FULL_PAPER_TABLE',
            epochs=50, checkpoint_epoch=49, counts=dict(frozen.EXPECTED_COUNTS),
            mmfi_protocol='protocol1-s1', split_seed=0, poison_rate=.1,
            target_joints=[2, 3], target_pck_threshold=.2, clean_scale_joints=[5, 12],
            dose_grid=list(frozen.GRID), dce_summary='mean over five positive doses',
            scratch=True, victim_loss='ordinary MPJPE', model_selection_on_test=False),
        limitations=[
            'Author model definitions preserved; native SmoothL1/bone loss is NOT used.',
            'Benchmark MPJPE uses the existing 17-joint MM-Fi order; no native bone graph is executed.',
            'AdamW lr1e-4/wd1e-4/betas(.9,.999)/clip1 from author scalar choices.',
            'Physical32 x accumulation2 = effective64; BN follows physical32, not physical64.',
            'Fixed LR and final epoch49 replace author validation-plateau/early-stop/selection.',
            'FP32 only; no AMP, automatic OOM fallback or skipped failed batches.',
            'Existing A1 CSI normalization, pose units, split, poison and target definitions unchanged.',
            'One seed; no mean/SD claim, other attacks, defense, second dataset or Graph replacement result.',
            'CPU synthetic tests do not verify full MM-Fi/CUDA convergence or clean utility.',
        ])
    matrix['plan_sha256'] = frozen._plan_hash(matrix)
    return matrix


def validate_manifest(matrix):
    if (matrix.get('schema') != 1 or matrix.get('profile') != PROFILE
            or matrix.get('plan_sha256') != frozen._plan_hash(matrix)):
        raise ValueError('Invalid or modified WiFlow manifest')
    canonical = build_matrix(matrix['data_home'], matrix['outdir'], matrix['cells'][0]['cfg']['device'])
    if matrix['plan_sha256'] != canonical['plan_sha256']:
        raise ValueError('WiFlow source/runtime/scientific plan changed; use a NEW output')
    for cell in matrix['cells']:
        # Plan hash ignores only documented execution fields, not the model/seed.
        if cell['cfg'] != configuration(matrix['data_home'], cell['cfg']['device'], cell['key']):
            raise ValueError('Noncanonical WiFlow configuration')


def selected_cells(matrix, clean_only=False):
    return [c for c in matrix['cells'] if not clean_only or c['key'] == 'clean']


def completed(cell, sources):
    return BASE.completed(cell, sources)


def execute_cell(cell, sources):
    return BASE.execute_cell(cell, sources)


def inspect_status(matrix, verify=False):
    missing, invalid = [], []
    for cell in matrix['cells']:
        path = Path(cell['artifact_dir']) / 'evaluation.json'
        if not path.is_file():
            missing.append(cell['method_key'])
            print(f'MISSING {cell["method_key"]}')
            continue
        try:
            if verify and completed(cell, matrix['sources']) is None:
                raise ValueError('Evaluation disappeared or is incomplete')
            print(f'{"VERIFIED" if verify else "PRESENT_UNVERIFIED"} {cell["method_key"]}')
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            invalid.append(cell['method_key'])
            print(f'INVALID {cell["method_key"]}: {exc}')
    return missing, invalid


def export_pilot(matrix):
    records = [completed(cell, matrix['sources']) for cell in matrix['cells']]
    if any(record is None for record in records):
        raise ValueError('Both authenticated Clean and A1 results required')
    identities = [r['provenance']['binding']['ordered_inputs'] for r in records]
    if identities[0] != identities[1]:
        raise ValueError('Clean/A1 ordered full-split identities differ')
    report = dict(status='WIFLOW_CLEAN_A1_SEED42_PILOT_COMPLETE_NOT_FULL_TABLE',
        plan_sha256=matrix['plan_sha256'], seed=42, backbone='wiflow', records=records,
        recipe=matrix['recipes']['wiflow'], limitations=matrix['limitations'])
    folder = Path(matrix['outdir'])
    folder.mkdir(parents=True, exist_ok=True)
    columns = ('t1_mpjpe_mm', 'target_pck_pct', 'clean_mpjpe_mm', 'clean_pampjpe_mm', 'mean_dce_mm')
    headers = ('Method', 'Seed', 'T-MPJPE (mm)', 'Target-PCK (%)', 'Clean MPJPE (mm)', 'Clean PA-MPJPE (mm)', 'DCE (mm)')
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    writer.writerow(headers)
    lines = ['# WiFlow seed42: Clean + A1 pilot (not a full comparison table)', '',
        'One seed; no across-seed uncertainty. Target metrics at d=1; DCE averages five positive doses.',
        'Errors in mm; Target-PCK@0.2 uses GT scale(5,12) on target joints[2,3].', '',
        '| ' + ' | '.join(headers) + ' |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for record in records:
        values = [record['metrics'][name] for name in columns]
        writer.writerow([record['key'], 42, *values])
        formatted = ['--' if value is None else f'{value:.2f}' for value in values]
        lines.append('| ' + ' | '.join([record['key'], '42', *formatted]) + ' |')
    from a1_three_tables_eval import _atomic_bytes
    _atomic_bytes(folder / 'wiflow_pilot.csv', buffer.getvalue().encode('utf-8'))
    _atomic_bytes(folder / 'wiflow_pilot.md', ('\n'.join(lines) + '\n').encode('utf-8'))
    frozen._atomic_json(folder / 'wiflow_pilot.complete.json', report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--clean-only', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    for name in ('dry-run', 'status', 'check-only', 'export-only'):
        mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--_cell', help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.device != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', args.device) is None:
        parser.error('Use cpu or a logical cuda:N device')
    if bool(args._cell) != bool(args._manifest):
        parser.error('Worker requires both --_cell and --_manifest')
    return args


def load_or_build(args):
    manifest = args.outdir.resolve() / MANIFEST
    if manifest.exists():
        matrix = frozen._json(manifest)
        validate_manifest(matrix)
        if (Path(matrix['outdir']) != args.outdir.resolve()
                or (args.data_home and Path(matrix['data_home']) != args.data_home.resolve())):
            raise ValueError('Requested roots differ from saved WiFlow plan')
    else:
        if args.status or args.check_only or args.export_only:
            raise FileNotFoundError(f'No existing WiFlow manifest: {manifest}')
        if args.data_home is None:
            raise ValueError('--data-home required for a new WiFlow pilot')
        if args.outdir.exists() and any(args.outdir.iterdir()):
            raise ValueError('Nonempty output without WiFlow manifest; choose a NEW folder')
        matrix = build_matrix(args.data_home, args.outdir, args.device)
    return matrix, manifest


def main(argv=None):
    args = parse_args(argv)
    if args._cell:
        if args._manifest.resolve() != args.outdir.resolve() / MANIFEST:
            raise ValueError('Worker manifest must belong to its exact output')
        matrix = frozen._json(args._manifest)
        validate_manifest(matrix)
        if Path(matrix['outdir']) != args.outdir.resolve():
            raise ValueError('Worker output differs from manifest')
        cell = next((c for c in matrix['cells'] if c['method_key'] == args._cell), None)
        if cell is None:
            raise ValueError('Unknown WiFlow worker cell')
        execute_cell(cell, matrix['sources'])
        return 0
    matrix, manifest = load_or_build(args)
    if args.dry_run:
        print(f'[wiflow] DRY RUN: {len(selected_cells(matrix, args.clean_only))}/2 real victims; '
            'seed42; 50 epochs; one WiFlow slot; workers0. NO writes/data/CUDA/training.')
        print(f'plan {matrix["plan_sha256"]}\n{manifest}')
        print(f'recipe {matrix["recipes"]["wiflow"]}')
        return 0
    if args.status or args.check_only:
        missing, invalid = inspect_status(matrix, verify=args.check_only)
        return int(bool(args.check_only and (missing or invalid)))
    if not args.export_only:
        frozen._check_inputs(matrix, [args.device])
    with frozen._cell_lock(args.outdir):
        if manifest.exists():
            matrix, manifest = load_or_build(args)
        else:
            frozen._atomic_json(manifest, matrix)
        chosen = selected_cells(matrix, args.clean_only)
        for cell in chosen:
            if (Path(cell['run_dir']) / 'run.lock').exists():
                with frozen._cell_lock(cell['run_dir']):
                    pass
        if not args.export_only:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            frozen._atomic_json(args.outdir / 'runtime_launches' / f'{stamp}_{os.getpid()}.json',
                dict(controller_pid=os.getpid(), plan_sha256=matrix['plan_sha256'],
                    device=args.device, jobs_per_gpu=1, cells=[c['method_key'] for c in chosen]))
            BASE.schedule(matrix, manifest, [args.device], 1, chosen)
        if args.clean_only and not (Path(matrix['cells'][1]['artifact_dir']) / 'evaluation.json').exists():
            print('[wiflow] clean-only finished; A1 still missing, pilot INCOMPLETE')
            return 0
        export_pilot(matrix)
        print(f'[wiflow] PILOT COMPLETE: {args.outdir / "wiflow_pilot.md"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[wiflow] ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
