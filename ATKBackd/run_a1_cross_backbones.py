"""Full MM-Fi T1/T2/T3 on MetaFi++ and GraphPose-Fi, three independent seeds.

The default plan contains 78 fresh victims. HPE-Li runs are never read as model
references or modified. Dry-run/status/check-only do not train or use a GPU.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import run_a1_three_tables as frozen
from a1_backbone_adapter import RECIPE_ID, recipe_for, recipe_config, training_engine, evaluation_engine
from models.cross_backbone_factory import BACKBONES

HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_a1_cross_backbones_full_v1'
MANIFEST = 'cross_backbones.resolved.json'
SEEDS = frozen.SEEDS
KEYS = frozen.KEYS


def runtime_versions():
    try:
        return dict(python=platform.python_version(), **{
            name: version(name) for name in ('torch', 'torchvision', 'numpy', 'scipy', 'PyYAML')})
    except PackageNotFoundError as exc:
        raise RuntimeError(f'Missing required package: {exc.name}. Install dependencies '
                           'in the existing experiment environment before creating a plan; '
                           'torchvision must match the installed torch build.') from exc


def source_hashes():
    names = {'run_a1_cross_backbones.py', 'a1_backbone_adapter.py', 'a1_backbone_report.py',
             'models/cross_backbone_factory.py', 'models/metafi_plus_plus.py',
             'models/metafi_channel_transformer.py', 'models/metafi_sources.json',
             'models/graphpose_fi.py', 'models/graphpose_sources.json'}
    names.update(str(p.relative_to(HERE)).replace('\\', '/')
                 for p in (HERE / 'models').glob('graphpose_*.py'))
    return dict(frozen._source_provenance(), **{
        name: frozen._file_sha(HERE / name) for name in sorted(names)})


def configuration(data_home, device, seed, key, backbone, graph_load_batch_size=32):
    cfg = frozen._configuration(Path(data_home), device, 0, seed, key)
    cfg = recipe_config(cfg, backbone, graph_load_batch_size=graph_load_batch_size)
    # The old confirmation-profile names identify attack variants, not models.
    # Retain them; the distinct backbone recipe/model/source binding is additive.
    cfg['cross_backbone_profile'] = PROFILE
    from train_backdoor import _resolve_training_config
    return _resolve_training_config(cfg)


def build_matrix(data_home, outdir, device='cuda:0', *, backbones=BACKBONES, graph_load_batch_size=32):
    backbones = list(backbones)
    if (not backbones or len(set(backbones)) != len(backbones)
            or set(backbones) - set(BACKBONES)):
        raise ValueError('Select distinct backbones metafiplusplus and/or graphposefi')
    backbones = [name for name in BACKBONES if name in backbones]
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    cells = []
    for backbone in backbones:
        for seed in SEEDS:
            clean_id = f'{backbone}__s{seed}__clean'
            a1 = configuration(data_home, device, seed, 'a1', backbone, graph_load_batch_size)
            for key in KEYS:
                name = f'{backbone}__s{seed}__{key}'
                folder = outdir / backbone / f'seed{seed}' / key
                cells.append(dict(method_key=name, backbone=backbone, key=key, seed=seed,
                    label=frozen.LABELS[key], origin='fresh',
                    cfg=configuration(data_home, device, seed, key, backbone, graph_load_batch_size),
                    tables=(['T1', 'T2', 'T3'] if key == 'a1' else
                            ['T3'] if key in ('a2', 'a3', 'a4') else
                            ['T2'] if key.startswith('rho') else ['T1']),
                    dependencies=[] if key == 'clean' else [clean_id],
                    run_dir=str(folder), ckpt_dir=str(folder),
                    eval_cache=str(folder / 'eval_cache.json'),
                    artifact_dir=str(outdir / backbone / 'evaluations' / f's{seed}__{key}')))
                if key == 'clean':
                    cells[-1]['evaluation_trigger_cfg'] = copy.deepcopy(a1)
    matrix = dict(schema=1, profile=PROFILE, dataset='mmfi', backbones=backbones,
        seeds=list(SEEDS), data_home=str(data_home), outdir=str(outdir), num_workers=0,
        created_utc=datetime.now(timezone.utc).isoformat(), cells=cells,
        sources=source_hashes(), runtime_versions=runtime_versions(),
        recipe_id=RECIPE_ID, graph_load_batch_size=graph_load_batch_size,
        recipes={name: recipe_for(name, graph_load_batch_size) for name in backbones},
        metrics_contract=dict(tables=['T1', 'T2', 'T3'], epochs=50,
            full_split_counts=dict(frozen.EXPECTED_COUNTS), mmfi_protocol='protocol1-s1',
            mmfi_split_seed=0, fresh_victims_per_backbone=39,
            training='ordinary MPJPE ERM; author scalar recipes per backbone; final epoch 49',
            scratch=True, model_selection_on_test=False, target_joints=[2, 3],
            target_pck_threshold=.2, clean_scale_joints=[5, 12],
            dose_grid=list(frozen.GRID), poison_rates=[.1, .2, .4],
            a1_shared_across_tables=True, uncertainty='mean +/- sample SD across 42/43/44',
            no_historical_victim_reuse=True, matches_realized_noise_budget=False,
            not_included=['second dataset', 'defense on new backbones', 'held-out subject split']),
        limitations=[
            'Independent source-informed implementations, not licensed copies or full paper reproductions.',
            'Scratch MetaFi omits ImageNet BatchNorm running statistics present in its public constructor.',
            'Final epoch is fixed; author best-TEST checkpoint selection is deliberately not reproduced.',
            'Graph uses explicit physical batch/accumulation with effective256; BatchNorm follows the physical batch.',
            'Accumulation resets per epoch and sample-weights the last partial group; author global-step carryover and fixed last-group denominator are not reproduced.',
            'Raw data, CSI normalization, attack operators and metric definitions use the existing A1 protocol.',
            'Inspect clean pilot utility before making scientific cross-backbone claims; no full GPU run verified here.',
        ])
    matrix['plan_sha256'] = frozen._plan_hash(matrix)
    return matrix


def validate_manifest(matrix):
    if (matrix.get('schema') != 1 or matrix.get('profile') != PROFILE
            or matrix.get('plan_sha256') != frozen._plan_hash(matrix)):
        raise ValueError('Invalid or modified cross-backbone manifest')
    canonical = build_matrix(matrix['data_home'], matrix['outdir'],
        matrix['cells'][0]['cfg']['device'], backbones=matrix['backbones'],
        graph_load_batch_size=matrix['graph_load_batch_size'])
    if matrix['plan_sha256'] != canonical['plan_sha256']:
        raise ValueError('Frozen scientific settings/source/runtime changed; use a NEW output directory')
    for cell in matrix['cells']:
        expected = configuration(matrix['data_home'], cell['cfg']['device'],
                                 cell['seed'], cell['key'], cell['backbone'], matrix['graph_load_batch_size'])
        if cell['cfg'] != expected:
            raise ValueError(f"{cell['method_key']}: noncanonical backbone/attack configuration")


def selected_cells(matrix, requested=None, pilot_clean=False):
    all_cells = {cell['method_key']: cell for cell in matrix['cells']}
    if pilot_clean and requested:
        raise ValueError('Use --pilot-clean or --cells, not both')
    if pilot_clean:
        return [cell for cell in matrix['cells'] if cell['key'] == 'clean']
    names = list(all_cells) if requested is None else list(requested)
    if not names or len(set(names)) != len(names) or set(names) - set(all_cells):
        raise ValueError('Unknown/duplicate cells; use metafiplusplus__s42__clean style IDs')
    # A requested attacked cell always includes its same-model, same-seed clean control.
    names = set(names)
    for name in tuple(names):
        names.update(all_cells[name]['dependencies'])
    return [cell for cell in matrix['cells'] if cell['method_key'] in names]


def current_inputs(cell):
    record = frozen._current_inputs(cell)
    record['profile'] = PROFILE
    return record


def completed(cell, sources):
    if not (Path(cell['artifact_dir']) / 'evaluation.json').is_file():
        return None
    run = frozen._read_completed(cell)
    inputs = current_inputs(cell)
    frozen._audit_run(cell, run, inputs)
    return evaluation_engine(sources).load_evaluation(dict(cell, verified_inputs=inputs),
                                                    cell['artifact_dir'])


def execute_cell(cell, sources):
    from train_backdoor import _config_fingerprint
    output = Path(cell['run_dir'])
    with frozen._cell_lock(output):
        current = current_inputs(cell)
        path = output / 'three_tables_inputs.json'
        if path.exists():
            if frozen._json(path) != current:
                raise ValueError('Training inputs changed; refuse to resume')
        elif any((output / name).exists() for name in ('checkpoint.pt', 'eval_cache.json')):
            raise ValueError('Checkpoint/cache has no training-time input audit')
        else:
            frozen._atomic_json(path, current)
        engine = training_engine(cell['cfg'])
        model, _ = engine.train(cell['cfg'], ckpt_dir=output)
        del model, engine
        if current_inputs(cell) != current:
            raise ValueError('Input identities/action changed during training/evaluation')
        run = frozen._read_completed(cell)
        audit = frozen._audit_run(cell, run, current)
        result = evaluation_engine(sources).evaluate_cell(
            dict(cell, verified_inputs=current), cell['artifact_dir'], cell['cfg']['device'])
        if run['fingerprint'] != _config_fingerprint(cell['cfg']):
            raise ValueError('Training engine changed the frozen configuration')
        frozen._atomic_json(output / 'scientific_audit.json', audit)
        print(f"[cross-backbones] completed {cell['method_key']}", flush=True)
        return result


def _stop_owned(running):
    for child, _, stream in running.values():
        try:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        finally:
            stream.close()


def schedule(matrix, manifest, devices, jobs_per_gpu, cells):
    pending, running, done = list(cells), {}, set()
    slots = [(device, i) for device in devices for i in range(jobs_per_gpu)]
    try:
        while pending or running:
            for slot in slots:
                if slot in running:
                    continue
                eligible = next((cell for cell in pending if set(cell['dependencies']) <= done), None)
                if eligible is None:
                    continue
                pending.remove(eligible)
                if completed(eligible, matrix['sources']) is not None:
                    done.add(eligible['method_key'])
                    print(f"[cross-backbones] verified complete: {eligible['method_key']}", flush=True)
                    continue
                eligible['cfg']['device'] = slot[0]
                frozen._atomic_json(manifest, matrix)
                folder = Path(eligible['run_dir'])
                folder.mkdir(parents=True, exist_ok=True)
                log = folder / 'console.log'
                stream = log.open('a', encoding='utf-8')
                env = os.environ.copy()
                env.setdefault('PYTHONUTF8', '1')
                env.setdefault('OMP_NUM_THREADS', '4')
                env.setdefault('MKL_NUM_THREADS', '4')
                try:
                    child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                        '--outdir', matrix['outdir'], '--_cell', eligible['method_key'],
                        '--_manifest', str(manifest)],
                        cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
                except BaseException:
                    stream.close()
                    raise
                running[slot] = child, eligible, stream
                print(f"[cross-backbones] {eligible['method_key']} -> {slot[0]}, "
                      f"slot={slot[1]+1}, PID={child.pid}\n       log: {log}", flush=True)
            for slot, (child, cell, stream) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                del running[slot]
                if code or completed(cell, matrix['sources']) is None:
                    raise RuntimeError(f"{cell['method_key']} failed (exit {code}); "
                                       f"inspect {cell['run_dir']}/console.log")
                done.add(cell['method_key'])
                print(f"[cross-backbones] finished {cell['method_key']}", flush=True)
            if pending and not running and not any(set(c['dependencies']) <= done for c in pending):
                raise RuntimeError('Missing clean-control dependency; refusing incomplete attack matrix')
            if running:
                time.sleep(1)
    except BaseException:
        _stop_owned(running)
        raise


def inspect_status(matrix, verify=False):
    missing, invalid = [], []
    for backbone in matrix['backbones']:
        for seed in SEEDS:
            cells = [c for c in matrix['cells'] if c['backbone'] == backbone and c['seed'] == seed]
            count = 0
            for cell in cells:
                if not (Path(cell['artifact_dir']) / 'evaluation.json').is_file():
                    missing.append(cell['method_key'])
                    continue
                try:
                    if verify and completed(cell, matrix['sources']) is None:
                        raise ValueError('Missing evaluation')
                    count += 1
                except (OSError, ValueError, KeyError) as exc:
                    invalid.append(cell['method_key'])
                    print(f"INVALID {cell['method_key']}: {exc}")
            print(f"{backbone} seed {seed}: {count}/13 {'VERIFIED' if verify else 'artifact files (not verified)'}")
    return missing, invalid


def export_reports(matrix):
    from a1_backbone_report import export_reports as export_three
    reports = {}
    for backbone in matrix['backbones']:
        records, identity = [], None
        for cell in [c for c in matrix['cells'] if c['backbone'] == backbone]:
            record = completed(cell, matrix['sources'])
            if record is None:
                raise ValueError(f"Missing authenticated evaluation: {cell['method_key']}")
            inputs = record['provenance']['binding']['ordered_inputs']
            if identity is not None and inputs != identity:
                raise ValueError('Backbone cells differ in ordered full split identities')
            identity = inputs
            records.append(record)
        reports[backbone] = export_three(records, Path(matrix['outdir']) / backbone,
            plan_sha256=matrix['plan_sha256'], provenance=dict(profile=PROFILE, backbone=backbone,
                recipe=matrix['recipes'][backbone], sources=matrix['sources'],
                runtime_versions=matrix['runtime_versions'], scratch=True,
                checkpoint_selection='fixed final epoch 49; never best TEST',
                limitations=matrix['limitations']))
    frozen._atomic_json(Path(matrix['outdir']) / 'cross_backbones.complete.json',
        dict(status='complete_full_declared_backbones_three_seeds', backbones=matrix['backbones'],
             plan_sha256=matrix['plan_sha256'], victims=len(matrix['cells']),
             reports={name: str(Path(matrix['outdir']) / name / 'three_main_tables.json')
                      for name in reports}))
    return reports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--backbones', nargs='+', choices=BACKBONES)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--jobs-per-gpu', type=int, choices=(1, 2), default=2)
    parser.add_argument('--num-workers', type=int, choices=(0,), default=0)
    parser.add_argument('--graph-load-batch-size', type=int, choices=(32, 64, 128, 256))
    parser.add_argument('--cells', nargs='+')
    parser.add_argument('--pilot-clean', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    for name in ('dry-run', 'status', 'check-only', 'export-only'):
        mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--_cell', help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (not args.devices or len(set(args.devices)) != len(args.devices)
            or any(d != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None for d in args.devices)):
        parser.error('Use distinct logical cuda:N IDs; slots are set by --jobs-per-gpu')
    if args.cells and args.pilot_clean:
        parser.error('--pilot-clean and --cells are mutually exclusive')
    return args


def load_or_build(args):
    manifest = args.outdir.resolve() / MANIFEST
    if manifest.exists():
        matrix = frozen._json(manifest)
        validate_manifest(matrix)
        if (Path(matrix['outdir']) != args.outdir.resolve()
                or (args.data_home and Path(matrix['data_home']) != args.data_home.resolve())
                or (args.backbones and set(args.backbones) != set(matrix['backbones']))
                or (args.graph_load_batch_size is not None
                    and args.graph_load_batch_size != matrix['graph_load_batch_size'])):
            raise ValueError('Requested roots/backbones differ from the saved plan')
    else:
        if args.status or args.check_only or args.export_only:
            raise FileNotFoundError(f'No existing manifest: {manifest}')
        if args.data_home is None:
            raise ValueError('--data-home is required for a new plan')
        if args.outdir.exists() and any(args.outdir.iterdir()):
            raise ValueError('Nonempty output without this manifest; choose a NEW directory')
        matrix = build_matrix(args.data_home, args.outdir, args.devices[0],
                              backbones=args.backbones or BACKBONES,
                              graph_load_batch_size=args.graph_load_batch_size or 32)
    return matrix, manifest


def main(argv=None):
    args = parse_args(argv)
    if args._cell:
        if args._manifest is None:
            raise ValueError('Worker requires its frozen manifest')
        matrix = frozen._json(args._manifest)
        validate_manifest(matrix)
        cell = next((c for c in matrix['cells'] if c['method_key'] == args._cell), None)
        if cell is None:
            raise ValueError('Worker cell is not in the frozen plan')
        execute_cell(cell, matrix['sources'])
        return 0
    matrix, manifest = load_or_build(args)
    chosen = selected_cells(matrix, args.cells, args.pilot_clean)
    if args.dry_run:
        print(f'[cross-backbones] DRY RUN: {len(matrix["cells"])} fresh scientific victims; '
              f'{len(chosen)} selected; {args.jobs_per_gpu} slot(s)/device; workers=0. '
              'NO output writes/data/GPU/training.')
        print(f'plan {matrix["plan_sha256"]}\n{manifest}')
        return 0
    if args.status or args.check_only:
        missing, invalid = inspect_status(matrix, verify=args.check_only)
        if args.check_only and (missing or invalid):
            print(f'INCOMPLETE: missing={len(missing)}, invalid={len(invalid)}')
            return 1
        return 0
    # Preflight before acquiring the lock allows a nonexistent output to remain
    # untouched on missing-data/CUDA failure. The lock is rechecked by workers.
    if not args.export_only:
        frozen._check_inputs(matrix, args.devices)
        for backbone in matrix['backbones']:
            # Real shape/forward checks are separate tests, not fake full-data validation.
            print(f'[cross-backbones] {backbone}: frozen recipe {matrix["recipes"][backbone]}', flush=True)
    with frozen._cell_lock(args.outdir):
        if manifest.exists():
            matrix, manifest = load_or_build(args)
        else:
            # Only run.lock was just created by this controller.
            frozen._atomic_json(manifest, matrix)
        chosen = selected_cells(matrix, args.cells, args.pilot_clean)
        for cell in chosen:
            if (Path(cell['run_dir']) / 'run.lock').exists():
                with frozen._cell_lock(cell['run_dir']):
                    pass
        if not args.export_only:
            launch = dict(controller_pid=os.getpid(), plan_sha256=matrix['plan_sha256'],
                devices=args.devices, jobs_per_gpu=args.jobs_per_gpu,
                cells=[c['method_key'] for c in chosen], time=datetime.now(timezone.utc).isoformat())
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            frozen._atomic_json(args.outdir / 'runtime_launches' / f'{stamp}_{os.getpid()}.json', launch)
            schedule(matrix, manifest, args.devices, args.jobs_per_gpu, chosen)
        missing = [c['method_key'] for c in matrix['cells']
                   if not (Path(c['artifact_dir']) / 'evaluation.json').is_file()]
        if missing:
            if args.cells or args.pilot_clean:
                print(f'[cross-backbones] selected cells finished; FULL TABLES INCOMPLETE: {len(missing)} cells remain')
                return 0
            raise ValueError(f'All declared cells required; missing {len(missing)}')
        export_reports(matrix)
        print(f'[cross-backbones] COMPLETE: {args.outdir / "cross_backbones.complete.json"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[cross-backbones] ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
