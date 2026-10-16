"""Resume an EXISTING frozen T1/T2/T3 manifest with multiple slots per GPU.

This sidecar deliberately does not modify the source-hashed scientific runner.
Stop its original controller and workers before a real resume. Dry-run/status
are read-only and may be used while the original controller is still running.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import time

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import run_a1_three_tables as frozen


def _positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--data-home', type=Path,
                        help='optional consistency check, never changes the saved plan')
    parser.add_argument('--num-workers', type=int,
                        help='optional consistency check, never changes the saved plan')
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--jobs-per-gpu', type=_positive_int, default=2)
    parser.add_argument('--cells', nargs='+')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--export-only', action='store_true')
    args = parser.parse_args(argv)
    if (len(set(args.devices)) != len(args.devices)
            or any(d != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None
                   for d in args.devices)):
        parser.error('--devices requires distinct logical cuda:N IDs (or cpu for tests)')
    return args


def _selected(matrix, requested):
    cells = {cell['method_key']: cell for cell in matrix['cells']}
    names = list(cells) if requested is None else list(requested)
    if not names or len(set(names)) != len(names) or set(names) - cells.keys():
        raise ValueError('Unknown, empty or duplicate --cells; use s42__clean style IDs')
    return [cell for cell in matrix['cells'] if cell['method_key'] in names]


def _load_existing(args):
    outdir = args.outdir.resolve()
    manifest = outdir / frozen.MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f'Existing manifest required; no new plan is created: {manifest}')
    matrix = frozen._json(manifest)
    frozen._validate_manifest(matrix)
    if Path(matrix['outdir']).resolve() != outdir:
        raise ValueError('Output path differs from the saved manifest; do not move/copy the run')
    if (args.data_home is not None
            and args.data_home.resolve() != Path(matrix['data_home']).resolve()):
        raise ValueError('--data-home differs from the saved manifest')
    if args.num_workers is not None and args.num_workers != matrix['num_workers']:
        raise ValueError('--num-workers differs from the saved manifest; preserve the worker policy')
    _selected(matrix, args.cells)
    return matrix, manifest


def _check_idle_cells(cells):
    """Reject surviving original workers; never create locks in historical runs."""
    seen = set()
    for cell in cells:
        folder = Path(cell['run_dir']).resolve()
        key = os.path.normcase(str(folder))
        if key in seen:
            continue
        seen.add(key)
        # Original workers take this lock before dataset/checkpoint access.
        # Absent locks mean an untouched cell, not permission to unlock a run.
        if (folder / 'run.lock').is_file():
            with frozen._cell_lock(folder):
                pass


def _completed(cell):
    """Skip only an input/poison/checkpoint/NPZ-bound verified evaluation."""
    if not (Path(cell['artifact_dir']) / 'evaluation.json').is_file():
        return False
    from a1_three_tables_eval import load_evaluation
    run = frozen._read_completed(cell)
    current = frozen._current_inputs(cell)
    frozen._audit_run(cell, run, current)
    evaluating = copy.deepcopy(cell)
    if cell['origin'] == 'reused_seed42' and cell['key'] == 'clean':
        evaluating['cfg'] = run['cfg']
    evaluating['verified_inputs'] = current
    return load_evaluation(evaluating, cell['artifact_dir']) is not None


def _stop_owned(running):
    """Only stop worker PIDs this controller started, never other GPU users."""
    for child, _, stream in running.values():
        try:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        except OSError as exc:
            # One already-exited/unreachable child must not prevent cleanup
            # of the other worker slots. Never search for unrelated PIDs.
            print(f'[parallel-resume] cleanup warning for owned PID {child.pid}: {exc}',
                  file=sys.stderr, flush=True)
        finally:
            try:
                stream.close()
            except OSError as exc:
                print(f'[parallel-resume] log-close warning: {exc}',
                      file=sys.stderr, flush=True)


def _schedule(matrix, manifest, devices, jobs_per_gpu, requested=None):
    if isinstance(jobs_per_gpu, bool) or not isinstance(jobs_per_gpu, int) or jobs_per_gpu < 1:
        raise ValueError('jobs_per_gpu must be a positive integer')
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('Distinct devices are required; use jobs_per_gpu for shared GPU slots')
    pending = _selected(matrix, requested)
    slots = [(device, index) for device in devices for index in range(jobs_per_gpu)]
    running = {}
    try:
        while pending or running:
            for slot in slots:
                if slot in running:
                    continue
                cell = None
                while pending:
                    candidate = pending.pop(0)
                    if _completed(candidate):
                        print(f"[parallel-resume] verified complete: {candidate['method_key']} (no training)",
                              flush=True)
                    else:
                        cell = candidate
                        break
                if cell is None:
                    continue
                device, slot_index = slot
                cell['cfg']['device'] = device
                # Devices are explicitly excluded from the existing plan hash.
                # Do not change workers, seed, recipe, origins or source hashes.
                frozen._atomic_json(manifest, matrix)
                folder = Path(cell['run_dir'])
                folder.mkdir(parents=True, exist_ok=True)
                log = folder / 'console.log'
                stream = log.open('a', encoding='utf-8')
                env = os.environ.copy()
                env.setdefault('PYTHONUTF8', '1')
                env.setdefault('OMP_NUM_THREADS', '4')
                env.setdefault('MKL_NUM_THREADS', '4')
                try:
                    child = subprocess.Popen([
                        sys.executable, '-u', str(Path(frozen.__file__).resolve()),
                        '--_cell', cell['method_key'], '--_manifest', str(manifest)],
                        cwd=frozen.HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
                except BaseException:
                    stream.close()
                    raise
                running[slot] = (child, cell, stream)
                print(f"[parallel-resume] {cell['method_key']} -> {device}, slot={slot_index + 1}, "
                      f"PID={child.pid}\n       log: {log}", flush=True)
            for slot, (child, cell, stream) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                del running[slot]
                if code:
                    raise RuntimeError(f"{cell['method_key']} failed (exit {code}); "
                                       f"read {cell['run_dir']}/console.log")
                if not _completed(cell):
                    raise RuntimeError(f"{cell['method_key']} exited without a verified evaluation")
                print(f"[parallel-resume] finished {cell['method_key']}", flush=True)
            if running:
                time.sleep(1)
    except BaseException:
        _stop_owned(running)
        raise


def _status(matrix):
    total = 0
    for seed in frozen.SEEDS:
        cells = [cell for cell in matrix['cells'] if cell['seed'] == seed]
        done = [cell['key'] for cell in cells
                if (Path(cell['artifact_dir']) / 'evaluation.json').is_file()]
        total += len(done)
        print(f"seed {seed}: {len(done)}/{len(cells)} evaluation artifacts; {', '.join(done)}")
    print(f'Total: {total}/39 evaluation artifacts')
    print('Counts alone are not verified results; --export-only performs the complete audit.')


def main(argv=None):
    args = parse_args(argv)
    matrix, manifest = _load_existing(args)
    if args.dry_run:
        cells = _selected(matrix, args.cells)
        print(f'[parallel-resume] DRY RUN: existing plan {matrix["plan_sha256"]}; '
              f'{len(cells)} selected cells; {len(args.devices) * args.jobs_per_gpu} worker slots; '
              f'{args.jobs_per_gpu} per device. NO writes/data/GPU/training.\n{manifest}')
        return 0
    if args.status:
        _status(matrix)
        return 0
    # The original controller holds the same OS lock for its whole run.
    # Never delete run.lock or bypass it to add live workers.
    with frozen._cell_lock(args.outdir):
        matrix, manifest = _load_existing(args)
        chosen = _selected(matrix, args.cells)
        _check_idle_cells(chosen)
        if not args.export_only:
            frozen._preflight(matrix, args.devices)
            now = datetime.now(timezone.utc)
            launch = dict(schema=1, kind='parallel_resume_runtime_only',
                created_utc=now.isoformat(), controller_pid=os.getpid(),
                plan_sha256=matrix['plan_sha256'], devices=args.devices,
                jobs_per_gpu=args.jobs_per_gpu, cells=[c['method_key'] for c in chosen],
                helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scientific_sources=matrix['sources'])
            record = args.outdir / 'runtime_launches' / (
                f'parallel_{now.strftime("%Y%m%dT%H%M%S%fZ")}_{os.getpid()}.json')
            frozen._atomic_json(record, launch)
            _schedule(matrix, manifest, args.devices, args.jobs_per_gpu, args.cells)
        missing = [cell['method_key'] for cell in matrix['cells']
                   if not (Path(cell['artifact_dir']) / 'evaluation.json').is_file()]
        if missing:
            if args.cells and not args.export_only:
                print(f'[parallel-resume] selected cells finished; complete tables await: {missing}', flush=True)
                return 0
            raise ValueError(f'All 39 cells are required; missing: {missing}')
        frozen._export(matrix)
        print(f'[parallel-resume] COMPLETE: {args.outdir / "three_main_tables.json"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[parallel-resume] ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
