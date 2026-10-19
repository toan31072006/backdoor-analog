"""Runtime-only, one/two-slot launcher for the unchanged, frozen T5 workers.

Stop the previous controller AND its workers before changing launchers. This
sidecar never changes the source-hashed repair/evaluation code or scientific
manifest. Existing completed records and epoch checkpoints remain reusable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import run_a1_defenses as frozen


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-outdir', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--jobs-per-gpu', type=int, choices=(1, 2), default=2,
                        help='runtime slots on this one device; does not change the recipe')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--methods', nargs='+', default=list(frozen.METHODS))
    parser.add_argument('--defenses', nargs='+', default=list(frozen.DEFENSES))
    parser.add_argument('--recipe', type=Path,
                        help='same explicit recipe mechanism as the original T5 runner')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--export-only', action='store_true')
    args = parser.parse_args(argv)
    if args.device != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', args.device) is None:
        parser.error('--device must be one logical cuda:N device (or cpu for tests)')
    return args


def _load_plan(args):
    recipe = frozen._json(args.recipe) if args.recipe else None
    path = args.outdir / frozen.MANIFEST
    if recipe is None and path.is_file():
        recipe = frozen._json(path)['recipe']
    plan, source = frozen._load_plan(args.source_outdir, args.outdir, recipe)
    frozen._selected(plan, args.seeds, args.methods, args.defenses)
    return plan, source


def _check_idle_cells(plan):
    """Include unselected cells: an orphan must not race the shared seed bank."""
    for cell in plan['cells']:
        folder = Path(cell['folder'])
        if (folder / 'run.lock').is_file():
            with frozen._cell_lock(folder):
                pass


def _prepare_banks(plan, source, selected):
    """Write each common bank once BEFORE workers can race its atomic temp file."""
    checked = set()
    for cell in selected:
        original = frozen._source_cell(source, cell)
        if original['method_key'] in checked:
            continue
        _, _, current, _ = frozen._audit_source(original)
        data, bank = frozen._bank(plan, original, current)
        del data, bank
        gc.collect()  # do not retain full TRAIN datasets alongside the two children
        checked.add(original['method_key'])


def _verified_record(plan, source, cell):
    if not (Path(cell['folder']) / 'evaluation.json').is_file():
        return None
    # _collect verifies the original source, inputs, bank, NPZ, repair and binding.
    # The full scientific plan/hash remains unchanged; only the read selection differs.
    records = frozen._collect(dict(plan, cells=[cell]), source)
    if len(records) != 1 or records[0]['cell_id'] != cell['cell_id']:
        raise ValueError(f"Missing verified evaluation for {cell['cell_id']}")
    return records[0]


def _stop_owned(running):
    """Release only worker PIDs started by THIS controller; never scan/kill others."""
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
            print(f'[T5-parallel] owned PID {child.pid} cleanup warning: {exc}',
                  file=sys.stderr, flush=True)
        finally:
            try:
                stream.close()
            except OSError as exc:
                print(f'[T5-parallel] log-close warning: {exc}', file=sys.stderr, flush=True)


def _schedule(plan, source, selected, device, jobs_per_gpu, manifest):
    if type(jobs_per_gpu) is not int or jobs_per_gpu not in (1, 2):
        raise ValueError('jobs_per_gpu must be 1 or 2')
    if len({c['cell_id'] for c in selected}) != len(selected):
        raise ValueError('Duplicate selected defense cells')
    pending = list(selected)
    running = {}
    try:
        while pending or running:
            # Reap failures BEFORE filling new slots: never start more work after
            # a known worker failure, and never silently retry/alter the recipe.
            for slot, (child, cell, stream) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                del running[slot]
                if code:
                    raise RuntimeError(f"{cell['cell_id']} failed (exit {code}); "
                        f"see {cell['folder']}/console.log. No automatic retry.")
                if _verified_record(plan, source, cell) is None:
                    raise RuntimeError(f"{cell['cell_id']} exited without a verified evaluation")
                print(f"[T5-parallel] finished {cell['cell_id']}", flush=True)
            for slot in range(jobs_per_gpu):
                if slot in running:
                    continue
                cell = None
                while pending:
                    candidate = pending.pop(0)
                    if _verified_record(plan, source, candidate) is not None:
                        print(f"[T5-parallel] verified complete: {candidate['cell_id']} "
                              '(no repair/evaluation)', flush=True)
                    else:
                        cell = candidate
                        break
                if cell is None:
                    continue
                folder = Path(cell['folder'])
                folder.mkdir(parents=True, exist_ok=True)
                log = folder / 'console.log'
                stream = log.open('a', encoding='utf-8')
                env = os.environ.copy()
                env.setdefault('PYTHONUTF8', '1')
                if os.name == 'nt':
                    env.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
                command = [sys.executable, '-u', str(Path(frozen.__file__).resolve()),
                           '--_manifest', str(manifest), '--_cell', cell['cell_id'],
                           '--device', device]
                try:
                    child = subprocess.Popen(command, cwd=frozen.HERE, env=env,
                                             stdout=stream, stderr=subprocess.STDOUT)
                except BaseException:
                    stream.close()
                    raise
                running[slot] = (child, cell, stream)
                print(f"[T5-parallel] {cell['cell_id']} -> {device}, slot={slot + 1}, "
                      f"PID={child.pid}\n       log: {log}", flush=True)
            if running:
                time.sleep(1)
    except BaseException:
        _stop_owned(running)
        raise


def main(argv=None):
    args = parse_args(argv)
    plan, source = _load_plan(args)
    selected = frozen._selected(plan, args.seeds, args.methods, args.defenses)
    if args.dry_run:
        print(f'[T5-parallel] DRY RUN: {len(selected)} selected evaluations; '
              f'{args.jobs_per_gpu} slots on {args.device}; workers=0; '
              'same frozen recipe. NO writes/data/checkpoints/GPU/training.')
        print(f"[T5-parallel] plan {plan['plan_sha256']}; source {plan['source_plan_sha256']}")
        return 0
    if args.status:
        present = sum((Path(c['folder']) / 'evaluation.json').is_file() for c in selected)
        print(f'[T5-parallel] artifact presence: {present}/{len(selected)} selected; '
              'NOT a fresh scientific audit')
        return 0
    output = Path(plan['outdir'])
    with frozen._cell_lock(output):
        # Recheck after taking the SAME controller lock as the original runner.
        plan, source = _load_plan(args)
        selected = frozen._selected(plan, args.seeds, args.methods, args.defenses)
        _check_idle_cells(plan)
        manifest = output / frozen.MANIFEST
        if not manifest.exists():
            frozen._atomic_json(manifest, plan)
        if not args.export_only:
            _prepare_banks(plan, source, selected)
            # Audit all historical records before accepting any skip or spawning work.
            frozen._collect(plan, source)
            now = datetime.now(timezone.utc)
            frozen._atomic_json(output / 'runtime_launches' /
                f"parallel_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}.json",
                dict(schema=1, kind='t5_parallel_runtime_only', created_utc=now.isoformat(),
                     controller_pid=os.getpid(), plan_sha256=plan['plan_sha256'],
                     device=args.device, jobs_per_gpu=args.jobs_per_gpu,
                     cells=[c['cell_id'] for c in selected],
                     helper_sha256=frozen._sha(Path(__file__)),
                     scientific_sources=plan['sources']))
            _schedule(plan, source, selected, args.device, args.jobs_per_gpu, manifest)
        records = frozen._collect(plan, source)
        status = frozen.export_records(plan, records)
        print(f"[{status}] {output / 'table5_defenses.md'}", flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[T5-parallel] ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
