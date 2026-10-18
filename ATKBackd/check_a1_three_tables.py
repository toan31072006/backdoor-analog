"""Read-only verification of completed seeds in an existing T1/T2/T3 plan.

This checks checkpoint/cache/input/poison/per-sample bindings, not log phrases
or artifact counts alone. It never trains, evaluates a model, exports, or
creates a manifest. Input identity auditing needs the original MM-Fi files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Importing this diagnostic must not create bytecode in the frozen code tree.
sys.dont_write_bytecode = True

import run_a1_three_tables as frozen
import run_a1_three_tables_parallel as parallel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, choices=frozen.SEEDS,
                        default=[42, 43])
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('--seeds must not contain duplicates')
    return args


def _load_existing(outdir):
    root = outdir.resolve()
    manifest = root / frozen.MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f'Existing manifest required: {manifest}')
    matrix = frozen._json(manifest)
    frozen._validate_manifest(matrix)
    if Path(matrix['outdir']).resolve() != root:
        raise ValueError('Output path differs from saved manifest; do not move/copy the run')
    return matrix


def main(argv=None):
    args = parse_args(argv)
    try:
        matrix = _load_existing(args.outdir)
        selected = {}
        for seed in args.seeds:
            cells = [cell for cell in matrix['cells'] if cell['seed'] == seed]
            if (len(cells) != len(frozen.KEYS)
                    or {cell['key'] for cell in cells} != set(frozen.KEYS)):
                raise ValueError(f'Seed {seed} does not contain the exact 13 planned cells')
            selected[seed] = cells
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'[seed-check] ERROR: {exc}', file=sys.stderr, flush=True)
        return 1

    print('[seed-check] READ ONLY: checkpoint/cache/input/poison/NPZ audit; '
          'no training, model evaluation, export or output writes.', flush=True)
    all_verified = True
    for seed, cells in selected.items():
        verified = missing = invalid = 0
        for cell in cells:
            name = cell['method_key']
            if not (Path(cell['artifact_dir']) / 'evaluation.json').is_file():
                missing += 1
                print(f'MISSING {name}: evaluation.json absent', flush=True)
                continue
            try:
                complete = parallel._completed(cell)
                if not complete:
                    raise ValueError('Evaluation artifact is incomplete or failed its bindings')
            except Exception as exc:
                invalid += 1
                reason = ' '.join(str(exc).split())
                print(f'INVALID {name}: {type(exc).__name__}: {reason}', flush=True)
            else:
                verified += 1
                print(f'VERIFIED {name}', flush=True)
        print(f'seed {seed}: VERIFIED {verified}/13; MISSING {missing}; '
              f'INVALID {invalid}', flush=True)
        all_verified = all_verified and verified == len(frozen.KEYS)
    if all_verified:
        print('ALL_SELECTED_SEEDS_VERIFIED', flush=True)
        return 0
    print('SELECTED_SEEDS_INCOMPLETE_OR_INVALID: inspect the named cells; '
          'nothing was retrained or rewritten.', flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
