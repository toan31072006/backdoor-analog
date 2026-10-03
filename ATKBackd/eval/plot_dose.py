"""Reproduce the manuscript's normalized dose-schedule shape figure.

Each input must be a raw per-run ``result.json`` written by
``run_experiments.py`` after the schedule-shape contract revision.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def summarize(result_paths):
    groups = defaultdict(list)
    for path in map(Path, result_paths):
        result = json.loads(path.read_text(encoding='utf-8'))
        shape = result.get('schedule_shape')
        if not isinstance(shape, dict):
            raise ValueError(
                f'{path} has no schedule_shape; rerun evaluation with the '
                'current metric schema')
        dataset = result.get('dataset', 'unknown')
        mode = result.get('dose_mode', shape.get('mode'))
        if not mode:
            raise ValueError(f'{path} does not identify dose_mode')
        groups[(dataset, mode)].append((path, result, shape))

    summary = []
    for (dataset, mode), runs in sorted(groups.items()):
        doses = np.asarray(runs[0][1]['dose_grid'], dtype=float)
        expected = np.asarray(
            runs[0][2]['expected_normalized'], dtype=float)
        measured = []
        mads = []
        seeds = []
        for path, result, shape in runs:
            run_doses = np.asarray(result['dose_grid'], dtype=float)
            run_expected = np.asarray(shape['expected_normalized'], dtype=float)
            if (run_doses.shape != doses.shape or
                    not np.allclose(run_doses, doses) or
                    not np.allclose(run_expected, expected)):
                raise ValueError(f'incompatible dose grid/schedule in {path}')
            measured.append(np.asarray(shape['measured_normalized'], dtype=float))
            mads.append(float(shape['mad']))
            seeds.append(result.get('seed'))
        measured = np.stack(measured)
        summary.append({
            'dataset': dataset,
            'dose_mode': mode,
            'n_seeds': len(runs),
            'seeds': seeds,
            'dose_grid': doses.tolist(),
            'expected_normalized': expected.tolist(),
            'measured_mean': measured.mean(axis=0).tolist(),
            'measured_std': measured.std(axis=0, ddof=0).tolist(),
            'mad_mean': float(np.mean(mads)),
            'mad_std': float(np.std(mads, ddof=0)),
        })
    return summary


def plot(summary, output):
    datasets = sorted({row['dataset'] for row in summary})
    fig, axes = plt.subplots(1, len(datasets), squeeze=False,
                             figsize=(6 * len(datasets), 4.5))
    colors = {'linear': '#0072B2', 'sqrt': '#D55E00', 'quad': '#009E73'}
    for ax, dataset in zip(axes[0], datasets):
        for row in [r for r in summary if r['dataset'] == dataset]:
            mode = row['dose_mode']
            dose = np.asarray(row['dose_grid'])
            mean = np.asarray(row['measured_mean'])
            std = np.asarray(row['measured_std'])
            expected = np.asarray(row['expected_normalized'])
            color = colors.get(mode)
            ax.plot(dose, expected, '--', color=color, alpha=0.75,
                    label=f'{mode}: prescribed')
            ax.plot(dose, mean, 'o-', color=color,
                    label=f'{mode}: measured (MAD={row["mad_mean"]:.3f})')
            ax.fill_between(dose, mean - std, mean + std,
                            color=color, alpha=0.15)
        ax.set_title(dataset)
        ax.set_xlabel('Trigger dose')
        ax.set_ylabel('Normalized displacement')
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='+', required=True,
                        help='per-run raw result.json files')
    parser.add_argument('--out', default='fig_dose.pdf')
    args = parser.parse_args()

    summary = summarize(args.results)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot(summary, output)
    summary_path = output.with_suffix('.json')
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'wrote {output} and {summary_path}')


if __name__ == '__main__':
    main()
