"""Read-only comparison of completed PiW v2 and the LR-only control.

Both runners replay their saved artifacts on CPU before comparison. This tool
does not train, evaluate a model, use CUDA, export files or alter either run.
"""
from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

V2_PROFILE = 'piw_hpeli_body_axis_s42_v2'
CONTROL_PROFILE = 'piw_hpeli_body_axis_s42_lr001_control_v3'
ADMIN_CONFIG_KEYS = frozenset(('profile', 'status', 'outdir'))
MATCHED_CONTEXT_KEYS = ('inputs', 'poison', 'trigger', 'test_truth_sha256', 'runtime')


def validate_pair(v2, v2_result, control, control_result):
    """Reject comparisons with any additional scientific configuration change."""
    if v2.get('profile') != V2_PROFILE or control.get('profile') != CONTROL_PROFILE:
        raise ValueError('Expected the frozen PiW v2 and LR-control profiles')
    left, right = v2['cfg'], control['cfg']
    if type(left.get('lr')) not in (int, float) or left['lr'] != .01:
        raise ValueError('Reference LR must be exactly .01')
    if type(right.get('lr')) not in (int, float) or right['lr'] != .001:
        raise ValueError('Control LR must be exactly .001')
    excluded = ADMIN_CONFIG_KEYS | {'lr'}
    a = {k: v for k, v in left.items() if k not in excluded}
    b = {k: v for k, v in right.items() if k not in excluded}
    if a != b:
        changed = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
        raise ValueError(f'Additional scientific configuration differences: {changed}')
    if Path(left['outdir']).resolve() == Path(right['outdir']).resolve():
        raise ValueError('Reference and control output directories must be separate')
    for key in MATCHED_CONTEXT_KEYS:
        if key not in v2 or key not in control or v2[key] != control[key]:
            raise ValueError(f'Runs do not share the exact {key} binding')
    if v2_result is None or control_result is None:
        raise ValueError('Both runs must have verified complete final-epoch artifacts')
    if ('execution' not in v2_result or 'execution' not in control_result
            or v2_result['execution'] != control_result['execution']):
        raise ValueError('Execution backend/GPU identity differs')


def compare_prediction_binding(v2_outdir, control_outdir):
    """Check exact saved test targets/identities without inference or raw access."""
    import numpy as np
    with np.load(Path(v2_outdir) / 'predictions.npz', allow_pickle=False) as left, \
            np.load(Path(control_outdir) / 'predictions.npz', allow_pickle=False) as right:
        for key in ('truth', 'names', 'doses', 'targets'):
            a, b = left[key], right[key]
            if (a.dtype != b.dtype or a.shape != b.shape
                    or np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes()):
                raise ValueError(f'Saved {key} arrays differ; not a matched LR-only comparison')
        return len(left['names'])


def dose_one(result):
    rows = [row for row in result['a1_metrics']['dose_rows'] if row.get('dose') == 1.]
    if len(rows) != 1:
        raise ValueError('Expected exactly one dose1 row')
    return rows[0]


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Comparison metrics must be finite real numbers')
    return float(value)


def report_rows(v2_result, control_result):
    """Numeric deltas always mean control(.001) minus reference(.01)."""
    def main(result):
        clean, a1 = result['clean_metrics'], result['a1_metrics']['mainmetrics']
        return [clean['clean_mpjpe_mm'], clean['clean_pampjpe_mm'],
                a1['t1_mpjpe_mm'], a1['target_pck_pct'], a1['clean_mpjpe_mm'],
                a1['clean_pampjpe_mm'], a1['mean_dce_mm']]

    labels = ['Clean reference MPJPE (mm, lower)', 'Clean reference PA-MPJPE (mm, lower)',
              'A1 T-MPJPE at d1 (mm, lower)', 'A1 Target-PCK at d1 (%, higher)',
              'A1 clean MPJPE (mm, lower)', 'A1 clean PA-MPJPE (mm, lower)',
              'A1 mean positive-dose DCE (mm, lower)']
    rows = [(label, _number(a), _number(b), _number(b)-_number(a))
            for label, a, b in zip(labels, main(v2_result), main(control_result))]
    keys = [('A1 no-trigger target error (mm)', 'no_trigger_t_mpjpe_mm'),
            ('A1 no-trigger Target-PCK (%)', 'no_trigger_target_pck_pct'),
            ('Clean model + trigger target error (mm)', 'reference_triggered_t_mpjpe_mm'),
            ('Clean model + trigger Target-PCK (%)', 'reference_triggered_target_pck_pct'),
            ('A1 triggered-to-GT target-joint error (mm)', 'triggered_gt_mpjpe_mm'),
            ('A1 nontarget drift (mm)', 'nontarget_drift_mm'),
            ('A1 DCE at d1 (mm)', 'dce_mm'), ('Stationary DCE at d1 (mm)', 'still_dce_mm')]
    a, b = dose_one(v2_result), dose_one(control_result)
    controls = [(label, _number(a[key]), _number(b[key]), _number(b[key])-_number(a[key]))
                for label, key in keys]
    for label, fn in [
        ('Within-run target-error gain at d1 (mm, higher)',
         lambda r: r['no_trigger_t_mpjpe_mm']-r['t_mpjpe_mm']),
        ('Within-run Target-PCK gain at d1 (pp, higher)',
         lambda r: r['target_pck_pct']-r['no_trigger_target_pck_pct'])]:
        x, y = _number(fn(a)), _number(fn(b))
        controls.append((label, x, y, y-x))
    return rows, controls


def compare_completed(v2_outdir, control_outdir):
    """Use each unchanged runner's own saved-artifact verifier, never main()."""
    v2_runner = importlib.import_module('run_piw_hpeli_v2')
    control_runner = importlib.import_module('run_piw_hpeli_lr_control')
    v2 = v2_runner.load_manifest(Path(v2_outdir) / v2_runner.MANIFEST)
    control = control_runner.load_manifest(Path(control_outdir) / control_runner.MANIFEST)
    v2_result = v2_runner.verify_complete(v2)
    control_result = control_runner.verify_complete(control)
    validate_pair(v2, v2_result, control, control_result)
    count = compare_prediction_binding(v2_outdir, control_outdir)
    rows, controls = report_rows(v2_result, control_result)
    return dict(n_samples=count, seed=v2['cfg']['seed'], rows=rows, controls=controls)


def print_report(report):
    print('# PiW LR-only control: paired seed42 comparison\n')
    print(f"Verified matched saved targets/GT/IDs: {report['n_samples']} samples; seed {report['seed']}.")
    print('Delta = LR .001 minus LR .01; Target-PCK deltas are percentage points.\n')
    for title, rows in [('Main metrics', report['rows']), ('Dose1 controls', report['controls'])]:
        print(title + '\n')
        print('| Metric | v2 LR .01 | Control LR .001 | Delta |')
        print('| --- | ---: | ---: | ---: |')
        for label, a, b, delta in rows:
            print(f'| {label} | {a:.4f} | {b:.4f} | {delta:+.4f} |')
        print()
    print('One-seed within-pair evidence; no across-seed SD/CI or population-level causal guarantee.')
    print('TEST was viewed during development. This is not a new blind test or a full comparison table.')
    print('Read-only saved-artifact replay: no training, model inference, CUDA or output writes.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2-outdir', type=Path, required=True)
    parser.add_argument('--control-outdir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print_report(compare_completed(args.v2_outdir, args.control_outdir))
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
        parser.exit(1, f'PiW comparison refused: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
