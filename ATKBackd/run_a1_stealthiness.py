"""T4: paired distortion and clean-trained detectors on frozen MM-Fi victims."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np

from eval_a1_target_pck import (
    BASELINES, _atomic_bytes, _dataset_record, _json, _load_weights,
    _read_run, _restore_trigger, _sha,
)

HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_a1_stealthiness_s42_v1'
COLUMNS = ('method_key', 'method', 'relative_l2', 'input_auroc', 'feature_auroc',
           'input_tpr_at_cal5fpr', 'feature_tpr_at_cal5fpr',
           'input_test_fpr', 'feature_test_fpr')


def _dependencies():
    try:
        import sklearn  # noqa: F401
        import joblib  # noqa: F401
        from eval import stealthiness
        return stealthiness
    except ImportError as exc:
        raise RuntimeError('T4 needs scikit-learn; install remote_linux/requirements_t4.txt '
                           'using the dose-backdoor Python environment') from exc


def _sources():
    names = ('run_a1_stealthiness.py', 'eval/stealthiness.py', 'eval_a1_target_pck.py',
             'eval/distortion.py', 'models/hpeli.py', 'models/sk_network.py')
    return {name: _sha(HERE / name) for name in names}


def _plan(data_home, comparison_run=None):
    from run_a1_native_comparison import build_summary
    from a1_frozen_reference import load_a1_reference
    from train_backdoor import _config_fingerprint
    comparison = Path(comparison_run or data_home / 'runs/mmfi_a1_native_comparison_s42_v3').resolve()
    matrix = _json(comparison / 'a1_native_comparison.resolved.json')
    original = build_summary(matrix)  # full original config/input/poison audit
    if [c['method_key'] for c in matrix['cells']] != list(BASELINES):
        raise ValueError('T4 requires exactly the six approved baselines plus A1')
    reference = load_a1_reference(matrix['metrics_contract']['a1_reference_path'])
    runs = []
    for cell in matrix['cells']:
        run = _read_run(cell['ckpt_dir'])
        if run['fingerprint'] != _config_fingerprint(cell['cfg']):
            raise ValueError(f'{cell["method_key"]}: manifest and checkpoint run differ')
        runs.append((cell['method_key'], cell['label'], run))
    a1 = _read_run(Path(matrix['metrics_contract']['a1_reference_path']) / 'full')
    runs.append(('a1', 'Proposed A1 (reused)', a1))
    return matrix, original, reference, runs


def _batches(data, indices, batch_size, trigger=None, cfg=None):
    """Read CSI only, retaining identical clean/triggered frame order."""
    for offset in range(0, len(indices), batch_size):
        ids = indices[offset:offset + batch_size]
        clean, hit, raw_clean, raw_hit = [], [], [], []
        for index in ids:
            raw = data.load_raw(data.items[int(index)]['csi'])
            clean.append(data.normalize(raw))
            if trigger is not None:
                perturbed = trigger.inject(raw, dose=1.0, eps=cfg['eps'])
                hit.append(data.normalize(perturbed))
                raw_clean.append(raw)
                raw_hit.append(perturbed)
        pair = None if trigger is None else (
            np.ascontiguousarray(np.stack(hit), dtype=np.float32),
            np.stack(raw_clean), np.stack(raw_hit))
        yield ids, np.ascontiguousarray(np.stack(clean), dtype=np.float32), pair


def _per_distortion(clean, hit):
    a, b = np.asarray(clean, dtype=np.float64), np.asarray(hit, dtype=np.float64)
    if (a.shape != b.shape or a.ndim != 4 or a.shape[1:] != (3, 114, 10)
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Paired CSI must have matching finite N,L,S,T shapes')
    delta = b - a
    axes = (1, 2, 3)
    return dict(signal_energy=np.sum(a * a, axis=axes),
                error_energy=np.sum(delta * delta, axis=axes),
                linf=np.max(np.abs(delta), axis=axes))


def _pool_distortion(samples):
    signal = float(samples['signal_energy'].sum())
    error = float(samples['error_energy'].sum())
    if signal <= 0:
        raise ValueError('Relative distortion requires positive clean CSI energy')
    count = len(samples['linf']) * 3 * 114 * 10
    return dict(relative_l2=float(np.sqrt(error / max(signal, 1e-300))),
        rmse=float(np.sqrt(error / count)), linf=float(samples['linf'].max()),
        snr_db=None if error == 0 else float(10 * np.log10(signal / error)),
        snr_is_infinite=error == 0, n_samples=len(samples['linf']),
        aggregation='pooled energies; maximum Linf over full TEST')


def _features(model, values, device):
    import torch
    with torch.no_grad():
        _, feature = model(torch.from_numpy(values).to(device))
    feature = feature.detach().cpu().numpy()
    if feature.shape != (len(values), 128) or not np.isfinite(feature).all():
        raise ValueError('Expected finite 128-dimensional frozen HPELi features')
    return feature


def _clean_bank(model, data, indices, batch_size, device, helpers):
    stats, features = [], []
    for _, clean, _ in _batches(data, indices, batch_size):
        stats.append(helpers.input_statistics(clean))
        features.append(_features(model, clean, device))
    return np.concatenate(stats), np.concatenate(features)


def _test_scores(model, data, trigger, cfg, device, input_detector, feature_detector, helpers):
    values = {name: [] for name in ('input_clean', 'input_trigger', 'feature_clean', 'feature_trigger')}
    model_dist, raw_dist = {k: [] for k in ('signal_energy', 'error_energy', 'linf')}, {
        k: [] for k in ('signal_energy', 'error_energy', 'linf')}
    indices = list(range(len(data)))
    for ids, clean, pair in _batches(data, indices, cfg['batch_size'], trigger, cfg):
        hit, raw, raw_hit = pair
        values['input_clean'].append(helpers.input_statistics(clean))
        values['input_trigger'].append(helpers.input_statistics(hit))
        values['feature_clean'].append(_features(model, clean, device))
        values['feature_trigger'].append(_features(model, hit, device))
        for distortion_values, dest in ((_per_distortion(clean, hit), model_dist),
                                        (_per_distortion(raw, raw_hit), raw_dist)):
            for key, value in distortion_values.items():
                dest[key].append(value)
        if int(ids[-1]) + 1 == len(data) or (int(ids[-1]) + 1) % (cfg['batch_size'] * 100) == 0:
            print(f'[T4] TEST {int(ids[-1]) + 1}/{len(data)}', flush=True)
    # Score the compact statistics/features in one call per detector/domain;
    # avoid traversing 200 isolation trees once for every 32-frame batch.
    scores = {key: (input_detector if key.startswith('input_') else feature_detector).score(
        np.concatenate(parts)) for key, parts in values.items()}
    return (scores,
            {k: np.concatenate(v) for k, v in model_dist.items()},
            {k: np.concatenate(v) for k, v in raw_dist.items()})


def _write_table(outdir, rows, details, provenance, protocol):
    import sklearn
    import torch
    report = dict(status='t4_stealthiness_complete_single_seed', profile=PROFILE,
        seed=42, created_utc=datetime.now(timezone.utc).isoformat(), table='T4',
        rows=rows, details=details, provenance=provenance, protocol=protocol,
        source_sha256=_sources(), versions=dict(numpy=np.__version__, sklearn=sklearn.__version__,
                                               torch=torch.__version__),
        limitations=[
            'Two fixed detectors on digital amplitude CSI; no RF/over-the-air stealthiness claim.',
            'AUROC is reported with fixed high-anomaly score direction; no TEST-driven score inversion.',
            'Feature scores use each frozen victim; defender has access to trusted clean TRAIN CSI and victim features.',
            'Fit/calibration are separate TRAIN sequences; the selected bank is a deterministic capped sample.',
            'FPR thresholds and feature transformations are fitted without TEST or trigger labels.',
            'Single seed 42; detector parameters were fixed before this TEST evaluation.',
            'Native method distortion and modified-input/cover counts differ.',
        ])
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(outdir / 'table4_stealthiness.csv', stream.getvalue().encode('utf-8'))
    lines = ['# T4 Stealthiness — seed 42', '',
        'Full MM-Fi TEST; d=1; rho=.1; original frozen victims. Relative L2 pools model-input energies.',
        'Detectors fit on trusted clean TRAIN; calibration uses disjoint TRAIN sequences. '
        'AUROC near .5 means this fixed detector separates poorly.', '',
        '| Method | Relative L2 ↓ | AUROC-input | AUROC-feature |',
        '| --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['method']} | {row['relative_l2']:.6f} | {row['input_auroc']:.4f} | {row['feature_auroc']:.4f} |")
    lines.extend(['', '| Method | TPR-input at cal-5%-FPR | Test FPR-input | TPR-feature at cal-5%-FPR | Test FPR-feature |',
        '| --- | ---: | ---: | ---: | ---: |'])
    for row in rows:
        lines.append(f"| {row['method']} | {100*row['input_tpr_at_cal5fpr']:.2f}% | {100*row['input_test_fpr']:.2f}% | "
                     f"{100*row['feature_tpr_at_cal5fpr']:.2f}% | {100*row['feature_test_fpr']:.2f}% |")
    lines.extend(['', 'Limitations:', *[f'- {item}' for item in report['limitations']]])
    _atomic_bytes(outdir / 'table4_stealthiness.md', ('\n'.join(lines) + '\n').encode('utf-8'))
    _atomic_bytes(outdir / 'table4_stealthiness.json',
                  (json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path, required=True)
    parser.add_argument('--comparison-run', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.device != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', args.device) is None:
        parser.error('--device must be cpu or a logical cuda:N device')
    import torch
    from run_mmfi_tables import _cell_lock
    from run_peak_confirmation import _pair_ids
    from train_backdoor import _load_dataset

    data_home = args.data_home.resolve()
    matrix, original, reference, runs = _plan(data_home, args.comparison_run)
    outdir = (args.outdir or data_home / 'runs' / PROFILE).resolve()
    protected = [r['folder'] for _, _, r in runs]
    if any(outdir == path or outdir in path.parents for path in protected):
        raise ValueError('Use a separate T4 output folder')
    if args.dry_run:
        for key, _, run in runs:
            print(f'[T4] reuse {key}: {run["folder"] / "checkpoint.pt"}')
        print(f'[T4] DRY RUN: 7 victims; no data/GPU/training. Output: {outdir}')
        return 0
    helpers = _dependencies()
    if args.device.startswith('cuda:'):
        torch.cuda.get_device_properties(torch.device(args.device))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    with _cell_lock(outdir):
        if (outdir / 'table4_stealthiness.json').exists():
            raise FileExistsError('Completed T4 exists; inspect it or choose a NEW --outdir')
        a1_cfg = runs[-1][2]['cfg']
        test, records, test_ids = _dataset_record(a1_cfg)
        if any(records[k] != reference['audit'][k] for k in ('train', 'eval')):
            raise ValueError('Current ordered data differs from T1/A1')
        train_data = _load_dataset(a1_cfg, 'training')
        bank = helpers.bank_split(train_data.items, fit_n=4096, cal_n=2048, seed=42)
        train_ids = _pair_ids(train_data)
        bank_record = dict(bank, fit_pair_ids=[train_ids[i] for i in bank['fit_indices']],
                           cal_pair_ids=[train_ids[i] for i in bank['cal_indices']])
        _atomic_bytes(outdir / 'trusted_clean_banks.json',
                      (json.dumps(bank_record, indent=2) + '\n').encode('utf-8'))
        if hasattr(test, 'enable_evaluation_cache'):
            test.enable_evaluation_cache()
        rows, details, provenance = [], [], []
        input_detector = input_cal = None
        for key, label, run in runs:
            print(f'[T4] {key}: loading original epoch-49 checkpoint', flush=True)
            paths = [run['folder'] / name for name in
                     ('checkpoint.pt', 'config.resolved.yaml', 'eval_cache.json')]
            hashes = {p.name: _sha(p) for p in paths}
            if any(hashes[k] != v for k, v in run['metadata_sha256'].items()):
                raise ValueError(f'{key}: original metadata changed since preflight')
            model, state = _load_weights(run, args.device)
            trigger = _restore_trigger(run['cfg'], state)
            print(f'[T4] {key}: clean TRAIN fit/calibration features', flush=True)
            fit_stats, fit_features = _clean_bank(model, train_data, bank['fit_indices'],
                run['cfg']['batch_size'], args.device, helpers)
            cal_stats, cal_features = _clean_bank(model, train_data, bank['cal_indices'],
                run['cfg']['batch_size'], args.device, helpers)
            if input_detector is None:
                input_detector = helpers.fit_input_detector(fit_stats, seed=42)
                input_cal = input_detector.score(cal_stats)
            feature_detector = helpers.fit_feature_detector(fit_features, seed=42, max_components=64)
            feature_cal = feature_detector.score(cal_features)
            scores, model_dist, raw_dist = _test_scores(model, test, trigger, run['cfg'],
                args.device, input_detector, feature_detector, helpers)
            input_metrics = helpers.detection_metrics(scores['input_clean'], scores['input_trigger'], input_cal, fpr=.05)
            feature_metrics = helpers.detection_metrics(scores['feature_clean'], scores['feature_trigger'], feature_cal, fpr=.05)
            distortion = _pool_distortion(model_dist)
            row = dict(method_key=key, method=label, relative_l2=distortion['relative_l2'],
                input_auroc=input_metrics['auroc'], feature_auroc=feature_metrics['auroc'],
                input_tpr_at_cal5fpr=input_metrics['tpr'], feature_tpr_at_cal5fpr=feature_metrics['tpr'],
                input_test_fpr=input_metrics['test_fpr'], feature_test_fpr=feature_metrics['test_fpr'])
            rows.append(row)
            details.append(dict(method_key=key, input_detector=input_metrics,
                feature_detector=feature_metrics, model_input_distortion=distortion,
                raw_processed_csi_distortion=_pool_distortion(raw_dist),
                input_fit_audit=input_detector.audit, feature_fit_audit=feature_detector.audit))
            import joblib
            detector_path = outdir / f'detectors_{key}.joblib'
            detector_buffer = io.BytesIO()
            joblib.dump(dict(input_detector=input_detector, feature_detector=feature_detector),
                        detector_buffer, compress=3)
            _atomic_bytes(detector_path, detector_buffer.getvalue())
            payload = io.BytesIO()
            np.savez_compressed(payload, pair_ids=np.asarray(test_ids), **scores,
                input_calibration_scores=input_cal, feature_calibration_scores=feature_cal,
                **{f'model_{k}': v for k, v in model_dist.items()},
                **{f'raw_{k}': v for k, v in raw_dist.items()})
            score_path = outdir / f'scores_{key}.npz'
            _atomic_bytes(score_path, payload.getvalue())
            if any(_sha(p) != hashes[p.name] for p in paths):
                raise ValueError(f'{key}: original checkpoint/cache changed during T4')
            provenance.append(dict(method_key=key, cfg=run['cfg'],
                original_folder=str(run['folder']), artifact_sha256=hashes,
                scores_file=score_path.name, scores_sha256=_sha(score_path),
                detectors_file=detector_path.name, detectors_sha256=_sha(detector_path),
                cfg_fingerprint=run['fingerprint']))
            print(f'[T4] {key}: L2={row["relative_l2"]:.6f}; '
                  f'AUROC input={row["input_auroc"]:.4f}, feature={row["feature_auroc"]:.4f}', flush=True)
            del model, trigger, fit_stats, fit_features, cal_stats, cal_features, feature_detector
        protocol = dict(dataset='mmfi', split='protocol1-s1', rho=.1, seed=42, dose=1.,
            test_samples=33264, victims_retrained=False, ordered_inputs=records,
            original_plan_sha256=matrix['plan_sha256'],
            fit_samples=len(bank['fit_indices']), calibration_samples=len(bank['cal_indices']),
            bank_seed=42, bank_file='trusted_clean_banks.json',
            bank_sha256=_sha(outdir / 'trusted_clean_banks.json'),
            feature_layer='HPELi forward second output: pooled SK2, 128D',
            threshold_fpr=.05, detector_recipe=helpers.DETECTOR_RECIPE,
            clean_bank_policy='trusted unmodified official TRAIN, group-disjoint fit/calibration sequences',
            input_domain='dataset.normalize(raw CSI), identical to victim input',
            original_comparison_limitations=original['limitations'])
        report = _write_table(outdir, rows, details, provenance, protocol)
        print(f'[{report["status"]}] {outdir / "table4_stealthiness.md"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
