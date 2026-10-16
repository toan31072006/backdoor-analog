"""Complete T1 from frozen MM-Fi checkpoints, without invoking training.

The original v3 comparison, A1 reference and clean run are read-only inputs.
New metrics and per-sample errors are written into a separate output folder.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re

# Match the repository's Windows NumPy/PyTorch runtime policy.
if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np
import yaml

from eval.target_pck import target_pck

HERE = Path(__file__).resolve().parent
BASELINES = ('badnets', 'wanet_source', 'ftrojan', 'fiba', 'sig_source', 'bpp_source')
PROFILE = 'mmfi_a1_target_pck_eval_v1'
COLUMNS = ('method_key', 'method', 't1_mpjpe_mm', 'target_pck_0.2_pct',
           'clean_mpjpe_mm', 'clean_pampjpe_mm')
CLEAN_KEYS = (
    'dataset_root', 'experiment_name', 'mmfi_protocol', 'mmfi_setting',
    'mmfi_random_ratio', 'mmfi_split_seed', 'model', 'pretrained',
    'data_parallel', 'num_person', 'n_ant', 'n_sub', 'n_pkt', 'seed',
    'victim_loss', 'batch_size', 'lr', 'optimizer', 'momentum', 'weight_decay',
    'lr_scheduler', 'epochs', 'victim_epochs', 'training_protocol',
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _read_run(folder):
    """Bind YAML, cache and metadata before loading any model weights."""
    from train_backdoor import _config_fingerprint, _RESULT_SCHEMA
    folder = Path(folder).resolve()
    cfg = yaml.safe_load((folder / 'config.resolved.yaml').read_text(encoding='utf-8'))
    recorded_fp = cfg.pop('config_fingerprint', None)
    cache = _json(folder / 'eval_cache.json')
    fp = _config_fingerprint(cfg)
    if (cache.get('cfg') != cfg or cache.get('cfg_fingerprint') != fp
            or recorded_fp not in (None, fp) or cache.get('trained_epochs') != 50
            or cache.get('result_schema') != _RESULT_SCHEMA or cfg.get('epochs') != 50):
        raise ValueError(f'{folder.name}: config/cache mismatch or incomplete run')
    if not (folder / 'checkpoint.pt').is_file():
        raise FileNotFoundError(folder / 'checkpoint.pt')
    return dict(folder=folder, cfg=cfg, res=cache['res'], fingerprint=fp,
                metadata_sha256={name: _sha(folder / name) for name in
                    ('config.resolved.yaml', 'eval_cache.json')})


def _validate_clean(clean_cfg, a1_cfg, clean_res):
    differences = [key for key in CLEAN_KEYS if clean_cfg.get(key) != a1_cfg.get(key)]
    if (differences or clean_cfg.get('rho') != 0.0
            or clean_cfg.get('clean_label_cover_ratio', 0) != 0
            or clean_cfg.get('wanet_cover_ratio', 0) != 0
            or clean_res.get('n_poison') != 0 or clean_res.get('n_cover', 0) != 0
            or clean_res.get('n_total') != 133056):
        raise ValueError(f'Clean reference recipe/poison count differs: {differences}')


def _load_weights(run, device):
    """Read a completed checkpoint; no optimizer, RNG resume or training loop."""
    import torch
    from models.factory import build_model
    from train_backdoor import _config_fingerprint
    checkpoint = run['folder'] / 'checkpoint.pt'
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (blob.get('epoch') != 49 or blob.get('cfg_fingerprint') != run['fingerprint']
            or not isinstance(blob.get('cfg'), dict)
            or _config_fingerprint(blob['cfg']) != run['fingerprint']):
        raise ValueError(f'{run["folder"].name}: checkpoint config/epoch mismatch')
    cfg = run['cfg']
    if cfg.get('data_parallel') or cfg.get('model') != 'hpeli':
        raise ValueError('This evaluator requires the original single-device HPELi victims')
    model = build_model(cfg['model'], num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=cfg.get('pretrained', False))
    model.load_state_dict(blob['model'], strict=True)
    return model.to(device).eval(), blob.get('trigger')


def _restore_trigger(cfg, state):
    from train_backdoor import build_trigger
    trigger = build_trigger(cfg)
    if getattr(trigger, 'requires_deferred_injection', False):
        raise ValueError('T1 contains only the frozen data-only trigger operators')
    if getattr(trigger, 'requires_stochastic_trigger_state', False):
        if state is None:
            raise ValueError('Checkpoint is missing its stochastic trigger state')
        trigger.load_state_dict(state)
    return trigger


def _dataset_record(cfg):
    """Reconstruct full ordered TRAIN/TEST identities and reject overlap."""
    from train_backdoor import _load_dataset
    from run_peak_confirmation import _pair_ids
    from run_method_drafts import _sha_json
    records, seen = {}, []
    test = None
    for name, split, count in (('train', 'training', 133056), ('eval', 'test', 33264)):
        data = _load_dataset(cfg, split)
        ids = _pair_ids(data)
        unique = {tuple(item) for item in ids}
        if len(data) != count or len(ids) != count or len(unique) != count:
            raise ValueError(f'{name}: wrong full split count or duplicate pairs')
        seen.append(unique)
        records[name] = dict(n=count, pair_ids_sha256=_sha_json(ids))
        if name == 'eval':
            test = data
    if seen[0].intersection(seen[1]):
        raise ValueError('TRAIN and TEST frame identities overlap')
    return test, records, _pair_ids(test)


def _assert_replay(res, metrics, tolerance_mm, *, attack=True):
    """A new PCK is accepted only if old scalar metrics reproduce."""
    pairs = [('clean_mpjpe_mm', float(res['clean_mpjpe']) * 1000),
             ('clean_pampjpe_mm', float(res['clean_pampjpe']) * 1000)]
    if attack:
        grid = res.get('dose_grid', [])
        if grid.count(1.0) != 1:
            raise ValueError('Cached dose grid must contain d=1 exactly once')
        index = grid.index(1.0)
        pairs.append(('t1_mpjpe_mm', float(res['tmpjpe'][index]) * 1000))
        if 'clean_to_target_tmpjpe' in res:
            pairs.append(('no_trigger_t1_mpjpe_mm',
                          float(res['clean_to_target_tmpjpe'][index]) * 1000))
    deltas = {}
    for key, expected in pairs:
        actual = metrics[key]
        if not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError(f'Nonfinite replay metric: {key}')
        delta = abs(actual - expected)
        if delta > tolerance_mm:
            raise ValueError(f'{key}: checkpoint replay differs by {delta:.6f} mm '
                             f'(new={actual:.6f}, old={expected:.6f}, '
                             f'tolerance={tolerance_mm:.6f}); no complete table exported')
        deltas[key] = delta
    return deltas


def _predict_pair(model, data, trigger, cfg, device):
    from torch.utils.data import DataLoader
    from attack.poison import PoisonedDataset, collate
    from train_backdoor import _predict
    def loader(mode):
        ds = PoisonedDataset(data, trigger, mode=mode, pivot=cfg['pivot'],
            axis=cfg['payload_axis'], dataset='mmfi', eps=cfg['eps'],
            theta_max_deg=cfg['theta_max_deg'], dose_mode=cfg['dose_mode'], fixed_dose=1.0)
        return DataLoader(ds, batch_size=cfg['batch_size'], collate_fn=collate,
                          num_workers=0, shuffle=False, pin_memory=False)
    print('[eval-only] clean pass', flush=True)
    clean, true, _, _ = _predict(model, loader('clean'), device)
    print('[eval-only] triggered pass d=1', flush=True)
    triggered, true_again, target, _ = _predict(model, loader('trigger@dose'), device)
    if not np.array_equal(true, true_again):
        raise ValueError('Clean/triggered frame order or true poses changed')
    return clean, triggered, true, target


def _measure(clean, triggered, true, target):
    from eval import metrics as M
    js = [2, 3]
    errors = np.linalg.norm(triggered[:, js] - target[:, js], axis=-1)
    baseline = np.linalg.norm(clean[:, js] - target[:, js], axis=-1)
    return {
        't1_mpjpe_mm': float(errors.mean()) * 1000,
        'target_pck_0.2_pct': 100 * target_pck(triggered, target, true),
        'clean_mpjpe_mm': float(M.mpjpe(clean, true).mean()) * 1000,
        'clean_pampjpe_mm': float(M.pa_mpjpe(clean, true).mean()) * 1000,
        'no_trigger_t1_mpjpe_mm': float(baseline.mean()) * 1000,
        'no_trigger_target_pck_0.2_pct': 100 * target_pck(clean, target, true),
    }, dict(
            target_error_mm=errors * 1000,
            no_trigger_target_error_mm=baseline * 1000,
            clean_scale_mm=np.linalg.norm(true[:, 5] - true[:, 12], axis=-1) * 1000)


def _write_report(outdir, rows, diagnostics, provenance, matrix, tolerance_mm, original):
    """Only called after all eight checkpoints pass their replay checks."""
    report = dict(status='t1_target_pck_complete_single_seed', profile=PROFILE,
        created_utc=datetime.now(timezone.utc).isoformat(), table='T1', seed=42,
        evaluation=dict(dataset='mmfi', protocol='protocol1-s1', samples=33264,
            dose=1.0, target_joints=[2, 3], target_pck_threshold=0.2,
            target_pck_scale='norm(clean_true_pose[5]-clean_true_pose[12])',
            target_pck_aggregation='fraction over samples and target joints; percent',
            replay_tolerance_mm=tolerance_mm, training_invoked=False),
        rows=rows, diagnostics=diagnostics, provenance=provenance,
        original_plan_sha256=matrix['plan_sha256'],
        original_comparison=original,
        evaluation_source_sha256={name: _sha(HERE / name) for name in
            ('eval_a1_target_pck.py', 'eval/target_pck.py')},
        limitations=[
            'Single seed 42; Proposed was selected during prior development.',
            'Native dose-adapted operators do not match realized L2/Linf or RF budgets.',
            'Clean reference has no attack rank; its A1-trigger diagnostic is separate.',
            'Clean historical run has no training-time pair-ID audit; its recipe, '
            'current ordered splits and replayed utility metrics were checked.',
            'Target-PCK is a target-joint hit fraction, not pose-level ASR.',
            'The selected comparison excludes Blended; no superiority over it is claimed.',
            'Frame identities are hashed; CSI and pose file contents are not fully hashed.',
        ])
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in COLUMNS} for row in rows)
    lines = ['# T1 comparison with Target-PCK', '',
        'MM-Fi protocol1-s1; seed 42; frozen 50-epoch checkpoints; rho=.1 (Clean: 0).',
        'Target-PCK@.2 uses target joints [2,3], attacker targets at d=1 and clean-pose scale.', '',
        '| Method | T-MPJPE@1 (mm) ↓ | Target-PCK@.2 (%) ↑ | Clean MPJPE (mm) ↓ | Clean PA-MPJPE (mm) ↓ |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        def fmt(key):
            value = row.get(key)
            return '—' if value is None else f'{value:.3f}'
        lines.append(f"| {row['method']} | {fmt('t1_mpjpe_mm')} | {fmt('target_pck_0.2_pct')} | "
                     f"{fmt('clean_mpjpe_mm')} | {fmt('clean_pampjpe_mm')} |")
    lines.extend(['', 'Clean reference is not ranked on attack metrics. Its A1-trigger '
                  'and every same-victim no-trigger control are in JSON diagnostics.',
                  'All replay checks passed. Per-sample NPZ files retain ordered frame '
                  'identities, target errors and clean-pose normalization scales.', '',
                  'Limitations:', *[f'- {item}' for item in report['limitations']]])
    _atomic_bytes(outdir / 'a1_target_pck_complete.csv', output.getvalue().encode('utf-8'))
    _atomic_bytes(outdir / 'a1_target_pck_complete.md', ('\n'.join(lines) + '\n').encode('utf-8'))
    # JSON is written last and serves as the completion marker.
    _atomic_bytes(outdir / 'a1_target_pck_complete.json',
                  (json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path, required=True)
    parser.add_argument('--comparison-run', type=Path)
    parser.add_argument('--clean-run', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true', help='Audit metadata and checkpoint paths only')
    args = parser.parse_args(argv)
    if args.device != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', args.device) is None:
        parser.error('--device must be cpu or a logical cuda:N device')

    import torch
    from a1_frozen_reference import load_a1_reference
    from run_a1_native_comparison import build_summary
    from run_mmfi_tables import _cell_lock
    from train_backdoor import _config_fingerprint

    root = args.data_home.resolve()
    comparison = (args.comparison_run or root / 'runs/mmfi_a1_native_comparison_s42_v3').resolve()
    matrix = _json(comparison / 'a1_native_comparison.resolved.json')
    existing = build_summary(matrix)  # original six-baseline/A1 audit, no writes
    if [cell['method_key'] for cell in matrix['cells']] != list(BASELINES):
        raise ValueError('T1 requires exactly the six approved baselines')
    a1_root = Path(matrix['metrics_contract']['a1_reference_path']).resolve()
    if not a1_root.is_dir():
        raise ValueError('A1 reference must be the original server directory with checkpoint.pt')
    reference = load_a1_reference(a1_root)
    a1 = _read_run(a1_root / 'full')
    clean = _read_run(args.clean_run or root / 'runs/mmfi_tables_s42_20261009_003044/clean')
    _validate_clean(clean['cfg'], a1['cfg'], clean['res'])
    runs = []
    for cell in matrix['cells']:
        run = _read_run(cell['ckpt_dir'])
        if run['fingerprint'] != _config_fingerprint(cell['cfg']):
            raise ValueError(f'{cell["method_key"]}: YAML differs from original comparison manifest')
        runs.append((cell['method_key'], cell['label'], run))
    runs.extend([('a1', 'Proposed A1 (reused)', a1), ('clean', 'Clean reference', clean)])
    outdir = (args.outdir or comparison / 'target_pck_eval').resolve()
    protected = [comparison, a1_root, *[r['folder'] for _, _, r in runs]]
    if outdir in protected or any(outdir == p.parent for p in protected):
        raise ValueError('Use a separate output directory, not an original run directory')
    if args.dry_run:
        for key, _, run in runs:
            print(f'OK {key}: {run["folder"] / "checkpoint.pt"}')
        print(f'[eval-only] metadata/path audit passed; no data/GPU/checkpoint loading. Output: {outdir}')
        return 0
    if args.device.startswith('cuda:'):
        torch.cuda.get_device_properties(torch.device(args.device))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tolerance_mm = 0.05
    with _cell_lock(outdir):
        # A completed export is immutable; choose a new folder to rerun it.
        if (outdir / 'a1_target_pck_complete.json').exists():
            raise FileExistsError('Completed evaluation exists; inspect it or use a NEW --outdir')
        print('[eval-only] auditing ordered full TRAIN/TEST frame identities', flush=True)
        data, records, pair_ids = _dataset_record(a1['cfg'])
        for name in ('train', 'eval'):
            if records[name] != reference['audit'][name]:
                raise ValueError(f'{name}: current ordered inputs differ from frozen A1')
        # Clean shares the checked split recipe/root, but its old run has no pair audit.
        if hasattr(data, 'enable_evaluation_cache'):
            data.enable_evaluation_cache()
        rows, diagnostics, provenance = [], [], []
        for key, label, run in runs:
            print(f'[eval-only] {key}: loading frozen epoch-49 checkpoint', flush=True)
            paths = [run['folder'] / name for name in
                     ('checkpoint.pt', 'config.resolved.yaml', 'eval_cache.json')]
            hashes = {p.name: _sha(p) for p in paths}
            if any(hashes[name] != digest for name, digest in run['metadata_sha256'].items()):
                raise ValueError(f'{key}: metadata changed since preflight')
            model, trigger_state = _load_weights(run, args.device)
            cfg = a1['cfg'] if key == 'clean' else run['cfg']
            trigger = _restore_trigger(cfg, None if key == 'clean' else trigger_state)
            clean_pred, triggered, true, target = _predict_pair(model, data, trigger, cfg, args.device)
            values, samples = _measure(clean_pred, triggered, true, target)
            deltas = _assert_replay(run['res'], values, tolerance_mm, attack=key != 'clean')
            if any(_sha(p) != hashes[p.name] for p in paths):
                raise ValueError(f'{key}: original artifact changed while evaluating')
            row = dict(method_key=key, method=label, **{k: values[k] for k in COLUMNS[2:]})
            if key == 'clean':
                row.update(t1_mpjpe_mm=None, **{'target_pck_0.2_pct': None})
            rows.append(row)
            diagnostics.append(dict(method_key=key, trigger=cfg['trigger'], **values,
                                    replay_abs_delta_mm=deltas))
            sample_path = outdir / f'per_sample_{key}.npz'
            buffer = io.BytesIO()
            np.savez_compressed(buffer, pair_ids=np.asarray(pair_ids), **samples)
            _atomic_bytes(sample_path, buffer.getvalue())
            provenance.append(dict(method_key=key, original_folder=str(run['folder']),
                cfg=run['cfg'], cfg_fingerprint=run['fingerprint'], artifact_sha256=hashes,
                per_sample_file=sample_path.name, per_sample_sha256=_sha(sample_path),
                ordered_inputs=records, checkpoint_epoch=49))
            print(f'[eval-only] {key}: replay passed; Target-PCK={values["target_pck_0.2_pct"]:.3f}%', flush=True)
            del model, trigger, clean_pred, triggered, true, target
        report = _write_report(outdir, rows, diagnostics, provenance, matrix, tolerance_mm, existing)
        print(f'[{report["status"]}] {outdir / "a1_target_pck_complete.md"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
