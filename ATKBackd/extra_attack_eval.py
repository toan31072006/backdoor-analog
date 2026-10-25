"""Frozen-checkpoint evaluation for the exploratory TSBA/LF MM-Fi runs.

Both attacks use the same P1S1 target and dose grid as the existing T1 table.
This module keeps their results separate because TSBA learns a generator and
LF may require a separately fixed trigger.  No checkpoint is selected on TEST.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np


GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TARGET_JOINTS = (2, 3)
REPLAY_TOLERANCE_MM = 0.05
EXPECTED_TEST_COUNT = 33264


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + '.saving')
    if temporary.exists():
        raise ValueError(f'Incomplete temporary export exists: {temporary}')
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _replay_mm(actual: float, cached_metres: float, name: str) -> float:
    actual = float(actual)
    expected = 1000.0 * float(cached_metres)
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f'{name}: nonfinite checkpoint/cache replay')
    delta = abs(actual - expected)
    if delta > REPLAY_TOLERANCE_MM:
        raise ValueError(f'{name}: checkpoint replay differs from cached '
                         f'evaluation by {delta:.6f} mm '
                         f'(limit {REPLAY_TOLERANCE_MM:.6f} mm)')
    return delta


def measure_pair(clean: np.ndarray, triggered: np.ndarray,
                 true: np.ndarray, target: np.ndarray) -> tuple[dict, dict]:
    """Return paired metrics and per-sample errors in millimetres.

    PCK uses the clean GT distance between joints 5 and 12, and only joints
    2 and 3 are scored. DCE compares the predicted change with the desired
    change; unchanged predictions cannot get a zero DCE at positive doses.
    """
    from eval import metrics as M
    from eval.target_pck import target_pck

    poses = [np.asarray(value) for value in (clean, triggered, true, target)]
    if any(value.shape != poses[0].shape or value.ndim != 3
           or value.shape[1:] != (17, 3)
           or not np.issubdtype(value.dtype, np.number)
           or np.issubdtype(value.dtype, np.complexfloating)
           or not np.isfinite(value).all() for value in poses):
        raise ValueError('Paired poses must be finite, matching (N, 17, 3) arrays')
    clean, triggered, true, target = [value.astype(np.float64, copy=False)
                                      for value in poses]
    js = list(TARGET_JOINTS)
    target_error = np.linalg.norm(triggered[:, js] - target[:, js], axis=-1)
    no_trigger_error = np.linalg.norm(clean[:, js] - target[:, js], axis=-1)
    scale = np.linalg.norm(true[:, 5] - true[:, 12], axis=-1)
    if np.any(scale <= 0):
        raise ValueError('Clean GT scale between joints 5 and 12 must be positive')
    dce = M.dose_control_error(triggered, clean, target, true, pivot=1)
    measured = {
        't_mpjpe_mm': 1000.0 * float(target_error.mean()),
        'no_trigger_t_mpjpe_mm': 1000.0 * float(no_trigger_error.mean()),
        'target_pck_pct': 100.0 * target_pck(triggered, target, true),
        'no_trigger_target_pck_pct': 100.0 * target_pck(clean, target, true),
        'dce_mm': 1000.0 * float(dce.mean()),
    }
    arrays = {
        'target_error_mm': 1000.0 * target_error,
        'no_trigger_target_error_mm': 1000.0 * no_trigger_error,
        'dce_mm': 1000.0 * dce,
        'clean_scale_mm': 1000.0 * scale,
    }
    replayed_pck = 100.0 * float(np.mean(
        arrays['target_error_mm'] <= 0.2 * arrays['clean_scale_mm'][:, None]))
    if abs(replayed_pck - measured['target_pck_pct']) > 1e-8:
        raise ValueError('Target-PCK does not replay from paired sample arrays')
    return measured, arrays


def _contract(cfg: dict, cached: dict) -> None:
    required = dict(experiment_name='mmfi', mmfi_protocol='protocol1',
                    mmfi_setting='s1', mmfi_split_seed=0,
                    mmfi_random_ratio=0.8, seed=42, epochs=50,
                    pivot=1, theta_max_deg=40, dose_mode='linear',
                    payload_axis=[0.0, 0.0, 1.0],
                    num_person=1, n_ant=3, n_sub=114, n_pkt=10,
                    pretrained=False, data_parallel=False,
                    victim_loss='mpjpe', rho=0.1, poison_select='uniform',
                    dose_min=0.2, dose_max=1.0, batch_size=32)
    recipes = {
        'hpeli': {'optimizer': 'sgd', 'lr': 0.001, 'momentum': 0.9,
                  'weight_decay': 0.0},
        'metafiplusplus': {'optimizer': 'sgd', 'lr': 0.01, 'momentum': 0.9,
                           'weight_decay': 0.0},
        'wiflow': {'optimizer': 'adamw', 'lr': 0.0001, 'weight_decay': 0.0001,
                   'backbone_effective_batch_size': 64,
                   'backbone_accumulation_steps': 2},
    }
    if cfg.get('model') not in recipes:
        raise ValueError('Extra attack evaluator supports HPE-Li/MetaFi++/WiFlow only')
    recipe = recipes[cfg['model']]
    required.update(recipe)
    if any(cfg.get(name) != expected for name, expected in required.items()):
        raise ValueError('Checkpoint config differs from frozen MM-Fi seed42 target')
    if cfg.get('dose_coupling', 'paired') != 'paired':
        raise ValueError('Only paired trigger/payload dose assignment is supported')
    if cfg.get('target_joints', [2, 3]) != [2, 3]:
        raise ValueError('Target joints differ from the P1S1 leg payload')
    if list(map(float, cfg.get('dose_grid', []))) != list(GRID):
        raise ValueError('Six-dose grid differs from frozen evaluation grid')
    if cfg.get('measure_dose_control') is not True:
        raise ValueError('Dose-control error is required for this comparison')
    if cached.get('dose_grid') != list(GRID):
        raise ValueError('Cached dose grid differs from frozen evaluation grid')
    for name in ('tmpjpe', 'clean_to_target_tmpjpe', 'dose_control_error'):
        value = cached.get(name)
        if not isinstance(value, list) or len(value) != len(GRID):
            raise ValueError(f'Cached {name} must contain all six doses')


def _loader(data, trigger, cfg: dict, mode: str, dose: float = 0.0):
    from torch.utils.data import DataLoader
    from attack.poison import PoisonedDataset, collate

    dataset = PoisonedDataset(data, trigger, mode=mode, pivot=cfg['pivot'],
                             axis=cfg['payload_axis'], dataset='mmfi',
                             eps=cfg['eps'], theta_max_deg=cfg['theta_max_deg'],
                             dose_mode=cfg['dose_mode'], fixed_dose=dose)
    return DataLoader(dataset, batch_size=cfg['batch_size'],
                      collate_fn=collate, num_workers=0, shuffle=False,
                      pin_memory=False)


def _read_existing(outdir: Path, binding: dict) -> dict | None:
    path = outdir / 'evaluation.json'
    if not path.exists():
        if outdir.exists() and any(outdir.iterdir()):
            raise ValueError('Incomplete evaluation directory; use a new output directory')
        return None
    report = json.loads(path.read_text(encoding='utf-8'))
    digest = report.pop('record_sha256', None)
    if (digest != _json_hash(report) or report.get('status') != 'complete'
            or report.get('binding') != binding
            or report.get('per_sample_sha256') != _sha256(outdir / 'per_sample.npz')):
        raise ValueError('Completed evaluation binding or per-sample file changed')
    report['record_sha256'] = digest
    return report


def _build_evaluation_model(cfg: dict, model_builder):
    """Use the MM-Fi geometry encoded in the checkpoint for every backbone."""
    return model_builder(cfg['model'], num_keypoints=17, subcarrier_num=114,
                         dataset='mmfi', pretrained=False)


def evaluate_checkpoint(run_dir, out_dir, device, *, model_builder,
                        trigger_builder=None) -> dict:
    """Replay a completed final49 victim without training or TEST selection.

    ``model_builder`` takes ``hpeli``, ``metafiplusplus`` or ``wiflow``. An optional
    ``trigger_builder(cfg)`` supplies the LF-CSI trigger; TSBA uses the frozen
    factory and must restore its trained generator state from the checkpoint.
    """
    import torch
    from train_backdoor import (_config_fingerprint, _load_dataset, _predict,
                                _RESULT_SCHEMA, build_trigger)
    from eval import metrics as M
    from eval_a1_target_pck import _read_run
    from run_peak_confirmation import _pair_ids

    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if run_dir == out_dir or out_dir in run_dir.parents or run_dir in out_dir.parents:
        raise ValueError('Evaluation output must be separate from checkpoint directory')
    run = _read_run(run_dir)
    cfg, cache = run['cfg'], run['res']
    _contract(cfg, cache)
    if cfg.get('device') != str(device):
        raise ValueError('Evaluation device must match the frozen training config')
    for name, digest in run['metadata_sha256'].items():
        if _sha256(run_dir / name) != digest:
            raise ValueError(f'Training artifact changed during preflight: {name}')
    blob = torch.load(run_dir / 'checkpoint.pt', map_location='cpu',
                      weights_only=False)
    if (blob.get('epoch') != 49 or blob.get('cfg_fingerprint') != run['fingerprint']
            or _config_fingerprint(blob.get('cfg', {})) != run['fingerprint']):
        raise ValueError('Expected the complete final49 checkpoint/config')
    raw_cache = json.loads((run_dir / 'eval_cache.json').read_text(encoding='utf-8'))
    if (raw_cache.get('result_schema') != _RESULT_SCHEMA
            or raw_cache.get('trained_epochs') != 50):
        raise ValueError('Expected complete six-dose final49 evaluation cache')
    binding = dict(schema=1, run_dir=str(run_dir),
                   config_fingerprint=run['fingerprint'],
                   checkpoint_sha256=_sha256(run_dir / 'checkpoint.pt'),
                   eval_cache_sha256=_sha256(run_dir / 'eval_cache.json'),
                   evaluator_sha256=_sha256(Path(__file__).resolve()),
                   metric_contract=dict(doses=list(GRID),
                       target_joints=list(TARGET_JOINTS),
                       pck_scale_gt_joints=[5, 12], pck_threshold=0.2,
                       replay_tolerance_mm=REPLAY_TOLERANCE_MM))
    found = _read_existing(out_dir, binding)
    if found is not None:
        return found

    model = _build_evaluation_model(cfg, model_builder)
    model.load_state_dict(blob['model'], strict=True)
    model = model.to(device).eval()
    trigger = (trigger_builder(cfg) if trigger_builder is not None
               else build_trigger(cfg))
    state = blob.get('trigger')
    if getattr(trigger, 'requires_deferred_injection', False) and state is None:
        raise ValueError('Learned trigger state missing from final checkpoint')
    if state is not None:
        if not callable(getattr(trigger, 'load_state_dict', None)):
            raise ValueError('Checkpoint trigger state cannot be restored')
        trigger.load_state_dict(state, strict=True)
    if getattr(trigger, 'requires_deferred_injection', False):
        trigger = trigger.to(device).eval()

    data = _load_dataset(cfg, 'test')
    if len(data) != EXPECTED_TEST_COUNT:
        raise ValueError(f'Official P1S1 TEST count changed: {len(data)}')
    pair_ids = _pair_ids(data)
    if len(pair_ids) != EXPECTED_TEST_COUNT or len({tuple(x) for x in pair_ids}) != len(pair_ids):
        raise ValueError('Official TEST ordered identities contain duplicates')
    if hasattr(data, 'enable_evaluation_cache'):
        data.enable_evaluation_cache()
    try:
        print('[extra eval] clean full TEST pass', flush=True)
        clean, true, _, _ = _predict(model, _loader(data, trigger, cfg, 'clean'), device)
        clean_mpjpe_mm = 1000.0 * float(M.mpjpe(clean, true).mean())
        clean_pa_mm = 1000.0 * float(M.pa_mpjpe(clean, true).mean())
        replay = dict(clean_mpjpe_mm=_replay_mm(
            clean_mpjpe_mm, cache['clean_mpjpe'], 'clean MPJPE'),
            clean_pampjpe_mm=_replay_mm(
            clean_pa_mm, cache['clean_pampjpe'], 'clean PA-MPJPE'))
        rows, errors, baselines, dces = [], [], [], []
        scale_mm = None
        for index, dose in enumerate(GRID):
            print(f'[extra eval] triggered full TEST pass d={dose:g}', flush=True)
            triggered, true_again, target, observed_dose = _predict(
                model, _loader(data, trigger, cfg, 'trigger@dose', dose),
                device, trig=trigger, cfg=cfg)
            if (not np.array_equal(true, true_again)
                    or not np.allclose(observed_dose, dose, rtol=0, atol=1e-7)):
                raise ValueError('Clean/triggered TEST order or requested dose changed')
            metrics, sample = measure_pair(clean, triggered, true, target)
            replay[f'd{dose:g}'] = {
                't_mpjpe_mm': _replay_mm(metrics['t_mpjpe_mm'],
                    cache['tmpjpe'][index], f'T-MPJPE d={dose:g}'),
                'no_trigger_t_mpjpe_mm': _replay_mm(metrics['no_trigger_t_mpjpe_mm'],
                    cache['clean_to_target_tmpjpe'][index], f'no-trigger d={dose:g}'),
                'dce_mm': _replay_mm(metrics['dce_mm'],
                    cache['dose_control_error'][index], f'DCE d={dose:g}'),
            }
            if scale_mm is None:
                scale_mm = sample['clean_scale_mm']
            elif not np.array_equal(scale_mm, sample['clean_scale_mm']):
                raise ValueError('GT reference scale changed across dose passes')
            rows.append(dict(dose=dose, **metrics))
            errors.append(sample['target_error_mm'])
            baselines.append(sample['no_trigger_target_error_mm'])
            dces.append(sample['dce_mm'])
        if rows[0]['dce_mm'] > REPLAY_TOLERANCE_MM:
            raise ValueError('DCE at identity dose should be zero')
        metrics = dict(t1_mpjpe_mm=rows[-1]['t_mpjpe_mm'],
                       target_pck_pct=rows[-1]['target_pck_pct'],
                       clean_mpjpe_mm=clean_mpjpe_mm,
                       clean_pampjpe_mm=clean_pa_mm,
                       mean_dce_mm=float(np.mean([row['dce_mm'] for row in rows[1:]])))
        buffer = io.BytesIO()
        np.savez_compressed(buffer,
            pair_ids=np.asarray(pair_ids, dtype=str),
            doses=np.asarray(GRID, dtype=np.float64),
            target_error_mm=np.stack(errors),
            no_trigger_target_error_mm=np.stack(baselines),
            dce_mm=np.stack(dces), clean_scale_mm=scale_mm)
        if _sha256(run_dir / 'checkpoint.pt') != binding['checkpoint_sha256'] or \
                _sha256(run_dir / 'eval_cache.json') != binding['eval_cache_sha256']:
            raise ValueError('Training artifacts changed during checkpoint evaluation')
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(out_dir / 'per_sample.npz', buffer.getvalue())
        report = dict(schema=1, status='complete', method=cfg['trigger'],
                      backbone=cfg['model'], seed=42, dataset='mmfi',
                      protocol='protocol1-s1', test_samples=len(data),
                      checkpoint_epoch=49, training_invoked=False,
                      metrics=metrics, dose_rows=rows,
                      replay_abs_delta_mm=replay, binding=binding,
                      per_sample_sha256=_sha256(out_dir / 'per_sample.npz'),
                      created_utc=datetime.now(timezone.utc).isoformat())
        report['record_sha256'] = _json_hash(report)
        _atomic_write(out_dir / 'evaluation.json',
                      (json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8'))
        return report
    finally:
        if hasattr(data, 'clear_evaluation_cache'):
            data.clear_evaluation_cache()
