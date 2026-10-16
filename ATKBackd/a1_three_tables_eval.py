"""Read-only, seed-generic checkpoint evaluation for the three-table suite.

The caller owns training/input/poison-plan audits. This module never invokes
training or modifies a checkpoint directory. Completed exports are reusable
only when their data, source, checkpoint and metadata bindings still agree.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from eval_a1_target_pck import (
    _assert_replay, _atomic_bytes, _dataset_record, _load_weights, _measure,
    _predict_pair, _read_run, _restore_trigger, _sha,
)

HERE = Path(__file__).resolve().parent
SCHEMA = 1
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COUNTS = {'train': 133056, 'eval': 33264}
T1_KEYS = frozenset(('clean', 'badnets', 'wanet_source', 'ftrojan', 'fiba',
                     'sig_source', 'bpp_source', 'a1'))
ALL_KEYS = T1_KEYS | frozenset(('a2', 'a3', 'a4', 'rho0p2', 'rho0p4'))
INPUT_FILES = ('three_tables_inputs.json', 'ablation_inputs.json',
               'comparison_inputs.json')
SOURCE_FILES = (
    'a1_three_tables_eval.py', 'eval_a1_target_pck.py', 'eval/target_pck.py',
    'eval/metrics.py', 'train_backdoor.py', 'data_utils/feeder.py',
    'data_utils/draft_subset.py', 'attack/trigger.py', 'attack/method_drafts.py',
    'attack/payload.py', 'attack/poison.py', 'attack/traditional.py',
    'attack/wanet_source.py', 'attack/frequency_baselines.py',
    'attack/static_source_baselines.py', 'models/factory.py',
    'models/hpeli.py', 'models/sk_network.py',
)


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes():
    return {name: _sha(HERE / name) for name in SOURCE_FILES}


def _number(value, name):
    if (isinstance(value, (bool, str)) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f'{name} must be a finite nonnegative number')
    return float(value)


def _series(value, name):
    if not isinstance(value, list) or len(value) != len(GRID):
        raise ValueError(f'{name} must contain all six evaluation doses')
    return [_number(item, name) for item in value]


def _cache_metrics(res, key):
    """Validate full-grid metrics; never infer PCK from aggregate target error."""
    if res.get('dose_grid') != GRID:
        raise ValueError('Cached dose grid differs from the six-dose contract')
    target = _series(res.get('tmpjpe'), 'tmpjpe')
    baseline = _series(res.get('clean_to_target_tmpjpe'), 'clean_to_target_tmpjpe')
    if key == 'clean' and 'dose_control_error' not in res:
        dce = [None] * len(GRID)  # historical clean victim has no DCE evaluation
    else:
        dce = _series(res.get('dose_control_error'), 'dose_control_error')
        if dce[0] > 1e-8:
            raise ValueError('DCE at the identity dose must be zero')
    values = dict(t1_mpjpe_mm=target[-1] * 1000,
        target_pck_pct=None,
        clean_mpjpe_mm=1000 * _number(res.get('clean_mpjpe'), 'clean_mpjpe'),
        clean_pampjpe_mm=1000 * _number(res.get('clean_pampjpe'), 'clean_pampjpe'),
        mean_dce_mm=None if dce[0] is None else sum(dce[1:]) * 1000 / 5)
    rows = [dict(d=d, tmpjpe_mm=target[i] * 1000,
                 dce_mm=None if dce[i] is None else dce[i] * 1000,
                 no_trigger_tmpjpe_mm=baseline[i] * 1000)
            for i, d in enumerate(GRID)]
    return values, rows


def _checkpoint_metadata(run):
    """Inspect a trusted local checkpoint without constructing a victim model."""
    import torch
    from train_backdoor import _config_fingerprint
    blob = torch.load(run['folder'] / 'checkpoint.pt', map_location='cpu',
                      weights_only=False)
    if (not isinstance(blob, dict) or blob.get('epoch') != 49
            or not isinstance(blob.get('cfg'), dict)
            or blob.get('cfg_fingerprint') != run['fingerprint']
            or _config_fingerprint(blob['cfg']) != run['fingerprint']):
        raise ValueError('Checkpoint config/epoch mismatch; require frozen epoch 49')
    return {'epoch': 49, 'config_fingerprint': run['fingerprint']}


def _input_splits(record):
    result = {}
    for split, expected in COUNTS.items():
        value = record.get(split)
        if (not isinstance(value, dict) or value.get('n') != expected
                or not isinstance(value.get('pair_ids_sha256'), str)
                or len(value['pair_ids_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in value['pair_ids_sha256'])):
            raise ValueError(f'{split}: missing full-split ordered-identity audit')
        result[split] = dict(n=expected, pair_ids_sha256=value['pair_ids_sha256'])
    return result


def _audit_inputs(run, current):
    """Existing input audits must agree with the freshly verified dataset IDs."""
    found = []
    for name in INPUT_FILES:
        path = run['folder'] / name
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        if _input_splits(record) != current:
            raise ValueError(f'{name}: current ordered inputs differ from training audit')
        if record.get('config_fingerprint') not in (None, run['fingerprint']):
            raise ValueError(f'{name}: configuration fingerprint differs')
        if 'action_file_sha256' in record:
            if record['action_file_sha256'] != _sha(run['cfg']['action_npy']):
                raise ValueError(f'{name}: reference action file changed')
        found.append(name)
    return found


def _artifact_hashes(run):
    names = ['checkpoint.pt', 'config.resolved.yaml', 'eval_cache.json']
    names += [name for name in (*INPUT_FILES, 'poison_manifest.json', 'results.json')
              if (run['folder'] / name).exists()]
    if 'action_npy' in run['cfg']:
        values = {'reference_action': _sha(run['cfg']['action_npy'])}
    else:
        values = {}
    values.update({name: _sha(run['folder'] / name) for name in names})
    return values


def _read_completed(outdir, binding):
    path = outdir / 'evaluation.json'
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding='utf-8'))
    stored_digest = report.pop('record_sha256', None)
    if stored_digest != _digest(report):
        raise ValueError('Completed evaluation record checksum mismatch')
    report['record_sha256'] = stored_digest
    if (report.get('schema') != SCHEMA or report.get('status') != 'complete'
            or report.get('binding_sha256') != _digest(binding)
            or report.get('provenance', {}).get('binding') != binding):
        raise ValueError('Completed evaluation bindings changed; use a NEW output directory')
    samples = report['provenance'].get('per_sample_file')
    if samples is not None:
        path = outdir / samples
        if samples != 'per_sample.npz' or _sha(path) != report['provenance']['per_sample_sha256']:
            raise ValueError('Completed per-sample artifact checksum mismatch')
    return report


def _preflight(cell, outdir):
    """Validate local run/config/checkpoint files without data or GPU access."""
    from train_backdoor import _config_fingerprint
    key, seed = cell.get('key'), cell.get('seed')
    if (key not in ALL_KEYS or isinstance(seed, bool) or not isinstance(seed, int)
            or seed < 0 or cell.get('origin') not in ('fresh', 'reused_seed42')
            or (cell.get('origin') == 'reused_seed42' and seed != 42)):
        raise ValueError('Invalid three-table cell key/seed/origin')
    cfg = cell['cfg']
    if (cfg.get('seed') != seed or cfg.get('epochs') != 50
            or cfg.get('experiment_name') != 'mmfi' or cfg.get('model') != 'hpeli'
            or cfg.get('pivot') != 1 or cfg.get('target_joints', [2, 3]) != [2, 3]
            or cfg.get('theta_max_deg') != 40
            or cfg.get('payload_axis') != [0., 0., 1.]
            or cfg.get('dose_mode') != 'linear'):
        raise ValueError('Cell differs from the MM-Fi full A1 target/training contract')
    run = _read_run(cell['ckpt_dir'])
    if run['fingerprint'] != _config_fingerprint(cfg):
        raise ValueError('Cell configuration differs from completed run fingerprint')
    if ('eval_cache' in cell and Path(cell['eval_cache']).resolve()
            != run['folder'] / 'eval_cache.json'):
        raise ValueError('Cell evaluation-cache path differs from checkpoint run')
    outdir = Path(outdir).resolve()
    if (outdir == run['folder'] or outdir in run['folder'].parents
            or run['folder'] in outdir.parents):
        raise ValueError('Evaluation output must be separate from the checkpoint directory')
    values, rows = _cache_metrics(run['res'], key)
    artifacts = _artifact_hashes(run)
    for name, digest in run['metadata_sha256'].items():
        if artifacts[name] != digest:
            raise ValueError('Run metadata changed after loading')
    result_path = run['folder'] / 'results.json'
    if result_path.exists():
        old = json.loads(result_path.read_text(encoding='utf-8'))
        if (old.get('res') != run['res'] or not isinstance(old.get('cfg'), dict)
                or _config_fingerprint(old['cfg']) != run['fingerprint']):
            raise ValueError('results.json differs from cache/config')
    _checkpoint_metadata(run)
    return key, seed, cfg, run, outdir, values, rows, artifacts


def _binding(cell, run, records, artifacts):
    """Prepare immutable evaluation inputs after a separate data audit."""
    from train_backdoor import _config_fingerprint
    key, seed, cfg = cell['key'], cell['seed'], cell['cfg']
    trigger_cfg = cell.get('evaluation_trigger_cfg', run['cfg'])
    if key != 'clean' and 'evaluation_trigger_cfg' in cell:
        raise ValueError('Only the clean control may override its diagnostic trigger')
    if key == 'clean':
        shared = ('seed', 'dataset_root', 'experiment_name', 'model', 'pivot',
                  'theta_max_deg', 'payload_axis', 'dose_mode')
        if (any(trigger_cfg.get(name) != cfg.get(name) for name in shared)
                or trigger_cfg.get('target_joints', [2, 3]) != [2, 3]):
            raise ValueError('Clean diagnostic trigger must share the seed/data/pose target')
    sources = _source_hashes()
    binding = dict(key=key, seed=seed, method_key=cell.get('method_key'),
        origin=cell['origin'], config_fingerprint=run['fingerprint'],
        checkpoint_folder=str(run['folder']), artifacts_sha256=artifacts,
        ordered_inputs=records, evaluation_source_sha256=sources,
        trigger_config_fingerprint=_config_fingerprint(trigger_cfg),
        diagnostic_action_sha256=(_sha(trigger_cfg['action_npy'])
                                 if 'action_npy' in trigger_cfg else None),
        metric_contract=dict(grid=GRID, target_joints=[2, 3],
            target_pck_threshold=.2, clean_scale_joints=[5, 12],
            replay_tolerance_mm=.05, positive_doses_only=True))
    return binding, trigger_cfg, sources


def load_evaluation(cell, outdir):
    """Read an export, verifying file/source/NPZ bindings without data/GPU.

    Supply ``verified_inputs`` from the caller's current full-split audit to
    check dataset freshness too. Otherwise this read-only helper checks the
    stored ordered-ID binding against all surviving training-time audits, but
    does not claim to have re-enumerated the underlying dataset.
    """
    _, _, _, run, outdir, _, _, artifacts = _preflight(cell, outdir)
    path = outdir / 'evaluation.json'
    if not path.exists():
        return None
    if 'verified_inputs' in cell:
        records = _input_splits(cell['verified_inputs'])
    else:
        stored = json.loads(path.read_text(encoding='utf-8'))
        records = _input_splits(stored.get('provenance', {}).get('binding', {})
                                .get('ordered_inputs', {}))
    _audit_inputs(run, records)
    binding, _, _ = _binding(cell, run, records, artifacts)
    return _read_completed(outdir, binding)


def evaluate_cell(cell, outdir, device):
    """Evaluate one completed seed/key victim or read a strictly bound export.

    Optional ``verified_inputs`` supplies the caller's current TRAIN/TEST audit
    to avoid enumerating data on an already-completed evaluation. Clean can
    supply ``evaluation_trigger_cfg`` for a same-seed A1-trigger diagnostic;
    its main attack metrics remain null. Non-T1 arms validate checkpoint
    metadata and extract existing six-dose/DCE metrics without victim forward.
    """
    key, seed, cfg, run, outdir, values, rows, artifacts = _preflight(cell, outdir)

    data, pair_ids = None, None
    if 'verified_inputs' in cell:
        records = _input_splits(cell['verified_inputs'])
    else:
        data, records, pair_ids = _dataset_record(run['cfg'])
        records = _input_splits(records)
    audits = _audit_inputs(run, records)
    binding, trigger_cfg, sources = _binding(cell, run, records, artifacts)
    completed = load_evaluation(dict(cell, verified_inputs=records), outdir)
    if completed is not None:
        print(f'[three-tables-eval] seed={seed} {key}: verified cache hit', flush=True)
        return completed

    outdir.mkdir(parents=True, exist_ok=True)
    diagnostics = dict(forward_evaluated=key in T1_KEYS,
        cache_only=key not in T1_KEYS, training_invoked=False,
        missing_training_time_input_audit=not audits,
        input_audit_files=audits)
    sample_digest = None
    if key in T1_KEYS:
        print(f'[three-tables-eval] seed={seed} {key}: frozen epoch-49 evaluation', flush=True)
        if data is None:
            data, observed, pair_ids = _dataset_record(run['cfg'])
            if _input_splits(observed) != records:
                raise ValueError('Dataset enumeration differs from caller verified input audit')
        if hasattr(data, 'enable_evaluation_cache'):
            data.enable_evaluation_cache()
        model, state = _load_weights(run, device)
        try:
            trigger = _restore_trigger(trigger_cfg, None if key == 'clean' else state)
            predictions = _predict_pair(model, data, trigger, trigger_cfg, device)
            measured, samples = _measure(*predictions)
            if len(pair_ids) != len(predictions[0]):
                raise ValueError('Prediction/ordered-frame identity count differs')
            replay = _assert_replay(run['res'], measured, .05, attack=key != 'clean')
            if any(delta >= .05 for delta in replay.values()):
                raise ValueError('Replay difference must be strictly below 0.05 mm')
            diagnostics.update(trigger=trigger_cfg.get('trigger'),
                replay_abs_delta_mm=replay, measured=measured)
            values.update(t1_mpjpe_mm=measured['t1_mpjpe_mm'],
                target_pck_pct=measured['target_pck_0.2_pct'],
                clean_mpjpe_mm=measured['clean_mpjpe_mm'],
                clean_pampjpe_mm=measured['clean_pampjpe_mm'])
            buffer = io.BytesIO()
            np.savez_compressed(buffer, pair_ids=np.asarray(pair_ids, dtype=str), **samples)
            _atomic_bytes(outdir / 'per_sample.npz', buffer.getvalue())
            sample_digest = _sha(outdir / 'per_sample.npz')
        finally:
            del model
            if hasattr(data, 'clear_evaluation_cache'):
                data.clear_evaluation_cache()
    if key == 'clean':
        values.update(t1_mpjpe_mm=None, target_pck_pct=None, mean_dce_mm=None)
        rows = []  # do not rank a clean reference using its unrelated stored trigger
    if _artifact_hashes(run) != artifacts or _source_hashes() != sources:
        raise ValueError('Original artifacts or evaluation source changed during evaluation')
    report = dict(schema=SCHEMA, status='complete', key=key, seed=seed,
        method_key=cell.get('method_key'), metrics=values, dose_rows=rows,
        config_fingerprint=run['fingerprint'], diagnostics=diagnostics,
        binding_sha256=_digest(binding),
        created_utc=datetime.now(timezone.utc).isoformat(),
        provenance=dict(origin=cell['origin'], checkpoint_epoch=49,
            binding=binding, per_sample_file='per_sample.npz' if sample_digest else None,
            per_sample_sha256=sample_digest,
            result_origin='checkpoint_forward_replay' if key in T1_KEYS else 'validated_six_dose_cache',
            identity_audit_scope='ordered file/frame identities, not full CSI/pose content hashes'))
    report['record_sha256'] = _digest(report)
    _atomic_bytes(outdir / 'evaluation.json',
                  (json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    print(f'[three-tables-eval] seed={seed} {key}: complete', flush=True)
    return report
