"""Read the already-audited A1 artifact without rewriting or retraining it.

This is an intentionally seed-42-specific frozen reference, not a permissive
historical-cache importer. Byte hashes pin the six original JSON/YAML records
from the user's audited ZIP. Directory and ZIP readers never extract files or
load pickle/checkpoint contents. New comparison runs keep their own provenance.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zipfile

import yaml

PROFILE = 'mmfi_proposed_factorial_ablation_v1'
PLAN = '3f4799c05797c760ea475329e51d6f7364e68abf3ac5144034e49a509919ff71'
RECORD_SHA256 = {
    'proposed_ablation.resolved.json': '7d52900f0d665bb9551d1ee2845f0a318a7b9df80181375cf5d2f777dee05072',
    'full/eval_cache.json': 'e0c56752171e35230c86a5276f5b61979e336629e390574828bd5bab9c377849',
    'full/results.json': 'f1f8a2e82713a29bf5f0c99b0ed888f92cbebc5092e3e2390de4d4ebfdf2ee02',
    'full/ablation_inputs.json': '2f8074b4f627c76bb4fed9ead1e9ec39816c7976b6527d4d442af8f42ab7a77a',
    'full/poison_manifest.json': 'a74492f2f3351c5916a1118614f4da2961e0d0a2dde8548be9369ceb19954238',
    'full/config.resolved.yaml': '57a4a584461cfd88eb8134fc8a30f8edede5adcae87e309b03e97328468e522b',
}
COUNTS = {'train': 133056, 'eval': 33264}
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COMMON_KEYS = (
    'experiment_name', 'mmfi_protocol', 'mmfi_setting', 'mmfi_random_ratio',
    'mmfi_split_seed', 'model', 'pretrained', 'data_parallel', 'num_person',
    'n_ant', 'n_sub', 'n_pkt', 'seed', 'victim_loss', 'batch_size', 'lr',
    'optimizer', 'momentum', 'weight_decay', 'lr_scheduler', 'epochs',
    'victim_epochs', 'rho', 'poison_select', 'dose_min', 'dose_max', 'dose_grid',
    'pivot', 'target_joints', 'theta_max_deg', 'payload_axis', 'dose_mode',
    'dose_coupling', 'training_protocol', 'threat_model', 'attacker_access',
    'measure_dose_control', 'eps',
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _records(path):
    path = Path(path).resolve()
    if path.is_dir():
        data = {}
        for name in RECORD_SHA256:
            item = path / name
            if item.stat().st_size > 16 * 1024 * 1024:
                raise ValueError('Unexpectedly large A1 reference record')
            data[name] = item.read_bytes()
    else:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            roots = [n[:-len('proposed_ablation.resolved.json')] for n in names
                     if n.endswith('/proposed_ablation.resolved.json')
                     or n == 'proposed_ablation.resolved.json']
            if len(roots) != 1:
                raise ValueError('A1 ZIP must contain exactly one ablation manifest')
            root = roots[0]
            data = {}
            for name in RECORD_SHA256:
                full = root + name
                if names.count(full) != 1 or archive.getinfo(full).file_size > 16 * 1024 * 1024:
                    raise ValueError('Missing, duplicate, or oversized A1 reference record')
                data[name] = archive.read(full)
    for name, expected in RECORD_SHA256.items():
        if _sha(data[name]) != expected:
            raise ValueError(f'A1 frozen reference changed: {name}; do not rewrite its metadata')
    return data


def load_a1_reference(path):
    """Validate original records and return a separate, immutable report row."""
    from run_mmfi_tables import _plan_fingerprint
    from train_backdoor import _config_fingerprint
    from mmfi_tables import metric_row, dose_rows

    records = _records(path)
    decoded = {n: json.loads(b) for n, b in records.items() if n.endswith('.json')}
    matrix = decoded['proposed_ablation.resolved.json']
    cache = decoded['full/eval_cache.json']
    result = decoded['full/results.json']
    inputs = decoded['full/ablation_inputs.json']
    poison = decoded['full/poison_manifest.json']
    if (matrix.get('confirmation_profile') != PROFILE or matrix.get('seed') != 42
            or matrix.get('plan_sha256') != PLAN or _plan_fingerprint(matrix) != PLAN):
        raise ValueError('Not the frozen, fully audited A1 ablation plan')
    cell = next(c for c in matrix['cells'] if c['method_key'] == 'full')
    cfg, res = cache['cfg'], cache['res']
    fingerprint = _config_fingerprint(cfg)
    yaml_cfg = yaml.safe_load(records['full/config.resolved.yaml'])
    yaml_cfg.pop('config_fingerprint', None)
    if (cache.get('trained_epochs') != 50 or cache.get('result_schema') != 10
            or cfg != cell['cfg'] or cfg != result.get('cfg') or cfg != yaml_cfg
            or res != result.get('res') or cache.get('cfg_fingerprint') != fingerprint
            or result.get('cfg_fingerprint') != fingerprint
            or inputs.get('config_fingerprint') != fingerprint
            or cfg.get('trigger') != 'md_multicarrier' or cfg.get('rho') != .1
            or cfg.get('confirmation_profile') != PROFILE
            or cfg.get('dose_coupling') != 'paired'
            or any(k.startswith('comparison_peak_') for k in cfg)):
        raise ValueError('A1 config/cache/result audit differs from the frozen reference')
    for split, n in COUNTS.items():
        if inputs.get(split, {}).get('n') != n:
            raise ValueError('A1 is not a complete official split')
    pairs = [[s['index'], s['dose']] for s in poison['samples']]
    digest = _sha(json.dumps(pairs, separators=(',', ':'), ensure_ascii=True).encode())
    if (len(pairs) != 13305 or poison.get('poison_plan_sha256') != digest
            or res.get('poison_plan_sha256') != digest
            or res.get('n_total') != COUNTS['train'] or res.get('n_poison') != 13305
            or res.get('n_cover', 0) != 0 or res.get('dose_grid') != GRID):
        raise ValueError('A1 poison/evaluation audit differs from the frozen reference')
    row = metric_row(cell, res)
    series, _ = dose_rows(cell, res)
    dce = res.get('dose_control_error', [])
    if (len(dce) != 6 or any(not math.isfinite(v) or v < 0 for v in dce)
            or [r['d'] for r in series] != GRID):
        raise ValueError('A1 six-dose/DCE record is incomplete')
    row.update(method_key='a1_reference', method='A1 Proposed (frozen ablation reference)',
        t1_mpjpe_mm=row.pop('tmpjpe_mm'), epochs=50, rho=.1,
        train_samples=COUNTS['train'], eval_samples=COUNTS['eval'], n_poison=13305,
        n_cover=0, mean_dose_control_error_mm=sum(dce[1:]) * 1000 / 5,
        mean_positive_dose_tmpjpe_mm=sum(r['triggered_tmpjpe_mm'] for r in series[1:]) / 5,
        cfg_fingerprint=fingerprint, result_origin='frozen_a1_reference',
        train_pair_ids_sha256=inputs['train']['pair_ids_sha256'],
        eval_pair_ids_sha256=inputs['eval']['pair_ids_sha256'], poison_plan_sha256=digest)
    series = [dict(r, method_key='a1_reference', method=row['method'],
                   dose_control_error_mm=1000 * dce[i]) for i, r in enumerate(series)]
    return dict(row=row, dose_response=series, cfg=cfg,
        audit={**{k: inputs[k] for k in ('train', 'eval', 'action_file_sha256')},
               'poison_plan_sha256': digest},
        provenance=dict(original_profile=PROFILE, plan_sha256=PLAN,
            cfg_fingerprint=fingerprint, record_sha256=dict(RECORD_SHA256),
            reference_policy='Previously trained A1; no new training, cache rewrite, or weight transfer'),
        source_sha256=matrix['sources']['source_sha256'])


def validate_a1_input_record(reference, record):
    """Fail on different ordered splits/action before costly victim training."""
    for name in ('train', 'eval', 'action_file_sha256'):
        if record.get(name) != reference['audit'][name]:
            raise ValueError(f'{name} identities differ from frozen A1')


def validate_a1_reference(reference, matrix, audit=None):
    """Check shared science early, and exact split/poison identity after training.

    Trigger-specific parameters, loader/runtime settings and native cover
    fractions deliberately differ; these differences must be reported. Neither
    equal epsilon nor this check means equal realized noise or minibatch RNG.
    """
    original = reference['cfg']
    for cell in matrix['cells']:
        cfg = cell['cfg']
        differences = [k for k in COMMON_KEYS if cfg.get(k) != original.get(k)]
        if differences:
            raise ValueError(f"{cell['method_key']}: scientific settings differ from A1: {differences}")
        if any(k.startswith('comparison_peak_') for k in cfg):
            raise ValueError('A1 native comparison must not project baseline peaks')
    if audit is not None:
        for cell in matrix['cells']:
            key = cell['method_key']
            current = audit['inputs'][key]
            validate_a1_input_record(reference, current)
            if audit['poison_plan_sha256'][key] != reference['audit']['poison_plan_sha256']:
                raise ValueError(f'{key}: poison indices/doses differ from frozen A1')
