"""Read-only reuse of the six completed, peak-only MM-Fi reference victims.

No training, trigger construction, re-evaluation, rewriting, or row selection
occurs here. Historical training/dispatch hashes are retained, not passed off
as current code. Unchanged native adapters, payload, data, model and metric
sources are checked against the local checkout before reference reuse.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

from mmfi_tables import config_fingerprint, metric_row, dose_rows
from run_method_drafts import _file_sha, _sha_json, _verified_cache
from run_mmfi_tables import _plan_fingerprint

HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_extended_shared_peak_comparison_v1'
STATUS = 'SHARED_PEAK_BUDGET_CSI_COMPARISON'
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EXPECTED_COUNTS = {'train': 133056, 'eval': 33264}
REFERENCE_KEYS = ('badnets', 'blended', 'wanet_source', 'ftrojan', 'fiba', 'proposed')
MAIN_METRICS = ('t1_mpjpe_mm', 'clean_mpjpe_mm', 'clean_pampjpe_mm',
                *[f'clean_pck_{t:.1f}_pct' for t in (.5, .4, .3, .2, .1)])
PROTECTED_SOURCES = (
    'configs/mmfi/attack_bend.yaml', 'attack/traditional.py',
    'attack/wanet_source.py', 'attack/peak_budget.py',
    'attack/frequency_baselines.py', 'attack/method_drafts.py',
    'attack/payload.py', 'attack/skeleton_mmfi.py', 'data_utils/feeder.py',
    'eval/distortion.py', 'eval/metrics.py', 'models/factory.py',
    'models/hpeli.py', 'models/sk_network.py', 'mmfi_tables.py',
    'third_party/backdoorbench/patch.py', 'third_party/backdoorbench/blended.py',
    'third_party/backdoorbench/SOURCE_MANIFEST.json',
)


def _digest(value, name):
    if re.fullmatch(r'[0-9a-f]{64}', str(value)) is None:
        raise ValueError(f'{name}: missing SHA256')
    return value


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Missing/invalid frozen reference file {Path(path).name}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Frozen {Path(path).name} must contain an object')
    return value


def _native_sources(matrix):
    historical = matrix.get('sources', {}).get('source_sha256', {})
    if not isinstance(historical, dict) or not historical:
        raise ValueError('Frozen reference has no historical source provenance')
    for name, value in historical.items():
        _digest(value, f'Historical source {name}')
    current = {}
    for name in PROTECTED_SOURCES:
        current[name] = _file_sha(HERE / name)
        # The historical Linux checkout used LF. Accept checkout-only CRLF
        # conversion, but retain both byte digests explicitly in provenance.
        normalized = hashlib.sha256((HERE / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if historical.get(name) not in (current[name], normalized):
            raise ValueError(f'Frozen native/payload/data/model/metric source changed: {name}')
    changed = {}
    for name, value in historical.items():
        path = HERE / name
        if path.is_file() and _file_sha(path) != value:
            changed[name] = dict(historical_sha256=value, current_sha256=_file_sha(path))
    return current, changed


def _validate_manifest(matrix):
    from run_traditional_comparison import _methods, _method_config, build_matrix
    required = dict(schema=1, dataset='mmfi', seed=42, fresh_results_only=True,
                    traditional_comparison_profile=PROFILE, status=STATUS)
    if any(matrix.get(k) != v for k, v in required.items()):
        raise ValueError('Expected completed full peak-only extended MM-Fi reference plan')
    if matrix.get('DRAFT_ONLY') or matrix.get('draft_profile') or matrix.get('simple_full_profile'):
        raise ValueError('Frozen reference must not be a draft or new simple/dual plan')
    cells = matrix.get('cells', [])
    if [c.get('method_key') for c in cells] != list(REFERENCE_KEYS):
        raise ValueError('All six frozen rows are required in original order; no row selection')
    if matrix.get('plan_sha256') != _plan_fingerprint(matrix):
        raise ValueError('Frozen reference plan fingerprint differs from contents')
    first = cells[0]['cfg']
    # This canonical plan only validates numeric/source-native settings. It
    # neither loads data nor starts a victim and its source-code hashes are
    # deliberately not substituted for the recorded historical hashes.
    canonical = build_matrix(HERE / '_reference_data_identity_only',
        HERE / '_reference_output_identity_only', first['device'], first['num_workers'],
        distortion_samples=matrix.get('metrics_contract', {}).get('distortion_samples', 256),
        budget_mode='shared_peak', baseline_set='extended')
    contract = copy.deepcopy(canonical['metrics_contract'])
    contract['full_split_counts'] = copy.deepcopy(EXPECTED_COUNTS)
    if matrix.get('metrics_contract') != contract:
        raise ValueError('Frozen reference metric/full-data/peak-only contract changed')
    if (matrix['sources'].get('references') != canonical['sources']['references']
            or matrix['sources'].get('baseline_config') != 'configs/mmfi/attack_bend.yaml'):
        raise ValueError('Frozen native paper/source defaults changed')
    ignore = {'dataset_root', 'action_npy', 'device', 'num_workers', 'ckpt_every'}
    for cell, (key, label, trigger) in zip(cells, _methods('extended')):
        cfg = cell['cfg']
        if key in ('wanet_source', 'fiba') and type(cfg.get('num_workers')) is not int:
            raise ValueError(f'{key}: frozen native cover RNG requires integer num_workers=0')
        if key in ('wanet_source', 'fiba') and cfg['num_workers'] != 0:
            raise ValueError(f'{key}: frozen native cover RNG requires num_workers=0')
        expected = _method_config(HERE / '_reference_data_identity_only',
            cfg['device'], cfg['num_workers'], key, trigger, 'shared_peak', 'extended')
        if ({k: v for k, v in cfg.items() if k not in ignore}
                != {k: v for k, v in expected.items() if k not in ignore}):
            raise ValueError(f'{key}: frozen native adapter or victim/payload settings changed')
        if any(isinstance(cfg.get(k), bool) or type(cfg.get(k)) is not int
               for k in ('epochs', 'victim_epochs', 'seed', 'batch_size', 'pivot')):
            raise ValueError(f'{key}: frozen integer settings must be actual integers')
        if (cell.get('label') != label or cell.get('group') != 'continuous_dose_task'
                or cell.get('dependencies') != [] or cell.get('tables') != []):
            raise ValueError(f'{key}: frozen row semantics changed')


def _input_identity(record):
    for name in ('train', 'eval'):
        item = record.get(name, {})
        if item.get('n') != EXPECTED_COUNTS[name] or type(item.get('n')) is not int:
            raise ValueError(f'Frozen reference requires full {name} identity')
        _digest(item.get('pair_ids_sha256'), f'{name} ordered pairs')
    _digest(record.get('action_file_sha256'), 'Action content')
    return dict(train=record['train'], eval=record['eval'],
                action_file_sha256=record['action_file_sha256'])


def _relative_pair_id(value):
    if not isinstance(value, str):
        raise ValueError('Poison sample requires CSI/pose file identity')
    parts = value.replace('\\', '/').split('/')
    if 'Compress' in parts:
        if parts.count('Compress') != 1:
            raise ValueError('Ambiguous MM-Fi poison file identity')
        parts = parts[parts.index('Compress') + 1:]
    if (not parts or any(p in ('', '.', '..') for p in parts)
            or not re.fullmatch(r'E[0-9]+', parts[0])):
        raise ValueError('Poison identity must be inside the MM-Fi Compress dataset')
    return '/'.join(parts)


def _poison_identity(poison):
    samples = poison.get('samples')
    n_poison = math.floor(.1 * EXPECTED_COUNTS['train'])
    if (poison.get('schema') not in (4, 5) or not isinstance(samples, list)
            or len(samples) != n_poison):
        raise ValueError('Frozen reference requires its complete poison sample plan')
    plan, normalized = [], []
    for sample in samples:
        i, d, frame = sample.get('index'), sample.get('dose'), sample.get('frame_idx')
        if (type(i) is not int or not 0 <= i < EXPECTED_COUNTS['train']
                or isinstance(d, bool) or not isinstance(d, (int, float))
                or not math.isfinite(d) or not .2 <= d <= 1
                or sample.get('payload_dose', d) != d or type(frame) is not int or frame < 0):
            raise ValueError('Frozen reference has an invalid paired poison sample')
        plan.append([i, d])
        normalized.append([i, d, _relative_pair_id(sample.get('csi_id')),
                           _relative_pair_id(sample.get('pose_id')), frame])
    digest = _sha_json(plan)
    required = dict(n_total=EXPECTED_COUNTS['train'], n_poison=n_poison,
        rho_requested=.1, seed=42, selection='uniform', dose_min=.2, dose_max=1.0,
        poison_plan_sha256=digest, pivot=1, target_joints=[2, 3], theta_max_deg=40.,
        payload_axis=[0., 0., 1.], dose_mode='linear')
    if (any(poison.get(k) != v for k, v in required.items())
            or poison.get('dose_coupling', 'paired') != 'paired'
            or len({p[0] for p in plan}) != n_poison):
        raise ValueError('Frozen poison manifest differs from the seed42 paired-dose task')
    if poison.get('schema') == 5 and poison.get('coupling_sha256') != _sha_json(
            [[i, d, d] for i, d in plan]):
        raise ValueError('Frozen native-cover manifest has a different paired coupling digest')
    return digest, normalized


def _validate_cover(poison, key):
    ratio = .2 if key == 'wanet_source' else .1 if key == 'fiba' else 0.
    indices = poison.get('cover_indices', [])
    count = math.floor(ratio * EXPECTED_COUNTS['train'])
    if (count and poison.get('schema') != 5 or not isinstance(indices, list) or len(indices) != count
            or poison.get('n_cover', 0) != count or poison.get('cover_ratio', 0.) != ratio
            or any(type(i) is not int or not 0 <= i < EXPECTED_COUNTS['train'] for i in indices)
            or len(set(indices)) != count
            or set(indices).intersection(s['index'] for s in poison['samples'])):
        raise ValueError(f'{key}: frozen native cover allowance/identities changed')
    return count


def _same_numeric_rows(actual, expected, keys):
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for saved, current in zip(actual, expected):
        if not isinstance(saved, dict) or saved.get('method_key') != current['method_key']:
            return False
        for key in keys:
            value = saved.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not math.isclose(value, current[key], rel_tol=0., abs_tol=1e-9)):
                return False
    return True


def load_frozen_references(reference_run, data_home=None):
    """Audit an existing reference directory without changing any of its bytes.

    Return all six metric rows, all 36 dose rows and immutable provenance.
    ``data_home`` additionally checks current full ordered data IDs/action bytes;
    it never constructs a reference trigger or re-evaluates a victim.
    """
    root = Path(reference_run).resolve()
    manifest_path = root / 'traditional_comparison.resolved.json'
    matrix = _read_json(manifest_path)
    _validate_manifest(matrix)
    current_sources, changed_sources = _native_sources(matrix)
    files = {manifest_path.name: _file_sha(manifest_path)}
    rows, doses, records, configs, poisons, cache_sources = [], [], {}, {}, {}, []
    common, common_poison = None, None
    for original in matrix['cells']:
        cell = copy.deepcopy(original)
        key, cfg = cell['method_key'], cell['cfg']
        folder = root / key
        cell['ckpt_dir'], cell['eval_cache'] = str(folder), str(folder / 'eval_cache.json')
        for name in ('eval_cache.json', 'comparison_inputs.json', 'poison_manifest.json'):
            path = folder / name
            files[f'{key}/{name}'] = _file_sha(path)
        record = _read_json(folder / 'comparison_inputs.json')
        if (record.get('schema') != 1 or record.get('profile') != PROFILE
                or record.get('config_fingerprint') != config_fingerprint(cfg)):
            raise ValueError(f'{key}: frozen input passport/config differs')
        identity = _input_identity(record)
        if common is not None and common != identity:
            raise ValueError('Frozen baselines have different ordered splits/action content')
        common = identity
        if key == 'fiba':
            if record.get('fiba_reference_train_index') != 0:
                raise ValueError('FIBA frozen key must use first TRAIN sample')
            for name in ('fiba_fixed_key_sha256', 'fiba_reference_sha256'):
                _digest(record.get(name), name)
        poison = _read_json(folder / 'poison_manifest.json')
        if poison.get('config_fingerprint') != config_fingerprint(cfg):
            raise ValueError(f'{key}: frozen poison/config fingerprint differs')
        digest, poison_identity = _poison_identity(poison)
        covers = _validate_cover(poison, key)
        if common_poison is not None and common_poison != poison_identity:
            raise ValueError('Frozen baselines require identical poison IDs AND paired doses')
        common_poison = poison_identity
        res, source = _verified_cache(cell)
        if key in ('wanet_source', 'fiba') and (
                type(source['cfg'].get('num_workers')) is not int
                or source['cfg']['num_workers'] != 0):
            raise ValueError(f'{key}: frozen cached native cover RNG requires num_workers=0')
        required_res = dict(training_contract='ordinary_erm', attacker_access='data_only',
            dose_coupling='paired', dose_grid=GRID, n_total=EXPECTED_COUNTS['train'],
            n_poison=len(poison['samples']), n_cover=covers, poison_plan_sha256=digest)
        if any(res.get(k) != v for k, v in required_res.items()):
            raise ValueError(f'{key}: frozen completed result disagrees with protocol audit')
        metrics, series = metric_row(cell, res), dose_rows(cell, res)[0]
        if [r['d'] for r in series] != GRID:
            raise ValueError(f'{key}: frozen reference needs all six exact evaluated doses')
        row = dict(method_key=key, method=cell['label'], seed=42, epochs=50, rho=.1,
                   train_samples=EXPECTED_COUNTS['train'], eval_samples=EXPECTED_COUNTS['eval'],
                   n_poison=len(poison['samples']), n_cover=covers, reference_role='frozen_historical',
                   t1_mpjpe_mm=metrics['tmpjpe_mm'], cfg_fingerprint=source['cfg_fingerprint'])
        row.update({k: metrics[k] for k in MAIN_METRICS if k != 't1_mpjpe_mm'})
        rows.append(row)
        doses.extend(dict(reference_role='frozen_historical', **r) for r in series)
        records[key], configs[key], poisons[key] = record, cfg, poison
        cache_sources.append(source)
        result_path = folder / 'results.json'
        if result_path.exists():
            saved = _read_json(result_path)
            if (saved.get('method_key') != key or saved.get('cfg_fingerprint') != source['cfg_fingerprint']
                    or saved.get('cfg') != cfg or saved.get('res') != res):
                raise ValueError(f'{key}: frozen results.json/cache/manifest disagree')
            files[f'{key}/results.json'] = _file_sha(result_path)
        yaml_path = folder / 'config.resolved.yaml'
        if yaml_path.exists():
            import yaml
            if yaml.safe_load(yaml_path.read_text(encoding='utf-8')) != cfg:
                raise ValueError(f'{key}: frozen YAML/manifest configuration disagree')
            files[f'{key}/config.resolved.yaml'] = _file_sha(yaml_path)
    for name, expected, keys in (
            ('traditional_comparison.json', rows, MAIN_METRICS),
            ('dose_response.json', doses, ('d', 'no_trigger_tmpjpe_mm',
                                         'triggered_tmpjpe_mm', 'improvement_mm'))):
        path = root / name
        if path.exists():
            saved = _read_json(path)
            if (saved.get('profile') != PROFILE or saved.get('plan_sha256') != matrix['plan_sha256']
                    or not _same_numeric_rows(saved.get('rows'), expected, keys)):
                raise ValueError(f'Frozen {name} disagrees with the six complete caches')
            files[name] = _file_sha(path)
    # Detect a reference file changing while its hashes/passports are read.
    if any(_file_sha(root / name) != value for name, value in files.items()):
        raise ValueError('Frozen reference files changed during the read-only audit')
    result = dict(rows=rows, dose_response=doses, inputs=records, configs=configs,
        poison_plans=poisons, common_inputs=common,
        common_poison_plan_sha256=_sha_json([[s[0], s[1]] for s in common_poison]),
        common_poison_identities=common_poison,
        provenance=dict(reference_run=str(root), reference_profile=PROFILE,
            plan_sha256=matrix['plan_sha256'], file_sha256=files,
            historical_sources=matrix['sources'], current_native_sha256=current_sources,
            changed_historical_sources=changed_sources, cache_provenance=cache_sources,
            content_scope='ordered dataset IDs/action bytes; not every CSI/pose file byte',
            budget='historical peak-only; no new L2 ceiling or baseline re-evaluation'))
    if data_home is not None:
        from train_backdoor import _load_dataset
        from run_peak_confirmation import _pair_ids
        cfg = dict(configs['blended'], dataset_root=str(Path(data_home) / 'datasets/Compress'),
                   action_npy=str(Path(data_home) / 'actions/data_bend.npy'), device='cpu')
        current = dict(action_file_sha256=_file_sha(cfg['action_npy']))
        for name, split in (('train', 'training'), ('eval', 'test')):
            ds = _load_dataset(cfg, split)
            ids = _pair_ids(ds)
            if len(ids) != len({tuple(v) for v in ids}):
                raise ValueError('Current reference data contains duplicate ordered pair IDs')
            current[name] = dict(n=len(ids), pair_ids_sha256=_sha_json(ids))
        validate_candidate_compatibility(result, current)
    return result


def validate_candidate_compatibility(references, input_record, poison_manifest=None):
    """Fail closed if a new Proposed victim uses a different benchmark task."""
    if _input_identity(input_record) != references['common_inputs']:
        raise ValueError('New Proposed uses different full ordered splits or action bytes from frozen references')
    if poison_manifest is not None:
        digest, identities = _poison_identity(poison_manifest)
        _validate_cover(poison_manifest, 'proposed')
        if (digest != references['common_poison_plan_sha256']
                or identities != references['common_poison_identities']):
            raise ValueError('New Proposed poison IDs/frames/paired doses differ from frozen references')
    return True
