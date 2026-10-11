"""Frozen native references are read-only, complete and task-compatible."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import simple_reference as reference
import run_traditional_comparison as traditional
from mmfi_tables import config_fingerprint, _trainer_schemas
from run_method_drafts import _sha_json, _file_sha
from run_mmfi_tables import _plan_fingerprint


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def forbidden(*args, **kwargs):
    pytest.fail('Frozen reference auditing must never train/evaluate/build a trigger')


def write_synthetic_reference(root, data_home):
    """Synthetic audit files only; no reference model is created or trained."""
    matrix = traditional.build_matrix(data_home, root, 'cpu', 0,
        budget_mode='shared_peak', baseline_set='extended')
    matrix['metrics_contract']['full_split_counts'] = dict(reference.EXPECTED_COUNTS)
    # Dispatch and training changed since the historical run. Native source
    # bytes must still match; do not pretend these historical bytes are current.
    for name in ('train_backdoor.py', 'attack/trigger.py', 'attack/poison.py',
                 'data_utils/draft_subset.py'):
        matrix['sources']['source_sha256'][name] = 'b' * 64
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    write(root / 'traditional_comparison.resolved.json', matrix)
    n_train, n_eval = reference.EXPECTED_COUNTS.values()
    indices = [1, 9]
    samples = [dict(index=i, dose=d,
        csi_id=str(data_home / f'datasets/Compress/E01/S01/A01/wifi-csi/frame{i:03d}_processed.npy'),
        pose_id=str(data_home / 'datasets/Compress/E01/S01/A01/ground_truth.npy'), frame_idx=i - 1)
        for i, d in zip(indices, [.3, .7])]
    assert len(samples) == int(.1 * n_train)
    digest = _sha_json([[s['index'], s['dose']] for s in samples])
    for cell in matrix['cells']:
        key, cfg = cell['method_key'], cell['cfg']
        folder = root / key
        fingerprint = config_fingerprint(cfg)
        record = dict(schema=1, profile=reference.PROFILE, config_fingerprint=fingerprint,
            action_file_sha256='a' * 64,
            train=dict(n=n_train, pair_ids_sha256='c' * 64),
            eval=dict(n=n_eval, pair_ids_sha256='d' * 64))
        if key == 'fiba':
            record.update(fiba_reference_train_index=0, fiba_fixed_key_sha256='e' * 64,
                          fiba_reference_sha256='f' * 64)
        write(folder / 'comparison_inputs.json', record)
        cover_ratio = .2 if key == 'wanet_source' else .1 if key == 'fiba' else 0.
        cover = list(range(10, 10 + int(cover_ratio * n_train)))
        poison = dict(schema=4, config_fingerprint=fingerprint, n_total=n_train,
            n_poison=len(samples), rho_requested=.1, seed=42, selection='uniform',
            dose_min=.2, dose_max=1., poison_plan_sha256=digest,
            pivot=1, target_joints=[2, 3], theta_max_deg=40.,
            payload_axis=[0., 0., 1.], dose_mode='linear', samples=copy.deepcopy(samples))
        if cover:
            poison.update(schema=5, dose_coupling='paired', cover_ratio=cover_ratio,
                n_cover=len(cover), cover_indices=cover,
                coupling_sha256=_sha_json([[s['index'], s['dose'], s['dose']] for s in samples]))
        write(folder / 'poison_manifest.json', poison)
        res = dict(clean_mpjpe=.174, clean_pampjpe=.104,
            **{f'clean_pck@{t:.1f}': .8 * t for t in (.5, .4, .3, .2, .1)},
            tmpjpe=[.1, .11, .12, .13, .14, .15],
            clean_to_target_tmpjpe=[.1, .15, .2, .25, .3, .4],
            dose_grid=reference.GRID, training_contract='ordinary_erm',
            attacker_access='data_only', dose_coupling='paired', n_total=n_train,
            n_poison=len(samples), n_cover=len(cover), poison_plan_sha256=digest)
        write(folder / 'eval_cache.json', dict(result_schema=_trainer_schemas()[1],
            cfg_fingerprint=fingerprint, trained_epochs=50, cfg=cfg, res=res))
    return matrix


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, 'EXPECTED_COUNTS', {'train': 20, 'eval': 4})
    root = tmp_path / 'unchanged_reference'
    data_home = tmp_path / 'data'
    matrix = write_synthetic_reference(root, data_home)
    import train_backdoor as trainer
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', forbidden)
    monkeypatch.setattr(trainer, 'evaluate', forbidden)
    return root, data_home, matrix


def test_reference_audit_retains_all_rows_doses_and_native_source_identity_without_writes(frozen):
    root, _, matrix = frozen
    before = {p.relative_to(root).as_posix(): _file_sha(p) for p in root.rglob('*') if p.is_file()}
    result = reference.load_frozen_references(root)
    after = {p.relative_to(root).as_posix(): _file_sha(p) for p in root.rglob('*') if p.is_file()}
    assert after == before
    assert [r['method_key'] for r in result['rows']] == list(reference.REFERENCE_KEYS)
    assert len(result['dose_response']) == 36
    assert result['provenance']['plan_sha256'] == matrix['plan_sha256']
    assert result['provenance']['file_sha256'] == before
    assert result['rows'][1]['t1_mpjpe_mm'] == 150
    assert all(k in result['rows'][1] for k in reference.MAIN_METRICS)
    assert set(result['provenance']['changed_historical_sources']) == {
        'train_backdoor.py', 'attack/trigger.py', 'attack/poison.py', 'data_utils/draft_subset.py'}
    assert result['poison_plans']['wanet_source']['n_cover'] == 4
    assert result['poison_plans']['fiba']['n_cover'] == 2
    assert result['common_poison_plan_sha256'] == result['poison_plans']['proposed']['poison_plan_sha256']


@pytest.mark.parametrize('filename', ['traditional_comparison.resolved.json',
    'blended/eval_cache.json', 'blended/comparison_inputs.json', 'blended/poison_manifest.json'])
def test_missing_required_reference_file_is_rejected_without_retraining(frozen, filename):
    root, _, _ = frozen
    (root / filename).unlink()
    with pytest.raises((ValueError, OSError)):
        reference.load_frozen_references(root)


@pytest.mark.parametrize('key,value', [('trained_epochs', 15), ('result_schema', 9),
                                    ('cfg_fingerprint', '0' * 64)])
def test_edited_partial_stale_cache_is_rejected(frozen, key, value):
    root, _, _ = frozen
    path = root / 'blended/eval_cache.json'
    blob = read(path)
    blob[key] = value
    write(path, blob)
    with pytest.raises(ValueError):
        reference.load_frozen_references(root)


@pytest.mark.parametrize('key,value', [('clean_mpjpe', float('nan')),
    ('clean_pampjpe', float('inf')), ('clean_pck@0.1', 2.),
    ('tmpjpe', [.1] * 5), ('dose_grid', [0., 1.]), ('n_total', 10),
    ('n_cover', 2), ('poison_plan_sha256', 'e' * 64), ('dose_coupling', 'shuffled')])
def test_incomplete_nonfinite_wrong_protocol_metrics_are_rejected(frozen, key, value):
    root, _, _ = frozen
    path = root / 'blended/eval_cache.json'
    blob = read(path)
    blob['res'][key] = value
    write(path, blob)
    with pytest.raises(ValueError):
        reference.load_frozen_references(root)


@pytest.mark.parametrize('key,value', [('epochs', 15), ('rho', .4), ('pivot', 0),
    ('dose_min', 1.), ('dose_coupling', 'shuffled'), ('simple_relative_l2', .1),
    ('blended_implementation', 'improved'), ('eps', .19), ('lr', .002),
    ('blended_source_commit', 'new'), ('victim_epochs', 50.)])
def test_frozen_reference_settings_cannot_be_changed_with_recomputed_plan(frozen, key, value):
    root, _, matrix = frozen
    matrix['cells'][1]['cfg'][key] = value
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    write(root / 'traditional_comparison.resolved.json', matrix)
    with pytest.raises(ValueError):
        reference.load_frozen_references(root)


@pytest.mark.parametrize('change', ['removed', 'reordered', 'peak_mode', 'count', 'native_hash'])
def test_reference_manifest_and_source_edits_fail_closed(frozen, change):
    root, _, matrix = frozen
    if change == 'removed':
        matrix['cells'].pop(1)
    elif change == 'reordered':
        matrix['cells'].reverse()
    elif change == 'peak_mode':
        matrix['metrics_contract']['budget_mode'] = 'dual'
    elif change == 'count':
        matrix['metrics_contract']['full_split_counts']['train'] = 10
    else:
        matrix['sources']['source_sha256']['attack/traditional.py'] = '0' * 64
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    write(root / 'traditional_comparison.resolved.json', matrix)
    with pytest.raises(ValueError):
        reference.load_frozen_references(root)


@pytest.mark.parametrize('change', ['action', 'split', 'count', 'fiba_key',
    'poison_dose', 'poison_frame', 'cover_count', 'coupling'])
def test_reference_input_and_native_poison_passports_must_match(frozen, change):
    root, _, _ = frozen
    if change in ('action', 'split', 'count', 'fiba_key'):
        path = root / ('fiba/comparison_inputs.json' if change == 'fiba_key'
                       else 'blended/comparison_inputs.json')
        value = read(path)
        if change == 'action':
            value['action_file_sha256'] = '0' * 64
        elif change == 'split':
            value['eval']['pair_ids_sha256'] = '0' * 64
        elif change == 'count':
            value['train']['n'] = 10
        else:
            value['fiba_reference_train_index'] = 1
    else:
        path = root / ('wanet_source/poison_manifest.json' if change in ('cover_count', 'coupling')
                       else 'blended/poison_manifest.json')
        value = read(path)
        if change == 'poison_dose':
            value['samples'][0]['payload_dose'] = .9
        elif change == 'poison_frame':
            value['samples'][0]['frame_idx'] = 100
        elif change == 'cover_count':
            value['cover_indices'].pop()
        else:
            value['coupling_sha256'] = '0' * 64
    write(path, value)
    with pytest.raises(ValueError):
        reference.load_frozen_references(root)


def test_candidate_compatibility_is_relocation_safe_but_checks_ids_dose_and_frames(frozen):
    root, data_home, _ = frozen
    result = reference.load_frozen_references(root)
    current = copy.deepcopy(result['inputs']['proposed'])
    poison = copy.deepcopy(result['poison_plans']['proposed'])
    assert reference.validate_candidate_compatibility(result, current)
    for sample in poison['samples']:
        for name in ('csi_id', 'pose_id'):
            sample[name] = sample[name].replace(str(data_home), '/different/current/data_home')
    assert reference.validate_candidate_compatibility(result, current, poison)
    poison['samples'][0]['frame_idx'] += 1
    with pytest.raises(ValueError, match='IDs/frames/paired doses'):
        reference.validate_candidate_compatibility(result, current, poison)


@pytest.mark.parametrize('change', ['action', 'train', 'dose', 'identity', 'cover'])
def test_new_proposed_cannot_use_a_different_task_under_frozen_numeric_rows(frozen, change):
    root, _, _ = frozen
    result = reference.load_frozen_references(root)
    current = copy.deepcopy(result['inputs']['proposed'])
    poison = copy.deepcopy(result['poison_plans']['proposed'])
    if change == 'action':
        current['action_file_sha256'] = '0' * 64
    elif change == 'train':
        current['train']['pair_ids_sha256'] = '0' * 64
    elif change == 'dose':
        poison['samples'][0]['dose'] = .4
    elif change == 'identity':
        poison['samples'][0]['csi_id'] = '../other.npy'
    else:
        poison['n_cover'], poison['cover_indices'] = 1, [19]
    with pytest.raises(ValueError):
        reference.validate_candidate_compatibility(result, current, poison)


@pytest.mark.parametrize('change', ['results', 'yaml'])
def test_optional_reference_result_and_resolved_yaml_are_bound_and_cross_checked(frozen, change):
    root, _, matrix = frozen
    cell = matrix['cells'][1]
    blob = read(root / 'blended/eval_cache.json')
    if change == 'results':
        path = root / 'blended/results.json'
        write(path, dict(method_key='blended', cfg=cell['cfg'], res=blob['res'],
                        cfg_fingerprint=config_fingerprint(cell['cfg'])))
        assert 'blended/results.json' in reference.load_frozen_references(root)['provenance']['file_sha256']
        value = read(path)
        value['res']['clean_mpjpe'] = .1
        write(path, value)
    else:
        import yaml
        path = root / 'blended/config.resolved.yaml'
        path.write_text(yaml.safe_dump(cell['cfg']), encoding='utf-8')
        assert 'blended/config.resolved.yaml' in reference.load_frozen_references(root)['provenance']['file_sha256']
        cfg = dict(cell['cfg'], lr=.01)
        path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(ValueError):
        reference.load_frozen_references(root)


@pytest.mark.parametrize('filename,metric', [('traditional_comparison.json', 't1_mpjpe_mm'),
                                         ('dose_response.json', 'triggered_tmpjpe_mm')])
def test_optional_exported_json_must_agree_with_complete_raw_caches(frozen, filename, metric):
    root, _, _ = frozen
    result = reference.load_frozen_references(root)
    rows = result['rows'] if filename == 'traditional_comparison.json' else result['dose_response']
    data = dict(profile=reference.PROFILE, plan_sha256=result['provenance']['plan_sha256'],
                rows=copy.deepcopy(rows))
    write(root / filename, data)
    assert filename in reference.load_frozen_references(root)['provenance']['file_sha256']
    data['rows'][1][metric] += 1.
    write(root / filename, data)
    with pytest.raises(ValueError, match='complete caches'):
        reference.load_frozen_references(root)


@pytest.mark.parametrize('key', ['wanet_source', 'fiba'])
@pytest.mark.parametrize('location', ['manifest', 'cache'])
@pytest.mark.parametrize('workers', [4, False])
def test_native_cover_workers_cannot_be_tampered_as_runtime_only(frozen, key, location, workers):
    root, _, matrix = frozen
    if location == 'manifest':
        cell = next(cell for cell in matrix['cells'] if cell['method_key'] == key)
        cell['cfg']['num_workers'] = workers
        # Runtime fields do not change the generic plan/config fingerprints;
        # the native cover RNG's source-specific contract still must reject.
        matrix['plan_sha256'] = _plan_fingerprint(matrix)
        write(root / 'traditional_comparison.resolved.json', matrix)
    else:
        path = root / key / 'eval_cache.json'
        blob = read(path)
        blob['cfg']['num_workers'] = workers
        write(path, blob)
    with pytest.raises(ValueError, match='native cover RNG'):
        reference.load_frozen_references(root)
