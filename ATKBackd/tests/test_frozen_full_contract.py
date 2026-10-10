"""Frozen-only full victim contracts; never load MM-Fi or start training."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frozen_full_contract as full
import run_paired_guard_drafts as drafts
import train_backdoor as trainer


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cfg(tmp_path):
    source_cfg = copy.deepcopy(drafts.build_matrix(tmp_path / 'data', tmp_path / 'old',
        'cpu', 0)['cells'][2]['cfg'])
    snapshot = tmp_path / 'source_frozen'
    snapshot.mkdir()
    action = Path(source_cfg['action_npy'])
    action.parent.mkdir(parents=True)
    action.write_bytes(b'action fixture bytes; never opened as an array')
    action_sha = hashlib.sha256(action.read_bytes()).hexdigest()
    recipe = 'a' * 64
    fitting = dict(variant='paired_guard', recipe_sha256=recipe,
        official_test_loaded=False, external_draft_holdout_loaded=False,
        no_initialization_weight_transfer=True, utility_gate_required=True,
        selected_utility_gate_passed=False, no_eligible_candidate=True,
        action_sha256=action_sha)
    fitting_sha = _write_json(snapshot / 'fitting.json', fitting)
    artifact_sha = _write_json(snapshot / 'learned_trigger.json', dict(
        variant='paired_guard', recipe_sha256=recipe, provenance=fitting))
    source_cfg.update(lc_recipe_sha256=recipe, lc_fitting_sha256=fitting_sha,
        lc_artifact_sha256=artifact_sha,
        lc_artifact_path='Z:/historical/source/no-longer-mounted/learned_trigger.json')
    prepared_sha = _write_json(snapshot / 'prepared_cfg.json', dict(
        schema=1, cfg=source_cfg, cfg_fingerprint=trainer._config_fingerprint(source_cfg),
        action_file_sha256=action_sha, artifact_sha256=artifact_sha,
        fitting_sha256=fitting_sha, recipe_sha256=recipe))
    result = {key: copy.deepcopy(value) for key, value in source_cfg.items()
              if not key.startswith('draft_')}
    result.update(method_draft=False, epochs=50, victim_epochs=50,
        confirmation_profile=full.FULL_CONFIRMATION_PROFILE,
        lc_artifact_path=str(snapshot / 'learned_trigger.json'),
        confirmation_source=dict(source_cell_dir=str(snapshot),
            prepared_cfg_sha256=prepared_sha, fitting_sha256=fitting_sha,
            artifact_sha256=artifact_sha, source_recipe_sha256=recipe,
            source_plan_sha256='b' * 64, selected_utility_gate_passed=False,
            no_eligible_candidate=True))
    return result


def forbidden(*args, **kwargs):
    raise AssertionError('dataset/model/fitting must not be opened')


def test_explicit_full_profile_validates_metadata_and_rebased_source(cfg):
    full.validate_frozen_full_config(cfg)
    full.validate_frozen_full_source_files(cfg)
    trainer._validate_training_contract(trainer._resolve_training_config(cfg))
    assert not Path('Z:/historical/source/no-longer-mounted/learned_trigger.json').exists()
    assert cfg['confirmation_source']['no_eligible_candidate'] is True


@pytest.mark.parametrize('key,value', [
    ('confirmation_profile', 'unregistered_full'), ('method_draft', True),
    ('draft_profile', 'paired_guard_screen_v1'), ('draft_train_samples', 0),
    ('draft_eval_source', 'official_test'), ('epochs', 15), ('victim_epochs', 15),
    ('strict_resume', False), ('seed', 0), ('seed', True), ('rho', .4),
    ('lr', .01), ('lr', float('nan')), ('batch_size', 64), ('optimizer', 'adamw'),
    ('momentum', 0), ('weight_decay', .01), ('model', 'other'),
    ('trigger', 'blended'), ('lc_variant', 'trainaware'), ('lc_bank_size', 8),
    ('lc_relative_l2', .2), ('lc_reference_eps', .2), ('eps', .2),
    ('lc_utility_mpjpe_tolerance', .005), ('lc_utility_pa_tolerance', .003),
    ('lc_utility_pck_tolerance', .01), ('lc_poison_indices', [1]),
    ('comparison_peak_budget', 'original_postclip_linf_v1'),
    ('victim_loss', 'mse'), ('attacker_access', 'white_box_training_control'),
    ('threat_model', 'inference_only'), ('training_protocol', 'staged'),
    ('dose_coupling', 'shuffled'), ('dose_min', 0), ('dose_max', .8),
    ('dose_grid', [0., 1.]), ('dose_grid', [False, .2, .4, .6, .8, 1.]),
    ('target_joints', [1, 2]), ('pivot', 2), ('theta_max_deg', 50),
    ('payload_axis', [1., 0., 0.]), ('pretrained', True), ('data_parallel', True),
    ('lr_scheduler', True), ('mmfi_protocol', 'protocol3'), ('mmfi_setting', 's3'),
    ('mmfi_split_seed', 1), ('clean_label_cover_ratio', .1)])
def test_full_scope_does_not_accept_operator_or_protocol_drift(cfg, key, value):
    cfg[key] = value
    with pytest.raises(ValueError):
        trainer._validate_training_contract(cfg)


@pytest.mark.parametrize('key', full._SOURCE_HASHES)
@pytest.mark.parametrize('value', ['', 'a' * 63, 'A' * 64, True, None])
def test_source_hash_types_shapes_fail_closed(cfg, key, value):
    cfg['confirmation_source'][key] = value
    with pytest.raises(ValueError, match='SHA256'):
        full.validate_frozen_full_config(cfg)


@pytest.mark.parametrize('field', ['lc_artifact_sha256', 'lc_fitting_sha256', 'lc_recipe_sha256'])
def test_source_config_hashes_cannot_be_unbound(cfg, field):
    cfg[field] = 'c' * 64
    with pytest.raises(ValueError, match='binding'):
        full.validate_frozen_full_config(cfg)


@pytest.mark.parametrize('field', full._SOURCE_FLAGS)
@pytest.mark.parametrize('value', [None, 1, 'True'])
def test_surrogate_gate_status_requires_explicit_booleans(cfg, field, value):
    cfg['confirmation_source'][field] = value
    with pytest.raises(ValueError, match='boolean'):
        full.validate_frozen_full_config(cfg)


def test_success_claim_cannot_replace_actual_source_fallback(cfg):
    cfg['confirmation_source'].update(selected_utility_gate_passed=True,
                                      no_eligible_candidate=False)
    with pytest.raises(ValueError, match='actual source fitting'):
        full.validate_frozen_full_source_files(cfg)


@pytest.mark.parametrize('filename', ['prepared_cfg.json', 'fitting.json', 'learned_trigger.json'])
def test_source_bytes_checked_before_any_dataset_or_cached_result(cfg, monkeypatch, filename):
    file = Path(cfg['confirmation_source']['source_cell_dir']) / filename
    file.write_bytes(file.read_bytes() + b' ')
    monkeypatch.setattr(trainer, '_load_dataset', forbidden)
    monkeypatch.setattr(trainer, '_load_cached_result', forbidden)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        trainer.train(cfg)


def test_source_action_bytes_checked_before_cache(cfg, monkeypatch):
    Path(cfg['action_npy']).write_bytes(b'changed action')
    monkeypatch.setattr(trainer, '_load_cached_result', forbidden)
    with pytest.raises(ValueError, match='action bytes changed'):
        trainer.train(cfg)


def test_fitting_options_cannot_be_edited_after_artifact_freeze(cfg):
    cfg['lc_outer_steps'] += 1
    with pytest.raises(ValueError, match='option changed'):
        full.validate_frozen_full_source_files(cfg)
    cfg['lc_outer_steps'] -= 1
    cfg['lc_new_secret_tuning'] = 1
    with pytest.raises(ValueError, match='cannot be added'):
        full.validate_frozen_full_source_files(cfg)


def test_prepared_fingerprint_still_checked_after_rebinding_bytes(cfg):
    path = Path(cfg['confirmation_source']['source_cell_dir']) / 'prepared_cfg.json'
    prepared = json.loads(path.read_text())
    prepared['cfg_fingerprint'] = 'c' * 64
    cfg['confirmation_source']['prepared_cfg_sha256'] = _write_json(path, prepared)
    with pytest.raises(ValueError, match='fingerprint'):
        full.validate_frozen_full_source_files(cfg)


def test_source_requires_official_train_only_provenance(cfg):
    folder = Path(cfg['confirmation_source']['source_cell_dir'])
    fitting = json.loads((folder / 'fitting.json').read_text())
    fitting['official_test_loaded'] = True
    fitting_sha = _write_json(folder / 'fitting.json', fitting)
    prepared = json.loads((folder / 'prepared_cfg.json').read_text())
    prepared['cfg']['lc_fitting_sha256'] = fitting_sha
    prepared['fitting_sha256'] = fitting_sha
    prepared['cfg_fingerprint'] = trainer._config_fingerprint(prepared['cfg'])
    cfg['lc_fitting_sha256'] = fitting_sha
    cfg['confirmation_source']['fitting_sha256'] = fitting_sha
    cfg['confirmation_source']['prepared_cfg_sha256'] = _write_json(folder / 'prepared_cfg.json', prepared)
    with pytest.raises(ValueError, match='TRAIN-only'):
        full.validate_frozen_full_source_files(cfg)


def test_complete_valid_cache_does_not_open_dataset_but_checks_snapshot(cfg, monkeypatch):
    monkeypatch.setattr(trainer, '_load_cached_result', lambda *args: {'fixture': 'done'})
    monkeypatch.setattr(trainer, '_load_dataset', forbidden)
    assert trainer.train(cfg) == (None, {'fixture': 'done'})


def test_full_loader_opens_true_splits_without_draft_views(cfg, monkeypatch):
    calls = []
    def dataset(**kwargs):
        calls.append(kwargs)
        return object()
    monkeypatch.setattr(trainer, 'MMFI', dataset)
    train = trainer._load_dataset(cfg, 'training')
    test = trainer._load_dataset(cfg, 'test')
    assert train is not test
    assert [call['split'] for call in calls] == ['training', 'test']
    assert all(call['protocol'] == 'protocol1' and call['setting'] == 's1' for call in calls)


def test_invalid_full_scope_fails_before_constructing_test_dataset(cfg, monkeypatch):
    cfg['draft_eval_source'] = 'official_test'
    monkeypatch.setattr(trainer, 'MMFI', forbidden)
    with pytest.raises(ValueError, match='draft/subset'):
        trainer._load_dataset(cfg, 'test')


def test_old_train_only_contract_is_not_weakened(cfg):
    old = json.loads((Path(cfg['confirmation_source']['source_cell_dir']) /
                      'prepared_cfg.json').read_text())['cfg']
    trainer._validate_training_contract(old)
    old['draft_eval_source'] = 'official_test'
    with pytest.raises(ValueError, match='training-holdout'):
        trainer._validate_training_contract(old)
    old.pop('draft_profile')
    with pytest.raises(ValueError):
        trainer._validate_training_contract(old)


@pytest.mark.parametrize('profile', [None, 'unregistered_full', 'mmfi_peak_confirmation_v1'])
def test_renamed_frozen_profile_cannot_bypass_source_checks(cfg, monkeypatch, profile):
    if profile is None:
        cfg.pop('confirmation_profile')
    else:
        cfg['confirmation_profile'] = profile
    monkeypatch.setattr(trainer, 'MMFI', forbidden)
    monkeypatch.setattr(trainer, '_load_cached_result', forbidden)
    with pytest.raises(ValueError, match='unknown frozen full'):
        trainer._load_dataset(cfg, 'test')
    with pytest.raises(ValueError, match='unknown frozen full'):
        trainer.train(cfg)


def test_existing_peak_confirmation_does_not_enter_frozen_source_contract(tmp_path, monkeypatch):
    import run_peak_confirmation as peak
    monkeypatch.setattr(peak, '_source_provenance', lambda: {'fixture.py': 'a' * 64})
    matrix = peak.build_matrix(tmp_path / 'data', tmp_path / 'old_full', 'cpu', 0)
    calls = []
    monkeypatch.setattr(full, 'validate_frozen_full_source_files', forbidden)
    monkeypatch.setattr(trainer, 'MMFI', lambda **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr(trainer, '_load_cached_result', lambda *args: {'fixture': 'old complete'})
    for cell in matrix['cells']:
        old = cell['cfg']
        trainer._validate_training_contract(old)
        trainer._load_dataset(old, 'training')
        trainer._load_dataset(old, 'test')
        assert trainer.train(old) == (None, {'fixture': 'old complete'})
    assert [call['split'] for call in calls] == ['training', 'test', 'training', 'test']
