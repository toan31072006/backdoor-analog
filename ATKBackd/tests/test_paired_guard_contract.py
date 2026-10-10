"""Synthetic integration contracts for the isolated paired-clean-twin draft."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import carrier_bank_fit as bank_fitting
import learned_carrier_fit as legacy_fitting
import train_backdoor as training
from attack.learned_carrier import (
    BANK_VARIANTS, LEGACY_VARIANTS, FrozenLearnedCarrier, TrainableCarrier, artifact_dict,
    operator_config_keys, resolve_learned_config, write_artifact,
)
from data_utils.draft_subset import DraftSubset, apply_draft_subset
from test_carrier_bank_operator import assert_budgets
from test_learned_carrier_fit import (
    SyntheticTrainingDataset, ToyHPE, config as legacy_config, original,
)


PROFILE = 'paired_guard_screen_v1'
ZERO_TOLERANCES = (
    'lc_utility_mpjpe_tolerance', 'lc_utility_pa_tolerance',
    'lc_utility_pck_tolerance',
)


def config(**updates):
    cfg = dict(experiment_name='mmfi', dataset_root='unused',
        trigger_zero_mean=True, n_ant=3, n_sub=8, n_pkt=4, eps=.185,
        method_draft=True, draft_profile=PROFILE,
        draft_eval_source='training_holdout', draft_subset_seed=0,
        draft_train_samples=40, draft_eval_samples=10,
        victim_loss='mpjpe', trigger='learned_carrier',
        lc_variant='paired_guard', lc_relative_l2=.1)
    cfg.update(updates)
    return cfg


class MetadataDataset:
    def __init__(self, *, split='training', **kwargs):
        self.split = split
        self.items = [dict(csi=f'x{i}', kpt=f'y{i}', frame_idx=i)
                      for i in range(100)]

    def __len__(self):
        return len(self.items)


def forbidden(*args, **kwargs):
    pytest.fail('An invalid paired-guard recipe must fail before data/model access')


def test_paired_guard_has_the_same_normalized_two_basis_operator_as_trainaware():
    base = original()
    paired = TrainableCarrier(base, config(), 'paired_guard')
    previous = TrainableCarrier(base, config(lc_variant='trainaware'), 'trainaware')
    assert paired.cfg['lc_bank_size'] == 2
    assert paired.cfg['lc_bank_seed'] == previous.cfg['lc_bank_seed'] == 42
    assert paired.carrier_bank.shape == (2, 3, 8, 4)
    assert torch.equal(paired.carrier_bank, previous.carrier_bank)
    assert torch.equal(paired.weights, previous.weights)
    assert paired.weights.requires_grad and not paired.mask_scores.requires_grad
    assert paired.export_amplitude() == 1
    assert paired.bank_diagnostics == previous.bank_diagnostics
    np.testing.assert_array_equal(paired.export_pattern(), previous.export_pattern())


@pytest.mark.parametrize('size', [8, 4, 1, True, 2.5])
def test_paired_guard_rejects_any_other_bank_size(size):
    with pytest.raises(ValueError, match='lc_bank_size'):
        resolve_learned_config(config(lc_bank_size=size))


def test_frozen_paired_guard_numpy_and_torch_parity_raw_bytes_hash_and_budgets(tmp_path):
    base = original()
    module = TrainableCarrier(base, config(), 'paired_guard')
    with torch.no_grad():
        module.weights.copy_(torch.tensor([-.7, 1.1], dtype=torch.float64))
    artifact = tmp_path / 'paired.json'
    digest = write_artifact(artifact, module, 'a' * 64, {'training_only': True})
    frozen_cfg = dict(module.cfg, lc_artifact_path=str(artifact),
        lc_artifact_sha256=digest, lc_recipe_sha256='a' * 64)
    frozen = FrozenLearnedCarrier(base, frozen_cfg)
    payload = json.loads(artifact.read_bytes())
    assert payload['variant'] == frozen.variant == 'paired_guard'
    assert payload['raw_weights'] == [-.7, 1.1]
    assert payload['operator_config']['lc_bank_size'] == 2
    assert payload['bank_diagnostics'] == module.bank_diagnostics
    assert digest == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert len(operator_config_keys(module.cfg)) == 8
    random = np.random.default_rng(814).uniform(0, 1, module.shape).astype(np.float32)
    inputs = np.stack([random, np.zeros_like(random), np.ones_like(random)])
    for dose in (0., .2, .4, .6, .8, 1.):
        actual = module.inject_tensor(torch.from_numpy(inputs), dose).detach().numpy()
        for value, output in zip(inputs, actual):
            np.testing.assert_allclose(output, frozen.inject(value, dose), rtol=0, atol=2e-7)
            assert_budgets(base, value, output, dose)
            assert_budgets(base, value, frozen.inject(value, dose), dose)
            if dose == 0:
                np.testing.assert_array_equal(output, value)
    artifact.write_bytes(artifact.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='bytes SHA256 mismatch'):
        FrozenLearnedCarrier(base, frozen_cfg)


def test_legacy_operator_defaults_and_artifact_shape_remain_unchanged():
    assert LEGACY_VARIANTS == ('weights', 'sparse', 'combined', 'gradient', 'energy')
    assert BANK_VARIANTS == ('trainaware', 'bank', 'bank_guard')
    cfg = {key: value for key, value in config(lc_variant='weights').items()
           if not key.startswith('lc_')}
    resolved = resolve_learned_config(cfg)
    assert 'lc_bank_size' not in resolved and 'lc_bank_seed' not in resolved
    assert len(operator_config_keys(resolved)) == 6
    module = TrainableCarrier(original(), cfg, 'weights')
    payload = artifact_dict(module, 'b' * 64)
    assert 'raw_weights' not in payload and 'bank_diagnostics' not in payload
    assert set(payload['operator_config']) == set(operator_config_keys(resolved))
    assert resolve_learned_config(config(lc_variant='trainaware'))['lc_bank_size'] == 2
    assert resolve_learned_config(config(lc_variant='bank'))['lc_bank_size'] == 8
    assert resolve_learned_config(config(lc_variant='bank_guard'))['lc_bank_size'] == 8


def test_existing_fitting_defaults_and_public_signatures_remain_unchanged():
    expected_legacy = dict(lc_warmup_epochs=3, lc_rounds=3, lc_inner_steps=32,
        lc_outer_steps=16, lc_batch_size=32, lc_lr=.02, lc_energy_weight=.2,
        lc_clean_weight=1., lc_fit_samples=4096, lc_inner_val_fraction=.2,
        lc_gradient_tensors=2, lc_fit_seed=4242)
    assert legacy_fitting.FIT_DEFAULTS == expected_legacy
    expected_bank = dict(expected_legacy, lc_warmup_epochs=10, lc_inner_steps=128,
        lc_outer_steps=24, lc_surrogate_count=2, lc_surrogate_seed_stride=101,
        lc_lookahead_steps=2, lc_pa_weight=1., lc_pck_weight=.1,
        lc_pck_temperature=.02, lc_utility_mpjpe_tolerance=.005,
        lc_utility_pa_tolerance=.003, lc_utility_pck_tolerance=.01)
    assert bank_fitting.BANK_FIT_DEFAULTS == expected_bank
    signatures = (
        (TrainableCarrier, '(base, cfg, variant)'),
        (FrozenLearnedCarrier, '(base, cfg)'),
        (legacy_fitting.prepare_learned_cell, '(cfg, folder, recipe_sha256)'),
        (legacy_fitting.fit_trigger,
         '(model, trigger, fit_x, fit_y, val_x, val_y, cfg, rng=None)'),
        (bank_fitting.fit_bank_trigger,
         '(models, trigger, fit_x, fit_y, val_x, val_y, cfg, rng=None)'),
    )
    for function, signature in signatures:
        assert str(inspect.signature(function)) == signature


@pytest.mark.parametrize('profile', [
    None, 'method_screening_v1', 'method_peak_control_v1',
    'learned_carrier_screen_v1', 'carrier_bank_screen_v1',
])
def test_paired_variant_is_exclusive_to_new_draft_profile(profile):
    with pytest.raises(ValueError, match='paired_guard_screen_v1'):
        training._validate_training_contract(config(draft_profile=profile))


@pytest.mark.parametrize('variant', ['paired_guard', 'trainaware'])
def test_new_profile_allows_paired_guard_and_existing_trainaware(variant):
    training._validate_training_contract(config(lc_variant=variant))


@pytest.mark.parametrize('updates', [
    {'draft_eval_source': 'official_test'}, {'draft_eval_source': None},
    {'method_draft': False}, {'experiment_name': 'one-person'},
])
def test_lower_dataset_loader_rejects_invalid_profile_before_data_factory(monkeypatch, updates):
    monkeypatch.setattr(training, 'MMFI', forbidden)
    monkeypatch.setattr(training, 'PersonInWiFi3D', forbidden)
    with pytest.raises(ValueError, match='training.holdout|training_holdout'):
        training._load_dataset(config(**updates), 'test')


def test_new_profile_alone_enforces_train_only_for_blended_without_learned_fields(monkeypatch):
    cfg = {key: value for key, value in config(trigger='blended').items()
           if not key.startswith('lc_')}
    cfg['draft_eval_source'] = 'official_test'
    monkeypatch.setattr(training, 'MMFI', forbidden)
    with pytest.raises(ValueError, match='training.holdout|training_holdout'):
        training._load_dataset(cfg, 'test')


def test_lower_preparer_refuses_official_test_before_dataset_or_action_access(tmp_path, monkeypatch):
    monkeypatch.setattr(training, '_load_dataset', forbidden)
    monkeypatch.setattr(legacy_fitting, '_file_sha', forbidden)
    with pytest.raises(ValueError, match='TRAIN|training.holdout|training_holdout'):
        legacy_fitting.prepare_learned_cell(config(draft_eval_source='official_test'),
            tmp_path / 'never-created', 'c' * 64)
    assert not (tmp_path / 'never-created').exists()


def test_public_paired_fitter_refuses_official_test_before_surrogate_or_tensor_access():
    import paired_guard_fit as fitting
    with pytest.raises(ValueError, match='TRAIN|training.holdout|training_holdout'):
        fitting.fit_paired_trigger(None, None, None, None, None, None,
                                  config(draft_eval_source='official_test'))


def test_paired_fit_defaults_are_strict_and_separate_from_existing_bank_defaults():
    from paired_guard_fit import PAIRED_FIT_DEFAULTS, resolve_paired_fit_config
    assert all(PAIRED_FIT_DEFAULTS[key] == 0 for key in ZERO_TOLERANCES)
    expected = dict(bank_fitting.BANK_FIT_DEFAULTS,
                    **{key: 0. for key in ZERO_TOLERANCES})
    assert PAIRED_FIT_DEFAULTS == expected
    resolved = resolve_paired_fit_config(config())
    assert resolved['lc_surrogate_count'] == resolved['lc_lookahead_steps'] == 2
    for key in ZERO_TOLERANCES:
        with pytest.raises(ValueError, match='zero|0'):
            resolve_paired_fit_config(config(**{key: 1e-15}))


@pytest.mark.parametrize('updates', [
    {'draft_profile': 'carrier_bank_screen_v1'}, {'method_draft': False},
    {'lc_variant': 'trainaware'}, {'lc_surrogate_count': 1},
    {'lc_lookahead_steps': 1},
])
def test_direct_paired_fitter_preserves_profile_and_paired_budget_contract(updates):
    from paired_guard_fit import resolve_paired_fit_config
    with pytest.raises(ValueError):
        resolve_paired_fit_config(config(**updates))


def test_train_and_external_holdout_use_disjoint_views_of_official_train(monkeypatch):
    opened = []

    def parent(**kwargs):
        opened.append(kwargs['split'])
        return MetadataDataset(**kwargs)

    monkeypatch.setattr(training, 'MMFI', parent)
    train = training._load_dataset(config(), 'training')
    holdout = training._load_dataset(config(), 'test')
    assert opened == ['training', 'training']
    assert len(train) == 40 and len(holdout) == 10
    assert not set(train.subset_indices) & set(holdout.subset_indices)
    assert train.draft_subset_manifest()['profile'] == PROFILE
    assert holdout.draft_subset_manifest()['profile'] == PROFILE
    existing = apply_draft_subset(MetadataDataset(),
        config(draft_profile='carrier_bank_screen_v1', lc_variant='trainaware'), 'training')
    assert train.subset_indices == existing.subset_indices


def test_direct_subset_entry_points_refuse_official_test():
    with pytest.raises(ValueError, match='never official_test'):
        apply_draft_subset(MetadataDataset(), config(draft_eval_source='official_test'), 'test')
    with pytest.raises(ValueError, match='never official_test'):
        DraftSubset(MetadataDataset(), [1, 2], split='test', requested_cap=2,
                    profile=PROFILE, eval_source='official_test')


def test_prepare_freezes_paired_audit_from_only_train_and_reuses_it(tmp_path, monkeypatch):
    import attack.trigger as triggers
    import models.factory as models
    opened = []

    def parent(**kwargs):
        opened.append(kwargs['split'])
        assert kwargs['split'] == 'training'
        dataset = SyntheticTrainingDataset()
        dataset.split = kwargs['split']
        return dataset

    monkeypatch.setattr(training, 'MMFI', parent)
    monkeypatch.setattr(triggers, 'build_trigger_by_name', lambda *args, **kwargs: original())
    monkeypatch.setattr(models, 'build_model', lambda *args, **kwargs: ToyHPE())
    cfg = legacy_config('paired_guard')
    cfg.update(draft_profile=PROFILE, dataset_root='unused', trigger='learned_carrier',
               draft_subset_seed=0, draft_train_samples=16, draft_eval_samples=4,
               lc_rounds=1)
    action = tmp_path / 'action.npy'
    action.write_bytes(b'synthetic TRAIN action identity')
    cfg['action_npy'] = str(action)
    folder = tmp_path / 'paired'
    ready = legacy_fitting.prepare_learned_cell(cfg, folder, 'e' * 64)
    assert opened == ['training']
    record = json.loads((folder / 'fitting.json').read_text(encoding='utf-8'))
    assert record['official_test_loaded'] is record['external_draft_holdout_loaded'] is False
    assert record['no_initialization_weight_transfer'] is True
    assert record['surrogate_initialization_seeds'] == [4242, 4343]
    assert record['utility_gate_required'] is True
    assert all(value == 0 for value in record['utility_tolerances'].values())
    for proof in record['clean_twin_clone_proof']:
        assert proof['model_state_equal'] and proof['optimizer_state_equal']
        assert proof['momentum_state_equal'] and proof['parameter_storage_independent']
    rounds = [row for row in record['history'] if row['stage'] == 'alternation']
    assert len(rounds) == 1
    selected = rounds[record['selected_round']]['inner_validation']
    assert record['selected_utility_gate_passed'] == selected['utility_gate_passed']
    assert record['no_eligible_candidate'] is (not selected['utility_gate_passed'])
    assert len(selected['per_surrogate']) == 2
    for member in selected['per_surrogate']:
        gate = member['utility_gate']
        checks = gate['checks']
        assert set(checks['pck']) == {'0.5', '0.4', '0.3', '0.2', '0.1'}
        assert gate['passed'] is (checks['mpjpe'] and checks['pa_mpjpe']
                                  and all(checks['pck'].values()))
        assert set(member['clean_twin_reference']) == {'clean_m', 'pa_m', 'pck'}
    assert ready['lc_variant'] == 'paired_guard' and ready['lc_bank_size'] == 2
    artifact = Path(ready['lc_artifact_path'])
    assert ready['lc_artifact_sha256'] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert ready == legacy_fitting.prepare_learned_cell(cfg, folder, 'e' * 64)
    assert opened == ['training']
    holdout = training._load_dataset(cfg, 'test')
    train_indices = record['draft_train_manifest']['subset_indices']
    fit_parent_indices = {train_indices[i] for i in record['inner_train_indices']}
    validation_parent_indices = {train_indices[i] for i in record['inner_validation_indices']}
    assert not fit_parent_indices & validation_parent_indices
    assert not (fit_parent_indices | validation_parent_indices) & set(holdout.subset_indices)
    artifact.write_bytes(artifact.read_bytes() + b' ')
    with pytest.raises(ValueError, match='missing or altered'):
        legacy_fitting.prepare_learned_cell(cfg, folder, 'e' * 64)


def test_three_arm_runner_preserves_blended_and_current_trainaware_configs(tmp_path, monkeypatch):
    import run_carrier_bank_drafts as previous
    import run_paired_guard_drafts as runner
    monkeypatch.setattr(previous, '_source_provenance', lambda: {'fixture': 'd' * 64})
    monkeypatch.setattr(runner, '_source_provenance', lambda: {'fixture': 'd' * 64})
    options = dict(epochs=3, train_samples=10, eval_samples=4, distortion_samples=4)
    current = runner.build_matrix(tmp_path / 'data', tmp_path / 'new', 'cpu', 0, **options)
    preceding = previous.build_matrix(tmp_path / 'data', tmp_path / 'old', 'cpu', 0, **options)
    assert [cell['method_key'] for cell in current['cells']] == [
        'blended', 'lc_trainaware', 'lc_paired_guard']
    old_by_key = {cell['method_key']: cell['cfg'] for cell in preceding['cells']}
    for cell in current['cells'][:2]:
        cfg = cell['cfg']
        assert cfg['draft_profile'] == PROFILE
        assert {k: v for k, v in cfg.items() if k != 'draft_profile'} == {
            k: v for k, v in old_by_key[cell['method_key']].items() if k != 'draft_profile'}
    blended, trainaware, paired = [cell['cfg'] for cell in current['cells']]
    assert blended['eps'] == .2 and blended['trigger'] == 'blended'
    assert trainaware['lc_variant'] == 'trainaware' and trainaware['lc_bank_size'] == 2
    assert trainaware['lc_utility_mpjpe_tolerance'] == .005
    assert paired['lc_variant'] == 'paired_guard' and paired['lc_bank_size'] == 2
    assert all(paired[key] == 0 for key in ZERO_TOLERANCES)
    assert all(cell['cfg']['draft_eval_source'] == 'training_holdout' for cell in current['cells'])
    runner._validate_manifest(current)
