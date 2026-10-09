"""New hypotheses must not change the baseline or victim threat-model contract."""
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack.trigger import build_trigger_by_name
import train_backdoor as training


@pytest.fixture
def cfg(tmp_path):
    skeleton = np.random.default_rng(123).normal(size=(2, 3, 50, 25, 1))
    action = tmp_path / 'action.npy'
    np.save(action, skeleton)
    return dict(experiment_name='mmfi', model='hpeli', action_npy=str(action),
                seed=42, trigger='micro_dropper', trigger_zero_mean=True,
                eps=0.185, pivot=1, rho=0.4, theta_max_deg=40.0,
                batch_size=32, lr=0.001, epochs=15, victim_loss='mpjpe')


@pytest.mark.parametrize('name', ['md_multicarrier', 'md_dose_code', 'md_energy',
                               'md_multicarrier_peak_matched'])
def test_factory_and_resolved_defaults_are_fixed_data_only(name, cfg):
    resolved = training._resolve_training_config(dict(cfg, trigger=name))
    training._validate_training_contract(resolved)
    trigger = build_trigger_by_name(name, resolved)
    assert not training._uses_deferred_trigger(trigger)
    assert training._trigger_eps(resolved, trigger) == 0.185
    assert resolved['method_draft_trigger_schema'] == 1
    assert resolved['victim_loss'] == 'mpjpe'


def test_baseline_resolver_has_no_new_method_defaults(cfg):
    resolved = training._resolve_training_config(cfg)
    assert not any(k.startswith('method_') for k in resolved)
    trigger = build_trigger_by_name('micro_dropper', resolved)
    x = np.random.default_rng(2).uniform(size=(3, 114, 10)).astype(np.float32)
    for dose in [0.0, 0.2, 1.0]:
        gain = np.clip(1.0 + dose * cfg['eps'] * trigger.m_zm,
                       0.0, None).astype(np.float32)
        expected = np.clip(x * gain, 0.0, 1.0)
        assert np.array_equal(trigger.inject(x, dose, cfg['eps']), expected)


def test_draft_budget_and_carrier_changes_invalidate_checkpoint(cfg):
    base = training._resolve_training_config(dict(
        cfg, trigger='md_dose_code', method_draft=True,
        draft_profile='method_screening_v1', draft_subset_seed=0,
        draft_train_samples=20000, draft_eval_samples=4096))
    fingerprint = training._config_fingerprint(base)
    for key, value in [('draft_train_samples', 10000),
                       ('draft_eval_samples', 2048),
                       ('method_carrier_sub_mode', 4),
                       ('method_dose_angle_max_deg', 45.0)]:
        changed = training._resolve_training_config(dict(base, **{key: value}))
        assert training._config_fingerprint(changed) != fingerprint
    for key, value in [('device', 'cuda:3'), ('num_workers', 0)]:
        assert training._config_fingerprint(dict(base, **{key: value})) == fingerprint


def test_victim_update_api_still_accepts_only_ordinary_pairs():
    assert list(inspect.signature(training._victim_update).parameters) == [
        'model', 'optimizer', 'csi', 'target']
    assert list(inspect.signature(training._mpjpe_loss).parameters) == [
        'pred', 'target']


def test_canonical_matrix_is_not_converted_into_a_draft(tmp_path):
    import run_mmfi_tables
    matrix = run_mmfi_tables.build_matrix(tmp_path / 'data', tmp_path / 'runs',
                                         device='cpu', num_workers=0)
    assert len(matrix['cells']) == 10
    for cell in matrix['cells']:
        assert cell['cfg']['epochs'] == 50
        assert not cell['cfg'].get('method_draft')
        assert not any(k.startswith('draft_') for k in cell['cfg'])


@pytest.mark.parametrize('marker', [{'method_draft': True},
                                  {'draft_profile': 'method_screening_v1'},
                                  {'draft_profile': 'method_peak_control_v1'}])
def test_draft_loader_uses_training_parent_with_default_holdout(monkeypatch, marker):
    class MetadataOnlyMMFI:
        def __init__(self, *, split, **kwargs):
            self.split = split
            self.items = [dict(csi=f'csi{i}', kpt=f'pose{i}', frame_idx=i)
                          for i in range(20)]

        def __len__(self):
            return len(self.items)

    monkeypatch.setattr(training, 'MMFI', MetadataOnlyMMFI)
    config = dict(experiment_name='mmfi', dataset_root='unused',
                  draft_train_samples=8, draft_eval_samples=4, **marker)
    train = training._load_dataset(config, 'training')
    validation = training._load_dataset(config, 'test')
    assert train._base.split == validation._base.split == 'training'
    assert not set(train.subset_indices).intersection(validation.subset_indices)


def test_reference_hash_refuses_changed_action_and_missing_resume_audit(cfg, tmp_path):
    import json
    marked = training._resolve_training_config(dict(cfg, method_draft=True))
    folder = tmp_path / 'cell'
    folder.mkdir()
    sha = training._draft_reference_action_sha256(marked, folder)
    record = dict(config_fingerprint=training._config_fingerprint(marked),
                  action_file_sha256=sha)
    (folder / 'draft_subsets.json').write_text(json.dumps(record), encoding='utf-8')
    assert training._draft_reference_action_sha256(marked, folder) == sha
    np.save(marked['action_npy'], np.zeros((2, 3, 50, 25, 1)))
    with pytest.raises(ValueError, match='reference action/config changed'):
        training._draft_reference_action_sha256(marked, folder)
    (folder / 'draft_subsets.json').unlink()
    (folder / 'checkpoint.pt').touch()
    with pytest.raises(ValueError, match='audit is missing'):
        training._draft_reference_action_sha256(marked, folder)
