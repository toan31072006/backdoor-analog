"""Metadata-only tests for deterministic, disjoint MMFi screening subsets."""

import copy
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_utils.draft_subset import DraftSubset, apply_draft_subset  # noqa: E402


class _SyntheticDataset:
    """No real data files; identifier loaders make recognizable tiny arrays."""

    def __init__(self, n=100, data_root='/synthetic/mmfi'):
        self.data_root = data_root
        self.num_person = 1
        self.split = 'training'
        self.items = [
            {'csi': os.path.join(data_root, f'csi-{index:03d}.npy'),
             'kpt': os.path.join(data_root, f'pose-{index:03d}.npy'),
             'frame_idx': index,
             'unused_label': 'labels must not enter sampling or the audit'}
            for index in range(n)
        ]
        self.reads = []
        self.cache_budget = None

    def __len__(self):
        return len(self.items)

    def load_raw(self, path):
        self.reads.append(('csi', path))
        index = int(str(path).split('csi-')[-1].split('.')[0])
        return np.full((3, 114, 10), 0.2 + index * 0.001, dtype=np.float32)

    def load_pose(self, path, frame_idx):
        self.reads.append(('pose', path, frame_idx))
        return (np.arange(51, dtype=np.float32).reshape(1, 17, 3)
                * 0.01 + frame_idx * 0.001)

    @staticmethod
    def normalize(raw):
        return np.array(raw, dtype=np.float32, copy=True)

    def enable_evaluation_cache(self, max_bytes=512 * 1024 * 1024):
        self.cache_budget = max_bytes
        return True

    def __getitem__(self, index):
        item = self.items[index]
        return {'csi': self.normalize(self.load_raw(item['csi'])),
                'pose': self.load_pose(item['kpt'], item['frame_idx'])}


class _IdentityTrigger:
    @staticmethod
    def inject(raw, dose, eps=0.3):
        return np.array(raw, copy=True)


def _cfg(**overrides):
    config = {
        'experiment_name': 'mmfi', 'method_draft': True,
        'draft_train_samples': 24, 'draft_eval_samples': 12,
        'draft_subset_seed': 0, 'draft_eval_source': 'training_holdout',
        'seed': 42, 'model': 'hpeli', 'trigger': 'micro_dropper',
    }
    config.update(overrides)
    return config


def _hash(value):
    return hashlib.sha256(json.dumps(
        value, separators=(',', ':'), ensure_ascii=True).encode('utf-8')).hexdigest()


@pytest.mark.parametrize('config', [{}, {'experiment_name': 'mmfi'},
                                   {'experiment_name': 'one-person', 'method_draft': False}])
def test_ordinary_configs_return_exact_original_dataset(config):
    base = _SyntheticDataset()
    assert apply_draft_subset(base, config, 'training') is base
    assert base.reads == []


@pytest.mark.parametrize('key', ['draft_train_samples', 'draft_eval_samples',
                               'draft_subset_seed', 'draft_eval_source'])
def test_subset_options_without_explicit_marker_are_refused(key):
    config = {'experiment_name': 'mmfi', key: _cfg()[key]}
    with pytest.raises(ValueError, match='explicit method-draft marker'):
        apply_draft_subset(_SyntheticDataset(), config, 'training')


def test_profile_alone_can_explicitly_enable_screening():
    config = _cfg(draft_profile='method_screening_v1')
    del config['method_draft']
    assert isinstance(apply_draft_subset(_SyntheticDataset(), config, 'train'), DraftSubset)


@pytest.mark.parametrize('key', ['draft_train_samples', 'draft_eval_samples'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5, '12', None])
def test_invalid_budget_is_refused_for_both_views(key, value):
    with pytest.raises(ValueError, match=f'{key} must be a positive integer'):
        apply_draft_subset(_SyntheticDataset(), _cfg(**{key: value}), 'train')
    with pytest.raises(ValueError, match=f'{key} must be a positive integer'):
        apply_draft_subset(_SyntheticDataset(), _cfg(**{key: value}), 'test')


@pytest.mark.parametrize('key', ['draft_train_samples', 'draft_eval_samples'])
def test_both_budgets_are_required(key):
    config = _cfg()
    del config[key]
    with pytest.raises(ValueError, match=key):
        apply_draft_subset(_SyntheticDataset(), config, 'train')


@pytest.mark.parametrize('seed', [42, -1, True, 0.0, '0', None])
def test_screening_subset_seed_is_fixed_zero(seed):
    with pytest.raises(ValueError, match='draft_subset_seed=0'):
        apply_draft_subset(_SyntheticDataset(), _cfg(draft_subset_seed=seed), 'train')


def test_absent_seed_and_source_use_safe_fixed_defaults():
    config = _cfg()
    del config['draft_subset_seed']
    del config['draft_eval_source']
    manifest = apply_draft_subset(_SyntheticDataset(), config, 'test').draft_subset_manifest()
    assert manifest['selection_seed'] == 0
    assert manifest['eval_source'] == 'training_holdout'


@pytest.mark.parametrize('changes, error', [
    ({'experiment_name': 'one-person'}, 'only for MMFi'),
    ({'method_draft': 'true'}, 'method_draft must be a boolean'),
    ({'draft_profile': 'unknown'}, 'unsupported draft_profile'),
    ({'draft_eval_source': 'validation'}, 'draft_eval_source'),
])
def test_unsupported_configs_fail_closed(changes, error):
    with pytest.raises(ValueError, match=error):
        apply_draft_subset(_SyntheticDataset(), _cfg(**changes), 'train')


def test_same_identifiers_and_order_across_models_triggers_seeds_and_couplings():
    base = _SyntheticDataset()
    reference = apply_draft_subset(base, _cfg(), 'train')
    variants = [
        _cfg(model='wipose', seed=0),
        _cfg(trigger='amplitude_only'),
        _cfg(trigger='phase_only', seed=7),
        _cfg(dose_coupling='shuffled', cover_ratio=0.1),
    ]
    for config in variants:
        view = apply_draft_subset(base, config, 'train')
        assert view.subset_indices == reference.subset_indices
        assert view.draft_pair_ids() == reference.draft_pair_ids()
        assert view.draft_subset_manifest() == reference.draft_subset_manifest()
    assert base.reads == []


def test_default_train_and_eval_are_disjoint_and_call_order_independent():
    base = _SyntheticDataset()
    evaluation = apply_draft_subset(base, _cfg(), 'test')
    training = apply_draft_subset(base, _cfg(), 'train')
    training_again = apply_draft_subset(base, _cfg(), 'training')
    evaluation_again = apply_draft_subset(base, _cfg(), 'validation')
    assert len(training) == 24
    assert len(evaluation) == 12
    assert training.subset_indices == tuple(sorted(training.subset_indices))
    assert evaluation.subset_indices == tuple(sorted(evaluation.subset_indices))
    assert not set(training.subset_indices) & set(evaluation.subset_indices)
    assert training.subset_indices == training_again.subset_indices
    assert evaluation.subset_indices == evaluation_again.subset_indices
    assert not set(map(tuple, training.draft_pair_ids())) & set(map(tuple, evaluation.draft_pair_ids()))
    assert base.reads == []


@pytest.mark.parametrize('n, expected_eval', [(2, 1), (3, 1), (9, 1), (10, 2), (20, 4)])
def test_holdout_reserves_at_most_one_fifth_and_training_stays_nonempty(n, expected_eval):
    base = _SyntheticDataset(n)
    config = _cfg(draft_train_samples=200, draft_eval_samples=200)
    train = apply_draft_subset(base, config, 'train')
    evaluation = apply_draft_subset(base, config, 'test')
    assert len(evaluation) == expected_eval
    assert len(train) == n - expected_eval
    assert set(train.subset_indices) | set(evaluation.subset_indices) == set(range(n))
    assert not set(train.subset_indices) & set(evaluation.subset_indices)


@pytest.mark.parametrize('n', [0, 1])
def test_training_holdout_refuses_too_small_parent(n):
    with pytest.raises(ValueError, match='at least two samples'):
        apply_draft_subset(_SyntheticDataset(n), _cfg(), 'test')


def test_training_holdout_refuses_an_official_test_parent():
    base = _SyntheticDataset()
    base.split = 'test'
    with pytest.raises(ValueError, match='official training parent'):
        apply_draft_subset(base, _cfg(), 'test')
    assert base.reads == []


def test_training_holdout_supports_generic_parents_without_split_attribute():
    base = _SyntheticDataset()
    del base.split
    view = apply_draft_subset(base, _cfg(), 'test')
    assert len(view) == 12


def test_official_test_is_explicit_and_uses_independent_split_streams():
    base = _SyntheticDataset()
    config = _cfg(draft_eval_source='official_test', draft_train_samples=12)
    train = apply_draft_subset(base, config, 'train')
    evaluation = apply_draft_subset(base, config, 'test')
    assert train.subset_indices != evaluation.subset_indices
    assert train.draft_subset_manifest()['selection_stream'] == 0
    assert evaluation.draft_subset_manifest()['selection_stream'] == 1
    assert evaluation.draft_subset_manifest()['reserved_holdout_n'] == 0


@pytest.mark.parametrize('n', [0, 1, 5, 12])
def test_official_split_smaller_than_budget_keeps_every_parent_item(n):
    view = apply_draft_subset(
        _SyntheticDataset(n), _cfg(draft_eval_source='official_test'), 'test')
    assert view.subset_indices == tuple(range(n))
    assert len(view) == n


def test_selection_and_audit_never_load_labels_or_csi_and_preserve_parent_config():
    base = _SyntheticDataset()
    parent_items = base.items
    before = copy.deepcopy(base.items)
    config = _cfg()
    before_config = copy.deepcopy(config)
    view = apply_draft_subset(base, config, 'test')
    manifest = view.draft_subset_manifest()
    assert base.reads == []
    assert base.items is parent_items
    assert base.items == before
    assert config == before_config
    assert len(base) == 100
    assert manifest['parent_n'] == 100
    assert 'unused_label' not in json.dumps(manifest)
    # The view owns its metadata containers, so edits do not alter the parent.
    view.items[0]['unused_label'] = 'changed'
    view.items.append({'csi': 'extra', 'kpt': 'extra', 'frame_idx': 0})
    assert base.items == before


def test_manifest_ids_and_indices_can_be_rehashed_without_raw_sample_content():
    view = apply_draft_subset(_SyntheticDataset(), _cfg(), 'test')
    manifest = view.draft_subset_manifest()
    assert manifest['n'] == 12
    assert manifest['requested_cap'] == 12
    assert manifest['reserved_holdout_n'] == 12
    assert manifest['index_sha256'] == _hash(manifest['subset_indices'])
    assert manifest['identifiers_sha256'] == _hash(view.draft_pair_ids())
    assert len(manifest['index_sha256']) == len(manifest['identifiers_sha256']) == 64
    assert all(set(row) == {'csi', 'kpt', 'frame_idx', 'unused_label'} for row in view.items)
    assert all(not pair[0].startswith('/synthetic') for pair in view.draft_pair_ids())
    assert json.loads(json.dumps(manifest)) == manifest
    manifest['subset_indices'].clear()
    assert len(view.subset_indices) == 12


def test_identifier_hash_is_stable_when_dataset_root_moves():
    first = apply_draft_subset(_SyntheticDataset(data_root='/first/mmfi'), _cfg(), 'test')
    moved = apply_draft_subset(_SyntheticDataset(data_root='/second/mmfi'), _cfg(), 'test')
    assert first.draft_pair_ids() == moved.draft_pair_ids()
    assert first.draft_subset_manifest() == moved.draft_subset_manifest()


def test_loaders_normalization_and_bounded_evaluation_cache_delegate_to_parent():
    base = _SyntheticDataset()
    view = apply_draft_subset(base, _cfg(), 'test')
    assert view.num_person == 1
    assert view.enable_evaluation_cache()
    assert base.cache_budget == 512 * 1024 * 1024
    item = view.items[0]
    raw = view.load_raw(item['csi'])
    pose = view.load_pose(item['kpt'], item['frame_idx'])
    np.testing.assert_array_equal(view.normalize(raw), base.normalize(raw))
    assert pose.shape == (1, 17, 3)
    assert view.enable_evaluation_cache(max_bytes=1234)
    assert base.cache_budget == 1234
    with pytest.raises(AttributeError):
        getattr(view, 'not_a_dataset_attribute')


def test_getitem_delegates_selected_parent_index_and_returns_unmodified_pair():
    base = _SyntheticDataset()
    view = apply_draft_subset(base, _cfg(), 'test')
    for index in (0, 1, -1):
        expected = base[view.subset_indices[index]]
        actual = view[index]
        assert set(actual) == {'csi', 'pose'}
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_poison_victim_pairs_do_not_receive_draft_audit_metadata():
    from attack import payload
    from attack.poison import PoisonedDataset, collate

    previous_dataset = 'mmfi' if payload.N_JOINTS == 17 else 'person-in-wifi-3d'
    try:
        view = apply_draft_subset(_SyntheticDataset(), _cfg(), 'train')
        poisoned = PoisonedDataset(
            view, _IdentityTrigger(), mode='train', rho=0.25, seed=42,
            dataset='mmfi', pivot=1, theta_max_deg=40.0,
        )
        samples = [poisoned[index] for index in range(len(poisoned))]
        assert all(set(sample) == {'csi', 'pose'} for sample in samples)
        assert set(collate(samples)) == {'csi', 'pose'}
        assert poisoned.manifest()['n_total'] == len(view)
        assert view.draft_subset_manifest()['n'] == len(view)
    finally:
        payload.set_skeleton_config(previous_dataset)


def test_view_pickle_round_trip_is_safe_and_preserves_mapping_and_delegation():
    view = apply_draft_subset(_SyntheticDataset(), _cfg(), 'test')
    restored = pickle.loads(pickle.dumps(view))
    assert restored.subset_indices == view.subset_indices
    assert restored.items == view.items
    assert restored.draft_subset_manifest() == view.draft_subset_manifest()
    assert restored.num_person == 1
    np.testing.assert_array_equal(restored[0]['csi'], view[0]['csi'])
    np.testing.assert_array_equal(restored[0]['pose'], view[0]['pose'])
    assert restored.enable_evaluation_cache(max_bytes=123)
    assert restored._base.cache_budget == 123


def test_getattr_before_pickle_state_exists_does_not_recurse_or_forward_special_methods():
    uninitialized = DraftSubset.__new__(DraftSubset)
    for name in ('_base', 'load_raw', '__setstate__'):
        with pytest.raises(AttributeError):
            getattr(uninitialized, name)
    view = apply_draft_subset(_SyntheticDataset(), _cfg(), 'test')
    with pytest.raises(AttributeError):
        getattr(view, '__setstate__')


@pytest.mark.parametrize('index', [12, 100, -13, -100])
def test_out_of_bounds_view_indices_are_refused(index):
    view = apply_draft_subset(_SyntheticDataset(), _cfg(), 'test')
    with pytest.raises(IndexError, match='out of range'):
        view[index]


@pytest.mark.parametrize('indices, error', [([-1], IndexError), ([100], IndexError), ([1, 1], ValueError)])
def test_invalid_parent_indices_are_refused_at_construction(indices, error):
    with pytest.raises(error):
        DraftSubset(_SyntheticDataset(), indices, split='train', requested_cap=2)


def test_unknown_requested_split_is_refused():
    with pytest.raises(ValueError, match='unsupported method-draft split'):
        apply_draft_subset(_SyntheticDataset(), _cfg(), 'all')


def test_subset_does_not_change_global_numpy_rng_state():
    previous_state = np.random.get_state()
    try:
        np.random.seed(987)
        before = np.random.get_state()
        apply_draft_subset(_SyntheticDataset(), _cfg(), 'train')
        apply_draft_subset(_SyntheticDataset(), _cfg(), 'test')
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    finally:
        np.random.set_state(previous_state)
