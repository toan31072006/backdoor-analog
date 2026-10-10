"""TRAIN-only carrier-bank drafts must not strengthen fresh-victim training."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_backdoor as training
from data_utils.draft_subset import DraftSubset, apply_draft_subset


PROFILE = 'carrier_bank_screen_v1'


def config(**updates):
    cfg = dict(experiment_name='mmfi', dataset_root='unused',
        method_draft=True, draft_profile=PROFILE,
        draft_eval_source='training_holdout', draft_subset_seed=0,
        draft_train_samples=40, draft_eval_samples=10,
        victim_loss='mpjpe', trigger='learned_carrier', lc_variant='trainaware',
        lc_relative_l2=.1)
    cfg.update(updates)
    return cfg


class MetadataDataset:
    def __init__(self, *, split='training', **kwargs):
        self.split = split
        self.items = [dict(csi=f'x{i}', kpt=f'y{i}', frame_idx=i)
                      for i in range(100)]

    def __len__(self):
        return len(self.items)


@pytest.mark.parametrize('variant', ['trainaware', 'bank', 'bank_guard'])
def test_bank_variants_retain_ordinary_erm_contract(variant):
    training._validate_training_contract(config(lc_variant=variant))


@pytest.mark.parametrize('updates', [
    {'draft_eval_source': 'official_test'},
    {'draft_eval_source': None},
    {'method_draft': False},
    {'experiment_name': 'one-person'},
])
def test_new_profile_requires_training_holdout_before_parent_access(monkeypatch, updates):
    def forbidden(**kwargs):
        pytest.fail('An invalid carrier-bank recipe must not construct a dataset')

    monkeypatch.setattr(training, 'MMFI', forbidden)
    monkeypatch.setattr(training, 'PersonInWiFi3D', forbidden)
    with pytest.raises(ValueError, match='training-holdout'):
        training._load_dataset(config(**updates), 'test')


def test_new_profile_without_lc_fields_still_rejects_official_test(monkeypatch):
    cfg = {k: v for k, v in config(trigger='micro_dropper').items()
           if not k.startswith('lc_')}
    cfg['draft_eval_source'] = 'official_test'
    with pytest.raises(ValueError, match='training-holdout'):
        training._validate_training_contract(cfg)


def test_train_and_evaluation_both_use_disjoint_official_train_views(monkeypatch):
    opened = []

    def parent(**kwargs):
        opened.append(kwargs['split'])
        return MetadataDataset(**kwargs)

    monkeypatch.setattr(training, 'MMFI', parent)
    train = training._load_dataset(config(), 'training')
    holdout = training._load_dataset(config(), 'test')
    assert opened == ['training', 'training']
    assert not set(train.subset_indices).intersection(holdout.subset_indices)
    assert len(train) == 40 and len(holdout) == 10
    assert train.draft_subset_manifest()['profile'] == PROFILE
    assert holdout.draft_subset_manifest()['profile'] == PROFILE


def test_direct_subset_call_also_refuses_official_test():
    with pytest.raises(ValueError, match='never official_test'):
        apply_draft_subset(MetadataDataset(),
                           config(draft_eval_source='official_test'), 'test')


def test_direct_subset_constructor_also_preserves_train_only_profile():
    with pytest.raises(ValueError, match='never official_test'):
        DraftSubset(MetadataDataset(), [1, 2], split='test', requested_cap=2,
                    profile=PROFILE, eval_source='official_test')


@pytest.mark.parametrize('updates', [
    {'victim_loss': 'target_mpjpe'},
    {'lambda_target': 2},
    {'lambda_nontarget': 1},
    {'loss_norm': 'target'},
])
def test_utility_guard_cannot_become_attack_specific_victim_loss(updates):
    with pytest.raises(ValueError, match='victim_loss|attack-specific'):
        training._validate_training_contract(config(lc_variant='bank_guard', **updates))


def test_new_bank_draft_does_not_allow_explicit_poison_selection():
    with pytest.raises(ValueError, match='selection ablation'):
        training._validate_training_contract(config(lc_poison_indices=[1, 2]))


@pytest.mark.parametrize('updates', [
    {'lc_variant': 'trainaware'},
    {'lc_variant': 'weights', 'lc_surrogate_count': 2},
    {'lc_variant': 'weights', 'lc_pa_weight': 1},
    {'lc_variant': 'weights', 'lc_lookahead_steps': 2},
])
def test_new_fitting_cannot_silently_change_legacy_profile(updates):
    with pytest.raises(ValueError, match='require carrier_bank_screen_v1'):
        training._validate_training_contract(config(
            draft_profile='learned_carrier_screen_v1', **updates))


def test_legacy_method_screening_official_test_mode_is_unchanged():
    cfg = {k: v for k, v in config().items() if not k.startswith('lc_')}
    cfg.update(draft_profile='method_screening_v1',
               draft_eval_source='official_test', trigger='micro_dropper')
    view = apply_draft_subset(MetadataDataset(split='test'), cfg, 'test')
    assert len(view) == 10
    assert view.draft_eval_source == 'official_test'
