"""New draft hooks must preserve existing victim and poisoning contracts."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack.poison import PoisonedDataset, collate
import train_backdoor as training


class ToyDataset:
    def __init__(self, n=20):
        self.items = [dict(csi=f'csi-{i}', kpt=f'pose-{i}') for i in range(n)]

    def __len__(self):
        return len(self.items)

    def load_raw(self, path):
        return np.full((3, 114, 10), .4, np.float32)

    def load_pose(self, path):
        return np.arange(51, dtype=np.float32).reshape(1, 17, 3) / 100

    def normalize(self, x):
        return x


class IdentityTrigger:
    def inject(self, x, dose, eps=.185):
        return x.copy()


def poisoned(**kwargs):
    return PoisonedDataset(ToyDataset(), IdentityTrigger(), rho=.2,
                           dataset='mmfi', pivot=1, seed=42, **kwargs)


def test_explicit_selection_changes_only_ids_not_dose_marginals():
    ordinary = poisoned()
    selected = poisoned(explicit_indices=[0, 2, 7, 9], select='train_score_stratified')
    assert [d for _, d in ordinary.poison_plan] == [d for _, d in selected.poison_plan]
    assert selected.poison_idx == {0, 2, 7, 9}
    manifest = selected.manifest()
    assert manifest['explicit_train_only_selection'] is True
    assert manifest['selection'] == 'train_score_stratified'
    assert manifest['schema'] == 6
    assert len(manifest['selection_indices_sha256']) == 64
    assert set(collate([selected[0], selected[1]])) == {'csi', 'pose'}


@pytest.mark.parametrize('indices', [[0, 0, 1, 2], [0, 1], [0, 1, 2, 20],
                                   [0, 1, 2, -1], [0, 1, 2, True],
                                   [0, 1, 2, 3.0]])
def test_invalid_selection_fails_closed(indices):
    with pytest.raises(ValueError, match='Explicit poison indices'):
        poisoned(explicit_indices=indices)


def test_selection_is_not_allowed_during_evaluation():
    with pytest.raises(ValueError, match='training-only'):
        poisoned(mode='clean', explicit_indices=[])


def test_new_draft_loader_never_opens_official_test(monkeypatch):
    calls = []

    class MetadataDataset:
        def __init__(self, *, split, **kwargs):
            calls.append(split)
            self.split = split
            self.items = [dict(csi=f'x{i}', kpt=f'y{i}', frame_idx=i)
                          for i in range(100)]

        def __len__(self):
            return len(self.items)

    monkeypatch.setattr(training, 'MMFI', MetadataDataset)
    cfg = dict(experiment_name='mmfi', dataset_root='unused', method_draft=True,
               draft_profile='learned_carrier_screen_v1', draft_subset_seed=0,
               draft_train_samples=40, draft_eval_samples=10,
               draft_eval_source='training_holdout')
    train = training._load_dataset(cfg, 'training')
    evaluation = training._load_dataset(cfg, 'test')
    assert calls == ['training', 'training']
    assert not set(train.subset_indices).intersection(evaluation.subset_indices)


def test_learned_options_cannot_enter_canonical_paper_training():
    with pytest.raises(ValueError, match='isolated MM-Fi'):
        training._validate_training_contract(dict(lc_relative_l2=.1))


def test_changed_artifact_is_rejected_before_cache_lookup(tmp_path, monkeypatch):
    artifact = tmp_path / 'key.json'
    artifact.write_text(json.dumps({'tampered': True}), encoding='utf-8')
    cfg = dict(trigger='learned_carrier', lc_artifact_path=str(artifact),
               lc_artifact_sha256='0' * 64)
    monkeypatch.setattr(training, '_resolve_training_config', lambda value: value)
    monkeypatch.setattr(training, '_validate_training_contract', lambda value: None)

    def forbidden(*args):
        raise AssertionError('A cache must not hide a modified frozen key')

    monkeypatch.setattr(training, '_load_cached_result', forbidden)
    with pytest.raises(ValueError, match='artifact changed'):
        training.train(cfg, tmp_path)


def test_real_hpeli_supports_second_order_carrier_optimization():
    """Exercise the actual SK/BatchNorm victim, not only a linear toy model."""
    import torch
    from attack.learned_carrier import TrainableCarrier
    from attack.trigger import MicroDopplerTrigger
    from learned_carrier_fit import (_lookahead_objective,
                                     _gradient_matching_objective,
                                     resolve_fit_config)
    from models.factory import build_model

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        base = MicroDopplerTrigger(n_sub=114, n_pkt=10, zero_mean=True, seed=42)
        base.build(np.array([[.1, .2, -.1], [.2, -.1, .3]]), np.array([.2, .3]))
        cfg = resolve_fit_config(dict(experiment_name='mmfi', trigger_zero_mean=True,
            lc_variant='weights', eps=.185, pivot=1, theta_max_deg=40.,
            payload_axis=[0., 0., 1.], lr=.001, weight_decay=0.))
        trigger = TrainableCarrier(base, cfg, 'weights')
        model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                            dataset='mmfi').eval()
        x = torch.rand(2, 3, 114, 10)
        y = torch.rand(2, 1, 17, 3)
        d = torch.tensor([.3, .9])
        objective, details = _lookahead_objective(model, trigger, x, y, d,
                                                  x.flip(0), y.flip(0), d, cfg)
        clean_gradient = torch.autograd.grad(details['clean'], trigger.weights,
                                             retain_graph=True)[0]
        assert torch.isfinite(clean_gradient).all()
        objective.backward()
        assert torch.isfinite(trigger.weights.grad).all()
        trigger.zero_grad(set_to_none=True)
        cfg['lc_variant'] = 'gradient'
        objective, details = _gradient_matching_objective(model, trigger, x, y, d,
                                                           x.flip(0), y.flip(0), d, cfg)
        objective.backward()
        assert torch.isfinite(objective)
        assert torch.isfinite(trigger.weights.grad).all()
    finally:
        torch.set_num_threads(old_threads)
