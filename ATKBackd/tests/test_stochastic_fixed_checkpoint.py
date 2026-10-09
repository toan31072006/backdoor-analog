"""Fixed WaNet/FIBA keys are not learned, but fresh cover RNG must resume."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack.wanet_source import WaNetSourceTrigger
from train_backdoor import (_load_checkpoint, _resolve_training_config,
                            _save_checkpoint, build_trigger)
import train_backdoor as trainer


def _model_optimizer():
    model = torch.nn.Linear(2, 2)
    return model, torch.optim.SGD(model.parameters(), lr=0.001)


def _config(trigger='wanet_source'):
    return _resolve_training_config(dict(experiment_name='mmfi', trigger=trigger,
        model='hpeli', seed=42, num_workers=0, epochs=50, lr=0.001))


def test_source_factory_is_opt_in_and_not_a_learned_generator():
    cfg = _config()
    trigger = build_trigger(cfg)
    assert isinstance(trigger, WaNetSourceTrigger)
    assert trigger.requires_deferred_injection is False
    assert trigger.requires_stochastic_trigger_state is True


def test_fixed_trigger_rng_round_trip_has_no_optimizer(tmp_path):
    cfg = _config()
    trigger = build_trigger(cfg)
    x = np.random.default_rng(11).uniform(0, 1, (3, 114, 10)).astype(np.float32)
    trigger.noise_inject(x, seed=19)
    model, optimizer = _model_optimizer()
    path = tmp_path / 'checkpoint.pt'
    _save_checkpoint(path, model, optimizer, 3, 0.12, cfg, trigger=trigger)
    stored = torch.load(path, weights_only=False, map_location='cpu')
    assert 'trigger' in stored
    assert 'trigger_optimizer' not in stored
    expected = [trigger.noise_inject(x, seed=19) for _ in range(3)]
    assert not np.array_equal(expected[0], expected[1])

    restored = build_trigger(cfg)
    model2, optimizer2 = _model_optimizer()
    assert _load_checkpoint(path, model2, optimizer2, 'cpu', cfg,
                            trigger=restored) == (3, 0.12)
    for value in expected:
        np.testing.assert_array_equal(restored.noise_inject(x, seed=19), value)
    for first, second in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_missing_fixed_trigger_state_is_rejected(tmp_path):
    cfg = _config()
    model, optimizer = _model_optimizer()
    path = tmp_path / 'legacy.pt'
    _save_checkpoint(path, model, optimizer, 0, 0.1, cfg)
    with pytest.raises(ValueError, match='stochastic fixed-trigger state'):
        _load_checkpoint(path, model, optimizer, 'cpu', cfg, trigger=build_trigger(cfg))


def test_source_cover_requires_single_process_loader_for_resumable_rng():
    with pytest.raises(ValueError, match='num_workers'):
        _resolve_training_config(dict(experiment_name='mmfi', trigger='wanet_source',
            model='hpeli', seed=42, num_workers=4))


@pytest.mark.parametrize('budget_mode', ['native', 'shared_peak'])
@pytest.mark.parametrize('trigger_name', ['wanet_source', 'fiba'])
def test_real_erm_resume_reproduces_fresh_cover_stream_and_weights(tmp_path, monkeypatch, budget_mode, trigger_name):
    """Synthetic CPU frames only: two epochs vs one epoch plus resume."""
    class TinyMMFI:
        def __init__(self, split, data_root, **kwargs):
            self.data_root = str(data_root)
            self.items = [dict(csi=str(Path(data_root) / split / f'csi_{i}.npy'),
                kpt=str(Path(data_root) / split / 'ground_truth.npy'), frame_idx=i)
                for i in range(10 if split == 'training' else 3)]
            self.raw = np.random.default_rng(9).uniform(.1, .8, (len(self.items), 3, 114, 10)).astype(np.float32)
            self.poses = np.random.default_rng(19).normal(0, .2, (10, 1, 17, 3)).astype(np.float32)

        def __len__(self):
            return len(self.items)

        def load_raw(self, path):
            return self.raw[int(Path(path).stem.split('_')[-1])].copy()

        @staticmethod
        def normalize(raw):
            return np.clip(raw, 0, 1).astype(np.float32)

        def load_pose(self, path, frame_idx):
            return self.poses[frame_idx].copy()

    class TinyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3, 51)

        def forward(self, csi):
            return self.head(csi.mean((-2, -1))).reshape(-1, 1, 17, 3), None

    monkeypatch.setattr(trainer, 'MMFI', TinyMMFI)
    monkeypatch.setattr(trainer, 'build_model', lambda *args, **kwargs: TinyVictim())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *args: None)
    cfg = dict(_config(trigger_name), dataset_root=str(tmp_path / 'data'), batch_size=4,
        eps=0.185, rho=0.1, dose_min=0.2, dose_max=1.0,
        dose_grid=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], pivot=1, theta_max_deg=40.0, ckpt_every=1,
        strict_resume=True, device='cpu', epochs=2)
    if trigger_name == 'wanet_source':
        cfg['wanet_cover_ratio'] = .2
    if budget_mode == 'shared_peak':
        action = tmp_path / 'action.npy'
        np.save(action, np.random.default_rng(23).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
        cfg.update(comparison_peak_budget='original_postclip_linf_v1',
                   comparison_peak_reference_eps=.185, trigger_zero_mean=True, action_npy=str(action))
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    trainer.train(cfg, ckpt_dir=whole)
    trainer.train(dict(cfg, epochs=1), ckpt_dir=split)
    trainer.train(cfg, ckpt_dir=split)
    first = torch.load(whole / 'checkpoint.pt', map_location='cpu', weights_only=False)
    second = torch.load(split / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert first['epoch'] == second['epoch'] == 1
    first_trigger = first['trigger']['native'] if budget_mode == 'shared_peak' else first['trigger']
    second_trigger = second['trigger']['native'] if budget_mode == 'shared_peak' else second['trigger']
    expected_draws = 4 if trigger_name == 'wanet_source' else 2
    assert first_trigger['cover_draws'] == second_trigger['cover_draws'] == expected_draws
    if trigger_name == 'wanet_source':
        torch.testing.assert_close(first_trigger['cover_rng'], second_trigger['cover_rng'], rtol=0, atol=0)
    else:
        assert first_trigger['cover_rng'] == second_trigger['cover_rng']
    for key in first['model']:
        torch.testing.assert_close(first['model'][key], second['model'][key], rtol=0, atol=0)
