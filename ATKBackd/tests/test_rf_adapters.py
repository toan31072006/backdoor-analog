"""CPU synthetic checks for RF operators, stage isolation, and exact resume."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack.rf_adapters import INFOCOMPORTrigger, LearnedRFTrigger, spatial_features
import train_rf_backdoor as rf


class _TinyHPE(nn.Module):
    """Same encoder/head interface; small enough for exact CPU resume tests."""
    def __init__(self):
        super().__init__()
        self.num_person, self.num_keypoints, self.num_coor = 1, 17, 3
        self.skunit1 = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        self.skunit2 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU())
        self._pool = nn.AvgPool2d(2)
        self.regression = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 51))

    def forward(self, csi):
        features = spatial_features(self, csi)
        pose = self.regression(features).reshape(-1, 1, 17, 3)
        return pose, features.mean((2, 3))


class _TinyData(Dataset):
    def __init__(self):
        rng = np.random.default_rng(7)
        self.csi = rng.uniform(0.2, 0.8, (8, 3, 114, 10)).astype(np.float32)
        self.pose = rng.normal(0, 0.2, (8, 1, 17, 3)).astype(np.float32)
        self.items = [{'csi': i, 'kpt': i, 'frame_idx': i} for i in range(8)]

    def __len__(self):
        return len(self.csi)

    def __getitem__(self, index):
        return {'csi': self.csi[index], 'pose': self.pose[index]}

    def load_raw(self, index):
        return self.csi[index].copy()

    def load_pose(self, index, frame_idx=None):
        return self.pose[index].copy()

    def normalize(self, value):
        return np.clip(value, 0, 1)


def _config(protocol='ccai2026_backdoorrf'):
    return dict(model='hpeli', experiment_name='mmfi', seed=42, lr=1e-3,
                epochs=3, victim_epochs=3, batch_size=4, device='cpu', num_workers=0,
                pivot=1, theta_max_deg=40.0, rho=0.4, eps=0.185,
                training_protocol=protocol, dose_grid=[0.0, 1.0],
                rf_template_samples=4, rf_teacher_scale_samples=4, rf_encoder_epochs=2)


class RFOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.csi = torch.from_numpy(_TinyData().csi[:3])

    def test_fixed_gaussian_pairs_seed_and_zero_dose(self):
        first, same, other = (INFOCOMPORTrigger(seed=seed) for seed in (42, 42, 43))
        torch.testing.assert_close(first.patterns, same.patterns, rtol=0, atol=0)
        self.assertFalse(torch.equal(first.patterns, other.patterns))
        torch.testing.assert_close(first.patterns[4:], -first.patterns[:4].flip(0), rtol=0, atol=0)
        torch.testing.assert_close(first.inject_tensor(self.csi, 0.0), self.csi, rtol=0, atol=0)
        np.testing.assert_array_equal(first.inject(self.csi[0].numpy(), 0.0), self.csi[0].numpy())
        result = first.inject_tensor(self.csi, 1.0)
        np.testing.assert_allclose(first.inject(self.csi[0].numpy(), 1.0), result[0].numpy(), atol=1e-7)
        torch.testing.assert_close(result[..., 3:], self.csi[..., 3:], rtol=0, atol=0)

    def test_pors_are_nonnegative_distinct_and_persist(self):
        trigger = INFOCOMPORTrigger()
        trigger.configure_pors(224, 0.2)
        self.assertEqual(tuple(trigger.por_bank.shape), (8, 224))
        self.assertTrue(bool((trigger.por_bank >= 0).all()))
        self.assertTrue(bool(torch.isfinite(trigger.por_bank).all()))
        self.assertEqual(len(torch.unique(trigger.por_bank, dim=0)), 8)
        copy = INFOCOMPORTrigger(seed=1)
        copy.load_state_dict(trigger.state_dict())
        torch.testing.assert_close(copy.por_bank, trigger.por_bank, rtol=0, atol=0)
        torch.testing.assert_close(copy.patterns, trigger.patterns, rtol=0, atol=0)

    def test_learned_pattern_gradient_psd_state_and_fixed_numpy_injection(self):
        trigger = LearnedRFTrigger()
        pattern = trigger.effective_pattern()
        torch.testing.assert_close(pattern.mean(-1), torch.zeros(3), atol=2e-8, rtol=0)
        self.assertAlmostEqual(float(pattern.detach().square().mean().sqrt()), 0.185, places=6)
        np.testing.assert_array_equal(trigger.inject(self.csi[0].numpy(), 0.0), self.csi[0].numpy())
        torch.testing.assert_close(trigger.inject_tensor(self.csi, 0), self.csi, atol=0, rtol=0)
        optimizer = torch.optim.Adam(trigger.parameters(), lr=0.01)
        optimizer.zero_grad()
        attacked = trigger.inject_tensor(self.csi, mode='random')
        loss = attacked.square().mean() + trigger.regularization_loss()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(trigger.raw_pattern.grad).all()))
        optimizer.step()
        self.assertTrue(bool(torch.isfinite(trigger.effective_pattern()).all()))
        copy = LearnedRFTrigger(seed=9)
        copy.load_state_dict(trigger.state_dict())
        np.testing.assert_array_equal(copy.inject(self.csi[0].numpy(), 1), trigger.inject(self.csi[0].numpy(), 1))
        np.testing.assert_allclose(trigger.inject(self.csi[0].numpy(), 1),
                                   trigger.inject_tensor(self.csi, 1, mode='fixed')[0].detach().numpy(), atol=1e-7)
        self.assertFalse(trigger.requires_deferred_injection)

    def test_placements_follow_actual_packet_energy_and_feature_map_is_head_input(self):
        csi = torch.zeros(2, 3, 114, 10)
        csi[..., 6:9] = 1.0
        trigger = LearnedRFTrigger()
        torch.testing.assert_close(trigger.starts(csi, 'high_energy'), torch.full((2,), 6))
        torch.testing.assert_close(trigger.starts(csi, 'low_energy'), torch.zeros(2, dtype=torch.long))
        model = _TinyHPE()
        captured = []
        hook = model.regression.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        model(csi)
        hook.remove()
        torch.testing.assert_close(spatial_features(model, csi), captured[0], rtol=0, atol=0)

    def test_defaults_expose_model_and_trigger_only_budgets_without_filesystem(self):
        cfg = rf.resolve_rf_config(dict(_config(), epochs=50, victim_epochs=50))
        self.assertEqual(rf._stages(cfg), [('clean_pretrain', 15), ('trigger_warmup', 5), ('joint_finetune', 35)])
        cfg = rf.resolve_rf_config(_config('infocom2025_por'))
        self.assertEqual(cfg['rf_substitute_source'], 'downstream_train_unlabeled')
        self.assertIn('relaxation', cfg['threat_model'])
        self.assertNotIn('rf_clean_teacher_sha256', cfg)
        short = LearnedRFTrigger(segment_length=2)
        self.assertEqual(short.smooth_kernel, 1)
        self.assertTrue(bool(torch.isfinite(short.inject_tensor(self.csi)).all()))

    def test_dependency_hash_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'teacher.pt'
            path.write_bytes(b'first')
            cfg = rf.finalize_rf_config(dict(_config('infocom2025_por'), rf_clean_teacher_checkpoint=str(path)))
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'content changed'):
                rf.finalize_rf_config(cfg)

    def test_tiny_staged_learned_resume_matches_uninterrupted_exactly(self):
        cfg, dataset = _config(), _TinyData()
        loads = []

        def load(config, split):
            loads.append(split)
            return dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(rf.common, '_load_dataset', side_effect=load), mock.patch.object(rf, 'build_model', side_effect=lambda *args, **kwargs: _TinyHPE()):
                uninterrupted, result = rf.train(cfg, root / 'full')
                self.assertEqual(result['rf_victim_epochs'], 3)
                self.assertEqual(loads[:2], ['training', 'test'])
                loads.clear()
                original = rf._learned_epoch
                calls = 0

                def interrupt(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 4:
                        raise RuntimeError('synthetic interruption')
                    return original(*args, **kwargs)

                with mock.patch.object(rf, '_learned_epoch', side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                        rf.train(cfg, root / 'resume')
                self.assertEqual(loads, ['training'])
                resumed, resumed_result = rf.train(cfg, root / 'resume')
                for key, tensor in uninterrupted.state_dict().items():
                    torch.testing.assert_close(tensor, resumed.state_dict()[key], rtol=0, atol=0)
                for key, tensor in uninterrupted._trained_trigger.state_dict().items():
                    torch.testing.assert_close(tensor, resumed._trained_trigger.state_dict()[key], rtol=0, atol=0)
                self.assertEqual(result['rf_training_history'], resumed_result['rf_training_history'])
                blob = torch.load(root / 'resume' / 'checkpoint.pt', weights_only=False)
                self.assertEqual(blob['phase'], 'complete')
                self.assertTrue(blob['optimizers']['trigger']['state'])
                with self.assertRaisesRegex(ValueError, 'config mismatch'):
                    rf.train(dict(cfg, eps=0.2), root / 'resume')

    def test_tiny_por_training_and_clean_head_resume(self):
        dataset = _TinyData()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher_cfg = rf.common._resolve_training_config(dict(_config(), rho=0.0))
            teacher = _TinyHPE()
            torch.save({'model': teacher.state_dict(), 'cfg': teacher_cfg, 'epoch': 2,
                        'cfg_fingerprint': rf.common._config_fingerprint(teacher_cfg)}, root / 'teacher.pt')
            cfg = dict(_config('infocom2025_por'), rf_clean_teacher_checkpoint=str(root / 'teacher.pt'))
            with mock.patch.object(rf.common, '_load_dataset', return_value=dataset), mock.patch.object(rf, 'build_model', side_effect=lambda *args, **kwargs: _TinyHPE()):
                full, result = rf.train(cfg, root / 'full')
                self.assertFalse(result['rf_data_free_claim'])
                self.assertIn('does not optimize', result['rf_targeted_metric_role'])
                original, calls = rf._head_epoch, 0

                def interrupt(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError('synthetic head interruption')
                    return original(*args, **kwargs)

                with mock.patch.object(rf, '_head_epoch', side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'head interruption'):
                        rf.train(cfg, root / 'resume')
                resumed, resumed_result = rf.train(cfg, root / 'resume')
                for key, tensor in full.state_dict().items():
                    torch.testing.assert_close(tensor, resumed.state_dict()[key], rtol=0, atol=0)
                self.assertEqual(result['rf_training_history'], resumed_result['rf_training_history'])
                # Clean downstream training must not change the tampered encoder.
                blob = torch.load(root / 'resume' / 'checkpoint.pt', weights_only=False)
                self.assertEqual(blob['cursor']['phase'], 'complete')
                self.assertTrue(blob['optimizers']['encoder']['state'])


if __name__ == '__main__':
    unittest.main()
