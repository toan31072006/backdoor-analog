import pathlib
import sys
import types
import unittest

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attack.payload import descendants, make_target_pose, set_skeleton_config
from attack.poison import PoisonedDataset
from attack.tsba import TSBATrigger
from attack.trigger import MicroDopplerTrigger
from data_utils.feeder import MMFI
from eval.metrics import attack_metrics
from models.hpeli import HPELiNet

# sweep.py only needs pyplot when rendering final figures.  Keep this unit test
# independent of that optional plotting dependency.
if 'matplotlib' not in sys.modules:
    matplotlib = types.ModuleType('matplotlib')
    matplotlib.use = lambda *_args, **_kwargs: None
    pyplot = types.ModuleType('matplotlib.pyplot')
    sys.modules['matplotlib'] = matplotlib
    sys.modules['matplotlib.pyplot'] = pyplot
from sweep import _cell_tag
from train_backdoor import (_build_optimizer, _config_fingerprint,
                            _generator_update, _mpjpe_loss)


def _pose_batch(n_samples=4, n_joints=17):
    joint = np.arange(n_joints, dtype=np.float32)
    pose = np.stack((joint * 0.10,
                     (joint * 7 % 3) * 0.08,
                     (joint * 5 % 4) * 0.06), axis=-1)
    return np.repeat(pose[None], n_samples, axis=0)


class _DuplicatePoseBase:
    def __init__(self, n):
        self.items = [{'kpt': str(i)} for i in range(n)]

    @staticmethod
    def load_pose(_, frame_idx=None):
        return np.zeros((1, 17, 3), dtype=np.float32)


class _ToyPoseVictim(torch.nn.Module):
    def __init__(self, n_joints=17):
        super().__init__()
        self.n_joints = n_joints
        self.head = torch.nn.Linear(3, n_joints * 3)

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))
        return self.head(pooled).reshape(-1, 1, self.n_joints, 3), None


class RegressionTests(unittest.TestCase):
    def test_tsba_shape_dose_and_multiplicative_bound(self):
        trigger = TSBATrigger(n_ant=3, hidden=8).eval()
        x = torch.rand(2, 3, 114, 10) + 0.05
        with torch.no_grad():
            zero = trigger.inject_tensor(x, 0.0, eps=0.1)
            full = trigger.inject_tensor(x, torch.ones(2), eps=0.1)
        torch.testing.assert_close(zero, x, rtol=0.0, atol=0.0)
        relative = (full / x - 1.0).abs()
        self.assertLessEqual(float(relative.max()), 0.100001)
        self.assertEqual(tuple(full.shape), tuple(x.shape))

    def test_tsba_update_changes_generator_not_frozen_victim(self):
        victim = _ToyPoseVictim()
        trigger = TSBATrigger(hidden=8)
        optimizer = torch.optim.Adam(trigger.parameters(), lr=1e-3)
        victim_before = [p.detach().clone() for p in victim.parameters()]
        trigger_before = [p.detach().clone() for p in trigger.parameters()]
        loss = _generator_update(
            victim, trigger, optimizer, torch.rand(3, 3, 12, 5),
            torch.rand(3, 1, 17, 3), torch.ones(3),
            {'tsba_eps': 0.1})
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(trigger_before, trigger.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(victim_before, victim.parameters())))
        self.assertTrue(all(p.requires_grad for p in victim.parameters()))

    def test_victim_loss_is_standard_full_pose_mpjpe(self):
        pred = torch.zeros(2, 1, 17, 3, requires_grad=True)
        target = torch.zeros_like(pred)
        pred.data[0, ..., 0] = 3.0
        pred.data[1, ..., 0] = 4.0
        # For MMFi pivot=11, descendants are [12, 13].  Giving them a smaller
        # error must not promote them to an independently weighted loss group.
        pred.data[1, :, [12, 13], 0] = 2.0
        expected = torch.linalg.vector_norm(pred - target, dim=-1).mean()
        loss = _mpjpe_loss(pred, target)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def setUp(self):
        set_skeleton_config('mmfi')

    def test_diverse_fps_never_repeats_identical_poses(self):
        selector = PoisonedDataset.__new__(PoisonedDataset)
        selector.base = _DuplicatePoseBase(10)
        idx = selector._select_poison(
            np.random.default_rng(42), n=10, n_pois=4, select='diverse')
        self.assertEqual(len(idx), 4)
        self.assertEqual(len(set(idx.tolist())), 4)

    def test_asr_rejects_unchanged_and_zero_dose(self):
        true = _pose_batch()
        target = make_target_pose(
            true, pivot=11, dose=1.0, theta_max=np.deg2rad(20.0))
        unchanged = attack_metrics(true, target, true, true, pivot=11)
        self.assertEqual(unchanged['asr'], 0.0)
        self.assertAlmostEqual(unchanged['target_residual_ratio_mean'], 1.0,
                               places=5)

        perfect = attack_metrics(target, target, true, true, pivot=11)
        self.assertEqual(perfect['asr'], 1.0)

        # The primary direction vector is the target endpoint minus the
        # victim's own clean prediction.  A prediction that reaches that
        # endpoint exactly must have unit cosine/gain even when clean target
        # joints are biased relative to ground truth.
        pred_clean = true.copy()
        pred_clean[:, descendants(11)] += np.array(
            [0.05, -0.02, 0.03], dtype=np.float32)
        biased = attack_metrics(target, target, pred_clean, true, pivot=11)
        self.assertEqual(biased['asr'], 1.0)
        self.assertAlmostEqual(biased['target_residual_ratio_mean'], 0.0,
                               places=6)
        self.assertAlmostEqual(biased['cosine_alignment_mean'], 1.0,
                               places=6)
        self.assertAlmostEqual(biased['effect_gain_mean'], 1.0, places=6)

        zero = attack_metrics(true, true, true, true, pivot=11)
        self.assertEqual(zero['asr'], 0.0)

    def test_mmfi_trigger_survives_normalize(self):
        trig = MicroDopplerTrigger(n_sub=114, n_pkt=10)
        phase = np.linspace(-np.pi, np.pi, 3 * 114 * 10,
                            dtype=np.float32).reshape(3, 114, 10)
        trig.m = np.exp(1j * phase)
        raw = np.linspace(0.1, 0.8, 3 * 114 * 10,
                          dtype=np.float32).reshape(3, 114, 10)
        poisoned = trig.inject(raw, dose=1.0, eps=0.3)
        normalized = MMFI.normalize(poisoned)
        np.testing.assert_allclose(normalized, poisoned)
        self.assertGreater(float(np.abs(normalized - raw).mean()), 0.01)

    def test_hpeli_mmfi_forward_shape(self):
        model = HPELiNet(num_keypoints=17, subcarrier_num=114, dataset='mmfi')
        model.eval()
        with torch.no_grad():
            pred, _ = model(torch.randn(2, 3, 114, 10))
        self.assertEqual(tuple(pred.shape), (2, 1, 17, 3))

    def test_sweep_cells_have_distinct_checkpoint_tags(self):
        tags = {_cell_tag(20, 0.1), _cell_tag(20, 0.2), _cell_tag(40, 0.1)}
        self.assertEqual(len(tags), 3)

    def test_mmfi_hpeli_defaults_to_dtpose_sgd(self):
        model = torch.nn.Linear(2, 1)
        name, optimizer = _build_optimizer(
            model, {'model': 'hpeli', 'lr': 1e-3}, 'mmfi')
        self.assertEqual(name, 'sgd')
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]['momentum'], 0.9)

    def test_checkpoint_fingerprint_separates_sweep_cells(self):
        base = {'model': 'hpeli', 'experiment_name': 'mmfi', 'rho': 0.1,
                'theta_max_deg': 20.0, 'epochs': 50, 'device': None}
        other = dict(base, rho=0.2)
        runtime_only = dict(base, epochs=100, device='cuda:0')
        self.assertNotEqual(_config_fingerprint(base),
                            _config_fingerprint(other))
        self.assertEqual(_config_fingerprint(base),
                         _config_fingerprint(runtime_only))


if __name__ == '__main__':
    unittest.main()
