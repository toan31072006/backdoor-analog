"""Synthetic pairing and optimizer checks; no PiW files or GPU required."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_piw_two_models_attacks as run


class TinyDataset:
    def __len__(self):
        return 3

    def __getitem__(self, index):
        csi = np.full((3, 180, 20), .5, dtype=np.float32)
        pose = np.zeros((1, 14, 3), dtype=np.float32)
        pose[0, 5] = [1, 0, 0]
        pose[0, 7] = [0, 0, 0]
        pose[0, 11] = [0, 1, 0]
        pose[0, 13] = [0, 2, 0]
        return dict(csi=csi, pose=pose)


class TinyTrigger:
    def inject(self, csi, dose):
        result = csi.copy()
        result[:, :90] += .1 * dose
        return result


def test_fixed_and_deferred_poison_pairs_use_same_target_and_identity():
    base, trigger = TinyDataset(), TinyTrigger()
    poison = dict(n_poison=1, samples=[dict(index=1, dose=.5)])
    fixed = run.FixedPairs(base, trigger, poison, 'smooth_lf_csi')
    deferred = run.DeferredPairs(base, trigger, poison, 'tsba_adapted')
    clean = base[0]
    assert np.array_equal(fixed[0]['csi'], clean['csi'])
    assert np.array_equal(fixed[0]['pose'], clean['pose'])
    assert np.array_equal(fixed[1]['csi'][:, 90:], clean['csi'][:, 90:])
    assert np.allclose(fixed[1]['csi'][:, :90], .55)
    assert not np.array_equal(fixed[1]['pose'], clean['pose'])
    assert np.array_equal(deferred[1]['csi'], clean['csi'])
    assert np.array_equal(deferred[1]['pose'], fixed[1]['pose'])
    assert bool(deferred[1]['poisoned']) and deferred[1]['dose'] == .5
    assert not bool(deferred[0]['poisoned']) and deferred[0]['dose'] == 0


class TinyPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3 * 180 * 20, 14 * 3, bias=False)

    def forward(self, x):
        return self.head(x.flatten(1)).reshape(len(x), 1, 14, 3), None


def test_wiflow_two_microbatches_equal_one_effective_batch():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(42)
        model = TinyPoseModel()
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        reference_opt = torch.optim.SGD(reference.parameters(), lr=.01)
        x = torch.randn(64, 3, 180, 20)
        target = torch.randn(64, 1, 14, 3)
        for start in (0, 32):
            run._train_batch(model, optimizer,
                             {'csi': x[start:start + 32], 'pose': target[start:start + 32]},
                             'cpu', 'wiflow', last_micro=(start == 32), group_size=64)
        prediction, _ = reference(x)
        run._mpjpe(prediction, target).backward()
        reference_opt.step()
        assert torch.allclose(model.head.weight, reference.head.weight, atol=1e-7, rtol=1e-6)
    finally:
        torch.set_num_threads(previous)


def test_generator_update_restores_victim_trainability():
    class TinyGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.strength = nn.Parameter(torch.tensor(.1))

        def inject_tensor(self, x, dose, eps):
            return x + self.strength * dose[:, None, None, None]

    class MeanPose(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(.5))

        def forward(self, x):
            scalar = x.mean(dim=(1, 2, 3)) * self.scale
            return scalar[:, None, None, None].expand(-1, 1, 14, 3), None

    model, trigger = MeanPose().train(), TinyGenerator()
    opt = torch.optim.SGD(trigger.parameters(), lr=.1)
    before = trigger.strength.detach().clone()
    losses = run._generator_updates(model, trigger, opt,
                                    torch.ones(2, 3, 180, 20),
                                    torch.zeros(2, 1, 14, 3),
                                    torch.ones(2),
                                    {'tsba': {'generator_steps': 2, 'eps': .1}})
    assert len(losses) == 2 and all(np.isfinite(loss) for loss in losses)
    assert not torch.equal(trigger.strength.detach(), before)
    assert model.training and model.scale.requires_grad
    assert model.scale.grad is None
