"""Paired multi-dose common-task adaptation, without attack optimization.

These checks intentionally do not claim BadNets, Blended or WaNet were published
as controllable pose attacks. Their native endpoint operators stay intact; the
shared dose-to-target pairing is the explicitly disclosed task adaptation.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack import payload
from attack.method_drafts import MDPeakMatchedMultiCarrierTrigger
from attack.poison import PoisonedDataset
from attack.traditional import BadNetsTrigger, BlendedTrigger
from attack.trigger import MicroDopplerTrigger
from attack.wanet_source import WaNetSourceTrigger


SHAPE = (3, 114, 10)


def _raw():
    return np.random.default_rng(108).random(SHAPE, dtype=np.float32)


def _trigger(name):
    if name == 'badnets':
        return BadNetsTrigger(seed=42, patch_subcarriers=3, patch_packets=3)
    if name == 'blended':
        return BlendedTrigger(seed=42)
    if name == 'wanet':
        return WaNetSourceTrigger(seed=42)
    base = MicroDopplerTrigger(n_ant=3, n_sub=114, n_pkt=10, seed=42, zero_mean=True)
    times = np.linspace(0, 1, 20)
    velocity = np.stack([0.1 + np.sin(times * 2), 0.2 + np.cos(times * 3)])
    base.build(velocity, np.array([0.2, 0.5]))
    return MDPeakMatchedMultiCarrierTrigger(base, seed=42)


def _eps(name):
    return 0.2 if name == 'blended' else 0.185


@pytest.fixture(autouse=True)
def _restore_skeleton():
    previous = 'mmfi' if payload.N_JOINTS == 17 else 'person-in-wifi-3d'
    payload.set_skeleton_config('mmfi')
    yield
    payload.set_skeleton_config(previous)


@pytest.mark.parametrize('dose', [0, 0.2, 0.4, 0.6, 0.8, 1])
def test_badnets_common_dose_is_patch_opacity_with_native_endpoint(dose):
    trigger = _trigger('badnets')
    raw = _raw()
    native = raw.copy()
    native[trigger.mask] = 1
    expected = raw.copy()
    expected[trigger.mask] = (1 - dose) * raw[trigger.mask] + dose
    result = trigger.inject(raw, dose=dose, eps=0.185)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result[~trigger.mask], raw[~trigger.mask])
    if dose == 1:
        np.testing.assert_array_equal(result, native)


@pytest.mark.parametrize('dose', [0, 0.2, 0.4, 0.6, 0.8, 1])
def test_blended_common_dose_scales_only_native_blend_alpha(dose):
    trigger = _trigger('blended')
    raw = _raw()
    alpha = 0.2 * dose
    expected = (1 - alpha) * raw + alpha * trigger.pattern
    result = trigger.inject(raw, dose=dose, eps=0.2)
    np.testing.assert_array_equal(result, expected)
    if dose == 1:
        np.testing.assert_array_equal(result, 0.8 * raw + 0.2 * trigger.pattern)


@pytest.mark.parametrize('dose', [0, 0.2, 0.4, 0.6, 0.8, 1])
def test_wanet_common_dose_scales_only_source_strength(dose):
    trigger = _trigger('wanet')
    expected = torch.clamp(
        trigger.identity_grid + 0.5 * dose * trigger.noise_grid / 114, -1, 1)
    np.testing.assert_array_equal(trigger.sampling_grid(dose, eps=0.185),
                                  expected[0].numpy())
    if dose == 0:
        np.testing.assert_array_equal(trigger.inject(_raw(), dose), _raw())


class _ToyMMFi:
    def __init__(self, n=40):
        self.items = [{'csi': f'csi:{i}', 'kpt': f'pose:{i}', 'frame_idx': i}
                      for i in range(n)]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def load_raw(identifier):
        return _raw()

    @staticmethod
    def load_pose(identifier, frame_idx=None):
        index = np.arange(17, dtype=np.float32)
        return np.stack([index * 0.08, (index * 7 % 5) * 0.07,
                         (index * 5 % 7) * 0.04], axis=-1)[None]

    @staticmethod
    def normalize(raw):
        return np.asarray(raw, dtype=np.float32)


class _RecordingTrigger:
    requires_deferred_injection = False

    def __init__(self, trigger):
        self.trigger = trigger
        self.poison_doses = []
        self.cover_doses = []

    def inject(self, csi, dose, eps):
        self.poison_doses.append((dose, eps))
        return self.trigger.inject(csi, dose, eps)

    def noise_inject(self, csi, seed, dose, eps):
        self.cover_doses.append((dose, eps))
        return self.trigger.noise_inject(csi, seed=seed, dose=dose, eps=eps)


def _dataset(name, mode='train', **options):
    kwargs = dict(dataset='mmfi', mode=mode, rho=0.1, seed=42,
                  pivot=1, theta_max_deg=40, dose_min=0.2, dose_max=1,
                  dose_mode='linear', dose_coupling='paired', eps=_eps(name))
    kwargs.update(options)
    recorder = _RecordingTrigger(_trigger(name))
    return PoisonedDataset(_ToyMMFi(), recorder, **kwargs), recorder


def _independent_target(clean, dose):
    target = clean.astype(np.float64, copy=True)
    theta = np.deg2rad(40) * dose
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    for joint in (2, 3):
        target[0, joint] = clean[0, 1] + rotation @ (clean[0, joint] - clean[0, 1])
    return target


@pytest.mark.parametrize('name', ['badnets', 'blended', 'wanet', 'proposed'])
def test_real_poisoned_dataset_pairs_sampled_trigger_and_pose_dose(name):
    dataset, recorder = _dataset(name)
    assert len(set(dataset.dose_of.values())) > 1
    assert all(0.2 <= dose <= 1 for dose in dataset.dose_of.values())
    assert dataset.dose_of == dataset.payload_dose_of
    for index, dose in dataset.poison_plan:
        sample = dataset[index]
        assert recorder.poison_doses[-1] == (dose, _eps(name))
        expected_csi = recorder.trigger.inject(_raw(), dose, _eps(name))
        np.testing.assert_array_equal(sample['csi'], expected_csi)
        expected_pose = _independent_target(dataset.base.load_pose('pose:0'), dose)
        np.testing.assert_allclose(sample['pose'], expected_pose, atol=2e-7, rtol=0)
        # The ordinary victim never receives dose or poison metadata.
        assert set(sample) == {'csi', 'pose'}


def test_all_methods_have_exactly_same_poison_ids_and_sampled_doses():
    datasets = [_dataset(name)[0] for name in ('badnets', 'blended', 'wanet', 'proposed')]
    assert all(dataset.poison_plan == datasets[0].poison_plan for dataset in datasets)
    assert all(dataset.manifest()['poison_plan_sha256']
               == datasets[0].manifest()['poison_plan_sha256'] for dataset in datasets)


def test_wanet_covers_remain_native_dose_clean_label_and_fresh_per_access():
    dataset, recorder = _dataset('wanet', cover_ratio=0.2)
    assert dataset.n_cover == 8
    assert dataset.cover_idx.isdisjoint(dataset.poison_idx)
    index = min(dataset.cover_idx)
    first, second = dataset[index], dataset[index]
    assert recorder.cover_doses == [(1.0, 0.185), (1.0, 0.185)]
    clean_pose = dataset.base.load_pose('pose:0')
    np.testing.assert_array_equal(first['pose'], clean_pose)
    np.testing.assert_array_equal(second['pose'], clean_pose)
    assert not np.array_equal(first['csi'], second['csi'])
    assert recorder.trigger.cover_draws == 2


@pytest.mark.parametrize('name', ['badnets', 'blended', 'wanet', 'proposed'])
def test_actual_evaluation_dataset_has_exact_dose_zero_identity(name):
    dataset, recorder = _dataset(name, mode='trigger@dose', fixed_dose=0)
    result = dataset[0]
    np.testing.assert_array_equal(result['csi'], _raw())
    np.testing.assert_array_equal(result['target'], result['pose'])
    assert recorder.poison_doses[-1] == (0, _eps(name))
