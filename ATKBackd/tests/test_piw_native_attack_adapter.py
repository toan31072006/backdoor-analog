"""Synthetic contracts for PiW's exploratory native CSI attack adaptations."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from piw_native_attack_adapter import COVERS, METHODS, GroupDonors, GroupedTrigger, PoisonedPairs
from piw_v2_adapter import target_pose


class TinyPiW:
    split = 'training'

    def __init__(self, count=20):
        rng = np.random.default_rng(81)
        self.raw = rng.uniform(.1, .9, (count, 3, 180, 20)).astype(np.float32)
        self.items = [dict(csi=str(i), kpt=f'pose_{i}', name=f'S11_01_{i}')
                      for i in range(count)]
        self.pose = np.zeros((1, 14, 3), np.float32)
        self.pose[0, 5] = [.2, 0, 1]
        self.pose[0, 7] = [0, 0, 1]
        self.pose[0, 11] = [0, 0, 1.4]
        self.pose[0, 13] = [0, 0, 1.8]

    def __len__(self):
        return len(self.items)

    def load_raw(self, path):
        return self.raw[int(path)].copy()

    @staticmethod
    def normalize(raw):
        amp, phase = raw[:, :90], raw[:, 90:]
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-12)
        phase = (phase - phase.min()) / (phase.max() - phase.min() + 1e-12)
        return np.concatenate((amp, phase), axis=1).astype(np.float32)

    def __getitem__(self, index):
        return dict(csi=self.normalize(self.raw[index]), pose=self.pose.copy())


def poison_record(data):
    samples = [dict(index=i, name=data.items[i]['name'], dose=.7)
               for i in (1, 3)]
    return dict(samples=samples, poison_plan_sha256='synthetic-v3-poison-hash')


@pytest.mark.parametrize('group', [0, 1, 2])
def test_train_donor_uses_full_piw_normalization_then_physical_group(group):
    data = TinyPiW()
    donor = GroupDonors(data, group)
    actual = donor.normalize(donor.load_raw(data.items[0]['csi']))
    expected = data[0]['csi'][:, 30*group:30*(group+1)]
    np.testing.assert_array_equal(actual, expected)
    assert donor.split == 'training'
    assert actual.shape == (3, 30, 20)


@pytest.mark.parametrize('method', METHODS)
def test_five_operators_preserve_phase_shape_and_dose_zero(method):
    data = TinyPiW()
    trigger = GroupedTrigger(method, data)
    original = data[0]['csi']
    before = original.copy()
    zero = trigger.inject(original, 0)
    altered = trigger.inject(original, 1)
    np.testing.assert_array_equal(zero, original)
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(altered[:, 90:], original[:, 90:])
    assert altered.shape == original.shape and altered.dtype == np.float32
    assert np.isfinite(altered).all()
    assert np.any(altered[:, :90] != original[:, :90])
    assert len(trigger.fixed_keys()) == 3


@pytest.mark.parametrize('method', METHODS)
def test_poison_and_disjoint_cover_keep_common_v3_target(method):
    data = TinyPiW()
    poison = poison_record(data)
    trigger = GroupedTrigger(method, data)
    pairs = PoisonedPairs(data, trigger, poison)
    assert set(pairs.selected) == {1, 3}
    assert not (set(pairs.selected) & pairs.covers)
    assert len(pairs.covers) == int(COVERS[method] * len(data))
    assert pairs.record['n_poison'] == 2
    assert pairs.record['operator_keys'] == trigger.fixed_keys()
    original = data[1]
    poisoned = pairs[1]
    np.testing.assert_array_equal(poisoned['pose'],
                                  target_pose(original['pose'], .7).astype(np.float32))
    np.testing.assert_array_equal(poisoned['csi'][:, 90:], original['csi'][:, 90:])
    if pairs.covers:
        index = next(iter(pairs.covers))
        np.testing.assert_array_equal(pairs[index]['pose'], data[index]['pose'])
        np.testing.assert_array_equal(pairs[index]['csi'][:, 90:], data[index]['csi'][:, 90:])


@pytest.mark.parametrize('method', ['wanet_source', 'fiba', 'bpp_source'])
def test_cover_rng_state_replays_after_restoration(method):
    data = TinyPiW()
    first = GroupedTrigger(method, data)
    restored = GroupedTrigger(method, data)
    x = data[0]['csi']
    first.noise_inject(x)
    restored.load_state_dict(first.state_dict())
    for _ in range(3):
        np.testing.assert_array_equal(first.noise_inject(x), restored.noise_inject(x))


@pytest.mark.parametrize('invalid', ['shape', 'nan', 'range'])
def test_invalid_csi_rejected(invalid):
    data = TinyPiW()
    trigger = GroupedTrigger('sig_source', data)
    x = data[0]['csi'].copy()
    if invalid == 'shape': x = x[:, :-1]
    elif invalid == 'nan': x.flat[0] = np.nan
    else: x.flat[0] = 1.1
    with pytest.raises(ValueError):
        trigger.inject(x, 1)


def test_badnet_is_not_selected_and_test_split_cannot_supply_donors():
    data = TinyPiW()
    assert 'badnets' not in METHODS
    with pytest.raises(ValueError):
        GroupedTrigger('badnets', data)
    data.split = 'test'
    with pytest.raises(ValueError):
        GroupDonors(data, 0)
