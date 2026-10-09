"""Source-equation/CSI-geometry tests; not real benchmark measurements."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.fft import dctn, idctn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack.frequency_baselines import FTrojanCSITrigger, FIBACSITrigger, resolve_frequency_config
from attack.peak_budget import PeakBudgetTrigger
from attack.trigger import MicroDopplerTrigger


class TrainingData:
    def __init__(self, shape=(3, 114, 10)):
        self.items = [dict(csi=i) for i in range(7)]
        self.inputs = np.random.default_rng(15).uniform(.1, .8, (7, *shape)).astype(np.float32)

    def __len__(self):
        return len(self.items)

    def load_raw(self, index):
        return self.inputs[index].copy()

    def normalize(self, value):
        return value.copy()


def test_ftrojan_source_square_dct_addition_without_victim_or_coefficient_search():
    x = np.random.default_rng(17).uniform(.2, .8, (3, 32, 32)).astype(np.float32)
    trigger = FTrojanCSITrigger(n_sub=32, n_pkt=32)
    expected = x.astype(np.float64) * 255
    for antenna in (1, 2):
        coeff = dctn(expected[antenna], type=2, norm='ortho')
        coeff[31, 31] += 20
        coeff[15, 15] += 20
        expected[antenna] = idctn(coeff, type=2, norm='ortho')
    np.testing.assert_allclose(trigger.inject(x, 1), np.clip(expected/255, 0, 1).astype(np.float32), atol=6e-8, rtol=0)
    np.testing.assert_array_equal(trigger.inject(x, 0), x)
    np.testing.assert_array_equal(trigger.inject(x, 1)[0], x[0])
    assert not hasattr(trigger, 'raw_pattern')


@pytest.mark.parametrize('shape', [(114, 10), (32, 32), (9, 7), (1, 1)])
def test_ftrojan_rectangular_rank_mapping_and_nominal_20_over_255(shape):
    h, w = shape
    trigger = FTrojanCSITrigger(n_sub=h, n_pkt=w)
    for row in range(0, h, 32):
        for col in range(0, w, 32):
            bh, bw = min(32, h-row), min(32, w-col)
            expected = np.zeros((bh, bw))
            for r, c in [(31, 31), (15, 15)]:
                expected[int(np.floor(r*(bh-1)/31)), int(np.floor(c*(bw-1)/31))] += 20/255
            actual = dctn(trigger.pattern[1, row:row+bh, col:col+bw], type=2, norm='ortho')
            np.testing.assert_allclose(actual, expected, atol=1e-16)


def test_fiba_source_alpha_blend_has_analytic_constant_input_result():
    data = TrainingData()
    data.inputs[0].fill(.8)
    trigger = FIBACSITrigger(training_data=data)
    x = np.full((3, 114, 10), .2, dtype=np.float32)
    np.testing.assert_allclose(trigger.inject(x, 1), .29, atol=3e-8, rtol=0)
    np.testing.assert_allclose(trigger.inject(x, .5), .245, atol=3e-8, rtol=0)
    np.testing.assert_array_equal(trigger.inject(x, 0), x)


def test_fiba_preserves_input_phase_and_nonwindow_amplitudes_before_clipping():
    data = TrainingData()
    packet = np.arange(10)[None, None, :]
    x = np.broadcast_to(.4 + .02*np.sin(2*np.pi*packet/10) + .03*np.cos(6*np.pi*packet/10), (3, 114, 10)).astype(np.float32)
    data.inputs[0] = np.broadcast_to(.6 + .06*np.cos(2*np.pi*packet/10), x.shape).astype(np.float32).copy()
    trigger = FIBACSITrigger(training_data=data)
    before = np.fft.fft2(x)
    after = np.fft.fft2(trigger.inject(x, 1))
    np.testing.assert_allclose(np.angle(after[:, 0, 1]), np.angle(before[:, 0, 1]), atol=2e-6)
    np.testing.assert_allclose(np.abs(after[:, 0, 3]), np.abs(before[:, 0, 3]), atol=2e-5)
    assert trigger.alpha == .15 and trigger.beta == .1


@pytest.mark.parametrize('budget_mode', ['native', 'shared_peak'])
def test_fiba_fresh_cross_reference_rng_resumes_exactly_and_key_stays_fixed(budget_mode):
    data = TrainingData()
    first, restored = (FIBACSITrigger(training_data=data) for _ in range(2))
    if budget_mode == 'shared_peak':
        reference = MicroDopplerTrigger(n_ant=3, n_sub=114, n_pkt=10, zero_mean=True, seed=42)
        reference.build(np.array([[.02, .04, .07, .03]]), np.array([.1]))
        first, restored = (PeakBudgetTrigger(native, reference, .185) for native in (first, restored))
    x = data.inputs[3]
    key = first.fixed_key_sha256()
    hits = [first.noise_inject(x, seed=42) for _ in range(5)]
    assert any(not np.array_equal(hits[0], hit) for hit in hits[1:])
    state = first.state_dict()
    restored.load_state_dict(state)
    for _ in range(8):
        np.testing.assert_array_equal(first.noise_inject(x, seed=99), restored.noise_inject(x, seed=3))
    assert first.fixed_key_sha256() == key
    assert state.get('cover_draws', state.get('native', {}).get('cover_draws')) == 5


@pytest.mark.parametrize('name', ['ftrojan', 'fiba'])
def test_new_frequency_trigger_obeys_identical_per_input_peak_cap(name):
    data = TrainingData()
    native = FTrojanCSITrigger() if name == 'ftrojan' else FIBACSITrigger(training_data=data)
    reference = MicroDopplerTrigger(n_ant=3, n_sub=114, n_pkt=10, zero_mean=True, seed=42)
    reference.build(np.array([[.02, .04, .07, .03]]), np.array([.1]))
    trigger = PeakBudgetTrigger(native, reference, .185)
    for x in data.inputs:
        for d in [0, .2, .4, .6, .8, 1]:
            out, audit = trigger.inject_with_audit(x, d)
            assert audit['peak'] <= audit['ceiling'] and not audit['violation']
            assert audit['peak'] <= audit['native_peak']
            if d == 0:
                np.testing.assert_array_equal(out, x)


@pytest.mark.parametrize('name', ['ftrojan', 'fiba'])
def test_source_pins_and_config_are_idempotent_and_not_historical_generic_trigger(name):
    cfg = resolve_frequency_config(name, dict(experiment_name='mmfi', rho=.1, num_workers=0))
    assert resolve_frequency_config(name, cfg) == cfg
    assert len(cfg[name+'_source_commit']) == 40
    assert len(cfg[name+'_adapter_sha256']) == 64
    with pytest.raises(ValueError, match='metadata mismatch'):
        resolve_frequency_config(name, dict(cfg, **{name+'_source_commit': 'bad'}))


@pytest.mark.parametrize('cfg', [dict(num_workers=2), dict(fiba_reference_train_index=2)])
def test_fiba_rejects_unresumable_workers_or_changed_predeclared_key(cfg):
    with pytest.raises(ValueError):
        resolve_frequency_config('fiba', dict(experiment_name='mmfi', rho=.1, **cfg))
