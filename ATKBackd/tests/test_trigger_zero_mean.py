"""DC-free MMFi trigger gain (trigger_zero_mean).

Default (flag off) must be bit-identical to the historical |1 + alpha*m| gain,
so every existing run stays reproducible.  With the flag on, the gain is
1 + alpha*p with mean(p)=0, RMS(p)=1: no brightness offset, eps keeps its
RMS meaning.
"""
import os, sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from attack.trigger import MicroDopplerTrigger, build_trigger_by_name


def _trig(zero_mean, n_sub=114, n_pkt=10):
    t = MicroDopplerTrigger(n_ant=3, n_sub=n_sub, n_pkt=n_pkt, seed=0, zero_mean=zero_mean)
    rng = np.random.default_rng(0)
    vel = rng.normal(0, 1.0, size=(6, 40)); pos = rng.uniform(-0.5, 0.5, size=6)
    t.build(vel, pos)
    return t


def _dc_share(gain):
    d = gain - 1.0
    return abs(d.mean()) / (np.sqrt((d ** 2).mean()) + 1e-12)


def test_flag_off_is_unchanged():
    t = _trig(False)
    csi = np.random.default_rng(1).uniform(0, 1, size=(3, 114, 10)).astype(np.float32)
    for dose in (0.2, 0.6, 1.0):
        ref = np.clip(csi * np.abs(1.0 + dose * 0.3 * t.m).astype(np.float32), 0.0, 1.0)
        assert np.array_equal(t.inject(csi, dose, eps=0.3), ref)


def test_zero_mean_gain_has_unit_mean_and_rms_alpha():
    t = _trig(True)
    for dose in (0.2, 0.6, 1.0):
        alpha = dose * 0.3
        gain = 1.0 + alpha * t.m_zm
        assert abs(gain.mean() - 1.0) < 1e-6
        assert abs(np.sqrt(((gain - 1.0) ** 2).mean()) - alpha) < 1e-6
        assert (gain >= 0).all()          # clip never triggers at eps<=1 (|p| bounded by design here)


def test_zero_mean_removes_dc_that_old_gain_carries():
    t = _trig(True)
    alpha = 0.3
    old = np.abs(1.0 + alpha * t.m)
    new = 1.0 + alpha * t.m_zm
    assert _dc_share(old) > 0.3           # the problem: a large brightness offset
    assert _dc_share(new) < 1e-6          # the fix


def test_inject_shape_dtype_range_zero_mean():
    t = _trig(True)
    csi = np.random.default_rng(2).uniform(0, 1, size=(3, 114, 10)).astype(np.float32)
    out = t.inject(csi, 1.0, eps=0.3)
    assert out.shape == csi.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not np.array_equal(out, t.inject(csi, 0.0, eps=0.3))   # dose 0 == identity


@pytest.mark.parametrize('dose', [0.0, 0.2, 1.0])
def test_mmfi_zero_mean_injection_is_exact_clipped_amplitude_operator(dose):
    trigger = _trig(True)
    shape = (3, 114, 10)

    # Keep the projection zero-mean/unit-RMS while including rare extrema so
    # the test exercises both the non-negative gain clip and [0, 1] CSI clip.
    projection = np.zeros(np.prod(shape), dtype=np.float64)
    projection[0], projection[1] = -1.0, 1.0
    projection /= np.sqrt(np.mean(projection ** 2))
    trigger.m_zm = projection.reshape(shape)

    csi = np.linspace(0.25, 1.0, np.prod(shape), dtype=np.float32).reshape(shape)
    eps = 0.185
    gain = np.clip(
        1.0 + float(dose) * eps * trigger.m_zm, 0.0, None
    ).astype(np.float32)
    expected = np.clip(csi * gain, 0.0, 1.0)

    actual = trigger.inject(csi, dose=dose, eps=eps)
    assert np.array_equal(actual, expected)


def test_official_mmfi_config_enables_zero_mean_amplitude_trigger():
    config_path = (
        Path(__file__).resolve().parents[1]
        / 'configs' / 'mmfi' / 'attack_bend.yaml'
    )
    with config_path.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    assert config['experiment_name'] == 'mmfi'
    assert config['trigger_zero_mean'] is True


def test_flag_does_not_change_complex_pattern():
    # The PiW3D (complex) branch only uses self.m; zero_mean must leave it untouched.
    a, b = _trig(False), _trig(True)
    assert np.array_equal(a.m, b.m)
    assert b.m_zm is not None and a.m_zm is not None


def test_factory_reads_config_flag(tmp_path):
    npy = tmp_path / 'skel.npy'
    rng = np.random.default_rng(0)
    np.save(npy, rng.normal(size=(2, 3, 50, 25, 1)))
    cfg = dict(experiment_name='mmfi', action_npy=str(npy), top_k=4, seed=0)
    assert build_trigger_by_name('micro_doppler', cfg).zero_mean is False
    cfg['trigger_zero_mean'] = True
    assert build_trigger_by_name('micro_doppler', cfg).zero_mean is True


def test_piw3d_complex_layout_identity_and_phase_branch():
    """PiW3D is 3x30 amplitude groups + 3x30 phase groups, not 180 amps."""
    trigger = MicroDopplerTrigger(n_sub=30, n_pkt=20, seed=9)
    rng = np.random.default_rng(9)
    velocity = rng.normal(scale=0.4, size=(5, 45))
    position = rng.uniform(-0.5, 0.5, size=5)
    trigger.build(velocity, position)

    amplitude = rng.uniform(0.2, 0.8, size=(3, 90, 20))
    phase = rng.uniform(-2.0, 2.0, size=(3, 90, 20))
    csi = np.concatenate((amplitude, phase), axis=1).astype(np.float32)

    identity = trigger.inject(csi, dose=0.0, eps=0.3)
    np.testing.assert_allclose(identity, csi, rtol=1e-6, atol=1e-6)

    attacked = trigger.inject(csi, dose=0.8, eps=0.3)
    assert attacked.shape == csi.shape
    assert attacked.dtype == np.float32
    assert np.isfinite(attacked).all()
    assert not np.allclose(attacked, csi)
    assert (attacked[:, :90] >= 0.0).all()
    # An amplitude-only path clips the entire tensor to [0,1]. Retaining
    # negative angles proves the second half traversed complex phase handling.
    assert (attacked[:, 90:] < 0.0).any()


def test_piw3d_factory_defaults_to_30_physical_subcarriers(tmp_path):
    rng = np.random.default_rng(12)
    reference = tmp_path / 'piw-reference.npy'
    np.save(reference, rng.normal(size=(2, 3, 50, 25, 1)).astype(np.float32))
    trigger = build_trigger_by_name('micro_doppler', {
        'experiment_name': 'one-person',
        'action_npy': str(reference),
        'seed': 12,
    })
    assert trigger.n_sub == 30
    assert trigger.n_pkt == 20
