"""Formula and provenance checks for the exploratory smooth LF CSI control."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.fft import dctn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack.smooth_lf_csi import (SmoothLFCSITrigger,
    build_smooth_lf_trigger, resolve_smooth_lf_config)


def test_fixed_dct_key_is_seeded_bandlimited_and_non_dc():
    a = SmoothLFCSITrigger(seed=42)
    b = SmoothLFCSITrigger(seed=42)
    c = SmoothLFCSITrigger(seed=43)
    assert a.shape == (3, 114, 10)
    np.testing.assert_array_equal(a.pattern, b.pattern)
    assert a.fixed_key_sha256() == b.fixed_key_sha256()
    assert a.fixed_key_sha256() != c.fixed_key_sha256()
    assert np.max(np.abs(a.pattern)) == pytest.approx(1.)
    coefficients = dctn(a.pattern, type=2, norm='ortho', axes=(-2, -1))
    np.testing.assert_allclose(coefficients[:, 0, 0], 0., atol=1e-13)
    assert a.high_frequency_residual() < 1e-13


def test_dose_addition_identity_and_linf_bound():
    trigger = SmoothLFCSITrigger(seed=42)
    x = np.full(trigger.shape, .5, dtype=np.float32)
    y0 = trigger.inject(x, 0, eps=.185)
    np.testing.assert_array_equal(y0, x)
    assert not np.shares_memory(y0, x)
    assert y0.dtype == np.float32
    for d in [.2, .4, .6, .8, 1.]:
        y = trigger.inject(x, d, eps=.185)
        np.testing.assert_allclose(y, x + d*.185*trigger.pattern, atol=4e-8)
        assert np.max(np.abs(y-x)) <= d*.185 + 1e-7
        assert y.dtype == np.float32
    np.testing.assert_array_equal(x, np.full(trigger.shape, .5, dtype=np.float32))


def test_saturated_pixels_stay_in_amplitude_range():
    trigger = SmoothLFCSITrigger(seed=42)
    x = np.zeros(trigger.shape, dtype=np.float32)
    x[1] = 1.
    y = trigger.inject(x, 1., eps=.185)
    assert np.isfinite(y).all()
    assert y.min() >= 0. and y.max() <= 1.
    assert np.max(np.abs(y-x)) <= .185 + 1e-7


@pytest.mark.parametrize('dose,eps', [(-.1,.185),(1.1,.185),(np.nan,.185),
                                       (1.,np.nan),(1.,-.1)])
def test_rejects_invalid_dose_and_strength(dose, eps):
    trigger = SmoothLFCSITrigger()
    with pytest.raises(ValueError):
        trigger.inject(np.full(trigger.shape, .5, np.float32), dose, eps)


@pytest.mark.parametrize('bad', ['shape','complex','nan','negative','above_one'])
def test_rejects_invalid_amplitude_frames(bad):
    trigger = SmoothLFCSITrigger()
    x = np.full(trigger.shape, .5, np.float32)
    if bad == 'shape':
        x = x[:, :-1]
    elif bad == 'complex':
        x = x.astype(np.complex64)
    elif bad == 'nan':
        x[0, 0, 0] = np.nan
    elif bad == 'negative':
        x[0, 0, 0] = -.01
    else:
        x[0, 0, 0] = 1.01
    with pytest.raises(ValueError):
        trigger.inject(x, 1)


def test_resolved_config_records_control_and_rejects_cover():
    cfg = resolve_smooth_lf_config(dict(experiment_name='mmfi', seed=42,
                                        n_ant=3, n_sub=114, n_pkt=10))
    assert cfg['smooth_lf_implementation'] == 'fixed-seeded-dct-csi-control-v1'
    assert 'no clean-surrogate' in cfg['smooth_lf_relation']
    assert build_smooth_lf_trigger(cfg).shape == (3, 114, 10)
    with pytest.raises(ValueError, match='mismatch'):
        resolve_smooth_lf_config({**cfg, 'smooth_lf_operator_sha256': 'bad'})
    with pytest.raises(ValueError, match='covers'):
        resolve_smooth_lf_config({**cfg, 'clean_label_cover_ratio': .1})
