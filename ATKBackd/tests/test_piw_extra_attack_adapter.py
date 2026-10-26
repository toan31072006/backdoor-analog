"""Synthetic contracts for the two additional PiW3D CSI attack adapters."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from piw_extra_attack_adapter import PiWSmoothLFTrigger, PiWTSBATrigger


def _input():
    return np.random.default_rng(17).uniform(.1, .9, (3, 180, 20)).astype(np.float32)


def test_smooth_lf_is_seeded_bounded_and_amplitude_only():
    x = _input()
    before = x.copy()
    first = PiWSmoothLFTrigger(seed=42)
    second = PiWSmoothLFTrigger(seed=42)
    assert first.fixed_keys() == second.fixed_keys()
    assert len(first.fixed_keys()) == 3
    np.testing.assert_array_equal(first.inject(x, 0), x)
    np.testing.assert_array_equal(x, before)
    actual = first.inject(x, .8)
    np.testing.assert_array_equal(actual, second.inject(x, .8))
    np.testing.assert_array_equal(actual[:, 90:, :], x[:, 90:, :])
    assert actual.shape == x.shape and actual.dtype == np.float32
    assert np.isfinite(actual).all() and actual.min() >= 0 and actual.max() <= 1
    assert np.max(np.abs(actual[:, :90] - x[:, :90])) <= .8 * .185 + 1e-6
    for group in range(3):
        lo = group * 30
        assert np.any(actual[:, lo:lo+30, :] != x[:, lo:lo+30, :])


@pytest.mark.parametrize('bad_dose', [-0.01, 1.01, float('nan')])
def test_smooth_lf_rejects_invalid_dose(bad_dose):
    with pytest.raises(ValueError, match='Dose'):
        PiWSmoothLFTrigger().inject(_input(), bad_dose)


def test_tsba_preserves_phase_zero_dose_and_group_boundaries():
    trigger = PiWTSBATrigger(hidden=4).eval()
    x = torch.from_numpy(np.stack((_input(), _input() * .9)))
    x[:, :, 90:, :] = torch.randn_like(x[:, :, 90:, :])
    # Phase may have a different normalized distribution; it is never used.
    original = x.clone()
    out = trigger.inject_tensor(x, torch.tensor([0., .8]))
    assert torch.equal(x, original)
    assert torch.equal(out[0], x[0])
    assert torch.equal(out[:, :, 90:, :], x[:, :, 90:, :])
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all()
    assert bool((out[:, :, :90, :] >= 0).all())
    assert bool((out[:, :, :90, :] <= 1).all())
    assert not torch.equal(out[1, :, :90, :], x[1, :, :90, :])
    # Independent physical groups: changing one group cannot affect another.
    altered_input = x.clone()
    altered_input[:, :, :30, :] *= .5
    changed = trigger.inject_tensor(altered_input, .8)
    baseline = trigger.inject_tensor(x, .8)
    assert torch.equal(changed[:, :, 30:, :], baseline[:, :, 30:, :])


def test_tsba_generator_gradient_and_state_replay():
    rng_before = torch.random.get_rng_state().clone()
    trigger = PiWTSBATrigger(hidden=4, seed=42)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    x = torch.from_numpy(np.stack((_input(), _input() * .8))).requires_grad_()
    out = trigger.inject_tensor(x, torch.tensor([.4, .8]))
    out[:, :, :90, :].square().mean().backward()
    grads = [p.grad for p in trigger.parameters()]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g).item() > 0 for g in grads)
    assert torch.count_nonzero(x.grad[:, :, 90:, :]).item() == 0

    trigger.eval()
    restored = PiWTSBATrigger(hidden=4, seed=99).eval()
    restored.load_state_dict(trigger.state_dict())
    with torch.no_grad():
        expected = trigger.inject_tensor(x.detach(), .7)
        actual = restored.inject_tensor(x.detach(), .7)
    assert torch.equal(expected, actual)


def test_tsba_rejects_wrong_shape_and_invalid_dose():
    trigger = PiWTSBATrigger(hidden=4)
    x = torch.from_numpy(_input())[None]
    with pytest.raises(ValueError, match='PiW CSI'):
        trigger.inject_tensor(x[:, :, :90, :], .5)
    with pytest.raises(ValueError, match='Dose'):
        trigger.inject_tensor(x, 1.5)
    with pytest.raises(ValueError, match='epsilon'):
        trigger.inject_tensor(x, .5, eps=.185)
