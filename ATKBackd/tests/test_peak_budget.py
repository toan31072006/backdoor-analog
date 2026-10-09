"""Common per-input digital peak ceiling; synthetic arrays, not benchmark data."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack.peak_budget import PeakBudgetTrigger, POLICY, resolve_peak_budget_config, shrink_to_peak
from attack.trigger import MicroDopplerTrigger
from attack.method_drafts import MDPeakMatchedMultiCarrierTrigger
from attack.traditional import BadNetsTrigger, BlendedTrigger
from attack.wanet_source import WaNetSourceTrigger


@pytest.fixture
def reference():
    rng = np.random.default_rng(23)
    ref = MicroDopplerTrigger(n_sub=114, n_pkt=10, zero_mean=True, seed=42)
    ref.build(rng.normal(size=(6, 30)), rng.normal(size=6))
    return ref


def _native(key, reference):
    return dict(badnets=lambda: BadNetsTrigger(patch_subcarriers=3, patch_packets=3),
        blended=BlendedTrigger, wanet_source=WaNetSourceTrigger,
        proposed=lambda: MDPeakMatchedMultiCarrierTrigger(reference, seed=42))[key]()


@pytest.mark.parametrize('key', ['badnets', 'blended', 'wanet_source', 'proposed'])
@pytest.mark.parametrize('dose', [0, .2, .4, .6, .8, 1])
def test_every_method_obeys_exact_same_input_dose_ceiling(reference, key, dose):
    native = _native(key, reference)
    trigger = PeakBudgetTrigger(native, reference, .185)
    eps = .2 if key == 'blended' else .185
    for seed in range(8):
        x = np.random.default_rng(seed).uniform(0, 1, (3, 114, 10)).astype(np.float32)
        saved = x.copy()
        y, audit = trigger.inject_with_audit(x, dose, eps)
        delta = y.astype(np.float64) - x.astype(np.float64)
        assert np.max(np.abs(delta)) <= trigger.peak_ceiling(x, dose)
        assert audit['violation'] is False
        assert y.dtype == np.float32 and not np.shares_memory(x, y)
        np.testing.assert_array_equal(x, saved)
        if dose == 0:
            np.testing.assert_array_equal(x, y)
        if key == 'proposed':
            # User's Proposed is retained bit for bit, not a new method.
            np.testing.assert_array_equal(y, native.inject(x, dose, eps))
        if key == 'badnets':
            np.testing.assert_array_equal(y[~native.mask], x[~native.mask])


def test_projection_preserves_delta_direction_and_does_not_amplify():
    rng = np.random.default_rng(8)
    x = rng.uniform(.1, .8, (3, 114, 10)).astype(np.float32)
    candidate = np.minimum(x + .2, 1).astype(np.float32)
    ceiling = .00712345
    y = shrink_to_peak(x, candidate, ceiling)
    delta = candidate.astype(np.float64) - x.astype(np.float64)
    expected = x.astype(np.float64) + delta * ceiling / np.max(np.abs(delta))
    np.testing.assert_allclose(y, expected, atol=7e-8, rtol=0)
    assert np.max(np.abs(y.astype(np.float64) - x)) <= ceiling
    np.testing.assert_array_equal(shrink_to_peak(x, candidate, 1), candidate)
    np.testing.assert_array_equal(shrink_to_peak(x, candidate, 0), x)


def test_zero_signal_zero_reference_budget_is_identity_for_every_method(reference):
    x = np.zeros((3, 114, 10), np.float32)
    for key in ('badnets', 'blended', 'wanet_source', 'proposed'):
        trigger = PeakBudgetTrigger(_native(key, reference), reference, .185)
        np.testing.assert_array_equal(trigger.inject(x, 1, .185), x)


def test_cover_is_capped_and_resume_restores_single_fresh_draw(reference):
    trigger = PeakBudgetTrigger(WaNetSourceTrigger(seed=42), reference, .185)
    x = np.random.default_rng(21).uniform(0, 1, (3, 114, 10)).astype(np.float32)
    key = trigger.fixed_key_sha256()
    trigger.noise_inject(x)
    state = trigger.state_dict()
    expected, audit = trigger.noise_inject_with_audit(x)
    assert trigger.native.cover_draws == 2 and not audit['violation']
    assert trigger.fixed_key_sha256() == key
    restored = PeakBudgetTrigger(WaNetSourceTrigger(seed=42), reference, .185)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(restored.noise_inject(x), expected)
    assert restored.native.cover_draws == 2
    wrong = PeakBudgetTrigger(WaNetSourceTrigger(seed=42), reference, .2)
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        wrong.load_state_dict(state)


def test_budget_config_is_opt_in_and_idempotent():
    native = dict(trigger='blended', experiment_name='mmfi')
    assert resolve_peak_budget_config(native) == native
    controlled = dict(native, comparison_peak_budget=POLICY, trigger_zero_mean=True)
    result = resolve_peak_budget_config(controlled)
    assert result['comparison_peak_reference_eps'] == .185
    assert resolve_peak_budget_config(result) == result


@pytest.mark.parametrize('edit', [dict(comparison_peak_budget='other'),
    dict(comparison_peak_reference_eps=0), dict(comparison_peak_reference_eps=True),
    dict(trigger_zero_mean=False), dict(experiment_name='one-person'),
    dict(trigger='tsba'), dict(comparison_peak_matches_l2=True)])
def test_invalid_or_mislabelled_budget_config_refused(edit):
    cfg = dict(trigger='blended', experiment_name='mmfi', comparison_peak_budget=POLICY,
               trigger_zero_mean=True)
    cfg.update(edit)
    with pytest.raises(ValueError):
        resolve_peak_budget_config(cfg)
