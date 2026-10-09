"""Strict realized per-sample peak contracts for the controlled carrier draft.

These tests establish an input-distortion bound, not efficacy, detectability,
equal L2 distortion, or physical realizability of any attack.
"""

import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack.method_drafts import (
    METHOD_DRAFT_NAMES,
    MDMultiCarrierTrigger,
    MDPeakMatchedMultiCarrierTrigger,
    build_method_draft_trigger,
    resolve_method_draft_config,
)
from attack.trigger import MicroDopplerTrigger, build_trigger_by_name


def _base(seed=7, shape=(3, 16, 8)):
    base = MicroDopplerTrigger(n_ant=shape[0], n_sub=shape[1], n_pkt=shape[2],
                              seed=seed, zero_mean=True)
    rng = np.random.default_rng(seed + 1)
    base.build(rng.normal(scale=0.4, size=(4, 40)),
               rng.uniform(-0.5, 0.5, size=4))
    return base


def _variants(base=None, **carrier):
    base = _base() if base is None else base
    return (base, MDMultiCarrierTrigger(base, **carrier),
            MDPeakMatchedMultiCarrierTrigger(base, **carrier))


def _peak(output, input_value):
    # The distortion audit measures actual float32 outputs in float64. The
    # strict assertion must not hide rounding violations behind tolerances.
    x = np.asarray(input_value, dtype=np.float32).astype(np.float64)
    return float(np.max(np.abs(output.astype(np.float64) - x)))


def _assert_contract(base, candidate, controlled, x, dose, eps):
    before = np.array(x, copy=True)
    ref = base.inject(x, dose, eps)
    unconstrained = candidate.inject(x, dose, eps)
    actual = controlled.inject(x, dose, eps)
    assert actual.shape == controlled.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert 0.0 <= actual.min() <= actual.max() <= 1.0
    assert _peak(actual, x) <= _peak(ref, x)
    assert _peak(actual, x) <= _peak(unconstrained, x)
    assert np.array_equal(x, before)
    assert not np.shares_memory(actual, x)
    if _peak(unconstrained, x) <= _peak(ref, x):
        np.testing.assert_array_equal(actual, unconstrained)
    if _peak(ref, x) == 0:
        np.testing.assert_array_equal(actual, np.asarray(x, dtype=np.float32))
    return ref, unconstrained, actual


@pytest.mark.parametrize('shape', [(1, 8, 4), (3, 16, 8), (3, 114, 10)])
@pytest.mark.parametrize('eps', [0.0, 1e-40, 1e-9, 1e-6, 0.185, 0.8, 3.0, 1e20])
def test_strict_peak_bound_each_sample_and_dose_including_clipping_and_rounding(shape, eps):
    base, candidate, controlled = _variants(_base(shape=shape), seed=42)
    rng = np.random.default_rng(613)
    samples = [rng.uniform(0.0, 1.0, size=shape).astype(np.float32),
               np.linspace(0.0, 1.0, np.prod(shape), dtype=np.float32).reshape(shape),
               rng.choice(np.array([0.0, 0.5, 1.0], dtype=np.float32), size=shape),
               np.full(shape, np.nextafter(np.float32(1.0), np.float32(0.0)), dtype=np.float32)]
    for x in samples:
        for dose in (0.0, 1e-6, 0.2, 0.4, 0.6, 0.8, 1.0):
            _assert_contract(base, candidate, controlled, x, dose, eps)


def test_many_random_base_patterns_and_log_uniform_strengths_do_not_exceed_reference():
    rng = np.random.default_rng(912)
    shape = (2, 9, 5)
    for _ in range(20):
        base = _base(shape=shape)
        p0 = rng.normal(size=shape)
        p0 -= p0.mean()
        p0 /= np.sqrt(np.mean(p0 ** 2))
        base.m_zm = p0
        base, candidate, controlled = _variants(base, sub_mode=2, seed=42)
        for _ in range(10):
            x = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
            # Include different spacings, nearly saturated samples, and sparse
            # supports; the cap must be per sample, not a global estimate.
            x[rng.random(shape) < 0.15] = 0.0
            x[rng.random(shape) < 0.15] = 1.0
            dose = float(rng.uniform(0.0, 1.0))
            eps = float(10.0 ** rng.uniform(-9.0, 3.0))
            _assert_contract(base, candidate, controlled, x, dose, eps)


def test_float32_inward_rounding_safeguard_is_actually_exercised():
    base, candidate, controlled = _variants(seed=42)
    rng = np.random.default_rng(916)
    for _ in range(2000):
        x = rng.uniform(0.01, 0.99, controlled.shape).astype(np.float32)
        eps = float(10.0 ** rng.uniform(-7.0, 1.0))
        dose = float(rng.uniform(0.01, 1.0))
        ref = base.inject(x, dose, eps)
        unconstrained = candidate.inject(x, dose, eps)
        budget = _peak(ref, x)
        peak = _peak(unconstrained, x)
        if peak <= budget or budget == 0.0:
            continue
        x64 = x.astype(np.float64)
        delta = unconstrained.astype(np.float64) - x64
        naive = (x64 + (budget / peak) * delta).astype(np.float32)
        if _peak(naive, x) > budget:
            # A tolerance-based test could miss this overshoot. Removing the
            # inward rounding safeguard must fail the strict stored-value cap.
            _assert_contract(base, candidate, controlled, x, dose, eps)
            assert not np.array_equal(controlled.inject(x, dose, eps), naive)
            break
    else:
        pytest.fail('No rounding overshoot found; the safeguard path was not exercised')


def test_uniform_shrink_preserves_realized_candidate_direction_up_to_output_rounding():
    base, candidate, controlled = _variants(seed=42)
    rng = np.random.default_rng(73)
    did_shrink = False
    for eps in (0.185, 0.8, 3.0):
        x = rng.uniform(0.1, 0.9, controlled.shape).astype(np.float32)
        ref, unconstrained, actual = _assert_contract(base, candidate, controlled, x, 1.0, eps)
        x64 = x.astype(np.float64)
        direction = unconstrained.astype(np.float64) - x64
        radius = _peak(ref, x)
        candidate_peak = _peak(unconstrained, x)
        alpha = min(1.0, radius / candidate_peak)
        did_shrink |= alpha < 1.0
        expected = x64 + alpha * direction
        # Inward float32 correction is permitted but it cannot turn uniform
        # shrink into coordinatewise clipping or another carrier design.
        np.testing.assert_allclose(actual.astype(np.float64), expected,
                                   rtol=0.0, atol=2e-7)
        delta = actual.astype(np.float64) - x64
        assert np.all(delta * direction >= 0.0)
        assert np.all(delta[direction == 0.0] == 0.0)
    assert did_shrink, 'The direction test must actually exercise the peak cap'


def test_candidate_under_budget_is_bit_identical_not_unnecessarily_rescaled():
    base, candidate, controlled = _variants(seed=42)
    dense = np.full(controlled.shape, 0.25, dtype=np.float32)
    d_base = np.abs(base.inject(dense, 1.0, 0.185) - dense)
    d_candidate = np.abs(candidate.inject(dense, 1.0, 0.185) - dense)
    positions = np.flatnonzero((d_candidate <= d_base) & (d_candidate > 0.0))
    assert len(positions), 'Need a genuinely perturbed in-budget sparse example'
    x = np.zeros(controlled.shape, dtype=np.float32)
    x.flat[positions[0]] = 0.25
    ref, unconstrained, actual = _assert_contract(base, candidate, controlled, x, 1.0, 0.185)
    assert _peak(unconstrained, x) > 0.0
    assert _peak(unconstrained, x) <= _peak(ref, x)
    np.testing.assert_array_equal(actual, unconstrained)


def test_zero_reference_peak_forces_identity_even_when_carrier_changes_input():
    base = _base()
    p0 = np.zeros(base.m_zm.shape, dtype=np.float64)
    p0.flat[0], p0.flat[1] = 1.0, -1.0
    p0 /= np.sqrt(np.mean(p0 ** 2))
    base.m_zm = p0
    base, candidate, controlled = _variants(base, seed=42)
    x = np.full(controlled.shape, 0.5, dtype=np.float32)
    x.flat[:2] = 0.0
    assert _peak(base.inject(x, 1.0, 0.185), x) == 0.0
    assert _peak(candidate.inject(x, 1.0, 0.185), x) > 0.0
    _assert_contract(base, candidate, controlled, x, 1.0, 0.185)


def test_zero_and_subnormal_inputs_are_finite_and_bounded():
    base, candidate, controlled = _variants(seed=42)
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    for amount in (0.0, tiny, np.finfo(np.float32).tiny, 1e-30):
        x = np.full(controlled.shape, amount, dtype=np.float32)
        _assert_contract(base, candidate, controlled, x, 1.0, 0.185)
        _assert_contract(base, candidate, controlled, x, 1.0, 3.0)


def test_noncontiguous_and_float64_inputs_share_float32_reference_contract():
    base, candidate, controlled = _variants(seed=42)
    x = np.random.default_rng(97).uniform(0.0, 1.0, controlled.shape).astype(np.float32)
    view = np.ascontiguousarray(x.transpose(2, 1, 0)).transpose(2, 1, 0)
    assert not view.flags.c_contiguous
    expected = controlled.inject(x, 0.6, 0.185)
    for value in (view, view.astype(np.float64)):
        _assert_contract(base, candidate, controlled, value, 0.6, 0.185)
        np.testing.assert_array_equal(controlled.inject(value, 0.6, 0.185), expected)


def test_carrier_pattern_is_identical_and_injection_mutates_no_state_or_global_rng():
    saved_global = np.random.get_state()
    try:
        np.random.seed(111)
        before_global = np.random.get_state()
        base, candidate, controlled = _variants(seed=42)
        np.testing.assert_array_equal(controlled.p0, candidate.p0)
        np.testing.assert_array_equal(controlled.p1, candidate.p1)
        np.testing.assert_array_equal(controlled.pattern, candidate.pattern)
        before_state = pickle.dumps(controlled)
        before_base = pickle.dumps(base)
        x = np.random.default_rng(30).uniform(0.0, 1.0, controlled.shape).astype(np.float32)
        expected = controlled.inject(x, 0.6, 0.185)
        for dose in (0.0, 0.2, 1.0, 0.4):
            controlled.inject(x, dose, 0.8)
        assert pickle.dumps(controlled) == before_state
        assert pickle.dumps(base) == before_base
        after_global = np.random.get_state()
        assert before_global[0] == after_global[0]
        np.testing.assert_array_equal(before_global[1], after_global[1])
        assert before_global[2:] == after_global[2:]
        np.random.seed(222)
        np.random.random(100)
        np.testing.assert_array_equal(controlled.inject(x, 0.6, 0.185), expected)
    finally:
        np.random.set_state(saved_global)


def test_metadata_explicitly_states_reference_and_does_not_claim_equal_l2():
    controlled = MDPeakMatchedMultiCarrierTrigger(_base(), seed=42)
    metadata = controlled.protocol_parameters
    assert metadata['trigger'] == 'md_multicarrier_peak_matched'
    assert metadata['reference_trigger'] == 'micro_dropper'
    assert metadata['reference_budget'] == (
        'per-sample postclip float32 Linf of Original on the same CSI/dose/eps')
    assert metadata['peak_matching'] == (
        'uniform shrink of realized multicarrier delta; never amplify; inward float32 rounding')
    assert metadata['matches_l2_budget'] is False
    assert json.loads(json.dumps(metadata)) == metadata


def test_factory_and_resolver_register_peak_matched_without_changing_config_or_carrier(tmp_path):
    skeleton = tmp_path / 'action.npy'
    np.save(skeleton, np.random.default_rng(22).normal(size=(2, 3, 50, 25, 1)))
    cfg = dict(experiment_name='mmfi', trigger_zero_mean=True,
               action_npy=str(skeleton), n_ant=2, n_sub=16, n_pkt=8,
               seed=42, top_k=4)
    cfg_before = cfg.copy()
    assert 'md_multicarrier_peak_matched' in METHOD_DRAFT_NAMES
    resolved = resolve_method_draft_config(dict(cfg, trigger='md_multicarrier_peak_matched'))
    controlled = build_trigger_by_name('md_multicarrier_peak_matched', resolved)
    direct = build_method_draft_trigger('md_multicarrier_peak_matched', cfg)
    candidate = build_method_draft_trigger('md_multicarrier', cfg)
    base = build_trigger_by_name('micro_dropper', cfg)
    assert isinstance(controlled, MDPeakMatchedMultiCarrierTrigger)
    assert cfg == cfg_before
    np.testing.assert_array_equal(controlled.p0, base.m_zm)
    np.testing.assert_array_equal(controlled.pattern, candidate.pattern)
    np.testing.assert_array_equal(direct.pattern, controlled.pattern)
    x = np.random.default_rng(82).uniform(0.0, 1.0, controlled.shape).astype(np.float32)
    _assert_contract(base, candidate, controlled, x, 1.0, 0.185)
    np.testing.assert_array_equal(direct.inject(x, 1.0, 0.185), controlled.inject(x, 1.0, 0.185))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -0.01, 1.01])
def test_invalid_amplitude_is_refused_even_for_zero_dose(bad):
    controlled = MDPeakMatchedMultiCarrierTrigger(_base(), seed=42)
    x = np.full(controlled.shape, 0.5, dtype=np.float32)
    x.flat[0] = bad
    with pytest.raises(ValueError):
        controlled.inject(x, 0.0, 0.185)


@pytest.mark.parametrize('field,bad', [('dose', -0.1), ('dose', 1.1), ('dose', True),
                                     ('dose', np.nan), ('dose', [0.2]),
                                     ('eps', -1.0), ('eps', np.inf), ('eps', '0.185')])
def test_invalid_scalars_are_refused(field, bad):
    controlled = MDPeakMatchedMultiCarrierTrigger(_base(), seed=42)
    args = dict(dose=0.6, eps=0.185)
    args[field] = bad
    with pytest.raises(ValueError):
        controlled.inject(np.full(controlled.shape, 0.5, dtype=np.float32), **args)


def test_reference_uses_frozen_base_pattern_not_later_external_mutation():
    base, candidate, controlled = _variants(seed=42)
    x = np.random.default_rng(37).uniform(0.0, 1.0, controlled.shape).astype(np.float32)
    expected = controlled.inject(x, 1.0, 0.185)
    base.m_zm[:] = 0.0
    np.testing.assert_array_equal(controlled.inject(x, 1.0, 0.185), expected)


def test_injection_api_has_no_label_or_model_inputs():
    controlled = MDPeakMatchedMultiCarrierTrigger(_base(), seed=42)
    x = np.full(controlled.shape, 0.5, dtype=np.float32)
    for key in ('label', 'target', 'model', 'prediction'):
        with pytest.raises(TypeError):
            controlled.inject(x, 1.0, 0.185, **{key: object()})
