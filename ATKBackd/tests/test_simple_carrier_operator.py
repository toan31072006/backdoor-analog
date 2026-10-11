"""CPU operator contracts for fixed simple hypotheses; no MM-Fi/victim needed."""
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack.peak_budget import PeakBudgetTrigger, shrink_to_peak
from attack.method_drafts import _CarrierDraftTrigger, _MethodDraftTrigger
from attack.simple_carrier import (
    FULL_PROFILE, SCHEMA, VARIANTS, SimpleCarrierTrigger, build_simple_trigger,
    resolve_simple_config,
)
from attack.trigger import MicroDopplerTrigger


@pytest.fixture
def base():
    built = MicroDopplerTrigger(n_ant=3, n_sub=8, n_pkt=4, seed=42, zero_mean=True)
    pattern = np.random.default_rng(7).normal(size=(3, 8, 4))
    pattern -= pattern.mean()
    built.m_zm = pattern / np.sqrt(np.mean(pattern ** 2))
    built.m = built.m_zm.astype(np.complex128)
    return built


@pytest.fixture
def cfg():
    return dict(experiment_name='mmfi', trigger_zero_mean=True, n_ant=3,
                n_sub=8, n_pkt=4, eps=.185, seed=42, simple_variant='additive')


@pytest.fixture
def x():
    return np.random.default_rng(11).uniform(0, 1, (3, 8, 4)).astype(np.float32)


def normalize(pattern):
    centered = np.asarray(pattern, dtype=np.float64) - np.mean(pattern)
    return centered / np.sqrt(np.mean(centered ** 2))


def test_config_resolution_is_pure_and_canonical(cfg):
    original = cfg.copy()
    resolved = resolve_simple_config(cfg)
    assert cfg == original
    assert resolve_simple_config(resolved) == resolved
    assert resolved['simple_schema'] == SCHEMA
    assert resolved['simple_full_profile'] == FULL_PROFILE
    assert 'simple_relative_l2' not in resolved
    assert resolved['simple_reference_eps'] == .185
    assert resolved['simple_sub_mode'] == 3
    assert resolved['simple_time_mode'] == 1
    assert resolved['simple_seed'] == 42
    defaults = resolve_simple_config(dict(experiment_name='mmfi', trigger_zero_mean=True,
                                         simple_variant='single'))
    assert (defaults['n_ant'], defaults['n_sub'], defaults['n_pkt']) == (3, 114, 10)
    reference_run = str(Path(__file__).resolve().parent)
    assert resolve_simple_config(dict(cfg, simple_reference_run=reference_run))['simple_reference_run'] == reference_run


@pytest.mark.parametrize('kwargs', [
    {'experiment_name': 'piw3d'}, {'trigger_zero_mean': False},
    {'trigger_zero_mean': 1}, {'simple_schema': 2}, {'simple_schema': True},
    {'simple_variant': 'adaptive'}, {'simple_variant': None},
    {'simple_sub_mode': 2}, {'simple_time_mode': 0}, {'simple_seed': 43},
    {'simple_seed': True}, {'n_sub': 3}, {'n_pkt': 1}, {'n_ant': 0},
    {'n_sub': 8.5}, {'simple_unknown': 'ignored'},
    {'simple_full_profile': 'draft'}, {'simple_relative_l2': .10}, {'simple_relative_l2': 0},
    {'simple_relative_l2': 1.01}, {'simple_relative_l2': float('nan')},
    {'simple_reference_eps': 0}, {'simple_reference_eps': True},
])
def test_invalid_configuration_fails(cfg, kwargs):
    with pytest.raises(ValueError):
        resolve_simple_config(dict(cfg, **kwargs))


@pytest.mark.parametrize('variant', VARIANTS)
def test_fixed_pattern_float32_identity_nonmutation_and_ranges(base, cfg, x, variant):
    before = x.copy()
    base_before = base.m_zm.copy()
    trigger = SimpleCarrierTrigger(base, dict(cfg, simple_variant=variant))
    pattern = trigger.pattern
    assert pattern.shape == x.shape
    assert np.isfinite(pattern).all()
    assert abs(float(pattern.mean())) < 1e-12
    assert np.mean(pattern ** 2) == pytest.approx(1., abs=1e-12)
    pattern.fill(0)
    assert np.mean(trigger.pattern ** 2) == pytest.approx(1.)
    for dose in (0., .2, .4, .6, .8, 1.):
        out = trigger.inject(x, dose, eps=.185)
        assert out.dtype == np.float32 and out.shape == x.shape
        assert np.isfinite(out).all() and np.all((out >= 0) & (out <= 1))
        assert not np.shares_memory(out, x)
        assert np.array_equal(out, trigger.inject(x, dose, eps=.185))
        if dose == 0:
            assert np.array_equal(out, x)
    assert np.array_equal(trigger.inject(x, 1., eps=0.), x)
    assert np.array_equal(x, before)
    assert np.array_equal(base.m_zm, base_before)
    assert not trigger.requires_deferred_injection
    assert not trigger.requires_event_metadata
    assert not trigger.requires_stochastic_trigger_state


def test_variants_change_exactly_the_declared_factor(base, cfg, x):
    reference = _CarrierDraftTrigger(base, sub_mode=3, time_mode=1, seed=42)
    baseline = normalize((reference.p0 + reference.p1) / np.sqrt(2.))
    triggers = {v: SimpleCarrierTrigger(base, dict(cfg, simple_variant=v)) for v in VARIANTS}
    np.testing.assert_array_equal(triggers['additive'].pattern, baseline)
    np.testing.assert_array_equal(triggers['single'].pattern, reference.p1)
    expected_shared = normalize(np.broadcast_to(baseline.mean(axis=0, keepdims=True), x.shape))
    np.testing.assert_allclose(triggers['shared_links'].pattern, expected_shared, rtol=0, atol=1e-14)
    for link in (1, 2):
        np.testing.assert_array_equal(triggers['shared_links'].pattern[0],
                                      triggers['shared_links'].pattern[link])
    frequency = _CarrierDraftTrigger(base, sub_mode=3, time_mode=0, seed=42)
    expected_frequency = normalize((frequency.p0 + frequency.p1) / np.sqrt(2.))
    np.testing.assert_array_equal(triggers['frequency_only'].pattern, expected_frequency)
    assert abs(np.mean(frequency.p0 * frequency.p1)) < 1e-12
    for v in ('single', 'shared_links', 'frequency_only'):
        np.testing.assert_array_equal(triggers[v].p0, reference.p0)
        assert not np.allclose(triggers[v].pattern, baseline)
        expected = _MethodDraftTrigger._gain_inject(x, triggers[v].pattern, .6 * .185)
        np.testing.assert_array_equal(triggers[v].inject(x, .6, eps=.185), expected)
        assert triggers[v].operator == 'gain'
    for v in ('additive', 'single', 'shared_links'):
        np.testing.assert_array_equal(triggers[v].p1, reference.p1)
    assert triggers['frequency_only'].time_mode == 0
    assert triggers['frequency_only'].cfg['simple_time_mode'] == 1
    assert triggers['frequency_only'].protocol_parameters['second_carrier_time_mode'] == 0


def test_additive_uses_per_input_rms_and_known_numeric_formula(base, cfg, x):
    trigger = SimpleCarrierTrigger(base, cfg)
    dose, eps = .6, .185
    for value in (x, x / 2, np.zeros_like(x), np.ones_like(x)):
        value64 = value.astype(np.float64)
        rms = float(np.sqrt(np.mean(value64 ** 2)))
        expected = np.clip(value64 + dose * eps * rms * trigger.pattern, 0., 1.).astype(np.float32)
        np.testing.assert_array_equal(trigger.inject(value, dose, eps=eps), expected)
    # This interior constant has a known RMS; signed q values remain unclipped.
    flat = np.full(x.shape, .5, dtype=np.float32)
    expected = (.5 + .1 * .2 * .5 * trigger.pattern).astype(np.float32)
    np.testing.assert_array_equal(trigger.inject(flat, .1, eps=.2), expected)
    assert trigger.operator == 'additive_input_rms'


@pytest.mark.parametrize('variant', VARIANTS)
def test_construction_injection_and_hash_preserve_global_rng(base, cfg, x, variant):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    trigger = SimpleCarrierTrigger(base, dict(cfg, simple_variant=variant))
    trigger.inject(x, .6, eps=.185)
    trigger.fixed_key_sha256()
    assert random.getstate() == python_state
    actual = np.random.get_state()
    assert actual[0] == numpy_state[0]
    np.testing.assert_array_equal(actual[1], numpy_state[1])
    assert actual[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_key_binds_variant_operator_pattern_and_returns_deterministically(base, cfg):
    keys = []
    for variant in VARIANTS:
        trigger = SimpleCarrierTrigger(base, dict(cfg, simple_variant=variant))
        key = trigger.fixed_key_sha256()
        assert len(key) == 64 and all(c in '0123456789abcdef' for c in key)
        assert key == SimpleCarrierTrigger(base, dict(cfg, simple_variant=variant)).fixed_key_sha256()
        original_operator = trigger.operator
        trigger.operator = 'different'
        assert trigger.fixed_key_sha256() != key
        trigger.operator = original_operator
        trigger._pattern = -trigger._pattern
        assert trigger.fixed_key_sha256() != key
        keys.append(key)
    assert len(set(keys)) == len(VARIANTS)


@pytest.mark.parametrize('bad_input', [
    np.zeros((3, 8, 5)), np.zeros((1, 3, 8, 4)),
    np.full((3, 8, 4), np.nan), np.full((3, 8, 4), np.inf),
    np.full((3, 8, 4), -.001), np.full((3, 8, 4), 1.001),
    np.ones((3, 8, 4), dtype=complex), np.ones((3, 8, 4), dtype=bool),
    np.full((3, 8, 4), '0.5'),
])
def test_bad_input_fails(base, cfg, bad_input):
    with pytest.raises(ValueError):
        SimpleCarrierTrigger(base, cfg).inject(bad_input, .6, eps=.185)


@pytest.mark.parametrize('dose,eps', [
    (-.1, .185), (1.01, .185), (True, .185), ([.6], .185),
    (float('nan'), .185), (.6, -.1), (.6, True), (.6, float('inf')),
])
def test_bad_strength_fails(base, cfg, x, dose, eps):
    with pytest.raises(ValueError):
        SimpleCarrierTrigger(base, cfg).inject(x, dose, eps=eps)


@pytest.mark.parametrize('variant', VARIANTS)
def test_unrepresentable_strength_fails(base, cfg, variant):
    trigger = SimpleCarrierTrigger(base, dict(cfg, simple_variant=variant))
    with pytest.raises(ValueError, match='finite'):
        trigger.inject(np.ones(trigger.shape, dtype=np.float32), 1., eps=1e308)


def test_bad_base_shape_and_degenerate_pattern_fail(base, cfg):
    with pytest.raises(ValueError, match='shape'):
        SimpleCarrierTrigger(base, dict(cfg, n_sub=9))
    base.m_zm = np.zeros_like(base.m_zm)
    with pytest.raises(ValueError, match='unit RMS'):
        SimpleCarrierTrigger(base, cfg)
    base.m_zm.fill(np.nan)
    with pytest.raises(ValueError, match='finite'):
        SimpleCarrierTrigger(base, cfg)


def test_shared_links_degeneracy_is_an_error(cfg):
    # Both p0 and seeded p1 have zero link mean here, so sharing erases the key.
    built = MicroDopplerTrigger(n_ant=2, n_sub=8, n_pkt=4, seed=42, zero_mean=True)
    one_link = np.random.default_rng(13).normal(size=(8, 4))
    built.m_zm = np.stack([one_link, -one_link])
    built.m_zm /= np.sqrt(np.mean(built.m_zm ** 2))
    with pytest.raises(ValueError, match='degenerate'):
        SimpleCarrierTrigger(built, dict(cfg, n_ant=2, simple_variant='shared_links'))


def test_builder_reuses_original_factory_without_budget_wrapper(base, cfg, monkeypatch):
    calls = []

    def factory(name, resolved):
        calls.append((name, resolved))
        return base

    monkeypatch.setattr('attack.trigger.build_trigger_by_name', factory)
    built = build_simple_trigger(cfg)
    assert type(built) is SimpleCarrierTrigger
    assert built.base is base
    assert len(calls) == 1 and calls[0][0] == 'micro_dropper'
    assert calls[0][1] == resolve_simple_config(cfg)


@pytest.mark.parametrize('variant', VARIANTS)
def test_historical_peak_budget_obeys_actual_cap_for_edge_inputs_and_doses(base, cfg, x, variant):
    native = SimpleCarrierTrigger(base, dict(cfg, simple_variant=variant))
    wrapped = PeakBudgetTrigger(native, base, reference_eps=.185)
    boundary = np.where(x < .5, np.float32(0), np.float32(1))
    subnormal = np.full_like(x, np.nextafter(np.float32(0), np.float32(1)))
    inputs = [x, np.zeros_like(x), np.ones_like(x), boundary, subnormal,
              np.full_like(x, np.nextafter(np.float32(1), np.float32(0)))]
    for value in inputs:
        before = value.copy()
        for dose in (0., 1e-12, .2, .4, .6, .8, 1.):
            out, audit = wrapped.inject_with_audit(value, dose, eps=.185)
            delta = out.astype(np.float64) - value.astype(np.float64)
            reference = base.inject(value, dose, eps=.185).astype(np.float64)
            peak_cap = float(np.max(np.abs(reference - value.astype(np.float64))))
            assert np.max(np.abs(delta)) <= peak_cap
            assert not audit['violation']
            assert 'l2_violation' not in audit
            assert out.dtype == np.float32 and np.isfinite(out).all()
            assert np.all((out >= 0) & (out <= 1))
            if dose == 0:
                np.testing.assert_array_equal(out, value)
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize('variant', VARIANTS)
def test_peak_profile_reuses_historical_projection_without_added_l2_ceiling(base, cfg, x, variant):
    native = SimpleCarrierTrigger(base, dict(cfg, simple_variant=variant))
    wrapped = PeakBudgetTrigger(native, base, reference_eps=.185)
    flat = np.full_like(x, .5)
    candidate = native.inject(flat, 1., eps=.185)
    peak = float(np.max(np.abs(base.inject(flat, 1., eps=.185).astype(np.float64)
                               - flat.astype(np.float64))))
    expected = shrink_to_peak(flat, candidate, peak)
    actual = wrapped.inject(flat, 1., eps=.185)
    np.testing.assert_array_equal(actual, expected)
    relative_l2 = (np.linalg.norm((actual.astype(np.float64) - flat.astype(np.float64)).ravel())
                   / np.linalg.norm(flat.astype(np.float64).ravel()))
    assert relative_l2 > .10
    assert 'simple_relative_l2' not in native.protocol_parameters
