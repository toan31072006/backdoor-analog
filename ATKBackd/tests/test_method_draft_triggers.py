"""Contracts for draft trigger mechanisms, independent of attack efficacy."""

import json
import multiprocessing as mp
import pickle
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack.method_drafts import (METHOD_DRAFT_NAMES, MDMultiCarrierTrigger,
                                 MDDoseCodeTrigger, MDEnergyTrigger,
                                 MDPeakMatchedMultiCarrierTrigger,
                                 build_method_draft_trigger,
                                 resolve_method_draft_config)
from attack.trigger import MicroDopplerTrigger, build_trigger_by_name


def _base(n_ant=3, n_sub=16, n_pkt=8, zero_mean=True):
    trigger = MicroDopplerTrigger(n_ant=n_ant, n_sub=n_sub, n_pkt=n_pkt,
                                 seed=7, zero_mean=zero_mean)
    rng = np.random.default_rng(8)
    trigger.build(rng.normal(scale=0.4, size=(4, 40)),
                  rng.uniform(-0.5, 0.5, size=4))
    return trigger


def _draft(name):
    return {'md_multicarrier': MDMultiCarrierTrigger,
            'md_multicarrier_peak_matched': MDPeakMatchedMultiCarrierTrigger,
            'md_dose_code': MDDoseCodeTrigger,
            'md_energy': MDEnergyTrigger}[name](_base())


def _amplitudes(shape):
    return np.random.default_rng(13).uniform(0.15, 0.85, size=shape).astype(np.float32)


def _spawn_inject(connection, trigger, csi):
    """An importable worker proves the actual spawn path, beyond pickle dumps."""
    try:
        connection.send(trigger.inject(csi, dose=0.6, eps=0.185))
    finally:
        connection.close()


@pytest.mark.parametrize('name', METHOD_DRAFT_NAMES)
def test_shape_dtype_range_identity_and_no_input_mutation(name):
    trigger = _draft(name)
    csi = _amplitudes(trigger.shape)
    before = csi.copy()
    for dose in (0.0, 0.2, 0.6, 1.0):
        out = trigger.inject(csi, dose, eps=0.185)
        assert out.shape == csi.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert 0.0 <= out.min() <= out.max() <= 1.0
        assert not np.shares_memory(out, csi)
        if dose == 0:
            assert np.array_equal(out, csi)
        else:
            assert not np.array_equal(out, csi)
    assert np.array_equal(trigger.inject(csi, 1.0, eps=0.0), csi)
    assert np.array_equal(csi, before)


@pytest.mark.parametrize('name', METHOD_DRAFT_NAMES)
def test_float64_and_noncontiguous_inputs(name):
    trigger = _draft(name)
    contiguous = _amplitudes(trigger.shape)
    view = np.ascontiguousarray(contiguous.transpose(2, 1, 0)).transpose(2, 1, 0)
    assert not view.flags.c_contiguous
    expected = trigger.inject(contiguous, 0.6, 0.185)
    assert np.array_equal(trigger.inject(view, 0.6, 0.185), expected)
    assert np.array_equal(trigger.inject(view.astype(np.float64), 0.6, 0.185), expected)
    assert np.array_equal(trigger.inject(view.astype(np.float64), 0.0), contiguous)


@pytest.mark.parametrize('name', METHOD_DRAFT_NAMES)
def test_deterministic_and_global_rng_independent(name):
    original = np.random.get_state()
    try:
        np.random.seed(41)
        before = np.random.get_state()
        trigger = _draft(name)
        csi = _amplitudes(trigger.shape)
        expected = trigger.inject(csi, 0.6, 0.185)
        after = np.random.get_state()
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]
        np.random.seed(999)
        np.random.random(100)
        other = _draft(name)
        assert np.array_equal(other.inject(csi, 0.6, 0.185), expected)
        trigger.inject(csi, 0.2, 0.5)
        assert np.array_equal(trigger.inject(csi, 0.6, 0.185), expected)
    finally:
        np.random.set_state(original)


@pytest.mark.parametrize('constructor', [MDMultiCarrierTrigger, MDDoseCodeTrigger])
def test_carrier_gram_orthogonality_zero_mean_unit_rms(constructor):
    base = _base()
    before = base.m_zm.copy()
    trigger = constructor(base, sub_mode=3, time_mode=1, seed=42)
    assert np.array_equal(base.m_zm, before)
    assert np.array_equal(trigger.p0, before)
    for p in (trigger.p0, trigger.p1):
        assert abs(p.mean()) < 1e-12
        assert np.sqrt(np.mean(p ** 2)) == pytest.approx(1.0, abs=1e-10)
    assert abs(np.mean(trigger.p0 * trigger.p1)) < 1e-12
    if isinstance(trigger, MDMultiCarrierTrigger):
        p = trigger.pattern
        assert abs(p.mean()) < 1e-12
        assert np.sqrt(np.mean(p ** 2)) == pytest.approx(1.0, abs=1e-12)
        expected = (trigger.p0 + trigger.p1) / np.sqrt(2.0)
        expected /= np.sqrt(np.mean(expected ** 2))
        np.testing.assert_array_equal(p, expected)


def test_multicarrier_uses_original_gain_and_both_clip_stages():
    trigger = MDMultiCarrierTrigger(_base())
    csi = np.linspace(0.25, 1.0, np.prod(trigger.shape), dtype=np.float32).reshape(trigger.shape)
    dose, eps = 1.0, 3.0
    raw_gain = 1.0 + dose * eps * trigger.pattern
    assert raw_gain.min() < 0.0
    gain = np.clip(raw_gain, 0.0, None).astype(np.float32)
    assert (csi * gain).max() > 1.0
    assert np.array_equal(trigger.inject(csi, dose, eps), np.clip(csi * gain, 0.0, 1.0))


def test_dose_code_constant_pattern_rms_exact_endpoint_and_distinct_inner_doses():
    base = _base()
    trigger = MDDoseCodeTrigger(base)
    csi = _amplitudes(trigger.shape)
    for dose in np.linspace(0.0, 1.0, 17):
        p = trigger.pattern(dose)
        assert abs(p.mean()) < 1e-12
        assert np.sqrt(np.mean(p ** 2)) == pytest.approx(1.0, abs=1e-10)
    for eps in (0.0, 0.185, 0.8, 3.0):
        assert np.array_equal(trigger.inject(csi, 1.0, eps), base.inject(csi, 1.0, eps))
    assert np.array_equal(trigger.inject(csi, 0.0, 0.185), csi)
    dose, eps = 0.6, 0.185
    assert not np.allclose(trigger.pattern(dose), base.m_zm)
    assert not np.array_equal(trigger.inject(csi, dose, eps), base.inject(csi, dose, eps))
    p = trigger.pattern(dose)
    ref = np.clip(csi * np.clip(1.0 + dose * eps * p, 0.0, None).astype(np.float32), 0.0, 1.0)
    assert np.array_equal(trigger.inject(csi, dose, eps), ref)
    angle = trigger.angle_max_rad * (1.0 - dose)
    assert np.mean(p * trigger.p0) == pytest.approx(np.cos(angle), abs=1e-10)
    assert np.mean(p * trigger.p1) == pytest.approx(np.sin(angle), abs=1e-10)


def test_energy_theoretical_preclip_norm_and_centered_direction():
    trigger = MDEnergyTrigger(_base())
    csi = _amplitudes(trigger.shape)
    dose, eps = 0.6, 1e-3
    delta = trigger.perturbation(csi, dose, eps)
    x = csi.astype(np.float64)
    direction = x * trigger.p0
    direction -= direction.mean()
    expected = dose * eps * direction * np.linalg.norm(x.ravel()) / np.linalg.norm(direction.ravel())
    np.testing.assert_allclose(delta, expected, rtol=1e-12, atol=1e-15)
    assert abs(delta.mean()) < 1e-16
    assert np.linalg.norm(delta.ravel()) / np.linalg.norm(x.ravel()) == pytest.approx(dose * eps, rel=1e-12)
    preclip = x + delta
    assert preclip.min() > 0.0 and preclip.max() < 1.0
    out = trigger.inject(csi, dose, eps)
    assert np.array_equal(out, preclip.astype(np.float32))
    assert np.linalg.norm((out.astype(np.float64) - x).ravel()) / np.linalg.norm(x.ravel()) == pytest.approx(dose * eps, rel=1e-4)


@pytest.mark.parametrize('dose,eps', [(0.2, 0.185), (1.0, 0.185), (1.0, 3.0)])
def test_energy_postclip_relative_norm_does_not_exceed_budget(dose, eps):
    trigger = MDEnergyTrigger(_base())
    csi = np.linspace(0.0, 1.0, np.prod(trigger.shape), dtype=np.float32).reshape(trigger.shape)
    out = trigger.inject(csi, dose, eps)
    actual = np.linalg.norm((out.astype(np.float64) - csi).ravel())
    assert actual <= dose * eps * np.linalg.norm(csi.astype(np.float64).ravel()) + 2e-6
    if eps == 3.0:
        preclip = csi + trigger.perturbation(csi, dose, eps)
        assert preclip.min() < 0.0 or preclip.max() > 1.0


def test_energy_zero_and_degenerate_direction_are_identity():
    trigger = MDEnergyTrigger(_base())
    zero = np.zeros(trigger.shape, dtype=np.float32)
    assert np.array_equal(trigger.inject(zero, 1.0, 0.185), zero)
    assert not trigger.perturbation(zero, 1.0).any()
    # Valid zero-mean/unit-RMS p0 with zeros: input supported only where p0=0.
    base = _base()
    p0 = np.zeros(base.m_zm.shape)
    p0.flat[0], p0.flat[1] = 1.0, -1.0
    p0 /= np.sqrt(np.mean(p0 ** 2))
    base.m_zm = p0
    degenerate = MDEnergyTrigger(base)
    csi = np.full(degenerate.shape, 0.5, dtype=np.float32)
    csi.flat[:2] = 0.0
    assert np.array_equal(degenerate.inject(csi, 1.0), csi)
    assert not degenerate.perturbation(csi, 1.0).any()


@pytest.mark.parametrize('name', METHOD_DRAFT_NAMES)
@pytest.mark.parametrize('invalid', ['shape', 'piw', 'nan', 'inf', 'negative', 'above_one', 'complex', 'object', 'bool'])
def test_bad_amplitude_inputs_refuse_even_at_zero_dose(name, invalid):
    trigger = _draft(name)
    csi = _amplitudes(trigger.shape)
    if invalid == 'shape':
        csi = csi[0]
    elif invalid == 'piw':
        csi = np.ones((trigger.n_ant, 6 * trigger.n_sub, trigger.n_pkt), dtype=np.float32)
    elif invalid in ('complex', 'object', 'bool'):
        csi = csi.astype({'complex': np.complex64, 'object': object, 'bool': bool}[invalid])
    else:
        csi.flat[0] = {'nan': np.nan, 'inf': np.inf, 'negative': -0.01, 'above_one': 1.01}[invalid]
    with pytest.raises(ValueError):
        trigger.inject(csi, 0.0)


@pytest.mark.parametrize('name', METHOD_DRAFT_NAMES)
@pytest.mark.parametrize('field,value', [('dose', -0.01), ('dose', 1.01), ('dose', np.nan),
                                        ('dose', np.inf), ('dose', [0.2]), ('dose', True),
                                        ('dose', np.complex128(0.2)), ('dose', '0.2'),
                                        ('eps', -1.0), ('eps', np.nan), ('eps', np.inf),
                                        ('eps', [0.1]), ('eps', True),
                                        ('eps', np.complex128(0.1)), ('eps', '0.1')])
def test_bad_scalars_refuse(name, field, value):
    trigger = _draft(name)
    kwargs = dict(dose=0.6, eps=0.185)
    kwargs[field] = value
    with pytest.raises(ValueError):
        trigger.inject(_amplitudes(trigger.shape), **kwargs)


@pytest.mark.parametrize('constructor', [MDMultiCarrierTrigger, MDDoseCodeTrigger, MDEnergyTrigger,
                                      MDPeakMatchedMultiCarrierTrigger])
def test_unbuilt_nonzero_mean_or_invalid_base_refused(constructor):
    with pytest.raises(ValueError):
        constructor(_base(zero_mean=False))
    with pytest.raises(ValueError):
        constructor(MicroDopplerTrigger(n_sub=16, n_pkt=8, zero_mean=True))
    with pytest.raises(ValueError):
        constructor(object())
    base = _base()
    base.m_zm = np.zeros_like(base.m_zm)
    with pytest.raises(ValueError):
        constructor(base)


@pytest.mark.parametrize('name', METHOD_DRAFT_NAMES)
def test_overflow_strength_is_refused_without_nonfinite_output(name):
    trigger = _draft(name)
    with pytest.raises(ValueError, match='finite'):
        trigger.inject(_amplitudes(trigger.shape), 1.0, eps=1e308)


@pytest.fixture
def cfg(tmp_path):
    skeleton = tmp_path / 'action.npy'
    np.save(skeleton, np.random.default_rng(22).normal(size=(2, 3, 50, 25, 1)))
    return dict(experiment_name='mmfi', trigger_zero_mean=True,
                action_npy=str(skeleton), n_ant=2, n_sub=16, n_pkt=8,
                seed=42, top_k=4)


@pytest.mark.parametrize('name', METHOD_DRAFT_NAMES)
def test_builder_inherits_baseline_shape_and_p0_and_exposes_json_metadata(name, cfg):
    original_cfg = cfg.copy()
    trigger = build_method_draft_trigger(name, cfg)
    base = build_trigger_by_name('micro_dropper', cfg)
    assert trigger.shape == (2, 16, 8)
    assert np.array_equal(trigger.p0, base.m_zm)
    assert cfg == original_cfg
    metadata = trigger.protocol_parameters
    assert metadata['trigger'] == name
    assert metadata['method_draft_trigger_schema'] == 1
    assert metadata['experimental_hypothesis'] is True
    json.dumps(metadata)


def test_resolver_defaults_and_baseline_noop(cfg):
    baseline = dict(cfg, trigger='micro_dropper')
    assert resolve_method_draft_config(baseline) == baseline
    assert resolve_method_draft_config(baseline) is not baseline
    resolved = resolve_method_draft_config(dict(cfg, trigger='md_dose_code'))
    assert resolved['method_carrier_sub_mode'] == 3
    assert resolved['method_carrier_time_mode'] == 1
    assert resolved['method_carrier_seed'] == cfg['seed']
    assert resolved['method_dose_angle_max_deg'] == 90.0
    assert resolved['method_draft_trigger_schema'] == 1


@pytest.mark.parametrize('overrides', [dict(experiment_name='person-in-wifi-3d'),
                                     dict(trigger_zero_mean=False), dict(trigger_zero_mean='true'),
                                     dict(method_draft_trigger_schema=2), dict(method_unknown=1),
                                     dict(method_carrier_sub_mode=16), dict(method_carrier_time_mode=8),
                                     dict(method_carrier_sub_mode=0, method_carrier_time_mode=0),
                                     dict(method_carrier_sub_mode=1.5), dict(method_carrier_seed=-1),
                                     dict(method_dose_angle_max_deg=0), dict(method_dose_angle_max_deg=181),
                                     dict(method_dose_angle_max_deg=np.nan)])
def test_builder_and_resolver_refuse_invalid_configs(overrides, cfg):
    bad = dict(cfg, trigger='md_dose_code', **overrides)
    with pytest.raises(ValueError):
        resolve_method_draft_config(bad)
    with pytest.raises(ValueError):
        build_method_draft_trigger('md_dose_code', bad)


def test_carrier_settings_change_pattern_and_dose_angle_changes_inner_code(cfg):
    first = build_method_draft_trigger('md_multicarrier', cfg)
    second = build_method_draft_trigger('md_multicarrier', dict(cfg, method_carrier_sub_mode=4))
    assert not np.allclose(first.p1, second.p1)
    code90 = build_method_draft_trigger('md_dose_code', cfg)
    code45 = build_method_draft_trigger('md_dose_code', dict(cfg, method_dose_angle_max_deg=45.0))
    assert not np.allclose(code90.pattern(0.6), code45.pattern(0.6))
    assert np.array_equal(code90.pattern(1.0), code45.pattern(1.0))
    with pytest.raises(ValueError, match='Unknown'):
        build_method_draft_trigger('unknown', cfg)


@pytest.mark.parametrize('name', METHOD_DRAFT_NAMES)
def test_pickle_and_spawn_preserve_injection(name):
    trigger = _draft(name)
    csi = _amplitudes(trigger.shape)
    expected = trigger.inject(csi, 0.6, 0.185)
    restored = pickle.loads(pickle.dumps(trigger))
    assert np.array_equal(restored.inject(csi, 0.6, 0.185), expected)
    assert restored.protocol_parameters == trigger.protocol_parameters
    context = mp.get_context('spawn')
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_spawn_inject, args=(sending, trigger, csi))
    process.start()
    sending.close()
    try:
        assert receiving.poll(45), 'Spawned worker did not return its injected sample'
        actual = receiving.recv()
        process.join(timeout=5)
        assert process.exitcode == 0
        assert np.array_equal(actual, expected)
    finally:
        receiving.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
