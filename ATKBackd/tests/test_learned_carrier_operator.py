"""CPU-only contracts for learned carriers; no dataset or real victim needed."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack.learned_carrier import (
    DualBudgetTrigger, FrozenLearnedCarrier, TrainableCarrier,
    _strict_project_numpy, artifact_dict, resolve_learned_config, write_artifact,
)
from attack.method_drafts import MDMultiCarrierTrigger
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
                n_sub=8, n_pkt=4, eps=.185, seed=42)


@pytest.fixture
def x():
    return np.random.default_rng(11).uniform(0, 1, size=(3, 8, 4)).astype(np.float32)


def freeze(tmp_path, module, cfg):
    target = tmp_path / 'carrier.json'
    recipe = 'a' * 64
    digest = write_artifact(target, module, recipe, {'training_only': True})
    return FrozenLearnedCarrier(module.base, dict(cfg, lc_artifact_path=str(target),
        lc_artifact_sha256=digest, lc_recipe_sha256=recipe, lc_variant=module.variant))


def assert_caps(base, cfg, value, out, dose):
    delta = out.astype(np.float64) - value.astype(np.float64)
    reference = base.inject(value, dose, eps=cfg.get('lc_reference_eps', .185))
    peak_cap = np.max(np.abs(reference.astype(np.float64) - value.astype(np.float64)))
    l2_cap = cfg.get('lc_relative_l2', .10) * dose * np.linalg.norm(value.astype(np.float64).ravel())
    assert np.max(np.abs(delta)) <= peak_cap
    assert np.linalg.norm(delta.ravel()) <= l2_cap
    assert out.dtype == np.float32
    assert np.all((out >= 0) & (out <= 1))


def test_resolution_is_pure_and_checks_defaults(cfg):
    original = cfg.copy()
    resolved = resolve_learned_config(cfg)
    assert cfg == original
    assert resolved['lc_relative_l2'] == .10
    assert resolved['lc_reference_eps'] == .185
    assert resolved['lc_mask_fraction'] == .25
    assert resolved['lc_carrier_seed'] == 42
    with pytest.raises(ValueError, match='Unknown'):
        resolve_learned_config(dict(cfg, lc_typo=True))


@pytest.mark.parametrize('key,value', [
    ('lc_relative_l2', 0), ('lc_relative_l2', float('nan')),
    ('lc_mask_fraction', 0), ('lc_mask_fraction', 1.1),
    ('lc_reference_eps', True), ('lc_reference_eps', -1),
    ('lc_carrier_sub_mode', 8), ('lc_carrier_time_mode', 4),
    ('lc_carrier_seed', 1.2), ('lc_schema', 2),
    ('lc_variant', 'imaginary'), ('lc_recipe_sha256', 'bad'),
])
def test_bad_settings_fail(cfg, key, value):
    with pytest.raises(ValueError):
        resolve_learned_config(dict(cfg, **{key: value}))


@pytest.mark.parametrize('variant', ['weights', 'sparse', 'combined', 'gradient', 'energy'])
def test_patterns_gradient_and_freeze_parity(base, cfg, x, tmp_path, variant):
    module = TrainableCarrier(base, cfg, variant)
    assert module.export_amplitude() == pytest.approx(.5 if variant == 'energy' else 1.)
    pattern = module.export_pattern()
    assert abs(pattern.mean()) < 1e-10
    assert np.mean(pattern ** 2) == pytest.approx(1)
    if variant in ('sparse', 'combined'):
        groups = np.any(pattern != 0, axis=-1)
        assert groups.sum() == module.mask_groups
        assert np.all(pattern[~groups] == 0)
    frozen = freeze(tmp_path, module, cfg)
    for dose in [0, .2, .4, .6, .8, 1]:
        expected = frozen.inject(x, dose)
        tensor = module.inject_tensor(torch.from_numpy(x), dose)
        actual = tensor.detach().numpy()
        assert_caps(base, cfg, x, expected, dose)
        assert_caps(base, cfg, x, actual, dose)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
        if dose == 0:
            assert np.array_equal(actual, x)
        else:
            loss = (tensor * torch.arange(x.size, dtype=torch.float32).reshape(x.shape)).sum()
            module.zero_grad()
            loss.backward()
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert any(torch.count_nonzero(g) > 0 for g in grads)
    assert len(frozen.fixed_key_sha256()) == 64


def test_batch_doses_and_extreme_inputs(base, cfg, x, tmp_path):
    module = TrainableCarrier(base, cfg, 'combined')
    frozen = freeze(tmp_path, module, cfg)
    inputs = np.stack([np.zeros_like(x), np.ones_like(x), x, x])
    doses = torch.tensor([1., 1., 0., .6])
    actual = module.inject_tensor(torch.from_numpy(inputs), doses).detach().numpy()
    for value, out, dose in zip(inputs, actual, doses.numpy()):
        assert_caps(base, cfg, value, out, float(dose))
        np.testing.assert_allclose(out, frozen.inject(value, float(dose)), rtol=0, atol=2e-7)


def test_dual_control_never_amplifies_and_key_binds_budgets(base, x):
    native = MDMultiCarrierTrigger(base, sub_mode=3, time_mode=1, seed=42)
    wrapper = DualBudgetTrigger(native, base, .185, .10)
    for dose in [0, .2, 1]:
        out, audit = wrapper.inject_with_audit(x, dose, eps=.185)
        assert_caps(base, {}, x, out, dose)
        assert 0 <= audit['shrink_factor'] <= 1
        assert not audit['violation'] and not audit['l2_violation']
        native_delta = native.inject(x, dose, eps=.185).astype(np.float64) - x
        assert np.all(np.abs(out.astype(np.float64) - x) <= np.abs(native_delta))
    other = DualBudgetTrigger(native, base, .185, .09)
    assert wrapper.fixed_key_sha256() != other.fixed_key_sha256()


def test_rounding_guard_handles_tiny_l2_and_boundary_values(x):
    candidate = np.ones_like(x)
    for cap in [0., 1e-100, 1e-12, 1e-7, .05, 100.]:
        out, factor = _strict_project_numpy(x, candidate, .03, cap)
        delta = out.astype(np.float64) - x.astype(np.float64)
        assert np.max(np.abs(delta)) <= .03
        assert np.linalg.norm(delta.ravel()) <= cap
        assert 0 <= factor <= 1


def test_frozen_is_idempotent_under_outer_common_wrapper(base, cfg, x, tmp_path):
    module = TrainableCarrier(base, cfg, 'weights')
    native = freeze(tmp_path, module, cfg)
    wrapper = DualBudgetTrigger(native, base, .185, .10)
    for dose in [0, .2, .6, 1]:
        assert np.array_equal(wrapper.inject(x, dose, eps=.185), native.inject(x, dose, eps=.185))


def test_tiny_tensor_budgets_strict_and_zero_eps_identity(base, cfg, x):
    module = TrainableCarrier(base, dict(cfg, lc_relative_l2=1e-100), 'weights')
    out = module.inject_tensor(torch.from_numpy(x), 1).detach().numpy()
    assert np.array_equal(out, x)
    standard = TrainableCarrier(base, cfg, 'weights')
    assert np.array_equal(standard.inject_tensor(torch.from_numpy(x), 1, eps=0).detach().numpy(), x)


def test_energy_attenuation_has_gradient_and_reduces_actual_l2(base, cfg, x, tmp_path):
    module = TrainableCarrier(base, cfg, 'energy')
    original_tensor = module.inject_tensor(torch.from_numpy(x), 1)
    original = original_tensor.detach().numpy()
    (original_tensor - torch.from_numpy(x)).square().sum().backward()
    assert module.amplitude_logit.grad is not None and module.amplitude_logit.grad > 0
    module.zero_grad()
    with torch.no_grad():
        module.amplitude_logit.fill_(np.log(.2 / .8))
    attenuated = module.inject_tensor(torch.from_numpy(x), 1)
    delta = attenuated - torch.from_numpy(x)
    energy = delta.square().sum()
    energy.backward()
    assert module.amplitude_logit.grad is not None
    assert torch.isfinite(module.amplitude_logit.grad)
    assert module.amplitude_logit.grad > 0
    assert np.linalg.norm(delta.detach().numpy().ravel()) < np.linalg.norm((original - x).ravel())
    frozen = freeze(tmp_path, module, cfg)
    assert frozen.amplitude == pytest.approx(.2)
    assert frozen.protocol_parameters['amplitude'] == pytest.approx(.2)
    assert frozen.protocol_parameters['amplitude_initial'] == .5
    np.testing.assert_allclose(frozen.inject(x, 1), attenuated.detach().numpy(), rtol=0, atol=2e-7)
    assert_caps(base, cfg, x, frozen.inject(x, 1), 1)


@pytest.mark.parametrize('variant,amplitude', [('energy', -1), ('energy', 1.1),
                                             ('energy', float('nan')), ('weights', .5)])
def test_artifact_invalid_amplitude_fails(base, cfg, tmp_path, variant, amplitude):
    module = TrainableCarrier(base, cfg, variant)
    obj = artifact_dict(module, 'a' * 64)
    obj['amplitude'] = amplitude
    data = json.dumps(obj).encode()
    path = tmp_path / 'invalid-amplitude.json'
    path.write_bytes(data)
    with pytest.raises(ValueError, match='amplitude'):
        FrozenLearnedCarrier(base, dict(cfg, lc_artifact_path=str(path),
            lc_artifact_sha256=hashlib.sha256(data).hexdigest(), lc_recipe_sha256='a' * 64))


def test_artifact_write_is_atomic_and_no_temporary_left(base, cfg, tmp_path):
    module = TrainableCarrier(base, cfg, 'weights')
    path = tmp_path / 'key.json'
    first = write_artifact(path, module, 'a' * 64)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first
    second = write_artifact(path, module, 'b' * 64)
    assert second != first
    assert json.loads(path.read_bytes())['recipe_sha256'] == 'b' * 64
    assert list(tmp_path.iterdir()) == [path]


def test_artifact_hash_recipe_and_budget_checks(base, cfg, tmp_path):
    module = TrainableCarrier(base, cfg, 'weights')
    frozen = freeze(tmp_path, module, cfg)
    config = frozen.cfg.copy()
    path = Path(config['lc_artifact_path'])
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == config['lc_artifact_sha256']
    path.write_bytes(data + b' ')
    with pytest.raises(ValueError, match='bytes SHA256'):
        FrozenLearnedCarrier(base, config)
    path.write_bytes(data)
    with pytest.raises(ValueError, match='recipe'):
        FrozenLearnedCarrier(base, dict(config, lc_recipe_sha256='b' * 64))
    with pytest.raises(ValueError, match='configuration'):
        FrozenLearnedCarrier(base, dict(config, lc_relative_l2=.09))
    obj = artifact_dict(module, 'a' * 64)
    obj['pattern'][0][0][0] = float('nan')
    bad = json.dumps(obj).encode()
    path.write_bytes(bad)
    with pytest.raises(ValueError, match='finite'):
        FrozenLearnedCarrier(base, dict(config, lc_artifact_sha256=hashlib.sha256(bad).hexdigest()))


@pytest.mark.parametrize('kind', ['negative', 'nan', 'complex', 'wrong_shape', 'double'])
def test_bad_tensor_inputs_fail(base, cfg, x, kind):
    module = TrainableCarrier(base, cfg, 'weights')
    value = torch.from_numpy(x.copy())
    if kind == 'negative':
        value[0, 0, 0] = -1
    elif kind == 'nan':
        value[0, 0, 0] = float('nan')
    elif kind == 'complex':
        value = value.to(torch.complex64)
    elif kind == 'wrong_shape':
        value = value[:, :-1]
    else:
        value = value.double()
    with pytest.raises(ValueError):
        module.inject_tensor(value, .5)


def test_cover_wrapper_draws_once_and_preserves_native_state(base, x):
    class Cover:
        requires_stochastic_trigger_state = True

        def __init__(self):
            self.draws = 0

        def noise_inject(self, value, seed=None, eps=.3, dose=1.):
            self.draws += 1
            return np.clip(value + eps * dose, 0, 1)

        def fixed_key_sha256(self):
            return 'f' * 64

        def state_dict(self):
            return {'draws': self.draws}

        def load_state_dict(self, state):
            self.draws = state['draws']

    native = Cover()
    wrapper = DualBudgetTrigger(native, base)
    out, audit = wrapper.noise_inject_with_audit(x)
    assert native.draws == 1
    assert not audit['l2_violation']
    assert_caps(base, {}, x, out, 1)
    state = wrapper.state_dict()
    native.draws = 7
    wrapper.load_state_dict(state)
    assert native.draws == 1
    with pytest.raises(ValueError, match='mismatch'):
        wrapper.load_state_dict(dict(state, key_sha256='e' * 64))
