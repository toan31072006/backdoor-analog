"""CPU contracts for the new train-aware bank; no MM-Fi data or real victim."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack.learned_carrier import (
    BANK_VARIANTS, FrozenLearnedCarrier, TrainableCarrier, artifact_dict,
    operator_config_keys, resolve_learned_config, write_artifact,
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
                n_sub=8, n_pkt=4, eps=.185, seed=42)


def freeze(tmp_path, module):
    path = tmp_path / 'key.json'
    digest = write_artifact(path, module, 'a' * 64, {'training_only': True})
    config = dict(module.cfg, lc_artifact_path=str(path), lc_artifact_sha256=digest,
                  lc_recipe_sha256='a' * 64)
    return FrozenLearnedCarrier(module.base, config), path, config


def assert_budgets(base, x, out, dose):
    delta = out.astype(np.float64) - x.astype(np.float64)
    reference = base.inject(x, dose, eps=.185).astype(np.float64)
    assert np.max(np.abs(delta)) <= np.max(np.abs(reference - x.astype(np.float64)))
    assert np.linalg.norm(delta.ravel()) <= .10 * dose * np.linalg.norm(x.astype(np.float64).ravel())
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert np.all((out >= 0) & (out <= 1))


def test_resolution_preserves_legacy_defaults_and_artifact(base, cfg):
    legacy = resolve_learned_config(cfg)
    assert 'lc_bank_size' not in legacy and 'lc_bank_seed' not in legacy
    module = TrainableCarrier(base, cfg, 'weights')
    obj = artifact_dict(module, 'a' * 64)
    assert 'raw_weights' not in obj and 'bank_diagnostics' not in obj
    assert len(operator_config_keys(legacy)) == 6
    assert set(obj['operator_config']) == set(operator_config_keys(legacy))
    for variant in BANK_VARIANTS:
        resolved = resolve_learned_config(dict(cfg, lc_variant=variant))
        assert resolved['lc_bank_size'] == (2 if variant == 'trainaware' else 8)
        assert resolved['lc_bank_seed'] == 42
        assert len(operator_config_keys(resolved)) == 8


@pytest.mark.parametrize('variant,size', [('trainaware', 8), ('bank', 2), ('bank_guard', 4)])
def test_bank_size_is_part_of_variant_contract(cfg, variant, size):
    with pytest.raises(ValueError, match='lc_bank_size'):
        resolve_learned_config(dict(cfg, lc_variant=variant, lc_bank_size=size))


@pytest.mark.parametrize('kwargs', [
    {'lc_variant': 'control', 'lc_bank_size': 8},
    {'lc_variant': 'weights', 'lc_bank_seed': 42},
    {'lc_variant': 'bank', 'lc_bank_seed': True},
    {'lc_variant': 'bank', 'lc_bank_seed': -1},
    {'lc_variant': 'bank', 'lc_bank_size': 8.5},
])
def test_bank_configuration_rejects_ambiguous_or_invalid_settings(cfg, kwargs):
    with pytest.raises(ValueError):
        resolve_learned_config(dict(cfg, **kwargs))


def test_variant_argument_cannot_disagree_with_configuration(base, cfg):
    with pytest.raises(ValueError, match='disagrees'):
        TrainableCarrier(base, dict(cfg, lc_variant='bank_guard'), 'bank')


@pytest.mark.parametrize('variant', BANK_VARIANTS)
def test_initial_pattern_is_equal_and_bank_is_independent(base, cfg, variant):
    original = TrainableCarrier(base, cfg, 'weights')
    module = TrainableCarrier(base, cfg, variant)
    bank = module.carrier_bank.detach().numpy()
    assert len(bank) == (2 if variant == 'trainaware' else 8)
    assert np.array_equal(bank[0], original.p0.numpy())
    assert np.array_equal(bank[1], original.p1.numpy())
    np.testing.assert_allclose(module.export_pattern(), original.export_pattern(), rtol=0, atol=1e-14)
    flat = bank.reshape(len(bank), -1)
    np.testing.assert_allclose(flat @ flat.T / flat.shape[1], np.eye(len(bank)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(bank.mean(axis=(1, 2, 3)), 0, rtol=0, atol=1e-14)
    assert module.export_amplitude() == 1
    assert not module.mask_scores.requires_grad
    assert module.weights.requires_grad
    assert module.bank_diagnostics['max_abs_gram_error'] < 1e-12


def test_bank_seed_is_local_deterministic_and_preserves_first_two(base, cfg):
    np.random.seed(119)
    expected = np.random.random(4)
    np.random.seed(119)
    first = TrainableCarrier(base, dict(cfg, lc_bank_seed=101), 'bank')
    actual = np.random.random(4)
    same = TrainableCarrier(base, dict(cfg, lc_bank_seed=101), 'bank')
    other = TrainableCarrier(base, dict(cfg, lc_bank_seed=102), 'bank')
    assert np.array_equal(expected, actual)
    assert torch.equal(first.carrier_bank, same.carrier_bank)
    assert torch.equal(first.carrier_bank[:2], other.carrier_bank[:2])
    assert not torch.equal(first.carrier_bank[2:], other.carrier_bank[2:])
    assert first.bank_diagnostics['basis_sha256'] != other.bank_diagnostics['basis_sha256']


@pytest.mark.parametrize('variant', BANK_VARIANTS)
def test_signed_weights_gradient_artifact_parity_and_strict_budgets(base, cfg, tmp_path, variant):
    module = TrainableCarrier(base, cfg, variant)
    with torch.no_grad():
        module.weights.copy_(torch.linspace(-.7, 1.1, len(module.weights), dtype=torch.float64))
    frozen, path, config = freeze(tmp_path, module)
    assert frozen.raw_weights[0] < 0
    assert frozen.bank_diagnostics == module.bank_diagnostics
    obj = json.loads(path.read_bytes())
    assert obj['operator_config']['lc_bank_size'] == len(module.weights)
    assert obj['raw_weights'] == module.weights.detach().tolist()
    assert frozen.protocol_parameters['lc_bank_size'] == len(module.weights)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config['lc_artifact_sha256']
    random = np.random.default_rng(123).uniform(0, 1, (3, 8, 4)).astype(np.float32)
    inputs = np.stack([random, np.zeros_like(random), np.ones_like(random)])
    for dose in [0, .2, .4, .6, .8, 1]:
        tensor = module.inject_tensor(torch.from_numpy(inputs), dose)
        actual = tensor.detach().numpy()
        for x, out in zip(inputs, actual):
            assert_budgets(base, x, out, dose)
            expected = frozen.inject(x, dose)
            np.testing.assert_allclose(out, expected, rtol=0, atol=2e-7)
            assert_budgets(base, x, expected, dose)
            if not dose:
                assert np.array_equal(out, x)
        if dose:
            module.zero_grad()
            loss = (tensor[0] * torch.arange(random.size).reshape(random.shape)).sum()
            loss.backward()
            assert module.weights.grad is not None
            assert torch.isfinite(module.weights.grad).all()
            assert torch.count_nonzero(module.weights.grad) > 0


def test_new_bank_coefficients_receive_gradient_at_common_initialization(base, cfg):
    module = TrainableCarrier(base, cfg, 'bank')
    pattern = module.pattern_tensor()
    (pattern * module.carrier_bank[7]).mean().backward()
    assert module.weights.grad[7].abs() > 0
    assert torch.isfinite(module.weights.grad).all()


@pytest.mark.parametrize('kind', ['weights', 'size', 'diagnostics', 'seed', 'string', 'bool', 'nan'])
def test_bank_artifact_is_self_consistent_and_configuration_bound(base, cfg, tmp_path, kind):
    module = TrainableCarrier(base, cfg, 'bank')
    _, path, config = freeze(tmp_path, module)
    obj = json.loads(path.read_bytes())
    if kind == 'weights':
        obj['raw_weights'][3] = .8
    elif kind == 'size':
        obj['raw_weights'].pop()
    elif kind == 'diagnostics':
        obj['bank_diagnostics']['basis_sha256'] = '0' * 64
    elif kind == 'seed':
        config['lc_bank_seed'] += 1
    elif kind == 'string':
        obj['raw_weights'] = [str(value) for value in obj['raw_weights']]
    elif kind == 'bool':
        obj['raw_weights'] = [True] * 8
    else:
        obj['raw_weights'][0] = float('nan')
    data = json.dumps(obj).encode()
    path.write_bytes(data)
    config['lc_artifact_sha256'] = hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError, match='weights|diagnostics|configuration'):
        FrozenLearnedCarrier(base, config)


def test_zero_bank_pattern_fails_without_silent_identity_fallback(base, cfg):
    module = TrainableCarrier(base, cfg, 'bank')
    with torch.no_grad():
        module.weights.zero_()
    with pytest.raises(ValueError, match='degenerate'):
        module.export_pattern()


def test_original_and_bank_caps_do_not_depend_on_learned_coefficients(base, cfg):
    module = TrainableCarrier(base, cfg, 'bank')
    x = np.random.default_rng(37).uniform(0, 1, (3, 8, 4)).astype(np.float32)
    for coefficients in [torch.linspace(-2, 2, 8), torch.linspace(3, -1, 8)]:
        with torch.no_grad():
            module.weights.copy_(coefficients)
        out = module.inject_tensor(torch.from_numpy(x), 1).detach().numpy()
        assert_budgets(base, x, out, 1)


def test_actual_mmfi_axes_and_all_extra_weights_have_independent_directions(tmp_path):
    shape = (3, 114, 10)
    base = MicroDopplerTrigger(n_ant=shape[0], n_sub=shape[1], n_pkt=shape[2],
                               seed=42, zero_mean=True)
    pattern = np.random.default_rng(47).normal(size=shape)
    pattern -= pattern.mean()
    base.m_zm = pattern / np.sqrt(np.mean(pattern ** 2))
    base.m = base.m_zm.astype(np.complex128)
    cfg = dict(experiment_name='mmfi', trigger_zero_mean=True,
               n_ant=3, n_sub=114, n_pkt=10, eps=.185)
    module = TrainableCarrier(base, cfg, 'bank_guard')
    assert module.carrier_bank.shape == (8, *shape)
    for index in range(2, 8):
        module.zero_grad()
        loss = (module.pattern_tensor() * module.carrier_bank[index]).mean()
        loss.backward()
        assert module.weights.grad[index].abs() > .1
    frozen, _, _ = freeze(tmp_path, module)
    x = np.random.default_rng(83).uniform(0, 1, shape).astype(np.float32)
    actual = frozen.inject(x, .6)
    assert_budgets(base, x, actual, .6)
    np.testing.assert_allclose(actual, module.inject_tensor(torch.from_numpy(x), .6).detach().numpy(),
                               atol=2e-7, rtol=0)
