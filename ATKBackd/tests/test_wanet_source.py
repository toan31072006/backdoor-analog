"""Source-formula regression checks; no model training or CSI data needed."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import attack.wanet_source as module
from attack.wanet_source import WaNetSourceTrigger, resolve_wanet_source_config


def _reference(seed=42, height=114, width=10, grid_size=4, strength=0.5,
               rescale=1.0):
    # Independent expression of the pinned author's train.py, with only its
    # square output changed to the native rectangular CSI dimensions.
    generator = torch.Generator().manual_seed(seed)
    key = torch.rand(1, 2, grid_size, grid_size, generator=generator) * 2 - 1
    key = key / torch.mean(torch.abs(key))
    noise = F.interpolate(key, size=(height, width), mode='bicubic',
                          align_corners=True).permute(0, 2, 3, 1)
    y, x = torch.meshgrid(torch.linspace(-1, 1, height),
                          torch.linspace(-1, 1, width), indexing='ij')
    identity = torch.stack((x, y), 2)[None, ...]
    grid = torch.clamp((identity + strength * noise / height) * rescale, -1, 1)
    return generator, key, noise, identity, grid


def _frame(shape=(3, 114, 10)):
    return np.random.default_rng(52).random(shape, dtype=np.float32)


def test_native_grid_and_output_equal_independent_source_expression():
    adapter = WaNetSourceTrigger(seed=42)
    _, key, noise, identity, grid = _reference()
    torch.testing.assert_close(adapter.coarse_grid, key, rtol=0, atol=0)
    torch.testing.assert_close(adapter.noise_grid, noise, rtol=0, atol=0)
    torch.testing.assert_close(adapter.identity_grid, identity, rtol=0, atol=0)
    np.testing.assert_array_equal(adapter.sampling_grid(), grid[0].numpy())
    raw = _frame()
    expected = F.grid_sample(torch.from_numpy(raw)[None], grid,
                             align_corners=True)[0].numpy()
    np.testing.assert_array_equal(adapter.inject(raw, 1, eps=0.185), expected)


def test_no_extra_noise_grid_maxabs_renormalization():
    adapter = WaNetSourceTrigger()
    _, _, reference, _, _ = _reference()
    assert float(reference.abs().max()) > 1.5
    torch.testing.assert_close(adapter.noise_grid, reference, rtol=0, atol=0)
    assert float(adapter.coarse_grid.abs().mean()) == pytest.approx(1, abs=1e-7)


@pytest.mark.parametrize('eps', [0, 0.01, 0.185, 0.3, 100])
def test_shared_amplitude_eps_never_scales_native_warp(eps):
    adapter = WaNetSourceTrigger()
    raw = _frame()
    np.testing.assert_array_equal(adapter.inject(raw, 1, eps), adapter.inject(raw, 1, 1))
    np.testing.assert_array_equal(adapter.sampling_grid(1, eps), adapter.sampling_grid(1, 1))


def test_zero_dose_is_exact_independent_identity_and_does_not_consume_rng():
    adapter = WaNetSourceTrigger()
    raw = _frame()
    rng_before = adapter.state_dict()['cover_rng']
    for operation in (adapter.inject, adapter.noise_inject):
        output = operation(raw, dose=0, eps=0.185)
        np.testing.assert_array_equal(output, raw)
        assert output.dtype == np.float32
        assert output.flags.c_contiguous
        assert not np.shares_memory(output, raw)
    torch.testing.assert_close(adapter.state_dict()['cover_rng'], rng_before, rtol=0, atol=0)
    assert adapter.cover_draws == 0


def test_rectangular_coordinate_axes_and_no_antenna_interpolation():
    adapter = WaNetSourceTrigger(n_ant=3, n_sub=114, n_pkt=10)
    grid = adapter.sampling_grid()
    assert grid.shape == (114, 10, 2)
    identity = adapter.identity_grid[0].numpy()
    np.testing.assert_array_equal(identity[:, :, 0], np.tile(identity[0, :, 0], (114, 1)))
    np.testing.assert_array_equal(identity[:, :, 1], np.tile(identity[:, :1, 1], (1, 10)))
    frequencies = np.linspace(0, 1, 114, dtype=np.float32)[:, None]
    packets = np.linspace(0, 1, 10, dtype=np.float32)[None, :]
    frame = np.stack([np.broadcast_to(frequencies, (114, 10)),
                      np.broadcast_to(packets, (114, 10)), np.full((114, 10), 0.7)])
    output = adapter.inject(frame, 1)
    np.testing.assert_allclose(output[0], (grid[:, :, 1] + 1) / 2, atol=2e-7)
    np.testing.assert_allclose(output[1], (grid[:, :, 0] + 1) / 2, atol=2e-7)
    np.testing.assert_allclose(output[2], 0.7, atol=1e-7)


def test_intermediate_dose_is_explicit_common_task_adaptation():
    adapter = WaNetSourceTrigger()
    _, _, noise, identity, _ = _reference()
    expected = (identity + 0.4 * 0.5 * noise / 114).clamp(-1, 1)
    np.testing.assert_array_equal(adapter.sampling_grid(0.4), expected[0].numpy())
    assert 'task adaptation' in adapter.dose_semantics


def test_source_cover_formula_is_fresh_and_not_sample_seed_frozen():
    adapter = WaNetSourceTrigger(seed=42)
    generator, _, _, _, grid = _reference()
    for _ in range(2):
        jitter = torch.rand(grid.shape, generator=generator) * 2 - 1
        expected = (grid + jitter / 114).clamp(-1, 1)
        actual = adapter.noise_sampling_grid(seed=17, eps=0.185)
        np.testing.assert_array_equal(actual, expected[0].numpy())
    assert adapter.cover_draws == 2


def test_cover_output_matches_grid_sample_and_retains_input_range():
    raw = _frame()
    adapter, duplicate = WaNetSourceTrigger(), WaNetSourceTrigger()
    grid = duplicate.noise_sampling_grid(seed=17)
    expected = F.grid_sample(torch.from_numpy(raw)[None], torch.from_numpy(grid)[None],
                             align_corners=True)[0].numpy()
    output = adapter.noise_inject(raw, seed=17)
    np.testing.assert_array_equal(output, expected)
    assert 0 <= output.min() <= output.max() <= 1
    np.testing.assert_array_equal(raw, _frame())


def test_equal_seed_and_equal_access_history_reproduce_fresh_cover_sequence():
    first, second = WaNetSourceTrigger(seed=42), WaNetSourceTrigger(seed=42)
    raw = _frame()
    previous = None
    for _ in range(3):
        current = first.noise_inject(raw, seed=17)
        np.testing.assert_array_equal(current, second.noise_inject(raw, seed=100))
        if previous is not None:
            assert not np.array_equal(current, previous)
        previous = current


def test_constructor_and_cover_draws_do_not_advance_global_rngs():
    torch_before, numpy_before = torch.get_rng_state(), np.random.get_state()
    adapter = WaNetSourceTrigger()
    adapter.inject(_frame(), dose=1)
    adapter.noise_inject(_frame(), seed=0)
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_checkpoint_resumes_fresh_cover_rng_and_immutable_key():
    first, resumed = WaNetSourceTrigger(), WaNetSourceTrigger()
    raw = _frame()
    key_sha = first.fixed_key_sha256()
    first.noise_inject(raw, seed=17)
    state = first.state_dict()
    expected = first.noise_inject(raw, seed=17)
    resumed.load_state_dict(state)
    np.testing.assert_array_equal(resumed.noise_inject(raw, seed=100), expected)
    assert resumed.cover_draws == 2
    assert first.fixed_key_sha256() == resumed.fixed_key_sha256() == key_sha
    # Caller mutation of a state snapshot cannot modify the adapter key.
    state['noise_grid'].zero_()
    assert first.fixed_key_sha256() == key_sha


def test_checkpoint_rejects_wrong_seed_key_and_bad_rng_type():
    adapter = WaNetSourceTrigger()
    with pytest.raises(ValueError, match='seed'):
        adapter.load_state_dict(WaNetSourceTrigger(seed=7).state_dict())
    state = adapter.state_dict()
    state['noise_grid'][0, 0, 0, 0] += 0.1
    with pytest.raises(ValueError, match='noise_grid'):
        adapter.load_state_dict(state)
    state = adapter.state_dict()
    state['cover_rng'] = state['cover_rng'].float()
    with pytest.raises(ValueError, match='ByteTensor'):
        adapter.load_state_dict(state)


def test_worker_copies_get_distinct_fresh_streams(monkeypatch):
    first, second = WaNetSourceTrigger(), WaNetSourceTrigger()
    monkeypatch.setattr(module, 'get_worker_info', lambda: SimpleNamespace(id=0, seed=100))
    first_grid = first.noise_sampling_grid(seed=17)
    second_first_grid = first.noise_sampling_grid(seed=17)
    assert not np.array_equal(first_grid, second_first_grid)
    monkeypatch.setattr(module, 'get_worker_info', lambda: SimpleNamespace(id=1, seed=101))
    second_grid = second.noise_sampling_grid(seed=17)
    assert not np.array_equal(first_grid, second_grid)
    assert first.cover_draws == 2
    assert second.cover_draws == 1


def test_source_resolver_binds_metadata_and_does_not_mutate_config():
    config = {'experiment_name': 'mmfi'}
    result = resolve_wanet_source_config('wanet_source', config)
    assert config == {'experiment_name': 'mmfi'}
    assert result['wanet_source_commit'] == '45e8c33c285cd55893ae4efcfcf7fe3d26170387'
    assert result['wanet_source_license'] == 'AGPL-3.0'
    assert result['wanet_grid_size'] == 4
    assert result['wanet_strength'] == 0.5
    assert result['wanet_grid_rescale'] == 1
    assert result['num_workers'] == 0
    assert len(result['wanet_adapter_sha256']) == 64
    assert resolve_wanet_source_config('blended', config) == config
    assert resolve_wanet_source_config('wanet_source', result) == result


def test_source_resolver_rejects_stale_marker_and_worker_rng_unresumability():
    with pytest.raises(ValueError, match='metadata mismatch'):
        resolve_wanet_source_config('wanet', {'wanet_implementation': 'legacy'})
    with pytest.raises(ValueError, match='num_workers=0'):
        resolve_wanet_source_config('wanet', {'num_workers': 4})
    with pytest.raises(ValueError, match='MMFi'):
        resolve_wanet_source_config('wanet', {'experiment_name': 'person-in-wifi-3d'})


@pytest.mark.parametrize('kwargs', [
    {'n_ant': 0}, {'n_sub': 1}, {'n_pkt': 1}, {'grid_size': 1},
    {'grid_rescale': 0}, {'strength': -1}, {'strength': float('nan')},
    {'grid_rescale': float('inf')}, {'n_ant': True}, {'seed': 0.5},
])
def test_rejects_invalid_construction(kwargs):
    with pytest.raises(ValueError):
        WaNetSourceTrigger(**kwargs)


@pytest.mark.parametrize('dose,eps', [(-1, 0.3), (1.1, 0.3),
    (float('nan'), 0.3), (0, float('nan')), (1, -1), (True, 0.3)])
def test_rejects_invalid_dose_interface(dose, eps):
    with pytest.raises(ValueError):
        WaNetSourceTrigger().inject(_frame(), dose, eps)


@pytest.mark.parametrize('kind', ['shape', 'complex', 'negative', 'large', 'nan'])
def test_rejects_invalid_amplitude_frames(kind):
    raw = _frame()
    if kind == 'shape':
        raw = raw[:, :, :-1]
    elif kind == 'complex':
        raw = raw.astype(np.complex64)
    else:
        raw[0, 0, 0] = {'negative': -1, 'large': 2, 'nan': np.nan}[kind]
    with pytest.raises(ValueError):
        WaNetSourceTrigger().inject(raw, 1)
