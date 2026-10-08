"""Checkpoint RNG restoration must tolerate CUDA-mapped byte tensors."""

from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_backdoor as trainer


class _CudaMappedState:
    """Simulate map_location='cuda:0' without requiring a CUDA runtime."""

    def __init__(self, cpu_state):
        self.cpu_state = cpu_state
        self.device = torch.device('cuda:0')
        self.dtype = cpu_state.dtype
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return self.cpu_state


def _cpu_rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': None}


def _assert_cpu_rng_state(expected):
    assert random.getstate() == expected['python']
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == expected['numpy'][0]
    np.testing.assert_array_equal(actual_numpy[1], expected['numpy'][1])
    assert actual_numpy[2:] == expected['numpy'][2:]
    assert torch.equal(torch.get_rng_state(), expected['torch'])


def _draw_cpu_randoms():
    return random.random(), np.random.random(4), torch.rand(4)


def _assert_cpu_draws_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def _require_cpu_byte_states(received):
    def setter(states):
        states = list(states)
        if any(not isinstance(state, torch.Tensor)
               or state.device.type != 'cpu' or state.dtype != torch.uint8
               for state in states):
            raise TypeError('RNG state must be a torch.ByteTensor')
        received.append([state.clone() for state in states])
    return setter


@pytest.fixture(autouse=True)
def _preserve_cpu_rng():
    before = _cpu_rng_state()
    yield
    random.setstate(before['python'])
    np.random.set_state(before['numpy'])
    torch.set_rng_state(before['torch'])


@pytest.mark.parametrize('mapped_to_cuda', [True, False])
def test_restore_cuda_rng_normalizes_each_state_to_cpu(monkeypatch, mapped_to_cuda):
    expected_state = _cpu_rng_state()
    expected_draws = _draw_cpu_randoms()
    cuda_states = [torch.tensor([1, 2, 3], dtype=torch.uint8),
                   torch.tensor([7, 8, 9], dtype=torch.uint8)]
    stored_states = ([_CudaMappedState(state) for state in cuda_states]
                     if mapped_to_cuda else cuda_states)
    stored = dict(expected_state, cuda=stored_states)
    received = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all',
                        _require_cpu_byte_states(received))

    trainer._restore_rng_state(stored)

    _assert_cpu_rng_state(expected_state)
    _assert_cpu_draws_equal(_draw_cpu_randoms(), expected_draws)
    assert len(received) == 1
    assert len(received[0]) == len(cuda_states)
    for actual, expected in zip(received[0], cuda_states):
        assert torch.equal(actual, expected)
    if mapped_to_cuda:
        assert all(state.cpu_calls == 1 for state in stored_states)


@pytest.mark.parametrize('cuda_available,has_cuda_state',
                         [(False, True), (False, False), (True, False)])
def test_restore_without_usable_cuda_still_restores_cpu_rng(
        monkeypatch, cuda_available, has_cuda_state):
    expected_state = _cpu_rng_state()
    expected_draws = _draw_cpu_randoms()
    mapped = _CudaMappedState(torch.tensor([11], dtype=torch.uint8))
    stored = dict(expected_state, cuda=[mapped] if has_cuda_state else None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all',
                        lambda states: pytest.fail('unexpected CUDA RNG setter'))

    trainer._restore_rng_state(stored)

    _assert_cpu_rng_state(expected_state)
    _assert_cpu_draws_equal(_draw_cpu_randoms(), expected_draws)
    assert mapped.cpu_calls == 0


@pytest.mark.parametrize('stored', [None, {}])
def test_missing_rng_state_keeps_current_generators(stored, capsys):
    expected = _cpu_rng_state()

    trainer._restore_rng_state(stored)

    _assert_cpu_rng_state(expected)
    assert 'no RNG state stored' in capsys.readouterr().out


def test_load_checkpoint_restores_cuda_mapped_rng(monkeypatch, tmp_path):
    cfg = dict(model='tiny', epochs=2, lr=0.02)
    source = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=cfg['lr'])
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.9)
    expected_rng = _cpu_rng_state()
    expected_draws = _draw_cpu_randoms()
    cpu_cuda_states = [torch.tensor([19, 23], dtype=torch.uint8),
                       torch.tensor([29, 31], dtype=torch.uint8)]
    mapped_cpu = _CudaMappedState(expected_rng['torch'])
    mapped_cuda = [_CudaMappedState(state) for state in cpu_cuda_states]
    checkpoint = {
        'epoch': 0, 'best_loss': 0.125, 'model': source.state_dict(),
        'optimizer': source_optimizer.state_dict(),
        'cfg_fingerprint': trainer._config_fingerprint(cfg),
        'rng': dict(expected_rng, torch=mapped_cpu, cuda=mapped_cuda),
    }
    path = tmp_path / 'checkpoint.pt'
    device = torch.device('cuda:0')
    load_calls = []
    received = []

    def load_checkpoint(actual_path, *, map_location, weights_only):
        load_calls.append((actual_path, map_location, weights_only))
        return checkpoint

    monkeypatch.setattr(torch, 'load', load_checkpoint)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all',
                        _require_cpu_byte_states(received))

    result = trainer._load_checkpoint(
        path, restored, restored_optimizer, device, cfg)

    assert result == (0, 0.125)
    assert load_calls == [(path, device, False)]
    for actual, expected in zip(restored.parameters(), source.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored_optimizer.param_groups[0]['lr'] == cfg['lr']
    _assert_cpu_rng_state(expected_rng)
    _assert_cpu_draws_equal(_draw_cpu_randoms(), expected_draws)
    assert mapped_cpu.cpu_calls == 1
    assert all(state.cpu_calls == 1 for state in mapped_cuda)
    assert len(received) == 1
    assert len(received[0]) == len(cpu_cuda_states)
    for actual, expected in zip(received[0], cpu_cuda_states):
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_real_cuda_checkpoint_resume_restores_random_streams(tmp_path):
    device = torch.device('cuda:0')
    before_cuda = torch.cuda.get_rng_state_all()
    try:
        cfg = dict(model='tiny', epochs=2, lr=0.02)
        source = torch.nn.Linear(2, 1).to(device)
        source_optimizer = torch.optim.SGD(source.parameters(), lr=cfg['lr'])
        inputs = torch.tensor([[0.25, -0.5]], device=device)
        expected_output = source(inputs).detach()
        checkpoint = tmp_path / 'cuda-checkpoint.pt'
        trainer._save_checkpoint(
            checkpoint, source, source_optimizer, 0, 0.125, cfg)
        expected_cpu_draws = _draw_cpu_randoms()
        expected_cuda_draws = torch.rand(4, device=device)

        mapped = torch.load(checkpoint, map_location=device, weights_only=False)
        assert mapped['rng']['torch'].device == device
        assert all(state.device == device for state in mapped['rng']['cuda'])
        restored = torch.nn.Linear(2, 1).to(device)
        restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.9)

        result = trainer._load_checkpoint(
            checkpoint, restored, restored_optimizer, device, cfg)

        assert result == (0, 0.125)
        assert restored_optimizer.param_groups[0]['lr'] == cfg['lr']
        torch.testing.assert_close(restored(inputs), expected_output, rtol=0, atol=0)
        _assert_cpu_draws_equal(_draw_cpu_randoms(), expected_cpu_draws)
        torch.testing.assert_close(torch.rand(4, device=device), expected_cuda_draws,
                                   rtol=0, atol=0)
    finally:
        torch.cuda.set_rng_state_all(before_cuda)
