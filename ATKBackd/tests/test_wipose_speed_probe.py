"""Bounded, CPU/synthetic contracts for the additive WiPose speed diagnostic.

These checks do not launch scientific training, use production datasets, or
change the frozen WiPose runner, model, inputs, or historical artifacts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np
import pytest
import torch

import probe_wipose_speed as probe
import wipose_data


class TinyPose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 36)
        # The production finite-gradient guard deliberately ignores None grads.
        self.unused = torch.nn.Parameter(torch.ones(1))
        self.register_buffer('counter', torch.zeros(1, dtype=torch.int64))

    def forward(self, x):
        features = x.mean(dim=(2, 3))
        return self.head(features).reshape(-1, 1, 18, 2), features


def tiny_batch():
    generator = torch.Generator().manual_seed(1234)
    return (torch.rand(2, 3, 90, 5, generator=generator),
            torch.rand(2, 1, 18, 2, generator=generator))


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def tree(root):
    return {path.relative_to(root).as_posix():
            (path.stat().st_size, path.stat().st_mtime_ns,
             hashlib.sha256(path.read_bytes()).hexdigest())
            for path in Path(root).rglob('*') if path.is_file()}


@pytest.mark.parametrize('gradients,expected', [
    ([], True), ([None], True), ([None, None], True),
    ([0.0], True), ([1.0, -2.0, None], True),
    ([float('nan')], False), ([float('inf')], False),
    ([-float('inf')], False), ([1.0, None, float('nan')], False),
])
def test_gradient_guards_agree_on_empty_none_finite_nan_and_inf(gradients, expected):
    parameters = []
    for value in gradients:
        parameter = torch.nn.Parameter(torch.zeros(2))
        if value is not None:
            parameter.grad = torch.full_like(parameter, value)
        parameters.append(parameter)
    before = [None if p.grad is None else p.grad.clone() for p in parameters]
    assert bool(probe.gradients_finite_legacy(iter(parameters))) is expected
    assert bool(probe.gradients_finite_grouped(iter(parameters))) is expected
    for parameter, original in zip(parameters, before):
        if original is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, original, rtol=0, atol=0,
                                       equal_nan=True)


def test_grouped_gradient_guard_handles_mixed_cpu_dtypes():
    parameters = []
    for dtype in (torch.float32, torch.float64):
        parameter = torch.nn.Parameter(torch.ones(3, dtype=dtype))
        parameter.grad = torch.ones_like(parameter)
        parameters.append(parameter)
    assert probe.gradients_finite_legacy(parameters)
    assert probe.gradients_finite_grouped(parameters)
    parameters[-1].grad[0] = float('inf')
    assert not probe.gradients_finite_legacy(parameters)
    assert not probe.gradients_finite_grouped(parameters)


@pytest.mark.parametrize('mode', ['legacy', 'grouped'])
def test_checked_step_uses_ordinary_mpjpe_and_returns_detached_prediction(mode):
    torch.manual_seed(42)
    model = TinyPose()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    x, y = tiny_batch()
    expected_prediction = model(x)[0].detach().clone()
    expected_loss = float(torch.linalg.vector_norm(expected_prediction-y, dim=-1).mean())
    before = copy.deepcopy(model.state_dict())
    prediction, loss = probe.checked_step(model, optimizer, x, y, mode=mode)
    assert torch.equal(prediction, expected_prediction)
    assert prediction.requires_grad is False and prediction.grad_fn is None
    assert loss == expected_loss
    assert model.unused.grad is None
    assert any(not torch.equal(before[name], value)
               for name, value in model.state_dict().items())


@pytest.mark.parametrize('mode', ['legacy', 'grouped'])
@pytest.mark.parametrize('failure', ['loss_nan', 'gradient_nan', 'gradient_inf'])
def test_checked_step_rejects_nonfinite_before_optimizer_update(monkeypatch, mode, failure):
    torch.manual_seed(42)
    model = TinyPose()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    x, y = tiny_batch()
    if failure == 'loss_nan':
        y[0, 0, 0, 0] = float('nan')
    else:
        value = float('nan') if failure == 'gradient_nan' else float('inf')
        model.head.weight.register_hook(lambda grad: torch.full_like(grad, value))
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())

    def forbidden_step(*args, **kwargs):
        pytest.fail('A nonfinite loss/gradient reached optimizer.step')

    monkeypatch.setattr(optimizer, 'step', forbidden_step)
    with pytest.raises((RuntimeError, ValueError), match='[Nn]onfinite|[Nn]on-finite'):
        probe.checked_step(model, optimizer, x, y, mode=mode)
    assert_same(model.state_dict(), before_model)
    assert_same(optimizer.state_dict(), before_optimizer)


def test_real_wipose_cpu_batch_two_two_step_parity(monkeypatch):
    from models.wipose_hpeli import WiPoseHPELi

    original = probe.checked_step
    calls = []

    def observe(model, optimizer, x, y, mode='legacy'):
        assert isinstance(model, WiPoseHPELi)
        assert tuple(x.shape) == (2, 3, 90, 5)
        assert tuple(y.shape) == (2, 1, 18, 2)
        assert x.device.type == y.device.type == 'cpu'
        result = original(model, optimizer, x, y, mode=mode)
        calls.append((mode, result[0].clone(), result[1],
                      copy.deepcopy(model.state_dict()),
                      copy.deepcopy(optimizer.state_dict()),
                      [None if p.grad is None else p.grad.clone()
                       for p in model.parameters()], torch.get_rng_state().clone()))
        return result

    def forbidden_cuda(*args, **kwargs):
        pytest.fail('CPU parity unexpectedly probed CUDA')

    monkeypatch.setattr(probe, 'checked_step', observe)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden_cuda)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        result = probe.parity_check(device='cpu', batch_size=2, steps=2)
    finally:
        torch.set_num_threads(previous_threads)
    assert result['ok'] is True
    legacy = [item for item in calls if item[0] == 'legacy']
    grouped = [item for item in calls if item[0] == 'grouped']
    assert len(legacy) == len(grouped) == 2
    for one, two in zip(legacy, grouped):
        # Includes train-mode BN buffers, every gradient, SGD momentum and RNG.
        assert_same(one[1:], two[1:])
    assert any(name.endswith('running_mean') for name in legacy[-1][3])
    assert legacy[-1][4]['state']


@pytest.mark.parametrize('changed', ['prediction', 'loss', 'gradient', 'parameter',
                                   'buffer', 'optimizer', 'rng'])
def test_parity_gate_rejects_every_compared_state_component(monkeypatch, changed):
    original = probe.checked_step
    monkeypatch.setattr(probe, '_configure', lambda device: None)
    monkeypatch.setattr(probe, '_model_and_batches',
                        lambda device, batch_size, batches:
                        (TinyPose(), [tiny_batch() for _ in range(batches)]))

    def tamper(model, optimizer, x, y, mode='legacy'):
        prediction, loss = original(model, optimizer, x, y, mode)
        if mode == 'grouped':
            if changed == 'prediction':
                prediction[0, 0, 0, 0] += 1
            elif changed == 'loss':
                loss += 1
            elif changed == 'gradient':
                model.head.weight.grad[0, 0] += 1
            elif changed == 'parameter':
                with torch.no_grad():
                    model.head.weight[0, 0] += 1
            elif changed == 'buffer':
                model.counter += 1
            elif changed == 'optimizer':
                optimizer.state[model.head.weight]['momentum_buffer'][0, 0] += 1
            elif changed == 'rng':
                torch.rand(1)
        return prediction, loss

    monkeypatch.setattr(probe, 'checked_step', tamper)
    with pytest.raises(RuntimeError, match='parity'):
        probe.parity_check(device='cpu', batch_size=2, steps=1)


@pytest.fixture
def io_dataset_root(tmp_path):
    root = tmp_path/'processed'
    root.mkdir()
    orders = {'Train': [4, 1, 8, 2, 7, 3, 6, 5, 9], 'Test': [1, 2]}
    for prefix, frames in orders.items():
        action = 'walk_001' if prefix == 'Train' else 'bend_002'
        names = [f'{action}-frame{number:03d}' for number in frames]
        (root/f'{prefix}_data_list.txt').write_text('\n'.join(names)+'\n', encoding='utf-8')
        folder = root/f'{prefix}_Amplitude_DWT'
        folder.mkdir()
        for number, name in zip(frames, names):
            csi = (np.arange(1350, dtype=np.float64)+1)**(1+number*.01)
            pose = np.arange(36, dtype=np.float64).reshape(2, 18)+number
            np.savez(folder/f'{name}.npz', CSI=csi.reshape(3, 90, 5),
                     SkeletonPoints=pose)
    return root


def test_io_reads_bounded_seeded_train_frames_with_exact_readonly_cache(io_dataset_root, monkeypatch):
    root = io_dataset_root
    dataset = wipose_data.WiPoseDataset(root, 'training')
    indices = np.random.RandomState(42).choice(len(dataset), size=4, replace=False)
    names = [dataset.items[int(index)]['name'] for index in indices]
    assert names != [item['name'] for item in dataset.items[:4]]
    reads = []
    original = wipose_data.read_frame

    def observe(path):
        pair = original(path)
        reads.append((Path(path), pair))
        return pair

    def forbidden_audit(*args, **kwargs):
        pytest.fail('Bounded IO diagnostic performed a full dataset audit')

    monkeypatch.setattr(wipose_data, 'read_frame', observe)
    monkeypatch.setattr(wipose_data, 'dataset_audit', forbidden_audit)
    before = tree(root)
    report = probe.io_probe(root, samples=4, seed=42)
    assert report['samples'] == 4 and report['retained_train'] == 8
    assert report['cache_coverage'] == .5
    assert report['cache_bytes'] == 4*(1350+36)*np.dtype(np.float32).itemsize
    assert report['cold_read_seconds'] >= 0 and report['cached_read_seconds'] >= 0
    assert report['bitwise_frame_parity'] is report['cache_arrays_readonly'] is True
    assert report['selected_names_sha256'] == hashlib.sha256(
        ('\n'.join(names)+'\n').encode()).hexdigest()
    assert len(reads) == 8
    assert [path.stem for path, _ in reads] == names+names
    assert all(path.parent.name == 'Train_Amplitude_DWT' for path, _ in reads)
    for (_, stored), (_, fresh) in zip(reads[:4], reads[4:]):
        for cached_array, fresh_array in zip(stored, fresh):
            assert cached_array.dtype == fresh_array.dtype == np.float32
            np.testing.assert_array_equal(cached_array, fresh_array)
            assert not cached_array.flags.writeable
            with pytest.raises(ValueError):
                cached_array.flat[0] = 999
            saved = cached_array.copy()
            fresh_array.flat[0] = 999
            np.testing.assert_array_equal(cached_array, saved)
    json.dumps(report, allow_nan=False)
    assert tree(root) == before


@pytest.mark.parametrize('samples', [0, -1, 1025, 100000])
def test_io_rejects_out_of_bound_counts_before_reading(tmp_path, monkeypatch, samples):
    monkeypatch.setattr(wipose_data, 'read_frame',
                        lambda *args: pytest.fail('Invalid IO count read a frame'))
    with pytest.raises(ValueError, match='samples'):
        probe.io_probe(tmp_path/'missing', samples=samples)


def test_io_rejects_request_exceeding_retained_train_without_writes(io_dataset_root):
    before = tree(io_dataset_root)
    with pytest.raises(ValueError, match='retained TRAIN'):
        probe.io_probe(io_dataset_root, samples=9)
    assert tree(io_dataset_root) == before


@pytest.mark.parametrize('stdout,returncode', [
    ('987654321\n', 0), ('N/A\n', 0), ('unexpected warning\n', 0), ('', 1),
])
@pytest.mark.parametrize('entrypoint', ['compute', 'parity'])
def test_cuda_probes_refuse_busy_or_unknown_host_before_touching_cuda(monkeypatch, stdout, returncode, entrypoint):
    calls = []

    def fake_process(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr='')

    monkeypatch.setattr(probe.subprocess, 'run', fake_process)
    monkeypatch.setattr(probe, '_configure',
                        lambda *args: pytest.fail('Rejected CUDA probe initialized CUDA'))
    with pytest.raises(RuntimeError):
        if entrypoint == 'compute':
            probe.compute_probe(device='cuda:0', steps=1, warmup=0, rounds=1)
        else:
            probe.parity_check(device='cuda:0', batch_size=2, steps=1)
    assert len(calls) == 1
    assert calls[0][0] == 'nvidia-smi'
    assert '--query-compute-apps=pid' in calls[0]


@pytest.mark.parametrize('mode', ['plan', 'io'])
def test_main_refuses_to_report_changed_source_hashes(io_dataset_root, monkeypatch, capsys, mode):
    sources = iter([{'fixture.py': '0'*64}, {'fixture.py': '1'*64}])
    monkeypatch.setattr(probe, '_source_hashes', lambda: next(sources))
    arguments = ['--mode', mode]
    if mode == 'io':
        arguments += ['--dataset-root', str(io_dataset_root), '--samples', '4']
    before = tree(io_dataset_root)
    with pytest.raises(RuntimeError, match='Source changed'):
        probe.main(arguments)
    assert capsys.readouterr().out == ''
    assert tree(io_dataset_root) == before


def run_cli(tmp_path, arguments, guarded=False):
    script = '''import builtins, runpy, sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'wipose_data', 'models',
                             'run_wipose_hpeli', 'run_wipose_hpeli_parallel'}:
        raise AssertionError('Forbidden plan import: '+name)
    return real_import(name, *args, **kwargs)
if sys.argv[1] == 'guard':
    builtins.__import__ = guarded_import
sys.argv = [sys.argv[2], *sys.argv[3:]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    return subprocess.run(
        [sys.executable, '-B', '-X', 'utf8', '-c', script,
         'guard' if guarded else 'allow', str(HERE/'probe_wipose_speed.py'),
         *arguments], cwd=tmp_path, capture_output=True, text=True,
        timeout=30, check=False,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(HERE)))


@pytest.mark.parametrize('arguments', [[], ['--mode', 'plan'],
                                     ['--mode', 'plan', '--device', 'cuda:0']])
def test_fresh_default_plan_imports_no_tensor_data_model_or_cuda_and_writes_nothing(tmp_path, arguments):
    before = tree(tmp_path)
    result = run_cli(tmp_path, arguments, guarded=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
    assert tree(tmp_path) == before


@pytest.mark.parametrize('arguments', [
    ['--steps', '0'], ['--steps', '100000'],
    ['--samples', '0'], ['--samples', '100000'],
    ['--warmup', '-1'], ['--warmup', '100000'],
    ['--rounds', '0'], ['--rounds', '100000'],
    ['--device', 'cuda'], ['--device', 'cuda:-1'],
    ['--device', 'auto'], ['--device', 'mps'],
    ['--dataset-root', 'missing'],
    ['--mode', 'compute', '--dataset-root', 'missing'],
    ['--mode', 'io'],
])
def test_cli_rejects_unbounded_or_ambiguous_invocations_before_runtime_imports(tmp_path, arguments):
    before = tree(tmp_path)
    result = run_cli(tmp_path, arguments, guarded=True)
    assert result.returncode != 0
    assert 'Forbidden plan import' not in result.stderr
    assert tree(tmp_path) == before
