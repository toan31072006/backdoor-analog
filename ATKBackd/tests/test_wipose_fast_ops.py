"""Exact cache/gradient-check contracts on local synthetic CPU fixtures."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wipose_fast_ops as ops
from wipose_data import WiPoseDataset


class TinyTraining:
    split = 'training'

    def __init__(self, count=12):
        self.items = [dict(name=f'walk_001-frame{index + 1:03d}')
                      for index in range(count)]
        self.calls = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        self.calls += 1
        if index < 0:
            index += len(self)
        csi = np.linspace(0, 1, 1350, dtype=np.float32).reshape(3, 90, 5)
        pose = (np.arange(36, dtype=np.float32).reshape(1, 18, 2) + index) * .001
        return dict(csi=csi, pose=pose, name=self.items[index]['name'])


def numpy_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_cache_exact_values_names_order_rng_and_payload():
    base = TinyTraining()
    random.seed(83)
    np.random.seed(87)
    torch.manual_seed(91)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cached = ops.cache_training(base)
    assert base.calls == len(base)
    assert random.getstate() == python_rng
    numpy_rng_equal(np.random.get_state(), numpy_rng)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert cached.items is base.items and cached.split == 'training'
    assert cached.payload_bytes == len(base) * 5544
    assert cached.max_bytes == 2 * 1024 ** 3
    assert len(cached) == len(base)
    for index in (0, 7, -1):
        got, expected = cached[index], base[index]
        assert set(got) == {'csi', 'pose', 'name'}
        assert got['name'] == expected['name']
        for field in ('csi', 'pose'):
            assert got[field].dtype == np.float32
            np.testing.assert_array_equal(got[field], expected[field])


def test_retrieval_is_owned_and_internal_arrays_are_read_only():
    base = TinyTraining()
    cached = ops.cache_training(base)
    assert not cached._csi.flags.writeable and not cached._poses.flags.writeable
    with pytest.raises(ValueError):
        cached._csi[0, 0, 0, 0] = 9
    first = cached[0]
    expected = base[0]
    for field in ('csi', 'pose'):
        assert first[field].flags.owndata and first[field].flags.writeable
        assert not np.shares_memory(first[field], cached[0][field])
        first[field][:] = 99
        np.testing.assert_array_equal(cached[0][field], expected[field])
    calls = base.calls
    cached[0]
    assert base.calls == calls


def test_capacity_rejected_before_allocation_or_reads(monkeypatch):
    base = TinyTraining()
    def forbidden(*args, **kwargs):
        raise AssertionError('Capacity rejection must precede allocation')
    monkeypatch.setattr(ops.np, 'empty', forbidden)
    with pytest.raises(MemoryError, match='no partial cache or fallback'):
        ops.cache_training(base, max_bytes=len(base) * 5544 - 1)
    assert base.calls == 0


def test_exact_capacity_accepted_and_no_writes(tmp_path):
    before = list(tmp_path.iterdir())
    base = TinyTraining(2)
    cached = ops.cache_training(base, max_bytes=2 * 5544)
    assert cached.payload_bytes == cached.max_bytes == 11088
    assert list(tmp_path.iterdir()) == before


@pytest.mark.parametrize('bound', [True, False, 0, -1, 2.0, '10000', None])
def test_invalid_capacity_rejected(bound):
    with pytest.raises(ValueError, match='positive integer'):
        ops.cache_training(TinyTraining(), max_bytes=bound)


def test_validation_and_empty_datasets_rejected():
    base = TinyTraining()
    base.split = 'validation'
    with pytest.raises(ValueError, match='TRAIN only'):
        ops.cache_training(base)
    with pytest.raises(ValueError, match='one ordered identity'):
        ops.cache_training(TinyTraining(0))


@pytest.mark.parametrize('issue', ['csi_shape', 'pose_shape', 'csi_dtype', 'pose_dtype',
                                    'csi_nan', 'pose_inf', 'range', 'identity'])
def test_invalid_cached_samples_fail_without_fallback(issue):
    class Bad(TinyTraining):
        def __getitem__(self, index):
            value = super().__getitem__(index)
            if issue == 'csi_shape': value['csi'] = value['csi'].reshape(3, 45, 10)
            elif issue == 'pose_shape': value['pose'] = value['pose'][0]
            elif issue == 'csi_dtype': value['csi'] = value['csi'].astype(np.float64)
            elif issue == 'pose_dtype': value['pose'] = value['pose'].astype(np.float64)
            elif issue == 'csi_nan': value['csi'][0, 0, 0] = np.nan
            elif issue == 'pose_inf': value['pose'][0, 0, 0] = np.inf
            elif issue == 'range': value['csi'][0, 0, 0] = 2
            elif issue == 'identity': value['name'] = 'other'
            return value
    with pytest.raises(ValueError):
        ops.cache_training(Bad())


def test_real_reader_exact_normalization_and_input_files_unchanged(tmp_path):
    root = tmp_path / 'processed'
    root.mkdir()
    names = ['walk_001-frame003', 'walk_001-frame001', 'walk_001-frame002']
    (root / 'Train_data_list.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
    folder = root / 'Train_Amplitude_DWT'
    folder.mkdir()
    for index, name in enumerate(names):
        csi = (np.arange(1350, dtype=np.uint16).reshape(3, 90, 5) * (index + 1))
        pose = np.arange(36, dtype=np.float64).reshape(2, 18) + index + .125
        np.savez_compressed(folder / f'{name}.npz', CSI=csi, SkeletonPoints=pose)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in root.rglob('*') if path.is_file()}
    base = WiPoseDataset(root, 'training')
    bindings = {str(path.resolve()): hashlib.sha256(contents).hexdigest()
                for path, (contents, _) in before.items() if path.suffix == '.npz'}
    cached = ops.cache_training(base, expected_hashes=bindings)
    assert [item['name'] for item in cached.items] == ['walk_001-frame001', 'walk_001-frame002']
    for index in range(len(base)):
        for field in ('csi', 'pose'):
            np.testing.assert_array_equal(base[index][field], cached[index][field])
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


def test_input_stat_change_during_preload_rejected(tmp_path):
    path = tmp_path / 'frame.npz'
    path.write_bytes(b'a')
    class Changing(TinyTraining):
        def __getitem__(self, index):
            value = super().__getitem__(index)
            path.write_bytes(b'changed-size')
            return value
    base = Changing(1)
    base.items[0]['path'] = path
    with pytest.raises(RuntimeError, match='changed during cache preload'):
        ops.cache_training(base)


def bound_dataset(tmp_path):
    base = TinyTraining(2)
    bindings = {}
    for index, item in enumerate(base.items):
        path = tmp_path / f'frame{index}.npz'
        path.write_bytes(f'bound-frame-{index}'.encode())
        item['path'] = path
        bindings[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return base, bindings


def test_cache_matches_every_audited_file_hash_before_and_after_read(tmp_path, monkeypatch):
    base, bindings = bound_dataset(tmp_path)
    calls = []
    original = ops._file_sha
    def counted(path):
        calls.append(path.resolve())
        return original(path)
    monkeypatch.setattr(ops, '_file_sha', counted)
    cached = ops.cache_training(base, expected_hashes=bindings)
    assert calls == [base.items[0]['path']] * 2 + [base.items[1]['path']] * 2
    for index in range(len(base)):
        np.testing.assert_array_equal(cached[index]['csi'], base[index]['csi'])
        np.testing.assert_array_equal(cached[index]['pose'], base[index]['pose'])


def test_wrong_audited_hash_fails_before_sample_read(tmp_path):
    base, bindings = bound_dataset(tmp_path)
    bindings[str(base.items[0]['path'])] = '0' * 64
    with pytest.raises(RuntimeError, match='SHA256 differs before'):
        ops.cache_training(base, expected_hashes=bindings)
    assert base.calls == 0


def test_missing_audited_path_fails_before_allocation(tmp_path, monkeypatch):
    base, bindings = bound_dataset(tmp_path)
    bindings.pop(str(base.items[1]['path']))
    def forbidden(*args, **kwargs):
        raise AssertionError('Missing binding must fail before allocation')
    monkeypatch.setattr(ops.np, 'empty', forbidden)
    with pytest.raises(ValueError, match='Unbound WiPose TRAIN path'):
        ops.cache_training(base, expected_hashes=bindings)
    assert base.calls == 0
    with pytest.raises(ValueError, match='Unbound WiPose TRAIN path'):
        ops.cache_training(TinyTraining(1), expected_hashes={})


@pytest.mark.parametrize('binding', [[], {'relative.npz': '0' * 64},
                                     {'/absolute': 'bad'}, {'/absolute': 'F' * 64}])
def test_invalid_hash_bindings_rejected(binding):
    with pytest.raises(ValueError, match='expected_hashes|absolute path'):
        ops.cache_training(TinyTraining(1), expected_hashes=binding)


def test_hash_detects_content_change_even_when_size_and_mtime_preserved(tmp_path):
    path = tmp_path / 'frame.npz'
    path.write_bytes(b'original-value')
    initial = path.stat()
    binding = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()}
    class Changing(TinyTraining):
        def __getitem__(self, index):
            value = super().__getitem__(index)
            path.write_bytes(b'changed--value')
            os.utime(path, ns=(initial.st_atime_ns, initial.st_mtime_ns))
            assert path.stat().st_size == initial.st_size
            assert path.stat().st_mtime_ns == initial.st_mtime_ns
            return value
    base = Changing(1)
    base.items[0]['path'] = path
    with pytest.raises(RuntimeError, match='SHA256 differs after'):
        ops.cache_training(base, expected_hashes=binding)


def test_native_a1_inputs_payload_poison_plan_identical_after_cache(tmp_path):
    import wipose_a1_adapter as adapter
    action = np.zeros((2, 3, 8, 25, 1), dtype=np.float32)
    for time in range(8):
        for joint in range(25):
            action[1, :, time, joint, 0] = [1 + .01 * (joint + 1) * time,
                                           .5 + .02 * joint, 2 + .01 * time]
    action_path = tmp_path / 'action.npy'
    np.save(action_path, action, allow_pickle=False)
    trigger = adapter.build_trigger(action_path)
    base = TinyTraining(24)
    cached = ops.cache_training(base)
    original = adapter.PoisonedWiPoseDataset(base, trigger)
    accelerated = adapter.PoisonedWiPoseDataset(cached, trigger)
    assert original.manifest() == accelerated.manifest()
    assert original._selected == accelerated._selected
    assert len(original._selected) > 0
    for index in range(len(base)):
        for field in ('csi', 'pose'):
            np.testing.assert_array_equal(original[index][field], accelerated[index][field])
    for record in accelerated.manifest()['samples']:
        index = record['index']
        expected = adapter.target_pose(base[index]['pose'], record['payload_dose']).astype(np.float32)
        np.testing.assert_array_equal(accelerated[index]['pose'], expected)
    np.testing.assert_array_equal(cached[0]['csi'], base[0]['csi'])


def legacy_finite(parameters):
    return not any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in parameters)


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), -float('inf')])
def test_grouped_gradient_predicate_matches_legacy_and_does_not_mutate(bad):
    parameters = [torch.nn.Parameter(torch.tensor([1., 2.])),
                  torch.nn.Parameter(torch.tensor([3.])), torch.nn.Parameter(torch.tensor([4.]))]
    parameters[0].grad = torch.tensor([.25, .5])
    parameters[2].grad = torch.tensor([.125 if bad is None else bad])
    before = [None if value.grad is None else value.grad.clone() for value in parameters]
    expected = legacy_finite(parameters)
    actual = ops.grouped_gradients_finite(iter(parameters))
    assert type(actual) is bool and actual == expected == (bad is None)
    for parameter, gradient in zip(parameters, before):
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0, equal_nan=True)
    assert ops.grouped_gradients_finite([]) is True
    assert ops.grouped_gradients_finite([torch.nn.Parameter(torch.tensor(1.))]) is True


def test_grouped_guard_preserves_complete_cpu_training_state():
    import copy
    torch.manual_seed(42)
    original = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4),
                                   torch.nn.ReLU(), torch.nn.Linear(4, 2))
    inputs = [(torch.rand(4, 4), torch.rand(4, 2)) for _ in range(3)]
    results = []
    for check in (legacy_finite, ops.grouped_gradients_finite):
        model = copy.deepcopy(original)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        for x, truth in inputs:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = torch.linalg.vector_norm(prediction - truth, dim=-1).mean()
            loss.backward()
            assert check(model.parameters())
            optimizer.step()
        results.append((model.state_dict(), optimizer.state_dict(),
                        [value.grad.clone() for value in model.parameters()]))
    for name in results[0][0]:
        assert torch.equal(results[0][0][name], results[1][0][name])
    for parameter in results[0][1]['state']:
        assert torch.equal(results[0][1]['state'][parameter]['momentum_buffer'],
                           results[1][1]['state'][parameter]['momentum_buffer'])
    assert all(torch.equal(left, right) for left, right in zip(results[0][2], results[1][2]))
