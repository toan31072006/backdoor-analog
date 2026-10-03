"""Focused tests for MM-Fi's process-local file caches."""

import os
import sys

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils.feeder as feeder  # noqa: E402
from data_utils.feeder import MMFI  # noqa: E402


def _bare_mmfi(items=None):
    """Build the cache-bearing part of MMFI without walking a dataset tree."""
    ds = MMFI.__new__(MMFI)
    ds.num_person = 1
    ds.items = list(items or [])
    ds._ground_truth_cache = {}
    ds._ground_truth_cache_pid = os.getpid()
    ds._raw_cache = None
    ds._raw_cache_pid = None
    ds._raw_cache_bytes = 0
    ds._raw_cache_max_bytes = 0
    return ds


def test_pose_cache_loads_each_sequence_once_and_closes_mmap(tmp_path,
                                                              monkeypatch):
    gt_path = tmp_path / 'ground_truth.npy'
    expected = np.arange(3 * 17 * 3, dtype=np.float64).reshape(3, 17, 3)
    np.save(gt_path, expected)

    real_load = np.load
    loads = []

    def counting_load(path, *args, **kwargs):
        arr = real_load(path, *args, **kwargs)
        loads.append((path, kwargs, arr))
        return arr

    monkeypatch.setattr(feeder.np, 'load', counting_load)
    ds = _bare_mmfi()

    pose0 = ds.load_pose(str(gt_path), 0)
    pose2 = ds.load_pose(str(gt_path), 2)

    assert len(loads) == 1
    assert loads[0][1]['mmap_mode'] == 'r'
    assert loads[0][1]['allow_pickle'] is False
    assert loads[0][2]._mmap.closed
    assert pose0.shape == (1, 17, 3)
    assert pose0.dtype == np.float32
    np.testing.assert_array_equal(pose0[0], expected[0].astype(np.float32))
    np.testing.assert_array_equal(pose2[0], expected[2].astype(np.float32))

    # A caller can safely edit its frame without corrupting the shared cache.
    pose0[...] = -1.0
    again = ds.load_pose(str(gt_path), 0)
    np.testing.assert_array_equal(again[0], expected[0].astype(np.float32))


def test_pose_cache_is_reset_when_pid_changes(tmp_path, monkeypatch):
    gt_path = tmp_path / 'ground_truth.npy'
    np.save(gt_path, np.zeros((2, 17, 3), dtype=np.float32))

    real_load = np.load
    load_count = 0

    def counting_load(path, *args, **kwargs):
        nonlocal load_count
        load_count += 1
        return real_load(path, *args, **kwargs)

    fake_pid = [101]
    monkeypatch.setattr(feeder.np, 'load', counting_load)
    monkeypatch.setattr(feeder.os, 'getpid', lambda: fake_pid[0])
    ds = _bare_mmfi()

    ds.load_pose(str(gt_path), 0)
    ds.load_pose(str(gt_path), 1)
    assert load_count == 1

    fake_pid[0] = 202
    ds.load_pose(str(gt_path), 0)
    assert load_count == 2
    assert ds._ground_truth_cache_pid == 202


def test_eval_raw_cache_reuses_exact_read_only_float32_array(tmp_path,
                                                              monkeypatch):
    csi_path = tmp_path / 'frame001_processed.npy'
    source = np.linspace(0.0, 1.0, 3 * 114 * 10, dtype=np.float64).reshape(
        3, 114, 10)
    np.save(csi_path, source)

    real_load = np.load
    load_count = 0

    def counting_load(path, *args, **kwargs):
        nonlocal load_count
        load_count += 1
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(feeder.np, 'load', counting_load)
    ds = _bare_mmfi([{'csi': str(csi_path)}])
    assert ds.enable_evaluation_cache()

    first = ds.load_raw(str(csi_path))
    second = ds.load_raw(str(csi_path))

    assert first is second
    assert load_count == 1
    assert first.dtype == np.float32
    assert not first.flags.writeable
    np.testing.assert_array_equal(first, source.astype(np.float32))
    # Normalization still allocates its result and leaves the cached input alone.
    normalized = ds.normalize(first)
    assert normalized is not first
    np.testing.assert_array_equal(normalized, first)


def test_eval_raw_cache_refuses_split_larger_than_budget():
    one_frame_bytes = 3 * 114 * 10 * np.dtype(np.float32).itemsize
    ds = _bare_mmfi([{'csi': 'one'}, {'csi': 'two'}])

    assert not ds.enable_evaluation_cache(max_bytes=one_frame_bytes)
    assert ds._raw_cache is None


def test_smaller_reenable_budget_clears_existing_eval_cache(tmp_path):
    csi_path = tmp_path / 'frame001_processed.npy'
    np.save(csi_path, np.zeros((3, 114, 10), dtype=np.float32))
    frame_bytes = 3 * 114 * 10 * np.dtype(np.float32).itemsize
    ds = _bare_mmfi([{'csi': str(csi_path)}, {'csi': str(csi_path)}])

    assert ds.enable_evaluation_cache(max_bytes=2 * frame_bytes)
    ds.load_raw(str(csi_path))
    assert ds._raw_cache

    assert not ds.enable_evaluation_cache(max_bytes=frame_bytes)
    assert ds._raw_cache is None
    assert ds._raw_cache_bytes == 0


def test_pickling_state_never_contains_cached_arrays(tmp_path):
    gt_path = tmp_path / 'ground_truth.npy'
    csi_path = tmp_path / 'frame001_processed.npy'
    np.save(gt_path, np.zeros((1, 17, 3), dtype=np.float32))
    np.save(csi_path, np.zeros((3, 114, 10), dtype=np.float32))

    ds = _bare_mmfi([{'csi': str(csi_path)}])
    ds.load_pose(str(gt_path), 0)
    assert ds.enable_evaluation_cache()
    ds.load_raw(str(csi_path))

    state = ds.__getstate__()
    assert state['_ground_truth_cache'] == {}
    assert state['_ground_truth_cache_pid'] is None
    assert state['_raw_cache'] == {}
    assert state['_raw_cache_pid'] is None
    assert state['_raw_cache_bytes'] == 0
