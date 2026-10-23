"""Exact-value TRAIN caching and grouped gradient checks for the fast sidecar.

This module does not import a runner, write files, change random state, or alter
normalization, A1, model math, optimization, or checkpoint semantics.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
import re

import numpy as np


DEFAULT_CACHE_BYTES = 2 * 1024 ** 3
FRAME_PAYLOAD_BYTES = (3 * 90 * 5 + 1 * 18 * 2) * np.dtype(np.float32).itemsize


class CachedWiPoseTraining:
    """Compact, read-only backing arrays; every retrieval returns owned copies."""

    def __init__(self, dataset, csi, poses, names, max_bytes):
        self.items = dataset.items
        self.split = 'training'
        self.max_bytes = max_bytes
        self.payload_bytes = csi.nbytes + poses.nbytes
        self._names = tuple(names)
        csi.setflags(write=False)
        poses.setflags(write=False)
        self._csi, self._poses = csi, poses

    def __len__(self):
        return len(self._names)

    def __getitem__(self, index):
        # A1 and callers may mutate these values without touching the cache.
        return dict(csi=self._csi[index].copy(), pose=self._poses[index].copy(),
                    name=self._names[index])


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_training(dataset, max_bytes=DEFAULT_CACHE_BYTES, expected_hashes=None):
    """Read the complete ordered TRAIN once, without consuming random draws.

    ``max_bytes`` bounds NumPy payload, not total process RSS. Capacity failure
    is fatal: there is no partial-cache or disk-reader fallback. The caller must
    retain its ordinary input audit/binding; this is not a replacement audit.
    When supplied, ``expected_hashes`` maps absolute paths to audited SHA256
    values. Every cached frame must match its binding before and after reading.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('WiPose cache max_bytes must be a positive integer')
    if getattr(dataset, 'split', None) != 'training':
        raise ValueError('WiPose RAM cache accepts TRAIN only')
    count = len(dataset)
    if count < 1 or len(dataset.items) != count:
        raise ValueError('WiPose TRAIN must have one ordered identity per sample')
    expected_bytes = count * FRAME_PAYLOAD_BYTES
    if expected_bytes > max_bytes:
        raise MemoryError(f'WiPose TRAIN cache requires {expected_bytes} payload bytes; '
                          f'limit={max_bytes}; no partial cache or fallback')
    names = tuple(item['name'] for item in dataset.items)
    if (any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != count):
        raise ValueError('WiPose TRAIN cache requires unique named identities')
    hashes = None
    if expected_hashes is not None:
        if not isinstance(expected_hashes, Mapping):
            raise ValueError('WiPose cache expected_hashes must map absolute paths to SHA256')
        hashes = {}
        for key, digest in expected_hashes.items():
            if (not isinstance(key, (str, Path)) or not Path(key).is_absolute()
                    or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
                raise ValueError('WiPose cache requires absolute path and lowercase SHA256 bindings')
            resolved = Path(key).resolve()
            if resolved in hashes and hashes[resolved] != digest:
                raise ValueError('Conflicting WiPose cache SHA256 bindings')
            hashes[resolved] = digest
        for record in dataset.items:
            if 'path' not in record or Path(record['path']).resolve() not in hashes:
                raise ValueError('Unbound WiPose TRAIN path in cache preload')
    # Allocate only after the full payload estimate has passed the strict cap.
    csi = np.empty((count, 3, 90, 5), dtype=np.float32)
    poses = np.empty((count, 1, 18, 2), dtype=np.float32)
    actual_bytes = csi.nbytes + poses.nbytes
    if actual_bytes != expected_bytes or actual_bytes > max_bytes:
        raise MemoryError('WiPose TRAIN cache allocation exceeded its payload bound')
    for index, name in enumerate(names):
        record = dataset.items[index]
        path = Path(record['path']) if 'path' in record else None
        before = path.stat() if path is not None else None
        if hashes is not None:
            expected_hash = hashes.get(path.resolve())
            if expected_hash is None or _file_sha(path) != expected_hash:
                raise RuntimeError(f'WiPose input SHA256 differs before cache preload: {path}')
        sample = dataset[index]
        if path is not None:
            if hashes is not None and _file_sha(path) != expected_hash:
                raise RuntimeError(f'WiPose input SHA256 differs after cache preload: {path}')
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'WiPose input changed during cache preload: {path}')
        if sample.get('name') != name:
            raise ValueError('WiPose TRAIN identity/order changed during cache preload')
        for label, value, shape in (('csi', sample['csi'], (3, 90, 5)),
                                    ('pose', sample['pose'], (1, 18, 2))):
            if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                    or value.shape != shape or not np.isfinite(value).all()):
                raise ValueError(f'WiPose TRAIN cache requires exact finite FP32 {label} {shape}')
        if np.any(sample['csi'] < 0) or np.any(sample['csi'] > 1):
            raise ValueError('WiPose TRAIN cache requires normalized [0,1] CSI')
        csi[index], poses[index] = sample['csi'], sample['pose']
        if (index + 1) % 10000 == 0:
            print(f'[WiPose RAM cache] {index + 1}/{count} TRAIN frames', flush=True)
    if len(dataset) != count or tuple(item['name'] for item in dataset.items) != names:
        raise RuntimeError('WiPose TRAIN identities changed during cache preload')
    result = CachedWiPoseTraining(dataset, csi, poses, names, max_bytes)
    print(f'[WiPose RAM cache] READY {count} TRAIN frames; '
          f'{result.payload_bytes} payload bytes; exact FP32 copies', flush=True)
    return result


def grouped_gradients_finite(parameters):
    """Preserve the current predicate with one host decision for populated grads."""
    import torch
    flags = [torch.isfinite(parameter.grad).all()
             for parameter in parameters if parameter.grad is not None]
    return True if not flags else bool(torch.stack(flags).all())
