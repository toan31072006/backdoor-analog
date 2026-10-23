"""Strict WiPose reader independently adapted from the DT-Pose data contract.

The public DT-Pose feeder consumes preprocessed ``CSI``/``SkeletonPoints`` NPZ
files.  We preserve its ordered lists, next-frame TRAIN selection, per-frame
global amplitude min-max and coordinate division by 1000.  We do not use its
pretraining pipeline or interpret the scaled image coordinates as metres.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np


CSI_SHAPE = (3, 90, 5)
POSE_SHAPE = (2, 18)
COORDINATE_SCALE = .001
SPLITS = {
    'training': ('Train_data_list.txt', 'Train_Amplitude_DWT'),
    'validation': ('Test_data_list.txt', 'Test_Amplitude_DWT'),
}
ACTIONS = ('bend', 'circle', 'crouch', 'jump', 'wave', 'walk', 'throw',
           'standup', 'sitdown', 'run', 'push', 'pull')
NAME_PATTERN = re.compile(r'([A-Za-z0-9_]+)-frame([0-9]{3,})\Z')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read_names(path):
    """Read original ordered identities without accepting traversal or aliases."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing official WiPose list: {path}')
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f'Empty list or interior blank line: {path}')
    names = [line.split()[0] for line in lines]
    if len(names) != len(set(names)):
        raise ValueError(f'Duplicate official WiPose identity: {path}')
    for name in names:
        if not NAME_PATTERN.fullmatch(name) or name.split('_')[0] not in ACTIONS:
            raise ValueError(f'Unsafe or unexpected official WiPose identity: {name!r}')
    return names


def next_frame_name(name):
    match = NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f'Invalid WiPose frame identity: {name!r}')
    # This is the exact DT-Pose zfill(3), not the input suffix width.
    return f'{match[1]}-frame{int(match[2]) + 1:03d}'


def read_frame(path):
    """Return normalized FP32 CSI and (1,18,2) scaled image coordinates."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing listed WiPose frame: {path}')
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not {'CSI', 'SkeletonPoints'}.issubset(archive.files):
                raise ValueError('expected CSI and SkeletonPoints arrays')
            csi, pose = archive['CSI'], archive['SkeletonPoints']
    except (OSError, ValueError, EOFError, KeyError) as exc:
        raise ValueError(f'Invalid safe numeric WiPose NPZ: {path}: {exc}') from exc
    for name, value, shape in (('CSI', csi, CSI_SHAPE),
                               ('SkeletonPoints', pose, POSE_SHAPE)):
        if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError(f'Invalid numeric finite {name} shape {value.shape}: {path}')
    # float64 arithmetic guards unsigned overflow; normalized FP32 output matches
    # the original collator's final conversion without modifying archived inputs.
    csi64 = csi.astype(np.float64)
    low, high = float(csi64.min()), float(csi64.max())
    if not high > low or not np.isfinite(high - low):
        raise ValueError(f'Degenerate or unrepresentable CSI min-max range: {path}')
    csi_out = ((csi64 - low) / (high - low)).astype(np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        pose_out = (pose.astype(np.float64).T * COORDINATE_SCALE).astype(np.float32)[None]
    if not np.isfinite(pose_out).all():
        raise ValueError(f'WiPose pose cannot be represented in FP32: {path}')
    if np.linalg.norm(pose_out[0, 5].astype(np.float64) -
                      pose_out[0, 8].astype(np.float64)) <= 0:
        raise ValueError(f'Degenerate clean GT/PCK reference scale (5,8): {path}')
    return csi_out, pose_out


class WiPoseDataset:
    """Map-style PyTorch-compatible dataset; TEST is not next-frame filtered."""
    def __init__(self, root, split):
        if split not in SPLITS:
            raise ValueError('WiPose split must be training or validation')
        self.root, self.split = Path(root).resolve(), split
        list_name, directory = SPLITS[split]
        self.list_path = self.root / list_name
        self.official_names = read_names(self.list_path)
        chosen = set(self.official_names)
        retained = ([name for name in self.official_names
                     if next_frame_name(name) in chosen]
                    if split == 'training' else list(self.official_names))
        if not retained:
            raise ValueError(f'No retained WiPose identities in {split}')
        self.items = [dict(name=name, path=self.root / directory / f'{name}.npz')
                      for name in retained]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        csi, pose = read_frame(item['path'])
        return dict(csi=csi, pose=pose, name=item['name'])


def dataset_audit(root):
    """Read/hash all official identities; return JSON-safe complete input binding.

    Includes last TRAIN frames omitted by the next-frame filter, because those
    frames determined that filter and are part of the supplied source dataset.
    No data/output files are created and no count is guessed from the paper.
    """
    root = Path(root).resolve()
    details, identities = {}, {}
    for split, (list_name, directory) in SPLITS.items():
        dataset = WiPoseDataset(root, split)
        identities[split] = set(dataset.official_names)
        files = []
        for index, name in enumerate(dataset.official_names):
            path = root / directory / f'{name}.npz'
            if not path.is_file():
                raise FileNotFoundError(f'Missing listed WiPose file: {path}')
            before = path.stat()
            read_frame(path)
            digest = file_sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'WiPose input changed during audit: {path}')
            files.append(dict(path=path.relative_to(root).as_posix(),
                              size=after.st_size, sha256=digest))
            if index and index % 10000 == 0:
                print(f'[WiPose preflight] {split}: {index}/{len(dataset.official_names)} identities', flush=True)
        names = [item['name'] for item in dataset.items]
        details[split] = dict(list_path=list_name, official_list_sha256=file_sha(root / list_name),
            official_count=len(dataset.official_names), official_names=dataset.official_names,
            retained_count=len(dataset), ordered_names=names,
            ordered_names_sha256=hashlib.sha256(('\n'.join(names) + '\n').encode('utf-8')).hexdigest(),
            files=files, files_sha256=object_sha(files))
        print(f'[WiPose preflight] {split}: {len(dataset)} retained / {len(dataset.official_names)} official', flush=True)
    overlap = identities['training'] & identities['validation']
    if overlap:
        raise ValueError(f'TRAIN/TEST identity overlap: {sorted(overlap)[:3]}')
    return dict(schema=1, dataset_root=str(root), splits=details,
                fingerprint=object_sha(details))
