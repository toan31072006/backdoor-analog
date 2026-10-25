"""Full-content, CPU-only input gate for the separate Vast seed-42 batch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

EXPECTED_ACTION_SHA256 = '2e8fc11192cc3ee7ee3f5544cb3c5e19f771f7f7265dde93af8969af41a2be3e'
COUNTS = {'train': 133056, 'eval': 33264}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def inspect(data_home, *, expected_action_sha=EXPECTED_ACTION_SHA256, counts=None):
    """Read every selected CSI/pose once. Never infer units, normalize, or split."""
    import numpy as np
    from data_utils.feeder import MMFI
    from run_peak_confirmation import _pair_ids
    data_home = Path(data_home).resolve()
    root = data_home / 'datasets' / 'Compress'
    action = data_home / 'actions' / 'data_bend.npy'
    if not action.is_file():
        raise FileNotFoundError(f'Missing ORIGINAL motion prior: {action}; copy it from Van, not a synthetic fixture')
    action_sha = sha256(action)
    if action_sha != expected_action_sha:
        raise ValueError('Original motion-prior SHA256 differs from the archived A1 protocol')
    prior = np.load(action, allow_pickle=False)
    if prior.ndim not in (4, 5) or prior.shape[1] != 3 or not np.isfinite(prior).all():
        raise ValueError('Invalid original action array')
    expected = COUNTS if counts is None else counts
    splits, files, identities = {}, {}, []
    for name, split in (('train', 'training'), ('eval', 'test')):
        dataset = MMFI(split=split, data_root=str(root), protocol='protocol1',
                       setting='s1', random_ratio=.8, random_seed=0)
        ids = _pair_ids(dataset)
        if len(ids) != expected[name] or len(set(map(tuple, ids))) != len(ids):
            raise ValueError(f'{name}: expected {expected[name]} distinct pairs, found {len(ids)}')
        identities.append(set(map(tuple, ids)))
        splits[name] = {'n': len(ids), 'pair_ids_sha256': digest(ids)}
        for item in dataset.items:
            for kind, path in (('csi', item['csi']), ('pose', item['kpt'])):
                path = Path(path)
                relative = path.relative_to(root).as_posix()
                if relative in files:
                    continue
                value = np.load(path, allow_pickle=False)
                if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value) or not np.isfinite(value).all():
                    raise ValueError(f'Non-finite/non-real {kind}: {relative}')
                if kind == 'csi' and (value.shape != (3, 114, 10) or value.min() < -1e-6 or value.max() > 1 + 1e-6):
                    raise ValueError(f'Invalid pre-normalized CSI: {relative}, {value.shape}')
                if kind == 'pose' and (value.ndim != 3 or value.shape[1:] != (17, 3)):
                    raise ValueError(f'Invalid 17-joint pose: {relative}')
                stat = path.stat()
                files[relative] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                   'sha256': sha256(path), 'shape': list(value.shape)}
                if len(files) % 10000 == 0:
                    print(f'[Vast input gate] {len(files)} unique selected files hashed', flush=True)
    if identities[0] & identities[1]:
        raise ValueError('TRAIN/TEST pair overlap')
    result = dict(schema=1, dataset='MM-Fi protocol1-s1', data_home=str(data_home),
        root=str(root), action_sha256=action_sha, splits=splits, files=files,
        normalized_by_gate=False, units='metres as supplied', content_hashed=True,
        shared_subjects_across_actions_possible=True)
    result['binding_sha256'] = digest(result)
    return result


def check_receipt(receipt, *, full_hash=False):
    """Stat-check all bound files; optionally reread all bytes (completion gate)."""
    expected = dict(receipt)
    claimed = expected.pop('binding_sha256', None)
    if claimed != digest(expected) or receipt.get('content_hashed') is not True:
        raise ValueError('Dataset receipt is not an intact full-content audit')
    root = Path(receipt['root'])
    home = Path(receipt['data_home']).resolve()
    if root.absolute() != home/'datasets/Compress':
        raise ValueError('Dataset receipt root differs from its data home')
    if sha256(Path(receipt['data_home']) / 'actions' / 'data_bend.npy') != receipt['action_sha256']:
        raise ValueError('Motion prior changed after preflight')
    for relative, bound in receipt['files'].items():
        relative_path = PurePosixPath(relative)
        if (relative_path.is_absolute() or '..' in relative_path.parts
                or '\\' in relative or ':' in relative or not relative_path.parts):
            raise ValueError('Unsafe dataset receipt member path')
        path = root / relative
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (bound['size'], bound['mtime_ns']):
            raise ValueError(f'Input file changed: {relative}')
        if full_hash and sha256(path) != bound['sha256']:
            raise ValueError(f'Input bytes changed: {relative}')
    # File-stat equality does not detect newly added/missing identities or a
    # changed split/order. Rebuild using the frozen loader without normalization.
    from data_utils.feeder import MMFI
    from run_peak_confirmation import _pair_ids
    for name, split in (('train', 'training'), ('eval', 'test')):
        dataset = MMFI(split=split, data_root=str(root), protocol='protocol1',
                       setting='s1', random_ratio=.8, random_seed=0)
        ids = _pair_ids(dataset)
        if {'n': len(ids), 'pair_ids_sha256': digest(ids)} != receipt['splits'][name]:
            raise ValueError('Dataset split identities/order changed: '+name)
    return True
