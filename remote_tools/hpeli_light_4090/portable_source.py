"""Authenticate local HPE-Li MM-Fi Clean/A1 final49 victims without copying them."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


BACKBONE = 'hpeli'
MANIFEST = 'three_tables.resolved.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cell(root, key):
    if key not in ('clean', 'a1'):
        raise ValueError('Only original Clean/A1 seed42 source is in scope')
    from run_a1_three_tables import _validate_manifest

    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    _validate_manifest(manifest)
    if root != Path(manifest['outdir']).resolve():
        raise ValueError('HPE-Li source root differs from frozen manifest path')
    found = [item for item in manifest['cells']
             if item['seed'] == 42 and item['key'] == key]
    if len(found) != 1:
        raise ValueError('Missing or duplicate seed42 source cell: ' + key)
    return found[0]


def checkpoint_path(root, key):
    return Path(cell(root, key)['ckpt_dir']) / 'checkpoint.pt'


def authenticate(root, native_dir=None, *, load_weights=True):
    """Replay the repository's saved source/evaluation audit for both controls."""
    import numpy as np
    import torch
    from a1_three_tables_eval import load_evaluation
    from eval_a1_target_pck import _load_weights
    from run_a1_defenses import _audit_source

    root = Path(root).resolve()
    result = dict(attack_plan_sha256=None, cells={})
    manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    result['attack_plan_sha256'] = manifest['plan_sha256']
    for key in ('clean', 'a1'):
        original = cell(root, key)
        run, artifacts, _, _ = _audit_source(original, data=False)
        evaluation = load_evaluation(original, original['artifact_dir'])
        if evaluation is None or evaluation['status'] != 'complete':
            raise ValueError(f'Original HPE-Li {key} evaluation is incomplete')
        binding = evaluation['provenance']['binding']
        if binding['key'] != key or binding['seed'] != 42:
            raise ValueError('Wrong HPE-Li source identity')
        if sha(checkpoint_path(root, key)) != artifacts['checkpoint.pt']:
            raise ValueError('HPE-Li source checkpoint changed')
        with np.load(Path(original['artifact_dir']) / 'per_sample.npz',
                     allow_pickle=False) as sample:
            if sample['pair_ids'].shape != (33264, 3):
                raise ValueError('HPE-Li original evaluation lacks full TEST')
        if load_weights:
            model, _ = _load_weights(run, 'cpu')
            if model is None:
                raise ValueError('HPE-Li source model did not load')
            del model
        result['cells'][key] = dict(
            checkpoint_sha256=artifacts['checkpoint.pt'],
            evaluation_record_sha256=evaluation['record_sha256'],
            cfg=copy.deepcopy(run['cfg']), original_metrics=evaluation['metrics'],
            inputs=binding['ordered_inputs'])
    if result['cells']['clean']['inputs'] != result['cells']['a1']['inputs']:
        raise ValueError('HPE-Li Clean/A1 original splits differ')
    return result


def local_config(source, data_home, device='cuda:0'):
    cfg = copy.deepcopy(source['cells']['a1']['cfg'])
    cfg.update(dataset_root=str(Path(data_home).resolve() / 'datasets/Compress'),
               action_npy=str(Path(data_home).resolve() / 'actions/data_bend.npy'),
               device=device, num_workers=0, data_parallel=False)
    return cfg


def bind_local_ids(source, data_home, train, test):
    from run_peak_confirmation import _pair_ids

    result = {}
    for name, dataset in (('train', train), ('eval', test)):
        ids = _pair_ids(dataset)
        item = dict(n=len(ids), pair_ids_sha256=digest(ids))
        if item != source['cells']['a1']['inputs'][name]:
            raise ValueError('HPE-Li local ordered split differs: ' + name)
        result[name] = item
    return result
