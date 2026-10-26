"""Authenticate a read-only Linux checkpoint import; create a new Windows run."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

EXPECTED_PRIOR = '2e8fc11192cc3ee7ee3f5544cb3c5e19f771f7f7265dde93af8969af41a2be3e'
BACKBONE = 'wiflow'
SNAPSHOT_SHA256 = 'b2c92d6fe20e992f42a7e19d514dbeb0fc43f5d91c1c58b19860010ec0181b03'
ATTACK_PREFIX = 'mmfi_wiflow_s42_attacks_only_v2/attacks'
SOURCE_PREFIX = 'backdoor-analog/ATKBackd'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def members(archive):
    if sha(archive) != SNAPSHOT_SHA256:
        raise ValueError('WiFlow snapshot SHA256 differs from the audited 20261025 archive')
    with zipfile.ZipFile(archive) as f:
        found = {}
        for m in f.infolist():
            name = PurePosixPath(m.filename)
            mode = m.external_attr >> 16
            if (name.is_absolute() or '..' in name.parts or '\\' in m.filename
                    or ':' in m.filename or not m.filename or stat.S_ISLNK(mode)):
                raise ValueError('Unsafe archive path: ' + m.filename)
            if m.filename in found:
                raise ValueError('Linked/duplicate archive member: ' + m.filename)
            found[m.filename] = m
        for required in (ATTACK_PREFIX + '/vast_seed42_attack.resolved.json',
                         SOURCE_PREFIX + '/vast_seed42_defense_adapter.py'):
            if required not in found:
                raise ValueError('WiFlow snapshot lacks required source: ' + required)
        if f.testzip() is not None:
            raise ValueError('WiFlow snapshot CRC failed')
        return found


def unpack(archive, target):
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError('Import exists: preserve it; do not overwrite')
    inventory = members(archive)
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as f:
        for name, member in inventory.items():
            path = target.joinpath(*PurePosixPath(name).parts)
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with f.open(member) as source, path.open('xb') as dest:
                    for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                        dest.write(chunk)


def authenticate(root, native_dir, *, load_weights=True):
    """Validate stored source hashes and real weights without rewriting paths."""
    import numpy as np
    root, native_dir = Path(root), Path(native_dir)
    manifest = json.loads((root / 'vast_seed42_attack.resolved.json').read_text())
    if (manifest['backbones'] != [BACKBONE] or manifest['seeds'] != [42]
            or digest({k: v for k, v in manifest.items() if k != 'plan_sha256'}) != manifest['plan_sha256']):
        raise ValueError('Not the canonical WiFlow seed42 attack manifest')
    result = dict(attack_plan_sha256=manifest['plan_sha256'], cells={})
    for key in ('clean', 'a1'):
        folder = root / BACKBONE / 'seed42' / key
        evaluation = root / BACKBONE / 'evaluations' / ('s42__' + key)
        r = json.loads((evaluation / 'evaluation.json').read_text())
        b = r['provenance']['binding']
        if (r['key'] != key or r['seed'] != 42 or r['status'] != 'complete'
                or r['provenance']['checkpoint_epoch'] != 49
                or digest({k: v for k, v in r.items() if k != 'record_sha256'}) != r['record_sha256']
                or digest(b) != r['binding_sha256']):
            raise ValueError('Original evaluation record differs: ' + key)
        for name, expected in b['evaluation_source_sha256'].items():
            if sha(native_dir / name) != expected:
                raise ValueError('Frozen scientific source differs: ' + name)
        for name, expected in b['artifacts_sha256'].items():
            if name == 'reference_action':
                if expected != EXPECTED_PRIOR:
                    raise ValueError('Original motion prior differs')
            elif sha(folder / name) != expected:
                raise ValueError('Original checkpoint/config artifact differs: ' + name)
        if sha(evaluation / 'per_sample.npz') != r['provenance']['per_sample_sha256']:
            raise ValueError('Original evaluation NPZ differs')
        cache = json.loads((folder / 'eval_cache.json').read_text())
        from train_backdoor import _config_fingerprint
        if (cache['trained_epochs'] != 50 or cache['cfg_fingerprint'] != r['config_fingerprint']
                or _config_fingerprint(cache['cfg']) != r['config_fingerprint']):
            raise ValueError('Original model configuration differs')
        if load_weights:
            import torch
            blob = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
            if (blob['epoch'] != 49 or blob['cfg_fingerprint'] != r['config_fingerprint']
                    or _config_fingerprint(blob['cfg']) != r['config_fingerprint']):
                raise ValueError('Original checkpoint is not final49')
            from vast_seed42_adapter import build_model
            model = build_model(BACKBONE, num_keypoints=17, num_coor=3, num_person=1,
                                subcarrier_num=114, dataset='mmfi', pretrained=False)
            model.load_state_dict(blob['model'], strict=True)
            del model, blob
        with np.load(evaluation / 'per_sample.npz', allow_pickle=False) as a:
            ids = a['pair_ids']
            if ids.shape != (33264, 3):
                raise ValueError('Original TEST is not full ordered split')
            converted = [[str(csi), str(pose), int(frame)] for csi, pose, frame in ids.tolist()]
            if digest(converted) != b['ordered_inputs']['eval']['pair_ids_sha256']:
                raise ValueError('Original NPZ identities differ')
        result['cells'][key] = dict(checkpoint_sha256=sha(folder / 'checkpoint.pt'),
            evaluation_record_sha256=r['record_sha256'], cfg=cache['cfg'],
            original_metrics=r['metrics'], inputs=b['ordered_inputs'])
    if result['cells']['clean']['inputs'] != result['cells']['a1']['inputs']:
        raise ValueError('Clean and A1 use different splits')
    return result


def local_config(source, data_home, device='cuda:0'):
    cfg = copy.deepcopy(source['cells']['a1']['cfg'])
    cfg.update(dataset_root=str(Path(data_home).resolve() / 'datasets/Compress'),
               action_npy=str(Path(data_home).resolve() / 'actions/data_bend.npy'),
               device=device, num_workers=0, data_parallel=False)
    return cfg


def bind_local_ids(source, data_home, train, test):
    """Check ordered dataset-relative IDs across the Linux and Windows roots."""
    from run_peak_confirmation import _pair_ids
    report = {}
    for name, dataset in (('train', train), ('eval', test)):
        ids = _pair_ids(dataset)
        value = dict(n=len(ids), pair_ids_sha256=digest(ids))
        if value != source['cells']['a1']['inputs'][name]:
            raise ValueError('Local ordered split differs from original: ' + name)
        report[name] = value
    return report
