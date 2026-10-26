"""Authenticate an immutable HPE-Li P1S1 Clean/A1 transfer from Van's PC."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import zipfile

BACKBONE = 'hpeli'
MANIFEST = 'three_tables.resolved.json'
EXPECTED_PRIOR = '2e8fc11192cc3ee7ee3f5544cb3c5e19f771f7f7265dde93af8969af41a2be3e'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unpack(archive, target, expected_sha256):
    """Import once; require the SHA printed on Van's source machine."""
    archive, target = Path(archive), Path(target).resolve()
    if not expected_sha256 or sha(archive).lower() != expected_sha256.lower():
        raise ValueError('HPE-Li source archive SHA256 differs from Van export receipt')
    if target.exists():
        raise FileExistsError('HPE-Li import exists; preserve it and inspect')
    with zipfile.ZipFile(archive) as z:
        names = set()
        for item in z.infolist():
            name = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (not item.filename or name.is_absolute() or '..' in name.parts
                    or '\\' in item.filename or ':' in item.filename
                    or stat.S_ISLNK(mode) or item.filename in names):
                raise ValueError('Unsafe or duplicate HPE-Li ZIP member: ' + item.filename)
            names.add(item.filename)
        if MANIFEST not in names or z.testzip() is not None:
            raise ValueError('HPE-Li archive lacks manifest or failed CRC')
        target.mkdir(parents=True)
        for item in z.infolist():
            path = target.joinpath(*PurePosixPath(item.filename).parts)
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as source, path.open('xb') as dest:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                        dest.write(block)


def _manifest(root):
    from run_a1_three_tables import _validate_manifest
    manifest = json.loads((Path(root) / MANIFEST).read_text(encoding='utf-8'))
    _validate_manifest(manifest)  # Validate original paths/config; never rewrite the manifest.
    return manifest


def _local_path(root, original, manifest):
    relative = PureWindowsPath(original).relative_to(PureWindowsPath(manifest['outdir']))
    if not relative.parts or any(part in ('', '.', '..') for part in relative.parts):
        raise ValueError('Source path escapes frozen HPE-Li run')
    return Path(root).joinpath(*relative.parts)


def cell(root, key):
    if key not in ('clean', 'a1'):
        raise ValueError('Only P1S1 seed42 Clean/A1 in scope')
    root = Path(root).resolve()
    manifest = _manifest(root)
    found = [item for item in manifest['cells'] if item['seed'] == 42 and item['key'] == key]
    if len(found) != 1:
        raise ValueError('Missing/duplicate HPE-Li source cell: ' + key)
    result = copy.deepcopy(found[0])
    for name in ('ckpt_dir', 'artifact_dir', 'eval_cache'):
        if name in result:
            result[name] = str(_local_path(root, result[name], manifest))
    return result


def checkpoint_path(root, key):
    return Path(cell(root, key)['ckpt_dir']) / 'checkpoint.pt'


def authenticate(root, native_dir=None, *, load_weights=True):
    import numpy as np
    import torch
    import yaml
    from train_backdoor import _config_fingerprint
    from models.factory import build_model

    root = Path(root).resolve()
    native_dir = Path(native_dir)
    manifest = _manifest(root)
    result = {'attack_plan_sha256': manifest['plan_sha256'], 'cells': {}}
    for key in ('clean', 'a1'):
        imported = cell(root, key)
        original = next(x for x in manifest['cells'] if x['seed'] == 42 and x['key'] == key)
        folder = Path(imported['ckpt_dir'])
        ev = Path(imported['artifact_dir'])
        record = json.loads((ev / 'evaluation.json').read_text(encoding='utf-8'))
        binding = record['provenance']['binding']
        if (record.get('status') != 'complete' or record.get('key') != key
                or record.get('seed') != 42 or record['provenance'].get('checkpoint_epoch') != 49
                or digest({k: v for k, v in record.items() if k != 'record_sha256'}) != record['record_sha256']
                or digest(binding) != record['binding_sha256']
                or PureWindowsPath(binding['checkpoint_folder']) != PureWindowsPath(original['ckpt_dir'])):
            raise ValueError('Original HPE-Li evaluation identity differs: ' + key)
        for name, expected in binding['evaluation_source_sha256'].items():
            if sha(native_dir / name) != expected:
                raise ValueError('Original HPE-Li evaluator source differs: ' + name)
        artifacts = binding['artifacts_sha256']
        for name, expected in artifacts.items():
            if name == 'reference_action':
                if expected != EXPECTED_PRIOR:
                    raise ValueError('Original HPE-Li motion prior differs')
            elif sha(folder / name) != expected:
                raise ValueError('Original HPE-Li artifact differs: ' + key + '/' + name)
        if sha(ev / 'per_sample.npz') != record['provenance']['per_sample_sha256']:
            raise ValueError('Original HPE-Li per-sample hash differs')
        cache = json.loads((folder / 'eval_cache.json').read_text(encoding='utf-8'))
        cfg = cache['cfg']
        yaml_cfg = yaml.safe_load((folder / 'config.resolved.yaml').read_text(encoding='utf-8'))
        recorded_fp = yaml_cfg.pop('config_fingerprint', None)
        if (cfg != original['cfg'] or yaml_cfg != cfg or cache['trained_epochs'] != 50
                or _config_fingerprint(cfg) != cache['cfg_fingerprint']
                or cache['cfg_fingerprint'] != record['config_fingerprint']
                or recorded_fp not in (None, record['config_fingerprint'])):
            raise ValueError('Original HPE-Li config/cache differs: ' + key)
        audit = json.loads((folder / 'three_tables_inputs.json').read_text(encoding='utf-8'))
        for split in ('train', 'eval'):
            expected = binding['ordered_inputs'][split]
            if {name: audit[split][name] for name in ('n', 'pair_ids_sha256')} != expected:
                raise ValueError('Original HPE-Li split audit differs')
        with np.load(ev / 'per_sample.npz', allow_pickle=False) as sample:
            ids = sample['pair_ids']
            if ids.shape != (33264, 3):
                raise ValueError('Original HPE-Li TEST is incomplete')
            converted = [[str(csi), str(pose), int(frame)] for csi, pose, frame in ids.tolist()]
            if digest(converted) != binding['ordered_inputs']['eval']['pair_ids_sha256']:
                raise ValueError('Original HPE-Li TEST identities differ')
        if load_weights:
            blob = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
            if (blob.get('epoch') != 49 or blob.get('cfg_fingerprint') != record['config_fingerprint']
                    or _config_fingerprint(blob['cfg']) != record['config_fingerprint']):
                raise ValueError('Original HPE-Li checkpoint is not final49')
            model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                                dataset='mmfi', pretrained=cfg.get('pretrained', False))
            model.load_state_dict(blob['model'], strict=True)
            del model, blob
        result['cells'][key] = dict(checkpoint_sha256=artifacts['checkpoint.pt'],
            evaluation_record_sha256=record['record_sha256'], cfg=cfg,
            original_metrics=record['metrics'], inputs=binding['ordered_inputs'])
    if result['cells']['clean']['inputs'] != result['cells']['a1']['inputs']:
        raise ValueError('HPE-Li Clean/A1 split differs')
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
        value = dict(n=len(ids), pair_ids_sha256=digest(ids))
        if value != source['cells']['a1']['inputs'][name]:
            raise ValueError('HPE-Li local ordered split differs: ' + name)
        result[name] = value
    return result
