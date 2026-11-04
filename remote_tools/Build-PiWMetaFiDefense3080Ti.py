"""Package authenticated 4090 source metadata; never include weights or raw CSI."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def build(root, nad_source, destination):
    plans = {}
    for role in ('clean', 'a1'):
        run = root / f'piw_metafi_{role}_vastenv_s42_4090_v1'
        plan = json.loads((run / 'plan.json').read_text())
        if digest({k:v for k,v in plan.items() if k != 'plan_sha256'}) != plan['plan_sha256']:
            raise ValueError('Source plan digest differs')
        plans[role] = plan
    a, c = plans['a1'], plans['clean']
    if (a['inputs'] != c['inputs'] or a['cfg']['target'] != c['cfg']['target']
            or a['clean_reference']['plan_sha256'] != c['plan_sha256']):
        raise ValueError('Fresh Clean/A1 source relationship differs')
    expected = {
        'clean': '2e61eda62f4a0292ce8a3ff9a7487018482707c1e07329d9e3cdea8783680287',
        'a1': 'c0bbb2440c8d7f235b3aa3e47b49fb85a185442ce9d470482c56fadb00e7f404'}
    checkpoints = {
        'clean': root / 'piw_metafi_clean_vastenv_s42_4090_v1/attacks/metafiplusplus/seed42/clean/checkpoint.pt',
        'a1': root / 'a1_checkpoint.pt'}
    if any(sha(checkpoints[r]) != expected[r] for r in expected):
        raise ValueError('Checkpoint pair differs from the user-confirmed CEI handoff')
    payload = {}
    frozen = root / 'piw_metafi_clean_vastenv_s42_4090_v1/source_snapshot/ATKBackd'
    for name in ('metafi_plus_plus_piw3d.py', 'metafi_plus_plus.py', 'metafi_channel_transformer.py'):
        relative = 'models/' + name
        data = (frozen / relative).read_bytes()
        if any(hashlib.sha256(data).hexdigest() != p['sources'][relative] for p in plans.values()):
            raise ValueError('Clean/A1 model source differs: ' + relative)
        payload['piw_metafi_frozen/' + name] = data
    payload['piw_metafi_frozen/__init__.py'] = b''
    payload['piw_metafi_nad_runtime_v2.py'] = nad_source.read_bytes()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / 'model_source.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(payload.items()):
            item = zipfile.ZipInfo(name, date_time=(2026, 11, 4, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(item, data)
    inputs = a['inputs']['splits']
    with (destination / 'source_inputs.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode(), mtime=0))
    metadata = dict(schema=1, checkpoints={r: dict(profile=plans[r]['profile'],
        method=r, plan_sha256=plans[r]['plan_sha256'], checkpoint_sha256=expected[r], epoch=199)
        for r in ('clean', 'a1')}, target=a['cfg']['target'], seed=42,
        action_sha256=a['action_sha256'], feeder_sha256=a['sources']['data_utils/feeder.py'],
        trigger_sources={n: a['sources'][n] for n in ('piw_v2_adapter.py', 'piw_a1_adapter.py',
            'attack/trigger.py', 'attack/method_drafts.py', 'attack/tsba.py')},
        trigger_protocol=a['trigger']['protocol'],
        source_files={n: hashlib.sha256(b).hexdigest() for n,b in payload.items()},
        assets={n: sha(destination/n) for n in ('model_source.zip', 'source_inputs.json.gz')},
        provenance='Fresh 4090 Clean/A1 plans and SHA-bound final199 checkpoints; no weights/raw CSI in this package')
    metadata['metadata_sha256'] = digest(metadata)
    with (destination / 'source_pair.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--nad-source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    result = build(args.source_root, args.nad_source, args.destination)
    print('PACKAGED_SOURCE_PAIR', result['metadata_sha256'])
