"""Export immutable HPE-Li P1S1 Clean/A1 sources from Van's Windows run.

Run on the source machine. This only reads the final-49 source and writes a
new transfer ZIP; it does not copy P3 checkpoints or alter any experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys
import zipfile


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def inventory(root: Path) -> dict[str, Path]:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / 'ATKBackd'))
    from run_a1_three_tables import _validate_manifest

    root = root.resolve()
    manifest_path = root / 'three_tables.resolved.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    _validate_manifest(manifest)
    if PureWindowsPath(manifest['outdir']) != PureWindowsPath(str(root)):
        raise ValueError('Source root differs from original frozen manifest')
    files: dict[str, Path] = {'three_tables.resolved.json': manifest_path}
    for key in ('clean', 'a1'):
        found = [c for c in manifest['cells'] if c['seed'] == 42 and c['key'] == key]
        if len(found) != 1:
            raise ValueError('Missing/duplicate HPE-Li seed42 ' + key)
        cell = found[0]
        ckpt_rel = PureWindowsPath(cell['ckpt_dir']).relative_to(PureWindowsPath(manifest['outdir']))
        eval_rel = PureWindowsPath(cell['artifact_dir']).relative_to(PureWindowsPath(manifest['outdir']))
        ckpt = root.joinpath(*ckpt_rel.parts)
        ev = root.joinpath(*eval_rel.parts)
        record = json.loads((ev / 'evaluation.json').read_text(encoding='utf-8'))
        binding = record['provenance']['binding']
        if (record['status'] != 'complete' or record['key'] != key or record['seed'] != 42
                or digest({k: v for k, v in record.items() if k != 'record_sha256'}) != record['record_sha256']
                or digest(binding) != record['binding_sha256']
                or PureWindowsPath(binding['checkpoint_folder']) != PureWindowsPath(cell['ckpt_dir'])):
            raise ValueError('Frozen evaluation identity differs: ' + key)
        for name, expected in binding['artifacts_sha256'].items():
            if name == 'reference_action':
                if sha(Path(cell['cfg']['action_npy'])) != expected:
                    raise ValueError('Motion prior differs')
                continue
            path = ckpt / name
            if sha(path) != expected:
                raise ValueError('Original artifact differs: ' + key + '/' + name)
            files['/'.join((*ckpt_rel.parts, name))] = path
        for name in ('evaluation.json', 'per_sample.npz'):
            path = ev / name
            files['/'.join((*eval_rel.parts, name))] = path
        if sha(ev / 'per_sample.npz') != record['provenance']['per_sample_sha256']:
            raise ValueError('Original per-sample file differs: ' + key)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path.home() / 'backdooranalog/runs/mmfi_three_tables_4090_win_fresh_v1')
    parser.add_argument('--output', type=Path, default=Path.home() / 'Downloads/mmfi_hpeli_p1s1_s42_clean_a1_source.zip')
    args = parser.parse_args()
    files = inventory(args.source_root)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, path in sorted(files.items()):
                archive.write(path, name)
    print(json.dumps({'archive': str(output), 'sha256': sha(output),
                      'files': len(files), 'source_root': str(args.source_root.resolve())}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
