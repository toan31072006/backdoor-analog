"""Export the completed PiW3D HPE-Li v3 Clean/A1 reference for another host.

The archive contains only fixed experiment artifacts. It never changes the
historical run and does not include the dataset, Python environment or keys.
Run with the same Python environment and source checkout as the v3 run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ATKBackd'))

import run_piw_hpeli_lr_control as v3  # noqa: E402


ARTIFACTS = (
    v3.MANIFEST,
    'evaluation.json',
    'predictions.npz',
    'poison.json',
    'hpeli/seed42/clean/checkpoint.pt',
    'hpeli/seed42/a1/checkpoint.pt',
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Artifact is not a regular file in the v3 run: {relative}')
    return path


def build_reference_record(root: Path):
    manifest = v3.load_manifest(root / v3.MANIFEST)
    evaluation = v3.verify_complete(manifest)
    if evaluation is None:
        raise ValueError('PiW HPE-Li v3 Clean/A1 evaluation is incomplete')
    cfg = manifest['cfg']
    if (cfg['seed'] != 42 or cfg['epochs'] != 200 or
            cfg['final_epoch'] != 199 or cfg['batch_size'] != 32 or
            cfg['lr'] != .001 or cfg['target']['version'] != 'piw_body_hip_axis_v2'):
        raise ValueError('Unexpected v3 scientific recipe')
    clean = v3.checkpoint(manifest, 'clean', evaluation['execution'], final=True)
    a1 = v3.checkpoint(manifest, 'a1', evaluation['execution'], final=True)
    model = v3.base.build_model()
    model.load_state_dict(clean['model'], strict=True)
    files = {}
    for name in ARTIFACTS:
        path = _inside(root, name)
        files[name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    if any(evaluation['artifacts'][name] != files[name]['sha256'] for name in
           ('predictions.npz', 'poison.json',
            'hpeli/seed42/clean/checkpoint.pt',
            'hpeli/seed42/a1/checkpoint.pt')):
        raise ValueError('v3 evaluation artifact hash differs from source')
    return dict(
        schema=1,
        profile='piw_hpeli_v3_portable_clean_reference_s42',
        source_profile=v3.PROFILE,
        source_run_name=root.name,
        source_plan_sha256=manifest['plan_sha256'],
        input_fingerprint=manifest['inputs']['fingerprint'],
        poison_plan_sha256=manifest['poison']['poison_plan_sha256'],
        target=cfg['target'],
        seed=42,
        final_epoch=199,
        training_recipe={key: cfg[key] for key in (
            'optimizer', 'lr', 'weight_decay', 'betas', 'eps',
            'batch_size', 'drop_last', 'amp', 'victim_loss')},
        test_truth_sha256=manifest['test_truth_sha256'],
        clean_checkpoint_epoch=clean['epoch'],
        a1_checkpoint_epoch=a1['epoch'],
        clean_metrics=evaluation['clean_metrics'],
        files=files,
        note='Historical v3 reference; portable weights and TEST arrays. '
             'Training and CUDA execution metadata remain in the original artifacts.'
    )


def export(root: Path, destination: Path) -> dict:
    root = root.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f'Export already exists: {destination}')
    if destination == root or root in destination.parents:
        raise ValueError('Export path must be outside the historical run')
    if not destination.parent.is_dir():
        raise FileNotFoundError(f'Output directory missing: {destination.parent}')
    record = build_reference_record(root)
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + '.', suffix='.writing', dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            archive.writestr('reference_manifest.json',
                             json.dumps(record, indent=2, sort_keys=True) + '\n')
            for name in ARTIFACTS:
                archive.write(_inside(root, name), arcname=name)
        with zipfile.ZipFile(temporary, 'r') as archive:
            if set(archive.namelist()) != {'reference_manifest.json', *ARTIFACTS}:
                raise ValueError('Unexpected ZIP member set')
            if archive.testzip() is not None:
                raise ValueError('ZIP CRC verification failed')
            if json.loads(archive.read('reference_manifest.json')) != record:
                raise ValueError('Reference manifest changed inside ZIP')
            for name, expected in record['files'].items():
                digest = hashlib.sha256()
                with archive.open(name) as source:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != expected['sha256']:
                    raise ValueError(f'ZIP member SHA256 mismatch: {name}')
        if destination.exists():
            raise FileExistsError(f'Export appeared during verification: {destination}')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'archive': str(destination), 'sha256': sha256(destination),
            'bytes': destination.stat().st_size, 'files': len(ARTIFACTS),
            'source_plan_sha256': record['source_plan_sha256'],
            'input_fingerprint': record['input_fingerprint']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    home = Path.home()
    parser.add_argument('--reference-dir', type=Path,
                        default=home / 'backdooranalog/runs/piw_hpeli_body_axis_s42_lr001_v3')
    parser.add_argument('--out', type=Path,
                        default=home / 'Downloads/piw_hpeli_v3_clean_reference.zip')
    args = parser.parse_args()
    print(json.dumps(export(args.reference_dir, args.out), indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
