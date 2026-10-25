"""Prepare the original processed MM-Fi archive for P3 on Windows.

The default mode only prints paths. Downloading needs explicit ``--download``;
extraction needs ``--mode prepare``. Neither mode starts training. Archive
validation and non-overwriting extraction reuse the audited Vast data helper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from remote_linux.vast_seed42.download_data import (  # noqa: E402
    ARCHIVE_NAME, DRIVE_ID, extract_verified, find_mmfi_root,
    safe_entries, validate_mmfi, write_new_json,
)


def inspect_archive(archive_path):
    """Validate the ZIP directory before any extraction or output writes."""
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f'MM-Fi archive missing: {archive_path}')
    with zipfile.ZipFile(archive_path) as archive:
        entries = safe_entries(archive)
    files = [name for name, info in entries if not info.is_dir()]
    if not files or any(not name.startswith('Compress/') for name in files):
        raise ValueError('Archive must contain only processed files under Compress/')
    if any(name != 'Compress/' and not name.startswith('Compress/')
           for name, _ in entries):
        raise ValueError('Archive has a member outside Compress/')
    missing = [scene for scene in ('E01', 'E02', 'E03', 'E04')
               if not any(name.startswith(f'Compress/{scene}/') for name in files)]
    if missing:
        raise ValueError(f'Archive is missing MM-Fi scenes: {missing}')
    return {
        'archive': str(archive_path),
        'compressed_bytes': archive_path.stat().st_size,
        'files': len(files),
        'uncompressed_bytes': sum(info.file_size for _, info in entries),
        'root': 'Compress',
        'first_members': [name for name, _ in entries[:8]],
    }


def download_archive(archive_path, tools_python, *, resume=False):
    """Run gdown only after the caller explicitly requested a download."""
    archive_path = Path(archive_path).absolute()
    tools_python = Path(tools_python).absolute()
    if archive_path.is_symlink():
        raise ValueError('Archive path must not be a symlink')
    if not tools_python.is_file():
        raise FileNotFoundError(f'gdown Python interpreter missing: {tools_python}')
    if archive_path.exists():
        if not resume:
            raise FileExistsError('Archive already exists; inspect it or use --resume')
        try:
            inspect_archive(archive_path)
        except zipfile.BadZipFile:
            pass  # Incomplete download; gdown --continue owns the resume.
        else:
            print(f'[MM-Fi] Existing complete ZIP retained: {archive_path}', flush=True)
            return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(tools_python), '-m', 'gdown',
               f'https://drive.google.com/uc?id={DRIVE_ID}',
               '-O', str(archive_path)]
    if resume:
        command.insert(-2, '--continue')
    subprocess.run(command, check=True)


def prepare_data(data_home, archive_path, *, resume=False):
    """Publish datasets/Compress only from the bound, validated archive."""
    data_home = Path(data_home).absolute()
    archive_path = Path(archive_path).absolute()
    metadata = inspect_archive(archive_path)
    destination = data_home / 'datasets'
    if destination.is_dir():
        if destination.is_symlink() or destination.resolve() != destination:
            raise ValueError('Dataset directory must not be a symlink')
        if next(destination.iterdir(), None) is None:
            # New-Item may have made this directory already. Remove only the
            # verified empty leaf, so extract_verified can create its binding.
            destination.rmdir()
    binding = extract_verified(archive_path, destination, resume=resume)
    root = find_mmfi_root(destination)
    expected = destination / 'Compress'
    if root != expected.resolve() or expected.is_symlink():
        raise ValueError(f'Unexpected processed MM-Fi root: {root}')
    validation = validate_mmfi(root)
    receipt = {
        'schema': 1,
        'drive_id': DRIVE_ID,
        'archive': str(archive_path),
        'destination': str(destination),
        'dataset_root': str(root),
        'archive_metadata': metadata,
        'archive_binding': binding,
        'validation': validation,
        'note': 'P1S1 sample validation only; P3 runner audits S1/S2/S3 before training',
    }
    write_new_json(data_home / 'dataset_receipt_p3_windows.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--mode', choices=('plan', 'list', 'prepare'), default='plan')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--tools-python', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)

    data_home = args.data_home.absolute()
    archive = (args.archive.absolute() if args.archive else
               data_home / 'downloads' / ARCHIVE_NAME)
    if args.mode == 'plan':
        if args.download or args.resume:
            parser.error('plan is read-only; use list/prepare for download or resume')
        print(json.dumps({'mode': 'plan', 'drive_id': DRIVE_ID,
                          'archive': str(archive),
                          'dataset_root': str(data_home / 'datasets' / 'Compress'),
                          'receipt': str(data_home / 'dataset_receipt_p3_windows.json'),
                          'download': False, 'extract': False, 'train': False}, indent=2))
        return 0
    if args.download:
        if args.tools_python is None:
            parser.error('--download requires --tools-python with gdown installed')
        download_archive(archive, args.tools_python, resume=args.resume)
    if args.mode == 'list':
        print(json.dumps(inspect_archive(archive), indent=2))
        print('LIST_ONLY; no extraction/training launched', flush=True)
        return 0
    receipt = prepare_data(data_home, archive, resume=args.resume)
    print('VERIFIED_PROCESSED_MMFI_READY; no training launched', flush=True)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
