"""Explicit MM-Fi download/extraction with verified, non-overwriting resume.

No download occurs without --download. Archive members may not escape the
staging directory, be links/devices, or overwrite existing different content.
The experiment receives the original processed arrays, not a new split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import zipfile
import zlib

DRIVE_ID = '1GXgLXzOhzdqyZpB0zDNqbhpzkuyA4TVn'
ARCHIVE_NAME = 'Compress_wifi_pose_only.zip'


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_new_json(path, record):
    path = Path(path)
    encoded = json.dumps(record, indent=2, sort_keys=True)+'\n'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('Existing receipt differs; preserve and inspect: '+str(path))
        return
    with path.open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def member_name(info):
    raw = info.filename.replace('\\', '/')
    path = PurePosixPath(raw)
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if (not raw or raw.startswith('/') or any(piece in ('..', '') or ':' in piece for piece in path.parts)
            or '\x00' in raw or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
            or info.flag_bits & 1 or path.name == '.extraction.json'
            or path.name.endswith('.operator-extracting')):
        raise ValueError('Unsafe/encrypted/special ZIP member: '+repr(info.filename))
    # Dot-only paths are not publishable; nested dot components are normalized.
    if not path.parts:
        raise ValueError('Empty ZIP member')
    return path.as_posix()


def safe_entries(archive):
    result, seen = [], set()
    for info in archive.infolist():
        name = member_name(info)
        folded = name.casefold()
        if folded in seen:
            raise ValueError('Duplicate/case-colliding archive member: '+name)
        seen.add(folded)
        result.append((name, info))
    files = {name.casefold() for name, info in result if not info.is_dir()}
    for name, _ in result:
        if any(parent.as_posix().casefold() in files for parent in PurePosixPath(name).parents
               if parent.parts):
            raise ValueError('Archive file/directory collision: '+name)
    return result


def assert_safe_destination(root, target):
    root, target = Path(root), Path(target)
    if root.is_symlink():
        raise ValueError('Extraction root must not be a symlink')
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError('Target escaped extraction root') from error
    for part in (target, *target.parents):
        if part.is_symlink():
            raise ValueError('Existing extraction path is a symlink: '+str(part))
        if part == root:
            break


def content_matches(path, info):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size != info.file_size:
        return False
    value = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value = zlib.crc32(chunk, value)
    return value & 0xffffffff == info.CRC


def extract_verified(archive_path, destination, resume=False):
    archive_path, destination = Path(archive_path).resolve(), Path(destination).absolute()
    if destination.is_symlink() or destination.resolve() != destination:
        raise ValueError('Extraction root must not be a symlink')
    archive_sha = file_sha(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        entries = safe_entries(archive)
        summary = [dict(name=name, size=info.file_size, crc=info.CRC,
                        directory=info.is_dir()) for name, info in entries]
        binding = dict(schema=1, archive_sha256=archive_sha,
                       archive_bytes=archive_path.stat().st_size,
                       members_sha256=hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest(),
                       files=sum(not row['directory'] for row in summary),
                       uncompressed_bytes=sum(row['size'] for row in summary))
        receipt = destination/'.extraction.json'
        if destination.exists():
            if not resume:
                raise FileExistsError('Extraction output exists; inspect and use explicit --resume')
            if not receipt.is_file() or json.loads(receipt.read_text(encoding='utf-8')) != binding:
                raise ValueError('Resume requires the same bound archive and extraction receipt')
        else:
            ancestor = destination.parent
            while not ancestor.exists():
                ancestor = ancestor.parent
            if shutil.disk_usage(ancestor).free < binding['uncompressed_bytes']:
                raise RuntimeError('Insufficient free space for archive uncompressed bytes')
            destination.mkdir(parents=True)
            write_new_json(receipt, binding)
        for index, (name, info) in enumerate(entries, 1):
            target = destination.joinpath(*PurePosixPath(name).parts)
            assert_safe_destination(destination, target)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not content_matches(target, info):
                    raise ValueError('Existing extracted file differs; never overwrite: '+str(target))
                continue
            temporary = target.with_name(target.name+'.operator-extracting')
            assert_safe_destination(destination, temporary)
            if temporary.exists():
                if not resume or not temporary.is_file():
                    raise ValueError('Unexpected partial extraction path: '+str(temporary))
                # This exact staging file is owned by the matching archive receipt.
                temporary.unlink()
            try:
                with archive.open(info) as source, temporary.open('xb') as sink:
                    for chunk in iter(lambda: source.read(1024*1024), b''):
                        sink.write(chunk)
                if not content_matches(temporary, info):
                    raise ValueError('ZIP member CRC/size mismatch: '+name)
                # Link publication is exclusive; never replaces an existing path.
                os.link(temporary, target)
                temporary.unlink()
            except BaseException:
                raise
            if index % 5000 == 0:
                print(f'[MM-Fi extraction] {index}/{len(entries)} members', flush=True)
        return binding


def find_mmfi_root(destination):
    destination = Path(destination).resolve()
    candidates = [destination, *(path for path in destination.rglob('*') if path.is_dir()
                                and path.name == 'E01')]
    roots = set()
    for candidate in candidates:
        root = candidate.parent if candidate.name == 'E01' else candidate
        if all((root/scene).is_dir() for scene in ('E01', 'E02', 'E03', 'E04')):
            roots.add(root)
    if len(roots) != 1:
        raise ValueError(f'Expected exactly one processed MM-Fi root E01..E04, found {len(roots)}')
    return roots.pop()


def validate_mmfi(root):
    """Use the existing loader without changing arrays or generating a split."""
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository/'ATKBackd'))
    import numpy as np
    from data_utils.feeder import MMFI
    datasets = {split: MMFI(split, str(root), protocol='protocol1', setting='s1',
                            random_ratio=.8, random_seed=0)
                for split in ('training', 'validation')}
    counts = {key: len(value) for key, value in datasets.items()}
    if counts != {'training': 133056, 'validation': 33264}:
        raise ValueError('Full frozen P1S1 counts differ; no subset/resplit: '+repr(counts))
    for dataset in datasets.values():
        for index in (0, len(dataset)-1):
            item = dataset[index]
            raw = np.load(dataset.items[index]['csi'], allow_pickle=False)
            if (raw.shape != (3, 114, 10) or not np.isfinite(raw).all()
                    or raw.min() < -1e-6 or raw.max() > 1+1e-6
                    or item['pose'].shape != (1, 17, 3) or not np.isfinite(item['pose']).all()):
                raise ValueError('Unexpected processed MM-Fi sample schema/range')
    identities = {key: hashlib.sha256('\n'.join(item['name'] for item in value.items).encode()).hexdigest()
                  for key, value in datasets.items()}
    return dict(protocol='protocol1', setting='s1', split_seed=0, train_ratio=.8,
                counts=counts, ordered_names_sha256=identities,
                csi_shape=[3, 114, 10], pose_shape=[1, 17, 3],
                note='Sample validation only; scientific runner performs its own complete input audit')


def publish_dataset_link(data_home, root):
    link = Path(data_home)/'datasets/Compress'
    root = Path(root).resolve()
    if link.exists() or link.is_symlink():
        if not link.is_dir() or link.resolve() != root:
            raise FileExistsError('Default MM-Fi path already points elsewhere; do not overwrite: '+str(link))
    else:
        if os.name != 'posix':
            raise RuntimeError('Default dataset link publication is Linux-only; extracted files remain intact')
        link.symlink_to(root, target_is_directory=True)
    return link


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', required=True, type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--tools-python', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args(argv)
    home = args.data_home.resolve()
    archive = args.archive.resolve() if args.archive else home/'downloads'/ARCHIVE_NAME
    if args.download:
        if args.list_only:
            parser.error('--list-only cannot download')
        if args.tools_python is None or not args.tools_python.is_file():
            parser.error('--download requires the isolated --tools-python interpreter')
        archive.parent.mkdir(parents=True, exist_ok=True)
        command = [str(args.tools_python), '-m', 'gdown',
                   f'https://drive.google.com/uc?id={DRIVE_ID}', '--continue', '-O', str(archive)]
        subprocess.run(command, check=True)
    if not archive.is_file():
        raise FileNotFoundError('Archive missing; download requires explicit --download: '+str(archive))
    if args.list_only:
        with zipfile.ZipFile(archive) as handle:
            entries = safe_entries(handle)
            print(json.dumps(dict(archive=str(archive), compressed_bytes=archive.stat().st_size,
                uncompressed_bytes=sum(info.file_size for _, info in entries),
                file_count=sum(not info.is_dir() for _, info in entries),
                first_members=[name for name, _ in entries[:12]]), indent=2))
        return 0
    destination = home/'datasets/mmfi_archive_v1'
    binding = extract_verified(archive, destination, args.resume)
    root = find_mmfi_root(destination)
    validated = validate_mmfi(root)
    link = publish_dataset_link(home, root)
    receipt = dict(schema=1, drive_id=DRIVE_ID, archive=str(archive), extraction=str(destination),
                   mmfi_root=str(root), default_root=str(link), archive_binding=binding,
                   validation=validated)
    write_new_json(home/'dataset_receipt.json', receipt)
    print('[MM-Fi] VERIFIED_PROCESSED_DATASET_READY; no experiment launched')
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
