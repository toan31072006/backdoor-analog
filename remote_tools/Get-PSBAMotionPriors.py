"""Download only official PSBA motion-prior files, without installing packages.

Google Drive IDs, byte counts and NPY headers were verified from the author's
public folder on 2026-10-30. SHA256 is measured after download, NOT a checksum
published by the author. Never overwrite an existing unsourced action file.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

PROJECT = 'https://qichenzheng.github.io/psba-website/'
FOLDER = 'https://drive.google.com/drive/folders/1ZdUKi9FpC3q-5CZmOx5lXOMyOEIcFJTU'
FILES = {
    'cross': dict(id='1irgTzLOMkpfOQjNHI9omf56iJIeHq3dk', bytes=225720128,
                  shape=[1254, 3, 300, 25, 2], label='crossing hands in front'),
    'nod': dict(id='1aGItQm8dSMk0FweZQnAdX7IuAaKHQx6A', bytes=237600128,
                shape=[1320, 3, 300, 25, 2], label='nodding'),
    'bend': dict(id='11EAVcLT0iZk2YeWPEXoSkfP66mVxi8uS', bytes=227520128,
                 shape=[1264, 3, 300, 25, 2], label='bending sideways'),
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Existing receipt differs; no overwrite: {path}')
        return
    # Exclusive creation: do not replace another user's file or receipt.
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def validate(path, action):
    import numpy as np
    expected = FILES[action]
    path = Path(path)
    if path.stat().st_size != expected['bytes']:
        raise ValueError(f'{action}: expected {expected["bytes"]} bytes; incomplete/foreign file')
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    if (list(value.shape) != expected['shape'] or value.dtype.str != '<f4'
            or not value.flags.c_contiguous):
        raise ValueError(f'{action}: official numeric float32 N,3,300,25,2 schema differs')
    flat = value.reshape(-1)
    any_nonzero = False
    for start in range(0, flat.size, 2 * 1024 * 1024):
        block = flat[start:start + 2 * 1024 * 1024]
        if not np.isfinite(block).all():
            raise ValueError(f'{action}: NaN/Inf in action file')
        any_nonzero = any_nonzero or bool(np.any(block != 0))
    if not any_nonzero:
        raise ValueError(f'{action}: empty motion-prior array')
    del block, flat, value  # Release all mmap views before Windows rename.
    return dict(sha256=sha(path), shape=expected['shape'], dtype='<f4', bytes=expected['bytes'])


def source(action):
    expected = FILES[action]
    return dict(schema=1, action=action, file_id=expected['id'],
        source_url=f'https://drive.google.com/file/d/{expected["id"]}/view',
        source_project=PROJECT, source_folder=FOLDER,
        expected_bytes=expected['bytes'], expected_shape=expected['shape'])


def validate_receipt(path, action, measured):
    receipt = json.loads(Path(path).read_text(encoding='utf-8'))
    if (any(receipt.get(k) != v for k, v in source(action).items())
            or any(receipt.get(k) != v for k, v in measured.items())
            or receipt.get('origin') != 'official_psba_download'
            or receipt.get('semantic_approved') is not True):
        raise ValueError('Missing/foreign official-download provenance; no automatic relabeling')
    return receipt


@contextmanager
def download_lock(outdir, action):
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'data_{action}.download.lock').open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def download(action, outdir, *, opener=urllib.request.urlopen):
    with download_lock(outdir, action):
        return _download(action, outdir, opener=opener)


def _download(action, outdir, *, opener=urllib.request.urlopen):
    expected = FILES[action]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / f'data_{action}.npy'
    receipt_path = Path(str(target) + '.provenance.json')
    if target.exists():
        measured = validate(target, action)
        if not receipt_path.is_file():
            raise FileExistsError(f'{target} already exists without official-download provenance. '
                'It was NOT overwritten or auto-approved; inspect it and use the runner manual '
                '--approve-action-labels option only if its semantic label is independently known.')
        validate_receipt(receipt_path, action, measured)
        print(f'VERIFIED_EXISTING {action}: {target}', flush=True)
        return target
    if receipt_path.exists():
        raise FileExistsError(f'Orphan provenance exists: {receipt_path}; inspect before downloading')
    partial = outdir / f'data_{action}.npy.download.part'
    ownership = Path(str(partial) + '.source.json')
    if partial.exists() and not ownership.exists():
        raise FileExistsError(f'Unowned partial file; no overwrite/resume: {partial}')
    write_new(ownership, source(action))
    offset = partial.stat().st_size if partial.exists() else 0
    if not 0 <= offset <= expected['bytes']:
        raise ValueError('Partial file exceeds official byte count')
    url = (f'https://drive.usercontent.google.com/download?id={expected["id"]}'
           '&export=download&confirm=t')
    if offset < expected['bytes']:
        request = urllib.request.Request(url, headers={
            'User-Agent': 'PSBA-motion-prior-downloader/1',
            'Range': f'bytes={offset}-{expected["bytes"] - 1}'})
        with opener(request, timeout=60) as response:
            if response.status == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                     response.headers.get('Content-Range', ''))
                if (not match or int(match[1]) != offset or int(match[3]) != expected['bytes']
                        or int(match[2]) != expected['bytes'] - 1):
                    raise ValueError('Server returned an unexpected/partial range; file left intact')
            elif response.status != 200 or offset != 0:
                raise ValueError('Server did not honor resume; no automatic overwrite/restart')
            kind = response.headers.get('Content-Type', '').lower()
            if 'text/html' in kind:
                raise ValueError('Drive returned a login/quota/confirmation page, not NPY bytes')
            with partial.open('ab' if partial.exists() else 'xb') as handle:
                written, next_report = offset, offset
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    if written + len(block) > expected['bytes']:
                        raise ValueError('Download exceeds official size; refusing extra bytes')
                    handle.write(block); written += len(block)
                    if written >= next_report:
                        print(f'{action}: {written}/{expected["bytes"]} bytes '
                              f'({100 * written / expected["bytes"]:.1f}%)', flush=True)
                        next_report = written + 16 * 1024 * 1024
    measured = validate(partial, action)
    if target.exists():
        raise FileExistsError('Action path appeared during download; no overwrite')
    # Publish without overwriting even if another program creates the target
    # after the existence check. Windows rename is no-replace; POSIX needs link.
    if os.name == 'nt':
        os.rename(partial, target)
    else:
        os.link(partial, target)
        partial.unlink()  # Same bytes survive in the newly published target.
    receipt = dict(source(action), **measured, semantic_approved=True,
        origin='official_psba_download', semantic_label=expected['label'],
        sha256_status='measured after HTTPS download; no author-published SHA256 available',
        downloaded_utc=datetime.now(timezone.utc).isoformat())
    write_new(receipt_path, receipt)
    print(f'DOWNLOADED_AND_VALIDATED {action}: {target}; SHA256={measured["sha256"]}', flush=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--priors', nargs='+', choices=FILES, default=['cross', 'nod'])
    parser.add_argument('--download', action='store_true', help='Explicitly allow HTTPS downloads')
    args = parser.parse_args(argv)
    if len(args.priors) != len(set(args.priors)):
        parser.error('Duplicate action names')
    for action in args.priors:
        if args.download:
            download(action, args.outdir)
        else:
            target = Path(args.outdir) / f'data_{action}.npy'
            measured = validate(target, action)
            validate_receipt(Path(str(target) + '.provenance.json'), action, measured)
            print(f'VERIFIED {action}: {target}', flush=True)
    print('ACTION_FILES_ONLY; no environment installation, GPU use, or training', flush=True)


if __name__ == '__main__':
    main()
