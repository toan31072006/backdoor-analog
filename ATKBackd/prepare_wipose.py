"""Explicit, non-overwriting WiPose MAT-to-NPZ preparation.

DT-Pose's active process.py branch does NOT call its DWT helper.  ``dtpose``
preserves that transpose/reshape; ``official`` declares the additional abs()
conversion needed by the public dataset's complex csi_serial format.  No
downloading, extraction, denoising, split reassignment or guessed unit scaling.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from wipose_data import (CSI_SHAPE, SPLITS, dataset_audit, file_sha,
                         object_sha, read_names)


def read_mat(path):
    """Numeric MATLAB reads only; mat73 is required only for v7.3 files."""
    from scipy.io import loadmat
    try:
        return loadmat(path)
    except NotImplementedError as exc:
        try:
            import mat73
        except ImportError as missing:
            raise RuntimeError('MAT v7.3 input requires mat73; install it explicitly') from missing
        return mat73.loadmat(str(path))


def convert_record(record, schema='auto'):
    if schema not in ('auto', 'dtpose', 'official'):
        raise ValueError('schema must be auto, dtpose or official')
    has_dt = {'CSI', 'SkeletonPoints'}.issubset(record)
    has_official = {'csi_serial', 'jointsVector'}.issubset(record)
    if schema == 'auto':
        if has_dt == has_official:
            raise ValueError('Unknown or ambiguous WiPose MAT field schema')
        schema = 'dtpose' if has_dt else 'official'
    if schema == 'dtpose':
        if not has_dt:
            raise ValueError('dtpose schema requires CSI and SkeletonPoints')
        csi = np.asarray(record['CSI'])
        pose = np.asarray(record['SkeletonPoints'])
        if csi.shape != (5, 30, 3, 3):
            raise ValueError(f'Expected dtpose CSI shape (5,30,3,3), got {csi.shape}')
        if csi.dtype.kind not in 'fiu' or (csi < 0).any():
            raise ValueError('dtpose active branch requires nonnegative real CSI; use explicit official schema for complex csi_serial')
        if pose.size != 54 or pose.dtype.kind not in 'fiu':
            raise ValueError('Expected dtpose numeric SkeletonPoints containing 54 coordinates/confidences')
        if not np.isfinite(csi).all():
            raise ValueError('Nonfinite WiPose source arrays')
        processed = csi.transpose(3, 2, 1, 0).reshape(CSI_SHAPE)
        points = pose.reshape(3, 18)[:2]
        transform = 'author active CSI transpose(3,2,1,0)/reshape; no DWT/abs'
    else:
        if not has_official:
            raise ValueError('official schema requires csi_serial and jointsVector')
        csi = np.asarray(record['csi_serial'])
        points_raw = np.asarray(record['jointsVector'])
        if csi.shape != (5, 30, 3, 3) or csi.dtype.kind not in 'fciu':
            raise ValueError('Expected numeric official csi_serial shape (5,30,3,3)')
        if points_raw.dtype.kind not in 'fiu':
            raise ValueError('Expected numeric official jointsVector')
        if not np.isfinite(csi).all():
            raise ValueError('Nonfinite WiPose source arrays')
        if points_raw.shape == (4, 18):
            points = points_raw[:2]
        elif points_raw.size == 72:
            points = points_raw.reshape(18, 4)[:, :2].T
        else:
            raise ValueError('Expected official jointsVector (18,4), (4,18), or flat 72 values')
        processed = np.abs(csi).transpose(3, 2, 1, 0).reshape(CSI_SHAPE)
        transform = 'public csi_serial abs amplitude then author transpose/reshape; no DWT'
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite WiPose source arrays')
    if not np.isfinite(processed).all():
        raise ValueError('Nonfinite WiPose source arrays')
    if np.max(processed) <= np.min(processed):
        raise ValueError('Degenerate WiPose CSI range')
    return dict(CSI=processed, SkeletonPoints=points), schema, transform


def _raw_names(raw_root, split, generate_lists):
    list_name = SPLITS[split][0]
    path = raw_root / list_name
    directory = 'Train' if split == 'training' else 'Test'
    if path.is_file():
        return read_names(path), dict(origin='supplied ordered list', path=list_name,
                                      sha256=file_sha(path))
    if not generate_lists:
        raise FileNotFoundError(f'Missing {path}; provide original list or explicitly use --generate-lists')
    folder = raw_root / directory
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing raw WiPose directory: {folder}')
    names = sorted(path.stem for path in folder.glob('*.mat') if path.is_file())
    if not names:
        raise ValueError(f'No .mat files in {folder}')
    # Validate with the same identity grammar without creating a temporary list.
    from wipose_data import ACTIONS, NAME_PATTERN
    if len(names) != len(set(names)) or any(not NAME_PATTERN.fullmatch(name)
            or name.split('_')[0] not in ACTIONS for name in names):
        raise ValueError('Unsafe, duplicate or unsupported raw WiPose names')
    return names, dict(origin='generated sorted per-split raw filenames; order adaptation',
                        ordered_names_sha256=object_sha(names))


def _publish_exclusive(path, writer):
    """Publish complete output without replacing another file, on NTFS/ext4.

    A killed writer leaves its explicit .saving file untouched.  Such a file
    requires operator inspection; resume never silently removes/replaces it.
    """
    path = Path(path)
    temporary = path.with_name(path.name + '.saving')
    if path.exists() or temporary.exists():
        raise FileExistsError(f'Refusing existing target or partial .saving file: {path}')
    with temporary.open('xb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    # link is atomic and fails if the destination already exists.  os.replace
    # would overwrite a concurrent target, which preparation must never do.
    os.link(temporary, path)
    temporary.unlink()


def _publish_json(path, value):
    _publish_exclusive(path, lambda stream: stream.write(
        (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')))


def _preparation_plan(raw_root, outdir, schema, generate_lists, selections):
    """Hash all raw inputs before the first processed file can be created."""
    source_bindings = {name: file_sha(Path(__file__).resolve().parent / name)
                       for name in ('prepare_wipose.py', 'wipose_data.py')}
    records = []
    for split, (names, _) in selections.items():
        folder = raw_root / ('Train' if split == 'training' else 'Test')
        for name in names:
            path = folder / f'{name}.mat'
            if not path.is_file():
                raise FileNotFoundError(f'Missing listed raw WiPose MAT: {path}')
            before = path.stat()
            digest = file_sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Raw WiPose MAT changed during initial hash: {path}')
            records.append(dict(split=split, name=name, path=path.relative_to(raw_root).as_posix(),
                                raw_size=after.st_size, raw_sha256=digest))
    plan = dict(schema=1, profile='wipose_explicit_preparation_v2',
        raw_root=str(raw_root), outdir=str(outdir), requested_schema=schema,
        generate_lists=bool(generate_lists), sources=source_bindings,
        lists={split: dict(names=names, provenance=provenance)
               for split, (names, provenance) in selections.items()}, raw_files=records)
    plan['fingerprint'] = object_sha(plan)
    return plan


def _verify_prepared_frame(path, expected):
    try:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(expected):
                raise ValueError('stored fields differ from expected conversion')
            for key, value in expected.items():
                actual = saved[key]
                if actual.dtype != value.dtype or not np.array_equal(actual, value):
                    raise ValueError(f'stored {key} differs from current bound conversion')
    except (OSError, ValueError, EOFError, KeyError) as exc:
        raise ValueError(f'Existing processed file failed exact resume validation: {path}: {exc}') from exc


def prepare(raw_root, outdir, schema='auto', generate_lists=False, resume=False):
    raw_root, outdir = Path(raw_root).resolve(), Path(outdir).resolve()
    if raw_root == outdir or raw_root in outdir.parents or outdir in raw_root.parents:
        raise ValueError('Raw and processed roots must be separate nonnested directories')
    if outdir.exists() and not resume:
        raise FileExistsError(f'Refusing existing preparation target: {outdir}')
    if not outdir.exists() and resume:
        raise FileNotFoundError('--resume requires an existing bound preparation output')
    selections = {split: _raw_names(raw_root, split, generate_lists) for split in SPLITS}
    if set(selections['training'][0]) & set(selections['validation'][0]):
        raise ValueError('TRAIN/TEST raw identity overlap')
    expected_plan = _preparation_plan(raw_root, outdir, schema, generate_lists, selections)
    intent_path = outdir / 'preparation.inprogress.json'
    if resume:
        if not intent_path.is_file():
            raise ValueError('Existing output has no bound preparation.inprogress.json; refusing resume')
        if json.loads(intent_path.read_text(encoding='utf-8')) != expected_plan:
            raise ValueError('Preparation source/raw/list/schema/root binding changed; refusing resume')
    else:
        outdir.mkdir(parents=True, exist_ok=False)
        _publish_json(intent_path, expected_plan)
    raw_bindings = {(record['split'], record['name']): record
                    for record in expected_plan['raw_files']}
    records, chosen_schema = [], None
    for split, (names, provenance) in selections.items():
        list_name, processed_dir = SPLITS[split]
        destination = outdir / processed_dir
        if destination.exists() and not destination.is_dir():
            raise ValueError(f'Processed directory target is not a directory: {destination}')
        destination.mkdir(exist_ok=resume)
        list_path = outdir / list_name
        list_bytes = ('\n'.join(names) + '\n').encode('utf-8')
        if list_path.exists():
            if not resume or list_path.read_bytes() != list_bytes:
                raise ValueError(f'Processed ordered list differs from bound intent: {list_path}')
        else:
            _publish_exclusive(list_path, lambda stream: stream.write(list_bytes))
        raw_dir = raw_root / ('Train' if split == 'training' else 'Test')
        for index, name in enumerate(names):
            source = raw_dir / f'{name}.mat'
            before = source.stat()
            value, detected, transform = convert_record(read_mat(source), schema)
            if chosen_schema is not None and chosen_schema != detected:
                raise ValueError('Mixed raw schemas are not accepted in one preparation')
            chosen_schema = detected
            raw_digest = file_sha(source)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Raw WiPose MAT changed during preparation: {source}')
            if (raw_digest != raw_bindings[(split, name)]['raw_sha256']
                    or after.st_size != raw_bindings[(split, name)]['raw_size']):
                raise RuntimeError(f'Raw WiPose MAT differs from initial binding: {source}')
            target = outdir / processed_dir / f'{name}.npz'
            if target.exists():
                if not resume:
                    raise FileExistsError(f'Refusing existing frame: {target}')
                _verify_prepared_frame(target, value)
            else:
                _publish_exclusive(target, lambda stream: np.savez(stream, **value))
            records.append(dict(split=split, name=name, raw_sha256=raw_digest,
                raw_size=after.st_size, processed_sha256=file_sha(target),
                conversion=transform))
            if index and index % 10000 == 0:
                print(f'[WiPose prepare] {split}: {index}/{len(names)}', flush=True)
    audit = dataset_audit(outdir)
    manifest = dict(schema=1, profile='wipose_explicit_preparation_v2',
        raw_root=str(raw_root), outdir=str(outdir), raw_schema=chosen_schema,
        list_provenance={split: data[1] for split, data in selections.items()},
        transforms=sorted({record['conversion'] for record in records}),
        coordinate_units='original image annotation coordinates; loader divides by1000; not metres',
        source_sha256=file_sha(__file__), sources=expected_plan['sources'],
        intent_fingerprint=expected_plan['fingerprint'], records=records,
        records_sha256=object_sha(records), inputs=audit)
    manifest['fingerprint'] = object_sha(manifest)
    final_path = outdir / 'preparation.json'
    if final_path.exists():
        if not resume or json.loads(final_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Existing final preparation metadata differs from exact replay')
    else:
        _publish_json(final_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--schema', choices=('auto', 'dtpose', 'official'), default='auto')
    parser.add_argument('--generate-lists', action='store_true',
                        help='explicit sorted-list adaptation when original lists are missing')
    parser.add_argument('--resume', action='store_true',
                        help='explicitly verify and resume a source/raw/list-bound partial preparation; never overwrite')
    args = parser.parse_args(argv)
    try:
        result = prepare(args.raw_root, args.outdir, args.schema, args.generate_lists, args.resume)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'[WiPose prepare] ERROR: {exc}\nNo raw files were overwritten. Existing partial outputs are preserved.\n')
    print(f'[WiPose prepare] COMPLETE {result["inputs"]["fingerprint"]}', flush=True)


if __name__ == '__main__':
    main()
