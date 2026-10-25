"""Archive the completed Vast WiFlow/MM-Fi T1 run for a later defense port.

Run this as a separate helper on the original Linux host, with the frozen v6
source and environment still in place. It does not run a defense or make the
Linux-bound attack manifest portable to Windows. Raw MM-Fi frames and the
installed Python environment are deliberately not copied.

The archive contains all eight attack cells, their final T1 table, the frozen
source, continuation tools, original motion prior, and dataset receipt. Every
member is SHA256-bound and reread from the finished ZIP before success prints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile


KEYS = ('clean', 'a1', 'badnets', 'wanet_source', 'ftrojan',
        'fiba', 'sig_source', 'bpp_source')
PROFILE = 'wiflow_mmfi_t1_transfer_archive_v1'
T1_PROFILE = 'wiflow_mmfi_two_gpu_attacks_only_s42_v2'
T1_STATUS = 'COMPLETE_T1_ONLY_DEFENSES_PAUSED'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha_stream(source, target=None):
    digest = hashlib.sha256()
    count = 0
    for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
        digest.update(block)
        count += len(block)
        if target is not None:
            target.write(block)
    return digest.hexdigest(), count


def file_sha(path):
    with Path(path).open('rb') as source:
        return sha_stream(source)[0]


def authenticate_t1(run_root, attack):
    """Re-authenticate every cell with the original frozen attack reader."""
    if Path(run_root).is_symlink():
        raise ValueError('WiFlow run root may not be a symlink')
    root = Path(run_root).resolve()
    final_path = root / 'T1_attack_results.json'
    if not final_path.is_file():
        raise ValueError('Final eight-record T1_attack_results.json is missing')
    final = read_json(final_path)
    if (final.get('profile') != T1_PROFILE or final.get('status') != T1_STATUS
            or final.get('seed') != 42 or final.get('n_seeds') != 1):
        raise ValueError('Final T1 status/profile/seed is not the WiFlow run')

    manifest = attack.load_manifest(root / 'attacks' / attack.MANIFEST)
    if (manifest.get('backbones') != ['wiflow']
            or Path(manifest.get('outdir', '')).resolve() != (root / 'attacks')):
        raise ValueError('Attack manifest is not the original WiFlow run')
    plan = read_json(root / 'attack_only_plan.json')
    if plan.get('profile') != T1_PROFILE or plan.get('attacks') != manifest:
        raise ValueError('WiFlow attack-only plan differs from its manifest')
    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or len(tasks) != len(KEYS):
        raise ValueError('WiFlow plan must contain exactly eight attack tasks')
    names = [task.get('cell') for task in tasks]
    if set(names) != {'wiflow__s42__' + key for key in KEYS} or len(set(names)) != len(KEYS):
        raise ValueError('WiFlow plan has missing or duplicate attack methods')
    records = []
    for task in tasks:
        cell = attack.cell_for(manifest, task['cell'], device=task['preferred_device'])
        record = attack.completed(cell, manifest['sources'])
        if record is None or record.get('status') != 'complete':
            raise ValueError('Attack cell lacks authenticated completion: ' + task['cell'])
        records.append(record)
        print('[archive] authenticated ' + task['cell'], flush=True)
    if final.get('records') != records:
        raise ValueError('Final T1 table differs from authenticated attack cells')
    gate = read_json(root / 'attack_only_prelaunch.json')
    if final.get('source_binding') != gate.get('binding'):
        raise ValueError('Final T1 source binding differs from prelaunch gate')
    return final, manifest


def collect(groups, output):
    output = Path(output).absolute()
    files = {}
    for label, root in groups.items():
        root = Path(root).absolute()
        if root.is_symlink() or not root.exists():
            raise ValueError('Missing or linked archive source: ' + str(root))
        if root.is_dir() and output.resolve().is_relative_to(root.resolve()):
            raise ValueError('Archive output must be outside all inputs')
        candidates = [root] if root.is_file() else sorted(root.rglob('*'))
        for path in candidates:
            if path.is_symlink():
                raise ValueError('Linked archive member: ' + str(path))
            if not path.is_file():
                continue
            relative = Path(path.name) if root.is_file() else path.relative_to(root)
            if ('__pycache__' in relative.parts or '.pytest_cache' in relative.parts
                    or relative.name.endswith('.lock')):
                continue
            name = label + '/' + relative.as_posix()
            if name in files:
                raise ValueError('Duplicate ZIP member: ' + name)
            files[name] = path
    if not files:
        raise ValueError('No archive input files')
    return files


def verify_archive(path):
    """Verify archived bytes without requiring Linux, PyTorch, or source data."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        index = json.loads(archive.read('_ARCHIVE_INDEX.json'))
        files = index.get('files')
        if (index.get('schema') != 1 or index.get('profile') != PROFILE
                or index.get('status') != 'BACKUP_ONLY_WINDOWS_DEFENSE_NOT_YET_VALIDATED'
                or not isinstance(files, dict)):
            raise ValueError('Unexpected WiFlow transfer archive index')
        expected = set(files) | {'_ARCHIVE_INDEX.json'}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError('Archive inventory differs from index')
        for name, bound in files.items():
            with archive.open(name) as source:
                digest, size = sha_stream(source)
            if digest != bound['sha256'] or size != bound['bytes']:
                raise ValueError('Archived member SHA256/size mismatch: ' + name)
    digest = file_sha(path)
    checksum = path.with_suffix(path.suffix + '.sha256')
    if checksum.exists():
        saved = checksum.read_text(encoding='utf-8').split()
        if len(saved) != 2 or saved != [digest, path.name]:
            raise ValueError('Whole ZIP SHA256 sidecar mismatch')
    return dict(status='VERIFIED_WIFLOW_ARCHIVE_BYTES', path=str(path),
                bytes=path.stat().st_size, sha256=digest, files=len(files))


def package(groups, output, metadata, *, progress=True):
    output = Path(output).absolute()
    checksum = output.with_suffix(output.suffix + '.sha256')
    if output.exists() or output.is_symlink() or checksum.exists():
        raise FileExistsError('Output/checksum exists; preserve it and choose a new name')
    files = collect(groups, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    before = {name: (path.stat().st_size, path.stat().st_mtime_ns)
              for name, path in files.items()}
    needed = sum(size for size, _ in before.values()) * 1.02 + 256 * 2**20
    if shutil.disk_usage(output.parent).free < needed:
        raise ValueError('Insufficient disk for checkpoint archive')
    records = {}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for index, (name, path) in enumerate(files.items(), 1):
            with path.open('rb') as source, archive.open(name, 'w', force_zip64=True) as target:
                digest, size = sha_stream(source, target)
            if (size, path.stat().st_mtime_ns) != before[name]:
                raise ValueError('Source changed during archive: ' + str(path))
            records[name] = dict(bytes=size, sha256=digest, original_path=str(path))
            if progress and (index % 25 == 0 or path.suffix.lower() in ('.pt', '.pth')):
                print(f'[archive] {index}/{len(files)} {name} ({size / 2**20:.1f} MiB)', flush=True)
        index = dict(schema=1, profile=PROFILE,
            status='BACKUP_ONLY_WINDOWS_DEFENSE_NOT_YET_VALIDATED',
            metadata=metadata, raw_dataset_included=False,
            installed_environment_included=False, original_files_deleted=False,
            files=records)
        archive.writestr('_ARCHIVE_INDEX.json', json.dumps(index, indent=2) + '\n')
    result = verify_archive(output)
    if collect(groups, output) != files or any(
            (path.stat().st_size, path.stat().st_mtime_ns) != before[name]
            for name, path in files.items()):
        raise ValueError('Inputs changed before archive verification finished')
    checksum.write_text(result['sha256'] + '  ' + output.name + '\n', encoding='utf-8')
    result['source_files_preserved'] = True
    print(json.dumps(result, indent=2), flush=True)
    return result


def _packages():
    names = ('torch', 'torchvision', 'numpy', 'scipy', 'PyYAML')
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path,
        default=Path('/workspace/runs/mmfi_wiflow_s42_attacks_only_v2'))
    parser.add_argument('--repo-dir', type=Path,
        default=Path('/workspace/mmfi_v6/backdoor-analog'))
    parser.add_argument('--tools-dir', type=Path,
        default=Path('/workspace/mmfi_wiflow_tools_v4'))
    parser.add_argument('--data-home', type=Path, default=Path('/workspace/mmfi'))
    parser.add_argument('--metafi-root', type=Path,
        default=Path('/workspace/runs/mmfi_metafiplusplus_s42_attack_defense_v1'))
    parser.add_argument('--backup-dir', type=Path, default=Path('/workspace/backups'))
    parser.add_argument('--output', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--verify', type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        print(json.dumps(verify_archive(args.verify), indent=2), flush=True)
        return 0
    if sys.platform != 'linux':
        raise RuntimeError('Create archive on original Vast Linux host; --verify works on Windows')
    sys.path.insert(0, str(args.repo_dir / 'ATKBackd'))
    import vast_seed42_attack as attack
    final, manifest = authenticate_t1(args.run_root, attack)
    receipt = args.metafi_root / 'dataset_content.json'
    groups = dict(wiflow_run=args.run_root, frozen_v6_code=args.repo_dir,
        continuation_tools=args.tools_dir,
        original_action_prior=args.data_home / 'actions/data_bend.npy',
        dataset_content_receipt=receipt)
    optional = dict(dataset_download_receipt=args.data_home / 'dataset_receipt.json',
        bootstrap_report=Path(sys.executable).parent.parent / 'bootstrap_report.json')
    for label, path in optional.items():
        if path.is_file():
            groups[label] = path
    if args.check_only:
        members = collect(groups, args.backup_dir / 'wiflow_archive_check_only.zip')
        print(f'AUTHENTICATED_WIFLOW_T1_EIGHT_ATTACKS; {len(members)} archive inputs present; no archive written', flush=True)
        return 0
    metadata = dict(original_run_root=str(args.run_root.resolve()),
        original_repo_root=str(args.repo_dir.resolve()),
        original_data_home=str(args.data_home.resolve()),
        original_tools_dir=str(args.tools_dir.resolve()),
        original_dataset_receipt=str(receipt.resolve()),
        t1_status=final['status'], attack_plan_sha256=manifest['plan_sha256'],
        eight_attack_checkpoint_pairs=list(KEYS),
        t1_sha256=file_sha(args.run_root / 'T1_attack_results.json'),
        runtime=dict(python=platform.python_version(), executable=sys.executable,
                     platform=platform.platform(), packages=_packages()),
        portability='Archive preserves Linux-bound evidence; Windows WiFlow defense requires a separately tested runner and capacity smoke.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = args.output or args.backup_dir / ('wiflow_s42_t1_for_van_' + stamp + '.zip')
    package(groups, output, metadata)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
