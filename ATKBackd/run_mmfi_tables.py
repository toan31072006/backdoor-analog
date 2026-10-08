"""Seed-42 MM-Fi matrix: ten unique result cells, four publication tables.

Fresh training is opt-in through --fresh and NEVER deletes previous results.
RF attacks use separate staged trainers; they are not renamed image triggers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# NumPy MKL and PyTorch can ship incompatible OpenMP runtimes on Windows.
# Serial MKL is safe for the small CPU metric SVDs; never allow duplicate OMP.
if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import yaml

HERE = Path(__file__).resolve().parent
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
RF_METHODS = {'infocom2025_por', 'ccai2026_backdoorrf'}
METHODS = (
    ('clean', 'Clean', 'internal', [3]),
    ('prop_rho0p4', 'Proposed', 'proposed', [1, 2, 3, 4]),
    ('shuffled', 'Proposed - shuffled coupling', 'internal', [3]),
    ('prop_rho0p1', 'Proposed (rho=0.1)', 'proposed', [4]),
    ('prop_rho0p2', 'Proposed (rho=0.2)', 'proposed', [4]),
    ('badnets', 'BadNets (CSI adaptation)', 'traditional', [1]),
    ('blended', 'Blended (CSI adaptation)', 'traditional', [1]),
    ('wanet', 'WaNet (CSI adaptation)', 'traditional', [1]),
    ('infocom2025_por', 'RF PTM / POR (HPE adaptation)', 'rf', [1]),
    ('ccai2026_backdoorrf', 'BackdoorRF (HPE adaptation)', 'rf', [1]),
)
SOURCES = {
    'badnets': {'paper': 'https://arxiv.org/abs/1708.06733',
                'implementation': 'Independent CSI adaptation of patch poisoning'},
    'blended': {'paper': 'https://arxiv.org/abs/1712.05526',
                'implementation': 'Independent CSI adaptation of the blend equation'},
    'wanet': {'paper': 'https://arxiv.org/abs/2102.10369',
              'code': 'https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release',
              'implementation': 'Independent frequency/time warp with clean-label noise covers'},
    'infocom2025_por': {'paper': 'https://arxiv.org/abs/2505.00881',
        'code': 'https://github.com/Tianyaz97/rf_backdoor',
        'commit': '989f148ed35c96f46bf96db797ff23376895fb25', 'license': 'MIT',
        'limitations': ['Pretrained-encoder tampering, not data poisoning',
            'Default substitute source uses downstream TRAIN CSI without labels: NOT data-free',
            'Original POR objective has no chosen input-dependent pose displacement',
            'Common T-MPJPE is a transferred diagnostic, not original-paper success']},
    'ccai2026_backdoorrf': {'code': 'https://github.com/NatsumiAi/BackdoorRF',
        'commit': '4b7d44fc4c939b6b2018e522f0f0c9801fa90abb',
        'license': 'No license file found; independent implementation, not copied code',
        'limitations': ['Conference association per repository/user confirmation; no invented DOI',
            'Targeted class loss replaced by pose regression; short-CSI PSD adaptation']},
}


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4,
                 rf_substitute_path=None):
    """Resolve all scientific settings without opening data or probing CUDA."""
    from train_backdoor import (_resolve_training_config, _validate_training_contract,
                                _CHECKPOINT_SCHEMA, _RESULT_SCHEMA)
    from train_rf_backdoor import resolve_rf_config
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    base = yaml.safe_load((HERE / 'configs/mmfi/attack_bend.yaml').read_text(encoding='utf-8'))
    base.update(dataset_root=str(data_home / 'datasets/Compress'),
        action_npy=str(data_home / 'actions/data_bend.npy'), seed=42, model='hpeli',
        device=device, num_workers=num_workers, epochs=50, victim_epochs=50,
        ckpt_every=1, strict_resume=True, dose_grid=GRID, payload_axis=[0.0, 0.0, 1.0],
        training_protocol='ordinary_erm', threat_model='training_data_poisoning',
        attacker_access='data_only', dose_coupling='paired')
    cells = []
    for key, label, group, tables in METHODS:
        cfg = copy.deepcopy(base)
        cfg['trigger'] = 'micro_dropper'
        dependencies = []
        if key == 'clean':
            cfg['rho'] = 0.0
        elif key == 'shuffled':
            cfg['dose_coupling'] = 'shuffled'
        elif key.startswith('prop_rho'):
            cfg['rho'] = {'prop_rho0p1': 0.1, 'prop_rho0p2': 0.2, 'prop_rho0p4': 0.4}[key]
        else:
            cfg.update(trigger=key, dose_min=1.0, dose_max=1.0)
            if key == 'wanet':
                cfg.update(wanet_cover_ratio=0.2, wanet_noise_strength=1.0,
                           wanet_grid_size=4, wanet_strength=0.5, wanet_grid_rescale=1.0)
            if key in RF_METHODS:
                cfg['training_protocol'] = key
                cfg.pop('attacker_access')
                cfg.pop('threat_model')
                if key == 'infocom2025_por':
                    dependencies = ['clean']
                    cfg['rf_clean_teacher_checkpoint'] = str(outdir / 'clean/checkpoint.pt')
                    cfg['rf_substitute_source'] = 'downstream_train_unlabeled'
                    if rf_substitute_path:
                        cfg.update(rf_substitute_source='external_unlabeled',
                                   rf_substitute_path=str(Path(rf_substitute_path).resolve()))
                cfg = resolve_rf_config(cfg)
        cfg = _resolve_training_config(cfg)
        _validate_training_contract(cfg)
        cell_dir = outdir / key
        cells.append(dict(method_key=key, label=label, group=group, tables=tables,
                          cfg=cfg, dependencies=dependencies,
                          ckpt_dir=str(cell_dir), eval_cache=str(cell_dir / 'eval_cache.json')))
    matrix = dict(schema=1, dataset='mmfi', seed=42, fresh_results_only=True,
                created_utc=datetime.now(timezone.utc).isoformat(),
                sources=SOURCES, cells=cells,
                metrics_contract={'errors': 'millimetres; pose arrays stored in metres',
                    'checkpoint_schema': _CHECKPOINT_SCHEMA, 'result_schema': _RESULT_SCHEMA,
                    'pck': 'relative threshold 0.5/0.4/0.3/0.2/0.1, NOT millimetres',
                    'target_joints': [2, 3], 'dose_grid': GRID,
                    'main_asr': False, 'seeds': [42], 'uncertainty': 'single seed: not estimated'})
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    return matrix


def _plan_fingerprint(matrix):
    """Recompute plan integrity, rather than trusting an editable stored hash."""
    ignored = {'device', 'num_workers', 'ckpt_every',
               'rf_clean_teacher_sha256', 'rf_substitute_sha256'}
    projection = []
    for cell in matrix['cells']:
        row = {k: cell[k] for k in ('method_key', 'label', 'group', 'tables',
                                    'dependencies', 'ckpt_dir', 'eval_cache')}
        row['cfg'] = {k: v for k, v in cell['cfg'].items() if k not in ignored}
        projection.append(row)
    payload = dict(schema=matrix['schema'], dataset=matrix['dataset'],
                   seed=matrix['seed'], fresh_results_only=matrix['fresh_results_only'],
                   cells=projection, sources=matrix['sources'],
                   metrics_contract=matrix['metrics_contract'])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()


@contextmanager
def _cell_lock(folder):
    """OS lock released automatically on a crash; never break another run's lock."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / 'run.lock').open('a+b')
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Another process is running this cell: {folder}') from exc
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _run_cell(cell):
    from train_backdoor import train, _config_fingerprint
    cfg = cell['cfg']
    with _cell_lock(cell['ckpt_dir']):
        if cell['method_key'] in RF_METHODS:
            from train_rf_backdoor import train as rf_train
            model, res = rf_train(cfg, ckpt_dir=cell['ckpt_dir'])
        else:
            model, res = train(cfg, ckpt_dir=cell['ckpt_dir'])
        del model
        _atomic_json(Path(cell['ckpt_dir']) / 'results.json', dict(
            method_key=cell['method_key'], cfg=cfg, res=res,
            cfg_fingerprint=_config_fingerprint(cfg)))
        print(f"[mmfi] completed {cell['method_key']}", flush=True)


def _complete(cell):
    from train_backdoor import _load_cached_result
    return _load_cached_result(cell['ckpt_dir'], cell['cfg']) is not None


def _check_inputs(matrix, devices):
    import torch
    cfg = matrix['cells'][0]['cfg']
    root = Path(cfg['dataset_root'])
    if not root.is_dir() or not next(root.glob('E*/S*/A*/ground_truth.npy'), None):
        raise FileNotFoundError(f'MM-Fi root has no expected ground_truth.npy: {root}')
    if not Path(cfg['action_npy']).is_file():
        raise FileNotFoundError(f"Missing action trigger: {cfg['action_npy']}")
    for device in devices:
        target = torch.device(device)
        if target.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; --dry-run does not test CUDA')
            torch.cuda.get_device_properties(target)
            torch.ones(1, device=target).sum().item()


def _select_cells(matrix, requested):
    all_cells = {c['method_key']: c for c in matrix['cells']}
    selected = set(requested or all_cells)
    unknown = selected - all_cells.keys()
    if unknown:
        raise ValueError(f'Unknown cells: {sorted(unknown)}')
    for key in tuple(selected):
        selected.update(all_cells[key]['dependencies'])
    return [c for c in matrix['cells'] if c['method_key'] in selected]


def run_matrix(matrix, manifest_path, devices, requested=None):
    """One independent process per device, with explicit clean-teacher dependency."""
    from train_rf_backdoor import finalize_rf_config
    pending = _select_cells(matrix, requested)
    completed = set()
    running = {}
    try:
        while pending or running:
            for device in devices:
                if device in running:
                    continue
                eligible = next((c for c in pending if set(c['dependencies']) <= completed), None)
                if eligible is None:
                    continue
                pending.remove(eligible)
                eligible['cfg']['device'] = device
                if eligible['method_key'] in RF_METHODS:
                    eligible['cfg'] = finalize_rf_config(eligible['cfg'])
                _atomic_json(manifest_path, matrix)
                logfile = Path(eligible['ckpt_dir']) / 'console.log'
                logfile.parent.mkdir(parents=True, exist_ok=True)
                stream = logfile.open('a', encoding='utf-8')
                command = [sys.executable, '-u', str(Path(__file__).resolve()),
                           '--_cell', eligible['method_key'], '--_manifest', str(manifest_path)]
                env = os.environ.copy()
                env.setdefault('OMP_NUM_THREADS', '4')
                env.setdefault('MKL_NUM_THREADS', '4')
                child = subprocess.Popen(command, cwd=HERE, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT)
                running[device] = (child, eligible, stream)
                print(f"[mmfi] {eligible['method_key']} -> {device}, PID={child.pid}\n"
                      f'       log: {logfile}', flush=True)
            for device, (child, cell, stream) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                del running[device]
                if code != 0:
                    raise RuntimeError(f"Cell {cell['method_key']} failed (exit {code}); "
                                       f"read {cell['ckpt_dir']}/console.log")
                if not _complete(cell):
                    raise RuntimeError(f"Cell {cell['method_key']} exited without a valid result")
                completed.add(cell['method_key'])
                print(f"[mmfi] finished {cell['method_key']}", flush=True)
            if pending and not running and not any(set(c['dependencies']) <= completed for c in pending):
                raise RuntimeError('Unresolved dependency cycle')
            if running:
                time.sleep(1)
    except BaseException:
        # Only children launched by THIS scheduler; never kill other GPU jobs.
        for child, _, stream in running.values():
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            stream.close()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cells', nargs='+', choices=[r[0] for r in METHODS])
    parser.add_argument('--rf-substitute-path', type=Path)
    parser.add_argument('--fresh', action='store_true', help='Require a NEW empty output directory')
    parser.add_argument('--dry-run', action='store_true', help='Write only the resolved matrix; no data/GPU/training')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--allow-partial', action='store_true', help='Export UNOFFICIAL partial tables only')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--_cell', help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args._cell:
        matrix = json.loads(args._manifest.read_text(encoding='utf-8'))
        cell = next(c for c in matrix['cells'] if c['method_key'] == args._cell)
        _run_cell(cell)
        return 0
    if args.data_home is None or args.outdir is None:
        raise ValueError('--data-home and --outdir are required')
    if args.num_workers < 0:
        raise ValueError('--num-workers cannot be negative')
    devices = args.devices
    if (len(set(devices)) != len(devices) or
            any(d != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None for d in devices)):
        raise ValueError('--devices must list distinct logical cuda:N devices, or cpu')
    outdir = args.outdir.resolve()
    if args.fresh and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('--fresh refuses a non-empty output directory; use a NEW path')
    # A second scheduler must not race manifest finalization or GPU allocation.
    # Child cells take their own locks in distinct directories.
    with _cell_lock(outdir):
        return _execute_matrix(args, outdir, devices)


def _execute_matrix(args, outdir, devices):
    manifest_path = outdir / 'mmfi_matrix.resolved.json'
    requested_matrix = build_matrix(args.data_home, outdir, devices[0], args.num_workers,
                                    args.rf_substitute_path)
    if manifest_path.exists():
        matrix = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (matrix.get('plan_sha256') != requested_matrix['plan_sha256'] or
                _plan_fingerprint(matrix) != requested_matrix['plan_sha256']):
            raise ValueError('Existing matrix differs; use a new output directory')
        for c in matrix['cells']:
            c['cfg']['num_workers'] = args.num_workers
    else:
        matrix = requested_matrix
    _atomic_json(manifest_path, matrix)
    if args.dry_run:
        print(f'[mmfi] DRY RUN: seed42, {len(matrix["cells"])} unique cells; NO training.\n{manifest_path}')
        return 0
    if not args.export_only:
        _check_inputs(matrix, devices)
        run_matrix(matrix, manifest_path, devices, args.cells)
    from mmfi_tables import export_tables
    missing = [c['method_key'] for c in matrix['cells'] if not _complete(c)]
    if missing and args.cells and not args.export_only and not args.allow_partial:
        print(f'[mmfi] Requested subset completed. Official tables pending: {missing}')
        return 0
    export_tables(manifest_path, outdir / 'tables', allow_partial=args.allow_partial,
                  plots=not args.no_plots)
    print(f'[mmfi] Tables: {outdir / "tables"}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f'[mmfi] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
