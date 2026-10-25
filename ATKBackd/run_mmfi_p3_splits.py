"""MM-Fi P3 S1/S2/S3 HPE-Li Clean+A1, seed 42, with explicit run gates.

This is a separate experiment from the frozen P1S1 tables.  It does not edit
the shared feeder, whose bytes are bound to active Person-in-WiFi-3D runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import run_a1_three_tables as frozen
from mmfi_s2_split import MMFIS2
from train_backdoor import _config_fingerprint, _load_cached_result


HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_p3_s1_s2_s3_hpeli_clean_a1_s42_v1'
SETTINGS = ('s3', 's2', 's1')  # harder held-out scene first
EXPECTED_SEQUENCES = {'s1': (864, 216), 's2': (864, 216), 's3': (810, 270)}
# Original MM-Fi A1 motion prior, taken from the completed Vast seed-42 A1
# three_tables_inputs.json (not from synthetic test data).
EXPECTED_ACTION_SHA256 = '2e8fc11192cc3ee7ee3f5544cb3c5e19f771f7f7265dde93af8969af41a2be3e'


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def _sources():
    sources = frozen._source_provenance()
    for name in ('run_mmfi_p3_splits.py', 'mmfi_s2_split.py'):
        sources[name] = _sha_file(HERE / name)
    return sources


def _configuration(data_home, device, setting, key, workers=0):
    if setting not in SETTINGS or key not in ('clean', 'a1'):
        raise ValueError('Only P3 S1/S2/S3 Clean+A1 are in this experiment')
    cfg = frozen._configuration(data_home, device, workers, 42, key)
    cfg.update(mmfi_protocol='protocol3', mmfi_setting=setting,
               mmfi_split_seed=0, mmfi_random_ratio=.8,
               confirmation_profile=PROFILE)
    # Clean and A1 retain the same trigger/target recipe. Clean has rho=0.
    from train_backdoor import _resolve_training_config, _validate_training_contract
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    return cfg


def build_plan(data_home, outdir, device='cuda:0', workers=0):
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    cells = []
    for setting in SETTINGS:
        for key in ('clean', 'a1'):
            cfg = _configuration(data_home, device, setting, key, workers)
            cells.append({'id': f'p3_{setting}_{key}', 'setting': setting,
                          'key': key, 'cfg': cfg,
                          'folder': str(outdir / f'p3_{setting}' / key)})
    plan = {'schema': 1, 'profile': PROFILE, 'seed': 42,
            'data_home': str(data_home), 'outdir': str(outdir),
            'device': device, 'workers': workers, 'cells': cells,
            'sources': _sources(),
            'contract': {'protocol': 'protocol3', 'actions': 27,
                         'settings': list(SETTINGS), 'split_seed': 0,
                         'model': 'hpeli', 'epochs': 50,
                         'selection': 'fixed final epoch49; no TEST tuning',
                         'victim': 'independent scratch; ordinary MPJPE',
                         'rho_a1': .1, 'rho_clean': 0., 'eps': .185,
                         'target_joints': [2, 3], 'dose_grid': list(frozen.GRID),
                         'expected_sequences': {k: list(v)
                                                for k, v in EXPECTED_SEQUENCES.items()},
                         'uncertainty': 'one seed; no seed-level SD/CI'}}
    plan['sha256'] = _sha_json(plan)
    return plan


def _plan_file(outdir):
    return Path(outdir) / 'p3_splits.plan.json'


def _verify_saved_plan(plan):
    path = _plan_file(plan['outdir'])
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved != plan:
            raise ValueError('Existing P3 plan differs; use a NEW output directory')
    else:
        _write_json(path, plan)


def _dataset(cfg, split):
    if cfg['mmfi_setting'] == 's2':
        return MMFIS2(split, cfg['dataset_root'],
                      protocol='protocol3', setting='s2')
    from data_utils.feeder import MMFI
    return MMFI(split, cfg['dataset_root'], protocol='protocol3',
                setting=cfg['mmfi_setting'], random_ratio=.8, random_seed=0)


def _identity_digest(dataset):
    digest = hashlib.sha256()
    sequences = set()
    for item in dataset.items:
        name = item['name']
        sequences.add(tuple(name.split('_')[:3]))
        digest.update(f"{name}\t{item['frame_idx']}\n".encode())
    if len({item['name'] for item in dataset.items}) != len(dataset):
        raise ValueError('Duplicate MM-Fi frame identities')
    return {'frames': len(dataset), 'sequences': len(sequences),
            'ordered_identity_sha256': digest.hexdigest(),
            'sequence_keys': sequences}


def preflight(plan):
    cfg = plan['cells'][0]['cfg']
    if not Path(cfg['action_npy']).is_file():
        raise FileNotFoundError(cfg['action_npy'])
    action_sha = _sha_file(cfg['action_npy'])
    if action_sha != EXPECTED_ACTION_SHA256:
        raise ValueError('Motion prior differs from completed MM-Fi A1 seed42; '
                         f'expected {EXPECTED_ACTION_SHA256}, got {action_sha}')
    import numpy as np
    action = np.load(cfg['action_npy'], allow_pickle=False)
    if not np.isfinite(action).all():
        raise ValueError('Motion prior has NaN/Inf')
    receipt = {'schema': 1, 'plan_sha256': plan['sha256'],
               'action_sha256': action_sha, 'settings': {}}
    for setting in SETTINGS:
        setting_cfg = next(c['cfg'] for c in plan['cells']
                           if c['setting'] == setting)
        train_ds = _dataset(setting_cfg, 'training')
        test_ds = _dataset(setting_cfg, 'test')
        train = _identity_digest(train_ds)
        test = _identity_digest(test_ds)
        if (train['sequences'], test['sequences']) != EXPECTED_SEQUENCES[setting]:
            raise ValueError(f'P3{setting}: incomplete official sequence coverage: '
                             f"{train['sequences']}/{test['sequences']}")
        if train['sequence_keys'] & test['sequence_keys']:
            raise ValueError(f'P3{setting}: TRAIN/TEST sequence overlap')
        if setting in ('s2', 's3'):
            train_subjects = {key[1] for key in train['sequence_keys']}
            test_subjects = {key[1] for key in test['sequence_keys']}
            if train_subjects & test_subjects:
                raise ValueError(f'P3{setting}: subject overlap')
        # CPU operator check on a real TEST frame, without victim inference.
        from attack.poison import PoisonedDataset
        from train_backdoor import build_trigger
        trigger = build_trigger(setting_cfg)
        triggered = PoisonedDataset(
            test_ds, trigger, mode='trigger@dose', fixed_dose=1.,
            rho=setting_cfg['rho'], eps=setting_cfg['eps'],
            pivot=setting_cfg['pivot'],
            theta_max_deg=setting_cfg['theta_max_deg'],
            dose_mode=setting_cfg['dose_mode'],
            axis=setting_cfg['payload_axis'], dataset='mmfi')
        sample = triggered[0]
        if (sample['csi'].shape != (3, 114, 10)
                or sample['pose'].shape != (1, 17, 3)
                or sample['target'].shape != (1, 17, 3)
                or any(not np.isfinite(sample[name]).all()
                       for name in ('csi', 'pose', 'target'))):
            raise ValueError(f'P3{setting}: trigger/payload sample invalid')
        train.pop('sequence_keys')
        test.pop('sequence_keys')
        receipt['settings'][setting] = {'train': train, 'test': test}
        print(f"[preflight] P3{setting}: {train['frames']} train, "
              f"{test['frames']} test frames", flush=True)
    _write_json(Path(plan['outdir']) / 'preflight.json', receipt)
    print('PREFLIGHT_PASS; no GPU/training launched', flush=True)


def _receipt(plan, name):
    path = Path(plan['outdir']) / name
    if not path.is_file():
        raise ValueError(f'Missing gate {path}')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if receipt.get('plan_sha256') != plan['sha256']:
        raise ValueError(f'Stale {name}; use a NEW output directory')
    if name == 'preflight.json':
        action_path = plan['cells'][0]['cfg']['action_npy']
        if receipt.get('action_sha256') != _sha_file(action_path):
            raise ValueError('Motion prior changed after preflight')
    return receipt


def _smoke_worker(device, batch, rendezvous, slot, expected):
    import torch
    from models.factory import build_model
    from train_backdoor import _mpjpe_loss
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False).to(device)
    x = torch.rand(batch, 3, 114, 10, device=device)
    y = torch.rand(batch, 1, 17, 3, device=device)
    ready = Path(rendezvous) / f'{slot}.ready'
    ready.write_text('ready', encoding='ascii')
    limit = time.monotonic() + 90
    while len(list(Path(rendezvous).glob('*.ready'))) < expected:
        if time.monotonic() > limit:
            raise TimeoutError('GPU smoke workers did not become concurrent')
        time.sleep(.1)
    opt = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    opt.zero_grad(set_to_none=True)
    pred, _ = model(x)
    loss = _mpjpe_loss(pred, y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize(device)
    if not torch.isfinite(loss):
        raise ValueError('Non-finite GPU smoke loss')
    print(f'GPU_SMOKE_WORKER_PASS slot={slot} loss={float(loss):.5f}', flush=True)


def gpu_smoke(plan, jobs):
    _receipt(plan, 'preflight.json')
    import tempfile
    with tempfile.TemporaryDirectory(dir=plan['outdir']) as barrier:
        children = [subprocess.Popen([
            sys.executable, '-u', str(HERE / 'run_mmfi_p3_splits.py'),
            '--_smoke-worker', '--device', plan['device'], '--batch', '32',
            '--_barrier', barrier, '--_slot', str(i), '--jobs-per-gpu', str(jobs)
        ]) for i in range(jobs)]
        codes = [child.wait() for child in children]
    if any(codes):
        raise RuntimeError(f'GPU smoke failed {codes}; no full training launched')
    _write_json(Path(plan['outdir']) / 'gpu_smoke.json',
                {'plan_sha256': plan['sha256'], 'jobs_per_gpu': jobs,
                 'physical_batch': 32, 'concurrent_workers': jobs})
    print('GPU_SMOKE_PASS; no full training launched', flush=True)


def _complete(cell):
    folder = Path(cell['folder'])
    ckpt = folder / 'checkpoint.pt'
    if not ckpt.is_file():
        return False
    import torch
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    if saved.get('epoch') != 49 or saved.get('cfg_fingerprint') != _config_fingerprint(cell['cfg']):
        return False
    if _load_cached_result(folder, cell['cfg']) is None:
        return False
    report_path = folder / 'p3_evaluation.json'
    samples = folder / 'p3_per_sample.npz'
    if not report_path.is_file() or not samples.is_file():
        return False
    report = json.loads(report_path.read_text(encoding='utf-8'))
    return (report.get('status') == 'complete'
            and report.get('config_fingerprint') == _config_fingerprint(cell['cfg'])
            and report.get('checkpoint_sha256') == _sha_file(ckpt)
            and report.get('per_sample_sha256') == _sha_file(samples))


def _evaluate_cell(plan, cell, cached):
    """Replay final weights and add target PCK on the exact P3 held-out split."""
    import numpy as np
    from eval_a1_target_pck import (_assert_replay, _load_weights, _measure,
                                    _predict_pair, _restore_trigger)

    cfg = cell['cfg']
    if cached.get('dose_grid') != list(frozen.GRID):
        raise ValueError('P3 cache lost the six-dose evaluation grid')
    if cell['key'] == 'a1':
        dce = np.asarray(cached.get('dose_control_error'), dtype=float)
        if (dce.shape != (6,) or not np.isfinite(dce).all()
                or abs(dce[0]) > 1e-8 or (dce < 0).any()):
            raise ValueError('P3 A1 cache has invalid six-dose DCE')
    folder = Path(cell['folder'])
    fingerprint = _config_fingerprint(cfg)
    model, state = _load_weights(
        {'folder': folder, 'cfg': cfg, 'fingerprint': fingerprint}, cfg['device'])
    data = _dataset(cfg, 'test')
    trigger = _restore_trigger(cfg, state)
    try:
        predictions = _predict_pair(model, data, trigger, cfg, cfg['device'])
        measured, samples = _measure(*predictions)
        replay = _assert_replay(cached, measured, .05, attack=cell['key'] == 'a1')
    finally:
        del model
    pair_ids = np.asarray([item['name'] for item in data.items], dtype=str)
    if len(pair_ids) != len(predictions[0]):
        raise ValueError('P3 evaluation identity/prediction length differs')
    binary = io.BytesIO()
    np.savez_compressed(binary, pair_ids=pair_ids, **samples)
    sample_path = folder / 'p3_per_sample.npz'
    tmp = folder / 'p3_per_sample.npz.tmp'
    tmp.write_bytes(binary.getvalue())
    os.replace(tmp, sample_path)
    metrics = {
        'clean_mpjpe_mm': measured['clean_mpjpe_mm'],
        'clean_pampjpe_mm': measured['clean_pampjpe_mm'],
        't_mpjpe_mm': measured['t1_mpjpe_mm'] if cell['key'] == 'a1' else None,
        'target_pck_pct': measured['target_pck_0.2_pct'] if cell['key'] == 'a1' else None,
        'no_trigger_t_mpjpe_mm': (measured['no_trigger_t1_mpjpe_mm']
                                 if cell['key'] == 'a1' else None),
        'no_trigger_target_pck_pct': (
            measured['no_trigger_target_pck_0.2_pct']
            if cell['key'] == 'a1' else None),
        'mean_positive_dce_mm': (sum(cached['dose_control_error'][1:]) * 1000. / 5.
                                 if cell['key'] == 'a1' else None),
    }
    report = {'status': 'complete', 'profile': PROFILE,
              'label': 'EXPLORATORY_T6_SPLIT_SLICE_NOT_FULL_T6',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'cell': cell['id'], 'seed': 42, 'setting': cell['setting'],
              'key': cell['key'], 'checkpoint_epoch': 49,
              'plan_sha256': plan['sha256'],
              'config_fingerprint': fingerprint,
              'checkpoint_sha256': _sha_file(folder / 'checkpoint.pt'),
              'per_sample_sha256': _sha_file(sample_path),
              'frames': len(pair_ids), 'metrics': metrics,
              'replay_abs_delta_mm': replay,
              'metric_contract': {'target_joints': [2, 3], 'dose': 1.,
                                  'target_pck_threshold': .2,
                                  'clean_scale_joints': [5, 12],
                                  'units': 'millimetres',
                                  'target_pck_is_pose_level_asr': False}}
    _write_json(folder / 'p3_evaluation.json', report)
    print(f"EVALUATION_COMPLETE {cell['id']}: {metrics}", flush=True)


def _train_cell(plan, cell_id):
    cell = next((c for c in plan['cells'] if c['id'] == cell_id), None)
    if cell is None:
        raise ValueError(f'Unknown cell {cell_id!r}')
    from run_mmfi_tables import _cell_lock
    with _cell_lock(cell['folder']):
        cfg = cell['cfg']
        if cfg['mmfi_setting'] == 's2':
            # Process-local adapter only. No edits to the frozen feeder or engine.
            import train_backdoor
            train_backdoor.MMFI = MMFIS2
        from train_backdoor import train
        trained_model, cached = train(cfg, ckpt_dir=cell['folder'])
        del trained_model
        if not _complete(cell):
            _evaluate_cell(plan, cell, cached)
        if not _complete(cell):
            raise RuntimeError(f'{cell_id}: final epoch49/evaluation missing')
    print(f'COMPLETE {cell_id}', flush=True)


def launch(plan, jobs):
    _receipt(plan, 'preflight.json')
    smoke = _receipt(plan, 'gpu_smoke.json')
    if smoke.get('jobs_per_gpu') != jobs:
        raise ValueError('GPU smoke concurrency differs from requested launch')
    from run_mmfi_tables import _cell_lock
    with _cell_lock(Path(plan['outdir']) / '.controller'):
        _launch_locked(plan, jobs)


def _launch_locked(plan, jobs):
    for setting in SETTINGS:
        cells = [c for c in plan['cells'] if c['setting'] == setting]
        running = []
        for cell in cells:
            if _complete(cell):
                print(f"VERIFIED_COMPLETE {cell['id']}", flush=True)
                continue
            folder = Path(cell['folder'])
            folder.mkdir(parents=True, exist_ok=True)
            stream = (folder / 'console.log').open('a', encoding='utf-8')
            child = subprocess.Popen([
                sys.executable, '-u', str(HERE / 'run_mmfi_p3_splits.py'),
                '--_cell', cell['id'], '--data-home', plan['data_home'],
                '--outdir', plan['outdir'], '--device', plan['device'],
                '--workers', str(plan['workers'])],
                stdout=stream, stderr=subprocess.STDOUT, cwd=HERE)
            running.append((child, stream, cell))
            print(f"START {cell['id']} PID={child.pid} log={folder / 'console.log'}",
                  flush=True)
            if len(running) == jobs:
                _wait_cells(running)
                running.clear()
        if running:
            _wait_cells(running)
        print(f'P3{setting} COMPLETE', flush=True)


def _wait_cells(running):
    codes = [child.wait() for child, _, _ in running]
    for _, stream, _ in running:
        stream.close()
    for (_, _, cell), code in zip(running, codes):
        if code or not _complete(cell):
            raise RuntimeError(f"{cell['id']} failed (exit {code}); inspect console.log")


def status(plan):
    for cell in plan['cells']:
        log = Path(cell['folder']) / 'console.log'
        epoch = '-'
        if log.exists():
            epochs = re.findall(r'epoch (\d+)/49:', log.read_text(encoding='utf-8',
                                                                   errors='replace'))
            epoch = epochs[-1] if epochs else '-'
        complete = _complete(cell)
        print(f"{cell['id']}: epoch={epoch}/49 complete={complete} log={log}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('dry-run', 'preflight', 'gpu-smoke',
                                           'launch', 'status'), default='dry-run')
    parser.add_argument('--data-home', default='/workspace/mmfi')
    parser.add_argument('--outdir', default='/workspace/runs/mmfi_p3_hpeli_s42_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--jobs-per-gpu', type=int, choices=(1, 2), default=1)
    parser.add_argument('--_cell')
    parser.add_argument('--_smoke-worker', action='store_true')
    parser.add_argument('--_barrier')
    parser.add_argument('--_slot', type=int)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args(argv)
    if args._smoke_worker:
        _smoke_worker(args.device, args.batch, args._barrier,
                      args._slot, args.jobs_per_gpu)
        return 0
    if args.workers < 0:
        parser.error('--workers must be nonnegative')
    plan = build_plan(args.data_home, args.outdir, args.device, args.workers)
    if args.mode == 'dry-run' and not args._cell:
        print(f"P3 HPE-Li Clean+A1 seed42 plan={plan['sha256']}")
        for cell in plan['cells']:
            cfg = cell['cfg']
            print(f"{cell['id']}: {cfg['epochs']} epochs, batch{cfg['batch_size']}, "
                  f"SGD lr={cfg['lr']}, rho={cfg['rho']}, {cfg['trigger']}")
        print('DRY_RUN_ONLY; no data/CUDA/training/output writes', flush=True)
        return 0
    if args.mode == 'status' and not args._cell:
        path = _plan_file(plan['outdir'])
        if not path.exists():
            print('NOT_PREPARED; no output writes')
            return 0
        if json.loads(path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Saved P3 plan differs from current source/settings')
        status(plan)
        return 0
    _verify_saved_plan(plan)
    if args._cell:
        _receipt(plan, 'preflight.json')
        _receipt(plan, 'gpu_smoke.json')
        _train_cell(plan, args._cell)
    elif args.mode == 'preflight':
        preflight(plan)
    elif args.mode == 'gpu-smoke':
        gpu_smoke(plan, args.jobs_per_gpu)
    elif args.mode == 'launch':
        launch(plan, args.jobs_per_gpu)
    else:
        status(plan)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
