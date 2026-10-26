"""MM-Fi P3 S1/S2/S3 Clean+A1 on MetaFi++ and WiFlow, one RTX 3090.

This is a new, one-seed split study. Each backbone runs its six cells in a
separate process; the controller never runs more than one worker per backbone
or two workers total. Historical HPE-Li and P1S1 outputs are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import run_a1_three_tables as frozen
import run_mmfi_p3_splits as p3


HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_p3_metafi_wiflow_clean_a1_s42_3090_v1'
BACKBONES = ('metafiplusplus', 'wiflow')
KEYS = ('clean', 'a1')
SETTINGS = p3.SETTINGS
EXPECTED_ACTION_SHA256 = p3.EXPECTED_ACTION_SHA256
EXPECTED_FRAMES = {'s1': (256608, 64152), 's2': (256608, 64152),
                   's3': (240570, 80190)}
RESULTS = 'p3_two_backbones_results.json'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_name(path.name + '.saving').exists():
        raise FileExistsError(f'Immutable output already exists: {path}')
    temporary = path.with_name(path.name + '.saving')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                             encoding='utf-8')
        if path.exists():
            raise FileExistsError(f'Immutable output already exists: {path}')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sources():
    names = set(frozen._source_provenance())
    names.update((
        'run_mmfi_p3_two_backbones_3090.py', 'run_mmfi_p3_splits.py',
        'mmfi_s2_split.py', 'a1_backbone_adapter.py',
        'a1_wiflow_adapter.py', 'vast_seed42_adapter.py',
        'models/cross_backbone_factory.py', 'models/metafi_plus_plus.py',
        'models/metafi_channel_transformer.py', 'models/metafi_sources.json',
        'models/wiflow_mmfi.py', 'models/wiflow_sources.json',
        'models/_vendor/wiflow_mmfi.py',
    ))
    return {name: _sha(HERE / name) for name in sorted(names)}


def _runtime():
    return dict(python=platform.python_version(), system=platform.system(),
                packages={name: version(name) for name in
                          ('torch', 'numpy', 'scipy', 'PyYAML')})


def _configuration(home, device, backbone, setting, key):
    if backbone not in BACKBONES or setting not in SETTINGS or key not in KEYS:
        raise ValueError('Only MetaFi++/WiFlow, P3 S1-S3, Clean/A1 are supported')
    from train_backdoor import _resolve_training_config, _validate_training_contract
    cfg = frozen._configuration(Path(home), device, 0, 42, key)
    if backbone == 'metafiplusplus':
        from a1_backbone_adapter import recipe_config
        cfg = recipe_config(cfg, backbone)
    else:
        from a1_wiflow_adapter import recipe_config
        cfg = recipe_config(cfg)
    cfg.update(mmfi_protocol='protocol3', mmfi_setting=setting,
               mmfi_random_ratio=.8, mmfi_split_seed=0,
               confirmation_profile=PROFILE, p3_two_backbones_profile=PROFILE)
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    _validate_p3_config(cfg)
    return cfg


def _validate_p3_config(cfg):
    backbone = cfg.get('model')
    if (backbone not in BACKBONES or cfg.get('p3_two_backbones_profile') != PROFILE
            or cfg.get('confirmation_profile') != PROFILE
            or cfg.get('mmfi_protocol') != 'protocol3'
            or cfg.get('mmfi_setting') not in SETTINGS
            or cfg.get('seed') != 42 or cfg.get('rho') not in (0., .1)
            or cfg.get('target_joints', [2, 3]) != [2, 3]
            or cfg.get('pivot') != 1 or cfg.get('theta_max_deg') != 40
            or cfg.get('payload_axis') != [0., 0., 1.]
            or cfg.get('dose_grid') != list(frozen.GRID)
            or cfg.get('dose_coupling') != 'paired'
            or cfg.get('victim_loss') != 'mpjpe'):
        raise ValueError('P3 model/split/target contract changed')
    # Reuse every frozen P1S1 model/optimizer check with only the declared
    # protocol and setting normalized for validation. Training sees P3 intact.
    reference = dict(cfg, mmfi_protocol='protocol1', mmfi_setting='s1')
    if backbone == 'metafiplusplus':
        from a1_backbone_adapter import validate_config
    else:
        from a1_wiflow_adapter import validate_config
    validate_config(reference)


def _training_engine(cfg):
    """Isolate the P3 split hook without mutating historical imported modules."""
    from a1_backbone_adapter import _private_module
    from mmfi_s2_split import MMFIS2
    _validate_p3_config(cfg)
    if cfg['model'] == 'metafiplusplus':
        adapter = _private_module('a1_backbone_adapter.py', '_p3_metafi_adapter')
    else:
        adapter = _private_module('a1_wiflow_adapter.py', '_p3_wiflow_adapter')
    adapter.validate_config = _validate_p3_config
    engine = adapter.training_engine(cfg)
    if cfg['mmfi_setting'] == 's2':
        engine.MMFI = MMFIS2
    return engine


def _gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    properties = torch.cuda.get_device_properties(0)
    if '3090' not in properties.name or properties.total_memory < 22 * 2**30:
        raise RuntimeError(f'Expected the Windows RTX 3090, found {properties.name}')
    return dict(name=properties.name, total_memory=properties.total_memory,
                capability=[properties.major, properties.minor],
                uuid=str(getattr(properties, 'uuid', '')))


def build_plan(home, output, device='cuda:0', jobs=2):
    home, output = Path(home).resolve(), Path(output).resolve()
    if device != 'cuda:0' or jobs != 2:
        raise ValueError('This profile requires two workers on one cuda:0 RTX 3090')
    runs = (home / 'runs').resolve()
    if not output.is_relative_to(runs) or output == runs:
        raise ValueError('Choose a dedicated output under data-home/runs')
    protected = (home / 'datasets', home / 'actions',
                 home / 'runs/mmfi_p3_hpeli_s42_v1')
    if any(output == p.resolve() or output in p.resolve().parents
           or p.resolve() in output.parents for p in protected):
        raise ValueError('Output overlaps historical runs or inputs')
    cells = []
    for backbone in BACKBONES:
        for setting in SETTINGS:
            for key in KEYS:
                folder = output / backbone / f'p3_{setting}' / key
                cells.append(dict(id=f'{backbone}__p3_{setting}__{key}',
                    backbone=backbone, setting=setting, key=key,
                    cfg=_configuration(home, device, backbone, setting, key),
                    folder=str(folder)))
    value = dict(schema=1, profile=PROFILE, data_home=str(home),
        outdir=str(output), device=device, jobs_per_gpu=2, seed=42,
        cells=cells, sources=_sources(),
        runtime=_runtime(),
        contract=dict(status='EXPLORATORY_P3_TWO_BACKBONES_ONE_SEED',
            protocol='protocol3', settings=list(SETTINGS),
            victims=12, independent_scratch=True, epochs=50, final_epoch=49,
            physical_batch=32, poison_rate_a1=.1, target_joints=[2, 3],
            dose_grid=list(frozen.GRID), selection='fixed final49; no TEST tuning',
            model_schedule='one MetaFi++ and one WiFlow process at once',
            other_backbone='historical HPE-Li P3 run remains separate',
            limitations=['One seed; no SD/CI.',
                'WiFlow uses common MPJPE, fixed LR and effective batch64.',
                'The two models use different optimizer recipes.',
                'Attack operators are digital CSI adaptations.']))
    value['plan_sha256'] = _digest(value)
    return value


def _plan_path(plan):
    return Path(plan['outdir']) / 'p3_two_backbones.plan.json'


def _read_plan(plan):
    saved = json.loads(_plan_path(plan).read_text(encoding='utf-8'))
    if saved != plan:
        raise ValueError('Saved P3 plan/source/runtime differs; use a NEW output')
    return saved


def _cell(plan, backbone, setting, key):
    rows = [row for row in plan['cells'] if row['backbone'] == backbone
            and row['setting'] == setting and row['key'] == key]
    if len(rows) != 1:
        raise ValueError('Missing or duplicate P3 cell')
    return rows[0]


def _preflight_data(plan):
    """Use the same official split adapter and counts as the finished HPE-Li P3."""
    import numpy as np
    cfg = plan['cells'][0]['cfg']
    action = Path(cfg['action_npy'])
    if _sha(action) != EXPECTED_ACTION_SHA256:
        raise ValueError('data_bend.npy differs from the P3 HPE-Li input')
    if not np.isfinite(np.load(action, allow_pickle=False)).all():
        raise ValueError('Motion prior has NaN/Inf')
    settings = {}
    for setting in SETTINGS:
        selected = _cell(plan, BACKBONES[0], setting, 'a1')['cfg']
        train = p3._identity_digest(p3._dataset(selected, 'training'))
        test = p3._identity_digest(p3._dataset(selected, 'test'))
        if ((train['sequences'], test['sequences']) != p3.EXPECTED_SEQUENCES[setting]
                or (train['frames'], test['frames']) != EXPECTED_FRAMES[setting]
                or train['sequence_keys'] & test['sequence_keys']):
            raise ValueError(f'P3{setting}: incomplete or overlapping split')
        if setting in ('s2', 's3') and ({k[1] for k in train['sequence_keys']}
                                       & {k[1] for k in test['sequence_keys']}):
            raise ValueError(f'P3{setting}: TRAIN/TEST subject overlap')
        train.pop('sequence_keys')
        test.pop('sequence_keys')
        settings[setting] = dict(train=train, test=test)
        print(f'[P3 preflight] {setting}: {train["frames"]} train / '
              f'{test["frames"]} test frames', flush=True)
    old = Path(plan['data_home']) / 'runs/mmfi_p3_hpeli_s42_v1/preflight.json'
    if old.is_file():
        reference = json.loads(old.read_text(encoding='utf-8'))
        if reference.get('settings') != settings:
            raise ValueError('P3 split differs from completed HPE-Li study')
    return dict(plan_sha256=plan['plan_sha256'], action_sha256=EXPECTED_ACTION_SHA256,
                settings=settings, hpeli_split_receipt_sha256=_sha(old) if old.is_file() else None)


def preflight(plan):
    from run_mmfi_tables import _cell_lock
    root = Path(plan['outdir'])
    with _cell_lock(root):
        path = _plan_path(plan)
        if path.exists():
            _read_plan(plan)
        elif any(p.name != 'run.lock' for p in root.iterdir()):
            raise ValueError('Nonempty output without P3 plan; choose a NEW folder')
        data = _preflight_data(plan)
        receipt = root / 'preflight.json'
        if path.exists():
            if not receipt.is_file() or json.loads(receipt.read_text(encoding='utf-8')) != data:
                raise ValueError('Existing P3 input receipt differs')
        else:
            _write_new(path, plan)
            _write_new(receipt, data)
    print('PREFLIGHT_PASS; no CUDA/training', flush=True)


def _receipt(plan, filename):
    path = Path(plan['outdir']) / filename
    row = json.loads(path.read_text(encoding='utf-8'))
    if row.get('plan_sha256') != plan['plan_sha256']:
        raise ValueError(f'Stale or foreign {filename}')
    if filename == 'preflight.json':
        action = Path(plan['cells'][0]['cfg']['action_npy'])
        if row.get('action_sha256') != _sha(action):
            raise ValueError('Motion prior changed after P3 preflight')
        old = Path(plan['data_home']) / 'runs/mmfi_p3_hpeli_s42_v1/preflight.json'
        expected = row.get('hpeli_split_receipt_sha256')
        if expected is not None and (not old.is_file() or _sha(old) != expected):
            raise ValueError('Completed HPE-Li split receipt changed')
    return row


def _smoke_worker(plan, backbone, attempt):
    import torch
    from vast_seed42_adapter import build_model
    cfg = _cell(plan, backbone, 's3', 'a1')['cfg']
    engine = _training_engine(cfg)
    gpu = _gpu()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(0)
    model = build_model(backbone, num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False).to('cuda:0')
    name, optimizer = engine._build_optimizer(model, cfg, 'mmfi')
    x = torch.rand(32, 3, 114, 10, device='cuda:0')
    y = torch.randn(32, 1, 17, 3, device='cuda:0')
    folder = Path(plan['outdir']) / 'smoke' / attempt
    ready = folder / (backbone + '.ready')
    ready.write_text('ready', encoding='ascii')
    deadline = time.monotonic() + 120
    while not all((folder / (name + '.ready')).is_file() for name in BACKBONES):
        if time.monotonic() > deadline or (folder / 'abort').exists():
            raise RuntimeError('Two-worker smoke failed to rendezvous')
        time.sleep(.1)
    for _ in range(3):
        loss = engine._victim_update(model, optimizer, x, y)
        if not math.isfinite(loss):
            raise RuntimeError('Nonfinite P3 model smoke loss')
    torch.cuda.synchronize()
    _write_new(folder / (backbone + '.json'), dict(plan_sha256=plan['plan_sha256'],
        attempt=attempt, backbone=backbone, gpu=gpu, batch_size=32,
        optimizer=name, steps=3, loss=loss,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(0)))


def gpu_smoke(plan):
    from run_mmfi_tables import _cell_lock
    _read_plan(plan)
    _receipt(plan, 'preflight.json')
    root = Path(plan['outdir'])
    with _cell_lock(root):
        gate_path = root / 'gpu_smoke.json'
        if gate_path.exists():
            _verify_gate(plan)
            print('GPU_SMOKE_PASS_EXISTING; no training', flush=True)
            return
        attempt = uuid.uuid4().hex
        folder = root / 'smoke' / attempt
        folder.mkdir(parents=True, exist_ok=False)
        workers = []
        for backbone in BACKBONES:
            log = (folder / (backbone + '.log')).open('x', encoding='utf-8')
            proc = subprocess.Popen(_command(plan, '--_smoke', backbone,
                                             '--_attempt', attempt), cwd=HERE,
                                    stdout=log, stderr=subprocess.STDOUT)
            log.close()
            workers.append((backbone, proc))
            print(f'[GPU smoke] {backbone} PID={proc.pid}', flush=True)
        codes = [proc.wait() for _, proc in workers]
        if any(codes):
            raise RuntimeError(f'Two-worker smoke failed {codes}; inspect {folder}')
        records = {}
        for backbone in BACKBONES:
            path = folder / (backbone + '.json')
            record = json.loads(path.read_text(encoding='utf-8'))
            if (record.get('plan_sha256') != plan['plan_sha256']
                    or record.get('attempt') != attempt
                    or record.get('backbone') != backbone
                    or record.get('batch_size') != 32
                    or record.get('steps') != 3
                    or record.get('peak_allocated_bytes', 0) <= 0):
                raise ValueError(f'Invalid two-job GPU smoke record: {backbone}')
            records[backbone] = _sha(path)
        _write_new(gate_path, dict(plan_sha256=plan['plan_sha256'], attempt=attempt,
            jobs_per_gpu=2, physical_batch=32, gpu=_gpu(), records=records))
    print('GPU_SMOKE_PASS; concurrent MetaFi++/WiFlow physical32', flush=True)


def _verify_gate(plan):
    root = Path(plan['outdir'])
    gate = _receipt(plan, 'gpu_smoke.json')
    if (gate.get('jobs_per_gpu') != 2 or gate.get('physical_batch') != 32
            or gate.get('gpu') != _gpu()):
        raise ValueError('GPU smoke device/concurrency changed')
    for backbone in BACKBONES:
        path = root / 'smoke' / gate['attempt'] / (backbone + '.json')
        if gate['records'].get(backbone) != _sha(path):
            raise ValueError('GPU smoke evidence changed')
    return gate


def _command(plan, private_flag, value, *extra):
    return [sys.executable, '-u', str(Path(__file__).resolve()),
        '--data-home', plan['data_home'], '--outdir', plan['outdir'],
        '--device', plan['device'], '--jobs-per-gpu', '2', private_flag, value,
        *extra]


def _complete(cell):
    from train_backdoor import _config_fingerprint, _load_cached_result
    import torch
    folder = Path(cell['folder'])
    path = folder / 'p3_evaluation.json'
    samples = folder / 'p3_per_sample.npz'
    ckpt = folder / 'checkpoint.pt'
    if not all(p.is_file() for p in (path, samples, ckpt)):
        return False
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    report = json.loads(path.read_text(encoding='utf-8'))
    return (saved.get('epoch') == 49
        and saved.get('cfg_fingerprint') == _config_fingerprint(cell['cfg'])
        and _load_cached_result(folder, cell['cfg']) is not None
        and report.get('status') == 'complete'
        and report.get('cell') == cell['id']
        and report.get('config_fingerprint') == _config_fingerprint(cell['cfg'])
        and report.get('checkpoint_sha256') == _sha(ckpt)
        and report.get('per_sample_sha256') == _sha(samples))


def _evaluate(plan, cell, cached):
    import numpy as np
    import torch
    from eval_a1_target_pck import _assert_replay, _measure, _predict_pair
    from train_backdoor import _config_fingerprint, build_trigger
    from vast_seed42_adapter import build_model
    folder = Path(cell['folder'])
    cfg = cell['cfg']
    ckpt = folder / 'checkpoint.pt'
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    fingerprint = _config_fingerprint(cfg)
    if (blob.get('epoch') != 49 or blob.get('cfg_fingerprint') != fingerprint
            or _config_fingerprint(blob.get('cfg', {})) != fingerprint):
        raise ValueError('Final49 checkpoint/config mismatch')
    model = build_model(cfg['model'], num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False)
    model.load_state_dict(blob['model'], strict=True)
    model = model.to(cfg['device']).eval()
    trigger = build_trigger(cfg)
    data = p3._dataset(cfg, 'test')
    predictions = _predict_pair(model, data, trigger, cfg, cfg['device'])
    measured, samples = _measure(*predictions)
    replay = _assert_replay(cached, measured, .05, attack=cell['key'] == 'a1')
    del model
    pair_ids = np.asarray([item['name'] for item in data.items], dtype=str)
    if len(pair_ids) != len(predictions[0]):
        raise ValueError('P3 evaluation identities/predictions differ')
    attack = cell['key'] == 'a1'
    dce = cached.get('dose_control_error')
    if attack and (not isinstance(dce, list) or len(dce) != 6
                   or any(not math.isfinite(float(x)) or float(x) < 0 for x in dce)
                   or abs(dce[0]) > 1e-8):
        raise ValueError('A1 six-dose DCE cache invalid')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, pair_ids=pair_ids, **samples)
    sample_path = folder / 'p3_per_sample.npz'
    temp = folder / 'p3_per_sample.npz.saving'
    if temp.exists():
        raise FileExistsError('Unfinished P3 per-sample write exists')
    if sample_path.exists():
        # A disconnect between NPZ and JSON is recoverable only when replay
        # produces identical arrays and ordered identities.
        with np.load(sample_path, allow_pickle=False) as old:
            expected = dict(pair_ids=pair_ids, **samples)
            if set(old.files) != set(expected) or any(
                    not np.array_equal(old[name], value)
                    for name, value in expected.items()):
                raise ValueError('Partial P3 per-sample arrays differ; preserve output')
    else:
        temp.write_bytes(buffer.getvalue())
        os.replace(temp, sample_path)
    metrics = dict(clean_mpjpe_mm=measured['clean_mpjpe_mm'],
        clean_pampjpe_mm=measured['clean_pampjpe_mm'],
        t_mpjpe_mm=measured['t1_mpjpe_mm'] if attack else None,
        target_pck_pct=measured['target_pck_0.2_pct'] if attack else None,
        no_trigger_t_mpjpe_mm=measured['no_trigger_t1_mpjpe_mm'] if attack else None,
        no_trigger_target_pck_pct=measured['no_trigger_target_pck_0.2_pct'] if attack else None,
        mean_positive_dce_mm=1000 * sum(dce[1:]) / 5 if attack else None)
    report = dict(status='complete', profile=PROFILE, cell=cell['id'],
        backbone=cell['backbone'], setting=cell['setting'], key=cell['key'],
        seed=42, checkpoint_epoch=49, frames=len(pair_ids), metrics=metrics,
        replay_abs_delta_mm=replay, plan_sha256=plan['plan_sha256'],
        config_fingerprint=fingerprint, checkpoint_sha256=_sha(ckpt),
        per_sample_sha256=_sha(sample_path),
        metric_contract=dict(target_joints=[2, 3], target_pck_threshold=.2,
            clean_scale_joints=[5, 12], dose=1.,
            dce='mean five positive doses', units='millimetres'),
        created_utc=datetime.now(timezone.utc).isoformat())
    _write_new(folder / 'p3_evaluation.json', report)
    print(f'EVALUATION_COMPLETE {cell["id"]}: {metrics}', flush=True)


def _train_cell(plan, cell):
    from run_mmfi_tables import _cell_lock
    from train_backdoor import _config_fingerprint
    with _cell_lock(cell['folder']):
        if _complete(cell):
            print(f'VERIFIED_COMPLETE {cell["id"]}', flush=True)
            return
        folder = Path(cell['folder'])
        cfg = cell['cfg']
        binding = dict(plan_sha256=plan['plan_sha256'], cell=cell['id'],
                       config_fingerprint=_config_fingerprint(cfg))
        source = folder / 'p3_cell_source.json'
        if source.exists():
            if json.loads(source.read_text(encoding='utf-8')) != binding:
                raise ValueError('P3 cell source/config binding changed')
        elif (folder / 'checkpoint.pt').exists():
            raise ValueError('Checkpoint without P3 source binding')
        else:
            _write_new(source, binding)
        engine = _training_engine(cfg)
        model, cached = engine.train(cfg, ckpt_dir=folder)
        del model, engine
        expected_train = EXPECTED_FRAMES[cell['setting']][0]
        if (cached.get('n_total') != expected_train
                or cached.get('n_poison') != math.floor(expected_train * cfg['rho'])):
            raise ValueError('P3 training/poison count changed')
        if not (folder / 'p3_evaluation.json').exists():
            _evaluate(plan, cell, cached)
        if not _complete(cell):
            raise RuntimeError(f'{cell["id"]}: final checkpoint/evaluation missing')
    print(f'COMPLETE {cell["id"]}', flush=True)


def _backbone_worker(plan, backbone):
    if backbone not in BACKBONES:
        raise ValueError('Unknown private backbone worker')
    _read_plan(plan)
    _receipt(plan, 'preflight.json')
    _verify_gate(plan)
    for setting in SETTINGS:
        for key in KEYS:
            cell = _cell(plan, backbone, setting, key)
            folder = Path(cell['folder'])
            folder.mkdir(parents=True, exist_ok=True)
            log = folder / 'console.log'
            with log.open('a', encoding='utf-8') as stream:
                old_out, old_err = sys.stdout, sys.stderr
                try:
                    sys.stdout = sys.stderr = stream
                    _train_cell(plan, cell)
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
            print(f'COMPLETE {cell["id"]}; log={log}', flush=True)


def launch(plan):
    from run_mmfi_tables import _cell_lock
    root = Path(plan['outdir'])
    with _cell_lock(root / '.controller'):
        _read_plan(plan)
        _receipt(plan, 'preflight.json')
        _verify_gate(plan)
        processes = []
        for backbone in BACKBONES:
            path = root / (backbone + '.controller.log')
            stream = path.open('a', encoding='utf-8')
            try:
                proc = subprocess.Popen(_command(plan, '--_backbone', backbone),
                    cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
            finally:
                stream.close()
            processes.append((backbone, proc))
            print(f'START {backbone} PID={proc.pid} log={path}', flush=True)
        codes = [(backbone, proc.wait()) for backbone, proc in processes]
        if any(code for _, code in codes):
            raise RuntimeError(f'P3 backbone worker failed {codes}; no automatic retry')
        records = []
        for cell in plan['cells']:
            if not _complete(cell):
                raise RuntimeError(f'Incomplete P3 cell: {cell["id"]}')
            records.append(json.loads((Path(cell['folder']) /
                                       'p3_evaluation.json').read_text(encoding='utf-8')))
        result = dict(status='COMPLETE_EXPLORATORY_P3_TWO_BACKBONES_S42',
                      plan_sha256=plan['plan_sha256'], records=records)
        path = root / RESULTS
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != result:
                raise ValueError('Existing final P3 report differs')
        else:
            _write_new(path, result)
    print('COMPLETE 12/12 independent P3 Clean/A1 victims', flush=True)


def status(plan):
    for cell in plan['cells']:
        folder = Path(cell['folder'])
        log = folder / 'console.log'
        epochs = (re.findall(r'epoch (\d+)/49:', log.read_text(encoding='utf-8',
            errors='replace')) if log.exists() else [])
        report = folder / 'p3_evaluation.json'
        print(f'{cell["id"]}: epoch={epochs[-1] if epochs else "-"}/49 '
              f'evaluation={report.exists()} log={log}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('dry-run', 'preflight', 'gpu-smoke',
        'launch', 'status', 'check-only'), default='dry-run')
    parser.add_argument('--data-home', type=Path,
                        default=Path.home() / 'backdooranalog')
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--jobs-per-gpu', type=int, default=2)
    parser.add_argument('--_smoke', choices=BACKBONES, help=argparse.SUPPRESS)
    parser.add_argument('--_backbone', choices=BACKBONES, help=argparse.SUPPRESS)
    parser.add_argument('--_attempt', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.outdir is None:
        args.outdir = args.data_home / 'runs/mmfi_p3_metafi_wiflow_s42_3090_v1'
    plan = build_plan(args.data_home, args.outdir, args.device, args.jobs_per_gpu)
    if args.mode == 'dry-run' and not (args._smoke or args._backbone):
        for cell in plan['cells']:
            cfg = cell['cfg']
            print(f'{cell["id"]}: 50 epochs final49, {cfg["optimizer"]} '
                  f'lr={cfg["lr"]}, B{cfg["batch_size"]}, rho={cfg["rho"]}')
        print('DRY_RUN_ONLY; 12 victims; max two simultaneous on cuda:0')
        return 0
    if args.mode == 'status' and not (args._smoke or args._backbone):
        status(plan)
        return 0
    if args._smoke:
        _read_plan(plan)
        if not args._attempt:
            raise ValueError('Smoke worker requires attempt')
        _smoke_worker(plan, args._smoke, args._attempt)
        return 0
    if args._backbone:
        _backbone_worker(plan, args._backbone)
        return 0
    if args._attempt:
        raise ValueError('Private attempt requires smoke worker')
    if args.mode == 'preflight':
        preflight(plan)
    elif args.mode == 'gpu-smoke':
        gpu_smoke(plan)
    elif args.mode == 'launch':
        launch(plan)
    elif args.mode == 'check-only':
        _read_plan(plan)
        _receipt(plan, 'preflight.json')
        _verify_gate(plan)
        if any(not _complete(cell) for cell in plan['cells']):
            raise ValueError('Some P3 cells are incomplete')
        final_path = Path(plan['outdir']) / RESULTS
        if not final_path.is_file():
            raise ValueError('Final P3 report is missing')
        final = json.loads(final_path.read_text(encoding='utf-8'))
        records = [json.loads((Path(cell['folder']) / 'p3_evaluation.json')
                              .read_text(encoding='utf-8')) for cell in plan['cells']]
        if (final.get('status') != 'COMPLETE_EXPLORATORY_P3_TWO_BACKBONES_S42'
                or final.get('plan_sha256') != plan['plan_sha256']
                or final.get('records') != records):
            raise ValueError('Final P3 report differs from authenticated cells')
        print('VERIFIED_COMPLETE 12/12', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, TimeoutError) as error:
        print(f'[P3 two backbones] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
