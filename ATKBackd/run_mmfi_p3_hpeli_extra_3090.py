"""Two HPE-Li CSI attack controls across official MM-Fi P3 S1/S2/S3.

Seed 42, six independent scratch victims, two concurrent batch-32 jobs on one
RTX 3090. This is an exploratory split extension, separate from the completed
HPE-Li Clean/A1 P3 and P1S1 attack runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
from importlib.metadata import version
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

import run_hpeli_extra_attacks_4090 as extra
import run_mmfi_p3_splits as p3


HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_p3_hpeli_smooth_lf_tsba_s42_3090_v1'
METHODS = extra.METHODS
SETTINGS = p3.SETTINGS
EXPECTED_FRAMES = {'s1': (256608, 64152), 's2': (256608, 64152),
                   's3': (240570, 80190)}
RESULTS = 'p3_extra_attack_results.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.saving')
    if path.exists() or temporary.exists():
        raise FileExistsError(f'Immutable or unfinished output: {path}')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                             encoding='utf-8')
        if path.exists():
            raise FileExistsError(f'Immutable output already exists: {path}')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def gpu_identity():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    info = torch.cuda.get_device_properties(0)
    if '3090' not in info.name or info.total_memory < 22 * 2**30:
        raise RuntimeError(f'Expected RTX 3090 on cuda:0, found {info.name}')
    return dict(name=info.name, total_memory=info.total_memory,
                capability=[info.major, info.minor],
                uuid=str(getattr(info, 'uuid', '')))


def config(home, setting, method):
    from train_backdoor import _resolve_training_config, _validate_training_contract
    if setting not in SETTINGS or method not in METHODS:
        raise ValueError('Only P3 S1/S2/S3 and Smooth-LF-CSI/TSBA adapted')
    cfg = extra.config(Path(home), 'cuda:0', method)
    cfg.update(mmfi_protocol='protocol3', mmfi_setting=setting,
               mmfi_split_seed=0, mmfi_random_ratio=.8,
               extra_attack_profile=PROFILE)
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    # The frozen HPE-Li A1 P3 recipe supplies the common victim and target
    # contract. Only the attack operator and its own extra fields may differ.
    reference = p3._configuration(Path(home), 'cuda:0', setting, 'a1')
    common = ('model', 'seed', 'experiment_name', 'mmfi_protocol', 'mmfi_setting',
              'mmfi_split_seed', 'mmfi_random_ratio', 'optimizer', 'lr',
              'momentum', 'weight_decay', 'batch_size', 'epochs', 'victim_epochs',
              'num_workers', 'pretrained', 'data_parallel', 'rho', 'pivot',
              'eps', 'theta_max_deg', 'payload_axis', 'target_joints',
              'dose_mode', 'dose_min', 'dose_max', 'dose_grid', 'dose_coupling',
              'victim_loss', 'poison_select')
    changed = [name for name in common if cfg.get(name) != reference.get(name)]
    if changed:
        raise ValueError(f'P3 HPE-Li common recipe changed: {changed}')
    return cfg


def build_plan(home, output, device='cuda:0', jobs=2):
    home, output = Path(home).resolve(), Path(output).resolve()
    if device != 'cuda:0' or jobs != 2:
        raise ValueError('This profile fixes two jobs on one cuda:0 RTX 3090')
    runs = (home / 'runs').resolve()
    if output == runs or not output.is_relative_to(runs):
        raise ValueError('Choose a dedicated output inside data-home/runs')
    protected = (home / 'datasets', home / 'actions',
                 home / 'runs/mmfi_p3_hpeli_s42_v1',
                 home / 'runs/mmfi_hpeli_extra_attacks_s42_4090_v1')
    if any(output == p.resolve() or output in p.resolve().parents
           or p.resolve() in output.parents for p in protected):
        raise ValueError('Output overlaps inputs or a completed experiment')
    cells = []
    for setting in SETTINGS:
        for method in METHODS:
            cells.append(dict(id=f'p3_{setting}__{method}', setting=setting,
                method=method, cfg=config(home, setting, method),
                folder=str(output / f'p3_{setting}' / method)))
    sources = set(extra.SOURCE_NAMES)
    sources.update(('run_mmfi_p3_hpeli_extra_3090.py', 'run_mmfi_p3_splits.py',
                    'mmfi_s2_split.py', 'extra_attack_eval.py'))
    plan = dict(schema=1, profile=PROFILE, data_home=str(home),
        outdir=str(output), device=device, jobs_per_gpu=2, seed=42,
        cells=cells, sources={name: sha(HERE / name) for name in sorted(sources)},
        runtime=dict(python=platform.python_version(), system=platform.system(),
                     packages={name: version(name) for name in
                               ('torch', 'numpy', 'scipy', 'PyYAML')}),
        contract=dict(status='EXPLORATORY_P3_HPELI_TWO_ADAPTED_ATTACKS_ONE_SEED',
            settings=list(SETTINGS), methods=list(METHODS), victims=6,
            clean_a1_reference='separate completed HPE-Li P3 S1/S2/S3 run',
            scratch=True, epochs=50, final_epoch=49, physical_batch=32,
            poison_rate=.1, target_joints=[2, 3],
            target_pck_threshold=.2, six_doses=[0., .2, .4, .6, .8, 1.],
            selection='fixed final49; no TEST selection',
            concurrent='one Smooth-LF and one TSBA process per P3 setting',
            limitations=['One seed; no across-seed SD/CI.',
                'Smooth-LF is a fixed CSI control, not original optimized LF.',
                'TSBA uses white-box generator training access.',
                'Attack budgets differ; report them explicitly.']))
    plan['plan_sha256'] = digest(plan)
    return plan


def plan_path(plan):
    return Path(plan['outdir']) / 'p3_extra_attack_plan.json'


def read_plan(plan):
    saved = json.loads(plan_path(plan).read_text(encoding='utf-8'))
    if saved != plan:
        raise ValueError('Saved source/config/runtime differs; use a NEW output')
    return saved


def cell(plan, setting, method):
    found = [row for row in plan['cells'] if row['setting'] == setting
             and row['method'] == method]
    if len(found) != 1:
        raise ValueError('Missing or duplicate P3 attack cell')
    return found[0]


def preflight(plan):
    import numpy as np
    from run_mmfi_tables import _cell_lock
    root = Path(plan['outdir'])
    with _cell_lock(root):
        if plan_path(plan).exists():
            read_plan(plan)
        elif any(p.name != 'run.lock' for p in root.iterdir()):
            raise ValueError('Nonempty output without P3 attack plan')
        action = Path(plan['cells'][0]['cfg']['action_npy'])
        if sha(action) != p3.EXPECTED_ACTION_SHA256:
            raise ValueError('data_bend.npy differs from completed HPE-Li P3')
        if not np.isfinite(np.load(action, allow_pickle=False)).all():
            raise ValueError('Nonfinite action prior')
        settings = {}
        for setting in SETTINGS:
            cfg = cell(plan, setting, METHODS[0])['cfg']
            train = p3._identity_digest(p3._dataset(cfg, 'training'))
            test_ds = p3._dataset(cfg, 'test')
            test = p3._identity_digest(test_ds)
            if ((train['frames'], test['frames']) != EXPECTED_FRAMES[setting]
                    or (train['sequences'], test['sequences']) !=
                       p3.EXPECTED_SEQUENCES[setting]
                    or train['sequence_keys'] & test['sequence_keys']):
                raise ValueError(f'P3{setting}: split count/identity mismatch')
            if setting in ('s2', 's3') and ({x[1] for x in train['sequence_keys']}
                                           & {x[1] for x in test['sequence_keys']}):
                raise ValueError(f'P3{setting}: TRAIN/TEST subject overlap')
            sample = test_ds[0]
            if (np.asarray(sample['csi']).shape != (3, 114, 10)
                    or np.asarray(sample['pose']).shape != (1, 17, 3)
                    or not np.isfinite(sample['csi']).all()
                    or not np.isfinite(sample['pose']).all()):
                raise ValueError(f'P3{setting}: invalid real TEST sample')
            train.pop('sequence_keys')
            test.pop('sequence_keys')
            settings[setting] = dict(train=train, test=test)
            print(f'[P3 attack preflight] {setting}: {train["frames"]} train / '
                  f'{test["frames"]} TEST', flush=True)
        old = Path(plan['data_home']) / 'runs/mmfi_p3_hpeli_s42_v1/preflight.json'
        if old.is_file() and json.loads(old.read_text(encoding='utf-8')).get('settings') != settings:
            raise ValueError('P3 split differs from completed HPE-Li P3')
        receipt = dict(plan_sha256=plan['plan_sha256'],
            action_sha256=p3.EXPECTED_ACTION_SHA256, settings=settings,
            hpeli_preflight_sha256=sha(old) if old.is_file() else None)
        path = root / 'preflight.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != receipt:
                raise ValueError('Saved P3 attack preflight differs')
        else:
            if not plan_path(plan).exists():
                write_new(plan_path(plan), plan)
            write_new(path, receipt)
    print('PREFLIGHT_PASS; no CUDA/training', flush=True)


def receipt(plan, name):
    path = Path(plan['outdir']) / name
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('plan_sha256') != plan['plan_sha256']:
        raise ValueError(f'Stale {name}')
    if name == 'preflight.json':
        if value.get('action_sha256') != sha(plan['cells'][0]['cfg']['action_npy']):
            raise ValueError('Motion prior changed after preflight')
        old_sha = value.get('hpeli_preflight_sha256')
        if old_sha:
            old = Path(plan['data_home']) / 'runs/mmfi_p3_hpeli_s42_v1/preflight.json'
            if not old.is_file() or sha(old) != old_sha:
                raise ValueError('Completed HPE-Li P3 receipt changed')
    return value


def smoke_folder(plan, attempt):
    return Path(plan['outdir']) / 'smoke' / attempt


def command(plan, private_flag, value, *more):
    return [sys.executable, '-u', str(Path(__file__).resolve()),
        '--data-home', plan['data_home'], '--outdir', plan['outdir'],
        '--device', 'cuda:0', '--jobs-per-gpu', '2', private_flag, value, *more]


def smoke_worker(plan, method, attempt):
    # Reuse the already exercised 4090 method-specific B32 update smoke.
    # The GPU identity and cell selection are replaced in this private process.
    extra.gpu_identity = lambda device: gpu_identity()
    smoke_plan = dict(plan, cells=[cell(plan, 's3', name) for name in METHODS])
    result = extra.smoke_worker(smoke_plan, method, attempt)
    write_new(smoke_folder(plan, attempt) / (method + '.json'), result)


def gpu_smoke(plan):
    from run_mmfi_tables import _cell_lock
    read_plan(plan)
    receipt(plan, 'preflight.json')
    root = Path(plan['outdir'])
    with _cell_lock(root):
        gate = root / 'gpu_smoke.json'
        if gate.exists():
            verify_gate(plan)
            print('GPU_SMOKE_PASS_EXISTING; no training', flush=True)
            return
        gpu = gpu_identity()
        attempt = uuid.uuid4().hex
        folder = smoke_folder(plan, attempt)
        folder.mkdir(parents=True, exist_ok=False)
        children = []
        for method in METHODS:
            log = (folder / f'{method}.log').open('x', encoding='utf-8')
            try:
                child = subprocess.Popen(command(plan, '--_smoke', method,
                    '--_attempt', attempt), cwd=HERE, stdout=log,
                    stderr=subprocess.STDOUT)
            finally:
                log.close()
            children.append((method, child))
            print(f'[P3 attack smoke] {method} PID={child.pid}', flush=True)
        while True:
            states = [(method, child.poll()) for method, child in children]
            failed = [(method, code) for method, code in states
                      if code not in (None, 0)]
            if failed and not (folder / 'abort.json').exists():
                write_new(folder / 'abort.json', dict(failed=failed))
            if all(code is not None for _, code in states):
                break
            time.sleep(.25)
        if failed:
            raise RuntimeError(f'Two-worker smoke failed {failed}; inspect {folder}')
        records = {}
        for method in METHODS:
            path = folder / f'{method}.json'
            row = json.loads(path.read_text(encoding='utf-8'))
            if (row.get('plan_sha256') != plan['plan_sha256']
                    or row.get('gpu') != gpu or row.get('batch_size') != 32
                    or row.get('concurrent_jobs') != 2 or row.get('steps') != 4):
                raise ValueError(f'Invalid physical32 paired smoke: {method}')
            records[method] = sha(path)
        write_new(gate, dict(plan_sha256=plan['plan_sha256'], gpu=gpu,
            attempt=attempt, jobs_per_gpu=2, physical_batch=32,
            records=records))
    print('GPU_SMOKE_PASS; two concurrent HPE-Li method workers', flush=True)


def verify_gate(plan):
    gate = receipt(plan, 'gpu_smoke.json')
    if (gate.get('gpu') != gpu_identity() or gate.get('jobs_per_gpu') != 2
            or gate.get('physical_batch') != 32):
        raise ValueError('RTX 3090 or two-worker smoke gate changed')
    folder = smoke_folder(plan, gate['attempt'])
    for method in METHODS:
        if gate['records'].get(method) != sha(folder / f'{method}.json'):
            raise ValueError(f'Paired smoke evidence changed: {method}')
    return gate


def _read_completed(row):
    from train_backdoor import _config_fingerprint, _load_cached_result
    import torch
    folder = Path(row['folder'])
    ckpt = folder / 'checkpoint.pt'
    report_path = folder / 'p3_evaluation.json'
    sample_path = folder / 'p3_per_sample.npz'
    if not all(p.is_file() for p in (ckpt, report_path, sample_path)):
        return None
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = row['cfg']
    if (saved.get('epoch') != 49
            or saved.get('cfg_fingerprint') != _config_fingerprint(cfg)
            or _load_cached_result(folder, cfg) is None):
        return None
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('status') != 'complete' or report.get('cell') != row['id']
            or report.get('checkpoint_sha256') != sha(ckpt)
            or report.get('per_sample_sha256') != sha(sample_path)
            or report.get('config_fingerprint') != _config_fingerprint(cfg)):
        return None
    return report


def evaluate(plan, row, cached):
    import numpy as np
    import torch
    from train_backdoor import (_config_fingerprint, _predict, build_trigger)
    from models.factory import build_model
    from eval import metrics as M
    from extra_attack_eval import GRID, _loader, _replay_mm, measure_pair
    from attack.smooth_lf_csi import build_smooth_lf_trigger

    folder = Path(row['folder'])
    cfg = row['cfg']
    if cached.get('dose_grid') != list(GRID):
        raise ValueError('Six-dose cache grid changed')
    for key in ('tmpjpe', 'clean_to_target_tmpjpe', 'dose_control_error'):
        values = cached.get(key)
        if (not isinstance(values, list) or len(values) != 6
                or any(not math.isfinite(float(x)) for x in values)):
            raise ValueError(f'Six-dose cache missing or invalid: {key}')
    if abs(cached['dose_control_error'][0]) > 1e-8:
        raise ValueError('Identity-dose DCE must be zero')
    blob = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    fingerprint = _config_fingerprint(cfg)
    if (blob.get('epoch') != 49 or blob.get('cfg_fingerprint') != fingerprint
            or _config_fingerprint(blob.get('cfg', {})) != fingerprint):
        raise ValueError('Final49 checkpoint/config mismatch')
    model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False)
    model.load_state_dict(blob['model'], strict=True)
    model = model.to('cuda:0').eval()
    trigger = (build_smooth_lf_trigger(cfg) if row['method'] == 'smooth_lf_csi'
               else build_trigger(cfg))
    trigger_state = blob.get('trigger')
    if getattr(trigger, 'requires_deferred_injection', False) and trigger_state is None:
        raise ValueError('TSBA generator state missing from checkpoint')
    if trigger_state is not None:
        trigger.load_state_dict(trigger_state, strict=True)
    if getattr(trigger, 'requires_deferred_injection', False):
        trigger = trigger.to('cuda:0').eval()
    data = p3._dataset(cfg, 'test')  # S2 uses the corrected official split.
    expected = EXPECTED_FRAMES[row['setting']][1]
    if len(data) != expected:
        raise ValueError(f'P3 TEST count changed: {len(data)}')
    pair_ids = np.asarray([item['name'] for item in data.items], dtype=str)
    if len(set(pair_ids)) != expected:
        raise ValueError('P3 TEST identities contain duplicates')
    if hasattr(data, 'enable_evaluation_cache'):
        data.enable_evaluation_cache()
    try:
        clean, true, _, _ = _predict(model, _loader(data, trigger, cfg, 'clean'),
                                     'cuda:0')
        clean_mm = 1000. * float(M.mpjpe(clean, true).mean())
        clean_pa_mm = 1000. * float(M.pa_mpjpe(clean, true).mean())
        replay = dict(clean_mpjpe_mm=_replay_mm(clean_mm, cached['clean_mpjpe'],
                                               'clean MPJPE'),
                      clean_pampjpe_mm=_replay_mm(clean_pa_mm,
                          cached['clean_pampjpe'], 'clean PA-MPJPE'))
        rows, target_errors, baseline_errors, dces = [], [], [], []
        scale = None
        for index, dose in enumerate(GRID):
            print(f'[P3 extra eval] {row["id"]} dose={dose:g}', flush=True)
            triggered, true_again, target, observed = _predict(
                model, _loader(data, trigger, cfg, 'trigger@dose', dose),
                'cuda:0', trig=trigger, cfg=cfg)
            if (not np.array_equal(true, true_again)
                    or not np.allclose(observed, dose, rtol=0, atol=1e-7)):
                raise ValueError('P3 clean/triggered TEST order or dose changed')
            measured, samples = measure_pair(clean, triggered, true, target)
            replay[f'd{dose:g}'] = {
                't_mpjpe_mm': _replay_mm(measured['t_mpjpe_mm'],
                    cached['tmpjpe'][index], f'T-MPJPE dose {dose:g}'),
                'no_trigger_t_mpjpe_mm': _replay_mm(
                    measured['no_trigger_t_mpjpe_mm'],
                    cached['clean_to_target_tmpjpe'][index],
                    f'no-trigger dose {dose:g}'),
                'dce_mm': _replay_mm(measured['dce_mm'],
                    cached['dose_control_error'][index], f'DCE dose {dose:g}'),
            }
            if scale is None:
                scale = samples['clean_scale_mm']
            elif not np.array_equal(scale, samples['clean_scale_mm']):
                raise ValueError('GT scale changed across dose passes')
            rows.append(dict(dose=dose, **measured))
            target_errors.append(samples['target_error_mm'])
            baseline_errors.append(samples['no_trigger_target_error_mm'])
            dces.append(samples['dce_mm'])
        if rows[0]['dce_mm'] > .05:
            raise ValueError('Identity dose DCE must be zero')
    finally:
        if hasattr(data, 'clear_evaluation_cache'):
            data.clear_evaluation_cache()
        del model
    metrics = dict(t_mpjpe_mm=rows[-1]['t_mpjpe_mm'],
        target_pck_pct=rows[-1]['target_pck_pct'],
        clean_mpjpe_mm=clean_mm, clean_pampjpe_mm=clean_pa_mm,
        mean_positive_dce_mm=float(np.mean([x['dce_mm'] for x in rows[1:]])))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, pair_ids=pair_ids, doses=np.asarray(GRID),
        target_error_mm=np.stack(target_errors),
        no_trigger_target_error_mm=np.stack(baseline_errors),
        dce_mm=np.stack(dces), clean_scale_mm=scale)
    sample_path = folder / 'p3_per_sample.npz'
    temp = folder / 'p3_per_sample.npz.saving'
    if temp.exists():
        raise FileExistsError(f'Unfinished per-sample file: {temp}')
    if sample_path.exists():
        with np.load(sample_path, allow_pickle=False) as old:
            with np.load(io.BytesIO(buffer.getvalue()), allow_pickle=False) as new:
                if set(old.files) != set(new.files) or any(
                        not np.array_equal(old[name], new[name]) for name in new.files):
                    raise ValueError('Partial P3 per-sample data differs')
    else:
        temp.write_bytes(buffer.getvalue())
        os.replace(temp, sample_path)
    report = dict(status='complete', cell=row['id'], profile=PROFILE,
        seed=42, setting=row['setting'], method=row['method'],
        checkpoint_epoch=49, frames=expected, metrics=metrics, dose_rows=rows,
        replay_abs_delta_mm=replay, plan_sha256=plan['plan_sha256'],
        config_fingerprint=fingerprint,
        checkpoint_sha256=sha(folder / 'checkpoint.pt'),
        per_sample_sha256=sha(sample_path),
        metric_contract=dict(target_joints=[2, 3], target_pck_threshold=.2,
            clean_scale_joints=[5, 12], units='millimetres'),
        created_utc=datetime.now(timezone.utc).isoformat())
    write_new(folder / 'p3_evaluation.json', report)
    print(f'EVALUATION_COMPLETE {row["id"]}: {metrics}', flush=True)


def train_worker(plan, row):
    from train_backdoor import _config_fingerprint
    from run_mmfi_tables import _cell_lock
    folder = Path(row['folder'])
    with _cell_lock(folder):
        if _read_completed(row) is not None:
            print(f'VERIFIED_COMPLETE {row["id"]}', flush=True)
            return
        cfg = row['cfg']
        binding = dict(plan_sha256=plan['plan_sha256'], cell=row['id'],
                       config_fingerprint=_config_fingerprint(cfg))
        source = folder / 'p3_cell_source.json'
        if source.exists():
            if json.loads(source.read_text(encoding='utf-8')) != binding:
                raise ValueError('Checkpoint source binding changed')
        elif (folder / 'checkpoint.pt').exists():
            raise ValueError('Checkpoint without P3 source binding')
        else:
            write_new(source, binding)
        engine = extra.private_engine(cfg, row['method'])
        if row['setting'] == 's2':
            from mmfi_s2_split import MMFIS2
            engine.MMFI = MMFIS2
        model, cached = engine.train(cfg, ckpt_dir=folder)
        del model, engine
        expected = EXPECTED_FRAMES[row['setting']][0]
        if (cached.get('n_total') != expected
                or cached.get('n_poison') != math.floor(expected * cfg['rho'])):
            raise ValueError('P3 poison/training count differs')
        if not (folder / 'p3_evaluation.json').exists():
            evaluate(plan, row, cached)
        if _read_completed(row) is None:
            raise RuntimeError(f'Incomplete final49 P3 attack cell: {row["id"]}')
    print(f'COMPLETE {row["id"]}', flush=True)


def wait_wave(running):
    states = [(row, child.wait()) for row, child in running]
    if any(code for _, code in states):
        raise RuntimeError(f'P3 attack worker failed: '
                           f'{[(r["id"], c) for r, c in states]}; no retry')
    if any(_read_completed(row) is None for row, _ in running):
        raise RuntimeError('P3 attack wave ended without verified outputs')


def launch(plan):
    from run_mmfi_tables import _cell_lock
    root = Path(plan['outdir'])
    with _cell_lock(root / '.controller'):
        read_plan(plan)
        receipt(plan, 'preflight.json')
        verify_gate(plan)
        for setting in SETTINGS:
            running = []
            for method in METHODS:
                row = cell(plan, setting, method)
                if _read_completed(row) is not None:
                    print(f'VERIFIED_COMPLETE {row["id"]}', flush=True)
                    continue
                folder = Path(row['folder'])
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder / 'console.log').open('a', encoding='utf-8')
                try:
                    child = subprocess.Popen(command(plan, '--_cell', row['id']),
                        cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
                finally:
                    log.close()
                running.append((row, child))
                print(f'START {row["id"]} PID={child.pid} log={folder / "console.log"}',
                      flush=True)
            if running:
                wait_wave(running)
            print(f'P3{setting} TWO ATTACKS COMPLETE', flush=True)
        records = []
        for row in plan['cells']:
            report = _read_completed(row)
            if report is None:
                raise RuntimeError(f'Incomplete P3 attack: {row["id"]}')
            records.append(report)
        result = dict(status='COMPLETE_EXPLORATORY_P3_HPELI_TWO_ATTACKS_S42',
                      plan_sha256=plan['plan_sha256'], records=records)
        final = root / RESULTS
        if final.exists():
            if json.loads(final.read_text(encoding='utf-8')) != result:
                raise ValueError('Existing final P3 result differs')
        else:
            write_new(final, result)
    print('COMPLETE 6/6 P3 HPE-Li attack victims', flush=True)


def status(plan):
    for row in plan['cells']:
        log = Path(row['folder']) / 'console.log'
        epochs = re.findall(r'epoch (\d+)/49:', log.read_text(encoding='utf-8',
            errors='replace')) if log.exists() else []
        report = Path(row['folder']) / 'p3_evaluation.json'
        print(f'{row["id"]}: epoch={epochs[-1] if epochs else "-"}/49 '
              f'evaluation={report.is_file()} log={log}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('dry-run', 'preflight', 'gpu-smoke',
        'launch', 'status', 'check-only'), default='dry-run')
    parser.add_argument('--data-home', type=Path,
                        default=Path.home() / 'backdooranalog')
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--jobs-per-gpu', type=int, default=2)
    parser.add_argument('--_smoke', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--_attempt', help=argparse.SUPPRESS)
    parser.add_argument('--_cell', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.outdir is None:
        args.outdir = args.data_home / 'runs/mmfi_p3_hpeli_extra_s42_3090_v1'
    plan = build_plan(args.data_home, args.outdir, args.device, args.jobs_per_gpu)
    if args.mode == 'dry-run' and not (args._smoke or args._cell):
        for row in plan['cells']:
            cfg = row['cfg']
            print(f'{row["id"]}: {cfg["epochs"]} epochs final49; '
                  f'{cfg["optimizer"]} lr={cfg["lr"]}; B{cfg["batch_size"]}; '
                  f'rho={cfg["rho"]}', flush=True)
        print('DRY_RUN_ONLY; six HPE-Li victims; max two jobs on cuda:0')
        return 0
    if args.mode == 'status' and not (args._smoke or args._cell):
        status(plan)
        return 0
    if args._smoke:
        read_plan(plan)
        if not args._attempt:
            raise ValueError('Smoke worker missing attempt')
        smoke_worker(plan, args._smoke, args._attempt)
        return 0
    if args._cell:
        read_plan(plan)
        receipt(plan, 'preflight.json')
        verify_gate(plan)
        rows = [row for row in plan['cells'] if row['id'] == args._cell]
        if len(rows) != 1:
            raise ValueError('Unknown private P3 attack cell')
        train_worker(plan, rows[0])
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
        read_plan(plan)
        receipt(plan, 'preflight.json')
        verify_gate(plan)
        records = [_read_completed(row) for row in plan['cells']]
        if any(report is None for report in records):
            raise ValueError('Some P3 HPE-Li attack cells are incomplete')
        final = json.loads((Path(plan['outdir']) / RESULTS).read_text(encoding='utf-8'))
        if (final.get('status') != 'COMPLETE_EXPLORATORY_P3_HPELI_TWO_ATTACKS_S42'
                or final.get('plan_sha256') != plan['plan_sha256']
                or final.get('records') != records):
            raise ValueError('Final P3 result differs from verified cells')
        print('VERIFIED_COMPLETE 6/6 P3 HPE-Li attack victims', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
