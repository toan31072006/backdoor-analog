"""Two exploratory MM-Fi HPE-Li attacks on one Windows RTX 4090.

Fresh seed-42 HPE-Li victims use the frozen P1S1/50-epoch recipe. LF and TSBA
train concurrently only after a paired real-model GPU smoke passes. Historical
runs and the completed attack table are never imported as writable outputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_hpeli_two_extra_attacks_s42_4090_v1'
METHODS = ('smooth_lf_csi', 'tsba_adapted')
EXPECTED_ACTION_SHA = '2e8fc11192cc3ee7ee3f5544cb3c5e19f771f7f7265dde93af8969af41a2be3e'
EXPECTED_COUNTS = dict(train=133056, test=33264)
SOURCE_NAMES = (
    'run_hpeli_extra_attacks_4090.py', 'extra_attack_eval.py',
    'run_a1_three_tables.py', 'run_proposed_ablation.py',
    'run_peak_confirmation.py', 'run_mmfi_tables.py',
    'run_piw_hpeli_clean.py', 'a1_backbone_adapter.py',
    'eval_a1_target_pck.py',
    'train_backdoor.py', 'attack/smooth_lf_csi.py', 'attack/tsba.py',
    'attack/traditional.py',
    'attack/trigger.py', 'attack/poison.py', 'attack/payload.py',
    'attack/skeleton_mmfi.py', 'data_utils/feeder.py',
    'models/factory.py', 'models/hpeli.py', 'models/sk_network.py',
    'eval/metrics.py', 'eval/target_pck.py',
    'configs/mmfi/attack_bend.yaml',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False,
                                     default=str).encode('utf-8')).hexdigest()


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Existing immutable receipt: {path}')
    temporary = path.with_name(path.name + '.saving')
    if temporary.exists():
        raise ValueError(f'Unfinished receipt: {temporary}')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        if path.exists():
            raise FileExistsError(f'Existing immutable receipt: {path}')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(root):
    from run_mmfi_tables import _cell_lock
    with _cell_lock(root):
        yield


def gpu_identity(device):
    import torch
    selected = torch.device(device)
    if selected.type != 'cuda' or selected.index != 0 or not torch.cuda.is_available():
        raise RuntimeError('This run requires the local cuda:0 RTX 4090')
    info = torch.cuda.get_device_properties(selected)
    if '4090' not in info.name or info.total_memory < 22 * 2**30:
        raise RuntimeError(f'Expected RTX 4090 with 24 GiB, found {info.name}')
    return dict(name=info.name, total_memory=info.total_memory,
                capability=[info.major, info.minor],
                uuid=str(getattr(info, 'uuid', '')))


def config(data_home, device, method):
    from run_a1_three_tables import _configuration
    from train_backdoor import _resolve_training_config, _validate_training_contract
    from attack.smooth_lf_csi import resolve_smooth_lf_config
    if method not in METHODS:
        raise ValueError('Unknown exploratory attack')
    original = _configuration(data_home, device, 0, 42, 'a1')
    cfg = {key: value for key, value in original.items()
           if key != 'confirmation_profile' and not key.startswith('method_')}
    cfg.update(trigger=method, extra_attack_profile=PROFILE,
               attacker_access=('white_box_training_control'
                                if method == 'tsba_adapted' else 'data_only'))
    if method == 'tsba_adapted':
        cfg.update(tsba_eps=0.1, tsba_hidden=32, tsba_lr=1e-3,
                   tsba_warmup_epochs=10, tsba_generator_steps=4)
    else:
        cfg = resolve_smooth_lf_config(cfg)
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    required = dict(experiment_name='mmfi', mmfi_protocol='protocol1',
                    mmfi_setting='s1', mmfi_split_seed=0, seed=42,
                    model='hpeli', pretrained=False, data_parallel=False,
                    optimizer='sgd', lr=.001, momentum=.9, weight_decay=0.,
                    batch_size=32, epochs=50, victim_epochs=50,
                    num_workers=0, strict_resume=True, pivot=1,
                    target_joints=[2, 3], theta_max_deg=40.,
                    payload_axis=[0., 0., 1.], rho=.1,
                    dose_min=.2, dose_max=1.,
                    dose_grid=[0., .2, .4, .6, .8, 1.],
                    dose_coupling='paired', measure_dose_control=True,
                    victim_loss='mpjpe')
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError('HPE-Li config differs from frozen MM-Fi P1S1 A1 recipe')
    return cfg


def build_plan(args):
    home, output = Path(args.data_home).resolve(), Path(args.outdir).resolve()
    if args.jobs_per_gpu != 2 or args.device != 'cuda:0':
        raise ValueError('This profile fixes two jobs on cuda:0')
    runs_root = (home / 'runs').resolve()
    if not output.is_relative_to(runs_root) or output == runs_root:
        raise ValueError('Output must be a dedicated directory under data-home/runs')
    for protected in (home / 'datasets', home / 'actions',
                      home / 'runs/mmfi_a1_three_tables_s42_43_44_v1',
                      home / 'runs/mmfi_proposed_factorial_ablation_s42_v1'):
        protected = protected.resolve()
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError(f'Output overlaps protected input/run: {protected}')
    cells = []
    for method in METHODS:
        cells.append(dict(id=f'hpeli__s42__{method}', method=method,
                          cfg=config(home, args.device, method),
                          run_dir=str(output / 'attacks/hpeli/seed42' / method),
                          eval_dir=str(output / 'evaluations/hpeli' / method)))
    value = dict(schema=1, profile=PROFILE, data_home=str(home), outdir=str(output),
                 device=args.device, jobs_per_gpu=2, seed=42,
                 status='EXPLORATORY_NOT_FROZEN_T1',
                 source_sha256={name: sha(HERE / name) for name in SOURCE_NAMES},
                 runtime=dict(python=platform.python_version()), cells=cells,
                 selection='50 scratch epochs, fixed final49; no TEST checkpoint selection',
                 caveats=['TSBA has white-box generator training access.',
                          'LF is a fixed smooth CSI control, not optimized LF reproduction.',
                          'Two attacks have different perturbation budgets and one seed.'])
    value['plan_sha256'] = digest(value)
    return value


def read_plan(args):
    path = Path(args.outdir).resolve() / 'extra_attack_plan.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved != build_plan(args):
        raise ValueError('Source/config/runtime/path changed; choose a NEW output')
    return saved


def cell(plan, method):
    matches = [item for item in plan['cells'] if item['method'] == method]
    if len(matches) != 1:
        raise ValueError('Missing or duplicate attack method')
    return matches[0]


def audit_inputs(plan):
    """Hash every selected CSI/GT file and check complete ordered P1S1 splits."""
    import numpy as np
    from train_backdoor import _load_dataset
    from run_peak_confirmation import _pair_ids
    cfg = plan['cells'][0]['cfg']
    root = Path(cfg['dataset_root']).resolve()
    action = Path(cfg['action_npy']).resolve()
    if sha(action) != EXPECTED_ACTION_SHA:
        raise ValueError('Motion prior data_bend.npy differs from authentic input')
    selected, splits, seen = {}, {}, []
    for name, split in (('train', 'training'), ('test', 'test')):
        ds = _load_dataset(cfg, split)
        ids = _pair_ids(ds)
        if len(ds) != EXPECTED_COUNTS[name] or len(ids) != len(ds):
            raise ValueError(f'Wrong full MM-Fi {name} split count')
        keys = {tuple(row) for row in ids}
        if len(keys) != len(ids):
            raise ValueError(f'Duplicate MM-Fi {name} identities')
        seen.append(keys)
        splits[name] = dict(n=len(ds), ordered_ids_sha256=digest(ids))
        for item in ds.items:
            for field in ('csi', 'kpt'):
                path = Path(item[field]).resolve()
                if not path.is_relative_to(root) or path.is_symlink():
                    raise ValueError('MM-Fi input path escapes the selected dataset')
                selected[path.relative_to(root).as_posix()] = path
        for index in (0, len(ds) // 2, len(ds) - 1):
            sample = ds[index]
            if (np.asarray(sample['csi']).shape != (3, 114, 10)
                    or np.asarray(sample['pose']).shape != (1, 17, 3)
                    or not np.isfinite(sample['csi']).all()
                    or not np.isfinite(sample['pose']).all()):
                raise ValueError(f'Invalid MM-Fi {name} data sample')
    if seen[0].intersection(seen[1]):
        raise ValueError('TRAIN/TEST frame identities overlap')
    records = []
    for index, (relative, path) in enumerate(sorted(selected.items()), 1):
        records.append((relative, path.stat().st_size, sha(path)))
        if index % 20000 == 0:
            print(f'[HPE-Li preflight] hashed {index}/{len(selected)} files', flush=True)
    return dict(schema=1, profile=PROFILE, data_home=plan['data_home'],
                action_sha256=EXPECTED_ACTION_SHA, splits=splits,
                unique_files=len(records), selected_content_sha256=digest(records))


def verify_input_receipt(plan, *, rehash):
    path = Path(plan['outdir']) / 'input_receipt.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('plan_sha256') != plan['plan_sha256']:
        raise ValueError('MM-Fi input receipt belongs to another plan')
    if rehash and saved.get('data') != audit_inputs(plan):
        raise ValueError('MM-Fi inputs changed after preflight')
    return sha(path)


def preflight(args, plan):
    root = Path(plan['outdir'])
    with run_lock(root):
        path = root / 'extra_attack_plan.json'
        if path.exists():
            read_plan(args)
        elif any(entry.name != 'run.lock' for entry in root.iterdir()):
            raise ValueError('Nonempty output without this plan; choose a NEW directory')
        data = audit_inputs(plan)
        if not path.exists():
            write_new_json(path, plan)
        receipt = root / 'input_receipt.json'
        content = dict(plan_sha256=plan['plan_sha256'], data=data)
        if receipt.exists():
            if json.loads(receipt.read_text(encoding='utf-8')) != content:
                raise ValueError('Existing input receipt differs; preserve old output')
        else:
            write_new_json(receipt, content)
    print('PREFLIGHT_PASS; full selected input bytes and splits verified; no training', flush=True)


def smoke_dir(plan, attempt):
    if not isinstance(attempt, str) or len(attempt) != 32 or any(
            char not in '0123456789abcdef' for char in attempt):
        raise ValueError('Invalid paired smoke attempt')
    return Path(plan['outdir']) / 'smoke' / attempt


def wait_barrier(folder, names, phase, *, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (folder / 'abort.json').is_file():
            raise RuntimeError('Paired GPU smoke peer failed')
        if all((folder / phase / (name + '.json')).is_file() for name in names):
            return
        time.sleep(.25)
    raise TimeoutError(f'Paired GPU smoke {phase} barrier timed out')


def private_engine(cfg, method):
    from a1_backbone_adapter import _private_module
    engine = _private_module('train_backdoor.py', '_hpeli_extra_train_engine')
    if method == 'smooth_lf_csi':
        from attack.smooth_lf_csi import build_smooth_lf_trigger
        engine.build_trigger = build_smooth_lf_trigger
    original_save = engine._save_checkpoint

    def atomic_save(path, *args, **kwargs):
        path = Path(path)
        temporary = path.with_name(path.name + '.saving')
        if temporary.exists():
            raise ValueError(f'Unfinished checkpoint: {temporary}')
        try:
            original_save(temporary, *args, **kwargs)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    engine._save_checkpoint = atomic_save
    return engine


def smoke_worker(plan, method, attempt):
    import numpy as np
    import torch
    from models.factory import build_model
    from run_piw_hpeli_clean import _setup_execution
    from train_backdoor import _generator_update
    worker = cell(plan, method)
    cfg = worker['cfg']
    device = plan['device']
    _setup_execution(device)
    gpu = gpu_identity(device)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False).to(device)
    engine = private_engine(cfg, method)
    optimizer_name, optimizer = engine._build_optimizer(model, cfg, 'mmfi')
    batch = cfg['batch_size']
    x = torch.rand(batch, 3, 114, 10, device=device)
    y = torch.randn(batch, 1, 17, 3, device=device)
    if method == 'tsba_adapted':
        trigger = engine.build_trigger(cfg).to(device)
        trigger_opt = torch.optim.Adam(trigger.parameters(), lr=cfg['tsba_lr'])
    else:
        trigger = engine.build_trigger(cfg)
        attacked = torch.from_numpy(np.stack([
            trigger.inject(frame, .6, cfg['eps']) for frame in x.cpu().numpy()
        ])).to(device)
    folder = smoke_dir(plan, attempt)
    marker = dict(plan_sha256=plan['plan_sha256'], method=method,
                  attempt=attempt, pid=os.getpid())
    write_new_json(folder / 'ready' / (method + '.json'), marker)
    wait_barrier(folder, METHODS, 'ready')
    for _ in range(4):
        if (folder / 'abort.json').is_file():
            raise RuntimeError('Paired GPU smoke peer failed')
        if method == 'tsba_adapted':
            doses = torch.full((batch,), .6, device=device)
            _generator_update(model, trigger, trigger_opt, x, y, doses, cfg)
            with torch.no_grad():
                attacked = trigger.inject_tensor(x, doses, eps=cfg['tsba_eps'])
        loss = engine._victim_update(model, optimizer, attacked, y)
        if not math.isfinite(loss):
            raise RuntimeError('Nonfinite actual HPE-Li smoke loss')
    torch.cuda.synchronize(device)
    write_new_json(folder / 'done' / (method + '.json'), marker)
    wait_barrier(folder, METHODS, 'done')
    return dict(method=method, attempt=attempt, plan_sha256=plan['plan_sha256'],
                gpu=gpu, batch_size=batch, optimizer=optimizer_name,
                concurrent_jobs=2, steps=4, loss=loss,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def worker_cmd(args, private_flag, method, *, attempt=None):
    command = [sys.executable, '-u', str(Path(__file__).resolve()),
               '--data-home', str(Path(args.data_home).resolve()),
               '--outdir', str(Path(args.outdir).resolve()),
               '--device', args.device, '--jobs-per-gpu', '2', private_flag, method]
    if attempt is not None:
        command += ['--_attempt', attempt]
    return command


def run_smoke(args, plan):
    attempt = uuid.uuid4().hex
    folder = smoke_dir(plan, attempt)
    children = []
    try:
        for method in METHODS:
            log = folder / (method + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('x', encoding='utf-8') as stream:
                proc = subprocess.Popen(worker_cmd(args, '--_smoke', method,
                                                   attempt=attempt),
                                        cwd=HERE, stdout=stream,
                                        stderr=subprocess.STDOUT)
            children.append((method, proc))
            print(f'[HPE-Li smoke] {method} PID={proc.pid} log={log}', flush=True)
    except OSError as error:
        write_new_json(folder / 'abort.json', dict(reason='smoke process launch failure'))
        raise RuntimeError(f'Could not start both smoke workers; inspect {folder}') from error
    while True:
        states = [(method, proc.poll()) for method, proc in children]
        failed = [(method, code) for method, code in states if code not in (None, 0)]
        if failed and not (folder / 'abort.json').exists():
            write_new_json(folder / 'abort.json', dict(failed=failed))
        if all(code is not None for _, code in states):
            break
        time.sleep(.25)
    if failed:
        raise RuntimeError(f'Paired GPU smoke failed {failed}; inspect {folder}')
    return attempt


def gate_payload(plan, attempt, gpu, receipt_sha):
    folder = smoke_dir(plan, attempt)
    if (folder / 'abort.json').exists():
        raise ValueError('Paired GPU smoke was aborted')
    value = dict(profile=PROFILE, plan_sha256=plan['plan_sha256'],
                 input_receipt_sha256=receipt_sha, gpu=gpu,
                 attempt=attempt, jobs_per_gpu=2, methods=list(METHODS),
                 smoke_sha256={}, barrier_sha256={})
    for method in METHODS:
        path = folder / (method + '.json')
        smoke = json.loads(path.read_text(encoding='utf-8'))
        if (smoke.get('method') != method or smoke.get('attempt') != attempt
                or smoke.get('plan_sha256') != plan['plan_sha256']
                or smoke.get('gpu') != gpu or smoke.get('concurrent_jobs') != 2
                or smoke.get('batch_size') != 32 or smoke.get('optimizer') != 'sgd'
                or smoke.get('steps') != 4
                or not isinstance(smoke.get('loss'), (int, float))
                or not math.isfinite(smoke['loss'])
                or smoke.get('peak_allocated_bytes', 0) <= 0):
            raise ValueError(f'Invalid HPE-Li smoke record for {method}')
        value['smoke_sha256'][method] = sha(path)
        for phase in ('ready', 'done'):
            marker = folder / phase / (method + '.json')
            record = json.loads(marker.read_text(encoding='utf-8'))
            if (record.get('method') != method or record.get('attempt') != attempt
                    or record.get('plan_sha256') != plan['plan_sha256']):
                raise ValueError(f'Invalid paired smoke {phase} marker')
            value['barrier_sha256'].setdefault(phase, {})[method] = sha(marker)
    return dict(value, gate_sha256=digest(value))


def verify_gate(plan):
    root = Path(plan['outdir'])
    gate = json.loads((root / 'prelaunch_gate.json').read_text(encoding='utf-8'))
    expected = gate_payload(plan, gate.get('attempt'), gpu_identity(plan['device']),
                            sha(root / 'input_receipt.json'))
    if gate != expected:
        raise ValueError('HPE-Li GPU smoke/input/source gate changed')
    return gate


def gpu_smoke(args, plan):
    root = Path(plan['outdir'])
    with run_lock(root):
        read_plan(args)
        if (root / 'prelaunch_gate.json').exists():
            verify_gate(plan)
            print('GPU_SMOKE_PASS_EXISTING; no training', flush=True)
            return
        verify_input_receipt(plan, rehash=True)
        gpu = gpu_identity(plan['device'])
        attempt = run_smoke(args, plan)
        receipt_sha = sha(root / 'input_receipt.json')
        write_new_json(root / 'prelaunch_gate.json',
                       gate_payload(plan, attempt, gpu, receipt_sha))
    print('GPU_SMOKE_PASS: two concurrent HPE-Li batch32 workers; no training', flush=True)


def train_cell(plan, method):
    from train_backdoor import _config_fingerprint
    from run_mmfi_tables import _cell_lock
    worker = cell(plan, method)
    run_dir = Path(worker['run_dir'])
    with _cell_lock(run_dir):
        receipt = run_dir / 'extra_source.json'
        binding = dict(plan_sha256=plan['plan_sha256'], method=method,
                       config_fingerprint=_config_fingerprint(worker['cfg']),
                       gpu=gpu_identity(plan['device']))
        if receipt.exists():
            if json.loads(receipt.read_text(encoding='utf-8')) != binding:
                raise ValueError('Checkpoint source/GPU binding changed')
        elif (run_dir / 'checkpoint.pt').exists():
            raise ValueError('Checkpoint without source binding; no silent resume')
        else:
            write_new_json(receipt, binding)
        engine = private_engine(worker['cfg'], method)
        model, result = engine.train(worker['cfg'], ckpt_dir=run_dir)
        del model, engine
        if result.get('n_total') != 133056 or result.get('n_poison') != 13305:
            raise ValueError('Completed HPE-Li poison count changed')
    print(f'TRAIN_COMPLETE {method}; fixed epoch49, TEST cache saved', flush=True)


def evaluate_cell(plan, method):
    from extra_attack_eval import evaluate_checkpoint
    from models.factory import build_model
    from attack.smooth_lf_csi import build_smooth_lf_trigger
    worker = cell(plan, method)
    result = evaluate_checkpoint(worker['run_dir'], worker['eval_dir'],
                                 plan['device'], model_builder=build_model,
                                 trigger_builder=build_smooth_lf_trigger
                                 if method == 'smooth_lf_csi' else None)
    if result.get('status') != 'complete' or result.get('checkpoint_epoch') != 49:
        raise ValueError(f'Incomplete HPE-Li evaluation: {method}')
    print(f'EVAL_COMPLETE {method}: {result["metrics"]}', flush=True)
    return result


def launch(args, plan):
    root = Path(plan['outdir'])
    with run_lock(root):
        read_plan(args)
        verify_gate(plan)
        verify_input_receipt(plan, rehash=True)
        children = []
        for method in METHODS:
            log = root / 'logs' / (method + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open('a', encoding='utf-8')
            try:
                proc = subprocess.Popen(worker_cmd(args, '--_train', method),
                                        cwd=HERE, stdout=stream,
                                        stderr=subprocess.STDOUT)
            finally:
                stream.close()
            children.append((method, proc))
            print(f'[HPE-Li TRAIN] {method} PID={proc.pid} log={log}', flush=True)
        failures = []
        for method, proc in children:
            code = proc.wait()
            if code:
                failures.append((method, code))
        if failures:
            raise RuntimeError(f'Training failed {failures}; checkpoints/logs kept; no automatic retry')
        verify_input_receipt(plan, rehash=True)
        records = [evaluate_cell(plan, method) for method in METHODS]
        result = dict(profile=PROFILE, plan_sha256=plan['plan_sha256'],
                      status='COMPLETE_EXPLORATORY_TWO_HPELI_ATTACKS',
                      seed=42, n_seeds=1, records=records)
        path = root / 'extra_attack_results.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != result:
                raise ValueError('Existing final result differs; preserve prior output')
        else:
            write_new_json(path, result)
    print('COMPLETE_EXPLORATORY_TWO_HPELI_ATTACKS', flush=True)


def check_only(args, plan):
    read_plan(args)
    verify_gate(plan)
    verify_input_receipt(plan, rehash=False)
    final_path = Path(plan['outdir']) / 'extra_attack_results.json'
    if not final_path.is_file():
        raise ValueError('Final result missing; inspect controller logs before CheckOnly')
    if any(not (Path(cell(plan, method)['eval_dir']) / 'evaluation.json').is_file()
           for method in METHODS):
        raise ValueError('An evaluation is missing; CheckOnly does not run inference')
    records = [evaluate_cell(plan, method) for method in METHODS]
    final = json.loads(final_path.read_text(encoding='utf-8'))
    if (final.get('status') != 'COMPLETE_EXPLORATORY_TWO_HPELI_ATTACKS'
            or final.get('plan_sha256') != plan['plan_sha256']
            or final.get('records') != records):
        raise ValueError('Final result does not match two frozen evaluations')
    print('VERIFIED_COMPLETE 2/2 HPE-Li exploratory attacks', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path,
                        default=Path.home() / 'backdooranalog')
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--jobs-per-gpu', type=int, default=2)
    modes = parser.add_mutually_exclusive_group()
    for name in ('dry-run', 'preflight', 'gpu-smoke', 'launch',
                 'status', 'check-only'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--_smoke', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--_train', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--_attempt', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.outdir is None:
        args.outdir = args.data_home / 'runs/mmfi_hpeli_extra_attacks_s42_4090_v1'
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(dict(status='DRY_RUN_ONLY', plan_sha256=plan['plan_sha256'],
                              profile=PROFILE, jobs_per_gpu=2,
                              cells=[dict(method=row['method'],
                                          config=row['cfg'], run_dir=row['run_dir'])
                                     for row in plan['cells']]),
                         indent=2, ensure_ascii=False))
        return 0
    if args.status:
        root = Path(plan['outdir'])
        for worker in plan['cells']:
            method = worker['method']
            print(f'{method}: checkpoint={(Path(worker["run_dir"])/"checkpoint.pt").is_file()}, '
                  f'evaluation={(Path(worker["eval_dir"])/"evaluation.json").is_file()}, '
                  f'log={root / "logs" / (method + ".log")}', flush=True)
        return 0
    if args._smoke or args._train:
        read_plan(args)
        if args._smoke:
            if not args._attempt:
                raise ValueError('Smoke worker missing attempt binding')
            result = smoke_worker(plan, args._smoke, args._attempt)
            write_new_json(smoke_dir(plan, args._attempt) /
                           (args._smoke + '.json'), result)
        else:
            verify_gate(plan)
            train_cell(plan, args._train)
        return 0
    if args._attempt:
        raise ValueError('Private attempt requires a smoke worker')
    if args.preflight:
        preflight(args, plan)
    elif args.gpu_smoke:
        gpu_smoke(args, plan)
    elif args.launch:
        launch(args, plan)
    elif args.check_only:
        check_only(args, plan)
    else:
        parser.error('Choose --dry-run, --preflight, --gpu-smoke, --launch, --status, or --check-only')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, TimeoutError, KeyError) as error:
        print(f'[HPE-Li extra attacks] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
