"""Exploratory TSBA and smooth-LF CSI attacks on MM-Fi P1S1, seed 42.

This overlay imports the byte-verified historical Vast source without editing it.
Four independent scratch victims are trained: two attacks on each of MetaFi++
and WiFlow.  These results are separate from the frozen eight-row T1 table.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid


HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_two_exploratory_attacks_s42_v3_two_per_gpu'
BACKBONES = ('metafiplusplus', 'wiflow')
METHODS = ('smooth_lf_csi', 'tsba_adapted')
DEFAULT_REPO = Path('/workspace/mmfi_v6/backdoor-analog')
DEFAULT_DATA = Path('/workspace/mmfi')
DEFAULT_OUT = Path('/workspace/runs/mmfi_two_extra_s42_v3')
DEFAULT_RECEIPT = Path('/workspace/runs/mmfi_metafiplusplus_s42_attack_defense_v1/dataset_content.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.saving')
    if temporary.exists():
        raise ValueError(f'Unfinished output exists: {temporary}')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def controller_lock(root):
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'controller.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another extra-attack controller holds this output') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def module_file(filename, repo):
    local = HERE / filename
    if local.is_file():
        return local
    # Local source checkout layout; the Vast overlay contains flattened files.
    subfolder = 'attack/' if filename == 'smooth_lf_csi.py' else ''
    return repo / 'ATKBackd' / subfolder / filename


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import overlay module {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def setup_source(repo):
    atk = repo / 'ATKBackd'
    if not (atk / 'vast_seed42_adapter.py').is_file():
        raise FileNotFoundError(f'Missing frozen Vast ATKBackd: {atk}')
    sys.path.insert(0, str(repo / 'remote_linux/vast_seed42'))
    sys.path.insert(0, str(atk))
    import vast_seed42_adapter as adapter
    from train_backdoor import _resolve_training_config, _validate_training_contract
    lf = load_file('extra_smooth_lf_csi', module_file('smooth_lf_csi.py', repo))
    ev = load_file('extra_attack_eval', module_file('extra_attack_eval.py', repo))
    return adapter, lf, ev, _resolve_training_config, _validate_training_contract


def configure(repo, data_home, device, backbone, method):
    if backbone not in BACKBONES or method not in METHODS:
        raise ValueError('Only two backbones and two declared exploratory attacks are permitted')
    adapter, lf, _, resolve, validate = setup_source(repo)
    cfg = adapter.canonical_config(data_home, device, 'a1', backbone)
    # Keep data, payload, poison rate, dose grid and victim recipe from the
    # frozen A1 configuration. Remove only A1 operator/profile metadata.
    cfg = {key: value for key, value in cfg.items()
           if key != 'confirmation_profile' and not key.startswith('method_')}
    cfg['trigger'] = method
    cfg['extra_attack_profile'] = PROFILE
    cfg['attacker_access'] = ('white_box_training_control' if method == 'tsba_adapted'
                              else 'data_only')
    if method == 'tsba_adapted':
        cfg.update(tsba_eps=0.1, tsba_hidden=32, tsba_lr=1e-3,
                   tsba_warmup_epochs=10, tsba_generator_steps=4)
    else:
        cfg = lf.resolve_smooth_lf_config(cfg)
    cfg = resolve(cfg)
    validate(cfg)
    # Catch accidental recipe changes before an expensive job is scheduled.
    expected = adapter.recipes()[backbone]
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError('Victim recipe differs from frozen MetaFi++/WiFlow recipe')
    fixed = dict(experiment_name='mmfi', mmfi_protocol='protocol1',
                 mmfi_setting='s1', mmfi_split_seed=0, seed=42,
                 pivot=1, target_joints=[2, 3], theta_max_deg=40.0,
                 payload_axis=[0.0, 0.0, 1.0], rho=0.1,
                 dose_min=0.2, dose_max=1.0,
                 dose_grid=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                 epochs=50, batch_size=32, victim_loss='mpjpe',
                 measure_dose_control=True)
    if any(cfg.get(key) != value for key, value in fixed.items()):
        raise ValueError('Data/target/dose contract differs from frozen P1S1 seed42')
    return cfg


def source_binding(repo):
    required = ('extra_attacks_runner.py', 'smooth_lf_csi.py', 'extra_attack_eval.py')
    values = {name: sha(module_file(name, repo)) for name in required}
    bundle_manifest = repo / 'BUNDLE_MANIFEST.json'
    if bundle_manifest.is_file():
        values['frozen_v6_bundle_manifest'] = sha(bundle_manifest)
    return values


def make_plan(args):
    repo, data, out = (Path(getattr(args, name)).resolve()
                       for name in ('repo_dir', 'data_home', 'outdir'))
    for protected in (repo, data / 'datasets', data / 'actions',
                      Path('/workspace/runs/mmfi_metafiplusplus_s42_attack_defense_v1'),
                      Path('/workspace/runs/mmfi_wiflow_s42_attacks_only_v2')):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError(f'Choose a new isolated output, not {protected}')
    cells = []
    for method in METHODS:
        for gpu, backbone in enumerate(BACKBONES):
            device = f'cuda:{gpu}'
            cfg = configure(repo, data, device, backbone, method)
            name = f'{backbone}__s42__{method}'
            cells.append(dict(id=name, backbone=backbone, method=method,
                              device=device, config=cfg,
                              run_dir=str(out / 'attacks' / backbone / 'seed42' / method),
                              eval_dir=str(out / 'evaluations' / backbone / method)))
    value = dict(schema=1, profile=PROFILE, dataset='MM-Fi P1S1', seed=42,
                 status='EXPLORATORY_NOT_FROZEN_T1',
                 repo_dir=str(repo), data_home=str(data), outdir=str(out),
                 dataset_receipt=str(Path(args.dataset_receipt).resolve()),
                 source_binding=source_binding(repo),
                 runtime=dict(python=platform.python_version(),
                              torch=version('torch'), numpy=version('numpy'),
                              scipy=version('scipy'), pyyaml=version('PyYAML')),
                 cells=cells,
                 scheduling='four concurrent processes; two per GPU; combined four-worker smoke barrier',
                 selection='scratch 50 epochs, fixed final49; no TEST checkpoint selection',
                 caveats=['TSBA adapted generator has white-box training access.',
                          'Smooth-LF is a fixed CSI control inspired by LF, not LF optimized-pattern reproduction.',
                          'Only one seed; do not append these results to frozen T1 without provenance.'])
    value['plan_sha256'] = digest(value)
    return value


def read_plan(args):
    path = Path(args.outdir).resolve() / 'extra_attack_plan.json'
    if not path.is_file():
        raise FileNotFoundError('Run --prepare-only first; no training launched')
    saved = json.loads(path.read_text(encoding='utf-8'))
    expected = make_plan(args)
    if saved != expected:
        raise ValueError('Plan/source/config/path changed; use a NEW output directory')
    return saved


def verify_frozen(repo):
    from verify_bundle import main as verify
    if not (repo / 'BUNDLE_MANIFEST.json').is_file():
        raise FileNotFoundError('Frozen v6 BUNDLE_MANIFEST.json missing')
    verify(['--root', str(repo)])


def verify_data(plan, *, full_hash):
    from vast_seed42_preflight import check_receipt
    receipt_path = Path(plan['dataset_receipt'])
    if not receipt_path.is_file():
        raise FileNotFoundError('Existing full MM-Fi content receipt missing: ' + str(receipt_path))
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('data_home') != plan['data_home'] or receipt.get('splits', {}).get('train', {}).get('n') != 133056 \
            or receipt.get('splits', {}).get('eval', {}).get('n') != 33264:
        raise ValueError('Dataset receipt differs from full P1S1 split')
    check_receipt(receipt, full_hash=full_hash)
    return sha(receipt_path)


def gpu_identity():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('This launcher requires exactly two visible CUDA GPUs')
    identities = []
    for index in range(2):
        p = torch.cuda.get_device_properties(index)
        if p.total_memory < 20 * 2**30:
            raise RuntimeError(f'cuda:{index} has under 20 GiB VRAM')
        identities.append(dict(index=index, name=p.name,
                               total_memory=p.total_memory,
                               capability=[p.major, p.minor],
                               uuid=str(getattr(p, 'uuid', ''))))
    return identities


def require_idle_gpus():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True, timeout=30)
    pids = {int(line.strip()) for line in result.stdout.splitlines()
            if line.strip() and line.strip().isdigit()}
    if any(line.strip() and not line.strip().isdigit()
           for line in result.stdout.splitlines()):
        raise RuntimeError('Cannot parse GPU compute process inventory')
    others = pids - {os.getpid()}
    if others:
        raise RuntimeError(f'Other GPU compute PIDs {sorted(others)} are active; inspect nvidia-smi')


def smoke_attempt_dir(plan, attempt):
    if not isinstance(attempt, str) or len(attempt) != 32 or any(
            char not in '0123456789abcdef' for char in attempt):
        raise ValueError('Invalid four-worker smoke attempt ID')
    return Path(plan['outdir']) / 'smoke' / 'attempts' / attempt


def wait_smoke_barrier(plan, attempt, phase, *, timeout=300):
    if phase not in ('ready', 'done'):
        raise ValueError('Unknown smoke barrier phase')
    folder = smoke_attempt_dir(plan, attempt)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (folder / 'abort.json').is_file():
            raise RuntimeError('Peer smoke failed; see four-worker attempt logs')
        complete = True
        for peer in plan['cells']:
            marker = folder / phase / (peer['id'] + '.json')
            if not marker.is_file():
                complete = False
                break
            record = json.loads(marker.read_text(encoding='utf-8'))
            if (record.get('cell') != peer['id'] or record.get('attempt') != attempt
                    or record.get('plan_sha256') != plan['plan_sha256']
                    or record.get('device') != peer['device']):
                raise ValueError(f'Invalid {phase} smoke marker: {marker}')
        if complete:
            return
        time.sleep(0.25)
    raise TimeoutError(f'Four-worker smoke {phase} barrier timed out')


def run_smoke_worker(plan, cell, attempt):
    import numpy as np
    import torch
    adapter, lf, _, _, _ = setup_source(Path(plan['repo_dir']))
    from train_backdoor import _generator_update
    from run_piw_hpeli_clean import _setup_execution
    cfg = cell['config']
    device = cell['device']
    _setup_execution(device)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(device)
    model = adapter.build_model(cell['backbone'], num_keypoints=17,
                                subcarrier_num=114, dataset='mmfi',
                                pretrained=False).to(device)
    engine = adapter.training_engine(cfg)
    if cell['method'] == 'smooth_lf_csi':
        engine.build_trigger = lf.build_smooth_lf_trigger
    optimizer_name, optimizer = engine._build_optimizer(model, cfg, 'mmfi')
    batch = cfg['batch_size']
    x = torch.rand(batch, 3, 114, 10, device=device)
    y = torch.randn(batch, 1, 17, 3, device=device)
    if cell['method'] == 'tsba_adapted':
        trigger = engine.build_trigger(cfg).to(device)
        trigger_opt = torch.optim.Adam(trigger.parameters(), lr=cfg['tsba_lr'])
    else:
        trigger = lf.build_smooth_lf_trigger(cfg)
        attacked = torch.from_numpy(np.stack([
            trigger.inject(frame, 0.6, cfg['eps']) for frame in x.cpu().numpy()
        ])).to(device)
    folder = smoke_attempt_dir(plan, attempt)
    marker = dict(cell=cell['id'], attempt=attempt,
                  plan_sha256=plan['plan_sha256'], device=device, pid=os.getpid())
    atomic_json(folder / 'ready' / (cell['id'] + '.json'), marker)
    wait_smoke_barrier(plan, attempt, 'ready')
    # All four models and optimizers are resident before any backward pass.
    # The learned generator gets a full physical batch here to stress VRAM.
    for _ in range(4):
        if (folder / 'abort.json').is_file():
            raise RuntimeError('Peer smoke failed; stopping synthetic steps')
        if cell['method'] == 'tsba_adapted':
            doses = torch.full((batch,), 0.6, device=device)
            _generator_update(model, trigger, trigger_opt, x, y, doses, cfg)
            with torch.no_grad():
                attacked = trigger.inject_tensor(x, doses, eps=cfg['tsba_eps'])
        loss = engine._victim_update(model, optimizer, attacked, y)
        if not np.isfinite(loss):
            raise RuntimeError('Nonfinite synthetic actual-model smoke loss')
    torch.cuda.synchronize(device)
    atomic_json(folder / 'done' / (cell['id'] + '.json'), marker)
    wait_smoke_barrier(plan, attempt, 'done')
    return dict(cell=cell['id'], physical_batch=batch, optimizer=optimizer_name,
                finite=True, loss=loss, attempt=attempt, concurrent_workers=4,
                gpu=gpu_identity()[int(device.split(':')[1])],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def run_cell(plan, cell):
    repo = Path(plan['repo_dir'])
    adapter, lf, ev, _, _ = setup_source(repo)
    from run_piw_hpeli_clean import _setup_execution
    from train_backdoor import _config_fingerprint
    cfg = cell['config']
    _setup_execution(cell['device'])
    verify_data(plan, full_hash=False)
    run_dir, eval_dir = Path(cell['run_dir']), Path(cell['eval_dir'])
    from run_mmfi_tables import _cell_lock
    with _cell_lock(run_dir):
        allowed = {'checkpoint.pt', 'config.resolved.yaml', 'poison_manifest.json',
                   'eval_cache.json', 'results.json', 'run.lock', 'extra_source.json'}
        if any(p.name not in allowed for p in run_dir.iterdir()):
            raise ValueError('Unexpected run artifact; preserve and inspect before resuming')
        receipt = run_dir / 'extra_source.json'
        binding = dict(plan_sha256=plan['plan_sha256'], cell=cell['id'],
                       config_fingerprint=_config_fingerprint(cfg),
                       gpu=gpu_identity()[int(cell['device'].split(':')[1])])
        if receipt.exists():
            if json.loads(receipt.read_text(encoding='utf-8')) != binding:
                raise ValueError('Checkpoint source/GPU binding differs; no cross-host resume')
        elif (run_dir / 'checkpoint.pt').exists():
            raise ValueError('Checkpoint without source receipt; no silent resume')
        else:
            atomic_json(receipt, binding)
        engine = adapter.training_engine(cfg)
        if cell['method'] == 'smooth_lf_csi':
            engine.build_trigger = lf.build_smooth_lf_trigger
        model, _ = engine.train(cfg, ckpt_dir=run_dir)
        del model, engine
        trigger_builder = (lf.build_smooth_lf_trigger
                           if cell['method'] == 'smooth_lf_csi' else None)
        report = ev.evaluate_checkpoint(run_dir, eval_dir, cell['device'],
                                        model_builder=adapter.build_model,
                                        trigger_builder=trigger_builder)
    if report['metrics']['t1_mpjpe_mm'] < 0:
        raise RuntimeError('Invalid completed metric')
    return report


def get_cell(plan, name):
    cells = [cell for cell in plan['cells'] if cell['id'] == name]
    if len(cells) != 1:
        raise ValueError('Unknown or duplicate cell')
    return cells[0]


def internal_command(args, mode, cell, *, smoke_attempt=None):
    command = [sys.executable, '-u', str(Path(__file__).resolve()),
            '--repo-dir', str(Path(args.repo_dir).resolve()),
            '--data-home', str(Path(args.data_home).resolve()),
            '--outdir', str(Path(args.outdir).resolve()),
            '--dataset-receipt', str(Path(args.dataset_receipt).resolve()),
            mode, cell['id']]
    if smoke_attempt is not None:
        command += ['--smoke-attempt', smoke_attempt]
    return command


def wave(args, plan, method, mode):
    children = []
    for cell in plan['cells']:
        if method is not None and cell['method'] != method:
            continue
        log_dir = Path(plan['outdir']) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / (cell['id'] + '.log')
        handle = path.open('a', encoding='utf-8')
        process = subprocess.Popen(internal_command(args, mode, cell),
                                   stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        print(f'[{mode[2:]}] {cell["id"]} PID={process.pid} GPU={cell["device"]} log={path}', flush=True)
        children.append((cell, process))
    failures = []
    for cell, process in children:
        code = process.wait()
        if code:
            failures.append((cell['id'], code))
    if failures:
        raise RuntimeError(f'{method} wave failed {failures}; logs/checkpoints retained; no automatic retry')


def smoke_all(args, plan):
    """Run the actual four-cell topology, with a resident-model barrier."""
    attempt = uuid.uuid4().hex
    folder = smoke_attempt_dir(plan, attempt)
    children = []
    try:
        for cell in plan['cells']:
            path = folder / 'logs' / (cell['id'] + '.log')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as handle:
                process = subprocess.Popen(
                    internal_command(args, '--smoke-worker', cell, smoke_attempt=attempt),
                    stdout=handle, stderr=subprocess.STDOUT)
            print(f'[concurrent-smoke] {cell["id"]} PID={process.pid} '
                  f'GPU={cell["device"]} log={path}', flush=True)
            children.append((cell, process))
    except OSError as error:
        atomic_json(folder / 'abort.json',
                    dict(attempt=attempt, reason='failed to launch all smoke workers'))
        raise RuntimeError(f'Four-worker smoke launch failed; preserve {folder}') from error
    failures = []
    while True:
        states = [(cell['id'], process.poll()) for cell, process in children]
        newly_failed = [(name, code) for name, code in states if code not in (None, 0)]
        if newly_failed and not (folder / 'abort.json').exists():
            atomic_json(folder / 'abort.json', dict(attempt=attempt, failures=newly_failed))
        if all(code is not None for _, code in states):
            failures = [(name, code) for name, code in states if code != 0]
            break
        time.sleep(0.25)
    if failures:
        raise RuntimeError(f'Four-worker concurrent GPU smoke failed {failures}; '
                           f'preserve {folder}; no training launched')
    return attempt


def gate_payload(plan, attempt, gpus, receipt_sha):
    folder = smoke_attempt_dir(plan, attempt)
    if (folder / 'abort.json').exists():
        raise ValueError('Smoke attempt was aborted')
    expected = dict(profile=PROFILE, plan_sha256=plan['plan_sha256'],
                    dataset_receipt_sha256=receipt_sha,
                    gpus=gpus, smoke_attempt=attempt,
                    concurrency=dict(workers=4, jobs_per_gpu=2,
                                     barrier='four resident models; full-batch TSBA backward'))
    for cell in plan['cells']:
        smoke = folder / (cell['id'] + '.json')
        if not smoke.is_file():
            raise FileNotFoundError(f'Missing actual-model GPU smoke {smoke}')
        value = json.loads(smoke.read_text(encoding='utf-8'))
        gpu_index = int(cell['device'].split(':')[1])
        if (value.get('cell') != cell['id'] or value.get('finite') is not True
                or value.get('gpu') != expected['gpus'][gpu_index]
                or value.get('attempt') != attempt
                or value.get('concurrent_workers') != 4
                or value.get('physical_batch') != cell['config']['batch_size']
                or value.get('optimizer') != cell['config']['optimizer']
                or not isinstance(value.get('loss'), (int, float))
                or not math.isfinite(value['loss'])
                or value.get('peak_allocated_bytes', 0) <= 0):
            raise ValueError('Failed actual-model GPU smoke')
        expected.setdefault('smoke_sha256', {})[cell['id']] = sha(smoke)
        for phase in ('ready', 'done'):
            marker = folder / phase / (cell['id'] + '.json')
            record = json.loads(marker.read_text(encoding='utf-8'))
            if (record.get('cell') != cell['id'] or record.get('attempt') != attempt
                    or record.get('plan_sha256') != plan['plan_sha256']
                    or record.get('device') != cell['device']):
                raise ValueError(f'Invalid {phase} smoke marker: {marker}')
            expected.setdefault('barrier_sha256', {}).setdefault(phase, {})[cell['id']] = sha(marker)
    return dict(expected, gate_sha256=digest(expected))


def validate_gate(args, plan):
    output = Path(plan['outdir'])
    gate_path = output / 'prelaunch_gate.json'
    if not gate_path.is_file():
        raise FileNotFoundError('Separate --prepare-only gate missing')
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    expected = gate_payload(plan, gate.get('smoke_attempt'), gpu_identity(),
                            sha(plan['dataset_receipt']))
    if gate != expected:
        raise ValueError('Prepared source/data/GPU/smoke gate changed')
    return gate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--data-home', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--dataset-receipt', type=Path, default=DEFAULT_RECEIPT)
    modes = parser.add_mutually_exclusive_group()
    for name in ('dry-run', 'prepare-only', 'launch', 'status', 'check-only'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--smoke-worker', metavar='CELL')
    parser.add_argument('--smoke-attempt', metavar='HEX_ID')
    parser.add_argument('--worker', metavar='CELL')
    args = parser.parse_args(argv)
    if not any((args.dry_run, args.prepare_only, args.launch, args.status,
                args.check_only, args.smoke_worker, args.worker)):
        parser.error('Choose a mode; default training is forbidden')
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    if args.dry_run:
        print('DRY_RUN_ONLY; four scratch victims, no data/CUDA/output writes')
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    if args.status:
        root = Path(plan['outdir'])
        for cell in plan['cells']:
            log = root / 'logs' / (cell['id'] + '.log')
            evaluation = Path(cell['eval_dir']) / 'evaluation.json'
            print(f'{cell["id"]}: eval={"present" if evaluation.is_file() else "missing"}; '
                  f'checkpoint={"present" if (Path(cell["run_dir"])/"checkpoint.pt").is_file() else "missing"}; '
                  f'log={log if log.exists() else "missing"}')
        return 0
    if args.smoke_worker or args.worker:
        saved = read_plan(args)
        cell = get_cell(saved, args.smoke_worker or args.worker)
        if args.smoke_worker:
            value = run_smoke_worker(saved, cell, args.smoke_attempt)
            atomic_json(smoke_attempt_dir(saved, args.smoke_attempt) /
                        (cell['id'] + '.json'), value)
            print(f'GPU_SMOKE_WORKER_PASS {cell["id"]}: {value}', flush=True)
        else:
            validate_gate(args, saved)
            report = run_cell(saved, cell)
            print(f'COMPLETE {cell["id"]}: {report["metrics"]}', flush=True)
        return 0
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        raise RuntimeError('Leave CUDA_VISIBLE_DEVICES unset; fixed logical GPU binding')
    if platform.system() != 'Linux':
        raise RuntimeError('This launch/gate is for the Vast Linux two-GPU host')
    repo = Path(args.repo_dir).resolve()
    verify_frozen(repo)
    if args.prepare_only:
        output = Path(plan['outdir'])
        with controller_lock(output):
            current_plan = output / 'extra_attack_plan.json'
            if current_plan.exists():
                saved = read_plan(args)
                if (output / 'prelaunch_gate.json').is_file():
                    validate_gate(args, saved)
                    print('PREPARED_ONLY_EXISTING; no training launched')
                    return 0
                print('MANUAL_PREPARE_CONTINUATION; new four-worker smoke attempt', flush=True)
            else:
                if any(p.name != 'controller.lock' for p in output.iterdir()):
                    raise ValueError('Output directory is nonempty without a plan; choose a NEW directory')
            require_idle_gpus()
            gpus = gpu_identity()
            receipt_sha = verify_data(plan, full_hash=True)
            if not current_plan.exists():
                atomic_json(current_plan, plan)
            attempt = smoke_all(args, plan)
            atomic_json(output / 'prelaunch_gate.json',
                        gate_payload(plan, attempt, gpus, receipt_sha))
        print('PREPARED_ONLY: full input audit + four GPU smoke passes; no full training launched')
        return 0
    saved = read_plan(args)
    validate_gate(args, saved)
    verify_data(saved, full_hash=False)
    if args.check_only:
        missing = []
        repo = Path(saved['repo_dir'])
        adapter, lf, ev, _, _ = setup_source(repo)
        for cell in saved['cells']:
            path = Path(cell['eval_dir']) / 'evaluation.json'
            if not path.is_file():
                missing.append(cell['id'])
                continue
            record = ev.evaluate_checkpoint(
                cell['run_dir'], cell['eval_dir'], cell['device'],
                model_builder=adapter.build_model,
                trigger_builder=lf.build_smooth_lf_trigger
                if cell['method'] == 'smooth_lf_csi' else None)
            if record.get('status') != 'complete' or record.get('checkpoint_epoch') != 49:
                raise ValueError(f'Invalid evaluation: {cell["id"]}')
            print(f'COMPLETE {cell["id"]}: {record["metrics"]}')
        if missing:
            print('INCOMPLETE: ' + ', '.join(missing))
        return int(bool(missing))
    with controller_lock(saved['outdir']):
        require_idle_gpus()
        wave(args, saved, None, '--worker')
        records = [json.loads((Path(cell['eval_dir']) / 'evaluation.json').read_text(encoding='utf-8'))
                   for cell in saved['cells']]
        verify_data(saved, full_hash=True)
        result = dict(profile=PROFILE, plan_sha256=saved['plan_sha256'],
                      status='COMPLETE_EXPLORATORY_FOUR_CELLS', seed=42,
                      n_seeds=1, records=records,
                      no_across_seed_uncertainty=True)
        result_path = Path(saved['outdir']) / 'extra_attack_results.json'
        if result_path.is_file():
            if json.loads(result_path.read_text(encoding='utf-8')) != result:
                raise ValueError('Existing final results differ; preserve old output')
        else:
            atomic_json(result_path, result)
    print('COMPLETE_EXPLORATORY_FOUR_CELLS; separate from frozen T1', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('[extra attacks] ERROR: ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
