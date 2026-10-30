"""Two new BadNets/MM-Fi P1S1 victims, seed42, strictly serial on cuda:0.

Reuse the historical native operator, per-backbone training adapters and
six-dose evaluator. No Clean/A1 victim, multi-seed sweep or other attack is
scheduled. Prepare is a full-content input audit and two disposable GPU smokes.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', PYTHONUTF8='1')
HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_p1s1_badnets_wiflow_metafi_s42_serial_v1'
BACKBONES = ('wiflow', 'metafiplusplus')
PLAN = 'badnets_two_models.plan.json'


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, *, immutable=False):
    from run_mmfi_tables import _atomic_json
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f'Existing binding differs; do not overwrite: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, value)


def sources():
    from vast_seed42_preflight import sha256
    # Bind scientific transitive dependencies too, not only this sidecar.
    return {p.relative_to(HERE).as_posix(): sha256(p)
            for p in sorted(HERE.rglob('*')) if p.is_file()
            and p.suffix in ('.py', '.yaml', '.yml', '.json')
            and not {'tests', '__pycache__'}.intersection(p.relative_to(HERE).parts)}


def adapter(backbone):
    if backbone == 'wiflow':
        import a1_wiflow_adapter as result
    elif backbone == 'metafiplusplus':
        import a1_backbone_adapter as result
    else:
        raise ValueError('Only WiFlow and MetaFi++ are authorized')
    return result


def configuration(home, backbone):
    import run_a1_three_tables as frozen
    from train_backdoor import _resolve_training_config
    cfg = frozen._configuration(Path(home), 'cuda:0', 0, 42, 'badnets')
    cfg = (adapter(backbone).recipe_config(cfg) if backbone == 'wiflow'
           else adapter(backbone).recipe_config(cfg, backbone))
    cfg['badnets_two_models_profile'] = PROFILE
    return _resolve_training_config(cfg)


def build_plan(home, outdir, device='cuda:0'):
    from run_a1_cross_backbones import runtime_versions
    home, outdir = Path(home).resolve(), Path(outdir).resolve()
    if device != 'cuda:0' or os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''):
        raise ValueError('Use physical cuda:0 without CUDA_VISIBLE_DEVICES remapping')
    if (outdir == home or outdir == Path(outdir.anchor)
            or home / 'runs' not in outdir.parents):
        raise ValueError('Outdir must be a distinct directory under DataHome/runs')
    cells = []
    for backbone in BACKBONES:
        folder = outdir / backbone / 'seed42' / 'badnets'
        cells.append(dict(method_key=f'{backbone}__s42__badnets', backbone=backbone,
            key='badnets', label='BadNets-CSI', seed=42, origin='fresh', tables=['T1'],
            dependencies=[], cfg=configuration(home, backbone), run_dir=str(folder),
            ckpt_dir=str(folder), eval_cache=str(folder / 'eval_cache.json'),
            artifact_dir=str(outdir / 'evaluations' / backbone)))
    value = dict(schema=1, profile=PROFILE, data_home=str(home), outdir=str(outdir),
        device=device, max_workers=1, cells=cells, sources=sources(),
        runtime=dict(runtime_versions(), executable=str(Path(sys.executable).resolve()),
                     system=platform.system(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
        contract=dict(dataset='MM-Fi', protocol='protocol1', setting='s1', split_seed=0,
            train=133056, test=33264, seed=42, epochs=50, final_epoch=49,
            rho=.1, poison_count=13305, dose_grid=[0., .2, .4, .6, .8, 1.],
            target_joints=[2, 3], pivot=1, theta_max_deg=40., payload_axis=[0., 0., 1.],
            target_pck_threshold=.2, scale_joints=[5, 12],
            badnets='native white3x3 replacement; opacity1; no external peak projection',
            train_order=list(BACKBONES), clean_training=False, a1_training=False,
            clean_reference_controls='not included; no matching reference was supplied',
            scope='two added single-seed rows, not a complete multi-seed paper table',
            uncertainty='not estimated; no equal-realized-noise-budget claim'))
    value['plan_sha256'] = digest(value)
    return value


def runtime(backbone):
    # Private globals only. Never mutate the historical cross-backbone runner.
    spec = importlib.util.spec_from_file_location('_badnets_cross_runtime',
                                                  HERE / 'run_a1_cross_backbones.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    result.PROFILE = PROFILE
    result.training_engine = adapter(backbone).training_engine
    result.evaluation_engine = adapter(backbone).evaluation_engine
    return result


def require_idle():
    from run_mmfi_p1s1_trojannn import classify_gpu, pending_experiment_conflicts
    report = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=30)
    records, names = [], None
    if os.name == 'nt':
        script = ('[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); '
                  '@(Get-CimInstance Win32_Process | Select-Object '
                  'ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine) | ConvertTo-Json -Compress')
        query = subprocess.run(['powershell', '-NoProfile', '-Command', script],
            capture_output=True, encoding='utf-8', check=True, timeout=30)
        records = json.loads(query.stdout.lstrip('\ufeff'))
        if not isinstance(records, list):
            records = [records]
        names = {int(r['ProcessId']): r['Name'] for r in records}
    conflicts = classify_gpu(report.stdout, names, os.getpid())
    # Ignore only actual ancestors (controller and Windows venv proxies), not
    # arbitrary other Python workers with similar command lines.
    by_pid = {r['ProcessId']: r for r in records}
    ancestors, current = {os.getpid()}, os.getpid()
    while current in by_pid:
        current = by_pid[current]['ParentProcessId']
        if current in ancestors:
            break
        ancestors.add(current)
    others = [r for r in records if r['ProcessId'] not in ancestors]
    conflicts += pending_experiment_conflicts(others, {}, os.getpid())
    if conflicts:
        raise ValueError(f'Other/unknown GPU or pending experiment work active: {conflicts}; '
                         'no process stopped; wait before this exclusive run')


def gpu_signature():
    report = subprocess.run(['nvidia-smi', '--id=0',
        '--query-gpu=uuid,name,memory.total,driver_version', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=30)
    rows = list(csv.reader(io.StringIO(report.stdout)))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError('Cannot identify physical GPU0')
    return [part.strip() for part in rows[0]]


def verify_inputs(plan, *, full_hash=False):
    from vast_seed42_preflight import check_receipt
    receipt = read(Path(plan['outdir']) / 'inputs.json')
    if (receipt.get('data_home') != plan['data_home'] or receipt.get('splits') is None
            or {k: v['n'] for k, v in receipt['splits'].items()} != dict(train=133056, eval=33264)):
        raise ValueError('Foreign/incomplete full-input receipt')
    check_receipt(receipt, full_hash=full_hash)
    return receipt


def binding(plan, cell):
    from vast_seed42_preflight import sha256
    return dict(plan_sha256=plan['plan_sha256'], method_key=cell['method_key'],
                inputs_sha256=sha256(Path(plan['outdir']) / 'inputs.json'), gpu=gpu_signature())


def gate_check(plan):
    verify_inputs(plan)
    for cell in plan['cells']:
        gate = read(Path(cell['run_dir']) / 'gpu_smoke.json')
        if (gate.get('binding') != binding(plan, cell) or gate.get('physical_batch') != 32
                or gate.get('batches') != 2 or gate.get('real_train_input') is not True
                or gate.get('poison_examples') != 8 or gate.get('parameter_updated') is not True
                or gate.get('optimizer_steps') != 2 // cell['cfg']['backbone_accumulation_steps']
                or len(gate.get('losses', [])) != 2
                or not all(math.isfinite(x) for x in gate['losses'])):
            raise ValueError('Missing/foreign B32 CUDA preparation gate')


def smoke(plan, cell):
    import numpy as np
    import torch
    from torch.utils.data import Subset
    from run_mmfi_p3_s3_three_attacks_4090 import paired_data
    from run_piw_hpeli_clean import _setup_execution
    cfg = cell['cfg']
    execution = _setup_execution(cfg['device'])
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    engine = adapter(cell['backbone']).training_engine(cfg)
    base = engine._load_dataset(cfg, 'training')
    pairs = paired_data(base, engine.build_trigger(cfg), cfg)
    if len(pairs) != 133056 or pairs.n_poison != 13305:
        raise ValueError('Full TRAIN/poison count differs')
    indices = sorted(pairs.poison_idx)[:8]
    indices += [i for i in range(len(pairs)) if i not in pairs.poison_idx][:56]
    model = engine.build_model(cfg['model']).to(cfg['device'])
    _, optimizer = engine._build_optimizer(model, cfg, 'mmfi')
    parameter = next(p for p in model.parameters() if p.requires_grad)
    before = parameter.detach().clone()
    losses, steps = [], []
    original_step = optimizer.step
    def counted_step(*args, **kwargs):
        steps.append(1)
        return original_step(*args, **kwargs)
    optimizer.step = counted_step
    for batch in engine._training_loader(Subset(pairs, indices), cfg):
        if tuple(batch['csi'].shape) != (32, 3, 114, 10):
            raise ValueError('Smoke requires physical B32 real TRAIN inputs')
        losses.append(engine._victim_update(model, optimizer,
            batch['csi'].to(cfg['device']), batch['pose'].to(cfg['device'])))
    updated = not torch.equal(before, parameter.detach())
    if (len(losses) != 2 or not all(math.isfinite(x) for x in losses) or not updated
            or len(steps) != 2 // cfg['backbone_accumulation_steps']):
        raise ValueError('CUDA smoke failed; no batch/recipe fallback')
    write(Path(cell['run_dir']) / 'gpu_smoke.json', dict(binding=binding(plan, cell),
        execution=execution, physical_batch=32, batches=2, losses=losses,
        optimizer_steps=len(steps), real_train_input=True, poison_examples=8,
        parameter_updated=updated), immutable=True)
    print(f"GPU_SMOKE_PASS {cell['backbone']}; disposable model, no full training", flush=True)


def validate_checkpoint(blob, cfg):
    from train_backdoor import _config_fingerprint
    if (not isinstance(blob, dict) or type(blob.get('epoch')) is not int
            or not 0 <= blob['epoch'] <= 49 or blob.get('cfg') != cfg
            or blob.get('cfg_fingerprint') != _config_fingerprint(cfg)
            or not isinstance(blob.get('model'), dict) or not blob['model']
            or not isinstance(blob.get('optimizer'), dict)
            or not blob['optimizer'].get('param_groups')
            or not blob['optimizer'].get('state')
            or not isinstance(blob.get('rng'), dict)
            or set(blob['rng']) != {'python', 'numpy', 'torch', 'cuda'}
            or any(blob['rng'][k] is None for k in blob['rng'])
            or not blob['rng']['cuda']
            or not isinstance(blob.get('best_loss'), (int, float))
            or not math.isfinite(blob['best_loss'])):
        raise ValueError('Incomplete/foreign epoch/optimizer/RNG checkpoint; no restart fallback')
    for group in blob['optimizer']['param_groups']:
        if group.get('lr') != cfg['lr'] or group.get('weight_decay') != cfg['weight_decay']:
            raise ValueError('Checkpoint optimizer recipe differs')
        if cfg['optimizer'] == 'sgd':
            if group.get('momentum') != .9 or 'betas' in group:
                raise ValueError('MetaFi++ MM-Fi checkpoint must retain SGD momentum .9')
        elif tuple(group.get('betas', ())) != (.9, .999) or group.get('eps') != 1e-8:
            raise ValueError('WiFlow AdamW checkpoint recipe differs')


def run_cell(plan, cell):
    import torch
    from run_piw_hpeli_clean import _setup_execution
    verify_inputs(plan)
    expected = binding(plan, cell)
    execution = _setup_execution('cuda:0')
    if execution != read(Path(cell['run_dir']) / 'gpu_smoke.json')['execution']:
        raise ValueError('CUDA execution differs from Prepare; no automatic re-prepare')
    folder = Path(cell['run_dir'])
    audit = folder / 'execution_binding.json'
    owned = dict(binding=expected, execution=execution)
    if (folder / 'checkpoint.pt').exists():
        if not audit.exists() or read(audit) != owned:
            raise ValueError('Checkpoint lacks its original execution/input/source binding')
        validate_checkpoint(torch.load(folder / 'checkpoint.pt', map_location='cpu',
                                       weights_only=False), cell['cfg'])
    write(audit, owned, immutable=True)
    result = runtime(cell['backbone']).execute_cell(cell, plan['sources'])
    verify_inputs(plan, full_hash=True)
    if binding(plan, cell) != expected:
        raise ValueError('Plan/inputs/GPU changed while running')
    write(folder / 'complete.json', dict(binding=owned,
        evaluation_sha256=result['record_sha256'], status='complete'), immutable=True)


def child(plan, cell, work):
    root = Path(plan['outdir']); logs = root / 'logs'; logs.mkdir(exist_ok=True)
    path = logs / f"{work}.{cell['backbone']}.{time.time_ns()}.log"
    print(f"[{work}] {cell['backbone']}: {path}", flush=True)
    command = [sys.executable, '-u', str(Path(__file__).resolve()),
               '--data-home', plan['data_home'], '--outdir', plan['outdir'],
               '--mode', 'worker', '--_backbone', cell['backbone'], '--_work', work]
    with path.open('x', encoding='utf-8') as stream:
        code = subprocess.run(command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f"{cell['backbone']} {work} failed; no retry/later job: {path}")


def completed(plan, cell):
    marker = Path(cell['run_dir']) / 'complete.json'
    if not marker.exists():
        return None
    owned = read(Path(cell['run_dir']) / 'execution_binding.json')
    gate = read(Path(cell['run_dir']) / 'gpu_smoke.json')
    from vast_seed42_preflight import sha256
    expected = dict(plan_sha256=plan['plan_sha256'], method_key=cell['method_key'],
                    inputs_sha256=sha256(Path(plan['outdir']) / 'inputs.json'),
                    gpu=gate['binding']['gpu'])
    if owned != dict(binding=expected, execution=gate['execution']):
        raise ValueError('Completed run belongs to a different plan/input/CUDA context')
    record = runtime(cell['backbone']).completed(cell, plan['sources'])
    if (record is None or read(marker) != dict(binding=owned,
            evaluation_sha256=record['record_sha256'], status='complete')):
        raise ValueError('Completed evaluation binding is invalid')
    return record


def export(plan):
    verify_inputs(plan)
    rows = []
    for cell in plan['cells']:
        record = completed(plan, cell)
        rows.append(dict(backbone=cell['backbone'], status='complete' if record else 'pending',
                         metrics=record['metrics'] if record else None))
    value = dict(profile=PROFILE, plan_sha256=plan['plan_sha256'], rows=rows,
                 clean_reference_controls='not included', uncertainty='single seed42')
    write(Path(plan['outdir']) / 'badnets_results.json', value)
    table = ['# MM-Fi P1S1 / BadNets-CSI / seed42', '',
             '| Victim | Status | T-MPJPE mm | Target-PCK % | DCE mm | Clean MPJPE mm | Clean PA mm |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        metrics = row['metrics']
        numbers = ([f'{metrics[key]:.2f}' for key in ('t1_mpjpe_mm', 'target_pck_pct',
                    'mean_dce_mm', 'clean_mpjpe_mm', 'clean_pampjpe_mm')]
                   if metrics else ['--'] * 5)
        table.append('| ' + ' | '.join([row['backbone'], row['status'], *numbers]) + ' |')
    table += ['', 'Single seed; matching clean-trained + trigger controls not included.',
              'DCE: mean over five positive doses. Target-PCK is not ASR.', '']
    from eval_a1_target_pck import _atomic_bytes
    _atomic_bytes(Path(plan['outdir']) / 'badnets_results.md', '\n'.join(table).encode('utf-8'))
    print('\n'.join(table))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0'])
    parser.add_argument('--mode', choices=['prepare', 'check', 'launch', 'status', 'export', 'worker'], required=True)
    parser.add_argument('--_backbone', choices=BACKBONES)
    parser.add_argument('--_work', choices=['smoke', 'run'])
    args = parser.parse_args(argv)
    plan = build_plan(args.data_home, args.outdir, args.device)
    root = Path(plan['outdir']); path = root / PLAN
    if args.mode == 'status':
        if path.exists() and read(path) != plan:
            raise ValueError('Source/runtime/config changed; use the original checkout/environment')
        for cell in plan['cells']:
            folder = Path(cell['run_dir'])
            print(f"{cell['backbone']}: checkpoint={(folder/'checkpoint.pt').exists()}, "
                  f"completion_marker={(folder/'complete.json').exists()} (existence only)")
        # PowerShell Tee-Object uses UTF-16 on PS5.1; cell logs below are UTF-8.
        logs = sorted((p for p in (root / 'logs').glob('*.log')
                       if p.name.startswith(('run.', 'smoke.'))), key=lambda p: p.stat().st_mtime)
        for log in logs[-2:]:
            print(f'LOG={log}')
            with log.open(encoding='utf-8', errors='replace') as handle:
                from collections import deque
                print(''.join(deque(handle, maxlen=12)))
        return
    if args.mode != 'prepare' and (not path.exists() or read(path) != plan):
        raise ValueError('Missing/mismatched immutable plan; Prepare first in the original checkout')
    from run_mmfi_tables import _cell_lock
    if args.mode == 'export':
        with _cell_lock(root):
            return export(plan)
    if args.mode == 'worker':
        if not args._backbone or not args._work:
            raise ValueError('Worker requires backbone/work')
        cell = next(c for c in plan['cells'] if c['backbone'] == args._backbone)
        # Device lock lives outside the run, covering different output folders.
        with _cell_lock(Path(plan['data_home']) / 'runs/.mmfi_badnets_gpu0_worker'):
            require_idle()
            gate_check(plan) if args._work == 'run' else verify_inputs(plan)
            (smoke if args._work == 'smoke' else run_cell)(plan, cell)
        return
    with _cell_lock(root):
        require_idle()
        if args.mode == 'prepare':
            write(path, plan, immutable=True)
            input_path = root / 'inputs.json'
            if input_path.exists():
                verify_inputs(plan)
            else:
                from vast_seed42_preflight import inspect
                write(input_path, inspect(plan['data_home']), immutable=True)
            for cell in plan['cells']:
                if not (Path(cell['run_dir']) / 'gpu_smoke.json').exists():
                    child(plan, cell, 'smoke')
            gate_check(plan)
            print('PREPARED_ONLY: full P1S1 audit + two B32 CUDA smokes; no full training', flush=True)
            return
        gate_check(plan)
        if args.mode == 'check':
            print('VERIFIED_PRELAUNCH_GATE; no training launched', flush=True)
            return
        for cell in plan['cells']:
            if completed(plan, cell) is None:
                require_idle()
                child(plan, cell, 'run')  # blocking: never two workers overlap
                if completed(plan, cell) is None:
                    raise ValueError('Worker exited without authenticated completion; no later job')
        export(plan)
        print('COMPLETE 2/2 BadNets victims; max one worker on GPU0', flush=True)


if __name__ == '__main__':
    main()
