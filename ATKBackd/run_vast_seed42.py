"""Eight-slot DAG controller; independent victims, never DDP or shortened runs.

Default: retrain MetaFi++ and WiFlow from scratch, then four repairs and a
paired No-defense evaluation per backbone. Historical HPE-Li is read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROFILE = 'vast_mmfi_seed42_four_tables_v1'
MANIFEST = 'vast_seed42.resolved.json'
TEST_GATE = 'prelaunch_tests.json'
DEFAULT_BACKBONES = ('metafiplusplus', 'wiflow')


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _imports():
    import vast_seed42_attack as attack
    import vast_seed42_defense as defense
    return attack, defense


def source_hashes():
    from vast_seed42_preflight import sha256
    # Freeze the complete shipped executable package, not only the new launcher.
    return {p.relative_to(HERE).as_posix(): sha256(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.json')
            and 'tests' not in p.relative_to(HERE).parts
            and '__pycache__' not in p.parts}


def make_plan(data_home, outdir, devices, jobs_per_gpu, backbones, defense_jobs_per_gpu=None,
              defense_activation_offload='none', experiment_scope='full', anp_host_exclusive=False):
    from vast_seed42_preflight import digest
    from vast_seed42_memory import validate_policy
    validate_policy(defense_activation_offload)
    if type(experiment_scope) is not str or experiment_scope not in ('full','attack-defense'):
        raise ValueError('Experiment scope must be full or attack-defense')
    if defense_activation_offload == 'cpu' and defense_jobs_per_gpu != 1:
        raise ValueError('CPU offload requires explicit exclusive defense capacity')
    if type(anp_host_exclusive) is not bool:
        raise ValueError('ANP host-exclusive policy must be a boolean')
    if anp_host_exclusive and (defense_activation_offload != 'cpu' or defense_jobs_per_gpu != 1):
        raise ValueError('ANP host-exclusive requires CPU offload and exclusive defense capacity')
    if not devices or len(set(devices)) != len(devices) or any(not d.startswith('cuda:') or not d[5:].isdigit() for d in devices):
        raise ValueError('Use distinct explicit CUDA device IDs')
    if type(jobs_per_gpu) is not int or jobs_per_gpu not in (1, 2):
        raise ValueError('This batch permits at most two independent jobs per GPU')
    if defense_jobs_per_gpu is not None and (type(defense_jobs_per_gpu) is not int
                                             or defense_jobs_per_gpu != 1):
        raise ValueError('Explicit defense capacity must be one exclusive job per GPU')
    if not backbones or len(set(backbones)) != len(backbones) or set(backbones) - {'hpeli', *DEFAULT_BACKBONES}:
        raise ValueError('Unknown/duplicate backbone')
    outdir = Path(outdir).resolve()
    data_home = Path(data_home).resolve()
    code = HERE.parent.resolve()
    for root, label in ((outdir, 'Output'), (data_home, 'Data home')):
        if root == code or root in code.parents or code in root.parents:
            raise ValueError(label+' must be outside the executable code tree')
    if outdir == data_home or outdir in data_home.parents:
        raise ValueError('Output must not be the data home or its ancestor')
    for root in (data_home/'datasets', data_home/'actions', data_home/'downloads'):
        if outdir == root or outdir in root.parents or root in outdir.parents:
            raise ValueError('Output must be separate from datasets/actions/downloads')
    attack, defense = _imports()
    scope_options = {'experiment_scope': experiment_scope} if experiment_scope != 'full' else {}
    attacks = attack.build_manifest(data_home, outdir / 'attacks', devices=devices,
                                    backbones=tuple(backbones), **scope_options)
    memory_options = {'activation_offload': 'cpu'} if defense_activation_offload == 'cpu' else {}
    defenses = defense.build_plan(data_home, outdir / 'defenses', outdir / 'attacks', devices=devices,
                                  backbones=tuple(backbones), **memory_options)
    cells = []
    for index, cell in enumerate(attacks['cells']):
        name = cell['method_key']
        cells.append(dict(id='attack/' + name, kind='attack', cell=name,
            dependencies=['attack/' + d for d in cell.get('dependencies', [])],
            folder=cell['run_dir'], artifact_dir=cell['artifact_dir'],
            preferred_device=devices[index % len(devices)]))
    for index, cell in enumerate(defenses['cells']):
        name = cell.get('cell_id', cell.get('method_key'))
        backbone = cell['backbone']
        folder = cell.get('folder', cell.get('run_dir'))
        cells.append(dict(id='defense/' + name, kind='defense', cell=name,
            dependencies=['attack/' + backbone + '__s42__clean', 'attack/' + backbone + '__s42__a1'],
            folder=folder, artifact_dir=folder,
            preferred_device=cell.get('device', devices[index % len(devices)])))
    plan = dict(schema=1, profile=PROFILE, seed=42, backbones=list(backbones),
        data_home=str(Path(data_home).resolve()), outdir=str(outdir), devices=list(devices),
        jobs_per_gpu=jobs_per_gpu, sources=source_hashes(), cells=cells,
        attacks=attacks, defenses=defenses,
        policy=dict(fresh_training=list(backbones), victim_epochs=50, selection='fixed final49; no TEST selection',
            defense_methods=['none', 'ft', 'anp', 'nad', 'clp'],
            historical_hpeli='read-only separate provenance; not same-environment fresh training',
            no_automatic_retry=True, no_batch_fallback=True, no_amp=True,
            seed_uncertainty='not estimated; n=1', excludes=['IAD', 'LIRA', 'GraphPose-Fi', 'second dataset', 'T4 stealth']))
    if defense_jobs_per_gpu is not None:
        plan['defense_jobs_per_gpu'] = defense_jobs_per_gpu
        plan['gpu_sharing'] = 'exclusive-defense'
    if defense_activation_offload == 'cpu':
        plan['defense_activation_offload'] = 'cpu'
    if anp_host_exclusive:
        plan['anp_host_exclusive'] = True
    if experiment_scope != 'full':
        plan['experiment_scope'] = experiment_scope
        plan['policy']['fresh_tables'] = ['T1','T5']
        plan['policy']['excludes'].append('T2/T3 fresh-backbone victims; HPE-Li historical only')
    # Worker plans carry timestamps, excluded from the controller scientific hash.
    hashable = {k: v for k, v in plan.items() if k not in ('attacks', 'defenses')}
    plan['plan_sha256'] = digest(hashable)
    return plan


def load_plan(outdir):
    outdir = Path(outdir).resolve()
    plan = json.loads((outdir / MANIFEST).read_text(encoding='utf-8'))
    if Path(plan['outdir']).resolve() != outdir:
        raise ValueError('Manifest is not at its original exact output path')
    rebuilt = make_plan(plan['data_home'], plan['outdir'], plan['devices'], plan['jobs_per_gpu'],
                        plan['backbones'], plan.get('defense_jobs_per_gpu'),
                        plan.get('defense_activation_offload', 'none'), plan.get('experiment_scope','full'),
                        plan.get('anp_host_exclusive', False))
    if plan.get('profile') != PROFILE or plan != rebuilt:
        raise ValueError('Runtime/source/root/plan changed; do not edit a bound manifest to resume')
    # Worker manifests enforce their own runtime and exact recipe bindings.
    return plan


def worker_command(plan, task, device):
    if task['kind'] == 'attack':
        return [sys.executable, '-u', str(HERE / 'vast_seed42_attack.py'),
                '--outdir', str(Path(plan['outdir']) / 'attacks'), '--device', device, '--cell', task['cell']]
    return [sys.executable, '-u', str(HERE / 'vast_seed42_defense.py'),
            '--data-home', plan['data_home'], '--outdir', str(Path(plan['outdir']) / 'defenses'),
            '--attack-outdir', str(Path(plan['outdir']) / 'attacks'), '--device', device,
            '--backbones', *plan['backbones'], '--cell', task['cell'],
            *(['--activation-offload', 'cpu'] if plan.get('defense_activation_offload') == 'cpu' else [])]


def completed(plan, task):
    attack, defense = _imports()
    if task['kind'] == 'attack':
        cell = next(c for c in plan['attacks']['cells'] if c['method_key'] == task['cell'])
        return attack.completed(cell, plan['attacks']['sources'])
    cell = next(c for c in plan['defenses']['cells'] if c.get('cell_id', c.get('method_key')) == task['cell'])
    return defense.completed(cell)


def run_queue(tasks, devices, jobs_per_gpu, outdir, command, verify, *, poll_seconds=1.,
              defense_jobs_per_gpu=None, anp_host_exclusive=False):
    """Bounded FIFO DAG. On failure, stop new launches and drain owned peers.

    No stale-lock deletion, subprocess kill, automatic retry, or silently
    discarded failed row. Function accepts injected commands for OS-level tests.
    """
    tasks = list(tasks)
    if (type(jobs_per_gpu) is not int or jobs_per_gpu not in (1, 2) or not devices
            or len(set(devices)) != len(devices) or poll_seconds < 0):
        raise ValueError('Invalid bounded queue capacity/devices/poll interval')
    if defense_jobs_per_gpu is not None and (type(defense_jobs_per_gpu) is not int
                                             or defense_jobs_per_gpu != 1):
        raise ValueError('Explicit defense capacity must be one exclusive job per GPU')
    if defense_jobs_per_gpu is not None and any(t.get('kind') not in ('attack', 'defense') for t in tasks):
        raise ValueError('Exclusive-defense scheduling requires an explicit attack/defense task kind')
    if type(anp_host_exclusive) is not bool or (anp_host_exclusive and defense_jobs_per_gpu != 1):
        raise ValueError('ANP host-exclusive scheduling requires explicit exclusive defense capacity')
    def is_anp(task):
        return task.get('kind') == 'defense' and task.get('cell', '').endswith('__anp')
    if anp_host_exclusive and any(t.get('kind') == 'defense' and (
            not isinstance(t.get('cell'), str)
            or t['cell'].rsplit('__', 1)[-1] not in ('none','ft','anp','nad','clp')) for t in tasks):
        raise ValueError('Host-exclusive defense scheduling requires explicit cell identities')
    by_id = {t['id']: t for t in tasks}
    if len(by_id) != len(tasks) or any(set(t['dependencies']) - set(by_id) for t in tasks):
        raise ValueError('Invalid DAG identities/dependencies')
    if any(task.get('preferred_device', devices[0]) not in devices for task in tasks):
        raise ValueError('Task preferred_device must be in the frozen device list')
    reachable = set()
    while True:
        added = {task['id'] for task in tasks
                 if set(task['dependencies']) <= reachable} - reachable
        if not added:
            break
        reachable.update(added)
    if len(reachable) != len(tasks):
        raise ValueError('Dependency cycle; nothing launched')
    pending, done, running, errors = list(tasks), set(), {}, []
    slots = [(device, index + 1) for device in devices for index in range(jobs_per_gpu)]
    def fits(task, slot):
        if anp_host_exclusive:
            if any(is_anp(peer) for _, peer, _, _ in running.values()):
                return False
            # CPU offload shares host/cgroup RAM across GPUs. ANP owns the
            # entire host while its two full backward graphs are resident.
            if is_anp(task) and running:
                return False
        if defense_jobs_per_gpu is None:
            return True
        residents = [peer for resident_slot, (_, peer, _, _) in running.items()
                     if resident_slot[0] == slot[0]]
        if task['kind'] == 'defense':
            # A defense owns the entire device, not just one of its victim slots.
            return slot[1] == 1 and not residents
        return not any(peer['kind'] == 'defense' for peer in residents)
    snapshot = Path(outdir) / 'controller_status.json'
    interrupted = False
    while pending or running:
        try:
            for slot, (child, task, stream, logfile) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                del running[slot]
                if code != 0:
                    errors.append(f"{task['id']}: exit{code}, log={logfile}")
                else:
                    try:
                        if not verify(task):
                            raise ValueError('Exit0 without authenticated completion')
                        done.add(task['id'])
                        print(f"[Vast] COMPLETE {task['id']}", flush=True)
                    except Exception as exc:
                        errors.append(f"{task['id']}: completion audit failed: {exc}")
            if not errors and not interrupted:
                for slot in slots:
                    if slot in running:
                        continue
                    task = next((t for t in pending if set(t['dependencies']) <= done
                                 and t.get('preferred_device', slot[0]) == slot[0]
                                 and fits(t, slot)), None)
                    if task is None:
                        continue
                    pending.remove(task)
                    if verify(task):
                        done.add(task['id'])
                        print(f"[Vast] VERIFIED_EXISTING {task['id']}", flush=True)
                        continue
                    folder = Path(task['folder'])
                    folder.mkdir(parents=True, exist_ok=True)
                    logfile = folder / 'console.log'
                    stream = logfile.open('a', encoding='utf-8', buffering=1)
                    try:
                        env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                                   MKL_THREADING_LAYER='SEQUENTIAL')
                        child = subprocess.Popen(command(task, slot[0]), stdout=stream, stderr=subprocess.STDOUT, env=env)
                    except BaseException:
                        stream.close()
                        raise
                    running[slot] = child, task, stream, logfile
                    print(f"[Vast] {task['id']} -> {slot[0]} slot{slot[1]} PID={child.pid}; {logfile}", flush=True)
            atomic_json(snapshot, dict(updated_utc=utc(), done=sorted(done), errors=errors,
                pending=[t['id'] for t in pending], interrupted=interrupted,
                running=[dict(id=t['id'], device=s[0], slot=s[1], pid=p.pid, log=str(l))
                         for s, (p, t, _, l) in running.items()]))
            if not running and pending and not errors and not interrupted:
                if not any(set(t['dependencies']) <= done for t in pending):
                    raise ValueError('Dependency cycle; nothing launched')
            if errors and not running:
                raise RuntimeError('Worker failure; pending work NOT launched. ' + '; '.join(errors))
            if interrupted and not running:
                raise KeyboardInterrupt
            if running:
                time.sleep(poll_seconds)
        except (KeyboardInterrupt, Exception) as exc:
            if not running:
                raise
            interrupted = True
            errors.append(f'Controller {type(exc).__name__}: {exc}')
            print('[Vast] No further launches; waiting for owned workers to exit. Do not kill the controller to resume.', flush=True)
    return done


def status(plan, verify=False):
    rows = []
    for task in plan['cells']:
        record = Path(task['artifact_dir']) / 'evaluation.json'
        state = 'record_present_unverified' if record.is_file() else 'pending_or_training'
        if verify:
            state = 'verified_complete' if completed(plan, task) else 'incomplete'
        rows.append(dict(id=task['id'], state=state, log=str(Path(task['folder']) / 'console.log')))
    snapshot = Path(plan['outdir']) / 'controller_status.json'
    return dict(tasks=rows, controller=json.loads(snapshot.read_text()) if snapshot.is_file() else None)


def validate_memory_smoke(plan, path):
    """Require actual concurrent-model probes; no CUDA/data in this receipt check."""
    script = HERE.parent / 'remote_linux' / 'vast_seed42' / 'model_smoke.py'
    spec = importlib.util.spec_from_file_location('_vast_memory_gate', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    profile = ({'defense_jobs_per_gpu': plan['defense_jobs_per_gpu']}
               if 'defense_jobs_per_gpu' in plan else {})
    if plan.get('defense_activation_offload') == 'cpu':
        profile['activation_offload'] = 'cpu'
    if plan.get('anp_host_exclusive') is True:
        profile['anp_host_exclusive'] = True
    module.validate_receipt(report, [int(device.split(':')[1]) for device in plan['devices']],
                            plan['jobs_per_gpu'], plan['backbones'], **profile)
    return report['receipt_sha256']


def dry_plan(plan):
    """Pure plan/config test. No data/GPU, output writes, or training."""
    print(f"DRY RUN: {len(plan['attacks']['cells'])} fresh50epoch victims + {len(plan['defenses']['cells'])} defense evaluations; "
          f"{len(plan['devices']) * plan['jobs_per_gpu']} slots; seed42. No writes/data/CUDA/training.")
    if plan.get('defense_jobs_per_gpu') == 1:
        print(f"GPU sharing: victims up to {plan['jobs_per_gpu']}/GPU; each defense EXCLUSIVE 1/GPU; "
              'no co-resident victim/defense; physical batches unchanged.')
    if plan.get('defense_activation_offload') == 'cpu':
        print('Defense saved tensors: CPU offload, no recompute/AMP/microbatch; actual GPU/RAM/parity smoke required.')
    if plan.get('anp_host_exclusive') is True:
        print('ANP: one process on the ENTIRE host; all other victim/defense jobs wait, including other GPUs.')
    print(json.dumps(dict(profile=PROFILE, plan_sha256=plan['plan_sha256'], policy=plan['policy'],
                         **({k: plan[k] for k in ('defense_jobs_per_gpu', 'gpu_sharing')}
                            if 'defense_jobs_per_gpu' in plan else {})), indent=2))


def test_gate_record(plan, memory_sha, dataset_receipt):
    from vast_seed42_preflight import digest
    result = dict(schema=1, profile=PROFILE, status='PRELAUNCH_TESTS_PASSED_NOT_TRAINED',
        plan_sha256=plan['plan_sha256'], memory_receipt_sha256=memory_sha,
        dataset_binding_sha256=dataset_receipt['binding_sha256'],
        tests=dict(dry_run=True, concurrent_actual_model_smoke=True, full_content_preflight=True),
        full_training_launched=False, explicit_next_command='--launch')
    result['receipt_sha256'] = digest(result)
    return result


def validate_test_gate(plan, memory_sha, dataset_receipt):
    path = Path(plan['outdir']) / TEST_GATE
    if not path.is_file():
        raise FileNotFoundError('Mandatory dry/smoke/preflight tests have not passed. '
                                'Run start.sh --prepare-only FIRST; no full training launched.')
    actual = json.loads(path.read_text(encoding='utf-8'))
    if actual != test_gate_record(plan, memory_sha, dataset_receipt):
        raise ValueError('Prelaunch test gate differs from this plan/data/model-smoke receipt; '
                         'do not bypass it or edit the receipt. Run tests for the intended new plan.')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--data-home')
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--jobs-per-gpu', type=int, default=2)
    parser.add_argument('--defense-jobs-per-gpu', type=int, choices=(1,),
                        help='Explicit memory profile: each defense exclusively owns its GPU; victim capacity unchanged')
    parser.add_argument('--defense-activation-offload', choices=('none', 'cpu'), default='none')
    parser.add_argument('--experiment-scope', choices=('full','attack-defense'), default='full')
    parser.add_argument('--anp-host-exclusive', action='store_true',
                        help='Serialize CPU-offloaded ANP across the entire host; no co-resident jobs on other GPUs')
    parser.add_argument('--backbones', nargs='+', default=list(DEFAULT_BACKBONES))
    parser.add_argument('--smoke-report', help='Concurrent actual-model GPU smoke receipt; default DATAHOME/model_smoke.json')
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ('dry-run', 'preflight', 'launch', 'status', 'check-only', 'export-only', 'check-test-gate'):
        mode.add_argument('--' + flag, action='store_true')
    args = parser.parse_args(argv)
    output = Path(args.outdir).resolve()
    if args.status or args.check_only or args.export_only or args.check_test_gate:
        plan = load_plan(output)
        if args.check_only or args.export_only or args.check_test_gate:
            from vast_seed42_preflight import check_receipt
            receipt = json.loads((output / 'dataset_content.json').read_text(encoding='utf-8'))
            check_receipt(receipt, full_hash=not args.check_test_gate)
        if args.check_test_gate:
            memory_sha = validate_memory_smoke(plan, args.smoke_report or Path(plan['data_home']) / 'model_smoke.json')
            validate_test_gate(plan, memory_sha, receipt)
            print('VERIFIED_PRELAUNCH_TEST_GATE; read-only, no full-data training launched')
            return 0
        if args.export_only:
            from vast_seed42_report import export
            export(plan)
        else:
            result = status(plan, verify=args.check_only)
            print(json.dumps(result, indent=2))
            if args.check_only and any(r['state'] != 'verified_complete' for r in result['tasks']):
                return 2
        return 0
    if not args.data_home:
        parser.error('--data-home is required for planning/preflight/launch')
    plan = make_plan(args.data_home, output, args.devices, args.jobs_per_gpu, args.backbones,
                     args.defense_jobs_per_gpu, args.defense_activation_offload, args.experiment_scope,
                     args.anp_host_exclusive)
    if args.dry_run:
        dry_plan(plan)
        return 0
    if args.preflight:
        dry_plan(plan)
    if args.launch and not (output / TEST_GATE).is_file():
        raise FileNotFoundError('Missing mandatory prelaunch test gate. Run start.sh --prepare-only FIRST; nothing launched.')
    memory_receipt = validate_memory_smoke(plan, args.smoke_report or Path(plan['data_home']) / 'model_smoke.json')
    if args.launch:
        if load_plan(output) != plan:
            raise ValueError('Launch does not match the separately tested plan')
        initial_receipt = json.loads((output / 'dataset_content.json').read_text(encoding='utf-8'))
        validate_test_gate(plan, memory_receipt, initial_receipt)
    from run_mmfi_tables import _cell_lock
    from vast_seed42_preflight import inspect, check_receipt
    with _cell_lock(output):
        if (output / MANIFEST).is_file():
            saved = load_plan(output)
            if saved['plan_sha256'] != plan['plan_sha256']:
                raise ValueError('Different launch options for an existing plan')
            plan = saved
        receipt_path = output / 'dataset_content.json'
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            check_receipt(receipt, full_hash=True)
        else:
            receipt = inspect(plan['data_home'])
            atomic_json(receipt_path, receipt)
        attack, _ = _imports()
        if not (output / MANIFEST).is_file():
            attack.write_manifest(plan['attacks'])
            atomic_json(output / MANIFEST, plan)
        if args.preflight:
            gate = test_gate_record(plan, memory_receipt, receipt)
            path = output / TEST_GATE
            if path.exists() and json.loads(path.read_text(encoding='utf-8')) != gate:
                raise ValueError('An existing prelaunch test gate differs; preserve it and inspect the changed plan')
            if not path.exists():
                atomic_json(path, gate)
            print('PRELAUNCH_TESTS_PASSED_NOT_TRAINED: dry-run, GPU smoke and full content verified. STOP; inspect logs before --launch.')
            return 0
        validate_test_gate(plan, memory_receipt, receipt)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA; will not fall back to CPU')
        for device in plan['devices']:
            torch.cuda.get_device_properties(device)
        atomic_json(output / 'memory_gate_used.json', dict(receipt_sha256=memory_receipt,
            source_report=str(Path(args.smoke_report or Path(plan['data_home']) / 'model_smoke.json').resolve()),
            limitation='resident model/state overlap; transient activation maxima and full-data OOM are not guaranteed'))
        run_queue(plan['cells'], plan['devices'], plan['jobs_per_gpu'], output,
                  lambda task, device: worker_command(plan, task, device), lambda task: completed(plan, task),
                  **({'defense_jobs_per_gpu': plan['defense_jobs_per_gpu']}
                     if 'defense_jobs_per_gpu' in plan else {}),
                  **({'anp_host_exclusive': True} if plan.get('anp_host_exclusive') is True else {}))
        check_receipt(receipt, full_hash=True)
        from vast_seed42_report import export
        export(plan)
        tables = 'T1/T5' if plan.get('experiment_scope') == 'attack-defense' else 'four tables'
        print(f'VAST_BATCH_COMPLETE_ONE_SEED; raw records and {tables} exported; no across-seed SD.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'[Vast] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
