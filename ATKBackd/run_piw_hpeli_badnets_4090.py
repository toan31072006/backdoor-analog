"""One BadNets-CSI / HPE-Li / PiW3D seed42 victim; exclusive Van4090.

Reuse the frozen v3 reference and the existing native-suite train/eval functions
in a PRIVATE module instance. Historical modules, gates and results are never
modified. Preflight is CPU-only; Prepare adds an actual poisoned B32 CUDA smoke
and stops. Launch is a separate foreground invocation, never a scheduled task.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROFILE = 'piw_hpeli_badnets_s42_exclusive4090_v1'
MANIFEST = 'badnets.plan.json'
METHOD = 'badnets'
DOSES = [0., .2, .4, .6, .8, 1.]
OPERATOR = dict(name='BadNets-CSI adapted', patch_subcarriers=3,
    patch_packets=3, patch_start=[27, 17], antennas=[0, 1, 2],
    pattern='white', opacity=1., groups=3, phase='unchanged',
    domain='normalized amplitude; no second normalization',
    dose='patch opacity=dose; shared attack_eps is NOT a peak bound')


def engine():
    """Do not mutate the five-method historical suite or its imported adapter."""
    spec = importlib.util.spec_from_file_location('_piw_badnets_private_engine',
        HERE / 'run_piw_hpeli_attack_suite.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    result.PROFILE = PROFILE
    result.METHODS = (METHOD,)
    result.GroupedTrigger = BadNetsGroups
    result.PoisonedPairs = BadNetsPairs
    original_checkpoint = result.checkpoint

    def checked_checkpoint(plan, method, *, final=False):
        blob = original_checkpoint(plan, method, final=final)
        validate_checkpoint(blob, plan)
        return blob

    result.checkpoint = checked_checkpoint
    return result


def validate_checkpoint(blob, plan):
    """Keep AdamW moments and exact-resume RNG; never fall back to scratch."""
    import torch
    cfg = plan['cfg']
    if blob.get('trigger_state') != OPERATOR:
        raise ValueError('Checkpoint frozen BadNets operator changed')
    if (not isinstance(blob.get('model'), dict) or not blob['model']
            or not isinstance(blob.get('execution'), dict) or not blob['execution']
            or type(blob.get('train_loss')) not in (int, float)
            or not math.isfinite(blob['train_loss'])):
        raise ValueError('Incomplete BadNets checkpoint; no restart fallback')
    rng = blob.get('rng', {})
    if (not isinstance(rng, dict) or set(rng) != {'python', 'numpy', 'torch', 'cuda'}
            or any(rng[k] is None for k in rng)):
        raise ValueError('Missing exact-resume RNG')
    for key in ('torch', 'cuda'):
        if not torch.is_tensor(rng[key]) or rng[key].dtype != torch.uint8 or rng[key].ndim != 1 or not rng[key].numel():
            raise ValueError('Missing Torch/CUDA RNG tensor')
    opt = blob.get('optimizer', {})
    if (not isinstance(opt, dict) or not isinstance(opt.get('state'), dict)
            or not opt['state'] or not isinstance(opt.get('param_groups'), list)
            or len(opt['param_groups']) != 1):
        raise ValueError('Missing complete AdamW optimizer state')
    group = opt['param_groups'][0]
    if (group.get('lr') != cfg['lr'] or group.get('weight_decay') != cfg['weight_decay']
            or tuple(group.get('betas', ())) != tuple(cfg['betas'])
            or group.get('eps') != cfg['eps'] or 'momentum' in group):
        raise ValueError('Checkpoint AdamW recipe differs')
    params = group.get('params', [])
    if not params or len(params) != len(set(params)) or set(opt['state']) != set(params):
        raise ValueError('Incomplete AdamW parameter moments')
    for state in opt['state'].values():
        if not {'step', 'exp_avg', 'exp_avg_sq'}.issubset(state):
            raise ValueError('Incomplete AdamW moments')
        a, b, step = state['exp_avg'], state['exp_avg_sq'], state['step']
        if (not torch.is_tensor(a) or not torch.is_tensor(b) or a.shape != b.shape
                or not torch.isfinite(a).all() or not torch.isfinite(b).all()
                or (b < 0).any() or float(step) <= 0 or not math.isfinite(float(step))):
            raise ValueError('Invalid AdamW moments')


class BadNetsGroups:
    """Native BackdoorBench white3x3 patch on each physical amplitude group."""
    method = METHOD

    def __init__(self, method=METHOD, train=None):
        if method != METHOD:
            raise ValueError('Only BadNets is scheduled')
        from attack.traditional import BadNetsTrigger
        self.native = [BadNetsTrigger(n_ant=3, n_sub=30, n_pkt=20,
            seed=42, patch_subcarriers=3, patch_packets=3, patch_opacity=1.)
            for _ in range(3)]

    def inject(self, csi, dose, eps=.185):
        import numpy as np
        from piw_native_attack_adapter import GroupedTrigger
        if eps != .185:
            raise ValueError('Frozen shared interface epsilon changed')
        output = GroupedTrigger._input(csi, dose)
        if dose == 0:
            return output
        for group, native in enumerate(self.native):
            lo = 30 * group
            output[:, lo:lo + 30] = native.inject(output[:, lo:lo + 30], dose, eps=eps)
        if not np.array_equal(output[:, 90:], np.asarray(csi, dtype=np.float32)[:, 90:]):
            raise ValueError('BadNets changed phase')
        return output

    def state_dict(self):
        return copy.deepcopy(OPERATOR)

    def load_state_dict(self, state):
        if state != OPERATOR:
            raise ValueError('Frozen BadNets operator changed')

    def fixed_keys(self):
        from run_piw_hpeli_clean import object_sha
        return [object_sha(OPERATOR)] * 3


class BadNetsPairs:
    """Same v3 poison indices AND paired doses; no clean-label covers."""
    def __init__(self, base, trigger, poison):
        self.base, self.trigger, self.items = base, trigger, base.items
        samples = poison['samples']
        selected = {}
        for row in samples:
            i, d = row['index'], row['dose']
            if (type(i) is not int or not 0 <= i < len(base) or i in selected
                    or type(d) not in (int, float) or not math.isfinite(d)
                    or not .2 <= d <= 1. or base.items[i]['name'] != row['name']):
                raise ValueError('PiW poison identity/dose mismatch')
            selected[i] = d
        if len(selected) != math.floor(.1 * len(base)):
            raise ValueError('Expected floor(0.1 * TRAIN) poison examples')
        self.selected, self.covers = selected, set()
        from run_piw_hpeli_clean import object_sha
        self.record = dict(method=METHOD, n_total=len(base), n_poison=len(selected),
            n_cover=0, cover_ratio=0., poison_sha256=poison['poison_plan_sha256'],
            cover_indices_sha256=object_sha([]), target='piw_body_hip_axis_v2',
            operator_keys=trigger.fixed_keys(), phase_policy='unchanged')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        import numpy as np
        from piw_v2_adapter import target_pose
        row = self.base[index]
        csi, pose = row['csi'], row['pose']
        if index in self.selected:
            d = self.selected[index]
            csi = self.trigger.inject(csi, d)
            pose = target_pose(pose, d).astype(np.float32)
        return dict(csi=csi, pose=pose)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, immutable=False):
    import run_piw_hpeli_clean as base
    path = Path(path)
    if path.exists() and immutable:
        if read(path) != value:
            raise ValueError(f'Existing immutable binding differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(path, value)


def sources():
    import run_mmfi_badnets_two_models_4090 as shared
    import run_piw_hpeli_clean as base
    result = shared.sources()
    result['../remote_tools/Start-PiWHPELiBadNets4090.ps1'] = base.file_sha(
        HERE.parent / 'remote_tools/Start-PiWHPELiBadNets4090.ps1')
    return result


def build_plan(home, output, reference_dir=None):
    import run_piw_hpeli_clean as base
    from piw_v2_adapter import target_contract
    home, output = Path(home).resolve(), Path(output).resolve()
    runs = home / 'runs'
    if output == runs or not output.is_relative_to(runs):
        raise ValueError('Output must be a strict descendant of DataHome/runs')
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        raise ValueError('Use physical GPU0 without CUDA_VISIBLE_DEVICES remapping')
    native = engine()
    root, ref, _ = native.reference(home, reference_dir)
    if output == root or output in root.parents or root in output.parents:
        raise ValueError('Historical v3 reference is read-only; choose separate output')
    cfg = copy.deepcopy(ref['cfg'])
    required = dict(dataset='person-in-wifi-3d', experiment_name='one-person',
        model='hpeli', seed=42, epochs=200, final_epoch=199, batch_size=32,
        num_workers=0, optimizer='adamw', lr=.001, weight_decay=.01,
        betas=[.9, .999], eps=1e-8, amp=False, drop_last=True,
        victim_loss='ordinary MPJPE', target=target_contract(), rho=.1,
        attack_eps=.185, dose_range=[.2, 1.], eval_doses=DOSES)
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError('Reference differs from frozen v3 HPE-Li scientific recipe')
    if cfg['device'] != 'cuda:0':
        raise ValueError('Historical v3 must use the local cuda:0 reference')
    cfg.update(profile=PROFILE, outdir=str(output), jobs_per_gpu=1, roles=[METHOD],
        initialization='independent scratch victim; never initialize from Clean or A1',
        trigger_policy=copy.deepcopy(OPERATOR), selection='fixed final199; no TEST selection',
        status='EXPLORATORY_PIW_BADNETS_ONE_SEED')
    names = (native.v3.MANIFEST, 'evaluation.json', 'predictions.npz', 'poison.json',
             'hpeli/seed42/clean/checkpoint.pt', 'hpeli/seed42/a1/checkpoint.pt')
    plan = dict(schema=1, profile=PROFILE, methods=[METHOD], max_workers=1,
        cfg=cfg, reference=str(root), reference_sha256={n: native.sha(root/n) for n in names},
        inputs_fingerprint=ref['inputs']['fingerprint'],
        poison_plan_sha256=ref['poison']['poison_plan_sha256'],
        operator=copy.deepcopy(OPERATOR), sources=sources(), runtime=base.runtime_versions(),
        policy=dict(clean_training=False, a1_training=False, no_automatic_retry=True,
            no_batch_fallback=True, gpu_sharing=False,
            note='Selected new BadNets row; not a completed multi-seed comparison'))
    plan['sha256'] = base.object_sha(plan)
    return plan


def verify_plan(plan):
    current = build_plan(plan['cfg']['data_home'], plan['cfg']['outdir'], plan['reference'])
    if current != plan:
        raise ValueError('Source/runtime/reference/config/plan changed; do not overwrite historical gates')
    return plan


def cell(plan):
    return Path(plan['cfg']['outdir']) / 'hpeli/seed42/badnets'


def receipt_path(plan, kind):
    return Path(plan['cfg']['outdir'] + f'.{kind}.json')


def receipt(plan, kind):
    value = read(receipt_path(plan, kind))
    if (value.get('status') != 'PASS' or value.get('kind') != kind
            or value.get('plan_sha256') != plan['sha256']
            or value.get('input_fingerprint') != plan['inputs_fingerprint']):
        raise ValueError(f'Missing/mismatched {kind} gate')
    return value


def full_audit(plan):
    native = engine()
    ref, ds = native.datasets(plan)
    current, _ = native.v3.base.audit_inputs(ref['cfg'])
    if current != ref['inputs']:
        raise ValueError('Full PiW TRAIN/TEST contents differ from v3 reference')
    if len(ds['training']) != 27876 or len(ds['validation']) != 2586:
        raise ValueError('Full official PiW splits required: TRAIN27876/TEST2586')
    return ref, ds


def preflight(plan):
    import numpy as np
    ref, ds = full_audit(plan)
    trigger = BadNetsGroups()
    poisoned = BadNetsPairs(ds['training'], trigger, ref['poison'])
    for i in (0, len(ds['training']) // 2):
        x = ds['training'][i]['csi']
        if not np.array_equal(trigger.inject(x, 0), x):
            raise ValueError('BadNets dose0 must be exact identity')
        y = trigger.inject(x, 1.)
        if np.array_equal(x, y) or not np.array_equal(x[:, 90:], y[:, 90:]):
            raise ValueError('Nonempty amplitude-only BadNets required')
    root = Path(plan['cfg']['outdir'])
    if root.exists() and not (root / MANIFEST).exists() and any(
            p.name != 'run.lock' for p in root.iterdir()):
        raise ValueError('Nonempty output without this plan; use a new folder')
    write(root / MANIFEST, plan, immutable=True)
    write(receipt_path(plan, 'preflight'), dict(kind='preflight', status='PASS',
        plan_sha256=plan['sha256'], input_fingerprint=plan['inputs_fingerprint'],
        n_train=27876, n_test=2586, poison=poisoned.record), immutable=True)
    print('PREFLIGHT_PASS: full PiW data + v3 reference + BadNets; CPU only, no training', flush=True)


def require_idle():
    from run_mmfi_badnets_two_models_4090 import require_idle as guard
    guard()


def gpu_signature():
    from run_mmfi_badnets_two_models_4090 import gpu_signature as signature
    value = signature()
    if '4090' not in value[1] or float(value[2]) < 22000:
        raise ValueError('Requires the local RTX4090/24GiB GPU0')
    return value


def smoke(plan):
    import torch
    from torch.utils.data import default_collate
    receipt(plan, 'preflight')
    require_idle()  # before any CUDA allocation; existing WiFlow is NOT stopped
    signature = gpu_signature()
    path = receipt_path(plan, 'gpu_smoke')
    if path.exists():
        check(plan)
        print('GPU_SMOKE_PASS_EXISTING; no new smoke or training', flush=True)
        return
    running = dict(kind='gpu_smoke', status='RUNNING', plan_sha256=plan['sha256'],
        input_fingerprint=plan['inputs_fingerprint'], gpu=signature)
    write(path, running)
    try:
        native = engine()
        execution = native.v3.seed_execution(plan['cfg'])
        ref, ds = native.datasets(plan)
        trigger = BadNetsGroups()
        pairs = BadNetsPairs(ds['training'], trigger, ref['poison'])
        indices = list(pairs.selected)[:4]
        indices += [i for i in range(len(pairs)) if i not in pairs.selected][:28]
        if len(indices) != 32:
            raise ValueError('Physical B32 smoke requires four poison and 28 clean')
        batch = default_collate([pairs[i] for i in indices])
        model = native.v3.base.build_model().to('cuda:0').train()
        optimizer = native.v3.optimizer_for(model, plan['cfg'])
        parameter = next(p for p in model.parameters() if p.requires_grad)
        before = parameter.detach().clone()
        torch.cuda.reset_peak_memory_stats('cuda:0')
        losses = [native.v3.step(model, optimizer, batch, 'cuda:0')[0] for _ in range(2)]
        torch.cuda.synchronize('cuda:0')
        if not all(math.isfinite(v) for v in losses) or torch.equal(before, parameter.detach()):
            raise ValueError('Nonfinite BadNets B32 smoke loss')
        record = dict(execution=execution, batch_size=32, steps=2, poison_examples=4,
            losses=losses, real_train_input=True, parameter_updated=True,
            poison=pairs.record,
            allocated_peak_bytes=torch.cuda.max_memory_allocated('cuda:0'),
            reserved_peak_bytes=torch.cuda.max_memory_reserved('cuda:0'))
        write(path, dict(running, status='PASS', records=[record]))
        print('GPU_SMOKE_PASS: real TRAIN B32, four poisoned samples, two steps; no full training', flush=True)
    except BaseException as error:
        write(path, dict(running, status='FAILED', error=str(error)))
        raise


def check(plan):
    verify_plan(plan)
    receipt(plan, 'preflight')
    value = receipt(plan, 'gpu_smoke')
    records = value.get('records', [])
    if (value.get('gpu') != gpu_signature() or len(records) != 1
            or records[0].get('batch_size') != 32 or records[0].get('steps') != 2
            or records[0].get('poison_examples') != 4 or not records[0].get('execution')
            or records[0].get('real_train_input') is not True
            or records[0].get('parameter_updated') is not True
            or records[0].get('poison') != receipt(plan, 'preflight').get('poison')
            or len(records[0].get('losses', [])) != 2
            or not all(type(x) in (int, float) and math.isfinite(x) for x in records[0]['losses'])):
        raise ValueError('Wrong GPU or incomplete real BadNets smoke receipt')
    return value


def run_phase(plan, work):
    """Exactly one blocking worker; on Ctrl+C drain it without process kill."""
    log = cell(plan) / f'{work}.{time.time_ns()}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, '-u', str(Path(__file__).resolve()), '--data-home',
        plan['cfg']['data_home'], '--outdir', plan['cfg']['outdir'],
        '--reference-dir', plan['reference'], '--mode', 'cell', '--work', work]
    with log.open('xb') as stream:
        process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT,
            cwd=str(HERE), creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0))
        print(f'[{work}] BadNets PID={process.pid}; LOG={log}', flush=True)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            print('INTERRUPTED: no later phase; draining owned worker without kill. Keep SSH open.', flush=True)
            while process.poll() is None:
                try:
                    process.wait(timeout=1)
                except (KeyboardInterrupt, subprocess.TimeoutExpired):
                    pass
            raise
    if code:
        print(log.read_text(encoding='utf-8', errors='replace')[-6000:], file=sys.stderr)
        raise RuntimeError(f'{work} failed (exit {code}); no retry/later phase. Log: {log}')


def export(plan):
    native = engine()
    value = native.verify_evaluation(plan, METHOD)
    if value is None:
        raise ValueError('Final199 evaluation is not complete')
    metrics = value['metrics']['mainmetrics']
    write(Path(plan['cfg']['outdir']) / 'badnets_results.json', value)
    print('EVALUATION_COMPLETE BadNets-CSI ' + json.dumps(metrics), flush=True)
    return value


def launch(plan):
    from run_mmfi_tables import _cell_lock
    with _cell_lock(Path(plan['cfg']['data_home']) / 'runs/.piw_hpeli_badnets_gpu0'):
        with _cell_lock(Path(plan['cfg']['outdir'])):
            require_idle()
            check(plan)
            full_audit(plan)
            native = engine()
            if native.verify_evaluation(plan, METHOD) is None:
                logs = list(cell(plan).glob('train.*.log'))
                if logs and not (cell(plan) / 'checkpoint.pt').exists():
                    raise ValueError('Training previously started but checkpoint missing; no silent restart')
                run_phase(plan, 'train')
                native.checkpoint(plan, METHOD, final=True)
                require_idle()
                check(plan)
                full_audit(plan)
                run_phase(plan, 'evaluate')
            export(plan)


def status(output):
    root = Path(output)
    folder = root / 'hpeli/seed42/badnets'
    print(f'OUTPUT={root}\nStatus reads logs/markers only; it does not prove a live worker.')
    for name in ('evaluation.json', 'checkpoint.pt'):
        p = folder / name
        print(f'{name}: {p.is_file()}')
    for work in ('train', 'evaluate'):
        logs = sorted(folder.glob(f'{work}.*.log'), key=lambda p: p.stat().st_mtime)
        if logs:
            print(f'LOG={logs[-1]}')
            print('\n'.join(logs[-1].read_text(encoding='utf-8', errors='replace').splitlines()[-8:]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path, default=Path.home() / 'backdooranalog')
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--mode', choices=('dry-run', 'preflight', 'prepare', 'check', 'launch', 'status', 'cell'), default='status')
    parser.add_argument('--work', choices=('train', 'evaluate'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.outdir or args.data_home / 'runs/piw_hpeli_badnets_s42_4090_v1'
    if args.mode == 'status':
        status(output)
        return 0
    if (args.mode == 'cell') != bool(args.work):
        parser.error('--work is reserved for an explicit cell invocation')
    plan = build_plan(args.data_home, output, args.reference_dir)
    if args.mode == 'dry-run':
        print(json.dumps(plan, indent=2))
        return 0
    if args.mode in ('preflight', 'prepare'):
        from run_mmfi_tables import _cell_lock
        with _cell_lock(output):
            preflight(plan)
            if args.mode == 'prepare':
                smoke(plan)
        print('PREPARED_ONLY; no full training. Invoke Launch separately.' if args.mode == 'prepare'
              else 'CPU_PREFLIGHT_ONLY; CUDA smoke still required before Launch.', flush=True)
        return 0
    if read(Path(output) / MANIFEST) != plan:
        raise ValueError('Saved BadNets plan differs')
    if args.mode == 'check':
        require_idle(); check(plan)
        print('VERIFIED_PRELAUNCH_GATE; no training launched', flush=True)
    elif args.mode == 'launch':
        launch(plan)
    else:
        from run_mmfi_tables import _cell_lock
        with _cell_lock(cell(plan)):
            require_idle(); check(plan)
            native = engine()
            if args.work == 'train':
                native.train(plan, METHOD)
            else:
                native.evaluate(plan, METHOD)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'[PiW HPE-Li BadNets] ERROR: {error}', file=sys.stderr, flush=True)
        raise
