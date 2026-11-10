"""Read-only import of a completed, source-pinned PiW BackTime attacker.

This is frozen-generator reuse, NOT migration of optimizer/RNG or victims.
The original plan, execution receipt, trigger and provenance stay unchanged.
"""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import numpy as np
import run_piw_hpeli_clean as base
from piw_backtime_clean_reference import regular, read


HERE = Path(__file__).resolve().parent
SOURCE_PROFILE = 'piw_backtime_hpeli_shared_three_backbones_s42_v1'
# Published 9d3c5da runner, in LF and Windows CRLF form. Never execute it.
RUNNER_HASHES = {
    '9ac48b1e33ef4e689ca4474e722eb83802b5c88598719eb85503ce7ec7033c24',
    '74b2bacdf7b71f389b9bd23eafa79bec1aa440d4989facafa323ed1c75e708ea',
}
SCIENTIFIC_SOURCES = (
    'run_piw_hpeli_clean.py', 'attack/backtime_piw.py', 'attack/backtime_mmfi.py',
    'piw_v2_adapter.py', 'piw_a1_adapter.py', 'data_utils/feeder.py',
    'models/hpeli.py', 'models/sk_network.py',
)


def require_idle(folder):
    """A forced stop may leave the old lock. Read it, never remove it."""
    lock = Path(folder) / 'runner.lock'
    if not lock.exists():
        return
    pid = read(regular(lock)).get('pid')
    if type(pid) is not int or pid <= 0:
        raise ValueError('Malformed source runner lock; preserve it for inspection')
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
        if not handle:
            if ctypes.get_last_error() == 87:  # no such process
                return
            raise ValueError('Cannot establish that source runner is stopped')
        try:
            active = kernel.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)  # POSIX liveness check; never used on Windows
            active = True
        except ProcessLookupError:
            active = False
        except PermissionError:
            active = True
    if active:
        raise ValueError(f'Source runner PID {pid} is still active; stop that job first')


def validate(blob, cfg, inputs):
    """Validate the frozen state on CPU without data loaders, TEST or fitting."""
    import torch
    from attack.backtime_piw import FrozenPiWBackTimeTrigger
    frozen = blob['trigger']
    if frozen.get('diagnostics', {}).get('fit_profile') != 'full_TRAIN_streaming_author_schedule_PiW14_adaptation':
        raise ValueError('Not a completed PiW14 TRAIN attacker')
    trigger = FrozenPiWBackTimeTrigger.from_state_dict(frozen)
    if json.loads(json.dumps(asdict(trigger.config))) != cfg['attacker']:
        raise ValueError('Frozen attacker configuration differs from new plan')
    n = inputs['splits']['train_data']['retained_count']
    doses = np.random.default_rng(42).uniform(.2, 1., n).astype(np.float32)
    indices, diag = trigger.training_indices, trigger.provenance()['diagnostics']
    k, steps = math.floor(.1*n), math.ceil(n/64)
    scores = trigger.training_selection_scores
    variables = trigger.generator.variables.cpu().numpy()
    if (not np.array_equal(blob['doses'], doses) or len(indices) != k
            or len(set(indices)) != k or any(type(i) is not int or not 0 <= i < n for i in indices)
            or len(variables) != 81 or not np.array_equal(variables, np.unique(variables))
            or np.any(variables < 0) or np.any(variables >= 270)
            or scores is None or scores.shape != (n,) or not torch.isfinite(scores).all()
            or diag.get('fit_epochs') != 100 or diag.get('test_used') is not False
            or diag.get('n_fit') != n or diag.get('n_poisoned_fit') != k
            or diag.get('generator_updates') != k*89 or diag.get('clean_warmup_epochs') != 11
            or diag.get('surrogate_updates') != steps*100
            or diag.get('fresh_surrogate_template_unchanged') is not True
            or diag.get('dataset_sha256') != inputs['fingerprint']):
        raise ValueError('Incomplete/foreign attacker, selection or paired doses')
    history = diag.get('history', [])
    if len(history) != 100:
        raise ValueError('Attacker must contain all 100 completed epoch records')
    for index, row in enumerate(history):
        active = index > 10
        if (row.get('epoch') != index+1 or row.get('epoch_index') != index
                or row.get('phase') != ('alternating' if active else 'clean_warmup')
                or row.get('surrogate_updates') != steps*(index+1)
                or row.get('generator_updates') != k*max(0, index-10)
                or any(not math.isfinite(float(row.get(name, float('nan')))) for name in
                       (('surrogate_smooth_l1', 'generator_target_mse', 'generator_dc_sum')
                        if active else ('surrogate_smooth_l1',)))):
            raise ValueError('Incomplete/nonfinite attacker epoch history')
    return trigger, doses


def files_match(record):
    for name, expected in record['files'].items():
        path = regular(name)
        if path.stat().st_size != expected['bytes'] or base.file_sha(path) != expected['sha256']:
            raise ValueError(f'Historical attacker changed: {path}')


def audit(folder, cfg, inputs):
    import torch
    folder = Path(folder).resolve()
    for other in (Path(cfg['outdir']).resolve(), Path(cfg['dataset_root']).resolve()):
        if folder == other or folder in other.parents or other in folder.parents:
            raise ValueError('Attacker source must be separate from new output and raw dataset')
    paths = {name: regular(folder/name) for name in
             ('plan.json', 'capacity.json', 'attacker/trigger.pt', 'attacker/provenance.json')}
    require_idle(folder)
    files = {str(p): dict(bytes=p.stat().st_size, sha256=base.file_sha(p)) for p in paths.values()}
    plan, receipt = read(paths['plan.json']), read(paths['capacity.json'])
    unsigned = {k:v for k,v in plan.items() if k != 'plan_sha256'}
    if (plan.get('profile') != SOURCE_PROFILE or plan.get('cfg', {}).get('profile') != SOURCE_PROFILE
            or plan.get('plan_sha256') != base.object_sha(unsigned)):
        raise ValueError('Unknown or changed original attacker plan')
    source = plan['cfg']
    if Path(source.get('outdir', '')).resolve() != folder:
        raise ValueError('Original attacker plan output differs from its source directory')
    baseline = base.configuration(cfg['data_home'], cfg['outdir'], cfg['device'], cfg['dataset_root'])
    for key in set(baseline) - {'profile', 'outdir', 'data_home', 'device'}:
        if source.get(key) != cfg[key]:
            raise ValueError(f'Original attacker data/training contract differs: {key}')
    for key in ('attacker', 'target', 'rho', 'eval_doses', 'recipes'):
        if source.get(key) != cfg[key]:
            raise ValueError(f'Original attacker scientific contract differs: {key}')
    if (plan.get('inputs') != inputs or inputs['fingerprint'] != base.object_sha(inputs['splits'])):
        raise ValueError('Original attacker full input/split/order fingerprint differs')
    hashes = plan.get('sources', {})
    if hashes.get('ATKBackd/run_piw_backtime_three_backbones.py') not in RUNNER_HASHES:
        raise ValueError('Original attacker runner is not the published v1 source')
    for name in SCIENTIFIC_SOURCES:
        if hashes.get('ATKBackd/'+name) != base.file_sha(HERE/name):
            raise ValueError(f'Original attacker scientific source differs: {name}')
    receipt_unsigned = {k:v for k,v in receipt.items() if k != 'receipt_sha256'}
    expected_smokes = [(b,r,32,32*cfg['recipes'][b]['accumulation'])
                       for b in ('hpeli','wiflow','metafiplusplus') for r in ('clean','backtime')]
    if (receipt.get('receipt_sha256') != base.object_sha(receipt_unsigned)
            or receipt.get('profile') != SOURCE_PROFILE
            or receipt.get('plan_sha256') != plan['plan_sha256']
            or not isinstance(receipt.get('execution'), dict) or not receipt['execution']
            or receipt.get('surrogate_batch') != 64 or receipt.get('generator_batch') != 1
            or receipt.get('training_launched') is not False or receipt.get('checkpoints_saved') is not False
            or [(s.get('backbone'),s.get('role'),s.get('physical_batch'),s.get('effective_batch'))
                for s in receipt.get('victim_smokes',[])] != expected_smokes
            or any(not math.isfinite(s.get('loss',float('nan'))) for s in receipt.get('victim_smokes',[]))):
        raise ValueError('Original attacker capacity receipt is incomplete/foreign')
    blob = torch.load(paths['attacker/trigger.pt'], map_location='cpu', weights_only=False)
    if blob.get('plan_sha256') != plan['plan_sha256'] or blob.get('execution') != receipt['execution']:
        raise ValueError('Frozen attacker original plan/execution binding differs')
    trigger, _ = validate(blob, cfg, inputs)
    if read(paths['attacker/provenance.json']) != json.loads(json.dumps(trigger.provenance())):
        raise ValueError('Original attacker provenance differs from frozen state')
    record = dict(source_run=str(folder), checkpoint=str(paths['attacker/trigger.pt']),
        source_profile=SOURCE_PROFILE, source_plan_sha256=plan['plan_sha256'],
        source_runtime=plan['runtime'], source_execution=receipt['execution'], files=files,
        frozen_state_sha256=blob['trigger']['sha256'], training_started=False,
        policy='read-only completed PiW attacker; no optimizer/RNG/victim import or refit')
    files_match(record)  # reject concurrent source changes during audit
    return record


def load(record, cfg, inputs):
    import torch
    files_match(record)
    require_idle(record['source_run'])
    blob = torch.load(regular(record['checkpoint']), map_location='cpu', weights_only=False)
    if (blob.get('plan_sha256') != record['source_plan_sha256']
            or blob.get('execution') != record['source_execution']
            or blob.get('trigger', {}).get('sha256') != record['frozen_state_sha256']):
        raise ValueError('Original frozen attacker binding changed')
    trigger, doses = validate(blob, cfg, inputs)
    files_match(record)
    return trigger, doses
