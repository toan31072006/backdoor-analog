"""Process-local execution adapter for an already-prepared TrojanNN victim.

The original scientific snapshot and its gate remain untouched.  This adapter
adds exactly one user-approved, creation-time-bound nscaling process tree to the
original A1-sharing guard.  It does not establish OS resource reservations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import ntpath
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


RUNNER_SHA256 = '0c335047a4869d0a354cf184ffb41ede04fd9e324452913372d31bfa9f565318'
PROFILE = 'mmfi_trojannn_existing_gate_shared4090_execution_v1'
NSCALING_PARENT = r'U:\ijoc\.venv\Scripts\python.exe'
NSCALING_CHILD = r'C:\Users\Van\AppData\Local\Python\pythoncore-3.12-64\python.exe'
NSCALING_SUFFIX = ['-u', r'scripts\nscaling.py', 'LandS,gbd,storm,20term,ssn', 'S1,S3', '5']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _path(value):
    if not isinstance(value, str) or not value or any(c in value for c in '\r\n\x00'):
        raise ValueError('Missing/invalid execution path')
    return ntpath.normcase(ntpath.normpath(value.replace('/', '\\')))


def _argv(line):
    # Keep the same strict Windows command-line subset as the frozen runner.
    import re
    token = r'(?:"[^"\r\n]*"|[^\s"\r\n]+)'
    if (not isinstance(line, str) or any(ord(c) < 32 for c in line)
            or not re.fullmatch(r'\s*' + token + r'(?:\s+' + token + r')*\s*', line)):
        raise ValueError('Cannot parse approved Windows command line')
    return [m.group()[1:-1] if m.group().startswith('"') else m.group()
            for m in re.finditer(token, line)]


def _nscaling_record(record, executable):
    if (not isinstance(record, dict) or type(record.get('ProcessId')) is not int
            or record['ProcessId'] <= 0 or type(record.get('ParentProcessId')) is not int
            or record['ParentProcessId'] < 0 or str(record.get('Name', '')).lower() != 'python.exe'
            or not isinstance(record.get('CreationDate'), str) or not record['CreationDate']):
        raise ValueError('Approval requires PID, parent, executable, argv and creation time')
    argv = _argv(record.get('CommandLine'))
    if (len(argv) != 6 or _path(record.get('ExecutablePath')) != _path(executable)
            or _path(argv[0]) != _path(executable)
            or argv[1] != NSCALING_SUFFIX[0]
            or _path(argv[2]) != _path(NSCALING_SUFFIX[1])
            or argv[3:] != NSCALING_SUFFIX[2:]):
        raise ValueError('Approval is restricted to the exact observed nscaling invocation')
    return (record['ProcessId'], record['ParentProcessId'], record['Name'].lower(),
            _path(record['ExecutablePath']), tuple([_path(argv[0]), argv[1],
            _path(argv[2]), *argv[3:]]), record['CreationDate'])


def validate_nscaling(records, approved):
    """Return only live members of the exact, immutable approved two-process tree.

    An absent PID is a completed process, not blanket permission for a new job.
    A reused PID or changed ancestry/command is rejected, even if presently idle.
    """
    if approved == []:
        return set()  # Already-finished peer grants no exception to any PID.
    if not isinstance(approved, list) or len(approved) != 2:
        raise ValueError('Exactly the observed parent and child must be approved')
    try:
        parent = next(r for r in approved
                      if _path(r.get('ExecutablePath')) == _path(NSCALING_PARENT))
        child = next(r for r in approved
                     if _path(r.get('ExecutablePath')) == _path(NSCALING_CHILD))
    except (StopIteration, AttributeError, TypeError):
        raise ValueError('Missing exact nscaling parent/child approval') from None
    parent_signature = _nscaling_record(parent, NSCALING_PARENT)
    child_signature = _nscaling_record(child, NSCALING_CHILD)
    signatures = {parent_signature[0]: parent_signature, child_signature[0]: child_signature}
    if len(signatures) != 2 or child['ParentProcessId'] != parent['ProcessId']:
        raise ValueError('Approved nscaling process tree is not the observed proxy chain')
    if records is None:
        raise ValueError('Live Windows process identities are required')
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError('Cannot parse live Windows process identities')
    current = {}
    for record in records:
        pid = record.get('ProcessId')
        if type(pid) is not int or pid < 0:
            raise ValueError('Live CIM snapshot has a missing/invalid process identity')
        if pid in signatures:
            if pid in current:
                raise ValueError('Duplicate live process identity')
            expected_exe = NSCALING_PARENT if pid == parent['ProcessId'] else NSCALING_CHILD
            if _nscaling_record(record, expected_exe) != signatures[pid]:
                raise ValueError(f'Approved nscaling PID {pid} was reused or changed')
            current[pid] = record
    if child['ProcessId'] in current and parent['ProcessId'] not in current:
        raise ValueError('Live nscaling child no longer has the approved parent')
    return set(current)


def windows_processes():
    if os.name != 'nt':
        raise ValueError('This execution exception requires real Windows CIM identities')
    script = ('[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); '
              '@(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,'
              'Name,ExecutablePath,CommandLine,@{Name="CreationDate";Expression={'
              '$_.CreationDate.ToUniversalTime().ToString("o")}}) | ConvertTo-Json -Compress')
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], check=True,
                            capture_output=True, encoding='utf-8', timeout=30)
    rows = json.loads(result.stdout.lstrip('\ufeff'))
    return rows if isinstance(rows, list) else [rows]


def install_guard(run, approved):
    """Keep the original guard and narrow only its returned GPU conflicts."""
    if getattr(run.classify_gpu, '_shared4090_exception', False):
        raise ValueError('Execution exception was already installed')
    # Validate the approval itself before importing any broad GPU allowance.
    validate_nscaling([], approved)
    original = run.classify_gpu

    def classified(report, windows_names=None, own_pid=None, *, plan=None,
                   windows_processes=None):
        conflicts = original(report, windows_names, own_pid, plan=plan,
                             windows_processes=windows_processes)
        permitted = validate_nscaling(globals()['windows_processes'](), approved)
        return [row for row in conflicts if row[0] not in permitted]

    classified._shared4090_exception = True
    run.classify_gpu = classified
    return original


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError('Shared execution request must be a JSON object')
    unsigned = dict(request)
    claimed = unsigned.pop('request_sha256', None)
    if request.get('schema') != 1 or claimed != digest(unsigned):
        raise ValueError('Missing/changed shared execution request')
    if request.get('trojan_runner_sha256') != RUNNER_SHA256:
        raise ValueError('Only the already-prepared frozen TrojanNN runner is supported')
    validate_nscaling([], request.get('approved_nscaling'))
    for key in ('home', 'python', 'trojan_repo', 'a1_repo', 'trojan_out', 'a1_out',
                'clean_checkpoint', 'smoke_receipt'):
        value = request.get(key)
        _path(value)
        if not Path(value).is_absolute():
            raise ValueError(f'Absolute shared execution path required: {key}')
    receipt = Path(request['smoke_receipt']).resolve()
    if any(receipt.is_relative_to(Path(request[k]).resolve())
           for k in ('trojan_repo', 'a1_repo', 'trojan_out', 'a1_out')):
        raise ValueError('Shared smoke receipt must not modify frozen sources or historical outputs')
    sources = request.get('source_sha256')
    required_sources = {Path(__file__).resolve(), Path(__file__).resolve().with_name('controller.py')}
    if (not isinstance(sources, dict)
            or {Path(path).resolve() for path in sources} != required_sources):
        raise ValueError('Shared execution must bind both external adapter sources')
    for path, expected in sources.items():
        if sha(path) != expected:
            raise ValueError(f'Shared execution source changed: {path}')
    return request


def read_request(path):
    return validate_request(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def import_runner(request):
    path = Path(request['trojan_repo']) / 'ATKBackd' / 'run_mmfi_p1s1_trojannn.py'
    if sha(path) != RUNNER_SHA256:
        raise ValueError('Frozen TrojanNN source changed; do not rewrite its prepared gate')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path.parent))
    run = importlib.import_module('run_mmfi_p1s1_trojannn')
    if Path(run.__file__).resolve() != path.resolve():
        raise ValueError('Foreign TrojanNN import context')
    return run


def build_trojan_plan(request, run):
    root = Path(request['trojan_out'])
    for filename, key in (('plan.json', 'trojan_plan_sha256'),
                          ('trigger.pt', 'trojan_trigger_sha256'),
                          ('gpu_gate.json', 'trojan_gate_sha256')):
        if sha(root / filename) != request.get(key):
            raise ValueError(f'Original prepared TrojanNN artifact changed: {filename}')
    plan = run.build_plan(request['home'], root, request['clean_checkpoint'],
                          'cuda:0', allow_concurrent_a1=True)
    if run.read(root / 'plan.json') != plan:
        raise ValueError('Frozen scientific plan differs; no re-Prepare or gate rewriting allowed')
    run.verify_inputs(plan)
    return plan


def smoke_binding(request, run, env):
    return dict(request_sha256=request['request_sha256'],
                adapter_sha256=sha(__file__),
                original_gate_sha256=request['trojan_gate_sha256'],
                frozen_trigger_sha256=request['trojan_trigger_sha256'],
                execution=env, profile=PROFILE)


def shared_smoke(request, run, plan):
    """Two B32 real-TRAIN updates on a throwaway victim, frozen trigger only."""
    import torch
    from attack.poison import collate
    from train_backdoor import _victim_update
    receipt = Path(request['smoke_receipt'])
    if receipt.exists():
        raise ValueError('Shared smoke receipt exists; inspect it, never overwrite it')
    run.require_idle(plan)
    env = run.gate_check(plan)
    trigger = run.load_trigger(plan, env)
    model, optimizer, pairs = run.components(plan, trigger)
    selected = sorted(pairs.poison_idx)[:4]
    selected += [i for i in range(len(pairs)) if i not in pairs.poison_idx][:28]
    batch = collate([pairs[i] for i in selected])
    if len(selected) != 32 or tuple(batch['csi'].shape) != (32, 3, 114, 10):
        raise ValueError('Shared context requires the original physical B32 real TRAIN smoke')
    before = next(model.parameters()).detach().clone()
    losses = [_victim_update(model, optimizer, batch['csi'].to('cuda:0'),
                             batch['pose'].to('cuda:0')) for _ in range(2)]
    torch.cuda.synchronize()
    if not all(math.isfinite(v) for v in losses) or torch.equal(before, next(model.parameters())):
        raise ValueError('Shared B32 CUDA smoke failed finite/update checks')
    # Recheck identities after CUDA work, before attesting to this context.
    run.require_idle(plan)
    permitted = validate_nscaling(windows_processes(), request['approved_nscaling'])
    row = dict(status='shared_smoke_passed', binding=smoke_binding(request, run, env),
               utc=datetime.now(timezone.utc).isoformat(), physical_batch=32, steps=2,
               real_train_input=True, poison_examples=4, losses=losses,
               approved_nscaling=request['approved_nscaling'],
               live_nscaling_pids=sorted(permitted),
               max_victim_workers=2, additional_approved_jobs=1,
               context_note='External sharing receipt; original gate certifies original context only. '
                            'No guarantee of speedup or future VRAM availability; no full training.')
    row['receipt_sha256'] = digest(row)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(row, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print('SHARED_TROJAN_B32_SMOKE_PASS; frozen trigger retained; no full training', flush=True)
    return row


def check_smoke(request, run, env):
    row = json.loads(Path(request['smoke_receipt']).read_text(encoding='utf-8'))
    unsigned = dict(row)
    claimed = unsigned.pop('receipt_sha256', None)
    if (claimed != digest(unsigned) or row.get('status') != 'shared_smoke_passed'
            or row.get('binding') != smoke_binding(request, run, env)
            or row.get('physical_batch') != 32 or row.get('steps') != 2
            or row.get('real_train_input') is not True or row.get('poison_examples') != 4
            or not isinstance(row.get('losses'), list) or len(row['losses']) != 2
            or not all(math.isfinite(v) for v in row['losses'])
            or row.get('approved_nscaling') != request['approved_nscaling']):
        raise ValueError('Missing/foreign shared-context CUDA smoke; original gate is insufficient')
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--phase', choices=('check', 'smoke', 'launch'), required=True)
    args = parser.parse_args(argv)
    request = read_request(args.request)
    if Path(sys.executable).resolve() != Path(request['python']).resolve():
        raise ValueError('Shared execution Python differs from the pinned prepared environment')
    run = import_runner(request)
    install_guard(run, request['approved_nscaling'])
    plan = build_trojan_plan(request, run)
    if args.phase == 'check':
        run.require_idle(plan)
        run.gate_check(plan)
        print('SHARED_TROJAN_FROZEN_GATE_CHECK_PASS; no training', flush=True)
        return
    if args.phase == 'smoke':
        from run_mmfi_tables import _cell_lock
        with _cell_lock(Path(plan['home'])/'tools/trojannn_gpu_locks/cuda0'), \
                _cell_lock(Path(plan['outdir'])/'.controller'):
            shared_smoke(request, run, plan)
        return
    env = run.gate_check(plan)
    check_smoke(request, run, env)
    # The original entry point retains pending-job checks, recipe, both output
    # locks, exact model/SGD/RNG resume, training and evaluation contract.
    run.main(['--data-home', request['home'], '--outdir', request['trojan_out'],
              '--clean-checkpoint', request['clean_checkpoint'], '--device', 'cuda:0',
              '--mode', 'launch', '--allow-concurrent-a1'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'[shared4090 TrojanNN] ERROR: {error}; no process stopped or automatic retry',
              file=sys.stderr, flush=True)
        raise
