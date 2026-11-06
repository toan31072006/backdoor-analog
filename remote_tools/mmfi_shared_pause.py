"""Windows-only, explicitly authorized pause of an authenticated old worker.

No wildcard process stops, no automatic checkpoint restoration, no CUDA calls.
Native Windows replacement can reject an open destination even with SHARE_DELETE.
Never open a live writable checkpoint: briefly suspend its exact bound process,
validate/copy the committed file, then terminate only after a verified backup.
Any validation exception resumes the original process in finally.
"""
from __future__ import annotations

import ctypes
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid


def process_row(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('An exact positive PID is required')
    script = (f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' | "
              'Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine,CreationDate | '
              'ConvertTo-Json -Compress')
    result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                            capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout) if result.stdout.strip() else None


def discover():
    """Return only original P3S3 worker PIDs, never unrelated command lines."""
    script = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*Run-MMFiBackTimeTwoSplits4090.py*' "
        "-and $_.CommandLine -match '--mode\\s+_worker' "
        "-and $_.CommandLine -match '--split\\s+p3s3' } | "
        'Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress')
    result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                            capture_output=True, text=True, check=True, timeout=30)
    values = json.loads(result.stdout) if result.stdout.strip() else []
    return values if isinstance(values, list) else [values]


def argv_windows(command):
    count = ctypes.c_int()
    function = ctypes.windll.shell32.CommandLineToArgvW
    function.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.POINTER(ctypes.c_wchar_p)
    pointer = function(command, ctypes.byref(count))
    if not pointer:
        raise ctypes.WinError()
    free = ctypes.windll.kernel32.LocalFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p
    try:
        return [pointer[i] for i in range(count.value)]
    finally:
        free(pointer)


def command_options(row, mode):
    """Parse only the exact old launch/worker vocabulary; fail closed."""
    args = argv_windows(row.get('CommandLine') or '')
    if len(args) < 4 or args[1:3] != ['-B', '-u']:
        raise ValueError('Not the original Python -B -u launcher')
    rest = args[4:]
    allowed = {'--mode', '--data-home', '--outdir'}
    if mode == '_worker':
        allowed |= {'--split', '--context'}
    if len(rest) != len(allowed)*2 or set(rest[::2]) != allowed:
        raise ValueError('Unexpected original-worker command arguments')
    options = dict(zip(rest[::2], rest[1::2]))
    if options['--mode'] != mode or (mode == '_worker' and options['--split'] != 'p3s3'):
        raise ValueError('Only the original P3S3 training worker can be paused')
    return Path(args[0]), Path(args[3]), options


class BoundProcess:
    """Keep a Windows handle open before validating PID identity (no PID reuse)."""
    def __init__(self, pid, *, terminate=False):
        if os.name != 'nt':
            raise RuntimeError('Authorized pause is Windows-only')
        self.kernel = ctypes.windll.kernel32
        self.kernel.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
        self.kernel.OpenProcess.restype = ctypes.c_void_p
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self.kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self.handle = self.kernel.OpenProcess(0x100000 | 0x1000 | (0x801 if terminate else 0), 0, pid)
        if not self.handle:
            raise ctypes.WinError()
        self.can_terminate = terminate
        self.suspended = False

    def suspend(self):
        if not self.can_terminate or self.suspended or not self.alive():
            raise ValueError('Only the active authenticated pause target may be suspended')
        function = ctypes.windll.ntdll.NtSuspendProcess
        function.argtypes, function.restype = [ctypes.c_void_p], ctypes.c_long
        status = function(self.handle)
        if status < 0:
            raise OSError(f'Native process suspension failed: NTSTATUS={status & 0xffffffff:08x}')
        self.suspended = True

    def resume(self):
        if self.suspended:
            function = ctypes.windll.ntdll.NtResumeProcess
            function.argtypes, function.restype = [ctypes.c_void_p], ctypes.c_long
            status = function(self.handle)
            if status < 0:
                raise OSError(f'Native process resume failed: NTSTATUS={status & 0xffffffff:08x}')
            self.suspended = False

    def alive(self):
        code = ctypes.c_uint()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError()
        return code.value == 259

    def stop_authorized(self):
        if not self.can_terminate or not self.alive():
            raise ValueError('Bound authorized worker is no longer active; nothing stopped')
        if not self.kernel.TerminateProcess(self.handle, 130):
            raise ctypes.WinError()
        self.suspended = False

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        try:
            if self.suspended and self.alive():
                self.resume()
        finally:
            self.kernel.CloseHandle(self.handle)


def checkpoint_choice(plan):
    root = Path(plan['outdir'])
    choices = [('victim', root/'backtime/checkpoint.pt'),
               ('trigger_complete', root/'backtime/trigger.pt'),
               ('attacker', root/'backtime/search_state.pt'),
               ('clean_surrogate', root/'clean_surrogate/checkpoint.pt')]
    chosen = next(((phase, path) for phase, path in choices if path.is_file()), None)
    if chosen is None:
        raise ValueError('No committed native resume checkpoint; refusing a fresh restart')
    return chosen


def checkpoint(runtime, plan, env, *, backup=None):
    """Validate the latest committed native resume point on CPU; no fresh fallback."""
    import torch
    import attack.backtime_mmfi as bt
    phase, path = checkpoint_choice(plan)
    runtime.reject_links(path)
    with path.open('rb') as stream:
        blob = torch.load(stream, map_location='cpu', weights_only=False)
        if phase in ('victim', 'clean_surrogate'):
            cell = 'backtime' if phase == 'victim' else phase
            runtime.validate_checkpoint(blob, plan, cell, env)
            epoch = blob['epoch'] + 1
        elif phase == 'trigger_complete':
            if blob.get('binding') != runtime.trigger_binding(plan, 'backtime', env):
                raise ValueError('Frozen BackTime trigger binding changed')
            if bt.FrozenBackTimeTrigger.from_state_dict(blob['trigger']).provenance() != blob.get('provenance'):
                raise ValueError('Frozen BackTime trigger provenance changed')
            epoch = 100
        else:
            if blob.get('binding') != runtime.trigger_binding(plan, 'backtime', env, for_search=True):
                raise ValueError('BackTime search checkpoint binding changed')
            state = dict(blob['state'])
            claimed = state.pop('sha256', None)
            required = {'generator','surrogate','surrogate_optimizer','generator_optimizer',
                'scheduler','numpy_rng','torch_rng','cuda_rng','history','resolved_doses',
                'online_scores','selection_scores','recipients','surrogate_updates','generator_updates'}
            epoch = state.get('epoch')
            if (claimed != bt._sha(state) or state.get('version') != bt.IMPLEMENTATION
                    or type(epoch) is not int or not 0 <= epoch <= 100
                    or not required.issubset(state) or state['cuda_rng'] is None):
                raise ValueError('Incomplete/corrupt native BackTime search state')
            inner = state.get('binding',{})
            if 'operators' in plan:
                expected = dict(implementation=bt.IMPLEMENTATION,
                    config=asdict(bt.BackTimeConfig(**plan['operators']['backtime'])),
                    dataset_sha256=runtime.verify_inputs(plan)['full_content_sha256'],
                    n_train=plan['contract']['counts']['training'],
                    n_poison=plan['contract']['n_poison'],
                    doses_sha256=bt._sha(runtime.full_train_doses(plan,'backtime')))
                if any(inner.get(k) != v for k,v in expected.items()):
                    raise ValueError('Native BackTime inner TRAIN/config/dose binding changed')
            for name in ('generator','surrogate'):
                if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state[name].values()):
                    raise ValueError('Nonfinite native BackTime model state')
        stream.seek(0)
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if backup is not None:
            backup = Path(backup)
            runtime.reject_links(backup)
            backup.parent.mkdir(parents=True, exist_ok=True)
            stream.seek(0)
            with backup.open('xb') as target:
                shutil.copyfileobj(stream, target)
                target.flush()
                os.fsync(target.fileno())
            if runtime.sha(backup) != digest:
                raise ValueError('Checkpoint backup checksum failed; nothing stopped')
    return dict(phase=phase, next_epoch=epoch, path=str(path), sha256=digest,
                backup=None if backup is None else str(backup))


def verify_owner(runtime, original, old_master, root, home, pid):
    row = process_row(pid)
    if row is None:
        raise ValueError('Original worker already exited; nothing stopped')
    exe, script, options = command_options(row, '_worker')
    if (exe.resolve() != Path(os.sys.executable).resolve()
            or Path(row.get('ExecutablePath') or '').resolve() != exe.resolve()
            or script.name != 'Run-MMFiBackTimeTwoSplits4090.py'
            or runtime.sha(script) != runtime.sha(original.__file__)
            or Path(options['--outdir']).resolve() != root.resolve()
            or Path(options['--data-home']).resolve() != home.resolve()):
        raise ValueError('Foreign worker/executable/source/path; nothing stopped')
    context_path = Path(options['--context'])
    runtime.reject_links(context_path)
    ctx = runtime.read(context_path)
    unsigned = dict(ctx)
    claimed = unsigned.pop('sha256', None)
    if (claimed != runtime.digest(unsigned) or ctx.get('profile') != original.PROFILE
            or ctx.get('action') != 'launch' or ctx.get('root') != str(root)
            or ctx.get('parent') != row['ParentProcessId']
            or set(ctx.get('workers', {})) != {'p1s1','p3s3'}
            or ctx['workers']['p3s3'] != pid or len(set(ctx['workers'].values())) != 2
            or ctx.get('script_sha256') != runtime.sha(original.__file__)
            or ctx.get('plan_hashes') != {s:p['plan_sha256'] for s,p in old_master['plans'].items()}
            or context_path != Path(ctx.get('directory',''))/'workers.json'
            or context_path.parent.parent != root/'executions'):
        raise ValueError('Original ownership context differs; nothing stopped')
    parent = process_row(row['ParentProcessId'])
    if parent is None:
        raise ValueError('Original parent is missing; explicit recovery required')
    parent_exe, parent_script, opts = command_options(parent, 'launch')
    if (parent_exe.resolve() != exe.resolve() or parent_script.resolve() != script.resolve()
            or Path(opts['--outdir']).resolve() != root.resolve()
            or Path(opts['--data-home']).resolve() != home.resolve()):
        raise ValueError('Original parent identity differs; nothing stopped')
    return dict(worker=row, parent=parent, context_sha256=runtime.sha(context_path), context=str(context_path))


def pause(runtime, original, master, root, home, output, env, pid):
    """Called only for explicit Pause/SwitchAndLaunch, never Inspect/Prepare/Launch."""
    receipt = output/'pause.json'
    if receipt.exists():
        raise ValueError('Pause already recorded; use Prepare/Launch, never stop another worker')
    with BoundProcess(pid, terminate=True) as handle:
        owner = verify_owner(runtime, original, master, root, home, pid)
        runtime.write(output/'pause_identity.json', dict(owner=owner,
            old_master_sha256=master['sha256'], temporary_suspend_before_checkpoint_open=True))
        print(f'TEMPORARILY_HOLD_AUTHENTICATED_P3S3 pid={pid}; '
              'validation failure resumes the original process', flush=True)
        handle.suspend()
        before = checkpoint(runtime, master['plans']['p3s3'], env,
            backup=output/'pause_backups'/uuid.uuid4().hex/'checkpoint.pt')
        runtime.write(output/'pause_intent.json', dict(owner=owner, before=before,
            old_master_sha256=master['sha256'], partial_epoch_may_replay=True))
        if not handle.alive():
            raise ValueError('Worker exited during backup; no stop attempted; inspect artifacts')
        # Re-check process/source/context after the potentially expensive backup.
        if verify_owner(runtime, original, master, root, home, pid) != owner:
            raise ValueError('Process changed during backup; nothing stopped')
        print(f'AUTHORIZED_PAUSE pid={pid}; BACKUP={before["backup"]}; '
              'only unfinished epoch may replay, never restart from scratch', flush=True)
        handle.stop_authorized()
        began = time.monotonic()
        while handle.alive():
            if time.monotonic()-began > 60:
                raise TimeoutError('Authorized worker did not exit; no further stop attempted')
            time.sleep(.2)
    # Original parent drains its finished P1 peer and releases its OS lock.
    from run_mmfi_tables import _cell_lock
    began = time.monotonic()
    while True:
        try:
            lock = _cell_lock(root/'.controller')
            lock.__enter__()
            break
        except RuntimeError:
            if time.monotonic()-began > 60:
                raise TimeoutError('Original controller still owns its lock; preserve everything')
            time.sleep(.2)
    try:
        split_root = root/'p3s3'
        renamed = []
        with _cell_lock(split_root/'.controller'):
            after = checkpoint(runtime, master['plans']['p3s3'], env)
            for item in sorted(split_root.rglob('*.saving')):
                runtime.reject_links(item)
                if not item.is_file() or not item.resolve().is_relative_to(split_root.resolve()):
                    raise ValueError('Unexpected staging target; preserve it and inspect')
                target = item.with_name(item.name+'.paused_'+uuid.uuid4().hex)
                item.rename(target)
                renamed.append(dict(original=str(item), preserved_as=str(target)))
            result = dict(status='PAUSED_WITH_VALID_NATIVE_CHECKPOINT', owner=owner,
                before=before, after=after, old_master_sha256=master['sha256'],
                preserved_staging=renamed, partial_epoch_may_replay=True)
            runtime.write(receipt, result)
            print('PAUSED_AND_BACKED_UP '+json.dumps(after), flush=True)
            return result
    finally:
        lock.__exit__(None, None, None)
