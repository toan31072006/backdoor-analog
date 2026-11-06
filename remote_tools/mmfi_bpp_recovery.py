"""Explicit recovery ONLY for the published8551376 Bpp Prepare role mismatch.

No fake readiness, forced process termination, checkpoint import or scientific
recipe changes. Preserve failed execution logs and the prior plan byte-for-byte.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import mmfi_bpp_handover as handover

LEGACY_WRAPPER_SHA256='8f3d5b17db0b830ee78a8fca9e9927f5b97a115a117c9663092d53d1d88e48b7'
LEGACY_WRAPPER_HASHES={LEGACY_WRAPPER_SHA256,
    '1a5a9dde27827b1ca59040812032329cf6c96f3560ef941126224d549f344d91'}  # Git LF/CRLF only.
WRAPPER='Run-MMFiBackTimeBppShared4090.py'


def legacy_plan(r,plan,wrapper_sha256=LEGACY_WRAPPER_SHA256):
    if wrapper_sha256 not in LEGACY_WRAPPER_HASHES:
        raise ValueError('Not the published8551376 wrapper')
    old=copy.deepcopy(plan)
    old['execution_sources'].pop('mmfi_bpp_recovery.py')
    old['execution_sources'][WRAPPER]=wrapper_sha256
    old.pop('sha256')
    old['sha256']=r.digest(old)
    return old


def failed_contexts(r,old):
    root=Path(old['root'])
    # This recovery cannot migrate a trained Bpp run or reuse a capacity gate.
    for name in ('shared.gate.json','results.json','bpp_source','evaluations',
                 'launch_resume_points','recovery.json'):
        if (root/name).exists():
            raise ValueError('Not the known failed Prepare; preserve and inspect '+name)
    result=[]
    directory=root/'executions'
    if not directory.is_dir():
        raise ValueError('Missing failed Prepare execution context')
    for folder in sorted(directory.iterdir()):
        r.reject_links(folder)
        if not folder.is_dir() or not folder.name.startswith('prepare_'):
            raise ValueError('Only failed Prepare executions may be recovered')
        filename=folder/'workers.json'
        r.reject_links(filename)
        ctx=r.read(filename)
        unsigned=dict(ctx); claimed=unsigned.pop('sha256',None)
        workers=ctx.get('workers',{})
        if (claimed!=r.digest(unsigned) or ctx.get('profile')!=old['profile']
                or ctx.get('action')!='prepare' or ctx.get('root')!=old['root']
                or ctx.get('directory')!=str(folder) or ctx.get('sources')!=old['execution_sources']
                or ctx.get('plan_sha256')!=old['sha256'] or set(workers)!={'backtime','bpp'}
                or any(type(pid) is not int or pid<=0 for pid in workers.values())
                or type(ctx.get('parent')) is not int or ctx['parent']<=0
                or len(set(workers.values())|{ctx['parent']})!=3):
            raise ValueError('Foreign/stale failed Prepare context')
        result.append(ctx)
    if not result:
        raise ValueError('No failed Prepare to recover')
    return result


def authenticate_process(r,row,ctx,old,role=None):
    """Refuse PID reuse/unrelated owners before sending a failure notification."""
    values=handover.native.argv_windows(row.get('CommandLine') or '')
    expected={'--data-home':old['data_home'],'--backtime-root':old['old_root'],
              '--outdir':old['root']}
    if role is not None:
        expected.update({'--role':role,'--context':str(Path(ctx['directory'])/'workers.json')})
    if len(values)!=4+2*(len(expected)+1) or values[1:3]!=['-B','-u']:
        raise ValueError('Not the owned failed Prepare process; nothing signalled')
    opts=dict(zip(values[4::2],values[5::2]))
    script=Path(values[3])
    r.reject_links(script)
    if (len(opts)!=len(expected)+1 or set(opts)!=set(expected)|{'--mode'}
            or Path(values[0]).resolve()!=Path(sys.executable).resolve()
            or Path(row.get('ExecutablePath') or '').resolve()!=Path(sys.executable).resolve()
            or script.name!=WRAPPER or r.sha(script)!=old['execution_sources'][WRAPPER]
            or any(opts[k]!=v for k,v in expected.items())
            or opts['--mode'] not in (('_smoke',) if role else ('prepare','switch-and-launch'))
            or row.get('ProcessId')!=(ctx['workers'][role] if role else ctx['parent'])
            or (role is not None and row.get('ParentProcessId')!=ctx['parent'])):
        raise ValueError('Foreign/reused Prepare process identity; nothing signalled')


def cancel_failed_smokes(r,old,contexts,timeout=60):
    """Graceful failure through the existing peer_failed channel, never kill."""
    pids=set()
    # Authenticate ALL live rows before writing ANY cancellation signal.
    for ctx in contexts:
        for role,pid in list(ctx['workers'].items())+[(None,ctx['parent'])]:
            pids.add(pid)
            row=handover.native.process_row(pid)
            if row is not None:
                authenticate_process(r,row,ctx,old,role)
    for ctx in contexts:
        signal=Path(ctx['directory'])/'peer_failed.json'
        r.reject_links(signal)
        if not signal.exists():
            r.write(signal,dict(role='operator_recovery',exit_code=1,
                reason='EXPLICIT_CANCEL_FAILED_PREPARE_ROLE_MISMATCH',
                context_sha256=ctx['sha256'],no_forced_process_stop=True))
    print('RECOVERY_CANCEL_FAILED_PREPARE: failure notification only; '
          'waiting for authenticated children/controller to drain; no process killed.',flush=True)
    began=time.monotonic()
    while any(handover.native.process_row(pid) is not None for pid in sorted(pids)):
        if time.monotonic()-began>timeout:
            raise TimeoutError('Old Prepare has not drained; no process killed or new job launched')
        time.sleep(.5)


def recover(engine,r,plan,master,args,env):
    if os.name!='nt':
        raise ValueError('Recovery requires Windows exact process identity checks')
    root=Path(plan['root'])
    r.reject_links(root)
    saved=r.read(root/'shared.plan.json')
    old=legacy_plan(r,plan,saved.get('execution_sources',{}).get(WRAPPER))
    if r.read(root/'shared.plan.json')!=old:
        raise ValueError('Recovery supports ONLY the exact8551376 failed Prepare plan')
    engine.pause_receipt(r,old)
    contexts=failed_contexts(r,old)
    cancel_failed_smokes(r,old,contexts)
    from run_mmfi_tables import _cell_lock
    with _cell_lock(root/'.controller'), _cell_lock(args.backtime_root/'.controller'), \
            _cell_lock(args.backtime_root/'p3s3/.controller'):
        if r.read(root/'shared.plan.json')!=old or failed_contexts(r,old)!=contexts:
            raise ValueError('Failed Prepare changed during recovery; nothing rebound')
        engine.pause_receipt(r,old)
        args_original=engine.original_adapter()
        args_original.require_idle()
        point=handover.checkpoint(r,master['plans']['p3s3'],env)
        paused=r.read(root/'pause.json')
        if any(point.get(k)!=paused['after'].get(k) for k in ('phase','next_epoch','path','sha256')):
            raise ValueError('Native P3S3 advanced/changed since pause; no automatic recovery')
        archive=root/'recovery_backups'/uuid.uuid4().hex
        r.reject_links(archive); archive.mkdir(parents=True,exist_ok=False)
        source=root/'shared.plan.json'; target=archive/'shared.plan.json'
        with source.open('rb') as reader, target.open('xb') as writer:
            shutil.copyfileobj(reader,writer)
        if r.sha(source)!=r.sha(target):
            raise ValueError('Failed plan backup differs; nothing rebound')
        receipt=dict(status='EXPLICIT_ROLE_FIX_PREPARE_REBOUND_NOT_TRAINED',
            prior_plan_sha256=old['sha256'],new_plan_sha256=plan['sha256'],
            prior_plan_backup=str(target),prior_plan_file_sha256=r.sha(target),
            pause_sha256=r.sha(root/'pause.json'),native_resume_point=point,
            failed_context_sha256=[ctx['sha256'] for ctx in contexts],
            no_forced_process_stop=True,no_checkpoint_import=True)
        r.write(archive/'recovery.json',receipt)
        r.write(root/'shared.plan.json',plan,immutable=False)
        r.write(root/'recovery.json',receipt)
    print('RECOVERY_PLAN_REBOUND; failed logs/plan/pause/backup preserved. '
          'New concurrent Prepare REQUIRED before native P3S3 resume + Bpp scratch50.',flush=True)
