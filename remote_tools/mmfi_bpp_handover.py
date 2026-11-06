"""Authenticate a legacy or dose-shared P3S3 owner; never stop arbitrary PIDs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import uuid

import mmfi_shared_pause as native
checkpoint,checkpoint_choice=native.checkpoint,native.checkpoint_choice


def private(filename):
    spec=importlib.util.spec_from_file_location('_bpp_handover_'+uuid.uuid4().hex,
        Path(__file__).with_name(filename))
    result=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=result
    spec.loader.exec_module(result)
    return result


def discover():
    script=("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { "
        "($_.CommandLine -like '*Run-MMFiBackTimeTwoSplits4090.py*' "
        "-and $_.CommandLine -match '--mode\\s+_worker' -and $_.CommandLine -match '--split\\s+p3s3') "
        "-or ($_.CommandLine -like '*Run-MMFiBackTimeDoseShared4090.py*' "
        "-and $_.CommandLine -match '--mode\\s+_worker' -and $_.CommandLine -match '--role\\s+backtime') "
        '} | Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress')
    result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',script],
        capture_output=True,text=True,check=True,timeout=30)
    value=json.loads(result.stdout) if result.stdout.strip() else []
    return value if isinstance(value,list) else [value]


def shared_options(row, *, child):
    values=native.argv_windows(row.get('CommandLine') or '')
    if len(values)<4 or values[1:3]!=['-B','-u']:
        raise ValueError('Not an exact Python shared worker')
    args=values[4:]
    allowed={'--mode','--data-home','--backtime-root','--outdir'}
    if child:
        allowed |= {'--role','--context'}
    if len(args)!=2*len(allowed) or set(args[::2])!=allowed:
        raise ValueError('Unknown shared process arguments')
    opts=dict(zip(args[::2],args[1::2]))
    if ((child and (opts['--mode']!='_worker' or opts['--role']!='backtime'))
            or (not child and opts['--mode'] not in ('launch','switch-and-launch'))):
        raise ValueError('Only the running BackTime training owner may be handed over')
    return Path(values[0]),Path(values[3]),opts


def completed_doses(runtime,old,shared_plan):
    """All final49/stochastic RNG/evaluation/NPZ bindings, no training/CUDA."""
    ref=old.dose.reference()
    result={}
    for cell in shared_plan['cells']:
        old.dose.validate_resume(cell)
        record=runtime.read(Path(cell['ckpt_dir'])/'three_tables_inputs.json')
        if (any(record[k]!=ref['ordered_inputs'][k] for k in ('train','eval'))
                or record['action_file_sha256']!=ref['action_sha256']):
            raise ValueError('Completed dose identities/prior differ from old T2')
        row=old.dose.evaluator().load_evaluation(dict(cell,verified_inputs=record),cell['artifact_dir'])
        if row is None:
            raise ValueError('Both1%/5% must have bound final49 evaluations before slot replacement')
        result[cell['key']]=dict(record_sha256=row['record_sha256'],
            checkpoint_sha256=runtime.sha(Path(cell['ckpt_dir'])/'checkpoint.pt'))
    if set(result)!={'rho0p01','rho0p05'}:
        raise ValueError('Expected the original two completed dose cells')
    return result


def verify_owner(runtime,original,master,root,home,pid):
    row=native.process_row(pid)
    if row is None:
        raise ValueError('BackTime already exited; nothing stopped')
    if 'Run-MMFiBackTimeTwoSplits4090.py' in (row.get('CommandLine') or ''):
        return native.verify_owner(runtime,original,master,root,home,pid)
    exe,script,opts=shared_options(row,child=True)
    old=private('Run-MMFiBackTimeDoseShared4090.py')
    if (exe.resolve()!=Path(sys.executable).resolve()
            or Path(row.get('ExecutablePath') or '').resolve()!=exe.resolve()
            or script.name!='Run-MMFiBackTimeDoseShared4090.py'
            or runtime.sha(script)!=runtime.sha(old.__file__)
            or Path(opts['--data-home']).resolve()!=home.resolve()
            or Path(opts['--backtime-root']).resolve()!=root.resolve()):
        raise ValueError('Foreign shared worker/executable/source/root; nothing stopped')
    shared_root=Path(opts['--outdir']).resolve()
    expected=old.build_plan(runtime,master,home,root,shared_root)
    if runtime.read(shared_root/'shared.plan.json')!=expected:
        raise ValueError('Old shared plan/source differs; no live source edits allowed')
    env=runtime.read(root/'two_splits.gate.json')['smokes']['p3s3']['execution']
    old.gate_check(runtime,expected,env)
    filename=Path(opts['--context'])
    runtime.reject_links(filename)
    ctx=runtime.read(filename)
    unsigned=dict(ctx); claimed=unsigned.pop('sha256',None)
    if (claimed!=runtime.digest(unsigned) or ctx.get('profile')!=old.PROFILE
            or ctx.get('action')!='launch' or ctx.get('parent')!=row['ParentProcessId']
            or ctx.get('root')!=str(shared_root) or ctx.get('plan_sha256')!=expected['sha256']
            or ctx.get('sources')!=expected['execution_sources']
            or set(ctx.get('workers',{}))!={'backtime','dose'}
            or ctx['workers']['backtime']!=pid or len(set(ctx['workers'].values()))!=2
            or filename!=Path(ctx.get('directory',''))/'workers.json'
            or filename.parent.parent!=shared_root/'executions'):
        raise ValueError('Foreign/stale previous shared worker context')
    if native.process_row(ctx['workers']['dose']) is not None:
        raise ValueError('Old dose worker PID is still active; wait for it to exit; nothing stopped')
    parent=native.process_row(row['ParentProcessId'])
    if parent is None:
        raise ValueError('Previous shared parent is missing; explicit recovery required')
    pexe,pscript,popts=shared_options(parent,child=False)
    if (pexe.resolve()!=exe.resolve() or pscript.resolve()!=script.resolve()
            or any(Path(popts[k]).resolve()!=Path(opts[k]).resolve()
                   for k in ('--data-home','--backtime-root','--outdir'))):
        raise ValueError('Previous shared parent identity differs')
    return dict(worker=row,parent=parent,context=str(filename),context_sha256=runtime.sha(filename),
                completed_doses=completed_doses(runtime,old,expected))


def pause(runtime,original,master,root,home,output,env,pid):
    # Private globals only: reuse the Windows handle/backup/pause protocol
    # already tested in6556c43; replace only its owner-authentication callback.
    module=private('mmfi_shared_pause.py')
    module.verify_owner=verify_owner
    return module.pause(runtime,original,master,root,home,output,env,pid)
