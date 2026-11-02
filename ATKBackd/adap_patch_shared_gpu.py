"""Narrow opt-in sharing guard: one P1S1 + the original P3S3 Adap-Patch.

Read-only process coordination, not an adversarial authentication boundary or
an OS GPU reservation. No process termination, elevation, or recipe changes.
The P3S3 source checkout and scientific artifacts remain read-only.
"""
from __future__ import annotations

import json
import ntpath
import os
from pathlib import Path
import subprocess


P3_RUN = 'mmfi_p3s3_hpeli_adap_patch_s42_4090_v1'
P3_ENTRY = 'run_mmfi_p3s3_adap_patch_4090.py'
P1_ENTRY = 'run_mmfi_p1s1_adap_patch_4090.py'
P3_RUNNER_SHA256 = 'efe69b53d195fc98db45a674359e9bed41ffce516142d9e3e6f2ba05e633476e'


def winpath(value):
    return ntpath.normcase(ntpath.normpath(str(value).replace('/', '\\')))


def process_argv(record):
    from run_mmfi_p1s1_trojannn import parse_windows_commandline
    try:
        return parse_windows_commandline(record.get('CommandLine'))
    except (ValueError, TypeError):
        return []


def invocation(record, plan, entry, *, peer=False):
    """Strictly recognize a worker/controller; never trust the Python name alone."""
    args=process_argv(record)
    if (type(record.get('ProcessId')) is not int or record['ProcessId']<=0
            or type(record.get('ParentProcessId')) is not int
            or str(record.get('Name','')).lower()!='python.exe'
            or len(args)<3 or args[1]!='-u'
            or not ntpath.isabs(args[0]) or not record.get('ExecutablePath')
            or winpath(args[0])!=winpath(record['ExecutablePath'])):
        return None
    home=plan['home']
    script=(ntpath.join(ntpath.dirname(home),'code_adap_patch_p3s3','ATKBackd',entry)
            if peer else ntpath.join(str(Path(__file__).parent),entry))
    if winpath(args[2])!=winpath(script):
        return None
    tail=args[3:]
    if tail.count('--allow-concurrent-p3s3')>1:
        return None
    sharing='--allow-concurrent-p3s3' in tail
    if sharing:
        tail=[value for value in tail if value!='--allow-concurrent-p3s3']
    if len(tail)%2 or any(not value.startswith('--') for value in tail[::2]):
        return None
    names=tail[::2]
    if len(set(names))!=len(names):
        return None
    options=dict(zip(names,tail[1::2]))
    expected={'--data-home','--outdir','--device','--jobs-per-gpu','--mode'}
    role=options.get('--mode')
    if role=='cell':
        expected.add('--work')
        if options.get('--work') not in (('train',) if peer else ('train','smoke','execution')):
            return None
    elif role not in (('launch',) if peer else ('launch','prepare','check')):
        return None
    if set(options)!=expected or (peer and sharing):
        return None
    output=ntpath.join(home,'runs',P3_RUN) if peer else plan['outdir']
    if (winpath(options['--data-home'])!=winpath(home)
            or winpath(options['--outdir'])!=winpath(output)
            or options['--device']!='cuda:0' or options['--jobs-per-gpu']!='1'):
        return None
    if not peer and sharing!=plan['execution_policy']['allow_concurrent_p3s3']:
        return None
    return {'args':args,'options':options,'role':role,'script':script}


def peer_pids(records, plan):
    """Authorize at most one linear venv launcher/worker chain, never two jobs."""
    if not plan['execution_policy']['allow_concurrent_p3s3']:
        return set()
    candidates={r['ProcessId']:(r,invocation(r,plan,P3_ENTRY,peer=True))
                for r in records if invocation(r,plan,P3_ENTRY,peer=True)}
    workers={pid:row for pid,row in candidates.items() if row[1]['role']=='cell'}
    if not workers:
        return set()
    roots=[pid for pid,(record,_) in workers.items() if record['ParentProcessId'] not in workers]
    if len(roots)!=1:
        raise ValueError('More than one P3S3 worker chain; no additional job authorized')
    root=roots[0]
    envs={winpath(ntpath.join(plan['home'],suffix)) for suffix in (
        r'envs\dose-backdoor\python.exe',r'envs\dose-backdoor\Scripts\python.exe',
        r'envs\mmfi-p3\Scripts\python.exe')}
    if winpath(workers[root][1]['args'][0]) not in envs:
        raise ValueError('P3S3 root is not the approved existing venv invocation')
    seen=set(); current=root
    while current is not None:
        if current in seen or len(seen)>=3:
            raise ValueError('Unexpected P3S3 venv process ancestry')
        seen.add(current)
        children=[pid for pid,(record,_) in workers.items() if record['ParentProcessId']==current]
        if len(children)>1:
            raise ValueError('Multiple P3S3 child workers; sharing blocked')
        current=children[0] if children else None
    if seen!=set(workers):
        raise ValueError('Unrelated P3S3 workers; sharing blocked')
    return seen


def own_ancestors(records, plan, own_pid):
    by_pid={r['ProcessId']:r for r in records}
    allowed={own_pid}; current=by_pid.get(own_pid)
    while current:
        parent=by_pid.get(current['ParentProcessId'])
        if not parent or parent['ProcessId'] in allowed or not invocation(parent,plan,P1_ENTRY):
            break
        allowed.add(parent['ProcessId']); current=parent
    return allowed


def sharing_conflicts(report, records, plan, own_pid):
    """Pure process-snapshot classifier, including known pre-CUDA workers."""
    from run_mmfi_p1s1_trojannn import classify_gpu
    peers=peer_pids(records,plan)
    ancestors=own_ancestors(records,plan,own_pid)
    names={r['ProcessId']:r['Name'] for r in records}
    by_pid={r['ProcessId']:r for r in records}
    controllers=set()
    for pid in peers:
        parent=by_pid.get(by_pid[pid]['ParentProcessId'])
        while parent and parent['ProcessId'] not in controllers:
            matched=invocation(parent,plan,P3_ENTRY,peer=True)
            if not matched or matched['role']!='launch':
                break
            controllers.add(parent['ProcessId'])
            parent=by_pid.get(parent['ParentProcessId'])
    conflicts=[row for row in classify_gpu(report,names,own_pid)
               if row[0] not in peers]
    # Even if currently CPU-loading, a second known training job is not ignored.
    for record in records:
        pid=record['ProcessId']
        if pid in ancestors or pid in peers:
            continue
        args=process_argv(record)
        if not args or str(record.get('Name','')).lower() not in ('python.exe','pythonw.exe'):
            continue
        entry=ntpath.basename(args[2]).lower() if len(args)>2 and args[1]=='-u' else ''
        if not (entry.startswith(('run_mmfi','run_piw','run_a1','run_hpeli'))
                or entry in ('train.py','train_backdoor.py')):
            continue
        # One original P3S3 controller is a CPU-only ancestor, not a worker.
        if pid in controllers and pid not in {p for p,_ in conflicts}:
            continue
        if pid not in {p for p,_ in conflicts}:
            conflicts.append((pid,record.get('CommandLine')))
    return conflicts, peers


def verify_peer_source(records, peers, plan):
    """Read-only pin validation of the original running checkout + resolved plan."""
    if not peers:
        return
    import hashlib
    entry=Path(plan['home']).parent/'code_adap_patch_p3s3'/'ATKBackd'/P3_ENTRY
    if hashlib.sha256(entry.read_bytes()).hexdigest()!=P3_RUNNER_SHA256:
        raise ValueError('Original P3S3 peer source pin differs; no sharing')
    path=Path(plan['home'])/'runs'/P3_RUN/'plan.json'
    saved=json.loads(path.read_text(encoding='utf-8-sig'))
    body={k:v for k,v in saved.items() if k!='plan_sha256'}
    claimed=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    cfg=saved.get('cfg',{})
    if (saved.get('plan_sha256')!=claimed or saved.get('profile')!=P3_RUN
            or winpath(saved.get('home',''))!=winpath(plan['home'])
            or winpath(saved.get('outdir',''))!=winpath(path.parent)
            or saved.get('jobs_per_gpu')!=1 or saved.get('device')!='cuda:0'
            or cfg.get('mmfi_protocol')!='protocol3' or cfg.get('mmfi_setting')!='s3'
            or cfg.get('seed')!=42 or cfg.get('model')!='hpeli'
            or saved.get('sources',{}).get(P3_ENTRY)!=P3_RUNNER_SHA256):
        raise ValueError('P3S3 peer plan/source/protocol mismatch; no sharing')


def require_allowed_gpu(plan):
    if os.name!='nt':
        if plan['execution_policy']['allow_concurrent_p3s3']:
            raise ValueError('Exact P3S3 sharing is Windows-only')
        from run_mmfi_p1s1_trojannn import require_idle
        return require_idle()
    report=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=30)
    script=("$ErrorActionPreference='Stop';"
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
        "@(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine) | ConvertTo-Json -Compress")
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],
        capture_output=True,encoding='utf-8-sig',check=True,timeout=30)
    records=json.loads(result.stdout.strip() or '[]')
    if isinstance(records,dict):
        records=[records]
    if (not isinstance(records,list) or any(not isinstance(r,dict)
            or type(r.get('ProcessId')) is not int or type(r.get('ParentProcessId')) is not int
            or not isinstance(r.get('Name'),str) for r in records)
            or len({r['ProcessId'] for r in records})!=len(records)):
        raise ValueError('Invalid Windows process snapshot; sharing blocked')
    conflicts,peers=sharing_conflicts(report.stdout,records,plan,os.getpid())
    if conflicts:
        raise ValueError(f'Other/unknown GPU or pending training work; no process stopped: {conflicts}')
    verify_peer_source(records,peers,plan)
    if peers:
        print(f'AUTHORIZED_P3S3_ADAP_PATCH PIDs={sorted(peers)}; at most one P1S1 + one P3S3. '
              'No speedup/VRAM guarantee; peer left unchanged.',flush=True)
    return sorted(peers)
