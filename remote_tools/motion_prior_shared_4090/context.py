"""Bound execution adapter: one A1 prior and one BadNets victim on Van4090.

No native source, prepared gate or training recipe is changed. The parent owns
both native output-root locks; each worker retains its native per-role GPU lock.
Only exact descendants of the current creation-time-bound controller may share.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import ntpath
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
    MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode=True
PROFILE='mmfi_a1_badnets_shared4090_v1'
ROLES={'a1':('cross','nod'),'badnets':('wiflow','metafiplusplus')}
RUNNERS={'a1':'run_mmfi_a1_motion_priors_4090.py','badnets':'run_mmfi_badnets_two_models_4090.py'}
RUNNER_SHA={'a1':'d6a777fb151ccf3216d0d0281f099ec6563da6c562cd473e3aef3985d529f262',
    'badnets':'2255b294d88490b75eb31b332c24f4f66c4cb0eb6e254890ea61cb0e76ee51a7'}
PLANS={'a1':'motion_priors.plan.json','badnets':'badnets_two_models.plan.json'}
PHASES=('audit','smoke','run','export')
DESKTOP={'explorer.exe','dwm.exe','chrome.exe','msedge.exe','msedgewebview2.exe',
    'code.exe','lockapp.exe','searchhost.exe','startmenuexperiencehost.exe',
    'shellexperiencehost.exe','shellhost.exe','textinputhost.exe','systemsettings.exe',
    'applicationframehost.exe','windowsterminal.exe','taskmgr.exe','snippingtool.exe',
    'crossdeviceresume.exe','copilot.exe','copilotapp.exe','copilotapphost.exe',
    'phoneexperiencehost.exe','teams.exe','ms-teams.exe','teamviewer.exe','logonui.exe',
    'anydesk.exe','system'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _path(value):
    if (not isinstance(value,str) or not value or any(c in value for c in '\r\n\x00"')
            or value.startswith(('\\\\','//'))):
        raise ValueError('Only ordinary absolute local execution paths are allowed')
    return ntpath.normcase(ntpath.normpath(value.replace('/','\\')))


def _absolute(value):
    _path(value)
    path=Path(value)
    if not path.is_absolute():raise ValueError('Absolute execution path required')
    return path.resolve()


def _argv(line):
    token=r'(?:"[^"\r\n]*"|[^\s"\r\n]+)'
    if (not isinstance(line,str) or any(ord(c)<32 for c in line)
            or not re.fullmatch(r'\s*'+token+r'(?:\s+'+token+r')*\s*',line)):
        raise ValueError('Cannot parse exact Windows process command line')
    return [m.group()[1:-1] if m.group().startswith('"') else m.group()
            for m in re.finditer(token,line)]


def _signature(record):
    if (not isinstance(record,dict) or type(record.get('ProcessId')) is not int
            or record['ProcessId']<=0 or type(record.get('ParentProcessId')) is not int
            or record['ParentProcessId']<0 or str(record.get('Name','')).lower()!='python.exe'
            or not isinstance(record.get('CreationDate'),str) or not record['CreationDate']):
        raise ValueError('Process needs PID, parent, Python executable, argv and creation time')
    return (record['ProcessId'],record['ParentProcessId'],record['CreationDate'],
        _path(record.get('ExecutablePath')),tuple(_argv(record.get('CommandLine'))))


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def gate_path(request,role,cell):
    return Path(request[role+'_out'])/cell/'seed42'/('a1' if role=='a1' else 'badnets')/'gpu_smoke.json'


def expected_bindings(request):
    result=set()
    for role,cells in ROLES.items():
        root=Path(request[role+'_out'])
        result.update({(root/PLANS[role]).resolve(),(root/'inputs.json').resolve()})
        result.update(gate_path(request,role,cell).resolve() for cell in cells)
    return result


def validate_request(request):
    if not isinstance(request,dict):raise ValueError('Execution request must be an object')
    unsigned=dict(request);claimed=unsigned.pop('request_sha256',None)
    if request.get('schema')!=1 or request.get('profile')!=PROFILE or claimed!=digest(unsigned):
        raise ValueError('Changed/foreign shared execution request')
    for key in ('home','python','a1_repo','a1_out','badnets_repo','badnets_out','root'):
        _absolute(request.get(key))
    if _path(str(Path(sys.executable).resolve()))!=_path(str(Path(request['python']).resolve())):
        raise ValueError('Use exactly the request-bound Python environment')
    root=Path(request['root']).resolve();home=Path(request['home']).resolve()
    if root==home/'tools' or home/'tools' not in root.parents:
        raise ValueError('Shared root must be a dedicated folder below DataHome/tools')
    for role in ROLES:
        output=Path(request[role+'_out']).resolve();repo=Path(request[role+'_repo']).resolve()
        if output==home/'runs' or home/'runs' not in output.parents:
            raise ValueError('Native output must be a dedicated DataHome/runs descendant')
        if root==output or output in root.parents or root in output.parents or root==repo or repo in root.parents or root in repo.parents:
            raise ValueError('Execution receipts must not modify native sources or outputs')
    a1_out=Path(request['a1_out']).resolve();badnets_out=Path(request['badnets_out']).resolve()
    if a1_out==badnets_out or a1_out in badnets_out.parents or badnets_out in a1_out.parents:
        raise ValueError('Two scientific outputs must remain separate')
    if Path(request['a1_repo']).resolve()==Path(request['badnets_repo']).resolve():
        raise ValueError('Keep the frozen BadNets checkout separate from the new A1 checkout')
    required={Path(__file__).resolve(),Path(__file__).resolve().with_name('controller.py')}
    sources=request.get('source_sha256')
    if not isinstance(sources,dict) or {_absolute(p) for p in sources}!=required:
        raise ValueError('Bind exactly both external execution sources')
    for path,expected in sources.items():
        if sha(path)!=expected:raise ValueError('External execution source changed')
    bindings=request.get('bindings')
    if not isinstance(bindings,dict) or {_absolute(p) for p in bindings}!=expected_bindings(request):
        raise ValueError('Bind all two native plans, two inputs and four original CUDA gates')
    for path,expected in bindings.items():
        if sha(path)!=expected:raise ValueError(f'Original plan/input/gate changed: {path}')
    return request


def read_request(path):return validate_request(_read(path))


def validate_session(path,request):
    path=_absolute(str(path));root=Path(request['root']).resolve()
    if root/'sessions' not in path.parents:raise ValueError('Session must be below bound root/sessions')
    session=_read(path);unsigned=dict(session);claimed=unsigned.pop('session_sha256',None)
    if (session.get('schema')!=1 or session.get('request_sha256')!=request['request_sha256']
            or claimed!=digest(unsigned)):
        raise ValueError('Changed/foreign controller session')
    signature=_signature(session.get('controller'))
    args=list(signature[-1])
    if (len(args)<3 or args[1]!='-u'
            or _path(args[2])!=_path(str(Path(__file__).resolve().with_name('controller.py')))
            or _path(args[0]) not in {_path(request['python']),signature[3]}):
        raise ValueError('Session controller must be the exact bound external coordinator')
    return session


def process_records():
    if os.name!='nt':raise ValueError('This bounded sharing policy requires real Windows CIM identities')
    script=('[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); '
        '@(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,'
        'ExecutablePath,CommandLine,@{Name="CreationDate";Expression={'
        '$_.CreationDate.ToUniversalTime().ToString("o")}}) | ConvertTo-Json -Compress')
    result=subprocess.run(['powershell','-NoProfile','-Command',script],check=True,
        capture_output=True,encoding='utf-8',timeout=30)
    records=json.loads(result.stdout.lstrip('\ufeff'))
    return records if isinstance(records,list) else [records]


def _snapshot(records):
    if not isinstance(records,list):raise ValueError('Invalid CIM snapshot')
    result={}
    for record in records:
        if (not isinstance(record,dict) or type(record.get('ProcessId')) is not int
                or record['ProcessId']<0 or record['ProcessId'] in result):
            raise ValueError('Missing/duplicate live process identity')
        result[record['ProcessId']]=record
    return result


def _ancestors(pid,records):
    result=set()
    while pid in records and pid!=0:
        if pid in result:raise ValueError('Cyclic process ancestry')
        result.add(pid);parent=records[pid].get('ParentProcessId')
        if type(parent) is not int or parent<0:raise ValueError('Missing process parent')
        pid=parent
    return result


def _pending(record):
    if str(record.get('Name','')).lower() not in ('python.exe','pythonw.exe'):return False
    try:args=_argv(record.get('CommandLine'))
    except ValueError:return False  # unknown GPU records are still rejected separately
    for value in args[1:]:
        name=ntpath.basename(value).lower()
        if (name.endswith('.py') and name.startswith(('run_mmfi','run_piw','run_a1','run_hpeli'))
                or name in ('train.py','train_backdoor.py','eval_offsets.py')
                or name in ('context.py','controller.py') and 'shared_4090' in value.lower()):return True
    return False


def _query_gpu():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=30)
    return result.stdout


def _gpu_conflicts(report,records,allowed):
    conflicts=[]
    for row in csv.reader(io.StringIO(report)):
        if not row:continue
        if len(row)!=2 or not row[0].strip().isdigit():raise ValueError('Cannot identify GPU compute process')
        pid=int(row[0]);name=str(records.get(pid,{}).get('Name') or row[1].strip())
        if pid in allowed or ntpath.basename(name.replace('/','\\')).lower() in DESKTOP:continue
        conflicts.append((pid,name))
    return conflicts


def require_exclusive_idle():
    records=_snapshot(process_records());own=os.getpid()
    if own not in records:raise ValueError('Own process is absent from CIM snapshot')
    permitted=_ancestors(own,records)
    conflicts=_gpu_conflicts(_query_gpu(),records,permitted)
    conflicts += [(pid,r.get('CommandLine')) for pid,r in records.items() if pid not in permitted and _pending(r)]
    if conflicts:raise ValueError(f'Other/native GPU or pending experiment work active: {conflicts}; no process stopped')


def _worker(record,request,request_path,session_path,controller,records):
    """Return exact role invocation, or None for an unrelated process."""
    if str(record.get('Name','')).lower() not in ('python.exe','pythonw.exe'):return None
    try:args=_argv(record.get('CommandLine'))
    except ValueError:return None
    context_path=_path(str(Path(__file__).resolve()))
    if len(args)<3 or args[1]!='-u' or _path(args[2])!=context_path:return None
    if len(args) not in (13,15):raise ValueError('Foreign context worker argv')
    expected=['--request',str(request_path),'--session',str(session_path),'--role',args[8],
              '--phase',args[10],'--cell',args[12]]
    actual=args[3:13]
    if (actual[::2]!=expected[::2] or _path(actual[1])!=_path(expected[1])
            or _path(actual[3])!=_path(expected[3]) or actual[5:]!=expected[5:]):
        raise ValueError('Context worker has a foreign request/session or argument order')
    role,phase,cell=args[8],args[10],args[12]
    if role not in ROLES or phase not in PHASES or cell not in ROLES[role]:
        raise ValueError('Context worker has an unauthorized role/cell/phase')
    pair=None
    if len(args)==15:
        if args[13]!='--pair' or phase!='smoke':raise ValueError('Pair is allowed only for shared smokes')
        pair=_absolute(args[14])
        if Path(request['root']).resolve() not in pair.parents:raise ValueError('Pair path escapes execution root')
    signature=_signature(record)
    exes={_path(request['python']),_path(controller['ExecutablePath']),_path(_argv(controller['CommandLine'])[0])}
    if signature[3] not in exes or _path(args[0]) not in exes:
        raise ValueError('Context worker Python environment differs from controller')
    if controller['ProcessId'] not in _ancestors(record['ProcessId'],records):
        raise ValueError('Context worker is not a descendant of the bound controller')
    return role,phase,cell,str(pair) if pair else None,tuple(args[1:])


def scoped_processes(request,session,records,request_path,session_path,own_pid):
    """Pure classifier: root identity, exact argv/ancestry and 1 worker/role."""
    rows=_snapshot(records);controller=session['controller'];pid=controller['ProcessId']
    if pid not in rows or _signature(rows[pid])!=_signature(controller):
        raise ValueError('Bound controller exited, changed or its PID was reused')
    if own_pid not in rows or pid not in _ancestors(own_pid,rows):
        raise ValueError('Current worker is not owned by the bound controller')
    permitted=_ancestors(pid,rows);workers={}
    for child_id,record in rows.items():
        worker=_worker(record,request,request_path,session_path,controller,rows)
        if worker is not None:workers[child_id]=worker
    if own_pid not in workers and own_pid!=pid:
        raise ValueError('Current process is not an exact bound context worker')
    counts={role:[] for role in ROLES}
    for child_id,worker in workers.items():
        record=rows[child_id];parent=record['ParentProcessId']
        # Windows venv launcher + base Python are one logical invocation only
        # when the complete suffix, role, phase, cell and pair are identical.
        if parent in workers and workers[parent]==worker:
            continue
        counts[worker[0]].append(child_id)
    if any(len(values)>1 for values in counts.values()):
        raise ValueError('At most one logical worker per role is authorized')
    permitted.update(workers)
    conflicts=[(p,r.get('CommandLine')) for p,r in rows.items() if p not in permitted and _pending(r)]
    if conflicts:raise ValueError(f'Native/foreign pending experiment processes remain: {conflicts}')
    return permitted,counts


def require_scoped_gpu(request,session):
    request_path=session.get('_request_path');session_path=session.get('_session_path')
    if not request_path or not session_path:raise ValueError('Session lacks its validated file paths')
    if read_request(request_path)!=request:raise ValueError('Request file changed during execution')
    saved=validate_session(session_path,request)
    if saved!={k:v for k,v in session.items() if not k.startswith('_')}:
        raise ValueError('Session file changed during execution')
    records=process_records();allowed,_=scoped_processes(request,session,records,
        request_path,session_path,os.getpid())
    conflicts=_gpu_conflicts(_query_gpu(),_snapshot(records),allowed)
    if conflicts:raise ValueError(f'Unknown GPU work active: {conflicts}; no process stopped')


def import_runner(request,role):
    path=Path(request[role+'_repo'])/'ATKBackd'/RUNNERS[role]
    if sha(path)!=RUNNER_SHA[role]:raise ValueError('Prepared native scientific runner changed')
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('_shared_motion_'+role,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=path.resolve():raise ValueError('Foreign native import context')
    return module


def native_plan(request,role,run):
    root=Path(request[role+'_out']);saved=run.read(root/PLANS[role])
    if role=='a1':
        if saved.get('priors')!=['cross','nod']:raise ValueError('Shared A1 queue is exactly cross then nod')
        plan=run.build_plan(request['home'],root,saved['priors'],saved['actions'],saved['approve_action_labels'])
    else:plan=run.build_plan(request['home'],root,'cuda:0')
    if saved!=plan:raise ValueError('Native scientific source/runtime/plan changed')
    run.verify_inputs(plan);run.gate_check(plan)
    return plan


def _cell(plan,role,cell):
    return next(c for c in plan['cells'] if c['prior' if role=='a1' else 'backbone']==cell)


def _checkpoint(run,cell,plan=None):
    path=Path(cell['run_dir'])/'checkpoint.pt'
    if not path.exists():
        if cell.get('backbone')=='wiflow':
            raise ValueError('Existing WiFlow checkpoint is required; no epoch-zero fallback')
        started=('execution_binding.json','three_tables_inputs.json','eval_cache.json',
                 'config.resolved.yaml','poison_manifest.json','results.json')
        if any((path.parent/name).exists() for name in started):
            raise ValueError('Previously started cell lost its checkpoint; no fresh restart fallback')
        return None
    import torch
    before=path.stat();payload=path.read_bytes();after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Checkpoint changed during stable read')
    blob=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    run.validate_checkpoint(blob,cell['cfg'])
    if plan is not None:
        gate=run.read(path.parent/'gpu_smoke.json')
        expected=dict(binding=run.binding(plan,cell),execution=gate['execution'])
        if run.read(path.parent/'execution_binding.json')!=expected:
            raise ValueError('Checkpoint execution/input/source binding differs from its original gate')
        inputs=path.parent/'three_tables_inputs.json'
        if not inputs.is_file():raise ValueError('Checkpoint lacks its original training input audit')
        current=(run.current_inputs(cell) if hasattr(run,'current_inputs')
                 else run.runtime(cell['backbone']).current_inputs(cell))
        if run.read(inputs)!=current:
            raise ValueError('Checkpoint training input audit changed')
    return blob['epoch']


def _facts(value):
    if (not isinstance(value,dict) or value.get('physical_batch')!=32
            or value.get('real_train_input') is not True or value.get('poison_examples')!=8
            or value.get('parameter_updated') is not True or len(value.get('losses',[]))!=2
            or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value['losses'])
            or not isinstance(value.get('execution'),dict) or not isinstance(value.get('binding'),dict)):
        raise ValueError('Incomplete/foreign shared B32 real-TRAIN smoke facts')


def _barrier(pair,request,session,role,cell,native):
    pair=_absolute(str(pair));root=Path(request['root']).resolve()
    if root not in pair.parents:raise ValueError('Pair directory escapes bound execution root')
    other='badnets' if role=='a1' else 'a1'
    expected={'cross':'wiflow','nod':'metafiplusplus','wiflow':'cross','metafiplusplus':'nod'}[cell]
    marker=dict(request_sha256=request['request_sha256'],session_sha256=session['session_sha256'],
                role=role,cell=cell,native_sha256=digest(native))
    _write_new(pair/(role+'.ready'),marker)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if (pair/'error').exists() or (pair/(other+'.error')).exists():
            raise ValueError('Peer shared smoke failed; no training launched')
        path=pair/(other+'.ready')
        if path.is_file():
            peer=_read(path)
            if (peer.get('request_sha256')!=marker['request_sha256'] or peer.get('session_sha256')!=marker['session_sha256']
                    or peer.get('role')!=other or peer.get('cell')!=expected
                    or not re.fullmatch('[0-9a-f]{64}',str(peer.get('native_sha256','')))):
                raise ValueError('Foreign/stale paired smoke barrier')
            return {p:sha(pair/(p+'.ready')) for p in ROLES}
        time.sleep(.2)
    raise TimeoutError('Paired B32 smoke did not become simultaneously resident within 300s')


def check_smokes(request):
    result={}
    for role,cells in ROLES.items():
        for cell in cells:
            path=Path(request['root'])/'smokes'/f'{role}.{cell}.json';receipt=_read(path)
            unsigned=dict(receipt);claimed=unsigned.pop('receipt_sha256',None)
            original_gate=gate_path(request,role,cell)
            if (claimed!=digest(unsigned) or receipt.get('request_sha256')!=request['request_sha256']
                    or receipt.get('role')!=role or receipt.get('cell')!=cell
                    or receipt.get('original_gate_sha256')!=sha(original_gate)
                    or not isinstance(receipt.get('session_sha256'),str)
                    or receipt.get('paired_residency') is not True):
                raise ValueError('Changed/foreign or unpaired shared context smoke')
            facts=receipt.get('native');_facts(facts)
            original=_read(original_gate)
            if facts['binding']!=original['binding'] or facts['execution']!=original['execution']:
                raise ValueError('Shared native gate binding/execution differs from original')
            steps=2 if role=='a1' or cell=='metafiplusplus' else 1
            if facts.get('optimizer_steps')!=steps:raise ValueError('Shared smoke optimizer steps differ')
            session=validate_session(receipt.get('session_path'),request)
            if session['session_sha256']!=receipt['session_sha256']:
                raise ValueError('Shared smoke belongs to a changed session')
            pair=_absolute(receipt.get('pair'))
            if Path(request['root']).resolve() not in pair.parents or (pair/'error').exists():
                raise ValueError('Shared pair proof is outside its root or failed')
            expected_cell={'cross':'wiflow','nod':'metafiplusplus','wiflow':'cross','metafiplusplus':'nod'}[cell]
            for peer_role in ROLES:
                marker_path=pair/(peer_role+'.ready');marker=_read(marker_path)
                peer_cell=cell if peer_role==role else expected_cell
                peer_native=facts if peer_role==role else _read(Path(request['root'])/'smokes'/f'{peer_role}.{peer_cell}.json')['native']
                if (receipt.get('pair_proof',{}).get(peer_role)!=sha(marker_path)
                        or marker!=dict(request_sha256=request['request_sha256'],
                            session_sha256=receipt['session_sha256'],role=peer_role,
                            cell=peer_cell,native_sha256=digest(peer_native))):
                    raise ValueError('Shared pair residency proof changed or mismatches native facts')
            result[f'{role}.{cell}']=sha(path)
    return result


def run_phase(request,session,role,phase,cell,pair=None):
    validate_request(request);require_scoped_gpu(request,session)
    run=import_runner(request,role);plan=native_plan(request,role,run);selected=_cell(plan,role,cell)
    epoch=_checkpoint(run,selected,plan)
    if phase=='audit':
        done=run.completed(plan,selected) is not None
        print(json.dumps(dict(role=role,cell=cell,complete=done,checkpoint_epoch=epoch)),flush=True);return
    if phase=='export':
        if any(run.completed(plan,c) is None for c in plan['cells']):raise ValueError('Cannot export an incomplete role queue')
        run.export(plan);return
    from run_mmfi_tables import _cell_lock
    lock=Path(request['home'])/'runs'/('.mmfi_motion_prior_gpu0_worker' if role=='a1' else '.mmfi_badnets_gpu0_worker')
    with _cell_lock(lock):
        require_scoped_gpu(request,session)
        if phase=='smoke':
            if pair is None:raise ValueError('Shared Prepare requires an explicit paired smoke barrier')
            native_write=run.write;destination=Path(request['root'])/'smokes'/f'{role}.{cell}.json'
            original_path=gate_path(request,role,cell).resolve();original_hash=sha(original_path)
            emitted=[]
            def shared_write(path,value,**kwargs):
                if Path(path).resolve()!=original_path:
                    raise ValueError('Unexpected native write during disposable shared smoke')
                _facts(value)
                proof=_barrier(pair,request,session,role,cell,value)
                receipt=dict(schema=1,role=role,cell=cell,request_sha256=request['request_sha256'],
                    session_sha256=session['session_sha256'],original_gate_sha256=original_hash,
                    session_path=session['_session_path'],pair=str(Path(pair).resolve()),
                    pair_proof=proof,native=value,paired_residency=True,
                    note='Separate execution-context receipt; original scientific gate remains unchanged')
                receipt['receipt_sha256']=digest(receipt)
                _write_new(destination,receipt);emitted.append(destination)
            run.write=shared_write
            try:run.smoke(plan,selected)
            finally:run.write=native_write
            if len(emitted)!=1 or sha(original_path)!=original_hash:raise ValueError('Original gate changed or smoke did not attest')
            require_scoped_gpu(request,session)
            print(json.dumps(dict(role=role,cell=cell,status='shared_smoke_passed',receipt=str(destination))),flush=True);return
        check_smokes(request)
        run.run_cell(plan,selected)
        completed=run.completed(plan,selected)
        if completed is None:raise ValueError('Native worker returned without authenticated completion')
        validate_request(request);require_scoped_gpu(request,session)
        receipt=dict(schema=1,role=role,cell=cell,status='complete',request_sha256=request['request_sha256'],
            session_sha256=session['session_sha256'],evaluation_sha256=completed['record_sha256'])
        receipt['receipt_sha256']=digest(receipt)
        _write_new(Path(request['root'])/'sessions'/Path(session['_session_path']).stem/'runs'/f'{role}.{cell}.json',receipt)
        print(json.dumps(receipt),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--session',required=True)
    parser.add_argument('--role',required=True,choices=ROLES);parser.add_argument('--phase',required=True,choices=PHASES)
    parser.add_argument('--cell',required=True);parser.add_argument('--pair')
    args=parser.parse_args(argv)
    request=read_request(args.request);session=validate_session(args.session,request)
    session.update(_request_path=str(Path(args.request).resolve()),_session_path=str(Path(args.session).resolve()))
    if args.cell not in ROLES[args.role]:raise ValueError('Cell does not belong to the exact requested role')
    try:run_phase(request,session,args.role,args.phase,args.cell,args.pair)
    except BaseException as exc:
        if args.pair and args.phase=='smoke':
            pair=_absolute(args.pair)
            if Path(request['root']).resolve() in pair.parents and not (pair/(args.role+'.error')).exists():
                _write_new(pair/(args.role+'.error'),dict(error=str(exc),request_sha256=request['request_sha256']))
        raise


if __name__=='__main__':main()
