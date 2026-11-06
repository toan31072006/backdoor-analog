"""Role-barrier and explicit failed-Prepare recovery; CPU/toy, not CUDA evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'remote_tools'))
spec=importlib.util.spec_from_file_location('bpp_recovery_adapter',
    ROOT/'remote_tools/Run-MMFiBackTimeBppShared4090.py')
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
fix=a.recovery


def test_real_worker_callbacks_cross_all_three_barriers_in_two_os_children(tmp_path,monkeypatch):
    engine=a.configure(); r=engine.original_adapter().load_runtime('p3s3')
    plan=dict(root=str(tmp_path),sha256='toy-plan',execution_sources={},cells=[{'key':'toy-bpp'}])
    r.write(tmp_path/'shared.plan.json',plan)
    r.write(tmp_path/'pause.json',{})
    original_popen=subprocess.Popen
    children=[]
    code=r'''
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
repo,context,role=sys.argv[1:]
spec=importlib.util.spec_from_file_location('real_worker_test',Path(repo)/'remote_tools/Run-MMFiBackTimeBppShared4090.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
engine=a.configure(); r=engine.original_adapter().load_runtime('p3s3')
folder=Path(context).parent; root=folder.parent.parent
plan=r.read(root/'shared.plan.json'); env={'fixture':'cpu'}
stages=('models_resident','B64_backward_resident','B32_complete')
args=SimpleNamespace(mode='_smoke',role=role,data_home=root/'home',backtime_root=root/'old',outdir=root,context=Path(context))
original=SimpleNamespace(require_idle=lambda *a:None)
def backtime_smoke(runtime,plan,ctx,split):
    for stage in stages: original.barrier(ctx,stage,split)
    runtime.write(folder/'smoke.p3s3.json',{'discarded':True})
original.smoke=backtime_smoke
def bpp_smoke(cell,callback):
    for stage in stages: callback(stage)
    return {'discarded':True}
a.bpp.smoke=bpp_smoke
a.bpp.execute=lambda *a:(_ for _ in ()).throw(AssertionError('Prepare trained Bpp'))
engine.original_adapter=lambda:original
engine.old_contract=lambda *a:(r,{'plans':{'p3s3':{}}},env)
engine.build_plan=lambda *a:plan
r.execution=lambda:env
base_barrier=engine.barrier
engine.barrier=lambda r,ctx,stage,role:base_barrier(r,ctx,stage,role,timeout=15)
engine.worker(args)
'''
    def popen(command,**kw):
        context=command[command.index('--context')+1]
        role=command[command.index('--role')+1]
        child=original_popen([sys.executable,'-B','-u','-c',code,str(ROOT),context,role],**kw)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess,'Popen',popen)
    monkeypatch.setattr(a,'gate_check',lambda *args:None)
    engine.pause_receipt=lambda *args:None
    args=SimpleNamespace(outdir=tmp_path,data_home=tmp_path/'home',backtime_root=tmp_path/'old')
    a.spawn_pair(engine,args,r,plan,'prepare',{'fixture':'cpu'},None,None)
    assert len(children)==2 and all(child.returncode==0 for child in children)
    folder=next((tmp_path/'executions').iterdir())
    assert {p.name for p in folder.glob('*.ready')}=={
        stage+'.'+role+'.ready' for stage in
        ('models_resident','B64_backward_resident','B32_complete') for role in ('backtime','bpp')}
    assert not list(folder.glob('*.dose.ready'))
    assert r.read(folder/'smoke.bpp.json')['execution']=={'fixture':'cpu'}
    assert not (tmp_path/'bpp_source').exists()


def fixture(tmp_path,monkeypatch):
    engine=a.configure(); r=engine.original_adapter().load_runtime('p3s3')
    oldroot=tmp_path/'runs/mmfi_backtime_old'
    root=tmp_path/'runs'/a.PROFILE
    oldroot.mkdir(parents=True); root.mkdir()
    (oldroot/'two_splits.gate.json').write_text('{}')
    for split in ('p1s1','p3s3'):
        (oldroot/split).mkdir(); (oldroot/split/'inputs.json').write_text('{}')
    master={'sha256':'native-master','plans':{'p3s3':{'outdir':str(oldroot/'p3s3')}}}
    args=SimpleNamespace(outdir=root,backtime_root=oldroot,data_home=tmp_path/'home')
    plan=engine.build_plan(r,master,args.data_home,oldroot,root)
    old=fix.legacy_plan(r,plan)
    r.write(root/'shared.plan.json',old)
    backup=root/'pause_backups/old/checkpoint.pt'; backup.parent.mkdir(parents=True)
    backup.write_bytes(b'committed native epoch98; no toy model here')
    native=oldroot/'p3s3/backtime/checkpoint.pt'; native.parent.mkdir()
    native.write_bytes(backup.read_bytes())
    point=dict(phase='search',next_epoch=99,path=str(native),sha256=r.sha(native),backup=None)
    r.write(root/'pause.json',dict(status='PAUSED_WITH_VALID_NATIVE_CHECKPOINT',
        old_master_sha256=master['sha256'],before=dict(point,backup=str(backup)),after=point))
    directory=root/'executions/prepare_failed'; directory.mkdir(parents=True)
    ctx=dict(profile=a.PROFILE,parent=10003,root=str(root),directory=str(directory),
        workers={'backtime':10001,'bpp':10002},action='prepare',
        plan_sha256=old['sha256'],sources=old['execution_sources'])
    ctx['sha256']=r.digest(ctx); r.write(directory/'workers.json',ctx)
    (directory/'models_resident.backtime.ready').touch()
    (directory/'models_resident.dose.ready').touch()
    (directory/'bpp.stdout.log').write_bytes(b'old deadlocked log')
    monkeypatch.setattr(fix.handover.native,'process_row',lambda pid:None)
    monkeypatch.setattr(fix.handover,'checkpoint',lambda *args:copy.deepcopy(point))
    idle=[]
    engine.original_adapter=lambda:SimpleNamespace(require_idle=lambda:idle.append(True))
    return engine,r,plan,old,master,args,point,ctx,idle


@pytest.mark.skipif(os.name!='nt',reason='Explicit Windows recovery')
def test_failed_prepare_rebind_preserves_plan_pause_native_backup_and_bad_ready_files(tmp_path,monkeypatch):
    engine,r,plan,old,master,args,point,ctx,idle=fixture(tmp_path,monkeypatch)
    root=args.outdir
    tracked=[root/'shared.plan.json',root/'pause.json',Path(point['path']),
             Path(r.read(root/'pause.json')['before']['backup']),Path(ctx['directory'])/'bpp.stdout.log']
    before={str(p):p.read_bytes() for p in tracked}
    fix.recover(engine,r,plan,master,args,{})
    receipt=r.read(root/'recovery.json')
    assert r.read(root/'shared.plan.json')==plan
    assert Path(receipt['prior_plan_backup']).read_bytes()==before[str(tracked[0])]
    for path in tracked[1:]: assert path.read_bytes()==before[str(path)]
    assert r.read(Path(ctx['directory'])/'peer_failed.json')['no_forced_process_stop']
    assert not (Path(ctx['directory'])/'models_resident.bpp.ready').exists()
    assert (Path(ctx['directory'])/'models_resident.dose.ready').exists()
    assert receipt['native_resume_point']==point and idle==[True]
    assert not (root/'shared.gate.json').exists() and not (root/'bpp_source').exists()
    engine.bind_plan(r,plan)
    with pytest.raises(ValueError,match='ONLY|published'):
        fix.recover(engine,r,plan,master,args,{})


@pytest.mark.skipif(os.name!='nt',reason='Explicit Windows recovery')
@pytest.mark.parametrize('bad',['plan','gate','trained','launch','context','pause','native','live'])
def test_recovery_refuses_other_plans_production_missing_backup_and_undrained_owner(tmp_path,monkeypatch,bad):
    engine,r,plan,old,master,args,point,ctx,idle=fixture(tmp_path,monkeypatch)
    root=args.outdir
    before=(root/'shared.plan.json').read_bytes()
    if bad=='plan':
        old['cells'][0]['cfg']['rho']=.05
        r.write(root/'shared.plan.json',old,immutable=False)
        before=(root/'shared.plan.json').read_bytes()
    if bad=='gate': r.write(root/'shared.gate.json',{})
    if bad=='trained': (root/'bpp_source').mkdir()
    if bad=='launch': (root/'executions/launch_other').mkdir()
    if bad=='context':
        ctx['workers']={'backtime':10001,'dose':10002}
        r.write(Path(ctx['directory'])/'workers.json',ctx,immutable=False)
    if bad=='pause': Path(r.read(root/'pause.json')['before']['backup']).write_bytes(b'changed')
    if bad=='native':
        monkeypatch.setattr(fix.handover,'checkpoint',lambda *args:dict(point,sha256='changed'))
    if bad=='live':
        monkeypatch.setattr(fix.handover.native,'process_row',lambda pid:{'CommandLine':'foreign'})
    with pytest.raises((ValueError,KeyError)):
        fix.recover(engine,r,plan,master,args,{})
    assert (root/'shared.plan.json').read_bytes()==before
    assert not (root/'recovery.json').exists()
    assert not (root/'shared.gate.json').exists() or bad=='gate'
    assert not list(root.glob('recovery_backups/*'))


def test_new_plan_cannot_silently_overwrite_old_immutable_plan(tmp_path,monkeypatch):
    engine,r,plan,old,master,args,point,ctx,idle=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='RecoverAndLaunch'):
        engine.bind_plan(r,plan)
    assert r.read(args.outdir/'shared.plan.json')==old


@pytest.mark.skipif(os.name!='nt',reason='Explicit Windows recovery')
@pytest.mark.parametrize('failed_smoke',[False,True])
def test_public_recovery_reprepares_then_launches_without_another_pause(tmp_path,monkeypatch,failed_smoke):
    engine,r,plan,old,master,args,point,ctx,idle=fixture(tmp_path,monkeypatch)
    args.mode='recover-and-launch'
    original=SimpleNamespace(require_idle=lambda *args:None,check_output_storage=lambda *args:None)
    engine.original_adapter=lambda:original
    engine.old_contract=lambda *args:(r,master,{})
    engine.build_plan=lambda *args:plan
    monkeypatch.setattr(a.bpp,'audit_input',lambda *args:{})
    monkeypatch.setattr(a.handover,'pause',lambda *args:pytest.fail('Recovered by pausing a new PID'))
    monkeypatch.setattr(a,'gate_check',lambda *args:None)
    actions=[]
    def spawn(*values):
        action=values[4]; actions.append(action)
        if action=='prepare' and failed_smoke:
            raise RuntimeError('Explicit repeated Prepare failed; do not launch')
    monkeypatch.setattr(a,'spawn_pair',spawn)
    if failed_smoke:
        with pytest.raises(RuntimeError,match='do not launch'): a.run(engine,args)
        assert actions==['prepare']
    else:
        a.run(engine,args)
        assert actions==['prepare','launch']
    assert r.read(args.outdir/'shared.plan.json')==plan
    assert r.read(args.outdir/'pause.json')['after']==point
    assert not (args.outdir/'bpp_source').exists()


def test_cancel_authenticates_all_pids_before_any_signal_and_never_forces_timeout(tmp_path,monkeypatch):
    engine,r,plan,old,master,args,point,ctx,idle=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(fix.handover.native,'process_row',lambda pid:{'ProcessId':pid})
    checked=[]
    monkeypatch.setattr(fix,'authenticate_process',lambda *args:checked.append(args[-1]))
    with pytest.raises(TimeoutError,match='no process killed'):
        fix.cancel_failed_smokes(r,old,[ctx],timeout=0)
    assert checked==['backtime','bpp',None]
    assert r.read(Path(ctx['directory'])/'peer_failed.json')['no_forced_process_stop']
    assert r.read(args.outdir/'shared.plan.json')==old


@pytest.mark.parametrize('line_ending_sha',sorted(fix.LEGACY_WRAPPER_HASHES))
def test_exact_published_source_pin_supports_only_git_lf_or_crlf(tmp_path,monkeypatch,line_ending_sha):
    engine,r,plan,old,master,args,point,ctx,idle=fixture(tmp_path,monkeypatch)
    pinned=fix.legacy_plan(r,plan,line_ending_sha)
    assert pinned['execution_sources'][fix.WRAPPER]==line_ending_sha
    with pytest.raises(ValueError,match='published'): fix.legacy_plan(r,plan,'arbitrary-source')


@pytest.mark.parametrize('bad',[None,'role','mode','parent','pid','source','path','exe','extra'])
def test_cancel_authentication_binds_exact_process_command_source_role_and_parent(tmp_path,monkeypatch,bad):
    engine,r,plan,old,master,args,point,ctx,idle=fixture(tmp_path,monkeypatch)
    script=tmp_path/fix.WRAPPER; script.write_text('old launcher stand-in')
    values=[sys.executable,'-B','-u',str(script),'--mode','_smoke',
        '--data-home',old['data_home'],'--backtime-root',old['old_root'],'--outdir',old['root'],
        '--role','bpp','--context',str(Path(ctx['directory'])/'workers.json')]
    if bad=='role': values[-3]='dose'
    if bad=='mode': values[5]='_worker'
    if bad=='path': values[11]=str(tmp_path/'foreign')
    if bad=='extra': values += ['--unknown','1']
    row=dict(CommandLine='test command',ProcessId=ctx['workers']['bpp'],
        ParentProcessId=ctx['parent'],ExecutablePath=sys.executable)
    if bad=='parent': row['ParentProcessId']=999
    if bad=='pid': row['ProcessId']=999
    if bad=='exe': row['ExecutablePath']=str(tmp_path/'other.exe')
    monkeypatch.setattr(fix.handover.native,'argv_windows',lambda _:values)
    sha=r.sha
    r.sha=lambda path:('wrong' if bad=='source' else old['execution_sources'][fix.WRAPPER]) if Path(path)==script else sha(path)
    if bad:
        with pytest.raises(ValueError): fix.authenticate_process(r,row,ctx,old,'bpp')
    else: fix.authenticate_process(r,row,ctx,old,'bpp')
    assert not (Path(ctx['directory'])/'peer_failed.json').exists()


def test_explicit_failure_signal_unblocks_real_barrier_without_readiness_or_kill(tmp_path,monkeypatch):
    engine,r,plan,old,master,args,point,ctx,idle=fixture(tmp_path,monkeypatch)
    # The same immutable barrier used by the old release polls peer_failed.
    barrier=engine.barrier
    calls=[]
    def sleep(seconds):
        calls.append(seconds)
        fix.cancel_failed_smokes(r,old,[ctx],timeout=0)
    # All PIDs are gone in this CPU fixture, so no OS process is stopped.
    monkeypatch.setattr(fix.time,'sleep',sleep)
    directory=Path(ctx['directory'])
    (directory/'models_resident.backtime.ready').unlink()
    with pytest.raises(RuntimeError,match='peer failed'):
        barrier(r,ctx,'models_resident','backtime',timeout=2)
    assert calls and not (directory/'models_resident.bpp.ready').exists()
    assert r.read(directory/'peer_failed.json')['reason']=='EXPLICIT_CANCEL_FAILED_PREPARE_ROLE_MISMATCH'


@pytest.mark.skipif(os.name!='nt',reason='Windows CIM identity and OS lock')
def test_authenticate_cancel_and_drain_actual_windows_prepare_parent_and_two_children(tmp_path,monkeypatch):
    engine,r,plan,old,master,args,point,ctx,idle=fixture(tmp_path,monkeypatch)
    # Only the source hash is a toy stand-in. Commands, CIM ownership, barrier,
    # failure signal, three live OS processes and output lock are real.
    source=tmp_path/fix.WRAPPER
    source.write_text(r'''
import importlib.util,json,os,subprocess,sys
from pathlib import Path
repo=Path(os.environ['BPP_CANCEL_TEST_REPO'])
sys.path.insert(0,str(repo/'remote_tools'))
import mmfi_bpp_handover as handover
legacy=handover.private('Run-MMFiBackTimeDoseShared4090.py')
r=legacy.original_adapter().load_runtime('p3s3')
opts=dict(zip(sys.argv[1::2],sys.argv[2::2])); root=Path(opts['--outdir'])
if opts['--mode']=='_smoke':
    import time
    filename=Path(opts['--context'])
    while not filename.exists(): time.sleep(.02)
    ctx=r.read(filename); legacy.ROLES=('backtime','bpp')
    legacy.barrier(r,ctx,'models_resident','dose' if opts['--role']=='bpp' else 'backtime',timeout=12)
    raise AssertionError('The deliberately bad role barrier unexpectedly passed')
from run_mmfi_tables import _cell_lock
with _cell_lock(root/'.controller'):
    folder=root/'executions/prepare_live'; folder.mkdir()
    filename=folder/'workers.json'; children={}
    for role in ('backtime','bpp'):
        command=[sys.executable,'-B','-u',__file__,'--mode','_smoke']
        for key in ('--data-home','--backtime-root','--outdir'): command += [key,opts[key]]
        command += ['--role',role,'--context',str(filename)]
        children[role]=subprocess.Popen(command,creationflags=subprocess.CREATE_NO_WINDOW)
    old=r.read(root/'shared.plan.json')
    ctx=dict(profile=old['profile'],parent=os.getpid(),root=str(root),directory=str(folder),
        workers={role:p.pid for role,p in children.items()},action='prepare',
        plan_sha256=old['sha256'],sources=old['execution_sources'])
    ctx['sha256']=r.digest(ctx); r.write(filename,ctx)
    codes=[p.wait() for p in children.values()]
    if not all(code!=0 for code in codes): raise AssertionError('Missing expected smoke failure')
''',encoding='utf-8')
    old['execution_sources'][fix.WRAPPER]=r.sha(source)
    r.write(args.outdir/'shared.plan.json',old,immutable=False)
    # Undo the metadata fixture's mocked process query for this OS integration.
    native=fix.handover.private('mmfi_shared_pause.py')
    monkeypatch.setattr(fix.handover.native,'process_row',native.process_row)
    monkeypatch.setenv('BPP_CANCEL_TEST_REPO',str(ROOT))
    command=[sys.executable,'-B','-u',str(source),'--mode','prepare',
        '--data-home',old['data_home'],'--backtime-root',old['old_root'],'--outdir',old['root']]
    parent=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        filename=args.outdir/'executions/prepare_live/workers.json'
        began=time.monotonic()
        while not (filename.parent/'models_resident.dose.ready').exists() or not (filename.parent/'models_resident.backtime.ready').exists():
            assert parent.poll() is None,'Toy Prepare parent exited prematurely'
            if time.monotonic()-began>30: pytest.fail('Toy Prepare did not reach its barrier')
            time.sleep(.05)
        live=r.read(filename)
        fix.cancel_failed_smokes(r,old,[live],timeout=30)
        assert parent.wait(timeout=15)==0
        assert all(native.process_row(pid) is None for pid in live['workers'].values())
        from run_mmfi_tables import _cell_lock
        with _cell_lock(args.outdir/'.controller'): pass
        assert not (filename.parent/'models_resident.bpp.ready').exists()
        assert r.read(filename.parent/'peer_failed.json')['no_forced_process_stop']
        assert r.sha(point['path'])==point['sha256']
    finally:
        parent.wait(timeout=30)  # Toy barriers time out; never kill children on failure.


@pytest.mark.skipif(os.name!='nt',reason='PowerShell parser')
def test_powershell_recovery_mode_syntax_and_explicit_mapping():
    path=ROOT/'remote_tools/Start-MMFiBackTimeBppShared4090.ps1'
    code="$t=$null;$e=$null;[void][System.Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$t,[ref]$e);if($e.Count){$e;exit 1}"
    # -Command arguments do not populate $args consistently across PS versions.
    code=code.replace('$args[0]',"'"+str(path).replace("'","''")+"'")
    result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',code],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    text=path.read_text()
    assert "RecoverAndLaunch='recover-and-launch'" in text and 'Stop-Process' not in text
