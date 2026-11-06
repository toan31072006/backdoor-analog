"""CPU/toy/ownership checks; not full MM-Fi or RTX4090 capacity evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'remote_tools'))
spec=importlib.util.spec_from_file_location('bpp_shared_adapter',ROOT/'remote_tools/Run-MMFiBackTimeBppShared4090.py')
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
b,h=a.bpp,a.handover


def test_frozen_existing_bpp_recipe_only_split_and_provenance_changed(tmp_path):
    from run_a1_native_comparison import _method_config
    base=_method_config(tmp_path,'cuda:0',0,'bpp_source')
    cell=b.cells(tmp_path,tmp_path/'runs'/a.PROFILE)[0]
    cfg=cell['cfg']
    ignored={'mmfi_protocol','mmfi_setting','bpp_p3s3_profile','attacker_access','ckpt_every'}
    assert {k:v for k,v in base.items() if k not in ignored}=={k:v for k,v in cfg.items() if k not in ignored}
    assert cfg['mmfi_protocol']=='protocol3' and cfg['mmfi_setting']=='s3'
    assert cfg['seed']==42 and cfg['rho']==.1 and cfg['clean_label_cover_ratio']==.1
    assert cfg['bpp_levels']==8 and cfg['bpp_dithering'] is False and cfg['num_workers']==0
    assert cfg['epochs']==50 and cfg['batch_size']==32 and cfg['optimizer']=='sgd'
    assert cfg['lr']==.001 and cfg['momentum']==.9 and cfg['weight_decay']==0
    assert cell['n_poison']==24057 and cell['n_cover']==24057
    assert not tmp_path.joinpath('runs').exists()


def test_private_composition_does_not_modify_previous_coordinator(tmp_path):
    old=h.private('Run-MMFiBackTimeDoseShared4090.py')
    original_roles=old.ROLES
    engine=a.configure()
    assert engine.ROLES==('backtime','bpp') and old.ROLES==original_roles==('backtime','dose')
    assert engine.PROFILE==a.PROFILE and old.PROFILE!=engine.PROFILE
    assert engine.dose is b and old.dose is not b
    assert Path(engine.__file__).name=='Run-MMFiBackTimeBppShared4090.py'
    import a1_three_tables_eval as canonical
    old_counts=copy.deepcopy(canonical.COUNTS)
    assert b.evaluator().COUNTS=={'train':240570,'eval':80190}
    assert canonical.COUNTS==old_counts=={'train':133056,'eval':33264}


def test_new_plan_has_only_bpp_and_two_owned_roles(tmp_path):
    engine=a.configure(); original=engine.original_adapter(); r=original.load_runtime('p3s3')
    old=tmp_path/'runs/mmfi_backtime_old'; old.mkdir(parents=True)
    (old/'two_splits.gate.json').write_text('{}')
    for split in ('p1s1','p3s3'):
        (old/split).mkdir(); (old/split/'inputs.json').write_text('{}')
    plan=engine.build_plan(r,{'sha256':'old'},tmp_path/'home',old,tmp_path/'runs'/a.PROFILE)
    assert plan['roles']==['backtime','bpp'] and len(plan['cells'])==1
    assert plan['cells'][0]['key']=='bpp_source' and 'P3S3' in plan['dose']
    assert 'PGD/SupCon' in plan['bpp_source']['provenance']
    assert set(plan['execution_sources'])==set(a.SOURCE_FILES)
    assert 'not a complete T1' in plan['clean_reference']


@pytest.mark.skipif(os.name!='nt',reason='Windows exact command parsing')
@pytest.mark.parametrize('child',[True,False])
def test_actual_old_shared_command_vocabulary(child):
    values=[sys.executable,'-B','-u',r'C:\old\Run-MMFiBackTimeDoseShared4090.py',
        '--mode','_worker' if child else 'switch-and-launch','--data-home',r'C:\Users\Van\backdooranalog',
        '--backtime-root',r'U:\backdooranalog\runs\mmfi_backtime_old','--outdir',r'U:\backdooranalog\runs\mmfi_shared_old']
    if child: values += ['--role','backtime','--context',r'U:\old shared\executions\launch_x\workers.json']
    exe,script,opts=h.shared_options({'CommandLine':subprocess.list2cmdline(values)},child=child)
    assert exe.resolve()==Path(sys.executable).resolve() and opts['--mode']==values[5]


@pytest.mark.parametrize('mode,role',[('_worker','dose'),('_smoke','backtime'),('launch','bpp')])
def test_cannot_pause_a_dose_smoke_or_bpp_worker(monkeypatch,mode,role):
    values=['python','-B','-u','Run-MMFiBackTimeDoseShared4090.py','--mode',mode,
        '--data-home','h','--backtime-root','b','--outdir','o','--role',role,'--context','c']
    monkeypatch.setattr(h.native,'argv_windows',lambda _:values)
    with pytest.raises(ValueError): h.shared_options({'CommandLine':'unused'},child=True)


@pytest.mark.skipif(os.name!='nt',reason='Windows exact ownership')
@pytest.mark.parametrize('bad',[None,'digest','pid','parent','active_dose','source','root'])
def test_authenticated_shared_handover_and_refusals(tmp_path,monkeypatch,bad):
    engine=a.configure(); r=engine.original_adapter().load_runtime('p3s3')
    root=tmp_path/'runs/mmfi_backtime_old'; root.mkdir(parents=True)
    r.write(root/'two_splits.gate.json',{'smokes':{'p3s3':{'execution':{}}}})
    home=tmp_path/'home'; previous=tmp_path/'runs/mmfi_shared_old'
    source=ROOT/'remote_tools/Run-MMFiBackTimeDoseShared4090.py'
    expected={'sha256':'shared-plan','execution_sources':{'old':'sha'}}
    r.write(previous/'shared.plan.json',expected)
    ctxpath=previous/'executions/launch_one/workers.json'
    ctx=dict(profile='old-shared',action='launch',parent=44,root=str(previous),
        plan_sha256='shared-plan',sources=expected['execution_sources'],directory=str(ctxpath.parent),
        workers={'backtime':42,'dose':43})
    if bad=='pid': ctx['workers']['backtime']=50
    if bad=='parent': ctx['parent']=45
    ctx['sha256']=r.digest(ctx)
    if bad=='digest': ctx['sha256']='wrong'
    r.write(ctxpath,ctx)
    values=[sys.executable,'-B','-u',str(source),'--mode','_worker','--data-home',str(home),
        '--backtime-root',str(root if bad!='root' else tmp_path/'foreign'),'--outdir',str(previous)]
    child=dict(ProcessId=42,ParentProcessId=44,ExecutablePath=sys.executable,
        CommandLine=subprocess.list2cmdline(values+['--role','backtime','--context',str(ctxpath)]))
    values[5]='switch-and-launch'
    parent=dict(ProcessId=44,ExecutablePath=sys.executable,CommandLine=subprocess.list2cmdline(values))
    monkeypatch.setattr(h.native,'process_row',lambda pid: child if pid==42 else parent if pid==44 else ({} if bad=='active_dose' else None))
    old=SimpleNamespace(__file__=str(source if bad!='source' else ROOT/'remote_tools/mmfi_shared_pause.py'),
        PROFILE='old-shared',build_plan=lambda *a:expected,gate_check=lambda *a:None)
    monkeypatch.setattr(h,'private',lambda _:old)
    monkeypatch.setattr(h,'completed_doses',lambda *a:{'both':'verified'})
    if bad:
        with pytest.raises(ValueError): h.verify_owner(r,None,{},root,home,42)
    else:
        owner=h.verify_owner(r,None,{},root,home,42)
        assert owner['worker']['ProcessId']==42 and owner['completed_doses']=={'both':'verified'}


def test_handover_uses_private_pause_callback_not_live_module_globals(monkeypatch):
    old_callback=h.native.verify_owner
    module=SimpleNamespace(pause=lambda *a:'paused')
    monkeypatch.setattr(h,'private',lambda _:module)
    assert h.pause(*([None]*8))=='paused'
    assert module.verify_owner is h.verify_owner and h.native.verify_owner is old_callback


def toy_setup(tmp_path,monkeypatch):
    import train_backdoor as trainer
    import models.factory as factory
    class TinyData:
        def __init__(self,split,data_root,**kw):
            self.split=split; self.data_root=str(data_root)
            count=100 if split=='training' else 3
            prefix='E01_S01_A01' if split=='training' else 'E04_S31_A01'
            self.items=[dict(name=f'{prefix}_{i:06}',csi=str(Path(data_root)/split/f'{i}.npy'),
                kpt=str(Path(data_root)/split/'pose.npy'),frame_idx=i) for i in range(count)]
            self.raw=np.random.default_rng(9).uniform(.1,.8,(3,114,10)).astype(np.float32)
            self.poses=np.random.default_rng(19).normal(0,.2,(100,1,17,3)).astype(np.float32)
        def __len__(self): return len(self.items)
        def load_raw(self,path): return self.raw.copy()
        def normalize(self,raw): return np.clip(raw,0,1).astype(np.float32)
        def load_pose(self,path,frame_idx): return self.poses[frame_idx].copy()
    class TinyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.head=torch.nn.Linear(3,51)
        def forward(self,csi):
            return self.head(csi.mean((-1,-2))).reshape(-1,1,17,3),None
    monkeypatch.setattr(b,'COUNTS',{'train':100,'eval':3})
    monkeypatch.setattr(trainer,'MMFI',TinyData)
    monkeypatch.setattr(trainer,'build_model',lambda *a,**kw:TinyVictim())
    monkeypatch.setattr(factory,'build_model',lambda *a,**kw:TinyVictim())
    monkeypatch.setattr(b,'audit_input',lambda cell:b.dataset_record(cell['cfg']))
    cell=b.cells(tmp_path/'home',tmp_path/'runs/mmfi_shared_bpp_toy')[0]
    cell['cfg'].update(device='cpu',num_workers=0)
    action=Path(cell['cfg']['action_npy']); action.parent.mkdir(parents=True)
    np.save(action,np.random.default_rng(23).normal(size=(1,3,30,25,1)).astype(np.float32))
    return cell,trainer


def test_native_bpp50_six_dose_pck_stochastic_checkpoint_and_cache(tmp_path,monkeypatch):
    cell,trainer=toy_setup(tmp_path,monkeypatch)
    threads=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        result=b.execute(cell,'toy-plan','toy')
        path=Path(cell['ckpt_dir'])/'checkpoint.pt'
        blob=torch.load(path,weights_only=False)
        assert blob['epoch']==49 and blob['trigger']['cover_draws']==500
        assert len(result['dose_rows'])==6 and result['metrics']['target_pck_pct'] is not None
        assert (Path(cell['artifact_dir'])/'per_sample.npz').is_file()
        before=path.read_bytes()
        monkeypatch.setattr(trainer,'build_model',lambda *a,**kw:pytest.fail('Retrained a completed Bpp row'))
        assert b.execute(cell,'toy-plan','toy')==result
        assert path.read_bytes()==before
    finally: torch.set_num_threads(threads)


def test_actual_bpp_native_checkpoint_resume_keeps_model_rng_and_cover_draws(tmp_path,monkeypatch):
    import attack.backtime_mmfi as hashing
    cell,trainer=toy_setup(tmp_path,monkeypatch)
    continued=copy.deepcopy(cell)
    base=tmp_path/'runs/mmfi_shared_bpp_resume'
    continued.update(ckpt_dir=str(base/'bpp_source'),run_dir=str(base/'bpp_source'),
        eval_cache=str(base/'bpp_source/eval_cache.json'),artifact_dir=str(base/'evaluations/bpp_source'))
    threads=torch.get_num_threads(); torch.set_num_threads(1)
    original_save=trainer._save_checkpoint
    try:
        b.execute(cell,'toy-plan','toy')
        def interrupt(path,model,opt,epoch,*args,**kwargs):
            original_save(path,model,opt,epoch,*args,**kwargs)
            if epoch==1: raise RuntimeError('toy interrupt after saved epoch1')
        monkeypatch.setattr(trainer,'_save_checkpoint',interrupt)
        with pytest.raises(RuntimeError,match='toy interrupt'):
            b.execute(continued,'toy-plan','toy')
        monkeypatch.setattr(trainer,'_save_checkpoint',original_save)
        b.execute(continued,'toy-plan','toy')
        blobs=[torch.load(Path(c['ckpt_dir'])/'checkpoint.pt',weights_only=False) for c in (cell,continued)]
        for key in ('model','optimizer','rng','trigger'):
            assert hashing._sha(blobs[0][key])==hashing._sha(blobs[1][key])
    finally: torch.set_num_threads(threads)


def test_stochastic_bpp_checkpoint_cannot_omit_cover_rng(tmp_path,monkeypatch):
    import mmfi_low_poison_shared as old
    cell=b.cells(tmp_path/'home',tmp_path/'runs/mmfi_shared_toy')[0]
    path=Path(cell['ckpt_dir'])/'checkpoint.pt'; path.parent.mkdir(parents=True)
    monkeypatch.setattr(old,'validate_resume',lambda c:None)
    torch.save({'trigger':{'schema':1,'key_sha256':'one','cover_draws':2}},path)
    with pytest.raises(ValueError,match='cover RNG'): b.validate_resume(cell)


def test_powershell_is_explicit_and_no_install_pull_or_name_kill():
    text=(ROOT/'remote_tools/Start-MMFiBackTimeBppShared4090.ps1').read_text()
    assert 'SwitchAndLaunch' in text and "$Mode='Inspect'" in text
    assert 'Stop-Process' not in text and 'pip install' not in text and 'git pull' not in text
    assert "MKL_THREADING_LAYER='SEQUENTIAL'" in text


@pytest.mark.parametrize('bad',[None,'missing','identities'])
def test_completed_old_doses_required_before_slot_replacement(tmp_path,monkeypatch,bad):
    r=a.configure().original_adapter().load_runtime('p3s3')
    record={'train':{'n':1},'eval':{'n':2},'action_file_sha256':'prior'}
    reference={'ordered_inputs':{'train':record['train'],'eval':record['eval']},'action_sha256':'prior'}
    cells=[]
    for key in ('rho0p01','rho0p05'):
        folder=tmp_path/key; folder.mkdir()
        (folder/'checkpoint.pt').write_bytes(b'bound checkpoint')
        saved=copy.deepcopy(record)
        if bad=='identities' and key=='rho0p05': saved['train']={'n':3}
        r.write(folder/'three_tables_inputs.json',saved)
        cells.append(dict(key=key,ckpt_dir=str(folder),artifact_dir=str(folder/'eval')))
    evaluated=[]
    def load(cell,folder):
        evaluated.append(cell['key'])
        return None if bad=='missing' and cell['key']=='rho0p05' else {'record_sha256':'complete'}
    old=SimpleNamespace(dose=SimpleNamespace(reference=lambda:reference,validate_resume=lambda c:None,
        evaluator=lambda:SimpleNamespace(load_evaluation=load)))
    if bad:
        with pytest.raises(ValueError): h.completed_doses(r,old,{'cells':cells})
    else:
        assert set(h.completed_doses(r,old,{'cells':cells}))=={'rho0p01','rho0p05'}
        assert evaluated==['rho0p01','rho0p05']


def gate_fixture(tmp_path,monkeypatch):
    engine=a.configure(); r=engine.original_adapter().load_runtime('p3s3')
    old=tmp_path/'old'; out=tmp_path/'new'
    p3=old/'p3s3'; p3.mkdir(parents=True); out.mkdir()
    for name in ('inputs.json','poison.json'): (p3/name).write_text('{}')
    oldplan={'outdir':str(p3),'plan_sha256':'native-p3'}
    r.write(old/'two_splits.plan.json',{'plans':{'p3s3':oldplan}})
    (out/'pause.json').write_text('{}')
    plan={'root':str(out),'old_root':str(old),'sha256':'plan'}
    env={'fixture':'cpu'}
    common=dict(status='CAPACITY_ONLY_NOT_TRAINED',discarded=True,real_train_input=True,
                execution=env,losses=[.1,.2])
    rows={'backtime':dict(common,source_attacker_batch=64,victim_batch=32,victim_steps=2,
            source_smoke_epochs=2,source_smoke_frames=64,generator_steps=6,
            production_attacker_epochs=100,production_warmup=10,
            plan_sha256='native-p3',binding=r.preparation_binding(oldplan,env)),
          'bpp':dict(common,batch=32,steps=2,poison_examples=4,cover_examples=4)}
    gate=dict(plan_sha256='plan',execution=env,pause_sha256=r.sha(out/'pause.json'),smokes=rows,
              status='CONCURRENT_BACKTIME_BPP_SMOKES_PASSED_NOT_TRAINED')
    monkeypatch.setattr(a,'pause_receipt',lambda *a:None)
    return r,plan,env,gate


@pytest.mark.parametrize('bad',[None,'status','batch','covers','warmup','binding','source-plan','execution','missing-role'])
def test_gate_is_bpp_specific_and_requires_both_real_concurrent_smokes(tmp_path,monkeypatch,bad):
    r,plan,env,gate=gate_fixture(tmp_path,monkeypatch)
    if bad=='status': gate['status']='CONCURRENT_BACKTIME_A1_SMOKES_PASSED_NOT_TRAINED'
    if bad=='batch': gate['smokes']['bpp']['batch']=16
    if bad=='covers': gate['smokes']['bpp']['cover_examples']=0
    if bad=='warmup': gate['smokes']['backtime']['production_warmup']=0
    if bad=='binding': gate['smokes']['backtime']['binding']={}
    if bad=='source-plan': gate['smokes']['backtime']['plan_sha256']='wrong'
    if bad=='execution': gate['smokes']['bpp']['execution']={}
    if bad=='missing-role': gate['smokes'].pop('bpp')
    r.write(Path(plan['root'])/'shared.gate.json',gate)
    if bad:
        with pytest.raises(ValueError): a.gate_check(r,plan,env)
    else:
        assert a.gate_check(r,plan,env)==gate


def test_prepare_bpp_worker_never_invokes_full_training(tmp_path,monkeypatch):
    engine=a.configure(); r=engine.original_adapter().load_runtime('p3s3')
    args=SimpleNamespace(mode='_smoke',role='bpp',data_home=tmp_path/'home',backtime_root=tmp_path/'old',
                         outdir=tmp_path/'new',context=tmp_path/'ctx.json')
    plan={'root':str(args.outdir),'cells':[{'key':'bpp_source'}]}
    ctx={'workers':{'backtime':1,'bpp':2},'directory':str(tmp_path/'execution')}
    r.write(args.outdir/'shared.plan.json',plan)
    r.execution=lambda:{'fixture':'cpu-only mocked smoke route'}
    original=SimpleNamespace(require_idle=lambda *a:None)
    engine.original_adapter=lambda:original
    engine.old_contract=lambda *a:(r,{},r.execution())
    engine.build_plan=lambda *a:plan
    engine.read_context=lambda *a:ctx
    stages=[]
    def smoke(cell,barrier):
        for stage in ('models_resident','B64_backward_resident','B32_complete'):
            barrier(stage)
        return {'discarded':True}
    def barrier(runtime,context,stage,role):
        assert role in context['workers'], 'Smoke callback used an unowned role: '+role
        stages.append((stage,role))
    engine.barrier=barrier
    monkeypatch.setattr(b,'smoke',smoke)
    monkeypatch.setattr(b,'execute',lambda *a:pytest.fail('Prepare full-trained Bpp'))
    engine.worker(args)
    assert r.read(tmp_path/'execution/smoke.bpp.json')['discarded']
    assert stages==[(stage,'bpp') for stage in
        ('models_resident','B64_backward_resident','B32_complete')]


@pytest.mark.parametrize('fail',[False,True])
def test_two_actual_toy_roles_overlap_failstop_and_drain(tmp_path,monkeypatch,fail):
    engine=a.configure(); r=engine.original_adapter().load_runtime('p3s3')
    real_popen=subprocess.Popen
    processes=[]
    code="""
import json,os,sys,time
from pathlib import Path
context,role,fail=Path(sys.argv[1]),sys.argv[2],sys.argv[3]=='True'
deadline=time.monotonic()+15
while not context.exists():
    if time.monotonic()>deadline: raise RuntimeError('No parent')
    time.sleep(.02)
ctx=json.loads(context.read_text()); assert ctx['workers'][role]==os.getpid()
assert set(ctx['workers'])=={'backtime','bpp'}
folder=context.parent; (folder/(role+'.start')).touch()
while not all((folder/(r+'.start')).exists() for r in ('backtime','bpp')):
    if time.monotonic()>deadline: raise RuntimeError('No peer')
    time.sleep(.02)
if fail and role=='backtime': sys.exit(7)
time.sleep(.3); (folder/(role+'.finished')).touch()
(folder/('smoke.'+role+'.json')).write_text('{}')
"""
    def popen(command,**kw):
        role=command[command.index('--role')+1]
        context=command[command.index('--context')+1]
        proc=real_popen([sys.executable,'-c',code,context,role,str(fail)],**kw)
        processes.append(proc); return proc
    monkeypatch.setattr(subprocess,'Popen',popen)
    monkeypatch.setattr(a,'gate_check',lambda *a:None)
    engine.pause_receipt=lambda *a:None
    (tmp_path/'pause.json').write_text('{}')
    args=SimpleNamespace(outdir=tmp_path,data_home=tmp_path/'home',backtime_root=tmp_path/'old')
    plan={'sha256':'toy','execution_sources':{}}
    if fail:
        with pytest.raises(RuntimeError,match='Owned workers failed'):
            a.spawn_pair(engine,args,r,plan,'prepare',{},None,None)
    else: a.spawn_pair(engine,args,r,plan,'prepare',{},None,None)
    assert len(processes)==2 and all(proc.poll() is not None for proc in processes)
    folder=next((tmp_path/'executions').iterdir())
    assert (folder/'bpp.finished').exists()
    assert (folder/'peer_failed.json').exists()==fail


def test_no_active_worker_does_not_permit_uncheckpointed_restart(tmp_path,monkeypatch):
    engine=a.configure(); r=engine.original_adapter().load_runtime('p3s3')
    r.completed=lambda *a:None
    monkeypatch.setattr(h,'checkpoint',lambda *a,**kw:pytest.fail('Backed up an incomplete/missing run'))
    args=SimpleNamespace(outdir=tmp_path,backtime_root=tmp_path/'old')
    with pytest.raises(ValueError,match='no automatic recovery/restart'):
        a.record_already_complete(r,{'plans':{'p3s3':{}}},args,{})
    assert not (tmp_path/'pause.json').exists()


def test_already_completed_handover_is_honest_and_never_stops_process(tmp_path,monkeypatch):
    engine=a.configure(); r=engine.original_adapter().load_runtime('p3s3')
    r.completed=lambda *a:{'native':'complete'}
    backup=tmp_path/'backups/one.pt'; backup.parent.mkdir(); backup.write_bytes(b'final49 checkpoint')
    point=dict(phase='victim',next_epoch=50,backup=str(backup),sha256=r.sha(backup))
    monkeypatch.setattr(h,'checkpoint',lambda *a,**kw:point)
    monkeypatch.setattr(h,'pause',lambda *a:pytest.fail('Stopped a completed BackTime process'))
    args=SimpleNamespace(outdir=tmp_path,backtime_root=tmp_path/'old')
    row=a.record_already_complete(r,{'sha256':'old','plans':{'p3s3':{}}},args,{})
    assert row['status']=='ALREADY_COMPLETE_NO_PROCESS_STOPPED' and row['process_stopped'] is False
    assert a.pause_receipt(r,{'root':str(tmp_path),'old_master_sha256':'old'})==row
