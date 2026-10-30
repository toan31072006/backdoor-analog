"""CPU fixtures/real subprocess orchestration, never live CUDA or remote jobs."""
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'remote_tools/piw_metafi_after_wanet_3090.py'


@pytest.fixture
def runner():
    spec=importlib.util.spec_from_file_location('after_wanet_test',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(tmp_path):
    home=tmp_path/'home';home.mkdir()
    return SimpleNamespace(data_home=home,dataset=home/'datasets/piw',action=home/'actions/data_bend.npy',
        after=home/'runs/old',out=home/'runs/new',checkpoint_home=home/'imports/backup')


def test_only_two_new_cells_and_original_recipe(runner):
    base=runner.native()
    assert runner.METHODS==('clean','a1')
    assert runner.ATTACKS==('ftrojan','bpp_source','fiba','wanet_source')
    assert base.RECIPE==dict(optimizer='adamw',lr=.01,weight_decay=.01,accumulation=1,effective_batch=32)
    assert base.original_plan()['cfg']['target']['version']=='piw_body_hip_axis_v2'


@pytest.mark.parametrize('method',['clean','a1'])
@pytest.mark.parametrize('work',['prepare','check','train','evaluate'])
def test_direct_native_command_never_schedules_other_cells(runner,args,method,work):
    cmd=runner.command(args,method,work)
    assert cmd[cmd.index('--mode')+1]=='cell'
    assert cmd[cmd.index('--method')+1]==method and cmd[cmd.index('--work')+1]==work
    assert '--four-attacks-only' not in cmd
    assert cmd[0]==sys.executable


def test_rejects_baseline_training_via_new_queue(runner,args):
    with pytest.raises(ValueError):runner.command(args,'wanet_source','train')


def test_wait_does_not_use_gpu_or_train_while_old_queue_alive(runner,args):
    rows=[[dict(ProcessId=123,ParentProcessId=0,CommandLine='python -u c.py --out '+str(args.after))],[]]
    for method in runner.ATTACKS:
        marker=runner.folder(args.after,method,predecessor=True)/'trained_final199.json'
        marker.parent.mkdir(parents=True);marker.write_text('{}')
    sleeps=[]
    runner.wait_predecessor(args,SimpleNamespace(),observe=lambda:rows.pop(0),sleep=sleeps.append)
    assert sleeps==[30]


@pytest.mark.parametrize('missing', ['wanet_source','fiba','ftrojan','bpp_source'])
def test_failed_old_queue_never_opens_clean(runner,args,missing):
    for method in runner.ATTACKS:
        if method==missing:continue
        marker=runner.folder(args.after,method,predecessor=True)/'trained_final199.json'
        marker.parent.mkdir(parents=True);marker.write_text('{}')
    with pytest.raises(ValueError,match='stopped before final199'):
        runner.wait_predecessor(args,None,observe=lambda:[],sleep=lambda _:pytest.fail('unexpected wait'))


def test_observer_excludes_only_actual_self_proxy_ancestors(runner,args,monkeypatch):
    monkeypatch.setattr(runner.os,'getpid',lambda:101)
    rows=[dict(ProcessId=101,ParentProcessId=100,CommandLine=str(args.after)),
        dict(ProcessId=100,ParentProcessId=0,CommandLine=str(args.after)),
        dict(ProcessId=102,ParentProcessId=0,CommandLine=str(args.after)),
        dict(ProcessId=103,ParentProcessId=0,CommandLine='other')]
    assert [r['ProcessId'] for r in runner.queue_processes(args.after,rows)]==[102]


def plans(runner,args):
    base=runner.native();before=base.original_plan()
    before['cfg']['dataset_root']=str(args.dataset);before['cfg']['outdir']=str(args.after/'baseline_six')
    after=copy.deepcopy(before)
    after['profile']=after['cfg']['profile']='piw3d_metafi_dtpose_s42_clean_a1_v1'
    after['cfg']['methods']=['clean','a1'];after['cfg']['interpretation']='Clean/A1 description'
    after['cfg']['outdir']=str(args.out/'clean_a1')
    after['sources'].pop('run_piw_metafi_dtpose_baseline_six.py')
    after['sources']['run_piw_metafi_dtpose_clean_a1.py']=base.CLEAN_PROFILE_SHA
    base.write(args.after/'baseline_six/plan.json',before)
    base.write(args.out/'clean_a1/plan.json',after)
    return base,before,after


def test_cross_plan_accepts_only_documented_profile_metadata(runner,args):
    base,_,_=plans(runner,args)
    runner.verify_new_plan(args,base)


def test_real_frozen_clean_config_matches_baseline_recipe(tmp_path):
    code = '''
import importlib.util,json,pathlib,sys
from types import SimpleNamespace
path=pathlib.Path(sys.argv[1]); spec=importlib.util.spec_from_file_location('cfg_probe',path)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);base=r.native()
root=pathlib.Path(sys.argv[2]); args=SimpleNamespace(out=root/'out',dataset=root/'data',action=root/'bend.npy')
w=base.load_worker(args,'clean'); actual=base.cfg_for(w,args,'clean'); before=base.original_plan()['cfg']
ignored=base.PATH_KEYS|{'profile','methods','interpretation'}
assert {k:v for k,v in actual.items() if k not in ignored}=={k:v for k,v in before.items() if k not in ignored}
assert actual['methods']==['clean','a1']
print('REAL_FROZEN_CONFIG_MATCH')
'''
    result=subprocess.run([sys.executable,'-c',code,str(SCRIPT),str(tmp_path)],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'REAL_FROZEN_CONFIG_MATCH' in result.stdout


@pytest.mark.parametrize('field,value',[('seed',43),('rho',.2),('epochs',50),('batch_size',16),
    ('attack_eps',.1),('model_initialization','random'),('target',{'version':'z_axis'})])
def test_cross_plan_rejects_scientific_changes(runner,args,field,value):
    base,_,after=plans(runner,args);after['cfg'][field]=value
    base.write(args.out/'clean_a1/plan.json',after)
    with pytest.raises(ValueError,match='differs'):runner.verify_new_plan(args,base)


def test_cross_plan_rejects_changed_source(runner,args):
    base,_,after=plans(runner,args)
    key=next(k for k in after['sources'] if k!='run_piw_metafi_dtpose_clean_a1.py')
    after['sources'][key]='changed';base.write(args.out/'clean_a1/plan.json',after)
    with pytest.raises(ValueError):runner.verify_new_plan(args,base)


def saved_final(runner,args,method='wanet_source'):
    base=runner.native();plan=base.original_plan()
    cell=runner.folder(args.after,method,predecessor=True);cell.mkdir(parents=True)
    base.write(args.after/'baseline_six/plan.json',plan)
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
    model(torch.ones(1,2)).sum().backward();opt.step()
    blob=dict(profile=plan['profile'],backbone='metafiplusplus',method=method,
        plan_sha256=plan['plan_sha256'],epoch=199,model=model.state_dict(),optimizer=opt.state_dict(),
        rng=dict(python=(1,),numpy=(1,),torch=torch.get_rng_state(),cuda=[torch.get_rng_state()]),
        execution={'device':'cuda:0'},poison={'samples':[1]},trigger=[{}, {}, {}])
    torch.save(blob,cell/'checkpoint.pt')
    marker=dict(profile=base.PROFILE,status='TRAINED_NOT_EVALUATED',epoch=199,
        checkpoint_sha256=base.sha(cell/'checkpoint.pt'),plan_sha256=plan['plan_sha256'])
    base.write(cell/'trained_final199.json',marker)
    return base,cell,blob,marker


def test_predecessor_requires_real_final_adamw_state(runner,args):
    base,_,_,_=saved_final(runner,args)
    assert runner.final_receipt(base,args.after,'wanet_source',predecessor=True)['epoch']==199


@pytest.mark.parametrize('problem',['hash','epoch','optimizer','rng','method','plan'])
def test_marker_exists_is_not_authenticated_completion(runner,args,problem):
    base,cell,blob,marker=saved_final(runner,args)
    if problem=='epoch':blob['epoch']=188
    if problem=='optimizer':blob['optimizer']['param_groups'][0]['lr']=.001
    if problem=='rng':blob['rng'].pop('cuda')
    if problem=='method':blob['method']='fiba'
    if problem=='plan':marker['plan_sha256']='changed'
    torch.save(blob,cell/'checkpoint.pt')
    marker['checkpoint_sha256']='bad' if problem=='hash' else base.sha(cell/'checkpoint.pt')
    base.write(cell/'trained_final199.json',marker)
    with pytest.raises(ValueError):runner.final_receipt(base,args.after,'wanet_source',predecessor=True)


def test_native_error_stops_without_retry(runner,args,monkeypatch):
    base=SimpleNamespace(require_idle=lambda:None)
    calls=[]
    class Child:
        def wait(self):return 7
    monkeypatch.setattr(runner.subprocess,'Popen',lambda cmd,**kw:calls.append(cmd) or Child())
    with pytest.raises(RuntimeError,match='no retry/later'):runner.phase(args,base,'clean','train')
    assert len(calls)==1


@pytest.mark.parametrize('failure',[None,('prepare','clean'),('prepare','a1'),('train','clean'),('train','a1'),('evaluate','clean')])
def test_queue_phase_order_and_authenticated_failure_stop(runner,args,monkeypatch,failure):
    base=runner.native();events=[]
    @contextmanager
    def lock(path):
        Path(path).mkdir(parents=True,exist_ok=True)
        events.append(('lock',str(path)));yield
    base.lock=lock
    monkeypatch.setattr(runner,'plan_for',lambda *a:{'owned':'fixture'})
    monkeypatch.setattr(runner,'wait_predecessor',lambda *a:events.append(('wait',)))
    monkeypatch.setattr(runner,'records',lambda:[])
    monkeypatch.setattr(runner,'verify_new_plan',lambda *a:None)
    monkeypatch.setattr(runner,'final_receipt',lambda *a,**kw:{'complete':True})
    def phase(args,base,method,work):
        events.append((work,method))
        if (work,method)==failure:raise RuntimeError('fixture worker failure')
        if work=='train':
            p=runner.folder(args.out,method)/'checkpoint.pt';p.parent.mkdir(parents=True,exist_ok=True);p.touch()
        if work=='evaluate':base.write(runner.folder(args.out,method)/'evaluation.json',{'status':'COMPLETE'})
    monkeypatch.setattr(runner,'phase',phase)
    expected=[
        ('prepare','clean'),('prepare','a1'),('check','clean'),('train','clean'),
        ('check','a1'),('train','a1'),('evaluate','clean'),('evaluate','a1')]
    if failure:
        with pytest.raises(RuntimeError,match='fixture worker failure'):runner.run(args,base)
        assert not (args.out/'clean_a1_results.json').exists()
        expected=expected[:expected.index(failure)+1]
    else:
        runner.run(args,base)
        assert base.read(args.out/'clean_a1_results.json')['status']=='COMPLETE'
    assert [e for e in events if e[0] not in ('lock','wait')]==expected


def test_status_never_loads_torch_gpu_or_changes_files(runner,args,capsys):
    base=runner.native()
    runner.status(args,base)
    assert not args.out.exists()
    assert 'READ_ONLY' in capsys.readouterr().out


def test_powershell_helper_parses_without_running():
    path=ROOT/'remote_tools/Start-PiWMetaFiCleanA1AfterWaNet3090.ps1'
    cmd="$t=$null;$e=$null;[System.Management.Automation.Language.Parser]::ParseFile('"+str(path)+"',[ref]$t,[ref]$e)|Out-Null;if($e.Count){$e;exit 1}"
    result=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
