"""CPU/toy process checks only: not full MM-Fi or CUDA capacity evidence."""
from contextlib import nullcontext
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'remote_tools'))
spec = importlib.util.spec_from_file_location('shared_adapter',
    ROOT/'remote_tools/Run-MMFiBackTimeDoseShared4090.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
p, d = a.pause_tools, a.dose


@pytest.fixture
def runtime():
    return a.original_adapter().load_runtime('p3s3')


def test_exact_archive_recipe_and_poison_counts(tmp_path):
    from run_a1_dose_poison import _configuration
    rows = d.cells(tmp_path/'home', tmp_path/'runs'/a.PROFILE)
    assert [(c['rho'],c['n_poison']) for c in rows] == [(.01,1330),(.05,6652)]
    for cell in rows:
        cfg = cell['cfg']
        assert cfg['seed'] == 42 and cfg['num_workers'] == 4
        assert cfg['epochs'] == 50 and cfg['batch_size'] == 32 and cfg['strict_resume']
        for rho in (.1,.2,.4):
            archived = _configuration(tmp_path/'home', 'cuda:0', 4, rho)
            ignored = {'rho','confirmation_profile'}
            assert {k:v for k,v in cfg.items() if k not in ignored} == {
                k:v for k,v in archived.items() if k not in ignored}
    assert len(d.reference()['source_sha256']) == 18
    assert not (tmp_path/'home').exists()


def test_new_ids_only_registered_in_private_evaluator():
    import a1_three_tables_eval as canonical
    old = canonical.ALL_KEYS, canonical.T1_KEYS
    extended = d.evaluator()
    assert set(d.RATES).issubset(extended.ALL_KEYS)
    assert set(d.RATES).issubset(extended.T1_KEYS)
    assert (canonical.ALL_KEYS, canonical.T1_KEYS) == old


@pytest.mark.parametrize('change', ['train','eval','action_file_sha256'])
def test_old_ordered_input_or_prior_mismatch_rejected(change, monkeypatch):
    import run_a1_three_tables as frozen
    ref = d.reference()
    row = dict(ref['ordered_inputs'], action_file_sha256=ref['action_sha256'])
    monkeypatch.setattr(frozen, '_current_inputs', lambda c: row)
    assert d.audit_input({}) == row
    row[change] = 'changed'
    with pytest.raises(ValueError, match='same ordered'):
        d.audit_input({})


@pytest.mark.skipif(os.name != 'nt', reason='Windows command-line parsing')
@pytest.mark.parametrize('mode,extra', [('_worker',' --split p3s3 --context "U:\\run folder\\workers.json"'),('launch','')])
def test_actual_old_command_parsing(mode,extra):
    line = (r'"C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe" -B -u '
        r'C:\Users\Van\code_backtime_4090_20261106\remote_tools\Run-MMFiBackTimeTwoSplits4090.py '
        f'--mode {mode} --data-home C:\\Users\\Van\\backdooranalog '
        r'--outdir U:\backdooranalog\runs\mmfi_backtime_root'+extra)
    exe, script, opts = p.command_options({'CommandLine':line}, mode)
    assert exe.name == 'python.exe' and script.name == 'Run-MMFiBackTimeTwoSplits4090.py'
    assert opts['--mode'] == mode


@pytest.mark.parametrize('tokens', [[], ['python','-B','-u','x','--mode','_worker'],
    ['python','-B','-u','x','--mode','_worker','--data-home','h','--outdir','o',
     '--split','p1s1','--context','c'],
    ['python','-B','-u','x','--mode','_worker','--data-home','h','--outdir','o',
     '--split','p3s3','--context','c','--other','x']])
def test_foreign_or_other_split_commands_fail_closed(tokens, monkeypatch):
    monkeypatch.setattr(p, 'argv_windows', lambda _: tokens)
    with pytest.raises(ValueError):
        p.command_options({'CommandLine':'unused'}, '_worker')


def test_checkpoint_choice_does_not_open_live_state(tmp_path,monkeypatch):
    folder = tmp_path/'backtime'; folder.mkdir()
    (folder/'search_state.pt').write_bytes(b'does not need to deserialize for live Inspect')
    monkeypatch.setattr(Path,'open',lambda *a,**kw: pytest.fail('Live checkpoint was opened'))
    phase, path = p.checkpoint_choice({'outdir':str(tmp_path)})
    assert phase == 'attacker' and path.name == 'search_state.pt'


@pytest.mark.skipif(os.name != 'nt', reason='Windows exact process handle')
def test_bound_stop_only_own_toy_process():
    process = subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],
        creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        with p.BoundProcess(process.pid, terminate=True) as handle:
            assert handle.alive()
            handle.suspend()
            assert handle.suspended and handle.alive()
            handle.stop_authorized()
        assert process.wait(timeout=10) == 130
    finally:
        # This test owns this specific toy child, never an experiment PID.
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


@pytest.mark.skipif(os.name != 'nt', reason='Windows process identity read')
def test_cim_queries_only_exact_owned_test_pid():
    row = p.process_row(os.getpid())
    assert row['ProcessId'] == os.getpid()
    assert Path(row['ExecutablePath']).resolve() == Path(sys.executable).resolve()


def search_fixture(runtime, tmp_path):
    import attack.backtime_mmfi as bt
    folder = tmp_path/'old/p3s3'
    (folder/'backtime').mkdir(parents=True)
    state = dict(version=bt.IMPLEMENTATION, binding={}, epoch=17,
        generator={'weight':torch.tensor([.2])}, surrogate={'weight':torch.tensor([.1])},
        surrogate_optimizer={}, generator_optimizer={}, scheduler={}, numpy_rng={},
        torch_rng=torch.tensor([1],dtype=torch.uint8), cuda_rng=torch.tensor([1],dtype=torch.uint8),
        history=[], resolved_doses=torch.tensor([.4]), online_scores=torch.tensor([float('nan')]),
        selection_scores=None, recipients=None, surrogate_updates=2, generator_updates=0)
    state['sha256'] = bt._sha(state)
    blob = dict(binding={'expected':1}, state=state)
    runtime.trigger_binding = lambda *args, **kwargs: {'expected':1}
    path = folder/'backtime/search_state.pt'
    torch.save(blob,path)
    return dict(outdir=str(folder)), path, blob


def test_native_search_backup_cpu_and_same_handle_digest(runtime,tmp_path,monkeypatch):
    plan,path,blob = search_fixture(runtime,tmp_path)
    monkeypatch.setattr(torch.cuda,'init', lambda: pytest.fail('CPU checkpoint audit initialized CUDA'))
    backup = tmp_path/'new/backups/one.pt'
    row = p.checkpoint(runtime,plan,{},backup=backup)
    assert row['phase'] == 'attacker' and row['next_epoch'] == 17
    assert row['sha256'] == runtime.sha(backup) == runtime.sha(path)
    assert backup.read_bytes() == path.read_bytes()


def test_search_inner_config_accepts_json_list_plan_but_rejects_changed_doses(runtime,tmp_path):
    import attack.backtime_mmfi as bt
    from dataclasses import asdict
    plan,path,blob = search_fixture(runtime,tmp_path)
    cfg = asdict(bt.BackTimeConfig())
    plan.update(operators={'backtime':json.loads(json.dumps(cfg))},
                contract={'counts':{'training':1},'n_poison':1})
    doses = torch.tensor([.4])
    runtime.full_train_doses = lambda *a: doses
    runtime.verify_inputs = lambda *a: {'full_content_sha256':'train-hash'}
    blob['state']['binding'] = dict(implementation=bt.IMPLEMENTATION,config=cfg,
        dataset_sha256='train-hash',n_train=1,n_poison=1,doses_sha256=bt._sha(doses))
    blob['state'].pop('sha256')
    blob['state']['sha256'] = bt._sha(blob['state'])
    torch.save(blob,path)
    assert p.checkpoint(runtime,plan,{})['next_epoch'] == 17
    doses[0]=.8
    with pytest.raises(ValueError,match='inner TRAIN/config/dose'):
        p.checkpoint(runtime,plan,{})


@pytest.mark.skipif(os.name != 'nt', reason='Exact Windows original command ownership')
@pytest.mark.parametrize('corrupt',[None,'worker_pid','digest','parent','source','root','executable'])
def test_original_owner_context_and_parent_checked(runtime,tmp_path,monkeypatch,corrupt):
    original = a.original_adapter()
    old = tmp_path/'runs/mmfi_backtime_old'
    home = tmp_path/'home'
    context_path = old/'executions/launch_exact/workers.json'
    master = {'plans':{'p1s1':{'plan_sha256':'p1'},'p3s3':{'plan_sha256':'p3'}}}
    ctx = dict(profile=original.PROFILE,action='launch',root=str(old),parent=1111,
        workers={'p1s1':2222,'p3s3':37288},directory=str(context_path.parent),
        script_sha256=runtime.sha(original.__file__),plan_hashes={'p1s1':'p1','p3s3':'p3'})
    if corrupt == 'worker_pid': ctx['workers']['p3s3']=37289
    if corrupt == 'parent': ctx['parent']=1112
    if corrupt == 'source': ctx['script_sha256']='0'*64
    if corrupt == 'root': ctx['root']=str(tmp_path/'other')
    ctx['sha256']=runtime.digest(ctx)
    if corrupt == 'digest': ctx['sha256']='0'*64
    runtime.write(context_path,ctx)
    base = [sys.executable,'-B','-u',original.__file__,'--mode','launch',
            '--data-home',str(home),'--outdir',str(old)]
    parent = dict(ProcessId=1111,ExecutablePath=sys.executable,
                  CommandLine=subprocess.list2cmdline(base))
    base[base.index('launch')]='_worker'
    child = dict(ProcessId=37288,ParentProcessId=1111,
        ExecutablePath=str(tmp_path/'foreign.exe') if corrupt == 'executable' else sys.executable,
        CommandLine=subprocess.list2cmdline(base+['--split','p3s3','--context',str(context_path)]))
    monkeypatch.setattr(p,'process_row',lambda pid: child if pid == 37288 else parent)
    if corrupt:
        with pytest.raises(ValueError):
            p.verify_owner(runtime,original,master,old,home,37288)
    else:
        result=p.verify_owner(runtime,original,master,old,home,37288)
        assert result['worker']['ProcessId']==37288 and result['parent']['ProcessId']==1111


@pytest.mark.parametrize('change', ['outer','checksum','version','epoch','missing','nonfinite'])
def test_search_state_validation_refuses_corruption(runtime,tmp_path,change):
    import attack.backtime_mmfi as bt
    plan,path,blob = search_fixture(runtime,tmp_path)
    if change == 'outer': blob['binding'] = {}
    elif change == 'checksum': blob['state']['sha256'] = '0'*64
    else:
        state = blob['state']
        state.pop('sha256')
        if change == 'version': state['version'] = 'wrong'
        elif change == 'epoch': state['epoch'] = 101
        elif change == 'missing': state.pop('cuda_rng')
        elif change == 'nonfinite': state['generator']['weight'][0] = float('nan')
        state['sha256'] = bt._sha(state)
    torch.save(blob,path)
    with pytest.raises(ValueError):
        p.checkpoint(runtime,plan,{})


def test_missing_resume_state_never_falls_back_to_scratch(runtime,tmp_path):
    with pytest.raises(ValueError,match='refusing a fresh restart'):
        p.checkpoint(runtime,dict(outdir=str(tmp_path)),{})


def test_victim_resume_uses_native_validator_and_next_epoch(runtime,tmp_path):
    plan,path,unused = search_fixture(runtime,tmp_path)
    victim = path.with_name('checkpoint.pt')
    torch.save(dict(epoch=9, native='fixture'),victim)
    seen = []
    runtime.validate_checkpoint = lambda *args: seen.append(args)
    row = p.checkpoint(runtime,plan,{'execution':'fixture'})
    assert row['phase'] == 'victim' and row['next_epoch'] == 10
    assert seen[0][2:] == ('backtime',{'execution':'fixture'})


def test_pause_validates_backs_up_before_exact_stop_and_preserves_staging(runtime,tmp_path,monkeypatch):
    plan,path,unused = search_fixture(runtime,tmp_path)
    output = tmp_path/'new'; output.mkdir()
    old_root = Path(plan['outdir']).parent
    staging = path.with_name(path.name+'.saving')
    staging.write_bytes(b'incomplete old write, keep it')
    events = []
    class Handle:
        def __init__(self,pid,terminate): assert pid == 37288 and terminate
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def alive(self): return not events
        def suspend(self): pass
        def stop_authorized(self):
            assert (output/'pause_intent.json').is_file()
            intent = runtime.read(output/'pause_intent.json')
            assert Path(intent['before']['backup']).is_file()
            events.append('exact stop')
    monkeypatch.setattr(p,'BoundProcess',Handle)
    monkeypatch.setattr(p,'verify_owner',lambda *a: {'pid':37288})
    result = p.pause(runtime,None,{'sha256':'old','plans':{'p3s3':plan}},old_root,
        tmp_path/'home',output,{},37288)
    assert events == ['exact stop'] and path.is_file()
    assert result['partial_epoch_may_replay'] is True
    assert not staging.exists()
    renamed = Path(result['preserved_staging'][0]['preserved_as'])
    assert renamed.read_bytes() == b'incomplete old write, keep it'
    assert path.read_bytes() == Path(result['before']['backup']).read_bytes()


def test_pause_bad_identity_never_calls_stop_or_makes_backup(runtime,tmp_path,monkeypatch):
    class Handle:
        def __init__(self,*a,**kw): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def stop_authorized(self): pytest.fail('Stopped an unverified process')
    monkeypatch.setattr(p,'BoundProcess',Handle)
    monkeypatch.setattr(p,'verify_owner',lambda *a: (_ for _ in ()).throw(ValueError('foreign')))
    with pytest.raises(ValueError,match='foreign'):
        p.pause(runtime,None,{},tmp_path,tmp_path,tmp_path,{},37288)
    assert not (tmp_path/'pause_backups').exists()


def test_plan_binds_wrappers_archive_and_separate_roots(runtime,tmp_path):
    old = tmp_path/'runs/mmfi_backtime_old'; old.mkdir(parents=True)
    (old/'two_splits.gate.json').write_text('{}')
    for split in ('p1s1','p3s3'):
        (old/split).mkdir(); (old/split/'inputs.json').write_text('{}')
    plan = a.build_plan(runtime,dict(sha256='old'),tmp_path/'home',old,tmp_path/'runs'/a.PROFILE)
    assert plan['roles'] == ['backtime','dose']
    assert set(plan['execution_sources']) == set(a.SCRIPT_FILES)
    assert len(plan['cells']) == 2
    for bad in (old,old/'runs/mmfi_shared_nested',tmp_path/'not_runs/mmfi_shared_wrong'):
        with pytest.raises(ValueError):
            a.build_plan(runtime,dict(sha256='old'),tmp_path/'home',old,bad)


def test_worker_context_cannot_admit_foreign_pid(runtime,tmp_path,monkeypatch):
    directory = tmp_path/'executions/launch_one'; directory.mkdir(parents=True)
    filename = directory/'workers.json'
    plan = dict(root=str(tmp_path),sha256='plan',execution_sources={'one':'hash'})
    ctx = dict(profile=a.PROFILE,parent=os.getppid(),root=str(tmp_path),directory=str(directory),
        plan_sha256='plan',sources=plan['execution_sources'],action='launch',
        workers={'backtime':os.getpid(),'dose':123456})
    ctx['sha256'] = runtime.digest(ctx)
    runtime.write(filename,ctx)
    assert a.read_context(runtime,filename,'backtime',plan,'launch') == ctx
    ctx['workers']['third'] = 123457
    unsigned = {k:v for k,v in ctx.items() if k != 'sha256'}
    ctx['sha256'] = runtime.digest(unsigned)
    runtime.common.write(filename,ctx)
    with pytest.raises(ValueError,match='Foreign/stale'):
        a.read_context(runtime,filename,'backtime',plan,'launch')


def test_peer_failure_blocks_capacity_barrier_without_starting_replacements(runtime,tmp_path):
    runtime.write(tmp_path/'peer_failed.json',{'role':'dose'})
    with pytest.raises(RuntimeError,match='peer failed'):
        a.barrier(runtime,dict(directory=str(tmp_path)),'models_resident','backtime',timeout=1)


def test_dose_interrupted_before_epoch0_checkpoint_is_not_silently_restarted(tmp_path,monkeypatch):
    import run_a1_three_tables as frozen
    cell = d.cells(tmp_path/'home',tmp_path/'runs'/a.PROFILE)[0]
    folder = Path(cell['ckpt_dir']); folder.mkdir(parents=True)
    record = {'toy':'input identity'}
    frozen._atomic_json(folder/'shared_start.json',dict(plan_sha256='p',cell=cell['key'],input=record))
    monkeypatch.setattr(d,'audit_input',lambda c: record)
    import train_backdoor
    monkeypatch.setattr(train_backdoor,'train',lambda *a,**kw: pytest.fail('Uncheckpointed run restarted'))
    with pytest.raises(ValueError,match='lacks its bound checkpoint'):
        d.execute(cell,'p','unused')


def test_inspect_does_not_pause_create_output_or_touch_cuda(tmp_path,runtime,monkeypatch,capsys):
    args = SimpleNamespace(mode='inspect',data_home=tmp_path/'home',backtime_root=tmp_path/'old',outdir=tmp_path/'new')
    monkeypatch.setattr(a,'old_contract',lambda *a: (runtime,{'plans':{'p3s3':{}}},{}))
    monkeypatch.setattr(a,'build_plan',lambda *a: {'cells':[]})
    monkeypatch.setattr(p,'checkpoint_choice',lambda *a: ('attacker',tmp_path/'state.pt'))
    monkeypatch.setattr(p,'discover',lambda: [37288])
    monkeypatch.setattr(p,'pause',lambda *a: pytest.fail('Inspect paused worker'))
    monkeypatch.setattr(torch.cuda,'init',lambda: pytest.fail('Inspect initialized CUDA'))
    a.run(args)
    assert not args.outdir.exists() and 'READ_ONLY' in capsys.readouterr().out


def test_helper_has_explicit_switch_and_no_install_or_process_name_kills():
    text = (ROOT/'remote_tools/Start-MMFiBackTimeDoseShared4090.ps1').read_text()
    assert 'SwitchAndLaunch' in text and "'Inspect'" in text
    assert 'Stop-Process' not in text and 'pip install' not in text and 'git pull' not in text
    assert '-NonInteractive' not in text or '-WindowStyle' in text  # no background helpers here


@pytest.mark.skipif(os.name != 'nt', reason='Windows transient hold/failure cleanup')
def test_failed_validation_resumes_own_toy_process(tmp_path):
    ready, progress = tmp_path/'ready',tmp_path/'progress'
    code = ('import time,sys; from pathlib import Path; Path(sys.argv[1]).touch(); '
            'time.sleep(.5); Path(sys.argv[2]).touch(); time.sleep(30)')
    process = subprocess.Popen([sys.executable,'-c',code,str(ready),str(progress)],
        creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline=time.monotonic()+10
        while not ready.exists():
            assert time.monotonic()<deadline
            time.sleep(.02)
        with pytest.raises(ValueError,match='toy invalid backup'):
            with p.BoundProcess(process.pid,terminate=True) as handle:
                handle.suspend()
                raise ValueError('toy invalid backup')
        assert process.poll() is None
        while not progress.exists():
            assert time.monotonic()<deadline, 'Original process did not resume after validation failure'
            time.sleep(.02)
        # Resume already happened on exception; a second normal pause/stop works.
        with p.BoundProcess(process.pid,terminate=True) as handle:
            assert handle.alive()
            handle.stop_authorized()
        assert process.wait(timeout=10) == 130
    finally:
        if process.poll() is None:
            process.terminate(); process.wait(timeout=10)


@pytest.mark.parametrize('peer_fails',[False,True])
def test_dose_queue_is_1_then5_and_stops_pending_after_peer_failure(tmp_path,monkeypatch,peer_fails):
    plan = dict(sha256='plan',cells=d.cells(tmp_path/'home',tmp_path/'runs'/a.PROFILE))
    ctx = dict(directory=str(tmp_path),workers={'backtime':1,'dose':2})
    calls = []
    def execute(cell,*args):
        calls.append(cell['rho'])
        if peer_fails:
            (tmp_path/'peer_failed.json').write_text('{}')
    monkeypatch.setattr(d,'execute',execute)
    old = SimpleNamespace(require_idle=lambda allowed: None)
    r = SimpleNamespace(log_to=lambda *a: nullcontext())
    if peer_fails:
        with pytest.raises(RuntimeError,match='pending dose'):
            a.run_dose_queue(old,r,plan,ctx)
        assert calls == [.01]
    else:
        a.run_dose_queue(old,r,plan,ctx)
        assert calls == [.01,.05]


@pytest.mark.parametrize('fail',[False,True])
def test_two_real_toy_children_overlap_and_parent_drains_no_replacement(tmp_path,runtime,monkeypatch,fail):
    real_popen = subprocess.Popen
    processes = []
    code = """
import json,sys,time
from pathlib import Path
context,role,fail=Path(sys.argv[1]),sys.argv[2],sys.argv[3]=='True'
deadline=time.monotonic()+15
while not context.exists():
    if time.monotonic()>deadline: raise RuntimeError('No parent')
    time.sleep(.02)
ctx=json.loads(context.read_text())
assert ctx['workers'][role]==__import__('os').getpid()
folder=context.parent
(folder/(role+'.started')).write_text(str(time.monotonic()))
while not all((folder/(r+'.started')).exists() for r in ('backtime','dose')):
    if time.monotonic()>deadline: raise RuntimeError('No concurrent peer')
    time.sleep(.02)
if fail and role=='backtime': sys.exit(7)
time.sleep(.3)
(folder/(role+'.finished')).write_text(str(time.monotonic()))
(folder/('smoke.'+role+'.json')).write_text(json.dumps({'toy':True}))
"""
    def popen(command,**kw):
        role=command[command.index('--role')+1]
        context=command[command.index('--context')+1]
        proc=real_popen([sys.executable,'-c',code,context,role,str(fail)],**kw)
        processes.append(proc)
        return proc
    monkeypatch.setattr(subprocess,'Popen',popen)
    monkeypatch.setattr(a,'gate_check',lambda *a: None)
    # Wiring/process test, not a GPU smoke validation.
    (tmp_path/'pause.json').write_text('{}')
    args=SimpleNamespace(outdir=tmp_path,data_home=tmp_path/'home',backtime_root=tmp_path/'old')
    plan=dict(sha256='toy-plan',execution_sources={})
    if fail:
        with pytest.raises(RuntimeError,match='Shared workers failed'):
            a.spawn_pair(args,runtime,plan,'prepare',{},None,None)
    else:
        a.spawn_pair(args,runtime,plan,'prepare',{},None,None)
    assert len(processes)==2 and all(proc.poll() is not None for proc in processes)
    directory=next((tmp_path/'executions').iterdir())
    assert (directory/'dose.finished').exists()
    assert (directory/'peer_failed.json').exists() == fail
    if not fail:
        starts=[float((directory/(r+'.started')).read_text()) for r in a.ROLES]
        finishes=[float((directory/(r+'.finished')).read_text()) for r in a.ROLES]
        assert max(starts)<min(finishes)


@pytest.mark.parametrize('rho_index',[0,1])
def test_actual_low_poison_native_50_epochs_eval_and_cache_reuse(tmp_path,monkeypatch,rho_index):
    """Toy CPU routing only; real native train/audits/eval, not HPE-Li efficacy."""
    import numpy as np
    import train_backdoor as trainer
    import models.factory as factory
    import run_a1_three_tables as frozen
    import run_peak_confirmation as confirmation
    class TinyData:
        def __init__(self,split,data_root,**kw):
            self.data_root=str(data_root)
            count=100 if split=='training' else 3
            self.items=[dict(csi=str(Path(data_root)/split/f'{i}.npy'),
                kpt=str(Path(data_root)/split/'pose.npy'),frame_idx=i) for i in range(count)]
            self.raw=np.random.default_rng(9).uniform(.1,.8,(3,114,10)).astype(np.float32)
            self.poses=np.random.default_rng(19).normal(0,.2,(100,1,17,3)).astype(np.float32)
        def __len__(self): return len(self.items)
        def load_raw(self,path): return self.raw.copy()
        def normalize(self,raw): return np.clip(raw,0,1).astype(np.float32)
        def load_pose(self,path,frame_idx): return self.poses[frame_idx].copy()
    class TinyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head=torch.nn.Linear(3,51)
        def forward(self,csi):
            return self.head(csi.mean((-1,-2))).reshape(-1,1,17,3),None
    cell=d.cells(tmp_path/'home',tmp_path/'runs'/a.PROFILE)[rho_index]
    cell['cfg'].update(device='cpu',num_workers=0)
    action=Path(cell['cfg']['action_npy']); action.parent.mkdir(parents=True)
    np.save(action,np.random.default_rng(23).normal(size=(1,3,30,25,1)).astype(np.float32))
    counts={'train':100,'eval':3}
    monkeypatch.setattr(confirmation,'EXPECTED_COUNTS',counts)
    monkeypatch.setattr(frozen,'EXPECTED_COUNTS',counts)
    monkeypatch.setattr(trainer,'MMFI',TinyData)
    monkeypatch.setattr(trainer,'build_model',lambda *a,**kw:TinyVictim())
    monkeypatch.setattr(factory,'build_model',lambda *a,**kw:TinyVictim())
    monkeypatch.setattr(d,'audit_input',frozen._current_inputs)
    original_evaluator=d.evaluator
    def evaluator():
        eva=original_evaluator(); eva.COUNTS=counts
        def dataset(cfg):
            from run_peak_confirmation import _pair_ids
            training=trainer._load_dataset(cfg,'training')
            test=trainer._load_dataset(cfg,'test')
            record={name:dict(n=len(data),pair_ids_sha256=frozen._sha_json(_pair_ids(data)))
                    for name,data in [('train',training),('eval',test)]}
            return test,record,_pair_ids(test)
        eva._dataset_record=dataset
        return eva
    monkeypatch.setattr(d,'evaluator',evaluator)
    threads=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        result=d.execute(cell,'toy-plan','toy')
        path=Path(cell['ckpt_dir'])/'checkpoint.pt'
        checkpoint=torch.load(path,weights_only=False)
        assert checkpoint['epoch']==49 and checkpoint['cfg']['rho']==cell['rho']
        manifest=frozen._json(Path(cell['ckpt_dir'])/'poison_manifest.json')
        assert manifest['n_poison']==(1 if rho_index==0 else 5)
        assert len(result['dose_rows'])==6 and result['metrics']['target_pck_pct'] is not None
        assert (Path(cell['artifact_dir'])/'per_sample.npz').exists()
        before=path.read_bytes()
        monkeypatch.setattr(trainer,'build_model',lambda *a,**kw:pytest.fail('Completed dose was retrained'))
        assert d.execute(cell,'toy-plan','toy')==result
        assert path.read_bytes()==before
    finally:
        torch.set_num_threads(threads)
