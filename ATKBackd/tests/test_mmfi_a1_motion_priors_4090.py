"""CPU contract fixtures; not evidence of full MM-Fi/CUDA efficacy."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_mmfi_a1_motion_priors_4090 as runner
import train_backdoor as original


def make_action(path, prior, phase=0., sidecar=True):
    values=np.zeros((2,3,10,25,1),dtype=np.float32)
    rng=np.random.default_rng(91)
    values[0,...,0]=rng.normal(size=(3,10,25))*.1+phase
    values[1,:,0:3,:,0]=1.
    path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,values)
    if sidecar:
        from vast_seed42_preflight import sha256
        runner.write(str(path)+'.provenance.json',dict(semantic_approved=True,action=prior,sha256=sha256(path)))
    return path


@pytest.fixture
def plan(tmp_path,monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(runner,'sources',lambda:{'fixture.py':'a'*64})
    home=tmp_path/'home'
    make_action(home/'actions/data_cross.npy','cross')
    make_action(home/'actions/data_nod.npy','nod',phase=.1)
    return runner.build_plan(home,home/'runs/new')


def args(plan,mode):
    return ['--data-home',plan['data_home'],'--outdir',plan['outdir'],'--mode',mode]


def test_two_new_scratch_a1_rows_without_hidden_victims(plan):
    assert plan['priors']==['cross','nod']
    assert plan['max_workers']==1
    assert [c['key'] for c in plan['cells']]==['a1','a1']
    assert all(c['origin']=='fresh' and not c['dependencies'] for c in plan['cells'])
    assert plan['contract']['clean_training'] is False
    assert plan['contract']['baseline_training'] is False
    assert 'not target-action' in plan['contract']['scope']
    assert plan['contract']['equal_realized_noise'] is False
    assert not Path(plan['outdir']).exists()


def test_only_action_and_explicit_profile_metadata_change_from_frozen_a1(plan):
    import run_a1_three_tables as frozen
    canonical=frozen._configuration(Path(plan['data_home']),'cuda:0',0,42,'a1')
    for cell in plan['cells']:
        cfg=copy.deepcopy(cell['cfg'])
        assert cfg.pop('motion_prior_profile')==runner.PROFILE
        assert cfg.pop('motion_prior_name')==cell['prior']
        cfg['action_npy']=canonical['action_npy']
        assert cfg==canonical
        assert (cfg['optimizer'],cfg['lr'],cfg['momentum'],cfg['weight_decay'])==('sgd',.001,.9,0.)
        assert (cfg['batch_size'],cfg['epochs'],cfg['victim_epochs'],cfg['num_workers'])==(32,50,50,0)
        assert cfg['rho']==.1 and cfg['eps']==.185
        assert cfg['trigger']=='md_multicarrier' and cfg['dose_coupling']=='paired'
        assert cfg['pivot']==1 and cfg['target_joints']==[2,3]
        assert cfg['payload_axis']==[0.,0.,1.] and cfg['theta_max_deg']==40.
        assert cfg['dose_grid']==runner.GRID and cfg['strict_resume'] is True
        assert cfg['mmfi_protocol']=='protocol1' and cfg['mmfi_setting']=='s1'


def test_selected_instance_and_movers_match_native_policy(plan):
    from attack.trigger import velocity_profiles_from_skeleton
    for cell in plan['cells']:
        record=cell['action']; velocity,position,movers=velocity_profiles_from_skeleton(record['path'],top_k=6)
        assert record['selected_instance']==0
        assert record['moving_joints']==list(map(int,movers))
        assert record['valid_frames']==10 and len(record['moving_joints'])==6
        assert record['fps']==30. and record['los']==[1.,0.,0.]
        assert velocity.shape==(6,9) and position.shape==(6,)


def test_native_trigger_deterministic_zero_identity_no_peak_reprojection(plan):
    cfg=plan['cells'][0]['cfg']
    trigger=original.build_trigger(cfg)
    data=np.random.default_rng(5).uniform(size=(3,114,10)).astype(np.float32)
    np.testing.assert_array_equal(trigger.inject(data.copy(),0.,cfg['eps']),data)
    expected=np.clip(data*np.clip(1.+cfg['eps']*trigger.pattern,0.,None).astype(np.float32),0.,1.)
    np.testing.assert_array_equal(trigger.inject(data.copy(),1.,cfg['eps']),expected)
    assert runner.trigger_record(cfg)==runner.trigger_record(cfg)
    assert set(runner.trigger_record(cfg))=={'base_complex','base_zero_mean','p0','p1','pattern'}


@pytest.mark.parametrize('shape',[(2,3,10,25),(0,3,10,25,1),(2,2,10,25,1),(2,3,1,25,1),(2,3,10,24,1)])
def test_invalid_action_shapes_fail_before_trigger(tmp_path,shape):
    path=tmp_path/'action.npy'; np.save(path,np.ones(shape,dtype=np.float32))
    with pytest.raises(ValueError,match='NTU'):
        runner.action_record(path,'cross',True)


@pytest.mark.parametrize('value',[np.nan,np.inf,complex(1,1)])
def test_nonfinite_or_complex_action_rejected(tmp_path,value):
    path=tmp_path/'action.npy'
    array=np.full((1,3,3,25,1),value,dtype=np.complex64 if isinstance(value,complex) else np.float32)
    np.save(path,array)
    with pytest.raises(ValueError,match='finite real'):
        runner.action_record(path,'nod',True)


def test_pickle_action_not_loaded(tmp_path):
    path=tmp_path/'object.npy'; np.save(path,np.array([{'untrusted':True}],dtype=object))
    with pytest.raises(ValueError): runner.action_record(path,'cross',True)


def test_missing_prior_never_substitutes_bend(tmp_path):
    with pytest.raises(FileNotFoundError,match='Missing nod'):
        runner.action_record(tmp_path/'data_nod.npy','nod',True)


def test_unsourced_prior_requires_explicit_semantic_attestation(tmp_path):
    path=make_action(tmp_path/'cross.npy','cross',sidecar=False)
    with pytest.raises(ValueError,match='attest'):
        runner.action_record(path,'cross')
    record=runner.action_record(path,'cross',True)
    assert record['provenance']['kind']=='explicit_user_action_label_attestation'


@pytest.mark.parametrize('field,value',[('action','nod'),('sha256','forged'),('semantic_approved',False)])
def test_incorrect_sidecar_never_bypassed_by_approval(tmp_path,field,value):
    path=make_action(tmp_path/'cross.npy','cross'); sidecar=Path(str(path)+'.provenance.json')
    record=runner.read(sidecar); record[field]=value; runner.write(sidecar,record)
    with pytest.raises(ValueError,match='provenance'):
        runner.action_record(path,'cross',True)


def test_bend_requires_historical_sha_not_just_filename(tmp_path):
    path=make_action(tmp_path/'data_bend.npy','bend')
    with pytest.raises(ValueError,match='archived original'):
        runner.action_record(path,'bend',True)


@pytest.mark.parametrize('kind',['allzero','static'])
def test_degenerate_motion_rejected(tmp_path,kind):
    path=tmp_path/'cross.npy'
    np.save(path,np.zeros((1,3,10,25,1),dtype=np.float32) if kind=='allzero' else np.ones((1,3,10,25,1),dtype=np.float32))
    with pytest.raises(ValueError,match='valid frames|degenerate radial'):
        runner.action_record(path,'cross',True)


@pytest.mark.parametrize('priors',[[],['cross','cross'],['unknown']])
def test_invalid_prior_selection_rejected(tmp_path,priors):
    with pytest.raises(ValueError,match='distinct'):
        runner.build_plan(tmp_path,tmp_path/'runs/new',priors)


@pytest.mark.parametrize('relative',['','runs','elsewhere/new'])
def test_output_must_be_new_dedicated_run(tmp_path,relative):
    with pytest.raises(ValueError,match='Outdir'):
        runner.build_plan(tmp_path,tmp_path/relative)


def test_same_action_bytes_under_two_names_rejected(tmp_path):
    home=tmp_path/'home'
    for name in ('cross','nod'): make_action(home/f'actions/data_{name}.npy',name)
    with pytest.raises(ValueError,match='same action bytes'):
        runner.build_plan(home,home/'runs/new')


def test_no_visible_device_remapping(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='physical cuda:0'):
        runner.build_plan(tmp_path,tmp_path/'runs/new')


def test_plan_hash_binds_prior_bytes_pattern_recipe_runtime(plan):
    payload=copy.deepcopy(plan); expected=payload.pop('plan_sha256')
    assert runner.digest(payload)==expected
    for key in ('action','trigger_arrays','cfg'):
        modified=copy.deepcopy(payload); modified['cells'][0][key]['new']='changed'
        assert runner.digest(modified)!=expected


def test_private_train_patch_is_only_atomic_checkpoint_write():
    before=original._save_checkpoint
    engine=runner.training_engine()
    assert engine is not original and engine._save_checkpoint is not before
    assert original._save_checkpoint is before
    assert engine._victim_update.__code__.co_code==original._victim_update.__code__.co_code
    assert engine.build_trigger.__code__.co_code==original.build_trigger.__code__.co_code


def test_atomic_checkpoint_and_exact_cpu_rng_optimizer_resume(tmp_path):
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    engine=runner.training_engine()
    model=torch.nn.Linear(4,3); optimizer=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9)
    cfg=dict(epochs=50,seed=42)
    optimizer.zero_grad(); model(torch.ones(2,4)).square().mean().backward(); optimizer.step()
    path=tmp_path/'checkpoint.pt'; engine._save_checkpoint(path,model,optimizer,7,.3,cfg)
    assert path.is_file() and not path.with_name(path.name+'.motion-prior.tmp').exists()
    expected=(torch.rand(3),np.random.random(3),random.random())
    saved=copy.deepcopy(optimizer.state_dict()); fresh=torch.nn.Linear(4,3)
    fresh_opt=torch.optim.SGD(fresh.parameters(),lr=.001,momentum=.9)
    assert engine._load_checkpoint(path,fresh,fresh_opt,'cpu',cfg)==(7,.3)
    torch.testing.assert_close(torch.rand(3),expected[0],rtol=0,atol=0)
    np.testing.assert_array_equal(np.random.random(3),expected[1]); assert random.random()==expected[2]
    for a,b in zip(fresh.parameters(),model.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert fresh_opt.state_dict()['param_groups']==saved['param_groups']
    for key in saved['state']:
        torch.testing.assert_close(fresh_opt.state_dict()['state'][key]['momentum_buffer'],saved['state'][key]['momentum_buffer'],rtol=0,atol=0)


def test_evaluator_private_hash_hook_does_not_modify_original():
    import a1_three_tables_eval
    from vast_seed42_preflight import sha256
    before=a1_three_tables_eval._source_hashes
    sources={'train_backdoor.py':sha256(runner.HERE/'train_backdoor.py')}
    engine=runner.evaluator(sources)
    assert engine._source_hashes()==sources
    assert a1_three_tables_eval._source_hashes is before


@contextmanager
def no_lock(path): yield


def setup_main(plan,monkeypatch):
    import run_mmfi_tables
    monkeypatch.setattr(runner,'build_plan',lambda *a,**k:copy.deepcopy(plan))
    monkeypatch.setattr(run_mmfi_tables,'_cell_lock',no_lock)


def test_cpu_audit_never_checks_gpu_or_launches_worker(plan,monkeypatch):
    import vast_seed42_preflight
    setup_main(plan,monkeypatch)
    monkeypatch.setattr(vast_seed42_preflight,'inspect',lambda home:{'full_cpu_fixture':True})
    monkeypatch.setattr(runner,'require_idle',lambda:pytest.fail('CPU audit touched GPU'))
    monkeypatch.setattr(runner,'child',lambda *a:pytest.fail('CPU audit launched worker'))
    monkeypatch.setattr(runner,'execution',lambda:pytest.fail('CPU audit initialized CUDA'))
    runner.main(args(plan,'audit'))
    assert runner.read(Path(plan['outdir'])/runner.PLAN)==plan
    assert runner.read(Path(plan['outdir'])/'inputs.json')=={'full_cpu_fixture':True}


def test_prepare_when_busy_cannot_launch_gpu_work(plan,monkeypatch):
    import vast_seed42_preflight
    setup_main(plan,monkeypatch)
    monkeypatch.setattr(vast_seed42_preflight,'inspect',lambda home:{'full_cpu_fixture':True})
    monkeypatch.setattr(runner,'require_idle',lambda:(_ for _ in ()).throw(ValueError('busy')))
    monkeypatch.setattr(runner,'child',lambda *a:pytest.fail('busy run launched smoke'))
    with pytest.raises(ValueError,match='busy'): runner.main(args(plan,'prepare'))


def test_serial_launch_requires_authenticated_completion_before_next(plan,monkeypatch):
    setup_main(plan,monkeypatch); runner.write(Path(plan['outdir'])/runner.PLAN,plan)
    monkeypatch.setattr(runner,'require_idle',lambda:None)
    monkeypatch.setattr(runner,'gate_check',lambda p:None)
    done=set(); events=[]
    monkeypatch.setattr(runner,'completed',lambda p,c:{'complete':True} if c['prior'] in done else None)
    def child(p,c,work):
        assert work=='run'; events.append(c['prior']); done.add(c['prior'])
    monkeypatch.setattr(runner,'child',child); monkeypatch.setattr(runner,'export',lambda p:events.append('export'))
    runner.main(args(plan,'launch'))
    assert events==['cross','nod','export']


@pytest.mark.parametrize('failure',['worker_error','missing_marker','busy','gate'])
def test_failure_stops_no_retry_or_later_prior(plan,monkeypatch,failure):
    setup_main(plan,monkeypatch); runner.write(Path(plan['outdir'])/runner.PLAN,plan)
    monkeypatch.setattr(runner,'require_idle',lambda:None)
    monkeypatch.setattr(runner,'gate_check',lambda p:None)
    monkeypatch.setattr(runner,'completed',lambda *a:None)
    calls=[]
    def fail(*a): raise ValueError('failure fixture')
    def child(p,c,w):
        calls.append(c['prior'])
        if failure=='worker_error':fail()
    monkeypatch.setattr(runner,'child',child)
    monkeypatch.setattr(runner,'export',lambda *a:pytest.fail('export after failure'))
    if failure=='busy':monkeypatch.setattr(runner,'require_idle',fail)
    if failure=='gate':monkeypatch.setattr(runner,'gate_check',fail)
    with pytest.raises(ValueError):runner.main(args(plan,'launch'))
    assert calls==([] if failure in ('busy','gate') else ['cross'])


def fake_gates(plan,monkeypatch):
    monkeypatch.setattr(runner,'verify_inputs',lambda *a,**k:None)
    monkeypatch.setattr(runner,'binding',lambda p,c:dict(prior=c['prior']))
    for c in plan['cells']:
        runner.write(Path(c['run_dir'])/'gpu_smoke.json',dict(binding=runner.binding(plan,c),
            physical_batch=32,optimizer_steps=2,real_train_input=True,poison_examples=8,
            parameter_updated=True,losses=[.4,.3]))


def test_both_b32_gates_required(plan,monkeypatch):
    fake_gates(plan,monkeypatch);runner.gate_check(plan)
    (Path(plan['cells'][1]['run_dir'])/'gpu_smoke.json').unlink()
    with pytest.raises(FileNotFoundError):runner.gate_check(plan)


@pytest.mark.parametrize('key,value',[('binding',{}),('physical_batch',16),('optimizer_steps',1),
    ('real_train_input',False),('poison_examples',0),('parameter_updated',False),('losses',[]),('losses',[.3,float('inf')])])
def test_altered_gate_fails_closed(plan,monkeypatch,key,value):
    fake_gates(plan,monkeypatch);path=Path(plan['cells'][0]['run_dir'])/'gpu_smoke.json'
    record=runner.read(path);record[key]=value;path.write_text(json.dumps(record),encoding='utf-8')
    with pytest.raises(ValueError,match='preparation gate'):runner.gate_check(plan)


def test_changed_action_or_sidecar_detected_after_audit(plan,monkeypatch):
    import vast_seed42_preflight
    monkeypatch.setattr(vast_seed42_preflight,'check_receipt',lambda *a,**k:None)
    runner.write(Path(plan['outdir'])/'inputs.json',dict(data_home=plan['data_home'],splits=dict(train={'n':133056},eval={'n':33264})))
    runner.verify_inputs(plan)
    sidecar=Path(plan['cells'][0]['action']['path']+'.provenance.json')
    provenance=runner.read(sidecar);provenance['new_metadata']='changed';runner.write(sidecar,provenance)
    with pytest.raises(ValueError,match='changed after Prepare'):runner.verify_inputs(plan)


def test_export_only_five_metrics_and_explicit_single_seed_limitations(plan,monkeypatch):
    metrics=dict(t1_mpjpe_mm=100.123,target_pck_pct=65.1,mean_dce_mm=55.2,clean_mpjpe_mm=174.3,clean_pampjpe_mm=104.1)
    monkeypatch.setattr(runner,'verify_inputs',lambda *a:None)
    monkeypatch.setattr(runner,'completed',lambda p,c:{'metrics':metrics} if c['prior']=='cross' else None)
    monkeypatch.setattr(runner,'child',lambda *a:pytest.fail('export launched training'))
    runner.export(plan)
    report=runner.read(Path(plan['outdir'])/'motion_prior_results.json')
    assert [r['status'] for r in report['rows']]==['historical_unverified_not_imported','complete','pending']
    assert report['bend_reference'].startswith('not imported')
    text=(Path(plan['outdir'])/'motion_prior_results.md').read_text(encoding='utf-8')
    assert '| cross | complete | 100.12 | 65.10 | 55.20 | 174.30 | 104.10 |' in text
    assert '| nod | pending | -- | -- | -- | -- | -- |' in text
    assert 'Single seed42' in text and 'not pose targets' in text


def test_child_uses_selected_priors_paths_and_explicit_approval_only(plan,monkeypatch):
    commands=[]
    monkeypatch.setattr(runner.subprocess,'run',lambda cmd,**kw:commands.append((cmd,kw)) or SimpleNamespace(returncode=0))
    runner.child(plan,plan['cells'][0],'run')
    cmd,kw=commands[0]
    assert cmd[cmd.index('--priors')+1:cmd.index('--priors')+3]==['cross','nod']
    assert '--cross-action' in cmd and '--nod-action' in cmd
    assert '--approve-action-labels' not in cmd
    assert kw['stderr']==runner.subprocess.STDOUT


def test_incomplete_resume_rejected_before_native_train(plan,monkeypatch):
    cell=plan['cells'][0];folder=Path(cell['run_dir'])
    runner.write(folder/'gpu_smoke.json',{'execution':{'fixture':True}})
    runner.write(folder/'execution_binding.json',dict(binding={'fixture':True},execution={'fixture':True}))
    (folder/'checkpoint.pt').touch()
    monkeypatch.setattr(runner,'verify_inputs',lambda *a:None)
    monkeypatch.setattr(runner,'execution',lambda:{'fixture':True})
    monkeypatch.setattr(runner,'binding',lambda *a:{'fixture':True})
    monkeypatch.setattr(runner,'current_inputs',lambda *a:{'fixture':True})
    monkeypatch.setattr(torch,'load',lambda *a,**k:{'epoch':3})
    monkeypatch.setattr(runner,'training_engine',lambda:pytest.fail('bad checkpoint reached train'))
    with pytest.raises(ValueError,match='no restart fallback'):runner.run_cell(plan,cell)


def test_nonempty_foreign_output_is_preserved(tmp_path):
    folder=tmp_path/'runs/historical'; folder.mkdir(parents=True)
    path=folder/'existing-result.json'; path.write_text('{"unchanged":true}',encoding='utf-8')
    with pytest.raises(ValueError,match='preserve historical'):
        runner.build_plan(tmp_path,folder)
    assert path.read_text(encoding='utf-8')=='{"unchanged":true}'
    assert not (folder/runner.PLAN).exists()


def test_evaluator_detects_source_mutation(tmp_path,monkeypatch):
    from vast_seed42_preflight import sha256
    path=tmp_path/'source.py'; path.write_text('original',encoding='utf-8')
    monkeypatch.setattr(runner,'HERE',tmp_path)
    monkeypatch.setattr(runner,'private_module',lambda name:SimpleNamespace())
    evaluator=runner.evaluator({'source.py':sha256(path)})
    assert evaluator._source_hashes()=={'source.py':sha256(path)}
    path.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='Scientific source changed'):evaluator._source_hashes()


def test_helper_device_argument_accepted_for_cpu_audit(plan,monkeypatch):
    import vast_seed42_preflight
    setup_main(plan,monkeypatch)
    monkeypatch.setattr(vast_seed42_preflight,'inspect',lambda home:{'cpu_fixture':True})
    monkeypatch.setattr(runner,'require_idle',lambda:pytest.fail('CPU audit initialized GPU'))
    runner.main(args(plan,'audit')+['--device','cuda:0'])
    assert runner.read(Path(plan['outdir'])/'inputs.json')=={'cpu_fixture':True}


class SyntheticMMFI:
    def __init__(self):
        rng=np.random.default_rng(88)
        self.raw=rng.uniform(.1,.9,size=(64,3,114,10)).astype(np.float32)
        self.pose=rng.normal(0,.2,size=(64,1,17,3)).astype(np.float32)
        self.items=[dict(csi=i,kpt=i,frame_idx=0) for i in range(64)]
    def __len__(self):return len(self.raw)
    def load_raw(self,index):return self.raw[index].copy()
    def load_pose(self,index,frame_idx):return self.pose[index].copy()
    def normalize(self,value):return value


def test_native_poison_ids_doses_and_target_labels_identical_across_priors(plan):
    from run_mmfi_p3_s3_three_attacks_4090 import paired_data
    base=SyntheticMMFI()
    pairs=[paired_data(base,original.build_trigger(c['cfg']),c['cfg']) for c in plan['cells']]
    assert pairs[0].poison_plan==pairs[1].poison_plan
    assert pairs[0].payload_dose_of==pairs[1].payload_dose_of
    assert pairs[0].n_poison==6
    for index in sorted(pairs[0].poison_idx):
        a,b=pairs[0][index],pairs[1][index]
        assert set(a)==set(b)=={'csi','pose'}
        np.testing.assert_array_equal(a['pose'],b['pose'])
        assert not np.array_equal(a['csi'],b['csi'])


def test_actual_hpeli_b32_two_sgd_updates_cpu_synthetic(plan):
    from attack.poison import collate
    from run_mmfi_p3_s3_three_attacks_4090 import paired_data
    cfg=plan['cells'][0]['cfg']; engine=runner.training_engine()
    torch.manual_seed(42)
    model=engine.build_model('hpeli',num_keypoints=17,subcarrier_num=114,dataset='mmfi',pretrained=False)
    _,optimizer=engine._build_optimizer(model,cfg,'mmfi')
    data=paired_data(SyntheticMMFI(),engine.build_trigger(cfg),cfg)
    parameter=next(p for p in model.parameters() if p.requires_grad)
    before=parameter.detach().clone(); losses=[]
    for start in (0,32):
        batch=collate([data[i] for i in range(start,start+32)])
        assert batch['csi'].shape==(32,3,114,10) and batch['pose'].shape==(32,1,17,3)
        losses.append(engine._victim_update(model,optimizer,batch['csi'],batch['pose']))
    assert np.isfinite(losses).all() and not torch.equal(before,parameter.detach())
    assert optimizer.param_groups[0]['lr']==.001 and optimizer.param_groups[0]['momentum']==.9


def test_status_excludes_utf16_controller_logs(plan,monkeypatch,capsys):
    setup_main(plan,monkeypatch)
    logs=Path(plan['outdir'])/'logs';logs.mkdir(parents=True)
    (logs/'run.cross.1.log').write_text('CELL_PROGRESS epoch5',encoding='utf-8')
    (logs/'controller.log').write_text('WRONG_CONTROLLER_PROGRESS',encoding='utf-16')
    runner.main(args(plan,'status'))
    output=capsys.readouterr().out
    assert 'CELL_PROGRESS epoch5' in output and 'controller.log' not in output
