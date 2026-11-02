"""Synthetic CPU tests; not full-data/CUDA convergence or efficacy evidence."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ATKBackd'))
import run_mmfi_p1s1_adap_patch_4090 as r


class Base:
    def __init__(self,n=100):
        self.items=[{'name':f'E1_S1_A1_{i}','csi':f'x{i}','kpt':f'y{i}','frame_idx':i} for i in range(n)]
    def __len__(self):
        return len(self.items)
    def load_raw(self,path):
        return np.full((3,114,10),.25,dtype=np.float32)
    def load_pose(self,path,index):
        pose=np.arange(51,dtype=np.float32).reshape(1,17,3)/100
        return pose+int(index)*.0001
    def normalize(self,x):
        return x.copy()


class Mark:
    def inject_train(self,x,dose,group):
        return x+float(dose)*.1


def pairs(n=100):
    return r.paired_data(Base(n),{'eps':.185,'payload_axis':[0.,0.,1.]})


def test_configuration_matches_common_a1(tmp_path):
    from run_a1_three_tables import _configuration
    common=_configuration(tmp_path,'cuda:0',0,42,'a1')
    cfg=r.configuration(tmp_path)
    for key in ('model','optimizer','lr','momentum','weight_decay','batch_size','epochs',
                'num_workers','rho','dose_min','dose_max','dose_coupling','target_joints',
                'theta_max_deg','victim_loss','mmfi_protocol','mmfi_setting'):
        assert cfg[key]==common[key]
    assert cfg['optimizer']=='sgd' and cfg['lr']==.001
    assert cfg['clean_label_cover_ratio']==.2
    assert cfg['adap_patch']['source_preset']=='gtsrb'
    assert not any(k.startswith('comparison_peak_') for k in cfg)


def test_plan_explicit_scope_and_output(tmp_path):
    plan=r.build_plan(tmp_path,tmp_path/'runs'/r.DEFAULT_RUN)
    assert plan['contract']['counts']=={'training':133056,'test':33264}
    assert plan['contract']['total_modified_rate']==.3
    assert plan['contract']['clean_model_trigger_control'] is False
    assert plan['operator']['test_names']==['firefox_corner_32.png','trojan_square_32.png']
    body={k:v for k,v in plan.items() if k!='plan_sha256'}
    assert r.digest(body)==plan['plan_sha256']
    for name in ('run_mmfi_p1s1_adap_patch_4090.py','attack/adap_patch_mmfi.py','run_mmfi_p3_splits.py'):
        assert name in plan['sources']


@pytest.mark.parametrize('device,jobs',[('cpu',1),('cuda:1',1),('cuda:0',2),('cuda:0',True)])
def test_serial_gpu_contract(tmp_path,device,jobs):
    with pytest.raises(ValueError):
        r.build_plan(tmp_path,tmp_path/'runs'/r.DEFAULT_RUN,device,jobs)


def test_historical_outputs_forbidden(tmp_path):
    with pytest.raises(ValueError,match='dedicated output'):
        r.build_plan(tmp_path,tmp_path/'runs'/'mmfi_p3_hpeli_s42_v1')


def test_common_poison_doses_preserved_and_covers_disjoint():
    data=pairs(); record,values=r.assignments(data)
    assert record['n_poison']==10 and record['n_cover']==20
    assert not set(values['poison_ids']) & set(values['cover_ids'])
    assert dict(zip(values['poison_ids'],values['poison_doses']))==data.dose_of
    assert r.assignments(pairs())[0]==record
    assert record['common_poison_sha256']==r.digest(data.manifest())
    assert values['poison_groups'].tolist()==[0,0,0,1,1,2,2,2,3,3]


def test_training_covers_keep_true_pose_and_only_normalized_input_changes():
    data=pairs(); rows=r.TrainingRows(data,Mark())
    assert set(rows[0])=={'csi','pose'}
    for index in rows.cover:
        before=data[index]; after=rows[index]
        assert np.array_equal(after['pose'],before['pose'])
        assert np.array_equal(after['pose'],before['clean_pose'])
        assert not np.array_equal(after['csi'],before['csi'])
    for index in rows.poison:
        assert np.array_equal(rows[index]['pose'],data[index]['pose'])
        assert not np.array_equal(rows[index]['pose'],data[index]['clean_pose'])
    index=next(i for i in range(len(rows)) if i not in rows.cover and i not in rows.poison)
    assert np.array_equal(rows[index]['csi'],data[index]['csi'])


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.fc=torch.nn.Linear(3,51)
    def forward(self,x):
        return self.fc(x.mean(dim=(-2,-1))).reshape(-1,1,17,3),None


def train_toy(epochs,start=None):
    from train_backdoor import _rng_state,_restore_rng_state
    import random
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model=Toy(); opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=0.)
    cfg={'test':'cfg'}; bind={'config_sha256':r.digest(cfg),'steps_per_epoch':4}
    rows=r.TrainingRows(pairs(100),Mark()); first=0; steps=0
    if start:
        r.validate_checkpoint(start,bind,cuda=False)
        model.load_state_dict(start['model']); opt.load_state_dict(start['optimizer'])
        _restore_rng_state(start['rng']); first=start['epoch']+1; steps=start['optimizer_steps']
    for epoch in range(first,epochs):
        for batch in r.loader(rows):
            assert np.isfinite(r.ordinary_step(model,opt,batch,'cpu')); steps+=1
    return deepcopy(dict(binding=bind,cfg=cfg,epoch=epochs-1,optimizer_steps=steps,
        model=model.state_dict(),optimizer=opt.state_dict(),rng=_rng_state()))


def test_exact_rng_optimizer_resume_cpu():
    final=train_toy(4); interrupted=train_toy(2); resumed=train_toy(4,interrupted)
    assert final['optimizer_steps']==resumed['optimizer_steps']==16
    for name,value in final['model'].items():
        assert torch.equal(value,resumed['model'][name])
    for key,entry in final['optimizer']['state'].items():
        assert torch.equal(entry['momentum_buffer'],resumed['optimizer']['state'][key]['momentum_buffer'])
    assert torch.equal(final['rng']['torch'],resumed['rng']['torch'])


@pytest.mark.parametrize('kind',['binding','epoch','steps','rng','momentum','lr','finite'])
def test_checkpoint_corruption_fails_closed(kind):
    blob=train_toy(1); binding=deepcopy(blob['binding'])
    if kind=='binding': blob['binding']={'bad':1}
    elif kind=='epoch': blob['epoch']=50
    elif kind=='steps': blob['optimizer_steps']=1
    elif kind=='rng': blob['rng'].pop('torch')
    elif kind=='momentum': next(iter(blob['optimizer']['state'].values())).pop('momentum_buffer')
    elif kind=='lr': blob['optimizer']['param_groups'][0]['lr']=.01
    elif kind=='finite': next(iter(blob['model'].values())).fill_(float('nan'))
    with pytest.raises(ValueError): r.validate_checkpoint(blob,binding,cuda=False)


def test_cuda_resume_requires_cuda_rng():
    blob=train_toy(1)
    with pytest.raises(ValueError,match='RNG'): r.validate_checkpoint(blob,blob['binding'])


def test_metrics_are_target_only_and_mean_five_positive_doses():
    rng=np.random.default_rng(3)
    truth=rng.normal(size=(8,17,3)); clean=truth.copy(); predictions=[]; targets=[]
    for dose in r.GRID:
        target=truth.copy(); target[:,[2,3],0]+=dose*.4
        pred=truth.copy(); pred[:,[2,3],0]+=dose*.3
        targets.append(target); predictions.append(pred)
    main,rows,arrays=r.evaluation_metrics(clean,truth,predictions,targets)
    assert main['t_mpjpe_mm']==pytest.approx(100)
    assert main['mean_positive_dce_mm']==pytest.approx(60)
    assert rows[0]['dce_mm']==0
    assert arrays[-1]['target_error_mm'].shape==(8,2)
    assert main['clean_mpjpe_mm']==0
    with pytest.raises(ValueError): r.evaluation_metrics(clean,truth,predictions[1:],targets[1:])


def test_npz_export_replay_and_no_overwrite(tmp_path):
    target=tmp_path/'per_sample.npz'; values={'truth':np.zeros((2,17,3)),'names':np.asarray(['a','b'])}
    r.export_arrays(target,values); first=r.sha(target); r.export_arrays(target,values)
    assert r.sha(target)==first
    with pytest.raises(ValueError,match='differs'):
        r.export_arrays(target,{**values,'truth':np.ones((2,17,3))})
    assert r.sha(target)==first


def test_status_reads_metadata_without_buildplan_gpu_checkpoint(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(r,'build_plan',lambda *a:pytest.fail('no build_plan in Status'))
    monkeypatch.setattr(r,'gpu',lambda:pytest.fail('no GPU in Status'))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('no checkpoint load in Status'))
    folder=tmp_path/r.METHOD; folder.mkdir(); (folder/'checkpoint.pt').write_bytes(b'not torch')
    r.main(['--outdir',str(tmp_path),'--mode','status'])
    assert 'UNKNOWN' in capsys.readouterr().out


def test_dryrun_has_no_gpu_network_or_output_writes(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(r,'gpu',lambda:pytest.fail('no GPU'))
    monkeypatch.setattr(r,'operator',lambda *a,**k:pytest.fail('no asset download'))
    r.main(['--data-home',str(tmp_path),'--mode','dry-run'])
    assert not (tmp_path/'runs').exists()
    assert 'DRY_RUN_ONLY' in capsys.readouterr().out


def test_child_failure_stops_without_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(r.subprocess,'run',lambda *a,**k:type('Exit',(),{'returncode':7})())
    with pytest.raises(RuntimeError,match='no retry'):
        r.child({'outdir':str(tmp_path),'home':str(tmp_path),'execution_policy':{'allow_concurrent_p3s3':False}},'train')


def test_gate_rejects_foreign_cuda_smoke(tmp_path,monkeypatch):
    plan={'outdir':str(tmp_path),'plan_sha256':'plan'}
    inputs={'poison':{'ids':[1]},'operator':{'asset':'bound'}}
    monkeypatch.setattr(r,'verify_inputs',lambda _:inputs)
    monkeypatch.setattr(r,'gpu',lambda:{'gpu':'4090'})
    r.write(tmp_path/'inputs.json',inputs)
    record=dict(plan_sha256='plan',execution={'gpu':'4090'},real_input=True,batch_size=32,
        steps=2,weights_changed=True,composition={'poison':4,'cover':4,'clean':24},
        poison_sha256=r.digest(inputs['poison']),operator=inputs['operator'],losses=[1.,.9])
    r.write(tmp_path/'smoke.json',record)
    gate=dict(plan_sha256='plan',execution={'gpu':'4090'},inputs_sha256=r.sha(tmp_path/'inputs.json'),
        smoke_sha256=r.sha(tmp_path/'smoke.json'),jobs_per_gpu=1)
    r.write(tmp_path/'gpu_gate.json',gate)
    assert r.gate_check(plan)==inputs
    record['weights_changed']=False; r.write(tmp_path/'smoke.json',record)
    with pytest.raises(ValueError): r.gate_check(plan)


def test_launch_controller_never_initializes_cuda_before_worker(tmp_path,monkeypatch):
    plan={'outdir':str(tmp_path),'home':str(tmp_path),'plan_sha256':'test'}
    monkeypatch.setattr(r,'build_plan',lambda *a:plan)
    monkeypatch.setattr(r,'require_idle',lambda *a:None)
    monkeypatch.setattr(r,'gpu',lambda:pytest.fail('controller must not initialize CUDA'))
    monkeypatch.setattr(r.shutil,'disk_usage',lambda p:type('Disk',(),{'free':6*2**30})())
    calls=[]
    monkeypatch.setattr(r,'execution_probe',lambda p:{'gpu':'4090'})
    monkeypatch.setattr(r,'gate_check',lambda p,**k:calls.append(('gate',k)))
    monkeypatch.setattr(r,'child',lambda p,work:calls.append(('child',work)))
    done=iter([None,{'status':'complete'}])
    monkeypatch.setattr(r,'completed',lambda p:next(done))
    r.main(['--data-home',str(tmp_path),'--mode','launch'])
    assert calls==[('gate',{'execution':{'gpu':'4090'}}),('child','train')]


def test_execution_probe_runs_short_lived_worker_and_parses_exact_receipt(monkeypatch):
    seen=[]
    def run(command,**kwargs):
        seen.append((command,kwargs))
        return type('Output',(),{'stdout':'EXECUTION_JSON={"gpu": "4090"}\n'})()
    monkeypatch.setattr(r.subprocess,'run',run)
    assert r.execution_probe({'home':'home','outdir':'output','execution_policy':{'allow_concurrent_p3s3':False}})=={'gpu':'4090'}
    assert seen[0][0][-2:]==['--work','execution']
    assert seen[0][1]['check'] is True


def test_completed_metrics_require_full_doses_and_consistent_summary():
    rng=np.random.default_rng(1); truth=rng.normal(size=(3,17,3))
    main,rows,_=r.evaluation_metrics(truth,truth,[truth]*6,[truth]*6)
    row={'metrics':main,'dose_rows':rows}; r.validate_metrics(row)
    bad=deepcopy(row); bad['metrics']['mean_positive_dce_mm']=1.
    with pytest.raises(ValueError): r.validate_metrics(bad)
    bad=deepcopy(row); bad['dose_rows'].pop()
    with pytest.raises(ValueError): r.validate_metrics(bad)
    bad=deepcopy(row); bad['metrics']['clean_mpjpe_mm']=float('nan')
    with pytest.raises(ValueError): r.validate_metrics(bad)


def test_symlink_and_reparse_roots_rejected_before_resolve(tmp_path,monkeypatch):
    target=tmp_path/'redirected'; target.mkdir()
    original=Path.is_symlink
    monkeypatch.setattr(Path,'is_symlink',lambda self:self==target or original(self))
    with pytest.raises(ValueError,match='symbolic link'): r.reject_links(target/'nested')


def test_low_disk_space_stops_before_prepare_or_train(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'build_plan',lambda *a:{'outdir':str(tmp_path),'home':str(tmp_path)})
    monkeypatch.setattr(r,'require_idle',lambda *a:None)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda p:type('Disk',(),{'free':4*2**30})())
    monkeypatch.setattr(r,'operator',lambda *a,**k:pytest.fail('no download'))
    with pytest.raises(ValueError,match='5 GiB'):
        r.main(['--data-home',str(tmp_path),'--mode','prepare'])


def test_p1s1_dataset_passes_protocol1_not_protocol3(tmp_path,monkeypatch):
    import data_utils.feeder as feeder
    seen=[]
    def factory(split,root,**kwargs):
        seen.append((split,root,kwargs))
        return range(r.COUNTS[split])
    monkeypatch.setattr(feeder,'MMFI',factory)
    cfg=r.configuration(tmp_path)
    assert len(r.dataset({'cfg':cfg},'test'))==33264
    assert seen[0][2]==dict(protocol='protocol1',setting='s1',random_ratio=.8,random_seed=0)


def test_p1s1_split_allows_global_subject_overlap_across_different_actions():
    a={(f'E{i%4}',f'S{i%40}',f'A{i}') for i in range(448)}
    b={(f'E{i%4}',f'S{i%40}',f'A{i+448}') for i in range(112)}
    data={key:type('Data',(),{'items':[{'name':'_'.join(s),'frame_idx':0} for s in seq]})()
          for key,seq in (('training',a),('test',b))}
    splits={key:{'sequence_keys':seq} for key,seq in (('training',a),('test',b))}
    r.split_coverage(data,splits)
    splits['test']['sequence_keys']={next(iter(a))}|set(list(b)[:111])
    with pytest.raises(ValueError,match='sequence'):r.split_coverage(data,splits)


def parent_fixture(tmp_path):
    from adap_patch_shared_gpu import P3_RUN,P3_ENTRY,P3_RUNNER_SHA256
    from run_mmfi_p3_s3_three_attacks_4090 import input_stats
    home=tmp_path/'home'; home.mkdir()
    data=home/'datasets'/'Compress'; data.mkdir(parents=True)
    value=data/'frame.npy'; value.write_bytes(b'opaque CSI bytes not loaded')
    records=[dict(path='frame.npy',bytes=value.stat().st_size,sha256=r.sha(value))]
    root=home/'runs'/P3_RUN; root.mkdir(parents=True)
    parent=dict(profile=P3_RUN,contract={'counts':{'training':240570,'test':80190}},
        sources={P3_ENTRY:P3_RUNNER_SHA256})
    parent['plan_sha256']=r.digest(parent)
    receipt=dict(plan_sha256=parent['plan_sha256'],dataset_root=str(data),files=records,
        full_content_sha256=r.digest(records),input_stat_fingerprint=input_stats(data,records))
    r.write(root/'plan.json',parent);r.write(root/'inputs.json',receipt)
    r.write(root/'gpu_gate.json',dict(plan_sha256=parent['plan_sha256'],inputs_sha256=r.sha(root/'inputs.json')))
    return dict(home=str(home),cfg={'dataset_root':str(data)}),value,root


def test_reuses_bound_p3_audit_with_no_dataset_content_reads(tmp_path,monkeypatch):
    plan,value,root=parent_fixture(tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('no data content re-audit'))
    original=Path.open
    def check_open(self,*args,**kwargs):
        if self==value:pytest.fail('must not read/hash original CSI bytes')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',check_open)
    before=(root/'inputs.json').read_bytes()
    receipt,provenance=r.parent_audit(plan)
    assert provenance['dataset_content_reread'] is False
    assert receipt['files'][0]['path']=='frame.npy'
    assert (root/'inputs.json').read_bytes()==before


def test_reused_audit_rejects_changed_file_or_receipt_without_full_reaudit(tmp_path):
    plan,value,root=parent_fixture(tmp_path)
    value.write_bytes(b'changed')
    with pytest.raises(ValueError):r.parent_audit(plan)
    receipt=r.read(root/'inputs.json');receipt['plan_sha256']='tampered'
    r.write(root/'inputs.json',receipt)
    with pytest.raises(ValueError,match='provenance'):r.parent_audit(plan)
