"""Synthetic/CPU gates only, not real PiW dataset/CUDA/efficacy validation."""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ATKBackd'))
from attack import backtime_mmfi as core
from attack import backtime_piw as a
import run_piw_backtime_three_backbones as r


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Dataset:
    split = 'training'
    def __init__(self,n=8):
        rng = np.random.default_rng(37)
        self.x = rng.uniform(.05,.95,(n,3,180,20)).astype(np.float32)
        self.y = rng.normal(size=(n,1,14,3)).astype(np.float32)
        self.items = [dict(name=f'fixture_{i}',kpt=i) for i in range(n)]
    def __len__(self):
        return len(self.x)
    def __getitem__(self,i):
        return dict(csi=self.x[i].copy(),pose=self.y[i].copy(),name=self.items[i]['name'])
    def load_pose(self,key):
        return self.y[key].copy()


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,42)
    def forward(self,x):
        return self.linear(x.mean((-1,-2))).reshape(-1,1,14,3), None


def trigger(cfg=None):
    cfg = cfg or a.configuration()
    variables = np.arange(cfg.n_attacked_variables)
    features = torch.linspace(-1,1,len(variables)*60).reshape(len(variables),60)
    return a.FrozenPiWBackTimeTrigger(cfg,core._GraphGenerator(cfg,features,variables,.5,.2))


def historical_clean(tmp_path, backbone, cfg, inputs, test, *, mutate_cfg=None, mutate_blob=None):
    """Historical layout fixture only, not evidence of full PiW training."""
    root = tmp_path / ('piw_historical_' + backbone)
    source = r.base.configuration(tmp_path/'home', root, device='cpu')
    source['drop_last'] = True
    if backbone == 'hpeli':
        profile, layout = 'piw_hpeli_body_axis_s42_lr001_control_v3', 'hpeli_v3'
        manifest_path, folder = root/'piw_lr_control.resolved.json', root/'hpeli/seed42/clean'
    else:
        profile, layout = 'piw3d_two_models_s42_six_baselines_v1', 'multi'
        source['recipes'] = deepcopy(r.RECIPES)
        source['test_checkpoint_selection'] = False
        manifest_path, folder = root/'plan.json', root/'attacks'/backbone/'seed42/clean'
    source['profile'] = profile
    if mutate_cfg:
        mutate_cfg(source)
    hashes = {n:r.base.file_sha(r.HERE/n) for n in ('data_utils/feeder.py', *r.clean_ref.MODEL_SOURCES[backbone])}
    manifest = r.signed(dict(profile=profile, cfg=source, inputs=inputs, sources=hashes, runtime={'fixture':True}))
    r.write_json(manifest_path, manifest)
    torch.manual_seed(42)
    model = Tiny()
    opt = r.optimizer(model, backbone)
    blob = dict(profile=profile, cfg=source, plan_sha256=manifest['plan_sha256'], epoch=199,
        model=model.state_dict(), optimizer=opt.state_dict(), rng={}, execution={'device':'historical'})
    if layout == 'hpeli_v3':
        blob['role'] = 'clean'
    else:
        blob.update(backbone=backbone, method='clean', poison={'n_poison':0,'n_cover':0}, trigger=None)
    if mutate_blob:
        mutate_blob(blob)
    folder.mkdir(parents=True)
    path = folder/'checkpoint.pt'
    torch.save(blob,path)
    truth = test.y[:,0]
    with torch.no_grad():
        pred = model(torch.tensor(test.x))[0][:,0].numpy()
    arrays = root/'predictions.npz' if layout == 'hpeli_v3' else folder/'predictions.npz'
    np.savez_compressed(arrays, truth=truth, names=[v['name'] for v in test.items],
                        **{('reference_predictions' if layout=='hpeli_v3' else 'predictions'):pred})
    result = dict(profile=profile, plan_sha256=manifest['plan_sha256'], clean_metrics=r.base._metrics(pred,truth))
    if layout == 'hpeli_v3':
        result.update(status='EXPLORATORY_PIW_LR_CONTROL_ONE_SEED_NOT_FULL_PAPER_TABLE',
                      artifacts={p.relative_to(root).as_posix():r.base.file_sha(p) for p in (path,arrays)})
        r.write_json(root/'evaluation.json',result)
    else:
        result.update(status='COMPLETE',backbone=backbone,method='clean',seed=42,
                      checkpoint_sha256=r.base.file_sha(path),predictions_sha256=r.base.file_sha(arrays))
        r.write_json(folder/'evaluation.json',result)
    return root


def toy_inputs(train, test):
    splits = {s:dict(retained_count=len(d),ordered_names=[v['name'] for v in d.items])
              for s,d in (('train_data',train),('test_data',test))}
    return dict(splits=splits,fingerprint=r.base.object_sha(splits))


def test_configuration_has_native_attacker_and_corrected_piw_recipes(tmp_path):
    cfg = r.config(tmp_path/'home',tmp_path/'output',device='cpu')
    assert cfg == json.loads(json.dumps(cfg))
    assert cfg['attacker']['shape'] == [3,90,20]
    assert cfg['attacker']['target_joints'] == [11,13]
    assert (cfg['attacker']['history_len'],cfg['attacker']['trigger_len']) == (12,8)
    assert cfg['attacker']['eps'] == .2 and cfg['attacker']['warmup'] == 10
    assert cfg['attacker']['fit_epochs'] == 100 and cfg['attacker']['batch_size'] == 64
    assert cfg['recipes']['metafiplusplus'] == dict(optimizer='adamw',lr=.01,weight_decay=.01,accumulation=1)
    assert cfg['recipes']['wiflow']['accumulation'] == 2
    assert cfg['epochs'] == 200 and cfg['final_epoch'] == 199 and cfg['drop_last']
    assert cfg['target']['pck_ref'] == [6,4] and cfg['target']['target_joints'] == [11,13]
    assert cfg['eval_doses'] == [0.,.2,.4,.6,.8,1.]
    assert cfg['backbones'] == ['hpeli','wiflow','metafiplusplus']
    assert cfg['roles'] == ['backtime']
    assert cfg['training_queue'] == [dict(backbone=b,role='backtime') for b in r.BACKBONES]
    assert cfg['clean_references'] == dict.fromkeys(r.BACKBONES)


@pytest.mark.parametrize('dose',[0.,.2,.6,1.,torch.tensor([0.,1.])])
def test_phase_history_and_sparse_variables_are_exact(dose):
    op = trigger()
    x = torch.tensor(Dataset(2).x)
    y = op.apply_tensor(x,dose)
    assert torch.equal(x[:,:,90:],y[:,:,90:])
    assert torch.equal(x[:,:,:90,:12],y[:,:,:90,:12])
    flat_x,flat_y = x[:,:,:90].reshape(2,270,20),y[:,:,:90].reshape(2,270,20)
    assert torch.equal(flat_x[:,81:],flat_y[:,81:])
    assert y.isfinite().all() and y.min()>=0 and y.max()<=1
    if isinstance(dose,torch.Tensor):
        assert torch.equal(x[0],y[0])
    if not isinstance(dose,torch.Tensor) and dose==0:
        assert torch.equal(x,y)
    np.testing.assert_array_equal(op.inject(x[0].numpy(),0),x[0].numpy())


@pytest.mark.parametrize('bad',[float('nan'),-.1,1.1,torch.ones(3)])
def test_bad_doses_rejected(bad):
    with pytest.raises(ValueError):
        trigger().apply_tensor(torch.tensor(Dataset(2).x),bad)


def test_invalid_inputs_and_piw_state_reject_mmfi():
    op = trigger()
    with pytest.raises(ValueError):
        op.apply_tensor(torch.ones(3,114,10),1)
    with pytest.raises(ValueError):
        op.inject(Dataset(1).x[0],1,eps=.185)
    restored = a.FrozenPiWBackTimeTrigger.from_state_dict(op.state_dict())
    x = torch.tensor(Dataset(1).x)
    assert torch.equal(op.apply_tensor(x,1),restored.apply_tensor(x,1))
    state = op.state_dict()
    state['generator']['conv1'][0,0] += 1
    with pytest.raises(ValueError,match='checksum'):
        a.FrozenPiWBackTimeTrigger.from_state_dict(state)
    with pytest.raises(ValueError):
        core.FrozenBackTimeTrigger.from_state_dict(op.state_dict())


def test_features_ignore_phase_and_include_all_amplitude_groups():
    data = Dataset(2)
    cfg = a.configuration()
    feat,mean,std = a.stream_features(data,cfg)
    raw = data.x[:,:,:90].astype(np.float64)
    assert mean == pytest.approx(raw.mean()) and std == pytest.approx(raw.std())
    data.x[:,:,90:] = .77
    other = a.stream_features(data,cfg)
    assert torch.equal(feat,other[0]) and (mean,std)==other[1:]
    data.x[:,2,80,:] += .01
    assert not torch.equal(feat,a.stream_features(data,cfg)[0])


def test_target_14joints_no_phase_or_global_axis_fallback():
    y = torch.tensor(Dataset(2).y)
    out = a.target_tensor(y,torch.tensor([0.,1.]))
    assert torch.equal(out[0],y[0])
    js = [j for j in range(14) if j not in (11,13)]
    assert torch.equal(out[:,:,js],y[:,:,js])
    y[:,:,5] = y[:,:,7]
    with pytest.raises(ValueError,match='Degenerate'):
        a.target_tensor(y,torch.ones(2))
    data = Dataset()
    data.y = np.zeros((8,1,17,3),np.float32)
    with pytest.raises(ValueError,match='14-joint'):
        a.stream_row(data,0,a.configuration())


def test_private_stream_loop_and_exact_resume():
    cfg = a.configuration(fit_epochs=4,warmup=0,batch_size=4,hidden_dim=4,variable_fraction=.02)
    data = Dataset(8)
    torch.manual_seed(42)
    model = Tiny()
    original = core._sha(model.state_dict())
    native_row, native_apply = core._stream_row,core._apply
    kw = dict(dataset_sha256='a'*64,n_poison=2,doses=np.linspace(.2,1,8,dtype=np.float32))
    full = a.fit_stream(model,data,cfg,**kw)
    snapshots = []
    def interrupt(state):
        snapshots.append(deepcopy(state))
        if state['epoch']==2:
            raise RuntimeError('test interruption')
    with pytest.raises(RuntimeError,match='test interruption'):
        a.fit_stream(model,data,cfg,save_state=interrupt,**kw)
    resumed = a.fit_stream(model,data,cfg,resume_state=snapshots[-1],**kw)
    assert core._sha(full.state_dict()) == core._sha(resumed.state_dict())
    assert core._sha(model.state_dict()) == original
    assert core._stream_row is native_row and core._apply is native_apply
    assert core.IMPLEMENTATION == 'independent-backtime-mmfi-pose-stream-v2'
    assert full.provenance()['diagnostics']['generator_updates'] == 6
    assert full.provenance()['diagnostics']['test_used'] is False
    altered = deepcopy(snapshots[-1])
    altered['binding']['dataset_sha256'] = 'b'*64
    with pytest.raises(ValueError,match='checksum'):
        a.fit_stream(model,data,cfg,resume_state=altered,**kw)
    data.split = 'validation'
    with pytest.raises(ValueError,match='TRAIN'):
        a.fit_stream(model,data,cfg,**kw)


def test_accumulation_tail_weighting_and_flush():
    assert [r.batch_groups(5,2,i) for i in range(5)] == [64,64,64,64,32]
    data = Dataset(96)
    from torch.utils.data import DataLoader
    torch.manual_seed(42)
    m = Tiny()
    opt = r.optimizer(m,'wiflow')
    reference = deepcopy(m)
    other = r.optimizer(reference,'wiflow')
    r.train_epoch(m,opt,DataLoader(data,batch_size=32,drop_last=True),'wiflow','cpu')
    for start,end in ((0,64),(64,96)):
        other.zero_grad(set_to_none=True)
        x,y = torch.tensor(data.x[start:end]),torch.tensor(data.y[start:end])
        torch.linalg.vector_norm(reference(x)[0]-y,dim=-1).mean().backward()
        other.step()
    for x,y in zip(m.parameters(),reference.parameters()):
        torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-7)
    assert all(p.grad is None for p in m.parameters())


def test_deferred_pairs_are_fixed_and_labels_only_target_joints():
    data = Dataset(8)
    doses = np.linspace(.2,1,8,dtype=np.float32)
    pairs = r.Pairs(data,[1,4],doses)
    for i in range(8):
        item = pairs[i]
        np.testing.assert_array_equal(item['csi'],data.x[i])
        assert item['poisoned'] == (i in (1,4))
        assert item['dose'] == (float(doses[i]) if i in (1,4) else 0.)
        np.testing.assert_array_equal(item['pose'][...,[0,1,2],:],data.y[i][...,[0,1,2],:])


@pytest.mark.parametrize('backbone',r.BACKBONES)
def test_real_piw_models_cpu_backward(backbone):
    r.seed('cpu')
    m = r.build_model(backbone,'cpu').train()
    x = torch.tensor(Dataset(2).x)
    y = torch.tensor(Dataset(2).y)
    opt = r.optimizer(m,backbone)
    loss = torch.linalg.vector_norm(r.prediction(m,x)-y,dim=-1).mean()
    loss.backward()
    r.finite_step(m,opt)
    assert torch.isfinite(loss)


def test_real_hpeli_generator_batch_one_train_mode_backward():
    r.seed('cpu')
    model = r.build_model('hpeli','cpu').train()
    data = Dataset(1)
    x,y = torch.tensor(data.x),torch.tensor(data.y)
    frozen = trigger()
    graph = frozen.generator
    for parameter in graph.parameters():
        parameter.requires_grad_(True)
    opt = torch.optim.Adam(graph.parameters(),lr=.005)
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    with core._frozen(model,eval_mode=False):
        changed,penalty = a.apply_tensor(graph,x,1.)
        pred = r.prediction(model,changed)
        target = a.target_tensor(y,torch.ones(1))
        loss = (pred[...,[11,13],:]-target[...,[11,13],:]).square().mean()
        (loss+.05*penalty.sum(-1).abs().mean()).backward()
        r.finite_step(graph,opt)
    # Stateful BN buffers may update in native train-mode; weights never do.
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
    assert torch.isfinite(loss) and model.training


def test_inspect_has_no_dataset_gpu_or_output_writes(tmp_path,capsys,monkeypatch):
    monkeypatch.setattr(base := r.base,'audit_inputs',lambda *_:pytest.fail('data access'))
    monkeypatch.setattr(base,'_setup_execution',lambda *_:pytest.fail('GPU access'))
    r.main(['--data-home',str(tmp_path/'home'),'--outdir',str(tmp_path/'output')])
    assert json.loads(capsys.readouterr().out)['profile']==r.PROFILE
    assert not (tmp_path/'output').exists()


def test_preflight_plan_binding_and_nonempty_output_guards(tmp_path,monkeypatch):
    cfg = r.config(tmp_path/'home',tmp_path/'output',device='cpu',
                   clean_references={b:tmp_path/b for b in r.BACKBONES})
    inputs = dict(splits={},fingerprint=r.base.object_sha({}))
    monkeypatch.setattr(r,'sources',lambda:dict(fixture='a'*64))
    monkeypatch.setattr(r.base,'audit_inputs',lambda _: (inputs,{}))
    monkeypatch.setattr(r.clean_ref,'audit',lambda _,b,*__:dict(backbone=b))
    monkeypatch.setattr(r.clean_ref,'files_match',lambda _:None)
    monkeypatch.setattr(r.clean_ref,'load',lambda _:dict(model=Tiny().state_dict()))
    monkeypatch.setattr(r,'build_model',lambda *_:Tiny())
    saved = r.preflight(cfg)
    assert r.load_plan(cfg)==saved
    changed = deepcopy(cfg)
    changed['recipes']['metafiplusplus']['lr'] = .001
    with pytest.raises(ValueError,match='binding'):
        r.load_plan(changed)
    changed = deepcopy(cfg)
    changed['training_queue'].reverse()
    with pytest.raises(ValueError,match='binding'):
        r.load_plan(changed)
    monkeypatch.setattr(r,'sources',lambda:dict(fixture='b'*64))
    with pytest.raises(ValueError,match='binding'):
        r.load_plan(cfg)
    old = tmp_path/'historical'
    old.mkdir()
    (old/'keep.txt').write_text('historical',encoding='utf-8')
    with pytest.raises(ValueError,match='Nonempty'):
        r.preflight(r.config(tmp_path/'home',old,device='cpu'))
    assert (old/'keep.txt').read_text()=='historical'


@pytest.mark.parametrize('mutation',['digest','parent','batch','role','effective','execution','saved','order'])
def test_capacity_gate_is_bound_to_parent_and_each_cell(tmp_path,monkeypatch,mutation):
    cfg = r.config(tmp_path/'home',tmp_path/'output',device='cuda:0')
    plan = dict(cfg=cfg,plan_sha256='f'*64)
    execution = dict(device='cuda:0',fixture=True)
    monkeypatch.setattr(r.base,'_setup_execution',lambda _:execution)
    value = dict(profile=r.PROFILE,plan_sha256=plan['plan_sha256'],execution=deepcopy(execution),
                 surrogate_batch=64,generator_batch=1,checkpoints_saved=False,training_launched=False,
                 victim_smokes=[dict(backbone=b,role=role,physical_batch=32,
                    effective_batch=32*r.RECIPES[b]['accumulation'],loss=.1)
                    for b in r.BACKBONES for role in ('clean','backtime')])
    path = Path(cfg['outdir'])/'capacity.json'
    r.write_json(path,{**value,'receipt_sha256':r.base.object_sha(value)})
    assert r.capacity(plan)==execution
    if mutation=='parent': value['plan_sha256']='a'*64
    if mutation=='batch': value['surrogate_batch']=32
    if mutation=='role': value['victim_smokes'].pop()
    if mutation=='effective':
        next(v for v in value['victim_smokes'] if v['backbone']=='wiflow')['effective_batch']=32
    if mutation=='execution': value['execution']['device']='cuda:1'
    if mutation=='saved': value['checkpoints_saved']=True
    if mutation=='order': value['victim_smokes'].reverse()
    r.write_json(path,{**value,'receipt_sha256':('0'*64 if mutation=='digest' else r.base.object_sha(value))})
    with pytest.raises(ValueError,match='capacity'):
        r.capacity(plan)


def test_owned_lock_and_smoke_cpu_rejection(tmp_path):
    cfg = r.config(tmp_path/'home',tmp_path/'output',device='cpu')
    with r.lock(cfg):
        with pytest.raises(FileExistsError):
            with r.lock(cfg):
                pytest.fail('duplicate lock accepted')
    assert not (Path(cfg['outdir'])/'runner.lock').exists()
    with pytest.raises(ValueError,match='CUDA'):
        r.smoke(dict(cfg=cfg))


def test_launch_dispatch_is_serial_hpeli_wiflow_metafi_with_no_extra_jobs(tmp_path,monkeypatch,capsys):
    cfg = r.config(tmp_path/'home',tmp_path/'output',device='cpu')
    plan = dict(cfg=cfg,plan_sha256='f'*64)
    calls = []
    monkeypatch.setattr(r,'lock',lambda _:nullcontext())
    monkeypatch.setattr(r,'load_plan',lambda *_,**__:plan)
    monkeypatch.setattr(r,'capacity',lambda _:calls.append('capacity'))
    monkeypatch.setattr(r,'fit_attacker',lambda _:(calls.append('attacker'),None))
    monkeypatch.setattr(r,'train',lambda _,b,role:calls.append((b,role)))
    def evaluate(_,b):
        calls.append(('evaluate',b))
        return dict(metrics=dict(mainmetrics={}))
    monkeypatch.setattr(r,'evaluate',evaluate)
    r.main(['--mode','launch','--data-home',str(tmp_path/'home'),
            '--outdir',str(tmp_path/'output'),'--device','cpu'])
    assert calls == ['capacity','attacker',
        ('hpeli','backtime'),('wiflow','backtime'),('metafiplusplus','backtime'),
        ('evaluate','hpeli'),('evaluate','wiflow'),('evaluate','metafiplusplus')]
    assert 'ALL_THREE_COMPLETE' in capsys.readouterr().out


def test_launch_stops_queue_on_failure_without_retry_or_starting_later_model(tmp_path,monkeypatch):
    cfg = r.config(tmp_path/'home',tmp_path/'output',device='cpu')
    monkeypatch.setattr(r,'lock',lambda _:nullcontext())
    monkeypatch.setattr(r,'load_plan',lambda *_,**__:dict(cfg=cfg))
    monkeypatch.setattr(r,'capacity',lambda _:None)
    monkeypatch.setattr(r,'fit_attacker',lambda _:(None,None))
    calls = []
    def train(_,b,role):
        calls.append((b,role))
        if role=='backtime':
            raise RuntimeError('fixture failure, not a real training crash')
    monkeypatch.setattr(r,'train',train)
    monkeypatch.setattr(r,'evaluate',lambda *_:pytest.fail('TEST opened after incomplete queue'))
    with pytest.raises(RuntimeError,match='fixture failure'):
        r.main(['--mode','launch','--data-home',str(tmp_path/'home'),
                '--outdir',str(tmp_path/'output'),'--device','cpu'])
    assert calls == [('hpeli','backtime')]


def test_full_cpu_toy_lifecycle_and_tamper(tmp_path,monkeypatch):
    """Attacker100 + three toy victims200 + read-only Clean; not full PiW."""
    train,test = Dataset(64),Dataset(4)
    cfg = r.config(tmp_path/'home',tmp_path/'output',device='cpu')
    inputs = toy_inputs(train,test)
    roots = {b:historical_clean(tmp_path,b,cfg,inputs,test) for b in r.BACKBONES}
    refs = {b:r.clean_ref.audit(roots[b],b,cfg,inputs) for b in r.BACKBONES}
    cfg['clean_references'] = {b:str(p) for b,p in roots.items()}
    plan = dict(cfg=cfg,plan_sha256='f'*64,inputs=inputs,clean_references=refs)
    monkeypatch.setattr(r,'capacity',lambda _:dict(device='cpu'))
    monkeypatch.setattr(r,'build_model',lambda *_:Tiny())
    monkeypatch.setattr(r,'datasets_for',lambda _:dict(training=train,validation=test))
    r.fit_attacker(plan)
    for b in r.BACKBONES:
        r.train(plan,b,'backtime')
    for b in r.BACKBONES:
        result = r.evaluate(plan,b)
        assert result['status']=='COMPLETE' and len(result['metrics']['dose_rows'])==6
        assert r.verify_evaluation(plan,b)==result
        assert result['clean_reference']['training_started'] is False
        assert not r.cell(plan,b,'clean').exists()
        r.clean_ref.files_match(refs[b])
    path = r.cell(plan,'hpeli','backtime')/'evaluation.json'
    result = r.read_json(path)
    result['metrics']['mainmetrics']['mean_dce_mm'] += 1
    r.write_json(path,result)
    with pytest.raises(ValueError,match='replay'):
        r.verify_evaluation(plan,'hpeli')


def test_clean_training_is_rejected_before_any_gpu_or_model_work(monkeypatch):
    monkeypatch.setattr(r,'capacity',lambda _:pytest.fail('GPU called'))
    monkeypatch.setattr(r,'build_model',lambda *_:pytest.fail('model built'))
    with pytest.raises(ValueError,match='Only BackTime'):
        r.train({},'hpeli','clean')


def test_missing_clean_refs_stop_before_data_audit_or_attacker(tmp_path,monkeypatch):
    monkeypatch.setattr(r.base,'audit_inputs',lambda _:pytest.fail('expensive data audit called'))
    cfg = r.config(tmp_path/'home',tmp_path/'output',device='cpu')
    with pytest.raises(ValueError,match='NOT be retrained'):
        r.preflight(cfg)
    assert not r.plan_path(cfg).exists()


@pytest.mark.parametrize('backbone',r.BACKBONES)
def test_historical_clean_audit_is_read_only_and_defers_test_arrays(tmp_path,monkeypatch,backbone):
    data = Dataset(4)
    cfg = r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    inputs = toy_inputs(data,data)
    root = historical_clean(tmp_path,backbone,cfg,inputs,data)
    before = {str(p):r.base.file_sha(p) for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(np,'load',lambda *_a,**_k:pytest.fail('TEST arrays opened in preflight'))
    record = r.clean_ref.audit(root,backbone,cfg,inputs)
    assert record['training_started'] is False
    assert r.clean_ref.load(record)['epoch'] == 199
    assert before == {str(p):r.base.file_sha(p) for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('field,value',[('drop_last',False),('seed',43),('epochs',50),
    ('amp',True),('pretrained',True),('lr_scheduler',True),('batch_size',64),('gradient_clip',1.)])
def test_mismatched_clean_recipe_never_silently_retrains(tmp_path,field,value):
    data=Dataset(4)
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    inputs=toy_inputs(data,data)
    root=historical_clean(tmp_path,'hpeli',cfg,inputs,data,mutate_cfg=lambda c:c.update({field:value}))
    with pytest.raises(ValueError,match='recipe mismatch'):
        r.clean_ref.audit(root,'hpeli',cfg,inputs)
    assert not Path(cfg['outdir']).exists()


@pytest.mark.parametrize('mutation',['partial','a1','optimizer','backbone','poison'])
def test_wrong_clean_checkpoint_is_rejected(tmp_path,mutation):
    data=Dataset(4)
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    inputs=toy_inputs(data,data)
    def change(blob):
        if mutation=='partial': blob['epoch']=198
        if mutation=='a1': blob['method']='a1'
        if mutation=='optimizer': blob['optimizer']['param_groups'][0]['lr']=.01
        if mutation=='backbone': blob['backbone']='metafiplusplus'
        if mutation=='poison': blob['poison']['n_poison']=1
    root=historical_clean(tmp_path,'wiflow',cfg,inputs,data,mutate_blob=change)
    with pytest.raises(ValueError):
        r.clean_ref.audit(root,'wiflow',cfg,inputs)


def test_changed_clean_input_source_or_file_is_rejected(tmp_path):
    data=Dataset(4)
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    inputs=toy_inputs(data,data)
    root=historical_clean(tmp_path,'hpeli',cfg,inputs,data)
    changed=deepcopy(inputs)
    changed['splits']['train_data']['ordered_names'].reverse()
    with pytest.raises(ValueError,match='fingerprint'):
        r.clean_ref.audit(root,'hpeli',cfg,changed)
    record=r.clean_ref.audit(root,'hpeli',cfg,inputs)
    with Path(record['checkpoint']).open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError,match='reference changed'):
        r.clean_ref.load(record)


def test_cli_duplicate_or_unknown_reference_is_rejected(tmp_path):
    args=['--data-home',str(tmp_path),'--outdir',str(tmp_path/'out')]
    with pytest.raises(SystemExit):
        r.main(args+['--clean-reference','unknown=x'])
    with pytest.raises(SystemExit):
        r.main(args+['--clean-reference','hpeli=x','--clean-reference','hpeli=y'])


def test_discovery_is_bounded_metadata_only_and_does_not_use_test_scores(tmp_path,monkeypatch,capsys):
    data=Dataset(4)
    cfg=r.config(tmp_path/'home',tmp_path/'output',device='cpu')
    inputs=toy_inputs(data,data)
    root=historical_clean(tmp_path,'hpeli',cfg,inputs,data)
    def forbidden(*_a,**_kw): pytest.fail('discovery opened tensors/TEST/GPU')
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(r.base,'_setup_execution',forbidden)
    found=r.clean_ref.discover(cfg)
    assert found['hpeli']==[str(root)] and not found['wiflow']
    r.main(['--mode','discover','--data-home',str(tmp_path/'home'),'--outdir',str(tmp_path/'output')])
    assert json.loads(capsys.readouterr().out)['candidates']==found
    assert not Path(cfg['outdir']).exists()


@pytest.mark.parametrize('ambiguous',[False,True])
def test_preflight_auto_freezes_unique_matching_clean_or_stops_ambiguity(tmp_path,monkeypatch,ambiguous):
    data=Dataset(4)
    cfg=r.config(tmp_path/'home',tmp_path/'output',device='cpu')
    inputs=toy_inputs(data,data)
    for b in r.BACKBONES:
        historical_clean(tmp_path,b,cfg,inputs,data)
    if ambiguous:
        duplicate=tmp_path/'piw_duplicate'
        duplicate.mkdir()
        # A second independent source, not an automatic choice by metric/name.
        other=historical_clean(duplicate,'hpeli',cfg,inputs,data)
        other.rename(tmp_path/'piw_hpeli_other')
    monkeypatch.setattr(r.base,'audit_inputs',lambda _:(inputs,{}))
    monkeypatch.setattr(r,'build_model',lambda *_:Tiny())
    if ambiguous:
        with pytest.raises(ValueError,match='found 2'):
            r.preflight(cfg)
        assert not r.plan_path(cfg).exists()
    else:
        plan=r.preflight(cfg)
        assert set(plan['clean_references'])==set(r.BACKBONES)
        assert all(plan['clean_references'][b]['backbone']==b for b in r.BACKBONES)
        assert all(v is None for v in plan['cfg']['clean_references'].values())


def test_evaluation_defers_reference_arrays_until_all_new_victims_final(tmp_path,monkeypatch):
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    plan=dict(cfg=cfg)
    monkeypatch.setattr(r,'checkpoint',lambda *_a,**_kw:(_ for _ in ()).throw(ValueError('victim incomplete')))
    monkeypatch.setattr(np,'load',lambda *_a,**_kw:pytest.fail('TEST/reference arrays opened early'))
    with pytest.raises(ValueError,match='victim incomplete'):
        r.evaluate(plan,'hpeli')


def test_reference_source_hash_mismatch_is_rejected(tmp_path):
    data=Dataset(4)
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    inputs=toy_inputs(data,data)
    root=historical_clean(tmp_path,'hpeli',cfg,inputs,data)
    path=root/'piw_lr_control.resolved.json'
    m=r.read_json(path)
    m.pop('plan_sha256')
    m['sources']['models/hpeli.py']='0'*64
    r.write_json(path,r.signed(m))
    with pytest.raises(ValueError,match='source differs'):
        r.clean_ref.audit(root,'hpeli',cfg,inputs)
