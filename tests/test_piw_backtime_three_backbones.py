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
    assert cfg['training_queue'] == [dict(backbone=b,role=role)
        for b in ('hpeli','wiflow','metafiplusplus') for role in ('clean','backtime')]


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
    cfg = r.config(tmp_path/'home',tmp_path/'output',device='cpu')
    inputs = dict(splits={},fingerprint=r.base.object_sha({}))
    monkeypatch.setattr(r,'sources',lambda:dict(fixture='a'*64))
    monkeypatch.setattr(r.base,'audit_inputs',lambda _: (inputs,{}))
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
        ('hpeli','clean'),('hpeli','backtime'),
        ('wiflow','clean'),('wiflow','backtime'),
        ('metafiplusplus','clean'),('metafiplusplus','backtime'),
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
    assert calls == [('hpeli','clean'),('hpeli','backtime')]


def test_full_cpu_toy_lifecycle_and_tamper(tmp_path,monkeypatch):
    """Native attacker100 + six toy victims200, NOT real-model full training."""
    train,test = Dataset(64),Dataset(4)
    cfg = r.config(tmp_path/'home',tmp_path/'output',device='cpu')
    plan = dict(cfg=cfg,plan_sha256='f'*64,inputs=dict(fingerprint='a'*64,splits={
        'train_data':dict(retained_count=64,ordered_names=[v['name'] for v in train.items]),
        'test_data':dict(retained_count=4,ordered_names=[v['name'] for v in test.items])}))
    monkeypatch.setattr(r,'capacity',lambda _:dict(device='cpu'))
    monkeypatch.setattr(r,'build_model',lambda *_:Tiny())
    monkeypatch.setattr(r,'datasets_for',lambda _:dict(training=train,validation=test))
    r.fit_attacker(plan)
    for b in r.BACKBONES:
        for role in ('clean','backtime'):
            r.train(plan,b,role)
    for b in r.BACKBONES:
        result = r.evaluate(plan,b)
        assert result['status']=='COMPLETE' and len(result['metrics']['dose_rows'])==6
        assert r.verify_evaluation(plan,b)==result
    path = r.cell(plan,'hpeli','backtime')/'evaluation.json'
    result = r.read_json(path)
    result['metrics']['mainmetrics']['mean_dce_mm'] += 1
    r.write_json(path,result)
    with pytest.raises(ValueError,match='replay'):
        r.verify_evaluation(plan,'hpeli')
