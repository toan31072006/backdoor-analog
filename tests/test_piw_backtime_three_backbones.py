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
    monkeypatch.setattr(r.clean_ref,'audit',lambda _,b,*__:dict(backbone=b,manifest=str(tmp_path/b/'plan.json')))
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
    monkeypatch.setattr(r,'require_trained',lambda _:None)
    monkeypatch.setattr(r,'evaluation_reference',lambda *_:dict(fixture=True))
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


@pytest.mark.parametrize('reuse_attacker',[False,True])
def test_full_cpu_toy_lifecycle_and_tamper(tmp_path,monkeypatch,reuse_attacker):
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
    if reuse_attacker:
        native = torch.load(r.attacker_path(plan), map_location='cpu', weights_only=False)
        frozen = a.FrozenPiWBackTimeTrigger.from_state_dict(native['trigger'])
        source = historical_attacker(tmp_path,cfg,inputs,frozen=frozen)
        cfg['attacker_reference'] = str(source)
        plan['attacker_reference'] = r.attacker_ref.audit(source,cfg,inputs)
        before = {str(p):r.base.file_sha(p) for p in source.rglob('*') if p.is_file()}
        monkeypatch.setattr(a,'fit_stream',lambda *_a,**_kw:pytest.fail('attacker was refitted'))
        reused,_ = r.fit_attacker(plan)
        assert core._sha(reused.state_dict()) == core._sha(frozen.state_dict())
    for b in r.BACKBONES:
        r.train(plan,b,'backtime')
    for b in r.BACKBONES:
        result = r.evaluate(plan,b)
        assert result['status']=='COMPLETE' and len(result['metrics']['dose_rows'])==6
        assert r.verify_evaluation(plan,b)==result
        assert result['clean_reference']['training_started'] is False
        assert not r.cell(plan,b,'clean').exists()
        r.clean_ref.files_match(refs[b])
    if reuse_attacker:
        assert before == {str(p):r.base.file_sha(p) for p in source.rglob('*') if p.is_file()}
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


def test_missing_clean_refs_do_not_block_training_plan_or_open_test(tmp_path,monkeypatch):
    inputs = dict(splits={},fingerprint=r.base.object_sha({}))
    monkeypatch.setattr(r.base,'audit_inputs',lambda _:(inputs,{}))
    monkeypatch.setattr(r,'build_model',lambda *_:pytest.fail('no Clean model needed'))
    monkeypatch.setattr(np,'load',lambda *_a,**_kw:pytest.fail('TEST arrays opened'))
    cfg = r.config(tmp_path/'home',tmp_path/'output',device='cpu')
    plan = r.preflight(cfg)
    assert plan['clean_references'] == dict.fromkeys(r.BACKBONES)
    assert all(v['status']=='EVALUATION_PENDING_CLEAN' for v in plan['clean_reference_status'].values())
    assert r.load_plan(cfg)==plan


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
def test_preflight_auto_freezes_unique_clean_or_defers_ambiguous_evaluation(tmp_path,monkeypatch,ambiguous):
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
        plan=r.preflight(cfg)
        assert plan['clean_references']['hpeli'] is None
        assert 'found 2' in plan['clean_reference_status']['hpeli']['detail']
        assert plan['clean_references']['wiflow'] is not None
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


def historical_attacker(tmp_path,cfg,inputs,*,frozen=None):
    """Synthetic original-plan fixture; only lifecycle test uses a real toy fit."""
    folder=tmp_path/'piw_attacker_v1'
    source=deepcopy(cfg)
    source.pop('attacker_reference',None)
    source.pop('clean_references',None)
    source.update(profile=r.attacker_ref.SOURCE_PROFILE,outdir=str(folder),
        roles=['clean','backtime'],control_policy='three new matched scratch Clean references; no historical checkpoint imports')
    source['training_queue']=[dict(backbone=b,role=role) for b in r.BACKBONES for role in ('clean','backtime')]
    hashes={'ATKBackd/'+n:r.base.file_sha(r.HERE/n) for n in r.attacker_ref.SCIENTIFIC_SOURCES}
    hashes['ATKBackd/run_piw_backtime_three_backbones.py']=sorted(r.attacker_ref.RUNNER_HASHES)[0]
    manifest=r.signed(dict(profile=source['profile'],cfg=source,sources=hashes,inputs=inputs,runtime={'historical':True}))
    r.write_json(folder/'plan.json',manifest)
    execution=dict(device='historical CPU fixture')
    receipt=dict(profile=source['profile'],plan_sha256=manifest['plan_sha256'],execution=execution,
        surrogate_batch=64,generator_batch=1,checkpoints_saved=False,training_launched=False,
        victim_smokes=[dict(backbone=b,role=role,physical_batch=32,
            effective_batch=32*r.RECIPES[b]['accumulation'],loss=.1)
            for b in r.BACKBONES for role in ('clean','backtime')])
    r.write_json(folder/'capacity.json',{**receipt,'receipt_sha256':r.base.object_sha(receipt)})
    n=inputs['splits']['train_data']['retained_count']
    if frozen is None:
        k,steps=int(.1*n),int(np.ceil(n/64))
        diag=dict(fit_profile='full_TRAIN_streaming_author_schedule_PiW14_adaptation',
            n_fit=n,n_poisoned_fit=k,poison_indices=list(range(k)),fit_epochs=100,clean_warmup_epochs=11,
            surrogate_updates=steps*100,generator_updates=k*89,test_used=False,
            dataset_sha256=inputs['fingerprint'],fresh_surrogate_template_unchanged=True,
            history=[dict(epoch=i+1,epoch_index=i,phase='alternating' if i>10 else 'clean_warmup',
                surrogate_updates=steps*(i+1),generator_updates=k*max(0,i-10),
                surrogate_smooth_l1=.1,generator_target_mse=.2 if i>10 else None,
                generator_dc_sum=.001 if i>10 else None) for i in range(100)])
        original=trigger()
        frozen=a.FrozenPiWBackTimeTrigger(original.config,original.generator,diag,selection_scores=torch.ones(n))
    blob=dict(plan_sha256=manifest['plan_sha256'],execution=execution,
        doses=np.random.default_rng(42).uniform(.2,1.,n).astype(np.float32),trigger=frozen.state_dict())
    (folder/'attacker').mkdir(exist_ok=True)
    torch.save(blob,folder/'attacker/trigger.pt')
    r.write_json(folder/'attacker/provenance.json',frozen.provenance())
    return folder


def test_completed_attacker_import_is_read_only_no_fit_gpu_or_test(tmp_path,monkeypatch):
    data=Dataset(64)
    cfg=r.config(tmp_path/'home',tmp_path/'new',device='cpu')
    inputs=toy_inputs(data,data)
    folder=historical_attacker(tmp_path,cfg,inputs)
    before={str(p):r.base.file_sha(p) for p in folder.rglob('*') if p.is_file()}
    forbidden=lambda *_a,**_kw:pytest.fail('import touched fitting/GPU/TEST/model')
    monkeypatch.setattr(a,'fit_stream',forbidden)
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(r.base,'_setup_execution',forbidden)
    monkeypatch.setattr(r,'datasets_for',forbidden)
    monkeypatch.setattr(r,'build_model',forbidden)
    cfg['attacker_reference']=str(folder)
    record=r.attacker_ref.audit(folder,cfg,inputs)
    plan=dict(cfg=cfg,inputs=inputs,attacker_reference=record)
    op,doses=r.fit_attacker(plan)
    source=torch.load(folder/'attacker/trigger.pt',map_location='cpu',weights_only=False)
    assert core._sha(op.state_dict()) == core._sha(source['trigger'])
    np.testing.assert_array_equal(doses,source['doses'])
    assert record['training_started'] is False
    assert r.attacker_path(plan)==folder/'attacker/trigger.pt'
    assert not Path(cfg['outdir']).exists()
    assert before == {str(p):r.base.file_sha(p) for p in folder.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mutation',['profile','plan_hash','target','rho','seed','inputs','outdir',
    'runner','source','capacity_hash','capacity_parent','execution','doses',
    'frozen_hash','partial','history','counter','recipients','nonfinite','provenance'])
def test_foreign_or_partial_attacker_import_rejected(tmp_path,mutation):
    data=Dataset(64)
    cfg=r.config(tmp_path/'home',tmp_path/'new',device='cpu')
    inputs=toy_inputs(data,data)
    folder=historical_attacker(tmp_path,cfg,inputs)
    manifest=r.read_json(folder/'plan.json')
    receipt=r.read_json(folder/'capacity.json')
    blob=torch.load(folder/'attacker/trigger.pt',map_location='cpu',weights_only=False)
    if mutation in ('profile','plan_hash','target','rho','seed','inputs','outdir','runner','source'):
        manifest.pop('plan_sha256')
        if mutation=='profile': manifest['profile']='MM-Fi'
        if mutation=='target': manifest['cfg']['target']['pivot']=1
        if mutation=='rho': manifest['cfg']['rho']=.2
        if mutation=='seed': manifest['cfg']['seed']=43
        if mutation=='inputs': manifest['inputs']['splits']['train_data']['ordered_names'].reverse()
        if mutation=='outdir': manifest['cfg']['outdir']=str(tmp_path/'different source')
        if mutation=='runner': manifest['sources']['ATKBackd/run_piw_backtime_three_backbones.py']='a'*64
        if mutation=='source': manifest['sources']['ATKBackd/attack/backtime_mmfi.py']='a'*64
        manifest=r.signed(manifest)
        if mutation=='plan_hash': manifest['plan_sha256']='a'*64
        r.write_json(folder/'plan.json',manifest)
    elif mutation in ('capacity_hash','capacity_parent'):
        if mutation=='capacity_parent': receipt['plan_sha256']='a'*64
        receipt.pop('receipt_sha256')
        receipt['receipt_sha256']=r.base.object_sha(receipt) if mutation=='capacity_parent' else 'a'*64
        r.write_json(folder/'capacity.json',receipt)
    elif mutation=='provenance':
        r.write_json(folder/'attacker/provenance.json',{'wrong':True})
    else:
        if mutation=='execution': blob['execution']['device']='changed'
        if mutation=='doses': blob['doses'][0]=.001
        frozen=blob['trigger']
        if mutation=='partial': frozen['diagnostics']['fit_epochs']=99
        if mutation=='history': frozen['diagnostics']['history'].pop()
        if mutation=='counter': frozen['diagnostics']['history'][-1]['generator_updates']-=1
        if mutation=='recipients': frozen['diagnostics']['poison_indices'][0]=len(data)
        if mutation=='nonfinite': frozen['generator']['conv1'][0,0]=float('nan')
        frozen.pop('sha256')
        frozen['sha256']=core._sha(frozen) if mutation!='frozen_hash' else 'a'*64
        torch.save(blob,folder/'attacker/trigger.pt')
    with pytest.raises(ValueError):
        r.attacker_ref.audit(folder,cfg,inputs)
    assert not Path(cfg['outdir']).exists()


def test_attacker_reference_rejects_active_source_and_keeps_lock(tmp_path):
    import os
    root=tmp_path/'source'
    r.write_json(root/'runner.lock',dict(pid=os.getpid(),token='owned'))
    with pytest.raises(ValueError,match='still active'):
        r.attacker_ref.require_idle(root)
    assert r.read_json(root/'runner.lock')['token']=='owned'


def test_stopped_source_stale_lock_is_kept_read_only(tmp_path):
    root=tmp_path/'source'
    r.write_json(root/'runner.lock',dict(pid=2147483647,token='stale-preserved'))
    r.attacker_ref.require_idle(root)
    assert r.read_json(root/'runner.lock')['token']=='stale-preserved'


def test_attacker_reference_missing_audit_never_falls_back_to_fit(tmp_path,monkeypatch):
    cfg=r.config(tmp_path/'home',tmp_path/'new',device='cpu',attacker_reference=tmp_path/'old')
    monkeypatch.setattr(a,'fit_stream',lambda *_a,**_kw:pytest.fail('refit'))
    monkeypatch.setattr(r,'capacity',lambda *_:pytest.fail('GPU called'))
    with pytest.raises(ValueError,match='no refit fallback'):
        r.fit_attacker(dict(cfg=cfg))


def test_attacker_reference_tamper_and_overlapping_output_rejected(tmp_path):
    data=Dataset(64)
    cfg=r.config(tmp_path/'home',tmp_path/'new',device='cpu')
    inputs=toy_inputs(data,data)
    folder=historical_attacker(tmp_path,cfg,inputs)
    record=r.attacker_ref.audit(folder,cfg,inputs)
    other=deepcopy(cfg)
    other['outdir']=str(folder/'nested')
    with pytest.raises(ValueError,match='separate'):
        r.attacker_ref.audit(folder,other,inputs)
    with (folder/'attacker/trigger.pt').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError,match='Historical attacker changed'):
        r.attacker_ref.load(record,cfg,inputs)


def test_import_preflight_freezes_origin_and_rechecks_on_resume(tmp_path,monkeypatch):
    data=Dataset(64)
    cfg=r.config(tmp_path/'home',tmp_path/'new',device='cpu')
    inputs=toy_inputs(data,data)
    for split,entry in inputs['splits'].items():
        path=Path(cfg['dataset_root'])/split/(split+'_list.txt')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('\n'.join(entry['ordered_names']),encoding='utf-8')
        entry.update(official_list_sha256=r.base.file_sha(path),files=[])
    inputs['fingerprint']=r.base.object_sha(inputs['splits'])
    folder=historical_attacker(tmp_path,cfg,inputs)
    cfg=r.config(tmp_path/'home',tmp_path/'new',device='cpu',attacker_reference=folder)
    for b in r.BACKBONES:
        historical_clean(tmp_path,b,cfg,inputs,data)
    monkeypatch.setattr(r.base,'audit_inputs',lambda _:(inputs,{}))
    monkeypatch.setattr(r,'build_model',lambda *_:Tiny())
    plan=r.preflight(cfg)
    assert r.load_plan(cfg)==plan
    assert plan['attacker_reference']['source_run']==str(folder)
    assert not (Path(cfg['outdir'])/'attacker').exists()
    with (folder/'attacker/provenance.json').open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError,match='Historical attacker changed'):
        r.load_plan(cfg)


def test_import_inspect_accepts_spaces_without_data_or_writes(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(r.attacker_ref,'audit',lambda *_:pytest.fail('inspect opened attacker'))
    r.main(['--data-home',str(tmp_path/'home'),'--outdir',str(tmp_path/'out'),
            '--attacker-reference',str(tmp_path/'old run with spaces')])
    cfg=json.loads(capsys.readouterr().out)
    assert cfg['attacker_reference']==str(tmp_path/'old run with spaces')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('invalid_kind',['recipe','weights'])
def test_incompatible_automatic_clean_is_pending_but_explicit_is_error(tmp_path,monkeypatch,invalid_kind):
    data=Dataset(4)
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    inputs=toy_inputs(data,data)
    root=historical_clean(tmp_path,'hpeli',cfg,inputs,data,
        mutate_cfg=(lambda c:c.update(drop_last=False)) if invalid_kind=='recipe' else None)
    monkeypatch.setattr(r.base,'audit_inputs',lambda _:(inputs,{}))
    monkeypatch.setattr(r,'build_model',lambda *_:nn.Linear(2,2) if invalid_kind=='weights' else Tiny())
    plan=r.preflight(cfg)
    assert plan['clean_references']['hpeli'] is None
    assert plan['clean_reference_status']['hpeli']['status']=='EVALUATION_PENDING_CLEAN'
    explicit=r.config(tmp_path/'home',tmp_path/'explicit',device='cpu',clean_references={'hpeli':root})
    with pytest.raises(ValueError,match='expected ONE'):
        r.preflight(explicit)
    assert not r.plan_path(explicit).exists()


def test_clean_files_are_not_a_resume_training_dependency(tmp_path,monkeypatch):
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    inputs=dict(splits={},fingerprint=r.base.object_sha({}))
    monkeypatch.setattr(r.base,'audit_inputs',lambda _:(inputs,{}))
    plan=r.preflight(cfg)
    plan.pop('plan_sha256')
    plan['clean_references']['hpeli']=dict(backbone='hpeli',files={'missing.pt':{}})
    r.write_json(r.plan_path(cfg),r.signed(plan))
    monkeypatch.setattr(r.clean_ref,'files_match',lambda *_:pytest.fail('Clean blocks training resume'))
    assert r.load_plan(cfg)['clean_references']['hpeli']['backbone']=='hpeli'


@pytest.mark.parametrize('mode',['start','launch'])
def test_pending_clean_launch_still_trains_all_three_in_order(tmp_path,monkeypatch,capsys,mode):
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    plan=dict(cfg=cfg,plan_sha256='f'*64,clean_references=dict.fromkeys(r.BACKBONES))
    calls=[]
    monkeypatch.setattr(r,'lock',lambda _:nullcontext())
    monkeypatch.setattr(r,'preflight',lambda *_:plan)
    monkeypatch.setattr(r,'load_plan',lambda *_,**__:plan)
    monkeypatch.setattr(r,'smoke',lambda _:calls.append('smoke'))
    monkeypatch.setattr(r,'capacity',lambda _:calls.append('capacity'))
    monkeypatch.setattr(r,'fit_attacker',lambda _:(calls.append('attacker'),None))
    monkeypatch.setattr(r,'train',lambda _,b,role:calls.append((b,role)))
    monkeypatch.setattr(r,'require_trained',lambda _:None)
    monkeypatch.setattr(r,'evaluate',lambda *_:pytest.fail('missing Clean evaluation attempted'))
    r.main(['--mode',mode,'--data-home',str(tmp_path/'home'),'--outdir',str(tmp_path/'out'),'--device','cpu'])
    assert calls==(['smoke'] if mode=='start' else [])+['capacity','attacker',
        ('hpeli','backtime'),('wiflow','backtime'),('metafiplusplus','backtime')]
    output=capsys.readouterr().out
    assert 'ALL_THREE_TRAINED_EVALUATION_PARTIAL' in output
    assert 'ALL_THREE_COMPLETE' not in output


def test_start_does_not_repeat_existing_capacity_or_retry_failure(tmp_path,monkeypatch):
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    r.write_json(Path(cfg['outdir'])/'capacity.json',{})
    monkeypatch.setattr(r,'lock',lambda _:nullcontext())
    monkeypatch.setattr(r,'preflight',lambda *_:dict(cfg=cfg))
    monkeypatch.setattr(r,'smoke',lambda _:pytest.fail('existing capacity repeated'))
    monkeypatch.setattr(r,'fit_attacker',lambda _:pytest.fail('invalid capacity bypassed'))
    calls=[]
    def capacity(_):
        calls.append('capacity')
        raise ValueError('fixture bad capacity')
    monkeypatch.setattr(r,'capacity',capacity)
    with pytest.raises(ValueError,match='fixture bad capacity'):
        r.main(['--mode','start','--data-home',str(tmp_path/'home'),
                '--outdir',str(tmp_path/'out'),'--device','cpu'])
    assert calls==['capacity']


def test_hpeli_only_clean_imported_attacker_three_victims_and_late_clean(tmp_path,monkeypatch,capsys):
    """Synthetic G/source fixture, real toy victim loops200; no full-data claim."""
    train,test=Dataset(64),Dataset(4)
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    inputs=toy_inputs(train,test)
    source=historical_attacker(tmp_path,cfg,inputs)
    cfg['attacker_reference']=str(source)
    hpeli=historical_clean(tmp_path,'hpeli',cfg,inputs,test)
    reference=r.clean_ref.audit(hpeli,'hpeli',cfg,inputs)
    refs=dict.fromkeys(r.BACKBONES)
    refs['hpeli']=reference
    plan=dict(profile=r.PROFILE,cfg=cfg,plan_sha256='f'*64,inputs=inputs,
        clean_references=refs,attacker_reference=r.attacker_ref.audit(source,cfg,inputs))
    r.write_json(r.plan_path(cfg),plan)
    old_files=[r.plan_path(cfg), *[p for root in (source,hpeli) for p in root.rglob('*') if p.is_file()]]
    old_hashes={str(p):r.base.file_sha(p) for p in old_files}
    monkeypatch.setattr(r,'capacity',lambda _:dict(device='cpu'))
    monkeypatch.setattr(r,'build_model',lambda *_:Tiny())
    monkeypatch.setattr(r,'datasets_for',lambda _:dict(training=train,validation=test))
    monkeypatch.setattr(a,'fit_stream',lambda *_a,**_kw:pytest.fail('completed attacker refit'))
    r.fit_attacker(plan)
    with pytest.raises(ValueError,match='Clean'):
        r.checkpoint(plan,'wiflow','clean')
    for b in r.BACKBONES:
        r.train(plan,b,'backtime')
        assert r.checkpoint(plan,b,'backtime',final=True)['epoch']==199
    victim_paths=[r.cell(plan,b,'backtime')/'checkpoint.pt' for b in r.BACKBONES]
    victim_hashes={str(p):r.base.file_sha(p) for p in victim_paths}
    with monkeypatch.context() as m:
        m.setattr(r,'build_model',lambda *_:pytest.fail('final victim rebuilt'))
        m.setattr(r,'load_attacker',lambda *_:pytest.fail('final victim loads G'))
        m.setattr(r,'datasets_for',lambda *_:pytest.fail('final victim reloads data'))
        for b in r.BACKBONES:
            r.train(plan,b,'backtime')
    summary=r.evaluate_available(plan)
    assert summary==dict(training_complete=True,evaluated=['hpeli'],
                         evaluation_pending_clean=['wiflow','metafiplusplus'])
    assert not (r.cell(plan,'wiflow','backtime')/'evaluation.json').exists()
    hpeli_result=r.cell(plan,'hpeli','backtime')/'evaluation.json'
    hpeli_sha=r.base.file_sha(hpeli_result)
    late={b:historical_clean(tmp_path,b,cfg,inputs,test) for b in r.BACKBONES[1:]}
    summary=r.evaluate_available(plan,late)
    assert summary['evaluated']==list(r.BACKBONES) and not summary['evaluation_pending_clean']
    assert r.base.file_sha(hpeli_result)==hpeli_sha
    assert old_hashes=={str(p):r.base.file_sha(p) for p in old_files}
    assert victim_hashes=={str(p):r.base.file_sha(p) for p in victim_paths}
    assert not (Path(cfg['outdir'])/'attacker').exists()
    assert all(not r.cell(plan,b,'clean').exists() for b in r.BACKBONES)
    before=r.base.file_sha(r.reference_path(plan,'wiflow'))
    r.register_evaluation_reference(plan,'wiflow',late['wiflow'])
    assert r.base.file_sha(r.reference_path(plan,'wiflow'))==before
    with pytest.raises(ValueError,match='already frozen'):
        r.register_evaluation_reference(plan,'wiflow',hpeli)
    path=r.reference_path(plan,'wiflow')
    record=r.read_json(path)
    record['training_plan_sha256']='0'*64
    r.write_json(path,r.signed({k:v for k,v in record.items() if k!='plan_sha256'}))
    with pytest.raises(ValueError,match='Late Clean reference binding'):
        r.verify_evaluation(plan,'wiflow')


def test_late_reference_cannot_open_test_before_all_three_final(tmp_path,monkeypatch):
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu')
    plan=dict(cfg=cfg)
    monkeypatch.setattr(r,'checkpoint',lambda *_a,**_kw:(_ for _ in ()).throw(ValueError('not final199')))
    monkeypatch.setattr(r.clean_ref,'audit',lambda *_:pytest.fail('late Clean audit before final victims'))
    monkeypatch.setattr(np,'load',lambda *_a,**_kw:pytest.fail('TEST opened'))
    with pytest.raises(ValueError,match='not final199'):
        r.register_evaluation_reference(plan,'wiflow',tmp_path/'source')


def test_evaluate_cli_late_reference_does_not_rebind_training_cfg(tmp_path,monkeypatch):
    cfg=r.config(tmp_path/'home',tmp_path/'out',device='cpu',
                 clean_references={'hpeli':tmp_path/'old_hpeli'})
    r.write_json(r.plan_path(cfg),dict(cfg=cfg))
    loaded=[]
    monkeypatch.setattr(r,'lock',lambda _:nullcontext())
    def load(actual,**_):
        loaded.append(actual)
        return dict(cfg=actual)
    monkeypatch.setattr(r,'load_plan',load)
    called=[]
    monkeypatch.setattr(r,'evaluate_available',lambda p,refs,**kw:called.append(refs))
    monkeypatch.setattr(r,'train',lambda *_:pytest.fail('evaluate retrains victim'))
    monkeypatch.setattr(r,'fit_attacker',lambda *_:pytest.fail('evaluate refits attacker'))
    r.main(['--mode','evaluate','--data-home',str(tmp_path/'home'),'--outdir',str(tmp_path/'out'),
            '--device','cpu','--clean-reference','wiflow='+str(tmp_path/'late_wiflow')])
    assert loaded==[cfg]
    assert called==[{'wiflow':str(tmp_path/'late_wiflow')}]


def test_helper_defaults_reuse_old_attacker_and_new_v4_output():
    source=(ROOT/'remote_tools/Start-PiWBackTimeThreeBackbones.ps1').read_text(encoding='utf-8')
    assert "[string]$AttackerRun = 'U:\\backdooranalog\\runs\\piw_backtime_three_backbones_s42_v1'" in source
    assert 'piw_backtime_three_backbones_train_first_s42_v4' in source
    assert "Start='start'" in source and "Evaluate='evaluate'" in source
    assert "$Mode = 'Inspect'" in source
    assert 'it will NOT fit a new attacker' in source


def test_partial_victim_resume_preserves_optimizer_rng_and_final_weights(tmp_path,monkeypatch):
    data=Dataset(64)
    cfg=r.config(tmp_path/'home',tmp_path/'full',device='cpu')
    op=trigger()
    op._diagnostics['poison_indices']=[1,7,19,24,38,50]
    doses=np.random.default_rng(42).uniform(.2,1.,64).astype(np.float32)
    frozen_path=tmp_path/'frozen_trigger.pt'
    torch.save(op.state_dict(),frozen_path)
    monkeypatch.setattr(r,'attacker_path',lambda _:frozen_path)
    monkeypatch.setattr(r,'load_attacker',lambda _:(op,doses))
    monkeypatch.setattr(r,'capacity',lambda _:dict(device='cpu'))
    monkeypatch.setattr(r,'build_model',lambda *_:Tiny())
    monkeypatch.setattr(r,'datasets_for',lambda _:dict(training=data))
    full=dict(cfg=cfg,plan_sha256='f'*64)
    r.train(full,'hpeli','backtime')
    expected=r.checkpoint(full,'hpeli','backtime',final=True)
    partial=dict(cfg=r.config(tmp_path/'home',tmp_path/'resumed',device='cpu'),plan_sha256='f'*64)
    save=r.base.atomic_checkpoint
    def interrupted(path,value):
        save(path,value)
        if value['epoch']==73:
            raise RuntimeError('fixture manual stop after atomic epoch73')
    with monkeypatch.context() as m:
        m.setattr(r.base,'atomic_checkpoint',interrupted)
        with pytest.raises(RuntimeError,match='manual stop'):
            r.train(partial,'hpeli','backtime')
    assert r.checkpoint(partial,'hpeli','backtime')['epoch']==73
    r.train(partial,'hpeli','backtime')
    actual=r.checkpoint(partial,'hpeli','backtime',final=True)
    for key in ('model','optimizer','rng','train_loss'):
        assert core._sha(actual[key])==core._sha(expected[key]),key
