"""CPU/synthetic contracts, not full MM-Fi/CUDA/Van verification."""
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

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('backtime_transfer',
    ROOT/'remote_tools/Run-MMFiBackTimeTransfer4090.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


@pytest.fixture
def old(tmp_path):
    adapter = a.load_adapter()
    runtime = adapter.load_runtime('p1s1')
    return adapter.build_plan(runtime, tmp_path/'home', tmp_path/'runs/mmfi_backtime_source', 'p1s1')


@pytest.fixture
def cfgs(old):
    return {b:a.configuration(old['cfg']['backtime'], b) for b in a.BACKBONES}


def test_recipes_and_poison_protocol_are_separate_and_unchanged(cfgs, old):
    meta, wifi = (cfgs[b] for b in a.BACKBONES)
    assert (meta['optimizer'], meta['lr'], meta['batch_size']) == ('sgd', .01, 32)
    assert (wifi['optimizer'], wifi['lr'], wifi['weight_decay']) == ('adamw', 1e-4, 1e-4)
    assert (wifi['backbone_accumulation_steps'], wifi['backbone_effective_batch_size']) == (2, 64)
    assert wifi['backbone_gradient_clip'] == 1.
    for cfg in cfgs.values():
        for k in ('poison_select', 'rho', 'dose_coupling', 'dose_min', 'dose_max',
                  'pivot', 'target_joints', 'payload_axis', 'theta_max_deg',
                  'mmfi_protocol', 'mmfi_setting', 'mmfi_split_seed', 'dose_grid'):
            assert cfg[k] == old['cfg']['backtime'][k]
        assert cfg['poison_select'] == 'backtime_train_error'
        assert cfg['epochs'] == 50 and cfg['num_workers'] == 0
        assert cfg['pretrained'] is False


def test_fresh_components_use_actual_architecture_optimizer(cfgs, monkeypatch):
    built = []
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layer = torch.nn.Linear(2, 2)
    monkeypatch.setattr(a, 'build_model', lambda b: (built.append(b) or Toy()))
    for b in a.BACKBONES:
        engine, model, optimizer = a.components(cfgs[b], 'cpu')
        assert isinstance(optimizer, torch.optim.SGD if b == 'metafiplusplus' else torch.optim.AdamW)
        assert not optimizer.state  # no clean/surrogate warm start
    assert built == list(a.BACKBONES)


@pytest.mark.parametrize('backbone,expected_updates', [('metafiplusplus', 3), ('wiflow', 2)])
def test_native_accumulation_tail_flush(cfgs, backbone, expected_updates):
    from attack.poison import collate
    cfg = cfgs[backbone]
    engine = a.engine_for(cfg)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1, 1, 17, 3))
        def forward(self, x):
            return self.weight.expand(len(x), -1, -1, -1), None
    model = Toy()
    _, optimizer = engine._build_optimizer(model, cfg, 'mmfi')
    steps = []
    original = optimizer.step
    optimizer.step = lambda: (steps.append(1), original())[1]
    rows = [dict(csi=torch.ones(3,114,10), pose=torch.zeros(1,17,3)) for _ in range(65)]
    for batch in engine._training_loader(rows, cfg):
        engine._victim_update(model, optimizer, batch['csi'], batch['pose'])
    assert len(steps) == expected_updates
    # A second epoch cannot carry an unfinished accumulation group.
    for batch in engine._training_loader(rows, cfg):
        engine._victim_update(model, optimizer, batch['csi'], batch['pose'])
    assert len(steps) == 2*expected_updates


def minimal_plan(tmp_path, cfgs):
    return dict(plan_sha256='plan', outdir=str(tmp_path), cfg=cfgs,
        source=dict(root='source', files=dict(trigger='frozen')),
        clean={b:dict(files=dict(checkpoint='clean')) for b in a.BACKBONES})


def checkpoint(plan, b):
    cfg = plan['cfg'][b]
    group = dict(lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    group.update(dict(momentum=.9) if b == 'metafiplusplus' else dict(betas=(.9,.999), eps=1e-8))
    return dict(epoch=0, cfg=cfg, binding=a.binding(plan,b,{}),
        model=dict(weight=torch.ones(1)), optimizer=dict(param_groups=[group],state={0:dict(v=torch.ones(1))}),
        rng=dict(python=[1], numpy=[1], torch=torch.get_rng_state(),cuda=[torch.get_rng_state()]),
        pending_accumulation=0)


@pytest.mark.parametrize('b', a.BACKBONES)
def test_checkpoint_exact_binding_recipe_rng_and_finite(tmp_path, cfgs, b):
    plan = minimal_plan(tmp_path, cfgs)
    blob = checkpoint(plan,b)
    a.validate_checkpoint(blob,plan,b,{})
    mutations = [lambda x:x.update(epoch=50), lambda x:x.update(pending_accumulation=1),
        lambda x:x['rng'].pop('cuda'), lambda x:x['binding'].update(plan_sha256='other'),
        lambda x:x['optimizer']['param_groups'][0].update(lr=999),
        lambda x:x['model'].update(weight=torch.tensor([float('nan')]))]
    for mutate in mutations:
        bad = copy.deepcopy(blob); mutate(bad)
        with pytest.raises(ValueError): a.validate_checkpoint(bad,plan,b,{})


def synthetic_arrays():
    from attack.payload import make_target_pose, set_skeleton_config
    set_skeleton_config('mmfi')
    from extra_attack_eval import measure_pair
    rng = np.random.default_rng(42)
    true = rng.normal(size=(4,17,3)).astype(np.float32)
    targets = np.stack([make_target_pose(true,1,d,np.deg2rad(40),target_joints=[2,3]).astype(np.float32)
                        for d in a.GRID])
    arrays = dict(true_pose=true, target_poses=targets, pair_ids=np.array(['a','b','c','d']), doses=np.array(a.GRID))
    records = {}
    for label in ('clean_reference','backtime'):
        clean = true + .01
        grid = np.stack([clean if d == 0 else clean+(targets[i]-true)*.8 for i,d in enumerate(a.GRID)])
        rows, residuals = [], []
        for i,d in enumerate(a.GRID):
            measured, per = measure_pair(clean, grid[i], true, targets[i])
            rows.append(dict(dose=d,**measured)); residuals.append(per)
        records[label] = dict(metrics=a.summarize(clean,true,rows),dose_rows=rows)
        arrays[label+'__clean_prediction']=clean
        arrays[label+'__trigger_predictions']=grid
        for name in ('target_error_mm','no_trigger_target_error_mm','dce_mm'):
            arrays[label+'__'+name]=np.stack([r[name] for r in residuals])
    return dict(records=records), arrays


def test_replay_all_six_doses_xyz_residuals_and_pck(monkeypatch):
    monkeypatch.setitem(a.COUNTS,'test',4)
    row, arrays = synthetic_arrays()
    a.replay(row,arrays,['a','b','c','d'])
    arrays['backtime__dce_mm'][1,0] += 1
    with pytest.raises(ValueError,match='residual'): a.replay(row,arrays)


@pytest.mark.parametrize('kind', ['ids','dose','target','prediction','metric'])
def test_replay_rejects_corruption(monkeypatch,kind):
    monkeypatch.setitem(a.COUNTS,'test',4)
    row,arrays=synthetic_arrays()
    if kind == 'ids': arrays['pair_ids'][1]='a'
    elif kind == 'dose': arrays['doses'][1]=.3
    elif kind == 'target': arrays['target_poses'][2,0,2,0] += 1
    elif kind == 'prediction': arrays['backtime__trigger_predictions'][0] += 1
    else: row['records']['backtime']['metrics']['mean_positive_dce_mm'] += 1
    with pytest.raises(ValueError): a.replay(row,arrays)


def test_output_source_overlaps_and_non_run_paths_rejected(tmp_path):
    good=tmp_path/'runs/mmfi_backtime_transfer_v1'
    source=tmp_path/'runs/source/p1s1'
    a.validate_paths(good,source,[tmp_path/'clean1',tmp_path/'clean2'])
    for root in (source,source/'runs/mmfi_backtime_transfer_inside', tmp_path/'other'):
        with pytest.raises(ValueError): a.validate_paths(root,source,[tmp_path/'clean1'])


def test_source_p1s1_plan_is_not_relabelled_or_written(old, tmp_path, monkeypatch):
    source=Path(old['outdir']); source.mkdir(parents=True)
    a.write(source/'plan.json',old)
    before=(source/'plan.json').read_bytes()
    runtime,plan=a.source_runtime(source)
    assert plan == old and runtime.COUNTS == a.COUNTS
    assert (source/'plan.json').read_bytes() == before
    changed=copy.deepcopy(old); changed['cfg']['backtime']['mmfi_setting']='s3'
    changed.pop('plan_sha256'); changed['plan_sha256']=a.digest(changed)
    a.write(source/'plan.json',changed,immutable=False)
    with pytest.raises(ValueError,match='P1S1'): a.source_runtime(source)


@pytest.fixture
def historical_inputs(old):
    """Small input receipts through the REAL old verify_inputs, not full MM-Fi."""
    runtime = a.load_adapter().load_runtime('p1s1')
    plan = copy.deepcopy(old)
    plan['sources'] = {n:h for n,h in old['sources'].items() if n not in a.AUDITED_SOURCE_ADDITIONS}
    plan.pop('plan_sha256'); plan['plan_sha256'] = a.digest(plan)
    root = Path(plan['outdir'])
    prior = Path(plan['cfg']['backtime']['action_npy'])
    a.write(prior, {'fixture':'prior'})
    runtime.ACTION_SHA = a.sha(prior)
    data = Path(plan['cfg']['backtime']['dataset_root'])
    a.write(data/'fixture.json', {'fixture':'input'})
    files = [dict(path='fixture.json', bytes=(data/'fixture.json').stat().st_size,
                  sha256=a.sha(data/'fixture.json'))]
    poison = dict(fixture='original poison')
    banks = {key:dict(indices=ids, split='training')
             for key,ids in runtime.bank_selection(a.COUNTS['training']).items()}
    inputs = dict(plan_sha256=plan['plan_sha256'], action_sha256=runtime.ACTION_SHA,
        dataset_root=str(data), files=files, full_content_sha256=a.digest(files),
        poison_sha256=a.digest(poison), banks=banks, banks_sha256=a.digest(banks),
        no_test_selection=True, input_stat_fingerprint=runtime.common.input_stats(data,files),
        splits=dict(training='fixture training IDs',test='fixture test IDs'))
    a.write(root/'plan.json',plan)
    a.write(root/'inputs.json',inputs)
    a.write(root/'poison.json',poison)
    return runtime,plan,inputs


def test_historical_native_verifier_allows_only_pinned_additions_without_global_patch(historical_inputs):
    runtime,plan,inputs = historical_inputs
    shared = runtime.common
    original_sources = shared.sources
    root = Path(plan['outdir'])
    before = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='Scientific source changed'):
        runtime.verify_inputs(plan,full_hash=True)
    receipt = a.bind_historical_sources(runtime,plan)
    assert runtime.verify_inputs(plan,full_hash=True) == inputs
    assert runtime.common is not shared and a.native.common is shared
    assert shared.sources is original_sources and shared.sources() != plan['sources']
    assert runtime.common.sources() == plan['sources']
    assert a.setup_sources()['scientific'] == shared.sources()
    assert receipt['additions'] == a.AUDITED_SOURCE_ADDITIONS
    assert receipt['recorded_sha256'] == a.digest(plan['sources'])
    assert receipt['current_sha256'] == a.digest(shared.sources())
    assert {p:p.read_bytes() for p in root.rglob('*') if p.is_file()} == before
    # A distinct native runtime retains the original strict equality guard.
    with pytest.raises(ValueError,match='Scientific source changed'):
        a.load_adapter().load_runtime('p1s1').verify_inputs(plan)


@pytest.mark.parametrize('kind', ['changed','missing','unapproved','addition_hash','unsafe'])
def test_historical_source_compatibility_rejects_drift(historical_inputs,monkeypatch,kind):
    runtime,plan,_ = historical_inputs
    shared = runtime.common
    current = shared.sources()
    if kind == 'changed': current['attack/backtime_mmfi.py'] = '0'*64
    elif kind == 'missing': current.pop('attack/backtime_mmfi.py')
    elif kind == 'unapproved': current['tests/test_action_targets_parallel.py'] = a.sha(ROOT/'ATKBackd/tests/test_action_targets_parallel.py')
    elif kind == 'addition_hash': current['attack/action_targets.py'] = '0'*64
    else: current['../unapproved.py'] = '0'*64
    monkeypatch.setattr(shared,'sources',lambda:current)
    with pytest.raises(ValueError): a.bind_historical_sources(runtime,plan)
    assert runtime.common is shared  # failed audit never installs even a private view


def test_historical_projection_rechecks_source_after_binding(historical_inputs,monkeypatch):
    runtime,plan,_ = historical_inputs
    shared = runtime.common
    current = shared.sources()
    a.bind_historical_sources(runtime,plan)
    monkeypatch.setattr(shared,'sources',lambda:current)
    current['attack/backtime_mmfi.py'] = '0'*64
    with pytest.raises(ValueError,match='missing/changed'):
        runtime.verify_inputs(plan,full_hash=True)


@pytest.mark.parametrize('kind', ['plan','poison','bank','prior','stat','content'])
def test_historical_source_adapter_keeps_native_input_guards(historical_inputs,kind):
    runtime,plan,inputs = historical_inputs
    a.bind_historical_sources(runtime,plan)
    root = Path(plan['outdir'])
    if kind == 'plan':
        changed=copy.deepcopy(plan); changed['home']='different'
        a.write(root/'plan.json',changed,immutable=False)
    elif kind == 'poison': a.write(root/'poison.json',{'changed':True},immutable=False)
    elif kind == 'prior': a.write(Path(plan['cfg']['backtime']['action_npy']),{'changed':True},immutable=False)
    elif kind == 'bank':
        inputs['banks']['fit']['split']='test'
        inputs['banks_sha256']=a.digest(inputs['banks'])
        a.write(root/'inputs.json',inputs,immutable=False)
    elif kind == 'stat':
        inputs['input_stat_fingerprint']='changed'
        a.write(root/'inputs.json',inputs,immutable=False)
    else:
        # Correct stat receipt but wrong bytes must still fail full-content hashing.
        inputs['files'][0]['sha256']='0'*64
        inputs['full_content_sha256']=a.digest(inputs['files'])
        a.write(root/'inputs.json',inputs,immutable=False)
    with pytest.raises(ValueError): runtime.verify_inputs(plan,full_hash=True)


def test_source_audit_records_compatibility_and_never_rewrites_old_artifacts(historical_inputs,monkeypatch):
    runtime,plan,_ = historical_inputs
    root=Path(plan['outdir'])
    provenance=dict(generator_sha256='fixture generator',diagnostics=dict(
        fit_profile='full_TRAIN_streaming_author_schedule_CSI_pose_adaptation',
        fit_epochs=100,test_used=False,n_fit=133056,n_poisoned_fit=13305,
        generator_updates=13305*89))
    trigger=SimpleNamespace(shape=(3,114,10),training_indices=[1,4],provenance=lambda:provenance)
    evaluation=dict(binding=dict(execution=dict(fixture='CPU, not CUDA')))
    a.write(root/'backtime/evaluation.json',evaluation)
    for name in ('clean_surrogate/checkpoint.pt','backtime/trigger.pt','backtime/trigger_provenance.json',
                 'backtime/poison.json','backtime/checkpoint.pt','backtime/per_sample.npz'):
        a.write(root/name,{'fixture':name})
    monkeypatch.setattr(a,'source_runtime',lambda source:(runtime,plan))
    monkeypatch.setattr(runtime,'completed',lambda *args:evaluation)
    monkeypatch.setattr(runtime,'backtime_poison',lambda *args:dict(indices=[1,4]))
    monkeypatch.setattr(runtime,'load_trigger',lambda *args:trigger)
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    _,returned,_,receipt=a.source_audit(root,full_hash=True)
    assert returned == plan
    assert receipt['source_compatibility']['additions'] == a.AUDITED_SOURCE_ADDITIONS
    assert receipt['attacker_refit'] is False
    assert receipt['files']['plan.json'] == a.sha(root/'plan.json')
    assert {p:p.read_bytes() for p in root.rglob('*') if p.is_file()} == before


def test_new_transfer_plan_binds_full_current_snapshot_and_compatibility(historical_inputs,tmp_path,monkeypatch):
    runtime,old,_ = historical_inputs
    compatibility = a.bind_historical_sources(runtime,old)
    source = dict(root=old['outdir'],files={'plan.json':a.sha(Path(old['outdir'])/'plan.json')},
                  source_compatibility=compatibility)
    monkeypatch.setattr(a,'source_audit',lambda *args,**kw:(runtime,old,None,source))
    inputs = dict(train='fixture train',eval='fixture test',action_file_sha256='fixture prior')
    def clean_audit(backbone,home,folder):
        return dict(root=str(folder),cfg=a.configuration(old['cfg']['backtime'],backbone),
                    inputs=inputs,files={'checkpoint.pt':'fixture clean'})
    monkeypatch.setattr(a,'clean_audit',clean_audit)
    import run_a1_three_tables as frozen
    monkeypatch.setattr(frozen,'_current_inputs',lambda cell:inputs)
    args=SimpleNamespace(outdir=tmp_path/'runs/mmfi_backtime_transfer_fixture',source=old['outdir'],
        data_home=old['home'],metafi_clean=tmp_path/'clean_meta',wiflow_clean=tmp_path/'clean_wifi')
    plan=a.make_plan(args,full_hash=True)
    assert plan['sources']['scientific'] == a.native.common.sources()
    assert all(plan['sources']['scientific'][n] == h for n,h in a.AUDITED_SOURCE_ADDITIONS.items())
    assert plan['source']['source_compatibility'] == compatibility
    a.signed(plan,'plan_sha256')
    a.write(Path(plan['outdir'])/'plan.json',plan)
    a.verify_plan(plan,full_hash=True)
    drift=copy.deepcopy(plan['sources'])
    drift['scientific']['attack/action_targets.py']='0'*64
    monkeypatch.setattr(a,'setup_sources',lambda:drift)
    with pytest.raises(ValueError,match='changed'): a.verify_plan(plan,full_hash=True)


def test_training_rows_route_to_original_adaptive_manifest(tmp_path, monkeypatch):
    marker=object(); trigger=object(); captured=[]
    runtime=SimpleNamespace(training_rows=lambda p,m,t:(captured.append((p,m,t)) or marker))
    monkeypatch.setattr(a,'source_runtime',lambda source:(runtime,{'old':'plan'}))
    assert a.reused_rows({'source':{'root':'oldroot'}},trigger) is marker
    assert captured == [({'old':'plan'},'backtime',trigger)]


def test_concurrent_gate_checks_cfg_backbone_and_no_training(tmp_path,cfgs):
    plan=minimal_plan(tmp_path,cfgs)
    for b in a.BACKBONES:
        a.write(tmp_path/f'gpu_gate.{b}.json',dict(binding=a.binding(plan,b,{}),physical_batch=32,
            effective_batch=cfgs[b]['backbone_effective_batch_size'],batches=2,losses=[1.,.5],
            real_train=True,concurrent_workers=2,discarded=True,status='PREPARED_NOT_TRAINED'))
    a.gate_check(plan,{})
    with pytest.raises(ValueError): a.gate_check(plan,{'different':'GPU'})


def test_status_is_stdlib_only_readonly_in_a_fresh_process(tmp_path):
    script = """import runpy,sys
source=sys.argv[2]
sys.argv=['runner','--mode','status','--outdir',sys.argv[1]]
runpy.run_path(source,run_name='__main__')
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
"""
    folder=tmp_path/'does not exist'
    result=subprocess.run([sys.executable,'-B','-c',script,str(folder),str(ROOT/'remote_tools/Run-MMFiBackTimeTransfer4090.py')],
        capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr
    assert not folder.exists()


def test_no_attacker_fitting_or_historical_source_edits():
    code=(ROOT/'remote_tools/Run-MMFiBackTimeTransfer4090.py').read_text(encoding='utf-8')
    assert 'fit_backtime_stream(' not in code and '.fit_trigger(' not in code
    assert "runtime.training_rows(old, 'backtime', trigger)" in code
    assert 'taskkill' not in code and '.kill(' not in code and '.terminate(' not in code


@pytest.mark.parametrize('backbone',a.BACKBONES)
def test_cpu_fixture_fifty_epochs_exact_epoch_resume(tmp_path,cfgs,monkeypatch,backbone):
    """Real native optimizer/accumulation/dropout/RNG; toy victim, no CUDA/data claim."""
    import train_backdoor
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.drop=torch.nn.Dropout(.2)
            self.fc=torch.nn.Linear(1,51)
        def forward(self,x):
            value=self.drop(x.mean(dim=(1,2,3)).unsqueeze(1))
            return self.fc(value).reshape(-1,1,17,3),None
    class Trigger:
        requires_deferred_injection=True
        def to(self,*args):return self
        def eval(self):return self
        def apply_tensor(self,x,d):return (x+.01*d[:,None,None,None]).clamp(0,1)
    rng=np.random.default_rng(42)
    rows=[dict(csi=rng.uniform(0,1,(3,114,10)).astype(np.float32),
               pose=rng.normal(size=(1,17,3)).astype(np.float32),dose=.5,poisoned=int(i%3==0)) for i in range(65)]
    monkeypatch.setitem(a.COUNTS,'training',65)
    monkeypatch.setattr(a,'source_audit',lambda *args,**kw:(None,None,Trigger(),{}))
    monkeypatch.setattr(a,'reused_rows',lambda *args:rows)
    monkeypatch.setattr(a,'build_model',lambda *args:Toy())
    real_components=a.components
    monkeypatch.setattr(a,'components',lambda cfg:real_components(cfg,'cpu'))
    tensor_to=torch.Tensor.to
    def cpu_to(tensor,*args,**kw):
        if args and str(args[0])=='cuda:0':args=('cpu',*args[1:])
        return tensor_to(tensor,*args,**kw)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    real_rng=train_backdoor._rng_state
    monkeypatch.setattr(train_backdoor,'_rng_state',lambda:dict(real_rng(),cuda=[torch.get_rng_state()]))
    full=minimal_plan(tmp_path/'full',cfgs)
    resumed=minimal_plan(tmp_path/'resumed',cfgs)
    for plan in (full,resumed):Path(plan['outdir']).mkdir()
    a.train(full,backbone,{})
    saved=a.load_checkpoint(full,backbone,{},final=True)
    original_save=a.native.save_checkpoint
    def interrupted_save(path,value):
        original_save(path,value)
        if value['epoch']==2:raise RuntimeError('fixture interruption after saved epoch2')
    monkeypatch.setattr(a.native,'save_checkpoint',interrupted_save)
    with pytest.raises(RuntimeError,match='fixture interruption'):a.train(resumed,backbone,{})
    monkeypatch.setattr(a.native,'save_checkpoint',original_save)
    a.train(resumed,backbone,{})
    replayed=a.load_checkpoint(resumed,backbone,{},final=True)
    assert all(torch.equal(v,replayed['model'][k]) for k,v in saved['model'].items())
    assert torch.equal(saved['rng']['torch'],replayed['rng']['torch'])
    for param,state in saved['optimizer']['state'].items():
        for name,value in state.items():
            other=replayed['optimizer']['state'][param][name]
            assert torch.equal(value,other) if torch.is_tensor(value) else value==other


@pytest.mark.parametrize('backbone',a.BACKBONES)
def test_cpu_fixture_evaluator_exports_full_xyz_controls_and_replays(tmp_path,cfgs,monkeypatch,backbone):
    from attack.payload import make_target_pose,set_skeleton_config
    import extra_attack_eval
    set_skeleton_config('mmfi')
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pose=torch.nn.Parameter(torch.arange(51,dtype=torch.float32).reshape(1,1,17,3)/51)
        def forward(self,x):return self.pose+x.mean((1,2,3))[:,None,None,None],None
    class Trigger:
        def to(self,*args):return self
        def eval(self):return self
        def apply_tensor(self,x,d):return x+.1*d
    rng=np.random.default_rng(42)
    xs=torch.tensor(rng.uniform(0,1,(4,3,114,10)),dtype=torch.float32)
    true=rng.normal(size=(4,17,3)).astype(np.float32)
    class Data:
        items=[{'name':str(i)} for i in range(4)]
        def __len__(self):return len(self.items)
    data=Data()
    runtime=SimpleNamespace(dataset=lambda *args:data)
    monkeypatch.setattr(a,'source_audit',lambda *args,**kw:(runtime,{},Trigger(),{}))
    monkeypatch.setattr(a,'source_runtime',lambda *args:(runtime,{}))
    monkeypatch.setattr(a,'verify_plan',lambda *args:None)
    monkeypatch.setattr(a,'build_model',lambda *args:Toy())
    monkeypatch.setitem(a.COUNTS,'test',4)
    tensor_to=torch.Tensor.to
    def cpu_to(tensor,*args,**kw):
        if args and str(args[0])=='cuda:0':args=('cpu',*args[1:])
        return tensor_to(tensor,*args,**kw)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    def loader(data,trigger,cfg,mode,dose):
        target=make_target_pose(true,1,dose,np.deg2rad(40),target_joints=[2,3]).astype(np.float32)
        yield dict(csi=xs,pose=torch.from_numpy(true[:,None]),target=torch.from_numpy(target[:,None]))
    monkeypatch.setattr(extra_attack_eval,'_loader',loader)
    plan=minimal_plan(tmp_path,cfgs)
    plan['contract']=dict(doses=list(a.GRID),target=[2,3])
    clean_root=tmp_path/'clean';clean_root.mkdir()
    plan['clean'][backbone]['root']=str(clean_root)
    clean=Toy()(xs)[0].detach().numpy()[:,0]
    dummy=[dict(t_mpjpe_mm=0.,target_pck_pct=0.,dce_mm=0.)]*6
    utility=a.summarize(clean,true,dummy)
    plan['clean'][backbone].update({k:utility[k] for k in ('clean_mpjpe_mm','clean_pampjpe_mm')})
    blob=dict(model=Toy().state_dict(),epoch=49)
    a.native.save_checkpoint(clean_root/'checkpoint.pt',blob)
    folder=tmp_path/backbone;folder.mkdir()
    a.native.save_checkpoint(folder/'checkpoint.pt',blob)
    monkeypatch.setattr(a,'load_checkpoint',lambda *args,**kw:blob)
    a.evaluate(plan,backbone,{})
    record=a.complete(plan,backbone,{})
    assert record['frames']==4 and set(record['records'])=={'clean_reference','backtime'}
    assert all(len(r['dose_rows'])==6 for r in record['records'].values())
    with np.load(folder/'per_sample.npz',allow_pickle=False) as arrays:
        assert arrays['target_poses'].shape==(6,4,17,3)
        assert arrays['backtime__trigger_predictions'].shape==(6,4,17,3)
        assert arrays['pair_ids'].tolist()==['0','1','2','3']


@pytest.mark.skipif(os.name != 'nt',reason='Windows PowerShell')
def test_powershell_parse():
    path=ROOT/'remote_tools/Start-MMFiBackTimeTransfer4090.ps1'
    result=subprocess.run(['powershell','-NoProfile','-Command',
        "$t=$null; $e=$null; [System.Management.Automation.Language.Parser]::ParseFile('"+str(path)+
        "',[ref]$t,[ref]$e) | Out-Null; if ($e.Count) {throw ($e|Out-String)}"],capture_output=True,text=True,timeout=30)
    assert result.returncode == 0,result.stderr
