"""CPU/mock and frozen-native synthetic checks; no remote CUDA claims."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'remote_tools/piw_metafi_four_attacks_4090.py'
HELPER = ROOT / 'remote_tools/Start-PiWMetaFiFourAttacks4090.ps1'


@pytest.fixture
def r():
    spec = importlib.util.spec_from_file_location('four_unit', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def x(r, tmp_path):
    a1 = r.a1_sidecar()
    base = a1.diagnostic().source_probe().base_module()
    args = NS(data_home=tmp_path, out=tmp_path / 'runs' / r.PROFILE,
              dataset=tmp_path / 'dataset', action=tmp_path / 'action.npy',
              clean_run=tmp_path / 'runs' / a1.CLEAN_PROFILE, mode='prepare')
    args.out.mkdir(parents=True)
    # Small metadata fixture; full-content validation is delegated to the
    # unchanged a1.reference, tested in test_piw_metafi_a1_4090.py.
    cfg = copy.deepcopy(base.original_plan()['cfg'])
    cfg.update(profile=r.PROFILE, outdir=str(args.out), dataset_root=str(args.dataset),
               action_npy=str(args.action), methods=list(r.METHODS), backbones=['metafiplusplus'])
    cfg['cover_ratios'].update(r.COVERS)
    cfg['recipes'] = {'metafiplusplus': copy.deepcopy(base.RECIPE)}
    ref_plan = dict(cfg=dict(cfg, profile=a1.CLEAN_PROFILE, methods=['clean'], outdir=str(args.clean_run)),
        inputs={'fingerprint': 'full-audit', 'splits': {}}, trigger={'frozen': True},
        poison={'samples': [], 'n_poison': 2787}, action_sha256='action',
        sources={'scientific': 'unchanged'}, runtime={'exact-env': True})
    ref = dict(plan=ref_plan, binding={'plan_sha256': 'clean', 'files': {'checkpoint.pt': 'clean-checkpoint'}},
        stats='stats', execution={'gpu_name': 'RTX4090 fixture'},
        evaluation={'clean_metrics': {'clean_mpjpe_mm': 163.63, 'clean_pampjpe_mm': 80.25}})
    def record(method):
        return dict(method=method, n_total=27876, n_poison=2787,
            n_cover=int(r.COVERS[method]*27876), cover_ratio=r.COVERS[method],
            target='piw_body_hip_axis_v2', phase_policy='unchanged',
            operator_keys=['key']*3, cover_indices_sha256='cover', poison_sha256='poison')
    records = {m: record(m) for m in r.METHODS}
    w = NS(plan=lambda c: dict(schema=1, profile=r.PROFILE, cfg=c,
            sources=ref_plan['sources'], runtime=ref_plan['runtime']),
        cfg_for=lambda *a: copy.deepcopy(cfg),
        manifest_path=lambda c: args.out / 'plan.json',
        receipt_path=lambda c: args.out / 'preflight_receipt.json',
        load_plan=lambda c: base.read(args.out / 'plan.json'),
        _datasets=lambda s: {'training': 'train'},
        make_trigger=lambda method, train, c: method,
        make_training_data=lambda s, m, ds, trigger: ('pairs', copy.deepcopy(records[m])))
    return NS(args=args, a1=a1, base=base, cfg=cfg, ref=ref, w=w, records=records)


def test_inspect_and_status_stdlib_readonly(tmp_path):
    code = """
import importlib.util,sys
s=importlib.util.spec_from_file_location('r',sys.argv[1]);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
for mode in ('inspect','status'):r.main(['--mode',mode,'--data-home',sys.argv[2]])
assert not any(k in sys.modules for k in ('torch','numpy','scipy'))
"""
    p = subprocess.run([sys.executable, '-c', code, str(SCRIPT), str(tmp_path)], capture_output=True, text=True, timeout=45)
    assert p.returncode == 0, p.stdout + p.stderr
    assert not (tmp_path / 'runs').exists()
    assert 'fiba' in p.stdout and 'bpp_source' in p.stdout


def test_exact_scope_and_recipe(r, x):
    assert r.METHODS == ('fiba', 'wanet_source', 'ftrojan', 'bpp_source')
    assert not set(r.METHODS).intersection(('clean','a1','tsba_adapted','smooth_lf_csi'))
    assert r.configuration(x.args, x.w)['recipes']['metafiplusplus'] == x.base.RECIPE


@pytest.mark.parametrize('field,value', [('methods', ['a1']), ('seed', 43), ('epochs', 50),
    ('batch_size', 16), ('amp', True), ('recipes', {'metafiplusplus': {'optimizer': 'sgd'}})])
def test_refuses_changed_training_recipe(r, x, field, value):
    x.cfg[field] = value
    with pytest.raises(ValueError, match='recipes'):
        r.configuration(x.args, x.w)


@pytest.mark.parametrize('method', ['fiba','wanet_source','ftrojan','bpp_source'])
@pytest.mark.parametrize('field,value', [('n_poison', 1), ('n_cover', -1), ('cover_ratio', .7),
    ('operator_keys', ['one']), ('phase_policy', 'changed'), ('target', 'other')])
def test_refuses_changed_actual_operator_or_cover(r, x, method, field, value):
    bad = dict(x.records[method], **{field: value})
    with pytest.raises(ValueError, match=method):
        r.validate_poison(method, bad)


def test_create_plan_reuses_clean_without_checkpoint_copy(r, x):
    saved = r.create_plan(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    assert saved['native_records'] == x.records
    assert saved['poison'] == x.ref['plan']['poison']
    assert saved['trigger'] == x.ref['plan']['trigger']
    assert saved['clean_reference'] == x.ref['binding']
    assert saved['plan_sha256'] == x.base.digest({k: v for k, v in saved.items() if k != 'plan_sha256'})
    assert x.base.read(x.args.out / 'preflight_receipt.json')['audit_reuse'] == x.ref['binding']
    assert not list(x.args.out.rglob('checkpoint.pt'))


def test_preexisting_checkpoint_never_imported(r, x):
    cp = x.args.out / r.CELL / 'fiba' / 'checkpoint.pt'
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b'foreign')
    with pytest.raises(ValueError, match='Ungated'):
        r.create_plan(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    assert cp.read_bytes() == b'foreign'


def test_prepare_only_four_native_smokes_and_no_training(r, x, monkeypatch):
    calls = []
    monkeypatch.setattr(x.a1, 'require_idle', lambda b: None)
    def smoke(saved, backbone, method, device):
        calls.append(method)
        print('GPU_SMOKE_WORKER_PASS ' + json.dumps(dict(cell=f'metafiplusplus__s42__{method}',
            device=device, batch=32, loss=3., generator_steps=0,
            peak_cuda_allocated=10, peak_cuda_reserved=20)))
    x.w.smoke_cell = smoke
    r.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    assert calls == list(r.METHODS)
    assert not list(x.args.out.rglob('checkpoint.pt'))
    r.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    assert calls == list(r.METHODS)  # gated Prepare never repeats smokes


def test_smoke_failure_stops_later_methods(r, x, monkeypatch):
    calls = []
    monkeypatch.setattr(x.a1, 'require_idle', lambda b: None)
    def smoke(*a):
        calls.append(a[-1])
        raise RuntimeError('capacity failure')
    monkeypatch.setattr(r, 'native_smoke', smoke)
    with pytest.raises(RuntimeError, match='capacity'):
        r.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    assert calls == ['fiba']
    assert not (x.args.out / 'prelaunch_gate.json').exists()


def test_launch_without_gate_cannot_train(r, x):
    r.create_plan(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    x.args.mode = 'launch'
    with pytest.raises(FileNotFoundError):
        r.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)


def setup_launch(r, x, monkeypatch):
    r.create_plan(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    x.args.mode = 'launch'
    events = []
    monkeypatch.setattr(x.a1, 'require_idle', lambda b: None)
    monkeypatch.setattr(r, 'check_gate', lambda *a: None)
    monkeypatch.setattr(r, 'own_checkpoint', lambda *a, **k: {'epoch': 199})
    monkeypatch.setattr(r, 'evaluate', lambda *a: events.append(('eval', a[-1])))
    monkeypatch.setattr(r, 'summary', lambda *a: events.append(('summary',)))
    x.w.train_cell = lambda s, b, m, d: events.append(('train', m))
    return events


def test_serial_train_all_then_evaluate_all(r, x, monkeypatch):
    events = setup_launch(r, x, monkeypatch)
    r.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    assert events == [('train', m) for m in r.METHODS] + [('eval', m) for m in r.METHODS] + [('summary',)]


def test_resume_skips_already_final_cells(r, x, monkeypatch):
    events = setup_launch(r, x, monkeypatch)
    for method in r.METHODS[:2]:
        cp = x.args.out / r.CELL / method / 'checkpoint.pt'
        cp.parent.mkdir(parents=True)
        cp.write_bytes(b'own-final-fixture')
    r.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    assert [e for e in events if e[0] == 'train'] == [('train', m) for m in r.METHODS[2:]]


def test_failed_training_never_launches_later_or_evaluate(r, x, monkeypatch):
    events = setup_launch(r, x, monkeypatch)
    def train(s, b, m, d):
        events.append(('train', m))
        if m == 'wanet_source':
            raise RuntimeError('interrupted')
    x.w.train_cell = train
    with pytest.raises(RuntimeError, match='interrupted'):
        r.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    assert events == [('train', 'fiba'), ('train', 'wanet_source')]


@pytest.mark.parametrize('key,value', [('clean_reference', {}), ('poison', {}), ('sources', {}), ('runtime', {})])
def test_plan_reference_source_drift_rejected(r, x, key, value):
    saved = r.create_plan(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    saved[key] = value
    with pytest.raises(ValueError):
        r.validate_plan(x.a1, saved, x.ref)


def make_existing_a1(x):
    root = x.args.data_home / 'runs' / x.a1.PROFILE
    folder = root / x.a1.CELL
    folder.mkdir(parents=True)
    (folder / 'checkpoint.pt').write_bytes(b'historical-a1-checkpoint')
    (folder / 'predictions.npz').write_bytes(b'historical-a1-predictions')
    saved = copy.deepcopy(x.ref['plan'])
    saved.update(profile=x.a1.PROFILE, clean_reference=x.ref['binding'])
    saved['cfg'].update(profile=x.a1.PROFILE, methods=['a1'], outdir=str(root))
    saved['plan_sha256'] = x.base.digest(saved)
    metrics = dict(t1_mpjpe_mm=210., target_pck_pct=30., mean_dce_mm=90.,
                   clean_mpjpe_mm=170., clean_pampjpe_mm=85.)
    result = dict(profile=x.a1.PROFILE, status='COMPLETE', method='a1', backbone='metafiplusplus',
        seed=42, plan_sha256=saved['plan_sha256'], poison=x.a1.POISON,
        clean_reference=x.ref['binding'], clean_checkpoint_sha256='clean-checkpoint',
        checkpoint_sha256=x.base.sha(folder / 'checkpoint.pt'),
        predictions_sha256=x.base.sha(folder / 'predictions.npz'), metrics={'mainmetrics': metrics})
    x.base.write(root / 'plan.json', saved)
    x.base.write(folder / 'evaluation.json', result)
    return root, folder, metrics


def test_existing_a1_is_readonly_and_report_includes_it_without_training(r, x):
    saved = r.create_plan(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    root, folder, metrics = make_existing_a1(x)
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert r.existing_a1(x.args, x.a1, x.base, x.ref) == metrics
    r.summary(x.args, x.a1, x.base, saved, x.ref)
    text = (x.args.out / 'RESULT.md').read_text(encoding='utf-8')
    assert 'A1 (existing; not retrained)' in text
    assert text.count('pending') == 4
    assert x.base.read(x.args.out / 'results.json')['status'] == 'PARTIAL'
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('key,value', [('seed', 43), ('method', 'clean'), ('status', 'TRAINED_NOT_EVALUATED'),
                                     ('clean_reference', {}), ('checkpoint_sha256', 'other')])
def test_foreign_existing_a1_not_silently_added(r, x, key, value):
    root, folder, _ = make_existing_a1(x)
    result = x.base.read(folder / 'evaluation.json')
    result[key] = value
    x.base.write(folder / 'evaluation.json', result)
    with pytest.raises(ValueError):
        r.existing_a1(x.args, x.a1, x.base, x.ref)


def test_missing_a1_never_schedules_retraining(r, x):
    assert r.existing_a1(x.args, x.a1, x.base, x.ref) is None


def test_frozen_native_operators_and_external_evaluation_synthetic(tmp_path):
    code = r'''
import importlib.util,sys,copy,hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('r',sys.argv[1]);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
a1=r.a1_sidecar();b=a1.diagnostic().source_probe().base_module();home=Path(sys.argv[2])
a=NS(data_home=home,out=home/'runs'/r.PROFILE,clean_run=home/'runs'/a1.CLEAN_PROFILE,dataset=home/'dataset',action=home/'action.npy')
w=b.load_worker(a,'a1');native={n:getattr(w,n) for n in ('train_cell','_build','_optimizer','_train_batch','_seed','measure_predictions','smoke_cell')}
w=r.worker(a,b);assert all(getattr(w,n) is fn for n,fn in native.items())
cfg=r.configuration(a,w);assert cfg['recipes']=={'metafiplusplus':b.RECIPE}
old=b.original_plan();names=old['inputs']['splits']['train_data']['ordered_names']
class Train:
 split='training'
 def __init__(self): self.items=[dict(name=n,csi='/local/csi/'+n+'.npy',kpt='/local/pose/'+n+'.npy') for n in names]
 def __len__(self): return len(self.items)
 def load_raw(self,path):
  shift=(int(hashlib.sha256(str(path).encode()).hexdigest()[:4],16)%100)/1000
  return (np.linspace(.1,.7,3*180*20).reshape(3,180,20)+shift).astype(np.float32)
 def normalize(self,v): return v
train=Train();saved=dict(cfg=cfg,poison=old['poison'])
records={};states={};value=np.random.default_rng(2).uniform(.1,.8,(3,180,20)).astype(np.float32)
for m in r.METHODS:
 trigger=w.make_trigger(m,train,cfg);_,record=w.make_training_data(saved,m,{'training':train},trigger)
 r.validate_poison(m,record);records[m]=record
 if m=='ftrojan': assert record['operator_keys']==[b.FT_PATTERN_SHA]*3
 assert np.array_equal(trigger.inject(value,0),value)
 assert np.array_equal(trigger.inject(value,.7)[:,90:],value[:,90:])
 if r.COVERS[m]:trigger.noise_inject(value,dose=1.)
 state=copy.deepcopy(trigger.state_dict());fresh=w.make_trigger(m,train,cfg);fresh.load_state_dict(state)
 assert fresh.fixed_keys()==trigger.fixed_keys()
 if r.COVERS[m]:assert np.array_equal(trigger.noise_inject(value,dose=1.),fresh.noise_inject(value,dose=1.))
 states[m]=state
assert records['wanet_source']['n_cover']==5575
# Exercise new external-reference evaluator with real frozen metric code,
# synthetic full2586 poses, toy model and a predictor spy (CPU only).
rng=np.random.default_rng(3);y=rng.normal(size=(2586,14,3)).astype(np.float32)
testnames=[str(i) for i in range(2586)];grid=np.stack([w.target_pose(y,d) for d in w.DOSES],axis=1)
plain=np.repeat(y[:,None],6,axis=1)
clean_dir=a.clean_run/a1.CLEAN_CELL;clean_dir.mkdir(parents=True)
np.savez_compressed(clean_dir/'predictions.npz',truth=y,names=np.asarray(testnames),predictions=y)
class Tiny(torch.nn.Module):
 def __init__(self):super().__init__();self.role=torch.nn.Parameter(torch.tensor(0.))
env={'gpu_name':'mock4090'}
torch.save(dict(epoch=199,model=Tiny().state_dict(),execution=env),clean_dir/'checkpoint.pt')
ref=dict(plan={'plan_sha256':'clean-plan'},execution=env,binding={'files':{'checkpoint.pt':b.sha(clean_dir/'checkpoint.pt')}})
saved.update(plan_sha256='new-plan',native_records=records,inputs={'splits':{'test_data':{'ordered_names':testnames}}})
native_metric=w.measure_predictions;native_npz=w._atomic_npz
w._datasets=lambda s:{'training':train,'validation':'test'};w._test_arrays=lambda d:(np.zeros((2586,1),np.float32),y,testnames)
w._build=lambda *a:Tiny()
loaded=[]
class Trigger:
 def load_state_dict(self,v):loaded.append(v)
w.make_trigger=lambda *a:Trigger()
w._predict_grid=lambda model,x,t,doses,device:grid if model.role.item()==1 else plain
calls=[];b.validate_blob=lambda blob,plan,method,poison:calls.append(method)
for m in r.METHODS:
 folder=a.out/r.CELL/m;folder.mkdir(parents=True)
 own=dict(epoch=199,model={'role':torch.tensor(1.)},execution=env,trigger=states[m],poison=records[m])
 torch.save(own,folder/'checkpoint.pt')
 w._checkpoint=lambda *a,**k:own
 before=(clean_dir/'checkpoint.pt').read_bytes()
 r.evaluate(a,a1,b,w,saved,ref,m)
 out=r.completed(a,b,saved,ref,m)
 assert out['metrics']==native_metric(y,grid[:,0],grid,y,plain)
 assert loaded[-1]==states[m]
 assert (clean_dir/'checkpoint.pt').read_bytes()==before
assert calls==[v for m in r.METHODS for v in (m,'clean')]
print('NATIVE_FOUR_OPERATORS_STATES_AND_EXTERNAL_METRIC_PARITY_PASS')
'''
    p = subprocess.run([sys.executable, '-c', code, str(SCRIPT), str(tmp_path / 'native')],
                       capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'METRIC_PARITY_PASS' in p.stdout


def test_helper_parser_no_installs_or_kills():
    text = HELPER.read_text(encoding='utf-8')
    assert 'pip install' not in text and 'Stop-Process' not in text and 'Start-Process' not in text
    assert 'piw-metafi-vast211-4090' in text and 'piw_metafi_four_attacks_4090.py' in text
    ps = shutil.which('powershell') or shutil.which('pwsh')
    if not ps:
        pytest.skip('PowerShell unavailable')
    path = str(HELPER).replace("'", "''")
    code = f"$t=$null;$e=$null;[Management.Automation.Language.Parser]::ParseFile('{path}',[ref]$t,[ref]$e)|Out-Null;if($e.Count){{$e;exit 1}}"
    p = subprocess.run([ps, '-NoProfile', '-Command', code], capture_output=True, text=True, timeout=40)
    assert p.returncode == 0, p.stdout + p.stderr
