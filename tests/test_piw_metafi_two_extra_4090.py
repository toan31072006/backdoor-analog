"""Local CPU/mock regression only; never claims full PiW or remote CUDA success."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'remote_tools/piw_metafi_two_extra_4090.py'
HELPER = ROOT / 'remote_tools/Start-PiWMetaFiTwoExtra4090.ps1'


def native_probe(code, folder):
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_THREADING_LAYER='SEQUENTIAL')
    code = 'import faulthandler\nfaulthandler.dump_traceback_later(60, repeat=True)\n' + code
    try:
        python = env.get('PIW_NATIVE_TEST_PYTHON', sys.executable)
        return subprocess.run([python, '-B', '-c', code, str(SCRIPT), str(folder)],
                              capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired as error:
        pytest.fail(f'Native CPU probe timed out; stdout={error.stdout!r}; stderr={error.stderr!r}')


@pytest.fixture
def r():
    spec = importlib.util.spec_from_file_location('two_extra_test', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def x(r, tmp_path):
    q = r.queue_module()
    a1 = q.a1_sidecar()
    base = a1.diagnostic().source_probe().base_module()
    args = NS(data_home=tmp_path, out=tmp_path / 'runs' / r.PROFILE,
        dataset=tmp_path / 'dataset', action=tmp_path / 'action.npy',
        clean_run=tmp_path / 'runs' / a1.CLEAN_PROFILE, mode='prepare')
    args.out.mkdir(parents=True)
    cfg = copy.deepcopy(base.original_plan()['cfg'])
    cfg.update(profile=r.PROFILE, outdir=str(args.out), dataset_root=str(args.dataset),
        action_npy=str(args.action), methods=list(r.METHODS), backbones=['metafiplusplus'])
    cfg['recipes'] = {'metafiplusplus': copy.deepcopy(base.RECIPE)}
    ref_plan = dict(cfg=dict(cfg, profile=a1.CLEAN_PROFILE, methods=['clean'], outdir=str(args.clean_run)),
        inputs={'fingerprint': 'full-audit', 'splits': {}}, trigger={'frozen': True},
        poison={'samples': [], 'n_poison': 2787, 'poison_plan_sha256': 'p' * 64},
        action_sha256='action', sources={'scientific': 'unchanged'}, runtime={'exact-env': True})
    ref = dict(plan=ref_plan, binding={'plan_sha256': 'clean', 'files': {'checkpoint.pt': 'clean-hash'}},
        stats='stats', execution={'gpu_name': 'RTX4090 fixture'},
        evaluation={'clean_metrics': {'clean_mpjpe_mm': 163.63, 'clean_pampjpe_mm': 80.25}})
    records = {m: dict(n_total=27876, n_poison=2787, n_cover=0, poison_sha256='p' * 64) for m in r.METHODS}
    w = NS(plan=lambda c: dict(schema=1, profile=r.PROFILE, cfg=c,
        sources=ref_plan['sources'], runtime=ref_plan['runtime']), cfg_for=lambda *a: copy.deepcopy(cfg),
        manifest_path=lambda c: args.out / 'plan.json', receipt_path=lambda c: args.out / 'preflight_receipt.json',
        load_plan=lambda c: base.read(args.out / 'plan.json'), _datasets=lambda s: {'training': 'train'},
        make_trigger=lambda m, train, c: m,
        make_training_data=lambda s, m, ds, trigger: ('pairs', copy.deepcopy(records[m])))
    return NS(q=q, a1=a1, base=base, args=args, cfg=cfg, ref=ref, w=w, records=records)


def smoke_row(method):
    return dict(cell=f'metafiplusplus__s42__{method}', device='cuda:0', batch=32,
        loss=3., generator_steps=4 if method == 'tsba_adapted' else 0,
        peak_cuda_allocated=1, peak_cuda_reserved=2)


def test_inspect_and_status_are_stdlib_readonly(tmp_path):
    code = '''
import importlib.util,sys
s=importlib.util.spec_from_file_location('r',sys.argv[1]);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
for mode in ('inspect','status'): r.main(['--mode',mode,'--data-home',sys.argv[2],'--out',sys.argv[3]])
assert not any(k in sys.modules for k in ('torch','numpy','scipy'))
'''
    p = subprocess.run([sys.executable, '-B', '-c', code, str(SCRIPT), str(tmp_path),
        str(tmp_path / 'new-out')], text=True, capture_output=True, timeout=30)
    assert p.returncode == 0, p.stdout + p.stderr
    assert not (tmp_path / 'new-out').exists()
    assert 'tsba_adapted' in p.stdout and '"external_gpu_process_limit": null' in p.stdout


def test_exact_two_method_scope_and_recipe(r, x):
    assert r.METHODS == ('tsba_adapted', 'smooth_lf_csi')
    cfg = r.configuration(x.q, x.args, x.w)
    assert cfg['recipes']['metafiplusplus'] == x.base.RECIPE
    assert cfg['seed'] == 42 and cfg['epochs'] == 200 and cfg['batch_size'] == 32
    assert cfg['target']['version'] == 'piw_body_hip_axis_v2'
    assert cfg['target']['target_joints'] == [11, 13]


@pytest.mark.parametrize('field,value', [('rho', .2), ('seed', 43), ('epochs', 50),
    ('batch_size', 16), ('amp', True), ('tsba', {'eps': .5}), ('eval_doses', [0., 1.])])
def test_changed_recipe_refused(r, x, field, value):
    x.cfg[field] = value
    with pytest.raises(ValueError):
        r.configuration(x.q, x.args, x.w)


@pytest.mark.parametrize('method', ['tsba_adapted', 'smooth_lf_csi'])
@pytest.mark.parametrize('field,value', [('n_total', 100), ('n_poison', 1), ('n_cover', 1),
    ('poison_sha256', ''), ('extra', True)])
def test_poison_drift_refused(r, x, method, field, value):
    with pytest.raises(ValueError):
        r.validate_poison(method, dict(x.records[method], **{field: value}))


@pytest.mark.parametrize('method', ['tsba_adapted', 'smooth_lf_csi'])
@pytest.mark.parametrize('field,value', [('batch', 16), ('device', 'cpu'), ('loss', float('nan')),
    ('loss', True), ('peak_cuda_allocated', -1), ('generator_steps', 2), ('cell', 'hpeli__s42__a1')])
def test_wrong_smoke_rejected(r, method, field, value):
    with pytest.raises(ValueError):
        r.validate_smoke(dict(smoke_row(method), **{field: value}), method)


def prepare(x):
    x.w.smoke_cell = lambda s, b, m, d: print('GPU_SMOKE_WORKER_PASS ' + json.dumps(smoke_row(m)))
    x.q.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    return x.w.load_plan(x.cfg)


def test_prepare_two_smokes_generator_four_steps_no_training(r, x):
    saved = prepare(x)
    gate = x.base.read(x.args.out / 'prelaunch_gate.json')
    assert gate['policy'] == r.POLICY
    assert gate['methods'] == list(r.METHODS)
    assert saved['native_records'] == x.records
    assert saved['poison'] == x.ref['plan']['poison']
    assert not list(x.args.out.rglob('checkpoint.pt'))
    x.w.smoke_cell = lambda *a: pytest.fail('no repeated smoke')
    x.q.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)


def test_external_jobs_do_not_trigger_pid_or_idle_query(r, x, monkeypatch):
    # The library's own process guard stays intact, but this queue never calls it.
    monkeypatch.setattr(x.a1, 'require_idle', lambda *a: pytest.fail('PID guard unexpectedly called'))
    monkeypatch.setattr(x.base.subprocess, 'run', lambda *a, **k: pytest.fail('process enumeration'))
    prepare(x)


def test_wrong_generator_smoke_stops_before_second_cell(r, x):
    seen = []
    def smoke(s, b, m, d):
        seen.append(m)
        print('GPU_SMOKE_WORKER_PASS ' + json.dumps(dict(smoke_row(m), generator_steps=0)))
    x.w.smoke_cell = smoke
    with pytest.raises(ValueError):
        x.q.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    assert seen == ['tsba_adapted']
    assert not (x.args.out / 'prelaunch_gate.json').exists()


def test_serial_launch_then_evaluation_and_no_global_idle_guard(r, x, monkeypatch):
    prepare(x)
    x.args.mode = 'launch'
    calls = []
    monkeypatch.setattr(x.q, 'own_checkpoint', lambda *a, **k: {'epoch': 199})
    monkeypatch.setattr(x.q, 'evaluate', lambda *a: calls.append(('eval', a[-1])))
    monkeypatch.setattr(x.q, 'summary', lambda *a: calls.append(('summary',)))
    x.w.train_cell = lambda s, b, m, d: calls.append(('train', m))
    x.q.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    assert calls == [('train', 'tsba_adapted'), ('train', 'smooth_lf_csi'),
                     ('eval', 'tsba_adapted'), ('eval', 'smooth_lf_csi'), ('summary',)]


def test_training_failure_never_starts_later_method_or_eval(r, x, monkeypatch):
    prepare(x)
    x.args.mode = 'launch'
    seen = []
    def fail(s, b, m, d):
        seen.append(m)
        raise RuntimeError('OOM fixture')
    x.w.train_cell = fail
    monkeypatch.setattr(x.q, 'evaluate', lambda *a: pytest.fail('evaluation after failure'))
    with pytest.raises(RuntimeError, match='OOM'):
        x.q.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    assert seen == ['tsba_adapted']


def test_gate_tamper_and_no_ungated_launch(r, x):
    x.args.mode = 'launch'
    with pytest.raises(FileNotFoundError):
        x.q.run(x.args, x.a1, x.base, x.w, x.cfg, x.ref)
    x.args.mode = 'prepare'
    saved = prepare(x)
    gate = x.base.read(x.args.out / 'prelaunch_gate.json')
    gate['policy']['max_owned_gpu_workers'] = 2
    x.base.write(x.args.out / 'prelaunch_gate.json', gate)
    with pytest.raises(ValueError, match='gate changed'):
        x.q.check_gate(x.args, x.a1, x.base, x.w, saved, x.ref)


def test_cli_method_only_for_evaluation(r):
    with pytest.raises(SystemExit):
        r.parse_args(['--mode', 'launch', '--method', 'tsba_adapted'])
    assert r.parse_args(['--mode', 'evaluate', '--method', 'tsba_adapted']).method == 'tsba_adapted'


def test_helper_syntax_no_install_kill_or_process_count():
    source = HELPER.read_text()
    for forbidden in ('Stop-Process', 'pip install', 'Get-CimInstance', 'Start-Process'):
        assert forbidden not in source
    assert "'Evaluate'" in source and "'Prepare'" in source
    ps = shutil.which('powershell') or shutil.which('pwsh')
    if not ps:
        pytest.skip('PowerShell unavailable')
    command = "$t=$null;$e=$null;[Management.Automation.Language.Parser]::ParseFile('" + str(HELPER).replace("'", "''") + "',[ref]$t,[ref]$e)|Out-Null;if($e.Count){$e;exit 1}"
    p = subprocess.run([ps, '-NoProfile', '-Command', command], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stdout + p.stderr


def test_package_is_self_contained_for_frozen_source_and_clean_bindings(tmp_path):
    spec = importlib.util.spec_from_file_location('two_extra_builder',
                                                ROOT / 'remote_tools/Build-PiWMetaFiTwoExtra4090.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    package = tmp_path / 'handoff.zip'
    builder.build(package)
    with pytest.raises(FileExistsError):
        builder.build(package)
    extracted = tmp_path / 'extracted'
    with zipfile.ZipFile(package) as z:
        z.extractall(extracted)
    spec = importlib.util.spec_from_file_location('extracted_two_extra',
        extracted / 'remote_tools/piw_metafi_two_extra_4090.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    q = runner.queue_module()
    a1 = q.a1_sidecar()
    base = a1.diagnostic().source_probe().base_module()
    assert a1.diagnostic().fresh_probe().code_binding(base)
    assert (base.bootstrap(tmp_path / 'run') / 'ATKBackd/run_piw_two_models_attacks.py').is_file()
    assert runner.main(['--mode', 'inspect', '--data-home', str(tmp_path / 'missing_data')]) == 0


def test_frozen_native_functions_and_real_cpu_generator_step(tmp_path):
    code = '''
import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace as NS
print('CPU_TEST: importing torch/numpy',flush=True)
import torch,numpy as np
print('CPU_TEST: imports complete',flush=True)
torch.set_num_threads(2)
s=importlib.util.spec_from_file_location('r',sys.argv[1]);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
q=r.queue_module();b=q.a1_sidecar().diagnostic().source_probe().base_module();a=NS(out=Path(sys.argv[2]))
w=b.load_worker(a,'a1'); native={k:getattr(w,k) for k in ('train_cell','_train_batch','_build','_optimizer','target_pose','measure_predictions','smoke_cell')}
w=q.worker(a,b)
print('CPU_TEST: frozen worker loaded',flush=True)
assert all(getattr(w,k) is v for k,v in native.items())
cfg=w.cfg_for(Path(sys.argv[2])/'data',Path(sys.argv[2])/'run',Path(sys.argv[2])/'action.npy')
trigger=w.make_trigger('tsba_adapted',None,cfg)
smooth=w.make_trigger('smooth_lf_csi',None,cfg)
x=torch.rand(4,3,180,20);phase=x[:,:,90:].clone()
assert torch.equal(trigger.inject_tensor(x,0),x)
assert torch.equal(trigger.inject_tensor(x,1)[:,:,90:],phase)
assert np.array_equal(smooth.inject(x[0].numpy(),1)[:,90:],phase[0].numpy())
class Tiny(torch.nn.Module):
 def __init__(self): super().__init__();self.fc=torch.nn.Linear(3,42)
 def forward(self,x): return self.fc(x.mean((2,3))).reshape(-1,1,14,3),None
model=Tiny();before=[p.detach().clone() for p in trigger.parameters()]
opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
gopt=torch.optim.Adam(trigger.parameters(),lr=.001)
y=torch.rand(4,1,14,3)
batch=dict(csi=x,pose=y,clean_pose=y,poisoned=torch.tensor([1,1,0,0],dtype=torch.bool),dose=torch.tensor([.5,1.,0.,0.]))
print('CPU_TEST: generator and victim updates',flush=True)
loss,n,steps=w._train_batch(model,opt,batch,'cpu','metafiplusplus',trigger,gopt,cfg,epoch=10,last_micro=True,group_size=4)
assert n==4 and len(steps)==4 and np.isfinite(loss)
assert any(not torch.equal(v,p) for v,p in zip(before,trigger.parameters()))
assert model.training
print('NATIVE_IDENTITY_PHASE_GRADIENT_4_STEPS_PASS; toy model only, not CUDA capacity')
'''
    p = native_probe(code, tmp_path / 'frozen')
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'NATIVE_IDENTITY_PHASE_GRADIENT_4_STEPS_PASS' in p.stdout


def test_full_shape_evaluation_restores_tsba_and_replays_metrics(tmp_path):
    code = '''
import importlib.util,sys,copy
from pathlib import Path
from types import SimpleNamespace as NS
print('EVAL_TEST: importing numpy/torch',flush=True)
import numpy as np,torch
print('EVAL_TEST: imports complete',flush=True)
torch.set_num_threads(2)
s=importlib.util.spec_from_file_location('r',sys.argv[1]);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
q=r.queue_module();a1=q.a1_sidecar();b=a1.diagnostic().source_probe().base_module()
root=Path(sys.argv[2]);a=NS(out=root/'out',clean_run=root/'clean')
w=q.worker(a,b);cfg=w.cfg_for(root/'data',a.out,root/'action.npy')
print('EVAL_TEST: frozen worker loaded',flush=True)
y=np.random.default_rng(123).normal(size=(2586,14,3)).astype(np.float32)
names=[str(i) for i in range(2586)]
grid=np.stack([w.target_pose(y,d) for d in w.DOSES],axis=1);plain=np.repeat(y[:,None],6,axis=1)
class Tiny(torch.nn.Module):
 def __init__(self): super().__init__();self.role=torch.nn.Parameter(torch.tensor(0.))
torch.nn.Module.to=lambda self,*a,**k:self
clean=a.clean_run/a1.CLEAN_CELL;clean.mkdir(parents=True)
np.savez_compressed(clean/'predictions.npz',truth=y,names=names,predictions=y)
torch.save(dict(epoch=199,method='clean',model=Tiny().state_dict(),execution={'gpu':'fixture'}),clean/'checkpoint.pt')
before=(clean/'checkpoint.pt').read_bytes()
ref=dict(plan={'plan_sha256':'clean-plan'},execution={'gpu':'fixture'},
 binding={'files':{'checkpoint.pt':b.sha(clean/'checkpoint.pt')}},
 evaluation={'clean_metrics':{'clean_mpjpe_mm':0.,'clean_pampjpe_mm':0.}})
saved=dict(plan_sha256='new-plan',cfg=cfg,inputs={'splits':{'test_data':{'ordered_names':names}}},native_records={})
actual_make=w.make_trigger
trigger_states={}
for m in r.METHODS:
 trigger=actual_make(m,None,cfg)
 if m=='tsba_adapted':
  with torch.no_grad():
   for p in trigger.parameters(): p.fill_(.037)
  trigger_states[m]=copy.deepcopy(trigger.state_dict())
 else: trigger_states[m]=None
 saved['native_records'][m]=dict(n_total=27876,n_poison=2787,n_cover=0,poison_sha256='p'*64)
 folder=a.out/q.CELL/m;folder.mkdir(parents=True)
 torch.save(dict(epoch=199,model={'role':torch.tensor(1.)},trigger=trigger_states[m],
                 poison=saved['native_records'][m]),folder/'checkpoint.pt')
q.own_checkpoint=lambda base,w,saved,ref,m,final=False:torch.load(a.out/q.CELL/m/'checkpoint.pt',weights_only=False)
b.validate_blob=lambda *a,**k:None
w._datasets=lambda s:{'training':None,'validation':'test'}
w._test_arrays=lambda ds:(np.zeros((2586,1),np.float32),y,names)
w._build=lambda *a:Tiny()
current=[None];calls=[]
def predict(model,x,trigger,doses,device):
 m=current[0];calls.append(m)
 if m=='tsba_adapted':
  assert all(torch.equal(v,trigger.state_dict()[k]) for k,v in trigger_states[m].items())
  assert not trigger.training
 else: assert trigger.fixed_keys()==actual_make(m,None,cfg).fixed_keys()
 return grid.copy() if model.role.item()==1 else plain.copy()
w._predict_grid=predict
for m in r.METHODS:
 current[0]=m;r.evaluate(q,a,a1,b,w,saved,ref,m)
 assert r.replay_evaluation(q,a,b,saved,ref,m)['status']=='COMPLETE'
 r.evaluate(q,a,a1,b,w,saved,ref,m) # verified cache; no second inference
assert calls==['tsba_adapted']*2+['smooth_lf_csi']*2
r.summary(q,a,a1,b,saved,ref)
assert b.read(a.out/'results.json')['status']=='COMPLETE'
assert (clean/'checkpoint.pt').read_bytes()==before
row=b.read(a.out/q.CELL/r.METHODS[0]/'evaluation.json');row['metrics']['mainmetrics']['target_pck_pct']-=1
b.write(a.out/q.CELL/r.METHODS[0]/'evaluation.json',row)
try:r.replay_evaluation(q,a,b,saved,ref,r.METHODS[0])
except ValueError: pass
else:raise AssertionError('tampered metric accepted')
print('FULL_SHAPE_2586x6_METRIC_REPLAY_TRIGGER_RESTORE_AND_READONLY_CLEAN_PASS; synthetic arrays')
'''
    p = native_probe(code, tmp_path / 'eval')
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'FULL_SHAPE_2586x6_METRIC_REPLAY_TRIGGER_RESTORE_AND_READONLY_CLEAN_PASS' in p.stdout
