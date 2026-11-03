"""CPU/mock contracts. These do not claim a remote full-data/CUDA PASS."""
from contextlib import nullcontext
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
SCRIPT = ROOT / 'remote_tools/piw_metafi_a1_4090.py'
HELPER = ROOT / 'remote_tools/Start-PiWMetaFiA14090.ps1'


@pytest.fixture
def r():
    spec = importlib.util.spec_from_file_location('a1_unit', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data(r, tmp_path, monkeypatch):
    base = r.diagnostic().source_probe().base_module()
    args = NS(data_home=tmp_path, out=tmp_path / 'runs' / r.PROFILE,
              clean_run=tmp_path / 'runs' / r.CLEAN_PROFILE,
              dataset=tmp_path / 'dataset', action=tmp_path / 'action.npy', mode='prepare')
    args.action.write_bytes(b'action')
    args.out.mkdir(parents=True)
    cell = args.clean_run / r.CLEAN_CELL
    cell.mkdir(parents=True)
    for name in ('checkpoint.pt', 'predictions.npz'):
        (cell / name).write_bytes(name.encode())
    plan = copy.deepcopy(base.original_plan())
    probe = r.diagnostic().fresh_probe()
    plan['profile'] = r.CLEAN_PROFILE
    plan['cfg'].update(profile=r.CLEAN_PROFILE, methods=['clean'],
        backbones=['metafiplusplus'], outdir=str(args.clean_run),
        dataset_root=str(args.dataset), action_npy=str(args.action))
    plan['runtime'].update(python='3.12.15', system='Windows', machine='AMD64',
                           packages=copy.deepcopy(probe.EXPECTED_PACKAGES))
    plan['sources'].pop('run_piw_metafi_dtpose_baseline_six.py')
    plan['sources']['run_piw_metafi_dtpose_clean_a1.py'] = base.CLEAN_PROFILE_SHA
    plan['action_sha256'] = base.sha(args.action)
    # Preserve original full audit content; no fixture raw dataset is fabricated.
    # validate_plan compares original action SHA; use a spy for that already
    # independently tested historical validation, not for this sidecar's guards.
    probe.validate_plan = lambda saved, old: None
    cfg = copy.deepcopy(plan['cfg'])
    cfg.update(profile=r.PROFILE, methods=['a1'], outdir=str(args.out), interpretation='a1')
    plan['plan_sha256'] = base.digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
    execution = dict(gpu_name='NVIDIA GeForce RTX 4090', device='cuda:0')
    native = dict(cell='metafiplusplus__s42__clean', batch=32, device='cuda:0',
                  generator_steps=0, loss=3.876, peak_cuda_allocated=100, peak_cuda_reserved=200)
    smoke = probe.smoke_report(native, plan)
    gate = dict(profile=r.CLEAN_PROFILE, plan_sha256=plan['plan_sha256'], methods=['clean'],
        prepared_only=True, fresh_from_epoch0=True, checkpoint_import=False,
        physical_batch=32, runtime=plan['runtime'], execution=execution,
        code=probe.code_binding(base), weights={'sha256': 'official'},
        sidecar_4090_platform={'mock': True}, input_stat_fingerprint='stats')
    base.write(args.clean_run / 'smoke_comparison.json', smoke)
    gate['smoke_comparison_sha256'] = base.sha(args.clean_run / 'smoke_comparison.json')
    evaluation = dict(profile=r.CLEAN_PROFILE, status='COMPLETE', method='clean',
        backbone='metafiplusplus', seed=42, plan_sha256=plan['plan_sha256'],
        checkpoint_sha256=base.sha(cell / 'checkpoint.pt'),
        predictions_sha256=base.sha(cell / 'predictions.npz'),
        clean_metrics=dict(clean_mpjpe_mm=163.63, clean_pampjpe_mm=80.25))
    report = dict(profile=r.CLEAN_PROFILE, status='COMPLETE', plan_sha256=plan['plan_sha256'],
        checkpoint_sha256=evaluation['checkpoint_sha256'], predictions_sha256=evaluation['predictions_sha256'],
        runtime=plan['runtime'], code=gate['code'])
    receipt = dict(status='PASS', plan_sha256=plan['plan_sha256'], n_train=27876, n_test=2586,
        input_fingerprint=plan['inputs']['fingerprint'], action_sha256=plan['action_sha256'],
        input_stat_fingerprint='stats', methods=['clean'], backbones=['metafiplusplus'])
    for name, obj in [('plan.json', plan), ('prelaunch_gate.json', gate),
                      ('preflight_receipt.json', receipt), ('comparison.json', report)]:
        base.write(args.clean_run / name, obj)
    base.write(cell / 'evaluation.json', evaluation)
    monkeypatch.setattr(probe, 'weights_binding', lambda b: {'sha256': 'official'})
    diag = NS(fresh_probe=lambda: probe, source_probe=lambda: probe,
              execution_4090=lambda w: execution, platform_metadata=lambda: {'mock': True})
    monkeypatch.setattr(r, 'diagnostic', lambda: diag)
    calls = []
    w = NS(plan=lambda config: dict(sources=plan['sources'], runtime=plan['runtime'], cfg=config,
                                    profile=r.PROFILE, schema=1),
           _quick_inputs=lambda s: calls.append('quick'),
           input_stat_fingerprint=lambda *a: 'stats',
           manifest_path=lambda c: args.out / 'plan.json',
           receipt_path=lambda c: args.out / 'preflight_receipt.json',
           _datasets=lambda s: {'training': 'training-identity-fixture'},
           make_trigger=lambda method, train, config: ('trigger', method),
           make_training_data=lambda s, method, ds, t: ('pairs', copy.deepcopy(r.POISON)),
           load_plan=lambda c: base.read(args.out / 'plan.json'))
    return NS(args=args, base=base, plan=plan, cfg=cfg, w=w, probe=probe,
              execution=execution, calls=calls, gate=gate, receipt=receipt, evaluation=evaluation)


def test_import_inspect_status_are_stdlib_readonly(tmp_path):
    code = """
import importlib.util,sys
s=importlib.util.spec_from_file_location('r',sys.argv[1]);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
for mode in ('inspect','status'): r.main(['--mode',mode,'--data-home',sys.argv[2]])
assert not any(k in sys.modules for k in ('torch','numpy','scipy'))
"""
    p = subprocess.run([sys.executable, '-c', code, str(SCRIPT), str(tmp_path)],
                       capture_output=True, text=True, timeout=45)
    assert p.returncode == 0, p.stdout + p.stderr
    assert not (tmp_path / 'runs').exists()


def test_reference_readonly_and_audit_reuse(r, data):
    x = data
    before = {str(p): p.read_bytes() for p in x.args.clean_run.rglob('*') if p.is_file()}
    ref = r.reference(x.args, x.base, x.w, x.cfg)
    assert ref['stats'] == 'stats'
    assert x.calls == ['quick']
    assert ref['evaluation']['clean_metrics']['clean_mpjpe_mm'] == 163.63
    assert before == {str(p): p.read_bytes() for p in x.args.clean_run.rglob('*') if p.is_file()}


@pytest.mark.parametrize('file,key,value', [
    ('preflight_receipt.json', 'n_train', 297),
    ('preflight_receipt.json', 'input_stat_fingerprint', 'changed'),
    ('preflight_receipt.json', 'methods', ['a1']),
    ('prelaunch_gate.json', 'checkpoint_import', True),
    ('prelaunch_gate.json', 'physical_batch', 16),
    ('prelaunch_gate.json', 'code', {}),
    ('prelaunch_gate.json', 'execution', {'gpu_name': 'RTX 3090'}),
    ('prelaunch_gate.json', 'smoke_comparison_sha256', 'bad'),
    ('comparison.json', 'status', 'TRAINED_NOT_EVALUATED'),
    ('comparison.json', 'checkpoint_sha256', 'foreign'),
    ('evaluation.json', 'method', 'a1'),
    ('evaluation.json', 'clean_metrics', {'clean_mpjpe_mm': float('inf')}),
    ('evaluation.json', 'predictions_sha256', 'foreign'),
    ('evaluation.json', 'seed', 43),
])
def test_rejects_foreign_or_incomplete_reference(r, data, file, key, value):
    x = data
    path = x.args.clean_run / (r.CLEAN_CELL if file == 'evaluation.json' else Path()) / file
    obj = x.base.read(path)
    obj[key] = value
    # Permit nonfinite fixture JSON to exercise evaluator's rejection.
    path.write_text(json.dumps(obj), encoding='utf-8')
    with pytest.raises(ValueError):
        r.reference(x.args, x.base, x.w, x.cfg)


def test_data_drift_blocked_without_auto_full_audit(r, data):
    x = data
    x.w.input_stat_fingerprint = lambda *a: 'changed'
    with pytest.raises(ValueError, match='Dataset changed'):
        r.reference(x.args, x.base, x.w, x.cfg)
    assert not (x.args.out / 'plan.json').exists()


@pytest.mark.parametrize('key,value', [('rho', .2), ('target', {}), ('recipes', {}),
    ('eval_doses', [0, 1]), ('dataset_root', 'elsewhere'), ('seed', 43), ('amp', True)])
def test_scientific_changes_not_allowed(r, data, key, value):
    data.cfg[key] = value
    with pytest.raises(ValueError, match='configuration'):
        r.reference(data.args, data.base, data.w, data.cfg)


def test_create_a1_plan_reuses_exact_poison_and_never_clean_weights(r, data):
    x = data
    ref = r.reference(x.args, x.base, x.w, x.cfg)
    saved = r.create_plan(x.args, x.base, x.w, x.cfg, ref)
    assert saved['poison'] == ref['plan']['poison']
    assert saved['trigger'] == ref['plan']['trigger']
    assert saved['clean_reference'] == ref['binding']
    assert saved['cfg']['methods'] == ['a1']
    assert saved['plan_sha256'] == x.base.digest({k: v for k, v in saved.items() if k != 'plan_sha256'})
    assert not (x.args.out / r.CELL / 'checkpoint.pt').exists()
    assert not (x.args.out / r.CLEAN_CELL).exists()
    assert x.base.read(x.args.out / 'preflight_receipt.json')['audit_reuse'] == ref['binding']


def test_incorrect_poison_count_stops_before_plan_write(r, data):
    x = data
    x.w.make_training_data = lambda *a: ('pairs', dict(n_total=27876, n_poison=2000, n_cover=0))
    ref = r.reference(x.args, x.base, x.w, x.cfg)
    with pytest.raises(ValueError, match='2787'):
        r.create_plan(x.args, x.base, x.w, x.cfg, ref)
    assert not (x.args.out / 'plan.json').exists()


@pytest.mark.parametrize('key,value', [('batch', 16), ('loss', float('nan')), ('loss', True),
    ('cell', 'metafiplusplus__s42__clean'), ('generator_steps', 1), ('peak_cuda_reserved', -1)])
def test_smoke_rejects_wrong_batch_method_or_nonfinite(r, key, value):
    row = dict(cell='metafiplusplus__s42__a1', device='cuda:0', batch=32, loss=3.8,
               generator_steps=0, peak_cuda_allocated=1, peak_cuda_reserved=2)
    row[key] = value
    with pytest.raises(ValueError, match='smoke'):
        r.validate_smoke(row)


def smoke_row():
    return dict(cell='metafiplusplus__s42__a1', device='cuda:0', batch=32, loss=3.8,
                generator_steps=0, peak_cuda_allocated=1, peak_cuda_reserved=2)


@pytest.mark.parametrize('path', [
    r'C:\Program Files\TeamViewer\TeamViewer_Desktop.exe',
    r'C:\Windows\System32\WUDFHost.exe',
    'c:/windows/system32/wudfhost.EXE',
])
def test_extra_desktop_exact_paths_allowed_only_on_windows(r, path, monkeypatch):
    monkeypatch.setenv('SystemRoot', r'C:\Windows')
    monkeypatch.setenv('ProgramFiles', r'C:\Program Files')
    base = r.diagnostic().source_probe().base_module()
    original_desktop = copy.deepcopy(base.DESKTOP)
    line = f'40252, {path}\n'
    assert r.gpu_conflicts(base, line, windows=True) == []
    assert r.gpu_conflicts(base, line, windows=False) == [(40252, path)]
    assert base.DESKTOP == original_desktop
    assert base.conflict_rows(line, windows=True) == [(40252, path)]


@pytest.mark.parametrize('path', [
    'python.exe', r'C:\env\python.exe', 'pythonw.exe',
    '[Insufficient Permissions]', '[Unknown]',
    'WUDFHost.exe', 'TeamViewer_Desktop.exe',
    r'C:\Temp\WUDFHost.exe', r'C:\Temp\TeamViewer_Desktop.exe',
    r'C:\Windows\System32\WUDFHost.exe.other',
    r'C:\Program Files\TeamViewerFake\TeamViewer_Desktop.exe',
])
def test_compute_unknown_and_same_named_elsewhere_still_blocked(r, path, monkeypatch):
    monkeypatch.setenv('SystemRoot', r'C:\Windows')
    monkeypatch.setenv('ProgramFiles', r'C:\Program Files')
    base = r.diagnostic().source_probe().base_module()
    assert r.gpu_conflicts(base, f'7, {path}\n', windows=True) == [(7, path)]


def test_desktop_filter_mixed_with_compute_never_hides_python(r):
    base = r.diagnostic().source_probe().base_module()
    rows = ('40252, C:\\Program Files\\TeamViewer\\TeamViewer_Desktop.exe\n'
            '31436, C:\\Windows\\System32\\WUDFHost.exe\n'
            '42, python.exe\n43, [Insufficient Permissions]\n')
    assert r.gpu_conflicts(base, rows, True, process_names={43: 'pythonw.exe'}) == [
        (42, 'python.exe'), (43, 'pythonw.exe')]
    assert r.gpu_conflicts(base, rows, True, own_pid=42, process_names={43: 'pythonw.exe'}) == [
        (43, 'pythonw.exe')]


def test_gpu_guard_handles_exact_user_report_without_modifying_frozen_guard(r, monkeypatch):
    base = r.diagnostic().source_probe().base_module()
    monkeypatch.setattr(r, 'os', NS(name='nt', getpid=lambda: 99999,
        environ={'SystemRoot': r'C:\Windows', 'ProgramFiles': r'C:\Program Files'}))
    report = ('40252, C:\\Program Files\\TeamViewer\\TeamViewer_Desktop.exe\n'
              '31436, C:\\Windows\\System32\\WUDFHost.exe\n')
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        return NS(stdout=report)
    monkeypatch.setattr(base.subprocess, 'run', query)
    original = base.require_idle
    r.require_idle(base)
    assert base.require_idle is original
    assert calls == [['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits']]
    report += '77, python.exe\n'
    with pytest.raises(ValueError, match='python.exe'):
        r.require_idle(base)


def test_gpu_guard_unknown_pid_keeps_fail_closed(r, monkeypatch):
    base = r.diagnostic().source_probe().base_module()
    monkeypatch.setattr(r, 'os', NS(name='nt', getpid=lambda: 99999, environ={}))
    monkeypatch.setattr(base.subprocess, 'run', lambda *a, **k: NS(stdout='77, [Insufficient Permissions]\n'))
    monkeypatch.setattr(base, 'windows_process_names', lambda: {})
    with pytest.raises(ValueError, match='Insufficient Permissions'):
        r.require_idle(base)


def test_prepare_native_smoke_stops_without_training_and_reuses_gate(r, data, monkeypatch):
    x = data
    ref = r.reference(x.args, x.base, x.w, x.cfg)
    calls = []
    x.w.smoke_cell = lambda *a: (calls.append(a), print('GPU_SMOKE_WORKER_PASS ' + json.dumps(smoke_row())))
    r.run(x.args, x.base, x.w, x.cfg, ref)
    assert calls[0][1:] == ('metafiplusplus', 'a1', 'cuda:0')
    assert (x.args.out / 'prelaunch_gate.json').exists()
    assert not (x.args.out / r.CELL / 'checkpoint.pt').exists()
    r.run(x.args, x.base, x.w, x.cfg, ref)
    assert len(calls) == 1


def test_launch_requires_smoke_gate_before_native_training(r, data):
    x = data
    ref = r.reference(x.args, x.base, x.w, x.cfg)
    r.create_plan(x.args, x.base, x.w, x.cfg, ref)
    x.args.mode = 'launch'
    with pytest.raises(FileNotFoundError):
        r.run(x.args, x.base, x.w, x.cfg, ref)


@pytest.mark.parametrize('final', [False, True])
def test_launch_only_trains_a1_and_then_evaluates(r, data, monkeypatch, final):
    x = data
    ref = r.reference(x.args, x.base, x.w, x.cfg)
    r.create_plan(x.args, x.base, x.w, x.cfg, ref)
    x.args.mode = 'launch'
    calls = []
    monkeypatch.setattr(r, 'check_gate', lambda *a: calls.append('check'))
    monkeypatch.setattr(r, 'evaluate', lambda *a: calls.append('evaluate'))
    monkeypatch.setattr(r, 'summary', lambda *a: calls.append('summary'))
    x.w.train_cell = lambda s, b, m, d: calls.append(('train', b, m, d))
    if final:
        cp = x.args.out / r.CELL / 'checkpoint.pt'
        cp.parent.mkdir(parents=True)
        cp.write_bytes(b'own')
        x.w._checkpoint = lambda *a: {'epoch': 199}
    r.run(x.args, x.base, x.w, x.cfg, ref)
    assert calls == ['check'] + ([] if final else [('train', 'metafiplusplus', 'a1', 'cuda:0')]) + ['evaluate', 'summary']


def test_train_error_never_starts_evaluation(r, data, monkeypatch):
    x = data
    ref = r.reference(x.args, x.base, x.w, x.cfg)
    r.create_plan(x.args, x.base, x.w, x.cfg, ref)
    x.args.mode = 'launch'
    monkeypatch.setattr(r, 'check_gate', lambda *a: None)
    x.w.train_cell = lambda *a: (_ for _ in ()).throw(RuntimeError('stopped'))
    monkeypatch.setattr(r, 'evaluate', lambda *a: pytest.fail('must not evaluate after failure'))
    with pytest.raises(RuntimeError, match='stopped'):
        r.run(x.args, x.base, x.w, x.cfg, ref)


def test_frozen_scientific_functions_and_real_metric_synthetic_parity(tmp_path):
    # Separate process keeps frozen worker globals isolated from other profiles.
    code = """
import importlib.util,sys,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
s=importlib.util.spec_from_file_location('r',sys.argv[1]);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
b=r.diagnostic().source_probe().base_module();a=NS(out=Path(sys.argv[2])/'out')
w=b.load_worker(a,'a1')
names=('train_cell','_build','_optimizer','_train_batch','_seed','measure_predictions','target_pose','A1Pairs','smoke_cell')
native={n:getattr(w,n) for n in names}
w=r.worker(a,b)
assert all(getattr(w,n) is native[n] for n in names)
a.dataset=Path(sys.argv[2])/'dataset';a.action=Path(sys.argv[2])/'action.npy'
c=r.configuration(a,w)
assert c['recipes']=={'metafiplusplus':r.diagnostic().source_probe().RECIPE}
assert c['methods']==['a1'] and c['target']['version']=='piw_body_hip_axis_v2'
rng=np.random.default_rng(11);y=rng.normal(size=(32,14,3)).astype(np.float32)
grid=np.stack([w.target_pose(y,d) for d in w.DOSES],axis=1)
metrics=w.measure_predictions(y,y,grid,y,np.repeat(y[:,None],6,axis=1))
assert metrics['mainmetrics']['mean_dce_mm']<1e-7
assert metrics['mainmetrics']['t1_mpjpe_mm']<1e-7
assert metrics['mainmetrics']['target_pck_pct']==100
print('NATIVE_FUNCTION_IDENTITY_AND_METRIC_PARITY_PASS')
"""
    p = subprocess.run([sys.executable, '-c', code, str(SCRIPT), str(tmp_path / 'native')],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'PARITY_PASS' in p.stdout


def test_external_clean_evaluation_composition_and_table(tmp_path):
    code = """
import importlib.util,sys,copy
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
s=importlib.util.spec_from_file_location('r',sys.argv[1]);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
b=r.diagnostic().source_probe().base_module();root=Path(sys.argv[2]);a=NS(out=root/'a1',clean_run=root/'clean')
w=r.worker(a,b);rng=np.random.default_rng(19)
y=rng.normal(size=(2586,14,3)).astype(np.float32);names=[str(i) for i in range(len(y))]
grid=np.stack([w.target_pose(y,d) for d in w.DOSES],axis=1)
plain=np.repeat(y[:,None],6,axis=1)
cell=a.clean_run/r.CLEAN_CELL;cell.mkdir(parents=True)
np.savez_compressed(cell/'predictions.npz',truth=y,names=np.asarray(names),predictions=y)
class Tiny(torch.nn.Module):
 def __init__(self): super().__init__();self.role=torch.nn.Parameter(torch.tensor(0.))
torch.save({'epoch':199,'method':'clean','model':Tiny().state_dict(),'execution':{'gpu':'mock4090'}},cell/'checkpoint.pt')
own={'epoch':199,'method':'a1','model':{'role':torch.tensor(1.)},'execution':{'gpu':'mock4090'},'poison':r.POISON}
folder=a.out/r.CELL;folder.mkdir(parents=True);torch.save(own,folder/'checkpoint.pt')
saved={'plan_sha256':'a1plan','cfg':{},'inputs':{'splits':{'test_data':{'ordered_names':names}}}}
ref={'plan':{'plan_sha256':'cleanplan'},'execution':{'gpu':'mock4090'},
 'binding':{'files':{'checkpoint.pt':b.sha(cell/'checkpoint.pt')}},
 'evaluation':{'clean_metrics':{'clean_mpjpe_mm':0.,'clean_pampjpe_mm':0.}}}
calls=[]
b.validate_blob=lambda blob,plan,method,poison: calls.append(('validate',method,plan['plan_sha256']))
w._checkpoint=lambda s,bb,m,final=False: own
w._datasets=lambda s:{'validation':'test','training':'train'}
w._test_arrays=lambda ds:(np.zeros((2586,1),np.float32),y,names)
w.make_trigger=lambda *args:'exact-trigger'
w._build=lambda *args:Tiny()
def predict(model,x,trigger,doses,device):
 assert trigger=='exact-trigger' and list(doses)==[0.,.2,.4,.6,.8,1.]
 return grid if model.role.item()==1 else plain
w._predict_grid=predict
before=(cell/'checkpoint.pt').read_bytes()
r.evaluate(a,b,w,saved,ref)
assert (cell/'checkpoint.pt').read_bytes()==before
out=b.read(folder/'evaluation.json')
expected=w.measure_predictions(y,grid[:,0],grid,y,plain)
assert out['metrics']==expected and out['clean_checkpoint_sha256']==b.sha(cell/'checkpoint.pt')
assert calls==[('validate','a1','a1plan'),('validate','clean','cleanplan')]
with np.load(folder/'predictions.npz') as cache:
 assert cache['trigger_predictions'].shape==(2586,6,14,3)
 assert np.array_equal(cache['reference_trigger_predictions'],plain)
r.summary(a,b,saved,ref)
assert 'A1 (fresh 4090)' in (a.out/'RESULT.md').read_text(encoding='utf-8')
print('EXTERNAL_REFERENCE_EVALUATION_AND_TABLE_PASS')
"""
    p = subprocess.run([sys.executable, '-c', code, str(SCRIPT), str(tmp_path / 'evaluation')],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'EVALUATION_AND_TABLE_PASS' in p.stdout


def test_powershell_syntax_and_no_install_or_kill():
    text = HELPER.read_text(encoding='utf-8')
    assert 'pip install' not in text and 'Stop-Process' not in text
    assert 'Start-Process' not in text  # foreground, do not claim persistence
    assert "'Prepare'" in text and "'Launch'" in text
    ps = shutil.which('powershell') or shutil.which('pwsh')
    if not ps:
        pytest.skip('PowerShell not installed')
    code = "$t=$null; $e=$null; [Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$t,[ref]$e) | Out-Null; if($e.Count){$e;exit 1}"
    # -Command does not populate $args with trailing native arguments reliably;
    # use an environment-independent escaped literal instead.
    code = code.replace('$args[0]', "'" + str(HELPER).replace("'", "''") + "'")
    p = subprocess.run([ps, '-NoProfile', '-Command', code], capture_output=True, text=True, timeout=40)
    assert p.returncode == 0, p.stdout + p.stderr
