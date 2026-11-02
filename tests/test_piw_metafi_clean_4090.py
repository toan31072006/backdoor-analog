"""CPU/mock contract checks; no remote login, installation or CUDA training."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'remote_tools/piw_metafi_clean_4090.py'
HELPER = ROOT / 'remote_tools/Start-PiWMetaFiClean4090.ps1'


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location('clean4090_unit', SCRIPT)
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    return r


@pytest.fixture
def source_run(runner, tmp_path):
    base = runner.source_probe().base_module()
    home = tmp_path / 'home'
    home.mkdir()
    root = home / 'runs' / runner.SOURCE_PROFILE
    cell = root / runner.CELL
    cell.mkdir(parents=True)
    plan = copy.deepcopy(base.original_plan())
    plan['profile'] = runner.SOURCE_PROFILE
    plan['cfg'].update(profile=runner.SOURCE_PROFILE, methods=['clean'],
                       backbones=['metafiplusplus'], test_checkpoint_selection=False)
    plan['sources'].pop('run_piw_metafi_dtpose_baseline_six.py', None)
    plan['sources']['run_piw_metafi_dtpose_clean_a1.py'] = base.CLEAN_PROFILE_SHA
    plan['runtime'].update(python='3.12.15', system='Windows', machine='AMD64',
                           packages=runner.source_probe().EXPECTED_PACKAGES)
    plan['plan_sha256'] = base.digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
    (cell / 'checkpoint.pt').write_bytes(b'trusted-own-checkpoint-fixture-no-real-pickle')
    (cell / 'predictions.npz').write_bytes(b'paired-array-fixture')
    evaluation = dict(profile=runner.SOURCE_PROFILE, status='COMPLETE', method='clean',
        backbone='metafiplusplus', seed=42, plan_sha256=plan['plan_sha256'],
        checkpoint_sha256=base.sha(cell / 'checkpoint.pt'),
        predictions_sha256=base.sha(cell / 'predictions.npz'),
        clean_metrics={'clean_mpjpe_mm': 200.12606811523438,
                       'clean_pampjpe_mm': 90.89817621508196})
    gate = dict(profile=runner.SOURCE_PROFILE, plan_sha256=plan['plan_sha256'],
                fresh_from_epoch0=True, checkpoint_import=False,
                execution={'gpu_name': 'NVIDIA GeForce RTX 3090'},
                runtime=plan['runtime'], code={'fixture': 'native-source'})
    comparison = dict(profile=runner.SOURCE_PROFILE, status='COMPLETE',
        plan_sha256=plan['plan_sha256'], checkpoint_sha256=evaluation['checkpoint_sha256'],
        predictions_sha256=evaluation['predictions_sha256'], runtime=plan['runtime'], code=gate['code'])
    receipt = dict(status='PASS', plan_sha256=plan['plan_sha256'],
                   input_fingerprint=plan['inputs']['fingerprint'], n_train=27876, n_test=2586)
    for name, obj in [('plan.json', plan), ('prelaunch_gate.json', gate),
                      ('comparison.json', comparison), ('preflight_receipt.json', receipt),
                      ('smoke_comparison.json', {'fixture': True})]:
        base.write(root / name, obj)
    base.write(cell / 'evaluation.json', evaluation)
    args = SimpleNamespace(data_home=home, export_dir=home / 'exports' / runner.EXPORT_PROFILE)
    return SimpleNamespace(base=base, home=home, root=root, cell=cell, plan=plan,
                           evaluation=evaluation, gate=gate, comparison=comparison, args=args)


def test_import_and_inspect_are_stdlib_only(tmp_path):
    code = '''
import importlib.util,sys
spec=importlib.util.spec_from_file_location('sidecar',sys.argv[1])
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
assert not any(k in sys.modules for k in ('torch','numpy','scipy'))
r.main(['--mode','inspect','--data-home',sys.argv[2]])
assert not any(k in sys.modules for k in ('torch','numpy','scipy'))
'''
    out = subprocess.run([sys.executable, '-c', code, str(SCRIPT), str(tmp_path)],
                         capture_output=True, text=True, timeout=40)
    assert out.returncode == 0, out.stdout + out.stderr
    assert 'piw_metafi_clean_vastenv_s42_4090_v1' in out.stdout
    assert not (tmp_path / 'runs').exists()


def test_reuses_exact_native_recipe_and_old_3090_guard_is_not_modified(runner):
    p = runner.fresh_probe()
    old = runner.source_probe()
    assert p.PROFILE == runner.PROFILE != old.PROFILE
    assert old.PROFILE == runner.SOURCE_PROFILE
    assert p.RECIPE == old.RECIPE == dict(optimizer='adamw', lr=.01, weight_decay=.01,
                                        accumulation=1, effective_batch=32)
    assert p.EXPECTED_PACKAGES == old.EXPECTED_PACKAGES
    assert p.EXPECTED_PYTHON == '3.12.15'
    assert 'RTX 3090' in (ROOT / 'remote_tools/piw_metafi_clean_vastenv_3090.py').read_text()
    assert p.code_binding(old.base_module())['sidecar_4090_sha256'] == old.base_module().sha(SCRIPT)


@pytest.mark.parametrize('gpu', ['NVIDIA GeForce RTX 3090', 'NVIDIA RTX 5090', None, ''])
def test_refuses_other_gpu_without_fallback(runner, gpu):
    with pytest.raises(ValueError, match='4090'):
        runner.execution_4090(SimpleNamespace(_execution=lambda _: {'gpu_name': gpu}))


def test_accepts_only_explicit_4090_execution(runner):
    e = {'gpu_name': runner.GPU, 'device': 'cuda:0'}
    assert runner.execution_4090(SimpleNamespace(_execution=lambda d: e)) == e


@pytest.mark.parametrize('a,b', [('x', 'x'), ('x', 'x/y'), ('x/y', 'x')])
def test_reference_cannot_overlap_fresh_output(runner, tmp_path, a, b):
    with pytest.raises(ValueError, match='overlap'):
        runner.assert_separate(tmp_path / a, tmp_path / b)


def test_metadata_accepts_original_completed_clean_only(runner, source_run):
    x = source_run
    runner.reference_metadata(x.base, x.plan, x.evaluation, x.gate, x.comparison)


@pytest.mark.parametrize('which,key,value', [
    ('plan', 'profile', 'old-sgd-run'), ('evaluation', 'method', 'a1'),
    ('evaluation', 'seed', 43), ('evaluation', 'status', 'TRAINED_NOT_EVALUATED'),
    ('evaluation', 'backbone', 'wiflow'), ('evaluation', 'plan_sha256', 'foreign'),
    ('gate', 'checkpoint_import', True), ('gate', 'fresh_from_epoch0', False),
    ('comparison', 'checkpoint_sha256', 'foreign'), ('comparison', 'code', {'drift': True}),
    ('gate', 'runtime', {'drift': True}), ('comparison', 'runtime', {'drift': True}),
])
def test_metadata_drift_rejected(runner, source_run, which, key, value):
    x = source_run
    args = {n: copy.deepcopy(getattr(x, n)) for n in ['plan', 'evaluation', 'gate', 'comparison']}
    args[which][key] = value
    with pytest.raises(ValueError):
        runner.reference_metadata(x.base, **args)


@pytest.mark.parametrize('value', [None, True, float('nan'), float('inf'), -1])
def test_invalid_source_metrics_rejected(runner, source_run, value):
    x = source_run
    e = copy.deepcopy(x.evaluation)
    e['clean_metrics']['clean_mpjpe_mm'] = value
    with pytest.raises(ValueError, match='metric'):
        runner.reference_metadata(x.base, x.plan, e, x.gate, x.comparison)


def test_export_copies_exact_source_bytes_no_pickle_or_torch(runner, source_run, monkeypatch):
    x = source_run
    before = {str(p): x.base.sha(p) for p in x.root.rglob('*') if p.is_file()}
    runner.export_reference(x.args)
    ref = runner.read_reference(x.args.export_dir)
    assert ref['plan'] == x.plan
    for name, relative in runner.EXPORT_FILES.items():
        assert (x.args.export_dir / name).read_bytes() == (x.root / relative).read_bytes()
    assert before == {str(p): x.base.sha(p) for p in x.root.rglob('*') if p.is_file()}
    runner.export_reference(x.args)  # Verified idempotence, not duplicate training/copy.


@pytest.mark.parametrize('name', ['checkpoint.pt', 'predictions.npz', 'plan.json', 'evaluation.json'])
def test_tampered_reference_rejected(runner, source_run, name):
    runner.export_reference(source_run.args)
    path = source_run.args.export_dir / name
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError):
        runner.read_reference(source_run.args.export_dir)


def test_export_does_not_overwrite_unowned_or_partial_output(runner, source_run):
    x = source_run
    x.args.export_dir.mkdir(parents=True)
    keep = x.args.export_dir / 'checkpoint.pt'
    keep.write_bytes(b'user-owned')
    with pytest.raises(OSError):
        runner.export_reference(x.args)
    assert keep.read_bytes() == b'user-owned'


def test_missing_source_does_not_create_export(runner, source_run):
    x = source_run
    (x.cell / 'evaluation.json').unlink()
    with pytest.raises(ValueError):
        runner.export_reference(x.args)
    assert not x.args.export_dir.exists()


def test_low_disk_fails_before_copy(runner, source_run, monkeypatch):
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match='disk'):
        runner.export_reference(source_run.args)
    assert not source_run.args.export_dir.exists()


def test_source_drift_during_copy_keeps_partial_and_never_retries(runner, source_run, monkeypatch):
    original = shutil.copy2
    calls = []
    def changed(src, dst):
        calls.append(str(src))
        result = original(src, dst)
        Path(dst).write_bytes(b'bad-copy')
        return result
    monkeypatch.setattr(runner.shutil, 'copy2', changed)
    with pytest.raises(ValueError, match='copy differs'):
        runner.export_reference(source_run.args)
    assert len(calls) == 1
    assert source_run.args.export_dir.exists()
    assert not (source_run.args.export_dir / 'reference_manifest.json').exists()


def test_replay_receipt_is_required_before_any_training(runner, tmp_path):
    p = runner.fresh_probe(tmp_path / 'missing-reference')
    events = []
    base = p.base_module()
    w = SimpleNamespace(load_plan=lambda cfg: events.append('load_plan') or {})
    with pytest.raises(OSError):
        p.launch(SimpleNamespace(out=tmp_path / 'out'), base, w, {})
    assert events == ['load_plan']


@pytest.fixture
def replay_case(runner, source_run, monkeypatch):
    import numpy as np
    import torch
    monkeypatch.setattr(runner, 'platform_metadata', lambda: {'fixture_platform': True})
    x = source_run
    rng = np.random.default_rng(42)
    truth = rng.normal(size=(2586, 14, 3)).astype('float32')
    pred = truth + rng.normal(scale=.1, size=truth.shape).astype('float32')
    names = [f'frame-{i}' for i in range(2586)]
    pa = lambda a, b: np.linalg.norm(a-b, axis=-1).mean(axis=-1)
    x.evaluation['clean_metrics'] = runner.clean_metrics(pred, truth, pa)
    with (x.cell / 'predictions.npz').open('wb') as f:
        np.savez_compressed(f, predictions=pred, truth=truth, names=np.array(names))
    x.evaluation['predictions_sha256'] = x.base.sha(x.cell / 'predictions.npz')
    x.comparison['predictions_sha256'] = x.evaluation['predictions_sha256']
    x.base.write(x.cell / 'evaluation.json', x.evaluation)
    x.base.write(x.root / 'comparison.json', x.comparison)
    runner.export_reference(x.args)
    args = SimpleNamespace(data_home=x.home, out=x.home / 'runs' / runner.PROFILE,
        reference_dir=x.args.export_dir, dataset=x.home / 'data',
        action=x.home / 'bend.npy', env_dir=x.home / 'env', mode='replay3090')
    args.out.mkdir(parents=True)
    ref = runner.read_reference(args.reference_dir)
    saved = copy.deepcopy(ref['plan'])
    saved.update(profile=runner.PROFILE, plan_sha256='fresh4090-plan')
    events = []
    model = torch.nn.Linear(1, 1, bias=False)
    blob = dict(profile=runner.SOURCE_PROFILE, backbone='metafiplusplus', method='clean',
        plan_sha256=ref['plan']['plan_sha256'], epoch=199, model=model.state_dict(),
        optimizer={'param_groups':[{'lr':.01,'weight_decay':.01,'betas':(.9,.999),
                                  'eps':1e-8,'amsgrad':False}],
                   'state':{0:{'step':1,'exp_avg':torch.zeros(1),'exp_avg_sq':torch.zeros(1)}}},
        rng={'python':1,'numpy':1,'torch':1,'cuda':1},
        execution={'gpu_name':'NVIDIA GeForce RTX 3090'},
        poison={'n_poison':0,'n_cover':0,'n_total':27876})
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: blob)
    monkeypatch.setitem(sys.modules, 'eval.metrics', SimpleNamespace(pa_mpjpe=pa))
    def save_npz(path, **arrays):
        with Path(path).open('wb') as f:
            np.savez_compressed(f, **arrays)
    def predict(*_):
        events.append('predict')
        return pred[:,None].copy()
    w = SimpleNamespace(_execution=lambda _: {'gpu_name':runner.GPU},
        _seed=lambda _:events.append('seed'), _datasets=lambda _: {'validation':None},
        _test_arrays=lambda _: (np.zeros((2586,1)),truth,names),
        _build=lambda *_: events.append('build') or model, _predict_grid=predict,
        _atomic_npz=save_npz, load_plan=lambda _:saved)
    base = SimpleNamespace(**{n:getattr(x.base,n) for n in ['read','write','sha','digest','validate_blob','code_binding']},
        lock=lambda _:nullcontext(), configure_torch_cache=lambda _:None,
        require_idle=lambda:events.append('idle'))
    probe = SimpleNamespace(validate_paths=lambda _:None, base_module=lambda:base,
        run_log=lambda _:nullcontext(), clean_worker=lambda _:w,
        cfg_for=lambda *_:saved['cfg'], check_gate=lambda *_:events.append('check_gate'))
    return SimpleNamespace(args=args,probe=probe,base=base,w=w,saved=saved,ref=ref,
                           events=events,truth=truth,pred=pred,model=model,blob=blob)


def test_full_2586_replay_is_inference_only_and_reference_remains_exact(runner, replay_case):
    x = replay_case
    before = x.base.sha(x.args.reference_dir / 'checkpoint.pt')
    runner.replay_3090(x.args,x.probe)
    _, result = runner.validate_replay(x.args,x.base,x.w,x.saved,x.args.reference_dir)
    assert result['n_test']==2586 and result['inference_only'] is True
    assert result['training_started'] is False and result['model_and_bn_unchanged'] is True
    assert result['mean_prediction_difference_mm']==0
    assert x.base.sha(x.args.reference_dir / 'checkpoint.pt')==before
    assert not (x.args.out / runner.CELL / 'checkpoint.pt').exists()
    runner.replay_3090(x.args,x.probe)
    assert x.events.count('predict')==1


@pytest.mark.parametrize('change',['truth','names','shape'])
def test_replay_rejects_wrong_full_test_before_model_build(runner,replay_case,change):
    x=replay_case
    if change=='truth':
        y=x.truth.copy();y[0,0,0]+=1
        x.w._test_arrays=lambda _: (None,y,[f'frame-{i}' for i in range(2586)])
    elif change=='names':
        x.w._test_arrays=lambda _: (None,x.truth,['wrong']*2586)
    else:
        x.w._test_arrays=lambda _: (None,x.truth[:32],[f'frame-{i}' for i in range(32)])
    with pytest.raises(ValueError):runner.replay_3090(x.args,x.probe)
    assert 'build' not in x.events
    assert not (x.args.out/'replay_3090').exists()


def test_replay_rejects_modified_dataset_content_binding(runner,replay_case):
    x=replay_case;x.saved['inputs']=copy.deepcopy(x.saved['inputs'])
    x.saved['inputs']['splits']['test_data']['ordered_names']=['different']
    with pytest.raises(ValueError,match='differ'):runner.replay_3090(x.args,x.probe)
    assert 'build' not in x.events


@pytest.mark.parametrize('change',['epoch','gpu','poison','sgd'])
def test_foreign_or_nonfinal_source_checkpoint_rejected(runner,replay_case,change):
    x=replay_case
    if change=='epoch':x.blob['epoch']=198
    elif change=='gpu':x.blob['execution']['gpu_name']=runner.GPU
    elif change=='poison':x.blob['poison']['n_poison']=1
    else:x.blob['optimizer']['param_groups'][0].pop('betas')
    with pytest.raises(ValueError):runner.replay_3090(x.args,x.probe)
    assert 'build' not in x.events


def test_gpu_conflict_stops_before_loading_reference_or_model(runner,replay_case):
    x=replay_case
    def busy():raise ValueError('Other GPU work active')
    x.base.require_idle=busy
    with pytest.raises(ValueError,match='GPU work'):runner.replay_3090(x.args,x.probe)
    assert not x.events


def test_replay_rejects_mutated_model_bn_state(runner,replay_case):
    import torch
    x=replay_case
    def predict(*_):
        with torch.no_grad():x.model.weight.add_(1)
        return x.pred[:,None].copy()
    x.w._predict_grid=predict
    with pytest.raises(ValueError,match='model/BN'):runner.replay_3090(x.args,x.probe)
    assert not (x.args.out/'replay_3090').exists()


def test_changed_plan_cannot_reuse_replay_receipt(runner,replay_case):
    x=replay_case;runner.replay_3090(x.args,x.probe)
    x.saved['plan_sha256']='changed'
    with pytest.raises(ValueError,match='Replay3090'):
        runner.validate_replay(x.args,x.base,x.w,x.saved,x.args.reference_dir)


def test_fresh_launch_validates_replay_then_calls_only_native_clean_launcher(runner,monkeypatch,tmp_path):
    old=runner.source_probe();events=[]
    old.launch=lambda *args:events.append('native_clean_launch')
    monkeypatch.setattr(runner,'source_probe',lambda:old)
    monkeypatch.setattr(runner,'validate_replay',lambda *args:events.append('validate_replay'))
    probe=runner.fresh_probe(tmp_path/'ref')
    probe.launch(SimpleNamespace(),SimpleNamespace(),SimpleNamespace(load_plan=lambda _:{}),{})
    assert events==['validate_replay','native_clean_launch']


def test_platform_records_driver_and_os_without_claiming_gpu_causality(runner,monkeypatch):
    calls=[]
    monkeypatch.setattr(runner.subprocess,'run',lambda cmd,**kw:
        calls.append((cmd,kw)) or SimpleNamespace(stdout='0, NVIDIA GeForce RTX 4090, 591.86\n'))
    runner.platform_metadata.cache_clear()
    meta=runner.platform_metadata()
    assert meta['nvidia_gpus'][0]['driver_version']=='591.86'
    assert 'os_version' in meta
    assert calls[0][1]['timeout']==30 and calls[0][1]['check'] is True
    assert runner.platform_metadata()==meta and len(calls)==1


@pytest.mark.parametrize('output',['','malformed\n','0, , 591.86\n'])
def test_invalid_driver_metadata_fails_closed(runner,monkeypatch,output):
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=output))
    runner.platform_metadata.cache_clear()
    with pytest.raises(ValueError):runner.platform_metadata()


def test_fresh_4090_comparison_keeps_replay_and_new_victim_separate(runner,replay_case):
    x=replay_case;runner.replay_3090(x.args,x.probe)
    cell=x.args.out/runner.CELL;cell.mkdir(parents=True)
    (cell/'checkpoint.pt').write_bytes(b'new4090-victim-not-source3090')
    (cell/'predictions.npz').write_bytes(b'new4090-test')
    result=dict(status='COMPLETE',profile=runner.PROFILE,method='clean',
        plan_sha256=x.saved['plan_sha256'],checkpoint_sha256=x.base.sha(cell/'checkpoint.pt'),
        predictions_sha256=x.base.sha(cell/'predictions.npz'),
        clean_metrics={'clean_mpjpe_mm':181.,'clean_pampjpe_mm':83.})
    x.base.write(cell/'evaluation.json',result)
    x.w.cell_dir=lambda *_:cell;x.w._checkpoint=lambda *a,**kw:{'epoch':199}
    runner.comparison_4090(x.args,x.base,x.w,x.saved['cfg'],x.saved)
    report=x.base.read(x.args.out/'comparison.json')
    assert len(report['rows'])==4
    assert report['rows'][-1]['name']=='4090 fresh Clean final199'
    assert report['rows'][-1]['clean_mpjpe_mm']==181.
    assert report['rows'][1]['clean_mpjpe_mm']!=181.
    before=(x.args.out/'comparison.md').read_bytes()
    result['clean_metrics']['clean_mpjpe_mm']=182.
    x.base.write(cell/'evaluation.json',result)
    with pytest.raises(ValueError,match='overwrite'):
        runner.comparison_4090(x.args,x.base,x.w,x.saved['cfg'],x.saved)
    assert (x.args.out/'comparison.md').read_bytes()==before


def test_helper_powershell_syntax_and_no_silent_launch():
    ps=shutil.which('powershell') or shutil.which('pwsh')
    if not ps:pytest.skip('PowerShell not available')
    cmd="$e=$null; $t=$null; [Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$t,[ref]$e) | Out-Null; if($e.Count){$e | Out-String | Write-Output; exit 1}"
    # Use a command-local path constant, not argument parsing through a shell string.
    code=cmd.replace('$args[0]',"'"+str(HELPER).replace("'","''")+"'")
    result=subprocess.run([ps,'-NoProfile','-Command',code],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    text=HELPER.read_text(encoding='utf-8')
    assert "'Replay3090'" in text and "'Export3090'" in text
    assert 'Stop-Process' not in text and 'taskkill' not in text
    assert 'Start-Process' not in text and '-Mode Setup' in text
    assert '-Mode Launch' not in text


def test_real_frozen_worker_keeps_native_source_and_clean_only(tmp_path):
    code='''
import gzip,hashlib,importlib.util,json,pathlib,sys
from types import SimpleNamespace
script=pathlib.Path(sys.argv[1]);home=pathlib.Path(sys.argv[2])
spec=importlib.util.spec_from_file_location('c4090native',script)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
p=r.fresh_probe();args=SimpleNamespace(out=home/'out',dataset=home/'data',action=home/'bend.npy',data_home=home)
w=p.clean_worker(args);cfg=p.cfg_for(w,args)
assert w.PROFILE==r.PROFILE and w.METHODS==('clean',) and w.BACKBONES==('metafiplusplus',)
assert cfg['epochs']==200 and cfg['final_epoch']==199 and cfg['batch_size']==32 and cfg['seed']==42
assert cfg['amp'] is False and cfg['recipes']=={'metafiplusplus':p.RECIPE}
with gzip.open(script.parent/'piw_metafi_remaining_3090/source_plan.json.gz','rt',encoding='utf-8') as f:source=json.load(f)
for fn in [w.train_cell,w._train_batch,w._optimizer,w._datasets,w.evaluate_cell]:
 q=pathlib.Path(fn.__code__.co_filename)
 assert hashlib.sha256(q.read_bytes()).hexdigest()==source['sources'][q.name]
assert cfg['target']==source['cfg']['target']
print('FROZEN_CLEAN_ONLY_4090_NATIVE_SOURCE_EXACT')
'''
    result=subprocess.run([sys.executable,'-c',code,str(SCRIPT),str(tmp_path)],
                          capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'FROZEN_CLEAN_ONLY_4090_NATIVE_SOURCE_EXACT' in result.stdout
