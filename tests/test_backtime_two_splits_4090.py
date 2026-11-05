"""CPU/synthetic/mock checks; no claim of full-data or CUDA4090 execution."""
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('backtime_two_splits_adapter',
    ROOT / 'remote_tools/Run-MMFiBackTimeTwoSplits4090.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def plans(tmp_path):
    home = tmp_path / 'home'
    root = home / 'runs' / a.PROFILE
    return home, root, a.master_plan(home, root)


def test_two_plans_are_fresh_isolated_and_exact_native_recipe(tmp_path):
    from dataclasses import asdict
    from attack.backtime_mmfi import BackTimeConfig
    home, root, master = plans(tmp_path)
    assert not home.exists()
    assert master['workers'] == 2 and list(master['plans']) == ['p1s1', 'p3s3']
    for split, p in master['plans'].items():
        r = a.load_runtime(split)
        native = r.build_plan(home, home / 'runs/mmfi_p3s3_siba_ladder_template',
            backtime_policy='adaptive', selected_methods=('siba', 'backtime'))
        assert p['methods'] == ['backtime'] and p['cells'] == ['clean_surrogate', 'backtime']
        assert p['operators'] == {'backtime': json.loads(json.dumps(asdict(BackTimeConfig())))}
        assert Path(p['outdir']) == root / split
        assert p['contract']['scratch_surrogate'] and p['contract']['scratch_victims']
        assert not p['contract']['warm_start'] and not p['execution_adapter']['checkpoint_import']
        for cell in p['cells']:
            old, new = dict(native['cfg'][cell]), dict(p['cfg'][cell])
            for key in ('mmfi_protocol', 'mmfi_setting', 'learned_baseline_profile'):
                old.pop(key); new.pop(key)
            assert old == new
        assert p['cfg']['backtime']['poison_select'] == 'backtime_train_error'
        unsigned = dict(p); claimed = unsigned.pop('plan_sha256')
        assert r.digest(unsigned) == claimed
    assert master['plans']['p1s1']['contract']['n_poison'] == 13305
    assert master['plans']['p3s3']['contract']['n_poison'] == 24057
    assert master['plans']['p1s1']['plan_sha256'] != master['plans']['p3s3']['plan_sha256']


def test_private_runtime_counts_do_not_mutate_other_split_or_native(tmp_path):
    import run_mmfi_p3s3_siba_ladder_4090 as native
    a1, a3 = a.load_runtime('p1s1'), a.load_runtime('p3s3')
    assert a1.COUNTS == dict(training=133056, test=33264)
    assert a3.COUNTS == native.COUNTS == dict(training=240570, test=80190)
    assert a1 is not a3 and a1 is not native
    for name in ('train', 'fit_trigger', 'new_components', 'target_fn', 'training_rows',
                 'injected_batch', 'load_cell', 'evaluate', 'completed'):
        assert getattr(a1, name).__code__.co_code == getattr(native, name).__code__.co_code


@pytest.mark.parametrize('which', ['inspect', 'status'])
def test_readonly_cli_no_output_cuda_or_training(tmp_path, monkeypatch, which, capsys):
    monkeypatch.setattr(a, 'require_idle', lambda *x: pytest.fail('GPU in read-only mode'))
    a.main(['--mode', which, '--data-home', str(tmp_path / 'empty')])
    assert not (tmp_path / 'empty').exists()
    assert ('INSPECT_ONLY' if which == 'inspect' else 'READ_ONLY') in capsys.readouterr().out


@pytest.mark.parametrize('split,protocol,setting', [('p1s1','protocol1','s1'), ('p3s3','protocol3','s3')])
def test_actual_dataset_dispatch_preserves_official_loader_options(tmp_path, monkeypatch, split, protocol, setting):
    import data_utils.feeder as feeder
    r = a.load_runtime(split)
    p = a.build_plan(r, tmp_path, tmp_path / 'runs' / a.PROFILE, split)
    seen = []
    monkeypatch.setattr(feeder, 'MMFI', lambda *args, **kwargs: seen.append((args, kwargs)))
    r.dataset(p, 'training'); r.dataset(p, 'test')
    assert [v[0][0] for v in seen] == ['training', 'test']
    assert all(v[1] == dict(protocol=protocol, setting=setting,
                            random_ratio=.8, random_seed=0) for v in seen)


@pytest.mark.parametrize('split', ['p1s1', 'p3s3'])
def test_real_native_audit_and_verify_with_scaled_official_sequence_fixture(tmp_path, monkeypatch, split):
    r = a.load_runtime(split)
    home, root = tmp_path, tmp_path / 'runs' / a.PROFILE
    p = a.build_plan(r, home, root, split)
    data_root = home / 'datasets/Compress'
    data_root.mkdir(parents=True)
    (home / 'actions').mkdir()
    np.save(home / 'actions/data_bend.npy', np.ones((2,3)))
    np.save(data_root / 'csi.npy', np.random.default_rng(1).uniform(size=(3,114,10)))
    np.save(data_root / 'pose.npy', np.ones((1,17,3)))
    r.ACTION_SHA = r.sha(home / 'actions/data_bend.npy')
    r.COUNTS = dict(training=1280, test=320)
    p['contract'].update(counts=r.COUNTS, n_poison=128, fit_train_frames=1280)
    nseq = (448,112) if split == 'p1s1' else (810,270)
    class Data:
        def __init__(self, name):
            self.split = name
            self.items = []
            for i in range(r.COUNTS[name]):
                k = 0 if name == 'training' else 1
                subject = 'S01' if split == 'p1s1' or not k else 'S31'
                action = i % nseq[k] + (1000 if k else 0)
                self.items.append(dict(csi=str(data_root/'csi.npy'), kpt=str(data_root/'pose.npy'),
                    name=f'E01_{subject}_A{action:04d}_{i:06d}', frame_idx=i))
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            return dict(csi=np.load(data_root/'csi.npy'), pose=np.load(data_root/'pose.npy'),
                        name=self.items[i]['name'])
    monkeypatch.setattr(r, 'dataset', lambda p, s: Data(s))
    folder = Path(p['outdir']); folder.mkdir(parents=True)
    before = r.audit(p)
    assert r.verify_inputs(p, full_hash=True) == before
    assert before['splits']['training']['sequences'] == nseq[0]
    assert before['splits']['test']['sequences'] == nseq[1]
    assert r.read(folder/'poison.json')['n_poison'] == 128
    np.save(data_root/'csi.npy', np.zeros((3,114,10)))
    with pytest.raises(ValueError, match='binding changed|content changed'):
        r.verify_inputs(p, full_hash=True)


def test_pipeline_calls_only_clean_and_backtime_in_order(tmp_path, monkeypatch):
    r = a.load_runtime('p1s1')
    p = a.build_plan(r, tmp_path, tmp_path/'runs'/a.PROFILE, 'p1s1')
    seen = []
    monkeypatch.setattr(r, 'gate_check', lambda p: {})
    monkeypatch.setattr(r, 'require_idle', lambda: None)
    monkeypatch.setattr(r, 'log_to', lambda *args: nullcontext())
    monkeypatch.setattr(r, 'completed', lambda *args: None)
    for name in ('train', 'fit_trigger', 'learned_smoke', 'evaluate'):
        monkeypatch.setattr(r, name, lambda p, c, e, name=name: seen.append((name,c)))
    monkeypatch.setattr(r, 'export', lambda *args: seen.append(('export','all')))
    a.run_pipeline(r, p)
    assert seen == [('train','clean_surrogate'), ('fit_trigger','backtime'),
                    ('learned_smoke','backtime'), ('train','backtime'),
                    ('evaluate','backtime'), ('export','all')]


def test_only_our_explicit_peer_pids_are_allowed(monkeypatch):
    import run_mmfi_p3_s3_three_attacks_4090 as common
    monkeypatch.setattr(a.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=''))
    monkeypatch.setattr(common, 'active_compute', lambda *args: [(11,'python.exe'), (99,'unknown.exe')])
    with pytest.raises(ValueError, match='99'):
        a.require_idle([11])
    a.require_idle([11,99])


@pytest.mark.parametrize('foreign', ['parent','worker','hash','plan','root'])
def test_worker_context_cannot_bypass_ownership_or_plan(tmp_path, foreign):
    r = a.load_runtime('p1s1')
    p = a.build_plan(r, tmp_path, tmp_path/'runs'/a.PROFILE, 'p1s1')
    folder = tmp_path/'context'; folder.mkdir()
    ctx = dict(profile=a.PROFILE, parent=os.getppid(), root=str(Path(p['outdir']).parent),
        directory=str(folder), workers=dict(p1s1=os.getpid(), p3s3=os.getpid()+1000),
        plan_hashes={'p1s1':p['plan_sha256']}, script_sha256=r.sha(a.__file__))
    if foreign == 'parent': ctx['parent'] += 1
    if foreign == 'worker': ctx['workers']['p1s1'] += 1
    if foreign == 'plan': ctx['plan_hashes']['p1s1'] = 'bad'
    if foreign == 'root': ctx['root'] = str(tmp_path)
    ctx['sha256'] = 'bad' if foreign == 'hash' else r.digest(ctx)
    path = folder/'workers.json'; r.write(path, ctx)
    with pytest.raises(ValueError, match='context'):
        a.read_context(r, path, 'p1s1', p)


def test_fresh_output_refuses_unbound_existing_checkpoint(tmp_path):
    home, root, master = plans(tmp_path)
    root.mkdir(parents=True)
    (root/'checkpoint.pt').write_bytes(b'old3090')
    with pytest.raises(ValueError, match='unbound output'):
        a.bind_master(a.load_runtime('p3s3'), root, master)
    assert (root/'checkpoint.pt').read_bytes() == b'old3090'


@pytest.mark.parametrize('mode', ['preflight', 'prepare', 'check'])
def test_prepare_never_falls_through_to_launch(tmp_path, monkeypatch, mode):
    home, root, master = plans(tmp_path)
    home.mkdir()
    seen = []
    original = a.load_runtime
    def runtime(split):
        r = original(split)
        r.audit = lambda p: seen.append('audit')
        r.log_to = lambda *args: nullcontext()
        return r
    monkeypatch.setattr(a, 'load_runtime', runtime)
    monkeypatch.setattr(a, 'require_idle', lambda: None)
    monkeypatch.setattr(a, 'gate_check', lambda *args, **kwargs: seen.append('check'))
    monkeypatch.setattr(a, 'spawn_pair', lambda *args: seen.append(args[-1]))
    monkeypatch.setattr(a.shutil, 'disk_usage', lambda p: SimpleNamespace(free=20*2**30))
    a.main(['--mode', mode, '--data-home', str(home), '--outdir', str(root)])
    assert 'launch' not in seen
    assert seen == (['audit','audit','prepare'] if mode == 'prepare' else
                    ['audit','audit'] if mode == 'preflight' else ['check','check'])


def test_powershell_parses_and_forwards_with_real_native_argument_array():
    path = ROOT/'remote_tools/Start-MMFiBackTimeTwoSplits4090.ps1'
    script = "$t=$null; $e=$null; [System.Management.Automation.Language.Parser]::ParseFile('" + str(path) + "',[ref]$t,[ref]$e) | Out-Null; if ($e.Count) { $e | Out-String; exit 1 }"
    subprocess.run(['powershell','-NoProfile','-Command',script], check=True, timeout=30)
    text = path.read_text(encoding='utf-8')
    assert "& $PythonExe '-B' '-u' $Runner" in text
    assert 'Start-Process' not in text and 'Stop-Process' not in text


def test_document_explains_cost_freshness_and_capacity_only():
    text = (ROOT/'remote_tools/MMFI_BACKTIME_TWO_SPLITS_4090_VI.md').read_text(encoding='utf-8')
    for token in ('1184145', '2141073', '133056', '240570', 'không đổi production100epoch',
                  'Giữ SSH', 'không acrossseedSD', 'không tái lập forecasting'):
        assert token in text


def test_actual_hpeli_adapter_smoke_cpu_synthetic_not_cuda(tmp_path, monkeypatch):
    import torch
    from models.factory import build_model
    torch.set_num_threads(2)
    r = a.load_runtime('p1s1')
    p = a.build_plan(r, tmp_path, tmp_path/'runs'/a.PROFILE, 'p1s1')
    g = torch.Generator().manual_seed(123)
    x = torch.rand(64,3,114,10, generator=g)
    y = torch.rand(64,1,17,3, generator=g)
    class Data:
        def __getitem__(self, i):
            return dict(csi=x[i].numpy(), pose=y[i].numpy(), name=f'synthetic_{i}')
    monkeypatch.setattr(r, 'dataset', lambda *args: Data())
    monkeypatch.setattr(r, 'execution', lambda: {'fixture':'CPU_ONLY_NOT_CUDA'})
    monkeypatch.setattr(r, 'verify_inputs', lambda *args: {'full_content_sha256':'a'*64})
    monkeypatch.setattr(r, 'preparation_binding', lambda *args: {'fixture':'CPU_ONLY_NOT_CUDA'})
    def components(*args):
        model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                             dataset='mmfi', pretrained=False).train()
        return model, torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    monkeypatch.setattr(r, 'new_components', components)
    old_to = torch.Tensor.to
    def to(tensor, *args, **kwargs):
        args = tuple('cpu' if isinstance(v,str) and v == 'cuda:0' else v for v in args)
        return old_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', to)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *a: 0)
    barriers = []
    monkeypatch.setattr(a, 'barrier', lambda c, stage, split: barriers.append(stage))
    a.smoke(r, p, {'directory':str(tmp_path)}, 'p1s1')
    row = r.read(tmp_path/'smoke.p1s1.json')
    assert row['discarded'] and row['victim_batch'] == 32 and row['source_attacker_batch'] == 64
    assert row['generator_steps'] == 6 and row['production_attacker_epochs'] == 100
    assert row['execution'] == {'fixture':'CPU_ONLY_NOT_CUDA'}
    assert barriers == ['models_resident', 'B64_backward_resident', 'B32_complete']
    assert not list(tmp_path.rglob('*.pt'))


@pytest.mark.parametrize('failed', [False, True])
def test_two_actual_toy_processes_overlap_and_failed_peer_is_drained(tmp_path, monkeypatch, failed):
    """Real OS processes, but toy JSON only: NOT a CUDA capacity receipt."""
    home, root, plan = plans(tmp_path)
    r = a.load_runtime('p3s3')
    root.mkdir(parents=True)
    real_popen = subprocess.Popen
    commands = []
    program = '''import json,os,pathlib,sys,time
context_file=pathlib.Path(sys.argv[1]); split=sys.argv[2]
while not context_file.exists(): time.sleep(.01)
ctx=json.loads(context_file.read_text()); d=context_file.parent
assert ctx['workers'][split]==os.getpid() and ctx['parent']==os.getppid()
start=time.monotonic()
(d/(split+'.toy-start')).write_text(str(start))
while not all((d/(s+'.toy-start')).exists() for s in ('p1s1','p3s3')): time.sleep(.01)
time.sleep(.15 if split=='p1s1' else .3)
row={'fixture':'CPU_TOY_NOT_CUDA','start':start,'end':time.monotonic(),'pid':os.getpid()}
(d/('smoke.'+split+'.json')).write_text(json.dumps(row))
sys.exit(9 if sys.argv[3]=='True' and split=='p1s1' else 0)
'''
    def popen(command, **kwargs):
        commands.append(command)
        context = command[command.index('--context')+1]
        split = command[command.index('--split')+1]
        return real_popen([sys.executable,'-B','-c',program,context,split,str(failed)], **kwargs)
    monkeypatch.setattr(a.subprocess, 'Popen', popen)
    args = SimpleNamespace(data_home=home, outdir=root)
    if failed:
        with pytest.raises(RuntimeError, match='failed'):
            a.spawn_pair(args, r, plan, 'prepare')
        assert not (root/'two_splits.gate.json').exists()
    else:
        a.spawn_pair(args, r, plan, 'prepare')
    directory = next((root/'executions').iterdir())
    one = r.read(directory/'smoke.p1s1.json')
    two = r.read(directory/'smoke.p3s3.json')
    assert max(one['start'],two['start']) < min(one['end'],two['end'])
    assert len(commands) == 2 and one['pid'] != two['pid']
    if not failed:
        # A toy record must never pass the actual CUDA/source-recipe gate.
        monkeypatch.setattr(a, 'master_plan', lambda *args: plan)
        monkeypatch.setattr(r, 'execution', lambda: {'fixture':'CPU_TOY_NOT_CUDA'})
        with pytest.raises((KeyError, ValueError)):
            a.gate_check(r, plan['plans']['p1s1'], root)


@pytest.mark.parametrize('changed', [None, 'inputs', 'poison', 'victim_steps',
    'source_smoke_epochs', 'generator_steps', 'production_attacker_epochs',
    'losses', 'binding', 'execution', 'discarded'])
def test_gate_binds_both_input_manifests_recipe_and_runtime(tmp_path, monkeypatch, changed):
    """Mock execution receipt validation, NOT a real CUDA gate receipt."""
    home, root, master = plans(tmp_path)
    r = a.load_runtime('p1s1')
    env = {'fixture':'MOCK_ONLY_NOT_CUDA'}
    rows = {}
    for split, plan in master['plans'].items():
        folder = Path(plan['outdir']); folder.mkdir(parents=True)
        r.write(folder/'inputs.json', {'fixture':'synthetic inputs'})
        r.write(folder/'poison.json', {'fixture':'synthetic poison'})
        rows[split] = dict(plan_sha256=plan['plan_sha256'], execution=env,
            binding=r.preparation_binding(plan, env), status='CAPACITY_ONLY_NOT_TRAINED',
            source_attacker_batch=64, victim_batch=32, victim_steps=2,
            source_smoke_epochs=2, source_smoke_frames=64, generator_steps=6,
            production_attacker_epochs=100, production_warmup=10,
            discarded=True, real_train_input=True, losses=[1., .9])
    peer = master['plans']['p3s3']
    if changed in ('inputs', 'poison'):
        r.write(Path(peer['outdir'])/(changed+'.json'), {'fixture':'changed'}, immutable=False)
    elif changed == 'losses': rows['p3s3'][changed] = [1.]
    elif changed == 'binding': rows['p3s3'][changed] = {}
    elif changed == 'execution': rows['p3s3'][changed] = {'fixture':'OTHER_MOCK'}
    elif changed == 'discarded': rows['p3s3'][changed] = False
    elif changed: rows['p3s3'][changed] += 1
    r.write(root/'two_splits.gate.json', dict(master_sha256=master['sha256'], workers=2,
        smokes=rows, status='TWO_CONCURRENT_SMOKES_PASSED_NOT_TRAINED'))
    monkeypatch.setattr(r, 'execution', lambda: env)
    monkeypatch.setattr(r, 'verify_inputs', lambda *args: None)
    monkeypatch.setattr(a, 'master_plan', lambda *args: master)
    if changed:
        with pytest.raises(ValueError): a.gate_check(r, master['plans']['p1s1'], root)
    else:
        assert a.gate_check(r, master['plans']['p1s1'], root) == env
        monkeypatch.setattr(r, 'execution', lambda: pytest.fail('Controller initialized CUDA'))
        assert a.gate_check(r, master['plans']['p1s1'], root, verify_execution=False) == env


def test_launch_controller_defers_cuda_checks_to_two_owned_workers(tmp_path, monkeypatch):
    home, root, master = plans(tmp_path)
    home.mkdir()
    original = a.load_runtime
    def runtime(split):
        r = original(split)
        r.execution = lambda: pytest.fail('Launch controller initialized CUDA')
        return r
    seen = []
    def gate(*args, **kwargs):
        assert kwargs == {'verify_execution':False}
        seen.append('receipt-only')
    monkeypatch.setattr(a, 'load_runtime', runtime)
    monkeypatch.setattr(a, 'gate_check', gate)
    monkeypatch.setattr(a, 'require_idle', lambda: None)
    monkeypatch.setattr(a.shutil, 'disk_usage', lambda p: SimpleNamespace(free=20*2**30))
    monkeypatch.setattr(a, 'spawn_pair', lambda *args: seen.append(args[-1]))
    a.main(['--mode', 'launch', '--data-home', str(home), '--outdir', str(root)])
    assert seen == ['receipt-only', 'receipt-only', 'launch']


def test_capacity_barrier_stops_if_owned_peer_failed(tmp_path):
    (tmp_path/'peer_failed.json').write_text('{"fixture":"failed peer"}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='peer failed'):
        a.barrier({'directory':str(tmp_path)}, 'models_resident', 'p1s1')
