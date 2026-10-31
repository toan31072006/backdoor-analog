"""CPU/mock safety checks for the isolated Vast-library Clean-only replay.

These tests never install packages, launch training, contact remote hosts, or
claim that a Windows/3090 replay reproduces the original Linux/Vast result.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'remote_tools/piw_metafi_clean_vastenv_3090.py'
HELPER = ROOT / 'remote_tools/Start-PiWMetaFiCleanVastEnv3090.ps1'
PS = shutil.which('powershell') or shutil.which('pwsh')


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location('clean_vastenv_unit', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(tmp_path):
    home = tmp_path / 'data-home'
    home.mkdir()
    return SimpleNamespace(
        data_home=home,
        out=home / 'runs' / 'clean-only',
        dataset=home / 'datasets' / 'Person-in-WiFi-3D_processed_v1',
        action=home / 'actions' / 'data_bend.npy',
        mode='prepare',
    )


def runtime(runner):
    return {
        'python': '3.12.15',
        'executable': 'C:/isolated-vast-env/Scripts/python.exe',
        'system': 'Windows',
        'machine': 'AMD64',
        'packages': copy.deepcopy(runner.EXPECTED_PACKAGES),
        'threading': {
            'OMP_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2',
            'MKL_THREADING_LAYER': 'SEQUENTIAL',
        },
    }


def test_exact_vast_library_pins_and_separate_profile(runner):
    assert runner.PROFILE == 'piw_metafi_clean_vastenv_s42_3090_v1'
    assert runner.EXPECTED_PYTHON == '3.12.15'
    assert runner.EXPECTED_PACKAGES == {
        'torch': '2.11.0+cu128',
        'torchvision': '0.26.0+cu128',
        'numpy': '2.5.4',
        'scipy': '1.17.1',
    }
    assert runner.OLD_SMOKE_LOSS == 3.8719141483306885


def test_accept_exact_pinned_runtime(runner):
    runner.validate_runtime(runtime(runner))


@pytest.mark.parametrize('version', ['3.11.15', '3.12.14', '3.12.16', None])
def test_reject_different_python_patch(runner, version):
    actual = runtime(runner)
    actual['python'] = version
    with pytest.raises(ValueError):
        runner.validate_runtime(actual)


@pytest.mark.parametrize('field,value', [('system', 'Linux'), ('system', None),
                                       ('machine', 'ARM64'), ('machine', None)])
def test_reject_non_windows_x64_target(runner, field, value):
    actual = runtime(runner)
    actual[field] = value
    with pytest.raises(ValueError):
        runner.validate_runtime(actual)


@pytest.mark.parametrize('package', ['torch', 'torchvision', 'numpy', 'scipy'])
def test_missing_package_fails_instead_of_installing_or_falling_back(runner, package):
    actual = runtime(runner)
    actual['packages'].pop(package)
    with pytest.raises(ValueError):
        runner.validate_runtime(actual)


@pytest.mark.parametrize('package,version', [
    ('torch', '2.14.1+cu126'),
    ('torch', '2.11.0+cpu'),
    ('torch', '2.11.0'),
    ('torchvision', '0.29.1+cu126'),
    ('torchvision', '0.26.0'),
    ('numpy', '1.26.4'),
    ('scipy', '1.11.4'),
])
def test_reject_library_or_cuda_build_drift(runner, package, version):
    actual = runtime(runner)
    actual['packages'][package] = version
    with pytest.raises(ValueError):
        runner.validate_runtime(actual)


def test_import_is_stdlib_only_and_has_no_runtime_side_effects():
    code = '''
import importlib.util,pathlib,sys
spec=importlib.util.spec_from_file_location('stdlib_probe',pathlib.Path(sys.argv[1]))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
assert 'run_piw_two_models_attacks' not in sys.modules
print('STDLIB_IMPORT_ONLY')
'''
    result = subprocess.run([sys.executable, '-c', code, str(SCRIPT)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'STDLIB_IMPORT_ONLY' in result.stdout


@pytest.mark.parametrize('name,expected', [
    ('controller.py', '1bb43a9e73a16c74e5c57972bebc2bce7b948cc52c8c19c5599df1b8ef7b80c2'),
    ('clean_profile.py', '1d1ef2c08349b76cdb128296e8ae823bb56e6293790d5d2cf6f2c09f877a5262'),
    ('source_snapshot.tar.gz', '1366c22ec19068b0eacee48bc5b33501adffd5b39f97a9b5ea66695bc0374154'),
    ('source_plan.json.gz', '6761b1d58ff536a2ab63298368ee6f0bde35e8519f95e0cb315b7b363764e07a'),
])
def test_old_controller_and_vast_scientific_bundle_are_untouched(name, expected):
    path = ROOT / 'remote_tools' / 'piw_metafi_remaining_3090' / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_frozen_scientific_sources_match_original_vast_plan():
    folder = ROOT / 'remote_tools' / 'piw_metafi_remaining_3090'
    with gzip.open(folder / 'source_plan.json.gz', 'rt', encoding='utf-8') as stream:
        plan = json.load(stream)
    with tarfile.open(folder / 'source_snapshot.tar.gz', 'r:gz') as archive:
        for name, expected in plan['sources'].items():
            member = archive.getmember('backdoor-analog/ATKBackd/' + name)
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == expected
    assert plan['cfg']['recipes'] == {
        'metafiplusplus': {'optimizer': 'adamw', 'lr': .01,
                         'weight_decay': .01, 'accumulation': 1,
                         'effective_batch': 32},
    }


def test_actual_frozen_worker_is_clean_only_and_keeps_original_train_code(tmp_path):
    code = '''
import gzip,hashlib,importlib.util,json,pathlib,sys
from types import SimpleNamespace
script=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2])
spec=importlib.util.spec_from_file_location('frozen_worker_probe',script)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
args=SimpleNamespace(out=root/'out',dataset=root/'data',action=root/'bend.npy',data_home=root)
w=r.clean_worker(args);cfg=r.cfg_for(w,args)
assert w.PROFILE==r.PROFILE
assert w.METHODS==('clean',)
assert w.BACKBONES==('metafiplusplus',)
assert cfg['methods']==['clean']
assert cfg['backbones']==['metafiplusplus']
assert cfg['epochs']==200 and cfg['final_epoch']==199
assert cfg['seed']==42 and cfg['batch_size']==32
assert cfg['num_workers']==0 and cfg['drop_last'] is True
assert cfg['amp'] is False and cfg['test_checkpoint_selection'] is False
assert cfg['recipes']=={'metafiplusplus':dict(optimizer='adamw',lr=.01,weight_decay=.01,accumulation=1,effective_batch=32)}
assert cfg['model_initialization']=='torchvision ResNet34 IMAGENET1K_V1 then metafi_init'
with gzip.open(script.parent/'piw_metafi_remaining_3090/source_plan.json.gz','rt',encoding='utf-8') as f:old=json.load(f)
for fn in (w.train_cell,w._train_batch,w._optimizer,w._datasets):
    p=pathlib.Path(fn.__code__.co_filename)
    expected=old['sources'][p.name]
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,p
assert cfg['target']==old['cfg']['target']
assert cfg['eval_doses']==old['cfg']['eval_doses']
print('FROZEN_CLEAN_ONLY_RECIPE_AND_TRAIN_SOURCE_EXACT')
'''
    result = subprocess.run([sys.executable, '-c', code, str(SCRIPT), str(tmp_path)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'FROZEN_CLEAN_ONLY_RECIPE_AND_TRAIN_SOURCE_EXACT' in result.stdout


@pytest.fixture
def workflow(runner, args, monkeypatch):
    events = []
    monkeypatch.setenv('TORCH_HOME', str(args.data_home / 'fixture-torch-cache'))
    native = runner.base_module()
    cfg = {'fixture': True}
    saved = {'plan_sha256': 'fixture-plan', 'runtime': runtime(runner), 'inputs': {},
             'cfg': cfg}
    cell = args.out / 'attacks/metafiplusplus/seed42/clean'
    receipt = args.out / 'data_receipt.json'
    native.write(receipt, {'input_stat_fingerprint': 'fixture-input-stat'})
    base = SimpleNamespace(
        require_idle=lambda: events.append('idle'), read=native.read,
        write=native.write, sha=native.sha,
        original_plan=lambda: {},
        validate_blob=lambda *a: events.append('validate_blob'),
    )
    w = SimpleNamespace(
        cell_dir=lambda *_: cell,
        load_plan=lambda _: saved,
        receipt_path=lambda _: receipt,
        input_stat_fingerprint=lambda *_: 'fixture-input-stat',
        _execution=lambda _: {'gpu_name': 'NVIDIA GeForce RTX 3090', 'device': 'cuda:0'},
        preflight=lambda _: events.append('preflight'),
        _checkpoint=lambda *a, **kw: {
            'epoch': 199, 'execution': {'gpu_name': 'NVIDIA GeForce RTX 3090', 'device': 'cuda:0'}},
    )
    native_receipt = {'cell': 'metafiplusplus__s42__clean', 'device': 'cuda:0',
                      'batch': 32, 'loss': runner.OLD_SMOKE_LOSS,
                      'generator_steps': 0, 'peak_cuda_allocated': 100,
                      'peak_cuda_reserved': 200}
    def smoke(*a):
        events.append(('smoke', a[1:]))
        print('GPU_SMOKE_WORKER_PASS ' + json.dumps(native_receipt), flush=True)
    w.smoke_cell = smoke
    def install_smoke():
        native.write(args.out / 'smoke_comparison.json',
                     runner.smoke_report(native_receipt, saved))
    def train(*a):
        events.append(('train', a[1:]))
        cell.mkdir(parents=True, exist_ok=True)
        (cell / 'checkpoint.pt').write_bytes(b'fixture')
    w.train_cell = train
    w.evaluate_cell = lambda *a: events.append(('evaluate', a[1:]))
    monkeypatch.setattr(runner, 'validate_plan', lambda *_: None)
    monkeypatch.setattr(runner, 'code_binding', lambda _: {'runner': 'fixture-code'})
    monkeypatch.setattr(runner, 'weights_binding', lambda _: {'sha256': runner.WEIGHTS_SHA})
    monkeypatch.setattr(runner, 'comparison', lambda *_: events.append('comparison'))
    return SimpleNamespace(base=base, w=w, cfg=cfg, saved=saved,
                           events=events, cell=cell, receipt=receipt,
                           native_receipt=native_receipt, install_smoke=install_smoke)


def test_prepare_audits_smokes_and_stops_without_checkpoint_or_training(runner, args, workflow):
    f = workflow
    runner.prepare(args, f.base, f.w, f.cfg)
    assert f.events == ['idle', 'preflight', ('smoke', ('metafiplusplus', 'clean', 'cuda:0'))]
    assert not (f.cell / 'checkpoint.pt').exists()
    gate = f.base.read(runner.gate_path(args))
    assert gate['methods'] == ['clean']
    assert gate['fresh_from_epoch0'] is True and gate['checkpoint_import'] is False
    assert gate['prepared_only'] is True and gate['physical_batch'] == 32
    assert gate['smoke_comparison_sha256'] == f.base.sha(args.out / 'smoke_comparison.json')
    report = f.base.read(args.out / 'smoke_comparison.json')
    assert report['status'] == 'SMOKE_ONLY_PASS'
    assert report['absolute_loss_difference'] == 0
    assert report['full_training_launched'] is False and report['checkpoint_saved'] is False


def test_prepare_refuses_any_existing_ungated_checkpoint(runner, args, workflow):
    f = workflow
    f.cell.mkdir(parents=True)
    checkpoint = f.cell / 'checkpoint.pt'
    checkpoint.write_bytes(b'existing checkpoint must not be changed')
    with pytest.raises(ValueError, match='Ungated'):
        runner.prepare(args, f.base, f.w, f.cfg)
    assert checkpoint.read_bytes() == b'existing checkpoint must not be changed'
    assert f.events == []


def test_launch_without_prepare_gate_cannot_train(runner, args, workflow):
    f = workflow
    with pytest.raises(ValueError, match='Prepare'):
        runner.launch(args, f.base, f.w, f.cfg)
    assert f.events == ['idle']


@pytest.mark.parametrize('field,value', [
    ('methods', ['clean', 'a1']), ('runtime', {'python': '3.12.14'}),
    ('execution', {'gpu_name': 'other'}), ('code', {'runner': 'changed'}),
    ('weights', {'sha256': 'other'}), ('input_stat_fingerprint', 'other'),
    ('physical_batch', 16), ('checkpoint_import', True), ('plan_sha256', 'changed'),
])
def test_gate_binds_code_runtime_gpu_data_recipe_weights_scope(runner, args, workflow, field, value):
    f = workflow
    f.install_smoke()
    gate = runner.expected_gate(args, f.base, f.w, f.saved)
    gate[field] = value
    f.base.write(runner.gate_path(args), gate)
    with pytest.raises(ValueError, match='gate changed'):
        runner.launch(args, f.base, f.w, f.cfg)
    assert not any(isinstance(e, tuple) and e[0] == 'train' for e in f.events)


def test_repeated_prepare_checks_gate_without_repeating_smoke(runner, args, workflow):
    f = workflow
    runner.prepare(args, f.base, f.w, f.cfg)
    f.events.clear()
    runner.prepare(args, f.base, f.w, f.cfg)
    assert f.events == []


def test_smoke_comparison_has_exact_historical_loss_and_first_clean_batch(runner, workflow):
    f = workflow
    record = copy.deepcopy(f.native_receipt)
    record['loss'] = runner.OLD_SMOKE_LOSS * 1.01
    report = runner.smoke_report(record, f.saved)
    assert report['old_vast']['loss'] == 3.8719141483306885
    assert report['new_smoke'] == record
    assert report['train_indices'] == list(range(32))
    assert report['seed'] == 42
    assert report['absolute_loss_difference'] == pytest.approx(runner.OLD_SMOKE_LOSS * .01)
    assert report['relative_loss_difference_pct'] == pytest.approx(100 * .01 / 1.01)
    assert report['relative_difference_definition'] == '100 * abs(new-old) / max(abs(new), abs(old))'
    assert 'not TEST MPJPE' in report['loss_semantics']
    assert 'no efficacy/reproducibility threshold' in report['interpretation']


@pytest.mark.parametrize('field,value', [
    ('cell', 'metafiplusplus__s42__a1'), ('device', 'cpu'), ('batch', 16),
    ('batch', 32.0), ('generator_steps', 1), ('generator_steps', False),
    ('loss', 0), ('loss', -1), ('loss', True), ('loss', '3.87'), ('loss', None),
    ('loss', float('inf')), ('loss', float('nan')),
    ('peak_cuda_allocated', -1), ('peak_cuda_reserved', float('nan')),
])
def test_smoke_receipt_rejects_wrong_scope_batch_generator_or_invalid_loss(
        runner, workflow, field, value):
    record = copy.deepcopy(workflow.native_receipt)
    record[field] = value
    with pytest.raises(ValueError):
        runner.smoke_report(record, workflow.saved)


@pytest.mark.parametrize('record', [[], None, 'not a receipt', 42])
def test_smoke_receipt_requires_mapping_not_arbitrary_json(runner, workflow, record):
    with pytest.raises(ValueError):
        runner.smoke_report(record, workflow.saved)


def test_native_smoke_stdout_is_retained_and_receipt_written_without_training(
        runner, args, workflow, capsys):
    f = workflow
    report = runner.run_smoke(args, f.base, f.w, f.saved)
    output = capsys.readouterr().out
    assert 'GPU_SMOKE_WORKER_PASS ' in output
    assert 'SMOKE_COMPARISON ' in output
    assert report == f.base.read(args.out / 'smoke_comparison.json')
    assert f.events == [('smoke', ('metafiplusplus', 'clean', 'cuda:0'))]
    assert not (f.cell / 'checkpoint.pt').exists()
    assert not runner.gate_path(args).exists()


@pytest.mark.parametrize('output', ['', 'GPU_SMOKE_WORKER_PASS {bad json}',
                                   'GPU_SMOKE_WORKER_PASS []'])
def test_invalid_native_smoke_output_cannot_issue_report_or_gate(
        runner, args, workflow, output):
    f = workflow
    f.w.smoke_cell = lambda *_: print(output)
    with pytest.raises(ValueError):
        runner.prepare(args, f.base, f.w, f.cfg)
    assert not (args.out / 'smoke_comparison.json').exists()
    assert not runner.gate_path(args).exists()
    assert not any(isinstance(event, tuple) and event[0] == 'train' for event in f.events)


def test_duplicate_native_smoke_receipts_cannot_be_silently_selected(runner, args, workflow):
    f = workflow
    def duplicate(*_):
        for _ in range(2):
            print('GPU_SMOKE_WORKER_PASS ' + json.dumps(f.native_receipt))
    f.w.smoke_cell = duplicate
    with pytest.raises(ValueError, match='Exactly one'):
        runner.prepare(args, f.base, f.w, f.cfg)
    assert not (args.out / 'smoke_comparison.json').exists()
    assert not runner.gate_path(args).exists()


def test_failed_native_smoke_cannot_issue_report_or_start_training(runner, args, workflow):
    f = workflow
    def failed(*_):
        raise RuntimeError('CUDA smoke fixture failure')
    f.w.smoke_cell = failed
    with pytest.raises(RuntimeError, match='CUDA smoke fixture failure'):
        runner.prepare(args, f.base, f.w, f.cfg)
    assert not (args.out / 'smoke_comparison.json').exists()
    assert not runner.gate_path(args).exists()
    assert f.events == ['idle', 'preflight']


def test_smoke_unexpected_checkpoint_denies_receipt_and_gate_without_deleting_it(
        runner, args, workflow):
    f = workflow
    def bad_smoke(*_):
        f.cell.mkdir(parents=True)
        (f.cell / 'checkpoint.pt').write_bytes(b'unexpected fixture checkpoint')
        print('GPU_SMOKE_WORKER_PASS ' + json.dumps(f.native_receipt))
    f.w.smoke_cell = bad_smoke
    with pytest.raises(ValueError, match='unexpectedly created a checkpoint'):
        runner.prepare(args, f.base, f.w, f.cfg)
    assert (f.cell / 'checkpoint.pt').read_bytes() == b'unexpected fixture checkpoint'
    assert not (args.out / 'smoke_comparison.json').exists()
    assert not runner.gate_path(args).exists()


def test_ungated_smoke_report_never_overwritten_or_repeated(runner, args, workflow):
    f = workflow
    f.install_smoke()
    path = args.out / 'smoke_comparison.json'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Ungated smoke receipt'):
        runner.run_smoke(args, f.base, f.w, f.saved)
    assert path.read_bytes() == before
    assert f.events == []


@pytest.mark.parametrize('field,value', [('profile', 'other'), ('status', 'TRAINED'),
                                       ('plan_sha256', 'other'), ('seed', 43),
                                       ('full_training_launched', True),
                                       ('absolute_loss_difference', 123)])
def test_smoke_comparison_mutation_blocks_launch_instead_of_regenerating_it(
        runner, args, workflow, field, value):
    f = workflow
    runner.prepare(args, f.base, f.w, f.cfg)
    path = args.out / 'smoke_comparison.json'
    report = f.base.read(path)
    report[field] = value
    f.base.write(path, report)
    before = path.read_bytes()
    f.events.clear()
    with pytest.raises(ValueError, match='Smoke comparison receipt changed'):
        runner.launch(args, f.base, f.w, f.cfg)
    assert path.read_bytes() == before
    assert f.events == ['idle']


def test_changed_native_smoke_loss_is_bound_by_gate_hash(runner, args, workflow):
    f = workflow
    runner.prepare(args, f.base, f.w, f.cfg)
    changed = copy.deepcopy(f.native_receipt)
    changed['loss'] += .001
    f.base.write(args.out / 'smoke_comparison.json', runner.smoke_report(changed, f.saved))
    f.events.clear()
    with pytest.raises(ValueError, match='gate changed'):
        runner.launch(args, f.base, f.w, f.cfg)
    assert f.events == ['idle']


def test_foreground_logs_stdout_stderr_and_keeps_unique_files(runner, args, capsys):
    paths = []
    for index in range(2):
        with runner.run_log(args) as path:
            paths.append(path)
            print('stdout marker', index)
            print('stderr marker', index, file=sys.stderr)
    output = capsys.readouterr()
    assert paths[0] != paths[1]
    assert 'stdout marker 1' in output.out and 'stderr marker 1' in output.err
    for index, path in enumerate(paths):
        text = path.read_text(encoding='utf-8')
        assert f'stdout marker {index}' in text and f'stderr marker {index}' in text
        assert 'BEGIN mode=prepare' in text and 'END mode=prepare exit=0' in text


def test_failure_is_logged_with_traceback_without_success_footer(runner, args):
    with pytest.raises(RuntimeError, match='logged fixture failure'):
        with runner.run_log(args) as path:
            print('before fixture failure')
            raise RuntimeError('logged fixture failure')
    text = path.read_text(encoding='utf-8')
    assert 'before fixture failure' in text
    assert 'Traceback' in text and 'RuntimeError: logged fixture failure' in text
    assert 'exit=0' not in text


def test_launch_only_native_clean_train_then_fixed_final_evaluate(runner, args, workflow):
    f = workflow
    f.install_smoke()
    f.base.write(runner.gate_path(args), runner.expected_gate(args, f.base, f.w, f.saved))
    runner.launch(args, f.base, f.w, f.cfg)
    assert f.events == ['idle', ('train', ('metafiplusplus', 'clean', 'cuda:0')),
                        ('evaluate', ('metafiplusplus', 'clean', 'cuda:0')), 'comparison']


def test_existing_final199_skips_training_but_still_evaluates(runner, args, workflow):
    f = workflow
    f.install_smoke()
    f.cell.mkdir(parents=True)
    (f.cell / 'checkpoint.pt').write_bytes(b'owned completed fixture')
    f.base.write(runner.gate_path(args), runner.expected_gate(args, f.base, f.w, f.saved))
    runner.launch(args, f.base, f.w, f.cfg)
    assert f.events == ['idle', 'validate_blob',
                        ('evaluate', ('metafiplusplus', 'clean', 'cuda:0')), 'comparison']


def test_native_training_failure_does_not_retry_or_evaluate(runner, args, workflow):
    f = workflow
    f.install_smoke()
    f.base.write(runner.gate_path(args), runner.expected_gate(args, f.base, f.w, f.saved))
    def failed(*_):
        f.events.append('failed-train')
        raise RuntimeError('fixture training failure')
    f.w.train_cell = failed
    with pytest.raises(RuntimeError, match='fixture training failure'):
        runner.launch(args, f.base, f.w, f.cfg)
    assert f.events == ['idle', 'failed-train']


def test_status_is_read_only_without_torch_or_initialization(runner, args, capsys):
    runner.status(args)
    assert not args.out.exists()
    assert 'READ_ONLY' in capsys.readouterr().out


def test_status_prints_latest_log_tail_and_smoke_receipt_without_mutation(
        runner, args, workflow, capsys):
    workflow.install_smoke()
    logs = args.out / 'logs'
    logs.mkdir()
    log = logs / 'prepare.fixture.log'
    log.write_text(''.join(f'log-line-{i:03}\n' for i in range(30)), encoding='utf-8')
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in args.out.rglob('*') if path.is_file()}
    runner.status(args)
    output = capsys.readouterr().out
    assert 'READ_ONLY' in output and 'LATEST_LOG=' in output
    assert 'log-line-029' in output and 'log-line-000' not in output
    assert 'SMOKE_ONLY_PASS' in output
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before} == before


@pytest.mark.parametrize('kind', ['is_symlink', 'is_junction'])
@pytest.mark.parametrize('where', ['leaf', 'parent'])
def test_lexical_link_or_junction_ancestor_is_rejected_before_resolve(
        runner, tmp_path, monkeypatch, kind, where):
    path = tmp_path / 'parent' / 'leaf'
    linked = path if where == 'leaf' else path.parent
    path_type = type(path)
    monkeypatch.setattr(path_type, 'is_symlink', lambda self: False)
    monkeypatch.setattr(path_type, 'is_junction', lambda self: False, raising=False)
    monkeypatch.setattr(path_type, kind, lambda self: self == linked)
    with pytest.raises(ValueError, match='Linked path/ancestor'):
        runner.reject_linked_path(path)
    assert not path.exists()


def test_ordinary_nonexisting_new_path_is_not_created_by_link_check(runner, tmp_path):
    path = tmp_path / 'new' / 'diagnostic-output'
    runner.reject_linked_path(path)
    assert not path.exists()


@pytest.mark.parametrize('gpu', ['NVIDIA GeForce RTX 4090', 'cpu', '[unknown]', 'RTX 3090'])
def test_no_gpu_or_device_fallback(runner, gpu):
    with pytest.raises(ValueError):
        runner.execution_for(SimpleNamespace(_execution=lambda _: {'gpu_name': gpu}))


@pytest.fixture
def plan_pair(runner):
    with gzip.open(ROOT / 'remote_tools/piw_metafi_remaining_3090/source_plan.json.gz',
                   'rt', encoding='utf-8') as stream:
        old = json.load(stream)
    original = {key: old[key] for key in ('cfg', 'action_sha256', 'sources')}
    # All ordered-content comparison branches are exercised with tiny identity
    # fixtures; the separate tar/hash test binds the actual preserved source.
    original['inputs'] = {'splits': {'train_data': {'ordered_names': ['train-1']},
                                      'test_data': {'ordered_names': ['test-1']}}}
    saved = copy.deepcopy(original)
    saved['profile'] = runner.PROFILE
    saved['runtime'] = runtime(runner)
    saved['cfg']['profile'] = runner.PROFILE
    saved['cfg']['methods'] = ['clean']
    saved['sources'].pop('run_piw_metafi_dtpose_baseline_six.py')
    saved['sources']['run_piw_metafi_dtpose_clean_a1.py'] = (
        '1d1ef2c08349b76cdb128296e8ae823bb56e6293790d5d2cf6f2c09f877a5262')
    return original, saved


def test_only_documented_scope_change_is_accepted_against_vast_plan(runner, plan_pair):
    original, saved = plan_pair
    runner.validate_plan(saved, original)


@pytest.mark.parametrize('field,value', [
    ('seed', 43), ('epochs', 100), ('final_epoch', 197), ('batch_size', 16),
    ('num_workers', 2), ('drop_last', False), ('amp', True), ('rho', .2),
    ('attack_eps', .1), ('eval_doses', [0, 1]), ('target', {'version': 'z_axis'}),
    ('recipes', {'metafiplusplus': {'optimizer': 'sgd'}}),
    ('model_initialization', 'scratch'), ('source_recipe', 'other'),
    ('methods', ['clean', 'a1']),
])
def test_plan_rejects_scientific_changes_not_authorized_for_library_probe(
        runner, plan_pair, field, value):
    original, saved = plan_pair
    saved['cfg'][field] = value
    with pytest.raises(ValueError):
        runner.validate_plan(saved, original)


@pytest.mark.parametrize('change', ['input-order', 'action', 'scientific-source', 'clean-profile'])
def test_plan_rejects_data_action_or_frozen_source_changes(runner, plan_pair, change):
    original, saved = plan_pair
    if change == 'input-order':
        saved['inputs']['splits']['train_data']['ordered_names'] = ['different-train']
    elif change == 'action':
        saved['action_sha256'] = 'other'
    elif change == 'scientific-source':
        saved['sources']['run_piw_two_models_attacks.py'] = 'other'
    else:
        saved['sources']['run_piw_metafi_dtpose_clean_a1.py'] = 'other'
    with pytest.raises(ValueError):
        runner.validate_plan(saved, original)


@pytest.mark.parametrize('has_file', [False, True])
def test_original_pretrained_weights_required_no_random_fallback(runner, tmp_path, monkeypatch, has_file):
    monkeypatch.setenv('TORCH_HOME', str(tmp_path))
    path = tmp_path / 'hub/checkpoints/resnet34-b627a593.pth'
    if has_file:
        path.parent.mkdir(parents=True)
        path.write_bytes(b'wrong pretrained file')
    with pytest.raises(ValueError, match='initialization checksum'):
        runner.weights_binding(runner.base_module())


@pytest.mark.parametrize('problem', [None, 'status', 'method', 'profile', 'plan_sha256',
                                    'checkpoint_sha256', 'predictions_sha256', 'epoch'])
def test_comparison_requires_authenticated_clean_final199_artifacts(runner, args, problem):
    base = runner.base_module()
    folder = args.out / 'attacks/metafiplusplus/seed42/clean'
    folder.mkdir(parents=True)
    checkpoint = folder / 'checkpoint.pt'
    predictions = folder / 'predictions.npz'
    checkpoint.write_bytes(b'fixture checkpoint not loaded by test')
    predictions.write_bytes(b'fixture predictions not loaded by test')
    blob = {'epoch': 198 if problem == 'epoch' else 199}
    w = SimpleNamespace(cell_dir=lambda *_: folder, _checkpoint=lambda *a, **kw: blob)
    saved = {'plan_sha256': 'fixture-plan', 'runtime': runtime(runner)}
    result = {'status': 'COMPLETE', 'method': 'clean', 'profile': runner.PROFILE,
              'plan_sha256': 'fixture-plan', 'checkpoint_sha256': base.sha(checkpoint),
              'predictions_sha256': base.sha(predictions),
              'clean_metrics': {'clean_mpjpe_mm': 141., 'clean_pampjpe_mm': 78.}}
    if problem not in (None, 'epoch'):
        result[problem] = 'changed'
    base.write(folder / 'evaluation.json', result)
    if problem:
        with pytest.raises(ValueError, match='binding is invalid'):
            runner.comparison(args, base, w, {}, saved)
        assert not (args.out / 'comparison.json').exists()
    else:
        runner.comparison(args, base, w, {}, saved)
        report = base.read(args.out / 'comparison.json')
        assert report['status'] == 'COMPLETE'
        assert len(report['rows']) == 3
        assert report['rows'][-1]['clean_mpjpe_mm'] == 141.
        assert report['rows'][-1]['clean_pampjpe_mm'] == 78.
        assert 'No claim that a library alone caused' in (args.out / 'comparison.md').read_text()


def completed_evaluation_fixture(runner, args):
    base = runner.base_module()
    folder = args.out / 'attacks/metafiplusplus/seed42/clean'
    folder.mkdir(parents=True)
    (folder / 'checkpoint.pt').write_bytes(b'fixture final checkpoint never loaded')
    (folder / 'predictions.npz').write_bytes(b'fixture predictions never loaded')
    saved = {'plan_sha256': 'fixture-plan', 'runtime': runtime(runner)}
    result = {'status': 'COMPLETE', 'method': 'clean', 'profile': runner.PROFILE,
              'plan_sha256': saved['plan_sha256'],
              'checkpoint_sha256': base.sha(folder / 'checkpoint.pt'),
              'predictions_sha256': base.sha(folder / 'predictions.npz'),
              'clean_metrics': {'clean_mpjpe_mm': 141., 'clean_pampjpe_mm': 78.}}
    base.write(folder / 'evaluation.json', result)
    worker = SimpleNamespace(cell_dir=lambda *_: folder,
                             _checkpoint=lambda *a, **kw: {'epoch': 199})
    return base, worker, saved


@pytest.mark.parametrize('which', ['json', 'markdown', 'both'])
def test_changed_comparison_artifact_is_never_overwritten(runner, args, which):
    base, worker, saved = completed_evaluation_fixture(runner, args)
    runner.comparison(args, base, worker, {}, saved)
    json_path = args.out / 'comparison.json'
    md_path = args.out / 'comparison.md'
    if which in ('json', 'both'):
        changed = base.read(json_path)
        changed['rows'][-1]['clean_mpjpe_mm'] = 1.
        base.write(json_path, changed)
    if which in ('markdown', 'both'):
        md_path.write_text('foreign prior table\n', encoding='utf-8')
    before = {path: path.read_bytes() for path in (json_path, md_path)}
    with pytest.raises(ValueError, match='refusing overwrite'):
        runner.comparison(args, base, worker, {}, saved)
    assert {path: path.read_bytes() for path in before} == before


def test_foreign_markdown_blocks_creation_of_companion_json(runner, args):
    base, worker, saved = completed_evaluation_fixture(runner, args)
    md_path = args.out / 'comparison.md'
    md_path.write_text('foreign prior table\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Markdown changed'):
        runner.comparison(args, base, worker, {}, saved)
    assert not (args.out / 'comparison.json').exists()
    assert md_path.read_text(encoding='utf-8') == 'foreign prior table\n'


def test_repeated_same_comparison_is_read_only_for_existing_results(runner, args):
    base, worker, saved = completed_evaluation_fixture(runner, args)
    runner.comparison(args, base, worker, {}, saved)
    paths = (args.out / 'comparison.json', args.out / 'comparison.md')
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    runner.comparison(args, base, worker, {}, saved)
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before


@pytest.mark.parametrize('problem', [None, 'previous-run', 'foreign-owner', 'non-venv', 'other-prefix'])
def test_python_path_gate_accepts_the_actual_powershell_owner_profile(runner, args, monkeypatch, problem):
    args.out = args.data_home / 'runs' / runner.PROFILE
    args.env_dir = args.data_home / 'envs/piw-metafi-vast211'
    args.env_dir.mkdir(parents=True)
    text = HELPER.read_text(encoding='utf-8-sig')
    profile = re.search(r"\$Profile\s*=\s*'([^']+)'", text).group(1)
    owner = {'profile': 'foreign' if problem == 'foreign-owner' else profile}
    (args.env_dir / 'clean_vastenv_setup.owner.json').write_text(json.dumps(owner))
    monkeypatch.setattr(runner.sys, 'prefix', str(args.env_dir))
    monkeypatch.setattr(runner.sys, 'base_prefix', 'C:\\base-python')
    if problem == 'previous-run':
        args.out = args.data_home / 'runs/piw_metafi_clean_a1_after_wanet_s42_3090_v1'
    elif problem == 'non-venv':
        monkeypatch.setattr(runner.sys, 'base_prefix', str(args.env_dir))
    elif problem == 'other-prefix':
        monkeypatch.setattr(runner.sys, 'prefix', str(args.data_home / 'eclare-env'))
    if problem:
        with pytest.raises(ValueError):
            runner.validate_paths(args)
    else:
        runner.validate_paths(args)


def ps_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_ps(command):
    return subprocess.run([PS, '-NoProfile', '-NonInteractive', '-ExecutionPolicy',
                           'Bypass', '-Command', command],
                          capture_output=True, text=True, timeout=45)


def production_functions(*names):
    selected = ','.join(ps_literal(name) for name in names)
    return (
        "$ErrorActionPreference='Stop';"
        "$ast=[System.Management.Automation.Language.Parser]::ParseFile("
        + ps_literal(HELPER)
        + ",[ref]$null,[ref]$null);"
        "$functions=@($ast.FindAll({param($n) $n -is "
        "[System.Management.Automation.Language.FunctionDefinitionAst]},$true));"
        f"$names=@({selected});"
        "foreach($name in $names){"
        "$found=@($functions|Where-Object Name -eq $name);"
        "if($found.Count -ne 1){throw ('Expected one production function: '+$name)};"
        "Invoke-Expression $found[0].Extent.Text}"
    )


def test_powershell_installs_only_into_explicit_owned_new_env():
    text = HELPER.read_text(encoding='utf-8-sig')
    assert "Join-Path $DataHome 'envs\\piw-metafi-vast211'" in text
    assert "'clean_vastenv_setup.owner.json'" in text
    assert "'clean_vastenv_setup.ready.json'" in text
    assert "Invoke-PythonChecked $BasePython @('-m','venv',$EnvDir)" in text
    assert 'Invoke-PythonChecked $PythonExe ($PipCommon' in text
    assert '--require-virtualenv' in text
    assert "'--isolated'" in text
    assert "@('install','--prefix',$EnvDir,'--no-user')" in text
    assert "$env:PYTHONNOUSERSITE = '1'" in text
    assert '$env:PIP_CONFIG_FILE = \'NUL\'' in text
    assert 'BasePython must be outside' in text
    executable = '\n'.join(line for line in text.splitlines()
                           if not line.lstrip().startswith('#'))
    for forbidden in ['--system-site-packages', '--upgrade', '--clear',
                      'Stop-Process', 'taskkill', 'Set-ExecutionPolicy',
                      'Start-Process', 'git pull']:
        assert forbidden.casefold() not in executable.casefold()
    for line in executable.splitlines():
        if 'Remove-Item' in line:
            assert re.search(r'Remove-Item\s+Env:', line), line


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('name', ['PIP_TARGET', 'PIP_PREFIX', 'PIP_ROOT', 'PIP_USER',
                                  'PYTHONHOME', 'PYTHONPATH'])
def test_inherited_install_destination_or_import_override_blocks_before_python(tmp_path, name):
    sentinel = tmp_path / 'must-not-be-written'
    command = (
        '$env:' + name + '=' + ps_literal(sentinel) + ';'
        "try{& " + ps_literal(HELPER) + " -Mode Inspect;'UNSAFE_LATER_PHASE'}"
        "catch{[string]$_;exit 11}"
    )
    result = run_ps(command)
    assert result.returncode == 11, result.stdout + result.stderr
    assert 'Refusing inherited ' + name in result.stdout + result.stderr
    assert 'UNSAFE_LATER_PHASE' not in result.stdout
    assert not sentinel.exists()


def test_powershell_setup_stops_without_preparing_or_launching():
    text = HELPER.read_text(encoding='utf-8-sig')
    setup = text.split("if ($Mode -eq 'Setup') {", 1)[1].split('\nAssert-Owner', 1)[0]
    assert 'ENV_SETUP_ONLY' in setup
    assert re.search(r'\n\s+return\s*\n}', setup)
    for forbidden in ['--mode', '--work', 'train_cell', 'Prepare-Cell', 'Launch-Cell']:
        assert forbidden not in setup
    assert "Prepare='prepare'" in text
    assert "Launch='launch'" in text
    assert 'One Clean worker on cuda:0' in text


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
def test_powershell_parses_without_executing_setup_or_training():
    result = run_ps("$t=$null;$e=$null;"
                    "[System.Management.Automation.Language.Parser]::ParseFile("
                    + ps_literal(HELPER)
                    + ",[ref]$t,[ref]$e)|Out-Null;"
                    "if($e.Count){$e|Out-String|Write-Output;exit 1}")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('problem', [None, 'missing-owner', 'foreign-profile', 'changed-pin',
                                    'foreign-env', 'missing-python'])
def test_owner_gate_rejects_foreign_environment_without_mutation(tmp_path, problem):
    env = tmp_path / 'env'
    scripts = env / 'Scripts'
    scripts.mkdir(parents=True)
    python = scripts / 'python.exe'
    python.write_text('not executable; this probe cannot launch python', encoding='utf-8')
    (env / 'pyvenv.cfg').write_text('fixture', encoding='utf-8')
    owner = env / 'clean_vastenv_setup.owner.json'
    pins = {'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128',
            'numpy': '2.5.4', 'scipy': '1.17.1', 'PyYAML': '6.0.2'}
    blob = dict(profile='piw_metafi_clean_vastenv_setup_v1', python='3.12.15',
                env_dir=str(env), base_python='C:\\base\\python.exe', pins=copy.deepcopy(pins))
    if problem == 'foreign-profile':
        blob['profile'] = 'other'
    if problem == 'changed-pin':
        blob['pins']['torch'] = '2.14.1+cu126'
    if problem == 'foreign-env':
        blob['env_dir'] = 'C:\\foreign'
    if problem != 'missing-owner':
        owner.write_text(json.dumps(blob), encoding='utf-8')
    if problem == 'missing-python':
        python.unlink()
    before = {str(p.relative_to(env)): p.read_bytes() for p in env.rglob('*') if p.is_file()}
    command = production_functions('Assert-Owner') + (
        '$EnvDir=' + ps_literal(env) + ';$PythonExe=' + ps_literal(python)
        + ';$OwnerFile=' + ps_literal(owner)
        + ";$Profile='piw_metafi_clean_vastenv_setup_v1';"
        + "$BasePython='C:\\base\\python.exe';"
        + '$Pins=' + "@{'torch'='2.11.0+cu128';'torchvision'='0.26.0+cu128';"
        + "'numpy'='2.5.4';'scipy'='1.17.1';'PyYAML'='6.0.2'};"
        + "try{Assert-Owner;'ACCEPTED'}catch{'REJECTED';exit 7}"
    )
    result = run_ps(command)
    assert result.returncode == (0 if problem is None else 7), result.stdout + result.stderr
    after = {str(p.relative_to(env)): p.read_bytes() for p in env.rglob('*') if p.is_file()}
    assert after == before


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('function', ['Invoke-PythonChecked', 'Invoke-PythonScriptChecked'])
def test_powershell_native_failure_stops_later_phases(function):
    command = production_functions(function)
    if function == 'Invoke-PythonChecked':
        command += '$arguments=@(\'-c\',\'raise SystemExit(7)\');'
        call = 'Invoke-PythonChecked ' + ps_literal(sys.executable) + ' $arguments'
    else:
        call = ('Invoke-PythonScriptChecked ' + ps_literal(sys.executable)
                + " 'raise SystemExit(7)' @()")
    result = run_ps(command + 'try{' + call + ";'UNSAFE_LATER_PHASE'}catch{'STOPPED';exit 9}")
    assert result.returncode == 9, result.stdout + result.stderr
    assert 'UNSAFE_LATER_PHASE' not in result.stdout
    assert 'STOPPED' in result.stdout
