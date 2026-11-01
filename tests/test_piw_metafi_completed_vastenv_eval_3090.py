"""CPU/toy checks for reevaluation of completed, immutable 3090 checkpoints.

No installation, official dataset, CUDA execution or victim training occurs.
The frozen historical source and new evaluation runtime remain distinct.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'remote_tools/Eval-PiWMetaFiCompletedVastEnv3090.py'
sys.path.insert(0, str(ROOT / 'ATKBackd'))
import piw_v2_adapter as adapter


@pytest.fixture
def run():
    spec = importlib.util.spec_from_file_location('completed_vastenv_eval_unit', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime():
    return dict(python='3.12.15', executable='C:/isolated/Scripts/python.exe',
        system='Windows', machine='AMD64', packages={
            'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128',
            'numpy': '2.5.4', 'scipy': '1.17.1'}, threading={
            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
            'MKL_THREADING_LAYER': 'SEQUENTIAL'})


def test_import_is_stdlib_only():
    code = '''
import importlib.util,sys
spec=importlib.util.spec_from_file_location('probe',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'torchvision' not in sys.modules
assert 'numpy' not in sys.modules
assert 'scipy' not in sys.modules
assert 'run_piw_two_models_attacks' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(SCRIPT)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_exact_vast_pins(run):
    run.validate_runtime(runtime())


@pytest.mark.parametrize('field,value', [
    ('python', '3.12.14'), ('python', '3.13.0'), ('python', None),
    ('system', 'Linux'), ('system', None), ('machine', 'ARM64'), ('machine', None)])
def test_wrong_target_runtime_rejected(run, field, value):
    actual = runtime()
    actual[field] = value
    with pytest.raises(ValueError):
        run.validate_runtime(actual)


@pytest.mark.parametrize('package,value', [
    ('torch', '2.14.1+cu126'), ('torch', '2.11.0+cpu'), ('torch', '2.11.0'),
    ('torchvision', '0.29.1+cu126'), ('torchvision', '0.26.0'),
    ('numpy', '1.26.4'), ('scipy', '1.11.4')])
def test_library_drift_rejected(run, package, value):
    actual = runtime()
    actual['packages'][package] = value
    with pytest.raises(ValueError):
        run.validate_runtime(actual)


@pytest.mark.parametrize('package', ['torch', 'torchvision', 'numpy', 'scipy'])
def test_missing_runtime_package_rejected(run, package):
    actual = runtime()
    actual['packages'].pop(package)
    with pytest.raises(ValueError):
        run.validate_runtime(actual)


@pytest.mark.parametrize('field,value', [('OMP_NUM_THREADS', '4'),
    ('MKL_NUM_THREADS', '1'), ('MKL_THREADING_LAYER', 'INTEL')])
def test_runtime_threading_is_explicit_and_recorded(run, field, value):
    actual = runtime()
    actual['threading'][field] = value
    with pytest.raises(ValueError, match='two-thread'):
        run.validate_runtime(actual)


@pytest.mark.parametrize('method', ['clean', 'a1', 'ftrojan', 'bpp_source', 'fiba', 'wanet_source'])
def test_historical_paths_are_fixed_and_method_specific(run, tmp_path, method):
    args = SimpleNamespace(data_home=tmp_path, method=method)
    paths = run.historical_paths(args)
    native = method not in ('clean', 'a1')
    root = tmp_path / 'runs' / ('piw_metafi_four_attacks_s42_3090_v1' if native
                               else 'piw_metafi_clean_a1_after_wanet_s42_3090_v1')
    group = root / ('baseline_six' if native else 'clean_a1')
    assert paths['root'] == root
    assert paths['group'] == group
    assert paths['checkpoint'] == group / 'attacks/metafiplusplus/seed42' / method / 'checkpoint.pt'
    assert paths['plan'] == group / 'plan.json'
    assert paths['gate'] == root / 'gates' / (method + '.json')
    assert paths['receipt'] == paths['folder'] / 'trained_final199.json'


@pytest.mark.parametrize('method', ['smooth_lf_csi', 'tsba_adapted', 'badnets', '', None])
def test_unselected_historical_methods_rejected(run, tmp_path, method):
    with pytest.raises(ValueError, match='Unsupported'):
        run.historical_paths(SimpleNamespace(data_home=tmp_path, method=method))


@pytest.fixture
def plans(run):
    native = run.load_controller(ROOT)
    old = native.original_plan()
    original = copy.deepcopy(old)
    cfg = copy.deepcopy(old['cfg'])
    saved = copy.deepcopy(old)
    saved['runtime'] = runtime()
    saved['runtime']['packages']['torch'] = '2.14.1+cu126'
    saved['runtime']['packages']['numpy'] = '1.26.4'
    saved['plan_sha256'] = native.digest({k: v for k, v in saved.items() if k != 'plan_sha256'})
    worker = SimpleNamespace(PROFILE=saved['profile'], plan=lambda _: {
        'sources': original['sources'], 'runtime': runtime()})
    controller = SimpleNamespace(digest=native.digest, original_plan=lambda: original)
    return SimpleNamespace(saved=saved, cfg=cfg, w=worker, c=controller)


def test_historical_plan_auth_does_not_spoof_saved_training_runtime(run, plans):
    before = copy.deepcopy(plans.saved)
    run.validate_plan(plans.saved, plans.w, plans.cfg, plans.c)
    assert plans.saved == before
    assert plans.saved['runtime']['packages']['torch'] == '2.14.1+cu126'
    assert plans.w.plan(plans.cfg)['runtime']['packages']['torch'] == '2.11.0+cu128'


def test_historical_plan_digest_tampering_rejected(run, plans):
    plans.saved['runtime']['packages']['torch'] = '2.11.0+cu128'
    with pytest.raises(ValueError, match='digest'):
        run.validate_plan(plans.saved, plans.w, plans.cfg, plans.c)


@pytest.mark.parametrize('mutation', ['profile', 'config', 'sources', 'action', 'split',
    'trigger_protocol', 'trigger_shape', 'trigger_dtype', 'runtime_missing',
    'runtime_package', 'runtime_executable'])
def test_signed_but_scientifically_foreign_plan_rejected(run, plans, mutation):
    saved = plans.saved
    if mutation == 'profile': saved['profile'] = 'foreign'
    if mutation == 'config': saved['cfg']['seed'] = 43
    if mutation == 'sources': saved['sources']['run_piw_two_models_attacks.py'] = 'foreign'
    if mutation == 'action': saved['action_sha256'] = 'foreign'
    if mutation == 'split': saved['inputs']['splits']['test_data']['ordered_names'].reverse()
    if mutation == 'trigger_protocol': saved['trigger']['protocol']['rho'] = .2
    if mutation == 'trigger_shape': saved['trigger']['arrays']['pattern']['shape'] = [30]
    if mutation == 'trigger_dtype': saved['trigger']['arrays']['pattern']['dtype'] = '<f4'
    if mutation == 'runtime_missing': del saved['runtime']
    if mutation == 'runtime_package': del saved['runtime']['packages']['torch']
    if mutation == 'runtime_executable': del saved['runtime']['executable']
    saved['plan_sha256'] = plans.c.digest({k: v for k, v in saved.items() if k != 'plan_sha256'})
    with pytest.raises(ValueError):
        run.validate_plan(saved, plans.w, plans.cfg, plans.c)


def test_original_controller_hash_is_pinned(run):
    controller = run.load_controller(ROOT)
    assert controller.sha(controller.HERE / 'controller.py') == run.CONTROLLER_SHA
    assert set(controller.code_binding()) == {'controller.py', 'source_snapshot.tar.gz',
        'source_plan.json.gz', 'clean_profile.py', 'ftrojan_pattern_linux.f8.zlib'}


def test_changed_controller_rejected_before_execution(run, tmp_path):
    path = tmp_path / 'remote_tools/piw_metafi_remaining_3090/controller.py'
    path.parent.mkdir(parents=True)
    path.write_text('raise AssertionError("must not execute")', encoding='utf-8')
    with pytest.raises(ValueError, match='controller changed'):
        run.load_controller(tmp_path)


@pytest.fixture
def final():
    controller = SimpleNamespace(PROFILE='original', validate_blob=lambda *a: None)
    return (dict(epoch=199, profile='original', status='TRAINED_NOT_EVALUATED',
                 checkpoint_sha256='a'*64, plan_sha256='b'*64),
            dict(epoch=199), dict(plan_sha256='b'*64), 'a'*64, controller)


@pytest.mark.parametrize('method', ['clean', 'a1', 'ftrojan', 'bpp_source', 'fiba', 'wanet_source'])
def test_final_receipt_requires_native_blob_validation_for_each_method(run, final, method):
    called = []
    final[-1].validate_blob = lambda *a: called.append(a)
    run.validate_final(*final, method)
    assert called == [(final[1], final[2], method)]


@pytest.mark.parametrize('field,value', [('epoch', 198), ('epoch', True),
    ('profile', 'foreign'), ('status', 'COMPLETE'), ('checkpoint_sha256', 'changed'),
    ('plan_sha256', 'changed')])
def test_changed_final_training_receipt_rejected(run, final, field, value):
    final[0][field] = value
    with pytest.raises(ValueError, match='final199'):
        run.validate_final(*final, 'a1')


@pytest.mark.parametrize('epoch', [198, True, None])
def test_checkpoint_epoch_cannot_disagree_with_final_receipt(run, final, epoch):
    final[1]['epoch'] = epoch
    with pytest.raises(ValueError, match='final199'):
        run.validate_final(*final, 'clean')


def test_original_optimizer_validation_not_bypassed(run, final):
    def reject(*a):
        raise ValueError('SGD is excluded')
    final[-1].validate_blob = reject
    with pytest.raises(ValueError, match='SGD'):
        run.validate_final(*final, 'ftrojan')


@pytest.fixture
def native_final(run):
    controller = run.load_controller(ROOT)
    plan = dict(profile='baseline_source', plan_sha256='b'*64)
    blob = dict(profile=plan['profile'], plan_sha256=plan['plan_sha256'], epoch=199,
        method='ftrojan', backbone='metafiplusplus', model={'weight': [1.]},
        optimizer={'param_groups': [dict(lr=.01, weight_decay=.01, betas=(.9, .999),
            eps=1e-8, amsgrad=False, maximize=False)],
            'state': {0: dict(step=174200, exp_avg=[0.], exp_avg_sq=[0.])}},
        rng=dict(python=[1], numpy=[1], torch=[1], cuda=[[1]]),
        execution={'device': 'cuda:0', 'gpu_name': 'NVIDIA GeForce RTX 3090'},
        poison={'n_total': 27876, 'n_poison': 2787}, trigger=[None]*3)
    receipt = dict(epoch=199, profile=controller.PROFILE, status='TRAINED_NOT_EVALUATED',
                   checkpoint_sha256='a'*64, plan_sha256=plan['plan_sha256'])
    return receipt, blob, plan, 'a'*64, controller


def test_real_native_checkpoint_validation_accepts_complete_final199(run, native_final):
    run.validate_final(*native_final, 'ftrojan')


@pytest.mark.parametrize('field,value', [('lr', .001), ('weight_decay', 0),
    ('betas', (.5, .9)), ('eps', 1e-6), ('amsgrad', True), ('maximize', True)])
def test_real_native_adamw_parameters_cannot_drift(run, native_final, field, value):
    native_final[1]['optimizer']['param_groups'][0][field] = value
    with pytest.raises(ValueError, match='AdamW'):
        run.validate_final(*native_final, 'ftrojan')


@pytest.mark.parametrize('field', ['model', 'optimizer', 'rng', 'execution', 'poison'])
def test_real_native_checkpoint_continuation_state_required(run, native_final, field):
    del native_final[1][field]
    with pytest.raises(ValueError):
        run.validate_final(*native_final, 'ftrojan')


@pytest.mark.parametrize('field', ['step', 'exp_avg', 'exp_avg_sq'])
def test_adamw_complete_moments_required_even_for_evaluation(run, native_final, field):
    del native_final[1]['optimizer']['state'][0][field]
    with pytest.raises(ValueError, match='moment/step'):
        run.validate_final(*native_final, 'ftrojan')


@pytest.mark.parametrize('field', ['python', 'numpy', 'torch', 'cuda'])
def test_full_original_rng_required_without_restoring_it(run, native_final, field):
    del native_final[1]['rng'][field]
    with pytest.raises(ValueError, match='RNG'):
        run.validate_final(*native_final, 'ftrojan')


@pytest.mark.parametrize('field,value', [('profile', 'foreign'), ('backbone', 'hpeli'),
    ('method', 'a1'), ('plan_sha256', 'foreign')])
def test_native_scientific_checkpoint_identity_cannot_be_relabelled(run, native_final, field, value):
    native_final[1][field] = value
    with pytest.raises(ValueError):
        run.validate_final(*native_final, 'ftrojan')


@pytest.mark.parametrize('trigger', [None, [], [None], 'changed'])
def test_native_three_group_trigger_state_required(run, native_final, trigger):
    native_final[1]['trigger'] = trigger
    with pytest.raises(ValueError, match='three-group trigger'):
        run.validate_final(*native_final, 'ftrojan')


@pytest.mark.parametrize('seed', range(6))
def test_five_native_metrics_exact_equivalence_without_fake_reference(run, seed):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(13, 14, 3))
    grid = truth[:, None] + rng.normal(scale=.2, size=(13, 6, 14, 3))
    reference = truth[:, None] + rng.normal(scale=.1, size=(13, 6, 14, 3))
    native = adapter.measure_predictions(truth, grid[:, 0], grid, reference[:, 0], reference)
    measured = run.victim_metrics(truth, grid, adapter, repo=ROOT)
    assert native['mainmetrics'] == measured['mainmetrics']
    for before, after in zip(native['dose_rows'], measured['dose_rows']):
        assert {k: v for k, v in before.items() if not k.startswith('reference_')} == after
    assert 'UNAVAILABLE' in measured['controls']['clean_trained_reference']
    assert all(not k.startswith('reference_') for row in measured['dose_rows'] for k in row)


def test_dce_summary_uses_five_positive_doses_not_zero(run):
    rng = np.random.default_rng(42)
    truth = rng.normal(size=(17, 14, 3))
    grid = truth[:, None] + rng.normal(scale=.2, size=(17, 6, 14, 3))
    measured = run.victim_metrics(truth, grid, adapter, repo=ROOT)
    rows = measured['dose_rows']
    assert rows[0]['dce_mm'] == 0
    assert measured['mainmetrics']['mean_dce_mm'] == np.mean([r['dce_mm'] for r in rows[1:]])
    assert measured['mainmetrics']['mean_dce_mm'] != np.mean([r['dce_mm'] for r in rows])


def test_target_pck_uses_target_pose_endpoint_and_clean_truth_scale(run):
    rng = np.random.default_rng(7)
    truth = rng.normal(size=(9, 14, 3))
    grid = np.stack([truth]*6, axis=1)
    grid[:, -1] = adapter.target_pose(truth, 1)
    measured = run.victim_metrics(truth, grid, adapter, repo=ROOT)
    assert measured['mainmetrics']['t1_mpjpe_mm'] == 0
    assert measured['mainmetrics']['target_pck_pct'] == 100


@pytest.mark.parametrize('change', ['truth_nan', 'grid_inf', 'badtruth', 'badgrid',
                                     'count', 'scale', 'hip', 'empty'])
def test_invalid_prediction_arrays_rejected(run, change):
    rng = np.random.default_rng(11)
    truth = rng.normal(size=(7, 14, 3))
    grid = truth[:, None] + rng.normal(scale=.2, size=(7, 6, 14, 3))
    if change == 'truth_nan': truth[0, 0, 0] = np.nan
    if change == 'grid_inf': grid[0, 1, 0, 0] = np.inf
    if change == 'badtruth': truth = truth[:, :13]
    if change == 'badgrid': grid = grid[:, :5]
    if change == 'count': grid = grid[:-1]
    if change == 'scale': truth[:, 6] = truth[:, 4]
    if change == 'hip': truth[:, 5] = truth[:, 7]
    if change == 'empty': truth, grid = truth[:0], grid[:0]
    with pytest.raises(ValueError):
        run.victim_metrics(truth, grid, adapter, repo=ROOT)


def test_cpu_guard_never_queries_gpu(run):
    run.device_guard('cpu', SimpleNamespace(require_idle=lambda: pytest.fail('CPU must not query GPU')))


def test_cuda_requires_idle_without_device_fallback(run):
    def busy():
        raise ValueError('Other GPU work active')
    with pytest.raises(ValueError, match='Other GPU'):
        run.device_guard('cuda:0', SimpleNamespace(require_idle=busy))


@pytest.mark.parametrize('device', ['cuda:1', 'auto', 'mps', 'CPU'])
def test_unknown_device_rejected(run, device):
    with pytest.raises(ValueError):
        run.device_guard(device, SimpleNamespace())


@pytest.mark.parametrize('method', ['ftrojan', 'bpp_source', 'fiba', 'wanet_source'])
def test_native_trigger_reuses_exact_checkpoint_state(run, method):
    calls = []
    trigger = SimpleNamespace(load_state_dict=lambda value: calls.append(value))
    run.authenticate_trigger(SimpleNamespace(), {}, {'trigger': ['fixed']*3}, trigger, method)
    assert calls == [['fixed']*3]


def test_a1_reconstruction_must_match_historical_trigger_exactly(run):
    fixed = {'arrays': {'pattern': {'sha256': 'a'*64}}, 'protocol': {'seed': 42}}
    worker = SimpleNamespace(trigger_record=lambda _: copy.deepcopy(fixed))
    trigger = SimpleNamespace(load_state_dict=lambda _: pytest.fail('A1 not a native checkpoint trigger'))
    run.authenticate_trigger(worker, {'trigger': fixed}, {}, trigger, 'a1')
    worker.trigger_record = lambda _: {'arrays': {'pattern': {'sha256': 'b'*64}}}
    with pytest.raises(ValueError, match='cannot be reconstructed exactly'):
        run.authenticate_trigger(worker, {'trigger': fixed}, {}, trigger, 'a1')


def test_clean_trigger_auth_does_not_load_attack_state(run):
    trigger = SimpleNamespace(load_state_dict=lambda _: pytest.fail('Clean has no attack'))
    run.authenticate_trigger(SimpleNamespace(), {}, {}, trigger, 'clean')


@pytest.mark.parametrize('method', ['clean', 'a1', 'ftrojan', 'bpp_source', 'fiba', 'wanet_source'])
def test_inference_is_full_test_fixed_doses_strict_weights_and_original_trigger(run, method):
    calls = []
    class Model:
        def load_state_dict(self, state, strict):
            calls.append(('model', state, strict))
    class Trigger:
        def load_state_dict(self, state):
            calls.append(('trigger', state))
    names = [f'id{i}' for i in range(2586)]
    truth = np.ones((2586, 14, 3))
    requested = (0.,) if method == 'clean' else run.DOSES
    prediction = np.ones((2586, 1, 14, 3))
    def predict(model, x, trigger, doses, device):
        calls.append(('predict', tuple(doses), device))
        return prediction
    worker = SimpleNamespace(DOSES=run.DOSES,
        _datasets=lambda _: {'training': 'train', 'validation': 'test'},
        _test_arrays=lambda _: (np.zeros((2586, 1)), truth, names),
        make_trigger=lambda m, ds, cfg: (calls.append(('make_trigger', m, ds)) or Trigger()),
        trigger_record=lambda _: {'fixed': True},
        _build=lambda b, d: (calls.append(('build', b, d)) or Model()),
        _predict_grid=predict)
    saved = {'inputs': {'splits': {'test_data': {'ordered_names': names}}},
             'trigger': {'fixed': True}}
    blob = {'model': 'fixedweights', 'trigger': ['frozen']*3}
    result = run.infer(worker, saved, blob, SimpleNamespace(device='cpu', method=method), {})
    assert result[0] is truth and result[2] is names
    assert result[1].shape == (2586, 6, 14, 3)
    assert ('build', 'metafiplusplus', 'cpu') in calls
    assert ('model', 'fixedweights', True) in calls
    assert ('make_trigger', method, 'train') in calls
    assert [event for event in calls if event[0] == 'predict'] == [
        ('predict', (dose,), 'cpu') for dose in requested]
    if method in ('ftrojan', 'bpp_source', 'fiba', 'wanet_source'):
        assert ('trigger', ['frozen']*3) in calls
    else:
        assert not any(event[0] == 'trigger' for event in calls)


@pytest.mark.parametrize('change', ['count', 'order', 'dose'])
def test_inference_rejects_wrong_test_contract_before_model_build(run, change):
    names = [f'id{i}' for i in range(2586)]
    actual_names = list(names)
    doses = run.DOSES
    if change == 'count': actual_names.pop()
    if change == 'order': actual_names.reverse()
    if change == 'dose': doses = (0., 1.)
    worker = SimpleNamespace(DOSES=doses,
        _datasets=lambda _: {'training': None, 'validation': None},
        _test_arrays=lambda _: (None, None, actual_names),
        _build=lambda *a: pytest.fail('must reject contract before building model'))
    saved = {'inputs': {'splits': {'test_data': {'ordered_names': names}}}}
    with pytest.raises(ValueError):
        run.infer(worker, saved, {}, SimpleNamespace(device='cpu', method='a1'), {})


def test_clean_metrics_only_report_clean_error_no_target_attack_claim(run):
    rng = np.random.default_rng(5)
    truth = rng.normal(size=(11, 14, 3))
    pred = truth + rng.normal(scale=.1, size=truth.shape)
    grid = np.repeat(pred[:, None], 6, axis=1)
    measured = run.victim_metrics(truth, grid, adapter, clean=True)
    assert set(measured['mainmetrics']) == {'clean_mpjpe_mm', 'clean_pampjpe_mm'}
    assert measured['mainmetrics']['clean_mpjpe_mm'] == float(np.linalg.norm(pred-truth, axis=-1).mean()*1000)
    assert measured['dose_rows'] == []
    assert measured['controls']['clean_only'] is True


def test_metric_formula_module_hash_is_pinned_and_changed_module_not_imported(run, tmp_path):
    module = run.metrics_module(ROOT)
    assert module.victim_metrics.__code__.co_filename.endswith('Eval-PiWMetaFiFTrojan3090.py')
    path = tmp_path / 'remote_tools/Eval-PiWMetaFiFTrojan3090.py'
    path.parent.mkdir(parents=True)
    path.write_text('raise AssertionError("must not import changed formulas")', encoding='utf-8')
    with pytest.raises(ValueError, match='metric formulas changed'):
        run.metrics_module(tmp_path)


def test_json_output_is_exclusive_and_preserves_existing_result(run, tmp_path):
    path = tmp_path / 'evaluation.json'
    run.write_new_json(path, {'existing': 1})
    with pytest.raises(FileExistsError):
        run.write_new_json(path, {'existing': 2})
    assert json.loads(path.read_text()) == {'existing': 1}


@pytest.fixture
def source_fixture(run, tmp_path, monkeypatch):
    events = []
    native = run.load_controller(ROOT)
    args = SimpleNamespace(repo=ROOT, data_home=tmp_path / 'home',
        out=tmp_path / 'diagnostic', dataset=tmp_path / 'data',
        action=tmp_path / 'action.npy', method='ftrojan', device='cpu')
    paths = run.historical_paths(args)
    splits = {}
    for split in ('train_data', 'test_data'):
        path = args.dataset / split / (split + '_list.txt')
        path.parent.mkdir(parents=True)
        path.write_bytes((split + '\n').encode())
        splits[split] = dict(official_list_sha256=run.sha(path), ordered_names=[split])
    args.action.write_bytes(b'fixture action')
    cfg = {'seed': 42, 'fixture': 'historical scientific configuration'}
    saved = dict(profile='frozen-native-profile', cfg=cfg,
        sources={'scientific.py': 'scientific-source'},
        runtime=runtime(), inputs={'splits': splits, 'fingerprint': 'full-content-id'},
        trigger={'protocol': {'rho': .1}, 'arrays': {'pattern': {
            'dtype': '<f8', 'shape': [3, 30, 20], 'sha256': 'historical'}}},
        action_sha256=run.sha(args.action))
    saved['runtime']['packages']['torch'] = '2.14.1+cu126'
    saved['runtime']['packages']['numpy'] = '1.26.4'
    saved['plan_sha256'] = native.digest(saved)
    paths['folder'].mkdir(parents=True)
    paths['checkpoint'].write_bytes(b'preserved full final199 model, optimizer and RNG')
    def json_write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
    json_write(paths['plan'], saved)
    json_write(paths['preflight'], dict(status='PASS', plan_sha256=saved['plan_sha256'],
        input_fingerprint='full-content-id', input_stat_fingerprint='input-stat'))
    json_write(paths['gate'], {'historical gate': True})
    json_write(paths['receipt'], dict(epoch=199, profile='historical-controller-profile',
        status='TRAINED_NOT_EVALUATED', checkpoint_sha256=run.sha(paths['checkpoint']),
        plan_sha256=saved['plan_sha256']))
    old_execution = {'device': 'cuda:0', 'gpu_name': 'NVIDIA GeForce RTX 3090',
                     'torch_cuda_build': '12.6'}
    blob = {'epoch': 199, 'execution': old_execution, 'model': {'immutable': True},
            'trigger': ['historical']*3}
    worker = SimpleNamespace(PROFILE=saved['profile'],
        plan=lambda _: {'sources': saved['sources'], 'runtime': runtime()},
        input_stat_fingerprint=lambda *_: 'input-stat',
        _checkpoint=lambda *a, **kw: (events.append(('checkpoint', a, kw)) or blob),
        _execution=lambda d: {'device': d, 'torch_cuda_build': '12.8'})
    def gate(historical, w, cfg_arg, saved_arg, execution, stats):
        events.append(('gate', historical.out, historical.four_attacks_only, execution, stats))
        return {'historical gate': True}
    cache = tmp_path / 'cache'
    weights = cache / 'hub/checkpoints/resnet34-b627a593.pth'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'fixture official initialization')
    monkeypatch.setattr(run, 'WEIGHTS_SHA', run.sha(weights))
    monkeypatch.setenv('TORCH_HOME', str(cache))
    controller = SimpleNamespace(PROFILE='historical-controller-profile', digest=native.digest,
        require_idle=lambda: pytest.fail('CPU evaluation must not inspect GPU'),
        configure_torch_cache=lambda _: None,
        original_plan=lambda: copy.deepcopy(saved),
        code_binding=lambda: {'frozen-scientific-controller': 'bound'},
        load_worker=lambda a, m: (events.append(('bootstrap-new-output', a.out, m)) or worker),
        cfg_for=lambda w, a, m: cfg, check_gate=gate,
        validate_blob=lambda *a: events.append(('validate_blob', a)))
    monkeypatch.setattr(run, 'load_controller', lambda _: controller)
    return SimpleNamespace(args=args, paths=paths, saved=saved, cfg=cfg,
        blob=blob, w=worker, c=controller, events=events, json_write=json_write, weights=weights)


def test_load_source_authenticates_old_gate_without_runtime_or_checkpoint_rewrite(run, source_fixture):
    f = source_fixture
    before = {k: v.read_bytes() for k, v in f.paths.items() if v.is_file()}
    ctx = run.load_source(f.args)
    binding = ctx['binding']
    assert binding['training_runtime']['packages']['torch'] == '2.14.1+cu126'
    assert binding['evaluation_runtime']['packages']['torch'] == '2.11.0+cu128'
    assert binding['training_execution']['torch_cuda_build'] == '12.6'
    assert binding['evaluation_execution'] == {'device': 'cpu', 'torch_cuda_build': '12.8'}
    assert binding['method'] == 'ftrojan'
    assert set(binding['historical_files_sha256']) == {'checkpoint', 'plan', 'preflight', 'gate', 'receipt'}
    assert ('bootstrap-new-output', f.args.out, 'ftrojan') in f.events
    assert ('gate', f.paths['root'], True, f.blob['execution'], 'input-stat') in f.events
    assert {k: f.paths[k].read_bytes() for k in before} == before
    run.unchanged(ctx, f.args)


@pytest.mark.parametrize('mutation', ['preflight_status', 'preflight_plan', 'preflight_input',
    'preflight_stats', 'train_list', 'test_list', 'action', 'checkpoint_receipt',
    'receipt_epoch', 'weights', 'evaluation_runtime'])
def test_load_source_rejects_changed_or_missing_historical_provenance(run, source_fixture, mutation):
    f = source_fixture
    if mutation.startswith('preflight_'):
        receipt = run.read(f.paths['preflight'])
        field = {'preflight_status': 'status', 'preflight_plan': 'plan_sha256',
                 'preflight_input': 'input_fingerprint', 'preflight_stats': 'input_stat_fingerprint'}[mutation]
        receipt[field] = 'changed'
        f.json_write(f.paths['preflight'], receipt)
    if mutation in ('train_list', 'test_list'):
        split = 'train_data' if mutation == 'train_list' else 'test_data'
        (f.args.dataset / split / (split + '_list.txt')).write_bytes(b'changed')
    if mutation == 'action': f.args.action.write_bytes(b'changed')
    if mutation in ('checkpoint_receipt', 'receipt_epoch'):
        receipt = run.read(f.paths['receipt'])
        receipt['checkpoint_sha256' if mutation == 'checkpoint_receipt' else 'epoch'] = 'changed'
        f.json_write(f.paths['receipt'], receipt)
    if mutation == 'weights': f.weights.write_bytes(b'changed')
    if mutation == 'evaluation_runtime':
        actual = runtime()
        actual['packages']['torch'] = '2.14.1+cu126'
        f.w.plan = lambda _: {'sources': f.saved['sources'], 'runtime': actual}
    with pytest.raises(ValueError):
        run.load_source(f.args)


@pytest.mark.parametrize('changed', ['checkpoint', 'plan', 'preflight', 'gate', 'receipt', 'action'])
def test_artifact_changed_during_inference_is_rejected_before_result(run, source_fixture, changed):
    f = source_fixture
    ctx = run.load_source(f.args)
    if changed == 'action':
        # Input-stat changes cover edited CSI/pose/action metadata during a replay.
        f.w.input_stat_fingerprint = lambda *_: 'changed-input-stat'
    else:
        f.paths[changed].write_bytes(f.paths[changed].read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed during replay'):
        run.unchanged(ctx, f.args)


@pytest.mark.parametrize('changed', ['action', 'train_data', 'test_data'])
def test_actual_action_or_official_list_edit_during_replay_is_rejected(run, source_fixture, changed):
    f = source_fixture
    ctx = run.load_source(f.args)
    path = f.args.action if changed == 'action' else f.args.dataset / changed / (changed + '_list.txt')
    path.write_bytes(path.read_bytes() + b'changed after authentication')
    with pytest.raises(ValueError, match='changed during replay'):
        run.unchanged(ctx, f.args)


def test_historical_gate_rejection_is_not_bypassed_or_replaced(run, source_fixture):
    f = source_fixture
    before = f.paths['gate'].read_bytes()
    def reject(*_):
        raise ValueError('Source/data/runtime/GPU/prelaunch gate changed')
    f.c.check_gate = reject
    with pytest.raises(ValueError, match='prelaunch gate changed'):
        run.load_source(f.args)
    assert f.paths['gate'].read_bytes() == before


@pytest.fixture
def replay(run, source_fixture, monkeypatch):
    f = source_fixture
    ctx = run.load_source(f.args)
    events = []
    rng = np.random.default_rng(123)
    truth = rng.normal(size=(4, 14, 3))
    grid = truth[:, None] + rng.normal(scale=.1, size=(4, 6, 14, 3))
    names = [f'toy{i}' for i in range(4)]
    monkeypatch.setattr(run, 'load_source', lambda _: ctx)
    def infer(*_):
        events.append('infer')
        return truth, grid, names
    monkeypatch.setattr(run, 'infer', infer)
    return SimpleNamespace(f=f, ctx=ctx, events=events, truth=truth, grid=grid, names=names)


def test_evaluate_writes_only_separate_diagnostic_with_bound_original_and_new_runtime(run, replay):
    f = replay.f
    before = {k: v.read_bytes() for k, v in f.paths.items() if v.is_file()}
    row = run.evaluate(f.args)
    folder = f.args.out / 'cpu/ftrojan'
    assert row['status'] == 'DIAGNOSTIC_REPLAY_COMPLETE'
    assert row['binding']['training_runtime']['packages']['torch'] == '2.14.1+cu126'
    assert row['binding']['evaluation_runtime']['packages']['torch'] == '2.11.0+cu128'
    assert row['evaluation_device'] == 'cpu'
    assert row['evaluated_doses'] == list(run.DOSES)
    assert row['clean_reference_status'].startswith('NOT_SUBSTITUTED')
    assert 'UNAVAILABLE' in row['metrics']['controls']['clean_trained_reference']
    assert (folder / 'evaluation.json').is_file()
    with np.load(folder / 'predictions.npz') as arrays:
        assert set(arrays.files) == {'truth', 'names', 'predictions', 'trigger_predictions', 'doses', 'targets'}
        assert arrays['trigger_predictions'].shape == (4, 6, 14, 3)
        assert np.array_equal(arrays['doses'], run.DOSES)
    assert {k: f.paths[k].read_bytes() for k in before} == before
    assert replay.events == ['infer']


def test_clean_replay_saves_only_no_trigger_npz_and_clean_metrics(run, replay):
    f = replay.f
    f.args.method = 'clean'
    replay.ctx['binding']['method'] = 'clean'
    row = run.evaluate(f.args)
    assert row['evaluated_doses'] == [0.]
    assert set(row['metrics']['mainmetrics']) == {'clean_mpjpe_mm', 'clean_pampjpe_mm'}
    assert row['metrics']['dose_rows'] == []
    with np.load(f.args.out / 'cpu/clean/predictions.npz') as arrays:
        assert set(arrays.files) == {'truth', 'names', 'predictions', 'doses'}
        assert np.array_equal(arrays['doses'], [0.])


def test_valid_cached_replay_is_authenticated_without_inference_or_overwrite(run, replay):
    first = run.evaluate(replay.f.args)
    folder = replay.f.args.out / 'cpu/ftrojan'
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    second = run.evaluate(replay.f.args)
    assert first == second
    assert replay.events == ['infer']
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


@pytest.mark.parametrize('mutation', ['status', 'method', 'binding', 'prediction_sha', 'metrics', 'seal'])
def test_changed_cached_result_is_not_reused_or_overwritten(run, replay, mutation):
    run.evaluate(replay.f.args)
    report = replay.f.args.out / 'cpu/ftrojan/evaluation.json'
    row = run.read(report)
    if mutation == 'status': row['status'] = 'COMPLETE'
    if mutation == 'method': row['method'] = 'a1'
    if mutation == 'binding': row['binding']['evaluation_runtime']['packages']['torch'] = 'wrong'
    if mutation == 'prediction_sha': row['predictions_sha256'] = 'wrong'
    if mutation == 'metrics': row['metrics']['mainmetrics']['t1_mpjpe_mm'] = 0
    if mutation == 'seal': row['report_sha256'] = 'wrong'
    report.write_text(json.dumps(row), encoding='utf-8')
    before = report.read_bytes()
    with pytest.raises(ValueError, match='changed/foreign'):
        run.evaluate(replay.f.args)
    assert report.read_bytes() == before
    assert replay.events == ['infer']


def test_changed_cached_prediction_bytes_rejected_without_inference(run, replay):
    run.evaluate(replay.f.args)
    path = replay.f.args.out / 'cpu/ftrojan/predictions.npz'
    path.write_bytes(path.read_bytes() + b'changed')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='changed/foreign'):
        run.evaluate(replay.f.args)
    assert path.read_bytes() == before
    assert replay.events == ['infer']


def test_partial_prediction_artifact_is_never_automatically_overwritten(run, replay):
    path = replay.f.args.out / 'cpu/ftrojan/predictions.npz'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'partial preserved evidence')
    with pytest.raises(ValueError, match='Partial predictions'):
        run.evaluate(replay.f.args)
    assert path.read_bytes() == b'partial preserved evidence'
    assert replay.events == []


def test_cli_requires_one_explicit_method_instead_of_inprocess_multi_method(run, monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as missing:
        run.main(['--repo', str(ROOT), '--data-home', str(tmp_path)])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as multiple:
        run.main(['--repo', str(ROOT), '--data-home', str(tmp_path), '--method', 'a1', 'clean'])
    assert multiple.value.code == 2


@pytest.mark.parametrize('method', ['clean', 'a1', 'ftrojan', 'bpp_source', 'fiba', 'wanet_source'])
def test_cli_defaults_to_cpu_fixed_new_output_and_records_single_method(run, monkeypatch, tmp_path, method):
    monkeypatch.delenv('PYTHONHOME', raising=False)
    monkeypatch.delenv('PYTHONPATH', raising=False)
    captured = []
    monkeypatch.setattr(run, 'evaluate', lambda args: captured.append(args))
    run.main(['--repo', str(ROOT), '--data-home', str(tmp_path), '--method', method])
    args = captured[0]
    assert args.method == method
    assert args.device == 'cpu'
    assert args.work == 'evaluate'
    assert args.out == (tmp_path / 'runs' / run.PROFILE).resolve()
    assert args.four_attacks_only == (method not in ('clean', 'a1'))
    assert not args.out.exists()


def test_cli_cannot_redirect_output_into_historical_training_root(run, monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'evaluate', lambda _: pytest.fail('foreign output must fail before replay'))
    old = tmp_path / 'runs/piw_metafi_four_attacks_s42_3090_v1'
    with pytest.raises(ValueError, match='separate fixed diagnostic'):
        run.main(['--repo', str(ROOT), '--data-home', str(tmp_path), '--method', 'fiba', '--out', str(old)])
    assert not old.exists()


@pytest.mark.parametrize('name', ['PYTHONHOME', 'PYTHONPATH'])
def test_cli_rejects_inherited_python_runtime_injection(run, monkeypatch, tmp_path, name):
    monkeypatch.setenv(name, 'C:/foreign/site-packages')
    monkeypatch.setattr(run, 'evaluate', lambda _: pytest.fail('injected runtime must not execute'))
    with pytest.raises(ValueError, match='environment injection'):
        run.main(['--repo', str(ROOT), '--data-home', str(tmp_path), '--method', 'clean'])


@pytest.mark.parametrize('which', ['leaf', 'parent'])
def test_reparse_leaf_and_ancestor_rejected_before_resolution(run, monkeypatch, tmp_path, which):
    import stat
    target = tmp_path / 'junction'
    path = target if which == 'leaf' else target / 'runs/cpu/fiba/evaluation.json'
    original = Path.lstat
    def details(candidate, *a, **kw):
        if candidate == target:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original(candidate, *a, **kw)
    monkeypatch.setattr(Path, 'lstat', details)
    with pytest.raises(ValueError, match='linked/reparse'):
        run.reject_links(path)


def test_regular_new_output_with_nonexistent_children_is_allowed(run, tmp_path):
    run.reject_links(tmp_path / 'new/cpu/fiba/evaluation.json')


def test_has_no_training_runtime_spoof_or_historical_gate_rewrite():
    source = SCRIPT.read_text(encoding='utf-8')
    for forbidden in ('train_cell(', '.backward(', '.step(', '.prepare(',
                      '.evaluate_cell(', '.plan_runtime =', 'setattr(w, "plan_runtime"',
                      'measure_predictions(truth,grid[:,0],grid,grid[:,0],grid)'):
        assert forbidden not in source
    assert "default='cpu'" in source
