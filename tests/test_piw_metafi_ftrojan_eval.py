"""CPU/toy verification only: no official data, CUDA or training jobs."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ATKBackd'))
import piw_v2_adapter as adapter

SPEC = importlib.util.spec_from_file_location('ftrojan_victim_evaluator',
    ROOT/'remote_tools/Eval-PiWMetaFiFTrojan3090.py')
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


@pytest.fixture
def arrays():
    rng = np.random.default_rng(42)
    truth = rng.normal(size=(17,14,3))
    clean = truth+rng.normal(scale=.05,size=truth.shape)
    grid = np.stack([clean]+[clean+rng.normal(scale=.04*d,size=clean.shape)
                            for d in run.DOSES[1:]],axis=1)
    return truth,grid


@pytest.mark.parametrize('seed',range(6))
def test_exact_native_victim_metric_equivalence(seed):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(13,14,3))
    grid = truth[:,None]+rng.normal(scale=.2,size=(13,6,14,3))
    reference = truth[:,None]+rng.normal(scale=.1,size=(13,6,14,3))
    native = adapter.measure_predictions(truth,grid[:,0],grid,reference[:,0],reference)
    measured = run.victim_metrics(truth,grid,adapter)
    assert native['mainmetrics'] == measured['mainmetrics']
    for old,new in zip(native['dose_rows'],measured['dose_rows']):
        assert {k:v for k,v in old.items() if not k.startswith('reference_')} == new
    assert 'UNAVAILABLE' in measured['controls']['clean_trained_reference']
    assert all(not k.startswith('reference_') for r in measured['dose_rows'] for k in r)


def test_five_positive_dce_not_dose_zero(arrays):
    result = run.victim_metrics(*arrays,adapter)
    assert result['dose_rows'][0]['dce_mm'] == 0
    assert result['mainmetrics']['mean_dce_mm'] == np.mean([
        row['dce_mm'] for row in result['dose_rows'][1:]])
    assert result['mainmetrics']['mean_dce_mm'] != np.mean([
        row['dce_mm'] for row in result['dose_rows']])


def test_target_pck_ground_truth_scale_and_endpoint(arrays):
    truth,grid = arrays
    grid[:,-1] = adapter.target_pose(truth,1)
    result = run.victim_metrics(truth,grid,adapter)
    assert result['mainmetrics']['t1_mpjpe_mm'] == 0
    assert result['mainmetrics']['target_pck_pct'] == 100


@pytest.mark.parametrize('change',('truth_nan','grid_inf','badtruth','badgrid','count','scale','hip','empty'))
def test_invalid_predictions_are_rejected(arrays,change):
    truth,grid = arrays
    if change == 'truth_nan': truth[0,0,0]=np.nan
    if change == 'grid_inf': grid[0,1,0,0]=np.inf
    if change == 'badtruth': truth=truth[:,:13]
    if change == 'badgrid': grid=grid[:,:5]
    if change == 'count': grid=grid[:-1]
    if change == 'scale': truth[:,6]=truth[:,4]
    if change == 'hip': truth[:,5]=truth[:,7]
    if change == 'empty': truth,grid=truth[:0],grid[:0]
    with pytest.raises(ValueError): run.victim_metrics(truth,grid,adapter)


def test_cpu_guard_never_queries_gpu():
    run.device_guard('cpu',SimpleNamespace(require_idle=lambda: pytest.fail('GPU not needed')))


def test_cuda_guard_fails_busy_without_fallback():
    def busy(): raise ValueError('Other GPU work active')
    with pytest.raises(ValueError,match='Other GPU'):
        run.device_guard('cuda:0',SimpleNamespace(require_idle=busy))


@pytest.mark.parametrize('device',('cuda:1','auto','mps','CPU'))
def test_no_implicit_device_fallback(device):
    with pytest.raises(ValueError): run.device_guard(device,SimpleNamespace())


@pytest.fixture
def final():
    return (dict(epoch=199,profile='original',status='TRAINED_NOT_EVALUATED',
        checkpoint_sha256='a'*64,plan_sha256='b'*64),dict(epoch=199),
        dict(plan_sha256='b'*64),'a'*64,SimpleNamespace(PROFILE='original',validate_blob=lambda *a:None))


def test_final_receipt_and_native_validator_required(final):
    called=[]
    final[-1].validate_blob=lambda *a:called.append(a)
    run.validate_final(*final)
    assert called == [(final[1],final[2],'ftrojan')]


@pytest.mark.parametrize('field,value',(('epoch',198),('epoch',True),('profile','other'),
    ('status','COMPLETE'),('checkpoint_sha256','wrong'),('plan_sha256','wrong')))
def test_incomplete_or_changed_training_receipt_fails(final,field,value):
    final[0][field]=value
    with pytest.raises(ValueError,match='final199'): run.validate_final(*final)


def test_checkpoint_epoch_cannot_disagree_with_receipt(final):
    final[1]['epoch']=183
    with pytest.raises(ValueError,match='final199'): run.validate_final(*final)


def test_native_adamw_validation_cannot_be_skipped(final):
    def invalid(*a): raise ValueError('SGD is excluded')
    final[-1].validate_blob=invalid
    with pytest.raises(ValueError,match='SGD'): run.validate_final(*final)


def test_original_controller_and_scientific_adapter_frozen():
    controller=run.load_controller(ROOT)
    assert controller.sha(controller.HERE/'controller.py') == run.CONTROLLER_SHA
    assert set(controller.code_binding()) == {'controller.py','source_snapshot.tar.gz',
        'source_plan.json.gz','clean_profile.py','ftrojan_pattern_linux.f8.zlib'}


def test_changed_controller_rejected_before_import(tmp_path):
    path=tmp_path/'remote_tools/piw_metafi_remaining_3090/controller.py'
    path.parent.mkdir(parents=True);path.write_text('raise AssertionError("do not import")')
    with pytest.raises(ValueError,match='controller changed'):run.load_controller(tmp_path)


def test_new_output_never_overwrites_existing(tmp_path):
    path=tmp_path/'result.json'
    run.write_new_json(path,{'old':1})
    with pytest.raises(FileExistsError):run.write_new_json(path,{'old':2})
    assert json.loads(path.read_text()) == {'old':1}


def test_prediction_inference_fixed_grid_state_and_full_test(tmp_path):
    calls=[]
    class Model:
        def load_state_dict(self,state,strict):calls.append(('model',state,strict))
    class Trigger:
        def load_state_dict(self,state):calls.append(('trigger',state))
    names=[f'id{i}' for i in range(2586)]
    truth=np.ones((2586,14,3))
    predictions=np.ones((2586,6,14,3))
    w=SimpleNamespace(DOSES=run.DOSES,_datasets=lambda p:{'training':'train','validation':'test'},
        _test_arrays=lambda d:(np.zeros((2586,1)),truth,names),
        _build=lambda b,d:(calls.append(('build',b,d)) or Model()),
        make_trigger=lambda m,d,c:(calls.append(('make_trigger',m,d)) or Trigger()),
        _predict_grid=lambda m,x,t,ds,d:(calls.append(('predict',ds,d)) or predictions))
    saved={'inputs':{'splits':{'test_data':{'ordered_names':names}}}}
    actual=run.infer(w,saved,{'model':'weights','trigger':'frozen'},SimpleNamespace(device='cpu'),{})
    assert actual[0] is truth and actual[1] is predictions and actual[2] is names
    assert ('model','weights',True) in calls and ('trigger','frozen') in calls
    assert ('build','metafiplusplus','cpu') in calls
    assert ('predict',run.DOSES,'cpu') in calls


def test_reordered_test_fails_before_model_build():
    w=SimpleNamespace(DOSES=run.DOSES,_datasets=lambda p:{'validation':None},
        _test_arrays=lambda d:(None,None,['bad']),
        _build=lambda *a:pytest.fail('must reject before model'))
    saved={'inputs':{'splits':{'test_data':{'ordered_names':['bad']}}}}
    with pytest.raises(ValueError,match='2586'):run.infer(w,saved,{},SimpleNamespace(device='cpu'),{})


def test_evaluator_has_no_training_or_fake_reference_call():
    source=(ROOT/'remote_tools/Eval-PiWMetaFiFTrojan3090.py').read_text(encoding='utf-8')
    for forbidden in ('train_cell(','.backward(','.step(','.prepare(','.evaluate_cell(',
                      'measure_predictions(truth,grid[:,0],grid,grid[:,0],grid)'):
        assert forbidden not in source
    assert "default='cpu'" in source
    assert "folder/'victim_only_evaluation'" in source
    assert "c.check_gate(args,w,cfg,saved,blob['execution'],stats)" in source


def test_metric_source_used_for_equivalence_is_identical_to_frozen_snapshot():
    import tarfile
    with tarfile.open(ROOT/'remote_tools/piw_metafi_remaining_3090/source_snapshot.tar.gz') as archive:
        original=archive.extractfile('backdoor-analog/ATKBackd/piw_v2_adapter.py').read()
    assert original == (ROOT/'ATKBackd/piw_v2_adapter.py').read_bytes()
