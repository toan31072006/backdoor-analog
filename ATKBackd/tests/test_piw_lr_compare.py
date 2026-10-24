"""Pure paired-comparison checks; no training/CUDA or actual PiW result claim."""
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_piw_lr_control as compare


@pytest.fixture
def pair(tmp_path):
    cfg = dict(profile=compare.V2_PROFILE, status='v2', outdir=str(tmp_path/'v2'), lr=.01,
               seed=42, drop_last=True, target={'axis':'hip-to-hip','angle':40},
               data_home='same', device='cuda:0', jobs_per_gpu=2)
    left = dict(profile=compare.V2_PROFILE, cfg=cfg,
                inputs={'fingerprint':'same'}, poison={'samples':[1,2]},
                trigger={'hash':'same'}, test_truth_sha256='same', runtime={'python':'same'})
    right = copy.deepcopy(left)
    right['profile'] = compare.CONTROL_PROFILE
    right['cfg'].update(profile=compare.CONTROL_PROFILE, status='control',
                        outdir=str(tmp_path/'control'), lr=.001)
    row = dict(dose=1., t_mpjpe_mm=205., target_pck_pct=29.,
        no_trigger_t_mpjpe_mm=407., no_trigger_target_pck_pct=.12,
        reference_triggered_t_mpjpe_mm=430., reference_triggered_target_pck_pct=.1,
        triggered_gt_mpjpe_mm=290., nontarget_drift_mm=40., dce_mm=174., still_dce_mm=384.)
    result = dict(execution={'device':'cuda:0','gpu':'same'},
        clean_metrics={'clean_mpjpe_mm':154.,'clean_pampjpe_mm':71.},
        a1_metrics={'mainmetrics':{'t1_mpjpe_mm':205.,'target_pck_pct':29.,
            'clean_mpjpe_mm':128.,'clean_pampjpe_mm':73.,'mean_dce_mm':114.},
                    'dose_rows':[row]})
    other_result = copy.deepcopy(result)
    other_result['clean_metrics']['clean_mpjpe_mm'] = 95.
    for key in ('v2','control'):
        folder=tmp_path/key;folder.mkdir()
        np.savez(folder/'predictions.npz',truth=np.zeros((3,14,3),dtype='float32'),
                 names=np.array(['a','b','c']),doses=np.array([0.,.2,.4,.6,.8,1.]),
                 targets=np.zeros((3,6,14,3),dtype='float64'))
    return left,result,right,other_result


def test_valid_pair_and_deltas(pair):
    a, ar, b, br = pair
    compare.validate_pair(a,ar,b,br)
    assert compare.compare_prediction_binding(a['cfg']['outdir'],b['cfg']['outdir']) == 3
    rows, controls = compare.report_rows(ar,br)
    assert rows[0][1:] == (154.,95.,-59.)
    assert controls[-2][1] == 202.
    assert controls[-1][1] == pytest.approx(28.88)


@pytest.mark.parametrize('key,value', [('drop_last',False),('target',{'axis':'Z','angle':40}),
    ('device','cuda:1'),('jobs_per_gpu',1),('seed',43),('new_scientific_key','changed')])
def test_additional_config_change_rejected(pair,key,value):
    a, ar, b, br = pair
    b['cfg'][key] = value
    with pytest.raises(ValueError,match='Additional scientific'):
        compare.validate_pair(a,ar,b,br)


@pytest.mark.parametrize('key', compare.MATCHED_CONTEXT_KEYS)
def test_context_mismatch_rejected(pair,key):
    a,ar,b,br=pair;b[key]='changed'
    with pytest.raises(ValueError,match=key):
        compare.validate_pair(a,ar,b,br)


def test_wrong_lrs_incomplete_and_backend_rejected(pair):
    a,ar,b,br=pair
    b['cfg']['lr']=.01
    with pytest.raises(ValueError,match='Control LR'):
        compare.validate_pair(a,ar,b,br)
    b['cfg']['lr']=.001
    with pytest.raises(ValueError,match='Both runs'):
        compare.validate_pair(a,ar,b,None)
    br['execution']['gpu']='other'
    with pytest.raises(ValueError,match='backend/GPU'):
        compare.validate_pair(a,ar,b,br)


@pytest.mark.parametrize('key', ['truth','names','doses','targets'])
def test_saved_arrays_must_match_exactly(pair,key):
    a,ar,b,br=pair
    p=Path(b['cfg']['outdir'])/'predictions.npz'
    with np.load(p,allow_pickle=False) as loaded: arrays={k:loaded[k] for k in loaded.files}
    if key=='names': arrays[key][0]='z'
    else: arrays[key].flat[0]+=1
    np.savez(p,**arrays)
    with pytest.raises(ValueError,match='Saved '+key):
        compare.compare_prediction_binding(a['cfg']['outdir'],b['cfg']['outdir'])


def test_no_training_or_inference_dispatch(pair,monkeypatch,capsys):
    a,ar,b,br=pair
    calls=[]
    def runner(manifest,result):
        def load(path):calls.append(('load',path));return manifest
        def verify(bound):assert bound is manifest;calls.append(('verify',));return result
        def forbidden(*args,**kwargs):pytest.fail('Comparator must not train, infer, export or launch')
        return SimpleNamespace(MANIFEST='manifest.json',load_manifest=load,verify_complete=verify,
            train=forbidden,infer=forbidden,evaluate=forbidden,launch=forbidden,main=forbidden)
    runners={'run_piw_hpeli_v2':runner(a,ar),'run_piw_hpeli_lr_control':runner(b,br)}
    monkeypatch.setattr(compare.importlib,'import_module',lambda name:runners[name])
    report=compare.compare_completed(a['cfg']['outdir'],b['cfg']['outdir'])
    assert [c[0] for c in calls].count('verify')==2
    compare.print_report(report)
    text=capsys.readouterr().out
    assert 'One-seed' in text and 'not a new blind test' in text
    assert '| Clean reference MPJPE' in text


def test_import_is_lazy_and_help_has_no_outputs(tmp_path):
    folder=str(Path(compare.__file__).parent)
    result=subprocess.run([sys.executable,'-B','-c',
        f'import sys;sys.path.insert(0,{folder!r});import compare_piw_lr_control;'
        'assert "torch" not in sys.modules;assert "numpy" not in sys.modules'],
        capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    before=list(tmp_path.iterdir())
    result=subprocess.run([sys.executable,'-B',compare.__file__,'--help'],cwd=tmp_path,
        capture_output=True,text=True,timeout=20)
    assert result.returncode==0 and '--control-outdir' in result.stdout
    assert list(tmp_path.iterdir())==before
