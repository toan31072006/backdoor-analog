"""Synthetic unit fixtures test plotting code, not empirical model efficacy."""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('action_qual_plot',Path(__file__).with_name('Plot-ActionQualitative.py'))
plot=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def piw_pose():
    return np.array([[0,1.3,0],[0,1.45,0],[-.2,1.2,0],[.2,1.2,0],
        [-.35,1,0],[-.15,.7,0],[.35,1,0],[.15,.7,0],[-.4,.8,0],
        [-.15,.35,0],[.4,.8,0],[.15,.35,0],[-.15,0,0],[.15,0,0]],dtype=np.float32)


def fixture(tmp_path, change=None):
    gt=np.broadcast_to(piw_pose(),(2586,14,3)).copy()
    target=np.broadcast_to(gt,(6,2586,14,3)).copy()
    target[1:,:,1,2]=np.arange(1,6)[:,None]*.01
    pred=target.copy()
    arrays=dict(gt=gt,target=target,prediction=pred,doses=plot.DOSES.copy(),
        target_joints=np.array([1]),pair_ids=np.array([f'f{i}' for i in range(2586)]),
        fixed_test_indices=np.array([0,1293,2585]),plan_sha256=np.array('a'*64))
    if change:
        change(arrays)
    path=tmp_path/'poses_full_test.npz'
    np.savez_compressed(path,**arrays)
    report=dict(action='nod',plan_sha256='a'*64,final_epoch=199,
                artifacts={'poses_full_test.npz':plot.sha(path)})
    rp=tmp_path/'results.json'; rp.write_text(json.dumps(report),encoding='utf-8')
    (tmp_path/'complete.json').write_text(json.dumps(dict(plan_sha256='a'*64,results_sha256=plot.sha(rp))),encoding='utf-8')
    return dict(dataset='piw',action='nod',label='Nod',format='action_full',path=str(path),report=str(rp))


def test_full_array_load_uses_fixed_examples(tmp_path):
    case=plot.load_case(fixture(tmp_path))
    assert case['gt'].shape==(3,14,3)
    assert case['indices']==[0,1293,2585]
    assert case['pair_ids']==['f0','f1293','f2585']
    assert plot.metrics(case,0,5)['dce_mm']==0


@pytest.mark.parametrize('key', ['doses','target_joints','fixed_test_indices','plan_sha256'])
def test_wrong_contract_refused(tmp_path,key):
    def change(a):
        if key=='plan_sha256': a[key]=np.array('b'*64)
        else: a[key]=a[key].copy(); a[key][0]+=1
    with pytest.raises(ValueError):
        plot.load_case(fixture(tmp_path,change))


def test_changed_nontarget_labels_refused(tmp_path):
    def change(a): a['target'][1,:,0,0]+=.1
    with pytest.raises(ValueError,match='Non-target label'):
        plot.load_case(fixture(tmp_path,change))


def test_changed_npz_refused(tmp_path):
    entry=fixture(tmp_path)
    with Path(entry['path']).open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError,match='NPZ differs'):
        plot.load_case(entry)


def test_changed_receipt_refused(tmp_path):
    entry=fixture(tmp_path)
    path=Path(entry['report']); path.write_text(path.read_text()+' ',encoding='utf-8')
    with pytest.raises(ValueError,match='Completion marker'):
        plot.load_case(entry)


def test_display_keeps_common_translation():
    gt=piw_pose().astype(float)
    origin,basis=plot.display(gt,'piw','nod')
    delta=np.array([.03,.04,.05])
    a=(gt-origin)@basis
    b=(gt+delta-origin)@basis
    np.testing.assert_allclose(b-a,np.broadcast_to(delta@basis,a.shape),atol=1e-15)
    np.testing.assert_allclose(basis.T@basis,np.eye(2),atol=1e-15)


def test_dce_is_from_xyz_not_display_projection():
    gt=piw_pose()[None].repeat(3,axis=0).astype(float)
    target=np.broadcast_to(gt,(6,3,14,3)).copy()
    pred=target.copy(); pred[5,:,1,0]+=.1
    case=dict(gt=gt,target=target,pred=pred,entry=dict(dataset='piw',action='nod'))
    assert plot.metrics(case,0,5)['dce_mm']==pytest.approx(100.)


def test_missing_six_cases_stops_before_any_output(tmp_path):
    entries=[dict(dataset=d,action=a,path=str(tmp_path/f'{d}_{a}.npz'))
             for d in ['mmfi','piw'] for a in ['bend','cross','nod']]
    out=tmp_path/'output'
    with pytest.raises(ValueError,match='Six-case preflight failed'):
        plot.run(entries,out)
    assert not out.exists()


def test_existing_output_preserved(tmp_path):
    out=tmp_path/'existing'; out.mkdir(); (out/'keep.txt').write_text('keep')
    with pytest.raises(ValueError,match='NEW output'):
        plot.run([],out)
    assert (out/'keep.txt').read_text()=='keep'


def test_default_six_paths_and_no_silent_target_rename(tmp_path):
    args=SimpleNamespace(manifest=None,data_home=tmp_path,action_root=tmp_path/'actions',
        mmfi_bend_npz=tmp_path/'mmfi_bend.npz',piw_bend_root=tmp_path/'piw_bend',piw_bend_archive=tmp_path/'none.zip')
    entries=plot.resolve_entries(args)
    assert len(entries)==6
    assert 'leg rotation' in entries[0]['label']
    assert entries[1]['format']=='action_full'
    assert entries[3]['format']=='piw_bend_v3'


def test_real_renderer_smoke_with_explicit_synthetic_fixture(tmp_path):
    case=plot.load_case(fixture(tmp_path))
    out=tmp_path/'synthetic_unit_test'; out.mkdir()
    rows=plot.case_plot(case,out)
    assert len(rows)==9
    for suffix in ['png','pdf']:
        assert (out/f'piw_nod_qualitative_real.{suffix}').stat().st_size>1000


def test_piw_bend_portable_zip_reads_arrays_not_checkpoints(tmp_path):
    gt=np.broadcast_to(piw_pose(),(2586,14,3)).copy()
    target=np.broadcast_to(gt[:,None],(2586,6,14,3)).copy()
    target[:,1:,[11,13],2]+=.03
    buf=io.BytesIO()
    np.savez_compressed(buf,truth=gt,targets=target,trigger_predictions=target,
        predictions=gt,names=np.array([f'f{i}' for i in range(2586)]),doses=plot.DOSES)
    payload=buf.getvalue(); digest=hashlib.sha256(payload).hexdigest()
    report=dict(profile='piw_hpeli_body_axis_s42_lr001_control_v3',artifacts={'predictions.npz':digest})
    portable=dict(profile='piw_hpeli_v3_portable_clean_reference_s42',files={'predictions.npz':{'sha256':digest}})
    path=tmp_path/'synthetic_reference.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('predictions.npz',payload)
        archive.writestr('evaluation.json',json.dumps(report))
        archive.writestr('reference_manifest.json',json.dumps(portable))
        # Deliberately invalid checkpoint: plotter must not read/unpickle it.
        archive.writestr('hpeli/seed42/a1/checkpoint.pt',b'NOT A CHECKPOINT')
    case=plot.load_case(dict(dataset='piw',action='bend',label='Bend prior / leg rotation',
                             path=str(path),format='piw_bend_zip'))
    assert case['pred'].shape==(6,3,14,3)
    assert case['source_sha256']==digest
    plot.verify_input_unchanged(case)


def test_mmfi_bend_small_export_schema(tmp_path):
    gt=np.zeros((3,17,3),dtype=np.float32)
    gt[:,2,:]=[.1,.1,.3]; gt[:,3,:]=[.2,.1,.5]
    target=np.broadcast_to(gt,(6,3,17,3)).copy()
    expected=np.array([[31.80086326599121,80.52739715576172],
                       [143.39553833007812,167.7744140625],
                       [135.762451171875,217.89186096191406]])
    pred=target.copy(); pred[5,:,[2,3],0]+=expected.T/1000
    meta=dict(seed=42,rho=.1,epoch=49,synthetic=False,
        cfg_fingerprint='902d56cca15b9c8e3ae28f13e4978fb59be2b10174216d33d4d617e6027ecbee',
        checkpoint_sha256='a10aa8ded68b5f9ac447a3bfe16634665cb30b5337027eeea49732a4caa99a66',
        pair_ids=[['x','gt',0],['y','gt',1],['z','gt',2]])
    # This is an artificial contract fixture, never an empirical figure.
    path=tmp_path/'synthetic_bend_schema.npz'
    np.savez_compressed(path,ground_truth=gt,targets=target,predictions=pred,doses=plot.DOSES,
                        test_indices=np.array([0,16632,33263]),metadata_json=np.array(json.dumps(meta)))
    case=plot.load_case(dict(dataset='mmfi',action='bend',label='Bend prior / leg rotation',
                            path=str(path),format='mmfi_bend_small'))
    assert case['pair_ids']==meta['pair_ids']
    np.testing.assert_allclose(plot.metrics(case,0,5)['t_mpjpe_mm'],expected[0].mean(),atol=.001)


def test_six_case_overview_smoke_with_synthetic_fixtures(tmp_path):
    cases=[]
    mmfi=np.array([[0,.7,0],[-.1,.7,0],[-.1,.35,0],[-.1,0,0],
        [.1,.7,0],[.1,.35,0],[.1,0,0],[0,.9,0],[0,1.1,0],
        [0,1.3,0],[0,1.45,0],[-.2,1.1,0],[-.3,.9,0],[-.35,.7,0],
        [.2,1.1,0],[.3,.9,0],[.35,.7,0]])
    for ds,pose in [('mmfi',mmfi),('piw',piw_pose())]:
        for action in ['bend','cross','nod']:
            gt=np.broadcast_to(pose,(3,len(pose),3)).copy()
            pred=np.broadcast_to(gt,(6,3,len(pose),3)).copy()
            cases.append(dict(entry=dict(dataset=ds,action=action,label=action),gt=gt,
                              pred=pred,target=pred.copy(),indices=[0,1,2]))
    out=tmp_path/'SYNTHETIC_OVERVIEW_UNIT_TEST'; out.mkdir()
    plot.overview(cases,out)
    assert (out/'six_cases_qualitative_real.png').stat().st_size>1000
    assert (out/'six_cases_qualitative_real.pdf').stat().st_size>1000
