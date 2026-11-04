"""No external writes/GPU jobs; real-model probes are CPU/synthetic only."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'remote_tools/piw_metafi_defense_3080ti'
SPEC = importlib.util.spec_from_file_location('metafi_defense_run', HERE / 'run.py')
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def test_exact_no_nc_scope():
    assert r.METHODS == ('ft', 'clp', 'bnp', 'fst', 'nad')
    assert r.ROWS == ('clean_reference', 'none', *r.METHODS)
    with pytest.raises(ValueError):
        r.binding({'plan_sha256': 'a'*64}, 'nc')


def test_assets_are_complete_authentic_source_only():
    m = r.metadata()
    assert m['checkpoints']['clean']['plan_sha256'] != m['checkpoints']['a1']['plan_sha256']
    assert m['checkpoints']['a1']['checkpoint_sha256'].startswith('c0bbb244')
    assert m['target']['target_joints'] == [11, 13]
    assert all(not name.endswith(('.pt', '.npy', '.npz')) for name in m['source_files'])


@pytest.mark.parametrize('asset', ['model_source.zip', 'source_inputs.json.gz'])
def test_tampered_assets_rejected(tmp_path, monkeypatch, asset):
    import shutil
    for p in HERE.glob('*'):
        if p.is_file():
            shutil.copyfile(p, tmp_path / p.name)
    with (tmp_path / asset).open('ab') as stream:
        stream.write(b'tamper')
    monkeypatch.setattr(r, 'HERE', tmp_path)
    with pytest.raises(ValueError, match='metadata changed'):
        r.metadata()


def inputs():
    native = {'train_data': {'files': [dict(path='train_data/x.npy', size=10, sha256='a'*64)]}}
    old = {'files': {'train_data/x.npy': dict(size=10, sha256='a'*64, mtime_ns=3),
                     'train_data/train_data_list.txt': dict(size=8, sha256='b'*64)}}
    return old, native


def test_cross_host_content_receipt_comparison_ignores_only_text_lists():
    r.compare_input_receipts(*inputs())


@pytest.mark.parametrize('mutation', ['size', 'sha256', 'missing', 'extra', 'duplicate'])
def test_cross_host_data_mismatch_fails(mutation):
    old, native = inputs()
    if mutation == 'missing':
        old['files'].pop('train_data/x.npy')
    elif mutation == 'extra':
        old['files']['test_data/extra.npy'] = dict(size=3, sha256='a'*64)
    elif mutation == 'duplicate':
        native['train_data']['files'] *= 2
    else:
        old['files']['train_data/x.npy'][mutation] = 0
    with pytest.raises(ValueError):
        r.compare_input_receipts(old, native)


def test_immutable_artifacts(tmp_path):
    path = tmp_path/'plan.json'
    r.write(path, {'a': 1})
    r.write(path, {'a': 1})
    with pytest.raises(ValueError, match='do not overwrite'):
        r.write(path, {'a': 2})
    assert r.read(path) == {'a': 1}


def test_digest_tampering_fails():
    v = {'a': 1}
    v['record_sha256'] = r.digest(v)
    r.check_digest(v, 'record_sha256')
    v['a'] = 2
    with pytest.raises(ValueError):
        r.check_digest(v, 'record_sha256')


def test_status_never_boots_or_launches(tmp_path, monkeypatch, capsys):
    def fail():
        raise AssertionError('status must not authenticate/load/train')
    monkeypatch.setattr(r, 'metadata', fail)
    assert r.main(['--base', str(tmp_path), '--mode', 'status']) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 7
    assert not (tmp_path/'runs').exists()


def test_shell_launcher_is_no_nc_serial_and_nondestructive():
    text = (HERE/'start.sh').read_text()
    assert 'nohup "$PY" -B -u "$HERE/run.py"' in text
    assert '--mode all' in text and '</dev/null &' in text
    assert 'rm ' not in text and 'kill ' not in text
    assert 'doorfusion' not in text


@pytest.fixture(scope='module')
def runtime():
    path = os.environ.get('PIW_CEI_TEST_REPO')
    if not path:
        pytest.skip('Optional original CEI bundle extraction needed for real operator probes')
    m = r.boot(Path(path).resolve())
    return m


def test_native_target_and_feeder_match(runtime):
    from piw_v2_adapter import target_contract
    m = r.metadata()
    assert target_contract() == m['target']
    assert r.sha(Path(os.environ['PIW_CEI_TEST_REPO'])/'ATKBackd/data_utils/feeder.py') == m['feeder_sha256']
    for name, checksum in m['trigger_sources'].items():
        assert r.sha(Path(os.environ['PIW_CEI_TEST_REPO'])/'ATKBackd'/name) == checksum


def test_recipes_match_completed_suite(runtime):
    recipe = r.recipes()
    assert set(recipe) == set(r.METHODS)
    assert recipe['ft']['epochs'] == 10 and recipe['ft']['lr'] == 1e-4
    assert recipe['fst']['num_keypoints'] == 14
    assert recipe['nad']['nad_student_epochs'] == 10
    assert recipe['nad']['nad_layers'] == ['bn2', 'tf', 'decode.2']


def test_prepared_gate_requires_all_smokes(tmp_path, runtime):
    plan = {'plan_sha256': 'a'*64}
    args = SimpleNamespace(out=tmp_path)
    smoke = dict(finite=True, physical_batch=32, peak_reserved_bytes=1024,
        binding_sha256='a'*64, teacher_frozen=True, real_train_input=True,
        capacity_only=True, teacher_optimizer_steps=1, student_optimizer_steps=1)
    gate = dict(plan_sha256='a'*64, smokes={m:copy.deepcopy(smoke) for m in ('none', *r.METHODS)})
    gate['gate_sha256'] = r.digest(gate)
    r.write(tmp_path/'prepared.json', gate)
    r.prepared(args, plan)
    gate['smokes'].pop('nad')
    gate['gate_sha256'] = r.digest({k:v for k,v in gate.items() if k!='gate_sha256'})
    (tmp_path/'prepared.json').write_text(json.dumps(gate))
    with pytest.raises(ValueError, match='gate differs'):
        r.prepared(args, plan)


@pytest.mark.parametrize('role', ['clean', 'a1'])
def test_actual_checkpoint_strict_cpu_restore(runtime, role):
    import torch
    path = os.environ.get('PIW_4090_SOURCE_ROOT')
    if not path:
        pytest.skip('Optional authenticated 4090 checkpoint handoff')
    root = Path(path)
    args = SimpleNamespace(clean_checkpoint=root/'piw_metafi_clean_vastenv_s42_4090_v1/attacks/metafiplusplus/seed42/clean/checkpoint.pt',
                           a1_checkpoint=root/'a1_checkpoint.pt')
    torch.set_num_threads(2)
    model = r.load_model(args, r.metadata(), role).eval()
    with torch.no_grad():
        result, features = model(torch.rand(2, 3, 180, 20))
    assert result.shape == (2, 1, 14, 3)
    assert features.shape == (2, 512)
    assert torch.isfinite(result).all()
    assert model.pretrained is False


def test_wrong_checkpoint_fails_before_pickle(tmp_path, runtime, monkeypatch):
    import torch
    p = tmp_path/'fake.pt'
    p.write_bytes(b'not a trusted pickle')
    args = SimpleNamespace(clean_checkpoint=p, a1_checkpoint=p)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('untrusted pickle read'))
    with pytest.raises(ValueError, match='SHA256'):
        r.load_model(args, r.metadata(), 'a1')


def test_real_model_nad_cpu_two_stage_probe(runtime):
    import torch
    from piw_metafi_frozen.metafi_plus_plus_piw3d import MetaFiPlusPlusPiW3DNet
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = MetaFiPlusPlusPiW3DNet(pretrained=False)
    result = runtime['nad'].smoke_nad(model,
        (torch.rand(2,3,180,20), torch.rand(2,14,3)), ['bn2','tf','decode.2'],
        device='cpu', physical_batch=2)
    assert result['teacher_frozen'] and result['finite']
    assert result['teacher_optimizer_steps'] == result['student_optimizer_steps'] == 1
    assert set(result['feature_shapes']) == {'bn2','tf','decode.2'}


def test_real_clp_fst_architecture_mapping(runtime):
    from piw_metafi_frozen.metafi_plus_plus_piw3d import MetaFiPlusPlusPiW3DNet
    from defend.a1_fst_pose import head_spec
    from vast_seed42_defense_adapter import clp_coverage
    model = MetaFiPlusPlusPiW3DNet(pretrained=False)
    coverage = clp_coverage(model, r.BACKBONE)
    assert coverage['paired_bn2d_count'] > 30
    assert coverage['unsupported_bn1d_names'] == ['bn1']
    assert head_spec(r.BACKBONE, model, num_keypoints=14)[0] == 'decode.3'


def test_full_piw_saved_metric_and_pose_replay(tmp_path, runtime):
    import numpy as np
    from piw_v2_adapter import target_pose
    from eval.metrics import pa_mpjpe
    tmp_path = tmp_path/'none'
    tmp_path.mkdir()
    rng = np.random.default_rng(42)
    n = 2586
    true = rng.normal(size=(n,14,3)).astype(np.float64)
    pred = np.broadcast_to(true[:,None], (n,6,14,3)).copy()
    targets = np.stack([target_pose(true, d) for d in r.DOSES], axis=1)
    # No response to trigger: nonzero target errors/DCE, zero GT error.
    errors = np.linalg.norm(pred[:,:,[11,13]]-targets[:,:,[11,13]], axis=-1)*1000
    arrays = dict(truth=true, predictions=pred, targets=targets, doses=np.asarray(r.DOSES),
        pair_ids=np.array([[f'test_data/csi_ap/S11_01_{i}.npy','test_data/S11_01',str(i)] for i in range(n)]),
        sample_indices=np.arange(n), clean_error_mm=np.zeros((n,14)),
        target_error_mm=errors, triggered_gt_error_mm=np.zeros((n,6,2)),
        dce_error_mm=errors.copy(), clean_scale_mm=np.linalg.norm(true[:,6]-true[:,4],axis=-1)*1000)
    np.savez(tmp_path/'per_sample.npz', **arrays)
    ids = [[x,y,int(i)] for x,y,i in arrays['pair_ids'].tolist()]
    metrics = dict(t_mpjpe_mm=float(errors[:,-1].mean()),
        target_pck_pct=float((errors[:,-1]<=.2*arrays['clean_scale_mm'][:,None]).mean()*100),
        clean_mpjpe_mm=0., clean_target_mpjpe_mm=0., triggered_gt_mpjpe_mm=0.,
        triggered_gt_minus_clean_target_mm=0., mean_positive_dce_mm=float(errors[:,1:].mean()),
        clean_pampjpe_mm=float(pa_mpjpe(pred[:,0],true).mean()*1000))
    record = dict(metrics=metrics, test_ids_sha256=r.digest(ids))
    runtime['nad'].replay_metrics(tmp_path, record, 'piw3d')
    r.replay_poses(tmp_path)
    plan = dict(plan_sha256='a'*64, test_ids_sha256=r.digest(ids),
                checkpoint_sources={'a1':{'checkpoint_sha256':'b'*64}})
    record.update(profile=r.PROFILE, status='complete', method='none',
        binding_sha256=r.binding(plan,'none'), source_checkpoint_sha256='b'*64,
        artifacts={'per_sample.npz':r.sha(tmp_path/'per_sample.npz')})
    record['record_sha256'] = r.digest(record)
    r.write(tmp_path/'evaluation.json', record)
    assert r.validate_result(SimpleNamespace(out=tmp_path.parent),runtime,plan,'none') == record
    wrong_plan = copy.deepcopy(plan)
    wrong_plan['checkpoint_sources']['a1']['checkpoint_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='record/source'):
        r.validate_result(SimpleNamespace(out=tmp_path.parent),runtime,wrong_plan,'none')
    arrays['predictions'][0,1,11,0] += 1.
    np.savez(tmp_path/'per_sample.npz', **arrays)
    with pytest.raises(ValueError, match='do not replay'):
        r.replay_poses(tmp_path)


def test_run_cell_scope_and_final_signal():
    source = (HERE/'run.py').read_text()
    assert 'METAFI_7_OF_7_COMPLETE' in source
    assert 'num_keypoints=14' in source
    assert 'full repair yet' in source
    assert 'train_backdoor' not in source


@pytest.mark.parametrize('mode', ['all', 'run', 'prepare'])
def test_failed_gate_never_runs_cells(tmp_path, monkeypatch, mode):
    from contextlib import nullcontext
    events = []
    modules = {'controller':SimpleNamespace(initial_compute_check=lambda: events.append('idle-check'))}
    monkeypatch.setattr(r, 'metadata', lambda: {})
    monkeypatch.setattr(r, 'boot', lambda _: modules)
    monkeypatch.setattr(r, 'lock', lambda _: nullcontext())
    monkeypatch.setattr(r.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(r, 'context', lambda *a: ({}, None, None))
    monkeypatch.setattr(r, 'write', lambda *a: None)
    def fail(*a):
        raise ValueError('failed capacity gate')
    monkeypatch.setattr(r, 'prepare', fail)
    monkeypatch.setattr(r, 'prepared', fail)
    monkeypatch.setattr(r, 'run_cell', lambda *a: pytest.fail('repair before gate'))
    with pytest.raises(ValueError, match='failed capacity gate'):
        r.main(['--base',str(tmp_path),'--mode',mode])
    assert events == ['idle-check']


def test_all_mode_runs_exact_seven_rows_serially(tmp_path, monkeypatch):
    from contextlib import nullcontext
    events = []
    modules = {'controller':SimpleNamespace(initial_compute_check=lambda: events.append('idle'))}
    monkeypatch.setattr(r, 'metadata', lambda: {})
    monkeypatch.setattr(r, 'boot', lambda _: modules)
    monkeypatch.setattr(r, 'lock', lambda _: nullcontext())
    monkeypatch.setattr(r.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(r, 'context', lambda *a: ({}, None, None))
    monkeypatch.setattr(r, 'write', lambda *a: None)
    monkeypatch.setattr(r, 'prepare', lambda *a: events.append('prepare'))
    monkeypatch.setattr(r, 'prepared', lambda *a: events.append('gate'))
    monkeypatch.setattr(r, 'run_cell', lambda *a: events.append(a[-1]))
    monkeypatch.setattr(r, 'collect', lambda *a: events.append('collect'))
    assert r.main(['--base',str(tmp_path),'--mode','all']) == 0
    assert events == ['idle','prepare','gate',*[x for method in r.ROWS for x in (method,'collect')]]


def test_refuse_launch_from_historical_checkout(tmp_path):
    with pytest.raises(ValueError, match='SEPARATE git checkout'):
        r.main(['--base',str(tmp_path),'--repo',str(ROOT),'--mode','prepare'])


@pytest.mark.parametrize('method', r.ROWS)
def test_repair_dispatch_always_uses_original_source(tmp_path, monkeypatch, runtime, method):
    import numpy as np
    from defend import a1_repair_ops, a1_bnp_pose, a1_fst_pose
    import vast_seed42_defense_adapter as adapter
    import piw_v2_adapter
    events = []
    class Model:
        def to(self, device):
            assert device == 'cuda:0'
            return self
    def load(args, meta, role):
        events.append(('source', role))
        return Model()
    def save(value, path):
        path.write_bytes(b'unit-fixture-not-a-real-model')
    def repair(model, fit, selection, defense, config, device, path, bound, seed):
        assert config == plan['recipes'][defense] and seed == 42
        events.append(('repair', defense))
        save({}, path)
        return model, {'complete':True}
    def fst(model, backbone, fit, selection, device, folder, bound, seed, **config):
        assert config == plan['recipes']['fst'] and config['num_keypoints'] == 14
        events.append(('repair','fst'))
        save({}, folder/'repair.pt'); save({}, folder/'fst_progress.pt')
        return model, {'complete':True}
    def one_pass(model, name):
        events.append(('repair', name))
        return model, {'complete':True}
    args = SimpleNamespace(out=tmp_path, base=tmp_path)
    ids = [['test/x.npy','test/sequence',0]]
    plan = dict(plan_sha256='a'*64, clean_bank={'unit':True}, recipes=r.recipes(),
                test_ids_sha256=r.digest(ids))
    meta = {'checkpoints': {role: {'checkpoint_sha256':role} for role in ('clean','a1')}}
    modules = {
        'run_mmfi':SimpleNamespace(seed_execution=lambda:None),
        'run_piw':SimpleNamespace(clean_loaders=lambda *a: ('fit','select'),
            evaluate=lambda *a: dict(metrics={}, arrays={'pair_ids':np.array(ids,dtype=str)}))}
    monkeypatch.setattr(r, 'validate_result', lambda *a: None)
    monkeypatch.setattr(r, 'load_model', load)
    monkeypatch.setattr(a1_repair_ops, 'train_repair', repair)
    monkeypatch.setattr(a1_repair_ops, '_atomic_save', save)
    monkeypatch.setattr(a1_repair_ops, '_state', lambda _: {})
    monkeypatch.setattr(a1_fst_pose, 'repair_fst', fst)
    monkeypatch.setattr(a1_bnp_pose, 'repair_bnp', lambda model,*a,**k:one_pass(model,'bnp'))
    monkeypatch.setattr(adapter, 'prune_clp', lambda model,*a:one_pass(model,'clp'))
    monkeypatch.setattr(piw_v2_adapter, 'build_trigger', lambda _:None)
    r.run_cell(args, modules, meta, plan, None, None, method)
    assert events[0] == ('source','clean' if method=='clean_reference' else 'a1')
    assert events[1:] == ([] if method not in r.METHODS else [('repair',method)])
    saved = r.read(tmp_path/method/'evaluation.json')
    assert saved['source_checkpoint_sha256'] == ('clean' if method=='clean_reference' else 'a1')
    assert 'clean_bank.json' in saved['artifacts'] if method in ('ft','bnp','fst','nad') else True


def test_cached_cell_does_not_load_or_evaluate(tmp_path, monkeypatch, runtime):
    monkeypatch.setattr(r, 'validate_result', lambda *a: {'status':'complete'})
    monkeypatch.setattr(r, 'load_model', lambda *a:pytest.fail('cached source reload'))
    result = r.run_cell(SimpleNamespace(out=tmp_path), {}, {}, {}, None, None, 'ft')
    assert result['status'] == 'complete'
