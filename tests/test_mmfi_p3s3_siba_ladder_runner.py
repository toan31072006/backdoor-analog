"""CPU/synthetic contracts; no dataset, CUDA, training efficacy, or remote claim."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ATKBackd'))
runner = importlib.import_module('run_mmfi_p3s3_siba_ladder_4090')


@pytest.fixture
def plan(tmp_path):
    return runner.build_plan(tmp_path, tmp_path / 'runs' / runner.PROFILE)


def test_exact_common_victim_recipe(plan):
    import run_mmfi_p3_splits as p3
    reference = p3._configuration(Path(plan['home']), 'cuda:0', 's3', 'a1')
    for name in runner.METHODS:
        cfg = plan['cfg'][name]
        for key in ('model', 'seed', 'mmfi_protocol', 'mmfi_setting', 'mmfi_split_seed',
                'mmfi_random_ratio', 'optimizer', 'lr', 'momentum', 'weight_decay',
                'batch_size', 'num_workers', 'epochs', 'victim_epochs', 'rho', 'pivot',
                'target_joints', 'theta_max_deg', 'payload_axis', 'dose_coupling',
                'dose_min', 'dose_max', 'dose_grid', 'poison_select', 'pretrained'):
            assert cfg[key] == reference[key], key
    assert plan['cfg']['clean_surrogate']['rho'] == 0
    assert plan['cfg']['siba']['lr'] == .001
    assert plan['contract']['scratch_surrogate']
    assert plan['contract']['scratch_victims']
    assert not plan['contract']['warm_start']
    assert not plan['contract']['test_used_for_search']


def test_plan_json_roundtrip_digest_and_counts(plan):
    assert json.loads(json.dumps(plan)) == plan
    unsigned = deepcopy(plan)
    claimed = unsigned.pop('plan_sha256')
    assert runner.digest(unsigned) == claimed
    assert plan['contract']['counts'] == {'training': 240570, 'test': 80190}
    assert plan['contract']['n_poison'] == 24057
    assert plan['operators']['ladder']['low_band'] == [49, 4]
    assert plan['operators']['siba']['eps'] == 8 / 255


@pytest.mark.parametrize('relative', ['runs', 'runs/mmfi_p3_hpeli_s42_v1',
    'runs/mmfi_p3s3_three_attacks_s42_4090_v1', 'other/mmfi_p3s3_siba_ladder_test',
    'runs/mmfi_p3s3_siba_ladder_test/nested'])
def test_rejects_historical_or_broad_outputs(tmp_path, relative):
    with pytest.raises(ValueError, match='NEW'):
        runner.build_plan(tmp_path, tmp_path / relative)


@pytest.mark.parametrize('device,jobs', [('cpu', 1), ('cuda:1', 1), ('cuda:0', 2), ('cuda:0', True)])
def test_rejects_cpu_fallback_or_parallelism(tmp_path, device, jobs):
    with pytest.raises(ValueError, match='serial'):
        runner.build_plan(tmp_path, tmp_path/'runs'/runner.PROFILE, device, jobs)


def test_unknown_method_rejected(tmp_path):
    with pytest.raises(ValueError):
        runner.configuration(tmp_path, 'a1')


def test_banks_are_fixed_unique_disjoint_train_indices():
    a = runner.bank_selection(240570)
    assert a == runner.bank_selection(240570)
    assert len(a['fit']) == 1024 and len(a['score']) == 256
    assert set(a['fit']).isdisjoint(a['score'])
    assert min(a['fit'] + a['score']) >= 0
    assert max(a['fit'] + a['score']) < 240570


def test_banks_too_small_rejected():
    with pytest.raises(ValueError):
        runner.bank_selection(100)


@pytest.mark.parametrize('with_person', [True, False])
def test_tensor_target_matches_native_payload(with_person):
    from attack.payload import make_target_pose
    rng = np.random.default_rng(42)
    shape = (7, 1, 17, 3) if with_person else (7, 17, 3)
    poses = rng.normal(size=shape).astype(np.float32)
    doses = np.asarray([0, .2, .4, .6, .8, 1., .37], dtype=np.float32)
    actual = runner.target_fn(torch.from_numpy(poses), torch.from_numpy(doses)).numpy()
    expected = np.stack([make_target_pose(p, 1, float(d), theta_max=np.deg2rad(40),
        mode='linear', axis=(0, 0, 1), target_joints=[2, 3]) for p, d in zip(poses, doses)])
    np.testing.assert_allclose(actual, expected, atol=8e-7)
    np.testing.assert_array_equal(actual[0], poses[0])
    np.testing.assert_array_equal(actual[..., [0, 1, *range(4, 17)], :],
                                  poses[..., [0, 1, *range(4, 17)], :])


@pytest.mark.parametrize('dose', [float('nan'), -1., 1.1])
def test_bad_target_doses_fail(dose):
    with pytest.raises(ValueError):
        runner.target_fn(torch.zeros(1, 1, 17, 3), torch.tensor([dose]))


def test_load_plan_refuses_mutated_source_runtime(plan, tmp_path):
    root = Path(plan['outdir'])
    runner.write(root/'plan.json', plan)
    changed = deepcopy(plan)
    changed['runtime']['python'] = 'different'
    with pytest.raises(ValueError, match='Source/config/runtime'):
        runner.load_plan(changed)


def make_blob(plan, cell, env):
    # Synthetic binding files, not real full-data preflight receipts.
    root = Path(plan['outdir'])
    runner.write(root/'inputs.json', {'fixture': True})
    runner.write(root/'poison.json', {'fixture': True})
    return dict(epoch=49, cfg=plan['cfg'][cell], binding=runner.cell_binding(plan, cell, env),
        model={'weight': torch.ones(1)}, optimizer={'param_groups': [dict(lr=.001,
            momentum=.9, weight_decay=0.)], 'state': {0: {'momentum_buffer': torch.ones(1)}}},
        rng={'python': [1], 'numpy': [1], 'torch': torch.ones(1), 'cuda': [torch.ones(1)]})


@pytest.fixture
def blob(plan):
    return make_blob(plan, 'clean_surrogate', {'test': 'cpu-only-contract'})


def test_checkpoint_contract_accepts_owned_final49(plan, blob):
    runner.validate_checkpoint(blob, plan, 'clean_surrogate', {'test': 'cpu-only-contract'})


@pytest.mark.parametrize('key,value', [('epoch', 50), ('epoch', True), ('epoch', -2),
    ('model', {}), ('optimizer', {}), ('rng', {}), ('binding', {})])
def test_checkpoint_rejects_partial_foreign_state(plan, blob, key, value):
    blob[key] = value
    with pytest.raises(ValueError):
        runner.validate_checkpoint(blob, plan, 'clean_surrogate', {'test': 'cpu-only-contract'})


@pytest.mark.parametrize('key,value', [('lr', .01), ('momentum', .8), ('weight_decay', .1), ('betas', [.9, .999])])
def test_checkpoint_recipe_cannot_drift(plan, blob, key, value):
    blob['optimizer']['param_groups'][0][key] = value
    with pytest.raises(ValueError, match='optimizer'):
        runner.validate_checkpoint(blob, plan, 'clean_surrogate', {'test': 'cpu-only-contract'})


def test_checkpoint_requires_cuda_rng_and_finite_weights(plan, blob):
    blob['rng']['cuda'] = None
    with pytest.raises(ValueError):
        runner.validate_checkpoint(blob, plan, 'clean_surrogate', {'test': 'cpu-only-contract'})
    blob['rng']['cuda'] = [torch.ones(1)]
    blob['model']['weight'][0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        runner.validate_checkpoint(blob, plan, 'clean_surrogate', {'test': 'cpu-only-contract'})


def test_only_fixed_final49_usable_for_search(plan, blob):
    blob['epoch'] = 48
    path = Path(plan['outdir'])/'clean_surrogate/checkpoint.pt'
    runner.save_checkpoint(path, blob)
    with pytest.raises(ValueError, match='final49'):
        runner.load_cell(plan, 'clean_surrogate', {'test': 'cpu-only-contract'}, final=True)


def test_atomic_save_keeps_interrupted_checkpoint(tmp_path):
    path = tmp_path/'checkpoint.pt'
    runner.save_checkpoint(path, {'epoch': 7})
    (tmp_path/'checkpoint.pt.saving').write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='Unfinished'):
        runner.save_checkpoint(path, {'epoch': 8})
    assert torch.load(path, weights_only=False)['epoch'] == 7


def test_npz_replay_checks_content_not_zip_timestamps(tmp_path):
    path = tmp_path/'samples.npz'
    arrays = dict(x=np.ones((2, 3)), y=np.arange(3))
    runner.save_npz(path, arrays)
    before = runner.sha(path)
    runner.save_npz(path, arrays)
    assert runner.sha(path) == before
    with pytest.raises(ValueError, match='partial'):
        runner.save_npz(path, dict(x=np.zeros((2, 3))))


def test_status_no_model_cuda_or_training(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    monkeypatch.setattr(runner, 'build_plan', lambda *a: pytest.fail('status must not build models/plan'))
    runner.main(['--data-home', str(tmp_path), '--mode', 'status'])
    assert 'READ_ONLY' in capsys.readouterr().out


def test_inspect_no_data_audit_cuda_or_training(tmp_path, monkeypatch, capsys):
    for name in ('audit', 'execution', 'require_idle', 'train', 'launch'):
        monkeypatch.setattr(runner, name, lambda *a: pytest.fail('inspect must be read-only'))
    runner.main(['--data-home', str(tmp_path), '--mode', 'inspect'])
    assert 'INSPECT_ONLY' in capsys.readouterr().out


def test_prepare_never_enters_full_training_or_search(plan, monkeypatch):
    plan = runner.build_plan(Path(plan['home']), Path(plan['home'])/'runs'/'mmfi_p3s3_siba_ladder_available_prepare',
        backtime_policy='adaptive', selected_methods=('siba', 'backtime'))
    # Existing preparation receipt path takes the checked short branch.
    root = Path(plan['outdir'])
    root.mkdir(parents=True)
    (root/'gpu_gate.json').write_text('{}')
    monkeypatch.setattr(runner, 'audit', lambda p: None)
    monkeypatch.setattr(runner, 'require_idle', lambda: None)
    monkeypatch.setattr(runner, 'execution', lambda: {})
    called = []
    monkeypatch.setattr(runner, 'gate_check', lambda p: called.append('checked'))
    monkeypatch.setattr(runner, 'train', lambda *a: pytest.fail('prepare must not train'))
    monkeypatch.setattr(runner, 'fit_trigger', lambda *a: pytest.fail('prepare must not search'))
    runner.prepare(plan)
    assert called == ['checked']


def test_launch_serial_sequence_and_no_warm_start(plan, monkeypatch):
    from contextlib import nullcontext
    plan = runner.build_plan(Path(plan['home']), Path(plan['home'])/'runs'/'mmfi_p3s3_siba_ladder_available_launch',
        backtime_policy='adaptive', selected_methods=('siba', 'backtime'))
    calls = []
    monkeypatch.setattr(runner, 'gate_check', lambda p: {})
    monkeypatch.setattr(runner, 'require_idle', lambda: None)
    monkeypatch.setattr(runner, 'log_to', lambda *a: nullcontext())
    monkeypatch.setattr(runner, 'completed', lambda *a: None)
    monkeypatch.setattr(runner, 'train', lambda p, c, e: calls.append(('train', c)))
    monkeypatch.setattr(runner, 'fit_trigger', lambda p, c, e: calls.append(('fit', c)))
    monkeypatch.setattr(runner, 'learned_smoke', lambda p, c, e: calls.append(('smoke', c)))
    monkeypatch.setattr(runner, 'evaluate', lambda p, c, e: calls.append(('eval', c)))
    monkeypatch.setattr(runner, 'export', lambda *a: calls.append(('export', 'all')))
    runner.launch(plan)
    assert calls == [('train', 'clean_surrogate'), ('fit', 'siba'), ('smoke', 'siba'),
        ('train', 'siba'), ('eval', 'siba'), ('fit', 'backtime'), ('smoke', 'backtime'),
        ('train', 'backtime'), ('eval', 'backtime'), ('export', 'all')]


def test_failed_search_stops_queue_no_retry(plan, monkeypatch):
    from contextlib import nullcontext
    plan = runner.build_plan(Path(plan['home']), Path(plan['home'])/'runs'/'mmfi_p3s3_siba_ladder_available_fail',
        backtime_policy='adaptive', selected_methods=('siba', 'backtime'))
    called = []
    monkeypatch.setattr(runner, 'gate_check', lambda p: {})
    monkeypatch.setattr(runner, 'require_idle', lambda: None)
    monkeypatch.setattr(runner, 'log_to', lambda *a: nullcontext())
    monkeypatch.setattr(runner, 'completed', lambda *a: None)
    monkeypatch.setattr(runner, 'train', lambda p, c, e: called.append(c))
    def fail(*a):
        called.append('search_failed')
        raise RuntimeError('synthetic failure')
    monkeypatch.setattr(runner, 'fit_trigger', fail)
    with pytest.raises(RuntimeError, match='synthetic'):
        runner.launch(plan)
    assert called == ['clean_surrogate', 'search_failed']
