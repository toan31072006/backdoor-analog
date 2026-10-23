"""WiPose pilot CPU-only contracts; synthetic TinyPose, not research results."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

import run_wipose_hpeli as runner
import wipose_a1_adapter as attack
import wipose_data


class TinyPose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.2)
        self.head = torch.nn.Linear(3, 36)

    def forward(self, csi):
        features = csi.mean(dim=(2, 3))
        if self.training:
            # All three application RNGs must resume exactly.
            features = self.dropout(features) + (random.random()+np.random.random())*.001
        return self.head(features).reshape(-1, 1, 18, 2), features


class FakeDataset:
    def __init__(self, count, split):
        self.items = [dict(name=f'{split}_frame{i:03}') for i in range(count)]
        rng = np.random.default_rng(123 if split == 'train' else 987)
        self.x = rng.uniform(size=(count, 3, 90, 5)).astype(np.float32)
        self.y = rng.uniform(.1, .8, size=(count, 1, 18, 2)).astype(np.float32)
        self.y[:, :, 5], self.y[:, :, 8] = [.2, .1], [.1, .7]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return dict(csi=self.x[i].copy(), pose=self.y[i].copy(), name=self.items[i]['name'])


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    rng = np.random.default_rng(555)
    action_path = tmp_path/'action.npy'
    np.save(action_path, rng.normal(size=(1, 3, 8, 25, 1)).astype(np.float32))
    datasets = dict(training=FakeDataset(35, 'train'), validation=FakeDataset(3, 'test'))
    inputs = dict(fingerprint='synthetic-input-hash', splits={
        'training': dict(official_count=35, retained_count=35),
        'validation': dict(official_count=3, retained_count=3)})
    monkeypatch.setattr(runner, 'EXPECTED_LIST_COUNTS', {'training': 35, 'validation': 3})
    monkeypatch.setattr(runner, 'source_hashes', lambda: {'fixture.py': 'fixed-test-source'})
    monkeypatch.setattr(wipose_data, 'dataset_audit', lambda root: copy.deepcopy(inputs))
    monkeypatch.setattr(runner, 'datasets_for', lambda cfg: datasets)
    monkeypatch.setattr(runner, 'build_model', TinyPose)
    return dict(home=tmp_path/'home', root=tmp_path/'dataset', out=tmp_path/'out',
                action=action_path, datasets=datasets, inputs=inputs)


def cfg(fixture, output=None):
    return runner.configuration(fixture['home'], output or fixture['out'], 'cpu',
                                fixture['root'], fixture['action'])


def cli(fixture, *extra):
    return ['--data-home', str(fixture['home']), '--outdir', str(fixture['out']),
        '--dataset-root', str(fixture['root']), '--action-npy', str(fixture['action']),
        '--device', 'cpu', *extra]


def manifest(fixture, output=None):
    prepared = runner.prepare(cfg(fixture, output))
    plan = prepared[0]
    Path(plan['cfg']['outdir']).mkdir(parents=True, exist_ok=True)
    runner.util.atomic_json(Path(plan['cfg']['outdir'])/runner.MANIFEST, plan)
    return prepared


def complete(fixture, output=None):
    plan, datasets, trigger, poisoned = manifest(fixture, output)
    for stage in runner.STAGES:
        runner.train_stage(plan, stage, datasets, trigger, poisoned)
    return plan


def tree(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, runner.util.file_sha(p))
            for p in Path(root).rglob('*') if p.is_file()}


def test_frozen_hpe_scratch_recipe_digital_2d_and_not_dtpose(fixture):
    value = cfg(fixture)
    assert value['profile'] == runner.PROFILE
    assert (value['epochs'], value['final_epoch'], value['batch_size']) == (50, 49, 32)
    assert (value['optimizer'], value['lr'], value['momentum'], value['weight_decay']) == ('sgd', .01, .9, 0)
    assert value['input_shape'] == [3, 90, 5]
    assert (value['num_keypoints'], value['num_coor']) == (18, 2)
    assert value['drop_last'] and value['shuffle'] and value['num_workers'] == 0
    assert not value['amp'] and not value['pretrained'] and not value['lr_scheduler']
    assert value['author_reproduction'] is False
    assert 'NOT mm' in value['report_units'] and 'NOT metres' in value['model_units']


def test_dryrun_no_data_model_cuda_or_output(fixture, monkeypatch, capsys):
    def fail(*a, **k):
        pytest.fail('Dry run touched data/model/CUDA')
    monkeypatch.setattr(runner, 'prepare', fail)
    monkeypatch.setattr(runner, 'build_model', fail)
    monkeypatch.setattr(torch.cuda, 'is_available', fail)
    assert runner.main(cli(fixture, '--dry-run')) == 0
    assert not fixture['out'].exists()
    assert 'NO data/model/CUDA/output writes/training' in capsys.readouterr().out


def test_true_fresh_cli_dryrun_does_not_import_tensor_or_dataset_modules(tmp_path):
    root = tmp_path/'nonexistent'
    script = '''import builtins, runpy, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'wipose_data', 'wipose_a1_adapter', 'models'}:
        raise AssertionError('Forbidden dry-run import: '+name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--data-home', sys.argv[2], '--dry-run']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', script,
                             runner.__file__, str(root)],
                            capture_output=True, text=True, timeout=30, check=False,
                            env=dict(os.environ, PYTHONPATH=str(Path(runner.__file__).parent)))
    assert result.returncode == 0, result.stderr
    assert 'NO data/model/CUDA/output writes/training' in result.stdout
    assert not root.exists()


@pytest.mark.parametrize('argument', ['--epochs', '--rho', '--batch-size', '--lr', '--seed', '--jobs-per-gpu'])
def test_no_scientific_or_multi_job_cli_override(fixture, argument):
    with pytest.raises(SystemExit):
        runner.main(cli(fixture, argument, '1'))


@pytest.mark.parametrize('device', ['cuda', 'CUDA:0', 'cuda:-1', 'auto', 'mps'])
def test_no_implicit_or_fallback_device(fixture, device):
    with pytest.raises(ValueError, match='cpu or explicit'):
        runner.configuration(fixture['home'], fixture['out'], device)


@pytest.mark.parametrize('overlap', ['same', 'child', 'parent'])
def test_dataset_output_no_overlap(fixture, overlap):
    dataset = fixture['root']
    output = {'same': dataset, 'child': dataset/'out', 'parent': dataset.parent}[overlap]
    with pytest.raises(ValueError, match='separate'):
        runner.configuration(fixture['home'], output, 'cpu', dataset, fixture['action'])


@pytest.mark.parametrize('change', ['epoch', 'lr', 'extra', 'source', 'runtime', 'planhash', 'root'])
def test_manifest_strict_hash_source_runtime_config_and_location(fixture, change):
    plan, *_ = manifest(fixture)
    value = copy.deepcopy(plan)
    if change == 'epoch':
        value['cfg']['epochs'] = 20
    elif change == 'lr':
        value['cfg']['lr'] = .1
    elif change == 'extra':
        value['cfg']['bogus_field'] = True
    elif change == 'source':
        value['sources']['fixture.py'] = 'wrong'
    elif change == 'runtime':
        value['runtime']['python'] = '0.0.0'
    elif change == 'root':
        value['cfg']['outdir'] = str(fixture['out']/'different')
    if change != 'planhash':
        value['plan_sha256'] = runner.util.object_sha({k: v for k, v in value.items() if k != 'plan_sha256'})
    else:
        value['plan_sha256'] = 'wrong'
    path = fixture['out']/runner.MANIFEST
    runner.util.atomic_json(path, value)
    with pytest.raises(ValueError):
        runner.load_manifest(path)


def test_prepare_readonly_cpu_smoke_and_three_real_paired_poisons(fixture, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CPU preflight probes GPU'))
    assert runner.main(cli(fixture, '--preflight')) == 0
    assert not fixture['out'].exists()
    plan, datasets, trigger, poisoned = runner.prepare(cfg(fixture))
    assert plan['poison']['n_poison'] == 3
    sample = plan['poison']['samples'][0]
    base, modified = datasets['training'][sample['index']], poisoned[sample['index']]
    assert np.any(base['csi'] != modified['csi'])
    assert np.any(base['pose'] != modified['pose'])
    assert set(modified) == {'csi', 'pose'}
    assert sample['dose'] == sample['payload_dose']
    assert trigger.state_record() == plan['operator']


@pytest.mark.parametrize('what', ['raw', 'action', 'poison', 'source', 'runtime'])
def test_changed_prepared_context_refuses_reuse(fixture, what):
    plan, *_ = manifest(fixture)
    actual = copy.deepcopy(plan)
    key = {'raw': 'inputs', 'action': 'operator', 'poison': 'poison',
           'source': 'sources', 'runtime': 'runtime'}[what]
    actual[key] = {'changed': True}
    with pytest.raises(ValueError, match='changed'):
        runner.same_context(plan, actual)


def test_a1_refuses_missing_clean_before_creating_stage(fixture):
    plan, datasets, trigger, poisoned = manifest(fixture)
    with pytest.raises(ValueError, match='Complete Clean'):
        runner.train_stage(plan, 'a1', datasets, trigger, poisoned)
    assert not runner.cell_dir(plan, 'a1').exists()


def test_full_clean_then_a1_50_epochs_replay_cache_and_reports_no_gpu(fixture, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CPU run probes GPU'))
    plan = complete(fixture)
    clean = runner.verify_stage(plan, 'clean')
    result = runner.verify_stage(plan, 'a1')
    assert clean['checkpoint_epoch'] == result['checkpoint_epoch'] == 49
    assert result['measurements']['controls']['d0_exact_identity']
    assert len(result['measurements']['dose_rows']) == 6
    assert result['clean_result_sha256'] == runner.util.object_sha(clean)
    with np.load(runner.cell_dir(plan, 'a1')/'predictions.npz', allow_pickle=False) as saved:
        assert saved['trigger_predictions'].shape == (3, 6, 18, 2)
        assert saved['reference_trigger_predictions'].shape == (3, 6, 18, 2)
        assert np.array_equal(saved['trigger_predictions'][:, 0], saved['predictions'])
        assert np.array_equal(saved['doses'], attack.GRID)
    runner.export_report(plan)
    assert (fixture['out']/'a1_six_doses.csv').is_file()
    assert (fixture['out']/'wipose_pilot.complete.json').is_file()
    text = (fixture['out']/'wipose_pilot.md').read_text(encoding='utf-8')
    assert 'not the proposed DT-Pose model' in text and 'never physical mm' in text
    before = tree(fixture['out'])
    monkeypatch.setattr(runner, 'build_model', lambda: pytest.fail('Cache model construction'))
    assert runner.main(cli(fixture, '--check-only')) == 0
    assert runner.main(cli(fixture, '--status')) == 0
    datasets = fixture['datasets']
    trigger = attack.build_trigger(fixture['action'])
    poisoned = attack.PoisonedWiPoseDataset(datasets['training'], trigger)
    for stage in runner.STAGES:
        runner.train_stage(plan, stage, datasets, trigger, poisoned)
    assert tree(fixture['out']) == before


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_same(left, right)
    else:
        assert a == b


@pytest.mark.parametrize('stage', ['clean', 'a1'])
def test_exact_optimizer_all_rng_and_final_weights_resume(fixture, monkeypatch, stage):
    uninterrupted = complete(fixture)
    baseline = runner.checkpoint_header(runner.cell_dir(uninterrupted, stage)/'checkpoint.pt',
                                        uninterrupted, stage, final=True)
    plan, datasets, trigger, poisoned = manifest(fixture, fixture['out'].parent/'resumed')
    if stage == 'a1':
        runner.train_stage(plan, 'clean', datasets, trigger, poisoned)
    original = runner.util.atomic_checkpoint
    def interrupt(path, blob):
        original(path, blob)
        if blob['stage'] == stage and blob['epoch'] == 17:
            raise RuntimeError('synthetic interruption after atomic epoch17')
    monkeypatch.setattr(runner.util, 'atomic_checkpoint', interrupt)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        runner.train_stage(plan, stage, datasets, trigger, poisoned)
    assert runner.checkpoint_header(runner.cell_dir(plan, stage)/'checkpoint.pt', plan, stage)['epoch'] == 17
    monkeypatch.setattr(runner.util, 'atomic_checkpoint', original)
    runner.train_stage(plan, stage, datasets, trigger, poisoned)
    actual = runner.checkpoint_header(runner.cell_dir(plan, stage)/'checkpoint.pt', plan, stage, final=True)
    for key in ('epoch', 'model', 'optimizer', 'rng', 'execution', 'train_loss'):
        assert_same(actual[key], baseline[key])


@pytest.mark.parametrize('field', ['t1_mpjpe_px', 'target_pck_pct', 'mean_dce_px'])
def test_saved_metrics_recomputed_not_trusted(fixture, field):
    plan = complete(fixture)
    result = runner.verify_stage(plan, 'a1')
    result['measurements']['mainmetrics'][field] += .01
    runner.util.atomic_json(runner.cell_dir(plan, 'a1')/'evaluation.json', result)
    with pytest.raises(ValueError, match='replay'):
        runner.verify_stage(plan, 'a1')


@pytest.mark.parametrize('field,value', [('schema', 9), ('seed', 43), ('units', 'mm')])
def test_result_headers_are_strict(fixture, field, value):
    plan = complete(fixture)
    result = runner.verify_stage(plan, 'a1')
    result[field] = value
    runner.util.atomic_json(runner.cell_dir(plan, 'a1')/'evaluation.json', result)
    with pytest.raises(ValueError, match='binding|schema|units'):
        runner.verify_stage(plan, 'a1')


@pytest.mark.parametrize('field', ['targets', 'truth', 'reference_predictions', 'trigger_predictions', 'doses', 'names'])
def test_npz_targets_controls_order_and_identity_replayed_even_after_rehash(fixture, field):
    plan = complete(fixture)
    folder = runner.cell_dir(plan, 'a1')
    path = folder/'predictions.npz'
    result = runner.verify_stage(plan, 'a1')
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if field == 'names':
        arrays[field] = arrays[field][::-1]
    else:
        arrays[field].flat[0] += .1
    runner._atomic_npz(path, **arrays)
    result['artifacts']['predictions.npz'] = runner.util.file_sha(path)
    runner.util.atomic_json(folder/'evaluation.json', result)
    with pytest.raises(ValueError):
        runner.verify_stage(plan, 'a1')


def test_clean_metrics_reject_complex_input_even_zero_imaginary(fixture):
    truth = fixture['datasets']['validation'].y[:, 0].astype(complex)
    with pytest.raises(ValueError):
        runner.clean_metrics(truth, truth)


def test_completed_clean_checkpoint_or_poison_modification_rejected(fixture):
    plan = complete(fixture)
    path = runner.cell_dir(plan, 'a1')/'poison.json'
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        runner.verify_stage(plan, 'a1')


def test_checkpoint_wrong_stage_epoch_execution_binding_rejected(fixture):
    plan = complete(fixture)
    path = runner.cell_dir(plan, 'clean')/'checkpoint.pt'
    with pytest.raises(ValueError, match='stage|binding'):
        runner.checkpoint_header(path, plan, 'a1', final=True)
    with pytest.raises(ValueError, match='Execution'):
        runner.checkpoint_header(path, plan, 'clean', execution={'changed': True})
    blob = torch.load(path, weights_only=False)
    blob['epoch'] = 48
    runner.util.atomic_checkpoint(path, blob)
    with pytest.raises(ValueError, match='final epoch'):
        runner.checkpoint_header(path, plan, 'clean', final=True)


def test_missing_a1_reports_partial_not_complete(fixture):
    plan, datasets, trigger, poisoned = manifest(fixture)
    runner.train_stage(plan, 'clean', datasets, trigger, poisoned)
    assert runner.main(cli(fixture, '--check-only')) == 1
    assert runner.main(cli(fixture, '--clean-only', '--check-only')) == 0
    runner.export_report(plan)
    text = (fixture['out']/'wipose_pilot.md').read_text(encoding='utf-8')
    assert 'CLEAN ONLY' in text
    assert not (fixture['out']/'wipose_pilot.complete.json').exists()


def test_no_automatic_restart_or_batch_fallback_on_nonfinite_gradient(fixture, monkeypatch):
    plan, datasets, trigger, poisoned = manifest(fixture)
    class BrokenPose(TinyPose):
        def forward(self, csi):
            output, feature = super().forward(csi)
            return output*float('nan'), feature
    monkeypatch.setattr(runner, 'build_model', BrokenPose)
    with pytest.raises(RuntimeError, match='Nonfinite loss'):
        runner.train_stage(plan, 'clean', datasets, trigger, poisoned)
    assert not (runner.cell_dir(plan, 'clean')/'checkpoint.pt').exists()
    assert plan['cfg']['batch_size'] == 32
