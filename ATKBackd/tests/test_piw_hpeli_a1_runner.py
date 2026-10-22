"""Synthetic CPU contracts: not real-data efficacy or CUDA timing tests."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

import run_piw_hpeli_clean as clean
import run_piw_hpeli_a1 as runner
from piw_a1_adapter import build_trigger, PoisonedPairs


class TinyPose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 42)

    def forward(self, csi):
        features = csi.mean(dim=(2, 3))
        return self.head(features).reshape(-1, 1, 14, 3), features


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    rng = np.random.default_rng(8174)
    root = tmp_path / 'dataset'
    expected = {}
    for split, prefix, count in [('train_data', 'S11_01', 13), ('test_data', 'S11_02', 3)]:
        names = [f'{prefix}_{index}' for index in range(count)]
        folder = root / split
        (folder / 'csi_ap').mkdir(parents=True)
        (folder / 'keypoint').mkdir()
        (folder / f'{split}_list.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
        for name in names:
            np.save(folder / 'csi_ap' / f'{name}.npy', rng.normal(size=(3, 180, 20)).astype(np.float32))
            np.save(folder / 'keypoint' / f'{name}.npy', rng.normal(size=(1, 14, 3)).astype(np.float32))
        selected = names[:-1] if split == 'train_data' else names
        expected[split] = (count, count, len(selected), hashlib.sha256(
            ('\n'.join(selected) + '\n').encode()).hexdigest())
    monkeypatch.setattr(clean, 'EXPECTED_IDENTITIES', expected)
    monkeypatch.setattr(clean, 'build_model', TinyPose)
    monkeypatch.setattr(runner, 'build_model', TinyPose)
    action = tmp_path / 'action.npy'
    np.save(action, rng.normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    return dict(root=root, home=tmp_path / 'home', clean=tmp_path / 'clean',
                out=tmp_path / 'a1', action=action)


def setup_clean(fixture):
    manifest = clean.plan(fixture['home'], fixture['clean'], 'cpu', fixture['root'])
    inputs, datasets = clean.audit_inputs(manifest['cfg'])
    manifest.update(inputs=inputs, created_utc='2026-10-23T00:00:00Z')
    manifest['plan_sha256'] = clean.object_sha(manifest)
    folder = clean.cell_dir(manifest)
    folder.mkdir(parents=True)
    clean.atomic_json(fixture['clean'] / clean.MANIFEST, manifest)
    clean.train_worker(manifest, datasets)
    return manifest


def make_manifest(fixture, out=None):
    candidate = runner.plan(fixture['home'], out or fixture['out'], 'cpu', fixture['root'],
                            fixture['clean'], fixture['action'])
    context, datasets, poisoned, trigger = runner.prepare(candidate)
    manifest = dict(candidate, **context, created_utc='2026-10-23T00:00:00Z')
    manifest['plan_sha256'] = runner.object_sha(manifest)
    runner.cell_dir(manifest).mkdir(parents=True)
    runner.atomic_json(Path(manifest['cfg']['outdir']) / runner.MANIFEST, manifest)
    return manifest, datasets, poisoned, trigger


def complete(fixture):
    setup_clean(fixture)
    manifest, datasets, poisoned, trigger = make_manifest(fixture)
    runner.train_worker(manifest, datasets, poisoned, trigger)
    return manifest


def cli(fixture, *extras):
    return ['--data-home', str(fixture['home']), '--outdir', str(fixture['out']),
            '--dataset-root', str(fixture['root']), '--clean-outdir', str(fixture['clean']),
            '--action-npy', str(fixture['action']), '--device', 'cpu', *extras]


def tree(root):
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns, clean.file_sha(path))
            for path in root.rglob('*') if path.is_file()}


def test_frozen_recipe_and_dataset_specific_metric(fixture):
    cfg = runner.configuration(fixture['home'], fixture['out'], 'cpu', fixture['root'],
                               fixture['clean'], fixture['action'])
    assert (cfg['epochs'], cfg['final_epoch'], cfg['batch_size']) == (200, 199, 32)
    assert (cfg['lr'], cfg['weight_decay'], cfg['eps'], cfg['attack_eps']) == (.001, .01, 1e-8, .185)
    assert cfg['pivot'] == 7 and cfg['target_joints'] == [11, 13]
    assert cfg['target_pck_ref'] == [6, 4] and cfg['rho'] == .1
    assert cfg['eval_doses'] == [0., .2, .4, .6, .8, 1.]
    assert not cfg['amp'] and not cfg['lr_scheduler'] and not cfg['pretrained']


def test_dry_run_no_action_clean_data_model_cuda_or_writes(fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('dry run accessed data/model/CUDA')
    monkeypatch.setattr(runner, 'prepare', forbidden)
    monkeypatch.setattr(runner, 'build_model', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    assert runner.main(cli(fixture, '--dry-run')) == 0
    assert not fixture['out'].exists()


def test_true_cli_dry_run_on_missing_inputs(fixture):
    result = subprocess.run([sys.executable, '-X', 'utf8', str(Path(runner.__file__)),
        *cli(fixture, '--dry-run')], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'no data/model/CUDA/output writes/training' in result.stdout
    assert not fixture['out'].exists()


@pytest.mark.parametrize('overlap', ['same', 'child', 'parent', 'dataset'])
def test_output_reference_and_raw_roots_never_overlap(fixture, overlap):
    reference = fixture['clean']
    out = {'same': reference, 'child': reference / 'x', 'parent': reference.parent,
           'dataset': fixture['root'] / 'x'}[overlap]
    with pytest.raises(ValueError, match='separate'):
        runner.configuration(fixture['home'], out, 'cpu', fixture['root'], reference, fixture['action'])


@pytest.mark.parametrize('argument', ['--epochs', '--rho', '--batch-size', '--learning-rate'])
def test_no_scientific_override_cli(fixture, argument):
    with pytest.raises(SystemExit):
        runner.main(cli(fixture, argument, '1'))


def test_missing_or_incomplete_clean_refuses_a1(fixture):
    candidate = runner.plan(fixture['home'], fixture['out'], 'cpu', fixture['root'],
                            fixture['clean'], fixture['action'])
    with pytest.raises(FileNotFoundError):
        runner.prepare(candidate)
    assert not fixture['out'].exists()


def test_preflight_cpu_readonly_and_actual_nonzero_poison(fixture, monkeypatch):
    setup_clean(fixture)
    before = tree(fixture['clean'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('preflight CUDA probe'))
    monkeypatch.setattr(clean, 'cpu_model_smoke', lambda: {'finite': True})
    assert runner.main(cli(fixture, '--preflight')) == 0
    assert tree(fixture['clean']) == before and not fixture['out'].exists()
    candidate = runner.plan(fixture['home'], fixture['out'], 'cpu', fixture['root'],
                            fixture['clean'], fixture['action'])
    context, datasets, poisoned, trigger = runner.prepare(candidate)
    assert len(poisoned) == 12
    assert context['poison']['n_poison'] == 1
    entries = context['poison']['samples']
    index = entries[0]['index']
    clean_sample, attack_sample = datasets['training'][index], poisoned[index]
    assert set(attack_sample) == {'csi', 'pose'}
    assert not np.array_equal(clean_sample['csi'][:, :90], attack_sample['csi'][:, :90])
    assert np.array_equal(clean_sample['csi'][:, 90:], attack_sample['csi'][:, 90:])
    assert not np.array_equal(clean_sample['pose'], attack_sample['pose'])


def test_full_200_epoch_artifacts_metrics_control_and_clean_immutable(fixture, monkeypatch):
    clean_manifest = setup_clean(fixture)
    before = tree(fixture['clean'])
    manifest, datasets, poisoned, trigger = make_manifest(fixture)
    runner.train_worker(manifest, datasets, poisoned, trigger)
    assert tree(fixture['clean']) == before
    result = runner.verify_complete(manifest)
    assert result['checkpoint_epoch'] == 199
    assert len(result['measurements']['dose_rows']) == 6
    assert set(result['measurements']['mainmetrics']) == {
        't1_mpjpe_mm', 'target_pck_pct', 'clean_mpjpe_mm', 'clean_pampjpe_mm', 'mean_dce_mm'}
    assert result['reference_reproduction']['max_joint_difference_mm'] == 0
    assert manifest['poison']['n_poison'] == 1
    with np.load(runner.cell_dir(manifest) / 'a1_predictions.npz') as value:
        assert value['trigger_predictions'].shape == (3, 6, 14, 3)
        assert np.array_equal(value['trigger_predictions'][:, 0], value['predictions'])
        assert np.array_equal(value['reference_trigger_predictions'][:, 0], value['reference_predictions'])
    assert clean.verify_complete(clean_manifest) is not None
    report = (fixture['out'] / 'a1_pilot.md').read_text()
    assert 'not pose-level ASR' in report and 'Clean reference | 42 | -- | --' in report
    before = tree(fixture['out'])
    monkeypatch.setattr(runner, 'build_model', lambda: pytest.fail('cache model construction'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('cache CUDA probe'))
    assert runner.main(cli(fixture, '--check-only')) == 0
    assert runner.main(cli(fixture, '--status')) == 0
    assert before == tree(fixture['out'])


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def test_exact_200_epoch_rng_optimizer_resume_matches_uninterrupted(fixture, monkeypatch):
    setup_clean(fixture)
    manifest, datasets, poisoned, trigger = make_manifest(fixture)
    runner.train_worker(manifest, datasets, poisoned, trigger)
    baseline = torch.load(runner.cell_dir(manifest) / 'checkpoint.pt', weights_only=False)
    resumed, datasets, poisoned, trigger = make_manifest(fixture, fixture['out'].parent / 'resumed')
    original = runner.atomic_checkpoint
    def interrupt(path, blob):
        original(path, blob)
        if blob['epoch'] == 97:
            raise RuntimeError('simulated process interruption')
    monkeypatch.setattr(runner, 'atomic_checkpoint', interrupt)
    with pytest.raises(RuntimeError, match='simulated'):
        runner.train_worker(resumed, datasets, poisoned, trigger)
    monkeypatch.setattr(runner, 'atomic_checkpoint', original)
    runner.train_worker(resumed, datasets, poisoned, trigger)
    actual = torch.load(runner.cell_dir(resumed) / 'checkpoint.pt', weights_only=False)
    for key in ('epoch', 'model', 'optimizer', 'rng', 'execution', 'train_loss'):
        assert_equal(actual[key], baseline[key])


@pytest.mark.parametrize('target', ['clean_reference', 'raw', 'action', 'poison'])
def test_changed_reference_raw_action_or_poison_refuses_reuse(fixture, target):
    manifest = complete(fixture)
    if target == 'clean_reference':
        path = fixture['clean'] / 'clean_result.json'
        path.write_text(path.read_text() + '\n')
        with pytest.raises(ValueError, match='Clean reference changed'):
            runner.verify_complete(manifest)
    elif target == 'poison':
        path = runner.cell_dir(manifest) / 'poison.json'
        path.write_text('{}')
        with pytest.raises(ValueError, match='artifact fingerprint'):
            runner.verify_complete(manifest)
    else:
        path = fixture['action'] if target == 'action' else fixture['root'] / 'train_data/csi_ap/S11_01_9.npy'
        value = np.load(path); value.flat[0] += .1; np.save(path, value)
        # A raw-data change is rejected even earlier by the authenticated
        # Clean input binding; action changes reach the A1 context comparison.
        with pytest.raises(ValueError, match='binding changed|fingerprint changed'):
            context, *_ = runner.prepare(manifest)
            runner._same_context(manifest, context)


@pytest.mark.parametrize('field', ['t1_mpjpe_mm', 'target_pck_pct', 'mean_dce_mm'])
def test_recorded_metrics_replayed_not_trusted(fixture, field):
    manifest = complete(fixture)
    result = runner.verify_complete(manifest)
    result['measurements']['mainmetrics'][field] += .001
    runner.atomic_json(runner.cell_dir(manifest) / 'evaluation.json', result)
    runner.atomic_json(fixture['out'] / 'a1_result.json', result)
    with pytest.raises(ValueError, match='do not reproduce'):
        runner.verify_complete(manifest)


@pytest.mark.parametrize('name', ['targets', 'truth', 'reference_predictions', 'trigger_predictions'])
def test_npz_geometry_reference_and_identity_proofs_replayed(fixture, name):
    manifest = complete(fixture)
    path = runner.cell_dir(manifest) / 'a1_predictions.npz'
    with np.load(path) as payload:
        data = {key: payload[key].copy() for key in payload.files}
    data[name].flat[0] += .1
    with path.open('wb') as stream:
        np.savez_compressed(stream, **data)
    result = json.loads((fixture['out'] / 'a1_result.json').read_text())
    result['artifacts']['a1_predictions.npz'] = runner.file_sha(path)
    runner.atomic_json(runner.cell_dir(manifest) / 'evaluation.json', result)
    runner.atomic_json(fixture['out'] / 'a1_result.json', result)
    with pytest.raises(ValueError):
        runner.verify_complete(manifest)


@pytest.mark.parametrize('missing', ['top', 'cell'])
def test_partial_result_commit_can_resume_evaluation(fixture, missing):
    manifest = complete(fixture)
    (fixture['out'] / 'a1_result.json' if missing == 'top' else
     runner.cell_dir(manifest) / 'evaluation.json').unlink()
    assert runner.verify_complete(manifest) is None
    context, datasets, poisoned, trigger = runner.prepare(manifest)
    runner.train_worker(manifest, datasets, poisoned, trigger)
    assert runner.verify_complete(manifest) is not None


def test_checkpoint_atomic_failure_keeps_previous(fixture, monkeypatch):
    path = fixture['out'] / 'checkpoint.pt'
    path.parent.mkdir()
    runner.atomic_checkpoint(path, {'epoch': 4})
    original = path.read_bytes()
    def fail(*args):
        raise PermissionError('simulated rename failure')
    monkeypatch.setattr(clean, 'atomic_replace', fail)
    with pytest.raises(PermissionError):
        runner.atomic_checkpoint(path, {'epoch': 5})
    assert path.read_bytes() == original
    assert not path.with_name('checkpoint.pt.saving').exists()


def test_real_hpeli_cpu_backward_piw_geometry(monkeypatch):
    monkeypatch.setattr(clean, 'build_model', runner.build_model)
    report = clean.cpu_model_smoke()
    assert report['output_shape'] == [2, 1, 14, 3] and report['finite']
