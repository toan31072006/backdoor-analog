"""CPU/synthetic tests; no full PiW3D dataset or CUDA experiment claim."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

import run_piw_hpeli_clean as runner


class TinyPose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 42)

    def forward(self, csi):
        features = csi.mean(dim=(2, 3))
        return self.head(features).reshape(-1, 1, 14, 3), features


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    expected = {}
    rng = np.random.default_rng(744)
    for split, prefix, count in [('train_data', 'S11_01', 5), ('test_data', 'S11_02', 3)]:
        names = [f'{prefix}_{index}' for index in range(count)]
        directory = root / split
        (directory / 'csi_ap').mkdir(parents=True)
        (directory / 'keypoint').mkdir()
        (directory / f'{split}_list.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
        for name in names:
            np.save(directory / 'csi_ap' / f'{name}.npy',
                    rng.normal(size=(3, 180, 20)).astype(np.float32))
            np.save(directory / 'keypoint' / f'{name}.npy',
                    rng.normal(size=(1, 14, 3)).astype(np.float32))
        selected = names[:-1] if split == 'train_data' else names
        expected[split] = (count, count, len(selected), hashlib.sha256(
            ('\n'.join(selected) + '\n').encode()).hexdigest())
    monkeypatch.setattr(runner, 'EXPECTED_IDENTITIES', expected)
    return dict(root=root, data_home=tmp_path / 'home', out=tmp_path / 'output')


def manifest_for(fixture, out=None):
    result = runner.plan(fixture['data_home'], out or fixture['out'], 'cpu', fixture['root'])
    inputs, datasets = runner.audit_inputs(result['cfg'])
    result.update(inputs=inputs, created_utc='2026-10-23T00:00:00Z')
    result['plan_sha256'] = runner.object_sha(result)
    root = Path(result['cfg']['outdir'])
    root.mkdir(parents=True)
    (root / runner.MANIFEST).write_text(json.dumps(result), encoding='utf-8')
    runner.cell_dir(result).mkdir(parents=True)
    return result, datasets


def tiny_complete(fixture, monkeypatch, out=None):
    manifest, datasets = manifest_for(fixture, out)
    monkeypatch.setattr(runner, 'build_model', TinyPose)
    runner.train_worker(manifest, datasets)
    return manifest


def cli(fixture, *extras):
    return ['--data-home', str(fixture['data_home']), '--dataset-root', str(fixture['root']),
            '--outdir', str(fixture['out']), '--device', 'cpu', *extras]


def test_frozen_200_epoch_clean_recipe(fixture):
    cfg = runner.configuration(fixture['data_home'], fixture['out'], 'cpu', fixture['root'])
    assert cfg['epochs'] == 200 and cfg['final_epoch'] == 199
    assert (cfg['batch_size'], cfg['lr'], cfg['weight_decay']) == (32, .001, .01)
    assert cfg['seed'] == 42 and cfg['num_workers'] == 0
    assert cfg['victim_loss'] == 'ordinary MPJPE'
    assert not cfg['pretrained'] and not cfg['lr_scheduler'] and not cfg['amp']
    assert not any(key in cfg for key in ('action_npy', 'trigger', 'pivot', 'rho'))
    assert cfg['clean_pck_ref'] == [6, 4]


def test_dry_run_no_data_gpu_model_or_output(fixture, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('Dry run touched data/model/GPU')
    monkeypatch.setattr(runner, 'audit_inputs', forbidden)
    monkeypatch.setattr(runner, 'build_model', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    assert runner.main(cli(fixture, '--dry-run')) == 0
    assert not fixture['out'].exists()
    assert 'NO data/GPU/output writes/training; NOT A1' in capsys.readouterr().out


@pytest.mark.parametrize('device', ['auto', 'cuda', 'cuda:-1', 'mps', 'cpu:0'])
def test_no_automatic_device_or_recipe_fallback(fixture, device):
    with pytest.raises(ValueError, match='Device'):
        runner.configuration(fixture['data_home'], fixture['out'], device, fixture['root'])


@pytest.mark.parametrize('relative', ['', 'child'])
def test_output_never_inside_dataset(fixture, relative):
    with pytest.raises(ValueError, match='separate'):
        runner.configuration(fixture['data_home'], fixture['root'] / relative, 'cpu', fixture['root'])


def test_official_order_next_frame_and_readonly_hash_audit(fixture):
    paths = sorted(fixture['root'].rglob('*'))
    before = {str(path): runner.file_sha(path) for path in paths if path.is_file()}
    cfg = runner.configuration(fixture['data_home'], fixture['out'], 'cpu', fixture['root'])
    inputs, datasets = runner.audit_inputs(cfg)
    assert len(datasets['training']) == 4 and len(datasets['validation']) == 3
    assert inputs['splits']['train_data']['ordered_names'] == [f'S11_01_{i}' for i in range(4)]
    assert len(inputs['splits']['train_data']['files']) == 10
    assert before == {str(path): runner.file_sha(path) for path in paths if path.is_file()}
    assert not fixture['out'].exists()
    sample = datasets['training'][0]
    assert sample['pose'].shape == (1, 14, 3)
    assert sample['csi'].shape == (3, 180, 20)
    assert np.isfinite(sample['csi']).all()
    assert sample['csi'].min() >= 0 and sample['csi'].max() <= 1


@pytest.mark.parametrize('case', ['missing', 'nan', 'wrong_csi', 'wrong_pose', 'multiperson_pose', 'object'])
def test_all_files_validated_not_only_first_sample(fixture, case):
    csi = fixture['root'] / 'train_data/csi_ap/S11_01_3.npy'
    pose = fixture['root'] / 'train_data/keypoint/S11_01_3.npy'
    if case == 'missing':
        csi.unlink()
    elif case == 'nan':
        values = np.load(csi); values[2, 179, 19] = np.nan; np.save(csi, values)
    elif case == 'wrong_csi':
        np.save(csi, np.zeros((3, 114, 10), np.float32))
    elif case == 'wrong_pose':
        np.save(pose, np.zeros((1, 17, 3), np.float32))
    elif case == 'multiperson_pose':
        np.save(pose, np.zeros((2, 14, 3), np.float32))
    else:
        np.save(csi, np.asarray([{}], dtype=object))
    cfg = runner.configuration(fixture['data_home'], fixture['out'], 'cpu', fixture['root'])
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.audit_inputs(cfg)


@pytest.mark.parametrize('bad_name', ['../escape', 'S11_01_0.npy', 'S11_01_x', 'S11/01/1'])
def test_names_cannot_escape_dataset(fixture, bad_name):
    path = fixture['root'] / 'train_data/train_data_list.txt'
    path.write_text(bad_name + '\n')
    cfg = runner.configuration(fixture['data_home'], fixture['out'], 'cpu', fixture['root'])
    with pytest.raises(ValueError, match='identity'):
        runner.audit_inputs(cfg)


def test_duplicate_official_identity_rejected(fixture):
    path = fixture['root'] / 'train_data/train_data_list.txt'
    path.write_text(path.read_text() + 'S11_01_0\n')
    cfg = runner.configuration(fixture['data_home'], fixture['out'], 'cpu', fixture['root'])
    with pytest.raises(ValueError, match='duplicate'):
        runner.audit_inputs(cfg)


def test_subset_or_reordered_official_lists_rejected(fixture):
    path = fixture['root'] / 'train_data/train_data_list.txt'
    path.write_text('\n'.join(reversed(path.read_text().splitlines())) + '\n')
    cfg = runner.configuration(fixture['data_home'], fixture['out'], 'cpu', fixture['root'])
    with pytest.raises(ValueError, match='frozen official'):
        runner.audit_inputs(cfg)


def test_real_hpeli_piw_forward_backward_cpu():
    report = runner.cpu_model_smoke()
    assert report['input_shape'] == [2, 3, 180, 20]
    assert report['output_shape'] == [2, 1, 14, 3]
    assert report['parameters'] > 800000 and report['finite']


def test_preflight_readonly_and_cpu_only(fixture, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('Preflight CUDA probe'))
    assert runner.main(cli(fixture, '--preflight')) == 0
    assert not fixture['out'].exists()


def test_tiny_200_epoch_final_eval_metrics_artifacts_and_no_model_cache(fixture, monkeypatch):
    manifest = tiny_complete(fixture, monkeypatch)
    result = runner.verify_complete(manifest)
    assert result['checkpoint_epoch'] == 199
    assert result['counts'] == dict(train_data=4, test_data=3)
    assert set(result['metrics']) == {'clean_mpjpe_mm', 'clean_pampjpe_mm', 'clean_pck_pct'}
    assert 'NOT_A1' in result['status']
    report = (fixture['out'] / 'clean_pilot.md').read_text()
    assert 'not Target-PCK or ASR' in report
    assert 'no across-seed SD/CI' in report
    monkeypatch.setattr(runner, 'build_model', lambda: pytest.fail('Completed cache constructed model'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('Completed cache CUDA probe'))
    before = {str(p): (p.stat().st_mtime_ns, runner.file_sha(p))
              for p in fixture['out'].rglob('*') if p.is_file()}
    assert runner.main(cli(fixture, '--check-only')) == 0
    assert runner.main(cli(fixture, '--status')) == 0
    assert before == {str(p): (p.stat().st_mtime_ns, runner.file_sha(p))
                      for p in fixture['out'].rglob('*') if p.is_file()}


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_exact_epoch_rng_optimizer_resume_matches_uninterrupted(fixture, monkeypatch):
    monkeypatch.setattr(runner, 'build_model', TinyPose)
    full, datasets = manifest_for(fixture)
    runner.train_worker(full, datasets)
    uninterrupted = torch.load(runner.cell_dir(full) / 'checkpoint.pt', weights_only=False)
    resumed, datasets = manifest_for(fixture, fixture['out'].parent / 'resumed')
    original = runner.atomic_checkpoint

    def interrupt_after_epoch97(path, blob):
        original(path, blob)
        if blob['epoch'] == 97:
            raise RuntimeError('Simulated process interruption after atomic epoch')

    monkeypatch.setattr(runner, 'atomic_checkpoint', interrupt_after_epoch97)
    with pytest.raises(RuntimeError, match='Simulated'):
        runner.train_worker(resumed, datasets)
    assert torch.load(runner.cell_dir(resumed) / 'checkpoint.pt', weights_only=False)['epoch'] == 97
    monkeypatch.setattr(runner, 'atomic_checkpoint', original)
    runner.train_worker(resumed, datasets)
    checkpoint = torch.load(runner.cell_dir(resumed) / 'checkpoint.pt', weights_only=False)
    for key in ('model', 'optimizer', 'rng', 'epoch', 'train_loss', 'execution'):
        assert_tree_equal(checkpoint[key], uninterrupted[key])
    assert runner.verify_complete(resumed)['metrics'] == runner.verify_complete(full)['metrics']


def test_atomic_checkpoint_failure_preserves_prior_epoch(tmp_path, monkeypatch):
    path = tmp_path / 'checkpoint.pt'
    runner.atomic_checkpoint(path, dict(epoch=17))
    before = path.read_bytes()
    def fail_save(blob, path):
        Path(path).write_bytes(b'broken temporary checkpoint')
        raise OSError('disk interrupted')
    monkeypatch.setattr(torch, 'save', fail_save)
    with pytest.raises(OSError, match='disk interrupted'):
        runner.atomic_checkpoint(path, dict(epoch=18))
    assert path.read_bytes() == before
    assert not path.with_name(path.name + '.saving').exists()


@pytest.mark.parametrize('field', ['cfg', 'sources', 'runtime', 'fingerprint', 'input_fingerprint'])
def test_manifest_recomputed_not_trusting_stored_hash(fixture, field):
    manifest, _ = manifest_for(fixture)
    if field == 'cfg':
        manifest['cfg']['epochs'] = 3
    elif field == 'sources':
        manifest['sources']['models/hpeli.py'] = 'x'
    elif field == 'runtime':
        manifest['runtime']['python'] = '0.0.0'
    elif field == 'input_fingerprint':
        manifest['inputs']['splits']['test_data']['ordered_names'].reverse()
    else:
        manifest['plan_sha256'] = 'x'
    if field != 'fingerprint':
        manifest['plan_sha256'] = runner.object_sha({k: v for k, v in manifest.items() if k != 'plan_sha256'})
    with pytest.raises(ValueError):
        runner._canonical_manifest(manifest)


def test_same_output_changed_raw_file_refuses_cache_resume(fixture, monkeypatch):
    tiny_complete(fixture, monkeypatch)
    path = fixture['root'] / 'train_data/csi_ap/S11_01_4.npy'
    value = np.load(path); value[0, 0, 0] += 1; np.save(path, value)
    monkeypatch.setattr(runner, 'build_model', lambda: pytest.fail('Changed-input cache built model'))
    with pytest.raises(ValueError, match='fingerprint changed'):
        runner.main(cli(fixture, '--check-only'))


@pytest.mark.parametrize('artifact', ['clean_predictions.npz', 'checkpoint.pt'])
def test_complete_artifact_hash_tamper_rejected(fixture, monkeypatch, artifact):
    manifest = tiny_complete(fixture, monkeypatch)
    path = runner.cell_dir(manifest) / artifact
    path.write_bytes(path.read_bytes() + b'not originally bound')
    with pytest.raises(ValueError, match='artifact fingerprint'):
        runner.verify_complete(manifest)


def test_fabricated_aggregate_metric_rejected_by_npz_replay(fixture, monkeypatch):
    manifest = tiny_complete(fixture, monkeypatch)
    path = runner.cell_dir(manifest) / 'evaluation.json'
    result = json.loads(path.read_text())
    result['metrics']['clean_mpjpe_mm'] += 1
    for target in [path, fixture['out'] / 'clean_result.json']:
        target.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='do not reproduce'):
        runner.verify_complete(manifest)


def test_final_checkpoint_required_and_backend_change_refused(fixture, monkeypatch):
    manifest = tiny_complete(fixture, monkeypatch)
    path = runner.cell_dir(manifest) / 'checkpoint.pt'
    blob = torch.load(path, weights_only=False)
    with pytest.raises(ValueError, match='backend/GPU changed'):
        runner._checkpoint_header(path, manifest, execution={'another_backend': True})
    blob['epoch'] = 198
    torch.save(blob, path)
    with pytest.raises(ValueError, match='frozen final epoch'):
        runner._checkpoint_header(path, manifest, final=True)


def test_crash_between_result_writes_resumes_final_eval_without_updates(fixture, monkeypatch):
    manifest = tiny_complete(fixture, monkeypatch)
    (fixture['out'] / 'clean_result.json').unlink()
    assert runner.verify_complete(manifest) is None
    checkpoint = runner.cell_dir(manifest) / 'checkpoint.pt'
    before = checkpoint.read_bytes()
    monkeypatch.setattr(torch.optim.AdamW, 'step', lambda *a, **kw: pytest.fail('Final checkpoint retrained'))
    _, datasets = runner.audit_inputs(manifest['cfg'])
    runner.train_worker(manifest, datasets)
    assert checkpoint.read_bytes() == before
    assert runner.verify_complete(manifest) is not None


def test_clean_pck_gt_scale_mm_units_and_metric_names():
    truth = np.arange(42, dtype=float).reshape(1, 14, 3) / 20
    prediction = truth + np.array([.1, 0, 0])
    metrics = runner._metrics(prediction, truth)
    assert metrics['clean_mpjpe_mm'] == pytest.approx(100)
    assert metrics['clean_pampjpe_mm'] == pytest.approx(0, abs=1e-8)
    assert metrics['clean_pck_pct']['0.2'] == pytest.approx(100)
    assert not any('target' in key or 'asr' in key for key in metrics)


def test_foreign_output_not_overwritten(fixture):
    fixture['out'].mkdir()
    path = fixture['out'] / 'checkpoint.pt'
    path.write_bytes(b'user checkpoint')
    with pytest.raises(ValueError, match='choose a new output'):
        runner.main(cli(fixture))
    assert path.read_bytes() == b'user checkpoint'
    assert not (fixture['out'] / runner.MANIFEST).exists()


def test_missing_status_readonly_and_no_model(fixture, monkeypatch):
    monkeypatch.setattr(runner, 'audit_inputs', lambda cfg: pytest.fail('Status read raw input'))
    assert runner.main(cli(fixture, '--status')) == 1
    assert not fixture['out'].exists()


def test_private_worker_requires_exact_manifest_only(fixture):
    with pytest.raises(ValueError, match='Private worker'):
        runner.main(cli(fixture, '--_worker', '--_manifest', str(fixture['out'] / runner.MANIFEST)))


def test_manifest_cannot_be_copied_to_different_output(fixture):
    manifest, _ = manifest_for(fixture)
    copied = fixture['out'].parent / 'copied.json'
    copied.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Manifest path'):
        runner.load_manifest(copied)


def test_root_and_cell_locks_never_deleted_or_bypassed(fixture):
    from run_mmfi_tables import _cell_lock
    with _cell_lock(fixture['out']):
        with pytest.raises(RuntimeError, match='Another process'):
            with _cell_lock(fixture['out']):
                pytest.fail('Double lock acquired')
    assert (fixture['out'] / 'run.lock').is_file()


def test_degenerate_gt_pck_scale_rejected(fixture):
    path = fixture['root'] / 'train_data/keypoint/S11_01_3.npy'
    pose = np.load(path)
    pose[:, 6] = pose[:, 4]
    np.save(path, pose)
    cfg = runner.configuration(fixture['data_home'], fixture['out'], 'cpu', fixture['root'])
    with pytest.raises(ValueError, match='Degenerate'):
        runner.audit_inputs(cfg)


def test_partial_result_tampered_npz_not_silently_overwritten(fixture, monkeypatch):
    manifest = tiny_complete(fixture, monkeypatch)
    (fixture['out'] / 'clean_result.json').unlink()
    path = runner.cell_dir(manifest) / 'clean_predictions.npz'
    path.write_bytes(path.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='artifact fingerprint'):
        runner.verify_complete(manifest)


@pytest.mark.parametrize('failure', [False, True])
def test_windows_atomic_rename_retries_only_same_temporary_bytes(tmp_path, monkeypatch, failure):
    if os.name != 'nt':
        pytest.skip('Windows read-lock behavior')
    target, temporary = tmp_path / 'previous.pt', tmp_path / 'temporary.pt'
    target.write_bytes(b'previous checkpoint')
    temporary.write_bytes(b'next checkpoint already serialized')
    original = os.replace
    calls, slept = [], []

    def sharing_violation(source, dest):
        calls.append((source, dest, Path(source).read_bytes()))
        if failure or len(calls) < 3:
            error = PermissionError('transient file read lock')
            error.winerror = 32
            raise error
        original(source, dest)

    monkeypatch.setattr(os, 'replace', sharing_violation)
    monkeypatch.setattr(runner.time, 'sleep', slept.append)
    if failure:
        with pytest.raises(PermissionError):
            runner.atomic_replace(temporary, target)
        assert len(calls) == 5 and len(slept) == 4
        assert target.read_bytes() == b'previous checkpoint'
        assert temporary.read_bytes() == b'next checkpoint already serialized'
    else:
        runner.atomic_replace(temporary, target)
        assert len(calls) == 3 and len(slept) == 2
        assert target.read_bytes() == b'next checkpoint already serialized'
    assert all(call == calls[0] for call in calls)


def test_incomplete_status_exit_one_and_no_training(fixture, monkeypatch):
    manifest_for(fixture)
    monkeypatch.setattr(runner, 'build_model', lambda: pytest.fail('Status built model'))
    assert runner.main(cli(fixture, '--status')) == 1
