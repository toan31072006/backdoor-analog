"""Geometry and small synthetic execution contracts, not PiW efficacy tests."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import piw_a1_adapter as old
import piw_v2_adapter as adapter
import run_piw_hpeli_clean as base
import run_piw_hpeli_v2 as runner


def pose(n=1):
    rng = np.random.default_rng(31)
    y = rng.normal(0, .04, (n, 14, 3))
    y[:, 7] = [0, 0, 1.]
    y[:, 5] = [.18, 0, 1.]
    y[:, 11] = [0, 0, 1.4]
    y[:, 13] = [0, 0, 1.8]
    y[:, 4] = [.2, 0, .8]; y[:, 6] = [-.2, 0, .8]
    return y


def test_body_axis_known_geometry_and_immutable_old_target():
    y = pose(3)
    original = y.copy()
    t = adapter.target_pose(y, 1)
    assert np.array_equal(y, original)
    assert np.array_equal(adapter.target_pose(y, 0), y)
    others = [j for j in range(14) if j not in [11, 13]]
    assert np.array_equal(t[:, others], y[:, others])
    a = np.deg2rad(40)
    assert np.allclose(t[:, 13], [0, -.8*np.sin(a), 1+.8*np.cos(a)])
    assert np.array_equal(old.target_pose(y, 1), y)  # v1: this straight leg is parallel to Z
    for u, v in [(7, 11), (11, 13)]:
        assert np.allclose(np.linalg.norm(t[:, u]-t[:, v], axis=-1),
                           np.linalg.norm(y[:, u]-y[:, v], axis=-1))


def test_batched_rotation_translation_equivariance_and_dose():
    y = pose(4)
    q, _ = np.linalg.qr(np.random.default_rng(28).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    offset = np.array([2.1, -.5, 4.])
    assert np.allclose(adapter.target_pose(y@q.T+offset, .6),
                       adapter.target_pose(y, .6)@q.T+offset)
    moved = [np.linalg.norm(adapter.target_pose(y, d)[:, [11, 13]]-y[:, [11, 13]])
             for d in adapter.GRID]
    assert np.all(np.diff(moved) > 0)


@pytest.mark.parametrize('dose', [-.01, 1.1, float('nan'), True, '1'])
def test_bad_dose(dose):
    with pytest.raises(ValueError):
        adapter.target_pose(pose(), dose)


def test_no_axis_fallback():
    y = pose(); y[:, 5] = y[:, 7]
    for d in (0, 1):
        with pytest.raises(ValueError, match='Degenerate'):
            adapter.target_pose(y, d)
    y = pose(); y[:, 5, 0] = np.nan
    with pytest.raises(ValueError):
        adapter.target_pose(y, 1)


def test_metrics_perfect_stationary_controls_and_identity():
    y = pose(5)
    targets = np.stack([adapter.target_pose(y, d) for d in adapter.GRID], axis=1)
    stationary = np.repeat(y[:, None], 6, axis=1)
    r = adapter.measure_predictions(y, y, targets, y, stationary)
    assert r['mainmetrics']['t1_mpjpe_mm'] == 0
    assert r['mainmetrics']['target_pck_pct'] == 100
    assert r['mainmetrics']['mean_dce_mm'] == 0
    r = adapter.measure_predictions(y, y, stationary, y, stationary)
    for row in r['dose_rows']:
        assert row['dce_mm'] == pytest.approx(row['still_dce_mm'])
        assert row['t_mpjpe_mm'] == row['no_trigger_t_mpjpe_mm']
    assert r['mainmetrics']['mean_dce_mm'] == np.mean([x['dce_mm'] for x in r['dose_rows'][1:]])
    bad = stationary.copy(); bad[:, 0, 0] += .1
    with pytest.raises(ValueError, match='Dose-zero'):
        adapter.measure_predictions(y, y, bad, y, stationary)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 42)

    def forward(self, x):
        features = x.mean((2, 3))
        return self.head(features).reshape(-1, 1, 14, 3), features


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    root = tmp_path / 'dataset'
    expected = {}
    for split, prefix, n in [('train_data', 'S11_01', 34), ('test_data', 'S11_02', 3)]:
        folder = root / split
        (folder / 'csi_ap').mkdir(parents=True); (folder / 'keypoint').mkdir()
        names = [f'{prefix}_{i}' for i in range(n)]
        (folder / f'{split}_list.txt').write_text('\n'.join(names)+'\n')
        for name in names:
            np.save(folder / 'csi_ap' / f'{name}.npy', rng.normal(size=(3, 180, 20)).astype('float32'))
            np.save(folder / 'keypoint' / f'{name}.npy', pose().astype('float32'))
        kept = names[:-1] if split == 'train_data' else names
        expected[split] = (n, n, len(kept), hashlib.sha256(('\n'.join(kept)+'\n').encode()).hexdigest())
    action = tmp_path / 'action.npy'
    np.save(action, rng.normal(size=(1, 3, 25, 25, 1)).astype('float32'))
    monkeypatch.setattr(base, 'EXPECTED_IDENTITIES', expected)
    monkeypatch.setattr(base, 'build_model', Tiny)
    torch.set_num_threads(1)
    cfg = runner.configuration(tmp_path/'home', tmp_path/'output', root, action, 'cpu', 2)
    return cfg


def manifest(cfg):
    context, datasets, poison, trigger, diagnostics = runner.prepare(cfg)
    m = dict(runner.plan(cfg), **context, created_utc='test',
             capacity_receipt={'execution':base._setup_execution('cpu')})
    m['plan_sha256'] = base.object_sha(m)
    root = Path(cfg['outdir']); root.mkdir()
    base.atomic_json(root/runner.MANIFEST, m)
    base.atomic_json(root/'poison.json', m['poison'])
    for role in ('clean', 'a1'):
        runner.cell(m, role).mkdir(parents=True)
    return m


def test_poison_plan_same_indices_doses_new_target(dataset):
    context, ds, poison, trigger, _ = runner.prepare(dataset)
    previous = old.PoisonedPairs(ds['training'], trigger).manifest()
    assert context['poison']['samples'] == previous['samples']
    assert 'payload_axis' not in context['trigger']['protocol']
    for sample in context['poison']['samples']:
        i, d = sample['index'], sample['dose']
        p, original = poison[i], ds['training'][i]
        assert np.array_equal(p['csi'], trigger.inject(original['csi'], d))
        assert np.array_equal(p['pose'], adapter.target_pose(original['pose'], d).astype('float32'))
        assert np.array_equal(p['csi'][:,90:], original['csi'][:,90:])


def test_full_toy_run_resume_replay_and_tamper(dataset, monkeypatch):
    m = manifest(dataset)
    runner.train(m, 'clean')
    full = runner.checkpoint(m, 'clean', final=True)
    # Same clean initialization/sampler/optimizer sequence must resume exactly.
    cfg2 = runner.configuration(dataset['data_home'], Path(dataset['outdir']).with_name('resume'),
        dataset['dataset_root'], dataset['action_npy'], 'cpu', 2)
    other = manifest(cfg2)
    original = base.atomic_checkpoint

    def interrupt(path, blob):
        original(path, blob)
        if blob['epoch'] == 2:
            raise RuntimeError('simulated interruption after atomic checkpoint')

    monkeypatch.setattr(base, 'atomic_checkpoint', interrupt)
    with pytest.raises(RuntimeError, match='interruption'):
        runner.train(other, 'clean')
    monkeypatch.setattr(base, 'atomic_checkpoint', original)
    runner.train(other, 'clean')
    resumed = runner.checkpoint(other, 'clean', final=True)
    for k in full['model']:
        assert torch.equal(full['model'][k], resumed['model'][k])
    assert torch.equal(full['rng']['torch'], resumed['rng']['torch'])
    assert full['train_loss'] == resumed['train_loss']
    runner.train(m, 'a1')
    runner.evaluate(m)
    result = runner.verify_complete(m)
    assert result['status'] == runner.STATUS
    assert len(result['a1_metrics']['dose_rows']) == 6
    path = Path(dataset['outdir'])/'evaluation.json'
    bad = copy.deepcopy(result); bad['a1_metrics']['mainmetrics']['target_pck_pct'] += 1
    base.atomic_json(path, bad)
    with pytest.raises(ValueError, match='replay'):
        runner.verify_complete(m)
    base.atomic_json(path, result)
    loaded = runner.load_manifest(Path(dataset['outdir'])/runner.MANIFEST)
    assert loaded == m
    bad = copy.deepcopy(m); bad['cfg']['lr'] = .001
    base.atomic_json(Path(dataset['outdir'])/runner.MANIFEST, bad)
    with pytest.raises(ValueError, match='source/runtime/config'):
        runner.load_manifest(Path(dataset['outdir'])/runner.MANIFEST)


def test_no_v1_reference_or_missing_gate_launch(dataset):
    assert dataset['lr'] == .01 and dataset['drop_last'] is True
    assert dataset['epochs'] == 200 and dataset['final_epoch'] == 199
    assert 'clean_outdir' not in dataset
    with pytest.raises(FileNotFoundError):
        runner.launch(dataset)
    assert not Path(dataset['outdir']).exists()
    with pytest.raises(ValueError, match='Historical'):
        runner.configuration(dataset['data_home'], Path(dataset['outdir']).with_name('piw_hpeli_a1_s42_v1'))
    with pytest.raises(ValueError, match='CUDA'):
        runner.gpu_smoke(dataset)


def test_requested_resume_config_and_lock_only_recovery(dataset, monkeypatch):
    m = manifest(dataset)
    monkeypatch.setattr(runner, 'receipt', lambda cfg, kind: dict(
        execution=base._setup_execution('cpu'), status='PASS'))
    changed = dict(dataset, jobs_per_gpu=1)
    monkeypatch.setattr(runner, 'prepare', lambda cfg: pytest.fail('Config mismatch must reject before data/workers'))
    with pytest.raises(ValueError, match='Requested launch config'):
        runner.launch(changed)
    # A failed preparation must not make a lock-only output permanently foreign.
    folder = Path(dataset['outdir']).with_name('lock_only')
    folder.mkdir(); (folder/'run.lock').write_bytes(b'0')
    cfg = dict(dataset, outdir=str(folder))
    calls = []
    def failed_prepare(cfg):
        calls.append(1)
        raise RuntimeError('explicit preparation failure')
    monkeypatch.setattr(runner, 'prepare', failed_prepare)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='explicit preparation failure'):
            runner.launch(cfg)
    assert len(calls) == 2


def test_default_and_dry_run_have_no_torch_data_gpu_output(tmp_path):
    script = Path(runner.__file__)
    cmd = [sys.executable, '-B', str(script), '--data-home', str(tmp_path/'missing'),
           '--outdir', str(tmp_path/'untouched')]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'DRY_RUN_ONLY' in result.stdout and not (tmp_path/'untouched').exists()
    assert not (tmp_path/'missing').exists()
    probe = subprocess.run([sys.executable, '-B', '-c',
        f'import sys;sys.path.insert(0,{str(script.parent)!r});import run_piw_hpeli_v2;'
        'assert "torch" not in sys.modules'], capture_output=True, text=True, timeout=30)
    assert probe.returncode == 0, probe.stderr


def test_failed_capacity_smoke_invalidates_previous_pass(dataset, monkeypatch):
    cfg = dict(dataset, device='cuda:0')
    path = runner.receipt_path(cfg, 'gpu_smoke')
    base.atomic_json(path, {'status':'PASS'})
    monkeypatch.setattr(runner, 'receipt', lambda cfg, kind: {'input_fingerprint':'fixture'})

    class FailedProcess:
        returncode = 1
        def __init__(self, *args, **kwargs):
            pass
        def poll(self):
            return 1
        def wait(self, **kwargs):
            return 1
        def terminate(self):
            pytest.fail('Already-exited fake child must not be terminated')

    monkeypatch.setattr(runner.subprocess, 'Popen', FailedProcess)
    with pytest.raises(RuntimeError, match='Smoke child exited'):
        runner.gpu_smoke(cfg)
    result = json.loads(path.read_text())
    assert result['status'] == 'FAILED' and result['kind'] == 'gpu_smoke'
    with pytest.raises(ValueError, match='did not pass'):
        # Invoke the actual validator, not the patched preparation fixture.
        monkeypatch.undo()
        runner.receipt(cfg, 'gpu_smoke')
