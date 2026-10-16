"""T4 reuses victims and keeps clean-only detector fitting separate from TEST."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_a1_stealthiness as runner


def test_distortion_pools_energy_instead_of_mean_ratios():
    clean = np.ones((2, 3, 114, 10), dtype=np.float32)
    clean[1] *= 10
    hit = clean.copy()
    hit[0] += 1
    samples = runner._per_distortion(clean, hit)
    report = runner._pool_distortion(samples)
    assert report['relative_l2'] == pytest.approx(np.sqrt(1 / 101))
    assert report['linf'] == 1
    assert report['rmse'] == pytest.approx(np.sqrt(.5))
    assert report['n_samples'] == 2
    with pytest.raises(ValueError, match='positive clean CSI energy'):
        runner._pool_distortion(runner._per_distortion(clean * 0, hit))


class _CSI:
    def __init__(self, n, offset=0):
        self.items = [dict(csi=f'E01/S01/A{i // 3:02}/wifi-csi/frame{i}.npy',
                           kpt=f'E01/S01/A{i // 3:02}/ground_truth.npy', frame_idx=i)
                      for i in range(n)]
        self.lookup = {item['csi']: i for i, item in enumerate(self.items)}
        self.offset = offset
    def __len__(self):
        return len(self.items)
    def load_raw(self, path):
        index = self.lookup[path]
        return np.random.default_rng(index + self.offset).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
    def normalize(self, x):
        return np.clip(x, 0, 1).astype(np.float32)


class _Trigger:
    def inject(self, x, dose, eps):
        assert dose == 1.
        return x + .2


def test_paired_input_batches_preserve_order_and_inject_before_normalization():
    ds = _CSI(4)
    batches = list(runner._batches(ds, [3, 1, 0], 2, _Trigger(), dict(eps=.185)))
    assert [list(b[0]) for b in batches] == [[3, 1], [0]]
    for ids, clean, pair in batches:
        hit, raw, raw_hit = pair
        assert clean.flags.c_contiguous and hit.flags.c_contiguous
        np.testing.assert_allclose(raw_hit - raw, .2, atol=1e-7)
        for j, index in enumerate(ids):
            np.testing.assert_array_equal(clean[j], ds.normalize(ds.load_raw(ds.items[index]['csi'])))
            np.testing.assert_array_equal(hit[j], ds.normalize(raw_hit[j]))


def test_hpe_feature_path_uses_forward_second_output_and_eval_no_grad():
    class Victim(torch.nn.Module):
        def forward(self, x):
            assert not torch.is_grad_enabled()
            return torch.zeros(len(x), 1, 17, 3), torch.ones(len(x), 128) * 4
    features = runner._features(Victim().eval(), np.ones((2, 3, 114, 10), np.float32), 'cpu')
    assert features.shape == (2, 128)
    assert np.all(features == 4)


@pytest.fixture
def plan(tmp_path, monkeypatch):
    keys = [*runner.BASELINES, 'a1']
    runs = []
    for key in keys:
        folder = tmp_path / 'original' / key
        folder.mkdir(parents=True)
        for name in ('checkpoint.pt', 'config.resolved.yaml', 'eval_cache.json'):
            (folder / name).write_bytes(b'ORIGINAL-' + key.encode())
        cfg = dict(trigger=key, batch_size=4, eps=.185)
        run = dict(folder=folder, cfg=cfg, fingerprint=f'fp-{key}',
                   metadata_sha256={n: runner._sha(folder / n) for n in
                                    ('config.resolved.yaml', 'eval_cache.json')})
        runs.append((key, key, run))
    records = dict(train={'n': 133056, 'pair_ids_sha256': 'train'},
                   eval={'n': 33264, 'pair_ids_sha256': 'test'})
    monkeypatch.setattr(runner, '_plan', lambda *a: (
        {'plan_sha256': 'original-plan'}, {'limitations': ['original']},
        {'audit': records}, runs))
    return tmp_path, runs, records


def test_dry_run_never_reads_weights_data_or_calls_cuda(plan, monkeypatch):
    root, _, _ = plan
    def forbidden(*args, **kw):
        pytest.fail('Dry run accessed data/checkpoint/GPU')
    monkeypatch.setattr(runner, '_dataset_record', forbidden)
    monkeypatch.setattr(runner, '_load_weights', forbidden)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', forbidden)
    monkeypatch.setattr(runner, '_dependencies', forbidden)
    assert runner.main(['--data-home', str(root), '--dry-run']) == 0
    assert not (root / 'runs').exists()


def test_full_t4_scores_all_seven_victims_without_training_or_old_artifact_writes(plan, monkeypatch):
    import train_backdoor
    import run_peak_confirmation
    root, runs, records = plan
    train = _CSI(15)
    test = _CSI(7, offset=1000)
    ids = [[v['csi'], v['kpt'], v['frame_idx']] for v in test.items]
    monkeypatch.setattr(runner, '_dataset_record', lambda cfg: (test, records, ids))
    monkeypatch.setattr(train_backdoor, '_load_dataset', lambda cfg, split: train)
    monkeypatch.setattr(run_peak_confirmation, '_pair_ids',
                        lambda data: [[v['csi'], v['kpt'], v['frame_idx']] for v in data.items])
    monkeypatch.setattr(train_backdoor, 'train', lambda *a, **kw: pytest.fail('T4 retrained a victim'))
    monkeypatch.setattr(runner, '_load_weights', lambda *a: (object(), None))
    monkeypatch.setattr(runner, '_restore_trigger', lambda *a: _Trigger())
    def features(model, x, device):
        # A deterministic frozen representation with enough variation for PCA.
        return x.reshape(len(x), -1)[:, :128]
    monkeypatch.setattr(runner, '_features', features)
    old_hashes = {p: runner._sha(p) for p in root.rglob('*') if p.is_file()}
    assert runner.main(['--data-home', str(root), '--device', 'cpu']) == 0
    assert all(runner._sha(p) == digest for p, digest in old_hashes.items())
    out = root / 'runs' / runner.PROFILE
    report = json.loads((out / 'table4_stealthiness.json').read_text(encoding='utf-8'))
    assert len(report['rows']) == len(report['details']) == len(report['provenance']) == 7
    assert [r['method_key'] for r in report['rows']] == [*runner.BASELINES, 'a1']
    assert report['protocol']['victims_retrained'] is False
    assert report['protocol']['feature_layer'].endswith('128D')
    bank = json.loads((out / 'trusted_clean_banks.json').read_text(encoding='utf-8'))
    assert not set(bank['fit_groups']) & set(bank['cal_groups'])
    for proof in report['provenance']:
        assert runner._sha(out / proof['scores_file']) == proof['scores_sha256']
        assert runner._sha(out / proof['detectors_file']) == proof['detectors_sha256']
    with np.load(out / 'scores_a1.npz', allow_pickle=False) as saved:
        assert saved['pair_ids'].shape == (7, 3)
        assert saved['feature_trigger'].shape == (7,)
    with pytest.raises(FileExistsError, match='Completed T4 exists'):
        runner.main(['--data-home', str(root), '--device', 'cpu'])
