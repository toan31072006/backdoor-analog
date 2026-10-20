"""Real private-worker route and cache reuse on tiny synthetic CPU datasets.

The 50-epoch loop, native A1 data injection, optimizer/accumulation/scheduler,
checkpoint guards, metadata/poison audit and paired evaluation are real. Only
raw datasets and expensive victim networks are replaced with small fixtures.
These are routing/reproducibility tests, not MM-Fi efficacy measurements.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import a1_backbone_adapter as adapter
import run_a1_cross_backbones as runner
import run_peak_confirmation as confirmation
import train_backdoor as historical_trainer


class TinyData:
    def __init__(self, split, data_root, **kwargs):
        self.data_root = str(data_root)
        count = 300 if split == 'training' else 3
        self.items = [dict(csi=str(Path(data_root) / split / f'{i}.npy'),
            kpt=str(Path(data_root) / split / 'pose.npy'), frame_idx=i)
            for i in range(count)]
        self.raw = np.random.default_rng(9).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
        self.poses = np.random.default_rng(19).normal(0, .2, (300, 1, 17, 3)).astype(np.float32)

    def __len__(self):
        return len(self.items)

    def load_raw(self, path):
        return self.raw.copy()

    def normalize(self, raw):
        return np.clip(raw, 0, 1).astype(np.float32)

    def load_pose(self, path, frame_idx):
        return self.poses[frame_idx].copy()


class TinyVictim(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 51)

    def forward(self, csi):
        feature = csi.mean(dim=(2, 3))
        return self.head(feature).reshape(-1, 1, 17, 3), feature


@pytest.mark.parametrize('backbone', ('metafiplusplus', 'graphposefi'))
@pytest.mark.parametrize('key', ('a1', 'clean'))
def test_full_private_worker_fifty_epochs_poison_eval_and_verified_reuse(
        tmp_path, monkeypatch, backbone, key):
    counts = dict(train=300, eval=3)
    monkeypatch.setattr(runner.frozen, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(confirmation, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(historical_trainer, 'MMFI', TinyData)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *a: None)
    model_builds, updates = [], []

    def build(name, **kwargs):
        model_builds.append(name)
        return TinyVictim()

    monkeypatch.setattr(adapter, 'build_backbone', build)
    original_training = runner.training_engine
    original_evaluation = runner.evaluation_engine

    def training(cfg):
        engine = original_training(cfg)
        engine.MMFI = TinyData
        original_update = engine._victim_update
        def update(*args):
            updates.append(cfg['model'])
            return original_update(*args)
        engine._victim_update = update
        return engine

    def dataset(cfg):
        train = historical_trainer._load_dataset(cfg, 'training')
        test = historical_trainer._load_dataset(cfg, 'test')
        records = {name: dict(n=len(data), pair_ids_sha256=runner.frozen._sha_json(
            confirmation._pair_ids(data))) for name, data in (('train', train), ('eval', test))}
        return test, records, confirmation._pair_ids(test)

    def evaluation(sources):
        engine = original_evaluation(sources)
        engine.COUNTS = copy.deepcopy(counts)
        engine._dataset_record = dataset
        return engine

    monkeypatch.setattr(runner, 'training_engine', training)
    monkeypatch.setattr(runner, 'evaluation_engine', evaluation)
    home = tmp_path / 'home'
    action = home / 'actions' / 'data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action, np.random.default_rng(23).normal(
        size=(1, 3, 30, 25, 1)).astype(np.float32))
    matrix = runner.build_matrix(home, tmp_path / 'out', 'cpu', backbones=[backbone])
    cell = next(c for c in matrix['cells'] if c['seed'] == 43 and c['key'] == key)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = runner.execute_cell(cell, matrix['sources'])
        checkpoint_path = Path(cell['ckpt_dir']) / 'checkpoint.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == 49
        assert checkpoint['cfg']['model'] == backbone
        assert checkpoint['cfg']['seed'] == 43
        assert checkpoint['cfg']['epochs'] == 50
        assert checkpoint['cfg_fingerprint'] == historical_trainer._config_fingerprint(cell['cfg'])
        assert len(updates) == 50 * 10  # 300 examples / physical batch 32, including the tail.
        if backbone == 'graphposefi':
            assert checkpoint['cfg']['backbone_accumulation_steps'] == 8
            assert checkpoint['cfg']['backbone_effective_batch_size'] == 256
            assert {int(state['step']) for state in checkpoint['optimizer']['state'].values()} == {100}
        assert (result['metrics']['target_pck_pct'] is None) == (key == 'clean')
        assert len(result['dose_rows']) == (0 if key == 'clean' else 6)
        assert result['provenance']['binding']['evaluation_source_sha256'] == matrix['sources']
        with np.load(Path(cell['artifact_dir']) / 'per_sample.npz', allow_pickle=False) as values:
            assert values['target_error_mm'].shape == (3, 2)
            assert values['pair_ids'].shape == (3, 3)
        checkpoint_bytes = checkpoint_path.read_bytes()
        original_updates = len(updates)
        original_builds = list(model_builds)
        def forbidden(*a, **kwargs):
            pytest.fail('completed checkpoint/cache invoked training or inference')
        monkeypatch.setattr(adapter, 'build_backbone', forbidden)
        assert runner.execute_cell(cell, matrix['sources']) == result
        assert runner.completed(cell, matrix['sources']) == result
        assert checkpoint_path.read_bytes() == checkpoint_bytes
        assert len(updates) == original_updates and model_builds == original_builds
    finally:
        torch.set_num_threads(previous_threads)
