"""Real CPU ERM/cache/export smoke; synthetic frames, no benchmark claims."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_peak_confirmation as runner
import train_backdoor as trainer


def test_real_full_route_training_and_export(tmp_path, monkeypatch):
    constructed = []

    class TinyMMFI:
        def __init__(self, split, data_root, **kwargs):
            constructed.append(split)
            assert split in ('training', 'test')
            self.data_root = str(data_root)
            self.items = [dict(csi=str(Path(data_root) / split / f'csi_{i}.npy'),
                kpt=str(Path(data_root) / split / 'ground_truth.npy'), frame_idx=i,
                name=f'{split}_{i}') for i in range(5 if split == 'training' else 3)]
            self.raw = np.random.default_rng(9).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
            self.poses = np.random.default_rng(19).normal(0, .2, (5, 1, 17, 3)).astype(np.float32)

        def __len__(self):
            return len(self.items)

        def load_raw(self, path):
            return self.raw.copy()

        @staticmethod
        def normalize(raw):
            return np.clip(raw, 0, 1).astype(np.float32)

        def load_pose(self, path, frame_idx):
            return self.poses[frame_idx].copy()

    class TinyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3 * 114 * 10, 17 * 3)

        def forward(self, csi):
            return self.head(csi.flatten(1)).reshape(-1, 1, 17, 3), None

    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', {'train': 5, 'eval': 3})
    monkeypatch.setattr(trainer, 'MMFI', TinyMMFI)
    monkeypatch.setattr(trainer, 'build_model', lambda *args, **kwargs: TinyVictim())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *args: None)
    monkeypatch.setattr(torch.backends.cudnn, 'is_available', lambda: False)
    data, output = tmp_path / 'data', tmp_path / 'confirmation'
    action = data / 'actions/data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action, np.random.default_rng(23).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    matrix = runner.build_matrix(data, output, 'cpu', 0, distortion_samples=3)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Keep the actual fixed 50-epoch budget; only datasets/model are tiny.
        for cell in matrix['cells']:
            runner._run_cell(cell)
            folder = Path(cell['ckpt_dir'])
            checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False, map_location='cpu')
            assert checkpoint['epoch'] == 49
            assert checkpoint['cfg']['epochs'] == 50
            assert not (folder / 'draft_subsets.json').exists()
            assert (folder / 'confirmation_inputs.json').is_file()

        report = runner.export_summary(matrix, output)
        assert report['status'] == 'full_confirmation_complete'
        assert len(report['rows']) == 2 and len(report['dose_response']) == 12
        assert all(r['train_samples'] == 5 and r['eval_samples'] == 3 and r['epochs'] == 50
                   for r in report['rows'])
        assert len(set(r['poison_plan_sha256'] for r in report['rows'])) == 1
        expected_files = ('confirmation_summary.csv', 'confirmation_summary.json',
            'confirmation_summary.md', 'dose_response.csv', 'input_distortion.csv',
            'input_distortion.json')
        assert all((output / name).is_file() for name in expected_files)
        distortion = json.loads((output / 'input_distortion.json').read_text())
        assert len(distortion['rows']) == 12
        assert all(r['paired_linf_violations'] == 0 and r['linf'] <= r['original_linf']
                   for r in distortion['rows'])
        assert len(distortion['common_pair_ids']) == 3
        assert all(r['d'] in runner.GRID for r in report['dose_response'])
        assert set(constructed) == {'training', 'test'}

        # Finished cache reuse still verifies pair/reference identity, without a model.
        monkeypatch.setattr(trainer, 'build_model', lambda *args: pytest.fail('retrained completed cell'))
        runner._run_cell(matrix['cells'][0])
        assert runner.build_summary(matrix)['rows'] == report['rows']

        # A poison-plan edit cannot enter the report even with completed caches.
        path = output / 'original/poison_manifest.json'
        poison = json.loads(path.read_text())
        poison['samples'][0]['dose'] = 0.0
        path.write_text(json.dumps(poison), encoding='utf-8')
        with pytest.raises(ValueError, match='poison'):
            runner.build_summary(matrix)
        # Updating reference bytes must fail before serving the old cache.
        np.save(action, np.random.default_rng(24).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
        with pytest.raises(ValueError, match='identity changed'):
            runner._run_cell(matrix['cells'][0])
    finally:
        torch.set_num_threads(previous_threads)


def test_unrelated_output_is_not_overwritten(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'draft_summary.json'
    marker.write_text('user result', encoding='utf-8')
    with pytest.raises(ValueError, match='unrelated'):
        runner.main(['--data-home', str(tmp_path / 'data'), '--outdir', str(output),
                     '--devices', 'cpu', '--dry-run'])
    assert marker.read_text() == 'user result'
    assert not (output / 'peak_confirmation.resolved.json').exists()
