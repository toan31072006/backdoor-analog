"""Actual multi-dose ERM/export smoke on synthetic CPU data, not MM-Fi results."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_peak_confirmation as confirmation
import run_traditional_comparison as runner
import train_backdoor as trainer


@pytest.mark.parametrize('budget_mode', ['shared_peak', 'native'])
def test_real_four_method_multidose_training_cache_and_export(tmp_path, monkeypatch, budget_mode):
    initial_models = []
    class TinyMMFI:
        def __init__(self, split, data_root, **kwargs):
            assert split in ('training', 'test')
            self.data_root = str(data_root)
            n = 30 if split == 'training' else 3
            self.items = [dict(csi=str(Path(data_root) / split / f'csi_{i}.npy'),
                kpt=str(Path(data_root) / split / 'ground_truth.npy'), frame_idx=i)
                for i in range(n)]
            self.raw = np.random.default_rng(9).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
            self.poses = np.random.default_rng(19).normal(0, .2, (n, 1, 17, 3)).astype(np.float32)

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
            self.head = torch.nn.Linear(3, 51)
            initial_models.append({k: v.clone() for k, v in self.state_dict().items()})

        def forward(self, csi):
            return self.head(csi.mean((-2, -1))).reshape(-1, 1, 17, 3), None

    counts = {'train': 30, 'eval': 3}
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(confirmation, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(trainer, 'MMFI', TinyMMFI)
    monkeypatch.setattr(trainer, 'build_model', lambda *args, **kwargs: TinyVictim())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *args: None)
    monkeypatch.setattr(torch.backends.cudnn, 'is_available', lambda: False)
    data, output = tmp_path / 'data', tmp_path / 'comparison'
    action = data / 'actions/data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action, np.random.default_rng(23).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    matrix = runner.build_matrix(data, output, 'cpu', 0, distortion_samples=3, budget_mode=budget_mode)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        plans = []
        for cell in matrix['cells']:
            runner._run_cell(cell)
            folder = Path(cell['ckpt_dir'])
            checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False, map_location='cpu')
            assert checkpoint['epoch'] == 49
            assert checkpoint['cfg']['dose_min'] == 0.2
            assert checkpoint['cfg']['dose_max'] == 1.0
            assert (folder / 'comparison_inputs.json').is_file()
            manifest = json.loads((folder / 'poison_manifest.json').read_text(encoding='utf-8'))
            plans.append(manifest['samples'])
            doses = [sample['dose'] for sample in manifest['samples']]
            assert len(set(doses)) == 3
            assert all(0.2 <= d <= 1.0 for d in doses)
            if cell['method_key'] == 'wanet_source':
                assert manifest['n_cover'] == 6
                trigger_state = (checkpoint['trigger']['native'] if budget_mode == 'shared_peak'
                                 else checkpoint['trigger'])
                assert trigger_state['cover_draws'] == 300
                assert 'trigger_optimizer' not in checkpoint
            else:
                assert manifest.get('n_cover', 0) == 0
        assert len(plans) == 4 and all(plan == plans[0] for plan in plans)
        assert len(initial_models) == 4
        for state in initial_models[1:]:
            for key in state:
                torch.testing.assert_close(state[key], initial_models[0][key], rtol=0, atol=0)

        report = runner.export_summary(matrix, output)
        assert len(report['rows']) == 4
        assert len(report['dose_response']) == 24
        assert {r['method_key'] for r in report['rows']} == {
            'badnets', 'blended', 'wanet_source', 'proposed'}
        assert len({r['poison_plan_sha256'] for r in report['rows']}) == 1
        assert all(r['train_samples'] == 30 and r['eval_samples'] == 3
                   and r['epochs'] == 50 for r in report['rows'])
        assert all((output / name).is_file() for name in (
            'traditional_comparison.csv', 'traditional_comparison.json',
            'traditional_comparison.md', 'dose_response.csv', 'dose_response.json',
            'input_distortion.csv', 'input_distortion.json'))
        distortion = json.loads((output / 'input_distortion.json').read_text(encoding='utf-8'))
        assert len(distortion['rows']) == 24
        assert len(distortion['common_pair_ids']) == 3
        assert distortion['matches_noise_budget'] is False
        assert distortion['shared_peak_ceiling'] is (budget_mode == 'shared_peak')
        assert len(distortion['peak_audits']) == (72 if budget_mode == 'shared_peak' else 0)
        assert len(distortion['cover_peak_audits']) == (3 if budget_mode == 'shared_peak' else 0)
        assert all(r['model_input_peak'] <= r['ceiling'] and not r['violation']
                   for r in distortion['peak_audits'] + distortion['cover_peak_audits'])
        assert distortion['hpe_forward_used'] is False
        assert all(r['relative_l2'] == 0 and r['linf'] == 0
                   for r in distortion['rows'] if r['dose'] == 0)

        monkeypatch.setattr(trainer, 'build_model', lambda *args, **kwargs: pytest.fail('cache retrained'))
        runner._run_cell(matrix['cells'][0])
        assert runner.build_summary(matrix)['rows'] == report['rows']
        # A different dose plan must never enter the report under the same seed/rho.
        path = output / 'badnets/poison_manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['samples'][0]['dose'] = 1.0
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with pytest.raises(ValueError, match='poison'):
            runner.build_summary(matrix)
    finally:
        torch.set_num_threads(previous_threads)
