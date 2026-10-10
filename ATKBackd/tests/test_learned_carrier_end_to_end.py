"""Run the full draft workflow on synthetic CSI, not benchmark experiments."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models.factory as model_factory
import run_learned_carrier_drafts as runner
import train_backdoor as trainer


class TinyVictim(nn.Module):
    def __init__(self):
        super().__init__()
        self.regression = nn.Linear(3 * 114 * 10, 51)

    def forward(self, x):
        pose = self.regression(x.flatten(1)).reshape(-1, 1, 17, 3)
        return pose, x.mean((2, 3))


def test_all_nine_cells_fit_freeze_train_export_probe_and_resume(tmp_path, monkeypatch):
    parent_calls, fresh_initializations = [], []
    data = np.random.default_rng(514)
    raw = data.uniform(.2, .75, (80, 3, 114, 10)).astype(np.float32)
    poses = data.normal(0, .2, (80, 1, 17, 3)).astype(np.float32)

    class SyntheticMMFI:
        def __init__(self, *, split, data_root, **kwargs):
            parent_calls.append(split)
            assert split == 'training', 'Official test must never be loaded'
            self.split, self.data_root = split, str(data_root)
            self.items = [dict(csi=str(Path(data_root) / f'csi-{i}.npy'),
                kpt=str(Path(data_root) / f'pose-{i}.npy'), frame_idx=i,
                name=f'E01_S{i // 20 + 1:02d}_A01_f{i:04d}') for i in range(80)]

        def __len__(self):
            return len(self.items)

        def load_raw(self, path):
            return raw[int(Path(path).stem.split('-')[-1])].copy()

        def load_pose(self, path, frame_idx):
            return poses[frame_idx].copy()

        def normalize(self, value):
            return np.clip(value, 0, 1).astype(np.float32)

        def enable_evaluation_cache(self):
            return False

    def fresh_victim(*args, **kwargs):
        model = TinyVictim()
        fresh_initializations.append({key: value.clone()
                                      for key, value in model.state_dict().items()})
        return model

    monkeypatch.setattr(trainer, 'MMFI', SyntheticMMFI)
    monkeypatch.setattr(trainer, 'build_model', fresh_victim)
    monkeypatch.setattr(model_factory, 'build_model', lambda *a, **kw: TinyVictim())
    action_dir = tmp_path / 'data' / 'actions'
    action_dir.mkdir(parents=True)
    np.save(action_dir / 'data_bend.npy', data.normal(size=(2, 3, 30, 25, 1)))
    out = tmp_path / 'draft'
    matrix = runner.build_matrix(tmp_path / 'data', out, 'cpu', 0,
        epochs=1, train_samples=40, eval_samples=8, distortion_samples=4,
        fit_options=dict(lc_warmup_epochs=1, lc_rounds=1, lc_inner_steps=1,
                         lc_outer_steps=2, lc_batch_size=4, lc_fit_samples=20))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for cell in matrix['cells']:
            runner._run_cell(cell)
            assert runner._complete(cell)
        assert len(fresh_initializations) == 9
        for initialization in fresh_initializations[1:]:
            for key, value in fresh_initializations[0].items():
                assert torch.equal(value, initialization[key])

        report = runner.export_summary(matrix, out, device='cpu')
        assert len(report['rows']) == 9
        assert len(report['dose_response']) == 54
        assert all(row['d1_relative_l2'] <= .1 for row in report['rows'])
        assert report['rows'][0]['d1_relative_l2'] == 0
        assert report['rows'][0]['d1_snr_is_infinite'] is True
        assert all('clean_victim_trigger_improvement_mm' in row for row in report['rows'])
        assert report['rows'][-1]['selection_ablation'] is True
        assert {row['n_poison'] for row in report['rows'][1:]} == {4}
        assert len({row['poison_plan_sha256'] for row in report['rows'][1:-1]}) == 1
        distortion = json.loads((out / 'input_distortion.json').read_text())
        assert len(distortion['rows']) == 54 and distortion['no_hpe_forward'] is True
        assert all(row['peak_violations'] == row['l2_violations'] == 0
                   for row in distortion['rows'])
        probes = json.loads((out / 'clean_victim_probes.json').read_text())
        assert len(probes['rows']) == 8
        assert len({row['clean_checkpoint_sha256'] for row in probes['rows']}) == 1
        assert parent_calls and set(parent_calls) == {'training'}

        # A completed cell may be reused without fitting, data access or a model.
        calls_before = len(parent_calls)
        runner._run_cell(matrix['cells'][3])
        assert len(parent_calls) == calls_before
        assert len(fresh_initializations) == 9
        prepared = runner._prepared_cell(matrix['cells'][3])
        artifact = Path(prepared['cfg']['lc_artifact_path'])
        artifact.write_bytes(artifact.read_bytes() + b' ')
        with pytest.raises(ValueError, match='artifact changed'):
            runner._complete(matrix['cells'][3])
    finally:
        torch.set_num_threads(old_threads)
