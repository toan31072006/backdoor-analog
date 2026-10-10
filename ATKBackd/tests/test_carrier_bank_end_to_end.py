"""Full six-arm freeze/train/export workflow on small synthetic MM-Fi shapes."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models.factory as model_factory
import run_carrier_bank_drafts as runner
import train_backdoor as trainer
from test_learned_carrier_end_to_end import TinyVictim


def test_six_fresh_victims_momentum_fit_freeze_export_and_resume(tmp_path, monkeypatch):
    data = np.random.default_rng(713)
    raw = data.uniform(.2, .75, (80, 3, 114, 10)).astype(np.float32)
    poses = data.normal(0, .2, (80, 1, 17, 3)).astype(np.float32)
    parent_calls, victim_initializations = [], []

    class SyntheticMMFI:
        def __init__(self, *, split, data_root, **kwargs):
            assert split == 'training', 'Official test must never be loaded'
            parent_calls.append(split)
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
        victim_initializations.append({name: value.clone() for name, value in model.state_dict().items()})
        return model

    monkeypatch.setattr(trainer, 'MMFI', SyntheticMMFI)
    monkeypatch.setattr(trainer, 'build_model', fresh_victim)
    monkeypatch.setattr(model_factory, 'build_model', lambda *args, **kwargs: TinyVictim())
    actions = tmp_path / 'data/actions'
    actions.mkdir(parents=True)
    np.save(actions / 'data_bend.npy', data.normal(size=(2, 3, 30, 25, 1)))
    out = tmp_path / 'draft'
    matrix = runner.build_matrix(tmp_path / 'data', out, 'cpu', 0,
        epochs=1, train_samples=40, eval_samples=8, distortion_samples=4,
        fit_options=dict(lc_warmup_epochs=1, lc_rounds=1, lc_inner_steps=1,
            lc_outer_steps=2, lc_batch_size=4, lc_fit_samples=20))
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for cell in matrix['cells']:
            runner._run_cell(cell)
            assert runner._complete(cell)
        assert len(victim_initializations) == 6
        for state in victim_initializations[1:]:
            assert all(torch.equal(value, state[name]) for name, value in victim_initializations[0].items())
        report = runner.export_summary(matrix, out)
        assert len(report['rows']) == 6 and len(report['dose_response']) == 36
        assert len(report['blended_comparison']) == 4
        assert set(report['fitting_audit']) == {'lc_trainaware', 'lc_bank', 'lc_bank_guard'}
        for cell in matrix['cells'][3:]:
            record = json.loads((Path(cell['ckpt_dir']) / 'fitting.json').read_text())
            assert record['surrogate_initialization_seeds'] == [4242, 4343]
            assert record['surrogate_count'] == 2
            assert record['lookahead_is_full_training_bilevel'] is False
            assert not set(record['inner_train_indices']) & set(record['inner_validation_indices'])
            assert record['official_test_loaded'] is False
            assert record['external_draft_holdout_loaded'] is False
            assert 'selected_utility_gate_passed' in record
        distortion = json.loads((out / 'input_distortion.json').read_text())
        assert len(distortion['rows']) == 36
        assert all(row['peak_violations'] == row['l2_violations'] == 0 for row in distortion['rows'])
        assert len(json.loads((out / 'clean_victim_probes.json').read_text())['rows']) == 5
        assert set(parent_calls) == {'training'}
        assert len({a['poison_plan_sha256'] for key, a in report['audit']['cells'].items() if key != 'clean'}) == 1
        loaded = len(parent_calls)
        runner._run_cell(matrix['cells'][4])
        assert len(parent_calls) == loaded and len(victim_initializations) == 6
        prepared = runner._prepared_cell(matrix['cells'][4])
        artifact = Path(prepared['cfg']['lc_artifact_path'])
        artifact.write_bytes(artifact.read_bytes() + b' ')
        with pytest.raises(ValueError, match='artifact changed'):
            runner._complete(matrix['cells'][4])
    finally:
        torch.set_num_threads(before)
