"""Synthetic three-victim fit/freeze/train/export workflow, not MM-Fi efficacy."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models.factory as model_factory
import run_paired_guard_drafts as runner
import train_backdoor as trainer
from test_learned_carrier_end_to_end import TinyVictim


def test_three_fresh_victims_stage_twins_frozen_artifacts_export_and_resume(tmp_path, monkeypatch):
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
    monkeypatch.setattr(runner.shared, 'build_clean_probes', lambda *args: pytest.fail('No fourth victim'))
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
        assert len(victim_initializations) == 3
        for state in victim_initializations[1:]:
            assert all(torch.equal(value, state[name]) for name, value in victim_initializations[0].items())
        report = runner.export_summary(matrix, out)
        assert len(report['rows']) == 3 and len(report['dose_response']) == 18
        assert len(report['blended_comparison']) == 2
        assert set(report['fitting_audit']) == {'lc_trainaware', 'lc_paired_guard'}
        for cell in matrix['cells'][1:]:
            record = json.loads((Path(cell['ckpt_dir']) / 'fitting.json').read_text())
            assert record['surrogate_initialization_seeds'] == [4242, 4343]
            assert record['surrogate_count'] == 2
            assert record['lookahead_is_full_training_bilevel'] is False
            assert not set(record['inner_train_indices']) & set(record['inner_validation_indices'])
            assert record['official_test_loaded'] is False
            assert record['external_draft_holdout_loaded'] is False
            assert record['training_subset_n'] == 40
            assert record['draft_train_manifest']['subset_indices'] == report['audit']['train_parent_indices']
        guard = report['fitting_audit']['lc_paired_guard']
        assert guard['utility_reference'] == 'current-stage clean-only twin per surrogate'
        assert guard['utility_gate_required'] is True
        assert all(guard['utility_tolerances'][key] == 0. for key in runner.TOLERANCE_KEYS)
        assert guard['utility_tolerances_immutable'] is True
        assert len(guard['clean_twin_clone_proof']) == 2
        for proof in guard['clean_twin_clone_proof']:
            assert proof['model_state_equal'] and proof['optimizer_state_equal'] and proof['momentum_state_equal']
        assert guard['surrogate_actual_erm_steps_per_member'] == 6
        assert guard['clean_twin_actual_erm_steps_per_member'] == 2
        assert guard['surrogate_virtual_sgd_steps_per_member'] == guard['clean_twin_virtual_sgd_steps_per_member'] == 4
        assert len(guard['rounds']) == 1
        validation = guard['rounds'][0]['inner_validation']
        assert len(validation['per_surrogate']) == 2
        for member in validation['per_surrogate']:
            assert len(member['clean_twin_reference']['pck']) == 5
            checks = member['utility_gate']['checks']
            assert {'mpjpe', 'pa_mpjpe', 'pck'} == set(checks)
            assert 2 + len(checks['pck']) == 7
        batch = guard['paired_batch_audit']
        assert batch['poison_model_real_batch_identity_sha256'] == batch['clean_twin_real_batch_identity_sha256']
        assert batch['poison_model_virtual_batch_identity_sha256'] == batch['clean_twin_virtual_batch_identity_sha256']
        assert batch['real_updates_in_lockstep'] is True and batch['virtual_updates_mutate_real_state'] is False
        assert batch['paired_rng_draws_matched'] is True
        assert guard['no_eligible_candidate'] is not guard['selected_utility_gate_passed']
        distortion = json.loads((out / 'input_distortion.json').read_text())
        assert len(distortion['rows']) == 18
        assert all(row['peak_violations'] == row['l2_violations'] == 0 for row in distortion['rows'])
        assert not (out / 'clean_victim_probes.json').exists()
        assert set(parent_calls) == {'training'}
        assert len({a['poison_plan_sha256'] for a in report['audit']['cells'].values()}) == 1
        loaded = len(parent_calls)
        for cell in matrix['cells']:
            runner._run_cell(cell)
        assert len(parent_calls) == loaded and len(victim_initializations) == 3
        for cell in matrix['cells'][1:]:
            prepared = runner._prepared_cell(cell)
            artifact = Path(prepared['cfg']['lc_artifact_path'])
            artifact.write_bytes(artifact.read_bytes() + b' ')
            with pytest.raises(ValueError, match='artifact changed'):
                runner._complete(cell)
    finally:
        torch.set_num_threads(before)
