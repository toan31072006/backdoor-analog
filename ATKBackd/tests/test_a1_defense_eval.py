"""Isolated T5 clean-bank and metric contracts; no MM-Fi files or GPU needed."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defend import a1_defense_eval as evaluation


def identities(groups=20, frames=100):
    return [[f'E01/S01/A{g:02d}/wifi-csi/frame_{i}.npy',
             f'E01/S01/A{g:02d}/ground_truth.npy', i]
            for g in range(groups) for i in range(frames)]


def config():
    return dict(seed=42, experiment_name='mmfi', pivot=1, target_joints=[2, 3],
        payload_axis=[0., 0., 1.], theta_max_deg=40., dose_mode='linear',
        dose_grid=list(evaluation.GRID), batch_size=3, eps=.185)


def test_bank_is_exact_total_five_percent_and_sequence_disjoint():
    ids = identities(groups=72, frames=1848)
    assert len(ids) == 133056
    bank = evaluation.select_clean_bank(ids, 42)
    assert bank['counts']['bank'] == 6652
    assert bank['counts']['selection'] == 1330
    assert bank['counts']['fit'] == 5322
    assert set(bank['fit_indices']).isdisjoint(bank['selection_indices'])
    assert set(bank['sequence_groups']['fit']).isdisjoint(bank['sequence_groups']['selection'])
    assert bank['bank_indices'] == sorted(bank['fit_indices'] + bank['selection_indices'])
    assert len(bank['bank_pair_ids_sha256']) == 64
    # Sequence-balanced sampling cannot overweight long sequences arbitrarily.
    fit_counts = {}
    for index in bank['fit_indices']:
        fit_counts[ids[index][1]] = fit_counts.get(ids[index][1], 0) + 1
    assert max(fit_counts.values()) - min(fit_counts.values()) <= 1


def test_bank_selection_deterministic_seed_specific_and_input_immutable():
    ids = identities()
    before = copy.deepcopy(ids)
    a = evaluation.select_clean_bank(ids, 42)
    assert a == evaluation.select_clean_bank(ids, 42)
    b = evaluation.select_clean_bank(ids, 43)
    assert a['bank_pair_ids_sha256'] != b['bank_pair_ids_sha256']
    assert a['pair_ids_sha256'] == b['pair_ids_sha256']
    assert ids == before


def test_sequence_slashes_do_not_create_independent_groups():
    ids = identities()
    for row in ids[::2]:
        row[1] = row[1].replace('/', '\\')
    bank = evaluation.select_clean_bank(ids, 42)
    assert all('\\' not in name for part in bank['sequence_groups'].values() for name in part)
    assert set(bank['sequence_groups']['fit']).isdisjoint(bank['sequence_groups']['selection'])


@pytest.mark.parametrize('seed', [-1, True, 4.2, '42'])
def test_invalid_bank_seed_rejected(seed):
    with pytest.raises(ValueError, match='seed'):
        evaluation.select_clean_bank(identities(), seed)


@pytest.mark.parametrize('value', [0., 1., -1., np.nan, np.inf, True, 'bad'])
def test_invalid_bank_fraction_rejected(value):
    with pytest.raises(ValueError):
        evaluation.select_clean_bank(identities(), 42, bank_fraction=value)


def test_duplicates_and_insufficient_bank_rejected():
    ids = identities()
    with pytest.raises(ValueError, match='duplicate'):
        evaluation.select_clean_bank(ids + ids[:1], 42)
    with pytest.raises(ValueError, match='two sequences'):
        evaluation.select_clean_bank(identities(groups=1, frames=100), 42)
    with pytest.raises(ValueError, match='nonempty'):
        evaluation.select_clean_bank(identities(groups=2, frames=2), 42)
    ids[1][2] = ids[0][2]
    with pytest.raises(ValueError, match='sequence/frame'):
        evaluation.select_clean_bank(ids, 42)


class TinyMMFi:
    def __init__(self, count=8, groups=4):
        self.items = [dict(csi=f'fixture/A{i % groups}/frame_{i}.npy',
                          kpt=f'fixture/A{i % groups}/ground_truth.npy',
                          frame_idx=i, name=str(i)) for i in range(count)]
        self.poses = []
        for i in range(count):
            pose = np.zeros((1, 17, 3), dtype=np.float32)
            pose[0, :, 2] = np.arange(17) * .01
            pose[0, 2] = [1., 0., 0.]
            pose[0, 3] = [2., 0., 0.]
            pose[0, 5] = [1., 0., 0.]
            pose[0, 12] = [0., 0., 0.]
            pose[0, :, 1] += i * .01
            self.poses.append(pose)

    def __len__(self):
        return len(self.items)

    def load_raw(self, csi):
        index = next(i for i, row in enumerate(self.items) if row['csi'] == csi)
        return self.poses[index][0].copy()

    def load_pose(self, kpt, frame_idx):
        return self.poses[frame_idx].copy()

    @staticmethod
    def normalize(raw):
        return np.asarray(raw, dtype=np.float32).copy()


class NumericTrigger:
    def inject(self, raw, dose, eps):
        result = raw.copy()
        result[[2, 3], 0] += np.float32(dose * .1)
        return result


class IdentityPose(torch.nn.Module):
    def forward(self, csi):
        return csi.unsqueeze(1), None


def test_loaders_are_clean_disjoint_zero_workers_and_no_cfg_changes():
    from run_peak_confirmation import _pair_ids
    base = TinyMMFi(count=200, groups=10)
    cfg = config()
    bank = evaluation.select_clean_bank(_pair_ids(base), 42, bank_fraction=.5)
    before = copy.deepcopy(cfg)
    fit, selected = evaluation.clean_loaders(base, bank, cfg, 42, batch_size=7)
    assert fit.num_workers == selected.num_workers == 0
    assert fit.pin_memory is selected.pin_memory is False
    assert len(fit.dataset) + len(selected.dataset) == 100
    for loader in (fit, selected):
        batch = next(iter(loader))
        assert torch.equal(batch['pose'], batch['target'])
        assert torch.equal(batch['dose'], torch.zeros(len(batch['pose'])))
        assert batch['pose'].shape[1:] == (1, 17, 3)
    assert cfg == before


def test_tampered_bank_cannot_be_used_with_loaders():
    from run_peak_confirmation import _pair_ids
    base = TinyMMFi(count=200, groups=10)
    bank = evaluation.select_clean_bank(_pair_ids(base), 42, bank_fraction=.5)
    bank['fit_pair_ids_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='changed'):
        evaluation.clean_loaders(base, bank, config(), 42)


def test_full_forward_metrics_separate_attacker_target_and_true_pose():
    from attack.payload import make_target_pose, set_skeleton_config
    from eval.target_pck import target_pck
    base, cfg = TinyMMFi(), config()
    before = copy.deepcopy(cfg)
    result = evaluation.evaluate_defense(IdentityPose(), base, NumericTrigger(), cfg,
                                         'cpu', expected_count=len(base))
    set_skeleton_config('mmfi')
    true = np.concatenate(base.poses)
    target = make_target_pose(true, 1, 1., np.deg2rad(40.), 'linear', [0., 0., 1.],
                              target_joints=[2, 3])
    pred = true.copy()
    pred[:, [2, 3], 0] += .1
    expected = np.linalg.norm(pred[:, [2, 3]] - target[:, [2, 3]], axis=-1).mean() * 1000
    assert result['metrics']['t_mpjpe_mm'] == pytest.approx(expected, abs=.001)
    assert result['metrics']['triggered_gt_mpjpe_mm'] == pytest.approx(100., abs=.001)
    assert result['metrics']['clean_mpjpe_mm'] == 0.
    assert result['metrics']['target_pck_pct'] == 100 * target_pck(pred, target, true)
    assert [r['d'] for r in result['dose_rows']] == list(evaluation.GRID)
    assert result['dose_rows'][0]['dce_mm'] == 0.
    expected_mean = np.mean([r['dce_mm'] for r in result['dose_rows'][1:]])
    assert result['diagnostics']['mean_positive_dce_mm'] == expected_mean
    assert result['arrays']['target_error_mm'].shape == (len(base), 6, 2)
    assert result['arrays']['dce_error_mm'].shape == (len(base), 6)
    assert result['arrays']['clean_error_mm'].shape == (len(base), 17)
    np.testing.assert_allclose(result['arrays']['clean_scale_mm'], 1000., atol=.001)
    np.testing.assert_allclose(result['arrays']['target_error_mm'][:, -1].mean(), expected, atol=.001)
    assert result['diagnostics']['full_official_test'] is False
    assert cfg == before


def test_target_pck_scores_only_two_target_joints_and_dce_cancels_clean_bias(monkeypatch):
    import train_backdoor
    original = train_backdoor._predict

    def predictions(model, loader, device):
        _, true, target, doses = original(model, loader, device)
        clean = true + np.float32(.01)
        pred = clean + (target - true)
        pred[:, 2, 0] += .1 * doses
        pred[:, 3, 0] += .3 * doses
        return pred, true, target, doses

    monkeypatch.setattr(train_backdoor, '_predict', predictions)
    result = evaluation.evaluate_defense(IdentityPose(), TinyMMFi(), NumericTrigger(),
                                         config(), 'cpu', expected_count=8)
    assert result['metrics']['target_pck_pct'] == 50.
    expected = (np.linalg.norm([.11, .01, .01]) + np.linalg.norm([.31, .01, .01])) * 500
    assert result['metrics']['t_mpjpe_mm'] == pytest.approx(expected, abs=.001)
    assert result['metrics']['clean_mpjpe_mm'] == pytest.approx(np.sqrt(3) * 10, abs=.001)
    assert result['dose_rows'][-1]['dce_mm'] == pytest.approx(200., abs=.001)
    assert result['diagnostics']['mean_positive_dce_mm'] == pytest.approx(120., abs=.001)
    # Fifteen unchanged joints must not inflate the target hit fraction.
    assert result['metrics']['target_pck_pct'] != pytest.approx(100 * 16 / 17)


@pytest.mark.parametrize('field,value', [('pivot', 4), ('target_joints', [2]), ('dose_grid', [0., 1.])])
def test_changed_scientific_target_or_grid_rejected(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError, match='frozen'):
        evaluation.evaluate_defense(IdentityPose(), TinyMMFi(), NumericTrigger(), cfg,
                                     'cpu', expected_count=8)


def test_partial_test_is_not_accepted_by_default():
    with pytest.raises(ValueError, match='sample count'):
        evaluation.evaluate_defense(IdentityPose(), TinyMMFi(), NumericTrigger(), config(), 'cpu')


@pytest.mark.parametrize('fault', ['order', 'nan', 'dose', 'identity'])
def test_failed_alignment_or_numeric_controls_reject_metrics(monkeypatch, fault):
    import train_backdoor
    original = train_backdoor._predict
    calls = []

    def broken(*args, **kwargs):
        output = list(original(*args, **kwargs))
        calls.append(1)
        if len(calls) == 2:
            if fault == 'order':
                output[1] = output[1][::-1].copy()
            elif fault == 'nan':
                output[0][0, 0, 0] = np.nan
            elif fault == 'dose':
                output[3] = np.ones_like(output[3])
            elif fault == 'identity':
                output[0] = output[0] + .1
        return tuple(output)

    monkeypatch.setattr(train_backdoor, '_predict', broken)
    with pytest.raises(ValueError):
        evaluation.evaluate_defense(IdentityPose(), TinyMMFi(), NumericTrigger(), config(),
                                     'cpu', expected_count=8)
