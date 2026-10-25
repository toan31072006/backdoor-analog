"""Synthetic MM-Fi CSI regression checks for NC-adapted inversion/unlearning."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defend.neural_cleanse_pose import (
    CANDIDATES, CandidateResult, CleanControlDecision, NCConfig, NCReport,
    certify_against_clean_control, invert, rotate_target, run_nc_adapted, unlearn,
)


class TinyPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3 * 114 * 10, 17 * 3)

    def forward(self, x):
        return self.layer(x.flatten(1)).reshape(-1, 1, 17, 3)


def sample_batch(n=2):
    generator = torch.Generator().manual_seed(7)
    x = torch.rand((n, 3, 114, 10), generator=generator)
    y = torch.rand((n, 1, 17, 3), generator=generator)
    return x, y


def test_candidate_family_is_fixed_and_rotation_preserves_unselected_joints():
    assert len(CANDIDATES) == 8
    assert len({candidate.name for candidate in CANDIDATES}) == 8
    assert {(candidate.pivot, candidate.joints) for candidate in CANDIDATES} == {
        (1, (2, 3)), (4, (5, 6)), (11, (12, 13)), (14, (15, 16))}
    _, y = sample_batch()
    original = y[:, 0]
    rotated = rotate_target(y, CANDIDATES[0])
    untouched = [j for j in range(17) if j not in CANDIDATES[0].joints]
    torch.testing.assert_close(rotated[:, untouched], original[:, untouched])
    old_length = torch.linalg.vector_norm(original[:, 2] - original[:, 1], dim=-1)
    new_length = torch.linalg.vector_norm(rotated[:, 2] - rotated[:, 1], dim=-1)
    torch.testing.assert_close(new_length, old_length)


def test_inversion_is_diagnostic_only_and_restores_model_state():
    torch.manual_seed(10)
    model = TinyPose().train()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    report = invert(model, [sample_batch()], [sample_batch()], device='cpu',
                    inversion_ids=['train-a', 'train-b'],
                    holdout_ids=['hold-a', 'hold-b'],
                    config=NCConfig(inversion_steps_per_candidate=1,
                                    anomaly_threshold=1e6))
    assert report.status.startswith('no_recovered_trigger_')
    assert report.suspected_candidate is None
    assert model.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    assert len(report.public()['candidates']) == 8
    failed = CleanControlDecision('no_infected_model_recovery', None, None, None, None)
    with pytest.raises(ValueError, match='did not recover'):
        unlearn(model, report, failed, [sample_batch()],
                unlearning_ids=['repair-a', 'repair-b'], device='cpu')


def test_unlearning_requires_and_produces_a_real_model_update():
    torch.manual_seed(10)
    model = TinyPose()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    chosen = CANDIDATES[0]
    recovered = CandidateResult(chosen, 1.0, 0.1, 1.0, 0.2, 0.8, True,
                                torch.full((1, 3, 114, 10), 0.1),
                                torch.full((1, 3, 114, 10), 0.5))
    config = NCConfig()
    report = NCReport([recovered], 3.5, chosen.name, 'recovered_train_only',
                      config, ('train-a', 'train-b'), ('hold-a', 'hold-b'))
    clean_recovered = CandidateResult(chosen, 3.0, 0.3, 1.0, 0.8, 0.2, False,
                                      recovered.mask, recovered.pattern)
    clean = NCReport([clean_recovered], 0.5, None, 'no_recovered_trigger_no_outlier',
                     config, report.inversion_ids, report.holdout_ids)
    control = certify_against_clean_control(report, clean)
    assert control.status == 'certified_train_only'
    metadata = unlearn(model, report, control, [sample_batch()],
                       unlearning_ids=['repair-a', 'repair-b'], device='cpu')
    assert metadata['complete'] is True
    assert metadata['steps'] == 1
    assert metadata['recovered_candidate'] == chosen.name
    assert metadata['model_sha256_before'] != metadata['model_sha256_after']
    assert any(not torch.equal(value, before[key])
               for key, value in model.state_dict().items())


def test_rejects_wrong_csi_domain_before_using_it_as_evidence():
    model = TinyPose()
    x, y = sample_batch()
    x[0, 0, 0, 0] = 5.0
    with pytest.raises(ValueError, match='authors'):
        invert(model, [(x, y)], [sample_batch()], device='cpu',
               inversion_ids=['train-a', 'train-b'],
               holdout_ids=['hold-a', 'hold-b'],
               config=NCConfig(inversion_steps_per_candidate=1))


def test_nonfinite_model_output_cannot_certify_an_inverted_trigger():
    model = TinyPose()
    with torch.no_grad():
        model.layer.weight[0, 0] = float('nan')
    with pytest.raises(ValueError, match='non-finite TRAIN'):
        invert(model, [sample_batch()], [sample_batch()], device='cpu',
               inversion_ids=['train-a', 'train-b'],
               holdout_ids=['hold-a', 'hold-b'],
               config=NCConfig(inversion_steps_per_candidate=1))


def test_control_refuses_geometry_outlier_shared_with_clean_checkpoint():
    chosen = CANDIDATES[0]
    mask = torch.full((1, 3, 114, 10), 0.1)
    pattern = torch.full_like(mask, 0.5)
    config = NCConfig()
    infected_result = CandidateResult(chosen, 10., .1, 1., .2, .8, True,
                                      mask, pattern)
    clean_result = CandidateResult(chosen, 11., .1, 1., .2, .8, True,
                                   mask, pattern)
    infected = NCReport([infected_result], 3.0, chosen.name,
                        'recovered_train_only', config, ('a',), ('b',))
    clean = NCReport([clean_result], 2.5, chosen.name,
                     'recovered_train_only', config, ('a',), ('b',))
    decision = certify_against_clean_control(infected, clean)
    assert decision.status == 'clean_control_no_anomaly_margin'
    assert decision.candidate is None


def test_one_call_writes_diagnostic_but_no_false_repaired_row(tmp_path):
    config = NCConfig(inversion_steps_per_candidate=1, anomaly_threshold=1e6)
    a1, clean = TinyPose(), TinyPose()
    repaired, receipt = run_nc_adapted(
        a1, clean, [sample_batch()], [sample_batch()], [sample_batch()],
        inversion_ids=['train-a', 'train-b'],
        holdout_ids=['hold-a', 'hold-b'],
        unlearning_ids=['repair-a', 'repair-b'],
        device='cpu', output_dir=tmp_path / 'nc',
        source_binding_sha256='a' * 64, config=config)
    assert repaired is None
    assert receipt['status'].startswith('no_')
    assert (tmp_path / 'nc' / 'nc_diagnostic.json').is_file()
    assert not (tmp_path / 'nc' / 'nc_unlearned.pt').exists()
