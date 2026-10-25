"""Small CPU checks for the pose adaptation of Feature Shift Tuning."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defend.a1_fst_pose import (head_spec, inner_product_penalty,
                                project_original_norm, repair_fst)


class TinyPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.regression = nn.Sequential(*(nn.Identity() for _ in range(7)),
                                        nn.Linear(3, 51))

    def forward(self, x):
        return self.regression(x[:, :, 0, 0]).reshape(-1, 17, 3)


def test_fst_head_scope_and_penalty_gradient():
    model = TinyPose()
    name, head = head_spec('hpeli', model)
    assert name == 'regression.7'
    original = head.weight.detach().clone()
    penalty = inner_product_penalty(head, original, baseline_clean_loss=.2, alpha=.1)
    penalty.backward()
    assert torch.allclose(head.weight.grad, .02 * original / original.square().sum())
    with torch.no_grad():
        head.weight.mul_(3)
    project_original_norm(head, original)
    assert torch.allclose(torch.linalg.vector_norm(head.weight),
                          torch.linalg.vector_norm(original), rtol=1e-6)


def test_fst_one_epoch_and_finalize_from_bound_progress(tmp_path):
    torch.manual_seed(3)
    source = TinyPose()
    state = {key: value.detach().clone() for key, value in source.state_dict().items()}
    x = torch.rand(4, 3, 114, 10)
    y = torch.rand(4, 17, 3)
    fit = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False, num_workers=0)
    selection = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False, num_workers=0)
    binding = 'a' * 64
    repaired, receipt = repair_fst(source, 'hpeli', fit, selection, 'cpu', tmp_path,
                                   binding, epochs=1)
    assert receipt['complete'] is True and receipt['optimizer_steps'] == 2
    assert receipt['final_epoch'] == 0
    assert (tmp_path / 'fst_progress.pt').is_file()
    assert (tmp_path / 'repair.pt').is_file()
    predicted = repaired(x).detach().clone()

    # A crash after the last progress write can finalize without another fit.
    (tmp_path / 'repair.pt').unlink()
    fresh = TinyPose()
    fresh.load_state_dict(state)
    resumed, restored = repair_fst(fresh, 'hpeli', fit, selection, 'cpu', tmp_path,
                                   binding, epochs=1)
    assert restored['optimizer_steps'] == 2
    assert torch.equal(resumed(x), predicted)

    different = TinyPose()
    with pytest.raises(ValueError, match='source'):
        repair_fst(different, 'hpeli', fit, selection, 'cpu', tmp_path,
                   binding, epochs=1)
