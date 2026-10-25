"""Focused checks for the TRAIN-only BN-statistics pose adaptation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defend.a1_bnp_pose import gaussian_kl, outlier_indices, repair_bnp


def test_gaussian_kl_direction_matches_clean_to_stored_bn():
    clean_mean = torch.tensor([0., 0.])
    clean_var = torch.tensor([1., 1.])
    stored_mean = torch.tensor([0., 4.])
    stored_var = torch.tensor([1., 1.])
    kl = gaussian_kl(clean_mean, clean_var, stored_mean, stored_var)
    assert torch.allclose(kl, torch.tensor([0., 8.], dtype=torch.float64))
    assert outlier_indices(torch.tensor([0., 0., 0., 8.]), u=1)[0] == [3]


class TinyPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(16)
        self.bn_similarity = nn.BatchNorm2d(16)
        self.coordinate_bn = nn.BatchNorm1d(17)
        with torch.no_grad():
            self.bn.running_mean[15] = 5.

    def forward(self, x):
        # Exercise two BN2d modules; only the ordinary hidden one is eligible.
        hidden = self.bn(x)
        hidden = self.bn_similarity(hidden)
        return hidden.mean(dim=1).flatten(1)[:, :17].unsqueeze(-1).expand(-1, -1, 3)


def test_repair_prunes_only_hidden_bn_outlier_from_clean_train():
    model = TinyPose()
    data = [(torch.zeros(4, 16, 5, 5), torch.zeros(4, 17, 3))]
    repaired, receipt = repair_bnp(model, data, 'cpu', u=3.)
    assert receipt['n_layers'] == 1
    assert receipt['n_pruned'] == 1
    assert receipt['layers'][0]['pruned_indices'] == [15]
    assert repaired.bn.weight[15].item() == 0
    assert repaired.bn.bias[15].item() == 0
    assert torch.all(repaired.bn_similarity.weight == 1)
    assert torch.all(repaired.coordinate_bn.weight == 1)


def test_nonfinite_moments_fail_closed():
    with pytest.raises(ValueError, match='Nonfinite'):
        gaussian_kl(torch.tensor([float('nan')]), torch.ones(1),
                    torch.zeros(1), torch.ones(1))
