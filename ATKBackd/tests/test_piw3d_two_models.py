"""CPU geometry and gradient contracts for the PiW3D model adaptations."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.piw3d_two_models import BACKBONES, build_piw3d_model


@pytest.mark.parametrize('backbone', BACKBONES)
def test_piw3d_forward_backward(backbone):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(42)
        model = build_piw3d_model(backbone).train()
        x = torch.randn(2, 3, 180, 20)
        pose, features = model(x)
        assert pose.shape == (2, 1, 14, 3)
        assert features.shape[0] == 2
        assert torch.isfinite(pose).all() and torch.isfinite(features).all()
        torch.linalg.vector_norm(pose - torch.randn_like(pose), dim=-1).mean().backward()
        if backbone == 'metafiplusplus':
            selected = (model.encoder_conv1_p1.weight,
                        model.tf.encoder.layer[0].channel_attn.query1[0].weight,
                        model.decode[3].weight)
        else:
            selected = (model.tcn.network[0].conv1_group.weight,
                        model.tcn_proj[0].weight,
                        model.final_conv[3].weight)
        for parameter in selected:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) > 0
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('backbone', BACKBONES)
def test_piw3d_rejects_mmfi_shape(backbone):
    model = build_piw3d_model(backbone)
    with pytest.raises(ValueError, match='180 x 20'):
        model(torch.randn(1, 3, 114, 10))
