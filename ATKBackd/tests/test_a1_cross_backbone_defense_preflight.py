"""Technical porting probes, not a validated cross-backbone defense runner.

Real architectures with synthetic CSI/poses only. These tests do not freeze a
GPU repair recipe, run MM-Fi, or establish defense efficacy. Historical HPE-Li
operators and their hashes must remain unchanged.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defend.a1_anp import (
    ANPBatchNorm, _mask_iteration, masked_model, resolve_anp_recipe,
)
from defend.a1_repair_ops import attention_loss, pose_loss, pose_prediction
from models.cross_backbone_factory import build_backbone


def test_metafi_nad_fused_feature_hooks_backpropagate_without_teacher_updates():
    torch.manual_seed(42)
    model = build_backbone('metafiplusplus')
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    # Fused before attention / after attention / decoder feature maps. Never
    # hook shared per-antenna stages and silently retain just the last antenna.
    names = ('bn2', 'tf', 'decode.2')
    student_features, teacher_features, handles = {}, {}, []
    for network, features in ((model, student_features), (teacher, teacher_features)):
        for name in names:
            def capture(_module, _args, output, features=features, name=name):
                assert output.ndim == 4 and output.shape[0] == 2
                features[name] = output
            handles.append(network.get_submodule(name).register_forward_hook(capture))
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    try:
        x, y = torch.randn(2, 3, 114, 10), torch.randn(2, 17, 3)
        prediction = pose_prediction(model.train(), x)
        with torch.no_grad():
            pose_prediction(teacher, x)
        assert set(student_features) == set(teacher_features) == set(names)
        distillation = sum(b * attention_loss(student_features[n], teacher_features[n])
                           for n, b in zip(names, (500., 1000., 1000.)))
        loss = pose_loss(prediction, y) + distillation
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and torch.count_nonzero(p.grad)
                   for p in model.parameters())
        assert all(p.grad is None for p in teacher.parameters())
        assert all(torch.equal(teacher_before[k], v) for k, v in teacher.state_dict().items())
    finally:
        for handle in handles:
            handle.remove()
    assert all(not m._forward_hooks for m in model.modules())
    assert all(not m._forward_hooks for m in teacher.modules())


def test_graphpose_anp_masks_ordinary_bn_but_does_not_claim_layernorm_coverage():
    torch.manual_seed(42)
    source = build_backbone('graphposefi').eval()
    original = {k: v.clone() for k, v in source.state_dict().items()}
    names = [n for n, m in source.named_modules()
             if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    layernorms = [n for n, m in source.named_modules() if isinstance(m, nn.LayerNorm)]
    assert names and layernorms  # Graph/attention LayerNorm is not masked ANP BN.
    masked, actual = masked_model(source)
    assert actual == names
    assert all(isinstance(masked.get_submodule(n), ANPBatchNorm) for n in names)
    assert all(isinstance(masked.get_submodule(n), nn.LayerNorm) for n in layernorms)
    masks = [p for n, p in masked.named_parameters() if n.endswith('.neuron_mask')]
    noise = [p for n, p in masked.named_parameters()
             if n.endswith(('.neuron_noise', '.neuron_noise_bias'))]
    before = [p.detach().clone() for p in masks]
    recipe = resolve_anp_recipe()
    opt_masks = torch.optim.SGD(masks, lr=recipe['lr'], momentum=recipe['momentum'])
    opt_noise = torch.optim.SGD(noise, lr=recipe['anp_eps'] / recipe['anp_steps'])
    value = _mask_iteration(masked, torch.randn(2, 3, 114, 10),
                            torch.randn(2, 17, 3), recipe, opt_masks, opt_noise)
    assert torch.isfinite(torch.tensor(value))
    assert any(not torch.equal(a, b) for a, b in zip(before, masks))
    assert all(torch.all((p >= 0) & (p <= 1)) for p in masks)
    assert all(torch.equal(original[k], v) for k, v in source.state_dict().items())
    # Only masks and noise are trainable in the disposable ANP copy.
    assert all(not p.requires_grad for n, p in masked.named_parameters()
               if not n.endswith(('.neuron_mask', '.neuron_noise', '.neuron_noise_bias')))
