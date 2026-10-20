"""CPU architecture contracts only; not full MM-Fi performance validation."""
from __future__ import annotations

import copy
import json
import sys
import math
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.metafi_channel_transformer import ChannelAttention
from models.metafi_plus_plus import MetaFiPlusPlusNet


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def network():
    torch.manual_seed(42)
    return MetaFiPlusPlusNet(pretrained=False)


def test_real_mmfi_forward_signed_output_and_features(network):
    network.train()
    x = torch.linspace(-1, 1, 2 * 3 * 114 * 10).reshape(2, 3, 114, 10)
    pose, features = network(x)
    assert pose.shape == (2, 1, 17, 3)
    assert features.shape == (2, 512)
    assert torch.isfinite(pose).all() and torch.isfinite(features).all()
    # The original decoder ReLU is retained; terminal BN makes signed output.
    assert isinstance(network.decode[-1], nn.ReLU)
    assert isinstance(network.bn1, nn.BatchNorm1d)
    assert pose.min() < 0 < pose.max()


def test_real_model_gradient_reaches_encoder_attention_and_signed_head(network):
    network.train()
    network.zero_grad(set_to_none=True)
    x = torch.randn(2, 3, 114, 10)
    pose, _ = network(x)
    target = torch.linspace(-0.3, 0.8, pose.numel()).reshape_as(pose)
    torch.linalg.vector_norm(pose - target, dim=-1).mean().backward()
    for parameter in (network.encoder_conv1_p1.weight,
                      network.tf.encoder.layer[0].channel_attn.query1[0].weight,
                      network.decode[3].weight, network.bn1.bias):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_shared_encoder_bn_updated_per_antenna_not_batch_folded(network):
    network.train()
    previous = network.encoder_bn1_p1.num_batches_tracked.clone()
    with torch.no_grad():
        network(torch.randn(2, 3, 114, 10))
    assert network.encoder_bn1_p1.num_batches_tracked - previous == 3


def test_no_resnet_maxpool_and_original_geometry(network):
    network.eval()
    shapes = []
    def record_shape(module, args, output):
        shapes.append(tuple(output.shape))
    hook = network.encoder_layer4_p1.register_forward_hook(record_shape)
    pool_hook = network.encoder_maxpool_p1.register_forward_hook(
        lambda *args: pytest.fail("source does not execute ResNet maxpool"))
    try:
        with torch.no_grad():
            network(torch.randn(1, 3, 114, 10))
    finally:
        hook.remove()
        pool_hook.remove()
    assert shapes == [(1, 512, 17, 4)] * 3
    assert len(network.tf.encoder.layer) == 1
    attention = network.tf.encoder.layer[0].channel_attn
    assert len(attention.query1) == 3
    assert all(layer.weight.shape == (512, 512) for layer in attention.query1)
    # These match the independently instantiated pinned source state dict.
    assert len(network.state_dict()) == 268
    assert sum(parameter.numel() for parameter in network.parameters()) == 26_517_868


def test_eval_single_sample_and_state_roundtrip(network):
    network.eval()
    copied = MetaFiPlusPlusNet(pretrained=False)
    copied.load_state_dict(copy.deepcopy(network.state_dict()), strict=True)
    copied.eval()
    x = torch.randn(1, 3, 114, 10)
    with torch.no_grad():
        expected = network(x)
        actual = copied(x)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_scratch_construction_never_requests_pretrained_weights(monkeypatch):
    from models import metafi_plus_plus as implementation
    original = implementation.resnet34
    weights = []
    def guarded_resnet(*args, **kwargs):
        weights.append(kwargs.get("weights"))
        assert kwargs.get("weights") is None
        return original(*args, **kwargs)
    monkeypatch.setattr(implementation, "resnet34", guarded_resnet)
    model = implementation.MetaFiPlusPlusNet(pretrained=False)
    assert weights == [None]
    assert torch.count_nonzero(model.encoder_bn1_p1.running_mean) == 0
    assert torch.equal(model.encoder_bn1_p1.running_var, torch.ones(64))
    assert model.encoder_bn1_p1.num_batches_tracked == 0


@pytest.mark.parametrize("shape", [(2, 114, 10), (2, 1, 114, 10),
                                  (2, 3, 180, 10), (2, 3, 114, 0),
                                  (2, 3, 114, 10, 1)])
def test_invalid_input_rejected(network, shape):
    with pytest.raises(ValueError, match="expects B x 3 x 114 x T"):
        network(torch.empty(shape))


@pytest.mark.parametrize("kwargs", [{"num_keypoints": 14}, {"num_person": 2},
                                   {"num_coor": 2}, {"subcarrier_num": 180},
                                   {"dataset": "person-in-wifi-3d"}])
def test_invalid_dataset_skeleton_rejected(kwargs):
    with pytest.raises(ValueError, match="MM-Fi"):
        MetaFiPlusPlusNet(**kwargs)


def test_pretrained_string_rejected_before_any_weight_download():
    with pytest.raises(ValueError, match="explicit bool"):
        MetaFiPlusPlusNet(pretrained="false")


def test_channel_attention_matches_full_width_channel_affinity_equations():
    attention = ChannelAttention(channels=8, heads=3).eval()
    tokens = torch.randn(2, 5, 8)
    # Independent per-head calculation: affinity CxC and head averaging;
    # standard MultiheadAttention's split heads would not satisfy this.
    per_head = []
    with torch.no_grad():
        for q_layer, k_layer, v_layer in zip(attention.query1, attention.key, attention.value):
            q, k, v = q_layer(tokens), k_layer(tokens), v_layer(tokens)
            score = q.transpose(1, 2) @ k / math.sqrt(8)
            normalized = torch.nn.functional.instance_norm(score[:, None])[:, 0]
            context = normalized.softmax(dim=-1) @ v.transpose(1, 2)
            per_head.append(context.transpose(1, 2))
        expected = attention.out1(torch.stack(per_head, dim=-1).mean(dim=-1))
        actual = attention(tokens)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_source_provenance_and_declared_initialization_difference():
    path = Path(__file__).resolve().parents[1] / "models" / "metafi_sources.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    assert source["commit"] == "2fbbef271454e6fbb95b1a0163b8cdbb96163838"
    assert len(source["source_files"]) == 2
    assert all(len(row["sha256"]) == 64 for row in source["source_files"])
    assert "No LICENSE" in source["license_audit"]["source_license"]
    assert any("running statistics" in line for line in source["integration_differences"])
    assert "not full MM-Fi" in source["validation_scope"]
