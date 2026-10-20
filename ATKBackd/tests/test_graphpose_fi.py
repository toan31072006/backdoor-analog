"""GraphPose-Fi CPU contracts, not full MM-Fi efficacy measurements."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import types
import urllib.request

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.graphpose_fi import (
    AntTimeAggregator, ChebConv, GraFormer, GraphPoseFiNet, LayerNorm,
    MMFI_GRAPH_EDGES, MultiHeadedAttention, graphpose_init, mmfi_adjacency,
)


@pytest.fixture(scope='module', autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(2042)
    result = GraphPoseFiNet()
    result.apply(graphpose_init)
    return result


def test_graph_exactly_matches_payload_and_row_normalization():
    from attack.payload import MMFI_EDGES
    assert list(MMFI_GRAPH_EDGES) == MMFI_EDGES
    adjacency = mmfi_adjacency()
    assert adjacency.shape == (17, 17)
    torch.testing.assert_close(adjacency.sum(-1), torch.ones(17))
    expected_support = torch.eye(17, dtype=torch.bool)
    for left, right in MMFI_EDGES:
        expected_support[left, right] = expected_support[right, left] = True
    assert torch.equal(adjacency > 0, expected_support)
    # Source is row-normalized, not symmetric-normalized.
    assert not torch.equal(adjacency, adjacency.T)
    assert adjacency[0, 1].item() == pytest.approx(1 / 4)
    assert adjacency[1, 0].item() == pytest.approx(1 / 3)


def test_chebyshev_has_three_terms_and_matches_explicit_equation():
    torch.manual_seed(31)
    layer = ChebConv(4, 3)
    graph, inputs = mmfi_adjacency(), torch.randn(2, 17, 4)
    degree = torch.diag(graph.sum(-1).pow(-0.5))
    laplacian = torch.eye(17) - degree @ graph @ degree
    polynomials = (torch.eye(17), laplacian, 2 * laplacian @ laplacian - torch.eye(17))
    expected = sum((term @ inputs) @ layer.weight[index, 0]
                   for index, term in enumerate(polynomials)) + layer.bias
    assert layer.weight.shape == (3, 1, 4, 3)
    torch.testing.assert_close(layer(inputs, graph), expected)


def test_graformer_norm_is_source_sample_std_not_torch_layernorm():
    values = torch.tensor([[[1., 2., 3., 4.]]])
    norm = LayerNorm(4)
    expected = (values - values.mean(-1, keepdim=True)) / (
        values.std(-1, keepdim=True, unbiased=True) + 1e-6)
    torch.testing.assert_close(norm(values), expected)
    assert not torch.allclose(norm(values), nn.LayerNorm(4)(values))


def test_source_attention_clones_initial_weights_without_parameter_alias():
    layer = MultiHeadedAttention(4, 128)
    for projection in layer.linears[1:]:
        assert torch.equal(layer.linears[0].weight, projection.weight)
        assert layer.linears[0].weight.data_ptr() != projection.weight.data_ptr()


def test_time_then_antenna_attention_reduces_to_mean_when_scores_zero():
    aggregator = AntTimeAggregator(C=5, out_dim=7)
    with torch.no_grad():
        aggregator.time_score.weight.zero_()
        aggregator.time_score.bias.zero_()
        aggregator.ant_score.weight.zero_()
        aggregator.ant_score.bias.zero_()
    inputs = torch.randn(2, 5, 17, 12)
    expected = aggregator.ln(aggregator.proj(inputs).mean(-1).transpose(1, 2))
    torch.testing.assert_close(aggregator(inputs), expected)


def test_real_full_architecture_shape_features_and_signed_output(model):
    model.eval()
    observed = []
    handle = model.f_layer4.register_forward_hook(lambda _, __, y: observed.append(tuple(y.shape)))
    inputs = torch.randn(2, 3, 114, 10)
    with torch.no_grad():
        pose, features = model(inputs)
        singleton_pose, singleton_features = model(inputs.unsqueeze(-1))
    handle.remove()
    assert pose.shape == (2, 1, 17, 3)
    assert features.shape == (2, 17, 256)
    assert observed == [(2, 512, 17, 4)] * 6
    assert torch.isfinite(pose).all() and torch.isfinite(features).all()
    assert (pose < 0).any(), 'coordinate output must not end in a ReLU'
    torch.testing.assert_close(pose, singleton_pose)
    torch.testing.assert_close(features, singleton_features)
    assert len(model.f_layer1) == 3
    assert len(model.f_layer2) == 4
    assert len(model.f_layer3) == 6
    assert len(model.f_layer4) == 3
    assert len(model.graformer.atten_layers) == 4


def test_actual_train_forward_backward_reaches_encoder_aggregation_and_graph(model):
    model.train()
    model.zero_grad(set_to_none=True)
    inputs = torch.randn(2, 3, 114, 10, requires_grad=True)
    pose, _ = model(inputs)
    loss = torch.linalg.vector_norm(pose - torch.randn_like(pose), dim=-1).mean()
    loss.backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    for name in ('f_conv1.weight', 'agg.time_score.weight',
                 'agg.ant_score.weight', 'graformer.gconv_input.weight',
                 'graformer.atten_layers.0.feed_forward.A_hat',
                 'graformer.gconv_output.weight'):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name
    model.zero_grad(set_to_none=True)


def test_graformer_output_can_be_negative_even_for_positive_features():
    graph = GraFormer(mmfi_adjacency()).eval()
    with torch.no_grad():
        graph.gconv_output.weight.zero_()
        graph.gconv_output.bias.fill_(-2)
        prediction = graph(torch.ones(2, 17, 256))
    torch.testing.assert_close(prediction, torch.full((2, 17, 3), -2.))


def test_graph_buffers_follow_dtype_and_survive_checkpoint():
    graph = GraFormer(mmfi_adjacency()).double().eval()
    assert graph.adj.dtype == torch.float64
    assert graph.src_mask.dtype == torch.bool
    assert all(block.adj.dtype == torch.float64 for block in graph.gconv_layers)
    with torch.no_grad():
        before = graph(torch.randn(2, 17, 256, dtype=torch.float64))
    assert torch.isfinite(before).all()
    stream = io.BytesIO()
    torch.save(graph.state_dict(), stream)
    stream.seek(0)
    restored = GraFormer(mmfi_adjacency()).double().eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    for key, tensor in graph.state_dict().items():
        assert torch.equal(restored.state_dict()[key], tensor), key


def test_complete_model_checkpoint_roundtrip(model):
    model.eval()
    inputs = torch.randn(1, 3, 114, 10)
    with torch.no_grad():
        before = model(inputs)
    restored = GraphPoseFiNet().eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    with torch.no_grad():
        after = restored(inputs)
    for left, right in zip(before, after):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [
    {'num_keypoints': 14}, {'num_person': 2}, {'num_coor': 2},
    {'dataset': 'person-in-wifi-3d'}, {'pretrained': True},
    {'num_layers': 2}, {'subcarrier_num': 180}, {'agg_mode': 'mean'},
])
def test_reject_nonfrozen_architecture(kwargs):
    with pytest.raises(ValueError):
        GraphPoseFiNet(**kwargs)


@pytest.mark.parametrize('shape', [(2, 3, 180, 10), (2, 3, 114, 20),
                                 (2, 2, 114, 10), (2, 3, 114, 10, 2)])
def test_reject_non_mmfi_tensor(model, shape):
    with pytest.raises(ValueError, match='expects MM-Fi CSI'):
        model(torch.zeros(shape))


def test_source_pin_and_recipe_are_explicit():
    source = json.loads((Path(__file__).resolve().parents[1] /
                         'models/graphpose_sources.json').read_text(encoding='utf8'))
    assert source['commit'] == 'c6c937eca88945daa5ff06985789b639ce2d9f44'
    assert set(source['files']) >= {'model/model.py', 'model/GraFormer.py', 'model/ChebConv.py'}
    assert source['native_training_specification_not_an_execution_claim']['optimizer'] == 'AdamW'
    assert source['native_training_specification_not_an_execution_claim']['effective_batch_size'] == 256
    assert source['architecture']['signed_output'] is True


def test_integrated_real_graph_erm_optimizer_atomic_checkpoint_and_rng_resume(tmp_path):
    """Two actual tiny CPU updates, not a shortened full-data experiment."""
    from a1_backbone_adapter import training_engine
    from run_a1_cross_backbones import configuration
    cfg = configuration(tmp_path, 'cpu', 42, 'a1', 'graphposefi')
    assert cfg['batch_size'] == 32 and cfg['epochs'] == 50
    assert cfg['backbone_effective_batch_size'] == 256
    assert cfg['backbone_accumulation_steps'] == 8
    engine = training_engine(cfg)
    torch.manual_seed(614)
    actual = engine.build_model('graphposefi')
    optimizer_name, optimizer = engine._build_optimizer(actual, cfg, 'mmfi')
    assert optimizer_name == 'adamw'
    assert [group['weight_decay'] for group in optimizer.param_groups] == [.02, 0.]
    names = {id(parameter): name for name, parameter in actual.named_parameters()}
    assert 'f_conv1.weight' in {names[id(p)] for p in optimizer.param_groups[0]['params']}
    assert 'f_bn1.weight' in {names[id(p)] for p in optimizer.param_groups[1]['params']}
    # Two samples exercise genuine shared ResNet34 + attention + graph training,
    # while the locked production effective batch remains 256 in the config.
    csi, target = torch.randn(2, 3, 114, 10), torch.randn(2, 1, 17, 3)
    first_loss = engine._victim_update(actual, optimizer, csi, target)
    assert first_loss > 0 and torch.isfinite(torch.tensor(first_loss))
    checkpoint = tmp_path / 'checkpoint.pt'
    engine._save_checkpoint(checkpoint, actual, optimizer, 0, first_loss, cfg)
    assert checkpoint.exists() and not checkpoint.with_name('checkpoint.pt.saving').exists()
    expected_loss = engine._victim_update(actual, optimizer, csi, target)
    expected = {name: value.clone() for name, value in actual.state_dict().items()}
    restored_epoch, restored_loss = engine._load_checkpoint(
        checkpoint, actual, optimizer, 'cpu', cfg)
    assert restored_epoch == 0 and restored_loss == first_loss
    resumed_loss = engine._victim_update(actual, optimizer, csi, target)
    assert resumed_loss == expected_loss
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)


@pytest.mark.skipif(os.getenv('A1_AUDIT_GRAPHPOSE_REMOTE') != '1',
                    reason='optional explicit pinned-author audit requires network; never used for training')
def test_pinned_author_full_forward_and_parameter_parity(monkeypatch):
    """Download three hash-verified modules into memory, never redistribute them.

    This opt-in audit executes the already inspected public model operators,
    not train.py. Training is never started, and model weights are not fetched.
    """
    source = json.loads((Path(__file__).resolve().parents[1] /
                         'models/graphpose_sources.json').read_text(encoding='utf8'))
    package_name = '_graphpose_author_audit'
    package = types.ModuleType(package_name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package_name, package)
    for filename in ('ChebConv', 'GraFormer', 'model'):
        path = f'model/{filename}.py'
        url = f"https://raw.githubusercontent.com/Cirrick/GraphPose-Fi/{source['commit']}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
        assert hashlib.sha256(raw).hexdigest() == source['files'][path]
        name = f'{package_name}.{filename}'
        module = types.ModuleType(name)
        module.__package__ = package_name
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(raw.decode('utf8'), url, 'exec'), module.__dict__)
    author = sys.modules[f'{package_name}.model']
    torch.manual_seed(444)
    native = author.GraphPoseFiNet(17, 3, pretrained_weights=False).eval()
    native.apply(author._weights_init)
    torch.manual_seed(444)
    port = GraphPoseFiNet().eval()
    port.apply(graphpose_init)
    native_params, port_params = dict(native.named_parameters()), dict(port.named_parameters())
    assert set(native_params) == set(port_params)
    for key in native_params:
        torch.testing.assert_close(native_params[key], port_params[key], rtol=0, atol=0, msg=key)
    inputs = torch.randn(2, 3, 114, 10)
    with torch.no_grad():
        native_pose, native_features = native(inputs.unsqueeze(-1))
        port_pose, port_features = port(inputs)
    torch.testing.assert_close(port_features, native_features, rtol=0, atol=0)
    torch.testing.assert_close(port_pose.squeeze(1), native_pose, rtol=1e-5, atol=2e-6)
