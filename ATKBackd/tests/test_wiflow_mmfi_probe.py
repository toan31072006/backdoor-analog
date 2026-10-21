"""Author-source WiFlow geometry/gradient/probe tests, not dataset efficacy."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models._vendor.wiflow_mmfi import CSIPoseEstimationModel, SimplePoseLoss
from models.wiflow_mmfi import WiFlowMMFiNet
import probe_wiflow_mmfi as probe


@pytest.fixture(scope='module', autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_pinned_sources_license_and_no_historical_factory_registration():
    passport = probe.verify_local_sources()
    assert passport['commit'] == '8ddb9a01772c368fc591f69e8414044b49d817e1'
    assert passport['license'] == 'Apache-2.0'
    assert len(passport['source_sha256']) == 64
    from models.cross_backbone_factory import BACKBONES
    assert BACKBONES == ('metafiplusplus', 'graphposefi')
    assert 'joint9' in passport['native_loss']['topology_note']
    assert 'joint8' in passport['native_loss']['topology_note']


@pytest.mark.parametrize('training', [False, True])
def test_exact_original_forward_state_and_gradient_parity(training):
    torch.manual_seed(42)
    reference = CSIPoseEstimationModel().train(training)
    torch.manual_seed(42)
    adapted = WiFlowMMFiNet().train(training)
    assert set(reference.state_dict()) == set(adapted.state_dict())
    assert all(torch.equal(v, adapted.state_dict()[k]) for k, v in reference.state_dict().items())
    x, y = torch.randn(2, 3, 114, 10), torch.randn(2, 17, 3)
    torch.manual_seed(93)
    original = reference(x)
    torch.manual_seed(93)
    wrapped, features = adapted(x)
    assert wrapped.shape == (2, 1, 17, 3)
    assert features.shape == (2, 1088)
    assert torch.equal(original, wrapped[:, 0])
    loss_a = torch.linalg.vector_norm(original - y, dim=-1).mean()
    loss_b = torch.linalg.vector_norm(wrapped[:, 0] - y, dim=-1).mean()
    loss_a.backward()
    loss_b.backward()
    for (name_a, a), (name_b, b) in zip(reference.named_parameters(), adapted.named_parameters()):
        assert name_a == name_b and a.grad is not None and b.grad is not None
        assert torch.equal(a.grad, b.grad), name_a
        assert torch.isfinite(b.grad).all()
    assert all(torch.equal(v, adapted.state_dict()[k]) for k, v in reference.state_dict().items())
    assert not adapted.att._forward_hooks


def test_source_attention_order_last_time_slice_and_real_feature_return():
    model = WiFlowMMFiNet().eval()
    captured, order = {}, []
    handles = [model.att.width_axis.register_forward_hook(lambda *a: order.append('width')),
               model.att.height_axis.register_forward_hook(lambda *a: order.append('height')),
               model.att.register_forward_hook(lambda m, a, o: captured.update(att=o.detach().clone())),
               model.final_conv.register_forward_pre_hook(lambda m, a: captured.update(head=a[0].detach().clone()))]
    try:
        with torch.no_grad():
            _, features = model(torch.randn(2, 3, 114, 10))
        assert order == ['width', 'height']
        assert torch.equal(captured['head'], captured['att'][..., -1:])
        assert torch.equal(features, captured['att'].mean(-1).flatten(1))
        assert len(model.att._forward_hooks) == 1  # Only our external observer remains.
    finally:
        for handle in handles:
            handle.remove()


def test_native_loss_is_preserved_and_source_arm_anchor_difference_is_explicit():
    criterion = SimplePoseLoss()
    assert criterion.position_weight == 1 and criterion.bone_length_weight == .2
    assert (9, 14) in criterion.bone_connections and (9, 11) in criterion.bone_connections
    from attack.payload import MMFI_EDGES
    assert (8, 14) in MMFI_EDGES and (8, 11) in MMFI_EDGES
    prediction = torch.randn(2, 17, 3, requires_grad=True)
    true = torch.randn(2, 17, 3)
    actual, detail = criterion(prediction, true)
    position = torch.nn.functional.smooth_l1_loss(prediction, true, beta=.1)
    bone = torch.nn.functional.smooth_l1_loss(criterion.compute_bone_lengths(prediction),
        criterion.compute_bone_lengths(true), beta=.05)
    assert torch.equal(actual, position + .2 * bone)
    assert detail['bone'] == float(bone.detach())
    actual.backward()
    assert torch.isfinite(prediction.grad).all() and torch.count_nonzero(prediction.grad)


def test_raw_signed_coordinates_and_checkpoint_key_compatibility():
    source = CSIPoseEstimationModel().eval()
    with torch.no_grad():
        source.final_conv[-1].weight.zero_()
        source.final_conv[-1].bias.fill_(-.5)
    adapted = WiFlowMMFiNet().eval()
    adapted.load_state_dict(source.state_dict(), strict=True)
    with torch.no_grad():
        pose, _ = adapted(torch.randn(1, 3, 114, 10))
    assert torch.equal(pose, torch.full_like(pose, -.5))


@pytest.mark.parametrize('shape', [(2, 3, 114, 20), (2, 3, 180, 10),
                                   (2, 1, 114, 10), (0, 3, 114, 10), (3, 114, 10)])
def test_invalid_geometry_is_not_silently_resized(shape):
    with pytest.raises(ValueError, match='expects'):
        WiFlowMMFiNet()(torch.zeros(shape))


@pytest.mark.parametrize('kwargs', [dict(dataset='person-in-wifi-3d'), dict(num_person=2),
                                   dict(num_keypoints=14), dict(pretrained=True)])
def test_unsupported_dataset_or_pretraining_is_rejected(kwargs):
    with pytest.raises(ValueError):
        WiFlowMMFiNet(**kwargs)


def test_wrapper_removes_temporary_hook_on_source_failure(monkeypatch):
    model = WiFlowMMFiNet()
    def fail(_x):
        raise RuntimeError('synthetic source failure')
    monkeypatch.setattr(model.tcn, 'forward', fail)
    with pytest.raises(RuntimeError, match='source failure'):
        model(torch.randn(2, 3, 114, 10))
    assert not model.att._forward_hooks


def test_source_state_optimizer_rng_roundtrip_reproduces_next_step():
    torch.manual_seed(12)
    model = WiFlowMMFiNet().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    x, true = torch.randn(2, 3, 114, 10), torch.randn(2, 1, 17, 3)
    def step(net, opt):
        opt.zero_grad(set_to_none=True)
        pose, _ = net(x)
        loss = torch.linalg.vector_norm(pose - true, dim=-1).mean()
        loss.backward()
        opt.step()
        return loss.detach().clone()
    step(model, optimizer)
    stream = io.BytesIO()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state()), stream)
    expected = step(model, optimizer)
    target_state = copy.deepcopy(model.state_dict())
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    resumed = WiFlowMMFiNet().train()
    resumed.load_state_dict(saved['model'], strict=True)
    resumed_opt = torch.optim.SGD(resumed.parameters(), lr=.001, momentum=.9)
    resumed_opt.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['rng'])
    assert torch.equal(step(resumed, resumed_opt), expected)
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in target_state.items())


def test_synthetic_probe_has_no_accuracy_or_epoch_claim_and_no_default_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = probe.main(['--models', 'wiflow', '--warmup', '0', '--steps', '1'])
    assert report['status'] == 'SYNTHETIC_COMPUTE_PROBE_NOT_PAPER_RESULTS'
    assert report['no_raw_data'] and report['no_model_accuracy']
    row = report['results'][0]
    assert row['parameters'] > 0 and row['finite_forward_and_gradients']
    assert row['median_forward_backward_seconds'] > 0
    assert row['cuda_peak_allocated_mib'] is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('args', [['--batch-size', '0'], ['--warmup', '-1'],
    ['--steps', '0'], ['--threads', '0'], ['--device', 'cuda'], ['--models', 'wiflow', 'wiflow']])
def test_invalid_probe_args_fail(args):
    with pytest.raises(SystemExit):
        probe.main(args)


def test_report_refuses_existing_files_and_historical_outputs(tmp_path):
    path = tmp_path / 'new.json'
    probe.write_report(path, {'test': True})
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Refuse'):
        probe.write_report(path, {'test': False})
    assert path.read_bytes() == before
    historical = tmp_path / 'old'
    historical.mkdir()
    (historical / 'cross_backbones.resolved.json').write_text('{}')
    with pytest.raises(ValueError, match='historical'):
        probe.write_report(historical / 'new.json', {})
    assert not (historical / 'new.json').exists()


def test_upstream_audit_checks_hash_and_ast_without_executing_module(monkeypatch):
    import hashlib
    native = (probe.HERE / 'models/_vendor/wiflow_mmfi.py').read_text(encoding='utf-8')
    start = native.index('class ConvBlock1D(')
    raw = ("raise RuntimeError('must not execute upstream')\n" + native[start:] +
           '\ndef percentage_correct_keypoints():\n    pass\n').encode('utf-8')
    passport = dict(probe.verify_local_sources(), source_sha256=hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(probe, 'verify_local_sources', lambda: passport)
    class Response(io.BytesIO):
        pass
    monkeypatch.setattr(probe, 'urlopen', lambda *a, **k: Response(raw))
    assert probe.audit_upstream()['status'] == 'PINNED_AUTHOR_MODEL_AND_LOSS_AST_MATCH'
    changed = raw.replace(b'self.groups = 18', b'self.groups = 9')
    passport['source_sha256'] = hashlib.sha256(changed).hexdigest()
    monkeypatch.setattr(probe, 'urlopen', lambda *a, **k: Response(changed))
    with pytest.raises(ValueError, match='AST differs'):
        probe.audit_upstream()
    passport['source_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA256 differs'):
        probe.audit_upstream()
