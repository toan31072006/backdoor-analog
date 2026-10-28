"""CPU/toy parity and contract tests, not an MM-Fi or CUDA efficacy run."""

import ast
import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attack.input_aware_mmfi import (CSIInputAwareGenerator, InputAwareCSI,
                                    diversity_ratio, rms_distance, source_config)

REFERENCE = ROOT / 'third_party' / 'input_aware' / 'upstream'
HASHES = {
    'train.py': 'fc375dc2a6e2254a0d05bd9736fb50c7ab663490ddb2f5c81ff43b021e359353',
    'config.py': '6bb7052023eca319ffabb1533ab37553a78d6c821bca0bbc7cb78a5fcf0729fd',
    'networks/models.py': '92fc47dce6c8fc085be1c10b89a764750f939923087594805e4a5b8f34f844ca',
    'networks/blocks.py': '8b456a0422ef86f21f8cf58dd61cb120aef716324e507a4d30f8c0b5f05c165d',
    'LICENSE': '7d8600c83754d8766cfb8a3e4c0133bae45069beb598b4895f761fc9dfd139b3',
}


@pytest.fixture(autouse=True)
def cpu_threads_and_seed():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


def x_batch(n=4):
    return torch.rand(n, 3, 114, 10)


def source_generator(out_channels):
    """Load only pinned generator/blocks classes; no torchvision imports."""
    namespace = {'torch': torch, 'nn': nn}
    tree = ast.parse((REFERENCE / 'networks' / 'blocks.py').read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    exec(compile(ast.Module(classes, type_ignores=[]), '<pinned blocks>', 'exec'), namespace)
    tree = ast.parse((REFERENCE / 'networks' / 'models.py').read_text())
    generator = [node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == 'Generator']
    exec(compile(ast.Module(generator, type_ignores=[]), '<pinned generator>', 'exec'), namespace)
    return namespace['Generator'](SimpleNamespace(dataset='gtsrb', input_channel=3),
                                  out_channels=out_channels)


def test_upstream_references_are_pinned_and_license_is_preserved():
    for path, expected in HASHES.items():
        assert hashlib.sha256((REFERENCE / path).read_bytes()).hexdigest() == expected
    assert b'Copyright (c) 2020 VinAI' in (REFERENCE / 'LICENSE').read_bytes()


def test_source_optimizer_and_mask_defaults_match_official_config():
    namespace = {}
    exec(compile((REFERENCE / 'config.py').read_text(), '<pinned config>', 'exec'), namespace)
    args = namespace['get_arguments']().parse_args([])
    cfg = source_config()
    assert cfg['generator_optimizer']['lr'] == args.lr_G == .01
    assert cfg['mask_optimizer']['lr'] == args.lr_M == .01
    assert cfg['mask_density'] == args.mask_density == .032
    assert cfg['lambda_norm'] == args.lambda_norm == 100
    assert cfg['lambda_div'] == args.lambda_div == 1
    assert cfg['epsilon'] == args.EPSILON == 1e-7
    assert cfg['mask_scheduler']['milestones'] == args.schedulerM_milestones == [10, 20]
    assert cfg['generator_scheduler']['milestones'] == args.schedulerG_milestones
    assert 'for i in range(25)' in (REFERENCE / 'train.py').read_text()
    assert cfg['mask_warmup_epochs'] == 25
    cfg['generator_optimizer']['lr'] = 42
    assert source_config()['generator_optimizer']['lr'] == .01


@pytest.mark.parametrize('out_channels', [1, 3])
@pytest.mark.parametrize('training', [False, True])
def test_padded_generator_forward_and_buffers_match_pinned_source(out_channels, training):
    adapted = CSIInputAwareGenerator(out_channels)
    upstream = source_generator(out_channels)
    upstream.load_state_dict(adapted.state_dict(), strict=True)
    adapted.train(training)
    upstream.train(training)
    x = x_batch(2)
    actual = adapted(x)
    expected = upstream(F.pad(x, (0, 6, 0, 6), mode='replicate'))[..., :114, :10]
    assert actual.shape == (2, out_channels, 114, 10)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key, value in upstream.state_dict().items():
        torch.testing.assert_close(adapted.state_dict()[key], value, rtol=0, atol=0)


def test_injection_and_mask_threshold_are_source_blend_with_dose_extension():
    attack = InputAwareCSI().eval()
    x, donor = x_batch(2), x_batch(2)
    with torch.no_grad():
        pattern = attack.pattern_generator(donor)
        mask_raw = attack.mask_generator(donor)
        mask = torch.tanh(mask_raw * 20 - 10) / (2 + 1e-7) + .5
        expected = x + (pattern - x) * mask
        actual = attack.inject_tensor(x, 1., donor=donor)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    assert actual.min() >= 0 and actual.max() <= 1


def test_zero_dose_exact_identity_without_generator_calls(monkeypatch):
    attack = InputAwareCSI()
    x = x_batch()
    def forbidden(*_args):
        raise AssertionError('zero dose must not call a generator')
    monkeypatch.setattr(attack.pattern_generator, 'forward', forbidden)
    monkeypatch.setattr(attack.mask_generator, 'forward', forbidden)
    assert attack.inject_tensor(x, 0.) is x
    assert attack.inject_tensor(x, torch.zeros(4)) is x


def test_vector_dose_zero_rows_bitwise_identity():
    attack = InputAwareCSI().eval()
    x = x_batch()
    actual = attack.inject_tensor(x, torch.tensor([0., .2, .7, 0.]))
    assert torch.equal(actual[[0, 3]], x[[0, 3]])
    assert not torch.equal(actual[1], x[1])


def test_diversity_formula_parity_and_zero_distance_finite_gradient():
    x, donor = x_batch(), x_batch()
    pattern = torch.rand(4, 1, 114, 10)
    donor_pattern = torch.rand_like(pattern)
    expected = (((x - donor).square().mean((1, 2, 3)).sqrt()) /
                ((pattern - donor_pattern).square().mean((1, 2, 3)).sqrt() + 1e-7)).mean()
    torch.testing.assert_close(diversity_ratio(x, donor, pattern, donor_pattern), expected)
    identical = torch.ones(4, 1, 114, 10, requires_grad=True)
    loss = diversity_ratio(x, donor, identical, torch.ones_like(identical))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(identical.grad).all()
    assert torch.equal(rms_distance(identical, identical), torch.zeros(4))


def test_actual_mask_step_changes_mask_not_pattern_and_freeze_sticks():
    attack = InputAwareCSI()
    before_mask = copy.deepcopy(attack.mask_generator.state_dict())
    before_pattern = copy.deepcopy(attack.pattern_generator.state_dict())
    opt = attack.make_mask_optimizer()
    result = attack.mask_step(x_batch(), x_batch(), opt)
    assert all(math.isfinite(value) for value in result.values())
    assert any(not torch.equal(value, attack.mask_generator.state_dict()[key])
               for key, value in before_mask.items())
    assert all(torch.equal(value, attack.pattern_generator.state_dict()[key])
               for key, value in before_pattern.items())
    assert opt.param_groups[0]['lr'] == .01
    assert opt.param_groups[0]['betas'] == (.5, .9)
    attack.freeze_mask().train()
    assert not attack.mask_generator.training
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in attack.mask_parameters())
    with pytest.raises(RuntimeError, match='frozen'):
        attack.mask_step(x_batch(), x_batch(), opt)


class ToyVictim(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 17 * 3)
        self.seen = None
        self.predicted = None

    def forward(self, x):
        self.seen = x.detach().clone()
        prediction = self.head(x.mean((2, 3))).reshape(-1, 1, 17, 3)
        self.predicted = prediction.detach().clone()
        return prediction, None


def joint_call(attack, victim, poison, cross, doses=None, donor=None):
    x = x_batch(4)
    y = torch.rand(4, 1, 17, 3)
    targets = y + .2
    ov = torch.optim.SGD(victim.parameters(), lr=.001, momentum=.9)
    og = attack.make_generator_optimizer()
    values = attack.train_step(victim, ov, og, x, y, donor if donor is not None else x_batch(4),
                               torch.tensor(poison), torch.tensor(cross),
                               torch.tensor(doses if doses is not None else [.7, 0., 0., 0.]),
                               targets)
    return values, x, y, targets


def test_joint_step_updates_victim_and_pattern_but_not_frozen_mask():
    attack = InputAwareCSI().freeze_mask()
    victim = ToyVictim()
    before_victim = copy.deepcopy(victim.state_dict())
    before_pattern = copy.deepcopy(attack.pattern_generator.state_dict())
    before_mask = copy.deepcopy(attack.mask_generator.state_dict())
    result, x, y, targets = joint_call(attack, victim,
        [True, False, False, False], [False, True, False, False])
    assert result['n_poison'] == result['n_cross'] == 1
    assert all(math.isfinite(value) for value in result.values())
    assert any(not torch.equal(value, victim.state_dict()[key])
               for key, value in before_victim.items())
    assert any(not torch.equal(value, attack.pattern_generator.state_dict()[key])
               for key, value in before_pattern.items())
    assert all(torch.equal(value, attack.mask_generator.state_dict()[key])
               for key, value in before_mask.items())
    assert torch.equal(victim.seen[2:], x[2:])
    # Poison labels change, cross labels do not. All joints have equal weight.
    expected_target = y.clone()
    expected_target[0] = targets[0]
    expected_loss = torch.linalg.vector_norm(victim.predicted - expected_target, dim=-1).mean()
    assert result['pose_loss'] == pytest.approx(float(expected_loss))


def test_cross_branch_uses_donor_at_full_dose_even_with_zero_poison_dose_vector():
    attack = InputAwareCSI().freeze_mask().train()
    victim = ToyVictim()
    donor = x_batch(4)
    # BN in training uses the current subgroup, not its running statistics.
    # Computing this once does not change the subsequent subgroup output.
    expected_pattern = attack.pattern_generator(donor[[1]]).detach()
    expected_mask = attack.threshold(attack.mask_generator(donor[[1]])).detach()
    result, x, _, _ = joint_call(attack, victim,
        [False] * 4, [False, True, False, False], doses=[0.] * 4, donor=donor)
    # Cross always changes the recipient despite zeros in the poison-dose vector.
    assert result['n_poison'] == 0 and result['n_cross'] == 1
    assert not torch.equal(victim.seen[1], x[1])
    expected = x[[1]] + expected_mask * (expected_pattern - x[[1]])
    torch.testing.assert_close(victim.seen[[1]], expected, rtol=0, atol=0)
    assert torch.equal(victim.seen[[0, 2, 3]], x[[0, 2, 3]])


def test_no_poison_no_cross_is_ordinary_erm_without_pattern_changes():
    attack = InputAwareCSI().freeze_mask()
    before = copy.deepcopy(attack.pattern_generator.state_dict())
    victim = ToyVictim()
    result, x, _, _ = joint_call(attack, victim, [False] * 4, [False] * 4, doses=[0.] * 4)
    assert result['diversity_loss'] == 0 and result['n_poison'] == result['n_cross'] == 0
    assert torch.equal(victim.seen, x)
    assert all(torch.equal(value, attack.pattern_generator.state_dict()[key])
               for key, value in before.items())


def test_state_dict_roundtrip_keeps_frozen_mask_and_exact_eval_injection():
    attack = InputAwareCSI().freeze_mask().eval()
    duplicate = InputAwareCSI()
    duplicate.load_state_dict(copy.deepcopy(attack.state_dict()), strict=True)
    duplicate.train()
    assert not duplicate.mask_generator.training
    assert all(not parameter.requires_grad for parameter in duplicate.mask_parameters())
    duplicate.eval()
    x = x_batch(2)
    torch.testing.assert_close(attack.inject_tensor(x, 1.), duplicate.inject_tensor(x, 1.),
                               rtol=0, atol=0)


@pytest.mark.parametrize('bad_dose', [-.1, 1.1, float('nan'), float('inf'), [0., 1., 0.]])
def test_invalid_doses_rejected(bad_dose):
    with pytest.raises(ValueError, match='dose'):
        InputAwareCSI().inject_tensor(x_batch(2), bad_dose)


@pytest.mark.parametrize('bad_input', [torch.rand(2, 3, 114, 9),
    torch.ones(2, 3, 114, 10) * 1.1, torch.ones(2, 3, 114, 10) * -.1,
    torch.full((2, 3, 114, 10), float('nan'))])
def test_wrong_shape_domain_or_nonfinite_inputs_rejected(bad_input):
    with pytest.raises(ValueError):
        InputAwareCSI().inject_tensor(bad_input, 1.)


def test_overlap_rejected_before_any_optimizer_step():
    attack = InputAwareCSI().freeze_mask()
    with pytest.raises(ValueError, match='disjoint'):
        joint_call(attack, ToyVictim(), [True, False, False, False],
                   [True, False, False, False])


def test_joint_training_requires_mask_warmup_freeze():
    with pytest.raises(RuntimeError, match='freeze_mask'):
        joint_call(InputAwareCSI(), ToyVictim(), [True, False, False, False],
                   [False, True, False, False])
