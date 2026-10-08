"""Victim-loss contract tests for the paper's ordinary-ERM protocol.

The historical suite exercised a poison-mask-aware decomposed loss. The paper
instead states that the victim receives ordinary ``(CSI, pose)`` pairs and
minimizes standard MPJPE, so these tests make attack metadata impossible to
pass to the loss/update APIs.
"""

import inspect
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_backdoor import (  # noqa: E402
    _mpjpe_loss,
    _resolve_training_config,
    _validate_training_contract,
    _victim_update,
)
from run_experiments import _apply_model_overrides  # noqa: E402


def test_mpjpe_matches_equation_over_all_samples_and_joints():
    generator = torch.Generator().manual_seed(7)
    pred = torch.randn(5, 1, 17, 3, generator=generator)
    target = torch.randn(5, 1, 17, 3, generator=generator)

    expected = torch.linalg.vector_norm(pred - target, dim=-1).mean()
    torch.testing.assert_close(_mpjpe_loss(pred, target), expected)


def test_mpjpe_gives_each_joint_ordinary_one_over_j_weight():
    # One joint with error J has the same mean error as J joints with error 1.
    # A target-subchain/group-balanced objective would not satisfy this check.
    n_joints = 17
    target = torch.zeros(1, 1, n_joints, 3)
    one_joint = target.clone()
    one_joint[..., 12, 0] = float(n_joints)
    every_joint = target.clone()
    every_joint[..., 0] = 1.0

    torch.testing.assert_close(_mpjpe_loss(one_joint, target), torch.tensor(1.0))
    torch.testing.assert_close(_mpjpe_loss(every_joint, target), torch.tensor(1.0))


def test_mpjpe_is_differentiable():
    pred = torch.ones(2, 1, 17, 3, requires_grad=True)
    target = torch.zeros_like(pred)
    loss = _mpjpe_loss(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad) == pred.numel()


@pytest.mark.parametrize(
    ('pred_shape', 'target_shape'),
    [((2, 17, 3), (2, 16, 3)), ((17, 3), (17, 3))],
)
def test_mpjpe_rejects_non_pose_or_mismatched_shapes(pred_shape, target_shape):
    with pytest.raises(ValueError):
        _mpjpe_loss(torch.zeros(pred_shape), torch.zeros(target_shape))


def test_victim_apis_cannot_receive_attack_metadata():
    assert tuple(inspect.signature(_mpjpe_loss).parameters) == ('pred', 'target')
    assert tuple(inspect.signature(_victim_update).parameters) == (
        'model', 'optimizer', 'csi', 'target')


def test_training_contract_accepts_only_standard_mpjpe():
    _validate_training_contract({})
    _validate_training_contract({'victim_loss': 'MPJPE'})

    with pytest.raises(ValueError, match='victim_loss'):
        _validate_training_contract({'victim_loss': 'smooth_l1'})
    for legacy_key in ('lambda_target', 'lambda_nontarget', 'loss_norm'):
        with pytest.raises(ValueError, match='attack-specific'):
            _validate_training_contract({legacy_key: 1.0})


def test_resolved_config_records_the_exact_paper_optimizer_recipe():
    mmfi = _resolve_training_config({
        'experiment_name': 'mmfi', 'model': 'hpeli',
    })
    assert mmfi['victim_loss'] == 'mpjpe'
    assert mmfi['optimizer'] == 'sgd'
    assert mmfi['momentum'] == pytest.approx(0.9)
    assert mmfi['weight_decay'] == pytest.approx(0.0)
    assert mmfi['seed'] == 42
    assert mmfi['poison_select'] == 'uniform'
    assert mmfi['dose_grid'] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert mmfi['mmfi_protocol'] == 'protocol1'
    assert mmfi['mmfi_setting'] == 's1'
    assert mmfi['mmfi_random_ratio'] == pytest.approx(0.8)
    assert mmfi['mmfi_split_seed'] == 0
    assert mmfi['num_person'] == 1

    piw = _resolve_training_config({
        'experiment_name': 'one-person', 'model': 'hpeli',
    })
    assert piw['victim_loss'] == 'mpjpe'
    assert piw['optimizer'] == 'adamw'
    assert piw['weight_decay'] == pytest.approx(0.01)
    assert piw['payload_axis'] == pytest.approx([0.0, 0.0, 1.0])

    tsba = _resolve_training_config({
        'experiment_name': 'mmfi', 'model': 'hpeli', 'trigger': 'tsba',
    })
    assert tsba['tsba_eps'] == pytest.approx(0.1)
    assert tsba['tsba_hidden'] == 32
    assert tsba['tsba_lr'] == pytest.approx(1e-3)
    assert tsba['tsba_warmup_epochs'] == 10
    assert tsba['tsba_generator_steps'] == 4


@pytest.mark.parametrize('learning_rate', [1e-3, 1e-2])
def test_piw3d_runner_preserves_configured_learning_rate(learning_rate):
    cfg = _apply_model_overrides(
        {'lr': learning_rate, 'optimizer': 'adamw'},
        model='hpeli', dataset_name='person-in-wifi-3d')
    assert cfg['lr'] == pytest.approx(learning_rate)
    assert cfg['optimizer'] == 'adamw'
    assert cfg['weight_decay'] == pytest.approx(0.01)


def test_piw3d_runner_defaults_to_original_wbackdoor_learning_rate():
    cfg = _apply_model_overrides(
        {}, model='hpeli', dataset_name='person-in-wifi-3d')
    assert cfg['lr'] == pytest.approx(1e-3)


class _ScalePose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, csi):
        return self.scale * csi, None


def test_victim_update_reports_pre_update_standard_mpjpe():
    model = _ScalePose()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    csi = torch.ones(2, 1, 17, 3)
    target = torch.zeros_like(csi)
    expected = float(_mpjpe_loss(model(csi)[0], target).detach())

    observed = _victim_update(model, optimizer, csi, target)

    assert observed == pytest.approx(expected)
    assert model.scale.item() < 1.0
