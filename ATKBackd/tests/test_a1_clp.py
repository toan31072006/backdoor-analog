"""CPU mechanism tests; not full MM-Fi or defense efficacy evidence."""
import copy
import json
import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defend import a1_clp as clp


def scalar_model(scores, *, gamma=None, beta=None, variance=None):
    n = len(scores)
    model = nn.Sequential(nn.Conv2d(1, n, 1, bias=False), nn.BatchNorm2d(n))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(scores).reshape(n, 1, 1, 1))
        model[1].weight.copy_(torch.tensor(gamma if gamma is not None else [1.] * n))
        model[1].bias.copy_(torch.tensor(beta if beta is not None else [0.] * n))
        model[1].running_var.copy_(torch.tensor(variance if variance is not None else [1.] * n))
    return model


def assert_state_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_spectral_score_is_matrix_singular_value_not_flattened_norm():
    model = nn.Sequential(nn.Conv2d(2, 3, (1, 2), bias=False), nn.BatchNorm2d(3))
    matrices = torch.tensor([[[3., 0.], [0., 4.]],
                             [[0., 0.], [0., 2.]],
                             [[1., 1.], [1., 1.]]])
    with torch.no_grad():
        model[0].weight.copy_(matrices.reshape(3, 2, 1, 2))
        model[1].weight.copy_(torch.tensor([-2., .5, 3.]))
        model[1].running_var.copy_(torch.tensor([4., 1., 9.]))
    _, meta = clp.prune_clp(model)
    assert meta['layers'][0]['scores'] == pytest.approx([4., 1., 2.], abs=1e-6)
    assert meta['layers'][0]['scores'][0] != pytest.approx(5.)


def test_bn_eps_is_not_added_to_source_variance_rule():
    model = scalar_model([2., 4., 6.], variance=[4., 4., 4.])
    model[1].eps = 100.
    _, meta = clp.prune_clp(model)
    assert meta['layers'][0]['scores'] == pytest.approx([1., 2., 3.])
    assert meta['variance_epsilon_added'] is False


def test_sample_std_not_population_std_and_no_threshold_sweep():
    model = scalar_model([0., 0., 1.])
    repaired, meta = clp.prune_clp(model, u=1.2)
    layer = meta['layers'][0]
    assert layer['score_sample_std'] == pytest.approx(1 / math.sqrt(3))
    assert layer['threshold'] > 1.
    assert layer['pruned_indices'] == []
    assert_state_equal(model.state_dict(), repaired.state_dict())
    assert meta['strength_candidates'] == [1.2]


def test_strict_threshold_comparison_and_zero_gamma_and_nonzero_beta():
    model = scalar_model([0., 1., 2.], beta=[.2, 5., -7.])
    repaired, meta = clp.prune_clp(model, u=0)
    assert meta['layers'][0]['threshold'] == pytest.approx(1.)
    assert meta['layers'][0]['pruned_indices'] == [2]
    assert torch.equal(repaired[1].weight, torch.tensor([1., 1., 0.]))
    assert torch.equal(repaired[1].bias, torch.tensor([.2, 5., 0.]))


def test_default_u_three_prunes_extreme_channel_and_preserves_everything_else():
    model = scalar_model([1.] * 63 + [20.], beta=[.5] * 64)
    model.train()
    before = copy.deepcopy(model.state_dict())
    rng_before = torch.random.get_rng_state().clone()
    repaired, meta = clp.prune_clp(model)
    assert repaired is not model and not repaired.training and model.training
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert_state_equal(before, model.state_dict())
    assert meta['pruned_channels'] == 1 and meta['u'] == 3.
    assert meta['training_steps'] == 0 and meta['fine_tuning_epochs'] == 0
    assert not meta['uses_data'] and not meta['uses_labels'] and not meta['uses_test_selection']
    assert torch.equal(repaired[0].weight, model[0].weight)
    for name in ('running_mean', 'running_var', 'num_batches_tracked'):
        assert torch.equal(getattr(repaired[1], name), getattr(model[1], name))
    assert repaired[1].weight[-1] == 0 and repaired[1].bias[-1] == 0
    json.dumps(meta, allow_nan=False)


def test_equal_channel_scores_prune_nothing():
    model = scalar_model([1.] * 5)
    repaired, meta = clp.prune_clp(model, u=0)
    assert meta['layers'][0]['score_sample_std'] == 0.
    assert meta['pruned_channels'] == 0
    assert_state_equal(model.state_dict(), repaired.state_dict())


def test_depthwise_channel_matrix_handles_groups():
    model = nn.Sequential(nn.Conv2d(4, 4, (1, 2), groups=4, bias=False), nn.BatchNorm2d(4))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[3., 4.], [0., 2.], [1., 0.], [0., 0.]]).reshape(4, 1, 1, 2))
    _, meta = clp.prune_clp(model, u=0)
    assert meta['layers'][0]['scores'] == pytest.approx([5., 2., 1., 0.])
    assert meta['layers'][0]['conv_groups'] == 4
    assert meta['layers'][0]['pruned_indices'] == [0]


@pytest.mark.parametrize('value', [True, False, -1., float('nan'), float('inf'), -float('inf'), '3', None, [3.]])
def test_invalid_strength_fails(value):
    with pytest.raises(ValueError, match='u must'):
        clp.prune_clp(scalar_model([1., 2.]), u=value)


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_running_variance_fails_without_mutating_input(value):
    model = scalar_model([1., 2.])
    with torch.no_grad():
        model[1].running_var[0] = value
    original = model[0].weight.clone()
    with pytest.raises(ValueError, match='variance|nonfinite'):
        clp.prune_clp(model)
    assert torch.equal(original, model[0].weight)
    assert model[1].weight[0] == 1


@pytest.mark.parametrize('location', ['conv', 'gamma', 'beta'])
def test_nonfinite_parameters_fail(location):
    model = scalar_model([1., 2.])
    with torch.no_grad():
        target = {'conv': model[0].weight, 'gamma': model[1].weight, 'beta': model[1].bias}[location]
        target.flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        clp.prune_clp(model)


def test_affine_and_running_statistics_are_required():
    for norm in (nn.BatchNorm2d(2, affine=False), nn.BatchNorm2d(2, track_running_stats=False)):
        with pytest.raises(ValueError, match='requires'):
            clp.prune_clp(nn.Sequential(nn.Conv2d(1, 2, 1), norm))


def test_channel_mismatch_and_one_channel_fail_closed():
    with pytest.raises(ValueError, match='mismatch'):
        clp.prune_clp(nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(3)))
    with pytest.raises(ValueError, match='at least two'):
        clp.prune_clp(scalar_model([1.]))


def test_unknown_unpaired_bn_cannot_be_silently_skipped_or_allowed():
    model = nn.Sequential(scalar_model([1., 2.]), nn.BatchNorm2d(2))
    with pytest.raises(ValueError, match='undeclared'):
        clp.prune_clp(model)
    with pytest.raises(ValueError, match='unsupported.*skip'):
        clp.prune_clp(model, allow_unpaired_bn2d=('1',))


@pytest.mark.parametrize('names', [('1',), ('1', '1'), '1', (1,)])
def test_stale_or_invalid_skip_declarations_fail(names):
    with pytest.raises(ValueError):
        clp.prune_clp(scalar_model([1., 2.]), allow_unpaired_bn2d=names)


def test_bn1d_is_explicitly_counted_but_untouched():
    model = nn.Sequential(scalar_model([1., 1., 9.]), nn.BatchNorm1d(3))
    before = copy.deepcopy(model[1].state_dict())
    repaired, meta = clp.prune_clp(model, u=0)
    assert meta['unsupported_bn1d_count'] == 1
    assert meta['unsupported_bn1d_names'] == ['1']
    assert_state_equal(before, repaired[1].state_dict())


def test_nonadjacent_layer_is_not_a_pair():
    model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.ReLU(), nn.BatchNorm2d(2))
    with pytest.raises(ValueError, match='undeclared'):
        clp.prune_clp(model)


def test_no_direct_pair_and_nonmodule_fail():
    with pytest.raises(ValueError, match='at least one'):
        clp.prune_clp(nn.Linear(2, 3))
    with pytest.raises(TypeError, match='nn.Module'):
        clp.prune_clp(None)


def test_shared_batchnorm_and_convolution_modules_fail():
    conv, norm = nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2)
    with pytest.raises(ValueError, match='shared BatchNorm2d'):
        clp.prune_clp(nn.Sequential(nn.Sequential(conv, norm), nn.Sequential(conv, norm)))
    with pytest.raises(ValueError, match='shared Conv2d'):
        clp.prune_clp(nn.Sequential(nn.Sequential(conv, nn.BatchNorm2d(2)),
                                   nn.Sequential(conv, nn.BatchNorm2d(2))))


def test_float64_scoring_is_not_silently_downcast():
    model = scalar_model([1., 2., 3.]).double()
    _, meta = clp.prune_clp(model)
    assert meta['layers'][0]['score_dtype'] == 'float64'


def test_operation_never_calls_model_forward():
    class WeightOnly(nn.Sequential):
        def forward(self, inputs):
            raise AssertionError('CLP must not consume data')
    source = scalar_model([1., 2.])
    model = WeightOnly(*list(source.children()))
    clp.prune_clp(model)


def test_hpeli_explicit_limited_coverage_and_real_forward():
    from models.hpeli import HPELiNet
    torch.set_num_threads(1)
    model = HPELiNet(num_keypoints=17, subcarrier_num=114, dataset='mmfi').eval()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='undeclared'):
        clp.prune_clp(model)
    repaired, meta = clp.prune_clp(model, allow_unpaired_bn2d=clp.HPELI_UNPAIRED_BN2D)
    assert meta['paired_bn2d_count'] == 10 and meta['total_bn2d_count'] == 12
    assert meta['paired_channels'] == 960
    assert meta['unsupported_bn1d_count'] == 2
    assert [row['name'] for row in meta['skipped_bn2d']] == list(clp.HPELI_UNPAIRED_BN2D)
    for name in clp.HPELI_UNPAIRED_BN2D:
        assert_state_equal(model.get_submodule(name).state_dict(), repaired.get_submodule(name).state_dict())
    assert_state_equal(before, model.state_dict())
    with torch.no_grad():
        pose, feature = repaired(torch.zeros(1, 3, 114, 10))
    assert pose.shape == (1, 1, 17, 3) and feature.shape == (1, 128)
    assert torch.isfinite(pose).all() and torch.isfinite(feature).all()
