"""Weight-only CLP mechanism for explicitly paired Conv2d/BatchNorm2d layers.

This is an independent implementation of the channel-Lipschitz pruning rule
used by pinned BackdoorBench CLP. It is not activation-based Fine-Pruning.
No clean bank, labels, test scores, optimization, or strength sweep is used.
Only structurally adjacent Conv2d -> BatchNorm2d pairs are interpreted as a
single channel transform. Composite SKConv output norms are not such pairs.
"""
from __future__ import annotations

import copy
import math
from numbers import Real

import torch
from torch import nn


DEFAULT_U = 3.0
HPELI_UNPAIRED_BN2D = ('skunit1.conv2_sk.1', 'skunit2.conv2_sk.1')


def _strength(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('CLP u must be a finite nonnegative scalar')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('CLP u must be a finite nonnegative scalar')
    return value


def _name(parent, child):
    return f'{parent}.{child}' if parent else child


def _pairing(model, allow_unpaired_bn2d):
    if isinstance(allow_unpaired_bn2d, str):
        raise ValueError('CLP allow_unpaired_bn2d must be a sequence of full names')
    allowed = tuple(allow_unpaired_bn2d)
    if any(not isinstance(x, str) or not x for x in allowed) or len(set(allowed)) != len(allowed):
        raise ValueError('CLP skipped BatchNorm2d names must be unique nonempty strings')
    modules = list(model.named_modules(remove_duplicate=False))
    norms = {name: module for name, module in modules if isinstance(module, nn.BatchNorm2d)}
    bn1d = [name for name, module in modules if isinstance(module, nn.BatchNorm1d)]
    if len({id(module) for module in norms.values()}) != len(norms):
        raise ValueError('CLP does not support shared BatchNorm2d module aliases')
    pairs = []
    for name, module in modules:
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module._modules.items())
        for (left_name, left), (right_name, right) in zip(children, children[1:]):
            if isinstance(left, nn.Conv2d) and isinstance(right, nn.BatchNorm2d):
                pairs.append((_name(name, left_name), left, _name(name, right_name), right))
    paired_names = [p[2] for p in pairs]
    if len(set(paired_names)) != len(paired_names):
        raise ValueError('CLP BatchNorm2d was mapped more than once')
    if len({id(p[1]) for p in pairs}) != len(pairs):
        raise ValueError('CLP does not support shared Conv2d module aliases')
    unpaired = set(norms) - set(paired_names)
    if set(allowed) != unpaired:
        missing = sorted(unpaired - set(allowed))
        stale = sorted(set(allowed) - unpaired)
        raise ValueError(f'CLP unpaired BatchNorm2d coverage mismatch: undeclared={missing}, stale={stale}')
    if not pairs:
        raise ValueError('CLP requires at least one direct Conv2d -> BatchNorm2d pair')
    skipped = []
    for name in allowed:
        # The production exception is intentionally specific: the composite
        # SKConv output has no single convolution to insert into the CLP rule.
        if name not in HPELI_UNPAIRED_BN2D:
            raise ValueError(f'CLP unsupported BatchNorm2d skip declaration: {name}')
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        children = list(parent._modules.items()) if isinstance(parent, nn.Sequential) else []
        index = next((i for i, (n, _) in enumerate(children) if n == child_name), -1)
        predecessor = children[index - 1][1] if index > 0 else None
        if predecessor is None or type(predecessor).__name__ != 'SKConv' or not hasattr(predecessor, 'convs'):
            raise ValueError(f'CLP skip is not the declared HPELi composite SKConv output: {name}')
        skipped.append({'name': name, 'channels': norms[name].num_features,
                        'reason': 'composite SKConv output; no single direct Conv2d pair'})
    return pairs, norms, bn1d, skipped


def _channel_scores(conv, norm, label):
    if conv.out_channels != norm.num_features:
        raise ValueError(f'CLP channel mismatch for {label}')
    if norm.num_features < 2:
        raise ValueError(f'CLP sample standard deviation requires at least two channels: {label}')
    if not norm.affine or norm.weight is None or norm.bias is None:
        raise ValueError(f'CLP requires affine BatchNorm2d: {label}')
    if not norm.track_running_stats or norm.running_var is None:
        raise ValueError(f'CLP requires BatchNorm2d running variance: {label}')
    # CPU and source dtype make this data-free operation independent of the
    # current CUDA device. Half/bfloat16 SVD is promoted and declared below.
    dtype = torch.float64 if conv.weight.dtype == torch.float64 else torch.float32
    weights = conv.weight.detach().to(device='cpu', dtype=dtype)
    gamma = norm.weight.detach().to(device='cpu', dtype=dtype)
    beta = norm.bias.detach().to(device='cpu', dtype=dtype)
    variance = norm.running_var.detach().to(device='cpu', dtype=dtype)
    if not all(torch.isfinite(v).all().item() for v in (weights, gamma, beta, variance)):
        raise ValueError(f'CLP nonfinite weights/BatchNorm parameters: {label}')
    if not torch.all(variance > 0).item():
        raise ValueError(f'CLP running variance must be strictly positive: {label}')
    # Each output kernel is (Cin/groups, Kh*Kw), including depthwise groups.
    # No epsilon is added: pinned CLP uses sqrt(running_var), not BN's eps.
    matrices = weights.reshape(conv.out_channels, conv.in_channels // conv.groups, -1)
    sigma = torch.linalg.svdvals(matrices)[:, 0]
    scores = sigma * gamma.abs() / variance.sqrt()
    if not torch.isfinite(scores).all().item():
        raise ValueError(f'CLP nonfinite channel score: {label}')
    return scores, str(dtype).removeprefix('torch.')


def prune_clp(model, u=DEFAULT_U, allow_unpaired_bn2d=()):
    """Return an eval-mode deep copy and JSON-safe audit metadata.

``u`` is the fixed outlier strength, not a test-tuned parameter. The production
HPELi caller explicitly supplies ``HPELI_UNPAIRED_BN2D``; all other unmatched
BatchNorm2d layers fail closed. BatchNorm1d is outside the source mechanism and
is left untouched with explicit coverage disclosure.
    """
    if not isinstance(model, nn.Module):
        raise TypeError('CLP model must be an nn.Module')
    u = _strength(u)
    pairs, norms, bn1d, skipped = _pairing(model, allow_unpaired_bn2d)
    layers = []
    # Audit every supported layer before copying or applying any pruning.
    for conv_name, conv, bn_name, norm in pairs:
        scores, dtype = _channel_scores(conv, norm, bn_name)
        mean = scores.mean()
        std = scores.std(correction=1)
        threshold = mean + u * std
        if not torch.isfinite(threshold).item():
            raise ValueError(f'CLP nonfinite pruning threshold: {bn_name}')
        indices = torch.nonzero(scores > threshold, as_tuple=False).flatten().tolist()
        layers.append({'conv_name': conv_name, 'bn_name': bn_name,
                       'channels': conv.out_channels, 'conv_groups': conv.groups,
                       'score_dtype': dtype, 'scores': scores.tolist(),
                       'score_mean': mean.item(), 'score_sample_std': std.item(),
                       'threshold': threshold.item(), 'pruned_indices': indices,
                       'pruned_channels': len(indices)})
    repaired = copy.deepcopy(model)
    with torch.no_grad():
        for layer in layers:
            norm = repaired.get_submodule(layer['bn_name'])
            norm.weight[layer['pruned_indices']] = 0
            norm.bias[layer['pruned_indices']] = 0
    repaired.eval()
    metadata = {
        'schema': 'a1_clp_weight_only_v1', 'method': 'clp', 'u': u,
        'score_rule': 'sigma_max(kernel[Cin/groups,Kh*Kw]) * abs(BN.gamma) / sqrt(BN.running_var)',
        'variance_epsilon_added': False, 'std_correction': 1,
        'prune_rule': 'score > mean + u * sample_std',
        'zeroed_parameters': ['BatchNorm2d.weight', 'BatchNorm2d.bias'],
        'uses_data': False, 'uses_labels': False, 'uses_test_selection': False,
        'training_steps': 0, 'fine_tuning_epochs': 0, 'strength_candidates': [u],
        'paired_bn2d_count': len(pairs), 'total_bn2d_count': len(norms),
        'paired_channels': sum(layer['channels'] for layer in layers),
        'pruned_channels': sum(layer['pruned_channels'] for layer in layers),
        'skipped_bn2d': skipped, 'unsupported_bn1d_names': bn1d,
        'unsupported_bn1d_count': len(bn1d), 'layers': layers,
        'scope': 'direct Conv2d -> BatchNorm2d pairs only; not full composite-layer coverage',
    }
    return repaired, metadata
