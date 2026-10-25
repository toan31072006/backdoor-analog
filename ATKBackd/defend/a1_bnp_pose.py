"""Clean-TRAIN BN-statistics pruning adapted from Zheng et al. (NeurIPS 2022).

The detector compares the frozen BN running moments from the A1 victim with
moments measured on a trusted clean TRAIN bank. It never sees the A1 trigger,
poisoned samples, or TEST. A channel whose Gaussian KL exceeds its layer mean
plus three population standard deviations is zeroed by setting BN gamma and
beta to zero. Only hidden feature-map BN2d layers are eligible; coordinate
BN1d and attention similarity normalizers have different semantics. This is
a pose-regression adaptation, not an author reproduction.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch
from torch import nn

from defend.a1_repair_ops import _batch, pose_prediction


BN_TYPE = nn.BatchNorm2d


def gaussian_kl(sample_mean, sample_var, running_mean, running_var, eps=1e-8):
    """D_KL(N_sample || N_BN) per channel, using population variances."""
    values = [sample_mean, sample_var, running_mean, running_var]
    if len({tuple(v.shape) for v in values}) != 1 or values[0].ndim != 1:
        raise ValueError('Gaussian moments must have matching channel vectors')
    if any(not bool(torch.isfinite(v).all()) for v in values):
        raise ValueError('Nonfinite BN or clean TRAIN moments')
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError('Invalid variance floor')
    sv = sample_var.double().clamp_min(eps)
    bv = running_var.double().clamp_min(eps)
    delta = sample_mean.double() - running_mean.double()
    kl = .5 * (torch.log(bv / sv) + (sv + delta.square()) / bv - 1.)
    if not bool(torch.isfinite(kl).all()):
        raise ValueError('Nonfinite BN-statistics KL')
    return kl.clamp_min(0)


def outlier_indices(scores, u=3.0):
    if scores.ndim != 1 or len(scores) < 2 or not bool(torch.isfinite(scores).all()):
        raise ValueError('BNP requires finite per-channel scores')
    if u <= 0 or not math.isfinite(u):
        raise ValueError('Invalid BNP threshold multiplier')
    mean = scores.double().mean()
    std = scores.double().std(unbiased=False)
    threshold = mean + u * std
    return torch.nonzero(scores > threshold, as_tuple=False).flatten().tolist(), float(threshold)


@torch.no_grad()
def repair_bnp(model, clean_loader, device, *, u=3.0):
    """One clean forward sweep and BN affine-channel pruning; no optimizer."""
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, BN_TYPE) and 'bn_similarity' not in name
               and module.affine
                and module.track_running_stats and module.running_mean is not None
                and module.weight is not None and module.bias is not None]
    if not modules:
        raise ValueError('BNP needs hidden affine BatchNorm2d with running statistics')
    if getattr(clean_loader, 'num_workers', 0) != 0:
        raise ValueError('BNP requires deterministic clean TRAIN loader with workers=0')
    accum = {name: dict(total=torch.zeros(module.num_features, dtype=torch.float64),
                        squares=torch.zeros(module.num_features, dtype=torch.float64),
                        count=0) for name, module in modules}
    handles = []

    def capture(name, module):
        def hook(_module, args):
            x = args[0].detach()
            if x.ndim not in (2, 3, 4) or x.shape[1] != module.num_features:
                raise ValueError('BNP encountered unexpected BN input geometry')
            dims = (0,) + tuple(range(2, x.ndim))
            count = x.numel() // x.shape[1]
            a = accum[name]
            double = x.double()
            a['total'] += double.sum(dim=dims).detach().cpu()
            a['squares'] += double.square().sum(dim=dims).detach().cpu()
            a['count'] += count
        return hook

    was_training = model.training
    model.eval()
    try:
        for name, module in modules:
            handles.append(module.register_forward_pre_hook(capture(name, module)))
        for batch in clean_loader:
            x, _ = _batch(batch, device)
            pose_prediction(model, x)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    records = []
    for name, module in modules:
        a = accum[name]
        if a['count'] <= 1:
            raise ValueError('Insufficient clean TRAIN activations for BNP: ' + name)
        sample_mean = a['total'] / a['count']
        sample_var = (a['squares'] / a['count'] - sample_mean.square()).clamp_min(0)
        scores = gaussian_kl(sample_mean, sample_var,
                             module.running_mean.detach().cpu(), module.running_var.detach().cpu())
        indices, threshold = outlier_indices(scores, u=u)
        with torch.no_grad():
            if indices:
                module.weight[indices] = 0
                module.bias[indices] = 0
        scores_bytes = scores.contiguous().numpy().tobytes()
        records.append(dict(layer=name, channels=module.num_features,
                            observed_values=a['count'], threshold=threshold,
                            pruned_indices=indices, scores_sha256=hashlib.sha256(scores_bytes).hexdigest()))
    model.eval()
    result = dict(complete=True, method='bnp_pose_adapted', source='Zheng et al., NeurIPS 2022',
                  u=float(u), clean_only=True, test_selection=False,
                  layer_scope='hidden BN2d excluding attention similarity normalizers and BN1d',
                  n_layers=len(records), n_pruned=sum(len(x['pruned_indices']) for x in records),
                  layers=records)
    json.dumps(result, allow_nan=False)
    return model, result


def smoke_bnp(backbone, model, device, physical_batch=32):
    """Exercise real-model BN hooks, pruning and a physical-B32 forward."""
    if type(physical_batch) is not int or physical_batch < 2:
        raise ValueError('BNP smoke requires physical batch >= 2')
    model = model.to(device)
    x = torch.rand(physical_batch, 3, 114, 10, device=device)
    y = torch.rand(physical_batch, 17, 3, device=device)
    model, receipt = repair_bnp(model, [(x[:8], y[:8])], device)
    with torch.no_grad():
        output = pose_prediction(model, x)
    if not bool(torch.isfinite(output).all()):
        raise ValueError('Nonfinite BNP real-model output')
    return dict(backbone=backbone, method='bnp', physical_batch=physical_batch,
                finite=True, n_layers=receipt['n_layers'], n_pruned=receipt['n_pruned'])
