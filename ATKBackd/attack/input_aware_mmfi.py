"""Input-Aware Dynamic Backdoor, explicitly adapted to MM-Fi pose regression.

Reference: Nguyen and Tran, NeurIPS 2020; VinAI MIT implementation pinned at
1750e4ff0cc2371b243f1fd0ee18971f7e49fe78. See third_party/input_aware/NOTICE.md.

Preserved: independent pattern/mask generators, source encoder/decoder widths,
sharp mask threshold, mask warm-up objective, frozen mask, joint victim/pattern
updates, clean-labelled cross-trigger branch, and diversity-ratio objective.

Adaptations: normalized CSI is already [0, 1]; pad/crop, not resize, accommodates
114x10 rather than 32x32 images. Labels/loss are full-pose targets/ordinary MPJPE,
not classes/cross entropy. The runner fixes poison/cross identities from TRAIN,
uses a paired dose, and retains the HPE-Li victim recipe. A second independently
shuffled TRAIN stream supplies donors. Diversity pairs each poisoned recipient
with its donor (rather than requiring equal poison/cross counts in every batch).
Both optimizers are zeroed per step; the upstream missing generator zero_grad is
not reproduced. A vector norm implements RMS with a finite subgradient at zero.

This is a white-box, joint-training CSI/HPE adaptation, not an exact reproduction
of the classification attack or a data-only attacker. No post-normalization,
amplitude clipping budget, image normalization, or TEST-dependent tuning is used.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


UPSTREAM_COMMIT = '1750e4ff0cc2371b243f1fd0ee18971f7e49fe78'
_EPS = 1e-7
_SOURCE_CONFIG = {
    'method': 'input_aware_dynamic_csi_adapted',
    'upstream_repository': 'https://github.com/VinAIResearch/input-aware-backdoor-attack-release',
    'upstream_commit': UPSTREAM_COMMIT,
    'upstream_license': 'MIT, Copyright (c) 2020 VinAI',
    'input_shape': [3, 114, 10],
    'input_domain': 'authors processed normalized MM-Fi amplitude [0,1]',
    'padding': {'bottom': 6, 'right': 6, 'mode': 'replicate',
                'network_shape': [3, 120, 16], 'output': 'crop to original shape'},
    'generator_channels': [32, 64, 128],
    'generator_downsampling_steps': 3,
    'batchnorm_eps': 1e-5,
    'batchnorm_momentum': 0.05,
    'pattern_rule': 'tanh(raw)/(2+1e-7)+0.5; no image-domain normalization',
    'mask_rule': 'tanh(mask_output*20-10)/(2+1e-7)+0.5',
    'injection': 'x + dose * threshold(M(donor)) * (G(donor)-x)',
    'cross_training_dose': 1.0,
    'dose_zero': 'exact input identity without generator calls',
    'mask_warmup_epochs': 25,
    'mask_density': 0.032,
    'lambda_norm': 100.0,
    'lambda_div': 1.0,
    'epsilon': _EPS,
    'generator_optimizer': {'name': 'Adam', 'lr': 0.01, 'betas': [0.5, 0.9],
                            'weight_decay': 0.0, 'eps': 1e-8},
    'mask_optimizer': {'name': 'Adam', 'lr': 0.01, 'betas': [0.5, 0.9],
                       'weight_decay': 0.0, 'eps': 1e-8},
    'generator_scheduler': {'name': 'MultiStepLR',
                            'milestones': [200, 300, 400, 500], 'gamma': 0.1},
    'mask_scheduler': {'name': 'MultiStepLR', 'milestones': [10, 20], 'gamma': 0.1},
    'upstream_attack_probability': 0.1,
    'upstream_cross_probability': 0.1,
    'mask_frozen_during_victim_training': True,
    'task_loss': 'ordinary full-pose MPJPE in pose units, no target-joint weighting',
    'attacker_access': ['clean TRAIN CSI/pose', 'victim parameters and gradients',
                        'joint victim/generator training',
                        'independent TRAIN donor CSI; no TEST fitting'],
    'adaptation_differences': [
        '3x114x10 CSI padded/cropped instead of 32x32 images',
        'CSI already [0,1]; no CIFAR channel normalization',
        'pose targets and ordinary MPJPE replace class labels and cross entropy',
        'paired doses scale the source blend; zero is an exact identity',
        'fixed shared TRAIN poison identities and disjoint fixed cross recipients',
        'diversity uses independently shuffled donor for every poisoned recipient',
        'victim HPE-Li SGD recipe/50 epochs rather than source classifier recipe',
        '25 additional TRAIN-only mask warm-up epochs are attacker compute',
        'explicit generator zero_grad each step; no unintended gradient accumulation',
        'RMS vector norm gives a finite zero-distance subgradient',
        'redundant upstream mask forward before overwrite is omitted',
    ],
}


def source_config() -> dict:
    """Return a fresh JSON-serializable, predeclared source/adaptation contract."""
    return copy.deepcopy(_SOURCE_CONFIG)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, relu: bool = True):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05)
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class _DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.maxpooling(x)


class _UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear',
                                   align_corners=False)

    def forward(self, x):
        return self.upsample(x)


class CSIInputAwareGenerator(nn.Sequential):
    """Source non-MNIST generator, with an explicit CSI pad/crop boundary."""

    def __init__(self, out_channels: int = 3):
        if out_channels not in (1, 3):
            raise ValueError('Input-aware pattern/mask output channels must be 3/1')
        layers = OrderedDict()
        current, following = 3, 32
        for step in range(3):
            layers[f'convblock_down_{2 * step}'] = _ConvBlock(current, following)
            layers[f'convblock_down_{2 * step + 1}'] = _ConvBlock(following, following)
            layers[f'downsample_{step}'] = _DownSample()
            if step < 2:
                current, following = following, following * 2
        layers['convblock_middle'] = _ConvBlock(following, following)
        current, following = following, following // 2
        for step in range(3):
            layers[f'upsample_{step}'] = _UpSample()
            layers[f'convblock_up_{2 * step}'] = _ConvBlock(current, current)
            layers[f'convblock_up_{2 * step + 1}'] = _ConvBlock(
                current, following, relu=step != 2)
            current, following = following, following // 2
            if step == 1:
                following = out_channels
        super().__init__(layers)

    def forward(self, x):
        height, width = x.shape[-2:]
        bottom, right = (-height) % 8, (-width) % 8
        x = F.pad(x, (0, right, 0, bottom), mode='replicate')
        for layer in self.children():
            x = layer(x)
        return (torch.tanh(x) / (2.0 + _EPS) + 0.5)[..., :height, :width]


def rms_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Source sqrt(mean((x-y)**2)), with a finite zero-distance subgradient."""
    if x.shape != y.shape or x.ndim < 2:
        raise ValueError('RMS arguments need matching batched tensor shapes')
    diff = (x - y).flatten(1)
    return torch.linalg.vector_norm(diff, dim=1) / math.sqrt(diff.shape[1])


def diversity_ratio(x, donor, pattern, donor_pattern):
    return (rms_distance(x, donor) /
            (rms_distance(pattern, donor_pattern) + _EPS)).mean()


def _check_gradients(parameters, label):
    checks = [torch.isfinite(parameter.grad).all() for parameter in parameters
              if parameter.grad is not None]
    # Keep the same fail-closed test, with one host/GPU synchronization rather
    # than one synchronization for every parameter tensor.
    if checks and not bool(torch.stack(checks).all()):
        raise FloatingPointError(f'Nonfinite {label} gradient; no optimizer step')


class InputAwareCSI(nn.Module):
    """Two-generator CSI attack with separately serialized mask/pattern state."""

    requires_deferred_injection = True
    requires_stochastic_trigger_state = True

    def __init__(self):
        super().__init__()
        self.pattern_generator = CSIInputAwareGenerator(out_channels=3)
        self.mask_generator = CSIInputAwareGenerator(out_channels=1)
        self.register_buffer('_mask_frozen', torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def source_config():
        return source_config()

    @staticmethod
    def _validate_x(x, name='CSI'):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 114, 10) or x.shape[0] == 0:
            raise ValueError(f'{name} must be nonempty (B,3,114,10), found {tuple(x.shape)}')
        if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
            raise ValueError(f'{name} must contain finite floating-point values')
        if bool((x < -1e-6).any()) or bool((x > 1.0 + 1e-6).any()):
            raise ValueError(f'{name} must use processed normalized amplitude [0,1]')

    @staticmethod
    def _doses(x, dose):
        doses = torch.as_tensor(dose, device=x.device, dtype=x.dtype)
        if doses.ndim == 0:
            doses = doses.expand(x.shape[0])
        if doses.shape != (x.shape[0],):
            raise ValueError(f'dose must be scalar or ({x.shape[0]},)')
        if not bool(torch.isfinite(doses).all()) or bool(((doses < 0) | (doses > 1)).any()):
            raise ValueError('dose must be finite in [0,1]')
        return doses

    @staticmethod
    def _selection(selection, x, name):
        selection = torch.as_tensor(selection, device=x.device)
        if selection.dtype != torch.bool or selection.shape != (x.shape[0],):
            raise ValueError(f'{name} must be a bool vector of shape ({x.shape[0]},)')
        return selection

    @staticmethod
    def threshold(mask):
        return torch.tanh(mask * 20.0 - 10.0) / (2.0 + _EPS) + 0.5

    def generator_parameters(self):
        return self.pattern_generator.parameters()

    def mask_parameters(self):
        return self.mask_generator.parameters()

    def make_mask_optimizer(self):
        return torch.optim.Adam(self.mask_parameters(), lr=0.01, betas=(0.5, 0.9),
                                eps=1e-8, weight_decay=0.0)

    def make_generator_optimizer(self):
        return torch.optim.Adam(self.generator_parameters(), lr=0.01,
                                betas=(0.5, 0.9), eps=1e-8, weight_decay=0.0)

    @staticmethod
    def make_mask_scheduler(optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1)

    @staticmethod
    def make_generator_scheduler(optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 300, 400, 500],
                                                    gamma=0.1)

    def train(self, mode=True):
        super().train(mode)
        if bool(self._mask_frozen):
            self.mask_generator.requires_grad_(False)
            self.mask_generator.eval()
        return self

    def freeze_mask(self):
        self._mask_frozen.fill_(True)
        self.mask_generator.requires_grad_(False)
        self.mask_generator.eval()
        for parameter in self.mask_parameters():
            parameter.grad = None
        return self

    def _parts(self, donor):
        pattern = self.pattern_generator(donor)
        mask = self.threshold(self.mask_generator(donor))
        return pattern, mask

    def inject_tensor(self, x, dose, donor=None):
        self._validate_x(x)
        doses = self._doses(x, dose)
        if donor is None:
            donor = x
        self._validate_x(donor, 'donor CSI')
        if donor.shape != x.shape or donor.device != x.device or donor.dtype != x.dtype:
            raise ValueError('donor must match recipient shape/device/dtype')
        if not bool((doses != 0).any()):
            return x
        pattern, mask = self._parts(donor)
        blended = x + doses[:, None, None, None] * mask * (pattern - x)
        return torch.where(doses[:, None, None, None] == 0, x, blended)

    def mask_step(self, x, donor, optimizer):
        """One source mask-warmup update; no victim model or TEST data involved."""
        self._validate_x(x)
        self._validate_x(donor, 'donor CSI')
        if donor.shape != x.shape or donor.device != x.device or donor.dtype != x.dtype:
            raise ValueError('mask donors must match recipients')
        if bool(self._mask_frozen):
            raise RuntimeError('Mask is frozen; do not restart mask warm-up')
        self.mask_generator.train()
        optimizer.zero_grad(set_to_none=True)
        mask = self.threshold(self.mask_generator(x))
        donor_mask = self.threshold(self.mask_generator(donor))
        loss_div = diversity_ratio(x, donor, mask, donor_mask)
        loss_norm = F.relu(mask - 0.032).mean()
        # Upstream applies lambda_div inside loss_div and outside again; its
        # fixed default is 1, hence exactly the following locked expression.
        total = 100.0 * loss_norm + loss_div
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('Nonfinite mask loss; no optimizer step')
        total.backward()
        _check_gradients(self.mask_parameters(), 'mask')
        optimizer.step()
        return {'total_loss': float(total.detach()), 'mask_norm_loss': float(loss_norm.detach()),
                'diversity_loss': float(loss_div.detach()), 'n_samples': int(x.shape[0])}

    def train_step(self, victim, opt_victim, opt_generator, x, y, donor,
                   poison_mask, cross_mask, doses, attack_targets):
        """One joint source-style victim/pattern update using ordinary MPJPE.

        ``y`` is the clean recipient pose. ``attack_targets`` contains the common
        P3 mapping for each recipient/dose; only poison rows replace clean labels.
        Cross rows use the *donor's* pattern/mask and retain recipient ``y``.
        """
        self._validate_x(x)
        self._validate_x(donor, 'donor CSI')
        if donor.shape != x.shape or donor.device != x.device or donor.dtype != x.dtype:
            raise ValueError('joint-training donors must match recipients')
        poison = self._selection(poison_mask, x, 'poison_mask')
        cross = self._selection(cross_mask, x, 'cross_mask')
        if bool((poison & cross).any()):
            raise ValueError('Poison and clean-labelled cross recipients must be disjoint')
        doses = self._doses(x, doses)
        if (y.shape != attack_targets.shape or y.ndim < 3 or y.shape[0] != x.shape[0]
                or y.shape[-1] != 3 or y.device != x.device or attack_targets.device != x.device
                or not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(attack_targets).all())):
            raise ValueError('Clean/attack pose targets must match finite (B,...,J,3) on CSI device')
        if not bool(self._mask_frozen):
            raise RuntimeError('Complete mask warm-up and freeze_mask before joint training')
        self.train()
        victim.train()
        opt_victim.zero_grad(set_to_none=True)
        opt_generator.zero_grad(set_to_none=True)
        mixed = x.clone()
        targets = y.clone()
        loss_div = x.new_zeros(())
        if bool(poison.any()):
            own_pattern, own_mask = self._parts(x[poison])
            mixed[poison] = (x[poison] + doses[poison, None, None, None] * own_mask
                             * (own_pattern - x[poison]))
            targets[poison] = attack_targets[poison]
            donor_pattern = self.pattern_generator(donor[poison])
            loss_div = diversity_ratio(x[poison], donor[poison], own_pattern, donor_pattern)
        if bool(cross.any()):
            pattern, mask = self._parts(donor[cross])
            # Cross recipients are clean-labelled controls, not poisons, so the
            # runner's poison-dose vector is zero there. Source cross injection
            # is full strength and must not disappear because of those zeros.
            mixed[cross] = x[cross] + mask * (pattern - x[cross])
        result = victim(mixed)
        prediction = result[0] if isinstance(result, (tuple, list)) else result
        if prediction.shape != targets.shape:
            raise ValueError(f'Victim pose shape {tuple(prediction.shape)} != {tuple(targets.shape)}')
        pose_loss = torch.linalg.vector_norm(prediction - targets, dim=-1).mean()
        total = pose_loss + loss_div
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('Nonfinite joint loss; no optimizer steps')
        total.backward()
        _check_gradients(victim.parameters(), 'victim')
        _check_gradients(self.generator_parameters(), 'pattern generator')
        opt_victim.step()
        opt_generator.step()
        return {'total_loss': float(total.detach()), 'pose_loss': float(pose_loss.detach()),
                'diversity_loss': float(loss_div.detach()), 'n_poison': int(poison.sum()),
                'n_cross': int(cross.sum()), 'n_samples': int(x.shape[0])}
