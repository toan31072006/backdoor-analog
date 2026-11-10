"""Scoped PiW BackTime adaptation; the historical MM-Fi implementation is read-only.

The original streaming schedule is rebound into a PRIVATE function namespace.
Only input/feature/pose adapters and frozen-state versioning differ. Nothing in
the imported MM-Fi module is monkeypatched. Full PiW amplitude+phase goes to the
surrogate; graph features and replacements use amplitude only (270 variables).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from types import FunctionType

import numpy as np
import torch

from . import backtime_mmfi as core

IMPLEMENTATION = 'independent-backtime-piw14-amp270-stream-v1'
SHAPE = (3, 180, 20)


def configuration(**overrides):
    cfg = core.BackTimeConfig(shape=(3, 90, 20), history_len=12, trigger_len=8,
                              target_joints=(11, 13), **overrides)
    return cfg


def validate(csi, config):
    if config.shape != (3, 90, 20) or config.target_joints != (11, 13):
        raise ValueError('PiW requires amplitude (3,90,20) and target joints 11/13')
    if (not isinstance(csi, torch.Tensor) or csi.ndim not in (3, 4)
            or tuple(csi.shape[-3:]) != SHAPE or not csi.is_floating_point()
            or not torch.isfinite(csi).all() or torch.any(csi < 0) or torch.any(csi > 1)):
        raise ValueError('Expected finite normalized PiW CHW/BCHW (3,180,20)')


def apply_tensor(generator, csi, doses):
    validate(csi, generator.config)
    amplitude, perturbation = core._apply(generator, csi[..., :90, :].contiguous(), doses)
    return torch.cat((amplitude, csi[..., 90:, :]), dim=-2), perturbation


def stream_row(dataset, index, config):
    row = dataset[int(index)]
    x = torch.as_tensor(row['csi'], dtype=torch.float32).detach().cpu().clone()
    y = torch.as_tensor(row['pose'], dtype=torch.float32).detach().cpu().clone()
    validate(x, config)
    if y.shape not in ((14, 3), (1, 14, 3)) or not torch.isfinite(y).all():
        raise ValueError('PiW requires finite 14-joint poses, not padded MM-Fi joints')
    return x, y


def stream_features(dataset, config, progress=None):
    total, square, count, features = 0., 0., 0, None
    for index in range(len(dataset)):
        x, _ = stream_row(dataset, index, config)
        flat = x[:, :90, :].numpy().reshape(config.n_variables, 20).astype(np.float64)
        value = core.author_frame_fft_features(flat, config.fft_components)
        features = value if features is None else features + value
        total += float(flat.sum())
        square += float(np.square(flat).sum())
        count += flat.size
        if progress and (index == 0 or (index + 1) % 10000 == 0 or index + 1 == len(dataset)):
            progress(dict(phase='full_TRAIN_amplitude_graph', frames=index + 1,
                          total_frames=len(dataset), phase_used_for_graph=False))
    mean = total / count
    std = math.sqrt(max(0., square / count - mean * mean))
    if not math.isfinite(std) or std <= 1e-12:
        raise ValueError('TRAIN amplitude standard deviation must be positive')
    return torch.from_numpy((features / len(dataset)).astype(np.float32)), mean, std


def target_tensor(y, doses):
    from piw_v2_adapter import target_pose
    d = core._doses(doses, torch.zeros((len(y), *SHAPE), device=y.device))
    return torch.as_tensor(np.stack([target_pose(p, float(v)) for p, v in
        zip(y.detach().cpu().numpy(), d.detach().cpu().numpy())]), dtype=y.dtype, device=y.device)


class FrozenPiWBackTimeTrigger(core.FrozenBackTimeTrigger):
    baseline_name = 'BackTime-PiW14 amplitude adaptation'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.shape != (3, 90, 20) or self.config.target_joints != (11, 13):
            raise ValueError('Foreign BackTime frozen configuration')
        self.shape = SHAPE
        if self._diagnostics.get('fit_profile'):
            self._diagnostics['fit_profile'] = 'full_TRAIN_streaming_author_schedule_PiW14_adaptation'

    def apply_tensor(self, csi, dose):
        return apply_tensor(self.generator, csi, dose)[0]

    def inject(self, csi, dose, eps=None):
        if eps is not None and float(eps) != self.config.eps:
            raise ValueError('BackTime epsilon is frozen, not WiFiDoor epsilon')
        value = np.asarray(csi)
        if value.shape != SHAPE or value.dtype.kind != 'f':
            raise ValueError('Expected floating PiW CHW')
        generator = self.generator if next(self.generator.parameters()).device.type == 'cpu' else deepcopy(self.generator).cpu()
        with torch.no_grad():
            return apply_tensor(generator, torch.from_numpy(value.copy()), dose)[0].numpy()

    def provenance(self):
        record = super().provenance()
        record['implementation'] = IMPLEMENTATION
        record['adaptation_differences'][0] = '270 amplitude variables; 20 packets split 12 history/8 replacement; all 90 phase rows unchanged'
        record['input_shape'] = list(SHAPE)
        return record

    def state_dict(self):
        state = super().state_dict()
        state.pop('sha256')
        state['version'] = IMPLEMENTATION
        return {**state, 'sha256': core._sha(state)}

    @classmethod
    def from_state_dict(cls, saved):
        state = core._cpu(saved)
        claimed = state.pop('sha256', None)
        if claimed != core._sha(state) or state.get('version') != IMPLEMENTATION:
            raise ValueError('PiW BackTime frozen checksum/version mismatch')
        cfg = core.BackTimeConfig(**state['config'])
        graph = state['generator']
        with torch.random.fork_rng(devices=[]):
            generator = core._GraphGenerator(cfg, graph['features'], graph['variables'],
                float(graph['train_mean']), float(graph['train_std']))
        generator.load_state_dict(graph, strict=True)
        return cls(cfg, generator, state['diagnostics'], selection_scores=state['selection_scores'])


def fit_stream(model_init, train_dataset, config, **kwargs):
    """Same native alternating loop, private adapters, no global mutation."""
    if max(config.target_joints) >= 14:
        raise ValueError('PiW target joint out of range')
    private = dict(vars(core))
    private.update(IMPLEMENTATION=IMPLEMENTATION, _stream_row=stream_row,
                   _stream_features=stream_features, _apply=apply_tensor,
                   FrozenBackTimeTrigger=FrozenPiWBackTimeTrigger)
    for name in ('_stream_batch', 'fit_backtime_stream'):
        original = getattr(core, name)
        function = FunctionType(original.__code__, private, original.__name__, original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        private[name] = function
    return private['fit_backtime_stream'](model_init, train_dataset, target_tensor, config, **kwargs)
