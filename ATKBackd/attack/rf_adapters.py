"""Independent RF-backdoor mechanisms adapted to amplitude CSI and HPELi.

Primary sources (audited 2026-10-08; no unlicensed source copied):
INFOCOM: https://arxiv.org/html/2505.00881v1, equations 4--6;
https://github.com/Tianyaz97/rf_backdoor at
989f148ed35c96f46bf96db797ff23376895fb25, Backdoor/wifi_attack.py.
BackdoorRF: https://github.com/NatsumiAi/BackdoorRF at
4b7d44fc4c939b6b2018e522f0f0c9801fa90abb, main.py and
util/{learnable_trigger,residual_prior}.py. Its README states CCAI2026;
formal paper metadata is not supplied by the repository.

Both adapters use real antenna amplitudes, broadcast a per-antenna temporal
segment over subcarriers, and clip to [0,1]. INFOCOM uses eight fixed signed
Gaussian patterns. Its cosine PORs are affinely mapped into the nonnegative
spatial feature space consumed by HPELi's head. BackdoorRF uses a learned,
smoothed, zero-mean segment with fixed RMS and training-derived log-PSD prior.
These are declared adaptations, not exact RF-fingerprinting reproductions.
Dose scales trigger amplitude; evaluation always uses a predeclared position.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def spatial_features(model, csi):
    """The actual pre-regression HPELi map, without changing its architecture."""
    if not all(hasattr(model, key) for key in ('skunit1', 'skunit2', '_pool', 'regression')):
        raise ValueError('RF adaptations require the HPELi encoder/regression interface')
    return model._pool(model.skunit2(model._pool(model.skunit1(csi))))


def _shape(n_ant, n_sub, n_pkt, segment_length):
    values = (n_ant, n_sub, n_pkt, segment_length)
    if any(isinstance(x, bool) or int(x) != x or int(x) < 1 for x in values):
        raise ValueError('CSI dimensions and segment length must be positive integers')
    shape = tuple(int(x) for x in values[:3])
    if not 2 <= segment_length <= shape[-1]:
        raise ValueError('RF temporal segment must have 2..n_pkt frames')
    return shape


class _RFTrigger(nn.Module):
    requires_deferred_injection = False
    requires_event_metadata = True

    def __init__(self, n_ant=3, n_sub=114, n_pkt=10, segment_length=3,
                 amplitude=0.185, seed=42, eval_start=0):
        super().__init__()
        self.shape = _shape(n_ant, n_sub, n_pkt, segment_length)
        self.segment_length = int(segment_length)
        self.amplitude = float(amplitude)
        self.seed = int(seed)
        self.eval_start = int(eval_start)
        if not np.isfinite(self.amplitude) or self.amplitude <= 0:
            raise ValueError('RF amplitude must be finite and positive')
        if not 0 <= self.eval_start <= self.shape[-1] - self.segment_length:
            raise ValueError('rf_eval_start must fit the temporal segment')

    def _validate_tensor(self, csi):
        if csi.ndim != 4 or tuple(csi.shape[1:]) != self.shape:
            raise ValueError(f'expected Bx{self.shape} amplitude CSI, got {tuple(csi.shape)}')
        if csi.is_complex() or not bool(torch.isfinite(csi).all()):
            raise ValueError('RF triggers require finite real amplitude CSI')

    def _inject_pattern(self, csi, pattern, dose, starts):
        self._validate_tensor(csi)
        doses = torch.as_tensor(dose, device=csi.device, dtype=csi.dtype)
        if doses.ndim == 0:
            doses = doses.expand(csi.shape[0])
        if doses.shape != (csi.shape[0],) or not bool(torch.isfinite(doses).all()) or bool((doses < 0).any()):
            raise ValueError('dose must be a finite nonnegative scalar or batch vector')
        temporal = torch.arange(csi.shape[-1], device=csi.device)[None, :] - starts[:, None]
        active = (temporal >= 0) & (temporal < self.segment_length)
        sample_pattern = pattern.gather(
            -1, temporal.clamp(0, self.segment_length - 1)[:, None, :].expand(-1, self.shape[0], -1))
        delta = sample_pattern[:, :, None, :] * active[:, None, None, :] * doses[:, None, None, None]
        attacked = (csi + delta).clamp(0.0, 1.0)
        # Preserve zero-dose input bit-for-bit, including signed-zero values.
        return torch.where((doses == 0)[:, None, None, None], csi, attacked)

    def inject(self, csi, dose, eps=0.185):
        """Detached CPU injection for common evaluation and worker-safe copies.

        eps is accepted for the shared interface; the recorded rf_trigger_amp
        defines this mechanism's RMS/standard-deviation amplitude budget.
        """
        array = np.asarray(csi)
        if array.shape != self.shape or np.iscomplexobj(array) or not np.isfinite(array).all():
            raise ValueError(f'expected finite real amplitude CSI {self.shape}')
        dose = float(dose)
        if not np.isfinite(dose) or dose < 0:
            raise ValueError('dose must be finite and nonnegative')
        if dose == 0:
            return array.astype(np.float32, copy=True)
        pattern = self.evaluation_pattern().detach().cpu().numpy()
        result = array.astype(np.float32, copy=True)
        start = self.eval_start
        result[:, :, start:start + self.segment_length] += dose * pattern[:, None, :]
        return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


class INFOCOMPORTrigger(_RFTrigger):
    """Fixed Gaussian/sign pairs; representation tampering lives in the trainer."""

    baseline_name = 'INFOCOM2025-POR-CSI-adapted'

    def __init__(self, *, eval_trigger_index=0, **kwargs):
        super().__init__(**kwargs)
        self.eval_trigger_index = int(eval_trigger_index)
        if not 0 <= self.eval_trigger_index < 8:
            raise ValueError('INFOCOM evaluation trigger index must be in 0..7')
        rng = np.random.default_rng(self.seed)
        first = rng.normal(0.0, self.amplitude,
                           (4, self.shape[0], self.segment_length)).astype(np.float32)
        self.register_buffer('patterns', torch.from_numpy(np.concatenate((first, -first[::-1].copy()))))
        self.register_buffer('por_bank', torch.empty(8, 0))

    def evaluation_pattern(self):
        return self.patterns[self.eval_trigger_index]

    def inject_tensor(self, csi, dose=1.0, trigger_ids=None):
        if trigger_ids is None:
            trigger_ids = torch.full((len(csi),), self.eval_trigger_index, device=csi.device, dtype=torch.long)
        patterns = self.patterns.to(csi)[trigger_ids]
        starts = torch.full((len(csi),), self.eval_start, device=csi.device, dtype=torch.long)
        return self._inject_pattern(csi, patterns, dose, starts)

    def configure_pors(self, feature_dimension, activation_scale):
        """Fixed affine nonnegative adaptation of paper equation 5.

        Scale is RMS of benign-teacher TRAIN activations only. Zero and one
        endpoint vectors remain distinct; cosine vectors use (1+cos)/2 so
        every raw HPELi ReLU feature target is attainable in sign.
        """
        feature_dimension = int(feature_dimension)
        scale = float(activation_scale)
        if feature_dimension < 1 or not np.isfinite(scale) or scale <= 0:
            raise ValueError('POR dimensions and training activation scale must be positive')
        t = torch.arange(feature_dimension, device=self.patterns.device, dtype=torch.float32) / feature_dimension
        bank = torch.zeros(8, feature_dimension, device=t.device)
        bank[-1] = scale
        for index in range(1, 7):
            sign = 1.0 if index <= 3 else -1.0
            frequency = index if index <= 3 else 7 - index
            bank[index] = scale * (1.0 + index / 8.0) * (1.0 + sign * torch.cos(2.0 * torch.pi * frequency * t)) / 2.0
        self.por_bank = bank

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if 'por_bank' in state_dict:
            self.por_bank = torch.empty_like(state_dict['por_bank'], device=self.patterns.device)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class LearnedRFTrigger(_RFTrigger):
    """A learned finite-support RF segment, with no invented learned mask."""

    baseline_name = 'CCAI2026-BackdoorRF-CSI-adapted'

    def __init__(self, *, smooth_kernel=None, **kwargs):
        super().__init__(**kwargs)
        kernel = int(smooth_kernel) if smooth_kernel is not None else (3 if self.segment_length >= 3 else 1)
        if kernel < 1 or kernel % 2 == 0 or kernel > self.segment_length:
            raise ValueError('RF smoothing kernel must be odd and <= segment length')
        self.smooth_kernel = kernel
        generator = torch.Generator().manual_seed(self.seed)
        self.raw_pattern = nn.Parameter(torch.randn(self.shape[0], self.segment_length, generator=generator))
        self.register_buffer('psd_template', torch.zeros(self.shape[0], self.segment_length // 2 + 1))

    def effective_pattern(self):
        smooth = F.avg_pool1d(self.raw_pattern[None], self.smooth_kernel,
                             stride=1, padding=self.smooth_kernel // 2)[0]
        centered = smooth - smooth.mean(-1, keepdim=True)
        return self.amplitude * centered / centered.square().mean().clamp_min(1e-12).sqrt()

    def evaluation_pattern(self):
        return self.effective_pattern()

    def starts(self, csi, mode='fixed'):
        count, _, _, time = csi.shape
        positions = time - self.segment_length + 1
        if mode == 'fixed':
            return torch.full((count,), self.eval_start, device=csi.device, dtype=torch.long)
        if mode == 'random':
            return torch.randint(positions, (count,), device=csi.device)
        if mode not in ('high_energy', 'low_energy'):
            raise ValueError(f'unsupported RF placement {mode!r}')
        energy = csi.square().mean((1, 2)).unfold(-1, self.segment_length, 1).mean(-1)
        return energy.argmax(-1) if mode == 'high_energy' else energy.argmin(-1)

    def inject_tensor(self, csi, dose=1.0, mode='fixed'):
        self._validate_tensor(csi)
        pattern = self.effective_pattern().to(csi)[None].expand(len(csi), -1, -1)
        return self._inject_pattern(csi, pattern, dose, self.starts(csi, mode))

    @staticmethod
    def log_psd(pattern):
        power = torch.fft.rfft(pattern, dim=-1).abs().square() / pattern.shape[-1]
        return (power + 1e-6).log()

    def regularization_loss(self, energy=1e-3, smooth=1e-3, environment=0.1):
        pattern = self.effective_pattern()
        return (float(energy) * pattern.square().mean()
                + float(smooth) * pattern.diff(dim=-1).square().mean()
                + float(environment) * F.mse_loss(self.log_psd(pattern), self.psd_template))


def build_rf_trigger(name, cfg):
    common = dict(n_ant=cfg.get('n_ant', 3), n_sub=cfg.get('n_sub', 114),
                  n_pkt=cfg.get('n_pkt', 10), seed=cfg.get('seed', 42),
                  segment_length=cfg.get('rf_segment_length', 3),
                  amplitude=cfg.get('rf_trigger_amp', cfg.get('eps', 0.185)),
                  eval_start=cfg.get('rf_eval_start', 0))
    name = str(name).lower().replace('-', '_')
    if name in ('infocom2025_por', 'rf_infocom2025', 'infocom2025_por_adapted'):
        return INFOCOMPORTrigger(eval_trigger_index=cfg.get('rf_eval_trigger_index', 0), **common)
    if name in ('ccai2026_backdoorrf', 'rf_ccai2026', 'backdoorrf_adapted'):
        return LearnedRFTrigger(smooth_kernel=cfg.get('rf_smooth_kernel'), **common)
    raise ValueError(f'unknown RF mechanism {name!r}')
