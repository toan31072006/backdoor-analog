"""Independent image-trigger adaptations for normalized MMFi amplitude CSI.

These are ``BadNets-CSI-adapted``, ``Blended-CSI-adapted`` and
``WaNet-CSI-adapted``, not RF-native attacks or exact paper reproductions.
The antenna axis stays a channel axis; frequency and packet axes replace image
height and width. All patterns are fixed by the experiment seed, and injection
needs neither event locations nor per-sample metadata.

Primary sources checked when implementing these adapters:
* BadNets: Gu, Dolan-Gavitt and Garg (2017),
  https://arxiv.org/abs/1708.06733 -- fixed local pattern and poisoned targets.
* Blended: Chen et al. (2017), https://arxiv.org/abs/1712.05526, Sec. III-B1
  -- ``(1-alpha)*x + alpha*k`` and a fixed uniform random key pattern.
* WaNet: Nguyen and Tran (ICLR 2021), https://arxiv.org/abs/2102.10369;
  author code: https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
  (``train.py``) -- smooth coarse random grid plus identity, then resampling.

An original author attack-code release was not located for BadNets or Blended;
their implementations below are derived from the papers. No third-party
implementation is represented as original author code.

Task-specific changes must be reported with results:
* BadNets uses a seeded binary frequency/packet patch and opacity
  ``alpha=min(dose*eps, 1)``. At alpha=1 it replaces the patch, as in a
  fixed-pattern patch attack; opacity below one is our dose extension.
* Blended uses a fixed same-shaped pattern sampled in [0,1], replacing the
  paper's [0,255] image scale, with ``alpha=min(dose*eps, 1)``. This preserves
  the published blending equation. Train/test dose handling is caller policy.
* WaNet uses one shared grid over frequency/packets, with no antenna mixing.
  SciPy cubic spline upsampling replaces the author's PyTorch bicubic
  upsampling; bilinear sampling follows an align-corners coordinate mapping.
  The smooth grid is normalized to maximum absolute magnitude one and eps
  scales normalized coordinate displacement, rather than applying the
  author's square-image ``s/input_height`` scaling to a 114-by-10 CSI frame.
  This is a geometric strength, NOT an amplitude L-infinity budget. Its
  ``noise_inject`` adds per-sample uniform coordinate jitter around the fixed
  warp, preserving the author's clean-label cover mechanism. Both smooth warp
  and jitter use dose*eps in place of the square-image 1/input_height factor;
  strength=0.5 and noise_strength=1 preserve the author's relative coefficients.
  Selecting disjoint cover samples and retaining their labels is caller policy.

For the traditional fixed-trigger comparison, train poisoned samples at
dose=1 and fixed target-payload dose=1. Vary only trigger dose at evaluation;
that experiment policy belongs to the dataset/training configuration.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def _nonnegative_finite(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative, got {value}')
    return value


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1:
        raise ValueError(f'{name} must be a positive integer, got {value}')
    return int(value)


class _AmplitudeTrigger:
    requires_deferred_injection = False
    requires_event_metadata = False

    def __init__(self, n_ant=3, n_sub=114, n_pkt=10, seed=42):
        self.n_ant = _positive_integer(n_ant, 'n_ant')
        self.n_sub = _positive_integer(n_sub, 'n_sub')
        self.n_pkt = _positive_integer(n_pkt, 'n_pkt')
        self.shape = (self.n_ant, self.n_sub, self.n_pkt)
        self.seed = int(seed)

    def _input(self, csi, dose, eps):
        value = np.asarray(csi)
        if value.shape != self.shape:
            raise ValueError(f'Expected amplitude CSI shape {self.shape}, got {value.shape}')
        if np.iscomplexobj(value):
            raise ValueError('Traditional CSI adapters require real [0,1] amplitudes')
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError('Traditional CSI adapters require finite [0,1] amplitudes')
        dose = _nonnegative_finite(dose, 'dose')
        eps = _nonnegative_finite(eps, 'eps')
        scale = dose * eps
        if not np.isfinite(scale):
            raise ValueError('dose*eps must be finite')
        # Always return independent storage. At zero scale this conversion is
        # the complete operation: no interpolation or normalization roundoff.
        return value.astype(np.float32, copy=True), scale


class BadNetsTrigger(_AmplitudeTrigger):
    """Fixed binary patch on supported amplitude bins and contiguous packets."""

    baseline_name = 'BadNets-CSI-adapted'
    dose_semantics = 'patch opacity = min(dose * eps, 1)'

    def __init__(self, n_ant=3, n_sub=114, n_pkt=10, seed=42,
                 patch_subcarriers=8, patch_packets=3, patch_start=None,
                 antennas=None):
        super().__init__(n_ant=n_ant, n_sub=n_sub, n_pkt=n_pkt, seed=seed)
        self.patch_subcarriers = _positive_integer(patch_subcarriers, 'patch_subcarriers')
        self.patch_packets = _positive_integer(patch_packets, 'patch_packets')
        if self.patch_subcarriers > self.n_sub or self.patch_packets > self.n_pkt:
            raise ValueError('BadNets patch must fit inside the amplitude frame')
        if patch_start is None:
            patch_start = (self.n_sub - self.patch_subcarriers,
                           self.n_pkt - self.patch_packets)
        if len(patch_start) != 2:
            raise ValueError('patch_start must contain subcarrier and packet start indices')
        starts = tuple(int(value) for value in patch_start)
        if any(start != value for start, value in zip(starts, patch_start)):
            raise ValueError('patch_start must contain integer indices')
        if (starts[0] < 0 or starts[1] < 0
                or starts[0] + self.patch_subcarriers > self.n_sub
                or starts[1] + self.patch_packets > self.n_pkt):
            raise ValueError('BadNets patch location must fit inside the amplitude frame')
        self.patch_start = starts

        if antennas is None:
            antennas = tuple(range(self.n_ant))
        if not antennas:
            raise ValueError('BadNets needs at least one selected antenna')
        selected = tuple(int(value) for value in antennas)
        if any(index != value for index, value in zip(selected, antennas)):
            raise ValueError('BadNets antenna indices must be integers')
        if len(set(selected)) != len(selected) or any(
                index < 0 or index >= self.n_ant for index in selected):
            raise ValueError('BadNets antenna indices must be unique and inside the frame')
        self.antennas = selected
        self.mask = np.zeros(self.shape, dtype=bool)
        frequency, packet = self.patch_start
        self.mask[list(selected), frequency:frequency + self.patch_subcarriers,
                  packet:packet + self.patch_packets] = True
        self.pattern = np.zeros(self.shape, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        self.pattern[self.mask] = rng.integers(0, 2, size=int(self.mask.sum()))

    def inject(self, csi, dose, eps=0.3):
        out, scale = self._input(csi, dose, eps)
        if scale == 0:
            return out
        alpha = min(scale, 1.0)
        out[self.mask] += alpha * (self.pattern[self.mask] - out[self.mask])
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


class BlendedTrigger(_AmplitudeTrigger):
    """Blend a single seeded, fixed uniform random key into the whole frame."""

    baseline_name = 'Blended-CSI-adapted'
    dose_semantics = 'blend alpha = min(dose * eps, 1)'

    def __init__(self, n_ant=3, n_sub=114, n_pkt=10, seed=42):
        super().__init__(n_ant=n_ant, n_sub=n_sub, n_pkt=n_pkt, seed=seed)
        self.pattern = np.random.default_rng(self.seed).uniform(
            0.0, 1.0, size=self.shape).astype(np.float32)

    def inject(self, csi, dose, eps=0.3):
        out, scale = self._input(csi, dose, eps)
        if scale == 0:
            return out
        alpha = min(scale, 1.0)
        out = (1.0 - alpha) * out + alpha * self.pattern
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


class WaNetTrigger(_AmplitudeTrigger):
    """Smooth fixed frequency/packet warp, shared across all antenna channels.

    Let I be the align-corners grid in [-1,1] and N the smooth random grid
    normalized to max(abs(N))=1. The full geometric key is
    G=(I+strength*N)*grid_rescale; injection samples at
    clip(I+dose*eps*(G-I), -1, 1). For grid_rescale=1, displacement is bounded
    by dose*eps*strength in normalized coordinates. One normalized coordinate
    unit is (axis_length-1)/2 bins or packets. There is no amplitude epsilon
    guarantee, and arbitrary signal distortion need not be monotonic in dose.
    """

    baseline_name = 'WaNet-CSI-adapted'
    dose_semantics = 'normalized warp displacement multiplier = dose * eps'

    def __init__(self, n_ant=3, n_sub=114, n_pkt=10, seed=42,
                 grid_size=4, strength=0.5, grid_rescale=1.0,
                 noise_strength=1.0):
        super().__init__(n_ant=n_ant, n_sub=n_sub, n_pkt=n_pkt, seed=seed)
        self.grid_size = _positive_integer(grid_size, 'grid_size')
        if self.grid_size < 2 or min(self.n_sub, self.n_pkt) < 2:
            raise ValueError('WaNet grid size and both CSI axes must be at least two')
        self.strength = _nonnegative_finite(strength, 'strength')
        self.grid_rescale = _nonnegative_finite(grid_rescale, 'grid_rescale')
        self.noise_strength = _nonnegative_finite(noise_strength, 'noise_strength')
        if self.grid_rescale == 0:
            raise ValueError('grid_rescale must be positive')

        frequency = np.linspace(-1.0, 1.0, self.n_sub)
        packet = np.linspace(-1.0, 1.0, self.n_pkt)
        packet_grid, frequency_grid = np.meshgrid(packet, frequency)
        # Match grid_sample's coordinate component order: x(packet), y(freq).
        self.identity_grid = np.stack((packet_grid, frequency_grid), axis=-1)
        rng = np.random.default_rng(self.seed)
        self.coarse_grid = rng.uniform(-1.0, 1.0, (2, self.grid_size, self.grid_size))
        self.coarse_grid /= max(float(np.mean(np.abs(self.coarse_grid))), 1e-12)
        query = np.array(np.meshgrid(
            np.linspace(0.0, self.grid_size - 1, self.n_sub),
            np.linspace(0.0, self.grid_size - 1, self.n_pkt), indexing='ij'))
        self.noise_grid = np.stack([
            map_coordinates(component, query, order=3, mode='nearest', prefilter=True)
            for component in self.coarse_grid
        ], axis=-1)
        self.noise_grid /= max(float(np.max(np.abs(self.noise_grid))), 1e-12)

    def sampling_grid(self, dose, eps=0.3):
        """Expose the geometric key for honest displacement measurements."""
        scale = _nonnegative_finite(dose, 'dose') * _nonnegative_finite(eps, 'eps')
        if not np.isfinite(scale):
            raise ValueError('dose*eps must be finite')
        full_grid = (self.identity_grid + self.strength * self.noise_grid) * self.grid_rescale
        return np.clip(self.identity_grid + scale * (full_grid - self.identity_grid), -1.0, 1.0)

    def inject(self, csi, dose, eps=0.3):
        out, scale = self._input(csi, dose, eps)
        if scale == 0 or (self.strength == 0 and self.grid_rescale == 1):
            return out
        grid = self.sampling_grid(dose=dose, eps=eps)
        return self._resample(out, grid)

    def noise_sampling_grid(self, seed, dose=1.0, eps=0.3):
        """Fixed warp plus sample-seeded jitter for clean-label cover samples.

        The official code clamps the backdoor grid, adds dense per-sample
        uniform coordinate noise divided by input_height, and clamps again.
        We preserve both clamps and use dose*eps*noise_strength for the noise
        coefficient on this rectangular CSI grid. The same grid samples every
        antenna, and no mutable RNG state depends on dataset access order.
        """
        grid = self.sampling_grid(dose=dose, eps=eps)
        scale = _nonnegative_finite(dose, 'dose') * _nonnegative_finite(eps, 'eps')
        jitter = np.random.default_rng(int(seed)).uniform(-1.0, 1.0, size=grid.shape)
        return np.clip(grid + scale * self.noise_strength * jitter, -1.0, 1.0)

    def noise_inject(self, csi, seed, eps=0.3, dose=1.0):
        """Warp one cover frame; its clean target must be retained by the caller."""
        out, scale = self._input(csi, dose, eps)
        if scale == 0:
            return out
        grid = self.noise_sampling_grid(seed=seed, dose=dose, eps=eps)
        return self._resample(out, grid)

    def _resample(self, amplitudes, grid):
        coordinates = np.stack((
            (grid[..., 1] + 1.0) * (self.n_sub - 1) / 2.0,
            (grid[..., 0] + 1.0) * (self.n_pkt - 1) / 2.0,
        ))
        # A 2D sampler for each antenna prevents interpolation between antennas.
        out = np.stack([
            map_coordinates(channel, coordinates, order=1, mode='nearest', prefilter=False)
            for channel in amplitudes
        ])
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def build_traditional_trigger(trigger_name: str, cfg: dict):
    """Build one MMFi adapter; the caller sets poison/target-dose policy."""
    if cfg.get('experiment_name', 'mmfi') != 'mmfi':
        raise ValueError('Traditional CSI adapters currently support MMFi amplitude CSI only')
    common = dict(n_ant=cfg.get('n_ant', 3), n_sub=cfg.get('n_sub', 114),
                  n_pkt=cfg.get('n_pkt', 10), seed=cfg.get('seed', 42))
    name = trigger_name.lower().replace('-', '_')
    if name in ('badnets', 'badnet', 'bad_nets', 'badnets_adapted'):
        return BadNetsTrigger(
            **common, patch_subcarriers=cfg.get('badnets_patch_subcarriers', 8),
            patch_packets=cfg.get('badnets_patch_packets', 3),
            patch_start=cfg.get('badnets_patch_start'),
            antennas=cfg.get('badnets_antennas'))
    if name in ('blended', 'blend', 'blended_adapted'):
        return BlendedTrigger(**common)
    if name in ('wanet', 'wa_net', 'wanet_adapted'):
        return WaNetTrigger(
            **common, grid_size=cfg.get('wanet_grid_size', 4),
            strength=cfg.get('wanet_strength', 0.5),
            grid_rescale=cfg.get('wanet_grid_rescale', 1.0),
            noise_strength=cfg.get('wanet_noise_strength', 1.0))
    raise ValueError(f'Unknown traditional trigger: {trigger_name}')
