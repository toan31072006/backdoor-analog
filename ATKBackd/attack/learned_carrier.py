"""Exploratory structured carriers with strict *realized* digital budgets.

These are CSI/HPE research hypotheses, not reproductions of SIBA, BLTO, LIRA,
or Sleeper Agent. Training this module does not register or modify a victim.
The attacker must freeze a JSON artifact before independent victim training.
Both ceilings concern stored, postclip CSI, not over-the-air stealthiness.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from attack.method_drafts import _CarrierDraftTrigger, _integer, _scalar

SCHEMA = 1
STATUS = 'DRAFT_ONLY_NOT_PAPER_RESULTS'
LEGACY_VARIANTS = ('weights', 'sparse', 'combined', 'gradient', 'energy')
BANK_VARIANTS = ('trainaware', 'bank', 'bank_guard')
PAIRED_VARIANTS = ('paired_guard',)
BANK_OPERATOR_VARIANTS = (*BANK_VARIANTS, *PAIRED_VARIANTS)
VARIANTS = (*LEGACY_VARIANTS, *BANK_OPERATOR_VARIANTS)
POLICY = 'original_postclip_linf_relative_l2_v1'
_OPERATOR_KEYS = ('lc_relative_l2', 'lc_reference_eps', 'lc_mask_fraction',
                  'lc_carrier_sub_mode', 'lc_carrier_time_mode', 'lc_carrier_seed')
_BANK_KEYS = ('lc_bank_size', 'lc_bank_seed')
_KNOWN = set(_OPERATOR_KEYS) | {
    'lc_schema', 'lc_variant', 'lc_recipe_sha256', 'lc_artifact_path',
    'lc_artifact_sha256', 'lc_warmup_epochs', 'lc_rounds', 'lc_inner_steps',
    'lc_outer_steps', 'lc_batch_size', 'lc_lr', 'lc_energy_weight',
    'lc_clean_weight', 'lc_fit_samples', 'lc_poison_indices', 'lc_selection',
    'lc_fit_seed', 'lc_fit_profile', 'lc_fitting_sha256', 'lc_poison_indices_sha256',
    'lc_inner_val_fraction', 'lc_gradient_tensors', 'lc_selection_sha256',
    'lc_bank_size', 'lc_bank_seed', 'lc_surrogate_count',
    'lc_surrogate_seed_stride', 'lc_lookahead_steps', 'lc_pa_weight',
    'lc_pck_weight', 'lc_pck_temperature', 'lc_utility_mpjpe_tolerance',
    'lc_utility_pa_tolerance', 'lc_utility_pck_tolerance',
}


def operator_config_keys(cfg):
    """Bind bank fields only for new variants; preserve legacy artifact schema."""
    return (*_OPERATOR_KEYS, *_BANK_KEYS) if cfg.get('lc_variant') in BANK_OPERATOR_VARIANTS else _OPERATOR_KEYS


def _sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError(f'{name} must be a lowercase SHA256 hex digest')
    return value


def resolve_learned_config(cfg):
    """Pure default resolution, shared by controls and learned candidates."""
    resolved = dict(cfg)
    unknown = sorted(k for k in resolved if k.startswith('lc_') and k not in _KNOWN)
    if unknown:
        raise ValueError(f'Unknown learned-carrier configuration keys: {unknown}')
    if resolved.get('experiment_name') != 'mmfi':
        raise ValueError('Learned carriers and dual-budget controls require MM-Fi')
    if resolved.get('trigger_zero_mean') is not True:
        raise ValueError('The Original reference requires trigger_zero_mean=true')
    schema = _integer(resolved.get('lc_schema', SCHEMA), 'lc_schema', minimum=1)
    if schema != SCHEMA:
        raise ValueError(f'lc_schema must equal {SCHEMA}')
    n_sub = _integer(resolved.get('n_sub', 114), 'n_sub', minimum=1)
    n_pkt = _integer(resolved.get('n_pkt', 10), 'n_pkt', minimum=1)
    _integer(resolved.get('n_ant', 3), 'n_ant', minimum=1)
    sub = _integer(resolved.get('lc_carrier_sub_mode', 3), 'lc_carrier_sub_mode')
    time = _integer(resolved.get('lc_carrier_time_mode', 1), 'lc_carrier_time_mode')
    seed = _integer(resolved.get('lc_carrier_seed', 42), 'lc_carrier_seed')
    if sub >= n_sub or time >= n_pkt or sub == time == 0:
        raise ValueError('Carrier modes must fit axes and not both be zero')
    relative = _scalar(resolved.get('lc_relative_l2', .10), 'lc_relative_l2', maximum=1)
    reference_eps = _scalar(resolved.get('lc_reference_eps', .185), 'lc_reference_eps')
    fraction = _scalar(resolved.get('lc_mask_fraction', .25), 'lc_mask_fraction', maximum=1)
    if relative <= 0 or reference_eps <= 0 or fraction <= 0:
        raise ValueError('Relative L2, reference epsilon and mask fraction must be positive')
    variant = resolved.get('lc_variant')
    if variant is not None and variant not in (*VARIANTS, 'selection', 'reference', 'control'):
        raise ValueError(f'Unsupported learned-carrier variant: {variant}')
    if variant in BANK_OPERATOR_VARIANTS:
        required_size = 2 if variant in ('trainaware', 'paired_guard') else 8
        size = _integer(resolved.get('lc_bank_size', required_size), 'lc_bank_size', minimum=2)
        if size != required_size:
            raise ValueError(f'lc_bank_size must equal {required_size} for {variant}')
        bank_seed = _integer(resolved.get('lc_bank_seed', 42), 'lc_bank_seed')
        resolved.update(lc_bank_size=size, lc_bank_seed=bank_seed)
    elif any(key in resolved for key in _BANK_KEYS):
        raise ValueError('Bank configuration requires a carrier-bank variant')
    for key in ('lc_recipe_sha256', 'lc_artifact_sha256'):
        if key in resolved:
            _sha256(resolved[key], key)
    artifact_keys = ('lc_artifact_path', 'lc_artifact_sha256')
    if any(k in resolved for k in artifact_keys) and not all(k in resolved for k in artifact_keys):
        raise ValueError('Artifact path and bytes SHA256 must be supplied together')
    if 'lc_artifact_path' in resolved and not isinstance(resolved['lc_artifact_path'], (str, Path)):
        raise ValueError('lc_artifact_path must be a filesystem path')
    resolved.update(lc_schema=schema, lc_relative_l2=relative,
                    lc_reference_eps=reference_eps, lc_mask_fraction=fraction,
                    lc_carrier_sub_mode=sub, lc_carrier_time_mode=time,
                    lc_carrier_seed=seed)
    return resolved


def _basis_sha256(bank):
    return hashlib.sha256(np.ascontiguousarray(bank, dtype='<f8').tobytes()).hexdigest()


def _smooth_carrier_bank(built, size, seed):
    """Keep Original/two-carrier bases, then add six smooth separable modes.

    Link profiles combine low antenna-axis DCT modes with decaying coefficients.
    Frequency/time ranks remain small, and double Gram-Schmidt gives independent
    keys without labels, victim outputs or data-dependent basis construction.
    """
    bases = [built.p0.copy(), built.p1.copy()]
    records = [dict(kind='original'), dict(kind='existing_carrier',
        sub_mode=built.sub_mode, time_mode=built.time_mode, seed=built.seed)]
    if size == 2:
        return np.stack(bases), records
    n_ant, n_sub, n_pkt = built.shape
    preferred = [(1, 0), (2, 0), (1, 1), (2, 1), (0, 2), (3, 2)]
    available = [(s, t) for t in range(min(n_pkt, 3))
                 for s in range(min(n_sub, 6)) if s or t]
    pairs = [pair for pair in preferred if pair in available]
    pairs.extend(pair for pair in available if pair not in pairs)
    rng = np.random.default_rng(seed)
    antenna_modes = np.arange(min(n_ant, 3))
    antenna_basis = np.cos(np.pi * antenna_modes[:, None]
                           * (np.arange(n_ant)[None, :] + .5) / n_ant)
    # Multiple antenna profiles per frequency/time pair support small fixtures.
    for repetition in range(max(1, n_ant)):
        for sub_mode, time_mode in pairs:
            coefficients = rng.normal(size=len(antenna_modes)) / (1 + antenna_modes) ** 2
            antenna = coefficients @ antenna_basis
            frequency = np.cos(np.pi * sub_mode * (np.arange(n_sub) + .5) / n_sub)
            packet = np.cos(np.pi * time_mode * (np.arange(n_pkt) + .5) / n_pkt)
            carrier = antenna[:, None, None] * frequency[None, :, None] * packet[None, None, :]
            for _ in range(2):
                carrier -= carrier.mean()
                for previous in bases:
                    carrier -= np.mean(carrier * previous) / np.mean(previous ** 2) * previous
            rms = float(np.sqrt(np.mean(carrier ** 2)))
            if not np.isfinite(rms) or rms <= 1e-12:
                continue
            bases.append(carrier / rms)
            records.append(dict(kind='smooth_dct', sub_mode=sub_mode, time_mode=time_mode,
                antenna_coefficients=coefficients.tolist(), repetition=repetition))
            if len(bases) == size:
                return np.stack(bases), records
    raise ValueError('CSI axes cannot support the declared independent carrier bank')


def _bank_diagnostics(bank, records, seed):
    flat = bank.reshape(len(bank), -1)
    gram = flat @ flat.T / flat.shape[1]
    return dict(schema=1, size=len(bank), seed=seed, shape=list(bank.shape[1:]),
                basis_sha256=_basis_sha256(bank), modes=records,
                max_abs_gram_error=float(np.max(np.abs(gram - np.eye(len(bank))))))


def _numpy_input(csi, shape, dose):
    value = np.asarray(csi)
    if (value.shape != shape or value.dtype.kind not in 'fiu'
            or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1)):
        raise ValueError(f'Expected finite real [0,1] CSI of shape {shape}')
    return value.astype(np.float32, copy=True), _scalar(dose, 'dose', maximum=1)


def _strict_project_numpy(value, candidate, peak_cap, l2_cap):
    """Shrink only; guarantee both caps after float32 storage, including ULPs."""
    x = np.asarray(value, dtype=np.float32)
    native = np.asarray(candidate)
    if (native.shape != x.shape or native.dtype.kind not in 'fiu'
            or not np.isfinite(native).all() or np.any(native < 0) or np.any(native > 1)):
        raise ValueError('Invalid dual-budget candidate')
    peak_cap = _scalar(peak_cap, 'peak_cap')
    l2_cap = _scalar(l2_cap, 'l2_cap')
    native = native.astype(np.float32)
    original_delta = native.astype(np.float64) - x.astype(np.float64)
    peak = float(np.max(np.abs(original_delta)))
    norm = float(np.linalg.norm(original_delta.ravel()))
    factor = min(1., peak_cap / peak if peak else 1., l2_cap / norm if norm else 1.)
    if factor == 0:
        return x.copy(), factor
    if factor == 1:
        return native.copy(), factor
    out = np.clip(x.astype(np.float64) + factor * original_delta, 0, 1).astype(np.float32)
    # Coordinate rounding inward enforces Linf. If L2 rounding is outward,
    # uniformly reshrink the actual residual, then move one ULP inward.
    for _ in range(8):
        delta = out.astype(np.float64) - x.astype(np.float64)
        outside = np.abs(delta) > peak_cap
        if np.any(outside):
            out[outside] = np.nextafter(out[outside], x[outside])
        delta = out.astype(np.float64) - x.astype(np.float64)
        actual_norm = float(np.linalg.norm(delta.ravel()))
        if actual_norm <= l2_cap and np.max(np.abs(delta)) <= peak_cap:
            return out, factor
        ratio = min(1., l2_cap / actual_norm if actual_norm else 1.)
        out = np.clip(x.astype(np.float64) + ratio * delta, 0, 1).astype(np.float32)
        out = np.nextafter(out, x)
    # A vanishing budget may be smaller than the smallest representable step.
    # Identity is always admissible; never return an unchecked approximation.
    return x.copy(), 0.


def _strict_project_tensor(x, candidate, peak_cap, l2_cap):
    """Differentiable shrink with a straight-through float32 storage guard."""
    delta = candidate.to(torch.float64) - x.to(torch.float64)
    dims = tuple(range(1, x.ndim))
    peak = delta.abs().amax(dims)
    norm = torch.linalg.vector_norm(delta.flatten(1), dim=1)
    one = torch.ones_like(peak)
    peak_ratio = torch.where(peak > 0, peak_cap / peak.clamp_min(1e-300), one)
    l2_ratio = torch.where(norm > 0, l2_cap / norm.clamp_min(1e-300), one)
    factor = torch.minimum(one, torch.minimum(peak_ratio, l2_ratio))
    view = (x.shape[0],) + (1,) * (x.ndim - 1)
    projected = (x.to(torch.float64) + factor.reshape(view) * delta).clamp(0, 1).to(torch.float32)
    with torch.no_grad():
        guarded = projected.detach().clone()
        original = x.detach().to(torch.float32)
        pc, nc = peak_cap.detach(), l2_cap.detach()
        for _ in range(8):
            actual = guarded.to(torch.float64) - original.to(torch.float64)
            outside = actual.abs() > pc.reshape(view)
            inward = torch.nextafter(guarded, original)
            guarded = torch.where(outside, inward, guarded)
            actual = guarded.to(torch.float64) - original.to(torch.float64)
            norms = torch.linalg.vector_norm(actual.flatten(1), dim=1)
            peaks = actual.abs().amax(dims)
            bad = (norms > nc) | (peaks > pc)
            if not bool(bad.any()):
                break
            ratio = torch.minimum(torch.ones_like(norms), nc / norms.clamp_min(1e-300))
            corrected = (original.to(torch.float64) + ratio.reshape(view) * actual).clamp(0, 1).to(torch.float32)
            corrected = torch.nextafter(corrected, original)
            guarded = torch.where(bad.reshape(view), corrected, guarded)
        actual = guarded.to(torch.float64) - original.to(torch.float64)
        bad = (torch.linalg.vector_norm(actual.flatten(1), dim=1) > nc) | (actual.abs().amax(dims) > pc)
        guarded = torch.where(bad.reshape(view), original, guarded)
    return guarded + (projected - projected.detach())


class TrainableCarrier(nn.Module):
    """Signed carrier-bank weights and/or a legacy group-sparse STE mask."""

    def __init__(self, base, cfg, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown carrier variant: {variant}')
        if cfg.get('lc_variant', variant) != variant:
            raise ValueError('Trainable carrier variant disagrees with configuration')
        self.cfg = resolve_learned_config(dict(cfg, lc_variant=variant) if variant in BANK_OPERATOR_VARIANTS else cfg)
        self.variant, self.base = variant, base
        built = _CarrierDraftTrigger(base, self.cfg['lc_carrier_sub_mode'],
                                     self.cfg['lc_carrier_time_mode'], self.cfg['lc_carrier_seed'])
        self.shape = built.shape
        self.register_buffer('p0', torch.from_numpy(built.p0.copy()))
        self.register_buffer('p1', torch.from_numpy(built.p1.copy()))
        if variant in BANK_OPERATOR_VARIANTS:
            bank, records = _smooth_carrier_bank(built, self.cfg['lc_bank_size'], self.cfg['lc_bank_seed'])
            self.bank_diagnostics = _bank_diagnostics(bank, records, self.cfg['lc_bank_seed'])
            self.register_buffer('carrier_bank', torch.from_numpy(bank))
            initial_weights = torch.zeros(len(bank), dtype=torch.float64)
            initial_weights[:2] = 1.
        else:
            initial_weights = torch.ones(2, dtype=torch.float64)
        self.weights = nn.Parameter(initial_weights, requires_grad=variant != 'sparse')
        rng = np.random.default_rng(self.cfg['lc_carrier_seed'])
        self.mask_scores = nn.Parameter(torch.from_numpy(rng.normal(0, .01, size=self.shape[:2])),
                                        requires_grad=variant in ('sparse', 'combined'))
        self.amplitude_initial = .5 if variant == 'energy' else 1.
        if variant == 'energy':
            self.amplitude_logit = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        else:
            self.register_buffer('amplitude_fixed', torch.tensor(1., dtype=torch.float64))
        self.mask_groups = max(1, int(np.ceil(np.prod(self.shape[:2]) * self.cfg['lc_mask_fraction'])))
        self.eps = _scalar(self.cfg.get('eps', .185), 'eps')

    def pattern_tensor(self):
        if self.variant in BANK_OPERATOR_VARIANTS:
            raw = (self.weights.reshape(-1, 1, 1, 1) * self.carrier_bank).sum(dim=0)
        else:
            raw = self.weights[0] * self.p0 + self.weights[1] * self.p1
        if self.variant in ('sparse', 'combined'):
            # Stable CPU/device sort gives deterministic tie handling.
            flat = self.mask_scores.flatten()
            indices = torch.argsort(flat, descending=True, stable=True)[:self.mask_groups]
            hard = torch.zeros_like(flat).scatter(0, indices, 1).reshape(self.shape[:2])
            soft = torch.sigmoid(self.mask_scores)
            mask = (hard + soft - soft.detach()).unsqueeze(-1).expand(self.shape)
            mean = (raw * mask).sum() / mask.sum().clamp_min(1)
            pattern = (raw - mean) * mask
        else:
            pattern = raw - raw.mean()
        rms = pattern.square().mean().sqrt()
        if not bool(torch.isfinite(rms)) or bool(rms <= 1e-12):
            raise ValueError('Learned carrier has a degenerate/nonfinite pattern')
        return pattern / rms

    def export_pattern(self):
        return self.pattern_tensor().detach().cpu().numpy().astype(np.float64, copy=True)

    def amplitude_tensor(self):
        """Energy-only learnable attenuation; it cannot amplify the carrier."""
        return self.amplitude_logit.sigmoid() if self.variant == 'energy' else self.amplitude_fixed

    def export_amplitude(self):
        return float(self.amplitude_tensor().detach().cpu())

    def inject_tensor(self, x, dose, eps=None):
        if not isinstance(x, torch.Tensor) or x.is_complex() or x.dtype != torch.float32:
            raise ValueError('inject_tensor requires a real float32 CSI tensor')
        single = x.ndim == 3
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 4 or tuple(x.shape[1:]) != self.shape or x.shape[0] == 0:
            raise ValueError(f'Expected batch CSI of shape (B,{self.shape})')
        if not bool(torch.isfinite(x).all()) or bool(((x < 0) | (x > 1)).any()):
            raise ValueError('CSI must be finite in [0,1]')
        if isinstance(dose, torch.Tensor):
            if dose.is_complex() or dose.dtype == torch.bool:
                raise ValueError('Dose must be real and not boolean')
            doses = dose.to(device=x.device, dtype=torch.float64)
            if doses.ndim == 0:
                doses = doses.expand(x.shape[0])
            if doses.shape != (x.shape[0],):
                raise ValueError('Dose tensor must be scalar or one value per sample')
            if not bool(torch.isfinite(doses).all()) or bool(((doses < 0) | (doses > 1)).any()):
                raise ValueError('Dose must be finite in [0,1]')
        else:
            doses = x.new_full((x.shape[0],), _scalar(dose, 'dose', maximum=1), dtype=torch.float64)
        eps = self.eps if eps is None else _scalar(eps, 'eps')
        pattern = self.pattern_tensor().to(x.device)
        scale = (doses * eps * self.amplitude_tensor().to(x.device)).reshape(-1, 1, 1, 1)
        gain = (1 + scale * pattern).clamp_min(0).to(torch.float32)
        if not bool(torch.isfinite(gain).all()):
            raise ValueError('Epsilon exceeds the finite gain range')
        candidate = (x * gain).clamp(0, 1)
        ref_gain = (1 + (doses * self.cfg['lc_reference_eps']).reshape(-1, 1, 1, 1)
                    * self.p0.to(x.device)).clamp_min(0).to(torch.float32)
        reference = (x * ref_gain).clamp(0, 1)
        peak_cap = (reference.to(torch.float64) - x.to(torch.float64)).abs().flatten(1).amax(1)
        l2_cap = self.cfg['lc_relative_l2'] * doses * torch.linalg.vector_norm(x.to(torch.float64).flatten(1), dim=1)
        out = _strict_project_tensor(x, candidate, peak_cap, l2_cap)
        return out[0] if single else out


def artifact_dict(module, recipe_sha256, provenance=None):
    """Create an inert artifact; provenance is JSON data, never executable."""
    _sha256(recipe_sha256, 'recipe_sha256')
    result = dict(schema=SCHEMA, status=STATUS, variant=module.variant,
                  shape=list(module.shape), pattern=module.export_pattern().tolist(),
                  amplitude=module.export_amplitude(), amplitude_initial=module.amplitude_initial,
                  recipe_sha256=recipe_sha256,
                  operator_config={k: module.cfg[k] for k in operator_config_keys(module.cfg)},
                  provenance={} if provenance is None else provenance)
    if module.variant in BANK_OPERATOR_VARIANTS:
        result.update(raw_weights=module.weights.detach().cpu().tolist(),
                      bank_diagnostics=module.bank_diagnostics)
    json.dumps(result, allow_nan=False)
    return result


def write_artifact(path, module, recipe_sha256, provenance=None):
    """Write a canonical JSON artifact and return SHA256 of its actual bytes."""
    data = json.dumps(artifact_dict(module, recipe_sha256, provenance), sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f'.{target.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return hashlib.sha256(data).hexdigest()


class FrozenLearnedCarrier:
    requires_deferred_injection = False
    requires_event_metadata = False

    def __init__(self, base, cfg):
        self.cfg = resolve_learned_config(cfg)
        self.base = base
        built = _CarrierDraftTrigger(base, self.cfg['lc_carrier_sub_mode'],
                                     self.cfg['lc_carrier_time_mode'], self.cfg['lc_carrier_seed'])
        self.shape = built.shape
        self.n_ant, self.n_sub, self.n_pkt = self.shape
        for key in ('lc_artifact_path', 'lc_artifact_sha256', 'lc_recipe_sha256'):
            if key not in self.cfg:
                raise ValueError(f'Frozen carrier requires {key}')
        data = Path(self.cfg['lc_artifact_path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != self.cfg['lc_artifact_sha256']:
            raise ValueError('Learned carrier artifact bytes SHA256 mismatch')
        obj = json.loads(data)
        if (obj.get('schema') != SCHEMA or obj.get('status') != STATUS
                or obj.get('recipe_sha256') != self.cfg['lc_recipe_sha256']
                or obj.get('variant') not in VARIANTS or obj.get('shape') != list(self.shape)):
            raise ValueError('Learned artifact schema, variant, shape or recipe mismatch')
        effective_cfg = resolve_learned_config(dict(self.cfg, lc_variant=obj['variant']))
        if obj.get('operator_config') != {k: effective_cfg[k] for k in operator_config_keys(effective_cfg)}:
            raise ValueError('Learned artifact operator configuration mismatch')
        if self.cfg.get('lc_variant', obj['variant']) != obj['variant']:
            raise ValueError('Learned artifact variant mismatch')
        amplitude = _scalar(obj.get('amplitude'), 'artifact amplitude', maximum=1)
        if obj['variant'] != 'energy' and amplitude != 1:
            raise ValueError('Only the energy variant may learn amplitude attenuation')
        self.amplitude_initial = .5 if obj['variant'] == 'energy' else 1.
        if obj.get('amplitude_initial') != self.amplitude_initial:
            raise ValueError('Learned artifact amplitude initialization mismatch')
        try:
            pattern = np.asarray(obj['pattern'], dtype=np.float64)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError('Invalid learned artifact pattern') from exc
        if (pattern.shape != self.shape or not np.isfinite(pattern).all()
                or abs(float(pattern.mean())) > 1e-7
                or not np.isclose(np.mean(pattern ** 2), 1, atol=1e-7, rtol=1e-7)):
            raise ValueError('Learned pattern must be finite, zero-mean and unit RMS')
        if obj['variant'] in ('sparse', 'combined'):
            groups = np.any(pattern != 0, axis=-1)
            maximum = max(1, int(np.ceil(np.prod(self.shape[:2]) * self.cfg['lc_mask_fraction'])))
            if int(groups.sum()) > maximum:
                raise ValueError('Learned pattern exceeds the declared sparse group fraction')
        if obj['variant'] in BANK_OPERATOR_VARIANTS:
            bank, records = _smooth_carrier_bank(built, effective_cfg['lc_bank_size'],
                                                effective_cfg['lc_bank_seed'])
            diagnostics = _bank_diagnostics(bank, records, effective_cfg['lc_bank_seed'])
            if obj.get('bank_diagnostics') != diagnostics:
                raise ValueError('Learned artifact bank diagnostics mismatch')
            try:
                native_weights = np.asarray(obj['raw_weights'])
                if native_weights.dtype.kind not in 'fiu':
                    raise ValueError('Raw bank weights must be real numeric values')
                weights = native_weights.astype(np.float64)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError('Invalid learned artifact raw bank weights') from exc
            if weights.shape != (len(bank),) or not np.isfinite(weights).all():
                raise ValueError('Learned artifact raw bank weights must be finite and match bank size')
            raw = np.sum(weights.reshape(-1, 1, 1, 1) * bank, axis=0)
            centered = raw - raw.mean()
            rms = float(np.sqrt(np.mean(centered ** 2)))
            if not np.isfinite(rms) or rms <= 1e-12:
                raise ValueError('Learned artifact raw bank weights produce a degenerate pattern')
            if not np.allclose(centered / rms, pattern, atol=1e-10, rtol=1e-10):
                raise ValueError('Learned artifact raw bank weights disagree with frozen pattern')
            self.raw_weights = weights.copy()
            self.bank_diagnostics = diagnostics
        self.cfg = effective_cfg
        self._pattern, self.variant, self.amplitude = pattern.copy(), obj['variant'], amplitude
        self.artifact_sha256 = self.cfg['lc_artifact_sha256']
        self.name = f'lc_{self.variant}'

    @property
    def pattern(self):
        return self._pattern.copy()

    @property
    def protocol_parameters(self):
        return dict(trigger=self.name, lc_schema=SCHEMA, status=STATUS,
                    variant=self.variant, amplitude=self.amplitude,
                    amplitude_initial=self.amplitude_initial, policy=POLICY,
                    artifact_sha256=self.artifact_sha256,
                    recipe_sha256=self.cfg['lc_recipe_sha256'],
                    **{k: self.cfg[k] for k in operator_config_keys(self.cfg)})

    def inject(self, csi, dose, eps=None):
        return self.inject_with_audit(csi, dose, eps)[0]

    def inject_with_audit(self, csi, dose, eps=None):
        value, dose = _numpy_input(csi, self.shape, dose)
        eps = _scalar(self.cfg.get('eps', .185) if eps is None else eps, 'eps')
        from attack.method_drafts import _MethodDraftTrigger
        candidate = _MethodDraftTrigger._gain_inject(value, self._pattern, dose * eps * self.amplitude)
        reference = self.base.inject(value, dose, eps=self.cfg['lc_reference_eps'])
        return _project_audit(value, candidate, reference, dose, self.cfg['lc_relative_l2'])

    def fixed_key_sha256(self):
        digest = hashlib.sha256(self.artifact_sha256.encode())
        digest.update(np.ascontiguousarray(self.base.m_zm, dtype='<f8').tobytes())
        return digest.hexdigest()


def _project_audit(value, candidate, reference, dose, relative_l2):
    x = value.astype(np.float64)
    reference = np.asarray(reference)
    if (reference.shape != value.shape or reference.dtype.kind not in 'fiu'
            or not np.isfinite(reference).all() or np.any(reference < 0) or np.any(reference > 1)):
        raise ValueError('Invalid dual-budget reference')
    peak_cap = float(np.max(np.abs(reference.astype(np.float32).astype(np.float64) - x)))
    l2_cap = float(relative_l2 * dose * np.linalg.norm(x.ravel()))
    out, factor = _strict_project_numpy(value, candidate, peak_cap, l2_cap)
    native_delta = np.asarray(candidate, dtype=np.float32).astype(np.float64) - x
    actual_delta = out.astype(np.float64) - x
    peak = float(np.max(np.abs(actual_delta)))
    norm = float(np.linalg.norm(actual_delta.ravel()))
    return out, dict(policy=POLICY, ceiling=peak_cap, l2_ceiling=l2_cap,
                     native_peak=float(np.max(np.abs(native_delta))),
                     native_l2=float(np.linalg.norm(native_delta.ravel())),
                     peak=peak, l2=norm, shrink_factor=factor, shrunk=factor < 1,
                     violation=peak > peak_cap, l2_violation=norm > l2_cap)


class DualBudgetTrigger:
    """Common cap wrapper for native controls, learned carriers and covers."""
    requires_deferred_injection = False
    requires_event_metadata = False

    def __init__(self, native, reference, reference_eps=.185, relative_l2=.10):
        self.native, self.reference = native, reference
        self.reference_eps = _scalar(reference_eps, 'reference_eps')
        self.relative_l2 = _scalar(relative_l2, 'relative_l2', maximum=1)
        if self.reference_eps <= 0 or self.relative_l2 <= 0:
            raise ValueError('Both budget strengths must be positive')
        if reference.zero_mean is not True or reference.m_zm is None:
            raise ValueError('Dual budget requires a built zero-mean Original reference')
        self.n_ant, self.n_sub, self.n_pkt = reference.n_ant, reference.n_sub, reference.n_pkt
        self.shape = (self.n_ant, self.n_sub, self.n_pkt)
        self.requires_stochastic_trigger_state = bool(getattr(native, 'requires_stochastic_trigger_state', False))

    def peak_ceiling(self, csi, dose):
        value, dose = _numpy_input(csi, self.shape, dose)
        reference = self.reference.inject(value, dose, eps=self.reference_eps)
        return float(np.max(np.abs(reference.astype(np.float64) - value.astype(np.float64))))

    def relative_l2_ceiling(self, csi, dose):
        value, dose = _numpy_input(csi, self.shape, dose)
        return self.relative_l2 * dose * float(np.linalg.norm(value.astype(np.float64).ravel()))

    def _project(self, value, candidate, dose):
        reference = self.reference.inject(value, dose, eps=self.reference_eps)
        return _project_audit(value, candidate, reference, dose, self.relative_l2)

    def inject(self, csi, dose, eps=.3):
        return self.inject_with_audit(csi, dose, eps)[0]

    def inject_with_audit(self, csi, dose, eps=.3):
        value, dose = _numpy_input(csi, self.shape, dose)
        eps = _scalar(eps, 'eps')
        if dose == 0:
            return self._project(value, value, dose)
        return self._project(value, self.native.inject(value, dose, eps=eps), dose)

    def noise_inject(self, csi, seed=None, eps=.3, dose=1.):
        return self.noise_inject_with_audit(csi, seed, eps, dose)[0]

    def noise_inject_with_audit(self, csi, seed=None, eps=.3, dose=1.):
        if not callable(getattr(self.native, 'noise_inject', None)):
            raise ValueError('Native trigger has no cover operator')
        value, dose = _numpy_input(csi, self.shape, dose)
        eps = _scalar(eps, 'eps')
        if dose == 0:
            return self._project(value, value, dose)
        # One draw only: a stochastic cover key must not be consumed twice.
        candidate = self.native.noise_inject(value, seed=seed, eps=eps, dose=dose)
        return self._project(value, candidate, dose)

    def fixed_key_sha256(self):
        digest = hashlib.sha256(json.dumps(dict(policy=POLICY, reference_eps=self.reference_eps,
                                                 relative_l2=self.relative_l2), sort_keys=True).encode())
        digest.update(type(self.native).__name__.encode())
        if callable(getattr(self.native, 'fixed_key_sha256', None)):
            digest.update(self.native.fixed_key_sha256().encode())
        else:
            for key in ('pattern', 'mask'):
                if hasattr(self.native, key):
                    value = np.asarray(getattr(self.native, key))
                    digest.update(key.encode())
                    digest.update(str((value.shape, value.dtype.str)).encode())
                    digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(np.ascontiguousarray(self.reference.m_zm, dtype='<f8').tobytes())
        return digest.hexdigest()

    def state_dict(self):
        if not self.requires_stochastic_trigger_state:
            raise ValueError('Deterministic dual-budget trigger has no mutable state')
        return dict(schema=SCHEMA, policy=POLICY, key_sha256=self.fixed_key_sha256(),
                    native=self.native.state_dict())

    def load_state_dict(self, state):
        if (state.get('schema') != SCHEMA or state.get('policy') != POLICY
                or state.get('key_sha256') != self.fixed_key_sha256()):
            raise ValueError('Dual-budget trigger checkpoint mismatch')
        self.native.load_state_dict(state['native'])
