"""Four fixed, one-factor MM-Fi carrier hypotheses.

These operators use no fitting data, labels, victim outputs, or training loss.
Their native gain/additive operators are deliberately separate from the common
postclip peak wrapper reused unchanged from the historical full comparison.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from attack.method_drafts import _CarrierDraftTrigger, _integer, _scalar


SCHEMA = 1
FULL_PROFILE = 'mmfi_simple_full_peak_v2'
VARIANTS = ('additive', 'single', 'shared_links', 'frequency_only')
_KNOWN = {
    'simple_schema', 'simple_variant', 'simple_sub_mode', 'simple_time_mode',
    'simple_seed', 'simple_full_profile',
    'simple_reference_eps', 'simple_reference_run',
}
_FIXED = dict(simple_sub_mode=3, simple_time_mode=1, simple_seed=42)


def resolve_simple_config(cfg):
    """Resolve the fixed hypothesis protocol before configuration hashing."""
    resolved = dict(cfg)
    unknown = sorted(k for k in resolved if k.startswith('simple_') and k not in _KNOWN)
    if unknown:
        raise ValueError(f'Unknown simple-carrier configuration keys: {unknown}')
    if resolved.get('experiment_name') != 'mmfi':
        raise ValueError('Simple carriers require experiment_name=mmfi')
    if resolved.get('trigger_zero_mean') is not True:
        raise ValueError('Simple carriers require trigger_zero_mean=true')
    schema = _integer(resolved.get('simple_schema', SCHEMA), 'simple_schema', minimum=1)
    if schema != SCHEMA:
        raise ValueError(f'simple_schema must equal {SCHEMA}')
    variant = resolved.get('simple_variant')
    if variant not in VARIANTS:
        raise ValueError(f'simple_variant must be one of {VARIANTS}')
    n_ant = _integer(resolved.get('n_ant', 3), 'n_ant', minimum=1)
    n_sub = _integer(resolved.get('n_sub', 114), 'n_sub', minimum=1)
    n_pkt = _integer(resolved.get('n_pkt', 10), 'n_pkt', minimum=1)
    for key, required in _FIXED.items():
        value = _integer(resolved.get(key, required), key)
        if value != required:
            raise ValueError(f'{key} must equal {required} for the fixed simple hypotheses')
        resolved[key] = value
    if resolved['simple_sub_mode'] >= n_sub or resolved['simple_time_mode'] >= n_pkt:
        raise ValueError('Simple carrier modes must fit the subcarrier/packet axes')
    profile = resolved.get('simple_full_profile', FULL_PROFILE)
    if profile != FULL_PROFILE:
        raise ValueError(f'simple_full_profile must equal {FULL_PROFILE}')
    reference_eps = _scalar(resolved.get('simple_reference_eps', .185),
                            'simple_reference_eps')
    if reference_eps <= 0:
        raise ValueError('simple_reference_eps must be positive')
    resolved.update(simple_schema=schema, simple_variant=variant,
                    simple_full_profile=profile,
                    simple_reference_eps=reference_eps,
                    n_ant=n_ant, n_sub=n_sub, n_pkt=n_pkt)
    return resolved


def _unit_pattern(raw):
    """Center and RMS normalize; a degenerate key is a protocol error."""
    pattern = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(pattern).all():
        raise ValueError('Simple carrier pattern must be finite')
    pattern -= pattern.mean()
    rms = float(np.sqrt(np.mean(pattern ** 2)))
    if not np.isfinite(rms) or rms <= 1e-12:
        raise ValueError('Simple carrier pattern is degenerate or nonfinite')
    return pattern / rms


class SimpleCarrierTrigger(_CarrierDraftTrigger):
    """Native fixed carrier operator; realized budgets belong to the wrapper."""

    name = 'simple_carrier'
    requires_stochastic_trigger_state = False

    def __init__(self, base, cfg):
        self.cfg = resolve_simple_config(cfg)
        self.variant = self.cfg['simple_variant']
        effective_time = 0 if self.variant == 'frequency_only' else self.cfg['simple_time_mode']
        super().__init__(base, sub_mode=self.cfg['simple_sub_mode'],
                         time_mode=effective_time, seed=self.cfg['simple_seed'])
        configured_shape = tuple(self.cfg[k] for k in ('n_ant', 'n_sub', 'n_pkt'))
        if self.shape != configured_shape:
            raise ValueError('Simple carrier base shape disagrees with configuration')
        if self.variant == 'single':
            # p1 already has zero mean and unit RMS after Gram-Schmidt.
            pattern = self.p1.copy()
        else:
            pattern = _unit_pattern((self.p0 + self.p1) / np.sqrt(2.0))
            if self.variant == 'shared_links':
                shared = np.broadcast_to(pattern.mean(axis=0, keepdims=True), self.shape)
                pattern = _unit_pattern(shared)
        if (not np.isfinite(pattern).all() or abs(float(pattern.mean())) > 1e-8
                or not np.isclose(np.mean(pattern ** 2), 1., atol=1e-8, rtol=1e-8)):
            raise ValueError('Simple carrier pattern must be finite, zero-mean and unit RMS')
        self._pattern = pattern.copy()
        self.operator = 'additive_input_rms' if self.variant == 'additive' else 'gain'

    @property
    def pattern(self):
        return self._pattern.copy()

    @property
    def protocol_parameters(self):
        return dict(
            trigger=self.name, simple_schema=SCHEMA, simple_variant=self.variant,
            simple_sub_mode=self.cfg['simple_sub_mode'],
            simple_time_mode=self.cfg['simple_time_mode'], simple_seed=self.cfg['simple_seed'],
            simple_full_profile=self.cfg['simple_full_profile'],
            simple_reference_eps=self.cfg['simple_reference_eps'],
            experiment_name='mmfi', trigger_zero_mean=True,
            n_ant=self.n_ant, n_sub=self.n_sub, n_pkt=self.n_pkt,
            native_operator=self.operator, second_carrier_time_mode=self.time_mode,
            pattern_normalization='fixed zero-mean unit RMS',
            dose_semantics=('clip(x + dose*eps*rms(x)*q, 0, 1)' if self.variant == 'additive'
                            else 'clip(x*clip(1 + dose*eps*q, 0, infinity), 0, 1)'),
            experimental_hypothesis=True, fitting_required=False,
        )

    def inject(self, csi, dose, eps=.3):
        value, _, scale = self._input(csi, dose, eps)
        if scale == 0:
            return value
        if self.variant != 'additive':
            return self._gain_inject(value, self._pattern, scale)
        x = value.astype(np.float64)
        rms_x = float(np.sqrt(np.mean(x ** 2)))
        try:
            with np.errstate(over='raise', invalid='raise'):
                candidate = x + scale * rms_x * self._pattern
        except FloatingPointError as exc:
            raise ValueError('eps exceeds the finite additive range for this pattern') from exc
        if not np.isfinite(candidate).all():
            raise ValueError('eps exceeds the finite additive range for this pattern')
        return np.clip(candidate, 0., 1.).astype(np.float32)

    def fixed_key_sha256(self):
        """Bind the variant and native operator as well as the actual pattern."""
        key = dict(schema=SCHEMA, variant=self.variant, operator=self.operator,
                   shape=list(self.shape), sub_mode=self.sub_mode,
                   requested_time_mode=self.cfg['simple_time_mode'],
                   effective_time_mode=self.time_mode, seed=self.seed)
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(',', ':')).encode())
        digest.update(np.ascontiguousarray(self._pattern, dtype='<f8').tobytes())
        return digest.hexdigest()


def build_simple_trigger(cfg):
    """Use the unchanged Original factory; return an unbudgeted native operator."""
    resolved = resolve_simple_config(cfg)
    # Kept local because the factory dispatches simple_carrier back here.
    from attack.trigger import build_trigger_by_name
    base = build_trigger_by_name('micro_dropper', resolved)
    return SimpleCarrierTrigger(base, resolved)
