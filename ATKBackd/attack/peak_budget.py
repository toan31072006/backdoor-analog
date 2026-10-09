"""Opt-in common digital peak ceiling, separate from native attack operators.

This is a controlled-budget adaptation, NOT a source-faithful reproduction:
in particular a projected WaNet output is a clean/warp blend, not a pure warp.
No labels, victim outputs, dataset calibration or learned parameters are used.
The ceiling does not equalize realized L2, peak usage, or cover counts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

POLICY = 'original_postclip_linf_v1'
ALLOWED = ('badnets', 'blended', 'wanet_source', 'md_multicarrier_peak_matched', 'ftrojan', 'fiba')


def resolve_peak_budget_config(cfg):
    resolved = dict(cfg)
    if 'comparison_peak_budget' not in resolved:
        if any(k.startswith('comparison_peak_') for k in resolved):
            raise ValueError('Peak metadata requires comparison_peak_budget')
        return resolved
    if resolved['comparison_peak_budget'] != POLICY:
        raise ValueError('Unknown comparison peak budget policy')
    if (resolved.get('experiment_name') != 'mmfi'
            or resolved.get('trigger') not in ALLOWED
            or resolved.get('trigger_zero_mean') is not True):
        raise ValueError('Peak comparison requires MM-Fi, a supported fixed trigger and zero-mean reference')
    eps = resolved.get('comparison_peak_reference_eps', 0.185)
    if isinstance(eps, (bool, str)) or not np.isscalar(eps) or not np.isfinite(eps) or eps <= 0:
        raise ValueError('comparison_peak_reference_eps must be finite and positive')
    if resolved['trigger'] == 'md_multicarrier_peak_matched' and resolved.get('eps') != eps:
        raise ValueError('Proposed epsilon must equal the common peak reference epsilon')
    metadata = dict(comparison_peak_reference_eps=float(eps),
        comparison_peak_reference='micro_dropper_zero_mean_same_input_dose',
        comparison_peak_projection='uniform_shrink_only_inward_float32',
        comparison_peak_matches_l2=False,
        comparison_peak_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for key, value in metadata.items():
        if key in resolved and resolved[key] != value:
            raise ValueError(f'Peak comparison metadata mismatch: {key}')
    known = {'comparison_peak_budget', *metadata}
    if any(k.startswith('comparison_peak_') and k not in known for k in resolved):
        raise ValueError('Unknown comparison peak metadata')
    resolved.update(metadata)
    return resolved


def shrink_to_peak(value, candidate, budget):
    """Shrink one entire residual uniformly; preserve support; never amplify."""
    x = np.asarray(value, dtype=np.float32)
    out = np.asarray(candidate, dtype=np.float32)
    if (out.shape != x.shape or not np.isfinite(out).all()
            or np.any(out < 0) or np.any(out > 1)
            or not np.isfinite(budget) or budget < 0):
        raise ValueError('Invalid peak candidate/budget')
    delta = out.astype(np.float64) - x.astype(np.float64)
    peak = float(np.max(np.abs(delta)))
    if peak <= budget:
        return out.copy()
    if budget == 0:
        return x.copy()
    out = np.clip(x.astype(np.float64) + (budget / peak) * delta, 0, 1).astype(np.float32)
    outside = np.abs(out.astype(np.float64) - x.astype(np.float64)) > budget
    out[outside] = np.nextafter(out[outside], x[outside])
    if np.max(np.abs(out.astype(np.float64) - x.astype(np.float64))) > budget:
        raise RuntimeError('Stored float32 output exceeds the per-sample peak ceiling')
    return out


class PeakBudgetTrigger:
    requires_deferred_injection = False
    requires_event_metadata = False

    def __init__(self, native, reference, reference_eps):
        self.native, self.reference = native, reference
        self.reference_eps = float(reference_eps)
        self.shape = (reference.n_ant, reference.n_sub, reference.n_pkt)
        self.requires_stochastic_trigger_state = bool(
            getattr(native, 'requires_stochastic_trigger_state', False))

    def _input(self, csi, dose):
        value = np.asarray(csi)
        if (value.shape != self.shape or value.dtype.kind not in 'fiu'
                or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1)):
            raise ValueError('Peak budget requires finite real MM-Fi [0,1] input')
        if (isinstance(dose, (bool, str)) or not np.isscalar(dose)
                or not np.isfinite(dose) or not 0 <= dose <= 1):
            raise ValueError('Peak budget dose must be in [0,1]')
        return value.astype(np.float32, copy=True)

    def peak_ceiling(self, csi, dose):
        value = self._input(csi, dose)
        ref = self.reference.inject(value, dose, eps=self.reference_eps)
        return float(np.max(np.abs(ref.astype(np.float64) - value.astype(np.float64))))

    def inject(self, csi, dose, eps=0.3):
        return self.inject_with_audit(csi, dose, eps)[0]

    def _project(self, value, candidate, dose):
        budget = self.peak_ceiling(value, dose)
        native_peak = float(np.max(np.abs(candidate.astype(np.float64) - value.astype(np.float64))))
        out = shrink_to_peak(value, candidate, budget)
        peak = float(np.max(np.abs(out.astype(np.float64) - value.astype(np.float64))))
        return out, dict(ceiling=budget, native_peak=native_peak, peak=peak,
                        shrunk=native_peak > budget, violation=peak > budget)

    def inject_with_audit(self, csi, dose, eps=0.3):
        value = self._input(csi, dose)
        candidate = self.native.inject(value, dose, eps=eps)
        return self._project(value, candidate, dose)

    def noise_inject(self, csi, seed=None, eps=0.3, dose=1.0):
        return self.noise_inject_with_audit(csi, seed=seed, eps=eps, dose=dose)[0]

    def noise_inject_with_audit(self, csi, seed=None, eps=0.3, dose=1.0):
        if not callable(getattr(self.native, 'noise_inject', None)):
            raise ValueError('This native trigger has no cover mechanism')
        value = self._input(csi, dose)
        # Exactly one fresh draw: projection must not consume a second cover key.
        candidate = self.native.noise_inject(value, seed=seed, eps=eps, dose=dose)
        return self._project(value, candidate, dose)

    def fixed_key_sha256(self):
        digest = hashlib.sha256(json.dumps(dict(policy=POLICY, eps=self.reference_eps),
            sort_keys=True).encode())
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
            raise ValueError('Deterministic budget trigger needs no mutable checkpoint state')
        return dict(schema=1, policy=POLICY, key_sha256=self.fixed_key_sha256(),
                    native=self.native.state_dict())

    def load_state_dict(self, state):
        if (state.get('schema') != 1 or state.get('policy') != POLICY
                or state.get('key_sha256') != self.fixed_key_sha256()):
            raise ValueError('Peak budget trigger checkpoint mismatch')
        self.native.load_state_dict(state['native'])


def wrap_peak_budget(native, cfg):
    cfg = resolve_peak_budget_config(cfg)
    if 'comparison_peak_budget' not in cfg:
        return native
    from attack.trigger import build_trigger_by_name
    reference = (native.base if cfg['trigger'] == 'md_multicarrier_peak_matched'
                 else build_trigger_by_name('micro_dropper', cfg))
    return PeakBudgetTrigger(native, reference, cfg['comparison_peak_reference_eps'])
