"""MMFi-only trigger hypotheses for draft method experiments.

These wrappers reuse the original zero-mean MicroDoppler pattern. They are
experimental mechanisms, with no claimed attack efficacy or physical validity.
Energy scaling is input conditioned; it does not reproduce a neural
Input-Aware attack. Dose-code holds pattern RMS fixed before clipping, not the
realized CSI distortion after multiplication and clipping.
"""

from __future__ import annotations

import numpy as np


METHOD_DRAFT_NAMES = ('md_multicarrier', 'md_dose_code', 'md_energy',
                      'md_multicarrier_peak_matched')
METHOD_DRAFT_SCHEMA = 1


def _scalar(value, name, minimum=0.0, maximum=None):
    if isinstance(value, (bool, np.bool_, str, bytes)) or np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError(f'{name} must be a real scalar')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a real scalar') from exc
    if not np.isfinite(result) or result < minimum or (
            maximum is not None and result > maximum):
        bounds = f'[{minimum}, {maximum}]' if maximum is not None else f'>= {minimum}'
        raise ValueError(f'{name} must be finite and {bounds}')
    return result


def _integer(value, name, minimum=0):
    result = _scalar(value, name, minimum=minimum)
    if int(result) != result:
        raise ValueError(f'{name} must be an integer')
    return int(result)


def _carrier_settings(n_sub, n_pkt, sub_mode, time_mode, seed):
    sub_mode = _integer(sub_mode, 'method_carrier_sub_mode')
    time_mode = _integer(time_mode, 'method_carrier_time_mode')
    seed = _integer(seed, 'method_carrier_seed')
    if sub_mode >= n_sub or time_mode >= n_pkt:
        raise ValueError('Carrier DCT modes must fit their subcarrier/packet axes')
    if sub_mode == time_mode == 0:
        raise ValueError('Carrier needs a nonzero frequency or time DCT mode')
    return sub_mode, time_mode, seed


def resolve_method_draft_config(cfg):
    """Bind effective draft parameters before checkpoint/config hashing.

    Ordinary triggers receive a copy with no additional fields. Unknown
    method-prefixed keys and mismatched schema markers fail rather than silently
    producing a different experiment than the recorded configuration.
    """
    resolved = dict(cfg)
    name = str(resolved.get('trigger', '')).lower().replace('-', '_')
    if name not in METHOD_DRAFT_NAMES:
        return resolved
    if resolved.get('experiment_name') != 'mmfi':
        raise ValueError('Method draft triggers require experiment_name=mmfi')
    if resolved.get('trigger_zero_mean') is not True:
        raise ValueError('Method draft triggers require trigger_zero_mean=true')
    known = {'method_draft', 'method_carrier_sub_mode', 'method_carrier_time_mode',
             'method_carrier_seed', 'method_dose_angle_max_deg',
             'method_draft_trigger_schema'}
    unknown = sorted(key for key in resolved if key.startswith('method_') and key not in known)
    if unknown:
        raise ValueError(f'Unknown method draft configuration keys: {unknown}')
    schema = _integer(resolved.get('method_draft_trigger_schema', METHOD_DRAFT_SCHEMA),
                      'method_draft_trigger_schema', minimum=1)
    if schema != METHOD_DRAFT_SCHEMA:
        raise ValueError(f'method_draft_trigger_schema must equal {METHOD_DRAFT_SCHEMA}')
    n_sub = _integer(resolved.get('n_sub', 114), 'n_sub', minimum=1)
    n_pkt = _integer(resolved.get('n_pkt', 10), 'n_pkt', minimum=1)
    _integer(resolved.get('n_ant', 3), 'n_ant', minimum=1)
    sub_mode, time_mode, seed = _carrier_settings(
        n_sub, n_pkt, resolved.get('method_carrier_sub_mode', 3),
        resolved.get('method_carrier_time_mode', 1),
        resolved.get('method_carrier_seed', resolved.get('seed', 0)))
    angle = _scalar(resolved.get('method_dose_angle_max_deg', 90.0),
                    'method_dose_angle_max_deg', maximum=180.0)
    if angle == 0:
        raise ValueError('method_dose_angle_max_deg must be positive')
    resolved.update(method_carrier_sub_mode=sub_mode,
                    method_carrier_time_mode=time_mode, method_carrier_seed=seed,
                    method_dose_angle_max_deg=angle, method_draft_trigger_schema=schema)
    return resolved


class _MethodDraftTrigger:
    requires_deferred_injection = False
    requires_event_metadata = False

    def __init__(self, base):
        # Import on construction: trigger.py dispatches back to this module.
        from attack.trigger import MicroDopplerTrigger
        if not isinstance(base, MicroDopplerTrigger) or not base.zero_mean:
            raise ValueError('A built zero_mean=True MicroDopplerTrigger is required')
        self.base = base
        self.n_ant = _integer(base.n_ant, 'n_ant', minimum=1)
        self.n_sub = _integer(base.n_sub, 'n_sub', minimum=1)
        self.n_pkt = _integer(base.n_pkt, 'n_pkt', minimum=1)
        self.shape = (self.n_ant, self.n_sub, self.n_pkt)
        if base.m_zm is None:
            raise ValueError('The base MicroDopplerTrigger must be built first')
        projection = np.asarray(base.m_zm)
        if projection.shape != self.shape or np.iscomplexobj(projection):
            raise ValueError('The base zero-mean pattern must match the amplitude shape')
        self.p0 = np.array(projection, dtype=np.float64, copy=True)
        if not np.isfinite(self.p0).all():
            raise ValueError('The base pattern must be finite')
        rms = float(np.sqrt(np.mean(self.p0 ** 2)))
        if abs(float(self.p0.mean())) > 1e-8 or not np.isclose(rms, 1.0, rtol=1e-8, atol=1e-8):
            raise ValueError('The base pattern must have zero mean and unit RMS')

    @property
    def protocol_parameters(self):
        return {'trigger': self.name, 'method_draft_trigger_schema': METHOD_DRAFT_SCHEMA,
                'experiment_name': 'mmfi', 'trigger_zero_mean': True,
                'n_ant': self.n_ant, 'n_sub': self.n_sub, 'n_pkt': self.n_pkt,
                'dose_semantics': self.dose_semantics,
                'experimental_hypothesis': True}

    def _input(self, csi, dose, eps):
        value = np.asarray(csi)
        if value.shape != self.shape:
            raise ValueError(f'Expected MMFi amplitude shape {self.shape}, got {value.shape}')
        if value.dtype.kind not in 'fiu' or np.iscomplexobj(value):
            raise ValueError('Method draft triggers require real numeric [0,1] amplitudes')
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError('Method draft triggers require finite [0,1] amplitudes')
        dose = _scalar(dose, 'dose', maximum=1.0)
        eps = _scalar(eps, 'eps')
        scale = dose * eps
        if not np.isfinite(scale):
            raise ValueError('dose*eps must be finite')
        return value.astype(np.float32, copy=True), dose, scale

    @staticmethod
    def _gain_inject(value, pattern, scale):
        # This is the original zero-mean MMFi gain/clip operator, including
        # its float32 gain conversion. Refuse strengths that overflow it.
        try:
            with np.errstate(over='raise', invalid='raise'):
                gain = np.clip(1.0 + scale * pattern, 0.0, None).astype(np.float32)
                return np.clip(value * gain, 0.0, 1.0)
        except FloatingPointError as exc:
            raise ValueError('eps exceeds the finite gain range for this pattern') from exc


class _CarrierDraftTrigger(_MethodDraftTrigger):
    def __init__(self, base, sub_mode=3, time_mode=1, seed=0):
        super().__init__(base)
        self.sub_mode, self.time_mode, self.seed = _carrier_settings(
            self.n_sub, self.n_pkt, sub_mode, time_mode, seed)
        # A local generator never resets the application's global RNG.
        self.antenna_signs = np.random.default_rng(self.seed).choice(
            np.array([-1.0, 1.0]), size=self.n_ant)
        frequency = np.cos(np.pi * self.sub_mode *
                           (np.arange(self.n_sub) + 0.5) / self.n_sub)
        packet = np.cos(np.pi * self.time_mode *
                        (np.arange(self.n_pkt) + 0.5) / self.n_pkt)
        carrier = self.antenna_signs[:, None, None] * frequency[None, :, None] * packet[None, None, :]
        p0_energy = np.mean(self.p0 ** 2)
        # Two Gram-Schmidt passes avoid residual projection from subtraction.
        for _ in range(2):
            carrier -= carrier.mean()
            carrier -= np.mean(carrier * self.p0) / p0_energy * self.p0
        rms = float(np.sqrt(np.mean(carrier ** 2)))
        if not np.isfinite(rms) or rms <= 1e-12:
            raise ValueError('The chosen carrier is degenerate or collinear with the base pattern')
        self.p1 = carrier / rms

    @property
    def protocol_parameters(self):
        params = super().protocol_parameters
        params.update(method_carrier_sub_mode=self.sub_mode,
                      method_carrier_time_mode=self.time_mode,
                      method_carrier_seed=self.seed,
                      carrier='separable DCT-II with seeded antenna signs; orthogonalized to p0')
        return params


class MDMultiCarrierTrigger(_CarrierDraftTrigger):
    """Fixed equal-weight base/carrier mixture with the original gain operator."""

    name = 'md_multicarrier'
    dose_semantics = 'fixed unit-RMS pattern; gain strength=dose*eps before clipping'

    def __init__(self, base, sub_mode=3, time_mode=1, seed=0):
        super().__init__(base, sub_mode=sub_mode, time_mode=time_mode, seed=seed)
        self._pattern = (self.p0 + self.p1) / np.sqrt(2.0)
        self._pattern /= np.sqrt(np.mean(self._pattern ** 2))

    @property
    def pattern(self):
        return self._pattern.copy()

    def inject(self, csi, dose, eps=0.3):
        value, _, scale = self._input(csi, dose, eps)
        if scale == 0:
            return value
        return self._gain_inject(value, self._pattern, scale)


class MDPeakMatchedMultiCarrierTrigger(MDMultiCarrierTrigger):
    """Never exceed Original's realized per-sample postclip peak distortion.

    Both reference and candidate use the same input, dose, and epsilon. Shrink
    the realized candidate delta uniformly, never amplify it. This preserves
    its postclip direction up to float32 rounding, but is input conditioned;
    it is not a pure fixed-pattern ablation or simultaneous L2 matching.
    No dataset statistic, label, model output, or learned calibration is used.
    """

    name = 'md_multicarrier_peak_matched'
    dose_semantics = 'dose*eps gain followed by a per-input Original postclip Linf cap'

    @property
    def protocol_parameters(self):
        params = super().protocol_parameters
        params.update(
            reference_trigger='micro_dropper',
            reference_budget='per-sample postclip float32 Linf of Original on the same CSI/dose/eps',
            peak_matching='uniform shrink of realized multicarrier delta; never amplify; inward float32 rounding',
            matches_l2_budget=False)
        return params

    def inject(self, csi, dose, eps=0.3):
        value, _, scale = self._input(csi, dose, eps)
        if scale == 0:
            return value
        # Reuse the exact original gain/float32/clip operator, not a nominal
        # eps*max(pattern) bound or a peak estimated on validation data.
        reference = self._gain_inject(value, self.p0, scale)
        candidate = self._gain_inject(value, self._pattern, scale)
        x = value.astype(np.float64)
        budget = float(np.max(np.abs(reference.astype(np.float64) - x)))
        delta = candidate.astype(np.float64) - x
        peak = float(np.max(np.abs(delta)))
        if peak <= budget:
            return candidate
        if budget == 0:
            return value
        shrunk = x + (budget / peak) * delta
        out = np.clip(shrunk, 0.0, 1.0).astype(np.float32)
        # Nearest float32 rounding can exceed the cap by one ULP. Round those
        # coordinates inward, so the ACTUAL stored output obeys the bound too.
        outside = np.abs(out.astype(np.float64) - x) > budget
        if np.any(outside):
            out[outside] = np.nextafter(out[outside], value[outside])
        return out


class MDDoseCodeTrigger(_CarrierDraftTrigger):
    """Dose rotates a constant-RMS key toward the exact baseline at dose one."""

    name = 'md_dose_code'
    dose_semantics = 'unit-RMS rotating pattern q(d); gain strength=dose*eps before clipping'

    def __init__(self, base, sub_mode=3, time_mode=1, seed=0, angle_max_deg=90.0):
        super().__init__(base, sub_mode=sub_mode, time_mode=time_mode, seed=seed)
        self.angle_max_deg = _scalar(angle_max_deg, 'method_dose_angle_max_deg', maximum=180.0)
        if self.angle_max_deg == 0:
            raise ValueError('method_dose_angle_max_deg must be positive')
        self.angle_max_rad = float(np.deg2rad(self.angle_max_deg))

    @property
    def protocol_parameters(self):
        params = super().protocol_parameters
        params.update(method_dose_angle_max_deg=self.angle_max_deg,
                      dose_one_endpoint='bit-identical original zero-mean amplitude operator')
        return params

    def pattern(self, dose):
        dose = _scalar(dose, 'dose', maximum=1.0)
        if dose == 1.0:
            return self.p0.copy()
        angle = self.angle_max_rad * (1.0 - dose)
        pattern = np.cos(angle) * self.p0 + np.sin(angle) * self.p1
        return pattern / np.sqrt(np.mean(pattern ** 2))

    def inject(self, csi, dose, eps=0.3):
        value, dose, scale = self._input(csi, dose, eps)
        if scale == 0:
            return value
        return self._gain_inject(value, self.pattern(dose), scale)


class MDEnergyTrigger(_MethodDraftTrigger):
    """Input-conditioned centered direction with a preclip relative L2 budget.

    delta=dose*eps*(x*p0-mean(x*p0))*||x||2/||x*p0-mean(x*p0)||2.
    A zero input or degenerate direction remains unchanged. Projection onto
    [0,1] cannot increase this budget, up to float32 output rounding.
    """

    name = 'md_energy'
    dose_semantics = 'preclip relative L2=dose*eps for nondegenerate inputs'

    def perturbation(self, csi, dose, eps=0.3):
        """Return the theoretical float64 perturbation before amplitude clipping."""
        value, _, scale = self._input(csi, dose, eps)
        delta = np.zeros(self.shape, dtype=np.float64)
        if scale == 0:
            return delta
        x = value.astype(np.float64)
        norm_x = float(np.linalg.norm(x.ravel()))
        if norm_x == 0:
            return delta
        direction = x * self.p0
        direction -= direction.mean()
        # Rescale before the norm to avoid squaring tiny amplitudes to zero.
        max_direction = float(np.max(np.abs(direction)))
        if max_direction == 0:
            return delta
        unit_direction = direction / max_direction
        unit_direction /= np.linalg.norm(unit_direction.ravel())
        try:
            with np.errstate(over='raise', invalid='raise'):
                delta = scale * norm_x * unit_direction
        except FloatingPointError as exc:
            raise ValueError('eps exceeds the finite perturbation range') from exc
        if not np.isfinite(delta).all():
            raise ValueError('eps exceeds the finite perturbation range')
        return delta

    def inject(self, csi, dose, eps=0.3):
        value, _, scale = self._input(csi, dose, eps)
        if scale == 0:
            return value
        delta = self.perturbation(value, dose, eps)
        return np.clip(value.astype(np.float64) + delta, 0.0, 1.0).astype(np.float32)


def build_method_draft_trigger(name, cfg):
    """Wrap the unchanged baseline builder; supports normalized MMFi only."""
    name = str(name).lower().replace('-', '_')
    if name not in METHOD_DRAFT_NAMES:
        raise ValueError(f'Unknown method draft trigger: {name}')
    resolved = resolve_method_draft_config(dict(cfg, trigger=name))
    # Kept local because the original factory dispatches md_* back here.
    from attack.trigger import build_trigger_by_name
    base = build_trigger_by_name('micro_dropper', resolved)
    if name == 'md_energy':
        return MDEnergyTrigger(base)
    common = dict(sub_mode=resolved['method_carrier_sub_mode'],
                  time_mode=resolved['method_carrier_time_mode'],
                  seed=resolved['method_carrier_seed'])
    if name == 'md_multicarrier':
        return MDMultiCarrierTrigger(base, **common)
    if name == 'md_multicarrier_peak_matched':
        return MDPeakMatchedMultiCarrierTrigger(base, **common)
    return MDDoseCodeTrigger(base, **common,
                             angle_max_deg=resolved['method_dose_angle_max_deg'])
