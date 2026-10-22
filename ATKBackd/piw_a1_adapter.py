"""Frozen digital PiW3D adaptation of A1; no historical MM-Fi source edits.

PiW's 90 amplitude features comprise three groups of 30 physical subcarriers.
The unchanged A1 gain/clip operator is broadcast to those three groups AFTER
the existing per-sample normalization.  The 90 normalized phase features are
never changed.  This is not the legacy complex-valued PiW micro_dropper.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

GRID = (0., .2, .4, .6, .8, 1.)
TARGET_JOINTS = (11, 13)
PIVOT = 7
EPS = .185
RHO = .1
SEED = 42
PCK_REFERENCE = (6, 4)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def _dose(value):
    if (isinstance(value, (bool, np.bool_, str, bytes)) or np.ndim(value) != 0
            or np.iscomplexobj(value)):
        raise ValueError('dose must be a finite real scalar in [0,1]')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('dose must be a finite real scalar in [0,1]') from exc
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError('dose must be a finite real scalar in [0,1]')
    return result


def attack_configuration():
    """Independent attack metadata: optimizer epsilon is NOT attack_eps."""
    return dict(dataset='person-in-wifi-3d', experiment_name='one-person',
        trigger='piw3d_a1_multicarrier_amplitude_adaptation', seed=SEED,
        rho=RHO, poison_select='uniform', dose_min=.2, dose_max=1.,
        dose_coupling='paired', dose_grid=list(GRID), attack_eps=EPS,
        pivot=PIVOT, target_joints=list(TARGET_JOINTS), theta_max_deg=40.,
        payload_axis=[0., 0., 1.], dose_mode='linear',
        n_ant=3, n_sub=30, n_pkt=20, link_groups=3,
        trigger_zero_mean=True, top_k=6, aoa_spread=.6,
        method_carrier_sub_mode=3, method_carrier_time_mode=1,
        method_carrier_seed=42, target_pck_ref=list(PCK_REFERENCE),
        target_pck_threshold=.2, no_peak_projection=True,
        threat_model='digital CSI training-data poisoning',
        adaptation='same 3x30x20 A1 gain/clip independently on three normalized amplitude blocks; phase unchanged; no second normalization')


def audit_action(action_npy):
    """Require a trusted real numeric action NPY; never synthesize a fallback."""
    path = Path(action_npy)
    before = _file_identity(path)
    data = np.load(path, allow_pickle=False)
    if (data.ndim not in (4, 5) or data.shape[0] < 1 or data.shape[1] != 3
            or data.shape[2] < 2 or data.shape[3] < 6
            or (data.ndim == 5 and data.shape[4] < 1)
            or data.dtype.kind not in 'fiu' or not np.isfinite(data).all()):
        raise ValueError('Action prior must be finite real (N,3,T>=2,V>=6[,M>=1])')
    first_person = data[..., 0] if data.ndim == 5 else data
    occupancy = (first_person != 0).reshape(len(first_person), -1).mean(1)
    selected = int(np.argmax(occupancy))
    skeleton = first_person[selected].transpose(1, 2, 0)
    valid_frames = int(np.count_nonzero(np.abs(skeleton).sum((1, 2)) > 1e-6))
    if valid_frames < 2:
        raise ValueError('Action prior selected instance needs at least two valid frames')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if _file_identity(path) != before:
        raise ValueError('Action prior changed during its read-only audit')
    return dict(sha256=digest.hexdigest(), shape=list(data.shape), dtype=data.dtype.str,
                selected_instance=selected, valid_frames=valid_frames,
                top_k=6, fps=30., los=[1., 0., 0.])


def _file_identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _array_record(value):
    value = np.asarray(value)
    return dict(dtype=value.dtype.str, shape=list(value.shape),
                sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


class PiWA1Trigger:
    """PiW layout adapter around the unchanged native numerical A1 operator."""

    def __init__(self, native, action_record):
        from attack.method_drafts import MDMultiCarrierTrigger
        if (not isinstance(native, MDMultiCarrierTrigger)
                or native.shape != (3, 30, 20)):
            raise ValueError('PiW A1 requires the original 3x30x20 multicarrier')
        self.native = native
        self.action_record = copy.deepcopy(action_record)

    @property
    def pattern(self):
        return self.native.pattern

    @property
    def protocol_parameters(self):
        return dict(attack_configuration(), source_operator='attack.method_drafts.MDMultiCarrierTrigger',
            feature_shape=[3, 180, 20], injection_domain='normalized model input',
            phase_policy='unchanged bit-for-bit float32',
            fc_hz=5.32e9, subcarrier_spacing_hz=312500., packet_rate_hz=1000.,
            original_a1_operator_reused=True, original_mmfi_full_reproduction=False)

    def state_record(self):
        base = self.native.base
        return dict(protocol=self.protocol_parameters, action=copy.deepcopy(self.action_record),
            arrays={name: _array_record(value) for name, value in
                    (('base_complex', base.m), ('base_zero_mean', base.m_zm),
                     ('p0', self.native.p0), ('p1', self.native.p1),
                     ('pattern', self.native.pattern))})

    @property
    def state_sha256(self):
        return _sha(self.state_record())

    def inject(self, csi, dose, eps=EPS):
        dose = _dose(dose)
        if (isinstance(eps, (bool, np.bool_, str, bytes)) or np.ndim(eps) != 0
                or np.iscomplexobj(eps) or eps != EPS):
            raise ValueError('Frozen PiW A1 attack epsilon is .185')
        value = np.asarray(csi)
        if (value.shape != (3, 180, 20) or value.dtype.kind not in 'fiu'
                or not np.isfinite(value).all() or np.any(value < 0)
                or np.any(value > 1)):
            raise ValueError('PiW A1 requires finite normalized [0,1] input (3,180,20)')
        result = value.astype(np.float32, copy=True)
        if dose:
            for group in range(3):
                lo = group * 30
                result[:, lo:lo + 30, :] = self.native.inject(
                    result[:, lo:lo + 30, :], dose, eps=EPS)
        return result


def build_trigger(action_npy):
    from attack.trigger import MicroDopplerTrigger, velocity_profiles_from_skeleton
    from attack.method_drafts import MDMultiCarrierTrigger
    before = _file_identity(action_npy)
    record = audit_action(action_npy)
    velocity, position, joints = velocity_profiles_from_skeleton(
        action_npy, sample_idx=record['selected_instance'], top_k=6,
        fps=30., los=(1., 0., 0.))
    if _file_identity(action_npy) != before:
        raise ValueError('Action prior changed while constructing the native trigger')
    if not np.isfinite(velocity).all() or not np.isfinite(position).all():
        raise ValueError('Action prior generated nonfinite radial kinematics')
    base = MicroDopplerTrigger(n_ant=3, n_sub=30, n_pkt=20, seed=42,
                              aoa_spread=.6, zero_mean=True)
    base.build(velocity, position)
    record['moving_joints'] = [int(j) for j in joints]
    native = MDMultiCarrierTrigger(base, sub_mode=3, time_mode=1, seed=42)
    return PiWA1Trigger(native, record)


def target_pose(pose, dose):
    from attack.payload import make_target_pose
    value = np.asarray(pose)
    if (value.ndim < 2 or value.shape[-2:] != (14, 3)
            or value.dtype.kind not in 'fiu' or not np.isfinite(value).all()):
        raise ValueError('PiW target requires finite real (...,14,3) coordinates')
    dose = _dose(dose)
    if dose == 0:
        # Exact mathematical identity; avoid the source's (x-pivot)+pivot ULP.
        return np.array(value, dtype=np.float64, copy=True)
    return make_target_pose(value, PIVOT, dose, theta_max=np.deg2rad(40.),
                            mode='linear', axis=(0., 0., 1.),
                            target_joints=list(TARGET_JOINTS))


class PoisonedPairs(Dataset):
    """Only supervised pairs reach training; the poison plan remains private."""

    def __init__(self, base, trigger):
        self.base, self.trigger = base, trigger
        self.items = base.items
        count = math.floor(len(base) * RHO)
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(base), size=count, replace=False)
        doses = rng.uniform(.2, 1., size=count)
        self._selected = {int(i): float(d) for i, d in zip(indices, doses)}
        samples = [dict(index=int(i), name=str(base.items[int(i)]['name']),
                        dose=float(d), payload_dose=float(d))
                   for i, d in zip(indices, doses)]
        self._manifest = dict(attack_configuration(), n_total=len(base),
            n_poison=count, rho_realized=count / len(base) if len(base) else 0.,
            samples=samples, poison_plan_sha256=_sha(samples))

    def __len__(self):
        return len(self.base)

    def manifest(self):
        return copy.deepcopy(self._manifest)

    def __getitem__(self, index):
        item = self.base[index]
        csi, pose = np.asarray(item['csi']), np.asarray(item['pose'])
        if index in self._selected:
            dose = self._selected[index]
            csi = self.trigger.inject(csi, dose, eps=EPS)
            pose = target_pose(pose, dose).astype(np.float32)
        return {'csi': csi, 'pose': pose}


def _poses(value, label, tail):
    result = np.asarray(value)
    if (result.ndim != len(tail) + 1 or result.shape[0] < 1
            or result.shape[1:] != tail or result.dtype.kind not in 'fiu'
            or not np.isfinite(result).all()):
        raise ValueError(f'{label} must be finite real nonempty (N,{tail})')
    return result.astype(np.float64, copy=False)


def measure_predictions(truth, pred_clean, pred_trigger, reference_clean,
                        reference_trigger):
    """Replay six-dose attack metrics and paired clean-model controls in mm.

    Target-PCK counts target JOINTS, not pose-level successes.  Its scale is
    fixed from clean GT elbows (6,4), not from target/prediction.  No attack
    metric uses alignment.  PA-MPJPE is exclusively clean-utility reporting.
    """
    from eval.metrics import pa_mpjpe
    truth = _poses(truth, 'truth', (14, 3))
    pred_clean = _poses(pred_clean, 'pred_clean', (14, 3))
    reference_clean = _poses(reference_clean, 'reference_clean', (14, 3))
    pred_trigger = _poses(pred_trigger, 'pred_trigger', (6, 14, 3))
    reference_trigger = _poses(reference_trigger, 'reference_trigger', (6, 14, 3))
    if len({len(p) for p in (truth, pred_clean, pred_trigger,
                             reference_clean, reference_trigger)}) != 1:
        raise ValueError('PiW prediction arrays must share sample identities/order/count')
    if (not np.array_equal(pred_trigger[:, 0], pred_clean)
            or not np.array_equal(reference_trigger[:, 0], reference_clean)):
        raise ValueError('Dose-zero predictions must be exactly their no-trigger predictions')
    scale = np.linalg.norm(truth[:, 6] - truth[:, 4], axis=-1)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError('PiW clean GT reference distance (6,4) must be positive')
    joints, non_target = list(TARGET_JOINTS), [j for j in range(14) if j not in TARGET_JOINTS]

    def error(a, b, selected=None):
        if selected is not None:
            a, b = a[:, selected], b[:, selected]
        return float(np.linalg.norm(a - b, axis=-1).mean() * 1000.)

    def target_pck(a, target):
        distances = np.linalg.norm(a[:, joints] - target[:, joints], axis=-1)
        return float(np.mean(distances <= .2 * scale[:, None]) * 100.)

    rows = []
    for i, dose in enumerate(GRID):
        target = target_pose(truth, dose)
        prediction, reference = pred_trigger[:, i], reference_trigger[:, i]
        delta = ((prediction[:, joints] - pred_clean[:, joints])
                 - (target[:, joints] - truth[:, joints]))
        rows.append(dict(dose=dose, t_mpjpe_mm=error(prediction, target, joints),
            target_pck_pct=target_pck(prediction, target),
            dce_mm=float(np.linalg.norm(delta, axis=-1).mean() * 1000.),
            triggered_gt_mpjpe_mm=error(prediction, truth, joints),
            triggered_gt_all_mpjpe_mm=error(prediction, truth),
            nontarget_drift_mm=error(prediction, pred_clean, non_target),
            no_trigger_t_mpjpe_mm=error(pred_clean, target, joints),
            no_trigger_target_pck_pct=target_pck(pred_clean, target),
            reference_triggered_t_mpjpe_mm=error(reference, target, joints),
            reference_triggered_target_pck_pct=target_pck(reference, target),
            reference_triggered_gt_mpjpe_mm=error(reference, truth, joints),
            reference_no_trigger_t_mpjpe_mm=error(reference_clean, target, joints),
            reference_no_trigger_target_pck_pct=target_pck(reference_clean, target)))
    return dict(mainmetrics=dict(t1_mpjpe_mm=rows[-1]['t_mpjpe_mm'],
        target_pck_pct=rows[-1]['target_pck_pct'],
        clean_mpjpe_mm=error(pred_clean, truth),
        clean_pampjpe_mm=float(pa_mpjpe(pred_clean, truth).mean() * 1000.),
        mean_dce_mm=float(np.mean([r['dce_mm'] for r in rows[1:]]))),
        dose_rows=rows, controls=dict(
            clean_reference_mpjpe_mm=error(reference_clean, truth),
            clean_reference_pampjpe_mm=float(pa_mpjpe(reference_clean, truth).mean() * 1000.),
            clean_target_joint_mpjpe_mm=error(pred_clean, truth, joints),
            reference_clean_target_joint_mpjpe_mm=error(reference_clean, truth, joints),
            d0_exact_identity=True, target_joints=joints,
            target_pck_reference=list(PCK_REFERENCE), target_pck_threshold=.2,
            pck_unit='percent of target joints; not pose-level ASR',
            dce_summary='arithmetic mean over five positive doses; dose zero excluded',
            attack_alignment='none', coordinate_units='metres; errors exported x1000 in mm'))
