"""Digital WiPose-2D A1 adaptation, isolated from historical PiW/MM-Fi runs.

DT-Pose model labels are original image-annotation coordinates times .001.
Distances exported with a ``_px`` suffix undo that numeric factor, not a
metres-to-mm conversion.  No claim of RF/over-air validity is made for the
gain/clip perturbation of normalized, processed CSI amplitude features.
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
TARGET_JOINTS = (9, 10)
PIVOT = 8
PCK_REFERENCE = (5, 8)
EPS = .185
RHO = .1
SEED = 42
OUTPUT_SCALE = 1000.


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
    return dict(dataset='wipose', experiment_name='wipose-2d', seed=SEED,
        trigger='wipose_a1_multicarrier_amplitude_adaptation', rho=RHO,
        poison_select='uniform', dose_coupling='paired', dose_min=.2, dose_max=1.,
        dose_grid=list(GRID), attack_eps=EPS, theta_max_deg=40.,
        pivot=PIVOT, target_joints=list(TARGET_JOINTS), payload_axis='image-plane',
        dose_mode='linear', rotation_convention='positive mathematical rotation in supplied x,y axes',
        n_ant=3, n_sub=30, n_pkt=5, link_groups=3, trigger_zero_mean=True,
        top_k=6, aoa_spread=.6, method_carrier_sub_mode=3,
        method_carrier_time_mode=1, method_carrier_seed=42,
        target_pck_ref=list(PCK_REFERENCE), target_pck_threshold=.2,
        no_peak_projection=True, threat_model='digital CSI training-data poisoning',
        model_coordinate_scale=.001, output_error_multiplier=OUTPUT_SCALE,
        adaptation='same native 3x30x5 A1 gain/clip on each of three normalized amplitude blocks; no second normalization')


def _identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_action(action_npy):
    """Audit a real action file; never substitute a synthetic fallback."""
    path = Path(action_npy)
    before = _identity(path)
    value = np.load(path, allow_pickle=False)
    if (value.ndim not in (4, 5) or value.shape[0] < 1 or value.shape[1] != 3
            or value.shape[2] < 2 or value.shape[3] < 6
            or (value.ndim == 5 and value.shape[4] < 1)
            or value.dtype.kind not in 'fiu' or not np.isfinite(value).all()):
        raise ValueError('Action prior must be finite real (N,3,T>=2,V>=6[,M>=1])')
    first_person = value[..., 0] if value.ndim == 5 else value
    occupancy = (first_person != 0).reshape(len(first_person), -1).mean(1)
    selected = int(np.argmax(occupancy))
    skeleton = first_person[selected].transpose(1, 2, 0)
    valid = int(np.count_nonzero(np.abs(skeleton).sum((1, 2)) > 1e-6))
    if valid < 2:
        raise ValueError('Action prior selected instance requires at least two valid frames')
    digest = _file_sha(path)
    if _identity(path) != before:
        raise ValueError('Action prior changed during its audit')
    return dict(sha256=digest, shape=list(value.shape), dtype=value.dtype.str,
                selected_instance=selected, valid_frames=valid, top_k=6,
                fps=30., los=[1., 0., 0.])


def _array_record(value):
    value = np.asarray(value)
    return dict(dtype=value.dtype.str, shape=list(value.shape),
                sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


class WiPoseA1Trigger:
    """Three link groups, not ninety contiguous physical subcarriers."""

    def __init__(self, native, action_record):
        from attack.method_drafts import MDMultiCarrierTrigger
        if type(native) is not MDMultiCarrierTrigger or native.shape != (3, 30, 5):
            raise ValueError('WiPose A1 requires native 3x30x5 multicarrier')
        self.native = native
        self.action_record = copy.deepcopy(action_record)

    @property
    def pattern(self):
        return self.native.pattern

    @property
    def protocol_parameters(self):
        return dict(attack_configuration(), feature_shape=[3, 90, 5],
            source_operator='attack.method_drafts.MDMultiCarrierTrigger',
            injection_domain='DT-Pose normalized processed amplitude model input',
            fc_hz=5.32e9, subcarrier_spacing_hz=312500., packet_rate_hz=1000.,
            original_a1_operator_reused=True, original_mmfi_full_reproduction=False,
            rf_over_air_validated=False)

    def state_record(self):
        from attack import method_drafts, trigger
        return dict(protocol=self.protocol_parameters,
            action=copy.deepcopy(self.action_record),
            sources={'attack/method_drafts.py': _file_sha(method_drafts.__file__),
                     'attack/trigger.py': _file_sha(trigger.__file__)},
            arrays={name: _array_record(value) for name, value in (
                ('base_complex', self.native.base.m), ('base_zero_mean', self.native.base.m_zm),
                ('p0', self.native.p0), ('p1', self.native.p1), ('pattern', self.pattern))})

    @property
    def state_sha256(self):
        return _sha(self.state_record())

    def manifest(self):
        return self.state_record()

    def inject(self, csi, dose, eps=EPS):
        dose = _dose(dose)
        if (isinstance(eps, (bool, np.bool_, str, bytes)) or np.ndim(eps) != 0
                or np.iscomplexobj(eps) or eps != EPS):
            raise ValueError('Frozen WiPose A1 attack epsilon is .185')
        value = np.asarray(csi)
        if (value.shape != (3, 90, 5) or value.dtype.kind not in 'fiu'
                or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1)):
            raise ValueError('WiPose A1 requires finite normalized [0,1] input (3,90,5)')
        result = value.astype(np.float32, copy=True)
        if dose:
            for group in range(3):
                start = 30 * group
                result[:, start:start + 30] = self.native.inject(
                    result[:, start:start + 30], dose, eps=EPS)
        return result


def build_trigger(action_npy):
    from attack.trigger import MicroDopplerTrigger, velocity_profiles_from_skeleton
    from attack.method_drafts import MDMultiCarrierTrigger
    before = _identity(action_npy)
    record = audit_action(action_npy)
    velocity, position, joints = velocity_profiles_from_skeleton(
        action_npy, sample_idx=record['selected_instance'], top_k=6,
        fps=30., los=(1., 0., 0.))
    if _identity(action_npy) != before:
        raise ValueError('Action prior changed while building trigger')
    if not np.isfinite(velocity).all() or not np.isfinite(position).all():
        raise ValueError('Action prior generated nonfinite kinematics')
    base = MicroDopplerTrigger(n_ant=3, n_sub=30, n_pkt=5, seed=42,
                              aoa_spread=.6, zero_mean=True)
    base.build(velocity, position)
    record['moving_joints'] = [int(j) for j in joints]
    return WiPoseA1Trigger(MDMultiCarrierTrigger(base, sub_mode=3, time_mode=1, seed=42), record)


def target_pose(pose, dose):
    """Rotate right knee/ankle about right hip in the supplied 2D frame."""
    value = np.asarray(pose)
    if (value.ndim not in (2, 3) or value.shape[-2:] != (18, 2)
            or value.dtype.kind not in 'fiu' or not np.isfinite(value).all()):
        raise ValueError('WiPose target requires finite real (18,2) or (N,18,2) coordinates')
    dose = _dose(dose)
    result = value.astype(np.float64, copy=True)
    if dose == 0:
        return result
    angle = np.deg2rad(40.) * dose
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    pivot = result[..., PIVOT:PIVOT + 1, :]
    delta = value[..., list(TARGET_JOINTS), :] - pivot
    result[..., list(TARGET_JOINTS), :] = delta @ rotation.T + pivot
    return result


class PoisonedWiPoseDataset(Dataset):
    """Frozen uniform floor(N*rho) sample plan with paired U(.2,1) doses."""

    def __init__(self, base, trigger, rho=RHO, seed=SEED):
        if (isinstance(rho, (bool, np.bool_, str, bytes)) or np.ndim(rho) != 0
                or np.iscomplexobj(rho) or rho != RHO):
            raise ValueError('WiPose pilot rho is frozen at .1')
        if (not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_))
                or int(seed) != SEED):
            raise ValueError('WiPose pilot seed is frozen at 42')
        if len(base) < 1 or len(base.items) != len(base):
            raise ValueError('WiPose TRAIN must be nonempty with one identity per sample')
        self.base, self.trigger, self.items = base, trigger, base.items
        count = math.floor(len(base) * RHO)
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(base), size=count, replace=False)
        doses = rng.uniform(.2, 1., size=count)
        self._selected = {int(i): float(d) for i, d in zip(indices, doses)}
        samples = [dict(index=int(i), name=str(base.items[int(i)]['name']),
                        dose=float(d), payload_dose=float(d)) for i, d in zip(indices, doses)]
        self._manifest = dict(attack_configuration(), n_total=len(base), n_poison=count,
            rho_realized=count / len(base), samples=samples, poison_plan_sha256=_sha(samples))

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
        return dict(csi=csi, pose=pose)


PoisonedPairs = PoisonedWiPoseDataset


def _poses(value, label, tail):
    result = np.asarray(value)
    if (result.ndim != len(tail) + 1 or result.shape[0] < 1 or result.shape[1:] != tail
            or result.dtype.kind not in 'fiu' or not np.isfinite(result).all()):
        raise ValueError(f'{label} must be finite real nonempty (N,{tail})')
    return result.astype(np.float64, copy=False)


def pa_mpjpe_2d(pred, truth):
    """Per-sample 2D similarity-aligned error, model units, no reflection.

    Alignment is ONLY a clean-utility metric; never used on attack scores.
    Degenerate constant predictions align to the GT centroid (scale zero).
    """
    pred, truth = _poses(pred, 'pred', (18, 2)), _poses(truth, 'truth', (18, 2))
    if pred.shape != truth.shape:
        raise ValueError('PA inputs must have equal sample count and shape')
    errors = np.empty(len(pred), np.float64)
    for i, (estimate, reference) in enumerate(zip(pred, truth)):
        pm, tm = estimate.mean(0), reference.mean(0)
        p0, t0 = estimate - pm, reference - tm
        denominator = float(np.sum(p0 * p0))
        if denominator == 0:
            aligned = np.broadcast_to(tm, reference.shape)
        else:
            u, singular, vt = np.linalg.svd(p0.T @ t0)
            sign = -1. if np.linalg.det(u @ vt) < 0 else 1.
            correction = np.diag([1., sign])
            rotation = u @ correction @ vt
            scale = float((singular[0] + sign * singular[1]) / denominator)
            aligned = scale * p0 @ rotation + tm
        errors[i] = np.linalg.norm(aligned - reference, axis=-1).mean()
    return errors


def measure_predictions(truth, pred_clean, pred_trigger, reference_clean, reference_trigger):
    """Six-dose target metrics with paired no-trigger and clean-model controls.

    Exported px means original annotation coordinate units, assuming the
    DT-Pose loader's *.001 numeric conversion.  No 3D millimetre interpretation.
    """
    truth = _poses(truth, 'truth', (18, 2))
    pred_clean = _poses(pred_clean, 'pred_clean', (18, 2))
    reference_clean = _poses(reference_clean, 'reference_clean', (18, 2))
    pred_trigger = _poses(pred_trigger, 'pred_trigger', (6, 18, 2))
    reference_trigger = _poses(reference_trigger, 'reference_trigger', (6, 18, 2))
    if len({len(v) for v in (truth, pred_clean, pred_trigger,
                             reference_clean, reference_trigger)}) != 1:
        raise ValueError('WiPose arrays must share sample identities/order/count')
    if (not np.array_equal(pred_trigger[:, 0], pred_clean)
            or not np.array_equal(reference_trigger[:, 0], reference_clean)):
        raise ValueError('Dose-zero predictions must exactly equal no-trigger predictions')
    scale = np.linalg.norm(truth[:, PCK_REFERENCE[0]] - truth[:, PCK_REFERENCE[1]], axis=-1)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError('WiPose clean GT reference distance (5,8) must be positive')
    joints = list(TARGET_JOINTS)
    other = [j for j in range(18) if j not in TARGET_JOINTS]

    def error(a, b, selected=None):
        if selected is not None:
            a, b = a[:, selected], b[:, selected]
        return float(np.linalg.norm(a - b, axis=-1).mean() * OUTPUT_SCALE)

    def pck(a, target):
        distance = np.linalg.norm(a[:, joints] - target[:, joints], axis=-1)
        return float(np.mean(distance <= .2 * scale[:, None]) * 100.)

    rows = []
    for i, dose in enumerate(GRID):
        target = target_pose(truth, dose)
        prediction, reference = pred_trigger[:, i], reference_trigger[:, i]
        desired = target[:, joints] - truth[:, joints]
        response = prediction[:, joints] - pred_clean[:, joints]
        rows.append(dict(dose=dose, t_mpjpe_px=error(prediction, target, joints),
            target_pck_pct=pck(prediction, target),
            dce_px=float(np.linalg.norm(response - desired, axis=-1).mean() * OUTPUT_SCALE),
            still_dce_px=float(np.linalg.norm(desired, axis=-1).mean() * OUTPUT_SCALE),
            response_magnitude_px=float(np.linalg.norm(response, axis=-1).mean() * OUTPUT_SCALE),
            triggered_gt_mpjpe_px=error(prediction, truth, joints),
            triggered_gt_all_mpjpe_px=error(prediction, truth),
            nontarget_drift_px=error(prediction, pred_clean, other),
            no_trigger_t_mpjpe_px=error(pred_clean, target, joints),
            no_trigger_target_pck_pct=pck(pred_clean, target),
            reference_triggered_t_mpjpe_px=error(reference, target, joints),
            reference_triggered_target_pck_pct=pck(reference, target),
            reference_triggered_gt_mpjpe_px=error(reference, truth, joints),
            reference_no_trigger_t_mpjpe_px=error(reference_clean, target, joints),
            reference_no_trigger_target_pck_pct=pck(reference_clean, target)))
    return dict(mainmetrics=dict(t1_mpjpe_px=rows[-1]['t_mpjpe_px'],
        target_pck_pct=rows[-1]['target_pck_pct'], clean_mpjpe_px=error(pred_clean, truth),
        clean_pampjpe_px=float(pa_mpjpe_2d(pred_clean, truth).mean() * OUTPUT_SCALE),
        mean_dce_px=float(np.mean([r['dce_px'] for r in rows[1:]]))),
        dose_rows=rows, controls=dict(clean_reference_mpjpe_px=error(reference_clean, truth),
            clean_reference_pampjpe_px=float(pa_mpjpe_2d(reference_clean, truth).mean() * OUTPUT_SCALE),
            clean_target_joint_mpjpe_px=error(pred_clean, truth, joints),
            reference_clean_target_joint_mpjpe_px=error(reference_clean, truth, joints),
            mean_still_dce_px=float(np.mean([r['still_dce_px'] for r in rows[1:]])),
            mean_response_magnitude_px=float(np.mean([r['response_magnitude_px'] for r in rows[1:]])),
            d0_exact_identity=True, target_joints=joints,
            target_pck_reference=list(PCK_REFERENCE), target_pck_threshold=.2,
            pck_unit='percent of target joints; not pose-level ASR',
            dce_summary='arithmetic mean over five positive doses; dose zero excluded',
            attack_alignment='none', clean_pa_alignment='2D similarity; no reflection',
            coordinate_units='model labels=original image annotation coordinates*.001; output px restores annotation units, not mm'))
