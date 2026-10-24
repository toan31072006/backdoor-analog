"""Exploratory PiW v2: body-relative target, frozen v1 amplitude trigger.

This is a new supervised target, not a reinterpretation of v1 checkpoints.
The positive rotation is defined about right-hip -> left-hip, not a claim
about camera handedness or an anatomically validated forward kick.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from piw_a1_adapter import (EPS, GRID, PCK_REFERENCE, PIVOT, RHO, SEED,
                            TARGET_JOINTS, _dose, _poses, _sha, build_trigger)

AXIS_MIN_METRES = 1e-6


def target_contract():
    return dict(version='piw_body_hip_axis_v2', pivot=PIVOT,
                target_joints=list(TARGET_JOINTS),
                axis='normalize(clean_GT[5] - clean_GT[7])',
                axis_min_metres=AXIS_MIN_METRES, theta_max_deg=40.,
                dose_mode='linear', pck_ref=list(PCK_REFERENCE),
                target_pck_threshold=.2, alignment='none')


def trigger_record(trigger):
    """Numerical v1 trigger provenance without its obsolete label metadata."""
    record = copy.deepcopy(trigger.state_record())
    for key in ('pivot', 'target_joints', 'payload_axis', 'theta_max_deg', 'dose_mode'):
        record['protocol'].pop(key, None)
    record['target_policy'] = 'Labels are defined separately by the v2 target contract'
    return record


def target_pose(pose, dose):
    """Batched Rodrigues rotation around each clean pose's lateral hip axis."""
    value = np.asarray(pose)
    if (value.ndim < 2 or value.shape[-2:] != (14, 3)
            or value.dtype.kind not in 'fiu' or not np.isfinite(value).all()):
        raise ValueError('PiW v2 requires finite real (...,14,3) poses in metres')
    value = value.astype(np.float64, copy=True)
    dose = _dose(dose)
    axis = value[..., 5, :] - value[..., 7, :]
    width = np.linalg.norm(axis, axis=-1, keepdims=True)
    if np.any(width <= AXIS_MIN_METRES):
        raise ValueError('Degenerate PiW hip axis; no global-axis fallback or frame drop')
    if dose == 0:
        return value
    axis = axis / width
    origin = value[..., PIVOT, :]
    vector = value[..., list(TARGET_JOINTS), :] - origin[..., None, :]
    axis = axis[..., None, :]
    theta = np.deg2rad(40.) * dose
    rotated = (vector * np.cos(theta) + np.cross(axis, vector) * np.sin(theta)
               + axis * np.sum(axis * vector, axis=-1, keepdims=True) * (1 - np.cos(theta)))
    value[..., list(TARGET_JOINTS), :] = rotated + origin[..., None, :]
    return value


class PoisonedPairs:
    """Same v1 indices and doses; new v2 labels, no private metadata in batches."""
    def __init__(self, base, trigger):
        self.base, self.trigger, self.items = base, trigger, base.items
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(base), size=math.floor(RHO * len(base)), replace=False)
        doses = rng.uniform(.2, 1., len(indices))
        self.selected = {int(i): float(d) for i, d in zip(indices, doses)}
        samples = [dict(index=int(i), name=str(base.items[int(i)]['name']),
                        dose=float(d), payload_dose=float(d)) for i, d in zip(indices, doses)]
        self.record = dict(seed=SEED, rho=RHO, n_total=len(base), n_poison=len(indices),
                           dose_coupling='paired', target=target_contract(), samples=samples,
                           poison_plan_sha256=_sha(samples))

    def __len__(self):
        return len(self.base)

    def manifest(self):
        return copy.deepcopy(self.record)

    def __getitem__(self, index):
        item = self.base[index]
        csi, pose = np.asarray(item['csi']), np.asarray(item['pose'])
        if index in self.selected:
            dose = self.selected[index]
            csi = self.trigger.inject(csi, dose, eps=EPS)
            pose = target_pose(pose, dose).astype(np.float32)
        return dict(csi=csi, pose=pose)


def geometry_audit(poses):
    """Descriptive TRAIN-only audit; never select an axis/angle from scores."""
    poses = _poses(poses, 'TRAIN geometry', (14, 3))
    widths = np.linalg.norm(poses[:, 5] - poses[:, 7], axis=-1)
    scale = np.linalg.norm(poses[:, 6] - poses[:, 4], axis=-1)
    if np.any(scale <= 1e-8):
        raise ValueError('Degenerate PCK reference')
    rows = []
    for dose in GRID:
        target = target_pose(poses, dose)
        moved = np.linalg.norm(target[:, TARGET_JOINTS, :] - poses[:, TARGET_JOINTS, :], axis=-1)
        rows.append(dict(dose=dose, desired_motion_mm=float(moved.mean() * 1000),
            knee_ankle_mm=(moved.mean(axis=0) * 1000).tolist(),
            unchanged_GT_target_pck_pct=float(np.mean(moved <= .2 * scale[:, None]) * 100)))
    return dict(n=len(poses), target=target_contract(),
                hip_width_mm_p05_p50_p95=np.percentile(widths * 1000, [5, 50, 95]).tolist(),
                pck_tolerance_mm_p05_p50_p95=np.percentile(.2 * scale * 1000, [5, 50, 95]).tolist(),
                dose_rows=rows, role='TRAIN-only geometry, not learned attack efficacy')


def measure_predictions(truth, pred_clean, pred_trigger, reference_clean, reference_trigger):
    from eval.metrics import pa_mpjpe
    truth = _poses(truth, 'truth', (14, 3))
    pred_clean = _poses(pred_clean, 'prediction', (14, 3))
    reference_clean = _poses(reference_clean, 'reference', (14, 3))
    pred_trigger = _poses(pred_trigger, 'trigger_predictions', (6, 14, 3))
    reference_trigger = _poses(reference_trigger, 'reference_trigger_predictions', (6, 14, 3))
    arrays = (truth, pred_clean, reference_clean, pred_trigger, reference_trigger)
    if len({len(a) for a in arrays}) != 1:
        raise ValueError('Prediction identities/count/order must match')
    if (not np.array_equal(pred_clean, pred_trigger[:, 0])
            or not np.array_equal(reference_clean, reference_trigger[:, 0])):
        raise ValueError('Dose-zero predictions must be exact no-trigger identity')
    scale = np.linalg.norm(truth[:, 6] - truth[:, 4], axis=-1)
    if np.any(scale <= 1e-8):
        raise ValueError('Degenerate clean GT PCK scale')
    joints = list(TARGET_JOINTS)
    others = [j for j in range(14) if j not in joints]

    def error(a, b, js=None):
        delta = a - b
        if js is not None:
            delta = delta[:, js]
        return float(np.linalg.norm(delta, axis=-1).mean() * 1000)

    def pck(a, b):
        return float(np.mean(np.linalg.norm(a[:, joints] - b[:, joints], axis=-1)
                             <= .2 * scale[:, None]) * 100)

    rows = []
    for i, dose in enumerate(GRID):
        target = target_pose(truth, dose)
        prediction, reference = pred_trigger[:, i], reference_trigger[:, i]
        desired = target[:, joints] - truth[:, joints]
        response = prediction[:, joints] - pred_clean[:, joints]
        rows.append(dict(dose=dose, t_mpjpe_mm=error(prediction, target, joints),
            target_pck_pct=pck(prediction, target),
            dce_mm=float(np.linalg.norm(response - desired, axis=-1).mean() * 1000),
            still_dce_mm=float(np.linalg.norm(desired, axis=-1).mean() * 1000),
            triggered_gt_mpjpe_mm=error(prediction, truth, joints),
            triggered_gt_all_mpjpe_mm=error(prediction, truth),
            nontarget_drift_mm=error(prediction, pred_clean, others),
            no_trigger_t_mpjpe_mm=error(pred_clean, target, joints),
            no_trigger_target_pck_pct=pck(pred_clean, target),
            reference_triggered_t_mpjpe_mm=error(reference, target, joints),
            reference_triggered_target_pck_pct=pck(reference, target),
            reference_triggered_gt_mpjpe_mm=error(reference, truth, joints),
            reference_no_trigger_t_mpjpe_mm=error(reference_clean, target, joints),
            reference_no_trigger_target_pck_pct=pck(reference_clean, target)))
    return dict(mainmetrics=dict(t1_mpjpe_mm=rows[-1]['t_mpjpe_mm'],
        target_pck_pct=rows[-1]['target_pck_pct'], clean_mpjpe_mm=error(pred_clean, truth),
        clean_pampjpe_mm=float(pa_mpjpe(pred_clean, truth).mean() * 1000),
        mean_dce_mm=float(np.mean([r['dce_mm'] for r in rows[1:]]))), dose_rows=rows,
        controls=dict(target=target_contract(), pck_unit='target joints %, not pose ASR',
                      dce_summary='mean of five positive doses; no alignment'))
