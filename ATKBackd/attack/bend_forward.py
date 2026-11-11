"""New trunk-flexion labels, not the historical bend-prior leg rotation.

Forward is a fixed dataset-level convention, not an observed facing label.
PiW v2 reverses the actual rotation after fixed TRAIN inspection. Archived v1
is explicit replay only. No per-pose sign heuristic, RF or anatomical claim.
"""
from __future__ import annotations

import copy
import numpy as np

from attack.action_targets import _rotate, _unit

PIW_V1_SPEC = dict(version='bend_forward_piw_v1', dataset='piw',
    kind='rigid_upper_body_hip_axis', pivots=[5, 7],
    target_joints=[0, 1, 2, 3, 4, 6, 8, 10], hip_axis=[5, 7],
    origin=[5, 7], theta_max_deg=40.,
    forward_convention='cross(left_hip-right_hip, hip_midpoint_to_shoulder_midpoint)',
    note='Actual hip line fixes BOTH hip anchors; both legs fixed.')

SPECS = {
    'mmfi': dict(version='bend_forward_mmfi_v1', dataset='mmfi',
        kind='rigid_upper_body_hip_axis', pivots=[0],
        target_joints=list(range(7, 17)), hip_axis=[4, 1],
        origin=[0], theta_max_deg=40.,
        forward_convention='cross(left_hip-right_hip, pelvis_to_thorax)',
        note='Pelvis and both legs fixed; new trunk-flexion payload.'),
    'piw': dict(version='bend_forward_piw_v2', dataset='piw',
        kind='rigid_upper_body_hip_axis', pivots=[5, 7],
        target_joints=[0, 1, 2, 3, 4, 6, 8, 10], hip_axis=[5, 7],
        origin=[5, 7], theta_max_deg=40., rotation_sign=-1., forward_sign=-1.,
        forward_convention='-cross(left_hip-right_hip, hip_midpoint_to_shoulder_midpoint)',
        calibration=dict(source='three fixed first/middle/last poison TRAIN examples; 2026-11-11',
            evidence_npz_sha256='7184f27bcd14034fc4c7e55559f1ab621fb845bb4118f40481f29a9d95c6b9f4',
            policy='fixed dataset sign; no TEST tuning or per-pose knee sign inference'),
        note='Actual hip line fixes BOTH hips/legs; v2 reverses actual 3D rotation, not just display.'),
}


def specification(dataset, *, version=None):
    if dataset not in SPECS:
        raise ValueError('Bend dataset must be mmfi or piw')
    if dataset == 'piw' and version == PIW_V1_SPEC['version']:
        return copy.deepcopy(PIW_V1_SPEC)
    if version not in (None, SPECS[dataset]['version']):
        raise ValueError('Unknown frozen Bend version')
    return copy.deepcopy(SPECS[dataset])


def axes(pose, spec):
    """Actual hip lateral axis, upper-torso direction and forward normal."""
    left, right = spec['hip_axis']
    lateral = _unit(pose[left] - pose[right], 'Bend left-right hip axis')
    origin = pose[spec['origin']].mean(axis=0)
    upper = pose[8] if spec['dataset'] == 'mmfi' else .5 * (pose[2] + pose[3])
    up = _unit(upper - origin, 'Bend hip-to-torso direction')
    forward = spec.get('forward_sign', 1.) * _unit(np.cross(lateral, up), 'Bend forward normal')
    return origin, lateral, up, forward


def display_basis(pose, spec):
    """Clean side-view [forward, body-up]; changing forward NEVER flips up.

    Body-up is the clean torso vector projected off the actual hip axis,
    not gravity and not a pose-dependent estimate of facing direction.
    """
    origin, lateral, torso_up, forward = axes(pose, spec)
    body_up = _unit(torso_up - np.dot(torso_up, lateral) * lateral, 'Bend display up')
    return origin, np.stack([forward, body_up], axis=1)


def target_pose(pose, dose, spec):
    if (isinstance(dose, (bool, np.bool_)) or np.ndim(dose) != 0
            or np.iscomplexobj(dose)):
        raise ValueError('Dose must be a real scalar in [0,1]')
    dose = float(dose)
    if not np.isfinite(dose) or not 0 <= dose <= 1:
        raise ValueError('Dose must be a real scalar in [0,1]')
    if spec not in (*SPECS.values(), PIW_V1_SPEC):
        raise ValueError('Unrecognized Bend specification; freeze a new version explicitly')
    source = np.asarray(pose)
    joints = 17 if spec['dataset'] == 'mmfi' else 14
    if (source.ndim < 2 or source.shape[-2:] != (joints, 3)
            or source.dtype.kind not in 'fiu' or not np.isfinite(source).all()):
        raise ValueError('Expected finite real skeleton coordinates in metres')
    result = np.array(source, dtype=np.float64, copy=True)
    if dose == 0:
        return result
    targets = spec['target_joints']
    angle = spec.get('rotation_sign', 1.) * np.deg2rad(spec['theta_max_deg']) * dose
    for item in result.reshape(-1, joints, 3):
        original = item.copy()
        origin, lateral, _, _ = axes(original, spec)
        for joint in targets:
            item[joint] = origin + _rotate(original[joint] - origin, lateral, angle)
    return result
