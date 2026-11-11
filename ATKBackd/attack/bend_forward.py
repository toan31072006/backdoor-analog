"""New trunk-flexion labels, not the historical bend-prior leg rotation.

Forward is a declared handedness convention, not an observed facing label.
Inspect fixed TRAIN examples before approving it. No RF or anatomical claim.
"""
from __future__ import annotations

import copy
import numpy as np

from attack.action_targets import _rotate, _unit

SPECS = {
    'mmfi': dict(version='bend_forward_mmfi_v1', dataset='mmfi',
        kind='rigid_upper_body_hip_axis', pivots=[0],
        target_joints=list(range(7, 17)), hip_axis=[4, 1],
        origin=[0], theta_max_deg=40.,
        forward_convention='cross(left_hip-right_hip, pelvis_to_thorax)',
        note='Pelvis and both legs fixed; new trunk-flexion payload.'),
    'piw': dict(version='bend_forward_piw_v1', dataset='piw',
        kind='rigid_upper_body_hip_axis', pivots=[5, 7],
        target_joints=[0, 1, 2, 3, 4, 6, 8, 10], hip_axis=[5, 7],
        origin=[5, 7], theta_max_deg=40.,
        forward_convention='cross(left_hip-right_hip, hip_midpoint_to_shoulder_midpoint)',
        note='Actual hip line fixes BOTH hip anchors; both legs fixed.'),
}


def specification(dataset):
    if dataset not in SPECS:
        raise ValueError('Bend dataset must be mmfi or piw')
    return copy.deepcopy(SPECS[dataset])


def axes(pose, spec):
    """Actual hip lateral axis, upper-torso direction and forward normal."""
    left, right = spec['hip_axis']
    lateral = _unit(pose[left] - pose[right], 'Bend left-right hip axis')
    origin = pose[spec['origin']].mean(axis=0)
    upper = pose[8] if spec['dataset'] == 'mmfi' else .5 * (pose[2] + pose[3])
    up = _unit(upper - origin, 'Bend hip-to-torso direction')
    forward = _unit(np.cross(lateral, up), 'Bend forward normal')
    return origin, lateral, up, forward


def target_pose(pose, dose, spec):
    if (isinstance(dose, (bool, np.bool_)) or np.ndim(dose) != 0
            or np.iscomplexobj(dose)):
        raise ValueError('Dose must be a real scalar in [0,1]')
    dose = float(dose)
    if not np.isfinite(dose) or not 0 <= dose <= 1:
        raise ValueError('Dose must be a real scalar in [0,1]')
    if spec not in SPECS.values():
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
    angle = np.deg2rad(spec['theta_max_deg']) * dose
    for item in result.reshape(-1, joints, 3):
        original = item.copy()
        origin, lateral, _, _ = axes(original, spec)
        for joint in targets:
            item[joint] = origin + _rotate(original[joint] - origin, lateral, angle)
    return result
