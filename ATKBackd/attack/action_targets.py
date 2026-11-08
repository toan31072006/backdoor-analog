"""Exploratory MM-Fi action payloads, not historical motion-prior ablations.

Nod rotates the head about the neck by 40*d degrees around the body lateral
axis. Cross uses bilateral two-bone IK for a prescribed crossed-wrist endpoint,
then interpolates upper/lower bone directions separately along great circles.
Cross is NOT the manuscript's single rigid subtree rotation; its joint angles
change and monotone wrist displacement is not a mathematical guarantee.
No CSI, victim output, TEST metric, or learned parameter enters either map.
"""
from __future__ import annotations

import copy
import numpy as np


VERSION = 'mmfi_action_geometry_v1'
SPECS = {
    'cross': dict(version=VERSION, kind='bilateral_two_bone_ik_slerp',
        pivots=[11, 14], target_joints=[12, 13, 15, 16],
        arms=[[11, 12, 13], [14, 15, 16]],
        wrist_across_width=.20, wrist_below_torso=.18,
        wrist_front_width=[.65, .80], wrist_height_separation_torso=.06,
        elbow_outward=.25, elbow_forward=.15, reach_margin=1e-5,
        forward_sign=1.,
        note='New multi-chain geometric target, not rigid-rotation theorem or action recognition.'),
    'nod': dict(version=VERSION, kind='neck_head_rigid_rotation',
        pivots=[9], target_joints=[10], theta_max_deg=40., forward_sign=1.,
        note='Head tilts toward the body-frame forward direction; neck stays fixed.'),
}
MIN_LENGTH = 1e-6  # metres; reject malformed/degenerate poses, never rescale them


def specification(action):
    if action not in SPECS:
        raise ValueError('Action must be cross or nod')
    return copy.deepcopy(SPECS[action])


def _unit(vector, name):
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < MIN_LENGTH:
        raise ValueError(f'Degenerate {name}; inspect original pose, do not skip or repair silently')
    return vector / length


def body_frame(pose, forward_sign=1.):
    """H36M/MM-Fi indices: pelvis0, thorax8, shoulders11/14, neck9, head10.

    The sign of cross(lateral,up) is an explicit geometry convention. It must
    be inspected on TRAIN before training; it is not inferred from TEST scores.
    """
    if forward_sign not in (-1., 1.):
        raise ValueError('forward_sign must be +1 or -1')
    up = _unit(pose[8] - pose[0], 'pelvis-to-thorax axis')
    shoulder = pose[14] - pose[11]
    lateral = _unit(shoulder - np.dot(shoulder, up) * up, 'shoulder lateral axis')
    forward = _unit(np.cross(lateral, up), 'body forward axis') * forward_sign
    return lateral, up, forward


def _rotate(vector, axis, angle):
    return (vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1 - np.cos(angle)))


def _direction_at(first, last, dose):
    first, last = _unit(first, 'initial bone'), _unit(last, 'endpoint bone')
    cosine = float(np.clip(np.dot(first, last), -1., 1.))
    cross = np.cross(first, last)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-10:
        if cosine > 0:
            return first.copy()
        # Deterministic antipodal route; no random or victim-dependent choice.
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        axis = _unit(np.cross(first, basis), 'antipodal interpolation axis')
    else:
        axis = cross / sine
    return _rotate(first, axis, np.arctan2(sine, cosine) * dose)


def _two_bone_endpoint(shoulder, elbow, wrist, goal, pole, margin):
    upper = float(np.linalg.norm(elbow - shoulder))
    lower = float(np.linalg.norm(wrist - elbow))
    if min(upper, lower) < MIN_LENGTH:
        raise ValueError('Degenerate arm bone')
    direction = _unit(goal - shoulder, 'wrist-goal direction')
    near, far = abs(upper - lower) + margin, upper + lower - margin
    if near >= far:
        raise ValueError('Arm lengths cannot satisfy the prescribed IK margin')
    distance = float(np.clip(np.linalg.norm(goal - shoulder), near, far))
    along = (upper**2 - lower**2 + distance**2) / (2 * distance)
    height = np.sqrt(max(0., upper**2 - along**2))
    normal = pole - np.dot(pole, direction) * direction
    if np.linalg.norm(normal) < MIN_LENGTH:
        basis = np.eye(3)[int(np.argmin(np.abs(direction)))]
        normal = basis - np.dot(basis, direction) * direction
    perpendicular = _unit(normal, 'elbow pole')
    return (shoulder + along * direction + height * perpendicular,
            shoulder + distance * direction)


def cross_endpoint(pose, spec):
    lateral, up, forward = body_frame(pose, spec['forward_sign'])
    center = .5 * (pose[11] + pose[14])
    width = float(np.linalg.norm(pose[14] - pose[11]))
    torso = float(np.linalg.norm(pose[8] - pose[0]))
    result = pose.copy()
    for k, (s, e, w) in enumerate(spec['arms']):
        across_sign = 1. if k == 0 else -1.
        below = spec['wrist_below_torso'] + k * spec['wrist_height_separation_torso']
        goal = (center + across_sign * spec['wrist_across_width'] * width * lateral
                - below * torso * up + spec['wrist_front_width'][k] * width * forward)
        pole = (-up - across_sign * spec['elbow_outward'] * lateral
                + spec['elbow_forward'] * forward)
        result[e], result[w] = _two_bone_endpoint(
            pose[s], pose[e], pose[w], goal, pole, spec['reach_margin'])
        # Clamping a physically unreachable goal must not turn Cross into a
        # same-side arm pose without an explicit failure visible to the user.
        if across_sign * np.dot(result[w] - center, lateral) <= 0:
            raise ValueError('Cross wrist cannot reach the opposite side with these bone lengths')
        if np.dot(result[w] - center, forward) <= 0:
            raise ValueError('Cross wrist is not in front of the chest')
    return result


def target_pose(pose, dose, spec):
    """Return a new float64 array of shape (...,17,3); dose is scalar in [0,1]."""
    if isinstance(dose, (bool, np.bool_)) or not np.isfinite(dose) or not 0 <= dose <= 1:
        raise ValueError('dose must be a finite scalar in [0,1]')
    source = np.asarray(pose)
    if (source.shape[-2:] != (17, 3) or source.dtype.kind not in 'fiu'
            or not np.isfinite(source).all()):
        raise ValueError('Expected finite MM-Fi poses (...,17,3) in metres')
    if spec not in SPECS.values():
        raise ValueError('Unrecognized payload specification; use a new version for changes')
    result = np.array(source, dtype=np.float64, copy=True)
    if dose == 0:
        return result
    for item in result.reshape(-1, 17, 3):
        original = item.copy()
        if spec['kind'] == 'neck_head_rigid_rotation':
            lateral, _, _ = body_frame(original, spec['forward_sign'])
            # Forward-sign changes reverse the nod angle, not the lateral axis.
            angle = np.deg2rad(spec['theta_max_deg']) * dose * spec['forward_sign']
            item[10] = original[9] + _rotate(original[10] - original[9], lateral, angle)
        else:
            endpoint = cross_endpoint(original, spec)
            if dose == 1:
                item[:] = endpoint
            else:
                for s, e, w in spec['arms']:
                    upper, lower = original[e] - original[s], original[w] - original[e]
                    item[e] = original[s] + np.linalg.norm(upper) * _direction_at(
                        upper, endpoint[e] - endpoint[s], dose)
                    item[w] = item[e] + np.linalg.norm(lower) * _direction_at(
                        lower, endpoint[w] - endpoint[e], dose)
    return result
