"""Exploratory Cross/Nod targets for the published PiW fourteen-joint layout.

Only numerical rotation/IK helpers are shared with MM-Fi. No old leg payload,
MM-Fi joint indices, CSI processing or learned model enters this definition.
Virtual shoulder/hip centres define a body frame; they are NOT extra labels.
"""
from __future__ import annotations

import copy
import numpy as np

from attack.action_targets import (MIN_LENGTH, _direction_at, _rotate,
                                   _two_bone_endpoint, _unit)
from piw_a1_adapter import _dose

VERSION = 'piw_action_geometry_v1'
# Neck0/head1; left shoulder2/elbow4/hand8/hip5/knee9/ankle12;
# right shoulder3/elbow6/hand10/hip7/knee11/ankle13. No GCN extra edges.
EDGES = ((0, 1), (0, 2), (2, 4), (4, 8), (2, 5), (5, 9), (9, 12),
         (0, 3), (3, 6), (6, 10), (3, 7), (7, 11), (11, 13))
SPECS = {
    'cross': dict(version=VERSION, kind='bilateral_two_bone_ik_slerp',
        pivots=[3, 2], target_joints=[6, 10, 4, 8], arms=[[3, 6, 10], [2, 4, 8]],
        wrist_across_width=.20, wrist_below_torso=.18,
        wrist_front_width=[.65, .80], wrist_height_separation_torso=.06,
        elbow_outward=.25, elbow_forward=.15, reach_margin=1e-5, forward_sign=1.,
        body_frame='up=shoulder_midpoint-hip_midpoint; lateral=left-right shoulders projected normal to up; forward=cross(lateral,up)',
        note='Two-chain IK, not rigid subtree rotation or action recognition.'),
    'nod': dict(version=VERSION, kind='neck_head_rigid_rotation',
        pivots=[0], target_joints=[1], theta_max_deg=40., forward_sign=1.,
        body_frame='up=shoulder_midpoint-hip_midpoint; lateral=left-right shoulders projected normal to up; forward=cross(lateral,up)',
        note='Head rotates about neck; forward sign must be inspected on TRAIN.'),
}


def specification(action):
    if action not in SPECS:
        raise ValueError('Action must be cross or nod')
    return copy.deepcopy(SPECS[action])


def body_frame(pose, forward_sign=1.):
    if forward_sign not in (-1., 1.):
        raise ValueError('forward_sign must be +1 or -1')
    center = .5 * (pose[2] + pose[3])
    pelvis = .5 * (pose[5] + pose[7])
    up = _unit(center - pelvis, 'PiW hip-to-shoulder midpoint axis')
    shoulder = pose[2] - pose[3]
    lateral = _unit(shoulder - np.dot(shoulder, up) * up, 'PiW shoulder lateral axis')
    return lateral, up, _unit(np.cross(lateral, up), 'PiW forward axis') * forward_sign


def cross_endpoint(pose, spec):
    lateral, up, forward = body_frame(pose, spec['forward_sign'])
    center = .5 * (pose[2] + pose[3])
    width = float(np.linalg.norm(pose[2] - pose[3]))
    torso = float(np.linalg.norm(center - .5 * (pose[5] + pose[7])))
    result = pose.copy()
    for k, (s, e, w) in enumerate(spec['arms']):
        across = 1. if k == 0 else -1.
        below = spec['wrist_below_torso'] + k * spec['wrist_height_separation_torso']
        goal = (center + across * spec['wrist_across_width'] * width * lateral
                - below * torso * up + spec['wrist_front_width'][k] * width * forward)
        pole = -up - across * spec['elbow_outward'] * lateral + spec['elbow_forward'] * forward
        result[e], result[w] = _two_bone_endpoint(
            pose[s], pose[e], pose[w], goal, pole, spec['reach_margin'])
        # IK radially clamps goals to feasible reach. Refuse failed semantics,
        # never silently accept a same-side/behind-chest wrist or drop a frame.
        if across * np.dot(result[w] - center, lateral) <= 0:
            raise ValueError('PiW Cross wrist cannot reach the opposite side')
        if np.dot(result[w] - center, forward) <= 0:
            raise ValueError('PiW Cross wrist is not in front of the chest')
    return result


def target_pose(pose, dose, spec):
    dose = _dose(dose)
    source = np.asarray(pose)
    if (source.ndim < 2 or source.shape[-2:] != (14, 3)
            or source.dtype.kind not in 'fiu' or not np.isfinite(source).all()):
        raise ValueError('Expected finite real PiW (...,14,3) poses in metres')
    if spec not in SPECS.values():
        raise ValueError('Unrecognized PiW payload; version changes must be explicit')
    result = np.array(source, dtype=np.float64, copy=True)
    if dose == 0:
        return result
    for item in result.reshape(-1, 14, 3):
        original = item.copy()
        if spec['kind'] == 'neck_head_rigid_rotation':
            lateral, _, _ = body_frame(original, spec['forward_sign'])
            head = original[1] - original[0]
            if np.linalg.norm(head) < MIN_LENGTH:
                raise ValueError('Degenerate PiW neck-head bone')
            item[1] = original[0] + _rotate(head, lateral,
                np.deg2rad(spec['theta_max_deg']) * dose * spec['forward_sign'])
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
