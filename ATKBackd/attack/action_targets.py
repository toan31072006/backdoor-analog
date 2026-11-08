"""Exploratory MM-Fi action payloads, not historical motion-prior ablations.

Nod rotates the head about the neck by 40*d degrees around the body lateral
axis. Cross uses bilateral two-bone IK for a prescribed crossed-wrist endpoint,
then interpolates upper/lower bone directions separately along great circles.
Cross v3 permits shoulder motion on fixed-anchor orbits as well as arm IK.
It is NOT the manuscript's single rigid subtree rotation; its joint angles
change and monotone wrist displacement is not a mathematical guarantee.
No CSI, victim output, TEST metric, or learned parameter enters either map.
"""
from __future__ import annotations

import copy
import numpy as np


VERSION = 'mmfi_action_geometry_v1'
SPECS = {
    'cross': dict(version='mmfi_action_cross_geometry_v3', kind='bilateral_shoulder_orbit_two_bone_ik_slerp',
        pivots=[8], target_joints=[11, 12, 13, 14, 15, 16],
        arms=[[11, 12, 13], [14, 15, 16]],
        shoulder_anchors=[[8], [8]],
        shoulder_candidate_fractions=[.2, .3, .4, .5, .6, .7, .8, .9, 1.],
        shoulder_root_steps=60,
        shoulder_policy='forward_orbit_fixed_anchors_clean_torso_v3',
        wrist_across_width=.20, wrist_below_torso=.18,
        wrist_front_width=[.65, .80], wrist_height_separation_torso=.06,
        elbow_outward=.25, elbow_forward=.15, reach_margin=1e-5,
        wrist_reach_policy='crossing_plane_then_perpendicular_radius_v2',
        wrist_cross_slack_fraction=.5,
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


def _reachable_cross_goal(shoulder, elbow, wrist, goal, center, lateral,
                          forward, across_sign, spec):
    """Fit a chest goal to this arm WITHOUT a same-side radial-clamp fallback.

    Already reachable goals are unchanged. Otherwise h is the shoulder's
    distance to the torso mid-plane and R is its maximum safe arm reach.
    Reserve min(preferred crossing, (R-h)/2) beyond that plane. At this fixed
    lateral coordinate, clamp only the perpendicular (height/front) radius
    to the two-bone reachable annulus. Its direction, hence forward sign, is
    preserved. No bone, shoulder, sample, dose or learned parameter is changed.
    This is an explicit v2 target mapping, not nearest-point projection or
    anatomical validation. A genuinely non-crossable arm still fails.
    """
    upper = float(np.linalg.norm(elbow - shoulder))
    lower = float(np.linalg.norm(wrist - elbow))
    near, far = abs(upper - lower) + spec['reach_margin'], upper + lower - spec['reach_margin']
    if min(upper, lower) < MIN_LENGTH or near >= far:
        raise ValueError('Degenerate arm or unsatisfiable IK margin')
    across_axis = across_sign * lateral
    h = float(np.dot(center - shoulder, across_axis))
    preferred = float(np.dot(goal - center, across_axis))
    if h <= 0 or preferred <= 0 or np.dot(goal - center, forward) <= 0:
        raise ValueError('Cross requires opposite-side shoulders and a front-chest goal')
    delta = goal - shoulder
    distance = float(np.linalg.norm(delta))
    if near <= distance <= far:
        return goal.copy()
    slack = far - h
    if slack <= 4 * MIN_LENGTH:
        raise ValueError(f'Cross arm cannot reach across the mid-plane: '
                         f'upper={upper:.9g}m lower={lower:.9g}m '
                         f'max_safe_reach={far:.9g}m mid_plane_distance={h:.9g}m; '
                         'no sample skipped or bone stretched')
    across = min(preferred, spec['wrist_cross_slack_fraction'] * slack)
    along = h + across
    perpendicular = delta - np.dot(delta, across_axis) * across_axis
    radius = float(np.linalg.norm(perpendicular))
    direction = _unit(perpendicular, 'front-chest perpendicular goal')
    # Factored differences avoid cancellation when reach is nearly exhausted.
    low = np.sqrt(max(0., (near - along) * (near + along)))
    high = np.sqrt(max(0., (far - along) * (far + along)))
    fitted = shoulder + along * across_axis + float(np.clip(radius, low, high)) * direction
    if (np.dot(fitted - center, across_axis) <= MIN_LENGTH
            or np.dot(fitted - center, forward) <= MIN_LENGTH):
        raise ValueError('Cross has insufficient strict crossing/front clearance; no fallback')
    return fitted


def _cross_arm_record(shoulder, elbow, wrist, nominal, fitted, result, center,
                      lateral, forward, across_sign):
    """TRAIN geometry diagnostics only; never used to select or discard a pose."""
    correction = float(np.linalg.norm(fitted - nominal))
    return dict(adapted=correction > 1e-12, goal_correction_m=correction,
        upper_m=float(np.linalg.norm(elbow - shoulder)),
        lower_m=float(np.linalg.norm(wrist - elbow)),
        nominal_distance_m=float(np.linalg.norm(nominal - shoulder)),
        opposite_side_m=float(across_sign * np.dot(result - center, lateral)),
        front_m=float(np.dot(result - center, forward)))


def cross_endpoint(pose, spec, *, diagnostics=None, shoulder_motions=None):
    from attack.cross_shoulders import endpoint
    torso = float(np.linalg.norm(pose[8] - pose[0]))
    return endpoint(pose, spec, body_frame(pose, spec['forward_sign']), torso,
                    diagnostics=diagnostics, shoulder_motions=shoulder_motions)


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
            from attack.cross_shoulders import interpolate
            motions = []
            endpoint = cross_endpoint(original, spec, shoulder_motions=motions)
            if dose == 1:
                item[:] = endpoint
            else:
                item[:] = interpolate(original, endpoint, motions, dose, spec)
    return result
