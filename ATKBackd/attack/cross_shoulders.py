"""Cross v3: six allowed joints, fixed-anchor shoulder orbits and two-bone IK.

Numerical label construction only: no victim, CSI, scores, randomness or fitted
parameters. PiW shoulders keep BOTH neck and hip distances. All body-frame
crossing/front checks refer to the CLEAN torso, never the moved shoulders.
"""
import numpy as np

from attack.action_targets import (MIN_LENGTH, _unit, _rotate, _direction_at,
    _two_bone_endpoint, _reachable_cross_goal, _cross_arm_record)


def orbit_for(pose, shoulder, anchors, up, forward, inward):
    origin = pose[shoulder]
    anchor = pose[anchors[0]]
    axis = up
    if len(anchors) == 2:
        line = pose[anchors[1]] - anchor
        if np.linalg.norm(line) >= MIN_LENGTH:
            axis = _unit(line, 'neck-hip shoulder constraint axis')
    if any(np.linalg.norm(origin-pose[a]) < MIN_LENGTH for a in anchors):
        raise ValueError('Degenerate shoulder-anchor bone; no bone repair')
    center = anchor + np.dot(origin-anchor, axis)*axis
    radial = origin-center
    if np.linalg.norm(radial) < MIN_LENGTH:
        # Tangent anchor spheres can pin a valid shoulder. Do not fabricate
        # extra mobility or stretch its constraints; the arm may still cross.
        angle = 0.
    else:
        desired = forward-np.dot(forward, axis)*axis
        if np.linalg.norm(desired) < MIN_LENGTH:
            desired = inward-np.dot(inward, axis)*axis
        desired = _unit(desired, 'shoulder orbit destination')
        first = radial / np.linalg.norm(radial)
        sine = float(np.dot(axis, np.cross(first, desired)))
        cosine = float(np.clip(np.dot(first, desired), -1., 1.))
        angle = float(np.arctan2(sine, cosine))
        if abs(sine) < 1e-12 and cosine < 0:
            # Antipodal tie uses a body-frame scalar, not a world-axis choice.
            angle = np.pi if np.dot(np.cross(axis, radial), inward) >= 0 else -np.pi
    return dict(origin=origin.copy(), center=center, radial=radial, axis=axis, angle=angle)


def orbit_position(motion, fraction):
    if fraction == 0 or motion['angle'] == 0:
        return motion['origin'].copy()
    return motion['center'] + _rotate(motion['radial'], motion['axis'], motion['angle']*fraction)


def _candidate_fractions(motion, center, inward, far, spec):
    """Fixed nominal candidates, then analytic lateral extrema/root witnesses.

    The extra witnesses prevent a coarse angle grid missing a narrow reachable
    interval. This is a prescribed IK solver, not a TEST/metric-based search.
    Failure means no valid candidate under THIS orbit policy, not a proof
    about all physically possible shoulder motions.
    """
    seen = set()
    for fraction in spec['shoulder_candidate_fractions']:
        seen.add(round(fraction, 14))
        yield fraction
    theta = motion['angle']
    if abs(theta) < 1e-12:
        return
    radial, axis = motion['radial'], motion['axis']
    b = -float(np.dot(radial, inward))
    c = -float(np.dot(np.cross(axis, radial), inward))
    critical = [0., 1.]
    phase = float(np.arctan2(c, b))
    for k in range(-2, 3):
        fraction = (phase+k*np.pi)/theta
        if 0 < fraction < 1:
            critical.append(fraction)
    critical.sort()
    h = lambda f: float(np.dot(center-orbit_position(motion, f), inward))
    initial_h = h(0.)
    wanted_h = min(.5*initial_h, .5*far)
    extra = list(critical[1:])
    if wanted_h > MIN_LENGTH:
        for left, right in zip(critical[:-1], critical[1:]):
            fl, fr = h(left)-wanted_h, h(right)-wanted_h
            if fl*fr > 0:
                continue
            for _ in range(spec['shoulder_root_steps']):
                middle = .5*(left+right)
                fm = h(middle)-wanted_h
                if fl*fm <= 0:
                    right, fr = middle, fm
                else:
                    left, fl = middle, fm
            extra.append(.5*(left+right))
    for fraction in sorted(extra):
        key = round(fraction, 14)
        if fraction > 1e-12 and key not in seen:
            seen.add(key)
            yield fraction


def endpoint(pose, spec, frame, torso, *, diagnostics=None, shoulder_motions=None):
    lateral, up, forward = frame
    shoulders = [arm[0] for arm in spec['arms']]
    center = .5*(pose[shoulders[0]]+pose[shoulders[1]])
    width = float(np.linalg.norm(pose[shoulders[1]]-pose[shoulders[0]]))
    result = pose.copy()
    for k, (s, e, w) in enumerate(spec['arms']):
        sign = 1. if k == 0 else -1.
        inward = sign*lateral
        goal = (center+spec['wrist_across_width']*width*inward
            -(spec['wrist_below_torso']+k*spec['wrist_height_separation_torso'])*torso*up
            +spec['wrist_front_width'][k]*width*forward)
        pole = -up-spec['elbow_outward']*inward+spec['elbow_forward']*forward
        upper, lower = pose[e]-pose[s], pose[w]-pose[e]
        upper_length, lower_length = np.linalg.norm(upper), np.linalg.norm(lower)
        if min(upper_length, lower_length) < MIN_LENGTH:
            raise ValueError('Degenerate arm bone; no bone repair')
        far = float(upper_length+lower_length-spec['reach_margin'])
        motion = orbit_for(pose, s, spec['shoulder_anchors'][k], up, forward, inward)
        failures, tested, best_h = [], 0, None
        for fraction in _candidate_fractions(motion, center, inward, far, spec):
            tested += 1
            shoulder = orbit_position(motion, fraction)
            h = float(np.dot(center-shoulder, inward))
            best_h = h if best_h is None else min(best_h, h)
            try:
                if h <= MIN_LENGTH:
                    raise ValueError('Shoulder would cross the clean torso mid-plane')
                fitted = _reachable_cross_goal(shoulder, shoulder+upper, shoulder+upper+lower,
                    goal, center, lateral, forward, sign, spec)
                elbow, wrist = _two_bone_endpoint(shoulder, shoulder+upper, shoulder+upper+lower,
                    fitted, pole, spec['reach_margin'])
                if np.dot(wrist-center, inward) <= 0 or np.dot(wrist-center, forward) <= 0:
                    raise ValueError('Post-IK wrist violates clean-torso crossing/front semantics')
            except ValueError as exc:
                failures.append(str(exc))
                continue
            result[s], result[e], result[w] = shoulder, elbow, wrist
            selected = dict(motion, angle=motion['angle']*fraction)
            if shoulder_motions is not None:
                shoulder_motions.append(selected)
            if diagnostics is not None:
                record = _cross_arm_record(pose[s], pose[e], pose[w], goal, fitted,
                    wrist, center, lateral, forward, sign)
                record.update(shoulder_displacement_m=float(np.linalg.norm(shoulder-pose[s])),
                    shoulder_rotation_deg=float(np.rad2deg(abs(selected['angle']))),
                    shoulder_fraction=float(fraction), candidates_tested=tested,
                    old_fixed_shoulder_non_crossable=far <= float(np.dot(center-pose[s], inward)))
                diagnostics.append(record)
            break
        else:
            raise ValueError(f'Cross v3 arm {s}/{e}/{w}: no valid shoulder-orbit candidate; '
                f'upper={upper_length:.9g}m lower={lower_length:.9g}m '
                f'min_candidate_mid_plane_distance={best_h}m candidates={tested}; '
                f'last_reason={failures[-1] if failures else "none"}; no sample dropped or bone stretched')
    return result


def interpolate(pose, target, motions, dose, spec):
    result = pose.copy()
    for (s, e, w), motion in zip(spec['arms'], motions):
        result[s] = orbit_position(motion, dose)
        upper, lower = pose[e]-pose[s], pose[w]-pose[e]
        result[e] = result[s]+np.linalg.norm(upper)*_direction_at(upper, target[e]-target[s], dose)
        result[w] = result[e]+np.linalg.norm(lower)*_direction_at(lower, target[w]-target[e], dose)
    return result
