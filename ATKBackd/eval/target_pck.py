"""Target-joint PCK for the MM-Fi dose-controlled pose attack.

This is an attack metric: predictions are compared with the *attacker target*,
while the normalization distance is taken from each sample's clean true pose.
It is deliberately separate from clean-pose PCK in :mod:`eval.metrics`.
"""

import numpy as np


def target_pck(pred, target, true_pose, target_joints=(2, 3), threshold=0.2):
    """Return the fraction of target joints reached within ``threshold`` scale.

    ``pred``, ``target`` and ``true_pose`` have shape ``(N, 17, 3)``. For sample
    ``i``, the fixed scale is ``||true_pose[i, 5] - true_pose[i, 12]||``. A joint
    counts as reached when its Euclidean distance to the attacker target is
    *at most* ``threshold * scale``. No pose alignment is applied.

    The result is in ``[0, 1]``; multiply by 100 only when reporting a percent.
    """
    poses = []
    for name, value in (('pred', pred), ('target', target), ('true_pose', true_pose)):
        pose = np.asarray(value)
        if pose.ndim != 3 or pose.shape[0] == 0 or pose.shape[1:] != (17, 3):
            raise ValueError(f'{name} must have a nonempty (N, 17, 3) shape')
        if (not np.issubdtype(pose.dtype, np.number)
                or np.issubdtype(pose.dtype, np.complexfloating)):
            raise ValueError(f'{name} must contain real numeric coordinates')
        if not np.isfinite(pose).all():
            raise ValueError(f'{name} must contain finite coordinates')
        poses.append(pose.astype(np.float64, copy=False))

    prediction, attacker_target, clean_truth = poses
    if prediction.shape != attacker_target.shape or prediction.shape != clean_truth.shape:
        raise ValueError('pred, target and true_pose must have matching shapes')

    try:
        joints = tuple(target_joints)
    except TypeError as exc:
        raise ValueError('target_joints must be a nonempty sequence of joint indices') from exc
    if (not joints or any(isinstance(j, (bool, np.bool_))
                          or not isinstance(j, (int, np.integer))
                          or not 0 <= int(j) < 17 for j in joints)):
        raise ValueError('target_joints must contain unique valid joint indices')
    joints = tuple(int(j) for j in joints)
    if len(set(joints)) != len(joints):
        raise ValueError('target_joints must contain unique valid joint indices')

    if isinstance(threshold, (bool, np.bool_)) or not np.isscalar(threshold):
        raise ValueError('threshold must be a finite nonnegative scalar')
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('threshold must be a finite nonnegative scalar') from exc
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be a finite nonnegative scalar')

    scale = np.linalg.norm(clean_truth[:, 5] - clean_truth[:, 12], axis=-1)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('true_pose reference distance (joints 5, 12) must be positive')
    distance = np.linalg.norm(prediction[:, joints] - attacker_target[:, joints], axis=-1)
    return float(np.mean(distance <= threshold * scale[:, None]))
