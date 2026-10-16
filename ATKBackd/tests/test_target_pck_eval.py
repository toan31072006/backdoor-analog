"""Target-PCK is an attack metric, not clean-pose PCK or pose-level ASR."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.target_pck import target_pck  # noqa: E402


def _poses(n=2):
    true = np.zeros((n, 17, 3), dtype=np.float64)
    true[:, 5, 0] = 1.0
    target = true.copy()
    target[:, (2, 3), 0] = 0.5
    return true, target


def test_scores_attacker_target_not_clean_pose():
    true, target = _poses()
    assert target_pck(target, target, true) == 1.0
    assert target_pck(true, target, true) == 0.0


def test_scale_is_from_each_samples_clean_true_pose():
    true, target = _poses()
    true[1, 5, 0] = 2.0
    prediction = target.copy()
    prediction[:, (2, 3), 0] += 0.3
    assert target_pck(prediction, target, true) == 0.5


def test_only_selected_target_joints_count():
    true, target = _poses()
    prediction = target.copy()
    prediction[:, 0, :] = 1000.0
    prediction[:, 4:17, 1] = 1000.0
    assert target_pck(prediction, target, true) == 1.0
    prediction[:, 2, 0] += 1.0
    assert target_pck(prediction, target, true) == 0.5
    assert target_pck(prediction, target, true, target_joints=(3,)) == 1.0


def test_boundary_ties_are_hits_and_zero_threshold_is_exact_match():
    true, target = _poses(1)
    prediction = target.copy()
    prediction[0, 2, 0] += 0.2
    assert target_pck(prediction, target, true) == 1.0
    assert target_pck(prediction, target, true, threshold=0.0) == 0.5


@pytest.mark.parametrize('bad', [
    np.zeros((0, 17, 3)), np.zeros((1, 16, 3)), np.zeros((1, 17, 2)),
    np.zeros((1, 1, 17, 3)), np.full((2, 17, 3), np.nan),
    np.full((2, 17, 3), np.inf), np.zeros((2, 17, 3), dtype=complex),
    np.full((2, 17, 3), 'invalid'),
])
def test_rejects_invalid_poses(bad):
    true, target = _poses()
    with pytest.raises(ValueError):
        target_pck(bad, target, true)
    with pytest.raises(ValueError):
        target_pck(target, bad, true)
    with pytest.raises(ValueError):
        target_pck(target, target, bad)


def test_rejects_shape_mismatch_and_nonpositive_reference_scale():
    true, target = _poses()
    with pytest.raises(ValueError, match='matching shapes'):
        target_pck(target[:1], target, true)
    true[0, 5] = true[0, 12]
    with pytest.raises(ValueError, match='must be positive'):
        target_pck(target, target, true)


@pytest.mark.parametrize('joints', [(), (2, 2), (-1,), (17,), (True,), 2, (2.0,)])
def test_rejects_invalid_target_joint_sets(joints):
    true, target = _poses()
    with pytest.raises(ValueError, match='target_joints'):
        target_pck(target, target, true, target_joints=joints)


@pytest.mark.parametrize('threshold', [-0.1, float('nan'), float('inf'), True, 1j])
def test_rejects_invalid_threshold(threshold):
    true, target = _poses()
    with pytest.raises(ValueError, match='threshold'):
        target_pck(target, target, true, threshold=threshold)
