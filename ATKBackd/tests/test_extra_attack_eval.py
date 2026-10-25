"""Paired target metrics for the separate TSBA/LF exploratory evaluator."""

import numpy as np
import pytest

from attack.payload import set_skeleton_config
from extra_attack_eval import _replay_mm, measure_pair


def test_paired_target_pck_and_dce_use_clean_gt_scale():
    set_skeleton_config('mmfi')
    true = np.zeros((2, 17, 3), dtype=np.float32)
    true[:, 12, 0] = 1.0              # each clean GT reference scale = 1 metre
    clean = true.copy()
    target = true.copy()
    target[:, [2, 3], 0] = 0.5
    triggered = target.copy()
    triggered[1, 3, 0] = 0.8           # one of four target joints misses @.2

    measured, sample = measure_pair(clean, triggered, true, target)

    assert measured['target_pck_pct'] == 75.0
    assert measured['no_trigger_target_pck_pct'] == 0.0
    assert measured['t_mpjpe_mm'] == pytest.approx(75.0)
    assert measured['no_trigger_t_mpjpe_mm'] == pytest.approx(500.0)
    assert measured['dce_mm'] == pytest.approx(75.0)
    assert sample['target_error_mm'].shape == (2, 2)
    assert np.all(sample['clean_scale_mm'] == 1000.0)


def test_unchanged_prediction_has_nonzero_dce_for_positive_payload():
    set_skeleton_config('mmfi')
    true = np.zeros((1, 17, 3), dtype=np.float32)
    true[:, 12, 0] = 1.0
    target = true.copy()
    target[:, [2, 3], 0] = 0.4
    measured, _ = measure_pair(true, true, true, target)
    assert measured['dce_mm'] == pytest.approx(400.0)
    assert measured['target_pck_pct'] == 0.0


def test_replay_gate_rejects_cache_drift():
    assert _replay_mm(100.03, 0.1, 'T-MPJPE') == pytest.approx(0.03)
    with pytest.raises(ValueError, match='checkpoint replay differs'):
        _replay_mm(100.06, 0.1, 'T-MPJPE')
