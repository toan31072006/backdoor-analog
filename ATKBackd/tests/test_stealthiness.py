"""T4 detectors use clean TRAIN for fitting and clean calibration for thresholds."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.stealthiness import (INPUT_STATISTICS, bank_split, detection_metrics,
                               fit_feature_detector, fit_input_detector,
                               input_statistics)


def _items(n_groups=10, frames=20, root='/data'):
    return [dict(csi=f'{root}/E01/S{group:02d}/A01/wifi-csi/frame{frame:03d}_processed.npy')
            for group in range(n_groups) for frame in range(frames)]


def test_bank_split_sequence_disjoint_reproducible_and_capped():
    items = _items()
    result = bank_split(items, fit_n=45, cal_n=25)
    assert result == bank_split(items, fit_n=45, cal_n=25)
    assert result['fit_n'] == 45 and result['cal_n'] == 25
    assert not set(result['fit_groups']) & set(result['cal_groups'])
    assert not set(result['fit_indices']) & set(result['cal_indices'])
    assert len(result['fit_groups']) == 8 and len(result['cal_groups']) == 2
    # Full sequence assignment, not just sampled frame indices, is disjoint.
    for key, groups in [('fit_indices', 'fit_groups'), ('cal_indices', 'cal_groups')]:
        for index in result[key]:
            parts = items[index]['csi'].split('/')
            assert '/'.join(parts[-5:-2]) in result[groups]
    assert result == bank_split(_items(root='D:\\MMFI'), fit_n=45, cal_n=25)
    assert result != bank_split(items, fit_n=45, cal_n=25, seed=43)


def test_bank_split_never_uses_test_and_requires_sequence_metadata():
    with pytest.raises(ValueError, match='CLEAN TRAIN'):
        bank_split(_items(), source_split='test')
    with pytest.raises(ValueError, match='at least two TRAIN sequences'):
        bank_split(_items(n_groups=1))
    with pytest.raises(ValueError, match='TRAIN CSI path'):
        bank_split([dict(csi='/data/frame.npy')] * 3)
    with pytest.raises(ValueError, match='positive integer'):
        bank_split(_items(), fit_n=True)


def test_input_statistics_axis_order_and_known_constant_link_values():
    x = np.ones((2, 3, 4, 5)) * np.arange(1, 4)[None, :, None, None]
    features = input_statistics(x).reshape(2, 3, len(INPUT_STATISTICS))
    for link in range(3):
        assert features[0, link, 0] == link + 1
        assert features[0, link, 2] == link + 1
        assert np.all(features[:, link, 6:] == 0)
    x = np.broadcast_to(np.arange(5)[None, None, None, :], (1, 1, 4, 5))
    feat = dict(zip(INPUT_STATISTICS, input_statistics(x)[0]))
    assert feat['packet_diff_abs'] == 1 and feat['packet_diff_rms'] == 1
    assert feat['subcarrier_diff_abs'] == 0
    assert feat['subcarrier_profile_std'] == 0
    assert feat['packet_profile_std'] > 0


def test_input_statistics_is_samplewise_and_has_no_test_fit():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4, 3, 8, 5))
    assert input_statistics(x).shape == (4, 36)
    np.testing.assert_array_equal(input_statistics(x[:1]), input_statistics(x)[:1])
    with pytest.raises(ValueError, match='finite real'):
        input_statistics(x.astype(complex))
    with pytest.raises(ValueError, match='two subcarriers'):
        input_statistics(x[:, :, :1])


def test_calibration_threshold_unchanged_when_test_scores_change():
    calibration = np.linspace(0, 1, 100)
    result = detection_metrics(np.zeros(20), np.ones(20) * 5, calibration)
    shifted = detection_metrics(np.ones(20) * 100, np.ones(20) * 200, calibration)
    assert result['threshold'] == shifted['threshold']
    assert result['calibration_fpr'] <= .05
    assert result['auroc'] == 1 and result['tpr'] == 1 and result['test_fpr'] == 0
    assert shifted['test_fpr'] == 1


def test_detection_never_flips_bad_direction_and_uses_strict_ties():
    result = detection_metrics(np.ones(20), np.zeros(20), np.zeros(100))
    assert result['auroc'] == 0 and result['tpr'] == 0 and result['test_fpr'] == 1
    assert result['calibration_fpr'] == 0
    equal = detection_metrics(np.ones(20), np.ones(20), np.ones(100))
    assert equal['auroc'] == .5 and equal['tpr'] == 0
    with pytest.raises(ValueError, match='identical paired'):
        detection_metrics(np.ones(5), np.ones(6), np.ones(10))


@pytest.mark.parametrize('fpr', [0, 1, True, float('nan')])
def test_detection_rejects_invalid_fpr(fpr):
    with pytest.raises(ValueError, match='FPR'):
        detection_metrics(np.zeros(2), np.ones(2), np.arange(50), fpr=fpr)


def test_isolation_forest_clean_fit_is_fixed_and_high_means_anomaly():
    rng = np.random.default_rng(42)
    clean = rng.normal(size=(200, 6))
    detector = fit_input_detector(clean)
    repeat = fit_input_detector(clean)
    np.testing.assert_array_equal(detector.score(clean), repeat.score(clean))
    before = detector.score(clean)
    assert detector.score(np.ones((20, 6)) * 40).mean() > before.mean()
    np.testing.assert_array_equal(detector.score(clean), before)
    assert detector.audit['n_fit'] == 200
    assert detector.audit['parameters']['n_estimators'] == 200
    with pytest.raises(ValueError, match='dimension'):
        detector.score(np.ones((2, 4)))


def test_feature_detector_pca_and_covariance_fit_only_clean_bank():
    rng = np.random.default_rng(42)
    clean = rng.normal(size=(150, 80))
    detector = fit_feature_detector(clean)
    assert detector.audit['pca_components'] == 64
    np.testing.assert_allclose(detector.scaler.mean_, clean.mean(axis=0))
    initial_mean = detector.scaler.mean_.copy()
    initial_cov = detector.precision.copy()
    before = detector.score(clean)
    assert detector.score(np.ones((30, 80)) * 50).mean() > before.mean()
    np.testing.assert_array_equal(detector.scaler.mean_, initial_mean)
    np.testing.assert_array_equal(detector.precision, initial_cov)
    np.testing.assert_array_equal(detector.score(clean), before)
    assert detector.audit['ridge'] > 0
    with pytest.raises(ValueError, match='dimension'):
        detector.score(np.ones((2, 4)))


def test_feature_detector_regularizes_degenerate_fit():
    # No singular inverse even if most dimensions are constant/duplicated.
    x = np.repeat(np.arange(8)[:, None], 70, axis=1).astype(float)
    detector = fit_feature_detector(x)
    assert np.isfinite(detector.score(x)).all()
    assert detector.audit['pca_components'] == 7
    with pytest.raises(ValueError, match='at least three'):
        fit_feature_detector(x[:2])
