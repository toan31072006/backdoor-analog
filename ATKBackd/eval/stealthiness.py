"""Fixed clean-trained input detectors for the MM-Fi T4 experiment.

Fit/calibration banks come only from trusted clean TRAIN. The caller evaluates
the frozen scores on paired clean/triggered TEST frames. A larger score always
means more anomalous; neither signs nor thresholds are selected using TEST.
These two modest detectors do not establish physical RF stealthiness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath

import numpy as np


INPUT_STATISTICS = (
    'mean', 'std', 'rms', 'q10', 'median', 'q90',
    'packet_diff_abs', 'packet_diff_rms',
    'subcarrier_diff_abs', 'subcarrier_diff_rms',
    'packet_profile_std', 'subcarrier_profile_std',
)

DETECTOR_RECIPE = {
    'schema': 'mmfi_t4_clean_detector_v1',
    'fitting_data': 'trusted clean TRAIN only; sequence-disjoint fit/calibration',
    'bank_fit_cap': 4096, 'bank_calibration_cap': 2048,
    'calibration_sequence_fraction': .2, 'seed': 42,
    'input_statistics_per_link': list(INPUT_STATISTICS),
    'input_detector': {
        'name': 'IsolationForest', 'n_estimators': 200,
        'max_samples': 'min(256,n_fit)', 'contamination': 'auto',
        'max_features': 1., 'bootstrap': False, 'n_jobs': 1,
        'score': 'negative score_samples; larger=triggered',
    },
    'feature_detector': {
        'standardization': 'StandardScaler fit on clean TRAIN fit bank only',
        'pca_dimension': 'min(64,feature_dimension,n_fit-1)',
        'pca_solver': 'randomized when dimension is reduced; full otherwise',
        'pca_iterated_power': 3, 'pca_whiten': False,
        'covariance': 'LedoitWolf fit on projected clean TRAIN fit bank only',
        'ridge': 'max(1e-6,mean_covariance_diagonal*1e-6)',
        'score': 'squared Mahalanobis distance; larger=triggered',
    },
    'threshold': {
        'requested_clean_calibration_fpr': .05,
        'rule': 'higher empirical .95 quantile of clean calibration scores',
        'decision': 'score > threshold',
    },
    'auroc': 'raw triggered-positive AUROC; no sign inversion selected from TEST',
}


def _real_finite(value, *, name, ndim):
    array = np.asarray(value)
    if (array.ndim != ndim or any(size == 0 for size in array.shape)
            or not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)
            or not np.isfinite(array).all()):
        raise ValueError(f'{name} must be a nonempty finite real {ndim}-D array')
    return array.astype(np.float64, copy=False)


def input_statistics(x):
    """Return 12 fixed statistics per link of ``x[N,link,subcarrier,packet]``.

    Quantiles and moments pool subcarriers/packets *within* each link. Difference
    statistics use the named axis, and profile standard deviations describe
    means along the other axis. No statistics are fit across input samples.
    The model input domain is preserved; there is no per-frame renormalization.
    Output columns are link-major, then :data:`INPUT_STATISTICS` order.
    """
    x = _real_finite(x, name='CSI input', ndim=4)
    if min(x.shape[2:]) < 2:
        raise ValueError('CSI statistics require at least two subcarriers and packets')
    packet_delta = np.diff(x, axis=3)
    subcarrier_delta = np.diff(x, axis=2)
    quantiles = np.quantile(x, [.1, .5, .9], axis=(2, 3))
    features = (
        x.mean(axis=(2, 3)), x.std(axis=(2, 3)),
        np.sqrt(np.mean(x * x, axis=(2, 3))),
        quantiles[0], quantiles[1], quantiles[2],
        np.abs(packet_delta).mean(axis=(2, 3)),
        np.sqrt(np.mean(packet_delta * packet_delta, axis=(2, 3))),
        np.abs(subcarrier_delta).mean(axis=(2, 3)),
        np.sqrt(np.mean(subcarrier_delta * subcarrier_delta, axis=(2, 3))),
        x.mean(axis=2).std(axis=2), x.mean(axis=3).std(axis=2),
    )
    return np.stack(features, axis=2).reshape(x.shape[0], -1)


def _sequence_key(item):
    # Official MM-Fi: E##/S##/A##/wifi-csi/frame###_processed.npy. Retaining
    # environment, subject and action makes IDs portable across mount roots.
    try:
        path = PurePosixPath(str(item['csi']).replace('\\', '/'))
    except (KeyError, TypeError) as exc:
        raise ValueError('each TRAIN item must provide its MM-Fi CSI path') from exc
    if path.parent.name.lower() != 'wifi-csi' or len(path.parent.parent.parts) < 3:
        raise ValueError('TRAIN CSI path must end E/S/A/wifi-csi/frame.npy')
    return '/'.join(path.parent.parent.parts[-3:])


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def bank_split(items, fit_n=4096, cal_n=2048, seed=42, *, source_split='training'):
    """Reserve sequence-disjoint clean TRAIN fit/calibration frame banks.

    Shuffle sorted E/S/A groups with the declared seed; reserve ceil(20%) of
    groups for calibration. Draw frames uniformly without replacement inside
    each partition using separate fixed RNG streams. Returned indices retain
    dataset order and caps never move frames across the sequence boundary.
    Only metadata is read. ``source_split`` rejects accidental TEST use.
    """
    if source_split not in ('train', 'training'):
        raise ValueError('detector fit/calibration banks must use CLEAN TRAIN only')
    fit_n, cal_n = _positive_int(fit_n, 'fit_n'), _positive_int(cal_n, 'cal_n')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    keys = [_sequence_key(item) for item in items]
    groups = sorted(set(keys))
    if len(groups) < 2:
        raise ValueError('at least two TRAIN sequences are needed for disjoint calibration')
    rng = np.random.default_rng(seed)
    group_order = rng.permutation(len(groups))
    n_cal_groups = min(len(groups) - 1, max(1, int(np.ceil(.2 * len(groups)))))
    cal_groups = sorted(groups[int(index)] for index in group_order[:n_cal_groups])
    fit_groups = sorted(groups[int(index)] for index in group_order[n_cal_groups:])
    cal_group_set = set(cal_groups)
    fit_candidates, cal_candidates = [], []
    for index, group in enumerate(keys):
        (cal_candidates if group in cal_group_set else fit_candidates).append(index)

    def sample(candidates, cap, stream):
        sample_rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        indices = sample_rng.choice(candidates, size=min(cap, len(candidates)), replace=False)
        return sorted(int(index) for index in indices)

    fit_indices = sample(fit_candidates, fit_n, 1)
    cal_indices = sample(cal_candidates, cal_n, 2)
    if len(fit_indices) < 3:
        raise ValueError('the clean TRAIN fit bank requires at least three frames')
    digest = hashlib.sha256(json.dumps(
        {'fit': fit_indices, 'cal': cal_indices, 'groups': keys},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(fit_indices=fit_indices, cal_indices=cal_indices,
                fit_groups=fit_groups, cal_groups=cal_groups,
                fit_n=len(fit_indices), cal_n=len(cal_indices), seed=seed,
                source_split='clean_training', group_unit='environment/subject/action',
                calibration_group_fraction=.2, selection_sha256=digest)


class _InputDetector:
    def __init__(self, clean_fit, seed):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(
            n_estimators=200, max_samples=min(256, len(clean_fit)),
            contamination='auto', max_features=1., bootstrap=False,
            random_state=seed, n_jobs=1).fit(clean_fit)
        self.n_features = clean_fit.shape[1]
        self.audit = dict(name='IsolationForest', n_fit=len(clean_fit),
                          input_dimension=self.n_features, seed=seed,
                          score='negative score_samples; larger=triggered',
                          parameters=self.model.get_params())

    def score(self, features):
        features = _real_finite(features, name='input statistics', ndim=2)
        if features.shape[1] != self.n_features:
            raise ValueError('input-statistics feature dimension differs from clean fit bank')
        return -self.model.score_samples(features)


class _FeatureDetector:
    def __init__(self, clean_fit, seed, max_components):
        from sklearn.covariance import LedoitWolf
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        self.n_features = clean_fit.shape[1]
        self.scaler = StandardScaler().fit(clean_fit)
        standardized = self.scaler.transform(clean_fit)
        n_components = min(max_components, self.n_features, len(clean_fit) - 1)
        solver = 'randomized' if n_components < min(standardized.shape) else 'full'
        self.pca = PCA(n_components=n_components, svd_solver=solver,
                       random_state=seed, iterated_power=3)
        self.pca.fit(standardized)
        projected = self.pca.transform(standardized)
        covariance = LedoitWolf(store_precision=False).fit(projected)
        self.location = covariance.location_
        ridge = max(1e-6, float(np.trace(covariance.covariance_) / n_components) * 1e-6)
        self.precision = np.linalg.inv(covariance.covariance_ + ridge * np.eye(n_components))
        self.audit = dict(name='standardized_PCA_LedoitWolf_Mahalanobis',
                          n_fit=len(clean_fit), input_dimension=self.n_features,
                          pca_components=n_components, pca_solver=solver,
                          pca_dimension_cap=max_components, seed=seed,
                          covariance_shrinkage=float(covariance.shrinkage_), ridge=ridge,
                          score='squared Mahalanobis distance; larger=triggered',
                          fitting_data='clean TRAIN fit bank only')

    def score(self, features):
        features = _real_finite(features, name='HPE features', ndim=2)
        if features.shape[1] != self.n_features:
            raise ValueError('HPE feature dimension differs from clean fit bank')
        delta = self.pca.transform(self.scaler.transform(features)) - self.location
        return np.maximum(0., np.einsum('ni,ij,nj->n', delta, self.precision, delta))


def fit_input_detector(clean_fit_statistics, seed=42):
    """Fit a fixed Isolation Forest on clean TRAIN statistics, without labels."""
    clean_fit = _real_finite(clean_fit_statistics, name='clean TRAIN statistics', ndim=2)
    if len(clean_fit) < 3:
        raise ValueError('detector clean fit bank requires at least three frames')
    return _InputDetector(clean_fit, seed)


def fit_feature_detector(clean_fit_features, seed=42, max_components=64):
    """Fit standardization, PCA and regularized covariance on clean TRAIN only."""
    clean_fit = _real_finite(clean_fit_features, name='clean TRAIN HPE features', ndim=2)
    if len(clean_fit) < 3:
        raise ValueError('detector clean fit bank requires at least three frames')
    max_components = _positive_int(max_components, 'max_components')
    return _FeatureDetector(clean_fit, seed, max_components)


def detection_metrics(clean_test_scores, trigger_test_scores, cal_clean_scores, fpr=.05):
    """Report raw AUROC and detection at a clean-calibration threshold.

    All scores already have the locked ``larger=triggered`` direction. The
    threshold is the higher empirical (1-FPR) quantile of CLEAN calibration;
    detection uses ``score > threshold`` (ties are not flagged). Report actual
    TEST FPR because calibration FPR does not guarantee TEST distribution fit.
    No TEST labels enter threshold selection and AUROC is never inverted.
    """
    from sklearn.metrics import roc_auc_score
    clean = _real_finite(clean_test_scores, name='clean TEST scores', ndim=1)
    triggered = _real_finite(trigger_test_scores, name='triggered TEST scores', ndim=1)
    calibration = _real_finite(cal_clean_scores, name='clean calibration scores', ndim=1)
    if clean.shape != triggered.shape:
        raise ValueError('clean/triggered TEST scores must cover identical paired frames')
    if isinstance(fpr, (bool, np.bool_)) or not np.isscalar(fpr) or not np.isfinite(fpr) or not 0 < fpr < 1:
        raise ValueError('calibration FPR must be a finite scalar strictly between 0 and 1')
    threshold = float(np.quantile(calibration, 1 - float(fpr), method='higher'))
    labels = np.r_[np.zeros(len(clean), dtype=np.uint8), np.ones(len(triggered), dtype=np.uint8)]
    return dict(auroc=float(roc_auc_score(labels, np.r_[clean, triggered])),
                tpr=float(np.mean(triggered > threshold)),
                test_fpr=float(np.mean(clean > threshold)), threshold=threshold,
                calibration_fpr=float(np.mean(calibration > threshold)),
                requested_calibration_fpr=float(fpr),
                n_clean_test=len(clean), n_trigger_test=len(triggered),
                n_clean_calibration=len(calibration), score_direction='larger=triggered',
                threshold_rule='clean calibration higher quantile; score > threshold')
