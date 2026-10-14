"""Opt-in payload-change calibration; legacy metrics remain unchanged."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack import payload  # noqa: E402
from eval.metrics import dose_control_error  # noqa: E402
import train_backdoor as trainer  # noqa: E402


@pytest.fixture(autouse=True)
def mmfi_skeleton():
    previous = 'mmfi' if payload.N_JOINTS == 17 else 'person-in-wifi-3d'
    payload.set_skeleton_config('mmfi')
    yield
    payload.set_skeleton_config(previous)


def _poses():
    rng = np.random.default_rng(17)
    clean = rng.normal(size=(3, 17, 3)) * 0.1
    target = clean.copy()
    target[:, [2, 3], 0] += 0.2
    return clean, target


def test_exact_payload_change_is_zero_despite_clean_prediction_bias():
    clean, target = _poses()
    bias = np.asarray([0.11, -0.03, 0.04])
    np.testing.assert_allclose(
        dose_control_error(target + bias, clean + bias, target, clean, 1),
        np.zeros(3), atol=1e-15)


def test_zero_dose_has_zero_error():
    clean, _ = _poses()
    prediction = clean + 0.17
    np.testing.assert_array_equal(
        dose_control_error(prediction, prediction, clean, clean, 1),
        np.zeros(3))


def test_unchanged_prediction_has_positive_payload_error():
    clean, target = _poses()
    prediction = clean + 0.17
    np.testing.assert_allclose(
        dose_control_error(prediction, prediction, target, clean, 1),
        np.full(3, 0.2), atol=1e-15)


def test_wrong_direction_is_penalized_even_for_equal_displacement_norm():
    clean, target = _poses()
    wrong = clean - (target - clean)
    np.testing.assert_allclose(
        dose_control_error(wrong, clean, target, clean, 1),
        np.full(3, 0.4), atol=1e-15)


def test_metric_is_per_sample_target_only_and_in_coordinate_units():
    clean, target = _poses()
    prediction = target.copy()
    prediction[0, [2, 3], 0] += 0.1
    prediction[1, [2, 3], 1] += 0.2
    prediction[:, 0, :] += 100.0  # Outside the pivot's descendants.
    np.testing.assert_allclose(
        dose_control_error(prediction, clean, target, clean, 1),
        [0.1, 0.2, 0.0], atol=1e-15)


@pytest.mark.parametrize('invalid', [
    np.zeros((3, 1, 17, 3)), np.zeros((3, 17, 2)), np.zeros((0, 17, 3)),
    np.zeros((3, 0, 3)), np.full((3, 17, 3), np.nan),
    np.full((3, 17, 3), np.inf), np.zeros((3, 17, 3), dtype=complex),
    np.full((3, 17, 3), 'not-a-coordinate'),
])
def test_invalid_pose_arrays_are_rejected(invalid):
    clean, target = _poses()
    with pytest.raises(ValueError):
        dose_control_error(invalid, clean, target, clean, 1)


def test_mismatched_shapes_are_rejected():
    clean, target = _poses()
    with pytest.raises(ValueError, match='matching'):
        dose_control_error(target[:2], clean, target, clean, 1)


@pytest.mark.parametrize('pivot', [-1, 17, 3, True, 1.5, '1'])
def test_invalid_or_leaf_pivot_is_rejected(pivot):
    clean, target = _poses()
    with pytest.raises(ValueError):
        dose_control_error(target, clean, target, clean, pivot)


def test_active_skeleton_must_fit_supplied_pose():
    short = np.zeros((3, 2, 3))
    with pytest.raises(ValueError, match='descendants'):
        dose_control_error(short, short, short, short, 1)


@pytest.mark.parametrize('enabled', [None, False, True])
def test_evaluation_adds_metric_only_when_requested(monkeypatch, enabled):
    clean, target = _poses()
    bias = np.asarray([0.11, -0.03, 0.04])
    cfg = {
        'experiment_name': 'mmfi', 'pivot': 1, 'batch_size': 3,
        'payload_axis': [0.0, 0.0, 1.0], 'eps': 0.185,
        'theta_max_deg': 40.0, 'dose_mode': 'linear',
        'dose_grid': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }
    if enabled is not None:
        cfg['measure_dose_control'] = enabled
    monkeypatch.setattr(trainer, 'PoisonedDataset',
                        lambda base, trig, mode, **kw: (mode, kw.get('fixed_dose', 0)))
    monkeypatch.setattr(trainer, 'DataLoader', lambda dataset, **kw: dataset)

    def predict(model, loader, device, **kw):
        mode, dose = loader
        if mode == 'clean':
            return clean + bias, clean, None, None
        prescribed = clean + dose * (target - clean)
        prediction = clean + bias + dose * 0.5 * (target - clean)
        return prediction, clean, prescribed, None

    monkeypatch.setattr(trainer, '_predict', predict)
    result = trainer.evaluate(None, object(), object(), cfg, 'cpu')
    if enabled:
        np.testing.assert_allclose(result['dose_control_error'],
                                   [0.0, 0.02, 0.04, 0.06, 0.08, 0.1], atol=1e-15)
        assert result['dose_control_error'] == [
            attack['dose_control_error_mean'] for attack in result['asr_curve']]
        assert result['asr@ref']['dose_control_error_mean'] == pytest.approx(0.1)
    else:
        assert 'dose_control_error' not in result
        assert all('dose_control_error_mean' not in attack
                   for attack in result['asr_curve'])


def _write_cache(path, cfg, result):
    path.write_text(json.dumps({
        'result_schema': trainer._RESULT_SCHEMA,
        'cfg_fingerprint': trainer._config_fingerprint(cfg),
        'trained_epochs': cfg['epochs'], 'res': result,
    }), encoding='utf-8')


def test_metric_flag_changes_fingerprint_without_changing_legacy_config():
    cfg = {'model': 'hpeli', 'experiment_name': 'mmfi'}
    legacy = trainer._resolve_training_config(cfg)
    assert 'measure_dose_control' not in legacy
    requested = trainer._resolve_training_config({**cfg, 'measure_dose_control': True})
    assert trainer._config_fingerprint(legacy) != trainer._config_fingerprint(requested)


@pytest.mark.parametrize('value', ['yes', 1, None])
def test_metric_flag_requires_a_boolean(value):
    with pytest.raises(ValueError, match='measure_dose_control'):
        trainer._resolve_training_config({
            'model': 'hpeli', 'experiment_name': 'mmfi', 'measure_dose_control': value})


@pytest.mark.parametrize('enabled', [False, True])
def test_legacy_cache_is_rejected_only_for_opt_in_metric(tmp_path, enabled):
    cfg = {'epochs': 50}
    if enabled:
        cfg['measure_dose_control'] = True
    old_result = {'dose_grid': [0.0, 1.0], 'asr_curve': [{}, {}]}
    _write_cache(tmp_path / 'eval_cache.json', cfg, old_result)
    result = trainer._load_cached_result(tmp_path, cfg)
    assert result is None if enabled else result == old_result


@pytest.mark.parametrize('invalid', [None, [0.0], [0.0, float('nan')], [0.0, -0.1]])
def test_opt_in_cache_rejects_missing_malformed_or_nonfinite_metric(tmp_path, invalid):
    cfg = {'epochs': 50, 'measure_dose_control': True}
    result = {'dose_grid': [0.0, 1.0], 'dose_control_error': invalid,
              'asr_curve': [{'dose_control_error_mean': 0.0},
                            {'dose_control_error_mean': 0.1}]}
    _write_cache(tmp_path / 'eval_cache.json', cfg, result)
    assert trainer._load_cached_result(tmp_path, cfg) is None


def test_opt_in_cache_requires_same_top_level_and_attack_curve(tmp_path):
    cfg = {'epochs': 50, 'measure_dose_control': True}
    result = {'dose_grid': [0.0, 1.0], 'dose_control_error': [0.0, 0.1],
              'asr_curve': [{'dose_control_error_mean': 0.0},
                            {'dose_control_error_mean': 0.2}]}
    _write_cache(tmp_path / 'eval_cache.json', cfg, result)
    assert trainer._load_cached_result(tmp_path, cfg) is None
    result['asr_curve'][1]['dose_control_error_mean'] = 0.1
    _write_cache(tmp_path / 'eval_cache.json', cfg, result)
    assert trainer._load_cached_result(tmp_path, cfg) == result
