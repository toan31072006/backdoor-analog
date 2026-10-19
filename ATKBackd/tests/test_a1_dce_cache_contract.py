"""Real-evaluator regressions for the historical per-frame DCE cache contract.

The production operator already averages the two target-joint norms.  Its
cached shape is (N, 6), not an invented (N, 6, 2) joint axis.  These tests
deliberately exercise the evaluator before the reporting helper, so an
internally consistent synthetic reporting fixture cannot hide that mismatch.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defend.a1_defense_eval import GRID, evaluate_defense
from defend.a1_defense_metrics import enrich_evaluation
from test_a1_defense_eval import IdentityPose, NumericTrigger, TinyMMFi, config


@pytest.fixture(scope='module')
def real_evaluation():
    """Run all seven CPU prediction passes, without mocking the DCE operator."""
    data = TinyMMFi(count=8)
    return evaluate_defense(IdentityPose(), data, NumericTrigger(), config(),
                            'cpu', expected_count=len(data))


def test_real_evaluator_caches_one_joint_averaged_dce_per_frame_and_dose(real_evaluation):
    result = real_evaluation
    errors = result['arrays']['dce_error_mm']
    assert errors.shape == (8, 6)
    assert result['arrays']['target_error_mm'].shape == (8, 6, 2)
    assert result['arrays']['triggered_gt_error_mm'].shape == (8, 6, 2)
    np.testing.assert_array_equal(errors[:, 0], np.zeros(8))
    assert np.all(errors[:, 1:] > 0.)
    for dose_index, dose in enumerate(GRID):
        row = result['dose_rows'][dose_index]
        assert row['d'] == dose
        assert row['dce_mm'] == pytest.approx(errors[:, dose_index].mean())
    positive_only = float(errors[:, 1:].mean())
    assert result['diagnostics']['mean_positive_dce_mm'] == pytest.approx(positive_only)
    assert positive_only != pytest.approx(errors.mean())
    # Eight synthetic frames are never described as a full official MM-Fi run.
    assert result['diagnostics']['full_official_test'] is False


def test_reporting_accepts_real_evaluator_dce_without_rewriting_cached_arrays(real_evaluation):
    source = copy.deepcopy(real_evaluation)
    enriched = enrich_evaluation(real_evaluation)
    for key, value in source['arrays'].items():
        np.testing.assert_array_equal(real_evaluation['arrays'][key], value)
        np.testing.assert_array_equal(enriched['arrays'][key], value)
    assert real_evaluation['metrics'] == source['metrics']
    assert real_evaluation['diagnostics'] == source['diagnostics']
    assert real_evaluation['dose_rows'] == source['dose_rows']
    for key, value in source['metrics'].items():
        assert enriched['metrics'][key] == value
    assert enriched['metrics']['mean_positive_dce_mm'] == pytest.approx(
        source['arrays']['dce_error_mm'][:, 1:].mean())
    for source_row, replayed_row in zip(source['dose_rows'], enriched['dose_rows']):
        assert replayed_row['dce_mm'] == source_row['dce_mm']
    assert enriched['diagnostics']['dce_role'] == 'diagnostic, not defense success'


def test_dce_operator_averages_target_joint_norms_and_returns_metres():
    from attack.payload import set_skeleton_config
    from eval.metrics import dose_control_error

    set_skeleton_config('mmfi')
    true = np.zeros((2, 17, 3), dtype=np.float64)
    target = true.copy()
    target[:, 2, 0] = .1
    target[:, 3, 1] = .2
    # A different common clean bias per frame and joint must cancel.
    clean = np.arange(true.size, dtype=np.float64).reshape(true.shape) * .01
    predicted = clean + target
    predicted[:, 2, 0] += [3., 6.]
    predicted[:, 3, 1] += [4., 8.]
    errors_metres = dose_control_error(predicted, clean, target, true, 1)
    assert errors_metres.shape == (2,)
    np.testing.assert_allclose(errors_metres, [3.5, 7.], rtol=0., atol=1.e-12)
    np.testing.assert_allclose(errors_metres * 1000., [3500., 7000.], atol=1.e-9)
    assert errors_metres[0] != pytest.approx(np.linalg.norm([3., 4.]))


def test_reporting_rejects_fabricated_target_joint_dce_axis(real_evaluation):
    source = copy.deepcopy(real_evaluation)
    source['arrays']['dce_error_mm'] = np.repeat(
        source['arrays']['dce_error_mm'][:, :, None], 2, axis=2)
    with pytest.raises(ValueError, match='dce_error_mm'):
        enrich_evaluation(source)


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -1.])
def test_reporting_keeps_strict_dce_numeric_validation(real_evaluation, invalid):
    source = copy.deepcopy(real_evaluation)
    source['arrays']['dce_error_mm'][0, 1] = invalid
    with pytest.raises(ValueError, match='dce_error_mm'):
        enrich_evaluation(source)


def test_v2_npz_reader_replays_real_operator_contract(tmp_path, real_evaluation):
    """Expand real CPU results solely to test the full-count cache reader gate.

    Replicated synthetic rows are not a full MM-Fi evaluation.  Unlike former
    reporting-only fixtures, their DCE shape and values originate in the actual
    frozen evaluator and survive the NPZ serialization boundary unchanged.
    """
    from run_a1_defenses_v2 import _enriched_record
    from run_method_drafts import _sha_json

    count = 33264
    copies = count // 8
    assert copies * 8 == count
    arrays = {key: (value.copy() if key == 'doses'
                    else np.concatenate([value] * copies, axis=0))
              for key, value in real_evaluation['arrays'].items()}
    ids = [[f'fixture/A00/frame_{index}.npy', 'fixture/A00/ground_truth.npy', index]
           for index in range(count)]
    arrays['sample_indices'] = np.arange(count, dtype=np.int64)
    arrays['pair_ids'] = np.asarray(ids, dtype=str)
    np.savez_compressed(tmp_path / 'per_sample.npz', **arrays)
    record = copy.deepcopy({key: value for key, value in real_evaluation.items()
                            if key != 'arrays'})
    record['diagnostics'].update(samples=count, full_official_test=True)
    record['binding'] = dict(inputs=dict(eval=dict(n=count,
                                                  pair_ids_sha256=_sha_json(ids))))
    replayed = _enriched_record(record, tmp_path)
    assert replayed['arrays']['dce_error_mm'].shape == (count, 6)
    np.testing.assert_array_equal(replayed['arrays']['dce_error_mm'],
                                  arrays['dce_error_mm'])
    assert replayed['metrics']['mean_positive_dce_mm'] == pytest.approx(
        real_evaluation['diagnostics']['mean_positive_dce_mm'])
    assert replayed['diagnostics']['samples'] == count
