"""Array-replayed T5 metrics and seed-paired summaries, without MM-Fi or GPU."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defend.a1_defense_metrics import enrich_evaluation, build_metric_report


GRID = np.array([0., .2, .4, .6, .8, 1.])


def evaluated_fixture():
    """Two frames deliberately distinguish target-joint from all-joint error."""
    clean = np.full((2, 17), 1000., dtype=np.float64)
    clean[:, [2, 3]] = [[10., 30.], [20., 40.]]
    scale = np.array([100., 200.])
    start = clean[:, [2, 3]]
    target_end = np.array([[10., 30.], [40., 60.]])
    gt_end = np.array([[12., 18.], [24., 30.]])
    no_trigger_end = np.array([[30., 50.], [60., 80.]])
    target = np.stack([(1. - d) * start + d * target_end for d in GRID], axis=1)
    gt = np.stack([(1. - d) * start + d * gt_end for d in GRID], axis=1)
    no_trigger = np.stack([(1. - d) * start + d * no_trigger_end for d in GRID], axis=1)
    dce = np.broadcast_to((10. * GRID)[None, :, None], (2, 6, 2)).copy()
    pck = lambda error: float(100. * np.mean(error <= .2 * scale[:, None]))
    rows = [dict(d=float(d), t_mpjpe_mm=float(target[:, i].mean()),
                 target_pck_pct=pck(target[:, i]),
                 triggered_gt_mpjpe_mm=float(gt[:, i].mean()),
                 dce_mm=float(dce[:, i].mean()),
                 no_trigger_tmpjpe_mm=float(no_trigger[:, i].mean()))
            for i, d in enumerate(GRID)]
    return dict(
        metrics=dict(t_mpjpe_mm=float(target[:, -1].mean()),
                     target_pck_pct=pck(target[:, -1]),
                     clean_mpjpe_mm=float(clean.mean()),
                     triggered_gt_mpjpe_mm=float(gt[:, -1].mean())),
        diagnostics=dict(mean_positive_dce_mm=6., no_trigger_t_mpjpe_mm=55.,
                         no_trigger_t1_mpjpe_mm=55., no_trigger_target_pck_pct=0.,
                         clean_pampjpe_mm=87.5, samples=2, full_official_test=False,
                         target_joints=[2, 3], target_pck_threshold=.2,
                         scale='norm(clean_true_pose[5]-clean_true_pose[12])',
                         main_dose=1., dose_grid=GRID.tolist(), alignment='none',
                         selection_on_test=False),
        dose_rows=rows,
        arrays=dict(sample_indices=np.arange(2), clean_error_mm=clean,
                    clean_scale_mm=scale, doses=GRID.copy(),
                    target_error_mm=target, triggered_gt_error_mm=gt,
                    no_trigger_target_error_mm=no_trigger, dce_error_mm=dce))


def assert_nested_equal(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_enrichment_replays_hand_calculated_target_scope_and_keeps_original():
    source = evaluated_fixture()
    source['provenance_marker'] = {'do_not_replace': True}
    before = copy.deepcopy(source)
    enriched = enrich_evaluation(source)
    assert_nested_equal(source, before)
    assert enriched is not source
    assert enriched['provenance_marker'] == source['provenance_marker']
    assert_nested_equal(enriched['arrays'], before['arrays'])
    metrics = enriched['metrics']
    assert metrics['clean_target_mpjpe_mm'] == pytest.approx(25.)
    assert metrics['clean_mpjpe_mm'] == pytest.approx(30100. / 34.)
    assert metrics['clean_pampjpe_mm'] == pytest.approx(87.5)
    assert metrics['t_mpjpe_mm'] == pytest.approx(35.)
    # Joint-level PCK is 2/4=50%; a pose-level "all joints hit" ASR would be 0%.
    assert metrics['target_pck_pct'] == pytest.approx(50.)
    assert metrics['no_trigger_target_pck_pct'] == pytest.approx(0.)
    assert metrics['no_trigger_t_mpjpe_mm'] == pytest.approx(55.)
    assert metrics['mean_positive_dce_mm'] == pytest.approx(6.)
    assert metrics['triggered_gt_minus_clean_target_mm'] == pytest.approx(-4.)
    assert not any('asr' in key.lower() for key in metrics)
    assert len(enriched['dose_rows']) == 6


@pytest.mark.parametrize('field', [
    'clean_error_mm', 'clean_scale_mm', 'doses', 'target_error_mm',
    'triggered_gt_error_mm', 'no_trigger_target_error_mm', 'dce_error_mm'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -1.])
def test_nonfinite_or_negative_array_values_fail_closed(field, bad):
    evaluated = evaluated_fixture()
    evaluated['arrays'][field].flat[0] = bad
    with pytest.raises(ValueError):
        enrich_evaluation(evaluated)


@pytest.mark.parametrize('field,shape', [
    ('clean_error_mm', (2, 16)), ('clean_scale_mm', (2, 1)),
    ('target_error_mm', (2, 6, 17)), ('triggered_gt_error_mm', (2, 5, 2)),
    ('no_trigger_target_error_mm', (1, 6, 2)), ('dce_error_mm', (2, 6)),
    ('doses', (1, 6))])
def test_wrong_array_shapes_rejected(field, shape):
    evaluated = evaluated_fixture()
    evaluated['arrays'][field] = np.ones(shape)
    with pytest.raises(ValueError):
        enrich_evaluation(evaluated)


def test_pck_scale_must_be_strictly_positive():
    evaluated = evaluated_fixture()
    evaluated['arrays']['clean_scale_mm'][0] = 0.
    with pytest.raises(ValueError):
        enrich_evaluation(evaluated)


@pytest.mark.parametrize('key,change', [
    ('t_mpjpe_mm', .051), ('clean_mpjpe_mm', .051),
    ('triggered_gt_mpjpe_mm', .051), ('target_pck_pct', .001)])
def test_existing_main_metric_cannot_be_silently_replaced_on_replay(key, change):
    evaluated = evaluated_fixture()
    evaluated['metrics'][key] += change
    with pytest.raises(ValueError):
        enrich_evaluation(evaluated)


@pytest.mark.parametrize('field', [
    'target_error_mm', 'triggered_gt_error_mm', 'no_trigger_target_error_mm'])
def test_dose_zero_must_reproduce_clean_error_on_target_joints(field):
    evaluated = evaluated_fixture()
    evaluated['arrays'][field][0, 0, 0] += 1.
    with pytest.raises(ValueError):
        enrich_evaluation(evaluated)


def test_dose_zero_dce_must_be_zero():
    evaluated = evaluated_fixture()
    evaluated['arrays']['dce_error_mm'][0, 0, 0] = 1.
    with pytest.raises(ValueError):
        enrich_evaluation(evaluated)


@pytest.mark.parametrize('field,value', [
    ('target_joints', [1, 2]), ('target_pck_threshold', .5),
    ('main_dose', .8), ('alignment', 'procrustes'), ('samples', 3),
    ('dose_grid', [0., .1, .4, .6, .8, 1.]),
    ('scale', 'predicted_pose_scale')])
def test_frozen_metric_protocol_mismatch_rejected(field, value):
    evaluated = evaluated_fixture()
    evaluated['diagnostics'][field] = value
    with pytest.raises(ValueError):
        enrich_evaluation(evaluated)


def test_independent_array_dose_order_mismatch_rejected():
    evaluated = evaluated_fixture()
    evaluated['arrays']['doses'][1] = .1
    with pytest.raises(ValueError):
        enrich_evaluation(evaluated)


@pytest.mark.parametrize('field,value', [
    ('no_trigger_target_pck_pct', 1.), ('no_trigger_t_mpjpe_mm', 100.),
    ('mean_positive_dce_mm', 100.)])
def test_existing_diagnostics_must_match_arrays(field, value):
    evaluated = evaluated_fixture()
    evaluated['diagnostics'][field] = value
    with pytest.raises(ValueError):
        enrich_evaluation(evaluated)


@pytest.mark.parametrize('field', [
    't_mpjpe_mm', 'target_pck_pct', 'triggered_gt_mpjpe_mm',
    'dce_mm', 'no_trigger_tmpjpe_mm'])
def test_six_dose_rows_are_replayed_not_blindly_trusted(field):
    evaluated = evaluated_fixture()
    evaluated['dose_rows'][2][field] += 1.
    with pytest.raises(ValueError):
        enrich_evaluation(evaluated)


DEFENSES = ('none', 'anp', 'nad', 'ft_sam')
METRICS = (
    't_mpjpe_mm', 'target_pck_pct', 'clean_mpjpe_mm',
    'triggered_gt_mpjpe_mm', 'clean_target_mpjpe_mm', 'clean_pampjpe_mm',
    'no_trigger_target_pck_pct', 'no_trigger_t_mpjpe_mm',
    'mean_positive_dce_mm', 'triggered_gt_minus_clean_target_mm')


def report_records(defenses=DEFENSES):
    records = []
    for i, seed in enumerate((42, 43, 44)):
        for defense in defenses:
            repaired = defense != 'none'
            t_delta = (10., -20., 30.)[i] if repaired else 0.
            pck_delta = (-10., -30., -20.)[i] if repaired else 0.
            clean_delta = float(i + 1) if repaired else 0.
            clean_floor = 80. + 100. * i + clean_delta
            gt_error = 300. + 100. * i - ((20., 30., 40.)[i] if repaired else 0.)
            metrics = dict(t_mpjpe_mm=100. + 100. * i + t_delta,
                target_pck_pct=60. + 10. * i + pck_delta,
                clean_mpjpe_mm=100. + 100. * i + clean_delta,
                triggered_gt_mpjpe_mm=gt_error,
                clean_target_mpjpe_mm=clean_floor,
                clean_pampjpe_mm=60. + 100. * i + clean_delta,
                no_trigger_target_pck_pct=10. + 10. * i,
                no_trigger_t_mpjpe_mm=200. + 100. * i,
                mean_positive_dce_mm=50. + 10. * i,
                triggered_gt_minus_clean_target_mm=gt_error - clean_floor)
            rows = [dict(d=float(d), t_mpjpe_mm=metrics['t_mpjpe_mm'],
                         target_pck_pct=metrics['target_pck_pct'],
                         triggered_gt_mpjpe_mm=gt_error,
                         dce_mm=metrics['mean_positive_dce_mm'] if d > 0 else 0.,
                         no_trigger_tmpjpe_mm=metrics['no_trigger_t_mpjpe_mm'])
                    for d in GRID]
            records.append(dict(seed=seed, method='a1', defense=defense,
                                metrics=metrics, dose_rows=rows,
                                diagnostics=dict(samples=2, target_joints=[2, 3],
                                    target_pck_threshold=.2, main_dose=1.,
                                    alignment='none', dose_grid=GRID.tolist())))
    return records


def row_for(rows, defense, seed=None):
    matches = [row for row in rows if row['defense'] == defense
               and (seed is None or row.get('seed') == seed)]
    assert len(matches) == 1
    return matches[0]


def test_complete_report_aggregates_independent_seeds_with_sample_sd():
    records = report_records()
    before = copy.deepcopy(records)
    report = build_metric_report(records, expected_defenses=DEFENSES)
    assert records == before
    assert report['complete'] is True
    assert len(report['per_seed']) == 12
    assert len(report['six_dose']) == 72
    assert len(report['paired_deltas']) == 9
    assert len(report['aggregates']) == 4
    assert len(report['paired_aggregates']) == 3
    aggregate = row_for(report['aggregates'], 'anp')
    assert aggregate['seeds'] == [42, 43, 44]
    assert aggregate['n_seeds'] == 3
    assert aggregate['complete'] is True
    assert aggregate['t_mpjpe_mm_mean'] == pytest.approx(np.mean([110., 180., 330.]))
    assert aggregate['t_mpjpe_mm_sd'] == pytest.approx(np.std([110., 180., 330.], ddof=1))
    assert aggregate['target_pck_pct_mean'] == pytest.approx(50.)
    assert aggregate['target_pck_pct_sd'] == pytest.approx(10.)
    assert set(METRICS).issubset(report['per_seed'][0])


def test_default_report_covers_ft_and_three_selected_repairs():
    records = report_records(('none', 'ft', 'anp', 'nad', 'ft_sam'))
    report = build_metric_report(records)
    assert report['complete'] is True
    assert len(report['per_seed']) == 15
    assert len(report['six_dose']) == 90
    assert {row['defense'] for row in report['aggregates']} == {
        'none', 'ft', 'anp', 'nad', 'ft_sam'}


def test_paired_changes_use_same_seed_and_percentage_points_not_relative_percent():
    # Reverse input order to make accidental zip/order-based pairing visible.
    records = list(reversed(report_records()))
    report = build_metric_report(records, expected_defenses=DEFENSES)
    paired = row_for(report['paired_deltas'], 'anp', seed=43)
    assert paired['t_mpjpe_mm_delta'] == pytest.approx(-20.)
    assert paired['target_pck_delta_pp'] == pytest.approx(-30.)
    assert paired['clean_mpjpe_mm_delta'] == pytest.approx(2.)
    assert paired['triggered_gt_mpjpe_mm_delta'] == pytest.approx(-30.)
    assert 'target_pck_pct_delta' not in paired
    summary = row_for(report['paired_aggregates'], 'anp')
    assert summary['target_pck_delta_pp_mean'] == pytest.approx(-20.)
    assert summary['target_pck_delta_pp_sd'] == pytest.approx(10.)
    true_paired_sd = float(np.std([10., -20., 30.], ddof=1))
    assert summary['t_mpjpe_mm_delta_sd'] == pytest.approx(true_paired_sd)
    repair = row_for(report['aggregates'], 'anp')
    baseline = row_for(report['aggregates'], 'none')
    assert summary['t_mpjpe_mm_delta_sd'] != pytest.approx(
        repair['t_mpjpe_mm_sd'] - baseline['t_mpjpe_mm_sd'])


def test_missing_seed_is_partial_and_cannot_advertise_complete_three_seed_row():
    records = [record for record in report_records() if record['seed'] == 42]
    report = build_metric_report(records, expected_defenses=DEFENSES)
    assert report['complete'] is False
    aggregate = row_for(report['aggregates'], 'anp')
    assert aggregate['complete'] is False
    assert aggregate['n_seeds'] == 1
    assert aggregate['t_mpjpe_mm_sd'] is None
    assert row_for(report['paired_aggregates'], 'anp')['target_pck_delta_pp_sd'] is None
    assert 'PARTIAL' in report['status'].upper() or 'INCOMPLETE' in report['status'].upper()


def test_absent_same_seed_no_defense_never_borrows_another_seed_baseline():
    records = [record for record in report_records()
               if not (record['seed'] == 43 and record['defense'] == 'none')]
    report = build_metric_report(records, expected_defenses=DEFENSES)
    assert report['complete'] is False
    assert not any(row['seed'] == 43 for row in report['paired_deltas'])
    paired = row_for(report['paired_aggregates'], 'anp')
    assert paired['complete'] is False
    assert paired['seeds'] == [42, 44]
    assert paired['n_seeds'] == 2


def test_no_baseline_at_all_leaves_paired_results_unavailable():
    records = [record for record in report_records() if record['defense'] != 'none']
    report = build_metric_report(records, expected_defenses=DEFENSES)
    assert report['complete'] is False
    assert report['paired_deltas'] == []
    for row in report['paired_aggregates']:
        assert row['complete'] is False
        assert row['n_seeds'] == 0
        assert row['target_pck_delta_pp_mean'] is None
        assert row['target_pck_delta_pp_sd'] is None


def test_duplicate_seed_defense_rows_rejected_not_pseudoreplicated():
    records = report_records()
    records.append(copy.deepcopy(records[0]))
    with pytest.raises(ValueError):
        build_metric_report(records, expected_defenses=DEFENSES)


@pytest.mark.parametrize('field,value', [
    ('seed', 45), ('seed', True), ('seed', '42'),
    ('defense', 'fp'), ('method', 'badnets')])
def test_out_of_plan_identity_rejected(field, value):
    records = report_records()
    records[0][field] = value
    with pytest.raises(ValueError):
        build_metric_report(records, expected_defenses=DEFENSES)


@pytest.mark.parametrize('field,value', [
    ('t_mpjpe_mm', np.nan), ('clean_target_mpjpe_mm', np.inf),
    ('clean_pampjpe_mm', -1.), ('target_pck_pct', 101.),
    ('no_trigger_target_pck_pct', -1.)])
def test_report_cannot_aggregate_invalid_metric_values(field, value):
    records = report_records()
    records[0]['metrics'][field] = value
    with pytest.raises(ValueError):
        build_metric_report(records, expected_defenses=DEFENSES)


def test_negative_gt_minus_clean_floor_is_valid_not_clipped_or_recovery_rate():
    records = report_records()
    records[0]['metrics']['triggered_gt_mpjpe_mm'] = 70.
    records[0]['metrics']['triggered_gt_minus_clean_target_mm'] = -10.
    report = build_metric_report(records, expected_defenses=DEFENSES)
    row = row_for(report['per_seed'], 'none', seed=42)
    assert row['triggered_gt_minus_clean_target_mm'] == pytest.approx(-10.)
    assert not any('recovery_pct' in key or 'asr' in key.lower() for key in row)
