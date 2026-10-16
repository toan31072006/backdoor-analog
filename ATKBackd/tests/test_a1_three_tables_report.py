import copy
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1_three_tables_report import (
    BASELINES, COMPLETION_FILE, GRID, KEYS, SEEDS, STATUS, build_report, export_reports,
)


def make_records():
    records = []
    for seed in SEEDS:
        offset = seed - 42
        for index, key in enumerate(KEYS):
            if key == 'clean':
                metrics = dict(t1_mpjpe_mm=None, target_pck_pct=None,
                    clean_mpjpe_mm=170 + offset, clean_pampjpe_mm=100 + 2 * offset,
                    mean_dce_mm=None)
                doses = []
            else:
                endpoint = 100 + index * 10 + offset * 2
                doses = [dict(d=d, tmpjpe_mm=endpoint - 30 * (1 - d),
                    dce_mm=(10 + index + offset * 3) * d, no_trigger_tmpjpe_mm=300 + d * 100)
                    for d in GRID]
                metrics = dict(t1_mpjpe_mm=endpoint,
                    target_pck_pct=60 - index + offset if key in (*BASELINES, 'a1') else None,
                    clean_mpjpe_mm=170 + index * .1 + offset,
                    clean_pampjpe_mm=100 + index * .2 + offset * 2,
                    mean_dce_mm=mean(row['dce_mm'] for row in doses[1:]))
            records.append(dict(seed=seed, key=key, metrics=metrics, dose_rows=doses,
                config_fingerprint=f'config-{seed}-{key}', diagnostics=dict(example=1)))
    return records


def test_complete_report_uses_sample_sd_and_all_seeds():
    report = build_report(make_records(), plan_sha256='plan')
    assert report['status'] == STATUS
    assert report['seeds'] == [42, 43, 44]
    row = next(row for row in report['tables']['T1']['rows'] if row['key'] == 'a1')
    assert row['t1_mpjpe_mm'] == dict(mean=172.0, sample_sd=2.0, n=3)
    assert report['aggregation']['ddof'] == 1
    assert report['aggregation']['frame_replicates'] is False


def test_comparison_order_clean_then_decreasing_baselines_then_a1():
    report = build_report(make_records()[::-1], plan_sha256='plan')
    rows = report['tables']['T1']['rows']
    assert rows[0]['key'] == 'clean'
    assert rows[-1]['key'] == 'a1'
    assert [row['t1_mpjpe_mm']['mean'] for row in rows[1:-1]] == sorted(
        [row['t1_mpjpe_mm']['mean'] for row in rows[1:-1]], reverse=True)
    assert rows[0]['target_pck_pct'] is None


def test_dce_summary_averages_within_seed_first():
    records = make_records()
    for record in records:
        if record['key'] == 'a1':
            a = record['seed'] - 41
            # Strongly different per-dose scatter with fixed means exposes
            # incorrect pooling over doses or averaging individual-dose SDs.
            values = [0, 10 * a, 100 - 10 * a, 20, 40, 80]
            for row, value in zip(record['dose_rows'], values):
                row['dce_mm'] = value
            record['metrics']['mean_dce_mm'] = mean(values[1:])
    report = build_report(records, plan_sha256='plan')
    row = report['tables']['T3']['rows'][0]
    assert row['mean_dce_mm'] == dict(mean=48.0, sample_sd=0.0, n=3)
    assert row['variant'] == 'A1--Full'


def test_matrix_has_six_dose_rows_three_rho_columns():
    report = build_report(make_records(), plan_sha256='plan')
    rows = report['tables']['T2']['dose_matrix']
    assert [row['d'] for row in rows] == list(GRID)
    assert set(rows[0]) == {'d', 'rho0p1', 'rho0p2', 'rho0p4'}
    assert rows[-1]['rho0p1']['mean'] == 172
    assert len(report['tables']['T2']['poison_summary']) == 3
    assert report['tables']['T3']['rows'][1]['auxiliary_carrier'] is True
    assert report['tables']['T3']['rows'][1]['dose_coupling'] is False
    assert report['tables']['T3']['rows'][2]['auxiliary_carrier'] is False
    assert report['tables']['T3']['rows'][2]['dose_coupling'] is True


def test_shared_a1_has_one_record_per_seed_not_three_replicates():
    report = build_report(make_records(), plan_sha256='plan')
    assert len(report['records']) == 39
    assert sum(record['key'] == 'a1' for record in report['records']) == 3
    assert report['tables']['T1']['rows'][-1]['t1_mpjpe_mm'] == \
        report['tables']['T3']['rows'][0]['t1_mpjpe_mm']


def test_build_report_does_not_mutate_input():
    records = make_records()
    before = copy.deepcopy(records)
    report = build_report(records, plan_sha256='plan', provenance={'input': ['original']})
    report['records'][0]['diagnostics']['example'] = 99
    assert records == before


@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate', 'wrong-seed', 'unknown-key', 'bool-seed'])
def test_rejects_incomplete_duplicate_or_unplanned_cells(change):
    records = make_records()
    if change == 'missing':
        records.pop()
    elif change == 'extra':
        records.append(copy.deepcopy(records[0]))
    elif change == 'duplicate':
        records[-1] = copy.deepcopy(records[-2])
    elif change == 'wrong-seed':
        records[-1]['seed'] = 45
    elif change == 'unknown-key':
        records[-1]['key'] = 'unplanned-method'
    else:
        records[-1]['seed'] = True
    with pytest.raises(ValueError):
        build_report(records, plan_sha256='plan')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True, '174'])
def test_rejects_nonfinite_negative_bool_or_string_metric(value):
    records = make_records()
    records[1]['metrics']['clean_mpjpe_mm'] = value
    with pytest.raises(ValueError):
        build_report(records, plan_sha256='plan')


def test_requires_target_pck_for_every_t1_targeted_method():
    records = make_records()
    records[1]['metrics']['target_pck_pct'] = None
    with pytest.raises(ValueError):
        build_report(records, plan_sha256='plan')


def test_rejects_target_pck_above_100():
    records = make_records()
    records[1]['metrics']['target_pck_pct'] = 100.01
    with pytest.raises(ValueError):
        build_report(records, plan_sha256='plan')


def test_clean_attack_columns_are_null_not_zero_or_trigger_diagnostic():
    records = make_records()
    records[0]['metrics']['target_pck_pct'] = 2.4
    with pytest.raises(ValueError, match='Clean reference'):
        build_report(records, plan_sha256='plan')


@pytest.mark.parametrize('change', ['missing-dose', 'duplicate-dose', 'near-dose', 'identity-dce', 'wrong-mean', 'endpoint'])
def test_rejects_curve_inconsistency(change):
    records = make_records()
    record = records[1]
    if change == 'missing-dose':
        record['dose_rows'].pop()
    elif change == 'duplicate-dose':
        record['dose_rows'][1]['d'] = 0
    elif change == 'near-dose':
        record['dose_rows'][1]['d'] = .2000001
    elif change == 'identity-dce':
        record['dose_rows'][0]['dce_mm'] = 1
    elif change == 'wrong-mean':
        record['metrics']['mean_dce_mm'] += .00001
    else:
        record['metrics']['t1_mpjpe_mm'] += .1
    with pytest.raises(ValueError):
        build_report(records, plan_sha256='plan')


def test_replay_float_tolerance_does_not_reject_audited_float32_difference():
    records = make_records()
    records[1]['metrics']['t1_mpjpe_mm'] += .00003
    build_report(records, plan_sha256='plan')


def test_accepts_unsorted_complete_dose_rows():
    records = make_records()
    records[1]['dose_rows'].reverse()
    report = build_report(records, plan_sha256='plan')
    assert [row['d'] for row in report['records'][1]['dose_rows']] == list(GRID)


def test_dose_summary_sd_uses_one_dose_scalar_per_seed():
    records = make_records()
    report = build_report(records, plan_sha256='plan')
    row = next(row for row in report['dose_response_summary'] if row['key'] == 'a1' and row['d'] == 1)
    assert row['tmpjpe_mm'] == dict(mean=172.0, sample_sd=2.0, n=3)
    assert row['dce_mm']['sample_sd'] == 3


def test_rejects_missing_fingerprint_or_non_json_metadata():
    records = make_records()
    records[0]['config_fingerprint'] = ''
    with pytest.raises(ValueError):
        build_report(records, plan_sha256='plan')
    records = make_records()
    records[0]['diagnostics'] = {'bad': float('nan')}
    with pytest.raises(ValueError):
        build_report(records, plan_sha256='plan')


def test_exports_all_tables_and_per_seed_artifacts(tmp_path):
    report = export_reports(make_records(), tmp_path, plan_sha256='plan', provenance={'test': True})
    assert (tmp_path / COMPLETION_FILE).exists()
    assert all((tmp_path / name).exists() for name in report['exported_files'])
    for stem in ('table1_comparison', 'table2_dose_poison', 'table3_ablation'):
        for suffix in ('.csv', '.json', '.md', '.tex'):
            assert (tmp_path / (stem + suffix)).is_file()
    for seed in SEEDS:
        with (tmp_path / f'per_seed_metrics_s{seed}.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 13
        assert {row['seed'] for row in rows} == {str(seed)}
    with (tmp_path / 'dose_response_per_seed.csv').open(newline='', encoding='utf-8') as stream:
        assert len(list(csv.DictReader(stream))) == 216
    with (tmp_path / 'dose_response_summary.csv').open(newline='', encoding='utf-8') as stream:
        assert len(list(csv.DictReader(stream))) == 72
    saved = json.loads((tmp_path / COMPLETION_FILE).read_text(encoding='utf-8'))
    assert saved['aggregation']['ddof'] == 1
    assert saved['status'] == STATUS


def test_export_headers_labels_and_no_extra_methods_or_plots(tmp_path):
    export_reports(make_records(), tmp_path, plan_sha256='plan')
    comparison = (tmp_path / 'table1_comparison.tex').read_text(encoding='utf-8')
    ablation = (tmp_path / 'table3_ablation.tex').read_text(encoding='utf-8')
    matrix = (tmp_path / 'table2_dose_poison.tex').read_text(encoding='utf-8')
    assert 'Target-PCK $' in comparison
    assert 'Target-PCK@0.2' not in comparison
    assert 'threshold 0.2' in comparison
    assert 'Auxiliary' in ablation and 'A1--Full' in ablation
    assert r'$\times$' in ablation and 'w/o' not in ablation
    assert r'\begin{tabular}{l|rrr}' in matrix
    assert not list(tmp_path.glob('*.png'))
    assert not any('Blended' in path.read_text(encoding='utf-8') for path in tmp_path.glob('*.md'))


def test_invalid_inputs_do_not_create_output_or_complete_marker(tmp_path):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        export_reports(make_records()[:-1], output, plan_sha256='plan')
    assert not output.exists()


def test_different_plan_cannot_overwrite_existing_report(tmp_path):
    export_reports(make_records(), tmp_path, plan_sha256='old')
    original = (tmp_path / COMPLETION_FILE).read_bytes()
    with pytest.raises(ValueError, match='different plan'):
        export_reports(make_records(), tmp_path, plan_sha256='new')
    assert (tmp_path / COMPLETION_FILE).read_bytes() == original


def test_export_failure_does_not_emit_new_complete_marker(tmp_path, monkeypatch):
    import a1_three_tables_report as module
    original = module._atomic_text

    def fail_on_second_table(path, value):
        if path.name == 'table2_dose_poison.tex':
            raise OSError('simulated export failure')
        original(path, value)

    monkeypatch.setattr(module, '_atomic_text', fail_on_second_table)
    with pytest.raises(OSError):
        export_reports(make_records(), tmp_path, plan_sha256='plan')
    assert not (tmp_path / COMPLETION_FILE).exists()
