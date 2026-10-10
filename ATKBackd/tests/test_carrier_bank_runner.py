"""Six-arm screening contracts without benchmark data or GPU access."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_carrier_bank_drafts as runner
import run_learned_carrier_drafts as previous
import train_backdoor as trainer
from carrier_bank_fit import BANK_FIT_DEFAULTS
from test_learned_carrier_runner import _completed_fixture, _prepared_fixture, forbidden


@pytest.fixture
def matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_source_provenance', lambda: {'fixture': 'a' * 64})
    return runner.build_matrix(tmp_path / 'data', tmp_path / 'out', 'cpu', 0,
        epochs=3, train_samples=10, eval_samples=4, distortion_samples=4)


def test_six_cells_shared_victim_and_separate_attacker_budget(matrix):
    assert [c['method_key'] for c in matrix['cells']] == [m[0] for m in runner.METHODS]
    assert len(matrix['cells']) == 6
    assert matrix['DRAFT_ONLY'] is True
    assert matrix['metrics_contract']['official_test_used'] is False
    for cell in matrix['cells']:
        cfg = cell['cfg']
        assert cfg['seed'] == 42 and cfg['epochs'] == 3
        assert cfg['lr'] == .001 and cfg['momentum'] == .9
        assert cfg['optimizer'] == 'sgd' and cfg['victim_loss'] == 'mpjpe'
        assert cfg['rho'] == (0 if cell['method_key'] == 'clean' else .1)
        assert cfg['draft_profile'] == runner.PROFILE
        assert cfg['draft_eval_source'] == 'training_holdout'
        assert cfg['lc_relative_l2'] == .1 and cfg['lc_reference_eps'] == .185
        assert cfg['dose_coupling'] == 'paired' and cfg['dose_grid'] == runner.GRID
        assert cfg['target_joints'] == [2, 3]
    for cell, size in zip(matrix['cells'][3:], (2, 8, 8)):
        cfg = cell['cfg']
        assert cfg['lc_bank_size'] == size
        assert cfg['lc_surrogate_count'] == 2 and cfg['lc_lookahead_steps'] == 2
        assert cfg['lc_warmup_epochs'] == 10 and cfg['lc_inner_steps'] == 128
    assert matrix['cells'][2]['cfg']['eps'] == .2
    runner._validate_manifest(matrix)


def test_three_control_configs_are_unchanged_except_profile(matrix):
    cfg = matrix['cells'][0]['cfg']
    old = previous.build_matrix(Path(cfg['dataset_root']).parents[1],
        Path(matrix['cells'][0]['ckpt_dir']).parent, 'cpu', 0,
        epochs=3, train_samples=10, eval_samples=4, distortion_samples=4)
    for current, preceding in zip(matrix['cells'][:3], old['cells'][:3]):
        clean = {k: v for k, v in current['cfg'].items() if k != 'draft_profile'}
        control = {k: v for k, v in preceding['cfg'].items() if k != 'draft_profile'}
        assert clean == control
        assert 'lc_bank_size' not in clean and 'lc_surrogate_count' not in clean


def test_runtime_change_allowed_scientific_change_rejected(matrix):
    changed = copy.deepcopy(matrix)
    for cell in changed['cells']:
        cell['cfg'].update(device='cuda:3', num_workers=8)
    assert runner._plan_sha(changed) == matrix['plan_sha256']
    runner._validate_manifest(changed)
    changed['cells'][3]['cfg']['lc_inner_steps'] += 1
    with pytest.raises(ValueError, match='fingerprint'):
        runner._validate_manifest(changed)


def test_manifest_cannot_hide_bad_shared_protocol_even_if_rehashed(matrix):
    changed = copy.deepcopy(matrix)
    changed['cells'][4]['cfg']['epochs'] += 1
    changed['cells'][4]['recipe_sha256'] = runner._recipe_sha(
        changed['cells'][4]['cfg'], changed['sources']['source_sha256'])
    changed['plan_sha256'] = runner._plan_sha(changed)
    with pytest.raises(ValueError, match='shared protocol'):
        runner._validate_manifest(changed)


def test_source_drift_rejected(matrix, monkeypatch):
    monkeypatch.setattr(runner, '_source_provenance', lambda: {'fixture': 'b' * 64})
    with pytest.raises(ValueError, match='source'):
        runner._validate_manifest(matrix)


@pytest.mark.parametrize('option,value', [
    ('lc_surrogate_count', 0), ('lc_surrogate_count', 5), ('lc_lookahead_steps', True),
    ('lc_surrogate_seed_stride', -1), ('lc_utility_mpjpe_tolerance', -1),
    ('lc_utility_pa_tolerance', float('nan')), ('lc_pck_temperature', 0),
])
def test_invalid_fitting_options_rejected_before_data(tmp_path, option, value):
    with pytest.raises(ValueError):
        runner.build_matrix(tmp_path, tmp_path / 'out', 'cpu', 0, fit_options={option: value})


def test_dry_run_never_trains_fits_loads_or_checks_cuda(tmp_path, monkeypatch):
    import learned_carrier_fit as fitting
    monkeypatch.setattr(runner, '_check_inputs', forbidden)
    monkeypatch.setattr(previous, 'run_matrix', forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', forbidden)
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', forbidden)
    monkeypatch.setattr(fitting, 'prepare_learned_cell', forbidden)
    out = tmp_path / 'dry'
    assert runner.main(['--data-home', str(tmp_path / 'absent'), '--outdir', str(out), '--dry-run']) == 0
    record = json.loads((out / 'carrier_bank.resolved.json').read_text())
    assert len(record['cells']) == 6
    assert not (out / 'blended').exists()


def test_fresh_never_deletes_historical_results(tmp_path):
    out = tmp_path / 'occupied'
    out.mkdir()
    old = out / 'user.json'
    old.write_text('user result')
    with pytest.raises(FileExistsError, match='never deletes'):
        runner.main(['--data-home', str(tmp_path), '--outdir', str(out), '--fresh', '--dry-run'])
    assert old.read_text() == 'user result'


def test_bank_artifact_and_science_bound_on_resume(matrix):
    cell = matrix['cells'][4]
    prepared = _prepared_fixture(cell)
    assert runner._prepared_cell(cell)['cfg'] == prepared['cfg']
    artifact = Path(prepared['cfg']['lc_artifact_path'])
    artifact.write_text(artifact.read_text() + '\n')
    with pytest.raises(ValueError, match='artifact changed'):
        runner._complete(cell)


def test_export_retains_all_main_metrics_rows_and_six_doses(matrix, monkeypatch):
    matrix['metrics_contract'].update(clean_probe='missing_by_explicit_skip',
        distortion='missing_by_explicit_skip')
    matrix['plan_sha256'] = runner._plan_sha(matrix)
    _completed_fixture(matrix)
    monkeypatch.setattr(previous, 'build_distortion', forbidden)
    monkeypatch.setattr(previous, 'build_clean_probes', forbidden)
    out = Path(matrix['cells'][0]['ckpt_dir']).parent
    report = runner.export_summary(matrix, out)
    assert len(report['rows']) == 6 and len(report['dose_response']) == 36
    assert len(report['blended_comparison']) == 4
    assert len({r['poison_plan_sha256'] for k, r in report['audit']['cells'].items() if k != 'clean'}) == 1
    with (out / 'draft_summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    for name in runner.MAIN_METRICS:
        assert name in rows[0]
    assert not any(name in rows[0] for name in ('i1_improvement_mm', 'asr', 'spearman', 'displacement'))
    assert 'lc_bank_guard' in report['fitting_audit']
    assert 'no eligible round' in (out / 'draft_summary.md').read_text(encoding='utf-8')


def test_missing_rows_cannot_export_success(matrix):
    _prepared_fixture(matrix['cells'][0])
    with pytest.raises(ValueError, match='All six cells'):
        runner.build_summary(matrix)


def test_pointwise_win_cannot_hide_a_worse_clean_metric():
    base = dict(method_key='blended', method='Blended', clean_mpjpe_mm=180.,
        clean_pampjpe_mm=104., t1_mpjpe_mm=200.,
        **{name: 80. for name in runner.MAIN_PCK})
    candidate = dict(base, method_key='lc_bank', method='bank',
        clean_mpjpe_mm=179., clean_pampjpe_mm=103., t1_mpjpe_mm=190.)
    tolerance = dict(mpjpe_mm=5., pampjpe_mm=3., pck_percentage_points=1.)
    result = runner.compare_to_blended([base, candidate], tolerance)[0]
    assert result['pointwise_dominates_blended'] is True
    assert result['delta_t1_mpjpe_mm'] == -10
    candidate[runner.MAIN_PCK[-1]] = 79.5
    result = runner.compare_to_blended([base, candidate], tolerance)[0]
    assert result['pointwise_dominates_blended'] is False
    assert result['lower_t1_with_tolerated_clean_tradeoff'] is True
    candidate[runner.MAIN_PCK[-1]] = 78.
    result = runner.compare_to_blended([base, candidate], tolerance)[0]
    assert result['lower_t1_with_tolerated_clean_tradeoff'] is False


def test_comparison_requires_finite_metrics_and_blended():
    with pytest.raises(ValueError, match='Blended'):
        runner.compare_to_blended([], {})
    row = dict(method_key='blended', **{name: 1. for name in runner.MAIN_METRICS})
    row['clean_pampjpe_mm'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        runner.compare_to_blended([row], {})


def test_shell_wrapper_forwards_devices_without_remote_actions():
    wrapper = (Path(__file__).resolve().parents[2] / 'remote_linux/10_run_carrier_bank_drafts.sh').read_text()
    assert '--python)' in wrapper and '--data-home)' in wrapper
    assert 'run_carrier_bank_drafts.py' in wrapper and '"${DOSE_FORWARDED[@]}"' in wrapper
    assert 'mmfi_carrier_bank_s42_v1' in wrapper
    assert 'git push' not in wrapper and 'rm -' not in wrapper


@pytest.mark.parametrize('devices', [['cuda:0', 'cuda:0'], ['cuda:-1'], ['gpu:0']])
def test_invalid_devices_fail_before_resolution(tmp_path, devices):
    with pytest.raises(ValueError, match='Devices'):
        runner.main(['--data-home', str(tmp_path), '--outdir', str(tmp_path / 'bad'),
                     '--devices', *devices, '--dry-run'])
