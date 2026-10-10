"""Three-arm recipes, exports and resume contracts without MM-Fi or CUDA."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_carrier_bank_drafts as base
import run_paired_guard_drafts as runner
import run_learned_carrier_drafts as shared
import train_backdoor as trainer
from carrier_bank_fit import BANK_FIT_DEFAULTS
from test_learned_carrier_runner import _completed_fixture, _prepared_fixture, forbidden


@pytest.fixture
def matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_source_provenance', lambda: {'fixture': 'a' * 64})
    return runner.build_matrix(tmp_path / 'data', tmp_path / 'out', 'cpu', 0,
        epochs=3, train_samples=10, eval_samples=4, distortion_samples=4)


def _rehash(matrix):
    for cell in matrix['cells']:
        cell['recipe_sha256'] = runner._recipe_sha(cell['cfg'], matrix['sources']['source_sha256'])
    matrix['plan_sha256'] = runner._plan_sha(matrix)
    return matrix


def test_exact_three_cells_common_protocol_and_seven_zero_guard_tolerances(matrix):
    assert [c['method_key'] for c in matrix['cells']] == ['blended', 'lc_trainaware', 'lc_paired_guard']
    assert matrix['DRAFT_ONLY'] is True
    assert matrix['metrics_contract']['official_test_used'] is False
    assert matrix['metrics_contract']['main_metrics'] == list(base.MAIN_METRICS)
    assert len(runner.MAIN_METRICS) == 8 and len(runner.MAIN_PCK) == 5
    assert matrix['metrics_contract']['clean_probe'] == 'missing_by_explicit_skip'
    assert matrix['metrics_contract']['clean_victim_probes'].startswith('unsupported_not_run')
    for cell in matrix['cells']:
        cfg = cell['cfg']
        assert cfg['rho'] == .1 and cfg['seed'] == 42
        assert cfg['epochs'] == cfg['victim_epochs'] == 3
        assert cfg['optimizer'] == 'sgd' and cfg['lr'] == .001 and cfg['momentum'] == .9
        assert cfg['weight_decay'] == 0 and cfg['batch_size'] == 32
        assert cfg['pivot'] == 1 and cfg['target_joints'] == [2, 3]
        assert cfg['dose_min'] == .2 and cfg['dose_max'] == 1
        assert cfg['dose_grid'] == runner.GRID and cfg['dose_coupling'] == 'paired'
        assert cfg['lc_reference_eps'] == .185 and cfg['lc_relative_l2'] == .1
        assert cfg['draft_eval_source'] == 'training_holdout'
        assert cfg['victim_loss'] == 'mpjpe' and cfg['attacker_access'] == 'data_only'
    for key in runner.TOLERANCE_KEYS:
        assert matrix['cells'][1]['cfg'][key] == BANK_FIT_DEFAULTS[key]
        assert matrix['cells'][2]['cfg'][key] == 0.
    assert matrix['cells'][0]['cfg']['eps'] == .2
    assert all(cell['cfg']['lc_bank_size'] == 2 for cell in matrix['cells'][1:])
    assert all(cell['cfg']['lc_surrogate_count'] == cell['cfg']['lc_lookahead_steps'] == 2
               for cell in matrix['cells'][1:])
    runner._validate_manifest(matrix)


def test_control_configs_are_exact_existing_blended_and_winner_except_profile(matrix):
    cfg = matrix['cells'][0]['cfg']
    old = base.build_matrix(Path(cfg['dataset_root']).parents[1],
        Path(matrix['cells'][0]['ckpt_dir']).parent, 'cpu', 0,
        epochs=3, train_samples=10, eval_samples=4, distortion_samples=4)
    by_key = {cell['method_key']: cell for cell in old['cells']}
    for current in matrix['cells'][:2]:
        strip = lambda value: {k: v for k, v in value.items() if k != 'draft_profile'}
        assert strip(current['cfg']) == strip(by_key[current['method_key']]['cfg'])


def test_fit_compute_overrides_are_common_but_control_tolerances_remain_default(tmp_path):
    result = runner.build_matrix(tmp_path, tmp_path / 'out', 'cpu', 0,
        fit_options=dict(lc_warmup_epochs=1, lc_rounds=1, lc_outer_steps=2,
                         lc_inner_steps=1, lc_batch_size=4, lc_fit_samples=20))
    for cell in result['cells'][1:]:
        assert cell['cfg']['lc_inner_steps'] == 1 and cell['cfg']['lc_outer_steps'] == 2
    for key in runner.TOLERANCE_KEYS:
        assert result['cells'][1]['cfg'][key] == BANK_FIT_DEFAULTS[key]
        assert result['cells'][2]['cfg'][key] == 0.


@pytest.mark.parametrize('key', runner.TOLERANCE_KEYS)
@pytest.mark.parametrize('value', [0., True, -.1, float('nan')])
def test_control_tolerances_cannot_be_overridden(tmp_path, key, value):
    with pytest.raises(ValueError, match='immutable'):
        runner.build_matrix(tmp_path, tmp_path / 'out', fit_options={key: value})


@pytest.mark.parametrize('option,value', [('unknown', 1), ('lc_surrogate_count', 3),
    ('lc_lookahead_steps', 3), ('lc_inner_steps', True), ('lc_pck_temperature', 0)])
def test_invalid_or_nonprotocol_fit_options_rejected_before_data(tmp_path, option, value):
    with pytest.raises(ValueError):
        runner.build_matrix(tmp_path, tmp_path / 'out', fit_options={option: value})


def test_runtime_changes_allowed_scientific_drift_requires_new_plan(matrix):
    changed = copy.deepcopy(matrix)
    for cell in changed['cells']:
        cell['cfg'].update(device='cuda:2', num_workers=8)
    assert runner._plan_sha(changed) == matrix['plan_sha256']
    runner._validate_manifest(changed)
    changed['cells'][2]['cfg']['lc_inner_steps'] += 1
    with pytest.raises(ValueError, match='fingerprint'):
        runner._validate_manifest(changed)


@pytest.mark.parametrize('index,key,value', [(0, 'eps', .1), (1, 'epochs', 4),
    (2, 'lc_bank_size', 8), (2, 'draft_eval_source', 'official_test'),
    (2, 'pivot', 2), (2, 'lc_utility_pa_tolerance', .003)])
def test_rehash_cannot_hide_operator_or_task_changes(matrix, index, key, value):
    changed = copy.deepcopy(matrix)
    changed['cells'][index]['cfg'][key] = value
    _rehash(changed)
    with pytest.raises(ValueError):
        runner._validate_manifest(changed)


def test_metric_contract_cannot_add_asr_or_claim_official_test(matrix):
    for field, value in [('main_metrics', list(runner.MAIN_METRICS) + ['asr']),
                         ('official_test_used', True), ('clean_probe', 'a fourth clean victim')]:
        changed = copy.deepcopy(matrix)
        changed['metrics_contract'][field] = value
        _rehash(changed)
        with pytest.raises(ValueError, match='contract'):
            runner._validate_manifest(changed)


def test_source_hash_binds_new_and_relevant_existing_implementation(matrix, monkeypatch):
    hashes = runner._source_provenance()
    assert hashes == {'fixture': 'a' * 64}
    monkeypatch.setattr(runner, '_source_provenance', lambda: {'fixture': 'b' * 64})
    with pytest.raises(ValueError, match='source'):
        runner._validate_manifest(matrix)


def test_actual_source_inventory_includes_paired_and_control_implementation():
    hashes = runner._source_provenance()
    required = ('run_paired_guard_drafts.py', 'paired_guard_fit.py', 'run_carrier_bank_drafts.py',
        'carrier_bank_fit.py', 'run_learned_carrier_drafts.py', 'learned_carrier_fit.py',
        'train_backdoor.py', 'attack/learned_carrier.py', 'attack/poison.py', 'attack/peak_budget.py',
        'eval/metrics.py', 'data_utils/draft_subset.py', 'third_party/backdoorbench/blended.py')
    assert set(required) <= set(hashes)
    assert all(len(digest) == 64 for digest in hashes.values())


def test_dry_run_skips_data_fitting_victims_and_cuda(tmp_path, monkeypatch):
    import learned_carrier_fit as fitting
    for name in ('_check_inputs', '_validate_cuda_devices'):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setattr(shared, 'run_matrix', forbidden)
    monkeypatch.setattr(shared, 'build_clean_probes', forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', forbidden)
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(fitting, 'prepare_learned_cell', forbidden)
    out = tmp_path / 'dry'
    assert runner.main(['--data-home', str(tmp_path / 'absent'), '--outdir', str(out), '--dry-run']) == 0
    record = json.loads((out / 'paired_guard.resolved.json').read_text())
    assert len(record['cells']) == 3 and not (out / 'blended').exists()


def test_dry_resume_allows_workers_but_rejects_changed_science(tmp_path):
    out = tmp_path / 'dry'
    args = ['--data-home', str(tmp_path), '--outdir', str(out), '--dry-run']
    assert runner.main(args) == runner.main(args + ['--devices', 'cpu', '--num-workers', '0']) == 0
    with pytest.raises(ValueError, match='NEW output'):
        runner.main(args + ['--epochs', '14'])


def test_fresh_never_deletes_historical_results(tmp_path):
    out = tmp_path / 'occupied'
    out.mkdir()
    old = out / 'user.json'
    old.write_text('user result')
    with pytest.raises(FileExistsError, match='never deletes'):
        runner.main(['--data-home', str(tmp_path), '--outdir', str(out), '--fresh', '--dry-run'])
    assert old.read_text() == 'user result'


@pytest.mark.parametrize('index', [0, 1, 2])
def test_all_three_completion_paths_verify_cache_and_action_bytes(matrix, index):
    _completed_fixture(matrix)
    assert all(runner._complete(cell) for cell in matrix['cells'])
    cell = matrix['cells'][index]
    cache_path = Path(cell['eval_cache'])
    record = json.loads(cache_path.read_text())
    record['cfg']['pivot'] = 2
    runner._atomic_json(cache_path, record)
    with pytest.raises(ValueError):
        runner._complete(cell)


@pytest.mark.parametrize('index', [1, 2])
def test_each_learned_artifact_sha_tampering_fails_completion(matrix, index):
    prepared = _prepared_fixture(matrix['cells'][index])
    assert runner._prepared_cell(matrix['cells'][index])['cfg'] == prepared['cfg']
    artifact = Path(prepared['cfg']['lc_artifact_path'])
    artifact.write_bytes(artifact.read_bytes() + b' ')
    with pytest.raises(ValueError, match='artifact changed'):
        runner._complete(matrix['cells'][index])


def test_fitting_record_cannot_claim_external_holdout_access_even_with_updated_sha(matrix):
    from mmfi_tables import config_fingerprint
    cell = matrix['cells'][2]
    prepared = _prepared_fixture(cell)
    path = Path(cell['ckpt_dir']) / 'fitting.json'
    record = json.loads(path.read_text())
    record['external_draft_holdout_loaded'] = True
    runner._atomic_json(path, record)
    cfg = prepared['cfg']
    cfg['lc_fitting_sha256'] = runner._file_sha(path)
    runner._atomic_json(Path(cell['ckpt_dir']) / 'prepared_cfg.json', dict(cfg=cfg,
        recipe_sha256=cell['recipe_sha256'], cfg_fingerprint=config_fingerprint(cfg),
        action_file_sha256=runner._file_sha(cfg['action_npy'])))
    with pytest.raises(ValueError, match='TRAIN-only'):
        runner._complete(cell)


def test_export_retains_three_rows_and_eighteen_doses_including_failed_guard(matrix, monkeypatch):
    matrix['metrics_contract']['distortion'] = 'missing_by_explicit_skip'
    _rehash(matrix)
    _completed_fixture(matrix)
    cell = matrix['cells'][2]
    path = Path(cell['ckpt_dir']) / 'fitting.json'
    record = json.loads(path.read_text())
    record.update(selected_utility_gate_passed=False, no_eligible_candidate=True,
        utility_gate_required=True, clean_twin_actual_erm_steps_per_member=4,
        clean_twin_virtual_sgd_steps_per_member=8)
    runner._atomic_json(path, record)
    prepared_path = Path(cell['ckpt_dir']) / 'prepared_cfg.json'
    prepared = json.loads(prepared_path.read_text())
    prepared['cfg']['lc_fitting_sha256'] = runner._file_sha(path)
    from mmfi_tables import config_fingerprint
    fingerprint = config_fingerprint(prepared['cfg'])
    prepared['cfg_fingerprint'] = fingerprint
    runner._atomic_json(prepared_path, prepared)
    for name in ('eval_cache.json', 'draft_subsets.json', 'poison_manifest.json'):
        blob_path = Path(cell['ckpt_dir']) / name
        blob = json.loads(blob_path.read_text())
        if name == 'eval_cache.json':
            blob['cfg'] = prepared['cfg']
            blob['cfg_fingerprint'] = fingerprint
            blob['res']['config_fingerprint'] = fingerprint
        else:
            blob['config_fingerprint'] = fingerprint
        runner._atomic_json(blob_path, blob)
    monkeypatch.setattr(shared, 'build_distortion', forbidden)
    monkeypatch.setattr(shared, 'build_clean_probes', forbidden)
    out = Path(matrix['cells'][0]['ckpt_dir']).parent
    report = runner.export_summary(matrix, out)
    assert len(report['rows']) == 3 and len(report['dose_response']) == 18
    assert len(report['blended_comparison']) == 2
    assert len({r['poison_plan_sha256'] for r in report['audit']['cells'].values()}) == 1
    assert set(report['fitting_audit']) == {'lc_trainaware', 'lc_paired_guard'}
    assert report['fitting_audit']['lc_paired_guard']['no_eligible_candidate'] is True
    with (out / 'draft_summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 and all(name in rows[0] for name in runner.MAIN_METRICS)
    assert not set(('asr', 'i1_improvement_mm', 'displacement', 'spearman')) & set(rows[0])
    assert 'no eligible round: True' in (out / 'draft_summary.md').read_text()
    assert not (out / 'clean_victim_probes.json').exists()


def test_export_requires_every_planned_row(matrix):
    _prepared_fixture(matrix['cells'][0])
    with pytest.raises(ValueError, match='All three cells'):
        runner.build_summary(matrix)


def test_comparison_preserves_strict_dominance_and_separate_tradeoff_flag():
    baseline = dict(method_key='blended', method='Blended', clean_mpjpe_mm=180.,
        clean_pampjpe_mm=104., t1_mpjpe_mm=200., **{name: 80. for name in runner.MAIN_PCK})
    candidate = dict(baseline, method_key='lc_paired_guard', method='paired',
        clean_mpjpe_mm=179., clean_pampjpe_mm=103., t1_mpjpe_mm=190.)
    tolerances = dict(mpjpe_mm=5., pampjpe_mm=3., pck_percentage_points=1.)
    assert runner.compare_to_blended([baseline, candidate], tolerances)[0]['pointwise_dominates_blended'] is True
    for name in runner.MAIN_PCK:
        changed = dict(candidate, **{name: 79.5})
        result = runner.compare_to_blended([baseline, changed], tolerances)[0]
        assert result['pointwise_dominates_blended'] is False
        assert result['lower_t1_with_tolerated_clean_tradeoff'] is True


@pytest.mark.parametrize('devices', [['cuda:0', 'cuda:0'], ['cuda:-1'], ['gpu:0']])
def test_invalid_device_names_fail_before_resolution(tmp_path, devices):
    with pytest.raises(ValueError, match='Devices'):
        runner.main(['--data-home', str(tmp_path), '--outdir', str(tmp_path / 'bad'),
                     '--devices', *devices, '--dry-run'])


def test_cuda_preflight_explains_masked_logical_ids(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,3')
    runner._validate_cuda_devices(['cuda:0', 'cuda:1', 'cuda:2'])
    with pytest.raises(ValueError, match='device_count=3') as error:
        runner._validate_cuda_devices(['cuda:3'])
    assert "CUDA_VISIBLE_DEVICES='0,1,3'" in str(error.value)
    assert 'cuda:0 cuda:1 cuda:2' in str(error.value)


def test_cuda_unavailable_failure_is_actionable_and_cpu_needs_no_probe(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='valid logical IDs=none'):
        runner._validate_cuda_devices(['cuda:0'])
    monkeypatch.setattr(torch.cuda, 'device_count', forbidden)
    runner._validate_cuda_devices(['cpu'])


def test_preflight_runs_before_data_check_and_scheduler(tmp_path, monkeypatch):
    def bad_devices(*args):
        raise ValueError('expected preflight failure')
    monkeypatch.setattr(runner, '_validate_cuda_devices', bad_devices)
    monkeypatch.setattr(runner, '_check_inputs', forbidden)
    monkeypatch.setattr(shared, 'run_matrix', forbidden)
    with pytest.raises(ValueError, match='expected preflight'):
        runner.main(['--data-home', str(tmp_path), '--outdir', str(tmp_path / 'out')])


def test_export_only_never_invokes_training_fitting_or_cuda(matrix, monkeypatch):
    monkeypatch.setattr(runner, 'build_matrix', lambda *args, **kwargs: copy.deepcopy(matrix))
    monkeypatch.setattr(runner, '_validate_cuda_devices', forbidden)
    monkeypatch.setattr(runner, '_check_inputs', forbidden)
    monkeypatch.setattr(shared, 'run_matrix', forbidden)
    monkeypatch.setattr(trainer, 'train', forbidden)
    with pytest.raises(ValueError, match='Missing valid completed cells'):
        runner.main(['--data-home', 'absent', '--outdir', str(Path(matrix['cells'][0]['ckpt_dir']).parent),
                     '--devices', 'cpu', '--export-only'])


def test_launcher_forwards_arguments_and_never_mutates_global_cuda_or_kills_jobs():
    wrapper = (Path(__file__).resolve().parents[2] / 'remote_linux/11_run_paired_guard_drafts.sh').read_text()
    assert '--python)' in wrapper and '--data-home)' in wrapper
    assert 'run_paired_guard_drafts.py' in wrapper and '"${DOSE_FORWARDED[@]}"' in wrapper
    assert 'mmfi_paired_guard_s42_v1' in wrapper
    assert wrapper.index('DOSE_PY="${DOSE_PY:-${DOSE_DATA_HOME}') > wrapper.index('\ndone\n')
    assert not any(word in wrapper for word in ('git push', 'rm -', 'kill ', 'pkill', 'export CUDA_VISIBLE_DEVICES'))
