"""Isolated draft runner checks; tiny metadata/NumPy fixtures, no victim training."""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_method_drafts as runner
import train_backdoor as trainer
from attack.poison import PoisonedDataset
from data_utils.draft_subset import apply_draft_subset
from mmfi_tables import config_fingerprint


def _forbidden(*args, **kwargs):
    pytest.fail('Dry run or draft measurement attempted data, CUDA, or victim training')


class _TinyDataset:
    def __init__(self, root, count=24):
        self.data_root = str(root)
        self.items = [dict(csi=str(root / f'csi_{i:03d}.mat'),
                           kpt=str(root / f'pose_{i:03d}.npy'), frame_idx=i)
                      for i in range(count)]
        self.raw = np.random.default_rng(9).uniform(0.1, 0.8, (3, 114, 10)).astype(np.float32)
        self.raw_loads = 0

    def __len__(self):
        return len(self.items)

    def load_raw(self, path):
        self.raw_loads += 1
        return self.raw.copy()

    @staticmethod
    def normalize(raw):
        return np.clip(raw, 0, 1).astype(np.float32)


@pytest.fixture
def matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_source_provenance', lambda: {'fixture': 'a' * 64})
    return runner.build_matrix(tmp_path / 'data', tmp_path / 'drafts', 'cpu', 0,
                               epochs=3, train_samples=10, eval_samples=4,
                               distortion_samples=4)


@pytest.fixture
def peak_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_source_provenance', lambda: {'fixture': 'a' * 64})
    return runner.build_matrix(tmp_path / 'data', tmp_path / 'peak_drafts', 'cpu', 0,
                               epochs=3, train_samples=10, eval_samples=4,
                               distortion_samples=4, profile='peak_control')


def _write_completed(matrix):
    datasets = {}
    for cell in matrix['cells']:
        cfg = cell['cfg']
        action_path = Path(cfg['action_npy'])
        if not action_path.exists():
            action_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(action_path, np.random.default_rng(1).normal(
                size=(1, 3, 30, 25, 1)).astype(np.float32))
        action_sha = runner._file_sha(action_path)
        parent = _TinyDataset(Path(cfg['dataset_root']))
        train = apply_draft_subset(parent, cfg, 'train')
        test = apply_draft_subset(parent, cfg, 'test')
        datasets[cell['method_key']] = (train, test)
        fingerprint = config_fingerprint(cfg)
        poison = PoisonedDataset(train, runner._CleanIdentityTrigger(), mode='train',
            rho=cfg['rho'], seed=42, select='uniform', dataset='mmfi',
            dose_min=0.2, dose_max=1.0, pivot=1, theta_max_deg=40,
            axis=[0., 0., 1.], dose_mode='linear').manifest()
        poison['config_fingerprint'] = fingerprint
        res = dict(clean_mpjpe=0.025, clean_pampjpe=0.015,
            dose_grid=runner.GRID, reference_dose=1.0,
            tmpjpe=[0.025, 0.03, 0.035, 0.04, 0.045, 0.05],
            clean_to_target_tmpjpe=[0.025, 0.04, 0.055, 0.07, 0.085, 0.10],
            target_joints=[2, 3], config_fingerprint=fingerprint,
            draft_action_sha256=action_sha,
            poison_plan_sha256=poison['poison_plan_sha256'],
            training_contract='ordinary_erm', attacker_access='data_only',
            dose_coupling='paired', n_total=len(train), n_poison=poison['n_poison'])
        for threshold in (0.5, 0.4, 0.3, 0.2, 0.1):
            res[f'clean_pck@{threshold:.1f}'] = threshold + 0.4
        runner._atomic_json(cell['eval_cache'], dict(result_schema=trainer._RESULT_SCHEMA,
            cfg=cfg, cfg_fingerprint=fingerprint, trained_epochs=cfg['epochs'], res=res))
        runner._atomic_json(Path(cell['ckpt_dir']) / 'poison_manifest.json', poison)
        runner._atomic_json(Path(cell['ckpt_dir']) / 'draft_subsets.json',
            dict(status=runner.STATUS, config_fingerprint=fingerprint,
                 action_file_sha256=action_sha,
                 train=train.draft_subset_manifest(), eval=test.draft_subset_manifest()))
    return datasets


def test_plan_is_separate_and_has_exact_common_scientific_settings(matrix):
    assert matrix['status'] == runner.STATUS and matrix['DRAFT_ONLY'] is True
    assert [c['method_key'] for c in matrix['cells']] == [r[0] for r in runner.METHODS]
    assert matrix['plan_sha256'] == runner._plan_fingerprint(matrix)
    for cell in matrix['cells']:
        cfg = cell['cfg']
        assert cfg['method_draft'] is True and cfg['draft_profile'] == 'method_screening_v1'
        assert cfg['draft_train_samples'] == 10 and cfg['draft_eval_samples'] == 4
        assert cfg['draft_subset_seed'] == 0 and cfg['draft_eval_source'] == 'training_holdout'
        assert cfg['seed'] == 42 and cfg['model'] == 'hpeli' and cfg['optimizer'] == 'sgd'
        assert cfg['epochs'] == cfg['victim_epochs'] == 3
        assert cfg['lr'] == 0.001 and cfg['batch_size'] == 32
        assert cfg['rho'] == (0.0 if cell['method_key'] == 'clean' else 0.4)
        assert cfg['dose_min'] == 0.2 and cfg['dose_max'] == 1.0
        assert cfg['pivot'] == 1 and cfg['target_joints'] == [2, 3]
        assert cfg['theta_max_deg'] == 40 and cfg['payload_axis'] == [0., 0., 1.]
        assert cfg['training_protocol'] == 'ordinary_erm' and cfg['attacker_access'] == 'data_only'
        assert cfg['trigger_zero_mean'] is True and cfg['strict_resume'] is True
        assert cell['dependencies'] == [] and cell['tables'] == []


def test_peak_profile_trains_only_new_candidate_with_unchanged_scientific_budget(matrix, peak_matrix):
    assert peak_matrix['draft_profile'] == runner.PEAK_PROFILE
    assert [cell['method_key'] for cell in peak_matrix['cells']] == ['md_multicarrier_peak_matched']
    assert peak_matrix['metrics_contract']['reference_training'] is False
    assert peak_matrix['metrics_contract']['matches_l2_budget'] is False
    runner._validate_manifest(peak_matrix)
    original = matrix['cells'][1]['cfg']
    peak = peak_matrix['cells'][0]['cfg']
    for key in ('seed', 'epochs', 'victim_epochs', 'draft_train_samples', 'draft_eval_samples',
                'draft_subset_seed', 'draft_eval_source', 'optimizer', 'lr', 'momentum',
                'weight_decay', 'batch_size', 'rho', 'eps', 'pivot', 'target_joints',
                'theta_max_deg', 'payload_axis', 'dose_mode', 'dose_grid', 'dose_min',
                'dose_max', 'dose_coupling', 'training_protocol', 'attacker_access'):
        assert peak[key] == original[key], key
    assert 'original' not in [cell['method_key'] for cell in peak_matrix['cells']]


def test_default_profile_retains_original_five_cells_and_config_markers(matrix):
    assert matrix['draft_profile'] == 'method_screening_v1'
    assert [cell['method_key'] for cell in matrix['cells']] == [row[0] for row in runner.METHODS]
    assert all(cell['cfg']['draft_profile'] == 'method_screening_v1' for cell in matrix['cells'])


def test_unknown_profile_is_rejected_before_data_access(tmp_path):
    with pytest.raises(ValueError, match='Unknown method draft profile'):
        runner.build_matrix(tmp_path, tmp_path / 'out', profile='not-a-profile')
    assert not (tmp_path / 'out').exists()


def test_peak_manifest_cannot_mislabel_another_trigger(peak_matrix):
    peak_matrix['cells'][0]['cfg']['trigger'] = 'md_multicarrier'
    peak_matrix['plan_sha256'] = runner._plan_fingerprint(peak_matrix)
    with pytest.raises(ValueError, match='trigger does not match'):
        runner._validate_manifest(peak_matrix)


@pytest.mark.parametrize('profile,cells', [('screening', ['md_multicarrier_peak_matched']),
                                         ('peak_control', ['original'])])
def test_requested_cells_cannot_cross_profiles(tmp_path, profile, cells):
    with pytest.raises(ValueError, match='selected --profile'):
        runner.main(['--data-home', str(tmp_path), '--outdir', str(tmp_path / 'out'),
                     '--profile', profile, '--cells', *cells, '--dry-run'])
    assert not (tmp_path / 'out').exists()


def test_peak_dry_run_is_one_cell_without_data_cuda_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_check_inputs', _forbidden)
    monkeypatch.setattr(runner, 'run_matrix', _forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', _forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', _forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', _forbidden)
    output = tmp_path / 'out'
    assert runner.main(['--data-home', str(tmp_path), '--outdir', str(output),
                        '--profile', 'peak_control', '--devices', 'cuda:999', '--dry-run']) == 0
    manifest = json.loads((output / 'method_drafts.resolved.json').read_text())
    assert len(manifest['cells']) == 1
    assert manifest['draft_profile'] == runner.PEAK_PROFILE


@pytest.mark.parametrize('key,value', [('epochs', True), ('epochs', 0),
    ('train_samples', -2), ('eval_samples', False), ('distortion_samples', 0)])
def test_plan_rejects_bool_and_nonpositive_budgets(tmp_path, key, value):
    with pytest.raises(ValueError, match='positive integer'):
        runner.build_matrix(tmp_path, tmp_path / 'out', **{key: value})


def test_dry_run_never_loads_data_trigger_cuda_or_trains(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_check_inputs', _forbidden)
    monkeypatch.setattr(runner, 'run_matrix', _forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', _forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', _forbidden)
    monkeypatch.setattr(trainer, 'train', _forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', _forbidden)
    output = tmp_path / 'out'
    assert runner.main(['--data-home', str(tmp_path / 'missing-data'), '--outdir', str(output),
                        '--devices', 'cuda:99', '--dry-run', '--fresh']) == 0
    manifest = json.loads((output / 'method_drafts.resolved.json').read_text())
    assert manifest['status'] == runner.STATUS
    assert not (output / 'mmfi_matrix.resolved.json').exists()
    assert not (output / 'draft_summary.csv').exists()


def test_real_cli_dry_run_does_not_need_data_or_cuda(tmp_path):
    result = subprocess.run([sys.executable, str(runner.HERE / 'run_method_drafts.py'),
        '--data-home', str(tmp_path / 'missing-data'), '--outdir', str(tmp_path / 'out'),
        '--devices', 'cuda:999', '--fresh', '--dry-run'], capture_output=True,
        text=True, timeout=45, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert runner.STATUS in result.stdout


@pytest.mark.parametrize('extra', [['--devices', 'cpu', 'cpu'], ['--devices', 'cuda'],
    ['--devices', 'cuda:01'], ['--num-workers', '-1'], ['--epochs', '0'],
    ['--eval-samples', '-2'], ['--cells', 'clean', 'clean']])
def test_bad_runtime_and_budget_inputs_create_no_output(tmp_path, extra):
    output = tmp_path / 'out'
    with pytest.raises(ValueError):
        runner.main(['--data-home', str(tmp_path), '--outdir', str(output), '--dry-run', *extra])
    assert not output.exists()


@pytest.mark.parametrize('extra', [['--epochs', '4'], ['--train-samples', '11'],
                                  ['--eval-samples', '5'], ['--profile', 'peak_control']])
def test_resume_rejects_changed_scientific_budget_without_overwrite(matrix, extra):
    folder = Path(matrix['cells'][0]['ckpt_dir']).parent
    path = folder / 'method_drafts.resolved.json'
    runner._atomic_json(path, matrix)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='NEW output directory'):
        runner.main(['--data-home', str(Path(matrix['cells'][0]['cfg']['dataset_root']).parents[1]),
            '--outdir', str(folder), '--epochs', '3', '--train-samples', '10',
            '--eval-samples', '4', '--distortion-samples', '4', '--dry-run', *extra])
    assert path.read_bytes() == before


def test_resume_allows_device_and_worker_changes(matrix):
    folder = Path(matrix['cells'][0]['ckpt_dir']).parent
    path = folder / 'method_drafts.resolved.json'
    runner._atomic_json(path, matrix)
    assert runner.main(['--data-home', str(Path(matrix['cells'][0]['cfg']['dataset_root']).parents[1]),
        '--outdir', str(folder), '--epochs', '3', '--train-samples', '10', '--eval-samples', '4',
        '--distortion-samples', '4', '--devices', 'cuda:9', '--num-workers', '8', '--dry-run']) == 0
    stored = json.loads(path.read_text())
    assert stored['plan_sha256'] == matrix['plan_sha256']
    assert all(c['cfg']['device'] == 'cuda:9' and c['cfg']['num_workers'] == 8 for c in stored['cells'])


def test_fresh_refuses_nonempty_directory_without_deleting(matrix):
    folder = Path(matrix['cells'][0]['ckpt_dir']).parent
    path = folder / 'method_drafts.resolved.json'
    runner._atomic_json(path, matrix)
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match='NEW path'):
        runner.main(['--data-home', str(folder), '--outdir', str(folder), '--fresh', '--dry-run'])
    assert path.read_bytes() == before


def test_summary_numeric_contract_preserves_all_rows(matrix):
    _write_completed(matrix)
    report = runner.build_summary(matrix)
    assert report['status'] == runner.STATUS
    assert [r['method_key'] for r in report['rows']] == [r[0] for r in runner.METHODS]
    assert len(report['dose_response']) == 30
    for row in report['rows']:
        assert row['clean_mpjpe_mm'] == pytest.approx(25)
        assert row['clean_pampjpe_mm'] == pytest.approx(15)
        assert row['clean_pck_0.5_pct'] == pytest.approx(90)
        assert row['t1_mpjpe_mm'] == pytest.approx(50)
        assert row['mean_positive_dose_tmpjpe_mm'] == pytest.approx(40)
        assert row['i1_improvement_mm'] == pytest.approx(50)
    assert len(set(r['train_subset_sha256'] for r in report['rows'])) == 1
    assert len(set(r['eval_subset_sha256'] for r in report['rows'])) == 1
    assert len(set(r['poison_plan_sha256'] for r in report['rows'][1:])) == 1
    assert not set(report['audit']['train_parent_indices']).intersection(report['audit']['eval_parent_indices'])


def test_peak_summary_requires_only_its_valid_completed_cache(peak_matrix, tmp_path):
    _write_completed(peak_matrix)
    report = runner.build_summary(peak_matrix)
    assert report['draft_profile'] == runner.PEAK_PROFILE
    assert len(report['rows']) == 1 and len(report['dose_response']) == 6
    assert report['rows'][0]['method_key'] == 'md_multicarrier_peak_matched'
    assert len(report['audit']['poison_plan_sha256']) == 1
    output = tmp_path / 'reports'
    runner.export_summary(peak_matrix, output, skip_distortion=True)
    with (output / 'draft_summary.csv').open(newline='') as handle:
        assert len(list(csv.DictReader(handle))) == 1
    markdown = (output / 'draft_summary.md').read_text()
    assert runner.PEAK_PROFILE in markdown and 'no Original victim result is imported' in markdown


def test_peak_profile_audits_exact_same_subset_and_poison_plan_as_original(matrix, peak_matrix):
    _write_completed(matrix)
    _write_completed(peak_matrix)
    original = runner.build_summary(matrix)
    peak = runner.build_summary(peak_matrix)
    candidate_key = peak_matrix['cells'][0]['method_key']
    assert original['audit']['train_parent_indices'] == peak['audit']['train_parent_indices']
    assert original['audit']['eval_parent_indices'] == peak['audit']['eval_parent_indices']
    assert original['audit']['subsets']['original'] == peak['audit']['subsets'][candidate_key]
    assert original['audit']['poison_plan_sha256']['original'] == peak['audit']['poison_plan_sha256'][candidate_key]


@pytest.mark.parametrize('mutation', ['subset_profile', 'subset_hash', 'action_hash', 'cache_epochs'])
def test_single_peak_cell_still_rejects_bad_audit_or_cache(peak_matrix, mutation):
    _write_completed(peak_matrix)
    cell = peak_matrix['cells'][0]
    if mutation == 'cache_epochs':
        path = Path(cell['eval_cache'])
        blob = json.loads(path.read_text())
        blob['trained_epochs'] -= 1
    else:
        path = Path(cell['ckpt_dir']) / 'draft_subsets.json'
        blob = json.loads(path.read_text())
        if mutation == 'subset_profile':
            blob['eval']['profile'] = runner.PROFILE
        elif mutation == 'subset_hash':
            blob['eval']['index_sha256'] = 'b' * 64
        else:
            blob['action_file_sha256'] = 'b' * 64
    runner._atomic_json(path, blob)
    with pytest.raises(ValueError):
        runner.build_summary(peak_matrix)


@pytest.mark.parametrize('mutation', ['missing', 'old_schema', 'epochs', 'fingerprint'])
def test_bad_cache_never_exports_fabricated_summary(matrix, tmp_path, mutation):
    _write_completed(matrix)
    cache = Path(matrix['cells'][-1]['eval_cache'])
    blob = json.loads(cache.read_text())
    if mutation == 'missing':
        cache.unlink()
    else:
        if mutation == 'old_schema':
            blob['result_schema'] = 9
        elif mutation == 'epochs':
            blob['trained_epochs'] -= 1
        else:
            blob['cfg_fingerprint'] = 'edited'
        runner._atomic_json(cache, blob)
    export = tmp_path / 'report'
    with pytest.raises(ValueError):
        runner.export_summary(matrix, export, skip_distortion=True)
    assert not export.exists()


def test_modified_subset_identity_is_rejected(matrix):
    _write_completed(matrix)
    cell = matrix['cells'][-1]
    path = Path(cell['ckpt_dir']) / 'draft_subsets.json'
    blob = json.loads(path.read_text())
    blob['eval']['identifiers_sha256'] = 'b' * 64
    runner._atomic_json(path, blob)
    with pytest.raises(ValueError, match='same train/eval'):
        runner.build_summary(matrix)


@pytest.mark.parametrize('recorded_sha', [None, 'invalid-sha256', 'b' * 64])
def test_training_action_file_audit_is_required_and_shared(matrix, recorded_sha):
    _write_completed(matrix)
    path = Path(matrix['cells'][-1]['ckpt_dir']) / 'draft_subsets.json'
    record = json.loads(path.read_text())
    record['action_file_sha256'] = recorded_sha
    runner._atomic_json(path, record)
    with pytest.raises(ValueError, match='action-file'):
        runner.build_summary(matrix)


def test_cached_result_action_hash_must_match_training_record(matrix):
    _write_completed(matrix)
    path = Path(matrix['cells'][-1]['eval_cache'])
    cached = json.loads(path.read_text())
    cached['res']['draft_action_sha256'] = 'b' * 64
    runner._atomic_json(path, cached)
    with pytest.raises(ValueError, match='cached result metadata'):
        runner.build_summary(matrix)


def test_changed_action_file_is_rejected_before_distortion_data_or_trigger_load(matrix, monkeypatch):
    _write_completed(matrix)
    report = runner.build_summary(matrix)
    action = Path(matrix['cells'][0]['cfg']['action_npy'])
    np.save(action, np.random.default_rng(2).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    monkeypatch.setattr(trainer, '_load_dataset', _forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', _forbidden)
    with pytest.raises(ValueError, match='current action file differs'):
        runner.build_distortion(matrix, report['audit'], n=4)


def test_modified_poison_doses_are_rejected_even_with_rehashed_plan(matrix):
    _write_completed(matrix)
    cell = matrix['cells'][-1]
    path = Path(cell['ckpt_dir']) / 'poison_manifest.json'
    blob = json.loads(path.read_text())
    blob['samples'][0]['dose'] = 0.3
    blob['poison_plan_sha256'] = runner._sha_json([[s['index'], s['dose']] for s in blob['samples']])
    runner._atomic_json(path, blob)
    with pytest.raises(ValueError, match='same poison indices and doses'):
        runner.build_summary(matrix)


def test_trigger_hash_is_deterministic_and_tracks_nested_numpy_state():
    class Fixed:
        def __init__(self):
            self.base = {'p0': np.arange(5, dtype=np.float64)}
            self.carrier = np.eye(3, dtype=np.float32)
            self.rng = np.random.default_rng(42)

    first, second = Fixed(), Fixed()
    assert runner.trigger_state_sha256(first) == runner.trigger_state_sha256(second)
    second.carrier[0, 0] += 1
    assert runner.trigger_state_sha256(first) != runner.trigger_state_sha256(second)


def test_distortion_uses_real_fixed_variants_common_ids_and_strict_json(matrix, tmp_path, monkeypatch):
    action = tmp_path / 'action.npy'
    np.save(action, np.random.default_rng(1).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    for cell in matrix['cells']:
        cell['cfg']['action_npy'] = str(action)
    matrix['plan_sha256'] = runner._plan_fingerprint(matrix)
    datasets = _write_completed(matrix)
    monkeypatch.setattr(trainer, '_load_dataset', lambda cfg, split:
        datasets['clean' if cfg['rho'] == 0 else ('original' if cfg['trigger'] == 'micro_dropper' else cfg['trigger'])][1])
    monkeypatch.setattr(trainer, 'build_model', _forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', _forbidden)
    report = runner.build_summary(matrix)
    distortion = runner.build_distortion(matrix, report['audit'], n=4)
    assert len(distortion['rows']) == 30
    assert len(set(r['common_pair_ids_sha256'] for r in distortion['rows'])) == 1
    assert len(distortion['common_pair_ids']) == 4
    for row in distortion['rows']:
        if row['method_key'] == 'clean' or row['dose'] == 0:
            assert row['relative_l2'] == row['rmse'] == row['linf'] == 0
            assert row['snr_db'] is None and row['snr_db_is_infinite'] is True
    assert 'Infinity' not in json.dumps(distortion, allow_nan=False)
    assert 'NaN' not in json.dumps(distortion, allow_nan=False)


def test_peak_distortion_records_actual_original_pairs_and_zero_tolerance_bound(peak_matrix, tmp_path, monkeypatch):
    datasets = _write_completed(peak_matrix)
    cell = peak_matrix['cells'][0]
    monkeypatch.setattr(trainer, '_load_dataset', lambda cfg, split: datasets[cell['method_key']][1])
    monkeypatch.setattr(trainer, 'build_model', _forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', _forbidden)
    report = runner.build_summary(peak_matrix)
    distortion = runner.build_distortion(peak_matrix, report['audit'], n=4)
    assert len(distortion['rows']) == 6 and len(distortion['paired_samples']) == 6
    assert distortion['reference_training'] is False and distortion['matches_l2_budget'] is False
    for row, audit in zip(distortion['rows'], distortion['paired_samples']):
        assert row['linf'] <= row['original_linf']
        assert row['paired_linf_violations'] == 0 and row['max_paired_linf_excess'] <= 0
        assert len(audit['pairs']) == 4
        assert [pair['pair_id'] for pair in audit['pairs']] == distortion['common_pair_ids']
        assert all(pair['candidate_linf'] <= pair['original_linf'] for pair in audit['pairs'])
        assert all(pair['linf_excess'] <= 0 for pair in audit['pairs'])
        assert row['original_trigger_state_sha256']
        if row['dose'] == 0:
            assert row['linf'] == row['original_linf'] == 0
            assert row['original_snr_db'] is None and row['original_snr_db_is_infinite']
    assert 'Infinity' not in json.dumps(distortion, allow_nan=False)
    output = tmp_path / 'reports'
    runner.export_summary(peak_matrix, output, distortion_samples=4)
    with (output / 'input_distortion.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(runner.PEAK_DISTORTION_COLUMNS)
        assert len(list(reader)) == 6


def test_peak_pair_audit_rejects_local_violation_even_when_global_maximum_is_lower():
    class PairDataset:
        items = [{'csi': 'first'}, {'csi': 'second'}]

        def load_raw(self, path):
            return np.full((1, 1, 1), 0.25 if path == 'first' else 0.5, dtype=np.float32)

        @staticmethod
        def normalize(raw):
            return raw.copy()

    class Candidate:
        def inject(self, raw, dose, eps):
            return raw + (0.2 if raw.item() == 0.25 else 0.1)

    class Original:
        def inject(self, raw, dose, eps):
            return raw + (0.1 if raw.item() == 0.25 else 0.3)

    # The candidate global maximum is .2 < .3, but its first pair violates .1.
    with pytest.raises(ValueError, match='per-sample Linf exceeds Original'):
        runner._measure_peak_pair({'eps': 0.185}, PairDataset(), [0, 1],
                                 ['first', 'second'], 1.0, Candidate(), Original())


def test_summary_writes_draft_names_and_marker(matrix, tmp_path):
    _write_completed(matrix)
    output = tmp_path / 'reports'
    runner.export_summary(matrix, output, skip_distortion=True)
    assert {p.name for p in output.iterdir()} == {'draft_summary.csv', 'draft_summary.json', 'draft_summary.md'}
    assert runner.STATUS in (output / 'draft_summary.md').read_text()
    with (output / 'draft_summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5 and all(r['status'] == runner.STATUS for r in rows)


def test_subset_run_reports_pending_without_running_other_cells(matrix, monkeypatch, capsys):
    folder = Path(matrix['cells'][0]['ckpt_dir']).parent
    runner._atomic_json(folder / 'method_drafts.resolved.json', matrix)
    selections = []
    monkeypatch.setattr(runner, '_check_inputs', lambda *args: None)
    monkeypatch.setattr(runner, 'run_matrix', lambda matrix, path, devices, requested:
                        selections.append(requested))
    monkeypatch.setattr(runner, '_complete', lambda cell: cell['method_key'] == 'original')
    monkeypatch.setattr(runner, 'export_summary', _forbidden)
    assert runner.main(['--data-home', str(Path(matrix['cells'][0]['cfg']['dataset_root']).parents[1]),
        '--outdir', str(folder), '--devices', 'cpu', '--epochs', '3', '--train-samples', '10',
        '--eval-samples', '4', '--distortion-samples', '4', '--cells', 'original']) == 0
    assert selections == [['original']]
    assert 'Draft summary pending:' in capsys.readouterr().out


def test_linux_launcher_targets_draft_driver_and_forwards_flags():
    source = (runner.HERE.parent / 'remote_linux/06_run_method_drafts.sh').read_text()
    assert 'DRAFT_ONLY_NOT_PAPER_RESULTS' in source
    assert 'ATKBackd/run_method_drafts.py' in source
    assert '--data-home "${DOSE_DATA_HOME}" "$@"' in source
    assert 'DOSE_PY' in source and 'envs/dose-backdoor/bin/python' in source
    assert 'rm ' not in source and 'run_mmfi_tables.py' not in source


@pytest.mark.parametrize('profile', ['screening', 'peak_control'])
def test_actual_train_evaluate_all_five_cells_produce_auditable_summary(matrix, tmp_path, monkeypatch, profile):
    """Smoke each profile with one-batch CPU victim updates.

    Only the external MMFi constructor and expensive HPELi model are replaced.
    Dataset routing, subset selection, fixed triggers, poison labels, ERM,
    evaluation metrics, checkpoints, caches, and the cell dispatcher are real.
    """
    if os.name == 'nt' and os.environ.get('METHOD_DRAFT_SMOKE_CHILD') != '1':
        # NumPy/PyTorch may already have loaded different OpenMP runtimes in
        # the full suite. Initialize serial MKL before imports in this child.
        environment = os.environ.copy()
        environment.update(METHOD_DRAFT_SMOKE_CHILD='1', MKL_THREADING_LAYER='SEQUENTIAL',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [sys.executable, '-m', 'pytest',
            str(Path(__file__).resolve()) + f'::test_actual_train_evaluate_all_five_cells_produce_auditable_summary[{profile}]',
            '-q', '--tb=short', '--basetemp', str(tmp_path / 'isolated_cpu_smoke')]
        result = subprocess.run(command, env=environment, capture_output=True,
                                text=True, timeout=60, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        return

    import run_mmfi_tables as cell_runner

    if profile == 'peak_control':
        matrix = runner.build_matrix(tmp_path / 'data', tmp_path / 'peak_drafts', 'cpu', 0,
            epochs=3, train_samples=10, eval_samples=4, distortion_samples=4, profile=profile)

    torch = trainer.torch
    constructed_splits = []

    class SyntheticMMFI(_TinyDataset):
        def __init__(self, split, data_root, **kwargs):
            constructed_splits.append(split)
            assert split == 'training', 'Draft validation must not open the official test split'
            super().__init__(Path(data_root), count=25)
            self.split = split
            self.poses = np.random.default_rng(19).normal(0, 0.2, (25, 1, 17, 3)).astype(np.float32)

        def load_pose(self, path, frame_idx):
            return self.poses[frame_idx].copy()

    class TinyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3 * 114 * 10, 17 * 3)

        def forward(self, csi):
            return self.head(csi.flatten(1)).reshape(-1, 1, 17, 3), None

    action = tmp_path / 'synthetic_reference_action.npy'
    np.save(action, np.random.default_rng(23).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    for cell in matrix['cells']:
        cell['cfg'].update(action_npy=str(action), epochs=1, victim_epochs=1)
    matrix['plan_sha256'] = runner._plan_fingerprint(matrix)
    monkeypatch.setattr(trainer, 'MMFI', SyntheticMMFI)
    monkeypatch.setattr(trainer, 'build_model', lambda *args, **kwargs: TinyVictim())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *args: None)
    monkeypatch.setattr(torch.backends.cudnn, 'is_available', lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for cell in matrix['cells']:
            cell_runner._run_cell(cell)
            folder = Path(cell['ckpt_dir'])
            assert all((folder / name).is_file() for name in (
                'checkpoint.pt', 'results.json', 'eval_cache.json',
                'config.resolved.yaml', 'poison_manifest.json', 'draft_subsets.json'))
            checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
            assert checkpoint['epoch'] == 0
            assert checkpoint['cfg_fingerprint'] == config_fingerprint(cell['cfg'])
            assert checkpoint['model'] and checkpoint['optimizer']
            cached = json.loads((folder / 'eval_cache.json').read_text())
            dispatched = json.loads((folder / 'results.json').read_text())
            assert cached['trained_epochs'] == 1
            assert cached['cfg_fingerprint'] == dispatched['cfg_fingerprint']
            assert cached['res'] == dispatched['res']
            assert cached['res']['training_contract'] == 'ordinary_erm'
            assert cached['res']['attacker_access'] == 'data_only'

        report = runner.build_summary(matrix)
        cell_count = 5 if profile == 'screening' else 1
        assert len(report['rows']) == cell_count and len(report['dose_response']) == 6 * cell_count
        assert all(row['epochs'] == 1 and row['train_samples'] == 10 and row['eval_samples'] == 4
                   for row in report['rows'])
        attack_rows = [row for row in report['rows'] if row['method_key'] != 'clean']
        assert len(set(row['poison_plan_sha256'] for row in attack_rows)) == 1
        assert constructed_splits == ['training'] * 2 * cell_count
        if profile == 'peak_control':
            distortion = runner.build_distortion(matrix, report['audit'], n=4)
            assert all(row['paired_linf_violations'] == 0 for row in distortion['rows'])
            assert all(row['linf'] <= row['original_linf'] for row in distortion['rows'])

        # Completed real caches must return before dataset/model/trigger construction.
        monkeypatch.setattr(trainer, 'MMFI', _forbidden)
        monkeypatch.setattr(trainer, 'build_model', _forbidden)
        monkeypatch.setattr(trainer, 'build_trigger', _forbidden)
        resume_cell = matrix['cells'][1 if profile == 'screening' else 0]
        cell_runner._run_cell(resume_cell)
        resumed = runner.build_summary(matrix)
        assert resumed['rows'] == report['rows']
        stored_action_sha = resumed['audit']['action_file_sha256']
        np.save(action, np.random.default_rng(24).normal(
            size=(1, 3, 30, 25, 1)).astype(np.float32))
        assert runner._file_sha(action) != stored_action_sha
        with pytest.raises(ValueError, match='action'):
            cell_runner._run_cell(resume_cell)
    finally:
        torch.set_num_threads(previous_threads)
