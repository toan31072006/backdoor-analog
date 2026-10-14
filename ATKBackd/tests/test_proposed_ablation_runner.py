"""Frozen best-Proposed ablation planning and resume guards; no benchmark data.

Synthetic metadata in these tests is deliberately not a scientific result.
The real runner must still enforce the full official MM-Fi split counts.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_peak_confirmation as confirmation
import run_proposed_ablation as runner
import train_backdoor as trainer


KEYS = ['full', 'w_o_coupling', 'w_o_auxiliary', 'w_o_peak']
TRIGGERS = ['md_multicarrier_peak_matched', 'md_multicarrier_peak_matched',
            'micro_dropper', 'md_multicarrier']
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def forbidden(*args, **kwargs):
    pytest.fail('Inert planning accessed data, CUDA, trigger fitting, or training')


def args(data_home, output, *extra):
    return ['--data-home', str(data_home), '--outdir', str(output),
            '--devices', 'cpu', '--num-workers', '0', '--dry-run', *extra]


def stored(output):
    return json.loads((output / 'proposed_ablation.resolved.json').read_text(encoding='utf-8'))


@pytest.fixture
def sources(monkeypatch):
    hashes = {'fixture.py': 'a' * 64}
    monkeypatch.setattr(runner, '_source_provenance', lambda: dict(hashes))
    return hashes


@pytest.fixture
def matrix(tmp_path, sources):
    return runner.build_matrix(tmp_path / 'missing_data', tmp_path / 'ablation', 'cpu', 0)


def test_plan_has_exactly_four_fresh_arms_of_the_same_frozen_proposed(matrix):
    assert runner.PROFILE == 'mmfi_proposed_ablation_v1'
    assert runner.STATUS == 'FULL_MMFI_ABLATION_SINGLE_SEED'
    assert runner.EXPECTED_COUNTS == {'train': 133056, 'eval': 33264}
    assert [cell['method_key'] for cell in matrix['cells']] == KEYS
    assert [cell['cfg']['trigger'] for cell in matrix['cells']] == TRIGGERS
    assert matrix['dataset'] == 'mmfi' and matrix['seed'] == 42
    assert matrix['status'] == runner.STATUS
    assert matrix['confirmation_profile'] == runner.PROFILE
    assert matrix['fresh_results_only'] is True
    assert matrix['plan_sha256'] == runner._plan_fingerprint(matrix)
    assert matrix['metrics_contract']['distortion_samples'] == 256
    assert matrix['metrics_contract']['dose_grid'] == GRID
    for cell in matrix['cells']:
        assert cell['dependencies'] == [] and cell['tables'] == []
        assert Path(cell['ckpt_dir']).name == cell['method_key']
        assert Path(cell['eval_cache']) == Path(cell['ckpt_dir']) / 'eval_cache.json'
    runner._validate_manifest(matrix)


def test_full_matches_the_best_full_confirmation_except_declared_metric_profile(matrix):
    cfg = matrix['cells'][0]['cfg']
    data_home = Path(cfg['dataset_root']).parent.parent
    previous = trainer._resolve_training_config(dict(
        confirmation._base_config(data_home, 'cpu', 0),
        trigger='md_multicarrier_peak_matched'))
    previous.update(confirmation_profile=runner.PROFILE, measure_dose_control=True)
    assert cfg == previous


def test_scientific_protocol_is_full_rho_point_four_50_epochs_for_every_arm(matrix):
    common = dict(experiment_name='mmfi', model='hpeli', seed=42,
        mmfi_protocol='protocol1', mmfi_setting='s1',
        mmfi_random_ratio=.8, mmfi_split_seed=0,
        epochs=50, victim_epochs=50, optimizer='sgd', lr=.001,
        momentum=.9, weight_decay=0.0, batch_size=32,
        rho=.4, eps=.185, poison_select='uniform', dose_min=.2, dose_max=1.0,
        dose_grid=GRID, dose_mode='linear', pivot=1, target_joints=[2, 3],
        theta_max_deg=40.0, payload_axis=[0.0, 0.0, 1.0],
        trigger_zero_mean=True, measure_dose_control=True,
        training_protocol='ordinary_erm', attacker_access='data_only',
        threat_model='training_data_poisoning', strict_resume=True,
        pretrained=False, data_parallel=False)
    for cell in matrix['cells']:
        for key, value in common.items():
            assert cell['cfg'][key] == value, (cell['method_key'], key)
        assert not any(k == 'method_draft' or k.startswith('draft_')
                       or 'subset' in k for k in cell['cfg'])
        assert cell['cfg']['dataset_root'].endswith(str(Path('datasets') / 'Compress'))
        assert cell['cfg']['action_npy'].endswith(str(Path('actions') / 'data_bend.npy'))


def test_each_ablation_changes_only_its_named_component(matrix):
    full, no_coupling, no_auxiliary, no_peak = [cell['cfg'] for cell in matrix['cells']]
    assert no_coupling == dict(full, dose_coupling='shuffled')
    assert no_peak == dict(full, trigger='md_multicarrier')
    assert no_auxiliary['dose_coupling'] == 'paired'
    assert no_auxiliary['trigger'] == 'micro_dropper'
    # Method parameters apply only to the multicarrier operators. No new
    # victim loss, dose distribution, poison amount, or optimizer is allowed.
    assert {k: v for k, v in full.items() if not k.startswith('method_') and k != 'trigger'} == {
        k: v for k, v in no_auxiliary.items() if not k.startswith('method_') and k != 'trigger'}
    assert all(cfg['method_carrier_seed'] == 42 for cfg in (full, no_coupling, no_peak))
    assert all(cfg['method_carrier_sub_mode'] == 3 for cfg in (full, no_coupling, no_peak))
    assert all(cfg['method_carrier_time_mode'] == 1 for cfg in (full, no_coupling, no_peak))


def test_mutable_config_fields_are_independent_between_arms(matrix):
    matrix['cells'][0]['cfg']['dose_grid'][1] = .123
    matrix['cells'][0]['cfg']['target_joints'].append(99)
    for cell in matrix['cells'][1:]:
        assert cell['cfg']['dose_grid'] == GRID
        assert cell['cfg']['target_joints'] == [2, 3]
    assert runner.GRID == GRID


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_future_replications_change_victim_poison_seed_not_frozen_carrier(tmp_path, sources, seed):
    matrix = runner.build_matrix(tmp_path / 'data', tmp_path / f'seed{seed}',
                                 'cpu', 0, seed=seed)
    assert matrix['seed'] == seed
    assert matrix['metrics_contract']['seeds'] == [seed]
    assert all(cell['cfg']['seed'] == seed for cell in matrix['cells'])
    assert all(cell['cfg']['method_carrier_seed'] == 42
               for cell in matrix['cells'] if cell['method_key'] != 'w_o_auxiliary')
    runner._validate_manifest(matrix)


@pytest.mark.parametrize('key,value', [
    ('epochs', 15), ('victim_epochs', 30), ('rho', .1), ('eps', .3),
    ('optimizer', 'adamw'), ('lr', .01), ('lr_scheduler', True),
    ('dose_min', 1.0), ('dose_max', .8), ('dose_grid', [0.0, 1.0]),
    ('dose_coupling', 'shuffled'), ('pivot', 0), ('target_joints', [4, 5]),
    ('theta_max_deg', 60.0), ('payload_axis', [1.0, 0.0, 0.0]),
    ('method_carrier_seed', 43), ('method_carrier_sub_mode', 2),
    ('pretrained', True), ('measure_dose_control', False),
    ('training_protocol', 'trigger_optimizer'), ('attacker_access', 'victim_gradients'),
    ('strict_resume', False), ('draft_train_samples', 20000), ('method_draft', True),
])
def test_manifest_rejects_scientific_mutation_even_with_recomputed_hash(matrix, key, value):
    changed = copy.deepcopy(matrix)
    changed['cells'][0]['cfg'][key] = value
    changed['plan_sha256'] = runner._plan_fingerprint(changed)
    with pytest.raises(ValueError):
        runner._validate_manifest(changed)


@pytest.mark.parametrize('change', ['order', 'missing', 'duplicate', 'status',
                                   'profile', 'dependency', 'table'])
def test_manifest_cannot_drop_reorder_or_import_historical_arms(matrix, change):
    altered = copy.deepcopy(matrix)
    if change == 'order':
        altered['cells'].reverse()
    elif change == 'missing':
        altered['cells'].pop()
    elif change == 'duplicate':
        altered['cells'].append(copy.deepcopy(altered['cells'][0]))
    elif change == 'status':
        altered['status'] = 'DRAFT_ONLY_NOT_PAPER_RESULTS'
    elif change == 'profile':
        altered['confirmation_profile'] = confirmation.PROFILE
    elif change == 'dependency':
        altered['cells'][0]['dependencies'] = ['old_full']
    else:
        altered['cells'][0]['tables'] = [1]
    altered['plan_sha256'] = runner._plan_fingerprint(altered)
    with pytest.raises(ValueError):
        runner._validate_manifest(altered)


def test_dry_run_needs_no_data_cuda_trigger_or_training(tmp_path, sources, monkeypatch):
    monkeypatch.setattr(runner, '_check_inputs', forbidden)
    monkeypatch.setattr(runner, 'run_matrix', forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', forbidden)
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(trainer.np, 'load', forbidden)
    monkeypatch.setattr(trainer.torch, 'load', forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'get_device_properties', forbidden)
    output = tmp_path / 'dry_run'
    assert runner.main(args(tmp_path / 'missing', output, '--fresh',
                            '--devices', 'cuda:999')) == 0
    assert not (tmp_path / 'missing').exists()
    assert {p.name for p in output.iterdir()} == {
        'proposed_ablation.resolved.json', 'run.lock'}
    assert [cell['method_key'] for cell in stored(output)['cells']] == KEYS
    runner._validate_manifest(stored(output))


def test_real_cli_dry_run_requires_neither_benchmark_data_nor_gpu(tmp_path):
    output = tmp_path / 'cli_output'
    result = subprocess.run([sys.executable, str(Path(runner.__file__).resolve()),
        *args(tmp_path / 'missing', output, '--fresh', '--devices', 'cuda:999')],
        capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert runner.STATUS in result.stdout
    assert [cell['method_key'] for cell in stored(output)['cells']] == KEYS
    assert not (tmp_path / 'missing').exists()


def test_seed_flag_is_recorded_and_default_is_42(tmp_path, sources):
    assert runner.main(args(tmp_path / 'data', tmp_path / 'default')) == 0
    assert stored(tmp_path / 'default')['seed'] == 42
    assert runner.main(args(tmp_path / 'data', tmp_path / 'next', '--seed', '43')) == 0
    assert stored(tmp_path / 'next')['seed'] == 43


def test_device_worker_changes_preserve_plan_and_safe_resume(tmp_path, sources):
    data, output = tmp_path / 'data', tmp_path / 'resume'
    cpu = runner.build_matrix(data, output, 'cpu', 0)
    gpu = runner.build_matrix(data, output, 'cuda:7', 8)
    assert cpu['plan_sha256'] == gpu['plan_sha256']
    assert runner.main(args(data, output, '--fresh')) == 0
    previous = stored(output)['plan_sha256']
    assert runner.main(args(data, output, '--devices', 'cuda:7', '--num-workers', '8')) == 0
    resumed = stored(output)
    assert resumed['plan_sha256'] == previous
    assert all(c['cfg']['device'] == 'cuda:7' and c['cfg']['num_workers'] == 8
               for c in resumed['cells'])


def test_fresh_does_not_remove_existing_result_or_checkpoint(tmp_path, sources):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'checkpoint.pt'
    marker.write_bytes(b'user checkpoint')
    with pytest.raises(FileExistsError):
        runner.main(args(tmp_path / 'data', output, '--fresh'))
    assert marker.read_bytes() == b'user checkpoint'
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize('change', ['source', 'seed', 'data_home', 'distortion_samples'])
def test_resume_refuses_changed_identity_without_overwriting(tmp_path, sources, change):
    data, output = tmp_path / 'data', tmp_path / 'changed'
    assert runner.main(args(data, output)) == 0
    manifest = output / 'proposed_ablation.resolved.json'
    before = manifest.read_bytes()
    extra = []
    if change == 'source':
        sources['fixture.py'] = 'b' * 64
    elif change == 'seed':
        extra = ['--seed', '43']
    elif change == 'data_home':
        data = tmp_path / 'other_data'
    else:
        extra = ['--distortion-samples', '257']
    with pytest.raises(ValueError):
        runner.main(args(data, output, *extra))
    assert manifest.read_bytes() == before


@pytest.mark.parametrize('extra', [
    ['--devices', 'cpu', 'cpu'], ['--devices', 'cuda'],
    ['--devices', 'cuda:01'], ['--devices', 'cuda:-1'],
    ['--num-workers', '-1'], ['--distortion-samples', '0'], ['--seed', '-1'],
])
def test_invalid_runtime_options_create_no_output(tmp_path, sources, extra):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        runner.main(args(tmp_path / 'data', output, *extra))
    assert not output.exists()


@pytest.mark.parametrize('flag', ['--epochs', '--rho', '--train-samples', '--eval-samples'])
def test_cli_cannot_request_shortened_or_capped_scientific_protocol(tmp_path, flag):
    output = tmp_path / 'capped'
    with pytest.raises(SystemExit):
        runner.main(args(tmp_path / 'data', output, flag, '15'))
    assert not output.exists()


def test_summary_only_never_accesses_cuda_or_schedules_training(tmp_path, sources, monkeypatch):
    # Existing manifest is inert; incomplete results must fail closed instead
    # of launching jobs during a status/export request.
    output = tmp_path / 'summary'
    assert runner.main(args(tmp_path / 'data', output)) == 0
    monkeypatch.setattr(runner, '_check_inputs', forbidden)
    monkeypatch.setattr(runner, 'run_matrix', forbidden)
    monkeypatch.setattr(runner, '_complete', lambda cell: False)
    with pytest.raises(ValueError):
        runner.main(['--data-home', str(tmp_path / 'data'), '--outdir', str(output),
                     '--devices', 'cpu', '--summary-only'])
    assert not (output / 'ablation_summary.json').exists()


@pytest.fixture
def input_cell(matrix, tmp_path, monkeypatch):
    cell = copy.deepcopy(matrix['cells'][0])
    action = tmp_path / 'action.npy'
    action.write_bytes(b'fixed reference bytes; metadata tests never decode NumPy')
    cell['cfg']['action_npy'] = str(action)

    class TinySplit:
        def __init__(self, split, count):
            self.data_root = str(tmp_path / 'dataset')
            self.items = [dict(
                csi=str(Path(self.data_root) / split / f'csi_{i}.npy'),
                kpt=str(Path(self.data_root) / split / 'ground_truth.npy'),
                frame_idx=i) for i in range(count)]

        def __len__(self):
            return len(self.items)

    datasets = {'training': TinySplit('training', 5), 'test': TinySplit('test', 3)}
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', {'train': 5, 'eval': 3})
    monkeypatch.setattr(trainer, '_load_dataset', lambda cfg, split: datasets[split])
    monkeypatch.setattr(trainer.np, 'load', forbidden)
    return cell, datasets, action


def test_input_passport_binds_own_profile_full_ordered_ids_and_reference(input_cell):
    cell, _, action = input_cell
    record = runner._collect_input_record(cell)
    assert record['schema'] == 1 and record['profile'] == runner.PROFILE
    assert record['config_fingerprint'] == trainer._config_fingerprint(cell['cfg'])
    assert record['action_file_sha256'] == hashlib.sha256(action.read_bytes()).hexdigest()
    for name, split, count in [('train', 'training', 5), ('eval', 'test', 3)]:
        ids = [[f'{split}/csi_{i}.npy', f'{split}/ground_truth.npy', i]
               for i in range(count)]
        assert record[name] == dict(n=count, pair_ids_sha256=runner._sha_json(ids))


@pytest.mark.parametrize('change', ['wrong_count', 'duplicate', 'overlap'])
def test_input_passport_rejects_incomplete_duplicate_or_overlapping_splits(input_cell, change):
    cell, datasets, _ = input_cell
    if change == 'wrong_count':
        datasets['training'].items.pop()
    elif change == 'duplicate':
        datasets['training'].items[1] = dict(datasets['training'].items[0])
    else:
        datasets['test'].items[0] = dict(datasets['training'].items[0])
    with pytest.raises(ValueError):
        runner._collect_input_record(cell)


def test_rebinding_same_inputs_does_not_rewrite_passport(input_cell):
    cell, _, _ = input_cell
    runner._bind_inputs(cell, runner._collect_input_record(cell))
    path = Path(cell['ckpt_dir']) / 'ablation_inputs.json'
    before = path.read_bytes()
    runner._bind_inputs(cell, runner._collect_input_record(cell))
    assert path.read_bytes() == before


@pytest.mark.parametrize('change', ['action', 'ordered_ids'])
def test_changed_input_identity_does_not_overwrite_existing_passport(input_cell, change):
    cell, datasets, action = input_cell
    runner._bind_inputs(cell, runner._collect_input_record(cell))
    path = Path(cell['ckpt_dir']) / 'ablation_inputs.json'
    before = path.read_bytes()
    if change == 'action':
        action.write_bytes(b'different reference version')
    else:
        datasets['test'].items.reverse()
    with pytest.raises(ValueError):
        runner._bind_inputs(cell, runner._collect_input_record(cell))
    assert path.read_bytes() == before


@pytest.mark.parametrize('artifact', ['checkpoint.pt', 'eval_cache.json'])
def test_unaudited_historical_checkpoint_or_cache_is_not_imported(input_cell, artifact):
    cell, _, _ = input_cell
    folder = Path(cell['ckpt_dir'])
    folder.mkdir(parents=True)
    old = folder / artifact
    old.write_bytes(b'unaudited historical victim; not a new ablation result')
    with pytest.raises(ValueError):
        runner._bind_inputs(cell, runner._collect_input_record(cell))
    assert old.read_bytes() == b'unaudited historical victim; not a new ablation result'
    assert not (folder / 'ablation_inputs.json').exists()


@pytest.fixture
def audited_matrix(tmp_path, sources, monkeypatch):
    """Construct only actual poison RNG metadata; no CSI or pose is read."""
    from attack.poison import PoisonedDataset

    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', {'train': 10, 'eval': 3})
    matrix = runner.build_matrix(tmp_path / 'data', tmp_path / 'audit', 'cpu', 0)

    class Base:
        items = [dict(csi=f'train/csi_{i}.npy', kpt='train/pose.npy', frame_idx=i)
                 for i in range(10)]

        def __len__(self):
            return len(self.items)

    for cell in matrix['cells']:
        folder = Path(cell['ckpt_dir'])
        folder.mkdir(parents=True)
        cfg = cell['cfg']
        record = dict(schema=1, profile=runner.PROFILE,
            config_fingerprint=trainer._config_fingerprint(cfg),
            action_file_sha256='c' * 64,
            train=dict(n=10, pair_ids_sha256='a' * 64),
            eval=dict(n=3, pair_ids_sha256='b' * 64))
        (folder / 'ablation_inputs.json').write_text(json.dumps(record), encoding='utf-8')
        poisoned = PoisonedDataset(Base(), object(), rho=.4, dose_min=.2,
            dose_max=1.0, eps=.185, pivot=1, theta_max_deg=40.0,
            seed=42, dataset='mmfi', dose_coupling=cfg['dose_coupling'])
        manifest = poisoned.manifest()
        manifest['config_fingerprint'] = trainer._config_fingerprint(cfg)
        (folder / 'poison_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return matrix


def test_poison_audit_checks_shared_indices_trigger_doses_and_correct_payload_rng(audited_matrix):
    audit = runner.audit_common_inputs(audited_matrix)
    assert list(audit['inputs']) == KEYS
    assert len(set(audit['poison_plan_sha256'].values())) == 1
    assignments = audit['payload_assignment_sha256']
    assert assignments['full'] == assignments['w_o_auxiliary'] == assignments['w_o_peak']
    assert assignments['w_o_coupling'] != assignments['full']


@pytest.mark.parametrize('change', ['payload_pairing', 'payload_marginal', 'seed',
                                   'cover', 'train_identity', 'action_identity'])
def test_poison_audit_refuses_hidden_intervention_changes(audited_matrix, change):
    cell = audited_matrix['cells'][1]
    folder = Path(cell['ckpt_dir'])
    if change in ('train_identity', 'action_identity'):
        path = folder / 'ablation_inputs.json'
        record = json.loads(path.read_text())
        if change == 'train_identity':
            record['train']['pair_ids_sha256'] = 'd' * 64
        else:
            record['action_file_sha256'] = 'd' * 64
    else:
        path = folder / 'poison_manifest.json'
        record = json.loads(path.read_text())
        if change == 'payload_pairing':
            for sample in record['samples']:
                sample['payload_dose'] = sample['dose']
        elif change == 'payload_marginal':
            record['samples'][0]['payload_dose'] = .21
        elif change == 'seed':
            record['seed'] = 43
        else:
            record['cover_ratio'] = .2
            record['n_cover'] = 2
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError):
        runner.audit_common_inputs(audited_matrix)


def test_real_cpu_four_cell_training_cache_and_export_are_consistent(tmp_path, monkeypatch):
    """Exercise 50 epochs on tiny synthetic frames, never a benchmark claim."""
    import numpy as np
    import torch

    constructed = []

    class TinyMMFI:
        def __init__(self, split, data_root, **kwargs):
            constructed.append(split)
            assert split in ('training', 'test')
            self.data_root = str(data_root)
            count = 10 if split == 'training' else 3
            self.items = [dict(csi=str(Path(data_root) / split / f'csi_{i}.npy'),
                kpt=str(Path(data_root) / split / 'ground_truth.npy'), frame_idx=i,
                name=f'{split}_{i}') for i in range(count)]
            self.raw = np.random.default_rng(9).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
            self.poses = np.random.default_rng(19).normal(0, .2, (10, 1, 17, 3)).astype(np.float32)

        def __len__(self):
            return len(self.items)

        def load_raw(self, path):
            return self.raw.copy()

        @staticmethod
        def normalize(raw):
            return np.clip(raw, 0, 1).astype(np.float32)

        def load_pose(self, path, frame_idx):
            return self.poses[frame_idx].copy()

    class TinyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3 * 114 * 10, 17 * 3)

        def forward(self, csi):
            return self.head(csi.flatten(1)).reshape(-1, 1, 17, 3), None

    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', {'train': 10, 'eval': 3})
    monkeypatch.setattr(trainer, 'MMFI', TinyMMFI)
    monkeypatch.setattr(trainer, 'build_model', lambda *a, **k: TinyVictim())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *args: None)
    monkeypatch.setattr(torch.backends.cudnn, 'is_available', lambda: False)
    data, output = tmp_path / 'data', tmp_path / 'real_cpu_route'
    action = data / 'actions/data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action, np.random.default_rng(23).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    matrix = runner.build_matrix(data, output, 'cpu', 0, distortion_samples=3)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for cell in matrix['cells']:
            runner._run_cell(cell)
            folder = Path(cell['ckpt_dir'])
            checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False, map_location='cpu')
            assert checkpoint['epoch'] == 49
            assert checkpoint['cfg']['epochs'] == 50
            assert not (folder / 'draft_subsets.json').exists()
            assert (folder / 'ablation_inputs.json').is_file()
            cache = json.loads((folder / 'eval_cache.json').read_text())
            assert cache['res']['dose_grid'] == GRID
            assert len(cache['res']['dose_control_error']) == 6
            assert cache['res']['dose_control_error'][0] == 0.0

        report = runner.export_summary(matrix, output)
        assert report['status'] == 'proposed_ablation_complete'
        assert len(report['rows']) == 4 and len(report['dose_response']) == 24
        assert [row['method_key'] for row in report['rows']] == KEYS
        assert all(row['train_samples'] == 10 and row['eval_samples'] == 3
                   and row['epochs'] == 50 for row in report['rows'])
        assert len(set(row['poison_plan_sha256'] for row in report['rows'])) == 1
        assignments = report['audit']['payload_assignment_sha256']
        assert assignments['full'] == assignments['w_o_auxiliary'] == assignments['w_o_peak']
        assert assignments['w_o_coupling'] != assignments['full']
        for key in KEYS:
            curve = [row['dose_control_error_mm'] for row in report['dose_response']
                     if row['method_key'] == key]
            summary = next(row for row in report['rows'] if row['method_key'] == key)
            assert summary['mean_dose_control_error_mm'] == pytest.approx(sum(curve[1:]) / 5)
        expected_files = ('ablation_summary.csv', 'ablation_summary.json',
            'ablation_summary.md', 'dose_response.csv', 'dose_response.json',
            'input_distortion.csv', 'input_distortion.json', 'dose_response.png')
        assert all((output / name).is_file() for name in expected_files)
        distortion = json.loads((output / 'input_distortion.json').read_text())
        assert len(distortion['rows']) == 24
        bounded = [row for row in distortion['rows'] if row['method_key'] != 'w_o_peak']
        assert all(row['paired_linf_violations'] == 0
                   and row['linf'] <= row['original_linf'] for row in bounded)
        assert all(not row['peak_bound_required'] for row in distortion['rows']
                   if row['method_key'] == 'w_o_peak')
        assert set(constructed) == {'training', 'test'}

        # Completed audited cache reuse must not initialize/retrain a victim.
        monkeypatch.setattr(trainer, 'build_model', forbidden)
        runner._run_cell(matrix['cells'][0])
        assert runner.build_summary(matrix)['rows'] == report['rows']

        # Rewriting the shuffled arm to paired labels must not legitimize an
        # old result, even when the trigger-dose plan remains untouched.
        path = output / 'w_o_coupling/poison_manifest.json'
        poison = json.loads(path.read_text())
        for sample in poison['samples']:
            sample['payload_dose'] = sample['dose']
        path.write_text(json.dumps(poison), encoding='utf-8')
        with pytest.raises(ValueError, match='payload-dose assignment'):
            runner.build_summary(matrix)
    finally:
        torch.set_num_threads(previous_threads)
