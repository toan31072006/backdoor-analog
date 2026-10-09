"""Full MM-Fi confirmation planning and fail-closed resume checks; no real data."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_peak_confirmation as runner
import train_backdoor as trainer


METHOD_KEYS = ['original', 'md_multicarrier_peak_matched']
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _forbidden(*args, **kwargs):
    pytest.fail('Confirmation planning attempted dataset access, CUDA, or training')


def _args(data_home, output, *extra):
    return ['--data-home', str(data_home), '--outdir', str(output),
            '--devices', 'cpu', '--dry-run', *extra]


def _stored(output):
    return json.loads((output / 'peak_confirmation.resolved.json').read_text(encoding='utf-8'))


@pytest.fixture
def sources(monkeypatch):
    provenance = {'fixture.py': 'a' * 64}
    monkeypatch.setattr(runner, '_source_provenance', lambda: dict(provenance))
    return provenance


@pytest.fixture
def matrix(tmp_path, sources):
    return runner.build_matrix(tmp_path / 'missing_data', tmp_path / 'confirmation',
                               'cpu', 0)


def test_full_plan_has_only_the_ordered_original_and_peak_candidate(matrix):
    assert [cell['method_key'] for cell in matrix['cells']] == METHOD_KEYS
    assert [cell['cfg']['trigger'] for cell in matrix['cells']] == [
        'micro_dropper', 'md_multicarrier_peak_matched']
    assert matrix['dataset'] == 'mmfi' and matrix['seed'] == 42
    assert matrix['status'] == runner.STATUS
    assert 'DRAFT_ONLY' not in matrix['status']
    assert matrix.get('DRAFT_ONLY') is not True
    assert matrix['confirmation_profile'] == 'mmfi_peak_confirmation_v1'
    assert matrix['fresh_results_only'] is True
    assert matrix['plan_sha256'] == runner._plan_fingerprint(matrix)
    assert matrix['metrics_contract']['distortion_samples'] == 256
    for cell in matrix['cells']:
        assert cell['dependencies'] == [] and cell['tables'] == []
        assert Path(cell['ckpt_dir']).name == cell['method_key']
        assert Path(cell['eval_cache']) == Path(cell['ckpt_dir']) / 'eval_cache.json'
    runner._validate_manifest(matrix)


def test_all_scientific_settings_are_full_protocol1_s1_and_fixed(matrix):
    expected = dict(
        experiment_name='mmfi', model='hpeli', seed=42,
        mmfi_protocol='protocol1', mmfi_setting='s1',
        mmfi_random_ratio=0.8, mmfi_split_seed=0,
        epochs=50, victim_epochs=50, optimizer='sgd', lr=0.001,
        momentum=0.9, weight_decay=0.0, batch_size=32,
        rho=0.4, eps=0.185, poison_select='uniform',
        dose_min=0.2, dose_max=1.0, dose_grid=GRID,
        dose_coupling='paired', dose_mode='linear',
        pivot=1, target_joints=[2, 3], theta_max_deg=40.0,
        payload_axis=[0.0, 0.0, 1.0], trigger_zero_mean=True,
        training_protocol='ordinary_erm', attacker_access='data_only',
        threat_model='training_data_poisoning', strict_resume=True,
        pretrained=False, data_parallel=False,
    )
    for cell in matrix['cells']:
        cfg = cell['cfg']
        for key, value in expected.items():
            assert cfg[key] == value, (cell['method_key'], key)
        assert 'method_draft' not in cfg and 'draft_profile' not in cfg
        assert not any(key.startswith('draft_') or 'subset' in key for key in cfg)
        assert cfg['dataset_root'].endswith(str(Path('datasets') / 'Compress'))
        assert cfg['action_npy'].endswith(str(Path('actions') / 'data_bend.npy'))


def test_candidate_changes_only_trigger_and_its_recorded_parameters(matrix):
    original, candidate = [cell['cfg'] for cell in matrix['cells']]
    for key in original:
        if key != 'trigger':
            assert candidate[key] == original[key], key
    extra = set(candidate) - set(original)
    assert extra and all(key.startswith('method_') for key in extra)
    assert 'method_draft' not in extra
    assert candidate['method_carrier_seed'] == 42
    assert candidate['method_carrier_sub_mode'] == 3
    assert candidate['method_carrier_time_mode'] == 1


def test_matrix_configs_do_not_share_mutable_lists(matrix):
    original, candidate = [cell['cfg'] for cell in matrix['cells']]
    original['dose_grid'][1] = 0.123
    original['target_joints'].append(99)
    assert candidate['dose_grid'] == GRID
    assert candidate['target_joints'] == [2, 3]
    assert runner.GRID == GRID


def test_confirmation_configs_load_entire_official_train_and_test_splits(matrix, monkeypatch):
    calls = []

    class TinyOfficialSplit:
        def __init__(self, *, split, **kwargs):
            self.split = split
            self.items = list(range(7 if split == 'training' else 3))
            calls.append((self, kwargs))

        def __len__(self):
            return len(self.items)

    monkeypatch.setattr(trainer, 'MMFI', TinyOfficialSplit)
    for cell in matrix['cells']:
        train = trainer._load_dataset(cell['cfg'], 'training')
        test = trainer._load_dataset(cell['cfg'], 'test')
        assert train is calls[-2][0] and test is calls[-1][0]
        assert train.split == 'training' and test.split == 'test'
        assert len(train) == 7 and len(test) == 3
        for _, options in calls[-2:]:
            assert options['protocol'] == 'protocol1'
            assert options['setting'] == 's1'
            assert options['random_seed'] == 0


@pytest.mark.parametrize('mutation', ['config', 'trigger', 'status', 'profile',
                                     'method_draft', 'draft_profile', 'draft_subset',
                                     'order', 'missing_cell', 'duplicate_cell'])
def test_manifest_rejects_altered_config_markers_or_cells(matrix, mutation):
    changed = copy.deepcopy(matrix)
    if mutation == 'config':
        changed['cells'][0]['cfg']['lr'] = 0.009
    elif mutation == 'trigger':
        changed['cells'][1]['cfg']['trigger'] = 'md_multicarrier'
    elif mutation == 'status':
        changed['status'] = 'DRAFT_ONLY_NOT_PAPER_RESULTS'
    elif mutation == 'profile':
        changed['confirmation_profile'] = 'method_peak_control_v1'
    elif mutation == 'method_draft':
        changed['cells'][0]['cfg']['method_draft'] = True
    elif mutation == 'draft_profile':
        changed['cells'][0]['cfg']['draft_profile'] = 'method_peak_control_v1'
    elif mutation == 'draft_subset':
        changed['cells'][0]['cfg']['draft_train_samples'] = 20
    elif mutation == 'order':
        changed['cells'].reverse()
    elif mutation == 'missing_cell':
        changed['cells'].pop()
    else:
        changed['cells'].append(copy.deepcopy(changed['cells'][0]))
    # A matching editable hash must not legitimize a different experiment.
    if mutation != 'config':
        changed['plan_sha256'] = runner._plan_fingerprint(changed)
    with pytest.raises(ValueError):
        runner._validate_manifest(changed)


def test_dry_run_uses_no_data_trigger_cuda_or_training(tmp_path, sources, monkeypatch):
    monkeypatch.setattr(runner, '_check_inputs', _forbidden)
    monkeypatch.setattr(runner, 'run_matrix', _forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', _forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', _forbidden)
    monkeypatch.setattr(trainer, 'train', _forbidden)
    monkeypatch.setattr(trainer.np, 'load', _forbidden)
    monkeypatch.setattr(trainer.torch, 'load', _forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', _forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'get_device_properties', _forbidden)
    output = tmp_path / 'dry_run'
    assert runner.main(_args(tmp_path / 'missing_data', output, '--fresh')) == 0
    stored = _stored(output)
    assert [cell['method_key'] for cell in stored['cells']] == METHOD_KEYS
    runner._validate_manifest(stored)
    assert not (tmp_path / 'missing_data').exists()
    assert {path.name for path in output.iterdir()} == {
        'peak_confirmation.resolved.json', 'run.lock'}


def test_real_cli_dry_run_needs_no_dataset_or_gpu(tmp_path):
    script = Path(runner.__file__).resolve()
    output = tmp_path / 'cli_output'
    result = subprocess.run([sys.executable, str(script),
        *_args(tmp_path / 'missing_data', output, '--fresh')],
        capture_output=True, text=True, timeout=45, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [cell['method_key'] for cell in _stored(output)['cells']] == METHOD_KEYS
    assert not (tmp_path / 'missing_data').exists()


def test_same_plan_can_resume_without_fresh(tmp_path, sources):
    data, output = tmp_path / 'missing_data', tmp_path / 'resume'
    assert runner.main(_args(data, output, '--fresh')) == 0
    first = _stored(output)
    assert runner.main(_args(data, output)) == 0
    assert _stored(output)['plan_sha256'] == first['plan_sha256']


def test_runtime_changes_keep_the_same_scientific_plan(tmp_path, sources):
    data, output = tmp_path / 'missing_data', tmp_path / 'runtime_change'
    cpu = runner.build_matrix(data, output, 'cpu', 0)
    gpu = runner.build_matrix(data, output, 'cuda:7', 8)
    assert cpu['plan_sha256'] == gpu['plan_sha256']
    assert runner.main(_args(data, output, '--num-workers', '0')) == 0
    before = _stored(output)['plan_sha256']
    assert runner.main(_args(data, output, '--devices', 'cuda:7',
                             '--num-workers', '8')) == 0
    stored = _stored(output)
    assert stored['plan_sha256'] == before
    assert all(cell['cfg']['device'] == 'cuda:7' and cell['cfg']['num_workers'] == 8
               for cell in stored['cells'])


def test_fresh_refuses_nonempty_directory_without_touching_results(tmp_path, sources):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'checkpoint.pt'
    marker.write_bytes(b'existing user checkpoint')
    with pytest.raises(FileExistsError):
        runner.main(_args(tmp_path / 'data', output, '--fresh'))
    assert marker.read_bytes() == b'existing user checkpoint'
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize('extra', [
    ['--devices', 'cpu', 'cpu'], ['--devices', 'cuda'],
    ['--devices', 'cuda:01'], ['--devices', 'cuda:-1'],
    ['--num-workers', '-1'], ['--distortion-samples', '0'],
])
def test_invalid_runtime_options_create_no_output(tmp_path, sources, extra):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        runner.main(_args(tmp_path / 'data', output, *extra))
    assert not output.exists()


@pytest.mark.parametrize('name,value', [('distortion_samples', 0),
    ('distortion_samples', True), ('distortion_samples', 1.5),
    ('num_workers', -1), ('num_workers', True)])
def test_build_rejects_invalid_caps_or_workers(tmp_path, sources, name, value):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        runner.build_matrix(tmp_path / 'data', output, **{name: value})
    assert not output.exists()


@pytest.mark.parametrize('change', ['distortion_samples', 'source', 'data_home'])
def test_resume_rejects_changed_plan_without_overwriting(tmp_path, sources, change):
    data, output = tmp_path / 'data', tmp_path / 'resume_changed'
    assert runner.main(_args(data, output)) == 0
    path = output / 'peak_confirmation.resolved.json'
    before = path.read_bytes()
    extra = []
    if change == 'distortion_samples':
        extra = ['--distortion-samples', '257']
    elif change == 'source':
        sources['fixture.py'] = 'b' * 64
    else:
        data = tmp_path / 'other_data'
    with pytest.raises(ValueError, match='NEW|differs'):
        runner.main(_args(data, output, *extra))
    assert path.read_bytes() == before


def test_resume_rejects_modified_stored_config_without_overwriting(tmp_path, sources):
    data, output = tmp_path / 'data', tmp_path / 'edited_manifest'
    assert runner.main(_args(data, output)) == 0
    stored = _stored(output)
    stored['cells'][0]['cfg']['epochs'] = 49
    path = output / 'peak_confirmation.resolved.json'
    path.write_text(json.dumps(stored), encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(ValueError):
        runner.main(_args(data, output))
    assert path.read_bytes() == before


@pytest.fixture
def input_cell(matrix, tmp_path, monkeypatch):
    cell = copy.deepcopy(matrix['cells'][0])
    action = tmp_path / 'reference_action.npy'
    action.write_bytes(b'synthetic action reference version one')
    cell['cfg']['action_npy'] = str(action)

    class TinySplit:
        def __init__(self, split, count):
            self.data_root = str(tmp_path / 'dataset')
            self.items = [dict(
                csi=str(Path(self.data_root) / split / f'csi_{i}.mat'),
                kpt=str(Path(self.data_root) / split / f'pose_{i}.npy'), frame_idx=i)
                for i in range(count)]

        def __len__(self):
            return len(self.items)

    datasets = {'training': TinySplit('training', 5), 'test': TinySplit('test', 3)}
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', {'train': 5, 'eval': 3})
    monkeypatch.setattr(trainer, '_load_dataset', lambda cfg, split: datasets[split])
    monkeypatch.setattr(trainer.np, 'load', _forbidden)
    return cell, datasets, action


def test_input_record_binds_ordered_full_splits_action_and_config(input_cell):
    cell, _, action = input_cell
    record = runner._collect_input_record(cell)
    assert record['schema'] == 1 and record['profile'] == runner.PROFILE
    assert record['config_fingerprint'] == trainer._config_fingerprint(cell['cfg'])
    assert record['action_file_sha256'] == hashlib.sha256(action.read_bytes()).hexdigest()
    for name, split, count in [('train', 'training', 5), ('eval', 'test', 3)]:
        expected_ids = [[f'{split}/csi_{i}.mat', f'{split}/pose_{i}.npy', i]
                        for i in range(count)]
        assert record[name] == dict(n=count, pair_ids_sha256=runner._sha_json(expected_ids))


@pytest.mark.parametrize('change', ['wrong_count', 'duplicate', 'overlap'])
def test_input_record_refuses_incomplete_duplicate_or_overlapping_splits(input_cell, change):
    cell, datasets, _ = input_cell
    if change == 'wrong_count':
        datasets['training'].items.pop()
    elif change == 'duplicate':
        datasets['training'].items[1] = dict(datasets['training'].items[0])
    else:
        datasets['test'].items[0] = dict(datasets['training'].items[0])
    with pytest.raises(ValueError):
        runner._collect_input_record(cell)


def test_input_record_preserves_order_in_split_identity(input_cell):
    cell, datasets, _ = input_cell
    first = runner._collect_input_record(cell)
    datasets['test'].items.reverse()
    changed = runner._collect_input_record(cell)
    assert changed['train'] == first['train']
    assert changed['eval']['n'] == first['eval']['n']
    assert changed['eval']['pair_ids_sha256'] != first['eval']['pair_ids_sha256']


def test_binding_same_inputs_can_resume_without_rewriting_audit(input_cell):
    cell, _, _ = input_cell
    record = runner._collect_input_record(cell)
    runner._bind_inputs(cell, record)
    path = Path(cell['ckpt_dir']) / 'confirmation_inputs.json'
    before = path.read_bytes()
    assert json.loads(before) == record
    runner._bind_inputs(cell, runner._collect_input_record(cell))
    assert path.read_bytes() == before


def test_changed_action_file_cannot_resume_or_overwrite_input_audit(input_cell):
    cell, _, action = input_cell
    first = runner._collect_input_record(cell)
    runner._bind_inputs(cell, first)
    path = Path(cell['ckpt_dir']) / 'confirmation_inputs.json'
    before = path.read_bytes()
    action.write_bytes(b'synthetic action reference version two')
    changed = runner._collect_input_record(cell)
    assert changed['config_fingerprint'] == first['config_fingerprint']
    assert changed['action_file_sha256'] != first['action_file_sha256']
    with pytest.raises(ValueError, match='NEW|changed'):
        runner._bind_inputs(cell, changed)
    assert path.read_bytes() == before


@pytest.mark.parametrize('artifact', ['checkpoint.pt', 'eval_cache.json'])
def test_missing_input_audit_refuses_old_checkpoint_or_cache_import(input_cell, artifact):
    cell, _, _ = input_cell
    folder = Path(cell['ckpt_dir'])
    folder.mkdir(parents=True)
    old = folder / artifact
    old.write_bytes(b'unaudited historical result')
    with pytest.raises(ValueError, match='old caches|Missing confirmation input audit'):
        runner._bind_inputs(cell, runner._collect_input_record(cell))
    assert old.read_bytes() == b'unaudited historical result'
    assert not (folder / 'confirmation_inputs.json').exists()
