"""Full simple-method plan and fail-closed scientific metadata; no data fitting."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_simple_full as runner
import simple_full_contract as contract
import train_backdoor as trainer


KEYS = ['clean', 'blended', 'proposed', 'additive', 'single',
        'shared_links', 'frequency_only']
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def forbidden(*args, **kwargs):
    pytest.fail('Planning or an invalid configuration touched data, a cache, CUDA, or training')


def args(data_home, output, *extra):
    return ['--data-home', str(data_home), '--outdir', str(output),
            '--devices', 'cpu', '--num-workers', '0', *extra]


def stored(output):
    return json.loads((output / 'simple_full.resolved.json').read_text(encoding='utf-8'))


@pytest.fixture
def sources(monkeypatch):
    values = {'fixture.py': 'a' * 64}
    monkeypatch.setattr(runner, '_source_provenance', lambda: dict(values))
    return values


@pytest.fixture
def matrix(tmp_path, sources):
    return runner.build_matrix(tmp_path / 'data', tmp_path / 'simple_full', 'cpu', 0)


def test_plan_retains_all_seven_fresh_full_cells_in_fixed_order(matrix):
    assert runner.PROFILE == contract.PROFILE == 'mmfi_simple_full_v1'
    assert runner.STATUS == contract.STATUS == 'FULL_EXPLORATORY_SINGLE_SEED'
    assert runner.EXPECTED_COUNTS == contract.EXPECTED_COUNTS == {
        'train': 133056, 'eval': 33264}
    assert runner.GRID == GRID
    assert matrix['simple_full_profile'] == runner.PROFILE
    assert matrix['status'] == runner.STATUS
    assert matrix['dataset'] == 'mmfi' and matrix['seed'] == 42
    assert matrix['fresh_results_only'] is True
    assert not matrix.get('DRAFT_ONLY')
    assert [cell['method_key'] for cell in matrix['cells']] == KEYS
    assert matrix['plan_sha256'] == runner._plan_fingerprint(matrix)
    for cell in matrix['cells']:
        assert cell['dependencies'] == cell['tables'] == []
        assert Path(cell['ckpt_dir']).name == cell['method_key']
        assert Path(cell['eval_cache']) == Path(cell['ckpt_dir']) / 'eval_cache.json'
    runner._validate_manifest(matrix)


def test_every_cell_uses_full_fixed_erm_payload_and_dual_budget(matrix):
    for cell in matrix['cells']:
        cfg = cell['cfg']
        for key, value in contract.FIXED.items():
            assert cfg[key] == value, (cell['method_key'], key)
        assert cfg['epochs'] == cfg['victim_epochs'] == 50
        assert cfg['rho'] == (0.0 if cell['method_key'] == 'clean' else .1)
        assert cfg['eps'] == (.2 if cell['method_key'] == 'blended' else .185)
        assert cfg['simple_reference_eps'] == .185
        assert cfg['simple_relative_l2'] == .10
        assert not any(key.startswith(('draft_', 'lc_', 'confirmation_', 'rf_'))
                       for key in cfg)
        assert 'method_draft' not in cfg
        assert cfg['dataset_root'].endswith(str(Path('datasets') / 'Compress'))
        assert cfg['action_npy'].endswith(str(Path('actions') / 'data_bend.npy'))
        contract.validate_simple_full_config(cfg)
    assert [cell['cfg']['trigger'] for cell in matrix['cells'][:3]] == [
        'micro_dropper', 'blended', 'md_multicarrier_peak_matched']
    for cell in matrix['cells'][3:]:
        assert cell['cfg']['trigger'] == 'simple_carrier'
        assert cell['cfg']['simple_variant'] == cell['method_key']
        assert cell['cfg']['simple_seed'] == 42
        assert cell['cfg']['simple_sub_mode'] == 3
        assert cell['cfg']['simple_time_mode'] == 1


def test_cells_own_their_mutable_dose_and_payload_lists(matrix):
    matrix['cells'][0]['cfg']['dose_grid'][1] = .123
    matrix['cells'][0]['cfg']['target_joints'].append(99)
    for cell in matrix['cells'][1:]:
        assert cell['cfg']['dose_grid'] == GRID
        assert cell['cfg']['target_joints'] == [2, 3]
    assert runner.GRID == contract.GRID == GRID


SCIENCE_MUTATIONS = [
    ('epochs', 15), ('victim_epochs', 15), ('epochs', True),
    ('draft_train_samples', 15), ('draft_eval_samples', 15),
    ('method_draft', True), ('lc_artifact_path', 'old/learned_trigger.json'),
    ('simple_reference_eps', .2), ('simple_relative_l2', .11),
    ('dose_min', 1.0), ('dose_max', .8), ('dose_grid', [0.0, 1.0]),
    ('dose_coupling', 'shuffled'), ('dose_mode', 'nonlinear'),
    ('rho', .4), ('pivot', 0), ('target_joints', [4, 5]),
    ('theta_max_deg', 60), ('payload_axis', [1.0, 0.0, 0.0]),
    ('lr_scheduler', True), ('optimizer', 'adamw'), ('lr', .01),
    ('pretrained', True), ('training_protocol', 'trigger_optimizer'),
    ('attacker_access', 'victim_gradients'), ('victim_loss', 'target_weighted'),
    ('lambda_target', 5), ('strict_resume', False),
]


@pytest.mark.parametrize('key,value', SCIENCE_MUTATIONS)
def test_manifest_rejects_science_mutations_even_with_a_recomputed_hash(matrix, key, value):
    changed = copy.deepcopy(matrix)
    changed['cells'][2]['cfg'][key] = value
    changed['plan_sha256'] = runner._plan_fingerprint(changed)
    with pytest.raises(ValueError):
        runner._validate_manifest(changed)


@pytest.mark.parametrize('entrypoint', ['dataset', 'train'])
@pytest.mark.parametrize('key,value', SCIENCE_MUTATIONS)
def test_invalid_science_is_rejected_before_dataset_or_cache_access(
        matrix, tmp_path, monkeypatch, entrypoint, key, value):
    cfg = copy.deepcopy(matrix['cells'][2]['cfg'])
    cfg[key] = value
    monkeypatch.setattr(trainer, 'MMFI', forbidden)
    monkeypatch.setattr(trainer, '_load_cached_result', forbidden)
    monkeypatch.setattr(trainer, 'build_model', forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', forbidden)
    monkeypatch.setattr(trainer.torch, 'load', forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', forbidden)
    with pytest.raises(ValueError):
        if entrypoint == 'dataset':
            trainer._load_dataset(cfg, 'training')
        else:
            trainer.train(cfg, ckpt_dir=tmp_path / 'old_cache')
    assert not (tmp_path / 'old_cache').exists()


@pytest.mark.parametrize('old_artifact', [None, 'checkpoint.pt', 'eval_cache.json'])
def test_direct_trainer_refuses_unaudited_initialization_or_results(
        matrix, tmp_path, monkeypatch, old_artifact):
    cfg = copy.deepcopy(matrix['cells'][2]['cfg'])
    folder = None
    if old_artifact:
        folder = tmp_path / 'historical'
        folder.mkdir()
        (folder / old_artifact).write_bytes(b'unaudited old victim or result')
    monkeypatch.setattr(trainer, 'MMFI', forbidden)
    monkeypatch.setattr(trainer, '_load_cached_result', forbidden)
    monkeypatch.setattr(trainer, 'build_model', forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', forbidden)
    monkeypatch.setattr(trainer.torch, 'load', forbidden)
    with pytest.raises(ValueError, match='passport|runner'):
        trainer.train(cfg, ckpt_dir=folder)
    if folder:
        assert (folder / old_artifact).read_bytes() == b'unaudited old victim or result'
        assert not (folder / 'simple_inputs.json').exists()


@pytest.mark.parametrize('mutation', [
    'remove_row', 'reorder_rows', 'duplicate_row', 'status', 'profile',
    'baseline_trigger', 'baseline_epsilon', 'simple_seed', 'simple_mode',
    'dependency', 'table', 'source',
])
def test_manifest_refuses_selection_and_baseline_changes(matrix, mutation):
    changed = copy.deepcopy(matrix)
    if mutation == 'remove_row':
        changed['cells'].pop()
    elif mutation == 'reorder_rows':
        changed['cells'].reverse()
    elif mutation == 'duplicate_row':
        changed['cells'].append(copy.deepcopy(changed['cells'][0]))
    elif mutation == 'status':
        changed['status'] = 'PAPER_CONFIRMED_WINNER'
    elif mutation == 'profile':
        changed['simple_full_profile'] = 'draft'
    elif mutation == 'baseline_trigger':
        changed['cells'][1]['cfg']['trigger'] = 'badnets'
    elif mutation == 'baseline_epsilon':
        changed['cells'][1]['cfg']['eps'] = .185
    elif mutation == 'simple_seed':
        changed['cells'][3]['cfg']['simple_seed'] = 43
    elif mutation == 'simple_mode':
        changed['cells'][3]['cfg']['simple_sub_mode'] = 2
    elif mutation == 'dependency':
        changed['cells'][3]['dependencies'] = ['clean']
    elif mutation == 'table':
        changed['cells'][3]['tables'] = [1]
    else:
        changed['sources']['source_sha256']['fixture.py'] = 'b' * 64
    changed['plan_sha256'] = runner._plan_fingerprint(changed)
    with pytest.raises(ValueError):
        runner._validate_manifest(changed)


def test_dry_run_is_metadata_only_even_with_an_unavailable_gpu(tmp_path, sources, monkeypatch):
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
    assert runner.main(args(tmp_path / 'missing_data', output, '--devices',
                            'cuda:999', '--fresh', '--dry-run')) == 0
    manifest = stored(output)
    assert [cell['method_key'] for cell in manifest['cells']] == KEYS
    runner._validate_manifest(manifest)
    assert not (tmp_path / 'missing_data').exists()
    assert {path.name for path in output.iterdir()} == {
        'simple_full.resolved.json', 'run.lock'}


def test_real_cli_dry_run_requires_neither_data_nor_cuda(tmp_path):
    output = tmp_path / 'cli'
    result = subprocess.run([sys.executable, str(Path(runner.__file__).resolve()),
        *args(tmp_path / 'missing_data', output, '--devices', 'cuda:999',
              '--fresh', '--dry-run')], capture_output=True, text=True,
        timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [cell['method_key'] for cell in stored(output)['cells']] == KEYS
    assert runner.STATUS in result.stdout
    assert not (tmp_path / 'missing_data').exists()


@pytest.mark.parametrize('flag', ['--epochs', '--victim-epochs', '--train-samples',
                                  '--eval-samples', '--rho'])
def test_cli_cannot_request_a_capped_or_shortened_experiment(tmp_path, flag):
    output = tmp_path / 'capped'
    with pytest.raises(SystemExit):
        runner.main(args(tmp_path / 'data', output, '--dry-run', flag, '15'))
    assert not output.exists()


@pytest.mark.parametrize('extra', [
    ['--devices', 'cpu', 'cpu'], ['--devices', 'cuda'],
    ['--devices', 'cuda:01'], ['--devices', 'cuda:-1'],
    ['--num-workers', '-1'], ['--distortion-samples', '0'],
    ['--cells', 'clean', 'clean'],
])
def test_invalid_runtime_options_create_no_output(tmp_path, sources, extra):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        runner.main(args(tmp_path / 'data', output, '--dry-run', *extra))
    assert not output.exists()


def test_device_and_worker_changes_preserve_scientific_plan_and_resume(tmp_path, sources):
    data, output = tmp_path / 'data', tmp_path / 'resume'
    cpu = runner.build_matrix(data, output, 'cpu', 0)
    gpu = runner.build_matrix(data, output, 'cuda:7', 8)
    assert cpu['plan_sha256'] == gpu['plan_sha256']
    assert runner.main(args(data, output, '--fresh', '--dry-run')) == 0
    first = stored(output)['plan_sha256']
    assert runner.main(args(data, output, '--devices', 'cuda:7',
                            '--num-workers', '8', '--dry-run')) == 0
    resumed = stored(output)
    assert resumed['plan_sha256'] == first
    assert all(cell['cfg']['device'] == 'cuda:7'
               and cell['cfg']['num_workers'] == 8 for cell in resumed['cells'])
    runner._validate_manifest(resumed)


def test_fresh_refuses_to_delete_existing_results(tmp_path, sources):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'checkpoint.pt'
    marker.write_bytes(b'user checkpoint')
    with pytest.raises(FileExistsError):
        runner.main(args(tmp_path / 'data', output, '--fresh', '--dry-run'))
    assert marker.read_bytes() == b'user checkpoint'
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize('change', ['distortion_samples', 'source', 'data_home', 'epochs'])
def test_resume_refuses_changed_science_without_rewriting_manifest(tmp_path, sources, change):
    data, output = tmp_path / 'data', tmp_path / 'changed'
    assert runner.main(args(data, output, '--dry-run')) == 0
    path = output / 'simple_full.resolved.json'
    extra = []
    if change == 'distortion_samples':
        extra = ['--distortion-samples', '257']
    elif change == 'source':
        sources['fixture.py'] = 'b' * 64
    elif change == 'data_home':
        data = tmp_path / 'different_data'
    else:
        altered = stored(output)
        altered['cells'][0]['cfg']['epochs'] = 15
        altered['plan_sha256'] = runner._plan_fingerprint(altered)
        path.write_text(json.dumps(altered), encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(ValueError):
        runner.main(args(data, output, '--dry-run', *extra))
    assert path.read_bytes() == before


def test_selected_cells_report_pending_completion_without_export(
        tmp_path, sources, monkeypatch, capsys):
    selected = []
    def run_selected(matrix, manifest_path, devices, requested=None, **kwargs):
        assert [cell['method_key'] for cell in matrix['cells']] == KEYS
        selected.extend(requested or [])
    monkeypatch.setattr(runner, '_check_inputs', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'run_matrix', run_selected)
    monkeypatch.setattr(runner, '_complete', lambda cell: cell['method_key'] == 'additive')
    monkeypatch.setattr(runner, 'export_summary', forbidden)
    output = tmp_path / 'partial'
    assert runner.main(args(tmp_path / 'data', output, '--cells', 'additive')) == 0
    assert selected == ['additive']
    assert [cell['method_key'] for cell in stored(output)['cells']] == KEYS
    message = capsys.readouterr().out.lower()
    assert any(word in message for word in ('awaits', 'incomplete', 'pending', 'remaining'))
    assert not (output / 'main_metrics.json').exists()
    with pytest.raises(ValueError):
        runner.main(args(tmp_path / 'data', output, '--export-only'))


@pytest.fixture
def input_cell(matrix, tmp_path, monkeypatch):
    cell = copy.deepcopy(matrix['cells'][3])
    action = tmp_path / 'action.npy'
    action.write_bytes(b'fixed action bytes; no NumPy loading in metadata audit')
    cell['cfg']['action_npy'] = str(action)

    class TinySplit:
        def __init__(self, split, count):
            self.data_root = str(tmp_path / 'dataset')
            self.items = [dict(
                csi=str(Path(self.data_root) / split / f'csi_{index}.mat'),
                kpt=str(Path(self.data_root) / split / f'pose_{index}.npy'), frame_idx=index)
                for index in range(count)]

        def __len__(self):
            return len(self.items)

    class FixedTriggerKey:
        @staticmethod
        def fixed_key_sha256():
            return 'b' * 64

    datasets = {'training': TinySplit('training', 5), 'test': TinySplit('test', 3)}
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', {'train': 5, 'eval': 3})
    monkeypatch.setattr(trainer, '_load_dataset', lambda cfg, split: datasets[split])
    monkeypatch.setattr(trainer, 'build_trigger', lambda cfg: FixedTriggerKey())
    monkeypatch.setattr(trainer.np, 'load', forbidden)
    return cell, datasets, action


def test_input_passport_records_exact_ordered_ids_action_config_and_key(input_cell):
    cell, _, action = input_cell
    record = runner._collect_input_record(cell)
    assert record['schema'] == 1 and record['profile'] == runner.PROFILE
    assert record['config_fingerprint'] == trainer._config_fingerprint(cell['cfg'])
    assert record['action_file_sha256'] == hashlib.sha256(action.read_bytes()).hexdigest()
    assert record['trigger_key_sha256'] == 'b' * 64
    for name, split, count in [('train', 'training', 5), ('eval', 'test', 3)]:
        ids = [[f'{split}/csi_{index}.mat', f'{split}/pose_{index}.npy', index]
               for index in range(count)]
        assert record[name] == dict(n=count, pair_ids_sha256=runner._sha_json(ids))


@pytest.mark.parametrize('change', ['incomplete', 'duplicate', 'overlap'])
def test_input_passport_rejects_incomplete_duplicate_or_overlapping_splits(input_cell, change):
    cell, datasets, _ = input_cell
    if change == 'incomplete':
        datasets['training'].items.pop()
    elif change == 'duplicate':
        datasets['training'].items[1] = dict(datasets['training'].items[0])
    else:
        datasets['test'].items[0] = dict(datasets['training'].items[0])
    with pytest.raises(ValueError):
        runner._collect_input_record(cell)


def test_ordered_input_identity_cannot_be_rebound_on_resume(input_cell):
    cell, datasets, _ = input_cell
    record = runner._collect_input_record(cell)
    runner._bind_inputs(cell, record)
    path = Path(cell['ckpt_dir']) / 'simple_inputs.json'
    before = path.read_bytes()
    runner._bind_inputs(cell, runner._collect_input_record(cell))
    assert path.read_bytes() == before
    datasets['test'].items.reverse()
    changed = runner._collect_input_record(cell)
    assert changed['train'] == record['train']
    assert changed['eval']['pair_ids_sha256'] != record['eval']['pair_ids_sha256']
    with pytest.raises(ValueError):
        runner._bind_inputs(cell, changed)
    assert path.read_bytes() == before

@pytest.mark.parametrize('artifact', ['checkpoint.pt', 'eval_cache.json'])
def test_historical_checkpoint_or_cache_without_a_passport_is_not_imported(input_cell, artifact):
    cell, _, _ = input_cell
    folder = Path(cell['ckpt_dir'])
    folder.mkdir(parents=True)
    old = folder / artifact
    old.write_bytes(b'unaudited historical victim')
    with pytest.raises(ValueError):
        runner._bind_inputs(cell, runner._collect_input_record(cell))
    assert old.read_bytes() == b'unaudited historical victim'
    assert not (folder / 'simple_inputs.json').exists()


def test_changed_action_bytes_do_not_overwrite_bound_inputs(input_cell):
    cell, _, action = input_cell
    first = runner._collect_input_record(cell)
    runner._bind_inputs(cell, first)
    path = Path(cell['ckpt_dir']) / 'simple_inputs.json'
    before = path.read_bytes()
    action.write_bytes(b'changed reference action')
    changed = runner._collect_input_record(cell)
    assert changed['config_fingerprint'] == first['config_fingerprint']
    assert changed['action_file_sha256'] != first['action_file_sha256']
    with pytest.raises(ValueError):
        runner._bind_inputs(cell, changed)
    assert path.read_bytes() == before
