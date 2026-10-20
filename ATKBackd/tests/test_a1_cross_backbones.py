"""New full-matrix orchestration; synthetic tests, never benchmark training."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_a1_cross_backbones as runner
import train_backdoor as trainer


@pytest.fixture
def matrix(tmp_path):
    return runner.build_matrix(tmp_path / 'data', tmp_path / 'new', 'cpu')


def test_full78_native_recipes_controls_and_outputs(matrix):
    assert len(matrix['cells']) == 78
    assert len({c['method_key'] for c in matrix['cells']}) == 78
    assert len({c['ckpt_dir'] for c in matrix['cells']}) == 78
    for backbone in runner.BACKBONES:
        for seed in runner.SEEDS:
            cells = [c for c in matrix['cells'] if c['backbone'] == backbone and c['seed'] == seed]
            assert [c['key'] for c in cells] == list(runner.KEYS)
            a1 = next(c for c in cells if c['key'] == 'a1')
            assert a1['tables'] == ['T1', 'T2', 'T3']
            for cell in cells:
                cfg = cell['cfg']
                assert cell['origin'] == 'fresh' and cfg['model'] == backbone
                assert cfg['num_workers'] == 0 and cfg['epochs'] == cfg['victim_epochs'] == 50
                assert cfg['pretrained'] is False and cfg['seed'] == seed
                assert cfg['batch_size'] == 32
                assert cfg['backbone_effective_batch_size'] == (32 if backbone == 'metafiplusplus' else 256)
                assert cfg['backbone_accumulation_steps'] == (1 if backbone == 'metafiplusplus' else 8)
                assert cfg['lr'] == (.01 if backbone == 'metafiplusplus' else .0003)
                assert cfg['dose_grid'] == [0., .2, .4, .6, .8, 1.]
                assert cfg['target_joints'] == [2, 3] and cfg['mmfi_split_seed'] == 0
                assert 'blended' not in cell['key']
                assert Path(matrix['outdir']) in Path(cell['ckpt_dir']).parents
                assert Path(cell['ckpt_dir']) not in Path(cell['artifact_dir']).parents
                if cell['key'] == 'clean':
                    assert cfg['rho'] == 0 and not cell['dependencies']
                    assert cell['evaluation_trigger_cfg'] == a1['cfg']
                else:
                    assert cell['dependencies'] == [f'{backbone}__s{seed}__clean']
    runner.validate_manifest(matrix)


def test_attack_operators_identical_to_existing_three_tables(matrix):
    changed = {'model', 'pretrained', 'data_parallel', 'lr', 'optimizer', 'momentum',
        'weight_decay', 'batch_size', 'lr_scheduler', 'warmup_epochs', 'cross_backbone_profile',
        'backbone_recipe_id', 'backbone_gradient_clip', 'backbone_optimizer_groups',
        'backbone_adam_betas', 'backbone_adam_eps',
        'backbone_effective_batch_size', 'backbone_accumulation_steps'}
    for cell in matrix['cells']:
        old = runner.frozen._configuration(Path(matrix['data_home']), 'cpu', 0, cell['seed'], cell['key'])
        assert {k: v for k, v in cell['cfg'].items() if k not in changed} == {
            k: v for k, v in old.items() if k not in changed}


@pytest.mark.parametrize('field', ['model', 'lr', 'batch_size', 'seed', 'dose_coupling', 'rho', 'epochs'])
def test_modified_even_rehashed_manifest_rejected(matrix, field):
    values = dict(model='hpeli', lr=.004, batch_size=8, seed=100,
                  dose_coupling='shuffled', rho=.5, epochs=20)
    matrix['cells'][1]['cfg'][field] = values[field]
    matrix['plan_sha256'] = runner.frozen._plan_hash(matrix)
    with pytest.raises(ValueError, match='settings|canonical'):
        runner.validate_manifest(matrix)


@pytest.mark.parametrize('field', ['sources', 'runtime_versions', 'dependencies', 'ckpt_dir'])
def test_provenance_and_dependency_tampering_rejected(matrix, field):
    if field in ('sources', 'runtime_versions'):
        matrix[field][next(iter(matrix[field]))] = 'changed'
    elif field == 'dependencies':
        matrix['cells'][1][field] = []
    else:
        matrix['cells'][1][field] += '_other'
    matrix['plan_sha256'] = runner.frozen._plan_hash(matrix)
    with pytest.raises(ValueError, match='settings'):
        runner.validate_manifest(matrix)


def test_runtime_device_changes_allowed_but_not_worker_recipe(matrix):
    matrix['cells'][0]['cfg']['device'] = 'cuda:1'
    matrix['cells'][0]['evaluation_trigger_cfg']['device'] = 'cuda:1'
    runner.validate_manifest(matrix)
    matrix['cells'][0]['cfg']['num_workers'] = 4
    with pytest.raises(ValueError, match='canonical'):
        runner.validate_manifest(matrix)


def test_single_backbone_declared39(tmp_path):
    value = runner.build_matrix(tmp_path, tmp_path / 'out', backbones=['graphposefi'])
    assert len(value['cells']) == 39 and value['backbones'] == ['graphposefi']
    runner.validate_manifest(value)


@pytest.mark.parametrize('load,steps', [(32, 8), (64, 4), (128, 2), (256, 1)])
def test_explicit_graph_microbatch_is_frozen_not_runtime(tmp_path, load, steps):
    matrix = runner.build_matrix(tmp_path, tmp_path / 'out', backbones=['graphposefi'],
                                 graph_load_batch_size=load)
    assert matrix['graph_load_batch_size'] == load
    assert all(c['cfg']['batch_size'] == load and c['cfg']['backbone_accumulation_steps'] == steps
               and c['cfg']['backbone_effective_batch_size'] == 256 for c in matrix['cells'])
    runner.validate_manifest(matrix)
    matrix['graph_load_batch_size'] = 32 if load != 32 else 64
    matrix['plan_sha256'] = runner.frozen._plan_hash(matrix)
    with pytest.raises(ValueError):
        runner.validate_manifest(matrix)


@pytest.mark.parametrize('models', [[], ['hpeli'], ['graphposefi', 'graphposefi']])
def test_invalid_models_rejected(tmp_path, models):
    with pytest.raises(ValueError):
        runner.build_matrix(tmp_path, tmp_path / 'out', backbones=models)


def test_pilot6_and_requested_dependencies(matrix):
    assert len(runner.selected_cells(matrix, pilot_clean=True)) == 6
    assert all(c['key'] == 'clean' for c in runner.selected_cells(matrix, pilot_clean=True))
    chosen = runner.selected_cells(matrix, ['graphposefi__s44__rho0p4'])
    assert [c['key'] for c in chosen] == ['clean', 'rho0p4']


@pytest.mark.parametrize('names', [[], ['s42__a1'], ['graphposefi__s44__a1'] * 2])
def test_unknown_duplicate_selected_cells(matrix, names):
    with pytest.raises(ValueError):
        runner.selected_cells(matrix, names)


def test_readonly_dry_run_no_data_gpu_or_directory(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('dry run performed expensive work')
    monkeypatch.setattr(runner.frozen, '_check_inputs', forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(runner, 'schedule', forbidden)
    output = tmp_path / 'output'
    assert runner.main(['--data-home', str(tmp_path / 'absent'), '--outdir', str(output),
                        '--devices', 'cuda:999', '--dry-run']) == 0
    assert not output.exists()
    assert '78 fresh scientific victims' in capsys.readouterr().out
    assert runner.parse_args(['--outdir', str(output)]).jobs_per_gpu == 2


def test_status_is_readonly_and_not_verification(matrix, monkeypatch, capsys):
    root = Path(matrix['outdir'])
    runner.frozen._atomic_json(root / runner.MANIFEST, matrix)
    path = Path(matrix['cells'][0]['artifact_dir']) / 'evaluation.json'
    runner.frozen._atomic_json(path, {})
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(runner, 'completed', lambda *a: pytest.fail('status audited or loaded a checkpoint'))
    assert runner.main(['--outdir', str(root), '--status']) == 0
    assert '1/13 artifact files (not verified)' in capsys.readouterr().out
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_resume_rejects_wrong_root_or_backbones(matrix):
    root = Path(matrix['outdir'])
    runner.frozen._atomic_json(root / runner.MANIFEST, matrix)
    with pytest.raises(ValueError, match='roots/backbones'):
        runner.main(['--outdir', str(root), '--backbones', 'graphposefi', '--dry-run'])


def test_missing_data_fails_without_creating_output(tmp_path, monkeypatch):
    def fail(*args):
        raise FileNotFoundError('missing raw data')
    monkeypatch.setattr(runner.frozen, '_check_inputs', fail)
    output = tmp_path / 'absent'
    with pytest.raises(FileNotFoundError, match='raw data'):
        runner.main(['--data-home', str(tmp_path), '--outdir', str(output)])
    assert not output.exists()


def test_worker_arguments_match_controller_protocol(matrix):
    name = matrix['cells'][0]['method_key']
    args = runner.parse_args(['--outdir', matrix['outdir'], '--_cell', name,
                              '--_manifest', str(Path(matrix['outdir']) / runner.MANIFEST)])
    assert args._cell == name


def test_two_slots_clean_dependencies_and_skip_resume(matrix, tmp_path, monkeypatch):
    # Synthetic subprocesses exercise ownership and worker argv; no model or GPU.
    cells = runner.selected_cells(matrix, ['metafiplusplus__s42__a1', 'graphposefi__s42__a1'])
    already, events, children = set(), [], []
    active = 0
    peak = 0

    class Child:
        def __init__(self, argv, **kwargs):
            nonlocal active, peak
            parsed = runner.parse_args(argv[3:])
            self.name, self.pid = parsed._cell, 100 + len(children)
            cell = next(c for c in cells if c['method_key'] == self.name)
            assert set(cell['dependencies']) <= already
            self.polls = 0
            children.append(self)
            events.append(self.name)
            active += 1
            peak = max(peak, active)

        def poll(self):
            nonlocal active
            self.polls += 1
            if self.polls == 1:
                return None
            if self.name not in already:
                already.add(self.name)
                active -= 1
            return 0

    monkeypatch.setattr(runner, 'completed', lambda c, s: {} if c['method_key'] in already else None)
    monkeypatch.setattr(runner.subprocess, 'Popen', Child)
    monkeypatch.setattr(runner.time, 'sleep', lambda *a: None)
    manifest = Path(matrix['outdir']) / runner.MANIFEST
    runner.schedule(matrix, manifest, ['cpu'], 2, cells)
    assert peak == 2 and len(events) == 4
    assert len(already) == 4
    runner.schedule(matrix, manifest, ['cpu'], 2, cells)
    assert len(events) == 4  # all verified, no duplicate training


def test_partial_never_exports_complete(matrix, monkeypatch):
    monkeypatch.setattr(runner, 'completed', lambda *a: None)
    with pytest.raises(ValueError, match='Missing authenticated'):
        runner.export_reports(matrix)
    assert not (Path(matrix['outdir']) / 'cross_backbones.complete.json').exists()


def test_sources_include_both_models_and_unchanged_metric_pipeline(matrix):
    for name in ('a1_backbone_adapter.py', 'a1_backbone_report.py', 'models/metafi_plus_plus.py',
                 'models/metafi_channel_transformer.py', 'models/graphpose_fi.py',
                 'models/metafi_sources.json', 'models/graphpose_sources.json',
                 'train_backdoor.py', 'attack/poison.py', 'a1_three_tables_eval.py',
                 'eval/target_pck.py', 'models/factory.py'):
        assert name in matrix['sources'] and len(matrix['sources'][name]) == 64
