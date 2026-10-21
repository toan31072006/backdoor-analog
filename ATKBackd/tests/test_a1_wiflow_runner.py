"""Frozen WiFlow pilot routing, resume and real source-model CPU checks.

Synthetic fixtures are never evidence of real MM-Fi accuracy or GPU cost.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import sys

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a1_backbone_adapter as historical_adapter
import a1_wiflow_adapter as adapter
import run_a1_wiflow as runner
import run_peak_confirmation as confirmation
import train_backdoor as historical_trainer


@pytest.fixture(scope='module', autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def matrix(tmp_path):
    return runner.build_matrix(tmp_path / 'missing data', tmp_path / 'new output', 'cpu')


def test_two_real_victims_one_seed_no_historical_registration(matrix):
    runner.validate_manifest(matrix)
    assert matrix['seeds'] == [42]
    assert [c['key'] for c in matrix['cells']] == ['clean', 'a1']
    assert matrix['cells'][1]['dependencies'] == ['wiflow__s42__clean']
    assert [c['cfg']['rho'] for c in matrix['cells']] == [0., .1]
    assert matrix['metrics_contract']['counts'] == {'train': 133056, 'eval': 33264}
    assert matrix['metrics_contract']['dose_grid'] == [0, .2, .4, .6, .8, 1]
    assert all(c['cfg']['victim_loss'] == 'mpjpe' for c in matrix['cells'])
    assert all(c['cfg']['epochs'] == 50 and c['cfg']['victim_epochs'] == 50 for c in matrix['cells'])
    assert matrix['cells'][0]['evaluation_trigger_cfg'] == matrix['cells'][1]['cfg']
    from models.cross_backbone_factory import BACKBONES
    assert BACKBONES == ('metafiplusplus', 'graphposefi')
    assert set(historical_adapter.RECIPES) == set(BACKBONES)
    assert not Path(matrix['outdir']).exists()


@pytest.mark.parametrize('name,value', [
    ('seed', 43), ('model', 'hpeli'), ('lr', .001), ('rho', .2),
    ('batch_size', 64), ('backbone_accumulation_steps', 1), ('num_workers', 2),
    ('epochs', 49), ('victim_loss', 'smooth_l1'), ('mmfi_split_seed', 1),
    ('lr_scheduler', True), ('theta_max_deg', 45), ('target_joints', [5, 6]),
])
def test_changed_scientific_recipe_rejected_even_with_rehashed_plan(matrix, name, value):
    matrix['cells'][1]['cfg'][name] = value
    matrix['plan_sha256'] = runner.frozen._plan_hash(matrix)
    with pytest.raises(ValueError, match='changed|Noncanonical'):
        runner.validate_manifest(matrix)


@pytest.mark.parametrize('field', ['sources', 'runtime_versions', 'dependencies', 'ckpt_dir'])
def test_provenance_and_paths_fail_closed(matrix, field):
    if field in ('sources', 'runtime_versions'):
        matrix[field][next(iter(matrix[field]))] = 'changed'
    else:
        matrix['cells'][1][field] = [] if field == 'dependencies' else 'other'
    matrix['plan_sha256'] = runner.frozen._plan_hash(matrix)
    with pytest.raises(ValueError, match='changed'):
        runner.validate_manifest(matrix)


def test_allowed_execution_device_but_not_workers(matrix):
    matrix['cells'][0]['cfg']['device'] = 'cuda:1'
    runner.validate_manifest(matrix)
    matrix['cells'][0]['cfg']['num_workers'] = 4
    with pytest.raises(ValueError, match='Noncanonical'):
        runner.validate_manifest(matrix)


def test_reject_output_inside_historical_run(tmp_path):
    old = tmp_path / 'historical'
    old.mkdir()
    (old / 'cross_backbones.resolved.json').write_text('{}')
    with pytest.raises(ValueError, match='historical'):
        runner.build_matrix(tmp_path, old / 'nested')
    assert list(old.iterdir()) == [old / 'cross_backbones.resolved.json']


def test_dry_run_has_no_data_gpu_model_or_output(tmp_path, monkeypatch, capsys):
    def forbidden(*a, **kw):
        pytest.fail('dry run attempted data/CUDA/model access')
    monkeypatch.setattr(adapter, 'build_wiflow', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(runner.frozen, '_check_inputs', forbidden)
    out = tmp_path / 'never-created'
    assert runner.main(['--data-home', str(tmp_path / 'absent'), '--outdir', str(out), '--dry-run']) == 0
    assert not out.exists()
    assert '2/2 real victims' in capsys.readouterr().out


def test_missing_data_fails_before_creating_output(tmp_path):
    out = tmp_path / 'never-created'
    with pytest.raises(FileNotFoundError):
        runner.main(['--data-home', str(tmp_path / 'absent'), '--outdir', str(out), '--device', 'cpu'])
    assert not out.exists()


def test_private_adapter_does_not_patch_old_model_or_training_globals(matrix):
    before = copy.deepcopy(historical_adapter.RECIPES)
    original_builder = historical_adapter.build_backbone
    original_update = historical_trainer._victim_update
    engine = adapter.training_engine(matrix['cells'][1]['cfg'])
    assert engine is not historical_trainer
    assert engine.build_model is adapter.build_wiflow
    assert historical_adapter.RECIPES == before
    assert historical_adapter.build_backbone is original_builder
    assert historical_trainer._victim_update is original_update


class Pairs(torch.utils.data.Dataset):
    def __init__(self, count=70):
        rng = np.random.default_rng(31)
        self.x = rng.normal(size=(count, 3, 2, 2)).astype(np.float32)
        self.y = rng.normal(size=(count, 1, 17, 3)).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return dict(csi=self.x[index], pose=self.y[index])


class Tiny(torch.nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(3, 51)

    def forward(self, csi):
        feature = self.dropout(csi.mean(dim=(2, 3)))
        return self.head(feature).reshape(-1, 1, 17, 3), feature


def test_effective64_and_weighted_sixsample_tail_match_reference(matrix):
    cfg = matrix['cells'][1]['cfg']
    engine = adapter.training_engine(cfg)
    model = Tiny()
    reference = copy.deepcopy(model)
    _, optimizer = engine._build_optimizer(model, cfg, 'mmfi')
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=1e-4)
    data = Pairs(70)
    ordered_x, ordered_y = [], []
    for batch in engine._training_loader(data, cfg):
        # Compare against the actual shuffled epoch, not natural dataset order.
        ordered_x.append(batch['csi'])
        ordered_y.append(batch['pose'])
        engine._victim_update(model, optimizer, batch['csi'], batch['pose'])
    assert len(ordered_x) == 3
    ordered_x, ordered_y = torch.cat(ordered_x), torch.cat(ordered_y)
    for start, end in ((0, 64), (64, 70)):
        ref_opt.zero_grad(set_to_none=True)
        pred, _ = reference(ordered_x[start:end])
        loss = torch.linalg.vector_norm(pred - ordered_y[start:end], dim=-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
        ref_opt.step()
    assert {int(state['step']) for state in optimizer.state.values()} == {2}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=2e-6, atol=1e-7)


def test_epoch_checkpoint_resume_exact_rng_optimizer_and_atomic_failure(matrix, tmp_path, monkeypatch):
    cfg = matrix['cells'][1]['cfg']
    engine = adapter.training_engine(cfg)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = Tiny(.2)
    _, opt = engine._build_optimizer(model, cfg, 'mmfi')

    def epoch(eng, net, optimizer):
        return [eng._victim_update(net, optimizer, b['csi'], b['pose'])
                for b in eng._training_loader(Pairs(), cfg)]

    epoch(engine, model, opt)
    checkpoint = tmp_path / 'checkpoint.pt'
    engine._save_checkpoint(checkpoint, model, opt, 0, .1, cfg)
    original_bytes = checkpoint.read_bytes()
    expected = epoch(engine, model, opt)
    expected_state = copy.deepcopy(model.state_dict())
    expected_rng = (random.random(), np.random.random(), torch.rand(4))
    resumed_engine = adapter.training_engine(cfg)
    resumed = Tiny(.2)
    _, resumed_opt = resumed_engine._build_optimizer(resumed, cfg, 'mmfi')
    assert resumed_engine._load_checkpoint(checkpoint, resumed, resumed_opt, 'cpu', cfg)[0] == 0
    assert epoch(resumed_engine, resumed, resumed_opt) == expected
    assert all(torch.equal(v, expected_state[k]) for k, v in resumed.state_dict().items())
    assert {int(state['step']) for state in resumed_opt.state.values()} == {4}
    actual_rng = (random.random(), np.random.random(), torch.rand(4))
    assert actual_rng[:2] == expected_rng[:2]
    assert torch.equal(actual_rng[2], expected_rng[2])
    def interrupted_save(*args, **kwargs):
        raise OSError('synthetic interrupted atomic write')
    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(OSError, match='interrupted'):
        resumed_engine._save_checkpoint(checkpoint, resumed, resumed_opt, 1, .1, cfg)
    assert checkpoint.read_bytes() == original_bytes
    assert not checkpoint.with_name('checkpoint.pt.saving').exists()


def test_real_author_model_ordinary_mpjpe_adamw_update(matrix):
    cfg = matrix['cells'][1]['cfg']
    engine = adapter.training_engine(cfg)
    model = engine.build_model('wiflow')
    _, opt = engine._build_optimizer(model, cfg, 'mmfi')
    before = model.final_conv[-1].weight.detach().clone()
    value = engine._victim_update(model, opt, torch.randn(32, 3, 114, 10), torch.randn(32, 1, 17, 3))
    assert np.isfinite(value)
    assert not torch.equal(before, model.final_conv[-1].weight)
    assert opt.param_groups[0]['lr'] == 1e-4 and opt.param_groups[0]['weight_decay'] == 1e-4
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not model.att._forward_hooks


def test_scheduler_spawns_wiflow_entrypoint_serially_with_clean_dependency(matrix, tmp_path, monkeypatch):
    done, calls = set(), []
    monkeypatch.setattr(runner.BASE, 'completed', lambda cell, sources: {} if cell['key'] in done else None)
    monkeypatch.setattr(runner.BASE.time, 'sleep', lambda *a: None)
    class Child:
        pid = 12345
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            self.key = argv[argv.index('--_cell') + 1].rsplit('__', 1)[1]
            if self.key == 'a1':
                assert 'clean' in done
        def poll(self):
            done.add(self.key)
            return 0
    monkeypatch.setattr(runner.BASE.subprocess, 'Popen', Child)
    runner.BASE.schedule(matrix, tmp_path / runner.MANIFEST, ['cpu'], 1, matrix['cells'])
    assert len(calls) == 2
    assert all(Path(argv[2]).name == 'run_a1_wiflow.py' for argv in calls)
    assert [argv[argv.index('--_cell') + 1] for argv in calls] == ['wiflow__s42__clean', 'wiflow__s42__a1']


class SyntheticMMFi:
    def __init__(self, split, data_root, **kwargs):
        self.data_root = str(data_root)
        count = 12 if split == 'training' else 3
        self.items = [dict(csi=str(Path(data_root) / split / f'{i}.npy'),
            kpt=str(Path(data_root) / split / 'pose.npy'), frame_idx=i) for i in range(count)]
        self.raw = np.random.default_rng(9).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
        self.pose = np.random.default_rng(19).normal(0, .2, (12, 1, 17, 3)).astype(np.float32)

    def __len__(self):
        return len(self.items)

    def load_raw(self, path):
        return self.raw.copy()

    def normalize(self, raw):
        return np.clip(raw, 0, 1).astype(np.float32)

    def load_pose(self, path, frame_idx):
        return self.pose[frame_idx].copy()


@pytest.mark.parametrize('real_model', [False, True])
@pytest.mark.parametrize('key', ['clean', 'a1'])
def test_fifty_epoch_worker_poison_eval_npz_and_no_retraining_on_resume(
        tmp_path, monkeypatch, real_model, key):
    counts = {'train': 12, 'eval': 3}
    monkeypatch.setattr(runner.frozen, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(confirmation, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(historical_trainer, 'MMFI', SyntheticMMFi)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *a: None)
    if not real_model:
        monkeypatch.setattr(adapter, 'build_wiflow', lambda *a, **kw: Tiny())
    original_train, original_eval = adapter.training_engine, adapter.evaluation_engine
    def train(cfg):
        engine = original_train(cfg)
        engine.MMFI = SyntheticMMFi
        return engine
    def dataset(cfg):
        train_data = historical_trainer._load_dataset(cfg, 'training')
        test_data = historical_trainer._load_dataset(cfg, 'test')
        records = {name: dict(n=len(data), pair_ids_sha256=runner.frozen._sha_json(
            confirmation._pair_ids(data))) for name, data in (('train', train_data), ('eval', test_data))}
        return test_data, records, confirmation._pair_ids(test_data)
    def evaluation(sources):
        engine = original_eval(sources)
        engine.COUNTS = copy.deepcopy(counts)
        engine._dataset_record = dataset
        return engine
    monkeypatch.setattr(runner.BASE, 'training_engine', train)
    monkeypatch.setattr(runner.BASE, 'evaluation_engine', evaluation)
    home = tmp_path / 'home'
    action = home / 'actions' / 'data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action, np.random.default_rng(23).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    matrix = runner.build_matrix(home, tmp_path / 'out', 'cpu')
    cell = next(c for c in matrix['cells'] if c['key'] == key)
    result = runner.execute_cell(cell, matrix['sources'])
    checkpoint_path = Path(cell['ckpt_dir']) / 'checkpoint.pt'
    blob = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert blob['epoch'] == 49 and blob['cfg']['model'] == 'wiflow'
    assert {int(state['step']) for state in blob['optimizer']['state'].values()} == {50}
    assert result['metrics']['clean_mpjpe_mm'] >= 0
    assert (result['metrics']['target_pck_pct'] is None) == (key == 'clean')
    assert len(result['dose_rows']) == (0 if key == 'clean' else 6)
    assert result['provenance']['binding']['evaluation_source_sha256'] == matrix['sources']
    with np.load(Path(cell['artifact_dir']) / 'per_sample.npz', allow_pickle=False) as data:
        assert data['target_error_mm'].shape == (3, 2)
        assert data['pair_ids'].shape == (3, 3)
    original_bytes = checkpoint_path.read_bytes()
    def forbidden(*a, **kw):
        pytest.fail('completed worker rebuilt model, trained or ran inference')
    monkeypatch.setattr(adapter, 'build_wiflow', forbidden)
    assert runner.execute_cell(cell, matrix['sources']) == result
    assert runner.completed(cell, matrix['sources']) == result
    assert checkpoint_path.read_bytes() == original_bytes


def test_worker_manifest_path_cannot_escape_output(matrix, tmp_path):
    with pytest.raises(ValueError, match='exact output'):
        runner.main(['--outdir', str(tmp_path / 'other'), '--_cell', 'wiflow__s42__a1',
                     '--_manifest', str(tmp_path / runner.MANIFEST)])


def test_worker_route_uses_saved_wifi_cell_and_rejects_unknown(matrix, tmp_path, monkeypatch):
    out = Path(matrix['outdir'])
    runner.frozen._atomic_json(out / runner.MANIFEST, matrix)
    calls = []
    monkeypatch.setattr(runner, 'execute_cell', lambda cell, sources: calls.append((cell, sources)))
    args = ['--outdir', str(out), '--_cell', 'wiflow__s42__a1', '--_manifest', str(out / runner.MANIFEST)]
    assert runner.main(args) == 0
    assert calls == [(matrix['cells'][1], matrix['sources'])]
    args[3] = 'graphposefi__s42__a1'
    with pytest.raises(ValueError, match='Unknown WiFlow'):
        runner.main(args)


def test_pilot_export_authenticates_both_rows_and_marks_single_seed(matrix, monkeypatch):
    metrics = dict(t1_mpjpe_mm=100., target_pck_pct=50., clean_mpjpe_mm=150.,
        clean_pampjpe_mm=110., mean_dce_mm=80.)
    records = [dict(key=key, metrics=dict(metrics),
        provenance=dict(binding=dict(ordered_inputs={'train': 'same', 'eval': 'same'}))) for key in runner.KEYS]
    for name in ('t1_mpjpe_mm', 'target_pck_pct', 'mean_dce_mm'):
        records[0]['metrics'][name] = None
    monkeypatch.setattr(runner, 'completed', lambda cell, sources: records[runner.KEYS.index(cell['key'])])
    report = runner.export_pilot(matrix)
    assert report['status'] == 'WIFLOW_CLEAN_A1_SEED42_PILOT_COMPLETE_NOT_FULL_TABLE'
    folder = Path(matrix['outdir'])
    text = (folder / 'wiflow_pilot.md').read_text(encoding='utf-8')
    assert 'One seed; no across-seed uncertainty' in text
    assert '| clean | 42 | -- | -- | 150.00 | 110.00 | -- |' in text
    assert (folder / 'wiflow_pilot.csv').is_file()
    records[1]['provenance']['binding']['ordered_inputs']['eval'] = 'different'
    with pytest.raises(ValueError, match='identities differ'):
        runner.export_pilot(matrix)
    monkeypatch.setattr(runner, 'completed', lambda *a: None)
    with pytest.raises(ValueError, match='Both authenticated'):
        runner.export_pilot(matrix)
