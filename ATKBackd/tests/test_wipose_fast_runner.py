"""Synthetic CPU contracts for the additive WiPose cache/resume sidecar.

Historical runner bytes, production datasets, and run artifacts are never used
as writable test fixtures. These checks establish numerical/provenance behavior,
not CUDA throughput or full-data attack efficacy.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np
import pytest
import torch

import run_wipose_hpeli_fast as runner
import run_wipose_hpeli as public_serial
import run_wipose_hpeli_parallel as public_parallel
import wipose_data


class TinyPose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3)
        self.dropout = torch.nn.Dropout(.2)
        self.head = torch.nn.Linear(3, 36)

    def forward(self, csi):
        features = self.bn(csi.mean(dim=(2, 3)))
        if self.training:
            features = self.dropout(features)+(random.random()+np.random.random())*.001
        return self.head(features).reshape(-1, 1, 18, 2), features


class FakeDataset:
    def __init__(self, count, split):
        self.split = 'training' if split == 'train' else 'validation'
        self.items = [dict(name=f'{split}_frame{index:03d}') for index in range(count)]
        rng = np.random.default_rng(123 if split == 'train' else 987)
        self.x = rng.uniform(size=(count, 3, 90, 5)).astype(np.float32)
        self.y = rng.uniform(.1, .8, size=(count, 1, 18, 2)).astype(np.float32)
        self.y[:, :, 5], self.y[:, :, 8] = [.2, .1], [.1, .7]
        self.reads = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        self.reads += 1
        return dict(csi=self.x[index].copy(), pose=self.y[index].copy(),
                    name=self.items[index]['name'])


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    action = tmp_path/'action.npy'
    np.save(action, np.random.default_rng(555).normal(size=(1, 3, 8, 25, 1)).astype(np.float32))
    datasets = dict(training=FakeDataset(35, 'train'), validation=FakeDataset(3, 'test'))
    inputs = dict(fingerprint='synthetic-input-hash', splits={
        'training': dict(official_count=35, retained_count=35),
        'validation': dict(official_count=3, retained_count=3)})
    modules = (runner, runner.ENGINE, runner.BASE, runner.LEGACY, runner.LEGACY.BASE)
    for module in modules:
        if hasattr(module, 'EXPECTED_LIST_COUNTS'):
            monkeypatch.setattr(module, 'EXPECTED_LIST_COUNTS', {'training': 35, 'validation': 3})
        if hasattr(module, 'source_hashes'):
            monkeypatch.setattr(module, 'source_hashes', lambda: {'fixture.py': 'fixed-test-source'})
        if hasattr(module, 'datasets_for'):
            monkeypatch.setattr(module, 'datasets_for', lambda cfg: datasets)
        if hasattr(module, 'build_model'):
            monkeypatch.setattr(module, 'build_model', TinyPose)
    monkeypatch.setattr(wipose_data, 'dataset_audit', lambda root: copy.deepcopy(inputs))
    return dict(home=tmp_path/'home', root=tmp_path/'dataset', out=tmp_path/'fast',
                old=tmp_path/'legacy', action=action, datasets=datasets, inputs=inputs)


def configuration(fixture, output=None, engine=runner, resume_from=None):
    arguments = (fixture['home'], output or fixture['out'], 'cpu',
                 fixture['root'], fixture['action'])
    return engine.configuration(*arguments, resume_from=resume_from) if engine is runner else engine.configuration(*arguments)


def manifest(fixture, output=None, engine=runner, resume_from=None):
    prepared = engine.prepare(configuration(fixture, output, engine, resume_from))
    plan = prepared[0]
    Path(plan['cfg']['outdir']).mkdir(parents=True, exist_ok=True)
    runner.BASE.util.atomic_json(Path(plan['cfg']['outdir'])/engine.MANIFEST, plan)
    return prepared


def cell(plan, stage, engine=runner):
    return engine.cell_dir(plan, stage)


def checkpoint(plan, stage, engine=runner, **kwargs):
    return engine.checkpoint_header(cell(plan, stage, engine)/'checkpoint.pt',
                                    plan, stage, **kwargs)


def tree(root):
    return {path.relative_to(root).as_posix():
            (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in Path(root).rglob('*') if path.is_file()}


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for one, two in zip(left, right):
            assert_same(one, two)
    else:
        assert left == right


TRAINING_STATE = ('epoch', 'model', 'optimizer', 'rng', 'execution', 'train_loss')


def assert_training_same(left, right):
    for key in TRAINING_STATE:
        assert_same(left[key], right[key])


def interrupt_legacy(fixture, monkeypatch, epochs=None):
    epochs = epochs or {'clean': 0, 'a1': 0}
    prepared = manifest(fixture, fixture['old'], runner.LEGACY)
    plan, datasets, trigger, poisoned = prepared
    original = runner.BASE.util.atomic_checkpoint

    def interrupted(path, blob):
        original(path, blob)
        if blob['epoch'] == epochs[blob['stage']]:
            raise RuntimeError('synthetic legacy checkpoint interruption')

    monkeypatch.setattr(runner.BASE.util, 'atomic_checkpoint', interrupted)
    try:
        for stage in epochs:
            with pytest.raises(RuntimeError, match='legacy checkpoint interruption'):
                runner.LEGACY.train_stage(plan, stage, datasets, trigger, poisoned)
    finally:
        monkeypatch.setattr(runner.BASE.util, 'atomic_checkpoint', original)
    # The real legacy controller/worker creates these persistent lock files.
    # Direct unit calls need equivalent inactive files for read-only adoption.
    for lock in [fixture['old']/'run.lock',
                 *(cell(plan, stage, runner.LEGACY)/'run.lock' for stage in runner.STAGES)]:
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_bytes(b'0')
    return prepared


def cli(fixture, *arguments):
    return ['--data-home', str(fixture['home']), '--outdir', str(fixture['out']),
            '--dataset-root', str(fixture['root']), '--action-npy', str(fixture['action']),
            '--device', 'cpu', *arguments]


def test_fast_profile_is_separate_and_scientific_recipe_is_unchanged(fixture):
    actual = configuration(fixture)
    legacy = configuration(fixture, engine=runner.LEGACY)
    assert runner.PROFILE == actual['profile'] == 'wipose_2d_hpeli_clean_a1_seed42_fast_v1'
    assert runner.MANIFEST == 'wipose_hpeli_fast.resolved.json'
    assert actual['optimization'] == dict(train_cache='exact_normalized_fp32_ram',
        gradient_guard='device_aggregate', cache_max_bytes=2147483648)
    assert {key: value for key, value in actual.items()
            if key not in ('profile', 'optimization', 'resume_from')} == {
        key: value for key, value in legacy.items() if key != 'profile'}
    default = runner.configuration(fixture['home'], device='cpu')
    assert Path(default['outdir']) == fixture['home']/'runs/wipose_hpeli_a1_s42_fast_v1'
    assert (actual['epochs'], actual['final_epoch'], actual['batch_size']) == (50, 49, 32)
    assert (actual['lr'], actual['momentum'], actual['weight_decay']) == (.01, .9, 0.)
    assert actual['drop_last'] and not actual['amp'] and actual['num_workers'] == 0
    assert actual['resume_from'] is None


def test_private_engine_does_not_change_public_historical_modules():
    assert runner.ENGINE is not public_parallel
    assert runner.BASE is runner.ENGINE.BASE
    assert runner.BASE is not public_serial
    assert runner.LEGACY is not runner.ENGINE
    assert runner.LEGACY.BASE is not runner.BASE
    assert public_parallel.PROFILE == 'wipose_2d_hpeli_clean_a1_seed42_parallel_v1'
    assert public_serial.PROFILE == 'wipose_2d_hpeli_clean_a1_seed42_v1'
    assert public_parallel.MANIFEST == 'wipose_hpeli_parallel.resolved.json'
    assert public_serial.MANIFEST == 'wipose_hpeli.resolved.json'


@pytest.mark.parametrize('stage', ['clean', 'a1'])
def test_full_cached_trajectory_matches_legacy_weights_bn_optimizer_and_all_rng(fixture, stage):
    old_plan, datasets, trigger, poisoned = manifest(fixture, fixture['old'], runner.LEGACY)
    runner.LEGACY.train_stage(old_plan, stage, datasets, trigger, poisoned)
    expected = checkpoint(old_plan, stage, runner.LEGACY, final=True)
    before = tree(fixture['old'])
    fast_plan, datasets, trigger, poisoned = manifest(fixture)
    runner.train_stage(fast_plan, stage, datasets, trigger, poisoned)
    actual = checkpoint(fast_plan, stage, final=True)
    assert_training_same(actual, expected)
    assert runner.verify_training(fast_plan, stage)['status'] == 'TRAINED_NOT_EVALUATED'
    assert tree(fixture['old']) == before


def test_fresh_dry_run_has_no_data_model_cuda_or_output(fixture, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('Dry run touched data/model/CUDA')

    monkeypatch.setattr(runner, 'prepare', forbidden)
    monkeypatch.setattr(runner, 'build_model', forbidden)
    monkeypatch.setattr(runner.BASE, 'build_model', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    assert runner.main(cli(fixture, '--dry-run')) == 0
    assert not fixture['out'].exists()
    assert 'NO data/model/CUDA' in capsys.readouterr().out


@pytest.mark.parametrize('arguments', [
    ['--epochs', '1'], ['--batch-size', '1'], ['--lr', '.1'], ['--seed', '43'],
    ['--rho', '.2'], ['--jobs-per-gpu', '0'], ['--jobs-per-gpu', '3'],
    ['--_worker', 'clean'], ['--_phase', 'train'], ['--_manifest', 'x.json'],
])
def test_invalid_cli_never_constructs_a_run(fixture, arguments):
    with pytest.raises((SystemExit, ValueError)):
        runner.main(cli(fixture, *arguments))
    assert not fixture['out'].exists()


@pytest.mark.parametrize('stage', ['clean', 'a1'])
def test_training_caches_complete_train_once_and_never_reads_test(fixture, stage):
    plan, datasets, trigger, poisoned = manifest(fixture)
    before_train = datasets['training'].reads
    before_test = datasets['validation'].reads
    runner.train_stage(plan, stage, datasets, trigger, poisoned)
    assert datasets['training'].reads-before_train == len(datasets['training'])
    assert datasets['validation'].reads == before_test
    assert checkpoint(plan, stage, final=True)['epoch'] == 49


@pytest.mark.parametrize('error', [MemoryError, ValueError, RuntimeError])
def test_cache_failure_is_fatal_before_training_or_checkpoint(fixture, monkeypatch, error):
    plan, datasets, trigger, poisoned = manifest(fixture)

    def fail_cache(*args, **kwargs):
        raise error('synthetic exact-cache failure; no fallback')

    monkeypatch.setattr(runner, 'cache_training', fail_cache)
    with pytest.raises(error, match='exact-cache failure'):
        runner.train_stage(plan, 'clean', datasets, trigger, poisoned)
    assert not (cell(plan, 'clean')/'checkpoint.pt').exists()
    assert not (cell(plan, 'clean')/'training.complete.json').exists()
    assert datasets['training'].reads == 0


def test_import_preserves_each_checkpoint_and_old_tree_with_different_stage_epochs(fixture, monkeypatch):
    old_plan, _, _, _ = interrupt_legacy(fixture, monkeypatch, {'clean': 0, 'a1': 2})
    old = {stage: checkpoint(old_plan, stage, runner.LEGACY)
           for stage in runner.STAGES}
    before = tree(fixture['old'])
    plan, _, _, _ = manifest(fixture, resume_from=fixture['old'])
    assert plan['resume']['source_root'] == str(fixture['old'].resolve())
    assert plan['resume']['source_plan_sha256'] == old_plan['plan_sha256']
    assert plan['resume']['source_manifest_sha256'] == runner.BASE.util.file_sha(
        fixture['old']/runner.LEGACY.MANIFEST)
    assert {stage: entry['epoch'] for stage, entry in plan['resume']['checkpoints'].items()} == {
        'clean': 0, 'a1': 2}
    runner.import_legacy_checkpoints(fixture['old'], plan)
    assert json.loads((fixture['out']/'resume.lineage.json').read_text(encoding='utf-8')) == plan['resume']
    for stage in runner.STAGES:
        imported = checkpoint(plan, stage)
        assert_training_same(imported, old[stage])
        assert imported['profile'] == runner.PROFILE
        assert imported['cfg'] == plan['cfg']
        assert imported['plan_sha256'] == plan['plan_sha256']
        assert imported['resume_lineage']['stage'] == stage
        assert imported['resume_lineage']['source_manifest_sha256'] == plan['resume']['source_manifest_sha256']
        assert runner.verify_training(plan, stage) is None
    assert tree(fixture['old']) == before
    new_before = tree(fixture['out'])
    runner.import_legacy_checkpoints(fixture['old'], plan)
    assert tree(fixture['out']) == new_before
    assert tree(fixture['old']) == before


@pytest.mark.parametrize('stage', ['clean', 'a1'])
def test_imported_epoch_zero_resumes_remaining_49_epochs_exactly(fixture, monkeypatch, stage):
    old_plan, datasets, trigger, poisoned = interrupt_legacy(fixture, monkeypatch)
    imported_state = checkpoint(old_plan, stage, runner.LEGACY)
    plan, datasets, trigger, poisoned = manifest(fixture, resume_from=fixture['old'])
    runner.import_legacy_checkpoints(fixture['old'], plan)
    source_before = tree(fixture['old'])
    saved_epochs = []
    atomic_checkpoint = runner.BASE.util.atomic_checkpoint

    def observe_checkpoint(path, blob):
        saved_epochs.append(blob['epoch'])
        return atomic_checkpoint(path, blob)

    with monkeypatch.context() as observed:
        observed.setattr(runner.BASE.util, 'atomic_checkpoint', observe_checkpoint)
        runner.train_stage(plan, stage, datasets, trigger, poisoned)
    assert saved_epochs == list(range(1, 50))
    actual = checkpoint(plan, stage, final=True)
    assert actual['resume_lineage'] == checkpoint(plan, stage)['resume_lineage']
    assert not torch.equal(actual['model']['head.weight'], imported_state['model']['head.weight'])
    assert tree(fixture['old']) == source_before
    runner.LEGACY.train_stage(old_plan, stage, datasets, trigger, poisoned)
    expected = checkpoint(old_plan, stage, runner.LEGACY, final=True)
    assert_training_same(actual, expected)
    # Only the explicit legacy continuation above changes the synthetic origin.
    assert source_before != tree(fixture['old'])


@pytest.mark.parametrize('changed', ['config', 'inputs', 'poison', 'operator', 'runtime',
                                   'sources', 'manifest_hash', 'checkpoint_plan', 'checkpoint_cfg'])
def test_legacy_binding_tampering_refuses_import_before_any_target_checkpoint(fixture, monkeypatch, changed):
    old_plan, _, _, _ = interrupt_legacy(fixture, monkeypatch)
    old_path = fixture['old']/runner.LEGACY.MANIFEST
    if changed in ('checkpoint_plan', 'checkpoint_cfg'):
        altered = checkpoint(old_plan, 'a1', runner.LEGACY)
        if changed == 'checkpoint_plan':
            altered['plan_sha256'] = 'wrong'
        else:
            altered['cfg']['lr'] = .02
        runner.BASE.util.atomic_checkpoint(cell(old_plan, 'a1', runner.LEGACY)/'checkpoint.pt', altered)
    else:
        altered = copy.deepcopy(old_plan)
        if changed == 'config':
            altered['cfg']['lr'] = .02
        elif changed == 'inputs':
            altered['inputs']['fingerprint'] = 'wrong'
        elif changed == 'poison':
            altered['poison']['n_poison'] += 1
        elif changed == 'operator':
            altered['operator']['tampered'] = True
        elif changed == 'runtime':
            altered['runtime']['python'] = 'wrong'
        elif changed == 'sources':
            altered['sources']['fixture.py'] = 'wrong'
        else:
            altered['plan_sha256'] = 'wrong'
        if changed != 'manifest_hash':
            altered['plan_sha256'] = runner.BASE.util.object_sha({
                key: value for key, value in altered.items() if key != 'plan_sha256'})
        runner.BASE.util.atomic_json(old_path, altered)
        if changed in ('inputs', 'poison', 'operator'):
            # Rebind both synthetic headers too, so this reaches the actual
            # prepared-context comparison rather than only a stale plan hash.
            for stage in runner.STAGES:
                path = cell(old_plan, stage, runner.LEGACY)/'checkpoint.pt'
                rebound = torch.load(path, map_location='cpu', weights_only=False)
                rebound['plan_sha256'] = altered['plan_sha256']
                runner.BASE.util.atomic_checkpoint(path, rebound)
            if changed == 'poison':
                runner.BASE.util.atomic_json(cell(old_plan, 'a1', runner.LEGACY)/'poison.json',
                                             altered['poison'])
    before = tree(fixture['old'])
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        prepared = runner.prepare(configuration(fixture, resume_from=fixture['old']))
        runner.import_legacy_checkpoints(fixture['old'], prepared[0])
    assert tree(fixture['old']) == before
    assert not list(fixture['out'].rglob('checkpoint.pt'))


@pytest.mark.parametrize('changed', ['missing', 'changed_stage', 'changed_origin'])
def test_imported_checkpoint_requires_authentic_lineage(fixture, monkeypatch, changed):
    interrupt_legacy(fixture, monkeypatch)
    plan, _, _, _ = manifest(fixture, resume_from=fixture['old'])
    runner.import_legacy_checkpoints(fixture['old'], plan)
    source_before = tree(fixture['old'])
    altered = checkpoint(plan, 'a1')
    if changed == 'missing':
        altered.pop('resume_lineage')
    elif changed == 'changed_stage':
        altered['resume_lineage']['stage'] = 'clean'
    else:
        altered['resume_lineage']['source_manifest_sha256'] = '0'*64
    runner.BASE.util.atomic_checkpoint(cell(plan, 'a1')/'checkpoint.pt', altered)
    before = tree(fixture['out'])
    with pytest.raises((ValueError, RuntimeError), match='[Ll]ineage|[Rr]esume|binding'):
        checkpoint(plan, 'a1')
    with pytest.raises((ValueError, RuntimeError)):
        runner.import_legacy_checkpoints(fixture['old'], plan)
    assert tree(fixture['out']) == before
    assert tree(fixture['old']) == source_before


def test_import_requires_explicit_source_and_preserves_existing_target_checkpoint(fixture, monkeypatch):
    interrupt_legacy(fixture, monkeypatch)
    scratch, _, _, _ = manifest(fixture)
    with pytest.raises((ValueError, RuntimeError)):
        runner.import_legacy_checkpoints(fixture['old'], scratch)
    plan, _, _, _ = manifest(fixture, output=fixture['out'].parent/'explicit', resume_from=fixture['old'])
    folder = cell(plan, 'a1')
    folder.mkdir(parents=True, exist_ok=True)
    sentinel = folder/'checkpoint.pt'
    runner.BASE.util.atomic_checkpoint(sentinel, {'preserve': 'existing unrelated target checkpoint'})
    before = sentinel.read_bytes(), sentinel.stat().st_mtime_ns
    with pytest.raises((ValueError, RuntimeError, OSError, EOFError, IndexError)):
        runner.import_legacy_checkpoints(fixture['old'], plan)
    assert (sentinel.read_bytes(), sentinel.stat().st_mtime_ns) == before


@pytest.mark.parametrize('resume', [False, True])
def test_true_fresh_dry_run_does_not_import_tensor_dataset_cache_or_model_modules(tmp_path, resume):
    script = '''import builtins, runpy, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'wipose_data', 'wipose_a1_adapter',
                             'wipose_fast_ops', 'models'}:
        raise AssertionError('Forbidden dry-run import: '+name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--data-home', sys.argv[2], '--dry-run', *sys.argv[3:]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    home = tmp_path/'nonexistent-home'
    arguments = ['--resume-from', str(tmp_path/'nonexistent-origin')] if resume else []
    before = tree(tmp_path)
    result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', script,
        str(HERE/'run_wipose_hpeli_fast.py'), str(home), *arguments],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(HERE)))
    assert result.returncode == 0, result.stderr
    assert 'NO data/model/CUDA' in result.stdout
    assert tree(tmp_path) == before


def test_resume_preflight_authenticates_source_but_writes_no_target(fixture, monkeypatch):
    interrupt_legacy(fixture, monkeypatch)
    before = tree(fixture['old'])
    assert runner.main(cli(fixture, '--resume-from', str(fixture['old']), '--preflight')) == 0
    assert not fixture['out'].exists()
    assert tree(fixture['old']) == before


def test_resume_prepare_requires_both_legacy_checkpoints_without_any_writes(fixture, monkeypatch):
    old_plan, _, _, _ = interrupt_legacy(fixture, monkeypatch, {'clean': 0})
    assert (cell(old_plan, 'clean', runner.LEGACY)/'checkpoint.pt').is_file()
    before = tree(fixture['old'])
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        runner.prepare(configuration(fixture, resume_from=fixture['old']))
    assert tree(fixture['old']) == before
    assert not fixture['out'].exists()


@pytest.mark.parametrize('device', ['cuda', 'CUDA:0', 'cuda:-1', 'auto', 'mps'])
def test_device_requires_cpu_or_an_explicit_cuda_index(fixture, device):
    with pytest.raises(ValueError, match='cpu or explicit'):
        runner.configuration(fixture['home'], fixture['out'], device)
    assert not fixture['out'].exists()


@pytest.mark.parametrize('nested', [False, True])
def test_fast_output_cannot_reuse_or_nest_in_legacy_origin(fixture, monkeypatch, nested):
    interrupt_legacy(fixture, monkeypatch)
    before = tree(fixture['old'])
    output = fixture['old']/'fast' if nested else fixture['old']
    with pytest.raises(ValueError):
        configuration(fixture, output=output, resume_from=fixture['old'])
    assert tree(fixture['old']) == before


def test_import_source_must_equal_explicit_frozen_resume_source(fixture, monkeypatch):
    interrupt_legacy(fixture, monkeypatch)
    plan, _, _, _ = manifest(fixture, resume_from=fixture['old'])
    before_source, before_target = tree(fixture['old']), tree(fixture['out'])
    with pytest.raises(ValueError):
        runner.import_legacy_checkpoints(fixture['old'].parent/'wrong-origin', plan)
    assert tree(fixture['old']) == before_source
    assert tree(fixture['out']) == before_target


def test_imported_checkpoint_rejects_changed_immutable_lineage_file(fixture, monkeypatch):
    interrupt_legacy(fixture, monkeypatch)
    plan, _, _, _ = manifest(fixture, resume_from=fixture['old'])
    runner.import_legacy_checkpoints(fixture['old'], plan)
    altered = copy.deepcopy(plan['resume'])
    altered['source_plan_sha256'] = '0'*64
    runner.BASE.util.atomic_json(fixture['out']/'resume.lineage.json', altered)
    before_source, before_target = tree(fixture['old']), tree(fixture['out'])
    with pytest.raises((ValueError, RuntimeError), match='[Ll]ineage|[Rr]esume|binding'):
        checkpoint(plan, 'clean')
    assert tree(fixture['old']) == before_source
    assert tree(fixture['out']) == before_target


def test_imported_final_checkpoints_only_publish_training_markers_without_new_epochs(fixture, monkeypatch):
    old_plan, _, _, _ = interrupt_legacy(fixture, monkeypatch, {'clean': 49, 'a1': 49})
    before_source = tree(fixture['old'])
    plan, datasets, trigger, poisoned = manifest(fixture, resume_from=fixture['old'])
    runner.import_legacy_checkpoints(fixture['old'], plan)
    before = {stage: checkpoint(plan, stage, final=True) for stage in runner.STAGES}
    saves = []
    atomic_checkpoint = runner.BASE.util.atomic_checkpoint

    def observe(path, blob):
        saves.append(blob['epoch'])
        return atomic_checkpoint(path, blob)

    monkeypatch.setattr(runner.BASE.util, 'atomic_checkpoint', observe)
    for stage in runner.STAGES:
        runner.train_stage(plan, stage, datasets, trigger, poisoned)
        assert runner.verify_training(plan, stage)['checkpoint_epoch'] == 49
        assert_training_same(checkpoint(plan, stage, final=True), before[stage])
    assert saves == []
    assert tree(fixture['old']) == before_source
    assert old_plan['cfg']['final_epoch'] == 49


def test_import_is_idempotent_after_fast_checkpoint_has_advanced(fixture, monkeypatch):
    old_plan, datasets, trigger, poisoned = interrupt_legacy(fixture, monkeypatch)
    plan, datasets, trigger, poisoned = manifest(fixture, resume_from=fixture['old'])
    runner.import_legacy_checkpoints(fixture['old'], plan)
    source_before = tree(fixture['old'])
    atomic_checkpoint = runner.BASE.util.atomic_checkpoint

    def interrupt_fast(path, blob):
        atomic_checkpoint(path, blob)
        if blob['stage'] == 'a1' and blob['epoch'] == 17:
            raise RuntimeError('synthetic fast interruption')

    with monkeypatch.context() as interrupted:
        interrupted.setattr(runner.BASE.util, 'atomic_checkpoint', interrupt_fast)
        with pytest.raises(RuntimeError, match='fast interruption'):
            runner.train_stage(plan, 'a1', datasets, trigger, poisoned)
    assert checkpoint(plan, 'a1')['epoch'] == 17
    target_before = tree(fixture['out'])
    runner.import_legacy_checkpoints(fixture['old'], plan)
    assert tree(fixture['out']) == target_before
    runner.train_stage(plan, 'a1', datasets, trigger, poisoned)
    actual = checkpoint(plan, 'a1', final=True)
    assert tree(fixture['old']) == source_before
    runner.LEGACY.train_stage(old_plan, 'a1', datasets, trigger, poisoned)
    assert_training_same(actual, checkpoint(old_plan, 'a1', runner.LEGACY, final=True))
