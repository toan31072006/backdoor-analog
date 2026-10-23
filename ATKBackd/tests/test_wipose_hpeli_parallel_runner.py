"""CPU/synthetic contracts for independent Clean/A1 training, not efficacy."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

import run_wipose_hpeli as serial
import run_wipose_hpeli_parallel as runner
import wipose_data


class TinyPose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.2)
        self.head = torch.nn.Linear(3, 36)

    def forward(self, csi):
        feature = csi.mean(dim=(2, 3))
        if self.training:
            feature = self.dropout(feature)+(random.random()+np.random.random())*.001
        return self.head(feature).reshape(-1, 1, 18, 2), feature


class FakeDataset:
    def __init__(self, count, split):
        self.items = [dict(name=f'{split}_frame{i:03}') for i in range(count)]
        rng = np.random.default_rng(123 if split == 'train' else 987)
        self.x = rng.uniform(size=(count, 3, 90, 5)).astype(np.float32)
        self.y = rng.uniform(.1, .8, size=(count, 1, 18, 2)).astype(np.float32)
        self.y[:, :, 5], self.y[:, :, 8] = [.2, .1], [.1, .7]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
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
    for module in (runner.BASE, runner, serial):
        if hasattr(module, 'EXPECTED_LIST_COUNTS'):
            monkeypatch.setattr(module, 'EXPECTED_LIST_COUNTS', {'training': 35, 'validation': 3})
        monkeypatch.setattr(module, 'source_hashes', lambda: {'fixture.py': 'fixed-test-source'})
        if hasattr(module, 'datasets_for'):
            monkeypatch.setattr(module, 'datasets_for', lambda cfg: datasets)
        monkeypatch.setattr(module, 'build_model', TinyPose)
    monkeypatch.setattr(wipose_data, 'dataset_audit', lambda root: copy.deepcopy(inputs))
    return dict(home=tmp_path/'home', root=tmp_path/'dataset', out=tmp_path/'parallel',
                action=action, datasets=datasets, inputs=inputs)


def cfg(fixture, output=None):
    return runner.configuration(fixture['home'], output or fixture['out'], 'cpu',
                                fixture['root'], fixture['action'])


def cli(fixture, *extra):
    return ['--data-home', str(fixture['home']), '--outdir', str(fixture['out']),
            '--dataset-root', str(fixture['root']), '--action-npy', str(fixture['action']),
            '--device', 'cpu', *extra]


def manifest(fixture, output=None):
    prepared = runner.prepare(cfg(fixture, output))
    plan = prepared[0]
    Path(plan['cfg']['outdir']).mkdir(parents=True, exist_ok=True)
    runner.BASE.util.atomic_json(Path(plan['cfg']['outdir'])/runner.MANIFEST, plan)
    return prepared


def cell(plan, stage):
    return runner.BASE.cell_dir(plan, stage)


def checkpoint(plan, stage, **kwargs):
    return runner.checkpoint_header(cell(plan, stage)/'checkpoint.pt', plan, stage, **kwargs)


def complete(fixture, output=None):
    plan, datasets, trigger, poisoned = manifest(fixture, output)
    # Intentionally train A1 first: Clean is an evaluation reference, not a
    # source of weights, labels, training selection or initialization for A1.
    for stage in ('a1', 'clean'):
        runner.train_stage(plan, stage, datasets, trigger, poisoned)
    for stage in ('clean', 'a1'):
        runner.evaluate_trained_stage(plan, stage, datasets, trigger)
    return plan


def tree(root):
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns,
            runner.BASE.util.file_sha(path)) for path in Path(root).rglob('*') if path.is_file()}


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_distinct_output_profile_and_identical_scientific_recipe(fixture):
    actual = cfg(fixture)
    assert actual['profile'] == 'wipose_2d_hpeli_clean_a1_seed42_parallel_v1'
    assert runner.MANIFEST == 'wipose_hpeli_parallel.resolved.json'
    default = runner.configuration(fixture['home'], device='cpu')
    assert Path(default['outdir']) == fixture['home']/'runs/wipose_hpeli_a1_s42_parallel_v1'
    original = serial.configuration(fixture['home'], fixture['out'], 'cpu',
                                    fixture['root'], fixture['action'])
    assert {k: v for k, v in actual.items() if k != 'profile'} == {
        k: v for k, v in original.items() if k != 'profile'}
    assert (actual['epochs'], actual['final_epoch'], actual['batch_size']) == (50, 49, 32)
    assert (actual['optimizer'], actual['lr'], actual['momentum'], actual['weight_decay']) == (
        'sgd', .01, .9, 0.)
    assert 'jobs_per_gpu' not in actual


def test_job_slots_are_operational_and_do_not_change_plan(fixture, capsys):
    plans = []
    for slots in ('1', '2'):
        args = runner.parse_args(cli(fixture, '--jobs-per-gpu', slots))
        assert args.jobs_per_gpu == int(slots)
        assert runner.main(cli(fixture, '--dry-run', '--jobs-per-gpu', slots)) == 0
        stdout = capsys.readouterr().out
        plans.append(json.loads(stdout[stdout.index('{'):]))
    assert plans[0] == plans[1]
    assert 'jobs_per_gpu' not in plans[0]['cfg']
    assert not fixture['out'].exists()


def test_private_clone_does_not_mutate_public_serial_module():
    assert runner.BASE is not serial
    assert serial.PROFILE == 'wipose_2d_hpeli_clean_a1_seed42_v1'
    assert serial.MANIFEST == 'wipose_hpeli.resolved.json'
    assert serial.configuration is not runner.configuration
    assert serial.checkpoint_header is not runner.checkpoint_header
    assert serial.verify_stage is not runner.verify_stage
    assert 'run_wipose_hpeli_parallel.py' not in serial.SOURCE_NAMES


@pytest.mark.parametrize('location', ['root', 'nested'])
def test_existing_serial_output_or_descendant_is_never_migrated(fixture, location):
    fixture['out'].mkdir(parents=True)
    old_manifest = fixture['out']/'wipose_hpeli.resolved.json'
    old_manifest.write_text('{"preserve": true}', encoding='utf-8')
    requested = fixture['out'] if location == 'root' else fixture['out']/'parallel'
    with pytest.raises(ValueError, match='NEW output|historical'):
        runner.configuration(fixture['home'], requested, 'cpu', fixture['root'], fixture['action'])
    assert old_manifest.read_text(encoding='utf-8') == '{"preserve": true}'
    assert not (fixture['out']/runner.MANIFEST).exists()


@pytest.mark.parametrize('slots', ['0', '3', '-1'])
def test_invalid_slot_counts_rejected(fixture, slots):
    with pytest.raises(SystemExit):
        runner.parse_args(cli(fixture, '--jobs-per-gpu', slots))


def test_dryrun_never_touches_data_model_gpu_or_output(fixture, monkeypatch, capsys):
    def fail(*args, **kwargs):
        pytest.fail('dry-run touched data/model/CUDA')
    monkeypatch.setattr(runner, 'prepare', fail)
    monkeypatch.setattr(runner, 'build_model', fail)
    monkeypatch.setattr(torch.cuda, 'is_available', fail)
    assert runner.main(cli(fixture, '--dry-run', '--jobs-per-gpu', '2')) == 0
    assert not fixture['out'].exists()
    assert 'DRY RUN' in capsys.readouterr().out


def test_true_fresh_dryrun_has_no_tensor_dataset_imports(tmp_path):
    root = tmp_path/'nonexistent'
    script = '''import builtins, runpy, sys
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'wipose_data', 'wipose_a1_adapter', 'models'}:
        raise AssertionError('Forbidden dry-run import: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
sys.argv = [sys.argv[1], '--data-home', sys.argv[2], '--dry-run']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', script, runner.__file__, str(root)],
        capture_output=True, text=True, timeout=30, check=False,
        env=dict(os.environ, PYTHONPATH=str(Path(runner.__file__).parent)))
    assert result.returncode == 0, result.stderr
    assert 'DRY RUN' in result.stdout
    assert not root.exists()


@pytest.mark.parametrize('field', ['cfg', 'sources', 'runtime', 'plan_sha256'])
def test_manifest_config_source_runtime_guards(fixture, field):
    plan, *_ = manifest(fixture)
    changed = copy.deepcopy(plan)
    if field == 'cfg':
        changed['cfg']['epochs'] = 20
    elif field == 'plan_sha256':
        changed[field] = 'wrong'
    else:
        changed[field] = {'changed': True}
    if field != 'plan_sha256':
        changed['plan_sha256'] = runner.BASE.util.object_sha(
            {k: v for k, v in changed.items() if k != 'plan_sha256'})
    path = fixture['out']/runner.MANIFEST
    runner.BASE.util.atomic_json(path, changed)
    with pytest.raises(ValueError):
        runner.load_manifest(path)


def test_a1_independent_training_precedes_clean_but_evaluation_refuses(fixture, monkeypatch):
    plan, datasets, trigger, poisoned = manifest(fixture)
    runner.train_stage(plan, 'a1', datasets, trigger, poisoned)
    marker = runner.verify_training(plan, 'a1')
    assert marker is not None
    assert 'TRAINED_NOT_EVALUATED' in json.dumps(marker)
    assert not cell(plan, 'clean').exists()
    assert not (cell(plan, 'a1')/'evaluation.json').exists()
    assert not (cell(plan, 'a1')/'predictions.npz').exists()
    blob = checkpoint(plan, 'a1', final=True)
    assert blob['epoch'] == 49
    assert 'clean_checkpoint_sha256' not in blob
    with pytest.raises(ValueError, match='Clean|clean'):
        runner.evaluate_trained_stage(plan, 'a1', datasets, trigger)
    before = tree(fixture['out'])
    monkeypatch.setattr(runner, 'build_model', lambda: pytest.fail('completed training constructs model'))
    runner.train_stage(plan, 'a1', datasets, trigger, poisoned)
    assert tree(fixture['out']) == before


def test_full_training_then_serial_evaluation_and_verified_cache(fixture, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CPU run probed CUDA'))
    plan = complete(fixture)
    clean, a1 = [runner.verify_stage(plan, stage) for stage in ('clean', 'a1')]
    assert clean['checkpoint_epoch'] == a1['checkpoint_epoch'] == 49
    assert a1['clean_result_sha256'] == runner.BASE.util.object_sha(clean)
    assert a1['measurements']['controls']['d0_exact_identity']
    assert len(a1['measurements']['dose_rows']) == 6
    runner.export_report(plan)
    assert (fixture['out']/'wipose_pilot.complete.json').is_file()
    before = tree(fixture['out'])
    monkeypatch.setattr(runner, 'build_model', lambda: pytest.fail('cache constructed model'))
    monkeypatch.setattr(runner.BASE, 'build_model', lambda: pytest.fail('cache constructed BASE model'))
    monkeypatch.setattr(runner.BASE, 'infer', lambda *a, **k: pytest.fail('cache inferred'))
    assert runner.main(cli(fixture, '--status')) == 0
    assert runner.main(cli(fixture, '--check-only')) == 0
    for stage in ('clean', 'a1'):
        runner.train_stage(plan, stage, fixture['datasets'], None, None)
        runner.evaluate_trained_stage(plan, stage, fixture['datasets'], None)
    assert tree(fixture['out']) == before


@pytest.mark.parametrize('stage', ['clean', 'a1'])
def test_exact_scratch_weights_optimizer_all_rng_resume(fixture, monkeypatch, stage):
    baseline, datasets, trigger, poisoned = manifest(fixture)
    runner.train_stage(baseline, stage, datasets, trigger, poisoned)
    expected = checkpoint(baseline, stage, final=True)
    resumed, datasets, trigger, poisoned = manifest(fixture, fixture['out'].parent/'resumed')
    original = runner.BASE.util.atomic_checkpoint
    def interrupt(path, blob):
        original(path, blob)
        if blob['stage'] == stage and blob['epoch'] == 17:
            raise RuntimeError('synthetic interruption after atomic epoch17')
    monkeypatch.setattr(runner.BASE.util, 'atomic_checkpoint', interrupt)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        runner.train_stage(resumed, stage, datasets, trigger, poisoned)
    assert checkpoint(resumed, stage)['epoch'] == 17
    assert runner.verify_training(resumed, stage) is None
    monkeypatch.setattr(runner.BASE.util, 'atomic_checkpoint', original)
    runner.train_stage(resumed, stage, datasets, trigger, poisoned)
    actual = checkpoint(resumed, stage, final=True)
    for key in ('epoch', 'model', 'optimizer', 'rng', 'execution', 'train_loss'):
        assert_same(actual[key], expected[key])


@pytest.mark.parametrize('stage', ['clean', 'a1'])
def test_same_frozen_single_job_numerics_as_original_serial_runner(fixture, stage):
    serial_path = Path(serial.__file__)
    old_sha = hashlib.sha256(serial_path.read_bytes()).hexdigest()
    original_cfg = serial.configuration(fixture['home'], fixture['out'].parent/'serial', 'cpu',
                                        fixture['root'], fixture['action'])
    original_plan, datasets, trigger, poisoned = serial.prepare(original_cfg)
    serial.train_stage(original_plan, 'clean', datasets, trigger, poisoned)
    if stage == 'a1':
        serial.train_stage(original_plan, 'a1', datasets, trigger, poisoned)
    baseline = serial.checkpoint_header(serial.cell_dir(original_plan, stage)/'checkpoint.pt',
                                        original_plan, stage, final=True)
    plan, datasets, trigger, poisoned = manifest(fixture)
    runner.train_stage(plan, stage, datasets, trigger, poisoned)
    actual = checkpoint(plan, stage, final=True)
    for key in ('epoch', 'model', 'optimizer', 'rng', 'execution', 'train_loss'):
        assert_same(actual[key], baseline[key])
    assert hashlib.sha256(serial_path.read_bytes()).hexdigest() == old_sha


@pytest.mark.parametrize('mutation', ['marker', 'checkpoint', 'poison'])
def test_training_marker_checkpoint_poison_tamper_rejected(fixture, mutation):
    plan, datasets, trigger, poisoned = manifest(fixture)
    runner.train_stage(plan, 'a1', datasets, trigger, poisoned)
    folder = cell(plan, 'a1')
    if mutation == 'marker':
        marker = runner.verify_training(plan, 'a1')
        marker['plan_sha256'] = 'wrong'
        runner.BASE.util.atomic_json(folder/'training.complete.json', marker)
    elif mutation == 'checkpoint':
        blob = checkpoint(plan, 'a1')
        blob['train_loss'] += .01
        runner.BASE.util.atomic_checkpoint(folder/'checkpoint.pt', blob)
    else:
        (folder/'poison.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError):
        runner.verify_training(plan, 'a1')


@pytest.mark.parametrize('field', ['t1_mpjpe_px', 'target_pck_pct', 'mean_dce_px'])
def test_result_metric_replay_rejects_scalar_tampering(fixture, field):
    plan = complete(fixture)
    result = runner.verify_stage(plan, 'a1')
    result['measurements']['mainmetrics'][field] += .01
    runner.BASE.util.atomic_json(cell(plan, 'a1')/'evaluation.json', result)
    with pytest.raises(ValueError, match='replay'):
        runner.verify_stage(plan, 'a1')


def test_no_retry_or_batch_fallback_on_nonfinite_training(fixture, monkeypatch):
    plan, datasets, trigger, poisoned = manifest(fixture)
    class BrokenPose(TinyPose):
        def forward(self, csi):
            output, feature = super().forward(csi)
            return output*float('nan'), feature
    monkeypatch.setattr(runner, 'build_model', BrokenPose)
    monkeypatch.setattr(runner.BASE, 'build_model', BrokenPose)
    with pytest.raises(RuntimeError, match='Nonfinite'):
        runner.train_stage(plan, 'clean', datasets, trigger, poisoned)
    assert not (cell(plan, 'clean')/'checkpoint.pt').exists()
    assert not (cell(plan, 'clean')/'training.complete.json').exists()
    assert plan['cfg']['batch_size'] == 32


def scheduler_probe(monkeypatch, slots, failing=None, verified=()):
    """Never starts a child process, reads data, or acquires a GPU."""
    launched, finished, active, markers, processes = [], [], set(), set(verified), []
    def verified_training(plan, stage):
        return {'status': 'TRAINED_NOT_EVALUATED'} if stage in markers else None
    monkeypatch.setattr(runner, 'verify_training', verified_training)
    monkeypatch.setattr(runner, 'verify_stage', lambda *args: None)
    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.stage = command[command.index('--_worker')+1]
            assert command[command.index('--_phase')+1] == 'train'
            assert self.stage not in launched, 'unexpected automatic retry'
            self.pid = 9000+len(launched)
            self.stdout = kwargs.get('stdout')
            self.returncode = None
            self.polls = 0
            launched.append(self.stage)
            active.add(self.stage)
            self.concurrent = len(active)
            processes.append(self)
            assert len(active) <= slots, 'scheduler overbooked its configured slots'

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            self.polls += 1
            if self.polls <= 2:
                return None
            self.returncode = 7 if self.stage == failing else 0
            active.remove(self.stage)
            finished.append(self.stage)
            if self.returncode == 0:
                markers.add(self.stage)
            return self.returncode

        def wait(self, timeout=None):
            while self.poll() is None:
                pass
            return self.returncode

        def terminate(self):
            pytest.fail('scheduler must drain its already-running workers, not kill them')

        def kill(self):
            pytest.fail('scheduler must not kill children automatically')
    monkeypatch.setattr(runner.subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(runner.time, 'sleep', lambda *args: None)
    return launched, finished, active, markers, processes


@pytest.mark.parametrize('slots', [1, 2])
def test_scheduler_respects_capacity_and_authenticates_all_owned_completions(fixture, monkeypatch, slots):
    plan, *_ = manifest(fixture)
    launched, finished, active, markers, processes = scheduler_probe(monkeypatch, slots)
    runner.schedule_training(plan, jobs_per_gpu=slots)
    assert set(launched) == set(finished) == set(markers) == {'clean', 'a1'}
    assert not active
    assert all(process.returncode == 0 for process in processes)
    assert max(process.concurrent for process in processes) == slots
    assert all(process.stdout is None or process.stdout.closed for process in processes)


def test_scheduler_reuses_verified_training_without_duplicate_worker(fixture, monkeypatch):
    plan, *_ = manifest(fixture)
    launched, finished, active, markers, _ = scheduler_probe(monkeypatch, 2, verified=('clean',))
    runner.schedule_training(plan, jobs_per_gpu=2)
    assert launched == finished == ['a1']
    assert not active and markers == {'clean', 'a1'}


def test_first_failure_stops_future_launches_without_automatic_retry(fixture, monkeypatch):
    plan, *_ = manifest(fixture)
    launched, finished, active, markers, processes = scheduler_probe(monkeypatch, 1, failing='clean')
    with pytest.raises(RuntimeError):
        runner.schedule_training(plan, jobs_per_gpu=1)
    assert launched == finished == ['clean']
    assert not active and not markers
    assert all(process.stdout is None or process.stdout.closed for process in processes)


def test_failure_drains_other_already_running_owned_worker_before_returning(fixture, monkeypatch):
    plan, *_ = manifest(fixture)
    launched, finished, active, markers, processes = scheduler_probe(monkeypatch, 2, failing='clean')
    with pytest.raises(RuntimeError):
        runner.schedule_training(plan, jobs_per_gpu=2)
    assert launched == ['clean', 'a1']
    assert set(finished) == {'clean', 'a1'}
    assert not active and markers == {'a1'}
    assert len(processes) == 2 and all(process.returncode is not None for process in processes)
    assert all(process.stdout is None or process.stdout.closed for process in processes)


def test_final_checkpoint_without_marker_recovers_without_optimizer_update(fixture, monkeypatch):
    plan, datasets, trigger, poisoned = manifest(fixture)
    original = runner.BASE.util.atomic_json
    def interrupt(path, value):
        if Path(path).name == 'training.complete.json':
            raise RuntimeError('interrupted before training marker publication')
        return original(path, value)
    monkeypatch.setattr(runner.BASE.util, 'atomic_json', interrupt)
    with pytest.raises(RuntimeError, match='before training marker'):
        runner.train_stage(plan, 'a1', datasets, trigger, poisoned)
    baseline = checkpoint(plan, 'a1', final=True)
    assert runner.verify_training(plan, 'a1') is None
    monkeypatch.setattr(runner.BASE.util, 'atomic_json', original)
    monkeypatch.setattr(torch.optim.SGD, 'step', lambda *a, **k: pytest.fail('recovery retrained final epoch'))
    runner.train_stage(plan, 'a1', datasets, trigger, poisoned)
    assert runner.verify_training(plan, 'a1') is not None
    actual = checkpoint(plan, 'a1', final=True)
    for key in ('epoch', 'model', 'optimizer', 'rng', 'execution', 'train_loss'):
        assert_same(actual[key], baseline[key])


@pytest.mark.parametrize('arguments', [
    ['--_worker', 'clean'], ['--_phase', 'train'], ['--_manifest', 'x.json'],
    ['--_worker', 'clean', '--_phase', 'train'],
    ['--_worker', 'clean', '--_manifest', 'x.json'],
    ['--_phase', 'train', '--_manifest', 'x.json'],
    ['--_worker', 'clean', '--_phase', 'eval', '--_manifest', 'x.json', '--status'],
])
def test_private_worker_requires_complete_phase_triplet_and_no_public_modes(fixture, arguments):
    with pytest.raises(SystemExit):
        runner.parse_args(cli(fixture, *arguments))


def test_trained_but_unevaluated_status_is_readonly_without_data_model_gpu(fixture, monkeypatch, capsys):
    plan, datasets, trigger, poisoned = manifest(fixture)
    runner.train_stage(plan, 'a1', datasets, trigger, poisoned)
    capsys.readouterr()
    before = tree(fixture['out'])
    def fail(*args, **kwargs):
        pytest.fail('status touched model/data/CUDA')
    monkeypatch.setattr(runner, 'prepare', fail)
    monkeypatch.setattr(runner, 'build_model', fail)
    monkeypatch.setattr(runner.BASE, 'build_model', fail)
    monkeypatch.setattr(torch.cuda, 'is_available', fail)
    assert runner.main(cli(fixture, '--status')) == 0
    stdout = capsys.readouterr().out
    assert 'a1: TRAINED_NOT_EVALUATED' in stdout
    assert 'clean: INCOMPLETE' in stdout
    assert 'VERIFIED_SAVED_ARTIFACTS_COMPLETE' not in stdout
    assert tree(fixture['out']) == before


def test_parallel_checkpoint_rejects_clean_dependency_wrong_epoch_and_execution(fixture):
    plan, datasets, trigger, poisoned = manifest(fixture)
    runner.train_stage(plan, 'a1', datasets, trigger, poisoned)
    baseline = checkpoint(plan, 'a1', final=True)
    candidate = cell(plan, 'a1')/'candidate_header.pt'
    for update in ({'clean_checkpoint_sha256': 'forbidden-warmstart-binding'},
                   {'epoch': -1}, {'epoch': 50}, {'train_loss': float('nan')},
                   {'stage': 'clean'}, {'profile': serial.PROFILE}):
        changed = copy.deepcopy(baseline)
        changed.update(update)
        runner.BASE.util.atomic_checkpoint(candidate, changed)
        with pytest.raises(ValueError, match='binding'):
            runner.checkpoint_header(candidate, plan, 'a1')
    with pytest.raises(ValueError, match='Execution'):
        checkpoint(plan, 'a1', execution={'changed': True})
    changed = copy.deepcopy(baseline)
    changed['epoch'] = 48
    runner.BASE.util.atomic_checkpoint(candidate, changed)
    with pytest.raises(ValueError, match='final epoch'):
        runner.checkpoint_header(candidate, plan, 'a1', final=True)


def test_second_worker_spawn_exception_drains_already_launched_peer(fixture, monkeypatch):
    plan, *_ = manifest(fixture)
    launched, finished, active, markers, processes = scheduler_probe(monkeypatch, 2)
    normal_popen = runner.subprocess.Popen
    rejected_logs = []
    def fail_second(command, **kwargs):
        stage = command[command.index('--_worker')+1]
        if stage == 'a1':
            rejected_logs.append(kwargs['stdout'])
            raise OSError('synthetic second worker spawn failure')
        return normal_popen(command, **kwargs)
    monkeypatch.setattr(runner.subprocess, 'Popen', fail_second)
    with pytest.raises(OSError, match='spawn failure'):
        runner.schedule_training(plan, jobs_per_gpu=2)
    assert launched == finished == ['clean']
    assert not active and markers == {'clean'}
    assert len(processes) == 1 and processes[0].returncode == 0
    assert processes[0].stdout.closed and all(log.closed for log in rejected_logs)


def test_real_two_toy_subprocesses_overlap_and_both_exit_without_gpu(fixture, monkeypatch):
    """Real OS processes exercise pool lifetime; toy markers are not training."""
    plan, *_ = manifest(fixture)
    script = '''import json, os, sys, time
from pathlib import Path
root, stage = Path(sys.argv[1]), sys.argv[2]
folder = root/'hpeli'/'seed42'/stage
folder.mkdir(parents=True, exist_ok=True)
(root/(stage+'.toy.started')).write_text(str(os.getpid()), encoding='utf-8')
deadline = time.monotonic()+8
while not all((root/(name+'.toy.started')).is_file() for name in ('clean', 'a1')):
    if time.monotonic() >= deadline:
        raise SystemExit(29)
    time.sleep(.01)
time.sleep(.05)
(folder/'training.complete.json').write_text(json.dumps({
    'toy_only': True, 'pid': os.getpid(), 'stage': stage,
    'status': 'TRAINED_NOT_EVALUATED', 'peer_started': True}), encoding='utf-8')
'''
    def command(manifest, stage, phase):
        assert phase == 'train'
        return [sys.executable, '-X', 'utf8', '-c', script, manifest['cfg']['outdir'], stage]
    def verify_toy(manifest, stage):
        path = cell(manifest, stage)/'training.complete.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    monkeypatch.setattr(runner, '_worker_command', command)
    monkeypatch.setattr(runner, 'verify_training', verify_toy)
    runner.schedule_training(plan, jobs_per_gpu=2)
    markers = [verify_toy(plan, stage) for stage in ('clean', 'a1')]
    assert all(marker['toy_only'] and marker['peer_started'] for marker in markers)
    assert len({marker['pid'] for marker in markers}) == 2
    assert all(marker['pid'] != os.getpid() for marker in markers)
    assert not any((cell(plan, stage)/'checkpoint.pt').exists() for stage in ('clean', 'a1'))


def test_normal_main_path_records_launch_trains_both_then_evaluates_serially(fixture, monkeypatch):
    timeline = []
    def train_selected(plan, jobs_per_gpu=2, stages=runner.STAGES):
        assert jobs_per_gpu == 2 and tuple(stages) == ('clean', 'a1')
        datasets = fixture['datasets']
        _, _, trigger, poisoned = runner.prepare(plan['cfg'])
        # Deliberately finish A1 first; evaluation must still wait for Clean.
        for stage in ('a1', 'clean'):
            runner.train_stage(plan, stage, datasets, trigger, poisoned)
            assert runner.verify_training(plan, stage) is not None
            timeline.append(('trained', stage))
    def evaluate_selected(plan, stage):
        assert all(runner.verify_training(plan, name) for name in ('clean', 'a1'))
        assert timeline[:2] == [('trained', 'a1'), ('trained', 'clean')]
        timeline.append(('evaluate', stage))
        _, datasets, trigger, _ = runner.prepare(plan['cfg'])
        runner.evaluate_trained_stage(plan, stage, datasets, trigger)
    monkeypatch.setattr(runner, 'schedule_training', train_selected)
    monkeypatch.setattr(runner, 'evaluate_in_worker', evaluate_selected)
    assert not fixture['out'].exists()
    assert runner.main(cli(fixture, '--jobs-per-gpu', '2')) == 0
    assert timeline == [('trained', 'a1'), ('trained', 'clean'),
                        ('evaluate', 'clean'), ('evaluate', 'a1')]
    plan = runner.load_manifest(fixture['out']/runner.MANIFEST)
    launches = list((fixture['out']/'runtime_launches').glob('*.json'))
    assert len(launches) == 1
    launch = json.loads(launches[0].read_text(encoding='utf-8'))
    assert launch['jobs_per_gpu'] == 2 and launch['stages'] == ['clean', 'a1']
    assert launch['plan_sha256'] == plan['plan_sha256']
    assert (fixture['out']/'wipose_pilot.md').is_file()
    assert (fixture['out']/'wipose_pilot.complete.json').is_file()
    before = tree(fixture['out'])
    monkeypatch.setattr(runner, 'build_model', lambda: pytest.fail('check-only model construction'))
    monkeypatch.setattr(runner.BASE, 'build_model', lambda: pytest.fail('check-only BASE model construction'))
    assert runner.main(cli(fixture, '--check-only')) == 0
    assert tree(fixture['out']) == before


def test_duplicate_stages_refused_before_any_worker_launch(fixture, monkeypatch):
    plan, *_ = manifest(fixture)
    def fail(*args, **kwargs):
        pytest.fail('duplicate stage launched worker')
    monkeypatch.setattr(runner.subprocess, 'Popen', fail)
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        runner.schedule_training(plan, jobs_per_gpu=2, stages=('clean', 'clean'))
