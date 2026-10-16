"""Resume-only scheduler tests: no benchmark training, data or CUDA access."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_a1_three_tables as frozen
import run_a1_three_tables_parallel as parallel


@pytest.fixture
def existing(tmp_path):
    matrix = frozen.build_matrix(tmp_path / 'data', tmp_path / 'out',
                                 retrain_seed42=True)
    root = Path(matrix['outdir'])
    root.mkdir(parents=True)
    manifest = root / frozen.MANIFEST
    manifest.write_text(json.dumps(matrix, indent=2), encoding='utf-8')
    return matrix, manifest


def _snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(root).rglob('*') if path.is_file()}


def _forbidden(*args, **kwargs):
    pytest.fail('Read-only or rejected request reached training/data/GPU/dispatch')


def test_defaults_resume_existing_plan_with_two_slots():
    args = parallel.parse_args(['--outdir', 'saved'])
    assert args.jobs_per_gpu == 2
    assert args.devices == ['cuda:0']
    assert args.data_home is None


def test_outdir_required():
    with pytest.raises(SystemExit):
        parallel.parse_args([])


@pytest.mark.parametrize('extra', [
    ['--jobs-per-gpu', '0'], ['--jobs-per-gpu', '-1'],
    ['--jobs-per-gpu', '1.5'], ['--devices', 'cuda:0', 'cuda:0'],
    ['--devices', 'cuda:-1'], ['--devices', 'cuda:01'],
    ['--dry-run', '--status'], ['--dry-run', '--export-only'],
])
def test_invalid_runtime_options_never_dispatch(existing, monkeypatch, extra):
    matrix, _ = existing
    monkeypatch.setattr(parallel, '_schedule', _forbidden)
    monkeypatch.setattr(frozen, '_preflight', _forbidden)
    with pytest.raises((ValueError, SystemExit)):
        parallel.main(['--outdir', matrix['outdir'], *extra])


@pytest.mark.parametrize('flag', ['--fresh', '--retrain-seed42'])
def test_no_plan_creation_or_retraining_flags(existing, flag):
    matrix, _ = existing
    with pytest.raises(SystemExit):
        parallel.parse_args(['--outdir', matrix['outdir'], flag])


def test_missing_manifest_rejected_without_creating_output(tmp_path, monkeypatch):
    root = tmp_path / 'absent'
    monkeypatch.setattr(parallel, '_schedule', _forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        parallel.main(['--outdir', str(root), '--dry-run'])
    assert not root.exists()


@pytest.mark.parametrize('option', ['--dry-run', '--status'])
def test_inspection_is_readonly_and_needs_no_data_gpu_lock(existing, monkeypatch, option):
    matrix, manifest = existing
    checkpoint = Path(matrix['cells'][0]['ckpt_dir']) / 'checkpoint.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'preserve checkpoint at saved epoch')
    root = Path(matrix['outdir'])
    before = _snapshot(root)
    monkeypatch.setattr(frozen, '_preflight', _forbidden)
    monkeypatch.setattr(frozen, '_current_inputs', _forbidden)
    monkeypatch.setattr(frozen, '_cell_lock', _forbidden)
    monkeypatch.setattr(parallel, '_schedule', _forbidden)
    assert parallel.main(['--outdir', str(root), option]) == 0
    assert _snapshot(root) == before
    assert json.loads(manifest.read_text())['plan_sha256'] == matrix['plan_sha256']
    assert not Path(matrix['data_home']).exists()


@pytest.mark.parametrize('field', ['data_home', 'num_workers'])
def test_cannot_change_existing_runtime_worker_or_data_policy(existing, field):
    matrix, _ = existing
    args = ['--outdir', matrix['outdir'], '--dry-run']
    args += (['--data-home', str(Path(matrix['data_home']).parent / 'different')]
             if field == 'data_home' else ['--num-workers', '3'])
    with pytest.raises(ValueError):
        parallel.main(args)


@pytest.mark.parametrize('mutation', ['plan_hash', 'scientific_config', 'source'])
def test_canonical_manifest_tampering_rejected(existing, monkeypatch, mutation):
    matrix, manifest = existing
    monkeypatch.setattr(parallel, '_schedule', _forbidden)
    if mutation == 'plan_hash':
        matrix['plan_sha256'] = '0' * 64
    elif mutation == 'scientific_config':
        matrix['cells'][1]['cfg']['epochs'] = 15
        matrix['plan_sha256'] = frozen._plan_hash(matrix)
    else:
        matrix['sources']['train_backdoor.py'] = '0' * 64
        matrix['plan_sha256'] = frozen._plan_hash(matrix)
    manifest.write_text(json.dumps(matrix), encoding='utf-8')
    before = _snapshot(Path(matrix['outdir']))
    with pytest.raises(ValueError):
        parallel.main(['--outdir', matrix['outdir'], '--dry-run'])
    assert _snapshot(Path(matrix['outdir'])) == before


def test_source_change_rejects_frozen_plan(existing, monkeypatch):
    matrix, _ = existing
    sources = frozen._source_provenance()
    monkeypatch.setattr(frozen, '_source_provenance',
                        lambda: dict(sources, changed_runtime_source='different'))
    with pytest.raises(ValueError, match='Scientific|source|changed|canonical'):
        parallel.main(['--outdir', matrix['outdir'], '--dry-run'])


def test_running_original_controller_lock_blocks_parallel_dispatch(existing, monkeypatch):
    matrix, _ = existing
    monkeypatch.setattr(parallel, '_schedule', _forbidden)
    monkeypatch.setattr(frozen, '_preflight', _forbidden)
    with frozen._cell_lock(matrix['outdir']):
        with pytest.raises(RuntimeError, match='Another|running|lock'):
            parallel.main(['--outdir', matrix['outdir']])


def test_surviving_original_worker_lock_blocks_dispatch(existing, monkeypatch):
    matrix, _ = existing
    cell = matrix['cells'][0]
    monkeypatch.setattr(parallel, '_schedule', _forbidden)
    monkeypatch.setattr(frozen, '_preflight', _forbidden)
    with frozen._cell_lock(cell['run_dir']):
        with pytest.raises(RuntimeError, match='Another|running|lock'):
            parallel.main(['--outdir', matrix['outdir'],
                           '--cells', cell['method_key']])


@pytest.mark.parametrize('cells', [
    ['s42__a1', 's42__a1'], ['s42__blended'], ['s41__a1'],
])
def test_invalid_selected_cells_rejected_before_dispatch(existing, monkeypatch, cells):
    matrix, _ = existing
    monkeypatch.setattr(parallel, '_schedule', _forbidden)
    with pytest.raises(ValueError):
        parallel.main(['--outdir', matrix['outdir'], '--cells', *cells,
                       '--dry-run'])


def _fake_processes(matrix, monkeypatch, *, fail_first=False):
    processes, active, peaks = [], [], []
    cells = {cell['method_key']: cell for cell in matrix['cells']}

    class Child:
        def __init__(self, command, **kwargs):
            self.command = command
            self.pid = 7000 + len(processes)
            self.polls = 0
            self.terminated = False
            self.waited = False
            self.stream = kwargs['stdout']
            self.cell = cells[command[command.index('--_cell') + 1]]
            self.device = self.cell['cfg']['device']
            processes.append(self)
            active.append(self)
            peaks.append(len(active))

        def poll(self):
            self.polls += 1
            if fail_first and self is processes[0]:
                active.remove(self)
                return 1
            if self.polls == 1:
                return None
            if self in active:
                active.remove(self)
                artifact = Path(self.cell['artifact_dir']) / 'evaluation.json'
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text('{}', encoding='utf-8')
            return 0

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

    monkeypatch.setattr(parallel.subprocess, 'Popen', Child)
    monkeypatch.setattr(parallel.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(parallel, '_completed', lambda cell:
                        (Path(cell['artifact_dir']) / 'evaluation.json').exists())
    return processes, peaks


@pytest.mark.parametrize('devices,slots,expected', [
    (['cuda:0'], 2, 2),
    (['cuda:0', 'cuda:1'], 2, 4),
    (['cuda:0'], 1, 1),
])
def test_parallel_slots_preserve_scientific_plan_and_resume_paths(
        existing, monkeypatch, devices, slots, expected):
    matrix, manifest = existing
    before = copy.deepcopy(matrix)
    selected = [cell['method_key'] for cell in matrix['cells'][:5]]
    checkpoint = Path(matrix['cells'][0]['ckpt_dir']) / 'checkpoint.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'saved optimizer RNG epoch state')
    source_hashes = frozen._source_provenance()
    processes, peaks = _fake_processes(matrix, monkeypatch)
    parallel._schedule(matrix, manifest, devices, slots, selected)
    assert len(processes) == len(selected)
    assert max(peaks) == expected
    assert {child.cell['method_key'] for child in processes} == set(selected)
    assert all(child.stream.closed for child in processes)
    for child in processes:
        assert child.command[0] == sys.executable
        assert Path(child.command[2]).name == 'run_a1_three_tables.py'
        assert child.command[child.command.index('--_manifest') + 1] == str(manifest)
        assert child.cell['ckpt_dir'] == next(
            cell['ckpt_dir'] for cell in before['cells']
            if cell['method_key'] == child.cell['method_key'])
    # The first simultaneous wave must not overbook any logical device.
    first_wave = processes[:expected]
    assert all(sum(child.device == device for child in first_wave) <= slots
               for device in devices)
    assert checkpoint.read_bytes() == b'saved optimizer RNG epoch state'
    assert frozen._source_provenance() == source_hashes
    assert frozen._plan_hash(matrix) == before['plan_sha256']
    saved = json.loads(manifest.read_text(encoding='utf-8'))
    assert saved['plan_sha256'] == before['plan_sha256']
    frozen._validate_manifest(saved)


def test_completed_cells_not_redispatched(existing, monkeypatch):
    matrix, manifest = existing
    selected = [cell['method_key'] for cell in matrix['cells'][:3]]
    complete = Path(matrix['cells'][0]['artifact_dir']) / 'evaluation.json'
    complete.parent.mkdir(parents=True)
    complete.write_text('{}', encoding='utf-8')
    processes, _ = _fake_processes(matrix, monkeypatch)
    parallel._schedule(matrix, manifest, ['cuda:0'], 2, selected)
    assert [child.cell['method_key'] for child in processes] == selected[1:]


def test_failed_child_terminates_only_this_scheduler_children(existing, monkeypatch):
    matrix, manifest = existing
    processes, _ = _fake_processes(matrix, monkeypatch, fail_first=True)
    with pytest.raises(RuntimeError, match='failed|exit'):
        parallel._schedule(matrix, manifest, ['cuda:0'], 2)
    assert len(processes) == 2
    assert not processes[0].terminated
    assert processes[1].terminated and processes[1].waited
    assert all(child.stream.closed for child in processes)


def test_popen_failure_closes_new_stream_and_stops_owned_children(existing, monkeypatch):
    matrix, manifest = existing
    processes, _ = _fake_processes(matrix, monkeypatch)
    factory = parallel.subprocess.Popen
    failed_streams = []

    def fail_second_launch(command, **kwargs):
        if processes:
            failed_streams.append(kwargs['stdout'])
            raise OSError('Synthetic second worker spawn failure')
        return factory(command, **kwargs)

    monkeypatch.setattr(parallel.subprocess, 'Popen', fail_second_launch)
    monkeypatch.setattr(parallel.subprocess, 'run', _forbidden)
    with pytest.raises(OSError, match='spawn failure'):
        parallel._schedule(matrix, manifest, ['cuda:0'], 2)
    assert len(processes) == 1
    assert processes[0].terminated and processes[0].waited
    assert processes[0].stream.closed
    assert len(failed_streams) == 1 and failed_streams[0].closed


def test_partial_checkpoint_resumes_actual_frozen_worker_training(
        tmp_path, monkeypatch, capsys):
    """Synthetic CPU victim: unchanged 50 epochs, interruption after saved epoch 0."""
    import numpy as np
    import torch
    import a1_three_tables_eval as evaluator
    import train_backdoor as trainer

    class TinyData:
        def __init__(self, split, data_root, **kwargs):
            self.data_root = str(data_root)
            count = 10 if split == 'training' else 3
            self.items = [dict(csi=str(Path(data_root) / split / f'csi_{index}.npy'),
                kpt=str(Path(data_root) / split / 'ground_truth.npy'), frame_idx=index)
                for index in range(count)]
            self.raw = np.random.default_rng(9).uniform(
                .1, .8, (count, 3, 114, 10)).astype(np.float32)
            self.poses = np.random.default_rng(19).normal(
                0, .2, (10, 1, 17, 3)).astype(np.float32)

        def __len__(self):
            return len(self.items)

        def load_raw(self, path):
            return self.raw[int(Path(path).stem.split('_')[-1])].copy()

        @staticmethod
        def normalize(raw):
            return np.clip(raw, 0, 1).astype(np.float32)

        def load_pose(self, path, frame_idx):
            return self.poses[frame_idx].copy()

    class TinyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3, 51)

        def forward(self, csi):
            return self.head(csi.mean((-2, -1))).reshape(-1, 1, 17, 3), None

    monkeypatch.setattr(trainer, 'MMFI', TinyData)
    monkeypatch.setattr(trainer, 'build_model', lambda *args, **kwargs: TinyVictim())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *args: None)
    plan = frozen.build_matrix(tmp_path / 'data', tmp_path / 'out',
                               device='cpu', num_workers=0, retrain_seed42=True)
    cell = next(item for item in plan['cells']
                if item['seed'] == 42 and item['key'] == 'badnets')
    assert cell['cfg']['epochs'] == 50
    monkeypatch.setattr(frozen, '_current_inputs', lambda item: {'synthetic': True})
    monkeypatch.setattr(frozen, '_read_completed', lambda item: {'cfg': item['cfg']})
    monkeypatch.setattr(frozen, '_audit_run', lambda *args: {'synthetic': True})
    monkeypatch.setattr(evaluator, 'evaluate_cell', lambda *args: {'synthetic': True})
    real_save = trainer._save_checkpoint
    saved_epochs = []

    class ControlledPause(RuntimeError):
        pass

    def save_then_pause(path, model, optimizer, epoch, best_loss, cfg, **kwargs):
        real_save(path, model, optimizer, epoch, best_loss, cfg, **kwargs)
        saved_epochs.append(epoch)
        if epoch == 0:
            raise ControlledPause('Pause after a valid saved checkpoint')

    monkeypatch.setattr(trainer, '_save_checkpoint', save_then_pause)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.raises(ControlledPause):
            frozen._execute_cell(cell)
        checkpoint = Path(cell['ckpt_dir']) / 'checkpoint.pt'
        partial = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert partial['epoch'] == 0 and partial['cfg']['epochs'] == 50
        assert partial['optimizer']['state']

        def save_remaining(path, model, optimizer, epoch, best_loss, cfg, **kwargs):
            saved_epochs.append(epoch)
            real_save(path, model, optimizer, epoch, best_loss, cfg, **kwargs)

        monkeypatch.setattr(trainer, '_save_checkpoint', save_remaining)
        assert frozen._execute_cell(cell) == {'synthetic': True}
        resumed = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert resumed['epoch'] == 49 and resumed['cfg']['epochs'] == 50
        assert saved_epochs == list(range(50))  # epoch 0 was not repeated
        assert '[ckpt] resumed from epoch 0' in capsys.readouterr().out

        monkeypatch.setattr(trainer, '_save_checkpoint', real_save)
        whole_dir = tmp_path / 'uninterrupted'
        trainer.train(cell['cfg'], ckpt_dir=whole_dir)
        whole = torch.load(whole_dir / 'checkpoint.pt', map_location='cpu', weights_only=False)
        for key in whole['model']:
            torch.testing.assert_close(whole['model'][key], resumed['model'][key],
                                       rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)


def test_missing_evaluation_is_not_completed(existing, monkeypatch):
    matrix, _ = existing
    monkeypatch.setattr(frozen, '_read_completed', _forbidden)
    assert parallel._completed(matrix['cells'][0]) is False


def test_existing_evaluation_needs_audit_and_verified_binding(existing, monkeypatch):
    import a1_three_tables_eval as evaluator
    matrix, _ = existing
    cell = matrix['cells'][1]
    path = Path(cell['artifact_dir']) / 'evaluation.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}', encoding='utf-8')
    current = {'ordered_inputs': 'unchanged'}
    calls = []
    monkeypatch.setattr(frozen, '_read_completed', lambda item:
                        calls.append('read') or {'cfg': item['cfg']})
    monkeypatch.setattr(frozen, '_current_inputs', lambda item:
                        calls.append('inputs') or current)
    monkeypatch.setattr(frozen, '_audit_run', lambda *args:
                        calls.append('audit') or {})

    def verify(evaluating, folder):
        calls.append('load')
        assert evaluating['verified_inputs'] == current
        assert str(folder) == cell['artifact_dir']
        return {'valid': True}

    monkeypatch.setattr(evaluator, 'load_evaluation', verify)
    assert parallel._completed(cell) is True
    assert set(calls) == {'read', 'inputs', 'audit', 'load'}
    assert calls.index('audit') < calls.index('load')


def test_stale_bound_evaluation_requires_replay(existing, monkeypatch):
    import a1_three_tables_eval as evaluator
    matrix, _ = existing
    cell = matrix['cells'][1]
    path = Path(cell['artifact_dir']) / 'evaluation.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(frozen, '_read_completed', lambda item: {'cfg': item['cfg']})
    monkeypatch.setattr(frozen, '_current_inputs', lambda item: {})
    monkeypatch.setattr(frozen, '_audit_run', lambda *args: {})
    monkeypatch.setattr(evaluator, 'load_evaluation', lambda *args: None)
    assert parallel._completed(cell) is False


def test_evaluation_audit_failure_is_not_silently_skipped(existing, monkeypatch):
    matrix, _ = existing
    cell = matrix['cells'][1]
    path = Path(cell['artifact_dir']) / 'evaluation.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(frozen, '_read_completed', lambda item: {'cfg': item['cfg']})
    monkeypatch.setattr(frozen, '_current_inputs', lambda item: {})

    def failed_audit(*args):
        raise ValueError('input identities changed')

    monkeypatch.setattr(frozen, '_audit_run', failed_audit)
    with pytest.raises(ValueError, match='identities'):
        parallel._completed(cell)


def test_selected_partial_never_exports_complete_tables(existing, monkeypatch):
    matrix, _ = existing
    selected = [matrix['cells'][0]['method_key']]
    dispatched = []
    monkeypatch.setattr(frozen, '_preflight', lambda *args: None)
    monkeypatch.setattr(frozen, '_export', _forbidden)
    monkeypatch.setattr(parallel, '_schedule', lambda *args, **kwargs:
                        dispatched.append((args, kwargs)))
    assert parallel.main(['--outdir', matrix['outdir'], '--cells', *selected]) == 0
    assert len(dispatched) == 1
    assert not (Path(matrix['outdir']) / 'three_main_tables.json').exists()


def test_cleanup_error_does_not_leave_other_owned_slots_running(tmp_path, capsys):
    class Child:
        def __init__(self, pid, fail=False):
            self.pid, self.fail, self.stopped = pid, fail, False

        def poll(self):
            return None

        def terminate(self):
            if self.fail:
                raise OSError('already unreachable')
            self.stopped = True

        def wait(self, timeout=None):
            return 0

    first, second = Child(100, True), Child(101)
    first_log, second_log = (tmp_path / 'one.log').open('a'), (tmp_path / 'two.log').open('a')
    parallel._stop_owned({('cuda:0', 0): (first, {}, first_log),
                          ('cuda:0', 1): (second, {}, second_log)})
    assert second.stopped
    assert first_log.closed and second_log.closed
    assert 'owned PID 100' in capsys.readouterr().err
