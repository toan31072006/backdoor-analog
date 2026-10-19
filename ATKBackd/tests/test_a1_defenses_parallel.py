"""Runtime scheduling tests; toy CPU artifacts, not full MM-Fi/GPU validation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import weakref

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_a1_defenses as frozen
import run_a1_defenses_parallel as parallel
from test_a1_defense_runner import plans, _write_record


def _args(root, output, *extra):
    return ['--source-outdir', str(root), '--outdir', str(output), *extra]


@pytest.mark.parametrize('jobs', ['0', '3', '-1', 'x'])
def test_reject_unsupported_slot_counts(jobs):
    with pytest.raises(SystemExit):
        parallel.parse_args(['--source-outdir', 's', '--outdir', 'o', '--jobs-per-gpu', jobs])


@pytest.mark.parametrize('device', ['cuda:0,cuda:1', 'cuda:-1', 'cuda', 'cuda:01'])
def test_single_device_only(device):
    with pytest.raises(SystemExit):
        parallel.parse_args(['--source-outdir', 's', '--outdir', 'o', '--device', device])


def test_read_only_modes_and_defaults_preserve_scientific_plan(plans, monkeypatch, capsys):
    _, expected, root, output = plans
    before = (root / frozen.frozen.MANIFEST).read_bytes()
    sources_before = frozen._sources()
    def forbidden(*a, **kw):
        pytest.fail('read-only mode accessed locks/data/checkpoint/GPU/processes')
    for name in ('_cell_lock', '_audit_source', '_bank', '_collect'):
        monkeypatch.setattr(frozen, name, forbidden)
    monkeypatch.setattr(parallel.subprocess, 'Popen', forbidden)
    args = parallel.parse_args(_args(root, output))
    plan, _ = parallel._load_plan(args)
    assert plan['plan_sha256'] == expected['plan_sha256']
    assert plan['sources'] == sources_before
    assert args.jobs_per_gpu == 2 and args.seeds == [42]
    assert plan['recipe']['batch_size'] == 256 and plan['contract']['num_workers'] == 0
    assert parallel.main(_args(root, output, '--dry-run')) == 0
    assert '5 selected evaluations; 2 slots' in capsys.readouterr().out
    assert parallel.main(_args(root, output, '--status')) == 0
    assert 'NOT a fresh scientific audit' in capsys.readouterr().out
    assert not output.exists()
    assert (root / frozen.frozen.MANIFEST).read_bytes() == before


def test_existing_custom_recipe_and_manifest_bytes_survive_sidecar(plans):
    matrix, _, root, output = plans
    plan = frozen.build_plan(matrix, root, output, {'epochs': 2})
    output.mkdir()
    path = output / frozen.MANIFEST
    path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    before = path.read_bytes()
    loaded, _ = parallel._load_plan(parallel.parse_args(_args(root, output)))
    assert loaded == plan
    assert path.read_bytes() == before
    assert 'run_a1_defenses_parallel.py' not in loaded['sources']


def test_changed_source_still_rejected(plans, monkeypatch):
    _, plan, root, output = plans
    output.mkdir()
    (output / frozen.MANIFEST).write_text(json.dumps(plan), encoding='utf-8')
    monkeypatch.setattr(frozen, '_sources', lambda: {'changed': 'bad'})
    with pytest.raises(ValueError, match='NEW output'):
        parallel._load_plan(parallel.parse_args(_args(root, output)))


def test_idle_check_includes_unselected_seed_and_never_creates_missing_locks(plans):
    _, plan, _, _ = plans
    orphan = plan['cells'][-1]  # seed44, although pilot selects seed42
    with frozen._cell_lock(orphan['folder']):
        with pytest.raises(RuntimeError, match='Another process'):
            parallel._check_idle_cells(plan)
    parallel._check_idle_cells(plan)
    assert not Path(plan['cells'][0]['folder']).exists()


def test_prepare_shared_bank_once_per_seed_and_release_dataset(plans, monkeypatch):
    matrix, plan, _, _ = plans
    selected = frozen._selected(plan, [42, 43], ['a1'], list(frozen.DEFENSES))
    audited, banks, references = [], [], []
    class Data:
        pass
    def audit(original):
        audited.append(original['seed'])
        return {}, {}, {'seed': original['seed']}, {}
    def bank(plan, original, current):
        data = Data()
        references.append(weakref.ref(data))
        banks.append((original['seed'], current['seed']))
        return data, {}
    monkeypatch.setattr(frozen, '_audit_source', audit)
    monkeypatch.setattr(frozen, '_bank', bank)
    parallel._prepare_banks(plan, matrix, selected)
    assert audited == [42, 43] and banks == [(42, 42), (43, 43)]
    assert all(ref() is None for ref in references)


def test_verified_record_requires_full_original_binding_not_file_presence(plans, monkeypatch):
    matrix, plan, _, _ = plans
    cell = plan['cells'][0]
    assert parallel._verified_record(plan, matrix, cell) is None
    record = _write_record(plan, cell)
    seen = []
    def collect(chosen, source):
        seen.append(chosen)
        assert source == matrix
        assert chosen['plan_sha256'] == plan['plan_sha256']
        raise ValueError('scientific binding mismatch')
    monkeypatch.setattr(frozen, '_collect', collect)
    with pytest.raises(ValueError, match='binding'):
        parallel._verified_record(plan, matrix, cell)
    assert seen[0]['cells'] == [cell]
    monkeypatch.setattr(frozen, '_collect', lambda p, s: [record])
    assert parallel._verified_record(plan, matrix, cell) == record
    assert len(plan['cells']) == 15  # no mutation to canonical plan


def test_two_slots_spawn_original_worker_append_logs_and_skip_verified(plans, monkeypatch):
    matrix, plan, _, output = plans
    cells = frozen._selected(plan, [42], ['a1'], list(frozen.DEFENSES))
    before = copy.deepcopy(plan)
    completed = {cells[0]['cell_id']}
    calls, active, peak = [], set(), [0]
    for cell in cells:
        folder = Path(cell['folder'])
        folder.mkdir(parents=True)
        (folder / 'console.log').write_text('old checkpoint log\n', encoding='utf-8')
    class Child:
        def __init__(self, name):
            self.name, self.pid = name, 100 + len(calls)
            self.polls = 0
            active.add(name)
            peak[0] = max(peak[0], len(active))
        def poll(self):
            self.polls += 1
            if self.polls == 1:
                return None
            active.discard(self.name)
            completed.add(self.name)
            return 0
    def popen(command, **kwargs):
        name = command[command.index('--_cell') + 1]
        calls.append((command, kwargs))
        kwargs['stdout'].write('resumed toy worker\n')
        kwargs['stdout'].flush()
        return Child(name)
    monkeypatch.setattr(parallel.subprocess, 'Popen', popen)
    monkeypatch.setattr(parallel.time, 'sleep', lambda _: None)
    monkeypatch.setattr(parallel, '_verified_record',
        lambda p, s, c: {'cell_id': c['cell_id']} if c['cell_id'] in completed else None)
    parallel._schedule(plan, matrix, cells, 'cuda:0', 2, output / frozen.MANIFEST)
    assert peak[0] == 2 and len(calls) == 4
    assert completed == {c['cell_id'] for c in cells}
    for command, kwargs in calls:
        assert command[:3] == [sys.executable, '-u', str(Path(frozen.__file__).resolve())]
        assert command[-2:] == ['--device', 'cuda:0']
        assert '--_manifest' in command and '--num-workers' not in command
        assert kwargs['env']['PYTHONUTF8'] == '1'
        assert kwargs['stdout'].closed
    assert plan == before
    assert 'old checkpoint log\nresumed toy worker' in (
        Path(cells[1]['folder']) / 'console.log').read_text(encoding='utf-8')


@pytest.mark.parametrize('fault', ['exit', 'no_record', 'launch', 'corrupt_record'])
def test_fail_closed_no_retry_only_owned_workers_stopped(plans, monkeypatch, fault):
    matrix, plan, _, output = plans
    cells = frozen._selected(plan, [42], ['a1'], list(frozen.DEFENSES))
    children, commands = [], []
    class Child:
        def __init__(self, name, index):
            self.name, self.index, self.pid = name, index, index + 10
            self.terminated = False
        def poll(self):
            if self.terminated:
                return -1
            if fault == 'launch':
                return None
            return (5 if fault == 'exit' else 0) if self.index == 0 else None
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return -1
    def popen(command, **kwargs):
        commands.append(command)
        if fault == 'launch' and len(commands) == 2:
            raise OSError('cannot launch')
        child = Child(command[command.index('--_cell') + 1], len(children))
        children.append(child)
        return child
    checks = {}
    def verified(p, s, c):
        checks[c['cell_id']] = checks.get(c['cell_id'], 0) + 1
        if fault == 'corrupt_record' and checks[c['cell_id']] > 1:
            raise ValueError('bad binding')
        return None
    monkeypatch.setattr(parallel, '_verified_record', verified)
    monkeypatch.setattr(parallel.subprocess, 'Popen', popen)
    monkeypatch.setattr(parallel.time, 'sleep', lambda _: None)
    with pytest.raises((RuntimeError, OSError, ValueError)):
        parallel._schedule(plan, matrix, cells, 'cpu', 2, output / frozen.MANIFEST)
    assert len(commands) == 2  # no third launch after failure; no retry
    survivor = children[-1] if len(children) == 2 else children[0]
    assert survivor.terminated


def test_main_single_owner_prepares_bank_before_schedule_and_exports_once(plans, monkeypatch):
    matrix, plan, root, output = plans
    output.mkdir()
    manifest = output / frozen.MANIFEST
    manifest.write_text(json.dumps(plan), encoding='utf-8')
    before = manifest.read_bytes()
    events = []
    monkeypatch.setattr(parallel, '_prepare_banks', lambda *a: events.append('bank'))
    monkeypatch.setattr(frozen, '_collect', lambda *a: events.append('audit') or [])
    def schedule(*args):
        assert events == ['bank', 'audit']
        events.append('schedule')
        assert args[4] == 2
    monkeypatch.setattr(parallel, '_schedule', schedule)
    monkeypatch.setattr(frozen, 'export_records', lambda *a: events.append('export') or 'T5_PARTIAL')
    with frozen._cell_lock(output):
        with pytest.raises(RuntimeError, match='Another process'):
            parallel.main(_args(root, output))
    assert not events
    assert parallel.main(_args(root, output, '--device', 'cpu')) == 0
    assert events == ['bank', 'audit', 'schedule', 'audit', 'export']
    assert manifest.read_bytes() == before
    launches = list((output / 'runtime_launches').glob('*.json'))
    assert len(launches) == 1
    launch = json.loads(launches[0].read_text(encoding='utf-8'))
    assert launch['jobs_per_gpu'] == 2 and launch['plan_sha256'] == plan['plan_sha256']
    assert launch['scientific_sources'] == plan['sources']


def test_export_only_never_prepares_or_launches(plans, monkeypatch):
    _, _, root, output = plans
    def forbidden(*a, **kw):
        pytest.fail('export-only started repair/prepared a dataset')
    monkeypatch.setattr(parallel, '_prepare_banks', forbidden)
    monkeypatch.setattr(parallel, '_schedule', forbidden)
    monkeypatch.setattr(frozen, '_collect', lambda *a: [])
    monkeypatch.setattr(frozen, 'export_records', lambda *a: 'T5_PARTIAL')
    assert parallel.main(_args(root, output, '--export-only')) == 0
    assert not (output / 'runtime_launches').exists()


def test_real_subprocesses_use_two_slots_without_touching_source(plans, monkeypatch, tmp_path):
    # Actual child processes exercise stdout handles/waiting on Windows and Linux.
    # The toy worker does not import torch or access benchmark/model data.
    matrix, plan, root, output = plans
    cells = frozen._selected(plan, [42], ['a1'], ['none', 'ft', 'fp'])
    output.mkdir()
    manifest = output / frozen.MANIFEST
    manifest.write_text(json.dumps(plan), encoding='utf-8')
    source_before = (root / frozen.frozen.MANIFEST).read_bytes()
    worker = tmp_path / 'toy_worker.py'
    worker.write_text('''import argparse, json, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--_manifest'); p.add_argument('--_cell'); p.add_argument('--device')
a=p.parse_args()
plan=json.loads(Path(a._manifest).read_text(encoding='utf-8'))
cell=next(c for c in plan['cells'] if c['cell_id']==a._cell)
folder=Path(cell['folder'])
begin=time.time_ns()
(folder/'toy.started').touch()
deadline=time.monotonic()+10
while len(list(Path(plan['outdir']).glob('seed42/a1/*/toy.started'))) < 2:
    if time.monotonic()>deadline:
        raise RuntimeError('second toy worker did not start')
    time.sleep(.02)
time.sleep(.2)
(folder/'toy.json').write_text(json.dumps({'begin':begin,'end':time.time_ns()}))
print('toy completed', a._cell, flush=True)
''', encoding='utf-8')
    monkeypatch.setattr(frozen, '__file__', str(worker))
    monkeypatch.setattr(parallel, '_verified_record', lambda p, s, c:
        {'ok': True} if (Path(c['folder']) / 'toy.json').exists() else None)
    parallel._schedule(plan, matrix, cells, 'cpu', 2, manifest)
    intervals = [json.loads((Path(c['folder']) / 'toy.json').read_text()) for c in cells]
    assert intervals[0]['begin'] < intervals[1]['end']
    assert intervals[1]['begin'] < intervals[0]['end']
    assert intervals[2]['begin'] >= min(x['end'] for x in intervals[:2])
    assert (root / frozen.frozen.MANIFEST).read_bytes() == source_before
    assert all('toy completed' in (Path(c['folder']) / 'console.log').read_text()
               for c in cells)
