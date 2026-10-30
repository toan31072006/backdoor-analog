"""Independent shared-execution review: real toy OS workers, never CUDA/CIM."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'remote_tools/motion_prior_shared_4090/controller.py'


def load_controller():
    spec = importlib.util.spec_from_file_location('_independent_shared_controller', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOY = r'''
import json,pathlib,sys,time
path,duration,code=sys.argv[1:]
result=dict(start=time.monotonic(),end=None,code=int(code))
pathlib.Path(path).write_text(json.dumps(result))
time.sleep(float(duration))
result['end']=time.monotonic()
pathlib.Path(path).write_text(json.dumps(result))
sys.exit(int(code))
'''


def fixture_queues(tmp_path, monkeypatch, *, failure=None, unauthenticated=None):
    controller = load_controller()
    execution = tmp_path / 'shared'
    execution.mkdir()
    request = dict(root=str(execution), request_sha256='test-request', python=sys.executable)
    request_file = execution / 'request.json'
    request_file.write_text(json.dumps(request), encoding='utf-8')
    session_file = execution / 'session.json'
    session_file.write_text(json.dumps(dict(controller='toy')), encoding='utf-8')
    owned, calls = [], []

    class Context:
        def validate_session(self, path, request):
            return json.loads(Path(path).read_text(encoding='utf-8'))

        def require_scoped_gpu(self, request, session):
            # A validated session needs ephemeral absolute filenames. Passing
            # raw JSON here used to reject every Launch before the first job.
            assert Path(session['_request_path']) == request_file.resolve()
            assert Path(session['_session_path']) == session_file.resolve()
            running = [item for item in owned if item['process'].poll() is None]
            assert len(running) <= 2
            assert len({item['role'] for item in running}) == len(running)

    def start_phase(request, request_path, session_path, role, phase, cell, pair=None):
        assert phase in ('run', 'export')
        running = [item for item in owned if item['process'].poll() is None]
        assert len(running) < 2
        assert not any(item['role'] == role for item in running)
        duration = .035 if role == 'a1' else .16
        if phase == 'export':
            duration = .01
        code = 7 if failure == (role, cell) and phase == 'run' else 0
        output = execution / f'{role}.{cell}.{phase}.json'
        stream = (execution / f'{role}.{cell}.{phase}.log').open('x', encoding='utf-8')
        process = subprocess.Popen(
            [sys.executable, '-u', '-c', TOY, str(output), str(duration), str(code)],
            stdout=stream, stderr=subprocess.STDOUT,
        )
        item = dict(process=process, stream=stream, role=role, cell=cell,
                    log=str(stream.name), output=output, phase=phase)
        owned.append(item)
        calls.append((phase, role, cell))
        return item

    def audit_cell(request, request_path, session_path, role, cell):
        path = execution / f'{role}.{cell}.run.json'
        complete = False
        if path.exists():
            value = json.loads(path.read_text())
            complete = value['end'] is not None and value['code'] == 0
        if unauthenticated == (role, cell):
            complete = False
        return dict(role=role, cell=cell, complete=complete)

    def audit_all(request, request_path, session_path):
        return {role: {cell: audit_cell(request, request_path, session_path, role, cell)
                       for cell in cells} for role, cells in controller.CELLS.items()}

    monkeypatch.setattr(controller, 'start_phase', start_phase)
    monkeypatch.setattr(controller, 'audit_cell', audit_cell)
    monkeypatch.setattr(controller, 'audit_all', audit_all)
    actual_sleep = time.sleep
    monkeypatch.setattr(controller, 'time', SimpleNamespace(sleep=lambda seconds: actual_sleep(min(seconds, .01))))
    initial = {role: {cell: dict(complete=False) for cell in cells}
               for role, cells in controller.CELLS.items()}
    return controller, request, request_file, session_file, Context(), initial, owned, calls


def cleanup(owned):
    # Test-only recovery: wait, never terminate. A failed assertion must not
    # leave even a tiny toy worker alive in the developer's workspace.
    for item in owned:
        item['process'].wait(timeout=15)
        item['stream'].close()


def test_real_toy_os_pair_serial_roles_max_two_and_native_source_unchanged(tmp_path, monkeypatch):
    protected = [ROOT / 'ATKBackd' / name for name in (
        'run_mmfi_badnets_two_models_4090.py', 'run_mmfi_a1_motion_priors_4090.py',
        'train_backdoor.py', 'a1_wiflow_adapter.py', 'a1_backbone_adapter.py',
    )]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    controller, request, request_file, session_file, context, audits, owned, calls = fixture_queues(tmp_path, monkeypatch)
    try:
        result = controller.launch_queues(request, request_file, session_file, context, audits)
        assert result['status'] == 'complete'
        assert [(role, cell) for phase, role, cell in calls if phase == 'run'] == [
            ('a1', 'cross'), ('badnets', 'wiflow'), ('a1', 'nod'), ('badnets', 'metafiplusplus')]
        intervals = []
        for item in owned:
            if item['phase'] == 'run':
                interval = json.loads(item['output'].read_text())
                intervals.append((interval['start'], 1, item['role']))
                intervals.append((interval['end'], -1, item['role']))
        count, maximum, active_roles = 0, 0, set()
        for timestamp, delta, role in sorted(intervals):
            if delta > 0:
                assert role not in active_roles
                active_roles.add(role)
            else:
                active_roles.remove(role)
            count += delta
            maximum = max(maximum, count)
            assert count <= 2
        assert maximum == 2
        assert count == 0
        assert all(item['process'].poll() == 0 for item in owned)
        assert all(item['stream'].closed for item in owned)
    finally:
        cleanup(owned)
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}


@pytest.mark.parametrize('kind', ['exit', 'unauthenticated'])
def test_real_toy_peer_failure_stops_both_pending_queues_and_drains(tmp_path, monkeypatch, kind):
    controller, request, request_file, session_file, context, audits, owned, calls = fixture_queues(
        tmp_path, monkeypatch, failure=('a1', 'cross') if kind == 'exit' else None,
        unauthenticated=('a1', 'cross') if kind == 'unauthenticated' else None,
    )
    try:
        with pytest.raises((RuntimeError, ValueError)):
            controller.launch_queues(request, request_file, session_file, context, audits)
        assert calls == [('run', 'a1', 'cross'), ('run', 'badnets', 'wiflow')]
        assert all(item['process'].poll() is not None for item in owned)
        assert all(item['stream'].closed for item in owned)
        state = json.loads(next((Path(request['root']) / 'executions').glob('*.json')).read_text())
        assert state['status'] == 'failed'
        assert state['drained_exits']['badnets'] == 0
    finally:
        cleanup(owned)


def test_failure_state_disk_error_cannot_orphan_owned_peer(tmp_path, monkeypatch):
    controller, request, request_file, session_file, context, audits, owned, calls = fixture_queues(
        tmp_path, monkeypatch, failure=('a1', 'cross'))
    original_write = controller.write_state

    def failing_write(path, value):
        if value.get('status') in ('draining_after_failure', 'failed'):
            raise OSError('mock disk full while writing failure diagnostics')
        return original_write(path, value)

    monkeypatch.setattr(controller, 'write_state', failing_write)
    try:
        with pytest.raises((RuntimeError, OSError)):
            controller.launch_queues(request, request_file, session_file, context, audits)
        assert calls == [('run', 'a1', 'cross'), ('run', 'badnets', 'wiflow')]
        assert all(item['process'].poll() is not None for item in owned), 'Diagnostic I/O failure skipped peer drain'
    finally:
        cleanup(owned)


def test_failed_pair_error_sentinel_io_cannot_skip_drain(tmp_path, monkeypatch):
    controller = load_controller()
    root = tmp_path / 'shared'
    root.mkdir()
    request = dict(root=str(root))
    created = []

    class FakeProcess:
        def __init__(self, code):
            self.code = code
            self.polls = 0

        def poll(self):
            self.polls += 1
            return self.code if self.code != 0 or self.polls >= 4 else None

    class FakeStream:
        closed = False

        def close(self):
            self.closed = True

    def start(request, request_file, session_file, role, phase, cell, pair):
        item = dict(process=FakeProcess(7 if role == 'a1' else 0), stream=FakeStream(), role=role, cell=cell)
        created.append(item)
        return item

    touch = Path.touch

    def failing_touch(path, *args, **kwargs):
        if path.name == 'error':
            raise OSError('mock read-only/full pair directory')
        return touch(path, *args, **kwargs)

    monkeypatch.setattr(controller, 'start_phase', start)
    monkeypatch.setattr(Path, 'touch', failing_touch)
    monkeypatch.setattr(controller, 'time', SimpleNamespace(sleep=lambda seconds: None))
    with pytest.raises((RuntimeError, OSError)):
        controller.smoke_pair(request, 'request.json', 'session.json', 1, dict(a1='cross', badnets='wiflow'))
    assert all(item['process'].poll() is not None for item in created), 'Pair diagnostic failure skipped drain'
    assert all(item['stream'].closed for item in created)


def test_post_spawn_broken_stdout_never_loses_child_ownership(tmp_path, monkeypatch):
    controller = load_controller()
    request = dict(root=str(tmp_path), python=sys.executable)

    class FakeProcess:
        pid = 999

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return 0 if self.polls >= 3 else None

    process = FakeProcess()
    monkeypatch.setattr(controller.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(controller, 'time', SimpleNamespace(sleep=lambda seconds: None))

    def broken_print(*args, **kwargs):
        raise BrokenPipeError('mock disconnected output pipe')

    monkeypatch.setattr(controller, 'print', broken_print, raising=False)
    try:
        item = controller.start_phase(request, 'request.json', 'session.json', 'a1', 'run', 'cross')
    except (BrokenPipeError, OSError):
        assert process.poll() is not None, 'Child was created but never returned/owned/drained'
    else:
        assert item['process'] is process
        item['stream'].close()
