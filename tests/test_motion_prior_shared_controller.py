"""CPU/offline coordinator policy tests; no scientific victim is launched."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'remote_tools/motion_prior_shared_4090/controller.py'


@pytest.fixture
def controller():
    spec = importlib.util.spec_from_file_location('_test_motion_prior_shared_controller', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def values(tmp_path):
    home = tmp_path / 'home'
    return dict(home=home, a1_repo=tmp_path / 'a1_code', badnets_repo=tmp_path / 'old_badnets_code',
                root=home / 'tools/shared_v1')


@pytest.fixture
def shared_request(controller, values):
    value = dict(schema=1, profile=controller.PROFILE,
        **{key: str(path) for key, path in values.items()}, python=sys.executable,
        a1_out=str(values['home'] / 'runs' / controller.OUTPUTS['a1']),
        badnets_out=str(values['home'] / 'runs' / controller.OUTPUTS['badnets']))
    return controller.signed(value, 'request_sha256')


@pytest.fixture
def files(controller, shared_request):
    root = Path(shared_request['root'])
    request_file, session_file = root / 'request.json', root / 'sessions/current.json'
    controller.write_new(request_file, shared_request)
    controller.write_new(session_file, dict(schema=1, request_sha256=shared_request['request_sha256'], controller={}))
    return request_file, session_file


class FakeProcess:
    serial = 100

    def __init__(self, states=(None, None, 0)):
        type(self).serial += 1
        self.pid = type(self).serial
        self.states = list(states)
        self.returncode = None
        self.poll_count = 0

    def poll(self):
        self.poll_count += 1
        if len(self.states) > 1:
            code = self.states.pop(0)
        else:
            code = self.states[0]
        if isinstance(code, BaseException):
            raise code
        if code is not None:
            self.returncode = code
        return code

    def wait(self):
        while self.poll() is None:
            pass
        return self.returncode

    def terminate(self):
        raise AssertionError('Never terminate an owned peer')

    def kill(self):
        raise AssertionError('Never kill an owned peer')


def owned(role, cell, process):
    return dict(role=role, cell=cell, process=process, stream=io.StringIO(), log='fixture.log')


def empty_audits(controller, completed=()):
    return {role: {cell: dict(role=role, cell=cell, complete=(role, cell) in completed,
                             checkpoint_epoch=49 if (role, cell) in completed else None)
                   for cell in cells} for role, cells in controller.CELLS.items()}


def fake_context(controller, shared_request, files):
    _, session_file = files
    return SimpleNamespace(validate_session=lambda path, req: controller.read(session_file),
                           require_scoped_gpu=lambda req, session: None)


@pytest.mark.parametrize('field', ['request_sha256', 'session_sha256', 'prepared_sha256'])
def test_signed_digest_binds_unsigned_content(controller, field):
    value = controller.signed(dict(a=1), field)
    assert value[field] == controller.digest(dict(a=1))
    assert value[field] != controller.digest(dict(a=2))


def test_paths_require_separate_new_and_frozen_checkouts(controller, values):
    result = controller.paths(SimpleNamespace(**{k: str(v) for k, v in values.items()}))
    assert result == values
    args = dict(values, a1_repo=values['badnets_repo'])
    with pytest.raises(ValueError, match='separate'):
        controller.paths(SimpleNamespace(**{k: str(v) for k, v in args.items()}))


@pytest.mark.parametrize('location', ['tools', 'runs/new', 'outside'])
def test_root_cannot_be_broad_or_native_output(controller, values, location):
    args = dict(values, root=values['home'] / location)
    with pytest.raises(ValueError, match='descendant'):
        controller.paths(SimpleNamespace(**{k: str(v) for k, v in args.items()}))


def test_root_cannot_be_inside_scientific_checkout(controller, values):
    args = dict(values, a1_repo=values['home'] / 'tools')
    with pytest.raises(ValueError, match='separate'):
        controller.paths(SimpleNamespace(**{k: str(v) for k, v in args.items()}))


def test_request_binds_both_plans_inputs_all_four_original_gates(controller, values, monkeypatch, tmp_path):
    for role, cells in controller.CELLS.items():
        output = values['home'] / 'runs' / controller.OUTPUTS[role]
        field = 'prior' if role == 'a1' else 'backbone'
        rows = [dict(**{field: cell}, run_dir=str(output / cell / 'seed42')) for cell in cells]
        controller.write_new(output / controller.PLANS[role],
            dict(data_home=str(values['home']), outdir=str(output), cells=rows))
        controller.write_new(output / 'inputs.json', dict(full=True))
        for cell in rows:
            controller.write_new(Path(cell['run_dir']) / 'gpu_smoke.json', dict(original=True))
    helper = tmp_path / 'helper'
    helper.mkdir()
    (helper / 'context.py').write_text('# fixture', encoding='utf-8')
    monkeypatch.setattr(controller, 'HERE', helper)
    shared_request = controller.make_request(values)
    assert len(shared_request['bindings']) == 8
    assert len(shared_request['source_sha256']) == 2
    assert shared_request['request_sha256'] == controller.digest({k:v for k,v in shared_request.items() if k!='request_sha256'})
    assert not (values['root'] / 'request.json').exists()


def test_write_new_cannot_overwrite_existing_record(controller, tmp_path):
    path = tmp_path / 'record.json'
    controller.write_new(path, dict(a=1))
    with pytest.raises(FileExistsError):
        controller.write_new(path, dict(a=2))
    assert controller.read(path) == dict(a=1)


def test_phase_command_is_strictly_ordered_for_context_identity(controller, shared_request, files):
    request_file, session_file = files
    command = controller.phase_command(shared_request, request_file, session_file, 'a1', 'smoke', 'cross', 'pairdir')
    assert command == [shared_request['python'], '-u', str(controller.HERE/'context.py'),
        '--request', str(request_file), '--session', str(session_file), '--role', 'a1',
        '--phase', 'smoke', '--cell', 'cross', '--pair', 'pairdir']


@pytest.mark.parametrize('role,phase,cell', [('a1','run','wiflow'), ('foreign','run','cross'),
                                          ('badnets','delete','wiflow')])
def test_phase_command_rejects_other_roles_cells_or_actions(controller, shared_request, files, role, phase, cell):
    with pytest.raises(ValueError):
        controller.phase_command(shared_request, *files, role, phase, cell)


def test_scoped_session_uses_validated_record_and_explicit_ephemeral_paths(controller, shared_request, files):
    request_file, session_file = files
    calls = []
    context = SimpleNamespace(validate_session=lambda path, req: calls.append(path) or controller.read(path))
    result = controller.scoped_session(shared_request, request_file, session_file, context)
    assert calls == [session_file]
    assert result['_request_path'] == str(request_file.resolve())
    assert result['_session_path'] == str(session_file.resolve())
    assert '_request_path' not in controller.read(session_file)


def test_session_binds_exact_cim_controller_identity(controller, shared_request, monkeypatch):
    record = dict(ProcessId=os.getpid(), ParentProcessId=7, CreationDate='2026-10-30T00:00:00Z',
                  ExecutablePath=sys.executable, CommandLine='fixture command')
    path = controller.session(shared_request, SimpleNamespace(process_records=lambda: [record]))
    value = controller.read(path)
    assert value['controller'] == record
    assert value['request_sha256'] == shared_request['request_sha256']
    assert value['session_sha256'] == controller.digest({k:v for k,v in value.items() if k!='session_sha256'})


@pytest.mark.parametrize('field', ['CreationDate','ExecutablePath','CommandLine'])
def test_missing_cim_identity_cannot_create_authority(controller, shared_request, field):
    record = dict(ProcessId=os.getpid(), ParentProcessId=7, CreationDate='timestamp',
                  ExecutablePath=sys.executable, CommandLine='command')
    record[field] = None
    with pytest.raises(ValueError, match='lacks'):
        controller.session(shared_request, SimpleNamespace(process_records=lambda: [record]))


def test_audit_nonzero_or_wrong_identity_fails_closed(controller, shared_request, files, monkeypatch):
    monkeypatch.setattr(controller.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=2, stdout='ERROR', stderr='fixture failure'))
    with pytest.raises(RuntimeError, match='audit failed'):
        controller.audit_cell(shared_request, *files, 'a1', 'cross')
    monkeypatch.setattr(controller.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=json.dumps(dict(role='a1',cell='nod',complete=True))))
    with pytest.raises(ValueError, match='role/cell/status'):
        controller.audit_cell(shared_request, *files, 'a1', 'cross')


def test_audit_accepts_only_last_json_line_with_exact_role_cell(controller, shared_request, files, monkeypatch):
    result = dict(role='badnets', cell='wiflow', complete=False, checkpoint_epoch=17)
    monkeypatch.setattr(controller.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout='diagnostic\n'+json.dumps(result)))
    assert controller.audit_cell(shared_request, *files, 'badnets', 'wiflow') == result


def test_drain_never_kills_and_waits_through_poll_error_and_interrupt(controller, monkeypatch):
    monkeypatch.setattr(controller.time, 'sleep', lambda *args: None)
    process = FakeProcess((OSError('transient'), KeyboardInterrupt(), None, 0))
    item = owned('a1', 'cross', process)
    assert controller.drain_owned({'a1': item}) == {'a1': 0}
    assert item['stream'].closed


def test_safe_output_failure_after_spawn_does_not_lose_owned_child(controller, shared_request, files, monkeypatch):
    process = FakeProcess((0,))
    monkeypatch.setattr(controller.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: (_ for _ in ()).throw(BrokenPipeError()))
    item = controller.start_phase(shared_request, *files, 'a1', 'run', 'cross')
    assert item['process'] is process
    controller.close_owned(item)


def test_prepare_rejects_existing_request_before_smoke_or_retry(controller, shared_request, files):
    values = {name: Path(shared_request[name]) for name in ('home','root','a1_repo','badnets_repo')}
    with pytest.raises(FileExistsError, match='never retried'):
        controller.prepare(values, SimpleNamespace())


def test_prepare_refuses_old_native_or_unrelated_gpu_work_before_native_prepare(controller, values, monkeypatch):
    called=[]
    context=SimpleNamespace(require_exclusive_idle=lambda: (_ for _ in ()).throw(ValueError('old queue active')))
    monkeypatch.setattr(controller,'native_a1_prepare',lambda *args:called.append(True))
    with pytest.raises(ValueError,match='old queue active'):
        controller.prepare(values,context)
    assert not called


def test_smoke_failure_drains_peer_and_never_launches_next_pair(controller, shared_request, files, monkeypatch):
    monkeypatch.setattr(controller.time,'sleep',lambda *args:None)
    processes={'a1':FakeProcess((None,None,0)), 'badnets':FakeProcess((9,))}
    items={role:owned(role,'cross' if role=='a1' else 'wiflow',process)
           for role,process in processes.items()}
    calls=[]
    def start(req,rfile,sfile,role,phase,cell,pair=None):
        calls.append((role,cell)); return items[role]
    monkeypatch.setattr(controller,'start_phase',start)
    with pytest.raises(RuntimeError,match='smoke failed'):
        controller.smoke_pair(shared_request,*files,1,dict(a1='cross',badnets='wiflow'))
    assert calls==[('a1','cross'),('badnets','wiflow')]
    assert processes['a1'].returncode==0
    assert all(item['stream'].closed for item in items.values())
    assert len(list((Path(shared_request['root'])/'pairs').glob('*/error')))==1


def test_smoke_sentinel_disk_error_still_drains_owned_peer(controller, shared_request, files, monkeypatch):
    monkeypatch.setattr(controller.time,'sleep',lambda *args:None)
    peer=FakeProcess((None,0))
    calls=[]
    def start(req,rfile,sfile,role,*args):
        calls.append(role)
        if role=='badnets':raise OSError('fixture Popen error')
        return owned(role,'cross',peer)
    monkeypatch.setattr(controller,'start_phase',start)
    monkeypatch.setattr(Path,'touch',lambda *a,**k:(_ for _ in ()).throw(OSError('full disk')))
    with pytest.raises(OSError,match='Popen error'):
        controller.smoke_pair(shared_request,*files,1,dict(a1='cross',badnets='wiflow'))
    assert peer.returncode==0 and calls==['a1','badnets']


def install_scheduler_fixture(controller, shared_request, files, monkeypatch, *, failure=None, unauthenticated=None,
                              completed=(), state_failure=False):
    context=fake_context(controller,shared_request,files)
    monkeypatch.setattr(controller.time,'sleep',lambda *args:None)
    starts=[]; processes={}; finished=set(completed)
    def start(req,rfile,sfile,role,phase,cell,pair=None):
        previous=processes.get(role)
        assert previous is None or previous.returncode is not None, 'Two workers on the same role'
        starts.append((role,phase,cell))
        code=7 if (role,cell)==failure and phase=='run' else 0
        process=FakeProcess((None,None,None,code) if phase=='run' else (0,))
        processes[role]=process
        return owned(role,cell,process)
    def audit(req,rfile,sfile,role,cell):
        if (role,cell)!=unauthenticated:finished.add((role,cell))
        return dict(role=role,cell=cell,complete=(role,cell) in finished,checkpoint_epoch=49)
    monkeypatch.setattr(controller,'start_phase',start)
    monkeypatch.setattr(controller,'audit_cell',audit)
    monkeypatch.setattr(controller,'audit_all',lambda *args:empty_audits(controller,finished))
    if state_failure:
        writes=[]
        original=controller.write_state
        def write(path,value):
            writes.append(True)
            if len(writes)>1:raise OSError('fixture full disk')
            return original(path,value)
        monkeypatch.setattr(controller,'write_state',write)
    return context,starts,processes,empty_audits(controller,completed)


def test_two_role_queues_overlap_but_each_role_remains_serial(controller,shared_request,files,monkeypatch):
    context,starts,processes,audits=install_scheduler_fixture(controller,shared_request,files,monkeypatch)
    result=controller.launch_queues(shared_request,*files,context,audits)
    run=[(r,c) for r,p,c in starts if p=='run']
    assert run[:2]==[('a1','cross'),('badnets','wiflow')]
    assert [c for r,c in run if r=='a1']==['cross','nod']
    assert [c for r,c in run if r=='badnets']==['wiflow','metafiplusplus']
    assert starts[-2:]==[('a1','export','nod'),('badnets','export','metafiplusplus')]
    assert result['status']=='complete'


def test_completed_cells_are_not_retrained(controller,shared_request,files,monkeypatch):
    done={('a1','cross'),('badnets','wiflow')}
    context,starts,processes,audits=install_scheduler_fixture(controller,shared_request,files,monkeypatch,completed=done)
    controller.launch_queues(shared_request,*files,context,audits)
    assert [(r,c) for r,p,c in starts if p=='run']==[('a1','nod'),('badnets','metafiplusplus')]


def test_worker_failure_globally_blocks_all_later_jobs_and_drains_peer(controller,shared_request,files,monkeypatch):
    context,starts,processes,audits=install_scheduler_fixture(controller,shared_request,files,monkeypatch,
                                                            failure=('badnets','wiflow'))
    with pytest.raises(RuntimeError,match='no (retry or )?later job'):
        controller.launch_queues(shared_request,*files,context,audits)
    assert [(r,p,c) for r,p,c in starts]==[('a1','run','cross'),('badnets','run','wiflow')]
    assert all(process.returncode is not None for process in processes.values())
    state=controller.read(next((Path(shared_request['root'])/'executions').glob('*.json')))
    assert state['status']=='failed'


def test_zero_exit_without_authenticated_completion_blocks_other_queues(controller,shared_request,files,monkeypatch):
    context,starts,processes,audits=install_scheduler_fixture(controller,shared_request,files,monkeypatch,
                                                            unauthenticated=('a1','cross'))
    with pytest.raises(ValueError,match='authenticated scientific completion'):
        controller.launch_queues(shared_request,*files,context,audits)
    assert all(cell not in ('nod','metafiplusplus') for role,phase,cell in starts)
    assert all(process.returncode is not None for process in processes.values())


def test_diagnostic_disk_failure_cannot_release_locks_before_owned_child_drained(controller,shared_request,files,monkeypatch):
    context,starts,processes,audits=install_scheduler_fixture(controller,shared_request,files,monkeypatch,state_failure=True)
    with pytest.raises(OSError,match='full disk'):
        controller.launch_queues(shared_request,*files,context,audits)
    assert starts==[('a1','run','cross')]
    assert processes['a1'].returncode==0


def test_status_has_no_context_import_gpu_check_torch_or_mutating_lock(controller,values,monkeypatch,capsys):
    monkeypatch.setattr(controller,'load_context',lambda:(_ for _ in ()).throw(AssertionError('context imported')))
    monkeypatch.setattr(controller,'os_lock',lambda *a:(_ for _ in ()).throw(AssertionError('lock mutated')))
    args=['--mode','status']
    for key,path in values.items():args += ['--'+key.replace('_','-'),str(path)]
    result=controller.main(args)
    assert result['prepared_receipt_present'] is False
    assert not values['root'].exists()
    assert 'does not establish a live worker' in capsys.readouterr().out


def test_status_excludes_powershell_tee_utf16_controller_logs(controller,values,capsys):
    logs=values['root']/'logs';logs.mkdir(parents=True)
    (logs/'controller.prepare.log').write_text('DO_NOT_DECODE_TEE',encoding='utf-16')
    (logs/'run.a1.cross.fixture.log').write_text('epoch 3/49 fixture',encoding='utf-8')
    controller.status(values)
    output=capsys.readouterr().out
    assert 'epoch 3/49 fixture' in output
    assert 'DO_NOT_DECODE_TEE' not in output
    assert 'controller.prepare.log' not in output


def test_nonstatus_mode_requires_explicit_concurrency_opt_in_before_mutation(controller,values,monkeypatch):
    monkeypatch.setattr(controller,'load_context',lambda:(_ for _ in ()).throw(AssertionError('loaded')))
    args=['--mode','prepare']
    for key,path in values.items():args += ['--'+key.replace('_','-'),str(path)]
    with pytest.raises(ValueError,match='allow-concurrent-badnets'):
        controller.main(args)
    assert not values['root'].exists()


def test_native_lock_stack_holds_both_outputs_and_unwinds_on_second_failure(controller,shared_request,monkeypatch):
    events=[]
    @contextmanager
    def lock(path):
        events.append(('enter',path))
        try:yield
        finally:events.append(('exit',path))
    monkeypatch.setattr(controller,'os_lock',lock)
    with controller.native_locks(shared_request):
        assert len(events)==2
    expected=sorted([shared_request['a1_out'],shared_request['badnets_out']])
    assert events==[('enter',expected[0]),('enter',expected[1]),('exit',expected[1]),('exit',expected[0])]


def test_native_prepare_interruption_drains_instead_of_killing(controller,values,monkeypatch):
    process=FakeProcess((None,None,0))
    process.wait=lambda:(_ for _ in ()).throw(KeyboardInterrupt())
    monkeypatch.setattr(controller.subprocess,'Popen',lambda *a,**k:process)
    monkeypatch.setattr(controller.time,'sleep',lambda *args:None)
    with pytest.raises(KeyboardInterrupt):
        controller.native_a1_prepare(values)
    assert process.returncode==0
