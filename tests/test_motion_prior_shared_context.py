"""Pure Windows identity/sidecar receipts and CPU resume fixtures, no GPU claim."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'remote_tools/motion_prior_shared_4090/context.py'


@pytest.fixture
def context():
    spec=importlib.util.spec_from_file_location('motion_shared_context_test',PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def dump(context,path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def bound(tmp_path,context,monkeypatch):
    context_path=tmp_path/'code/remote_tools/motion_prior_shared_4090/context.py'
    context_path.parent.mkdir(parents=True);context_path.write_text('context fixture',encoding='utf-8')
    controller_path=context_path.with_name('controller.py');controller_path.write_text('controller fixture',encoding='utf-8')
    monkeypatch.setattr(context,'__file__',str(context_path))
    home=tmp_path/'home'
    value=dict(schema=1,profile=context.PROFILE,home=str(home),python=sys.executable,
        a1_repo=str(tmp_path/'a1_repo'),badnets_repo=str(tmp_path/'old_badnets_repo'),
        a1_out=str(home/'runs/a1'),badnets_out=str(home/'runs/badnets'),root=str(home/'tools/sharing'))
    sources={str(p):context.sha(p) for p in (context_path,controller_path)}
    value['source_sha256']=sources;value['bindings']={}
    for path in context.expected_bindings(value):
        dump(context,path,{'native_fixture':str(path)})
        value['bindings'][str(path)]=context.sha(path)
    value['request_sha256']=context.digest(value)
    return value


def controller_record(context,bound,pid=100):
    return dict(ProcessId=pid,ParentProcessId=4,Name='python.exe',ExecutablePath=bound['python'],
        CommandLine=subprocess.list2cmdline([bound['python'],'-u',str(Path(context.__file__).with_name('controller.py')),'--mode','Launch']),
        CreationDate='2026-10-30T00:00:00.0000000Z')


def session_files(context,bound):
    root=Path(bound['root']);request_path=root/'request.json'
    dump(context,request_path,bound)
    controller=controller_record(context,bound)
    session=dict(schema=1,request_sha256=bound['request_sha256'],controller=controller)
    session['session_sha256']=context.digest(session)
    session_path=root/'sessions/test_session.json';dump(context,session_path,session)
    session.update(_request_path=str(request_path),_session_path=str(session_path))
    return session,request_path,session_path


def worker(context,bound,session_path,request_path,role,cell,pid,parent=100,phase='run',pair=None):
    args=[bound['python'],'-u',context.__file__,'--request',str(request_path),
        '--session',str(session_path),'--role',role,'--phase',phase,'--cell',cell]
    if pair:args+=['--pair',str(pair)]
    return dict(ProcessId=pid,ParentProcessId=parent,Name='python.exe',ExecutablePath=bound['python'],
        CommandLine=subprocess.list2cmdline(args),CreationDate=f'2026-10-30T00:00:{pid:04d}Z')


def base_records(context,bound,session):
    return [dict(ProcessId=0,ParentProcessId=0,Name='System Idle Process'),
        dict(ProcessId=4,ParentProcessId=0,Name='System'),copy.deepcopy(session['controller'])]


def test_actual_native_sources_are_not_modified(context):
    for role,filename in context.RUNNERS.items():
        assert context.sha(ROOT/'ATKBackd'/filename)==context.RUNNER_SHA[role]


def test_request_all_original_bindings_and_external_sources_checked(context,bound):
    assert context.validate_request(bound)==bound
    assert len(bound['bindings'])==8
    assert len(bound['source_sha256'])==2


@pytest.mark.parametrize('field,value',[('schema',2),('profile','other'),('python','C:/foreign/python.exe'),
    ('request_sha256','forged')])
def test_changed_request_rejected(context,bound,field,value):
    bound[field]=value
    with pytest.raises(ValueError):context.validate_request(bound)


@pytest.mark.parametrize('group',['bindings','source_sha256'])
def test_bound_file_mutation_rejected(context,bound,group):
    path=Path(next(iter(bound[group])));path.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'):context.validate_request(bound)


def test_missing_original_gate_not_silently_reprepared(context,bound):
    path=context.gate_path(bound,'badnets','wiflow');path.unlink()
    with pytest.raises(FileNotFoundError):context.validate_request(bound)


@pytest.mark.parametrize('key,value',[('root','C:/'),('home','relative'),('a1_repo','\\\\host\\share'),
    ('root','C:/Users/Van/backdooranalog/runs/shared')])
def test_unsafe_execution_paths_rejected(context,bound,key,value):
    bound[key]=value;bound.pop('request_sha256');bound['request_sha256']=context.digest(bound)
    with pytest.raises(ValueError):context.validate_request(bound)


def test_session_binds_controller_identity_and_exact_script(context,bound):
    session,_,path=session_files(context,bound)
    assert context.validate_session(path,bound)['controller']==session['controller']
    session.pop('_session_path');session.pop('_request_path')
    session['controller']['CommandLine']='python unrelated.py'
    session.pop('session_sha256');session['session_sha256']=context.digest(session);dump(context,path,session)
    with pytest.raises(ValueError,match='exact bound'):context.validate_session(path,bound)


def test_pid0_terminates_ancestry_without_false_cycle(context):
    rows={0:dict(ProcessId=0,ParentProcessId=0),4:dict(ProcessId=4,ParentProcessId=0),
        100:dict(ProcessId=100,ParentProcessId=4),101:dict(ProcessId=101,ParentProcessId=100)}
    assert context._ancestors(101,rows)=={101,100,4}


def test_exact_two_role_descendants_authorized(context,bound):
    session,rp,sp=session_files(context,bound)
    rows=base_records(context,bound,session)+[
        worker(context,bound,sp,rp,'a1','cross',101),worker(context,bound,sp,rp,'badnets','wiflow',102)]
    permitted,counts=context.scoped_processes(bound,session,rows,rp,sp,101)
    assert {100,101,102}.issubset(permitted)
    assert counts==dict(a1=[101],badnets=[102])


def test_venv_proxy_is_one_role_worker_not_two(context,bound):
    session,rp,sp=session_files(context,bound)
    proxy=worker(context,bound,sp,rp,'a1','cross',101)
    real=copy.deepcopy(proxy);real.update(ProcessId=102,ParentProcessId=101)
    rows=base_records(context,bound,session)+[proxy,real]
    allowed,counts=context.scoped_processes(bound,session,rows,rp,sp,102)
    assert {101,102}.issubset(allowed) and counts['a1']==[101]


@pytest.mark.parametrize('field,value',[('CreationDate','reused'),('ParentProcessId',99),
    ('ExecutablePath','C:/other/python.exe'),('CommandLine','python train.py')])
def test_changed_controller_pid_identity_fails(context,bound,field,value):
    session,rp,sp=session_files(context,bound)
    rows=base_records(context,bound,session)+[worker(context,bound,sp,rp,'a1','cross',101)]
    rows[2][field]=value
    with pytest.raises(ValueError,match='controller exited'):context.scoped_processes(bound,session,rows,rp,sp,101)


def test_controller_exit_is_not_permission_to_share(context,bound):
    session,rp,sp=session_files(context,bound)
    rows=base_records(context,bound,session)[:2]+[worker(context,bound,sp,rp,'a1','cross',101)]
    with pytest.raises(ValueError,match='controller exited'):context.scoped_processes(bound,session,rows,rp,sp,101)


@pytest.mark.parametrize('role,cell',[('a1','cross'),('badnets','wiflow')])
def test_second_independent_same_role_worker_rejected(context,bound,role,cell):
    session,rp,sp=session_files(context,bound)
    rows=base_records(context,bound,session)+[worker(context,bound,sp,rp,role,cell,101),worker(context,bound,sp,rp,role,cell,102)]
    with pytest.raises(ValueError,match='one logical worker'):context.scoped_processes(bound,session,rows,rp,sp,101)


def test_foreign_request_context_not_whitelisted(context,bound):
    session,rp,sp=session_files(context,bound)
    rows=base_records(context,bound,session)+[worker(context,bound,sp,rp,'a1','cross',101),
        worker(context,bound,sp,rp.with_name('foreign.json'),'badnets','wiflow',102)]
    with pytest.raises(ValueError,match='foreign request/session'):context.scoped_processes(bound,session,rows,rp,sp,101)


def test_unknown_old_native_queue_rejected_even_before_cuda(context,bound):
    session,rp,sp=session_files(context,bound)
    rows=base_records(context,bound,session)+[worker(context,bound,sp,rp,'a1','cross',101),
        dict(ProcessId=102,ParentProcessId=4,Name='python.exe',CommandLine='python.exe -u C:/old/ATKBackd/run_mmfi_badnets_two_models_4090.py --mode launch')]
    with pytest.raises(ValueError,match='Native/foreign pending'):context.scoped_processes(bound,session,rows,rp,sp,101)


def test_unowned_matching_worker_is_not_authorized(context,bound):
    session,rp,sp=session_files(context,bound)
    rows=base_records(context,bound,session)+[worker(context,bound,sp,rp,'a1','cross',101,parent=4)]
    with pytest.raises(ValueError,match='not owned'):context.scoped_processes(bound,session,rows,rp,sp,101)


@pytest.mark.parametrize('name,expected',[('AnyDesk.exe',[]),('TeamViewer.exe',[]),('python.exe',[(200,'python.exe')]),
    ('[Insufficient Permissions]',[(200,'[Insufficient Permissions]')])])
def test_wddm_desktop_names_only_not_generic_compute(context,name,expected):
    assert context._gpu_conflicts('200, [Insufficient Permissions]\n',{200:{'Name':name}},set())==expected


def test_exclusive_entry_does_not_allow_old_native_queue(context,monkeypatch):
    monkeypatch.setattr(context.os,'getpid',lambda:100)
    monkeypatch.setattr(context,'process_records',lambda:[dict(ProcessId=0,ParentProcessId=0),
        dict(ProcessId=4,ParentProcessId=0),dict(ProcessId=100,ParentProcessId=4),
        dict(ProcessId=101,ParentProcessId=4,Name='python.exe',CommandLine='python -u C:/old/run_mmfi_badnets_two_models_4090.py')])
    monkeypatch.setattr(context,'_query_gpu',lambda:'')
    with pytest.raises(ValueError,match='no process stopped'):context.require_exclusive_idle()


def test_exclusive_guard_allows_only_self_ancestors_and_desktop(context,monkeypatch):
    monkeypatch.setattr(context.os,'getpid',lambda:100)
    monkeypatch.setattr(context,'process_records',lambda:[dict(ProcessId=0,ParentProcessId=0),
        dict(ProcessId=4,ParentProcessId=0),dict(ProcessId=100,ParentProcessId=4),
        dict(ProcessId=101,ParentProcessId=4,Name='AnyDesk.exe')])
    monkeypatch.setattr(context,'_query_gpu',lambda:'101, [Insufficient Permissions]\n')
    context.require_exclusive_idle()


def test_missing_wiflow_checkpoint_never_restarts_from_zero(context,tmp_path):
    with pytest.raises(ValueError,match='WiFlow checkpoint'):
        context._checkpoint(SimpleNamespace(),dict(run_dir=str(tmp_path),backbone='wiflow',cfg={}))


@pytest.mark.parametrize('filename',['execution_binding.json','three_tables_inputs.json','eval_cache.json','config.resolved.yaml','poison_manifest.json'])
def test_started_cell_without_checkpoint_rejected(context,tmp_path,filename):
    (tmp_path/filename).write_text('fixture',encoding='utf-8')
    with pytest.raises(ValueError,match='lost its checkpoint'):
        context._checkpoint(SimpleNamespace(),dict(run_dir=str(tmp_path),prior='cross',cfg={}))


def test_genuinely_unstarted_metafi_or_a1_can_start_fresh(context,tmp_path):
    assert context._checkpoint(SimpleNamespace(),dict(run_dir=str(tmp_path),backbone='metafiplusplus',cfg={})) is None
    assert context._checkpoint(SimpleNamespace(),dict(run_dir=str(tmp_path),prior='cross',cfg={})) is None


def test_stable_checkpoint_header_is_validated_not_loaded_as_model(context,tmp_path):
    import torch
    path=tmp_path/'checkpoint.pt';torch.save(dict(epoch=7),path)
    calls=[]
    run=SimpleNamespace(validate_checkpoint=lambda blob,cfg:calls.append((blob,cfg)))
    assert context._checkpoint(run,dict(run_dir=str(tmp_path),cfg={'recipe':'frozen'}))==7
    assert calls==[(dict(epoch=7),{'recipe':'frozen'})]


def test_corrupt_checkpoint_raises_without_retry(context,tmp_path):
    (tmp_path/'checkpoint.pt').write_bytes(b'not a checkpoint')
    with pytest.raises(Exception):context._checkpoint(SimpleNamespace(),dict(run_dir=str(tmp_path),cfg={}))


def facts(role,cell):
    return dict(binding={'native':'bound'},execution={'cuda':'frozen'},physical_batch=32,
        optimizer_steps=1 if cell=='wiflow' else 2,real_train_input=True,poison_examples=8,
        parameter_updated=True,losses=[.4,.3])


@pytest.mark.parametrize('field,value',[('physical_batch',16),('real_train_input',False),('parameter_updated',False),
    ('poison_examples',0),('losses',[]),('losses',[True,.3]),('losses',[float('nan'),.3])])
def test_incomplete_shared_smoke_facts_not_accepted(context,field,value):
    value_facts=facts('a1','cross');value_facts[field]=value
    with pytest.raises(ValueError,match='B32'):context._facts(value_facts)


def smoke_fixture(context,bound):
    session,rp,sp=session_files(context,bound)
    # Original native gates are immutable facts prepared under their old context.
    for role,cells in context.ROLES.items():
        for cell in cells:
            path=context.gate_path(bound,role,cell);dump(context,path,facts(role,cell))
            bound['bindings'][str(path)]=context.sha(path)
    bound.pop('request_sha256');bound['request_sha256']=context.digest(bound)
    session,rp,sp=session_files(context,bound)
    for first,second in (('cross','wiflow'),('nod','metafiplusplus')):
        pair=Path(bound['root'])/'pairs'/first
        markers={}
        for role,cell in (('a1',first),('badnets',second)):
            marker=dict(request_sha256=bound['request_sha256'],session_sha256=session['session_sha256'],role=role,cell=cell,native_sha256=context.digest(facts(role,cell)))
            path=pair/(role+'.ready');dump(context,path,marker);markers[role]=context.sha(path)
        for role,cell in (('a1',first),('badnets',second)):
            receipt=dict(schema=1,role=role,cell=cell,request_sha256=bound['request_sha256'],session_sha256=session['session_sha256'],
                session_path=str(sp),original_gate_sha256=context.sha(context.gate_path(bound,role,cell)),
                native=facts(role,cell),paired_residency=True,pair=str(pair),pair_proof=markers)
            receipt['receipt_sha256']=context.digest(receipt)
            dump(context,Path(bound['root'])/'smokes'/f'{role}.{cell}.json',receipt)
    return session


def test_four_shared_receipts_require_exact_pairs_original_gates_and_sessions(context,bound):
    smoke_fixture(context,bound)
    assert set(context.check_smokes(bound))=={'a1.cross','a1.nod','badnets.wiflow','badnets.metafiplusplus'}


@pytest.mark.parametrize('change',['missing_peer_ready','changed_peer_ready','failed_pair','changed_gate','changed_session','changed_receipt','wrong_steps'])
def test_failed_or_tampered_shared_smokes_cannot_launch(context,bound,change):
    session=smoke_fixture(context,bound);root=Path(bound['root']);pair=root/'pairs/cross'
    if change=='missing_peer_ready':(pair/'badnets.ready').unlink()
    if change=='changed_peer_ready':(pair/'badnets.ready').write_text('{}',encoding='utf-8')
    if change=='failed_pair':(pair/'error').write_text('peer failed',encoding='utf-8')
    if change=='changed_gate':context.gate_path(bound,'badnets','wiflow').write_text('{}',encoding='utf-8')
    if change=='changed_session':Path(session['_session_path']).write_text('{}',encoding='utf-8')
    if change=='changed_receipt':(root/'smokes/a1.cross.json').write_text('{}',encoding='utf-8')
    if change=='wrong_steps':
        path=root/'smokes/a1.cross.json';receipt=context._read(path);receipt['native']['optimizer_steps']=1
        receipt.pop('receipt_sha256');receipt['receipt_sha256']=context.digest(receipt);dump(context,path,receipt)
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):context.check_smokes(bound)


def test_barrier_does_not_create_final_smoke_receipt(context,bound,monkeypatch):
    session,rp,sp=session_files(context,bound);pair=Path(bound['root'])/'pairs/test'
    monkeypatch.setattr(context.time,'monotonic',iter([0.,301.]).__next__)
    with pytest.raises(TimeoutError):context._barrier(pair,bound,session,'a1','cross',facts('a1','cross'))
    assert (pair/'a1.ready').exists()
    assert not (Path(bound['root'])/'smokes/a1.cross.json').exists()
