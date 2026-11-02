"""Pure process-snapshot tests: never launch, stop or attach to GPU processes."""
from copy import deepcopy
import hashlib
import json
import ntpath
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ATKBackd'))
import adap_patch_shared_gpu as g
import run_mmfi_p1s1_adap_patch_4090 as r


def plan(home=r'C:\Users\Van\backdooranalog',sharing=True):
    return {'home':str(home),'outdir':ntpath.join(str(home),'runs',r.DEFAULT_RUN),
            'execution_policy':{'allow_concurrent_p3s3':sharing}}


def record(p, pid=20, parent=10, role='cell', executable=None, own=False):
    executable=executable or ntpath.join(p['home'],r'envs\dose-backdoor\python.exe')
    script=(ntpath.join(str(Path(g.__file__).parent),g.P1_ENTRY) if own else
            ntpath.join(ntpath.dirname(p['home']),'code_adap_patch_p3s3','ATKBackd',g.P3_ENTRY))
    out=p['outdir'] if own else ntpath.join(p['home'],'runs',g.P3_RUN)
    args=[executable,'-u',script,'--data-home',p['home'],'--outdir',out,
          '--device','cuda:0','--jobs-per-gpu','1','--mode',role]
    if role=='cell':
        args+=['--work','train']
    if own and p['execution_policy']['allow_concurrent_p3s3']:
        args+=['--allow-concurrent-p3s3']
    return dict(ProcessId=pid,ParentProcessId=parent,Name='python.exe',ExecutablePath=executable,
                CommandLine=' '.join('"'+str(a)+'"' for a in args))


def test_explicit_policy_and_exact_single_p3_worker_only():
    p=plan(); worker=record(p)
    assert g.peer_pids([worker],p)=={20}
    report='20, [Insufficient Permissions]\n'
    assert g.sharing_conflicts(report,[worker],p,99)==([],{20})
    p['execution_policy']['allow_concurrent_p3s3']=False
    conflicts,permitted=g.sharing_conflicts(report,[worker],p,99)
    assert permitted==set() and conflicts[0][0]==20


@pytest.mark.parametrize('bad',[
    ('cuda:0','cuda:1'), ('"--jobs-per-gpu" "1"','"--jobs-per-gpu" "2"'),
    ('"--work" "train"','"--work" "smoke"'),
    ('code_adap_patch_p3s3','other_checkout'),
    (g.P3_RUN,'other_run'), (g.P3_ENTRY,'run_mmfi_p3_splits.py'),
    ('"--mode" "cell"','"--mode" "launch"'),
])
def test_lookalike_invocations_not_authorized(bad):
    p=plan(); worker=record(p)
    worker['CommandLine']=worker['CommandLine'].replace(*bad)
    assert g.invocation(worker,p,g.P3_ENTRY,peer=True) is None
    conflicts,permitted=g.sharing_conflicts('20, python.exe',[worker],p,99)
    assert conflicts and not permitted


@pytest.mark.parametrize('field,value',[
    ('ExecutablePath',None),('ExecutablePath',r'C:\other\python.exe'),
    ('Name','pythonw.exe'),('ProcessId',True),('ParentProcessId',None),
])
def test_unverifiable_process_fails_closed(field,value):
    p=plan(); worker=record(p); worker[field]=value
    assert g.invocation(worker,p,g.P3_ENTRY,peer=True) is None


def test_venv_linear_chain_is_one_worker_but_two_roots_or_fork_are_not():
    p=plan(); a=record(p)
    b=record(p,pid=21,parent=20,executable=r'C:\Python312\python.exe')
    assert g.peer_pids([a,b],p)=={20,21}
    with pytest.raises(ValueError,match='More than one'):
        g.peer_pids([a,record(p,pid=22,parent=10)],p)
    with pytest.raises(ValueError,match='Multiple'):
        g.peer_pids([a,b,record(p,pid=22,parent=20)],p)
    with pytest.raises(ValueError,match='approved'):
        g.peer_pids([record(p,executable=r'C:\Python312\python.exe')],p)


def test_controller_ancestors_are_cpu_only_and_no_arbitrary_python_bypass():
    p=plan()
    rows=[record(p,pid=8,parent=1,role='launch'),record(p,pid=10,parent=8,role='launch'),
          record(p,pid=20,parent=10),record(p,pid=21,parent=20,executable=r'C:\Python312\python.exe')]
    assert g.sharing_conflicts('21, python.exe',rows,p,99)==([],{20,21})
    # Controller is not a GPU-worker exemption.
    assert g.sharing_conflicts('10, python.exe\n21, python.exe',rows,p,99)[0][0][0]==10
    rogue=dict(ProcessId=44,ParentProcessId=1,Name='python.exe',ExecutablePath=r'C:\Python312\python.exe',
               CommandLine='"C:\\Python312\\python.exe" -u "C:\\train.py"')
    assert g.sharing_conflicts('21, python.exe\n44, python.exe',rows+[rogue],p,99)[0][0][0]==44
    # A known experiment not yet on CUDA is blocked too.
    assert g.sharing_conflicts('21, python.exe',rows+[rogue],p,99)[0][0][0]==44


def test_own_controller_and_venv_ancestors_are_not_foreign_pending_jobs():
    p=plan(); rows=[record(p,pid=90,parent=1,role='launch',own=True),
        record(p,pid=91,parent=90,role='cell',own=True),record(p,pid=20)]
    assert g.own_ancestors(rows,p,91)=={90,91}
    assert not g.sharing_conflicts('20, python.exe',rows,p,91)[0]
    # A second unrelated P1S1 controller must not be exempted by filename.
    rows.append(record(p,pid=92,parent=1,role='launch',own=True))
    assert g.sharing_conflicts('20, python.exe',rows,p,91)[0][0][0]==92


def test_helper_flag_before_mode_and_worker_flag_at_end_are_both_exact():
    p=plan(); controller=record(p,pid=90,parent=1,role='launch',own=True)
    controller['CommandLine']=controller['CommandLine'].replace(' "--allow-concurrent-p3s3"','')
    controller['CommandLine']=controller['CommandLine'].replace(' "--mode"',
        ' "--allow-concurrent-p3s3" "--mode"')
    assert g.invocation(controller,p,g.P1_ENTRY)['role']=='launch'
    child=record(p,pid=91,parent=90,own=True)
    assert g.sharing_conflicts('',[controller,child],p,91)==([],set())
    controller['CommandLine']+=' "--allow-concurrent-p3s3"'
    assert g.invocation(controller,p,g.P1_ENTRY) is None


def test_unknown_gpu_and_desktop_names_are_distinguished():
    p=plan()
    desktop=dict(ProcessId=5,ParentProcessId=1,Name='AnyDesk.exe')
    assert not g.sharing_conflicts('5, [Insufficient Permissions]',[desktop],p,99)[0]
    assert g.sharing_conflicts('6, [Insufficient Permissions]',[],p,99)[0]==[(6,'[Insufficient Permissions]')]


def test_actual_p3_source_pin_is_still_unchanged():
    path=ROOT/'ATKBackd'/g.P3_ENTRY
    assert hashlib.sha256(path.read_bytes()).hexdigest()==g.P3_RUNNER_SHA256


def test_peer_plan_and_source_are_read_only_verified(tmp_path):
    home=tmp_path/'backdooranalog'; home.mkdir()
    p=plan(home)
    path=home.parent/'code_adap_patch_p3s3'/'ATKBackd'/g.P3_ENTRY
    path.parent.mkdir(parents=True); path.write_bytes((ROOT/'ATKBackd'/g.P3_ENTRY).read_bytes())
    peer_root=home/'runs'/g.P3_RUN; peer_root.mkdir(parents=True)
    saved=dict(profile=g.P3_RUN,home=str(home),outdir=str(peer_root),jobs_per_gpu=1,device='cuda:0',
        cfg={'mmfi_protocol':'protocol3','mmfi_setting':'s3','seed':42,'model':'hpeli'},
        sources={g.P3_ENTRY:g.P3_RUNNER_SHA256})
    saved['plan_sha256']=r.digest(saved)
    blob=json.dumps(saved).encode()
    target=peer_root/'plan.json'; target.write_bytes(blob)
    g.verify_peer_source([], {20},p)
    assert target.read_bytes()==blob
    saved['cfg']['seed']=43; target.write_text(json.dumps(saved),encoding='utf-8')
    with pytest.raises(ValueError,match='mismatch'):g.verify_peer_source([],{20},p)


def test_sharing_is_bound_in_plan_and_forwarded_to_each_child(tmp_path):
    out=tmp_path/'runs'/r.DEFAULT_RUN
    exclusive=r.build_plan(tmp_path,out)
    shared=r.build_plan(tmp_path,out,allow_concurrent_p3s3=True)
    assert exclusive['plan_sha256']!=shared['plan_sha256']
    assert exclusive['cfg']==shared['cfg']
    for work in ('smoke','train','execution'):
        assert '--allow-concurrent-p3s3' in r.command(shared,work)
        assert '--allow-concurrent-p3s3' not in r.command(exclusive,work)
    assert shared['contract']['counts']=={'training':133056,'test':33264}
    assert shared['cfg']['mmfi_protocol']=='protocol1' and shared['cfg']['mmfi_setting']=='s1'
