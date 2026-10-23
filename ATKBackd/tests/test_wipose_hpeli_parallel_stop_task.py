"""Operator stop gates with real temporary files and mocked process/task calls."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT/'remote_tools/Stop-WiPoseHPELiParallelTask.ps1'
OLD = ROOT/'remote_tools/Start-WiPoseHPELiParallelTask.ps1'
RUNNER = ROOT/'ATKBackd/run_wipose_hpeli_parallel.py'
POWERSHELL = shutil.which('powershell.exe')


def ps(source, **values):
    if os.name != 'nt' or not POWERSHELL:
        pytest.skip('Windows PowerShell integration')
    values.update(HELPER=HELPER, OLD=OLD, RUNNER=RUNNER, CODE=ROOT,
                  PREFIX=Path(sys.executable).parent, PYTHON=sys.executable)
    for name, value in values.items():
        source = source.replace('__'+name+'__', str(value).replace("'", "''"))
    encoded = base64.b64encode(source.encode('utf-16-le')).decode('ascii')
    return subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                          capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)


def ok(result, marker):
    assert result.returncode == 0, result.stdout+result.stderr
    assert marker in result.stdout, result.stdout+result.stderr


@pytest.fixture
def lineage(tmp_path):
    data=tmp_path/'data'
    output=data/'runs/wipose_hpeli_a1_s42_parallel_v1'
    output.mkdir(parents=True)
    cfg=dict(outdir=str(output), data_home=str(data),
             dataset_root=str(data/'datasets/WiPose_processed_v1'),
             action_npy=str(data/'actions/data_bend.npy'), seed=42)
    manifest=dict(schema=1, profile='wipose_2d_hpeli_clean_a1_seed42_parallel_v1', cfg=cfg,
                  plan_sha256='f'*64,
                  sources={'run_wipose_hpeli_parallel.py':hashlib.sha256(RUNNER.read_bytes()).hexdigest()})
    (output/'wipose_hpeli_parallel.resolved.json').write_text(json.dumps(manifest), encoding='utf-8')
    for stage in ('clean','a1'):
        folder=output/'hpeli/seed42'/stage
        folder.mkdir(parents=True)
        (folder/'checkpoint.pt').write_bytes(f'atomic last complete {stage}'.encode())
        if stage=='a1':
            (folder/'poison.json').write_text('{"seed":42}', encoding='utf-8')
    return dict(DATA=data, OUT=output, BACKUP=tmp_path/'backups')


MOCK = r'''
$ErrorActionPreference='Stop'
$global:WEvents=@(); $global:WProcesses=@(); $global:WTaskStopped=$false
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$cfg=@{helper='wipose_hpeli_parallel_task';output='__OUT__';code='__CODE__';data='__DATA__';dataset=(Join-Path '__DATA__' 'datasets\WiPose_processed_v1');prefix='__PREFIX__';action=(Join-Path '__DATA__' 'actions\data_bend.npy');conda='__PYTHON__';clean_only=$false;jobs_per_gpu=2}|ConvertTo-Json -Compress
$config64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__OLD__',[ref]$tokens,[ref]$errors)
$template=$ast.Find({param($node) $node -is [System.Management.Automation.Language.StringConstantExpressionAst] -and $node.Value.StartsWith("`$ErrorActionPreference = 'Stop'") -and $node.Value.Contains('$cfg =')},$true).Value
$body=$template.Replace('__CONFIG__',$config64)
$global:WFakeTask=[pscustomobject]@{State='Running';Principal=[pscustomobject]@{UserId=$sid};Actions=@([pscustomobject]@{Execute="$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe";Arguments=('-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body)))})}
function Get-ScheduledTask {param($TaskName,$ErrorAction) $global:WFakeTask}
function Get-CimInstance {param($ClassName,$Filter)
    if($Filter -match 'ProcessId = (\d+)'){$global:WProcesses|Where-Object{$_.ProcessId -eq [int]$Matches[1]}}
    else {$global:WProcesses}
}
function Invoke-CimMethod {param($InputObject,$MethodName,$ErrorAction) [pscustomobject]@{ReturnValue=0;Sid=$sid}}
function Stop-ScheduledTask {param($TaskName,$ErrorAction)
    if(@(Get-ChildItem -LiteralPath '__BACKUP__' -Recurse -Filter hashes.json).Count -ne 1){throw 'no verified backup before mutation'}
    $global:WTaskStopped=$true; $global:WEvents+='task'; $global:WFakeTask.State='Ready'
}
function Stop-Process {param($Id,$ErrorAction)
    if(!$global:WTaskStopped){throw 'process stopped before task controller'}
    $global:WEvents+=[string]$Id
    $global:WProcesses=@($global:WProcesses|Where-Object{$_.ProcessId -ne $Id})
}
function Register-ScheduledTask {throw 'UNSAFE register'}
function Start-ScheduledTask {throw 'UNSAFE launch'}
'''

CALL = r"& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__' -BackupRoot '__BACKUP__'"

PROCESSES = r'''
$global:WProcesses=@(
 [pscustomobject]@{Name='python.exe';ProcessId=42420;CreationDate=[datetime]'2026-10-23T10:00:00';ExecutablePath='__PYTHON__';CommandLine='"__PYTHON__" -u "__RUNNER__" --outdir "__OUT__" --dataset-root "__DATA__\datasets\WiPose_processed_v1"'},
 [pscustomobject]@{Name='python.exe';ProcessId=42421;CreationDate=[datetime]'2026-10-23T10:01:00';ExecutablePath='__PYTHON__';CommandLine='"__PYTHON__" -u "__RUNNER__" --outdir "__OUT__" --_worker a1 --_phase train --_manifest "__OUT__\wipose_hpeli_parallel.resolved.json"'},
 [pscustomobject]@{Name='python.exe';ProcessId=42422;CreationDate=[datetime]'2026-10-23T10:02:00';ExecutablePath='__PYTHON__';CommandLine='"__PYTHON__" -u "C:\other\run_a1_wiflow.py" --outdir "__OUT__"'}
)
'''


def test_default_is_readonly_with_no_task_or_process_mutation(lineage):
    before={p:p.read_bytes() for p in Path(lineage['OUT']).rglob('*') if p.is_file()}
    result=ps(MOCK+PROCESSES+r'''
function New-Item {throw 'UNSAFE write in dry-run'}
'''+CALL+r'''
if($global:WEvents.Count){throw 'dry-run mutated state'}
'DRY_RUN_OK'
''', **lineage)
    ok(result,'DRY_RUN_OK')
    assert not Path(lineage['BACKUP']).exists()
    assert {p:p.read_bytes() for p in before}==before


def test_explicit_stop_backs_up_before_task_then_fresh_controller_and_workers(lineage):
    before={p:p.read_bytes() for p in Path(lineage['OUT']).rglob('*') if p.is_file()}
    result=ps(MOCK+PROCESSES+CALL+r''' -Stop
if(($global:WEvents -join ',') -ne 'task,42420,42421'){throw ('wrong stop scope/order '+($global:WEvents -join ','))}
if(($global:WProcesses.ProcessId -join ',') -ne '42422'){throw 'unrelated job changed'}
'STOP_OK'
''', **lineage)
    ok(result,'STOP_OK')
    assert {p:p.read_bytes() for p in before}==before
    hashes=list(Path(lineage['BACKUP']).rglob('hashes.json'))
    assert len(hashes)==2
    for path in hashes:
        for record in json.loads(path.read_text(encoding='utf-8-sig')):
            copied=Path(record['copy']).read_bytes()
            assert len(copied)==record['bytes']
            assert hashlib.sha256(copied).hexdigest()==record['sha256'].lower()
            assert copied==Path(record['source']).read_bytes()


@pytest.mark.parametrize('problem', ['saving','missing_clean','empty_a1','manifest','source'])
def test_invalid_artifacts_refuse_before_backup_or_stop(lineage, problem):
    output=Path(lineage['OUT'])
    if problem=='saving':
        (output/'hpeli/seed42/a1/checkpoint.pt.saving').write_bytes(b'ongoing')
    elif problem=='missing_clean':
        (output/'hpeli/seed42/clean/checkpoint.pt').unlink()
    elif problem=='empty_a1':
        (output/'hpeli/seed42/a1/checkpoint.pt').write_bytes(b'')
    else:
        path=output/'wipose_hpeli_parallel.resolved.json'
        saved=json.loads(path.read_text(encoding='utf-8'))
        if problem=='manifest':
            saved['cfg']['outdir']=str(output)+'_different'
        else:
            saved['sources']['run_wipose_hpeli_parallel.py']='0'*64
        path.write_text(json.dumps(saved),encoding='utf-8')
    result=ps(MOCK+r'''
try {
'''+CALL+r''' -Stop
 throw 'expected refusal'
} catch {if($_.Exception.Message -notmatch 'Atomic .saving|Missing/empty|manifest profile|source hash'){throw}}
if($global:WEvents.Count){throw 'unsafe mutation'}
'ARTIFACT_REFUSAL_OK'
''', **lineage)
    ok(result,'ARTIFACT_REFUSAL_OK')
    assert not Path(lineage['BACKUP']).exists()


@pytest.mark.parametrize('problem', ['owner','output','payload'])
def test_task_provenance_refusal_before_any_mutation(lineage, problem):
    mutate=dict(owner="$global:WFakeTask.Principal.UserId='S-1-5-18'",
                output="$altered=$cfg|ConvertFrom-Json; $altered.output='__OUT___other'; $cfg=$altered|ConvertTo-Json -Compress; $config64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg)); $body=$template.Replace('__CONFIG__',$config64)",
                payload="$body+='# altered'")[problem]
    result=ps(MOCK+mutate+r'''
if('__PROBLEM__' -ne 'owner'){$global:WFakeTask.Actions[0].Arguments='-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))}
try {
'''+CALL+r''' -Stop
 throw 'expected refusal'
} catch {if($_.Exception.Message -notmatch 'another account|mismatch'){throw}}
if($global:WEvents.Count){throw 'unsafe mutation'}
'TASK_REFUSAL_OK'
''', PROBLEM=problem, **lineage)
    ok(result,'TASK_REFUSAL_OK')
    assert not Path(lineage['BACKUP']).exists()


def test_failed_backup_never_stops_anything(lineage):
    result=ps(MOCK+PROCESSES+r'''
function Copy-Item {throw 'simulated backup failure'}
try {
'''+CALL+r''' -Stop
 throw 'expected refusal'
} catch {if($_.Exception.Message -notmatch 'simulated backup failure'){throw}}
if($global:WEvents.Count){throw 'stop before verified backup'}
'BACKUP_REFUSAL_OK'
''', **lineage)
    ok(result,'BACKUP_REFUSAL_OK')


def test_save_started_during_backup_refuses_task_stop(lineage):
    result=ps(MOCK+r'''
function Copy-Item {param($LiteralPath,$Destination,$ErrorAction)
 Microsoft.PowerShell.Management\Copy-Item @PSBoundParameters
 [IO.File]::WriteAllBytes('__OUT__\hpeli\seed42\a1\checkpoint.pt.saving',[byte[]]@(1,2,3))
}
try {
'''+CALL+r''' -Stop
 throw 'expected refusal'
} catch {if($_.Exception.Message -notmatch 'Atomic .saving'){throw}}
if($global:WEvents.Count){throw 'stop during atomic save'}
'SAVING_DURING_BACKUP_OK'
''', **lineage)
    ok(result,'SAVING_DURING_BACKUP_OK')
    assert (Path(lineage['OUT'])/'hpeli/seed42/a1/checkpoint.pt.saving').read_bytes()==b'\x01\x02\x03'


def test_after_stop_backup_captures_epoch_completed_after_before_snapshot(lineage):
    result=ps(MOCK+r'''
function Stop-ScheduledTask {param($TaskName,$ErrorAction)
 if(@(Get-ChildItem -LiteralPath '__BACKUP__' -Recurse -Filter hashes.json).Count -ne 1){throw 'no verified backup before stop'}
 [IO.File]::WriteAllBytes('__OUT__\hpeli\seed42\clean\checkpoint.pt',[Text.Encoding]::UTF8.GetBytes('newly completed clean epoch'))
 $global:WEvents+='task';$global:WTaskStopped=$true;$global:WFakeTask.State='Ready'
}
'''+CALL+r''' -Stop
'LATEST_CHECKPOINT_OK'
''', **lineage)
    ok(result,'LATEST_CHECKPOINT_OK')
    backup=next(Path(lineage['BACKUP']).iterdir())
    rel='output/hpeli/seed42/clean/checkpoint.pt'
    assert (backup/'before_stop'/rel).read_bytes()==b'atomic last complete clean'
    assert (backup/'after_stop'/rel).read_bytes()==b'newly completed clean epoch'
    assert (Path(lineage['OUT'])/'hpeli/seed42/clean/checkpoint.pt').read_bytes()==b'newly completed clean epoch'


def test_pid_reuse_is_rechecked_before_process_stop(lineage):
    result=ps(MOCK+PROCESSES+r'''
function Get-CimInstance {param($ClassName,$Filter)
 if($Filter){$copy=$global:WProcesses[0].PSObject.Copy();$copy.CreationDate=$copy.CreationDate.AddSeconds(1);$copy}
 else {$global:WProcesses}
}
try {
'''+CALL+r''' -Stop
 throw 'expected refusal'
} catch {if($_.Exception.Message -notmatch 'identity changed'){throw}}
if(($global:WEvents -join ',') -ne 'task'){throw 'PID reuse terminated'}
'PID_REUSE_OK'
''', **lineage)
    ok(result,'PID_REUSE_OK')


@pytest.mark.parametrize('problem', ['executable','owner','other_output','other_checkout','non_runner_argument'])
def test_process_scope_excludes_untrusted_or_other_jobs(lineage, problem):
    mutation=dict(executable=r"$global:WProcesses[0].ExecutablePath='C:\other\python.exe'",
                  owner="function Invoke-CimMethod {param($InputObject,$MethodName,$ErrorAction) [pscustomobject]@{ReturnValue=0;Sid='S-1-5-18'}}",
                  other_output="$global:WProcesses[0].CommandLine=$global:WProcesses[0].CommandLine.Replace('__OUT__','__OUT___other')",
                  other_checkout=r"$global:WProcesses[0].CommandLine=$global:WProcesses[0].CommandLine.Replace('__RUNNER__','C:\other\run_wipose_hpeli_parallel.py')",
                  non_runner_argument=r'''$global:WProcesses[0].CommandLine='"__PYTHON__" -u C:\other.py --note "__RUNNER__" --outdir "__OUT__"' ''')[problem]
    result=ps(MOCK+PROCESSES+r'''
$global:WProcesses=@($global:WProcesses[0])
'''+mutation+'\n'+CALL+r''' -Stop
if(($global:WEvents -join ',') -ne 'task' -or $global:WProcesses.Count -ne 1){throw 'unrelated process terminated'}
'PROCESS_SCOPE_OK'
''', **lineage)
    ok(result,'PROCESS_SCOPE_OK')


def test_stop_source_never_deletes_locks_or_uses_historical_saved_pids():
    raw=HELPER.read_text(encoding='utf-8')
    for forbidden in ('Remove-Item','taskkill','Suspend-Process','Wait-Process','controller_pid','runtime_launches','--benchmark'):
        assert forbidden not in raw
    assert "[switch]$Stop" in raw
    assert "if (!$Stop)" in raw
    assert "-Filter '*.saving'" in raw
    assert 'CreationDate -ne $candidate.CreationDate' in raw
