"""Windows cross-backbone helper contract; never register a real scheduled task."""
from __future__ import annotations

import base64
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[2] / 'remote_tools' / 'Start-A1CrossBackboneTask.ps1'
POWERSHELL = shutil.which('powershell.exe')


def _powershell(source):
    if os.name != 'nt' or not POWERSHELL:
        pytest.skip('Windows PowerShell integration')
    encoded = base64.b64encode(source.encode('utf-16-le')).decode('ascii')
    return subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive',
                           '-EncodedCommand', encoded], capture_output=True,
                          text=True, encoding='utf-8', errors='replace', timeout=90)


def _script(source, **values):
    values['HELPER'] = HELPER
    for name, value in values.items():
        source = source.replace('__' + name + '__', str(value).replace("'", "''"))
    return source


def test_helper_and_payload_parse_without_execution():
    result = _powershell(_script(r'''
$ErrorActionPreference='Stop'
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
$payload=$ast.Find({param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst] -and $n.Value.StartsWith("`$ErrorActionPreference = 'Stop'") -and $n.Value.Contains('$cfg =')},$true).Value
if(!$payload){throw 'No task payload'}
$tokens=$null; $errors=$null
$null=[System.Management.Automation.Language.Parser]::ParseInput($payload,[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
foreach($required in @('run_a1_cross_backbones\.py','--data-home','--outdir','--backbones','--num-workers 0','--jobs-per-gpu','--graph-load-batch-size','--pilot-clean','exit \$p.ExitCode','NewGuid','-WindowStyle Hidden')){
    if($payload -notmatch $required){throw ('Missing payload contract: '+$required)}
}
'PARSE_OK'
'''))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PARSE_OK' in result.stdout


def test_owner_binding_is_exact_output_runner_and_current_account():
    result = _powershell(_script(r'''
$ErrorActionPreference='Stop'
$Outdir='C:\Users\Van\backdooranalog\runs\cross model'
$currentSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
foreach($name in @('runnerPattern','outputEscaped','manifestEscaped','bindingPattern')){
    $assignment=$ast.Find({param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and $n.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and $n.Left.VariablePath.UserPath -eq $name},$true)
    . ([scriptblock]::Create($assignment.Extent.Text))
}
foreach($name in @('Test-OwnedTask','Test-OwnedWorker')){
    $fn=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
    . ([scriptblock]::Create($fn.Extent.Text))
}
function Task($output,$runner='run_a1_cross_backbones.py'){
    $cfg=@{output=$output;helper='a1_cross_backbone_task'}|ConvertTo-Json -Compress
    $cfg64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
    $body="FromBase64String('$cfg64'); $runner"
    $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
    [pscustomobject]@{Principal=[pscustomobject]@{UserId=$currentSid};Actions=@([pscustomobject]@{Arguments="-EncodedCommand $encoded"})}
}
if(!(Test-OwnedTask (Task $Outdir))){throw 'Own task rejected'}
if(Test-OwnedTask (Task ($Outdir+'_other'))){throw 'Sibling output accepted'}
if(Test-OwnedTask (Task $Outdir 'run_a1_defenses_v2.py')){throw 'Defense task accepted'}
$other=Task $Outdir; $other.Principal.UserId='S-1-5-18'
if(Test-OwnedTask $other){throw 'Other account accepted'}
function Invoke-CimMethod { param($InputObject,$MethodName,$ErrorAction)
    [pscustomobject]@{ReturnValue=0;Sid=$(if($InputObject.ProcessId -eq 8){'S-1-5-18'}else{$currentSid})}
}
$cases=@(
    [pscustomobject]@{Name='python.exe';ProcessId=1;CommandLine=('python run_a1_cross_backbones.py --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=2;CommandLine=('python run_a1_cross_backbones.py --_cell graphposefi__s42__a1 --_manifest "'+$Outdir+'\cross_backbones.resolved.json"')},
    [pscustomobject]@{Name='python.exe';ProcessId=3;CommandLine=('python run_a1_cross_backbones.py --outdir "'+$Outdir+'_other"')},
    [pscustomobject]@{Name='python.exe';ProcessId=4;CommandLine=('python run_a1_defenses_v2.py --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=5;CommandLine=('python yolo.py --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=6;CommandLine=('python run_a1_cross_backbones.py --data-home "'+$Outdir+'" --outdir elsewhere')},
    [pscustomobject]@{Name='python.exe';ProcessId=7;CommandLine=('python run_a1_cross_backbones.py --_manifest "'+$Outdir+'\nested\cross_backbones.resolved.json"')},
    [pscustomobject]@{Name='python.exe';ProcessId=8;CommandLine=('python run_a1_cross_backbones.py --outdir "'+$Outdir+'"')}
)
$owned=@($cases | Where-Object {Test-OwnedWorker $_})
if(($owned.ProcessId -join ',') -ne '1,2'){throw ('Wrong owners: '+($owned.ProcessId -join ','))}
'OWNERSHIP_OK'
'''))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OWNERSHIP_OK' in result.stdout


def test_real_dry_run_does_not_write_or_inspect_tasks(tmp_path):
    before = sorted(str(p) for p in tmp_path.rglob('*'))
    result = _powershell(_script(r'''
$ErrorActionPreference='Stop'
function Register-ScheduledTask { throw 'must not register' }
function Start-ScheduledTask { throw 'must not start' }
function Stop-ScheduledTask { throw 'must not stop' }
function Stop-Process { throw 'must not kill' }
function Get-ScheduledTask { throw 'must not inspect tasks in dry-run' }
function Get-CimInstance { throw 'must not inspect processes in dry-run' }
& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUTPUT__' -EnvironmentPrefix '__PREFIX__' -PilotClean -DryRun
''', DATA=tmp_path / 'missing data', OUTPUT=tmp_path / 'not created output',
        PREFIX=Path(sys.executable).parent))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'DRY RUN' in result.stdout
    assert sorted(str(p) for p in tmp_path.rglob('*')) == before


def test_mock_registration_saves_pilot_and_user_requested_two_jobs(tmp_path):
    result = _powershell(_script(r'''
$ErrorActionPreference='Stop'
$env:CONDA_EXE='__PYTHON__'
$global:CBRegistered=0; $global:CBStarted=0
function Get-CimInstance { }
function Get-ScheduledTask { }
function New-ScheduledTaskAction { param($Execute,$Argument)
    $global:CBAction=[pscustomobject]@{Execute=$Execute;Arguments=$Argument}
    $global:CBAction
}
function New-ScheduledTaskPrincipal { param($UserId,$LogonType,$RunLevel)
    if($LogonType -ne 'S4U'){throw 'Wrong persistence account'}
    [pscustomobject]@{UserId=$UserId}
}
function New-ScheduledTaskSettingsSet {
    param($ExecutionTimeLimit,$MultipleInstances,[switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries,[switch]$Hidden)
    if($ExecutionTimeLimit -ne [TimeSpan]::Zero -or $MultipleInstances -ne 'IgnoreNew' -or !$Hidden){throw 'Wrong task settings'}
    [pscustomobject]@{}
}
function Register-ScheduledTask { param($TaskName,$Action,$Principal,$Settings,[switch]$Force)
    if($TaskName -ne 'MMFI-A1-CrossBackbones'){throw 'Wrong task name'}
    $global:CBRegistered++
}
function Start-ScheduledTask { param($TaskName) $global:CBStarted++ }
function Stop-ScheduledTask { throw 'Unexpected task stop' }
function Stop-Process { throw 'Unexpected process stop' }
& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUTPUT__' -EnvironmentPrefix '__PREFIX__' -PilotClean -GraphLoadBatchSize 64
if($global:CBRegistered -ne 1 -or $global:CBStarted -ne 1){throw 'Registration missing'}
if($global:CBAction.Arguments -notmatch '-EncodedCommand\s+([A-Za-z0-9+/=]+)'){throw 'No encoded payload'}
$payload=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
if($payload -notmatch "FromBase64String\('([A-Za-z0-9+/=]+)'\)"){throw 'No encoded config'}
$cfg=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Matches[1])) | ConvertFrom-Json
if(($cfg.backbones -join ',') -ne 'metafiplusplus,graphposefi' -or $cfg.jobs -ne 2 -or !$cfg.pilot -or $cfg.graphBatch -ne 64){throw 'Wrong model/job/pilot/batch config'}
if($cfg.helper -ne 'a1_cross_backbone_task' -or $cfg.output -ne '__OUTPUT__'){throw 'Wrong binding marker'}
'MOCK_REGISTER_OK'
''', DATA=tmp_path / 'missing data', OUTPUT=tmp_path / 'output',
        PREFIX=Path(sys.executable).parent, PYTHON=sys.executable))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'MOCK_REGISTER_OK' in result.stdout
    assert not (tmp_path / 'output').exists()


def test_duplicate_backbones_rejected_before_preflight_or_task_changes(tmp_path):
    result = _powershell(_script(r'''
$ErrorActionPreference='Stop'
function Get-CimInstance { throw 'UNSAFE process inspection' }
function Register-ScheduledTask { throw 'UNSAFE registration' }
try {
    & '__HELPER__' -DataHome '__DATA__' -Outdir '__OUTPUT__' -EnvironmentPrefix '__PREFIX__' -Backbones metafiplusplus,metafiplusplus
    throw 'expected duplicate failure'
} catch {
    if($_.Exception.Message -notmatch 'duplicate backbones'){throw}
    'REJECTED_SAFELY'
}
''', DATA=tmp_path / 'data', OUTPUT=tmp_path / 'output',
        PREFIX=Path(sys.executable).parent))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'REJECTED_SAFELY' in result.stdout
    assert not (tmp_path / 'output').exists()


def test_existing_exact_output_worker_blocks_registration_without_killing(tmp_path):
    result = _powershell(_script(r'''
$ErrorActionPreference='Stop'
$env:CONDA_EXE='__PYTHON__'
function Get-ScheduledTask { }
function Get-CimInstance {
    [pscustomobject]@{Name='python.exe';ProcessId=987654321;ParentProcessId=123;CommandLine='python run_a1_cross_backbones.py --outdir "__OUTPUT__"'}
}
function Invoke-CimMethod { param($InputObject,$MethodName,$ErrorAction)
    [pscustomobject]@{ReturnValue=0;Sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value}
}
function Register-ScheduledTask { throw 'UNSAFE registration' }
function Start-ScheduledTask { throw 'UNSAFE start' }
function Stop-ScheduledTask { throw 'UNSAFE stop task' }
function Stop-Process { throw 'UNSAFE stop process' }
try {
    & '__HELPER__' -DataHome '__DATA__' -Outdir '__OUTPUT__' -EnvironmentPrefix '__PREFIX__'
    throw 'expected active-worker failure'
} catch {
    if($_.Exception.Message -notmatch 'still active'){throw}
    'ACTIVE_OWNER_BLOCKED'
}
''', DATA=tmp_path / 'data', OUTPUT=tmp_path / 'output',
        PREFIX=Path(sys.executable).parent, PYTHON=sys.executable))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ACTIVE_OWNER_BLOCKED' in result.stdout
    assert not (tmp_path / 'output').exists()


def test_foreign_task_name_collision_cannot_be_overwritten(tmp_path):
    result = _powershell(_script(r'''
$ErrorActionPreference='Stop'
$env:CONDA_EXE='__PYTHON__'
function Get-ScheduledTask {
    [pscustomobject]@{Principal=[pscustomobject]@{UserId='S-1-5-18'};Actions=@();State='Ready'}
}
function Get-CimInstance { throw 'Must fail before process inspection' }
function Register-ScheduledTask { throw 'UNSAFE foreign task overwrite' }
function Start-ScheduledTask { throw 'UNSAFE start' }
try {
    & '__HELPER__' -DataHome '__DATA__' -Outdir '__OUTPUT__' -EnvironmentPrefix '__PREFIX__'
    throw 'expected task collision'
} catch {
    if($_.Exception.Message -notmatch 'different account/action/output'){throw}
    'FOREIGN_TASK_BLOCKED'
}
''', DATA=tmp_path / 'data', OUTPUT=tmp_path / 'output',
        PREFIX=Path(sys.executable).parent, PYTHON=sys.executable))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'FOREIGN_TASK_BLOCKED' in result.stdout
    assert not (tmp_path / 'output').exists()


def test_no_autokill_retries_deletion_or_batch_override():
    helper = HELPER.read_text(encoding='utf-8')
    assert '[ValidateSet(1, 2)][int]$JobsPerGpu = 2' in helper
    assert '[ValidateSet(32, 64, 128, 256)][int]$GraphLoadBatchSize = 32' in helper
    assert "'MMFI-A1-CrossBackbones'" in helper
    assert 'cross_backbones\\.resolved\\.json' in helper
    assert '-LogonType S4U' in helper and '-WindowStyle Hidden' in helper
    assert '-ExecutionTimeLimit ([TimeSpan]::Zero)' in helper
    assert '--num-workers 0' in helper
    assert '--graph-load-batch-size' in helper
    for forbidden in ('Stop-Process', 'Stop-ScheduledTask', 'Remove-Item',
                      'RestartCount', 'StopExisting', '--batch-size'):
        assert forbidden not in helper
    assert helper.index('if ($DryRun) { return }') < helper.index('Get-ScheduledTask -TaskName')
    assert helper.index('if ($DryRun) { return }') < helper.index('New-Item -ItemType Directory')
