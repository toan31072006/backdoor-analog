"""PowerShell WiFlow helper checks; no actual task registration or GPU job."""
from __future__ import annotations

import base64
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

HELPER = Path(__file__).resolve().parents[2] / 'remote_tools/Start-A1WiFlowTask.ps1'
POWERSHELL = shutil.which('powershell.exe')


def ps(source, **values):
    if os.name != 'nt' or not POWERSHELL:
        pytest.skip('Windows PowerShell integration')
    values['HELPER'] = HELPER
    for name, value in values.items():
        source = source.replace('__' + name + '__', str(value).replace("'", "''"))
    encoded = base64.b64encode(source.encode('utf-16-le')).decode('ascii')
    return subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)


def test_helper_payload_parses_and_never_stops_other_jobs():
    result = ps(r'''
$ErrorActionPreference='Stop'
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
$payload=$ast.Find({param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst] -and $n.Value.StartsWith("`$ErrorActionPreference = 'Stop'") -and $n.Value.Contains('$cfg =')},$true).Value
if(!$payload){throw 'No payload'}
$null=[System.Management.Automation.Language.Parser]::ParseInput($payload,[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
foreach($required in @('run_a1_wiflow\.py','--device cuda:0','--data-home','--outdir','--clean-only','exit \$p.ExitCode','-WindowStyle Hidden','NewGuid')){
    if($payload -notmatch $required){throw ('Missing '+$required)}
}
if($payload -match 'Stop-Process|Stop-ScheduledTask|run_a1_cross_backbones.py|--jobs-per-gpu|--_cell') {throw 'Unexpected expanded scope'}
'PARSE_OK'
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PARSE_OK' in result.stdout


def test_real_dry_run_does_not_inspect_tasks_data_or_write_output(tmp_path):
    before = list(tmp_path.rglob('*'))
    result = ps(r'''
$ErrorActionPreference='Stop'
function Register-ScheduledTask {throw 'must not register'}
function Start-ScheduledTask {throw 'must not start'}
function Get-ScheduledTask {throw 'must not inspect tasks'}
function Get-CimInstance {throw 'must not inspect processes'}
& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__' -DryRun
''', DATA=tmp_path / 'missing data', OUT=tmp_path / 'new output', PREFIX=Path(sys.executable).parent)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '2/2 real victims' in result.stdout
    assert list(tmp_path.rglob('*')) == before


def test_task_and_worker_ownership_exact_account_runner_output():
    result = ps(r'''
$ErrorActionPreference='Stop'
$Outdir='C:\Users\Van\backdooranalog\runs\wiflow pilot'
$currentSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
foreach($name in @('runnerPattern','outputEscaped','manifestEscaped','bindingPattern')){
    $n=$ast.Find({param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and $n.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and $n.Left.VariablePath.UserPath -eq $name},$true)
    . ([scriptblock]::Create($n.Extent.Text))
}
foreach($name in @('Test-OwnedTask','Test-OwnedWorker')){
    $n=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
    . ([scriptblock]::Create($n.Extent.Text))
}
function Task($output,$runner='run_a1_wiflow.py'){
    $cfg=@{helper='a1_wiflow_task';output=$output}|ConvertTo-Json -Compress
    $b64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
    $body="FromBase64String('$b64'); $runner"
    $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
    [pscustomobject]@{Principal=[pscustomobject]@{UserId=$currentSid};Actions=@([pscustomobject]@{Arguments="-EncodedCommand $encoded"})}
}
if(!(Test-OwnedTask (Task $Outdir))){throw 'own task rejected'}
if(Test-OwnedTask (Task ($Outdir+'_other'))){throw 'sibling accepted'}
if(Test-OwnedTask (Task $Outdir 'run_a1_cross_backbones.py')){throw 'Graph/Meta task accepted'}
$other=Task $Outdir; $other.Principal.UserId='S-1-5-18'
if(Test-OwnedTask $other){throw 'other account accepted'}
function Invoke-CimMethod {param($InputObject,$MethodName,$ErrorAction)
    [pscustomobject]@{ReturnValue=0;Sid=$(if($InputObject.ProcessId -eq 7){'S-1-5-18'}else{$currentSid})}
}
$cases=@(
    [pscustomobject]@{Name='python.exe';ProcessId=1;CommandLine=('python run_a1_wiflow.py --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=2;CommandLine=('python run_a1_wiflow.py --_cell wiflow__s42__a1 --_manifest "'+$Outdir+'\wiflow_a1.resolved.json"')},
    [pscustomobject]@{Name='python.exe';ProcessId=3;CommandLine=('python run_a1_cross_backbones.py --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=4;CommandLine=('python run_a1_wiflow.py --outdir "'+$Outdir+'_other"')},
    [pscustomobject]@{Name='python.exe';ProcessId=5;CommandLine=('python yolo.py --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=6;CommandLine=('python run_a1_wiflow.py --data-home "'+$Outdir+'" --outdir elsewhere')},
    [pscustomobject]@{Name='python.exe';ProcessId=7;CommandLine=('python run_a1_wiflow.py --outdir "'+$Outdir+'"')}
)
$owned=@($cases | Where-Object {Test-OwnedWorker $_})
if(($owned.ProcessId -join ',') -ne '1,2'){throw 'wrong worker ownership'}
'OWNERSHIP_OK'
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OWNERSHIP_OK' in result.stdout


def test_mock_registration_hidden_one_wiflow_slot_no_real_tasks(tmp_path):
    result = ps(r'''
$ErrorActionPreference='Stop'
$env:CONDA_EXE='__PYTHON__'
$global:Registered=0; $global:Started=0
function Get-CimInstance {}
function Get-ScheduledTask {}
function New-ScheduledTaskAction {param($Execute,$Argument)
    $global:Action=[pscustomobject]@{Execute=$Execute;Arguments=$Argument}; $global:Action
}
function New-ScheduledTaskPrincipal {param($UserId,$LogonType,$RunLevel)
    if($LogonType -ne 'S4U'){throw 'wrong account persistence'}
    [pscustomobject]@{UserId=$UserId}
}
function New-ScheduledTaskSettingsSet {param($ExecutionTimeLimit,$MultipleInstances,[switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries,[switch]$Hidden)
    if($ExecutionTimeLimit -ne [TimeSpan]::Zero -or $MultipleInstances -ne 'IgnoreNew' -or !$Hidden){throw 'wrong settings'}
    [pscustomobject]@{}
}
function Register-ScheduledTask {param($TaskName,$Action,$Principal,$Settings,[switch]$Force)
    if($TaskName -ne 'MMFI-A1-WiFlow-Seed42'){throw 'wrong task name'}
    $global:Registered++
}
function Start-ScheduledTask {param($TaskName) $global:Started++}
function Stop-ScheduledTask {throw 'UNSAFE stop task'}
function Stop-Process {throw 'UNSAFE kill process'}
& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__' -CleanOnly
if($global:Registered -ne 1 -or $global:Started -ne 1){throw 'no registration'}
if($global:Action.Arguments -notmatch '-EncodedCommand\s+([A-Za-z0-9+/=]+)'){throw 'no payload'}
$payload=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
if($payload -notmatch "FromBase64String\('([A-Za-z0-9+/=]+)'\)"){throw 'no config'}
$cfg=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Matches[1]))|ConvertFrom-Json
if(!$cfg.cleanOnly -or $cfg.output -ne '__OUT__' -or $cfg.helper -ne 'a1_wiflow_task'){throw 'wrong config'}
'REGISTER_OK'
''', DATA=tmp_path / 'missing data', OUT=tmp_path / 'new output', PREFIX=Path(sys.executable).parent,
        PYTHON=sys.executable)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'REGISTER_OK' in result.stdout
    assert not (tmp_path / 'new output').exists()


@pytest.mark.parametrize('conflict', ['active', 'foreign'])
def test_active_or_foreign_task_prevents_registration(tmp_path, conflict):
    result = ps(r'''
$ErrorActionPreference='Stop'
$env:CONDA_EXE='__PYTHON__'
$Out='__OUT__'
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$cfg=@{helper='a1_wiflow_task';output=$(if('__CONFLICT__' -eq 'foreign'){$Out+'_foreign'}else{$Out})}|ConvertTo-Json -Compress
$b64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
$body="FromBase64String('$b64'); run_a1_wiflow.py"
$encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
$global:FakeTask=[pscustomobject]@{State='Running';Principal=[pscustomobject]@{UserId=$sid};Actions=@([pscustomobject]@{Arguments="-EncodedCommand $encoded"})}
function Get-ScheduledTask {param($TaskName,$ErrorAction) $global:FakeTask}
function Get-CimInstance {}
function Register-ScheduledTask {throw 'UNSAFE registration'}
function Start-ScheduledTask {throw 'UNSAFE start'}
function Stop-Process {throw 'UNSAFE kill'}
try {
  & '__HELPER__' -DataHome '__DATA__' -Outdir $Out -EnvironmentPrefix '__PREFIX__'
  throw 'expected failure'
} catch {
  if($_.Exception.Message -notmatch 'output is active|different account/action/output'){throw}
  'CONFLICT_REJECTED'
}
''', DATA=tmp_path / 'missing', OUT=tmp_path / 'new output', PREFIX=Path(sys.executable).parent,
        PYTHON=sys.executable, CONFLICT=conflict)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CONFLICT_REJECTED' in result.stdout
    assert not (tmp_path / 'new output').exists()
