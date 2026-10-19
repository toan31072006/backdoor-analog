"""V2 Windows helper: parsing, mocks and read-only dry-run; never launch a task."""
from __future__ import annotations

import base64
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_a1_defense_runner import plans

HELPER = Path(__file__).resolve().parents[2] / 'remote_tools' / 'Start-A1DefenseTaskV2.ps1'
POWERSHELL = shutil.which('powershell.exe')
pytestmark = pytest.mark.skipif(os.name != 'nt' or not POWERSHELL,
                                reason='Windows PowerShell 5.1 integration')


def _powershell(source):
    encoded = base64.b64encode(source.encode('utf-16-le')).decode('ascii')
    return subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                          capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=90)


def _script(source):
    return source.replace('__HELPER__', str(HELPER).replace("'", "''"))


def test_helper_and_payload_parse_without_executing():
    result = _powershell(_script(r'''
$ErrorActionPreference='Stop'
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
$payload=$ast.Find({param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst] -and $n.Value.StartsWith("`$ErrorActionPreference = 'Stop'") -and $n.Value.Contains('$cfg =')},$true).Value
if(!$payload){throw 'No scheduled payload'}
$tokens=$null; $errors=$null
$null=[System.Management.Automation.Language.Parser]::ParseInput($payload,[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
if($payload -notmatch 'exit \$p.ExitCode'){throw 'Missing exit status propagation'}
if($payload -notmatch 'run_a1_defenses_v2\.py' -or $payload -notmatch '--reuse-outdir'){throw 'Wrong runner'}
if($payload -notmatch 'NewGuid'){throw 'Logs must be unique'}
'PARSE_OK'
'''))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PARSE_OK' in result.stdout


def test_only_same_user_v2_exact_output_workers_and_tasks_are_owned():
    result = _powershell(_script(r'''
$ErrorActionPreference='Stop'
$Outdir='C:\Users\Van\backdooranalog\runs\defenses v2'
$currentSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
foreach($name in @('runnerPattern','outputEscaped','manifestEscaped','bindingPattern')){
    $assignment=$ast.Find({param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and $n.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and $n.Left.VariablePath.UserPath -eq $name},$true)
    . ([scriptblock]::Create($assignment.Extent.Text))
}
foreach($name in @('Test-OwnedTask','Test-OwnedWorker','Get-OwnedWorkers')){
    $fn=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
    . ([scriptblock]::Create($fn.Extent.Text))
}
function Task($output,$runner='run_a1_defenses_v2.py'){
    $cfg=@{output=$output;helper='a1_defense_v2_task'}|ConvertTo-Json -Compress
    $cfg64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
    $body="FromBase64String('$cfg64'); $runner"
    $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
    [pscustomobject]@{Principal=[pscustomobject]@{UserId=$currentSid};Actions=@([pscustomobject]@{Arguments="-EncodedCommand $encoded"})}
}
if(!(Test-OwnedTask (Task $Outdir))){throw 'Own V2 task rejected'}
if(Test-OwnedTask (Task ($Outdir+'_other'))){throw 'Sibling task accepted'}
if(Test-OwnedTask (Task $Outdir 'run_a1_defenses_parallel.py')){throw 'V1 task accepted'}
$other=Task $Outdir; $other.Principal.UserId='S-1-5-18'
if(Test-OwnedTask $other){throw 'Other account task accepted'}
function Invoke-CimMethod { param($InputObject,$MethodName,$ErrorAction)
    if($MethodName -ne 'GetOwnerSid'){throw 'Wrong owner query'}
    [pscustomobject]@{ReturnValue=0;Sid=$(if($InputObject.ProcessId -eq 7){'S-1-5-18'}else{$currentSid})}
}
function Get-CimInstance {
    [pscustomobject]@{Name='python.exe';ProcessId=1;CommandLine=('python run_a1_defenses_v2.py --outdir "'+$Outdir+'"')}
    [pscustomobject]@{Name='python.exe';ProcessId=2;CommandLine=('python run_a1_defenses_v2.py --_cell s42__a1__anp --_manifest "'+$Outdir+'\a1_defenses_v2.resolved.json"')}
    [pscustomobject]@{Name='python.exe';ProcessId=3;CommandLine=('python run_a1_defenses_v2.py --outdir "'+$Outdir+'_other"')}
    [pscustomobject]@{Name='python.exe';ProcessId=4;CommandLine=('python run_a1_defenses.py --outdir "'+$Outdir+'"')}
    [pscustomobject]@{Name='python.exe';ProcessId=5;CommandLine=('python yolo.py --outdir "'+$Outdir+'"')}
    [pscustomobject]@{Name='python.exe';ProcessId=6;CommandLine=('python run_a1_defenses_v2.py --source-outdir "'+$Outdir+'" --outdir elsewhere')}
    [pscustomobject]@{Name='python.exe';ProcessId=7;CommandLine=('python run_a1_defenses_v2.py --outdir "'+$Outdir+'"')}
    [pscustomobject]@{Name='python.exe';ProcessId=8;CommandLine=('python run_a1_defenses_v2.py --_manifest "'+$Outdir+'\nested\a1_defenses_v2.resolved.json"')}
}
$owned=@(Get-OwnedWorkers)
if(($owned.ProcessId -join ',') -ne '1,2'){throw ('Ownership incorrect: '+($owned.ProcessId -join ','))}
'OWNERSHIP_OK'
'''))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OWNERSHIP_OK' in result.stdout


def test_real_dry_run_is_read_only_even_with_stop_existing(plans):
    _, _, root, output = plans
    before = sorted(str(p) for p in root.parent.rglob('*'))
    script = _script(r'''
$ErrorActionPreference='Stop'
function Register-ScheduledTask { throw 'must not register' }
function Start-ScheduledTask { throw 'must not start' }
function Stop-ScheduledTask { throw 'must not stop' }
function Stop-Process { throw 'must not stop process' }
function Get-CimInstance { throw 'must not inspect processes during dry-run' }
& '__HELPER__' -SourceOutdir '__ROOT__' -Outdir '__OUTPUT__' -EnvironmentPrefix '__PREFIX__' -ReuseOutdir '' -DryRun -StopExisting
''')
    for key, value in {'__ROOT__': root, '__OUTPUT__': output,
                       '__PREFIX__': Path(sys.executable).parent}.items():
        script = script.replace(key, str(value).replace("'", "''"))
    result = _powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'DRY RUN: 15/15 evaluations' in result.stdout
    assert '2 slots; workers=0' in result.stdout
    assert not output.exists()
    assert sorted(str(p) for p in root.parent.rglob('*')) == before


def test_invalid_seed_parameters_cannot_stop_or_register_task(plans):
    _, _, root, output = plans
    script = _script(r'''
$ErrorActionPreference='Stop'
function Register-ScheduledTask { throw 'UNSAFE register' }
function Start-ScheduledTask { throw 'UNSAFE start' }
function Stop-ScheduledTask { throw 'UNSAFE stop task' }
function Stop-Process { throw 'UNSAFE stop process' }
function Get-CimInstance { throw 'UNSAFE process inspection' }
try {
    & '__HELPER__' -SourceOutdir '__ROOT__' -Outdir '__OUTPUT__' -EnvironmentPrefix '__PREFIX__' -ReuseOutdir '' -Seeds 42,42 -StopExisting
    throw 'expected invalid seeds'
} catch {
    if($_.Exception.Message -notmatch 'duplicate seeds'){throw}
    'FAILED_SAFELY'
}
''')
    for key, value in {'__ROOT__': root, '__OUTPUT__': output,
                       '__PREFIX__': Path(sys.executable).parent}.items():
        script = script.replace(key, str(value).replace("'", "''"))
    result = _powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'FAILED_SAFELY' in result.stdout
    assert not output.exists()


def test_mock_registration_encodes_default_seeds_and_distinct_task_without_launch(plans):
    _, _, root, output = plans
    script = _script(r'''
$ErrorActionPreference='Stop'
$env:CONDA_EXE='__PYTHON__'
$global:DOSEMockRegistered=0; $global:DOSEMockStarted=0
function Get-CimInstance { }
function Get-ScheduledTask { }
function New-ScheduledTaskAction { param($Execute,$Argument)
    $global:DOSEMockAction=[pscustomobject]@{Execute=$Execute;Arguments=$Argument}
    $global:DOSEMockAction
}
function New-ScheduledTaskPrincipal { param($UserId,$LogonType,$RunLevel)
    if($LogonType -ne 'S4U'){throw 'Wrong persistence account'}
    [pscustomobject]@{UserId=$UserId}
}
function New-ScheduledTaskSettingsSet {
    param($ExecutionTimeLimit,$MultipleInstances,[switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries,[switch]$Hidden)
    if($ExecutionTimeLimit -ne [TimeSpan]::Zero -or $MultipleInstances -ne 'IgnoreNew' -or !$Hidden){throw 'Wrong persistence settings'}
    [pscustomobject]@{}
}
function Register-ScheduledTask { param($TaskName,$Action,$Principal,$Settings,[switch]$Force)
    if($TaskName -ne 'MMFI-A1-Defense-V2'){throw 'Must not replace V1 task'}
    $global:DOSEMockRegistered++
}
function Start-ScheduledTask { param($TaskName)
    if($TaskName -ne 'MMFI-A1-Defense-V2'){throw 'Wrong task'}
    $global:DOSEMockStarted++
}
function Stop-ScheduledTask { throw 'Unexpected stop' }
function Stop-Process { throw 'Unexpected process stop' }
& '__HELPER__' -SourceOutdir '__ROOT__' -Outdir '__OUTPUT__' -EnvironmentPrefix '__PREFIX__' -ReuseOutdir ''
if($global:DOSEMockRegistered -ne 1 -or $global:DOSEMockStarted -ne 1){throw 'Mock registration missing'}
if($global:DOSEMockAction.Arguments -notmatch '-EncodedCommand\s+([A-Za-z0-9+/=]+)'){throw 'No encoded payload'}
$payload=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
if($payload -notmatch "FromBase64String\('([A-Za-z0-9+/=]+)'\)"){throw 'No config'}
$cfg=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Matches[1])) | ConvertFrom-Json
if(($cfg.seeds -join ',') -ne '42,43,44' -or $cfg.jobs -ne 2 -or $cfg.output -ne '__OUTPUT__'){throw 'Incorrect config'}
if($cfg.helper -ne 'a1_defense_v2_task' -or $cfg.reuse){throw 'Incorrect reuse marker'}
'MOCK_REGISTER_OK'
''')
    for key, value in {'__ROOT__': root, '__OUTPUT__': output,
                       '__PREFIX__': Path(sys.executable).parent,
                       '__PYTHON__': Path(sys.executable)}.items():
        script = script.replace(key, str(value).replace("'", "''"))
    result = _powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'MOCK_REGISTER_OK' in result.stdout
    assert not output.exists()


def test_two_jobs_three_seed_defaults_and_no_retry_or_deletion():
    helper = HELPER.read_text(encoding='utf-8')
    assert "[ValidateSet(1, 2)][int]$JobsPerGpu = 2" in helper
    assert '[int[]]$Seeds = @(42, 43, 44)' in helper
    assert "'MMFI-A1-Defense-V2'" in helper
    assert '-LogonType S4U' in helper and '-WindowStyle Hidden' in helper
    assert '-ExecutionTimeLimit ([TimeSpan]::Zero)' in helper
    assert 'RestartCount' not in helper and 'Remove-Item' not in helper
    assert helper.index('if ($DryRun) { return }') < helper.index('Stop-ScheduledTask -InputObject')
