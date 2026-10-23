"""WiPose persistent helper contracts; Python/Task/process calls are mocked.

No Task Scheduler registration, GPU allocation, training, download or extraction.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[2] / 'remote_tools/Start-WiPoseHPELiTask.ps1'
POWERSHELL = shutil.which('powershell.exe')


def ps(source, **values):
    if os.name != 'nt' or not POWERSHELL:
        pytest.skip('Windows PowerShell integration')
    values.update(HELPER=HELPER, PREFIX=Path(sys.executable).parent, PYTHON=sys.executable)
    for name, value in values.items():
        source = source.replace('__' + name + '__', str(value).replace("'", "''"))
    encoded = base64.b64encode(source.encode('utf-16-le')).decode('ascii')
    return subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)


MOCK = r'''
$ErrorActionPreference='Stop'
$env:CONDA_EXE='__PYTHON__'
$global:WCalls=@(); $global:WFailFlag=''; $global:WRegistered=0; $global:WStarted=0
$mockPython=Join-Path '__PREFIX__' 'python.exe'
Set-Item -LiteralPath "Function:global:$mockPython" -Value {
    param([Parameter(ValueFromRemainingArguments=$true)][object[]]$RunnerArgs)
    $global:WCalls+=,@($RunnerArgs)
    $global:LASTEXITCODE=0
    if($global:WFailFlag -and $RunnerArgs -contains $global:WFailFlag){$global:LASTEXITCODE=3}
}
function Stop-ScheduledTask {throw 'UNSAFE stop task'}
function Stop-Process {throw 'UNSAFE stop process'}
function Test-Path {param($LiteralPath,$PathType)
    if($LiteralPath -like '*\ATKBackd\run_wipose_hpeli.py'){return $true}
    Microsoft.PowerShell.Management\Test-Path @PSBoundParameters
}
'''

NO_TASKS = r'''
function Register-ScheduledTask {throw 'UNSAFE register task'}
function Start-ScheduledTask {throw 'UNSAFE start task'}
function Get-ScheduledTask {throw 'UNSAFE inspect task'}
function Get-CimInstance {throw 'UNSAFE inspect process'}
'''

REGISTER = r'''
function Get-CimInstance {}
function Get-ScheduledTask {}
function New-ScheduledTaskAction {param($Execute,$Argument)
    $global:WAction=[pscustomobject]@{Execute=$Execute;Arguments=$Argument}; $global:WAction
}
function New-ScheduledTaskPrincipal {param($UserId,$LogonType,$RunLevel)
    if($LogonType -ne 'S4U' -or $RunLevel -ne 'Limited'){throw 'wrong persistence'}
    [pscustomobject]@{UserId=$UserId}
}
function New-ScheduledTaskSettingsSet {param($ExecutionTimeLimit,$MultipleInstances,[switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries,[switch]$Hidden)
    if($ExecutionTimeLimit -ne [TimeSpan]::Zero -or $MultipleInstances -ne 'IgnoreNew' -or !$Hidden){throw 'wrong settings'}
    [pscustomobject]@{}
}
function Register-ScheduledTask {param($TaskName,$Action,$Principal,$Settings,[switch]$Force)
    $global:WTaskName=$TaskName; $global:WRegistered++
}
function Start-ScheduledTask {param($TaskName) $global:WStarted++}
'''


def ok(result, marker):
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker in result.stdout, result.stdout + result.stderr


def test_static_scope_has_no_stops_deletes_download_or_extraction():
    raw = HELPER.read_text(encoding='utf-8')
    for forbidden in ('Stop-Process', 'Stop-ScheduledTask', 'Remove-Item', 'Suspend-Process',
                      'Invoke-WebRequest', 'Expand-Archive', '--jobs-per-gpu'):
        assert forbidden not in raw
    assert "'hpeli\\seed42\\clean\\run.lock'" in raw
    assert "'hpeli\\seed42\\a1\\run.lock'" in raw
    assert raw.count('Assert-WiPoseIdle\n') == 2


def test_helper_and_nested_encoded_payload_parse():
    result = ps(r'''
$ErrorActionPreference='Stop'
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
$payload=$ast.Find({param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst] -and $n.Value.StartsWith("`$ErrorActionPreference = 'Stop'") -and $n.Value.Contains('$cfg =')},$true).Value
if(!$payload){throw 'No payload'}
$null=[System.Management.Automation.Language.Parser]::ParseInput($payload,[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
foreach($required in @('run_wipose_hpeli\.py','--device cuda:0','--data-home','--dataset-root','--outdir','--action-npy','--clean-only','exit \$p.ExitCode','-WindowStyle Hidden','NewGuid')){
    if($payload -notmatch $required){throw ('Missing '+$required)}
}
'PARSE_OK'
''')
    ok(result, 'PARSE_OK')


def test_real_runner_dryrun_has_no_data_task_or_output_access(tmp_path):
    result = ps(NO_TASKS + r'''
$ErrorActionPreference='Stop'
& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__' -DryRun
'REAL_DRYRUN_OK'
''', DATA=tmp_path / 'missing data', OUT=tmp_path / 'new output')
    ok(result, 'REAL_DRYRUN_OK')
    assert 'DRY RUN:' in result.stdout
    assert 'NO data/model/CUDA/output writes/training' in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mode,count', [('DryRun', 1), ('PreflightOnly', 2)])
def test_readonly_modes_do_not_inspect_tasks_or_write_output(tmp_path, mode, count):
    result = ps(MOCK + NO_TASKS + r'''
& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__' -__MODE__
if($global:WCalls.Count -ne __COUNT__ -or $global:WCalls[0] -notcontains '--dry-run'){throw 'wrong calls'}
if('__MODE__' -eq 'PreflightOnly' -and $global:WCalls[1] -notcontains '--preflight'){throw 'no preflight'}
'READONLY_OK'
''', DATA=tmp_path / 'data', OUT=tmp_path / 'output', MODE=mode, COUNT=count)
    ok(result, 'READONLY_OK')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mode', ['', '-CleanOnly'])
def test_registration_binds_exact_config_and_serial_clean_a1(tmp_path, mode):
    result = ps(MOCK + REGISTER + r'''
& '__HELPER__' -DataHome '__DATA__' -DatasetRoot '__DATASET__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__' -ActionNpy '__ACTION__' __MODE__
if($global:WRegistered -ne 1 -or $global:WStarted -ne 1){throw 'no registration'}
if($global:WCalls.Count -ne 2 -or $global:WCalls[0] -notcontains '--dry-run' -or $global:WCalls[1] -notcontains '--preflight'){throw 'preflight missing/out of order'}
if($global:WAction.Arguments -notmatch '-EncodedCommand\s+([A-Za-z0-9+/=]+)'){throw 'no payload'}
$payload=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
if($payload -notmatch "FromBase64String\('([A-Za-z0-9+/=]+)'\)"){throw 'no config'}
$cfg=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Matches[1]))|ConvertFrom-Json
if($cfg.output -ne '__OUT__' -or $cfg.dataset -ne '__DATASET__' -or $cfg.data -ne '__DATA__' -or $cfg.prefix -ne '__PREFIX__' -or $cfg.helper -ne 'wipose_hpeli_task' -or $cfg.action -ne '__ACTION__'){throw 'wrong config'}
$expectedClean='__MODE__' -eq '-CleanOnly'
if($cfg.clean_only -ne $expectedClean){throw 'wrong clean selection'}
if($global:WTaskName -ne $(if($expectedClean){'WiPose-HPELi-Clean-Seed42'}else{'WiPose-HPELi-Seed42'})){throw 'wrong task name'}
foreach($call in $global:WCalls){
    if(($call -contains '--clean-only') -ne $expectedClean){throw 'mode not forwarded'}
    if($call[([Array]::IndexOf($call,'--action-npy')+1)] -ne '__ACTION__'){throw 'action not forwarded'}
}
'REGISTER_OK'
''', DATA=tmp_path / 'data home', DATASET=tmp_path / 'processed data',
        OUT=tmp_path / 'output', ACTION=tmp_path / 'action prior.npy', MODE=mode)
    ok(result, 'REGISTER_OK')
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('failure', ['--dry-run', '--preflight'])
def test_preflight_failure_prevents_registration(tmp_path, failure):
    result = ps(MOCK + REGISTER + r'''
$global:WFailFlag='__FAILURE__'
try {
    & '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__'
    throw 'expected failure'
} catch {if($_.Exception.Message -notmatch 'dry-run failed|preflight failed'){throw}}
if($global:WRegistered -or $global:WStarted){throw 'unsafe registration'}
'FAILURE_OK'
''', DATA=tmp_path / 'data', OUT=tmp_path / 'output', FAILURE=failure)
    ok(result, 'FAILURE_OK')
    assert not (tmp_path / 'output').exists()


def test_task_and_worker_ownership_reject_other_output_account_and_modified_payload():
    result = ps(r'''
$ErrorActionPreference='Stop'
$Outdir='C:\Users\Van\backdooranalog\runs\wipose pilot'; $code='C:\Users\Van\code'
$runner=$code+'\ATKBackd\run_wipose_hpeli.py'; $DataHome='C:\Users\Van\backdooranalog'
$DatasetRoot=$DataHome+'\datasets\WiPose_processed_v1'; $EnvironmentPrefix=$DataHome+'\envs\dose-backdoor'
$ActionNpy=$DataHome+'\actions\data_bend.npy'; $CleanOnly=$false; $env:CONDA_EXE='C:\Conda\conda.exe'
$currentSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
foreach($name in @('runnerPattern','outputEscaped','manifestEscaped','bindingPattern')){
    $n=$ast.Find({param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and $n.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and $n.Left.VariablePath.UserPath -eq $name},$true)
    . ([scriptblock]::Create($n.Extent.Text))
}
foreach($name in @('Get-WiPosePayload','Test-OwnedTask','Test-OwnedWorker')){
    $n=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
    . ([scriptblock]::Create($n.Extent.Text))
}
function Task($output,$clean=$false){
    $cfg=@{helper='wipose_hpeli_task';output=$output;code=$code;data=$DataHome;dataset=$DatasetRoot;prefix=$EnvironmentPrefix;action=$ActionNpy;conda=$env:CONDA_EXE;clean_only=$clean}|ConvertTo-Json -Compress
    $body=Get-WiPosePayload ([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg)))
    [pscustomobject]@{Principal=[pscustomobject]@{UserId=$currentSid};Actions=@([pscustomobject]@{Execute="$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe";Arguments=('-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body)))})}
}
if(!(Test-OwnedTask (Task $Outdir))){throw 'own task rejected'}
if(Test-OwnedTask (Task ($Outdir+'_other'))){throw 'other output accepted'}
if(Test-OwnedTask (Task $Outdir $true)){throw 'different selection accepted for overwrite'}
if(!(Test-OwnedTask (Task $Outdir $true) -OutputOnly)){throw 'same-output CleanOnly active task missed'}
$other=Task $Outdir; $other.Principal.UserId='S-1-5-18'
if(Test-OwnedTask $other){throw 'other account accepted'}
$other=Task $Outdir; $other.Actions[0].Execute='cmd.exe'
if(Test-OwnedTask $other){throw 'other executable accepted'}
$other=Task $Outdir
$body=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String(($other.Actions[0].Arguments -split ' ')[-1]))+'; Write-Host injected'
$other.Actions[0].Arguments='-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
if(Test-OwnedTask $other){throw 'modified payload accepted'}
foreach($field in @('ActionNpy','DatasetRoot','EnvironmentPrefix')){
    $original=Get-Variable -Name $field -ValueOnly; $own=Task $Outdir
    Set-Variable -Name $field -Value ($original+'_other')
    if(Test-OwnedTask $own){throw ('mismatched '+$field+' accepted')}
    Set-Variable -Name $field -Value $original
}
function Invoke-CimMethod {param($InputObject,$MethodName,$ErrorAction)
    [pscustomobject]@{ReturnValue=0;Sid=$(if($InputObject.ProcessId -eq 8){'S-1-5-18'}else{$currentSid})}
}
$cases=@(
    [pscustomobject]@{Name='python.exe';ProcessId=1;CommandLine=('python "'+$runner+'" --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=2;CommandLine=('python "'+$runner+'" --_worker a1 --_manifest "'+$Outdir+'\wipose_hpeli.resolved.json"')},
    [pscustomobject]@{Name='python.exe';ProcessId=3;CommandLine=('python C:\foreign\run_wipose_hpeli.py --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=4;CommandLine=('python "'+$runner+'" --outdir "'+$Outdir+'_other"')},
    [pscustomobject]@{Name='python.exe';ProcessId=5;CommandLine=('python run_a1_wiflow.py --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=6;CommandLine=('python "'+$runner+'" --data-home "'+$Outdir+'" --outdir elsewhere')},
    [pscustomobject]@{Name='python.exe';ProcessId=7;CommandLine=('python "'+$runner+'" --_manifest "'+$Outdir+'\nested\wipose_hpeli.resolved.json"')},
    [pscustomobject]@{Name='python.exe';ProcessId=8;CommandLine=('python "'+$runner+'" --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=9;CommandLine=('python "'+$runner.Replace('\','/')+'" --outdir "'+$Outdir.Replace('\','/')+'"')}
)
$owned=@($cases | Where-Object {Test-OwnedWorker $_})
if(($owned.ProcessId -join ',') -ne '1,2,9'){throw ('wrong workers '+($owned.ProcessId -join ','))}
'OWNERSHIP_OK'
''')
    ok(result, 'OWNERSHIP_OK')


@pytest.mark.parametrize('conflict', ['active', 'queued', 'clean_only', 'foreign_output', 'foreign_account'])
def test_running_queued_or_foreign_named_task_prevents_registration(tmp_path, conflict):
    result = ps(MOCK + REGISTER + r'''
$Out='__OUT__'; $WCode=[IO.Path]::GetFullPath((Split-Path -Parent (Split-Path -Parent '__HELPER__')))
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$cfg=@{helper='wipose_hpeli_task';output=$(if('__CONFLICT__' -eq 'foreign_output'){$Out+'_other'}else{$Out});code=$WCode;data='__DATA__';dataset=(Join-Path '__DATA__' 'datasets\WiPose_processed_v1');prefix='__PREFIX__';action=(Join-Path '__DATA__' 'actions\data_bend.npy');conda=$env:CONDA_EXE;clean_only=('__CONFLICT__' -eq 'clean_only')}|ConvertTo-Json -Compress
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
$fn=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Get-WiPosePayload'},$true)
. ([scriptblock]::Create($fn.Extent.Text))
$body=Get-WiPosePayload ([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg)))
$encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
$global:WFakeTask=[pscustomobject]@{State=$(if('__CONFLICT__' -eq 'queued'){'Queued'}else{'Running'});Principal=[pscustomobject]@{UserId=$(if('__CONFLICT__' -eq 'foreign_account'){'S-1-5-18'}else{$sid})};Actions=@([pscustomobject]@{Execute="$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe";Arguments="-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded"})}
function Get-ScheduledTask {param($TaskName,$ErrorAction)
    # A differently named CleanOnly task shares the same output and must block.
    if(!($TaskName -and '__CONFLICT__' -eq 'clean_only')){$global:WFakeTask}
}
try {
    & '__HELPER__' -DataHome '__DATA__' -Outdir $Out -EnvironmentPrefix '__PREFIX__'
    throw 'expected failure'
} catch {if($_.Exception.Message -notmatch 'output is active|different account/action/output/config'){throw}}
if($global:WRegistered -or $global:WStarted -or $global:WCalls.Count -ne 1){throw 'unsafe registration/preflight'}
'TASK_CONFLICT_OK'
''', DATA=tmp_path / 'data', OUT=tmp_path / 'output', CONFLICT=conflict)
    ok(result, 'TASK_CONFLICT_OK')
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('kind', ['root', 'clean', 'a1'])
def test_locked_output_or_cell_rejected_without_deleting_lock(tmp_path, kind):
    output = tmp_path / 'output'
    lock = output / ({'root': 'run.lock', 'clean': 'hpeli/seed42/clean/run.lock',
                      'a1': 'hpeli/seed42/a1/run.lock'}[kind])
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b'LOCK')
    result = ps(MOCK + REGISTER + r'''
$global:WFailFlag='-c'
try {
    & '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__'
    throw 'expected failure'
} catch {if($_.Exception.Message -notmatch 'output lock unavailable'){throw}}
if($global:WRegistered -or $global:WStarted){throw 'unsafe registration'}
if($global:WCalls.Count -ne 2 -or $global:WCalls[1][-1] -ne '__LOCK__'){throw 'wrong lock probe'}
'LOCK_OK'
''', DATA=tmp_path / 'data', OUT=output, LOCK=lock)
    ok(result, 'LOCK_OK')
    assert lock.read_bytes() == b'LOCK'


@pytest.mark.parametrize('late', [False, True])
def test_worker_active_before_or_during_preflight_prevents_registration(tmp_path, late):
    result = ps(MOCK + REGISTER + r'''
$Out='__OUT__'; $WCode=[IO.Path]::GetFullPath((Split-Path -Parent (Split-Path -Parent '__HELPER__')))
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
function Get-CimInstance {
    if($global:WCalls.Count -ge __MIN_CALLS__){
        [pscustomobject]@{Name='python.exe';ProcessId=42;CommandLine=('python "'+$WCode+'\ATKBackd\run_wipose_hpeli.py" --outdir "'+$Out+'"')}
    }
}
function Invoke-CimMethod {param($InputObject,$MethodName,$ErrorAction) [pscustomobject]@{ReturnValue=0;Sid=$sid}}
try {
    & '__HELPER__' -DataHome '__DATA__' -Outdir $Out -EnvironmentPrefix '__PREFIX__'
    throw 'expected failure'
} catch {if($_.Exception.Message -notmatch 'output is active'){throw}}
if($global:WRegistered -or $global:WStarted -or $global:WCalls.Count -ne __MIN_CALLS__){throw 'unsafe registration/preflight'}
'WORKER_OK'
''', DATA=tmp_path / 'data', OUT=tmp_path / 'output', MIN_CALLS=2 if late else 1)
    ok(result, 'WORKER_OK')
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('parameter', ['DataHome', 'DatasetRoot', 'Outdir', 'EnvironmentPrefix', 'ActionNpy'])
def test_quoted_paths_rejected_before_python_and_tasks(tmp_path, parameter):
    result = ps(MOCK + NO_TASKS + r'''
$kw=@{DataHome='__DATA__';EnvironmentPrefix='__PREFIX__';DryRun=$true}
$kw['__PARAMETER__']='C:\invalid"quoted\path'
try {& '__HELPER__' @kw; throw 'expected failure'}
catch {if($_.Exception.Message -notmatch 'without double quotes|Illegal characters'){throw}}
if($global:WCalls.Count){throw 'unexpected Python'}
'PATH_OK'
''', DATA=tmp_path / 'data', PARAMETER=parameter)
    ok(result, 'PATH_OK')


def test_combined_modes_rejected_before_python_and_tasks(tmp_path):
    result = ps(MOCK + NO_TASKS + r'''
try {
    & '__HELPER__' -DataHome '__DATA__' -EnvironmentPrefix '__PREFIX__' -DryRun -PreflightOnly
    throw 'expected failure'
} catch {if($_.Exception.Message -notmatch 'Choose only one'){throw}}
if($global:WCalls.Count){throw 'unexpected Python'}
'MODE_OK'
''', DATA=tmp_path / 'data')
    ok(result, 'MODE_OK')
