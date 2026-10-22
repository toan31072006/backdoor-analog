"""Windows PiW helper contract; task/process calls and Python are mocked.

No actual Task Scheduler registration, CUDA allocation, training or raw-data IO.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[2] / 'remote_tools/Start-PiWHPELiCleanTask.ps1'
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


MOCK_RUNNER = r'''
$ErrorActionPreference='Stop'
$env:CONDA_EXE='__PYTHON__'
$global:PiwCalls=@()
$global:PiwFailFlag=''
$mockPython=Join-Path '__PREFIX__' 'python.exe'
Set-Item -LiteralPath "Function:global:$mockPython" -Value {
    param([Parameter(ValueFromRemainingArguments=$true)][object[]]$RunnerArgs)
    $global:PiwCalls+=,@($RunnerArgs)
    $global:LASTEXITCODE=0
    if($global:PiwFailFlag -and $RunnerArgs -contains $global:PiwFailFlag){$global:LASTEXITCODE=3}
}
function Stop-ScheduledTask {throw 'UNSAFE stop task'}
function Stop-Process {throw 'UNSAFE stop process'}
'''

NO_TASKS = r'''
function Register-ScheduledTask {throw 'UNSAFE register task'}
function Start-ScheduledTask {throw 'UNSAFE start task'}
function Get-ScheduledTask {throw 'UNSAFE inspect task'}
function Get-CimInstance {throw 'UNSAFE inspect process'}
'''

REGISTRATION = r'''
$global:PiwRegistered=0; $global:PiwStarted=0
function Get-CimInstance {}
function Get-ScheduledTask {}
function New-ScheduledTaskAction {param($Execute,$Argument)
    $global:PiwAction=[pscustomobject]@{Execute=$Execute;Arguments=$Argument}; $global:PiwAction
}
function New-ScheduledTaskPrincipal {param($UserId,$LogonType,$RunLevel)
    if($LogonType -ne 'S4U' -or $RunLevel -ne 'Limited'){throw 'wrong account persistence'}
    [pscustomobject]@{UserId=$UserId}
}
function New-ScheduledTaskSettingsSet {param($ExecutionTimeLimit,$MultipleInstances,[switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries,[switch]$Hidden)
    if($ExecutionTimeLimit -ne [TimeSpan]::Zero -or $MultipleInstances -ne 'IgnoreNew' -or !$Hidden){throw 'wrong settings'}
    [pscustomobject]@{}
}
function Register-ScheduledTask {param($TaskName,$Action,$Principal,$Settings,[switch]$Force)
    if($TaskName -ne 'PIW-HPELi-Clean-Seed42'){throw 'wrong task name'}
    $global:PiwRegistered++
}
function Start-ScheduledTask {param($TaskName) $global:PiwStarted++}
'''


def check(result, marker):
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker in result.stdout, result.stdout + result.stderr


def test_helper_and_encoded_payload_parse_clean_only_no_other_jobs():
    result = ps(r'''
$ErrorActionPreference='Stop'
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
$payload=$ast.Find({param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst] -and $n.Value.StartsWith("`$ErrorActionPreference = 'Stop'") -and $n.Value.Contains('$cfg =')},$true).Value
if(!$payload){throw 'No payload'}
$null=[System.Management.Automation.Language.Parser]::ParseInput($payload,[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
foreach($required in @('run_piw_hpeli_clean\.py','--device cuda:0','--data-home','--dataset-root','--outdir','exit \$p.ExitCode','-WindowStyle Hidden','NewGuid')){
    if($payload -notmatch $required){throw ('Missing '+$required)}
}
$raw=Get-Content -LiteralPath '__HELPER__' -Raw
if($raw -match 'Stop-Process|Stop-ScheduledTask|Suspend-Process|Remove-Item|--jobs-per-gpu|--_cell\s|--attack|--a1') {throw 'Unexpected mutation/scope'}
if($raw -notmatch "'hpeli\\seed42\\clean\\run.lock'"){throw 'Missing cell lock probe'}
'PARSE_OK'
''')
    check(result, 'PARSE_OK')


def test_real_runner_dryrun_reads_no_missing_dataset_or_tasks_and_writes_nothing(tmp_path):
    before = list(tmp_path.rglob('*'))
    result = ps(NO_TASKS + r'''
$ErrorActionPreference='Stop'
& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__' -DryRun
'REAL_DRYRUN_OK'
''', DATA=tmp_path / 'missing data', OUT=tmp_path / 'new output')
    check(result, 'REAL_DRYRUN_OK')
    assert 'DRY RUN: 1 clean victim' in result.stdout
    assert list(tmp_path.rglob('*')) == before


@pytest.mark.parametrize('mode,expected', [('DryRun', 1), ('PreflightOnly', 2)])
def test_dryrun_and_preflight_only_do_not_touch_tasks_or_output(tmp_path, mode, expected):
    before = list(tmp_path.rglob('*'))
    result = ps(MOCK_RUNNER + NO_TASKS + r'''
& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__' -__MODE__
if($global:PiwCalls.Count -ne __COUNT__){throw 'wrong calls'}
if($global:PiwCalls[0] -notcontains '--dry-run'){throw 'dry-run not first'}
if('__MODE__' -eq 'PreflightOnly' -and $global:PiwCalls[1] -notcontains '--preflight'){throw 'missing preflight'}
'NO_TASKS_OK'
''', DATA=tmp_path / 'data', OUT=tmp_path / 'new output', MODE=mode, COUNT=expected)
    check(result, 'NO_TASKS_OK')
    assert list(tmp_path.rglob('*')) == before


def test_combined_modes_rejected_before_python_or_tasks(tmp_path):
    result = ps(MOCK_RUNNER + NO_TASKS + r'''
try {
    & '__HELPER__' -DataHome '__DATA__' -EnvironmentPrefix '__PREFIX__' -DryRun -PreflightOnly
    throw 'expected failure'
} catch {if($_.Exception.Message -notmatch 'Choose only one'){throw}}
if($global:PiwCalls.Count){throw 'unexpected Python'}
'MODES_REJECTED'
''', DATA=tmp_path / 'data')
    check(result, 'MODES_REJECTED')


def test_mock_registration_has_preflight_first_exact_config_hidden_one_clean_worker(tmp_path):
    result = ps(MOCK_RUNNER + REGISTRATION + r'''
& '__HELPER__' -DataHome '__DATA__' -DatasetRoot '__DATASET__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__'
if($global:PiwRegistered -ne 1 -or $global:PiwStarted -ne 1){throw 'no registration'}
if($global:PiwCalls.Count -ne 2 -or $global:PiwCalls[0] -notcontains '--dry-run' -or $global:PiwCalls[1] -notcontains '--preflight'){throw 'preflight missing/out of order'}
if($global:PiwAction.Arguments -notmatch '-EncodedCommand\s+([A-Za-z0-9+/=]+)'){throw 'no payload'}
$payload=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
if($payload -notmatch "FromBase64String\('([A-Za-z0-9+/=]+)'\)"){throw 'no config'}
$cfg=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Matches[1]))|ConvertFrom-Json
if($cfg.output -ne '__OUT__' -or $cfg.dataset -ne '__DATASET__' -or $cfg.data -ne '__DATA__' -or $cfg.prefix -ne '__PREFIX__' -or $cfg.helper -ne 'piw_hpeli_clean_task'){throw 'wrong config'}
'REGISTER_OK'
''', DATA=tmp_path / 'data home', DATASET=tmp_path / 'processed data', OUT=tmp_path / 'new output')
    check(result, 'REGISTER_OK')
    assert not (tmp_path / 'new output').exists()


@pytest.mark.parametrize('failure', ['--dry-run', '--preflight'])
def test_preflight_failure_prevents_task_registration(tmp_path, failure):
    result = ps(MOCK_RUNNER + REGISTRATION + r'''
$global:PiwFailFlag='__FAILURE__'
try {
    & '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__'
    throw 'expected failure'
} catch {if($_.Exception.Message -notmatch 'dry-run failed|preflight failed'){throw}}
if($global:PiwRegistered -or $global:PiwStarted){throw 'unsafe task creation'}
'PREFLIGHT_FAILURE_OK'
''', DATA=tmp_path / 'data', OUT=tmp_path / 'new output', FAILURE=failure)
    check(result, 'PREFLIGHT_FAILURE_OK')
    assert not (tmp_path / 'new output').exists()


def test_owner_checks_bind_exact_absolute_runner_output_account_and_task_config():
    result = ps(r'''
$ErrorActionPreference='Stop'
$Outdir='C:\Users\Van\backdooranalog\runs\piw pilot'
$code='C:\Users\Van\code'
$runner=$code+'\ATKBackd\run_piw_hpeli_clean.py'
$DataHome='C:\Users\Van\backdooranalog'
$DatasetRoot=$DataHome+'\datasets\Person-in-WiFi-3D_processed_v1'
$EnvironmentPrefix=$DataHome+'\envs\dose-backdoor'
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
function Task($output,$name='run_piw_hpeli_clean.py'){
    $cfg=@{helper='piw_hpeli_clean_task';output=$output;code=$code;data=$DataHome;dataset=$DatasetRoot;prefix=$EnvironmentPrefix}|ConvertTo-Json -Compress
    $b64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
    $body="FromBase64String('$b64'); $name"
    $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
    [pscustomobject]@{Principal=[pscustomobject]@{UserId=$currentSid};Actions=@([pscustomobject]@{Execute="$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe";Arguments="-EncodedCommand $encoded"})}
}
if(!(Test-OwnedTask (Task $Outdir))){throw 'own task rejected'}
if(Test-OwnedTask (Task ($Outdir+'_other'))){throw 'sibling output accepted'}
if(Test-OwnedTask (Task $Outdir 'run_a1_wiflow.py')){throw 'WiFlow task accepted'}
$other=Task $Outdir; $other.Principal.UserId='S-1-5-18'
if(Test-OwnedTask $other){throw 'other account accepted'}
$other=Task $Outdir; $other.Actions[0].Execute='cmd.exe'
if(Test-OwnedTask $other){throw 'other action accepted'}
function Invoke-CimMethod {param($InputObject,$MethodName,$ErrorAction)
    [pscustomobject]@{ReturnValue=0;Sid=$(if($InputObject.ProcessId -eq 8){'S-1-5-18'}else{$currentSid})}
}
$cases=@(
    [pscustomobject]@{Name='python.exe';ProcessId=1;CommandLine=('python "'+$runner+'" --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=2;CommandLine=('python "'+$runner+'" --_manifest "'+$Outdir+'\piw_hpeli_clean.resolved.json"')},
    [pscustomobject]@{Name='python.exe';ProcessId=3;CommandLine=('python C:\foreign\run_piw_hpeli_clean.py --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=4;CommandLine=('python "'+$runner+'" --outdir "'+$Outdir+'_other"')},
    [pscustomobject]@{Name='python.exe';ProcessId=5;CommandLine=('python run_a1_wiflow.py --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=6;CommandLine=('python "'+$runner+'" --data-home "'+$Outdir+'" --outdir elsewhere')},
    [pscustomobject]@{Name='python.exe';ProcessId=7;CommandLine=('python "'+$runner+'" --_manifest "'+$Outdir+'\nested\piw_hpeli_clean.resolved.json"')},
    [pscustomobject]@{Name='python.exe';ProcessId=8;CommandLine=('python "'+$runner+'" --outdir "'+$Outdir+'"')},
    [pscustomobject]@{Name='python.exe';ProcessId=9;CommandLine=('python "'+$runner.Replace('\','/')+'" --outdir "'+$Outdir.Replace('\','/')+'"')}
)
$owned=@($cases | Where-Object {Test-OwnedWorker $_})
if(($owned.ProcessId -join ',') -ne '1,2,9'){throw ('wrong workers '+($owned.ProcessId -join ','))}
'OWNERSHIP_OK'
''')
    check(result, 'OWNERSHIP_OK')


@pytest.mark.parametrize('conflict', ['active', 'foreign', 'worker'])
def test_active_or_foreign_scope_prevents_registration_before_data_preflight(tmp_path, conflict):
    result = ps(MOCK_RUNNER + REGISTRATION + r'''
$Out='__OUT__'
$PiwCode=[IO.Path]::GetFullPath((Split-Path -Parent (Split-Path -Parent '__HELPER__')))
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$cfg=@{helper='piw_hpeli_clean_task';output=$(if('__CONFLICT__' -eq 'foreign'){$Out+'_foreign'}else{$Out});code=$PiwCode;data='__DATA__';dataset=(Join-Path '__DATA__' 'datasets\Person-in-WiFi-3D_processed_v1');prefix='__PREFIX__'}|ConvertTo-Json -Compress
$b64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
$body="FromBase64String('$b64'); run_piw_hpeli_clean.py"
$encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
$global:PiwFakeTask=[pscustomobject]@{State='Running';Principal=[pscustomobject]@{UserId=$sid};Actions=@([pscustomobject]@{Execute="$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe";Arguments="-EncodedCommand $encoded"})}
function Get-ScheduledTask {param($TaskName,$ErrorAction) if('__CONFLICT__' -ne 'worker'){$global:PiwFakeTask}}
function Get-CimInstance {
    if('__CONFLICT__' -eq 'worker'){
        [pscustomobject]@{Name='python.exe';ProcessId=42;CommandLine=('python "'+$PiwCode+'\ATKBackd\run_piw_hpeli_clean.py" --outdir "'+$Out+'"')}
    }
}
function Invoke-CimMethod {param($InputObject,$MethodName,$ErrorAction) [pscustomobject]@{ReturnValue=0;Sid=$sid}}
try {
    & '__HELPER__' -DataHome '__DATA__' -Outdir $Out -EnvironmentPrefix '__PREFIX__'
    throw 'expected failure'
} catch {if($_.Exception.Message -notmatch 'output is active|different account/action/output/config'){throw}}
if($global:PiwRegistered -or $global:PiwStarted -or $global:PiwCalls.Count -ne 1){throw 'unsafe registration/preflight'}
'CONFLICT_REJECTED'
''', DATA=tmp_path / 'data', OUT=tmp_path / 'new output', CONFLICT=conflict)
    check(result, 'CONFLICT_REJECTED')
    assert not (tmp_path / 'new output').exists()


@pytest.mark.parametrize('cell_lock', [False, True])
def test_locked_root_or_cell_rejected_without_deleting_lock(tmp_path, cell_lock):
    output = tmp_path / 'output'
    lock = output / ('hpeli/seed42/clean/run.lock' if cell_lock else 'run.lock')
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b'LOCK')
    result = ps(MOCK_RUNNER + REGISTRATION + r'''
$global:PiwFailFlag='-c'
try {
    & '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__'
    throw 'expected failure'
} catch {if($_.Exception.Message -notmatch 'output lock unavailable'){throw}}
if($global:PiwRegistered -or $global:PiwStarted){throw 'unsafe registration'}
if($global:PiwCalls.Count -ne 2 -or $global:PiwCalls[1][-1] -ne '__LOCK__'){throw 'wrong lock probe'}
'LOCK_REJECTED'
''', DATA=tmp_path / 'data', OUT=output, LOCK=lock)
    check(result, 'LOCK_REJECTED')
    assert lock.read_bytes() == b'LOCK'
