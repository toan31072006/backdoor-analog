"""Fast WiPose task operator gates; no real Task/GPU/training calls."""
from __future__ import annotations

import base64
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[2] / 'remote_tools/Start-WiPoseHPELiFastTask.ps1'
SERIAL_HELPER = HELPER.with_name('Start-WiPoseHPELiTask.ps1')
POWERSHELL = shutil.which('powershell.exe')


def ps(source, **values):
    if os.name != 'nt' or not POWERSHELL:
        pytest.skip('Windows PowerShell integration')
    values.update(HELPER=HELPER, SERIAL_HELPER=SERIAL_HELPER,
                  PREFIX=Path(sys.executable).parent, PYTHON=sys.executable)
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
    $global:WCalls+=,@($RunnerArgs); $global:LASTEXITCODE=0
    if($global:WFailFlag -and $RunnerArgs -contains $global:WFailFlag){$global:LASTEXITCODE=3}
}
function Stop-ScheduledTask {throw 'UNSAFE stop task'}
function Stop-Process {throw 'UNSAFE stop process'}
function Test-Path {param($LiteralPath,$PathType)
    if($LiteralPath -like '*\ATKBackd\run_wipose_hpeli_fast.py'){return $true}
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

FUNCTIONS = r'''
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
foreach($fn in $ast.FindAll({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst]},$true)){
    . ([scriptblock]::Create($fn.Extent.Text))
}
'''


def ok(result, marker):
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker in result.stdout, result.stdout + result.stderr


def test_static_scope_preserves_historical_outputs_and_limits_jobs():
    raw = HELPER.read_text(encoding='utf-8')
    for forbidden in ('Stop-Process', 'Stop-ScheduledTask', 'Remove-Item', 'Suspend-Process',
                      'Invoke-WebRequest', 'Expand-Archive', 'Set-Content', '--batch-size'):
        assert forbidden not in raw
    assert '[ValidateSet(1, 2)][int]$JobsPerGpu = 2' in raw
    assert "'hpeli\\seed42\\clean\\run.lock'" in raw
    assert "'hpeli\\seed42\\a1\\run.lock'" in raw
    assert raw.count('Assert-WiPoseFastIdle\n') == 2
    assert 'wipose_hpeli_a1_s42_fast_v1' in raw
    assert 'wipose_hpeli_fast.resolved.json' in raw


def test_helper_and_encoded_payload_parse():
    result = ps(r'''
$ErrorActionPreference='Stop'
''' + FUNCTIONS + r'''
$payload=Get-WiPoseFastPayload 'CONFIG'
$null=[System.Management.Automation.Language.Parser]::ParseInput($payload,[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
foreach($required in @('run_wipose_hpeli_fast\.py','--device cuda:0','--jobs-per-gpu','--data-home','--dataset-root','--outdir','--action-npy','--clean-only','exit \$p.ExitCode','-WindowStyle Hidden','NewGuid')){
    if($payload -notmatch $required){throw ('Missing '+$required)}
}
'PARSE_OK'
''')
    ok(result, 'PARSE_OK')


@pytest.mark.parametrize('mode,count', [('DryRun', 1), ('PreflightOnly', 2)])
@pytest.mark.parametrize('jobs', [1, 2])
def test_readonly_modes_forward_jobs_without_tasks_or_writes(tmp_path, mode, count, jobs):
    result = ps(MOCK + NO_TASKS + r'''
& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__' -JobsPerGpu __JOBS__ -__MODE__
if($global:WCalls.Count -ne __COUNT__ -or $global:WCalls[0] -notcontains '--dry-run'){throw 'wrong calls'}
foreach($call in $global:WCalls){
    if($call[([Array]::IndexOf($call,'--jobs-per-gpu')+1)] -ne '__JOBS__'){throw 'jobs not forwarded'}
}
if('__MODE__' -eq 'PreflightOnly' -and $global:WCalls[1] -notcontains '--preflight'){throw 'no preflight'}
'READONLY_OK'
''', DATA=tmp_path/'data', OUT=tmp_path/'out', JOBS=jobs, MODE=mode, COUNT=count)
    ok(result, 'READONLY_OK')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('jobs', [0, 3, 4, -1])
def test_invalid_slot_count_rejected_before_python_or_tasks(tmp_path, jobs):
    result = ps(MOCK + NO_TASKS + r'''
try {
    & '__HELPER__' -DataHome '__DATA__' -EnvironmentPrefix '__PREFIX__' -JobsPerGpu __JOBS__ -DryRun
    throw 'expected failure'
} catch {if($_.FullyQualifiedErrorId -notmatch 'ParameterArgumentValidationError'){throw}}
if($global:WCalls.Count){throw 'unexpected Python'}
'JOBS_OK'
''', DATA=tmp_path/'data', JOBS=jobs)
    ok(result, 'JOBS_OK')


@pytest.mark.parametrize('mode', ['', '-CleanOnly'])
@pytest.mark.parametrize('jobs', [1, 2])
def test_registration_binds_jobs_roots_and_clean_selection(tmp_path, mode, jobs):
    result = ps(MOCK + REGISTER + r'''
& '__HELPER__' -DataHome '__DATA__' -DatasetRoot '__DATASET__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__' -ActionNpy '__ACTION__' -JobsPerGpu __JOBS__ __MODE__
if($global:WRegistered -ne 1 -or $global:WStarted -ne 1){throw 'no registration'}
if($global:WCalls.Count -ne 2 -or $global:WCalls[0] -notcontains '--dry-run' -or $global:WCalls[1] -notcontains '--preflight'){throw 'preflight missing/out of order'}
if($global:WAction.Arguments -notmatch '-EncodedCommand\s+([A-Za-z0-9+/=]+)'){throw 'no payload'}
$payload=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
if($payload -notmatch "FromBase64String\('([A-Za-z0-9+/=]+)'\)"){throw 'no config'}
$cfg=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Matches[1]))|ConvertFrom-Json
if($cfg.resume_from -ne (Join-Path '__DATA__' 'runs\wipose_hpeli_a1_s42_parallel_v1')){throw 'wrong default lineage'}
if($cfg.output -ne '__OUT__' -or $cfg.dataset -ne '__DATASET__' -or $cfg.data -ne '__DATA__' -or $cfg.prefix -ne '__PREFIX__' -or $cfg.helper -ne 'wipose_hpeli_fast_task' -or $cfg.action -ne '__ACTION__' -or $cfg.jobs_per_gpu -ne __JOBS__){throw 'wrong config'}
$expectedClean='__MODE__' -eq '-CleanOnly'
if($cfg.clean_only -ne $expectedClean){throw 'wrong clean selection'}
if($global:WTaskName -ne $(if($expectedClean){'WiPose-HPELi-Fast-Clean-Seed42'}else{'WiPose-HPELi-Fast-Seed42'})){throw 'wrong task name'}
foreach($call in $global:WCalls){
    if(($call -contains '--clean-only') -ne $expectedClean){throw 'mode not forwarded'}
    if($call[([Array]::IndexOf($call,'--jobs-per-gpu')+1)] -ne '__JOBS__'){throw 'jobs not forwarded'}
}
'REGISTER_OK'
''', DATA=tmp_path/'data home', DATASET=tmp_path/'processed data', OUT=tmp_path/'out',
        ACTION=tmp_path/'action prior.npy', MODE=mode, JOBS=jobs)
    ok(result, 'REGISTER_OK')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('failure', ['--dry-run', '--preflight'])
def test_preflight_failure_never_registers(tmp_path, failure):
    result = ps(MOCK + REGISTER + r'''
$global:WFailFlag='__FAILURE__'
try {& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__'; throw 'expected failure'}
catch {if($_.Exception.Message -notmatch 'dry-run failed|preflight failed'){throw}}
if($global:WRegistered -or $global:WStarted){throw 'unsafe registration'}
'FAILURE_OK'
''', DATA=tmp_path/'data', OUT=tmp_path/'out', FAILURE=failure)
    ok(result, 'FAILURE_OK')


def test_exact_task_ownership_jobs_config_and_payload_tamper_guards():
    result = ps(r'''
$ErrorActionPreference='Stop'
$Outdir='C:\Users\Van\runs\parallel'; $code='C:\Users\Van\code'; $DataHome='C:\Users\Van\data'
$DatasetRoot=$DataHome+'\datasets\WiPose_processed_v1'; $EnvironmentPrefix='C:\Van\env'; $ActionNpy='C:\Van\actions\bend.npy'
$ResumeFrom='C:\Users\Van\runs\old_parallel'
$CleanOnly=$false; $JobsPerGpu=2; $env:CONDA_EXE='C:\Conda\conda.exe'
$currentSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
''' + FUNCTIONS + r'''
function Task($changes=@{}) {
    $cfg=@{helper='wipose_hpeli_fast_task';output=$Outdir;code=$code;data=$DataHome;dataset=$DatasetRoot;prefix=$EnvironmentPrefix;action=$ActionNpy;resume_from=$ResumeFrom;conda=$env:CONDA_EXE;clean_only=$false;jobs_per_gpu=2}
    foreach($key in $changes.Keys){$cfg[$key]=$changes[$key]}
    $json=$cfg|ConvertTo-Json -Compress
    $body=Get-WiPoseFastPayload ([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json)))
    [pscustomobject]@{Principal=[pscustomobject]@{UserId=$currentSid};Actions=@([pscustomobject]@{Execute="$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe";Arguments=('-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body)))})}
}
if(!(Test-OwnedTask (Task))){throw 'own task rejected'}
if(!(Test-OwnedTask (Task @{jobs_per_gpu=1}))){throw 'safe idle slot-count change rejected'}
foreach($change in @(@{resume_from=$ResumeFrom+'_other'},@{output=$Outdir+'_other'},@{code=$code+'_other'},@{dataset=$DatasetRoot+'_other'},@{prefix=$EnvironmentPrefix+'_other'},@{action=$ActionNpy+'_other'},@{jobs_per_gpu=3},@{jobs_per_gpu=$true},@{jobs_per_gpu='2'},@{clean_only=$true},@{helper='wipose_hpeli_task'})){
    if(Test-OwnedTask (Task $change)){throw 'different config accepted for overwrite'}
}
if(!(Test-OwnedTask (Task @{output=$Outdir+'_other';jobs_per_gpu=1;clean_only=$true}) -DatasetOnly)){throw 'same dataset cross-output task missed'}
if(Test-OwnedTask (Task @{dataset=$DatasetRoot+'_other'}) -DatasetOnly){throw 'different dataset blocked'}
$other=Task; $other.Principal.UserId='S-1-5-18'; if(Test-OwnedTask $other){throw 'other account accepted'}
$other=Task; $other.Actions[0].Execute='cmd.exe'; if(Test-OwnedTask $other){throw 'wrong executable accepted'}
$other=Task
$body=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String(($other.Actions[0].Arguments -split ' ')[-1]))+'; Write-Host injected'
$other.Actions[0].Arguments='-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
if(Test-OwnedTask $other){throw 'modified payload accepted for overwrite'}
'OWNERSHIP_OK'
''')
    ok(result, 'OWNERSHIP_OK')


@pytest.mark.parametrize('kind', ['root', 'clean', 'a1'])
def test_held_root_or_cell_lock_is_preserved(tmp_path, kind):
    output = tmp_path/'output'
    lock = output/({'root':'run.lock','clean':'hpeli/seed42/clean/run.lock',
                    'a1':'hpeli/seed42/a1/run.lock'}[kind])
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b'LOCK')
    result = ps(MOCK + REGISTER + r'''
$global:WFailFlag='-c'
try {& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__'; throw 'expected failure'}
catch {if($_.Exception.Message -notmatch 'output lock unavailable'){throw}}
if($global:WRegistered -or $global:WStarted){throw 'unsafe registration'}
if($global:WCalls.Count -ne 2 -or $global:WCalls[1][-1] -ne '__LOCK__'){throw 'wrong lock probe'}
'LOCK_OK'
''', DATA=tmp_path/'data', OUT=output, LOCK=lock)
    ok(result, 'LOCK_OK')
    assert lock.read_bytes() == b'LOCK'


@pytest.mark.parametrize('legacy,state', [(False,'Running'),(False,'Queued'),(True,'Running'),(True,'Queued')])
def test_active_owned_serial_or_parallel_payload_blocks_same_dataset_other_output(tmp_path, legacy, state):
    result = ps(MOCK + REGISTER + FUNCTIONS + r'''
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$cfg=@{helper='__HELPER_KIND__';output='__OTHER_OUT__';code='__CODE__';data='__DATA__';dataset=(Join-Path '__DATA__' 'datasets\WiPose_processed_v1');prefix='__PREFIX__';action=(Join-Path '__DATA__' 'actions\data_bend.npy');conda=$env:CONDA_EXE;clean_only=$false;jobs_per_gpu=2}|ConvertTo-Json -Compress
$base64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
if('__LEGACY__' -eq 'True') {
    $old=[System.Management.Automation.Language.Parser]::ParseFile('__SERIAL_HELPER__',[ref]$tokens,[ref]$errors)
    $template=$old.Find({param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst] -and $n.Value.StartsWith("`$ErrorActionPreference = 'Stop'") -and $n.Value.Contains('$cfg =')},$true).Value
    $body=$template.Replace('__CONFIG__',$base64)
} else {$body=Get-WiPoseFastPayload $base64}
$global:WFakeTask=[pscustomobject]@{State='__STATE__';Principal=[pscustomobject]@{UserId=$sid};Actions=@([pscustomobject]@{Execute="$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe";Arguments=('-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body)))})}
function Get-ScheduledTask {param($TaskName,$ErrorAction) if(!$TaskName){$global:WFakeTask}}
try {& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__'; throw 'expected failure'}
catch {if($_.Exception.Message -notmatch 'dataset is active'){throw}}
if($global:WRegistered -or $global:WStarted -or $global:WCalls.Count -ne 1){throw 'unsafe registration/preflight'}
'TASK_ACTIVE_OK'
''', DATA=tmp_path/'data', OUT=tmp_path/'new output', OTHER_OUT=tmp_path/'old output',
        CODE=HELPER.parents[1], LEGACY=legacy, STATE=state,
        HELPER_KIND='wipose_hpeli_task' if legacy else 'wipose_hpeli_fast_task')
    ok(result, 'TASK_ACTIVE_OK')


@pytest.mark.parametrize('kind', ['serial_python','parallel_python','serial_worker','parallel_worker',
                                 'serial_helper','parallel_helper','parallel_preflight'])
@pytest.mark.parametrize('late', [False, True])
def test_owned_same_dataset_process_in_other_output_blocks_startup_and_recheck(tmp_path, kind, late):
    dataset = tmp_path/'data'/'datasets'/'WiPose_processed_v1'
    other = tmp_path/'other output'
    if kind.endswith('_worker'):
        other.mkdir()
        import json
        name = 'wipose_hpeli_fast.resolved.json' if kind.startswith('parallel') else 'wipose_hpeli.resolved.json'
        (other/name).write_text(json.dumps({'cfg':{'dataset_root':str(dataset)}}), encoding='utf-8')
    result = ps(MOCK + REGISTER + r'''
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$global:WPCommand=''; $global:WPName='python.exe'
$kind='__KIND__'; $old=$kind.StartsWith('serial'); $script=$(if($old){'run_wipose_hpeli.py'}else{'run_wipose_hpeli_fast.py'})
if($kind.EndsWith('_worker')){
    $manifest=$(if($old){'wipose_hpeli.resolved.json'}else{'wipose_hpeli_fast.resolved.json'})
    $global:WPCommand='python "C:\different checkout\ATKBackd\'+$script+'" --_worker a1 --_phase train --outdir "__OTHER__" --_manifest "__OTHER__\'+$manifest+'"'
} elseif($kind.EndsWith('_helper')) {
    $global:WPName='powershell.exe'; $script=$(if($old){'Start-WiPoseHPELiTask.ps1'}else{'Start-WiPoseHPELiFastTask.ps1'})
    $global:WPCommand='powershell -NoProfile -File ".\remote_tools\'+$script+'" -DataHome "__DATA__" -Outdir "__OTHER__"'
} else {
    $global:WPCommand='python "C:\different checkout\ATKBackd\'+$script+'" --dataset-root "__DATASET__" --outdir "__OTHER__"'
    if($kind -eq 'parallel_preflight'){$global:WPCommand+=' --preflight'}
}
function Get-CimInstance {
    if($global:WCalls.Count -ge __MIN_CALLS__){[pscustomobject]@{Name=$global:WPName;ProcessId=424242;CommandLine=$global:WPCommand}}
}
function Invoke-CimMethod {param($InputObject,$MethodName,$ErrorAction) [pscustomobject]@{ReturnValue=0;Sid=$sid}}
try {& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__'; throw 'expected failure'}
catch {if($_.Exception.Message -notmatch 'dataset is active'){throw}}
if($global:WRegistered -or $global:WStarted -or $global:WCalls.Count -ne __MIN_CALLS__){throw 'unsafe registration/preflight'}
'PROCESS_ACTIVE_OK'
''', DATA=tmp_path/'data', DATASET=dataset, OUT=tmp_path/'new output', OTHER=other,
        KIND=kind, MIN_CALLS=2 if late else 1)
    ok(result, 'PROCESS_ACTIVE_OK')


def test_process_guard_ignores_other_dataset_nonwipose_foreign_owner_and_self():
    result = ps(r'''
$ErrorActionPreference='Stop'
$DatasetRoot='C:\data\WiPose'; $currentSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
''' + FUNCTIONS + r'''
function Invoke-CimMethod {param($InputObject,$MethodName,$ErrorAction)
    [pscustomobject]@{ReturnValue=0;Sid=$(if($InputObject.ProcessId -eq 8){'S-1-5-18'}else{$currentSid})}
}
$cases=@(
    [pscustomobject]@{Name='python.exe';ProcessId=1;CommandLine='python run_wipose_hpeli.py --dataset-root C:\data\WiPose --outdir C:\other'},
    [pscustomobject]@{Name='python.exe';ProcessId=2;CommandLine='python run_wipose_hpeli_fast.py --dataset-root C:\data\WiPose_other --outdir C:\other'},
    [pscustomobject]@{Name='python.exe';ProcessId=3;CommandLine='python run_a1_wiflow.py --dataset-root C:\data\WiPose'},
    [pscustomobject]@{Name='cmd.exe';ProcessId=4;CommandLine='cmd run_wipose_hpeli.py --dataset-root C:\data\WiPose'},
    [pscustomobject]@{Name='python.exe';ProcessId=5;CommandLine='python run_wipose_hpeli_fast.py --outdir C:\missing'},
    [pscustomobject]@{Name='powershell.exe';ProcessId=6;CommandLine='powershell -File Start-WiPoseHPELiTask.ps1 -DatasetRoot C:\data\WiPose_other'},
    [pscustomobject]@{Name='powershell.exe';ProcessId=$PID;CommandLine='powershell -File Start-WiPoseHPELiFastTask.ps1 -DatasetRoot C:\data\WiPose'},
    [pscustomobject]@{Name='python.exe';ProcessId=8;CommandLine='python run_wipose_hpeli_fast.py --dataset-root C:\data\WiPose'},
    [pscustomobject]@{Name='python.exe';ProcessId=9;CommandLine='python C:/other/run_wipose_hpeli_fast.py --dataset-root C:/data/WiPose'}
)
$owned=@($cases|Where-Object{Test-OwnedWorker $_})
if(($owned.ProcessId -join ',') -ne '1,5,9'){throw ('wrong workers '+($owned.ProcessId -join ','))}
'PROCESS_SCOPE_OK'
''')
    ok(result, 'PROCESS_SCOPE_OK')


@pytest.mark.parametrize('parameter', ['DataHome','DatasetRoot','Outdir','EnvironmentPrefix','ActionNpy'])
def test_invalid_paths_rejected_before_python(tmp_path, parameter):
    result = ps(MOCK + NO_TASKS + r'''
$kw=@{DataHome='__DATA__';EnvironmentPrefix='__PREFIX__';DryRun=$true}
$kw['__PARAMETER__']='C:\invalid"quoted\path'
try {& '__HELPER__' @kw; throw 'expected failure'}
catch {if($_.Exception.Message -notmatch 'without double quotes|Illegal characters'){throw}}
if($global:WCalls.Count){throw 'unexpected Python'}
'PATH_OK'
''', DATA=tmp_path/'data', PARAMETER=parameter)
    ok(result, 'PATH_OK')


def test_combined_modes_are_rejected_before_python(tmp_path):
    result = ps(MOCK + NO_TASKS + r'''
try {& '__HELPER__' -DataHome '__DATA__' -EnvironmentPrefix '__PREFIX__' -DryRun -PreflightOnly; throw 'expected failure'}
catch {if($_.Exception.Message -notmatch 'Choose only one'){throw}}
if($global:WCalls.Count){throw 'unexpected Python'}
'MODE_OK'
''', DATA=tmp_path/'data')
    ok(result, 'MODE_OK')


def test_idle_authenticated_task_allows_two_to_one_worker_resume(tmp_path):
    result = ps(MOCK + REGISTER + FUNCTIONS + r'''
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$WCode=[IO.Path]::GetFullPath((Split-Path -Parent (Split-Path -Parent '__HELPER__')))
$cfg=@{helper='wipose_hpeli_fast_task';output='__OUT__';code=$WCode;data='__DATA__';dataset=(Join-Path '__DATA__' 'datasets\WiPose_processed_v1');prefix='__PREFIX__';action=(Join-Path '__DATA__' 'actions\data_bend.npy');resume_from=(Join-Path '__DATA__' 'runs\wipose_hpeli_a1_s42_parallel_v1');conda=$env:CONDA_EXE;clean_only=$false;jobs_per_gpu=2}|ConvertTo-Json -Compress
$body=Get-WiPoseFastPayload ([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg)))
$global:WFakeTask=[pscustomobject]@{State='Ready';Principal=[pscustomobject]@{UserId=$sid};Actions=@([pscustomobject]@{Execute="$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe";Arguments=('-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body)))})}
function Get-ScheduledTask {param($TaskName,$ErrorAction) $global:WFakeTask}
& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__' -JobsPerGpu 1
if($global:WRegistered -ne 1 -or $global:WStarted -ne 1){throw 'safe idle operational resume blocked'}
if($global:WAction.Arguments -notmatch '-EncodedCommand\s+([A-Za-z0-9+/=]+)'){throw 'no replacement payload'}
$body=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
if($body -notmatch "FromBase64String\('([A-Za-z0-9+/=]+)'\)"){throw 'no replacement config'}
$saved=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Matches[1]))|ConvertFrom-Json
if($saved.jobs_per_gpu -ne 1){throw 'requested slots not saved'}
'IDLE_RESUME_OK'
''', DATA=tmp_path/'data', OUT=tmp_path/'output')
    ok(result, 'IDLE_RESUME_OK')


@pytest.mark.parametrize('kind', ['serial','parallel'])
def test_encoded_persistent_wrapper_process_blocks_same_dataset(kind):
    result = ps(r'''
$ErrorActionPreference='Stop'
$DatasetRoot='C:\data\WiPose'; $currentSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
''' + FUNCTIONS + r'''
function Invoke-CimMethod {param($InputObject,$MethodName,$ErrorAction) [pscustomobject]@{ReturnValue=0;Sid=$currentSid}}
$cfg=@{helper='__HELPER_KIND__';dataset=$DatasetRoot;output='C:\other output';jobs_per_gpu=2;clean_only=$false}|ConvertTo-Json -Compress
$base64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
if('__KIND__' -eq 'serial'){
    $old=[System.Management.Automation.Language.Parser]::ParseFile('__SERIAL_HELPER__',[ref]$tokens,[ref]$errors)
    $template=$old.Find({param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst] -and $n.Value.StartsWith("`$ErrorActionPreference = 'Stop'") -and $n.Value.Contains('$cfg =')},$true).Value
    $body=$template.Replace('__CONFIG__',$base64)
}else{$body=Get-WiPoseFastPayload $base64}
$line='powershell -NoProfile -EncodedCommand '+[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
if(!(Test-OwnedWorker ([pscustomobject]@{Name='powershell.exe';ProcessId=1234;CommandLine=$line}))){throw 'encoded live wrapper missed'}
'WRAPPER_OK'
''', KIND=kind, HELPER_KIND='wipose_hpeli_task' if kind=='serial' else 'wipose_hpeli_fast_task')
    ok(result, 'WRAPPER_OK')


@pytest.mark.parametrize('kind', ['task','python','worker','helper','wrapper'])
@pytest.mark.parametrize('late', [False, True])
def test_old_parallel_task_and_processes_block_fast_start_and_post_preflight_recheck(tmp_path, kind, late):
    other=tmp_path/'old output'
    other.mkdir()
    import json
    dataset=tmp_path/'data/datasets/WiPose_processed_v1'
    (other/'wipose_hpeli_parallel.resolved.json').write_text(
        json.dumps({'cfg':{'dataset_root':str(dataset)}}), encoding='utf-8')
    result=ps(MOCK+REGISTER+r'''
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$cfg=@{helper='wipose_hpeli_parallel_task';dataset='__DATASET__';output='__OTHER__'}|ConvertTo-Json -Compress
$config64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
$tokens=$null;$errors=$null
$old=[System.Management.Automation.Language.Parser]::ParseFile('__OLD_PARALLEL__',[ref]$tokens,[ref]$errors)
$template=$old.Find({param($node) $node -is [System.Management.Automation.Language.StringConstantExpressionAst] -and $node.Value.StartsWith("`$ErrorActionPreference = 'Stop'") -and $node.Value.Contains('$cfg =')},$true).Value
$body=$template.Replace('__CONFIG__',$config64)
$encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
$global:WOldTask=[pscustomobject]@{State='Running';Principal=[pscustomobject]@{UserId=$sid};Actions=@([pscustomobject]@{Execute="$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe";Arguments=('-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$encoded)})}
$global:WName='python.exe'
$global:WLine='python "C:\old checkout\ATKBackd\run_wipose_hpeli_parallel.py" --dataset-root "__DATASET__" --outdir "__OTHER__"'
if('__KIND__' -eq 'worker'){$global:WLine='python "C:\old checkout\ATKBackd\run_wipose_hpeli_parallel.py" --_worker a1 --outdir "__OTHER__" --_manifest "__OTHER__\wipose_hpeli_parallel.resolved.json"'}
if('__KIND__' -eq 'helper'){$global:WName='powershell.exe';$global:WLine='powershell -File Start-WiPoseHPELiParallelTask.ps1 -DatasetRoot "__DATASET__"'}
if('__KIND__' -eq 'wrapper'){$global:WName='powershell.exe';$global:WLine='powershell -NoProfile -EncodedCommand '+$encoded}
function Get-ScheduledTask {param($TaskName,$ErrorAction)
 if('__KIND__' -eq 'task' -and !$TaskName -and $global:WCalls.Count -ge __MIN_CALLS__){$global:WOldTask}
}
function Get-CimInstance {
 if('__KIND__' -ne 'task' -and $global:WCalls.Count -ge __MIN_CALLS__){[pscustomobject]@{Name=$global:WName;ProcessId=424242;CommandLine=$global:WLine}}
}
function Invoke-CimMethod {param($InputObject,$MethodName,$ErrorAction) [pscustomobject]@{ReturnValue=0;Sid=$sid}}
try {& '__HELPER__' -DataHome '__DATA__' -Outdir '__OUT__' -EnvironmentPrefix '__PREFIX__';throw 'expected refusal'}
catch {if($_.Exception.Message -notmatch 'dataset is active'){throw}}
if($global:WRegistered -or $global:WStarted -or $global:WCalls.Count -ne __MIN_CALLS__){throw 'unsafe launch/preflight'}
'OLD_PARALLEL_BLOCKED_OK'
''', DATA=tmp_path/'data', OUT=tmp_path/'fast output', OTHER=other, DATASET=dataset,
        OLD_PARALLEL=HELPER.with_name('Start-WiPoseHPELiParallelTask.ps1'),
        KIND=kind, MIN_CALLS=2 if late else 1)
    ok(result,'OLD_PARALLEL_BLOCKED_OK')
