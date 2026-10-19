"""Windows helper tests: parse/mocks/dry-run only, never register/stop a task."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_a1_defense_runner import plans

HELPER = Path(__file__).resolve().parents[2] / 'remote_tools' / 'Start-A1DefenseTask.ps1'
POWERSHELL = shutil.which('powershell.exe')
pytestmark = pytest.mark.skipif(os.name != 'nt' or not POWERSHELL,
                                reason='Windows PowerShell 5.1 integration')


def _powershell(source):
    encoded = base64.b64encode(source.encode('utf-16-le')).decode('ascii')
    return subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                          capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=60)


def test_helper_and_scheduled_action_parse_without_executing():
    result = _powershell(r'''
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
'PARSE_OK'
'''.replace('__HELPER__', str(HELPER).replace("'", "''")))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PARSE_OK' in result.stdout


def test_task_and_process_ownership_reject_other_outputs_no_processes_touched():
    result = _powershell(r'''
$ErrorActionPreference='Stop'
$Outdir='C:\Users\Van\backdooranalog\runs\defenses'
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__HELPER__',[ref]$tokens,[ref]$errors)
foreach($name in @('Test-OwnedTask','Get-OwnedWorkers')){
    $fn=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
    . ([scriptblock]::Create($fn.Extent.Text))
}
function Task($body){
    $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
    [pscustomobject]@{
        Principal=[pscustomobject]@{UserId=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value}
        Actions=@([pscustomobject]@{Arguments="-EncodedCommand $encoded"})
    }
}
$old=Task ('python run_a1_defenses.py --outdir "'+$Outdir+'"')
if(!(Test-OwnedTask $old)){throw 'Old sequential task not recognized'}
$otherAccount=Task ('python run_a1_defenses.py --outdir "'+$Outdir+'"')
$otherAccount.Principal.UserId='S-1-5-18'
if(Test-OwnedTask $otherAccount){throw 'Other account task recognized'}
if(Test-OwnedTask (Task ('python run_a1_defenses.py --outdir "'+$Outdir+'_other"'))){throw 'Sibling old task recognized'}
$cfg=@{output=$Outdir}|ConvertTo-Json -Compress
$cfg64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
$body="FromBase64String('$cfg64'); run_a1_defenses_parallel.py"
if(!(Test-OwnedTask (Task $body))){throw 'New encoded config not recognized'}
if(Test-OwnedTask (Task 'python other.py')){throw 'Unrelated task recognized'}
$cfg=@{output=$Outdir+'_other'}|ConvertTo-Json -Compress
$cfg64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cfg))
if(Test-OwnedTask (Task "FromBase64String('$cfg64'); run_a1_defenses_parallel.py")){throw 'Sibling task recognized'}
$pattern=[regex]::Escape($Outdir)+'(?=[\\/"\s]|$)'
function Get-CimInstance {
    [pscustomobject]@{Name='python.exe';ProcessId=1;CommandLine=('python run_a1_defenses.py --outdir "'+$Outdir+'"')}
    [pscustomobject]@{Name='python.exe';ProcessId=2;CommandLine=('python run_a1_defenses.py --_manifest "'+$Outdir+'\a1_defenses.resolved.json"')}
    [pscustomobject]@{Name='python.exe';ProcessId=3;CommandLine=('python run_a1_defenses.py --outdir "'+$Outdir+'_other"')}
    [pscustomobject]@{Name='python.exe';ProcessId=4;CommandLine='python yolo.py'}
}
$owned=@(Get-OwnedWorkers)
if(($owned.ProcessId -join ',') -ne '1,2'){throw 'Process ownership incorrect'}
'OWNERSHIP_OK'
'''.replace('__HELPER__', str(HELPER).replace("'", "''")))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OWNERSHIP_OK' in result.stdout


def test_helper_real_dry_run_creates_no_plan_or_tasks(plans):
    _, _, root, output = plans
    before = list(root.parent.rglob('*'))
    script = r'''
$ErrorActionPreference='Stop'
function Register-ScheduledTask { throw 'must not register task' }
function Start-ScheduledTask { throw 'must not start task' }
function Stop-ScheduledTask { throw 'must not stop task' }
function Stop-Process { throw 'must not stop process' }
& '__HELPER__' -SourceOutdir '__ROOT__' -Outdir '__OUTPUT__' -EnvironmentPrefix '__PREFIX__' -DryRun
'''
    for key, value in {'__HELPER__': HELPER, '__ROOT__': root, '__OUTPUT__': output,
                       '__PREFIX__': Path(sys.executable).parent}.items():
        script = script.replace(key, str(value).replace("'", "''"))
    result = _powershell(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'DRY RUN: 5 selected evaluations; 2 slots' in result.stdout
    assert not output.exists()
    assert list(root.parent.rglob('*')) == before
