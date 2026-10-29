"""SSH-independent Windows handoff tests; never register real tasks or train."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "remote_tools/Start-PiWMetaFiFourAttacks3090Task.ps1"
PS = shutil.which("powershell") or shutil.which("pwsh")


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_ps(command):
    return subprocess.run(
        [PS, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, timeout=45,
    )


def assert_ok(result):
    assert result.returncode == 0, result.stdout + result.stderr


def source():
    return HELPER.read_text(encoding="utf-8-sig")


def functions_prelude(*names):
    """Load production functions via AST only; never execute the script body."""
    selected = ",".join(literal(name) for name in names)
    return (
        "$ErrorActionPreference='Stop';"
        "$ast=[System.Management.Automation.Language.Parser]::ParseFile("
        + literal(HELPER)
        + ",[ref]$null,[ref]$null);"
        "$functions=@($ast.FindAll({param($n) $n -is "
        "[System.Management.Automation.Language.FunctionDefinitionAst]},$true));"
        f"$names=@({selected});"
        "foreach($name in $names){"
        "$found=@($functions|Where-Object Name -eq $name);"
        "if($found.Count -ne 1){throw ('Expected one production function: '+$name)};"
        "Invoke-Expression $found[0].Extent.Text}"
    )


def test_sidecar_keeps_existing_gate_and_four_attack_queue():
    text = source()
    assert "piw_metafi_remaining_3090" in text
    assert "controller.py" in text
    assert "piw_metafi_four_attacks_s42_3090_v1" in text
    assert "--four-attacks-only" in text
    assert "--mode" in text
    assert "Prepare" not in re.search(r"\[ValidateSet\((.*?)\)\]", text, re.S).group(1)
    for forbidden in (
        "Remove-Item", "Stop-Process", "taskkill", "Unregister-ScheduledTask",
        "Set-ScheduledTask", "Invoke-WebRequest", "pip install", "git pull",
        "File]::Replace", "Set-ExecutionPolicy", "--jobs-per-gpu",
        "--mode prepare", "--mode inspect", "--method",
    ):
        assert forbidden.casefold() not in text.casefold(), forbidden


def test_original_controller_gate_source_is_not_rewritten_by_this_repair():
    frozen = ROOT / "remote_tools/piw_metafi_remaining_3090/controller.py"
    expected = "1bb43a9e73a16c74e5c57972bebc2bce7b948cc52c8c19c5599df1b8ef7b80c2"
    # This handoff deliberately keeps the prepared run's source binding intact.
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == expected
    assert expected in source().lower()


def test_scheduler_has_no_ssh_parent_or_short_default_timeout():
    text = source()
    assert "New-ScheduledTaskPrincipal" in text
    assert "S4U" in text
    assert "Limited" in text
    assert "New-ScheduledTaskSettingsSet" in text
    assert "ExecutionTimeLimit" in text
    assert "IgnoreNew" in text
    assert "AllowStartIfOnBatteries" in text
    assert "DontStopIfGoingOnBatteries" in text
    assert "Register-ScheduledTask" in text
    assert "Start-ScheduledTask" in text
    # A detached Start-Process inside an SSH-descended worker was the old bug.
    assert not re.search(r"(?im)^\s*(?:\$\w+\s*=\s*)?Start-Process\b", text)
    assert "New-ScheduledTaskTrigger" not in text
    assert "RestartInterval" not in text
    assert "-Force" not in text
    assert "-Password" not in text
    assert "RunLevel Highest" not in text


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
def test_powershell_parses_without_executing_task_or_controller():
    result = run_ps(
        "$t=$null;$e=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile("
        + literal(HELPER)
        + ",[ref]$t,[ref]$e)|Out-Null;"
        "if($e.Count){$e|Out-String|Write-Output;exit 1}"
    )
    assert_ok(result)


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("value,expected", [
    ("plain", '"plain"'),
    ("C:\\path with spaces\\file.py", '"C:\\path with spaces\\file.py"'),
    ("C:\\", '"C:\\\\"'),
    ('a"b', '"a\\"b"'),
    ("", '""'),
])
def test_native_task_arguments_preserve_spaces_and_trailing_backslashes(value, expected):
    result = run_ps(
        functions_prelude("Quote-WindowsArgument")
        + "Quote-WindowsArgument " + literal(value) + "|ConvertTo-Json -Compress"
    )
    assert_ok(result)
    assert json.loads(result.stdout.strip()) == expected


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("value", ["bad\nline", "bad\rline", "bad\x00nul"])
def test_native_task_arguments_fail_closed_on_control_injection(value):
    # Pass escaped JSON instead of a literal NUL in the subprocess command line.
    encoded = json.dumps(value)
    result = run_ps(
        functions_prelude("Quote-WindowsArgument")
        + "$bad=ConvertFrom-Json " + literal(encoded) + ";"
        + "try{Quote-WindowsArgument $bad|Out-Null;throw 'ACCEPTED_UNSAFE_VALUE'}"
        + "catch{if($_.Exception.Message -eq 'ACCEPTED_UNSAFE_VALUE'){throw};'REJECTED'}"
    )
    assert_ok(result)
    assert "REJECTED" in result.stdout


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("phase", ["check", "launch"])
def test_controller_command_preserves_exact_original_resume_contract(phase):
    request = {
        "Controller": "C:\\frozen repo\\controller.py",
        "DataHome": "C:\\home",
        "Dataset": "C:\\data",
        "Action": "C:\\action.npy",
        "Outdir": "C:\\existing run",
        "CheckpointHome": "C:\\import",
    }
    ps_request = "@{" + ";".join(k + "=" + literal(v) for k, v in request.items()) + "}"
    result = run_ps(
        functions_prelude("Get-ControllerArguments")
        + "$request=" + ps_request + ";"
        + "@(Get-ControllerArguments $request " + literal(phase) + ")|ConvertTo-Json -Compress"
    )
    assert_ok(result)
    assert json.loads(result.stdout) == [
        "-u", request["Controller"], "--data-home", request["DataHome"],
        "--dataset", request["Dataset"], "--action", request["Action"],
        "--out", request["Outdir"], "--checkpoint-home", request["CheckpointHome"],
        "--four-attacks-only", "--mode", phase,
    ]


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("phase", ["prepare", "inspect", "cell", "train", "evaluate", ""])
def test_controller_wrapper_cannot_prepare_a_new_run_or_schedule_arbitrary_cells(phase):
    result = run_ps(
        functions_prelude("Get-ControllerArguments")
        + "try{Get-ControllerArguments @{} " + literal(phase) + "|Out-Null;throw 'BAD_ACCEPT'}"
        + "catch{if($_.Exception.Message -eq 'BAD_ACCEPT'){throw};'REJECTED'}"
    )
    assert_ok(result)
    assert "REJECTED" in result.stdout


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
def test_real_foreground_process_drains_both_pipes_and_preserves_nonzero_exit(tmp_path):
    fixture = tmp_path / "fake controller with spaces.py"
    fixture.write_text(
        "import json, os, sys, time\n"
        "print(json.dumps({'arg': sys.argv[1], 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'order': os.environ.get('CUDA_DEVICE_ORDER'), 'cache': os.environ.get('TORCH_HOME'), "
        "'utf8': os.environ.get('PYTHONUTF8')}), flush=True)\n"
        "sys.stdout.write('O' * 160000); sys.stdout.flush()\n"
        "sys.stderr.write('E' * 160000); sys.stderr.flush()\n"
        "time.sleep(0.1)\n"
        "print('\\nFINISHED_FOREGROUND', flush=True)\n"
        "sys.exit(37)\n",
        encoding="utf-8",
    )
    prefix = tmp_path / "new unique controller"
    arg = 'space and "quoted" value with trailing\\'
    command = (
        functions_prelude("Quote-WindowsArgument", "Invoke-LoggedProcess")
        + "$request=@{PythonExe=" + literal(sys.executable) + ";Repo=" + literal(tmp_path)
        + ";Environment=@{CUDA_VISIBLE_DEVICES='0';CUDA_DEVICE_ORDER='PCI_BUS_ID';TORCH_HOME='C:\\cached weights'}};"
        + "$code=Invoke-LoggedProcess $request @(" + literal(fixture) + "," + literal(arg)
        + ") " + literal(prefix) + ";Write-Output ('EXIT_CODE='+$code)"
    )
    result = run_ps(command)
    assert_ok(result)
    assert "EXIT_CODE=37" in result.stdout
    out = Path(str(prefix) + ".stdout.log").read_text(encoding="utf-8")
    err = Path(str(prefix) + ".stderr.log").read_text(encoding="utf-8")
    metadata = json.loads(out.splitlines()[0])
    assert metadata == {
        "arg": arg, "cuda": "0", "order": "PCI_BUS_ID",
        "cache": "C:\\cached weights", "utf8": "1",
    }
    assert "O" * 160000 in out
    assert out.endswith("FINISHED_FOREGROUND\n")
    assert err == "E" * 160000


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
def test_foreground_logs_are_never_overwritten(tmp_path):
    prefix = tmp_path / "existing log"
    out = Path(str(prefix) + ".stdout.log")
    out.write_text("preserve historical log", encoding="utf-8")
    result = run_ps(
        functions_prelude("Quote-WindowsArgument", "Invoke-LoggedProcess")
        + "$request=@{PythonExe=" + literal(sys.executable) + ";Repo=" + literal(tmp_path)
        + ";Environment=@{}};"
        + "try{Invoke-LoggedProcess $request @('-c','raise SystemExit(0)') "
        + literal(prefix) + ";throw 'BAD_OVERWRITE'}"
        + "catch{if($_.Exception.Message -eq 'BAD_OVERWRITE'){throw};'PRESERVED'}"
    )
    assert_ok(result)
    assert "PRESERVED" in result.stdout
    assert out.read_text(encoding="utf-8") == "preserve historical log"
    assert not Path(str(prefix) + ".stderr.log").exists()


def valid_task_fixture():
    return r"""
$spec=@{
    Name='PiWMetaFi4-fixture';UserSID='S-1-5-21-1-2-3-1001';
    Execute='C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe';
    Arguments='"-NoProfile" "-File" "C:\helper.ps1" "-Mode" "Run"';
    WorkingDirectory='C:\frozen original source';
    Description='PiWMetaFi4 S4U continuation; request SHA256=fixture'
}
$task=[pscustomobject]@{
    TaskName=$spec.Name;TaskPath='\';State='Ready';Description=$spec.Description;
    Actions=@([pscustomobject]@{Execute=$spec.Execute;Arguments=$spec.Arguments;WorkingDirectory=$spec.WorkingDirectory});
    Principal=[pscustomobject]@{UserId=$spec.UserSID;LogonType='S4U';RunLevel='Limited'};
    Triggers=@();
    Settings=[pscustomobject]@{
        ExecutionTimeLimit='PT0S';MultipleInstances='IgnoreNew';RestartCount=0;
        DisallowStartIfOnBatteries=$false;StopIfGoingOnBatteries=$false;
        RunOnlyIfIdle=$false;RunOnlyIfNetworkAvailable=$false;
        AllowDemandStart=$true;Enabled=$true
    }
}
"""


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
def test_real_definition_builder_uses_safe_scheduler_parameters_with_mocked_cmdlets():
    # All Task Scheduler cmdlets are replaced in-process; no task is registered.
    command = functions_prelude("New-TaskDefinition") + valid_task_fixture() + r"""
$script:calls=@()
function New-ScheduledTaskAction($Execute,$Argument,$WorkingDirectory){
    $script:calls+='action'
    return [pscustomobject]@{Execute=$Execute;Arguments=$Argument;WorkingDirectory=$WorkingDirectory}
}
function New-ScheduledTaskPrincipal($UserId,$LogonType,$RunLevel){
    $script:calls+='principal'
    return [pscustomobject]@{UserId=$UserId;LogonType=$LogonType;RunLevel=$RunLevel}
}
function New-ScheduledTaskSettingsSet([timespan]$ExecutionTimeLimit,$MultipleInstances,
    [switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries){
    $script:calls+='settings'
    if($ExecutionTimeLimit.Ticks -ne 0){throw 'Short timeout'}
    if(!$AllowStartIfOnBatteries -or !$DontStopIfGoingOnBatteries){throw 'Battery restrictions'}
    return [pscustomobject]@{ExecutionTimeLimit='PT0S';MultipleInstances=$MultipleInstances;RestartCount=0}
}
function New-ScheduledTask($Action,$Principal,$Settings,$Description){
    $script:calls+='definition'
    return [pscustomobject]@{Actions=@($Action);Principal=$Principal;Settings=$Settings;Description=$Description}
}
$definition=New-TaskDefinition $spec
@{Calls=@($script:calls);Definition=$definition}|ConvertTo-Json -Depth 8 -Compress
"""
    result = run_ps(command)
    assert_ok(result)
    actual = json.loads(result.stdout)
    assert actual["Calls"] == ["action", "principal", "settings", "definition"]
    definition = actual["Definition"]
    assert definition["Principal"] == {
        "UserId": "S-1-5-21-1-2-3-1001", "LogonType": "S4U", "RunLevel": "Limited",
    }
    assert definition["Settings"] == {
        "ExecutionTimeLimit": "PT0S", "MultipleInstances": "IgnoreNew", "RestartCount": 0,
    }
    assert len(definition["Actions"]) == 1
    assert definition["Actions"][0]["WorkingDirectory"] == r"C:\frozen original source"


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("change", [
    "", "$task.Triggers=$null", "$task.Principal.RunLevel='LeastPrivilege'",
    "$task.Principal.LogonType=2;$task.Principal.RunLevel=0;$task.Settings.MultipleInstances=2",
])
def test_task_verifier_accepts_safe_no_trigger_definition_and_cim_enum_values(change):
    result = run_ps(
        functions_prelude("ConvertTo-UserSID", "Assert-TaskDefinition") + valid_task_fixture()
        + change + ";Assert-TaskDefinition $task $spec;'ACCEPTED'"
    )
    assert_ok(result)
    assert "ACCEPTED" in result.stdout


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("change", [
    "$task.TaskName='foreign task'",
    "$task.TaskPath='\\foreign\\'",
    "$task.Actions=@($task.Actions[0],$task.Actions[0])",
    "$task.Actions[0].Execute='foreign.exe'",
    "$task.Actions[0].Arguments+=' -foreign'",
    "$task.Actions[0].WorkingDirectory='C:\\foreign'",
    "$task.Principal.UserId='S-1-5-18'",
    "$task.Principal.LogonType='InteractiveToken'",
    "$task.Principal.LogonType=3",
    "$task.Principal.RunLevel='HighestAvailable'",
    "$task.Principal.RunLevel=1",
    "$task.Description='unrelated task'",
    "$task.Triggers=@([pscustomobject]@{Enabled=$true})",
    "$task.Settings.ExecutionTimeLimit='PT72H'",
    "$task.Settings.MultipleInstances='Parallel'",
    "$task.Settings.MultipleInstances=0",
    "$task.Settings.RestartCount=1",
    "$task.Settings.DisallowStartIfOnBatteries=$true",
    "$task.Settings.StopIfGoingOnBatteries=$true",
    "$task.Settings.RunOnlyIfIdle=$true",
    "$task.Settings.RunOnlyIfNetworkAvailable=$true",
    "$task.Settings.AllowDemandStart=$false",
    "$task.Settings.Enabled=$false",
])
def test_task_verifier_rejects_foreign_or_unsafe_definition_without_mutation(change):
    result = run_ps(
        functions_prelude("ConvertTo-UserSID", "Assert-TaskDefinition") + valid_task_fixture()
        + "Assert-TaskDefinition $task $spec;" + change + ";"
        + "try{Assert-TaskDefinition $task $spec;throw 'BAD_ACCEPT'}"
        + "catch{if($_.Exception.Message -eq 'BAD_ACCEPT'){throw};'REJECTED_NO_MUTATION'}"
    )
    assert_ok(result)
    assert "REJECTED_NO_MUTATION" in result.stdout


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
def test_request_bytes_are_stable_across_hashtable_insertion_order():
    result = run_ps(
        functions_prelude("ConvertTo-RequestJSON", "Get-TextSHA") + r"""
$one=@{Profile='fixed';Repo='C:\repo';Environment=@{TORCH_HOME=$null;CUDA_VISIBLE_DEVICES='0';CUDA_DEVICE_ORDER='PCI_BUS_ID'}}
$two=@{Environment=@{CUDA_DEVICE_ORDER='PCI_BUS_ID';CUDA_VISIBLE_DEVICES='0';TORCH_HOME=$null};Repo='C:\repo';Profile='fixed'}
$a=ConvertTo-RequestJSON $one
$b=ConvertTo-RequestJSON $two
if($a -cne $b -or (Get-TextSHA $a) -cne (Get-TextSHA $b)){throw 'Unstable content addressed request'}
$a
"""
    )
    assert_ok(result)
    parsed = json.loads(result.stdout)
    assert list(parsed) == ["Environment", "Profile", "Repo"]
    assert list(parsed["Environment"]) == ["CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "TORCH_HOME"]


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("kind", ["present", "not_found", "access_denied", "service_failure"])
def test_owned_task_lookup_does_not_treat_permission_or_service_errors_as_absence(kind):
    command = functions_prelude("Find-OwnedTask") + r"""
$script:kind=KIND
function Get-ScheduledTask {
    [CmdletBinding()]param($TaskName,$TaskPath)
    if($TaskName -ne 'fixed-owned-task' -or $TaskPath -ne '\'){throw 'Broad or wrong task lookup'}
    if($script:kind -eq 'present'){return [pscustomobject]@{TaskName=$TaskName}}
    if($script:kind -eq 'not_found'){
        $record=[Management.Automation.ErrorRecord]::new(
            [Exception]::new('absent'),'CmdletizationQuery_NotFound',
            [Management.Automation.ErrorCategory]::ObjectNotFound,$TaskName)
    } elseif($script:kind -eq 'access_denied'){
        $record=[Management.Automation.ErrorRecord]::new(
            [UnauthorizedAccessException]::new('denied'),'AccessDenied',
            [Management.Automation.ErrorCategory]::PermissionDenied,$TaskName)
    } else {
        $record=[Management.Automation.ErrorRecord]::new(
            [Exception]::new('service unavailable'),'ServiceUnavailable',
            [Management.Automation.ErrorCategory]::ResourceUnavailable,$TaskName)
    }
    $PSCmdlet.ThrowTerminatingError($record)
}
try{
    $value=Find-OwnedTask 'fixed-owned-task'
    if($script:kind -in @('access_denied','service_failure')){throw 'SWALLOWED_FAILURE'}
    if($script:kind -eq 'not_found' -and $null -ne $value){throw 'Not found should return null'}
    if($script:kind -eq 'present' -and $value.TaskName -ne 'fixed-owned-task'){throw 'Lost task'}
    'EXPECTED_LOOKUP_RESULT'
}catch{
    if($script:kind -notin @('access_denied','service_failure') -or $_.Exception.Message -eq 'SWALLOWED_FAILURE'){throw}
    'EXPECTED_LOOKUP_RESULT'
}
"""
    command = command.replace("KIND", literal(kind))
    result = run_ps(command)
    assert_ok(result)
    assert "EXPECTED_LOOKUP_RESULT" in result.stdout


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
def test_actual_internal_run_rejects_changed_request_before_any_native_child(tmp_path):
    request = tmp_path / "changed request.json"
    request.write_text('{"Schema":1,"Profile":"changed"}', encoding="utf-8")
    result = subprocess.run(
        [PS, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(HELPER), "-Mode", "Run", "-RequestFile", str(request),
         "-RequestSHA", "0" * 64],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "Internal Run requires an intact Launch request" in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == [request.name]
