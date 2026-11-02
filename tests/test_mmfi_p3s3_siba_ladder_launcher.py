"""PowerShell wrapper contracts: no dataset/GPU/network or real training."""
from pathlib import Path
import json
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "remote_tools/Start-MMFiP3S3SibaLadder4090.ps1"
DOC = ROOT / "remote_tools/MMFI_P3S3_SIBA_LADDER_4090_VI.md"
PS = shutil.which("powershell") or shutil.which("pwsh")


def text():
    return HELPER.read_text(encoding="utf-8")


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


@pytest.fixture
def mock_helper(tmp_path):
    """The wrapper sees a checkout but never executes the placeholder runner."""
    checkout = tmp_path / "mock checkout"
    helper = checkout / "remote_tools" / HELPER.name
    runner = checkout / "ATKBackd/run_mmfi_p3s3_siba_ladder_4090.py"
    helper.parent.mkdir(parents=True)
    runner.parent.mkdir(parents=True)
    helper.write_text(text(), encoding="utf-8")
    runner.write_text("# not executed by these wrapper tests\n", encoding="utf-8")
    return helper


def test_only_new_runner_one_worker_and_fresh_surrogate():
    source = text()
    assert "ATKBackd\\run_mmfi_p3s3_siba_ladder_4090.py" in source
    assert "mmfi_p3s3_siba_ladder_source_adapted_s42_4090_v2" in source
    assert "[ValidateSet(1)][int]$JobsPerGpu = 1" in source
    assert "[ValidateSet('cuda:0')][string]$Device = 'cuda:0'" in source
    assert "--jobs-per-gpu" in source
    assert "--clean-checkpoint" not in source
    assert "$CleanCheckpoint" not in source
    assert "NEW Clean surrogate" in source
    assert "[ValidateSet('adaptive')][string]$BackTimePolicy = ''" in source
    assert "[ValidateSet('Requested','SibaBackTime')][string]$MethodSet = 'Requested'" in source
    assert "if ($BackTimePolicy) { $Common += @('--backtime-policy',$BackTimePolicy) }" in source
    assert "if ($MethodSet -eq 'SibaBackTime') { $BackTimePolicy = 'adaptive' }" in source
    assert "if ($MethodSet -eq 'SibaBackTime') { $Common += @('--methods','siba','backtime') }" in source
    assert 'mmfi_p3s3_siba_ladder_backtime_adaptive_source_adapted_s42_4090_v2' in source
    assert "$RunName.Replace('_s42_','_siba_backtime_only_s42_')" in source
    assert "OWN scratch attacker surrogate for100 epochs:11 clean warmup +89 alternating" in source
    assert "SIBA full-TRAIN search200/B128" in source
    assert "LADDER remains blocked; it is not silently skipped" in source


def test_no_expansive_mutations_or_unverified_persistence():
    source = text()
    for forbidden in ("Start-Process", "Stop-Process", "taskkill", "schtasks", "Invoke-CimMethod",
                      "Invoke-WebRequest", "pip install", "Remove-Item", "git clone", "git pull",
                      "AllowConcurrent", "--allow-concurrent"):
        assert forbidden.casefold() not in source.casefold()
    assert "FOREGROUND: keep this SSH session open" in source
    assert "No environment was installed" in source
    assert "Outdir must be a separate run folder" in source
    assert "[IO.Path]::GetPathRoot($Outdir)" in source


def test_prepare_stops_and_launch_checks_without_auto_prepare():
    source = text()
    prepare = source.split("if ($Mode -eq 'Prepare') {", 1)[1].split("if ($Mode -eq 'CheckOnly')", 1)[0]
    assert "Invoke-Runner 'prepare'" in prepare
    assert "return" in prepare
    assert "Invoke-Runner 'launch'" not in prepare
    assert "native Clean B32 CUDA smoke" in prepare
    assert "no full Clean/victim training or full trigger search" in prepare
    launch = source.split("# Check never prepares", 1)[1]
    assert launch.index("Invoke-Runner 'check'") < launch.index("Invoke-Runner 'launch'")
    assert "Invoke-Runner 'prepare'" not in launch
    assert "no later phase was started by this invocation" in source
    assert "no full training started" not in source


def test_defaults_existing_envs_and_array_argument_forwarding():
    source = text()
    assert "envs\\dose-backdoor\\python.exe" in source
    assert "envs\\mmfi-p3\\Scripts\\python.exe" in source
    assert "& $PythonExe @Common '--mode' $Phase" in source
    assert "Push-Location -LiteralPath $Repo" in source
    assert "finally" in source and "Pop-Location" in source


def test_document_discloses_adaptation_selection_and_cost():
    doc = " ".join(DOC.read_text(encoding="utf-8").split()).casefold()
    for required in ("Material Passport", "exploratory", "P3S3", "seed42", "240570 TRAIN / 80190 TEST",
                     "3 full scientific victim/reference trainings", "từ scratch", "cap loss10",
                     "200epoch", "B128", "mỗi5epoch", "Clean B32 CUDA smoke2updates", "tối đa300tọa độ",
                     "8/255", "lr.001", "official TEST", "không phải mean ± SD",
                     "giữ SSH mở", "Chưa có kết quả mới trên full MM-Fi/CUDA4090", "LADDER", "BLOCKED"):
        assert required.casefold() in doc


def test_document_backtime_requires_explicit_policy_and_disclosure():
    doc = " ".join(DOC.read_text(encoding="utf-8").split()).casefold()
    for required in ("-BackTimePolicy adaptive", "-MethodSet SibaBackTime", "Không âm thầm bỏ LADDER",
                     "surrogate riêng từ scratch100epoch/B64", "warmup epoch0..10",
                     "error-driven selection", "online per-sample SmoothL1", "uniform variant bỏ khỏi plan",
                     "historical anchor", "actual residual Linf", "mọi mode",
                     "Không thay warmup bằng checkpoint Clean50", "thiếu author trigger-search source/artifact"):
        assert required.casefold() in doc


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
def test_powershell_parses_without_execution():
    command = (
        f"$t=$null; $e=$null; [System.Management.Automation.Language.Parser]::ParseFile({literal(HELPER)},"
        "[ref]$t,[ref]$e)|Out-Null; if($e.Count){$e|Out-String|Write-Output;exit 1}"
    )
    result = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("policy,method_set", [(None, "Requested"), ("adaptive", "Requested"),
                                             (None, "SibaBackTime"), ("adaptive", "SibaBackTime")])
@pytest.mark.parametrize("mode,expected", [
    ("Inspect", ["inspect"]), ("Preflight", ["preflight"]),
    ("Prepare", ["prepare"]), ("CheckOnly", ["check"]),
    ("Status", ["status"]), ("Launch", ["check", "launch"]),
])
def test_modes_forward_exactly_and_paths_with_spaces(tmp_path, mock_helper, mode, expected, policy, method_set):
    capture = tmp_path / "calls.jsonl"
    stub = tmp_path / "stub python.ps1"
    home = tmp_path / "data home"
    home.mkdir()
    stub.write_text(
        "[IO.File]::AppendAllText($env:SIBA_LADDER_TEST_CAPTURE,"
        "(ConvertTo-Json -InputObject @($args) -Compress)+[Environment]::NewLine);"
        "$global:LASTEXITCODE=0", encoding="utf-8",
    )
    command = (
        "$ErrorActionPreference='Stop'; "
        f"$env:SIBA_LADDER_TEST_CAPTURE={literal(capture)}; "
        f"& {literal(mock_helper)} -Mode {mode} -DataHome {literal(home)} -PythonExe {literal(stub)} "
        f"-MethodSet {method_set} " + (f"-BackTimePolicy {policy}" if policy else "")
    )
    result = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-Command", command], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
    assert [args[args.index("--mode") + 1] for args in calls] == expected
    for args in calls:
        assert args[args.index("--data-home") + 1] == str(home)
        assert args[args.index("--device") + 1] == "cuda:0"
        assert args[args.index("--jobs-per-gpu") + 1] == "1"
        includes_backtime = policy is not None or method_set == "SibaBackTime"
        expected_run = ("mmfi_p3s3_siba_ladder_backtime_adaptive_source_adapted_s42_4090_v2" if includes_backtime
                        else "mmfi_p3s3_siba_ladder_source_adapted_s42_4090_v2")
        if method_set == "SibaBackTime":
            expected_run = expected_run.replace("_s42_", "_siba_backtime_only_s42_")
        assert Path(args[args.index("--outdir") + 1]).name == expected_run
        assert args.count("--backtime-policy") == int(includes_backtime)
        if includes_backtime:
            assert args[args.index("--backtime-policy") + 1] == "adaptive"
        assert args.count("--methods") == int(method_set == "SibaBackTime")
        if method_set == "SibaBackTime":
            idx = args.index("--methods")
            assert args[idx + 1:idx + 3] == ["siba", "backtime"]
        assert "--clean-checkpoint" not in args
    assert not (home / "runs").exists(), "mock wrapper must not create artifacts itself"


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("failure_phase", ["prepare", "check", "launch"])
def test_failures_do_not_start_later_phase(tmp_path, mock_helper, failure_phase):
    capture = tmp_path / "calls.jsonl"
    stub = tmp_path / "fail stub.ps1"
    stub.write_text(
        "[IO.File]::AppendAllText($env:SIBA_LADDER_TEST_CAPTURE,"
        "(ConvertTo-Json -InputObject @($args) -Compress)+[Environment]::NewLine);"
        "$phase=$args[[array]::IndexOf($args,'--mode')+1];"
        "if($phase -eq $env:SIBA_LADDER_FAIL_PHASE){$global:LASTEXITCODE=7}"
        "else{$global:LASTEXITCODE=0}", encoding="utf-8",
    )
    mode = "Prepare" if failure_phase == "prepare" else "Launch"
    command = (
        "$ErrorActionPreference='Stop'; "
        f"$env:SIBA_LADDER_TEST_CAPTURE={literal(capture)}; "
        f"$env:SIBA_LADDER_FAIL_PHASE={literal(failure_phase)}; "
        f"& {literal(mock_helper)} -Mode {mode} -DataHome {literal(tmp_path)} -PythonExe {literal(stub)}"
    )
    result = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-Command", command], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    calls = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
    phases = [args[args.index("--mode") + 1] for args in calls]
    assert phases == {"prepare": ["prepare"], "check": ["check"], "launch": ["check", "launch"]}[failure_phase]


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("arguments", ["-JobsPerGpu 2", "-Device cuda:1", "-BackTimePolicy invalid",
                                       "-BackTimePolicy uniform", "-MethodSet invalid"])
def test_other_slot_or_gpu_policy_rejected_before_invocation(tmp_path, arguments):
    capture = tmp_path / "must not exist.jsonl"
    command = f"& {literal(HELPER)} -Mode Inspect {arguments}"
    result = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-Command", command], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert not capture.exists()
