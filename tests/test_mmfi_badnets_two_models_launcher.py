"""PowerShell launcher contract; no dataset, CUDA, training, or network."""
from pathlib import Path
import json
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "remote_tools/Start-MMFiBadNetsTwoModels4090.ps1"
DOC = ROOT / "remote_tools/MMFI_BADNETS_TWO_MODELS_4090_VI.md"
PS = shutil.which("powershell") or shutil.which("pwsh")


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def ps_run(command):
    return subprocess.run(
        [PS, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True, text=True, timeout=45,
    )


def mock_project(tmp_path):
    repo = tmp_path / "code separate"
    (repo / "remote_tools").mkdir(parents=True)
    (repo / "ATKBackd").mkdir()
    helper = repo / "remote_tools" / HELPER.name
    shutil.copyfile(HELPER, helper)
    (repo / "ATKBackd/run_mmfi_badnets_two_models_4090.py").write_text("# stub", encoding="utf-8")
    data = tmp_path / "data existing"
    data.mkdir()
    capture = tmp_path / "calls.jsonl"
    stub = tmp_path / "stub-python.ps1"
    stub.write_text(
        '[IO.File]::AppendAllText($env:BADNETS_CAPTURE, '
        '(ConvertTo-Json -InputObject @($args) -Compress)+[Environment]::NewLine); '
        '$phase=$args[([array]::IndexOf($args,"--mode"))+1]; '
        'if ($phase -eq $env:BADNETS_FAIL_PHASE) {$global:LASTEXITCODE=23} '
        'else {$global:LASTEXITCODE=0}; Write-Output "MOCK_PHASE=$phase"', encoding="utf-8",
    )
    return helper, data, capture, stub


def test_fixed_scope_and_no_concurrency_override():
    source = HELPER.read_text(encoding="utf-8")
    for required in (
        "ATKBackd\\run_mmfi_badnets_two_models_4090.py",
        "mmfi_p1s1_badnets_wiflow_metafi_s42_4090_v1",
        "envs\\dose-backdoor\\python.exe",
        "[ValidateSet('cuda:0')]", "[string]$Mode = 'Status'",
        "[StringComparison]::OrdinalIgnoreCase", "strict descendant of DataHome\\runs",
        "$env:MKL_THREADING_LAYER = 'SEQUENTIAL'",
    ):
        assert required in source
    for prohibited in (
        "AllowConcurrent", "--jobs-per-gpu", "--method", "Start-Process",
        "Stop-Process", "taskkill", "Remove-Item", "pip install", "Invoke-WebRequest",
        "New-ScheduledTask", "Invoke-CimMethod",
    ):
        assert prohibited not in source


def test_prepare_does_not_fall_through_to_launch():
    source = HELPER.read_text(encoding="utf-8")
    branch = source.split("if ($Mode -eq 'Prepare') {", 1)[1].split("if ($Mode -eq 'CheckOnly')", 1)[0]
    assert "Invoke-Runner 'prepare'" in branch
    assert "return" in branch
    assert "Invoke-Runner 'launch'" not in branch
    launch = source.split("# Launch does not prepare implicitly.", 1)[1]
    assert launch.index("Invoke-Runner 'check'") < launch.index("Invoke-Runner 'launch'")
    assert "Tee-Object -FilePath $Log" in source
    assert "[guid]::NewGuid()" in source
    assert "keep this SSH session open" in source


def test_docs_match_scope_and_limitations():
    text = " ".join(DOC.read_text(encoding="utf-8").split())
    for required in (
        r"C:\Users\Van\code_badnets_mmfi", "git -C $Repo pull --ff-only",
        "P1S1", "133056", "33264", "50 epochs/final49", "SGD",
        "0.01", "AdamW", "0.0001", "2 / 64", "seed42", "Giữ SSH mở",
        "không tự thêm hai job Clean", "unavailable", "không phải toàn bộ T1",
        "Local tests không thay cho audit", "mean +/- độ lệch chuẩn", "[2,3]",
    ):
        assert required.casefold() in text.casefold()


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
def test_powershell_parses_without_execution():
    result = ps_run(
        "$t=$null; $e=$null; [System.Management.Automation.Language.Parser]::ParseFile("
        f"{literal(HELPER)},[ref]$t,[ref]$e)|Out-Null; "
        "if($e.Count){$e|Out-String|Write-Output;exit 1}"
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("mode,phases", [
    ("Prepare", ["prepare"]), ("CheckOnly", ["check"]),
    ("Status", ["status"]), ("Export", ["export"]),
    ("Launch", ["check", "launch"]),
])
def test_foreground_phase_forwarding_and_default_paths(tmp_path, mode, phases):
    helper, data, capture, stub = mock_project(tmp_path)
    result = ps_run(
        f"$env:BADNETS_CAPTURE={literal(capture)}; $env:BADNETS_FAIL_PHASE=''; "
        f"& {literal(helper)} -Mode {mode} -DataHome {literal(data)} -PythonExe {literal(stub)}"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
    assert [args[args.index("--mode") + 1] for args in calls] == phases
    for args in calls:
        assert args[args.index("--device") + 1] == "cuda:0"
        assert Path(args[args.index("--outdir") + 1]) == (
            data / "runs/mmfi_p1s1_badnets_wiflow_metafi_s42_4090_v1"
        )
        assert args[args.index("--data-home") + 1] == str(data)
    controller_logs = list(data.glob("runs/*/logs/controller.*.log"))
    assert len(controller_logs) == int(mode == "Launch")
    if mode == "Launch":
        raw = controller_logs[0].read_bytes()
        decoded = raw.decode("utf-16") if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else raw.decode("utf-8-sig")
        assert "MOCK_PHASE=launch" in decoded


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("failure,expected", [("prepare", ["prepare"]), ("check", ["check"]),
                                              ("launch", ["check", "launch"])])
def test_nonzero_exit_stops_without_retry(tmp_path, failure, expected):
    helper, data, capture, stub = mock_project(tmp_path)
    mode = "Prepare" if failure == "prepare" else "Launch"
    result = ps_run(
        f"$env:BADNETS_CAPTURE={literal(capture)}; $env:BADNETS_FAIL_PHASE={literal(failure)}; "
        f"& {literal(helper)} -Mode {mode} -DataHome {literal(data)} -PythonExe {literal(stub)}"
    )
    assert result.returncode != 0
    calls = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
    assert [args[args.index("--mode") + 1] for args in calls] == expected


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("outside", ["data", "runs", "sibling", "prefix"])
def test_outdir_escape_is_blocked_before_worker(tmp_path, outside):
    helper, data, capture, stub = mock_project(tmp_path)
    targets = {"data": data, "runs": data / "runs", "sibling": tmp_path / "outside",
               "prefix": data / "runs-other/run"}
    result = ps_run(
        f"$env:BADNETS_CAPTURE={literal(capture)}; & {literal(helper)} -Mode Launch "
        f"-DataHome {literal(data)} -PythonExe {literal(stub)} -Outdir {literal(targets[outside])}"
    )
    assert result.returncode != 0
    assert not capture.exists()


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("bad_option", ["-Device cuda:1", "-AllowConcurrentA1", "-JobsPerGpu 2"])
def test_unknown_sharing_options_are_rejected(tmp_path, bad_option):
    helper, data, capture, stub = mock_project(tmp_path)
    result = ps_run(
        f"$env:BADNETS_CAPTURE={literal(capture)}; & {literal(helper)} -Mode Launch "
        f"-DataHome {literal(data)} -PythonExe {literal(stub)} {bad_option}"
    )
    assert result.returncode != 0
    assert not capture.exists()
