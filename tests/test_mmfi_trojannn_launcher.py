"""Windows wrapper contract tests; no dataset, GPU, training or network calls."""
from pathlib import Path
import json
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "remote_tools/Start-MMFiP1S1TrojanNN.ps1"
DOC = ROOT / "remote_tools/MMFI_P1S1_TROJANNN_VI.md"
PS = shutil.which("powershell") or shutil.which("pwsh")


def test_only_p1s1_and_one_runner_are_scheduled():
    source = HELPER.read_text(encoding="utf-8")
    assert "ATKBackd\\run_mmfi_p1s1_trojannn.py" in source
    assert "mmfi_p1s1_hpeli_trojannn_s42_v1" in source
    assert source.count("$Child = Start-Process ") == 1
    assert "--jobs-per-gpu" not in source
    assert "--_cell" not in source
    assert "[switch]$AllowConcurrentA1" in source
    assert "[switch]$AllowConcurrentA1 = $true" not in source
    assert "if ($AllowConcurrentA1) { $Common += '--allow-concurrent-a1' }" in source
    assert "mmfi_p1s1_hpeli_trojannn_s42_with_a1_v1" in source
    assert "Stop-Process" not in source
    assert "taskkill" not in source.lower()
    assert "File]::Replace" not in source
    assert "--clean-checkpoint" in source


def test_prepare_never_falls_through_to_launch():
    source = HELPER.read_text(encoding="utf-8")
    prepare = source.split("if ($Mode -eq 'Prepare') {", 1)[1].split("if ($Mode -eq 'CheckOnly')", 1)[0]
    assert "Invoke-Runner 'prepare'" in prepare
    assert "return" in prepare
    assert "Start-Process" not in prepare
    launch = source.split("# The runner rechecks", 1)[1]
    assert launch.index("Invoke-Runner 'check'") < launch.index("Start-Process")


def test_hidden_unique_logging_and_early_exit_are_explicit():
    source = HELPER.read_text(encoding="utf-8")
    assert "-WindowStyle Hidden" in source
    assert "-RedirectStandardOutput $Stdout" in source
    assert "-RedirectStandardError $Stderr" in source
    assert "[guid]::NewGuid()" in source
    assert '"controller.$Stamp.stdout.log"' in source
    assert '"controller.$Stamp.stderr.log"' in source
    assert "$Child.WaitForExit(1500)" in source
    assert "A live PID is not proof" in source


def test_matching_clean_default_and_no_installs():
    source = HELPER.read_text(encoding="utf-8")
    assert "mmfi_three_tables_4090_win_fresh_v1\\seed42\\clean\\checkpoint.pt" in source
    assert "envs\\dose-backdoor\\python.exe" in source
    assert "-LiteralPath $CleanCheckpoint -PathType Leaf" in source
    assert "Outdir must be a separate run folder" in source
    assert "pip install" not in source
    assert "Invoke-WebRequest" not in source
    assert "Remove-Item" not in source


def test_docs_do_not_claim_cuda_or_scientific_results():
    text = DOC.read_text(encoding="utf-8")
    for required in ("P1S1", "không chạy P3S3", "model-informed", "từ khởi tạo mới",
                     "không fine-tune/warm-start", "chưa train full victim", "official TEST",
                     "độ lệch chuẩn", "không thay cho audit"):
        assert required.casefold() in text.casefold()


def test_docs_use_separate_checkout_and_explicit_peer_flag():
    text = " ".join(DOC.read_text(encoding="utf-8").split())
    for required in (
        r"C:\Users\Van\code_trojannn", "-AllowConcurrentA1",
        "mmfi_p1s1_hpeli_trojannn_s42_with_a1_v1", "epoch27/49",
        "không chứng minh", "không tự khởi động lại A1",
        "không bảo đảm sống sau khi ngắt SSH", "không bảo đảm vừa VRAM",
    ):
        assert required.casefold() in text.casefold()
    assert "git -C $Repo pull --ff-only" in text
    assert "git clone https://github.com/toan31072006/backdoor-analog.git $Repo" in text


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("mode,expected_phases", [
    ("Prepare", ["prepare"]), ("CheckOnly", ["check"]),
    ("Status", ["status"]), ("Export", ["export"]),
    ("Launch", ["check"]),
])
@pytest.mark.parametrize("allow", [False, True])
def test_opt_in_and_default_forwarding_with_mock_worker(tmp_path, mode, expected_phases, allow):
    """Execute wrapper with a harmless PS stub, never Python/CUDA/training."""
    capture = tmp_path / "calls.jsonl"
    launch_capture = tmp_path / "child.json"
    stub = tmp_path / "stub-python.ps1"
    clean = tmp_path / "clean.pt"
    clean.write_text("mock-checkpoint-not-loaded", encoding="utf-8")
    stub.write_text(
        '[IO.File]::AppendAllText($env:TROJANNN_TEST_CAPTURE, '
        '(ConvertTo-Json -InputObject @($args) -Compress)+[Environment]::NewLine); '
        '$global:LASTEXITCODE=0', encoding="utf-8"
    )
    literal = lambda value: "'" + str(value).replace("'", "''") + "'"
    mock = (
        "function Start-Process { param($FilePath,$ArgumentList,$WorkingDirectory,"
        "$WindowStyle,$RedirectStandardOutput,$RedirectStandardError,[switch]$PassThru); "
        "[IO.File]::WriteAllText($env:TROJANNN_TEST_CHILD,"
        "(ConvertTo-Json -InputObject @{Arguments=$ArgumentList;FilePath=$FilePath} -Compress)); "
        "$child=[pscustomobject]@{Id=42}; "
        "$child|Add-Member -MemberType ScriptMethod -Name WaitForExit -Value {param($ms) $false}; "
        "return $child }; "
    )
    command = (
        "$ErrorActionPreference='Stop'; "
        f"$env:TROJANNN_TEST_CAPTURE={literal(capture)}; "
        f"$env:TROJANNN_TEST_CHILD={literal(launch_capture)}; " + mock +
        f"& {literal(HELPER)} -Mode {mode} -DataHome {literal(tmp_path)} "
        f"-PythonExe {literal(stub)} -CleanCheckpoint {literal(clean)} " +
        ("-AllowConcurrentA1" if allow else "")
    )
    result = subprocess.run(
        [PS, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
    assert [args[args.index("--mode") + 1] for args in calls] == expected_phases
    expected_run = ("mmfi_p1s1_hpeli_trojannn_s42_with_a1_v1" if allow
                    else "mmfi_p1s1_hpeli_trojannn_s42_v1")
    for args in calls:
        assert args.count("--allow-concurrent-a1") == int(allow)
        assert Path(args[args.index("--outdir") + 1]).name == expected_run
        assert args[args.index("--device") + 1] == "cuda:0"
    if mode == "Launch":
        child = json.loads(launch_capture.read_text(encoding="utf-8"))
        assert child["Arguments"].count('"--allow-concurrent-a1"') == int(allow)
        assert '"--mode" "launch"' in child["Arguments"]
        assert expected_run in child["Arguments"]
    else:
        assert not launch_capture.exists()


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
def test_powershell_parses_without_execution():
    path = str(HELPER).replace("'", "''")
    command = (f"$t=$null; $e=$null; [System.Management.Automation.Language.Parser]::ParseFile('{path}',"
               "[ref]$t,[ref]$e)|Out-Null; if($e.Count){$e|Out-String|Write-Output;exit 1}")
    result = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
@pytest.mark.parametrize("value,expected", [
    ("plain", '"plain"'),
    ("C:\\path with spaces\\file.py", '"C:\\path with spaces\\file.py"'),
    ("C:\\", '"C:\\\\"'),
    ('a"b', '"a\\"b"'),
    ("", '""'),
])
def test_native_argument_quotes_backslashes_and_spaces(value, expected):
    path = str(HELPER).replace("'", "''")
    literal = value.replace("'", "''")
    command = (
        f"$t=$null; $e=$null; $a=[System.Management.Automation.Language.Parser]::ParseFile('{path}',"
        "[ref]$t,[ref]$e); $f=$a.FindAll({param($n) $n -is "
        "[System.Management.Automation.Language.FunctionDefinitionAst] -and "
        "$n.Name -eq 'ConvertTo-NativeArgument'},$true); "
        "Invoke-Expression $f[0].Extent.Text; "
        f"ConvertTo-NativeArgument '{literal}' | ConvertTo-Json -Compress"
    )
    result = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip()) == expected
