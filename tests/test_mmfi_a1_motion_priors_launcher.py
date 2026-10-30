"""Launcher contracts only: PowerShell stubs, no CUDA, data, or downloads."""
from pathlib import Path
import json
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'remote_tools/Start-MMFiA1MotionPriors4090.ps1'
DOC = ROOT / 'remote_tools/MMFI_A1_MOTION_PRIORS_4090_VI.md'
PS = shutil.which('powershell') or shutil.which('pwsh')


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def ps_run(command):
    return subprocess.run(
        [PS, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', command],
        capture_output=True, text=True, timeout=45,
    )


def mock_project(tmp_path):
    repo = tmp_path / 'code separate'
    (repo / 'remote_tools').mkdir(parents=True)
    (repo / 'ATKBackd').mkdir()
    helper = repo / 'remote_tools' / HELPER.name
    shutil.copyfile(HELPER, helper)
    (repo / 'ATKBackd/run_mmfi_a1_motion_priors_4090.py').write_text('# stub', encoding='utf-8')
    data = tmp_path / 'data existing'
    data.mkdir()
    capture = tmp_path / 'calls.jsonl'
    stub = tmp_path / 'stub-python.ps1'
    stub.write_text(
        '[IO.File]::AppendAllText($env:MOTION_PRIOR_CAPTURE, '
        '(ConvertTo-Json -InputObject @($args) -Compress)+[Environment]::NewLine); '
        '$phase=$args[([array]::IndexOf($args,"--mode"))+1]; '
        'if ($phase -eq $env:MOTION_PRIOR_FAIL_PHASE) {$global:LASTEXITCODE=23} '
        'else {$global:LASTEXITCODE=0}; Write-Output "MOCK_PHASE=$phase"', encoding='utf-8',
    )
    return helper, data, capture, stub


def test_scope_and_no_implicit_mutation():
    source = HELPER.read_text(encoding='utf-8')
    for required in (
        'ATKBackd\\run_mmfi_a1_motion_priors_4090.py',
        'mmfi_p1s1_hpeli_a1_motion_priors_s42_4090_v1',
        'envs\\dose-backdoor\\python.exe', "[string]$Mode = 'Status'",
        "[ValidateSet('cuda:0')]", "@('cross','nod')", '--approve-action-labels',
        "$env:MKL_THREADING_LAYER = 'SEQUENTIAL'", 'strict descendant of DataHome\\runs',
    ):
        assert required in source
    for prohibited in (
        'AllowConcurrent', '--jobs-per-gpu', 'Start-Process', 'Stop-Process', 'taskkill',
        'Remove-Item', 'pip install', 'Invoke-WebRequest', 'New-ScheduledTask', 'Invoke-CimMethod',
    ):
        assert prohibited not in source
    prepare = source.split("if ($Mode -eq 'Prepare') {", 1)[1].split("if ($Mode -eq 'CheckOnly')", 1)[0]
    assert "Invoke-Runner 'prepare'" in prepare
    assert 'return' in prepare
    assert "Invoke-Runner 'launch'" not in prepare
    launch = source.split('# Launch never prepares implicitly', 1)[1]
    assert launch.index("Invoke-Runner 'check'") < launch.index("Invoke-Runner 'launch'")
    assert 'Tee-Object -FilePath $Log' in source
    assert '[guid]::NewGuid()' in source
    audit = source.split("if ($Mode -eq 'AuditOnly') {", 1)[1].split("if ($Mode -eq 'Prepare')", 1)[0]
    assert "Invoke-Runner 'audit'" in audit
    assert 'return' in audit
    assert "Invoke-Runner 'prepare'" not in audit


def test_docs_explicitly_exploratory_and_not_ready_to_share_gpu():
    text = ' '.join(DOC.read_text(encoding='utf-8').split())
    for required in (
        r'C:\Users\Van\code_motion_priors', 'git -C $Repo pull --ff-only',
        'P1S1', '133056', '33264', '50 epochs/final49', 'SGD lr0.001',
        'cross → nod', 'seed42', 'Giữ SSH mở', 'exclusive', 'code_badnets_mmfi',
        'Get-PSBAMotionPriors.py', '441.9 MiB', '-ApproveActionLabels', 'không phải ba',
        '1irgTzLOMkpfOQjNHI9omf56iJIeHq3dk', '1aGItQm8dSMk0FweZQnAdX7IuAaKHQx6A',
        'target-action scenarios', '[2,3]', 'unavailable', 'mean +/- độ lệch chuẩn',
        'Local tests không thay cho audit', 'không claim equal-noise', 'AuditOnly',
        'bendSideways', 'crossHandsFront', 'nodHead', 'semantic_approved:true',
    ):
        assert required.casefold() in text.casefold()


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
def test_powershell_parses_without_execution():
    result = ps_run(
        '$t=$null; $e=$null; [System.Management.Automation.Language.Parser]::ParseFile('
        f'{literal(HELPER)},[ref]$t,[ref]$e)|Out-Null; '
        'if($e.Count){$e|Out-String|Write-Output;exit 1}'
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('mode,phases', [
    ('AuditOnly', ['audit']), ('Prepare', ['prepare']), ('CheckOnly', ['check']), ('Status', ['status']),
    ('Export', ['export']), ('Launch', ['check', 'launch']),
])
def test_foreground_phases_and_defaults(tmp_path, mode, phases):
    helper, data, capture, stub = mock_project(tmp_path)
    result = ps_run(
        f'$env:MOTION_PRIOR_CAPTURE={literal(capture)}; $env:MOTION_PRIOR_FAIL_PHASE=""; '
        f'& {literal(helper)} -Mode {mode} -DataHome {literal(data)} -PythonExe {literal(stub)}'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text(encoding='utf-8').splitlines()]
    assert [args[args.index('--mode') + 1] for args in calls] == phases
    for args in calls:
        assert args[args.index('--device') + 1] == 'cuda:0'
        assert Path(args[args.index('--outdir') + 1]) == data / 'runs/mmfi_p1s1_hpeli_a1_motion_priors_s42_4090_v1'
        assert args[args.index('--data-home') + 1] == str(data)
        start = args.index('--priors')
        assert args[start + 1:start + 3] == ['cross', 'nod']
        assert '--approve-action-labels' not in args
    logs = list(data.glob('runs/*/logs/controller.*.log'))
    assert len(logs) == int(mode == 'Launch')
    if logs:
        raw = logs[0].read_bytes()
        decoded = raw.decode('utf-16') if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else raw.decode('utf-8-sig')
        assert 'MOCK_PHASE=launch' in decoded


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
def test_action_paths_priors_and_semantic_attestation_are_forwarded(tmp_path):
    helper, data, capture, stub = mock_project(tmp_path)
    action = tmp_path / 'motions quoted space'
    action.mkdir()
    result = ps_run(
        f'$env:MOTION_PRIOR_CAPTURE={literal(capture)}; & {literal(helper)} -Mode Prepare '
        f'-DataHome {literal(data)} -PythonExe {literal(stub)} '
        f'-CrossAction {literal(action / "data_cross.npy")} '
        f'-NodAction {literal(action / "data_nod.npy")} '
        f'-BendAction {literal(action / "data_bend.npy")} '
        '-Priors @("bend","nod") -ApproveActionLabels'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(capture.read_text(encoding='utf-8').strip())
    for label in ('cross', 'nod', 'bend'):
        assert Path(args[args.index(f'--{label}-action') + 1]) == action / f'data_{label}.npy'
    assert '--approve-action-labels' in args
    start = args.index('--priors')
    assert args[start + 1:start + 3] == ['bend', 'nod']


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('failure,expected', [
    ('prepare', ['prepare']), ('check', ['check']), ('launch', ['check', 'launch']),
])
def test_nonzero_exit_stops_without_retry(tmp_path, failure, expected):
    helper, data, capture, stub = mock_project(tmp_path)
    mode = 'Prepare' if failure == 'prepare' else 'Launch'
    result = ps_run(
        f'$env:MOTION_PRIOR_CAPTURE={literal(capture)}; $env:MOTION_PRIOR_FAIL_PHASE={literal(failure)}; '
        f'& {literal(helper)} -Mode {mode} -DataHome {literal(data)} -PythonExe {literal(stub)}'
    )
    assert result.returncode != 0
    calls = [json.loads(line) for line in capture.read_text(encoding='utf-8').splitlines()]
    assert [args[args.index('--mode') + 1] for args in calls] == expected


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('outside', ['data', 'runs', 'sibling', 'prefix'])
def test_outdir_escape_blocked_before_worker(tmp_path, outside):
    helper, data, capture, stub = mock_project(tmp_path)
    targets = {'data': data, 'runs': data / 'runs', 'sibling': tmp_path / 'outside', 'prefix': data / 'runs-other/run'}
    result = ps_run(
        f'$env:MOTION_PRIOR_CAPTURE={literal(capture)}; & {literal(helper)} -Mode Launch '
        f'-DataHome {literal(data)} -PythonExe {literal(stub)} -Outdir {literal(targets[outside])}'
    )
    assert result.returncode != 0
    assert not capture.exists()


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('options', [
    '-Device cuda:1', '-AllowConcurrentBadNets', '-JobsPerGpu 2',
    '-Priors @("cross","cross")', '-Priors @("other")',
])
def test_sharing_unknown_priors_and_duplicate_priors_rejected(tmp_path, options):
    helper, data, capture, stub = mock_project(tmp_path)
    result = ps_run(
        f'$env:MOTION_PRIOR_CAPTURE={literal(capture)}; & {literal(helper)} -Mode Launch '
        f'-DataHome {literal(data)} -PythonExe {literal(stub)} {options}'
    )
    assert result.returncode != 0
    assert not capture.exists()
