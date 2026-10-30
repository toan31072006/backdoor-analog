"""Shared launcher contracts using PowerShell stubs; never CUDA/training."""
from pathlib import Path
import json
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'remote_tools/Start-MMFiA1BadNetsShared4090.ps1'
DOC = ROOT / 'remote_tools/MMFI_A1_BADNETS_SHARED_4090_VI.md'
PS = shutil.which('powershell') or shutil.which('pwsh')


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def ps_run(command):
    return subprocess.run(
        [PS, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', command],
        capture_output=True, text=True, timeout=45,
    )


def mock_project(tmp_path):
    repo = tmp_path / 'motion code separate'
    (repo / 'remote_tools/motion_prior_shared_4090').mkdir(parents=True)
    helper = repo / 'remote_tools' / HELPER.name
    shutil.copyfile(HELPER, helper)
    (repo / 'remote_tools/motion_prior_shared_4090/controller.py').write_text('# stub', encoding='utf-8')
    data = tmp_path / 'data existing'
    data.mkdir()
    old_repo = tmp_path / 'badnets code frozen'
    old_repo.mkdir()
    capture = tmp_path / 'calls.jsonl'
    stub = tmp_path / 'stub-python.ps1'
    stub.write_text(
        '[IO.File]::AppendAllText($env:SHARED_MOTION_CAPTURE, '
        '(ConvertTo-Json -InputObject @($args) -Compress)+[Environment]::NewLine); '
        '$phase=$args[([array]::IndexOf($args,"--mode"))+1]; '
        'if ($phase -eq $env:SHARED_MOTION_FAIL_PHASE) {$global:LASTEXITCODE=23} '
        'else {$global:LASTEXITCODE=0}; Write-Output "MOCK_PHASE=$phase"', encoding='utf-8',
    )
    return helper, repo, old_repo, data, capture, stub


def test_explicit_scope_and_no_implicit_mutation():
    source = HELPER.read_text(encoding='utf-8')
    for required in (
        "[string]$Mode = 'Status'", 'remote_tools\\motion_prior_shared_4090\\controller.py',
        'mmfi_a1_badnets_shared_v1', '--allow-concurrent-badnets', '--badnets-repo',
        '--a1-repo', '--home', '--root', 'envs\\dose-backdoor\\python.exe',
        'strict descendant of DataHome\\tools', 'outside both scientific source checkouts',
        "$Mode -ne 'Status' -and !$AllowConcurrentBadNets", 'Tee-Object -FilePath $Log',
        '[guid]::NewGuid()', "$env:MKL_THREADING_LAYER = 'SEQUENTIAL'",
    ):
        assert required in source
    for prohibited in (
        'Start-Process', 'Stop-Process', 'taskkill', 'Remove-Item', 'Invoke-CimMethod',
        'New-ScheduledTask', 'pip install', 'Invoke-WebRequest', 'git pull', '--jobs-per-gpu',
    ):
        assert prohibited not in source
    assert source.index("if ($Mode -eq 'Launch') { Invoke-Runner 'check' }") < source.index('Invoke-Runner $Phase $ControllerLog')
    status = source.split("if ($Mode -eq 'Status') {", 1)[1].split("if ($Mode -eq 'CheckOnly')", 1)[0]
    assert "Invoke-Runner 'status'; return" in status


def test_docs_require_user_pause_and_frozen_checkout():
    text = ' '.join(DOC.read_text(encoding='utf-8').split())
    for required in (
        'Ctrl+C', '[ckpt] saved', 'không kill process', 'không còn dòng nào',
        'không tự quay về epoch0', 'optimizer/RNG', r'C:\Users\Van\code_badnets_mmfi',
        r'C:\Users\Van\code_motion_priors', 'git -C $Repo pull --ff-only',
        '-AllowConcurrentBadNets', 'không train full victim', 'Giữ SSH mở',
        'gate cũ chỉ chứng nhận context cũ', 'không bảo đảm',
        'một A1', 'một BadNets', 'Local CPU/stub tests không thay',
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
    ('Prepare', ['prepare']), ('CheckOnly', ['check']), ('Launch', ['check', 'launch']), ('Status', ['status']),
])
def test_phases_and_fixed_paths(tmp_path, mode, phases):
    helper, repo, old_repo, data, capture, stub = mock_project(tmp_path)
    approval = '' if mode == 'Status' else '-AllowConcurrentBadNets'
    result = ps_run(
        f'$env:SHARED_MOTION_CAPTURE={literal(capture)}; $env:SHARED_MOTION_FAIL_PHASE=""; '
        f'& {literal(helper)} -Mode {mode} -DataHome {literal(data)} '
        f'-BadNetsRepo {literal(old_repo)} -PythonExe {literal(stub)} {approval}'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text(encoding='utf-8').splitlines()]
    assert [args[args.index('--mode') + 1] for args in calls] == phases
    for args in calls:
        assert Path(args[args.index('--a1-repo') + 1]) == repo
        assert Path(args[args.index('--badnets-repo') + 1]) == old_repo
        assert Path(args[args.index('--root') + 1]) == data / 'tools/mmfi_a1_badnets_shared_v1'
        assert Path(args[args.index('--home') + 1]) == data
        assert ('--allow-concurrent-badnets' in args) == (mode != 'Status')
    logs = list(data.glob('tools/*/logs/controller.*.log'))
    assert len(logs) == int(mode in ('Prepare', 'Launch'))
    if logs:
        raw = logs[0].read_bytes()
        decoded = raw.decode('utf-16') if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else raw.decode('utf-8-sig')
        assert f'MOCK_PHASE={mode.lower()}' in decoded
    if mode == 'Status':
        assert not (data / 'tools').exists()


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('mode', ['Prepare', 'CheckOnly', 'Launch'])
def test_no_approval_no_invocation_or_directory_write(tmp_path, mode):
    helper, repo, old_repo, data, capture, stub = mock_project(tmp_path)
    result = ps_run(
        f'$env:SHARED_MOTION_CAPTURE={literal(capture)}; & {literal(helper)} -Mode {mode} '
        f'-DataHome {literal(data)} -BadNetsRepo {literal(old_repo)} -PythonExe {literal(stub)}'
    )
    assert result.returncode != 0
    assert not capture.exists()
    assert not (data / 'tools').exists()


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('failure,expected', [
    ('prepare', ['prepare']), ('check', ['check']), ('launch', ['check', 'launch']),
])
def test_failure_no_retry_or_fallthrough(tmp_path, failure, expected):
    helper, repo, old_repo, data, capture, stub = mock_project(tmp_path)
    mode = 'Prepare' if failure == 'prepare' else 'Launch'
    result = ps_run(
        f'$env:SHARED_MOTION_CAPTURE={literal(capture)}; $env:SHARED_MOTION_FAIL_PHASE={literal(failure)}; '
        f'& {literal(helper)} -Mode {mode} -DataHome {literal(data)} '
        f'-BadNetsRepo {literal(old_repo)} -PythonExe {literal(stub)} -AllowConcurrentBadNets'
    )
    assert result.returncode != 0
    calls = [json.loads(line) for line in capture.read_text(encoding='utf-8').splitlines()]
    assert [args[args.index('--mode') + 1] for args in calls] == expected


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('outside', ['data', 'tools', 'sibling', 'prefix'])
def test_shared_root_escape_blocked(tmp_path, outside):
    helper, repo, old_repo, data, capture, stub = mock_project(tmp_path)
    targets = {'data': data, 'tools': data / 'tools', 'sibling': tmp_path / 'outside', 'prefix': data / 'tools-other/shared'}
    result = ps_run(
        f'$env:SHARED_MOTION_CAPTURE={literal(capture)}; & {literal(helper)} -Mode Launch '
        f'-DataHome {literal(data)} -BadNetsRepo {literal(old_repo)} -PythonExe {literal(stub)} '
        f'-SharedDir {literal(targets[outside])} -AllowConcurrentBadNets'
    )
    assert result.returncode != 0
    assert not capture.exists()


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('which_repo', ['motion', 'badnets'])
def test_shared_logs_cannot_modify_source_tree(tmp_path, which_repo):
    helper, repo, old_repo, data, capture, stub = mock_project(tmp_path)
    # This setup deliberately locates a source checkout under DataHome/tools.
    source = data / 'tools/source'
    source.mkdir(parents=True)
    replacement = repo if which_repo == 'badnets' else source
    badnets = source if which_repo == 'badnets' else old_repo
    result = ps_run(
        f'$env:SHARED_MOTION_CAPTURE={literal(capture)}; & {literal(helper)} -Mode Launch '
        f'-DataHome {literal(data)} -MotionRepo {literal(replacement)} '
        f'-BadNetsRepo {literal(badnets)} -PythonExe {literal(stub)} '
        f'-SharedDir {literal(source / "shared")} -AllowConcurrentBadNets'
    )
    assert result.returncode != 0
    assert not capture.exists()


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
def test_same_checkout_is_rejected(tmp_path):
    helper, repo, old_repo, data, capture, stub = mock_project(tmp_path)
    result = ps_run(
        f'$env:SHARED_MOTION_CAPTURE={literal(capture)}; & {literal(helper)} -Mode Prepare '
        f'-DataHome {literal(data)} -BadNetsRepo {literal(repo)} -PythonExe {literal(stub)} -AllowConcurrentBadNets'
    )
    assert result.returncode != 0
    assert not capture.exists()
