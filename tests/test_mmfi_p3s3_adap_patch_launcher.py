"""PowerShell handoff tests; no network, dataset, CUDA or victim training."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'remote_tools/Start-MMFiP3S3AdapPatch4090.ps1'
DOC = ROOT / 'remote_tools/MMFI_P3S3_ADAP_PATCH_4090_VI.md'
PS = shutil.which('powershell') or shutil.which('pwsh')


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def ps_run(command):
    return subprocess.run(
        [PS, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
         '-Command', command], capture_output=True, text=True, timeout=45,
    )


def mock_project(tmp_path):
    repo = tmp_path / 'separate code'
    (repo / 'remote_tools').mkdir(parents=True)
    (repo / 'ATKBackd').mkdir()
    helper = repo / 'remote_tools' / HELPER.name
    shutil.copyfile(HELPER, helper)
    (repo / 'ATKBackd/run_mmfi_p3s3_adap_patch_4090.py').write_text('# stub', encoding='utf-8')
    data = tmp_path / 'existing data'
    data.mkdir()
    capture = tmp_path / 'calls.jsonl'
    stub = tmp_path / 'stub-python.ps1'
    stub.write_text(
        '[IO.File]::AppendAllText($env:ADAPPATCH_TEST_CALLS, '
        '(ConvertTo-Json -InputObject @($args) -Compress)+[Environment]::NewLine); '
        '$phase=$args[([array]::IndexOf($args,"--mode"))+1]; '
        'if ($phase -eq $env:ADAPPATCH_TEST_FAIL) {$global:LASTEXITCODE=23} '
        'else {$global:LASTEXITCODE=0}; Write-Output "MOCK_PHASE=$phase"',
        encoding='utf-8',
    )
    return helper, data, capture, stub


def calls(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_fixed_single_worker_scope_and_no_system_mutations():
    source = HELPER.read_text(encoding='utf-8')
    for value in (
        "[string]$Mode = 'Status'", "[ValidateSet('cuda:0')]",
        'ATKBackd\\run_mmfi_p3s3_adap_patch_4090.py',
        'mmfi_p3s3_hpeli_adap_patch_s42_4090_v1',
        "'--jobs-per-gpu','1'", "'--device',$Device",
        'strict descendant of DataHome\\runs',
        '[StringComparison]::OrdinalIgnoreCase',
        'keep this SSH session open', 'Tee-Object -FilePath $Log',
    ):
        assert value in source
    for forbidden in (
        'AllowConcurrent', 'Stop-Process', 'taskkill', 'Remove-Item',
        'Start-Process', 'New-ScheduledTask', 'Invoke-CimMethod',
        'pip install', 'Invoke-WebRequest', 'git clone',
    ):
        assert forbidden not in source


def test_documentation_discloses_adaptation_budget_and_limits():
    text = ' '.join(DOC.read_text(encoding='utf-8').split())
    for value in (
        'P3S3', 'seed42', '240570', '80190', '50 epochs/final49',
        'GTSRB', 'phoenix_corner2_32.png', '48114', '72171', '30%',
        'cover:poison 2:1', 'không phải GTSRB', 'bilinear', 'nearest',
        'không train lại Clean', 'Giữ SSH mở', 'năm dose dương',
        'không phải clean PCK', 'không tự retry', 'equal-modified-data',
        'không có independent clean-trained-model', 'Local tests không thay',
    ):
        assert value.casefold() in text.casefold()


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
    ('Inspect', ['inspect']), ('DryRun', ['dry-run']),
    ('Prepare', ['prepare']), ('CheckOnly', ['check']),
    ('Status', ['status']), ('Launch', ['check', 'launch']),
])
def test_exact_phase_forwarding_and_no_implicit_training(tmp_path, mode, phases):
    helper, data, capture, stub = mock_project(tmp_path)
    result = ps_run(
        f'$env:ADAPPATCH_TEST_CALLS={literal(capture)}; $env:ADAPPATCH_TEST_FAIL=""; '
        f'& {literal(helper)} -Mode {mode} -DataHome {literal(data)} -PythonExe {literal(stub)}'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = calls(capture)
    assert [a[a.index('--mode') + 1] for a in records] == phases
    for args in records:
        assert args[args.index('--jobs-per-gpu') + 1] == '1'
        assert args[args.index('--device') + 1] == 'cuda:0'
        assert Path(args[args.index('--outdir') + 1]) == (
            data / 'runs/mmfi_p3s3_hpeli_adap_patch_s42_4090_v1'
        )
    assert len(list(data.glob('runs/*/logs/controller.*.log'))) == int(mode == 'Launch')
    if mode != 'Launch':
        assert not (data / 'runs').exists()


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('failure,expected', [
    ('prepare', ['prepare']), ('check', ['check']), ('launch', ['check', 'launch']),
])
def test_nonzero_exit_stops_without_retry(tmp_path, failure, expected):
    helper, data, capture, stub = mock_project(tmp_path)
    mode = 'Prepare' if failure == 'prepare' else 'Launch'
    result = ps_run(
        f'$env:ADAPPATCH_TEST_CALLS={literal(capture)}; $env:ADAPPATCH_TEST_FAIL={literal(failure)}; '
        f'& {literal(helper)} -Mode {mode} -DataHome {literal(data)} -PythonExe {literal(stub)}'
    )
    assert result.returncode != 0
    assert [a[a.index('--mode') + 1] for a in calls(capture)] == expected


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('location', ['data', 'runs', 'sibling', 'prefix'])
def test_output_escape_is_blocked_before_python(tmp_path, location):
    helper, data, capture, stub = mock_project(tmp_path)
    targets = {'data': data, 'runs': data / 'runs', 'sibling': tmp_path / 'other',
               'prefix': data / 'runs-other/run'}
    result = ps_run(
        f'$env:ADAPPATCH_TEST_CALLS={literal(capture)}; & {literal(helper)} -Mode Launch '
        f'-DataHome {literal(data)} -PythonExe {literal(stub)} -Outdir {literal(targets[location])}'
    )
    assert result.returncode != 0
    assert not capture.exists()


@pytest.mark.skipif(not PS, reason='PowerShell unavailable')
@pytest.mark.parametrize('option', ['-Device cuda:1', '-JobsPerGpu 2', '-AllowConcurrentA1'])
def test_unsupported_sharing_options_are_rejected(tmp_path, option):
    helper, data, capture, stub = mock_project(tmp_path)
    result = ps_run(
        f'$env:ADAPPATCH_TEST_CALLS={literal(capture)}; & {literal(helper)} -Mode Launch '
        f'-DataHome {literal(data)} -PythonExe {literal(stub)} {option}'
    )
    assert result.returncode != 0
    assert not capture.exists()
