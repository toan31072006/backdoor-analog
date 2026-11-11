"""Synthetic Windows/NVIDIA fixtures; no CUDA job or remote process mutation."""
import copy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from bend_gpu_guard import desktop_exemption, parse_gpu_xml, policy
import run_bend_forward as runner


DESKTOPS = [
    ('explorer.exe', r'C:\Windows\Explorer.exe'),
    ('dwm.exe', r'C:\Windows\System32\dwm.exe'),
    ('ShellHost.exe', r'C:\Windows\System32\ShellHost.exe'),
    ('ApplicationFrameHost.exe', r'C:\Windows\System32\ApplicationFrameHost.exe'),
    ('LogonUI.exe', r'C:\Windows\System32\LogonUI.exe'),
    *[(name, 'C:\\Windows\\SystemApps\\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\\'+name)
      for name in ('CrossDeviceResume.exe', 'SearchHost.exe', 'TextInputHost.exe', 'ShellExperienceHost.exe')],
    ('StartMenuExperienceHost.exe', r'C:\Windows\SystemApps\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\StartMenuExperienceHost.exe'),
    ('LockApp.exe', r'C:\Windows\SystemApps\Microsoft.LockApp_cw5n1h2txyewy\LockApp.exe'),
    ('SystemSettings.exe', r'C:\Windows\ImmersiveControlPanel\SystemSettings.exe'),
    ('Code.exe', r'C:\Program Files\Microsoft VS Code\Code.exe'),
    ('TeamViewer.exe', r'C:\Program Files\TeamViewer\TeamViewer.exe'),
    ('msedge.exe', r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'),
    ('msedgewebview2.exe', r'C:\Program Files (x86)\Microsoft\EdgeWebView\Application\157.0.4341.42\msedgewebview2.exe'),
    ('copilotapp.exe', r'C:\Program Files (x86)\Microsoft\Copilot\Application\copilotapp.exe'),
    ('copilotapphost.exe', r'C:\Program Files (x86)\Microsoft\Copilot\Application\undocked_copilot\copilotapphost.exe'),
    *[(exe, f'C:\\Program Files\\WindowsApps\\{family}_1.2.3.0_x64__8wekyb3d8bbwe\\{exe}')
      for exe, family in [('WindowsTerminal.exe', 'Microsoft.WindowsTerminal'),
        ('SnippingTool.exe', 'Microsoft.ScreenSketch'), ('ms-teams.exe', 'MSTeams'),
        ('PhoneExperienceHost.exe', 'Microsoft.YourPhone'),
        ('Microsoft.Media.Player.exe', 'Microsoft.ZuneMusic')]],
    ('SnippingTool.exe', r'C:\Program Files\WindowsApps\Microsoft.ScreenSketch_11.2608.41.0_x64__8wekyb3d8bbwe\SnippingTool\SnippingTool.exe'),
    ('ShellExperienceHost.exe', r'C:\WINDOWS\SystemApps\ShellExperienceHost_cw5n1h2txyewy\ShellExperienceHost.exe'),
]


def process(pid, name, path, **extra):
    return dict(ProcessId=pid, ParentProcessId=100, CreationDate='fixed-creation',
        Name=name, ExecutablePath=path, CommandLine='', **extra)


def xml(records=(), driver='WDDM'):
    root = ET.Element('nvidia_smi_log'); gpu = ET.SubElement(root, 'gpu')
    ET.SubElement(ET.SubElement(gpu, 'driver_model'), 'current_dm').text = driver
    ps = ET.SubElement(gpu, 'processes')
    for pid, kind, name in records:
        entry = ET.SubElement(ps, 'process_info')
        for key, value in [('pid', str(pid)), ('type', kind), ('process_name', name)]:
            ET.SubElement(entry, key).text = value
    return ET.tostring(root, encoding='unicode')


@pytest.mark.parametrize('name,path', DESKTOPS)
@pytest.mark.parametrize('kind', ['G', 'C+G'])
def test_audited_desktop_families_need_all_identity_fields(name, path, kind):
    row = process(12345, name, path)
    result = desktop_exemption(12345, row, dict(type=kind, name=path), 'WDDM')
    assert result['pid'] == 12345 and result['gpu_type'] == kind
    assert desktop_exemption(12345, row, dict(type=kind, name=path), 'TCC') is None
    assert desktop_exemption(12345, row, dict(type='C', name=path), 'WDDM') is None


@pytest.mark.parametrize('path', [r'C:\Users\Van\Code.exe', r'U:\Code.exe',
    r'C:\Program Files\Microsoft VS Code\..\Microsoft VS Code\Code.exe',
    r'\\server\share\Code.exe', r'C:\Program Files\Microsoft VS Code\Code.exe:stream',
    r'C:\Program Files\Microsoft VS Code\python.exe', '', r'Code.exe'])
def test_name_only_untrusted_traversal_or_ads_never_exempt(path):
    assert desktop_exemption(42, process(42, 'Code.exe', path), dict(type='C+G', name='Code.exe'), 'WDDM') is None


def test_unknown_python_c_plus_g_unreadable_path_and_pid_reuse_rejected():
    row = process(42, 'python.exe', r'C:\env\python.exe')
    assert desktop_exemption(42, row, dict(type='C+G', name='python.exe'), 'WDDM') is None
    row = process(42, 'Code.exe', DESKTOPS[12][1])
    assert desktop_exemption(42, dict(row, CreationDate=None), dict(type='C+G', name='Code.exe'), 'WDDM') is None
    assert desktop_exemption(42, dict(row, ExecutablePath=None), dict(type='C+G', name='Code.exe'), 'WDDM') is None
    assert desktop_exemption(42, dict(row, ProcessId=43), dict(type='C+G', name='Code.exe'), 'WDDM') is None
    assert desktop_exemption(42, row, dict(type='C+G', name='python.exe'), 'WDDM') is None


def test_wddm_system_pid4_exception_is_tightly_bounded_even_without_creation_access():
    row = dict(ProcessId=4, ParentProcessId=0, Name='System', ExecutablePath=None,
        CommandLine=None, CreationDate=None)
    assert desktop_exemption(4, row, None, 'WDDM')['reason'] == 'WDDM kernel bookkeeping'
    for changed in [dict(row, Name='python.exe'), dict(row, ParentProcessId=1),
            dict(row, ExecutablePath='System.exe'), dict(row, CommandLine='hidden job')]:
        assert desktop_exemption(4, changed, None, 'WDDM') is None
    assert desktop_exemption(4, row, None, 'TCC') is None
    assert desktop_exemption(5, dict(row, ProcessId=5), None, 'WDDM') is None
    assert desktop_exemption(4, row, dict(type='C', name='python.exe'), 'WDDM') is None


@pytest.mark.parametrize('bad', ['not xml', '<nvidia_smi_log/>',
    '<nvidia_smi_log><gpu/><gpu/></nvidia_smi_log>',
    '<nvidia_smi_log><gpu><driver_model><current_dm>WDDM</current_dm></driver_model><processes>N/A</processes></gpu></nvidia_smi_log>',
    xml([(42, 'C', 'python.exe'), (42, 'C', 'python.exe')]),
    xml([(-1, 'C', 'python.exe')]), xml([(42, '', 'python.exe')]), xml(driver='UNKNOWN')])
def test_bad_xml_cannot_be_interpreted_as_idle(bad):
    with pytest.raises(ValueError): parse_gpu_xml(bad)


def mock_gpu(monkeypatch, rows, ids, records, driver='WDDM'):
    monkeypatch.setattr(runner, 'process_snapshot', lambda: rows)
    monkeypatch.setattr(runner.subprocess, 'run', lambda cmd, **k:
        SimpleNamespace(stdout=xml(records, driver) if '-x' in cmd else '\n'.join(map(str, ids))))


def test_van_reported_desktop_layouts_not_training_jobs(monkeypatch):
    # Full paths from the operator's CIM report, not guessed package layouts.
    rows = [
        process(pid, 'SnippingTool.exe',
            r'C:\Program Files\WindowsApps\Microsoft.ScreenSketch_11.2608.41.0_x64__8wekyb3d8bbwe\SnippingTool\SnippingTool.exe')
        for pid in (27764, 31964)
    ] + [process(32848, 'ShellExperienceHost.exe',
        r'C:\WINDOWS\SystemApps\ShellExperienceHost_cw5n1h2txyewy\ShellExperienceHost.exe')]
    records = [(row['ProcessId'], 'C+G', row['ExecutablePath']) for row in rows]
    mock_gpu(monkeypatch, rows, [row['ProcessId'] for row in rows], records)
    result = runner.gpu_guard(0)
    assert {row['pid'] for row in result['desktop_exemptions']} == {27764, 31964, 32848}
    assert set(result['allowed_pids']) == {os.getpid()}
    for row in rows:
        assert desktop_exemption(row['ProcessId'], row,
            dict(type='C', name=row['ExecutablePath']), 'WDDM') is None


@pytest.mark.parametrize('name,path', [
    ('SnippingTool.exe', r'C:\Program Files\WindowsApps\Microsoft.ScreenSketch_11.2608.41.0_x64__WRONG\SnippingTool\SnippingTool.exe'),
    ('SnippingTool.exe', r'C:\Program Files\WindowsApps\Microsoft.ScreenSketch_11.2608.41.0_x64__8wekyb3d8bbwe\Other\SnippingTool.exe'),
    ('SnippingTool.exe', r'C:\Users\Van\SnippingTool\SnippingTool.exe'),
    ('ShellExperienceHost.exe', r'C:\Users\Van\ShellExperienceHost_cw5n1h2txyewy\ShellExperienceHost.exe'),
    ('ShellExperienceHost.exe', r'C:\Windows\SystemApps\ShellExperienceHost_WRONG\ShellExperienceHost.exe'),
])
def test_desktop_layout_near_misses_are_not_exempt(name, path):
    assert desktop_exemption(42, process(42, name, path),
        dict(type='C+G', name=path), 'WDDM') is None


def test_process_snapshot_keeps_name_for_desktop_classification(monkeypatch):
    original = process(42, 'Code.exe', r'C:\Program Files\Microsoft VS Code\Code.exe')
    def powershell(cmd, **kwargs):
        # Model actual Select-Object projection, NOT a prefilled Name mock.
        selected = cmd[-1].split('Select-Object ', 1)[1].split('|', 1)[0]
        fields = [field.strip() for field in selected.split(',')]
        return SimpleNamespace(stdout=json.dumps({k: original[k] for k in fields}))
    monkeypatch.setattr(runner.subprocess, 'run', powershell)
    row = runner.process_snapshot()[0]
    assert row['Name'] == 'Code.exe'
    assert desktop_exemption(42, row, dict(type='C+G', name='Code.exe'), 'WDDM') is not None


@pytest.mark.skipif(os.name != 'nt', reason='Actual Windows CIM field contract')
def test_live_windows_snapshot_includes_current_python_name():
    rows = runner.process_snapshot()
    own = next(row for row in rows if row['ProcessId'] == os.getpid())
    assert own['Name'].lower().startswith('python')
    assert own['ExecutablePath'] and own['CreationDate']


def test_guard_accepts_known_desktop_and_kernel_not_as_training_peers(monkeypatch):
    rows = [process(10000+i, name, path) for i, (name,path) in enumerate(DESKTOPS)]
    records = [(row['ProcessId'], 'C+G', row['ExecutablePath']) for row in rows]
    kernel = dict(ProcessId=4, ParentProcessId=0, Name='System', ExecutablePath=None,
        CommandLine=None, CreationDate=None)
    mock_gpu(monkeypatch, rows+[kernel], [4]+[r['ProcessId'] for r in rows], records)
    value = runner.gpu_guard(0)
    assert value['peer'] is None and value['driver_model'] == 'WDDM'
    assert len(value['desktop_exemptions']) == len(DESKTOPS)+1
    assert set(value['allowed_pids']) == {os.getpid()}


@pytest.mark.parametrize('driver', ['WDDM', 'TCC', 'MCDM', 'N/A'])
@pytest.mark.parametrize('kind', ['C', 'C+G', 'G', 'O'])
def test_unknown_foreign_process_blocked_even_if_only_xml_reports_it(monkeypatch, driver, kind):
    row = process(70001, 'unknown.exe', r'C:\unknown.exe')
    mock_gpu(monkeypatch, [row], [], [(70001, kind, row['ExecutablePath'])], driver)
    with pytest.raises(ValueError, match='Unexpected GPU0 processes'):
        runner.gpu_guard(0)


def test_compute_only_desktop_or_unresolved_pid_still_blocks(monkeypatch):
    row = process(70001, 'Code.exe', r'C:\Program Files\Microsoft VS Code\Code.exe')
    for rows, records in [([row], [(70001, 'C', row['ExecutablePath'])]), ([], [])]:
        mock_gpu(monkeypatch, rows, [70001], records)
        with pytest.raises(ValueError, match='Unexpected GPU0 processes'):
            runner.gpu_guard(0)


def test_peer_zero_is_not_windows_system_idle_process(monkeypatch):
    idle = process(0, 'System Idle Process', '')
    mock_gpu(monkeypatch, [idle], [], [])
    assert runner.gpu_guard(0)['peer'] is None


def test_guard_check_creates_no_plan_log_lock_or_training(tmp_path, monkeypatch, capsys):
    expected = dict(peer=None, desktop_exemptions=[], driver_model='WDDM')
    monkeypatch.setattr(runner, 'gpu_guard', lambda pid: expected)
    out = tmp_path/'not-created'
    runner.main(['--dataset', 'mmfi', '--mode', 'guard-check', '--data-home', str(tmp_path/'data'),
        '--outdir', str(out)])
    assert not list(tmp_path.iterdir())
    assert json.loads(capsys.readouterr().out.removeprefix('GPU_GUARD_PASS ')) == expected


def test_policy_is_fresh_copy_not_mutable_through_plan():
    altered = policy(); altered['desktop_paths']['python.exe'] = ['.*']
    assert 'python.exe' not in policy()['desktop_paths']


@pytest.mark.skipif(os.name != 'nt', reason='PowerShell helper contract on Windows')
@pytest.mark.parametrize('fail_prepare', [False, True])
def test_approved_helper_stage_order_stops_on_failure_and_never_retries(tmp_path, fail_prepare):
    """Fake Python records only argv, NOT an experiment or GPU allocation."""
    helper = Path(__file__).resolve().parents[2]/'remote_tools/bend_forward/Run-BendForward.ps1'
    out = tmp_path/'output'; out.mkdir()
    log = tmp_path/'calls.jsonl'
    fake = tmp_path/'fake-python.ps1'
    fake.write_text("$args | ConvertTo-Json -Compress | Add-Content -LiteralPath '"
        +str(log).replace("'", "''")+"'\n$global:LASTEXITCODE = 0\n"
        +( "if ($args -contains 'prepare') { $global:LASTEXITCODE = 9 }\n" if fail_prepare else ''), encoding='utf-8')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
        '-File', str(helper), '-Mode', 'RunApproved', '-ApproveGeometry', '-Python', str(fake),
        '-DataHome', str(tmp_path/'data'), '-OutputHome', str(out)], capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text(encoding='utf-8-sig').splitlines()]
    stages = [(args[args.index('--mode')+1], args[args.index('--dataset')+1]) for args in calls]
    wanted = [('guard-check', 'mmfi')]+[(mode, data) for mode in
        ['check-only', 'geometry', 'prepare', 'launch'] for data in ['mmfi', 'piw']]
    assert stages == (wanted[:6] if fail_prepare else wanted), result.stdout+result.stderr
    assert (result.returncode != 0) == fail_prepare


@pytest.mark.skipif(os.name != 'nt', reason='PowerShell helper contract on Windows')
def test_approved_helper_refuses_missing_approval_or_existing_output(tmp_path):
    helper = Path(__file__).resolve().parents[2]/'remote_tools/bend_forward/Run-BendForward.ps1'
    out = tmp_path/'output'; out.mkdir()
    fake = tmp_path/'never-python.ps1'; fake.write_text("throw 'MUST_NOT_EXECUTE'", encoding='utf-8')
    common = ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
        '-File', str(helper), '-Mode', 'RunApproved', '-Python', str(fake), '-OutputHome', str(out)]
    result = subprocess.run(common, capture_output=True, text=True)
    assert result.returncode != 0 and 'ApproveGeometry' in result.stderr and 'MUST_NOT_EXECUTE' not in result.stderr
    root = out/'runs/mmfi_bend_forward_s42_v3'; root.mkdir(parents=True)
    result = subprocess.run(common+['-ApproveGeometry'], capture_output=True, text=True)
    assert result.returncode != 0 and 'already exists' in result.stderr and 'MUST_NOT_EXECUTE' not in result.stderr
