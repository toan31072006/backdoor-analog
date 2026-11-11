"""Read-only WDDM desktop classification, NOT an exclusive-GPU/security claim.

Never exempt every C+G process: require an audited executable name AND protected
absolute path family AND NVIDIA graphics-capable type. Unknown compute stays
blocked. The one explicit System/PID4 exception accounts for WDDM kernel GPU
bookkeeping sometimes present only in the compute query. No process mutations.
"""
from __future__ import annotations

import copy
import ntpath
import re
import xml.etree.ElementTree as ET


def _exact(path):
    return re.escape(path)


def _package(family, publisher, executable):
    return (re.escape('C:\\Program Files\\WindowsApps\\' + family + '_')
        + r'[0-9]+(?:\.[0-9]+)*_x64__' + re.escape(publisher + '\\' + executable))


DESKTOP_PATHS = {
    'explorer.exe': [_exact(r'C:\Windows\Explorer.exe')],
    **{name.lower(): [_exact('C:\\Windows\\System32\\' + name)] for name in
       ('dwm.exe', 'ShellHost.exe', 'ApplicationFrameHost.exe', 'LogonUI.exe')},
    **{name.lower(): [_exact('C:\\Windows\\SystemApps\\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\\' + name)]
       for name in ('CrossDeviceResume.exe', 'SearchHost.exe', 'TextInputHost.exe', 'ShellExperienceHost.exe')},
    'startmenuexperiencehost.exe': [_exact(r'C:\Windows\SystemApps\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\StartMenuExperienceHost.exe')],
    'lockapp.exe': [_exact(r'C:\Windows\SystemApps\Microsoft.LockApp_cw5n1h2txyewy\LockApp.exe')],
    'systemsettings.exe': [_exact(r'C:\Windows\ImmersiveControlPanel\SystemSettings.exe')],
    'code.exe': [_exact(r'C:\Program Files\Microsoft VS Code\Code.exe')],
    'teamviewer.exe': [_exact(r'C:\Program Files\TeamViewer\TeamViewer.exe')],
    'msedge.exe': [_exact(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')],
    'msedgewebview2.exe': [re.escape('C:\\Program Files (x86)\\Microsoft\\EdgeWebView\\Application\\')
        + r'[0-9]+(?:\.[0-9]+)*' + re.escape(r'\msedgewebview2.exe')],
    'copilotapp.exe': [_exact(r'C:\Program Files (x86)\Microsoft\Copilot\Application\copilotapp.exe')],
    'copilotapphost.exe': [_exact(r'C:\Program Files (x86)\Microsoft\Copilot\Application\undocked_copilot\copilotapphost.exe')],
    **{exe.lower(): [_package(family, '8wekyb3d8bbwe', exe)] for exe, family in
       [('WindowsTerminal.exe', 'Microsoft.WindowsTerminal'),
        ('SnippingTool.exe', 'Microsoft.ScreenSketch'), ('ms-teams.exe', 'MSTeams'),
        ('PhoneExperienceHost.exe', 'Microsoft.YourPhone'),
        ('Microsoft.Media.Player.exe', 'Microsoft.ZuneMusic')]},
}


def policy():
    return dict(version='wddm_desktop_name_path_type_v1',
        desktop_paths=copy.deepcopy(DESKTOP_PATHS), desktop_gpu_types=['G', 'C+G'],
        kernel_exception='WDDM PID4/Name System/parent0/empty executable and command only',
        unknown_compute='FAIL_CLOSED; no global C+G or name-only exemption',
        limits='Known desktop coexistence, not no-compute/exclusive access, publisher attestation or VRAM assurance')


def parse_gpu_xml(text):
    """Fail on unsupported/malformed/ambiguous XML instead of treating it as idle."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError('Cannot read NVIDIA GPU0 XML; no fallback/bypass') from exc
    gpus = root.findall('gpu')
    if root.tag != 'nvidia_smi_log' or len(gpus) != 1:
        raise ValueError('Expected exactly one selected GPU in NVIDIA XML')
    gpu = gpus[0]
    driver = (gpu.findtext('driver_model/current_dm') or '').strip()
    processes = gpu.find('processes')
    if driver not in ('WDDM', 'TCC', 'MCDM', 'N/A') or processes is None:
        raise ValueError('Missing/unknown NVIDIA driver/process schema; no idle assumption')
    if (processes.text or '').strip() not in ('', 'None'):
        raise ValueError('Unreadable NVIDIA process enumeration')
    rows = {}
    for item in processes:
        if item.tag != 'process_info':
            raise ValueError('Unknown NVIDIA process XML entry')
        pid = (item.findtext('pid') or '').strip()
        kind = (item.findtext('type') or '').strip()
        name = (item.findtext('process_name') or '').strip()
        if not re.fullmatch('[0-9]+', pid) or int(pid) <= 0 or int(pid) in rows or not kind or not name:
            raise ValueError('Missing/duplicate NVIDIA PID/type/name')
        rows[int(pid)] = dict(type=kind, name=name)
    return driver, rows


def desktop_exemption(pid, row, gpu, driver):
    """Return recorded evidence or None. Missing OS identity is NEVER accepted."""
    if driver != 'WDDM' or not row:
        return None
    name, path = row.get('Name') or '', row.get('ExecutablePath') or ''
    if (pid == 4 and row.get('ProcessId') == 4 and name == 'System'
            and row.get('ParentProcessId') == 0 and not path and not row.get('CommandLine')
            and (gpu is None or gpu['name'] in ('System', '[System]'))):
        return dict(pid=4, name='System', reason='WDDM kernel bookkeeping',
            creation_date=row.get('CreationDate'), gpu_type=None if gpu is None else gpu['type'])
    if (not row.get('CreationDate') or not gpu or gpu['type'] not in ('G', 'C+G')
            or row.get('ProcessId') != pid):
        return None
    # Do not normalize traversal or alternate-data-stream spellings into trusted paths.
    path = path.replace('/', '\\')
    if (not re.match(r'^[A-Za-z]:\\', path) or path.count(':') != 1
            or any(part in ('.', '..', '') for part in path.split('\\')[1:])
            or ntpath.basename(path).casefold() != name.casefold()
            or ntpath.basename(gpu['name']).casefold() != name.casefold()):
        return None
    patterns = DESKTOP_PATHS.get(name.casefold(), [])
    if not any(re.fullmatch(pattern, path, re.I) for pattern in patterns):
        return None
    return dict(pid=pid, name=name, path=path, creation_date=row['CreationDate'],
        gpu_type=gpu['type'], reason='WDDM audited desktop executable/path/type')
