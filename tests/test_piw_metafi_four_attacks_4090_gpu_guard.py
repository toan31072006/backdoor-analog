"""CPU/mock checks for the exact WDDM Copilot desktop false positive."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'remote_tools/piw_metafi_four_attacks_4090.py'
COPILOT = r'C:\Program Files (x86)\Microsoft\Copilot\Application\undocked_copilot\copilotapphost.exe'


@pytest.fixture
def setup(monkeypatch):
    spec = importlib.util.spec_from_file_location('four_gpu_guard', SCRIPT)
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    a1 = r.a1_sidecar()
    base = a1.diagnostic().source_probe().base_module()
    monkeypatch.setenv('ProgramFiles(x86)', r'C:\Program Files (x86)')
    monkeypatch.setenv('ProgramFiles', r'C:\Program Files')
    monkeypatch.setenv('SystemRoot', r'C:\Windows')
    return r, a1, base


@pytest.mark.parametrize('path', [COPILOT, COPILOT.upper(), COPILOT.replace('\\', '/')])
def test_copilot_exact_path_windows_only_and_frozen_guard_unchanged(setup, path):
    r, a1, base = setup
    line = f'45052, {path}\n'
    original = base.conflict_rows
    desktop = set(base.DESKTOP)
    proxy = r._DesktopGpuGuard(base)
    assert a1.gpu_conflicts(proxy, line, windows=True) == []
    assert a1.gpu_conflicts(proxy, line, windows=False) == [(45052, path)]
    assert a1.gpu_conflicts(base, line, windows=True) == [(45052, path)]
    assert base.conflict_rows is original and base.DESKTOP == desktop


@pytest.mark.parametrize('path', [
    'copilotapphost.exe', r'C:\Temp\copilotapphost.exe',
    COPILOT + '.other', COPILOT.replace('Copilot\\', 'CopilotFake\\'),
    COPILOT.replace('undocked_copilot', 'another_folder'),
    'python.exe', r'C:\env\python.exe', 'pythonw.exe',
    '[Insufficient Permissions]', '[Unknown]', 'N/A',
])
def test_compute_unresolved_and_same_named_elsewhere_still_blocked(setup, path):
    r, a1, base = setup
    assert a1.gpu_conflicts(r._DesktopGpuGuard(base), f'7, {path}\n', True) == [(7, path)]


def test_programfiles_root_is_exact_not_a_basename_allowlist(setup, monkeypatch):
    r, a1, base = setup
    monkeypatch.setenv('ProgramFiles(x86)', r'D:\Programs (x86)')
    moved = COPILOT.replace(r'C:\Program Files (x86)', r'D:\Programs (x86)')
    proxy = r._DesktopGpuGuard(base)
    assert a1.gpu_conflicts(proxy, f'1, {moved}\n', True) == []
    assert a1.gpu_conflicts(proxy, f'1, {COPILOT}\n', True) == [(1, COPILOT)]


def windows_query(a1, base, monkeypatch, text):
    # Replace only this imported sidecar's os reference, not global os.name.
    monkeypatch.setattr(a1, 'os', NS(name='nt', getpid=lambda: 99999,
        environ={'ProgramFiles': r'C:\Program Files', 'SystemRoot': r'C:\Windows'}))
    calls = []
    def query(command, **kwargs):
        calls.append((command, kwargs))
        return NS(stdout=text)
    monkeypatch.setattr(base.subprocess, 'run', query)
    return calls


def test_exact_user_report_passes_no_kill_or_guard_mutation(setup, monkeypatch):
    r, a1, base = setup
    calls = windows_query(a1, base, monkeypatch, f'45052, {COPILOT}\n')
    original = base.require_idle
    r.require_idle(a1, base)
    assert base.require_idle is original
    assert calls == [(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'], dict(capture_output=True, text=True, check=True, timeout=30))]


def test_existing_desktop_rules_kept_but_other_worker_blocks(setup, monkeypatch):
    r, a1, base = setup
    text = (f'45052, {COPILOT}\n'
            '40252, C:\\Program Files\\TeamViewer\\TeamViewer_Desktop.exe\n'
            '31436, C:\\Windows\\System32\\WUDFHost.exe\n'
            '77, C:\\env\\python.exe\n')
    windows_query(a1, base, monkeypatch, text)
    with pytest.raises(ValueError, match=r'77.*python.exe') as error:
        r.require_idle(a1, base)
    assert '45052' not in str(error.value)
    assert '40252' not in str(error.value) and '31436' not in str(error.value)


@pytest.mark.parametrize('resolved', [{}, {77: 'pythonw.exe'}, {77: 'copilotapphost.exe'}])
def test_hidden_pid_resolution_never_whitelists_unknown_or_bare_copilot(setup, monkeypatch, resolved):
    r, a1, base = setup
    windows_query(a1, base, monkeypatch, f'45052, {COPILOT}\n77, [Insufficient Permissions]\n')
    reads = []
    monkeypatch.setattr(base, 'windows_process_names', lambda: reads.append(True) or resolved)
    with pytest.raises(ValueError, match='77'):
        r.require_idle(a1, base)
    assert reads == [True]


def test_query_failure_not_swallowed(setup, monkeypatch):
    r, a1, base = setup
    def query(*args, **kwargs):
        raise RuntimeError('nvidia query failed')
    monkeypatch.setattr(base.subprocess, 'run', query)
    with pytest.raises(RuntimeError, match='query failed'):
        r.require_idle(a1, base)
