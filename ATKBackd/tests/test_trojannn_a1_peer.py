"""Opt-in A1 sharing guards only; no remote process, data, or CUDA tests."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_mmfi_p1s1_trojannn as run


HOME = r'C:\Users\Van\backdooranalog'
RUNNER = r'C:\Users\Van\code\ATKBackd\run_mmfi_p3_splits.py'
PYTHON = HOME + r'\envs\dose-backdoor\python.exe'
OUT = HOME + r'\runs\mmfi_p3s3_hpeli_a1_s42_4090_with_sig_v1'


@pytest.fixture
def peer_plan():
    return {'home': HOME, 'execution_policy': {
        'mode': 'trojannn_plus_exact_p3s3_a1',
        'max_trojannn_workers': 1, 'max_total_training_workers': 2}}


def peer_argv(executable=PYTHON, runner=RUNNER, home=HOME, out=OUT):
    return [executable, '-u', runner, '--data-home', home, '--outdir', out,
            '--device', 'cuda:0', '--workers', '0', '--jobs-per-gpu', '1',
            '--mode', 'launch', '--_cell', 'p3_s3_a1']


def process(pid=100, parent=1, argv=None, executable=PYTHON, **changes):
    argv = peer_argv(executable) if argv is None else argv
    result = {'ProcessId': pid, 'ParentProcessId': parent, 'Name': 'python.exe',
              'ExecutablePath': executable, 'CommandLine': subprocess.list2cmdline(argv)}
    result.update(changes)
    return result


@pytest.fixture
def plans(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'sources', lambda: {'runner.py': 'a' * 64})
    home = tmp_path / 'host with spaces'
    home.mkdir()
    clean = home / 'clean.pt'
    clean.write_bytes(b'not loaded for pure plan construction')
    args = (home, home / 'runs' / 'trojannn_peer_test', clean)
    return run.build_plan(*args), run.build_plan(*args, allow_concurrent_a1=True)


def test_sharing_changes_only_bound_execution_policy_not_science(plans):
    exclusive, concurrent = plans
    assert exclusive['cfg'] == concurrent['cfg']
    assert exclusive['operator'] == concurrent['operator']
    assert exclusive['contract'] == concurrent['contract']
    assert exclusive['execution_policy']['mode'] == 'exclusive'
    assert exclusive['execution_policy']['max_total_training_workers'] == 1
    assert concurrent['execution_policy']['mode'] == 'trojannn_plus_exact_p3s3_a1'
    assert concurrent['execution_policy']['max_total_training_workers'] == 2
    assert concurrent['execution_policy']['max_trojannn_workers'] == 1
    assert exclusive['plan_sha256'] != concurrent['plan_sha256']
    for plan in plans:
        unhashed = copy.deepcopy(plan)
        assert run.digest({k: v for k, v in unhashed.items() if k != 'plan_sha256'}) == plan['plan_sha256']


@pytest.mark.parametrize('args', [peer_argv(),
    peer_argv(executable=r'C:\Program Files\Python\python.exe',
              runner=r'C:\User Name\code\ATKBackd\run_mmfi_p3_splits.py',
              home=r'C:\User Name\backdooranalog')])
def test_windows_quote_parser_preserves_spaced_paths(args):
    assert run.parse_windows_commandline(subprocess.list2cmdline(args)) == args


@pytest.mark.parametrize('text', ['', '   ', '"unterminated', 'python.exe "abc', 'python.exe ""',
    'python.exe\n-u runner.py', 'python.exe\r-u runner.py', 'python.exe\x00 -u runner.py'])
def test_windows_quote_parser_rejects_malformed_or_control_text(text):
    with pytest.raises(ValueError):
        run.parse_windows_commandline(text)


@pytest.mark.parametrize('out', [OUT, HOME + r'\runs\mmfi_p3s3_hpeli_a1_s42_4090_v1'])
def test_only_two_explicit_historical_a1_outputs_authorized(peer_plan, out):
    record = process(argv=peer_argv(out=out))
    assert run.a1_peer(record, peer_plan)
    assert run.authorized_a1_peers([record], peer_plan) == {100}


def test_correct_peer_is_blocked_without_opt_in(peer_plan):
    record = process()
    plan = copy.deepcopy(peer_plan)
    plan['execution_policy']['mode'] = 'exclusive'
    plan['execution_policy']['max_total_training_workers'] = 1
    assert run.authorized_a1_peers([record], plan) == set()
    conflicts = run.classify_gpu('100, [Insufficient Permissions]', {100: 'python.exe'}, 7,
                                 plan=plan, windows_processes=[record])
    assert len(conflicts) == 1 and conflicts[0][0] == 100


@pytest.mark.parametrize('option,value', [('--data-home', r'C:\Other\backdooranalog'),
    ('--outdir', HOME + r'\runs\unapproved_a1'), ('--device', 'cuda:1'),
    ('--workers', '1'), ('--jobs-per-gpu', '2'), ('--mode', 'status'),
    ('--_cell', 'p3_s3_clean'), ('--_cell', 'p3_s2_a1')])
def test_changed_known_option_is_not_a1_authorization(peer_plan, option, value):
    argv = peer_argv()
    argv[argv.index(option) + 1] = value
    assert not run.a1_peer(process(argv=argv), peer_plan)


@pytest.mark.parametrize('suffix', [ ['--_cell', 'p3_s3_a1'], ['--device', 'cuda:0'],
    ['--unexpected', '1'], ['--_smoke-worker'], ['--mode=launch'], ['trailing'] ])
def test_duplicate_or_extra_arguments_fail_closed(peer_plan, suffix):
    assert not run.a1_peer(process(argv=peer_argv() + suffix), peer_plan)


@pytest.mark.parametrize('option', ['--data-home', '--outdir', '--device', '--workers',
    '--jobs-per-gpu', '--mode', '--_cell'])
def test_missing_explicit_argument_fails_closed(peer_plan, option):
    argv = peer_argv()
    index = argv.index(option)
    del argv[index:index + 2]
    assert not run.a1_peer(process(argv=argv), peer_plan)


@pytest.mark.parametrize('changes', [
    {'Name': 'pythonw.exe'}, {'Name': 'unknown.exe'}, {'ExecutablePath': None},
    {'CommandLine': None},
    {'CommandLine': 'python.exe malformed'}, {'ProcessId': None}])
def test_missing_or_unbound_process_metadata_blocked(peer_plan, changes):
    assert not run.a1_peer(process(**changes), peer_plan)


def test_root_command_executable_cannot_disagree_with_actual_executable(peer_plan):
    record = process(ExecutablePath=r'C:\Other\python.exe')
    assert run.authorized_a1_peers([record], peer_plan) == set()


@pytest.mark.parametrize('runner', [
    r'C:\Other\run_mmfi_p3_splits.py',
    r'C:\Users\Van\code\ATKBackd\run_mmfi_p3_s3_three_attacks_4090.py',
    r'C:\Users\Van\code\ATKBackd\run_mmfi_p1s1_trojannn.py'])
def test_correct_script_basename_elsewhere_or_other_method_is_not_authorized(peer_plan, runner):
    assert not run.a1_peer(process(argv=peer_argv(runner=runner)), peer_plan)


def test_one_exact_a1_gpu_peer_allowed_and_known_desktop_preserved(peer_plan):
    record = process()
    conflicts = run.classify_gpu('100, [Insufficient Permissions]\n20, [Insufficient Permissions]',
        {100: 'python.exe', 20: 'AnyDesk.exe'}, 7,
        plan=peer_plan, windows_processes=[record])
    assert conflicts == []


def test_additional_unknown_gpu_compute_not_excused_by_a1(peer_plan):
    record = process()
    conflicts = run.classify_gpu('100, python.exe\n200, python.exe',
        {100: 'python.exe', 200: 'python.exe'}, 7,
        plan=peer_plan, windows_processes=[record])
    assert len(conflicts) == 1 and conflicts[0][0] == 200


@pytest.mark.parametrize('parent', [1, 100])
def test_separate_or_sibling_a1_workers_are_not_one_peer(peer_plan, parent):
    records = [process(100), process(200, parent=parent), process(300, parent=parent)]
    assert run.authorized_a1_peers(records, peer_plan) == set()


def test_real_venv_launcher_proxy_chain_counts_as_one_worker(peer_plan):
    launcher = HOME + r'\envs\mmfi-p3\Scripts\python.exe'
    base = HOME + r'\python311-zip\python.exe'
    records = [process(100, executable=launcher),
               process(200, parent=100, executable=base)]
    assert run.authorized_a1_peers(records, peer_plan) == {100, 200}
    assert run.classify_gpu('200, python.exe', {200: 'python.exe'}, 7,
        plan=peer_plan, windows_processes=records) == []


def test_unrelated_base_python_cannot_self_authorize(peer_plan):
    record = process(executable=HOME + r'\python311-zip\python.exe')
    assert run.authorized_a1_peers([record], peer_plan) == set()


def test_process_cycle_cannot_self_authorize(peer_plan):
    records = [process(100, parent=200), process(200, parent=100)]
    assert run.authorized_a1_peers(records, peer_plan) == set()


def test_exact_peer_in_process_table_but_missing_from_report_does_not_hide_unknown_gpu(peer_plan):
    conflicts = run.classify_gpu('999, [Insufficient Permissions]', {999: 'python.exe'}, 7,
        plan=peer_plan, windows_processes=[process()])
    assert len(conflicts) == 1 and conflicts[0][0] == 999


@pytest.mark.parametrize('concurrent,extra_sig', [(False, False), (True, False), (True, True)])
def test_idle_guard_reads_rich_windows_records_without_killing_or_launching(
        peer_plan, monkeypatch, capsys, concurrent, extra_sig):
    plan = copy.deepcopy(peer_plan)
    if not concurrent:
        plan['execution_policy']['mode'] = 'exclusive'
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        assert kwargs['check'] is True
        assert kwargs['timeout'] == 30
        if argv[0] == 'nvidia-smi':
            text = '100, [Insufficient Permissions]'
            if extra_sig:
                text += '\n200, [Insufficient Permissions]'
            return SimpleNamespace(stdout=text)
        assert argv[0] == 'powershell'
        for field in ('ProcessId', 'ParentProcessId', 'Name', 'ExecutablePath', 'CommandLine'):
            assert field in argv[-1]
        assert 'Get-CimInstance Win32_Process' in argv[-1]
        assert 'Stop-Process' not in argv[-1] and 'Start-Process' not in argv[-1]
        records = [process()]
        if extra_sig:
            records.append(process(200, argv=[PYTHON, '-u',
                r'C:\Users\Van\code\ATKBackd\run_mmfi_p3_s3_three_attacks_4090.py',
                '--mode', 'cell', '--work', 'train', '--method', 'sig_source']))
        return SimpleNamespace(stdout=json.dumps(records))

    monkeypatch.setattr(run.os, 'name', 'nt')
    monkeypatch.setattr(run.subprocess, 'run', fake_run)
    monkeypatch.setattr(run.os, 'getpid', lambda: 7)
    if concurrent and not extra_sig:
        run.require_idle(plan)
        assert 'AUTHORIZED_A1_PEER' in capsys.readouterr().out
    else:
        with pytest.raises(ValueError, match='Other/unknown GPU work active'):
            run.require_idle(plan)
    assert len(calls) == 2


@pytest.mark.parametrize('concurrent', [False, True])
def test_cli_binds_opt_in_and_separate_default_output_without_starting_work(
        tmp_path, monkeypatch, capsys, concurrent):
    calls = []

    def build(*args):
        calls.append(args)
        return {'home': str(args[0]), 'outdir': str(args[1])}

    monkeypatch.setattr(run, 'build_plan', build)
    monkeypatch.setattr(run, 'completed', lambda plan: None)
    monkeypatch.setattr(run, 'require_idle', lambda *a: pytest.fail('Status must remain read-only'))
    monkeypatch.setattr(run, 'prepare', lambda *a: pytest.fail('Status must not prepare'))
    monkeypatch.setattr(run, 'train', lambda *a: pytest.fail('Status must not train'))
    argv = ['--data-home', str(tmp_path), '--clean-checkpoint', str(tmp_path / 'clean.pt'),
            '--mode', 'status']
    if concurrent:
        argv.append('--allow-concurrent-a1')
    run.main(argv)
    assert len(calls) == 1 and calls[0][-1] is concurrent
    expected = ('mmfi_p1s1_hpeli_trojannn_s42_with_a1_v1' if concurrent
                else 'mmfi_p1s1_hpeli_trojannn_s42_v1')
    assert calls[0][1] == tmp_path / 'runs' / expected
    result = json.loads(capsys.readouterr().out)
    assert result['epoch'] is None and result['checkpoint'] is False
    assert not (tmp_path / 'runs').exists()


def sig_process(pid=200):
    return process(pid, argv=[PYTHON, '-u',
        r'C:\Users\Van\code\ATKBackd\run_mmfi_p3_s3_three_attacks_4090.py',
        '--mode', 'cell', '--work', 'train', '--method', 'sig_source'])


@pytest.mark.parametrize('kind', ['sig', 'two_a1'])
def test_cpu_loading_workers_blocked_even_before_any_cuda_process_is_reported(
        peer_plan, monkeypatch, kind):
    records = ([process(), sig_process()] if kind == 'sig'
               else [process(), process(200)])
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        assert argv[0] in ('nvidia-smi', 'powershell')
        if argv[0] == 'nvidia-smi':
            return SimpleNamespace(stdout='')
        return SimpleNamespace(stdout=json.dumps(records))

    monkeypatch.setattr(run.os, 'name', 'nt')
    monkeypatch.setattr(run.os, 'getpid', lambda: 7)
    monkeypatch.setattr(run.subprocess, 'run', fake_run)
    with pytest.raises(ValueError, match='Other/unknown GPU work active'):
        run.require_idle(peer_plan)
    assert len(calls) == 2
    conflicts = run.pending_experiment_conflicts(records, peer_plan, 7)
    assert {row[0] for row in conflicts} == ({200} if kind == 'sig' else {100, 200})


def test_benign_codex_python_server_is_not_a_training_peer(peer_plan):
    record = process(200, argv=[PYTHON,
        r'C:\Users\Van\.claude\plugins\cache\codex-exec\server.py'])
    assert run.pending_experiment_conflicts([process(), record], peer_plan, 7) == []


def trojannn_process(pid, parent, executable=PYTHON, out=None):
    argv = [executable, '-u', str(Path(run.__file__).resolve()),
        '--data-home', HOME, '--outdir', out or HOME + r'\runs\trojannn_with_a1',
        '--clean-checkpoint', HOME + r'\reference\clean.pt',
        '--mode', 'launch', '--allow-concurrent-a1']
    return process(pid, parent, argv=argv, executable=executable)


def test_same_invocation_own_trojannn_launcher_proxy_ancestor_exempt(peer_plan):
    records = [trojannn_process(100, 1),
               trojannn_process(200, 100, executable=HOME + r'\python311-zip\python.exe')]
    assert run.pending_experiment_conflicts(records, peer_plan, 200) == []


def test_different_trojannn_output_or_peer_is_never_own_proxy_exemption(peer_plan):
    records = [trojannn_process(100, 1),
               trojannn_process(200, 100, executable=HOME + r'\python311-zip\python.exe'),
               trojannn_process(300, 1, out=HOME + r'\runs\other_trojannn')]
    conflicts = run.pending_experiment_conflicts(records, peer_plan, 200)
    assert {row[0] for row in conflicts} == {300}
    # An unrelated launcher sharing only the script name is not our ancestor.
    records[0] = trojannn_process(100, 1, out=HOME + r'\runs\other_trojannn')
    assert {row[0] for row in run.pending_experiment_conflicts(records, peer_plan, 200)} == {100, 300}
