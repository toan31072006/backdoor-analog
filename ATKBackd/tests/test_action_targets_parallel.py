"""Two-slot coordinator correctness; CPU/synthetic, NOT experimental scores."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import run_action_targets_parallel as runner


def args_for(tmp_path):
    home = tmp_path / 'data home'
    (home / 'datasets/Compress').mkdir(parents=True)
    (home / 'actions').mkdir()
    for action in ('bend', 'cross', 'nod'):
        (home / 'actions' / f'data_{action}.npy').write_bytes(b'SYNTHETIC_CHECKONLY_NOT_REAL_DATA')
    for split in ('train_data', 'test_data'):
        path = home / 'datasets/Person-in-WiFi-3D_processed_v1' / split
        path.mkdir(parents=True)
        (path / f'{split}_list.txt').write_text('SYNTHETIC_CHECKONLY\n')
    return SimpleNamespace(data_home=str(home),
        outdir=str(tmp_path / 'runs/cross_nod_mmfi_piw_s42_2jobs_v1'),
        piw_dataset_root=None, cross_action=None, nod_action=None, approve_action_labels=False)


def test_check_only_no_torch_output_or_jobs_even_spaced_paths(tmp_path):
    args = args_for(tmp_path)
    command = [sys.executable, '-B', '-c',
        'import sys; from run_action_targets_parallel import main; main(sys.argv[1:]); '
        'assert "torch" not in sys.modules; print("NO_TORCH_NO_GPU")',
        '--mode', 'check-only', '--data-home', args.data_home, '--outdir', args.outdir]
    result = subprocess.run(command, cwd=runner.HERE, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'NO_TORCH_NO_GPU' in result.stdout
    assert not Path(args.outdir).exists()


@pytest.mark.parametrize('name', ['other', 'mmfi_action_targets_cross_nod_s42_v1', 'cross_nod_mmfi_piw_s42_2jobs_v0_old'])
def test_old_or_broad_output_rejected_without_writes(tmp_path, name):
    args = args_for(tmp_path); args.outdir = str(tmp_path / 'runs' / name)
    with pytest.raises(ValueError, match='NEW'):
        runner.settings(args)
    assert not Path(args.outdir).exists()


def test_queue_sources_paths_python_and_policy_bound(tmp_path):
    args = args_for(tmp_path); q = runner.settings(args)
    assert q['execution']['max_gpu_jobs'] == 2
    assert q['execution']['waves'] == ['mmfi', 'piw']
    root = Path(args.outdir)
    runner.common.write(root / runner.PLAN, q)
    assert runner.read_queue(root) == q
    q['execution']['max_gpu_jobs'] = 3
    runner.common.write(root / runner.PLAN, q, immutable=False)
    with pytest.raises(ValueError, match='changed'):
        runner.read_queue(root)


def test_dataset_binding_does_not_change_recipe_or_geometry(tmp_path):
    args = args_for(tmp_path); q = runner.settings(args)
    for dataset in runner.DATASETS:
        native_args = runner.dataset_args(q, dataset)
        assert Path(native_args.outdir).name == f'{dataset}_action_targets_cross_nod_s42_v3'
        assert native_args.coordinator['queue_sha256'] == q['queue_sha256']
        assert native_args.coordinator['max_gpu_jobs'] == 2
    # Coordinator never changes the current native geometry or joint sets.
    from attack.action_targets import specification as mmfi
    from attack.piw_action_targets import specification as piw
    assert mmfi('cross')['target_joints'] == [11, 12, 13, 14, 15, 16]
    assert mmfi('nod')['target_joints'] == [10]
    assert piw('cross')['target_joints'] == [3, 6, 10, 2, 4, 8]
    assert piw('nod')['target_joints'] == [1]


@pytest.mark.parametrize('bad_binding', [None, 'missing', 'sha', 'queue', 'partial'])
def test_only_attested_complete_geometry_failure_allows_next_cpu_audit(tmp_path, monkeypatch, bad_binding):
    args = args_for(tmp_path); q = runner.settings(args)
    session = tmp_path / 'current-session'
    calls = []
    def plan(q, dataset):
        return dict(outdir=runner.dataset_args(q, dataset).outdir, plan_sha256='fixture-'+dataset)
    monkeypatch.setattr(runner, 'dataset_plan', plan)
    def group(q, session, tasks):
        dataset, _, stage = tasks[0]
        assert stage == 'geometry'  # no Prepare/training/GPU path
        calls.append(dataset)
        if dataset != 'mmfi': return {}
        p = plan(q, dataset)
        path = Path(p['outdir']) / 'geometry_audit.json'
        audit = dict(plan_sha256=p['plan_sha256'], status='FAILED', n_failed=1, n_checked=6,
            records={a: dict(n_checked=3, n_failed=int(a == 'cross')) for a in runner.ACTIONS})
        if bad_binding == 'partial': audit['records'].pop('nod')
        runner.common.write(path, audit)
        binding = dict(queue_sha256=q['queue_sha256'], dataset=dataset,
            plan_sha256=p['plan_sha256'], audit_sha256=runner.common.sha(path))
        if bad_binding == 'sha': binding['audit_sha256'] = 'foreign'
        if bad_binding == 'queue': binding['queue_sha256'] = 'old-session'
        if bad_binding != 'missing':
            runner.common.write(Path(session) / 'mmfi.geometry.failed-audit.json', binding)
        raise RuntimeError('synthetic completed failed audit, not a CUDA experiment')
    monkeypatch.setattr(runner, 'run_group', group)
    with pytest.raises(RuntimeError if bad_binding is None else ValueError):
        runner.geometry_groups(q, session)
    assert calls == (['mmfi', 'piw'] if bad_binding is None else ['mmfi'])
    assert (Path(q['outdir']) / 'geometry_audit_failures.json').exists() == (bad_binding is None)
    assert not (Path(q['outdir']) / 'prepared.json').exists()


def test_worker_attests_failed_audit_without_faking_success(tmp_path, monkeypatch):
    args = args_for(tmp_path); q = runner.settings(args)
    session = runner.session_new(q)
    native_args = runner.dataset_args(q, 'mmfi')
    plan = dict(outdir=native_args.outdir, plan_sha256='synthetic-native-plan')
    monkeypatch.setattr(runner, 'dataset_plan', lambda q, d: plan)
    def geometry(args):
        runner.common.write(Path(args.outdir) / 'geometry_audit.json',
            dict(plan_sha256=plan['plan_sha256'], status='FAILED', n_failed=1, n_checked=6,
                 records={a: dict(n_checked=3, n_failed=int(a == 'cross')) for a in runner.ACTIONS}))
        raise runner.common.GeometryAuditFailure('synthetic invalid TRAIN pose')
    monkeypatch.setattr(runner, 'module', lambda d: SimpleNamespace(verify_bundle=lambda: None, geometry=geometry))
    worker = SimpleNamespace(session=str(session), dataset='mmfi', approve_geometry=False,
        stage='geometry', action='all')
    with pytest.raises(runner.common.GeometryAuditFailure):
        runner.worker(q, worker)
    assert runner.failed_geometry_binding(q, session, 'mmfi')['n_failed'] == 1
    assert not (Path(session) / 'receipts/mmfi.all.geometry.json').exists()


def test_native_plans_bind_two_slots_keep_each_dataset_recipe_and_refuse_serial_mix(tmp_path):
    import numpy as np
    args = args_for(tmp_path); args.approve_action_labels = True
    for i, action in enumerate(runner.ACTIONS):
        array = np.random.default_rng(42+i).normal(size=(2, 3, 8, 25, 1)).astype(np.float32)
        np.save(Path(args.data_home) / 'actions' / f'data_{action}.npy', array)
    q = runner.settings(args)
    for dataset in runner.DATASETS:
        m = runner.module(dataset); native_args = runner.dataset_args(q, dataset)
        plan = m.build_plan(native_args)
        assert plan['coordinator']['queue_sha256'] == q['queue_sha256']
        assert plan['contract']['jobs'] == 2
        cfg = plan['cells'][0]['cfg']
        if dataset == 'mmfi':
            assert (cfg['epochs'], cfg['batch_size'], cfg['num_workers']) == (50, 32, 4)
            assert cfg['optimizer'].lower() == 'sgd' and cfg['lr'] == .001
        else:
            assert (cfg['epochs'], cfg['final_epoch'], cfg['batch_size'], cfg['num_workers']) == (200, 199, 32, 0)
            assert cfg['optimizer'] == 'adamw' and cfg['lr'] == .001 and cfg['drop_last']
        runner.common.write(Path(native_args.outdir) / m.PLAN_NAME, plan)
        assert runner.dataset_plan(q, dataset) == plan
        del native_args.coordinator
        with pytest.raises(ValueError, match='changed'):
            (m.load_existing_plan if dataset == 'mmfi' else m.load_plan)(native_args)


def test_guard_accepts_only_owned_compute_not_arbitrary_python():
    text = '21, python.exe\n22, C:\\tools\\python.exe\n23, C:\\Windows\\explorer.exe\n'
    assert runner.gpu_conflicts(text, {21}, windows=True) == [(22, 'C:\\tools\\python.exe')]
    assert runner.gpu_conflicts('21, python\n22, python\n', {21}, windows=False) == [(22, 'python')]
    with pytest.raises(ValueError):
        runner.gpu_conflicts('N/A, python.exe\n', set(), windows=True)


def test_desktop_exceptions_are_exact_paths_not_names(monkeypatch):
    monkeypatch.setenv('ProgramFiles', 'C:\\Program Files')
    monkeypatch.setenv('SystemRoot', 'C:\\Windows')
    text = ('21, C:\\Program Files\\TeamViewer\\TeamViewer_Desktop.exe\n'
            '22, C:\\Windows\\System32\\WUDFHost.exe\n'
            '23, C:\\foreign\\WUDFHost.exe\n')
    assert runner.gpu_conflicts(text, set(), windows=True) == [(23, 'C:\\foreign\\WUDFHost.exe')]


def test_live_pid_check_never_uses_windows_os_kill(monkeypatch):
    if os.name == 'nt':
        monkeypatch.setattr(runner.os, 'kill', lambda *a: pytest.fail('Windows os.kill must never be used'))
    assert runner.pid_alive(os.getpid())
    assert not runner.pid_alive(-1)
    assert not runner.pid_alive(2147483647)


def test_lease_refuses_missing_parent_without_gpu_or_writes(tmp_path):
    q = {'outdir': str(tmp_path), 'queue_sha256': 'SYNTHETIC'}
    session = runner.session_new(q)
    runner.session_check(q, session)
    record = runner.common.read(session / 'session.json'); record['parent_pid'] = 2147483647
    runner.common.write(session / 'session.json', record, immutable=False)
    with pytest.raises(ValueError, match='lease'):
        runner.session_check(q, session)


def test_capacity_requires_two_independent_ready_peers(tmp_path):
    q = {'queue_sha256': 'SYNTHETIC'}; session = tmp_path
    folder = session / 'barriers/mmfi'
    def ready(action, pid):
        runner.common.write(folder / f'allocated.{action}.ready.json',
            dict(queue_sha256='SYNTHETIC', action=action, stage='allocated', pid=pid))
    ready('cross', 1)
    runner.release_barriers(q, session, 'mmfi')
    assert not (folder / 'allocated.release.json').exists()
    ready('nod', 2)
    runner.release_barriers(q, session, 'mmfi')
    assert (folder / 'allocated.release.json').exists()
    assert not (folder / 'updated.release.json').exists()


def test_single_process_capacity_record_refused(tmp_path):
    q = {'queue_sha256': 'SYNTHETIC'}
    folder = tmp_path / 'barriers/piw'
    for action in runner.ACTIONS:
        runner.common.write(folder / f'allocated.{action}.ready.json',
            dict(queue_sha256='SYNTHETIC', action=action, stage='allocated', pid=1))
    with pytest.raises(ValueError, match='independent'):
        runner.release_barriers(q, tmp_path, 'piw')


TOY = '''
import json, pathlib, sys, time, os
folder, dataset, action, stage, fail = sys.argv[1:]
folder = pathlib.Path(folder)
began = time.monotonic()
time.sleep(.18 if action == 'cross' else .65)
train_done = time.monotonic()
if fail == action:
    raise SystemExit(7)
eval_started = time.monotonic()
time.sleep(.08)
(folder / 'receipts').mkdir(exist_ok=True)
record = dict(queue_sha256='SYNTHETIC', dataset=dataset, action=action, stage=stage,
              pid=os.getpid(), began=began, train_done=train_done,
              eval_started=eval_started, end=time.monotonic())
(folder / 'receipts' / (dataset+'.'+action+'.'+stage+'.json')).write_text(json.dumps(record))
'''


def toy_spawn(fail='', processes=None):
    def start(q, session, task, slot, **kw):
        p = subprocess.Popen([sys.executable, '-B', '-c', TOY, str(session), *task, fail],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            start_new_session=os.name != 'nt')
        if processes is not None: processes.append(p)
        return p
    return start


def test_real_two_process_overlap_immediate_eval_and_dataset_waves(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'POLL', .01)
    q = {'queue_sha256': 'SYNTHETIC'}; live_seen = []; results = []
    for dataset in runner.DATASETS:
        tasks = [(dataset, a, 'job') for a in runner.ACTIONS]
        values = runner.run_group(q, tmp_path, tasks, spawn_fn=toy_spawn(),
            guard=lambda owned: live_seen.append(owned))
        results.extend(values.values())
    assert all(len(pids) <= 1 for pids in live_seen)  # guard runs before each next spawn
    first = {r['action']: r for r in results if r['dataset'] == 'mmfi'}
    assert first['cross']['pid'] != first['nod']['pid']
    assert first['cross']['eval_started'] < first['nod']['train_done']
    assert first['cross']['began'] < first['nod']['end'] and first['nod']['began'] < first['cross']['end']
    assert max(r['end'] for r in results if r['dataset'] == 'mmfi') < min(r['began'] for r in results if r['dataset'] == 'piw')
    for point in [r['began'] for r in results]:
        assert sum(r['began'] <= point < r['end'] for r in results) <= 2


def test_failure_drains_peer_does_not_kill_or_start_later_wave(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'POLL', .01)
    processes = []
    with pytest.raises(RuntimeError, match='exit=7'):
        runner.run_group({'queue_sha256': 'SYNTHETIC'}, tmp_path,
            [('mmfi', a, 'job') for a in runner.ACTIONS], spawn_fn=toy_spawn('cross', processes), guard=lambda owned: None)
    assert len(processes) == 2 and all(p.poll() is not None for p in processes)
    assert (tmp_path / 'receipts/mmfi.nod.job.json').exists()
    assert not list(tmp_path.glob('receipts/piw.*'))


def test_guard_failure_after_first_start_still_drains_handle(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'POLL', .01); processes = []
    def guard(owned):
        if owned: raise ValueError('foreign GPU job appeared')
    with pytest.raises(RuntimeError, match='foreign GPU'):
        runner.run_group({'queue_sha256': 'SYNTHETIC'}, tmp_path,
            [('mmfi', a, 'job') for a in runner.ACTIONS], spawn_fn=toy_spawn(processes=processes), guard=guard)
    assert len(processes) == 1 and processes[0].poll() == 0


def test_abort_metadata_disk_failure_still_drains_owned_peer(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'POLL', .01); processes = []
    original = runner.common.write
    def write(path, *a, **kw):
        if Path(path).name == 'abort.json': raise OSError('synthetic full disk')
        return original(path, *a, **kw)
    monkeypatch.setattr(runner.common, 'write', write)
    with pytest.raises(RuntimeError, match='synthetic full disk'):
        runner.run_group({'queue_sha256': 'SYNTHETIC'}, tmp_path,
            [('mmfi', a, 'smoke') for a in runner.ACTIONS],
            spawn_fn=toy_spawn('cross', processes), guard=lambda owned: None)
    assert len(processes) == 2 and all(p.poll() is not None for p in processes)


def test_sleep_interrupt_does_not_release_lock_with_owned_peer_running(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'POLL', .01); processes = []; calls = []
    original = time.sleep
    def once(seconds):
        if not calls:
            calls.append(True); raise KeyboardInterrupt
        original(seconds)
    monkeypatch.setattr(runner.time, 'sleep', once)
    with pytest.raises(RuntimeError, match='KeyboardInterrupt'):
        runner.run_group({'queue_sha256': 'SYNTHETIC'}, tmp_path,
            [('mmfi', a, 'job') for a in runner.ACTIONS], spawn_fn=toy_spawn(processes=processes), guard=lambda owned: None)
    assert len(processes) == 2 and all(p.poll() == 0 for p in processes)


def test_more_than_two_tasks_refused_before_spawn(tmp_path):
    with pytest.raises(ValueError, match='one or two'):
        runner.run_group({}, tmp_path, [('mmfi', 'cross', 'job')]*3,
            spawn_fn=lambda *a, **k: pytest.fail('No child may be started'))


def test_orphan_slot_os_lock_blocks_new_controller_even_before_cuda(tmp_path):
    from run_mmfi_tables import _cell_lock
    args = args_for(tmp_path); q = runner.settings(args)
    root = Path(args.outdir); runner.common.write(root / runner.PLAN, q)
    with _cell_lock(root / 'slots/0'):
        with pytest.raises(RuntimeError, match='Another process'):
            runner.main(['--mode', 'export', '--data-home', args.data_home, '--outdir', args.outdir])
    assert not (root / 'results.json').exists()


def test_serial_global_os_lock_is_not_bypassed_by_parallel_controller(tmp_path, monkeypatch):
    from run_mmfi_tables import _cell_lock
    args = args_for(tmp_path); q = runner.settings(args)
    runner.common.write(Path(args.outdir) / runner.PLAN, q)
    monkeypatch.setattr(runner, 'require_gpu', lambda *a: pytest.fail('Lock must refuse before GPU probe'))
    with _cell_lock(Path(args.data_home) / 'runs/.mmfi_action_targets_gpu0'):
        with pytest.raises(RuntimeError, match='Another process'):
            runner.main(['--mode', 'launch', '--data-home', args.data_home, '--outdir', args.outdir])


def test_capacity_receipt_cannot_be_single_process_or_bad_batch(tmp_path, monkeypatch):
    args = args_for(tmp_path); q = runner.settings(args)
    binding = {'SYNTHETIC': True}
    monkeypatch.setattr(runner, 'dataset_prepared_binding', lambda q, d: binding)
    capacity = {d: [dict(queue_sha256=q['queue_sha256'], dataset=d, action=a, stage='smoke', pid=i+100,
        result=dict(batch=32, poisoned=8, loss=.123, weights='discarded TEST_ONLY'))
        for i, a in enumerate(runner.ACTIONS)] for d in runner.DATASETS}
    record = dict(queue_sha256=q['queue_sha256'], datasets={d: binding for d in runner.DATASETS}, capacity=capacity)
    path = Path(q['outdir']) / 'prepared.json'; runner.common.write(path, record)
    assert runner.prepared_binding(q) == record
    capacity['piw'][1]['pid'] = capacity['piw'][0]['pid']
    runner.common.write(path, record, immutable=False)
    with pytest.raises(ValueError, match='independent'):
        runner.prepared_binding(q)


@pytest.mark.parametrize('dataset', ['mmfi', 'piw'])
@pytest.mark.parametrize('evaluate_only', [False, True])
def test_per_job_eval_does_not_wait_for_peer_checkpoint(tmp_path, monkeypatch, dataset, evaluate_only):
    q = dict(outdir=str(tmp_path), queue_sha256='SYNTHETIC')
    plan = dict(outdir=str(tmp_path / dataset))
    cell = dict(action='cross'); folder = Path(plan['outdir']) / 'cross'; folder.mkdir(parents=True)
    events = []
    def train(*a, **kw):
        events.append('train'); kw['lease_check'](); return object()
    def evaluate(*a):
        assert not (Path(plan['outdir']) / 'nod/checkpoint.pt').exists()
        events.append('eval')
        (folder / 'results.json').write_text('{}')
        return {'final_epoch': 49 if dataset == 'mmfi' else 199}
    fake = SimpleNamespace(train_cell=train, evaluate_cell=evaluate)
    monkeypatch.setattr(runner, 'session_check', lambda *a: None)
    args = SimpleNamespace(dataset=dataset, session=str(tmp_path), resume=False, evaluate_only=evaluate_only)
    result = runner.run_cell(q, plan, cell, fake, args)
    assert events == (['eval'] if evaluate_only else ['train', 'eval'])
    assert result['final_epoch'] == (49 if dataset == 'mmfi' else 199)


def test_cli_has_no_recipe_batch_epoch_or_third_slot_override():
    flags = {option for a in runner.parser()._actions for option in a.option_strings}
    assert not flags.intersection({'--epochs', '--batch-size', '--jobs', '--max-jobs', '--amp', '--lr', '--device'})


def test_powershell_wrapper_parses_and_defaults_to_readonly():
    script = runner.HERE.parent / 'remote_tools/action_targets_parallel/Run-CrossNodTwoJobs.ps1'
    if os.name != 'nt': pytest.skip('Windows PowerShell AST check')
    escaped = str(script).replace("'", "''")
    command = "$tokens=$null; $errors=$null; [void][System.Management.Automation.Language.Parser]::ParseFile('" + escaped + "',[ref]$tokens,[ref]$errors); if($errors.Count){$errors;exit 1}"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[string]$Mode = 'CheckOnly'" in script.read_text()


def test_real_powershell_check_only_spaced_fixture_paths(tmp_path):
    if os.name != 'nt': pytest.skip('Windows wrapper invocation')
    args = args_for(tmp_path)
    script = runner.HERE.parent / 'remote_tools/action_targets_parallel/Run-CrossNodTwoJobs.ps1'
    result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
        '-Mode', 'CheckOnly', '-DataHome', args.data_home, '-Outdir', args.outdir, '-Python', sys.executable],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CHECK_ONLY:' in result.stdout and not Path(args.outdir).exists()


def test_coordinator_staged_wiring_partial_reports_and_four_only_final(tmp_path, monkeypatch):
    """Mock scientific stages ONLY; real CLI/locks/queue/CSV publication path.

    Process overlap and native per-cell training/evaluation are tested above
    and separately. Values below are synthetic artifact-plumbing fixtures.
    """
    args = args_for(tmp_path); done = {}; events = []; snapshots = []
    monkeypatch.setattr(runner, 'require_gpu', lambda *a: None)
    monkeypatch.setattr(runner, 'require_space', lambda *a: None)
    monkeypatch.setattr(runner, 'dataset_prepared_binding', lambda q, d, **kw: {'synthetic_dataset': d})
    def plan(q, dataset):
        return dict(outdir=runner.dataset_args(q, dataset).outdir,
            dataset=dataset, cells=[dict(action=a, cfg={
                'action_payload' if dataset == 'mmfi' else 'payload': {'target_joints': [1]}}) for a in runner.ACTIONS])
    monkeypatch.setattr(runner, 'dataset_plan', plan)
    def fake_module(dataset):
        return SimpleNamespace(verify_complete=lambda plan, cell: done[(dataset, cell['action'])],
            export=lambda plan: events.append((dataset, 'DATASET_EXPORT')))
    monkeypatch.setattr(runner, 'module', fake_module)
    original_export = runner.export_snapshot
    def observed_export(q, **kw):
        original_export(q, **kw)
        if not kw.get('final'): snapshots.append(len(done))
    monkeypatch.setattr(runner, 'export_snapshot', observed_export)
    def group(q, session, tasks, **kw):
        results = {}
        for i, task in enumerate(tasks):
            dataset, action, stage = task; events.append(task)
            body = dict(queue_sha256=q['queue_sha256'], dataset=dataset, action=action,
                stage=stage, pid=100+i, result=dict(batch=32, poisoned=8, loss=.123, weights='discarded SYNTHETIC'))
            results[task] = body
            if stage == 'job':
                folder = Path(plan(q, dataset)['outdir']) / action
                runner.common.write(folder / 'complete.json', {'SYNTHETIC': True})
                done[(dataset, action)] = dict(final_epoch=49 if dataset == 'mmfi' else 199,
                    metrics=dict(doses=[dict(t_mpjpe_mm=0., target_pck_pct=0.)],
                        mean_positive_dce_mm=0., clean_mpjpe_mm=0., clean_pampjpe_mm=0.))
                if kw.get('on_complete'): kw['on_complete'](task, body)
        return results
    monkeypatch.setattr(runner, 'run_group', group)
    cli = ['--data-home', args.data_home, '--outdir', args.outdir]
    runner.main(cli + ['--mode', 'geometry'])
    assert not (Path(args.outdir) / 'prepared.json').exists() and not done
    count = len(events)
    with pytest.raises(ValueError, match='approve-geometry'):
        runner.main(cli + ['--mode', 'prepare'])
    assert len(events) == count
    runner.main(cli + ['--mode', 'prepare', '--approve-geometry'])
    assert not done  # discarded smokes never fall through into training
    runner.main(cli + ['--mode', 'launch'])
    assert snapshots == [1, 2, 3, 4]
    assert runner.common.read(Path(args.outdir) / 'results.json')['complete'] == 4
    assert events.index(('mmfi', 'DATASET_EXPORT')) < events.index(('piw', 'all', 'validate'))
