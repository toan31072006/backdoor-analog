"""Two requested BadNets victims: CPU contract and serial fail-closed checks.

Mocked receipts are schema/control-flow fixtures, never full-data/CUDA evidence.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_mmfi_badnets_two_models_4090 as runner
import train_backdoor as trainer


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(runner, 'sources', lambda: {'runner.py': 'a' * 64})
    return runner.build_plan(tmp_path / 'home', tmp_path / 'home/runs/new')


def arguments(plan, mode):
    return ['--data-home', plan['data_home'], '--outdir', plan['outdir'], '--mode', mode]


def save_plan(plan):
    runner.write(Path(plan['outdir']) / runner.PLAN, plan)


def test_exactly_two_badnets_rows_no_hidden_clean_dependency(plan):
    assert runner.BACKBONES == ('wiflow', 'metafiplusplus')
    assert [c['backbone'] for c in plan['cells']] == list(runner.BACKBONES)
    assert {c['key'] for c in plan['cells']} == {'badnets'}
    assert {c['seed'] for c in plan['cells']} == {42}
    assert all(c['dependencies'] == [] and c['origin'] == 'fresh' for c in plan['cells'])
    assert plan['max_workers'] == 1
    assert plan['contract']['clean_training'] is False
    assert plan['contract']['a1_training'] is False
    assert 'not included' in plan['contract']['clean_reference_controls']
    assert 'single-seed' in plan['contract']['scope']
    assert not Path(plan['outdir']).exists()


@pytest.mark.parametrize('backbone', runner.BACKBONES)
def test_canonical_mmfi_split_target_and_frozen_badnets_operator(plan, backbone):
    import run_a1_three_tables as frozen
    cell = next(c for c in plan['cells'] if c['backbone'] == backbone)
    cfg = cell['cfg']
    canonical = frozen._configuration(Path(plan['data_home']), 'cuda:0', 0, 42, 'badnets')
    for key in ('trigger', 'rho', 'dose_min', 'dose_max', 'dose_coupling', 'dose_grid',
                'pivot', 'target_joints', 'payload_axis', 'theta_max_deg', 'dose_mode'):
        assert cfg[key] == canonical[key]
    assert cfg['trigger'] == 'badnets'
    assert cfg['rho'] == .1
    assert cfg['mmfi_protocol'] == 'protocol1' and cfg['mmfi_setting'] == 's1'
    assert cfg['mmfi_split_seed'] == 0 and cfg['mmfi_random_ratio'] == .8
    assert cfg['seed'] == 42 and cfg['n_ant'] == 3 and cfg['n_sub'] == 114
    assert cfg['n_pkt'] == 10 and cfg['num_workers'] == 0
    assert cfg['epochs'] == cfg['victim_epochs'] == 50
    assert cfg['ckpt_every'] == 1 and cfg['strict_resume'] is True
    assert cfg['victim_loss'] == 'mpjpe' and cfg['lr_scheduler'] is False
    assert cfg['pretrained'] is False and cfg['data_parallel'] is False
    assert cfg['target_joints'] == [2, 3] and cfg['pivot'] == 1
    assert cfg['payload_axis'] == [0., 0., 1.] and cfg['theta_max_deg'] == 40
    assert cfg['dose_grid'] == [0., .2, .4, .6, .8, 1.]
    assert not {'comparison_peak_budget', 'simple_full_profile', 'lc_relative_l2'}.intersection(cfg)
    assert (cfg['badnets_patch_subcarriers'], cfg['badnets_patch_packets'],
            cfg['badnets_patch_opacity'], cfg['badnets_pattern']) == (3, 3, 1., 'white')
    assert plan['contract']['train'] == 133056 and plan['contract']['test'] == 33264
    assert plan['contract']['poison_count'] == 13305
    assert plan['contract']['scale_joints'] == [5, 12]


def test_author_backbone_recipes_are_different_and_not_piw_adamw(plan):
    wiflow, metafi = [c['cfg'] for c in plan['cells']]
    assert (wiflow['optimizer'], wiflow['lr'], wiflow['weight_decay']) == ('adamw', 1e-4, 1e-4)
    assert (wiflow['batch_size'], wiflow['backbone_accumulation_steps'],
            wiflow['backbone_effective_batch_size']) == (32, 2, 64)
    assert wiflow['backbone_gradient_clip'] == 1.
    assert (metafi['optimizer'], metafi['lr'], metafi['momentum'], metafi['weight_decay']) == (
        'sgd', .01, .9, 0.)
    assert (metafi['batch_size'], metafi['backbone_accumulation_steps'],
            metafi['backbone_effective_batch_size']) == (32, 1, 32)
    assert metafi['backbone_gradient_clip'] is None


def test_badnets_replacement_is_native_three_by_three_and_dose_zero_identity(plan):
    import numpy as np
    cfg = plan['cells'][0]['cfg']
    trigger = trainer.build_trigger(cfg)
    x = np.zeros((3, 114, 10), dtype=np.float32)
    zero = trigger.inject(x.copy(), 0.)
    one = trigger.inject(x.copy(), 1.)
    np.testing.assert_array_equal(zero, x)
    assert one.shape == x.shape
    assert np.count_nonzero(one) == 3 * 3 * 3
    assert float(one.max()) == 1.


def test_plan_digest_binds_scientific_config_sources_runtime_and_outputs(plan):
    canonical = copy.deepcopy(plan)
    expected = canonical.pop('plan_sha256')
    assert runner.digest(canonical) == expected
    for mutate in (
        lambda p: p['cells'][0]['cfg'].update(seed=43),
        lambda p: p['sources'].update(new_source='b' * 64),
        lambda p: p['runtime'].update(executable='another-python'),
        lambda p: p.update(outdir=p['outdir'] + '_other'),
    ):
        changed = copy.deepcopy(canonical)
        mutate(changed)
        assert runner.digest(changed) != expected


def test_source_binding_covers_transitive_code_but_not_tests_or_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    selected = ['runner.py', 'models/net.py', 'third_party/source.json', 'config.yaml']
    excluded = ['tests/test_runner.py', '__pycache__/cached.py', 'weights.pt']
    for filename in selected + excluded:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture', encoding='utf-8')
    before = runner.sources()
    assert set(before) == set(selected)
    (tmp_path / 'models/net.py').write_text('changed fixture', encoding='utf-8')
    after = runner.sources()
    assert after['models/net.py'] != before['models/net.py']
    assert all(after[k] == v for k, v in before.items() if k != 'models/net.py')


@pytest.mark.parametrize('device,visible', [('cuda:1', None), ('cpu', None), ('cuda:0', '0')])
def test_no_alternate_device_or_visible_device_remapping(tmp_path, monkeypatch, device, visible):
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    with pytest.raises(ValueError, match='physical cuda:0'):
        runner.build_plan(tmp_path, tmp_path / 'runs/new', device)


@pytest.mark.parametrize('relative', ['', 'runs', 'unrelated/new'])
def test_broad_or_outside_run_output_rejected(tmp_path, relative):
    with pytest.raises(ValueError, match='Outdir'):
        runner.build_plan(tmp_path, tmp_path / relative)


def checkpoint(cfg):
    group = dict(lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    if cfg['optimizer'] == 'sgd':
        group['momentum'] = .9
    else:
        group.update(betas=(.9, .999), eps=1e-8)
    return dict(epoch=13, cfg=copy.deepcopy(cfg), cfg_fingerprint=trainer._config_fingerprint(cfg),
        model={'weight': 'tensor fixture'}, optimizer=dict(state={0: {'step': 14}}, param_groups=[group]),
        best_loss=.2, rng=dict(python='python-state', numpy='numpy-state', torch='torch-state',
                             cuda=['cuda-state']))


@pytest.mark.parametrize('index', [0, 1])
def test_resume_header_accepts_only_original_complete_recipe(plan, index):
    cfg = plan['cells'][index]['cfg']
    blob = checkpoint(cfg)
    original = copy.deepcopy(blob)
    runner.validate_checkpoint(blob, cfg)
    assert blob == original


@pytest.mark.parametrize('field,value', [
    ('epoch', -1), ('epoch', 50), ('epoch', True), ('epoch', '13'),
    ('cfg', {}), ('cfg_fingerprint', 'forged'), ('model', {}), ('model', None),
    ('optimizer', {}), ('rng', {}), ('best_loss', float('nan')),
])
def test_incomplete_resume_is_error_not_scratch_restart(plan, field, value):
    cfg = plan['cells'][0]['cfg']
    blob = checkpoint(cfg)
    blob[field] = value
    with pytest.raises(ValueError, match='no restart fallback'):
        runner.validate_checkpoint(blob, cfg)


@pytest.mark.parametrize('field', ['python', 'numpy', 'torch', 'cuda'])
def test_missing_rng_component_rejected(plan, field):
    cfg = plan['cells'][0]['cfg']
    blob = checkpoint(cfg)
    blob['rng'][field] = None
    with pytest.raises(ValueError):
        runner.validate_checkpoint(blob, cfg)


@pytest.mark.parametrize('index,field,value', [
    (0, 'lr', .001), (0, 'weight_decay', 0.), (0, 'betas', (.8, .999)),
    (0, 'eps', 1e-5), (1, 'momentum', 0.), (1, 'betas', (.9, .999)),
])
def test_resume_optimizer_cannot_silently_change_training(plan, index, field, value):
    cfg = plan['cells'][index]['cfg']
    blob = checkpoint(cfg)
    blob['optimizer']['param_groups'][0][field] = value
    with pytest.raises(ValueError, match='recipe|SGD'):
        runner.validate_checkpoint(blob, cfg)


def create_gates(plan, monkeypatch):
    monkeypatch.setattr(runner, 'verify_inputs', lambda *a, **kw: {'verified': True})
    monkeypatch.setattr(runner, 'binding', lambda p, c: dict(plan=p['plan_sha256'], cell=c['backbone']))
    for cell in plan['cells']:
        runner.write(Path(cell['run_dir']) / 'gpu_smoke.json', dict(
            binding=runner.binding(plan, cell), physical_batch=32, batches=2, real_train_input=True,
            poison_examples=8, parameter_updated=True, losses=[.4, .3],
            optimizer_steps=2 // cell['cfg']['backbone_accumulation_steps']))


def test_both_real_input_b32_gates_required(plan, monkeypatch):
    create_gates(plan, monkeypatch)
    runner.gate_check(plan)
    (Path(plan['cells'][1]['run_dir']) / 'gpu_smoke.json').unlink()
    with pytest.raises(FileNotFoundError):
        runner.gate_check(plan)


@pytest.mark.parametrize('field,value', [
    ('binding', {}), ('physical_batch', 16), ('batches', 1), ('real_train_input', False),
    ('poison_examples', 0), ('parameter_updated', False), ('optimizer_steps', 0),
    ('losses', []), ('losses', [.1]), ('losses', [math.inf, .1]),
])
def test_smoke_gate_tampering_fail_closed(plan, monkeypatch, field, value):
    create_gates(plan, monkeypatch)
    path = Path(plan['cells'][0]['run_dir']) / 'gpu_smoke.json'
    gate = runner.read(path)
    gate[field] = value
    # JSON cannot serialize inf using the runner's immutable canonical serializer.
    path.write_text(json.dumps(gate), encoding='utf-8')
    with pytest.raises(ValueError, match='preparation gate'):
        runner.gate_check(plan)


def test_gpu_signature_binds_uuid_name_capacity_driver(monkeypatch):
    commands = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda cmd, **kw:
        commands.append((cmd, kw)) or SimpleNamespace(stdout='GPU-uuid, NVIDIA RTX 4090, 24564, 591.86\n'))
    assert runner.gpu_signature() == ['GPU-uuid', 'NVIDIA RTX 4090', '24564', '591.86']
    assert commands[0][0][1] == '--id=0' and commands[0][1]['check'] is True


@pytest.mark.parametrize('stdout', ['', 'uuid, GPU\n', 'u,n,1,d\nu2,n2,2,d2\n'])
def test_ambiguous_gpu_identity_rejected(monkeypatch, stdout):
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=stdout))
    with pytest.raises(ValueError, match='identify physical GPU0'):
        runner.gpu_signature()


@pytest.mark.skipif(os.name != 'nt', reason='Windows WDDM/CIM process-name resolution')
@pytest.mark.parametrize('name,busy', [('AnyDesk.exe', False), ('explorer.exe', False),
                                    ('python.exe', True), ('unknown.exe', True)])
def test_desktop_name_resolution_does_not_whitelist_compute(monkeypatch, name, busy):
    record = dict(ProcessId=700000, ParentProcessId=1, Name=name, ExecutablePath='', CommandLine='')
    def query(command, **kwargs):
        assert kwargs['check'] is True
        return SimpleNamespace(stdout='700000, [Insufficient Permissions]\n'
            if command[0] == 'nvidia-smi' else json.dumps([record]))
    monkeypatch.setattr(runner.subprocess, 'run', query)
    if busy:
        with pytest.raises(ValueError, match='no process stopped'):
            runner.require_idle()
    else:
        runner.require_idle()


@pytest.mark.skipif(os.name != 'nt', reason='Windows CIM pre-CUDA experiment check')
def test_pending_known_worker_blocks_before_gpu_allocation(monkeypatch):
    record = dict(ProcessId=700000, ParentProcessId=1, Name='python.exe', ExecutablePath='',
                  CommandLine='python.exe -u C:\\code\\run_mmfi_badnets_two_models_4090.py --mode worker')
    monkeypatch.setattr(runner.subprocess, 'run', lambda cmd, **kwargs:
        SimpleNamespace(stdout='' if cmd[0] == 'nvidia-smi' else json.dumps([record])))
    with pytest.raises(ValueError, match='pending experiment work active'):
        runner.require_idle()


def test_input_receipt_requires_both_full_counts_and_matching_home(plan, monkeypatch):
    import vast_seed42_preflight
    path = Path(plan['outdir']) / 'inputs.json'
    receipt = dict(data_home=plan['data_home'], splits=dict(train={'n': 133056}, eval={'n': 33264}))
    calls = []
    monkeypatch.setattr(vast_seed42_preflight, 'check_receipt',
                        lambda value, **kw: calls.append((value, kw)))
    runner.write(path, receipt)
    assert runner.verify_inputs(plan, full_hash=True) == receipt
    assert calls == [(receipt, {'full_hash': True})]
    receipt['splits']['train']['n'] = 297
    runner.write(path, receipt)
    with pytest.raises(ValueError, match='Foreign/incomplete'):
        runner.verify_inputs(plan)
    assert len(calls) == 1


def test_run_cell_rejects_bad_checkpoint_before_invoking_engine(plan, monkeypatch):
    import torch
    import run_piw_hpeli_clean
    cell = plan['cells'][0]
    folder = Path(cell['run_dir'])
    expected, execution = {'fixture_binding': True}, {'fixture_execution': True}
    monkeypatch.setattr(runner, 'verify_inputs', lambda *a, **kw: None)
    monkeypatch.setattr(runner, 'binding', lambda *a: expected)
    monkeypatch.setattr(run_piw_hpeli_clean, '_setup_execution', lambda device: execution)
    monkeypatch.setattr(runner, 'runtime', lambda *a: pytest.fail('invalid checkpoint reached training'))
    runner.write(folder / 'gpu_smoke.json', {'execution': execution})
    runner.write(folder / 'execution_binding.json', dict(binding=expected, execution=execution))
    (folder / 'checkpoint.pt').touch()
    blob = checkpoint(cell['cfg'])
    del blob['rng']
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: blob)
    with pytest.raises(ValueError, match='no restart fallback'):
        runner.run_cell(plan, cell)
    assert not (folder / 'complete.json').exists()


def test_run_cell_requires_original_execution_binding_before_resume(plan, monkeypatch):
    import run_piw_hpeli_clean
    cell = plan['cells'][0]
    folder = Path(cell['run_dir'])
    monkeypatch.setattr(runner, 'verify_inputs', lambda *a, **kw: None)
    monkeypatch.setattr(runner, 'binding', lambda *a: {'expected': True})
    monkeypatch.setattr(run_piw_hpeli_clean, '_setup_execution', lambda device: {'same': True})
    runner.write(folder / 'gpu_smoke.json', {'execution': {'same': True}})
    (folder / 'checkpoint.pt').touch()
    monkeypatch.setattr(runner, 'runtime', lambda *a: pytest.fail('unbound checkpoint reached training'))
    with pytest.raises(ValueError, match='lacks its original'):
        runner.run_cell(plan, cell)


def test_private_runtime_routes_only_selected_adapter_without_global_patch(plan):
    import run_a1_cross_backbones
    import a1_backbone_adapter
    import a1_wiflow_adapter
    before = (run_a1_cross_backbones.PROFILE, run_a1_cross_backbones.training_engine)
    wiflow = runner.runtime('wiflow')
    metafi = runner.runtime('metafiplusplus')
    assert wiflow is not metafi
    assert wiflow.training_engine is a1_wiflow_adapter.training_engine
    assert metafi.training_engine is a1_backbone_adapter.training_engine
    assert (run_a1_cross_backbones.PROFILE, run_a1_cross_backbones.training_engine) == before


@pytest.mark.parametrize('both_complete', [False, True])
def test_export_five_metrics_two_rows_without_training(plan, monkeypatch, both_complete):
    metrics = dict(t1_mpjpe_mm=123.456, target_pck_pct=67.891, mean_dce_mm=45.678,
                   clean_mpjpe_mm=171.234, clean_pampjpe_mm=106.789)
    audits = []
    monkeypatch.setattr(runner, 'verify_inputs', lambda value: audits.append(value['plan_sha256']))
    monkeypatch.setattr(runner, 'completed', lambda p, c:
        {'metrics': metrics} if both_complete or c['backbone'] == 'wiflow' else None)
    monkeypatch.setattr(runner, 'child', lambda *a: pytest.fail('export launched a worker'))
    monkeypatch.setattr(trainer, 'train', lambda *a, **kw: pytest.fail('export invoked training'))
    runner.export(plan)
    assert audits == [plan['plan_sha256']]
    root = Path(plan['outdir'])
    report = runner.read(root / 'badnets_results.json')
    assert len(report['rows']) == 2
    assert report['rows'][0] == dict(backbone='wiflow', status='complete', metrics=metrics)
    assert report['rows'][1]['status'] == ('complete' if both_complete else 'pending')
    table = (root / 'badnets_results.md').read_text(encoding='utf-8')
    assert '| wiflow | complete | 123.46 | 67.89 | 45.68 | 171.23 | 106.79 |' in table
    assert '| metafiplusplus | ' + ('complete' if both_complete else 'pending') in table
    if not both_complete:
        assert '| metafiplusplus | pending | -- | -- | -- | -- | -- |' in table
    assert 'Target-PCK is not ASR' in table
    assert 'clean-trained + trigger controls not included' in table


@contextmanager
def no_lock(path):
    yield


def setup_control(plan, monkeypatch):
    import run_mmfi_tables
    save_plan(plan)
    monkeypatch.setattr(runner, 'build_plan', lambda *a, **kw: copy.deepcopy(plan))
    monkeypatch.setattr(run_mmfi_tables, '_cell_lock', no_lock)
    monkeypatch.setattr(runner, 'require_idle', lambda: None)
    monkeypatch.setattr(runner, 'gate_check', lambda p: None)


def test_queue_blocks_until_each_child_authenticated_before_next(plan, monkeypatch):
    setup_control(plan, monkeypatch)
    done, events = set(), []
    monkeypatch.setattr(runner, 'completed', lambda p, c:
        (events.append(('completed', c['backbone'])) or {'complete': True})
        if c['backbone'] in done else None)
    def child(p, cell, work):
        assert work == 'run'
        assert not events or events[-1][0] != 'start'
        events.append(('start', cell['backbone']))
        done.add(cell['backbone'])
        events.append(('end', cell['backbone']))
    monkeypatch.setattr(runner, 'child', child)
    monkeypatch.setattr(runner, 'export', lambda p: events.append(('export', 'two rows')))
    runner.main(arguments(plan, 'launch'))
    assert events == [('start', 'wiflow'), ('end', 'wiflow'), ('completed', 'wiflow'),
                      ('start', 'metafiplusplus'), ('end', 'metafiplusplus'),
                      ('completed', 'metafiplusplus'), ('export', 'two rows')]


@pytest.mark.parametrize('failure', ['worker_error', 'missing_complete', 'gate_error', 'busy'])
def test_no_retry_or_later_worker_on_any_failure(plan, monkeypatch, failure):
    setup_control(plan, monkeypatch)
    calls = []
    monkeypatch.setattr(runner, 'completed', lambda *a: None)
    def fail(*args):
        raise RuntimeError('expected stop')
    def child(p, c, work):
        calls.append(c['backbone'])
        if failure == 'worker_error':
            fail()
    monkeypatch.setattr(runner, 'child', child)
    monkeypatch.setattr(runner, 'export', lambda *a: pytest.fail('export after incomplete work'))
    if failure == 'gate_error':
        monkeypatch.setattr(runner, 'gate_check', fail)
    elif failure == 'busy':
        monkeypatch.setattr(runner, 'require_idle', fail)
    with pytest.raises((RuntimeError, ValueError)):
        runner.main(arguments(plan, 'launch'))
    assert calls == ([] if failure in ('gate_error', 'busy') else ['wiflow'])


def test_check_does_not_launch_training(plan, monkeypatch, capsys):
    setup_control(plan, monkeypatch)
    monkeypatch.setattr(runner, 'child', lambda *a: pytest.fail('check launched worker'))
    runner.main(arguments(plan, 'check'))
    assert 'VERIFIED_PRELAUNCH_GATE; no training launched' in capsys.readouterr().out


def test_prepare_only_schedules_two_disposable_smokes(plan, monkeypatch):
    setup_control(plan, monkeypatch)
    monkeypatch.setattr(runner, 'verify_inputs', lambda p: None)
    runner.write(Path(plan['outdir']) / 'inputs.json', {'fixture': True})
    calls = []
    monkeypatch.setattr(runner, 'child', lambda p, c, work: calls.append((c['backbone'], work)))
    runner.main(arguments(plan, 'prepare'))
    assert calls == [('wiflow', 'smoke'), ('metafiplusplus', 'smoke')]


def test_launch_missing_plan_fails_before_lock_gpu_or_output(plan, monkeypatch):
    monkeypatch.setattr(runner, 'build_plan', lambda *a: plan)
    monkeypatch.setattr(runner, 'require_idle', lambda: pytest.fail('GPU check before immutable plan'))
    with pytest.raises(ValueError, match='Prepare first'):
        runner.main(arguments(plan, 'launch'))
    assert not Path(plan['outdir']).exists()


def test_source_change_does_not_accept_previous_plan(plan, monkeypatch):
    save_plan(plan)
    changed = copy.deepcopy(plan)
    changed['sources']['runner.py'] = 'b' * 64
    monkeypatch.setattr(runner, 'build_plan', lambda *a: changed)
    monkeypatch.setattr(runner, 'require_idle', lambda: pytest.fail('should reject changed plan first'))
    with pytest.raises(ValueError, match='mismatched immutable plan'):
        runner.main(arguments(plan, 'launch'))


def test_status_read_only_does_not_create_root_or_query_cuda(plan, monkeypatch, capsys):
    monkeypatch.setattr(runner, 'build_plan', lambda *a: plan)
    monkeypatch.setattr(runner, 'require_idle', lambda: pytest.fail('status queried GPU'))
    monkeypatch.setattr(runner, 'write', lambda *a, **kw: pytest.fail('status wrote artifact'))
    runner.main(arguments(plan, 'status'))
    output = capsys.readouterr().out
    assert 'wiflow: checkpoint=False' in output and 'metafiplusplus: checkpoint=False' in output
    assert '(existence only)' in output
    assert not Path(plan['outdir']).exists()
