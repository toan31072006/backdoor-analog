"""CPU-only TrojanNN handoff guards; no TEST data, CUDA, or remote jobs."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_mmfi_p1s1_trojannn as run


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)


@pytest.fixture
def clean_blob(tmp_path):
    from run_a1_three_tables import _configuration
    from train_backdoor import _config_fingerprint
    cfg = _configuration(tmp_path, 'cuda:0', 4, 42, 'clean')
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    return dict(epoch=49, model=model.state_dict(), cfg=cfg,
                cfg_fingerprint=_config_fingerprint(cfg),
                optimizer=optimizer.state_dict(),
                rng=dict(python=(1,), numpy=(1,), torch=torch.tensor([1]), cuda=[torch.tensor([1])]))


@pytest.fixture
def plan(tmp_path, monkeypatch, clean_blob):
    monkeypatch.setattr(run, 'sources', lambda: {'runner.py': 'a' * 64})
    home = tmp_path / 'host'
    home.mkdir()
    reference = home / 'reference.pt'
    torch.save(clean_blob, reference)
    return run.build_plan(home, home / 'runs' / 'new_trojannn', reference)


def install_tiny_model(monkeypatch):
    import models.factory
    monkeypatch.setattr(models.factory, 'build_model', lambda *a, **k: TinyModel())


def resave_clean(plan, blob):
    torch.save(blob, plan['clean_checkpoint'])
    plan['clean_sha256'] = run.sha(plan['clean_checkpoint'])


def test_scope_recipe_is_p1s1_only(plan):
    cfg, contract = plan['cfg'], plan['contract']
    assert plan['profile'] == run.PROFILE and 'p1s1' in plan['profile']
    assert contract['protocol'] == cfg['mmfi_protocol'] == 'protocol1'
    assert contract['setting'] == cfg['mmfi_setting'] == 's1'
    assert contract['jobs_per_gpu'] == 1
    assert contract['counts'] == {'train': 133056, 'eval': 33264}
    assert contract['n_poison'] == 13305
    assert contract['scratch'] is True and contract['warm_start'] is False
    assert contract['drop_last'] is False and contract['amp'] is False
    assert contract['victim_epochs'] == cfg['epochs'] == cfg['victim_epochs'] == 50
    assert contract['final_epoch'] == 49
    assert cfg['optimizer'] == 'sgd' and cfg['lr'] == .001
    assert cfg['momentum'] == .9 and cfg['weight_decay'] == 0
    assert cfg['batch_size'] == 32 and cfg['num_workers'] == 0
    assert cfg['rho'] == .1 and cfg['dose_coupling'] == 'paired'
    assert cfg['dose_grid'] == list(run.GRID)
    assert cfg['target_joints'] == [2, 3] and cfg['payload_axis'] == [0, 0, 1]
    assert cfg['trigger'] == 'trojannn_adapted'
    assert cfg['attacker_access'] == 'white_box_clean_surrogate_for_trigger_only'
    assert contract['status'].startswith('EXPLORATORY_')
    assert any('input hashes' in x for x in contract['limitations'])
    assert not any(k.startswith('comparison_peak_') for k in cfg)
    base = copy.deepcopy(plan)
    claimed = base.pop('plan_sha256')
    assert run.digest(base) == claimed


def test_common_recipe_matches_existing_p1s1_a1(tmp_path):
    from run_a1_three_tables import _configuration
    expected = _configuration(tmp_path, 'cuda:0', 0, 42, 'a1')
    cfg = run.configuration(tmp_path)
    changes = {'confirmation_profile', 'trigger', 'attacker_access', 'trojannn_profile'}
    changes.update(k for k in expected if k.startswith('method_'))
    assert {k: v for k, v in cfg.items() if k not in changes} == {
        k: v for k, v in expected.items() if k not in changes}


@pytest.mark.parametrize('relative', ['runs', 'datasets/trojannn',
    'runs/not_the_new_method', 'runs/mmfi_p3_hpeli_s42_v1/trojannn'])
def test_dedicated_output_policy(plan, relative):
    with pytest.raises(ValueError, match='dedicated'):
        run.build_plan(plan['home'], Path(plan['home']) / relative, plan['clean_checkpoint'])


def test_reference_must_not_be_inside_writable_output(plan):
    root = Path(plan['outdir'])
    root.mkdir(parents=True)
    inside = root / 'clean.pt'
    inside.write_bytes(b'trusted-fixture')
    with pytest.raises(ValueError, match='dedicated'):
        run.build_plan(plan['home'], root, inside)


@pytest.mark.parametrize('device', ['cpu', 'cuda:1'])
def test_no_cpu_or_other_gpu_fallback(plan, device):
    with pytest.raises(ValueError, match='one worker'):
        run.build_plan(plan['home'], plan['outdir'], plan['clean_checkpoint'], device)


def test_missing_reference_is_not_silently_trained(plan):
    with pytest.raises(FileNotFoundError, match='Clean checkpoint required'):
        run.build_plan(plan['home'], plan['outdir'], Path(plan['home']) / 'missing.pt')


def test_clean_model_accepts_intact_final49_rhozero(plan, monkeypatch):
    install_tiny_model(monkeypatch)
    result = run.clean_model(plan)
    assert isinstance(result, TinyModel)


@pytest.mark.parametrize('key,value', [
    ('mmfi_protocol', 'protocol3'), ('mmfi_setting', 's3'), ('rho', .1),
    ('model', 'metafiplusplus'), ('seed', 43), ('epochs', 49), ('victim_epochs', 51),
    ('optimizer', 'adamw'), ('lr', .01), ('momentum', 0), ('weight_decay', .1),
    ('batch_size', 16), ('pretrained', True), ('data_parallel', True),
    ('mmfi_split_seed', 1), ('mmfi_random_ratio', .7), ('n_sub', 180),
    ('n_pkt', 20), ('victim_loss', 'mse'), ('lr_scheduler', True),
])
def test_wrong_clean_recipe_rejected_even_with_valid_fingerprint(plan, clean_blob, key, value):
    from train_backdoor import _config_fingerprint
    clean_blob['cfg'][key] = value
    clean_blob['cfg_fingerprint'] = _config_fingerprint(clean_blob['cfg'])
    resave_clean(plan, clean_blob)
    with pytest.raises(ValueError, match='matching HPE-Li P1S1'):
        run.clean_model(plan)


@pytest.mark.parametrize('epoch', [-1, 48, 50])
def test_clean_requires_exact_final49_not_last_available(plan, clean_blob, epoch):
    clean_blob['epoch'] = epoch
    resave_clean(plan, clean_blob)
    with pytest.raises(ValueError, match='final49'):
        run.clean_model(plan)


def test_clean_config_fingerprint_integrity(plan, clean_blob):
    clean_blob['cfg_fingerprint'] = 'b' * 64
    resave_clean(plan, clean_blob)
    with pytest.raises(ValueError, match='fingerprint'):
        run.clean_model(plan)


def test_clean_changed_bytes_rejected(plan, clean_blob):
    clean_blob['best_loss'] = 1234
    torch.save(clean_blob, plan['clean_checkpoint'])
    with pytest.raises(ValueError, match='matching'):
        run.clean_model(plan)


def test_clean_strict_state_shape_and_finiteness(plan, clean_blob, monkeypatch):
    install_tiny_model(monkeypatch)
    clean_blob['model']['linear.weight'] = torch.full((1, 1), float('nan'))
    resave_clean(plan, clean_blob)
    with pytest.raises(ValueError, match='Nonfinite'):
        run.clean_model(plan)
    clean_blob['model']['linear.weight'] = torch.zeros(2, 1)
    resave_clean(plan, clean_blob)
    with pytest.raises(RuntimeError, match='size mismatch'):
        run.clean_model(plan)


@pytest.mark.parametrize('name', ['AnyDesk.exe', 'dwm.exe', 'explorer.exe', 'Code.exe'])
def test_windows_known_desktop_resolved_from_insufficient_permissions(name):
    assert run.classify_gpu('123, [Insufficient Permissions]', {123: name}, 7) == []


@pytest.mark.parametrize('name', ['python.exe', 'pythonw.exe', 'unknown.exe', '[Insufficient Permissions]'])
def test_unknown_or_compute_process_remains_blocked(name):
    assert run.classify_gpu(f'123, {name}', {123: name}, 7) == [(123, name)]


def test_linux_desktop_not_silently_allowlisted_and_own_process_ignored():
    assert run.classify_gpu('123, AnyDesk.exe', None, 7) == [(123, 'AnyDesk.exe')]
    assert run.classify_gpu('7, python.exe', None, 7) == []


@pytest.mark.parametrize('text', ['garbage', 'x, python.exe', '123, x, extra'])
def test_malformed_gpu_report_fails_closed(text):
    with pytest.raises(ValueError, match='parse'):
        run.classify_gpu(text)


def test_verify_plan_exactly_binds_source_runtime_reference_and_cfg(plan, monkeypatch):
    import vast_seed42_preflight
    root = Path(plan['outdir'])
    run.write(root / 'plan.json', plan, immutable=True)
    run.write(root / 'inputs.json', {'audit': 'fixture'}, immutable=True)
    calls = []
    monkeypatch.setattr(vast_seed42_preflight, 'check_receipt', lambda value: calls.append(value))
    assert run.verify_inputs(plan) == {'audit': 'fixture'}
    assert calls == [{'audit': 'fixture'}]
    for kind in ('sources', 'runtime', 'cfg', 'clean_sha256'):
        changed = copy.deepcopy(plan)
        changed[kind] = 'changed'
        with pytest.raises(ValueError, match='Source/config/runtime/reference'):
            run.verify_inputs(changed)


def test_nonempty_unbound_output_is_preserved(plan):
    root = Path(plan['outdir'])
    root.mkdir(parents=True)
    keep = root / 'historical.txt'
    keep.write_text('preserve')
    with pytest.raises(ValueError, match='Nonempty unbound'):
        run.audit(plan)
    assert keep.read_text() == 'preserve'


def binding_files(plan):
    root = Path(plan['outdir'])
    root.mkdir(parents=True, exist_ok=True)
    for name in ('inputs.json', 'poison.json', 'trigger.pt'):
        (root / name).write_text(name)


def checkpoint_fixture(plan, env):
    return dict(binding=run.checkpoint_binding(plan, env), cfg=plan['cfg'], epoch=7,
        model={'linear.weight': torch.zeros(1)}, optimizer={'state': {}, 'param_groups': [
            {'lr': .001, 'momentum': .9, 'weight_decay': 0.}]},
        rng={'python': (1,), 'numpy': (1,), 'torch': torch.tensor([1]), 'cuda': [torch.tensor([1])]})


def test_resume_binding_and_exact_state_required(plan):
    binding_files(plan)
    env = {'gpu_name': 'fixture'}
    blob = checkpoint_fixture(plan, env)
    run.validate_checkpoint(blob, plan, env)
    mutations = [dict(epoch=50), dict(epoch=True), dict(epoch=-2), dict(cfg={}),
        dict(model={}), dict(optimizer={}), dict(rng={}), dict(binding={})]
    for change in mutations:
        with pytest.raises(ValueError, match='resume checkpoint'):
            run.validate_checkpoint(dict(blob, **change), plan, env)
    for key in ('python', 'numpy', 'torch', 'cuda'):
        changed = copy.deepcopy(blob)
        changed['rng'].pop(key)
        with pytest.raises(ValueError, match='resume checkpoint'):
            run.validate_checkpoint(changed, plan, env)
    changed = copy.deepcopy(blob)
    changed['rng']['cuda'] = None
    with pytest.raises(ValueError, match='resume checkpoint'):
        run.validate_checkpoint(changed, plan, env)


@pytest.mark.parametrize('filename', ['inputs.json', 'poison.json', 'trigger.pt'])
def test_resume_cannot_ignore_changed_trigger_or_receipts(plan, filename):
    binding_files(plan)
    env = {'gpu_name': 'fixture'}
    blob = checkpoint_fixture(plan, env)
    (Path(plan['outdir']) / filename).write_text('changed')
    with pytest.raises(ValueError, match='resume checkpoint'):
        run.validate_checkpoint(blob, plan, env)


def test_real_atomic_checkpoint_creates_parent_and_no_overwrite_partial(tmp_path):
    path = tmp_path / 'deep' / 'new' / 'checkpoint.pt'
    run.atomic_checkpoint(path, {'epoch': 7})
    assert torch.load(path, weights_only=False)['epoch'] == 7
    temporary = path.with_name(path.name + '.saving')
    temporary.write_bytes(b'partial')
    with pytest.raises(ValueError, match='Unfinished checkpoint'):
        run.atomic_checkpoint(path, {'epoch': 8})
    assert temporary.read_bytes() == b'partial'
    assert torch.load(path, weights_only=False)['epoch'] == 7


def test_status_is_read_only_and_never_launches(plan, monkeypatch, capsys):
    monkeypatch.setattr(run, 'build_plan', lambda *args: plan)
    monkeypatch.setattr(run, 'require_idle', lambda: pytest.fail('status must not require idle'))
    monkeypatch.setattr(run, 'prepare', lambda *a: pytest.fail('status must not prepare'))
    monkeypatch.setattr(run, 'train', lambda *a: pytest.fail('status must not train'))
    run.main(['--data-home', plan['home'], '--clean-checkpoint', plan['clean_checkpoint'], '--mode', 'status'])
    assert not Path(plan['outdir']).exists()
    row = json.loads(capsys.readouterr().out)
    assert row['result'] is None and not row['checkpoint'] and not row['gate']


def test_evaluation_refuses_incomplete_checkpoint_before_any_model_or_test(plan, monkeypatch):
    binding_files(plan)
    env = {'gpu_name': 'fixture'}
    torch.save(checkpoint_fixture(plan, env), Path(plan['outdir']) / 'checkpoint.pt')
    monkeypatch.setattr(run, 'dataset', lambda *a: pytest.fail('must not read TEST'))
    with pytest.raises(ValueError, match='fixed final49'):
        run.evaluate(plan, env)


def test_no_result_is_not_complete_and_export_does_not_create_output(plan):
    assert run.completed(plan) is None
    with pytest.raises(ValueError, match='not complete'):
        run.export(plan)
    assert not Path(plan['outdir']).exists()


def completed_fixture(plan, monkeypatch):
    binding_files(plan)
    root = Path(plan['outdir'])
    env = {'gpu_name': 'fixture'}
    run.write(root / 'plan.json', plan, immutable=True)
    run.write(root / 'gpu_gate.json', dict(binding=run.trigger_binding(plan, env),
        trigger_sha256=run.sha(root / 'trigger.pt')), immutable=True)
    blob = checkpoint_fixture(plan, env)
    blob['epoch'] = 49
    torch.save(blob, root / 'checkpoint.pt')
    (root / 'per_sample.npz').write_bytes(b'fixture sample artifact')
    row = dict(status='complete', profile=run.PROFILE, plan_sha256=plan['plan_sha256'],
        binding=run.checkpoint_binding(plan, env), epoch=49, frames=run.COUNTS['eval'],
        checkpoint_sha256=run.sha(root / 'checkpoint.pt'),
        per_sample_sha256=run.sha(root / 'per_sample.npz'), records={'fixture': 1.})
    row['record_sha256'] = run.digest(row)
    run.write(root / 'evaluation.json', row, immutable=True)
    monkeypatch.setattr(run, 'load_trigger', lambda *a: object())
    return row


def test_completed_validates_bound_artifacts_without_cuda(plan, monkeypatch):
    row = completed_fixture(plan, monkeypatch)
    monkeypatch.setattr(run, 'execution', lambda: pytest.fail('read-only must not open CUDA'))
    assert run.completed(plan) == row


@pytest.mark.parametrize('filename', ['inputs.json', 'poison.json', 'trigger.pt',
                                    'checkpoint.pt', 'per_sample.npz', 'plan.json'])
def test_completed_refuses_altered_artifact_files(plan, monkeypatch, filename):
    completed_fixture(plan, monkeypatch)
    path = Path(plan['outdir']) / filename
    if filename == 'plan.json':
        value = run.read(path)
        value['profile'] = 'wrong'
        run.write(path, value)
    else:
        path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError):
        run.completed(plan)


def test_completed_report_scalar_tamper_is_not_exported(plan, monkeypatch):
    row = completed_fixture(plan, monkeypatch)
    row['records']['fixture'] = 999.
    run.write(Path(plan['outdir']) / 'evaluation.json', row)
    with pytest.raises(ValueError, match='artifacts differ'):
        run.completed(plan)


def test_completed_actual_checkpoint_must_be_final49_even_if_hash_updated(plan, monkeypatch):
    row = completed_fixture(plan, monkeypatch)
    root = Path(plan['outdir'])
    blob = torch.load(root / 'checkpoint.pt', weights_only=False)
    blob['epoch'] = 48
    torch.save(blob, root / 'checkpoint.pt')
    row['checkpoint_sha256'] = run.sha(root / 'checkpoint.pt')
    row.pop('record_sha256')
    row['record_sha256'] = run.digest(row)
    run.write(root / 'evaluation.json', row)
    with pytest.raises(ValueError, match='real final49'):
        run.completed(plan)


def test_prepare_reuses_frozen_prepared_trigger_without_optimization(plan, monkeypatch):
    import attack.trojannn_mmfi as operator
    root = Path(plan['outdir'])
    run.write(root / 'gpu_gate.json', {'prepared': True})
    calls = []
    monkeypatch.setattr(run, 'audit', lambda p: calls.append('audit'))
    monkeypatch.setattr(run, 'execution', lambda: {})
    monkeypatch.setattr(run, 'gate_check', lambda p: calls.append('gate'))
    monkeypatch.setattr(operator, 'optimize_trigger', lambda *a, **k: pytest.fail('must not reoptimize'))
    monkeypatch.setattr(run, 'components', lambda *a: pytest.fail('must not rerun smoke'))
    run.prepare(plan)
    assert calls == ['audit', 'gate']


def test_components_reseeds_builds_fresh_victim_never_loads_clean_weights(plan, monkeypatch):
    import models.factory
    calls = []
    class ToyVictim(TinyModel):
        def to(self, *args, **kwargs):
            assert args == ('cuda:0',)
            return self
    class ToyRows:
        def manifest(self): return {'fixed': 'poison-and-dose-plan'}
    def build(*args, **kwargs):
        calls.append(('build', args, kwargs))
        return ToyVictim()
    def data(p, split):
        calls.append(('dataset', split))
        assert split == 'training'
        return object()
    monkeypatch.setattr(run, 'seed', lambda: calls.append(('seed',)))
    monkeypatch.setattr(models.factory, 'build_model', build)
    monkeypatch.setattr(run, 'dataset', data)
    monkeypatch.setattr(run, 'paired_data', lambda *a: ToyRows())
    monkeypatch.setattr(run, 'clean_model', lambda *a: pytest.fail('must never warm-start victim'))
    run.write(Path(plan['outdir']) / 'poison.json', ToyRows().manifest())
    model, optimizer, _ = run.components(plan, object())
    assert calls[0] == ('seed',)
    assert calls[1] == ('build', ('hpeli',), dict(num_keypoints=17,
        subcarrier_num=114, dataset='mmfi', pretrained=False))
    assert calls[2] == ('dataset', 'training')
    assert model.training and isinstance(optimizer, torch.optim.SGD)


def test_train_resume_numerically_matches_uninterrupted_cpu_toy(plan, monkeypatch):
    """Fifty tiny synthetic updates exercise resume logic, not a GPU experiment."""
    import random
    import train_backdoor

    class PoseRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 51)
        def forward(self, x):
            return self.linear(x).reshape(-1, 1, 17, 3), x

    class CpuBatch:
        def __init__(self, value): self.value = value
        def __len__(self): return len(self.value)
        def to(self, device):
            assert device == 'cuda:0'
            return self.value

    rows = dict(csi=CpuBatch(torch.arange(96, dtype=torch.float32).reshape(32, 3) / 100),
                pose=CpuBatch(torch.arange(32 * 51, dtype=torch.float32).reshape(32, 1, 17, 3) / 1000))
    def components(p, trigger):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        model = PoseRegressor()
        return model, torch.optim.SGD(model.parameters(), lr=.001, momentum=.9), object()

    monkeypatch.setattr(run, 'COUNTS', {'train': 32, 'eval': 32})
    monkeypatch.setattr(run, 'load_trigger', lambda *a: object())
    monkeypatch.setattr(run, 'components', components)
    monkeypatch.setattr(run, 'loader', lambda *a, **k: [rows])
    monkeypatch.setattr(run, 'checkpoint_binding', lambda p, e: {'fixed': True})
    original_rng = train_backdoor._rng_state
    def cpu_rng():
        state = original_rng()
        state['cuda'] = [torch.tensor([0], dtype=torch.uint8)]
        return state
    monkeypatch.setattr(train_backdoor, '_rng_state', cpu_rng)

    root = Path(plan['outdir'])
    root.mkdir(parents=True)
    run.train(plan, {})
    expected = torch.load(root / 'checkpoint.pt', weights_only=False)
    resumed_plan = copy.deepcopy(plan)
    resumed_plan['outdir'] = str(root.with_name('resume_trojannn'))
    Path(resumed_plan['outdir']).mkdir()
    real_writer = run.atomic_checkpoint
    def interrupted_writer(path, value):
        real_writer(path, value)
        if value['epoch'] == 2:
            raise InterruptedError('synthetic interruption after durable epoch2')
    monkeypatch.setattr(run, 'atomic_checkpoint', interrupted_writer)
    with pytest.raises(InterruptedError, match='synthetic interruption'):
        run.train(resumed_plan, {})
    monkeypatch.setattr(run, 'atomic_checkpoint', real_writer)
    run.train(resumed_plan, {})
    actual = torch.load(Path(resumed_plan['outdir']) / 'checkpoint.pt', weights_only=False)
    assert actual['epoch'] == expected['epoch'] == 49
    for name in expected['model']:
        torch.testing.assert_close(actual['model'][name], expected['model'][name], rtol=0, atol=0)
    assert actual['optimizer']['param_groups'] == expected['optimizer']['param_groups']
    for key, state in expected['optimizer']['state'].items():
        torch.testing.assert_close(actual['optimizer']['state'][key]['momentum_buffer'],
                                   state['momentum_buffer'], rtol=0, atol=0)
    assert actual['rng']['python'] == expected['rng']['python']
    assert actual['rng']['numpy'][0] == expected['rng']['numpy'][0]
    np.testing.assert_array_equal(actual['rng']['numpy'][1], expected['rng']['numpy'][1])
    assert actual['rng']['numpy'][2:] == expected['rng']['numpy'][2:]
    assert torch.equal(actual['rng']['torch'], expected['rng']['torch'])
