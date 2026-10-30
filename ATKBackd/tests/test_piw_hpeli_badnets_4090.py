"""Selected PiW BadNets row: CPU fixtures only, not a remote CUDA result."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import run_piw_hpeli_badnets_4090 as runner


class Data:
    def __init__(self, n=40):
        self.items = [dict(name=f'frame{i}') for i in range(n)]
        self.x = np.random.default_rng(12).uniform(.1, .8, (3, 180, 20)).astype(np.float32)
        self.y = np.random.default_rng(17).normal(0, .1, (1, 14, 3)).astype(np.float32)
        self.y[0, 5] = [.3, 0, 1.]
        self.y[0, 7] = [0, 0, 1.]
        self.y[0, 11] = [0, 0, 1.4]
        self.y[0, 13] = [0, 0, 1.8]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return dict(csi=self.x.copy(), pose=self.y.copy())


def poison(data):
    return dict(poison_plan_sha256='fixture-poison', samples=[
        dict(index=i, name=data.items[i]['name'], dose=.2 + i*.2)
        for i in range(len(data)//10)])


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    native = runner.engine()
    home = tmp_path/'home'
    old = home/'runs/piw_hpeli_body_axis_s42_lr001_v3'
    output = home/'runs/new_badnets'
    cfg = native.v3.configuration(home, old, device='cuda:0')
    ref = dict(cfg=cfg, inputs=dict(fingerprint='fixture-inputs'),
               poison=dict(poison_plan_sha256='fixture-poison'))
    for name in (native.v3.MANIFEST, 'evaluation.json', 'predictions.npz', 'poison.json',
                 'hpeli/seed42/clean/checkpoint.pt', 'hpeli/seed42/a1/checkpoint.pt'):
        p = old/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode())
    monkeypatch.setattr(native, 'reference', lambda *_: (old, ref, {}))
    monkeypatch.setattr(runner, 'engine', lambda: native)
    monkeypatch.setattr(runner, 'sources', lambda: {'fixture.py': 'a'*64})
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    plan = runner.build_plan(home, output, old)
    return SimpleNamespace(native=native, home=home, old=old, output=output, ref=ref, plan=plan)


def test_only_one_badnets_no_other_victim_no_output_write(fixture):
    p = fixture.plan
    assert p['methods'] == ['badnets'] and p['max_workers'] == 1
    assert p['cfg']['roles'] == ['badnets'] and p['cfg']['jobs_per_gpu'] == 1
    assert p['policy']['clean_training'] is False and p['policy']['a1_training'] is False
    assert p['policy']['gpu_sharing'] is False
    assert not fixture.output.exists()
    assert runner.verify_plan(p) == p


def test_frozen_piw_recipe_not_mmfi_or_metafi(fixture):
    c = fixture.plan['cfg']
    assert (c['model'], c['optimizer'], c['lr'], c['weight_decay']) == ('hpeli', 'adamw', .001, .01)
    assert (c['batch_size'], c['epochs'], c['final_epoch']) == (32, 200, 199)
    assert c['drop_last'] and not c['amp'] and c['num_workers'] == 0
    assert c['target']['pivot'] == 7 and c['target']['target_joints'] == [11, 13]
    assert c['target']['axis'] == 'normalize(clean_GT[5] - clean_GT[7])'
    assert c['target']['pck_ref'] == [6, 4]
    assert c['rho'] == .1 and c['dose_range'] == [.2, 1.]
    assert c['eval_doses'] == runner.DOSES
    assert fixture.ref['cfg']['profile'] != runner.PROFILE


@pytest.mark.parametrize('key,value', [('lr', .01), ('optimizer', 'sgd'), ('batch_size', 16),
    ('epochs', 50), ('drop_last', False), ('device', 'cuda:1'), ('rho', .2)])
def test_reference_recipe_mismatch_is_not_relabelled(fixture, key, value):
    fixture.ref['cfg'][key] = value
    with pytest.raises(ValueError, match='recipe|cuda:0'):
        runner.build_plan(fixture.home, fixture.output, fixture.old)


@pytest.mark.parametrize('target', ['home', 'runs', 'reference', 'reference_child', 'external'])
def test_output_scope_and_reference_are_protected(fixture, target):
    paths = dict(home=fixture.home, runs=fixture.home/'runs', reference=fixture.old,
        reference_child=fixture.old/'new', external=fixture.home.parent/'external')
    with pytest.raises(ValueError, match='Output|reference'):
        runner.build_plan(fixture.home, paths[target], fixture.old)


def test_visible_device_remapping_rejected(fixture, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(ValueError, match='remapping'):
        runner.build_plan(fixture.home, fixture.output, fixture.old)


@pytest.mark.parametrize('kind', ['config', 'source', 'reference', 'digest'])
def test_bindings_cannot_be_edited_to_reuse_receipts(fixture, monkeypatch, kind):
    plan = copy.deepcopy(fixture.plan)
    if kind == 'config': plan['cfg']['lr'] = .01
    elif kind == 'source': monkeypatch.setattr(runner, 'sources', lambda: {'changed': 'b'*64})
    elif kind == 'reference': (fixture.old/'poison.json').write_bytes(b'changed')
    else: plan['sha256'] = '0'*64
    with pytest.raises(ValueError, match='changed'):
        runner.verify_plan(plan)


def test_private_engine_does_not_mutate_old_suite_or_adapter():
    import run_piw_hpeli_attack_suite as old
    import piw_native_attack_adapter as adapter
    before = (old.PROFILE, old.METHODS, old.GroupedTrigger, old.PoisonedPairs,
              adapter.METHODS, copy.deepcopy(adapter.COVERS))
    runtime = runner.engine()
    assert runtime.METHODS == ('badnets',) and runtime.PROFILE == runner.PROFILE
    assert runtime.train.__code__ is not None
    assert runtime.train.__code__.co_code == old.train.__code__.co_code
    assert runtime.evaluate.__code__.co_code == old.evaluate.__code__.co_code
    assert before == (old.PROFILE, old.METHODS, old.GroupedTrigger, old.PoisonedPairs,
                      adapter.METHODS, adapter.COVERS)
    assert 'badnets' not in old.METHODS


def test_native_patch_three_groups_and_opacity_phase_identity():
    data = Data()
    x = data.x.copy()
    trigger = runner.BadNetsGroups()
    np.testing.assert_array_equal(trigger.inject(x, 0), x)
    mask = np.zeros(x.shape, dtype=bool)
    for group in range(3): mask[:, group*30+27:group*30+30, 17:20] = True
    assert mask.sum() == 3*3*3*3
    for dose in (.2, .4, .6, .8, 1.):
        actual = trigger.inject(x, dose)
        expected = x.copy()
        expected[mask] = (1-dose)*x[mask] + dose
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(actual[:, 90:], x[:, 90:])
        np.testing.assert_array_equal(actual[~mask], x[~mask])
        for group, native in enumerate(trigger.native):
            block = x[:, group*30:(group+1)*30]
            np.testing.assert_array_equal(actual[:, group*30:(group+1)*30], native.inject(block, dose, eps=.185))
    assert np.max(np.abs(trigger.inject(x, 1.) - x)) > .185
    np.testing.assert_array_equal(x, data.x)


@pytest.mark.parametrize('dose', [-1, 1.1, float('nan'), True, '0.2'])
def test_trigger_rejects_invalid_dose(dose):
    with pytest.raises(ValueError): runner.BadNetsGroups().inject(Data().x, dose)


def test_static_trigger_state_and_epsilon_are_bound():
    t = runner.BadNetsGroups()
    t.load_state_dict(t.state_dict())
    state = t.state_dict(); state['patch_subcarriers'] = 8
    with pytest.raises(ValueError): t.load_state_dict(state)
    with pytest.raises(ValueError): t.inject(Data().x, 1., eps=.1)
    assert t.state_dict() == runner.OPERATOR


def test_poison_selection_and_target_match_v3_and_no_covers():
    from piw_v2_adapter import target_pose
    ds = Data(); p = poison(ds)
    pairs = runner.BadNetsPairs(ds, runner.BadNetsGroups(), p)
    assert pairs.selected == {row['index']: row['dose'] for row in p['samples']}
    assert pairs.record['n_poison'] == 4 and pairs.record['n_cover'] == 0 and not pairs.covers
    assert pairs.record['poison_sha256'] == p['poison_plan_sha256']
    for i, d in pairs.selected.items():
        row = pairs[i]
        np.testing.assert_array_equal(row['csi'][:, 90:], ds[i]['csi'][:, 90:])
        np.testing.assert_array_equal(row['pose'], target_pose(ds[i]['pose'], d).astype(np.float32))
    np.testing.assert_array_equal(pairs[20]['csi'], ds[20]['csi'])
    np.testing.assert_array_equal(pairs[20]['pose'], ds[20]['pose'])


@pytest.mark.parametrize('bad', ['duplicate', 'wrong_name', 'wrong_dose', 'wrong_count'])
def test_poison_mismatch_is_fatal(bad):
    ds = Data(); p = poison(ds)
    if bad == 'duplicate': p['samples'][1] = p['samples'][0].copy()
    elif bad == 'wrong_name': p['samples'][0]['name'] = 'wrong'
    elif bad == 'wrong_dose': p['samples'][0]['dose'] = .1
    else: p['samples'].pop()
    with pytest.raises(ValueError): runner.BadNetsPairs(ds, runner.BadNetsGroups(), p)


def test_missing_gates_and_failed_smoke_never_launch(fixture, monkeypatch):
    monkeypatch.setattr(runner, 'require_idle', lambda: None)
    monkeypatch.setattr(runner, 'run_phase', lambda *_: pytest.fail('No worker may launch'))
    with pytest.raises(FileNotFoundError): runner.launch(fixture.plan)
    runner.write(runner.receipt_path(fixture.plan, 'preflight'), dict(kind='preflight',
        status='PASS', plan_sha256=fixture.plan['sha256'], input_fingerprint='fixture-inputs'))
    runner.write(runner.receipt_path(fixture.plan, 'gpu_smoke'), dict(kind='gpu_smoke',
        status='FAILED', plan_sha256=fixture.plan['sha256'], input_fingerprint='fixture-inputs'))
    with pytest.raises(ValueError, match='gate'): runner.launch(fixture.plan)


def test_cpu_preflight_never_checks_cuda_or_launches(fixture, monkeypatch):
    train = Data(27876)
    p = dict(samples=[dict(index=i, dose=.5, name=train.items[i]['name']) for i in range(2787)],
        poison_plan_sha256='fixture-poison')
    monkeypatch.setattr(runner, 'full_audit', lambda _: (dict(poison=p), dict(training=train)))
    monkeypatch.setattr(runner, 'require_idle', lambda: pytest.fail('CPU preflight does not probe GPU'))
    monkeypatch.setattr(runner, 'run_phase', lambda *_: pytest.fail('No training'))
    runner.preflight(fixture.plan)
    assert runner.receipt(fixture.plan, 'preflight')['n_train'] == 27876
    assert not runner.receipt_path(fixture.plan, 'gpu_smoke').exists()
    original = (fixture.output/runner.MANIFEST).read_bytes()
    runner.preflight(fixture.plan)
    assert (fixture.output/runner.MANIFEST).read_bytes() == original


def test_status_does_not_build_reference_or_import_torch(fixture, monkeypatch, capsys):
    monkeypatch.setattr(runner, 'build_plan', lambda *_: pytest.fail('No plan/Torch/GPU for status'))
    assert runner.main(['--outdir', str(fixture.output), '--mode', 'status']) == 0
    assert 'does not prove a live worker' in capsys.readouterr().out


def test_train_then_evaluate_serial_fail_stop(fixture, monkeypatch):
    events = []
    monkeypatch.setattr(runner, 'require_idle', lambda: None)
    monkeypatch.setattr(runner, 'check', lambda _: None)
    monkeypatch.setattr(runner, 'full_audit', lambda _: None)
    monkeypatch.setattr(fixture.native, 'verify_evaluation', lambda *_: None)
    monkeypatch.setattr(fixture.native, 'checkpoint', lambda *_, **__: {'epoch': 199})
    monkeypatch.setattr(runner, 'export', lambda _: events.append('export'))
    monkeypatch.setattr(runner, 'run_phase', lambda _, work: events.append(work))
    runner.launch(fixture.plan)
    assert events == ['train', 'evaluate', 'export']
    events.clear()
    def fail(_, work):
        events.append(work)
        raise RuntimeError('worker failure')
    monkeypatch.setattr(runner, 'run_phase', fail)
    with pytest.raises(RuntimeError, match='failure'): runner.launch(fixture.plan)
    assert events == ['train']


def test_prior_train_log_missing_checkpoint_blocks_silent_restart(fixture, monkeypatch):
    monkeypatch.setattr(runner, 'require_idle', lambda: None)
    monkeypatch.setattr(runner, 'check', lambda _: None)
    monkeypatch.setattr(runner, 'full_audit', lambda _: None)
    monkeypatch.setattr(fixture.native, 'verify_evaluation', lambda *_: None)
    folder = runner.cell(fixture.plan); folder.mkdir(parents=True)
    (folder/'train.1.log').write_text('epoch0')
    monkeypatch.setattr(runner, 'run_phase', lambda *_: pytest.fail('Must not restart'))
    with pytest.raises(ValueError, match='checkpoint missing'): runner.launch(fixture.plan)


def test_immutable_write_does_not_replace_failed_or_foreign_record(tmp_path):
    path = tmp_path/'receipt.json'
    runner.write(path, {'status': 'FAILED'}, immutable=True)
    with pytest.raises(ValueError, match='immutable'):
        runner.write(path, {'status': 'PASS'}, immutable=True)
    assert runner.read(path) == {'status': 'FAILED'}


def test_source_binding_includes_native_patch_target_and_launcher():
    bound = runner.sources()
    for name in ('run_piw_hpeli_badnets_4090.py', 'run_piw_hpeli_attack_suite.py',
                 'piw_v2_adapter.py', 'models/hpeli.py', 'attack/traditional.py',
                 'third_party/backdoorbench/patch.py',
                 '../remote_tools/Start-PiWHPELiBadNets4090.ps1'):
        assert len(bound[name]) == 64


def test_launcher_parse_and_no_task_or_kill():
    script = runner.HERE.parent/'remote_tools/Start-PiWHPELiBadNets4090.ps1'
    text = script.read_text(encoding='utf-8')
    assert "[string]$Mode = 'Status'" in text
    for forbidden in ('Stop-Process', 'Register-ScheduledTask', 'Invoke-CimMethod', 'Start-Process', 'AllowConcurrent'):
        assert forbidden not in text
    import subprocess
    result = subprocess.run(['powershell', '-NoProfile', '-Command',
        "$e=$null; $t=$null; [void][Management.Automation.Language.Parser]::ParseFile('"
        + str(script) + "',[ref]$t,[ref]$e); if($e.Count){$e | Out-String; exit 1}"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_hpeli_cpu_b32_step_with_four_badnets_examples():
    import torch
    from torch.utils.data import default_collate
    import run_piw_hpeli_lr_control as v3
    ds = Data(); pairs = runner.BadNetsPairs(ds, runner.BadNetsGroups(), poison(ds))
    indices = list(pairs.selected) + list(range(4, 32))
    batch = default_collate([pairs[i] for i in indices])
    cfg = v3.configuration(Path('fixture-home'), Path('fixture-home/runs/new'), device='cpu')
    model = v3.base.build_model().train()
    opt = v3.optimizer_for(model, cfg)
    before = next(model.parameters()).detach().clone()
    loss, count = v3.step(model, opt, batch, 'cpu')
    assert np.isfinite(loss) and count == 32
    assert not torch.equal(before, next(model.parameters()).detach())
    assert opt.param_groups[0]['lr'] == .001
    blob = dict(model=model.state_dict(), optimizer=opt.state_dict(), train_loss=loss,
        trigger_state=runner.BadNetsGroups().state_dict(), execution={'fixture': 'cpu'},
        rng=v3.base._rng_state('cpu'))
    # CPU schema fixture only; this is explicitly NOT a CUDA RNG/capacity proof.
    blob['rng']['cuda'] = torch.get_rng_state()
    runner.validate_checkpoint(blob, {'cfg': cfg})


def checkpoint_fixture(plan):
    import torch
    import random
    return dict(model={'weight': torch.ones(2)}, execution={'fixture': 'cpu'}, train_loss=.2,
        trigger_state=copy.deepcopy(runner.OPERATOR),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(), cuda=torch.get_rng_state()),
        optimizer=dict(param_groups=[dict(params=[0], lr=.001, weight_decay=.01,
            betas=(.9, .999), eps=1e-8)], state={0: dict(step=torch.tensor(1.),
            exp_avg=torch.ones(2), exp_avg_sq=torch.ones(2))}))


@pytest.mark.parametrize('field', ['trigger', 'rng', 'cuda_rng', 'torch_rng', 'optimizer',
    'lr', 'betas', 'sgd', 'moments', 'param_ids', 'loss', 'moment_finite'])
def test_resume_refuses_incomplete_or_foreign_state(fixture, field):
    import torch
    blob = checkpoint_fixture(fixture.plan)
    runner.validate_checkpoint(blob, fixture.plan)
    if field == 'trigger': blob['trigger_state']['patch_packets'] = 8
    elif field == 'rng': blob['rng'].pop('numpy')
    elif field == 'cuda_rng': blob['rng']['cuda'] = None
    elif field == 'torch_rng': blob['rng']['torch'] = 'missing'
    elif field == 'optimizer': blob['optimizer']['state'] = {}
    elif field == 'lr': blob['optimizer']['param_groups'][0]['lr'] = .01
    elif field == 'betas': blob['optimizer']['param_groups'][0]['betas'] = (.8, .999)
    elif field == 'sgd': blob['optimizer']['param_groups'][0]['momentum'] = .9
    elif field == 'moments': blob['optimizer']['state'][0].pop('exp_avg')
    elif field == 'param_ids': blob['optimizer']['param_groups'][0]['params'].append(1)
    elif field == 'loss': blob['train_loss'] = float('nan')
    else: blob['optimizer']['state'][0]['exp_avg_sq'][0] = float('nan')
    with pytest.raises(ValueError): runner.validate_checkpoint(blob, fixture.plan)


def test_gpu_smoke_refuses_active_mmfi_before_cuda(fixture, monkeypatch):
    import torch
    runner.write(runner.receipt_path(fixture.plan, 'preflight'), dict(kind='preflight',
        status='PASS', plan_sha256=fixture.plan['sha256'], input_fingerprint='fixture-inputs'))
    def busy(): raise ValueError('MM-Fi WiFlow active; no process stopped')
    monkeypatch.setattr(runner, 'require_idle', busy)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *_: pytest.fail('No CUDA'))
    with pytest.raises(ValueError, match='WiFlow active'): runner.smoke(fixture.plan)
    assert not runner.receipt_path(fixture.plan, 'gpu_smoke').exists()


def test_exclusive_guard_resolves_desktop_but_blocks_compute(monkeypatch):
    import run_mmfi_p1s1_trojannn as classify
    rows = '24476, [Insufficient Permissions]\n6352, [Insufficient Permissions]\n'
    conflicts = classify.classify_gpu(rows, {24476: 'python.exe', 6352: 'AnyDesk.exe'}, 99999)
    assert conflicts == [(24476, 'python.exe')]
    assert classify.classify_gpu('999, unknown\n', {}, 1) == [(999, 'unknown')]
    assert classify.classify_gpu('6352, anydesk.exe\n', {6352: 'AnyDesk.exe'}, 1) == []


def test_actual_native_200_epoch_loop_and_exact_checkpoint_resume(fixture, monkeypatch):
    """Toy CPU victim, real poisoning/AdamW/loader/RNG/checkpoint train loop."""
    import random
    import torch
    native = fixture.native
    ds = Data()
    ref = dict(poison=poison(ds))
    execution = {'fixture': 'toy CPU; not CUDA evidence'}

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(1, 14, 3))

        def forward(self, x):
            signal = x[:, :90].mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)
            return signal*self.weight[None], None

    def seed(cfg):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        return execution

    original_rng = native.v3.base._rng_state
    def rng(device):
        value = original_rng(device)
        value['cuda'] = torch.get_rng_state()  # schema fixture; no fake GPU gate
        return value

    monkeypatch.setattr(native, 'datasets', lambda _: (ref, {'training': ds}))
    monkeypatch.setattr(native, 'receipt', lambda *_: {'records': [{'execution': execution}]})
    monkeypatch.setattr(native.v3, 'seed_execution', seed)
    monkeypatch.setattr(native.v3.base, 'build_model', Toy)
    monkeypatch.setattr(native.v3.base, '_rng_state', rng)
    p1 = copy.deepcopy(fixture.plan)
    p1['cfg'].update(device='cpu', outdir=str(fixture.output/'full'))
    native.train(p1, 'badnets')
    full = native.checkpoint(p1, 'badnets', final=True)
    assert full['epoch'] == 199

    p2 = copy.deepcopy(p1); p2['cfg']['outdir'] = str(fixture.output/'resumed')
    original_save = native.v3.base.atomic_checkpoint
    def interrupt(path, blob):
        original_save(path, blob)
        if blob['epoch'] == 97: raise RuntimeError('explicit fixture interruption')
    monkeypatch.setattr(native.v3.base, 'atomic_checkpoint', interrupt)
    with pytest.raises(RuntimeError, match='fixture interruption'):
        native.train(p2, 'badnets')
    assert native.checkpoint(p2, 'badnets')['epoch'] == 97
    monkeypatch.setattr(native.v3.base, 'atomic_checkpoint', original_save)
    native.train(p2, 'badnets')
    resumed = native.checkpoint(p2, 'badnets', final=True)

    def equal(a, b):
        if torch.is_tensor(a): assert torch.equal(a, b)
        elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a: equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for v, w in zip(a, b): equal(v, w)
        else: assert a == b
    for key in ('model', 'optimizer', 'rng', 'epoch', 'train_loss', 'trigger_state', 'poison'):
        equal(full[key], resumed[key])
