"""Synthetic/CPU correctness only; NOT full-data/CUDA/attack evidence."""
import copy
from pathlib import Path
from types import SimpleNamespace
import os
import sys

import numpy as np
import pytest

from attack.bend_forward import axes, display_basis, specification, target_pose
from attack.payload import MMFI_EDGES
from attack.piw_action_targets import EDGES
import run_bend_forward as runner


def pose(dataset):
    if dataset == 'mmfi':
        return np.array([[0, 0, 1], [.12, 0, 1], [.13, 0, .55], [.14, .02, .1],
            [-.12, 0, 1], [-.13, 0, .55], [-.14, .02, .1], [0, 0, 1.2],
            [0, 0, 1.45], [0, 0, 1.6], [0, 0, 1.8], [-.2, 0, 1.45],
            [-.22, 0, 1.15], [-.23, 0, .9], [.2, 0, 1.45], [.22, 0, 1.15],
            [.23, 0, .9]], dtype=np.float64)
    return np.array([[0, 0, 1.6], [0, 0, 1.8], [-.2, 0, 1.45], [.2, 0, 1.45],
        [-.22, 0, 1.15], [-.12, 0, 1], [.22, 0, 1.15], [.12, 0, 1],
        [-.23, 0, .9], [-.13, 0, .55], [.23, 0, .9], [.13, 0, .55],
        [-.14, .02, .1], [.14, .02, .1]], dtype=np.float64)


@pytest.mark.parametrize('dataset', ['mmfi', 'piw'])
@pytest.mark.parametrize('dose', [0., .2, .4, .6, .8, 1.])
def test_bones_legs_targets_identity_and_no_mutation(dataset, dose):
    original, spec = pose(dataset), specification(dataset)
    before = original.copy()
    result = target_pose(original, dose, spec)
    edges = MMFI_EDGES if dataset == 'mmfi' else EDGES + ((5, 7),)
    for a, b in edges:
        np.testing.assert_allclose(np.linalg.norm(result[a]-result[b]),
            np.linalg.norm(original[a]-original[b]), atol=1e-12, rtol=0)
    fixed = sorted(set(range(len(original))) - set(spec['target_joints']))
    np.testing.assert_array_equal(result[fixed], original[fixed])
    np.testing.assert_array_equal(original, before)
    if dose == 0:
        np.testing.assert_array_equal(result, original)
    else:
        assert np.linalg.norm(result-original) > .1
        _, _, _, forward = axes(original, spec)
        upper = 8 if dataset == 'mmfi' else 0
        assert np.dot(result[upper]-original[upper], forward) > 0


@pytest.mark.parametrize('dataset', ['mmfi', 'piw'])
def test_exact_40_degree_schedule_monotonic_and_se3_equivariance(dataset):
    original, spec = pose(dataset), specification(dataset)
    origin, axis, up, forward = axes(original, spec)
    upper = 8 if dataset == 'mmfi' else 0
    initial = original[upper] - origin
    initial -= np.dot(initial, axis)*axis
    previous = np.zeros(len(spec['target_joints']))
    q, _ = np.linalg.qr(np.random.default_rng(42).normal(size=(3, 3)))
    q[:, 0] *= np.linalg.det(q)
    shift = np.array([2., -3., .7])
    for dose in np.linspace(0, 1, 21):
        result = target_pose(original, dose, spec)
        offset = result[upper] - origin
        offset -= np.dot(offset, axis)*axis
        angle = np.arctan2(np.dot(np.cross(initial, offset), axis), np.dot(initial, offset))
        expected_sign = 1 if dataset == 'mmfi' else -1  # independent of display axes/spec fields
        np.testing.assert_allclose(np.rad2deg(angle), expected_sign*40*dose, atol=1e-10, rtol=0)
        movement = np.linalg.norm(result[spec['target_joints']]-original[spec['target_joints']], axis=-1)
        assert np.all(movement >= previous-1e-12)
        previous = movement
        transformed = target_pose(original @ q.T + shift, dose, spec)
        np.testing.assert_allclose(transformed, result @ q.T + shift, atol=2e-12, rtol=0)
    batch = np.stack([original, original])[None]
    assert target_pose(batch, .6, spec).shape == batch.shape


def test_piw_actual_hip_axis_not_projected_or_virtual_single_anchor():
    original = pose('piw')
    original[5] += [.01, .05, .08]  # hip line not orthogonal to torso
    spec = specification('piw')
    result = target_pose(original, 1, spec)
    for a, b in EDGES:
        np.testing.assert_allclose(np.linalg.norm(result[a]-result[b]),
            np.linalg.norm(original[a]-original[b]), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(result[[5, 7, 9, 11, 12, 13]], original[[5, 7, 9, 11, 12, 13]])


@pytest.mark.parametrize('dataset', ['mmfi', 'piw'])
@pytest.mark.parametrize('dose', [-.1, 1.1, np.nan, np.inf, True, [1], 1j])
def test_invalid_dose_rejected(dataset, dose):
    with pytest.raises(ValueError):
        target_pose(pose(dataset), dose, specification(dataset))


@pytest.mark.parametrize('dataset', ['mmfi', 'piw'])
def test_bad_shape_spec_degenerate_axes_fail_not_repaired(dataset):
    p, spec = pose(dataset), specification(dataset)
    changed = copy.deepcopy(spec); changed['theta_max_deg'] = 60
    with pytest.raises(ValueError):
        target_pose(p, 1, changed)
    with pytest.raises(ValueError):
        target_pose(p[:-1], 1, spec)
    left, right = spec['hip_axis']; p[left] = p[right]
    with pytest.raises(ValueError, match='hip axis'):
        target_pose(p, 1, spec)


@pytest.mark.parametrize('dataset', ['mmfi', 'piw'])
def test_plan_native_recipe_new_target_and_no_clean(tmp_path, monkeypatch, dataset):
    import run_mmfi_a1_motion_priors_4090 as prior
    import run_piw_action_targets as piw
    monkeypatch.setattr(prior, 'action_record', lambda *a: {'sha256': 'prior'})
    monkeypatch.setattr(prior, 'trigger_record', lambda *a: {'pattern': 'native'})
    monkeypatch.setattr(piw, 'trigger_record', lambda *a, **k: {'pattern': 'native', **k})
    monkeypatch.setattr(runner, 'source_hashes', lambda: {'test': 'source'})
    args = SimpleNamespace(dataset=dataset, data_home=str(tmp_path/'data'),
        outdir=str(tmp_path/'runs'/f'{dataset}_bend_forward_s42_v2'), dataset_root=None)
    plan = runner.build_plan(args)
    cfg = plan['cells'][0]['cfg']
    spec = cfg['action_payload'] if dataset == 'mmfi' else cfg['payload']
    assert spec == specification(dataset)
    assert plan['profile'] == f'{dataset}_hpeli_a1_bend_forward_s42_v2'
    assert plan['contract']['signed_angle_deg'] == (40 if dataset == 'mmfi' else -40)
    assert len(plan['cells']) == 1 and plan['contract']['clean_training'] is False
    assert plan['contract']['clean_weight_import'] is False
    assert plan['contract']['doses'] == [0, .2, .4, .6, .8, 1]
    assert cfg['seed'] == 42 and cfg['rho'] == .1
    if dataset == 'mmfi':
        assert cfg['epochs'] == 50 and cfg['batch_size'] == 32
        assert cfg['lr'] == .001 and cfg['num_workers'] == 4
    else:
        assert cfg['epochs'] == 200 and cfg['final_epoch'] == 199
        assert cfg['lr'] == .001 and cfg['weight_decay'] == .01
        assert cfg['drop_last'] is True and cfg['num_workers'] == 0


def test_mmfi_explicit_bend_adapter_native_poison_parity():
    from attack.poison import PoisonedDataset
    import run_mmfi_action_targets as mmfi
    from run_a1_three_tables import _configuration
    class Base:
        items = [{'csi': str(i), 'kpt': str(i), 'frame_idx': 0} for i in range(40)]
        def __len__(self): return len(self.items)
        def load_pose(self, *a): return pose('mmfi')[None].astype(np.float32)
        def load_raw(self, *a): return np.full((3, 114, 10), .5, np.float32)
        def normalize(self, raw): return raw
    class Trigger:
        def inject(self, raw, dose, eps): return raw * (1 + eps*dose)
    cfg = _configuration(Path('synthetic'), 'cpu', 0, 42, 'a1')
    cfg.update(action_target_name=runner.ACTION, action_payload=specification('mmfi'),
        pivot=0, target_joints=list(range(7, 17)))
    data = mmfi.action_data(Base(), Trigger(), cfg)
    native = PoisonedDataset(Base(), Trigger(), mode='train', rho=.1, dose_min=.2,
        dose_max=1, eps=.185, seed=42, select='uniform', dataset='mmfi', dose_coupling='paired')
    assert data.poison_plan == native.poison_plan
    for index, dose in data.poison_plan:
        item = data[index]
        assert set(item) == {'csi', 'pose'}
        np.testing.assert_array_equal(item['pose'], target_pose(Base().load_pose(index), dose, specification('mmfi')))
        np.testing.assert_array_equal(item['csi'], native[index]['csi'])


def test_piw_explicit_bend_adapter_native_poison_and_phase_parity():
    from piw_action_target_data import ActionTargetPairs
    from piw_a1_adapter import PoisonedPairs
    class Base:
        items = [{'name': f'synthetic{i}', 'kpt': str(i)} for i in range(40)]
        def __len__(self): return len(self.items)
        def __getitem__(self, index):
            return dict(csi=np.full((3, 180, 20), .5, np.float32), pose=pose('piw')[None].astype(np.float32))
        def load_pose(self, *a): return pose('piw')[None].astype(np.float32)
    class Trigger:
        def inject(self, csi, dose, eps):
            result = csi.copy(); result[:, :90] *= 1 + dose*eps; return result
    data = ActionTargetPairs(Base(), Trigger(), runner.ACTION, payload=specification('piw'), target_fn=target_pose)
    native = PoisonedPairs(Base(), Trigger())
    assert data.selected == native._selected
    for index, dose in data.selected.items():
        item, original = data[index], Base()[index]
        np.testing.assert_array_equal(item['csi'], native[index]['csi'])
        np.testing.assert_array_equal(item['csi'][:, 90:], original['csi'][:, 90:])
        np.testing.assert_array_equal(item['pose'], target_pose(original['pose'], dose, specification('piw')).astype(np.float32))
        assert set(item) == {'csi', 'pose'}


def process(pid, parent=0, command=''):
    return dict(ProcessId=pid, ParentProcessId=parent, CreationDate='frozen',
        ExecutablePath='python.exe', CommandLine=command)


@pytest.mark.parametrize('value', ['999999', 'N/A', 'not a PID'])
def test_gpu_guard_rejects_foreign_or_unenumerable_processes(monkeypatch, value):
    monkeypatch.setattr(runner, 'process_snapshot', lambda: [])
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=value))
    with pytest.raises(ValueError): runner.gpu_guard()


def test_gpu_guard_allows_only_named_peer_and_own_children(monkeypatch):
    peer = 12345 if os.getpid() != 12345 else 12346
    rows = [process(peer, command='python run_a1_wiflow.py --mode launch'),
        process(12347, peer), process(12348, os.getpid())]
    monkeypatch.setattr(runner, 'process_snapshot', lambda: rows)
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=f'{peer}\n12347\n12348'))
    assert runner.gpu_guard(peer)['peer'] == rows[0]
    rows[0]['CommandLine'] = 'python unrelated.py'
    with pytest.raises(ValueError, match='not a recognized'): runner.gpu_guard(peer)


def test_helper_serial_order_and_no_process_mutation():
    source = (Path(__file__).resolve().parents[2]/'remote_tools/bend_forward/Run-BendForward.ps1').read_text()
    assert "foreach ($bendDataset in @('mmfi','piw'))" in source
    assert 'Start-Process' not in source and 'Stop-Process' not in source
    assert 'ApproveGeometry' in source and 'if ($LASTEXITCODE -ne 0)' in source
    assert 'U:\\backdooranalog' in source
    assert '[ValidateRange(2,9999)][int]$RunVersion = 2' in source


def test_check_only_never_imports_cuda_or_writes_output(tmp_path, capsys):
    home = tmp_path/'home'; (home/'actions').mkdir(parents=True)
    (home/'actions/data_bend.npy').touch(); (home/'datasets/Compress').mkdir(parents=True)
    out = tmp_path/'uncreated'
    runner.main(['--mode', 'check-only', '--dataset', 'mmfi', '--data-home', str(home), '--outdir', str(out)])
    assert not out.exists() and 'CHECK_ONLY' in capsys.readouterr().out


@pytest.mark.parametrize('dataset', ['mmfi', 'piw'])
def test_bend_profile_exact_optimizer_rng_resume(tmp_path, dataset):
    import torch
    from torch.utils.data import DataLoader
    import run_piw_hpeli_clean as piw_base
    import train_backdoor as mmfi_native
    from run_a1_three_tables import _configuration
    torch.set_num_threads(2)
    module = runner.module_for(dataset)
    context = {'cpu_synthetic_only': True}
    epochs = 50 if dataset == 'mmfi' else 200
    cfg = (_configuration(tmp_path/'data', 'cpu', 0, 42, 'a1') if dataset == 'mmfi'
        else piw_base.configuration(tmp_path/'data', tmp_path/'output', 'cpu'))
    cfg.update(action=runner.ACTION, action_target_name=runner.ACTION, batch_size=4,
        drop_last=True, payload=specification(dataset), action_payload=specification(dataset))
    joints = len(pose(dataset))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.drop = torch.nn.Dropout(.2); self.linear = torch.nn.Linear(3, joints*3)
        def forward(self, value):
            return self.linear(self.drop(value)).reshape(-1, 1, joints, 3), None
    samples = [dict(csi=np.array([i*.03, .4, -.1], np.float32),
        pose=target_pose(pose(dataset), .6, specification(dataset))[None].astype(np.float32)) for i in range(8)]
    results = []
    for name, interrupted in [('full', False), ('resumed', True)]:
        folder = tmp_path/name; folder.mkdir()
        runner.write(folder/'inputs.json', {'synthetic': True})
        runner.write(folder/'prepared.json', {'execution': context})
        plan = dict(outdir=str(folder), profile=f'{dataset}_hpeli_a1_bend_forward_s42_v2', plan_sha256='BEND_TEST_ONLY')
        if dataset == 'mmfi': module.seed_all()
        else: module.seed_all('cpu')
        model = Tiny()
        optimizer = (mmfi_native._build_optimizer(model, cfg, 'mmfi')[1] if dataset == 'mmfi'
            else module.optimizer_for(model, cfg))
        loader = DataLoader(samples, batch_size=4, shuffle=True, drop_last=True)
        path = folder/'checkpoint.pt'
        if interrupted:
            module.train_epochs(model, optimizer, loader, cfg, path, plan, context, device='cpu', stop=epochs//2)
            saved = module.checkpoint(path, cfg, plan)
            assert saved['profile'] == plan['profile']
            if dataset == 'mmfi': module.seed_all()
            else: module.seed_all('cpu')
            model = Tiny()
            optimizer = (mmfi_native._build_optimizer(model, cfg, 'mmfi')[1] if dataset == 'mmfi'
                else module.optimizer_for(model, cfg))
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            if dataset == 'mmfi': mmfi_native._restore_rng_state(saved['rng'])
            else: piw_base._restore_rng(saved['rng'], 'cpu')
            module.train_epochs(model, optimizer, loader, cfg, path, plan, context, device='cpu', start=epochs//2)
        else:
            module.train_epochs(model, optimizer, loader, cfg, path, plan, context, device='cpu')
        saved = module.checkpoint(path, cfg, plan)
        assert saved['epoch'] == epochs-1 and saved['profile'] == plan['profile']
        with pytest.raises(ValueError, match='Foreign'):
            module.checkpoint(path, cfg, dict(plan, profile='old_leg_rotation'))
        with pytest.raises(ValueError, match='Foreign'):
            module.checkpoint(path, cfg, dict(plan, profile=f'{dataset}_hpeli_a1_bend_forward_s42_v1'))
        results.append(copy.deepcopy(model.state_dict()))
    for key in results[0]:
        torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)


@pytest.mark.parametrize('dataset', ['mmfi', 'piw'])
def test_real_hpeli_b32_new_bend_cpu_update(dataset):
    import torch
    import train_backdoor as native
    import run_piw_hpeli_clean as piw_base
    from run_piw_hpeli_lr_control import step
    from run_a1_three_tables import _configuration
    torch.set_num_threads(2)
    spec = specification(dataset)
    truth = pose(dataset)
    labels = np.stack([target_pose(truth, .6 if i < 8 else 0, spec)[None] for i in range(32)]).astype(np.float32)
    shape = (3, 114, 10) if dataset == 'mmfi' else (3, 180, 20)
    csi = torch.from_numpy(np.random.default_rng(42).uniform(size=(32,)+shape).astype(np.float32))
    model = (native.build_model('hpeli', num_keypoints=17, subcarrier_num=114,
        dataset='mmfi', pretrained=False) if dataset == 'mmfi' else piw_base.build_model())
    before = next(model.parameters()).detach().clone()
    if dataset == 'mmfi':
        cfg = _configuration(Path('synthetic'), 'cpu', 0, 42, 'a1')
        optimizer = native._build_optimizer(model, cfg, 'mmfi')[1]
        loss = native._victim_update(model, optimizer, csi, torch.from_numpy(labels))
    else:
        cfg = piw_base.configuration('synthetic', 'synthetic_output', 'cpu')
        optimizer = runner.module_for('piw').optimizer_for(model, cfg)
        loss, count = step(model, optimizer, dict(csi=csi, pose=torch.from_numpy(labels)), 'cpu')
        assert count == 32
    assert np.isfinite(loss) and not torch.equal(before, next(model.parameters()))
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize('dataset', ['mmfi', 'piw'])
def test_full_poison_train_geometry_and_explicit_approval_gate(tmp_path, monkeypatch, dataset):
    """Real Bend adapter/audit/NPZ/side-view plot on a tiny SYNTHETIC TRAIN set."""
    from run_a1_three_tables import _configuration
    import run_piw_hpeli_clean as piw_base
    module = runner.module_for(dataset)
    root = tmp_path/'output'; root.mkdir()
    label_file = tmp_path/'SYNTHETIC_TRAIN_ONLY.npy'
    np.save(label_file, pose(dataset)[None].astype(np.float32))
    spec = specification(dataset)
    cfg = (_configuration(tmp_path/'data', 'cpu', 0, 42, 'a1') if dataset == 'mmfi'
        else piw_base.configuration(tmp_path/'data', root, 'cpu'))
    cfg.update(action=runner.ACTION, action_target_name=runner.ACTION,
        action_payload=spec, payload=spec, pivot=0, target_joints=spec['target_joints'])
    class Base:
        items = [dict(csi='NEVER_OPEN_CSI', kpt=str(label_file), frame_idx=0, name=f'synthetic{i}') for i in range(40)]
        def __len__(self): return len(self.items)
        def load_pose(self, *a): return np.load(label_file, allow_pickle=False)
        def load_raw(self, *a): raise AssertionError('Geometry must not read CSI')
    ids = [item['name'] for item in Base.items]
    def dataset_for(cfg, split):
        assert split == 'training', 'Geometry must NEVER open TEST'
        return Base(), ids
    monkeypatch.setattr(module, 'dataset', dataset_for)
    plan = dict(profile=f'SYNTHETIC_{dataset}_BEND', dataset=dataset, outdir=str(root),
        cells=[dict(action=runner.ACTION, cfg=cfg)], plan_sha256='SYNTHETIC_ONLY')
    monkeypatch.setattr(runner, 'build_plan', lambda args: plan)
    runner.geometry(SimpleNamespace(dataset=dataset))
    audit = runner.read(root/'geometry_audit.json')
    assert audit['status'] == 'PASS' and audit['n_checked'] == 4 and audit['n_failed'] == 0
    assert audit['records'][runner.ACTION]['max_bone_delta_m'] < 1e-12
    assert audit['records'][runner.ACTION]['decreasing_joint_displacement_steps'] == 0
    record = module.verify_geometry(plan)
    assert len(record['files']) == 3
    with np.load(root/'geometry_bend_forward.npz', allow_pickle=False) as saved:
        assert saved['gt'].shape == (3, len(pose(dataset)), 3)
        np.testing.assert_array_equal(saved['targets'][0], saved['gt'])
    runner.verify_train_pose_hashes(plan)
    with pytest.raises(ValueError, match='approve-geometry'):
        module.prepare(SimpleNamespace(approve_geometry=False), plan,
            idle_check=lambda: pytest.fail('No GPU check before geometry approval'))
    # Bind actual TRAIN label bytes, not just names/pose shapes.
    np.save(label_file, (pose(dataset)[None]+.001).astype(np.float32))
    with pytest.raises(ValueError, match='TRAIN pose changed'):
        runner.verify_train_pose_hashes(plan)


@pytest.mark.parametrize('mode,resume', [('launch', False), ('resume', True), ('evaluate', None)])
def test_stage_routes_only_one_new_victim_and_then_evaluation(tmp_path, monkeypatch, mode, resume):
    events = []
    peer = process(23456, command='python run_a1_wiflow.py')
    cell = dict(action=runner.ACTION, cfg={'new_bend_only': True})
    plan = dict(dataset='mmfi', outdir=str(tmp_path), cells=[cell], plan_sha256='new-bend')
    runner.write(tmp_path/'coexistence.json', dict(plan_sha256='new-bend', peer_pid=23456,
        evidence=dict(peer=peer)))
    fake = SimpleNamespace(
        verify_prepared=lambda p, **k: (events.append(('verify', k)), {'execution': 'CPU_TEST'})[1],
        execution=lambda: 'CPU_TEST',
        train_cell=lambda p, c, **k: events.append(('train', c, k['resume'])),
        evaluate_cell=lambda p, c: events.append(('evaluate', c)),
        export=lambda p: events.append(('export',)))
    monkeypatch.setattr(runner, 'module_for', lambda d: fake)
    monkeypatch.setattr(runner, 'load_plan', lambda a: plan)
    monkeypatch.setattr(runner, 'verify_train_pose_hashes', lambda p: None)
    monkeypatch.setattr(runner, 'gpu_guard', lambda pid: dict(peer=peer))
    runner.run_stage(SimpleNamespace(dataset='mmfi', mode=mode, peer_pid=23456))
    if resume is None:
        assert not any(e[0] == 'train' for e in events)
    else:
        assert [e for e in events if e[0] == 'train'] == [('train', cell, resume)]
    assert events[-2:] == [('evaluate', cell), ('export',)]
    assert events[0] == ('verify', {'full_hash': True})


def test_recycled_peer_identity_refused_before_training(tmp_path, monkeypatch):
    expected = process(23456, command='python run_a1_wiflow.py')
    runner.write(tmp_path/'coexistence.json', dict(plan_sha256='new', peer_pid=23456,
        evidence=dict(peer=expected)))
    plan = dict(dataset='mmfi', outdir=str(tmp_path), cells=[{}], plan_sha256='new')
    changed = dict(expected, CreationDate='different-process')
    monkeypatch.setattr(runner, 'module_for', lambda d: SimpleNamespace())
    monkeypatch.setattr(runner, 'load_plan', lambda a: plan)
    monkeypatch.setattr(runner, 'gpu_guard', lambda pid: dict(peer=changed))
    with pytest.raises(ValueError, match='identity changed'):
        runner.run_stage(SimpleNamespace(dataset='mmfi', mode='launch', peer_pid=23456))


def test_main_geometry_writes_plan_before_own_logs_not_unrelated_output(tmp_path, monkeypatch):
    root = tmp_path/'runs/mmfi_bend_forward_s42_v2'
    plan = dict(outdir=str(root), profile='SYNTHETIC_ONLY', plan_sha256='test')
    monkeypatch.setattr(runner, 'build_plan', lambda a: plan)
    def stage(args):
        assert runner.read(root/runner.PLAN_NAME) == plan
        assert len(list((root/'logs').glob('geometry.*.log'))) == 1
    monkeypatch.setattr(runner, 'run_stage', stage)
    runner.main(['--dataset', 'mmfi', '--mode', 'geometry', '--data-home', str(tmp_path/'home'),
        '--outdir', str(root)])


def test_mmfi_spec_and_actual_rotation_unchanged_from_published_v1():
    expected = dict(version='bend_forward_mmfi_v1', dataset='mmfi',
        kind='rigid_upper_body_hip_axis', pivots=[0], target_joints=list(range(7, 17)),
        hip_axis=[4, 1], origin=[0], theta_max_deg=40.,
        forward_convention='cross(left_hip-right_hip, pelvis_to_thorax)',
        note='Pelvis and both legs fixed; new trunk-flexion payload.')
    assert specification('mmfi') == expected
    source = pose('mmfi'); source[4] += [.02, .01, .03]
    axis = source[4]-source[1]; axis /= np.linalg.norm(axis)
    for dose in [0., .2, .4, .6, .8, 1.]:
        angle = np.deg2rad(40*dose)
        reference = source.copy()
        for j in range(7, 17):
            v = source[j]-source[0]
            reference[j] = source[0] + (np.cos(angle)*v + np.sin(angle)*np.cross(axis, v)
                + (1-np.cos(angle))*np.dot(axis, v)*axis)
        np.testing.assert_allclose(target_pose(source, dose, expected), reference, atol=2e-15, rtol=0)


def test_piw_v2_actual_flexion_is_toward_independently_declared_knee_side():
    # Synthetic coordinates: superior +z, anterior -y, actual left-minus-right hip axis -x.
    # Knees bend toward -y. Do NOT obtain anterior from the operator's forward vector.
    source = pose('piw')
    source[[9, 11], 1] = -.12
    source[[12, 13], 1] = 0.
    anterior = np.array([0., -1., 0.])
    old = specification('piw', version='bend_forward_piw_v1')
    current = specification('piw')
    assert current['version'] == 'bend_forward_piw_v2'
    for dose in [.2, .4, .6, .8, 1.]:
        result = target_pose(source, dose, current)
        prior = target_pose(source, dose, old)
        assert (result[0]-source[0]) @ anterior > 0
        assert (prior[0]-source[0]) @ anterior < 0
        # Independent Cartesian expectation for rotating neck about -x by -40d.
        angle = np.deg2rad(40*dose)
        np.testing.assert_allclose(result[0], [0, -.6*np.sin(angle), 1+.6*np.cos(angle)],
            atol=1e-12, rtol=0)
        np.testing.assert_array_equal(result[[5, 7, 9, 11, 12, 13]], source[[5, 7, 9, 11, 12, 13]])


def test_explicit_legacy_spec_replay_and_no_implicit_migration():
    current = specification('piw')
    old = specification('piw', version='bend_forward_piw_v1')
    assert old['version'] == 'bend_forward_piw_v1' and 'rotation_sign' not in old
    source = pose('piw')
    result = target_pose(source, 1., old)
    np.testing.assert_allclose(result[0], [0, .6*np.sin(np.deg2rad(40)),
        1+.6*np.cos(np.deg2rad(40))], atol=1e-12, rtol=0)
    assert not np.allclose(result, target_pose(source, 1., current))
    old['note'] = 'mutated copy'
    assert specification('piw', version='bend_forward_piw_v1')['note'] != old['note']
    for dataset, version in [('mmfi', 'bend_forward_piw_v1'), ('piw', 'v999')]:
        with pytest.raises(ValueError, match='Unknown frozen'):
            specification(dataset, version=version)


def test_preview_forward_sign_never_flips_body_up_even_with_tilted_hips():
    source = pose('piw'); source[5] += [.01, .05, .08]
    old = specification('piw', version='bend_forward_piw_v1')
    current = specification('piw')
    origin, old_basis = display_basis(source, old)
    origin2, new_basis = display_basis(source, current)
    np.testing.assert_array_equal(origin, origin2)
    np.testing.assert_array_equal(old_basis[:, 1], new_basis[:, 1])
    np.testing.assert_array_equal(old_basis[:, 0], -new_basis[:, 0])
    np.testing.assert_allclose(new_basis.T @ new_basis, np.eye(2), atol=1e-12, rtol=0)
    raw_up = .5*(source[2]+source[3])-origin
    assert raw_up @ new_basis[:, 1] > 0
    assert target_pose(source, 1., current)[0] @ new_basis[:, 1] != source[0] @ new_basis[:, 1]
    q, _ = np.linalg.qr(np.random.default_rng(84).normal(size=(3, 3)))
    q[:, 0] *= np.linalg.det(q)
    shift = np.array([2., -3., .7])
    transformed_origin, transformed_basis = display_basis(source @ q.T + shift, current)
    np.testing.assert_allclose(transformed_origin, origin @ q.T+shift, atol=1e-12, rtol=0)
    np.testing.assert_allclose(transformed_basis, q @ new_basis, atol=1e-12, rtol=0)


@pytest.mark.parametrize('dataset', ['mmfi', 'piw'])
@pytest.mark.parametrize('version', [0, 1])
def test_v1_output_root_rejected_before_any_write(tmp_path, dataset, version):
    root = tmp_path/'runs'/f'{dataset}_bend_forward_s42_v{version}'
    with pytest.raises(ValueError, match='preserve v1'):
        runner.main(['--dataset', dataset, '--mode', 'geometry', '--data-home', str(tmp_path/'home'),
            '--outdir', str(root)])
    assert not root.parent.exists()


def test_archived_plan_untouched_and_refused_not_migrated(tmp_path):
    root = tmp_path/'runs/piw_bend_forward_s42_v1'; root.mkdir(parents=True)
    frozen = dict(profile='piw_hpeli_a1_bend_forward_s42_v1',
        payload=specification('piw', version='bend_forward_piw_v1'))
    runner.write(root/runner.PLAN_NAME, frozen)
    before = (root/runner.PLAN_NAME).read_bytes()
    with pytest.raises(ValueError, match='preserve v1'):
        runner.load_plan(SimpleNamespace(dataset='piw', data_home=str(tmp_path/'home'),
            outdir=str(root), dataset_root=None))
    assert (root/runner.PLAN_NAME).read_bytes() == before
    assert sorted(p.name for p in root.iterdir()) == [runner.PLAN_NAME]
