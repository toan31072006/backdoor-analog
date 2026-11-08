"""Local CPU/synthetic correctness tests, NOT PiW efficacy or real-data results."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import run_piw_action_targets as runner
import run_piw_hpeli_clean as native
from attack.piw_action_targets import EDGES, body_frame, specification, target_pose
from piw_action_target_data import ActionTargetPairs
from piw_a1_adapter import PoisonedPairs, build_trigger


def pose():
    return np.array([[0, 0, 1.60], [0, 0, 1.80],
        [.20, 0, 1.45], [-.20, 0, 1.45], [.22, .01, 1.15],
        [.10, 0, .95], [-.22, .02, 1.15], [-.10, 0, .95],
        [.24, .02, .86], [.11, 0, .55], [-.24, .03, .86],
        [-.11, 0, .55], [.12, 0, .12], [-.12, 0, .12]], dtype=np.float64)


def bones(p):
    return np.array([np.linalg.norm(p[a]-p[b]) for a, b in EDGES])


@pytest.mark.parametrize('action', ['cross', 'nod'])
@pytest.mark.parametrize('dose', runner.GRID)
def test_actual_piw_layout_bones_identity_and_untouched(action, dose):
    p = pose(); original = p.copy(); spec = specification(action)
    result = target_pose(p, dose, spec)
    np.testing.assert_array_equal(p, original)
    np.testing.assert_allclose(bones(result), bones(p), atol=1e-12, rtol=0)
    others = [i for i in range(14) if i not in spec['target_joints']]
    np.testing.assert_array_equal(result[others], p[others])
    if dose == 0:
        np.testing.assert_array_equal(result, p)
    else:
        assert np.linalg.norm(result-p) > .01


def test_cross_wrist_semantics_and_fixed_shoulders():
    p = pose(); q = target_pose(p, 1., specification('cross'))
    lateral, _, forward = body_frame(p)
    center = .5*(p[2]+p[3])
    assert (q[10]-center) @ lateral > 0 and (q[8]-center) @ lateral < 0
    assert (q[10]-center) @ forward > 0 and (q[8]-center) @ forward > 0
    np.testing.assert_array_equal(q[[0, 1, 2, 3, 5, 7, 9, 11, 12, 13]], p[[0, 1, 2, 3, 5, 7, 9, 11, 12, 13]])


def test_nod_around_piw_neck_not_old_hip():
    p = pose(); q = target_pose(p, 1., specification('nod'))
    _, _, forward = body_frame(p)
    a, b = p[1]-p[0], q[1]-q[0]
    np.testing.assert_allclose(np.rad2deg(np.arccos(a@b/np.linalg.norm(a)/np.linalg.norm(b))), 40., atol=1e-10)
    assert b @ forward > a @ forward
    np.testing.assert_array_equal(q[[11, 13]], p[[11, 13]])


@pytest.mark.parametrize('action', ['cross', 'nod'])
def test_batched_rigid_coordinate_equivariance_and_determinism(action):
    p = pose(); spec = specification(action)
    theta = .76
    rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    translation = np.array([2., -3., .8])
    expected = target_pose(p, .6, spec) @ rotation.T + translation
    actual = target_pose(p @ rotation.T + translation, .6, spec)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    result = target_pose(np.stack([p, p])[None], .6, spec)
    assert result.shape == (1, 2, 14, 3)
    np.testing.assert_array_equal(result[0, 0], result[0, 1])


@pytest.mark.parametrize('dose', [-.1, 1.1, np.nan, np.inf, True, '0.2', [0.2], 1j])
def test_invalid_dose_refused(dose):
    with pytest.raises(ValueError):
        target_pose(pose(), dose, specification('nod'))


@pytest.mark.parametrize('kind', ['shape', 'complex', 'nonfinite', 'shoulder', 'torso', 'head', 'arm', 'reach', 'spec'])
def test_malformed_geometry_refused_not_repaired_or_dropped(kind):
    p = pose(); action = 'nod' if kind == 'head' else 'cross'; spec = specification(action)
    if kind == 'shape': p = p[:13]
    if kind == 'complex': p = p.astype(complex)
    if kind == 'nonfinite': p[0, 0] = np.nan
    if kind == 'shoulder': p[2] = p[3]
    if kind == 'torso': p[[5, 7]] += [0., 0., .5]
    if kind == 'head': p[1] = p[0]
    if kind == 'arm': p[6] = p[3]
    if kind == 'reach': p[6] = p[3]+[0, 0, -.02]; p[10] = p[6]+[0, 0, -.02]
    if kind == 'spec': spec['wrist_across_width'] = .01
    with pytest.raises(ValueError):
        target_pose(p, 1., spec)


class TinyBase:
    def __init__(self, count=40):
        self.items = [{'name': f'S11_01_{i:05d}', 'kpt': str(i)} for i in range(count)]
        self.raw = np.random.default_rng(71).normal(size=(3, 180, 20)).astype(np.float32)

    def __len__(self): return len(self.items)

    def load_pose(self, path): return pose()[None].astype(np.float32)

    def __getitem__(self, index):
        from data_utils.feeder import PersonInWiFi3D
        return dict(csi=PersonInWiFi3D.normalize(self.raw), pose=self.load_pose(''), name=self.items[index]['name'])


def action_file(path, offset=0.):
    value = np.random.default_rng(42).normal(size=(2, 3, 8, 25, 1)).astype(np.float32) + offset
    np.save(path, value)
    return path


@pytest.mark.parametrize('action', ['cross', 'nod'])
def test_same_native_poison_plan_amp_once_phase_untouched_and_new_labels(tmp_path, action):
    trigger = build_trigger(action_file(tmp_path / 'prior.npy'))
    base = TinyBase()
    native_pairs = PoisonedPairs(base, trigger)
    pairs = ActionTargetPairs(base, trigger, action)
    assert pairs.manifest()['poison_plan_sha256'] == native_pairs.manifest()['poison_plan_sha256']
    assert pairs.manifest()['samples'] == native_pairs.manifest()['samples']
    assert 'pivot' not in pairs.manifest() and pairs.manifest()['payload'] == specification(action)
    index = next(iter(pairs.selected)); dose = pairs.selected[index]
    before = base[index]; after = pairs[index]
    assert set(after) == {'csi', 'pose'}
    np.testing.assert_array_equal(after['csi'][:, 90:], before['csi'][:, 90:])
    np.testing.assert_array_equal(after['csi'], trigger.inject(before['csi'], dose))
    np.testing.assert_array_equal(after['pose'], target_pose(before['pose'], dose, specification(action)).astype(np.float32))
    assert not np.array_equal(after['pose'], native_pairs[index]['pose'])
    other = next(i for i in range(len(base)) if i not in pairs.selected)
    np.testing.assert_array_equal(pairs[other]['pose'], base[other]['pose'])
    np.testing.assert_array_equal(pairs[other]['csi'], base[other]['csi'])


@pytest.mark.parametrize('action', ['cross', 'nod'])
def test_metrics_perfect_still_reverse_and_npz_replay(tmp_path, action):
    gt = np.stack([pose(), pose()+[.2, -.1, .1]])
    spec = specification(action); joints = spec['target_joints']
    targets = np.stack([target_pose(gt, d, spec) for d in runner.GRID])
    exact = runner.metrics_from_arrays(gt, targets, targets, joints)
    assert exact['mean_positive_dce_mm'] == 0 and exact['clean_mpjpe_mm'] == 0
    assert all(row['target_pck_pct'] == 100 for row in exact['doses'])
    still = np.repeat(gt[None], 6, axis=0)
    static = runner.metrics_from_arrays(gt, targets, still, joints)
    reverse = runner.metrics_from_arrays(gt, targets, 2*still-targets, joints)
    for a, b in zip(static['doses'], reverse['doses']):
        np.testing.assert_allclose(a['dce_mm'], a['requested_mm'], atol=1e-12)
        np.testing.assert_allclose(b['dce_mm'], 2*a['requested_mm'], atol=1e-10)
    artifact = tmp_path / 'arrays.npz'
    runner.write_npz(artifact, gt=gt, target=targets, prediction=2*still-targets)
    with np.load(artifact, allow_pickle=False) as saved:
        assert runner.metrics_from_arrays(saved['gt'], saved['target'], saved['prediction'], joints) == reverse


def test_pck_scale_uses_clean_elbows_not_target_elbows_or_mmfi_indices():
    gt = np.stack([pose(), pose()])
    target = np.repeat(gt[None], 6, axis=0)
    target[1:, :, 6, 0] += 3
    pred = target.copy(); pred[:, :, 1, 0] += .1
    # Clean elbow width .44 -> threshold .088, so .1 is a miss, even when
    # target elbows are deliberately far apart. MM-Fi's (5,12) is not used.
    scores = runner.metrics_from_arrays(gt, target, pred, [1])
    assert all(r['target_pck_pct'] == 0. for r in scores['doses'])


def args_for(tmp_path):
    home = tmp_path / 'data home'; home.mkdir()
    actions = home / 'actions'; actions.mkdir()
    action_file(actions / 'data_cross.npy')
    action_file(actions / 'data_nod.npy', .1)
    return runner.parser().parse_args(['--data-home', str(home), '--outdir',
        str(home / 'runs/piw_action_targets_cross_nod_s42_v1'), '--mode', 'geometry', '--approve-action-labels'])


def test_frozen_piw_v3_recipe_not_mmfi_or_v2(tmp_path):
    args = args_for(tmp_path)
    plan = runner.build_plan(args)
    a, b = plan['cells']
    for cell in (a, b):
        cfg = cell['cfg']
        assert (cfg['epochs'], cfg['final_epoch'], cfg['batch_size'], cfg['num_workers']) == (200, 199, 32, 0)
        assert cfg['optimizer'] == 'adamw' and cfg['lr'] == .001 and cfg['weight_decay'] == .01
        assert cfg['drop_last'] is True and cfg['pretrained'] is False
        assert cfg['target_pck_reference'] == [6, 4]
        assert cfg['csi_shape'] == [3, 180, 20]
    assert a['cfg']['payload']['target_joints'] == [6, 10, 4, 8]
    assert b['cfg']['payload']['target_joints'] == [1]
    assert a['action_record']['sha256'] != b['action_record']['sha256']
    runner.write(Path(args.outdir) / runner.PLAN_NAME, plan)
    assert runner.load_plan(args) == plan
    args.nod_action = args.cross_action = a['cfg']['action_npy']
    with pytest.raises(ValueError, match='identical'):
        runner.build_plan(args)


def test_no_geometry_approval_no_smoke_or_training(tmp_path):
    args = args_for(tmp_path)
    with pytest.raises(ValueError, match='approve-geometry'):
        runner.prepare(args, {'outdir': args.outdir})
    assert not Path(args.outdir).exists()


def test_unrelated_output_refused_and_own_lock_allowed(tmp_path):
    args = args_for(tmp_path)
    root = Path(args.outdir); root.mkdir(parents=True)
    (root / 'run.lock').write_text('own lock')
    runner.build_plan(args)
    (root / 'historical.txt').write_text('preserve')
    with pytest.raises(ValueError, match='unrelated'):
        runner.build_plan(args)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear = torch.nn.Linear(2, 42); self.drop = torch.nn.Dropout(.2)

    def forward(self, x): return self.linear(self.drop(x)).reshape(-1, 1, 14, 3), None


class TinyTraining:
    def __len__(self): return 8
    def __getitem__(self, i):
        return dict(csi=np.array([i*.03, -.2], dtype=np.float32), pose=(pose()[None]+i*.001).astype(np.float32))


def test_200_epoch_adamw_exact_optimizer_rng_resume(tmp_path, capsys):
    args = args_for(tmp_path)
    cfg = runner.configuration(args, 'nod', Path(args.data_home) / 'actions/data_nod.npy')
    cfg['batch_size'] = 4  # CPU test-only fixture, not a CLI or scientific recipe override
    root = tmp_path / 'fixture'; root.mkdir()
    plan = dict(plan_sha256='synthetic-test', outdir=str(root))
    runner.write(root / 'inputs.json', {'synthetic': True})
    runner.write(root / 'prepared.json', {'execution': {'device': 'cpu'}})
    context = {'device': 'cpu'}
    def setup():
        runner.seed_all('cpu'); model = TinyModel(); opt = runner.optimizer_for(model, cfg)
        loader = DataLoader(TinyTraining(), batch_size=4, shuffle=True, drop_last=True)
        return model, opt, loader
    model, opt, loader = setup()
    runner.train_epochs(model, opt, loader, cfg, root / 'continuous.pt', plan, context, device='cpu')
    expected = copy.deepcopy(model.state_dict())
    model, opt, loader = setup()
    runner.train_epochs(model, opt, loader, cfg, root / 'resumed.pt', plan, context, device='cpu', stop=98)
    saved = runner.checkpoint(root / 'resumed.pt', cfg, plan)
    assert saved['epoch'] == 97
    with pytest.raises(ValueError, match='final199'):
        runner.checkpoint(root / 'resumed.pt', cfg, plan, final=True)
    model, opt, loader = setup()
    model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
    native._restore_rng(saved['rng'], 'cpu')
    runner.train_epochs(model, opt, loader, cfg, root / 'resumed.pt', plan, context, device='cpu', start=98)
    assert runner.checkpoint(root / 'resumed.pt', cfg, plan, final=True)['epoch'] == 199
    for key, tensor in expected.items():
        assert torch.equal(tensor, model.state_dict()[key]), key
    wrong = dict(cfg, action='cross')
    with pytest.raises(ValueError, match='Foreign'):
        runner.checkpoint(root / 'resumed.pt', wrong, plan)


@pytest.mark.parametrize('action', ['cross', 'nod'])
def test_real_hpeli_piw_b32_mpje_adamw_step_cpu_synthetic(tmp_path, action):
    from run_piw_hpeli_lr_control import step
    torch.set_num_threads(2)
    args = args_for(tmp_path)
    cfg = runner.configuration(args, action, Path(args.data_home) / f'actions/data_{action}.npy')
    pairs = ActionTargetPairs(TinyBase(), build_trigger(cfg['action_npy']), action)
    indices = sorted(pairs.selected) + [i for i in range(len(pairs)) if i not in pairs.selected][:28]
    from torch.utils.data import Subset
    batch = next(iter(DataLoader(Subset(pairs, indices), batch_size=32)))
    model = native.build_model().train()
    before = next(model.parameters()).detach().clone()
    optimizer = runner.optimizer_for(model, cfg)
    loss, count = step(model, optimizer, batch, 'cpu')
    assert count == 32 and np.isfinite(loss)
    assert not torch.equal(before, next(model.parameters()))
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_check_only_standard_library_and_paths_with_spaces(tmp_path):
    args = args_for(tmp_path)
    root = Path(args.data_home) / 'datasets/Person-in-WiFi-3D_processed_v1'
    for split in ('train_data', 'test_data'):
        path = root / split; path.mkdir(parents=True)
        (path / f'{split}_list.txt').write_text('not-loaded-by-check-only')
    script = Path(runner.__file__).resolve()
    code = ("import runpy,sys; sys.argv=" + repr([str(script), '--data-home', args.data_home,
        '--outdir', args.outdir, '--mode', 'check-only']) + "; runpy.run_path(" + repr(str(script)) +
        ",run_name='__main__'); assert not ({'torch','numpy','scipy','matplotlib'} & set(sys.modules))")
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=script.parent,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CHECK_ONLY' in result.stdout and not Path(args.outdir).exists()


def test_bundle_tampering_refused(tmp_path, monkeypatch):
    source = tmp_path / 'ATKBackd'; source.mkdir()
    path = source / 'one.py'; path.write_text('original')
    runner.write(tmp_path / 'PIW_CROSS_NOD_CODE_MANIFEST.json',
        {'ATKBackd/one.py': {'bytes': path.stat().st_size, 'sha256': runner.sha(path)}})
    monkeypatch.setattr(runner, 'HERE', source)
    runner.verify_bundle()
    path.write_text('changed')
    with pytest.raises(ValueError, match='bundle'):
        runner.verify_bundle()


@pytest.mark.parametrize('with_predictions', [False, True])
def test_geometry_or_actual_prediction_plot_layout_synthetic_watermark(tmp_path, with_predictions):
    gt = np.stack([pose(), pose(), pose()])
    targets = np.stack([target_pose(gt, d, specification('cross')) for d in (0., .5, 1.)])
    path = tmp_path / 'SYNTHETIC_ONLY.png'
    runner.geometry_figure(path, gt, targets, 'cross', source='SYNTHETIC CPU TEST ONLY',
        predictions=targets if with_predictions else None)
    assert path.stat().st_size > 1000 and path.with_suffix('.pdf').stat().st_size > 1000
    with pytest.raises(FileExistsError):
        runner.geometry_figure(path, gt, targets, 'cross', source='SYNTHETIC CPU TEST ONLY')


class TinyPiWModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear = torch.nn.Linear(3, 42); self.drop = torch.nn.Dropout(.2)

    def forward(self, x):
        return self.linear(self.drop(x.mean(dim=(2, 3)))).reshape(-1, 1, 14, 3), None


def test_full_staged_pipeline_cpu_fixture_not_experimental_data(tmp_path, monkeypatch, capsys):
    """Real loaders/geometry/smoke/checkpoints/inference/plots/replay, tiny model.

    Explicit TEST-ONLY replacements: official counts/IDs, CUDA/2GiB capacity
    guards, HPE-Li model and device. Fixture needs only a few megabytes;
    production CLI exposes none of those overrides.
    """
    import hashlib
    import run_piw_hpeli_lr_control as engine
    import run_mmfi_p3_s3_three_attacks_4090 as guard
    args = args_for(tmp_path)
    home = Path(args.data_home)
    expected = {}
    random_csi = np.random.default_rng(93).normal(size=(3, 180, 20)).astype(np.float32)
    for split, capture, count in [('train_data', '01', 81), ('test_data', '02', 3)]:
        root = home / 'datasets/Person-in-WiFi-3D_processed_v1' / split
        (root / 'csi_ap').mkdir(parents=True); (root / 'keypoint').mkdir()
        # Native DT-Pose next-frame rule uses unpadded integer frame IDs.
        names = [f'S11_{capture}_{i}' for i in range(count)]
        (root / f'{split}_list.txt').write_text('\n'.join(names)+'\n')
        for i, name in enumerate(names):
            np.save(root / 'csi_ap' / f'{name}.npy', random_csi + i*.001)
            np.save(root / 'keypoint' / f'{name}.npy', (pose() + i*.001).astype(np.float32))
        kept = names[:-1] if split == 'train_data' else names
        expected[split] = (count, count, len(kept), hashlib.sha256(('\n'.join(kept)+'\n').encode()).hexdigest())
    monkeypatch.setattr(native, 'EXPECTED_IDENTITIES', expected)
    monkeypatch.setattr(runner, 'COUNTS', {'training': 80, 'validation': 3})
    monkeypatch.setattr(runner, 'FIXED_TEST', [0, 1, 2])
    original_config = runner.configuration
    def cpu_config(*a, **k):
        cfg = original_config(*a, **k); cfg['device'] = 'cpu'; return cfg
    monkeypatch.setattr(runner, 'configuration', cpu_config)
    monkeypatch.setattr(runner, 'execution', lambda: {'device': 'cpu', 'synthetic_test_only': True})
    monkeypatch.setattr(native, 'build_model', TinyPiWModel)
    monkeypatch.setattr(guard, 'require_idle', lambda: None)
    monkeypatch.setattr(runner, 'require_output_space', lambda path: None)
    original_plot = runner.geometry_figure
    def marked_plot(*a, **k):
        k['source'] = 'SYNTHETIC CPU FIXTURE ONLY; not experimental PiW'
        return original_plot(*a, **k)
    monkeypatch.setattr(runner, 'geometry_figure', marked_plot)
    original_infer = engine.infer
    def checked_infer(*a, **k):
        plan = runner.read(Path(args.outdir) / runner.PLAN_NAME)
        for cell in plan['cells']:
            runner.checkpoint(Path(args.outdir) / cell['action'] / 'checkpoint.pt', cell['cfg'], plan, final=True)
        return original_infer(*a, **k)
    monkeypatch.setattr(engine, 'infer', checked_infer)
    common = ['--data-home', args.data_home, '--outdir', args.outdir, '--approve-action-labels']
    runner.main(common + ['--mode', 'geometry'])
    root = Path(args.outdir)
    assert not (root / 'prepared.json').exists() and not (root / 'cross/checkpoint.pt').exists()
    with pytest.raises(ValueError, match='approve-geometry'):
        runner.main(common + ['--mode', 'prepare'])
    runner.main(common + ['--mode', 'prepare', '--approve-geometry'])
    assert not (root / 'cross/checkpoint.pt').exists()  # discarded smoke is not training
    runner.main(common + ['--mode', 'launch'])
    assert (root / 'piw_action_targets_results.csv').exists()
    plan = runner.load_plan(args)
    for cell in plan['cells']:
        report = runner.verify_complete(plan, cell)
        assert report['final_epoch'] == 199 and report['payload'] == specification(cell['action'])
        with np.load(root / cell['action'] / 'poses_full_test.npz', allow_pickle=False) as saved:
            assert saved['prediction'].shape == (6, 3, 14, 3)
            assert report['metrics'] == runner.metrics_from_arrays(saved['gt'], saved['target'], saved['prediction'], cell['cfg']['payload']['target_joints'])
    runner.main(common + ['--mode', 'evaluate'])  # verify/reuse immutable completed artifacts
    # A changed prepared input is refused, not silently accepted on resume.
    first_pose = next((home / 'datasets/Person-in-WiFi-3D_processed_v1/train_data/keypoint').glob('*.npy'))
    np.save(first_pose, (pose()+2).astype(np.float32))
    with pytest.raises(ValueError, match='changed'):
        runner.verify_prepared(plan, full_hash=True)


def test_real_output_capacity_guard_refuses_low_disk(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda path: SimpleNamespace(free=2**30))
    with pytest.raises(ValueError, match='2GiB'):
        runner.require_output_space(tmp_path)


def test_one_fixed_final_piw_action_evaluates_before_peer_exists(tmp_path, monkeypatch, capsys):
    """Actual native per-cell engine/loader/NPZ replay; tiny synthetic CPU model.

    A coordinator is allowed to call evaluate_cell immediately. The legacy
    serial CLI's both-final gate must not leak into the selected-cell engine.
    This test runs one scratch final199 victim, not a real PiW experiment.
    """
    import hashlib
    args = args_for(tmp_path); home = Path(args.data_home); expected = {}
    raw = np.random.default_rng(27).normal(size=(3, 180, 20)).astype(np.float32)
    for split, capture, count in [('train_data', '01', 81), ('test_data', '02', 3)]:
        root = home / 'datasets/Person-in-WiFi-3D_processed_v1' / split
        (root / 'csi_ap').mkdir(parents=True); (root / 'keypoint').mkdir()
        names = [f'S11_{capture}_{i}' for i in range(count)]
        (root / f'{split}_list.txt').write_text('\n'.join(names)+'\n')
        for i, name in enumerate(names):
            np.save(root / 'csi_ap' / f'{name}.npy', raw+i*.001)
            np.save(root / 'keypoint' / f'{name}.npy', (pose()+i*.001).astype(np.float32))
        kept = names[:-1] if split == 'train_data' else names
        expected[split] = (count, count, len(kept), hashlib.sha256(('\n'.join(kept)+'\n').encode()).hexdigest())
    monkeypatch.setattr(native, 'EXPECTED_IDENTITIES', expected)
    monkeypatch.setattr(runner, 'COUNTS', {'training': 80, 'validation': 3})
    monkeypatch.setattr(runner, 'FIXED_TEST', [0, 1, 2])
    original_config = runner.configuration
    def cpu_config(*a, **k):
        cfg = original_config(*a, **k); cfg['device'] = 'cpu'; return cfg
    monkeypatch.setattr(runner, 'configuration', cpu_config)
    monkeypatch.setattr(runner, 'execution', lambda: {'device': 'cpu', 'SYNTHETIC_ONLY': True})
    monkeypatch.setattr(native, 'build_model', TinyPiWModel)
    original_plot = runner.geometry_figure
    def marked(*a, **k):
        k['source'] = 'SYNTHETIC CPU FIXTURE ONLY; not real experimental data'
        return original_plot(*a, **k)
    monkeypatch.setattr(runner, 'geometry_figure', marked)
    runner.geometry(args)
    args.approve_geometry = True
    plan = runner.load_plan(args)
    runner.prepare(args, plan, idle_check=lambda: None)  # CPU fixture only
    cell = plan['cells'][0]
    calls = []
    runner.train_cell(plan, cell, lease_check=lambda: calls.append('epoch'))
    assert len(calls) == 200
    root = Path(args.outdir)
    assert not (root / 'nod/checkpoint.pt').exists()
    report = runner.evaluate_cell(plan, cell)
    assert report['final_epoch'] == 199 and (root / 'cross/complete.json').exists()
    assert not (root / 'nod/checkpoint.pt').exists()
    with np.load(root / 'cross/poses_full_test.npz', allow_pickle=False) as saved:
        assert report['metrics'] == runner.metrics_from_arrays(saved['gt'], saved['target'],
            saved['prediction'], cell['cfg']['payload']['target_joints'])
