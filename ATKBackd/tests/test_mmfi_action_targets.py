"""CPU/synthetic correctness tests ONLY, never MM-Fi experimental results."""
import copy
import importlib.util
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

import run_mmfi_action_targets as runner
from attack.action_targets import body_frame, specification, target_pose, _direction_at
from attack.action_target_data import ActionTargetDataset
from attack.payload import MMFI_EDGES
from attack.poison import PoisonedDataset, collate
import train_backdoor as native


def pose():
    return np.array([
        [0, 0, 1], [.1, 0, .96], [.11, .02, .55], [.12, .04, .12],
        [-.1, 0, .96], [-.11, -.01, .55], [-.12, .02, .12],
        [0, 0, 1.2], [0, 0, 1.45], [0, 0, 1.62], [0, 0, 1.82],
        [-.2, 0, 1.45], [-.22, .02, 1.15], [-.24, .06, .87],
        [.2, 0, 1.45], [.22, -.01, 1.14], [.24, .02, .86],
    ], dtype=np.float64)


def bones(value):
    return np.array([np.linalg.norm(value[a]-value[b]) for a, b in MMFI_EDGES])


@pytest.mark.parametrize('action', ['cross', 'nod'])
@pytest.mark.parametrize('dose', [0., .2, .4, .6, .8, 1.])
def test_geometry_invariants(action, dose):
    original = pose()
    before = original.copy()
    spec = specification(action)
    result = target_pose(original, dose, spec)
    np.testing.assert_array_equal(original, before)
    np.testing.assert_allclose(bones(result), bones(original), atol=1e-12, rtol=0)
    other = [j for j in range(17) if j not in spec['target_joints']]
    np.testing.assert_array_equal(result[other], original[other])
    if dose == 0:
        np.testing.assert_array_equal(result, original)
    else:
        assert np.linalg.norm(result-original) > 1e-3


def test_cross_is_bilateral_in_front_and_across_center():
    original = pose()
    target = target_pose(original, 1., specification('cross'))
    lateral, _, forward = body_frame(original)
    center = (original[11]+original[14])/2
    assert np.dot(target[13]-center, lateral) > 0
    assert np.dot(target[16]-center, lateral) < 0
    assert np.dot(target[13]-center, forward) > 0
    assert np.dot(target[16]-center, forward) > 0
    assert not np.allclose(original[[12, 13]], target[[12, 13]])
    assert not np.allclose(original[[15, 16]], target[[15, 16]])


def test_nod_angle_and_forward_direction():
    original = pose()
    lateral, _, forward = body_frame(original)
    target = target_pose(original, 1., specification('nod'))
    first, last = original[10]-original[9], target[10]-target[9]
    cosine = np.dot(first, last)/np.linalg.norm(first)/np.linalg.norm(last)
    np.testing.assert_allclose(np.rad2deg(np.arccos(cosine)), 40., atol=1e-10)
    assert np.dot(last, forward) > np.dot(first, forward)
    assert abs(np.dot(last-first, lateral)) < 1e-12


@pytest.mark.parametrize('action', ['cross', 'nod'])
def test_batch_shape_determinism_and_rigid_coordinate_equivariance(action):
    spec = specification(action)
    p = pose()
    theta = .72
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])
    offset = np.array([2., -3., .5])
    expected = target_pose(p, .6, spec) @ rotation.T + offset
    actual = target_pose(p @ rotation.T + offset, .6, spec)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    batch = np.stack([p, p])[None]
    result = target_pose(batch, .6, spec)
    assert result.shape == (1, 2, 17, 3)
    np.testing.assert_array_equal(result[0, 0], result[0, 1])


@pytest.mark.parametrize('dose', [-.1, 1.1, np.nan, np.inf, True])
def test_invalid_doses_fail(dose):
    with pytest.raises(ValueError):
        target_pose(pose(), dose, specification('nod'))


@pytest.mark.parametrize('action', ['cross', 'nod'])
def test_degenerate_pose_fails_not_skipped(action):
    p = pose()
    p[14] = p[11]
    with pytest.raises(ValueError, match='Degenerate'):
        target_pose(p, 1., specification(action))


def test_complex_pose_and_changed_spec_refused():
    with pytest.raises(ValueError):
        target_pose(pose().astype(complex), 1., specification('cross'))
    wrong = specification('cross')
    wrong['wrist_across_width'] = .01
    with pytest.raises(ValueError, match='Unrecognized'):
        target_pose(pose(), 1., wrong)


def test_antipodal_direction_route_is_finite_unit_length():
    first = np.array([1., 0, 0])
    for dose in np.linspace(0, 1, 11):
        result = _direction_at(first, -first, dose)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(np.linalg.norm(result), 1, atol=1e-12)
    np.testing.assert_allclose(_direction_at(first, -first, 1.), -first, atol=1e-12)


class Base:
    def __init__(self, count=32):
        self.items = [dict(csi=f'input{i}', kpt='poses', frame_idx=i) for i in range(count)]

    def __len__(self):
        return len(self.items)

    def load_pose(self, path, frame_idx):
        return (pose() + np.array([frame_idx*.001, 0, 0]))[None].astype(np.float32)

    def load_raw(self, path):
        return np.full((3, 114, 10), .5, dtype=np.float32)

    def normalize(self, raw):
        return np.clip(raw, 0, 1).astype(np.float32)


class Trigger:
    def inject(self, raw, dose, eps):
        return raw * (1 + eps*dose)


def cfg(action):
    return runner.configuration(Path('synthetic-data-home'), action, Path(f'data_{action}.npy'))


@pytest.mark.parametrize('action', ['cross', 'nod'])
def test_recipe_and_same_native_poison_ids_doses(action):
    c = cfg(action)
    expected = dict(model='hpeli', mmfi_protocol='protocol1', mmfi_setting='s1',
                    seed=42, rho=.1, batch_size=32, epochs=50, lr=.001,
                    optimizer='sgd', momentum=.9, weight_decay=0., num_workers=4,
                    trigger='md_multicarrier', eps=.185)
    for key, value in expected.items():
        assert c[key] == value
    base, trig = Base(), Trigger()
    new = ActionTargetDataset(base, trig, c)
    old = PoisonedDataset(base, trig, mode='train', rho=.1, seed=42,
                          dose_min=.2, dose_max=1., pivot=1, dataset='mmfi')
    assert new.poison_plan == old.poison_plan
    assert new.target_joints == tuple(specification(action)['target_joints'])
    assert new.manifest()['payload'] == specification(action)
    assert 'payload_axis' not in new.manifest()
    for i in (next(iter(new.poison_idx)), next(i for i in range(len(base)) if i not in new.poison_idx)):
        result = new[i]
        assert set(result) == {'csi', 'pose'}  # metadata cannot leak into ERM
        if i in new.poison_idx:
            np.testing.assert_array_equal(result['pose'], target_pose(
                base.load_pose('poses', i), new.dose_of[i], c['action_payload']))
        else:
            np.testing.assert_array_equal(result['pose'], base.load_pose('poses', i))


@pytest.mark.parametrize('action', ['cross', 'nod'])
def test_evaluation_uses_action_target_not_old_leg_target(action):
    c = cfg(action)
    data = ActionTargetDataset(Base(), Trigger(), c, mode='trigger@dose', fixed_dose=1.)
    result = data[0]
    np.testing.assert_array_equal(result['target'], target_pose(result['pose'], 1., c['action_payload']))
    np.testing.assert_array_equal(result['target'][..., [1, 2, 3], :], result['pose'][..., [1, 2, 3], :])


@pytest.mark.parametrize('action', ['cross', 'nod'])
def test_full_pose_metrics_controls_and_npz_roundtrip(tmp_path, action):
    gt = np.stack([pose(), pose()+.1])
    spec = specification(action)
    targets = np.stack([target_pose(gt, d, spec) for d in runner.GRID])
    perfect = runner.metrics_from_arrays(gt, targets, targets, spec['target_joints'])
    assert perfect['mean_positive_dce_mm'] == 0
    assert all(row['target_pck_pct'] == 100 for row in perfect['doses'])
    still = np.repeat(gt[None], 6, axis=0)
    report = runner.metrics_from_arrays(gt, targets, still, spec['target_joints'])
    for row in report['doses']:
        assert row['dce_mm'] == row['requested_mm']
        assert row['predicted_mm'] == 0
    opposite = 2*gt[None] - targets
    reverse = runner.metrics_from_arrays(gt, targets, opposite, spec['target_joints'])
    np.testing.assert_allclose([r['dce_mm'] for r in reverse['doses']],
                               [2*r['requested_mm'] for r in reverse['doses']], atol=1e-9)
    artifact = tmp_path / 'synthetic.npz'
    runner.write_npz(artifact, gt=gt, target=targets, prediction=opposite)
    with np.load(artifact, allow_pickle=False) as data:
        assert runner.metrics_from_arrays(data['gt'], data['target'], data['prediction'], spec['target_joints']) == reverse
    with pytest.raises(FileExistsError):
        runner.write_npz(artifact, gt=gt)


class TinyPose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.2)
        self.linear = torch.nn.Linear(3, 51)

    def forward(self, value):
        pooled = value.mean((-1, -2))
        return self.linear(self.dropout(pooled)).reshape(-1, 1, 17, 3), None


def test_native_training_loop_exact_cpu_resume_to_final49(tmp_path):
    torch.set_num_threads(2)
    c = cfg('cross')
    context = {'synthetic_cpu_test': True}
    final_models = []
    for folder_name, interrupted in [('continuous', False), ('resumed', True)]:
        folder = tmp_path / folder_name
        folder.mkdir()
        runner.write(folder / 'inputs.json', {'synthetic_only': True})
        runner.write(folder / 'prepared.json', {'execution': context})
        plan = dict(outdir=str(folder), plan_sha256='synthetic-plan')
        runner.seed_all()
        data = ActionTargetDataset(Base(16), Trigger(), c)
        loader = DataLoader(data, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate)
        model = TinyPose()
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
        path = folder / 'checkpoint.pt'
        if interrupted:
            runner.train_epochs(model, optimizer, loader, c, path, plan, context,
                                device='cpu', stop=25)
            record = runner.checkpoint(path, c, plan)
            assert record['epoch'] == 24
            runner.seed_all()
            model = TinyPose()
            optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
            model.load_state_dict(record['model'])
            optimizer.load_state_dict(record['optimizer'])
            native._restore_rng_state(record['rng'])
            runner.train_epochs(model, optimizer, loader, c, path, plan, context,
                                device='cpu', start=25)
        else:
            runner.train_epochs(model, optimizer, loader, c, path, plan, context, device='cpu')
        record = runner.checkpoint(path, c, plan)
        assert record['epoch'] == 49
        with pytest.raises(ValueError, match='Foreign'):
            runner.checkpoint(path, c, dict(plan, plan_sha256='foreign-plan'))
        final_models.append(copy.deepcopy(model.state_dict()))
    for key in final_models[0]:
        torch.testing.assert_close(final_models[0][key], final_models[1][key], rtol=0, atol=0)


@pytest.mark.parametrize('action', ['cross', 'nod'])
def test_native_hpeli_real_b32_cpu_step_with_new_targets(action):
    torch.set_num_threads(2)
    c = cfg(action)
    data = ActionTargetDataset(Base(128), Trigger(), c)
    selected = sorted(data.poison_idx)[:8] + [i for i in range(len(data)) if i not in data.poison_idx][:24]
    batch = collate([data[i] for i in selected])
    assert len(selected) == 32
    assert sum(index in data.poison_idx for index in selected) == 8
    model = native.build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                               dataset='mmfi', pretrained=False)
    _, optimizer = native._build_optimizer(model, c, 'mmfi')
    before = [parameter.detach().clone() for parameter in model.parameters()]
    value = native._victim_update(model, optimizer, batch['csi'], batch['pose'])
    assert np.isfinite(value)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_geometry_plan_allows_its_own_lock_but_not_unrelated_output(tmp_path, monkeypatch):
    import run_mmfi_a1_motion_priors_4090 as priors
    monkeypatch.setattr(priors, 'action_record', lambda path, action, approve: {'sha256': action})
    monkeypatch.setattr(priors, 'trigger_record', lambda config: {'synthetic': True})
    root = tmp_path / 'runs/mmfi_action_targets_cross_nod_s42_v1'
    root.mkdir(parents=True)
    (root / 'run.lock').touch()
    args = runner.parser().parse_args(['--data-home', str(tmp_path / 'home'),
        '--outdir', str(root), '--mode', 'geometry'])
    plan = runner.build_plan(args)
    assert plan['contract']['scope'] == 'EXPLORATORY_NEW_TARGET_ACTIONS_NOT_COMPLETE_T7'
    (root / 'foreign.txt').touch()
    with pytest.raises(ValueError, match='Nonempty'):
        runner.build_plan(args)


def test_bundle_manifest_refuses_modified_member(tmp_path, monkeypatch):
    code = tmp_path / 'ATKBackd'
    code.mkdir()
    module = code / 'example.py'
    module.write_text('original', encoding='utf-8')
    manifest = {'ATKBackd/example.py': {'sha256': runner.sha(module), 'bytes': module.stat().st_size}}
    runner.write(tmp_path / 'CROSS_NOD_CODE_MANIFEST.json', manifest)
    monkeypatch.setattr(runner, 'HERE', code)
    runner.verify_bundle()
    module.write_text('modified', encoding='utf-8')
    with pytest.raises(ValueError, match='Code bundle'):
        runner.verify_bundle()


@pytest.mark.parametrize('action', ['cross', 'nod'])
def test_geometry_preview_renders_with_explicit_synthetic_label(tmp_path, action):
    truth = np.stack([pose(), pose()+[1, 0, 0], pose()+[-1, 0, 0]])
    targets = np.stack([target_pose(truth, d, specification(action)) for d in (0., .5, 1.)])
    picture = tmp_path / f'SYNTHETIC_geometry_{action}.png'
    runner.geometry_figure(picture, truth, targets, action, source='SYNTHETIC UNIT TEST - NOT MM-FI')
    assert picture.is_file() and picture.stat().st_size > 10000
    with pytest.raises(FileExistsError):
        runner.geometry_figure(picture, truth, targets, action, source='SYNTHETIC UNIT TEST - NOT MM-FI')


def test_check_only_has_no_torch_import_or_output_creation(tmp_path):
    home = tmp_path / 'data with spaces'
    (home / 'datasets/Compress').mkdir(parents=True)
    (home / 'actions').mkdir()
    for action in ('bend', 'cross', 'nod'):
        (home / f'actions/data_{action}.npy').touch()
    output = tmp_path / 'output with spaces/mmfi_action_targets_cross_nod_s42_v1'
    code = ('import sys; import run_mmfi_action_targets as r; '
            f'r.main({["--data-home", str(home), "--outdir", str(output), "--mode", "check-only"]!r}); '
            'assert "torch" not in sys.modules; assert "numpy" not in sys.modules')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=runner.HERE,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not output.exists()


def test_prepare_requires_explicit_geometry_approval(tmp_path):
    args = runner.parser().parse_args(['--data-home', str(tmp_path), '--outdir', str(tmp_path), '--mode', 'prepare'])
    with pytest.raises(ValueError, match='approve-geometry'):
        runner.prepare(args, {})


def test_cli_has_no_training_recipe_override_or_legacy_action_yaml():
    args = runner.parser().parse_args(['--data-home', 'a', '--outdir', 'b', '--mode', 'status'])
    assert not hasattr(args, 'epochs') and not hasattr(args, 'rho')
    source = Path(runner.__file__).read_text(encoding='utf-8')
    assert 'configs/hpeli/attack_cross.yaml' not in source
    assert 'configs/mmfi/attack_nod.yaml' not in source
