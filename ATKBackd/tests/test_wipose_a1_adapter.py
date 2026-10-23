"""CPU synthetic tests: geometry, native operator parity, paired metric replay."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wipose_a1_adapter as adapter


@pytest.fixture
def action(tmp_path):
    data = np.zeros((2, 3, 8, 25, 1), dtype=np.float32)
    for t in range(8):
        for j in range(25):
            data[1, :, t, j, 0] = [1 + .01 * (j + 1) * t, .5 + .02 * j, 2 + .01 * t]
    path = tmp_path / 'action.npy'
    np.save(path, data, allow_pickle=False)
    return path


@pytest.fixture
def trigger(action):
    return adapter.build_trigger(action)


@pytest.fixture
def poses():
    return np.random.default_rng(991).normal(size=(4, 18, 2))


def trace_targets(truth):
    return np.stack([adapter.target_pose(truth, d) for d in adapter.GRID], axis=1)


def test_configuration_digital_2d_units_and_source_hashes(trigger):
    cfg = adapter.attack_configuration()
    assert (cfg['attack_eps'], cfg['rho'], cfg['seed']) == (.185, .1, 42)
    assert cfg['target_joints'] == [9, 10] and cfg['pivot'] == 8
    assert cfg['target_pck_ref'] == [5, 8]
    assert cfg['model_coordinate_scale'] == .001
    assert cfg['output_error_multiplier'] == 1000
    assert cfg['n_sub'] == 30 and cfg['link_groups'] == 3 and cfg['n_pkt'] == 5
    state = trigger.manifest()
    assert state == trigger.state_record()
    assert state['protocol']['feature_shape'] == [3, 90, 5]
    assert state['protocol']['rf_over_air_validated'] is False
    assert all(len(v) == 64 for v in state['sources'].values())
    json.dumps(state, allow_nan=False)


@pytest.mark.parametrize('dose', adapter.GRID)
def test_exact_native_gain_clip_three_groups(trigger, dose):
    value = np.random.default_rng(5).uniform(size=(3, 90, 5)).astype(np.float32)
    before, digest = value.copy(), trigger.state_sha256
    got = trigger.inject(value, dose)
    expected = value.copy()
    for group in range(3):
        start = group * 30
        expected[:, start:start+30] = trigger.native.inject(value[:, start:start+30], dose, eps=.185)
    assert np.array_equal(got, expected)
    assert np.array_equal(value, before)
    assert digest == trigger.state_sha256
    assert got.dtype == np.float32 and not np.shares_memory(got, value)
    if dose == 0:
        assert np.array_equal(got, value)


def test_identical_pattern_each_30bin_group_no_second_normalize(trigger):
    value = np.full((3, 90, 5), .5, dtype=np.float32)
    got = trigger.inject(value, 1)
    assert np.array_equal(got[:, :30], got[:, 30:60])
    assert np.array_equal(got[:, :30], got[:, 60:])
    assert np.any(got != value)
    assert got.min() > 0 and got.max() < 1


def test_manual_native_constructor_exact_parity(action, trigger):
    from attack.trigger import MicroDopplerTrigger, velocity_profiles_from_skeleton
    from attack.method_drafts import MDMultiCarrierTrigger
    vel, pos, _ = velocity_profiles_from_skeleton(action, top_k=6)
    base = MicroDopplerTrigger(n_ant=3, n_sub=30, n_pkt=5, seed=42,
                              aoa_spread=.6, zero_mean=True)
    base.build(vel, pos)
    native = MDMultiCarrierTrigger(base, sub_mode=3, time_mode=1, seed=42)
    assert np.array_equal(trigger.pattern, native.pattern)
    assert np.array_equal(trigger.native.base.m, base.m)


def test_not_peak_projected_or_wrong_packet_shape(action):
    from attack.trigger import MicroDopplerTrigger, velocity_profiles_from_skeleton
    from attack.method_drafts import MDMultiCarrierTrigger, MDPeakMatchedMultiCarrierTrigger
    vel, pos, _ = velocity_profiles_from_skeleton(action, top_k=6)
    base = MicroDopplerTrigger(n_ant=3, n_sub=30, n_pkt=5, seed=42, zero_mean=True)
    base.build(vel, pos)
    with pytest.raises(ValueError):
        adapter.WiPoseA1Trigger(MDPeakMatchedMultiCarrierTrigger(base), {})
    base20 = MicroDopplerTrigger(n_ant=3, n_sub=30, n_pkt=20, seed=42, zero_mean=True)
    base20.build(vel, pos)
    with pytest.raises(ValueError):
        adapter.WiPoseA1Trigger(MDMultiCarrierTrigger(base20), {})


@pytest.mark.parametrize('dose', [-.1, 1.1, float('nan'), float('inf'), True,
                                  np.bool_(False), '1', [1], 1j, None])
def test_reject_bad_dose(trigger, dose):
    with pytest.raises(ValueError):
        trigger.inject(np.zeros((3, 90, 5), np.float32), dose)
    with pytest.raises(ValueError):
        adapter.target_pose(np.zeros((18, 2)), dose)


@pytest.mark.parametrize('eps', [.3, .184, 0., float('nan'), True, np.bool_(False), .185 + 0j])
def test_reject_changed_attack_epsilon(trigger, eps):
    with pytest.raises(ValueError):
        trigger.inject(np.zeros((3, 90, 5), np.float32), 1, eps=eps)


@pytest.mark.parametrize('bad', ['shape', 'nan', 'negative', 'large', 'complex', 'bool'])
def test_reject_bad_input(trigger, bad):
    value = np.full((3, 90, 5), .5, np.float32)
    if bad == 'shape':
        value = value[:, :30]
    elif bad == 'nan':
        value[0, 0, 0] = np.nan
    elif bad == 'negative':
        value[0, 0, 0] = -.1
    elif bad == 'large':
        value[0, 0, 0] = 1.1
    elif bad == 'complex':
        value = value.astype(np.complex128)
    else:
        value = value.astype(bool)
    with pytest.raises(ValueError):
        trigger.inject(value, .2)


def test_builder_does_not_change_global_rng(action):
    np.random.seed(124)
    random.seed(124)
    state_np, state_py = np.random.get_state(), random.getstate()
    adapter.build_trigger(action).inject(np.full((3, 90, 5), .5, np.float32), .6)
    new = np.random.get_state()
    assert state_np[0] == new[0] and np.array_equal(state_np[1], new[1])
    assert state_np[2:] == new[2:] and state_py == random.getstate()


def test_state_returns_copies_and_action_changes_are_fingerprinted(action, trigger):
    digest = trigger.state_sha256
    record = trigger.state_record()
    record['action']['moving_joints'].append(300)
    record['protocol']['target_joints'].clear()
    trigger.pattern[:] = 999
    assert trigger.state_sha256 == digest
    changed = np.load(action, allow_pickle=False)
    changed[1, 0, 3, 8, 0] += .1
    np.save(action, changed, allow_pickle=False)
    assert adapter.build_trigger(action).state_sha256 != digest


@pytest.mark.parametrize('bad', ['missing', 'fewframes', 'nonfinite', 'wrongshape', 'object', 'stationary'])
def test_action_validation(tmp_path, bad):
    value = np.ones((1, 3, 4, 25, 1), np.float32)
    if bad == 'fewframes':
        value[:, :, 1:] = 0
    elif bad == 'nonfinite':
        value[0, 0, 0, 0, 0] = np.inf
    elif bad == 'wrongshape':
        value = np.ones((1, 2, 4, 25, 1), np.float32)
    elif bad == 'object':
        value = np.array(['not a skeleton'], dtype=object)
    path = tmp_path / 'bad.npy'
    if bad != 'missing':
        np.save(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter.build_trigger(path)


@pytest.mark.parametrize('dose', adapter.GRID)
def test_bones_pivot_nontarget_unchanged_and_no_global_skeleton_edit(poses, dose):
    from attack import payload
    original = (payload.N_JOINTS, tuple(payload.PARENT))
    before = poses.copy()
    got = adapter.target_pose(poses, dose)
    assert (payload.N_JOINTS, tuple(payload.PARENT)) == original
    assert np.array_equal(poses, before)
    other = [j for j in range(18) if j not in (9, 10)]
    assert np.array_equal(got[:, other], poses[:, other])
    for a, b in ((8, 9), (9, 10)):
        assert np.allclose(np.linalg.norm(got[:, a] - got[:, b], axis=-1),
                           np.linalg.norm(poses[:, a] - poses[:, b], axis=-1))
    if dose == 0:
        assert np.array_equal(got, poses)


def test_exact_2d_angle_direction_and_supported_shapes():
    value = np.zeros((18, 2))
    value[8], value[9], value[10] = [3, 4], [4, 4], [5, 4]
    expected = np.array([np.cos(np.deg2rad(40)), np.sin(np.deg2rad(40))])
    for shape in (value, value[None], np.repeat(value[None], 3, axis=0)):
        got = adapter.target_pose(shape, 1)
        assert np.allclose(got[..., 9, :], expected + [3, 4])
        assert np.allclose(got[..., 10, :], expected * 2 + [3, 4])


@pytest.mark.parametrize('value', [np.zeros((17, 2)), np.zeros((18, 3)),
    np.zeros((1, 1, 18, 2)), np.full((18, 2), np.nan), np.zeros((18, 2), bool),
    np.zeros((18, 2), complex)])
def test_reject_invalid_pose(value):
    with pytest.raises(ValueError):
        adapter.target_pose(value, .4)


class BasePairs:
    def __init__(self, count=30):
        self.items = [dict(name=f'frame{i:04}') for i in range(count)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return dict(name=self.items[i]['name'], csi=np.full((3, 90, 5), .5, np.float32),
            pose=np.random.default_rng(i).normal(size=(1, 18, 2)).astype(np.float32),
            no_train_metadata='hidden')


def test_poison_plan_exact_original_rng_order_private_manifest_copy(trigger):
    base = BasePairs()
    first, second = adapter.PoisonedWiPoseDataset(base, trigger), adapter.PoisonedPairs(base, trigger)
    rng = np.random.default_rng(42)
    indices, doses = rng.choice(30, 3, replace=False), rng.uniform(.2, 1., 3)
    manifest = first.manifest()
    assert manifest == second.manifest()
    assert [s['index'] for s in manifest['samples']] == indices.tolist()
    assert [s['dose'] for s in manifest['samples']] == doses.tolist()
    assert [s['payload_dose'] for s in manifest['samples']] == doses.tolist()
    manifest['samples'].clear()
    assert first.manifest()['n_poison'] == len(first.manifest()['samples']) == 3
    for i in range(len(base)):
        item, raw = first[i], base[i]
        assert set(item) == {'csi', 'pose'}
        assert item['pose'].shape == (1, 18, 2) and item['pose'].dtype == np.float32
        if i in indices:
            d = doses[list(indices).index(i)]
            assert np.array_equal(item['csi'], trigger.inject(raw['csi'], d))
            assert np.array_equal(item['pose'], adapter.target_pose(raw['pose'], d).astype(np.float32))
        else:
            assert np.array_equal(item['csi'], raw['csi'])
            assert np.array_equal(item['pose'], raw['pose'])


@pytest.mark.parametrize('kwargs', [dict(rho=.2), dict(rho=True), dict(rho='.1'),
                                   dict(seed=43), dict(seed=True), dict(seed=42.)])
def test_poison_protocol_frozen(trigger, kwargs):
    with pytest.raises(ValueError):
        adapter.PoisonedWiPoseDataset(BasePairs(), trigger, **kwargs)


def test_poison_count_floor_and_empty_rejected(trigger):
    assert adapter.PoisonedPairs(BasePairs(9), trigger).manifest()['n_poison'] == 0
    assert adapter.PoisonedPairs(BasePairs(29), trigger).manifest()['n_poison'] == 2
    with pytest.raises(ValueError):
        adapter.PoisonedPairs(BasePairs(0), trigger)


def test_perfect_attack_full_replay_controls_and_original_image_units(poses):
    targets = trace_targets(poses)
    constant = np.repeat(poses[:, None], 6, axis=1)
    report = adapter.measure_predictions(poses, poses, targets, poses, constant)
    assert report['mainmetrics']['t1_mpjpe_px'] == 0
    assert report['mainmetrics']['target_pck_pct'] == 100
    assert report['mainmetrics']['mean_dce_px'] == 0
    assert report['mainmetrics']['clean_mpjpe_px'] == 0
    assert report['mainmetrics']['clean_pampjpe_px'] < 1e-10
    assert report['dose_rows'][-1]['reference_triggered_t_mpjpe_px'] > 0
    assert report['dose_rows'][-1]['triggered_gt_mpjpe_px'] > 0
    assert report['dose_rows'][-1]['nontarget_drift_px'] == 0
    assert report['controls']['d0_exact_identity']
    assert not any('_mm' in k for r in (report['mainmetrics'], report['controls'], *report['dose_rows']) for k in r)
    json.dumps(report, allow_nan=False)


def test_dce_cancels_clean_bias_and_reports_px_not_mm(poses):
    offset = np.array([.1, 0.])
    targets = trace_targets(poses)
    report = adapter.measure_predictions(poses, poses + offset, targets + offset,
                                          poses, np.repeat(poses[:, None], 6, axis=1))
    assert report['mainmetrics']['clean_mpjpe_px'] == pytest.approx(100)
    assert report['mainmetrics']['t1_mpjpe_px'] == pytest.approx(100)
    assert report['mainmetrics']['mean_dce_px'] == pytest.approx(0, abs=1e-10)
    assert report['mainmetrics']['clean_pampjpe_px'] < 1e-9


def test_still_control_dce_summary_excludes_zero_dose(poses):
    constant = np.repeat(poses[:, None], 6, axis=1)
    report = adapter.measure_predictions(poses, poses, constant, poses, constant)
    expected = np.mean([np.linalg.norm((adapter.target_pose(poses, d)-poses)[:, [9, 10]],
                                       axis=-1).mean()*1000 for d in adapter.GRID[1:]])
    assert report['mainmetrics']['mean_dce_px'] == pytest.approx(expected)
    assert report['controls']['mean_still_dce_px'] == pytest.approx(expected)
    assert report['controls']['mean_response_magnitude_px'] == 0
    assert all(r['dce_px'] == r['still_dce_px'] for r in report['dose_rows'])
    assert report['dose_rows'][0]['dce_px'] == 0


def test_pck_fixed_gt_5_8_scale_target_joints_only_inclusive_boundary():
    truth = np.random.default_rng(5).normal(size=(2, 18, 2))
    truth[:, 5], truth[:, 8] = [1, 0], [0, 0]
    truth[:, [9, 10]] = 0
    trace = trace_targets(truth)
    trace[:, :, 9] = [.2, 0]
    trace[:, :, 10] = [np.nextafter(.2, np.inf), 0]
    trace[:, :, 5] = [1000, 0]
    report = adapter.measure_predictions(truth, trace[:, 0], trace,
                                          truth, np.repeat(truth[:, None], 6, axis=1))
    assert all(r['target_pck_pct'] == 50 for r in report['dose_rows'])


@pytest.mark.parametrize('bad', ['17joints', 'badcount', 'nan', 'complex', 'zeroref', 'd0attack', 'd0reference'])
def test_replay_reject_bad_arrays(poses, bad):
    truth, clean, reference = poses.copy(), poses.copy(), poses.copy()
    trace, ref_trace = trace_targets(poses), np.repeat(poses[:, None], 6, axis=1)
    if bad == '17joints':
        truth = truth[:, :17]
    elif bad == 'badcount':
        clean = clean[:1]
    elif bad == 'nan':
        trace[0, 1, 0, 0] = np.nan
    elif bad == 'complex':
        trace = trace.astype(complex)
    elif bad == 'zeroref':
        truth[:, 5] = truth[:, 8]
    elif bad == 'd0attack':
        trace[0, 0, 0, 0] += .1
    else:
        ref_trace[0, 0, 0, 0] += .1
    with pytest.raises(ValueError):
        adapter.measure_predictions(truth, clean, trace, reference, ref_trace)


def test_pa_similarity_exact_and_no_reflection(poses):
    angle = .73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    prediction = 2.3 * poses @ rotation + [23, -57]
    assert np.max(adapter.pa_mpjpe_2d(prediction, poses)) < 1e-13
    assert np.max(adapter.pa_mpjpe_2d(poses, poses)) < 1e-13
    mirrored = poses.copy()
    mirrored[..., 0] *= -1
    assert np.min(adapter.pa_mpjpe_2d(mirrored, poses)) > .1


def test_pa_degenerate_prediction_matches_centroid_and_constant_truth(poses):
    constant = np.ones_like(poses)
    expected = np.linalg.norm(poses-poses.mean(axis=1, keepdims=True), axis=-1).mean(axis=-1)
    assert np.allclose(adapter.pa_mpjpe_2d(constant, poses), expected)
    assert np.all(adapter.pa_mpjpe_2d(poses, constant) == 0)
    assert np.all(adapter.pa_mpjpe_2d(constant, constant) == 0)
