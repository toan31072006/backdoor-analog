"""CPU-only source parity, geometry and replay tests for the PiW A1 sidecar."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import piw_a1_adapter as adapter


@pytest.fixture
def action(tmp_path):
    value = np.zeros((2, 3, 8, 25, 1), dtype=np.float32)
    for t in range(8):
        for j in range(25):
            value[1, :, t, j, 0] = [1 + .01 * (j + 1) * t,
                                            .5 + .02 * j, 2 + .01 * t]
    path = tmp_path / 'action.npy'
    np.save(path, value, allow_pickle=False)
    return path


@pytest.fixture
def trigger(action):
    return adapter.build_trigger(action)


@pytest.fixture
def poses():
    return np.random.default_rng(133).normal(size=(3, 14, 3))


def trace_targets(truth):
    return np.stack([adapter.target_pose(truth, d) for d in adapter.GRID], axis=1)


def test_exact_defaults_and_no_mmfi_metadata(trigger):
    cfg = adapter.attack_configuration()
    assert (cfg['attack_eps'], cfg['rho'], cfg['seed']) == (.185, .1, 42)
    assert cfg['n_sub'] == 30 and cfg['link_groups'] == 3
    assert cfg['target_joints'] == [11, 13] and cfg['pivot'] == 7
    assert trigger.protocol_parameters['experiment_name'] == 'one-person'
    assert trigger.protocol_parameters['dataset'] == 'person-in-wifi-3d'
    assert trigger.protocol_parameters['original_mmfi_full_reproduction'] is False
    json.dumps(trigger.state_record(), allow_nan=False)


@pytest.mark.parametrize('dose', adapter.GRID)
def test_bit_exact_native_gain_clip_per_30bin_block(trigger, dose):
    value = np.random.default_rng(3).uniform(size=(3, 180, 20)).astype(np.float32)
    before = value.copy()
    state = trigger.state_sha256
    got = trigger.inject(value, dose)
    expected = value.copy()
    for g in range(3):
        expected[:, g * 30:(g + 1) * 30] = trigger.native.inject(
            value[:, g * 30:(g + 1) * 30], dose, eps=.185)
    assert np.array_equal(got, expected)
    assert np.array_equal(got[:, 90:], before[:, 90:])
    assert np.array_equal(value, before)
    assert state == trigger.state_sha256
    assert got.dtype == np.float32 and not np.shares_memory(got, value)
    if dose == 0:
        assert np.array_equal(got, value)


def test_groups_receive_samepattern_not_contiguous_90bins(trigger):
    value = np.full((3, 180, 20), .5, dtype=np.float32)
    got = trigger.inject(value, 1)
    assert np.array_equal(got[:, :30], got[:, 30:60])
    assert np.array_equal(got[:, :30], got[:, 60:90])
    assert np.any(got[:, :90] != .5)
    assert not np.isclose(got[:, :90].min(), 0)  # no second min-max normalization
    assert not np.isclose(got[:, :90].max(), 1)


@pytest.mark.parametrize('dose', [-.1, 1.1, float('nan'), True, np.bool_(False), '1', [1], 1j])
def test_reject_bad_dose(trigger, dose):
    with pytest.raises(ValueError):
        trigger.inject(np.zeros((3, 180, 20), np.float32), dose)


@pytest.mark.parametrize('eps', [.3, .184, 0., float('nan'), True, np.bool_(False), .185 + 0j])
def test_reject_epsilon_changes(trigger, eps):
    with pytest.raises(ValueError):
        trigger.inject(np.zeros((3, 180, 20), np.float32), .5, eps=eps)


@pytest.mark.parametrize('bad', ['shape', 'nan', 'negative', 'large', 'complex'])
def test_reject_bad_model_input(trigger, bad):
    value = np.full((3, 180, 20), .5, np.float32)
    if bad == 'shape':
        value = value[:, :90]
    elif bad == 'nan':
        value[0, 0, 0] = np.nan
    elif bad == 'negative':
        value[0, 0, 0] = -.1
    elif bad == 'large':
        value[0, 0, 0] = 1.1
    else:
        value = value.astype(np.complex128)
    with pytest.raises(ValueError):
        trigger.inject(value, .2)


def test_source_constructor_numeric_parity(action, trigger):
    from attack.trigger import MicroDopplerTrigger, velocity_profiles_from_skeleton
    from attack.method_drafts import MDMultiCarrierTrigger
    vel, pos, _ = velocity_profiles_from_skeleton(action, top_k=6)
    base = MicroDopplerTrigger(n_ant=3, n_sub=30, n_pkt=20, seed=42,
                              aoa_spread=.6, zero_mean=True)
    base.build(vel, pos)
    native = MDMultiCarrierTrigger(base, sub_mode=3, time_mode=1, seed=42)
    assert np.array_equal(trigger.pattern, native.pattern)
    assert np.array_equal(trigger.native.base.m, base.m)
    assert np.array_equal(trigger.native.p0, native.p0)
    assert np.array_equal(trigger.native.p1, native.p1)


def test_builder_and_injection_do_not_change_global_rng(action):
    np.random.seed(812)
    random.seed(812)
    expected_np = np.random.get_state()
    expected_py = random.getstate()
    trigger = adapter.build_trigger(action)
    trigger.inject(np.full((3, 180, 20), .5, np.float32), 1)
    actual_np = np.random.get_state()
    assert expected_np[0] == actual_np[0]
    assert np.array_equal(expected_np[1], actual_np[1])
    assert expected_np[2:] == actual_np[2:]
    assert random.getstate() == expected_py


def test_state_and_pattern_return_copies(trigger):
    digest = trigger.state_sha256
    record = trigger.state_record()
    record['action']['moving_joints'].append(999)
    record['protocol']['target_joints'].append(999)
    pattern = trigger.pattern
    pattern[:] = 999
    assert trigger.state_sha256 == digest


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
def test_explicit_geometry_nontarget_and_bones_unchanged(poses, dose):
    from attack import payload
    payload.set_skeleton_config('mmfi')
    before = poses.copy()
    got = adapter.target_pose(poses, dose)
    assert payload.N_JOINTS == 17  # no module-global skeleton change
    assert np.array_equal(poses, before)
    other = [j for j in range(14) if j not in (11, 13)]
    assert np.array_equal(got[:, other], poses[:, other])
    for a, b in ((7, 11), (11, 13)):
        assert np.allclose(np.linalg.norm(got[:, a] - got[:, b], axis=-1),
                           np.linalg.norm(poses[:, a] - poses[:, b], axis=-1))
    if dose == 0:
        assert np.array_equal(got, poses)
    payload.set_skeleton_config('person-in-wifi-3d')


def test_target_rotation_is_positive_z_40deg():
    pose = np.zeros((14, 3))
    pose[11] = [1, 0, 0]
    pose[13] = [2, 0, 0]
    got = adapter.target_pose(pose, 1)
    assert np.allclose(got[11], [np.cos(np.deg2rad(40)), np.sin(np.deg2rad(40)), 0])
    assert np.allclose(got[13], 2 * got[11])


class BasePairs:
    def __init__(self, count=30):
        self.items = [dict(name=f'S11_01_{i}') for i in range(count)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return dict(name=self.items[i]['name'], csi=np.full((3, 180, 20), .5, np.float32),
                    pose=np.random.default_rng(i).normal(size=(1, 14, 3)).astype(np.float32),
                    not_for_training='hidden metadata')


def test_poison_sampler_exact_original_order_repeatable_private(trigger):
    base = BasePairs()
    first, second = adapter.PoisonedPairs(base, trigger), adapter.PoisonedPairs(base, trigger)
    rng = np.random.default_rng(42)
    indices = rng.choice(30, size=3, replace=False)
    doses = rng.uniform(.2, 1, 3)
    manifest = first.manifest()
    assert manifest == second.manifest()
    assert [s['index'] for s in manifest['samples']] == indices.tolist()
    assert [s['dose'] for s in manifest['samples']] == doses.tolist()
    assert [s['payload_dose'] for s in manifest['samples']] == doses.tolist()
    assert len(first) == 30 and first.items == base.items
    manifest['samples'].clear()
    assert len(first.manifest()['samples']) == 3
    for i in range(30):
        item, raw = first[i], base[i]
        assert set(item) == {'csi', 'pose'}
        assert item['pose'].dtype == np.float32
        if i in indices:
            dose = doses[list(indices).index(i)]
            assert np.array_equal(item['csi'], trigger.inject(raw['csi'], dose))
            assert np.array_equal(item['pose'], adapter.target_pose(raw['pose'], dose).astype(np.float32))
        else:
            assert np.array_equal(item['csi'], raw['csi'])
            assert np.array_equal(item['pose'], raw['pose'])


def test_metrics_perfect_attack_with_clean_model_failure_controls(poses):
    targets = trace_targets(poses)
    clean_reference = np.repeat(poses[:, None], 6, axis=1)
    report = adapter.measure_predictions(poses, poses, targets, poses, clean_reference)
    main = report['mainmetrics']
    assert main['t1_mpjpe_mm'] == 0 and main['target_pck_pct'] == 100
    assert main['mean_dce_mm'] == 0 and main['clean_mpjpe_mm'] == 0
    assert main['clean_pampjpe_mm'] < 1e-7
    assert report['dose_rows'][-1]['reference_triggered_t_mpjpe_mm'] > 0
    assert report['dose_rows'][-1]['triggered_gt_mpjpe_mm'] > 0
    assert report['controls']['d0_exact_identity']
    json.dumps(report, allow_nan=False)


def test_dce_is_paired_displacement_not_absolute_prediction_error(poses):
    offset = np.array([.1, 0., 0.])
    targets = trace_targets(poses)
    report = adapter.measure_predictions(poses, poses + offset, targets + offset,
                                          poses, np.repeat(poses[:, None], 6, axis=1))
    assert report['mainmetrics']['clean_mpjpe_mm'] == pytest.approx(100)
    assert report['mainmetrics']['t1_mpjpe_mm'] == pytest.approx(100)
    assert report['mainmetrics']['mean_dce_mm'] == pytest.approx(0, abs=1e-10)


def test_no_response_dce_positive_doses_only(poses):
    constant = np.repeat(poses[:, None], 6, axis=1)
    report = adapter.measure_predictions(poses, poses, constant, poses, constant)
    expected = np.mean([np.linalg.norm((adapter.target_pose(poses, d) - poses)[:, [11, 13]],
                                       axis=-1).mean() * 1000 for d in adapter.GRID[1:]])
    assert report['mainmetrics']['mean_dce_mm'] == pytest.approx(expected)
    assert report['dose_rows'][0]['dce_mm'] == 0


def test_target_pck_boundary_fixed_gt_elbow_scale_not_prediction():
    truth = np.random.default_rng(5).normal(size=(2, 14, 3))
    truth[:, 6] = [1, 0, 0]
    truth[:, 4] = [0, 0, 0]
    targets = trace_targets(truth)
    pred = targets.copy()
    pred[:, :, 11] += [0.199999, 0, 0]
    pred[:, :, 13] += [.200001, 0, 0]
    # Non-target wildly wrong predictions cannot change scale or target-only PCK.
    pred[:, :, 6] = [1000, 0, 0]
    pred[:, :, 4] = [0, 0, 0]
    report = adapter.measure_predictions(truth, pred[:, 0], pred, truth,
                                          np.repeat(truth[:, None], 6, axis=1))
    assert report['mainmetrics']['target_pck_pct'] == 50


def test_target_pck_exact_inclusive_boundary():
    truth = np.random.default_rng(83).normal(size=(2, 14, 3))
    truth[:, 6] = [1, 0, 0]
    truth[:, 4] = [0, 0, 0]
    truth[:, [7, 11, 13]] = 0
    trace = trace_targets(truth)
    trace[:, :, 11] = [.2, 0, 0]
    trace[:, :, 13] = [np.nextafter(.2, np.inf), 0, 0]
    report = adapter.measure_predictions(truth, trace[:, 0], trace, truth,
                                          np.repeat(truth[:, None], 6, axis=1))
    assert all(row['target_pck_pct'] == 50 for row in report['dose_rows'])


@pytest.mark.parametrize('bad', ['17joints', 'badcount', 'nan', 'complex', 'zeroref', 'd0attack', 'd0reference'])
def test_metric_replay_rejects_bad_arrays(poses, bad):
    truth, clean, ref = poses.copy(), poses.copy(), poses.copy()
    trace, ref_trace = trace_targets(poses), np.repeat(poses[:, None], 6, axis=1)
    if bad == '17joints':
        truth = np.zeros((3, 17, 3))
    elif bad == 'badcount':
        clean = clean[:1]
    elif bad == 'nan':
        trace[0, 1, 0, 0] = np.nan
    elif bad == 'complex':
        trace = trace.astype(np.complex128)
    elif bad == 'zeroref':
        truth[:, 6] = truth[:, 4]
    elif bad == 'd0attack':
        trace[0, 0, 0, 0] += .1
    else:
        ref_trace[0, 0, 0, 0] += .1
    with pytest.raises(ValueError):
        adapter.measure_predictions(truth, clean, trace, ref, ref_trace)
