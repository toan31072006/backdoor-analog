"""Reach-adaptive v2 regression tests. Invented poses, NOT dataset results.

The short-arm fixture is a synthetic analogue of the reported TRAIN failure;
no raw MM-Fi frame is distributed with the code.
"""
import copy
import importlib
import os

os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
import numpy as np
import pytest

from attack.action_targets import _two_bone_endpoint, _direction_at, _rotate
from run_mmfi_action_targets import new_cross_reach_audit, add_cross_reach_records


@pytest.fixture(params=['mmfi', 'piw'])
def geometry(request):
    module = importlib.import_module('attack.' + ('action_targets' if request.param == 'mmfi' else 'piw_action_targets'))
    if request.param == 'mmfi':
        pose = np.array([[0, 0, 1], [.1, 0, .96], [.11, 0, .55], [.12, 0, .12],
            [-.1, 0, .96], [-.11, 0, .55], [-.12, 0, .12], [0, 0, 1.2],
            [0, 0, 1.61], [0, 0, 1.75], [0, 0, 1.95],
            [-.13, 0, 1.61], [-.13, 0, 1.31], [-.13, 0, 1.03],
            [.13, 0, 1.61], [.13, 0, 1.31], [.13, 0, 1.03]], dtype=np.float64)
        from attack.payload import MMFI_EDGES
        edges = MMFI_EDGES
    else:
        pose = np.array([[0, 0, 1.75], [0, 0, 1.95], [.13, 0, 1.61],
            [-.13, 0, 1.61], [.13, 0, 1.31], [.10, 0, 1.],
            [-.13, 0, 1.31], [-.10, 0, 1.], [.13, 0, 1.03],
            [.11, 0, .55], [-.13, 0, 1.03], [-.11, 0, .55],
            [.12, 0, .12], [-.12, 0, .12]], dtype=np.float64)
        edges = module.EDGES
    return module, pose, edges


def lengths(pose, edges):
    return np.array([np.linalg.norm(pose[a]-pose[b]) for a, b in edges])


def set_arm(pose, arm, upper, lower):
    s, e, w = arm
    pose[e] = pose[s] + [0, 0, -upper]
    pose[w] = pose[e] + [0, 0, -lower]


def nominal_endpoint(module, pose, spec):
    """Original v1 radial-only IK, for a regression counterexample/reference."""
    lateral, up, forward = module.body_frame(pose)
    center = .5*(pose[spec['pivots'][0]]+pose[spec['pivots'][1]])
    width = np.linalg.norm(pose[spec['pivots'][1]]-pose[spec['pivots'][0]])
    torso = np.linalg.norm(pose[8]-pose[0]) if len(pose) == 17 else np.linalg.norm(center-.5*(pose[5]+pose[7]))
    result = pose.copy()
    for k, (s, e, w) in enumerate(spec['arms']):
        sign = 1 if k == 0 else -1
        goal = (center+sign*spec['wrist_across_width']*width*lateral
            -(spec['wrist_below_torso']+k*spec['wrist_height_separation_torso'])*torso*up
            +spec['wrist_front_width'][k]*width*forward)
        pole = -up-sign*spec['elbow_outward']*lateral+spec['elbow_forward']*forward
        result[e], result[w] = _two_bone_endpoint(pose[s], pose[e], pose[w], goal, pole, spec['reach_margin'])
    return result


def assert_semantics(module, original, result, spec):
    lateral, _, forward = module.body_frame(original)
    center = .5*(original[spec['pivots'][0]]+original[spec['pivots'][1]])
    for k, (_, _, w) in enumerate(spec['arms']):
        assert (1 if k == 0 else -1)*np.dot(result[w]-center, lateral) > 0
        assert np.dot(result[w]-center, forward) > 0


@pytest.mark.parametrize('dose', [0., .2, .4, .6, .8, 1.])
def test_short_arm_no_stretch_or_skipped_frame_at_any_dose(geometry, dose):
    module, p, edges = geometry
    spec = module.specification('cross')
    set_arm(p, spec['arms'][0], .092, .077)
    before = p.copy()
    # The old formula fails even though 169mm > the 130mm mid-plane distance.
    old = nominal_endpoint(module, p, spec)
    lateral, _, _ = module.body_frame(p)
    center = .5*(p[spec['pivots'][0]]+p[spec['pivots'][1]])
    assert np.dot(old[spec['arms'][0][2]]-center, lateral) < 0
    q = module.target_pose(p, dose, spec)
    np.testing.assert_array_equal(p, before)
    np.testing.assert_allclose(lengths(q, edges), lengths(p, edges), atol=1e-12, rtol=0)
    other = [j for j in range(len(p)) if j not in spec['target_joints']]
    np.testing.assert_array_equal(q[other], p[other])
    if dose == 0:
        np.testing.assert_array_equal(q, p)
    if dose == 1:
        assert_semantics(module, p, q, spec)


def test_reachable_nominal_targets_are_bitwise_unchanged(geometry):
    module, p, _ = geometry
    spec = module.specification('cross')
    expected = nominal_endpoint(module, p, spec)
    records = []
    q = module.cross_endpoint(p, spec, diagnostics=records)
    np.testing.assert_array_equal(q, expected)
    assert [r['adapted'] for r in records] == [False, False]
    for dose in (.2, .4, .6, .8):
        wanted = p.copy()
        for s, e, w in spec['arms']:
            upper, lower = p[e]-p[s], p[w]-p[e]
            wanted[e] = p[s]+np.linalg.norm(upper)*_direction_at(upper, expected[e]-p[s], dose)
            wanted[w] = wanted[e]+np.linalg.norm(lower)*_direction_at(lower, expected[w]-expected[e], dose)
        np.testing.assert_array_equal(module.target_pose(p, dose, spec), wanted)


def test_inner_reach_annulus_and_short_bone_remain_valid(geometry):
    module, p, edges = geometry
    spec = module.specification('cross')
    set_arm(p, spec['arms'][0], .55, .06)
    q = module.target_pose(p, 1., spec)
    assert_semantics(module, p, q, spec)
    np.testing.assert_allclose(lengths(q, edges), lengths(p, edges), atol=1e-12, rtol=0)
    s, _, w = spec['arms'][0]
    np.testing.assert_allclose(np.linalg.norm(q[w]-q[s]), .49+spec['reach_margin'], atol=1e-12)


def test_truly_non_crossable_pose_refuses_whole_batch(geometry):
    module, p, _ = geometry
    spec = module.specification('cross')
    bad = p.copy()
    set_arm(bad, spec['arms'][0], .02, .02)
    with pytest.raises(ValueError, match='cannot reach across.*mid-plane'):
        module.target_pose(np.stack([p, bad]), .2, spec)


def test_changed_version_refuses_v1_cross_but_nod_definition_is_unchanged(geometry):
    module, p, _ = geometry
    cross = module.specification('cross')
    assert cross['version'].endswith('_v2')
    old = copy.deepcopy(cross)
    old['version'] = module.VERSION
    old.pop('wrist_reach_policy'); old.pop('wrist_cross_slack_fraction')
    with pytest.raises(ValueError, match='Unrecognized'):
        module.target_pose(p, .6, old)
    nod = module.specification('nod')
    assert nod['version'] == module.VERSION and nod['version'].endswith('_v1')
    pivot, head = nod['pivots'][0], nod['target_joints'][0]
    lateral, _, _ = module.body_frame(p)
    for dose in (0., .2, .4, .6, .8, 1.):
        expected = p.copy()
        expected[head] = p[pivot]+_rotate(p[head]-p[pivot], lateral, np.deg2rad(40.)*dose)
        np.testing.assert_array_equal(module.target_pose(p, dose, nod), expected)


def test_adapted_target_is_deterministic_rigid_coordinate_equivariant(geometry):
    module, p, _ = geometry
    spec = module.specification('cross')
    set_arm(p, spec['arms'][0], .092, .077)
    rotation, _ = np.linalg.qr(np.random.default_rng(17).normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    translation = np.array([2., -3., .7])
    for dose in (0., .2, .6, 1.):
        wanted = module.target_pose(p, dose, spec) @ rotation.T + translation
        q = module.target_pose(p @ rotation.T+translation, dose, spec)
        np.testing.assert_allclose(q, wanted, atol=3e-12, rtol=0)
    np.testing.assert_array_equal(module.target_pose(p, 1., spec), module.target_pose(p, 1., spec))


def test_random_feasible_short_asymmetric_arms_keep_bones_and_semantics(geometry):
    module, p, edges = geometry
    spec = module.specification('cross')
    rng = np.random.default_rng(613)
    for _ in range(100):
        q = p.copy()
        half_width = np.linalg.norm(p[spec['pivots'][0]]-p[spec['pivots'][1]])/2
        for arm in spec['arms']:
            reach = rng.uniform(half_width+.002, .85)
            upper = reach*rng.uniform(.1, .9)
            set_arm(q, arm, upper, reach-upper)
        target = module.target_pose(q, 1., spec)
        assert_semantics(module, q, target, spec)
        np.testing.assert_allclose(lengths(target, edges), lengths(q, edges), atol=1e-12, rtol=0)


def test_adaptation_counts_and_clearances_are_audited_not_hidden(geometry):
    module, p, _ = geometry
    spec = module.specification('cross')
    audit = new_cross_reach_audit()
    for index in (0, 1):
        if index: set_arm(p, spec['arms'][0], .092, .077)
        records = []
        module.cross_endpoint(p, spec, diagnostics=records)
        add_cross_reach_records(audit, index, records)
    assert (audit['poses'], audit['arms'], audit['adapted_poses'], audit['adapted_arms']) == (2, 4, 1, 1)
    assert audit['max_correction_train_index'] == 1
    assert audit['max_goal_correction_m'] > .01
    assert audit['min_opposite_side_m'] > 0 and audit['min_front_m'] > 0


def test_mmfi_geometry_audits_real_payload_on_every_fixture_poison(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import run_mmfi_action_targets as runner
    import attack.action_targets as module
    # Synthetic labels only; actual geometry loop/JSON and payload code execute.
    p = np.zeros((17, 3))
    p[8] = [0, 0, .61]; p[9] = [0, 0, .75]; p[10] = [0, 0, .95]
    p[11] = [-.13, 0, .61]; p[14] = [.13, 0, .61]
    spec = module.specification('cross')
    set_arm(p, spec['arms'][0], .092, .077)
    set_arm(p, spec['arms'][1], .3, .28)
    rotation, _ = np.linalg.qr(np.random.default_rng(61).normal(size=(3, 3)))
    p = (p @ rotation.T + [2., -3., .8]).astype(np.float32)
    cells = [dict(action=a, cfg=dict(action_payload=module.specification(a),
        target_joints=module.specification(a)['target_joints'])) for a in ('cross', 'nod')]
    plan = dict(outdir=str(tmp_path), plan_sha256='synthetic-fixture', cells=cells)
    monkeypatch.setattr(runner, 'build_plan', lambda args: plan)
    monkeypatch.setattr(runner, 'dataset', lambda cfg, split: (None, [['synthetic', i] for i in range(3)]))
    data = SimpleNamespace(poison_idx={0, 1, 2}, n_poison=3,
        clean_pose=lambda i: p[None].copy(), manifest=lambda: dict(poison_plan_sha256='fixture'))
    monkeypatch.setattr(runner, 'action_data', lambda *a, **k: data)
    monkeypatch.setattr(runner, 'geometry_figure', lambda path, *a, **k: path.write_bytes(b'SYNTHETIC ONLY'))
    runner.geometry(SimpleNamespace())
    record = runner.read(tmp_path / 'geometry.json')['records']
    reach = record['cross']['reach_adaptation']
    assert (reach['poses'], reach['arms'], reach['adapted_poses'], reach['adapted_arms']) == (3, 6, 3, 3)
    assert record['cross']['max_bone_delta_m'] < 1e-12
    assert 'reach_adaptation' not in record['nod']
