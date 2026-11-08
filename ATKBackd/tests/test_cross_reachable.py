"""Six-joint Cross v3 regression tests. Invented poses, NOT dataset results.

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
    a, b = [arm[0] for arm in spec['arms']]
    center = .5*(pose[a]+pose[b])
    width = np.linalg.norm(pose[b]-pose[a])
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
    a, b = [arm[0] for arm in spec['arms']]
    center = .5*(original[a]+original[b])
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
    a, b = [arm[0] for arm in spec['arms']]
    center = .5*(p[a]+p[b])
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


def test_nominal_shoulders_move_with_all_anchor_bones_preserved(geometry):
    module, p, edges = geometry
    spec = module.specification('cross')
    records = []
    q = module.cross_endpoint(p, spec, diagnostics=records)
    assert len(spec['target_joints']) == 6
    for s, _, _ in spec['arms']:
        assert np.linalg.norm(q[s]-p[s]) > .001
    np.testing.assert_allclose(lengths(q, edges), lengths(p, edges), atol=1e-12, rtol=0)
    assert [r['adapted'] for r in records] == [False, False]
    assert all(r['shoulder_displacement_m'] > .001 for r in records)


def test_inner_reach_annulus_and_short_bone_remain_valid(geometry):
    module, p, edges = geometry
    spec = module.specification('cross')
    set_arm(p, spec['arms'][0], .55, .06)
    q = module.target_pose(p, 1., spec)
    assert_semantics(module, p, q, spec)
    np.testing.assert_allclose(lengths(q, edges), lengths(p, edges), atol=1e-12, rtol=0)
    s, _, w = spec['arms'][0]
    np.testing.assert_allclose(np.linalg.norm(q[w]-q[s]), .49+spec['reach_margin'], atol=1e-12)


def test_locked_non_crossable_shoulder_refuses_whole_batch(geometry):
    module, p, _ = geometry
    spec = module.specification('cross')
    bad = p.copy()
    s = spec['arms'][0][0]
    if len(p) == 17:
        bad[s] = bad[8]+[0, 0, -.05]
    else:
        bad[s] = .5*(bad[0]+bad[7])
    set_arm(bad, spec['arms'][0], .02, .02)
    with pytest.raises(ValueError, match='no valid shoulder-orbit candidate'):
        module.target_pose(np.stack([p, bad]), .2, spec)


def test_changed_version_refuses_v1_cross_but_nod_definition_is_unchanged(geometry):
    module, p, _ = geometry
    cross = module.specification('cross')
    assert cross['version'].endswith('_v3')
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
        a, b = [arm[0] for arm in spec['arms']]
        half_width = np.linalg.norm(p[a]-p[b])/2
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
    runner.verify_geometry(plan)
    audit = runner.read(tmp_path / 'geometry_audit.json')
    assert audit['status'] == 'PASS' and audit['n_checked'] == 6
    audit['status'] = 'FAILED'
    (tmp_path / 'geometry_audit.json').write_text(__import__('json').dumps(audit))
    with pytest.raises(ValueError, match='changed complete'):
        runner.verify_geometry(plan)


@pytest.mark.parametrize('dose', [0., .2, .4, .6, .8, 1.])
def test_v2_impossible_forearm_lengths_now_cross_with_moving_shoulders(geometry, dose):
    module, p, edges = geometry
    spec = module.specification('cross')
    # Synthetic analogue of the second reported failure, NOT the raw frame.
    set_arm(p, spec['arms'][0], .1076, .0157)
    a, b = [arm[0] for arm in spec['arms']]
    lateral, _, _ = module.body_frame(p)
    assert .1076+.0157 < np.dot(.5*(p[a]+p[b])-p[a], lateral)
    q = module.target_pose(p, dose, spec)
    np.testing.assert_allclose(lengths(q, edges), lengths(p, edges), atol=1e-12, rtol=0)
    for arm, anchors in zip(spec['arms'], spec['shoulder_anchors']):
        for anchor in anchors:
            np.testing.assert_allclose(np.linalg.norm(q[arm[0]]-q[anchor]),
                np.linalg.norm(p[arm[0]]-p[anchor]), atol=1e-12, rtol=0)
    if dose == 1:
        assert_semantics(module, p, q, spec)


def test_geometry_records_all_failures_and_audits_nod_without_success_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import run_mmfi_action_targets as runner
    import attack.action_targets as module
    p = np.zeros((17, 3))
    p[8] = [0, 0, .61]; p[9] = [0, 0, .75]; p[10] = [0, 0, .95]
    p[11] = [-.13, 0, .61]; p[14] = [.13, 0, .61]
    spec = module.specification('cross')
    set_arm(p, spec['arms'][0], .3, .28); set_arm(p, spec['arms'][1], .3, .28)
    bad = p.copy(); bad[11] = bad[8]+[0, 0, -.05]
    set_arm(bad, spec['arms'][0], .02, .02)  # valid bones, locked shoulder, no feasible orbit
    samples = [bad, p, bad]
    visited = []
    def load(index):
        visited.append(index)
        return samples[index][None].copy()
    cells = [dict(action=a, cfg=dict(action_payload=module.specification(a),
        target_joints=module.specification(a)['target_joints'])) for a in ('cross', 'nod')]
    plan = dict(outdir=str(tmp_path), plan_sha256='synthetic-failed-audit', cells=cells)
    monkeypatch.setattr(runner, 'build_plan', lambda args: plan)
    monkeypatch.setattr(runner, 'dataset', lambda cfg, split: (None, [['fixture', i] for i in range(3)]))
    pairs = SimpleNamespace(poison_idx={0, 1, 2}, n_poison=3, clean_pose=load,
        manifest=lambda: dict(poison_plan_sha256='same-ids-not-filtered'))
    monkeypatch.setattr(runner, 'action_data', lambda *a, **k: pairs)
    monkeypatch.setattr(runner, 'geometry_figure', lambda path, *a, **k: path.write_bytes(b'SYNTHETIC ONLY'))
    with pytest.raises(runner.GeometryAuditFailure, match='2 invalid poses of 6'):
        runner.geometry(SimpleNamespace())
    assert {0, 1, 2}.issubset(visited)
    audit = runner.read(tmp_path / 'geometry_audit.json')
    assert audit['status'] == 'FAILED' and audit['n_checked'] == 6
    assert audit['records']['cross']['n_failed'] == 2
    assert audit['records']['nod']['n_checked'] == 3 and audit['records']['nod']['n_failed'] == 0
    failures = runner.read(tmp_path / 'geometry_failures.json')['failures']
    assert [f['index'] for f in failures] == [0, 2]
    assert not (tmp_path / 'geometry.json').exists()
    assert not (tmp_path / 'prepared.json').exists()


def test_analytic_candidate_does_not_miss_narrow_reachable_interval():
    from attack.cross_shoulders import _candidate_fractions, orbit_position
    phi = .45*np.pi
    radial = .2*np.array([np.cos(phi), -np.sin(phi), 0])
    motion = dict(origin=radial, center=np.zeros(3), radial=radial, axis=np.array([0., 0., 1.]), angle=np.pi)
    center, inward, far = np.array([.21, 0, 0]), np.array([1., 0, 0]), .01001
    spec = dict(shoulder_candidate_fractions=[.2, .3, .4, .5, .6, .7, .8, .9, 1.], shoulder_root_steps=60)
    h = lambda f: np.dot(center-orbit_position(motion, f), inward)
    assert all(h(f) > far for f in spec['shoulder_candidate_fractions'])
    candidates = list(_candidate_fractions(motion, center, inward, far, spec))
    assert any(0 < h(f) < far for f in candidates)
