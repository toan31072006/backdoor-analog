"""Data-only poisoning and kinematic-payload contract tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from attack import payload as skeleton  # noqa: E402
from attack.poison import PoisonedDataset, collate  # noqa: E402


class _ToyPoseDataset:
    def __init__(self, n):
        self.items = [
            {'csi': f'csi-{i:03d}', 'kpt': f'pose-{i:03d}'} for i in range(n)
        ]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def load_raw(csi_id):
        index = int(csi_id.rsplit('-', 1)[1])
        return np.full((3, 114, 10), 0.2 + index * 1e-3, dtype=np.float32)

    @staticmethod
    def load_pose(pose_id, frame_idx=None):
        del frame_idx
        index = int(pose_id.rsplit('-', 1)[1])
        joint = np.arange(17, dtype=np.float32)
        pose = np.stack((joint * 0.10, joint * 0.03, joint * -0.02), axis=-1)
        return (pose + index * 1e-4)[None]

    @staticmethod
    def normalize(raw):
        return np.asarray(raw, dtype=np.float32)


class _FixedTrigger:
    def inject(self, csi, dose, eps=0.3):
        return np.asarray(csi, dtype=np.float32) * (1.0 + float(dose) * eps)


def _poisoned(n=20, **kwargs):
    return PoisonedDataset(
        _ToyPoseDataset(n), _FixedTrigger(), mode='train', dataset='mmfi',
        pivot=1, theta_max_deg=40.0, dose_mode='linear', select='uniform',
        dose_min=0.2, dose_max=1.0, **kwargs)


def test_poison_count_uses_paper_floor_not_rounding():
    dataset = _poisoned(n=7, rho=0.25, seed=3)
    assert dataset.n_poison == 1
    assert dataset.n_poison == int(np.floor(0.25 * dataset.n_total))


def test_poison_plan_and_doses_are_reproducible_and_hashed():
    first = _poisoned(n=30, rho=0.4, seed=42).manifest()
    replay = _poisoned(n=30, rho=0.4, seed=42).manifest()
    other_seed = _poisoned(n=30, rho=0.4, seed=0).manifest()

    assert first == replay
    assert first['samples'] != other_seed['samples']
    assert first['poison_plan_sha256'] == replay['poison_plan_sha256']
    assert first['poison_plan_sha256'] != other_seed['poison_plan_sha256']
    assert len(first['poison_plan_sha256']) == 64
    assert all(0.2 <= row['dose'] <= 1.0 for row in first['samples'])


def test_fixed_trigger_victim_batch_contains_only_csi_and_pose():
    dataset = _poisoned(n=8, rho=0.5, seed=42)
    samples = [dataset[i] for i in range(len(dataset))]

    assert all(set(sample) == {'csi', 'pose'} for sample in samples)
    batch = collate(samples)
    assert set(batch) == {'csi', 'pose'}
    assert tuple(batch['csi'].shape) == (8, 3, 114, 10)
    assert tuple(batch['pose'].shape) == (8, 1, 17, 3)
    # Audit information remains available out of band, never in the victim batch.
    assert dataset.manifest()['n_poison'] == 4


@pytest.mark.parametrize(
    ('dataset_name', 'n_joints', 'root'),
    [('mmfi', 17, 0), ('person-in-wifi-3d', 14, 0)],
)
def test_payload_topology_is_a_connected_kinematic_tree(
        dataset_name, n_joints, root):
    skeleton.set_skeleton_config(dataset_name)

    assert len(skeleton._CURRENT_EDGES) == n_joints - 1
    assert set(skeleton.PARENT) == set(range(n_joints))
    assert skeleton.PARENT[root] is None
    assert all(
        skeleton.PARENT[joint] is not None
        for joint in range(n_joints) if joint != root)


def test_piw3d_official_joint_order_and_manuscript_pivots():
    skeleton.set_skeleton_config('person-in-wifi-3d')
    assert skeleton.PWIF3D_JOINT_NAMES == (
        'neck', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'left_hip', 'right_elbow', 'right_hip',
        'left_hand', 'left_knee', 'right_hand', 'right_knee',
        'left_ankle', 'right_ankle',
    )
    assert skeleton.descendants(3) == [6, 7, 10, 11, 13]
    assert skeleton.descendants(5) == [9, 12]
    assert skeleton.descendants(7) == [11, 13]

    model_edges = {
        frozenset(edge) for edge in skeleton.PWIF3D_MODEL_GRAPH_EDGES
    }
    kinematic_edges = {
        frozenset(edge) for edge in skeleton.PWIF3D_KINEMATIC_EDGES
    }
    assert model_edges - kinematic_edges == {frozenset((5, 7))}


def test_canonical_piw3d_config_targets_only_right_hip_leg_subtree():
    config_path = (
        Path(__file__).resolve().parents[1]
        / 'configs' / 'hpeli' / 'attack_ln.yaml'
    )
    with config_path.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    skeleton.set_skeleton_config('person-in-wifi-3d')
    pivot = config['pivot']
    target_joints = skeleton.descendants(pivot)

    assert pivot == 7
    assert config['lr'] == pytest.approx(1e-3)
    assert config['optimizer'] == 'adamw'
    assert config['payload_axis'] == pytest.approx([0.0, 0.0, 1.0])
    assert skeleton.PWIF3D_JOINT_NAMES[pivot] == 'right_hip'
    assert target_joints == [11, 13]
    assert [skeleton.PWIF3D_JOINT_NAMES[joint] for joint in target_joints] == [
        'right_knee', 'right_ankle',
    ]


@pytest.mark.parametrize('relative_path', [
    'configs/attack.yaml', 'configs/hpeli/attack_ln.yaml',
])
def test_piw3d_config_preserves_original_wbackdoor_parameters(relative_path):
    from run_experiments import _apply_model_overrides, _prepare_run_config

    config_path = Path(__file__).resolve().parents[1] / relative_path
    with config_path.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    expected = {
        'experiment_name': 'one-person', 'model': 'hpeli', 'pretrained': False,
        'top_k': 6, 'aoa_spread': 0.6, 'eps': 0.3,
        'theta_max_deg': 60.0, 'dose_mode': 'linear',
        'poison_select': 'diverse', 'rho': 0.1,
        'dose_min': 0.2, 'dose_max': 1.0,
        'dose_grid': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'batch_size': 32, 'lr': 0.001, 'epochs': 200,
        'device': None, 'data_parallel': False, 'seed': 0,
    }
    for key, value in expected.items():
        assert config[key] == value, key
    assert config['pivot'] == 7  # intended right leg, not the original wrong tree
    assert config['payload_axis'] == [0.0, 0.0, 1.0]
    resolved = _prepare_run_config(
        'hpeli', 'micro_dropper',
        _apply_model_overrides(config.copy(), 'hpeli', 'person-in-wifi-3d'),
        device=None, epochs=None, seed=config['seed'])
    for key, value in expected.items():
        assert resolved[key] == value, key
    assert resolved['pivot'] == 7


@pytest.mark.parametrize('dataset_name', ['mmfi', 'person-in-wifi-3d'])
def test_rotation_preserves_every_bone_for_every_nonleaf_pivot(dataset_name):
    skeleton.set_skeleton_config(dataset_name)
    rng = np.random.default_rng(11)
    pose = rng.normal(size=(4, skeleton.N_JOINTS, 3))
    before = skeleton.all_bone_lengths(pose)

    tested = 0
    for pivot in range(skeleton.N_JOINTS):
        if not skeleton.descendants(pivot):
            continue
        transformed = skeleton.make_target_pose(
            pose, pivot=pivot, dose=0.73, theta_max=np.deg2rad(67.0),
            axis=(0.3, -0.4, 0.5))
        after = skeleton.all_bone_lengths(transformed)
        np.testing.assert_allclose(after, before, rtol=1e-10, atol=1e-10)
        tested += 1

    assert tested > 0
