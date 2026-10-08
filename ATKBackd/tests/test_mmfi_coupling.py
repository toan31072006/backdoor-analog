"""Toy-data integration checks for MMFi dose coupling, covers and evaluation."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack import payload
from attack.poison import PoisonedDataset, collate
from attack.traditional import WaNetTrigger
from eval import metrics
from train_backdoor import (_RESULT_SCHEMA, _load_cached_result,
                            _save_cached_result, evaluate)


def _template_pose():
    joint = np.arange(17, dtype=np.float32)
    return np.stack((joint * 0.08, (joint * 7 % 5) * 0.07,
                     (joint * 5 % 7) * 0.04), axis=-1)


class _ToyMMFi:
    def __init__(self, n=30):
        self.items = [{'csi': f'csi:{i}', 'kpt': f'pose:{i}', 'frame_idx': i}
                      for i in range(n)]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def load_raw(csi_id):
        index = int(csi_id.split(':')[1])
        antenna = np.arange(3, dtype=np.float32)[:, None, None] * 0.025
        frequency = np.linspace(0.0, 0.20, 114, dtype=np.float32)[None, :, None]
        packet = np.linspace(0.0, 0.05, 10, dtype=np.float32)[None, None, :]
        return (0.2 + index * 0.002 + antenna + frequency + packet).astype(np.float32)

    @staticmethod
    def load_pose(pose_id, frame_idx=None):
        index = int(pose_id.split(':')[1])
        if frame_idx is not None:
            assert index == frame_idx
        translation = index * np.array([0.001, -0.0005, 0.0003], dtype=np.float32)
        return (_template_pose() + translation)[None]

    @staticmethod
    def normalize(raw):
        return np.asarray(raw, dtype=np.float32)


class _FixedTrigger:
    requires_deferred_injection = False

    @staticmethod
    def inject(csi, dose, eps=0.185):
        return np.clip(np.asarray(csi, dtype=np.float32) * (1.0 + dose * eps),
                       0.0, 1.0).astype(np.float32)


class _DeferredTrigger:
    requires_deferred_injection = True


class _ToyVictim(torch.nn.Module):
    """A known translation-biased pose predictor sensitive only to CSI."""

    def __init__(self):
        super().__init__()
        self.register_buffer('template', torch.from_numpy(_template_pose())[None, None])
        self.inputs = []

    def forward(self, csi):
        assert torch.is_tensor(csi) and tuple(csi.shape[1:]) == (3, 114, 10)
        self.inputs.append(csi.detach().cpu().clone())
        mean = csi.mean(dim=(1, 2, 3))
        shift = torch.stack((0.08 + 0.18 * mean, -0.03 + 0.07 * mean,
                             0.04 - 0.11 * mean), dim=-1)
        return self.template + shift[:, None, None, :], None


@pytest.fixture(autouse=True)
def _mmfi_topology():
    previous = 'mmfi' if payload.N_JOINTS == 17 else 'person-in-wifi-3d'
    payload.set_skeleton_config('mmfi')
    yield
    payload.set_skeleton_config(previous)


def _dataset(base=None, trigger=None, **overrides):
    options = dict(mode='train', dataset='mmfi', rho=0.4, seed=42,
                   pivot=1, theta_max_deg=40.0, dose_mode='linear',
                   dose_min=0.2, dose_max=1.0, eps=0.185, select='uniform')
    options.update(overrides)
    return PoisonedDataset(_ToyMMFi() if base is None else base,
                           _FixedTrigger() if trigger is None else trigger,
                           **options)


def _sha256(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode('utf-8')).hexdigest()


def test_shuffling_changes_only_payload_pairing_with_identical_marginals():
    paired = _dataset(dose_coupling='paired')
    shuffled = _dataset(dose_coupling='shuffled')
    assert paired.n_poison == shuffled.n_poison == 12
    assert paired.poison_plan == shuffled.poison_plan
    assert paired.poison_idx == shuffled.poison_idx
    assert paired.dose_of == shuffled.dose_of
    assert paired.payload_dose_of == paired.dose_of
    assert sorted(paired.payload_dose_of.values()) == sorted(shuffled.payload_dose_of.values())
    assert any(paired.payload_dose_of[i] != shuffled.payload_dose_of[i]
               for i in paired.poison_idx)

    changed_labels = 0
    for i in range(len(paired)):
        first, second = paired[i], shuffled[i]
        assert set(first) == set(second) == {'csi', 'pose'}
        np.testing.assert_array_equal(first['csi'], second['csi'])
        clean = paired.base.load_pose(f'pose:{i}', i)
        if i in paired.poison_idx:
            for dataset, sample in ((paired, first), (shuffled, second)):
                expected = payload.make_target_pose(
                    clean, pivot=1, dose=dataset.payload_dose_of[i],
                    theta_max=np.deg2rad(40.0), mode='linear')
                np.testing.assert_array_equal(sample['pose'], expected)
            changed_labels += not np.array_equal(first['pose'], second['pose'])
        else:
            np.testing.assert_array_equal(first['pose'], clean)
            np.testing.assert_array_equal(second['pose'], clean)
    assert changed_labels > 0
    victim_batch = collate([shuffled[i] for i in range(len(shuffled))])
    assert set(victim_batch) == {'csi', 'pose'}
    assert tuple(victim_batch['csi'].shape) == (30, 3, 114, 10)
    assert tuple(victim_batch['pose'].shape) == (30, 1, 17, 3)


def test_private_manifests_hash_the_input_plan_and_actual_payload_assignment():
    paired = _dataset(dose_coupling='paired')
    shuffled = _dataset(dose_coupling='shuffled')
    audit = shuffled.manifest()
    plan = [[i, dose] for i, dose in shuffled.poison_plan]
    assignment = [[i, dose, shuffled.payload_dose_of[i]]
                  for i, dose in shuffled.poison_plan]
    assert paired.manifest()['poison_plan_sha256'] == audit['poison_plan_sha256'] == _sha256(plan)
    assert audit['schema'] >= 5
    assert audit['dose_coupling'] == 'shuffled'
    assert audit['coupling_sha256'] == _sha256(assignment)
    assert audit['coupling_sha256'] != _sha256([[i, dose, dose] for i, dose in paired.poison_plan])
    assert audit == _dataset(dose_coupling='shuffled').manifest()
    assert sorted(row['payload_dose'] for row in audit['samples']) == sorted(
        row['dose'] for row in audit['samples'])
    for row in audit['samples']:
        assert row['payload_dose'] == shuffled.payload_dose_of[row['index']]
        assert row['csi_id'] == f'csi:{row["index"]}'
        assert row['pose_id'] == f'pose:{row["index"]}'
    # Auditing detects a changed assignment even if both marginals stay fixed.
    first, second = next(iter(shuffled.poison_plan))[0], shuffled.poison_plan[1][0]
    shuffled.payload_dose_of[first], shuffled.payload_dose_of[second] = (
        shuffled.payload_dose_of[second], shuffled.payload_dose_of[first])
    changed = shuffled.manifest()
    assert changed['poison_plan_sha256'] == audit['poison_plan_sha256']
    assert changed['coupling_sha256'] != audit['coupling_sha256']


def test_wanet_covers_are_disjoint_clean_label_and_order_independent():
    first = _dataset(trigger=WaNetTrigger(seed=42), cover_ratio=0.2)
    replay = _dataset(trigger=WaNetTrigger(seed=42), cover_ratio=0.2)
    assert first.n_poison == 12 and first.n_cover == 6
    assert first.poison_idx.isdisjoint(first.cover_idx)
    assert first.cover_idx == replay.cover_idx
    assert first.poison_plan == _dataset().poison_plan
    audit = first.manifest()
    assert audit['cover_indices'] == sorted(first.cover_idx)
    assert audit['n_cover'] == 6 and audit['cover_ratio'] == 0.2
    forward = {i: first[i] for i in range(len(first))}
    backward = {i: replay[i] for i in reversed(range(len(replay)))}
    for i in range(len(first)):
        np.testing.assert_array_equal(forward[i]['csi'], backward[i]['csi'])
        np.testing.assert_array_equal(forward[i]['pose'], backward[i]['pose'])
        assert set(forward[i]) == {'csi', 'pose'}
        if i not in first.poison_idx:
            np.testing.assert_array_equal(forward[i]['pose'], first.base.load_pose(f'pose:{i}', i))
        if i in first.cover_idx:
            raw = first.base.load_raw(f'csi:{i}')
            assert not np.array_equal(forward[i]['csi'], raw)
            assert np.isfinite(forward[i]['csi']).all()
            assert 0.0 <= float(forward[i]['csi'].min()) <= float(forward[i]['csi'].max()) <= 1.0
        elif i not in first.poison_idx:
            np.testing.assert_array_equal(forward[i]['csi'], first.base.load_raw(f'csi:{i}'))
    assert set(collate(list(forward.values()))) == {'csi', 'pose'}


@pytest.mark.parametrize('options', [
    {'dose_coupling': 'unrecognized'},
    {'cover_ratio': -0.01},
    {'cover_ratio': 1.01},
    {'rho': 0.81, 'cover_ratio': 0.2},
    {'dose_min': 0.8, 'dose_max': 0.2},
    {'dose_min': -0.1},
    {'dose_max': 1.1},
    {'rho': -0.1},
])
def test_invalid_poison_and_cover_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        _dataset(trigger=WaNetTrigger(seed=42), **options)


def test_covers_require_noise_injection_and_shuffling_requires_fixed_trigger():
    with pytest.raises(ValueError, match='noise_inject'):
        _dataset(cover_ratio=0.2)
    with pytest.raises(ValueError, match='fixed triggers'):
        _dataset(trigger=_DeferredTrigger(), dose_coupling='shuffled')


def _evaluation_config(**overrides):
    options = dict(experiment_name='mmfi', pivot=1, payload_axis=[0.0, 0.0, 1.0],
                   eps=0.185, theta_max_deg=40.0, dose_mode='linear', batch_size=2,
                   dose_grid=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], epochs=1, seed=42)
    options.update(overrides)
    return options


def test_six_clean_to_target_baselines_use_same_model_and_dose_specific_target():
    base, model, trigger = _ToyMMFi(n=4), _ToyVictim(), _FixedTrigger()
    cfg = _evaluation_config()
    raw = torch.from_numpy(np.stack([base.load_raw(f'csi:{i}') for i in range(len(base))]))
    with torch.no_grad():
        clean_prediction = model(raw)[0].numpy()[:, 0]
    model.inputs.clear()
    clean_true = np.stack([base.load_pose(f'pose:{i}', i) for i in range(len(base))])[:, 0]
    result = evaluate(model, base, trigger, cfg, device='cpu')
    assert _RESULT_SCHEMA == 10
    assert result['dose_grid'] == cfg['dose_grid']
    assert len(result['clean_to_target_tmpjpe']) == 6
    assert payload.descendants(1) == [2, 3]

    expected = []
    for dose in cfg['dose_grid']:
        # Independent z-axis rotation of exactly the two attacked MMFi joints.
        target = clean_true.astype(np.float64, copy=True)
        angle = np.deg2rad(40.0) * dose
        for joint in (2, 3):
            relative = clean_true[:, joint] - clean_true[:, 1]
            target[:, joint, 0] = clean_true[:, 1, 0] + relative[:, 0] * np.cos(angle) - relative[:, 1] * np.sin(angle)
            target[:, joint, 1] = clean_true[:, 1, 1] + relative[:, 0] * np.sin(angle) + relative[:, 1] * np.cos(angle)
        errors = np.linalg.norm(clean_prediction[:, [2, 3]] - target[:, [2, 3]], axis=-1)
        expected.append(float(errors.mean()))
    np.testing.assert_allclose(result['clean_to_target_tmpjpe'], expected, rtol=1e-6, atol=1e-7)
    assert len(set(np.round(expected, 6))) > 1
    assert result['clean_to_target_tmpjpe'][0] == pytest.approx(result['clean_target_floor'])
    assert result['tmpjpe'][0] == pytest.approx(expected[0])
    assert result['displacement'][0] == 0.0
    assert result['nontarget_mpjpe'][0] == 0.0
    assert result['asr_curve'][0]['asr'] == 0.0
    # The biased prediction makes raw and PA errors observably different.
    assert result['clean_to_target_tmpjpe'][0] > 0.1
    assert result['tpampjpe'][0] < 1e-5
    assert result['reference_dose'] == 1.0
    assert len(model.inputs) == 14  # One clean pass plus six two-batch dose passes.
    for first, zero_dose in zip(model.inputs[:2], model.inputs[2:4]):
        torch.testing.assert_close(first, zero_dose, rtol=0.0, atol=0.0)


def test_two_joint_target_metric_is_raw_unaligned_euclidean_error():
    target = np.zeros((1, 17, 3), dtype=np.float32)
    prediction = np.full_like(target, 100.0)
    prediction[0, 2] = [3.0, 0.0, 0.0]
    prediction[0, 3] = [0.0, 4.0, 0.0]
    np.testing.assert_array_equal(metrics.target_mpjpe(prediction, target, pivot=1), [3.5])


def test_zero_dose_dataset_retains_exact_clean_csi_and_pose():
    base = _ToyMMFi(n=4)
    clean = _dataset(base=base, mode='clean')
    zero = _dataset(base=base, mode='trigger@dose', fixed_dose=0.0)
    for i in range(len(base)):
        np.testing.assert_array_equal(zero[i]['csi'], clean[i]['csi'])
        np.testing.assert_array_equal(zero[i]['target'], clean[i]['pose'])
        np.testing.assert_array_equal(zero[i]['pose'], clean[i]['pose'])


def test_schema_ten_cache_keeps_baselines_and_rejects_older_schema(tmp_path):
    cfg = _evaluation_config()
    result = {'clean_to_target_tmpjpe': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    _save_cached_result(tmp_path, cfg, result)
    assert _load_cached_result(tmp_path, cfg) == result
    cache_path = tmp_path / 'eval_cache.json'
    blob = json.loads(cache_path.read_text(encoding='utf-8'))
    assert blob['result_schema'] == 10
    blob['result_schema'] = 9
    cache_path.write_text(json.dumps(blob), encoding='utf-8')
    assert _load_cached_result(tmp_path, cfg) is None
