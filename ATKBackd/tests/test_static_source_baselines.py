"""Independent formula/provenance/state tests, never real benchmark scores."""
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack.static_source_baselines import (SOURCES, BppSourceCSITrigger,
    SIGSourceCSITrigger, build_static_source_trigger, resolve_static_source_config)


class TrainingData:
    def __init__(self, *, split='training', count=8, shape=(3, 114, 10)):
        self.split = split
        self.items = [dict(csi=str(i), kpt='pose.npy', frame_idx=i) for i in range(count)]
        self.inputs = np.random.default_rng(17).uniform(.1, .9, (count, *shape)).astype(np.float32)
        self.loads = []

    def __len__(self):
        return len(self.items)

    def load_raw(self, index):
        self.loads.append(int(index))
        return self.inputs[int(index)].copy()


def test_sig_backdoorbench_default40_f6_zero_based_columns_without_learning():
    trigger = SIGSourceCSITrigger(n_ant=2, n_sub=7, n_pkt=19)
    x = np.full(trigger.shape, .5, np.float32)
    for d in [0, .2, .4, .6, .8, 1]:
        expected = np.clip(x.astype(float) + d * (40/255) *
            np.sin(2*np.pi*np.arange(19)*6/19)[None, None, :], 0, 1).astype(np.float32)
        np.testing.assert_array_equal(trigger.inject(x, d), expected)
    assert np.all(trigger.pattern[..., 0] == 0)
    assert not hasattr(trigger, 'noise_inject')


def test_bpp_backdoorbench_default8_quantization_endpoint_and_linear_dose():
    data = TrainingData()
    trigger = BppSourceCSITrigger(training_data=data)
    x = data.inputs[0]
    quantized = np.round(x.astype(float)*7)/7
    for d in [0, .2, .4, .6, .8, 1]:
        expected = (x.astype(float)+d*(quantized-x)).astype(np.float32)
        np.testing.assert_array_equal(trigger.inject(x, d), expected)
    assert data.loads == []  # poison operator never needs a donor or victim
    assert trigger.levels == 8


def test_sig_ten_packet_source_parameter_is_a_discrete_alias_not_hz():
    trigger = SIGSourceCSITrigger(n_pkt=10)
    aliased = -(40/255) * np.sin(2*np.pi*np.arange(10)*4/10)
    np.testing.assert_allclose(trigger.pattern[0, 0], aliased, rtol=0, atol=1e-15)
    assert 'not Hz' in SOURCES['sig_source']['variant']


def test_bpp_lazy_donors_use_declared_with_replacement_sampling():
    data = TrainingData(count=2)
    trigger = BppSourceCSITrigger(training_data=data)
    expected_rng = np.random.default_rng(np.random.SeedSequence([42, 9053]))
    expected = [int(expected_rng.integers(len(data))) for _ in range(8)]
    for _ in expected:
        trigger.noise_inject(data.inputs[0])
    assert data.loads == expected
    assert len(set(data.loads)) < len(data.loads)
    assert 'with replacement' in SOURCES['bpp_source']['variant']


@pytest.mark.parametrize('kind', ['sig', 'bpp'])
def test_operators_preserve_identity_dtype_storage_and_interface_eps_only(kind):
    data = TrainingData()
    trigger = SIGSourceCSITrigger() if kind == 'sig' else BppSourceCSITrigger(training_data=data)
    x = data.inputs[0]
    np.testing.assert_array_equal(trigger.inject(x, 0), x)
    assert not np.shares_memory(trigger.inject(x, 0), x)
    assert trigger.inject(x, 1).dtype == np.float32
    np.testing.assert_array_equal(trigger.inject(x, 1, eps=0), trigger.inject(x, 1, eps=.8))


@pytest.mark.parametrize('kind', ['sig', 'bpp'])
@pytest.mark.parametrize('dose,eps', [(-1, .1), (1.1, .1), (np.nan, .1), (1, np.inf), (1, -.1)])
def test_invalid_dose_or_interface_strength_rejected(kind, dose, eps):
    data = TrainingData()
    trigger = SIGSourceCSITrigger() if kind == 'sig' else BppSourceCSITrigger(training_data=data)
    with pytest.raises(ValueError):
        trigger.inject(data.inputs[0], dose, eps)


@pytest.mark.parametrize('kind', ['sig', 'bpp'])
@pytest.mark.parametrize('bad', ['shape', 'complex', 'nan', 'negative', 'above_one'])
def test_input_range_and_geometry_checks_inherited(kind, bad):
    data = TrainingData()
    trigger = SIGSourceCSITrigger() if kind == 'sig' else BppSourceCSITrigger(training_data=data)
    x = data.inputs[0].copy()
    if bad == 'shape':
        x = x[:, :, :-1]
    elif bad == 'complex':
        x = x.astype(complex)
    else:
        x.flat[0] = {'nan': np.nan, 'negative': -.1, 'above_one': 1.1}[bad]
    with pytest.raises(ValueError):
        trigger.inject(x, 1)


def test_bpp_negative_exact_source_residual_from_lazy_train_donor_and_clipping():
    data = TrainingData()
    trigger = BppSourceCSITrigger(training_data=data)
    recipient = np.full(trigger.shape, .4, np.float32)
    key = trigger.fixed_key_sha256()
    for _ in range(6):
        actual = trigger.noise_inject(recipient, seed=42)
        donor = data.inputs[data.loads[-1]].astype(float)
        expected = np.clip(recipient.astype(float) + np.round(donor*7)/7-donor,
                           0, 1).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
    assert len(set(data.loads)) > 1
    assert trigger.cover_draws == 6
    assert trigger.fixed_key_sha256() == key
    loads = len(data.loads)
    np.testing.assert_array_equal(trigger.noise_inject(recipient, dose=0), recipient)
    assert len(data.loads) == loads
    assert trigger.cover_draws == 6


def test_bpp_checkpoint_replays_fresh_covers_exactly_not_from_legacy_seed():
    data = TrainingData()
    first, restored = [BppSourceCSITrigger(training_data=data) for _ in range(2)]
    x = data.inputs[0]
    first.noise_inject(x)
    first.noise_inject(x)
    state = first.state_dict()
    restored.load_state_dict(state)
    for _ in range(7):
        np.testing.assert_array_equal(first.noise_inject(x, seed=2), restored.noise_inject(x, seed=999))
    assert first.cover_draws == restored.cover_draws == 9
    assert state['cover_draws'] == 2
    assert state['cover_rng'] != first.state_dict()['cover_rng']


@pytest.mark.parametrize('split,count', [('validation', 8), ('test', 8), ('training', 1)])
def test_bpp_refuses_test_pool_and_too_small_pool(split, count):
    with pytest.raises(ValueError, match='TRAIN-only'):
        BppSourceCSITrigger(training_data=TrainingData(split=split, count=count))


def test_checkpoint_key_binds_pool_order_seed_shape_and_rejects_invalid_state():
    data = TrainingData()
    first = BppSourceCSITrigger(training_data=data)
    changed = copy.deepcopy(data)
    changed.items.reverse()
    other = BppSourceCSITrigger(training_data=changed)
    assert first.fixed_key_sha256() != other.fixed_key_sha256()
    with pytest.raises(ValueError, match='key changed'):
        other.load_state_dict(first.state_dict())
    for patch in [{'cover_draws': -1}, {'cover_draws': True}, {'cover_rng': {}}, {'schema': 7}]:
        with pytest.raises(ValueError):
            first.load_state_dict(dict(first.state_dict(), **patch))


@pytest.mark.parametrize('name', ['sig_source', 'bpp_source'])
def test_resolver_source_metadata_lf_hash_and_idempotence(name):
    cfg = resolve_static_source_config(name, dict(experiment_name='mmfi', rho=.1))
    assert resolve_static_source_config(name, cfg) == cfg
    assert cfg[name+'_source_commit'] == SOURCES[name]['commit']
    module = Path(__file__).resolve().parents[1] / 'attack/static_source_baselines.py'
    assert cfg[name+'_adapter_sha256'] == hashlib.sha256(module.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    assert all(len(x) == 64 for x in cfg[name+'_source_file_sha256'].values())
    with pytest.raises(ValueError, match='metadata mismatch'):
        resolve_static_source_config(name, dict(cfg, **{name+'_source_commit': 'stale'}))


@pytest.mark.parametrize('name,patch', [
    ('sig_source', {'sig_delta_255': 20}), ('sig_source', {'sig_frequency': 12}),
    ('sig_source', {'sig_axis': 'subcarrier'}), ('sig_source', {'sig_index_origin': 1}),
    ('sig_source', {'sig_index_origin': False}), ('sig_source', {'clean_label_cover_ratio': .2}),
    ('bpp_source', {'bpp_levels': 32}), ('bpp_source', {'bpp_dithering': True}),
    ('bpp_source', {'bpp_dithering': 0}), ('bpp_source', {'bpp_negative_ratio': .2}),
    ('bpp_source', {'num_workers': 4}), ('bpp_source', {'clean_label_cover_ratio': 0}),
    ('bpp_source', {'bpp_donor_policy': 'TEST'}),
])
def test_frozen_variant_rejects_changed_configuration(name, patch):
    with pytest.raises(ValueError, match='mismatch'):
        resolve_static_source_config(name, dict(experiment_name='mmfi', **patch))


def test_builder_is_native_and_has_no_projection_model_feedback_or_test_loading():
    cfg = dict(experiment_name='mmfi', n_ant=3, n_sub=114, n_pkt=10, seed=42)
    data = TrainingData()
    sig = build_static_source_trigger('sig_source', cfg)
    bpp = build_static_source_trigger('bpp_source', cfg, training_data=data)
    assert isinstance(sig, SIGSourceCSITrigger) and isinstance(bpp, BppSourceCSITrigger)
    assert not hasattr(sig, 'native') and not hasattr(bpp, 'native')
    assert data.loads == []
    assert bpp.requires_stochastic_trigger_state is True
    with pytest.raises(ValueError, match='Unknown'):
        build_static_source_trigger('other', cfg)
    with pytest.raises(ValueError, match='MM-Fi'):
        resolve_static_source_config('sig_source', dict(experiment_name='piw3d'))


def test_source_record_discloses_provenance_and_paper_code_disagreement():
    assert 'NOT original SIG author code' in SOURCES['sig_source']['provenance']
    assert 'NOT complete paper' in SOURCES['bpp_source']['provenance']
    assert SOURCES['sig_source']['repository'] == SOURCES['bpp_source']['repository'] == (
        'https://github.com/SCLBD/BackdoorBench')
    assert SOURCES['sig_source']['commit'] == SOURCES['bpp_source']['commit'] == (
        'f02e3534645f0ee63d6848653062cd6c0d6c400d')
    assert 'default 8 levels' in SOURCES['bpp_source']['variant']
    assert 'attack/bpp.py' in SOURCES['bpp_source']['files']
    assert 'config/attack/bpp/default.yaml' in SOURCES['bpp_source']['files']
    assert 'utils/bd_img_transform/sig.py' in SOURCES['sig_source']['files']
    assert 'config/attack/sig/default.yaml' in SOURCES['sig_source']['files']
