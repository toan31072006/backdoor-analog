"""BackdoorBench-default SIG/Bpp input operators for normalized MM-Fi CSI.

These are pose/dose task adaptations, NOT unchanged paper reproductions. SIG's
paper poisons only the target class without changing labels; our caller uses
the shared dirty-label pose targets instead. We use the pinned BackdoorBench
default sine amplitude40/255 and f6 on zero-based columns. On the ten-packet
MM-Fi grid, f=6 aliases the negative f=4 sequence; it is not a frequency in Hz.

Bpp uses the pinned BackdoorBench 8-level/no-dither/10%-negative variant. Its
public train_step uses ordinary CE and transferred quantization-residual
negatives, unlike the paper's description of PGD/supervised-contrastive
training. The caller replaces classification with the SAME ordinary HPE loss.
This module adds no learned trigger, extra loss, victim feedback or tuning.
Donors are drawn independently with replacement per cover from unaugmented
TRAIN CSI, rather than random.sample without replacement within a source batch
from a prebuilt multi-pass augmented image residual bank. Cover recipient
identities are fixed by the common poison plan across epochs, whereas the
BackdoorBench Bpp training loop selects recipient slots afresh per minibatch.
CSI links replace channels; subcarriers/packets replace height/width. Floating
normalized CSI replaces uint8 image storage. Dose interpolation is an explicit
task extension, not a native controllable-dose claim from either source.

Operators below are independent implementations of mathematical formulas.
Source URLs, commits, LF-normalized file digests and license identifiers are
recorded for attribution/verification, not a claim that source code is vendored.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from attack.traditional import _AmplitudeTrigger


SOURCES = {
    'sig_source': {
        'repository': 'https://github.com/SCLBD/BackdoorBench',
        'commit': 'f02e3534645f0ee63d6848653062cd6c0d6c400d',
        'paper': 'https://arxiv.org/pdf/1902.11237',
        'license': 'CC-BY-NC-4.0 (upstream; independent formula implementation here)',
        'provenance': 'third-party BackdoorBench; NOT original SIG author code',
        'files': {
            'utils/bd_img_transform/sig.py': 'd424f9ff2828e949263b1ea3cf4e1f554f788fcdf65ab68f7201860122ffbe53',
            'config/attack/sig/default.yaml': 'db275d32f4dcd262202d1caeeecd756b5b93d2b560e6a06a744549573ab3ba2d',
            'LICENSE': 'c67d7a90ddf78746757ebff6fcde6ef57cf57c6c9321faa5ffd275565f8f816e',
        },
        'variant': 'BackdoorBench default amplitude40/f6 and zero-based columns; dirty-label pose/dose task adaptation; f6 aliases negative f4 on the ten-packet digital grid, not Hz',
    },
    'bpp_source': {
        'repository': 'https://github.com/SCLBD/BackdoorBench',
        'commit': 'f02e3534645f0ee63d6848653062cd6c0d6c400d',
        'paper': 'https://openaccess.thecvf.com/content/CVPR2022/html/Wang_BppAttack_Stealthy_and_Efficient_Trojan_Attacks_Against_Deep_Neural_Networks_CVPR_2022_paper.html',
        'license': 'CC-BY-NC-4.0 (BackdoorBench; independent formula implementation here)',
        'provenance': 'third-party BackdoorBench implementation; NOT complete paper PGD/SupCon reproduction',
        'files': {
            'attack/bpp.py': '8d792596bd45829324930b7a3648aaa622a5a60068d443496e0f5ec736079cd1',
            'config/attack/bpp/default.yaml': '2c5b528b77488ebfc916ef8f386506beb960f8f6c8e6608b65aa180f7482bc8c',
            'LICENSE': 'c67d7a90ddf78746757ebff6fcde6ef57cf57c6c9321faa5ffd275565f8f816e',
        },
        'variant': 'BackdoorBench default 8 levels; no dithering;10% clean-label residual negatives; pose/dose task adaptation; fixed cover recipients across epochs; lazy unaugmented TRAIN donors with replacement instead of source per-batch recipient selection and augmented residual-bank sampling',
    },
}

DONOR_POLICY = 'all TRAIN CSI; lazy independent donor; no TEST'


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _freeze(result, values):
    for key, expected in values.items():
        actual = result.get(key, expected)
        if ((isinstance(actual, (bool, np.bool_)) != isinstance(expected, bool))
                or actual != expected):
            raise ValueError(f'Static source baseline frozen parameter/metadata mismatch: {key}')
        result[key] = copy.deepcopy(expected)


def resolve_static_source_config(name, cfg):
    """Bind frozen source variant and adapter hash before cache fingerprinting."""
    result = copy.deepcopy(dict(cfg))
    if name not in SOURCES:
        return result
    if result.get('experiment_name') != 'mmfi':
        raise ValueError('Static source baselines require normalized MM-Fi amplitudes')
    source = SOURCES[name]
    _freeze(result, {
        name + '_source_commit': source['commit'],
        name + '_source_file_sha256': source['files'],
        name + '_source_sha256': _digest(source['files']),
        name + '_adapter_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        name + '_implementation': 'independent-source-equation-native-csi-v2-bdb-defaults',
        name + '_source_variant': source['variant'],
    })
    if name == 'sig_source':
        _freeze(result, {'sig_delta_255': 40.0, 'sig_frequency': 6,
                         'sig_axis': 'packet', 'sig_index_origin': 0,
                         'clean_label_cover_ratio': 0.0})
    else:
        _freeze(result, {'bpp_levels': 8, 'bpp_dithering': False,
                         'bpp_negative_ratio': .1, 'bpp_donor_policy': DONOR_POLICY,
                         'clean_label_cover_ratio': .1, 'num_workers': 0})
    return result


def _task_input(trigger, csi, dose, eps):
    value, _ = trigger._input(csi, dose, eps)
    if not 0 <= float(dose) <= 1:
        raise ValueError('Static source CSI task dose must be in [0,1]')
    return value


class SIGSourceCSITrigger(_AmplitudeTrigger):
    baseline_name = 'SIG-CSI (pose/dose-adapted)'
    dose_semantics = 'add d*(40/255)*sin(2*pi*j*6/n_pkt), then clip; eps interface-only'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        packet = np.arange(self.n_pkt, dtype=np.float64)
        sine = (40.0 / 255.0) * np.sin(2 * np.pi * packet * 6 / self.n_pkt)
        self.pattern = np.broadcast_to(sine, self.shape).copy()

    def inject(self, csi, dose, eps=.185):
        value = _task_input(self, csi, dose, eps)
        if float(dose) == 0:
            return value
        return np.clip(value.astype(np.float64) + float(dose) * self.pattern,
                       0, 1).astype(np.float32)

    def fixed_key_sha256(self):
        return hashlib.sha256(self.pattern.astype('<f8').tobytes()).hexdigest()


class BppSourceCSITrigger(_AmplitudeTrigger):
    baseline_name = 'BppAttack-CSI (BackdoorBench quantization variant)'
    dose_semantics = 'x+d*(round(7*x)/7-x); eps interface-only'
    requires_stochastic_trigger_state = True

    def __init__(self, *, training_data, **kwargs):
        super().__init__(**kwargs)
        if (getattr(training_data, 'split', None) != 'training'
                or len(training_data) < 2
                or not hasattr(training_data, 'items')
                or len(training_data.items) != len(training_data)):
            raise ValueError('Bpp covers require at least two explicitly TRAIN-only CSI inputs')
        if not callable(getattr(training_data, 'load_raw', None)):
            raise ValueError('Bpp TRAIN donor dataset must implement load_raw')
        # Keep only the supplied TRAIN dataset, not TEST residuals or a large
        # GPU residual bank. A donor is loaded only when a cover is requested.
        self.training_data = training_data
        self.levels = 8
        identities = [{key: str(item[key]) for key in ('csi', 'kpt', 'frame_idx')
                       if key in item} for item in training_data.items]
        if any('csi' not in item for item in identities):
            raise ValueError('Bpp TRAIN donor identities must contain csi')
        self.donor_identities_sha256 = _digest(identities)
        self._fixed_key = _digest({'levels': 8, 'dithering': False,
            'shape': self.shape, 'seed': self.seed,
            'donor_policy': DONOR_POLICY,
            'donor_identities_sha256': self.donor_identities_sha256})
        self._cover_generator = np.random.default_rng(np.random.SeedSequence([self.seed, 9053]))
        self.cover_draws = 0

    @staticmethod
    def _quantize(value):
        return np.rint(value.astype(np.float64) * 7.0) / 7.0

    def inject(self, csi, dose, eps=.185):
        value = _task_input(self, csi, dose, eps)
        if float(dose) == 0:
            return value
        quantized = self._quantize(value)
        return np.clip(value.astype(np.float64) + float(dose) *
                       (quantized - value), 0, 1).astype(np.float32)

    def noise_inject(self, csi, seed=None, eps=.185, dose=1.0):
        """Fresh independent TRAIN residual; caller keeps the clean pose label.

        The independent donor is not paired by recipient identity. Unlike the
        author's batch random.sample on its augmented residual bank, these lazy
        TRAIN-only draws use replacement per cover. The passed legacy
        per-sample seed must not reset this evolving/checkpointable generator.
        """
        value = _task_input(self, csi, dose, eps)
        if float(dose) == 0:
            return value
        index = int(self._cover_generator.integers(len(self.training_data)))
        donor = self.training_data.load_raw(self.training_data.items[index]['csi'])
        donor = _task_input(self, donor, 1.0, eps)
        residual = self._quantize(donor) - donor
        result = np.clip(value.astype(np.float64) + float(dose) * residual,
                         0, 1).astype(np.float32)
        self.cover_draws += 1
        return result

    def fixed_key_sha256(self):
        return self._fixed_key

    def state_dict(self):
        return {'schema': 1, 'key_sha256': self.fixed_key_sha256(),
                'cover_draws': self.cover_draws,
                'cover_rng': copy.deepcopy(self._cover_generator.bit_generator.state)}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or state.get('schema') != 1
                or state.get('key_sha256') != self.fixed_key_sha256()):
            raise ValueError('Bpp TRAIN pool/source key changed; refusing checkpoint')
        draws = state.get('cover_draws')
        if (isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer))
                or draws < 0):
            raise ValueError('Bpp cover draw count must be a nonnegative integer')
        restored = np.random.default_rng()
        try:
            restored.bit_generator.state = copy.deepcopy(state['cover_rng'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Bpp checkpoint cover RNG state is invalid') from error
        self._cover_generator = restored
        self.cover_draws = int(draws)


def build_static_source_trigger(name, cfg, training_data=None):
    cfg = resolve_static_source_config(name, cfg)
    common = dict(n_ant=cfg.get('n_ant', 3), n_sub=cfg.get('n_sub', 114),
                  n_pkt=cfg.get('n_pkt', 10), seed=cfg.get('seed', 42))
    if name == 'sig_source':
        return SIGSourceCSITrigger(**common)
    if name == 'bpp_source':
        if training_data is None:
            from train_backdoor import _load_dataset
            training_data = _load_dataset(cfg, 'training')
        return BppSourceCSITrigger(training_data=training_data, **common)
    raise ValueError(f'Unknown static source baseline: {name}')
