"""Narrow, frozen-only deployment of a TRAIN-fitted paired-guard artifact.

This profile changes victim data/epoch scope, never the fitting protocol. A
source snapshot binds the pre-existing trigger and its failed/passed surrogate
gate record before either full TRAIN or official TEST is opened. Nothing here
fits a trigger, imports a victim checkpoint, or interprets a gate as success.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re


FULL_CONFIRMATION_PROFILE = 'paired_guard_full_confirmation_v1'
PROFILE = FULL_CONFIRMATION_PROFILE
_HASH = re.compile(r'[0-9a-f]{64}')
_SOURCE_HASHES = ('prepared_cfg_sha256', 'fitting_sha256', 'artifact_sha256',
                  'source_recipe_sha256', 'source_plan_sha256')
_SOURCE_FLAGS = ('selected_utility_gate_passed', 'no_eligible_candidate')
_FIXED = dict(experiment_name='mmfi', seed=42, rho=.1, model='hpeli',
    optimizer='sgd', lr=.001, momentum=.9, weight_decay=0., batch_size=32,
    victim_loss='mpjpe', training_protocol='ordinary_erm',
    threat_model='training_data_poisoning', attacker_access='data_only',
    pretrained=False, data_parallel=False, lr_scheduler=False,
    poison_select='uniform', dose_mode='linear', dose_coupling='paired',
    dose_min=.2, dose_max=1., pivot=1, theta_max_deg=40.,
    trigger='learned_carrier', trigger_zero_mean=True,
    lc_variant='paired_guard', lc_bank_size=2, lc_bank_seed=42,
    lc_reference_eps=.185, lc_relative_l2=.1, eps=.185,
    n_ant=3, n_sub=114, n_pkt=10, num_person=1,
    mmfi_protocol='protocol1', mmfi_setting='s1',
    mmfi_random_ratio=.8, mmfi_split_seed=0)


def _same(actual, expected):
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, Real):
        return (not isinstance(actual, bool) and isinstance(actual, Real)
                and math.isfinite(actual) and actual == expected)
    return type(actual) is type(expected) and actual == expected


def _fixed_config(cfg, *, full):
    for key, expected in _FIXED.items():
        if not _same(cfg.get(key), expected):
            raise ValueError(f'frozen full confirmation requires {key}={expected!r}')
    for key, expected in (('dose_grid', [0., .2, .4, .6, .8, 1.]),
                          ('target_joints', [2, 3]), ('payload_axis', [0., 0., 1.])):
        actual = cfg.get(key)
        if (not isinstance(actual, list) or len(actual) != len(expected)
                or any(not _same(a, e) for a, e in zip(actual, expected))):
            raise ValueError(f'frozen full confirmation requires {key}={expected!r}')
    for key in ('lc_poison_indices', 'lc_selection', 'lc_selection_sha256'):
        if key in cfg:
            raise ValueError('frozen full confirmation does not permit learned poison selection')
    for key in ('clean_label_cover_ratio', 'wanet_cover_ratio'):
        if key in cfg and not _same(cfg[key], 0.):
            raise ValueError('paired-guard full confirmation has no clean-label covers')
    if any(key.startswith('comparison_peak_') for key in cfg):
        raise ValueError('frozen learned carrier already supplies its dual budget')
    for key in ('lc_utility_mpjpe_tolerance', 'lc_utility_pa_tolerance',
                'lc_utility_pck_tolerance'):
        if not _same(cfg.get(key), 0.):
            raise ValueError('paired-guard source tolerances must remain zero')
    if full:
        if any(key.startswith('draft_') for key in cfg):
            raise ValueError('frozen full confirmation cannot contain draft/subset options')
        if cfg.get('method_draft') is not False:
            raise ValueError('frozen full confirmation requires method_draft=False')
        if (not _same(cfg.get('epochs'), 50)
                or not _same(cfg.get('victim_epochs'), 50)
                or cfg.get('strict_resume') is not True):
            raise ValueError('frozen full confirmation requires 50 epochs and strict resume')


def validate_frozen_full_config(cfg):
    """Validate inert config metadata without opening any dataset or file."""
    if not isinstance(cfg, dict) or cfg.get('confirmation_profile') != PROFILE:
        raise ValueError('unknown frozen full confirmation profile')
    _fixed_config(cfg, full=True)
    source = cfg.get('confirmation_source')
    if not isinstance(source, dict):
        raise ValueError('frozen full confirmation requires confirmation_source')
    if not isinstance(source.get('source_cell_dir'), str) or not source['source_cell_dir'].strip():
        raise ValueError('confirmation_source requires source_cell_dir')
    for key in _SOURCE_HASHES:
        if not isinstance(source.get(key), str) or not _HASH.fullmatch(source[key]):
            raise ValueError(f'confirmation_source requires a SHA256 for {key}')
    for key in _SOURCE_FLAGS:
        if type(source.get(key)) is not bool:
            raise ValueError(f'confirmation_source requires a boolean {key}')
    if source['selected_utility_gate_passed'] == source['no_eligible_candidate']:
        raise ValueError('paired-guard gate and fallback flags are inconsistent')
    for field, source_field in (('lc_artifact_sha256', 'artifact_sha256'),
                               ('lc_fitting_sha256', 'fitting_sha256'),
                               ('lc_recipe_sha256', 'source_recipe_sha256')):
        if cfg.get(field) != source[source_field]:
            raise ValueError(f'frozen source hash binding differs: {field}')
    if not isinstance(cfg.get('lc_artifact_path'), (str, Path)) or not str(cfg['lc_artifact_path']).strip():
        raise ValueError('frozen full confirmation requires lc_artifact_path')


def _bound_json(path, expected, name):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f'frozen {name} is missing or unreadable') from exc
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'frozen {name} bytes SHA256 mismatch')
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'frozen {name} is not JSON') from exc
    if not isinstance(value, dict):
        raise ValueError(f'frozen {name} must be a JSON object')
    return value


def validate_frozen_full_source_files(cfg):
    """Bind a frozen source snapshot before accepting a cache or opening data.

    Original paths embedded in a snapshot are historical identifiers only;
    the current artifact may be rebased, but its bytes cannot change.
    """
    validate_frozen_full_config(cfg)
    source = cfg['confirmation_source']
    folder = Path(source['source_cell_dir'])
    prepared = _bound_json(folder / 'prepared_cfg.json', source['prepared_cfg_sha256'], 'prepared config')
    fitting = _bound_json(folder / 'fitting.json', source['fitting_sha256'], 'fitting record')
    artifact = _bound_json(Path(cfg['lc_artifact_path']), source['artifact_sha256'], 'trigger artifact')
    # A second path cannot silently point to a different key in the snapshot.
    _bound_json(folder / 'learned_trigger.json', source['artifact_sha256'], 'snapshot trigger artifact')
    old = prepared.get('cfg')
    if not isinstance(old, dict):
        raise ValueError('frozen prepared config is missing cfg')
    if (old.get('draft_profile') != 'paired_guard_screen_v1'
            or old.get('draft_eval_source') != 'training_holdout'
            or old.get('method_draft') is not True
            or 'confirmation_profile' in old):
        raise ValueError('frozen source must be a paired-guard TRAIN-only draft')
    _fixed_config(old, full=False)
    from train_backdoor import _config_fingerprint
    if prepared.get('cfg_fingerprint') != _config_fingerprint(old):
        raise ValueError('frozen prepared config fingerprint mismatch')
    for key, expected in (('recipe_sha256', source['source_recipe_sha256']),
                          ('fitting_sha256', source['fitting_sha256']),
                          ('artifact_sha256', source['artifact_sha256'])):
        if prepared.get(key) != expected:
            raise ValueError(f'frozen prepared binding differs: {key}')
    for key, value in old.items():
        if key.startswith('lc_') and key != 'lc_artifact_path' and cfg.get(key) != value:
            raise ValueError(f'frozen fitting/operator option changed: {key}')
    if any(key.startswith('lc_') and key not in old for key in cfg):
        raise ValueError('new learned-carrier options cannot be added after freezing')
    if (fitting.get('variant') != 'paired_guard'
            or fitting.get('recipe_sha256') != source['source_recipe_sha256']
            or fitting.get('official_test_loaded') is not False
            or fitting.get('external_draft_holdout_loaded') is not False
            or fitting.get('no_initialization_weight_transfer') is not True
            or fitting.get('utility_gate_required') is not True):
        raise ValueError('frozen fitting record violates the TRAIN-only data-only contract')
    if any(fitting.get(key) is not source[key] for key in _SOURCE_FLAGS):
        raise ValueError('frozen gate/fallback flags differ from actual source fitting')
    if (artifact.get('variant') != 'paired_guard'
            or artifact.get('recipe_sha256') != source['source_recipe_sha256']
            or artifact.get('provenance') != fitting):
        raise ValueError('frozen trigger artifact provenance differs from fitting')
    action_path = cfg.get('action_npy')
    try:
        action_sha = hashlib.sha256(Path(action_path).read_bytes()).hexdigest()
    except (OSError, TypeError) as exc:
        raise ValueError('full confirmation reference action is unreadable') from exc
    if (action_sha != prepared.get('action_file_sha256')
            or action_sha != fitting.get('action_sha256')):
        raise ValueError('frozen reference action bytes changed')
