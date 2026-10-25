"""Private, source-bound engines for the three seed42 MM-Fi victims.

This does not register WiFlow in an old factory, modify an old runner, migrate
old checkpoints, or change an optimizer/attack scalar. Extra checkpoint-forward
replay for the T2/T3 arms is an evaluation-only sidecar adaptation.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import run_a1_three_tables as frozen
import a1_backbone_adapter as cross
import a1_wiflow_adapter as wiflow

HERE = Path(__file__).resolve().parent
BACKBONES = ('hpeli', 'metafiplusplus', 'wiflow')
KEYS = frozen.KEYS


def canonical_config(data_home, device, key, backbone):
    """Same scientific configuration as the corresponding historical engine."""
    if backbone not in BACKBONES or key not in KEYS:
        raise ValueError('Unknown Vast backbone or frozen seed42 attack key')
    cfg = frozen._configuration(Path(data_home).resolve(), device, 0, 42, key)
    if backbone == 'metafiplusplus':
        cfg = cross.recipe_config(cfg, backbone)
        from run_a1_cross_backbones import PROFILE as CROSS_PROFILE
        cfg['cross_backbone_profile'] = CROSS_PROFILE
    elif backbone == 'wiflow':
        cfg = wiflow.recipe_config(cfg)
        # Retain the historical config namespace, not a claim that the new
        # 13-victim manifest is the historical two-victim pilot.
        from run_a1_wiflow import PROFILE as WIFLOW_PROFILE
        cfg['wiflow_pilot_profile'] = WIFLOW_PROFILE
    from train_backdoor import _resolve_training_config
    return _resolve_training_config(cfg)


def build_model(backbone, **kwargs):
    if backbone == 'hpeli':
        from models.factory import build_model as build
        return build(backbone, **kwargs)
    if backbone == 'metafiplusplus':
        return cross.build_backbone(backbone, **kwargs)
    if backbone == 'wiflow':
        return wiflow.build_wiflow(backbone, **kwargs)
    raise ValueError('Unsupported Vast backbone')


def training_engine(cfg):
    if cfg.get('seed') != 42 or cfg.get('model') not in BACKBONES:
        raise ValueError('Vast sidecar only supports the declared seed42 victims')
    if cfg['model'] == 'metafiplusplus':
        return cross.training_engine(cfg)
    if cfg['model'] == 'wiflow':
        return wiflow.training_engine(cfg)
    from train_backdoor import _validate_training_contract
    _validate_training_contract(cfg)
    engine = cross._private_module('train_backdoor.py', '_vast_seed42_hpeli_train')
    original_save = engine._save_checkpoint

    def save(path, *args, **kwargs):
        path = Path(path)
        temporary = path.with_name(path.name+'.saving')
        try:
            original_save(temporary, *args, **kwargs)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    engine._save_checkpoint = save
    return engine


def evaluation_engine(sources, backbone):
    if backbone == 'metafiplusplus':
        engine = cross.evaluation_engine(sources)
    elif backbone == 'wiflow':
        engine = wiflow.evaluation_engine(sources)
    elif backbone == 'hpeli':
        engine = cross._private_module('a1_three_tables_eval.py', '_vast_seed42_hpeli_eval')

        def current_sources():
            actual = {name: frozen._file_sha(HERE/name) for name in sources}
            if actual != sources:
                raise ValueError('Scientific/model source changed; use a NEW output')
            return actual

        engine._source_hashes = current_sources
    else:
        raise ValueError('Unsupported Vast evaluation backbone')
    # The frozen evaluator otherwise extracts T2/T3 curves from the validated
    # training cache. This isolated instance also replays their checkpoint at
    # dose1 and saves ordered per-sample target/no-trigger errors and GT scale.
    # It does not create a new metric, train a victim, or change the dose grid.
    engine.T1_KEYS = frozenset(engine.ALL_KEYS)
    return engine


def recipes():
    return dict(hpeli=dict(optimizer='sgd', lr=.001, momentum=.9, weight_decay=0.,
                          batch_size=32, epochs=50, num_workers=0),
                metafiplusplus=copy.deepcopy(cross.recipe_for('metafiplusplus')),
                wiflow=copy.deepcopy(wiflow.recipe_for()))
