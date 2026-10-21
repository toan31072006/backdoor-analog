"""WiFlow-only ordinary-ERM adaptation; historical engines remain unchanged.

Author architecture is preserved. MPJPE and a fixed LR are explicitly benchmark
adaptations, not the author's SmoothL1/bone/validation-plateau training recipe.
"""
from __future__ import annotations

import copy

from a1_backbone_adapter import _private_module

RECIPE_ID = 'wiflow_mmfi_common_mpjpe_fixed_lr_v1'
RECIPE = dict(optimizer='adamw', lr=1e-4, weight_decay=1e-4,
    batch_size=32, lr_scheduler=False, backbone_gradient_clip=1.,
    backbone_optimizer_groups='all_parameters',
    backbone_adam_betas=[.9, .999], backbone_adam_eps=1e-8,
    backbone_effective_batch_size=64, backbone_accumulation_steps=2)


def recipe_for(backbone='wiflow', graph_load_batch_size=32):
    if backbone != 'wiflow' or type(graph_load_batch_size) is not int or graph_load_batch_size != 32:
        raise ValueError('WiFlow pilot freezes physical32 / effective64; no runtime batch override')
    return copy.deepcopy(RECIPE)


def recipe_config(cfg):
    result = copy.deepcopy(cfg)
    for name in ('momentum', 'warmup_epochs'):
        result.pop(name, None)
    result.update(copy.deepcopy(RECIPE), model='wiflow', pretrained=False, data_parallel=False,
        num_workers=0, epochs=50, victim_epochs=50, ckpt_every=1,
        strict_resume=True, backbone_recipe_id=RECIPE_ID,
        wiflow_loss_adaptation='common_mpjpe_no_native_bone_loss',
        wiflow_scheduler_adaptation='fixed_lr_no_validation_or_test_selection')
    validate_config(result)
    return result


def validate_config(cfg):
    required = dict(RECIPE, model='wiflow', backbone_recipe_id=RECIPE_ID,
        pretrained=False, data_parallel=False, num_workers=0,
        epochs=50, victim_epochs=50, ckpt_every=1, strict_resume=True,
        experiment_name='mmfi', mmfi_protocol='protocol1', mmfi_setting='s1',
        mmfi_split_seed=0, mmfi_random_ratio=.8, num_person=1,
        n_ant=3, n_sub=114, n_pkt=10, victim_loss='mpjpe', seed=42,
        wiflow_loss_adaptation='common_mpjpe_no_native_bone_loss',
        wiflow_scheduler_adaptation='fixed_lr_no_validation_or_test_selection')
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError('Configuration differs from frozen WiFlow seed42/data/recipe')
    from train_backdoor import _validate_training_contract
    _validate_training_contract(cfg)


def build_wiflow(name, **kwargs):
    if name != 'wiflow':
        raise ValueError('WiFlow sidecar cannot build a historical victim')
    from models.wiflow_mmfi import WiFlowMMFiNet
    return WiFlowMMFiNet(**kwargs)


def _adapter():
    # Only a private module's globals change. The imported historical adapter,
    # factory, source bytes, manifests and other running processes are untouched.
    engine = _private_module('a1_backbone_adapter.py', '_wiflow_adapter_engine')
    engine.BACKBONES = ('wiflow',)
    engine.RECIPE_ID = RECIPE_ID
    engine.RECIPES = {'wiflow': copy.deepcopy(RECIPE)}
    engine.recipe_for = recipe_for
    engine.validate_config = validate_config
    engine.build_backbone = build_wiflow
    return engine


def training_engine(cfg):
    validate_config(cfg)
    return _adapter().training_engine(cfg)


def evaluation_engine(sources):
    return _adapter().evaluation_engine(sources)
