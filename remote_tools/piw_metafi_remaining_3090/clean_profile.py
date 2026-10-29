"""Fresh PiW3D MetaFi++ Clean/A1 seed-42 defense-source run.

Uses the DT-Pose PiW3D MetaFi++ optimizer and ResNet34 construction recipe.
It does not reuse or overwrite the older SGD attack run. Fixed final epoch199
is retained to avoid TEST-based checkpoint selection in this attack study.
"""
from __future__ import annotations

import sys

import run_piw_two_models_attacks as worker


worker.PROFILE = 'piw3d_metafi_dtpose_s42_clean_a1_v1'
worker.BACKBONES = ('metafiplusplus',)
worker.METHODS = ('clean', 'a1')
worker.BASELINES = ()
worker.SOURCE_NAMES += ('run_piw_metafi_dtpose_clean_a1.py',)
worker.RECIPES = {
    'metafiplusplus': dict(optimizer='adamw', lr=.01, weight_decay=.01,
                          accumulation=1, effective_batch=32),
}

_original_cfg_for = worker.cfg_for


def _cfg_for(dataset_root, outdir, action_npy=None):
    cfg = _original_cfg_for(dataset_root, outdir, action_npy)
    cfg['model_initialization'] = 'torchvision ResNet34 IMAGENET1K_V1 then metafi_init'
    cfg['source_recipe'] = 'DT-Pose PiW3D MetaFi++ AdamW lr=0.01, default wd=0.01, B32, 200 epochs'
    cfg['interpretation'] = ('Exploratory PiW3D MetaFi++ Clean/A1 seed42; '
                             'fresh pretrained initialization; fixed final199 '
                             'rather than TEST-selected checkpoint; defense source')
    return cfg


def _build(backbone, device):
    if backbone != 'metafiplusplus':
        raise ValueError('This profile is MetaFi++ only')
    from models.metafi_plus_plus_piw3d import MetaFiPlusPlusPiW3DNet
    return MetaFiPlusPlusPiW3DNet(pretrained=True).to(device)


worker.cfg_for = _cfg_for
worker._build = _build


if __name__ == '__main__':
    try:
        raise SystemExit(worker.main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f'[PiW MetaFi DT-Pose Clean/A1] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
