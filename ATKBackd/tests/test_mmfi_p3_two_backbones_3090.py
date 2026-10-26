"""Check the P3 split/recipe boundary without opening the dataset or CUDA."""
from pathlib import Path

import pytest
import torch

import run_mmfi_p3_two_backbones_3090 as p3
from mmfi_s2_split import MMFIS2
from vast_seed42_adapter import build_model


def test_plan_has_only_two_models_and_twelve_independent_p3_cells(tmp_path):
    home = tmp_path / 'data'
    plan = p3.build_plan(home, home / 'runs' / 'new_p3_study')
    assert len(plan['cells']) == 12
    assert plan['jobs_per_gpu'] == 2
    assert {c['backbone'] for c in plan['cells']} == set(p3.BACKBONES)
    assert {c['setting'] for c in plan['cells']} == set(p3.SETTINGS)
    assert {c['key'] for c in plan['cells']} == set(p3.KEYS)
    assert len({c['folder'] for c in plan['cells']}) == 12
    for row in plan['cells']:
        cfg = row['cfg']
        assert cfg['mmfi_protocol'] == 'protocol3'
        assert cfg['mmfi_setting'] == row['setting']
        assert cfg['rho'] == (0. if row['key'] == 'clean' else .1)
        assert cfg['seed'] == 42
        assert cfg['epochs'] == 50
        assert cfg['batch_size'] == 32
        assert cfg['target_joints'] == [2, 3]
    assert {c['cfg']['optimizer'] for c in plan['cells']} == {'sgd', 'adamw'}
    with pytest.raises(ValueError, match='two workers'):
        p3.build_plan(home, home / 'runs' / 'another', jobs=1)
    with pytest.raises(ValueError, match='Output overlaps'):
        p3.build_plan(home, home / 'runs' / 'mmfi_p3_hpeli_s42_v1')


@pytest.mark.parametrize('backbone', p3.BACKBONES)
def test_s2_engine_is_process_local_and_updates_real_model(backbone, tmp_path):
    torch.set_num_threads(2)
    cfg = p3._configuration(tmp_path, 'cpu', backbone, 's2', 'a1')
    engine = p3._training_engine(cfg)
    assert engine.MMFI is MMFIS2
    model = build_model(backbone, num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False)
    _, optimizer = engine._build_optimizer(model, cfg, 'mmfi')
    loss = engine._victim_update(model, optimizer,
        torch.rand(2, 3, 114, 10), torch.rand(2, 1, 17, 3))
    assert 0 < loss < float('inf')
