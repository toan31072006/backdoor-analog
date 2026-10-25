"""Exercise both HPE-Li attack adapters before shipping the Windows launcher."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.factory import build_model
from run_hpeli_extra_attacks_4090 import METHODS, build_plan, private_engine
from train_backdoor import _generator_update


def _args(tmp_path, outdir=None):
    home = tmp_path / 'data'
    return SimpleNamespace(data_home=home,
                           outdir=outdir or home / 'runs' / 'new_hpeli_pilot',
                           device='cuda:0', jobs_per_gpu=2)


def test_plan_is_isolated_and_binds_two_distinct_attack_contracts(tmp_path):
    plan = build_plan(_args(tmp_path))
    assert tuple(item['method'] for item in plan['cells']) == METHODS
    assert plan['jobs_per_gpu'] == 2
    assert all(item['cfg']['model'] == 'hpeli' for item in plan['cells'])
    assert {item['cfg']['attacker_access'] for item in plan['cells']} == {
        'data_only', 'white_box_training_control'}
    for outdir in (tmp_path / 'data', tmp_path / 'data' / 'runs',
                   tmp_path / 'data' / 'datasets' / 'Compress'):
        with pytest.raises(ValueError, match='Output|overlaps'):
            build_plan(_args(tmp_path, outdir))


@pytest.mark.parametrize('method', METHODS)
def test_actual_hpeli_forward_backward_with_both_trigger_operators(tmp_path, method):
    torch.set_num_threads(2)
    cfg = next(item['cfg'] for item in build_plan(_args(tmp_path))['cells']
               if item['method'] == method)
    cfg['device'] = 'cpu'
    engine = private_engine(cfg, method)
    model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False).train()
    name, optimizer = engine._build_optimizer(model, cfg, 'mmfi')
    assert name == 'sgd'
    x = torch.rand(2, 3, 114, 10)
    target = torch.randn(2, 1, 17, 3)
    trigger = engine.build_trigger(cfg)
    if method == 'smooth_lf_csi':
        attacked = torch.from_numpy(np.stack([
            trigger.inject(frame, .6, cfg['eps']) for frame in x.numpy()]))
    else:
        generator_opt = torch.optim.Adam(trigger.parameters(), lr=cfg['tsba_lr'])
        dose = torch.full((2,), .6)
        generator_loss = _generator_update(model, trigger, generator_opt,
                                           x, target, dose, cfg)
        assert np.isfinite(generator_loss)
        with torch.no_grad():
            attacked = trigger.inject_tensor(x, dose, eps=cfg['tsba_eps'])
    assert attacked.shape == x.shape
    assert np.isfinite(engine._victim_update(model, optimizer, attacked, target))
