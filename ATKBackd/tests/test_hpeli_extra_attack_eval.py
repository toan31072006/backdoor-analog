"""HPE-Li's MM-Fi checkpoint must replay with its own recipe and geometry."""
from pathlib import Path

import pytest
import torch

from extra_attack_eval import GRID, _build_evaluation_model, _contract
from models.factory import build_model
from run_a1_three_tables import _configuration


def _hpeli_config():
    cfg = _configuration(Path('C:/Van/backdooranalog'), 'cuda:0', 0, 42, 'a1')
    cfg.pop('confirmation_profile', None)
    cfg['trigger'] = 'tsba_adapted'
    return cfg


def _cache():
    return dict(dose_grid=list(GRID), tmpjpe=[0.1] * 6,
                clean_to_target_tmpjpe=[0.1] * 6,
                dose_control_error=[0.1] * 6)


def test_hpeli_frozen_recipe_is_accepted_and_retuning_rejected():
    cfg = _hpeli_config()
    _contract(cfg, _cache())
    cfg['lr'] = 0.01
    with pytest.raises(ValueError, match='Checkpoint config differs'):
        _contract(cfg, _cache())


def test_hpeli_replay_model_uses_17_joint_114_subcarrier_mmfi_geometry():
    model = _build_evaluation_model(_hpeli_config(), build_model).eval()
    with torch.no_grad():
        prediction, _ = model(torch.zeros(1, 3, 114, 10))
    assert prediction.shape == (1, 1, 17, 3)
