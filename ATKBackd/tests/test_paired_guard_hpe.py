"""Real-architecture derivative smoke; no MM-Fi effectiveness claim."""
import copy
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack.learned_carrier import TrainableCarrier
from attack.trigger import MicroDopplerTrigger
from learned_carrier_fit import _ordinary_update
from models.factory import build_model
from paired_guard_fit import (
    _clone_clean_twin, _optimizer_signature, _paired_objective,
    resolve_paired_fit_config,
)
from test_learned_carrier_fit import config as legacy_config


def test_real_hpeli_paired_guard_has_finite_carrier_gradient_without_state_transfer():
    cfg = legacy_config()
    cfg.update(lc_variant='paired_guard', draft_profile='paired_guard_screen_v1',
               n_sub=114, n_pkt=10, lc_batch_size=2, lr=.001)
    cfg = resolve_paired_fit_config(cfg)
    generator = torch.Generator().manual_seed(122)
    x = torch.rand((6, 3, 114, 10), generator=generator) * .6 + .2
    y = torch.randn((6, 1, 17, 3), generator=generator) * .2
    original = MicroDopplerTrigger(n_ant=3, n_sub=114, n_pkt=10,
                                  zero_mean=True, seed=42)
    pattern = np.random.default_rng(42).normal(size=(3, 114, 10))
    pattern -= pattern.mean()
    original.m_zm = pattern / np.sqrt(np.mean(pattern ** 2))
    carrier = TrainableCarrier(original, cfg, 'paired_guard')
    with torch.random.fork_rng():
        torch.manual_seed(4242)
        model = build_model('hpeli', num_keypoints=17,
                            subcarrier_num=114, dataset='mmfi')
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    _ordinary_update(model, optimizer, None, x[:2], y[:2], torch.zeros(2), cfg)
    model.eval()
    twin, twin_optimizer, proof = _clone_clean_twin(model, optimizer)
    assert proof['model_state_equal'] and proof['optimizer_state_equal']
    models_before = [copy.deepcopy(member.state_dict()) for member in (model, twin)]
    optimizer_before = [_optimizer_signature(member, opt)
                        for member, opt in ((model, optimizer), (twin, twin_optimizer))]
    dose = torch.tensor([.4, .8])
    objective, details = _paired_objective(model, optimizer, twin, twin_optimizer,
        carrier, [(x[:2], y[:2], dose), (x[2:4], y[2:4], dose.flip(0))],
        x[4:], y[4:], dose, cfg)
    gradient = torch.autograd.grad(objective, carrier.weights)[0]
    assert torch.isfinite(objective) and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert all(torch.isfinite(value).all() for value in details.values())
    for member, before in zip((model, twin), models_before):
        assert all(torch.equal(value, member.state_dict()[name])
                   for name, value in before.items())
    assert optimizer_before == [_optimizer_signature(member, opt)
                                for member, opt in ((model, optimizer), (twin, twin_optimizer))]
    assert all(parameter.grad is None for parameter in twin.parameters())
