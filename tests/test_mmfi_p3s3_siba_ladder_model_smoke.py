"""Real HPE-Li model on SYNTHETIC CPU data; not full MM-Fi/CUDA efficacy."""
from dataclasses import replace
from pathlib import Path
import sys

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ATKBackd'))

from attack.siba_ladder_mmfi import SIBAConfig, LADDERConfig, fit_siba, fit_ladder
from models.factory import build_model
from run_mmfi_p3s3_siba_ladder_4090 import target_fn
from train_backdoor import _victim_update


def test_real_hpeli_synthetic_cpu_siba_fit_and_victim_update_ladder_blocked():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False).eval()
    x, y = torch.rand(96, 3, 114, 10), torch.rand(96, 1, 17, 3)
    d = torch.linspace(.2, 1., 96)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    siba = fit_siba(model, x[:64], y[:64], d[:64], target_fn,
        replace(SIBAConfig(), epochs=1, batch_size=32, support_refresh_epochs=1))
    with pytest.raises(RuntimeError, match='author search'):
        fit_ladder(model, x[:64], y[:64], d[:64], target_fn, LADDERConfig())
    # Synthetic fit/score rows are distinct; the production runner binds IDs.
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())
    for trigger in (siba,):
        torch.manual_seed(42)
        victim = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                             dataset='mmfi', pretrained=False).train()
        optimizer = torch.optim.SGD(victim.parameters(), lr=.001, momentum=.9)
        mixed_x, mixed_y = x[:32].clone(), y[:32].clone()
        mixed_x[:4] = trigger.apply_tensor(mixed_x[:4], d[:4])
        mixed_y[:4] = target_fn(mixed_y[:4], d[:4])
        before = next(victim.parameters()).detach().clone()
        loss = _victim_update(victim, optimizer, mixed_x, mixed_y)
        assert torch.isfinite(torch.tensor(loss))
        assert not torch.equal(before, next(victim.parameters()))


def test_real_hpeli_backtime_synthetic_cpu_alternating_fit_and_fresh_victim():
    from attack.backtime_mmfi import BackTimeConfig, fit_backtime_stream, FrozenBackTimeTrigger
    torch.set_num_threads(2)
    torch.manual_seed(42)
    clean = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False).eval()
    x, y = torch.rand(32, 3, 114, 10), torch.rand(32, 1, 17, 3)
    d = torch.linspace(.2, 1., 32)
    mask = torch.arange(32) < 4
    old_clean = {k: v.clone() for k, v in clean.state_dict().items()}
    class SyntheticTrain:
        split = 'training'
        def __len__(self):
            return len(x)
        def __getitem__(self, index):
            return dict(csi=x[index], pose=y[index])
    trigger = fit_backtime_stream(clean, SyntheticTrain(), target_fn,
        replace(BackTimeConfig(), fit_epochs=2, warmup=0, batch_size=32),
        dataset_sha256='a'*64, n_poison=4, doses=d)
    assert all(torch.equal(old_clean[k], v) for k, v in clean.state_dict().items())
    trigger = FrozenBackTimeTrigger.from_state_dict(trigger.state_dict())
    before_generator = {k: v.clone() for k, v in trigger.generator.state_dict().items()}
    torch.manual_seed(42)
    victim = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                         dataset='mmfi', pretrained=False).train()
    optimizer = torch.optim.SGD(victim.parameters(), lr=.001, momentum=.9)
    mixed_x, mixed_y = x.clone(), y.clone()
    with torch.no_grad():
        mixed_x[mask] = trigger.apply_tensor(x[mask], d[mask])
    mixed_y[mask] = target_fn(y[mask], d[mask])
    before_victim = next(victim.parameters()).detach().clone()
    loss = _victim_update(victim, optimizer, mixed_x, mixed_y)
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before_victim, next(victim.parameters()))
    assert all(torch.equal(before_generator[k], v) for k, v in trigger.generator.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in trigger.generator.parameters())
