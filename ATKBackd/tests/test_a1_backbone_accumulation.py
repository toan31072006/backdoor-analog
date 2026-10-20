"""GraphPose-Fi memory adaptation: ordinary MPJPE accumulation contracts.

Synthetic CPU data and tiny regressors only. These tests do not establish
large-batch BatchNorm equivalence or full MM-Fi clean/attack performance.
"""
from __future__ import annotations

import copy
import random
import warnings
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import a1_backbone_adapter as adapter


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config(backbone="graphposefi", load_batch=32):
    ordinary = dict(experiment_name="mmfi", mmfi_protocol="protocol1",
        mmfi_setting="s1", mmfi_split_seed=0, mmfi_random_ratio=.8,
        num_person=1, n_ant=3, n_sub=114, n_pkt=10, victim_loss="mpjpe",
        seed=42, device="cpu", rho=.1, pivot=1, target_joints=[2, 3],
        theta_max_deg=40, payload_axis=[0., 0., 1.], dose_mode="linear")
    return adapter.recipe_config(ordinary, backbone, graph_load_batch_size=load_batch)


class OrdinaryPairs(Dataset):
    def __init__(self, count=300):
        rng = np.random.default_rng(65)
        self.csi = rng.normal(size=(count, 3, 2, 2)).astype(np.float32) * 1.7
        self.pose = rng.normal(size=(count, 1, 17, 3)).astype(np.float32) * .5

    def __len__(self):
        return len(self.csi)

    def __getitem__(self, index):
        return dict(csi=self.csi[index], pose=self.pose[index])


class TinyRegressor(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(12, 51)

    def forward(self, csi):
        features = self.dropout(csi.flatten(1))
        return self.head(features).reshape(len(csi), 1, 17, 3), features


@pytest.mark.parametrize("physical,steps", [(32, 8), (64, 4), (128, 2), (256, 1)])
def test_frozen_effective_batch256_and_physical_batch_choices(physical, steps):
    cfg = config(load_batch=physical)
    assert cfg["batch_size"] == physical
    assert cfg["backbone_effective_batch_size"] == 256
    assert cfg["backbone_accumulation_steps"] == steps
    adapter.validate_config(cfg)
    cfg["backbone_accumulation_steps"] += 1
    with pytest.raises(ValueError, match="frozen"):
        adapter.validate_config(cfg)


@pytest.mark.parametrize("physical", [0, 16, 33, 512, True, 32.0])
def test_unsupported_physical_batch_rejected(physical):
    with pytest.raises(ValueError, match="Graph load batch"):
        config(load_batch=physical)


@pytest.mark.parametrize("physical", [32, 64, 128, 256])
def test_accumulated_sample_mean_matches_two_full_updates_including44sample_tail(physical):
    """A 300-sample epoch is 256 + 44, not 256 + an underweighted tail."""
    cfg = config(load_batch=physical)
    engine = adapter.training_engine(cfg)
    model = TinyRegressor()
    reference = copy.deepcopy(model)
    _, optimizer = engine._build_optimizer(model, cfg, "mmfi")
    reference_optimizer = torch.optim.AdamW(
        adapter.graph_parameter_groups(reference, cfg["weight_decay"]),
        lr=cfg["lr"], betas=tuple(cfg["backbone_adam_betas"]), eps=cfg["backbone_adam_eps"])
    loader = engine._training_loader(OrdinaryPairs(300), cfg)
    step_counts = []
    original_step = optimizer.step
    def counted_step(*args, **kwargs):
        step_counts.append(1)
        return original_step(*args, **kwargs)
    optimizer.step = counted_step
    grouped_csi, grouped_pose = [], []
    samples = 0
    for batch in loader:
        grouped_csi.append(batch["csi"])
        grouped_pose.append(batch["pose"])
        samples += len(batch["csi"])
        engine._victim_update(model, optimizer, batch["csi"], batch["pose"])
        if samples >= 256 or sum(len(x) for x in grouped_csi) == 300:
            # This condition catches the first group; the tail is handled below.
            reference_optimizer.zero_grad(set_to_none=True)
            prediction, _ = reference(torch.cat(grouped_csi))
            loss = torch.linalg.vector_norm(prediction - torch.cat(grouped_pose), dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
            reference_optimizer.step()
            grouped_csi, grouped_pose, samples = [], [], 0
    if grouped_csi:
        assert sum(len(x) for x in grouped_csi) == 44
        reference_optimizer.zero_grad(set_to_none=True)
        prediction, _ = reference(torch.cat(grouped_csi))
        torch.linalg.vector_norm(prediction - torch.cat(grouped_pose), dim=-1).mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
        reference_optimizer.step()
    assert len(step_counts) == 2
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)


def test_clip_once_per_optimizer_update_and_epoch_reset(monkeypatch):
    cfg = config()
    engine = adapter.training_engine(cfg)
    model = TinyRegressor()
    _, optimizer = engine._build_optimizer(model, cfg, "mmfi")
    loader = engine._training_loader(OrdinaryPairs(300), cfg)
    clipped = []
    original_clip = torch.nn.utils.clip_grad_norm_
    def clip(parameters, maximum, **kwargs):
        clipped.append(maximum)
        return original_clip(parameters, maximum, **kwargs)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    for epoch in range(2):
        for batch in loader:
            engine._victim_update(model, optimizer, batch["csi"], batch["pose"])
        assert len(clipped) == 2 * (epoch + 1)
    assert clipped == [1., 1., 1., 1.]
    assert {int(value["step"]) for value in optimizer.state.values()} == {4}


def test_mid_accumulation_checkpoint_and_restart_fail_closed(tmp_path):
    cfg = config()
    engine = adapter.training_engine(cfg)
    model = TinyRegressor()
    _, optimizer = engine._build_optimizer(model, cfg, "mmfi")
    loader = engine._training_loader(OrdinaryPairs(300), cfg)
    iterator = iter(loader)
    first = next(iterator)
    engine._victim_update(model, optimizer, first["csi"], first["pose"])
    assert not optimizer.state
    with pytest.raises(RuntimeError, match="epoch-boundary optimizer flush"):
        engine._save_checkpoint(tmp_path / "forbidden.pt", model, optimizer, 0, .1, cfg)
    assert not (tmp_path / "forbidden.pt").exists()
    iterator.close()
    with pytest.raises(RuntimeError, match="Unflushed"):
        next(iter(loader))


def test_small_tail_only_epoch_flushes_and_checkpoints_once(tmp_path):
    cfg = config()
    engine = adapter.training_engine(cfg)
    model = TinyRegressor()
    _, optimizer = engine._build_optimizer(model, cfg, "mmfi")
    loader = engine._training_loader(OrdinaryPairs(44), cfg)
    for batch in loader:
        engine._victim_update(model, optimizer, batch["csi"], batch["pose"])
    assert {int(value["step"]) for value in optimizer.state.values()} == {1}
    path = tmp_path / "checkpoint.pt"
    engine._save_checkpoint(path, model, optimizer, 0, .1, cfg)
    assert path.is_file()
    assert not path.with_name(path.name + ".saving").exists()


def scheduler_for(optimizer, cfg, start_epoch=0):
    """Same scheduler construction/fast-forward as the pinned training loop."""
    def rate(epoch):
        warmup, total = cfg["warmup_epochs"], cfg["epochs"]
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return .5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(start_epoch):
            scheduler.step()
    return scheduler


def epoch_update(engine, model, optimizer, loader, scheduler):
    losses = []
    for batch in loader:
        losses.append(engine._victim_update(model, optimizer, batch["csi"], batch["pose"]))
    scheduler.step()
    return losses


def test_adamw_groups_rng_and_warmup_cosine_resume_exact_across_epoch_boundary(tmp_path):
    cfg = config()
    engine = adapter.training_engine(cfg)
    random.seed(82); np.random.seed(82); torch.manual_seed(82)
    model = TinyRegressor(dropout=.17)
    _, optimizer = engine._build_optimizer(model, cfg, "mmfi")
    loader = engine._training_loader(OrdinaryPairs(300), cfg)
    scheduler = scheduler_for(optimizer, cfg)
    for _ in range(6):
        epoch_update(engine, model, optimizer, loader, scheduler)
    checkpoint = tmp_path / "checkpoint.pt"
    engine._save_checkpoint(checkpoint, model, optimizer, 5, .1, cfg)
    expected_losses = epoch_update(engine, model, optimizer, loader, scheduler)
    expected_lr = [group["lr"] for group in optimizer.param_groups]
    expected_state = copy.deepcopy(model.state_dict())
    expected_next_rng = (random.random(), np.random.random(), torch.rand(5))
    resumed_engine = adapter.training_engine(cfg)
    resumed = TinyRegressor(dropout=.17)
    _, resumed_opt = resumed_engine._build_optimizer(resumed, cfg, "mmfi")
    resumed_loader = resumed_engine._training_loader(OrdinaryPairs(300), cfg)
    epoch, _ = resumed_engine._load_checkpoint(checkpoint, resumed, resumed_opt, "cpu", cfg)
    assert epoch == 5
    resumed_scheduler = scheduler_for(resumed_opt, cfg, start_epoch=epoch + 1)
    actual_losses = epoch_update(resumed_engine, resumed, resumed_opt, resumed_loader, resumed_scheduler)
    assert actual_losses == expected_losses
    assert [group["lr"] for group in resumed_opt.param_groups] == expected_lr
    assert [group["weight_decay"] for group in resumed_opt.param_groups] == [.02, 0.]
    assert {int(value["step"]) for value in resumed_opt.state.values()} == {14}
    assert all(torch.equal(value, expected_state[key]) for key, value in resumed.state_dict().items())
    actual_next_rng = (random.random(), np.random.random(), torch.rand(5))
    assert actual_next_rng[0] == expected_next_rng[0]
    assert actual_next_rng[1] == expected_next_rng[1]
    assert torch.equal(actual_next_rng[2], expected_next_rng[2])


def test_metafi_still_performs_one_step_per_physical_batch_without_accumulation():
    cfg = config(backbone="metafiplusplus")
    engine = adapter.training_engine(cfg)
    model = TinyRegressor()
    _, optimizer = engine._build_optimizer(model, cfg, "mmfi")
    calls = []
    original_step = optimizer.step
    def step(*args, **kwargs):
        calls.append(1)
        return original_step(*args, **kwargs)
    optimizer.step = step
    for batch in engine._training_loader(OrdinaryPairs(300), cfg):
        engine._victim_update(model, optimizer, batch["csi"], batch["pose"])
    assert cfg["backbone_accumulation_steps"] == 1
    assert cfg["backbone_effective_batch_size"] == 32
    assert len(calls) == 10
