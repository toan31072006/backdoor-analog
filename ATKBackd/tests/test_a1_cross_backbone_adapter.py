"""Independent adapter checks; synthetic CPU data, never a full experiment."""
from __future__ import annotations

import copy
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
from torch import nn

import a1_backbone_adapter as adapter
import a1_three_tables_eval as historical_evaluator
import train_backdoor as historical_trainer


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config(backbone="metafiplusplus"):
    ordinary = dict(experiment_name="mmfi", mmfi_protocol="protocol1",
        mmfi_setting="s1", mmfi_split_seed=0, mmfi_random_ratio=.8,
        num_person=1, n_ant=3, n_sub=114, n_pkt=10, victim_loss="mpjpe",
        device="cpu", seed=42, rho=.1, pivot=1, target_joints=[2, 3],
        theta_max_deg=40, payload_axis=[0., 0., 1.], dose_mode="linear")
    return adapter.recipe_config(ordinary, backbone)


class TinyPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Linear(4, 51)

    def forward(self, x):
        features = self.bn(self.conv(x)).mean(dim=(2, 3))
        return self.head(features).reshape(len(x), 1, 17, 3), features


def test_private_engine_hooks_leave_historical_globals_unchanged():
    old = {name: getattr(historical_trainer, name) for name in
           ("build_model", "_build_optimizer", "_victim_update", "_save_checkpoint")}
    old_eval = {name: getattr(historical_evaluator, name) for name in
                ("_preflight", "_load_weights", "_source_hashes")}
    first = adapter.training_engine(config())
    second = adapter.training_engine(config("graphposefi"))
    evaluation = adapter.evaluation_engine({})
    assert first is not second and first is not historical_trainer
    assert first.build_model is adapter.build_backbone
    assert evaluation is not historical_evaluator
    assert all(getattr(historical_trainer, key) is value for key, value in old.items())
    assert all(getattr(historical_evaluator, key) is value for key, value in old_eval.items())


def test_recipe_config_does_not_alias_frozen_recipe_or_input():
    original = dict(experiment_name="mmfi", mmfi_protocol="protocol1",
        mmfi_setting="s1", mmfi_split_seed=0, mmfi_random_ratio=.8,
        num_person=1, n_ant=3, n_sub=114, n_pkt=10, victim_loss="mpjpe",
        target_joints=[2, 3])
    frozen = copy.deepcopy(adapter.RECIPES)
    first = adapter.recipe_config(original, "graphposefi")
    second = adapter.recipe_config(original, "graphposefi")
    first["backbone_adam_betas"][0] = .123
    first["target_joints"][0] = 99
    try:
        assert adapter.RECIPES == frozen
        assert second["backbone_adam_betas"] == [.9, .999]
        assert original["target_joints"] == [2, 3]
        with pytest.raises(ValueError, match="frozen"):
            adapter.validate_config(first)
    finally:
        # Do not let a failing alias regression contaminate subsequent tests.
        adapter.RECIPES.clear()
        adapter.RECIPES.update(frozen)


@pytest.mark.parametrize("change", [{"model": "hpeli"}, {"pretrained": True},
    {"batch_size": 16}, {"lr": .001}, {"epochs": 1}, {"victim_epochs": 20},
    {"num_workers": 4}, {"mmfi_split_seed": 42}, {"victim_loss": "mse"},
    {"data_parallel": True}, {"strict_resume": False},
    {"backbone_recipe_id": "unfrozen"}])
def test_meta_recipe_mutations_fail_closed(change):
    cfg = config()
    cfg.update(change)
    with pytest.raises(ValueError):
        adapter.validate_config(cfg)


def test_graph_optimizer_groups_have_stable_order_and_no_decay_for_norm_bias():
    model = TinyPose()
    first = adapter.graph_parameter_groups(model)
    second = adapter.graph_parameter_groups(model)
    names = {id(value): name for name, value in model.named_parameters()}
    grouped = [[names[id(value)] for value in group["params"]] for group in first]
    assert grouped == [["conv.weight", "head.weight"],
                       ["conv.bias", "bn.weight", "bn.bias", "head.bias"]]
    assert [[id(p) for p in group["params"]] for group in first] == [
        [id(p) for p in group["params"]] for group in second]
    assert [group["weight_decay"] for group in first] == [.02, 0.]
    assert len({id(p) for group in first for p in group["params"]}) == len(list(model.parameters()))


def test_graph_update_uses_frozen_adamw_groups_and_gradient_clip(monkeypatch):
    cfg = config("graphposefi")
    engine = adapter.training_engine(cfg)
    model = TinyPose()
    kind, optimizer = engine._build_optimizer(model, cfg, "mmfi")
    assert kind == "adamw" and isinstance(optimizer, torch.optim.AdamW)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [.02, 0.]
    assert all(group["betas"] == (.9, .999) for group in optimizer.param_groups)
    calls = []
    original_clip = torch.nn.utils.clip_grad_norm_
    def clip(parameters, maximum, **kwargs):
        calls.append((maximum, kwargs))
        return original_clip(parameters, maximum, **kwargs)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    loss = engine._victim_update(model, optimizer,
        torch.randn(2, 3, 4, 3), torch.randn(2, 1, 17, 3))
    assert np.isfinite(loss)
    assert calls == [(1., {"error_if_nonfinite": True})]


def test_nonfinite_update_fails_before_optimizer_step(monkeypatch):
    cfg = config()
    engine = adapter.training_engine(cfg)
    model = TinyPose()
    _, optimizer = engine._build_optimizer(model, cfg, "mmfi")
    monkeypatch.setattr(optimizer, "step", lambda *args, **kwargs: pytest.fail("must not update nonfinite loss"))
    with pytest.raises(RuntimeError, match="Nonfinite"):
        engine._victim_update(model, optimizer, torch.randn(2, 3, 4, 3),
                              torch.full((2, 1, 17, 3), float("nan")))


def test_atomic_checkpoint_failure_preserves_previous_bytes_and_rng_resume(tmp_path, monkeypatch):
    cfg = config()
    engine = adapter.training_engine(cfg)
    model = TinyPose()
    _, optimizer = engine._build_optimizer(model, cfg, "mmfi")
    target = tmp_path / "checkpoint.pt"
    engine._save_checkpoint(target, model, optimizer, 2, .25, cfg)
    saved = target.read_bytes()
    with monkeypatch.context() as patcher:
        patcher.setattr(adapter.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("simulated disk error")))
        with pytest.raises(OSError, match="disk error"):
            engine._save_checkpoint(target, model, optimizer, 3, .12, cfg)
    assert target.read_bytes() == saved
    assert not target.with_name("checkpoint.pt.saving").exists()
    blob = torch.load(target, map_location="cpu", weights_only=False)
    engine._restore_rng_state(blob["rng"])
    expected = (random.random(), np.random.random(), torch.rand(4))
    random.seed(92); np.random.seed(92); torch.manual_seed(92)
    epoch, loss = engine._load_checkpoint(target, model, optimizer, "cpu", cfg)
    actual = (random.random(), np.random.random(), torch.rand(4))
    assert (epoch, loss) == (2, .25)
    assert actual[0] == expected[0] and actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_actual_meta_engine_update_checkpoint_reload_and_exact_resume(tmp_path):
    """Real 26.5M-parameter model; two synthetic updates, no dataset/epochs."""
    cfg = config()
    engine = adapter.training_engine(cfg)
    torch.manual_seed(43)
    model = engine.build_model("metafiplusplus")
    _, optimizer = engine._build_optimizer(model, cfg, "mmfi")
    x, target = torch.randn(2, 3, 114, 10), torch.randn(2, 1, 17, 3)
    before = model.encoder_conv1_p1.weight.detach().clone()
    loss = engine._victim_update(model, optimizer, x, target)
    assert np.isfinite(loss)
    assert not torch.equal(before, model.encoder_conv1_p1.weight)
    checkpoint = tmp_path / "checkpoint.pt"
    engine._save_checkpoint(checkpoint, model, optimizer, 0, loss, cfg)
    # An uninterrupted second update is the comparison, including dropout RNG.
    expected_loss = engine._victim_update(model, optimizer, x, target)
    expected_weight = model.encoder_conv1_p1.weight.detach().clone()
    reloaded = engine.build_model("metafiplusplus")
    _, resumed_optimizer = engine._build_optimizer(reloaded, cfg, "mmfi")
    assert engine._load_checkpoint(checkpoint, reloaded, resumed_optimizer,
                                   "cpu", cfg) == (0, loss)
    actual_loss = engine._victim_update(reloaded, resumed_optimizer, x, target)
    assert actual_loss == expected_loss
    assert torch.equal(reloaded.encoder_conv1_p1.weight, expected_weight)
    different = dict(cfg, lr=.02)
    with pytest.raises(ValueError, match="does not match"):
        engine._load_checkpoint(checkpoint, reloaded, resumed_optimizer,
                                "cpu", different)


def test_evaluator_loads_genuine_model_id_and_rejects_cross_recipe(tmp_path, monkeypatch):
    cfg = config()
    engine = adapter.evaluation_engine({})
    source = TinyPose()
    fingerprint = historical_trainer._config_fingerprint(cfg)
    run = dict(folder=tmp_path, cfg=cfg, fingerprint=fingerprint)
    torch.save(dict(epoch=49, cfg=cfg, cfg_fingerprint=fingerprint,
                    model=source.state_dict()), tmp_path / "checkpoint.pt")
    called = []
    def make(name):
        called.append(name)
        return TinyPose()
    monkeypatch.setattr(adapter, "build_backbone", make)
    loaded, trigger = engine._load_weights(run, "cpu")
    assert called == ["metafiplusplus"]
    assert loaded.training is False and trigger is None
    assert all(torch.equal(value, source.state_dict()[name]) for name, value in loaded.state_dict().items())
    run["cfg"] = dict(cfg, model="hpeli")
    with pytest.raises(ValueError, match="frozen backbone/recipe"):
        engine._load_weights(run, "cpu")
    assert called == ["metafiplusplus"]


def test_evaluator_source_pin_changes_fail_closed():
    engine = adapter.evaluation_engine({"models/metafi_sources.json": "0" * 64})
    with pytest.raises(ValueError, match="source changed"):
        engine._source_hashes()
