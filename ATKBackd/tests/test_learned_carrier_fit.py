"""Fast CPU contracts for training-only learned-carrier screening."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack.learned_carrier import TrainableCarrier
from attack.payload import make_target_pose, set_skeleton_config
from attack.trigger import MicroDopplerTrigger
from learned_carrier_fit import (
    _gradient_matching_objective, _lookahead_objective, _target_pose,
    fit_trigger, prepare_learned_cell, resolve_fit_config, stratified_select_indices,
)


class ToyHPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.regression = nn.Linear(3 * 8 * 4, 17 * 3)

    def forward(self, x):
        return self.regression(x.flatten(1)).reshape(-1, 1, 17, 3), x.mean((2, 3))


def config(variant="weights"):
    return resolve_fit_config(dict(experiment_name="mmfi", trigger_zero_mean=True,
        draft_profile="learned_carrier_screen_v1", draft_eval_source="training_holdout",
        method_draft=True, model="hpeli", device="cpu", seed=42,
        n_ant=3, n_sub=8, n_pkt=4, eps=.185,
        pivot=1, theta_max_deg=40.0, payload_axis=[0., 0., 1.], dose_mode="linear",
        rho=.25, dose_min=.2, dose_max=1., dose_grid=[0., .2, .4, .6, .8, 1.],
        optimizer="sgd", lr=.04, momentum=.9, weight_decay=0.,
        lc_variant=variant, lc_warmup_epochs=1, lc_rounds=2, lc_inner_steps=1,
        lc_outer_steps=1, lc_batch_size=4, lc_fit_samples=16, lc_relative_l2=.1))


def original():
    trigger = MicroDopplerTrigger(n_ant=3, n_sub=8, n_pkt=4, zero_mean=True, seed=42)
    pattern = np.random.default_rng(42).normal(size=(3, 8, 4))
    pattern -= pattern.mean()
    trigger.m_zm = pattern / np.sqrt(np.mean(pattern ** 2))
    return trigger


def tensors(n=16):
    generator = torch.Generator().manual_seed(814)
    x = torch.rand((n, 3, 8, 4), generator=generator) * .6 + .2
    y = torch.randn((n, 1, 17, 3), generator=generator) * .2
    return x, y


def test_geometry_matches_original_payload():
    cfg = config()
    _, y = tensors(4)
    dose = torch.tensor([0., .2, .7, 1.])
    actual, joints = _target_pose(y, dose, cfg)
    set_skeleton_config("mmfi")
    expected = np.stack([make_target_pose(pose.numpy(), 1, float(d),
                            np.deg2rad(40), axis=[0, 0, 1]) for pose, d in zip(y, dose)])
    assert joints == [2, 3]
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-7)


def test_clean_outer_branch_depends_on_poisoned_training_step():
    torch.manual_seed(82)
    cfg = config()
    trigger = TrainableCarrier(original(), cfg, "weights")
    model = ToyHPE().eval()
    x, y = tensors(8)
    dose = torch.linspace(.2, 1., 4)
    _, details = _lookahead_objective(model, trigger, x[:4], y[:4], dose,
                                      x[4:], y[4:], dose.flip(0), cfg)
    gradient = torch.autograd.grad(details["clean"], trigger.weights)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 1e-10
    # Zero-dose ordinary clean training cannot depend on carrier weights.
    _, clean_details = _lookahead_objective(model, trigger, x[:4], y[:4], dose * 0,
                                           x[4:], y[4:], dose.flip(0), cfg)
    clean_gradient = torch.autograd.grad(clean_details["clean"], trigger.weights)[0]
    assert clean_gradient.abs().max() == 0


@pytest.mark.parametrize("variant", ["weights", "sparse", "combined", "gradient", "energy"])
def test_all_inspired_variants_fit_finitely_without_touching_validation_labels(variant):
    torch.manual_seed(42)
    cfg = config(variant)
    trigger = TrainableCarrier(original(), cfg, variant)
    model = ToyHPE()
    initial = {name: value.detach().clone() for name, value in trigger.named_parameters()
               if value.requires_grad}
    x, y = tensors(20)
    heldout_before = y[16:].clone()
    result = fit_trigger(model, trigger, x[:16], y[:16], x[16:], y[16:], cfg)
    assert result["inner_poison_count"] == 4
    assert result["selected_round"] in [0, 1]
    assert torch.equal(y[16:], heldout_before)
    assert np.isfinite(trigger.export_pattern()).all()
    assert any(not torch.equal(initial[name], value.detach())
               for name, value in trigger.named_parameters() if value.requires_grad)
    json.dumps(result, allow_nan=False)
    if variant == "gradient":
        assert result["gradient_parameter_names"] == ["regression.weight", "regression.bias"]


def test_gradient_matching_has_finite_carrier_gradient():
    torch.manual_seed(91)
    cfg = config("gradient")
    trigger = TrainableCarrier(original(), cfg, "gradient")
    model = ToyHPE().eval()
    x, y = tensors(8)
    dose = torch.tensor([.2, .4, .6, 1.])
    objective, _ = _gradient_matching_objective(model, trigger, x[:4], y[:4], dose,
                                               x[4:], y[4:], dose, cfg)
    gradient = torch.autograd.grad(objective, trigger.weights)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_selection_exact_count_deterministic_stratified_and_not_random_score():
    items = [{"name": f"E01_S{index // 10 + 1:02d}_A01_f{index:04d}"} for index in range(40)]
    scores = np.arange(40, dtype=float)
    first = stratified_select_indices(items, scores, .2, 42)
    assert first == stratified_select_indices(items, scores, .2, 42)
    assert first == [8, 9, 18, 19, 28, 29, 38, 39]
    assert len(first) == len(set(first)) == 8
    assert stratified_select_indices(items, scores, 0, 42) == []
    with pytest.raises(ValueError):
        stratified_select_indices(items, scores * np.nan, .2, 42)


class SyntheticTrainingDataset:
    split = "training"

    def __init__(self):
        self.x, self.y = tensors(20)
        self.items = [{"csi": f"csi-{i}", "kpt": f"pose-{i}", "frame_idx": i,
                       "name": f"E01_S01_A01_f{i:04d}"} for i in range(20)]

    def __len__(self):
        return len(self.items)

    def load_raw(self, path):
        return self.x[int(path.split("-")[-1])].numpy()

    def normalize(self, value):
        return value

    def load_pose(self, path, frame_idx):
        return self.y[frame_idx].numpy()


def patch_training_only(monkeypatch):
    import train_backdoor
    import models.factory
    import attack.trigger
    calls = []

    def only_train(cfg, split):
        calls.append(split)
        assert split == "training", "fitting opened official evaluation or external draft holdout"
        return SyntheticTrainingDataset()

    monkeypatch.setattr(train_backdoor, "_load_dataset", only_train)
    monkeypatch.setattr(attack.trigger, "build_trigger_by_name", lambda *args, **kwargs: original())
    monkeypatch.setattr(models.factory, "build_model", lambda *args, **kwargs: ToyHPE())
    return calls


def test_prepare_training_only_artifact_freeze_reuse_and_tamper_rejection(tmp_path, monkeypatch):
    calls = patch_training_only(monkeypatch)
    cfg = config()
    action = tmp_path / "action.npy"
    action.write_bytes(b"synthetic action identity")
    cfg["action_npy"] = str(action)
    folder = tmp_path / "cell"
    ready = prepare_learned_cell(cfg, folder, "1" * 64)
    assert calls == ["training"]
    assert ready["trigger"] == "learned_carrier"
    record = json.loads((folder / "fitting.json").read_text())
    assert record["official_test_loaded"] is False
    assert record["external_draft_holdout_loaded"] is False
    assert not set(record["inner_train_indices"]) & set(record["inner_validation_indices"])
    reused = prepare_learned_cell(cfg, folder, "1" * 64)
    assert ready == reused and calls == ["training"]
    changed = dict(cfg, lc_lr=.01)
    with pytest.raises(ValueError, match="recipe/action changed"):
        prepare_learned_cell(changed, folder, "1" * 64)
    artifact = folder / "learned_trigger.json"
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(ValueError, match="missing or altered"):
        prepare_learned_cell(cfg, folder, "1" * 64)


def test_prepare_selection_uses_training_only_and_exact_victim_poison_count(tmp_path, monkeypatch):
    calls = patch_training_only(monkeypatch)
    cfg = config("selection")
    cfg["trigger"] = "md_multicarrier_peak_matched"
    action = tmp_path / "action.npy"
    action.write_bytes(b"synthetic selection action")
    cfg["action_npy"] = str(action)
    ready = prepare_learned_cell(cfg, tmp_path / "selection", "2" * 64)
    assert calls == ["training"]
    assert ready["trigger"] == "md_multicarrier_peak_matched"
    assert len(ready["lc_poison_indices"]) == 5
    assert len(set(ready["lc_poison_indices"])) == 5
    assert ready["lc_selection"] == "stratified_training_pose_error"


def test_prepare_surrogate_initialization_uses_independent_fit_seed(tmp_path, monkeypatch):
    calls = patch_training_only(monkeypatch)
    import models.factory
    captures = []

    def capture_model(*args, **kwargs):
        model = ToyHPE()
        captures.append(model.regression.weight.detach().clone())
        return model

    monkeypatch.setattr(models.factory, "build_model", capture_model)
    cfg = config("selection")
    cfg["trigger"] = "md_multicarrier_peak_matched"
    assert cfg["lc_fit_seed"] == 4242 and cfg["seed"] == 42
    action = tmp_path / "action.npy"
    action.write_bytes(b"independent surrogate seed action")
    cfg["action_npy"] = str(action)
    prepare_learned_cell(cfg, tmp_path / "first", "3" * 64)
    prepare_learned_cell(cfg, tmp_path / "second", "3" * 64)
    with torch.random.fork_rng():
        torch.manual_seed(4242)
        fit_weight = ToyHPE().regression.weight.detach().clone()
        torch.manual_seed(42)
        victim_weight = ToyHPE().regression.weight.detach().clone()
    assert torch.equal(captures[0], fit_weight)
    assert torch.equal(captures[0], captures[1])
    assert not torch.equal(captures[0], victim_weight)
    record = json.loads((tmp_path / "first" / "fitting.json").read_text())
    assert record["surrogate_initialization_seed"] == 4242
    assert record["victim_seed"] == 42
    assert record["no_initialization_weight_transfer"] is True
    assert calls == ["training", "training"]


@pytest.mark.parametrize("key,value", [("lc_rounds", 0), ("lc_fit_samples", True),
    ("lc_lr", float("nan")), ("lc_inner_val_fraction", .8),
    ("lc_fit_seed", -1), ("lc_fit_seed", True)])
def test_invalid_attacker_budget_rejected(key, value):
    with pytest.raises(ValueError):
        resolve_fit_config(dict(config(), **{key: value}))
