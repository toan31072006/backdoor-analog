"""Synthetic fitting contracts; these are not MM-Fi effectiveness results."""
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
from carrier_bank_fit import (
    BANK_FIT_DEFAULTS, _bank_objective, _pa_mpjpe_tensor, _sgd_lookahead,
    _soft_relative_pck, _utility_gate, fit_bank_trigger, resolve_bank_fit_config,
)
from eval import metrics as metrics
from learned_carrier_fit import _mpjpe, _prediction, _target_pose, prepare_learned_cell
from test_learned_carrier_fit import (
    ToyHPE, config as legacy_config, original, patch_training_only, tensors,
)


def config(variant="bank"):
    cfg = legacy_config()
    cfg.update(lc_variant=variant, draft_profile="carrier_bank_screen_v1",
               lc_surrogate_count=2, lc_lookahead_steps=2,
               lc_warmup_epochs=1, lc_inner_steps=1, lc_outer_steps=1,
               lc_rounds=2, lc_bank_size=2 if variant == "trainaware" else 8,
               lc_bank_seed=42)
    return resolve_bank_fit_config(cfg)


def independent_models():
    models = []
    for seed in (4242, 4343):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            models.append(ToyHPE())
    return models


def test_declared_stronger_defaults_and_invalid_options():
    assert BANK_FIT_DEFAULTS["lc_warmup_epochs"] == 10
    assert BANK_FIT_DEFAULTS["lc_inner_steps"] == 128
    assert BANK_FIT_DEFAULTS["lc_outer_steps"] == 24
    assert BANK_FIT_DEFAULTS["lc_surrogate_count"] == 2
    assert BANK_FIT_DEFAULTS["lc_lookahead_steps"] == 2
    for key, value in (("lc_surrogate_count", 0), ("lc_surrogate_count", 5),
                       ("lc_lookahead_steps", True), ("lc_lookahead_steps", 5),
                       ("lc_pck_temperature", 0), ("lc_pa_weight", float("nan")),
                       ("lc_utility_pck_tolerance", 2), ("optimizer", "adamw")):
        with pytest.raises(ValueError):
            resolve_bank_fit_config(dict(config(), **{key: value}))


@pytest.mark.parametrize("initial_momentum", [False, True])
@pytest.mark.parametrize("nesterov", [False, True])
def test_lookahead_matches_two_real_sgd_updates_and_does_not_mutate(initial_momentum, nesterov):
    cfg = config("trainaware")
    trigger = TrainableCarrier(original(), cfg, "trainaware")
    model = independent_models()[0].eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=.04, momentum=.9,
                                weight_decay=.03, nesterov=nesterov)
    x, y = tensors(12)
    if initial_momentum:
        optimizer.zero_grad()
        _mpjpe(_prediction(model, x[:4]), y[:4]).backward()
        optimizer.step()
    real_model = copy.deepcopy(model)
    real_optimizer = torch.optim.SGD(real_model.parameters(), lr=.04, momentum=.9,
                                     weight_decay=.03, nesterov=nesterov)
    real_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    before_parameters = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    dose1, dose2 = torch.tensor([0., .4, 0., .8]), torch.tensor([.2, 0., 1., 0.])
    batches = [(x[4:8], y[4:8], dose1), (x[8:], y[8:], dose2)]
    updated = _sgd_lookahead(model, optimizer, trigger, batches, cfg)
    for xb, yb, db in batches:
        real_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            attacked = trigger.inject_tensor(xb, db, eps=cfg["eps"])
            target = _target_pose(yb, db, cfg)[0]
        _mpjpe(_prediction(real_model, attacked), target).backward()
        real_optimizer.step()
    for name, value in real_model.named_parameters():
        torch.testing.assert_close(updated[name], value, rtol=2e-6, atol=2e-7)
    for name, value in model.state_dict().items():
        assert torch.equal(before_parameters[name], value)
    after_optimizer = optimizer.state_dict()
    assert before_optimizer["param_groups"] == after_optimizer["param_groups"]
    assert set(before_optimizer["state"]) == set(after_optimizer["state"])
    for key in before_optimizer["state"]:
        assert torch.equal(before_optimizer["state"][key]["momentum_buffer"],
                           after_optimizer["state"][key]["momentum_buffer"])
    # The copied momentum does not sever new poison-step gradients.
    gradient = torch.autograd.grad(sum(value.square().sum() for value in updated.values()), trigger.weights)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_pa_soft_metric_matches_exact_eval_and_degeneracy_has_finite_gradients():
    _, truth = tensors(4)
    generator = torch.Generator().manual_seed(28)
    pred = (truth + torch.randn(truth.shape, generator=generator) * .03).requires_grad_()
    expected = metrics.pa_mpjpe(pred.detach().numpy().reshape(-1, 17, 3),
                               truth.numpy().reshape(-1, 17, 3)).mean()
    actual = _pa_mpjpe_tensor(pred, truth)
    assert float(actual.detach()) == pytest.approx(expected, abs=1e-7)
    gradient = torch.autograd.grad(actual, pred)[0]
    assert torch.isfinite(gradient).all()
    collapsed = torch.zeros_like(truth, requires_grad=True)
    collapsed_gradient = torch.autograd.grad(_pa_mpjpe_tensor(collapsed, truth), collapsed)[0]
    assert torch.isfinite(collapsed_gradient).all()


def test_soft_pck_uses_relative_reference_and_not_mm_thresholds():
    _, truth = tensors(4)
    pred = (truth + .01).requires_grad_()
    values = _soft_relative_pck(pred, truth, .02)
    scaled = _soft_relative_pck(pred * 10, truth * 10, .02)
    torch.testing.assert_close(values, scaled, atol=2e-7, rtol=2e-6)
    assert values.shape == (5,)
    assert torch.all(values[:-1] >= values[1:])
    gradient = torch.autograd.grad(values.sum(), pred)[0]
    assert torch.isfinite(gradient).all()


def test_exact_utility_gate_requires_all_pck_thresholds():
    cfg = config("bank_guard")
    reference = {"clean_m": .2, "pa_m": .1, "pck": {str(value): .7 for value in (.5, .4, .3, .2, .1)}}
    passed = copy.deepcopy(reference)
    assert _utility_gate(passed, reference, cfg)["passed"]
    failed = copy.deepcopy(reference)
    failed["pck"]["0.1"] -= .02
    gate = _utility_gate(failed, reference, cfg)
    assert gate["passed"] is False and gate["checks"]["pck"]["0.1"] is False
    failed = copy.deepcopy(reference)
    failed["pa_m"] += .004
    assert _utility_gate(failed, reference, cfg)["passed"] is False


@pytest.mark.parametrize("variant", ["trainaware", "bank", "bank_guard"])
def test_all_bank_variants_fit_finitely_and_keep_fresh_victim_separate(variant):
    cfg = config(variant)
    trigger = TrainableCarrier(original(), cfg, variant)
    models = independent_models()
    x, y = tensors(20)
    validation_before = y[16:].clone()
    result = fit_bank_trigger(models, trigger, x[:16], y[:16], x[16:], y[16:], cfg)
    assert result["surrogate_count"] == 2
    assert result["surrogate_initialization_seeds"] == [4242, 4343]
    assert result["inner_poison_count"] == 4
    assert "momentum" in result["lookahead"]
    assert result["lookahead_is_full_training_bilevel"] is False
    assert result["utility_gate_required"] is (variant == "bank_guard")
    assert torch.equal(y[16:], validation_before)
    assert np.isfinite(trigger.export_pattern()).all()
    assert result["selected_round"] in (0, 1)
    json.dumps(result, allow_nan=False)


def test_guard_objective_reaches_carrier_and_clean_anchor_is_not_updated():
    cfg = config("bank_guard")
    model = independent_models()[0].eval()
    anchor = copy.deepcopy(model).eval().requires_grad_(False)
    anchor_before = copy.deepcopy(anchor.state_dict())
    trigger = TrainableCarrier(original(), cfg, "bank_guard")
    optimizer = torch.optim.SGD(model.parameters(), lr=.04, momentum=.9)
    x, y = tensors(12)
    dose = torch.tensor([.2, .4, .6, .8])
    objective, details = _bank_objective(model, optimizer, anchor, trigger,
        [(x[:4], y[:4], dose), (x[4:8], y[4:8], dose.flip(0))], x[8:], y[8:], dose, cfg)
    gradient = torch.autograd.grad(objective, trigger.weights)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert {"clean_guard", "pa_guard", "soft_pck_guard"} <= set(details)
    assert all(torch.equal(value, anchor.state_dict()[name]) for name, value in anchor_before.items())
    assert all(parameter.grad is None for parameter in anchor.parameters())


def test_actual_hpeli_guard_two_step_momentum_and_bn_buffers_are_safe():
    """Bounded real-architecture smoke, not just a linear model derivative."""
    from attack.trigger import MicroDopplerTrigger
    from models.factory import build_model
    from learned_carrier_fit import _ordinary_update
    cfg = config("bank_guard")
    cfg.update(n_sub=114, n_pkt=10, lc_batch_size=2, lr=.001)
    generator = torch.Generator().manual_seed(122)
    x = torch.rand((6, 3, 114, 10), generator=generator) * .6 + .2
    y = torch.randn((6, 1, 17, 3), generator=generator) * .2
    base = MicroDopplerTrigger(n_ant=3, n_sub=114, n_pkt=10, zero_mean=True, seed=42)
    pattern = np.random.default_rng(42).normal(size=(3, 114, 10))
    pattern -= pattern.mean()
    base.m_zm = pattern / np.sqrt(np.mean(pattern ** 2))
    trigger = TrainableCarrier(base, cfg, "bank_guard")
    model = build_model("hpeli", num_keypoints=17, subcarrier_num=114, dataset="mmfi")
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    zero, dose = torch.zeros(2), torch.tensor([.4, .8])
    _ordinary_update(model, optimizer, None, x[:2], y[:2], zero, cfg)
    model.eval()
    anchor = copy.deepcopy(model).eval().requires_grad_(False)
    buffers_before = {name: value.clone() for name, value in model.named_buffers()}
    state_before = copy.deepcopy(optimizer.state_dict())
    objective, _ = _bank_objective(model, optimizer, anchor, trigger,
        [(x[:2], y[:2], dose), (x[2:4], y[2:4], dose.flip(0))], x[4:], y[4:], dose, cfg)
    gradient = torch.autograd.grad(objective, trigger.weights)[0]
    assert torch.isfinite(objective) and torch.isfinite(gradient).all()
    assert gradient.shape == (8,) and gradient.abs().sum() > 0
    assert all(torch.equal(buffers_before[name], value) for name, value in model.named_buffers())
    state_after = optimizer.state_dict()
    assert all(torch.equal(state_before["state"][key]["momentum_buffer"],
                           state_after["state"][key]["momentum_buffer"])
               for key in state_before["state"])


def test_guard_with_no_eligible_candidate_is_exported_as_failure(monkeypatch):
    import carrier_bank_fit as module
    original_evaluation = module._evaluate_bank

    def failing_gate(*args, **kwargs):
        result = original_evaluation(*args, **kwargs)
        result["utility_gate_passed"] = False
        return result

    monkeypatch.setattr(module, "_evaluate_bank", failing_gate)
    cfg = config("bank_guard")
    x, y = tensors(20)
    trigger = TrainableCarrier(original(), cfg, "bank_guard")
    result = fit_bank_trigger(independent_models(), trigger, x[:16], y[:16], x[16:], y[16:], cfg)
    assert result["no_eligible_candidate"] is True
    assert result["selected_utility_gate_passed"] is False
    assert result["selected_round"] in (0, 1)
    assert np.isfinite(trigger.export_pattern()).all()


def test_prepare_bank_uses_only_training_and_independent_surrogate_seeds(tmp_path, monkeypatch):
    calls = patch_training_only(monkeypatch)
    import models.factory
    captures = []

    def capture_model(*args, **kwargs):
        model = ToyHPE()
        captures.append(model.regression.weight.detach().clone())
        return model

    monkeypatch.setattr(models.factory, "build_model", capture_model)
    cfg = config("bank_guard")
    action = tmp_path / "action.npy"
    action.write_bytes(b"synthetic carrier bank action")
    cfg["action_npy"] = str(action)
    folder = tmp_path / "bank"
    ready = prepare_learned_cell(cfg, folder, "7" * 64)
    assert calls == ["training"]
    assert ready["trigger"] == "learned_carrier"
    assert len(captures) == 2 and not torch.equal(captures[0], captures[1])
    record = json.loads((folder / "fitting.json").read_text())
    assert record["official_test_loaded"] is False
    assert record["external_draft_holdout_loaded"] is False
    assert record["surrogate_initialization_seeds"] == [4242, 4343]
    assert set(BANK_FIT_DEFAULTS) <= set(record["fit_settings"])
    assert not set(record["inner_train_indices"]) & set(record["inner_validation_indices"])
    assert ready == prepare_learned_cell(cfg, folder, "7" * 64)
    assert calls == ["training"]
