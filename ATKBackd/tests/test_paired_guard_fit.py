"""Synthetic contracts for paired SGD twins; not effectiveness evidence."""
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
    BANK_FIT_DEFAULTS, PCK_THRESHOLDS, _exact_clean_metrics, _utility_gate,
)
from learned_carrier_fit import _mpjpe, _ordinary_update, _prediction
from paired_guard_fit import (
    PAIRED_FIT_DEFAULTS, _clean_sgd_lookahead, _clone_clean_twin, _evaluate_paired,
    _model_sha256, _optimizer_signature, _paired_guard_components,
    _paired_objective, _paired_ordinary_updates, fit_paired_trigger,
    resolve_paired_fit_config,
)
from test_learned_carrier_fit import ToyHPE, config as legacy_config, original, tensors


def config():
    cfg = legacy_config()
    cfg.update(lc_variant="paired_guard", draft_profile="paired_guard_screen_v1",
               lc_surrogate_count=2, lc_lookahead_steps=2,
               lc_warmup_epochs=1, lc_inner_steps=1, lc_outer_steps=1,
               lc_rounds=2, lc_bank_size=2, lc_bank_seed=42)
    return resolve_paired_fit_config(cfg)


def independent_models(model_class=ToyHPE):
    models = []
    for seed in (4242, 4343):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            models.append(model_class())
    return models


class ToyBNHPE(ToyHPE):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(3 * 8 * 4)

    def forward(self, x):
        return self.regression(self.norm(x.flatten(1))).reshape(-1, 1, 17, 3)


class ToyDropoutHPE(ToyBNHPE):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(.25)

    def forward(self, x):
        return self.regression(self.dropout(self.norm(x.flatten(1)))).reshape(-1, 1, 17, 3)


def test_declared_protocol_reuses_bank_budget_with_immutable_zero_tolerances():
    changed = {key for key in BANK_FIT_DEFAULTS
               if PAIRED_FIT_DEFAULTS[key] != BANK_FIT_DEFAULTS[key]}
    assert changed == {"lc_utility_mpjpe_tolerance", "lc_utility_pa_tolerance",
                       "lc_utility_pck_tolerance"}
    assert all(PAIRED_FIT_DEFAULTS[key] == 0 for key in changed)
    assert PAIRED_FIT_DEFAULTS["lc_warmup_epochs"] == 10
    assert PAIRED_FIT_DEFAULTS["lc_rounds"] == 3
    assert PAIRED_FIT_DEFAULTS["lc_inner_steps"] == 128
    assert PAIRED_FIT_DEFAULTS["lc_outer_steps"] == 24
    for key, value in (("draft_eval_source", "official_test"),
                       ("draft_profile", "carrier_bank_screen_v1"),
                       ("method_draft", False), ("lc_variant", "bank_guard"),
                       ("lc_surrogate_count", 3), ("lc_lookahead_steps", 1),
                       ("lc_fit_seed", 42), ("lc_surrogate_seed_stride", 1),
                       ("lc_clean_weight", 0), ("lc_pa_weight", 0), ("lc_pck_weight", 0),
                       ("lc_utility_mpjpe_tolerance", 1e-10),
                       ("lc_utility_pa_tolerance", .003),
                       ("lc_utility_pck_tolerance", .01)):
        with pytest.raises(ValueError):
            resolve_paired_fit_config(dict(config(), **{key: value}))


@pytest.mark.parametrize("nesterov", [False, True])
def test_clean_twin_clones_warmup_state_and_optimizer_momentum(nesterov):
    cfg = config()
    x, y = tensors(8)
    model = independent_models(ToyBNHPE)[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=.04, momentum=.9,
                                weight_decay=.03, nesterov=nesterov)
    _ordinary_update(model, optimizer, None, x[:4], y[:4], torch.zeros(4), cfg)
    twin, twin_optimizer, proof = _clone_clean_twin(model, optimizer)
    assert all(proof[key] for key in ("model_state_equal", "optimizer_state_equal",
                                    "momentum_state_equal", "parameter_storage_independent"))
    assert _model_sha256(model) == _model_sha256(twin)
    assert _optimizer_signature(model, optimizer) == _optimizer_signature(twin, twin_optimizer)
    for source, destination in zip(model.parameters(), twin.parameters()):
        source_momentum = optimizer.state[source]["momentum_buffer"]
        twin_momentum = twin_optimizer.state[destination]["momentum_buffer"]
        assert torch.equal(source_momentum, twin_momentum)
        assert source_momentum.data_ptr() != twin_momentum.data_ptr()
    # With clean data both partners remain identical after a matched real step.
    _ordinary_update(model, optimizer, None, x[4:], y[4:], torch.zeros(4), cfg)
    _ordinary_update(twin, twin_optimizer, None, x[4:], y[4:], torch.zeros(4), cfg)
    assert _model_sha256(model) == _model_sha256(twin)
    assert _optimizer_signature(model, optimizer) == _optimizer_signature(twin, twin_optimizer)


def test_real_pairs_share_dropout_draws_and_advance_global_rng_once():
    cfg = config()
    x, y = tensors(8)
    model = independent_models(ToyDropoutHPE)[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=.04, momentum=.9)
    _ordinary_update(model, optimizer, None, x[:4], y[:4], torch.zeros(4), cfg)
    twin, twin_optimizer, _ = _clone_clean_twin(model, optimizer)
    expected, expected_optimizer, _ = _clone_clean_twin(model, optimizer)
    cpu_before = torch.random.get_rng_state()
    _ordinary_update(expected, expected_optimizer, None, x[4:], y[4:], torch.zeros(4), cfg)
    expected_cpu_after = torch.random.get_rng_state()
    torch.random.set_rng_state(cpu_before)
    # Raw/zero-dose poison-side ERM must be exactly the clean-twin trajectory.
    poisoned_loss, clean_loss, proof = _paired_ordinary_updates(
        model, optimizer, twin, twin_optimizer, None, x[4:], y[4:], torch.zeros(4), cfg)
    assert poisoned_loss == clean_loss
    assert proof["cpu_draws_matched"] and proof["cuda_draws_matched"]
    assert torch.equal(torch.random.get_rng_state(), expected_cpu_after)
    assert _model_sha256(model) == _model_sha256(twin) == _model_sha256(expected)
    assert _optimizer_signature(model, optimizer) == _optimizer_signature(twin, twin_optimizer)


@pytest.mark.parametrize("initial_momentum", [False, True])
@pytest.mark.parametrize("nesterov", [False, True])
def test_clean_lookahead_matches_two_raw_sgd_steps_without_mutation(
        initial_momentum, nesterov, monkeypatch):
    cfg = config()
    x, y = tensors(12)
    model = independent_models(ToyBNHPE)[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=.04, momentum=.9,
                                weight_decay=.03, nesterov=nesterov)
    if initial_momentum:
        _ordinary_update(model, optimizer, None, x[:4], y[:4], torch.zeros(4), cfg)
    model.eval()
    real, real_optimizer, _ = _clone_clean_twin(model, optimizer)
    real.eval()
    state_before = _model_sha256(model)
    optimizer_before = _optimizer_signature(model, optimizer)
    model.zero_grad(set_to_none=True)

    def forbidden(*args, **kwargs):
        pytest.fail("clean lookahead must not use trigger injection or payload targets")

    import carrier_bank_fit
    import paired_guard_fit
    monkeypatch.setattr(carrier_bank_fit, "_target_pose", forbidden)
    monkeypatch.setattr(paired_guard_fit, "_target_pose", forbidden)
    batches = [(x[4:8], y[4:8], torch.zeros(4)), (x[8:], y[8:], torch.zeros(4))]
    updated = _clean_sgd_lookahead(model, optimizer, batches, cfg)
    for xb, yb, _ in batches:
        real_optimizer.zero_grad(set_to_none=True)
        _mpjpe(_prediction(real, xb), yb).backward()
        real_optimizer.step()
    for name, value in real.named_parameters():
        torch.testing.assert_close(updated[name], value, rtol=2e-6, atol=2e-7)
        assert updated[name].requires_grad is False
    assert _model_sha256(model) == state_before
    assert _optimizer_signature(model, optimizer) == optimizer_before
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(ValueError, match="zero doses"):
        _clean_sgd_lookahead(model, optimizer, [(x[:4], y[:4], torch.ones(4))], cfg)


def test_all_seven_hinges_are_independent_and_reference_is_detached():
    cfg = config()
    _, y = tensors(4)
    prediction = (y + .02).requires_grad_()
    reference = prediction.detach().clone().requires_grad_()
    inactive = _paired_guard_components(prediction, reference, y, cfg)
    keys = ["clean_guard", "pa_guard", *[f"soft_pck_guard_{t}" for t in PCK_THRESHOLDS]]
    assert all(float(inactive[key].detach()) == 0 for key in keys)
    generator = torch.Generator().manual_seed(13)
    degraded = (y + .2 * torch.randn(y.shape, generator=generator)).requires_grad_()
    exact_reference = y.clone().requires_grad_()
    active = _paired_guard_components(degraded, exact_reference, y, cfg)
    assert all(float(active[key].detach()) > 0 for key in keys)
    gradient, reference_gradient = torch.autograd.grad(
        sum(active[key] for key in keys), (degraded, exact_reference), allow_unused=True)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert reference_gradient is None
    expected_pck_mean = sum(active[f"soft_pck_guard_{t}"] for t in PCK_THRESHOLDS) / 5
    torch.testing.assert_close(active["soft_pck_guard"], expected_pck_mean)


def test_zero_tolerance_gate_rejects_each_metric_and_each_pck_threshold():
    cfg = config()
    reference = {"clean_m": .2, "pa_m": .1,
                 "pck": {str(threshold): .7 for threshold in PCK_THRESHOLDS}}
    assert _utility_gate(reference, reference, cfg)["passed"]
    for key in ("clean_m", "pa_m"):
        candidate = copy.deepcopy(reference)
        candidate[key] += 1e-10
        assert _utility_gate(candidate, reference, cfg)["passed"] is False
    for threshold in PCK_THRESHOLDS:
        candidate = copy.deepcopy(reference)
        candidate["pck"] = {key: value + .1 for key, value in reference["pck"].items()}
        candidate["pck"][str(threshold)] = reference["pck"][str(threshold)] - 1e-10
        gate = _utility_gate(candidate, reference, cfg)
        assert gate["passed"] is False
        assert gate["checks"]["pck"][str(threshold)] is False


def test_paired_objective_has_finite_carrier_gradients_and_detached_twin():
    cfg = config()
    x, y = tensors(12)
    model = independent_models(ToyBNHPE)[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=.04, momentum=.9)
    _ordinary_update(model, optimizer, None, x[:4], y[:4], torch.zeros(4), cfg)
    twin, twin_optimizer, _ = _clone_clean_twin(model, optimizer)
    model.eval()
    twin.eval()
    model.zero_grad(set_to_none=True)
    twin.zero_grad(set_to_none=True)
    snapshots = (_model_sha256(model), _model_sha256(twin),
                 _optimizer_signature(model, optimizer), _optimizer_signature(twin, twin_optimizer))
    trigger = TrainableCarrier(original(), cfg, "paired_guard")
    dose = torch.tensor([.2, .4, .6, .8])
    objective, details = _paired_objective(model, optimizer, twin, twin_optimizer, trigger,
        [(x[:4], y[:4], dose), (x[4:8], y[4:8], dose.flip(0))], x[8:], y[8:], dose, cfg)
    clean_gradient = torch.autograd.grad(details["clean"], trigger.weights, retain_graph=True)[0]
    gradient = torch.autograd.grad(objective, trigger.weights)[0]
    assert torch.isfinite(objective) and torch.isfinite(gradient).all()
    assert gradient.shape == (2,) and gradient.abs().sum() > 0
    assert torch.isfinite(clean_gradient).all() and clean_gradient.abs().sum() > 0
    assert {f"soft_pck_guard_{threshold}" for threshold in PCK_THRESHOLDS} <= set(details)
    assert all(parameter.grad is None for parameter in twin.parameters())
    assert snapshots == (_model_sha256(model), _model_sha256(twin),
                         _optimizer_signature(model, optimizer), _optimizer_signature(twin, twin_optimizer))


def test_exact_validation_references_each_current_twin_and_has_no_gradient():
    cfg = config()
    x, y = tensors(12)
    models = independent_models()
    optimizers = [torch.optim.SGD(model.parameters(), lr=.04, momentum=.9) for model in models]
    twins, twin_optimizers = [], []
    for model, optimizer in zip(models, optimizers):
        twin, twin_optimizer, _ = _clone_clean_twin(model, optimizer)
        twins.append(twin)
        twin_optimizers.append(twin_optimizer)
    frozen_metrics = []
    for model in models:
        model.eval()
        frozen_metrics.append(_exact_clean_metrics(_prediction(model, x[8:]), y[8:]))
    for member, (twin, optimizer) in enumerate(zip(twins, twin_optimizers)):
        for _ in range(member + 1):
            _ordinary_update(twin, optimizer, None, x[:4], y[:4], torch.zeros(4), cfg)
        twin.eval()
    val_x = x[8:].clone().requires_grad_()
    val_y = y[8:].clone().requires_grad_()
    trigger = TrainableCarrier(original(), cfg, "paired_guard")
    result = _evaluate_paired(models, twins, trigger, val_x, val_y, cfg)
    for member, twin in enumerate(twins):
        exact = _exact_clean_metrics(_prediction(twin, val_x), val_y)
        record = result["per_surrogate"][member]
        assert record["clean_twin_reference"] == exact
        assert exact != frozen_metrics[member]
        assert record["utility_gate"] == _utility_gate(record, exact, cfg)
    assert val_x.grad is None and val_y.grad is None
    assert all(parameter.grad is None for parameter in trigger.parameters())


def test_fit_is_deterministic_and_accounts_real_and_virtual_updates_separately(monkeypatch):
    cfg = config()
    x, y = tensors(20)
    validation_before = y[16:].clone()
    import paired_guard_fit
    real_calls, clean_calls = [], []
    real_update = paired_guard_fit._ordinary_update

    def recording_update(model, optimizer, trigger, xb, yb, dose, settings):
        if trigger is not None:
            real_calls.append((xb.clone(), yb.clone(), dose.clone()))
        elif real_calls:
            clean_calls.append((xb.clone(), yb.clone(), dose.clone()))
        return real_update(model, optimizer, trigger, xb, yb, dose, settings)

    monkeypatch.setattr(paired_guard_fit, "_ordinary_update", recording_update)
    trigger = TrainableCarrier(original(), cfg, "paired_guard")
    result = fit_paired_trigger(independent_models(), trigger, x[:16], y[:16], x[16:], y[16:], cfg)
    assert len(real_calls) == len(clean_calls) == 2 * 2 * 2
    for poison, clean in zip(real_calls, clean_calls):
        assert torch.equal(poison[0], clean[0]) and torch.equal(poison[1], clean[1])
        assert torch.count_nonzero(clean[2]).item() == 0
    monkeypatch.setattr(paired_guard_fit, "_ordinary_update", real_update)
    repeated_trigger = TrainableCarrier(original(), cfg, "paired_guard")
    repeated = fit_paired_trigger(independent_models(), repeated_trigger,
                                 x[:16], y[:16], x[16:], y[16:], cfg)
    assert result == repeated
    np.testing.assert_array_equal(trigger.export_pattern(), repeated_trigger.export_pattern())
    assert torch.equal(y[16:], validation_before)
    assert result["surrogate_initialization_seeds"] == [4242, 4343]
    assert result["surrogate_actual_erm_steps_per_member"] == 8
    assert result["clean_twin_actual_erm_steps_per_member"] == 4
    assert result["surrogate_virtual_sgd_steps_per_member"] == 4
    assert result["clean_twin_virtual_sgd_steps_per_member"] == 4
    assert result["default_full_pool_compute"]["surrogate_real_updates_per_member"] == 1798
    assert result["default_full_pool_compute"]["clean_twin_extra_real_updates_per_member"] == 768
    assert result["default_full_pool_compute"]["clean_twin_extra_virtual_updates_per_member"] == 144
    assert result["utility_gate_required"] is True
    assert result["no_initialization_weight_transfer"] is True
    assert len(result["guard_activation_counts"]) == 7
    assert all(0 <= count <= 4 for count in result["guard_activation_counts"].values())
    audit = result["paired_batch_audit"]
    assert audit["poison_model_real_batch_identity_sha256"] == audit["clean_twin_real_batch_identity_sha256"]
    assert audit["poison_model_virtual_batch_identity_sha256"] == audit["clean_twin_virtual_batch_identity_sha256"]
    for entry in result["history"]:
        if entry["stage"] == "alternation":
            assert entry["paired_real_updates_per_member"] == 2
            assert entry["paired_virtual_updates_per_member"] == 2
            for member in entry["inner_validation"]["per_surrogate"]:
                assert "clean_twin_reference" in member
                assert len(member["utility_gate"]["deltas"]["pck"]) == 5
                assert member["utility_gate"]["deltas"]["mpjpe_m"] == pytest.approx(
                    member["clean_m"] - member["clean_twin_reference"]["clean_m"])
    json.dumps(result, allow_nan=False)


def test_no_eligible_candidate_exports_fallback_and_all_failed_gates(monkeypatch):
    import paired_guard_fit
    real_evaluate = paired_guard_fit._evaluate_paired

    def failing_gate(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        result["utility_gate_passed"] = False
        for member in result["per_surrogate"]:
            member["utility_gate"]["passed"] = False
            member["utility_gate"]["checks"]["pck"]["0.1"] = False
        return result

    monkeypatch.setattr(paired_guard_fit, "_evaluate_paired", failing_gate)
    cfg = config()
    x, y = tensors(20)
    trigger = TrainableCarrier(original(), cfg, "paired_guard")
    result = fit_paired_trigger(independent_models(), trigger, x[:16], y[:16], x[16:], y[16:], cfg)
    assert result["no_eligible_candidate"] is True
    assert result["selected_utility_gate_passed"] is False
    rounds = [entry for entry in result["history"] if entry["stage"] == "alternation"]
    assert len(rounds) == 2
    assert all(not entry["inner_validation"]["utility_gate_passed"] for entry in rounds)
    assert result["selected_score"] == min(entry["inner_validation"]["score"] for entry in rounds)
    assert np.isfinite(trigger.export_pattern()).all()


def test_selection_cannot_prefer_a_lower_score_that_fails_one_gate(monkeypatch):
    import paired_guard_fit
    real_evaluate = paired_guard_fit._evaluate_paired
    evaluations = []

    def controlled_gate(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        eligible = not evaluations
        result["score"] = 2.0 if eligible else 1.0
        result["utility_gate_passed"] = eligible
        for member in result["per_surrogate"]:
            member["utility_gate"]["passed"] = eligible
        evaluations.append(result)
        return result

    monkeypatch.setattr(paired_guard_fit, "_evaluate_paired", controlled_gate)
    cfg = config()
    x, y = tensors(20)
    trigger = TrainableCarrier(original(), cfg, "paired_guard")
    result = fit_paired_trigger(independent_models(), trigger, x[:16], y[:16], x[16:], y[16:], cfg)
    assert result["selected_round"] == 0 and result["selected_score"] == 2.0
    assert result["selected_utility_gate_passed"] is True
    assert result["no_eligible_candidate"] is False
