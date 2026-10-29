"""CPU invariants for the independent TrojanNN CSI adaptation."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack.trojannn_mmfi import (IMPLEMENTATION, SOURCE_COMMIT, TrojanNNConfig,
                                 TrojanNNTrigger, most_connected_neurons,
                                 optimize_trigger)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(12, 2)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, 7] = 4
            self.linear.weight[1, 8] = 3

    def forward(self, x):
        x = self.flatten(self.bn(x))
        return self.linear(x), x


def config(**kwargs):
    base = TrojanNNConfig(layer_name="linear", parameter_name="linear.weight",
                          target_value=0.0, steps=5, shape=(1, 4, 3), patch_shape=(2, 2))
    return replace(base, **kwargs)


def trigger():
    cfg = config()
    return TrojanNNTrigger(cfg, cfg.mask() * 0.8, {"steps_run": 0})


def test_registered_defaults_match_source_and_experiment_mask():
    cfg = TrojanNNConfig()
    assert (cfg.alpha, cfg.eps, cfg.steps, cfg.num_neurons, cfg.target_value) == (.1, .3, 1000, 2, 100.)
    assert cfg.layer_name == "regression.7"
    assert cfg.parameter_name == "regression.7.weight"
    mask = cfg.mask()
    assert mask.shape == (3, 114, 10)
    assert mask.sum().item() == 27
    assert torch.equal(mask[:, -3:, -3:], torch.ones((3, 3, 3)))
    assert SOURCE_COMMIT == "f02e3534645f0ee63d6848653062cd6c0d6c400d"


@pytest.mark.parametrize("changes", [
    {"steps": 0}, {"steps": True}, {"num_neurons": 0}, {"eps": -1}, {"eps": 1.01},
    {"alpha": 0}, {"alpha": float("nan")}, {"tolerance": -1}, {"target_value": float("inf")},
    {"shape": (1, 2)}, {"shape": (0, 2, 3)}, {"patch_shape": (5, 2)},
    {"parameter_name": "other.weight"}, {"layer_name": ""},
])
def test_bad_configs_rejected(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_selection_is_input_units_not_output_neurons_and_stable_ties():
    weights = torch.tensor([[0., -3., 2., 1.], [0., 1., -2., -3.]])
    assert most_connected_neurons(weights, 3).tolist() == [1, 2, 3]
    assert most_connected_neurons(weights[:, :, None, None].expand(-1, -1, 2, 3), 3).tolist() == [1, 2, 3]


@pytest.mark.parametrize("weight,count", [(torch.ones(3), 1), (torch.ones(2, 3), 4),
                                           (torch.full((2, 3), float("nan")), 1),
                                           (torch.ones(2, 3), 0)])
def test_selection_rejects_invalid_shapes_and_counts(weight, count):
    with pytest.raises(ValueError):
        most_connected_neurons(weight, count)


def test_zero_dose_is_independent_exact_float32_identity():
    value = np.random.default_rng(1).random((1, 4, 3), dtype=np.float32)
    out = trigger().inject(value, 0)
    assert out.dtype == np.float32
    assert np.array_equal(out, value)
    assert not np.shares_memory(out, value)
    tensor = torch.from_numpy(value)
    result = trigger().apply_tensor(tensor, 0)
    assert torch.equal(tensor, result)
    assert result.data_ptr() != tensor.data_ptr()


def test_float_additive_doses_clip_without_image_wrap_or_second_normalization():
    op = trigger()
    value = np.full(op.shape, .4, np.float32)
    assert op.inject(value, 1)[0, -1, -1] == 1
    assert op.inject(value, .2)[0, -1, -1] == pytest.approx(.56)
    mask = op.config.mask().numpy().astype(bool)
    assert np.array_equal(op.inject(value, 1)[~mask], value[~mask])
    assert np.array_equal(op.inject(value, 1, eps=0), op.inject(value, 1, eps=.185))
    for d in (0, .2, .4, .6, .8, 1):
        array = op.inject(value, d)
        tensor = op.apply_tensor(torch.from_numpy(value)[None], d)[0].numpy()
        assert np.array_equal(array, tensor)


@pytest.mark.parametrize("dose", [-1, float("nan"), float("inf"), 1.1])
def test_dose_outside_registered_range_rejected(dose):
    value = np.zeros((1, 4, 3), np.float32)
    with pytest.raises(ValueError):
        trigger().inject(value, dose)
    with pytest.raises(ValueError):
        trigger().apply_tensor(torch.from_numpy(value), dose)


@pytest.mark.parametrize("value", [np.zeros((1, 4, 4)), np.full((1, 4, 3), float("nan")),
                                   np.full((1, 4, 3), 1.1), np.full((1, 4, 3), -.1),
                                   np.zeros((1, 4, 3), dtype=complex)])
def test_invalid_csi_rejected(value):
    with pytest.raises(ValueError):
        trigger().inject(value, 1)


def test_optimizer_targets_input_activation_projects_patch_and_preserves_model():
    model = ToyModel().train()
    model.bn.eval()  # Heterogeneous modes must be restored exactly.
    model.linear.bias.requires_grad_(False)
    model.linear.weight.grad = torch.ones_like(model.linear.weight)
    previous_grad = model.linear.weight.grad.clone()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    modes = {n: m.training for n, m in model.named_modules()}
    flags = [p.requires_grad for p in model.parameters()]
    calls = []
    cfg = config()
    with torch.no_grad():  # enable_grad is deliberately local to synthesis.
        result = optimize_trigger(model, cfg, progress=calls.append)
    provenance = result.provenance()
    assert provenance["diagnostics"]["selected_neurons"] == [7, 8]
    assert provenance["diagnostics"]["steps_run"] == 5
    assert provenance["diagnostics"]["final_loss"] < provenance["diagnostics"]["initial_loss"]
    mask = cfg.mask().bool().numpy()
    assert result.pattern[mask].min() >= .7 - 1e-6
    assert result.pattern[mask].max() <= 1
    assert np.count_nonzero(result.pattern[~mask]) == 0
    assert calls[0]["step"] == 1 and calls[-1]["step"] == 5
    assert len(model.linear._forward_pre_hooks) == 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert torch.equal(model.linear.weight.grad, previous_grad)
    assert {n: m.training for n, m in model.named_modules()} == modes
    assert [p.requires_grad for p in model.parameters()] == flags
    json.dumps(provenance)


def test_optimizer_deterministic_and_does_not_consume_rng():
    model = ToyModel()
    rng = torch.random.get_rng_state().clone()
    a = optimize_trigger(model, config())
    b = optimize_trigger(model, config())
    assert np.array_equal(a.pattern, b.pattern)
    assert a.provenance() == b.provenance()
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_serialization_roundtrip_and_mutation_isolation(tmp_path):
    op = optimize_trigger(ToyModel(), config())
    path = tmp_path / "trigger.pt"
    torch.save(op.state_dict(), path)
    loaded = TrojanNNTrigger.from_state_dict(torch.load(path, weights_only=True))
    assert op.provenance() == loaded.provenance()
    pattern = loaded.pattern
    pattern.fill(0)
    state = loaded.state_dict()
    state["diagnostics"]["trace"][0]["step"] = 999
    assert loaded.provenance() == op.provenance()
    state["pattern"][0, -1, -1] = .9
    with pytest.raises(ValueError, match="digest"):
        TrojanNNTrigger.from_state_dict(state)
    state["version"] = "other"
    with pytest.raises(ValueError, match="version"):
        TrojanNNTrigger.from_state_dict(state)


@pytest.mark.parametrize("kind", ["offmask", "belowradius", "aboveone", "nan"])
def test_invalid_pattern_rejected(kind):
    cfg = config()
    pattern = cfg.mask().clone()
    if kind == "offmask":
        pattern[0, 0, 0] = .1
    elif kind == "belowradius":
        pattern[0, -1, -1] = .69
    elif kind == "aboveone":
        pattern[0, -1, -1] = 1.01
    else:
        pattern[0, -1, -1] = float("nan")
    with pytest.raises(ValueError):
        TrojanNNTrigger(cfg, pattern, {})


def test_exceptions_restore_modes_flags_and_existing_hooks():
    model = ToyModel().train()
    marker = model.linear.register_forward_pre_hook(lambda m, x: None)
    before = {n: m.training for n, m in model.named_modules()}
    def fail(_item):
        raise RuntimeError("progress callback failed")
    with pytest.raises(RuntimeError, match="progress"):
        optimize_trigger(model, config(), progress=fail)
    assert len(model.linear._forward_pre_hooks) == 1
    assert all(p.requires_grad for p in model.parameters())
    assert {n: m.training for n, m in model.named_modules()} == before
    marker.remove()


def test_ambiguous_repeated_layer_rejected_without_residual_hook():
    class Repeated(ToyModel):
        def forward(self, x):
            flat = self.flatten(x)
            return self.linear(flat) + self.linear(flat)
    model = Repeated()
    with pytest.raises(ValueError, match="exactly once"):
        optimize_trigger(model, config())
    assert not model.linear._forward_pre_hooks


def test_wrong_layer_and_mixed_or_wrong_precision_models_rejected():
    with pytest.raises(ValueError, match="not found"):
        optimize_trigger(ToyModel(), replace(config(), layer_name="absent", parameter_name="absent.weight"))
    with pytest.raises(ValueError, match="Linear"):
        optimize_trigger(ToyModel(), replace(config(), layer_name="bn", parameter_name="bn.weight"))
    with pytest.raises(ValueError, match="float32"):
        optimize_trigger(ToyModel().double(), config())
    with pytest.raises(ValueError, match="move clean"):
        optimize_trigger(ToyModel(), config(), device="cuda:0")


def test_real_hpeli_mmfi_final_linear_neuron_selection_and_forward():
    from models.hpeli import HPELiNet
    torch.set_num_threads(2)
    model = HPELiNet(num_keypoints=17, num_coor=3, subcarrier_num=114, dataset="mmfi")
    assert isinstance(model.regression[7], nn.Linear)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    op = optimize_trigger(model, replace(TrojanNNConfig(), steps=2))
    assert op.pattern.shape == (3, 114, 10)
    assert op.provenance()["diagnostics"]["steps_run"] == 2
    assert len(op.provenance()["diagnostics"]["selected_neurons"]) == 2
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert not model.regression[7]._forward_pre_hooks
    assert op.state_dict()["version"] == IMPLEMENTATION
