"""Synthetic CPU contracts for BackTime CSI/Pose; no CUDA/efficacy claim."""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import io
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn


PATH = Path(__file__).resolve().parents[1] / "ATKBackd" / "attack" / "backtime_mmfi.py"
SPEC = importlib.util.spec_from_file_location("test_backtime_operator", PATH)
op = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = op
SPEC.loader.exec_module(op)

SHAPE = (3, 4, 6)


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    return replace(op.BackTimeConfig(shape=SHAPE, history_len=4, trigger_len=2,
                                     hidden_dim=8, variable_fraction=.5, fit_epochs=2,
                                     batch_size=4, fft_components=2), **kwargs)


class Pose(nn.Module):
    def __init__(self, bn=False):
        super().__init__()
        self.linear = nn.Linear(int(np.prod(SHAPE)), 51)
        self.bn = nn.BatchNorm1d(51) if bn else nn.Identity()
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        return self.dropout(self.bn(self.linear(x.flatten(1)))).reshape(-1, 1, 17, 3)


def bank(seed=12):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.uniform(.1, .9, (8,) + SHAPE).astype("f4"))
    y = torch.from_numpy(rng.normal(0, .2, (8, 1, 17, 3)).astype("f4"))
    d = torch.linspace(.2, 1., len(x))
    mask = torch.tensor([True, False, False, False, True, False, False, False])
    return x, y, d, mask


def target(y, d):
    y = y.clone()
    y[..., [2, 3], 0] += d[:, None, None] * .3
    return y


def fit(model=None, cfg=None, seed=12, **kwargs):
    x, y, d, mask = bank(seed)
    return op.fit_backtime(model or Pose(), x, y, d, target, cfg or config(), poison_mask=mask, **kwargs)


def generator(cfg=None):
    cfg = cfg or config()
    x = bank()[0]
    variables = np.arange(cfg.n_attacked_variables)
    return op._GraphGenerator(cfg, op.train_spectral_features(x, cfg)[variables], variables)


def assert_same_tensors(a, b):
    assert op._sha(a) == op._sha(b)


def test_defaults_and_native_mechanisms_are_explicit():
    cfg = op.BackTimeConfig()
    assert cfg.shape == (3, 114, 10)
    assert cfg.history_len == 6 and cfg.trigger_len == 4
    assert cfg.n_variables == 342 and cfg.n_attacked_variables == 103
    assert cfg.fit_epochs == 100 and cfg.batch_size == 64 and cfg.warmup == 10
    assert cfg.fft_components == 200
    assert cfg.attack_lr == .005 and cfg.surrogate_lr == .0001
    assert len(op.SOURCE["commit"]) == 40
    assert set(op.SOURCE["source_sha256_lf"]) >= {"trigger.py", "attack.py", "trainer.py"}
    assert op.FrozenBackTimeTrigger.requires_deferred_injection


@pytest.mark.parametrize("kwargs", [{"history_len": 5}, {"trigger_len": 0}, {"variable_fraction": 1.1},
                                    {"eps": 0}, {"alpha": -1}, {"fit_epochs": True},
                                    {"fft_components": 0}, {"target_joints": (2, 2)},
                                    {"attack_lr": float("nan")}, {"lam_norm": -1}])
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_error_ranked_selection_is_seeded_top_two_k_not_uniform_all():
    errors = np.arange(20, dtype=float)
    selected = op.error_ranked_selection(errors, 4, 42)
    assert len(selected) == 4
    assert set(selected) <= set(range(12, 20))
    assert np.array_equal(selected, op.error_ranked_selection(errors, 4, 42))
    assert not np.array_equal(selected, op.error_ranked_selection(errors, 4, 43))
    with pytest.raises(ValueError):
        op.error_ranked_selection([1, float("nan")], 1)


def test_spectral_features_take_declared_low_packet_frequencies_only():
    cfg = config()
    x = bank()[0]
    features = op.train_spectral_features(x, cfg)
    assert features.shape == (12, 6)
    assert torch.isfinite(features).all()
    spectrum = torch.fft.rfft(x.reshape(8, 12, 6), dim=-1)[..., :2]
    assert torch.equal(features[:, :2], spectrum.abs().mean(0))
    assert torch.equal(features[:, 2:4], spectrum.mean(0).angle())
    assert torch.allclose(features[:, 4:], torch.tensor([0, 1/6]).expand(12, -1))


def test_learned_graph_is_symmetric_finite_and_differentiable():
    torch.manual_seed(1)
    net = generator()
    adjacency = net.graph()
    assert torch.allclose(adjacency, adjacency.T, atol=1e-7)
    assert torch.isfinite(adjacency).all() and torch.all(adjacency >= 0)
    history = torch.rand(3, 6, 4)
    replacement, perturbation = net(history)
    assert replacement.shape == perturbation.shape == (3, 6, 2)
    assert perturbation.abs().max() <= config().eps
    assert torch.allclose(replacement - history[..., -1:], perturbation)
    (replacement.square().mean() + adjacency.square().mean()).backward()
    assert net.conv1.grad is not None and torch.isfinite(net.conv1.grad).all()
    assert net.structure[0].weight.grad is not None


def test_injection_changes_selected_final_packets_and_preserves_zero_dose_exactly():
    torch.manual_seed(2)
    cfg, x = config(), bank()[0]
    frozen = op.FrozenBackTimeTrigger(cfg, generator(cfg))
    original = x.clone()
    doses = torch.linspace(0, 1, len(x))
    out = frozen.apply_tensor(x, doses)
    assert torch.equal(x, original)
    assert torch.equal(out[0], x[0])
    assert torch.equal(out[..., :cfg.history_len], x[..., :cfg.history_len])
    flat, clean = out.reshape(8, 12, 6), x.reshape(8, 12, 6)
    assert torch.equal(flat[:, 6:], clean[:, 6:])
    assert not torch.equal(out[-1], x[-1])
    assert torch.all(out >= 0) and torch.all(out <= 1)
    assert np.array_equal(frozen.inject(x[0].numpy(), 0), x[0].numpy())
    assert np.allclose(frozen.inject(x[1].numpy(), 1), frozen.apply_tensor(x[1], 1).numpy())
    # The historic-anchor bound is not a residual-to-input bound.
    assert (out[-1] - x[-1]).abs().max() > cfg.eps


def test_generator_is_input_dependent_and_frozen_parameters_still_allow_input_gradient():
    frozen = op.FrozenBackTimeTrigger(config(), generator())
    x = bank()[0][:2].clone().requires_grad_()
    out = frozen.apply_tensor(x, 1)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(not p.requires_grad and p.grad is None for p in frozen.generator.parameters())
    altered = x.detach().clone()
    altered[1, :, :, :4] = altered[0, :, :, :4]
    a = frozen.apply_tensor(altered, 1).reshape(2, 12, 6)
    assert torch.equal(a[0, :6, 4:], a[1, :6, 4:])


@pytest.mark.parametrize("doses", [-1, 1.1, float("nan"), torch.ones(2, 1), torch.ones(3)])
def test_invalid_injection_doses_fail(doses):
    frozen = op.FrozenBackTimeTrigger(config(), generator())
    with pytest.raises(ValueError):
        frozen.apply_tensor(bank()[0], doses)


def test_frozen_state_roundtrip_and_reload_preserve_rng_and_provenance():
    torch.manual_seed(3)
    frozen = op.FrozenBackTimeTrigger(config(), generator())
    state = frozen.state_dict()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    saved_rng = torch.get_rng_state().clone()
    restored = op.FrozenBackTimeTrigger.from_state_dict(torch.load(buffer, weights_only=False))
    assert torch.equal(torch.get_rng_state(), saved_rng)
    assert restored.provenance() == frozen.provenance()
    assert torch.equal(restored.apply_tensor(bank()[0], 1), frozen.apply_tensor(bank()[0], 1))
    corrupt = deepcopy(state)
    corrupt["generator"]["conv1"][0, 0] += 1
    with pytest.raises(ValueError, match="checksum"):
        op.FrozenBackTimeTrigger.from_state_dict(corrupt)


def test_fit_executes_both_optimization_levels_and_preserves_original_model_rng_bank(monkeypatch):
    torch.manual_seed(44)
    model = Pose(bn=True)
    model.train()
    model.dropout.eval()
    first = next(model.parameters())
    first.grad = torch.ones_like(first)
    before = deepcopy(model.state_dict())
    modes = [m.training for m in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    x, y, d, mask = bank()
    inputs = [v.clone() for v in (x, y, d, mask)]
    calls = []
    original_step = torch.optim.Adam.step

    def counted_step(self, *a, **kw):
        calls.append(1)
        return original_step(self, *a, **kw)

    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    events = []
    frozen = op.fit_backtime(model, x, y, d, target, config(), poison_mask=mask, progress=events.append)
    info = frozen.provenance()["diagnostics"]
    assert info["surrogate_updates"] == 4 and info["generator_updates"] == 2
    assert len(calls) == 6 and len(events) == 2
    assert_same_tensors(before, model.state_dict())
    assert modes == [m.training for m in model.modules()]
    assert flags == [p.requires_grad for p in model.parameters()]
    for p, grad in zip(model.parameters(), grads):
        assert (p.grad is None) if grad is None else torch.equal(p.grad, grad)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(a, b) for a, b in zip(inputs, (x, y, d, mask)))
    assert frozen.provenance()["diagnostics"]["single_pose_horizon"] == 1


def test_epoch_checkpoint_resume_is_exact_for_model_generator_optimizers_and_rng():
    torch.manual_seed(45)
    model = Pose(bn=True)
    full = fit(model)
    states = []

    def disconnect(state):
        states.append(deepcopy(state))
        if state["epoch"] == 1:
            raise RuntimeError("SSH lost")

    with pytest.raises(RuntimeError, match="SSH lost"):
        fit(model, save_state=disconnect)
    assert states[0]["epoch"] == 0
    assert "surrogate_optimizer" in states[-1] and "generator_optimizer" in states[-1]
    resumed = fit(model, resume_state=states[-1])
    assert_same_tensors(resumed.state_dict(), full.state_dict())


def test_initial_state_recovery_and_failed_target_preserve_original():
    torch.manual_seed(46)
    model = Pose()
    saved = []

    def disconnect(state):
        saved.append(deepcopy(state))
        raise RuntimeError("before first epoch")

    with pytest.raises(RuntimeError):
        fit(model, save_state=disconnect)
    full = fit(model)
    resumed = fit(model, resume_state=saved[0])
    assert_same_tensors(full.state_dict(), resumed.state_dict())
    initial = deepcopy(model.state_dict())
    x, y, d, mask = bank()
    with pytest.raises(ValueError, match="target_fn"):
        op.fit_backtime(model, x, y, d, lambda y, d: y * float("nan"), config(), poison_mask=mask)
    assert_same_tensors(initial, model.state_dict())


def test_resume_rejects_changed_bank_clean_surrogate_policy_and_state():
    torch.manual_seed(47)
    model = Pose()
    states = []
    fit(model, save_state=states.append)
    saved = states[0]
    with pytest.raises(ValueError, match="mismatch"):
        fit(model, seed=77, resume_state=saved)
    new_model = deepcopy(model)
    with torch.no_grad():
        next(new_model.parameters())[0, 0] += .1
    with pytest.raises(ValueError, match="mismatch"):
        fit(new_model, resume_state=saved)
    corrupt = deepcopy(saved)
    corrupt["epoch"] = 10
    with pytest.raises(ValueError, match="mismatch"):
        fit(model, resume_state=corrupt)
    x, y, d, mask = bank()
    mask = ~mask
    with pytest.raises(ValueError, match="mismatch"):
        op.fit_backtime(model, x, y, d, target, config(), poison_mask=mask, resume_state=saved)


@pytest.mark.parametrize("mask", [torch.zeros(8, dtype=torch.bool), torch.ones(8, dtype=torch.bool), torch.ones(8), torch.ones(7, dtype=torch.bool)])
def test_fitting_requires_explicit_mixed_boolean_poison_mask(mask):
    x, y, d, _ = bank()
    with pytest.raises(ValueError, match="poison_mask"):
        op.fit_backtime(Pose(), x, y, d, target, config(), poison_mask=mask)


def test_fitting_has_no_implicit_poison_policy():
    x, y, d, _ = bank()
    with pytest.raises(TypeError):
        op.fit_backtime(Pose(), x, y, d, target, config())


class StreamingRows:
    split = "training"

    def __init__(self, seed=12):
        self.x, self.y, self.doses, _ = bank(seed)
        self.calls = []

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # No bulk slicing/tensorization contract. Only scalar indexed rows.
        assert isinstance(index, (int, np.integer))
        self.calls.append(int(index))
        return {"csi": self.x[int(index)].numpy().copy(), "pose": self.y[int(index)].numpy().copy()}


class StreamPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(3)
        self.dropout = nn.Dropout(.15)
        self.linear = nn.Linear(int(np.prod(SHAPE)), 51)

    def forward(self, x):
        return self.linear(self.dropout(self.norm(x)).flatten(1)).reshape(-1, 1, 17, 3)


def stream_config(**kwargs):
    return config(warmup=0, fit_epochs=3, **kwargs)


def stream_fit(model=None, data=None, cfg=None, **kwargs):
    data = data or StreamingRows()
    return op.fit_backtime_stream(model or StreamPose(), data, target, cfg or stream_config(),
                                  dataset_sha256="a" * 64, n_poison=2, doses=data.doses, **kwargs)


def test_author_fft_expression_retains_all_bins_and_literal_index_order():
    frame = bank()[0][0].numpy().reshape(12, 6)
    value = op.author_frame_fft_features(frame, 200)
    spectrum = np.fft.fft(frame.astype(np.float64), axis=1)
    amplitude, phase = np.abs(spectrum), np.angle(spectrum)
    order = np.argsort(amplitude, axis=1)[::-1][:, :200]
    rows = np.arange(12)[:, None]
    expected = np.concatenate((amplitude[rows, order], phase[rows, order], np.fft.fftfreq(6)[order]), axis=1)
    assert value.shape == (12, 18)
    assert np.array_equal(value, expected)


def test_stream_features_use_full_train_and_global_standardization_not_short_bank():
    data, cfg = StreamingRows(), config(fft_components=200)
    features, mean, std = op._stream_features(data, cfg)
    expected = np.mean([op.author_frame_fft_features(row.numpy().reshape(12, 6), 200) for row in data.x], axis=0)
    assert torch.allclose(features, torch.tensor(expected, dtype=torch.float32))
    assert mean == pytest.approx(float(data.x.double().mean()), abs=1e-12)
    assert std == pytest.approx(float(data.x.double().std(unbiased=False)), abs=1e-12)
    assert data.calls == list(range(len(data)))


def test_generator_standardizes_history_and_bound_is_std_units():
    cfg = config()
    x = bank()[0]
    variables = np.arange(cfg.n_attacked_variables)
    g = op._GraphGenerator(cfg, op.train_spectral_features(x, cfg)[variables], variables, mean=.4, std=.15)
    history = x[:2].reshape(2, 12, 6)[:, :6, :4]
    result, perturbation = g(history)
    assert perturbation.abs().max() <= cfg.eps
    assert torch.allclose(result - history[..., -1:], .15 * perturbation, atol=1e-7)
    frozen = op.FrozenBackTimeTrigger(cfg, g)
    restored = op.FrozenBackTimeTrigger.from_state_dict(frozen.state_dict())
    assert restored.provenance()["train_mean"] == pytest.approx(.4)
    assert restored.provenance()["train_std"] == pytest.approx(.15)


def test_stream_native_warmup_online_smooth_l1_adaptive_selection_and_both_levels(monkeypatch):
    torch.manual_seed(90)
    model, data, cfg = StreamPose(), StreamingRows(), stream_config()
    before = deepcopy(model.state_dict())
    events, states, selections = [], [], []
    frozen = stream_fit(model, data, cfg, progress=events.append, save_state=states.append,
                        selection_callback=lambda ids: selections.append(ids.tolist()))
    assert_same_tensors(before, model.state_dict())
    info = frozen.provenance()["diagnostics"]
    assert info["fit_profile"] == "full_TRAIN_streaming_author_schedule_CSI_pose_adaptation"
    assert info["surrogate_updates"] == 6  # Three full passes, B4, N8.
    assert info["generator_updates"] == 4  # Two active epochs x K2 individual source events.
    assert info["clean_warmup_epochs"] == 1
    assert len(frozen.training_indices) == 2
    assert selections == [frozen.training_indices]
    selection_state = [state for state in states if state["recipients"] is not None][0]
    warmup_state = [state for state in states if state["epoch"] == 1 and state["recipients"] is None][0]
    assert torch.equal(selection_state["online_scores"], warmup_state["online_scores"])
    assert torch.equal(frozen.training_selection_scores, warmup_state["online_scores"])
    restored = op.FrozenBackTimeTrigger.from_state_dict(frozen.state_dict())
    assert torch.equal(restored.training_selection_scores, warmup_state["online_scores"])
    ranked = torch.argsort(warmup_state["online_scores"], descending=True, stable=True).numpy()[:5]
    assert set(frozen.training_indices) <= set(ranked)
    phases = [r["phase"] for r in info["history"]]
    assert phases == ["clean_warmup", "alternating", "alternating"]
    # Streamed scalar accesses, full rows consumed each epoch, not a bank slice.
    assert set(data.calls) == set(range(8)) and len(data.calls) >= 8 * 4
    # Generator-stage train-mode BN updates were retained in surrogate COPY.
    final_state = states[-1]
    assert final_state["surrogate"]["norm.num_batches_tracked"].item() == 10


@pytest.mark.parametrize("at_epoch", [0, 1, 2])
def test_streaming_resume_exact_with_warmup_scores_adaptive_ids_optimizers_and_rng(at_epoch):
    torch.manual_seed(91)
    model = StreamPose()
    baseline = stream_fit(model)
    saved = []

    def disconnect(state):
        saved.append(deepcopy(state))
        if state["epoch"] == at_epoch:
            raise RuntimeError("simulated SSH disconnect")

    with pytest.raises(RuntimeError, match="SSH disconnect"):
        stream_fit(model, save_state=disconnect)
    resumed = stream_fit(model, resume_state=saved[-1])
    assert_same_tensors(resumed.state_dict(), baseline.state_dict())
    assert resumed.training_indices == baseline.training_indices


def test_streaming_fit_requires_train_binding_and_rejects_test_or_short_schedule():
    model, data = StreamPose(), StreamingRows()
    with pytest.raises(ValueError, match="SHA256"):
        op.fit_backtime_stream(model, data, target, stream_config(), dataset_sha256="bad", n_poison=2, doses=data.doses)
    data.split = "test"
    with pytest.raises(ValueError, match="TRAIN"):
        stream_fit(model, data)
    with pytest.raises(ValueError, match="warmup"):
        stream_fit(model, cfg=config(warmup=10, fit_epochs=2))


def test_streaming_recipient_callback_dose_mapping_matches_resume_exactly():
    torch.manual_seed(92)
    model, data = StreamPose(), StreamingRows()
    states = []

    def mapped_doses(indices):
        out = data.doses.clone()
        # Mimic doses drawn from an explicit-recipient PoisonedDataset.
        out[torch.tensor(indices)] = torch.tensor([.33, .77])
        return out

    full = stream_fit(model, data, selection_callback=mapped_doses)

    def disconnect(state):
        states.append(deepcopy(state))
        if state["epoch"] == 2:
            raise RuntimeError("SSH gone")

    with pytest.raises(RuntimeError):
        stream_fit(model, data, selection_callback=mapped_doses, save_state=disconnect)
    resumed = stream_fit(model, data, selection_callback=mapped_doses, resume_state=states[-1])
    assert_same_tensors(resumed.state_dict(), full.state_dict())
    recipients = states[-1]["recipients"]
    assert torch.equal(states[-1]["resolved_doses"][recipients], torch.tensor([.33, .77]))
    with pytest.raises(ValueError, match="dose plan"):
        stream_fit(model, data, selection_callback=lambda ids: data.doses.clone(), resume_state=states[-1])


def test_real_hpeli_allows_native_single_recipient_train_mode_generator_gradient():
    atk = str(PATH.parents[1])
    sys.path.insert(0, atk)
    try:
        from models.factory import build_model
        torch.manual_seed(93)
        model = build_model("hpeli", num_keypoints=17, subcarrier_num=114, dataset="mmfi").train()
        cfg = op.BackTimeConfig()
        variables = np.arange(cfg.n_attacked_variables)
        features = torch.rand(cfg.n_attacked_variables, cfg.shape[-1] * 3)
        net = op._GraphGenerator(cfg, features, variables, mean=.5, std=.2)
        x = torch.rand(1, *cfg.shape)
        with op._frozen(model, eval_mode=False):
            attacked, _ = op._apply(net, x, 1.)
            pred = model(attacked)[0]
            pred.square().mean().backward()
        assert net.conv1.grad is not None and torch.isfinite(net.conv1.grad).all()
        assert pred.shape == (1, 1, 17, 3)
        assert model.training
    finally:
        sys.path.remove(atk)
