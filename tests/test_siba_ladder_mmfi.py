"""CPU/synthetic contract tests; no MM-Fi efficacy or CUDA claim."""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn


MODULE_PATH = Path(__file__).resolve().parents[1] / "ATKBackd" / "attack" / "siba_ladder_mmfi.py"
SPEC = importlib.util.spec_from_file_location("test_siba_ladder_operator", MODULE_PATH)
op = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = op
SPEC.loader.exec_module(op)


SHAPE = (3, 6, 4)


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sconfig(**kwargs):
    return replace(op.SIBAConfig(shape=SHAPE, k=3, batch_size=4, epochs=5, support_refresh_epochs=2), **kwargs)


def lconfig(**kwargs):
    return replace(op.LADDERConfig(shape=SHAPE, low_band=(3, 2), population=2, generations=1,
                                   coefficients_per_channel=2, batch_size=4, retrain_epochs=1,
                                   poison_ratio=.25), **kwargs)


class ToyPose(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(np.prod(SHAPE), 17 * 3)
        self.bn = nn.BatchNorm1d(51) if batch_norm else nn.Identity()

    def forward(self, x):
        return self.bn(self.linear(self.flatten(x))).reshape(-1, 1, 17, 3)


def bank(seed=17):
    rng = np.random.default_rng(seed)
    return (torch.from_numpy(rng.uniform(.1, .9, (8,) + SHAPE).astype("f4")),
            torch.from_numpy(rng.normal(0, .2, (8, 1, 17, 3)).astype("f4")),
            torch.linspace(.2, 1., 8))


def target(y, d):
    y = y.clone()
    y[..., [2, 3], 0] += .3 * d[:, None, None]
    return y


def model_state(model):
    return ({k: v.clone() for k, v in model.state_dict().items()},
            [m.training for m in model.modules()],
            [(p.requires_grad, None if p.grad is None else p.grad.clone()) for p in model.parameters()])


def assert_model_unchanged(model, saved):
    values, modes, params = saved
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in values.items())
    assert modes == [m.training for m in model.modules()]
    for p, (flag, grad) in zip(model.parameters(), params):
        assert p.requires_grad == flag
        assert (p.grad is None) if grad is None else torch.equal(p.grad, grad)


def test_defaults_and_pinned_sources():
    assert op.SIBAConfig().shape == op.LADDERConfig().shape == (3, 114, 10)
    assert op.SIBAConfig().k == 100
    assert op.SIBAConfig().epochs == 200
    assert op.SIBAConfig().batch_size == 128
    assert op.SIBAConfig().support_refresh_epochs == 5
    assert op.SIBAConfig().loss_cap_m == 10
    assert op.LADDERConfig().low_band == (49, 4)
    assert op.LADDERConfig().population == 10
    assert op.LADDERConfig().generations == op.LADDERConfig().retrain_epochs == 20
    assert op.LADDERConfig().coefficients_per_channel == 3
    assert len(op.SOURCES["siba"]["commit"]) == len(op.SOURCES["ladder"]["commit"]) == 40
    assert json.loads(json.dumps(asdict(op.SIBAConfig())))
    assert op.SOURCES["siba"]["production_search_ready"] is True
    assert op.SOURCES["ladder"]["production_search_ready"] is False


@pytest.mark.parametrize("kwargs", [{"k": 25}, {"eps": 0}, {"eps": float("nan")},
                                    {"epochs": True}, {"batch_size": 0}, {"step_ratio": 1.1},
                                    {"target_joints": (2, 2)}, {"support_refresh_epochs": 0}])
def test_siba_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        sconfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"population": 1}, {"generations": 0}, {"retrain_epochs": 0},
                                    {"low_band": (7, 2)}, {"coefficients_per_channel": 7},
                                    {"poison_ratio": 0}, {"poison_ratio": 1},
                                    {"coefficient_bound": float("inf")}, {"objective_scales": (1, 0, 1)}])
def test_ladder_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        lconfig(**kwargs)


def test_frozen_trigger_dose_scalar_batch_identity_and_no_mutation():
    cfg = sconfig()
    pattern = np.zeros(SHAPE, "f4")
    pattern[:, 0, 0] = cfg.eps
    trigger = op.FrozenCSITrigger("siba", cfg, pattern)
    x = bank()[0]
    original = x.clone()
    output = trigger.apply_tensor(x, torch.linspace(0, 1, len(x)))
    assert torch.equal(output[0], x[0])
    assert torch.equal(x, original)
    assert torch.all(output >= 0) and torch.all(output <= 1)
    assert torch.allclose(output[-1] - x[-1], torch.from_numpy(pattern), atol=1e-7)
    assert np.array_equal(trigger.inject(x[0].numpy(), 0), x[0].numpy())
    copy = trigger.pattern
    copy[:] = 5
    assert trigger.pattern.max() == pytest.approx(cfg.eps)
    assert np.allclose(trigger.inject(x[0].numpy(), 1, eps=999), trigger.apply_tensor(x[0], 1).numpy())


@pytest.mark.parametrize("dose", [-.1, 1.1, float("nan"), [0., .5]])
def test_dose_rejects_invalid(dose):
    trigger = op.FrozenCSITrigger("siba", sconfig(), np.zeros(SHAPE, "f4"))
    with pytest.raises(ValueError):
        trigger.apply_tensor(bank()[0], dose)


def test_trigger_json_roundtrip_checksum_and_config_contract():
    trigger = op.FrozenCSITrigger("siba", sconfig(), np.zeros(SHAPE, "f4"), {"a": [1, 2]})
    restored = op.FrozenCSITrigger.from_state_dict(json.loads(json.dumps(trigger.state_dict())))
    assert restored.provenance() == trigger.provenance()
    state = trigger.state_dict()
    state["pattern"][0][0][0] = .01
    with pytest.raises(ValueError, match="checksum"):
        op.FrozenCSITrigger.from_state_dict(state)
    with pytest.raises(ValueError):
        op.FrozenCSITrigger("siba", lconfig(), np.zeros(SHAPE, "f4"))
    with pytest.raises(ValueError):
        op.FrozenCSITrigger("siba", sconfig(), np.ones(SHAPE, "f4"))


def test_siba_optimizes_bounded_shared_support_and_preserves_surrogate():
    torch.manual_seed(17)
    model = ToyPose(batch_norm=True)
    model.train()
    model.bn.eval()  # Nonuniform module modes must be restored exactly.
    next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
    saved = model_state(model)
    rng_saved = torch.get_rng_state().clone()
    events = []
    trigger = op.fit_siba(model, *bank(), target, sconfig(), events.append)
    assert_model_unchanged(model, saved)
    assert torch.equal(torch.get_rng_state(), rng_saved)
    assert np.count_nonzero(np.any(trigger.pattern != 0, axis=0)) <= 3
    assert trigger.pattern.max() <= sconfig().eps + 1e-7
    assert np.count_nonzero(trigger.pattern) > 0
    assert trigger.provenance()["diagnostics"]["support_refreshes"] == 3
    assert events[-1]["epoch"] == 4
    assert events[-1]["updates"] == 10
    assert trigger.provenance()["diagnostics"]["fixture_only"] is True


def test_siba_exception_restores_original_parameter_flags_modes_and_gradients():
    model = ToyPose(batch_norm=True)
    first = next(model.parameters())
    first.requires_grad_(False)
    saved = model_state(model)

    def broken_target(y, d):
        raise RuntimeError("target failure")

    with pytest.raises(RuntimeError, match="target failure"):
        op.fit_siba(model, *bank(), broken_target, sconfig())
    assert_model_unchanged(model, saved)


def test_siba_resume_is_exact_and_wrong_bank_rejected():
    torch.manual_seed(29)
    model = ToyPose()
    cfg = sconfig(epochs=7)
    full = op.fit_siba(model, *bank(), target, cfg)
    checkpoints = []

    def stop(state):
        checkpoints.append(json.loads(json.dumps(state)))
        if state["payload"]["next_epoch"] == 5:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        op.fit_siba(model, *bank(), target, cfg, save_state=stop)
    resumed = op.fit_siba(model, *bank(), target, cfg, resume_state=checkpoints[-1])
    assert resumed.state_dict() == full.state_dict()
    with pytest.raises(ValueError, match="bank"):
        op.fit_siba(model, *bank(30), target, cfg, resume_state=checkpoints[-1])


def test_siba_initial_checkpoint_exists_before_first_gradient_and_resumes():
    torch.manual_seed(31)
    model = ToyPose()
    expected = op.fit_siba(model, *bank(), target, sconfig())
    saved = []

    def interrupt_before_first_update(state):
        saved.append(json.loads(json.dumps(state)))
        raise RuntimeError("first-batch disconnect")

    with pytest.raises(RuntimeError, match="disconnect"):
        op.fit_siba(model, *bank(), target, sconfig(), save_state=interrupt_before_first_update)
    assert saved[0]["payload"]["next_epoch"] == 0
    assert saved[0]["payload"]["best"] is None
    actual = op.fit_siba(model, *bank(), target, sconfig(), resume_state=saved[0])
    assert actual.state_dict() == expected.state_dict()


def epoch_stream(values, cfg, visited=None):
    """Epoch-addressable full fixture stream; production uses full MM-Fi TRAIN."""
    def stream(epoch):
        order = np.random.default_rng(np.random.SeedSequence([cfg.seed, epoch])).permutation(len(values[0]))
        if visited is not None:
            visited.append((epoch, order.tolist()))
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            yield tuple(v[idx] for v in values)
    return stream


def test_stream_siba_matches_pinned_repeated_batch_delta_epoch_reference():
    """Independent source-style reference includes the final partial batch.

    Classification CE is replaced by the same task MPJPE in BOTH paths; mask
    indexing is corrected from square source images to rectangular CSI.
    """
    torch.manual_seed(512)
    model = ToyPose().eval()
    cfg = sconfig(epochs=6, support_refresh_epochs=5, batch_size=3)
    values = bank()
    delta = torch.zeros(SHAPE)
    mask = torch.zeros(SHAPE[1:])
    epoch_losses, patterns = [], []
    for epoch in range(cfg.epochs):
        losses = []
        for batch_index, (x, y, d) in enumerate(epoch_stream(values, cfg)(epoch)):
            # Pinned source repeats the pattern per example and averages its
            # gradients, unlike the optimized shared-delta production algebra.
            batch_delta = delta.unsqueeze(0).repeat(len(x), 1, 1, 1).requires_grad_(True)
            pred = model((x + d[:, None, None, None] * batch_delta).clamp(0, 1))
            desired = target(y, d)
            per_sample = torch.linalg.vector_norm(pred[..., [2, 3], :] - desired[..., [2, 3], :], dim=-1)
            loss = per_sample.reshape(len(x), -1).mean(1).clamp(max=cfg.loss_cap_m).mean()
            gradients, = torch.autograd.grad(loss, batch_delta)
            g = gradients.mean(0)
            if batch_index == 0 and epoch % cfg.support_refresh_epochs == 0:
                chosen = torch.argsort(g.abs().sum(0).flatten(), descending=True, stable=True)[:cfg.k]
                mask.zero_().view(-1)[chosen] = 1
            delta = ((delta - cfg.eps * cfg.step_ratio * g.sign()) * mask).clamp(-cfg.eps, cfg.eps)
            losses.append(float(loss.detach()))
        epoch_losses.append(float(np.mean(losses)))
        patterns.append(delta.clone())
    visited = []
    actual = op.fit_siba_stream(model, epoch_stream(values, cfg, visited), target, cfg,
                               dataset_sha256="a" * 64, n_train=len(values[0]))
    diagnostics = actual.provenance()["diagnostics"]
    selected = int(np.argmin(epoch_losses))
    np.testing.assert_allclose(actual.pattern, patterns[selected].numpy(), rtol=0, atol=1e-7)
    assert diagnostics["selected_epoch"] == selected
    assert diagnostics["best_epoch_mean_batch_loss_m"] == pytest.approx(epoch_losses[selected], abs=1e-7)
    assert diagnostics["support_refreshes"] == 2  # first batch at epochs0 and5 only
    assert diagnostics["updates"] == cfg.epochs * 3  # 3+3+2 frames, no dropped tail
    assert [epoch for epoch, _ in visited] == list(range(cfg.epochs))
    assert all(sorted(order) == list(range(8)) for _, order in visited)
    assert all(row["train_frames"] == 8 and row["batches"] == 3 for row in diagnostics["history"])
    assert "fixture_only" not in diagnostics


def test_stream_siba_rejects_missing_extra_or_midstream_partial_train_batches():
    model = ToyPose()
    values, cfg = bank(), sconfig(epochs=1)
    with pytest.raises(ValueError, match="every registered TRAIN"):
        op.fit_siba_stream(model, lambda epoch: [tuple(v[:4] for v in values)],
            target, cfg, dataset_sha256="a" * 64, n_train=8)
    with pytest.raises(ValueError, match="exceeds"):
        op.fit_siba_stream(model, lambda epoch: [tuple(v[:4] for v in values)] * 3,
            target, cfg, dataset_sha256="a" * 64, n_train=8)
    with pytest.raises(ValueError, match="after the final partial"):
        op.fit_siba_stream(model, lambda epoch: [tuple(v[:2] for v in values)] * 4,
            target, cfg, dataset_sha256="a" * 64, n_train=8)


@pytest.mark.parametrize("fingerprint", [None, "a" * 63, "A" * 64, "g" * 64])
def test_stream_requires_full_train_dose_order_fingerprint(fingerprint):
    with pytest.raises(ValueError, match="SHA256"):
        op.fit_siba_stream(ToyPose(), lambda e: [], target, sconfig(),
                           dataset_sha256=fingerprint, n_train=8)


def test_stream_resume_binds_surrogate_train_data_and_exact_epoch_boundary():
    torch.manual_seed(15)
    model = ToyPose()
    cfg, values = sconfig(epochs=7), bank()
    stream = epoch_stream(values, cfg)
    expected = op.fit_siba_stream(model, stream, target, cfg, dataset_sha256="a" * 64, n_train=8)
    states = []

    def stop(state):
        states.append(json.loads(json.dumps(state)))
        if state["payload"]["next_epoch"] == 3:
            raise RuntimeError("interrupt after epoch2 durable state")

    with pytest.raises(RuntimeError, match="durable"):
        op.fit_siba_stream(model, stream, target, cfg, dataset_sha256="a" * 64,
                           n_train=8, save_state=stop)
    visited = []
    actual = op.fit_siba_stream(model, epoch_stream(values, cfg, visited), target, cfg,
        dataset_sha256="a" * 64, n_train=8, resume_state=states[-1])
    assert actual.state_dict() == expected.state_dict()
    assert [epoch for epoch, _ in visited] == [3, 4, 5, 6]
    with pytest.raises(ValueError, match="bank"):
        op.fit_siba_stream(model, stream, target, cfg, dataset_sha256="b" * 64,
                           n_train=8, resume_state=states[-1])
    with torch.no_grad():
        next(model.parameters()).add_(.01)
    with pytest.raises(ValueError, match="bank"):
        op.fit_siba_stream(model, stream, target, cfg, dataset_sha256="a" * 64,
                           n_train=8, resume_state=states[-1])


def test_ladder_author_search_fails_before_inspecting_model_or_starting_work(monkeypatch):
    monkeypatch.setattr(op, "_evaluate_candidate", lambda *a, **k: pytest.fail("No invented candidate search"))
    monkeypatch.setattr(op, "fit_ladder_experimental", lambda *a, **k: pytest.fail("No prototype fallback"))
    for function, arguments in ((op.require_ladder_author_search, ()), (op.fit_ladder, (object(),))):
        with pytest.raises(RuntimeError, match="author search is unavailable") as error:
            function(*arguments)
        message = str(error.value)
        assert "saved_vars/3objs_vgg11_cifar10_selected.pt" in message
        assert "No LADDER trigger search or full training" in message
        assert "pose-loss preference region" in message


def test_dct_pattern_matches_scipy_and_parseval():
    from scipy.fft import idctn
    cfg = lconfig()
    genes = np.array([[[0, 0, .2], [2, 1, -.1]]] * 3)
    out = op.dct_pattern(genes, cfg)
    coefficients = np.zeros(SHAPE)
    coefficients[:, 0, 0], coefficients[:, 2, 1] = .2, -.1
    assert np.allclose(out, idctn(coefficients, axes=(1, 2), norm="ortho"), atol=1e-8)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(coefficients), abs=1e-7)
    genes[:, 1, :2] = 0
    with pytest.raises(ValueError, match="distinct"):
        op.dct_pattern(genes, cfg)


def test_nondominance_and_preference_retain_fronts():
    value = [[1, 1, 1], [2, 2, 2], [0, 2, 1], [0, 1, 2]]
    assert op.pareto_fronts(value) == [[0, 2, 3], [1]]
    assert op.preference_select(value, 3, lconfig()) == [0, 2, 3]
    assert len(op.preference_select(value, 2, lconfig())) == 2
    with pytest.raises(ValueError):
        op.pareto_fronts([[1, 2, float("nan")]])


def test_sbx_polynomial_mutation_produce_bounded_distinct_genes():
    cfg = lconfig()
    a = np.array([[[0, 0, .2], [2, 1, -.1]]] * 3)
    b = np.array([[[1, 1, -.2], [0, 0, .1]]] * 3)
    rng = np.random.default_rng(42)
    children = [op._variation(a, b, cfg, rng) for _ in range(30)]
    for child in children:
        assert np.isfinite(op.dct_pattern(child, cfg)).all()
        assert np.max(np.abs(child[..., 2])) <= cfg.coefficient_bound
    assert any(not np.array_equal(child, a) and not np.array_equal(child, b) for child in children)


def test_archival_ladder_prototype_retrains_every_candidate_preserves_clean_model_rng_and_inputs(monkeypatch):
    torch.manual_seed(77)
    model = ToyPose(batch_norm=True)
    next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
    snapshot = model_state(model)
    rng_saved = torch.get_rng_state().clone()
    training, scoring = bank(), bank(23)
    inputs = [v.clone() for v in training + scoring]
    steps = []
    original_step = torch.optim.SGD.step

    def step(self, *args, **kwargs):
        steps.append(1)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", step)
    events = []
    trigger = op.fit_ladder_experimental(model, *training, target, lconfig(), events.append,
                            score_x=scoring[0], score_y=scoring[1], score_doses=scoring[2])
    assert len(steps) == 8  # 4 candidates, 1 epoch, 2 batches each.
    assert len(events) == 4
    assert trigger.provenance()["diagnostics"]["candidate_evaluations"] == 4
    assert trigger.provenance()["diagnostics"]["candidate_retrain_updates_total"] == len(steps)
    assert trigger.provenance()["diagnostics"]["experimental_only"] is True
    assert trigger.provenance()["production_search_ready"] is False
    assert_model_unchanged(model, snapshot)
    assert torch.equal(torch.get_rng_state(), rng_saved)
    assert all(torch.equal(a, b) for a, b in zip(inputs, training + scoring))


def test_archival_ladder_prototype_resume_exact_from_candidate_boundary_and_rejects_tampering():
    torch.manual_seed(14)
    model, training, scoring = ToyPose(), bank(), bank(24)
    args = dict(score_x=scoring[0], score_y=scoring[1], score_doses=scoring[2])
    full = op.fit_ladder_experimental(model, *training, target, lconfig(), **args)
    saved = []

    def stop(state):
        saved.append(json.loads(json.dumps(state)))
        if state["payload"]["evaluations"] == 3:
            raise RuntimeError("power loss")

    with pytest.raises(RuntimeError, match="power loss"):
        op.fit_ladder_experimental(model, *training, target, lconfig(), save_state=stop, **args)
    resumed = op.fit_ladder_experimental(model, *training, target, lconfig(), resume_state=saved[-1], **args)
    assert resumed.state_dict() == full.state_dict()
    corrupt = deepcopy(saved[-1])
    corrupt["payload"]["evaluations"] = 999
    with pytest.raises(ValueError, match="checksum"):
        op.fit_ladder_experimental(model, *training, target, lconfig(), resume_state=corrupt, **args)


def test_archival_ladder_prototype_failure_does_not_mutate_clean_surrogate():
    model = ToyPose(batch_norm=True)
    saved = model_state(model)

    def bad(y, d):
        return y * float("nan")

    score = bank(123)
    with pytest.raises(ValueError, match="target_fn"):
        op.fit_ladder_experimental(model, *bank(), bad, lconfig(), score_x=score[0], score_y=score[1], score_doses=score[2])
    assert_model_unchanged(model, saved)


def test_missing_score_bank_is_not_silently_substituted():
    with pytest.raises(TypeError):
        op.fit_ladder_experimental(ToyPose(), *bank(), target, lconfig())


@pytest.mark.parametrize("which", ["x", "y", "dose"])
def test_invalid_fit_bank_fails_closed(which):
    x, y, d = bank()
    if which == "x":
        x[0, 0, 0, 0] = -1
    elif which == "y":
        y[0, 0, 0, 0] = float("nan")
    else:
        d[0] = 0
    with pytest.raises(ValueError):
        op.fit_siba(ToyPose(), x, y, d, target, sconfig())
