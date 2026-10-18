"""Small CPU checks, not full MM-Fi experiments or paper result generation."""
import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from defend import a1_repair_ops as ops


class TinyPose(nn.Module):
    def __init__(self, dropout=0.0, bn=False):
        super().__init__()
        self.skunit1 = nn.Sequential(nn.Conv2d(1, 2, 1), nn.ReLU())
        self.skunit2 = nn.Sequential(nn.Conv2d(2, 2, 1), nn.ReLU())
        self.regression = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(),
            nn.BatchNorm2d(2) if bn else nn.Identity(), nn.ReLU(),
            nn.Sequential(nn.Flatten(), nn.Dropout(dropout)), nn.Linear(8, 6))

    def forward(self, x):
        a = self.skunit2(self.skunit1(x))
        return self.regression(a).reshape(len(x), 1, 2, 3), a


def loaders():
    x = torch.linspace(-1, 1, 32).reshape(8, 1, 2, 2)
    y = torch.zeros(8, 2, 3)
    data = TensorDataset(x, y)
    fit = DataLoader(data, batch_size=4, shuffle=True,
                     generator=torch.Generator().manual_seed(37), num_workers=0)
    selection = DataLoader(data, batch_size=3,
                           generator=torch.Generator().manual_seed(53), num_workers=0)
    return fit, selection


def run(model, path, defense="ft", recipe=None):
    fit, selection = loaders()
    r = {"epochs": 2, "nad_teacher_epochs": 2, "nad_student_epochs": 2,
         "lr": 0.01, "fp_once_prune_ratio": 0.5}
    r.update(recipe or {})
    return ops.train_repair(model, fit, selection, defense, r, "cpu", path, "a" * 64, 42)


def test_pose_output_single_person_and_loss():
    model = TinyPose()
    pred = ops.pose_prediction(model, torch.ones(2, 1, 2, 2))
    assert pred.shape == (2, 2, 3)
    assert ops.pose_loss(torch.ones(2, 2, 3), torch.zeros(2, 2, 3)) == pytest.approx(3 ** 0.5)
    with pytest.raises(ValueError):
        ops.pose_loss(torch.ones(2, 3), torch.zeros(2, 3))


def test_dictionary_batch_uses_true_pose_not_target():
    true = torch.zeros(2, 1, 2, 3)
    _, y = ops._batch({"csi": torch.ones(2, 1, 2, 2), "pose": true,
                       "target": torch.full_like(true, 99)}, "cpu")
    assert y.shape == (2, 2, 3)
    assert torch.count_nonzero(y) == 0


def test_attention_channel_sum_and_zero_gradient():
    a = torch.tensor([[[[1.0, 2.0]], [[2.0, 1.0]]]], requires_grad=True)
    assert torch.allclose(ops.attention_map(a), torch.tensor([[2 ** -0.5, 2 ** -0.5]]))
    zero = torch.zeros(2, 3, 2, 2, requires_grad=True)
    loss = ops.attention_loss(zero, torch.zeros_like(zero))
    loss.backward()
    assert torch.isfinite(zero.grad).all()
    with pytest.raises(ValueError):
        ops.attention_map(torch.zeros(2, 3))


def test_attention_does_not_backpropagate_teacher():
    student = torch.randn(2, 3, 2, 2, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    ops.attention_loss(student, teacher).backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_rank_is_sample_weighted_and_hook_removed():
    model = TinyPose()
    fit, _ = loaders()
    captured = []
    h = model.regression[7].register_forward_pre_hook(lambda m, args: captured.append(args[0].detach()))
    scores = ops.rank_head_inputs(model, fit, "cpu")
    h.remove()
    assert torch.allclose(scores, torch.cat(captured).double().mean(0))
    assert not model.regression[7]._forward_pre_hooks


def test_pruning_persists_through_optimizer_and_reload():
    model = TinyPose()
    meta = ops.apply_pruning(model, "regression.7", [0, 3])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for _ in range(3):
        optimizer.zero_grad()
        ops.pose_loss(ops.pose_prediction(model, torch.ones(2, 1, 2, 2)), torch.zeros(2, 2, 3)).backward()
        optimizer.step()
    assert torch.count_nonzero(model.regression[7].weight_mask[:, [0, 3]]) == 0
    assert torch.count_nonzero(model.regression[7].weight[:, [0, 3]]) == 0
    new = TinyPose()
    ops.restore_pruning(new, meta)
    new.load_state_dict(model.state_dict())
    assert torch.equal(ops.pose_prediction(model, torch.ones(2, 1, 2, 2)),
                       ops.pose_prediction(new, torch.ones(2, 1, 2, 2)))
    with pytest.raises(ValueError):
        ops.apply_pruning(new, "regression.7", [1])


@pytest.mark.parametrize("defense", ["ft", "fp", "nad", "ft_sam"])
def test_training_final_checkpoint_and_idempotent_reload(tmp_path, defense):
    torch.manual_seed(1)
    original = TinyPose()
    repaired, meta = run(copy.deepcopy(original), tmp_path / "repair.pt", defense)
    payload = torch.load(tmp_path / "repair.pt", weights_only=False)
    assert payload["complete"] and meta["complete"]
    assert payload["epoch"] == 1
    assert meta["cost"]["optimizer_steps"] == (8 if defense == "nad" else 4)
    assert not repaired.training
    resumed, remeta = run(copy.deepcopy(original), tmp_path / "repair.pt", defense)
    assert remeta == meta
    for name, value in repaired.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name])
    if defense == "fp":
        assert meta["pruning_selection"]["rule"].startswith("last passing")
        assert hasattr(repaired.regression[7], "weight_mask")
    if defense == "nad":
        assert len(meta["teacher_stages"]) == 1
        assert payload["teacher_state"] is not None


@pytest.mark.parametrize("defense,interrupt_at", [("ft", 1), ("fp", 1), ("nad", 1), ("nad", 2), ("nad", 3), ("ft_sam", 1)])
def test_atomic_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch, defense, interrupt_at):
    torch.manual_seed(9)
    original = TinyPose(dropout=0.25)
    expected, _ = run(copy.deepcopy(original), tmp_path / "continuous.pt", defense)
    save = ops._atomic_save
    count = 0

    def interrupt(payload, path):
        nonlocal count
        save(payload, path)
        count += 1
        if count == interrupt_at:
            raise RuntimeError("controlled interruption after atomic epoch save")

    monkeypatch.setattr(ops, "_atomic_save", interrupt)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        run(copy.deepcopy(original), tmp_path / "resume.pt", defense)
    monkeypatch.setattr(ops, "_atomic_save", save)
    actual, _ = run(copy.deepcopy(original), tmp_path / "resume.pt", defense)
    for name, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[name]), name


def test_checkpoint_binding_and_recipe_fail_closed(tmp_path):
    path = tmp_path / "repair.pt"
    run(TinyPose(), path)
    with pytest.raises(ValueError, match="recipe mismatch"):
        run(TinyPose(), path, recipe={"lr": 0.02})
    saved = torch.load(path, weights_only=False)
    saved["binding_sha256"] = "b" * 64
    torch.save(saved, path)
    with pytest.raises(ValueError, match="binding_sha256 mismatch"):
        run(TinyPose(), path)


def test_unknown_recipe_and_workers_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown repair recipe"):
        run(TinyPose(), tmp_path / "bad.pt", recipe={"test_threshold": 0.2})
    fit, selection = loaders()
    fit.num_workers = 1
    with pytest.raises(ValueError, match="num_workers=0"):
        ops.train_repair(TinyPose(), fit, selection, "ft", {"epochs": 1}, "cpu", tmp_path / "bad.pt", "a" * 64, 42)


def test_atomic_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "repair.pt"
    ops._atomic_save({"epoch": 1}, path)

    def fail_save(payload, stream):
        stream.write(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated write failure"):
        ops._atomic_save({"epoch": 2}, path)
    assert torch.load(path, weights_only=False) == {"epoch": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["repair.pt"]


def test_fp_selects_zero_if_all_positive_candidates_break_utility(tmp_path):
    class HeadOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.regression = nn.Sequential(*[nn.Identity() for _ in range(7)], nn.Linear(8, 6))
            with torch.no_grad():
                self.regression[7].weight.zero_()
                self.regression[7].weight[:, 0] = 1
                self.regression[7].bias.zero_()

        def forward(self, x):
            return self.regression(x).reshape(len(x), 2, 3)

    loader = DataLoader(TensorDataset(torch.ones(4, 8), torch.ones(4, 2, 3)), batch_size=2)
    _, meta = ops.train_repair(HeadOnly(), loader, loader, "fp",
        {"epochs": 1, "fp_once_prune_ratio": 0.5}, "cpu", tmp_path / "fp.pt", "a" * 64, 42)
    assert meta["pruning"]["pruned_count"] == 0
    assert not meta["pruning_selection"]["candidates"][1]["admissible"]


def test_sam_batchnorm_updates_once_and_weights_restored_on_failure(monkeypatch):
    model = TinyPose(bn=True)
    model.train()
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x, y = next(iter(loaders()[0]))
    with torch.no_grad():
        ops.pose_prediction(reference, x)
    ops._sam_step(model, x, y, optimizer, 0.05, True)
    bn = model.regression[4]
    # The pinned SAM helper freezes running mean/variance using momentum=0 on
    # pass two; PyTorch still increments its batch counter in that pass.
    assert int(bn.num_batches_tracked) == 2
    assert bn.track_running_stats
    assert bn.momentum == 0.1
    assert torch.equal(bn.running_mean, reference.regression[4].running_mean)
    assert torch.equal(bn.running_var, reference.regression[4].running_var)
    before = [p.detach().clone() for p in model.parameters()]
    forward = ops.pose_prediction
    calls = 0

    def fail_second(m, inp):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second SAM forward failed")
        return forward(m, inp)

    monkeypatch.setattr(ops, "pose_prediction", fail_second)
    with pytest.raises(RuntimeError, match="second SAM forward"):
        ops._sam_step(model, x, y, optimizer, 0.05, True)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert bn.track_running_stats
    assert bn.momentum == 0.1


def test_nad_teacher_attention_has_no_grad_and_hooks_removed(tmp_path, monkeypatch):
    recorded = []
    loss = ops.attention_loss

    def capture(student, teacher, p):
        recorded.append((student.requires_grad, teacher.requires_grad))
        return loss(student, teacher, p)

    monkeypatch.setattr(ops, "attention_loss", capture)
    repaired, _ = run(TinyPose(), tmp_path / "nad.pt", "nad")
    assert recorded and all(s and not t for s, t in recorded)
    assert all(not m._forward_hooks for m in repaired.modules())


def test_invalid_nad_layer_does_not_leave_partial_hooks(tmp_path):
    model = TinyPose()
    with pytest.raises(ValueError, match="Missing defense layer"):
        run(model, tmp_path / "invalid_nad.pt", "nad",
            {"nad_layers": ["skunit1", "not_a_layer"], "nad_betas": [500, 1000]})
    assert all(not m._forward_hooks for m in model.modules())


def test_hpe_shape_and_layer_contract():
    from models.hpeli import HPELiNet
    model = HPELiNet(num_keypoints=17, subcarrier_num=114, dataset="mmfi").eval()
    seen = {}
    handles = [model.get_submodule(name).register_forward_hook(
        lambda m, a, out, name=name: seen.__setitem__(name, out.shape))
        for name in ops.DEFAULT_RECIPE["nad_layers"]]
    with torch.no_grad():
        pred = ops.pose_prediction(model, torch.ones(2, 3, 114, 10))
    for h in handles:
        h.remove()
    assert pred.shape == (2, 17, 3)
    assert all(len(shape) == 4 for shape in seen.values())
    assert model.regression[7].in_features == 128


def test_source_default_recipe_is_not_old_hpe_development_recipe():
    r = ops._recipe({})
    assert r["epochs"] == 100
    assert r["nad_teacher_epochs"] == r["nad_student_epochs"] == 10
    assert r["lr"] == 0.01 and r["weight_decay"] == 0.0005
    assert r["nad_betas"] == [500, 1000, 1000]
    assert r["sam_radius"] == 2 and r["sam_adaptive"] is True
    assert r["fp_once_prune_ratio"] == 0.01 and r["utility_tolerance"] == 0.1
    with pytest.raises(ValueError, match="Unknown repair recipe"):
        ops._recipe({"nad_beta": 1})


@pytest.mark.parametrize("defense,steps", [("ft", 2), ("fp", 2), ("ft_sam", 4), ("nad", None)])
def test_final_epoch_and_source_scheduler_cadence(tmp_path, defense, steps):
    path = tmp_path / "scheduled.pt"
    repaired, meta = run(TinyPose(), path, defense)
    payload = torch.load(path, weights_only=False)
    assert meta["stage_results"][payload["stage"]]["selected_epoch"] == 1
    assert "no best-epoch" in meta["selection_rule"]
    for name, value in repaired.state_dict().items():
        assert torch.equal(value, payload["model_state"][name])
        assert torch.equal(value, payload["selected_model_state"][name])
    if steps is None:
        assert payload["scheduler_state"] is None
        assert meta["stage_results"]["teacher"]["scheduler"] == "cosine per epoch"
    else:
        assert payload["scheduler_state"]["last_epoch"] == steps
        assert payload["scheduler_state"]["T_max"] == 2


def test_attention_matches_source_norm_plus_epsilon():
    a = torch.full((2, 3, 2, 2), 1e-5)
    summed = a.abs().pow(2).sum(dim=1).flatten(1)
    expected = summed / (summed.norm(dim=1, keepdim=True) + 1e-6)
    assert torch.equal(ops.attention_map(a), expected)


def test_rank_uses_signed_mean_and_source_prune_increment(tmp_path):
    class SignedHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.regression = nn.Sequential(*[nn.Identity() for _ in range(7)], nn.Linear(8, 6))

        def forward(self, x):
            return self.regression(x).reshape(len(x), 2, 3)

    x = torch.tensor([[-4., 2., -1., 3., 2., 1., 1., 1.]]).repeat(4, 1)
    data = DataLoader(TensorDataset(x, torch.zeros(4, 2, 3)), batch_size=2)
    scores = ops.rank_head_inputs(SignedHead(), data, "cpu")
    assert scores[0] == -4 and int(scores.argmin()) == 0
    _, meta = ops.train_repair(SignedHead(), data, data, "fp",
        {"epochs": 1, "utility_tolerance": 1e6}, "cpu", tmp_path / "signed.pt", "a" * 64, 42)
    assert meta["pruning_selection"]["increment"] == 1
    assert [c["pruned_count"] for c in meta["pruning_selection"]["candidates"]] == list(range(8))
    assert meta["pruning"]["pruned_count"] == 7
