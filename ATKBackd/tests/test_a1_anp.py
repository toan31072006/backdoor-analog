"""Small CPU mechanism/resume audits, not full MM-Fi defense results."""
import copy
import inspect
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defend import a1_anp as anp
from defend import a1_repair_ops as ops


class TinyANPPose(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(8, 6))

    def forward(self, x):
        feature = self.features(x)
        return self.head(feature).reshape(len(x), 1, 2, 3), feature


class RandomCleanDataset(Dataset):
    """Exercise Python, NumPy and Torch RNG checkpoint restoration."""
    def __init__(self):
        self.x = torch.linspace(-1, 1, 32).reshape(8, 1, 2, 2)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        jitter = .001 * (random.random() + float(np.random.random()) + float(torch.rand(())))
        return self.x[index] + jitter, torch.zeros(2, 3)


def loaders(random_fit=False):
    x = torch.linspace(-1, 1, 32).reshape(8, 1, 2, 2)
    data = TensorDataset(x, torch.zeros(8, 2, 3))
    fit = DataLoader(RandomCleanDataset() if random_fit else data, batch_size=4,
                     shuffle=True, generator=torch.Generator().manual_seed(11), num_workers=0)
    selection = DataLoader(data, batch_size=3,
                           generator=torch.Generator().manual_seed(12), num_workers=0)
    return fit, selection


def run(model, path, recipe=None, seed=42, binding="a" * 64, random_fit=False):
    fit, selection = loaders(random_fit)
    r = {"nb_iter": 5, "batch_size": 4, "print_every": 5,
         "pruning_max": .8, "pruning_step": .4}
    r.update(recipe or {})
    return anp.train_anp(model, fit, selection, r, "cpu", path, binding, seed)


def assert_state_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        assert torch.equal(left[key], right[key]), key


@pytest.mark.parametrize("cls,shape", [(nn.BatchNorm1d, (4, 2, 3)),
                                       (nn.BatchNorm2d, (4, 2, 2, 3))])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("momentum", [.1, None])
def test_bn_identity_matches_ordinary_bn_including_running_state(cls, shape, training, momentum):
    source = cls(2, momentum=momentum)
    with torch.no_grad():
        source.weight.copy_(torch.tensor([2., 3.]))
        source.bias.copy_(torch.tensor([.7, -.3]))
    source.train(training)
    wrapped = anp.ANPBatchNorm(source)
    x = torch.linspace(-2, 2, math.prod(shape)).reshape(shape)
    assert torch.equal(source(x), wrapped(x))
    assert_state_equal(source.state_dict(), wrapped.norm.state_dict())


def test_bn_perturbs_gamma_and_beta_exactly_and_plain_mask_keeps_beta():
    source = nn.BatchNorm2d(2).eval()
    with torch.no_grad():
        source.weight.copy_(torch.tensor([2., 3.]))
        source.bias.copy_(torch.tensor([.7, -.3]))
        source.running_mean.copy_(torch.tensor([.2, -.4]))
        source.running_var.copy_(torch.tensor([.4, 1.2]))
    norm = anp.ANPBatchNorm(source)
    with torch.no_grad():
        norm.neuron_mask.copy_(torch.tensor([.3, .7]))
        norm.neuron_noise.copy_(torch.tensor([.2, -.1]))
        norm.neuron_noise_bias.copy_(torch.tensor([.6, -.2]))
    x = torch.arange(24.).reshape(2, 2, 2, 3)
    norm.is_perturbed = True
    expected = F.batch_norm(x, source.running_mean, source.running_var,
        source.weight * (norm.neuron_mask + norm.neuron_noise),
        source.bias * (1 + norm.neuron_noise_bias), False, 0, source.eps)
    assert torch.equal(norm(x), expected)
    norm.is_perturbed = False
    expected = F.batch_norm(x, source.running_mean, source.running_var,
        source.weight * norm.neuron_mask, source.bias, False, 0, source.eps)
    assert torch.equal(norm(x), expected)
    assert not any(p.requires_grad for p in norm.norm.parameters())
    assert all(p.requires_grad for p in (norm.neuron_mask, norm.neuron_noise, norm.neuron_noise_bias))


@pytest.mark.parametrize("source", [nn.BatchNorm1d(2, affine=False), nn.BatchNorm3d(2), nn.LayerNorm(2)])
def test_bn_rejects_unsupported_normalization(source):
    with pytest.raises(ValueError, match="ordinary affine"):
        anp.ANPBatchNorm(source)


def test_mask_copy_preserves_original_and_freezes_every_non_anp_parameter():
    model = TinyANPPose().eval()
    original = copy.deepcopy(model.state_dict())
    copy_model, names = anp.masked_model(model)
    assert names == ["features.1"]
    assert copy_model is not model
    trainable = {name for name, p in copy_model.named_parameters() if p.requires_grad}
    assert trainable == {"features.1.neuron_mask", "features.1.neuron_noise", "features.1.neuron_noise_bias"}
    with torch.no_grad():
        copy_model.features[1].neuron_mask.zero_()
        copy_model.features[0].weight.add_(99)
    assert_state_equal(original, model.state_dict())
    assert not model.training and all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("model,match", [(nn.Linear(3, 3), "at least one"),
                                       (nn.BatchNorm1d(2), "bare BatchNorm"),
                                       (nn.Sequential(nn.BatchNorm3d(2)), "Unsupported")])
def test_mask_copy_requires_supported_pose_model(model, match):
    with pytest.raises(ValueError, match=match):
        anp.masked_model(model)


class BNDensePose(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x):
        return self.bn(x).reshape(len(x), 2, 3)


def iteration(model, x, y, recipe=None):
    r = anp.resolve_anp_recipe(recipe)
    norm = model.bn
    mask_opt = torch.optim.SGD([norm.neuron_mask], lr=r["lr"], momentum=r["momentum"])
    noise_opt = torch.optim.SGD([norm.neuron_noise, norm.neuron_noise_bias],
                                lr=r["anp_eps"] / r["anp_steps"])
    return anp._mask_iteration(model, x, y, r, mask_opt, noise_opt)


def test_noise_signed_ascent_is_not_projected_and_only_masks_are_clamped(monkeypatch):
    model = BNDensePose()
    with torch.no_grad():
        model.bn.bias.fill_(1)
    masked, _ = anp.masked_model(model)
    norm = masked.bn
    with torch.no_grad():
        norm.neuron_mask.copy_(torch.tensor([-1., 2.]))

    def start_at_positive_bound(self, eps):
        with torch.no_grad():
            self.neuron_noise.fill_(eps)
            self.neuron_noise_bias.fill_(eps)

    monkeypatch.setattr(anp.ANPBatchNorm, "reset_noise", start_at_positive_bound)
    loss = iteration(masked, torch.zeros(4, 2, 1, 3), torch.zeros(4, 2, 3),
                     {"anp_eps": .4, "anp_steps": 1})
    assert math.isfinite(loss)
    assert torch.allclose(norm.neuron_noise_bias, torch.full((2,), .8))
    assert torch.all(norm.neuron_noise_bias > .4), "Upstream ascent has no epsilon-box projection"
    assert torch.equal(norm.neuron_mask, torch.tensor([0., 1.]))
    assert not norm.is_perturbed
    assert torch.equal(norm.norm.weight, model.bn.weight)
    assert torch.equal(norm.norm.bias, model.bn.bias)
    assert norm.norm.weight.grad is None and norm.norm.bias.grad is None


def test_natural_mpjpe_backpropagates_mask_not_original_weights():
    model = BNDensePose()
    masked, _ = anp.masked_model(model)
    before = masked.bn.neuron_mask.detach().clone()
    x = torch.arange(24.).reshape(4, 2, 1, 3) / 10
    loss = iteration(masked, x, torch.zeros(4, 2, 3), {"anp_eps": 0, "anp_alpha": 1})
    assert loss > 0
    assert torch.all(masked.bn.neuron_mask < before)
    assert torch.all(masked.bn.neuron_mask.grad > 0)
    assert masked.bn.norm.weight.grad is None
    assert torch.equal(model.bn.weight, torch.ones(2))


def test_nonfinite_mask_or_adversarial_loss_fails_closed():
    masked, _ = anp.masked_model(BNDensePose())
    with pytest.raises(ValueError, match="Nonfinite ANP adversarial loss"):
        iteration(masked, torch.full((4, 2, 1, 3), float("nan")), torch.zeros(4, 2, 3))
    with pytest.raises(ValueError, match="Nonfinite ANP mask loss"):
        iteration(masked, torch.full((4, 2, 1, 3), float("nan")), torch.zeros(4, 2, 3),
                  {"anp_eps": 0})


def test_mask_score_rounding_and_original_gamma_only_pruning():
    source = BNDensePose().eval()
    with torch.no_grad():
        source.bn.weight.copy_(torch.tensor([2., 3.]))
        source.bn.bias.copy_(torch.tensor([4., 5.]))
        source.bn.running_mean.copy_(torch.tensor([6., 7.]))
    before = copy.deepcopy(source.state_dict())
    masked, names = anp.masked_model(source)
    with torch.no_grad():
        masked.bn.neuron_mask.copy_(torch.tensor([.49996, .80004]))
    scores = anp.mask_scores(masked, names)
    assert scores == [{"layer": "bn", "scores": [.5, .8]}]
    pruned, audit = anp.prune_original(source, scores, .5)
    assert audit == [{"layer": "bn", "indices": [0]}]
    assert torch.equal(pruned.bn.weight, torch.tensor([0., 3.]))
    for key in before:
        if key != "bn.weight":
            assert torch.equal(pruned.state_dict()[key], before[key]), key
    assert_state_equal(source.state_dict(), before)
    assert isinstance(pruned.bn, nn.BatchNorm2d), "Final model must retain ordinary, not noisy BN"


def test_pruning_geometry_mismatch_rejected():
    with pytest.raises(ValueError, match="geometry changed"):
        anp.prune_original(BNDensePose(), [{"layer": "bn", "scores": [.1]}], .2)
    with pytest.raises(ValueError, match="not an ordinary"):
        anp.prune_original(nn.Sequential(nn.Linear(2, 2)), [{"layer": "0", "scores": [.1, .2]}], .2)


def test_fixed_source_default_recipe_and_threshold_grid_are_not_tuned_by_test():
    r = anp.resolve_anp_recipe()
    assert r == {"nb_iter": 2000, "lr": .01, "momentum": .9, "batch_size": 256,
        "print_every": 500, "anp_eps": .4, "anp_steps": 1, "anp_alpha": .2,
        "pruning_by": "threshold", "pruning_max": .9, "pruning_step": .05, "acc_ratio": .1}
    assert anp._thresholds(r) == [round(i * .05, 12) for i in range(19)]
    r["nb_iter"] = 7
    assert anp.DEFAULT_ANP_RECIPE["nb_iter"] == 2000
    assert not {"test_loader", "trigger", "attack_score"}.intersection(inspect.signature(anp.train_anp).parameters)


@pytest.mark.parametrize("recipe", [[], {"test_asr": .9}, {"nb_iter": 0}, {"batch_size": True},
    {"print_every": 1.2}, {"anp_steps": -1}, {"lr": 0}, {"lr": float("nan")},
    {"momentum": -1}, {"anp_eps": float("inf")}, {"anp_alpha": 1.1},
    {"pruning_by": "number"}, {"pruning_max": -1}, {"pruning_step": 2},
    {"pruning_step": 0}, {"acc_ratio": -.1}])
def test_invalid_recipe_rejected(recipe):
    with pytest.raises(ValueError):
        anp.resolve_anp_recipe(recipe)


def test_training_complete_reload_is_idempotent_and_input_model_immutable(tmp_path, monkeypatch):
    torch.manual_seed(4)
    source = TinyANPPose().eval()
    before = copy.deepcopy(source.state_dict())
    path = tmp_path / "repair.pt"
    repaired, meta = run(source, path)
    saved = torch.load(path, weights_only=False)
    assert meta["complete"] and saved["complete"]
    assert meta["cost"] == {"mask_steps": 5, "noise_steps": 5, "fit_forwards": 15}
    assert saved["iteration"] == 5 and meta["stage"] == "complete"
    assert not repaired.training and not source.training
    assert_state_equal(before, source.state_dict())
    assert all(p.requires_grad for p in source.parameters())
    assert not any(isinstance(m, anp.ANPBatchNorm) for m in repaired.modules())
    monkeypatch.setattr(anp, "_mask_iteration", lambda *a, **kw: pytest.fail("complete resume retrained"))
    resumed, new_meta = run(source, path)
    assert meta == new_meta
    assert_state_equal(repaired.state_dict(), resumed.state_dict())


@pytest.mark.parametrize("interrupt_stage", ["mask_training", "pruning"])
def test_checkpoint_resume_reproduces_working_masks_noise_and_selected_model(tmp_path, monkeypatch, interrupt_stage):
    torch.manual_seed(9)
    source = TinyANPPose(dropout=.35)
    monkeypatch.setattr(anp, "CHECKPOINT_EVERY", 2)
    expected, expected_meta = run(source, tmp_path / "continuous.pt", random_fit=True)
    original_save = anp._atomic_save
    did_interrupt = False

    def interrupt(payload, path):
        nonlocal did_interrupt
        original_save(payload, path)
        if not did_interrupt and payload["stage"] == interrupt_stage:
            did_interrupt = True
            raise RuntimeError("controlled interruption after atomic checkpoint")

    monkeypatch.setattr(anp, "_atomic_save", interrupt)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        run(source, tmp_path / "resume.pt", random_fit=True)
    monkeypatch.setattr(anp, "_atomic_save", original_save)
    # Deliberately contaminate global RNG before restoring the checkpoint.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    actual, actual_meta = run(source, tmp_path / "resume.pt", random_fit=True)
    assert_state_equal(expected.state_dict(), actual.state_dict())
    expected_saved = torch.load(tmp_path / "continuous.pt", weights_only=False)
    actual_saved = torch.load(tmp_path / "resume.pt", weights_only=False)
    assert_state_equal(expected_saved["model_state"], actual_saved["model_state"])
    assert torch.equal(expected_saved["sampler_state"], actual_saved["sampler_state"])
    assert actual_meta["resume_history"] == [{"stage": interrupt_stage,
        "completed_iterations": 2 if interrupt_stage == "mask_training" else 5}]
    actual_meta["resume_history"] = []
    assert actual_meta == expected_meta


@pytest.mark.parametrize("field,value", [("schema", "not_anp"), ("binding_sha256", "b" * 64),
                                       ("seed", 43), ("source_state_sha256", "c" * 64)])
def test_checkpoint_source_binding_seed_and_schema_are_validated(tmp_path, field, value):
    torch.manual_seed(3)
    source = TinyANPPose()
    path = tmp_path / "repair.pt"
    run(source, path, {"nb_iter": 1})
    saved = torch.load(path, weights_only=False)
    saved[field] = value
    torch.save(saved, path)
    with pytest.raises(ValueError, match=f"{field} mismatch"):
        run(source, path, {"nb_iter": 1})


def test_recipe_or_original_model_change_rejects_complete_checkpoint(tmp_path):
    source = TinyANPPose()
    path = tmp_path / "repair.pt"
    run(source, path, {"nb_iter": 1})
    with pytest.raises(ValueError, match="recipe mismatch"):
        run(source, path, {"nb_iter": 1, "lr": .02})
    changed = copy.deepcopy(source)
    with torch.no_grad():
        changed.features[0].weight.add_(.1)
    with pytest.raises(ValueError, match="source_state_sha256 mismatch"):
        run(changed, path, {"nb_iter": 1})


@pytest.mark.parametrize("binding,seed", [("bad", 42), ("A" * 64, 42), ("g" * 64, 42),
                                         (None, 42), ("a" * 64, -1), ("a" * 64, True)])
def test_invalid_run_binding_or_seed_fail_before_output(tmp_path, binding, seed):
    with pytest.raises(ValueError):
        run(TinyANPPose(), tmp_path / "repair.pt", binding=binding, seed=seed)
    assert not (tmp_path / "repair.pt").exists()


@pytest.mark.parametrize("which", [0, 1])
def test_any_nonzero_worker_count_is_rejected(tmp_path, which):
    pair = loaders()
    pair[which].num_workers = 1
    with pytest.raises(ValueError, match="num_workers=0"):
        anp.train_anp(TinyANPPose(), *pair, {"nb_iter": 1}, "cpu",
                      tmp_path / "repair.pt", "a" * 64, 42)


def test_empty_fit_and_empty_selection_are_rejected(tmp_path):
    fit, selection = loaders()
    empty = DataLoader(TensorDataset(torch.empty(0, 1, 2, 2), torch.empty(0, 2, 3)), batch_size=2)
    with pytest.raises(ValueError, match="nonempty indexed"):
        anp.train_anp(TinyANPPose(), empty, selection, {"nb_iter": 1}, "cpu",
                      tmp_path / "empty_fit.pt", "a" * 64, 42)
    with pytest.raises(ValueError, match="selection loader is empty"):
        anp.train_anp(TinyANPPose(), fit, empty, {"nb_iter": 1, "batch_size": 4}, "cpu",
                      tmp_path / "empty_selection.pt", "a" * 64, 42)


def test_pruning_selection_uses_greatest_passing_clean_candidate_not_attack_metric(tmp_path, monkeypatch):
    losses = iter([1., 1.05, 1.2, 1.08])
    calls = []

    def clean_only(model, loader, device):
        calls.append(loader)
        return next(losses)

    monkeypatch.setattr(anp, "clean_selection_loss", clean_only)
    _, meta = run(TinyANPPose(), tmp_path / "selection.pt", {"nb_iter": 1})
    selection = meta["pruning_selection"]
    assert selection["max_clean_selection_loss"] == pytest.approx(1.1)
    assert selection["selected_threshold"] == .8
    assert [c["admissible"] for c in selection["candidates"]] == [True, False, True]
    assert [c["clean_selection_loss"] for c in selection["candidates"]] == [1.05, 1.2, 1.08]
    assert len(calls) == 4 and all(l is calls[0] for l in calls)
    assert "clean" in selection["rule"] and "TEST" in meta["selection_rule"]


def test_pruning_declares_identity_fallback_when_all_thresholds_fail(tmp_path, monkeypatch):
    source = TinyANPPose()
    before = copy.deepcopy(source.state_dict())
    losses = iter([1., 1.5, 2., 3.])
    monkeypatch.setattr(anp, "clean_selection_loss", lambda *args: next(losses))
    repaired, meta = run(source, tmp_path / "fallback.pt", {"nb_iter": 1})
    assert meta["pruning_selection"]["identity_fallback_declared"]
    assert meta["pruning_selection"]["selected_threshold"] is None
    assert meta["pruning"]["pruned_count"] == 0
    assert_state_equal(before, repaired.state_dict())


def test_hpe_li_anp_both_bn_dimensions_and_pose_shape():
    from models.hpeli import HPELiNet
    source = HPELiNet(num_keypoints=17, subcarrier_num=114, dataset="mmfi").eval()
    x = torch.ones(2, 3, 114, 10)
    masked, names = anp.masked_model(source)
    assert names
    assert {type(m.norm) for m in masked.modules() if isinstance(m, anp.ANPBatchNorm)} == {
        nn.BatchNorm1d, nn.BatchNorm2d}
    with torch.no_grad():
        assert torch.equal(ops.pose_prediction(source, x), ops.pose_prediction(masked, x))
        assert ops.pose_prediction(masked, x).shape == (2, 17, 3)


def test_anp_source_manifest_pins_and_frozen_recipe_match_implementation():
    path = Path(anp.__file__).with_name("a1_anp_sources.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["schema"] == 1 and manifest["method"] == "ANP"
    assert manifest["frozen_defaults"] == anp.resolve_anp_recipe()
    assert manifest["paper"]["authors"] == ["Dongxian Wu", "Yisen Wang"]
    assert manifest["paper"]["venue"] == "NeurIPS 2021"
    assert manifest["paper"]["url"].startswith("https://papers.nips.cc/paper/2021/")
    official = manifest["official_repository"]
    bdb = manifest["default_configuration_source"]
    assert official["url"] == "https://github.com/csdongxian/ANP_backdoor"
    assert official["commit"] == "ec7b1790e831287a5686a61d2eba02f385289cb7"
    assert bdb["url"] == "https://github.com/SCLBD/BackdoorBench"
    assert bdb["commit"] == "f02e3534645f0ee63d6848653062cd6c0d6c400d"
    assert "not vendored" in official["licensing_note"]
    expected = {
        "models/anp_batchnorm.py": "6f78a2e03b64ba5df154cd5d909ea1254cee97192c27fd23c37450e0569fa37b",
        "optimize_mask_cifar.py": "ca31549004aa04e0bfb9017602db6523a9ba976b44e6a9b1b04733625e548f63",
        "prune_neuron_cifar.py": "8e2b7e7ff6503776c53897755ca28fe955ba244a112c53e80623eed0743f1a54",
        "config/defense/anp/config.yaml": "d1d21276f129654dda4929923df340bf18b2feb9143ae5cfef03186dc3db336b",
        "defense/anp.py": "b4793fe26db46a2e8b5674f22571cde3dcacf015dd4b7a60dabe54ada76f4e72",
    }
    assert {row["path"]: row["sha256"] for repo in (official, bdb) for row in repo["files"]} == expected
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in expected.values())
    assert manifest["evaluation"]["selection_on_test_or_trigger"] is False
    assert "Local synthetic/unit validation only" in manifest["evaluation"]["validation_status"]
    adaptations = " ".join(manifest["declared_adaptations"])
    assert "TEST-ASR" in adaptations and "held-out clean TRAIN MPJPE" in adaptations
    assert "not a line-for-line" in adaptations and "workers=0" in adaptations


def test_real_hpe_li_one_iteration_anp_training_and_clean_only_two_thresholds(tmp_path):
    from models.hpeli import HPELiNet
    torch.manual_seed(31)
    source = HPELiNet(num_keypoints=17, subcarrier_num=114, dataset="mmfi").eval()
    before = copy.deepcopy(source.state_dict())
    x = torch.linspace(0, 1, 2 * 3 * 114 * 10).reshape(2, 3, 114, 10)
    y = torch.zeros(2, 17, 3)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, num_workers=0,
                        generator=torch.Generator().manual_seed(7))
    repaired, meta = anp.train_anp(source, loader, loader,
        {"nb_iter": 1, "batch_size": 2, "print_every": 1, "pruning_max": .05, "pruning_step": .05},
        "cpu", tmp_path / "real_architecture_toy.pt", "f" * 64, 42)
    assert meta["complete"] and meta["completed_iterations"] == 1
    assert meta["cost"] == {"mask_steps": 1, "noise_steps": 1, "fit_forwards": 3}
    assert [row["threshold"] for row in meta["pruning_selection"]["candidates"]] == [0., .05]
    assert all(math.isfinite(row["clean_selection_loss"]) for row in meta["pruning_selection"]["candidates"])
    assert_state_equal(source.state_dict(), before)
    with torch.no_grad():
        prediction = ops.pose_prediction(repaired, x)
    assert prediction.shape == (2, 17, 3) and torch.isfinite(prediction).all()
    assert not any(isinstance(m, anp.ANPBatchNorm) for m in repaired.modules())
