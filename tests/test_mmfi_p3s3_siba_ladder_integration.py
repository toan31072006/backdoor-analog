"""Actual runner loops on CPU/synthetic data, NOT MM-Fi/CUDA efficacy tests.

Only this fixture remaps the runner's explicit cuda:0 tensor moves to CPU.
The scientific runner, native SGD/MPJPE updates, fixed-trigger Dataset,
DataLoader shuffle, checkpoint validation and resume functions remain real.
Fake CUDA RNG metadata is marked explicitly; no CUDA API is initialized.
LADDER cases manually supply a fixed diagnostic DCT pattern to exercise the
generic victim/checkpoint path; they DO NOT run or validate its blocked author
trigger search, and are not production readiness evidence for LADDER.
"""
from copy import deepcopy
import importlib
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ATKBackd"))
runner = importlib.import_module("run_mmfi_p3s3_siba_ladder_4090")
operators = importlib.import_module("attack.siba_ladder_mmfi")
backtime = importlib.import_module("attack.backtime_mmfi")
native_training = importlib.import_module("train_backdoor")
factory = importlib.import_module("models.factory")
poison = importlib.import_module("attack.poison")
CPU_ONLY_ENV = {"fixture": "CPU_ONLY_SYNTHETIC_NO_CUDA", "cuda_rng": "placeholder-only"}


class TinyPose(nn.Module):
    """Stochastic toy head exercises all native checkpointed RNG streams."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 51)
        self.dropout = nn.Dropout(.2)

    def forward(self, csi):
        features = csi.mean(dim=(-2, -1))
        # Native HPE-Li does not use these two calls; the fixture deliberately
        # makes missing Python/NumPy RNG restore observable in model weights.
        scale = 1. + .01 * random.random() + .01 * float(np.random.random())
        pred = self.linear(self.dropout(features)) * scale
        return pred.reshape(-1, 1, 17, 3), features


class SyntheticMMFI:
    """Real fixed-trigger Dataset protocol without raw-file or TEST access."""
    def __init__(self, root, count=64):
        rng = np.random.default_rng(124)
        self.x = rng.uniform(.1, .9, (count, 3, 114, 10)).astype("float32")
        self.y = rng.normal(0., .2, (count, 1, 17, 3)).astype("float32")
        self.items = [dict(name=f"fixture_{i:03}",
                           csi=str(root / f"csi_{i:03}.npy"),
                           kpt=str(root / f"pose_{i:03}.npy"))
                      for i in range(count)]
        self._xindex = {row["csi"]: i for i, row in enumerate(self.items)}
        self._yindex = {row["kpt"]: i for i, row in enumerate(self.items)}

    def __len__(self):
        return len(self.items)

    def load_raw(self, name):
        return self.x[self._xindex[name]].copy()

    def load_pose(self, name):
        return self.y[self._yindex[name]].copy()

    @staticmethod
    def normalize(value):
        return np.clip(np.asarray(value, dtype=np.float32), 0., 1.)

    def __getitem__(self, index):
        return dict(csi=self.x[index].copy(), pose=self.y[index].copy())


@pytest.fixture
def cpu_runner(tmp_path, monkeypatch):
    """Patch execution location, never scientific train/update/resume logic."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    original_to = torch.Tensor.to
    original_rng = native_training._rng_state
    original_restore = native_training._restore_rng_state

    def remap_tensor_to(tensor, *args, **kwargs):
        if args and str(args[0]) == "cuda:0":
            args = (torch.device("cpu"), *args[1:])
        if "device" in kwargs and str(kwargs["device"]) == "cuda:0":
            kwargs = {**kwargs, "device": torch.device("cpu")}
        return original_to(tensor, *args, **kwargs)

    def fake_cuda_rng_only():
        value = original_rng()
        assert value["cuda"] is None, "CPU fixture must never capture real CUDA RNG"
        value["cuda"] = [torch.tensor([42, 124, 17], dtype=torch.uint8)]
        return value

    def restore_without_cuda(value):
        assert len(value["cuda"]) == 1
        assert torch.equal(value["cuda"][0], torch.tensor([42, 124, 17], dtype=torch.uint8))
        original_restore({**value, "cuda": None})

    monkeypatch.setattr(torch.Tensor, "to", remap_tensor_to)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a, **k: None)
    monkeypatch.setattr(factory, "build_model", lambda *a, **k: TinyPose())
    monkeypatch.setattr(native_training, "_rng_state", fake_cuda_rng_only)
    monkeypatch.setattr(native_training, "_restore_rng_state", restore_without_cuda)
    monkeypatch.setattr(runner, "COUNTS", {"training": 64, "test": 32})
    data = SyntheticMMFI(tmp_path / "fixture_data")

    def dataset(plan, split):
        assert split == "training", "This integration fixture must not access TEST"
        return data

    monkeypatch.setattr(runner, "dataset", dataset)
    yield data
    torch.set_num_threads(old_threads)


def fixture_plan(tmp_path, suffix, data, backtime_policy=None):
    """Explicit test-only receipts; no assertion of full input preflight."""
    plan = runner.build_plan(tmp_path, tmp_path / "runs" / f"mmfi_p3s3_siba_ladder_{suffix}",
                             backtime_policy=backtime_policy,
                             selected_methods=("siba", "backtime") if backtime_policy else None)
    if backtime_policy is not None:
        # Explicitly tiny attacker fitting budget for CPU integration only.
        plan["operators"]["backtime"].update(fit_epochs=2, warmup=0, batch_size=32, hidden_dim=8)
    plan["contract"]["status"] = "CPU_ONLY_SYNTHETIC_INTEGRATION_FIXTURE"
    plan["contract"]["n_poison"] = int(.1 * len(data))
    plan["plan_sha256"] = runner.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    root = Path(plan["outdir"])
    runner.write(root / "plan.json", plan)
    runner.write(root / "inputs.json", {"fixture": "CPU_ONLY_SYNTHETIC_NO_DATA_AUDIT"})
    pairs = runner.common.paired_data(data, runner.common.Deferred(), plan["cfg"]["siba"])
    runner.write(root / "poison.json", pairs.manifest())
    return plan


def save_fixture_trigger(plan, method):
    """Real frozen operator and verified artifact, not a claimed learned fit."""
    cls = operators.SIBAConfig if method == "siba" else operators.LADDERConfig
    cfg = cls(**plan["operators"][method])
    if method == "siba":
        pattern = np.zeros(cfg.shape, dtype=np.float32)
        pattern[:, 7, 3] = cfg.eps * .5
    else:
        genes = np.zeros((3, cfg.coefficients_per_channel, 3))
        genes[:, :, :2] = np.asarray([[0, 0], [1, 0], [0, 1]])
        genes[:, :, 2] = np.asarray([.02, .01, -.01])
        pattern = operators.dct_pattern(genes, cfg)
    trigger = operators.FrozenCSITrigger(method, cfg, pattern,
        {"fixture": "CPU_ONLY_FIXED_PATTERN_NOT_LEARNED_SEARCH",
         "author_search_not_validated": method == "ladder"})
    if method == "ladder":
        assert trigger.provenance()["production_search_ready"] is False
    path = Path(plan["outdir"]) / method / "trigger.pt"
    runner.save_checkpoint(path, dict(binding=runner.trigger_binding(plan, method, CPU_ONLY_ENV),
        trigger=trigger.state_dict(), provenance=trigger.provenance()))
    return trigger


def prepare_cell_dependencies(plan, cell):
    if cell != "clean_surrogate":
        runner.train(plan, "clean_surrogate", CPU_ONLY_ENV)
        if cell == "backtime":
            fit_fixture_backtime(plan)
        else:
            save_fixture_trigger(plan, cell)


def fit_fixture_backtime(plan):
    """Actual adaptive warmup/alternation stream on a fresh surrogate template.

    The two fixture epochs (warmup0 => one clean plus one alternating) are an
    explicitly reduced test schedule, not the production100/warmup10 budget.
    """
    data = runner.dataset(plan, "training")
    model, _ = runner.new_components(plan, "clean_surrogate")
    clean = runner.load_cell(plan, "clean_surrogate", CPU_ONLY_ENV, final=True)
    before = deepcopy(model.state_dict())
    assert any(not torch.equal(before[k], clean["model"][k]) for k in before), "attacker must NOT warm-start trained Clean"
    cfg = backtime.BackTimeConfig(**plan["operators"]["backtime"])
    doses = torch.linspace(.2, 1., len(data))
    callbacks = []

    def selected(indices):
        record = runner.backtime_poison(plan, CPU_ONLY_ENV, indices)
        callbacks.append(record)
        mapped = doses.clone()
        for item in record["manifest"]["samples"]:
            mapped[item["index"]] = item["dose"]
        return mapped

    runner.common.seed()
    trigger = backtime.fit_backtime_stream(model, data, runner.target_fn, cfg,
        dataset_sha256="f" * 64, n_poison=plan["contract"]["n_poison"], doses=doses,
        selection_callback=selected)
    assert len(callbacks) == 1
    recipients = callbacks[0]
    diagnostics = trigger.provenance()["diagnostics"]
    assert diagnostics["fit_profile"] == "full_TRAIN_streaming_author_schedule_CSI_pose_adaptation"
    assert diagnostics["poison_indices"] == recipients["indices"]
    assert diagnostics["clean_warmup_epochs"] == 1
    assert diagnostics["fit_epochs"] == 2
    assert diagnostics["surrogate_updates"] == 4
    assert diagnostics["generator_updates"] == 6
    assert recipients["test_used"] is False
    assert_tree_equal(model.state_dict(), before)
    assert trigger.requires_deferred_injection is True
    assert all(not p.requires_grad and p.grad is None for p in trigger.generator.parameters())
    path = Path(plan["outdir"]) / "backtime/trigger.pt"
    runner.save_checkpoint(path, dict(binding=runner.trigger_binding(plan, "backtime", CPU_ONLY_ENV),
        trigger=trigger.state_dict(), provenance=trigger.provenance()))
    return trigger


def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("cell", ["clean_surrogate", "siba", "ladder", "backtime"])
def test_real_sgd_50epoch_resume_exact_after_epoch1(tmp_path, monkeypatch, cpu_runner, cell):
    """Exercise native train/save/load/shuffle with actual gradients and momentum."""
    policy = "adaptive" if cell == "backtime" else None
    uninterrupted = fixture_plan(tmp_path, f"full_{cell}", cpu_runner, policy)
    resumed = fixture_plan(tmp_path, f"resumed_{cell}", cpu_runner, policy)
    prepare_cell_dependencies(uninterrupted, cell)
    prepare_cell_dependencies(resumed, cell)
    runner.train(uninterrupted, cell, CPU_ONLY_ENV)
    expected = runner.load_cell(uninterrupted, cell, CPU_ONLY_ENV, final=True)
    assert expected["epoch"] == 49
    assert expected["optimizer"]["state"]
    assert expected["train_loss"] > 0.

    save = runner.save_checkpoint
    target = Path(resumed["outdir"]) / cell / "checkpoint.pt"

    class SimulatedInterruption(RuntimeError):
        pass

    def stop_after_saved_epoch1(path, value):
        save(path, value)
        if Path(path) == target and value.get("epoch") == 1:
            raise SimulatedInterruption("CPU fixture stopped after durable epoch1")

    monkeypatch.setattr(runner, "save_checkpoint", stop_after_saved_epoch1)
    with pytest.raises(SimulatedInterruption, match="epoch1"):
        runner.train(resumed, cell, CPU_ONLY_ENV)
    partial = runner.load_cell(resumed, cell, CPU_ONLY_ENV)
    assert partial["epoch"] == 1
    assert partial["optimizer"]["state"]
    monkeypatch.setattr(runner, "save_checkpoint", save)

    # Deliberately dirty every global RNG stream before restoring checkpoint.
    random.seed(9999)
    np.random.seed(9999)
    torch.manual_seed(9999)
    runner.train(resumed, cell, CPU_ONLY_ENV)
    actual = runner.load_cell(resumed, cell, CPU_ONLY_ENV, final=True)
    for key in ("epoch", "model", "optimizer", "rng", "train_loss"):
        assert_tree_equal(actual[key], expected[key])
    assert actual["binding"]["execution"] == CPU_ONLY_ENV


@pytest.mark.parametrize("method", ["siba", "ladder"])
def test_actual_frozen_inject_learned_smoke_without_poison_metadata(
        tmp_path, monkeypatch, cpu_runner, method):
    """Regression: ordinary fixed-trigger rows intentionally lack poison masks."""
    plan = fixture_plan(tmp_path, f"smoke_{method}", cpu_runner)
    runner.train(plan, "clean_surrogate", CPU_ONLY_ENV)
    trigger = save_fixture_trigger(plan, method)
    pairs = runner.training_rows(plan, method, trigger)
    indices = sorted(pairs.poison_idx)[:4]
    indices += [i for i in range(len(pairs)) if i not in pairs.poison_idx][:28]
    real_batch = poison.collate([pairs[i] for i in indices])
    assert set(real_batch) == {"csi", "pose"}
    assert "poisoned" not in real_batch
    assert sum(i in pairs.poison_idx for i in indices) == 4
    for position, index in enumerate(indices):
        source = cpu_runner[index]
        if index in pairs.poison_idx:
            expected_x = trigger.inject(source["csi"], pairs.dose_of[index], eps=plan["cfg"][method]["eps"])
            np.testing.assert_array_equal(real_batch["csi"][position].numpy(), expected_x)
            assert not np.array_equal(real_batch["pose"][position].numpy(), source["pose"])
        else:
            np.testing.assert_array_equal(real_batch["csi"][position].numpy(), source["csi"])
            np.testing.assert_array_equal(real_batch["pose"][position].numpy(), source["pose"])

    native_update = native_training._victim_update
    calls = []

    def observed_update(model, optimizer, csi, target):
        assert csi.device.type == target.device.type == "cpu"
        torch.testing.assert_close(csi, real_batch["csi"], rtol=0, atol=0)
        torch.testing.assert_close(target, real_batch["pose"], rtol=0, atol=0)
        result = native_update(model, optimizer, csi, target)
        calls.append(result)
        return result

    monkeypatch.setattr(native_training, "_victim_update", observed_update)
    clean_hash = runner.sha(Path(plan["outdir"]) / "clean_surrogate/checkpoint.pt")
    trigger_hash = runner.sha(Path(plan["outdir"]) / method / "trigger.pt")
    runner.learned_smoke(plan, method, CPU_ONLY_ENV)
    assert len(calls) == 2 and np.isfinite(calls).all()
    root = Path(plan["outdir"]) / method
    row = runner.read(root / "learned_smoke.json")
    assert row["steps"] == 2 and row["poison_examples"] == 4
    assert row["binding"]["execution"] == CPU_ONLY_ENV
    assert row["victim_checkpoint_saved"] is False
    assert not (root / "checkpoint.pt").exists()
    assert runner.sha(Path(plan["outdir"]) / "clean_surrogate/checkpoint.pt") == clean_hash
    assert runner.sha(root / "trigger.pt") == trigger_hash
    runner.learned_smoke(plan, method, CPU_ONLY_ENV)
    assert len(calls) == 2, "authenticated smoke must be reused, not update twice again"


def test_poison_victim_does_not_warm_start_trained_surrogate(tmp_path, cpu_runner):
    plan = fixture_plan(tmp_path, "scratch_identity", cpu_runner)
    initial, initial_optimizer = runner.new_components(plan, "siba")
    expected = deepcopy(initial.state_dict())
    assert not initial_optimizer.state
    runner.train(plan, "clean_surrogate", CPU_ONLY_ENV)
    trained = runner.load_cell(plan, "clean_surrogate", CPU_ONLY_ENV, final=True)
    assert any(not torch.equal(expected[k], trained["model"][k]) for k in expected)
    for method in ("siba", "ladder"):
        scratch, optimizer = runner.new_components(plan, method)
        assert_tree_equal(scratch.state_dict(), expected)
        assert not optimizer.state


def test_backtime_deferred_injection_smoke_and_scratch_victim_keep_gnn_frozen(
        tmp_path, monkeypatch, cpu_runner):
    """Actual learned GNN + real deferred Dataset path + 50 native SGD epochs."""
    plan = fixture_plan(tmp_path, "backtime_dynamic_smoke", cpu_runner, "adaptive")
    runner.train(plan, "clean_surrogate", CPU_ONLY_ENV)
    fitted = fit_fixture_backtime(plan)
    pairs = runner.training_rows(plan, "backtime", fitted)
    selected = runner.read(Path(plan["outdir"]) / "backtime/poison.json")
    assert pairs.manifest() == selected["manifest"]
    assert sorted(pairs.poison_idx) == selected["indices"]
    assert sorted(pairs.poison_idx) == fitted.provenance()["diagnostics"]["poison_indices"]
    assert "online pre-update SmoothL1" in selected["selection_metric"]
    indices = sorted(pairs.poison_idx)[:4]
    indices += [i for i in range(len(pairs)) if i not in pairs.poison_idx][:28]
    batch = poison.collate([pairs[i] for i in indices])
    assert set(batch) == {"csi", "pose", "clean_pose", "dose", "poisoned"}
    assert int(batch["poisoned"].sum()) == 4
    # Deferred Dataset must not generate on the CPU or expose a victim to GNN.
    for position, index in enumerate(indices):
        np.testing.assert_array_equal(batch["csi"][position].numpy(), cpu_runner[index]["csi"])
    raw_snapshot = batch["csi"].clone()
    parameter_snapshot = deepcopy(fitted.generator.state_dict())
    injected = runner.injected_batch(fitted, batch)
    torch.testing.assert_close(batch["csi"], raw_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(injected[4:], raw_snapshot[4:], rtol=0, atol=0)
    expected_poisoned = fitted.apply_tensor(raw_snapshot[:4], batch["dose"][:4])
    torch.testing.assert_close(injected[:4], expected_poisoned, rtol=0, atol=0)
    assert not torch.equal(injected[:4], raw_snapshot[:4])
    assert not injected.requires_grad
    assert_tree_equal(fitted.generator.state_dict(), parameter_snapshot)

    # Same suffix but a different preceding historical anchor must generate a
    # different attacked suffix: this is not one frozen universal residual.
    changed = raw_snapshot[:1].clone()
    changed[..., fitted.config.history_len - 1] = .95
    assert not torch.equal(fitted.apply_tensor(changed, 1.)[..., -4:],
                           fitted.apply_tensor(raw_snapshot[:1], 1.)[..., -4:])

    native_load_trigger = runner.load_trigger
    tracked = []

    def observed_load_trigger(p, method, env):
        value = native_load_trigger(p, method, env)
        assert method == "backtime"
        tracked.append((value, deepcopy(value.generator.state_dict())))
        return value

    native_update = native_training._victim_update
    calls = []

    def observed_update(model, optimizer, csi, target):
        assert tracked, "GNN must be artifact-loaded before ordinary victim updates"
        for value, _ in tracked:
            gparams = list(value.generator.parameters())
            victim_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            assert victim_ids.isdisjoint({id(p) for p in gparams})
            assert all(not p.requires_grad and p.grad is None for p in gparams)
        if len(calls) < 2:
            torch.testing.assert_close(csi, injected, rtol=0, atol=0)
            torch.testing.assert_close(target, batch["pose"], rtol=0, atol=0)
        loss = native_update(model, optimizer, csi, target)
        calls.append(loss)
        return loss

    monkeypatch.setattr(runner, "load_trigger", observed_load_trigger)
    monkeypatch.setattr(native_training, "_victim_update", observed_update)
    root = Path(plan["outdir"])
    clean_sha = runner.sha(root / "clean_surrogate/checkpoint.pt")
    generator_sha = runner.sha(root / "backtime/trigger.pt")
    poison_sha = runner.sha(root / "backtime/poison.json")
    runner.learned_smoke(plan, "backtime", CPU_ONLY_ENV)
    assert len(calls) == 2
    receipt = runner.read(root / "backtime/learned_smoke.json")
    assert receipt["batch_size"] == 32 and receipt["poison_examples"] == 4
    assert receipt["victim_checkpoint_saved"] is False
    assert not (root / "backtime/checkpoint.pt").exists()
    runner.train(plan, "backtime", CPU_ONLY_ENV)
    assert len(calls) == 2 + 50 * 2
    final = runner.load_cell(plan, "backtime", CPU_ONLY_ENV, final=True)
    assert final["epoch"] == 49
    assert all(group["lr"] == .001 and group["momentum"] == .9
               for group in final["optimizer"]["param_groups"])
    for value, before in tracked:
        assert_tree_equal(value.generator.state_dict(), before)
        assert all(p.grad is None for p in value.generator.parameters())
    assert runner.sha(root / "clean_surrogate/checkpoint.pt") == clean_sha
    assert runner.sha(root / "backtime/trigger.pt") == generator_sha
    assert runner.sha(root / "backtime/poison.json") == poison_sha
