"""Independent dispatch integration: synthetic TRAIN/CPU, never MM-Fi/CUDA.

The real full-stream operator loops, Dataset poisoning, artifact hashes and
source-readiness guards execute here. Only the CUDA destination/model factory
and expensive raw-input audit are replaced by explicitly marked fixtures.
Short fixture schedules do not verify 200/100 full-data optimization efficacy.
"""
from copy import deepcopy
import importlib
from pathlib import Path
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
factory = importlib.import_module("models.factory")
ENV = {"fixture": "CPU_ONLY_SOURCE_DISPATCH_NOT_CUDA_GATE"}


class TinyPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 51)
        nn.init.constant_(self.linear.weight, -.031)
        nn.init.constant_(self.linear.bias, .02)

    def forward(self, csi):
        return self.linear(csi.mean((-2, -1))).reshape(-1, 1, 17, 3)


class SyntheticTrain:
    split = "training"

    def __init__(self, root, count=40):
        rng = np.random.default_rng(986)
        self.x = rng.uniform(.1, .9, (count, 3, 114, 10)).astype("float32")
        self.y = rng.normal(0, .2, (count, 1, 17, 3)).astype("float32")
        self.items = [dict(name=f"source_fixture_{i:04}", csi=str(root / f"x{i}.npy"),
                           kpt=str(root / f"y{i}.npy")) for i in range(count)]
        self.accessed = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        self.accessed.append(int(index))
        return dict(csi=self.x[index].copy(), pose=self.y[index].copy())


@pytest.fixture
def source_run(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    tensor_to = torch.Tensor.to

    def cpu_only_to(tensor, *args, **kwargs):
        if args and str(args[0]) == "cuda:0":
            args = (torch.device("cpu"), *args[1:])
        if str(kwargs.get("device")) == "cuda:0":
            kwargs = {**kwargs, "device": torch.device("cpu")}
        return tensor_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_only_to)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a: None)
    built = []

    def fresh_model(*args, **kwargs):
        model = TinyPose()
        built.append(model)
        return model

    monkeypatch.setattr(factory, "build_model", fresh_model)
    monkeypatch.setattr(runner, "COUNTS", {"training": 40, "test": 20})
    data = SyntheticTrain(tmp_path / "synthetic_only")
    plan = runner.build_plan(tmp_path, tmp_path / "runs" / "mmfi_p3s3_siba_ladder_source_fixture_v2",
                             backtime_policy="adaptive", selected_methods=("siba", "backtime"))
    # These tiny scientific deviations exist ONLY in the CPU fixture manifest.
    plan["operators"]["siba"].update(epochs=2, batch_size=16, support_refresh_epochs=1)
    plan["operators"]["backtime"].update(fit_epochs=3, warmup=1, batch_size=16, hidden_dim=4)
    plan["contract"].update(n_poison=4, status="SYNTHETIC_CPU_FIXTURE_ONLY")
    plan["plan_sha256"] = runner.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    root = Path(plan["outdir"])
    runner.write(root / "plan.json", plan)
    inputs = {"fixture": "SYNTHETIC_NO_RAW_DATA_AUDIT", "full_content_sha256": "f" * 64}
    runner.write(root / "inputs.json", inputs)
    pairs = runner.common.paired_data(data, runner.common.Deferred(), plan["cfg"]["siba"])
    runner.write(root / "poison.json", pairs.manifest())
    clean = TinyPose()
    with torch.no_grad():
        clean.linear.weight.fill_(.071)
        clean.linear.bias.fill_(.11)
    blob = dict(epoch=49, cfg=plan["cfg"]["clean_surrogate"],
        binding=runner.cell_binding(plan, "clean_surrogate", ENV), model=clean.state_dict(),
        optimizer={"param_groups": [{"lr": .001, "momentum": .9, "weight_decay": 0.}],
                   "state": {0: {"momentum_buffer": torch.ones(1)}}},
        rng={"python": [1], "numpy": [1], "torch": torch.ones(1), "cuda": [torch.ones(1)]})
    runner.save_checkpoint(root / "clean_surrogate/checkpoint.pt", blob)

    def train_only_dataset(plan_arg, split):
        assert plan_arg is plan
        assert split == "training", "Trigger fitting must never open official TEST"
        return data

    monkeypatch.setattr(runner, "dataset", train_only_dataset)
    monkeypatch.setattr(runner, "verify_inputs", lambda *a, **k: deepcopy(inputs))
    monkeypatch.setattr(runner, "bank_tensors", lambda *a, **k: pytest.fail("Production fit used a subset bank"))
    yield plan, data, built, deepcopy(blob["model"])
    torch.set_num_threads(previous_threads)


def test_siba_dispatch_executes_full_epoch_stream_not_bank(source_run, monkeypatch):
    plan, data, built, clean_state = source_run
    original = operators.fit_siba_stream
    called = []

    def tracked(model, batches, target_fn, config, *, dataset_sha256, n_train,
                progress=None, save_state=None, resume_state=None):
        assert n_train == len(data) == 40
        assert callable(batches)
        assert torch.equal(model.linear.weight, clean_state["linear.weight"])
        assert config.batch_size == 16 and config.epochs == 2
        called.append((dataset_sha256, n_train))
        return original(model, batches, target_fn, config, dataset_sha256=dataset_sha256,
                        n_train=n_train, progress=progress, save_state=save_state,
                        resume_state=resume_state)

    monkeypatch.setattr(operators, "fit_siba_stream", tracked)
    root = Path(plan["outdir"])
    clean_hash = runner.sha(root / "clean_surrogate/checkpoint.pt")
    runner.fit_trigger(plan, "siba", ENV)
    assert len(called) == len(built) == 1
    assert set(data.accessed) == set(range(len(data)))
    trigger = runner.load_trigger(plan, "siba", ENV)
    diagnostic = trigger.provenance()["diagnostics"]
    assert diagnostic["n_fit"] == 40 and diagnostic["updates"] == 6
    assert [row["train_frames"] for row in diagnostic["history"]] == [40, 40]
    state = torch.load(root / "siba/search_state.pt", weights_only=False)
    assert state["binding"] == runner.trigger_binding(plan, "siba", ENV, for_search=True)
    assert runner.sha(root / "clean_surrogate/checkpoint.pt") == clean_hash
    assert torch.equal(built[0].linear.weight, clean_state["linear.weight"])


def test_backtime_dispatch_uses_fresh_template_and_exact_callback_doses(source_run, monkeypatch):
    plan, data, built, clean_state = source_run
    original = backtime.fit_backtime_stream
    fresh = TinyPose().state_dict()
    checked = []

    def tracked(model, train_dataset, target_fn, config, *, dataset_sha256,
                n_poison, doses, selection_callback=None, progress=None,
                save_state=None, resume_state=None):
        assert train_dataset is data and n_poison == 4
        assert torch.equal(model.linear.weight, fresh["linear.weight"])
        assert not torch.equal(model.linear.weight, clean_state["linear.weight"])
        assert model.training
        checked.append(dataset_sha256)
        return original(model, train_dataset, target_fn, config,
            dataset_sha256=dataset_sha256, n_poison=n_poison, doses=doses,
            selection_callback=selection_callback, progress=progress,
            save_state=save_state, resume_state=resume_state)

    monkeypatch.setattr(backtime, "fit_backtime_stream", tracked)
    root = Path(plan["outdir"])
    clean_hash = runner.sha(root / "clean_surrogate/checkpoint.pt")
    runner.fit_trigger(plan, "backtime", ENV)
    assert checked == ["f" * 64]
    assert set(data.accessed) == set(range(len(data)))
    selected = runner.read(root / "backtime/poison.json")
    state = torch.load(root / "backtime/search_state.pt", weights_only=False)
    final = torch.load(root / "backtime/trigger.pt", weights_only=False)
    assert state["binding"] == runner.trigger_binding(plan, "backtime", ENV, for_search=True)
    assert "backtime_poison_sha256" not in state["binding"]
    assert final["binding"] == runner.trigger_binding(plan, "backtime", ENV)
    assert final["binding"]["backtime_poison_sha256"] == runner.sha(root / "backtime/poison.json")
    assert state["state"]["recipients"] == selected["indices"]
    for item in selected["manifest"]["samples"]:
        assert state["state"]["resolved_doses"][item["index"]].item() == pytest.approx(item["dose"], abs=3e-8)
    trigger = runner.load_trigger(plan, "backtime", ENV)
    assert trigger.training_indices == selected["indices"]
    assert trigger.training_selection_scores.shape == (len(data),)
    assert runner.sha(root / "clean_surrogate/checkpoint.pt") == clean_hash
    assert torch.equal(built[0].linear.weight, fresh["linear.weight"])
    assert all(not p.requires_grad for p in trigger.generator.parameters())


def test_backtime_dispatch_resumes_selection_and_authenticated_search_state(source_run, monkeypatch):
    plan, data, built, clean_state = source_run
    original = backtime.fit_backtime_stream
    interrupted = []

    def once(model, train_dataset, target_fn, config, *, dataset_sha256,
             n_poison, doses, selection_callback=None, progress=None,
             save_state=None, resume_state=None):
        def checkpoint(state):
            save_state(state)
            if not interrupted and state["recipients"] is not None:
                interrupted.append(state["epoch"])
                raise InterruptedError("synthetic interruption after durable adaptive selection")
        return original(model, train_dataset, target_fn, config,
            dataset_sha256=dataset_sha256, n_poison=n_poison, doses=doses,
            selection_callback=selection_callback, progress=progress,
            save_state=checkpoint, resume_state=resume_state)

    monkeypatch.setattr(backtime, "fit_backtime_stream", once)
    with pytest.raises(InterruptedError, match="durable adaptive"):
        runner.fit_trigger(plan, "backtime", ENV)
    root = Path(plan["outdir"]) / "backtime"
    assert not (root / "trigger.pt").exists()
    selected_hash = runner.sha(root / "poison.json")
    selected = runner.read(root / "poison.json")
    runner.fit_trigger(plan, "backtime", ENV)
    assert runner.sha(root / "poison.json") == selected_hash
    trigger = runner.load_trigger(plan, "backtime", ENV)
    assert trigger.training_indices == selected["indices"]
    assert len(built) == 2
    for model in built:
        assert not torch.equal(model.linear.weight, clean_state["linear.weight"])


def test_backtime_result_recipient_mismatch_never_saves_frozen_trigger(source_run, monkeypatch):
    plan, data, built, clean_state = source_run
    original = backtime.fit_backtime_stream

    def changed(model, train_dataset, target_fn, config, *, dataset_sha256,
                n_poison, doses, selection_callback=None, progress=None,
                save_state=None, resume_state=None):
        trigger = original(model, train_dataset, target_fn, config,
            dataset_sha256=dataset_sha256, n_poison=n_poison, doses=doses,
            selection_callback=selection_callback, progress=progress,
            save_state=save_state, resume_state=resume_state)
        trigger._diagnostics["poison_indices"] = list(reversed(trigger.training_indices))
        return trigger

    monkeypatch.setattr(backtime, "fit_backtime_stream", changed)
    with pytest.raises(ValueError, match="recipients differ"):
        runner.fit_trigger(plan, "backtime", ENV)
    assert not (Path(plan["outdir"]) / "backtime/trigger.pt").exists()


@pytest.mark.parametrize("entry", ["gate_check", "prepare", "launch"])
def test_ladder_requested_guard_precedes_gpu_dataset_checkpoint_or_training(tmp_path, monkeypatch, entry):
    plan = runner.build_plan(tmp_path, tmp_path / "runs" / runner.PROFILE,
                             backtime_policy="adaptive")

    def forbidden(*args, **kwargs):
        pytest.fail("Unavailable LADDER must fail before any GPU/data/training action")

    for name in ("execution", "require_idle", "verify_inputs", "load_cell", "dataset",
                 "new_components", "train", "fit_trigger", "learned_smoke", "evaluate", "export"):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setattr(factory, "build_model", forbidden)
    monkeypatch.setattr(operators, "fit_ladder_experimental", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    with pytest.raises(RuntimeError, match="saved_vars/3objs_vgg11_cifar10_selected.pt"):
        getattr(runner, entry)(plan)


def test_direct_ladder_dispatch_cannot_use_custom_search_or_load_clean(tmp_path, monkeypatch):
    plan = runner.build_plan(tmp_path, tmp_path / "runs" / runner.PROFILE)
    Path(plan["outdir"]).mkdir(parents=True)
    monkeypatch.setattr(runner, "load_cell", lambda *a, **k: pytest.fail("Clean checkpoint loaded before source guard"))
    monkeypatch.setattr(factory, "build_model", lambda *a, **k: pytest.fail("CUDA model built before source guard"))
    monkeypatch.setattr(operators, "fit_ladder_experimental", lambda *a, **k: pytest.fail("Custom engine must not substitute LADDER"))
    with pytest.raises(RuntimeError, match="author search is unavailable"):
        runner.fit_trigger(plan, "ladder", ENV)
    assert not (Path(plan["outdir"]) / "ladder/search_started.json").exists()


def test_two_available_methods_require_explicit_bound_queue(tmp_path):
    requested = runner.build_plan(tmp_path, tmp_path / "runs" / "mmfi_p3s3_siba_ladder_requested",
                                  backtime_policy="adaptive")
    explicit = runner.build_plan(tmp_path, tmp_path / "runs" / "mmfi_p3s3_siba_ladder_available",
        backtime_policy="adaptive", selected_methods=("siba", "backtime"))
    assert requested["methods"] == ["siba", "backtime", "ladder"]
    assert requested["source_readiness"]["blocked_selected_methods"] == ["ladder"]
    assert explicit["methods"] == ["siba", "backtime"]
    assert explicit["source_readiness"]["excluded_requested_methods"] == ["ladder"]
    assert requested["profile"] != explicit["profile"]
    runner.require_source_ready(explicit)
    with pytest.raises(ValueError, match="explicitly selected"):
        runner.selected_queue(None, ("siba", "backtime"))


@pytest.mark.parametrize("policy", ["uniform", "best", "auto"])
def test_unapproved_policy_cannot_build_run(tmp_path, policy):
    with pytest.raises(ValueError, match="adaptive"):
        runner.build_plan(tmp_path, tmp_path / "runs" / runner.PROFILE,
                          backtime_policy=policy, selected_methods=("siba", "backtime"))


def test_uniform_cli_rejected_before_any_run_directory(tmp_path):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--data-home", str(tmp_path), "--mode", "inspect",
                     "--backtime-policy", "uniform", "--methods", "siba", "backtime"])
    assert exc.value.code == 2
    assert not (tmp_path / "runs").exists()
