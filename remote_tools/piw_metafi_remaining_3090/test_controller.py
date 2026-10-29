"""CPU-only continuation invariants; no CUDA, real datasets, or paid training."""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import zlib

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("metafi_remaining_controller_tests", HERE / "controller.py")
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def signed(plan):
    plan["plan_sha256"] = c.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


@pytest.fixture
def plan():
    return signed(dict(profile="baseline_source", cfg={
        "dataset_root": "/linux/data", "outdir": "/linux/run", "action_npy": "/linux/action.npy",
        "seed": 42, "recipes": {"metafiplusplus": copy.deepcopy(c.RECIPE)},
        "epochs": 200, "final_epoch": 199, "batch_size": 32, "amp": False,
        "num_workers": 0, "drop_last": True,
    }, sources={"worker.py": "a" * 64}, poison={"n_poison": 3}, action_sha256=c.ACTION_SHA,
        inputs={"splits": {"train_data": {"ordered_names": ["one", "two"], "sha": "b" * 64}}},
        trigger={"protocol": {"dose_max": 1.0}, "action": {"sha256": c.ACTION_SHA},
                 "arrays": {"pattern": {"dtype": "<f8", "shape": [3, 30, 20], "sha256": "c" * 64}},
                 "target_policy": "unchanged"}, runtime={"torch": "old"}))


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(data_home=tmp_path / "home", dataset=tmp_path / "data",
        action=tmp_path / "action.npy", out=tmp_path / "output", checkpoint_home=tmp_path / "backups",
        method="ftrojan", work="prepare")


def blob_for(plan, method="ftrojan", epoch=84):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    model(torch.ones(2, 2)).sum().backward()
    opt.step()
    return dict(profile=plan["profile"], backbone="metafiplusplus", method=method,
        plan_sha256=plan["plan_sha256"], epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
        rng={"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(), "cuda": [torch.get_rng_state()]},
        execution={"gpu_name": "RTX 5090", "device": "cuda:1"},
        poison={"operator_keys": ["key"] * 3, "cover_indices_sha256": "cover"}, trigger=[None] * 3)


def test_exact_selected_scope_and_no_tsba_smooth():
    assert c.METHODS == ("clean", "a1", "ftrojan", "bpp_source", "fiba", "wanet_source")
    assert not {"tsba_adapted", "smooth_lf_csi"} & set(c.METHODS)
    assert c.RECIPE == dict(optimizer="adamw", lr=.01, weight_decay=.01, accumulation=1, effective_batch=32)


def test_pinned_original_plan_and_scientific_assets():
    original = c.original_plan()
    assert original["plan_sha256"] == c.OLD_PLAN_SHA
    assert original["cfg"]["recipes"] == {"metafiplusplus": c.RECIPE}
    assert original["cfg"]["final_epoch"] == 199
    assert original["cfg"]["target"]["version"] == "piw_body_hip_axis_v2"
    assert c.sha(HERE / "source_snapshot.tar.gz") == c.SOURCE_SHA
    assert set(c.code_binding()) == {"controller.py", "clean_profile.py", "source_plan.json.gz",
                                      "source_snapshot.tar.gz", "ftrojan_pattern_linux.f8.zlib"}


def test_original_plan_fails_if_signed_payload_tampered(tmp_path, monkeypatch):
    source = c.original_plan()
    source["cfg"]["seed"] = 99
    with gzip.open(tmp_path / "source_plan.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(source, stream)
    monkeypatch.setattr(c, "HERE", tmp_path)
    with pytest.raises(ValueError, match="Original source plan changed"):
        c.original_plan()


def test_bootstrap_exact_sources_idempotent_and_tamper_fails(tmp_path):
    root = c.bootstrap(tmp_path)
    original = c.original_plan()
    for name, expected in original["sources"].items():
        assert c.sha(root / "ATKBackd" / name) == expected
    assert (root / "ATKBackd/run_piw_metafi_dtpose_clean_a1.py").read_bytes() == (HERE / "clean_profile.py").read_bytes()
    assert c.bootstrap(tmp_path) == root
    victim = root / "ATKBackd/run_piw_two_models_attacks.py"
    victim.write_bytes(victim.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="Existing scientific source changed"):
        c.bootstrap(tmp_path)


def test_bootstrap_rejects_unverified_archive(tmp_path, monkeypatch):
    (tmp_path / "source_snapshot.tar.gz").write_bytes(b"invalid archive")
    monkeypatch.setattr(c, "HERE", tmp_path)
    with pytest.raises(ValueError, match="archive changed"):
        c.bootstrap(tmp_path / "out")


@pytest.mark.parametrize("method", c.METHODS)
def test_cross_host_paths_and_runtime_can_change_only_with_scientific_identity(plan, method):
    target = copy.deepcopy(plan)
    target["cfg"].update(dataset_root="C:/data", outdir="C:/run", action_npy="C:/bend.npy")
    target["runtime"] = {"torch": "new host"}
    assert c.validate_plans(plan, target, method) == []


@pytest.mark.parametrize("key", ["profile", "sources", "poison", "action_sha256"])
def test_source_plan_identity_fields_cannot_change(plan, key):
    target = copy.deepcopy(plan)
    target[key] = "changed"
    with pytest.raises(ValueError, match=key):
        c.validate_plans(plan, target, "ftrojan")


@pytest.mark.parametrize("key,value", [("seed", 43), ("epochs", 201), ("batch_size", 16), ("amp", True)])
def test_scientific_recipe_mutations_refused(plan, key, value):
    target = copy.deepcopy(plan)
    target["cfg"][key] = value
    with pytest.raises(ValueError, match="Scientific configuration differs"):
        c.validate_plans(plan, target, "ftrojan")


def test_original_plan_signature_checked(plan):
    plan["cfg"]["seed"] = 43
    with pytest.raises(ValueError, match="Source plan digest mismatch"):
        c.validate_plans(plan, copy.deepcopy(plan), "ftrojan")


def test_dataset_list_content_and_order_cannot_change(plan):
    target = copy.deepcopy(plan)
    target["inputs"]["splits"]["train_data"]["ordered_names"].reverse()
    with pytest.raises(ValueError, match="dataset/list/content/order"):
        c.validate_plans(plan, target, "fiba")


@pytest.mark.parametrize("method", c.NATIVE)
def test_native_only_unused_a1_array_hash_difference_explicitly_recorded(plan, method):
    before = copy.deepcopy(plan)
    target = copy.deepcopy(plan)
    target["trigger"]["arrays"]["pattern"]["sha256"] = "d" * 64
    diffs = c.validate_plans(plan, target, method)
    assert diffs == [dict(array="pattern", previous="c" * 64, current="d" * 64)]
    assert plan == before
    assert target["trigger"]["arrays"]["pattern"]["sha256"] == "d" * 64


@pytest.mark.parametrize("method", ("clean", "a1"))
def test_clean_a1_array_hashes_never_relaxed(plan, method):
    target = copy.deepcopy(plan)
    target["trigger"]["arrays"]["pattern"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="Actual A1 trigger/protocol changed"):
        c.validate_plans(plan, target, method)


@pytest.mark.parametrize("field,value", [("dtype", "<f4"), ("shape", [3, 30, 21])])
def test_unused_a1_shape_dtype_not_relaxed(plan, field, value):
    target = copy.deepcopy(plan)
    target["trigger"]["arrays"]["pattern"][field] = value
    with pytest.raises(ValueError, match="shape/dtype"):
        c.validate_plans(plan, target, "ftrojan")


def test_native_a1_protocol_target_and_action_never_relaxed(plan):
    target = copy.deepcopy(plan)
    target["trigger"]["protocol"]["dose_max"] = .9
    with pytest.raises(ValueError, match="Actual A1 trigger/protocol changed"):
        c.validate_plans(plan, target, "bpp_source")


def test_exact_linux_ft_pattern_bitwise_shape_dtype_and_independent_copies():
    value = c.frozen_pattern()
    assert value.shape == (3, 30, 20)
    assert value.dtype == np.dtype("<f8")
    assert np.isfinite(value).all()
    assert hashlib.sha256(value.tobytes()).hexdigest() == c.FT_PATTERN_SHA
    value[:] = 0
    assert hashlib.sha256(c.frozen_pattern().tobytes()).hexdigest() == c.FT_PATTERN_SHA


@pytest.mark.parametrize("raw", [b"zero", bytes(14400), bytes(14408)], ids=["short", "wrong-digest", "wrong-size"])
def test_ft_wrong_pattern_never_falls_back(tmp_path, monkeypatch, raw):
    (tmp_path / "ftrojan_pattern_linux.f8.zlib").write_bytes(zlib.compress(raw))
    monkeypatch.setattr(c, "HERE", tmp_path)
    with pytest.raises(ValueError, match="checksum differs"):
        c.frozen_pattern()


def test_canonical_donor_path_maps_to_audited_local_files_only():
    class Dataset:
        split = "training"
        items = [dict(name="a", csi="C:/data/a.npy", kpt="C:/pose/a.npy"),
                 dict(name="b", csi="C:/data/b.npy", kpt="C:/pose/b.npy")]
        def __len__(self): return len(self.items)
        def load_raw(self, path): return ("loaded", path)
        def normalize(self, value): return ("normalized", value)
    actual = Dataset()
    canonical = c.CanonicalTrain(actual, "/workspace/original", ["a", "b"])
    assert len(canonical) == 2
    assert canonical.items[0]["csi"] == "/workspace/original/train_data/csi_ap/a.npy"
    assert canonical.load_raw(canonical.items[0]["csi"]) == ("loaded", "C:/data/a.npy")
    assert canonical.items[1]["kpt"] == "/workspace/original/train_data/keypoint/b.npy"
    assert canonical.normalize(1) == ("normalized", 1)
    assert actual.items[0]["csi"] == "C:/data/a.npy"
    with pytest.raises(ValueError, match="audited TRAIN"):
        canonical.load_raw("/workspace/original/test_data/csi_ap/a.npy")
    with pytest.raises(ValueError, match="exact ordered TRAIN"):
        c.CanonicalTrain(actual, "/workspace/original", ["b", "a"])
    actual.split = "validation"
    with pytest.raises(ValueError, match="exact ordered TRAIN"):
        c.CanonicalTrain(actual, "/workspace/original", ["a", "b"])


@pytest.mark.parametrize("method", ["bpp_source", "fiba"])
def test_real_grouped_native_donor_keys_and_stochastic_resume_preserved(method, monkeypatch):
    repo = HERE.parents[1]
    monkeypatch.syspath_prepend(str(repo / "ATKBackd"))
    from piw_native_attack_adapter import GroupedTrigger
    source = c.original_plan()
    for name in ("piw_native_attack_adapter.py", "attack/static_source_baselines.py", "attack/frequency_baselines.py"):
        assert c.sha(repo / "ATKBackd" / name) == source["sources"][name]
    rng = np.random.default_rng(42)
    arrays = {name: rng.uniform(.05, .95, (3, 180, 20)).astype(np.float32) for name in ("a", "b", "c")}
    class Dataset:
        split = "training"
        def __init__(self, root):
            self.items = [dict(name=name, csi=f"{root}/train_data/csi_ap/{name}.npy",
                              kpt=f"{root}/train_data/keypoint/{name}.npy") for name in arrays]
        def __len__(self): return len(self.items)
        def load_raw(self, path): return arrays[str(path).replace("\\", "/").rsplit("/", 1)[-1][:-4]].copy()
        @staticmethod
        def normalize(value): return value
    linux_root = "/workspace/original"
    original = GroupedTrigger(method, Dataset(linux_root))
    actual = Dataset("C:/hostdata")
    canonical = c.CanonicalTrain(actual, linux_root, list(arrays))
    resumed = GroupedTrigger(method, canonical)
    assert original.fixed_keys() == resumed.fixed_keys()
    assert original.fixed_keys() != GroupedTrigger(method, actual).fixed_keys()
    value = arrays["a"]
    for _ in range(3):
        assert np.array_equal(original.noise_inject(value), resumed.noise_inject(value))
    state = copy.deepcopy(original.state_dict())
    restored = GroupedTrigger(method, canonical)
    restored.load_state_dict(state)
    for _ in range(3):
        assert np.array_equal(original.noise_inject(value), restored.noise_inject(value))
    assert np.array_equal(restored.inject(value, .7)[:, 90:, :], value[:, 90:, :])
    assert np.array_equal(restored.inject(value, 0), value)


def test_valid_original_adamw_full_resume_state(plan):
    blob = blob_for(plan)
    c.validate_blob(blob, plan, "ftrojan", blob["poison"])
    c.validate_blob(blob_for(plan, epoch=199), plan, "ftrojan")


@pytest.mark.parametrize("epoch", [-1, 200, True, 84.0, "84"])
def test_invalid_resume_epoch_refused(plan, epoch):
    with pytest.raises(ValueError, match="Foreign or invalid"):
        c.validate_blob(blob_for(plan, epoch=epoch), plan, "ftrojan")


@pytest.mark.parametrize("field", ["model", "optimizer", "rng", "execution", "poison"])
def test_missing_complete_resume_state_refused(plan, field):
    blob = blob_for(plan)
    blob[field] = {}
    with pytest.raises(ValueError, match="complete continuation state"):
        c.validate_blob(blob, plan, "ftrojan")


@pytest.mark.parametrize("field,value", [("lr", .001), ("weight_decay", 0)])
def test_sgd_or_other_recipe_state_cannot_be_imported(plan, field, value):
    blob = blob_for(plan)
    blob["optimizer"]["param_groups"][0][field] = value
    with pytest.raises(ValueError, match="AdamW"):
        c.validate_blob(blob, plan, "ftrojan")


def test_sgd_param_groups_and_missing_adamw_moments_refused(plan):
    blob = blob_for(plan)
    del blob["optimizer"]["param_groups"][0]["betas"]
    with pytest.raises(ValueError, match="AdamW"):
        c.validate_blob(blob, plan, "ftrojan")
    blob = blob_for(plan)
    blob["optimizer"]["state"] = {}
    with pytest.raises(ValueError, match="Optimizer state missing"):
        c.validate_blob(blob, plan, "ftrojan")


@pytest.mark.parametrize("field,value", [("betas", (.5, .9)), ("eps", 1e-6), ("amsgrad", True), ("maximize", True)])
def test_adamw_scalar_defaults_are_frozen(plan, field, value):
    blob = blob_for(plan)
    blob["optimizer"]["param_groups"][0][field] = value
    with pytest.raises(ValueError, match="original AdamW"):
        c.validate_blob(blob, plan, "ftrojan")


@pytest.mark.parametrize("field", ["step", "exp_avg", "exp_avg_sq"])
def test_every_adamw_moment_and_step_is_required(plan, field):
    blob = blob_for(plan)
    state = next(iter(blob["optimizer"]["state"].values()))
    del state[field]
    with pytest.raises(ValueError, match="moment/step state"):
        c.validate_blob(blob, plan, "ftrojan")


@pytest.mark.parametrize("missing", ["python", "numpy", "torch", "cuda"])
def test_full_rng_required(plan, missing):
    blob = blob_for(plan)
    del blob["rng"][missing]
    with pytest.raises(ValueError, match="CPU/CUDA RNG"):
        c.validate_blob(blob, plan, "ftrojan")


def test_native_poison_cover_operator_keys_and_three_states_required(plan):
    blob = blob_for(plan)
    with pytest.raises(ValueError, match="operator/poison/cover"):
        c.validate_blob(blob, plan, "ftrojan", {"changed": True})
    blob["trigger"] = [None, None]
    with pytest.raises(ValueError, match="three-group"):
        c.validate_blob(blob, plan, "ftrojan")


def test_candidate_discovery_known_layouts_only_and_excludes_new_output(args):
    exact = args.checkpoint_home / "metafiplusplus/seed42/ftrojan/checkpoint.pt"
    old = args.data_home / "runs/old/attacks/metafiplusplus/seed42/ftrojan/checkpoint.pt"
    wrong = args.data_home / "runs/old/attacks/wiflow/seed42/ftrojan/checkpoint.pt"
    for path in (exact, old, wrong):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"cp")
    assert set(c.candidate_paths(args, "ftrojan")) == {exact.resolve(), old.resolve()}


def test_legacy_host_zip_backup_is_discovered(args):
    path = args.data_home / "code/legacy/backdoor-analog/recovery_inputs/ftrojan_epoch84.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"backup")
    assert c.candidate_paths(args, "ftrojan") == [path.resolve()]


def test_missing_backup_methods_refuse_silent_epoch0(args, plan, monkeypatch):
    monkeypatch.setattr(c, "candidate_paths", lambda *_: [])
    for method in c.BACKUPS:
        with pytest.raises(ValueError, match="not restart epoch0"):
            c.select_candidate(args, method, plan, {})
    for method in ("clean", "a1", "wanet_source"):
        assert c.select_candidate(args, method, plan, {})["epoch"] == -1


def test_candidate_highest_saved_epoch_not_score_and_sgd_is_skipped(args, plan, monkeypatch):
    args.checkpoint_home.mkdir(parents=True)
    paths = [args.checkpoint_home / name for name in ("p10.pt", "p199.pt", "p84.pt", "sgd.pt")]
    for path, epoch in zip(paths, (10, 199, 84, 199)):
        torch.save(blob_for(plan, epoch=epoch), path)
    sgd = copy.deepcopy(plan)
    sgd["cfg"]["recipes"]["metafiplusplus"]["optimizer"] = "sgd"
    signed(sgd)
    monkeypatch.setattr(c, "candidate_paths", lambda *_: paths)
    monkeypatch.setattr(c, "candidate_plan", lambda p, _: sgd if p.name == "sgd.pt" else plan)
    selected = c.select_candidate(args, "ftrojan", plan, blob_for(plan)["poison"])
    assert selected["path"] == str(paths[1])
    assert selected["epoch"] == 199
    assert selected["sha256"] == c.sha(paths[1])
    assert selected["ignored_non_adamw"][0]["path"] == str(paths[-1])


def test_candidate_rejects_foreign_adamw_instead_of_restart(args, plan, monkeypatch):
    path = args.out / "checkpoint.pt"; path.parent.mkdir()
    broken = blob_for(plan); broken["method"] = "bpp_source"
    torch.save(broken, path)
    monkeypatch.setattr(c, "candidate_paths", lambda *_: [path])
    monkeypatch.setattr(c, "candidate_plan", lambda *_: plan)
    with pytest.raises(ValueError, match="Foreign or invalid"):
        c.select_candidate(args, "ftrojan", plan, {})


def test_planless_sgd_is_skipped_but_unknown_adamw_is_not_imported(args, plan, monkeypatch):
    path = args.out / "checkpoint.pt"; path.parent.mkdir()
    model = torch.nn.Linear(2, 1)
    torch.save({"optimizer": torch.optim.SGD(model.parameters(), lr=.01, momentum=.9).state_dict()}, path)
    monkeypatch.setattr(c, "candidate_paths", lambda *_: [path])
    def missing(*_): raise ValueError("Missing source plan")
    monkeypatch.setattr(c, "candidate_plan", missing)
    selected = c.select_candidate(args, "clean", plan, {})
    assert selected["epoch"] == -1
    assert "SGD checkpoint without plan" in selected["ignored_non_adamw"][0]["reason"]
    torch.save(blob_for(plan), path)
    with pytest.raises(ValueError, match="Missing source plan"):
        c.select_candidate(args, "clean", plan, {})


def test_other_experiment_profiles_not_relabelled_as_current(args, plan, monkeypatch):
    other = copy.deepcopy(plan); other["profile"] = "old SGD or unrelated AdamW study"; signed(other)
    path = args.out / "checkpoint.pt"
    monkeypatch.setattr(c, "candidate_paths", lambda *_: [path])
    monkeypatch.setattr(c, "candidate_plan", lambda *_: other)
    selected = c.select_candidate(args, "clean", plan, {})
    assert selected["epoch"] == -1
    assert selected["ignored_non_adamw"][0]["reason"] == "other experiment profile; never imported"


def test_sample_batch_contains_actual_poison_cover_and_native_physical32():
    data = SimpleNamespace()
    class Data:
        covers = {8, 9}
        def __len__(self): return 80
        def __getitem__(self, index): return {"index": index}
    saved = {"poison": {"samples": [{"index": i} for i in range(7)]}}
    values = c.sample_batch(None, saved, "fiba", Data())["index"].tolist()
    assert len(values) == 32
    assert values[:5] == [0, 1, 2, 3, 8]
    assert len(set(values)) == 32
    assert not {4, 5, 6} & set(values)
    clean = c.sample_batch(None, saved, "clean", Data())["index"].tolist()
    assert clean == list(range(32))


def test_gpu_desktop_filter_preserves_unknown_or_other_compute_conflicts():
    text = '4, C:\\Windows\\explorer.exe\n5, C:\\env\\python.exe\n6, C:\\Windows\\dwm.exe\n'
    assert c.conflict_rows(text, windows=True) == [(5, "C:\\env\\python.exe")]
    assert len(c.conflict_rows(text, windows=False)) == 3
    assert c.conflict_rows("5, python.exe\n", own_pid=5) == []
    with pytest.raises(ValueError, match="Unidentified"):
        c.conflict_rows("unknown\n")


def test_lock_rejects_overlap_and_releases_on_error(tmp_path):
    with pytest.raises(RuntimeError, match="release"):
        with c.lock(tmp_path):
            with pytest.raises(OSError):
                with c.lock(tmp_path):
                    pytest.fail("overlapping controller acquired lock")
            raise RuntimeError("release")
    with c.lock(tmp_path):
        pass


def cli(args, mode):
    result = ["--mode", mode]
    for key in ("data_home", "dataset", "action", "out", "checkpoint_home"):
        result += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    return result


def setup_controller_mock(args, monkeypatch):
    monkeypatch.setattr(c, "sha", lambda _: c.ACTION_SHA)
    monkeypatch.setattr(c, "code_binding", lambda: {"code": "test"})
    monkeypatch.setattr(c, "require_idle", lambda: None)
    monkeypatch.setattr(c.shutil, "disk_usage", lambda _: SimpleNamespace(free=8 * 2**30))
    monkeypatch.setattr(c, "configure_torch_cache", lambda _: None)


def test_inspect_status_never_create_outputs_or_audit_or_load_gpu(args, monkeypatch, capsys):
    monkeypatch.setattr(c, "sha", lambda *_: pytest.fail("read-only inspect audited inputs"))
    monkeypatch.setattr(c, "require_idle", lambda: pytest.fail("read-only inspect queried GPU"))
    assert c.main(cli(args, "inspect")) == 0
    assert c.main(cli(args, "status")) == 0
    assert not args.out.exists()
    assert "serial 1 job/cuda:0" in capsys.readouterr().out


def test_serial_launch_checks_all_then_trains_all_then_evaluates_no_overlap(args, monkeypatch):
    setup_controller_mock(args, monkeypatch)
    events = []
    def run(command, **kwargs):
        events.append((command[command.index("--method") + 1], command[command.index("--work") + 1]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.subprocess, "run", run)
    monkeypatch.setattr(c, "summarize", lambda _: events.append(("summary", "summary")))
    assert c.main(cli(args, "launch")) == 0
    assert events == [(m, phase) for phase in ("check", "train", "evaluate") for m in c.METHODS] + [("summary", "summary")]


def test_queue_failure_stops_later_jobs_and_never_evaluates(args, monkeypatch):
    setup_controller_mock(args, monkeypatch)
    events = []
    def run(command, **kwargs):
        event = (command[command.index("--method") + 1], command[command.index("--work") + 1])
        events.append(event)
        return SimpleNamespace(returncode=7 if event == ("ftrojan", "train") else 0)
    monkeypatch.setattr(c.subprocess, "run", run)
    monkeypatch.setattr(c, "summarize", lambda _: pytest.fail("summarized failed queue"))
    with pytest.raises(RuntimeError, match="no retry/later jobs"):
        c.main(cli(args, "launch"))
    assert events[-1] == ("ftrojan", "train")
    assert not any(phase == "evaluate" for _, phase in events)
    assert [m for m, phase in events if phase == "train"] == ["clean", "a1", "ftrojan"]


def test_prepare_never_launches_full_training(args, monkeypatch):
    setup_controller_mock(args, monkeypatch)
    events = []
    def run(command, **kwargs):
        events.append(command[command.index("--work") + 1])
        method = command[command.index("--method") + 1]
        c.write(c.gate_path(args, method), {"checkpoint_selection": {"path": None, "epoch": -1}})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.subprocess, "run", run)
    assert c.main(cli(args, "prepare")) == 0
    assert events == ["prepare"] * 6
    scope = c.read(args.out / "execution_scope.json")
    assert scope["selected_methods"] == list(c.METHODS)
    assert scope["excluded_methods"] == ["tsba_adapted", "smooth_lf_csi"]
    assert scope["jobs_per_gpu"] == 1
    assert scope["final_epoch"] == 199


def test_changed_prelaunch_gate_blocks_training(args, plan, monkeypatch):
    cfg = {"outdir": str(args.out)}
    execution = {"gpu_name": "RTX 3090"}
    w = SimpleNamespace(cell_dir=lambda *_: args.out / "cell")
    monkeypatch.setattr(c, "code_binding", lambda: {"code": "new"})
    gate = dict(profile=c.PROFILE, method=args.method, plan_sha256=plan["plan_sha256"],
                execution=execution, input_stat_fingerprint="stats", code={"code": "old"},
                physical_batch=32, prepared_only=True)
    c.write(c.gate_path(args, args.method), gate)
    with pytest.raises(ValueError, match="prelaunch gate changed"):
        c.check_gate(args, w, cfg, plan, execution, "stats")


def test_deleted_migrated_checkpoint_cannot_be_restarted_from_epoch0(args, plan, monkeypatch):
    execution = {"gpu_name": "RTX 3090"}
    w = SimpleNamespace(cell_dir=lambda *_: args.out / "cell")
    monkeypatch.setattr(c, "code_binding", lambda: {})
    gate = dict(profile=c.PROFILE, method=args.method, plan_sha256=plan["plan_sha256"],
        execution=execution, input_stat_fingerprint="stats", code={}, physical_batch=32,
        prepared_only=True, checkpoint_selection={"path": "original.pt"},queue_scope=c.queue_scope(args))
    c.write(c.gate_path(args, args.method), gate)
    with pytest.raises(ValueError, match="refusing an epoch0 restart"):
        c.check_gate(args, w, {}, plan, execution, "stats")


def test_prepare_migration_preserves_model_optimizer_rng_trigger_after_mutating_smoke(args, plan, monkeypatch):
    original = blob_for(plan)
    source_path = args.out / "source.pt"; source_path.parent.mkdir()
    torch.save(original, source_path)
    source_sha = c.sha(source_path)
    target_plan = copy.deepcopy(plan)
    target_plan["runtime"] = {"torch": "new host"}
    signed(target_plan)
    selection = dict(path=str(source_path), epoch=84, sha256=source_sha, source_plan=plan,
                     unused_a1_metadata_differences=[])
    folder = args.out / "cell"
    trigger = SimpleNamespace(load_state_dict=lambda _: None)
    def smoke(model, optimizer, *unused, **kwargs):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(32, 2)).square().mean(); loss.backward(); optimizer.step()
        return float(loss.detach()), 32, {}
    def save(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path)
    w = SimpleNamespace(cell_dir=lambda *_: folder, _datasets=lambda _: {"training": object()},
        make_trigger=lambda *_: trigger, make_training_data=lambda *_: (object(), original["poison"]),
        _seed=lambda _: None, _build=lambda *_: torch.nn.Linear(2, 1),
        _optimizer=lambda model, _: torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01),
        _train_batch=smoke, base=SimpleNamespace(_restore_rng=lambda *_: None, atomic_checkpoint=save))
    monkeypatch.setattr(c, "select_candidate", lambda *_: copy.deepcopy(selection))
    monkeypatch.setattr(c, "sample_batch", lambda *_: {"unused": 1})
    monkeypatch.setattr(c, "code_binding", lambda: {})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    execution = {"gpu_name": "RTX 3090", "device": "cuda:0"}
    c.prepare_cell(args, w, {}, target_plan, execution, "stats")
    migrated = torch.load(folder / "checkpoint.pt", weights_only=False)
    assert migrated["epoch"] == original["epoch"] == 84
    assert migrated["execution"] == execution
    assert migrated["plan_sha256"] == target_plan["plan_sha256"]
    assert c.sha(source_path) == source_sha
    for name, tensor in original["model"].items():
        assert torch.equal(tensor, migrated["model"][name])
    for pid, state in original["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, migrated["optimizer"]["state"][pid][name])
    assert torch.equal(original["rng"]["torch"], migrated["rng"]["torch"])
    assert torch.equal(original["rng"]["cuda"][0], migrated["rng"]["cuda"][0])
    assert original["rng"]["python"] == migrated["rng"]["python"]
    assert np.array_equal(original["rng"]["numpy"][1], migrated["rng"]["numpy"][1])
    assert original["trigger"] == migrated["trigger"]
    receipt = c.read(folder / "migration_receipt.json")
    assert receipt["next_epoch"] == 85
    assert receipt["model_optimizer_rng_trigger_unchanged"] is True
    assert receipt["no_bitwise_cross_platform_trajectory_guarantee"] is True
    assert c.read(c.gate_path(args, args.method))["prepared_only"] is True


def test_four_attack_scope_does_not_modify_the_original_scientific_scope(args):
    args.four_attacks_only = True
    assert c.selected_methods(args) == ('ftrojan','bpp_source','fiba','wanet_source')
    assert c.queue_scope(args)['training_only'] is True
    assert c.METHODS == ('clean','a1','ftrojan','bpp_source','fiba','wanet_source')
    cmd = c.command(args,'ftrojan','train')
    assert '--four-attacks-only' in cmd
    assert cmd.count('--four-attacks-only') == 1


def test_four_attack_prepare_only_schedules_four_smokes(args,monkeypatch):
    setup_controller_mock(args,monkeypatch)
    events = []
    def run(command,**kwargs):
        method = command[command.index('--method')+1]
        work = command[command.index('--work')+1]
        events.append((method,work))
        assert '--four-attacks-only' in command
        c.write(c.gate_path(args,method),{'checkpoint_selection':{'epoch':84,'path':'backup'}})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.subprocess,'run',run)
    assert c.main(cli(args,'prepare')+['--four-attacks-only']) == 0
    assert events == [(m,'prepare') for m in c.NATIVE]
    scope = c.read(args.out/'execution_scope.json')
    assert scope['selected_methods'] == list(c.NATIVE)
    assert scope['excluded_methods'] == ['clean','a1','tsba_adapted','smooth_lf_csi']
    assert scope['queue_scope']['training_only'] is True


def test_four_attack_queue_has_no_clean_a1_evaluate_or_retry(args,monkeypatch,capsys):
    setup_controller_mock(args,monkeypatch)
    events = []
    def run(command,**kwargs):
        assert '--four-attacks-only' in command
        events.append((command[command.index('--method')+1],command[command.index('--work')+1]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.subprocess,'run',run)
    monkeypatch.setattr(c,'summarize',lambda _: pytest.fail('No comparison without Clean reference'))
    monkeypatch.setattr(c,'summarize_training',lambda _: events.append(('summary','trained_only')))
    assert c.main(cli(args,'launch')+['--four-attacks-only']) == 0
    assert events == [(m,phase) for phase in ('check','train') for m in c.NATIVE]+[('summary','trained_only')]
    out = capsys.readouterr().out
    assert 'ALL_FOUR_TRAINED_NOT_EVALUATED' in out
    assert 'ALL_SIX_COMPLETE' not in out


def test_four_attack_queue_failure_stops_pending_without_retry(args,monkeypatch):
    setup_controller_mock(args,monkeypatch)
    events = []
    def run(command,**kwargs):
        event = (command[command.index('--method')+1],command[command.index('--work')+1])
        events.append(event)
        return SimpleNamespace(returncode=1 if event == ('bpp_source','train') else 0)
    monkeypatch.setattr(c.subprocess,'run',run)
    monkeypatch.setattr(c,'summarize_training',lambda _: pytest.fail('Failed queue cannot finish'))
    with pytest.raises(RuntimeError,match='no retry/later jobs'):
        c.main(cli(args,'launch')+['--four-attacks-only'])
    assert [m for m,phase in events if phase == 'train'] == ['ftrojan','bpp_source']


@pytest.mark.parametrize('method,work',[('clean','train'),('a1','prepare'),('ftrojan','evaluate')])
def test_four_attack_internal_cell_rejects_expanded_scope(args,method,work):
    with pytest.raises(SystemExit):
        c.main(cli(args,'cell')+['--four-attacks-only','--method',method,'--work',work])


def test_scope_change_invalidates_existing_cell_gate(args,plan,monkeypatch):
    monkeypatch.setattr(c,'code_binding',lambda: {})
    execution = {'gpu_name':'RTX 3090'}
    gate = dict(profile=c.PROFILE,method=args.method,plan_sha256=plan['plan_sha256'],
        execution=execution,input_stat_fingerprint='stats',code={},physical_batch=32,
        prepared_only=True,queue_scope=c.queue_scope(args))
    c.write(c.gate_path(args,args.method),gate)
    args.four_attacks_only = True
    with pytest.raises(ValueError,match='prelaunch gate changed'):
        c.check_gate(args,SimpleNamespace(),{},plan,execution,'stats')


def test_training_summary_binds_four_final_checkpoints_without_fake_metrics(args):
    args.four_attacks_only = True
    for method in c.NATIVE:
        folder = c.run_out(args,method)/'attacks/metafiplusplus/seed42'/method
        folder.mkdir(parents=True)
        checkpoint = folder/'checkpoint.pt'
        checkpoint.write_bytes(method.encode())
        c.write(folder/'trained_final199.json',dict(profile=c.PROFILE,status='TRAINED_NOT_EVALUATED',
            epoch=199,checkpoint_sha256=c.sha(checkpoint),plan_sha256='plan'))
    c.summarize_training(args)
    result = c.read(args.out/'training_results.json')
    assert result['status'] == 'TRAINED_NOT_EVALUATED'
    assert [r['method'] for r in result['rows']] == list(c.NATIVE)
    assert all('metrics' not in r for r in result['rows'])
    checkpoint.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='receipt mismatch'):
        c.summarize_training(args)
