"""Shared-4090 execution guards, without CIM, CUDA, datasets or training."""
from __future__ import annotations

import copy
import ast
import csv
import hashlib
import importlib.util
import ntpath
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "remote_tools" / "shared_4090"


def load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def context():
    return load_file(SHARED / "trojan_context.py", "shared4090_context_test")


@pytest.fixture
def nscaling():
    """Two observed Windows venv/real-Python records, not two GPU jobs."""
    executable = r"C:\Users\Van\AppData\Local\Python\pythoncore-3.12-64\python.exe"
    proxy = r"U:\ijoc\.venv\Scripts\python.exe"
    tail = " -u scripts\\nscaling.py LandS,gbd,storm,20term,ssn S1,S3 5"
    return [
        dict(ProcessId=26344, ParentProcessId=44248, Name="python.exe",
             ExecutablePath=executable, CommandLine='"' + executable + '"' + tail,
             CreationDate="2026-10-29T18:01:00.0000000+07:00"),
        dict(ProcessId=44248, ParentProcessId=26844, Name="python.exe",
             ExecutablePath=proxy, CommandLine=proxy + tail,
             CreationDate="2026-10-29T18:00:59.0000000+07:00"),
    ]


def test_frozen_scientific_runner_and_operator_are_unchanged():
    expected = {
        "ATKBackd/run_mmfi_p1s1_trojannn.py":
            "0c335047a4869d0a354cf184ffb41ede04fd9e324452913372d31bfa9f565318",
        "ATKBackd/attack/trojannn_mmfi.py":
            "81df9011814de7443d4f02068b88e37790f719afb9f2ed39f4ced947df7537d5",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_exact_observed_nscaling_pair_is_allowed(context, nscaling):
    assert context.validate_nscaling(copy.deepcopy(nscaling), nscaling) == {26344, 44248}


@pytest.mark.parametrize("present, expected", [([], set()), ([1], {44248})])
def test_finished_approved_worker_does_not_block(context, nscaling, present, expected):
    records = [copy.deepcopy(nscaling[i]) for i in present]
    assert context.validate_nscaling(records, nscaling) == expected


def test_live_child_requires_observed_parent(context, nscaling):
    with pytest.raises(ValueError):
        context.validate_nscaling([copy.deepcopy(nscaling[0])], nscaling)


@pytest.mark.parametrize("key,value", [
    ("CreationDate", "2026-10-30T00:00:00Z"),
    ("Name", "pythonw.exe"),
    ("ParentProcessId", 123),
    ("ExecutablePath", r"C:\Other\python.exe"),
    ("CommandLine", 'python train.py'),
    ("CommandLine", 'python -u scripts\\nscaling.py LandS,gbd,storm,20term,ssn S1,S3 5 --extra'),
])
def test_pid_reuse_or_changed_identity_fails_closed(context, nscaling, key, value):
    records = copy.deepcopy(nscaling)
    records[0][key] = value
    with pytest.raises(ValueError):
        context.validate_nscaling(records, nscaling)


@pytest.mark.parametrize("key", ["ProcessId", "ParentProcessId", "Name", "ExecutablePath",
                                 "CommandLine", "CreationDate"])
def test_missing_live_process_metadata_is_rejected(context, nscaling, key):
    records = copy.deepcopy(nscaling)
    del records[0][key]
    with pytest.raises(ValueError):
        context.validate_nscaling(records, nscaling)


@pytest.mark.parametrize("key", ["ProcessId", "ParentProcessId", "Name", "ExecutablePath",
                                 "CommandLine", "CreationDate"])
def test_missing_approval_metadata_is_rejected(context, nscaling, key):
    approved = copy.deepcopy(nscaling)
    del approved[0][key]
    with pytest.raises(ValueError):
        context.validate_nscaling(nscaling, approved)


@pytest.mark.parametrize("old,new", [
    ("scripts\\nscaling.py", "scripts\\eval_offsets.py"),
    ("LandS,gbd,storm,20term,ssn", "LandS"),
    ("S1,S3", "S3"),
    (" 5", " 6"),
])
def test_cannot_approve_different_script_or_scientific_arguments(context, nscaling, old, new):
    approved = copy.deepcopy(nscaling)
    for record in approved:
        record["CommandLine"] = record["CommandLine"].replace(old, new)
    with pytest.raises(ValueError):
        context.validate_nscaling(approved, approved)


def test_unapproved_relaunch_is_not_inherited_by_pid_or_script_name(context, nscaling):
    extra = copy.deepcopy(nscaling[0])
    extra.update(ProcessId=99999, CreationDate="2026-10-30T00:00:00Z")
    assert context.validate_nscaling(nscaling + [extra], nscaling) == {26344, 44248}


def test_installed_guard_filters_only_exact_approved_conflict(context, nscaling, monkeypatch):
    calls = []
    def original(report, windows_names=None, own_pid=None, *, plan=None, windows_processes=None):
        calls.append((report, windows_names, own_pid, plan, windows_processes))
        return [(26344, "python.exe"), (777, "python.exe"), (778, "[Insufficient Permissions]")]
    pending = object()
    run = SimpleNamespace(classify_gpu=original, pending_experiment_conflicts=pending)
    monkeypatch.setattr(context, "windows_processes", lambda: copy.deepcopy(nscaling))
    context.install_guard(run, nscaling)
    names = {26344: "python.exe", 777: "python.exe"}
    plan, records = {"fixed": "recipe"}, [{"unmodified": "snapshot"}]
    conflicts = run.classify_gpu("report", names, 99, plan=plan, windows_processes=records)
    assert conflicts == [(777, "python.exe"), (778, "[Insufficient Permissions]")]
    assert calls == [("report", names, 99, plan, records)]
    assert run.pending_experiment_conflicts is pending


def test_installed_guard_revalidates_identity_each_call(context, nscaling, monkeypatch):
    records = copy.deepcopy(nscaling)
    run = SimpleNamespace(classify_gpu=lambda *a, **k: [(26344, "python.exe")])
    monkeypatch.setattr(context, "windows_processes", lambda: records)
    context.install_guard(run, nscaling)
    assert run.classify_gpu("first") == []
    records[0]["CreationDate"] = "2026-10-30T01:00:00Z"
    with pytest.raises(ValueError):
        run.classify_gpu("second")


def test_finished_nscaling_does_not_allow_other_python_or_sig(context, nscaling, monkeypatch):
    run = SimpleNamespace(classify_gpu=lambda *a, **k: [(777, "python.exe"), (778, "sig_source")])
    monkeypatch.setattr(context, "windows_processes", lambda: [])
    context.install_guard(run, nscaling)
    assert run.classify_gpu("report") == [(777, "python.exe"), (778, "sig_source")]


def test_no_nscaling_approval_confers_no_compute_permission(context, monkeypatch):
    assert context.validate_nscaling([], []) == set()
    run = SimpleNamespace(classify_gpu=lambda *a, **k: [(26344, "python.exe")])
    monkeypatch.setattr(context, "windows_processes", lambda: [])
    context.install_guard(run, [])
    assert run.classify_gpu("report") == [(26344, "python.exe")]


def test_duplicate_identity_and_duplicate_install_are_rejected(context, nscaling, monkeypatch):
    with pytest.raises(ValueError, match="Duplicate"):
        context.validate_nscaling(nscaling + [copy.deepcopy(nscaling[0])], nscaling)
    run = SimpleNamespace(classify_gpu=lambda *a, **k: [])
    monkeypatch.setattr(context, "windows_processes", lambda: nscaling)
    context.install_guard(run, nscaling)
    with pytest.raises(ValueError, match="already installed"):
        context.install_guard(run, nscaling)


def test_original_gpu_parse_failure_is_not_swallowed(context, nscaling, monkeypatch):
    def original(*a, **k):
        raise ValueError("Cannot parse GPU process")
    run = SimpleNamespace(classify_gpu=original)
    monkeypatch.setattr(context, "windows_processes", lambda: nscaling)
    context.install_guard(run, nscaling)
    with pytest.raises(ValueError, match="Cannot parse GPU process"):
        run.classify_gpu("malformed")


def original_gpu_guard():
    """Load actual stdlib-only guard functions, not a model or training runner."""
    source_path = ROOT / "ATKBackd/run_mmfi_p1s1_trojannn.py"
    functions = {"parse_windows_commandline", "_windows_path", "_a1_environments",
                 "a1_peer", "authorized_a1_peers", "classify_gpu"}
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
    namespace = dict(csv=csv, ntpath=ntpath, re=re)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in functions})


@pytest.fixture
def a1_peer():
    home = r"C:\Users\Van\backdooranalog"
    python = home + r"\envs\dose-backdoor\python.exe"
    script = r"C:\Users\Van\code\ATKBackd\run_mmfi_p3_splits.py"
    output = home + r"\runs\mmfi_p3s3_hpeli_a1_s42_4090_with_sig_v1"
    args = [python, "-u", script, "--data-home", home, "--outdir", output,
            "--device", "cuda:0", "--workers", "0", "--jobs-per-gpu", "1",
            "--mode", "launch", "--_cell", "p3_s3_a1"]
    return dict(ProcessId=12345, ParentProcessId=12340, Name="python.exe",
                ExecutablePath=python, CommandLine=" ".join('"' + x + '"' for x in args),
                CreationDate="2026-10-29T18:00:00Z")


def test_original_exact_a1_allowance_is_preserved(context, nscaling, a1_peer, monkeypatch):
    run = original_gpu_guard()
    monkeypatch.setattr(context, "windows_processes", lambda: nscaling + [a1_peer])
    context.install_guard(run, nscaling)
    plan = dict(home=r"C:\Users\Van\backdooranalog",
                execution_policy=dict(mode="trojannn_plus_exact_p3s3_a1"))
    assert run.classify_gpu("26344, python.exe\n12345, python.exe", {
        26344: "python.exe", 12345: "python.exe"}, 999, plan=plan,
        windows_processes=nscaling + [a1_peer]) == []


@pytest.mark.parametrize("old,new", [("p3_s3_a1", "p3_s3_clean"),
                                    ("p3_s3_a1", "sig_source"),
                                    ('"--workers" "0"', '"--workers" "2"'),
                                    ('"--device" "cuda:0"', '"--device" "cuda:1"')])
def test_narrow_nscaling_exception_never_weakens_a1_identity(
        context, nscaling, a1_peer, monkeypatch, old, new):
    a1_peer["CommandLine"] = a1_peer["CommandLine"].replace(old, new)
    run = original_gpu_guard()
    monkeypatch.setattr(context, "windows_processes", lambda: nscaling + [a1_peer])
    context.install_guard(run, nscaling)
    plan = dict(home=r"C:\Users\Van\backdooranalog",
                execution_policy=dict(mode="trojannn_plus_exact_p3s3_a1"))
    assert run.classify_gpu("26344, python.exe\n12345, python.exe", {
        26344: "python.exe", 12345: "python.exe"}, 999, plan=plan,
        windows_processes=nscaling + [a1_peer]) == [(12345, "python.exe")]


@pytest.fixture(name="execution_request")
def request_fixture(context, nscaling, tmp_path, monkeypatch):
    # A synthetic execution receipt; no checkpoint/model is loaded here.
    request = dict(schema=1, trojan_runner_sha256=context.RUNNER_SHA256,
                   approved_nscaling=copy.deepcopy(nscaling))
    for key in ("home", "python", "trojan_repo", "a1_repo", "trojan_out", "a1_out",
                "clean_checkpoint", "smoke_receipt"):
        request[key] = str((tmp_path / key).resolve())
    request["source_sha256"] = {
        str(SHARED / "trojan_context.py"): "a" * 64,
        str(SHARED / "controller.py"): "a" * 64,
    }
    monkeypatch.setattr(context, "sha", lambda path: "a" * 64)
    request["trojan_gate_sha256"] = "b" * 64
    request["trojan_trigger_sha256"] = "c" * 64
    request["request_sha256"] = context.digest(request)
    return request


def resign(context, request):
    request = copy.deepcopy(request)
    request.pop("request_sha256", None)
    request["request_sha256"] = context.digest(request)
    return request


def test_valid_external_request_binds_adapter_sources_without_changing_recipe(context, execution_request):
    request = execution_request
    assert context.validate_request(request) == request


@pytest.mark.parametrize("key", ["home", "python", "trojan_repo", "a1_repo", "trojan_out",
                                 "a1_out", "clean_checkpoint", "smoke_receipt"])
def test_request_rejects_relative_paths_even_with_recomputed_digest(context, execution_request, key):
    request = execution_request
    request[key] = "relative/path"
    with pytest.raises(ValueError):
        context.validate_request(resign(context, request))


def test_request_tamper_requires_new_bound_receipt(context, execution_request):
    request = execution_request
    request["trojan_trigger_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="request"):
        context.validate_request(request)


@pytest.mark.parametrize("where", ["trojan_repo", "a1_repo", "trojan_out", "a1_out"])
def test_shared_receipt_cannot_modify_frozen_source_or_historical_output(context, execution_request, where):
    request = execution_request
    request["smoke_receipt"] = str(Path(request[where]) / "shared-smoke.json")
    with pytest.raises(ValueError, match="frozen"):
        context.validate_request(resign(context, request))


def test_changed_scientific_runner_hash_is_not_a_sharing_exception(context, execution_request):
    request = execution_request
    request["trojan_runner_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen TrojanNN"):
        context.validate_request(resign(context, request))


def test_both_adapter_sources_must_be_bound_exactly(context, execution_request):
    request = execution_request
    request["source_sha256"].pop(str(SHARED / "controller.py"))
    with pytest.raises(ValueError, match="both external"):
        context.validate_request(resign(context, request))


def test_source_change_invalidates_request(context, execution_request, monkeypatch):
    request = execution_request
    monkeypatch.setattr(context, "sha", lambda path: "e" * 64)
    with pytest.raises(ValueError, match="source changed"):
        context.validate_request(request)


@pytest.fixture
def smoke(context, execution_request):
    request = execution_request
    env = {"gpu_name": "mock 4090", "physical_batch": 32}
    row = dict(status="shared_smoke_passed",
               binding=context.smoke_binding(request, None, env), physical_batch=32,
               steps=2, real_train_input=True, poison_examples=4, losses=[.3, .2],
               approved_nscaling=copy.deepcopy(request["approved_nscaling"]))
    row["receipt_sha256"] = context.digest(row)
    return env, row


def write_smoke(request, row):
    import json
    Path(request["smoke_receipt"]).write_text(json.dumps(row), encoding="utf-8")


def test_shared_cuda_smoke_receipt_replays_only_matching_context(context, execution_request, smoke):
    request = execution_request
    env, row = smoke
    write_smoke(request, row)
    assert context.check_smoke(request, None, env) == row


@pytest.mark.parametrize("key,value", [("status", "prepared_only"), ("physical_batch", 16),
                                     ("steps", 1), ("real_train_input", False),
                                     ("poison_examples", 0), ("losses", [float("inf"), .1]),
                                     ("approved_nscaling", [])])
def test_malformed_or_relabelled_cuda_receipt_is_rejected(context, execution_request, smoke, key, value):
    request = execution_request
    env, row = smoke
    row.pop("receipt_sha256")
    row[key] = value
    # Nonfinite fields are rejected by canonical digest itself; the checker must
    # never accept them even with an arbitrary digest claim.
    try:
        row["receipt_sha256"] = context.digest(row)
    except ValueError:
        row["receipt_sha256"] = "f" * 64
    write_smoke(request, row)
    with pytest.raises(ValueError):
        context.check_smoke(request, None, env)


def test_foreign_execution_or_request_cannot_reuse_prior_shared_smoke(context, execution_request, smoke):
    request = execution_request
    env, row = smoke
    write_smoke(request, row)
    with pytest.raises(ValueError, match="shared-context"):
        context.check_smoke(request, None, {**env, "gpu_name": "different GPU"})
    changed = copy.deepcopy(request)
    changed["request_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="shared-context"):
        context.check_smoke(changed, None, env)


@pytest.fixture
def controller():
    sys.path.insert(0, str(SHARED))
    return load_file(SHARED / "controller.py", "shared4090_controller_test")


def test_request_digest_is_order_invariant_and_ignores_only_own_digest(controller):
    value = {"b": 2, "a": {"y": 3, "x": 4}}
    digest = controller.request_digest(value)
    assert digest == controller.request_digest({"a": {"x": 4, "y": 3}, "b": 2})
    assert digest == controller.request_digest({**value, "request_sha256": "ignore-self-only"})
    assert digest != controller.request_digest({**value, "b": 3})


def test_execution_json_is_create_exclusive_and_never_overwrites(controller, tmp_path):
    import json
    path = tmp_path / "request.json"
    controller.write_new_json(path, {"immutable": "original"})
    original = path.read_bytes()
    assert json.loads(original) == {"immutable": "original"}
    with pytest.raises((ValueError, FileExistsError)):
        controller.write_new_json(path, {"immutable": "new"})
    assert path.read_bytes() == original


@pytest.fixture
def controller_inputs(controller, nscaling, tmp_path, monkeypatch):
    home = tmp_path / "home"
    python = home / "envs/dose-backdoor/python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"mock-python-never-executed")
    a1_out = home / "runs/mmfi_p3s3_hpeli_a1_s42_4090_with_sig_v1"
    checkpoint = a1_out / "p3_s3/a1/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"mock-checkpoint-never-loaded")
    def source_sha(path):
        return {
            "run_mmfi_p1s1_trojannn.py": controller.TROJAN_SHA,
            "run_mmfi_p3_splits.py": controller.A1_SHA,
        }.get(Path(path).name, "d" * 64)
    monkeypatch.setattr(controller, "sha", source_sha)
    inputs = (home, tmp_path / "trojan_repo", tmp_path / "code", python,
              copy.deepcopy(nscaling))
    return inputs


def test_make_request_records_only_execution_context_and_does_not_create_runs(
        controller, controller_inputs):
    request = controller.make_request(*controller_inputs)
    assert request["request_sha256"] == controller.request_digest(request)
    assert request["policy"]["max_model_training_workers"] == 2
    assert request["policy"]["max_total_compute_jobs"] == 3
    assert request["policy"]["old_gate_certifies_shared_context"] is False
    assert request["policy"]["no_kill"] is True
    assert request["policy"]["no_retry"] is True
    assert {r["ProcessId"] for r in request["approved_nscaling"]} == {26344, 44248}
    assert not Path(request["execution_dir"]).exists()
    assert not Path(request["smoke_receipt"]).exists()
    assert set(request["source_sha256"]) == {
        str(SHARED / "controller.py"), str(SHARED / "trojan_context.py")}


def test_make_request_without_saved_a1_checkpoint_never_falls_back_to_epoch_zero(
        controller, controller_inputs):
    checkpoint = (controller_inputs[0] /
                  "runs/mmfi_p3s3_hpeli_a1_s42_4090_with_sig_v1/p3_s3/a1/checkpoint.pt")
    checkpoint.unlink()
    with pytest.raises(ValueError, match="no epoch-zero fallback"):
        controller.make_request(*controller_inputs)


def test_make_request_refuses_changed_prepared_source(controller, controller_inputs, monkeypatch):
    monkeypatch.setattr(controller, "sha", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="scientific runner"):
        controller.make_request(*controller_inputs)


def test_make_request_does_not_approve_a_partial_nscaling_lineage(controller, controller_inputs):
    inputs = list(controller_inputs)
    inputs[4] = inputs[4][:1]
    with pytest.raises(ValueError, match="lineage is incomplete"):
        controller.make_request(*inputs)


def test_make_request_rejects_substituted_a1_checkout(controller, controller_inputs):
    inputs = list(controller_inputs)
    inputs[2] = inputs[0].parent / "unrelated_a1_repo"
    with pytest.raises(ValueError, match="original sibling code"):
        controller.make_request(*inputs)


def test_a1_command_is_exact_historical_serial_resume_only(controller, controller_inputs):
    request = controller.make_request(*controller_inputs)
    args = controller.a1_command(request)
    assert len(args) == 17
    assert args[:3] == [request["python"], "-u",
                        str(Path(request["a1_repo"]) / "ATKBackd/run_mmfi_p3_splits.py")]
    assert dict(zip(args[3::2], args[4::2])) == {
        "--data-home": request["home"], "--outdir": request["a1_out"],
        "--device": "cuda:0", "--workers": "0", "--jobs-per-gpu": "1",
        "--mode": "launch", "--_cell": "p3_s3_a1"}
    assert not any("sig" == arg or "prepare" == arg for arg in args)


@pytest.fixture
def mocked_launch(controller, controller_inputs, monkeypatch):
    request = controller.make_request(*controller_inputs)
    calls = []
    children = []
    state = dict(epoch=27, next_epoch=28, checkpoint_sha256="f" * 64,
                 plan_sha256="a" * 64, complete=False)
    class Child:
        def __init__(self, args):
            self.args = args
            self.pid = 9000 + len(children)
            self.mutated = False
        def poll(self):
            return 0
        def terminate(self):
            self.mutated = True
            pytest.fail("No worker may be terminated")
        def kill(self):
            self.mutated = True
            pytest.fail("No worker may be killed")
    def popen(args, **kwargs):
        calls.append(("popen", args))
        child = Child(args)
        children.append(child)
        return child
    def run(args, **kwargs):
        calls.append(("run", args))
        return SimpleNamespace(returncode=0)
    def audit(req):
        calls.append(("audit", req))
        return state
    monkeypatch.setattr(controller, "child_phase", lambda path, phase: ["adapter", phase])
    monkeypatch.setattr(controller.subprocess, "run", run)
    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    monkeypatch.setattr(controller, "audit_a1", audit)
    monkeypatch.setattr(controller.context, "windows_processes", lambda: controller_inputs[4])
    monkeypatch.setattr(controller, "a1_live", lambda req, records: set())
    return request, calls, children, state


def test_launch_checks_gate_then_a1_audit_before_starting_any_worker(controller, mocked_launch):
    request, calls, children, state = mocked_launch
    controller.launch(request)
    assert [(kind, args if kind == "run" else None) for kind, args in calls[:3]] == [
        ("run", ["adapter", "check"]), ("audit", None), ("popen", None)]
    assert len(children) == 2
    assert children[0].args == controller.a1_command(request)
    assert children[1].args == ["adapter", "launch"]
    assert all(not child.mutated for child in children)


def test_failed_gate_starts_no_a1_or_trojan_worker(controller, mocked_launch, monkeypatch):
    request, calls, children, state = mocked_launch
    def fail_check(args, **kwargs):
        assert args == ["adapter", "check"]
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(controller.subprocess, "run", fail_check)
    with pytest.raises(subprocess.CalledProcessError):
        controller.launch(request)
    assert children == []
    assert calls == []


def test_failed_a1_audit_starts_no_worker(controller, mocked_launch, monkeypatch):
    request, calls, children, state = mocked_launch
    def failed_audit(request):
        raise ValueError("A1 checkpoint config mismatch")
    monkeypatch.setattr(controller, "audit_a1", failed_audit)
    with pytest.raises(ValueError, match="config mismatch"):
        controller.launch(request)
    assert children == []


def test_smoke_failure_never_starts_trojan_or_kills_owned_a1(controller, mocked_launch, monkeypatch):
    request, calls, children, state = mocked_launch
    def run(args, **kwargs):
        if args == ["adapter", "smoke"]:
            raise subprocess.CalledProcessError(1, args)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(controller.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        controller.launch(request)
    assert len(children) == 1
    assert children[0].args == controller.a1_command(request)
    assert not children[0].mutated


@pytest.mark.parametrize("already_running,already_complete", [(True, False), (False, True)])
def test_existing_or_complete_a1_is_never_launched_twice(
        controller, mocked_launch, monkeypatch, already_running, already_complete):
    request, calls, children, state = mocked_launch
    state["complete"] = already_complete
    monkeypatch.setattr(controller, "a1_live", lambda *args: {567} if already_running else set())
    controller.launch(request)
    assert len(children) == 1
    assert children[0].args == ["adapter", "launch"]


def test_a1_audit_preserves_strict_resume_and_requires_model_optimizer_rng(controller):
    audit = controller.A1_AUDIT
    for token in ("strict_resume", "checkpoint.pt", "cfg_fingerprint", "_config_fingerprint",
                  "optimizer", "rng", "0<=epoch<=49", "preflight.json", "gpu_smoke.json"):
        assert token in audit
    assert "torch.save" not in audit
    assert "unlink" not in audit
    assert "_receipt(plan" in audit


def test_nonzero_owned_worker_does_not_release_lock_while_peer_still_alive(controller, monkeypatch):
    sleeps = []
    class Child:
        def __init__(self, values):
            self.values = iter(values)
            self.last = None
        def poll(self):
            self.last = next(self.values, self.last)
            return self.last
        def terminate(self):
            pytest.fail("Peer must be drained, never terminated")
        def kill(self):
            pytest.fail("Peer must be drained, never killed")
    monkeypatch.setattr(controller.time, "sleep", lambda delay: sleeps.append(delay))
    owned = {"failed": Child([1, 1, 1]), "still_running": Child([None, None, 0])}
    assert controller.drain_owned(owned) == {"failed": 1, "still_running": 0}
    assert len(sleeps) == 2


def test_smoke_failure_drains_live_owned_a1_before_returning(controller, mocked_launch, monkeypatch):
    request, calls, children, state = mocked_launch
    drained = []
    old_popen = controller.subprocess.Popen
    def popen(*args, **kwargs):
        child = old_popen(*args, **kwargs)
        child.poll = lambda: None
        return child
    def run(args, **kwargs):
        if args == ["adapter", "smoke"]:
            raise subprocess.CalledProcessError(1, args)
        return SimpleNamespace(returncode=0)
    def drain(owned):
        drained.append(set(owned))
        assert set(owned) == {"a1"}
        return {"a1": 0}
    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    monkeypatch.setattr(controller.subprocess, "run", run)
    monkeypatch.setattr(controller, "drain_owned", drain)
    with pytest.raises(subprocess.CalledProcessError):
        controller.launch(request)
    assert drained == [{"a1"}]
    assert len(children) == 1 and not children[0].mutated
    assert (Path(request["execution_dir"]) / "exit_after_failure.json").is_file()


@pytest.fixture
def a1_audit_fixture(controller, tmp_path, monkeypatch):
    import json
    root = tmp_path / "a1"
    root.mkdir()
    (root / "checkpoint.pt").write_bytes(b"not-a-real-torch-file")
    plan = {"sha256": "c" * 64, "cells": [{"id": "p3_s3_a1", "folder": str(root),
            "cfg": {"strict_resume": True, "epochs": 50}}]}
    plan_file = root / "plan.json"
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    blob = {"epoch": 27, "cfg_fingerprint": "d" * 64, "model": {"weight": 1},
            "optimizer": {"state": {}, "param_groups": [{"lr": .001, "momentum": .9,
                                                       "weight_decay": 0.}]},
            "rng": {"python": [1], "numpy": [2], "torch": [3], "cuda": [[4]]}}
    receipts = []
    run = SimpleNamespace(build_plan=lambda *args: plan, _plan_file=lambda *args: plan_file,
                          _receipt=lambda p, name: receipts.append(name),
                          _config_fingerprint=lambda cfg: "d" * 64,
                          _complete=lambda cell: False)
    monkeypatch.setitem(sys.modules, "run_mmfi_p3_splits", run)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *args, **kw: blob))
    monkeypatch.setattr(sys, "argv", ["audit", str(tmp_path), str(tmp_path), str(root)])
    return blob, plan, plan_file, receipts


def test_a1_audit_executes_read_only_against_mock_checkpoint(
        controller, a1_audit_fixture, capsys):
    import json
    blob, plan, plan_file, receipts = a1_audit_fixture
    before = (plan_file.read_bytes(), (plan_file.parent / "checkpoint.pt").read_bytes())
    exec(controller.A1_AUDIT, {})
    output = json.loads(capsys.readouterr().out)
    assert output["epoch"] == 27 and output["next_epoch"] == 28
    assert output["complete"] is False
    assert receipts == ["preflight.json", "gpu_smoke.json"]
    assert (plan_file.read_bytes(), (plan_file.parent / "checkpoint.pt").read_bytes()) == before


@pytest.mark.parametrize("key,value", [("epoch", -1), ("epoch", 50), ("epoch", True),
                                     ("cfg_fingerprint", "foreign"), ("model", {}),
                                     ("optimizer", {}), ("rng", {})])
def test_a1_audit_executes_strict_state_rejection(controller, a1_audit_fixture, key, value):
    blob, plan, plan_file, receipts = a1_audit_fixture
    blob[key] = value
    with pytest.raises(ValueError):
        exec(controller.A1_AUDIT, {})


def test_a1_audit_rejects_non_strict_resume_before_checkpoint_loading(
        controller, a1_audit_fixture):
    import json
    blob, plan, plan_file, receipts = a1_audit_fixture
    plan["cells"][0]["cfg"]["strict_resume"] = False
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="strict resume"):
        exec(controller.A1_AUDIT, {})


@pytest.mark.parametrize("missing", ["python", "numpy", "torch", "cuda"])
def test_a1_audit_requires_every_rng_state(controller, a1_audit_fixture, missing):
    blob, plan, plan_file, receipts = a1_audit_fixture
    del blob["rng"][missing]
    with pytest.raises(ValueError, match="exact RNG"):
        exec(controller.A1_AUDIT, {})


def test_a1_audit_rejects_absent_cuda_rng(controller, a1_audit_fixture):
    blob, plan, plan_file, receipts = a1_audit_fixture
    blob["rng"]["cuda"] = None
    with pytest.raises(ValueError, match="exact RNG"):
        exec(controller.A1_AUDIT, {})


@pytest.mark.parametrize("key,value", [("lr", .01), ("momentum", 0),
                                     ("weight_decay", .01), ("betas", [.9, .99])])
def test_a1_audit_rejects_optimizer_recipe_change(controller, a1_audit_fixture, key, value):
    blob, plan, plan_file, receipts = a1_audit_fixture
    blob["optimizer"]["param_groups"][0][key] = value
    with pytest.raises(ValueError, match="saved optimizer"):
        exec(controller.A1_AUDIT, {})


HELPER = ROOT / "remote_tools/Start-MMFiA1TrojanNNShared4090.ps1"
DOC = ROOT / "remote_tools/MMFI_A1_TROJANNN_SHARED4090_VI.md"
PS = shutil.which("powershell") or shutil.which("pwsh")


def test_powershell_shared_exception_is_explicit_and_foreground():
    source = HELPER.read_text(encoding="utf-8")
    assert "[switch]$AllowKnownNScaling" in source
    assert "if ($Mode -eq 'Launch' -and !$AllowKnownNScaling)" in source
    assert "& $PythonExe @Arguments" in source
    assert "'--trojan-repo',$Repo,'--a1-repo',$A1Repo" in source
    for forbidden in ("Stop-Process", "taskkill", "Start-Process", "Register-ScheduledTask",
                      "Invoke-CimMethod", "Set-ExecutionPolicy", "pip install", "Remove-Item"):
        assert forbidden.casefold() not in source.casefold()
    assert "-and !$AllowKnownNScaling" in source
    assert "[string]$Mode='Status'" in source


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
def test_shared_helper_parses_without_execution():
    literal = str(HELPER).replace("'", "''")
    code = ("$t=$null;$e=$null;"
            f"[System.Management.Automation.Language.Parser]::ParseFile('{literal}',"
            "[ref]$t,[ref]$e)|Out-Null;if($e.Count){$e|Out-String|Write-Output;exit 1}")
    result = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-Command", code],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not PS, reason="PowerShell unavailable")
def test_shared_helper_refuses_launch_without_explicit_exception_before_any_work():
    result = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-File", str(HELPER), "-Mode", "Launch"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Explicit -AllowKnownNScaling is required" in result.stdout + result.stderr


def test_shared_docs_distinguish_execution_test_from_full_training_and_durability():
    text = " ".join(DOC.read_text(encoding="utf-8").split())
    for required in ("P1S1", "P3S3", "-AllowKnownNScaling", "giữ SSH mở",
                     "không bảo đảm", "VRAM", "không fallback epoch0",
                     "không tối ưu lại trigger", "append-only", "không retry",
                     "PID", "creation time", "không pull/sửa checkout đó"):
        assert required.casefold() in text.casefold()
