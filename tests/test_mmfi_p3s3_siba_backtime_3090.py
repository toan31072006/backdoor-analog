"""RTX3090 execution-port tests: CPU/mock only, no remote efficacy claim."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ATKBackd"))
spec = importlib.util.spec_from_file_location("siba_backtime3090_adapter",
    ROOT / "remote_tools/Run-MMFiP3S3SibaBackTime3090.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
native = importlib.import_module("run_mmfi_p3s3_siba_ladder_4090")


@pytest.fixture
def runtime():
    return adapter.load_runtime(ROOT)


def plan(r, home):
    return r.build_plan(home, home / "runs" / r.output_profile("adaptive", adapter.METHODS),
                        backtime_policy="adaptive", selected_methods=adapter.METHODS)


def test_private_port_does_not_mutate_native_module_or_source(runtime, tmp_path):
    source = ROOT / "ATKBackd/run_mmfi_p3s3_siba_ladder_4090.py"
    before = adapter.file_sha(source)
    original_profile, original_execution = native.PROFILE, native.execution
    result = plan(runtime, tmp_path)
    assert native.PROFILE == original_profile and native.execution is original_execution
    assert "4090" in original_profile and "3090" in runtime.PROFILE
    assert native is not runtime and native.__name__ != runtime.__name__
    assert adapter.file_sha(source) == before == result["execution_adapter"]["scientific_engine_sha256"]


def test_scientific_recipe_operators_and_methods_unchanged(runtime, tmp_path):
    a, b = plan(native, tmp_path), plan(runtime, tmp_path)
    assert a["operators"] == b["operators"]
    assert a["contract"] == b["contract"]
    assert a["sources"] == b["sources"]
    assert b["methods"] == ["siba", "backtime"]
    assert b["cells"] == ["clean_surrogate", "siba", "backtime"]
    assert b["backtime_recipient_policy"] == "adaptive" and b["jobs_per_gpu"] == 1
    for method in b["cells"]:
        old, new = dict(a["cfg"][method]), dict(b["cfg"][method])
        assert old.pop("learned_baseline_profile") != new.pop("learned_baseline_profile")
        assert old == new
    for name in ("dataset", "target_fn", "new_components", "train", "fit_trigger", "learned_smoke",
                 "evaluate", "export", "require_idle", "launch", "prepare", "gate_check"):
        assert getattr(runtime, name).__code__.co_code == getattr(native, name).__code__.co_code
    assert b["contract"]["counts"] == dict(training=240570, test=80190)
    assert b["operators"]["siba"]["epochs"] == 200
    assert b["operators"]["backtime"]["fit_epochs"] == 100


def test_profile_output_and_adapter_fingerprint_isolated(runtime, tmp_path):
    p = plan(runtime, tmp_path)
    assert "_3090_v2" in p["profile"] and "4090" not in p["profile"]
    assert p["profile"] == Path(p["outdir"]).name
    assert p["execution_adapter"]["script_sha256"] == adapter.file_sha(adapter.__file__)
    assert p["execution_adapter"]["scientific_change"] is False
    assert p["execution_adapter"]["required_gpu"] == "NVIDIA GeForce RTX 3090"
    unsigned = dict(p); claimed = unsigned.pop("plan_sha256")
    assert runtime.digest(unsigned) == claimed
    assert p["plan_sha256"] != plan(native, tmp_path)["plan_sha256"]
    assert p["contract"]["warm_start"] is False


def test_wrong_method_set_is_not_silently_substituted(runtime, tmp_path):
    with pytest.raises(ValueError, match="exactly SIBA"):
        runtime.build_plan(tmp_path, tmp_path / "runs/mmfi_p3s3_siba_ladder_wrong")


def test_status_and_inspect_are_read_only_no_cuda(runtime, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(adapter, "load_runtime", lambda repo:runtime)
    for name in ("execution", "require_idle", "audit", "prepare", "train", "fit_trigger", "launch"):
        monkeypatch.setattr(runtime, name, lambda *a:pytest.fail("read-only mode called scientific/GPU work"))
    for mode in ("status", "inspect"):
        adapter.main(["--mode", mode, "--data-home", str(tmp_path)])
    output = capsys.readouterr().out
    assert "INSPECT_ONLY" in output and "READ_ONLY" in output
    assert "_3090_v2" in output and not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode,called", [("prepare", "prepare"), ("check", "gate_check"),
                                         ("launch", "launch"), ("preflight", "audit")])
def test_cli_dispatch_keeps_exact_queue_and_no_automatic_later_phase(runtime, tmp_path, monkeypatch, mode, called):
    from contextlib import nullcontext
    seen = []
    monkeypatch.setattr(adapter, "load_runtime", lambda repo:runtime)
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _:SimpleNamespace(free=10*2**30))
    monkeypatch.setattr(runtime, "require_idle", lambda:None)
    monkeypatch.setattr(runtime, "log_to", lambda *a:nullcontext())
    for function in ("prepare", "gate_check", "launch", "audit"):
        def record(p, function=function):
            assert p["methods"] == list(adapter.METHODS)
            assert p["backtime_recipient_policy"] == "adaptive"
            assert "_3090_v2" in p["profile"]
            seen.append(function)
        monkeypatch.setattr(runtime, function, record)
    adapter.main(["--mode", mode, "--data-home", str(tmp_path)])
    assert seen == [called]


def test_3090_queue_reuses_native_serial_order(runtime, tmp_path, monkeypatch):
    from contextlib import nullcontext
    seen = []
    monkeypatch.setattr(runtime, "gate_check", lambda _: {})
    monkeypatch.setattr(runtime, "require_idle", lambda:None)
    monkeypatch.setattr(runtime, "completed", lambda *a:None)
    monkeypatch.setattr(runtime, "log_to", lambda *a:nullcontext())
    for function in ("train", "fit_trigger", "learned_smoke", "evaluate"):
        monkeypatch.setattr(runtime, function, lambda p,m,e,function=function:seen.append((function,m)))
    monkeypatch.setattr(runtime, "export", lambda *a:seen.append(("export","all")))
    runtime.launch(plan(runtime, tmp_path))
    assert seen == [("train","clean_surrogate"),
        ("fit_trigger","siba"),("learned_smoke","siba"),("train","siba"),("evaluate","siba"),
        ("fit_trigger","backtime"),("learned_smoke","backtime"),("train","backtime"),("evaluate","backtime"),
        ("export","all")]


@pytest.mark.parametrize("args", [["--jobs-per-gpu","2"], ["--device","cpu"],
                                  ["--device","cuda:1"], ["--methods","ladder"]])
def test_cli_rejects_scope_expansion(args):
    with pytest.raises(SystemExit):
        adapter.main(["--mode","inspect", *args])


@pytest.fixture
def gpu_mocks(monkeypatch):
    import run_mmfi_badnets_two_models_4090 as common_gpu
    props = SimpleNamespace(name="NVIDIA GeForce RTX 3090", total_memory=24*2**30)
    cuda = SimpleNamespace(is_available=lambda:True, device_count=lambda:1,
                           get_device_properties=lambda _:props)
    fake = SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "run_piw_hpeli_clean",
                        SimpleNamespace(_setup_execution=lambda device:{"device":device,"deterministic":True}))
    monkeypatch.setattr(common_gpu, "gpu_signature", lambda:["GPU-mock",props.name,"24576","mock-driver"])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return props, cuda, common_gpu


def test_3090_hardware_guard_keeps_native_execution_signature(gpu_mocks):
    env = adapter.gpu_execution()
    assert env["torch"] == {"device":"cuda:0","deterministic":True}
    assert env["physical_gpu0"] == ["GPU-mock","NVIDIA GeForce RTX 3090","24576","mock-driver"]


@pytest.mark.parametrize("reason", ["cpu", "two", "4090", "small", "remap", "physical_mismatch"])
def test_wrong_hardware_is_not_accepted_or_fallback(gpu_mocks, monkeypatch, reason):
    props, cuda, common_gpu = gpu_mocks
    if reason == "cpu": cuda.is_available = lambda:False
    if reason == "two": cuda.device_count = lambda:2
    if reason == "4090": props.name = "NVIDIA GeForce RTX 4090"
    if reason == "small": props.total_memory = 12*2**30
    if reason == "remap": monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    if reason == "physical_mismatch":
        monkeypatch.setattr(common_gpu,"gpu_signature",lambda:["GPU-other","NVIDIA GeForce RTX 4090","24576","x"])
    with pytest.raises(ValueError):
        adapter.gpu_execution()


def test_document_discloses_port_only_scope_and_cost():
    doc = (ROOT / "remote_tools/MMFI_P3S3_SIBA_BACKTIME_3090_VI.md").read_text(encoding="utf-8")
    for required in ("Material Passport", "200/B128", "100/B64", "240570 TRAIN / 80190 TEST",
                     "Exploratory T6", "keep SSH open", "No 4090 checkpoint", "Nothing is killed"):
        assert required in doc
