"""Execution-only RTX3090 port of the published SIBA/BackTime P3S3 queue.

Loads the unchanged scientific runner in a private module. Only the hardware
guard, profile labels and execution-adapter binding change. No job termination,
library installation, automatic retry, checkpoint import or shortened schedule.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import uuid


PROFILE = "mmfi_p3s3_siba_ladder_source_adapted_s42_3090_v2"
METHODS = ("siba", "backtime")
MODES = ("inspect", "preflight", "prepare", "check", "launch", "status")


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gpu_execution():
    import torch
    from run_mmfi_badnets_two_models_4090 import gpu_signature
    from run_piw_hpeli_clean import _setup_execution

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ValueError("Use physical GPU0 without CUDA_VISIBLE_DEVICES remapping")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible RTX3090 CUDA GPU is required; no CPU fallback")
    props = torch.cuda.get_device_properties(0)
    if props.name != "NVIDIA GeForce RTX 3090" or props.total_memory < 22 * 2**30:
        raise ValueError(f"Expected RTX3090 with24GiB, found {props.name}")
    physical = gpu_signature()
    if physical[1] != props.name:
        raise ValueError("PyTorch cuda:0 differs from physical GPU0")
    return dict(torch=_setup_execution("cuda:0"), physical_gpu0=physical)


def load_runtime(repo):
    """Keep original imported modules and their scientific implementations intact."""
    repo = Path(repo).resolve()
    atk = repo / "ATKBackd"
    source = atk / "run_mmfi_p3s3_siba_ladder_4090.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing published scientific runner: {source}")
    sys.path.insert(0, str(atk))
    name = "_mmfi_siba_backtime3090_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, source)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    if Path(result.common.__file__).resolve().parent != atk:
        raise ValueError("Scientific dependency imported from a different checkout")
    result.PROFILE = PROFILE
    original_profile, original_plan = result.backtime_profile, result.build_plan

    def backtime_profile(policy):
        value = original_profile(policy)
        if not value.endswith("_4090_v2"):
            raise ValueError("Published source profile changed; inspect before porting")
        return value.removesuffix("_4090_v2") + "_3090_v2"

    def build_plan(*args, **kwargs):
        plan = original_plan(*args, **kwargs)
        if plan["methods"] != list(METHODS) or plan["backtime_recipient_policy"] != "adaptive":
            raise ValueError("3090 queue is exactly SIBA + adaptive BackTime; no LADDER or substitution")
        plan.pop("plan_sha256")
        plan["execution_adapter"] = dict(
            script_sha256=file_sha(__file__), required_gpu="NVIDIA GeForce RTX 3090",
            scientific_engine_sha256=file_sha(source),
            scientific_change=False, max_workers=1,
            migration="fresh isolated 3090 run; no 4090 checkpoint/gate reuse")
        plan["plan_sha256"] = result.digest(plan)
        return plan

    result.backtime_profile = backtime_profile
    result.build_plan = build_plan
    result.execution = gpu_execution
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-home", type=Path, default=Path.home() / "backdooranalog")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--jobs-per-gpu", type=int, choices=(1,), default=1)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args(argv)
    # A source port, not a new algorithm. Keep all attacker/victim/metric paths.
    runtime = load_runtime(args.repo)
    out = args.outdir or args.data_home / "runs" / runtime.output_profile("adaptive", METHODS)
    print(f"RTX3090_SOURCE_ADAPTED_QUEUE mode={args.mode} methods=siba,backtime "
          "policy=adaptive workers=1; no process stopped", flush=True)
    return runtime.main([
        "--mode", args.mode, "--data-home", str(args.data_home), "--outdir", str(out),
        "--device", args.device, "--jobs-per-gpu", str(args.jobs_per_gpu),
        "--backtime-policy", "adaptive", "--methods", *METHODS])


if __name__ == "__main__":
    main()
