"""Vast controller for two exploratory HPE-Li / PiW3D attack controls.

The scientific worker is ATKBackd/run_piw_hpeli_two_extra.py. This module
only validates a fixed two-GPU execution and schedules its two independent
seed-42 cells. No model, attack, target, optimizer or metric is defined here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


PROFILE = "EXPLORATORY_PIW3D_HPELI_TWO_EXTRA_S42_V1"
CELLS = (("tsba_adapted", "cuda:0"), ("smooth_lf_csi", "cuda:1"))
REPO_DEFAULT = Path(__file__).resolve().parents[2]
HOME_DEFAULT = Path("/workspace/piw3d")
OUT_DEFAULT = Path("/workspace/runs/piw3d_hpeli_two_extra_s42_v1")
WORKER = Path("ATKBackd/run_piw_hpeli_two_extra.py")


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_sha(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing record differs: {path}")
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".writing")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


@contextmanager
def controller_lock(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "controller.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Controller lock exists: {path}; inspect it before retry") from error
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()} started={time.time()}\n")
        yield
    finally:
        path.unlink()


def paths(args):
    repo = args.repo_dir.resolve()
    dataset = args.dataset_root.resolve()
    action = args.action_npy.resolve()
    reference = args.reference_archive.resolve()
    out = args.outdir.resolve()
    for path in (repo / WORKER, dataset / "train_data/train_data_list.txt",
                 dataset / "test_data/test_data_list.txt", action, reference):
        if not path.is_file():
            raise FileNotFoundError(f"Required PiW input missing: {path}")
    for protected in (repo, dataset, action.parent, reference.parent):
        if (out == protected or out.is_relative_to(protected) or
                protected.is_relative_to(out)):
            raise ValueError("Output must be isolated from source, dataset, action and reference")
    if out in (Path("/workspace"), Path("/workspace/runs")):
        raise ValueError("Output path is too broad")
    return repo, dataset, action, reference, out


def selected_sources(repo: Path) -> dict[str, str]:
    selected = (repo / WORKER,
                repo / "remote_linux/piw3d_hpeli_extra/controller.py",
                repo / "remote_linux/piw3d_hpeli_extra/start.sh")
    if any(not path.is_file() or path.is_symlink() for path in selected):
        raise ValueError("PiW HPE-Li launcher source is missing or linked")
    return {path.relative_to(repo).as_posix(): sha(path) for path in selected}


def gpu_identity() -> list[dict]:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("This profile needs exactly two visible CUDA GPUs")
    result = []
    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        if "5090" not in props.name or props.total_memory < 30 * 2**30:
            raise RuntimeError(f"cuda:{index} must be RTX 5090 with >=30 GiB")
        result.append(dict(index=index, name=props.name,
                           total_memory=props.total_memory,
                           capability=[props.major, props.minor],
                           uuid=str(getattr(props, "uuid", ""))))
    return result


def require_idle_gpus() -> None:
    report = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=30)
    pids = [line.strip() for line in report.stdout.splitlines() if line.strip()]
    if pids:
        raise RuntimeError(f"GPU compute processes are active: {pids}. Wait for the current Vast run to finish")


def quick_inputs(dataset: Path) -> dict[str, str]:
    return {name: sha(dataset / name) for name in
            ("train_data/train_data_list.txt", "test_data/test_data_list.txt")}


def worker_command(args, *extra: str) -> list[str]:
    repo, dataset, action, reference, out = paths(args)
    return [sys.executable, "-u", str(repo / WORKER),
            "--dataset-root", str(dataset), "--action-npy", str(action),
            "--reference-archive", str(reference), "--outdir", str(out), *extra]


def worker_dry_run(args) -> dict:
    report = subprocess.run(worker_command(args, "--dry-run"),
                            capture_output=True, text=True, check=True)
    try:
        value = json.loads(report.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Worker dry-run did not emit JSON: {report.stdout[-800:]}") from error
    if not isinstance(value, dict):
        raise ValueError("Worker dry-run must emit an object")
    return value


def gate_path(out: Path) -> Path:
    return out / "hpeli_two_extra_gate.json"


def smoke(args, attempt: str) -> list[dict]:
    out = args.outdir.resolve()
    folder = out / "smoke" / attempt
    folder.mkdir(parents=True, exist_ok=False)
    children = []
    try:
        for method, device in CELLS:
            log = folder / f"{method}.log"
            with log.open("x", encoding="utf-8") as stream:
                child = subprocess.Popen(
                    worker_command(args, "--cell", method, "--device", device,
                                   "--smoke-only"), stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT)
            children.append((method, device, child, log))
            print(f"[smoke] {method} PID={child.pid} {device} log={log}", flush=True)
        failed = []
        while any(child.poll() is None for _, _, child, _ in children):
            for method, _, child, log in children:
                if child.poll() not in (None, 0):
                    failed.append((method, child.returncode, str(log)))
            if failed:
                for _, _, child, _ in children:
                    if child.poll() is None:
                        child.terminate()
                break
            time.sleep(1)
        for method, _, child, log in children:
            code = child.wait(timeout=30)
            if code != 0 and not any(row[0] == method for row in failed):
                failed.append((method, code, str(log)))
        if failed:
            raise RuntimeError(f"Paired GPU smoke failed: {failed}; no full training launched")
        records = []
        for method, device, _, log in children:
            content = log.read_text(encoding="utf-8", errors="replace")
            if "GPU_SMOKE_WORKER_PASS" not in content:
                raise RuntimeError(f"Smoke worker did not report PASS: {log}")
            records.append(dict(method=method, device=device,
                                log_path=str(log), log_sha256=sha(log)))
        return records
    except BaseException:
        for _, _, child, _ in children:
            if child.poll() is None:
                child.terminate()
        for _, _, child, _ in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
        raise


def prepare(args) -> None:
    repo, dataset, action, reference, out = paths(args)
    with controller_lock(out):
        if gate_path(out).exists():
            raise ValueError("Gate already exists; use --check-only or choose a new output")
        require_idle_gpus()
        identity = gpu_identity()
        dry = worker_dry_run(args)
        subprocess.run(worker_command(args, "--preflight-only"), check=True)
        plan = out / "plan.json"
        receipt = out / "preflight_receipt.json"
        if not plan.is_file() or not receipt.is_file():
            raise RuntimeError("Worker preflight omitted plan.json or preflight_receipt.json")
        attempt = uuid.uuid4().hex
        records = smoke(args, attempt)
        gate = dict(schema=1, profile=PROFILE, cells=list(map(list, CELLS)),
                    worker_dry_run=dry, sources=selected_sources(repo),
                    plan_sha256=sha(plan), preflight_receipt_sha256=sha(receipt),
                    archive_sha256=sha(reference), action_sha256=sha(action),
                    dataset_lists_sha256=quick_inputs(dataset),
                    gpus=identity, smoke_attempt=attempt, smoke=records,
                    policy=dict(seed=42, jobs_per_gpu=1, physical_batch=32,
                                epochs=200, final_epoch=199, no_automatic_retry=True,
                                no_batch_fallback=True, test_checkpoint_selection=False,
                                status="EXPLORATORY_ONE_SEED_PIW3D_HPELI_TWO_EXTRA"))
        gate["gate_sha256"] = object_sha(gate)
        atomic_new_json(gate_path(out), gate)
    print(f"PREPARED_ONLY: {gate_path(out)}; two real GPU smokes passed, no full training", flush=True)


def validate_gate(args) -> dict:
    repo, dataset, action, reference, out = paths(args)
    path = gate_path(out)
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("gate_sha256") != object_sha(
            {key: value for key, value in gate.items() if key != "gate_sha256"}):
        raise ValueError("Controller gate digest differs")
    if gate.get("profile") != PROFILE or gate.get("cells") != list(map(list, CELLS)):
        raise ValueError("Controller profile/cells differ")
    if gate["sources"] != selected_sources(repo):
        raise ValueError("PiW source bytes changed after prepare")
    if gate["worker_dry_run"] != worker_dry_run(args):
        raise ValueError("Worker dry-run plan differs")
    if (gate["plan_sha256"] != sha(out / "plan.json") or
            gate["preflight_receipt_sha256"] != sha(out / "preflight_receipt.json") or
            gate["archive_sha256"] != sha(reference) or
            gate["action_sha256"] != sha(action) or
            gate["dataset_lists_sha256"] != quick_inputs(dataset)):
        raise ValueError("PiW input/plan/preflight bytes changed after prepare")
    if gate["gpus"] != gpu_identity():
        raise ValueError("GPU identity changed after smoke")
    if len(gate["smoke"]) != 2 or any(
            row["log_sha256"] != sha(Path(row["log_path"])) for row in gate["smoke"]):
        raise ValueError("Paired GPU smoke record differs")
    return gate


def launch_workers(args, phase: str):
    out = args.outdir.resolve()
    folder = out / "logs" / phase
    folder.mkdir(parents=True, exist_ok=True)
    children = []
    try:
        for method, device in CELLS:
            result = out / "hpeli/seed42" / method / "evaluation.json"
            if result.exists():
                subprocess.run(worker_command(args, "--cell", method,
                    "--device", device, "--verify-cell"), check=True)
                print(f"[{phase}] VERIFIED_COMPLETE {method}", flush=True)
                continue
            log = folder / f"{method}.{uuid.uuid4().hex[:12]}.log"
            extra = ["--cell", method, "--device", device]
            if phase == "evaluate":
                extra.append("--evaluate-only")
            with log.open("x", encoding="utf-8") as stream:
                child = subprocess.Popen(worker_command(args, *extra),
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            children.append((method, child, log))
            print(f"[{phase}] {method} PID={child.pid} {device} log={log}", flush=True)
        failed = []
        while any(child.poll() is None for _, child, _ in children):
            for method, child, log in children:
                if child.poll() not in (None, 0):
                    failed.append((method, child.returncode, str(log)))
            if failed:
                for _, child, _ in children:
                    if child.poll() is None:
                        child.terminate()
                break
            time.sleep(5)
        for method, child, log in children:
            code = child.wait(timeout=30)
            if code != 0 and not any(row[0] == method for row in failed):
                failed.append((method, code, str(log)))
        if failed:
            raise RuntimeError(f"{phase} failed: {failed}; no automatic retry")
    except BaseException:
        for _, child, _ in children:
            if child.poll() is None:
                child.terminate()
        for _, child, _ in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
        raise


def launch(args):
    _, _, _, _, out = paths(args)
    validate_gate(args)
    with controller_lock(out):
        require_idle_gpus()
        launch_workers(args, "train")
        for method, _ in CELLS:
            if not (out / "hpeli/seed42" / method / "checkpoint.pt").is_file():
                raise RuntimeError(f"Training returned without checkpoint: {method}")
        launch_workers(args, "evaluate")
        for method, device in CELLS:
            subprocess.run(worker_command(args, "--cell", method,
                "--device", device, "--verify-cell"), check=True)
        atomic_new_json(out / "controller_complete.json",
            dict(profile=PROFILE, gate_sha256=json.loads(gate_path(out).read_text())["gate_sha256"],
                 methods=[method for method, _ in CELLS], status="COMPLETE_EXPLORATORY_S42"))
    print("COMPLETE_EXPLORATORY_S42; two HPE-Li baseline victims evaluated", flush=True)


def status(args) -> None:
    out = args.outdir.resolve()
    rows = []
    for method, device in CELLS:
        folder = out / "hpeli/seed42" / method
        rows.append(dict(method=method, device=device,
                         checkpoint=(folder / "checkpoint.pt").is_file(),
                         evaluated=(folder / "evaluation.json").is_file()))
    print(json.dumps(dict(profile=PROFILE, gate=gate_path(out).is_file(),
                          complete=(out / "controller_complete.json").is_file(),
                          cells=rows), indent=2))


def unlock_stale(args) -> None:
    _, _, _, _, out = paths(args)
    lock = out / "controller.lock"
    match = re.fullmatch(r"pid=(\d+) started=.+\n", lock.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError("Unknown controller lock format")
    pid = int(match.group(1))
    if Path("/proc").joinpath(str(pid)).exists():
        raise RuntimeError(f"PID {pid} still exists; inspect it before unlocking")
    require_idle_gpus()
    archived = out / f"controller.lock.stale.{uuid.uuid4().hex[:12]}"
    os.replace(lock, archived)
    print(f"STALE_LOCK_ARCHIVED {archived}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--data-home", type=Path, default=HOME_DEFAULT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--action-npy", type=Path)
    parser.add_argument("--reference-archive", type=Path)
    parser.add_argument("--outdir", type=Path, default=OUT_DEFAULT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--unlock-stale", action="store_true")
    args = parser.parse_args(argv)
    args.dataset_root = args.dataset_root or args.data_home / "datasets/Person-in-WiFi-3D_processed_v1"
    args.action_npy = args.action_npy or args.data_home / "actions/data_bend.npy"
    args.reference_archive = (args.reference_archive or
        args.data_home / "references/piw_hpeli_v3_clean_reference.zip")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None and not args.status:
        raise RuntimeError("Leave CUDA_VISIBLE_DEVICES unset: logical GPU bindings are fixed")
    if args.status:
        status(args)
    elif args.dry_run:
        print(json.dumps(dict(profile=PROFILE, cells=list(map(list, CELLS)),
                              worker=worker_dry_run(args)), indent=2))
        print("DRY_RUN_ONLY; no data hash/CUDA/training/output writes")
    elif args.prepare_only:
        prepare(args)
    elif args.check_only:
        validate_gate(args)
        print("VERIFIED_PRELAUNCH_GATE; no training launched")
    elif args.unlock_stale:
        unlock_stale(args)
    else:
        launch(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[PiW HPE-Li two-extra controller] ERROR: {error}", file=sys.stderr,
              flush=True)
        raise SystemExit(1)
