"""Isolated two-GPU controller for the exploratory PiW3D attack comparison.

The scientific implementation lives in ATKBackd/run_piw_two_models_attacks.py.
This controller owns the input/smoke gate and process scheduling only.  It never
changes a physical batch, an epoch count, or an attack configuration on failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid


BACKBONES = ("metafiplusplus", "wiflow")
METHOD_GROUPS = (
    ("clean", "a1", "wanet_source"),
    ("ftrojan", "fiba", "bpp_source"),
    ("smooth_lf_csi", "tsba_adapted"),
)
JOBS_PER_GPU = 3
PROFILE = "EXPLORATORY_PIW3D_TWO_MODELS_S42_3PERGPU_V2"
DEFAULT_REPO = Path(__file__).resolve().parents[2]
DEFAULT_HOME = Path("/workspace/piw3d")
DEFAULT_OUT = Path("/workspace/runs/piw3d_two_models_s42_3pergpu_v2")


def file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False) + "\n"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing record differs; use a new output: {path}")
        return
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x", encoding="utf-8") as sink:
        sink.write(encoded)
    os.link(temporary, path)
    temporary.unlink()


@contextmanager
def controller_lock(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    lock = outdir / "controller.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Another controller holds {lock}; inspect it") from error
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()} started={time.time()}\n")
        yield
    finally:
        lock.unlink()


def paths(args):
    repo = args.repo_dir.resolve()
    dataset = args.dataset_root.resolve()
    action = args.action_npy.resolve()
    out = args.outdir.resolve()
    if not (repo / "ATKBackd/run_piw_two_models_attacks.py").is_file():
        raise FileNotFoundError("PiW two-model worker is missing from repo")
    if any(out == protected or out.is_relative_to(protected)
           or protected.is_relative_to(out)
           for protected in (repo, dataset, action.parent)):
        raise ValueError("Output must be isolated from source, dataset and action")
    if out == Path("/workspace/runs") or out == Path("/workspace"):
        raise ValueError("Output path is too broad")
    return repo, dataset, action, out


def source_binding(repo: Path) -> dict[str, str]:
    """Bind all local Python science sources and the controller, not run artifacts."""
    selected = [p for p in (repo / "ATKBackd").rglob("*.py")
                if "__pycache__" not in p.parts and "envs" not in p.parts]
    selected += [p for p in (repo / "remote_linux/piw3d_two_models").glob("*.py")]
    result = {}
    for path in sorted(set(selected)):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Linked or missing source: {path}")
        result[path.relative_to(repo).as_posix()] = file_sha(path)
    if not result:
        raise ValueError("No science sources found")
    manifest = repo / "BUNDLE_MANIFEST.json"
    if manifest.is_file():
        result["BUNDLE_MANIFEST.json"] = file_sha(manifest)
    return result


def gpu_identity() -> list[dict]:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Exactly two visible CUDA GPUs are required")
    result = []
    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        if "5090" not in props.name or props.total_memory < 30 * 2**30:
            raise RuntimeError(f"cuda:{index} is not an RTX 5090 with at least 30 GiB VRAM")
        result.append(dict(index=index, name=props.name,
                           total_memory=props.total_memory,
                           capability=[props.major, props.minor],
                           uuid=str(getattr(props, "uuid", ""))))
    return result


def require_idle_gpus() -> None:
    report = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30)
    processes = [line.strip() for line in report.stdout.splitlines() if line.strip()]
    if processes:
        raise RuntimeError(f"CUDA GPU processes are active: {processes}; inspect first")


def run_worker(args, *extra, capture=False, env=None):
    repo, dataset, action, out = paths(args)
    command = [sys.executable, "-u", str(repo / "ATKBackd/run_piw_two_models_attacks.py"),
               "--dataset-root", str(dataset), "--action-npy", str(action),
               "--outdir", str(out), *extra]
    return subprocess.run(command, check=True, capture_output=capture,
                          text=capture, env=env)


def parse_worker_json(stdout: str) -> dict:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Worker did not emit exactly one JSON object: {stdout[-800:]}") from error
    if not isinstance(value, dict):
        raise ValueError("Worker JSON must be an object")
    return value


def cell(backbone: str, method: str, gpu: int) -> dict:
    return dict(id=f"{backbone}__s42__{method}", backbone=backbone,
                method=method, device=f"cuda:{gpu}")


def waves() -> list[list[dict]]:
    result = []
    for wave_index, methods in enumerate(METHOD_GROUPS):
        # The first two waves fill three slots per GPU. Flip the 2:1 model
        # mix between waves so the smoke exercises both memory topologies.
        wave = []
        for method_index, method in enumerate(methods):
            metafi_gpu = (method_index + wave_index) % 2
            wave.append(cell("metafiplusplus", method, metafi_gpu))
            wave.append(cell("wiflow", method, 1 - metafi_gpu))
        result.append(wave)
    return result


def dataset_quick_binding(dataset: Path) -> dict:
    """Fast launch guard; the full per-file content audit runs in prepare only."""
    result = {}
    for split in ("train_data", "test_data"):
        official = dataset / split / (split + "_list.txt")
        if not official.is_file() or official.is_symlink():
            raise FileNotFoundError(f"Official PiW3D list missing: {official}")
        result[official.relative_to(dataset).as_posix()] = file_sha(official)
    return result


def plan(args, dry_worker: dict | None = None) -> dict:
    repo, dataset, action, out = paths(args)
    result = dict(schema=1, profile=PROFILE, seed=42,
                  dataset="Person-in-WiFi-3D one-person official TRAIN/TEST",
                  repo_dir=str(repo), dataset_root=str(dataset),
                  action_npy=str(action), outdir=str(out),
                  source_binding=source_binding(repo),
                  waves=waves(), jobs_per_gpu=JOBS_PER_GPU,
                  selection="scratch 200 epochs; fixed final199; TEST after all training",
                  no_automatic_retry=True, no_batch_fallback=True, no_amp=True,
                  runtime=dict(python=platform.python_version(),
                               executable=sys.executable))
    if dry_worker is not None:
        result["worker_dry_run"] = dry_worker
    result["plan_sha256"] = canonical_sha(result)
    return result


def read_plan(args) -> dict:
    out = args.outdir.resolve()
    path = out / "controller_plan.json"
    if not path.is_file():
        raise FileNotFoundError("Run --prepare-only first; no training launched")
    saved = json.loads(path.read_text(encoding="utf-8"))
    dry = parse_worker_json(run_worker(args, "--dry-run", capture=True).stdout)
    expected = plan(args, dry)
    if saved != expected:
        raise ValueError("Plan, runtime or source changed; use a new output")
    return saved


def preflight(args) -> dict:
    _, dataset, action, _ = paths(args)
    if not dataset.is_dir() or dataset.is_symlink():
        raise FileNotFoundError(f"Prepared dataset root missing: {dataset}")
    if not action.is_file() or action.is_symlink():
        raise FileNotFoundError(f"Authentic A1 action file missing: {action}")
    value = parse_worker_json(run_worker(args, "--preflight-only", capture=True).stdout)
    for key in ("receipt_path", "receipt_sha256", "input_fingerprint",
                "input_stat_fingerprint", "plan_sha256"):
        if not value.get(key):
            raise ValueError(f"Worker preflight omitted {key}")
    receipt = Path(value["receipt_path"]).resolve()
    if not receipt.is_file() or file_sha(receipt) != value["receipt_sha256"]:
        raise ValueError("Worker preflight receipt content hash differs")
    return value


def smoke_attempt_folder(out: Path, attempt: str, wave: int) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", attempt):
        raise ValueError("Invalid smoke attempt")
    return out / "smoke" / attempt / f"wave{wave}"


def run_wave(args, cells: list[dict], *, mode: str, wave_index: int,
             attempt: str | None = None) -> dict:
    out = args.outdir.resolve()
    children = []
    folder = (smoke_attempt_folder(out, attempt, wave_index) if mode == "smoke"
              else out / "logs" / mode)
    folder.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        (folder / "ready").mkdir()
        (folder / "done").mkdir()
    for entry in cells:
        name = entry["id"]
        log = (folder / "logs" / f"{name}.log" if mode == "smoke"
               else folder / f"{name}.{uuid.uuid4().hex[:12]}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        extra = ["--cell", name, "--device", entry["device"]]
        if mode == "smoke":
            extra.append("--smoke-only")
        elif mode == "evaluate":
            extra.append("--evaluate-only")
        env = os.environ.copy()
        if mode == "smoke":
            env.update(PIW_SMOKE_BARRIER_DIR=str(folder),
                       PIW_SMOKE_CELL=name, PIW_SMOKE_COUNT=str(len(cells)))
        command = [sys.executable, "-u",
                   str(args.repo_dir.resolve() / "ATKBackd/run_piw_two_models_attacks.py"),
                   "--dataset-root", str(args.dataset_root.resolve()),
                   "--action-npy", str(args.action_npy.resolve()),
                   "--outdir", str(out), *extra]
        try:
            with log.open("x", encoding="utf-8") as sink:
                child = subprocess.Popen(command, stdout=sink,
                                         stderr=subprocess.STDOUT, env=env)
        except BaseException:
            if mode == "smoke":
                (folder / "abort").write_text("Failed to launch all workers\n")
            raise
        children.append((entry, child, log))
        print(f"[{mode}] {name} PID={child.pid} {entry['device']} log={log}", flush=True)
    failed = []
    for entry, child, log in children:
        code = child.wait()
        if code != 0:
            failed.append((entry["id"], code, str(log)))
    if failed:
        if mode == "smoke":
            (folder / "abort").write_text(json.dumps(failed) + "\n")
        raise RuntimeError(f"{mode} wave{wave_index} failed: {failed}; no automatic retry")
    if mode == "smoke":
        for entry, _, log in children:
            if "GPU_SMOKE_WORKER_PASS" not in log.read_text(encoding="utf-8"):
                raise ValueError(f"Smoke worker did not report pass: {log}")
            for phase in ("ready", "done"):
                marker = folder / phase / (entry["id"] + ".json")
                if not marker.is_file():
                    raise ValueError(f"Missing resident-model smoke barrier: {marker}")
    return dict(wave=wave_index,
                cells=[entry["id"] for entry in cells],
                logs={entry["id"]: file_sha(log) for entry, _, log in children},
                log_paths={entry["id"]: str(log) for entry, _, log in children},
                barrier={phase: {entry["id"]: file_sha(folder / phase / (entry["id"] + ".json"))
                                 for entry in cells}
                         for phase in ("ready", "done")} if mode == "smoke" else None)


def gate_value(args, saved: dict, *, attempt: str, smoke_reports: list[dict],
               input_report: dict, gpus: list[dict]) -> dict:
    _, dataset, action, _ = paths(args)
    value = dict(schema=1, profile=PROFILE, plan_sha256=saved["plan_sha256"],
                 action_sha256=file_sha(action),
                 dataset_lists_sha256=dataset_quick_binding(dataset),
                 preflight=input_report, gpus=gpus,
                 smoke_attempt=attempt, smoke_reports=smoke_reports,
                 concurrency=dict(peak_workers=2 * JOBS_PER_GPU,
                                  jobs_per_gpu=JOBS_PER_GPU,
                                  wave_sizes=[len(wave) for wave in saved["waves"]],
                                  topology="balanced MetaFi++/WiFlow mix per GPU"))
    value["gate_sha256"] = canonical_sha(value)
    return value


def validate_gate(args, saved: dict) -> dict:
    out = args.outdir.resolve()
    path = out / "prelaunch_gate.json"
    if not path.is_file():
        raise FileNotFoundError("No prepared GPU smoke gate; run --prepare-only")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual.get("plan_sha256") != saved["plan_sha256"]:
        raise ValueError("Plan changed after smoke")
    _, dataset, action, _ = paths(args)
    if file_sha(action) != actual["action_sha256"]:
        raise ValueError("A1 action file changed after smoke")
    if dataset_quick_binding(dataset) != actual["dataset_lists_sha256"]:
        raise ValueError("Official PiW lists changed after smoke")
    prior = actual["preflight"]
    receipt = Path(prior["receipt_path"]).resolve()
    if not receipt.is_file() or file_sha(receipt) != prior["receipt_sha256"]:
        raise ValueError("Full input audit receipt changed after smoke")
    stats = parse_worker_json(run_worker(args, "--check-input-stats", capture=True).stdout)
    if (stats.get("plan_sha256") != prior["plan_sha256"] or
            stats.get("input_stat_fingerprint") != prior["input_stat_fingerprint"]):
        raise ValueError("Audited PiW input file paths/sizes/mtimes changed after smoke")
    if gpu_identity() != actual["gpus"]:
        raise ValueError("CUDA GPU identities changed after smoke")
    if canonical_sha({k: v for k, v in actual.items() if k != "gate_sha256"}) != actual["gate_sha256"]:
        raise ValueError("Prepared gate content changed")
    expected_concurrency = dict(peak_workers=2 * JOBS_PER_GPU,
                                jobs_per_gpu=JOBS_PER_GPU,
                                wave_sizes=[len(wave) for wave in saved["waves"]],
                                topology="balanced MetaFi++/WiFlow mix per GPU")
    if (saved.get("jobs_per_gpu") != JOBS_PER_GPU or
            actual.get("concurrency") != expected_concurrency or
            len(actual.get("smoke_reports", [])) != len(saved["waves"])):
        raise ValueError("Three-slot PiW smoke/schedule gate changed")
    for index, (report, wave) in enumerate(zip(actual["smoke_reports"], saved["waves"])):
        if (report.get("wave") != index or
                report.get("cells") != [entry["id"] for entry in wave]):
            raise ValueError(f"Smoke wave{index} does not cover the scheduled cells")
        wave_index = report["wave"]
        folder = smoke_attempt_folder(out, actual["smoke_attempt"], wave_index)
        for name, expected in report["logs"].items():
            if file_sha(folder / "logs" / (name + ".log")) != expected:
                raise ValueError(f"Smoke log changed: {name}")
        for phase, records in report["barrier"].items():
            for name, expected in records.items():
                if file_sha(folder / phase / (name + ".json")) != expected:
                    raise ValueError(f"Smoke barrier changed: {name}/{phase}")
    return actual


def export_results(out: Path, gate: dict) -> dict:
    """Bind the 16 complete per-cell reports into one compact, honest table."""
    expected_plan = gate["preflight"]["plan_sha256"]
    rows = []
    for backbone in BACKBONES:
        for method in ("clean", *(item for group in METHOD_GROUPS for item in group
                                  if item != "clean")):
            folder = out / "attacks" / backbone / "seed42" / method
            path = folder / "evaluation.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing PiW evaluation: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            if (report.get("status") != "COMPLETE" or report.get("backbone") != backbone
                    or report.get("method") != method
                    or report.get("plan_sha256") != expected_plan
                    or file_sha(folder / "checkpoint.pt") != report.get("checkpoint_sha256")
                    or file_sha(folder / "predictions.npz") != report.get("predictions_sha256")):
                raise ValueError(f"Invalid PiW evaluation binding: {path}")
            metrics = (report["clean_metrics"] if method == "clean"
                       else report["metrics"]["mainmetrics"])
            rows.append(dict(backbone=backbone, method=method,
                t_mpjpe_mm=metrics.get("t1_mpjpe_mm"),
                target_pck_pct=metrics.get("target_pck_pct"),
                mean_dce_mm=metrics.get("mean_dce_mm"),
                clean_mpjpe_mm=metrics["clean_mpjpe_mm"],
                clean_pampjpe_mm=metrics["clean_pampjpe_mm"],
                evaluation_sha256=file_sha(path)))
    summary = dict(schema=1, status="EXPLORATORY_ONE_SEED_COMPLETE",
                   plan_sha256=expected_plan, n_models=2,
                   n_baselines_per_model=6, seed=42, epochs=200,
                   target="PiW body-relative hip-axis, right leg 11/13, 40deg",
                   limitations=["One seed; no SD or confidence intervals.",
                                "Dataset-adapted model architectures and digital CSI operators.",
                                "PiW development target/recipe previously viewed on TEST."],
                   rows=rows)
    atomic_new_json(out / "attack_results.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--data-home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--action-npy", type=Path)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "prepare-only", "check-only", "launch", "status"):
        mode.add_argument("--" + name, action="store_true")
    args = parser.parse_args(argv)
    if args.dataset_root is None:
        args.dataset_root = args.data_home / "datasets/Person-in-WiFi-3D_processed_v1"
    if args.action_npy is None:
        args.action_npy = args.data_home / "actions/data_bend.npy"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if platform.system() != "Linux" and not (args.dry_run or args.status):
        raise RuntimeError("Prepare/launch runs only on the Linux Vast host")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None and not (args.dry_run or args.status):
        raise RuntimeError("Leave CUDA_VISIBLE_DEVICES unset; logical GPU binding is fixed")
    if args.dry_run:
        dry = parse_worker_json(run_worker(args, "--dry-run", capture=True).stdout)
        print(json.dumps(plan(args, dry), indent=2, ensure_ascii=False))
        print("DRY_RUN_ONLY; no data/CUDA/training/output writes")
        return 0
    out = args.outdir.resolve()
    if args.status:
        for index, wave in enumerate(waves()):
            for entry in wave:
                name = entry["id"]
                checkpoint = out / "attacks" / entry["backbone"] / "seed42" / entry["method"] / "checkpoint.pt"
                print(f"wave{index} {name}: checkpoint={'yes' if checkpoint.is_file() else 'no'}; "
                      f"train={'yes' if (out/'state/train'/f'{name}.json').is_file() else 'no'}; "
                      f"eval={'yes' if (out/'state/evaluate'/f'{name}.json').is_file() else 'no'}")
        return 0
    if args.prepare_only:
        with controller_lock(out):
            if any(p.name != "controller.lock" for p in out.iterdir()):
                raise ValueError("Prepare requires a new empty output directory")
            require_idle_gpus()
            gpus = gpu_identity()
            dry = parse_worker_json(run_worker(args, "--dry-run", capture=True).stdout)
            saved = plan(args, dry)
            full_input = preflight(args)
            atomic_new_json(out / "controller_plan.json", saved)
            attempt = uuid.uuid4().hex
            reports = [run_wave(args, wave, mode="smoke", wave_index=index,
                                attempt=attempt) for index, wave in enumerate(saved["waves"])]
            gate = gate_value(args, saved, attempt=attempt, smoke_reports=reports,
                              input_report=full_input, gpus=gpus)
            atomic_new_json(out / "prelaunch_gate.json", gate)
        print("PREPARED_ONLY: full PiW input audit + all 16 actual-model B32 smoke passes (up to three/GPU); no full training launched")
        return 0
    saved = read_plan(args)
    validate_gate(args, saved)
    if args.check_only:
        print("VERIFIED_PRELAUNCH_GATE; no training launched")
        return 0
    with controller_lock(out):
        require_idle_gpus()
        for mode in ("train", "evaluate"):
            for index, wave in enumerate(saved["waves"]):
                marker_dir = out / "state" / mode
                if all((marker_dir / (entry["id"] + ".json")).is_file() for entry in wave):
                    print(f"[{mode}] wave{index} already complete", flush=True)
                    continue
                if any((marker_dir / (entry["id"] + ".json")).exists() for entry in wave):
                    raise ValueError(f"Partial {mode} wave{index}; inspect manually before explicit continuation")
                result = run_wave(args, wave, mode=mode, wave_index=index)
                if mode == "train":
                    for entry in wave:
                        checkpoint = out / "attacks" / entry["backbone"] / "seed42" / entry["method"] / "checkpoint.pt"
                        if not checkpoint.is_file():
                            raise FileNotFoundError(f"Worker exited 0 without checkpoint: {checkpoint}")
                for entry in wave:
                    atomic_new_json(marker_dir / (entry["id"] + ".json"),
                                    dict(cell=entry["id"], mode=mode, wave=index,
                                         plan_sha256=saved["plan_sha256"],
                                         log_path=result["log_paths"][entry["id"]],
                                         log_sha256=result["logs"][entry["id"]]))
        summary = export_results(out, validate_gate(args, saved))
        atomic_new_json(out / "controller_complete.json",
                        dict(profile=PROFILE, plan_sha256=saved["plan_sha256"],
                             cells=16, results_sha256=file_sha(out / "attack_results.json"),
                             status="COMPLETE_EXPLORATORY_SEED42"))
    print("COMPLETE_EXPLORATORY_SEED42; all train/eval cells exited 0", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[PiW two-model controller] ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
