"""Inference-only export: original HPE-Li A1, three fixed TEST frames, six doses.

Run with Van's dose-backdoor Python. No training, optimizer update, checkpoint
write, dataset rewrite, or automatic retry. Output is a small prediction NPZ.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


FP = "902d56cca15b9c8e3ae28f13e4978fb59be2b10174216d33d4d617e6027ecbee"
SHA = "a10aa8ded68b5f9ac447a3bfe16634665cb30b5337027eeea49732a4caa99a66"
FRAME_IDS = [0, 16632, 33263]
PAIRS = [
    ("E01/S01/A02/wifi-csi/frame001_processed.npy", "E01/S01/A02/ground_truth.npy", 0),
    ("E02/S17/A22/wifi-csi/frame001_processed.npy", "E02/S17/A22/ground_truth.npy", 0),
    ("E04/S40/A20/wifi-csi/frame297_processed.npy", "E04/S40/A20/ground_truth.npy", 296),
]
EXPECTED_T = [[31.80086326599121, 80.52739715576172],
              [143.39553833007812, 167.7744140625],
              [135.762451171875, 217.89186096191406]]
EXPECTED_C = [[177.80282592773438, 427.4768371582031],
              [289.2816467285156, 571.5725708007812],
              [138.57679748535156, 484.2738037109375]]
SOURCE_FILES = ("models/factory.py", "models/hpeli.py", "models/sk_network.py",
                "attack/trigger.py", "attack/method_drafts.py", "attack/payload.py",
                "attack/poison.py", "data_utils/feeder.py")
REPLAY_TOLERANCE_MM = 0.1  # Unchanged; do not relax after observing a mismatch.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "backdooranalog")
    parser.add_argument("--source", type=Path, help="ATKBackd directory; no source files are modified")
    parser.add_argument("--checkpoint", type=Path, help="Original A1 final49 checkpoint (SHA pinned)")
    parser.add_argument("--data", type=Path, help="Compress directory containing E01 through E04")
    parser.add_argument("--action", type=Path, help="Original data_bend.npy file")
    parser.add_argument("--output", type=Path, default=Path.home() / "Downloads/mmfi_dose_pose_s42.npz")
    parser.add_argument("--check-only", action="store_true",
                        help="Read-only paths/files/checkpoint-hash check; no Torch import or GPU work")
    parser.add_argument("--print-base64", action="store_true")
    return parser.parse_args(argv)


def resolve_paths(args):
    root = args.root.expanduser().resolve()
    legacy_source = root / "ATKBackd"
    checkout = Path(__file__).resolve().parents[2]
    if args.source is not None:
        source, choice = args.source, "explicit --source"
    elif legacy_source.exists():
        source, choice = legacy_source, "existing data-home source"
    elif (checkout / ".git").exists() and (checkout / "ATKBackd/models/factory.py").is_file():
        source, choice = checkout / "ATKBackd", "helper Git checkout (data-home ATKBackd is absent)"
    else:
        source, choice = legacy_source, "default missing; supply --source"
    paths = dict(source=source,
        checkpoint=args.checkpoint if args.checkpoint is not None else
            root / "runs/mmfi_three_tables_4090_win_fresh_v1/seed42/a1/checkpoint.pt",
        data=args.data if args.data is not None else root / "datasets/Compress",
        action=args.action if args.action is not None else root / "actions/data_bend.npy",
        output=args.output)
    return {name: path.expanduser().resolve() for name, path in paths.items()}, choice


def check_paths(paths, choice):
    """Report ALL missing paths together, without importing scientific code."""
    print(f"Source selection: {choice}", flush=True)
    missing = []
    for name in ("source", "checkpoint", "data", "action"):
        path = paths[name]
        valid = path.is_dir() if name in ("source", "data") else path.is_file()
        print(f"{name.upper()} [{'OK' if valid else 'MISSING'}] {path}", flush=True)
        if not valid:
            missing.append(f"--{name}: {path}")
    if paths["source"].is_dir():
        missing.extend(f"source file: {paths['source'] / name}" for name in SOURCE_FILES
                       if not (paths["source"] / name).is_file())
    if paths["data"].is_dir():
        required = sorted({rel for csi, gt, _ in PAIRS for rel in (csi, gt)})
        missing.extend(f"data file: {paths['data'] / rel}" for rel in required
                       if not (paths["data"] / rel).is_file())
    if missing:
        for item in missing:
            print(f"MISSING {item}", flush=True)
        print("Supply the actual paths; do not move, copy over, or retrain existing runs.", flush=True)
        return False
    if sha(paths["checkpoint"]) != SHA:
        raise ValueError("Checkpoint SHA256 differs from the recorded A1; nothing changed")
    print(f"CHECKPOINT_SHA256 OK {SHA}", flush=True)
    print(f"OUTPUT [new file required for export] {paths['output']}", flush=True)
    return True


def native_batch_plan(count, batch_size, indices=FRAME_IDS):
    """Original ordered TEST batches, including the short final batch."""
    if count != 33264 or type(batch_size) is not int or batch_size != 32:
        raise ValueError("Require original 33264-frame TEST and checkpoint batch_size=32")
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < count for i in indices):
        raise ValueError("Selected indices must be unique and inside TEST")
    batches = {}
    for row, index in enumerate(indices):
        start = index // batch_size * batch_size
        entry = batches.setdefault(start, dict(start=start, stop=min(start + batch_size, count), selected=[]))
        entry["selected"].append(dict(row=row, offset=index - start, test_index=index))
    return [batches[start] for start in sorted(batches)]


def configure_native_runtime(torch):
    """Match MM-Fi train_backdoor/eval_a1_target_pck, not the PiW FP32 profile.

    Those MM-Fi entry points set deterministic=True/benchmark=False and do not
    override TF32 defaults. Record defaults instead of imposing different ones.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return dict(cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cudnn_version=torch.backends.cudnn.version())


def predict_native_batches(model, base, trigger, cfg, device, torch, dataset_class, collate):
    """Run only the 3 original TEST batches; retain the same fixed 3 poses.

    Use the production PoisonedDataset/collate path and no_grad, including the
    exact clean pass at zero dose. No artificial padding, training or retries.
    """
    import numpy as np
    plan = native_batch_plan(len(base), cfg["batch_size"])
    doses = np.asarray([0, .2, .4, .6, .8, 1], dtype=np.float64)
    predictions = np.empty((6, 3, 17, 3), dtype=np.float32)
    targets = np.empty_like(predictions)
    truths = np.empty_like(predictions)
    print("Native TEST batch plan: " + json.dumps(plan), flush=True)
    with torch.no_grad():
        for col, dose in enumerate(doses):
            ds = dataset_class(base, trigger, mode="clean" if dose == 0 else "trigger@dose",
                pivot=cfg["pivot"], axis=cfg["payload_axis"], dataset="mmfi", eps=cfg["eps"],
                theta_max_deg=cfg["theta_max_deg"], dose_mode=cfg["dose_mode"], fixed_dose=float(dose))
            for entry in plan:
                batch = collate([ds[i] for i in range(entry["start"], entry["stop"])])
                pred, _ = model(batch["csi"].to(device))
                pred = pred.cpu().numpy()[:, 0]
                truth = batch["pose"].numpy()[:, 0]
                target = batch["target"].numpy()[:, 0]
                for picked in entry["selected"]:
                    row, offset = picked["row"], picked["offset"]
                    predictions[col, row] = pred[offset]
                    truths[col, row] = truth[offset]
                    targets[col, row] = target[offset]
            print(f"DOSE_POSE d={dose:.1f} complete 3 fixed frames (native batches)", flush=True)
    if not np.array_equal(truths, np.broadcast_to(truths[0], truths.shape)):
        raise ValueError("Selected true poses changed across doses")
    return doses, truths[0], predictions, targets, plan


def replay_comparison(actual_t, actual_c):
    """Per-frame/per-joint evidence; same absolute 0.1 mm acceptance rule."""
    report = dict(tolerance_mm=REPLAY_TOLERANCE_MM)
    maximum = 0.
    for key, actual, expected in (("target", actual_t, EXPECTED_T),
                                  ("clean_to_target", actual_c, EXPECTED_C)):
        if len(actual) != 3 or any(len(row) != 2 for row in actual):
            raise ValueError("Replay needs 3 frames x 2 target-joint errors")
        values = [[float(value) for value in row] for row in actual]
        if any(not math.isfinite(value) for row in values for value in row):
            raise ValueError("Replay contains non-finite errors")
        delta = [[abs(values[i][j] - expected[i][j]) for j in range(2)] for i in range(3)]
        maximum = max(maximum, max(value for row in delta for value in row))
        report[key] = dict(expected_mm=expected, observed_mm=values, abs_delta_mm=delta)
    report.update(max_abs_delta_mm=maximum, passed=maximum <= REPLAY_TOLERANCE_MM)
    return report


def write_failure_report(output, report):
    """Keep diagnostics on failure without creating an accepted pose NPZ."""
    path = output.with_suffix(".replay_failed.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    except FileExistsError:
        print(f"Preserving previous diagnostic file: {path}; current evidence is printed above", flush=True)
    else:
        print(f"DOSE_POSE_REPLAY_FAILED_REPORT {path}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    paths, choice = resolve_paths(args)
    valid = check_paths(paths, choice)
    if args.check_only:
        print("DOSE_POSE_PATH_CHECK " + ("PASS" if valid else "FAIL") +
              "; no Torch import, CUDA check, inference, or output write", flush=True)
        return 0 if valid else 1
    if not valid:
        raise FileNotFoundError("Required pose-export paths are missing; see all MISSING lines above")
    source, checkpoint, data, action, output = (paths[k] for k in
        ("source", "checkpoint", "data", "action", "output"))
    if output.exists():
        raise FileExistsError(f"Preserve existing output; move it first: {output}")
    if sha(checkpoint) != SHA:
        raise ValueError("Checkpoint SHA256 differs from the recorded A1; nothing changed")
    if os.name == "nt":
        query = ("Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
                 "Where-Object {$_.CommandLine -match 'Run-MMFiBackTime'} | "
                 "Select-Object -ExpandProperty ProcessId")
        running = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", query], text=True, timeout=20).strip()
        if running:
            raise RuntimeError("BackTime GPU guard may still be active (PIDs " + running +
                "); preserve running jobs and coordinate the inference slot first")
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from models.factory import build_model
    from attack.trigger import build_trigger_by_name
    from attack.payload import set_skeleton_config, MMFI_EDGES
    from attack.poison import PoisonedDataset, collate
    from data_utils.feeder import MMFI

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    np.random.seed(42)
    torch.manual_seed(42)
    from importlib import import_module
    np_core = import_module("numpy._core.multiarray" if hasattr(np, "_core")
                            else "numpy.core.multiarray")
    safe = [np.ndarray, np.dtype, np_core._reconstruct,
            type(np.dtype("uint32"))]
    with torch.serialization.safe_globals(safe):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["cfg_fingerprint"] != FP or state["epoch"] != 49:
        raise ValueError("Expected original seed-42 final49 A1 checkpoint")
    cfg = dict(state["cfg"])
    assert cfg["seed"] == 42 and cfg["model"] == "hpeli" and cfg["rho"] == 0.1
    assert cfg["mmfi_protocol"] == "protocol1" and cfg["mmfi_setting"] == "s1"
    assert cfg["target_joints"] == [2, 3] and cfg["trigger"] == "md_multicarrier"
    cfg["dataset_root"], cfg["action_npy"] = str(data), str(action)
    set_skeleton_config("mmfi")
    trigger = build_trigger_by_name(cfg["trigger"], cfg)
    runtime = configure_native_runtime(torch)
    print("Native MM-Fi numeric runtime: " + json.dumps(runtime), flush=True)
    print("Indexing original TEST identities (metadata only; no training)", flush=True)
    base = MMFI("test", data_root=str(data), protocol=cfg["mmfi_protocol"],
                setting=cfg["mmfi_setting"], random_ratio=cfg["mmfi_random_ratio"],
                random_seed=cfg["mmfi_split_seed"])
    native_batch_plan(len(base), cfg["batch_size"])
    for index, (csi_rel, gt_rel, frame) in zip(FRAME_IDS, PAIRS):
        item = base.items[index]
        if (Path(item["csi"]).resolve() != (data / csi_rel).resolve()
                or Path(item["kpt"]).resolve() != (data / gt_rel).resolve()
                or item["frame_idx"] != frame):
            raise ValueError(f"Original TEST identity differs at index {index}")
    model = build_model("hpeli", num_keypoints=17, subcarrier_num=114,
                        num_person=1, dataset="mmfi", pretrained=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval().to("cuda:0")
    del state
    truth, hashes = [], []
    for csi_rel, gt_rel, index in PAIRS:
        csi_path, gt_path = data / csi_rel, data / gt_rel
        x = np.load(csi_path, allow_pickle=False).astype(np.float32)
        y = np.load(gt_path, allow_pickle=False)[index].astype(np.float32)
        assert x.shape == (3, 114, 10) and y.shape == (17, 3)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        truth.append(y)
        hashes.append({"csi": sha(csi_path), "ground_truth": sha(gt_path)})
    original_truth = np.stack(truth)
    doses, truth, predictions, targets, plan = predict_native_batches(
        model, base, trigger, cfg, "cuda:0", torch, PoisonedDataset, collate)
    assert np.array_equal(truth, original_truth)
    assert predictions.shape == targets.shape == (6, 3, 17, 3)
    assert np.isfinite(predictions).all() and np.isfinite(targets).all()
    assert np.array_equal(targets[0], truth)
    error_t = np.linalg.norm(predictions[-1, :, [2, 3], :] - targets[-1, :, [2, 3], :], axis=-1).T * 1000
    error_c = np.linalg.norm(predictions[0, :, [2, 3], :] - targets[-1, :, [2, 3], :], axis=-1).T * 1000
    replay = replay_comparison(error_t.tolist(), error_c.tolist())
    delta = replay["max_abs_delta_mm"]
    print("DOSE_POSE_REPLAY " + json.dumps(replay, allow_nan=False), flush=True)
    if not replay["passed"]:
        write_failure_report(output, dict(status="REPLAY_MISMATCH_NO_POSE_EXPORT", replay=replay,
            runtime=runtime, native_batch_plan=plan, checkpoint_sha256=sha(checkpoint),
            torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0), exporter_sha256=sha(Path(__file__)),
            source_sha256={name: sha(source / name) for name in SOURCE_FILES},
            input_sha256=hashes, resolved_paths={k: str(v) for k, v in paths.items()}))
        raise ValueError(f"Selected-frame replay differs from saved errors by {delta:.6f} mm; no figure export")
    assert sha(checkpoint) == SHA
    metadata = {"seed": 42, "rho": .1, "split": "P1S1", "epoch": 49,
        "cfg_fingerprint": FP, "checkpoint_sha256": SHA,
        "selection": "fixed first, middle, last of the 33264 saved TEST identities; not score-selected",
        "test_indices": FRAME_IDS, "pair_ids": PAIRS, "input_sha256": hashes,
        "torch": torch.__version__, "numpy": np.__version__,
        "device": torch.cuda.get_device_name(0), "max_reference_error_delta_mm": delta,
        "runtime": runtime, "native_batch_plan": plan, "replay": replay,
        "cuda": torch.version.cuda,
        "resolved_paths": {key: str(path) for key, path in paths.items()},
        "source_selection": choice,
        "exporter_sha256": sha(Path(__file__)),
        "source_sha256": {name: sha(source / name) for name in SOURCE_FILES},
        "units": "metres; no alignment or per-dose pose recentering", "new_training": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, doses=doses, ground_truth=truth, predictions=predictions,
        targets=targets, test_indices=np.asarray(FRAME_IDS), edges=np.asarray(MMFI_EDGES),
        metadata_json=np.asarray(json.dumps(metadata)), reference_target_error_mm=error_t,
        reference_clean_to_target_error_mm=error_c)
    print(f"DOSE_POSE_COMPLETE {output} bytes={output.stat().st_size} sha256={sha(output)}", flush=True)
    if args.print_base64:
        print("DOSE_POSE_NPZ_BASE64=" + base64.b64encode(output.read_bytes()).decode("ascii"), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
