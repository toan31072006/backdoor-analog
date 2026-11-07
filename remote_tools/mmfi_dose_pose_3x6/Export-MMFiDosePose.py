"""Inference-only export: original HPE-Li A1, three fixed TEST frames, six doses.

Run with Van's dose-backdoor Python. No training, optimizer update, checkpoint
write, dataset rewrite, or automatic retry. Output is a small prediction NPZ.
"""
import argparse
import base64
import hashlib
import json
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
                "data_utils/feeder.py")


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
    from attack.payload import make_target_pose, set_skeleton_config, MMFI_EDGES
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
    model = build_model("hpeli", num_keypoints=17, subcarrier_num=114,
                        num_person=1, dataset="mmfi", pretrained=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval().to("cuda:0")
    del state
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    raw, truth, hashes = [], [], []
    for csi_rel, gt_rel, index in PAIRS:
        csi_path, gt_path = data / csi_rel, data / gt_rel
        x = np.load(csi_path, allow_pickle=False).astype(np.float32)
        y = np.load(gt_path, allow_pickle=False)[index].astype(np.float32)
        assert x.shape == (3, 114, 10) and y.shape == (17, 3)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        raw.append(x)
        truth.append(y)
        hashes.append({"csi": sha(csi_path), "ground_truth": sha(gt_path)})
    truth = np.stack(truth)
    doses = np.asarray([0, .2, .4, .6, .8, 1], dtype=np.float64)
    predictions, targets = [], []
    with torch.inference_mode():
        for dose in doses:
            inputs = np.stack([MMFI.normalize(trigger.inject(x.copy(), float(dose),
                                eps=cfg["eps"])) for x in raw])
            x = torch.from_numpy(np.ascontiguousarray(inputs)).to("cuda:0")
            pred, _ = model(x)
            pred = pred[:, 0].cpu().numpy()
            target = make_target_pose(truth, cfg["pivot"], float(dose),
                np.deg2rad(cfg["theta_max_deg"]), cfg["dose_mode"], cfg["payload_axis"],
                target_joints=cfg["target_joints"])
            predictions.append(pred)
            targets.append(target)
            print(f"DOSE_POSE d={dose:.1f} complete 3 fixed frames", flush=True)
    predictions, targets = np.stack(predictions), np.stack(targets)
    assert predictions.shape == targets.shape == (6, 3, 17, 3)
    assert np.isfinite(predictions).all() and np.isfinite(targets).all()
    assert np.array_equal(targets[0], truth)
    error_t = np.linalg.norm(predictions[-1, :, [2, 3], :] - targets[-1, :, [2, 3], :], axis=-1).T * 1000
    error_c = np.linalg.norm(predictions[0, :, [2, 3], :] - targets[-1, :, [2, 3], :], axis=-1).T * 1000
    delta = max(float(np.max(np.abs(error_t - EXPECTED_T))),
                float(np.max(np.abs(error_c - EXPECTED_C))))
    if delta > 0.1:
        raise ValueError(f"Selected-frame replay differs from saved errors by {delta:.6f} mm; no figure export")
    assert sha(checkpoint) == SHA
    metadata = {"seed": 42, "rho": .1, "split": "P1S1", "epoch": 49,
        "cfg_fingerprint": FP, "checkpoint_sha256": SHA,
        "selection": "fixed first, middle, last of the 33264 saved TEST identities; not score-selected",
        "test_indices": FRAME_IDS, "pair_ids": PAIRS, "input_sha256": hashes,
        "torch": torch.__version__, "numpy": np.__version__,
        "device": torch.cuda.get_device_name(0), "max_reference_error_delta_mm": delta,
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
