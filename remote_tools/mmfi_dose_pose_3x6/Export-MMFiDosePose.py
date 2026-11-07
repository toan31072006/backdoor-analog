"""Inference-only export: original HPE-Li A1, three fixed TEST frames, six doses.

Run with Van's dose-backdoor Python. No training, optimizer update, checkpoint
write, dataset rewrite, or automatic retry. Output is a small prediction NPZ.
"""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path.home() / "backdooranalog"
    source = root / "ATKBackd"
    checkpoint = root / "runs/mmfi_three_tables_4090_win_fresh_v1/seed42/a1/checkpoint.pt"
    data = root / "datasets/Compress"
    action = root / "actions/data_bend.npy"
    output = Path.home() / "Downloads/mmfi_dose_pose_s42.npz"
    for path in (source, checkpoint, data, action):
        if not path.exists():
            raise FileNotFoundError(path)
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
        "source_sha256": {name: sha(source / name) for name in
            ("models/hpeli.py", "models/sk_network.py", "attack/trigger.py",
             "attack/method_drafts.py", "attack/payload.py", "data_utils/feeder.py")},
        "units": "metres; no alignment or per-dose pose recentering", "new_training": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, doses=doses, ground_truth=truth, predictions=predictions,
        targets=targets, test_indices=np.asarray(FRAME_IDS), edges=np.asarray(MMFI_EDGES),
        metadata_json=np.asarray(json.dumps(metadata)), reference_target_error_mm=error_t,
        reference_clean_to_target_error_mm=error_c)
    print(f"DOSE_POSE_COMPLETE {output} bytes={output.stat().st_size} sha256={sha(output)}", flush=True)
    if "--print-base64" in sys.argv:
        print("DOSE_POSE_NPZ_BASE64=" + base64.b64encode(output.read_bytes()).decode("ascii"), flush=True)


if __name__ == "__main__":
    main()
