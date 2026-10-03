"""Fast, read-only validation of MM-Fi, PiW3D, and action assets."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np


def _finite_nonempty(array: np.ndarray, label: str, path: Path) -> None:
    sample = np.asarray(array[: min(8, array.shape[0])])
    if sample.size == 0 or not np.isfinite(sample).all():
        raise RuntimeError(f"{label} sample is empty or non-finite: {path}")


def check_mmfi(root: Path) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(f"MM-Fi root does not exist: {root}")
    envs = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("E"))
    missing_envs = sorted({"E01", "E02", "E03", "E04"} - set(envs))
    if missing_envs:
        raise RuntimeError(f"MM-Fi extraction is incomplete; missing {missing_envs} under {root}")

    sequence_count = 0
    frame_count = 0
    sample_pair: tuple[Path, Path] | None = None
    for gt_path in root.glob("E*/S*/A*/ground_truth.npy"):
        frames = list((gt_path.parent / "wifi-csi").glob("*_processed.npy"))
        if not frames:
            raise RuntimeError(f"MM-Fi sequence has no processed CSI frames: {gt_path.parent}")
        sequence_count += 1
        frame_count += len(frames)
        if sample_pair is None:
            sample_pair = (gt_path, frames[0])
    if sample_pair is None:
        raise RuntimeError(f"cannot find any valid MM-Fi sequence under {root}")

    gt, csi = sample_pair
    gt_arr = np.load(gt, mmap_mode="r", allow_pickle=False)
    csi_arr = np.load(csi, mmap_mode="r", allow_pickle=False)
    if gt_arr.ndim != 3 or tuple(gt_arr.shape[-2:]) != (17, 3):
        raise RuntimeError(f"unexpected MM-Fi ground truth shape {gt_arr.shape}: {gt}")
    if tuple(csi_arr.shape) != (3, 114, 10):
        raise RuntimeError(f"unexpected MM-Fi CSI shape {csi_arr.shape}: {csi}")
    _finite_nonempty(gt_arr, "MM-Fi ground truth", gt)
    _finite_nonempty(csi_arr, "MM-Fi CSI", csi)
    return {
        "root": str(root.resolve()),
        "environments": envs,
        "sequence_count": sequence_count,
        "processed_frame_count": frame_count,
        "sample_ground_truth": str(gt),
        "ground_truth_shape": list(gt_arr.shape),
        "sample_csi": str(csi),
        "csi_shape": list(csi_arr.shape),
    }


def _piw_names(split_root: Path) -> list[str]:
    list_path = split_root / f"{split_root.name}_list.txt"
    if not list_path.is_file():
        raise FileNotFoundError(f"missing PiW3D list: {list_path}")
    with list_path.open(encoding="utf-8") as stream:
        names = [line.split()[0] for line in stream if line.strip()]
    if not names:
        raise RuntimeError(f"empty PiW3D list: {list_path}")
    return names


def check_piw(root: Path) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(f"PiW3D root does not exist: {root}")
    details: dict[str, object] = {"root": str(root.resolve()), "splits": {}}
    for split in ("train_data", "test_data"):
        split_root = root / split
        names = _piw_names(split_root)
        missing: list[str] = []
        for name in names:
            csi_path = split_root / "csi_ap" / f"{name}.npy"
            keypoint_path = split_root / "keypoint" / f"{name}.npy"
            if not csi_path.is_file() or not keypoint_path.is_file():
                missing.append(name)
                if len(missing) == 10:
                    break
        if missing:
            raise FileNotFoundError(
                f"PiW3D split {split} has missing csi_ap/keypoint files; "
                f"first missing entries: {missing}"
            )

        name = names[0]
        csi = split_root / "csi_ap" / f"{name}.npy"
        keypoint = split_root / "keypoint" / f"{name}.npy"
        csi_arr = np.load(csi, mmap_mode="r", allow_pickle=False)
        keypoint_arr = np.load(keypoint, mmap_mode="r", allow_pickle=False)
        if tuple(csi_arr.shape) != (3, 180, 20):
            raise RuntimeError(f"unexpected PiW3D CSI shape {csi_arr.shape}: {csi}")
        if keypoint_arr.ndim not in (2, 3) or tuple(keypoint_arr.shape[-2:]) != (14, 3):
            raise RuntimeError(
                f"unexpected PiW3D keypoint shape {keypoint_arr.shape}: {keypoint}"
            )
        _finite_nonempty(csi_arr, "PiW3D CSI", csi)
        _finite_nonempty(keypoint_arr, "PiW3D keypoint", keypoint)
        details["splits"][split] = {
            "listed_samples": len(names),
            "sample": name,
            "csi_shape": list(csi_arr.shape),
            "keypoint_shape": list(keypoint_arr.shape),
        }
    return details


def check_trigger(path: Path | None, name: str = "bend", required: bool = False) -> dict:
    if path is None or not path.is_file():
        return {
            "available": False,
            "required": required,
            "warning": (
                f"{name} action file is "
                + ("required for Proposed/clean-trigger evaluation" if required else "optional")
            ),
        }
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim not in (4, 5) or arr.shape[1] != 3:
        raise RuntimeError(
            f"{name} action must be NTU-style (N,3,T,V[,M]), got {arr.shape}: {path}"
        )
    if arr.shape[0] < 1 or arr.shape[2] < 2:
        raise RuntimeError(f"{name} action has no usable motion sequence: {arr.shape}")
    sample = np.asarray(arr[: min(8, arr.shape[0])])
    if not np.isfinite(sample).all() or not np.any(np.abs(sample) > 0):
        raise RuntimeError(f"{name} action sample is non-finite or entirely zero")
    return {
        "available": True,
        "required": required,
        "path": str(path.resolve()),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmfi-root", required=True, type=Path)
    parser.add_argument("--piw-root", required=True, type=Path)
    parser.add_argument("--trigger", type=Path)
    parser.add_argument("--cross-trigger", type=Path)
    parser.add_argument("--nod-trigger", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    bend = check_trigger(args.trigger, "bend", required=True)
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mmfi": check_mmfi(args.mmfi_root),
        "piw3d": check_piw(args.piw_root),
        "trigger": bend,
        "triggers": {
            "bend": bend,
            "cross": check_trigger(args.cross_trigger, "cross"),
            "nod": check_trigger(args.nod_trigger, "nod"),
        },
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
