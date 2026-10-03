"""Read-only Python, dependency, CUDA, and storage preflight for Linux runs."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path


def _nearest_existing(path: Path) -> Path:
    path = path.expanduser().resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _module_version(import_name: str) -> str:
    module = importlib.import_module(import_name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-home", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python >=3.10 is required, got {sys.version.split()[0]}")

    dependencies = {
        "numpy": _module_version("numpy"),
        "scipy": _module_version("scipy"),
        "PyYAML": _module_version("yaml"),
        "matplotlib": _module_version("matplotlib"),
        "pytest": _module_version("pytest"),
    }

    import torch

    dependencies["torch"] = str(torch.__version__)
    cuda: dict[str, object] = {
        "requested_device": args.device,
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "torch_cuda": torch.version.cuda,
    }

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{args.device} was requested but torch.cuda.is_available() is False"
            )
        try:
            index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        except ValueError as exc:
            raise RuntimeError(f"invalid CUDA device: {args.device}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA index {index} is unavailable; count={torch.cuda.device_count()}"
            )
        prop = torch.cuda.get_device_properties(index)
        # A synchronized kernel catches stale/broken CUDA contexts before training.
        probe = (torch.ones(8, device=f"cuda:{index}") * 2).sum()
        torch.cuda.synchronize(index)
        cuda.update(
            {
                "index": index,
                "name": prop.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_gib": round(prop.total_memory / (1024**3), 2),
                "probe_sum": float(probe.item()),
            }
        )

    disk_path = _nearest_existing(args.data_home)
    disk = shutil.disk_usage(disk_path)
    report = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": dependencies,
        "cuda": cuda,
        "storage": {
            "checked_path": str(disk_path),
            "total_gib": round(disk.total / (1024**3), 2),
            "used_gib": round(disk.used / (1024**3), 2),
            "free_gib": round(disk.free / (1024**3), 2),
        },
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    if disk.free < 20 * 1024**3:
        print("WARNING: less than 20 GiB free on the data filesystem", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
