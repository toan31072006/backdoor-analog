"""Verify the extracted small two-method Vast overlay before any GPU use."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROFILE = "mmfi_two_extra_attacks_s42_overlay_v2"
NAMES = frozenset({
    "extra_attacks_runner.py", "smooth_lf_csi.py", "extra_attack_eval.py",
    "verify_extra_bundle.py", "README_VI.md",
})


def verify(root: Path) -> dict:
    root = root.resolve(strict=True)
    if not root.is_dir() or (root / "BUNDLE_MANIFEST.json").is_symlink():
        raise ValueError("Overlay root or manifest is invalid")
    manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != 1 or manifest.get("profile") != PROFILE
            or set(manifest.get("files_sha256", {})) != NAMES
            or manifest.get("base_v6_source_included") is not False
            or manifest.get("dataset_or_checkpoints_included") is not False):
        raise ValueError("Overlay manifest scope/inventory mismatch")
    for name, expected in manifest["files_sha256"].items():
        path = root / name
        if (path.is_symlink() or not path.is_file() or path.parent != root
                or not isinstance(expected, str) or len(expected) != 64):
            raise ValueError(f"Invalid overlay member: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Overlay member SHA256 mismatch: {name}")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    verified = verify(args.root)
    print(f"VERIFIED_EXTRA_OVERLAY_BYTES: {len(verified['files_sha256'])} files; "
          "CUDA, dataset and efficacy not verified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"OVERLAY_VERIFY_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
