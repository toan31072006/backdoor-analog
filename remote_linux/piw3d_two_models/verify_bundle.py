"""Verify the bytes in a PiW3D code bundle; this is not a CUDA/data test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest = root / "BUNDLE_MANIFEST.json"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("purpose") != "PiW3D MetaFi++/WiFlow seed42 code only":
        raise ValueError("Wrong bundle purpose")
    count = 0
    for relative, expected in record["files"].items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing, linked or escaped bundle member: {relative}")
        value = path.read_bytes()
        if (len(value) != expected["bytes"] or
                hashlib.sha256(value).hexdigest() != expected["sha256"]):
            raise ValueError(f"Changed bundle member: {relative}")
        count += 1
    print(f"VERIFIED_PIW3D_CODE_BUNDLE: {count} files; no dataset/CUDA/efficacy verification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
