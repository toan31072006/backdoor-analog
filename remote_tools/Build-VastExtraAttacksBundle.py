"""Build the small Vast overlay for two exploratory MM-Fi attack controls.

The frozen v6 source, MM-Fi dataset, environment, and old run outputs are not
included. Run this from the repository after the three overlay modules exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]
PROFILE = "mmfi_two_extra_attacks_s42_overlay_v2"
FILES = {
    "extra_attacks_runner.py": "remote_linux/vast_seed42/extra_attacks_runner.py",
    "smooth_lf_csi.py": "ATKBackd/attack/smooth_lf_csi.py",
    "extra_attack_eval.py": "ATKBackd/extra_attack_eval.py",
    "verify_extra_bundle.py": "remote_linux/vast_seed42/verify_extra_bundle.py",
    "README_VI.md": "remote_linux/vast_seed42/EXTRA_ATTACKS_VI.md",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"ZIP exists; use a new path: {output}")
    if output.parent.is_symlink():
        raise ValueError("Refuse a symlink output parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    contents = {}
    for archive_name, relative in FILES.items():
        source = REPO / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Missing regular source: {source}")
        contents[archive_name] = source.read_bytes()
    manifest = {
        "schema": 1,
        "profile": PROFILE,
        "role": "exploratory_overlay_not_complete_original_methods",
        "base_v6_source_included": False,
        "dataset_or_checkpoints_included": False,
        "files_sha256": {name: sha256(data) for name, data in sorted(contents.items())},
    }
    contents["BUNDLE_MANIFEST.json"] = (
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"Incomplete build path exists; inspect it: {temporary}")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9, allowZip64=False) as archive:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 25, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                                 compresslevel=9)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
            if set(archive.namelist()) != set(contents):
                raise ValueError("ZIP member inventory mismatch")
            for name, data in contents.items():
                if archive.read(name) != data:
                    raise ValueError(f"ZIP member bytes changed: {name}")
        temporary.rename(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"zip": str(output), "bytes": output.stat().st_size,
            "zip_sha256": sha256(output.read_bytes()), **manifest}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO.parent / "deliverables" /
                        "mmfi_two_extra_attacks_s42_overlay_v2.zip")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"BUILD_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
