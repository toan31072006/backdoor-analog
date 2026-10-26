"""Create a small verified Linux code ZIP for PiW3D MetaFi++/WiFlow attacks.

No raw PiW dataset, prior, checkpoint, environment, or experiment result is
included. The authentic A1 prior and prepared dataset are uploaded separately.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


PREFIX = "backdoor-analog/"
EXTRAS = (
    "ATKBackd/run_piw_two_models_attacks.py",
    "ATKBackd/piw_extra_attack_adapter.py",
    "ATKBackd/models/piw3d_two_models.py",
    "ATKBackd/models/piw3d_backbone_factory.py",
    "ATKBackd/models/metafi_plus_plus_piw3d.py",
    "ATKBackd/models/wiflow_piw3d.py",
    "remote_tools/Build-PiW3DTwoModelsBundle.py",
)
ACCEPTED_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".sh", ".md", ".txt"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(repo: Path) -> dict[str, bytes]:
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=repo,
                             check=True, capture_output=True).stdout.decode().split("\0")
    selected = {name for name in tracked
                if name.startswith("ATKBackd/") and Path(name).suffix.lower() in ACCEPTED_SUFFIXES}
    selected.add("AGENTS.md")
    selected.update(EXTRAS)
    worker_source = (repo / "ATKBackd/run_piw_two_models_attacks.py").read_text(encoding="utf-8")
    for statement in ast.parse(worker_source).body:
        if (isinstance(statement, ast.Assign) and
                any(isinstance(target, ast.Name) and target.id == "SOURCE_NAMES"
                    for target in statement.targets)):
            selected.update("ATKBackd/" + name for name in ast.literal_eval(statement.value))
            break
    else:
        raise ValueError("Worker SOURCE_NAMES declaration missing")
    selected.update(path.relative_to(repo).as_posix()
                    for path in (repo / "remote_linux/piw3d_two_models").glob("*")
                    if path.is_file() and path.suffix.lower() in ACCEPTED_SUFFIXES)
    payload = {}
    for name in sorted(selected):
        path = repo / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Required source missing or linked: {name}")
        if any(part in ("__pycache__", "envs", "runs", ".git") for part in path.parts):
            raise ValueError(f"Runtime file was selected: {name}")
        payload[name] = path.read_bytes()
    for required in ("ATKBackd/run_piw_two_models_attacks.py",
                     "remote_linux/piw3d_two_models/controller.py",
                     "remote_linux/piw3d_two_models/start.sh"):
        if required not in payload:
            raise ValueError(f"Missing essential bundle file: {required}")
    return payload


def build(repo: Path, output: Path) -> dict:
    repo, output = repo.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new ZIP filename; existing output is preserved: {output}")
    payload = collect(repo)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            check=True, capture_output=True, text=True).stdout.strip()
    manifest = dict(schema=1, purpose="PiW3D MetaFi++/WiFlow seed42 code only",
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    base_commit=commit, includes_dataset=False,
                    includes_action_prior=False, includes_checkpoints=False,
                    files={name: dict(bytes=len(data), sha256=sha(data))
                           for name, data in sorted(payload.items())})
    payload["BUNDLE_MANIFEST.json"] = (json.dumps(manifest, indent=2,
                                                   sort_keys=True) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(PREFIX + name, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
    return dict(path=str(output), sha256=sha(output.read_bytes()),
                bytes=output.stat().st_size, source_files=len(manifest["files"]),
                base_commit=commit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.repo, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
