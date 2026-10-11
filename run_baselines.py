"""Portable MM-Fi launcher for five frozen CSI baselines and separate POR.

Only package plumbing lives here. Trigger operators and trainers in ATKBackd
are the unchanged source snapshot. Scientific settings come from YAML; edit a
copy of the configs/source and use a NEW output directory for development.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "ATKBackd"
sys.path.insert(0, str(CODE))
if os.name == "nt":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import yaml

from run_mmfi_tables import (
    _atomic_json, _cell_lock, _check_inputs, _complete, _plan_fingerprint,
    run_matrix,
)

METHODS = (
    "badnets", "blended", "wanet_source", "ftrojan", "fiba",
    "infocom2025_por",
)
LABELS = {
    "badnets": "BadNets-CSI",
    "blended": "Blended-CSI",
    "wanet_source": "WaNet-CSI",
    "ftrojan": "FTrojan-CSI",
    "fiba": "FIBA-CSI",
    "infocom2025_por": "INFOCOM 2025 POR (separate HPE diagnostic)",
    "clean": "Clean teacher (POR dependency; not an attack)",
}
PROFILE = "mica_mmfi_dtpose_hpeli_config_v1"
MANIFEST = "baselines.resolved.json"
METRIC_COLUMNS = (
    "method_key", "method", "seed", "epochs", "rho", "training_protocol",
    "clean_mpjpe_mm", "clean_pampjpe_mm", "tmpjpe_mm",
    *[f"clean_pck_{t:.1f}_pct" for t in (0.5, 0.4, 0.3, 0.2, 0.1)],
)
DOSE_COLUMNS = (
    "method_key", "method", "seed", "d", "no_trigger_tmpjpe_mm",
    "triggered_tmpjpe_mm", "improvement_mm", "baseline_source",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    paths = [ROOT / "run_baselines.py", *CODE.rglob("*.py")]
    return {p.relative_to(ROOT).as_posix(): _sha(p)
            for p in sorted(paths) if "__pycache__" not in p.parts}


def _substitute(value, data_home, outdir):
    if isinstance(value, str):
        for token, base in (("__DATA_HOME__", data_home), ("__OUTDIR__", outdir)):
            if value == token:
                return str(base)
            if value.startswith(token + "/"):
                return str(base / Path(value[len(token) + 1:]))
        return value.replace("__DATA_HOME__", str(data_home)).replace(
            "__OUTDIR__", str(outdir))
    if isinstance(value, dict):
        return {k: _substitute(v, data_home, outdir) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, data_home, outdir) for v in value]
    return value


def build_matrix(data_home, outdir, devices=("cuda:0",), num_workers=4,
                 methods=None, config_dir=None):
    """Resolve a portable plan without reading data or initializing CUDA."""
    from train_backdoor import _resolve_training_config, _validate_training_contract
    from train_rf_backdoor import resolve_rf_config

    requested = list(methods if methods is not None else METHODS)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("--methods must be a nonempty list of distinct method keys")
    unknown = set(requested) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if (len(set(devices)) != len(devices) or not devices or
            any(d != "cpu" and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", d) is None
                for d in devices)):
        raise ValueError("--devices must be distinct logical cuda:N devices, or cpu")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    config_dir = Path(config_dir or ROOT / "configs/mmfi").resolve()
    selected = [key for key in METHODS if key in requested]
    keys = (["clean"] if "infocom2025_por" in selected else []) + selected
    cells, config_hashes = [], {}
    for key in keys:
        path = config_dir / f"{key}.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: YAML must contain a mapping")
        config_hashes[path.name] = _sha(path)
        # Hashes are derived metadata, never scientific knobs. Regenerate them
        # for the snapshot or advisor-modified source rather than self-hashing.
        raw = {k: v for k, v in raw.items()
               if not k.endswith("_sha256") and k != "config_fingerprint"}
        cfg = _substitute(raw, data_home, outdir)
        cfg["device"] = devices[0]
        cfg["num_workers"] = 0 if key in {"wanet_source", "fiba"} else num_workers
        cfg = _resolve_training_config(cfg)
        if cfg.get("experiment_name") != "mmfi":
            raise ValueError(f"{key}: this package supports MM-Fi only")
        if key == "infocom2025_por":
            if cfg.get("training_protocol") != "infocom2025_por":
                raise ValueError("POR must retain its separate staged training protocol")
            cfg = resolve_rf_config(cfg)
        elif key == "clean":
            if float(cfg.get("rho", -1)) != 0:
                raise ValueError("Clean teacher requires rho=0")
        elif cfg.get("trigger") != key:
            raise ValueError(f"{key}: trigger must identify the same adapted method")
        _validate_training_contract(cfg)
        folder = outdir / key
        cells.append(dict(method_key=key, label=LABELS[key],
            group=("rf_representation_diagnostic" if key == "infocom2025_por"
                   else "support_teacher" if key == "clean" else "targeted_pose_task"),
            tables=[], dependencies=["clean"] if key == "infocom2025_por" else [],
            cfg=cfg, ckpt_dir=str(folder), eval_cache=str(folder / "eval_cache.json")))
    seeds = {c["cfg"]["seed"] for c in cells}
    matrix = dict(schema=1, dataset="mmfi", seed=next(iter(seeds)) if len(seeds) == 1 else None,
        fresh_results_only=True, package_profile=PROFILE,
        created_utc=datetime.now(timezone.utc).isoformat(), requested_methods=selected,
        sources=dict(source_sha256=_source_hashes(), config_sha256=config_hashes),
        metrics_contract=dict(profile=PROFILE, requested_methods=selected,
            runtime_devices="logical indices within CUDA_VISIBLE_DEVICES",
            ordinary_role="five dose-adapted CSI/HPE baselines; no Proposed scheduled",
            por_role="separate encoder-tampering diagnostic; not a targeted-pose ranking",
            config_mode="editable YAML; any config or Python source change needs a new outdir",
            errors="mm", pck="relative thresholds 0.5/0.4/0.3/0.2/0.1; percent",
            uncertainty="single seed; not estimated"), cells=cells)
    matrix["plan_sha256"] = _plan_fingerprint(matrix)
    return matrix


def _verify_manifest(matrix):
    if (matrix.get("package_profile") != PROFILE or
            matrix.get("plan_sha256") != _plan_fingerprint(matrix)):
        raise ValueError("Invalid package plan fingerprint")
    if matrix["sources"]["source_sha256"] != _source_hashes():
        raise ValueError("Bundled source changed; use a NEW output directory")
    keys = [c["method_key"] for c in matrix["cells"]]
    requested = matrix["metrics_contract"]["requested_methods"]
    expected = (["clean"] if "infocom2025_por" in requested else []) + requested
    if keys != expected or set(requested) - set(METHODS):
        raise ValueError("Package plan contains an unrequested or unsupported method")


def _write_csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(title, rows, notice):
    lines = [f"# {title}", "", notice, "",
        "Errors are in mm; T1 is target-joint T-MPJPE at d=1. PCK thresholds are relative, not mm.", "",
        "| Method | MPJPE | PA-MPJPE | T1 | PCK .5/.4/.3/.2/.1 (%) |",
        "| --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        pck = "/".join(f"{row[f'clean_pck_{t:.1f}_pct']:.2f}"
                       for t in (0.5, 0.4, 0.3, 0.2, 0.1))
        lines.append(f"| {row['method']} | {row['clean_mpjpe_mm']:.3f} | "
            f"{row['clean_pampjpe_mm']:.3f} | {row['tmpjpe_mm']:.3f} | {pck} |")
    lines.extend(["", "All requested rows are retained without ranking or selection.",
        "No statistical superiority, RF stealthiness, or over-the-air feasibility is established.", ""])
    return "\n".join(lines)


def export_results(matrix, outdir):
    """Verify every selected cache before writing separate protocol reports."""
    from mmfi_tables import dose_rows, metric_row
    from run_method_drafts import _verified_cache

    _verify_manifest(matrix)
    groups = {"baseline": [], "por": [], "clean": []}
    doses = {"baseline": [], "por": [], "clean": []}
    provenance = []
    for cell in matrix["cells"]:
        res, audit = _verified_cache(cell)
        row = metric_row(cell, res)
        row.update(epochs=cell["cfg"]["epochs"], rho=cell["cfg"]["rho"])
        series, _ = dose_rows(cell, res)
        group = ("por" if cell["method_key"] == "infocom2025_por" else
                 "clean" if cell["method_key"] == "clean" else "baseline")
        groups[group].append(row)
        doses[group].extend(series)
        provenance.append(audit)
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    names = {"baseline": "baseline_metrics", "por": "por_diagnostics", "clean": "clean_teacher_metrics"}
    notices = {
        "baseline": "Five adapted traditional baselines use the packaged YAML settings. No Proposed is run here. Check configs before claiming a controlled comparison.",
        "por": "POR is a separate pretrained-encoder-tampering objective, not targeted pose poisoning. T1 is an additional diagnostic only; its rho/stages/budget differ. Do not rank this row against the traditional table.",
        "clean": "Clean teacher is an automatic support dependency for POR, not an attack baseline or ranked competitor.",
    }
    for group, rows in groups.items():
        if not rows:
            continue
        stem = names[group]
        _write_csv(output / f"{stem}.csv", rows, METRIC_COLUMNS)
        (output / f"{stem}.md").write_text(
            _markdown(stem.replace("_", " ").title(), rows, notices[group]), encoding="utf-8")
        dose_name = "dose_response" if group == "baseline" else f"{group}_dose_response"
        _write_csv(output / f"{dose_name}.csv", doses[group], DOSE_COLUMNS)
    report = dict(status="selected_baselines_complete", plan_sha256=matrix["plan_sha256"],
        requested_methods=matrix["metrics_contract"]["requested_methods"],
        traditional_rows=groups["baseline"], por_diagnostics=groups["por"],
        clean_teacher_rows=groups["clean"], provenance=provenance,
        sources=matrix["sources"], metrics_contract=matrix["metrics_contract"])
    _atomic_json(output / "baseline_report.json", report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--config-dir", type=Path, help="Editable copies of the seven MM-Fi YAML files")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Resolve only; no data/GPU/training")
    parser.add_argument("--export-only", action="store_true", help="Fail unless every selected cache is complete")
    parser.add_argument("--_cell", help=argparse.SUPPRESS)
    parser.add_argument("--_manifest", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args._cell:
        matrix = json.loads(args._manifest.read_text(encoding="utf-8"))
        _verify_manifest(matrix)
        cell = next(c for c in matrix["cells"] if c["method_key"] == args._cell)
        if args._cell in {"clean", "infocom2025_por"}:
            from run_mmfi_tables import _run_cell
        else:
            from run_traditional_comparison import _run_cell
        _run_cell(cell)
        return 0
    if args.data_home is None or args.outdir is None:
        raise ValueError("--data-home and --outdir are required")
    if args.dry_run and args.export_only:
        raise ValueError("--dry-run and --export-only are mutually exclusive")
    outdir = args.outdir.resolve()
    requested = build_matrix(args.data_home, outdir, args.devices, args.num_workers,
                             args.methods, args.config_dir)
    manifest_path = outdir / MANIFEST
    with _cell_lock(outdir):
        if manifest_path.exists():
            matrix = json.loads(manifest_path.read_text(encoding="utf-8"))
            _verify_manifest(matrix)
            if matrix["plan_sha256"] != requested["plan_sha256"]:
                raise ValueError("Existing config/selection/source differs; use a NEW output directory")
            for cell in matrix["cells"]:
                cell["cfg"]["num_workers"] = 0 if cell["method_key"] in {
                    "wanet_source", "fiba"} else args.num_workers
        else:
            if any(p.name != "run.lock" for p in outdir.iterdir()):
                raise FileExistsError("Nonempty output lacks a package manifest; use a NEW directory")
            if args.export_only:
                raise FileNotFoundError("--export-only requires an existing package manifest")
            matrix = requested
        _atomic_json(manifest_path, matrix)
        if args.dry_run:
            print(f"[baselines] DRY RUN: {len(matrix['cells'])} cells; no data/GPU/training.\n{manifest_path}")
            return 0
        if not args.export_only:
            print("[baselines] Devices are logical CUDA indices; physical mapping follows CUDA_VISIBLE_DEVICES.", flush=True)
            _check_inputs(matrix, args.devices)
            run_matrix(matrix, manifest_path, args.devices, worker_script=Path(__file__))
        missing = [c["method_key"] for c in matrix["cells"] if not _complete(c)]
        if missing:
            raise RuntimeError(f"Missing completed cells: {missing}; no partial report was exported")
        export_results(matrix, outdir)
        print(f"[baselines] Complete: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"[baselines] ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
