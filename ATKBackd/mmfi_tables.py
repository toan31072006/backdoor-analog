"""Export the single-seed MM-Fi four-table experiment from verified caches.

This module only reads evaluation caches; importing it never imports a model or
starts training. ``export_tables(manifest, outdir)`` accepts a resolved manifest
path, a manifest dictionary, or a list of cells. Cells contain ``method_key``,
``cfg``, and ``eval_cache`` (or ``ckpt_dir``), with optional ``label``, ``group``,
and ``tables``. Relative cache paths are relative to the manifest file.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class ReportValidationError(ValueError):
    """A cache or experiment matrix cannot support the requested report."""


TABLE_NAMES = {
    1: "table1_main",
    2: "table2_dose_response",
    3: "table3_coupling_ablation",
    4: "table4_poison_rate",
}
METRIC_COLUMNS = (
    "clean_mpjpe_mm", "clean_pampjpe_mm", "clean_pck_0.5_pct",
    "clean_pck_0.4_pct", "clean_pck_0.3_pct", "clean_pck_0.2_pct",
    "clean_pck_0.1_pct", "tmpjpe_mm",
)
DOSE_COLUMNS = (
    "method_key", "method", "seed", "d", "no_trigger_tmpjpe_mm",
    "triggered_tmpjpe_mm", "improvement_mm", "baseline_source",
)
RATE_KEYS = {"prop_rho0p1": 0.1, "prop_rho0p2": 0.2, "prop_rho0p4": 0.4}
ABLATION_KEYS = ("clean", "shuffled", "prop_rho0p4")
MAIN_KEYS = ("infocom2025_por", "ccai2026_backdoorrf", "badnets", "blended", "wanet", "prop_rho0p4")
_PROTOCOLS = {"infocom2025_por": "infocom2025_por", "ccai2026_backdoorrf": "ccai2026_backdoorrf"}
ROW_METADATA = ("training_protocol", "threat_model", "attacker_access")
REPORT_LIMITATIONS = (
    "The INFOCOM RF PTM/POR adaptation optimizes predefined encoder representations and then trains a clean pose head. Its T-MPJPE to the input-dependent pose target is a transferred diagnostic: POR does not optimize that HPE target.",
    "The default INFOCOM substitute is unlabeled downstream TRAIN CSI. This adaptation is not data-free: it also uses a supervised clean HPE teacher and downstream TRAIN CSI to calibrate POR scale, even with an external substitute. Any external pool still requires independent provenance.",
    "RF baselines use white-box access and distinct staged training budgets, whereas the proposed and traditional poisoning methods use data-only victim ERM. Interpret comparisons within these declared protocols; the tables do not establish a universally fair ranking across threat models or training budgets.",
    "BadNets and Blended call pinned BackdoorBench trigger operators (a third-party implementation); WaNet remains independently adapted. These CSI/HPE experiments are not exact reproductions of the original image-domain papers. BadNets uses an opaque white patch at dose1, not the shared gain epsilon. Patch opacity, blend alpha, RF segment RMS or Gaussian scale, and geometric warp strength have different meanings; these runs do not claim an equal L-infinity perturbation budget.",
)
_RUNTIME_KEYS = {"device", "num_workers", "ckpt_every", "data_parallel"}
_FINGERPRINT_RUNTIME_KEYS = {"device", "epochs", "ckpt_every", "num_workers"}
# These options identify the intervention. Victim architecture, optimizer,
# learning rate, payload geometry, dataset split, and victim budget stay shared.
_METHOD_KEYS = {
    "trigger", "action_npy", "top_k", "aoa_spread", "eps",
    "trigger_zero_mean", "rho", "coupling", "dose_coupling",
    "poison_coupling", "training_protocol", "attack_training_protocol",
    "attacker_access", "threat_model", "epochs", "victim_epochs",
    "config_fingerprint", "dose_min", "dose_max", "clean_teacher_checkpoint",
}
_METHOD_PREFIXES = ("tsba_", "wiba_", "badnet_", "badnets_", "sig_", "blend_", "blended_", "wanet_", "rf_")


def _trainer_schemas() -> tuple[int, int]:
    """Read scalar schema constants without loading torch or the trainer."""
    source = Path(__file__).with_name("train_backdoor.py")
    constants = {}
    for node in ast.parse(source.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "_CHECKPOINT_SCHEMA", "_RESULT_SCHEMA"
                }:
                    constants[target.id] = ast.literal_eval(node.value)
    return constants["_CHECKPOINT_SCHEMA"], constants["_RESULT_SCHEMA"]


def config_fingerprint(cfg: Mapping[str, Any], checkpoint_schema: int | None = None) -> str:
    """Mirror the trainer's fingerprint, keeping this exporter model-free."""
    if checkpoint_schema is None:
        checkpoint_schema, _ = _trainer_schemas()
    stable = {k: v for k, v in cfg.items() if k not in _FINGERPRINT_RUNTIME_KEYS}
    payload = json.dumps(
        {"schema": checkpoint_schema, "config": stable},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _number(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportValidationError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ReportValidationError(f"{name} must be finite and nonnegative")
    return value


def _array(value: Any, name: str, length: int | None = None) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ReportValidationError(f"{name} must be an array")
    if length is not None and len(value) != length:
        raise ReportValidationError(f"{name} length does not match dose_grid")
    return [_number(item, f"{name}[{i}]") for i, item in enumerate(value)]


def _dose_grid(res: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[float]:
    grid = _array(res.get("dose_grid"), "dose_grid")
    if len(grid) < 2 or any(d > 1 for d in grid):
        raise ReportValidationError("dose_grid requires at least two doses in [0, 1]")
    if any(a >= b for a, b in zip(grid, grid[1:])):
        raise ReportValidationError("dose_grid must be strictly increasing")
    if sum(math.isclose(d, 1.0, abs_tol=1e-9) for d in grid) != 1:
        raise ReportValidationError("dose_grid must contain d=1 exactly once")
    if cfg.get("dose_grid") is not None and grid != list(cfg["dose_grid"]):
        raise ReportValidationError("cached dose_grid differs from resolved cfg")
    if "reference_dose" in res and not math.isclose(
        _number(res["reference_dose"], "reference_dose"), 1.0, abs_tol=1e-9
    ):
        raise ReportValidationError("reference_dose must be d=1")
    return grid


def _ref_index(grid: Sequence[float]) -> int:
    return next(i for i, d in enumerate(grid) if math.isclose(d, 1, abs_tol=1e-9))


def metric_row(cell: Mapping[str, Any], res: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the eight main metrics, using d=1 rather than a guessed index."""
    cfg = cell.get("cfg", cell.get("config", {}))
    grid = _dose_grid(res, cfg)
    tmpjpe = _array(res.get("tmpjpe"), "tmpjpe", len(grid))
    row = {
        "method_key": cell["method_key"],
        "method": cell.get("label", cell["method_key"]),
        "seed": cfg["seed"],
        "clean_mpjpe_mm": _number(res.get("clean_mpjpe"), "clean_mpjpe") * 1000,
        "clean_pampjpe_mm": _number(res.get("clean_pampjpe"), "clean_pampjpe") * 1000,
    }
    for threshold in (0.5, 0.4, 0.3, 0.2, 0.1):
        value = _number(res.get(f"clean_pck@{threshold:.1f}"), f"clean_pck@{threshold:.1f}")
        if value > 1:
            raise ReportValidationError("clean PCK cache values must be fractions in [0, 1]")
        row[f"clean_pck_{threshold:.1f}_pct"] = value * 100
    row["tmpjpe_mm"] = tmpjpe[_ref_index(grid)] * 1000
    for name in ROW_METADATA:
        row[name] = cfg.get(name, "ordinary_erm" if name == "training_protocol" else "")
    return row


def dose_rows(
    cell: Mapping[str, Any], res: Mapping[str, Any], *, allow_partial: bool = False
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compare triggered and no-trigger predictions to each dose's same target.

    Legacy ``asr@ref.target_scale_mean`` is only a d=1 fallback. Other doses
    require the freshly evaluated clean-to-target array; clean_target_floor is
    deliberately never used as a baseline for rotated targets.
    """
    cfg = cell.get("cfg", cell.get("config", {}))
    grid = _dose_grid(res, cfg)
    triggered = _array(res.get("tmpjpe"), "tmpjpe", len(grid))
    source = "clean_to_target_tmpjpe"
    if "clean_to_target_tmpjpe" in res:
        clean = _array(res["clean_to_target_tmpjpe"], source, len(grid))
    else:
        clean = [None] * len(grid)
        source = "asr@ref.target_scale_mean (legacy d=1 only)"
        reference = res.get("asr@ref", {})
        if isinstance(reference, Mapping) and "target_scale_mean" in reference:
            clean[_ref_index(grid)] = _number(reference["target_scale_mean"], source)
    rows, issues = [], []
    for i, d in enumerate(grid):
        if clean[i] is None:
            issues.append(f"{cell['method_key']}: missing same-model no-trigger T-MPJPE at d={d:g}")
            continue
        rows.append({
            "method_key": cell["method_key"], "method": cell.get("label", cell["method_key"]),
            "seed": cfg["seed"], "d": d,
            "no_trigger_tmpjpe_mm": clean[i] * 1000,
            "triggered_tmpjpe_mm": triggered[i] * 1000,
            "improvement_mm": (clean[i] - triggered[i]) * 1000,
            "baseline_source": source,
        })
    if issues and not allow_partial:
        raise ReportValidationError("; ".join(issues))
    return rows, issues


def _normalise_manifest(manifest: Any) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    base = Path.cwd()
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).resolve()
        base = path.parent
        manifest = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(manifest, list):
        manifest = {"cells": manifest}
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("cells"), list):
        raise ReportValidationError("manifest must contain a cells list")
    cells = []
    for raw in manifest["cells"]:
        if not isinstance(raw, Mapping):
            raise ReportValidationError("each manifest cell must be a dictionary")
        cell = dict(raw)
        cfg = cell.get("cfg", cell.get("config"))
        if not isinstance(cfg, Mapping):
            raise ReportValidationError("each cell requires a resolved cfg dictionary")
        cell["cfg"] = dict(cfg)
        key = cell.get("method_key")
        if not isinstance(key, str) or not key:
            raise ReportValidationError("each cell requires a nonempty method_key")
        cell.setdefault("label", key)
        memberships = cell.get("tables")
        if memberships is None:
            memberships = [1] if key not in {"clean", "shuffled", "prop_rho0p1", "prop_rho0p2"} else []
        try:
            cell["tables"] = [int(t) for t in memberships]
        except (ValueError, TypeError):
            raise ReportValidationError(f"{key}: tables must contain table numbers") from None
        if any(t not in TABLE_NAMES for t in cell["tables"]):
            raise ReportValidationError(f"{key}: table number must be one of 1, 2, 3, 4")
        cache = cell.get("eval_cache", cell.get("cache_path"))
        if cache is None:
            directory = cell.get("ckpt_dir", cell.get("cell_dir"))
            if directory is None:
                raise ReportValidationError(f"{key}: eval_cache or ckpt_dir is required")
            cache = Path(directory) / "eval_cache.json"
        cache = Path(cache)
        cell["eval_cache"] = str(cache if cache.is_absolute() else base / cache)
        cells.append(cell)
    return dict(manifest), cells, base


def _protocol(cfg: Mapping[str, Any], key: str) -> None:
    if cfg.get("seed") != 42 or isinstance(cfg.get("seed"), bool):
        raise ReportValidationError(f"{key}: this report requires the single seed 42")
    if str(cfg.get("mmfi_protocol", "")).lower() != "protocol1" or str(cfg.get("mmfi_setting", "")).lower() != "s1":
        raise ReportValidationError(f"{key}: this report requires MM-Fi protocol1-s1")
    if cfg.get("pivot") != 1:
        raise ReportValidationError(f"{key}: pivot=1 targets MM-Fi joints [2, 3]")
    if "target_joints" in cfg and list(cfg["target_joints"]) != [2, 3]:
        raise ReportValidationError(f"{key}: target_joints must be [2, 3]")
    if str(cfg.get("victim_loss", "mpjpe")).lower() != "mpjpe":
        raise ReportValidationError(f"{key}: victim_loss must be standard MPJPE")
    if key in RATE_KEYS and not math.isclose(_number(cfg.get("rho"), f"{key}.rho"), RATE_KEYS[key], abs_tol=1e-9):
        raise ReportValidationError(f"{key}: rho must be {RATE_KEYS[key]}")
    if key == "clean" and _number(cfg.get("rho"), "clean.rho") != 0:
        raise ReportValidationError("clean: clean-only training requires rho=0")
    if key in {*MAIN_KEYS, "shuffled"} and not math.isclose(_number(cfg.get("rho"), f"{key}.rho"), 0.4, abs_tol=1e-9):
        raise ReportValidationError(f"{key}: main and shuffled configurations require rho=0.4")
    training_protocol = cfg.get("training_protocol", "ordinary_erm")
    if training_protocol != _PROTOCOLS.get(key, "ordinary_erm"):
        raise ReportValidationError(f"{key}: training_protocol must document {_PROTOCOLS.get(key, 'ordinary_erm')}")
    coupling = cfg.get("dose_coupling", cfg.get("poison_coupling", cfg.get("coupling", "paired")))
    if key == "shuffled" and coupling != "shuffled":
        raise ReportValidationError("shuffled: dose_coupling must be shuffled")
    if key in RATE_KEYS and coupling not in {"paired", "coupled"}:
        raise ReportValidationError(f"{key}: proposed dose_coupling must be paired")
    if "dose_min" in cfg or "dose_max" in cfg:
        expected_range = (1.0, 1.0) if key in MAIN_KEYS[:5] else (0.2, 1.0)
        actual_range = (_number(cfg.get("dose_min"), f"{key}.dose_min"), _number(cfg.get("dose_max"), f"{key}.dose_max"))
        if actual_range != expected_range:
            raise ReportValidationError(f"{key}: documented training dose range must be {expected_range}")
    if cfg.get("epochs") is None or _number(cfg["epochs"], f"{key}.epochs") <= 0:
        raise ReportValidationError(f"{key}: epochs must be positive")


def _shared_projection(cfg: Mapping[str, Any]) -> dict[str, Any]:
    shared = {k: v for k, v in cfg.items() if k not in _RUNTIME_KEYS | _METHOD_KEYS and not k.startswith(_METHOD_PREFIXES)}
    shared["victim_epochs"] = cfg.get("victim_epochs", cfg.get("epochs"))
    return shared


def _validate_matrix(manifest: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = [cell["method_key"] for cell in cells]
    if len(set(keys)) != len(keys):
        raise ReportValidationError("manifest contains duplicate method keys")
    if manifest.get("seed", 42) != 42:
        raise ReportValidationError("manifest seed must be 42")
    if str(manifest.get("dataset", "mmfi")).lower() != "mmfi":
        raise ReportValidationError("manifest dataset must be mmfi")
    if not cells:
        raise ReportValidationError("manifest cells cannot be empty")
    reference = _shared_projection(cells[0]["cfg"])
    for cell in cells:
        key, cfg = cell["method_key"], cell["cfg"]
        _protocol(cfg, key)
        shared = _shared_projection(cfg)
        changed = sorted(k for k in set(reference) | set(shared) if reference.get(k) != shared.get(k) or (k in reference) != (k in shared))
        if changed:
            raise ReportValidationError(f"{key}: shared victim/data/payload hyperparameters differ: {', '.join(changed)}")
        if cfg.get("epochs") != cfg.get("victim_epochs", cfg.get("epochs")) and not any(cfg.get(k) for k in ("training_protocol", "attack_training_protocol", "rf_training_protocol")):
            raise ReportValidationError(f"{key}: staged training requires documented training_protocol")
    main = [cell["method_key"] for cell in cells if 1 in cell["tables"]]
    if "clean" in main or "shuffled" in main:
        raise ReportValidationError("clean and shuffled controls belong only in the coupling ablation")
    if "prop_rho0p1" in main or "prop_rho0p2" in main:
        raise ReportValidationError("the main comparison uses proposed rho=0.4 only")
    required = (*MAIN_KEYS, "clean", "shuffled", "prop_rho0p1", "prop_rho0p2")
    issues = [f"manifest missing required method {key}" for key in required if key not in keys]
    issues.extend(f"Table 1 is missing required main method {key}" for key in MAIN_KEYS if key not in main)
    return issues


def _load_verified_cache(cell: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    key, cfg, path = cell["method_key"], cell["cfg"], Path(cell["eval_cache"])
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportValidationError(f"{key}: cannot read completed eval_cache.json: {exc}") from exc
    if not isinstance(blob, Mapping):
        raise ReportValidationError(f"{key}: evaluation cache must be a wrapper object")
    checkpoint_schema, result_schema = _trainer_schemas()
    # Schema 9 predates the per-dose clean-to-target array; its d=1 baseline
    # remains usable, but it cannot silently complete a multi-dose table.
    if blob.get("result_schema") not in {result_schema, 9}:
        raise ReportValidationError(f"{key}: stale or unknown evaluation result schema")
    cached_cfg = blob.get("cfg")
    if not isinstance(cached_cfg, Mapping):
        raise ReportValidationError(f"{key}: cache is missing its resolved cfg")
    expected = config_fingerprint(cfg, checkpoint_schema)
    if blob.get("cfg_fingerprint") != expected or config_fingerprint(cached_cfg, checkpoint_schema) != expected:
        raise ReportValidationError(f"{key}: cache fingerprint does not match the resolved manifest config")
    if blob.get("trained_epochs") != cfg["epochs"] or cached_cfg.get("epochs") != cfg["epochs"]:
        raise ReportValidationError(f"{key}: cache has not completed the requested epoch budget")
    res = blob.get("res")
    if not isinstance(res, dict):
        raise ReportValidationError(f"{key}: cache is missing its result dictionary")
    if "target_joints" in res and list(res["target_joints"]) != [2, 3]:
        raise ReportValidationError(f"{key}: cached target_joints must be [2, 3]")
    provenance = {
        "method_key": key, "label": cell["label"], "group": cell.get("group"),
        "eval_cache": str(path.resolve()), "cfg_fingerprint": expected,
        "result_schema": blob["result_schema"], "trained_epochs": blob["trained_epochs"],
        "cfg": dict(cached_cfg),
    }
    for name in ("threat_model", "attacker_access", "training_protocol", "attack_training_protocol", "rf_training_protocol"):
        if name in cfg:
            provenance[name] = cfg[name]
    return res, provenance


def build_tables(manifest: Any, *, allow_partial: bool = False) -> dict[str, Any]:
    """Validate and construct a report without writing files."""
    metadata, cells, _ = _normalise_manifest(manifest)
    issues = _validate_matrix(metadata, cells)
    table_rows = {name: [] for name in TABLE_NAMES.values()}
    rows, results, provenance = {}, {}, []
    for cell in cells:
        try:
            res, audit = _load_verified_cache(cell)
            rows[cell["method_key"]] = metric_row(cell, res)
            results[cell["method_key"]] = res
            provenance.append(audit)
        except ReportValidationError as exc:
            issues.append(str(exc))
    for cell in cells:
        if 1 in cell["tables"] and cell["method_key"] in rows:
            table_rows[TABLE_NAMES[1]].append(rows[cell["method_key"]])
    for key in ABLATION_KEYS:
        if key in rows:
            table_rows[TABLE_NAMES[3]].append(rows[key])
    for key, rho in RATE_KEYS.items():
        if key in rows:
            table_rows[TABLE_NAMES[4]].append({**rows[key], "rho": rho})
    proposed = next((c for c in cells if c["method_key"] == "prop_rho0p4"), None)
    if proposed is not None and "prop_rho0p4" in results:
        try:
            dose, missing = dose_rows(proposed, results["prop_rho0p4"], allow_partial=True)
            table_rows[TABLE_NAMES[2]] = dose
            issues.extend(missing)
        except ReportValidationError as exc:
            issues.append(str(exc))
    if issues and not allow_partial:
        raise ReportValidationError("refusing to export official incomplete tables: " + "; ".join(issues))
    return {
        "schema": 1, "status": "unofficial_partial" if issues else "official_complete",
        "seed": 42, "dataset": "mmfi", "protocol": "protocol1-s1",
        "target_joints": [2, 3], "reference_dose": 1.0,
        "units": {"errors": "mm", "relative_pck": "%", "dose": "unitless"},
        "metric_definition": {
            "tmpjpe": "Mean target-joint Euclidean error at d=1 over MM-Fi joints [2, 3]",
            "dose_baseline": "Same model, no trigger, compared with G(Y, d) at each dose",
            "improvement": "no_trigger_tmpjpe_mm - triggered_tmpjpe_mm",
            "relative_pck": "Five relative thresholds: 0.5, 0.4, 0.3, 0.2, 0.1; fractions converted to percentages",
        },
        "tables": table_rows, "issues": issues, "provenance": provenance,
        "sources": metadata.get("sources", {}),
        "limitations": list(REPORT_LIMITATIONS),
    }


def _columns(table: int) -> tuple[str, ...]:
    if table == 2:
        return DOSE_COLUMNS
    common = ("method_key", "method", "seed", *ROW_METADATA)
    return common + (("rho",) if table == 4 else ()) + METRIC_COLUMNS


def _display(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown(report: Mapping[str, Any]) -> str:
    status = report["status"]
    lines = ["# MM-Fi experiment tables", "", f"Status: **{status}**. Seed 42; protocol1-s1; target joints [2, 3].", "",
             "Errors are in mm. PCK is a percentage at relative thresholds 0.5/0.4/0.3/0.2/0.1. Main-table T-MPJPE is evaluated at d=1.", "",
             "The dose table uses the same rho=0.4 model and the same target G(Y,d) for triggered and no-trigger predictions. Positive improvement means the trigger reduced target error. Single-seed estimates have no uncertainty interval.", ""]
    lines.extend(["## Interpretation limits", ""])
    for limitation in report["limitations"]:
        lines.extend([limitation, ""])
    if report["issues"]:
        lines.extend(["This is an unofficial partial report. Missing or invalid measurements were not filled in.", ""])
        lines.extend(f"- {issue}" for issue in report["issues"])
        lines.append("")
    titles = {1: "Table 1: Main comparison", 2: "Table 2: Dose response", 3: "Table 3: Coupling ablation", 4: "Table 4: Poison-rate sensitivity"}
    for number, name in TABLE_NAMES.items():
        columns = _columns(number)
        lines.extend([f"## {titles[number]}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"])
        for row in report["tables"][name]:
            lines.append("| " + " | ".join(_display(row[column]) for column in columns) + " |")
        if not report["tables"][name]:
            lines.extend(["", "No verified measurements available."])
        lines.append("")
    sources = report.get("sources", {})
    if isinstance(sources, Mapping) and sources:
        lines.extend(["## Source provenance", ""])
        labels = {item["method_key"]: item["label"] for item in report["provenance"]}
        for key, source in sources.items():
            if not isinstance(source, Mapping):
                continue
            references = []
            for field, label in (("paper", "paper"), ("code", "source code"), ("repository", "repository")):
                url = source.get(field)
                if isinstance(url, str) and url.startswith(("https://", "http://")):
                    references.append(f"[{label}]({url})")
            details = references[:]
            for field in ("commit", "license", "source_file", "source_sha256", "implementation"):
                if source.get(field):
                    details.append(f"{field}: {_display(source[field])}")
            lines.extend([f"{_display(labels.get(key, key))}: " + "; ".join(details) + ".", ""])
            caveats = source.get("limitations", [])
            if isinstance(caveats, list) and caveats:
                lines.extend(["Declared source/adaptation caveats: " + "; ".join(_display(value) for value in caveats) + ".", ""])
    return "\n".join(lines)


def _plots(report: Mapping[str, Any], outdir: Path, suffix: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return []
    paths = []
    dose = report["tables"][TABLE_NAMES[2]]
    if dose:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["d"] for r in dose], [r["no_trigger_tmpjpe_mm"] for r in dose], "o-", label="No trigger, same target")
        ax.plot([r["d"] for r in dose], [r["triggered_tmpjpe_mm"] for r in dose], "s-", label="Triggered")
        ax.set(xlabel="Dose d", ylabel="T-MPJPE to G(Y,d) (mm)", title="Proposed, rho=0.4, seed 42")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = outdir / f"dose_response{suffix}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    rate = report["tables"][TABLE_NAMES[4]]
    if rate:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["rho"] for r in rate], [r["tmpjpe_mm"] for r in rate], "o-", label="T-MPJPE at d=1")
        ax.plot([r["rho"] for r in rate], [r["clean_mpjpe_mm"] for r in rate], "s-", label="Clean MPJPE")
        ax.set(xlabel="Poison rate rho", ylabel="Error (mm)", title="Proposed, seed 42")
        ax.set_xticks([0.1, 0.2, 0.4])
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = outdir / f"poison_rate{suffix}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def export_tables(
    manifest: Any, outdir: str | Path, *, allow_partial: bool = False, plots: bool = True
) -> dict[str, Any]:
    """Write four CSVs, a JSON/Markdown report, and optional real-data plots.

    Official validation finishes before any output is created. Explicit partial
    exports always use ``.unofficial`` filenames, even when currently complete.
    The returned dictionary includes the report plus an ``artifacts`` map.
    """
    report = build_tables(manifest, allow_partial=allow_partial)
    if allow_partial:
        report["status"] = "unofficial_partial" if report["issues"] else "unofficial_complete"
    suffix = ".unofficial" if allow_partial else ""
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for number, name in TABLE_NAMES.items():
        path = output / f"{name}{suffix}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_columns(number))
            writer.writeheader()
            writer.writerows(report["tables"][name])
        artifacts[name] = str(path)
    json_path = output / f"mmfi_tables{suffix}.json"
    markdown_path = output / f"mmfi_tables{suffix}.md"
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    artifacts["markdown"] = str(markdown_path)
    artifacts["json"] = str(json_path)
    artifacts["plots"] = _plots(report, output, suffix) if plots else []
    report["artifacts"] = artifacts
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
