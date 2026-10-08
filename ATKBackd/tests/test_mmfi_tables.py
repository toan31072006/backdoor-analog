"""Model-free checks for the MM-Fi four-table exporter."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmfi_tables import (  # noqa: E402
    METRIC_COLUMNS,
    ROW_METADATA,
    ReportValidationError,
    _trainer_schemas,
    build_tables,
    config_fingerprint,
    dose_rows,
    export_tables,
    metric_row,
)


def _cfg(**overrides):
    cfg = {
        "experiment_name": "mmfi", "dataset_root": "/data/MMFI",
        "mmfi_protocol": "protocol1", "mmfi_setting": "s1",
        "mmfi_split_seed": 0, "seed": 42, "model": "hpeli",
        "pivot": 1, "theta_max_deg": 40.0, "payload_axis": [0.0, 0.0, 1.0],
        "dose_mode": "linear", "dose_grid": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "rho": 0.4, "poison_select": "uniform", "poison_coupling": "coupled",
        "epochs": 50, "victim_epochs": 50, "batch_size": 32,
        "optimizer": "sgd", "lr": 0.001, "momentum": 0.9,
        "weight_decay": 0.0, "victim_loss": "mpjpe", "trigger": "micro_dropper",
        "eps": 0.185, "num_workers": 4, "device": "cpu",
    }
    cfg.update(overrides)
    return cfg


def _res(**overrides):
    res = {
        "clean_mpjpe": 0.10, "clean_pampjpe": 0.08,
        "dose_grid": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "reference_dose": 1.0,
        "tmpjpe": [0.07, 0.06, 0.055, 0.05, 0.04, 0.03],
        "clean_to_target_tmpjpe": [0.07, 0.09, 0.11, 0.13, 0.15, 0.17],
        "clean_target_floor": 0.07, "asr@ref": {"asr": 0.5, "target_scale_mean": 0.17},
    }
    for threshold in (0.5, 0.4, 0.3, 0.2, 0.1):
        res[f"clean_pck@{threshold:.1f}"] = threshold + 0.4
    res.update(overrides)
    return res


def _write_cache(path, config, result=None, **wrapper_changes):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "result_schema": _trainer_schemas()[1], "cfg_fingerprint": config_fingerprint(config),
        "trained_epochs": config["epochs"], "cfg": config, "res": _res() if result is None else result,
    }
    blob.update(wrapper_changes)
    path.write_text(json.dumps(blob), encoding="utf-8")


@pytest.fixture
def manifest(tmp_path):
    specifications = [
        ("infocom2025_por", "external", 0.4, [1]), ("ccai2026_backdoorrf", "external", 0.4, [1]),
        ("badnets", "traditional", 0.4, [1]), ("wanet", "traditional", 0.4, [1]),
        ("blended", "traditional", 0.4, [1]), ("prop_rho0p4", "proposed", 0.4, [1, 2, 3, 4]),
        ("clean", "clean", 0.0, [3]), ("shuffled", "shuffled", 0.4, [3]),
        ("prop_rho0p1", "proposed", 0.1, [4]), ("prop_rho0p2", "proposed", 0.2, [4]),
    ]
    cells = []
    for key, group, rho, tables in specifications:
        cfg = _cfg(rho=rho, trigger=key if group in {"external", "traditional"} else "micro_dropper")
        cfg["training_protocol"] = key if group == "external" else "ordinary_erm"
        if key == "shuffled":
            cfg["poison_coupling"] = "shuffled"
        path = tmp_path / key / "eval_cache.json"
        _write_cache(path, cfg)
        cells.append({
            "method_key": key, "label": key, "group": group,
            "tables": tables, "cfg": cfg, "eval_cache": str(path),
        })
    return {"schema": 1, "dataset": "mmfi", "seed": 42, "cells": cells}


def _cell(manifest, key):
    return next(cell for cell in manifest["cells"] if cell["method_key"] == key)


def _rewrite_cfg(manifest, key, **changes):
    cell = _cell(manifest, key)
    cell["cfg"].update(changes)
    _write_cache(Path(cell["eval_cache"]), cell["cfg"])


def test_complete_four_tables_have_exact_metrics_and_membership(manifest):
    report = build_tables(manifest)
    assert report["status"] == "official_complete"
    tables = report["tables"]
    assert len(tables["table1_main"]) == 6
    assert len(tables["table2_dose_response"]) == 6
    assert [r["method_key"] for r in tables["table3_coupling_ablation"]] == ["clean", "shuffled", "prop_rho0p4"]
    assert [r["rho"] for r in tables["table4_poison_rate"]] == [0.1, 0.2, 0.4]
    for row in tables["table1_main"]:
        assert set(row) == {"method_key", "method", "seed", *ROW_METADATA, *METRIC_COLUMNS}
        assert row["clean_mpjpe_mm"] == pytest.approx(100)
        assert row["clean_pampjpe_mm"] == pytest.approx(80)
        assert row["clean_pck_0.5_pct"] == pytest.approx(90)
        assert row["tmpjpe_mm"] == pytest.approx(30)
    assert "asr" not in str(tables["table1_main"])
    assert "±" not in json.dumps(report)


def test_dose_curve_uses_same_model_and_each_doses_target(manifest):
    proposed = _cell(manifest, "prop_rho0p4")
    rows, issues = dose_rows(proposed, _res())
    assert not issues
    assert [r["no_trigger_tmpjpe_mm"] for r in rows] == pytest.approx([70, 90, 110, 130, 150, 170])
    assert rows[-1]["improvement_mm"] == pytest.approx(140)
    assert rows[0]["improvement_mm"] == pytest.approx(0)
    assert all(r["baseline_source"] == "clean_to_target_tmpjpe" for r in rows)


def test_legacy_baseline_is_only_allowed_at_dose_one(manifest):
    res = _res()
    del res["clean_to_target_tmpjpe"]
    cell = _cell(manifest, "prop_rho0p4")
    with pytest.raises(ReportValidationError, match="d=0"):
        dose_rows(cell, res)
    rows, issues = dose_rows(cell, res, allow_partial=True)
    assert len(rows) == 1 and rows[0]["d"] == 1
    assert rows[0]["no_trigger_tmpjpe_mm"] == pytest.approx(170)
    assert "legacy" in rows[0]["baseline_source"]
    assert len(issues) == 5
    del res["asr@ref"]
    rows, issues = dose_rows(cell, res, allow_partial=True)
    assert rows == [] and len(issues) == 6


def test_missing_cache_fails_before_creating_official_outputs(manifest, tmp_path):
    Path(_cell(manifest, "infocom2025_por")["eval_cache"]).unlink()
    output = tmp_path / "official"
    with pytest.raises(ReportValidationError, match="refusing to export official incomplete"):
        export_tables(manifest, output, plots=False)
    assert not output.exists()


def test_partial_export_is_separate_and_never_invents_a_row(manifest, tmp_path):
    Path(_cell(manifest, "infocom2025_por")["eval_cache"]).unlink()
    report = export_tables(manifest, tmp_path / "partial", allow_partial=True, plots=False)
    assert report["status"] == "unofficial_partial"
    assert len(report["tables"]["table1_main"]) == 5
    assert not any(row["method_key"] == "infocom2025_por" for row in report["tables"]["table1_main"])
    outputs = list((tmp_path / "partial").iterdir())
    assert len(outputs) == 6
    assert all(".unofficial." in path.name for path in outputs)
    stored = json.loads(Path(report["artifacts"]["json"]).read_text(encoding="utf-8"))
    assert stored["issues"] and stored["status"] == "unofficial_partial"


def test_explicit_partial_mode_keeps_unofficial_names_when_complete(manifest, tmp_path):
    report = export_tables(manifest, tmp_path / "output", allow_partial=True, plots=False)
    assert report["status"] == "unofficial_complete"
    assert "unofficial" in report["artifacts"]["json"]


@pytest.mark.parametrize("changes, error", [
    ({"seed": 0}, "seed 42"), ({"mmfi_setting": "s3"}, "protocol1-s1"),
    ({"mmfi_protocol": "protocol2"}, "protocol1-s1"), ({"pivot": 11}, "joints"),
    ({"rho": 0.2}, "rho must be 0.4"), ({"victim_loss": "mse"}, "standard MPJPE"),
    ({"lr": 0.01}, "hyperparameters differ"), ({"optimizer": "adamw"}, "hyperparameters differ"),
    ({"theta_max_deg": 60}, "hyperparameters differ"), ({"victim_epochs": 30}, "hyperparameters differ"),
])
def test_protocol_and_shared_hyperparameters_are_validated(manifest, changes, error):
    _rewrite_cfg(manifest, "prop_rho0p4", **changes)
    with pytest.raises(ReportValidationError, match=error):
        build_tables(manifest)


@pytest.mark.parametrize("wrapper_changes, error", [
    ({"trained_epochs": 49}, "epoch budget"), ({"cfg_fingerprint": "wrong"}, "fingerprint"),
    ({"result_schema": -1}, "schema"), ({"cfg": {}}, "fingerprint"), ({"res": None}, "result dictionary"),
])
def test_stale_and_incomplete_cache_is_rejected(manifest, wrapper_changes, error):
    cell = _cell(manifest, "infocom2025_por")
    _write_cache(Path(cell["eval_cache"]), cell["cfg"], **wrapper_changes)
    with pytest.raises(ReportValidationError, match=error):
        build_tables(manifest)


def test_all_methods_must_complete_even_if_only_main_cache_missing(manifest):
    Path(_cell(manifest, "prop_rho0p2")["eval_cache"]).unlink()
    with pytest.raises(ReportValidationError, match="prop_rho0p2"):
        build_tables(manifest)


def test_manifest_cannot_hide_a_missing_main_method(manifest):
    manifest["cells"] = [cell for cell in manifest["cells"] if cell["method_key"] != "wanet"]
    with pytest.raises(ReportValidationError, match="missing required method wanet"):
        build_tables(manifest)
    report = build_tables(manifest, allow_partial=True)
    assert report["status"] == "unofficial_partial"
    assert any("wanet" in issue for issue in report["issues"])


def test_fixed_baselines_and_shuffled_training_protocol_are_explicit(manifest):
    for key in ("badnets", "blended", "wanet"):
        _rewrite_cfg(manifest, key, dose_min=1.0, dose_max=1.0)
    for key in ("prop_rho0p1", "prop_rho0p2", "prop_rho0p4", "shuffled", "clean"):
        _rewrite_cfg(manifest, key, dose_min=0.2, dose_max=1.0)
    _rewrite_cfg(manifest, "wanet", wanet_cover_ratio=0.2)
    assert build_tables(manifest)["status"] == "official_complete"
    _rewrite_cfg(manifest, "shuffled", dose_coupling="paired")
    with pytest.raises(ReportValidationError, match="dose_coupling must be shuffled"):
        build_tables(manifest)


def test_duplicate_keys_and_clean_in_main_are_rejected(manifest):
    duplicate = copy.deepcopy(manifest)
    duplicate["cells"].append(copy.deepcopy(duplicate["cells"][0]))
    with pytest.raises(ReportValidationError, match="duplicate"):
        build_tables(duplicate)
    _cell(manifest, "clean")["tables"].append(1)
    with pytest.raises(ReportValidationError, match="only in the coupling ablation"):
        build_tables(manifest)


def test_documented_staged_attack_can_have_extra_budget(manifest):
    _rewrite_cfg(manifest, "infocom2025_por", epochs=60, victim_epochs=50, training_protocol="infocom2025_por", rf_generator_warmup_epochs=10)
    report = build_tables(manifest)
    audit = next(item for item in report["provenance"] if item["method_key"] == "infocom2025_por")
    assert audit["trained_epochs"] == 60
    assert audit["training_protocol"] == "infocom2025_por"
    assert audit["cfg"]["rf_generator_warmup_epochs"] == 10


def test_undocumented_extra_training_budget_is_rejected(manifest):
    _rewrite_cfg(manifest, "infocom2025_por", epochs=60, victim_epochs=50, training_protocol="ordinary_erm")
    with pytest.raises(ReportValidationError, match="training_protocol must document"):
        build_tables(manifest)


@pytest.mark.parametrize("changes, error", [
    ({"tmpjpe": [0.1]}, "length"), ({"clean_to_target_tmpjpe": [0.1]}, "length"),
    ({"clean_mpjpe": float("nan")}, "finite"), ({"clean_pck@0.5": 90}, "fractions"),
    ({"clean_pck@0.4": None}, "finite number"),
    ({"dose_grid": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]}, "d=1"),
    ({"target_joints": [12, 13]}, "target_joints"),
])
def test_malformed_metrics_cannot_produce_official_report(manifest, changes, error):
    cell = _cell(manifest, "prop_rho0p4")
    _write_cache(Path(cell["eval_cache"]), cell["cfg"], _res(**changes))
    with pytest.raises(ReportValidationError, match=error):
        build_tables(manifest)


def test_manifest_path_resolves_relative_cache_locations(manifest, tmp_path):
    for cell in manifest["cells"]:
        cell["eval_cache"] = str(Path(cell["eval_cache"]).relative_to(tmp_path))
    path = tmp_path / "mmfi_matrix.resolved.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = export_tables(path, tmp_path / "output", plots=False)
    assert report["status"] == "official_complete"
    assert len(list((tmp_path / "output").glob("*.csv"))) == 4


def test_downloadable_reports_include_interpretation_limits_and_sources(manifest, tmp_path):
    manifest["sources"] = {
        "infocom2025_por": {
            "paper": "https://arxiv.org/abs/2505.00881",
            "code": "https://github.com/Tianyaz97/rf_backdoor",
            "limitations": ["Default substitute uses downstream TRAIN CSI without labels"],
        },
        "wanet": {"paper": "https://arxiv.org/abs/2102.10369"},
    }
    report = export_tables(manifest, tmp_path / "with_sources", plots=False)
    stored = json.loads(Path(report["artifacts"]["json"]).read_text(encoding="utf-8"))
    markdown = Path(report["artifacts"]["markdown"]).read_text(encoding="utf-8")
    for warning in ("does not optimize", "not data-free", "white-box", "equal L-infinity"):
        assert warning in markdown
        assert warning in " ".join(stored["limitations"])
    assert "https://arxiv.org/abs/2505.00881" in markdown
    assert "https://github.com/Tianyaz97/rf_backdoor" in markdown
    assert "https://arxiv.org/abs/2102.10369" in markdown
    assert stored["sources"] == manifest["sources"]


def test_plots_are_created_from_verified_dose_and_rate_rows(manifest, tmp_path):
    pytest.importorskip("matplotlib")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    if (
        getattr(pyplot, "__spec__", None) is None
        and getattr(pyplot, "__file__", None) is None
        and not hasattr(pyplot, "subplots")
    ):
        pytest.skip("test_regressions.py installed its optional matplotlib.pyplot collection stub")
    report = export_tables(manifest, tmp_path / "with_plots", plots=True)
    assert len(report["artifacts"]["plots"]) == 2
    for name in report["artifacts"]["plots"]:
        path = Path(name)
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert path.stat().st_size > 1000


def test_metrics_are_selected_at_dose_one_explicitly(manifest):
    cell = _cell(manifest, "prop_rho0p4")
    row = metric_row(cell, _res())
    assert row["tmpjpe_mm"] == pytest.approx(30)
    assert "clean_target_floor_mm" not in row
    assert "asr" not in row


def test_runtime_only_changes_keep_fingerprint():
    cfg = _cfg()
    altered = dict(cfg, device="cuda", num_workers=0, epochs=75)
    assert config_fingerprint(cfg) == config_fingerprint(altered)
    assert config_fingerprint(cfg) != config_fingerprint(dict(cfg, lr=0.02))
