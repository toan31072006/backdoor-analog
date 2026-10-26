"""Controller scheduling/result-binding tests without PiW data or CUDA."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "remote_linux/piw3d_two_models/controller.py"
spec = importlib.util.spec_from_file_location("piw_two_models_controller", SOURCE)
assert spec and spec.loader
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_six_baselines_plus_a1_clean_on_two_balanced_gpus():
    waves = controller.waves()
    assert len(waves) == 4
    assert len({cell["id"] for wave in waves for cell in wave}) == 16
    for wave in waves:
        for gpu in ("cuda:0", "cuda:1"):
            assigned = [cell for cell in wave if cell["device"] == gpu]
            assert len(assigned) == 2
            assert {cell["backbone"] for cell in assigned} == {"metafiplusplus", "wiflow"}


def test_result_export_requires_complete_bound_artifacts(tmp_path):
    gate = {"preflight": {"plan_sha256": "test-science-plan"}}
    for backbone in controller.BACKBONES:
        for method in ("clean", "a1", "wanet_source", "ftrojan", "fiba",
                       "bpp_source", "smooth_lf_csi", "tsba_adapted"):
            folder = tmp_path / "attacks" / backbone / "seed42" / method
            folder.mkdir(parents=True)
            (folder / "checkpoint.pt").write_bytes(b"ckpt")
            (folder / "predictions.npz").write_bytes(b"paired")
            metrics = dict(clean_mpjpe_mm=100., clean_pampjpe_mm=60.)
            if method != "clean":
                metrics.update(t1_mpjpe_mm=140., target_pck_pct=55., mean_dce_mm=70.)
            report = dict(status="COMPLETE", backbone=backbone, method=method,
                          plan_sha256="test-science-plan",
                          checkpoint_sha256=controller.file_sha(folder / "checkpoint.pt"),
                          predictions_sha256=controller.file_sha(folder / "predictions.npz"))
            report.update(clean_metrics=metrics) if method == "clean" else report.update(
                metrics=dict(mainmetrics=metrics))
            (folder / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    summary = controller.export_results(tmp_path, gate)
    assert len(summary["rows"]) == 16
    assert sum(row["method"] == "clean" for row in summary["rows"]) == 2
    assert (tmp_path / "attack_results.json").is_file()

    (tmp_path / "attacks/wiflow/seed42/a1/predictions.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Invalid PiW evaluation binding"):
        controller.export_results(tmp_path, gate)
