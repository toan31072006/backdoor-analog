"""CPU-only tests for the DT-Pose HPELi/MM-Fi MICA package; no training."""
from pathlib import Path
import hashlib
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_baselines as launcher

DTPOSE_SHA = "237d6097c6106052abf0bd5b79cc2e26e3f7f05fa5062ca53d9bd846785e0997"
ORDINARY = launcher.METHODS[:5]


def _plan(tmp_path, **kwargs):
    return launcher.build_matrix(tmp_path / "no_data", tmp_path / "out", **kwargs)


def _configs(tmp_path):
    return {c["method_key"]: c["cfg"] for c in _plan(tmp_path)["cells"]}


def test_raw_dtpose_mmfi_config_is_pinned_and_not_relabelled_as_hpeli():
    path = ROOT / "original_configs/dtpose_mmfi_original.yaml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == DTPOSE_SHA
    original = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert original["total_epoch"] == 50
    assert original["seed"] == 42
    assert original["batch_size"] == 32
    assert original["train_loader"]["batch_size"] == 32
    assert original["validation_loader"]["batch_size"] == 32
    assert original["base_learning_rate"] == 0.001
    assert original["protocol"] == "protocol1"
    assert original["setting"] == "protocol1-s1"
    assert original["random_split"]["ratio"] == 0.8
    assert original["random_split"]["random_seed"] == 0
    assert original["dataset_name"] == "mmfi-csi"
    # This untouched source YAML defaults to pretrained DT-Pose, not HPELi.
    assert original["experiment_name"] == "DT-Pose-ratio80"
    assert original["pretrained_model_path"] is not None


def test_default_has_six_attacks_and_only_por_clean_dependency(tmp_path):
    matrix = _plan(tmp_path)
    assert matrix["package_profile"] == "mica_mmfi_dtpose_hpeli_config_v1"
    assert matrix["seed"] == 42
    assert [c["method_key"] for c in matrix["cells"]] == ["clean", *launcher.METHODS]
    assert not {"proposed", "ccai2026_backdoorrf"}.intersection(
        c["method_key"] for c in matrix["cells"])
    assert matrix["cells"][-1]["dependencies"] == ["clean"]
    assert matrix["cells"][-1]["cfg"]["rf_clean_teacher_checkpoint"] == str(
        tmp_path / "out" / "clean" / "checkpoint.pt")
    for cell in matrix["cells"]:
        assert cell["dependencies"] == (["clean"] if cell["method_key"] == "infocom2025_por" else [])


def test_active_training_matches_dtpose_mmfi_hpeli_scratch_branch(tmp_path):
    from train_backdoor import _get_num_keypoints
    from attack.payload import descendants, set_skeleton_config
    set_skeleton_config("mmfi")
    original = yaml.safe_load((ROOT / "original_configs/dtpose_mmfi_original.yaml").read_text(
        encoding="utf-8"))
    for cfg in _configs(tmp_path).values():
        assert cfg["epochs"] == cfg["victim_epochs"] == original["total_epoch"] == 50
        assert cfg["seed"] == original["seed"] == 42
        assert cfg["batch_size"] == original["batch_size"] == 32
        assert cfg["lr"] == original["base_learning_rate"] == 0.001
        assert cfg["model"] == "hpeli"
        assert cfg["pretrained"] is False
        assert cfg["optimizer"] == "sgd"
        assert cfg["momentum"] == 0.9
        assert cfg["weight_decay"] == 0
        assert cfg["victim_loss"] == "mpjpe"
        assert cfg["lr_scheduler"] is False
        assert cfg["mmfi_protocol"] == original["protocol"] == "protocol1"
        assert cfg["mmfi_setting"] == "s1"
        assert cfg["mmfi_random_ratio"] == original["random_split"]["ratio"] == 0.8
        assert cfg["mmfi_split_seed"] == original["random_split"]["random_seed"] == 0
        assert (cfg["n_ant"], cfg["n_sub"], cfg["n_pkt"]) == (3, 114, 10)
        assert _get_num_keypoints(cfg) == 17
        assert cfg["pivot"] == 1
        # The legacy clean/POR configs derive this from the dataset skeleton;
        # ordinary comparison configs also bind it explicitly.
        assert cfg.get("target_joints", list(descendants(cfg["pivot"]))) == [2, 3]
        assert cfg["dataset_root"] == str(tmp_path / "no_data/datasets/Compress")
        assert cfg["action_npy"] == str(tmp_path / "no_data/actions/data_bend.npy")
        assert cfg["data_parallel"] is False
        assert not {"max_train_samples", "max_test_samples", "train_cap", "test_cap", "method_draft"}.intersection(cfg)
        assert not any(key.startswith("draft_") for key in cfg)


def test_actual_optimizer_is_source_hpeli_sgd_not_pretrained_vit_weight_decay(tmp_path):
    import torch
    from train_backdoor import _build_optimizer
    for cfg in _configs(tmp_path).values():
        name, optimizer = _build_optimizer(torch.nn.Linear(2, 1), cfg, "mmfi")
        assert name == "sgd"
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[0]["momentum"] == 0.9
        assert optimizer.param_groups[0]["weight_decay"] == 0


def test_canonical_configuration_matches_existing_builders_without_baseline_changes(tmp_path):
    from run_traditional_comparison import build_matrix as traditional_plan
    from run_mmfi_tables import build_matrix as historical_plan
    data, out = tmp_path / "no_data", tmp_path / "out"
    reference = traditional_plan(data, out, "cuda:0", 4,
                                 budget_mode="shared_peak", baseline_set="extended")
    original = historical_plan(data, out, "cuda:0", 4)
    expected = {c["method_key"]: c["cfg"] for c in reference["cells"]}
    expected.update({c["method_key"]: c["cfg"] for c in original["cells"]
                     if c["method_key"] in {"clean", "infocom2025_por"}})
    for cell in _plan(tmp_path)["cells"]:
        assert cell["cfg"] == expected[cell["method_key"]]


def test_native_knobs_existing_shared_peak_and_worker_cover_behavior_preserved(tmp_path):
    configs = _configs(tmp_path)
    for key in ORDINARY:
        cfg = configs[key]
        assert cfg["rho"] == 0.1
        assert cfg["poison_select"] == "uniform"
        assert cfg["dose_min"] == 0.2
        assert cfg["dose_max"] == 1
        assert cfg["dose_coupling"] == "paired"
        assert cfg["theta_max_deg"] == 40
        assert cfg["comparison_peak_budget"] == "original_postclip_linf_v1"
        assert cfg["comparison_peak_reference_eps"] == 0.185
        assert cfg["comparison_peak_matches_l2"] is False
        assert "comparison_l2_budget" not in cfg
        assert not any(k.startswith("comparison_l2") for k in cfg)
    badnets = configs["badnets"]
    assert (badnets["badnets_patch_subcarriers"], badnets["badnets_patch_packets"]) == (3, 3)
    assert badnets["badnets_patch_opacity"] == 1
    assert configs["blended"]["eps"] == 0.2
    wanet = configs["wanet_source"]
    assert (wanet["wanet_grid_size"], wanet["wanet_strength"], wanet["wanet_grid_rescale"]) == (4, 0.5, 1.0)
    assert wanet["wanet_cover_ratio"] == 0.2
    assert wanet["num_workers"] == 0
    ftrojan = configs["ftrojan"]
    assert ftrojan["ftrojan_magnitude_255"] == 20
    assert ftrojan["ftrojan_channels"] == [1, 2]
    assert ftrojan["ftrojan_positions_32"] == [[31, 31], [15, 15]]
    fiba = configs["fiba"]
    assert (fiba["fiba_alpha"], fiba["fiba_beta"], fiba["fiba_cross_ratio"]) == (0.15, 0.1, 1.0)
    assert fiba["clean_label_cover_ratio"] == 0.1
    assert fiba["fiba_reference_train_index"] == 0
    assert fiba["num_workers"] == 0


def test_por_retains_historical_unlike_ordinary_objective_staged_protocol(tmp_path):
    configs = _configs(tmp_path)
    por, clean = configs["infocom2025_por"], configs["clean"]
    assert por["training_protocol"] == "infocom2025_por"
    assert por["rho"] == 0.4
    assert por["poison_select"] == "uniform"
    assert por["dose_min"] == por["dose_max"] == 1
    assert por["rf_training_payload_dose"] == por["rf_training_trigger_dose"] == 1
    assert por["rf_substitute_source"] == "downstream_train_unlabeled"
    assert por["rf_trigger_amp"] == 0.185
    assert por["rf_encoder_epochs"] == 50
    assert por["rf_downstream_head_epochs"] == por["victim_epochs"] == 50
    assert clean["rho"] == 0
    assert clean["epochs"] == 50
    assert "comparison_peak_budget" not in por
    assert clean["epochs"] + por["rf_encoder_epochs"] + por["rf_downstream_head_epochs"] == 150


def test_method_selection_does_not_require_other_yaml_files(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_dir.joinpath("blended.yaml").write_bytes((ROOT / "configs/mmfi/blended.yaml").read_bytes())
    matrix = _plan(tmp_path, methods=["blended"], config_dir=config_dir)
    assert [c["method_key"] for c in matrix["cells"]] == ["blended"]
    assert matrix["cells"][0]["dependencies"] == []


def test_por_selection_automatically_adds_clean(tmp_path):
    assert [c["method_key"] for c in _plan(tmp_path, methods=["infocom2025_por"])["cells"]] == [
        "clean", "infocom2025_por"]


def test_dry_run_touches_neither_dataset_nor_cuda(tmp_path, monkeypatch):
    import torch
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run attempted data/GPU/training")
    monkeypatch.setattr(launcher, "_check_inputs", forbidden)
    monkeypatch.setattr(launcher, "run_matrix", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    out = tmp_path / "dry"
    assert launcher.main(["--data-home", str(tmp_path / "absent"), "--outdir", str(out),
                          "--devices", "cuda:0", "cuda:1", "cuda:2", "--dry-run"]) == 0
    assert (out / launcher.MANIFEST).is_file()
    assert not (out / "baseline_report.json").exists()


@pytest.mark.parametrize("methods", [["proposed"], ["ccai2026_backdoorrf"], [], ["badnets", "badnets"]])
def test_unknown_empty_or_duplicate_method_fails(tmp_path, methods):
    with pytest.raises(ValueError):
        _plan(tmp_path, methods=methods)


def test_cli_unknown_method_fails():
    with pytest.raises(SystemExit):
        launcher.parse_args(["--methods", "proposed"])


def test_modified_config_refuses_existing_output(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "badnets.yaml"
    config_path.write_bytes((ROOT / "configs/mmfi/badnets.yaml").read_bytes())
    args = ["--data-home", str(tmp_path / "no_data"), "--outdir", str(tmp_path / "out"),
            "--config-dir", str(config_dir), "--methods", "badnets", "--dry-run"]
    assert launcher.main(args) == 0
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["lr"] = 0.002
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="NEW output directory"):
        launcher.main(args)


def test_python_source_edits_change_plan(tmp_path, monkeypatch):
    first = _plan(tmp_path)
    base = launcher._source_hashes()
    monkeypatch.setattr(launcher, "_source_hashes", lambda: {
        **base, "ATKBackd/attack/traditional.py": "a" * 64})
    assert _plan(tmp_path)["plan_sha256"] != first["plan_sha256"]
    with pytest.raises(ValueError, match="source changed"):
        launcher._verify_manifest(first)


def test_existing_foreign_output_is_not_imported(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "checkpoint.pt").write_bytes(b"foreign checkpoint")
    with pytest.raises(FileExistsError, match="NEW directory"):
        launcher.main(["--data-home", str(tmp_path / "no_data"), "--outdir", str(out), "--dry-run"])


def _fake_result():
    return dict(clean_mpjpe=0.18, clean_pampjpe=0.105,
        **{f"clean_pck@{t:.1f}": t for t in (0.5, 0.4, 0.3, 0.2, 0.1)},
        dose_grid=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], tmpjpe=[0.12] * 6,
        clean_to_target_tmpjpe=[0.12, 0.14, 0.20, 0.26, 0.33, 0.40])


def test_export_keeps_por_and_clean_separate(tmp_path, monkeypatch):
    import run_method_drafts
    monkeypatch.setattr(run_method_drafts, "_verified_cache", lambda cell: (
        _fake_result(), dict(method_key=cell["method_key"], cfg=cell["cfg"])))
    report = launcher.export_results(_plan(tmp_path), tmp_path / "export")
    assert [r["method_key"] for r in report["traditional_rows"]] == list(ORDINARY)
    assert [r["method_key"] for r in report["por_diagnostics"]] == ["infocom2025_por"]
    assert [r["method_key"] for r in report["clean_teacher_rows"]] == ["clean"]
    assert (tmp_path / "export/por_diagnostics.csv").is_file()
    assert (tmp_path / "export/clean_teacher_metrics.csv").is_file()
    csv = (tmp_path / "export/baseline_metrics.csv").read_text(encoding="utf-8")
    assert "infocom2025_por" not in csv
    assert "proposed" not in csv
    assert all(r["epochs"] == 50 and r["seed"] == 42 for r in report["traditional_rows"])


def test_export_does_not_silently_omit_failed_method(tmp_path, monkeypatch):
    import run_method_drafts
    def load(cell):
        if cell["method_key"] == "blended":
            raise ValueError("missing valid completed cache")
        return _fake_result(), {"method_key": cell["method_key"]}
    monkeypatch.setattr(run_method_drafts, "_verified_cache", load)
    with pytest.raises(ValueError, match="completed cache"):
        launcher.export_results(_plan(tmp_path), tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_export_only_fails_if_missing_cache(tmp_path):
    args = ["--data-home", str(tmp_path / "no_data"), "--outdir", str(tmp_path / "out"),
            "--methods", "badnets"]
    assert launcher.main([*args, "--dry-run"]) == 0
    with pytest.raises(RuntimeError, match="Missing completed cells"):
        launcher.main([*args, "--export-only"])
