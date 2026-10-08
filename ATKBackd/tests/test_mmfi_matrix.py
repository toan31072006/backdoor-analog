"""Model-free matrix, scheduling, and resume checks for the four-table runner."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_mmfi_tables as runner  # noqa: E402
import train_backdoor as common  # noqa: E402
import train_rf_backdoor as rf  # noqa: E402


MAIN_KEYS = {"infocom2025_por", "ccai2026_backdoorrf", "badnets", "blended", "wanet", "prop_rho0p4"}


@pytest.fixture
def matrix(tmp_path):
    return runner.build_matrix(tmp_path / "absent_data", tmp_path / "runs", device="cpu", num_workers=0)


def _by_key(matrix):
    return {cell["method_key"]: cell for cell in matrix["cells"]}


def _forbidden(*args, **kwargs):
    raise AssertionError("This test must not load data, inspect CUDA, or train a model")


def test_matrix_has_ten_unique_cells_and_six_main_methods(matrix):
    cells = matrix["cells"]
    assert len(cells) == 10
    assert len({cell["method_key"] for cell in cells}) == 10
    assert len({cell["label"] for cell in cells}) == 10
    assert len({cell["ckpt_dir"] for cell in cells}) == 10
    assert {cell["method_key"] for cell in cells if 1 in cell["tables"]} == MAIN_KEYS
    assert matrix["seed"] == 42 and matrix["dataset"] == "mmfi"
    assert matrix["fresh_results_only"] is True
    assert matrix["metrics_contract"]["main_asr"] is False
    assert matrix["metrics_contract"]["uncertainty"] == "single seed: not estimated"
    assert all(Path(cell["eval_cache"]).parent == Path(cell["ckpt_dir"]) for cell in cells)


def test_seed_recipe_split_and_target_geometry_are_shared(matrix):
    expected = {
        "seed": 42, "model": "hpeli", "epochs": 50, "victim_epochs": 50,
        "optimizer": "sgd", "lr": 0.001, "momentum": 0.9, "weight_decay": 0.0,
        "batch_size": 32, "victim_loss": "mpjpe", "lr_scheduler": False,
        "mmfi_protocol": "protocol1", "mmfi_setting": "s1", "mmfi_split_seed": 0,
        "mmfi_random_ratio": 0.8, "pivot": 1, "theta_max_deg": 40.0,
        "payload_axis": [0.0, 0.0, 1.0], "dose_grid": runner.GRID,
    }
    for cell in matrix["cells"]:
        for key, value in expected.items():
            assert cell["cfg"][key] == value, (cell["method_key"], key)
    assert matrix["metrics_contract"]["target_joints"] == [2, 3]


def test_table_membership_and_poison_rates_do_not_duplicate_training(matrix):
    cells = _by_key(matrix)
    assert cells["prop_rho0p4"]["tables"] == [1, 2, 3, 4]
    assert cells["clean"]["tables"] == [3]
    assert cells["shuffled"]["tables"] == [3]
    assert cells["prop_rho0p1"]["tables"] == cells["prop_rho0p2"]["tables"] == [4]
    assert cells["clean"]["cfg"]["rho"] == 0.0
    for key in MAIN_KEYS | {"shuffled"}:
        assert cells[key]["cfg"]["rho"] == 0.4
    assert cells["prop_rho0p1"]["cfg"]["rho"] == 0.1
    assert cells["prop_rho0p2"]["cfg"]["rho"] == 0.2
    assert cells["shuffled"]["cfg"]["dose_coupling"] == "shuffled"
    for key in ("clean", "prop_rho0p1", "prop_rho0p2", "prop_rho0p4"):
        assert cells[key]["cfg"]["dose_coupling"] == "paired"


def test_fixed_baseline_and_continuous_proposed_training_doses_are_distinct(matrix):
    for cell in matrix["cells"]:
        cfg = cell["cfg"]
        assert cfg["dose_max"] == 1.0
        assert cfg["dose_min"] == (1.0 if cell["method_key"] in MAIN_KEYS - {"prop_rho0p4"} else 0.2)
    assert _by_key(matrix)["wanet"]["cfg"]["wanet_cover_ratio"] == 0.2


def test_rf_protocols_keep_their_stages_and_explicit_clean_dependency(matrix):
    cells = _by_key(matrix)
    info, backdoor = cells["infocom2025_por"], cells["ccai2026_backdoorrf"]
    assert info["dependencies"] == ["clean"]
    assert info["cfg"]["rf_clean_teacher_checkpoint"] == str(Path(cells["clean"]["ckpt_dir"]) / "checkpoint.pt")
    assert info["cfg"]["training_protocol"] == "infocom2025_por"
    assert info["cfg"]["rf_substitute_source"] == "downstream_train_unlabeled"
    assert info["cfg"]["rf_encoder_epochs"] == 50
    assert info["cfg"]["rf_downstream_head_epochs"] == 50
    assert backdoor["dependencies"] == []
    assert backdoor["cfg"]["training_protocol"] == "ccai2026_backdoorrf"
    assert backdoor["cfg"]["rf_clean_pretrain_epochs"] == 15
    assert backdoor["cfg"]["rf_joint_epochs"] == 35
    assert backdoor["cfg"]["rf_trigger_warmup_epochs"] == 5


def test_real_resolved_matrix_is_compatible_with_four_table_exporter(matrix):
    from mmfi_tables import build_tables

    result = {
        "clean_mpjpe": 0.10, "clean_pampjpe": 0.08,
        "dose_grid": list(runner.GRID), "reference_dose": 1.0,
        "tmpjpe": [0.07, 0.065, 0.06, 0.05, 0.04, 0.03],
        "clean_to_target_tmpjpe": [0.07, 0.09, 0.11, 0.13, 0.15, 0.17],
    }
    for threshold in (0.5, 0.4, 0.3, 0.2, 0.1):
        result[f"clean_pck@{threshold:.1f}"] = threshold + 0.4
    for cell in matrix["cells"]:
        runner._atomic_json(cell["eval_cache"], {
            "cfg": cell["cfg"], "cfg_fingerprint": common._config_fingerprint(cell["cfg"]),
            "trained_epochs": cell["cfg"]["epochs"], "result_schema": common._RESULT_SCHEMA,
            "res": result,
        })
    report = build_tables(matrix)
    assert report["status"] == "official_complete"
    assert len(report["tables"]["table1_main"]) == 6
    assert len(report["tables"]["table2_dose_response"]) == 6


def test_info_threat_model_discloses_the_downstream_data_relaxation(matrix):
    cfg = _by_key(matrix)["infocom2025_por"]["cfg"]
    assert "downstream" in cfg["threat_model"]
    assert "unlabeled" in cfg["threat_model"]
    assert "substitute" in cfg["attacker_access"]


def test_external_substitute_is_resolved_without_reading_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_sha256", _forbidden)
    substitute = tmp_path / "nonexistent_substitute.npy"
    matrix = runner.build_matrix(tmp_path / "absent_data", tmp_path / "output", "cpu", 0, substitute)
    cfg = _by_key(matrix)["infocom2025_por"]["cfg"]
    assert cfg["rf_substitute_source"] == "external_unlabeled"
    assert cfg["rf_substitute_path"] == str(substitute.resolve())
    assert not substitute.exists()


def test_matrix_configs_are_independent_copies(matrix):
    cells = _by_key(matrix)
    cells["prop_rho0p4"]["cfg"]["dose_grid"][1] = 0.123
    assert cells["clean"]["cfg"]["dose_grid"] == runner.GRID
    assert cells["shuffled"]["cfg"]["dose_grid"] == runner.GRID


def test_scientific_plan_hash_ignores_device_and_loader_settings(tmp_path):
    cpu = runner.build_matrix(tmp_path / "data", tmp_path / "run", "cpu", 0)
    cuda = runner.build_matrix(tmp_path / "data", tmp_path / "run", "cuda:7", 8)
    assert cpu["plan_sha256"] == cuda["plan_sha256"]
    changed = runner.build_matrix(tmp_path / "other_data", tmp_path / "run", "cpu", 0)
    assert cpu["plan_sha256"] != changed["plan_sha256"]


def test_dry_run_writes_manifest_and_lock_without_data_gpu_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_check_inputs", _forbidden)
    monkeypatch.setattr(runner, "run_matrix", _forbidden)
    monkeypatch.setattr(common, "train", _forbidden)
    monkeypatch.setattr(common, "_load_dataset", _forbidden)
    monkeypatch.setattr(rf, "train", _forbidden)
    monkeypatch.setattr(rf, "finalize_rf_config", _forbidden)
    monkeypatch.setattr(rf, "_sha256", _forbidden)
    monkeypatch.setattr(common.np, "load", _forbidden)
    monkeypatch.setattr(common.torch, "load", _forbidden)
    monkeypatch.setattr(common.torch.cuda, "is_available", _forbidden)
    monkeypatch.setattr(common.torch.cuda, "get_device_properties", _forbidden)
    output = tmp_path / "fresh_dry_run"
    assert runner.main(["--data-home", str(tmp_path / "missing_data"), "--outdir", str(output), "--fresh", "--dry-run"]) == 0
    assert {path.name for path in output.iterdir()} == {"mmfi_matrix.resolved.json", "run.lock"}
    assert (output / "run.lock").read_bytes() == b"0"
    stored = json.loads((output / "mmfi_matrix.resolved.json").read_text(encoding="utf-8"))
    assert len(stored["cells"]) == 10


def test_fresh_refuses_old_nonempty_directory_without_changing_anything(tmp_path, monkeypatch):
    output = tmp_path / "old_results"
    output.mkdir()
    marker = output / "checkpoint.pt"
    marker.write_bytes(b"existing user checkpoint")
    monkeypatch.setattr(runner, "build_matrix", _forbidden)
    with pytest.raises(FileExistsError, match="NEW path"):
        runner.main(["--data-home", str(tmp_path / "data"), "--outdir", str(output), "--fresh", "--dry-run"])
    assert marker.read_bytes() == b"existing user checkpoint"
    assert list(output.iterdir()) == [marker]


def test_selecting_info_implicitly_schedules_its_clean_teacher(matrix):
    selected = runner._select_cells(matrix, ["infocom2025_por"])
    assert [cell["method_key"] for cell in selected] == ["clean", "infocom2025_por"]
    selected = runner._select_cells(matrix, ["prop_rho0p4"])
    assert [cell["method_key"] for cell in selected] == ["prop_rho0p4"]
    with pytest.raises(ValueError, match="Unknown cells"):
        runner._select_cells(matrix, ["unknown"])


def test_cell_lock_rejects_concurrent_owner_and_releases_after_exception(tmp_path):
    folder = tmp_path / "cell"
    with runner._cell_lock(folder):
        with pytest.raises(RuntimeError, match="Another process"):
            with runner._cell_lock(folder):
                pytest.fail("A second owner acquired the same cell lock")
    with pytest.raises(RuntimeError, match="simulated failure"):
        with runner._cell_lock(folder):
            raise RuntimeError("simulated failure")
    with runner._cell_lock(folder):
        pass


def test_os_releases_cell_lock_after_worker_crash(tmp_path):
    folder = tmp_path / "crashed_worker"
    script = (
        "import os, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "from run_mmfi_tables import _cell_lock; "
        "lock = _cell_lock(sys.argv[2]); lock.__enter__(); "
        "print('LOCKED', flush=True); sys.stdin.readline(); os._exit(23)"
    )
    command = [sys.executable, "-u", "-c", script, str(runner.HERE), str(folder)]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RuntimeError, match="Another process"):
            with runner._cell_lock(folder):
                pytest.fail("Acquired a live worker's lock")
        child.stdin.write("exit\n")
        child.stdin.flush()
        assert child.wait(timeout=10) == 23
        with runner._cell_lock(folder):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()


class _Process:
    def __init__(self, name, code=0):
        self.name, self.code = name, code
        self.pid = 1000
        self.terminated = self.killed = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def test_scheduler_finalizes_info_only_after_clean_dependency_completed(matrix, tmp_path, monkeypatch):
    events, children = [], []

    def launch(command, **kwargs):
        method = command[command.index("--_cell") + 1]
        events.append(f"launch:{method}")
        child = _Process(method)
        children.append(child)
        return child

    def complete(cell):
        events.append(f"complete:{cell['method_key']}")
        return True

    def finalize(cfg):
        assert "complete:clean" in events
        events.append("finalize:infocom2025_por")
        return dict(cfg, rf_clean_teacher_sha256="bound-to-new-clean-teacher")

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner, "_complete", complete)
    monkeypatch.setattr(rf, "finalize_rf_config", finalize)
    monkeypatch.setattr(runner.time, "sleep", _forbidden)
    path = tmp_path / "manifest.json"
    runner.run_matrix(matrix, path, ["cpu"], requested=["infocom2025_por"])
    assert [child.name for child in children] == ["clean", "infocom2025_por"]
    assert events.index("complete:clean") < events.index("finalize:infocom2025_por") < events.index("launch:infocom2025_por")
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert _by_key(stored)["infocom2025_por"]["cfg"]["rf_clean_teacher_sha256"] == "bound-to-new-clean-teacher"
    assert all(not child.terminated for child in children)


def test_scheduler_failure_stops_only_its_owned_running_children(matrix, tmp_path, monkeypatch):
    own = []
    foreign = _Process("unrelated_gpu_job", None)

    def launch(command, **kwargs):
        method = command[command.index("--_cell") + 1]
        child = _Process(method, 7 if method == "clean" else None)
        own.append(child)
        return child

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner, "_complete", _forbidden)
    monkeypatch.setattr(runner.os, "kill", _forbidden)
    monkeypatch.setattr(runner.time, "sleep", _forbidden)
    with pytest.raises(RuntimeError, match="clean failed"):
        runner.run_matrix(matrix, tmp_path / "manifest.json", ["cuda:0", "cuda:1"], requested=["clean", "prop_rho0p4"])
    assert len(own) == 2
    assert own[1].terminated is True
    assert foreign.terminated is False and foreign.killed is False


def test_scheduler_rejects_worker_success_without_completed_cache(matrix, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: _Process("prop_rho0p4"))
    monkeypatch.setattr(runner, "_complete", lambda cell: False)
    with pytest.raises(RuntimeError, match="without a valid result"):
        runner.run_matrix(matrix, tmp_path / "manifest.json", ["cpu"], requested=["prop_rho0p4"])


def test_resume_rejects_requested_scientific_plan_change_before_overwriting_manifest(tmp_path, monkeypatch):
    output = tmp_path / "resumed"
    old = runner.build_matrix(tmp_path / "original_data", output, "cpu", 0)
    path = output / "mmfi_matrix.resolved.json"
    runner._atomic_json(path, old)
    before = path.read_bytes()
    monkeypatch.setattr(runner, "_check_inputs", _forbidden)
    with pytest.raises(ValueError, match="Existing matrix differs"):
        runner.main(["--data-home", str(tmp_path / "different_data"), "--outdir", str(output), "--devices", "cpu", "--dry-run"])
    assert path.read_bytes() == before


def test_resume_rejects_modified_stored_cfg_even_if_hash_field_was_not_changed(tmp_path, monkeypatch):
    output, data = tmp_path / "resumed", tmp_path / "data"
    old = runner.build_matrix(data, output, "cpu", 0)
    _by_key(old)["badnets"]["cfg"]["lr"] = 0.009
    path = output / "mmfi_matrix.resolved.json"
    runner._atomic_json(path, old)
    before = path.read_bytes()
    monkeypatch.setattr(runner, "_check_inputs", _forbidden)
    with pytest.raises(ValueError, match="matrix|config|plan|differs"):
        runner.main(["--data-home", str(data), "--outdir", str(output), "--devices", "cpu", "--dry-run"])
    assert path.read_bytes() == before


def test_resume_allows_runtime_change_and_keeps_finalized_teacher_hash(tmp_path):
    output, data = tmp_path / "resumed", tmp_path / "data"
    old = runner.build_matrix(data, output, "cpu", 0)
    _by_key(old)["infocom2025_por"]["cfg"]["rf_clean_teacher_sha256"] = "previously-bound-teacher"
    path = output / "mmfi_matrix.resolved.json"
    runner._atomic_json(path, old)
    assert runner.main(["--data-home", str(data), "--outdir", str(output), "--devices", "cuda:7", "--num-workers", "8", "--dry-run"]) == 0
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["plan_sha256"] == old["plan_sha256"]
    assert all(cell["cfg"]["num_workers"] == 8 for cell in stored["cells"])
    assert _by_key(stored)["infocom2025_por"]["cfg"]["rf_clean_teacher_sha256"] == "previously-bound-teacher"


@pytest.mark.parametrize("extra, error", [
    (["--devices", "cpu", "cpu"], "distinct"), (["--devices", "cuda"], "cuda:N"),
    (["--num-workers", "-1"], "cannot be negative"),
])
def test_invalid_runtime_options_fail_before_outputs(tmp_path, extra, error):
    output = tmp_path / "no_output"
    with pytest.raises(ValueError, match=error):
        runner.main(["--data-home", str(tmp_path / "data"), "--outdir", str(output), "--dry-run", *extra])
    assert not output.exists()
