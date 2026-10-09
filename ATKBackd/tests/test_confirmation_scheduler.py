"""The confirmation runner shares the independent, per-device MM-Fi scheduler."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_mmfi_tables as runner  # noqa: E402


@pytest.mark.parametrize("custom_worker", [False, True], ids=["default", "custom"])
def test_two_devices_run_concurrently_with_selected_worker(tmp_path, monkeypatch, custom_worker):
    matrix = {"cells": [
        {"method_key": key, "dependencies": [], "ckpt_dir": str(tmp_path / key),
         "cfg": {"device": "cpu", "epochs": 50, "seed": 42, "rho": 0.4}}
        for key in ("original", "peak_matched")
    ]}
    original_matrix = copy.deepcopy(matrix)
    manifest = tmp_path / "matrix.json"
    devices = ["cuda:0", "cuda:1"]
    spawned = []
    active_at_sleep = []
    completed = []

    def unexpected_rf_config(cfg):
        pytest.fail("Ordinary confirmation cells must not finalize an RF config")

    monkeypatch.setitem(sys.modules, "train_rf_backdoor", SimpleNamespace(
        finalize_rf_config=unexpected_rf_config))

    class FakeChild:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            self.pid = 100 + len(spawned)
            self.returncode = None
            self.key = command[4]
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            cell = next(cell for cell in saved["cells"] if cell["method_key"] == self.key)
            self.device = cell["cfg"]["device"]
            spawned.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self, timeout=None):
            return self.returncode

    def finish_workers(seconds):
        assert seconds == 1
        live = [child for child in spawned if child.poll() is None]
        active_at_sleep.append([(child.key, child.device) for child in live])
        for child in live:
            child.returncode = 0

    def complete(cell):
        completed.append(cell["method_key"])
        return True

    monkeypatch.setattr(runner.subprocess, "Popen", FakeChild)
    monkeypatch.setattr(runner.time, "sleep", finish_workers)
    monkeypatch.setattr(runner, "_complete", complete)
    monkeypatch.chdir(tmp_path)
    kwargs = {"worker_script": Path("run_peak_confirmation.py")} if custom_worker else {}

    runner.run_matrix(matrix, manifest, devices, **kwargs)

    assert active_at_sleep == [[("original", "cuda:0"), ("peak_matched", "cuda:1")]]
    assert completed == ["original", "peak_matched"]
    expected_script = (tmp_path / "run_peak_confirmation.py" if custom_worker
                       else Path(runner.__file__)).resolve()
    assert [child.command for child in spawned] == [
        [sys.executable, "-u", str(expected_script), "--_cell", key,
         "--_manifest", str(manifest)]
        for key in ("original", "peak_matched")
    ]
    assert all(child.kwargs["cwd"] == runner.HERE for child in spawned)
    assert all(child.kwargs["stderr"] == runner.subprocess.STDOUT for child in spawned)
    assert all(child.kwargs["stdout"].closed for child in spawned)
    for cell, original in zip(matrix["cells"], original_matrix["cells"]):
        cell = copy.deepcopy(cell)
        cell["cfg"]["device"] = "cpu"
        assert cell == original
