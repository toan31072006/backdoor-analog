"""MICA launch wiring and safe logical GPU remapping; no dataset or GPU access."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'remote_linux' / '15_run_proposed_ablation.sh'
DOC = ROOT / 'remote_linux' / 'PROPOSED_ABLATION_VI.md'


def test_launcher_uses_new_runner_and_preserves_forwarded_arguments():
    script = LAUNCHER.read_text(encoding='utf-8')
    assert 'set -euo pipefail' in script
    assert 'ATKBackd/run_proposed_ablation.py' in script
    assert '--data-home "${DOSE_DATA_HOME}" "${DOSE_FORWARDED[@]}"' in script
    assert '--python)' in script
    assert '--data-home)' in script
    assert 'DOSE_DATA_HOME}/envs/dose-backdoor/bin/python' in script
    assert 'DOSE_PY="$2"; shift 2' in script
    assert '\r' not in script
    assert 'kill ' not in script


def test_mica_docs_use_three_logical_devices_for_physical_0_2_3():
    doc = DOC.read_text(encoding='utf-8')
    assert 'CUDA_VISIBLE_DEVICES=0,2,3' in doc
    assert '--devices cuda:0 cuda:1 cuda:2' in doc
    assert '--devices cuda:0 cuda:1 cuda:3' not in doc
    assert 'mmfi-prop-abl' in doc
    assert 'mmfi_proposed_ablation_s42_v1' in doc
    assert 'w_o_coupling' in doc
    assert 'w_o_auxiliary' in doc
    assert 'w_o_peak' in doc
    assert '--summary-only' in doc


def test_launcher_bash_syntax():
    candidates = [Path('C:/Program Files/Git/bin/bash.exe'),
                  Path('C:/Program Files/Git/usr/bin/bash.exe')]
    bash = next((str(path) for path in candidates if path.is_file()),
                shutil.which('bash'))
    if bash is None:
        pytest.skip('Bash is not installed; static launcher checks still run')
    completed = subprocess.run([bash, '-n', str(LAUNCHER)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
