"""New no-peak launch wiring and data/CUDA-free plan generation."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'remote_linux' / '16_run_proposed_no_peak_ablation.sh'
DOC = ROOT / 'remote_linux' / 'PROPOSED_NO_PEAK_ABLATION_VI.md'
RUNNER = ROOT / 'ATKBackd' / 'run_proposed_ablation.py'


def _bash():
    candidates = [Path('C:/Program Files/Git/bin/bash.exe'),
                  Path('C:/Program Files/Git/usr/bin/bash.exe')]
    return next((str(path) for path in candidates if path.is_file()),
                shutil.which('bash'))


def test_no_peak_launcher_uses_same_runner_and_safe_forwarding():
    script = LAUNCHER.read_text(encoding='utf-8')
    assert 'set -euo pipefail' in script
    assert 'ATKBackd/run_proposed_ablation.py' in script
    assert '--no-peak --data-home "${DOSE_DATA_HOME}" "${DOSE_FORWARDED[@]}"' in script
    assert '--python)' in script
    assert '--data-home)' in script
    assert 'DOSE_DATA_HOME}/envs/dose-backdoor/bin/python' in script
    assert 'DOSE_PY="$2"; shift 2' in script
    assert b'\r' not in LAUNCHER.read_bytes()
    assert 'kill ' not in script
    assert 'git push' not in script
    assert 'ssh ' not in script


def test_no_peak_docs_separate_profile_and_clean_relative_pck():
    doc = DOC.read_text(encoding='utf-8')
    assert 'mmfi_proposed_no_peak_ablation_v1' in doc
    assert 'mmfi_proposed_no_peak_ablation_s42_v1' in doc
    assert 'CUDA_VISIBLE_DEVICES=0,2,3' in doc
    assert '--devices cuda:0 cuda:1 cuda:2' in doc
    assert '--devices cuda:0 cuda:1 cuda:3' not in doc
    assert 'w_o_coupling' in doc and 'w_o_auxiliary' in doc
    assert 'Clean PCK@0.2' in doc
    assert '0.2 * ||gt[5] - gt[12]||' in doc
    assert 'không phải attack PCK' in doc
    assert 'không phải 20 mm' in doc
    assert 'profile **ba nhánh, rho=0.4**' in doc
    assert '17_run_proposed_factorial_ablation.sh' in doc
    assert '--summary-only' in doc


def test_no_peak_launcher_bash_syntax():
    bash = _bash()
    if bash is None:
        pytest.skip('Bash is not installed; static launcher checks still run')
    completed = subprocess.run([bash, '-n', str(LAUNCHER)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize('option', ['--python', '--data-home'])
@pytest.mark.parametrize('empty_value', [False, True])
def test_no_peak_launcher_rejects_missing_path_values(option, empty_value):
    bash = _bash()
    if bash is None:
        pytest.skip('Bash is not installed; static launcher checks still run')
    args = [bash, str(LAUNCHER), option]
    if empty_value:
        args.append('')
    completed = subprocess.run(args, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 2
    assert f'Missing {option} value' in completed.stderr


@pytest.mark.parametrize('through_launcher', [False, True])
def test_no_peak_runner_dry_run_requires_no_data_or_gpu(tmp_path, through_launcher):
    data_home = tmp_path / 'no dataset here'
    outdir = tmp_path / 'no peak plan'
    if through_launcher:
        bash = _bash()
        if bash is None:
            pytest.skip('Bash is not installed; direct runner test still runs')
        command = [bash, str(LAUNCHER), '--python', Path(sys.executable).as_posix()]
    else:
        command = [sys.executable, str(RUNNER), '--no-peak']
    completed = subprocess.run(
        [*command, '--data-home', data_home.as_posix(),
         '--outdir', outdir.as_posix(), '--seed', '42', '--devices', 'cuda:999',
         '--num-workers', '4', '--dry-run'],
        capture_output=True, text=True, encoding='utf-8', timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'DRY RUN' in completed.stdout
    assert 'NO data/GPU/training' in completed.stdout
    assert not data_home.exists()
    manifest = json.loads((outdir / 'proposed_ablation.resolved.json').read_text(
        encoding='utf-8'))
    assert manifest['confirmation_profile'] == 'mmfi_proposed_no_peak_ablation_v1'
    assert [cell['method_key'] for cell in manifest['cells']] == [
        'full', 'w_o_coupling', 'w_o_auxiliary']
    assert [cell['cfg']['trigger'] for cell in manifest['cells']] == [
        'md_multicarrier', 'md_multicarrier', 'micro_dropper']
    assert [cell['cfg']['dose_coupling'] for cell in manifest['cells']] == [
        'paired', 'shuffled', 'paired']
    assert not list(outdir.rglob('eval_cache.json'))
    assert not list(outdir.rglob('checkpoint.pt'))
