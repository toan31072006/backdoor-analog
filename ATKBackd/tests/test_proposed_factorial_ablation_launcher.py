"""Factorial launch wiring and full-plan dry-run without dataset/CUDA access."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'remote_linux' / '17_run_proposed_factorial_ablation.sh'
DOC = ROOT / 'remote_linux' / 'PROPOSED_FACTORIAL_ABLATION_VI.md'
RUNNER = ROOT / 'ATKBackd' / 'run_proposed_ablation.py'


def _bash():
    candidates = [Path('C:/Program Files/Git/bin/bash.exe'),
                  Path('C:/Program Files/Git/usr/bin/bash.exe')]
    return next((str(path) for path in candidates if path.is_file()),
                shutil.which('bash'))


def test_factorial_launcher_selects_new_profile_and_quotes_paths():
    script = LAUNCHER.read_text(encoding='utf-8')
    assert 'set -euo pipefail' in script
    assert 'ATKBackd/run_proposed_ablation.py' in script
    assert '--factorial --data-home "${DOSE_DATA_HOME}" "${DOSE_FORWARDED[@]}"' in script
    assert '--no-peak' not in script
    assert '--python)' in script
    assert '--data-home)' in script
    assert 'DOSE_DATA_HOME}/envs/dose-backdoor/bin/python' in script
    assert 'DOSE_PY="$2"; shift 2' in script
    assert b'\r' not in LAUNCHER.read_bytes()
    assert 'kill ' not in script
    assert 'git push' not in script
    assert 'ssh ' not in script


def test_factorial_docs_declare_four_arms_rho_and_legacy_pck():
    doc = DOC.read_text(encoding='utf-8')
    assert 'mmfi_proposed_factorial_ablation_v1' in doc
    assert 'mmfi_proposed_factorial_ablation_s42_v1' in doc
    assert 'rho=0.1' in doc and '50 epoch' in doc
    assert 'mmfi-prop-factorial-s42' in doc
    assert 'CUDA_VISIBLE_DEVICES=0,2,3' in doc
    assert '--devices cuda:0 cuda:1 cuda:2' in doc
    assert '--devices cuda:0 cuda:1 cuda:3' not in doc
    assert all(key in doc for key in ('full', 'w_o_coupling', 'w_o_auxiliary', 'w_o_both'))
    assert 'Clean PCK@0.2' in doc
    assert '0.2 * ||gt[5] - gt[12]||' in doc
    assert 'không phải Target-PCK' in doc
    assert 'không phải 20 mm' in doc
    assert 'chưa được triển khai' in doc
    assert '--summary-only' in doc and '--fresh' in doc
    assert 'scp "mica-server:' in doc
    assert b'\r' not in DOC.read_bytes()


def test_factorial_launcher_bash_syntax():
    bash = _bash()
    if bash is None:
        pytest.skip('Bash is not installed; static launcher checks still run')
    completed = subprocess.run([bash, '-n', str(LAUNCHER)],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize('option', ['--python', '--data-home'])
@pytest.mark.parametrize('empty_value', [False, True])
def test_factorial_launcher_rejects_missing_path_values(option, empty_value):
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
def test_factorial_dry_run_needs_no_dataset_or_gpu(tmp_path, through_launcher):
    data_home = tmp_path / 'no dataset here'
    outdir = tmp_path / 'factorial plan with spaces'
    if through_launcher:
        bash = _bash()
        if bash is None:
            pytest.skip('Bash is not installed; direct runner test still runs')
        command = [bash, str(LAUNCHER), '--python', Path(sys.executable).as_posix()]
    else:
        command = [sys.executable, str(RUNNER), '--factorial']
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
    assert manifest['confirmation_profile'] == 'mmfi_proposed_factorial_ablation_v1'
    assert manifest['seed'] == 42
    assert manifest['metrics_contract']['rho'] == 0.1
    assert manifest['metrics_contract']['epochs'] == 50
    assert manifest['metrics_contract']['full_split_counts'] == {
        'train': 133056, 'eval': 33264}
    assert manifest['metrics_contract']['peak_bound_required_keys'] == []
    assert [cell['method_key'] for cell in manifest['cells']] == [
        'full', 'w_o_coupling', 'w_o_auxiliary', 'w_o_both']
    assert [cell['cfg']['trigger'] for cell in manifest['cells']] == [
        'md_multicarrier', 'md_multicarrier', 'micro_dropper', 'micro_dropper']
    assert [cell['cfg']['dose_coupling'] for cell in manifest['cells']] == [
        'paired', 'shuffled', 'paired', 'shuffled']
    assert all(cell['cfg']['rho'] == 0.1 for cell in manifest['cells'])
    assert not list(outdir.rglob('eval_cache.json'))
    assert not list(outdir.rglob('checkpoint.pt'))
