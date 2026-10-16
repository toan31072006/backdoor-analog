"""T1/T2/T3 launcher wiring; no benchmark training or CUDA is used here."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'remote_linux' / '23_run_a1_three_tables.sh'
DOC = ROOT / 'remote_linux' / 'A1_THREE_TABLES_VI.md'
RUNNER = ROOT / 'ATKBackd' / 'run_a1_three_tables.py'


def _bash():
    candidates = [Path('C:/Program Files/Git/bin/bash.exe'),
                  Path('C:/Program Files/Git/usr/bin/bash.exe')]
    return next((str(path) for path in candidates if path.is_file()),
                shutil.which('bash'))


def test_three_tables_launcher_quotes_paths_and_forwards_user_options():
    script = LAUNCHER.read_text(encoding='utf-8')
    assert 'set -euo pipefail' in script
    assert 'ATKBackd/run_a1_three_tables.py' in script
    assert '--data-home "${DOSE_DATA_HOME}" "${DOSE_FORWARDED[@]}"' in script
    assert '--python)' in script and '--data-home)' in script
    assert 'DOSE_DATA_HOME}/envs/dose-backdoor/bin/python' in script
    assert 'DOSE_PY="$2"; shift 2' in script
    assert b'\r' not in LAUNCHER.read_bytes()
    for disallowed in ('kill ', 'git push', 'ssh ', 'cuda_visible_devices=',
                       '--retrain-seed42', '--fresh', 'blended'):
        assert disallowed not in script.lower()


def test_three_tables_docs_scope_reuse_metrics_and_gpu_mapping():
    doc = DOC.read_text(encoding='utf-8')
    assert 'mmfi_a1_three_tables_s42_43_44_v1' in doc
    assert '26 victim mới' in doc and '39 victim mới' in doc
    assert 'Không train A1 ba lần' in doc
    assert all(path in doc for path in (
        'mmfi_a1_native_comparison_s42_v3',
        'mmfi_proposed_factorial_ablation_s42_v1',
        'mmfi_a1_dose_poison_s42_v1',
        'mmfi_tables_s42_20261009_003044/clean'))
    assert all(option in doc for option in (
        '--seed42-comparison', '--seed42-ablation', '--seed42-dose-poison',
        '--seed42-clean', '--retrain-seed42', '--export-only', '--status', '--fresh',
        '--cells s43__badnets'))
    assert 'Target-PCK' in doc and 'clean_gt[5] - clean_gt[12]' in doc
    assert 'Không phải 20 mm' in doc
    assert 'mean ± sample SD (ddof=1)' in doc
    assert '50 epochs' in doc and '133056' in doc and '33264' in doc
    assert 'SIG giữ delta=40, f=6' in doc
    assert 'squeeze_num=8' in doc and 'neg_ratio=0.1' in doc
    assert 'CUDA_VISIBLE_DEVICES=0,2,3' in doc
    assert '--devices cuda:0 cuda:1 cuda:2' in doc
    assert '--devices cuda:0 cuda:1 cuda:3' not in doc
    assert 'set -o pipefail;' in doc
    assert '"$DOSE_THREE_OUT"/seed*/*/console.log' in doc
    assert 'three_main_tables.json' in doc and '39/39' in doc
    assert '*.npz' in doc and "'*.pt'" not in doc
    assert 'scp "mica-server:' in doc
    assert 'blended' not in doc.lower()
    assert b'\r' not in DOC.read_bytes()


def test_three_tables_launcher_bash_syntax():
    bash = _bash()
    if bash is None:
        pytest.skip('Bash is not installed; static checks still run')
    completed = subprocess.run([bash, '-n', str(LAUNCHER)],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize('option', ['--python', '--data-home'])
@pytest.mark.parametrize('empty_value', [False, True])
def test_three_tables_launcher_rejects_missing_path_values(option, empty_value):
    bash = _bash()
    if bash is None:
        pytest.skip('Bash is not installed; static checks still run')
    args = [bash, str(LAUNCHER), option]
    if empty_value:
        args.append('')
    completed = subprocess.run(args, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 2
    assert f'Missing {option} value' in completed.stderr


def test_three_tables_launcher_reports_missing_interpreter(tmp_path):
    bash = _bash()
    if bash is None:
        pytest.skip('Bash is not installed; static checks still run')
    interpreter = tmp_path / 'missing Python with spaces'
    completed = subprocess.run(
        [bash, str(LAUNCHER), '--python', interpreter.as_posix(), '--dry-run'],
        capture_output=True, text=True, timeout=20)
    assert completed.returncode == 1
    assert 'Missing Python:' in completed.stderr
    assert 'Set DOSE_PY or use --python.' in completed.stderr


@pytest.mark.parametrize('through_launcher', [False, True])
def test_three_tables_dry_run_no_data_gpu_or_references_needed(tmp_path, through_launcher):
    data_home = tmp_path / 'no dataset here'
    outdir = tmp_path / 'three table plan with spaces'
    reference_options = []
    for name in ('comparison', 'ablation', 'dose-poison', 'clean'):
        reference_options.extend([
            f'--seed42-{name}', (tmp_path / f'absent reference {name}').as_posix()])
    if through_launcher:
        bash = _bash()
        if bash is None:
            pytest.skip('Bash is not installed; direct runner test still runs')
        command = [bash, str(LAUNCHER), '--python', Path(sys.executable).as_posix()]
    else:
        command = [sys.executable, str(RUNNER)]
    completed = subprocess.run(
        [*command, '--data-home', data_home.as_posix(),
         '--outdir', outdir.as_posix(), *reference_options,
         '--devices', 'cuda:999', '--num-workers', '4', '--dry-run'],
        capture_output=True, text=True, encoding='utf-8', timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'DRY RUN' in completed.stdout
    assert not data_home.exists()
    assert not list(outdir.rglob('eval_cache.json'))
    assert not list(outdir.rglob('checkpoint.pt'))
    manifest_path = outdir / 'three_tables.resolved.json'
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert 'mmfi_a1_three_tables_v1' in json.dumps(manifest)
