"""Native comparison launcher; no benchmark training or GPU is used here."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'remote_linux' / '18_run_a1_native_comparison.sh'
DOC = ROOT / 'remote_linux' / 'A1_NATIVE_COMPARISON_VI.md'
RUNNER = ROOT / 'ATKBackd' / 'run_a1_native_comparison.py'


def _bash():
    candidates = [Path('C:/Program Files/Git/bin/bash.exe'),
                  Path('C:/Program Files/Git/usr/bin/bash.exe')]
    return next((str(path) for path in candidates if path.is_file()),
                shutil.which('bash'))


def test_native_launcher_quotes_paths_and_has_no_hidden_training_flags():
    script = LAUNCHER.read_text(encoding='utf-8')
    assert 'set -euo pipefail' in script
    assert 'ATKBackd/run_a1_native_comparison.py' in script
    assert '--data-home "${DOSE_DATA_HOME}" "${DOSE_FORWARDED[@]}"' in script
    assert '--python)' in script
    assert '--data-home)' in script
    assert 'DOSE_DATA_HOME}/envs/dose-backdoor/bin/python' in script
    assert 'DOSE_PY="$2"; shift 2' in script
    assert b'\r' not in LAUNCHER.read_bytes()
    for disallowed in ('kill ', 'git push', 'ssh ', '--seed', '--factorial',
                       '--no-peak', '--fresh', 'blended'):
        assert disallowed not in script.lower()


def test_native_docs_freeze_a1_and_explain_native_comparison():
    doc = DOC.read_text(encoding='utf-8')
    assert 'mmfi_a1_native_comparison_v3' in doc
    assert 'mmfi_a1_native_comparison_s42_v3' in doc
    assert 'mmfi_proposed_factorial_ablation_s42_v1/full' in doc
    assert 'mmfi-a1-compare-s42' in doc
    assert 'rho=0.1' in doc and '50 epochs' in doc
    assert all(key in doc for key in ('badnets', 'wanet_source', 'ftrojan', 'fiba',
                                      'sig_source', 'bpp_source'))
    assert '133056' in doc and '33264' in doc
    assert '--a1-reference' in doc
    assert 'Không train lại A1' in doc
    assert 'Không match peak thực tế hoặc L2' in doc
    assert 'num_workers=0' in doc
    assert 'dirty-label pose adaptation' in doc
    assert 'không phải tái hiện đầy đủ' in doc
    assert 'delta=40' in doc and 'f=6' in doc
    assert 'squeeze_num=8' in doc and 'neg_ratio=0.1' in doc
    assert 'không vendored' in doc
    assert 'cố định 10% recipient cover' in doc
    assert 'chọn lại vị trí mẫu negative recipient' in doc
    assert 'Target-PCK' in doc and 'không phải 20 mm' in doc
    assert 'BackdoorBench' in doc
    assert 'CUDA_VISIBLE_DEVICES=0,2,3' in doc
    assert '--devices cuda:0 cuda:1 cuda:2' in doc
    assert '--devices cuda:0 cuda:1 cuda:3' not in doc
    assert '--summary-only' in doc and '--fresh' in doc
    assert 'sáu cache baseline hợp lệ' in doc
    assert 'summary bảy hàng' in doc
    assert 'scp "mica-server:' in doc
    assert 'Blended không thuộc lượt comparison' in doc
    assert 'mmfi_a1_native_comparison_s42_v2' not in doc
    assert b'\r' not in DOC.read_bytes()


def test_native_launcher_bash_syntax():
    bash = _bash()
    if bash is None:
        pytest.skip('Bash is not installed; static launcher checks still run')
    completed = subprocess.run([bash, '-n', str(LAUNCHER)],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize('option', ['--python', '--data-home'])
@pytest.mark.parametrize('empty_value', [False, True])
def test_native_launcher_rejects_missing_path_values(option, empty_value):
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
def test_native_dry_run_no_data_gpu_or_reference_needed(tmp_path, through_launcher):
    data_home = tmp_path / 'no dataset here'
    reference = tmp_path / 'no A1 reference yet.zip'
    outdir = tmp_path / 'native comparison with spaces'
    if through_launcher:
        bash = _bash()
        if bash is None:
            pytest.skip('Bash is not installed; direct runner test still runs')
        command = [bash, str(LAUNCHER), '--python', Path(sys.executable).as_posix()]
    else:
        command = [sys.executable, str(RUNNER)]
    completed = subprocess.run(
        [*command, '--data-home', data_home.as_posix(),
         '--outdir', outdir.as_posix(), '--a1-reference', reference.as_posix(),
         '--devices', 'cuda:999', '--num-workers', '4', '--dry-run'],
        capture_output=True, text=True, encoding='utf-8', timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'DRY RUN' in completed.stdout
    assert not data_home.exists()
    assert not reference.exists()
    manifest = json.loads((outdir / 'a1_native_comparison.resolved.json').read_text(
        encoding='utf-8'))
    assert manifest['traditional_comparison_profile'] == 'mmfi_a1_native_comparison_v3'
    assert manifest['seed'] == 42
    assert [cell['method_key'] for cell in manifest['cells']] == [
        'badnets', 'wanet_source', 'ftrojan', 'fiba', 'sig_source', 'bpp_source']
    assert all(cell['cfg']['rho'] == 0.1 for cell in manifest['cells'])
    assert all(cell['cfg']['epochs'] == 50 for cell in manifest['cells'])
    configs = {cell['method_key']: cell['cfg'] for cell in manifest['cells']}
    assert configs['sig_source']['sig_delta_255'] == 40.0
    assert configs['sig_source']['sig_frequency'] == 6
    assert configs['bpp_source']['bpp_levels'] == 8
    assert configs['bpp_source']['bpp_negative_ratio'] == 0.1
    assert configs['bpp_source']['clean_label_cover_ratio'] == 0.1
    assert not list(outdir.rglob('eval_cache.json'))
    assert not list(outdir.rglob('checkpoint.pt'))
