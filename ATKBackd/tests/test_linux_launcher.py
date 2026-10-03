"""Regression tests for the Linux full-run launcher arguments."""

from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which('bash')
CAN_RUN_BASH = BASH is not None and os.name != 'nt'


def _run_common(*args):
    command = (
        'source "$1"; shift; parse_common_args "$@"; finalize_common; '
        'printf "%s|%s" "${SEEDS[*]}" "$WORKER_TIMEOUT"'
    )
    return subprocess.run(
        [BASH, '-c', command, 'launcher-test',
         str(REPO_ROOT / 'remote_linux' / 'common.sh'), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.skipif(not CAN_RUN_BASH, reason='a native bash is required for launcher tests')
def test_common_launcher_seed_and_timeout_defaults():
    result = _run_common()

    assert result.returncode == 0, result.stderr
    assert result.stdout == '42 0 1|0'


@pytest.mark.skipif(not CAN_RUN_BASH, reason='a native bash is required for launcher tests')
def test_common_launcher_accepts_seed_list_and_timeout():
    result = _run_common('--seeds', '42', '7', '--worker-timeout', '86400')

    assert result.returncode == 0, result.stderr
    assert result.stdout == '42 7|86400'


@pytest.mark.skipif(not CAN_RUN_BASH, reason='a native bash is required for launcher tests')
def test_common_launcher_rejects_duplicate_seeds():
    result = _run_common('--seeds', '42', '42')

    assert result.returncode != 0
    assert 'duplicate value: 42' in result.stderr


@pytest.mark.skipif(not CAN_RUN_BASH, reason='a native bash is required for launcher tests')
def test_common_launcher_rejects_equivalent_zero_padded_seeds():
    result = _run_common('--seeds', '0', '00')

    assert result.returncode != 0
    assert 'duplicate value: 0' in result.stderr


@pytest.mark.skipif(not CAN_RUN_BASH, reason='a native bash is required for launcher tests')
def test_main_launcher_forwards_seed_subset_and_custom_timeout(tmp_path):
    mmfi_root = tmp_path / 'mmfi'
    mmfi_root.mkdir()
    action_npy = tmp_path / 'data_bend.npy'
    action_npy.touch()
    argv_log = tmp_path / 'argv.log'
    fake_python = tmp_path / 'fake-python'
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf \'%q \' "$@" >> "$ARGV_LOG"\n'
        'printf \'\\n\' >> "$ARGV_LOG"\n',
        encoding='utf-8',
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env['ARGV_LOG'] = str(argv_log)
    result = subprocess.run(
        [BASH, str(REPO_ROOT / 'remote_linux' / '03_run_main.sh'),
         '--dataset', 'mmfi', '--python', str(fake_python),
         '--mmfi-root', str(mmfi_root), '--action-npy', str(action_npy),
         '--runs-root', str(tmp_path / 'runs'), '--seeds', '42', '7',
         '--worker-timeout', '86400'],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = [shlex.split(line) for line in argv_log.read_text().splitlines()]
    assert len(calls) == 2  # attacked conditions, then paired clean control
    for call in calls:
        seed_index = call.index('--seeds')
        assert call[seed_index + 1:seed_index + 3] == ['42', '7']
        assert call[seed_index + 3] == '--dataset-root'
        timeout_index = call.index('--worker-timeout')
        assert call[timeout_index + 1] == '86400'
