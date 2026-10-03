"""Fail-closed checks for the paper experiment runner."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run_experiments import (  # noqa: E402
    _assert_complete_matrix,
    _wait_for_workers,
    main as runner_main,
)


class _FakeProcess:
    def __init__(self, alive_checks, exitcode=0):
        self._alive_checks = iter(alive_checks)
        self.exitcode = exitcode
        self.join_calls = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def join(self, timeout=None):
        self.join_calls.append(timeout)

    def is_alive(self):
        return next(self._alive_checks)

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1


def _row(model='hpeli', scenario='bend', trigger='micro_dropper', seed=42):
    return {
        'model': model,
        'scenario': scenario,
        'trigger': trigger,
        'seed': seed,
    }


def test_complete_matrix_is_accepted():
    cells = [('hpeli', 'bend', 'micro_dropper', 42),
             ('hpeli', 'bend', 'micro_dropper', 0)]
    _assert_complete_matrix(cells, [_row(), _row(seed=0)])


@pytest.mark.parametrize(
    ('rows', 'failed'),
    [([_row()], []),
     ([_row(), _row()], []),
     ([_row(), _row(seed=0)], [1])],
)
def test_missing_duplicate_or_failed_worker_is_rejected(rows, failed):
    cells = [('hpeli', 'bend', 'micro_dropper', 42),
             ('hpeli', 'bend', 'micro_dropper', 0)]
    with pytest.raises(RuntimeError, match='refusing to write partial'):
        _assert_complete_matrix(cells, rows, failed)


def test_parallel_worker_has_no_implicit_timeout_or_termination():
    process = _FakeProcess([False])

    assert _wait_for_workers([(0, process)]) == []
    assert process.join_calls == [None]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0


def test_explicit_worker_timeout_terminates_and_marks_worker_failed():
    process = _FakeProcess([True, False])

    assert _wait_for_workers([(3, process)], worker_timeout=12) == [3]
    assert process.join_calls == [12, 30]
    assert process.terminate_calls == 1
    assert process.kill_calls == 0


def test_explicit_worker_timeout_escalates_if_terminate_does_not_stop_it():
    process = _FakeProcess([True, True])

    assert _wait_for_workers([(4, process)], worker_timeout=5) == [4]
    assert process.join_calls == [5, 30, 30]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1


@pytest.mark.parametrize('seeds', [('0', '0'), ('0', '00')])
def test_cli_rejects_duplicate_seed_cells_before_training(monkeypatch, seeds):
    monkeypatch.setattr(sys, 'argv', ['run_experiments.py', '--seeds', *seeds])

    with pytest.raises(SystemExit) as exc:
        runner_main()

    assert exc.value.code == 2


def test_cli_rejects_negative_seed_before_training(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_experiments.py', '--seeds', '-1'])

    with pytest.raises(SystemExit) as exc:
        runner_main()

    assert exc.value.code == 2
