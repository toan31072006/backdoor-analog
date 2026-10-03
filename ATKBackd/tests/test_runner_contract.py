"""Fail-closed checks for the paper experiment runner."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run_experiments import _assert_complete_matrix  # noqa: E402


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
