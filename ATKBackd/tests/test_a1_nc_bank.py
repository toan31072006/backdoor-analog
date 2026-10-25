"""NC bank selection must not leak holdout/TEST into inversion or unlearning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defend.a1_nc_bank import partition, identity


def test_disjoint_fixed_train_nc_partitions_and_replay():
    ids = [[f'csi/{i}.npy', f'pose/{i//10}.npy', i % 10] for i in range(400)]
    bank = dict(fit_indices=list(range(220)), selection_indices=list(range(220, 380)))
    first = partition(bank, ids)
    assert partition(bank, ids) == first
    inversion = set(first['inversion_indices'])
    holdout = set(first['holdout_indices'])
    unlearning = set(first['unlearning_indices'])
    assert (len(inversion), len(holdout), len(unlearning)) == (128, 128, 92)
    assert not inversion & holdout and not inversion & unlearning and not holdout & unlearning
    assert inversion | unlearning == set(bank['fit_indices'])
    assert holdout <= set(bank['selection_indices'])
    assert not (inversion | holdout | unlearning) & set(range(380, 400))
    expected = hashlib.sha256(json.dumps([identity(ids[i]) for i in sorted(inversion)],
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    assert first['inversion_ids_sha256'] == expected


def test_nc_partition_rejects_tampered_fit_selection_overlap():
    ids = [[f'csi/{i}.npy', f'pose/{i//10}.npy', i % 10] for i in range(400)]
    bank = dict(fit_indices=list(range(220)), selection_indices=list(range(220, 380)))
    bank['selection_indices'][0] = 219
    bank['selection_indices'].sort()
    with pytest.raises(ValueError, match='overlap'):
        partition(bank, ids)
