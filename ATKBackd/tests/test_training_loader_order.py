"""Common multi-dose loader policy aligns parent RNG across worker counts."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_backdoor import _resolve_training_config, _training_loader


class OrderedPairs(Dataset):
    """Top-level class so Windows DataLoader workers can pickle it."""
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return dict(csi=np.array([index, index / 12], dtype=np.float32),
                    pose=np.array([index / 24], dtype=np.float32))


def _run(workers, resume=None):
    torch.manual_seed(314)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.0001, momentum=.9)
    loader = _training_loader(OrderedPairs(), dict(batch_size=4, num_workers=workers,
                              loader_persistent_workers=False))
    first_epoch = 0
    if resume is not None:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['rng'])
        first_epoch = 1
    orders, first = [], None
    for epoch in range(first_epoch, 3):
        ids = []
        for batch in loader:
            ids.extend(batch['csi'][:, 0].tolist())
            optimizer.zero_grad()
            loss = (model(batch['csi']) - batch['pose']).square().mean()
            loss.backward()
            optimizer.step()
        orders.append(ids)
        if epoch == 0:
            first = dict(model=copy.deepcopy(model.state_dict()),
                         optimizer=copy.deepcopy(optimizer.state_dict()),
                         rng=torch.get_rng_state().clone())
    return orders, model.state_dict(), torch.get_rng_state(), first


def test_nonpersistent_worker_counts_match_shuffle_weights_and_epoch_resume():
    zero, many = _run(0), _run(2)
    assert zero[0] == many[0]
    for key in zero[1]:
        torch.testing.assert_close(zero[1][key], many[1][key], rtol=0, atol=0)
    torch.testing.assert_close(zero[2], many[2], rtol=0, atol=0)
    resumed = _run(0, resume=many[3])
    assert resumed[0] == many[0][1:]
    for key in many[1]:
        torch.testing.assert_close(resumed[1][key], many[1][key], rtol=0, atol=0)
    torch.testing.assert_close(resumed[2], many[2], rtol=0, atol=0)


def test_legacy_loader_keeps_its_persistent_worker_default():
    assert _training_loader(OrderedPairs(), dict(batch_size=4, num_workers=2)).persistent_workers
    assert not _training_loader(OrderedPairs(), dict(batch_size=4, num_workers=0)).persistent_workers


@pytest.mark.parametrize('value', [1, 0, None, 'False'])
def test_persistence_flag_is_boolean_and_scientifically_bound(value):
    cfg = dict(loader_persistent_workers=value)
    with pytest.raises(ValueError, match='must be a boolean'):
        _resolve_training_config(cfg)
    with pytest.raises(ValueError, match='must be a boolean'):
        _training_loader(OrderedPairs(), dict(cfg, batch_size=4, num_workers=0))
