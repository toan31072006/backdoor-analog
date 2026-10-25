"""TRAIN-only, disjoint clean banks for the MM-Fi NC pose adaptation.

The fixed budget is sealed by the owning defense runner before TEST evaluation.
No source trigger, attacked labels, or official TEST frames enter these loaders.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


INVERSION_FRAMES = 128
HOLDOUT_FRAMES = 128
SEED = 42


def identity(row):
    if len(row) != 3:
        raise ValueError('Expected CSI, pose and frame identity')
    return json.dumps([str(row[0]).replace('\\', '/'),
                       str(row[1]).replace('\\', '/'), int(row[2])],
                      separators=(',', ':'), ensure_ascii=False)


def partition(bank, pair_ids):
    """Use trusted fit frames for inversion/unlearning and distinct selection frames for holdout."""
    fit = bank['fit_indices']
    selection = bank['selection_indices']
    total = len(pair_ids)
    for name, indices in (('fit', fit), ('selection', selection)):
        if (not isinstance(indices, list) or not indices
                or any(isinstance(i, (bool, np.bool_))
                       or not isinstance(i, (int, np.integer))
                       or not 0 <= int(i) < total for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError(f'Invalid trusted TRAIN {name} indices')
    if set(fit).intersection(selection):
        raise ValueError('Trusted fit and holdout selection overlap')
    if {identity(pair_ids[i]) for i in fit}.intersection(
            identity(pair_ids[i]) for i in selection):
        raise ValueError('Trusted fit and holdout frame identities overlap')
    if {pair_ids[i][1] for i in fit}.intersection(
            pair_ids[i][1] for i in selection):
        raise ValueError('Trusted fit and holdout sequence identities overlap')
    if len(fit) <= INVERSION_FRAMES or len(selection) < HOLDOUT_FRAMES:
        raise ValueError('Trusted bank too small for predeclared NC partitions')
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 90417]))
    inversion = sorted(int(i) for i in rng.choice(fit, INVERSION_FRAMES, replace=False))
    holdout = sorted(int(i) for i in rng.choice(selection, HOLDOUT_FRAMES, replace=False))
    unlearning = sorted(set(fit) - set(inversion))
    if set(inversion) & set(holdout) or set(unlearning) & (set(inversion) | set(holdout)):
        raise AssertionError('NC TRAIN partitions overlap')
    result = dict(inversion_indices=inversion, holdout_indices=holdout,
                  unlearning_indices=unlearning)
    for key, values in tuple(result.items()):
        result[key.replace('_indices', '_ids_sha256')] = hashlib.sha256(
            json.dumps([identity(pair_ids[i]) for i in values],
                       separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    return result


class PairBatches:
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for item in self.loader:
            yield item['csi'], item['pose']


def loaders(train, cfg, bank, pair_ids):
    """Return two small materialized banks and a streaming unlearning bank."""
    from torch.utils.data import DataLoader, Subset
    from attack.poison import PoisonedDataset, collate

    sections = partition(bank, pair_ids)
    clean = PoisonedDataset(train, None, mode='clean', pivot=cfg['pivot'],
                            axis=cfg['payload_axis'], dataset='mmfi')

    def make(indices, batch):
        return PairBatches(DataLoader(Subset(clean, indices), batch_size=batch,
                                      shuffle=False, collate_fn=collate,
                                      num_workers=0, pin_memory=False, drop_last=False))

    inv = list(make(sections['inversion_indices'], 8))
    hold = list(make(sections['holdout_indices'], 8))
    unlearn = make(sections['unlearning_indices'], 32)
    ids = {name: [identity(pair_ids[i]) for i in sections[name + '_indices']]
           for name in ('inversion', 'holdout', 'unlearning')}
    return inv, hold, unlearn, ids, sections


def run(model, clean_model, train, cfg, bank, *, device, folder, binding):
    """Invert on A1 and paired clean checkpoints, then unlearn if TRAIN-certified."""
    from run_peak_confirmation import _pair_ids
    from defend.neural_cleanse_pose import run_nc_adapted

    pair_ids = _pair_ids(train)
    inversion, holdout, unlearning, ids, sections = loaders(train, cfg, bank, pair_ids)
    repaired, receipt = run_nc_adapted(
        model, clean_model, inversion, holdout, unlearning,
        inversion_ids=ids['inversion'], holdout_ids=ids['holdout'],
        unlearning_ids=ids['unlearning'], device=device,
        output_dir=folder, source_binding_sha256=binding)
    return repaired, receipt, sections
