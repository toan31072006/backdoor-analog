"""Deterministic metadata-only subsets for the MMFi method-screening draft.

The default screening evaluation is a held-out part of the official training
split.  Selection uses only the parent length and fixed RNG seed, never CSI,
pose values, evaluation results, or the attack/model variant.
"""

import hashlib
import json
import operator
import os
from numbers import Integral

import numpy as np


DRAFT_PROFILE = 'method_screening_v1'
DRAFT_PROFILES = (DRAFT_PROFILE, 'method_peak_control_v1')
_DRAFT_OPTIONS = (
    'draft_train_samples', 'draft_eval_samples', 'draft_subset_seed',
    'draft_eval_source',
)


def _positive_integer(cfg, key):
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{key} must be a positive integer for a method draft')
    return int(value)


def _canonical_split(split):
    if split in ('train', 'training'):
        return 'train'
    if split in ('test', 'validation', 'val', 'eval'):
        return 'test'
    raise ValueError(f'unsupported method-draft split: {split!r}')


def _json_sha256(value):
    encoded = json.dumps(
        value, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _relative_identifier(value, data_root):
    identifier = (str(os.fspath(value)) if isinstance(value, os.PathLike)
                  else str(value))
    if data_root is not None:
        try:
            identifier = os.path.relpath(identifier, os.fspath(data_root))
        except (TypeError, ValueError):
            # A synthetic identifier or a path on another drive remains an ID;
            # resolving it must never involve opening a data file.
            pass
    return identifier.replace('\\', '/')


class DraftSubset:
    """A small dataset view preserving the parent loader and sample contract.

    ``items`` is a new list of shallow-copied metadata dictionaries.  File
    loaders and cache controls delegate to the parent.  Sampling/audit metadata
    is available out of band and is never inserted into returned samples.
    """

    def __init__(self, base, indices, *, split, requested_cap,
                 selection_seed=0, selection_stream=0,
                 eval_source='training_holdout', reserved_holdout_n=0,
                 profile=DRAFT_PROFILE):
        if profile not in DRAFT_PROFILES:
            raise ValueError(f'unsupported draft_profile: {profile!r}')
        parent_n = len(base)
        if len(base.items) != parent_n:
            raise ValueError('draft parent items and dataset length disagree')
        selected = tuple(operator.index(index) for index in indices)
        if any(index < 0 or index >= parent_n for index in selected):
            raise IndexError('draft subset contains an out-of-bounds parent index')
        if len(set(selected)) != len(selected):
            raise ValueError('draft subset indices must be unique')
        self._base = base
        self.subset_indices = selected
        self.items = [dict(base.items[index]) for index in selected]
        self.split = _canonical_split(split)
        self.parent_n = parent_n
        self.requested_cap = int(requested_cap)
        self.selection_seed = int(selection_seed)
        self.selection_stream = int(selection_stream)
        self.draft_eval_source = eval_source
        self.reserved_holdout_n = int(reserved_holdout_n)
        self.draft_profile = profile

    def __getattr__(self, name):
        # Pickle asks for special methods before restoring __dict__.  Forwarding
        # those, or accessing self._base recursively, can corrupt worker spawn.
        if name.startswith('__'):
            raise AttributeError(name)
        base = object.__getattribute__(self, '__dict__').get('_base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __len__(self):
        return len(self.subset_indices)

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError('draft subset index out of range')
        return self._base[self.subset_indices[index]]

    def draft_pair_ids(self):
        """Return ordered file/frame identifiers, with no CSI or pose content.

        The compact JSON encoding of this list is exactly the input to the
        manifest's ``identifiers_sha256``.  This lets a distortion report prove
        that it used the same held-out frames as method evaluation.
        """
        data_root = getattr(self._base, 'data_root', None)
        return [
            [_relative_identifier(item['csi'], data_root),
             _relative_identifier(item['kpt'], data_root),
             int(item['frame_idx'])]
            for item in self.items
        ]

    def draft_subset_manifest(self):
        """Return a JSON-safe audit record without sample arrays or labels."""
        indices = list(self.subset_indices)
        return {
            'schema': 1,
            'profile': self.draft_profile,
            'split': self.split,
            'eval_source': self.draft_eval_source,
            'parent_n': self.parent_n,
            'n': len(self),
            'requested_cap': self.requested_cap,
            'selection_seed': self.selection_seed,
            'selection_stream': self.selection_stream,
            'reserved_holdout_n': self.reserved_holdout_n,
            'subset_indices': indices,
            'index_sha256': _json_sha256(indices),
            'identifiers_sha256': _json_sha256(self.draft_pair_ids()),
        }


def apply_draft_subset(base, cfg, split):
    """Apply explicitly marked MMFi screening budgets, otherwise return base.

    ``training_holdout`` expects both views to receive the same official
    training parent from the dataset factory.  One seeded permutation reserves
    at most one fifth for evaluation; training draws from the remainder.
    ``official_test`` is an explicit alternative using independent split RNG
    streams.  Both modes preserve parent order within each selected view.
    """
    marker = cfg.get('method_draft', False)
    if not isinstance(marker, bool):
        raise ValueError('method_draft must be a boolean')
    profile = cfg.get('draft_profile')
    if profile is not None and profile not in DRAFT_PROFILES:
        raise ValueError(f'unsupported draft_profile: {profile!r}')
    enabled = marker or profile in DRAFT_PROFILES
    if not enabled:
        if any(key in cfg for key in _DRAFT_OPTIONS):
            raise ValueError('draft subset options require an explicit method-draft marker')
        return base
    if cfg.get('experiment_name', 'one-person') != 'mmfi':
        raise ValueError('method-draft subsets are supported only for MMFi')

    train_cap = _positive_integer(cfg, 'draft_train_samples')
    eval_cap = _positive_integer(cfg, 'draft_eval_samples')
    seed = cfg.get('draft_subset_seed', 0)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed != 0:
        raise ValueError(f'{DRAFT_PROFILE} requires draft_subset_seed=0')
    eval_source = cfg.get('draft_eval_source', 'training_holdout')
    if eval_source not in ('training_holdout', 'official_test'):
        raise ValueError('draft_eval_source must be training_holdout or official_test')

    split = _canonical_split(split)
    parent_n = len(base)
    cap = train_cap if split == 'train' else eval_cap
    stream = 0
    holdout_n = 0
    if eval_source == 'training_holdout':
        parent_split = getattr(base, 'split', None)
        if parent_split is not None and parent_split not in ('train', 'training'):
            raise ValueError('training_holdout requires an official training parent dataset')
        if parent_n < 2:
            raise ValueError('training-holdout method draft requires at least two samples')
        holdout_n = min(eval_cap, max(1, parent_n // 5))
        permutation = np.random.default_rng(
            np.random.SeedSequence([int(seed), stream])).permutation(parent_n)
        if split == 'train':
            selected = permutation[holdout_n:holdout_n + min(train_cap, parent_n - holdout_n)]
        else:
            selected = permutation[:holdout_n]
    else:
        stream = 0 if split == 'train' else 1
        selected = np.random.default_rng(
            np.random.SeedSequence([int(seed), stream])).choice(
                parent_n, size=min(cap, parent_n), replace=False)
    return DraftSubset(
        base, sorted(map(int, selected)), split=split, requested_cap=cap,
        selection_seed=int(seed), selection_stream=stream,
        eval_source=eval_source, reserved_holdout_n=holdout_n,
        profile=profile if profile is not None else DRAFT_PROFILE,
    )
