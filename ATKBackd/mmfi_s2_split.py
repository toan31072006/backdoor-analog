"""MM-Fi P1/P2/P3 S2 split as a sidecar to the frozen dataset feeder.

DT-Pose's ``cross_subject_split`` is a fixed subject partition, shared by
every action in the selected protocol.  The source is the DT-Pose repository
at commit 2fbbef271454e6fbb95b1a0163b8cdbb96163838, specifically
``config/mmfi/pose_config.yaml`` and ``feeder/mmfi.py``:
https://github.com/cseeyangchen/DT-Pose/tree/2fbbef271454e6fbb95b1a0163b8cdbb96163838

This module does not alter ``data_utils.feeder``.  Existing PiW runs bind that
file by hash.  ``MMFIS2`` retains its frame-number CSI/pose pairing and CSI
normalization while selecting the official S2 subject-action keys.
"""

from data_utils.feeder import MMFI, _MMFI_ALL_SUBJECTS, _MMFI_PROTOCOL_ACTIONS


_MMFI_S2_VAL_SUBJECTS = tuple(f'S{i:02d}' for i in range(5, 41, 5))
_MMFI_S2_TRAIN_SUBJECTS = tuple(
    subject for subject in _MMFI_ALL_SUBJECTS
    if subject not in _MMFI_S2_VAL_SUBJECTS
)


def _mmfi_cross_subject_split(protocol):
    """Return official S2 (train, validation) ``(subject, action)`` sets.

    ``protocol1`` selects 14 daily actions, ``protocol2`` selects 13
    rehabilitation actions, and ``protocol3``/``all`` select all 27 actions.
    Unlike S1, S2 does not redraw subjects for each action or use an RNG seed.
    ``random_split`` is a legacy feeder option, not a valid DT-Pose protocol.
    """
    if protocol not in _MMFI_PROTOCOL_ACTIONS:
        raise ValueError(
            f'S2 needs a DT-Pose action protocol; got {protocol!r}. '
            f'Choose one of {sorted(_MMFI_PROTOCOL_ACTIONS)}.'
        )
    actions = _MMFI_PROTOCOL_ACTIONS[protocol]
    train = {(subject, action)
             for subject in _MMFI_S2_TRAIN_SUBJECTS for action in actions}
    validation = {(subject, action)
                  for subject in _MMFI_S2_VAL_SUBJECTS for action in actions}
    return train, validation


class MMFIS2(MMFI):
    """MMFI loader with the official fixed S2 split and original frame pairing.

    ``split`` is ``training``, ``validation`` or ``test`` (test is the held-out
    validation split, as in the existing feeder).  The base feeder is called
    with a ratio of 1 solely to enumerate every valid frame once.  Its legacy
    random split never determines membership: this subclass filters each item
    by the official, fixed subject-action sets.  This also retains the base
    loader's filename-derived frame index so missing CSI frames cannot shift
    pose labels.  ``random_ratio``/``random_seed`` are accepted for interface
    compatibility but do not affect S2.
    """

    def __init__(self, split, data_root, num_person=1,
                 protocol='protocol3', setting='s2',
                 random_ratio=0.8, random_seed=0):
        if split not in ('training', 'validation', 'test'):
            raise ValueError(f'unsupported MMFi S2 split {split!r}')
        if setting != 's2':
            raise ValueError(f'MMFIS2 requires setting="s2", got {setting!r}')
        train, validation = _mmfi_cross_subject_split(protocol)
        keep = train if split == 'training' else validation

        # ratio=1 includes all sequences.  Reuse the base feeder's frame
        # enumeration and CSI/pose methods; its shuffled indices are a full
        # permutation and do not affect the returned, directory-sorted order.
        super().__init__(
            split='training', data_root=data_root, num_person=num_person,
            protocol='random_split', setting='s1', random_ratio=1.0,
            random_seed=0,
        )
        self.items = [item for item in self.items
                      if tuple(item['name'].split('_')[1:3]) in keep]
        self.split = split
        self.protocol = protocol
        self.setting = setting
        if not self.items:
            raise RuntimeError(
                f'MMFI S2: no items for {split}/{protocol} under {self.data_root}'
            )


__all__ = ['MMFIS2', '_mmfi_cross_subject_split',
           '_MMFI_S2_TRAIN_SUBJECTS', '_MMFI_S2_VAL_SUBJECTS']
