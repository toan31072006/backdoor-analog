"""DT-Pose MM-Fi S2 subject split contracts without touching frozen loaders.

The official S2 train and validation subjects are fixed across P1/P2/P3.  This
sidecar is deliberately separate from ``data_utils.feeder`` because completed
PiW runs bind that source file by hash.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils.feeder import (  # noqa: E402
    _MMFI_PROTOCOL_ACTIONS, _mmfi_scene_split, _mmfi_subject_split,
)
from mmfi_s2_split import (  # noqa: E402
    MMFIS2, _MMFI_S2_TRAIN_SUBJECTS, _MMFI_S2_VAL_SUBJECTS,
    _mmfi_cross_subject_split,
)


HELD_OUT = {'S05', 'S10', 'S15', 'S20', 'S25', 'S30', 'S35', 'S40'}
ALL_SUBJECTS = {f'S{i:02d}' for i in range(1, 41)}
def test_s2_uses_official_fixed_subjects():
    train = set(_MMFI_S2_TRAIN_SUBJECTS)
    validation = set(_MMFI_S2_VAL_SUBJECTS)
    assert validation == HELD_OUT
    assert train == ALL_SUBJECTS - HELD_OUT
    assert len(train) == 32 and len(validation) == 8
    assert train.isdisjoint(validation)


@pytest.mark.parametrize('protocol,n_actions', [
    ('protocol1', 14), ('protocol2', 13), ('protocol3', 27), ('all', 27),
])
def test_s2_subject_action_pairs_follow_protocol(protocol, n_actions):
    train, validation = _mmfi_cross_subject_split(protocol)
    actions = set(_MMFI_PROTOCOL_ACTIONS[protocol])
    assert len(train) == 32 * n_actions
    assert len(validation) == 8 * n_actions
    assert not train & validation
    assert {subject for subject, _ in train} == set(_MMFI_S2_TRAIN_SUBJECTS)
    assert {subject for subject, _ in validation} == HELD_OUT
    assert {action for _, action in train} == actions
    assert {action for _, action in validation} == actions
    for action in actions:
        assert sum(a == action for _, a in train) == 32
        assert sum(a == action for _, a in validation) == 8


def test_s2_rejects_unknown_protocol_instead_of_falling_back_to_p3():
    with pytest.raises(ValueError):
        _mmfi_cross_subject_split('random_split')


def test_s2_is_distinct_from_existing_s1_and_s3():
    s2_train, s2_val = _mmfi_cross_subject_split('protocol3')
    s1_train, s1_val = _mmfi_subject_split('protocol3', .8, 0)
    s3_train, s3_val = _mmfi_scene_split('protocol3')
    assert (len(s1_train), len(s1_val)) == (32 * 27, 8 * 27)
    assert (len(s3_train), len(s3_val)) == (30 * 27, 10 * 27)
    assert (s1_train, s1_val) != (s2_train, s2_val)
    assert {subject for subject, _ in s2_train}.isdisjoint(
        {subject for subject, _ in s2_val})
    assert {subject for _, subject, _ in s3_train}.isdisjoint(
        {subject for _, subject, _ in s3_val})


def test_s2_dataset_filters_subjects_and_keeps_frame_number_pairing(tmp_path):
    for subject in ('S01', 'S05'):
        sequence = tmp_path / 'E01' / subject / 'A01'
        wifi = sequence / 'wifi-csi'
        wifi.mkdir(parents=True)
        truth = np.zeros((3, 17, 3), np.float32)
        truth[:, 0, 0] = [11., 22., 33.]
        np.save(sequence / 'ground_truth.npy', truth)
        np.save(wifi / 'frame001_processed.npy',
                np.full((3, 114, 10), .5, np.float32))
        np.save(wifi / 'frame003_processed.npy',
                np.full((3, 114, 10), .5, np.float32))

    train = MMFIS2('training', str(tmp_path), protocol='protocol3')
    validation = MMFIS2('validation', str(tmp_path), protocol='protocol3')
    validation_test_alias = MMFIS2('test', str(tmp_path), protocol='protocol3',
                                    random_ratio=.1, random_seed=999)
    assert [item['name'] for item in train.items] == [
        'E01_S01_A01_f0001', 'E01_S01_A01_f0003']
    assert [item['name'] for item in validation.items] == [
        'E01_S05_A01_f0001', 'E01_S05_A01_f0003']
    assert [item['name'] for item in validation_test_alias.items] == [
        item['name'] for item in validation.items]
    assert [item['frame_idx'] for item in train.items] == [0, 2]
    assert train[1]['pose'][0, 0, 0] == 33.
    assert validation[1]['pose'][0, 0, 0] == 33.
