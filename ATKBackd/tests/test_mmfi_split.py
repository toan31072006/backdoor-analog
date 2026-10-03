"""Regression tests for the MMFi protocol/setting split.

These lock down three things that were each wrong at some point:

* setting='s2' used to fall through to the s1 branch, silently running a random
  split while the config and checkpoint claimed cross-subject.
* _mmfi_scene_split built a Cartesian product of scenes x subjects, reporting
  1260 training keys where the real count is 420 — each MMFi subject appears in
  exactly one environment.
* The S1 split must reproduce DT-Pose's per-action reseed exactly (448/112),
  or every MMFi number becomes incomparable with the upstream table.

Run:  python tests/test_mmfi_split.py     (no pytest needed)
      pytest tests/test_mmfi_split.py     (also works)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.feeder import (                                   # noqa: E402
    _build_mmfi_items, _mmfi_scene_split, _mmfi_subject_split,
    _MMFI_PROTOCOL_ACTIONS, _MMFI_SUBJECT_SCENE,
    _MMFI_S3_TRAIN_SUBJECTS, _MMFI_S3_VAL_SUBJECTS)


# ── 1. an unimplemented setting must be refused, not silently downgraded ────
def test_unsupported_setting_is_refused():
    for bad in ('s2', 'S3', 'cross_subject', 'cross_subject_split', '', None):
        try:
            _build_mmfi_items('/nonexistent', 'training', setting=bad)
        except ValueError as e:
            assert 's2' in str(e) or 'setting' in str(e), str(e)
            continue
        except (FileNotFoundError, NotADirectoryError, RuntimeError):
            raise AssertionError(
                f'setting={bad!r} reached the dataset instead of being refused; '
                f'it would run the s1 split under another name')
        raise AssertionError(f'setting={bad!r} should have been refused')


def test_supported_settings_reach_the_dataset():
    """s1/s3 must pass validation (and then fail on the bogus path, not before)."""
    for ok in ('s1', 's3'):
        try:
            _build_mmfi_items('/nonexistent', 'training', setting=ok)
        except (FileNotFoundError, NotADirectoryError, RuntimeError):
            continue                      # validation passed, data path did not
        except ValueError as e:
            raise AssertionError(f'setting={ok!r} must be accepted, got: {e}')


# ── 2. S3 counts subjects once, not once per environment ───────────────────
def test_s3_counts_are_not_a_cartesian_product():
    n_act = len(_MMFI_PROTOCOL_ACTIONS['protocol1'])
    train, val = _mmfi_scene_split('protocol1')
    assert len(train) == len(_MMFI_S3_TRAIN_SUBJECTS) * n_act, (
        f'expected {len(_MMFI_S3_TRAIN_SUBJECTS) * n_act} training keys '
        f'(30 subjects x {n_act} actions), got {len(train)} — a scenes x '
        f'subjects product would give {3 * len(_MMFI_S3_TRAIN_SUBJECTS) * n_act}')
    assert len(val) == len(_MMFI_S3_VAL_SUBJECTS) * n_act, len(val)


def test_s3_uses_each_subject_own_environment():
    train, val = _mmfi_scene_split('protocol1')
    for scene, subject, _ in train | val:
        assert scene == _MMFI_SUBJECT_SCENE[subject], (
            f'{subject} paired with {scene}, but it is recorded in '
            f'{_MMFI_SUBJECT_SCENE[subject]}')


# ── 3. S3 is genuinely cross-environment AND cross-subject ─────────────────
def test_s3_has_no_environment_or_subject_overlap():
    train, val = _mmfi_scene_split('protocol1')
    env_tr = {e for e, _, _ in train}
    env_va = {e for e, _, _ in val}
    sub_tr = {s for _, s, _ in train}
    sub_va = {s for _, s, _ in val}
    assert env_tr == {'E01', 'E02', 'E03'}, sorted(env_tr)
    assert env_va == {'E04'}, sorted(env_va)
    assert not (env_tr & env_va), 'S3 must not share an environment'
    assert not (sub_tr & sub_va), 'S3 must not share a subject'


# ── 4. S1 still reproduces DT-Pose exactly (448/112, per-action reseed) ────
def test_s1_matches_dtpose_reference():
    train, val = _mmfi_subject_split('protocol1', 0.8, 0)
    assert (len(train), len(val)) == (448, 112), (len(train), len(val))

    subjects = [f'S{i:02d}' for i in range(1, 41)]
    actions = _MMFI_PROTOCOL_ACTIONS['protocol1']
    rs, ref_tr, ref_va = 0, set(), set()
    for action in actions:                       # DT-Pose's legacy global RNG
        np.random.seed(rs)
        idx = np.random.permutation(len(subjects))
        n = int(np.floor(0.8 * len(subjects)))
        for s in np.array(subjects)[idx[:n]]:
            ref_tr.add((str(s), action))
        for s in np.array(subjects)[idx[n:]]:
            ref_va.add((str(s), action))
        rs += 1
    assert train == ref_tr and val == ref_va, 'S1 no longer matches DT-Pose'


def test_s1_shares_subjects_but_s3_does_not():
    """The two settings must actually differ in the way that matters."""
    tr1, va1 = _mmfi_subject_split('protocol1', 0.8, 0)
    assert {s for s, _ in tr1} & {s for s, _ in va1}, \
        'S1 is a random split: subjects are expected on both sides'
    tr3, va3 = _mmfi_scene_split('protocol1')
    assert not ({s for _, s, _ in tr3} & {s for _, s, _ in va3})


# ── 5. protocol names, including the paper's P3 spelling ───────────────────
def test_protocol_names_and_sizes():
    assert len(_MMFI_PROTOCOL_ACTIONS['protocol1']) == 14
    assert len(_MMFI_PROTOCOL_ACTIONS['protocol2']) == 13
    assert len(_MMFI_PROTOCOL_ACTIONS['all']) == 27
    assert _MMFI_PROTOCOL_ACTIONS['protocol3'] == _MMFI_PROTOCOL_ACTIONS['all'], \
        'the paper calls the 27-action set P3; both spellings must work'
    assert not (set(_MMFI_PROTOCOL_ACTIONS['protocol1'])
                & set(_MMFI_PROTOCOL_ACTIONS['protocol2'])), \
        'P1 and P2 must be disjoint action sets'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e}')
        except Exception as e:                    # noqa: BLE001
            failed += 1
            print(f'ERROR {t.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
