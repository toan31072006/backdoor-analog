"""Defense reporting must distinguish correct-pose recovery from attack failure."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defend.a1_three_model_report import (BACKBONES, METAFI_WINDOWS_PROFILE,
                                          LIGHT_WINDOWS_PROFILES,
                                          _digest, _sha,
                                          build_report, load_verified_record,
                                          markdown_report)
from defend.a1_defense_metrics import enrich_evaluation


def item(backbone, defense, *, clean_target=25., triggered_gt=50., clean_all=100.,
         t_mpjpe=80., pck=60.):
    return dict(backbone=backbone, method='a1', defense=defense, seed=42,
        metrics=dict(clean_mpjpe_mm=clean_all,
                     clean_target_mpjpe_mm=clean_target,
                     triggered_gt_mpjpe_mm=triggered_gt,
                     triggered_gt_minus_clean_target_mm=triggered_gt-clean_target,
                     t_mpjpe_mm=t_mpjpe, target_pck_pct=pck),
        pair_ids_sha256='same-ordered-full-test',
        source_identity_sha256='a1-checkpoint-' + backbone,
        clean_scale_mm=np.array([100., 200.]),
        source_path=f'/audit/{backbone}/{defense}/evaluation.json',
        record_sha256='record-' + backbone + '-' + defense,
        per_sample_sha256='samples-' + backbone + '-' + defense,
        profile='historical-or-current')


def records():
    result = []
    for backbone in BACKBONES:
        result.append(item(backbone, 'none'))
        # Attack score moves in defender's expected direction, but recovery
        # must be judged against true pose and the same two-joint clean floor.
        result.append(item(backbone, 'neural_cleanse', clean_target=27.,
                           triggered_gt=35., clean_all=103.,
                           t_mpjpe=105., pck=20.))
    return result


def test_three_model_a1_report_recovery_uses_same_two_joint_clean_floor():
    report = build_report(records(), required_defenses=('neural_cleanse',))
    assert report['status'] == 'EXPLORATORY_A1_ONLY_THREE_MODEL_DEFENSE'
    assert report['n_seeds'] == 1
    assert report['attack_methods'] == ['a1']
    assert report['excluded_attack_methods'] == ['badnets']
    rows = {(r['backbone'], r['defense']): r for r in report['rows']}
    for backbone in BACKBONES:
        repaired = rows[backbone, 'neural_cleanse']
        assert repaired['clean_mpjpe_mm'] == 103.
        assert repaired['clean_target_mpjpe_mm'] == 27.
        assert repaired['triggered_gt_mpjpe_mm'] == 35.
        assert repaired['triggered_gt_minus_clean_target_mm'] == 8.
        assert repaired['clean_target_mpjpe_mm_delta_vs_none'] == 2.
        assert repaired['triggered_gt_mpjpe_mm_delta_vs_none'] == -15.
        assert repaired['triggered_gt_minus_clean_target_mm_delta_vs_none'] == -17.
        assert repaired['target_pck_pct_delta_vs_none'] == -40.
    table = markdown_report(report)
    assert 'CleanTarget2' in table and 'TriggeredGT2' in table
    assert 'A1-only' in table and 'BadNets excluded' in table
    assert '| hpeli | neural_cleanse | 103.00 | 27.00 | 35.00 | 8.00 |' in table


def test_missing_pair_or_required_repair_cannot_be_called_complete():
    full = records()
    with pytest.raises(ValueError, match='Missing paired NoDef'):
        build_report([r for r in full if (r['backbone'], r['defense']) != ('wiflow', 'none')])
    with pytest.raises(ValueError, match='Missing required'):
        build_report([r for r in full if (r['backbone'], r['defense']) !=
                      ('metafiplusplus', 'neural_cleanse')],
                     required_defenses=('neural_cleanse',))


def test_report_rejects_cross_attack_and_mismatched_test_pairing():
    full = records()
    full[0]['method'] = 'badnets'
    with pytest.raises(ValueError, match='Only seed42 A1'):
        build_report(full)
    full = records()
    full[-1]['pair_ids_sha256'] = 'different-test'
    with pytest.raises(ValueError, match='ordered TEST identities'):
        build_report(full)
    full = records()
    full[-1]['source_identity_sha256'] = 'different-a1-victim'
    with pytest.raises(ValueError, match='same A1 checkpoint'):
        build_report(full)


def test_clean_target_recovery_gap_never_uses_all_joint_clean_error():
    full = records()
    full[-1]['metrics']['triggered_gt_minus_clean_target_mm'] = 35. - 103.
    with pytest.raises(ValueError, match='recovery gap inconsistent'):
        build_report(full)


def test_clean_gt_scale_must_match_between_defenses():
    full = records()
    full[-1]['clean_scale_mm'][0] = 120.
    with pytest.raises(ValueError, match='PCK scales differ'):
        build_report(full)


def test_record_loader_requires_bound_full_test_arrays_and_replay(tmp_path):
    n = 33264
    grid = np.array([0., .2, .4, .6, .8, 1.])
    errors = np.full((n, 6, 2), 10., dtype=np.float64)
    arrays = dict(sample_indices=np.arange(n),
        pair_ids=np.array([(f'csi{i}', f'pose{i}', str(i)) for i in range(n)]),
        clean_error_mm=np.full((n, 17), 10., dtype=np.float64),
        clean_scale_mm=np.full(n, 100., dtype=np.float64),
        doses=grid, target_error_mm=errors.copy(),
        triggered_gt_error_mm=errors.copy(),
        no_trigger_target_error_mm=errors.copy(),
        dce_error_mm=np.zeros((n, 6), dtype=np.float64))
    rows = [dict(d=float(d), t_mpjpe_mm=10., target_pck_pct=100.,
                 triggered_gt_mpjpe_mm=10., dce_mm=0.,
                 no_trigger_tmpjpe_mm=10.) for d in grid]
    diagnostic = dict(samples=n, full_official_test=True,
        target_joints=[2, 3], target_pck_threshold=.2,
        scale='norm(clean_true_pose[5]-clean_true_pose[12])',
        main_dose=1., dose_grid=grid.tolist(), alignment='none',
        selection_on_test=False, clean_pampjpe_mm=8.,
        no_trigger_target_pck_pct=100., no_trigger_t_mpjpe_mm=10.,
        mean_positive_dce_mm=0.)
    measured = enrich_evaluation(dict(metrics=dict(t_mpjpe_mm=10.,
        target_pck_pct=100., clean_mpjpe_mm=10.,
        triggered_gt_mpjpe_mm=10.), diagnostics=diagnostic,
        dose_rows=rows, arrays=arrays))
    npz = tmp_path / 'per_sample.npz'
    np.savez(npz, **arrays)
    record = dict(schema=1, profile='synthetic-test', status='complete',
        seed=42, method='a1', defense='none', metrics=measured['metrics'],
        diagnostics=measured['diagnostics'], dose_rows=measured['dose_rows'],
        binding={'source_artifacts': {'checkpoint.pt': 'synthetic-checkpoint'}},
        artifacts={'per_sample.npz': _sha(npz)})
    record['record_sha256'] = _digest(record)
    path = tmp_path / 'evaluation.json'
    path.write_text(json.dumps(record), encoding='utf-8')
    loaded = load_verified_record('hpeli', path)
    assert loaded['metrics']['clean_target_mpjpe_mm'] == 10.
    assert loaded['metrics']['triggered_gt_minus_clean_target_mm'] == 0.
    legacy = dict(record)
    legacy.pop('record_sha256')
    legacy.pop('defense')
    legacy.pop('binding')
    legacy.update(profile=METAFI_WINDOWS_PROFILE, method='none',
                  source_checkpoint_sha256='a' * 64)
    legacy['record_sha256'] = _digest(legacy)
    path.write_text(json.dumps(legacy), encoding='utf-8')
    assert load_verified_record('metafiplusplus', path)['method'] == 'a1'
    light = dict(legacy)
    light.pop('record_sha256')
    light.update(profile=LIGHT_WINDOWS_PROFILES['wiflow'], method='nc')
    light['record_sha256'] = _digest(light)
    path.write_text(json.dumps(light), encoding='utf-8')
    assert load_verified_record('wiflow', path)['defense'] == 'nc'
    light['status'] = 'nc_diagnostic_only'
    light.pop('record_sha256')
    light['record_sha256'] = _digest(light)
    path.write_text(json.dumps(light), encoding='utf-8')
    with pytest.raises(ValueError, match='diagnostic without repair'):
        load_verified_record('wiflow', path)
    path.write_text(json.dumps(legacy), encoding='utf-8')
    npz.write_bytes(npz.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='Per-sample artifact changed'):
        load_verified_record('metafiplusplus', path)
