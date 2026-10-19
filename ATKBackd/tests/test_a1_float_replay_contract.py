"""Full-count synthetic NPZ roundtrips must tolerate only float reduction noise.

No MM-Fi data, model loading, training, or GPU is used. The nonconstant arrays
and their actual memory layout reproduce the evaluator's per-joint stacking,
unlike a constant/broadcast fixture whose reductions are accidentally exact.
"""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_a1_defenses as legacy
import run_a1_defenses_v2 as runner
import run_a1_clp as clp_runner
from defend.a1_defense_metrics import GRID, METRICS, enrich_evaluation
from run_method_drafts import _sha_json


@pytest.fixture(scope='module')
def nonconstant_roundtrip():
    n = 33264
    rng = np.random.default_rng(20261020)
    start = np.asfortranarray(rng.uniform(100., 300., (n, 2)))
    clean = rng.uniform(100., 300., (n, 17))
    clean[:, [2, 3]] = start
    assert start.dtype == np.float64 and start.flags.f_contiguous
    assert np.unique(start).size == start.size
    # Each source is Fortran ordered, like advanced target-joint indexing.
    # Stacking them at axis 1 produces an array that is neither C nor F ordered.
    target = np.stack([start * (1. + .2 * d) for d in GRID], axis=1)
    gt = np.stack([start * (1. - .1 * d) for d in GRID], axis=1)
    no_trigger = np.stack([start * (1. - d) + start * .9 * d for d in GRID], axis=1)
    assert not no_trigger.flags.c_contiguous
    assert not no_trigger.flags.f_contiguous
    scale = rng.uniform(100., 1800., n)
    dce = rng.uniform(10., 150., (n, 1)) * np.array(GRID)[None, :]
    pairs = np.array([['S01', 'A01', str(i)] for i in range(n)])
    assert len({tuple(pair) for pair in pairs.tolist()}) == n
    pck = lambda error: float(100. * np.mean(error <= .2 * scale[:, None]))
    rows = [dict(d=d, t_mpjpe_mm=float(target[:, i].mean()),
                 target_pck_pct=pck(target[:, i]),
                 triggered_gt_mpjpe_mm=float(gt[:, i].mean()),
                 dce_mm=float(dce[:, i].mean()),
                 no_trigger_tmpjpe_mm=float(no_trigger[:, i].mean()))
            for i, d in enumerate(GRID)]
    evaluated = dict(
        metrics=dict(t_mpjpe_mm=float(target[:, -1].mean()),
                     target_pck_pct=pck(target[:, -1]),
                     clean_mpjpe_mm=float(clean.mean()),
                     triggered_gt_mpjpe_mm=float(gt[:, -1].mean())),
        diagnostics=dict(samples=n, full_official_test=True,
                         target_joints=[2, 3], target_pck_threshold=.2,
                         main_dose=1., dose_grid=list(GRID), alignment='none',
                         scale='norm(clean_true_pose[5]-clean_true_pose[12])',
                         selection_on_test=False, clean_pampjpe_mm=123.456789,
                         no_trigger_t_mpjpe_mm=float(no_trigger[:, -1].mean()),
                         no_trigger_target_pck_pct=pck(no_trigger[:, -1]),
                         mean_positive_dce_mm=float(dce[:, 1:].mean())),
        dose_rows=rows,
        arrays=dict(sample_indices=np.arange(n), pair_ids=pairs,
                    clean_error_mm=clean, clean_scale_mm=scale,
                    doses=np.array(GRID), target_error_mm=target,
                    triggered_gt_error_mm=gt,
                    no_trigger_target_error_mm=no_trigger, dce_error_mm=dce))
    # This is the production ordering: derive metrics before saving the NPZ.
    enriched = enrich_evaluation(evaluated)
    saved = io.BytesIO()
    np.savez_compressed(saved, **enriched['arrays'])
    with np.load(io.BytesIO(saved.getvalue()), allow_pickle=False) as data:
        restored = {name: data[name] for name in data.files}
    replay = enrich_evaluation(dict(enriched, arrays=restored))
    assert restored['no_trigger_target_error_mm'].flags.c_contiguous
    for name, values in enriched['arrays'].items():
        np.testing.assert_array_equal(restored[name], values)
    changed = {name: abs(enriched['metrics'][name] - replay['metrics'][name])
               for name in METRICS
               if enriched['metrics'][name] != replay['metrics'][name]}
    assert changed, 'Fixture must expose an actual reduction-order discrepancy'
    assert max(changed.values()) < 1.e-10
    identity = dict(n=n, pair_ids_sha256=_sha_json(
        [[subject, action, int(frame)] for subject, action, frame in pairs.tolist()]))
    return enriched, saved.getvalue(), identity, changed


@pytest.fixture
def saved_record(nonconstant_roundtrip, tmp_path):
    enriched, sample_bytes, identity, changed = nonconstant_roundtrip
    folder = tmp_path / 'seed43' / 'a1' / 'none'
    folder.mkdir(parents=True)
    (folder / 'per_sample.npz').write_bytes(sample_bytes)
    binding = dict(cell_id='s43__a1__none', inputs=dict(eval=copy.deepcopy(identity)))
    values = copy.deepcopy({key: value for key, value in enriched.items() if key != 'arrays'})
    record = dict(schema=1, status='complete', profile=runner.PROFILE,
                  cell_id=binding['cell_id'], seed=43, method='a1', defense='none',
                  control=False, binding=binding, binding_sha256=legacy._digest(binding),
                  per_sample_file='per_sample.npz',
                  per_sample_sha256=legacy._sha(folder / 'per_sample.npz'), **values)
    record = runner._write_result(folder, record)
    return folder, record, changed


def test_real_noncontiguous_roundtrip_is_accepted_without_rewriting_record(saved_record, monkeypatch):
    folder, record, changed = saved_record
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    assert changed and record['metrics']['triggered_gt_minus_clean_target_mm'] < 0
    # Reinstall just the old exact-comparison rule: this same valid record
    # must fail before the fix, without checking out or altering source files.
    def old_exact_comparison(saved, replayed):
        if saved != replayed:
            raise ValueError('Old exact float replay comparison failed')
    with monkeypatch.context() as old:
        old.setattr(runner, '_assert_metric_replay', old_exact_comparison)
        with pytest.raises(ValueError, match='Old exact float replay'):
            runner._read_record(folder, record['binding'])
    assert runner._read_record(folder, record['binding']) == record
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


@pytest.mark.parametrize('metric', [
    'clean_target_mpjpe_mm', 'no_trigger_t_mpjpe_mm',
    'mean_positive_dce_mm', 'triggered_gt_minus_clean_target_mm'])
def test_rehashed_derived_error_of_one_micro_mm_is_not_roundoff(saved_record, metric):
    folder, record, _ = saved_record
    record['metrics'][metric] += 1.e-6
    runner._write_result(folder, record)
    with pytest.raises(ValueError):
        runner._read_record(folder)


def test_rehashed_pck_error_above_numerical_tolerance_is_rejected(saved_record):
    folder, record, _ = saved_record
    record['metrics']['no_trigger_target_pck_pct'] += 1.e-7
    runner._write_result(folder, record)
    with pytest.raises(ValueError):
        runner._read_record(folder)


@pytest.mark.parametrize('bad', [True, False, None, '123', float('nan'), float('inf')])
def test_rehashed_invalid_derived_scalar_is_rejected(saved_record, bad):
    folder, record, _ = saved_record
    record['metrics']['clean_target_mpjpe_mm'] = bad
    with pytest.raises(ValueError):
        # Canonical JSON rejects NaN/Inf even before a record can be rehashed;
        # finite invalid types reach the real authenticated record reader.
        runner._write_result(folder, record)
        runner._read_record(folder)


@pytest.mark.parametrize('fault', [
    'missing_metric', 'extra_metric', 'schema', 'record_hash', 'npz_hash',
    'array_shape', 'array_nan', 'pair_identity', 'sample_order'])
def test_numeric_tolerance_does_not_weaken_authentication_or_array_schema(saved_record, fault):
    folder, record, _ = saved_record
    if fault == 'missing_metric':
        record['metrics'].pop('clean_target_mpjpe_mm')
    elif fault == 'extra_metric':
        record['metrics']['unapproved_metric'] = 1.
    elif fault == 'schema':
        record['schema'] = 999
    elif fault == 'record_hash':
        record['metrics']['clean_target_mpjpe_mm'] += 1.
        (folder / 'evaluation.json').write_text(json.dumps(record), encoding='utf-8')
    elif fault == 'npz_hash':
        # ZIP tolerates trailing bytes, but the cryptographic byte hash must not.
        with (folder / 'per_sample.npz').open('ab') as stream:
            stream.write(b'changed artifact bytes')
    else:
        with np.load(folder / 'per_sample.npz', allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        if fault == 'array_shape':
            arrays['dce_error_mm'] = arrays['dce_error_mm'][:, :, None]
        elif fault == 'array_nan':
            arrays['no_trigger_target_error_mm'][0, -1, 0] = np.nan
        elif fault == 'pair_identity':
            arrays['pair_ids'][0, 2] = '1'
        else:
            arrays['sample_indices'][[0, 1]] = arrays['sample_indices'][[1, 0]]
        np.savez_compressed(folder / 'per_sample.npz', **arrays)
        record['per_sample_sha256'] = legacy._sha(folder / 'per_sample.npz')
    if fault not in ('record_hash', 'npz_hash'):
        runner._write_result(folder, record)
    with pytest.raises(ValueError):
        runner._read_record(folder)


@pytest.fixture
def saved_clp_record(saved_record):
    source_folder, record, changed = saved_record
    record = copy.deepcopy(record)
    folder = source_folder.parent / 'clp'
    folder.mkdir()
    (folder / 'per_sample.npz').write_bytes((source_folder / 'per_sample.npz').read_bytes())
    (folder / 'repair.pt').write_bytes(b'authenticated synthetic CLP repair checkpoint')
    binding = dict(record['binding'], cell_id='s43__a1__clp',
                   data_free_repair=True, recipe=copy.deepcopy(clp_runner.RECIPE))
    metadata = dict(
        complete=True, optimizer_steps=0, schema='a1_clp_weight_only_v1',
        method='clp', u=3., uses_data=False, uses_labels=False,
        uses_test_selection=False, paired_bn2d_count=10, total_bn2d_count=12,
        paired_channels=960, unsupported_bn1d_count=2, training_steps=0,
        fine_tuning_epochs=0, training_samples=0, selection_samples=0,
        post_pruning_finetuning=False, strength_candidates=[3.],
        std_correction=1, variance_epsilon_added=False,
        skipped_bn2d=[{'name': name} for name in clp_runner.RECIPE['allow_unpaired_bn2d']])
    record.update(profile=clp_runner.PROFILE, cell_id=binding['cell_id'], defense='clp',
                  binding=binding, binding_sha256=legacy._digest(binding),
                  repair_file='repair.pt', repair_sha256=legacy._sha(folder / 'repair.pt'),
                  repair=metadata)
    record = runner._write_result(folder, record)
    return folder, record, changed


def test_clp_noncontiguous_roundtrip_is_accepted_without_rewriting_artifacts(saved_clp_record, monkeypatch):
    folder, record, changed = saved_clp_record
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    assert changed and record['metrics']['triggered_gt_minus_clean_target_mm'] < 0
    def old_exact_comparison(saved, replayed):
        if saved != replayed:
            raise ValueError('Old exact CLP replay comparison failed')
    with monkeypatch.context() as old:
        old.setattr(runner, '_assert_metric_replay', old_exact_comparison)
        with pytest.raises(ValueError, match='Old exact CLP replay'):
            clp_runner._read_record(folder, record['binding'])
    assert clp_runner._read_record(folder, record['binding']) == record
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


def test_clp_rehashed_derived_tamper_is_not_treated_as_roundoff(saved_clp_record):
    folder, record, _ = saved_clp_record
    record['metrics']['no_trigger_t_mpjpe_mm'] += 1.e-6
    runner._write_result(folder, record)
    with pytest.raises(ValueError):
        clp_runner._read_record(folder)
