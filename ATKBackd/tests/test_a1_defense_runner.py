"""T5 runner provenance/CLI integration; CPU toy fixtures, never full MM-Fi."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_a1_defenses as runner
import run_a1_three_tables as source
from eval_a1_target_pck import _sha


@pytest.fixture
def plans(tmp_path):
    root = tmp_path / 'data' / 'runs' / 'three_tables'
    root.mkdir(parents=True)
    matrix = source.build_matrix(tmp_path / 'data', root, 'cpu', 4, retrain_seed42=True)
    (root / source.MANIFEST).write_text(json.dumps(matrix), encoding='utf-8')
    output = root.parent / 'defenses'
    return matrix, runner.build_plan(matrix, root, output), root, output


def test_only_a1_three_seed_five_defense_plan_preserves_originals(plans):
    matrix, plan, root, output = plans
    before = copy.deepcopy(matrix)
    again = runner.build_plan(matrix, root, output)
    assert len(plan['cells']) == 15
    assert {c['method'] for c in plan['cells']} == {'a1'}
    assert sum(c['defense'] != 'none' for c in plan['cells']) == 12
    assert plan['plan_sha256'] == again['plan_sha256']
    assert matrix == before
    assert plan['recipe']['batch_size'] == 256
    assert plan['contract']['original_victim_training'] is False
    assert plan['contract']['num_workers'] == 0
    assert plan['contract']['target_joints'] == [2, 3]
    assert plan['contract']['main_metrics'] == list(runner.METRICS)
    assert not output.exists()


def test_default_recipe_preserves_pinned_upstream_hyperparameters():
    r = runner.resolve_recipe()
    assert (r['epochs'], r['lr'], r['momentum'], r['weight_decay']) == (100, .01, .9, .0005)
    assert r['sam_radius'] == 2. and r['sam_adaptive'] is True
    assert r['nad_betas'] == [500., 1000., 1000.]
    assert r['nad_teacher_epochs'] == r['nad_student_epochs'] == 10
    assert r['fp_once_prune_ratio'] == .01 and r['utility_tolerance'] == .1


@pytest.mark.parametrize('value', [[], 2, {'unknown': 1}, {'batch_size': False},
                                    {'bank_fraction': .1}, {'selection_fraction': .3}])
def test_recipe_rejects_unknown_or_changed_bank(value):
    with pytest.raises((ValueError, TypeError)):
        runner.resolve_recipe(value)


def test_output_cannot_overlap_any_source_or_data(plans):
    matrix, _, root, _ = plans
    for output in (root, root / 'new', root.parent,
                   Path(matrix['data_home']) / 'datasets' / 'new',
                   Path(matrix['data_home']) / 'actions',
                   Path(matrix['cells'][0]['ckpt_dir']) / 'defense'):
        with pytest.raises(ValueError, match='output'):
            runner.build_plan(matrix, root, output)


def test_saved_manifest_rejects_rehashed_recipe_and_source_mutations(plans):
    matrix, plan, root, output = plans
    output.mkdir()
    path = output / runner.MANIFEST
    changed = copy.deepcopy(plan)
    changed['recipe']['lr'] = .001
    changed['plan_sha256'] = runner._plan_hash(changed)
    path.write_text(json.dumps(changed), encoding='utf-8')
    with pytest.raises(ValueError, match='NEW output'):
        runner._load_plan(root, output)
    path.write_text(json.dumps(plan), encoding='utf-8')
    matrix['cells'][1]['cfg']['theta_max_deg'] = 30.
    matrix['plan_sha256'] = source._plan_hash(matrix)
    (root / source.MANIFEST).write_text(json.dumps(matrix), encoding='utf-8')
    with pytest.raises(ValueError):
        runner._load_plan(root, output)


def test_new_run_rejects_unowned_nonempty_output(plans):
    _, _, root, output = plans
    output.mkdir()
    (output / 'checkpoint.pt').write_bytes(b'not ours')
    with pytest.raises(ValueError, match='not empty'):
        runner._load_plan(root, output)


def test_dry_run_and_status_are_read_only_without_checkpoint_data_or_cuda(plans, monkeypatch, capsys):
    _, _, root, output = plans
    original = (root / source.MANIFEST).read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('dry run must not audit data/checkpoints or initialize CUDA')
    monkeypatch.setattr(runner, '_audit_source', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    args = ['--source-outdir', str(root), '--outdir', str(output)]
    assert runner.main([*args, '--dry-run']) == 0
    assert '5 selected evaluations, 4 repairs' in capsys.readouterr().out
    assert runner.main([*args, '--status']) == 0
    assert 'NOT a fresh scientific audit' in capsys.readouterr().out
    assert not output.exists()
    assert (root / source.MANIFEST).read_bytes() == original


@pytest.mark.parametrize('seeds,methods,defenses', [([42, 42], ['a1'], ['none']),
    ([45], ['a1'], ['ft']), ([42], ['blended'], ['ft']), ([42], ['a1'], ['nc'])])
def test_invalid_runtime_selections_fail_closed(plans, seeds, methods, defenses):
    with pytest.raises(ValueError):
        runner._selected(plans[1], seeds, methods, defenses)


def _record(plan, cell):
    return dict(cell_id=cell['cell_id'], seed=cell['seed'], method=cell['method'],
        defense=cell['defense'], control=False,
        metrics=dict(t_mpjpe_mm=100. + cell['seed'] - 42,
                     target_pck_pct=50., clean_mpjpe_mm=170., triggered_gt_mpjpe_mm=120.),
        dose_rows=[dict(d=d, t_mpjpe_mm=100., target_pck_pct=50.,
                       triggered_gt_mpjpe_mm=120., dce_mm=20., no_trigger_tmpjpe_mm=400.)
                   for d in (0., .2, .4, .6, .8, 1.)])


def _write_record(plan, cell):
    folder = Path(cell['folder'])
    folder.mkdir(parents=True)
    sample = io.BytesIO()
    np.savez_compressed(sample, values=np.array([1., 2.]))
    (folder / 'per_sample.npz').write_bytes(sample.getvalue())
    record = dict(_record(plan, cell), schema=1, status='complete',
        binding={'cell_id': cell['cell_id']}, per_sample_file='per_sample.npz',
        per_sample_sha256=_sha(folder / 'per_sample.npz'))
    record['binding_sha256'] = runner._digest(record['binding'])
    if cell['defense'] != 'none':
        (folder / 'repair.pt').write_bytes(b'bound toy checkpoint')
        record.update(repair_file='repair.pt', repair_sha256=_sha(folder / 'repair.pt'))
    record['record_sha256'] = runner._digest(record)
    (folder / 'evaluation.json').write_text(json.dumps(record), encoding='utf-8')
    return record


def test_completed_record_hashes_binding_samples_and_repaired_weights(plans):
    plan = plans[1]
    cell = next(c for c in plan['cells'] if c['defense'] == 'fp')
    record = _write_record(plan, cell)
    assert runner._read_record(cell['folder'], record['binding']) == record
    with pytest.raises(ValueError, match='scientific binding'):
        runner._read_record(cell['folder'], {'cell_id': 'wrong'})
    (Path(cell['folder']) / 'repair.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        runner._read_record(cell['folder'])


@pytest.mark.parametrize('fault', ['metric', 'npz', 'nan', 'grid'])
def test_corrupt_or_invalid_record_rejected(plans, fault):
    plan = plans[1]
    cell = plan['cells'][0]
    record = _write_record(plan, cell)
    folder = Path(cell['folder'])
    if fault == 'npz':
        (folder / 'per_sample.npz').write_bytes(b'corrupt')
    else:
        record['metrics']['target_pck_pct'] = 101. if fault == 'metric' else 50.
        if fault == 'nan':
            record['metrics']['clean_mpjpe_mm'] = None
        if fault == 'grid':
            record['dose_rows'] = record['dose_rows'][:1]
        record.pop('record_sha256')
        record['record_sha256'] = runner._digest(record)
        (folder / 'evaluation.json').write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError):
        runner._read_record(folder)


def test_partial_seed_never_claims_complete_or_frame_based_sd(plans):
    _, plan, _, output = plans
    output.mkdir()
    records = [_record(plan, c) for c in plan['cells'] if c['seed'] == 42]
    status = runner.export_records(plan, records)
    report = json.loads((output / 'defense_summary.json').read_text(encoding='utf-8'))
    assert status.startswith('T5_PARTIAL')
    assert report['complete_records'] == 5 and report['expected_records'] == 15
    assert all(r['n_seeds'] == 1 and not r['complete'] for r in report['aggregates'])
    assert all(r['t_mpjpe_mm_sd'] is None for r in report['aggregates'])
    assert '42 | 100.00' in (output / 'table5_defenses.md').read_text(encoding='utf-8')


def test_full_three_seed_report_uses_sample_sd_of_seeds(plans):
    _, plan, _, output = plans
    output.mkdir()
    records = [_record(plan, c) for c in plan['cells']]
    assert runner.export_records(plan, records).startswith('T5_COMPLETE_THREE_SEEDS')
    report = json.loads((output / 'defense_summary.json').read_text(encoding='utf-8'))
    assert all(r['n_seeds'] == 3 and r['complete'] for r in report['aggregates'])
    assert all(r['t_mpjpe_mm_mean'] == 101. and r['t_mpjpe_mm_sd'] == 1.
               for r in report['aggregates'])
    with pytest.raises(ValueError, match='Duplicate'):
        runner.export_records(plan, [*records, records[0]])


def test_worker_zero_exit_without_record_is_not_success(plans, monkeypatch):
    _, plan, _, _ = plans
    class ExitZero:
        returncode = 0
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: ExitZero())
    with pytest.raises(RuntimeError, match='without a complete'):
        runner._run_sequential(plan, None, [plan['cells'][0]], 'cpu', 'unused.json')


@pytest.mark.parametrize('defense', ['ft', 'fp', 'nad', 'ft_sam'])
def test_real_clean_loader_to_repair_to_six_dose_eval_pipeline(tmp_path, defense):
    # Reuse small canonical pose/trigger fixtures, not benchmark files.
    from test_a1_defense_eval import TinyMMFi, NumericTrigger, config
    from run_peak_confirmation import _pair_ids
    from defend import a1_defense_eval as evaluation
    from defend import a1_repair_ops as ops
    from torch import nn
    class ToyPose(nn.Module):
        def __init__(self):
            super().__init__()
            self.skunit1 = nn.Sequential(nn.Conv2d(1, 2, 1), nn.ReLU())
            self.skunit2 = nn.Sequential(nn.Conv2d(2, 2, 1), nn.ReLU())
            self.regression = nn.Sequential(*[nn.Identity() for _ in range(5)],
                nn.ReLU(), nn.Flatten(), nn.Linear(2 * 17 * 3, 51))
        def forward(self, x):
            feature = self.skunit2(self.skunit1(x.unsqueeze(1)))
            return self.regression(feature).reshape(len(x), 1, 17, 3), feature
    base = TinyMMFi(count=200, groups=10)
    cfg = config()
    bank = evaluation.select_clean_bank(_pair_ids(base), 42)
    fit, selection = evaluation.clean_loaders(base, bank, cfg, 42, batch_size=4)
    recipe = dict(epochs=1, nad_teacher_epochs=1, nad_student_epochs=1)
    model, metadata = ops.train_repair(ToyPose(), fit, selection, defense, recipe,
        'cpu', tmp_path / 'repair.pt', 'a' * 64, 42)
    assert metadata['complete']
    result = evaluation.evaluate_defense(model, TinyMMFi(), NumericTrigger(), cfg,
                                         'cpu', expected_count=8)
    assert set(result['metrics']) == set(runner.METRICS)
    assert all(np.isfinite(value) for value in result['metrics'].values())
    assert result['arrays']['target_error_mm'].shape == (8, 6, 2)
