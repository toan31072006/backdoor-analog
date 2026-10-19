"""Versioned T5 orchestration and audited reuse; CPU fixtures, never MM-Fi jobs."""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_a1_defenses as legacy
import run_a1_defenses_v2 as runner
import run_a1_three_tables as source
from defend.a1_defense_metrics import enrich_evaluation
from run_method_drafts import _sha_json
from test_a1_defense_metrics import evaluated_fixture, report_records


@pytest.fixture(scope='module')
def canonical_plan(tmp_path_factory):
    # Validate the large immutable scientific matrix once. Each test gets a
    # distinct output and fresh code hashes; production validation is unmocked.
    root = tmp_path_factory.mktemp('v2_scientific_matrix') / 'data' / 'runs' / 'three_tables'
    root.mkdir(parents=True)
    matrix = source.build_matrix(root.parent.parent, root, 'cpu', 4, retrain_seed42=True)
    (root / source.MANIFEST).write_text(json.dumps(matrix), encoding='utf-8')
    output = root.parent / 'defenses_v2'
    return matrix, runner.build_plan(matrix, root, output), root, output


@pytest.fixture
def plans(canonical_plan, tmp_path):
    matrix, canonical, root, _ = canonical_plan
    output = root.parent / tmp_path.name
    plan = copy.deepcopy(canonical)
    plan.update(outdir=str(output.resolve()), sources=runner._sources())
    for cell in plan['cells']:
        cell['folder'] = str(output / f"seed{cell['seed']}" / 'a1' / cell['defense'])
    plan['plan_sha256'] = legacy._plan_hash(plan)
    return copy.deepcopy(matrix), plan, root, output


def _args(root, output, *extra):
    return ['--source-outdir', str(root), '--outdir', str(output), *extra]


def test_plan_is_fifteen_a1_only_records_preserving_original_recipe(plans):
    matrix, plan, root, output = plans
    before = copy.deepcopy(matrix)
    again = runner.build_plan(matrix, root, output)
    assert len(plan['cells']) == 15
    assert {c['method'] for c in plan['cells']} == {'a1'}
    assert {c['defense'] for c in plan['cells']} == {'none', 'ft', 'anp', 'nad', 'ft_sam'}
    assert sum(c['defense'] != 'none' for c in plan['cells']) == 12
    assert plan['plan_sha256'] == again['plan_sha256']
    assert plan['schema'] == 2 and plan['profile'] == runner.PROFILE
    assert matrix == before and not output.exists()
    assert plan['recipe'] == legacy.resolve_recipe()
    assert plan['recipe']['batch_size'] == plan['anp_recipe']['batch_size'] == 256
    assert plan['contract']['num_workers'] == 0
    assert plan['contract']['original_victim_training'] is False
    assert plan['contract']['expected_records'] == 15
    assert plan['contract']['main_metrics'] == list(runner.MAIN_METRICS)
    assert plan['contract']['reporting_metrics'] == list(runner.METRICS)
    assert 'after viewing pilot' in plan['contract']['historical_evaluation']
    assert runner.MANIFEST != legacy.MANIFEST
    assert set(legacy._sources()).issubset(plan['sources'])
    assert 'defend/a1_anp_sources.json' in plan['sources']


def test_anp_and_clean_bank_batch_sizes_must_match(plans):
    matrix, _, root, output = plans
    with pytest.raises(ValueError, match='batch size'):
        runner.build_plan(matrix, root, output, anp_recipe={'batch_size': 128})


def test_saved_plan_resume_rejects_rehashed_recipe_or_source_change(plans, monkeypatch):
    matrix, plan, root, output = plans
    output.mkdir()
    path = output / runner.MANIFEST
    path.write_text(json.dumps(plan), encoding='utf-8')
    before = path.read_bytes()
    assert runner._load_plan(root, output) == (plan, matrix)
    assert path.read_bytes() == before
    changed = copy.deepcopy(plan)
    changed['anp_recipe']['anp_eps'] = .3
    changed['plan_sha256'] = legacy._plan_hash(changed)
    path.write_text(json.dumps(changed), encoding='utf-8')
    assert changed['plan_sha256'] != plan['plan_sha256']
    with pytest.raises(ValueError, match='Frozen V2 scalar defaults changed'):
        runner._load_plan(root, output)
    path.write_text(json.dumps(plan), encoding='utf-8')
    monkeypatch.setattr(runner, '_sources', lambda: {'different': 'source'})
    with pytest.raises(ValueError, match='NEW output'):
        runner._load_plan(root, output)


def test_new_output_must_be_empty_and_cannot_overlap_source(plans):
    matrix, _, root, output = plans
    with pytest.raises(ValueError, match='output'):
        runner.build_plan(matrix, root, root / 'new')
    output.mkdir()
    (output / 'repair.pt').write_bytes(b'unowned')
    with pytest.raises(ValueError, match='not empty'):
        runner._load_plan(root, output)


def test_dry_run_and_presence_status_never_touch_data_gpu_locks_or_processes(plans, monkeypatch, capsys):
    _, _, root, output = plans
    original = (root / source.MANIFEST).read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail('read-only mode touched data/checkpoint/CUDA/locks/processes')
    for name in ('_audit_source', '_bank', '_cell_lock', '_atomic_bytes', '_atomic_json'):
        monkeypatch.setattr(legacy, name, forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(runner.subprocess, 'Popen', forbidden)
    assert runner.main(_args(root, output, '--dry-run')) == 0
    assert '15/15 evaluations' in capsys.readouterr().out
    assert runner.main(_args(root, output, '--status')) == 0
    assert 'NOT a fresh scientific audit' in capsys.readouterr().out
    assert not output.exists() and (root / source.MANIFEST).read_bytes() == original


@pytest.mark.parametrize('seeds,defenses', [([42, 42], ['ft']), ([45], ['none']),
    ([42], ['fp']), ([42], ['ft', 'ft']), ([], ['anp']), ([42], [])])
def test_runtime_selection_rejects_invalid_duplicate_or_removed_method(plans, seeds, defenses):
    with pytest.raises(ValueError):
        runner._selected(plans[1], seeds, defenses)


@pytest.mark.parametrize('extra', [('--jobs-per-gpu', '0'), ('--jobs-per-gpu', '3'),
    ('--device', 'cuda:01'), ('--device', 'cuda:0,cuda:1')])
def test_cli_rejects_unsupported_slots_and_multiple_devices(extra):
    with pytest.raises(SystemExit):
        runner.main(['--source-outdir', 's', '--outdir', 'o', *extra])


def _full_fixture():
    evaluated = evaluated_fixture()
    for name, array in list(evaluated['arrays'].items()):
        if name == 'doses':
            continue
        evaluated['arrays'][name] = np.tile(array, (16632,) + (1,) * (array.ndim - 1))
    evaluated['arrays']['sample_indices'] = np.arange(33264)
    evaluated['arrays']['pair_ids'] = np.array([['S01', 'A01', str(i)] for i in range(33264)])
    evaluated['diagnostics'].update(samples=33264, full_official_test=True)
    identity = {'n': 33264, 'pair_ids_sha256': _sha_json(
        [[csi, pose, int(frame)] for csi, pose, frame in evaluated['arrays']['pair_ids'].tolist()])}
    return evaluated, identity


def _write_record(folder, cell, binding, *, version2=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    evaluated, identity = _full_fixture()
    binding = copy.deepcopy(binding)
    binding.setdefault('inputs', {}).setdefault('eval', identity)
    if version2:
        evaluated = enrich_evaluation(evaluated)
    sample = io.BytesIO()
    np.savez_compressed(sample, **evaluated.pop('arrays'))
    (folder / 'per_sample.npz').write_bytes(sample.getvalue())
    record = dict(schema=1, status='complete', cell_id=cell['cell_id'], seed=cell['seed'],
        method='a1', defense=cell['defense'], control=False, binding=binding,
        binding_sha256=legacy._digest(binding), per_sample_file='per_sample.npz',
        per_sample_sha256=legacy._sha(folder / 'per_sample.npz'), **evaluated)
    if cell['defense'] != 'none':
        (folder / 'repair.pt').write_bytes(b'authenticated original repair checkpoint')
        record.update(repair_file='repair.pt', repair_sha256=legacy._sha(folder / 'repair.pt'))
    if version2:
        record['profile'] = runner.PROFILE
    record['record_sha256'] = legacy._digest(record)
    (folder / 'evaluation.json').write_text(json.dumps(record), encoding='utf-8')
    return record


def test_record_reader_replays_all_metrics_and_rejects_rehashed_false_scalar(plans):
    _, plan, _, _ = plans
    cell = plan['cells'][0]
    record = _write_record(cell['folder'], cell, {'cell_id': cell['cell_id']}, version2=True)
    assert runner._read_record(cell['folder'], record['binding']) == record
    assert record['metrics']['clean_target_mpjpe_mm'] == 25.
    assert record['metrics']['triggered_gt_minus_clean_target_mm'] == -4.
    with pytest.raises(ValueError, match='scientific binding'):
        runner._read_record(cell['folder'], {'wrong': 'binding'})
    record['metrics']['clean_target_mpjpe_mm'] += 5.
    runner._write_result(cell['folder'], record)
    with pytest.raises(ValueError, match='replay'):
        runner._read_record(cell['folder'])


@pytest.mark.parametrize('fault', ['missing_metric', 'wrong_profile', 'npz', 'repair'])
def test_version2_record_corruption_is_rejected(plans, fault):
    _, plan, _, _ = plans
    cell = next(c for c in plan['cells'] if c['defense'] == 'ft')
    record = _write_record(cell['folder'], cell, {'cell_id': cell['cell_id']}, version2=True)
    folder = Path(cell['folder'])
    if fault == 'npz':
        (folder / 'per_sample.npz').write_bytes(b'broken')
    elif fault == 'repair':
        (folder / 'repair.pt').write_bytes(b'changed')
    else:
        if fault == 'missing_metric':
            record['metrics'].pop('clean_target_mpjpe_mm')
        else:
            record['profile'] = legacy.PROFILE
        runner._write_result(folder, record)
    with pytest.raises(ValueError):
        runner._read_record(folder)


@pytest.mark.parametrize('field,value', [('seed', 43), ('method', 'badnets'),
    ('defense', 'none'), ('cell_id', 's44__a1__none')])
def test_rehashed_record_identity_must_match_its_scientific_binding(plans, field, value):
    _, plan, _, _ = plans
    cell = next(c for c in plan['cells'] if c['defense'] == 'ft') if field == 'defense' else plan['cells'][0]
    record = _write_record(cell['folder'], cell, {'cell_id': cell['cell_id']}, version2=True)
    record[field] = value
    runner._write_result(cell['folder'], record)
    with pytest.raises(ValueError):
        runner._read_record(cell['folder'])


@pytest.fixture
def reuse_case(plans, monkeypatch, request):
    matrix, _, root, output = plans
    oldroot = root.parent / (output.name + '_v1')
    oldroot.mkdir()
    oldplan = legacy.build_plan(matrix, root, oldroot)
    (oldroot / legacy.MANIFEST).write_text(json.dumps(oldplan), encoding='utf-8')
    plan = runner.build_plan(matrix, root, output, reuse_outdir=oldroot)
    defense = getattr(request, 'param', 'ft')
    cell = next(c for c in plan['cells'] if c['seed'] == 42 and c['defense'] == defense)
    oldcell = next(c for c in oldplan['cells'] if c['cell_id'] == cell['cell_id'])
    bank = {'fit': [1, 3], 'selection': [2], 'seed': 42}
    (oldroot / 'seed42').mkdir()
    (oldroot / 'seed42' / 'clean_bank.json').write_text(json.dumps(bank), encoding='utf-8')
    original = legacy._source_cell(matrix, cell)
    run = dict(folder=original['ckpt_dir'], cfg=original['cfg'], fingerprint='f' * 64)
    _, identity = _full_fixture()
    artifacts, inputs = {'checkpoint': 'a' * 64}, {'train': {'identity': 'T'}, 'eval': identity}
    oldbinding = legacy._binding(oldplan, oldcell, original, run, artifacts, inputs, bank)
    newbinding = legacy._binding(plan, cell, original, run, artifacts, inputs, bank)
    oldrecord = _write_record(oldcell['folder'], oldcell, oldbinding)
    monkeypatch.setattr(runner, '_context', lambda *a, **k:
        (original, run, artifacts, inputs, {'passed': True}, bank, newbinding))
    return dict(matrix=matrix, plan=plan, cell=cell, oldplan=oldplan, oldcell=oldcell,
        oldrecord=oldrecord, bank=bank, newbinding=newbinding)


def test_reuse_copies_authenticated_artifacts_without_mutating_or_evaluating_v1(reuse_case, monkeypatch):
    case = reuse_case
    oldfolder = Path(case['oldcell']['folder'])
    before = {p.name: p.read_bytes() for p in oldfolder.iterdir()}
    import defend.a1_defense_eval as evaluation
    monkeypatch.setattr(evaluation, 'evaluate_defense', lambda *a, **kw: pytest.fail('unexpected inference'))
    record = runner._reuse_record(case['plan'], case['matrix'], case['cell'])
    folder = Path(case['cell']['folder'])
    assert record['profile'] == runner.PROFILE
    assert record['binding'] == case['newbinding']
    assert record['reuse_provenance']['original_binding'] == case['oldrecord']['binding']
    assert record['reuse_provenance']['source_plan_sha256'] == case['oldplan']['plan_sha256']
    assert record['metrics']['clean_target_mpjpe_mm'] == 25.
    assert record['metrics']['triggered_gt_minus_clean_target_mm'] == -4.
    assert (folder / 'per_sample.npz').read_bytes() == before['per_sample.npz']
    assert (folder / 'repair.pt').read_bytes() == before['repair.pt']
    assert (folder / 'reused_v1_evaluation.json').read_bytes() == before['evaluation.json']
    assert {p.name: p.read_bytes() for p in oldfolder.iterdir()} == before
    assert runner._read_record(folder, case['newbinding']) == record
    (folder / 'reused_v1_evaluation.json').write_bytes(b'tampered old provenance')
    with pytest.raises(ValueError, match='V1 record changed'):
        runner._read_record(folder)


@pytest.mark.parametrize('fault', ['binding', 'bank', 'source_plan', 'new_checkpoint'])
def test_reuse_rejects_incompatible_source_bank_or_existing_new_repair(reuse_case, fault):
    c = reuse_case
    oldfolder = Path(c['oldcell']['folder'])
    if fault == 'binding':
        old = copy.deepcopy(c['oldrecord'])
        old['binding']['inputs']['eval'] = {'identity': 'different'}
        old['binding_sha256'] = legacy._digest(old['binding'])
        runner._write_result(oldfolder, old)
    elif fault == 'bank':
        bankfile = Path(c['oldplan']['outdir']) / 'seed42' / 'clean_bank.json'
        bankfile.write_text(json.dumps({'changed': True}), encoding='utf-8')
    elif fault == 'source_plan':
        oldplan = copy.deepcopy(c['oldplan'])
        oldplan['plan_sha256'] = '0' * 64
        (Path(oldplan['outdir']) / legacy.MANIFEST).write_text(json.dumps(oldplan), encoding='utf-8')
    else:
        folder = Path(c['cell']['folder'])
        folder.mkdir(parents=True)
        (folder / 'repair.pt').write_bytes(b'new in-progress repair must survive')
    with pytest.raises(ValueError):
        runner._reuse_record(c['plan'], c['matrix'], c['cell'])
    if fault == 'new_checkpoint':
        assert (Path(c['cell']['folder']) / 'repair.pt').read_bytes() == b'new in-progress repair must survive'


def test_interrupted_reuse_with_identical_copied_checkpoint_is_idempotent(reuse_case):
    c = reuse_case
    oldfolder, folder = Path(c['oldcell']['folder']), Path(c['cell']['folder'])
    folder.mkdir(parents=True)
    (folder / 'repair.pt').write_bytes((oldfolder / 'repair.pt').read_bytes())
    record = runner._reuse_record(c['plan'], c['matrix'], c['cell'])
    assert runner._read_record(folder, c['newbinding']) == record


def test_anp_is_never_imported_from_old_non_anp_plan(reuse_case):
    c = reuse_case
    cell = next(r for r in c['plan']['cells'] if r['defense'] == 'anp')
    assert runner._reuse_record(c['plan'], c['matrix'], cell) is None
    assert not Path(cell['folder']).exists()


@pytest.mark.parametrize('reuse_case', ['none'], indirect=True)
def test_reused_no_defense_requires_no_repaired_checkpoint(reuse_case):
    c = reuse_case
    record = runner._reuse_record(c['plan'], c['matrix'], c['cell'])
    folder = Path(c['cell']['folder'])
    assert record['defense'] == 'none'
    assert 'repair_file' not in record and 'repair_sha256' not in record
    assert not (folder / 'repair.pt').exists()
    assert runner._read_record(folder, c['newbinding']) == record


@pytest.mark.parametrize('fault', ['order', 'pair_ids', 'partial', 'declared_full',
                                  'frame_decimal', 'frame_alpha', 'frame_leading_zero'])
def test_rehashed_npz_still_requires_full_ordered_test_identity(plans, fault):
    _, plan, _, _ = plans
    cell = plan['cells'][0]
    record = _write_record(cell['folder'], cell, {'cell_id': cell['cell_id']}, version2=True)
    folder = Path(cell['folder'])
    with np.load(folder / 'per_sample.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if fault == 'order':
        arrays['sample_indices'] = arrays['sample_indices'][::-1]
    elif fault == 'pair_ids':
        arrays['pair_ids'][0, 0] = 'other'
    elif fault == 'partial':
        record['binding']['inputs']['eval']['n'] = 2
        record['binding_sha256'] = legacy._digest(record['binding'])
    elif fault.startswith('frame_'):
        arrays['pair_ids'][1, 2] = {'frame_decimal': '1.0', 'frame_alpha': 'abc',
                                   'frame_leading_zero': '01'}[fault]
    else:
        record['diagnostics']['full_official_test'] = False
    content = io.BytesIO()
    np.savez_compressed(content, **arrays)
    (folder / 'per_sample.npz').write_bytes(content.getvalue())
    record['per_sample_sha256'] = legacy._sha(folder / 'per_sample.npz')
    runner._write_result(folder, record)
    with pytest.raises(ValueError):
        runner._read_record(folder)


def test_complete_export_contains_paired_seed_sd_and_six_dose_csvs(plans):
    _, plan, _, output = plans
    output.mkdir()
    records = report_records(runner.DEFENSES)
    for record in records:
        record.update(cell_id=f"s{record['seed']}__a1__{record['defense']}", record_sha256='a' * 64)
    assert 'COMPLETE_THREE_SEEDS' in runner.export_records(plan, records)
    summary = json.loads((output / 'defense_summary.json').read_text(encoding='utf-8'))
    assert summary['complete_records'] == summary['expected_records'] == 15
    assert len(summary['six_dose']) == 90 and len(summary['paired_deltas']) == 12
    anp = next(a for a in summary['paired_aggregates'] if a['defense'] == 'anp')
    assert anp['t_mpjpe_mm_delta_sd'] == pytest.approx(np.std([10., -20., 30.], ddof=1))
    assert anp['target_pck_delta_pp_mean'] == -20.
    assert anp['target_pck_delta_pp_sd'] == 10.
    for name, count in [('per_seed', 15), ('six_dose', 90), ('paired_deltas', 12),
                        ('aggregates', 5), ('paired_aggregates', 4)]:
        with (output / f'defense_{name}.csv').open(encoding='utf-8', newline='') as handle:
            assert len(list(csv.DictReader(handle))) == count
    text = (output / 'table5_defenses.md').read_text(encoding='utf-8')
    assert 'FT/FP pilot results remain historical' in text
    assert '| ANP (HPE-adapted) | 42/43/44 |' in text


def test_partial_export_never_advertises_three_seed_sd(plans):
    _, plan, _, output = plans
    output.mkdir()
    records = [r for r in report_records(runner.DEFENSES) if r['seed'] == 42]
    for record in records:
        record.update(cell_id=f"s42__a1__{record['defense']}", record_sha256='a' * 64)
    assert 'PARTIAL' in runner.export_records(plan, records)
    summary = json.loads((output / 'defense_summary.json').read_text(encoding='utf-8'))
    assert summary['complete_records'] == 5
    assert all(r['t_mpjpe_mm_sd'] is None for r in summary['aggregates'])
    assert all(r['t_mpjpe_mm_delta_sd'] is None for r in summary['paired_aggregates'])


def test_scheduler_two_slots_preserves_logs_and_skips_complete(plans, monkeypatch):
    matrix, plan, _, _ = plans
    cells = runner._selected(plan, [42], list(runner.DEFENSES))
    completed = {cells[0]['folder']}
    calls, active, peak = [], set(), [0]
    for cell in cells:
        folder = Path(cell['folder'])
        folder.mkdir(parents=True)
        (folder / 'console.log').write_text('old checkpoint log\n', encoding='utf-8')
    class Child:
        def __init__(self, cell):
            self.cell, self.polls, self.pid = cell, 0, 100 + len(calls)
            active.add(cell['folder'])
            peak[0] = max(peak[0], len(active))
        def poll(self):
            self.polls += 1
            if self.polls == 1:
                return None
            active.discard(self.cell['folder'])
            completed.add(self.cell['folder'])
            return 0
    def popen(command, **kwargs):
        cell = next(c for c in cells if c['cell_id'] == command[command.index('--_cell') + 1])
        calls.append((command, kwargs))
        kwargs['stdout'].write('resumed worker\n')
        kwargs['stdout'].flush()
        return Child(cell)
    monkeypatch.setattr(runner, '_read_record', lambda folder, *a:
        {'verified': True} if str(folder) in completed else None)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    runner._schedule(plan, matrix, cells, 'cuda:0', 2)
    assert peak[0] == 2 and len(calls) == 4
    assert completed == {c['folder'] for c in cells}
    for command, kwargs in calls:
        assert command[:3] == [sys.executable, '-u', str(Path(runner.__file__).resolve())]
        assert command[-2:] == ['--device', 'cuda:0']
        assert kwargs['stdout'].closed
        assert runner.MANIFEST in command[command.index('--_manifest') + 1]
    assert 'old checkpoint log\nresumed worker' in (
        Path(cells[1]['folder']) / 'console.log').read_text(encoding='utf-8')


@pytest.mark.parametrize('jobs', [0, 3, False, 1.0])
def test_direct_scheduler_invalid_slot_count_fails_before_spawning(plans, monkeypatch, jobs):
    matrix, plan, _, _ = plans
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('invalid scheduling launched worker'))
    with pytest.raises(ValueError):
        runner._schedule(plan, matrix, [], 'cpu', jobs)


def test_scheduler_duplicate_cells_rejected_before_spawn(plans, monkeypatch):
    matrix, plan, _, _ = plans
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('duplicate scheduled'))
    with pytest.raises(ValueError):
        runner._schedule(plan, matrix, [plan['cells'][0]] * 2, 'cpu', 2)


def test_main_has_one_owner_prepares_bank_then_audits_reuses_and_exports(plans, monkeypatch):
    matrix, plan, root, output = plans
    output.mkdir()
    manifest = output / runner.MANIFEST
    manifest.write_text(json.dumps(plan), encoding='utf-8')
    before = manifest.read_bytes()
    events = []
    monkeypatch.setattr(runner.runtime_v1, '_prepare_banks', lambda *a: events.append('bank'))
    monkeypatch.setattr(runner, '_collect', lambda *a: events.append('audit') or [])
    monkeypatch.setattr(runner, '_reuse_record', lambda *a: events.append('reuse'))
    def schedule(*args):
        assert events == ['bank', 'audit', *(['reuse'] * 5)]
        assert len(args[2]) == 5 and args[3:] == ('cpu', 2)
        events.append('schedule')
    monkeypatch.setattr(runner, '_schedule', schedule)
    monkeypatch.setattr(runner, 'export_records', lambda *a: events.append('export') or 'T5_V2_PARTIAL')
    with legacy._cell_lock(output):
        with pytest.raises(RuntimeError, match='Another process'):
            runner.main(_args(root, output, '--seeds', '42', '--device', 'cpu'))
    assert not events
    assert runner.main(_args(root, output, '--seeds', '42', '--device', 'cpu')) == 0
    assert events == ['bank', 'audit', *(['reuse'] * 5), 'schedule', 'audit', 'export']
    assert manifest.read_bytes() == before
    launches = list((output / 'runtime_launches').glob('*.json'))
    assert len(launches) == 1
    runtime = json.loads(launches[0].read_text(encoding='utf-8'))
    assert runtime['jobs_per_gpu'] == 2 and len(runtime['cells']) == 5


def test_export_only_audits_completed_records_without_repair_or_bank_creation(plans, monkeypatch):
    _, plan, root, output = plans
    events = []
    def forbidden(*a, **kw):
        pytest.fail('export-only must not fit/evaluate models or spawn workers')
    monkeypatch.setattr(runner.runtime_v1, '_prepare_banks', forbidden)
    monkeypatch.setattr(runner, '_schedule', forbidden)
    monkeypatch.setattr(runner, '_reuse_record', forbidden)
    monkeypatch.setattr(runner, '_collect', lambda *a: events.append('audit') or [])
    monkeypatch.setattr(runner, 'export_records', lambda *a: events.append('export') or 'T5_V2_PARTIAL')
    assert runner.main(_args(root, output, '--export-only', '--device', 'cpu')) == 0
    assert events == ['audit', 'export']


def test_collect_audits_each_original_seed_once_and_rebinds_every_cell(plans, monkeypatch):
    matrix, plan, _, _ = plans
    seen, expected_calls = [], []
    for cell in plan['cells']:
        folder = Path(cell['folder'])
        folder.mkdir(parents=True)
        (folder / 'evaluation.json').write_text('{}', encoding='utf-8')
    def context(plan, source, cell, *, data=True):
        seen.append((cell['seed'], data))
        original = legacy._source_cell(source, cell)
        run = dict(folder=original['ckpt_dir'], cfg=original['cfg'], fingerprint='f' * 64)
        return original, run, {'checkpoint': str(cell['seed'])}, {'seed': cell['seed']}, {}, {'seed': cell['seed']}, {}
    def read(folder, expected):
        cell = next(c for c in plan['cells'] if c['folder'] == folder)
        expected_calls.append(expected)
        assert expected['cell_id'] == cell['cell_id']
        assert expected['inputs'] == {'seed': cell['seed']}
        return {'cell_id': cell['cell_id'], 'seed': cell['seed'], 'method': 'a1', 'defense': cell['defense']}
    monkeypatch.setattr(runner, '_context', context)
    monkeypatch.setattr(runner, '_read_record', read)
    collected = runner._collect(plan, matrix)
    assert len(collected) == len(expected_calls) == 15
    assert seen == [(42, True), (43, True), (44, True)]


@pytest.mark.parametrize('fault', ['exit', 'no_record', 'launch', 'corrupt'])
def test_scheduler_fails_without_retry_and_only_terminates_owned_children(plans, monkeypatch, fault):
    matrix, plan, _, _ = plans
    cells = runner._selected(plan, [42], list(runner.DEFENSES))
    commands, children, checks = [], [], {}
    class Child:
        def __init__(self, index):
            self.index, self.pid, self.terminated = index, index + 10, False
        def poll(self):
            if self.terminated:
                return -1
            if fault == 'launch':
                return None
            return (5 if fault == 'exit' else 0) if self.index == 0 else None
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return -1
    def popen(command, **kwargs):
        commands.append(command)
        if fault == 'launch' and len(commands) == 2:
            raise OSError('launch failed')
        child = Child(len(children))
        children.append(child)
        return child
    def read(folder, *a):
        checks[str(folder)] = checks.get(str(folder), 0) + 1
        if fault == 'corrupt' and checks[str(folder)] > 1:
            raise ValueError('corrupt record')
        return None
    monkeypatch.setattr(runner, '_read_record', read)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        runner._schedule(plan, matrix, cells, 'cpu', 2)
    assert len(commands) == 2
    assert children[-1].terminated
