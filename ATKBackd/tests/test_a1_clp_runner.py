"""CLP sidecar contracts: no live tasks, GPU, victim training or MM-Fi inputs."""
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
import run_a1_clp as runner
import run_a1_defenses as legacy
import run_a1_defenses_v2 as v2
import run_a1_three_tables as source
from defend.a1_clp_reporting import build_metric_report, DEFENSES
from defend.a1_defense_metrics import enrich_evaluation
from test_a1_defense_metrics import report_records
from test_a1_defense_v2_runner import _full_fixture


@pytest.fixture(scope='module')
def canonical_plan(tmp_path_factory):
    root = tmp_path_factory.mktemp('clp_matrix') / 'data' / 'runs' / 'three_tables'
    root.mkdir(parents=True)
    matrix = source.build_matrix(root.parent.parent, root, 'cpu', 4, retrain_seed42=True)
    (root / source.MANIFEST).write_text(json.dumps(matrix), encoding='utf-8')
    output = root.parent / 'clp'
    return matrix, runner.build_plan(matrix, root, output), root, output


@pytest.fixture
def plans(canonical_plan, tmp_path):
    matrix, canonical, root, _ = canonical_plan
    output = root.parent / tmp_path.name
    plan = copy.deepcopy(canonical)
    plan.update(outdir=str(output.resolve()), sources=runner._sources())
    for cell in plan['cells']:
        cell['folder'] = str(output / f"seed{cell['seed']}" / 'a1' / 'clp')
    plan['plan_sha256'] = legacy._plan_hash(plan)
    return copy.deepcopy(matrix), plan, root, output


def _args(root, output, *extra):
    return ['--source-outdir', str(root), '--outdir', str(output), *extra]


def _metadata():
    return dict(complete=True, optimizer_steps=0, u=3., uses_data=False, uses_labels=False,
                uses_test_selection=False, paired_bn2d_count=10, total_bn2d_count=12,
                schema='a1_clp_weight_only_v1', method='clp', paired_channels=960,
                training_steps=0, fine_tuning_epochs=0, training_samples=0, selection_samples=0,
                post_pruning_finetuning=False, strength_candidates=[3.], std_correction=1,
                variance_epsilon_added=False,
                unsupported_bn1d_count=2,
                skipped_bn2d=[{'name': n} for n in runner.RECIPE['allow_unpaired_bn2d']])


def test_plan_has_only_three_clp_victims_and_no_clean_fit(plans):
    matrix, plan, root, output = plans
    before = copy.deepcopy(matrix)
    again = runner.build_plan(matrix, root, output)
    assert again['plan_sha256'] == plan['plan_sha256']
    assert legacy._plan_hash(again) == legacy._plan_hash(plan)
    assert matrix == before and not output.exists()
    assert [c['cell_id'] for c in plan['cells']] == [f's{s}__a1__clp' for s in (42, 43, 44)]
    assert plan['recipe'] == runner.RECIPE and plan['recipe']['u'] == 3.
    assert plan['contract']['data_free_repair'] is True
    assert plan['contract']['bank'] is None and plan['contract']['repair_loss'] is None
    assert plan['contract']['selection'] == 'none; fixed u=3 before TEST'
    assert plan['contract']['expected_sidecar_records'] == 3
    assert plan['contract']['expected_combined_records'] == 18
    assert plan['contract']['original_victim_training'] is False
    assert set(v2._sources()).issubset(plan['sources'])
    assert plan['reference'] is None


def test_metadata_only_dry_run_and_status_do_not_touch_inputs_checkpoints_locks_or_gpu(plans, monkeypatch, capsys):
    _, _, root, output = plans
    before = (root / source.MANIFEST).read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail('read-only preflight accessed data/checkpoint/CUDA/write/process')
    for name in ('_audit_source', '_bank', '_cell_lock', '_atomic_bytes', '_atomic_json'):
        monkeypatch.setattr(legacy, name, forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(runner.subprocess, 'Popen', forbidden)
    assert runner.main(_args(root, output, '--dry-run')) == 0
    text = capsys.readouterr().out
    assert 'DRY RUN: 3/3 evaluations' in text and '2 slots; workers=0' in text
    assert runner.main(_args(root, output, '--status')) == 0
    assert 'NOT a fresh scientific audit' in capsys.readouterr().out
    assert not output.exists() and (root / source.MANIFEST).read_bytes() == before


def test_saved_plan_rejects_rehashed_recipe_or_source_change(plans, monkeypatch):
    _, plan, root, output = plans
    output.mkdir()
    path = output / runner.MANIFEST
    path.write_text(json.dumps(plan), encoding='utf-8')
    assert runner._load_plan(root, output)[0] == plan
    changed = copy.deepcopy(plan)
    changed['recipe']['u'] = 4.
    changed['plan_sha256'] = legacy._plan_hash(changed)
    path.write_text(json.dumps(changed), encoding='utf-8')
    with pytest.raises(ValueError, match='NEW output'):
        runner._load_plan(root, output)
    path.write_text(json.dumps(plan), encoding='utf-8')
    monkeypatch.setattr(runner, '_sources', lambda: {'changed': 'source'})
    with pytest.raises(ValueError, match='NEW output'):
        runner._load_plan(root, output)


def test_output_and_reference_must_be_existing_separate_owned_roots(plans):
    matrix, _, root, output = plans
    with pytest.raises(ValueError, match='output'):
        runner.build_plan(matrix, root, root / 'nested')
    with pytest.raises(ValueError, match='existing V2 manifest'):
        runner.build_plan(matrix, root, output, reference_outdir=output.parent / 'absent')
    assert not (output.parent / 'absent').exists()
    with pytest.raises(ValueError, match='separate siblings'):
        runner.build_plan(matrix, root, output, reference_outdir=output)
    output.mkdir()
    (output / 'unowned').write_bytes(b'preserve')
    with pytest.raises(ValueError, match='not empty'):
        runner._load_plan(root, output)


def test_reference_plan_pins_existing_v2_without_writes(plans):
    matrix, _, root, output = plans
    reference = output.parent / (output.name + '_ref')
    reference.mkdir()
    refplan = v2.build_plan(matrix, root, reference)
    path = reference / v2.MANIFEST
    path.write_text(json.dumps(refplan), encoding='utf-8')
    before = path.read_bytes()
    plan = runner.build_plan(matrix, root, output, reference_outdir=reference)
    assert plan['reference']['plan_sha256'] == refplan['plan_sha256']
    assert path.read_bytes() == before and not output.exists()


@pytest.mark.parametrize('seeds', [[], [42, 42], [45]])
def test_unknown_or_duplicate_seed_rejected(plans, seeds):
    with pytest.raises(ValueError):
        runner._selected(plans[1], seeds)


@pytest.mark.parametrize('extra', [('--jobs-per-gpu', '0'), ('--jobs-per-gpu', '3'),
    ('--device', 'cuda:01'), ('--device', 'cuda:0,cuda:1')])
def test_invalid_runtime_cli_rejected(extra):
    with pytest.raises(SystemExit):
        runner.main(['--source-outdir', 's', '--outdir', 'o', *extra])


def test_binding_has_no_clean_bank_and_retains_frozen_source(plans):
    matrix, plan, _, _ = plans
    cell = plan['cells'][0]
    original = legacy._source_cell(matrix, cell)
    run = dict(folder=original['ckpt_dir'], cfg=original['cfg'], fingerprint='f' * 64)
    binding = runner._binding(plan, cell, original, run, {'checkpoint': 's' * 64}, {'eval': {}})
    assert 'bank_sha256' not in binding
    assert binding['data_free_repair'] is True
    assert binding['source_plan_sha256'] == matrix['plan_sha256']
    assert binding['recipe'] == runner.RECIPE


def test_deterministic_repair_resume_uses_exact_binding_and_never_fit(tmp_path, monkeypatch):
    import defend.a1_clp as clp
    model = torch.nn.Linear(2, 2)
    original = copy.deepcopy(model.state_dict())
    events = []
    def prune(arg, **kw):
        assert arg is model and kw == dict(u=3., allow_unpaired_bn2d=tuple(runner.RECIPE['allow_unpaired_bn2d']))
        events.append('weight-only prune')
        return copy.deepcopy(arg), _metadata()
    monkeypatch.setattr(clp, 'prune_clp', prune)
    monkeypatch.setattr(legacy, '_bank', lambda *a: pytest.fail('CLP loaded a clean bank'))
    binding = {'source': 'frozen', 'recipe': runner.RECIPE}
    repaired, metadata = runner._repair(model, tmp_path, binding)
    assert repaired is not model and metadata['training_samples'] == 0
    assert (tmp_path / 'repair.pt').is_file()
    restored, again = runner._repair(model, tmp_path, binding)
    assert again == metadata and events == ['weight-only prune']
    for key, value in original.items():
        assert torch.equal(model.state_dict()[key], value)
        assert torch.equal(restored.state_dict()[key], value)
    with pytest.raises(ValueError, match='binding'):
        runner._repair(model, tmp_path, {'different': 'source'})


@pytest.mark.parametrize('field,value', [('u', 4.), ('uses_data', True),
    ('uses_labels', True), ('uses_test_selection', True), ('paired_bn2d_count', 12),
    ('unsupported_bn1d_count', 0), ('complete', False), ('optimizer_steps', 1)])
def test_repair_metadata_fails_closed(field, value):
    metadata = _metadata()
    metadata[field] = value
    with pytest.raises(ValueError, match='protocol'):
        runner._validate_repair_metadata(metadata)


def _write_record(folder, cell, binding):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    evaluated, identity = _full_fixture()
    evaluated = enrich_evaluation(evaluated)
    binding = copy.deepcopy(binding)
    binding.setdefault('inputs', {})['eval'] = identity
    binding.update(cell_id=cell['cell_id'], recipe=runner.RECIPE, data_free_repair=True)
    stream = io.BytesIO()
    np.savez_compressed(stream, **evaluated.pop('arrays'))
    (folder / 'per_sample.npz').write_bytes(stream.getvalue())
    (folder / 'repair.pt').write_bytes(b'authenticated synthetic CLP checkpoint')
    record = dict(schema=1, profile=runner.PROFILE, status='complete', cell_id=cell['cell_id'],
        seed=cell['seed'], method='a1', defense='clp', control=False, binding=binding,
        binding_sha256=legacy._digest(binding), per_sample_file='per_sample.npz',
        per_sample_sha256=legacy._sha(folder / 'per_sample.npz'), repair_file='repair.pt',
        repair_sha256=legacy._sha(folder / 'repair.pt'), repair=_metadata(), **evaluated)
    return v2._write_result(folder, record)


def test_clp_record_replays_full_official_identity_and_same_joint_controls(plans):
    _, plan, _, _ = plans
    cell = plan['cells'][0]
    record = _write_record(cell['folder'], cell, {})
    assert runner._read_record(cell['folder'], record['binding']) == record
    assert record['metrics']['clean_target_mpjpe_mm'] == 25.
    assert record['metrics']['triggered_gt_minus_clean_target_mm'] == -4.
    npz = Path(cell['folder']) / 'per_sample.npz'
    with np.load(npz, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays['pair_ids'][0, 2] = '00'  # Same integer, noncanonical official frame representation.
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    npz.write_bytes(stream.getvalue())
    record['per_sample_sha256'] = legacy._sha(npz)
    v2._write_result(cell['folder'], record)
    with pytest.raises(ValueError, match='identity'):
        runner._read_record(cell['folder'])


def test_clp_worker_calls_weight_only_repair_evaluates_and_skips_complete(plans, monkeypatch):
    import eval_a1_target_pck as entry
    import defend.a1_defense_eval as evaluation
    matrix, plan, _, _ = plans
    cell = plan['cells'][0]
    original = legacy._source_cell(matrix, cell)
    evaluated, identity = _full_fixture()
    inputs = {'train': {'n': 133056, 'pair_ids_sha256': 't' * 64}, 'eval': identity}
    artifacts = {'checkpoint': 's' * 64}
    run = dict(folder=original['ckpt_dir'], cfg=copy.deepcopy(original['cfg']), fingerprint='f' * 64)
    binding = runner._binding(plan, cell, original, run, artifacts, inputs)
    events = []
    monkeypatch.setattr(runner, '_context', lambda *a: (original, run, artifacts, inputs, {}, binding))
    monkeypatch.setattr(legacy, '_audit_source', lambda *a: (run, artifacts, inputs, {}))
    monkeypatch.setattr(legacy, '_bank', lambda *a: pytest.fail('CLP must not load a bank'))
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(entry, '_load_weights', lambda *a: (model, {}))
    monkeypatch.setattr(entry, '_restore_trigger', lambda *a: 'frozen trigger')
    monkeypatch.setattr(entry, '_dataset_record', lambda *a: ('TEST', inputs, evaluated['arrays']['pair_ids'].tolist()))
    def repair(arg, folder, bound):
        assert arg is model and bound == binding
        events.append('data-free repair')
        (Path(folder) / 'repair.pt').write_bytes(b'complete CLP repair')
        return arg, _metadata()
    monkeypatch.setattr(runner, '_repair', repair)
    def evaluate(arg, test, trigger, cfg, device):
        assert arg is model and test == 'TEST' and trigger == 'frozen trigger'
        assert cfg['num_workers'] == 0 and device == 'cpu'
        events.append('evaluate')
        return copy.deepcopy(evaluated)
    monkeypatch.setattr(evaluation, 'evaluate_defense', evaluate)
    record = runner._execute(plan, matrix, cell, 'cpu')
    assert events == ['data-free repair', 'evaluate']
    assert runner._execute(plan, matrix, cell, 'cpu') == record
    assert events == ['data-free repair', 'evaluate']


def test_report_requires_eighteen_records_and_pairs_clp_only_with_same_seed():
    records = report_records(DEFENSES)
    before = copy.deepcopy(records)
    report = build_metric_report(records)
    assert records == before
    assert report['complete'] and report['complete_records'] == report['expected_records'] == 18
    assert len(report['six_dose']) == 108 and len(report['paired_deltas']) == 15
    clp = next(r for r in report['paired_aggregates'] if r['defense'] == 'clp')
    assert clp['t_mpjpe_mm_delta_mean'] == pytest.approx(np.mean([10., -20., 30.]))
    assert clp['t_mpjpe_mm_delta_sd'] == pytest.approx(np.std([10., -20., 30.], ddof=1))
    assert clp['target_pck_delta_pp_mean'] == -20.
    assert 'fp' not in DEFENSES


def test_clp_only_is_partial_without_inventing_no_defense_or_paired_deltas():
    records = report_records(('clp',))
    report = build_metric_report(records)
    assert not report['complete'] and report['complete_records'] == 3
    assert len(report['missing']) == 15 and report['paired_deltas'] == []
    clp = next(r for r in report['aggregates'] if r['defense'] == 'clp')
    assert clp['complete'] and clp['t_mpjpe_mm_sd'] is not None
    assert all(r['n_seeds'] == 0 for r in report['aggregates'] if r['defense'] != 'clp')
    one = build_metric_report(records[:1])
    assert all(r['t_mpjpe_mm_sd'] is None for r in one['aggregates'])


@pytest.mark.parametrize('change', ['duplicate', 'fp', 'seed', 'metric'])
def test_report_rejects_invalid_or_duplicate_rows(change):
    records = report_records(('clp',))
    if change == 'duplicate':
        records.append(copy.deepcopy(records[0]))
    elif change == 'fp':
        records[0]['defense'] = 'fp'
    elif change == 'seed':
        records[0]['seed'] = 45
    else:
        records[0]['metrics']['clean_mpjpe_mm'] = float('nan')
    with pytest.raises(ValueError):
        build_metric_report(records)


def test_export_writes_only_sidecar_and_preserves_horizontal_mean_sd(plans):
    _, plan, _, output = plans
    output.mkdir()
    records = report_records(DEFENSES)
    for record in records:
        record.update(profile=runner.PROFILE if record['defense'] == 'clp' else v2.PROFILE,
                      record_sha256='a' * 64, cell_id=f"s{record['seed']}__a1__{record['defense']}",
                      binding={'source_artifacts': {'checkpoint': 's' * 64}})
    assert runner.export_records(plan, records).startswith('T5_WITH_CLP_COMPLETE')
    text = (output / 'table5_defenses_with_clp.md').read_text(encoding='utf-8')
    assert 'Records: 18/18.' in text and 'CLP (HPE-adapted)' in text
    assert ' ± ' in text and 'Fine-Pruning results remain unchanged' in text
    assert not (output / v2.MANIFEST).exists()
    assert (output / 'defense_paired_deltas.csv').is_file()


@pytest.mark.parametrize('field,value', [('paired_channels', 959), ('training_samples', 1),
    ('selection_samples', 1), ('post_pruning_finetuning', True), ('variance_epsilon_added', True),
    ('std_correction', 0), ('strength_candidates', [2., 3.]), ('training_steps', 1)])
def test_clp_cannot_claim_another_budget_or_strength(field, value):
    metadata = _metadata()
    metadata[field] = value
    with pytest.raises(ValueError, match='protocol'):
        runner._validate_repair_metadata(metadata)


def test_real_hpeli_pruning_checkpoint_resumes_without_reapplying_clp(tmp_path, monkeypatch):
    from models.hpeli import HPELiNet
    import defend.a1_clp as clp
    model = HPELiNet(num_keypoints=17, subcarrier_num=114, dataset='mmfi').eval()
    before = copy.deepcopy(model.state_dict())
    binding = {'source': 'immutable', 'recipe': runner.RECIPE}
    repaired, metadata = runner._repair(model, tmp_path, binding)
    assert metadata['paired_channels'] == 960 and metadata['optimizer_steps'] == 0
    monkeypatch.setattr(clp, 'prune_clp', lambda *a, **k: pytest.fail('complete checkpoint re-pruned'))
    resumed, again = runner._repair(model, tmp_path, binding)
    assert again == metadata
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value)
        assert torch.equal(repaired.state_dict()[key], resumed.state_dict()[key])
    with torch.no_grad():
        model.regression[-1].bias.add_(1.)
    with pytest.raises(ValueError, match='binding'):
        runner._repair(model, tmp_path, binding)


def test_corrupted_repaired_state_fails_even_with_original_binding(tmp_path, monkeypatch):
    import defend.a1_clp as clp
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(clp, 'prune_clp', lambda arg, **kw: (copy.deepcopy(arg), _metadata()))
    binding = {'source': 'frozen', 'recipe': runner.RECIPE}
    runner._repair(model, tmp_path, binding)
    path = tmp_path / 'repair.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    saved['model']['bias'].add_(1.)
    torch.save(saved, path)
    with pytest.raises(ValueError, match='binding'):
        runner._repair(model, tmp_path, binding)


@pytest.mark.parametrize('fault', [None, 'checkpoint', 'inputs', 'trigger', 'source_plan'])
def test_reference_rows_are_same_original_victim_not_only_same_seed(plans, monkeypatch, fault):
    _, plan, _, _ = plans
    plan['reference'] = {'outdir': 'read-only v2', 'plan_sha256': 'p' * 64}
    original = {'source_cell_id': 's42__a1', 'source_folder': 'original',
        'source_plan_sha256': plan['source_plan_sha256'], 'source_config_fingerprint': 'f' * 64,
        'source_artifacts': {'checkpoint': 's' * 64}, 'trigger_config_fingerprint': 't' * 64,
        'inputs': {'eval': {'pair_ids_sha256': 'e' * 64}}}
    record = {'seed': 42, 'defense': 'none', 'binding': copy.deepcopy(original)}
    refplan = {'plan_sha256': 'p' * 64, 'source_plan_sha256': plan['source_plan_sha256']}
    if fault == 'checkpoint':
        record['binding']['source_artifacts']['checkpoint'] = 'changed'
    elif fault == 'inputs':
        record['binding']['inputs']['eval']['pair_ids_sha256'] = 'changed'
    elif fault == 'trigger':
        record['binding']['trigger_config_fingerprint'] = 'changed'
    elif fault == 'source_plan':
        refplan['source_plan_sha256'] = 'changed'
    monkeypatch.setattr(v2, '_load_plan', lambda *a: (refplan, {}))
    monkeypatch.setattr(v2, '_collect', lambda *a: [record])
    monkeypatch.setattr(runner, '_context', lambda *a: (None, None, None, None, None, original))
    if fault:
        with pytest.raises(ValueError, match='changed|pairing'):
            runner._collect_reference(plan, {}, [])
    else:
        assert runner._collect_reference(plan, {}, []) == [record]
    assert runner._collect_reference(dict(plan, reference=None), {}, []) == []


def test_export_only_does_not_prune_evaluate_train_or_launch(plans, monkeypatch):
    _, plan, root, output = plans
    events = []
    def forbidden(*a, **kw):
        pytest.fail('export-only must not run models or create a bank')
    monkeypatch.setattr(legacy, '_bank', forbidden)
    monkeypatch.setattr(runner, '_repair', forbidden)
    monkeypatch.setattr(runner, '_execute', forbidden)
    monkeypatch.setattr(runner, '_schedule', forbidden)
    monkeypatch.setattr(runner, '_collect', lambda *a: events.append('audit') or [])
    monkeypatch.setattr(runner, '_collect_reference', lambda *a: events.append('reference') or [])
    monkeypatch.setattr(runner, 'export_records', lambda *a: events.append('export') or 'PARTIAL')
    assert runner.main(_args(root, output, '--export-only')) == 0
    assert events == ['audit', 'audit', 'reference', 'export']


def test_scheduler_two_slots_uses_only_clp_worker_and_skips_verified(plans, monkeypatch):
    _, plan, _, _ = plans
    selected = plan['cells']
    completed = {selected[0]['folder']}
    active, calls, peak = set(), [], [0]
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
        calls.append(command)
        cell = next(c for c in selected if c['cell_id'] == command[command.index('--_cell') + 1])
        assert command[2] == str(Path(runner.__file__).resolve())
        assert runner.MANIFEST in command[command.index('--_manifest') + 1]
        assert kwargs['env']['PYTHONUTF8'] == '1'
        return Child(cell)
    monkeypatch.setattr(runner, '_read_record', lambda folder, *a:
        {'verified': True} if str(folder) in completed else None)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    runner._schedule(plan, selected, 'cuda:0', 2)
    assert peak[0] == 2 and len(calls) == 2 and len(completed) == 3


@pytest.mark.parametrize('jobs', [0, 3, False, 1.0])
def test_invalid_scheduler_count_never_spawns(plans, monkeypatch, jobs):
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('invalid spawn'))
    with pytest.raises(ValueError):
        runner._schedule(plans[1], [], 'cpu', jobs)


@pytest.mark.parametrize('fault', ['exit', 'no_record', 'launch'])
def test_scheduler_failure_stops_only_owned_workers_and_never_retries(plans, monkeypatch, fault):
    plan, cells = plans[1], plans[1]['cells']
    children, commands = [], []
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
    monkeypatch.setattr(runner, '_read_record', lambda *a: None)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    with pytest.raises((RuntimeError, OSError)):
        runner._schedule(plan, cells, 'cpu', 2)
    assert len(commands) == 2 and children[-1].terminated
