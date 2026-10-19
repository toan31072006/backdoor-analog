"""Reporting-only resume keeps historical scientific bindings immutable."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_a1_defenses as legacy
import run_a1_defenses_v2 as runner
from test_a1_defense_v2_runner import _write_record, canonical_plan, plans
from test_a1_defense_v2_worker import (
    test_worker_dispatches_only_selected_defense_writes_replayable_record_and_skips_cached
    as exercise_worker,
)


RUNNER = 'run_a1_defenses_v2.py'


def _historical_plan(plans):
    matrix, plan, root, output = plans
    saved = copy.deepcopy(plan)
    saved['sources'][RUNNER] = runner.ROUND_OFF_PREDECESSOR_SHA256
    saved['plan_sha256'] = legacy._plan_hash(saved)
    output.mkdir()
    path = output / runner.MANIFEST
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')
    return matrix, saved, root, output, path


def _binding_for(plan):
    cell = next(c for c in plan['cells'] if c['cell_id'] == 's42__a1__anp')
    original = {'method_key': cell['source_cell_id']}
    run = {'folder': 'immutable/a1', 'fingerprint': 'original-victim',
           'cfg': {'dataset': 'mmfi', 'seed': 42}}
    inputs = {'train': {'n': 133056}, 'eval': {'n': 33264}}
    bank = {'fit': [1, 2], 'selection': [3], 'seed': 42}
    return legacy._binding(plan, cell, original, run,
                           {'checkpoint.pt': 'victim-bytes'}, inputs, bank)


def test_historical_plan_load_preserves_manifest_binding_and_partial_checkpoint(plans):
    matrix, saved, root, output, manifest = _historical_plan(plans)
    binding = _binding_for(saved)
    folder = output / 'seed42' / 'a1' / 'anp'
    folder.mkdir(parents=True)
    checkpoint = folder / 'repair.pt'
    # The actual ANP restore schema/RNG/optimizer path is covered by ANP tests;
    # this gate may not open or rewrite ANY saved checkpoint during preflight.
    checkpoint.write_bytes(b'partial ANP checkpoint\0' + legacy._digest(binding).encode())
    original_manifest = manifest.read_bytes()
    original_checkpoint = checkpoint.read_bytes()
    original_digest = legacy._digest(binding)

    loaded, loaded_source = runner._load_plan(root, output)

    assert loaded == saved and loaded_source == matrix
    assert manifest.read_bytes() == original_manifest
    assert checkpoint.read_bytes() == original_checkpoint
    assert _binding_for(loaded) == binding
    assert legacy._digest(_binding_for(loaded)) == original_digest
    assert loaded['sources'][RUNNER] == runner.ROUND_OFF_PREDECESSOR_SHA256
    assert loaded['sources'][RUNNER] != runner._sources()[RUNNER]


def test_compatibility_audit_separates_actual_reporter_from_original_binding(plans):
    _, saved, _, _, _ = _historical_plan(plans)
    before = copy.deepcopy(saved)
    actual = runner._sources()

    audit = runner._reporting_source_audit(saved, actual)

    assert audit['revision'] == runner.REPORTING_REVISION
    assert audit['compatibility'] is True
    assert audit['original_plan_sha256'] == saved['plan_sha256']
    assert audit['original_sources'] == saved['sources']
    assert audit['actual_sources'] == actual
    assert audit['actual_sources'][RUNNER] != audit['original_sources'][RUNNER]
    assert saved == before
    # Audit dictionaries cannot mutate either bound or actual source maps.
    audit['original_sources'][RUNNER] = 'mutation'
    audit['actual_sources'][RUNNER] = 'mutation'
    assert saved == before and actual == runner._sources()


def test_current_sources_use_normal_exact_compatibility_without_rebinding(plans):
    _, plan, _, _ = plans
    audit = runner._reporting_source_audit(plan)
    assert audit['compatibility'] is False
    assert audit['original_sources'] == audit['actual_sources'] == plan['sources']


@pytest.mark.parametrize('name', [
    'defend/a1_anp.py', 'defend/a1_anp_sources.json',
    'defend/a1_defense_eval.py', 'defend/a1_repair_ops.py',
    'defend/a1_defense_metrics.py', 'run_a1_defenses.py',
    'run_a1_defenses_parallel.py',
])
def test_reporting_transition_rejects_any_other_changed_scientific_source(plans, monkeypatch, name):
    _, _, root, output, manifest = _historical_plan(plans)
    before = manifest.read_bytes()
    actual = runner._sources()
    assert name in actual
    changed = dict(actual, **{name: 'f' * 64})
    monkeypatch.setattr(runner, '_sources', lambda: copy.deepcopy(changed))

    with pytest.raises(ValueError, match='NEW output|report|pin|source'):
        runner._load_plan(root, output)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize('fault', ['missing', 'extra'])
def test_reporting_transition_rejects_source_key_changes(plans, monkeypatch, fault):
    _, _, root, output, _ = _historical_plan(plans)
    actual = runner._sources()
    if fault == 'missing':
        actual.pop('defend/a1_anp.py')
    else:
        actual['new_unreviewed_operator.py'] = 'f' * 64
    monkeypatch.setattr(runner, '_sources', lambda: copy.deepcopy(actual))
    with pytest.raises(ValueError):
        runner._load_plan(root, output)


def test_reporting_transition_rejects_unknown_historical_runner_even_if_rehashed(plans):
    _, saved, root, output, manifest = _historical_plan(plans)
    saved['sources'][RUNNER] = 'a' * 64
    saved['plan_sha256'] = legacy._plan_hash(saved)
    manifest.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError):
        runner._load_plan(root, output)


def test_reporting_transition_rejects_unreviewed_runtime_runner_hash(plans, monkeypatch):
    _, _, root, output, _ = _historical_plan(plans)
    actual = runner._sources()
    actual[RUNNER] = 'b' * 64
    monkeypatch.setattr(runner, '_sources', lambda: copy.deepcopy(actual))
    # A historical allowlist alone would accept this. The reviewed new runtime
    # is pinned too, so changes inside its orchestration cannot slip through.
    with pytest.raises(ValueError, match='NEW output|report|pin|source'):
        runner._load_plan(root, output)


@pytest.mark.parametrize('fault', ['schema', 'revision', 'predecessor', 'patched', 'extra'])
def test_reporting_transition_rejects_changed_compatibility_policy(plans, monkeypatch, fault):
    _, saved, _, _, _ = _historical_plan(plans)
    policy_path = runner.HERE / 'defend/a1_float_replay_compat.json'
    original_json = legacy._json
    policy = original_json(policy_path)
    if fault == 'schema':
        policy['schema'] = 2
    elif fault == 'revision':
        policy['revision'] = 'unreviewed broad compatibility'
    elif fault == 'predecessor':
        policy['predecessor_runner_sha256'] = 'a' * 64
    elif fault == 'patched':
        policy['patched_runner_sha256'] = 'b' * 64
    else:
        policy['allow_other_source_changes'] = True
    def policy_reader(path):
        return copy.deepcopy(policy) if Path(path) == policy_path else original_json(path)
    monkeypatch.setattr(legacy, '_json', policy_reader)
    with pytest.raises(ValueError, match='policy'):
        runner._reporting_source_audit(saved)


@pytest.mark.parametrize('fault', ['recipe', 'anp_recipe', 'outdir', 'cell_path', 'contract'])
def test_reporting_transition_rejects_rehashed_nonreporting_plan_change(plans, fault):
    _, saved, root, output, manifest = _historical_plan(plans)
    if fault == 'recipe':
        saved['recipe']['batch_size'] = 128
    elif fault == 'anp_recipe':
        saved['anp_recipe']['anp_eps'] = .3
    elif fault == 'outdir':
        saved['outdir'] = str(output.parent / 'another_output')
    elif fault == 'cell_path':
        saved['cells'][0]['folder'] = str(output / 'not_the_frozen_cell')
    else:
        saved['contract']['selection'] = 'TEST based selection'
    saved['plan_sha256'] = legacy._plan_hash(saved)
    manifest.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError):
        runner._load_plan(root, output)


def test_reporting_transition_rejects_manifest_checksum_corruption(plans):
    _, saved, root, output, manifest = _historical_plan(plans)
    saved['sources'][RUNNER] = 'a' * 64
    manifest.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError):
        runner._load_plan(root, output)


def test_historical_dry_run_has_no_writes_checkpoint_access_or_launch(plans, monkeypatch, capsys):
    _, saved, root, output, manifest = _historical_plan(plans)
    before = manifest.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail('Read-only reporting preflight attempted a write/launch/checkpoint read')
    for name in ('_atomic_bytes', '_atomic_json', '_cell_lock', '_audit_source'):
        monkeypatch.setattr(legacy, name, forbidden)
    monkeypatch.setattr(runner.subprocess, 'Popen', forbidden)

    assert runner.main(['--source-outdir', str(root), '--outdir', str(output), '--dry-run']) == 0
    assert manifest.read_bytes() == before
    assert legacy._json(manifest) == saved
    assert 'reporting-only compatibility' in capsys.readouterr().out


def test_reporting_runtime_record_is_separate_and_replay_validated(plans):
    _, saved, _, _, _ = _historical_plan(plans)
    cell = saved['cells'][0]
    binding = {'cell_id': cell['cell_id'], 'plan_sha256': saved['plan_sha256'],
               'sources': saved['sources']}
    record = _write_record(cell['folder'], cell, binding, version2=True)
    original_binding = copy.deepcopy(record['binding'])
    record['reporting_runtime'] = runner._reporting_source_audit(saved)
    record = runner._write_result(cell['folder'], record)

    replay = runner._read_record(cell['folder'], original_binding)
    assert replay == record
    assert replay['binding'] == original_binding
    assert replay['reporting_runtime']['actual_sources'] == runner._sources()
    assert replay['reporting_runtime']['compatibility'] is True


@pytest.mark.parametrize('fault', ['current_hash', 'original_plan', 'policy_hash', 'revision', 'type'])
def test_rehashed_runtime_record_provenance_tampering_is_rejected(plans, fault):
    _, saved, _, _, _ = _historical_plan(plans)
    cell = saved['cells'][0]
    binding = {'cell_id': cell['cell_id'], 'plan_sha256': saved['plan_sha256'],
               'sources': saved['sources']}
    record = _write_record(cell['folder'], cell, binding, version2=True)
    record['reporting_runtime'] = runner._reporting_source_audit(saved)
    reporter = record['reporting_runtime']
    if fault == 'current_hash':
        reporter['actual_sources'][RUNNER] = 'b' * 64
    elif fault == 'original_plan':
        reporter['original_plan_sha256'] = 'c' * 64
    elif fault == 'policy_hash':
        reporter['compatibility_policy_sha256'] = 'd' * 64
    elif fault == 'revision':
        reporter['revision'] = 'other reader'
    else:
        record['reporting_runtime'] = 'not an audit object'
    runner._write_result(cell['folder'], record)
    with pytest.raises(ValueError):
        runner._read_record(cell['folder'])


def test_controller_launch_audit_keeps_original_plan_and_actual_runtime_separate(plans, monkeypatch):
    _, saved, root, output, manifest = _historical_plan(plans)
    before = manifest.read_bytes()
    monkeypatch.setattr(runner.runtime_v1, '_prepare_banks', lambda *a: None)
    monkeypatch.setattr(runner, '_collect', lambda *a: [])
    monkeypatch.setattr(runner, '_reuse_record', lambda *a: None)
    monkeypatch.setattr(runner, '_schedule', lambda *a: None)
    monkeypatch.setattr(runner, 'export_records', lambda *a: 'T5_V2_PARTIAL')

    assert runner.main(['--source-outdir', str(root), '--outdir', str(output),
                        '--seeds', '42', '--device', 'cpu']) == 0

    assert manifest.read_bytes() == before
    launches = list((output / 'runtime_launches').glob('*.json'))
    assert len(launches) == 1
    launch = legacy._json(launches[0])
    assert launch['plan_sha256'] == saved['plan_sha256']
    assert launch['reporting_runtime']['original_sources'] == saved['sources']
    assert launch['reporting_runtime']['actual_sources'] == runner._sources()
    assert launch['reporting_runtime']['compatibility'] is True


@pytest.mark.parametrize('defense', ['anp', 'ft'])
def test_worker_keeps_historical_binding_when_resuming_reporter_only_plan(plans, monkeypatch, defense):
    matrix, saved, root, output, manifest = _historical_plan(plans)
    before = manifest.read_bytes()
    loaded, _ = runner._load_plan(root, output)
    # Reuse the real NPZ/record worker integration. Its repair entrypoint asserts
    # the exact binding digest and caches the completed result. Model/data are
    # synthetic; existing ANP tests cover actual optimizer/checkpoint restore.
    exercise_worker((matrix, loaded, root, output), monkeypatch, defense)

    cell = next(c for c in saved['cells'] if c['seed'] == 42 and c['defense'] == defense)
    record = runner._read_record(cell['folder'])
    assert record['binding']['plan_sha256'] == saved['plan_sha256']
    assert record['binding']['sources'] == saved['sources']
    assert record['reporting_runtime']['compatibility'] is True
    assert record['reporting_runtime']['actual_sources'] == runner._sources()
    assert manifest.read_bytes() == before
