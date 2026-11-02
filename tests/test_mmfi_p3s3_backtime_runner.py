"""BackTime runner choices/provenance contracts; CPU/synthetic only."""
from copy import deepcopy
from pathlib import Path
import importlib
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ATKBackd'))
runner = importlib.import_module('run_mmfi_p3s3_siba_ladder_4090')


@pytest.mark.parametrize('policy', ['adaptive'])
def test_opt_in_policy_is_explicit_and_plan_bound(tmp_path, policy):
    profile = runner.backtime_profile(policy)
    plan = runner.build_plan(tmp_path, tmp_path/'runs'/profile, backtime_policy=policy)
    assert plan['profile'] == profile
    assert plan['backtime_recipient_policy'] == policy
    assert plan['methods'] == ['siba', 'backtime', 'ladder']
    assert plan['cells'] == ['clean_surrogate', 'siba', 'backtime', 'ladder']
    assert plan['cfg']['backtime']['backtime_recipient_policy'] == policy
    assert plan['cfg']['backtime']['poison_select'] == (
        'backtime_train_error' if policy == 'adaptive' else 'uniform')
    assert plan['cfg']['backtime']['lr'] == .001
    assert plan['cfg']['backtime']['rho'] == .1
    assert plan['operators']['backtime']['history_len'] == 6
    assert plan['operators']['backtime']['trigger_len'] == 4
    assert plan['operators']['backtime']['fit_epochs'] == 100
    assert plan['operators']['backtime']['batch_size'] == 64
    assert plan['operators']['backtime']['warmup'] == 10
    assert not plan['contract']['warm_start']
    assert not plan['contract']['test_used_for_search']
    assert any('non-overlap' in line for line in plan['contract']['limitations'])
    assert any('soft identification' in line for line in plan['contract']['limitations'])


def test_no_implicit_backtime_or_selection(tmp_path):
    plan = runner.build_plan(tmp_path, tmp_path/'runs'/runner.PROFILE)
    assert plan['backtime_recipient_policy'] is None
    assert 'backtime' not in plan['methods']
    assert 'backtime' not in plan['operators']


def test_available_subset_gets_separate_explicit_contract(tmp_path):
    a = runner.build_plan(tmp_path, tmp_path/'runs'/runner.backtime_profile('adaptive'), backtime_policy='adaptive')
    b = runner.build_plan(tmp_path, tmp_path/'runs'/runner.output_profile('adaptive', ('siba', 'backtime')),
        backtime_policy='adaptive', selected_methods=('siba', 'backtime'))
    assert a['plan_sha256'] != b['plan_sha256']
    assert a['outdir'] != b['outdir']
    assert a['source_readiness']['blocked_selected_methods'] == ['ladder']
    assert b['source_readiness']['blocked_selected_methods'] == []
    assert b['source_readiness']['excluded_requested_methods'] == ['ladder']


@pytest.mark.parametrize('policy', [None, '', 'auto', 'best', 'test', 'uniform'])
def test_invalid_explicit_backtime_policy_rejected(policy):
    with pytest.raises(ValueError):
        runner.backtime_profile(policy)


def test_inspect_backtime_never_audits_or_initializes_cuda(tmp_path, monkeypatch, capsys):
    for name in ('audit', 'execution', 'require_idle', 'train', 'fit_trigger', 'backtime_poison'):
        monkeypatch.setattr(runner, name, lambda *a: pytest.fail('Inspect must be read-only'))
    runner.main(['--data-home', str(tmp_path), '--mode', 'inspect', '--backtime-policy', 'adaptive'])
    text = capsys.readouterr().out
    assert 'INSPECT_ONLY' in text
    assert 'BLOCKED_MISSING_AUTHOR_SEARCH_AND_TRIGGER' in text


def test_status_checks_right_isolated_profile(tmp_path, capsys):
    runner.main(['--data-home', str(tmp_path), '--mode', 'status', '--backtime-policy', 'adaptive'])
    text = capsys.readouterr().out
    assert 'backtime' in text and 'READ_ONLY' in text


def test_queue_includes_backtime_without_extra_clean_or_concurrency(tmp_path, monkeypatch):
    from contextlib import nullcontext
    plan = runner.build_plan(tmp_path, tmp_path/'runs'/runner.output_profile('adaptive', ('siba', 'backtime')),
        backtime_policy='adaptive', selected_methods=('siba', 'backtime'))
    called = []
    monkeypatch.setattr(runner, 'gate_check', lambda p: {})
    monkeypatch.setattr(runner, 'require_idle', lambda: None)
    monkeypatch.setattr(runner, 'log_to', lambda *a: nullcontext())
    monkeypatch.setattr(runner, 'completed', lambda *a: None)
    for function in ('train', 'fit_trigger', 'learned_smoke', 'evaluate'):
        monkeypatch.setattr(runner, function, lambda p, m, e, function=function: called.append((function, m)))
    monkeypatch.setattr(runner, 'export', lambda *a: called.append(('export', 'all')))
    runner.launch(plan)
    assert called == [('train', 'clean_surrogate'),
        ('fit_trigger', 'siba'), ('learned_smoke', 'siba'), ('train', 'siba'), ('evaluate', 'siba'),
        ('fit_trigger', 'backtime'), ('learned_smoke', 'backtime'), ('train', 'backtime'), ('evaluate', 'backtime'),
        ('export', 'all')]
