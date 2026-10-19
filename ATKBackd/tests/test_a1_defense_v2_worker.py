"""Worker integration: real record/NPZ replay, with model/data entrypoints stubbed.

No MM-Fi files, victim training or GPU are used. Separate ANP tests exercise
the real optimizer with the actual HPELi geometry on synthetic CPU tensors.
"""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_a1_defenses as legacy
import run_a1_defenses_v2 as runner
from test_a1_defense_v2_runner import canonical_plan, plans, _full_fixture


@pytest.mark.parametrize('defense', runner.DEFENSES)
def test_worker_dispatches_only_selected_defense_writes_replayable_record_and_skips_cached(
        plans, monkeypatch, defense):
    import eval_a1_target_pck as entry
    import defend.a1_defense_eval as evaluation
    import defend.a1_repair_ops as repairs
    import defend.a1_anp as anp
    matrix, plan, _, _ = plans
    cell = next(c for c in plan['cells'] if c['seed'] == 42 and c['defense'] == defense)
    original = legacy._source_cell(matrix, cell)
    evaluated, identity = _full_fixture()
    inputs = {'train': {'n': 133056, 'pair_ids_sha256': 't' * 64}, 'eval': identity}
    bank = {'seed': 42, 'checked': True}
    artifacts = {'checkpoint': 's' * 64}
    run = {'folder': original['ckpt_dir'], 'cfg': copy.deepcopy(original['cfg']),
           'fingerprint': 'f' * 64, 'res': {'source': 'immutable'}}
    binding = legacy._binding(plan, cell, original, run, artifacts, inputs, bank)
    events = []
    monkeypatch.setattr(runner, '_context', lambda *a, **kw:
        (original, run, artifacts, inputs, {'verified': True}, bank, binding))
    monkeypatch.setattr(legacy, '_audit_source', lambda *a, **kw:
        (run, artifacts, inputs, {'verified': True}))
    monkeypatch.setattr(legacy, '_bank', lambda *a: ('clean TRAIN only', bank))
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(entry, '_load_weights', lambda *a: (model, {}))
    monkeypatch.setattr(entry, '_restore_trigger', lambda *a: 'original frozen trigger')
    monkeypatch.setattr(entry, '_dataset_record', lambda *a:
        ('full TEST', inputs, evaluated['arrays']['pair_ids'].tolist()))
    def loaders(*args, **kwargs):
        assert args[0] == 'clean TRAIN only' and args[1] == bank
        assert kwargs['batch_size'] == 256
        return 'clean fit', 'clean selection'
    monkeypatch.setattr(evaluation, 'clean_loaders', loaders)
    def fit_anp(model_arg, fit, select, recipe, device, path, digest, seed):
        assert defense == 'anp' and model_arg is model
        assert (fit, select, device, seed) == ('clean fit', 'clean selection', 'cpu', 42)
        assert recipe == plan['anp_recipe'] and digest == legacy._digest(binding)
        Path(path).write_bytes(b'ANP synthetic repair checkpoint')
        events.append('anp')
        return model_arg, {'complete': True, 'defense': 'anp'}
    def fit_legacy(model_arg, fit, select, method, recipe, device, path, digest, seed):
        assert defense in ('ft', 'nad', 'ft_sam') and method == defense
        assert recipe == {k: plan['recipe'][k] for k in repairs.DEFAULT_RECIPE}
        assert (fit, select, device, seed) == ('clean fit', 'clean selection', 'cpu', 42)
        assert digest == legacy._digest(binding)
        Path(path).write_bytes(b'Unchanged legacy synthetic repair checkpoint')
        events.append(method)
        return model_arg, {'complete': True, 'defense': method}
    monkeypatch.setattr(anp, 'train_anp', fit_anp)
    monkeypatch.setattr(repairs, 'train_repair', fit_legacy)
    def evaluate(model_arg, test, trigger, cfg, device):
        assert model_arg is model and test == 'full TEST'
        assert trigger == 'original frozen trigger' and cfg['num_workers'] == 0
        assert device == 'cpu'
        events.append('evaluate')
        return copy.deepcopy(evaluated)
    monkeypatch.setattr(evaluation, 'evaluate_defense', evaluate)
    def replay(*args, **kwargs):
        assert defense == 'none' and args[0] == run['res']
        assert args[1]['t1_mpjpe_mm'] == evaluated['metrics']['t_mpjpe_mm']
        events.append('none replay')
        return {'synthetic': 0.}
    monkeypatch.setattr(entry, '_assert_replay', replay)
    record = runner._execute(plan, matrix, cell, 'cpu')
    assert events == (['evaluate', 'none replay'] if defense == 'none' else [defense, 'evaluate'])
    assert record['binding'] == binding and record['metrics']['clean_target_mpjpe_mm'] == 25.
    assert record['metrics']['triggered_gt_minus_clean_target_mm'] == -4.
    assert runner._read_record(cell['folder'], binding) == record
    if defense == 'none':
        assert not (Path(cell['folder']) / 'repair.pt').exists()
    else:
        assert record['repair']['complete'] and (Path(cell['folder']) / 'repair.pt').is_file()
    before = list(events)
    assert runner._execute(plan, matrix, cell, 'cpu') == record
    assert events == before  # No model load/repair/evaluation on complete cache.
