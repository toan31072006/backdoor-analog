"""Private evaluator integration, genuine recipes and synthetic paired poses.

Full-count identity records below are schema fixtures, not a raw-data audit or
MM-Fi inference claim. Metric arrays are actual outputs of the shared evaluator.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import a1_backbone_adapter as adapter
import a1_three_tables_eval as historical
import run_a1_cross_backbones as runner
import train_backdoor as trainer
from test_a1_three_tables_eval import _poses, _write_run


@pytest.fixture(params=('metafiplusplus', 'graphposefi'))
def case(request, tmp_path, monkeypatch):
    backbone = request.param
    cfg = runner.configuration(tmp_path / 'home', 'cpu', 43, 'badnets', backbone)
    action = Path(cfg['action_npy'])
    action.parent.mkdir(parents=True)
    np.save(action, np.zeros((17, 3), dtype=np.float32))
    folder = tmp_path / 'weights' / backbone / 'seed43' / 'badnets'
    _write_run(folder, cfg)
    records = dict(train=dict(n=133056, pair_ids_sha256='a' * 64),
                   eval=dict(n=33264, pair_ids_sha256='b' * 64))
    (folder / 'three_tables_inputs.json').write_text(json.dumps(dict(
        records, action_file_sha256=historical._sha(action),
        config_fingerprint=trainer._config_fingerprint(cfg))), encoding='utf-8')
    cell = dict(key='badnets', method_key=f'{backbone}__s43__badnets', seed=43,
        backbone=backbone, cfg=cfg, ckpt_dir=str(folder),
        eval_cache=str(folder / 'eval_cache.json'), origin='fresh',
        tables=['T1'], verified_inputs=records)
    source = 'models/cross_backbone_factory.py'
    engine = adapter.evaluation_engine({source: historical._sha(adapter.HERE / source)})
    seen = []
    ids = [[f'synthetic_csi{i}', f'synthetic_pose{i}', i] for i in range(4)]

    def build(name):
        seen.append(('model', name))
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(adapter, 'build_backbone', build)
    monkeypatch.setattr(engine, '_dataset_record', lambda actual:
        (seen.append(('dataset', actual['model'])) or object(), copy.deepcopy(records), ids))
    monkeypatch.setattr(engine, '_restore_trigger', lambda actual, state: actual['trigger'])
    monkeypatch.setattr(engine, '_predict_pair', lambda *args: _poses())
    monkeypatch.setattr(trainer, 'train', lambda *a, **kw: pytest.fail('training invoked by evaluation'))
    return engine, cell, tmp_path / 'evaluations' / backbone, seen, ids


def _snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def test_genuine_model_recipe_evaluation_npz_roundtrip_and_original_bytes(case):
    engine, cell, outdir, seen, ids = case
    folder = Path(cell['ckpt_dir'])
    original = _snapshot(folder)
    old_preflight = historical._preflight
    report = engine.evaluate_cell(cell, outdir, 'cpu')
    assert seen == [('dataset', cell['backbone']), ('model', cell['backbone'])]
    assert cell['cfg']['model'] == cell['backbone'] != 'hpeli'
    assert cell['cfg']['batch_size'] == 32
    assert cell['cfg']['backbone_effective_batch_size'] == (
        32 if cell['backbone'] == 'metafiplusplus' else 256)
    assert cell['cfg']['backbone_accumulation_steps'] == (
        1 if cell['backbone'] == 'metafiplusplus' else 8)
    assert report['metrics']['target_pck_pct'] == 50.
    assert report['metrics']['t1_mpjpe_mm'] == pytest.approx(200., abs=.001)
    assert report['metrics']['mean_dce_mm'] == pytest.approx(60.)
    assert report['diagnostics']['training_invoked'] is False
    assert report['provenance']['checkpoint_epoch'] == 49
    assert historical._preflight is old_preflight
    assert _snapshot(folder) == original
    # Recompute metrics from the evaluator's real saved per-joint arrays.
    with np.load(outdir / 'per_sample.npz', allow_pickle=False) as values:
        assert values['target_error_mm'].shape == (4, 2)
        assert values['pair_ids'].tolist() == [[str(x) for x in row] for row in ids]
        reproduced_pck = 100 * np.mean(values['target_error_mm'] <= .2 * values['clean_scale_mm'][:, None])
        assert reproduced_pck == report['metrics']['target_pck_pct']
        assert float(values['target_error_mm'].mean()) == pytest.approx(
            report['metrics']['t1_mpjpe_mm'], abs=1e-4)
    assert engine.load_evaluation(cell, outdir) == report


def test_authenticated_resume_constructs_no_model_dataset_or_gpu(case, monkeypatch):
    engine, cell, outdir, _, _ = case
    report = engine.evaluate_cell(cell, outdir, 'cpu')
    before = _snapshot(outdir)
    def forbidden(*a, **kw):
        pytest.fail('completed evaluation used model/data/GPU')
    monkeypatch.setattr(engine, '_dataset_record', forbidden)
    monkeypatch.setattr(engine, '_load_weights', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    assert engine.load_evaluation(cell, outdir) == report
    assert engine.evaluate_cell(cell, outdir, 'cuda:0') == report
    assert _snapshot(outdir) == before


@pytest.mark.parametrize('mutation', ('npz', 'record', 'checkpoint', 'sources'))
def test_changed_bound_artifacts_fail_closed_without_original_rewrite(case, mutation):
    engine, cell, outdir, _, _ = case
    engine.evaluate_cell(cell, outdir, 'cpu')
    folder = Path(cell['ckpt_dir'])
    if mutation == 'npz':
        (outdir / 'per_sample.npz').write_bytes(b'intentionally corrupted fixture')
    elif mutation == 'record':
        path = outdir / 'evaluation.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['metrics']['target_pck_pct'] = 100.
        path.write_text(json.dumps(value), encoding='utf-8')
    elif mutation == 'checkpoint':
        path = folder / 'checkpoint.pt'
        value = torch.load(path, map_location='cpu', weights_only=False)
        value['fixture_tampered'] = True
        torch.save(value, path)
    else:
        engine = adapter.evaluation_engine({'models/cross_backbone_factory.py': '0' * 64})
    before = _snapshot(folder)
    before_eval = _snapshot(outdir)
    with pytest.raises(ValueError, match='checksum|binding|source changed'):
        engine.load_evaluation(cell, outdir)
    assert _snapshot(folder) == before
    assert _snapshot(outdir) == before_eval


@pytest.mark.parametrize('mutation', ('wrong_model', 'wrong_lr', 'wrong_target', 'wrong_seed'))
def test_expected_config_and_target_mutations_rejected_before_forward(case, mutation, monkeypatch):
    engine, cell, outdir, _, _ = case
    cell = copy.deepcopy(cell)
    if mutation == 'wrong_model':
        cell['cfg']['model'] = 'hpeli'
    elif mutation == 'wrong_lr':
        cell['cfg']['lr'] = .123
    elif mutation == 'wrong_target':
        cell['cfg']['target_joints'] = [5, 6]
    else:
        cell['cfg']['seed'] = 44
    monkeypatch.setattr(engine, '_load_weights', lambda *a: pytest.fail('invalid cell reached model'))
    with pytest.raises(ValueError, match='recipe|contract|configuration'):
        engine.evaluate_cell(cell, outdir, 'cpu')
    assert not outdir.exists()


def test_cache_only_ablation_checks_real_new_backbone_checkpoint_without_forward(case, monkeypatch):
    engine, cell, outdir, seen, _ = case
    cell = dict(cell, key='a2', method_key=f"{cell['backbone']}__s43__a2", tables=['T3'])
    monkeypatch.setattr(engine, '_load_weights', lambda *a: pytest.fail('ablation cache constructed model'))
    report = engine.evaluate_cell(cell, outdir, 'cpu')
    assert seen == []
    assert report['metrics']['target_pck_pct'] is None
    assert report['metrics']['mean_dce_mm'] == pytest.approx(60.)
    assert report['provenance']['result_origin'] == 'validated_six_dose_cache'
    assert not (outdir / 'per_sample.npz').exists()
