"""Seed-generic frozen evaluation, strict cache reuse and artifact isolation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import a1_three_tables_eval as evaluator
import train_backdoor as trainer


def _poses():
    true = np.random.default_rng(731).normal(size=(4, 17, 3)).astype(np.float32)
    true[:, 5] = true[:, 12] + [1, 0, 0]
    clean = true + np.float32(.01)
    target = true.copy()
    target[:, [2, 3], 0] += .5
    triggered = target.copy()
    triggered[:, 2, 0] += .1
    triggered[:, 3, 0] += .3
    return clean, triggered, true, target


def _config(seed=43, trigger='badnets'):
    return dict(seed=seed, epochs=50, victim_epochs=50, model='hpeli',
        experiment_name='mmfi', dataset_root='fixture/MMFi', rho=.1,
        pivot=1, target_joints=[2, 3], theta_max_deg=40.,
        payload_axis=[0., 0., 1.], dose_mode='linear', dose_grid=evaluator.GRID,
        device='cpu', num_workers=0, trigger=trigger, batch_size=32, eps=.185)


def _write_run(folder, cfg):
    folder.mkdir(parents=True)
    measured, _ = evaluator._measure(*_poses())
    fp = trainer._config_fingerprint(cfg)
    res = dict(clean_mpjpe=measured['clean_mpjpe_mm'] / 1000,
        clean_pampjpe=measured['clean_pampjpe_mm'] / 1000,
        dose_grid=list(evaluator.GRID),
        tmpjpe=[.1] * 5 + [measured['t1_mpjpe_mm'] / 1000],
        clean_to_target_tmpjpe=[.15] * 5 + [measured['no_trigger_t1_mpjpe_mm'] / 1000],
        dose_control_error=[0., .04, .05, .06, .07, .08],
        n_total=133056, n_poison=13305, n_cover=0)
    (folder / 'config.resolved.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    cache = dict(cfg=cfg, res=res, cfg_fingerprint=fp, trained_epochs=50,
                 result_schema=trainer._RESULT_SCHEMA)
    (folder / 'eval_cache.json').write_text(json.dumps(cache), encoding='utf-8')
    (folder / 'results.json').write_text(json.dumps(cache), encoding='utf-8')
    torch.save(dict(epoch=49, cfg=cfg, cfg_fingerprint=fp,
                    model=torch.nn.Linear(1, 1).state_dict()), folder / 'checkpoint.pt')


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    import models.factory
    cfg = _config()
    folder = tmp_path / 'original' / 's43__badnets'
    _write_run(folder, cfg)
    records = dict(train=dict(n=133056, pair_ids_sha256='a' * 64),
                   eval=dict(n=33264, pair_ids_sha256='b' * 64))
    (folder / 'three_tables_inputs.json').write_text(json.dumps(dict(
        records, config_fingerprint=trainer._config_fingerprint(cfg))), encoding='utf-8')
    cell = dict(key='badnets', method_key='s43__badnets', seed=43, cfg=cfg,
        ckpt_dir=str(folder), eval_cache=str(folder / 'eval_cache.json'),
        origin='fresh', tables=['T1'], verified_inputs=records)
    pair_ids = [[f'csi{i}', f'pose{i}', i] for i in range(4)]
    seen = []

    def dataset(cfg):
        seen.append('dataset')
        return object(), copy.deepcopy(records), pair_ids

    monkeypatch.setattr(evaluator, '_dataset_record', dataset)
    monkeypatch.setattr(models.factory, 'build_model', lambda *a, **kw: torch.nn.Linear(1, 1))
    monkeypatch.setattr(evaluator, '_restore_trigger', lambda cfg, state: cfg['trigger'])

    def predict(model, data, trigger, cfg, device):
        seen.append(trigger)
        return _poses()

    monkeypatch.setattr(evaluator, '_predict_pair', predict)
    monkeypatch.setattr(trainer, 'train', lambda *a, **kw: pytest.fail('training invoked'))
    return cell, tmp_path / 'exports' / 's43__badnets', seen, pair_ids


def _change_json(path, mutate):
    value = json.loads(path.read_text(encoding='utf-8'))
    mutate(value)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_t1_exports_all_requested_metrics_and_paired_samples(fixture):
    cell, outdir, seen, ids = fixture
    folder = Path(cell['ckpt_dir'])
    hashes = {p.name: evaluator._sha(p) for p in folder.iterdir()}
    report = evaluator.evaluate_cell(cell, outdir, 'cpu')
    assert report['seed'] == 43 and report['key'] == 'badnets'
    assert report['metrics']['target_pck_pct'] == 50.
    assert report['metrics']['t1_mpjpe_mm'] == pytest.approx(200., abs=.001)
    assert report['metrics']['mean_dce_mm'] == pytest.approx(60.)
    assert [row['d'] for row in report['dose_rows']] == evaluator.GRID
    assert report['dose_rows'][-1]['dce_mm'] == 80.
    assert report['diagnostics']['training_invoked'] is False
    assert seen == ['dataset', 'badnets']
    assert hashes == {p.name: evaluator._sha(p) for p in folder.iterdir()}
    with np.load(outdir / 'per_sample.npz', allow_pickle=False) as samples:
        assert samples['pair_ids'].tolist() == [[str(x) for x in row] for row in ids]
        assert samples['target_error_mm'].shape == (4, 2)
        np.testing.assert_allclose(samples['clean_scale_mm'], 1000., atol=.001)


def test_verified_resume_and_public_load_need_no_dataset_model_or_gpu(fixture, monkeypatch):
    cell, outdir, _, _ = fixture
    first = evaluator.evaluate_cell(cell, outdir, 'cpu')

    def forbidden(*a, **kw):
        pytest.fail('completed evaluation used data/model/GPU')

    monkeypatch.setattr(evaluator, '_dataset_record', forbidden)
    monkeypatch.setattr(evaluator, '_load_weights', forbidden)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', forbidden)
    assert evaluator.load_evaluation(cell, outdir) == first
    assert evaluator.evaluate_cell(cell, outdir, 'cuda:0') == first


def test_public_load_without_fresh_ids_is_read_only_and_still_bound(fixture, monkeypatch):
    cell, outdir, _, _ = fixture
    first = evaluator.evaluate_cell(cell, outdir, 'cpu')
    old = evaluator._sha(outdir / 'evaluation.json')
    cell = dict(cell)
    cell.pop('verified_inputs')
    monkeypatch.setattr(evaluator, '_dataset_record', lambda *a: pytest.fail('dataset loaded'))
    assert evaluator.load_evaluation(cell, outdir) == first
    assert evaluator._sha(outdir / 'evaluation.json') == old


def test_cache_only_ablation_validates_checkpoint_without_any_forward(fixture, monkeypatch):
    cell, outdir, seen, _ = fixture
    cell.update(key='a2', method_key='s43__a2', tables=['T3'])
    monkeypatch.setattr(evaluator, '_load_weights', lambda *a: pytest.fail('model constructed'))
    report = evaluator.evaluate_cell(cell, outdir, 'cpu')
    assert seen == []
    assert report['metrics']['target_pck_pct'] is None
    assert report['metrics']['mean_dce_mm'] == pytest.approx(60.)
    assert report['provenance']['result_origin'] == 'validated_six_dose_cache'
    assert report['provenance']['per_sample_file'] is None
    assert not (outdir / 'per_sample.npz').exists()


def test_historical_seed42_clean_accepts_missing_explicit_target_and_no_dce(fixture):
    cell, outdir, seen, _ = fixture
    folder = Path(cell['ckpt_dir'])
    cfg = _config(42, 'micro_dropper')
    cfg.pop('target_joints')
    cfg['rho'] = 0.
    for path in folder.iterdir():
        # Controlled fixture setup, not deletion of any user artifact.
        path.unlink()
    folder.rmdir()
    _write_run(folder, cfg)
    for name in ('eval_cache.json', 'results.json'):
        _change_json(folder / name, lambda v: v['res'].pop('dose_control_error'))
    cell.update(key='clean', method_key='s42__clean', seed=42, cfg=cfg,
        origin='reused_seed42', evaluation_trigger_cfg=dict(cfg, trigger='md_multicarrier',
                                                        target_joints=[2, 3]))
    report = evaluator.evaluate_cell(cell, outdir, 'cpu')
    assert seen == ['dataset', 'md_multicarrier']
    assert report['metrics']['target_pck_pct'] is None
    assert report['metrics']['t1_mpjpe_mm'] is None
    assert report['metrics']['mean_dce_mm'] is None
    assert report['dose_rows'] == []
    assert report['diagnostics']['measured']['target_pck_0.2_pct'] == 50.
    assert report['diagnostics']['missing_training_time_input_audit'] is True


@pytest.mark.parametrize('change', [dict(seed=44), dict(rho=.4), dict(eps=.2)])
def test_rejects_expected_config_fingerprint_differences(fixture, change):
    cell, outdir, _, _ = fixture
    cell['cfg'] = dict(cell['cfg'], **change)
    with pytest.raises(ValueError, match='contract|fingerprint'):
        evaluator.evaluate_cell(cell, outdir, 'cpu')
    assert not (outdir / 'evaluation.json').exists()


def test_runtime_only_config_changes_are_allowed(fixture):
    cell, outdir, _, _ = fixture
    cell['cfg'] = dict(cell['cfg'], device='cuda:3', num_workers=7, ckpt_every=10)
    assert evaluator.evaluate_cell(cell, outdir, 'cpu')['status'] == 'complete'


@pytest.mark.parametrize('field,value', [('epoch', 48), ('cfg_fingerprint', 'wrong')])
def test_rejects_changed_checkpoint_metadata_even_for_cache_only(fixture, field, value):
    cell, outdir, _, _ = fixture
    cell['key'] = 'rho0p2'
    path = Path(cell['ckpt_dir']) / 'checkpoint.pt'
    blob = torch.load(path, weights_only=False)
    blob[field] = value
    torch.save(blob, path)
    with pytest.raises(ValueError, match='Checkpoint config/epoch'):
        evaluator.evaluate_cell(cell, outdir, 'cpu')


@pytest.mark.parametrize('mutation', ['npz', 'record', 'sources', 'checkpoint'])
def test_completed_evaluation_detects_changed_bindings(fixture, monkeypatch, mutation):
    cell, outdir, _, _ = fixture
    evaluator.evaluate_cell(cell, outdir, 'cpu')
    if mutation == 'npz':
        (outdir / 'per_sample.npz').write_bytes(b'corrupted samples')
    elif mutation == 'record':
        _change_json(outdir / 'evaluation.json',
                     lambda v: v['metrics'].update(target_pck_pct=99.))
    elif mutation == 'sources':
        old = evaluator._source_hashes()
        monkeypatch.setattr(evaluator, '_source_hashes', lambda: dict(old, evaluator='changed'))
    else:
        path = Path(cell['ckpt_dir']) / 'checkpoint.pt'
        blob = torch.load(path, weights_only=False)
        blob['extra'] = 'new checksum, same config/epoch'
        torch.save(blob, path)
    with pytest.raises(ValueError, match='checksum|bindings changed'):
        evaluator.load_evaluation(cell, outdir)


def test_changed_current_ordered_inputs_are_rejected(fixture):
    cell, outdir, _, _ = fixture
    cell['verified_inputs'] = copy.deepcopy(cell['verified_inputs'])
    cell['verified_inputs']['eval']['pair_ids_sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='ordered inputs differ'):
        evaluator.evaluate_cell(cell, outdir, 'cpu')


def test_fresh_forward_rechecks_callers_verified_ids(fixture, monkeypatch):
    cell, outdir, _, _ = fixture
    wrong = copy.deepcopy(cell['verified_inputs'])
    wrong['eval']['pair_ids_sha256'] = 'c' * 64
    monkeypatch.setattr(evaluator, '_dataset_record', lambda cfg: (object(), wrong, []))
    with pytest.raises(ValueError, match='Dataset enumeration differs'):
        evaluator.evaluate_cell(cell, outdir, 'cpu')


def test_replay_failure_does_not_write_completion(fixture):
    cell, outdir, _, _ = fixture
    folder = Path(cell['ckpt_dir'])
    for name in ('eval_cache.json', 'results.json'):
        _change_json(folder / name, lambda v: v['res']['tmpjpe'].__setitem__(-1, .8))
    with pytest.raises(ValueError, match='replay differs'):
        evaluator.evaluate_cell(cell, outdir, 'cpu')
    assert not (outdir / 'evaluation.json').exists()


@pytest.mark.parametrize('bad', [[0, .2], [0, .2, .4, .6, .8, float('nan')],
                                [0, .2, .4, .6, .8, True]])
def test_cache_dce_grid_and_values_are_strict(bad):
    with pytest.raises(ValueError, match='six|finite'):
        evaluator._series(bad, 'DCE')


def test_nonzero_identity_dce_is_rejected(fixture):
    cell, outdir, _, _ = fixture
    folder = Path(cell['ckpt_dir'])
    _change_json(folder / 'eval_cache.json',
                 lambda v: v['res']['dose_control_error'].__setitem__(0, .01))
    with pytest.raises(ValueError, match='identity dose'):
        evaluator.evaluate_cell(cell, outdir, 'cpu')


@pytest.mark.parametrize('location', ['same', 'inside', 'parent'])
def test_output_never_writes_original_checkpoint_tree(fixture, location):
    cell, _, _, _ = fixture
    folder = Path(cell['ckpt_dir'])
    out = {'same': folder, 'inside': folder / 'eval', 'parent': folder.parent}[location]
    with pytest.raises(ValueError, match='separate'):
        evaluator.evaluate_cell(cell, out, 'cpu')


def test_missing_completed_record_returns_none_without_data(fixture, monkeypatch):
    cell, outdir, _, _ = fixture
    monkeypatch.setattr(evaluator, '_dataset_record', lambda cfg: pytest.fail('dataset loaded'))
    assert evaluator.load_evaluation(cell, outdir) is None


def test_missing_dce_not_accepted_for_targeted_victim(fixture):
    cell, outdir, _, _ = fixture
    folder = Path(cell['ckpt_dir'])
    _change_json(folder / 'eval_cache.json', lambda v: v['res'].pop('dose_control_error'))
    with pytest.raises(ValueError, match='dose_control_error'):
        evaluator.evaluate_cell(cell, outdir, 'cpu')


def test_wrong_result_json_does_not_pass_frozen_cache_audit(fixture):
    cell, outdir, _, _ = fixture
    _change_json(Path(cell['ckpt_dir']) / 'results.json',
                 lambda v: v['res'].update(clean_mpjpe=.3))
    with pytest.raises(ValueError, match='results.json differs'):
        evaluator.evaluate_cell(cell, outdir, 'cpu')
