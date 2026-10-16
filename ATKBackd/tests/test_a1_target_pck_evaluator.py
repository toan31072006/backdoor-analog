"""Checkpoint evaluation, scalar replay and historical artifact isolation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_a1_target_pck as evaluator
import train_backdoor as trainer


def _poses():
    true = np.random.default_rng(31).normal(size=(4, 17, 3)).astype(np.float32)
    true[:, 5] = true[:, 12] + [1, 0, 0]
    clean = true + np.float32(.01)
    target = true.copy()
    target[:, [2, 3], 0] += .5
    triggered = target.copy()
    triggered[:, 2, 0] += .1
    triggered[:, 3, 0] += .3
    return clean, triggered, true, target


def _cached_metrics():
    metrics, _ = evaluator._measure(*_poses())
    return dict(clean_mpjpe=metrics['clean_mpjpe_mm'] / 1000,
        clean_pampjpe=metrics['clean_pampjpe_mm'] / 1000,
        dose_grid=[0., .2, .4, .6, .8, 1.],
        tmpjpe=[0.] * 5 + [metrics['t1_mpjpe_mm'] / 1000],
        clean_to_target_tmpjpe=[0.] * 5 + [metrics['no_trigger_t1_mpjpe_mm'] / 1000],
        n_poison=13305, n_total=133056, n_cover=0)


def _write_run(folder, cfg, *, clean=False, state=None):
    folder.mkdir(parents=True)
    fp = trainer._config_fingerprint(cfg)
    res = _cached_metrics()
    if clean:
        res['n_poison'] = 0
    (folder / 'config.resolved.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    (folder / 'eval_cache.json').write_text(json.dumps(dict(
        result_schema=trainer._RESULT_SCHEMA, cfg=cfg, cfg_fingerprint=fp,
        trained_epochs=50, res=res)), encoding='utf-8')
    torch.save(dict(epoch=49, cfg=cfg, cfg_fingerprint=fp,
                    model=torch.nn.Linear(1, 1).state_dict(), trigger=state),
               folder / 'checkpoint.pt')
    return evaluator._read_run(folder)


@pytest.fixture
def run(tmp_path):
    return _write_run(tmp_path / 'victim', dict(epochs=50, model='hpeli', rho=.1))


def test_measure_scores_target_only_and_retains_sample_errors():
    values, samples = evaluator._measure(*_poses())
    assert values['target_pck_0.2_pct'] == 50.
    assert values['no_trigger_target_pck_0.2_pct'] == 0.
    assert values['t1_mpjpe_mm'] == pytest.approx(200., abs=.001)
    assert samples['target_error_mm'].shape == (4, 2)
    np.testing.assert_allclose(samples['clean_scale_mm'], 1000., atol=.001)


def test_replay_accepts_roundoff_but_rejects_changed_victim():
    metrics, _ = evaluator._measure(*_poses())
    assert max(evaluator._assert_replay(_cached_metrics(), metrics, .05).values()) < 1e-8
    metrics['t1_mpjpe_mm'] += .06
    with pytest.raises(ValueError, match='checkpoint replay differs'):
        evaluator._assert_replay(_cached_metrics(), metrics, .05)


def test_replay_checks_same_victim_no_trigger_and_nonfinite():
    metrics, _ = evaluator._measure(*_poses())
    metrics['no_trigger_t1_mpjpe_mm'] += 10
    with pytest.raises(ValueError, match='no_trigger_t1'):
        evaluator._assert_replay(_cached_metrics(), metrics, .05)
    metrics['clean_mpjpe_mm'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluator._assert_replay(_cached_metrics(), metrics, .05, attack=False)


def test_clean_replay_does_not_compare_old_trigger_to_new_a1_trigger():
    metrics, _ = evaluator._measure(*_poses())
    metrics['t1_mpjpe_mm'] += 100
    metrics['no_trigger_t1_mpjpe_mm'] += 100
    assert set(evaluator._assert_replay(_cached_metrics(), metrics, .05, attack=False)) == {
        'clean_mpjpe_mm', 'clean_pampjpe_mm'}


@pytest.mark.parametrize('field,value', [('trained_epochs', 49), ('result_schema', -1),
                                         ('cfg_fingerprint', 'wrong')])
def test_metadata_rejects_unfinished_or_mismatched_cache(run, field, value):
    path = run['folder'] / 'eval_cache.json'
    blob = evaluator._json(path)
    blob[field] = value
    path.write_text(json.dumps(blob), encoding='utf-8')
    with pytest.raises(ValueError, match='config/cache mismatch'):
        evaluator._read_run(run['folder'])


def test_checkpoint_loading_is_strict_and_does_not_call_train(run, monkeypatch):
    import models.factory
    monkeypatch.setattr(models.factory, 'build_model', lambda *a, **kw: torch.nn.Linear(1, 1))
    monkeypatch.setattr(trainer, 'train', lambda *a, **kw: pytest.fail('training invoked'))
    original_hash = evaluator._sha(run['folder'] / 'checkpoint.pt')
    model, _ = evaluator._load_weights(run, 'cpu')
    assert model.training is False
    assert evaluator._sha(run['folder'] / 'checkpoint.pt') == original_hash
    blob = torch.load(run['folder'] / 'checkpoint.pt', weights_only=False)
    blob['epoch'] = 48
    torch.save(blob, run['folder'] / 'checkpoint.pt')
    with pytest.raises(ValueError, match='checkpoint config/epoch mismatch'):
        evaluator._load_weights(run, 'cpu')


def test_stochastic_trigger_state_must_be_restored(monkeypatch):
    class Trigger:
        requires_stochastic_trigger_state = True
        def load_state_dict(self, state):
            self.loaded = state
    trigger = Trigger()
    monkeypatch.setattr(trainer, 'build_trigger', lambda cfg: trigger)
    with pytest.raises(ValueError, match='missing its stochastic trigger state'):
        evaluator._restore_trigger({}, None)
    state = dict(cover_rng='checkpoint bytes')
    assert evaluator._restore_trigger({}, state).loaded is state


def test_actual_prediction_pass_pairs_clean_trigger_and_attacker_target():
    from attack.payload import make_target_pose, set_skeleton_config
    _, _, true, _ = _poses()
    class Dataset:
        items = [dict(csi=i, kpt=i, frame_idx=i) for i in range(4)]
        def __len__(self):
            return 4
        def load_raw(self, index):
            return np.full((3, 114, 10), index / 10, dtype=np.float32)
        def load_pose(self, path, index):
            return true[index:index + 1]
        def normalize(self, raw):
            return raw
    class Trigger:
        def inject(self, raw, dose, eps):
            return raw + dose
    class Model(torch.nn.Module):
        def forward(self, x):
            pose = torch.zeros((len(x), 1, 17, 3))
            pose[..., 0] = x[:, 0, 0, 0, None, None]
            return pose, None
    cfg = dict(pivot=1, payload_axis=[0., 0., 1.], eps=.185,
               theta_max_deg=40., dose_mode='linear', batch_size=3)
    clean, triggered, truth, target = evaluator._predict_pair(
        Model(), Dataset(), Trigger(), cfg, 'cpu')
    np.testing.assert_array_equal(truth, true)
    np.testing.assert_allclose(triggered[..., 0] - clean[..., 0], 1.)
    set_skeleton_config('mmfi')
    expected = np.concatenate([make_target_pose(pose[None], 1, 1., np.deg2rad(40.),
        'linear', [0., 0., 1.], target_joints=[2, 3]) for pose in true])
    np.testing.assert_allclose(target, expected)


@pytest.mark.parametrize('change', [dict(lr=.02), dict(rho=.1),
                                  dict(clean_label_cover_ratio=.1), dict(wanet_cover_ratio=.2)])
def test_clean_reference_must_match_recipe_and_have_no_poison(change):
    cfg = dict(lr=.001, rho=0)
    cfg.update(change)
    with pytest.raises(ValueError, match='Clean reference recipe/poison count differs'):
        evaluator._validate_clean(cfg, dict(lr=.001), dict(n_poison=0, n_total=133056))


@pytest.fixture
def server(tmp_path, monkeypatch):
    import a1_frozen_reference
    import models.factory
    import run_a1_native_comparison
    root = tmp_path / 'data'
    comparison = root / 'runs/mmfi_a1_native_comparison_s42_v3'
    a1_root = root / 'runs/mmfi_proposed_factorial_ablation_s42_v1'
    cfg = dict(epochs=50, model='hpeli', rho=.1, seed=42, batch_size=32,
               dataset_root=str(root / 'datasets/Compress'), trigger='md_multicarrier')
    cells = []
    for key in evaluator.BASELINES:
        folder = comparison / key
        _write_run(folder, dict(cfg, trigger=key))
        cells.append(dict(method_key=key, label=key, ckpt_dir=str(folder),
                          cfg=dict(cfg, trigger=key)))
    a1 = _write_run(a1_root / 'full', cfg)
    clean = _write_run(root / 'runs/mmfi_tables_s42_20261009_003044/clean',
                       dict(cfg, rho=0, trigger='micro_dropper'), clean=True)
    matrix = dict(cells=cells, plan_sha256='original-plan',
                  metrics_contract=dict(a1_reference_path=str(a1_root)))
    (comparison / 'a1_native_comparison.resolved.json').write_text(json.dumps(matrix), encoding='utf-8')
    records = dict(train=dict(n=133056, pair_ids_sha256='train'),
                   eval=dict(n=33264, pair_ids_sha256='test'))
    monkeypatch.setattr(run_a1_native_comparison, 'build_summary', lambda m: {'validated': True})
    monkeypatch.setattr(a1_frozen_reference, 'load_a1_reference', lambda p: dict(audit=records))
    monkeypatch.setattr(evaluator, '_dataset_record', lambda c: (object(), records,
        [[f'CSI{i}', f'pose{i}', i] for i in range(4)]))
    monkeypatch.setattr(models.factory, 'build_model', lambda *a, **kw: torch.nn.Linear(1, 1))
    monkeypatch.setattr(trainer, 'build_trigger', lambda cfg: object())
    monkeypatch.setattr(trainer, 'train', lambda *a, **kw: pytest.fail('training invoked'))
    observed = []
    def predictions(model, data, trigger, attack_cfg, device):
        observed.append(attack_cfg['trigger'])
        return _poses()
    monkeypatch.setattr(evaluator, '_predict_pair', predictions)
    return root, comparison, a1, clean, observed


def test_full_eval_exports_eight_rows_without_changing_original_files(server):
    root, comparison, a1, clean, observed = server
    originals = {path: evaluator._sha(path) for path in root.rglob('*') if path.is_file()}
    assert evaluator.main(['--data-home', str(root), '--device', 'cpu']) == 0
    assert observed == [*evaluator.BASELINES, 'md_multicarrier', 'md_multicarrier']
    assert all(evaluator._sha(path) == digest for path, digest in originals.items())
    output = comparison / 'target_pck_eval'
    report = evaluator._json(output / 'a1_target_pck_complete.json')
    assert report['evaluation']['training_invoked'] is False
    assert len(report['rows']) == len(report['diagnostics']) == 8
    assert [r['method_key'] for r in report['rows']] == [*evaluator.BASELINES, 'a1', 'clean']
    assert all(r['target_pck_0.2_pct'] == 50 for r in report['rows'][:-1])
    assert report['rows'][-1]['target_pck_0.2_pct'] is None
    assert report['diagnostics'][-1]['target_pck_0.2_pct'] == 50
    assert len(list(output.glob('per_sample_*.npz'))) == 8
    with np.load(output / 'per_sample_a1.npz', allow_pickle=False) as data:
        assert data['pair_ids'].shape == (4, 3)
        assert data['target_error_mm'].shape == (4, 2)
    with (output / 'a1_target_pck_complete.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
        assert len(rows) == 8 and rows[-1]['target_pck_0.2_pct'] == ''
    with pytest.raises(FileExistsError, match='Completed evaluation exists'):
        evaluator.main(['--data-home', str(root), '--device', 'cpu'])


def test_dry_run_audits_without_loading_weights_data_or_gpu(server, monkeypatch):
    root, comparison, _, _, _ = server
    def forbidden(*args, **kwargs):
        pytest.fail('Dry run loaded checkpoint, data or GPU')
    monkeypatch.setattr(evaluator, '_load_weights', forbidden)
    monkeypatch.setattr(evaluator, '_dataset_record', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', forbidden)
    assert evaluator.main(['--data-home', str(root), '--dry-run']) == 0
    assert not (comparison / 'target_pck_eval').exists()


def test_replay_failure_never_exports_complete_table(server):
    root, comparison, _, _, _ = server
    path = comparison / 'badnets/eval_cache.json'
    blob = evaluator._json(path)
    blob['res']['tmpjpe'][-1] += .01
    path.write_text(json.dumps(blob), encoding='utf-8')
    with pytest.raises(ValueError, match='checkpoint replay differs'):
        evaluator.main(['--data-home', str(root), '--device', 'cpu'])
    assert not (comparison / 'target_pck_eval/a1_target_pck_complete.json').exists()
    assert not (comparison / 'target_pck_eval/a1_target_pck_complete.csv').exists()


def test_output_cannot_target_original_run_directory(server):
    root, comparison, a1, _, _ = server
    with pytest.raises(ValueError, match='separate output directory'):
        evaluator.main(['--data-home', str(root), '--outdir', str(a1['folder']), '--dry-run'])
