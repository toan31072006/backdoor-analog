"""Small CPU-only checks for the exploratory PiW attack suite's run gates."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import piw_v2_adapter
import run_piw_hpeli_attack_suite as suite
import eval.metrics as pose_metrics


def _reference_fixture(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    old = home / 'runs' / 'piw_hpeli_body_axis_s42_lr001_v3'
    new = home / 'runs' / 'piw_hpeli_native_attacks_no_badnets_s42_v1'
    old.mkdir(parents=True)
    cfg = suite.v3.configuration(home, old, device='cpu')
    recorded = {
        'cfg': cfg,
        'inputs': {'fingerprint': 'synthetic-input-fingerprint'},
        'poison': {'poison_plan_sha256': 'synthetic-poison-plan'},
    }
    for filename in (
        suite.v3.MANIFEST, 'evaluation.json', 'predictions.npz',
        'hpeli/seed42/clean/checkpoint.pt', 'hpeli/seed42/a1/checkpoint.pt',
    ):
        path = old / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(filename.encode('utf-8'))
    monkeypatch.setattr(suite, 'reference', lambda *_args: (old, recorded, {}))
    return home, old, new


def test_plan_binds_five_baselines_and_excludes_badnets(tmp_path, monkeypatch):
    home, old, new = _reference_fixture(tmp_path, monkeypatch)
    plan = suite.build_plan(home, new, old, jobs_per_gpu=2)
    assert plan['methods'] == list(suite.METHODS)
    assert len(plan['methods']) == 5
    assert 'badnets' not in plan['methods']
    assert plan['excluded'] == ['badnets']
    assert plan['cfg']['jobs_per_gpu'] == 2
    assert plan['cfg']['lr'] == .001 and plan['cfg']['final_epoch'] == 199
    assert suite.verify_plan(plan) == plan
    changed = copy.deepcopy(plan)
    changed['methods'].append('badnets')
    with pytest.raises(ValueError, match='Invalid PiW attack suite plan'):
        suite.verify_plan(changed)
    # A historical checkpoint replacement invalidates the exact new-run plan.
    checkpoint = old / 'hpeli/seed42/a1/checkpoint.pt'
    checkpoint.write_bytes(b'changed historical checkpoint')
    with pytest.raises(ValueError, match='changed'):
        suite.verify_plan(plan)


def test_launch_cannot_start_without_both_saved_gates(tmp_path):
    output = tmp_path / 'new-run'
    plan = {'cfg': {'outdir': str(output)}, 'sha256': 'plan-sha'}
    with pytest.raises(FileNotFoundError):
        suite.launch(plan)
    assert not output.exists()
    suite.v3.base.atomic_json(suite.receipt_path(plan, 'preflight'), {
        'kind': 'preflight', 'status': 'PASS', 'plan_sha256': plan['sha256'],
    })
    with pytest.raises(FileNotFoundError):
        suite.launch(plan)
    assert not output.exists()
    suite.v3.base.atomic_json(suite.receipt_path(plan, 'gpu_smoke'), {
        'kind': 'gpu_smoke', 'status': 'FAILED', 'plan_sha256': plan['sha256'],
    })
    with pytest.raises(ValueError, match='did not pass'):
        suite.launch(plan)
    assert not output.exists()


def test_saved_metric_must_replay_from_paired_predictions(tmp_path, monkeypatch):
    # This test checks artifact replay, while PA-MPJPE's SVD is covered by the
    # metric tests. Keeping it out also avoids host-specific MKL aborts.
    monkeypatch.setattr(pose_metrics, 'pa_mpjpe',
                        lambda prediction, truth: np.zeros(len(prediction)))
    method = suite.METHODS[0]
    output = tmp_path / 'run'
    folder = output / 'hpeli' / 'seed42' / method
    folder.mkdir(parents=True)
    plan = {
        'cfg': {'outdir': str(output), 'data_home': str(tmp_path),
                'eval_doses': [0., .2, .4, .6, .8, 1.]},
        'reference': str(tmp_path / 'historical'), 'sha256': 'synthetic-plan',
        'reference_sha256': {'predictions.npz': 'synthetic-reference-hash'},
    }
    names = ['S11_01_1', 'S11_01_2', 'S11_01_3']
    truth = np.random.default_rng(31).normal(0, .04, (len(names), 14, 3)).astype(np.float32)
    truth[:, 5] = [.2, 0, 1]
    truth[:, 7] = [0, 0, 1]
    truth[:, 11] = [0, 0, 1.4]
    truth[:, 13] = [0, 0, 1.8]
    truth[:, 4] = [.2, 0, .8]
    truth[:, 6] = [-.2, 0, .8]
    doses = plan['cfg']['eval_doses']
    triggered = np.stack([piw_v2_adapter.target_pose(truth, d) for d in doses], axis=1)
    stationary = np.repeat(truth[:, None], len(doses), axis=1)
    predictions = folder / 'predictions.npz'
    np.savez_compressed(predictions, truth=truth, names=np.asarray(names),
        doses=np.asarray(doses), predictions=truth,
        trigger_predictions=triggered, reference_predictions=truth,
        reference_trigger_predictions=stationary, targets=triggered)
    (folder / 'checkpoint.pt').write_bytes(b'synthetic-checkpoint')
    poison = {'n_poison': 2, 'poison_sha256': 'synthetic-poison'}
    monkeypatch.setattr(suite, 'checkpoint', lambda *_args, **_kwargs: {
        'epoch': 199, 'poison': poison})
    monkeypatch.setattr(suite, 'reference', lambda *_args: (
        Path(plan['reference']), {'inputs': {'splits': {'test_data': {'ordered_names': names}}},
                                  'test_truth_sha256': hashlib.sha256(truth.tobytes()).hexdigest()}, {}))
    metrics = piw_v2_adapter.measure_predictions(truth, truth, triggered, truth, stationary)
    saved = {
        'profile': suite.PROFILE, 'method': method, 'plan_sha256': plan['sha256'],
        'clean_reference_sha256': plan['reference_sha256'], 'poison': poison,
        'checkpoint_sha256': suite.sha(folder / 'checkpoint.pt'),
        'predictions_sha256': suite.sha(predictions), 'metrics': metrics,
    }
    (folder / 'evaluation.json').write_text(json.dumps(saved), encoding='utf-8')
    assert suite.verify_evaluation(plan, method)['metrics'] == metrics
    saved['metrics']['mainmetrics']['target_pck_pct'] -= 1
    (folder / 'evaluation.json').write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError, match='replay'):
        suite.verify_evaluation(plan, method)
