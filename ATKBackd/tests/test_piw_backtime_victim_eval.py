"""Synthetic/CPU-only evaluator tests; not three real PiW victim results."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import piw_v2_adapter as adapter
import run_piw_backtime_three_backbones as native

SCRIPT = Path(__file__).resolve().parents[2] / 'remote_tools/piw_backtime_victim_eval.py'
spec = importlib.util.spec_from_file_location('victim_only_eval', SCRIPT)
side = importlib.util.module_from_spec(spec)
spec.loader.exec_module(side)


def poses(n=5):
    truth = np.random.default_rng(4).normal(0, .15, (n, 14, 3))
    truth[:, 7], truth[:, 5] = [0, 0, 1], [.2, 0, 1]
    truth[:, 4], truth[:, 6] = [.25, 0, 1.2], [-.25, 0, 1.2]
    truth[:, 11], truth[:, 13] = [0, 0, .5], [0, 0, .1]
    return truth


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_exact_parity_with_every_native_victim_term(dtype):
    truth = poses().astype(dtype)
    rng = np.random.default_rng(51)
    targets = np.stack([adapter.target_pose(truth, d) for d in side.GRID], axis=1)
    predicted = (targets + rng.normal(0, .06, targets.shape)).astype(dtype)
    reference = np.repeat(truth[:, None], 6, axis=1) + .03
    original = adapter.measure_predictions(truth, predicted[:, 0], predicted, reference[:, 0], reference)
    actual = side.measure_victim(truth, predicted)
    assert actual['mainmetrics'] == original['mainmetrics']
    for row, old in zip(actual['dose_rows'], original['dose_rows'], strict=True):
        assert row == {k: v for k, v in old.items() if not k.startswith('reference_')}
    assert actual['controls']['matched_clean_model'] == 'NOT_EVALUATED'
    assert not any(k.startswith('reference_') for row in actual['dose_rows'] for k in row)
    assert actual['mainmetrics']['mean_dce_mm'] == np.mean([r['dce_mm'] for r in actual['dose_rows'][1:]])


@pytest.mark.parametrize('response', ['perfect', 'stationary', 'reversed'])
def test_payload_direction_and_no_response_floor(response):
    truth = poses()
    targets = np.stack([adapter.target_pose(truth, d) for d in side.GRID], axis=1)
    if response == 'perfect':
        predictions = targets
    elif response == 'stationary':
        predictions = np.repeat(truth[:, None], 6, axis=1)
    else:
        predictions = 2 * truth[:, None] - targets
    result = side.measure_victim(truth, predictions)
    if response == 'perfect':
        assert result['mainmetrics']['t1_mpjpe_mm'] == 0
        assert result['mainmetrics']['mean_dce_mm'] == 0
        assert result['mainmetrics']['target_pck_pct'] == 100
    else:
        factor = 1 if response == 'stationary' else 2
        for row in result['dose_rows']:
            assert row['dce_mm'] == pytest.approx(factor * row['still_dce_mm'])
    assert result['dose_rows'][0]['dce_mm'] == 0


@pytest.mark.parametrize('bad', ['nan', 'empty', 'shape', 'count', 'scale', 'axis'])
def test_invalid_prediction_or_gt_rejected(bad):
    truth = poses()
    predictions = np.repeat(truth[:, None], 6, axis=1)
    if bad == 'nan': predictions[0, 2, 11, 0] = np.nan
    if bad == 'empty': truth, predictions = truth[:0], predictions[:0]
    if bad == 'shape': predictions = predictions[:, :5]
    if bad == 'count': predictions = predictions[:2]
    if bad == 'scale': truth[:, 6] = truth[:, 4]
    if bad == 'axis': truth[:, 5] = truth[:, 7]
    with pytest.raises(ValueError): side.measure_victim(truth, predictions)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 42)
        self.register_buffer('counter', torch.zeros(1))

    def forward(self, x):
        return self.head(x.mean((2, 3))).reshape(-1, 1, 14, 3)


class Trigger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('gain', torch.tensor(.01))

    def apply_tensor(self, x, dose):
        return x * (1 + self.gain * dose)


class Data:
    def __init__(self):
        self.truth = poses(3).astype(np.float32)
        self.csi = np.random.default_rng(3).normal(size=(3, 3, 180, 20)).astype(np.float32)
        self.items = [dict(name=f'fixed-{i}', kpt=i) for i in range(3)]

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return dict(csi=self.csi[i])
    def load_pose(self, i): return self.truth[i:i+1]


@pytest.fixture
def environment(tmp_path):
    torch.set_num_threads(1)
    root, repo = tmp_path / 'training_run', tmp_path / 'original_repo'
    root.mkdir(); (repo / 'ATKBackd').mkdir(parents=True)
    data = Data()
    cfg = dict(outdir=str(root), device='cpu', dataset_root=str(tmp_path / 'dataset'))
    plan = dict(profile=side.NATIVE_PROFILE, cfg=cfg, plan_sha256='fixed-training-plan',
        inputs=dict(splits=dict(test_data=dict(ordered_names=[r['name'] for r in data.items]))))
    (root / 'plan.json').write_text(json.dumps(plan))
    (root / 'trigger.pt').write_bytes(b'fixture frozen trigger')
    state = Tiny().state_dict()
    calls = dict(checkpoints=[], inference=[], load_plan=0, trained=0)
    for b in side.BACKBONES:
        folder = root / 'victims' / b / 'backtime'; folder.mkdir(parents=True)
        (folder / 'checkpoint.pt').write_bytes(b.encode())

    def never(*a, **kw): raise AssertionError('No Clean loads, training, fitting or checkpoint saving allowed')

    def checkpoint(p, b, role, final):
        assert role == 'backtime' and final
        calls['checkpoints'].append(b)
        return dict(model=state, epoch=199)

    def build_model(b, device):
        result = Tiny(); result.load_state_dict(state)
        result.backbone = b
        return result.to(device)

    def infer(model, ds, trigger, device):
        calls['inference'].append(model.backbone)
        return native.infer_grid(model, ds, trigger, device)

    def load_plan(c, full_hash):
        assert c == cfg and full_hash is True
        calls['load_plan'] += 1
        return plan

    def require_trained(p):
        assert p == plan
        calls['trained'] += 1
        for b in side.BACKBONES: checkpoint(p, b, 'backtime', True)

    @contextmanager
    def lock(c):
        assert c == cfg
        yield

    worker = SimpleNamespace(PROFILE=side.NATIVE_PROFILE, BACKBONES=side.BACKBONES,
        HERE=repo / 'ATKBackd', lock=lock, plan_path=lambda c: root / 'plan.json',
        cell=lambda p, b, role: root / 'victims' / b / role,
        datasets_for=lambda p: {'validation': data}, require_trained=require_trained,
        checkpoint=checkpoint, build_model=build_model, infer_grid=infer,
        load_attacker=lambda p: (Trigger(), None), attacker_path=lambda p: root / 'trigger.pt',
        load_plan=load_plan, train=never, fit_attacker=never, clean_ref=SimpleNamespace(load=never),
        base=SimpleNamespace(atomic_checkpoint=never))
    return SimpleNamespace(root=root, repo=repo, cfg=cfg, plan=plan, data=data,
        worker=worker, calls=calls, state=state, out=tmp_path / 'evaluation')


def cache_hpeli(env):
    model = env.worker.build_model('hpeli', 'cpu')
    predictions = native.infer_grid(model, env.data, Trigger(), 'cpu')
    truth = env.data.truth
    reference = np.repeat(truth[:, None], 6, axis=1)
    folder = env.worker.cell(env.plan, 'hpeli', 'backtime')
    arrays = folder / 'predictions.npz'
    np.savez_compressed(arrays, truth=truth, predictions=predictions,
        targets=np.stack([adapter.target_pose(truth, d) for d in side.GRID], axis=1),
        names=np.asarray([r['name'] for r in env.data.items]), doses=side.GRID)
    report = dict(profile=side.NATIVE_PROFILE, status='COMPLETE', backbone='hpeli',
        plan_sha256=env.plan['plan_sha256'],
        metrics=adapter.measure_predictions(truth, predictions[:, 0], predictions, truth, reference),
        artifacts={'checkpoint.pt': side.sha(folder / 'checkpoint.pt'),
            'predictions.npz': side.sha(arrays), 'trigger.pt': side.sha(env.root / 'trigger.pt')})
    side.write_new(folder / 'evaluation.json', report)
    return folder / 'evaluation.json'


def test_real_cpu_inference_three_outputs_no_clean_or_training(environment, monkeypatch):
    env = environment
    cache_hpeli(env)
    before = {str(p): side.sha(p) for p in env.root.rglob('*') if p.is_file()}
    monkeypatch.setattr(side, 'load_worker', lambda repo: env.worker)
    side.main(['--run', str(env.root), '--source-repo', str(env.repo), '--outdir', str(env.out)])
    assert env.calls['inference'] == ['wiflow', 'metafiplusplus']
    assert env.calls['trained'] == 1 and env.calls['load_plan'] == 2
    assert before == {str(p): side.sha(p) for p in env.root.rglob('*') if p.is_file()}
    summary = side.read(env.out / 'summary.json')
    assert summary['status'] == 'THREE_VICTIM_METRICS_COMPLETE_NO_MATCHED_CLEAN'
    assert [r['backbone'] for r in summary['results']] == list(side.BACKBONES)
    assert len((env.out / 'victim_results.csv').read_text().splitlines()) == 4
    assert not list(env.out.rglob('*.pt'))
    for b in side.BACKBONES:
        report = side.read(env.out / b / 'evaluation.json')
        assert report['clean_model_loaded'] is False and report['model_training_started'] is False
        assert report['final_epoch'] == 199
        assert len(report['metrics']['dose_rows']) == 6


@pytest.mark.parametrize('bad', ['binding', 'metrics', 'arrays'])
def test_existing_cache_tampering_never_silently_reruns(environment, bad):
    env = environment
    path = cache_hpeli(env)
    report = side.read(path)
    if bad == 'binding': report['plan_sha256'] = 'wrong'
    if bad == 'metrics': report['metrics']['mainmetrics']['mean_dce_mm'] += 1
    if bad == 'arrays': report['artifacts']['predictions.npz'] = 'wrong'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError): side.evaluate(env.worker, env.plan, env.out)
    assert env.calls['inference'] == []
    assert not (env.out / 'summary.json').exists()


def test_bn_or_trigger_mutation_refuses_result(environment):
    env = environment
    def mutated(model, data, trigger, device):
        result = native.infer_grid(model, data, trigger, device)
        model.counter += 1
        return result
    env.worker.infer_grid = mutated
    with pytest.raises(ValueError, match='mutated'):
        side.evaluate(env.worker, env.plan, env.out)
    assert not (env.out / 'summary.json').exists()


@pytest.mark.parametrize('location', ['run', 'repo', 'dataset', 'ancestor', 'existing'])
def test_output_cannot_overwrite_original_or_existing(environment, location):
    env = environment
    paths = dict(run=env.root / 'evaluation', repo=env.repo / 'evaluation',
        dataset=Path(env.cfg['dataset_root']) / 'evaluation', ancestor=env.root.parent,
        existing=env.root)
    with pytest.raises(ValueError):
        side.validate_output(paths[location], env.repo, env.root, env.cfg['dataset_root'])


def test_write_new_refuses_replacement(tmp_path):
    path = tmp_path / 'existing.json'
    side.write_new(path, {'keep': True})
    with pytest.raises(FileExistsError): side.write_new(path, {'replace': True})
    assert side.read(path) == {'keep': True}


def test_native_final_gate_failure_precedes_any_output(environment):
    env = environment
    def unfinished(plan): raise ValueError('not final199')
    env.worker.require_trained = unfinished
    with pytest.raises(ValueError, match='final199'): side.evaluate(env.worker, env.plan, env.out)
    assert not env.out.exists() and env.calls['inference'] == []
