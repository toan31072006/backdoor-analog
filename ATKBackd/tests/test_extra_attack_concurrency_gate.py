"""The four-job launch gate must reject a partial or mixed GPU smoke."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parents[2] / 'remote_linux/vast_seed42/extra_attacks_runner.py'
spec = importlib.util.spec_from_file_location('extra_attacks_runner_gate_test', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _attempt(tmp_path):
    attempt = 'a' * 32
    cells = [dict(id=f'{backbone}__s42__{method}', device=f'cuda:{gpu}',
                  config=dict(batch_size=32, optimizer='sgd' if gpu == 0 else 'adamw'))
             for method in ('smooth_lf_csi', 'tsba_adapted')
             for gpu, backbone in enumerate(('metafiplusplus', 'wiflow'))]
    plan = dict(outdir=str(tmp_path), plan_sha256='plan', cells=cells)
    gpus = [dict(index=gpu, name='GPU', total_memory=32 * 2**30)
            for gpu in range(2)]
    folder = runner.smoke_attempt_dir(plan, attempt)
    for cell in cells:
        marker = dict(cell=cell['id'], attempt=attempt,
                      plan_sha256='plan', device=cell['device'], pid=123)
        for phase in ('ready', 'done'):
            path = folder / phase / (cell['id'] + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(marker), encoding='utf-8')
        receipt = dict(cell=cell['id'], finite=True, attempt=attempt,
                       concurrent_workers=4, gpu=gpus[int(cell['device'][-1])],
                       physical_batch=32, optimizer=cell['config']['optimizer'],
                       loss=1.0, peak_allocated_bytes=1024)
        (folder / (cell['id'] + '.json')).write_text(json.dumps(receipt), encoding='utf-8')
    return plan, attempt, gpus, folder


def test_four_worker_gate_requires_both_barriers(tmp_path):
    plan, attempt, gpus, folder = _attempt(tmp_path)
    valid = runner.gate_payload(plan, attempt, gpus, 'dataset')
    assert valid['concurrency']['jobs_per_gpu'] == 2
    assert len(valid['barrier_sha256']['done']) == 4

    (folder / 'done' / (plan['cells'][-1]['id'] + '.json')).unlink()
    with pytest.raises(FileNotFoundError):
        runner.gate_payload(plan, attempt, gpus, 'dataset')


def test_four_worker_gate_rejects_mixed_attempt(tmp_path):
    plan, attempt, gpus, folder = _attempt(tmp_path)
    marker = folder / 'ready' / (plan['cells'][0]['id'] + '.json')
    changed = json.loads(marker.read_text(encoding='utf-8'))
    changed['attempt'] = 'b' * 32
    marker.write_text(json.dumps(changed), encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid ready smoke marker'):
        runner.gate_payload(plan, attempt, gpus, 'dataset')
