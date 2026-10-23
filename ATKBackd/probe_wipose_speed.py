"""Bounded WiPose timing diagnostics, never a training launcher or resume tool.

The default is a metadata-only plan. IO reads a small selected TRAIN subset.
Compute uses disposable synthetic batches/models, compares the current finite-
gradient guard with an on-device aggregate, and never opens a run/checkpoint.
Nothing here changes the frozen WiPose implementation or proves a GPU speedup.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


PROFILE = 'wipose_speed_diagnostic_v1_not_a_paper_experiment'
HERE = Path(__file__).resolve().parent


def gradients_finite_legacy(parameters):
    """Current runner semantics: a host decision for each populated gradient."""
    import torch
    return not any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in parameters)


def gradients_finite_grouped(parameters):
    """Same predicate, but read one device-side aggregate on the host."""
    import torch
    flags = [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
    return True if not flags else bool(torch.stack(flags).all())


def checked_step(model, optimizer, x, y, mode='legacy'):
    import torch
    checks = {'legacy': gradients_finite_legacy, 'grouped': gradients_finite_grouped}
    if mode not in checks:
        raise ValueError('Unknown finite-gradient check mode')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = model(x)
    if prediction.shape != y.shape:
        raise ValueError('WiPose training shape mismatch')
    loss = torch.linalg.vector_norm(prediction - y, dim=-1).mean()
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss; no optimizer update')
    loss.backward()
    if not checks[mode](model.parameters()):
        raise RuntimeError('Nonfinite gradient; no optimizer update')
    optimizer.step()
    return prediction.detach().clone(), float(loss.detach())


def _configure(device):
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if device != 'cpu' and not re.fullmatch(r'cuda:[0-9]+', device):
        raise ValueError('Use cpu or explicit cuda:N; no device fallback')
    if device.startswith('cuda:'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU fallback')
        torch.cuda.set_device(device)


def _rng(device):
    import torch
    state = {'cpu': torch.get_rng_state().clone()}
    if device.startswith('cuda:'):
        state['cuda'] = torch.cuda.get_rng_state(device).clone()
    return state


def _restore_rng(state, device):
    import torch
    torch.set_rng_state(state['cpu'])
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'], device)


def _cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _exact(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    return left == right


def _model_and_batches(device, batch_size, batches):
    import torch
    from models.wipose_hpeli import build_wipose_hpeli
    torch.manual_seed(42)
    if device.startswith('cuda:'):
        torch.cuda.manual_seed(42)
    model = build_wipose_hpeli().to(device)
    generator = torch.Generator(device='cpu').manual_seed(20261024)
    # CPU-generated common inputs: no dataset/TEST access, independent of timing mode.
    inputs = [(torch.rand(batch_size, 3, 90, 5, generator=generator).to(device),
               torch.rand(batch_size, 1, 18, 2, generator=generator).to(device))
              for _ in range(batches)]
    return model, inputs


def _optimizer(model):
    import torch
    return torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=0.)


def parity_check(device='cpu', batch_size=2, steps=2):
    """Compare active real-model math, not original-author initialization."""
    if not 1 <= steps <= 8 or not 2 <= batch_size <= 32:
        raise ValueError('Parity probe requires 1..8 steps and batch size 2..32')
    if device.startswith('cuda:'):
        _idle_gpu_check()
    _configure(device)
    original, inputs = _model_and_batches(device, batch_size, steps)
    initial_rng = _rng(device)
    records = []
    for mode in ('legacy', 'grouped'):
        model = copy.deepcopy(original)
        optimizer = _optimizer(model)
        _restore_rng(initial_rng, device)
        trace = []
        for x, y in inputs:
            prediction, loss = checked_step(model, optimizer, x, y, mode)
            trace.append(_cpu_copy(dict(prediction=prediction, loss=loss,
                gradients={name: p.grad for name, p in model.named_parameters()},
                model=model.state_dict(), optimizer=optimizer.state_dict(), rng=_rng(device))))
        records.append(trace)
    if not _exact(records[0], records[1]):
        raise RuntimeError('Finite-check candidate failed bitwise training-state parity')
    return dict(ok=True, bitwise=True, device=device, steps=steps, batch_size=batch_size,
        compared=['predictions', 'loss', 'gradients', 'parameters', 'BN_buffers',
                  'optimizer', 'Torch_CPU_CUDA_RNG'],
        scope='synthetic steps only; not full-data resume or scientific efficacy')


def _idle_gpu_check():
    """Reject visible compute clients; a snapshot is not an idle reservation.

    WDDM/process-reporting limitations can hide clients. The operator must also
    verify that the training tasks are idle before invoking the CUDA mode.
    """
    command = ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10,
                                check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('Cannot establish an idle GPU host; no compute probe launched') from exc
    if result.returncode:
        raise RuntimeError('nvidia-smi failed; no compute probe launched')
    others = []
    for row in result.stdout.splitlines():
        value = row.strip()
        if not value:
            continue
        if not value.isdigit():
            raise RuntimeError('Unrecognized GPU process listing; refusing compute probe')
        if int(value) != os.getpid():
            others.append(int(value))
    if others:
        raise RuntimeError(f'GPU compute clients still active: {others}; no probe/stop/retry')


def _sync(device):
    if device.startswith('cuda:'):
        import torch
        torch.cuda.synchronize(device)


def compute_probe(device='cpu', steps=20, warmup=2, rounds=2, max_seconds=60.):
    if not 1 <= steps <= 200 or not 0 <= warmup <= 10 or not 1 <= rounds <= 3:
        raise ValueError('Bounded probe: steps1..200, warmup0..10, rounds1..3')
    if not 1 <= max_seconds <= 120:
        raise ValueError('max_seconds must be 1..120 per timing case')
    if device != 'cpu' and not re.fullmatch(r'cuda:[0-9]+', device):
        raise ValueError('Use cpu or explicit cuda:N; no device fallback')
    if device.startswith('cuda:'):
        _idle_gpu_check()
    _configure(device)
    parity = parity_check(device, batch_size=2, steps=2)
    original, inputs = _model_and_batches(device, 32, 2)
    initial_rng = _rng(device)
    timings = []
    for trial in range(rounds):
        order = ('legacy', 'grouped') if trial % 2 == 0 else ('grouped', 'legacy')
        for mode in order:
            model = copy.deepcopy(original)
            optimizer = _optimizer(model)
            _restore_rng(initial_rng, device)
            for index in range(warmup):
                checked_step(model, optimizer, *inputs[index % len(inputs)], mode)
            _sync(device)
            began = time.perf_counter()
            for index in range(steps):
                checked_step(model, optimizer, *inputs[index % len(inputs)], mode)
                if time.perf_counter() - began > max_seconds:
                    raise RuntimeError('Timing case exceeded bound; incomplete, no retry')
            _sync(device)
            elapsed = time.perf_counter() - began
            if elapsed > max_seconds:
                raise RuntimeError('Timing case exceeded bound; incomplete, no retry')
            timings.append(dict(round=trial, mode=mode, steps=steps, seconds=elapsed,
                                seconds_per_batch=elapsed / steps))
    import torch
    return dict(profile=PROFILE, mode='compute', device=device, batch_size=32,
        parity=parity, timings=timings, python=platform.python_version(), torch=torch.__version__,
        interpretation='Synthetic disposable models; no NPZ/trigger/checkpoint/epoch timing. '
                       'CPU measurements cannot establish CUDA gains. No full-epoch ETA.',
        writes='No dataset, production run, manifest or checkpoint writes')


def io_probe(root, samples=128, seed=42):
    import numpy as np
    from wipose_data import WiPoseDataset, read_frame
    if not 1 <= samples <= 1024:
        raise ValueError('IO probe samples must be 1..1024')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('IO selection seed must be an unsigned 32-bit integer')
    dataset = WiPoseDataset(root, 'training')
    if samples > len(dataset):
        raise ValueError('Requested IO sample count exceeds retained TRAIN')
    indices = np.random.RandomState(seed).choice(len(dataset), size=samples, replace=False)
    selection = [dataset.items[int(index)] for index in indices]
    cache = {}
    began = time.perf_counter()
    for item in selection:
        pair = read_frame(item['path'])
        for value in pair:
            value.setflags(write=False)
        cache[item['name']] = pair
    preload = time.perf_counter() - began
    began = time.perf_counter()
    observed = [cache[item['name']] for item in selection]
    cached = time.perf_counter() - began
    # Exact re-read is outside both timers. Candidate caches final normalized FP32,
    # never recomputes normalization or modifies input files.
    for item, pair in zip(selection, observed):
        fresh = read_frame(item['path'])
        if any(a.dtype != b.dtype or not np.array_equal(a, b) for a, b in zip(pair, fresh)):
            raise RuntimeError('Input changed or cached output parity failed')
    names = [item['name'] for item in selection]
    return dict(profile=PROFILE, mode='io', samples=samples, seed=seed,
        retained_train=len(dataset), cache_coverage=samples / len(dataset),
        selected_names_sha256=hashlib.sha256(('\n'.join(names) + '\n').encode()).hexdigest(),
        cold_read_seconds=preload, cached_read_seconds=cached,
        cache_bytes=sum(value.nbytes for pair in cache.values() for value in pair),
        bitwise_frame_parity=True, cache_arrays_readonly=True,
        interpretation='First pass is not verified cold disk: OS cache/contention uncontrolled. '
                       'Bounded RAM lookup excludes collation/transfer/augmentation; '
                       'do not extrapolate it to a full shuffled epoch.',
        writes='No input, dataset, production run, manifest or checkpoint writes')


def _source_hashes():
    names = ('probe_wipose_speed.py', 'run_wipose_hpeli_parallel.py',
             'wipose_data.py', 'models/wipose_hpeli.py', 'models/sk_network.py')
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'io', 'compute'), default='plan')
    parser.add_argument('--dataset-root')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args(argv)
    if args.mode == 'io' and not args.dataset_root:
        parser.error('--dataset-root is required only for --mode io')
    if args.mode != 'io' and args.dataset_root:
        parser.error('--dataset-root is accepted only for --mode io')
    if not 1 <= args.samples <= 1024 or not 1 <= args.steps <= 200:
        parser.error('samples1..1024 and steps1..200 are required')
    if not 0 <= args.warmup <= 10 or not 1 <= args.rounds <= 3:
        parser.error('warmup0..10 and rounds1..3 are required')
    if args.device != 'cpu' and not re.fullmatch(r'cuda:[0-9]+', args.device):
        parser.error('Use cpu or explicit cuda:N')
    before_sources = _source_hashes()
    if args.mode == 'plan':
        report = dict(profile=PROFILE, mode='plan',
            status='METADATA_ONLY_NO_DATA_MODEL_CUDA_OR_TRAINING',
            io='Bounded TRAIN-only read; can compete for disk with active training',
            compute='Synthetic disposable models; CUDA refuses visible compute clients',
            production='Never changes runner, data, manifests, checkpoints or task state')
    elif args.mode == 'io':
        report = io_probe(args.dataset_root, samples=args.samples)
    else:
        report = compute_probe(args.device, args.steps, args.warmup, args.rounds)
    after_sources = _source_hashes()
    if before_sources != after_sources:
        raise RuntimeError('Source changed during probe; refusing to publish timings')
    report['sources_sha256'] = before_sources
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'[WiPose speed probe] ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
