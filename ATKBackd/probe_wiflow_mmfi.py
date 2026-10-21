"""Source audit / synthetic compute probe; NOT a MM-Fi training/evaluation run.

Default is CPU, one model at a time, random inputs/poses. No raw dataset,
checkpoint, pretrained weights or historical output is read/written. No
optimizer step, AMP, model shrinking, automatic retry or timing-to-epoch claim.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import time
from urllib.request import Request, urlopen

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

HERE = Path(__file__).resolve().parent
PASSPORT = HERE / 'models/wiflow_sources.json'
MODELS = ('hpeli', 'metafiplusplus', 'wiflow')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_local_sources():
    passport = json.loads(PASSPORT.read_text(encoding='utf-8'))
    for name, expected in passport['local_sha256'].items():
        if digest(HERE / name) != expected:
            raise ValueError(f'WiFlow pinned local source differs: {name}')
    return passport


def definition_ast(text):
    return ast.dump(ast.Module(body=[node for node in ast.parse(text).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))], type_ignores=[]),
        include_attributes=False)


def audit_upstream():
    """Download only pinned public text; parse AST, never execute its module."""
    passport = verify_local_sources()
    request = Request(passport['source_url'], headers={'User-Agent': 'WiFlow-source-audit'})
    with urlopen(request, timeout=30) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != passport['source_sha256']:
        raise ValueError('Pinned public source SHA256 differs; no source execution')
    source = raw.decode('utf-8-sig')
    start = source.index('class ConvBlock1D(')
    end = source.index('def percentage_correct_keypoints(')
    local = (HERE / 'models/_vendor/wiflow_mmfi.py').read_text(encoding='utf-8')
    if definition_ast(source[start:end]) != definition_ast(local):
        raise ValueError('Vendored model/loss AST differs from the pinned author definitions')
    return dict(status='PINNED_AUTHOR_MODEL_AND_LOSS_AST_MATCH', commit=passport['commit'],
                source_sha256=passport['source_sha256'], source_url=passport['source_url'])


def build(name):
    if name == 'hpeli':
        from models.factory import build_model
        return build_model('hpeli', num_keypoints=17, num_coor=3,
                           num_person=1, subcarrier_num=114, dataset='mmfi')
    if name == 'metafiplusplus':
        from models.cross_backbone_factory import build_backbone
        return build_backbone(name)
    if name == 'wiflow':
        from models.wiflow_mmfi import WiFlowMMFiNet
        return WiFlowMMFiNet()
    raise ValueError(f'Unknown probe model: {name}')


def probe_model(name, device, batch_size, warmup, steps, seed):
    import torch
    torch.manual_seed(seed)
    model = build(name).to(device).train()
    parameters = sum(p.numel() for p in model.parameters())
    # Identical synthetic values for each model, independent of its init RNG.
    generator = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(batch_size, 3, 114, 10, generator=generator).to(device)
    target = torch.randn(batch_size, 1, 17, 3, generator=generator).to(device)
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
    def step(validate=False):
        model.zero_grad(set_to_none=True)
        pose, feature = model(x)
        if pose.shape != target.shape:
            raise ValueError('Invalid synthetic pose/feature output')
        loss = torch.linalg.vector_norm(pose - target, dim=-1).mean()
        if validate and (not torch.isfinite(loss) or not torch.isfinite(feature).all()):
            raise ValueError('Nonfinite synthetic MPJPE; no auto-retry')
        loss.backward()
        if validate and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite synthetic gradient; no auto-retry')
        return tuple(pose.shape), tuple(feature.shape)
    # Keep per-parameter finite checks (especially CUDA synchronizations) out
    # of the timed region. One validation pass is separate from warmup/timing.
    step(validate=True)
    for _ in range(warmup):
        step()
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(steps):
        if cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        pose_shape, feature_shape = step()
        if cuda:
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Nonfinite final gradient; no report accepted')
    result = dict(model=name, parameters=parameters, batch_size=batch_size,
        pose_shape=list(pose_shape), feature_shape=list(feature_shape),
        finite_forward_and_gradients=True, dtype='float32', amp=False,
        untimed_validation_passes=1,
        median_forward_backward_seconds=statistics.median(samples),
        seconds_per_step=samples, cuda_peak_allocated_mib=peak)
    model.zero_grad(set_to_none=True)
    del model, x, target
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()  # Only this probe process's unused allocations.
    return result


def write_report(path, report):
    path = Path(path).resolve()
    if path.suffix.lower() != '.json':
        raise ValueError('Probe report must use a new .json path')
    protected = ('cross_backbones.resolved.json', 'three_tables.resolved.json',
                 'a1_defenses.resolved.json', 'a1_defenses_v2.resolved.json', 'checkpoint.pt')
    if path.exists() or any(any((parent / name).exists() for name in protected)
                            for parent in path.parents):
        raise ValueError('Refuse existing report or historical experiment output')
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-upstream', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if (args.batch_size < 1 or args.warmup < 0 or args.steps < 1 or args.threads < 1
            or len(set(args.models)) != len(args.models)
            or (args.device != 'cpu' and not re.fullmatch(r'cuda:(0|[1-9][0-9]*)', args.device))):
        parser.error('Positive batch/steps/threads, nonnegative warmup and distinct models; device cpu or cuda:N')
    passport = verify_local_sources()
    if args.audit_upstream:
        report = audit_upstream()
    else:
        import torch
        torch.set_num_threads(args.threads)
        if args.device != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; no CPU fallback')
        report = dict(status='SYNTHETIC_COMPUTE_PROBE_NOT_PAPER_RESULTS',
            device=args.device, device_name=(torch.cuda.get_device_name(torch.device(args.device))
                if args.device != 'cpu' else platform.processor()),
            python=platform.python_version(), runtime={k: version(k) for k in ('torch', 'torchvision', 'numpy')},
            seed=args.seed, threads=args.threads, warmup=args.warmup, steps=args.steps,
            objective='Common synthetic MPJPE forward/backward only; no optimizer update or author recipe claim',
            no_raw_data=True, no_pretrained_checkpoint=True, no_model_accuracy=True,
            wiflow_commit=passport['commit'], sources={**passport['local_sha256'],
                'probe_wiflow_mmfi.py': digest(__file__), 'models/hpeli.py': digest(HERE / 'models/hpeli.py'),
                'models/sk_network.py': digest(HERE / 'models/sk_network.py'),
                'models/metafi_plus_plus.py': digest(HERE / 'models/metafi_plus_plus.py'),
                'models/metafi_channel_transformer.py': digest(HERE / 'models/metafi_channel_transformer.py')},
            results=[])
        for name in args.models:
            print(f'[synthetic probe] {name}: forward/backward on {args.device}', flush=True)
            report['results'].append(probe_model(name, args.device, args.batch_size,
                args.warmup, args.steps, args.seed))
    if args.output:
        write_report(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    main()
