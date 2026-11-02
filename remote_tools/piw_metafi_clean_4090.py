"""Separate Clean-only 4090 diagnostic; immutable 3090 replay, then fresh training.

Scientific functions and the historical 3090 launcher are not edited. An
inference-only replay never migrates optimizer/RNG/execution metadata, never
becomes a training checkpoint, and is not an efficacy/reproduction pass gate.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import csv
from functools import lru_cache
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROFILE = 'piw_metafi_clean_vastenv_s42_4090_v1'
SOURCE_PROFILE = 'piw_metafi_clean_vastenv_s42_3090_v1'
EXPORT_PROFILE = 'piw_metafi_clean_3090_reference_v1'
GPU = 'NVIDIA GeForce RTX 4090'
CELL = Path('attacks/metafiplusplus/seed42/clean')
EXPORT_FILES = {
    'plan.json': 'plan.json',
    'preflight_receipt.json': 'preflight_receipt.json',
    'prelaunch_gate.json': 'prelaunch_gate.json',
    'comparison.json': 'comparison.json',
    'smoke_comparison.json': 'smoke_comparison.json',
    'checkpoint.pt': str(CELL / 'checkpoint.pt'),
    'evaluation.json': str(CELL / 'evaluation.json'),
    'predictions.npz': str(CELL / 'predictions.npz'),
}


def source_probe():
    spec = importlib.util.spec_from_file_location(
        '_unchanged_clean_3090_for_4090', HERE / 'piw_metafi_clean_vastenv_3090.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execution_4090(w):
    execution = w._execution('cuda:0')
    if execution.get('gpu_name') != GPU:
        raise ValueError('This separate diagnostic requires exactly RTX 4090 on cuda:0')
    return execution


@lru_cache(maxsize=1)
def platform_metadata():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version',
        '--format=csv,noheader,nounits'], capture_output=True, text=True,
        check=True, timeout=30)
    gpus = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 3 or not all(v.strip() for v in row):
            raise ValueError('Cannot record NVIDIA GPU/driver metadata')
        gpus.append(dict(index=row[0].strip(), name=row[1].strip(), driver_version=row[2].strip()))
    if not gpus:
        raise ValueError('NVIDIA GPU/driver metadata is empty')
    return dict(system=platform.system(), release=platform.release(),
                os_version=platform.version(), nvidia_gpus=gpus)


def fresh_probe(reference_dir=None):
    """Reuse unchanged native recipe, validation, smoke and train implementation."""
    probe = source_probe()
    probe.PROFILE = PROFILE
    probe.execution_for = execution_4090
    original_binding = probe.code_binding
    probe.code_binding = lambda base: {
        **original_binding(base), 'sidecar_4090_sha256': base.sha(__file__),
        'helper_4090_sha256': base.sha(HERE / 'Start-PiWMetaFiClean4090.ps1'),
    }
    original_gate = probe.expected_gate

    def expected_gate(*args):
        result = original_gate(*args)
        result['sidecar_4090_platform'] = platform_metadata()
        return result

    probe.expected_gate = expected_gate
    original_launch = probe.launch

    def launch(args, base, w, cfg):
        if reference_dir is None:
            raise ValueError('Replay3090 must complete before fresh Clean Launch')
        validate_replay(args, base, w, w.load_plan(cfg), reference_dir)
        return original_launch(args, base, w, cfg)

    probe.launch = launch
    probe.comparison = comparison_4090
    return probe


def assert_separate(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError('Reference/export and experiment outputs must not overlap')


def reference_metadata(base, plan, evaluation, gate, comparison):
    """Pure metadata check; no tensor deserialization or GPU use."""
    if (plan.get('profile') != SOURCE_PROFILE or
            plan.get('plan_sha256') != base.digest({k: v for k, v in plan.items()
                                                   if k != 'plan_sha256'})):
        raise ValueError('Foreign or corrupt source 3090 plan')
    source_probe().validate_plan(plan, base.original_plan())
    cfg = plan['cfg']
    if (cfg.get('methods') != ['clean'] or cfg.get('backbones') != ['metafiplusplus']
            or cfg.get('test_checkpoint_selection') is not False):
        raise ValueError('Only the completed Clean-only 3090 diagnostic may be exported')
    if (evaluation.get('status') != 'COMPLETE' or evaluation.get('profile') != SOURCE_PROFILE
            or evaluation.get('method') != 'clean' or evaluation.get('backbone') != 'metafiplusplus'
            or evaluation.get('seed') != 42 or evaluation.get('plan_sha256') != plan['plan_sha256']):
        raise ValueError('Source Clean evaluation does not match its plan')
    if (gate.get('profile') != SOURCE_PROFILE or gate.get('plan_sha256') != plan['plan_sha256']
            or gate.get('fresh_from_epoch0') is not True or gate.get('checkpoint_import') is not False
            or gate.get('runtime') != plan['runtime']
            or gate.get('execution', {}).get('gpu_name') != 'NVIDIA GeForce RTX 3090'):
        raise ValueError('Source fresh-training 3090 gate does not match')
    if (comparison.get('profile') != SOURCE_PROFILE or comparison.get('status') != 'COMPLETE'
            or comparison.get('plan_sha256') != plan['plan_sha256']
            or comparison.get('checkpoint_sha256') != evaluation['checkpoint_sha256']
            or comparison.get('predictions_sha256') != evaluation['predictions_sha256']
            or comparison.get('runtime') != plan['runtime'] or comparison.get('code') != gate.get('code')):
        raise ValueError('Source completed comparison does not match')
    metrics = evaluation.get('clean_metrics', {})
    for name in ('clean_mpjpe_mm', 'clean_pampjpe_mm'):
        v = metrics.get(name)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError('Source Clean metric is invalid')


def read_reference(folder):
    probe = source_probe()
    base = probe.base_module()
    folder = Path(folder)
    probe.reject_linked_path(folder)
    manifest_path = folder / 'reference_manifest.json'
    manifest = base.read(manifest_path)
    if (manifest.get('profile') != EXPORT_PROFILE or manifest.get('schema') != 1
            or set(manifest.get('files', {})) != set(EXPORT_FILES)
            or manifest.get('manifest_sha256') != base.digest(
                {k: v for k, v in manifest.items() if k != 'manifest_sha256'})):
        raise ValueError('Reference manifest is foreign, incomplete or corrupt')
    for name, record in manifest['files'].items():
        path = folder / name
        probe.reject_linked_path(path)
        if (not path.is_file() or path.stat().st_size != record.get('bytes')
                or base.sha(path) != record.get('sha256')):
            raise ValueError(f'Reference file checksum/size differs: {name}')
    plan, evaluation, gate, comparison = [base.read(folder / n) for n in
        ('plan.json', 'evaluation.json', 'prelaunch_gate.json', 'comparison.json')]
    reference_metadata(base, plan, evaluation, gate, comparison)
    receipt = base.read(folder / 'preflight_receipt.json')
    if (receipt.get('status') != 'PASS' or receipt.get('plan_sha256') != plan['plan_sha256']
            or receipt.get('input_fingerprint') != plan['inputs']['fingerprint']
            or receipt.get('n_train') != 27876 or receipt.get('n_test') != 2586):
        raise ValueError('Reference full-input preflight receipt does not match')
    if (base.sha(folder / 'checkpoint.pt') != evaluation['checkpoint_sha256'] or
            base.sha(folder / 'predictions.npz') != evaluation['predictions_sha256']):
        raise ValueError('Reference evaluation artifact hashes differ')
    return dict(folder=folder, manifest=manifest, plan=plan, evaluation=evaluation,
                manifest_sha256=base.sha(manifest_path))


def export_reference(args):
    """Copy exact original bytes; no torch import, pickle load, or source writes."""
    probe = source_probe()
    base = probe.base_module()
    root = args.data_home / 'runs' / SOURCE_PROFILE
    dest = args.export_dir
    for p in (root, dest):
        probe.reject_linked_path(p)
    assert_separate(root, dest)
    if dest.parent != args.data_home / 'exports' or dest.name != EXPORT_PROFILE:
        raise ValueError('Export is restricted to the dedicated DataHome/exports child')
    if dest.exists():
        reference = read_reference(dest)
        print('EXISTING_VERIFIED_REFERENCE ' + reference['manifest_sha256'], flush=True)
        return
    inputs = {name: root / relative for name, relative in EXPORT_FILES.items()}
    for path in inputs.values():
        probe.reject_linked_path(path)
        if not path.is_file():
            raise ValueError(f'Missing completed source artifact: {path}')
    plan, evaluation, gate, comparison = [base.read(inputs[n]) for n in
        ('plan.json', 'evaluation.json', 'prelaunch_gate.json', 'comparison.json')]
    reference_metadata(base, plan, evaluation, gate, comparison)
    records = {n: {'bytes': p.stat().st_size, 'sha256': base.sha(p)}
               for n, p in inputs.items()}
    if (records['checkpoint.pt']['sha256'] != evaluation['checkpoint_sha256'] or
            records['predictions.npz']['sha256'] != evaluation['predictions_sha256']):
        raise ValueError('Source checkpoint/predictions differ from completed evaluation')
    if shutil.disk_usage(args.data_home).free < sum(r['bytes'] for r in records.values()) + 64 * 1024**2:
        raise ValueError('Not enough disk for exact reference copy; nothing removed')
    dest.mkdir(parents=True, exist_ok=False)
    for name, path in inputs.items():
        shutil.copy2(path, dest / name)
        if base.sha(path) != records[name]['sha256'] or base.sha(dest / name) != records[name]['sha256']:
            raise ValueError('Source changed or copy differs; partial export retained, no retry')
    manifest = dict(schema=1, profile=EXPORT_PROFILE, source_run=str(root),
                    files=records, plan_sha256=plan['plan_sha256'],
                    note='Exact trusted local checkpoint bytes; inference-only reference, never a training resume.')
    manifest['manifest_sha256'] = base.digest(manifest)
    base.write(dest / 'reference_manifest.json', manifest)
    reference = read_reference(dest)
    print('EXPORTED_REFERENCE=' + str(dest), flush=True)
    print('MANIFEST_SHA256=' + reference['manifest_sha256'], flush=True)
    print('No training, GPU use, checkpoint conversion, or remote upload.', flush=True)


def validate_source_blob(base, blob, reference):
    base.validate_blob(blob, reference['plan'], 'clean')
    if (blob['epoch'] != 199 or blob['execution'].get('gpu_name') != 'NVIDIA GeForce RTX 3090'
            or blob['poison'].get('n_poison') != 0 or blob['poison'].get('n_cover', 0) != 0):
        raise ValueError('Only the completed unpoisoned 3090 Clean victim can be replayed')


def clean_metrics(pred, truth, pa_mpjpe):
    import numpy as np
    if pred.shape != truth.shape or pred.shape != (2586, 14, 3):
        raise ValueError('Full TEST must have exactly 2586 x 14 x 3 predictions/truth')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('Nonfinite full TEST predictions/truth')
    return dict(clean_mpjpe_mm=float(np.linalg.norm(pred-truth, axis=-1).mean()*1000),
                clean_pampjpe_mm=float(pa_mpjpe(pred, truth).mean()*1000))


def replay_binding(args, base, w, saved, reference):
    return dict(profile=PROFILE, plan_sha256=saved['plan_sha256'],
                reference_manifest_sha256=reference['manifest_sha256'],
                source_checkpoint_sha256=reference['evaluation']['checkpoint_sha256'],
                code=fresh_probe().code_binding(base), execution=execution_4090(w),
                platform=platform_metadata())


def validate_replay(args, base, w, saved, reference_dir):
    reference = read_reference(reference_dir)
    assert_separate(args.out, reference_dir)
    result = base.read(args.out / 'replay_3090' / 'evaluation.json')
    if (result.get('status') != 'COMPLETE' or result.get('inference_only') is not True
            or result.get('n_test') != 2586 or result.get('training_started') is not False
            or result.get('binding') != replay_binding(args, base, w, saved, reference)
            or result.get('predictions_sha256') != base.sha(args.out / 'replay_3090/predictions.npz')):
        raise ValueError('Run Replay3090 with this exact reference/plan/GPU before Launch')
    return reference, result


def replay_3090(args, probe):
    import numpy as np
    import torch

    probe.validate_paths(args)
    base = probe.base_module()
    base.configure_torch_cache(args)
    assert_separate(args.out, args.reference_dir)
    with ExitStack() as stack:
        stack.enter_context(base.lock(args.data_home / 'tools/clean_vastenv_gpu0_lock'))
        stack.enter_context(base.lock(args.out))
        stack.enter_context(probe.run_log(args))
        base.require_idle()
        w = probe.clean_worker(args)
        cfg = probe.cfg_for(w, args)
        saved = w.load_plan(cfg)
        probe.check_gate(args, base, w, cfg, saved)
        reference = read_reference(args.reference_dir)
        source = reference['plan']
        if (saved['inputs']['splits'] != source['inputs']['splits'] or
                saved['sources'] != source['sources'] or
                saved['action_sha256'] != source['action_sha256']):
            raise ValueError('4090 audited data/order/scientific sources differ from 3090')
        output = args.out / 'replay_3090'
        if (output / 'evaluation.json').exists():
            _, result = validate_replay(args, base, w, saved, args.reference_dir)
            print(json.dumps(result, indent=2), flush=True)
            return
        if output.exists():
            raise ValueError('Partial replay output exists; inspect it, do not overwrite or retry')
        # Check all bindings before loading trusted pickle or creating a CUDA model.
        path = args.reference_dir / 'checkpoint.pt'
        blob = torch.load(path, map_location='cpu', weights_only=False)
        validate_source_blob(base, blob, reference)
        original_execution = copy.deepcopy(blob['execution'])
        w._seed('cuda:0')
        data = w._datasets(saved)
        x, truth, names = w._test_arrays(data['validation'])
        with np.load(args.reference_dir / 'predictions.npz', allow_pickle=False) as cache:
            old_truth = cache['truth'].copy()
            old_pred = cache['predictions'].copy()
            old_names = cache['names'].tolist()
        if names != old_names or not np.array_equal(truth, old_truth):
            raise ValueError('Reference ordered TEST names/truth do not match exactly')
        from eval.metrics import pa_mpjpe
        checked_source_metrics = clean_metrics(old_pred, truth, pa_mpjpe)
        if any(abs(checked_source_metrics[k] - reference['evaluation']['clean_metrics'][k]) > .001
               for k in checked_source_metrics):
            raise ValueError('Saved source arrays do not reproduce source metrics within 0.001 mm')
        model = w._build('metafiplusplus', 'cuda:0')
        model.load_state_dict(blob['model'], strict=True)
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pred = w._predict_grid(model, x, None, [0.], 'cuda:0')[:, 0]
        metrics = clean_metrics(pred, truth, pa_mpjpe)
        if any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()):
            raise ValueError('Inference changed model/BN state; result rejected')
        if original_execution != blob['execution'] or base.sha(path) != reference['evaluation']['checkpoint_sha256']:
            raise ValueError('Reference checkpoint/execution metadata changed')
        result = dict(status='COMPLETE', inference_only=True, training_started=False,
            n_test=len(names), binding=replay_binding(args, base, w, saved, reference),
            source_execution=original_execution, source_metrics=checked_source_metrics,
            replay_metrics=metrics,
            metric_difference_mm={k: metrics[k]-checked_source_metrics[k] for k in metrics},
            mean_prediction_difference_mm=float(np.linalg.norm(pred-old_pred, axis=-1).mean()*1000),
            max_coordinate_difference_mm=float(np.abs(pred-old_pred).max()*1000),
            model_and_bn_unchanged=True,
            interpretation='Measurement only; COMPLETE is not a similarity threshold or final-training reproducibility claim.')
        output.mkdir(exist_ok=False)
        w._atomic_npz(output / 'predictions.npz', truth=truth, names=np.asarray(names), predictions=pred)
        result['predictions_sha256'] = base.sha(output / 'predictions.npz')
        base.write(output / 'evaluation.json', result)
        print(json.dumps(result, indent=2), flush=True)
        print('REPLAY_ONLY_COMPLETE; no fresh training launched. Inspect, then invoke Launch separately.', flush=True)


def comparison_4090(args, base, w, cfg, saved):
    folder = w.cell_dir(cfg, 'metafiplusplus', 'clean')
    result = base.read(folder / 'evaluation.json')
    blob = w._checkpoint(saved, 'metafiplusplus', 'clean', final=True)
    if (result.get('status') != 'COMPLETE' or result.get('profile') != PROFILE
            or result.get('method') != 'clean' or result.get('plan_sha256') != saved['plan_sha256']
            or result.get('checkpoint_sha256') != base.sha(folder / 'checkpoint.pt')
            or result.get('predictions_sha256') != base.sha(folder / 'predictions.npz')
            or blob['epoch'] != 199):
        raise ValueError('Fresh 4090 final199 evaluation artifact binding differs')
    replay = base.read(args.out / 'replay_3090/evaluation.json')
    probe = fresh_probe()
    rows = [dict(name='Vast historical reference', **probe.OLD_CLEAN),
            dict(name='3090 source checkpoint', **replay['source_metrics']),
            dict(name='Same 3090 checkpoint evaluated on 4090', **replay['replay_metrics']),
            dict(name='4090 fresh Clean final199', **result['clean_metrics'])]
    report = dict(profile=PROFILE, status='COMPLETE', rows=rows,
                  plan_sha256=saved['plan_sha256'], runtime=saved['runtime'],
                  execution=execution_4090(w), code=probe.code_binding(base),
                  platform=platform_metadata(),
                  checkpoint_sha256=result['checkpoint_sha256'],
                  predictions_sha256=result['predictions_sha256'],
                  replay_binding=replay['binding'],
                  limitations=['One seed; this is a platform diagnostic, not an isolated GPU causal experiment.',
                               'Driver/OS/backend differences are recorded, not assumed identical.',
                               'Reference replay never initializes or resumes the fresh training victim.'])
    text = '# Clean-only 4090 platform diagnostic\n\n'
    text += '| Run | Clean MPJPE mm | Clean PA-MPJPE mm |\n| --- | ---: | ---: |\n'
    text += '\n'.join(f"| {r['name']} | {r['clean_mpjpe_mm']:.6f} | {r['clean_pampjpe_mm']:.6f} |" for r in rows)
    text += '\n\nOne seed; fixed final199, no TEST-selected checkpoint or hyperparameter change.\n'
    for path, value in ((args.out / 'comparison.json', report), (args.out / 'comparison.md', text)):
        if path.exists():
            existing = base.read(path) if path.suffix == '.json' else path.read_text(encoding='utf-8')
            if existing != value:
                raise ValueError('Existing comparison differs; no overwrite permitted')
    if not (args.out / 'comparison.json').exists():
        base.write(args.out / 'comparison.json', report)
    if not (args.out / 'comparison.md').exists():
        with (args.out / 'comparison.md').open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inspect', 'prepare', 'check', 'launch', 'status', 'export3090', 'replay3090'), default='inspect')
    parser.add_argument('--data-home', type=Path, default=Path.home() / 'backdooranalog')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--action', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--env-dir', type=Path)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--export-dir', type=Path)
    args = parser.parse_args(argv)
    args.data_home = args.data_home.resolve()
    args.out = args.out or args.data_home / 'runs' / PROFILE
    args.dataset = args.dataset or args.data_home / 'datasets/Person-in-WiFi-3D_processed_v1'
    args.action = args.action or args.data_home / 'actions/data_bend.npy'
    args.env_dir = args.env_dir or args.data_home / 'envs/piw-metafi-vast211-4090'
    args.reference_dir = args.reference_dir or args.data_home / 'imports' / EXPORT_PROFILE
    args.export_dir = args.export_dir or args.data_home / 'exports' / EXPORT_PROFILE
    for name in ('out', 'dataset', 'action', 'env_dir', 'reference_dir', 'export_dir'):
        path = getattr(args, name)
        source_probe().reject_linked_path(path)
        setattr(args, name, path.resolve())
    if args.mode == 'export3090':
        export_reference(args)
        return 0
    probe = fresh_probe(args.reference_dir)
    if args.mode == 'replay3090':
        os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                          OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MKL_THREADING_LAYER='SEQUENTIAL')
        if os.environ.get('PIW_SMOKE_BARRIER_DIR'):
            raise ValueError('Foreign parallel smoke barrier is set')
        replay_3090(args, probe)
        return 0
    forwarded = ['--mode', args.mode]
    for option, value in (('data-home', args.data_home), ('out', args.out),
                          ('dataset', args.dataset), ('action', args.action), ('env-dir', args.env_dir)):
        forwarded.extend(['--'+option, str(value)])
    return probe.main(forwarded)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'[Clean 4090 diagnostic] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
