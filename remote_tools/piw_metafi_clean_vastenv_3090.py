"""One fresh Clean-only PiW3D MetaFi++ library-matching diagnostic.

Uses the original frozen Vast scientific source. This is not an A1 run, a
full paper table, or a promise of Linux/Windows bitwise training equivalence.
The existing eclare environment and historical runs are never modified.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import traceback
import uuid

HERE = Path(__file__).resolve().parent
PROFILE = 'piw_metafi_clean_vastenv_s42_3090_v1'
SETUP_PROFILE = 'piw_metafi_clean_vastenv_setup_v1'
EXPECTED_PYTHON = '3.12.15'
EXPECTED_PACKAGES = {'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128',
                     'numpy': '2.5.4', 'scipy': '1.17.1'}
RECIPE = {'optimizer': 'adamw', 'lr': .01, 'weight_decay': .01,
          'accumulation': 1, 'effective_batch': 32}
WEIGHTS_SHA = 'b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f'
OLD_CLEAN = {'clean_mpjpe_mm': 136.16445922851562,
             'clean_pampjpe_mm': 76.86581054870948}
PREVIOUS_3090 = {'clean_mpjpe_mm': 200.1260771485875,
                 'clean_pampjpe_mm': 90.89817437952546}
OLD_SMOKE_LOSS = 3.8719141483306885
OLD_SMOKE = {
    'profile': 'piw3d_metafi_dtpose_s42_clean_a1_v1',
    'cell': 'metafiplusplus__s42__clean', 'device': 'cuda:0', 'batch': 32,
    'loss': OLD_SMOKE_LOSS,
    'log_sha256': 'd7dae8778964e545d45d70146091f74788273045007129d8dd81c5d56eb06bbf',
    'evidence': 'Preserved Vast terminal output, historical execution 2026-10-27; original smoke log not in local backups.',
}


class Tee:
    """Keep foreground output visible while preserving a separate diagnostic log."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    @property
    def encoding(self):
        return getattr(self.streams[0], 'encoding', 'utf-8')

    def isatty(self):
        return False


@contextmanager
def run_log(args):
    folder = args.out / 'logs'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = folder / f'{args.mode}.{stamp}.{uuid.uuid4().hex}.log'
    print(f'DIAGNOSTIC_LOG={path}', flush=True)
    with path.open('x', encoding='utf-8') as stream:
        with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
            print(f'BEGIN mode={args.mode} pid={os.getpid()} utc={stamp}', flush=True)
            try:
                yield path
            except BaseException:
                traceback.print_exc()
                raise
            else:
                print(f'END mode={args.mode} exit=0', flush=True)


def smoke_report(record, saved):
    if (not isinstance(record, dict)
            or record.get('cell') != 'metafiplusplus__s42__clean'
            or record.get('device') != 'cuda:0'
            or type(record.get('batch')) is not int or record['batch'] != 32
            or type(record.get('generator_steps')) is not int or record['generator_steps'] != 0):
        raise ValueError('Native Clean CUDA B32 smoke receipt is invalid')
    loss = record.get('loss')
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss <= 0:
        raise ValueError('Native smoke loss must be finite and positive')
    for key in ('peak_cuda_allocated', 'peak_cuda_reserved'):
        if type(record.get(key)) is not int or record[key] < 0:
            raise ValueError('Native smoke CUDA memory receipt is invalid')
    json.dumps(record, allow_nan=False)  # Reject non-finite/unserializable extras before any write.
    difference = abs(loss - OLD_SMOKE_LOSS)
    return {
        'profile': PROFILE, 'status': 'SMOKE_ONLY_PASS',
        'plan_sha256': saved['plan_sha256'], 'runtime': saved['runtime'],
        'seed': 42, 'train_indices': list(range(32)),
        'loss_semantics': 'Native ordinary MPJPE TRAIN loss before the single AdamW update; not TEST MPJPE.',
        'old_vast': OLD_SMOKE, 'new_smoke': record,
        'absolute_loss_difference': difference,
        'relative_loss_difference_pct': 100 * (difference / max(abs(loss), abs(OLD_SMOKE_LOSS))),
        'relative_difference_definition': '100 * abs(new-old) / max(abs(new), abs(old))',
        'full_training_launched': False, 'checkpoint_saved': False,
        'interpretation': 'Numeric first-batch comparison only; no efficacy/reproducibility threshold or guarantee of final199 agreement.',
    }


def run_smoke(args, base, w, saved):
    path = args.out / 'smoke_comparison.json'
    if path.exists():
        raise ValueError('Ungated smoke receipt exists; inspect it, never overwrite or silently repeat')
    capture = io.StringIO()
    with redirect_stdout(Tee(sys.stdout, capture)):
        w.smoke_cell(saved, 'metafiplusplus', 'clean', 'cuda:0')
    prefix = 'GPU_SMOKE_WORKER_PASS '
    records = [json.loads(line[len(prefix):]) for line in capture.getvalue().splitlines()
               if line.startswith(prefix)]
    if len(records) != 1:
        raise ValueError('Exactly one native GPU_SMOKE_WORKER_PASS receipt is required')
    report = smoke_report(records[0], saved)
    if checkpoint_path(w, saved['cfg']).exists():
        raise ValueError('Smoke unexpectedly created a checkpoint; no gate will be issued')
    weights_binding(base)  # Also verify a cache downloaded by the native build.
    base.write(path, report)
    print('SMOKE_COMPARISON ' + json.dumps(report, allow_nan=False), flush=True)
    return report


def base_module():
    spec = importlib.util.spec_from_file_location(
        '_clean_vastenv_base', HERE / 'piw_metafi_remaining_3090/controller.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_runtime(runtime):
    if runtime.get('python') != EXPECTED_PYTHON:
        raise ValueError(f'Exact Python {EXPECTED_PYTHON} required; no fallback')
    if runtime.get('system') != 'Windows' or str(runtime.get('machine', '')).lower() not in ('amd64', 'x86_64'):
        raise ValueError('This diagnostic requires Windows x86-64')
    for name, version in EXPECTED_PACKAGES.items():
        if runtime.get('packages', {}).get(name) != version:
            raise ValueError(f'Exact {name}=={version} required; no fallback')


def clean_worker(args):
    w = base_module().load_worker(args, 'clean')
    # Change only the explicitly requested scheduling/profile scope. Never edit
    # extracted scientific files, replace the model, or import a checkpoint.
    w.PROFILE = PROFILE
    w.METHODS = ('clean',)
    w.BACKBONES = ('metafiplusplus',)
    w.BASELINES = ()
    return w


def cfg_for(w, args):
    cfg = w.cfg_for(args.dataset, args.out, args.action)
    cfg['interpretation'] = ('Clean-only fixed-final199 library-matching diagnostic; '
        'fresh training, not an A1 experiment or bitwise Vast/Linux reproduction')
    if (cfg['recipes'] != {'metafiplusplus': RECIPE} or cfg['methods'] != ['clean']
            or cfg['backbones'] != ['metafiplusplus'] or cfg['seed'] != 42
            or cfg['epochs'] != 200 or cfg['final_epoch'] != 199
            or cfg['batch_size'] != 32 or cfg['num_workers'] != 0
            or not cfg['drop_last'] or cfg['amp']):
        raise ValueError('Clean-only original AdamW/B32/FP32/final199 recipe required')
    return cfg


def validate_plan(saved, original):
    validate_runtime(saved['runtime'])
    if saved.get('profile') != PROFILE:
        raise ValueError('Foreign diagnostic plan')
    keys = ('seed', 'epochs', 'final_epoch', 'batch_size', 'num_workers',
            'drop_last', 'amp', 'target', 'rho', 'attack_eps', 'eval_doses',
            'recipes', 'model_initialization', 'source_recipe')
    if any(saved['cfg'].get(k) != original['cfg'].get(k) for k in keys):
        raise ValueError('Scientific recipe differs from the actual Vast source plan')
    if saved['cfg'].get('methods') != ['clean']:
        raise ValueError('Only Clean is authorized in this diagnostic')
    if (saved['inputs']['splits'] != original['inputs']['splits'] or
            saved['action_sha256'] != original['action_sha256']):
        raise ValueError('Full dataset content/order or action differs from Vast')
    for name, value in original['sources'].items():
        if name != 'run_piw_metafi_dtpose_baseline_six.py' and saved['sources'].get(name) != value:
            raise ValueError(f'Frozen scientific source differs: {name}')
    if saved['sources'].get('run_piw_metafi_dtpose_clean_a1.py') != base_module().CLEAN_PROFILE_SHA:
        raise ValueError('Original Clean profile differs')


def code_binding(base):
    return {'runner_sha256': base.sha(__file__), 'native': base.code_binding()}


def weights_binding(base):
    path = Path(os.environ['TORCH_HOME']) / 'hub/checkpoints/resnet34-b627a593.pth'
    if not path.is_file() or base.sha(path) != WEIGHTS_SHA:
        raise ValueError('Original official ResNet34 V1 initialization checksum required')
    return {'sha256': WEIGHTS_SHA}


def gate_path(args):
    return args.out / 'prelaunch_gate.json'


def checkpoint_path(w, cfg):
    return w.cell_dir(cfg, 'metafiplusplus', 'clean') / 'checkpoint.pt'


def execution_for(w):
    execution = w._execution('cuda:0')
    if execution.get('gpu_name') != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Exactly the RTX 3090 on cuda:0 is required; no fallback')
    return execution


def expected_gate(args, base, w, saved):
    smoke_path = args.out / 'smoke_comparison.json'
    smoke = base.read(smoke_path)
    if smoke != smoke_report(smoke['new_smoke'], saved):
        raise ValueError('Smoke comparison receipt changed or belongs to another plan')
    return {'schema': 1, 'profile': PROFILE, 'methods': ['clean'],
            'plan_sha256': saved['plan_sha256'], 'runtime': saved['runtime'],
            'execution': execution_for(w), 'code': code_binding(base),
            'weights': weights_binding(base), 'physical_batch': 32,
            'input_stat_fingerprint': w.input_stat_fingerprint(args.dataset, saved['inputs']),
            'fresh_from_epoch0': True, 'checkpoint_import': False,
            'smoke_comparison_sha256': base.sha(smoke_path),
            'prepared_only': True}


def prepare(args, base, w, cfg):
    if gate_path(args).is_file():
        saved = w.load_plan(cfg)
        check_gate(args, base, w, cfg, saved)
        print('PREPARED_ALREADY; no training or repeated smoke', flush=True)
        return
    if checkpoint_path(w, cfg).exists():
        raise ValueError('Ungated checkpoint exists; never import or overwrite it')
    base.require_idle()
    w.preflight(cfg)  # Full original content/list/skeleton audit, CPU/disk only.
    saved = w.load_plan(cfg)
    validate_plan(saved, base.original_plan())
    execution_for(w)
    weights_path = Path(os.environ['TORCH_HOME']) / 'hub/checkpoints/resnet34-b627a593.pth'
    if weights_path.exists():
        weights_binding(base)  # Refuse a corrupt existing cache before CUDA work.
    # Original real physical-B32 forward/backward/AdamW smoke, including native
    # seed reset. It saves no model/checkpoint and is not the training start.
    run_smoke(args, base, w, saved)
    if checkpoint_path(w, cfg).exists():
        raise ValueError('Prepare unexpectedly created a checkpoint; do not launch')
    base.write(gate_path(args), expected_gate(args, base, w, saved))
    print('PREPARED_ONLY: full input audit + original CUDA B32 smoke; no training', flush=True)


def check_gate(args, base, w, cfg, saved):
    validate_plan(saved, base.original_plan())
    if not gate_path(args).is_file():
        raise ValueError('Run Prepare successfully first; no ungated training')
    gate = base.read(gate_path(args))
    if gate != expected_gate(args, base, w, saved):
        raise ValueError('Code/runtime/GPU/data/initialization gate changed')
    receipt = base.read(w.receipt_path(cfg))
    if receipt.get('input_stat_fingerprint') != gate['input_stat_fingerprint']:
        raise ValueError('Dataset changed since its full content audit')
    if checkpoint_path(w, cfg).exists():
        blob = w._checkpoint(saved, 'metafiplusplus', 'clean')
        base.validate_blob(blob, saved, 'clean',
            {'n_total': 27876, 'n_poison': 0, 'n_cover': 0})
        if blob['execution'] != gate['execution']:
            raise ValueError('Checkpoint execution changed; do not migrate this probe')
    return gate


def comparison(args, base, w, cfg, saved):
    folder = w.cell_dir(cfg, 'metafiplusplus', 'clean')
    path = folder / 'evaluation.json'
    result = base.read(path)
    blob = w._checkpoint(saved, 'metafiplusplus', 'clean', final=True)
    if (result.get('status') != 'COMPLETE' or result.get('method') != 'clean'
            or result.get('profile') != PROFILE or result.get('plan_sha256') != saved['plan_sha256']
            or result.get('checkpoint_sha256') != base.sha(folder / 'checkpoint.pt')
            or result.get('predictions_sha256') != base.sha(folder / 'predictions.npz')
            or blob['epoch'] != 199):
        raise ValueError('Completed Clean evaluation/artifact binding is invalid')
    metrics = result['clean_metrics']
    rows = [{'name': 'Vast historical reference', **OLD_CLEAN},
            {'name': '3090 previous environment', **PREVIOUS_3090},
            {'name': '3090 library-matching probe', **metrics}]
    comparison_path = args.out / 'comparison.json'
    comparison_record = {'profile': PROFILE,
        'status': 'COMPLETE', 'rows': rows, 'plan_sha256': saved['plan_sha256'],
        'code': code_binding(base), 'runtime': saved['runtime'],
        'checkpoint_sha256': result['checkpoint_sha256'],
        'predictions_sha256': result['predictions_sha256'],
        'limitations': ['One seed and one new victim; no TEST-selected checkpoint.',
            'Windows/3090 and its fixed thread policy remain different from Vast/Linux.',
            'This changes a package/toolchain bundle, not one isolated library.',
            'Historical rows are authenticated archived results, not re-evaluated old weights.']}
    lines = ['# Clean-only library-matching diagnostic', '',
        '| Run | Clean MPJPE mm | Clean PA-MPJPE mm |', '| --- | ---: | ---: |']
    lines += [f"| {r['name']} | {r['clean_mpjpe_mm']:.6f} | {r['clean_pampjpe_mm']:.6f} |" for r in rows]
    lines += ['', 'Diagnostic only; no A1 training. Full final199 TEST, not best-checkpoint selection.',
              'No claim that a library alone caused an improvement/degradation or that the Linux trajectory was reproduced.']
    report = args.out / 'comparison.md'
    report_text = '\n'.join(lines) + '\n'
    # Check every existing artifact before writing anything. Never mask an old
    # stale/foreign table with a newly regenerated JSON or vice versa.
    if comparison_path.exists() and base.read(comparison_path) != comparison_record:
        raise ValueError('Existing comparison JSON changed; refusing overwrite')
    if report.exists() and report.read_text(encoding='utf-8') != report_text:
        raise ValueError('Existing comparison Markdown changed; refusing overwrite')
    if not comparison_path.exists():
        base.write(comparison_path, comparison_record)
    if not report.exists():
        with report.open('x', encoding='utf-8') as stream:
            stream.write(report_text)
    print(json.dumps({'status': 'COMPLETE', 'clean_metrics': metrics,
                      'comparison': str(report)}, indent=2), flush=True)


def launch(args, base, w, cfg):
    base.require_idle()
    saved = w.load_plan(cfg)
    check_gate(args, base, w, cfg, saved)
    folder = w.cell_dir(cfg, 'metafiplusplus', 'clean')
    if (folder / 'evaluation.json').exists():
        comparison(args, base, w, cfg, saved)
        return
    final = checkpoint_path(w, cfg).is_file() and w._checkpoint(saved, 'metafiplusplus', 'clean')['epoch'] == 199
    if not final:
        # Resumes only this newly owned run. Never calls select_candidate,
        # prepare_cell, migration, or another method's training function.
        w.train_cell(saved, 'metafiplusplus', 'clean', 'cuda:0')
    w._checkpoint(saved, 'metafiplusplus', 'clean', final=True)
    w.evaluate_cell(saved, 'metafiplusplus', 'clean', 'cuda:0')
    comparison(args, base, w, cfg, saved)


def status(args):
    print('READ_ONLY; no environment/GPU/model initialization', flush=True)
    folder = args.out / 'attacks/metafiplusplus/seed42/clean'
    logs = sorted((args.out / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    if logs:
        print(f'LATEST_LOG={logs[-1]}', flush=True)
        with logs[-1].open(encoding='utf-8', errors='replace') as stream:
            print(''.join(deque(stream, maxlen=16)), flush=True)
    for path in (args.out / 'comparison.json', folder / 'evaluation.json', args.out / 'smoke_comparison.json'):
        if path.is_file():
            print(path.read_text(encoding='utf-8-sig'), flush=True)
            return
    print('No completed Clean evaluation in this diagnostic output.', flush=True)
    if (folder / 'checkpoint.pt').is_file():
        print('Checkpoint exists; inspect changing epoch logs, not only file existence.', flush=True)


def validate_paths(args):
    if args.out.name != PROFILE or args.out.parent != args.data_home / 'runs':
        raise ValueError('Output is restricted to the dedicated new diagnostic run')
    if args.out.is_symlink() or args.dataset.is_symlink() or args.env_dir.is_symlink():
        raise ValueError('Linked output/dataset/environment is not supported')
    if args.out == args.dataset or args.out in args.dataset.parents or args.dataset in args.out.parents:
        raise ValueError('Output and raw data must not overlap')
    if Path(sys.prefix).resolve() != args.env_dir or sys.prefix == sys.base_prefix:
        raise ValueError('Use the new isolated venv, not eclare-env or system Python')
    owner = args.env_dir / 'clean_vastenv_setup.owner.json'
    if not owner.is_file() or json.loads(owner.read_text(encoding='utf-8-sig')).get('profile') != SETUP_PROFILE:
        raise ValueError('Isolated setup ownership marker required')


def reject_linked_path(path):
    # Inspect lexical ancestors BEFORE resolve() follows a Windows junction or
    # symlink. A dedicated leaf under a redirected parent is not isolated.
    lexical = Path(os.path.abspath(path))
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink() or getattr(candidate, 'is_junction', lambda: False)():
            raise ValueError(f'Linked path/ancestor is unsupported: {candidate}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inspect', 'prepare', 'check', 'launch', 'status'), default='inspect')
    parser.add_argument('--data-home', type=Path, default=Path.home() / 'backdooranalog')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--action', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--env-dir', type=Path)
    args = parser.parse_args(argv)
    args.dataset = args.dataset or args.data_home / 'datasets/Person-in-WiFi-3D_processed_v1'
    args.action = args.action or args.data_home / 'actions/data_bend.npy'
    args.out = args.out or args.data_home / 'runs' / PROFILE
    args.env_dir = args.env_dir or args.data_home / 'envs/piw-metafi-vast211'
    for path in (args.data_home, args.dataset, args.action, args.out, args.env_dir):
        if path is not None:
            reject_linked_path(path)
    args.data_home = args.data_home.resolve()
    args.dataset = args.dataset.resolve()
    args.action = args.action.resolve()
    args.out = args.out.resolve()
    args.env_dir = args.env_dir.resolve()
    if args.mode == 'status':
        status(args)
        return 0
    if args.mode == 'inspect':
        print(json.dumps({'profile': PROFILE, 'methods': ['clean'], 'epochs': 200,
            'seed': 42, 'recipe': RECIPE, 'python': EXPECTED_PYTHON,
            'packages': EXPECTED_PACKAGES, 'out': str(args.out),
            'note': 'No environment changes, GPU use or training.'}, indent=2))
        return 0
    validate_paths(args)
    os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MKL_THREADING_LAYER='SEQUENTIAL')
    sys.dont_write_bytecode = True
    if os.environ.get('PIW_SMOKE_BARRIER_DIR'):
        raise ValueError('Foreign parallel smoke barrier is set; do not join it')
    base = base_module()
    base.configure_torch_cache(args)
    with ExitStack() as stack:
        stack.enter_context(base.lock(args.data_home / 'tools/clean_vastenv_gpu0_lock'))
        stack.enter_context(base.lock(args.out))
        stack.enter_context(run_log(args))
        w = clean_worker(args)
        cfg = cfg_for(w, args)
        validate_runtime(w.plan(cfg)['runtime'])
        if args.mode == 'prepare':
            prepare(args, base, w, cfg)
        elif args.mode == 'check':
            base.require_idle()
            check_gate(args, base, w, cfg, w.load_plan(cfg))
            print('VERIFIED_PRELAUNCH_GATE; no training', flush=True)
        else:
            launch(args, base, w, cfg)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'[Clean Vast-env diagnostic] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
