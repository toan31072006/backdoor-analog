"""Frozen HPE-Li BackTime -> fresh MetaFi++ / WiFlow, MM-Fi P1S1 seed42.

Only execution/model-recipe adapters live here. Historical ATKBackd sources
and artifacts are read-only. Prepare is not Launch; there is no attacker fit,
warm start, automatic retry, batch fallback, process kill, or TEST selection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', MKL_THREADING_LAYER='SEQUENTIAL')
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'ATKBackd'))
import run_mmfi_p3s3_siba_ladder_4090 as native

PROFILE = 'mmfi_backtime_hpeli_to_metafi_wiflow_p1s1_s42_4090_v1'
BACKBONES = ('metafiplusplus', 'wiflow')
COUNTS = dict(training=133056, test=33264)
GRID = (0., .2, .4, .6, .8, 1.)
sha, digest, read, write = native.sha, native.digest, native.read, native.write


def signed(row, key):
    unsigned = dict(row)
    claimed = unsigned.pop(key, None)
    if claimed != digest(unsigned):
        raise ValueError(f'Invalid {key} checksum')


def load_adapter():
    spec = importlib.util.spec_from_file_location('_transfer_twosplits',
        REPO / 'remote_tools/Run-MMFiBackTimeTwoSplits4090.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_runtime(source):
    runtime = load_adapter().load_runtime('p1s1')
    plan = read(Path(source) / 'plan.json')
    signed(plan, 'plan_sha256')
    cfg = plan['cfg']['backtime']
    required = dict(model='hpeli', seed=42, mmfi_protocol='protocol1', mmfi_setting='s1',
        mmfi_split_seed=0, rho=.1, poison_select='backtime_train_error',
        dose_coupling='paired', dose_min=.2, dose_max=1., pivot=1,
        target_joints=[2, 3], payload_axis=[0., 0., 1.], theta_max_deg=40.,
        victim_loss='mpjpe', epochs=50)
    if (plan['outdir'] != str(Path(source).resolve()) or plan['methods'] != ['backtime']
            or plan['contract']['counts'] != COUNTS
            or plan['backtime_recipient_policy'] != 'adaptive'
            or any(cfg.get(k) != v for k, v in required.items())
            or list(cfg['dose_grid']) != list(GRID)):
        raise ValueError('Source must be the completed HPE-Li BackTime P1S1/seed42 run')
    return runtime, plan


def source_audit(source, *, full_hash=False):
    """Validate OLD bindings as OLD bindings, never relabel/copy a trigger."""
    runtime, plan = source_runtime(source)
    inputs = runtime.verify_inputs(plan, full_hash=full_hash)
    evaluation = read(Path(source) / 'backtime/evaluation.json')
    env = evaluation['binding']['execution']
    if runtime.completed(plan, 'backtime', env) != evaluation:
        raise ValueError('Source BackTime evaluation incomplete')
    poison = runtime.backtime_poison(plan, env)  # existing file only; required below
    trigger = runtime.load_trigger(plan, 'backtime', env)
    diagnostics = trigger.provenance()['diagnostics']
    if (tuple(trigger.shape) != (3, 114, 10)
            or diagnostics.get('fit_profile') != 'full_TRAIN_streaming_author_schedule_CSI_pose_adaptation'
            or diagnostics.get('fit_epochs') != 100 or diagnostics.get('test_used') is not False
            or diagnostics.get('n_fit') != COUNTS['training']
            or diagnostics.get('n_poisoned_fit') != 13305
            or diagnostics.get('generator_updates') != 13305 * 89
            or list(trigger.training_indices) != poison['indices']):
        raise ValueError('Incomplete/foreign attacker or selected poison identities')
    names = ('plan.json', 'inputs.json', 'poison.json', 'clean_surrogate/checkpoint.pt',
             'backtime/trigger.pt', 'backtime/trigger_provenance.json',
             'backtime/poison.json', 'backtime/checkpoint.pt', 'backtime/evaluation.json',
             'backtime/per_sample.npz')
    record = dict(root=str(Path(source).resolve()),
        files={n: sha(Path(source) / n) for n in names},
        input_content_sha256=inputs['full_content_sha256'],
        ordered_splits=inputs['splits'], generator_sha256=trigger.provenance()['generator_sha256'],
        original_execution=env, surrogate='hpeli', fit_epochs=100,
        recipient_policy='backtime_train_error', attacker_refit=False)
    return runtime, plan, trigger, record


def clean_cell(backbone, data_home, folder):
    if backbone == 'metafiplusplus':
        from run_a1_cross_backbones import configuration
        cfg = configuration(data_home, 'cuda:0', 42, 'clean', backbone)
    elif backbone == 'wiflow':
        from run_a1_wiflow import configuration
        cfg = configuration(data_home, 'cuda:0', 'clean')
    else:
        raise ValueError('Unknown victim architecture')
    return dict(method_key=f'{backbone}__s42__clean', backbone=backbone,
                key='clean', seed=42, origin='fresh', cfg=cfg, ckpt_dir=str(folder))


def clean_audit(backbone, data_home, folder):
    import run_a1_three_tables as frozen
    from a1_three_tables_eval import _checkpoint_metadata, _artifact_hashes, _cache_metrics
    cell = clean_cell(backbone, data_home, folder)
    run = frozen._read_completed(cell)
    current = frozen._current_inputs(cell)
    frozen._audit_run(cell, run, current)
    _checkpoint_metadata(run)
    values, _ = _cache_metrics(run['res'], 'clean')
    return dict(root=str(Path(folder).resolve()), cfg=run['cfg'],
        files=_artifact_hashes(run), inputs=current,
        clean_mpjpe_mm=values['clean_mpjpe_mm'], clean_pampjpe_mm=values['clean_pampjpe_mm'],
        limitation='Historical clean training records bind ordered identities, not full content at training time.')


def configuration(source_cfg, backbone):
    if backbone == 'metafiplusplus':
        from a1_backbone_adapter import recipe_config
        cfg = recipe_config(source_cfg, backbone)
    elif backbone == 'wiflow':
        from a1_wiflow_adapter import recipe_config
        cfg = recipe_config(source_cfg)
    else:
        raise ValueError('Only MetaFi++ and WiFlow requested')
    cfg.update(learned_baseline_profile=PROFILE,
        attacker_access='frozen_shared_hpeli_surrogate_no_victim_feedback',
        attacker_reuse_profile=PROFILE)
    return cfg


def setup_sources():
    names = ('remote_tools/Run-MMFiBackTimeTransfer4090.py',
             'remote_tools/Start-MMFiBackTimeTransfer4090.ps1',
             'remote_tools/Run-MMFiBackTimeTwoSplits4090.py')
    return dict(scientific=native.common.sources(), execution={n: sha(REPO / n) for n in names})


def validate_paths(outdir, source, cleans):
    paths = [Path(p).resolve() for p in (outdir, source, *cleans)]
    for p in (outdir, source, *cleans):
        native.reject_links(p)
    root = paths[0]
    if root.parent.name.lower() != 'runs' or not root.name.startswith('mmfi_backtime_transfer_'):
        raise ValueError('Use a NEW mmfi_backtime_transfer_* directory directly under runs')
    for donor in paths[1:]:
        if root == donor or root in donor.parents or donor in root.parents:
            raise ValueError('Output must not overlap a source run')


def make_plan(args, *, full_hash=False):
    validate_paths(args.outdir, args.source, (args.metafi_clean, args.wiflow_clean))
    runtime, old, trigger, receipt = source_audit(args.source, full_hash=full_hash)
    if Path(args.data_home).resolve() != Path(old['home']).resolve():
        raise ValueError('DataHome must match the source run; no input relocation')
    refs = {b: clean_audit(b, args.data_home, p) for b, p in zip(BACKBONES,
                            (args.metafi_clean, args.wiflow_clean))}
    configs = {b: configuration(old['cfg']['backtime'], b) for b in BACKBONES}
    # The clean reference recipe must agree, not just its model name.
    for b, cfg in configs.items():
        keys = ('model', 'seed', 'optimizer', 'lr', 'weight_decay', 'momentum', 'batch_size',
                'backbone_accumulation_steps', 'backbone_effective_batch_size',
                'backbone_gradient_clip', 'backbone_adam_betas', 'backbone_adam_eps',
                'lr_scheduler', 'epochs', 'num_workers', 'pretrained', 'victim_loss',
                'dataset_root', 'action_npy', 'mmfi_protocol', 'mmfi_setting', 'mmfi_split_seed')
        if any(cfg.get(k) != refs[b]['cfg'].get(k) for k in keys):
            raise ValueError(f'{b}: clean reference recipe differs from the victim')
        current = __import__('run_a1_three_tables')._current_inputs(dict(
            cfg=cfg, key='clean', ckpt_dir=refs[b]['root']))
        if any(current[k] != refs[b]['inputs'][k] for k in ('train', 'eval', 'action_file_sha256')):
            raise ValueError('Clean and victim ordered inputs differ')
    plan = dict(profile=PROFILE, outdir=str(Path(args.outdir).resolve()),
        data_home=str(Path(args.data_home).resolve()), source=receipt, clean=refs, cfg=configs,
        sources=setup_sources(), runtime=dict(python=platform.python_version(), executable=sys.executable),
        contract=dict(status='EXPLORATORY_SHARED_SURROGATE_SINGLE_SEED_NOT_FULL_T1',
            protocol='protocol1-s1', counts=COUNTS, seed=42, rho=.1, n_poison=13305,
            epochs=50, final_epoch=49, doses=list(GRID), target=[2, 3], pivot=1,
            axis=[0., 0., 1.], angle=40., pck_scale=[5, 12], pck_threshold=.2,
            amp=False, workers=0, max_gpu_jobs=2, scratch_victims=True, warm_start=False,
            attacker_refit=False, poison_manifest='source/backtime/poison.json',
            dce='mean five positive doses', no_test_selection=True))
    plan['plan_sha256'] = digest(plan)
    return plan


def plan_args(plan):
    return argparse.Namespace(outdir=plan['outdir'], data_home=plan['data_home'],
        source=plan['source']['root'], metafi_clean=plan['clean']['metafiplusplus']['root'],
        wiflow_clean=plan['clean']['wiflow']['root'])


def verify_plan(plan, *, full_hash=False):
    signed(plan, 'plan_sha256')
    if read(Path(plan['outdir']) / 'plan.json') != plan or make_plan(
            plan_args(plan), full_hash=full_hash) != plan:
        raise ValueError('Source/clean/input/config/runtime changed; preserve run and inspect')


def engine_for(cfg):
    if cfg['model'] == 'wiflow':
        from a1_wiflow_adapter import training_engine
    else:
        from a1_backbone_adapter import training_engine
    return training_engine(cfg)


def build_model(backbone):
    if backbone == 'wiflow':
        from a1_wiflow_adapter import build_wiflow
        return build_wiflow(backbone)
    from models.cross_backbone_factory import build_backbone
    return build_backbone(backbone)


def components(cfg, device='cuda:0'):
    native.common.seed()
    engine = engine_for(cfg)
    model = build_model(cfg['model']).to(device).train()
    name, optimizer = engine._build_optimizer(model, cfg, 'mmfi')
    if name != cfg['optimizer']:
        raise ValueError('Wrong backbone optimizer')
    return engine, model, optimizer


def reused_rows(plan, trigger):
    runtime, old = source_runtime(plan['source']['root'])
    return runtime.training_rows(old, 'backtime', trigger)


def binding(plan, backbone, env):
    return dict(plan_sha256=plan['plan_sha256'], backbone=backbone, execution=env,
        source=plan['source']['files'], clean=plan['clean'][backbone]['files'],
        cfg_sha256=digest(plan['cfg'][backbone]))


def validate_checkpoint(blob, plan, backbone, env):
    import torch
    cfg = plan['cfg'][backbone]
    if (blob.get('binding') != binding(plan, backbone, env) or blob.get('cfg') != cfg
            or type(blob.get('epoch')) is not int or not -1 <= blob['epoch'] <= 49
            or not blob.get('model') or not blob.get('optimizer') or not blob.get('rng')
            or set(blob['rng']) != {'python', 'numpy', 'torch', 'cuda'}
            or not blob['rng']['cuda'] or blob.get('pending_accumulation') != 0):
        raise ValueError('Incomplete/foreign epoch-boundary exact-resume checkpoint')
    groups = blob['optimizer'].get('param_groups', [])
    if len(groups) != 1:
        raise ValueError('Wrong optimizer parameter groups')
    for g in groups:
        if g.get('lr') != cfg['lr'] or g.get('weight_decay') != cfg['weight_decay']:
            raise ValueError('Wrong saved optimizer recipe')
        if backbone == 'metafiplusplus':
            if g.get('momentum') != .9 or 'betas' in g:
                raise ValueError('MetaFi requires SGD momentum .9')
        elif tuple(g.get('betas', ())) != (.9, .999) or g.get('eps') != 1e-8:
            raise ValueError('WiFlow requires native AdamW settings')
    if blob['epoch'] >= 0 and not blob['optimizer'].get('state'):
        raise ValueError('Missing trained optimizer state')
    values = list(blob['model'].values())
    for state in blob['optimizer'].get('state', {}).values():
        values.extend(v for v in state.values() if torch.is_tensor(v))
    if any(not torch.isfinite(v).all() for v in values):
        raise ValueError('Nonfinite checkpoint state')


def load_checkpoint(plan, backbone, env, *, final=False):
    import torch
    path = Path(plan['outdir']) / backbone / 'checkpoint.pt'
    native.reject_links(path)
    blob = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint(blob, plan, backbone, env)
    if final and blob['epoch'] != 49:
        raise ValueError('Only fixed final49 can be evaluated')
    return blob


def train(plan, backbone, env):
    import torch
    from train_backdoor import _rng_state, _restore_rng_state
    cfg = plan['cfg'][backbone]
    _, _, trigger, _ = source_audit(plan['source']['root'])
    rows = reused_rows(plan, trigger)
    trigger.to('cuda:0').eval()
    engine, model, optimizer = components(cfg)
    folder = Path(plan['outdir']) / backbone
    folder.mkdir(exist_ok=True)
    path = folder / 'checkpoint.pt'
    start = 0
    def save(epoch, loss=None):
        native.save_checkpoint(path, dict(epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(), rng=_rng_state(), cfg=cfg,
            binding=binding(plan, backbone, env), pending_accumulation=0, train_loss=loss))
    if path.exists():
        blob = load_checkpoint(plan, backbone, env)
        model.load_state_dict(blob['model'], strict=True)
        optimizer.load_state_dict(blob['optimizer'])
        _restore_rng_state(blob['rng'])
        start = blob['epoch'] + 1
        del blob
        print(f'RESUME {backbone} next_epoch={start}; model/optimizer/all RNG restored', flush=True)
    else:
        save(-1)
    batches = engine._training_loader(rows, cfg)
    for epoch in range(start, 50):
        began, total, n = time.monotonic(), 0., 0
        for batch in batches:
            x = native.injected_batch(trigger, batch)
            loss = engine._victim_update(model, optimizer, x, batch['pose'].to('cuda:0'))
            if not math.isfinite(loss):
                raise ValueError('Nonfinite loss; no retry')
            total += loss * len(x)
            n += len(x)
        if n != COUNTS['training']:
            raise ValueError('Full TRAIN was not consumed')
        save(epoch, total / n)
        print(f'{backbone} epoch {epoch}/49 loss={total/n:.8f} t={time.monotonic()-began:.1f}s', flush=True)
    del model, optimizer, engine, trigger, rows, batches
    torch.cuda.empty_cache()


def summarize(clean, true, dose_rows):
    import numpy as np
    from eval import metrics as M
    return dict(t_mpjpe_mm=dose_rows[-1]['t_mpjpe_mm'],
        target_pck_pct=dose_rows[-1]['target_pck_pct'],
        mean_positive_dce_mm=float(np.mean([r['dce_mm'] for r in dose_rows[1:]])),
        clean_mpjpe_mm=float(M.mpjpe(clean, true).mean()*1000),
        clean_pampjpe_mm=float(M.pa_mpjpe(clean, true).mean()*1000))


def replay(row, arrays, expected_names=None):
    """Recompute all scalar metrics and stored residuals from full XYZ/IDs."""
    import numpy as np
    from extra_attack_eval import measure_pair
    from attack.payload import make_target_pose, set_skeleton_config
    set_skeleton_config('mmfi')
    true, doses, ids = arrays['true_pose'], arrays['doses'], arrays['pair_ids']
    if (true.shape != (COUNTS['test'], 17, 3) or not np.array_equal(doses, GRID)
            or ids.shape != (len(true),) or len(set(ids.tolist())) != len(true)
            or (expected_names is not None and not np.array_equal(ids, expected_names))):
        raise ValueError('Replay TEST counts/ordered identities/doses differ')
    for label in ('clean_reference', 'backtime'):
        clean = arrays[label + '__clean_prediction']
        grid = arrays[label + '__trigger_predictions']
        if grid.shape != (6, *true.shape) or not np.array_equal(grid[0], clean):
            raise ValueError('Replay dose0 identity/topology differs')
        rows = []
        for i, d in enumerate(GRID):
            target = arrays['target_poses'][i]
            geometry = make_target_pose(true, pivot=1, dose=d, theta_max=np.deg2rad(40.),
                                        axis=(0., 0., 1.), target_joints=[2, 3])
            # Labels are constructed from original float64 poses, then collated
            # to float32. Recomputing from saved float32 GT can differ <.001mm.
            if not np.allclose(target, geometry, rtol=0, atol=1e-6):
                raise ValueError('Replay target geometry differs')
            metrics, per = measure_pair(clean, grid[i], true, target)
            rows.append(dict(dose=d, **metrics))
            for name in ('target_error_mm', 'no_trigger_target_error_mm', 'dce_mm'):
                if not np.allclose(arrays[label+'__'+name][i], per[name], rtol=0, atol=1e-7):
                    raise ValueError(f'Replay residual differs: {label}/{name}')
        expected = row['records'][label]
        if len(expected['dose_rows']) != 6:
            raise ValueError('Replay requires all six dose rows')
        for actual, cached in zip(rows, expected['dose_rows']):
            if set(actual) != set(cached) or any(abs(actual[k]-cached[k]) > 1e-6 for k in actual):
                raise ValueError('Replay dose metrics differ')
        actual = summarize(clean, true, rows)
        if any(abs(v-expected['metrics'][k]) > 1e-6 for k, v in actual.items()):
            raise ValueError('Replay summary differs')


def complete(plan, backbone, env, *, replay_arrays=True):
    folder = Path(plan['outdir']) / backbone
    path = folder / 'evaluation.json'
    if not path.exists():
        return None
    row = read(path)
    signed(row, 'record_sha256')
    if (row.get('status') != 'complete' or row.get('profile') != PROFILE
            or row.get('backbone') != backbone or row.get('method') != 'backtime'
            or row.get('metric_contract') != plan['contract']
            or row.get('binding') != binding(plan, backbone, env)
            or row.get('epoch') != 49 or row.get('frames') != COUNTS['test']
            or row.get('checkpoint_sha256') != sha(folder/'checkpoint.pt')
            or row.get('per_sample_sha256') != sha(folder/'per_sample.npz')):
        raise ValueError('Completed evaluation/checkpoint binding changed')
    if replay_arrays:
        import numpy as np
        runtime, old = source_runtime(plan['source']['root'])
        names = [r['name'] for r in runtime.dataset(old, 'test').items]
        with np.load(folder/'per_sample.npz', allow_pickle=False) as arrays:
            replay(row, arrays, names)
    return row


def evaluate(plan, backbone, env):
    import numpy as np
    import torch
    from extra_attack_eval import _loader, measure_pair
    cfg = plan['cfg'][backbone]
    runtime, old, trigger, _ = source_audit(plan['source']['root'], full_hash=True)
    data = runtime.dataset(old, 'test')
    if len(data) != COUNTS['test']:
        raise ValueError('Full official P1S1 TEST required')
    blob = load_checkpoint(plan, backbone, env, final=True)
    folder = Path(plan['outdir']) / backbone
    records, arrays, truth = {}, {}, None
    for label in ('clean_reference', 'backtime'):
        if label == 'clean_reference':
            saved = torch.load(Path(plan['clean'][backbone]['root'])/'checkpoint.pt',
                               map_location='cpu', weights_only=False)
        else:
            saved = blob
        model = build_model(backbone)
        model.load_state_dict(saved['model'], strict=True)
        model.to('cuda:0').eval()
        trigger.to('cuda:0').eval()
        @torch.no_grad()
        def predict(mode, dose=0.):
            preds, truths, targets = [], [], []
            for batch in _loader(data, trigger, cfg, mode, dose):
                x = batch['csi'].to('cuda:0')
                if mode == 'trigger@dose':
                    x = trigger.apply_tensor(x, dose)
                output = model(x)[0]
                if tuple(output.shape[1:]) != (1, 17, 3) or not torch.isfinite(output).all():
                    raise ValueError('Invalid predicted pose')
                preds.append(output.cpu().numpy()[:, 0])
                truths.append(batch['pose'].numpy()[:, 0])
                targets.append(batch['target'].numpy()[:, 0])
            return np.concatenate(preds), np.concatenate(truths), np.concatenate(targets)
        clean, true, _ = predict('clean')
        if truth is not None and not np.array_equal(truth, true):
            raise ValueError('Clean and poisoned victim TEST order differs')
        truth = true
        rows, grid, per_rows, targets = [], [], [], []
        for d in GRID:
            print(f'EVALUATE {backbone}/{label} dose={d:g}', flush=True)
            pred, again, target = predict('trigger@dose', d)
            if not np.array_equal(again, true) or (d == 0 and not np.array_equal(clean, pred)):
                raise ValueError('TEST order or dose0 identity changed')
            metrics, per = measure_pair(clean, pred, true, target)
            rows.append(dict(dose=d, **metrics)); grid.append(pred); per_rows.append(per)
            targets.append(target)
        metrics = summarize(clean, true, rows)
        if label == 'clean_reference':
            for k in ('clean_mpjpe_mm', 'clean_pampjpe_mm'):
                if abs(metrics[k] - plan['clean'][backbone][k]) > .05:
                    raise ValueError('Historical clean utility does not replay within .05mm')
        records[label] = dict(metrics=metrics, dose_rows=rows)
        if 'target_poses' in arrays and not np.array_equal(arrays['target_poses'], np.stack(targets)):
            raise ValueError('Clean reference and victim targets differ')
        arrays['target_poses'] = np.stack(targets)
        arrays[label+'__clean_prediction'] = clean
        arrays[label+'__trigger_predictions'] = np.stack(grid)
        for name in ('target_error_mm', 'no_trigger_target_error_mm', 'dce_mm'):
            arrays[label+'__'+name] = np.stack([p[name] for p in per_rows])
        del model, saved
        torch.cuda.empty_cache()
    arrays.update(true_pose=truth, doses=np.asarray(GRID),
                  pair_ids=np.asarray([r['name'] for r in data.items]))
    row = dict(profile=PROFILE, status='complete', backbone=backbone, method='backtime',
        epoch=49, frames=len(data), binding=binding(plan, backbone, env), records=records,
        checkpoint_sha256=sha(folder/'checkpoint.pt'), metric_contract=plan['contract'])
    replay(row, arrays)
    native.save_npz(folder/'per_sample.npz', arrays)
    verify_plan(plan)  # donor checkpoint/trigger/config cannot change during evaluation
    row['per_sample_sha256'] = sha(folder/'per_sample.npz')
    row['record_sha256'] = digest(row)
    write(folder/'evaluation.json', row)
    print('EVALUATION_COMPLETE '+backbone+' '+json.dumps(records['backtime']['metrics']), flush=True)


@contextmanager
def owned_lock(root):
    """No stale-lock deletion or PID-based killing. An OS handle owns the lock."""
    import run_mmfi_tables
    with run_mmfi_tables._cell_lock(root):
        yield


def barrier(context, stage, backbone, timeout=600):
    folder = Path(context)
    write(folder/f'{stage}.{backbone}.json', dict(pid=os.getpid(), stage=stage))
    began = time.monotonic()
    while not all((folder/f'{stage}.{b}.json').exists() for b in BACKBONES):
        if list(folder.glob('failed.*.json')) or time.monotonic()-began > timeout:
            raise RuntimeError('Concurrent capacity gate failed/timed out; no training or retry')
        time.sleep(.2)


def smoke(plan, backbone, context, env):
    import torch
    from torch.utils.data import Subset
    cfg = plan['cfg'][backbone]
    _, _, trigger, _ = source_audit(plan['source']['root'])
    rows = reused_rows(plan, trigger)
    # 64 genuine TRAIN recipients exercise the frozen trigger at B32.
    _, old = source_runtime(plan['source']['root'])
    indices = read(Path(old['outdir'])/'backtime/poison.json')['indices'][:64]
    sample = Subset(rows, indices)
    trigger.to('cuda:0').eval()
    engine, model, optimizer = components(cfg)
    initial = next(model.parameters()).detach().clone()
    barrier(context, 'resident', backbone)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for i, batch in enumerate(engine._training_loader(sample, cfg)):
        barrier(context, f'batch{i}', backbone)
        x = native.injected_batch(trigger, batch)
        if tuple(x.shape) != (32, 3, 114, 10):
            raise ValueError('Physical B32 required, no automatic shrink')
        losses.append(engine._victim_update(model, optimizer, x, batch['pose'].to('cuda:0')))
    if len(losses) != 2 or not all(math.isfinite(v) for v in losses) or torch.equal(initial, next(model.parameters())):
        raise ValueError('Real CUDA/optimizer/frozen-trigger smoke did not update finite weights')
    barrier(context, 'updated', backbone)
    write(Path(plan['outdir'])/f'gpu_gate.{backbone}.json', dict(binding=binding(plan, backbone, env),
        physical_batch=32, batches=2, effective_batch=cfg['backbone_effective_batch_size'],
        losses=losses, real_train=True, concurrent_workers=2, discarded=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), status='PREPARED_NOT_TRAINED'))


def gate_check(plan, env):
    for b in BACKBONES:
        row = read(Path(plan['outdir'])/f'gpu_gate.{b}.json')
        if (row.get('binding') != binding(plan, b, env) or row.get('physical_batch') != 32
                or row.get('effective_batch') != plan['cfg'][b]['backbone_effective_batch_size']
                or row.get('batches') != 2 or row.get('real_train') is not True
                or row.get('concurrent_workers') != 2 or row.get('discarded') is not True
                or row.get('status') != 'PREPARED_NOT_TRAINED'
                or len(row.get('losses', [])) != 2 or not all(math.isfinite(v) for v in row['losses'])):
            raise ValueError('Missing/foreign concurrent native GPU gate; run Prepare first')


def worker(args):
    plan = read(Path(args.outdir)/'plan.json')
    backbone = args.worker
    verify_plan(plan)
    env = native.execution()
    try:
        with owned_lock(Path(args.outdir)/backbone):
            with native.log_to(args.outdir, args.mode+'.'+backbone):
                if args.mode == 'prepare':
                    smoke(plan, backbone, args.context, env)
                else:
                    gate_check(plan, env)
                    if complete(plan, backbone, env) is None:
                        train(plan, backbone, env)
                        evaluate(plan, backbone, env)
                    complete(plan, backbone, env)
    except BaseException:
        if args.context:
            write(Path(args.context)/f'failed.{backbone}.json', dict(pid=os.getpid(), failed=True))
        raise


def spawn_pair(plan, mode, context):
    processes = {}
    for b in BACKBONES:
        cmd = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
               '--mode', mode, '--outdir', plan['outdir'], '--worker', b, '--context', str(context)]
        processes[b] = subprocess.Popen(cmd, cwd=REPO)
        print(f'OWNED {b} pid={processes[b].pid}', flush=True)
    # A failed job is never retried; the other owned job drains without a kill.
    reported, last = set(), time.monotonic()
    while any(p.poll() is None for p in processes.values()):
        for b, p in processes.items():
            if p.poll() is not None and b not in reported:
                print(f'EXIT {b} code={p.returncode}; other job not killed', flush=True)
                reported.add(b)
        if time.monotonic()-last > 30:
            print('ALIVE '+json.dumps({b:p.poll() is None for b,p in processes.items()}), flush=True)
            last = time.monotonic()
        time.sleep(.5)
    if any(p.returncode != 0 for p in processes.values()):
        raise RuntimeError('Worker failed; preserve logs/checkpoints. No automatic retry.')


def export(plan, env):
    rows = [complete(plan, b, env) for b in BACKBONES]
    if any(r is None for r in rows):
        raise ValueError('Both independent victims must complete before export')
    stream = io.StringIO(newline='')
    columns = ['backbone', 'surrogate', 'seed', 't_mpjpe_mm', 'target_pck_pct',
               'mean_positive_dce_mm', 'clean_mpjpe_mm', 'clean_pampjpe_mm']
    writer = csv.DictWriter(stream, columns)
    writer.writeheader()
    for b, row in zip(BACKBONES, rows):
        writer.writerow(dict(backbone=b, surrogate='hpeli', seed=42, **row['records']['backtime']['metrics']))
    write(Path(plan['outdir'])/'summary.json', dict(profile=PROFILE, contract=plan['contract'],
        records={b:r['records'] for b,r in zip(BACKBONES,rows)}, csv=stream.getvalue()))
    from extra_attack_eval import _atomic_write
    path = Path(plan['outdir'])/'summary.csv'
    payload = stream.getvalue().encode('utf-8')
    if path.exists() and path.read_bytes() != payload:
        raise ValueError('Existing summary CSV differs; preserve it')
    if not path.exists():
        _atomic_write(path, payload)
    print(stream.getvalue(), flush=True)


def inspect(args):
    print(f'{PROFILE}: frozen HPE-Li attacker, two scratch victims; no GPU/write/fit.')
    for label, folder in (('source', args.source), ('MetaFi clean', args.metafi_clean),
                          ('WiFlow clean', args.wiflow_clean), ('new output', args.outdir)):
        print(f'{label}: {folder} exists={Path(folder).exists()}')
        if label in ('MetaFi clean', 'WiFlow clean'):
            for name in ('checkpoint.pt','config.resolved.yaml','eval_cache.json','three_tables_inputs.json'):
                print(f'  {name}: {(Path(folder)/name).is_file()}')
    if args.mode == 'status':
        for b in BACKBONES:
            folder = Path(args.outdir)/b
            print(json.dumps(dict(backbone=b, checkpoint=(folder/'checkpoint.pt').exists(),
                evaluation=(folder/'evaluation.json').exists())))
        logs = Path(args.outdir)/'logs'
        for path in sorted(logs.glob('*.log'), key=lambda p:p.stat().st_mtime)[-2:]:
            print(str(path))
            with path.open(encoding='utf-8') as stream:
                from collections import deque
                print(''.join(deque(stream, maxlen=8)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('inspect','check','prepare','launch','status'), default='inspect')
    p.add_argument('--data-home', default=str(Path.home()/'backdooranalog'))
    p.add_argument('--source', default=r'U:\backdooranalog\runs\mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_u_v1\p1s1')
    p.add_argument('--metafi-clean')
    p.add_argument('--wiflow-clean')
    p.add_argument('--outdir')
    p.add_argument('--worker', choices=BACKBONES, help=argparse.SUPPRESS)
    p.add_argument('--context', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    home = Path(args.data_home)
    args.metafi_clean = args.metafi_clean or str(home/'runs/mmfi_cross_backbones_s42_43_44_v1/metafiplusplus/seed42/clean')
    args.wiflow_clean = args.wiflow_clean or str(home/'runs/mmfi_wiflow_a1_s42_v1/wiflow/seed42/clean')
    args.outdir = args.outdir or str(Path(args.source).parent.parent/'mmfi_backtime_transfer_metafi_wiflow_p1s1_s42_4090_v1')
    if args.worker:
        return worker(args)
    if args.mode in ('inspect','status'):
        return inspect(args)
    plan = make_plan(args, full_hash=True)
    root = Path(args.outdir)
    if args.mode == 'check':
        if (root/'plan.json').exists():
            verify_plan(plan)
        print('CHECK_PASS: source+clean+full content hashes; no CUDA or training', flush=True)
        return
    adapter = load_adapter()
    adapter.check_output_storage(root)
    adapter.require_idle()
    env = native.execution()
    root.mkdir(parents=True, exist_ok=True)
    with owned_lock(root):
        if not (root/'plan.json').exists() and any(x.name != 'run.lock' for x in root.iterdir()):
            raise ValueError('Nonempty unbound output; choose a NEW root')
        if args.mode == 'launch' and not (root/'plan.json').exists():
            raise ValueError('Run Prepare first; Launch never auto-prepares')
        write(root/'plan.json', plan)
        if args.mode == 'prepare':
            if all((root/f'gpu_gate.{b}.json').exists() for b in BACKBONES):
                gate_check(plan, env)
                print('PREPARED_ALREADY; no repeated smoke or training', flush=True)
                return
            context = root/'capacity_checks'/uuid.uuid4().hex
            context.mkdir(parents=True)
            spawn_pair(plan, 'prepare', context)
            gate_check(plan, env)
            print('PREPARED_ONLY. No attacker fit or full victim training.', flush=True)
        else:
            gate_check(plan, env)
            spawn_pair(plan, 'launch', '')
            export(plan, env)


if __name__ == '__main__':
    main()
