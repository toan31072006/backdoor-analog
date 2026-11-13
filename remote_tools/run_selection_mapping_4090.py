"""Isolated MM-Fi/P1S1 HPE-Li 2x3 TRAIN-selection/payload-mapping study.

No historical source/result changes, no clean training, no automatic retries.
Production commands require full-content preflight and two simultaneous smokes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', PYTHONUTF8='1')
REPO = Path(__file__).resolve().parents[1]
SCIENCE = REPO / 'ATKBackd'
sys.path.insert(0, str(SCIENCE))
PROFILE = 'mmfi_hpeli_selection_mapping_s42_v1'
GRID = [0., .2, .4, .6, .8, 1.]
CELLS = [f'{selection}_{mode}' for mode in ('linear', 'sqrt', 'quad')
         for selection in ('uniform', 'diverse')]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    from vast_seed42_preflight import sha256
    return sha256(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f'Immutable file differs: {path}; preserve output')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def atomic_torch(path, value):
    import torch
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def source_record():
    paths = [p for p in SCIENCE.rglob('*') if p.is_file()
             and p.suffix in ('.py', '.yaml', '.yml', '.json')
             and not {'tests', '__pycache__'}.intersection(p.relative_to(SCIENCE).parts)]
    paths += [Path(__file__), REPO/'remote_tools/Start-SelectionMapping4090.ps1']
    return {p.relative_to(REPO).as_posix(): sha(p) for p in sorted(paths)}


def configuration(home, key):
    from run_a1_three_tables import _configuration
    from train_backdoor import _resolve_training_config, _validate_training_contract
    if key not in CELLS:
        raise ValueError('Unknown cell')
    selection, mode = key.split('_')
    cfg = _configuration(Path(home), 'cuda:0', 0, 42, 'a1')
    cfg.update(confirmation_profile=PROFILE, poison_select=selection, dose_mode=mode)
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    return cfg


def plan(home, root):
    from run_piw_hpeli_clean import runtime_versions
    value = dict(profile=PROFILE, data_home=str(Path(home).resolve()),
        outdir=str(Path(root).resolve()), sources=source_record(),
        runtime=runtime_versions(), python=str(Path(sys.executable).resolve()),
        cells={key: configuration(home, key) for key in CELLS},
        contract=dict(status='EXPLORATORY_SINGLE_SEED_NOT_FULL_T1_T2_T3',
            seed=42, train=133056, test=33264, rho=.1, n_poison=13305,
            target=[2, 3], pivot=1, axis=[0., 0., 1.], theta_max_deg=40.,
            payload='historical LEG ROTATION; NOT forward trunk Bend',
            selection='native FPS on full unmodified flattened TRAIN poses; no centering/subset',
            doses='one shared vector, native uniform RNG state after choice; separate copy from selector RNG',
            trigger='native md_multicarrier; epsilon .185; dose-linear injection unchanged',
            max_workers=2, epochs=50, final_epoch=49, batch=32, workers=0,
            optimizer='SGD .001 momentum .9 wd0; fixed LR; FP32 ordinary all-joint MPJPE',
            clean_training=False, clean_trigger_control='NOT_SUPPLIED',
            historical_import='NONE; six fresh matched victims, no unverified historical reuse',
            metrics='endpoint@1; mean DCE over five positive doses; mean ratio of dataset means nDCE',
            uncertainty='one seed; no across-seed SD/significance'))
    value['plan_sha256'] = digest(value)
    return value


def output_root(path, home):
    root = Path(path).resolve()
    if (root.parent.name.lower() != 'runs'
            or not root.name.startswith('mmfi_selection_mapping_s42_v')
            or root.is_relative_to(REPO) or root == Path(home).resolve()
            or root.is_relative_to(Path(home).resolve()/'datasets')):
        raise ValueError('Use a NEW runs/mmfi_selection_mapping_s42_v1 outside code/datasets')
    return root


def seed_all():
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def shared_selection(base, count):
    """Native uniform doses retained; FPS selection cannot advance dose RNG."""
    import numpy as np
    from attack.poison import PoisonedDataset
    start = time.perf_counter()
    uniform = np.random.default_rng(42).choice(len(base), count, replace=False)
    uniform_seconds = time.perf_counter()-start
    # Replay only native uniform RNG consumption in an independent generator.
    dose_rng = np.random.default_rng(42)
    dose_rng.choice(len(base), count, replace=False)
    doses = dose_rng.uniform(.2, 1., count)
    start = time.perf_counter()
    shell = object.__new__(PoisonedDataset)
    shell.base = base
    diverse = shell._select_poison(np.random.default_rng(42), len(base), count, 'diverse')
    return dict(uniform=uniform.tolist(), diverse=diverse.tolist(), doses=doses.tolist(),
        timing_seconds=dict(uniform=uniform_seconds, diverse=time.perf_counter()-start),
        policy='native full TRAIN FPS on raw flattened coordinates, first random seed42; shared rank-wise doses')


def paired(base, trigger, cfg, selection):
    import numpy as np
    from attack.poison import PoisonedDataset
    ids = selection[cfg['poison_select']]
    result = PoisonedDataset(base, trigger, mode='train', rho=.1,
        dose_min=.2, dose_max=1., eps=.185, pivot=1, theta_max_deg=40.,
        dose_mode=cfg['dose_mode'], axis=[0., 0., 1.], seed=42,
        select=cfg['poison_select'], dataset='mmfi', dose_coupling='paired',
        explicit_indices=ids)
    if (list(result.poison_plan) != list(zip(ids, selection['doses']))
            or result.n_cover != 0 or not np.isfinite(selection['doses']).all()):
        raise ValueError('Shared poison/dose plan mismatch')
    return result


def metrics(gt, prediction, targets):
    import numpy as np
    from eval.metrics import mpjpe, pa_mpjpe
    from eval.target_pck import target_pck
    gt, prediction, targets = [np.asarray(a, dtype=np.float64) for a in (gt, prediction, targets)]
    if (gt.ndim != 3 or gt.shape[1:] != (17, 3) or len(gt) == 0
            or prediction.shape != (6, *gt.shape) or targets.shape != prediction.shape
            or not all(np.isfinite(a).all() for a in (gt, prediction, targets))
            or not np.array_equal(targets[0], gt)):
        raise ValueError('Expected finite six-dose full XYZ and identity dose0 target')
    js, other = [2, 3], [j for j in range(17) if j not in (2, 3)]
    clean = prediction[0]
    rows = []
    for index, dose in enumerate(GRID):
        pred, target = prediction[index], targets[index]
        requested = np.linalg.norm((target-gt)[:, js], axis=-1).mean()*1000
        dce = np.linalg.norm(((pred-clean)-(target-gt))[:, js], axis=-1).mean()*1000
        if dose > 0 and requested <= 0:
            raise ValueError('Cannot normalize zero requested movement')
        rows.append(dict(dose=dose,
            t_mpjpe_mm=float(np.linalg.norm((pred-target)[:, js], axis=-1).mean()*1000),
            target_pck_pct=100*target_pck(pred, target, gt), dce_mm=float(dce),
            ndce=None if dose == 0 else float(dce/requested), requested_mm=float(requested),
            predicted_mm=float(np.linalg.norm((pred-clean)[:, js], axis=-1).mean()*1000),
            nt_drift_mm=float(np.linalg.norm((pred-clean)[:, other], axis=-1).mean()*1000),
            no_trigger_t_mpjpe_mm=float(np.linalg.norm((clean-target)[:, js], axis=-1).mean()*1000),
            no_trigger_target_pck_pct=100*target_pck(clean, target, gt),
            triggered_gt_target_mm=float(np.linalg.norm((pred-gt)[:, js], axis=-1).mean()*1000)))
    summary = dict(t_mpjpe_mm=rows[-1]['t_mpjpe_mm'], target_pck_pct=rows[-1]['target_pck_pct'],
        mean_dce_mm=float(np.mean([r['dce_mm'] for r in rows[1:]])),
        mean_ndce=float(np.mean([r['ndce'] for r in rows[1:]])),
        clean_mpjpe_mm=float(mpjpe(clean, gt).mean()*1000),
        clean_pampjpe_mm=float(pa_mpjpe(clean, gt).mean()*1000),
        nt_drift_mm=rows[-1]['nt_drift_mm'])
    return dict(summary=summary, dose_rows=rows)


def verify(root):
    from vast_seed42_preflight import check_receipt
    bound = read(root/'plan.json')
    current = plan(bound['data_home'], root)
    if bound != current:
        raise ValueError('Plan/source/runtime/config changed; preserve run, no automatic migration')
    receipt = read(root/'inputs.json')
    check_receipt(receipt)
    selection = read(root/'selection.json')
    if sha(root/'selection.json') != read(root/'prepared.json')['selection_sha256']:
        raise ValueError('Selection file changed')
    validate_selection(selection, bound['contract']['train'], bound['contract']['n_poison'])
    return bound, receipt, selection


def validate_selection(selection, n, count):
    import numpy as np
    for name in ('uniform', 'diverse'):
        ids = selection[name]
        if (len(ids) != count or len(set(ids)) != count
                or any(type(i) is not int or not 0 <= i < n for i in ids)):
            raise ValueError('Selection IDs invalid')
    doses = np.asarray(selection['doses'])
    expected_rng = np.random.default_rng(42)
    expected_ids = expected_rng.choice(n, count, replace=False)
    expected_doses = expected_rng.uniform(.2, 1., count)
    if (not np.array_equal(selection['uniform'], expected_ids)
            or not np.array_equal(doses, expected_doses)):
        raise ValueError('Native uniform/shared dose vector changed')


def components(cfg, selection):
    import torch
    from train_backdoor import _load_dataset, build_trigger, _build_optimizer
    from models.factory import build_model
    seed_all()
    base = _load_dataset(cfg, 'training')
    if len(base) != 133056:
        raise ValueError('Full P1S1 TRAIN required')
    trigger = build_trigger(cfg)
    pairs = paired(base, trigger, cfg, selection)
    model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False).to('cuda:0')
    _, optimizer = _build_optimizer(model, cfg, 'mmfi')
    return base, trigger, pairs, model, optimizer


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def checkpoint_binding(bound, receipt, selection, key):
    return dict(plan=bound['plan_sha256'], inputs=receipt['binding_sha256'],
                selection=digest(selection), cell=key, cfg=bound['cells'][key])


def check_checkpoint(blob, binding):
    import torch
    if (blob.get('binding') != binding or type(blob.get('epoch')) is not int
            or not -1 <= blob['epoch'] <= 49 or not blob.get('model')
            or 'optimizer' not in blob or not blob.get('initial_model_sha256')
            or set(blob.get('rng', {})) != {'python', 'numpy', 'torch', 'cuda'}
            or not blob['rng']['cuda']
            or any(not torch.isfinite(v).all() for v in blob['model'].values())):
        raise ValueError('Incomplete/mismatched checkpoint; no scratch fallback')
    groups = blob['optimizer'].get('param_groups', [])
    if not groups or any(g['lr'] != .001 or g['momentum'] != .9 or g['weight_decay'] != 0 for g in groups):
        raise ValueError('SGD recipe changed')


def evaluate(root, key, cfg, model, trigger, binding):
    import numpy as np
    from torch.utils.data import DataLoader
    from attack.poison import PoisonedDataset, collate
    from train_backdoor import _load_dataset, _predict
    from run_peak_confirmation import _pair_ids
    base = _load_dataset(cfg, 'test')
    if len(base) != 33264:
        raise ValueError('Full P1S1 TEST required')
    if hasattr(base, 'enable_evaluation_cache'):
        base.enable_evaluation_cache()
    predictions, targets, gt = [], [], None
    for dose in GRID:
        pairs = PoisonedDataset(base, trigger, mode='trigger@dose', fixed_dose=dose,
            eps=.185, pivot=1, theta_max_deg=40., dose_mode=cfg['dose_mode'],
            axis=[0., 0., 1.], dataset='mmfi')
        pred, truth, target, _ = _predict(model, DataLoader(pairs, batch_size=32,
            num_workers=0, collate_fn=collate), 'cuda:0')
        if gt is not None and not np.array_equal(gt, truth):
            raise ValueError('Dose TEST identity/truth drift')
        gt = truth
        predictions.append(pred)
        targets.append(target)
        print(f'[{key}] evaluated dose={dose:g}', flush=True)
    directory = root/key
    arrays = directory/'poses_full_test.npz'
    with arrays.with_suffix('.npz.tmp').open('wb') as handle:
        np.savez_compressed(handle, gt=gt, prediction=np.stack(predictions), target=np.stack(targets),
            doses=np.array(GRID), pair_ids_json=np.array(json.dumps(_pair_ids(base))),
            binding_json=np.array(json.dumps(binding, sort_keys=True)))
    os.replace(arrays.with_suffix('.npz.tmp'), arrays)
    # Report from saved bytes, not a separate metric implementation/cache.
    with np.load(arrays, allow_pickle=False) as saved:
        report = metrics(saved['gt'], saved['prediction'], saved['target'])
    report.update(status='complete', profile=PROFILE, cell=key, binding=binding,
        frames=33264, final_epoch=49, arrays_sha256=sha(arrays),
        checkpoint_sha256=sha(directory/'checkpoint.pt'),
        initial_model_sha256=read(directory/'initialization.json')['model_sha256'],
        clean_trained_trigger_control='NOT_SUPPLIED', source='saved full TEST XYZ; no alignment')
    write(directory/'evaluation.json', report)
    write(directory/'complete.json', dict(evaluation_sha256=sha(directory/'evaluation.json'),
        checkpoint_sha256=report['checkpoint_sha256'], arrays_sha256=report['arrays_sha256'],
        binding=binding))


def worker(root, key, smoke=False):
    import torch
    import numpy as np
    from train_backdoor import _training_loader, _victim_update, _rng_state, _restore_rng_state
    from attack.poison import collate
    from run_mmfi_tables import _cell_lock
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''):
        raise ValueError('Use physical GPU0; CUDA_VISIBLE_DEVICES remapping is not allowed')
    bound, receipt, selection = verify(root)
    cfg = bound['cells'][key]
    directory = root/key
    with _cell_lock(directory):
        _, trigger, pairs, model, optimizer = components(cfg, selection)
        if torch.cuda.get_device_name(0).find('4090') < 0:
            raise ValueError('This capacity profile requires RTX4090')
        if smoke:
            torch.cuda.reset_peak_memory_stats()
            ids = selection[cfg['poison_select']][:4]
            ids += [i for i in range(len(pairs)) if i not in pairs.poison_idx][:28]
            batch = collate([pairs[i] for i in ids])
            # Both full B32 graphs remain resident at the barrier (not just weights).
            model.train()
            before = next(model.parameters()).detach().clone()
            output, _ = model(batch['csi'].to('cuda:0'))
            ready = root/'smoke'/f'{key}.ready.json'
            write(ready, dict(pid=os.getpid()))
            other = root/'smoke'/f'{"diverse_quad" if key == "uniform_sqrt" else "uniform_sqrt"}.ready.json'
            deadline = time.monotonic()+180
            while not other.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Two-worker smoke barrier timed out; no retry')
                time.sleep(.5)
            loss = torch.linalg.vector_norm(output-batch['pose'].to('cuda:0'), dim=-1).mean()
            loss.backward()
            optimizer.step()
            if (not torch.isfinite(loss) or torch.equal(before, next(model.parameters()))
                    or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())):
                raise ValueError('Nonfinite smoke')
            write(root/'smoke'/f'{key}.json', dict(loss=float(loss.detach()),
                peak_bytes=torch.cuda.max_memory_allocated(), discarded=True,
                plan_sha256=bound['plan_sha256']))
            return
        binding = checkpoint_binding(bound, receipt, selection, key)
        start = 0
        checkpoint = directory/'checkpoint.pt'
        init_hash = model_digest(model)
        if checkpoint.exists():
            blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
            check_checkpoint(blob, binding)
            if blob['initial_model_sha256'] != init_hash:
                raise ValueError('Initialization changed')
            model.load_state_dict(blob['model'])
            optimizer.load_state_dict(blob['optimizer'])
            _restore_rng_state(blob['rng'])
            start = blob['epoch']+1
        elif any(directory.iterdir()):
            if any(p.name not in ('run.lock', 'console.log') for p in directory.iterdir()):
                raise ValueError('Unbound partial cell; preserve artifacts')
        write(directory/'initialization.json', dict(model_sha256=init_hash), immutable=True)
        write(directory/'poison_manifest.json', pairs.manifest(), immutable=True)
        if not checkpoint.exists():
            atomic_torch(checkpoint, dict(epoch=-1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=_rng_state(), binding=binding, initial_model_sha256=init_hash))
        loader = _training_loader(pairs, cfg)
        for epoch in range(start, 50):
            model.train()
            losses, started = [], time.monotonic()
            for batch in loader:
                value = _victim_update(model, optimizer, batch['csi'].to('cuda:0'), batch['pose'].to('cuda:0'))
                if not np.isfinite(value):
                    raise ValueError('Nonfinite training loss; no skipped batch')
                losses.append(value)
            atomic_torch(checkpoint, dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=_rng_state(), binding=binding, initial_model_sha256=init_hash))
            print(f'[{key}] epoch {epoch}/49 loss={np.mean(losses):.7f} seconds={time.monotonic()-started:.1f}', flush=True)
        evaluate(root, key, cfg, model, trigger, binding)


def completed(root, key, bound, receipt, selection):
    directory = root/key
    if not (directory/'complete.json').exists():
        return False
    complete = read(directory/'complete.json')
    expected = checkpoint_binding(bound, receipt, selection, key)
    if complete['binding'] != expected:
        raise ValueError('Completed binding mismatch')
    for name, field in [('evaluation.json', 'evaluation_sha256'), ('checkpoint.pt', 'checkpoint_sha256'),
                        ('poses_full_test.npz', 'arrays_sha256')]:
        if sha(directory/name) != complete[field]:
            raise ValueError(f'Completed artifact changed: {key}/{name}')
    return True


def run_queue(commands, max_workers=2):
    """Stop scheduling on first failure; drain owned peers, never terminate jobs."""
    pending, active, failed = list(commands), [], []
    last_heartbeat = 0.
    try:
        while pending or active:
            # Observe failures BEFORE scheduling another cell after a slot frees.
            for entry in active[:]:
                key, process, stream = entry
                code = process.poll()
                if code is not None:
                    stream.close()
                    active.remove(entry)
                    print(f'EXIT {key} code={code}', flush=True)
                    if code:
                        failed.append((key, code))
            if failed and not active:
                raise RuntimeError(f'Queue stopped; peers drained, no retry: {failed}')
            while pending and len(active) < max_workers and not failed:
                key, command, logfile = pending.pop(0)
                logfile.parent.mkdir(parents=True, exist_ok=True)
                stream = logfile.open('a', encoding='utf-8')
                process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
                active.append((key, process, stream))
                print(f'START {key} PID={process.pid}; log={logfile}', flush=True)
            for entry in active[:]:
                key, process, stream = entry
                code = process.poll()
                if code is not None:
                    stream.close()
                    active.remove(entry)
                    print(f'EXIT {key} code={code}', flush=True)
                    if code:
                        failed.append((key, code))
            if failed and not active:
                raise RuntimeError(f'Queue stopped; peers drained, no retry: {failed}')
            if time.monotonic()-last_heartbeat > 30 and active:
                print('ALIVE '+str([(key, p.pid) for key, p, _ in active]), flush=True)
                last_heartbeat = time.monotonic()
            if active:
                time.sleep(1)
    finally:
        # Interrupting controller is NOT permission to kill training peers.
        for key, process, stream in active:
            print(f'Controller exiting; worker {key} PID={process.pid} NOT stopped', flush=True)
            stream.close()


def commands(root, keys, smoke=False):
    return [(key, [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
        '--mode', '_smoke' if smoke else '_worker', '--outdir', str(root), '--cell', key],
        root/('smoke' if smoke else key)/('console.log' if not smoke else key+'.log')) for key in keys]


def gpu_check():
    from run_bend_forward import gpu_guard
    return gpu_guard(0)


def export(root):
    bound, receipt, selection = verify(root)
    rows = []
    for selection_name in ('uniform', 'diverse'):
        for mode in ('linear', 'sqrt', 'quad'):
            key = selection_name+'_'+mode
            if not completed(root, key, bound, receipt, selection):
                raise ValueError('All six verified results required; no partial complete table')
            result = read(root/key/'evaluation.json')
            rows.append(dict(selection=selection_name, mapping=mode, **result['summary']))
    inits = {read(root/key/'evaluation.json')['initial_model_sha256'] for key in CELLS}
    if len(inits) != 1:
        raise ValueError('Six initializations differ')
    with (root/'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(root/'results.json', dict(status='COMPLETE_EXPLORATORY_SINGLE_SEED',
        plan_sha256=bound['plan_sha256'], rows=rows, selection_timing=selection['timing_seconds'],
        ndce='mean_d>0(mean_test,joint(DCE)/mean_test,joint(requested)); not mean per-sample ratios',
        clean_trained_trigger_control='NOT_SUPPLIED; no clean training'))
    headings = ['Selection','Mapping','T-MPJPE','T-PCK (%)','DCE','nDCE','Clean MPJPE','NT-drift']
    columns = ['t_mpjpe_mm','target_pck_pct','mean_dce_mm','mean_ndce','clean_mpjpe_mm','nt_drift_mm']
    lines = ['# Selection x dose mapping — HPE-Li/MM-Fi P1S1/seed42', '',
        '| '+' | '.join(headings)+' |', '| '+' | '.join(['---']*len(headings))+' |']
    for row in rows:
        values = [row['selection'],row['mapping']]+[f'{row[c]:.3f}' if c=='mean_ndce' else f'{row[c]:.2f}' for c in columns]
        lines.append('| '+' | '.join(values)+' |')
    lines += ['', 'All errors mm, PCK percent, nDCE dimensionless. Endpoint d=1; DCE/nDCE mean five positive doses.',
        'No independent clean-model+trigger control supplied. Single seed; no significance/across-seed uncertainty.',
        'Historical leg target, not forward trunk Bend. Six matched fresh victims. PA-MPJPE in CSV/JSON.',
        'Selection timing (seconds, this machine only): '+json.dumps(selection['timing_seconds'])]
    (root/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'COMPLETE: {root / "results.csv"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['inspect', 'prepare', 'capacity', 'launch', 'resume', 'status', 'export', '_worker', '_smoke'], required=True)
    parser.add_argument('--data-home', default=r'C:\Users\Van\backdooranalog')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--cell', choices=CELLS)
    parser.add_argument('--fresh-linear', action='store_true', help='Explicitly authorize sixth matched fresh victim; no historical import')
    args = parser.parse_args(argv)
    root = output_root(args.outdir, args.data_home)
    if args.mode == 'inspect':
        print(json.dumps(dict(cells=CELLS, jobs=2, epochs=50, clean_training=False,
            policy='six fresh victims only with --fresh-linear; historical reuse not assumed',
            data_home=args.data_home, outdir=str(root)), indent=2))
        return
    if args.mode == 'status':
        for key in CELLS:
            directory = root/key
            print(key, 'complete='+str((directory/'complete.json').exists()))
            logfile = directory/'console.log'
            if logfile.exists():
                with logfile.open(encoding='utf-8', errors='replace') as handle:
                    from collections import deque
                    print(''.join(deque(handle, maxlen=3)))
        return
    if args.mode in ('_worker', '_smoke'):
        if not args.cell:
            parser.error('--cell required')
        return worker(root, args.cell, smoke=args.mode == '_smoke')
    from run_mmfi_tables import _cell_lock
    with _cell_lock(root):
        if args.mode == 'prepare':
            if not args.fresh_linear:
                raise ValueError('Supply --fresh-linear to authorize six fresh matched victims; no silent old-row retraining')
            if (root/'plan.json').exists() or any(p.name != 'run.lock' for p in root.iterdir()):
                raise ValueError('Prepare requires NEW empty output; preserve partial artifacts')
            import shutil
            if shutil.disk_usage(root).free < 10*1024**3:
                raise ValueError('At least 10GiB free on output volume required')
            from vast_seed42_preflight import inspect
            from train_backdoor import _load_dataset
            bound = plan(args.data_home, root)
            receipt = inspect(args.data_home)
            base = _load_dataset(bound['cells']['uniform_linear'], 'training')
            selection = shared_selection(base, 13305)
            validate_selection(selection, 133056, 13305)
            write(root/'plan.json', bound, immutable=True)
            write(root/'inputs.json', receipt, immutable=True)
            write(root/'selection.json', selection, immutable=True)
            write(root/'prepared.json', dict(selection_sha256=sha(root/'selection.json'),
                plan_sha256=bound['plan_sha256']), immutable=True)
            print('PREPARED_CPU_ONLY: full input audit/FPS once; next run Capacity', flush=True)
            return
        bound, receipt, selection = verify(root)
        if args.mode == 'export':
            return export(root)
        if args.mode == 'capacity':
            if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''):
                raise ValueError('No CUDA device remapping')
            gpu_check()
            if (root/'smoke').exists():
                raise ValueError('Smoke artifacts already exist; no automatic retry/stale barrier reuse')
            run_queue(commands(root, ['uniform_sqrt', 'diverse_quad'], smoke=True))
            gate = dict(plan_sha256=bound['plan_sha256'], gpu=torch_gpu(),
                reports={key: read(root/'smoke'/f'{key}.json') for key in ('uniform_sqrt','diverse_quad')})
            gate['gate_sha256'] = digest(gate)
            write(root/'capacity.json', gate, immutable=True)
            print('CAPACITY_PASS: two real B32 graphs/backward; weights discarded; no full training', flush=True)
            return
        if args.mode == 'launch' and any((root/key/'checkpoint.pt').exists() for key in CELLS):
            raise ValueError('Checkpoint exists; use explicit Resume, not first Launch')
        gate = read(root/'capacity.json')
        projection = dict(gate)
        claimed = projection.pop('gate_sha256', None)
        if (gate['plan_sha256'] != bound['plan_sha256'] or claimed != digest(projection)
                or set(gate['reports']) != {'uniform_sqrt','diverse_quad'}
                or any(r.get('discarded') is not True or r.get('peak_bytes',0) <= 0
                       or r.get('plan_sha256') != bound['plan_sha256'] for r in gate['reports'].values())):
            raise ValueError('Capacity plan mismatch')
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''):
            raise ValueError('No CUDA device remapping')
        gpu_check()
        if gate['gpu'] != torch_gpu():
            raise ValueError('GPU/runtime differs from capacity gate')
        pending = [key for key in CELLS if not completed(root, key, bound, receipt, selection)]
        run_queue(commands(root, pending))
        from vast_seed42_preflight import check_receipt
        check_receipt(receipt, full_hash=True)
        export(root)


def torch_gpu():
    # Controller never creates a CUDA context alongside the two actual workers.
    report = subprocess.run(['nvidia-smi','--id=0',
        '--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=30)
    rows = [line.strip() for line in report.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or '4090' not in rows[0]:
        raise ValueError('Expected physical RTX4090 GPU0')
    return rows[0]


if __name__ == '__main__':
    main()
