"""PiW HPE-Li Cross/Nod target actions; explicit preparation, never auto-launch.

Separate exploratory victims, seed42, native PiW v3 recipe (AdamW .001,
drop_last=True, B32, 200 epochs/final199). Not the old motion-prior ablation,
not a complete T7/comparison table and not physical RF validation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from importlib.metadata import version

os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', PYTHONUTF8='1')

from run_mmfi_action_targets import (Tee, digest, read, sha, source_hashes,
                                    write, write_npz, new_cross_reach_audit,
                                    add_cross_reach_records, audit_action_geometry,
                                    finish_geometry_audit, verify_geometry_audit)

HERE = Path(__file__).resolve().parent
PROFILE = 'piw_hpeli_cross_nod_targets_s42_v1'
PLAN_NAME = 'piw_action_targets.plan.json'
GRID = [0., .2, .4, .6, .8, 1.]
COUNTS = {'training': 27876, 'validation': 2586}
FIXED_TEST = [0, 1293, 2585]


def verify_bundle():
    manifest = HERE.parent / 'PIW_CROSS_NOD_CODE_MANIFEST.json'
    if not manifest.exists():
        return
    for name, record in read(manifest).items():
        path = (HERE.parent / name).resolve()
        if (not path.is_relative_to(HERE.parent) or not path.is_file()
                or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']):
            raise ValueError('Missing/changed/unsafe code bundle member: ' + name)


def check_paths(args):
    """Standard library only: no scientific imports, CUDA, data loading or writes."""
    root = Path(args.dataset_root or Path(args.data_home) / 'datasets/Person-in-WiFi-3D_processed_v1')
    required = {'TRAIN list': root / 'train_data/train_data_list.txt',
        'TEST list': root / 'test_data/test_data_list.txt',
        'cross prior': Path(args.cross_action or Path(args.data_home) / 'actions/data_cross.npy'),
        'nod prior': Path(args.nod_action or Path(args.data_home) / 'actions/data_nod.npy')}
    required.update({key: HERE / key for key in (
        'attack/piw_action_targets.py', 'piw_action_target_data.py',
        'piw_a1_adapter.py', 'run_piw_hpeli_clean.py', 'models/hpeli.py')})
    missing = []
    for label, path in required.items():
        ok = path.is_file()
        print(f'{label}: {"OK" if ok else "MISSING"} {path.resolve()}', flush=True)
        if not ok:
            missing.append(label)
    print(f'Output: {Path(args.outdir).resolve()}\nCHECK_ONLY: no training/inference or output writes.', flush=True)
    if missing:
        raise FileNotFoundError('Missing: ' + ', '.join(missing))


def configuration(args, action, action_file):
    import run_piw_hpeli_clean as base
    from attack.piw_action_targets import specification
    cfg = base.configuration(args.data_home, args.outdir, 'cuda:0', args.dataset_root)
    cfg.update(profile=PROFILE, action=action, payload=specification(action),
        action_npy=str(Path(action_file).resolve()), drop_last=True,
        rho=.1, attack_eps=.185, training_doses=[.2, 1.], eval_doses=GRID,
        target_pck_reference=[6, 4], target_pck_threshold=.2,
        recipe_reference='run_piw_hpeli_lr_control.py v3: AdamW .001/wd.01/drop_lastTrue',
        initialization='independent scratch per action; no historical warm start',
        status='EXPLORATORY_NEW_ACTION_TARGETS_NOT_COMPLETE_T7')
    return cfg


def trigger_record(path):
    from piw_a1_adapter import build_trigger
    record = build_trigger(path).state_record()
    for key in ('pivot', 'target_joints', 'payload_axis', 'theta_max_deg', 'dose_mode'):
        record['protocol'].pop(key, None)
    record['target_policy'] = 'Separately bound new PiW Cross/Nod action payload, not native leg rotation'
    return record


def build_plan(args):
    import run_piw_hpeli_clean as base
    from run_mmfi_a1_motion_priors_4090 import action_record
    root, home = Path(args.outdir).resolve(), Path(args.data_home).resolve()
    if (not re.fullmatch(r'piw_action_targets_cross_nod_s42_v[0-9]+', root.name)
            or root.parent.name.lower() != 'runs' or root.is_relative_to(HERE)
            or root == home or home.is_relative_to(root)):
        raise ValueError('Use a NEW dedicated runs/piw_action_targets_cross_nod_s42_v1 (or v2, v3, ...)')
    if (root.exists() and any(p.name != 'run.lock' for p in root.iterdir())
            and not (root / PLAN_NAME).is_file()):
        raise ValueError('Nonempty unrelated output retained; choose a new directory')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''):
        raise ValueError('Physical cuda:0 only; no CUDA_VISIBLE_DEVICES remapping')
    cells = []
    for action in ('cross', 'nod'):
        path = getattr(args, action + '_action') or home / 'actions' / f'data_{action}.npy'
        if Path(path).resolve().is_relative_to(root):
            raise ValueError('Read-only action inputs must be outside experiment output')
        cells.append(dict(action=action, cfg=configuration(args, action, path),
            action_record=action_record(path, action, args.approve_action_labels),
            trigger_record=trigger_record(path)))
    if cells[0]['action_record']['sha256'] == cells[1]['action_record']['sha256']:
        raise ValueError('Cross/Nod must not be identical prior files')
    runtime = base.runtime_versions()
    runtime['packages']['matplotlib'] = version('matplotlib')
    value = dict(schema=1, profile=PROFILE, outdir=str(root), data_home=str(home),
        cells=cells, sources=source_hashes(), runtime=runtime,
        contract=dict(dataset='Person-in-WiFi-3D one-person', counts=COUNTS,
            full_official_lists=True, TRAIN_next_frame_filter=True,
            seed=42, rho=.1, n_poison=2787, doses=GRID, epochs=200, final_epoch=199,
            optimizer='AdamW lr.001 wd.01 betas(.9,.999) eps1e-8 fixed LR',
            batch=32, workers=0, drop_last=True, FP32=True, ordinary_all_joint_MPJPE=True,
            jobs=1, pretrained=False, pck_clean_gt_reference=[6, 4], pck_threshold=.2,
            fixed_test_indices=FIXED_TEST, test_selection='first/middle/last official TEST indices, not score selected',
            limitations=['Single-seed new-target extension, not the old motion-prior table or complete T7.',
                'Cross uses bilateral IK and bone-direction interpolation, not a rigid subtree; no monotonic/near-linear theorem.',
                'Body frame uses virtual hip/shoulder centres; forward convention requires TRAIN inspection.',
                'Bone lengths do not establish anatomical limits, collision/contact consistency or action recognition.',
                'No new Clean/baseline/defense training and no clean-trained+trigger control supplied.',
                'Old bend-prior leg rotation is not a trunk-bend target.']))
    coordinator = getattr(args, 'coordinator', None)
    if coordinator is not None:
        value['coordinator'] = coordinator
        value['contract'].update(jobs=2, evaluation_policy='each fixed-final cell immediately; two-slot coordinator')
    value['plan_sha256'] = digest(value)
    return value


def load_plan(args):
    value = read(Path(args.outdir) / PLAN_NAME)
    unsigned = {k: v for k, v in value.items() if k != 'plan_sha256'}
    if value.get('plan_sha256') != digest(unsigned) or value != build_plan(args):
        raise ValueError('Source/runtime/config/action/plan changed; use original checkout or a NEW output')
    return value


def dataset(cfg, split):
    from data_utils.feeder import PersonInWiFi3D
    import run_piw_hpeli_clean as base
    result = PersonInWiFi3D(split, cfg['dataset_root'], 'one-person', 1)
    ids = [item['name'] for item in result.items]
    name = 'train_data' if split == 'training' else 'test_data'
    expected = base.EXPECTED_IDENTITIES[name]
    actual_hash = __import__('hashlib').sha256(('\n'.join(ids) + '\n').encode()).hexdigest()
    if (len(result) != COUNTS[split] or len(set(ids)) != len(ids) or actual_hash != expected[3]):
        raise ValueError('PiW official ordered identities/counts changed')
    return result, ids


def geometry_figure(path, truth, targets, action, *, source, predictions=None, ids=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    from attack.piw_action_targets import EDGES, specification
    path = Path(path)
    if path.exists() or path.with_suffix('.pdf').exists():
        raise FileExistsError('Existing plot retained: ' + str(path))
    # One clean-GT hip midpoint per example, display only, shared by all poses.
    roots = .5 * (truth[:, 5:6] + truth[:, 7:8])
    gt, tar = truth - roots, targets - roots[None]
    pred = None if predictions is None else predictions - roots[None]
    values = [gt.reshape(-1, 3), tar.reshape(-1, 3)]
    if pred is not None:
        values.append(pred.reshape(-1, 3))
    all_values = np.concatenate(values)
    center = .5 * (all_values.min(0) + all_values.max(0))
    radius = max(.1, .55 * float(np.ptp(all_values, axis=0).max()))
    fig = plt.figure(figsize=(8.4, 7.8))
    label = 'GT / TARGET GEOMETRY ONLY; no model predictions' if pred is None else 'ACTUAL FINAL199 MODEL PREDICTIONS'
    fig.suptitle(f'{action.upper()} — {source}\n{label}', fontsize=11)
    spec = specification(action)
    doses = [0., .5, 1.] if pred is None else [0., .6, 1.]
    for row in range(3):
        for col, dose in enumerate(doses):
            ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection='3d')
            layers = [(gt[row], '#999999', '-'), (tar[col, row], '#d9720a', '--')]
            if pred is not None:
                layers.append((pred[col, row], '#2166ac', '-'))
            for value, color, style in layers:
                for a, b in EDGES:
                    ax.plot(*value[[a, b]].T, color=color, ls=style, lw=1.1)
                ax.scatter(*value.T, c=color, s=18, edgecolors='black', linewidths=.3)
            ax.scatter(*tar[col, row, spec['pivots']].T, c='#c62728', s=28, edgecolors='black', linewidths=.4)
            ax.set(xlim=(center[0]-radius, center[0]+radius),
                ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius),
                xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(3))
                axis.label.set_fontsize(7)
                axis.labelpad = 0
            ax.set_box_aspect((1, 1, 1)); ax.set_proj_type('ortho'); ax.view_init(elev=15, azim=-65)
            ax.tick_params(labelsize=5, pad=0)
            ax.set_title(f'{ids[row] if ids is not None else row}: d={dose:g}', fontsize=7)
    legend = [Line2D([0], [0], color='#999999', label='True pose'),
              Line2D([0], [0], color='#d9720a', ls='--', label='Target')]
    if pred is not None:
        legend.append(Line2D([0], [0], color='#2166ac', label='Prediction'))
    fig.legend(handles=legend, loc='lower center', ncol=len(legend), frameon=False)
    fig.subplots_adjust(top=.86, bottom=.08, hspace=.22, wspace=.1)
    fig.savefig(path, dpi=180); fig.savefig(path.with_suffix('.pdf')); plt.close(fig)


def geometry(args):
    import numpy as np
    from attack.piw_action_targets import EDGES, target_pose, cross_endpoint
    from piw_action_target_data import ActionTargetPairs
    plan = build_plan(args)
    root = Path(plan['outdir'])
    write(root / PLAN_NAME, plan)
    if (root / 'geometry.json').exists():
        raise ValueError('Geometry already exists; inspect it, then Prepare; no overwrite')
    records, files, failures = {}, {}, []
    for cell in plan['cells']:
        cfg, action = cell['cfg'], cell['action']
        base, ids = dataset(cfg, 'training')  # no TEST labels or CSI for geometry design
        pairs = ActionTargetPairs(base, None, action)
        selected = sorted(pairs.selected)
        chosen = [selected[0], selected[len(selected)//2], selected[-1]]
        report, bad, pose_files = audit_action_geometry(selected, pairs.clean_pose, ids,
            cfg['payload'], EDGES, target_pose, cross_endpoint, action,
            pose_hash=lambda index: sha(base.items[index]['kpt']), progress_every=500)
        records[action] = dict(report, train_ids_sha256=digest(ids), poison=pairs.manifest(),
            pose_file_sha256=pose_files, fixed_train_indices=chosen)
        if bad:
            failures.extend(dict(entry, action=action) for entry in bad)
            write(root / f'geometry_failure_{action}.json',
                dict(index=bad[0]['index'], name=bad[0]['identity'], error=bad[0]['error']))
            continue
        gt = np.stack([pairs.clean_pose(i)[0] for i in chosen])
        targets = np.stack([target_pose(gt, d, cfg['payload']) for d in (0., .5, 1.)])
        path = root / f'geometry_{action}.npz'
        write_npz(path, gt=gt, targets=targets, doses=np.array([0., .5, 1.]),
            indices=np.array(chosen), pair_ids=np.array([ids[i] for i in chosen]))
        picture = root / f'geometry_{action}.png'
        geometry_figure(picture, gt, targets, action, source='REAL TRAIN; geometry inspection', ids=[ids[i] for i in chosen])
        for artifact in (path, picture, picture.with_suffix('.pdf')):
            files[artifact.name] = sha(artifact)
    audit_sha = finish_geometry_audit(root, plan, records, failures)
    write(root / 'geometry.json', dict(plan_sha256=plan['plan_sha256'], files=files,
        source='TRAIN_ONLY_NO_MODEL_PREDICTIONS', records=records, audit_file_sha256=audit_sha))
    print(f'GEOMETRY_ONLY: inspect {root}/geometry_cross.png and geometry_nod.png.\n'
          'No training; Prepare requires explicit --approve-geometry.', flush=True)


def verify_geometry(plan):
    root = Path(plan['outdir'])
    record = read(root / 'geometry.json')
    if record['plan_sha256'] != plan['plan_sha256']:
        raise ValueError('Geometry/plan mismatch')
    verify_geometry_audit(root, plan, record)
    for name, expected in record['files'].items():
        if sha(root / name) != expected:
            raise ValueError('Geometry artifact changed: ' + name)
    for cell in plan['cells']:
        base, ids = dataset(cell['cfg'], 'training')
        entry = record['records'][cell['action']]
        if digest(ids) != entry['train_ids_sha256']:
            raise ValueError('TRAIN identity order changed after geometry inspection')
        for index, expected in entry['pose_file_sha256'].items():
            if sha(base.items[int(index)]['kpt']) != expected:
                raise ValueError('TRAIN pose changed after geometry inspection')
    return record


def execution():
    import torch
    import run_piw_hpeli_clean as base
    result = base._setup_execution('cuda:0')
    if '4090' not in result['gpu_name'] or result['gpu_total_memory'] < 22 * 2**30:
        raise ValueError('Expected physical RTX4090/24GiB; no device fallback')
    torch.set_num_threads(2)
    return result


def seed_all(device):
    import numpy as np
    import torch
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if device.startswith('cuda:'):
        torch.cuda.manual_seed(42)


def optimizer_for(model, cfg):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'],
        betas=tuple(cfg['betas']), eps=cfg['eps'])


def pairs_for(plan, cell):
    from piw_action_target_data import ActionTargetPairs
    from piw_a1_adapter import build_trigger
    if trigger_record(cell['cfg']['action_npy']) != cell['trigger_record']:
        raise ValueError('Native PiW trigger changed')
    base, _ = dataset(cell['cfg'], 'training')
    pairs = ActionTargetPairs(base, build_trigger(cell['cfg']['action_npy']), cell['action'])
    if pairs.manifest() != read(Path(plan['outdir']) / 'geometry.json')['records'][cell['action']]['poison']:
        raise ValueError('Poison identities/doses/payload changed')
    return pairs


def smoke(plan, cell, rendezvous=None):
    import torch
    from torch.utils.data import DataLoader, Subset
    import run_piw_hpeli_clean as base
    from run_piw_hpeli_lr_control import step
    device = cell['cfg']['device']
    seed_all(device)
    pairs = pairs_for(plan, cell)
    indices = sorted(pairs.selected)[:8] + [i for i in range(len(pairs)) if i not in pairs.selected][:24]
    batch = next(iter(DataLoader(Subset(pairs, indices), batch_size=32, num_workers=0)))
    model = base.build_model().to(device).train()
    optimizer = optimizer_for(model, cell['cfg'])
    if rendezvous is not None:
        rendezvous('allocated')
    loss, count = step(model, optimizer, batch, device)
    if count != 32 or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite/incomplete B32 real TRAIN smoke')
    if rendezvous is not None:
        torch.cuda.synchronize()
        rendezvous('updated')
    del model, optimizer, batch, pairs
    gc.collect(); torch.cuda.empty_cache()
    return dict(batch=32, poisoned=8, finite_loss=loss, weights='discarded; not a training checkpoint')


def prepare(args, plan, *, idle_check=None):
    if not args.approve_geometry:
        raise ValueError('Inspect both TRAIN geometry plots, then explicitly pass --approve-geometry')
    root = Path(plan['outdir'])
    if (root / 'prepared.json').exists():
        raise ValueError('Already prepared; no repeated smoke; use Launch/Resume')
    verify_geometry(plan)
    from run_mmfi_p3_s3_three_attacks_4090 import require_idle
    import run_piw_hpeli_clean as base
    (idle_check or require_idle)()
    inputs, _ = base.audit_inputs(plan['cells'][0]['cfg'])
    write(root / 'inputs.json', inputs)
    context = execution()
    smokes = {cell['action']: smoke(plan, cell) for cell in plan['cells']}
    verify_geometry(plan)
    write(root / 'prepared.json', dict(plan_sha256=plan['plan_sha256'],
        inputs_sha256=sha(root / 'inputs.json'), geometry_sha256=sha(root / 'geometry.json'),
        execution=context, smokes=smokes, geometry_approval='explicit --approve-geometry'))
    print('PREPARED_ONLY: full input audit + real CUDA B32 smokes; weights discarded. Launch separately.', flush=True)


def verify_prepared(plan, full_hash=False):
    root = Path(plan['outdir'])
    record = read(root / 'prepared.json')
    if (record['plan_sha256'] != plan['plan_sha256'] or record['inputs_sha256'] != sha(root / 'inputs.json')
            or record['geometry_sha256'] != sha(root / 'geometry.json')):
        raise ValueError('Prepared input/geometry/plan binding changed')
    verify_geometry(plan)
    if full_hash:
        import run_piw_hpeli_clean as base
        current, _ = base.audit_inputs(plan['cells'][0]['cfg'])
        if current != read(root / 'inputs.json'):
            raise ValueError('Full PiW input bytes/identities changed')
    return record


def checkpoint(path, cfg, plan, final=False):
    import torch
    record = torch.load(path, map_location='cpu', weights_only=False)
    if (record.get('profile') != PROFILE or record.get('plan_sha256') != plan['plan_sha256']
            or record.get('cfg') != cfg or record.get('inputs_sha256') != sha(Path(plan['outdir']) / 'inputs.json')
            or record.get('execution') != read(Path(plan['outdir']) / 'prepared.json')['execution']
            or type(record.get('epoch')) is not int or not 0 <= record['epoch'] < cfg['epochs']
            or any(k not in record for k in ('model', 'optimizer', 'rng'))):
        raise ValueError('Foreign/incomplete checkpoint; no scratch fallback')
    if final and record['epoch'] != cfg['final_epoch']:
        raise ValueError('Evaluation requires the selected action fixed final199')
    return record


def train_epochs(model, optimizer, loader, cfg, path, plan, context, *, device, start=0, stop=None, lease_check=None):
    """PiW ordinary MPJPE/AdamW; private stop hook is CPU-test-only."""
    import run_piw_hpeli_clean as base
    from run_piw_hpeli_lr_control import step
    stop = cfg['epochs'] if stop is None else stop
    for epoch in range(start, stop):
        if lease_check is not None:
            lease_check()
        model.train()
        total, count, began = 0., 0, time.monotonic()
        heartbeat = began
        for index, batch in enumerate(loader):
            loss, n = step(model, optimizer, batch, device)
            total += loss * n; count += n
            if index % 250 == 0 or time.monotonic() - heartbeat >= 30:
                print(f'[{cfg["action"]}] epoch{epoch}/{cfg["final_epoch"]} step{index}/{len(loader)} loss={loss:.7f}', flush=True)
                heartbeat = time.monotonic()
        expected = len(loader.dataset) // cfg['batch_size'] * cfg['batch_size']
        if not count or count != expected:
            raise ValueError('Training coverage differs from drop_last=True recipe')
        base.atomic_checkpoint(path, dict(profile=PROFILE, plan_sha256=plan['plan_sha256'],
            cfg=cfg, inputs_sha256=sha(Path(plan['outdir']) / 'inputs.json'),
            execution=context, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
            rng=base._rng_state(device), train_loss=total/count))
        print(f'[{cfg["action"]}] epoch{epoch}/{cfg["final_epoch"]}: MPJPE={total/count:.7f} seconds={time.monotonic()-began:.1f}', flush=True)


def train_cell(plan, cell, resume=False, lease_check=None):
    import torch
    from torch.utils.data import DataLoader
    import run_piw_hpeli_clean as base
    folder, cfg = Path(plan['outdir']) / cell['action'], cell['cfg']
    folder.mkdir(exist_ok=True)
    path = folder / 'checkpoint.pt'
    if path.exists() and not resume:
        raise ValueError('Checkpoint exists; use explicit Resume')
    context = execution()
    if context != read(Path(plan['outdir']) / 'prepared.json')['execution']:
        raise ValueError('CUDA backend changed after Prepare')
    device = cfg['device']
    seed_all(device)
    pairs = pairs_for(plan, cell)
    write(folder / 'config.resolved.json', cfg); write(folder / 'poison_manifest.json', pairs.manifest())
    loader = DataLoader(pairs, batch_size=32, num_workers=0, shuffle=True, drop_last=True)
    model = base.build_model().to(device)
    optimizer = optimizer_for(model, cfg)
    start = 0
    if path.exists():
        saved = checkpoint(path, cfg, plan)
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        base._restore_rng(saved['rng'], device); start = saved['epoch'] + 1
        del saved
    train_epochs(model, optimizer, loader, cfg, path, plan, context, device=device, start=start, lease_check=lease_check)
    del loader, pairs, model, optimizer
    gc.collect(); torch.cuda.empty_cache()


def metrics_from_arrays(gt, targets, predictions, joints):
    import numpy as np
    from eval.metrics import pa_mpjpe
    gt, targets, predictions = [np.array(x, dtype=np.float64, order='C', copy=True)
                               for x in (gt, targets, predictions)]
    if (gt.shape != (len(gt), 14, 3) or not len(gt)
            or targets.shape != (6, len(gt), 14, 3) or predictions.shape != targets.shape
            or any(not np.isfinite(x).all() for x in (gt, targets, predictions))
            or not joints or len(set(joints)) != len(joints) or any(j not in range(14) for j in joints)):
        raise ValueError('Invalid PiW full pose arrays/target joints')
    np.testing.assert_array_equal(targets[0], gt)
    clean = predictions[0]
    scale = np.linalg.norm(gt[:, 6]-gt[:, 4], axis=-1)  # CLEAN GT elbows; never target elbows
    if np.any(scale <= 1e-8):
        raise ValueError('Degenerate PiW clean-GT PCK reference (6,4)')
    others = [j for j in range(14) if j not in joints]
    rows = []
    for k, dose in enumerate(GRID):
        pred, target = predictions[k], targets[k]
        error = np.linalg.norm(pred[:, joints]-target[:, joints], axis=-1)
        control = np.linalg.norm(clean[:, joints]-target[:, joints], axis=-1)
        requested, induced = target[:, joints]-gt[:, joints], pred[:, joints]-clean[:, joints]
        rows.append(dict(dose=dose, t_mpjpe_mm=float(error.mean()*1000),
            target_pck_pct=float((error <= .2*scale[:, None]).mean()*100),
            no_trigger_t_mpjpe_mm=float(control.mean()*1000),
            no_trigger_target_pck_pct=float((control <= .2*scale[:, None]).mean()*100),
            requested_mm=float(np.linalg.norm(requested, axis=-1).mean()*1000),
            predicted_mm=float(np.linalg.norm(induced, axis=-1).mean()*1000),
            dce_mm=float(np.linalg.norm(induced-requested, axis=-1).mean()*1000),
            triggered_gt_target_mm=float(np.linalg.norm(pred[:, joints]-gt[:, joints], axis=-1).mean()*1000),
            nontarget_drift_mm=float(np.linalg.norm(pred[:, others]-clean[:, others], axis=-1).mean()*1000)))
    return dict(clean_mpjpe_mm=float(np.linalg.norm(clean-gt, axis=-1).mean()*1000),
        clean_pampjpe_mm=float(pa_mpjpe(clean.copy(), gt.copy()).mean()*1000),
        clean_target_mpjpe_mm=float(np.linalg.norm(clean[:, joints]-gt[:, joints], axis=-1).mean()*1000),
        mean_positive_dce_mm=float(np.mean([r['dce_mm'] for r in rows[1:]])), doses=rows)


def evaluate_cell(plan, cell):
    import numpy as np
    import torch
    import run_piw_hpeli_clean as base
    from run_piw_hpeli_lr_control import infer
    from attack.piw_action_targets import target_pose
    from piw_a1_adapter import build_trigger
    cfg, root = cell['cfg'], Path(plan['outdir'])
    folder = root / cell['action']
    if (folder / 'complete.json').exists():
        return verify_complete(plan, cell)
    if execution() != read(root / 'prepared.json')['execution']:
        raise ValueError('Evaluation backend changed')
    saved = checkpoint(folder / 'checkpoint.pt', cfg, plan, final=True)
    model = base.build_model().to(cfg['device'])
    model.load_state_dict(saved['model'], strict=True); del saved
    ds, ids = dataset(cfg, 'validation')
    trigger = build_trigger(cfg['action_npy'])
    if trigger_record(cfg['action_npy']) != cell['trigger_record']:
        raise ValueError('Evaluation trigger changed')
    predictions, gt = [], None
    for dose in GRID:
        pred, truth, names = infer(model, ds, cfg, trigger, dose)
        if names != ids:
            raise ValueError('Evaluation identity order changed')
        if gt is None:
            gt = truth
        else:
            np.testing.assert_array_equal(gt, truth)
        predictions.append(pred)
        print(f'[eval] {cell["action"]} d={dose:g}: {len(truth)} TEST frames', flush=True)
    predictions = np.stack(predictions)
    targets = np.stack([target_pose(gt, dose, cfg['payload']) for dose in GRID])
    metrics = metrics_from_arrays(gt, targets, predictions, cfg['payload']['target_joints'])
    artifact = folder / 'poses_full_test.npz'
    write_npz(artifact, gt=gt, target=targets, prediction=predictions, doses=np.array(GRID),
        target_joints=np.array(cfg['payload']['target_joints']), pair_ids=np.array(ids),
        fixed_test_indices=np.array(FIXED_TEST), plan_sha256=np.array(plan['plan_sha256']))
    with np.load(artifact, allow_pickle=False) as cache:
        if metrics != metrics_from_arrays(cache['gt'], cache['target'], cache['prediction'], cfg['payload']['target_joints']):
            raise ValueError('NPZ replay failed to reproduce metrics exactly')
    picture = folder / 'qualitative_fixed_test.png'
    columns = [0, 3, 5]
    geometry_figure(picture, gt[FIXED_TEST], targets[columns][:, FIXED_TEST], cell['action'],
        source='REAL TEST; fixed official identities', predictions=predictions[columns][:, FIXED_TEST],
        ids=[ids[i] for i in FIXED_TEST])
    report = dict(profile=PROFILE, action=cell['action'], plan_sha256=plan['plan_sha256'],
        payload=cfg['payload'], metrics=metrics, final_epoch=199,
        clean_trained_trigger_control='NOT_SUPPLIED', pck_reference='clean GT elbows6,4',
        artifacts={p.name: sha(p) for p in (artifact, picture, picture.with_suffix('.pdf'), folder / 'checkpoint.pt')})
    verify_prepared(plan, full_hash=True)
    write(folder / 'results.json', report)
    write(folder / 'complete.json', dict(plan_sha256=plan['plan_sha256'], results_sha256=sha(folder / 'results.json')))
    del model, ds, predictions, targets
    gc.collect(); torch.cuda.empty_cache()
    return report


def verify_complete(plan, cell):
    folder = Path(plan['outdir']) / cell['action']
    if read(folder / 'complete.json') != dict(plan_sha256=plan['plan_sha256'], results_sha256=sha(folder / 'results.json')):
        raise ValueError('Completion marker differs')
    report = read(folder / 'results.json')
    if report['plan_sha256'] != plan['plan_sha256'] or report['payload'] != cell['cfg']['payload']:
        raise ValueError('Evaluation payload changed')
    for name, expected in report['artifacts'].items():
        if sha(folder / name) != expected:
            raise ValueError('Completed artifact changed: ' + name)
    return report


def export(plan):
    rows = []
    for cell in plan['cells']:
        m = verify_complete(plan, cell)['metrics']
        rows.append(dict(action=cell['action'], target_joints=json.dumps(cell['cfg']['payload']['target_joints']),
            t_mpjpe_mm=m['doses'][-1]['t_mpjpe_mm'], target_pck_pct=m['doses'][-1]['target_pck_pct'],
            mean_positive_dce_mm=m['mean_positive_dce_mm'], clean_mpjpe_mm=m['clean_mpjpe_mm'],
            clean_pampjpe_mm=m['clean_pampjpe_mm']))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    path = Path(plan['outdir']) / 'piw_action_targets_results.csv'
    contents = buffer.getvalue().encode('utf-8')
    if path.exists() and path.read_bytes() != contents:
        raise ValueError('Existing summary differs; not overwritten')
    if not path.exists():
        with path.open('xb') as stream:
            stream.write(contents)
    print(buffer.getvalue(), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-home', required=True); p.add_argument('--outdir', required=True)
    p.add_argument('--dataset-root'); p.add_argument('--cross-action'); p.add_argument('--nod-action')
    p.add_argument('--approve-action-labels', action='store_true'); p.add_argument('--approve-geometry', action='store_true')
    p.add_argument('--mode', required=True, choices=['check-only', 'geometry', 'prepare', 'launch', 'resume', 'evaluate', 'status', 'export'])
    return p


def require_output_space(path):
    if shutil.disk_usage(path).free < 2 * 2**30:
        raise ValueError('Need at least2GiB free; no automatic cleanup')


def main(argv=None):
    args = parser().parse_args(argv)
    verify_bundle()
    if args.mode == 'check-only':
        return check_paths(args)
    if args.mode == 'status':
        for action in ('cross', 'nod'):
            folder = Path(args.outdir) / action
            print(f'{action}: checkpoint={(folder/"checkpoint.pt").is_file()}, complete={(folder/"complete.json").is_file()} (presence only)')
        return
    from run_mmfi_tables import _cell_lock
    if args.mode == 'geometry':
        build_plan(args)  # refuse unrelated outputs before creating the lock
        with _cell_lock(args.outdir):
            return geometry(args)
    plan = load_plan(args)
    with _cell_lock(args.outdir):
        if args.mode == 'export':
            verify_prepared(plan)
            return export(plan)
        # Same new-action lock as MM-Fi prevents check-then-launch races
        # between these two helpers. Never remove a foreign lock or job.
        with _cell_lock(Path(args.data_home) / 'runs/.mmfi_action_targets_gpu0'):
            if args.mode == 'prepare':
                return prepare(args, plan)
            from run_mmfi_p3_s3_three_attacks_4090 import require_idle
            require_idle()
            receipt = verify_prepared(plan, full_hash=True)
            if execution() != receipt['execution']:
                raise ValueError('CUDA backend differs from Prepare')
            require_output_space(args.outdir)
            logs = Path(args.outdir) / 'logs'; logs.mkdir(exist_ok=True)
            with (logs / f'{args.mode}.{time.time_ns()}.log').open('x', encoding='utf-8') as stream:
                previous = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = Tee(previous[0], stream), Tee(previous[1], stream)
                try:
                    if args.mode != 'evaluate':
                        for cell in plan['cells']:
                            if (Path(args.outdir) / cell['action'] / 'complete.json').exists():
                                verify_complete(plan, cell)
                                continue
                            train_cell(plan, cell, resume=args.mode == 'resume')
                    for cell in plan['cells']:  # TEST deferred until BOTH fixed-final victims exist
                        saved = checkpoint(Path(args.outdir) / cell['action'] / 'checkpoint.pt', cell['cfg'], plan, final=True)
                        del saved
                    for cell in plan['cells']:
                        evaluate_cell(plan, cell)
                    export(plan)
                finally:
                    sys.stdout, sys.stderr = previous


if __name__ == '__main__':
    main()
