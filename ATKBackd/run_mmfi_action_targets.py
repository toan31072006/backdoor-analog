"""Prepare new Cross/Nod targets for MM-Fi P1S1 HPE-Li seed42.

Explicit stages: check-only -> geometry (TRAIN only) -> prepare with geometry
approval -> launch. Cross/Nod train serially, each from scratch for50epochs.
No historical checkpoint is imported. No auto launch, retry, process kill,
batch shrink, early stopping or TEST selection. These are exploratory new
target-action runs, not replacements for motion-prior Table2 or full T7.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import sys
import time
from importlib.metadata import version

os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', PYTHONUTF8='1')
HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_p1s1_hpeli_cross_nod_targets_s42_v1'
COUNTS = {'train': 133056, 'eval': 33264}
GRID = [0., .2, .4, .6, .8, 1.]
FIXED_TEST = [0, 16632, 33263]
PLAN_NAME = 'action_targets.plan.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, immutable=True):
    path = Path(path)
    if path.exists():
        if immutable:
            if read(path) != value:
                raise ValueError(f'Existing record differs; preserve it: {path}')
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.saving')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def write_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Existing artifact retained: {path}')
    temporary = path.with_name(path.name + '.saving')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def source_hashes():
    result = {}
    for path in HERE.rglob('*'):
        relative = path.relative_to(HERE)
        if any(part.startswith('.') or part in ('tests', '__pycache__', 'runs', 'envs')
               for part in relative.parts):
            continue
        if path.is_file() and path.suffix in ('.py', '.json', '.yaml', '.yml'):
            result[relative.as_posix()] = sha(path)
    return result


def check_paths(args):
    """Read-only and standard-library-only: no Torch, CUDA or output writes."""
    required = {
        'dataset': Path(args.data_home) / 'datasets/Compress',
        'bend reference': Path(args.data_home) / 'actions/data_bend.npy',
        'cross prior': Path(args.cross_action or Path(args.data_home) / 'actions/data_cross.npy'),
        'nod prior': Path(args.nod_action or Path(args.data_home) / 'actions/data_nod.npy'),
    }
    missing = []
    for name, path in required.items():
        ok = path.is_dir() if name == 'dataset' else path.is_file()
        print(f'{name}: {"OK" if ok else "MISSING"} {path.resolve()}', flush=True)
        if not ok:
            missing.append(name)
    for relative in ('train_backdoor.py', 'attack/action_targets.py',
                     'attack/action_target_data.py', 'models/hpeli.py'):
        if not (HERE / relative).is_file():
            missing.append('source: ' + relative)
    print(f'Output: {Path(args.outdir).resolve()}\nNo training/inference performed.', flush=True)
    if missing:
        raise FileNotFoundError('Missing: ' + ', '.join(missing))


def verify_bundle():
    """Optional delivered code manifest, without importing scientific packages."""
    manifest_path = HERE.parent / 'CROSS_NOD_CODE_MANIFEST.json'
    if not manifest_path.exists():
        return
    for name, record in read(manifest_path).items():
        path = (HERE.parent / name).resolve()
        if (not path.is_relative_to(HERE.parent) or not path.is_file()
                or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']):
            raise ValueError('Code bundle missing/changed/unsafe member: ' + name)


def configuration(home, action, action_file):
    from run_a1_three_tables import _configuration
    from train_backdoor import _resolve_training_config, _validate_training_contract
    from attack.action_targets import specification
    cfg = _configuration(Path(home), 'cuda:0', 4, 42, 'a1')
    spec = specification(action)
    cfg.update(confirmation_profile=PROFILE, action_target_name=action,
        action_payload=spec, action_npy=str(Path(action_file).resolve()),
        pivot=spec['pivots'][0], target_joints=spec['target_joints'],
        payload_axis=[0., 0., 1.], theta_max_deg=40., ckpt_every=1)
    # pivot/axis/theta remain inert compatibility fields for the native recipe;
    # the complete new action_payload, not those fields, defines labels/metrics.
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    return cfg


def build_plan(args):
    from run_mmfi_a1_motion_priors_4090 import action_record, trigger_record
    root, home = Path(args.outdir).resolve(), Path(args.data_home).resolve()
    if (root == Path(root.anchor) or root == home or root == home / 'runs'
            or not re.fullmatch(r'mmfi_action_targets_cross_nod_s42_v[0-9]+', root.name)
            or root.parent.name.lower() != 'runs'):
        raise ValueError('Use a NEW dedicated output runs/mmfi_action_targets_cross_nod_s42_v1 (or v2, v3, ...)')
    if root.is_relative_to(HERE) or home.is_relative_to(root):
        raise ValueError('Output must not contain data home or lie inside scientific source')
    if (root.exists() and any(p.name != 'run.lock' for p in root.iterdir())
            and not (root / PLAN_NAME).is_file()):
        raise ValueError('Nonempty output has no matching plan; nothing will be overwritten')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''):
        raise ValueError('No CUDA_VISIBLE_DEVICES remapping in this profile')
    cells = []
    for action in ('cross', 'nod'):
        path = getattr(args, action + '_action') or home / 'actions' / f'data_{action}.npy'
        cfg = configuration(home, action, path)
        cells.append(dict(action=action, cfg=cfg,
            action_record=action_record(path, action, args.approve_action_labels),
            trigger_arrays=trigger_record(cfg)))
    if cells[0]['action_record']['sha256'] == cells[1]['action_record']['sha256']:
        raise ValueError('Cross and Nod must not reuse identical motion-prior bytes')
    value = dict(schema=1, profile=PROFILE, data_home=str(home), outdir=str(root),
        cells=cells, sources=source_hashes(),
        runtime=dict(python=platform.python_version(), executable=str(Path(sys.executable).resolve()),
            packages={key: version(key) for key in ('numpy', 'torch', 'scipy', 'PyYAML', 'matplotlib')}),
        contract=dict(scope='EXPLORATORY_NEW_TARGET_ACTIONS_NOT_COMPLETE_T7',
            dataset='MM-Fi', protocol='protocol1', setting='s1', split_seed=0,
            counts=COUNTS, seed=42, rho=.1, n_poison=13305, doses=GRID,
            epochs=50, final_epoch=49, physical_batch=32, workers=4,
            optimizer='SGD lr=.001 momentum=.9 wd0', victim_loss='all-joint MPJPE',
            training_doses='fixed paired U(.2,1)', scratch=True, pretrained=False,
            trigger='native md_multicarrier', eps=.185, jobs=1,
            pck_threshold=.2, pck_clean_gt_reference=[5, 12],
            test_indices=FIXED_TEST, uncertainty='single seed; no seed variability estimated',
            limitations=['Cross is two-bone IK + bone-direction interpolation, not a rigid subtree.',
                'No rigid-payload near-linearity/monotonicity guarantee is claimed for Cross.',
                'Bone lengths do not prove joint-angle limits, collision/contact consistency or action recognition.',
                'No new Clean/baseline/defense training; clean-trained+trigger control not supplied.',
                'Old bend motion-prior/leg-rotation result is not a trunk-bend target.',
                'Do not replace or merge historical motion-prior table rows with these target-action runs.']))
    coordinator = getattr(args, 'coordinator', None)
    if coordinator is not None:
        value['coordinator'] = coordinator
        value['contract'].update(jobs=2, evaluation_policy='each fixed-final cell immediately; two-slot coordinator')
    value['plan_sha256'] = digest(value)
    return value


def load_existing_plan(args):
    plan = read(Path(args.outdir) / PLAN_NAME)
    if (plan['profile'] != PROFILE or plan['data_home'] != str(Path(args.data_home).resolve())
            or plan['outdir'] != str(Path(args.outdir).resolve())):
        raise ValueError('Foreign plan or paths')
    check = dict(plan)
    claimed = check.pop('plan_sha256')
    if digest(check) != claimed or source_hashes() != plan['sources']:
        raise ValueError('Plan/scientific sources changed; original checkout or new output required')
    expected = build_plan(args)
    if expected != plan:
        raise ValueError('Configuration/action/runtime changed after geometry inspection')
    return plan


def dataset(cfg, split):
    from train_backdoor import _load_dataset
    from run_peak_confirmation import _pair_ids
    base = _load_dataset(cfg, split)
    ids = _pair_ids(base)
    key = 'train' if split == 'training' else 'eval'
    if len(base) != COUNTS[key] or len(set(map(tuple, ids))) != len(ids):
        raise ValueError(f'Incomplete/duplicate {split} split: {len(base)}')
    return base, ids


def action_data(base, trigger, cfg, **kwargs):
    from attack.action_target_data import ActionTargetDataset
    return ActionTargetDataset(base, trigger, cfg, **kwargs)


def geometry_figure(path, truth, targets, action, *, source='REAL TRAIN'):
    if path.exists():
        raise FileExistsError(path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from attack.payload import MMFI_EDGES
    # One common GT-pelvis translation per example, never fit to predictions.
    # This affects display only; saved targets and training labels stay absolute.
    roots = truth[:, 0:1, :]
    truth, targets = truth - roots, targets - roots[None]
    values = np.concatenate([truth.reshape(-1, 3), targets.reshape(-1, 3)])
    low, high = values.min(0), values.max(0)
    # World coordinate axes are shown; no assumption that a global axis is up.
    fig = plt.figure(figsize=(8.4, 7.8))
    fig.suptitle(f'{action.upper()}: {source} GT / TARGET GEOMETRY ONLY\n'
                 'No model predictions; GT-root centered for display only', fontsize=11)
    center = .5 * (low + high)
    radius = .55 * float(np.max(high - low))
    for row in range(3):
        for col, dose in enumerate((0., .5, 1.)):
            ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection='3d')
            for a, b in MMFI_EDGES:
                ax.plot(*truth[row, [a, b]].T, color='#999999', lw=1)
                ax.plot(*targets[col, row, [a, b]].T, color='#d9720a', lw=1.2, ls='--')
            ax.scatter(*targets[col, row].T, s=13, c='#d9720a', edgecolors='black', linewidths=.3)
            ax.set(xlim=(center[0]-radius, center[0]+radius),
                   ylim=(center[1]-radius, center[1]+radius),
                   zlim=(center[2]-radius, center[2]+radius),
                   xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
            ax.set_box_aspect((1, 1, 1))
            ax.set_proj_type('ortho')
            ax.view_init(elev=15, azim=-65)
            ax.set_title(f'TRAIN example {row+1}, d={dose:g}', fontsize=8)
            ax.tick_params(labelsize=5)
    fig.subplots_adjust(top=.86, hspace=.22, wspace=.10)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def new_cross_reach_audit():
    return dict(poses=0, arms=0, adapted_poses=0, adapted_arms=0,
        max_goal_correction_m=0., max_correction_train_index=None,
        min_opposite_side_m=None, min_front_m=None,
        moved_shoulders=0, old_fixed_shoulder_non_crossable_arms=0,
        max_shoulder_displacement_m=0., max_shoulder_rotation_deg=0.)


def add_cross_reach_records(audit, index, records):
    audit['poses'] += 1
    audit['adapted_poses'] += int(any(r['adapted'] for r in records))
    for record in records:
        audit['arms'] += 1
        audit['adapted_arms'] += int(record['adapted'])
        audit['moved_shoulders'] += int(record.get('shoulder_displacement_m', 0.) > 1e-12)
        audit['old_fixed_shoulder_non_crossable_arms'] += int(record.get('old_fixed_shoulder_non_crossable', False))
        for key in ('shoulder_displacement_m', 'shoulder_rotation_deg'):
            audit['max_'+key] = max(audit['max_'+key], record.get(key, 0.))
        if record['goal_correction_m'] > audit['max_goal_correction_m']:
            audit['max_goal_correction_m'] = record['goal_correction_m']
            audit['max_correction_train_index'] = index
        for key in ('opposite_side_m', 'front_m'):
            name = 'min_' + key
            audit[name] = record[key] if audit[name] is None else min(audit[name], record[key])


def audit_action_geometry(indices, clean_pose, ids, spec, edges, target_fn,
                          endpoint_fn, action, *, pose_hash=None, progress_every=2000):
    """Check EVERY selected TRAIN pose; collect geometry failures, drop none.

    Unexpected I/O/programming failures still abort immediately. Geometric
    ValueError/AssertionError entries are collected; any one blocks Prepare.
    This loop never trains, opens TEST, edits labels or chooses new poison IDs.
    """
    import numpy as np
    reach, failures, pose_files = new_cross_reach_audit(), [], {}
    worst, decreasing = 0., 0
    for count, index in enumerate(indices):
        if count % progress_every == 0:
            print(f'[geometry] {action}: {count}/{len(indices)} poison TRAIN poses; '
                  f'geometry failures={len(failures)}', flush=True)
        try:
            truth = np.array(clean_pose(index)[0], dtype=np.float64, copy=True)
            originals = np.array([np.linalg.norm(truth[a]-truth[b]) for a, b in edges])
            other = [j for j in range(len(truth)) if j not in spec['target_joints']]
            previous, local_decreasing, local_worst = np.zeros(len(spec['target_joints'])), 0, 0.
            arm_records = []
            if action == 'cross':
                endpoint_fn(truth, spec, diagnostics=arm_records)
            for dose in GRID:
                target = target_fn(truth, dose, spec)
                lengths = np.array([np.linalg.norm(target[a]-target[b]) for a, b in edges])
                local_worst = max(local_worst, float(np.max(np.abs(lengths-originals))))
                np.testing.assert_array_equal(target[other], truth[other])
                moved = np.linalg.norm(target[spec['target_joints']]-truth[spec['target_joints']], axis=-1)
                local_decreasing += int(np.any(moved+1e-9 < previous))
                previous = moved
            if local_worst > 1e-8:
                raise ValueError(f'Bone-length preservation failed: {local_worst:.9g}m >1e-8m')
            worst = max(worst, local_worst)
            decreasing += local_decreasing
            if action == 'cross':
                add_cross_reach_records(reach, index, arm_records)
        except (ValueError, AssertionError) as exc:
            failures.append(dict(index=index, identity=ids[index], error=str(exc)))
        if pose_hash is not None:
            pose_files[str(index)] = pose_hash(index)
    record = dict(n_checked=len(indices), n_failed=len(failures),
        max_bone_delta_m=worst, decreasing_joint_displacement_steps=decreasing)
    if action == 'cross':
        record['reach_adaptation'] = reach
        print('[geometry] Cross v3 reach/shoulder audit: '+json.dumps(reach), flush=True)
    return record, failures, pose_files


class GeometryAuditFailure(ValueError):
    """Complete TRAIN audit failed; distinct from interrupted I/O/code failures."""


def finish_geometry_audit(root, plan, records, failures):
    audit = dict(schema=1, plan_sha256=plan['plan_sha256'],
        source='ALL_POISON_TRAIN_ONLY_NO_MODEL_NO_DROPPED_SAMPLES',
        status='FAILED' if failures else 'PASS', records=records,
        n_failed=len(failures), n_checked=sum(r['n_checked'] for r in records.values()))
    write(root / 'geometry_audit.json', audit)
    if failures:
        write(root / 'geometry_failures.json', dict(plan_sha256=plan['plan_sha256'], failures=failures))
        raise GeometryAuditFailure(f'Completed poison TRAIN geometry audit: {len(failures)} invalid poses '
            f'of {audit["n_checked"]} action/pose checks; see {root}/geometry_failures.json. '
            'No samples dropped, no success receipt, no training started.')
    return sha(root / 'geometry_audit.json')


def verify_geometry_audit(root, plan, record):
    path = root / 'geometry_audit.json'
    if not path.is_file() or sha(path) != record.get('audit_file_sha256'):
        raise ValueError('Missing/changed complete TRAIN geometry audit')
    audit = read(path)
    if (audit.get('status') != 'PASS' or audit.get('n_failed') != 0
            or audit.get('plan_sha256') != plan['plan_sha256']):
        raise ValueError('TRAIN geometry audit did not fully pass this plan')


def geometry(args):
    import numpy as np
    from attack.action_targets import target_pose, cross_endpoint
    from attack.payload import MMFI_EDGES
    plan = build_plan(args)
    root = Path(plan['outdir'])
    write(root / PLAN_NAME, plan)
    if (root / 'geometry.json').exists():
        raise ValueError('Geometry already exists; inspect it, then Prepare. No overwrite.')
    records, files, failures = {}, {}, []
    for cell in plan['cells']:
        action, cfg = cell['action'], cell['cfg']
        base, ids = dataset(cfg, 'training')  # no TEST labels used for design
        data = action_data(base, None, cfg, cache_targets=False)
        indices = sorted(data.poison_idx)
        chosen = [indices[0], indices[len(indices)//2], indices[-1]]
        report, bad, _ = audit_action_geometry(indices, data.clean_pose, ids,
            cfg['action_payload'], MMFI_EDGES, target_pose, cross_endpoint, action)
        records[action] = dict(report, train_pair_ids_sha256=digest(ids), n_poison=data.n_poison,
            poison_plan_sha256=data.manifest()['poison_plan_sha256'], selected_train_indices=chosen,
            selection='first/middle/last sorted poison IDs, not score-selected')
        if bad:
            failures.extend(dict(entry, action=action) for entry in bad)
            write(root / f'geometry_failure_{action}.json',
                dict(index=bad[0]['index'], pair_id=bad[0]['identity'], error=bad[0]['error']))
            continue
        truth = np.stack([data.clean_pose(i)[0] for i in chosen])
        targets = np.stack([target_pose(truth, dose, cfg['action_payload']) for dose in (0., .5, 1.)])
        artifact = root / f'geometry_{action}.npz'
        write_npz(artifact, gt=truth, targets=targets, doses=np.array([0., .5, 1.]),
                  indices=np.array(chosen), pair_ids=np.array([json.dumps(ids[i]) for i in chosen]))
        picture = root / f'geometry_{action}.png'
        geometry_figure(picture, truth, targets, action)
        files[artifact.name], files[picture.name] = sha(artifact), sha(picture)
    audit_sha = finish_geometry_audit(root, plan, records, failures)
    write(root / 'geometry.json', dict(plan_sha256=plan['plan_sha256'],
        source='TRAIN_ONLY_NO_MODEL_PREDICTIONS', records=records, files=files,
        audit_file_sha256=audit_sha))
    print(f'GEOMETRY_ONLY: inspect {root}/geometry_cross.png and geometry_nod.png.\n'
          'Prepare requires your explicit --approve-geometry; no training has run.', flush=True)


def verify_geometry(plan):
    root = Path(plan['outdir'])
    record = read(root / 'geometry.json')
    if record['plan_sha256'] != plan['plan_sha256']:
        raise ValueError('Geometry belongs to a different plan')
    verify_geometry_audit(root, plan, record)
    for name, expected in record['files'].items():
        if sha(root / name) != expected:
            raise ValueError('Geometry artifact changed: ' + name)
    return record


def execution():
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; no CPU fallback')
    device = torch.cuda.get_device_properties(0)
    if '4090' not in device.name or device.total_memory < 22 * 2**30:
        raise ValueError('Expected RTX4090 with24GiB for this prepared profile')
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return dict(gpu_name=device.name, memory=device.total_memory,
        torch=torch.__version__, cuda=torch.version.cuda,
        deterministic=True, benchmark=False, amp=False,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32)


def seed_all():
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def smoke(plan, cell, rendezvous=None):
    import gc
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    import train_backdoor as native
    from attack.poison import collate
    cfg = cell['cfg']
    seed_all()
    base, _ = dataset(cfg, 'training')
    data = action_data(base, native.build_trigger(cfg), cfg)
    selected = sorted(data.poison_idx)[:8] + [i for i in range(len(data)) if i not in data.poison_idx][:24]
    loader = DataLoader(Subset(data, selected), batch_size=32, collate_fn=collate, num_workers=0)
    model = native.build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                               dataset='mmfi', pretrained=False).to('cuda:0')
    _, optimizer = native._build_optimizer(model, cfg, 'mmfi')
    model.train()
    batch = next(iter(loader))
    if rendezvous is not None:
        rendezvous('allocated')
    loss = native._victim_update(model, optimizer, batch['csi'].to('cuda:0'), batch['pose'].to('cuda:0'))
    if not np.isfinite(loss) or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite real B32 TRAIN capacity smoke')
    if rendezvous is not None:
        torch.cuda.synchronize()
        rendezvous('updated')
    del model, optimizer, batch, data, loader, base
    gc.collect()
    torch.cuda.empty_cache()
    return dict(batch=32, poisoned=8, loss=loss, weights='discarded, not used for training')


def prepare(args, plan, *, idle_check=None):
    if not args.approve_geometry:
        raise ValueError('Inspect both TRAIN geometry PNGs, then explicitly pass --approve-geometry')
    verify_geometry(plan)
    root = Path(plan['outdir'])
    if (root / 'prepared.json').exists():
        raise ValueError('Already prepared; use Launch or explicit Resume, not repeated Prepare')
    from run_mmfi_p3_s3_three_attacks_4090 import require_idle
    from vast_seed42_preflight import inspect
    (idle_check or require_idle)()  # No job is paused/killed or allowed through an old guard.
    if (root / 'inputs.json').exists():
        from vast_seed42_preflight import check_receipt
        inputs = read(root / 'inputs.json')
        check_receipt(inputs, full_hash=True)
    else:
        inputs = inspect(plan['data_home'])
        write(root / 'inputs.json', inputs)
    for cell in plan['cells']:
        record = read(root / 'geometry.json')['records'][cell['action']]
        if record['train_pair_ids_sha256'] != inputs['splits']['train']['pair_ids_sha256']:
            raise ValueError('TRAIN changed since geometry inspection')
    context = execution()
    smokes = {cell['action']: smoke(plan, cell) for cell in plan['cells']}
    write(root / 'prepared.json', dict(plan_sha256=plan['plan_sha256'],
        inputs_sha256=sha(root / 'inputs.json'), geometry_sha256=sha(root / 'geometry.json'),
        geometry_approval='explicit --approve-geometry after TRAIN-only inspection',
        execution=context, smokes=smokes))
    print('PREPARED_ONLY: real CUDA B32 smokes passed and discarded. Launch is a separate command.', flush=True)


def verify_prepared(plan, full_hash=False):
    from vast_seed42_preflight import check_receipt
    root = Path(plan['outdir'])
    record = read(root / 'prepared.json')
    if (record['plan_sha256'] != plan['plan_sha256']
            or record['inputs_sha256'] != sha(root / 'inputs.json')
            or record['geometry_sha256'] != sha(root / 'geometry.json')):
        raise ValueError('Prepared input/geometry/plan binding differs')
    verify_geometry(plan)
    check_receipt(read(root / 'inputs.json'), full_hash=full_hash)
    return record


def checkpoint(path, cfg, plan):
    import torch
    import train_backdoor as native
    record = torch.load(path, map_location='cpu', weights_only=False)
    if (record.get('profile') != PROFILE or record.get('plan_sha256') != plan['plan_sha256']
            or record.get('cfg_fingerprint') != native._config_fingerprint(cfg)
            or record.get('inputs_sha256') != sha(Path(plan['outdir']) / 'inputs.json')
            or record.get('execution') != read(Path(plan['outdir']) / 'prepared.json')['execution']
            or not isinstance(record.get('epoch'), int) or not 0 <= record['epoch'] < 50
            or not record.get('rng') or 'optimizer' not in record):
        raise ValueError('Foreign/incomplete checkpoint; no scratch fallback')
    return record


def save_checkpoint(path, model, optimizer, epoch, cfg, plan, context):
    import torch
    import train_backdoor as native
    record = dict(profile=PROFILE, plan_sha256=plan['plan_sha256'], cfg=cfg,
        cfg_fingerprint=native._config_fingerprint(cfg),
        inputs_sha256=sha(Path(plan['outdir']) / 'inputs.json'), execution=context,
        epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), rng=native._rng_state())
    temporary = path.with_name(path.name + '.saving')
    with temporary.open('xb') as stream:
        torch.save(record, stream)
    os.replace(temporary, path)


def train_epochs(model, optimizer, loader, cfg, path, plan, context, *,
                 device, start=0, stop=50, lease_check=None):
    """Ordinary native MPJPE/SGD loop; the private stop hook is for CPU tests."""
    import numpy as np
    import train_backdoor as native
    for epoch in range(start, stop):
        if lease_check is not None:
            lease_check()
        model.train()
        began, losses = time.monotonic(), []
        last_heartbeat = began
        for step, batch in enumerate(loader):
            loss = native._victim_update(model, optimizer, batch['csi'].to(device),
                                         batch['pose'].to(device))
            if not np.isfinite(loss):
                raise ValueError(f'Nonfinite loss at {cfg["action_target_name"]} epoch{epoch}/step{step}; no retry')
            losses.append(loss)
            if step % 500 == 0 or time.monotonic() - last_heartbeat >= 30:
                print(f'[{cfg["action_target_name"]}] epoch{epoch}/49 step{step}/{len(loader)} loss={loss:.6f}', flush=True)
                last_heartbeat = time.monotonic()
        if not losses:
            raise ValueError('Empty training loader')
        save_checkpoint(path, model, optimizer, epoch, cfg, plan, context)
        print(f'[{cfg["action_target_name"]}] epoch{epoch}/49 mean_loss={np.mean(losses):.6f} '
              f'seconds={time.monotonic()-began:.1f}', flush=True)


def train_cell(plan, cell, resume=False, lease_check=None):
    import numpy as np
    import torch
    import train_backdoor as native
    cfg = cell['cfg']
    root, folder = Path(plan['outdir']), Path(plan['outdir']) / cell['action']
    folder.mkdir(exist_ok=True)
    path = folder / 'checkpoint.pt'
    if path.exists() and not resume:
        raise ValueError('Checkpoint already exists; use explicit Resume, not Launch')
    context = execution()
    if context != read(root / 'prepared.json')['execution']:
        raise ValueError('Execution environment changed after Prepare')
    seed_all()
    base, ids = dataset(cfg, 'training')
    data = action_data(base, native.build_trigger(cfg), cfg)
    receipt = read(root / 'geometry.json')['records'][cell['action']]
    if (digest(ids) != receipt['train_pair_ids_sha256']
            or data.manifest()['poison_plan_sha256'] != receipt['poison_plan_sha256']):
        raise ValueError('Training identities/poison plan changed')
    write(folder / 'config.resolved.json', cfg)
    write(folder / 'poison_manifest.json', data.manifest())
    loader = native._training_loader(data, cfg)
    model = native.build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                               dataset='mmfi', pretrained=False).to('cuda:0')
    _, optimizer = native._build_optimizer(model, cfg, 'mmfi')
    start = 0
    if path.exists():
        record = checkpoint(path, cfg, plan)
        model.load_state_dict(record['model'], strict=True)
        optimizer.load_state_dict(record['optimizer'])
        native._restore_rng_state(record['rng'])
        start = record['epoch'] + 1
        del record
    train_epochs(model, optimizer, loader, cfg, path, plan, context,
                 device='cuda:0', start=start, lease_check=lease_check)
    # Closing the iterator before evaluation releases this run's own workers.
    del loader, data, base, optimizer
    return model


def metrics_from_arrays(gt, targets, predictions, joints):
    import numpy as np
    from eval.metrics import pa_mpjpe
    gt, targets, predictions = [np.array(value, dtype=np.float64, order='C', copy=True)
                               for value in (gt, targets, predictions)]
    if (targets.shape != predictions.shape or targets.shape != (6, len(gt), 17, 3)
            or any(not np.isfinite(value).all() for value in (gt, targets, predictions))):
        raise ValueError('Invalid full pose arrays')
    np.testing.assert_allclose(targets[0], gt, atol=0., rtol=0.)
    clean = predictions[0]
    scale = np.linalg.norm(gt[:, 5] - gt[:, 12], axis=-1)
    if np.any(scale <= 0):
        raise ValueError('Degenerate clean-GT PCK reference')
    others = [j for j in range(17) if j not in joints]
    rows = []
    for k, dose in enumerate(GRID):
        prediction, target = predictions[k], targets[k]
        error = np.linalg.norm(prediction[:, joints]-target[:, joints], axis=-1)
        no_trigger = np.linalg.norm(clean[:, joints]-target[:, joints], axis=-1)
        requested = target[:, joints]-gt[:, joints]
        induced = prediction[:, joints]-clean[:, joints]
        rows.append(dict(dose=dose, t_mpjpe_mm=float(error.mean()*1000),
            target_pck_pct=float((error <= .2*scale[:, None]).mean()*100),
            no_trigger_t_mpjpe_mm=float(no_trigger.mean()*1000),
            no_trigger_target_pck_pct=float((no_trigger <= .2*scale[:, None]).mean()*100),
            requested_mm=float(np.linalg.norm(requested, axis=-1).mean()*1000),
            predicted_mm=float(np.linalg.norm(induced, axis=-1).mean()*1000),
            dce_mm=float(np.linalg.norm(induced-requested, axis=-1).mean()*1000),
            triggered_gt_target_mm=float(np.linalg.norm(prediction[:, joints]-gt[:, joints], axis=-1).mean()*1000),
            nontarget_drift_mm=float(np.linalg.norm(prediction[:, others]-clean[:, others], axis=-1).mean()*1000)))
    return dict(clean_mpjpe_mm=float(np.linalg.norm(clean-gt, axis=-1).mean()*1000),
        clean_pampjpe_mm=float(pa_mpjpe(clean.copy(), gt.copy()).mean()*1000),
        clean_target_mpjpe_mm=float(np.linalg.norm(clean[:, joints]-gt[:, joints], axis=-1).mean()*1000),
        mean_positive_dce_mm=float(np.mean([row['dce_mm'] for row in rows[1:]])),
        doses=rows)


def evaluate_cell(plan, cell, model=None):
    import gc
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_backdoor as native
    from attack.poison import collate
    folder = Path(plan['outdir']) / cell['action']
    if (folder / 'complete.json').exists():
        return verify_complete(plan, cell)
    cfg = cell['cfg']
    context = execution()
    if context != read(Path(plan['outdir']) / 'prepared.json')['execution']:
        raise ValueError('Execution changed before evaluation')
    record = checkpoint(folder / 'checkpoint.pt', cfg, plan)
    if record['epoch'] != 49:
        raise ValueError('Evaluation requires final49, not an intermediate or best-TEST checkpoint')
    # Always reconstruct final49: never evaluate a discarded smoke or mutable
    # in-memory set of weights as if it were the bound disk checkpoint.
    del model
    model = native.build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                               dataset='mmfi', pretrained=False).to('cuda:0')
    model.load_state_dict(record['model'], strict=True)
    del record
    base, ids = dataset(cfg, 'test')
    input_record = read(Path(plan['outdir']) / 'inputs.json')
    if digest(ids) != input_record['splits']['eval']['pair_ids_sha256']:
        raise ValueError('TEST identity order changed')
    trigger = native.build_trigger(cfg)
    predictions, labels = [], []
    gt = None
    for dose in GRID:
        data = action_data(base, trigger, cfg, mode='trigger@dose', fixed_dose=dose)
        loader = DataLoader(data, batch_size=32, shuffle=False, collate_fn=collate,
                            num_workers=0, pin_memory=True)
        predicted, truth, target, _ = native._predict(model, loader, 'cuda:0')
        if gt is None:
            gt = truth.copy()
        else:
            np.testing.assert_array_equal(gt, truth)
        predictions.append(predicted)
        labels.append(target)
        print(f'[eval] {cell["action"]} d={dose:g}: {len(truth)} TEST frames', flush=True)
    predictions, targets = np.stack(predictions), np.stack(labels)
    metrics = metrics_from_arrays(gt, targets, predictions, cfg['target_joints'])
    artifact = folder / 'poses_full_test.npz'
    # Failed evaluations preserve orphan outputs. Explicit operator inspection
    # is required rather than overwriting/guessing which artifact is correct.
    write_npz(artifact, gt=gt, target=targets, prediction=predictions,
        doses=np.array(GRID), target_joints=np.array(cfg['target_joints']),
        pair_ids=np.array([json.dumps(value) for value in ids]),
        fixed_test_indices=np.array(FIXED_TEST),
        plan_sha256=np.array(plan['plan_sha256']))
    with np.load(artifact, allow_pickle=False) as saved:
        replay = metrics_from_arrays(saved['gt'], saved['target'], saved['prediction'], cfg['target_joints'])
    if replay != metrics:
        raise ValueError('Saved full pose arrays do not exactly reproduce reported metrics')
    report = dict(profile=PROFILE, action=cell['action'], plan_sha256=plan['plan_sha256'],
        payload=cfg['action_payload'], metrics=metrics, final_epoch=49,
        clean_trained_trigger_control='NOT_SUPPLIED',
        artifacts={'poses_full_test.npz': sha(artifact), 'checkpoint.pt': sha(folder / 'checkpoint.pt')})
    verify_prepared(plan, full_hash=True)
    write(folder / 'results.json', report)
    write(folder / 'complete.json', dict(plan_sha256=plan['plan_sha256'], results_sha256=sha(folder / 'results.json')))
    del model, predictions, targets, base, loader, data
    gc.collect()
    torch.cuda.empty_cache()
    return report


def verify_complete(plan, cell):
    folder = Path(plan['outdir']) / cell['action']
    marker = read(folder / 'complete.json')
    if marker != dict(plan_sha256=plan['plan_sha256'], results_sha256=sha(folder / 'results.json')):
        raise ValueError('Completion marker changed')
    report = read(folder / 'results.json')
    if report['plan_sha256'] != plan['plan_sha256'] or report['payload'] != cell['cfg']['action_payload']:
        raise ValueError('Foreign evaluation payload')
    for name, expected in report['artifacts'].items():
        if sha(folder / name) != expected:
            raise ValueError('Completed artifact changed: ' + name)
    return report


def export(plan):
    rows = []
    for cell in plan['cells']:
        if not (Path(plan['outdir']) / cell['action'] / 'complete.json').is_file():
            raise ValueError('Both actions must finish; no partial table relabeled complete')
        result = verify_complete(plan, cell)
        m = result['metrics']
        rows.append(dict(action=cell['action'], target_joints=json.dumps(cell['cfg']['target_joints']),
            t_mpjpe_mm=m['doses'][-1]['t_mpjpe_mm'], target_pck_pct=m['doses'][-1]['target_pck_pct'],
            mean_positive_dce_mm=m['mean_positive_dce_mm'], clean_mpjpe_mm=m['clean_mpjpe_mm'],
            clean_pampjpe_mm=m['clean_pampjpe_mm']))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    destination = Path(plan['outdir']) / 'action_targets_results.csv'
    data = buffer.getvalue().encode('utf-8')
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError('Existing summary differs; not overwritten')
    if not destination.exists():
        with destination.open('xb') as stream:
            stream.write(data)
    print(buffer.getvalue(), flush=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-home', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--mode', required=True, choices=['check-only', 'geometry', 'prepare',
                   'launch', 'resume', 'evaluate', 'status', 'export'])
    p.add_argument('--cross-action')
    p.add_argument('--nod-action')
    p.add_argument('--approve-action-labels', action='store_true')
    p.add_argument('--approve-geometry', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    verify_bundle()
    if args.mode == 'check-only':
        return check_paths(args)
    if args.mode == 'status':
        root = Path(args.outdir)
        for action in ('cross', 'nod'):
            folder = root / action
            print(f'{action}: checkpoint={(folder/"checkpoint.pt").is_file()}, '
                  f'complete={(folder/"complete.json").is_file()} (file presence only)')
        return
    # Native imports happen only after explicit operator selection. Same
    # process-local CPU threading as the previous4090 launchers, no GPU use yet.
    os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', PYTHONUTF8='1')
    from run_mmfi_tables import _cell_lock
    if args.mode == 'geometry':
        build_plan(args)  # Refuse unrelated output before a lock can create it.
        with _cell_lock(args.outdir):
            return geometry(args)
    plan = load_existing_plan(args)
    with _cell_lock(args.outdir):
        if args.mode == 'prepare':
            # A cross-output lock stops two copies of this NEW launcher; the
            # existing launchers' own guards remain entirely unchanged.
            with _cell_lock(Path(args.data_home) / 'runs/.mmfi_action_targets_gpu0'):
                return prepare(args, plan)
        verify_prepared(plan)
        if args.mode == 'export':
            return export(plan)
        from run_mmfi_p3_s3_three_attacks_4090 import require_idle
        with _cell_lock(Path(args.data_home) / 'runs/.mmfi_action_targets_gpu0'):
            require_idle()
            context = execution()
            if context != read(Path(args.outdir) / 'prepared.json')['execution']:
                raise ValueError('CUDA execution differs from Prepare')
            if shutil.disk_usage(args.outdir).free < 2 * 2**30:
                raise ValueError('Need at least2GiB free output space; nothing deleted')
            logs = Path(args.outdir) / 'logs'
            logs.mkdir(exist_ok=True)
            with (logs / f'{args.mode}.{time.time_ns()}.log').open('x', encoding='utf-8') as stream:
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = Tee(old_out, stream), Tee(old_err, stream)
                try:
                    for cell in plan['cells']:
                        if (Path(args.outdir) / cell['action'] / 'complete.json').is_file():
                            verify_complete(plan, cell)
                            continue
                        if args.mode != 'evaluate':
                            model = train_cell(plan, cell, resume=args.mode == 'resume')
                            del model
                        evaluate_cell(plan, cell)
                    export(plan)
                finally:
                    sys.stdout, sys.stderr = old_out, old_err


if __name__ == '__main__':
    main()
