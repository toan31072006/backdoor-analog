"""New A1 forward Bend: one scratch HPE-Li per dataset, NO Clean retraining.

Explicit Geometry -> visual approval -> Prepare/B32 smoke -> Launch/Resume.
The helper queues MM-Fi then PiW in ONE additional slot alongside a named peer.
No module-global patching, old target relabeling, process kill or auto retry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from importlib.metadata import version

from run_mmfi_action_targets import (HERE, GRID, Tee, digest, read, sha,
    source_hashes, write, write_npz, audit_action_geometry, finish_geometry_audit)

ACTION = 'bend_forward'
PLAN_NAME = 'bend_forward.plan.json'


def module_for(dataset):
    if dataset == 'mmfi':
        import run_mmfi_action_targets as runner
    elif dataset == 'piw':
        import run_piw_action_targets as runner
    else:
        raise ValueError('Dataset must be mmfi or piw')
    return runner


def check_paths(args):
    home = Path(args.data_home)
    paths = [home / 'actions/data_bend.npy']
    if args.dataset == 'mmfi':
        paths.append(home / 'datasets/Compress')
    else:
        root = Path(args.dataset_root or home / 'datasets/Person-in-WiFi-3D_processed_v1')
        paths += [root / f'{s}/{s}_list.txt' for s in ('train_data', 'test_data')]
    for path in paths:
        print(f'{"OK" if path.exists() else "MISSING"}: {path}', flush=True)
    if any(not p.exists() for p in paths):
        raise FileNotFoundError('Required existing inputs missing; no download/fallback')
    print('CHECK_ONLY: no training, CUDA, data arrays or output writes.', flush=True)


def build_plan(args):
    import run_piw_hpeli_clean as piw_base
    from attack.bend_forward import specification
    from run_mmfi_a1_motion_priors_4090 import action_record, trigger_record
    home, root = Path(args.data_home).resolve(), Path(args.outdir).resolve()
    root_match = re.fullmatch(rf'{args.dataset}_bend_forward_s42_v([0-9]+)', root.name)
    if (not root_match or int(root_match.group(1)) < 3
            or root.parent.name.lower() != 'runs' or root.is_relative_to(HERE)
            or home.is_relative_to(root) or root.is_relative_to(home / 'datasets')
            or (root.exists() and not (root / PLAN_NAME).exists()
                and any(p.name != 'run.lock' for p in root.iterdir()))):
        raise ValueError('Use NEW runs/<mmfi|piw>_bend_forward_s42_v3 (or higher); preserve v1/v2/source/data')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''):
        raise ValueError('Physical cuda:0 only, no remapping')
    prior = home / 'actions/data_bend.npy'
    spec = specification(args.dataset)
    runner = module_for(args.dataset)
    profile = f'{args.dataset}_hpeli_a1_bend_forward_s42_v3'
    if args.dataset == 'mmfi':
        from run_a1_three_tables import _configuration
        from train_backdoor import _resolve_training_config, _validate_training_contract
        cfg = _configuration(home, 'cuda:0', 4, 42, 'a1')
        cfg.update(confirmation_profile=profile, action_target_name=ACTION,
            action_payload=spec, action_npy=str(prior), pivot=0,
            target_joints=spec['target_joints'], payload_axis=[0., 0., 1.],
            theta_max_deg=40., ckpt_every=1)
        cfg = _resolve_training_config(cfg)
        _validate_training_contract(cfg)
        cell = dict(action=ACTION, cfg=cfg, trigger_arrays=trigger_record(cfg))
    else:
        cfg = piw_base.configuration(home, root, 'cuda:0', args.dataset_root)
        cfg.update(profile=profile, action=ACTION, payload=spec,
            action_npy=str(prior), drop_last=True, rho=.1, attack_eps=.185,
            training_doses=[.2, 1.], eval_doses=GRID, target_pck_reference=[6, 4],
            target_pck_threshold=.2, initialization='independent scratch; no Clean weight import')
        cell = dict(action=ACTION, cfg=cfg, trigger_record=runner.trigger_record(prior, payload=spec))
    cell['action_record'] = action_record(prior, 'bend')
    runtime = piw_base.runtime_versions()
    runtime['packages']['matplotlib'] = version('matplotlib')
    from bend_gpu_guard import policy
    value = dict(schema=1, profile=profile, dataset=args.dataset, data_home=str(home),
        outdir=str(root), cells=[cell], sources=source_hashes(), runtime=runtime,
        contract=dict(status='EXPLORATORY_NEW_TARGET_NOT_COMPLETE_T7', seed=42,
            counts=runner.COUNTS, doses=GRID, rho=.1, eps=.185,
            training_doses='fixed paired U(.2,1); native A1 selection/operator unchanged',
            payload=spec, signed_angle_deg=spec.get('rotation_sign', 1.) * spec['theta_max_deg'],
            correction='WDDM guard only; exact approved MM-Fi v1/PiW v2 payloads unchanged; fresh v3 plans',
            gpu_policy=policy(),
            physical_batch=32, epochs=50 if args.dataset == 'mmfi' else 200,
            final_epoch=49 if args.dataset == 'mmfi' else 199,
            recipe='SGD .001/momentum.9/wd0/workers4' if args.dataset == 'mmfi'
                else 'AdamW .001/wd.01/betas(.9,.999)/eps1e-8/workers0/drop_lastTrue',
            ordinary_all_joint_MPJPE=True, pretrained=False, amp=False,
            jobs=1, queue_order='MM-Fi then PiW; one extra worker alongside explicit peer',
            clean_training=False, clean_weight_import=False,
            clean_control='NOT_SUPPLIED; reuse only after separate provenance audit, never auto-train',
            fixed_test_indices=runner.FIXED_TEST,
            limitations=['Fixed dataset forward convention must be inspected on TRAIN before approval.',
                'Signed 40d-degree incremental rigid trunk rotation, not conversion of every pose to an action class.',
                'Bone preservation does not prove anatomy, foot contact, balance or action recognition.',
                'No new baselines, Clean or clean-model+trigger control; not full T7.',
                'One seed; no uncertainty/significance or physical RF feasibility claim.']))
    value['plan_sha256'] = digest(value)
    return value


def load_plan(args):
    plan = read(Path(args.outdir) / PLAN_NAME)
    if plan != build_plan(args):
        raise ValueError('Source/runtime/input/config changed; use frozen checkout, never edit an old plan')
    return plan


def preview_figure(path, truth, targets, spec):
    """Display-only SIDE projection in the SAME clean body frame at every dose."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from attack.bend_forward import display_basis
    from attack.payload import MMFI_EDGES
    from attack.piw_action_targets import EDGES
    if path.exists() or path.with_suffix('.pdf').exists():
        raise FileExistsError('Preserve existing geometry figure')
    edges = MMFI_EDGES if spec['dataset'] == 'mmfi' else EDGES
    projected = []
    for original, generated in zip(truth, targets.transpose(1, 0, 2, 3)):
        origin, basis = display_basis(original, spec)
        projected.append(((original-origin) @ basis, (generated-origin) @ basis))
    values = np.concatenate([p.reshape(-1, 2) for pair in projected for p in pair])
    low, high = values.min(0), values.max(0)
    margin = max(.1, .08*float(np.ptp(values, axis=0).max()))
    fig, axs = plt.subplots(3, 3, figsize=(9, 9), squeeze=False)
    angle = spec.get('rotation_sign', 1.) * spec['theta_max_deg']
    fig.suptitle(f'{spec["version"]} / signed angle {angle:+g}d deg / TRAIN TARGET GEOMETRY ONLY\n'
        'Grey = true pose; orange = target. Right = declared forward; no model prediction.', fontsize=10)
    for row, (original, generated) in enumerate(projected):
        for col, dose in enumerate((0., .5, 1.)):
            ax = axs[row, col]
            for value, color, style in ((original, '#999999', '-'), (generated[col], '#d9720a', '--')):
                for a, b in edges:
                    ax.plot(*value[[a, b]].T, color=color, ls=style, lw=1.3)
                ax.scatter(*value.T, color=color, s=12)
            ax.scatter(*generated[col, spec['pivots']].T, color='#c62728', s=28, zorder=5)
            ax.set(xlim=(low[0]-margin, high[0]+margin), ylim=(low[1]-margin, high[1]+margin),
                xlabel='declared forward (m)', ylabel='clean body up (m)',
                title=f'fixed TRAIN example {row+1}; d={dose:g}')
            ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.15)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=160); fig.savefig(path.with_suffix('.pdf')); plt.close(fig)


def geometry(args):
    import numpy as np
    from attack.bend_forward import target_pose
    from attack.payload import MMFI_EDGES
    from attack.piw_action_targets import EDGES
    from piw_action_target_data import ActionTargetPairs
    plan = build_plan(args)
    root, runner, cell = Path(plan['outdir']), module_for(args.dataset), plan['cells'][0]
    if (root / 'geometry.json').exists():
        raise ValueError('Already inspected; no overwrite. Inspect PNG then Prepare.')
    write(root / PLAN_NAME, plan)
    cfg = cell['cfg']
    base, ids = runner.dataset(cfg, 'training')
    if args.dataset == 'mmfi':
        data = runner.action_data(base, None, cfg, cache_targets=False)
        selected = sorted(data.poison_idx)
        spec, edges = cfg['action_payload'], MMFI_EDGES
    else:
        data = ActionTargetPairs(base, None, ACTION, payload=cfg['payload'], target_fn=target_pose)
        selected = sorted(data.selected)
        spec, edges = cfg['payload'], EDGES
    chosen = [selected[0], selected[len(selected)//2], selected[-1]]
    report, failures, pose_files = audit_action_geometry(selected, data.clean_pose, ids,
        spec, edges, target_pose, None, ACTION,
        pose_hash=lambda i: sha(base.items[i]['kpt']), progress_every=500)
    entry = dict(report, selected_train_indices=chosen, pose_file_sha256=pose_files)
    if args.dataset == 'mmfi':
        entry.update(train_pair_ids_sha256=digest(ids), n_poison=data.n_poison,
            poison_plan_sha256=data.manifest()['poison_plan_sha256'])
    else:
        entry.update(train_ids_sha256=digest(ids), poison=data.manifest())
    records = {ACTION: entry}
    audit_sha = finish_geometry_audit(root, plan, records, failures)
    truth = np.stack([data.clean_pose(i)[0] for i in chosen])
    targets = np.stack([target_pose(truth, d, spec) for d in (0., .5, 1.)])
    artifact, picture = root / 'geometry_bend_forward.npz', root / 'geometry_bend_forward.png'
    write_npz(artifact, gt=truth, targets=targets, doses=np.array([0., .5, 1.]),
        indices=np.array(chosen), pair_ids=np.array([json.dumps(ids[i]) for i in chosen]))
    preview_figure(picture, truth, targets, spec)
    files = {p.name: sha(p) for p in (artifact, picture, picture.with_suffix('.pdf'))}
    write(root / 'geometry.json', dict(plan_sha256=plan['plan_sha256'], records=records,
        files=files, audit_file_sha256=audit_sha, source='ALL_POISON_TRAIN_ONLY_NO_MODEL'))
    print(f'GEOMETRY_ONLY: inspect {picture}; explicitly approve geometry. No training.', flush=True)


def process_snapshot():
    """Read-only Windows process evidence; never kill/suspend or interpolate input."""
    script = ('Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,'
              'CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
        check=True, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
    value = json.loads(result.stdout)
    return value if isinstance(value, list) else [value]


def gpu_guard(peer_pid=0):
    """Allow only this worker and ONE explicitly named existing transfer peer.

    Known desktop apps need WDDM + audited name/path + G/C+G XML type. Unknown
    compute still fails closed. This is NOT exclusive GPU use/VRAM assurance.
    """
    if peer_pid < 0 or peer_pid == os.getpid():
        raise ValueError('Invalid peer PID')
    from bend_gpu_guard import desktop_exemption, parse_gpu_xml, policy
    result = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=30)
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not re.fullmatch(r'[0-9]+', value) or int(value) <= 0 for value in values):
        raise ValueError('Cannot enumerate GPU0 compute PIDs; do not bypass coexistence guard')
    xml = subprocess.run(['nvidia-smi', '--id=0', '-q', '-x'],
        check=True, capture_output=True, text=True, timeout=30)
    driver, gpu_records = parse_gpu_xml(xml.stdout)
    # Take current OS identities AFTER GPU queries; disappearing PIDs fail closed.
    rows = process_snapshot()
    by_pid = {int(p['ProcessId']): p for p in rows}
    # Windows has a real System Idle Process PID0: --peer-pid 0 means NO peer.
    peer = by_pid.get(peer_pid) if peer_pid else None
    allowed = {os.getpid()}
    if peer:
        command = peer.get('CommandLine') or ''
        if not re.search(r'run_(?:a1_wiflow|piw_backtime_three_backbones|backtime_transfer[^\s]*)\.py', command, re.I):
            raise ValueError('Peer is not a recognized WiFlow/BackTime transfer runner; inspect its command first')
        allowed.add(peer_pid)
    for _ in range(len(rows)):
        expanded = {p for p, row in by_pid.items() if int(row['ParentProcessId']) in allowed}
        if expanded <= allowed:
            break
        allowed |= expanded
    gpu_pids = set(map(int, values)) | set(gpu_records)
    exemptions = []
    for pid in sorted(gpu_pids - allowed):
        evidence = desktop_exemption(pid, by_pid.get(pid), gpu_records.get(pid), driver)
        if evidence is not None:
            exemptions.append(evidence)
    foreign = gpu_pids - allowed - {entry['pid'] for entry in exemptions}
    if foreign:
        raise ValueError(f'Unexpected GPU0 processes {sorted(foreign)}; no jobs stopped or modified')
    return dict(peer=peer, gpu_pids=sorted(gpu_pids), allowed_pids=sorted(allowed),
        driver_model=driver, desktop_exemptions=exemptions, policy=policy()['version'])


def verify_train_pose_hashes(plan):
    runner = module_for(plan['dataset'])
    cell = plan['cells'][0]
    base, _ = runner.dataset(cell['cfg'], 'training')
    record = read(Path(plan['outdir']) / 'geometry.json')['records'][ACTION]
    for index, expected in record['pose_file_sha256'].items():
        if sha(base.items[int(index)]['kpt']) != expected:
            raise ValueError('TRAIN pose changed since geometry approval')


def run_stage(args):
    runner = module_for(args.dataset)
    if args.mode == 'geometry':
        return geometry(args)
    plan = load_plan(args)
    root, cell = Path(plan['outdir']), plan['cells'][0]
    if args.mode == 'export':
        runner.verify_prepared(plan)
        return runner.export(plan)
    if shutil.disk_usage(root).free < 2 * 2**30:
        raise ValueError('Need at least 2GiB free on output volume; no cleanup/fallback')
    path = root / 'coexistence.json'
    coexistence = read(path) if path.exists() else None
    if coexistence is not None and (coexistence['plan_sha256'] != plan['plan_sha256']
            or coexistence['peer_pid'] != args.peer_pid):
        raise ValueError('Use the same approved peer PID, including after it exits')
    def guard():
        current = gpu_guard(args.peer_pid)
        if coexistence is not None and current['peer'] is not None:
            if current['peer'] != coexistence['evidence']['peer']:
                raise ValueError('Peer PID identity changed; no automatic substitution')
        return current
    guard()
    verify_train_pose_hashes(plan)
    if args.mode == 'prepare':
        runner.prepare(args, plan, idle_check=guard)
        write(path, dict(plan_sha256=plan['plan_sha256'], peer_pid=args.peer_pid,
            evidence=guard(), policy='one new Bend worker; never stop/retry peers'))
        return
    receipt = runner.verify_prepared(plan, full_hash=True)
    if coexistence is None:
        raise ValueError('Missing coexistence Prepare receipt')
    if runner.execution() != receipt['execution']:
        raise ValueError('CUDA backend changed since Prepare')
    if (root / ACTION / 'complete.json').exists():
        runner.verify_complete(plan, cell)
    else:
        if args.mode != 'evaluate':
            model = runner.train_cell(plan, cell, resume=args.mode == 'resume', lease_check=guard)
            del model
        runner.evaluate_cell(plan, cell)  # TEST only after this dataset's fixed final
    runner.export(plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['mmfi', 'piw'])
    parser.add_argument('--data-home', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--dataset-root')
    parser.add_argument('--peer-pid', type=int, default=0)
    parser.add_argument('--approve-geometry', action='store_true')
    parser.add_argument('--mode', required=True, choices=['check-only', 'guard-check', 'geometry', 'prepare',
        'launch', 'resume', 'evaluate', 'export', 'status'])
    args = parser.parse_args(argv)
    if args.mode == 'check-only':
        return check_paths(args)
    if args.mode == 'guard-check':
        print('GPU_GUARD_PASS '+json.dumps(gpu_guard(args.peer_pid)), flush=True)
        return  # read-only: no CUDA allocation, plan, locks, output/log writes
    if args.mode == 'status':
        root = Path(args.outdir) / ACTION
        print(f'{args.dataset}: checkpoint={(root/"checkpoint.pt").is_file()}, '
              f'complete={(root/"complete.json").is_file()} (presence only)')
        return
    plan = build_plan(args) if args.mode == 'geometry' else load_plan(args)
    from run_mmfi_tables import _cell_lock
    with _cell_lock(Path(args.outdir).resolve().parent / '.bend_forward_gpu0_slot'):
        with _cell_lock(args.outdir):
            if args.mode == 'geometry':
                write(Path(args.outdir) / PLAN_NAME, plan)
            logs = Path(args.outdir) / 'logs'; logs.mkdir(exist_ok=True)
            with (logs / f'{args.mode}.{time.time_ns()}.log').open('x', encoding='utf-8') as stream:
                previous = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = Tee(previous[0], stream), Tee(previous[1], stream)
                try:
                    return run_stage(args)
                finally:
                    sys.stdout, sys.stderr = previous


if __name__ == '__main__':
    main()
