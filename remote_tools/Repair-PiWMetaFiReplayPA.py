"""Canonicalize ONLY the confirmed one-ULP Smooth-LF PA JSON replay mismatch.

Default is read-only. --apply preserves the exact original evaluation JSON
and a hashed proposal before atomically updating that ONE derived scalar from
unchanged saved predictions. Never trains, infers, edits source/checkpoints,
weakens the original comparator or changes any other metric. Final completion
still requires the unchanged launcher --mode evaluate and all its guards.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
PROFILE = 'piw_metafi_two_extra_s42_4090_serial_v1'
CELL = Path('attacks/metafiplusplus/seed42')
OLD = 81.46046470956578
NEW = 81.4604647095658
TAG = 'pa_replay_roundoff_20261108_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe_file(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if (path.is_symlink() or not path.resolve().is_relative_to(root)
            or not path.is_file()):
        raise ValueError('Missing/linked/unsafe input: ' + str(path))
    return path


def checked_plan(root):
    path = safe_file(root, 'plan.json')
    plan = read(path)
    if plan.get('plan_sha256') != digest({k: v for k, v in plan.items() if k != 'plan_sha256'}):
        raise ValueError('Plan digest differs')
    cfg = plan['cfg']
    if (plan.get('profile') != PROFILE or cfg.get('methods') != ['tsba_adapted', 'smooth_lf_csi']
            or cfg.get('seed') != 42 or cfg.get('epochs') != 200 or cfg.get('final_epoch') != 199
            or cfg.get('eval_doses') != [0., .2, .4, .6, .8, 1.]):
        raise ValueError('Not the confirmed two-extra/final199 scope')
    snapshot = Path(root) / 'source_snapshot/ATKBackd'
    for name, expected in plan['sources'].items():
        if sha(safe_file(snapshot, name)) != expected:
            raise ValueError('Frozen scientific source changed: ' + name)
    return plan, snapshot


def validate_report(root, plan, method):
    folder = Path(root) / CELL / method
    report = read(safe_file(folder, 'evaluation.json'))
    own = safe_file(folder, 'checkpoint.pt')
    pred = safe_file(folder, 'predictions.npz')
    ref = plan['clean_reference']
    if (report.get('profile') != PROFILE or report.get('status') != 'COMPLETE'
            or report.get('method') != method or report.get('backbone') != 'metafiplusplus'
            or report.get('seed') != 42 or report.get('epoch') != 199
            or report.get('plan_sha256') != plan['plan_sha256']
            or report.get('poison') != plan['native_records'][method]
            or report.get('clean_reference') != ref
            or report.get('checkpoint_sha256') != sha(own)
            or report.get('predictions_sha256') != sha(pred)
            or report.get('clean_checkpoint_sha256') != ref['files']['checkpoint.pt']):
        raise ValueError('Evaluation/checkpoint/prediction binding differs: ' + method)
    return report


def replay(root, plan, method, measure, target_pose):
    import numpy as np
    folder = Path(root) / CELL / method
    truth_names = plan['inputs']['splits']['test_data']['ordered_names']
    with np.load(folder / 'predictions.npz', allow_pickle=False) as p:
        if (p['truth'].shape != (2586, 14, 3) or p['trigger_predictions'].shape != (2586, 6, 14, 3)
                or p['names'].tolist() != truth_names
                or not np.array_equal(p['doses'], [0., .2, .4, .6, .8, 1.])
                or not np.array_equal(p['targets'], np.stack([target_pose(p['truth'], d) for d in p['doses']], axis=1))
                or not np.array_equal(p['predictions'], p['trigger_predictions'][:, 0])
                or not np.array_equal(p['reference_predictions'], p['reference_trigger_predictions'][:, 0])):
            raise ValueError('Full TEST identity/target/dose schema differs: ' + method)
        clean_path = Path(plan['clean_reference']['run']) / CELL / 'clean/predictions.npz'
        with np.load(clean_path, allow_pickle=False) as clean:
            if clean['names'].tolist() != truth_names or not np.array_equal(p['truth'], clean['truth']):
                raise ValueError('External Clean TEST identities/truth differ')
        return measure(p['truth'], p['predictions'], p['trigger_predictions'],
                       p['reference_predictions'], p['reference_trigger_predictions'])


def corrected_report(report, actual):
    """No general tolerance: exact known Smooth-LF field/value pair only."""
    if digest(report['metrics']) == digest(actual):
        return copy.deepcopy(report), False
    if (report.get('method') != 'smooth_lf_csi'
            or report['metrics']['mainmetrics']['clean_pampjpe_mm'] != OLD
            or actual['mainmetrics']['clean_pampjpe_mm'] != NEW
            or NEW - OLD != math.ulp(OLD)):
        raise ValueError('Not the single confirmed PA one-ULP mismatch; no changes')
    metrics = copy.deepcopy(report['metrics'])
    metrics['mainmetrics']['clean_pampjpe_mm'] = actual['mainmetrics']['clean_pampjpe_mm']
    if digest(metrics) != digest(actual):
        raise ValueError('Additional metric/schema differences; no changes')
    result = copy.deepcopy(report)
    result['metrics'] = metrics
    return result, True


def immutable_bytes(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing backup/audit differs; preserve it: ' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def apply_correction(path, report, actual, protected, *, apply=False):
    """Backup + proposal BEFORE replace. Protected files are rehashed, not edited."""
    path = Path(path)
    before = path.read_bytes()
    if read(path) != report:
        raise ValueError('Evaluation changed during inspection')
    result, changed = corrected_report(report, actual)
    if not changed:
        print('ALREADY_EXACT_REPLAY; no file changed', flush=True)
        return False
    print(f'CONFIRMED_PA_ONLY: {OLD!r} -> {NEW!r}; delta={NEW-OLD}; exactly1ULP', flush=True)
    if not apply:
        print('READ_ONLY: pass --apply only to preserve original JSON and canonicalize this scalar.', flush=True)
        return False
    if any(sha(p) != expected for p, expected in protected.items()):
        raise ValueError('Protected input changed; no correction')
    after = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    backup = path.with_name('evaluation.before_' + TAG + '.json')
    proposal = path.with_name(TAG + '.proposal.json')
    audit = dict(profile=TAG, method='smooth_lf_csi', scope='derived JSON scalar only; no source/NPZ/checkpoint edits',
        old_value=OLD, new_value=NEW, delta_mm=NEW-OLD, ulps=1,
        original_sha256=hashlib.sha256(before).hexdigest(), replacement_sha256=hashlib.sha256(after).hexdigest(),
        plan_sha256=report['plan_sha256'], protected={str(p): value for p, value in protected.items()},
        repair_script_sha256=sha(__file__))
    immutable_bytes(backup, before)
    immutable_bytes(proposal, encoded(audit) + b'\n')
    temporary = path.with_name(path.name + '.' + TAG + '.saving')
    immutable_bytes(temporary, after)
    if path.read_bytes() != before or any(sha(p) != expected for p, expected in protected.items()):
        raise ValueError('Inputs changed after backup; replacement refused')
    os.replace(temporary, path)  # exact named derived JSON; original bytes recoverable above
    if path.read_bytes() != after or any(sha(p) != expected for p, expected in protected.items()):
        raise ValueError('Post-correction verification failed; retain backup/proposal for inspection')
    immutable_bytes(path.with_name(TAG + '.applied.json'), encoded(audit) + b'\n')
    print('PA_JSON_CANONICALIZED; original backup=' + str(backup), flush=True)
    print('PATCHED_ONLY: run the unchanged launcher -Mode Evaluate, NOT Launch.', flush=True)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args(argv)
    root = args.run.resolve()
    controller = safe_file(args.checkout, 'remote_tools/piw_metafi_remaining_3090/controller.py')
    spec = importlib.util.spec_from_file_location('_unchanged_replay_lock', controller)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    if not root.is_dir() or root.name != PROFILE:
        raise ValueError('Use the existing dedicated two-extra run; nothing is created')
    # Same real OS output lock as the original controller. No lock deletion,
    # process termination, gate alteration or concurrent writer override.
    with base.lock(root):
        plan, snapshot = checked_plan(root)
        gate_path = safe_file(root, 'prelaunch_gate.json')
        gate = read(gate_path)
        if gate.get('plan_sha256') != plan['plan_sha256']:
            raise ValueError('Prelaunch gate/plan mismatch')
        for key, name in {'runner': 'piw_metafi_two_extra_4090.py',
                'helper': 'Start-PiWMetaFiTwoExtra4090.ps1', 'orchestration': 'piw_metafi_four_attacks_4090.py',
                'a1_sidecar': 'piw_metafi_a1_4090.py'}.items():
            if sha(safe_file(args.checkout / 'remote_tools', name)) != gate['code'][key]:
                raise ValueError('Original orchestration changed: ' + name)
        protected = {safe_file(root, 'plan.json'): sha(root / 'plan.json'), gate_path: sha(gate_path)}
        for name, expected in plan['sources'].items():
            protected[safe_file(snapshot, name)] = expected
        for name in ('piw_metafi_two_extra_4090.py', 'Start-PiWMetaFiTwoExtra4090.ps1',
                     'piw_metafi_four_attacks_4090.py', 'piw_metafi_a1_4090.py'):
            path = safe_file(args.checkout / 'remote_tools', name)
            protected[path] = sha(path)
        protected[controller] = sha(controller)
        ref = plan['clean_reference']
        clean_root = Path(ref['run'])
        for name, expected in ref['files'].items():
            relative = CELL / 'clean' / name if name in ('checkpoint.pt', 'predictions.npz', 'evaluation.json') else Path(name)
            path = safe_file(clean_root, relative)
            if sha(path) != expected:
                raise ValueError('Clean reference changed: ' + name)
            protected[path] = expected
        sys.path.insert(0, str(snapshot))
        from piw_v2_adapter import measure_predictions, target_pose
        reports, metrics = {}, {}
        for method in ('tsba_adapted', 'smooth_lf_csi'):
            reports[method] = validate_report(root, plan, method)
            for name in ('checkpoint.pt', 'predictions.npz'):
                path = safe_file(root, CELL / method / name); protected[path] = sha(path)
            metrics[method] = replay(root, plan, method, measure_predictions, target_pose)
            # Separate fresh NPZ loads must reproduce bit-for-bit before any write.
            if digest(metrics[method]) != digest(replay(root, plan, method, measure_predictions, target_pose)):
                raise ValueError('CPU replay is unstable; no correction')
            corrected_report(reports[method], metrics[method])
        tsba_path = safe_file(root, CELL / 'tsba_adapted/evaluation.json')
        protected[tsba_path] = sha(tsba_path)
        path = safe_file(root, CELL / 'smooth_lf_csi/evaluation.json')
        changed = apply_correction(path, reports['smooth_lf_csi'], metrics['smooth_lf_csi'], protected, apply=args.apply)
        if changed and digest(read(path)['metrics']) != digest(replay(root, plan, 'smooth_lf_csi', measure_predictions, target_pose)):
            raise ValueError('Final strict replay failed; preserve backup and inspect')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print('REPAIR_REFUSED: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
