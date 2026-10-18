"""Clean-bank partitioning and frozen-model evaluation for the isolated T5 study.

Nothing in this module writes to victim runs or tunes a defense on official TEST.
The caller audits the checkpoint/config before passing a model and its trigger.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

GRID = (0., .2, .4, .6, .8, 1.)
TARGET_JOINTS = (2, 3)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True,
        sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('seed must be a nonnegative integer')
    return int(value)


def _fraction(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must lie strictly between zero and one')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must lie strictly between zero and one') from exc
    if not math.isfinite(result) or not 0. < result < 1.:
        raise ValueError(f'{name} must lie strictly between zero and one')
    return result


def _identities(pair_ids):
    """Canonical IDs are [relative CSI path, sequence GT path, frame index]."""
    ids = []
    groups = {}
    if not isinstance(pair_ids, (list, tuple)) or not pair_ids:
        raise ValueError('pair_ids must be a nonempty ordered list')
    for index, item in enumerate(pair_ids):
        if (not isinstance(item, (list, tuple)) or len(item) != 3
                or not isinstance(item[0], str) or not item[0]
                or not isinstance(item[1], str) or not item[1]
                or isinstance(item[2], (bool, np.bool_))
                or not isinstance(item[2], (int, np.integer)) or item[2] < 0):
            raise ValueError('invalid canonical CSI/pose/frame identity')
        # Slash normalization avoids treating Windows/POSIX spelling as two groups.
        row = [item[0].replace('\\', '/'), item[1].replace('\\', '/'), int(item[2])]
        ids.append(row)
        groups.setdefault(row[1], []).append(index)
    if len({tuple(row) for row in ids}) != len(ids):
        raise ValueError('duplicate training frame identities')
    if len({(row[1], row[2]) for row in ids}) != len(ids):
        raise ValueError('duplicate sequence/frame identities')
    return ids, groups


def _balanced_sample(group_names, groups, count, rng):
    """Sequence-balanced quotas, then sample uniformly within each sequence."""
    quotas = {name: 0 for name in group_names}
    remaining = int(count)
    active = list(group_names)
    while remaining:
        active = [name for name in active if quotas[name] < len(groups[name])]
        if not active:
            raise ValueError('insufficient sequence capacity for clean bank')
        for name in rng.permutation(active).tolist():
            quotas[name] += 1
            remaining -= 1
            if not remaining:
                break
    chosen = []
    for name in group_names:
        if quotas[name]:
            chosen.extend(map(int, rng.choice(groups[name], size=quotas[name], replace=False)))
    return sorted(chosen)


def select_clean_bank(pair_ids, seed, bank_fraction=.05, selection_fraction=.2):
    """Select exactly floor(bank_fraction*N) trusted, clean TRAIN frames.

    The bank is a *total* budget: selection/calibration is carved from this bank,
    not added on top. All selected frames from a sequence belong to one side.
    Model scores, attack identities, labels and official TEST never enter selection.
    The caller must supply IDs from canonical ``training``, not MM-Fi's
    misleading ``validation`` alias (which is official TEST).
    """
    seed = _seed(seed)
    fraction = _fraction(bank_fraction, 'bank_fraction')
    select_fraction = _fraction(selection_fraction, 'selection_fraction')
    ids, groups = _identities(pair_ids)
    total = int(math.floor(len(ids) * fraction))
    n_selection = int(math.floor(total * select_fraction))
    n_fit = total - n_selection
    if min(n_fit, n_selection) < 1 or len(groups) < 2:
        raise ValueError('clean bank requires nonempty fit/selection and two sequences')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 518071]))
    names = rng.permutation(sorted(groups)).tolist()
    capacities = np.cumsum([len(groups[name]) for name in names])
    candidates = [cut for cut in range(1, len(names))
                  if capacities[cut - 1] >= n_selection
                  and len(ids) - capacities[cut - 1] >= n_fit]
    if not candidates:
        raise ValueError('cannot separate enough fit/selection frames by sequence')
    # Closest TRAIN frame-capacity fraction, with deterministic tie breaking.
    cut = min(candidates, key=lambda c: (abs(capacities[c - 1] / len(ids) - select_fraction), c))
    selection = _balanced_sample(names[:cut], groups, n_selection, rng)
    fit = _balanced_sample(names[cut:], groups, n_fit, rng)
    bank = sorted(fit + selection)
    fit_groups = sorted({ids[i][1] for i in fit})
    selection_groups = sorted({ids[i][1] for i in selection})
    if set(fit_groups).intersection(selection_groups):
        raise AssertionError('internal sequence partition overlap')
    hashes = {name + '_pair_ids_sha256': _digest([ids[i] for i in indices])
              for name, indices in (('bank', bank), ('fit', fit), ('selection', selection))}
    return dict(schema=1, seed=seed, bank_fraction=fraction,
        selection_fraction=select_fraction, bank_indices=bank,
        fit_indices=fit, selection_indices=selection,
        pair_ids_sha256=_digest(ids), **hashes,
        counts=dict(train=len(ids), bank=len(bank), fit=len(fit), selection=len(selection),
                    fit_sequences=len(fit_groups), selection_sequences=len(selection_groups)),
        sequence_groups=dict(fit=fit_groups, selection=selection_groups),
        allocation='exact total budget; sequence-disjoint selection; sequence-balanced fit/selection',
        selection_scope='clean official TRAIN identities supplied by audited caller; no TEST selection')


def _checked_indices(bank, name, size):
    indices = bank.get(name)
    if (not isinstance(indices, list) or not indices
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < size for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError(f'{name} must be sorted unique in-range nonempty indices')
    return indices


def clean_loaders(base_train, bank, cfg, seed, batch_size=32):
    """Return clean fit/selection loaders, avoiding Windows shared-memory queues."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from attack.poison import PoisonedDataset, collate
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    seed = _seed(seed)
    if bank.get('seed') != seed or bank.get('counts', {}).get('train') != len(base_train):
        raise ValueError('clean bank seed or TRAIN count differs')
    # Reconstruct selection from identities as an anti-tampering check. No CSI is read.
    from run_peak_confirmation import _pair_ids
    expected = select_clean_bank(_pair_ids(base_train), seed,
        bank_fraction=bank['bank_fraction'], selection_fraction=bank['selection_fraction'])
    if bank != expected:
        raise ValueError('clean bank metadata/identities changed')
    fit = _checked_indices(bank, 'fit_indices', len(base_train))
    selection = _checked_indices(bank, 'selection_indices', len(base_train))
    if set(fit).intersection(selection):
        raise ValueError('clean fit/selection overlap')
    ds = PoisonedDataset(base_train, None, mode='clean', pivot=cfg['pivot'],
        axis=cfg['payload_axis'], dataset='mmfi')
    generator = torch.Generator().manual_seed(seed + 518072)
    common = dict(batch_size=batch_size, collate_fn=collate,
                  num_workers=0, pin_memory=False, drop_last=False)
    return (DataLoader(Subset(ds, fit), shuffle=True, generator=generator, **common),
            DataLoader(Subset(ds, selection), shuffle=False, **common))


def _validate_pose(pose, count, name):
    value = np.asarray(pose)
    if (value.shape != (count, 17, 3) or not np.issubdtype(value.dtype, np.number)
            or np.issubdtype(value.dtype, np.complexfloating) or not np.isfinite(value).all()):
        raise ValueError(f'{name} must be finite real ({count},17,3) coordinates')
    return value.astype(np.float64, copy=False)


def evaluate_defense(model, data_test, trigger, cfg, device, *, expected_count=33264):
    """Evaluate a frozen repaired model on the complete, paired TEST grid.

    Main metrics are at d=1. DCE uses only the five positive doses in its
    summary and is retained as a diagnostic, not a defense-success score.
    Passing another explicit expected_count is for synthetic tests only.
    """
    from torch.utils.data import DataLoader
    from attack.poison import PoisonedDataset, collate
    from attack.payload import set_skeleton_config
    from eval import metrics as M
    from eval.target_pck import target_pck
    from train_backdoor import _predict
    count = len(data_test)
    if (isinstance(expected_count, bool) or not isinstance(expected_count, int)
            or expected_count < 1 or count != expected_count):
        raise ValueError('official TEST sample count differs; no partial metric accepted')
    if (cfg.get('pivot') != 1 or list(cfg.get('target_joints', [2, 3])) != [2, 3]
            or list(cfg.get('dose_grid', GRID)) != list(GRID)):
        raise ValueError('T5 requires frozen MM-Fi target joints and six-dose grid')
    if getattr(trigger, 'requires_deferred_injection', False):
        raise ValueError('T5 frozen baseline suite does not include learned trigger generators')
    set_skeleton_config('mmfi')

    def loader(mode, dose=None):
        ds = PoisonedDataset(data_test, trigger, mode=mode, pivot=1,
            axis=cfg['payload_axis'], dataset='mmfi', eps=cfg['eps'],
            theta_max_deg=cfg['theta_max_deg'], dose_mode=cfg['dose_mode'], fixed_dose=dose)
        return DataLoader(ds, batch_size=cfg['batch_size'], collate_fn=collate,
                          shuffle=False, num_workers=0, pin_memory=False)

    print('[T5 eval] clean full TEST pass', flush=True)
    clean, true, _, _ = _predict(model, loader('clean'), device)
    clean = _validate_pose(clean, count, 'clean prediction')
    true = _validate_pose(true, count, 'clean ground truth')
    clean_error = np.linalg.norm(clean - true, axis=-1) * 1000.
    rows = []
    dose_errors, gt_errors, dce_errors, no_trigger_errors = [], [], [], []
    endpoint = None
    for dose in GRID:
        print(f'[T5 eval] triggered full TEST pass d={dose:g}', flush=True)
        pred, true_again, target, returned_doses = _predict(model, loader('trigger@dose', dose), device)
        pred = _validate_pose(pred, count, 'triggered prediction')
        target = _validate_pose(target, count, 'attacker target')
        true_again = _validate_pose(true_again, count, 'triggered ground truth')
        if not np.array_equal(true, true_again):
            raise ValueError('clean/triggered frame order or ground truth changed')
        if returned_doses is None or np.asarray(returned_doses).shape != (count,) or not np.allclose(returned_doses, dose, rtol=0, atol=1.e-7):
            raise ValueError('returned evaluation doses differ from frozen dose')
        if dose == 0 and (not np.array_equal(target, true)
                          or not np.allclose(pred, clean, rtol=0, atol=1.e-6)):
            raise ValueError('dose-zero identity control failed')
        target_error = np.linalg.norm(pred[:, TARGET_JOINTS] - target[:, TARGET_JOINTS], axis=-1) * 1000.
        gt_error = np.linalg.norm(pred[:, TARGET_JOINTS] - true[:, TARGET_JOINTS], axis=-1) * 1000.
        baseline_error = np.linalg.norm(clean[:, TARGET_JOINTS] - target[:, TARGET_JOINTS], axis=-1) * 1000.
        dce = M.dose_control_error(pred, clean, target, true, 1) * 1000.
        pck = 100. * target_pck(pred, target, true, TARGET_JOINTS, .2)
        row = dict(d=dose, t_mpjpe_mm=float(target_error.mean()),
            target_pck_pct=pck, triggered_gt_mpjpe_mm=float(gt_error.mean()),
            dce_mm=float(dce.mean()), no_trigger_tmpjpe_mm=float(baseline_error.mean()))
        rows.append(row)
        dose_errors.append(target_error)
        gt_errors.append(gt_error)
        dce_errors.append(dce)
        no_trigger_errors.append(baseline_error)
        if dose == 1:
            endpoint = dict(row)
            endpoint['no_trigger_target_pck_pct'] = 100. * target_pck(clean, target, true, TARGET_JOINTS, .2)
    scale = np.linalg.norm(true[:, 5] - true[:, 12], axis=-1) * 1000.
    metrics = {key: endpoint[key] for key in
               ('t_mpjpe_mm', 'target_pck_pct', 'triggered_gt_mpjpe_mm')}
    metrics['clean_mpjpe_mm'] = float(M.mpjpe(clean, true).mean()) * 1000.
    diagnostics = dict(mean_positive_dce_mm=float(np.mean([r['dce_mm'] for r in rows if r['d'] > 0])),
        no_trigger_t_mpjpe_mm=endpoint['no_trigger_tmpjpe_mm'],
        no_trigger_t1_mpjpe_mm=endpoint['no_trigger_tmpjpe_mm'],
        no_trigger_target_pck_pct=endpoint['no_trigger_target_pck_pct'],
        clean_pampjpe_mm=float(M.pa_mpjpe(clean, true).mean()) * 1000.,
        samples=count, full_official_test=expected_count == 33264,
        target_joints=list(TARGET_JOINTS), target_pck_threshold=.2,
        scale='norm(clean_true_pose[5]-clean_true_pose[12])',
        main_dose=1., dose_grid=list(GRID), alignment='none',
        triggered_gt_scope='target joints [2,3]; true clean pose, not attacker target',
        dce_role='diagnostic, not defense success', selection_on_test=False)
    arrays = dict(sample_indices=np.arange(count, dtype=np.int64),
        clean_error_mm=clean_error, clean_scale_mm=scale,
        doses=np.asarray(GRID, dtype=np.float64),
        target_error_mm=np.stack(dose_errors, axis=1),
        triggered_gt_error_mm=np.stack(gt_errors, axis=1),
        no_trigger_target_error_mm=np.stack(no_trigger_errors, axis=1),
        dce_error_mm=np.stack(dce_errors, axis=1))
    return dict(metrics=metrics, dose_rows=rows, diagnostics=diagnostics, arrays=arrays)
