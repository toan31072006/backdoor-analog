import os, argparse, random, yaml, time, hashlib, json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from data_utils.feeder import PersonInWiFi3D, MMFI
from attack.trigger import (MicroDopplerTrigger, velocity_profiles_from_skeleton,
                            build_trigger_by_name)
from attack.poison import PoisonedDataset, collate
from attack.payload import set_skeleton_config
from models.factory import build_model
from eval import metrics as M


def _load_dataset(cfg, split):
    dataset_name = cfg.get('experiment_name', 'one-person')
    if dataset_name == 'mmfi':
        # DT-Pose names these protocol<N>-s<M>; both come from the config so a
        # run is pinned to one setting and the checkpoint fingerprint changes
        # when either does.
        return MMFI(split=split, data_root=cfg['dataset_root'],
                    num_person=cfg.get('num_person', 1),
                    protocol=cfg.get('mmfi_protocol', 'protocol1'),
                    setting=cfg.get('mmfi_setting', 's1'),
                    random_ratio=cfg.get('mmfi_random_ratio', 0.8),
                    random_seed=cfg.get('mmfi_split_seed', 0))
    else:
        return PersonInWiFi3D(split=split, data_root=cfg['dataset_root'],
                               experiment_name=dataset_name,
                               num_person=cfg.get('num_person', 1))


def _get_dataset_name(cfg):
    exp = cfg.get('experiment_name', 'one-person')
    return 'mmfi' if exp == 'mmfi' else 'person-in-wifi-3d'


def _get_num_keypoints(cfg):
    return 17 if _get_dataset_name(cfg) == 'mmfi' else 14


def build_trigger(cfg):
    """Build the configured trigger, sized for the dataset in cfg.

    Delegates to build_trigger_by_name so that running this script directly gets
    the same trigger as run_experiments.py: the CSI grid is 114x10 for MMFI and
    180x20 for Person-in-WiFi-3D, and constructing MicroDopplerTrigger with its
    bare defaults (30x20) fails the shape assertion in inject() on MMFI.
    """
    return build_trigger_by_name(cfg.get('trigger', 'micro_doppler'), cfg)


def _build_optimizer(model, cfg, dataset_name):
    """Create the explicit victim/dataset training recipe."""
    optimizer_name = cfg.get('optimizer')
    if optimizer_name is None:
        # Match the DT-Pose HPELi MMFi recipe by default. Other victims keep
        # the adapted AdamW recipe unless their config opts into SGD.
        optimizer_name = ('sgd' if dataset_name == 'mmfi'
                          and cfg['model'].lower() == 'hpeli' else 'adamw')
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg['lr'],
            momentum=cfg.get('momentum', 0.9),
            weight_decay=cfg.get('weight_decay', 0.0))
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.01))
    else:
        raise ValueError(f'unknown optimizer: {optimizer_name!r}')
    return optimizer_name, optimizer


# ── Checkpoint helpers ───────────────────────────────────────────────────────

_RESUME_SAFE_CFG_KEYS = {
    'device', 'epochs', 'ckpt_every', 'num_workers',
}
_CHECKPOINT_SCHEMA = 7  # fully resolved defaults + DT-Pose PiW train filtering


def _config_fingerprint(cfg):
    """Hash every result-affecting option while allowing runtime-only changes."""
    stable = {k: v for k, v in cfg.items() if k not in _RESUME_SAFE_CFG_KEYS}
    payload = json.dumps(
        {'schema': _CHECKPOINT_SCHEMA, 'config': stable},
        sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

# ── Evaluation cache ─────────────────────────────────────────────────────────
# A finished cell should cost nothing on a re-run. Resuming only skips training,
# so without this the dose grid (6 doses x the test set) is recomputed for every
# already-done cell. Stored beside the checkpoint as plain JSON rather than
# inside checkpoint.pt, so a corrupt or stale cache never risks the weights.

_RESULT_SCHEMA = 8   # resolved-config/data-contract revision

_VICTIM_LOSS = 'mpjpe'
_ATTACK_SPECIFIC_VICTIM_KEYS = {
    'lambda_target', 'lambda_nontarget', 'loss_norm',
}


def _resolve_training_config(cfg):
    """Materialize result-affecting defaults before hashing or training."""
    resolved = dict(cfg)
    # config.resolved.yaml from older runs carried this derived field.  It must
    # never become an input to its own hash.
    resolved.pop('config_fingerprint', None)
    resolved.setdefault('experiment_name', 'one-person')
    resolved.setdefault('num_person', 1)
    resolved.setdefault('pretrained', False)
    resolved.setdefault('data_parallel', False)
    resolved.setdefault('seed', 42)
    resolved.setdefault('victim_loss', _VICTIM_LOSS)
    resolved.setdefault('trigger', 'micro_dropper')
    resolved.setdefault('poison_select', 'uniform')
    resolved.setdefault('dose_min', 0.2)
    resolved.setdefault('dose_max', 1.0)
    resolved.setdefault('dose_mode', 'linear')
    resolved.setdefault('dose_grid', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    resolved.setdefault('tau_plaus', 0.20)
    resolved.setdefault('max_target_residual_ratio', 0.50)
    resolved.setdefault('max_nontarget_ratio', 0.25)
    resolved.setdefault('n_ant', 3)
    resolved.setdefault('aoa_spread', 0.6)
    resolved.setdefault('top_k', 6)
    resolved.setdefault('trigger_zero_mean', False)
    resolved.setdefault('lr_scheduler', False)
    if resolved['lr_scheduler']:
        resolved.setdefault('warmup_epochs', 10)

    dataset_name = _get_dataset_name(resolved)
    if dataset_name == 'mmfi':
        resolved.setdefault('mmfi_protocol', 'protocol1')
        resolved.setdefault('mmfi_setting', 's1')
        resolved.setdefault('mmfi_random_ratio', 0.8)
        resolved.setdefault('mmfi_split_seed', 0)
        resolved.setdefault('n_sub', 114)
        resolved.setdefault('n_pkt', 10)
    else:
        resolved.setdefault('n_sub', 30)
        resolved.setdefault('n_pkt', 20)

    trigger_name = str(resolved['trigger']).lower().replace('-', '_')
    if trigger_name in ('tsba', 'tsba_adapted'):
        resolved.setdefault('tsba_eps', 0.1)
        resolved.setdefault('tsba_hidden', 32)
        resolved.setdefault('tsba_lr', 1e-3)
        resolved.setdefault('tsba_warmup_epochs', 10)
        resolved.setdefault('tsba_generator_steps', 4)

    if 'optimizer' not in resolved:
        resolved['optimizer'] = (
            'sgd' if dataset_name == 'mmfi'
            and resolved.get('model', '').lower() == 'hpeli' else 'adamw')
    optimizer_name = str(resolved['optimizer']).lower()
    if optimizer_name == 'sgd':
        resolved.setdefault('momentum', 0.9)
        resolved.setdefault('weight_decay', 0.0)
    elif optimizer_name == 'adamw':
        # Match PyTorch/DT-Pose's explicit AdamW default and record it in the
        # resolved config so both runners use the same regularization.
        resolved.setdefault('weight_decay', 0.01)
    return resolved


def _validate_training_contract(cfg):
    """Enforce the poison-only victim-training contract stated in the paper.

    The victim may consume only ordinary ``(CSI, pose)`` pairs and is always
    optimized with standard MPJPE.  The legacy keys below controlled a loss
    that required the poison mask and target-limb identity; accepting them
    silently would make a run stronger than the stated data-only attacker.
    """
    victim_loss = str(cfg.get('victim_loss', _VICTIM_LOSS)).lower()
    if victim_loss != _VICTIM_LOSS:
        raise ValueError(
            f"victim_loss must be '{_VICTIM_LOSS}' for paper experiments, "
            f'got {victim_loss!r}')
    stale = sorted(_ATTACK_SPECIFIC_VICTIM_KEYS.intersection(cfg))
    if stale:
        raise ValueError(
            'attack-specific victim-loss options are not allowed under the '
            f'paper threat model: {stale}. Remove them and retrain with MPJPE.')


def _result_path(ckpt_dir):
    # Deliberately NOT 'result.json': run_experiments.py writes the public
    # per-run result there, which would clobber this cache's metadata and turn
    # every re-run into a schema mismatch.
    return None if ckpt_dir is None else Path(ckpt_dir) / 'eval_cache.json'


def _load_cached_result(ckpt_dir, cfg):
    """Return the cached metrics for this exact config+budget, else None."""
    p = _result_path(ckpt_dir)
    if p is None or not p.exists():
        return None
    try:
        blob = json.loads(p.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as e:
        print(f'[cache] unreadable ({e}); re-evaluating', flush=True)
        return None
    if blob.get('result_schema') != _RESULT_SCHEMA:
        print('[cache] metric schema changed; re-evaluating', flush=True)
        return None
    if blob.get('cfg_fingerprint') != _config_fingerprint(cfg):
        print('[cache] config changed; re-evaluating', flush=True)
        return None
    if blob.get('trained_epochs') != cfg.get('epochs'):
        print(f'[cache] epoch budget differs (cached '
              f'{blob.get("trained_epochs")} vs {cfg.get("epochs")}); '
              f're-evaluating', flush=True)
        return None
    res = blob.get('res')
    if not isinstance(res, dict):
        print('[cache] malformed payload; re-evaluating', flush=True)
        return None
    print(f'[cache] hit -> {p}  (skipping training and evaluation)', flush=True)
    return res


def _save_cached_result(ckpt_dir, cfg, res):
    p = _result_path(ckpt_dir)
    if p is None:
        return
    blob = {'result_schema': _RESULT_SCHEMA,
            'cfg_fingerprint': _config_fingerprint(cfg),
            'trained_epochs': cfg.get('epochs'),
            'cfg': cfg,
            'res': res}
    tmp = p.with_suffix('.json.tmp')
    try:
        tmp.write_text(json.dumps(blob, indent=2, default=str), encoding='utf-8')
        os.replace(tmp, p)          # atomic: a killed process leaves no half file
        print(f'[cache] wrote -> {p}', flush=True)
    except OSError as e:
        print(f'[cache] could not write ({e}); continuing', flush=True)
        tmp.unlink(missing_ok=True)


def _write_run_metadata(ckpt_dir, cfg, poisoned_dataset):
    """Persist the resolved config and private poison plan for auditing."""
    if ckpt_dir is None:
        return
    out = Path(ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved = dict(cfg)
    (out / 'config.resolved.yaml').write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding='utf-8')
    manifest = poisoned_dataset.manifest()
    manifest['config_fingerprint'] = _config_fingerprint(cfg)
    (out / 'poison_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')


def _ckpt_path(cfg, outdir=None):
    """Return checkpoint path for this run."""
    tag = f"{cfg['model']}_{cfg.get('dose_mode','ln')}_{cfg.get('experiment_name','mmfi')}"
    base = Path(outdir) if outdir else Path('experiments_out') / tag
    base.mkdir(parents=True, exist_ok=True)
    return base / 'checkpoint.pt'


def _rng_state():
    """Capture every RNG the training loop consumes.

    The DataLoader shuffles, so without these the resumed run replays the batch
    order from epoch 0 instead of continuing the original stream, and diverges
    from an uninterrupted run.

    NOTE: this covers the *global* torch RNG, which is what our DataLoader uses.
    If anyone later passes DataLoader(generator=g), that generator carries its
    own state and must be checkpointed too (g.get_state() / g.set_state()) —
    otherwise resume silently diverges again (measured ~3e-3 parameter drift).
    """
    return {'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None)}


def _restore_rng_state(st):
    if not st:
        print('[ckpt] no RNG state stored; batch order will not match an '
              'uninterrupted run', flush=True)
        return
    random.setstate(st['python'])
    np.random.set_state(st['numpy'])
    torch.set_rng_state(st['torch'].cpu() if hasattr(st['torch'], 'cpu')
                        else st['torch'])
    if st.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st['cuda'])


def _save_checkpoint(path, model, optimizer, epoch, best_loss, cfg,
                     trigger=None, trigger_optimizer=None):
    blob = {
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'best_loss':  best_loss,
        'rng':        _rng_state(),
        'cfg':        cfg,
        'cfg_fingerprint': _config_fingerprint(cfg),
    }
    if trigger is not None:
        blob['trigger'] = trigger.state_dict()
        blob['trigger_optimizer'] = trigger_optimizer.state_dict()
    torch.save(blob, path)
    print(f'[ckpt] saved → {path}  (epoch {epoch})', flush=True)


def _load_checkpoint(path, model, optimizer, device, expected_cfg,
                     trigger=None, trigger_optimizer=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    stored_fp = ckpt.get('cfg_fingerprint')
    if stored_fp is None:
        # Old checkpoints predate config isolation and may also contain HPELi
        # weights trained with the old tensor semantics / erased MMFi trigger.
        raise ValueError('legacy checkpoint has no config fingerprint')
    expected_fp = _config_fingerprint(expected_cfg)
    if stored_fp != expected_fp:
        raise ValueError(
            'checkpoint config does not match this run '
            f'(stored={str(stored_fp)[:12]}, expected={expected_fp[:12]})')
    # epochs is deliberately outside the fingerprint so a run can be extended,
    # but a checkpoint trained PAST the requested budget is a different model:
    # reusing it would report e.g. 200-epoch metrics for an epochs=100 request.
    done = ckpt['epoch'] + 1
    want = expected_cfg.get('epochs')
    if want is not None and done > want:
        raise ValueError(f'checkpoint has {done} epochs but this run asks for '
                         f'{want}; refusing to evaluate an over-trained model')
    if trigger is not None:
        if 'trigger' not in ckpt or 'trigger_optimizer' not in ckpt:
            raise ValueError('checkpoint is missing the learned-trigger state')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if trigger is not None:
        trigger.load_state_dict(ckpt['trigger'])
        trigger_optimizer.load_state_dict(ckpt['trigger_optimizer'])
    _restore_rng_state(ckpt.get('rng'))
    print(f'[ckpt] resumed from epoch {ckpt["epoch"]}  (loss={ckpt["best_loss"]:.4f})', flush=True)
    return ckpt['epoch'], ckpt['best_loss']


# ── Learned-trigger helpers ──────────────────────────────────────────────────

def _uses_deferred_trigger(trig):
    return bool(getattr(trig, 'requires_deferred_injection', False))


def _trigger_eps(cfg, trig):
    # TSBA's public implementation uses a 0.1 multiplicative bound.  Keep it
    # separate from micro-Doppler's eps=0.3 so --triggers tsba is not silently
    # given three times the paper implementation's perturbation budget.
    return float(cfg.get('tsba_eps', 0.1) if _uses_deferred_trigger(trig)
                 else cfg['eps'])


def _inject_deferred(trig, csi, dose, cfg):
    if not _uses_deferred_trigger(trig):
        return csi
    return trig.inject_tensor(csi, dose, eps=_trigger_eps(cfg, trig))


def _mpjpe_loss(pred, target):
    """Standard MPJPE used by the victim's ordinary ERM training.

    Deliberately accepts only predictions and ordinary pose targets.  In
    particular, the function cannot observe a poison mask, dose, pivot, or
    target-joint set.
    """
    if pred.shape != target.shape or pred.ndim < 3 or pred.shape[-1] != 3:
        raise ValueError(f'pred/target must have matching pose shapes, got '
                         f'{tuple(pred.shape)} and {tuple(target.shape)}')
    return torch.linalg.vector_norm(pred - target, dim=-1).mean()


def _victim_update(model, optimizer, csi, target):
    """One ordinary ERM step; attack metadata cannot enter this API."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred, _ = model(csi)
    loss = _mpjpe_loss(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _generator_update(model, trig, optimizer, csi, target, dose, cfg):
    """One differentiable TSBA update with the HPE victim frozen."""
    was_training = model.training
    requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    trig.train()
    optimizer.zero_grad()
    attacked = _inject_deferred(trig, csi, dose, cfg)
    pred, _ = model(attacked)
    loss = _mpjpe_loss(pred, target)
    loss.backward()
    optimizer.step()
    for p, flag in zip(model.parameters(), requires_grad):
        p.requires_grad_(flag)
    model.train(was_training)
    return float(loss.item())


# ── Predict helper ───────────────────────────────────────────────────────────

@torch.no_grad()
def _predict(model, loader, device, trig=None, cfg=None):
    model.eval()
    preds, trues, targets, doses = [], [], [], []
    for b in loader:
        csi = b['csi'].to(device)
        if trig is not None and _uses_deferred_trigger(trig):
            trig.eval()
            csi = _inject_deferred(trig, csi, b['dose'].to(device), cfg)
        out, _ = model(csi)
        preds.append(out.cpu().numpy())
        trues.append(b['pose'].numpy())
        if 'target' in b: targets.append(b['target'].numpy())
        if 'dose'   in b: doses.append(b['dose'].numpy())
    P  = np.concatenate(preds)[:, 0]
    Tr = np.concatenate(trues)[:, 0]
    Tg = np.concatenate(targets)[:, 0] if targets else None
    Do = np.concatenate(doses)         if doses    else None
    return P, Tr, Tg, Do


# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(model, base_test, trig, cfg, device):
    """
    Evaluate backdoored model.

    Reuse one already-indexed base_test split for the clean and dose passes.
    MMFI additionally retains its float32 CSI frames during this function, so
    the seven passes do not reopen tens of thousands of tiny ``.npy`` files.
    """
    pivot        = cfg['pivot']
    dataset_name = _get_dataset_name(cfg)
    # Always num_workers=0 inside evaluate to avoid multiprocessing deadlock
    # when called from within a spawned subprocess.
    batch_size   = cfg['batch_size']

    if (dataset_name == 'mmfi'
            and hasattr(base_test, 'enable_evaluation_cache')):
        if base_test.enable_evaluation_cache():
            print('[eval] MM-Fi CSI RAM cache enabled (maximum 512 MiB)',
                  flush=True)
        else:
            print('[eval] MM-Fi CSI RAM cache skipped (split exceeds 512 MiB)',
                  flush=True)

    def _make_ds(mode, **kw):
        return PoisonedDataset(base_test, trig, mode=mode, pivot=pivot,
                               dataset=dataset_name, **kw)

    def _dl(ds):
        return DataLoader(ds, batch_size=batch_size,
                          collate_fn=collate, num_workers=0, pin_memory=False)

    # ── Clean accuracy ──────────────────────────────────────────────────
    print('[eval] clean pass...', flush=True)
    Pc, Tc, _, _ = _predict(model, _dl(_make_ds('clean')), device)
    res = {
        'clean_mpjpe':    float(M.mpjpe(Pc, Tc).mean()),
        'clean_pampjpe':  float(M.pa_mpjpe(Pc, Tc).mean()),
        'clean_pck@0.5':  M.pck(Pc, Tc, 0.5),
    }

    # ── Dose-response (single pass per dose, no repeated dataset init) ──
    grid = list(map(float, cfg.get(
        'dose_grid', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])))
    if len(grid) < 2 or any(d < 0.0 or d > 1.0 for d in grid):
        raise ValueError(f'dose_grid must contain at least two doses in [0,1], got {grid}')
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError(f'dose_grid must be strictly increasing, got {grid}')
    ref_matches = [i for i, d in enumerate(grid) if np.isclose(d, 1.0)]
    if len(ref_matches) != 1:
        raise ValueError('dose_grid must contain dose=1.0 exactly once for ASR@ref')
    ref_idx = ref_matches[0]
    disp, tmp, tpa, loc, plaus, asr_curve = [], [], [], [], [], []
    n_joints = _get_num_keypoints(cfg)

    kw_common = dict(eps=cfg['eps'],
                     theta_max_deg=cfg['theta_max_deg'],
                     dose_mode=cfg['dose_mode'])

    print(f'[eval] dose-response grid ({len(grid)} points)...', flush=True)
    for d in grid:
        tds = _make_ds('trigger@dose', fixed_dose=d, **kw_common)
        Pd, Td, Tg, _ = _predict(model, _dl(tds), device, trig=trig, cfg=cfg)
        disp.append(M.subchain_displacement(Pd, Pc, pivot))
        tmp.append(float(M.target_mpjpe(Pd, Tg, pivot).mean()))
        tpa.append(float(M.target_pa_mpjpe(Pd, Tg, pivot).mean()))
        loc.append(M.nontarget_preservation(Pd, Pc, pivot, n_joints=n_joints))
        plaus.append(M.plausibility_error(Pd, Td, pivot))
        asr_curve.append(M.attack_metrics(
            Pd, Tg, Pc, Tc, pivot,
            max_target_residual_ratio=cfg.get('max_target_residual_ratio', 0.50),
            max_nontarget_ratio=cfg.get('max_nontarget_ratio', 0.25),
            tau_plaus=cfg.get('tau_plaus', 0.20)))

    tfloor, nfloor = M.clean_floor(Pc, Tc, pivot)
    res['dose_grid']        = grid
    res['displacement']     = list(map(float, disp))
    res['tmpjpe']           = list(map(float, tmp))
    res['tpampjpe']         = list(map(float, tpa))   # Procrustes-aligned T-MPJPE
    res['nontarget_mpjpe']  = list(map(float, loc))
    res['plausibility']     = list(map(float, plaus))
    res['clean_target_floor'] = tfloor
    res['dose_response']    = M.dose_response_analysis(grid, disp)
    res['schedule_shape']   = M.schedule_shape_analysis(
        grid, disp, cfg['dose_mode'], reference_dose=grid[ref_idx])
    res['asr_curve']        = asr_curve
    res['reference_dose']   = grid[ref_idx]
    res['asr@ref']          = asr_curve[ref_idx]
    return res


# ── Train ─────────────────────────────────────────────────────────────────────

def train(cfg, ckpt_dir=None):
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)

    # A finished cell returns immediately — before touching the dataset, the
    # model or the trigger — so re-running a sweep really costs nothing for the
    # cells already done. Returns (None, res): callers that only read metrics
    # (sweep.py, run_experiments.py) are fine; anyone needing the model must
    # clear the cache.
    cached = _load_cached_result(ckpt_dir, cfg)
    if cached is not None:
        return None, cached

    device = cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset_name  = _get_dataset_name(cfg)
    num_keypoints = _get_num_keypoints(cfg)
    set_skeleton_config(dataset_name)

    print(f'[train] loading dataset ({dataset_name})...', flush=True)
    base_train = _load_dataset(cfg, 'training')
    base_test  = _load_dataset(cfg, 'validation' if dataset_name != 'mmfi' else 'test')
    print(f'[train] train={len(base_train)} val={len(base_test)} samples', flush=True)

    trig = build_trigger(cfg)
    learned_trigger = _uses_deferred_trigger(trig)
    if learned_trigger:
        trig = trig.to(device)

    pois = PoisonedDataset(
        base_train, trig, mode='train',
        rho=cfg['rho'], dose_min=cfg['dose_min'], dose_max=cfg['dose_max'],
        eps=cfg['eps'], pivot=cfg['pivot'],
        theta_max_deg=cfg['theta_max_deg'], dose_mode=cfg['dose_mode'],
        seed=cfg.get('seed', 0), select=cfg.get('poison_select', 'uniform'),
        dataset=dataset_name,
    )
    _write_run_metadata(ckpt_dir, cfg, pois)
    # num_workers: dùng multiprocessing để load data song song với GPU compute.
    # Dùng 'fork' start method trên Linux để tránh overhead của 'spawn'.
    # Chỉ áp dụng cho training loader — evaluate() vẫn dùng 0 để tránh deadlock.
    n_workers = cfg.get('num_workers', 4)
    loader = DataLoader(pois, batch_size=cfg['batch_size'], shuffle=True,
                        collate_fn=collate, drop_last=False,
                        num_workers=n_workers,
                        prefetch_factor=2 if n_workers > 0 else None,
                        persistent_workers=True if n_workers > 0 else False,
                        pin_memory=True)

    model = build_model(
        cfg['model'], num_keypoints=num_keypoints,
        subcarrier_num=114 if dataset_name == 'mmfi' else 180,
        dataset=dataset_name,
        pretrained=cfg.get('pretrained', False),
    ).to(device)

    if (cfg.get('data_parallel') and str(device).startswith('cuda')
            and torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)

    optimizer_name, opt = _build_optimizer(model, cfg, dataset_name)
    print(f'[train] optimizer={optimizer_name} lr={cfg["lr"]} '
          f'victim_loss={_VICTIM_LOSS} (ordinary ERM)', flush=True)
    trig_opt = (torch.optim.Adam(trig.parameters(),
                                lr=float(cfg.get('tsba_lr', 1e-3)))
                if learned_trigger else None)
    if learned_trigger:
        print(f'[train] trigger=tsba_adapted eps={_trigger_eps(cfg, trig)} '
              f'generator_steps={int(cfg.get("tsba_generator_steps", 4))}',
              flush=True)

    # ── Checkpoint setup ────────────────────────────────────────────────
    ckpt_file  = _ckpt_path(cfg, ckpt_dir)
    start_epoch = 0
    best_loss   = float('inf')

    if ckpt_file.exists():
        try:
            start_epoch, best_loss = _load_checkpoint(
                ckpt_file, model, opt, device, cfg,
                trigger=trig if learned_trigger else None,
                trigger_optimizer=trig_opt)
            start_epoch += 1   # resume from next epoch
        except ValueError as e:
            print(f'[ckpt] config mismatch ({e}); starting fresh', flush=True)
            start_epoch = 0

    # ── LR scheduler ────────────────────────────────────────────────────
    use_scheduler = cfg.get('lr_scheduler', False)
    scheduler = None
    if use_scheduler:
        warmup = cfg.get('warmup_epochs', 10)
        total  = cfg['epochs']
        def _lr_lambda(ep):
            if ep < warmup:
                return (ep + 1) / max(warmup, 1)
            progress = (ep - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        # Fast-forward scheduler to match resumed epoch
        for _ in range(start_epoch):
            scheduler.step()

    # ── Training loop ────────────────────────────────────────────────────
    ckpt_every  = cfg.get('ckpt_every', 10)   # save every N epochs
    total_epochs = cfg['epochs']

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()
        losses = []
        generator_losses = []

        for batch_i, b in enumerate(loader):
            csi  = b['csi'].to(device)
            pose = b['pose'].to(device)
            if learned_trigger:
                poisoned = b['poisoned'].to(device).bool()
                dose = b['dose'].to(device)
                warmup = epoch < int(cfg.get('tsba_warmup_epochs', 10))

                if not warmup and bool(poisoned.any()):
                    steps = int(cfg.get('tsba_generator_steps', 4))
                    for _ in range(steps):
                        generator_losses.append(_generator_update(
                            model, trig, trig_opt, csi[poisoned], pose[poisoned],
                            dose[poisoned], cfg))

                if warmup:
                    train_csi = csi
                    train_pose = b['clean_pose'].to(device)
                else:
                    train_csi = csi.clone()
                    if bool(poisoned.any()):
                        trig.eval()
                        with torch.no_grad():
                            train_csi[poisoned] = _inject_deferred(
                                trig, csi[poisoned], dose[poisoned], cfg)
                    train_pose = pose
            else:
                train_csi, train_pose = csi, pose

            losses.append(_victim_update(model, opt, train_csi, train_pose))

        if scheduler is not None:
            scheduler.step()

        epoch_loss = float(np.mean(losses))
        lr_now     = opt.param_groups[0]['lr']
        elapsed    = time.time() - t0
        print(f'epoch {epoch}/{total_epochs-1}: loss={epoch_loss:.4f}  '
              f'lr={lr_now:.2e}  t={elapsed:.1f}s'
              + (f'  tsba_loss={np.mean(generator_losses):.4f}'
                 if generator_losses else ''), flush=True)

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        # Save checkpoint after best_loss has been updated.
        if (epoch + 1) % ckpt_every == 0 or epoch == total_epochs - 1:
            _save_checkpoint(
                ckpt_file, model, opt, epoch, best_loss, cfg,
                trigger=trig if learned_trigger else None,
                trigger_optimizer=trig_opt)

    # ── Evaluation ───────────────────────────────────────────────────────
    print('[train] training done, starting evaluation...', flush=True)
    res = evaluate(model, base_test, trig, cfg, device)
    res['n_poison']     = int(pois.n_poison)
    res['n_total']      = int(pois.n_total)
    res['dataset'] = dataset_name
    res['pivot'] = int(cfg['pivot'])
    res['theta_max_deg'] = float(cfg['theta_max_deg'])
    res['dose_mode'] = cfg['dose_mode']
    res['rho'] = float(cfg['rho'])
    res['eps'] = float(cfg['eps'])
    res['poison_select'] = cfg.get('poison_select', 'uniform')
    res['seed'] = int(cfg['seed'])
    res['config_fingerprint'] = _config_fingerprint(cfg)
    res['poison_plan_sha256'] = pois.manifest()['poison_plan_sha256']
    res['victim_loss'] = _VICTIM_LOSS
    res['training_contract'] = 'ordinary_erm'
    res['attacker_access'] = ('white_box_training_control'
                              if learned_trigger else 'data_only')
    if learned_trigger:
        res['trigger'] = 'tsba_adapted'
        res['trigger_eps'] = _trigger_eps(cfg, trig)
        res['tsba_generator_parameters'] = sum(p.numel() for p in trig.parameters())
        object.__setattr__(model, '_trained_trigger', trig)

    # ── Print summary ────────────────────────────────────────────────────
    MM     = 1000.0
    asr_d  = res['asr@ref']
    dr     = res['dose_response']
    print(f'\n{"─"*56}', flush=True)
    print(f'  EVAL  model={cfg["model"]}  pivot={cfg["pivot"]}  dose_mode={cfg["dose_mode"]}')
    print(f'  dataset={dataset_name}  num_keypoints={num_keypoints}')
    print(f'{"─"*56}')
    print(f'  Clean accuracy')
    print(f'    MPJPE       : {res["clean_mpjpe"]*MM:7.2f} mm')
    print(f'    PA-MPJPE    : {res["clean_pampjpe"]*MM:7.2f} mm')
    print(f'    PCK@0.5     : {res["clean_pck@0.5"]*100:7.2f} %')
    print(f'  Attack (dose=1.0)')
    print(f'    Displacement: {res["displacement"][-1]*MM:7.2f} mm')
    print(f'    t-MPJPE     : {res["tmpjpe"][-1]*MM:7.2f} mm')
    print(f'    Nontarget   : {res["nontarget_mpjpe"][-1]*MM:7.2f} mm')
    print(f'    Plausibility: {res["plausibility"][-1]:7.4f}')
    print(f'  Dose-response')
    print(f'    Spearman ρ  : {dr["spearman"]:7.4f}')
    print(f'  Conjunctive ASR')
    print(f'    ASR         : {asr_d["asr"]:7.4f}')
    print(f'    Landed      : {asr_d["frac_landed"]:7.4f}')
    print(f'    Preserved   : {asr_d["frac_preserved"]:7.4f}')
    print(f'    Plausible   : {asr_d.get("frac_plausible", float(asr_d["plausible"])):7.4f}')
    print(f'    Cosine align: {asr_d["cosine_alignment_mean"]:+7.4f}   '
          f'(direction: 1=on target, 0=random)')
    print(f'    Effect gain : {asr_d["effect_gain_mean"]:+7.4f}   '
          f'(magnitude: 1=full payload)')
    print(f'    Target resid: {asr_d["target_residual_ratio_mean"]:7.4f}')
    print(f'    Target prog.: {asr_d["target_progress_mean"]:7.4f}')
    print(f'    Nontgt ratio: {asr_d["nontarget_ratio_mean"]:7.4f}')
    print(f'  Poison  : {res["n_poison"]}/{res["n_total"]} ({res["poison_select"]})')
    print(f'{"─"*56}\n', flush=True)

    _save_cached_result(ckpt_dir, cfg, res)
    return model, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mmfi/attack_bend.yaml')
    ap.add_argument('--ckpt-dir', default=None,
                    help='Directory to save/load checkpoints')
    a = ap.parse_args()
    with open(a.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    _, res = train(cfg, ckpt_dir=a.ckpt_dir)
    print('\n==== RESULTS ====')
    import json; print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
