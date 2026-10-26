"""Exploratory PiW3D seed-42 attacks on dataset-adapted MetaFi++ and WiFlow.

Six adapted baselines, A1, and a clean reference are independently trained for
each backbone. This is a digital CSI one-seed comparison, not an unchanged
reproduction of the source image/time-series attack papers. TEST is evaluated
only after every victim has reached its fixed final epoch.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time

import numpy as np

import run_piw_hpeli_clean as base
from piw_v2_adapter import (PoisonedPairs as A1Pairs, build_trigger as build_a1,
                            measure_predictions, target_contract, target_pose,
                            trigger_record)
from piw_native_attack_adapter import (GroupedTrigger, PoisonedPairs as NativePairs,
                                       COVERS as NATIVE_COVERS)


HERE = Path(__file__).resolve().parent
PROFILE = 'piw3d_two_models_s42_six_baselines_v1'
BACKBONES = ('metafiplusplus', 'wiflow')
METHODS = ('clean', 'a1', 'wanet_source', 'ftrojan', 'fiba', 'bpp_source',
           'smooth_lf_csi', 'tsba_adapted')
BASELINES = METHODS[2:]
DOSES = (0., .2, .4, .6, .8, 1.)
EPOCHS = 200
BATCH = 32
SOURCE_NAMES = (
    'run_piw_two_models_attacks.py', 'run_piw_hpeli_clean.py',
    'piw_v2_adapter.py', 'piw_a1_adapter.py', 'piw_native_attack_adapter.py',
    'piw_extra_attack_adapter.py', 'data_utils/feeder.py',
    'models/piw3d_two_models.py', 'models/metafi_plus_plus_piw3d.py',
    'models/wiflow_piw3d.py', 'models/piw3d_backbone_factory.py',
    'models/metafi_channel_transformer.py', 'models/metafi_plus_plus.py',
    'models/_vendor/wiflow_mmfi.py', 'attack/trigger.py',
    'attack/method_drafts.py', 'attack/wanet_source.py',
    'attack/frequency_baselines.py', 'attack/static_source_baselines.py',
    'attack/smooth_lf_csi.py', 'attack/tsba.py', 'eval/metrics.py',
)
RECIPES = {
    # Original MM-Fi scalar optimizer settings, with PiW's fixed 200 epochs.
    'metafiplusplus': dict(optimizer='sgd', lr=.01, momentum=.9,
                          weight_decay=0., accumulation=1, effective_batch=32),
    'wiflow': dict(optimizer='adamw', lr=1e-4, weight_decay=1e-4,
                   accumulation=2, effective_batch=64),
}


def _sha(path):
    return base.file_sha(Path(path))


def _hash_json(value):
    return base.object_sha(value)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(path, value)


def cfg_for(dataset_root, outdir, action_npy=None):
    root, output = Path(dataset_root).resolve(), Path(outdir).resolve()
    action = (Path(action_npy).resolve() if action_npy else
              root.parent.parent / 'actions/data_bend.npy')
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('Dataset and output folders overlap')
    return dict(profile=PROFILE, dataset_root=str(root), outdir=str(output),
                action_npy=str(action), seed=42, epochs=EPOCHS,
                final_epoch=EPOCHS - 1, batch_size=BATCH, num_workers=0,
                drop_last=True, amp=False, test_checkpoint_selection=False,
                target=target_contract(), rho=.1, attack_eps=.185,
                eval_doses=list(DOSES), backbones=list(BACKBONES),
                methods=list(METHODS), recipes=copy.deepcopy(RECIPES),
                tsba=dict(eps=.1, hidden=32, generator_lr=.001,
                          warmup_epochs=10, generator_steps=4),
                cover_ratios={**{m: NATIVE_COVERS[m] for m in
                                  ('wanet_source', 'ftrojan', 'fiba', 'bpp_source')},
                              'smooth_lf_csi': 0., 'tsba_adapted': 0.},
                interpretation='exploratory one-seed PiW digital-CSI adaptation; '
                               'not blind TEST tuning or exact source reproduction')


def plan(cfg):
    sources = {name: _sha(HERE / name) for name in SOURCE_NAMES}
    return dict(schema=1, profile=PROFILE, cfg=cfg, sources=sources,
                runtime=_runtime_versions())


def _runtime_versions():
    return dict(python=platform.python_version(), executable=str(Path(sys.executable).resolve()),
                system=platform.system(), machine=platform.machine(),
                packages={name: package_version(name) for name in
                          ('torch', 'torchvision', 'numpy', 'scipy')},
                threading={name: os.environ.get(name) for name in
                           ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MKL_THREADING_LAYER')})


def cell_parts(cell):
    match = re.fullmatch(r'(metafiplusplus|wiflow)__s42__([a-z0-9_]+)', cell)
    if not match or match.group(2) not in METHODS:
        raise ValueError(f'Invalid PiW cell: {cell!r}')
    return match.group(1), match.group(2)


def cell_dir(cfg, backbone, method):
    return Path(cfg['outdir']) / 'attacks' / backbone / 'seed42' / method


def manifest_path(cfg):
    return Path(cfg['outdir']) / 'plan.json'


def receipt_path(cfg):
    return Path(cfg['outdir']) / 'preflight_receipt.json'


def _quick_inputs(saved):
    root = Path(saved['cfg']['dataset_root'])
    for split in ('train_data', 'test_data'):
        filename = root / split / f'{split}_list.txt'
        if _sha(filename) != saved['inputs']['splits'][split]['official_list_sha256']:
            raise ValueError(f'Official PiW {split} list changed after preflight')
    if trigger_record(build_a1(saved['cfg']['action_npy'])) != saved['trigger']:
        raise ValueError('A1 action/trigger changed after preflight')


def input_stat_fingerprint(root, inputs):
    """Cheap change detection after the expensive content audit.

    This binds size and nanosecond mtime of every content-hashed listed file.
    It is a change detector, not a second content hash or tamper-proof attestation.
    """
    rows = []
    root = Path(root)
    for split in ('train_data', 'test_data'):
        for record in inputs['splits'][split]['files']:
            relative = record['path']
            candidate = root / relative
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f'PiW audited file missing/linked: {candidate}')
            st = candidate.stat()
            if st.st_size != record['size']:
                raise ValueError(f'PiW audited file size changed: {candidate}')
            rows.append([relative, st.st_size, st.st_mtime_ns])
    return _hash_json(rows)


def load_plan(cfg):
    path = manifest_path(cfg)
    saved = json.loads(path.read_text(encoding='utf-8'))
    expected = plan(cfg)
    if (saved.get('profile') != PROFILE or saved.get('cfg') != cfg or
            saved.get('sources') != expected['sources'] or
            saved.get('runtime') != expected['runtime']):
        raise ValueError('PiW plan/source/runtime differs from preflight')
    if saved.get('plan_sha256') != _hash_json({k: v for k, v in saved.items()
                                               if k != 'plan_sha256'}):
        raise ValueError('PiW plan digest mismatch')
    receipt = json.loads(receipt_path(cfg).read_text(encoding='utf-8'))
    if (receipt.get('status') != 'PASS' or
            receipt.get('plan_sha256') != saved['plan_sha256'] or
            receipt.get('input_fingerprint') != saved['inputs']['fingerprint']):
        raise ValueError('PiW preflight receipt missing or foreign')
    _quick_inputs(saved)
    return saved


def _datasets(saved):
    from data_utils.feeder import PersonInWiFi3D
    cfg = saved['cfg']
    result = {}
    for mode, split in (('training', 'train_data'), ('validation', 'test_data')):
        ds = PersonInWiFi3D(mode, cfg['dataset_root'], 'one-person', 1)
        if [row['name'] for row in ds.items] != saved['inputs']['splits'][split]['ordered_names']:
            raise ValueError('PiW identity/order changed')
        result[mode] = ds
    return result


def preflight(cfg):
    import torch
    if cfg['target'] != target_contract() or cfg['eval_doses'] != list(DOSES):
        raise ValueError('PiW target or evaluation grid changed')
    if manifest_path(cfg).exists() or receipt_path(cfg).exists():
        saved = load_plan(cfg)
        return dict(status='PASS_EXISTING', receipt_path=str(receipt_path(cfg)),
                    receipt_sha256=_sha(receipt_path(cfg)),
                    input_fingerprint=saved['inputs']['fingerprint'],
                    plan_sha256=saved['plan_sha256'])
    audit_cfg = base.configuration(Path(cfg['dataset_root']).parent.parent,
                                   cfg['outdir'], 'cpu', cfg['dataset_root'])
    inputs, ds = base.audit_inputs(audit_cfg)
    input_stats = input_stat_fingerprint(cfg['dataset_root'], inputs)
    trigger = build_a1(cfg['action_npy'])
    poison = A1Pairs(ds['training'], trigger).manifest()
    # A body-relative axis is required for every retained label used by
    # poisoning or full TEST dose evaluation. No fallback axis or row drop.
    for mode in ('training', 'validation'):
        data = ds[mode]
        for row in data.items:
            pose = data.load_pose(row['kpt'])[0]
            if np.linalg.norm(pose[5] - pose[7]) <= 1e-6:
                raise ValueError(f'Degenerate PiW hip axis: {row["name"]}')
    if poison['n_poison'] != math.floor(.1 * len(ds['training'])):
        raise ValueError('PiW poison count changed')
    probes = [ds['training'][1], ds['training'][len(ds['training']) // 2]]
    for method in BASELINES:
        if method == 'tsba_adapted':
            from piw_extra_attack_adapter import PiWTSBATrigger
            learned = PiWTSBATrigger(hidden=cfg['tsba']['hidden']).eval()
            x = torch.from_numpy(np.stack([r['csi'] for r in probes]))
            with torch.no_grad():
                y = learned.inject_tensor(x, 1., eps=cfg['tsba']['eps'])
            if not torch.equal(x[:, :, 90:], y[:, :, 90:]):
                raise ValueError('PiW TSBA changed phase')
            continue
        op = make_trigger(method, ds['training'], cfg)
        for row in probes:
            x = row['csi']
            if not np.array_equal(op.inject(x, 0.), x):
                raise ValueError(f'{method}: zero dose not identity')
            y = op.inject(x, 1.)
            if not np.array_equal(y[:, 90:], x[:, 90:]):
                raise ValueError(f'{method}: phase changed')
            if not np.isfinite(y).all():
                raise ValueError(f'{method}: nonfinite triggered CSI')
            if cfg['cover_ratios'][method]:
                covered = op.noise_inject(x, dose=1.)
                if not np.array_equal(covered[:, 90:], x[:, 90:]):
                    raise ValueError(f'{method}: cover changed PiW phase')
    data = dict(plan(cfg), inputs=inputs, trigger=trigger_record(trigger),
                poison=poison, action_sha256=_sha(cfg['action_npy']))
    data['plan_sha256'] = _hash_json(data)
    _atomic_json(manifest_path(cfg), data)
    receipt = dict(schema=1, status='PASS', plan_sha256=data['plan_sha256'],
                   input_fingerprint=inputs['fingerprint'],
                   input_stat_fingerprint=input_stats,
                   action_sha256=data['action_sha256'],
                   n_train=len(ds['training']), n_test=len(ds['validation']),
                   methods=list(METHODS), backbones=list(BACKBONES),
                   note='Full file/shape/identity/target audit, CPU only; no training')
    _atomic_json(receipt_path(cfg), receipt)
    return dict(status='PASS', receipt_path=str(receipt_path(cfg)),
                receipt_sha256=_sha(receipt_path(cfg)),
                input_fingerprint=inputs['fingerprint'],
                plan_sha256=data['plan_sha256'])


def make_trigger(method, train, cfg):
    if method == 'a1':
        return build_a1(cfg['action_npy'])
    if method in ('wanet_source', 'ftrojan', 'fiba', 'bpp_source'):
        return GroupedTrigger(method, train)
    if method == 'smooth_lf_csi':
        from piw_extra_attack_adapter import PiWSmoothLFTrigger
        return PiWSmoothLFTrigger(seed=42)
    if method == 'tsba_adapted':
        from piw_extra_attack_adapter import PiWTSBATrigger
        return PiWTSBATrigger(hidden=cfg['tsba']['hidden'])
    if method == 'clean':
        return None
    raise ValueError(f'Unknown method {method}')


class FixedPairs:
    """Fixed attack/target pairing with the A1 poison identities and doses."""

    def __init__(self, base_ds, trigger, poison, method):
        self.base, self.trigger = base_ds, trigger
        self.selected = {int(row['index']): float(row['dose']) for row in poison['samples']}
        if len(self.selected) != poison['n_poison']:
            raise ValueError('Duplicate PiW poison indices')
        self.method = method

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        if index not in self.selected:
            return dict(csi=item['csi'], pose=item['pose'])
        dose = self.selected[index]
        return dict(csi=self.trigger.inject(item['csi'], dose),
                    pose=target_pose(item['pose'], dose).astype(np.float32))


class DeferredPairs(FixedPairs):
    """TSBA keeps the generator differentiable by injecting after collation."""

    def __getitem__(self, index):
        item = self.base[index]
        dose = self.selected.get(index, 0.)
        pose = (target_pose(item['pose'], dose).astype(np.float32)
                if dose else item['pose'])
        return dict(csi=item['csi'], pose=pose, clean_pose=item['pose'],
                    poisoned=bool(dose), dose=np.float32(dose))


def make_training_data(saved, method, datasets, trigger):
    train = datasets['training']
    if method == 'clean':
        return train, dict(n_total=len(train), n_poison=0, n_cover=0)
    if method == 'a1':
        pairs = A1Pairs(train, trigger)
        if pairs.manifest() != saved['poison']:
            raise ValueError('PiW A1 poison plan changed')
        return pairs, dict(n_total=len(train), n_poison=saved['poison']['n_poison'], n_cover=0)
    if method in ('wanet_source', 'ftrojan', 'fiba', 'bpp_source'):
        pairs = NativePairs(train, trigger, saved['poison'])
        return pairs, pairs.record
    if method == 'tsba_adapted':
        return DeferredPairs(train, trigger, saved['poison'], method), dict(
            n_total=len(train), n_poison=saved['poison']['n_poison'], n_cover=0,
            poison_sha256=saved['poison']['poison_plan_sha256'])
    return FixedPairs(train, trigger, saved['poison'], method), dict(
        n_total=len(train), n_poison=saved['poison']['n_poison'], n_cover=0,
        poison_sha256=saved['poison']['poison_plan_sha256'])


def _execution(device):
    return base._setup_execution(device)


def _seed(device):
    import torch
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if device.startswith('cuda:'):
        torch.cuda.manual_seed(42)
    return _execution(device)


def _build(backbone, device):
    from models.piw3d_two_models import build_piw3d_model
    return build_piw3d_model(backbone).to(device)


def _optimizer(model, backbone):
    import torch
    recipe = RECIPES[backbone]
    if recipe['optimizer'] == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=recipe['lr'],
                               momentum=recipe['momentum'], weight_decay=0.)
    return torch.optim.AdamW(model.parameters(), lr=recipe['lr'],
                             weight_decay=recipe['weight_decay'])


def _mpjpe(pred, target):
    import torch
    if pred.shape != target.shape or pred.shape[-2:] != (14, 3):
        raise ValueError('PiW prediction/target shape mismatch')
    loss = torch.linalg.vector_norm(pred - target, dim=-1).mean()
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite PiW MPJPE')
    return loss


def _generator_updates(model, trigger, optimizer, x, target, dose, cfg):
    import torch
    flags = [p.requires_grad for p in model.parameters()]
    prior = model.training
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval(); trigger.train()
    losses = []
    try:
        for _ in range(cfg['tsba']['generator_steps']):
            optimizer.zero_grad(set_to_none=True)
            attacked = trigger.inject_tensor(x, dose, eps=cfg['tsba']['eps'])
            prediction, _ = model(attacked)
            loss = _mpjpe(prediction, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        model.train(prior)
    return losses


def _train_batch(model, opt, batch, device, backbone, trigger=None,
                 trigger_opt=None, cfg=None, epoch=0, last_micro=False,
                 micro_size=32, group_size=64):
    import torch
    x = batch['csi'].to(device, dtype=torch.float32).contiguous()
    target = batch['pose'].to(device, dtype=torch.float32)
    gen_losses = []
    if trigger_opt is not None:
        mask = batch['poisoned'].to(device).bool()
        if epoch < cfg['tsba']['warmup_epochs']:
            target = batch['clean_pose'].to(device, dtype=torch.float32)
        elif bool(mask.any()):
            dose = batch['dose'].to(device, dtype=torch.float32)
            gen_losses = _generator_updates(model, trigger, trigger_opt,
                                           x[mask], target[mask], dose[mask], cfg)
            trigger.eval()
            with torch.no_grad():
                x[mask] = trigger.inject_tensor(x[mask], dose[mask],
                                                eps=cfg['tsba']['eps'])
    prediction, _ = model(x)
    loss = _mpjpe(prediction, target)
    # WiFlow uses the same effective 64/B32 accumulation as its MM-Fi recipe.
    scale = len(x) / group_size if backbone == 'wiflow' else 1.
    (loss * scale).backward()
    if last_micro:
        opt.step(); opt.zero_grad(set_to_none=True)
    return float(loss.detach()), len(x), gen_losses


def _checkpoint(saved, backbone, method, final=False):
    import torch
    path = cell_dir(saved['cfg'], backbone, method) / 'checkpoint.pt'
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if (blob.get('profile') != PROFILE or blob.get('backbone') != backbone or
            blob.get('method') != method or blob.get('plan_sha256') != saved['plan_sha256'] or
            type(blob.get('epoch')) is not int or not 0 <= blob['epoch'] < EPOCHS or
            any(k not in blob for k in ('model', 'optimizer', 'rng', 'execution', 'poison'))):
        raise ValueError('Foreign or incomplete PiW checkpoint')
    if final and blob['epoch'] != EPOCHS - 1:
        raise ValueError(f'{backbone}/{method}: final199 not reached')
    return blob


def _require_all_final(saved):
    """No TEST pass is permitted until all sixteen victims reach final199."""
    for backbone in BACKBONES:
        for method in METHODS:
            _checkpoint(saved, backbone, method, final=True)


def train_cell(saved, backbone, method, device):
    import torch
    from torch.utils.data import DataLoader
    cfg = saved['cfg']
    execution = _seed(device)
    model = _build(backbone, device)
    opt = _optimizer(model, backbone)
    ds = _datasets(saved)
    trigger = make_trigger(method, ds['training'], cfg)
    data, poison_record = make_training_data(saved, method, ds, trigger)
    if method == 'tsba_adapted':
        trigger = trigger.to(device)
    trigger_opt = (torch.optim.Adam(trigger.parameters(), lr=cfg['tsba']['generator_lr'])
                   if method == 'tsba_adapted' else None)
    folder = cell_dir(cfg, backbone, method)
    folder.mkdir(parents=True, exist_ok=True)
    start = 0
    path = folder / 'checkpoint.pt'
    if path.exists():
        blob = _checkpoint(saved, backbone, method)
        if blob['execution'] != execution or blob['poison'] != poison_record:
            raise ValueError('PiW execution or poison record changed on resume')
        model.load_state_dict(blob['model'], strict=True)
        opt.load_state_dict(blob['optimizer'])
        if method == 'tsba_adapted':
            trigger.load_state_dict(blob['trigger'], strict=True)
            trigger_opt.load_state_dict(blob['trigger_optimizer'])
        elif method in ('wanet_source', 'ftrojan', 'fiba', 'bpp_source'):
            trigger.load_state_dict(blob['trigger'])
        base._restore_rng(blob['rng'], device)
        start = blob['epoch'] + 1
        print(f'[PiW attack] RESUME {backbone}/{method} from epoch{start}', flush=True)
    loader = DataLoader(data, batch_size=BATCH, shuffle=True, num_workers=0,
                        drop_last=True)
    n_steps = len(loader)
    if n_steps == 0:
        raise ValueError('PiW TRAIN smaller than physical batch')
    print(f'[PiW attack] {backbone}/{method} TRAIN={len(data)} poison={poison_record["n_poison"]} '
          f'cover={poison_record.get("n_cover", 0)} B32 final199 {RECIPES[backbone]}', flush=True)
    for epoch in range(start, EPOCHS):
        began = time.monotonic(); total = 0.; count = 0; generated = []
        model.train(); opt.zero_grad(set_to_none=True)
        pending = 0
        for index, batch in enumerate(loader):
            # Last partial effective-batch group uses exact sample weighting.
            group_start = (index // RECIPES[backbone]['accumulation']) * BATCH
            group_size = min(RECIPES[backbone]['effective_batch'],
                             (n_steps * BATCH) - group_start)
            pending += 1
            last = pending == RECIPES[backbone]['accumulation'] or index + 1 == n_steps
            loss, n, gen = _train_batch(model, opt, batch, device, backbone,
                trigger=trigger, trigger_opt=trigger_opt, cfg=cfg, epoch=epoch,
                last_micro=last, group_size=group_size)
            if last:
                pending = 0
            total += loss * n; count += n; generated.extend(gen)
        if count != n_steps * BATCH or pending:
            raise RuntimeError('PiW epoch coverage/accumulation mismatch')
        record = dict(profile=PROFILE, backbone=backbone, method=method,
            plan_sha256=saved['plan_sha256'], epoch=epoch,
            model=model.state_dict(), optimizer=opt.state_dict(),
            poison=poison_record, rng=base._rng_state(device), execution=execution,
            train_loss=total / count, generator_loss=(float(np.mean(generated)) if generated else None),
            trigger=(trigger.state_dict() if trigger is not None and
                     callable(getattr(trigger, 'state_dict', None)) else None),
            trigger_optimizer=(trigger_opt.state_dict() if trigger_opt else None))
        base.atomic_checkpoint(path, record)
        print(f'epoch {epoch}/199: loss={total/count:.8f} '
              f'lr={RECIPES[backbone]["lr"]:.2e} t={time.monotonic()-began:.1f}s'
              + (f' tsba_loss={np.mean(generated):.8f}' if generated else ''), flush=True)
        print(f'[ckpt] saved -> {path} (epoch{epoch})', flush=True)
    _checkpoint(saved, backbone, method, final=True)
    print(f'TRAIN_COMPLETE {backbone}__s42__{method}; TEST deferred', flush=True)


def _test_arrays(dataset):
    xs, ys, names = [], [], []
    for index in range(len(dataset)):
        item = dataset[index]
        xs.append(item['csi']); ys.append(item['pose'][0]); names.append(item['name'])
    return np.stack(xs), np.stack(ys), names


def _predict_grid(model, x, trigger, doses, device):
    import torch
    model.eval()
    if trigger is not None and isinstance(trigger, torch.nn.Module):
        trigger.eval()
    outputs = []
    with torch.no_grad():
        for dose in doses:
            rows = []
            for start in range(0, len(x), BATCH):
                clean = torch.from_numpy(x[start:start + BATCH]).to(device, dtype=torch.float32)
                if dose and trigger is not None:
                    if callable(getattr(trigger, 'inject_tensor', None)):
                        inp = trigger.inject_tensor(clean, dose, eps=.1)
                    else:
                        inp = torch.from_numpy(np.stack([
                            trigger.inject(value, dose) for value in x[start:start + BATCH]
                        ])).to(device, dtype=torch.float32)
                else:
                    inp = clean
                prediction, _ = model(inp)
                rows.append(prediction[:, 0].cpu().numpy())
            outputs.append(np.concatenate(rows))
            print(f'[PiW eval] dose={dose:g} n={len(x)}', flush=True)
    return np.stack(outputs, axis=1)


def _atomic_npz(path, **arrays):
    temp = path.with_name(path.name + '.saving')
    try:
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        base.atomic_replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def evaluate_cell(saved, backbone, method, device):
    from eval.metrics import pa_mpjpe
    cfg = saved['cfg']
    _require_all_final(saved)
    own = _checkpoint(saved, backbone, method, final=True)
    if own['execution'] != _execution(device):
        raise ValueError('PiW evaluation GPU/backend differs from checkpoint')
    ds = _datasets(saved)
    x, truth, names = _test_arrays(ds['validation'])
    if names != saved['inputs']['splits']['test_data']['ordered_names']:
        raise ValueError('PiW TEST order differs from preflight')
    model = _build(backbone, device)
    model.load_state_dict(own['model'], strict=True)
    folder = cell_dir(cfg, backbone, method)
    if method == 'clean':
        plain = _predict_grid(model, x, None, [0.], device)[:, 0]
        result = dict(profile=PROFILE, status='COMPLETE', backbone=backbone,
            method=method, seed=42, plan_sha256=saved['plan_sha256'],
            checkpoint_sha256=_sha(folder / 'checkpoint.pt'),
            clean_metrics=dict(clean_mpjpe_mm=float(np.linalg.norm(plain-truth, axis=-1).mean()*1000),
                               clean_pampjpe_mm=float(pa_mpjpe(plain, truth).mean()*1000)))
        _atomic_npz(folder / 'predictions.npz', truth=truth, names=np.asarray(names),
                    predictions=plain)
        result['predictions_sha256'] = _sha(folder / 'predictions.npz')
        _atomic_json(folder / 'evaluation.json', result)
        print(f'EVAL_COMPLETE {backbone}__s42__clean', flush=True)
        return
    clean_blob = _checkpoint(saved, backbone, 'clean', final=True)
    clean_model = _build(backbone, device)
    clean_model.load_state_dict(clean_blob['model'], strict=True)
    trigger = make_trigger(method, ds['training'], cfg)
    if method == 'tsba_adapted':
        trigger.load_state_dict(own['trigger'], strict=True)
        trigger = trigger.to(device)
    elif method in ('wanet_source', 'ftrojan', 'fiba', 'bpp_source'):
        trigger.load_state_dict(own['trigger'])
    victim_grid = _predict_grid(model, x, trigger, DOSES, device)
    reference_grid = _predict_grid(clean_model, x, trigger, DOSES, device)
    metrics = measure_predictions(truth, victim_grid[:, 0], victim_grid,
                                  reference_grid[:, 0], reference_grid)
    targets = np.stack([target_pose(truth, d) for d in DOSES], axis=1)
    _atomic_npz(folder / 'predictions.npz', truth=truth, names=np.asarray(names),
                doses=np.asarray(DOSES), predictions=victim_grid[:, 0],
                trigger_predictions=victim_grid,
                reference_predictions=reference_grid[:, 0],
                reference_trigger_predictions=reference_grid, targets=targets)
    result = dict(profile=PROFILE, status='COMPLETE', backbone=backbone,
        method=method, seed=42, plan_sha256=saved['plan_sha256'],
        checkpoint_sha256=_sha(folder / 'checkpoint.pt'),
        clean_checkpoint_sha256=_sha(cell_dir(cfg, backbone, 'clean') / 'checkpoint.pt'),
        predictions_sha256=_sha(folder / 'predictions.npz'),
        poison=own['poison'], metrics=metrics,
        limitations=['One seed; no uncertainty estimate.',
                     'PiW input/model and attack operators are dataset adaptations.',
                     'No image/classification or physical RF reproduction claim.',
                     'Historical PiW target/LR were viewed during development.'])
    _atomic_json(folder / 'evaluation.json', result)
    print(f'EVAL_COMPLETE {backbone}__s42__{method}: {metrics["mainmetrics"]}', flush=True)


def _barrier(device, phase):
    folder = os.environ.get('PIW_SMOKE_BARRIER_DIR')
    if not folder:
        return
    import torch
    name = os.environ['PIW_SMOKE_CELL']
    count = int(os.environ.get('PIW_SMOKE_COUNT', '4'))
    root = Path(folder) / phase
    root.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize(device)
    _atomic_json(root / f'{name}.json', dict(cell=name, device=device,
        peak_allocated=torch.cuda.max_memory_allocated(device),
        peak_reserved=torch.cuda.max_memory_reserved(device)))
    deadline = time.monotonic() + 240
    while len(list(root.glob('*.json'))) < count:
        if time.monotonic() > deadline:
            raise RuntimeError(f'PiW {count}-worker {phase} barrier timeout')
        time.sleep(.1)


def smoke_cell(saved, backbone, method, device):
    import torch
    _seed(device)
    model = _build(backbone, device).train()
    opt = _optimizer(model, backbone)
    train = _datasets(saved)['training']
    trigger = make_trigger(method, train, saved['cfg'])
    data, _ = make_training_data(saved, method, {'training': train}, trigger)
    # Use a physical B32 with the actual 10% poison mix. Include a cover for
    # methods that have one, so their source operator is exercised as well.
    indices = []
    if method != 'clean':
        indices.extend(int(r['index']) for r in saved['poison']['samples'][:4])
        if getattr(data, 'covers', None):
            indices.append(min(data.covers))
    used = set(indices)
    for index in range(len(data)):
        if len(indices) == BATCH:
            break
        if index not in used and (method == 'clean' or index not in data.selected):
            indices.append(index)
            used.add(index)
    if len(indices) != BATCH:
        raise ValueError('Could not construct the physical PiW B32 smoke batch')
    samples = [data[i] for i in indices]
    batch = {key: torch.as_tensor(np.stack([row[key] for row in samples]))
             for key in samples[0]}
    if method == 'tsba_adapted':
        trigger = trigger.to(device)
        trigger_opt = torch.optim.Adam(trigger.parameters(),
                                       lr=saved['cfg']['tsba']['generator_lr'])
    else:
        trigger_opt = None
    torch.cuda.reset_peak_memory_stats(device)
    # All four physical B32 tensors/models must be resident before backward.
    resident = batch['csi'].to(device, dtype=torch.float32)
    _barrier(device, 'ready')
    del resident
    loss, n, gen = _train_batch(model, opt, batch, device, backbone,
        trigger=trigger, trigger_opt=trigger_opt, cfg=saved['cfg'],
        epoch=saved['cfg']['tsba']['warmup_epochs'], last_micro=True,
        group_size=BATCH)
    if n != BATCH or not math.isfinite(loss):
        raise ValueError('PiW B32 smoke failed')
    _barrier(device, 'done')
    print('GPU_SMOKE_WORKER_PASS ' + json.dumps(dict(
        cell=f'{backbone}__s42__{method}', device=device, batch=n, loss=loss,
        generator_steps=len(gen),
        peak_cuda_allocated=torch.cuda.max_memory_allocated(device),
        peak_cuda_reserved=torch.cuda.max_memory_reserved(device))), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path(
        '/workspace/piw3d/datasets/Person-in-WiFi-3D_processed_v1'))
    parser.add_argument('--outdir', type=Path, default=Path(
        '/workspace/runs/piw3d_two_models_s42_v1'))
    parser.add_argument('--action-npy', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cell')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--check-input-stats', action='store_true')
    modes.add_argument('--smoke-only', action='store_true')
    modes.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args(argv)
    cfg = cfg_for(args.dataset_root, args.outdir, args.action_npy)
    if args.dry_run:
        print(json.dumps(dict(plan(cfg), cells=[f'{b}__s42__{m}'
            for b in BACKBONES for m in METHODS]), indent=2))
        return 0
    if args.preflight_only:
        # The full audit reports progress; keep stdout machine-readable for
        # the Vast controller's exact JSON receipt parser.
        with redirect_stdout(sys.stderr):
            result = preflight(cfg)
        print(json.dumps(result, indent=2))
        return 0
    if args.check_input_stats:
        saved = load_plan(cfg)
        print(json.dumps(dict(plan_sha256=saved['plan_sha256'],
            input_stat_fingerprint=input_stat_fingerprint(cfg['dataset_root'],
                                                          saved['inputs']))))
        return 0
    if not args.cell:
        parser.error('--cell required for smoke/train/evaluate')
    if not re.fullmatch(r'cuda:[0-9]+', args.device):
        parser.error('Explicit CUDA device required; no CPU fallback')
    backbone, method = cell_parts(args.cell)
    saved = load_plan(cfg)
    if args.smoke_only:
        smoke_cell(saved, backbone, method, args.device)
    else:
        gate = json.loads((Path(cfg['outdir']) / 'prelaunch_gate.json').read_text(
            encoding='utf-8'))
        if gate.get('preflight', {}).get('plan_sha256') != saved['plan_sha256']:
            raise ValueError('Exact PiW GPU smoke gate missing/foreign')
        if args.evaluate_only:
            evaluate_cell(saved, backbone, method, args.device)
        else:
            train_cell(saved, backbone, method, args.device)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f'[PiW two-model attack] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
