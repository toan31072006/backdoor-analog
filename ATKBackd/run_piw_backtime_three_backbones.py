"""Setup-only by default: PiW HPE-Li BackTime -> three independent victims.

New serial experiment: HPE-Li -> WiFlow -> MetaFi++, BackTime victims only.
Reuse audited historical Clean controls read-only; no MM-Fi job management.
Missing Clean delays only evaluation, never victim training. Start explicitly
audits, capacity-checks and runs the queue; Inspect remains read-only/default.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import uuid

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np
import run_piw_hpeli_clean as base
import piw_backtime_clean_reference as clean_ref
import piw_backtime_attacker_reference as attacker_ref

HERE = Path(__file__).resolve().parent
PROFILE = 'piw_backtime_hpeli_shared_three_backbones_train_first_s42_v4'
BACKBONES = ('hpeli', 'wiflow', 'metafiplusplus')
GRID = [0., .2, .4, .6, .8, 1.]
RECIPES = {
    'hpeli': dict(optimizer='adamw', lr=.001, weight_decay=.01, accumulation=1),
    'wiflow': dict(optimizer='adamw', lr=.0001, weight_decay=.0001, accumulation=2),
    # Corrected PiW MetaFi++ recipe, NOT the obsolete SGD MM-Fi carry-over.
    'metafiplusplus': dict(optimizer='adamw', lr=.01, weight_decay=.01, accumulation=1),
}


def config(data_home, outdir, dataset=None, device='cuda:0', clean_references=None, attacker_reference=None):
    from piw_v2_adapter import target_contract
    from attack.backtime_piw import configuration
    cfg = base.configuration(data_home, outdir, device, dataset)
    refs = clean_references or {}
    if set(refs) - set(BACKBONES):
        raise ValueError('Unknown Clean reference backbone')
    cfg.update(profile=PROFILE, backbones=list(BACKBONES), roles=['backtime'],
        training_queue=[dict(backbone=b, role='backtime') for b in BACKBONES],
        clean_references={b: str(Path(refs[b]).resolve()) if refs.get(b) else None for b in BACKBONES},
        attacker_reference=str(Path(attacker_reference).resolve()) if attacker_reference else None,
        drop_last=True, target=target_contract(), eval_doses=GRID, rho=.1,
        recipes=deepcopy(RECIPES), attacker=asdict(configuration()),
        execution_policy='one serial PiW worker: HPE-Li -> WiFlow -> MetaFi++; may coexist with the existing MM-Fi WiFlow job after capacity gate; no automatic retry/kill/fallback',
        attacker_policy=('reuse completed source-pinned PiW v1 attacker read-only; no refit or optimizer/RNG/victim import'
                         if attacker_reference else 'fresh HPE-Li TRAIN surrogate; shared frozen attacker and adaptive IDs/doses for three scratch victims'),
        control_policy='read-only audited historical Clean references; freeze unique compatible sources without TEST-score selection; missing/incompatible/ambiguous auto-discovery delays only evaluation; explicit invalid sources are errors; never train Clean; late references use separate immutable evaluation sidecars',
        interpretation='exploratory PiW14 digital-amplitude BackTime adaptation, seed42, not exact forecasting reproduction or complete paper table')
    return json.loads(json.dumps(cfg, allow_nan=False))


def sources():
    files = [p for p in HERE.rglob('*.py') if 'tests' not in p.relative_to(HERE).parts
             and '__pycache__' not in p.parts]
    files += [HERE.parent / 'remote_tools/Start-PiWBackTimeThreeBackbones.ps1']
    if any(p.is_symlink() for p in files):
        raise ValueError('Linked scientific source is not accepted')
    return {p.relative_to(HERE.parent).as_posix(): base.file_sha(p) for p in sorted(files)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(path, value)


def signed(value):
    result = deepcopy(value)
    result['plan_sha256'] = base.object_sha(value)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plan_path(cfg):
    return Path(cfg['outdir']) / 'plan.json'


def runtime():
    value = base.runtime_versions()
    value['packages']['torchvision'] = package_version('torchvision')
    return value


@contextmanager
def lock(cfg):
    root = Path(cfg['outdir'])
    root.mkdir(parents=True, exist_ok=True)
    path, token = root / 'runner.lock', uuid.uuid4().hex
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), token=token)))
    try:
        yield
    finally:
        if read_json(path).get('token') != token:
            raise RuntimeError('Owned lock changed; refusing removal')
        path.unlink()


def preflight(cfg):
    from piw_v2_adapter import target_pose
    if plan_path(cfg).exists():
        return load_plan(cfg, full_hash=True)
    root = Path(cfg['outdir'])
    if root.exists() and any(p.name != 'runner.lock' for p in root.iterdir()):
        raise ValueError('Nonempty unbound output; preserve it and choose a new directory')
    if any(root.rglob('*.pt')):
        raise ValueError('Unbound checkpoints found; choose a new output')
    found = clean_ref.discover(cfg)
    candidates = {b: [cfg['clean_references'][b]] if cfg['clean_references'][b] else found[b] for b in BACKBONES}
    inputs, datasets = base.audit_inputs(cfg)
    for dataset in datasets.values():
        for row in dataset.items:
            target_pose(dataset.load_pose(row['kpt']), 1.)
    attacker_reference = (attacker_ref.audit(cfg['attacker_reference'], cfg, inputs)
                          if cfg['attacker_reference'] else None)
    references, reference_status = dict.fromkeys(BACKBONES), {}
    for b in BACKBONES:
        valid, errors = [], []
        for folder in candidates[b]:
            try:
                reference = clean_ref.audit(folder, b, cfg, inputs)
                model = build_model(b, 'cpu')
                model.load_state_dict(clean_ref.load(reference)['model'], strict=True)
                del model
                valid.append(reference)
            except (ValueError, OSError, KeyError, RuntimeError) as exc:
                errors.append(f'{folder}: {exc}')
        if len(valid) != 1:
            message = (f'{b}: expected ONE matching historical Clean, found {len(valid)}. '
                       f'Valid: {[v["manifest"] for v in valid]}. Rejected: {errors}. '
                       'No Clean retraining.')
            if cfg['clean_references'][b]:
                raise ValueError(message)
            reference_status[b] = dict(status='EVALUATION_PENDING_CLEAN', detail=message)
            print(message + ' Victim training is NOT blocked.', flush=True)
        else:
            references[b] = valid[0]
            reference_status[b] = dict(status='READY', manifest=valid[0]['manifest'])
    record = signed(dict(profile=PROFILE, cfg=cfg, sources=sources(),
                         runtime=runtime(), inputs=inputs, clean_references=references,
                         clean_reference_status=reference_status,
                         attacker_reference=attacker_reference))
    write_json(plan_path(cfg), record)
    return record


def load_plan(cfg, full_hash=False):
    saved = read_json(plan_path(cfg))
    unsigned = {k: v for k, v in saved.items() if k != 'plan_sha256'}
    canonical = config(cfg['data_home'], cfg['outdir'], cfg['dataset_root'], cfg['device'],
                       cfg['clean_references'], cfg['attacker_reference'])
    if (saved.get('plan_sha256') != base.object_sha(unsigned) or cfg != canonical
            or saved.get('cfg') != cfg or saved.get('sources') != sources()
            or saved.get('runtime') != runtime()
            or saved.get('profile') != PROFILE
            or saved.get('inputs', {}).get('fingerprint') != base.object_sha(saved['inputs']['splits'])):
        raise ValueError('Plan/config/source/runtime/input binding changed')
    if set(saved.get('clean_references', {})) != set(BACKBONES):
        raise ValueError('Invalid historical Clean reference registry')
    for b, record in saved['clean_references'].items():
        if record is not None and record.get('backbone') != b:
            raise ValueError('Historical Clean reference backbone changed')
        # Clean files are an evaluation dependency, not a training dependency.
        # Their frozen hashes are rechecked before any evaluation uses them.
    if cfg['attacker_reference']:
        reference = saved.get('attacker_reference')
        if not reference or reference.get('source_run') != cfg['attacker_reference']:
            raise ValueError('Missing/foreign historical attacker audit; no refit fallback')
        attacker_ref.files_match(reference)
        attacker_ref.require_idle(reference['source_run'])
    elif saved.get('attacker_reference') is not None:
        raise ValueError('Unexpected historical attacker reference')
    for split, entry in saved['inputs']['splits'].items():
        path = Path(cfg['dataset_root']) / split / (split + '_list.txt')
        if base.file_sha(path) != entry['official_list_sha256']:
            raise ValueError('Official list changed')
        for row in entry['files']:
            path = Path(cfg['dataset_root']) / row['path']
            if not path.is_file() or path.stat().st_size != row['size']:
                raise ValueError('Audited input missing or resized')
            if full_hash and base.file_sha(path) != row['sha256']:
                raise ValueError('Audited input content changed')
    return saved


def datasets_for(plan):
    from data_utils.feeder import PersonInWiFi3D
    result = {}
    for mode, split in (('training', 'train_data'), ('validation', 'test_data')):
        data = PersonInWiFi3D(mode, plan['cfg']['dataset_root'], 'one-person', 1)
        if [r['name'] for r in data.items] != plan['inputs']['splits'][split]['ordered_names']:
            raise ValueError('Loader identities/order differ from audited split')
        result[mode] = data
    return result


def seed(device):
    import torch
    execution = base._setup_execution(device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.startswith('cuda:'):
        torch.cuda.manual_seed(42)
    return execution


def build_model(backbone, device):
    if backbone == 'hpeli':
        return base.build_model().to(device)
    if backbone not in BACKBONES:
        raise ValueError('Unknown PiW backbone')
    from models.piw3d_two_models import build_piw3d_model
    return build_piw3d_model(backbone).to(device)


def optimizer(model, backbone):
    import torch
    recipe = RECIPES[backbone]
    return torch.optim.AdamW(model.parameters(), lr=recipe['lr'],
        weight_decay=recipe['weight_decay'], betas=(.9, .999), eps=1e-8)


def prediction(model, x):
    import torch
    result = model(x)
    result = result[0] if isinstance(result, (tuple, list)) else result
    if (result.shape != (len(x), 1, 14, 3) or not torch.isfinite(result).all()):
        raise ValueError('PiW model output must be finite Bx1x14x3')
    return result


def finite_step(model, opt):
    import torch
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError('Missing/nonfinite gradients; no optimizer step')
    opt.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite weights; no retry')
    opt.zero_grad(set_to_none=True)


def batch_groups(n_batches, accumulation, index, physical_batch=32):
    start = (index // accumulation) * accumulation
    return min(accumulation, n_batches - start) * physical_batch


def smoke(plan):
    import torch
    from attack import backtime_piw as attack
    cfg = plan['cfg']
    if not cfg['device'].startswith('cuda:'):
        raise ValueError('Production smoke requires explicit CUDA, no CPU fallback')
    path = Path(cfg['outdir']) / 'capacity.json'
    if path.exists():
        raise ValueError('Capacity receipt exists; inspect it, do not silently repeat')
    execution = seed(cfg['device'])
    data = datasets_for(plan)['training']
    rows = [data[i] for i in range(64)]
    x = torch.as_tensor(np.stack([r['csi'] for r in rows]), device=cfg['device'])
    y = torch.as_tensor(np.stack([r['pose'] for r in rows]), device=cfg['device'])
    acfg = attack.configuration()
    features = attack.core.author_frame_fft_features(x[0, :, :90].cpu().numpy().reshape(270, 20))
    variables = np.sort(np.random.default_rng(42).choice(270, 81, replace=False))
    graph = attack.core._GraphGenerator(acfg, features[variables], variables,
                float(x[:, :, :90].mean()), float(x[:, :, :90].std())).to(cfg['device'])
    surrogate = build_model('hpeli', cfg['device']).train()
    opt = torch.optim.Adam(surrogate.parameters(), lr=acfg.surrogate_lr)
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.smooth_l1_loss(prediction(surrogate, x), y)
    loss.backward()
    finite_step(surrogate, opt)
    graph_opt = torch.optim.Adam(graph.parameters(), lr=acfg.attack_lr)
    with attack.core._frozen(surrogate, eval_mode=False):
        changed, penalty = attack.apply_tensor(graph, x[:1], 1.)
        target = attack.target_tensor(y[:1], torch.ones(1, device=x.device))
        loss = torch.nn.functional.mse_loss(prediction(surrogate, changed)[..., [11,13], :], target[..., [11,13], :])
        (loss + acfg.lam_norm * penalty.sum(-1).abs().mean()).backward()
        finite_step(graph, graph_opt)
    frozen = attack.FrozenPiWBackTimeTrigger(acfg, graph).to(cfg['device'])
    del opt, surrogate, graph_opt, graph
    results = []
    for backbone in BACKBONES:
        seed(cfg['device'])
        model = build_model(backbone, cfg['device']).train()
        opt = optimizer(model, backbone)
        opt.zero_grad(set_to_none=True)
        for role in ('clean', 'backtime'):
            for _ in range(RECIPES[backbone]['accumulation']):
                xb, yb = x[:32], y[:32]
                if role == 'backtime':
                    xb = frozen.apply_tensor(xb, 1.)
                    yb = attack.target_tensor(yb, torch.ones(32, device=x.device))
                    if not torch.equal(xb[:, :, 90:], x[:32, :, 90:]):
                        raise ValueError('Smoke changed phase')
                loss = torch.linalg.vector_norm(prediction(model, xb) - yb, dim=-1).mean()
                (loss / RECIPES[backbone]['accumulation']).backward()
            finite_step(model, opt)
            results.append(dict(backbone=backbone, role=role, physical_batch=32,
                                effective_batch=32*RECIPES[backbone]['accumulation'], loss=float(loss)))
        del model, opt
        torch.cuda.empty_cache()
    receipt = dict(profile=PROFILE, plan_sha256=plan['plan_sha256'], execution=execution,
                   surrogate_batch=64, generator_batch=1, victim_smokes=results,
                   checkpoints_saved=False, training_launched=False)
    write_json(path, {**receipt, 'receipt_sha256':base.object_sha(receipt)})
    return receipt


def capacity(plan):
    record = read_json(Path(plan['cfg']['outdir']) / 'capacity.json')
    unsigned = {k:v for k,v in record.items() if k != 'receipt_sha256'}
    # Receipt has its own digest; parent plan hash remains in the payload.
    if (record.get('receipt_sha256') != base.object_sha(unsigned)
            or record.get('plan_sha256') != plan['plan_sha256']
            or record.get('execution') != base._setup_execution(plan['cfg']['device'])
            or record.get('profile') != PROFILE or record.get('surrogate_batch') != 64
            or record.get('generator_batch') != 1 or record.get('checkpoints_saved') is not False
            or record.get('training_launched') is not False
            or [(r.get('backbone'), r.get('role'), r.get('physical_batch'), r.get('effective_batch'))
                for r in record.get('victim_smokes',[])] !=
                [(b,r,32,32*RECIPES[b]['accumulation']) for b in BACKBONES for r in ('clean','backtime')]
            or any(not math.isfinite(r.get('loss',float('nan'))) for r in record.get('victim_smokes',[]))):
        raise ValueError('Missing/foreign capacity gate')
    return record['execution']


def attacker_path(plan):
    if plan['cfg'].get('attacker_reference'):
        # The historical file is never copied, relabeled or rebound to v3.
        return Path(plan['cfg']['attacker_reference']) / 'attacker/trigger.pt'
    return Path(plan['cfg']['outdir']) / 'attacker/trigger.pt'


def fit_attacker(plan):
    import torch
    from attack import backtime_piw as attack
    cfg = plan['cfg']
    if cfg.get('attacker_reference'):
        print('REUSE_COMPLETED_ATTACKER: no attacker training; original artifacts read-only', flush=True)
        return load_attacker(plan)
    execution = capacity(plan)
    seed(cfg['device'])
    folder = attacker_path(plan).parent
    folder.mkdir(parents=True, exist_ok=True)
    if attacker_path(plan).exists():
        return load_attacker(plan)
    data = datasets_for(plan)['training']
    model = build_model('hpeli', cfg['device'])
    doses = np.random.default_rng(42).uniform(.2, 1., len(data)).astype(np.float32)
    path = folder / 'fit_checkpoint.pt'
    resume = None
    if path.exists():
        blob = torch.load(path, map_location='cpu', weights_only=False)
        if blob.get('plan_sha256') != plan['plan_sha256'] or blob.get('execution') != execution:
            raise ValueError('Foreign attacker resume binding')
        resume = blob['state']
    def save(state):
        base.atomic_checkpoint(path, dict(plan_sha256=plan['plan_sha256'], execution=execution, state=state))
    trigger = attack.fit_stream(model, data, attack.configuration(),
        dataset_sha256=plan['inputs']['fingerprint'], n_poison=math.floor(.1*len(data)),
        doses=doses, resume_state=resume, save_state=save,
        progress=lambda r: print(json.dumps(r), flush=True))
    blob = dict(plan_sha256=plan['plan_sha256'], execution=execution,
                doses=doses, trigger=trigger.state_dict())
    base.atomic_checkpoint(attacker_path(plan), blob)
    write_json(folder/'provenance.json', trigger.provenance())
    return load_attacker(plan)


def load_attacker(plan):
    import torch
    if plan['cfg'].get('attacker_reference'):
        reference = plan.get('attacker_reference')
        if not reference or reference.get('source_run') != plan['cfg']['attacker_reference']:
            raise ValueError('Missing/foreign historical attacker audit; no refit fallback')
        trigger, doses = attacker_ref.load(reference, plan['cfg'], plan['inputs'])
        return trigger.to(plan['cfg']['device']), doses
    blob = torch.load(attacker_path(plan), map_location='cpu', weights_only=False)
    if blob.get('plan_sha256') != plan['plan_sha256'] or blob.get('execution') != capacity(plan):
        raise ValueError('Frozen attacker plan/execution changed')
    trigger, doses = attacker_ref.validate(blob, plan['cfg'], plan['inputs'])
    return trigger.to(plan['cfg']['device']), doses


class Pairs:
    def __init__(self, data, indices, doses):
        self.base, self.selected, self.doses = data, set(indices), doses

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        from piw_v2_adapter import target_pose
        item = self.base[index]
        marked = index in self.selected
        dose = float(self.doses[index]) if marked else 0.
        y = np.array(item['pose'], copy=True)
        if marked:
            y = target_pose(y, dose).astype(np.float32)
        return dict(csi=item['csi'], pose=y, dose=dose, poisoned=marked)


def cell(plan, backbone, role):
    if backbone not in BACKBONES or role not in ('clean', 'backtime'):
        raise ValueError('Unknown victim cell')
    return Path(plan['cfg']['outdir']) / 'victims' / backbone / role


def reference_path(plan, backbone):
    if backbone not in BACKBONES:
        raise ValueError('Unknown Clean backbone')
    return Path(plan['cfg']['outdir']) / 'evaluation_references' / (backbone + '.json')


def evaluation_reference(plan, backbone):
    """Get a frozen reference, never rediscover or select by TEST results."""
    path = reference_path(plan, backbone)
    original = plan.get('clean_references', {}).get(backbone)
    if not path.exists():
        return original
    record = read_json(clean_ref.regular(path))
    unsigned = {k: v for k, v in record.items() if k != 'plan_sha256'}
    reference = record.get('reference', {})
    if (record.get('plan_sha256') != base.object_sha(unsigned)
            or record.get('profile') != PROFILE
            or record.get('training_plan_sha256') != plan['plan_sha256']
            or record.get('backbone') != backbone or reference.get('backbone') != backbone
            or original is not None):
        raise ValueError('Late Clean reference binding changed or shadows the training plan')
    return reference


def require_trained(plan):
    # Keep the original deferred-TEST contract: all three final199 first.
    for backbone in BACKBONES:
        checkpoint(plan, backbone, 'backtime', final=True)


def register_evaluation_reference(plan, backbone, folder):
    """Attach a compatible Clean AFTER training, without changing plan/victims."""
    require_trained(plan)
    current = evaluation_reference(plan, backbone)
    if current is not None:
        if Path(folder).resolve() != Path(current['manifest']).parent.resolve():
            raise ValueError('Clean reference is already frozen; cannot replace it')
        clean_ref.files_match(current)
        return current
    reference = clean_ref.audit(folder, backbone, plan['cfg'], plan['inputs'])
    model = build_model(backbone, 'cpu')
    model.load_state_dict(clean_ref.load(reference)['model'], strict=True)
    del model
    path = reference_path(plan, backbone)
    if (cell(plan, backbone, 'backtime') / 'evaluation.json').exists():
        raise ValueError('Cannot attach a different Clean to an existing evaluation')
    write_json(path, signed(dict(profile=PROFILE, training_plan_sha256=plan['plan_sha256'],
                                backbone=backbone, reference=reference)))
    return evaluation_reference(plan, backbone)


def checkpoint(plan, backbone, role, final=False):
    import torch
    from attack.backtime_mmfi import _sha
    if role == 'clean':
        if backbone not in BACKBONES:
            raise ValueError('Unknown Clean backbone')
        reference = evaluation_reference(plan, backbone)
        if reference is None:
            raise ValueError(f'{backbone}: EVALUATION_PENDING_CLEAN; never train a Clean fallback')
        return clean_ref.load(reference)
    blob = torch.load(cell(plan, backbone, role)/'checkpoint.pt', map_location='cpu', weights_only=False)
    value = dict(blob)
    claimed = value.pop('sha256', None)
    if (claimed != _sha(value) or blob.get('plan_sha256') != plan['plan_sha256']
            or blob.get('backbone') != backbone or blob.get('role') != role
            or blob.get('execution') != capacity(plan)
            or blob.get('attacker_sha256') != base.file_sha(attacker_path(plan))
            or type(blob.get('epoch')) is not int or not 0 <= blob['epoch'] < 200
            or any(k not in blob for k in ('model','optimizer','rng'))
            or (final and blob['epoch'] != 199)):
        raise ValueError('Victim checkpoint binding/checksum/final epoch mismatch')
    return blob


def train_epoch(model, opt, loader, backbone, device, trigger=None):
    import torch
    model.train()
    opt.zero_grad(set_to_none=True)
    total, count = 0., 0
    accumulation = RECIPES[backbone]['accumulation']
    for index, batch in enumerate(loader):
        x = batch['csi'].to(device, dtype=torch.float32)
        y = batch['pose'].to(device, dtype=torch.float32)
        if trigger is not None:
            with torch.no_grad():
                x = trigger.apply_tensor(x, batch['dose'].to(device))
        loss = torch.linalg.vector_norm(prediction(model, x) - y, dim=-1).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite victim MPJPE')
        (loss * len(x) / batch_groups(len(loader), accumulation, index)).backward()
        if (index+1) % accumulation == 0 or index+1 == len(loader):
            finite_step(model, opt)
        total += float(loss.detach())*len(x)
        count += len(x)
    if count != len(loader)*32 or not count:
        raise ValueError('Wrong physical-batch TRAIN coverage')
    return total/count


def train(plan, backbone, role):
    if backbone not in BACKBONES or role != 'backtime':
        raise ValueError('Only BackTime victims may train; Clean is a read-only historical reference')
    from torch.utils.data import DataLoader
    from attack.backtime_mmfi import _sha
    cfg = plan['cfg']
    execution = capacity(plan)
    folder = cell(plan, backbone, role)
    path, start, resume = folder/'checkpoint.pt', 0, None
    if path.exists():
        resume = checkpoint(plan, backbone, role)
        if resume['epoch'] == 199:
            print(f'SKIP_TRAINED: {backbone}/backtime final199 verified; no retraining', flush=True)
            return
    trigger, doses = load_attacker(plan)
    seed(cfg['device'])
    model = build_model(backbone, cfg['device'])
    opt = optimizer(model, backbone)
    data = datasets_for(plan)['training']
    data = Pairs(data, trigger.training_indices, doses)
    folder.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        model.load_state_dict(resume['model'], strict=True)
        opt.load_state_dict(resume['optimizer'])
        base._restore_rng(resume['rng'], cfg['device'])
        start = resume['epoch']+1
        print(f'RESUME: {backbone}/backtime next epoch={start}/199', flush=True)
    loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    attacker_hash = base.file_sha(attacker_path(plan))
    for epoch in range(start, 200):
        loss = train_epoch(model, opt, loader, backbone, cfg['device'], trigger if role=='backtime' else None)
        blob = dict(plan_sha256=plan['plan_sha256'], backbone=backbone, role=role,
                    epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                    rng=base._rng_state(cfg['device']), execution=execution,
                    attacker_sha256=attacker_hash, train_loss=loss)
        base.atomic_checkpoint(path, {**blob,'sha256':_sha(blob)})
        print(f'{backbone}/{role} epoch={epoch}/199 MPJPE={loss:.8f}', flush=True)


def infer_grid(model, dataset, trigger, device):
    import torch
    from torch.utils.data import DataLoader
    model.eval()
    grid = []
    with torch.no_grad():
        for dose in GRID:
            result = []
            for batch in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
                x = batch['csi'].to(device, dtype=torch.float32)
                if dose:
                    x = trigger.apply_tensor(x, dose)
                result.append(prediction(model, x)[:,0].cpu().numpy())
            grid.append(np.concatenate(result))
    return np.stack(grid, axis=1)


def evaluate(plan, backbone):
    import torch
    from piw_v2_adapter import measure_predictions, target_pose
    from run_piw_hpeli_lr_control import atomic_npz
    require_trained(plan)
    checkpoint(plan, backbone, 'clean', final=True)
    reference = evaluation_reference(plan, backbone)
    folder = cell(plan,backbone,'backtime')
    if (folder/'evaluation.json').exists():
        return verify_evaluation(plan,backbone)
    cfg = plan['cfg']
    trigger,_ = load_attacker(plan)
    data = datasets_for(plan)['validation']
    truth = np.stack([data.load_pose(r['kpt'])[0] for r in data.items])
    names = np.asarray([r['name'] for r in data.items])
    grids = {}
    for role in ('clean','backtime'):
        blob = checkpoint(plan,backbone,role,final=True)
        model = build_model(backbone,cfg['device'])
        model.load_state_dict(blob['model'],strict=True)
        from attack.backtime_mmfi import _sha
        before = _sha(model.state_dict())
        grids[role] = infer_grid(model,data,trigger,cfg['device'])
        if _sha(model.state_dict()) != before:
            raise ValueError('Evaluation changed model/BN state')
        del model
        if cfg['device'].startswith('cuda:'):
            torch.cuda.empty_cache()
    reference_audit = clean_ref.replay(reference, truth, names.tolist(), grids['clean'][:,0])
    metrics = measure_predictions(truth, grids['backtime'][:,0], grids['backtime'], grids['clean'][:,0], grids['clean'])
    targets = np.stack([target_pose(truth,d) for d in GRID],axis=1)
    residual = (grids['backtime'][:,:, [11,13]]-grids['backtime'][:,0:1,[11,13]])-(targets[:,:,[11,13]]-truth[:,None,[11,13]])
    atomic_npz(folder/'predictions.npz',truth=truth,names=names,doses=np.asarray(GRID),
        predictions=grids['backtime'],reference_predictions=grids['clean'],targets=targets,
        displacement_residuals=residual)
    result = dict(profile=PROFILE, status='COMPLETE', plan_sha256=plan['plan_sha256'],backbone=backbone,
                  metrics=metrics, clean_reference=reference_audit,
                  clean_reference_binding_sha256=base.object_sha(reference),
                  attacker_reference=plan.get('attacker_reference'), artifacts={
                      'predictions.npz':base.file_sha(folder/'predictions.npz'),
                      'checkpoint.pt':base.file_sha(folder/'checkpoint.pt'),
                      'clean_checkpoint.pt':base.file_sha(reference['checkpoint']),
                      'trigger.pt':base.file_sha(attacker_path(plan))})
    write_json(folder/'evaluation.json',result)
    return verify_evaluation(plan,backbone)


def verify_evaluation(plan, backbone):
    from piw_v2_adapter import measure_predictions, target_pose
    from run_piw_hpeli_lr_control import replay_equal
    require_trained(plan)
    reference = evaluation_reference(plan, backbone)
    if reference is None:
        raise ValueError(f'{backbone}: EVALUATION_PENDING_CLEAN')
    clean_ref.files_match(reference)
    folder = cell(plan,backbone,'backtime')
    result = read_json(folder/'evaluation.json')
    paths = {'predictions.npz':folder/'predictions.npz', 'checkpoint.pt':folder/'checkpoint.pt',
             'clean_checkpoint.pt':Path(reference['checkpoint']),'trigger.pt':attacker_path(plan)}
    if (result.get('profile') != PROFILE or result.get('status') != 'COMPLETE'
            or result.get('backbone') != backbone or result.get('plan_sha256') != plan['plan_sha256']
            or result.get('clean_reference_binding_sha256') != base.object_sha(reference)
            or result.get('artifacts') != {k:base.file_sha(p) for k,p in paths.items()}):
        raise ValueError('Evaluation binding/artifact changed')
    with np.load(folder/'predictions.npz',allow_pickle=False) as z:
        if z['names'].tolist() != plan['inputs']['splits']['test_data']['ordered_names'] or not np.array_equal(z['doses'],GRID):
            raise ValueError('Evaluation identities/doses changed')
        data = datasets_for(plan)['validation']
        truth = np.stack([data.load_pose(r['kpt'])[0] for r in data.items])
        if not np.array_equal(z['truth'],truth):
            raise ValueError('Saved GT differs from official dataset')
        targets = np.stack([target_pose(truth,d) for d in GRID],axis=1)
        if not np.array_equal(targets,z['targets']):
            raise ValueError('Saved targets changed')
        residual = (z['predictions'][:,:,[11,13]]-z['predictions'][:,0:1,[11,13]])-(targets[:,:,[11,13]]-truth[:,None,[11,13]])
        if not np.array_equal(residual,z['displacement_residuals']):
            raise ValueError('Saved displacement residuals changed')
        metrics = measure_predictions(truth,z['predictions'][:,0],z['predictions'],z['reference_predictions'][:,0],z['reference_predictions'])
        reference_audit = clean_ref.replay(reference, truth, z['names'].tolist(), z['reference_predictions'][:,0])
    replay_equal(result['metrics'],metrics)
    replay_equal(result['clean_reference'], reference_audit)
    replay_equal(result.get('attacker_reference'), plan.get('attacker_reference'))
    return result


def evaluate_available(plan, references=None, verify_only=False):
    require_trained(plan)
    for backbone, folder in (references or {}).items():
        register_evaluation_reference(plan, backbone, folder)
    complete, pending = [], []
    for backbone in BACKBONES:
        if evaluation_reference(plan, backbone) is None:
            pending.append(backbone)
            print(f'EVALUATION_PENDING_CLEAN: {backbone}; victim saved, no retraining', flush=True)
            continue
        result = verify_evaluation(plan, backbone) if verify_only else evaluate(plan, backbone)
        complete.append(backbone)
        print(json.dumps({backbone:result['metrics']['mainmetrics']}), flush=True)
    summary = dict(training_complete=True, evaluated=complete, evaluation_pending_clean=pending)
    print(json.dumps(summary), flush=True)
    print('ALL_THREE_COMPLETE' if not pending else 'ALL_THREE_TRAINED_EVALUATION_PARTIAL', flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['inspect','discover','check','prepare','start','launch','evaluate','status','verify'],default='inspect')
    parser.add_argument('--data-home',required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--outdir',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--attacker-reference', metavar='RUN_DIR',
                        help='Completed PiW v1 attacker run, read-only; no attacker refit or victim-state migration')
    parser.add_argument('--clean-reference',action='append',default=[],metavar='BACKBONE=RUN_DIR',
                        help='Optional explicit historical Clean run directory; otherwise bounded metadata discovery, no Clean retraining')
    args = parser.parse_args(argv)
    references = {}
    for value in args.clean_reference:
        b, sep, path = value.partition('=')
        if not sep or not path or b not in BACKBONES or b in references:
            parser.error('Use one --clean-reference BACKBONE=RUN_DIR per known backbone')
        references[b] = path
    cfg = config(args.data_home,args.outdir,args.dataset,args.device,references,args.attacker_reference)
    if args.mode == 'inspect':
        print(json.dumps(cfg,indent=2))
        return
    if args.mode == 'discover':
        print(json.dumps(dict(candidates=clean_ref.discover(cfg),
            note='Metadata candidates only, not validated references. CheckOnly audits recipes/inputs/weights. No training or writes.'),indent=2))
        return
    if args.mode == 'status':
        root = Path(cfg['outdir'])
        saved = read_json(plan_path(cfg)) if plan_path(cfg).is_file() else None
        print(json.dumps(dict(plan=plan_path(cfg).is_file(), capacity=(root/'capacity.json').is_file(),
            attacker=attacker_path(dict(cfg=cfg)).is_file(),
            clean_available={b:bool(saved and evaluation_reference(saved,b)) for b in BACKBONES},
            checkpoints={b:(root/'victims'/b/'backtime/checkpoint.pt').is_file() for b in BACKBONES},
            evaluations={b:(root/'victims'/b/'backtime/evaluation.json').is_file() for b in BACKBONES}),indent=2))
        return
    with lock(cfg):
        if args.mode in ('evaluate','verify'):
            # New references are evaluation-only inputs, not modifications of
            # the signed training configuration or checkpoint binding.
            original = read_json(plan_path(cfg))['cfg']['clean_references']
            eval_cfg = config(args.data_home,args.outdir,args.dataset,args.device,original,args.attacker_reference)
            plan = load_plan(eval_cfg,full_hash=True)
            evaluate_available(plan,references if args.mode=='evaluate' else None,
                               verify_only=args.mode=='verify')
            return
        if args.mode in ('check','prepare'):
            plan = preflight(cfg)
            if args.mode == 'prepare':
                smoke(plan)
            print('CHECK_PASS' if args.mode=='check' else 'PREPARED_NOT_LAUNCHED',flush=True)
            return
        plan = preflight(cfg) if args.mode=='start' else load_plan(cfg,full_hash=True)
        if args.mode=='start' and not (Path(cfg['outdir'])/'capacity.json').exists():
            smoke(plan)
        capacity(plan)
        trigger,_ = fit_attacker(plan)
        del trigger
        for queued in cfg['training_queue']:
            train(plan,queued['backbone'],queued['role'])
        print('ALL_THREE_TRAINED: hpeli -> wiflow -> metafiplusplus, final199',flush=True)
        evaluate_available(plan)


if __name__ == '__main__':
    main()
