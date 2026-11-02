"""Exploratory, serial HPE-Li/MM-Fi P3S3 SIBA and LADDER pose adaptations.

An independently initialized Clean surrogate is trained on full TRAIN first.
Only full TRAIN informs the source-aligned trigger fitting. Each victim starts
afresh; final49, six-dose TEST, no TEST selection, no implicit job retry.
Prepare performs full input audit and native CUDA B32 smoke, NOT training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid
from importlib.metadata import version

os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  MKL_THREADING_LAYER='SEQUENTIAL', PYTHONUTF8='1')

import run_mmfi_p3_s3_three_attacks_4090 as common

PROFILE = 'mmfi_p3s3_siba_ladder_source_adapted_s42_4090_v2'
METHODS = ('siba', 'ladder')
CELLS = ('clean_surrogate',) + METHODS
BACKTIME_POLICIES = ('adaptive',)
COUNTS = {'training': 240570, 'test': 80190}
GRID = (0., .2, .4, .6, .8, 1.)
FIT_COUNT, SCORE_COUNT = 1024, 256
ACTION_SHA = common.ACTION_SHA
HERE = Path(__file__).resolve().parent
sha, digest, read = common.sha, common.digest, common.read


def reject_links(path):
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        if item.is_symlink() or getattr(item, 'is_junction', lambda: False)():
            raise ValueError(f'Linked path/ancestor is not supported: {item}')


def write(path, value, *, immutable=True):
    reject_links(path)
    common.write(path, value, immutable=immutable)


def save_checkpoint(path, value):
    reject_links(path)
    reject_links(str(path) + '.saving')
    common.atomic_checkpoint(path, value)


def configuration(home, method):
    import run_mmfi_p3_splits as p3
    if method not in CELLS + ('backtime',):
        raise ValueError('Only Clean reference, SIBA, BackTime and LADDER are in scope')
    cfg = p3._configuration(Path(home), 'cuda:0', 's3',
                            'clean' if method == 'clean_surrogate' else 'a1')
    cfg = {k: v for k, v in cfg.items()
           if k != 'confirmation_profile' and not k.startswith('method_')}
    cfg.update(trigger=method, learned_baseline_profile=PROFILE,
               attacker_access=('train_only_scratch_warmup_and_alternating_surrogate'
                   if method == 'backtime' else 'train_only_clean_surrogate_for_trigger_search'), ckpt_every=1)
    required = dict(model='hpeli', seed=42, mmfi_protocol='protocol3', mmfi_setting='s3',
        mmfi_split_seed=0, mmfi_random_ratio=.8, optimizer='sgd', lr=.001,
        momentum=.9, weight_decay=0., batch_size=32, epochs=50, victim_epochs=50,
        num_workers=0, pretrained=False, data_parallel=False, victim_loss='mpjpe',
        pivot=1, target_joints=[2, 3], theta_max_deg=40., payload_axis=[0., 0., 1.],
        dose_min=.2, dose_max=1., dose_coupling='paired', poison_select='uniform')
    if any(cfg.get(k) != v for k, v in required.items()) or list(cfg['dose_grid']) != list(GRID):
        raise ValueError('Common historical P3S3 HPE-Li victim/target recipe changed')
    if cfg['rho'] != (0. if method == 'clean_surrogate' else .1):
        raise ValueError('Unexpected poison rate')
    return cfg


def operator_configs(include_backtime=False):
    from attack.siba_ladder_mmfi import SIBAConfig, LADDERConfig
    configs = {'siba': asdict(SIBAConfig()), 'ladder': asdict(LADDERConfig())}
    if include_backtime:
        from attack.backtime_mmfi import BackTimeConfig
        configs['backtime'] = asdict(BackTimeConfig())
    return json.loads(json.dumps(configs))


def backtime_profile(policy):
    if policy not in BACKTIME_POLICIES:
        raise ValueError('Source-aligned BackTime requires adaptive recipient selection; uniform is not approved')
    return f'mmfi_p3s3_siba_ladder_backtime_{policy}_source_adapted_s42_4090_v2'


def selected_queue(backtime_policy=None, selected_methods=None):
    methods = tuple(selected_methods) if selected_methods is not None else (
        ('siba', 'backtime', 'ladder') if backtime_policy else METHODS)
    if methods not in (METHODS, ('siba', 'backtime'), ('siba', 'backtime', 'ladder')):
        raise ValueError('Use the declared ordered methods: siba ladder, siba backtime, or siba backtime ladder')
    if ('backtime' in methods) != (backtime_policy is not None):
        raise ValueError('BackTime must be explicitly selected with its adaptive policy')
    if backtime_policy is not None:
        backtime_profile(backtime_policy)
    return methods


def output_profile(backtime_policy=None, selected_methods=None):
    methods = selected_queue(backtime_policy, selected_methods)
    value = backtime_profile(backtime_policy) if backtime_policy else PROFILE
    return value.replace('_s42_', '_siba_backtime_only_s42_') if methods == ('siba', 'backtime') else value


def require_source_ready(plan):
    # Do not train Clean/SIBA first and only then discover the requested LADDER
    # search is absent. No experimental MOEA or FTrojan fallback is permitted.
    if 'ladder' in plan['methods']:
        from attack.siba_ladder_mmfi import require_ladder_author_search
        require_ladder_author_search()


def build_plan(home, outdir, device='cuda:0', jobs=1, backtime_policy=None, selected_methods=None):
    for path in (home, outdir):
        reject_links(path)
    home, outdir = Path(home).resolve(), Path(outdir).resolve()
    if (outdir.parent != home / 'runs' or
            not outdir.name.startswith('mmfi_p3s3_siba_ladder_')):
        raise ValueError('Use a NEW mmfi_p3s3_siba_ladder_* directory directly under data-home/runs')
    if device != 'cuda:0' or type(jobs) is not int or jobs != 1:
        raise ValueError('Exactly one serial worker on cuda:0 is required')
    profile = output_profile(backtime_policy, selected_methods)
    methods = selected_queue(backtime_policy, selected_methods)
    cells = ('clean_surrogate',) + methods
    configs = {m: configuration(home, m) for m in cells}
    for cfg in configs.values():
        cfg['learned_baseline_profile'] = profile
    if backtime_policy is not None:
        configs['backtime']['backtime_recipient_policy'] = backtime_policy
        if backtime_policy == 'adaptive':
            configs['backtime']['poison_select'] = 'backtime_train_error'
    for key in ('dataset_root', 'action_npy'):
        reject_links(configs['siba'][key])
    plan = dict(profile=profile, schema=1, home=str(home), outdir=str(outdir),
        device=device, jobs_per_gpu=1, methods=list(methods), cells=list(cells),
        backtime_recipient_policy=backtime_policy,
        cfg=configs, operators=operator_configs(backtime_policy is not None), sources=common.sources(),
        runtime=dict(python=platform.python_version(), executable=sys.executable,
            system=platform.system(), packages={n: version(n) for n in ('torch', 'numpy', 'scipy', 'PyYAML')},
            threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MKL_THREADING_LAYER')}),
        source_readiness=dict(ladder='BLOCKED_MISSING_AUTHOR_SEARCH_AND_TRIGGER',
            blocked_selected_methods=['ladder'] if 'ladder' in methods else [],
            excluded_requested_methods=['ladder'] if 'ladder' not in methods else [],
            no_automatic_substitution=True),
        contract=dict(protocol='protocol3', setting='s3', seed=42, counts=COUNTS,
            scratch_surrogate=True, scratch_victims=True, warm_start=False,
            epochs=50, final_epoch=49, batch=32, drop_last=False,
            optimizer='sgd', lr=.001, momentum=.9, weight_decay=0., amp=False,
            rho=.1, n_poison=24057, doses=list(GRID), target_joints=[2, 3],
            pck_scale=[5, 12], pck_threshold=.2, fit_train_frames=COUNTS['training'],
            score_train_frames=0, test_used_for_search=False,
            selection='fixed final49; no TEST tuning',
            status='EXPLORATORY_T6_EXTENSION_SINGLE_SEED',
            limitations=[
                'Independent CSI/pose/dose adaptations, not original image-classification reproductions.',
                'Clean trained from scratch on full TRAIN, read-only during trigger synthesis.',
                'SIBA uses full TRAIN/200 epochs/B128; old diagnostic banks are not used for production fitting.',
                'LADDER author repository lacks search code and selected trigger; production launch is blocked, not reconstructed.',
                'Each victim is independently initialized; no surrogate model/optimizer/BN warm-start.',
                'Native amplitude/coefficient constraints are not equal realized noise with A1.',
                'Single seed; no across-seed SD or claim of achieved attack efficacy.']))
    if backtime_policy is not None:
        plan['contract']['limitations'] += [
            'BackTime is an independent input-dependent GNN CSI/pose adaptation, not a forecasting reproduction.',
            'CSI packet-history6/trigger4 replaces forecasting history/future; pose horizon1 has no native overlapping-window soft identification.',
            'BackTime is last-history-value-anchored replacement with dose interpolation, not eps-bounded residual to the original input.',
            'Independent MM-Fi frames do not retain native continuous-series timestamp non-overlap rules.',
            'BackTime uses online SmoothL1 errors from its own scratch-surrogate clean warmup, not final49 Clean MPJPE.',
            'BackTime uses 100 full TRAIN passes/B64; epochs0..10 clean then89 alternating epochs; generator updates one recipient at a time.',
            'Packet-window FFT features are pooled over TRAIN without stitching independent frames into a fictitious continuous series.',
            'Paired-dose training/rho.1 are the common pose benchmark extension, not native forecasting hyperparameters.',
        ]
    plan['plan_sha256'] = digest(plan)
    return plan


def load_plan(plan):
    if read(Path(plan['outdir']) / 'plan.json') != plan:
        raise ValueError('Source/config/runtime/output changed; preserve old run')


def dataset(plan, split):
    import run_mmfi_p3_splits as p3
    return p3._dataset(plan['cfg']['siba'], split)


def bank_selection(n):
    import numpy as np
    if n < FIT_COUNT + SCORE_COUNT:
        raise ValueError('Insufficient TRAIN rows for disjoint optimization banks')
    rng = np.random.default_rng(np.random.SeedSequence([42, 1102]))
    selected = rng.choice(n, FIT_COUNT + SCORE_COUNT, replace=False)
    return dict(fit=sorted(map(int, selected[:FIT_COUNT])),
                score=sorted(map(int, selected[FIT_COUNT:])))


def audit(plan):
    import numpy as np
    import run_mmfi_p3_splits as p3
    root = Path(plan['outdir'])
    if (root / 'plan.json').exists():
        load_plan(plan)
        if (root / 'inputs.json').exists():
            return verify_inputs(plan, full_hash=True)
    elif any(p.name not in ('.controller', 'logs') for p in root.iterdir()):
        raise ValueError('Nonempty unbound output; use a new directory, do not overwrite')
    cfg = plan['cfg']['siba']
    action = Path(cfg['action_npy'])
    if sha(action) != ACTION_SHA or not np.isfinite(np.load(action, allow_pickle=False)).all():
        raise ValueError('Authentic finite MM-Fi motion prior required')
    data = {s: dataset(plan, s) for s in COUNTS}
    if any(len(data[s]) != count for s, count in COUNTS.items()):
        raise ValueError('Full P3S3 TRAIN/TEST counts required')
    splits = {s: p3._identity_digest(ds) for s, ds in data.items()}
    a, b = splits['training']['sequence_keys'], splits['test']['sequence_keys']
    if (len(a) != 810 or len(b) != 270 or a & b or
            {x[1] for x in a} & {x[1] for x in b}):
        raise ValueError('P3S3 sequence/subject separation differs')
    for row in splits.values():
        row.pop('sequence_keys')
    data_root = Path(cfg['dataset_root']).resolve()
    selected = {}
    for ds in data.values():
        for row in ds.items:
            for key in ('csi', 'kpt'):
                reject_links(row[key])
                path = Path(row[key]).resolve()
                if not path.is_relative_to(data_root):
                    raise ValueError('Input escapes dataset root')
                selected[path.relative_to(data_root).as_posix()] = (path, key)
    records = []
    for i, (name, (path, kind)) in enumerate(sorted(selected.items()), 1):
        before = path.stat()
        value = np.load(path, allow_pickle=False, mmap_mode='r')
        if (not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value)
                or not np.isfinite(value).all()
                or (kind == 'csi' and (value.shape != (3, 114, 10)
                    or value.min() < -1e-6 or value.max() > 1 + 1e-6))
                or (kind == 'kpt' and (value.ndim != 3 or value.shape[1:] != (17, 3)))):
            raise ValueError(f'Invalid full input: {path}')
        del value
        records.append(dict(path=name, bytes=before.st_size, sha256=sha(path)))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Input changed during audit')
        if i % 10000 == 0:
            print(f'[preflight] {i}/{len(selected)} unique files validated/hashed', flush=True)
    pairs = common.paired_data(data['training'], common.Deferred(), cfg)
    poison = pairs.manifest()
    if poison['n_poison'] != 24057:
        raise ValueError('Expected 24057 paired poison recipients')
    selection = bank_selection(len(data['training']))
    rng = np.random.default_rng(np.random.SeedSequence([42, 1103]))
    banks = {}
    for key, indices in selection.items():
        doses = rng.uniform(.2, 1., len(indices))
        banks[key] = dict(indices=indices, doses=doses.tolist(),
            names=[data['training'].items[i]['name'] for i in indices], split='training')
    receipt = dict(plan_sha256=plan['plan_sha256'], dataset_root=str(data_root),
        splits=splits, files=records, full_content_sha256=digest(records),
        input_stat_fingerprint=common.input_stats(data_root, records),
        poison_sha256=digest(poison), banks=banks, banks_sha256=digest(banks),
        action_sha256=ACTION_SHA, no_test_selection=True)
    write(root / 'plan.json', plan)
    write(root / 'poison.json', poison)
    write(root / 'inputs.json', receipt)
    print('FULL_INPUT_AUDIT_PASS; production fitting uses full TRAIN; saved bank IDs are audit-only; no training', flush=True)
    return receipt


def verify_inputs(plan, *, full_hash=False):
    load_plan(plan)
    if common.sources() != plan['sources']:
        raise ValueError('Scientific source changed since plan creation')
    root = Path(plan['outdir'])
    row = read(root / 'inputs.json')
    if (row['plan_sha256'] != plan['plan_sha256'] or row['action_sha256'] != ACTION_SHA
            or sha(plan['cfg']['siba']['action_npy']) != ACTION_SHA
            or row['full_content_sha256'] != digest(row['files'])
            or row['poison_sha256'] != digest(read(root / 'poison.json'))
            or row['banks_sha256'] != digest(row['banks'])
            or row.get('no_test_selection') is not True
            or common.input_stats(Path(row['dataset_root']), row['files']) != row['input_stat_fingerprint']):
        raise ValueError('Audited input/poison/bank/prior binding changed')
    selection = bank_selection(COUNTS['training'])
    for key, indices in selection.items():
        if row['banks'][key]['split'] != 'training' or row['banks'][key]['indices'] != indices:
            raise ValueError('Optimization bank is not the registered TRAIN partition')
    if full_hash:
        for item in row['files']:
            path = Path(row['dataset_root']) / item['path']
            reject_links(path)
            if sha(path) != item['sha256']:
                raise ValueError(f'Audited input content changed: {path}')
    return row


def execution():
    from run_mmfi_badnets_two_models_4090 import gpu_signature
    return dict(torch=common.gpu(), physical_gpu0=gpu_signature())


def require_idle():
    from run_mmfi_badnets_two_models_4090 import require_idle as exclusive
    exclusive()


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def log_to(root, phase):
    import traceback
    folder = Path(root) / 'logs'
    reject_links(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = folder / f'{phase}.{stamp}.{uuid.uuid4().hex}.log'
    print(f'LOG={path}', flush=True)
    with path.open('x', encoding='utf-8') as stream:
        with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
            print(f'BEGIN {phase} pid={os.getpid()} utc={stamp}', flush=True)
            try:
                yield
            except BaseException:
                traceback.print_exc()
                raise
            print(f'END {phase} exit=0', flush=True)


def new_components(plan, cell):
    from models.factory import build_model
    from train_backdoor import _build_optimizer
    common.seed()  # Discard all previous model/search RNG before a fresh victim.
    model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False).to('cuda:0').train()
    name, optimizer = _build_optimizer(model, plan['cfg'][cell], 'mmfi')
    if name != 'sgd':
        raise ValueError('Expected native HPE-Li SGD')
    return model, optimizer


def preparation_binding(plan, env):
    root = Path(plan['outdir'])
    return dict(plan_sha256=plan['plan_sha256'], execution=env,
                inputs_sha256=sha(root / 'inputs.json'), poison_sha256=sha(root / 'poison.json'))


def prepare(plan):
    require_source_ready(plan)
    import torch
    from attack.poison import collate
    from train_backdoor import _victim_update
    audit(plan)
    require_idle()
    env, root = execution(), Path(plan['outdir'])
    if (root / 'gpu_gate.json').exists():
        gate_check(plan)
        print('PREPARED_ALREADY; no repeated smoke or full training', flush=True)
        return
    model, optimizer = new_components(plan, 'clean_surrogate')
    data = dataset(plan, 'training')
    batch = collate([data[i] for i in range(32)])
    if tuple(batch['csi'].shape) != (32, 3, 114, 10):
        raise ValueError('Real physical B32 smoke required')
    before = next(model.parameters()).detach().clone()
    losses = [_victim_update(model, optimizer, batch['csi'].to('cuda:0'),
                            batch['pose'].to('cuda:0')) for _ in range(2)]
    if not all(math.isfinite(x) for x in losses) or torch.equal(before, next(model.parameters())):
        raise ValueError('Native CUDA smoke did not update finite parameters')
    write(root / 'gpu_gate.json', dict(binding=preparation_binding(plan, env),
        batch_size=32, steps=2, real_train_input=True, losses=losses,
        status='NATIVE_CLEAN_SMOKE_ONLY', learned_trigger_smoke='after search, before each victim'))
    print('PREPARED_ONLY: full input audit + native Clean CUDA B32 smoke; no full training', flush=True)


def gate_check(plan):
    require_source_ready(plan)
    verify_inputs(plan)
    root, env = Path(plan['outdir']), execution()
    row = read(root / 'gpu_gate.json')
    if (row.get('binding') != preparation_binding(plan, env) or row.get('batch_size') != 32
            or row.get('steps') != 2 or row.get('real_train_input') is not True
            or row.get('status') != 'NATIVE_CLEAN_SMOKE_ONLY'
            or len(row.get('losses', [])) != 2 or not all(math.isfinite(x) for x in row['losses'])):
        raise ValueError('Missing/foreign native CUDA gate')
    return env


def cell_binding(plan, cell, env):
    result = dict(**preparation_binding(plan, env), cell=cell,
                  config_sha256=digest(plan['cfg'][cell]))
    if cell != 'clean_surrogate':
        root = Path(plan['outdir'])
        result.update(clean_sha256=sha(root / 'clean_surrogate/checkpoint.pt'),
                      trigger_sha256=sha(root / cell / 'trigger.pt'))
        if cell == 'backtime':
            result['backtime_poison_sha256'] = sha(root / 'backtime/poison.json')
    return result


def validate_checkpoint(blob, plan, cell, env):
    import torch
    if (blob.get('binding') != cell_binding(plan, cell, env)
            or blob.get('cfg') != plan['cfg'][cell]
            or type(blob.get('epoch')) is not int or not -1 <= blob['epoch'] <= 49
            or any(not blob.get(k) for k in ('model', 'optimizer', 'rng'))
            or set(blob['rng']) != {'python', 'numpy', 'torch', 'cuda'}
            or not isinstance(blob['rng']['cuda'], (list, tuple)) or not blob['rng']['cuda']):
        raise ValueError('Incomplete/foreign exact-resume checkpoint; no epoch0 fallback')
    groups = blob['optimizer'].get('param_groups', [])
    if not groups or any(g.get('lr') != .001 or g.get('momentum') != .9
            or g.get('weight_decay') != 0. or 'betas' in g for g in groups):
        raise ValueError('Checkpoint optimizer differs from HPE-Li SGD recipe')
    if blob['epoch'] >= 0 and not blob['optimizer'].get('state'):
        raise ValueError('Trained checkpoint lacks SGD momentum state')
    for state in blob['optimizer'].get('state', {}).values():
        if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Nonfinite saved optimizer state')
    if any(not torch.isfinite(v).all() for v in blob['model'].values()):
        raise ValueError('Nonfinite saved model')


def load_cell(plan, cell, env, *, final=False):
    import torch
    path = Path(plan['outdir']) / cell / 'checkpoint.pt'
    reject_links(path)
    blob = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint(blob, plan, cell, env)
    if final and blob['epoch'] != 49:
        raise ValueError('Only fixed final49 may be used for search/evaluation')
    return blob


def training_rows(plan, cell, trigger=None):
    data = dataset(plan, 'training')
    if cell == 'clean_surrogate':
        return data
    if cell == 'backtime':
        from attack.poison import PoisonedDataset
        row = read(Path(plan['outdir']) / 'backtime/poison.json')
        indices = row['indices']
        rows = PoisonedDataset(data, trigger, mode='train', rho=.1, dose_min=.2,
            dose_max=1., eps=plan['cfg'][cell]['eps'], pivot=1, theta_max_deg=40.,
            dose_mode='linear', axis=[0., 0., 1.], seed=42,
            select=('backtime_train_error' if plan['backtime_recipient_policy'] == 'adaptive' else 'uniform'),
            dataset='mmfi', dose_coupling='paired', cover_ratio=0., explicit_indices=indices)
        if rows.manifest() != row['manifest']:
            raise ValueError('BackTime selected poison identity/dose plan changed')
        return rows
    rows = common.paired_data(data, trigger, plan['cfg'][cell])
    if rows.manifest() != read(Path(plan['outdir']) / 'poison.json'):
        raise ValueError('Poison identity/dose/target plan differs')
    return rows


def injected_batch(trigger, batch):
    """Fixed triggers are already injected; frozen BackTime runs on CUDA batches."""
    import torch
    x = batch['csi'].to('cuda:0')
    if trigger is not None and getattr(trigger, 'requires_deferred_injection', False):
        trigger.to('cuda:0').eval()
        mask = batch['poisoned'].to('cuda:0').bool()
        with torch.no_grad():
            x = x.clone()
            if mask.any():
                x[mask] = trigger.apply_tensor(x[mask], batch['dose'].to('cuda:0')[mask])
    return x


def train(plan, cell, env):
    import torch
    from train_backdoor import _rng_state, _restore_rng_state, _victim_update
    root = Path(plan['outdir']) / cell
    reject_links(root)
    root.mkdir(exist_ok=True)
    trigger = None if cell == 'clean_surrogate' else load_trigger(plan, cell, env)
    model, optimizer = new_components(plan, cell)
    data = training_rows(plan, cell, trigger)
    path, start = root / 'checkpoint.pt', 0
    def save(epoch, loss=None):
        save_checkpoint(path, dict(epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(), rng=_rng_state(), train_loss=loss,
            cfg=plan['cfg'][cell], binding=cell_binding(plan, cell, env)))
    if path.exists():
        old = load_cell(plan, cell, env)
        model.load_state_dict(old['model'], strict=True)
        optimizer.load_state_dict(old['optimizer'])
        _restore_rng_state(old['rng'])
        start = old['epoch'] + 1
        print(f'RESUME {cell} next_epoch={start}; model/SGD/Python/NumPy/Torch/CUDA RNG restored', flush=True)
        del old
    else:
        save(-1)
    model.train()
    batches = common.loader(data, train=True)
    for epoch in range(start, 50):
        began, total, n = time.monotonic(), 0., 0
        for batch in batches:
            loss = _victim_update(model, optimizer, injected_batch(trigger, batch), batch['pose'].to('cuda:0'))
            if not math.isfinite(loss):
                raise ValueError('Nonfinite training loss; no retry')
            total += float(loss) * len(batch['csi'])
            n += len(batch['csi'])
        if n != COUNTS['training']:
            raise ValueError('Not every full TRAIN frame was consumed')
        save(epoch, total / n)
        print(f'{cell} epoch {epoch}/49: loss={total/n:.8f} lr=1.00e-03 t={time.monotonic()-began:.1f}s', flush=True)
    load_cell(plan, cell, env, final=True)
    print(f'TRAINED_NOT_EVALUATED {cell} final49', flush=True)


def target_fn(poses, doses):
    """Native MM-Fi pivot1 -> joints2/3, +Z40deg*d; Torch label equivalent."""
    import torch
    if poses.ndim not in (3, 4) or poses.shape[-2:] != (17, 3):
        raise ValueError('Expected Bx[1x]17x3 poses')
    doses = torch.as_tensor(doses, device=poses.device, dtype=poses.dtype)
    if doses.shape != (len(poses),) or not torch.isfinite(poses).all() or not torch.isfinite(doses).all():
        raise ValueError('Finite paired per-example doses required')
    if torch.any(doses < 0) or torch.any(doses > 1):
        raise ValueError('Dose must lie in [0,1]')
    angle = doses * (40. * math.pi / 180.)
    shape = (len(poses),) + (1,) * (poses.ndim - 2)
    cosine, sine = angle.cos().reshape(shape), angle.sin().reshape(shape)
    output = poses.clone()
    pivot = poses[..., 1:2, :]
    delta = poses[..., 2:4, :] - pivot
    rotated = delta.clone()
    rotated[..., 0] = delta[..., 0] * cosine - delta[..., 1] * sine
    rotated[..., 1] = delta[..., 0] * sine + delta[..., 1] * cosine
    output[..., 2:4, :] = rotated + pivot
    return torch.where((doses == 0).reshape((len(poses),) + (1,) * (poses.ndim - 1)), poses, output)


def bank_tensors(plan):
    """Legacy diagnostic helper; no source-aligned production fit calls this."""
    import torch
    import numpy as np
    inputs = verify_inputs(plan)
    data = dataset(plan, 'training')  # Never instantiate TEST for trigger fitting.
    result = {}
    for key, record in inputs['banks'].items():
        rows = [data[i] for i in record['indices']]
        if [data.items[i]['name'] for i in record['indices']] != record['names']:
            raise ValueError('TRAIN bank identities changed')
        x = torch.from_numpy(np.stack([r['csi'] for r in rows])).float().contiguous()
        y = torch.from_numpy(np.stack([r['pose'] for r in rows])).float().contiguous()
        d = torch.tensor(record['doses'], dtype=torch.float32)
        if x.shape != (len(rows), 3, 114, 10) or y.shape != (len(rows), 1, 17, 3):
            raise ValueError('TRAIN bank CSI/pose geometry differs')
        result[key] = (x, y, d)
    return result


def trigger_binding(plan, method, env, *, for_search=False):
    root = Path(plan['outdir'])
    binding = dict(**preparation_binding(plan, env), method=method,
        operator=plan['operators'][method], clean_sha256=sha(root / 'clean_surrogate/checkpoint.pt'))
    if method == 'backtime' and not for_search:
        binding['backtime_poison_sha256'] = sha(root / 'backtime/poison.json')
    return binding


def load_trigger(plan, method, env):
    import torch
    from attack.siba_ladder_mmfi import FrozenCSITrigger
    path = Path(plan['outdir']) / method / 'trigger.pt'
    reject_links(path)
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if blob.get('binding') != trigger_binding(plan, method, env):
        raise ValueError('Frozen trigger source/surrogate/input binding changed')
    if method == 'backtime':
        from attack.backtime_mmfi import FrozenBackTimeTrigger
        trigger = FrozenBackTimeTrigger.from_state_dict(blob['trigger'])
    else:
        trigger = FrozenCSITrigger.from_state_dict(blob['trigger'])
    if trigger.provenance() != blob.get('provenance'):
        raise ValueError('Frozen trigger provenance changed')
    return trigger


def backtime_selection_binding(plan, env):
    return dict(**preparation_binding(plan, env), policy=plan['backtime_recipient_policy'],
                clean_sha256=sha(Path(plan['outdir']) / 'clean_surrogate/checkpoint.pt'))


def backtime_poison(plan, env, indices=None):
    """Persist native warmup's selection; never substitute final49 Clean ranking."""
    from attack.poison import PoisonedDataset
    root = Path(plan['outdir']) / 'backtime'
    root.mkdir(exist_ok=True)
    path = root / 'poison.json'
    expected = backtime_selection_binding(plan, env)
    if path.exists():
        row = read(path)
        unsigned = dict(row)
        claimed = unsigned.pop('record_sha256', None)
        if claimed != digest(unsigned) or row.get('binding') != expected:
            raise ValueError('BackTime recipient policy/data/surrogate binding changed')
        if indices is not None and list(map(int, indices)) != row['indices']:
            raise ValueError('Warmup resume changed the already selected TRAIN recipients')
        training_rows(plan, 'backtime', common.Deferred())
        return row
    if indices is None:
        raise ValueError('BackTime recipients are selected inside its scratch warmup, not from Clean reference')
    data = dataset(plan, 'training')
    if len(indices) != plan['contract']['n_poison'] or len(set(map(int, indices))) != plan['contract']['n_poison']:
        raise ValueError('BackTime must retain exactly floor(.1*TRAIN) poison recipients')
    indices = list(map(int, indices))
    selected = PoisonedDataset(data, common.Deferred(), mode='train', rho=.1,
        dose_min=.2, dose_max=1., eps=plan['cfg']['backtime']['eps'], pivot=1,
        theta_max_deg=40., dose_mode='linear', axis=[0., 0., 1.], seed=42,
        select='backtime_train_error',
        dataset='mmfi', dose_coupling='paired', cover_ratio=0., explicit_indices=indices)
    row = dict(binding=expected, indices=indices, manifest=selected.manifest(),
        selection_metric='online pre-update SmoothL1 from scratch-surrogate epoch10',
        selection_state_artifact='search_state.pt', test_used=False,
        spacing='independent frame adaptation; native continuous-series non-overlap is not claimed')
    row['record_sha256'] = digest(row)
    write(path, row)
    print(f'BACKTIME_RECIPIENTS_LOCKED policy={plan["backtime_recipient_policy"]} '
          f'n={plan["contract"]["n_poison"]} TRAIN only', flush=True)
    return row


def full_train_doses(plan, method):
    """Registered paired-dose benchmark extension, not an author-native claim."""
    import numpy as np
    import torch
    values = np.random.default_rng(np.random.SeedSequence([42, 1103])).uniform(
        .2, 1., COUNTS['training'])
    if method == 'siba':
        for row in read(Path(plan['outdir']) / 'poison.json')['samples']:
            values[row['index']] = row['dose']
    return torch.tensor(values, dtype=torch.float32)


def source_epoch_batches(data, doses, batch_size):
    """Reiterable full TRAIN; no materialized full CSI tensor or TEST access."""
    import torch
    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(data)
        def __getitem__(self, index):
            row = data[index]
            return (torch.as_tensor(row['csi'], dtype=torch.float32),
                    torch.as_tensor(row['pose'], dtype=torch.float32), doses[index])
    rows = Rows()
    def batches(epoch):
        generator = torch.Generator().manual_seed(42 + int(epoch))
        return torch.utils.data.DataLoader(rows, batch_size=batch_size, shuffle=True,
            generator=generator, num_workers=0, drop_last=False, pin_memory=True)
    return batches


def fit_trigger(plan, method, env):
    import torch
    import attack.siba_ladder_mmfi as operators
    from models.factory import build_model
    root = Path(plan['outdir']) / method
    reject_links(root)
    root.mkdir(exist_ok=True)
    if method == 'ladder':
        operators.require_ladder_author_search()
    if (root / 'trigger.pt').exists():
        load_trigger(plan, method, env)
        print(f'FROZEN_TRIGGER_REUSED {method}; no new search', flush=True)
        return
    expected_binding = trigger_binding(plan, method, env, for_search=True)
    blob = load_cell(plan, 'clean_surrogate', env, final=True)
    if sha(Path(plan['outdir']) / 'clean_surrogate/checkpoint.pt') != expected_binding['clean_sha256']:
        raise ValueError('Clean surrogate changed while loading')
    common.seed()
    model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False).to('cuda:0')
    if method != 'backtime':
        model.load_state_dict(blob['model'], strict=True)
        model.eval()
    del blob
    data = dataset(plan, 'training')
    if len(data) != COUNTS['training']:
        raise ValueError('Source-aligned fitting requires every TRAIN frame')
    doses = full_train_doses(plan, method)
    common.seed()
    if method == 'backtime':
        import attack.backtime_mmfi as backtime
        config = backtime.BackTimeConfig(**plan['operators'][method])
    else:
        config = (operators.SIBAConfig if method == 'siba' else operators.LADDERConfig)(**plan['operators'][method])
    kwargs = dict(progress=lambda row: print(f'[{method} search] ' + json.dumps(row, allow_nan=False), flush=True))
    if method == 'backtime':
        function = backtime.fit_backtime_stream
        def selected(indices):
            record = backtime_poison(plan, env, indices)
            resolved = doses.clone()
            for item in record['manifest']['samples']:
                resolved[item['index']] = item['dose']
            return resolved
        kwargs.update(dataset_sha256=verify_inputs(plan)['full_content_sha256'],
            n_poison=plan['contract']['n_poison'], doses=doses,
            selection_callback=selected)
    else:
        function = operators.fit_siba_stream
        kwargs.update(dataset_sha256=digest(dict(
            inputs=verify_inputs(plan)['full_content_sha256'], doses=doses.tolist(),
            shuffle='epoch-local torch.Generator seed42+epoch')),
            n_train=len(data))
    # A state callback is used when supplied by the operator. Never silently
    # restart an interrupted expensive search without an authenticated state.
    import inspect
    state_path = root / 'search_state.pt'
    reject_links(state_path)
    parameters = inspect.signature(function).parameters
    if 'save_state' in parameters and 'resume_state' in parameters:
        if state_path.exists():
            state_blob = torch.load(state_path, map_location='cpu', weights_only=False)
            if state_blob.get('binding') != expected_binding:
                raise ValueError('Interrupted search state binding changed')
            kwargs['resume_state'] = state_blob['state']
            print(f'RESUME_TRIGGER_SEARCH {method}', flush=True)
        kwargs['save_state'] = lambda state: save_checkpoint(state_path,
            dict(binding=expected_binding, state=state))
    marker = root / 'search_started.json'
    if marker.exists() and not state_path.exists():
        raise ValueError('Interrupted search without resumable state; preserve artifacts and request explicit recovery')
    write(marker, dict(binding=expected_binding))
    began = time.monotonic()
    if method == 'backtime':
        trigger = function(model, data, target_fn, config, **kwargs)
        if trigger.training_indices != read(root / 'poison.json')['indices']:
            raise ValueError('Fitted BackTime recipients differ from victim poison manifest')
    else:
        trigger = function(model, source_epoch_batches(data, doses, config.batch_size),
                           target_fn, config, **kwargs)
    if sha(Path(plan['outdir']) / 'clean_surrogate/checkpoint.pt') != expected_binding['clean_sha256']:
        raise ValueError('Clean surrogate checkpoint changed during search')
    verify_inputs(plan)
    save_checkpoint(root / 'trigger.pt', dict(binding=trigger_binding(plan, method, env),
        trigger=trigger.state_dict(), provenance=trigger.provenance(), search_seconds=time.monotonic()-began))
    write(root / 'trigger_provenance.json', trigger.provenance())
    print(f'FROZEN_TRIGGER_COMPLETE {method}; seconds={time.monotonic()-began:.1f}', flush=True)


def learned_smoke(plan, method, env):
    import torch
    from attack.poison import collate
    from train_backdoor import _victim_update
    root = Path(plan['outdir']) / method
    path = root / 'learned_smoke.json'
    expected = cell_binding(plan, method, env)
    if path.exists():
        row = read(path)
        if (row.get('binding') != expected or row.get('batch_size') != 32
                or row.get('poison_examples') != 4 or row.get('steps') != 2
                or row.get('real_train_input') is not True or len(row.get('losses', [])) != 2
                or not all(math.isfinite(x) for x in row['losses'])):
            raise ValueError('Learned-trigger real CUDA gate changed')
        return
    trigger = load_trigger(plan, method, env)
    model, optimizer = new_components(plan, method)
    pairs = training_rows(plan, method, trigger)
    indices = sorted(pairs.poison_idx)[:4]
    indices += [i for i in range(len(pairs)) if i not in pairs.poison_idx][:28]
    batch = collate([pairs[i] for i in indices])
    before = next(model.parameters()).detach().clone()
    losses = [_victim_update(model, optimizer, injected_batch(trigger, batch), batch['pose'].to('cuda:0'))
              for _ in range(2)]
    if (tuple(batch['csi'].shape) != (32, 3, 114, 10)
            or sum(i in pairs.poison_idx for i in indices) != 4
            or not all(math.isfinite(x) for x in losses) or torch.equal(before, next(model.parameters()))):
        raise ValueError('Learned-trigger B32 finite/update/poison smoke failed')
    write(path, dict(binding=expected, batch_size=32, steps=2, poison_examples=4,
                     real_train_input=True, losses=losses, victim_checkpoint_saved=False))
    print(f'LEARNED_TRIGGER_CUDA_SMOKE_PASS {method}; fresh victim used, updates discarded', flush=True)


def save_npz(path, arrays):
    import numpy as np
    reject_links(path)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    if path.exists():
        with np.load(path, allow_pickle=False) as a, np.load(io.BytesIO(buffer.getvalue()), allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in b.files):
                raise ValueError('Existing partial predictions differ; preserve them')
        return
    temporary = path.with_name(path.name + '.saving')
    reject_links(temporary)
    with temporary.open('xb') as stream:
        stream.write(buffer.getvalue())
    os.replace(temporary, path)


def evaluate(plan, method, env):
    import numpy as np
    import torch
    from models.factory import build_model
    from extra_attack_eval import _loader, measure_pair
    from eval import metrics as M
    root = Path(plan['outdir']) / method
    old = load_cell(plan, method, env, final=True)
    if (root / 'evaluation.json').exists():
        return completed(plan, method, env)
    verify_inputs(plan, full_hash=True)
    trigger = load_trigger(plan, method, env)
    data = dataset(plan, 'test')
    if len(data) != COUNTS['test']:
        raise ValueError('Full official P3S3 TEST required')
    records, arrays, truth = {}, {}, None
    for label in ('clean_reference', method):
        saved = load_cell(plan, 'clean_surrogate', env, final=True) if label == 'clean_reference' else old
        model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                            dataset='mmfi', pretrained=False).to('cuda:0').eval()
        model.load_state_dict(saved['model'], strict=True)
        @torch.no_grad()
        def predict(mode, dose=0.):
            preds, truths, targets = [], [], []
            for batch in _loader(data, trigger, plan['cfg'][method], mode, dose):
                x = batch['csi'].to('cuda:0')
                if getattr(trigger, 'requires_deferred_injection', False) and mode == 'trigger@dose':
                    trigger.to('cuda:0').eval()
                    x = trigger.apply_tensor(x, dose)
                output = model(x)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if tuple(output.shape[1:]) != (1, 17, 3) or not torch.isfinite(output).all():
                    raise ValueError('Finite native HPE-Li output topology required')
                preds.append(output.cpu().numpy()[:, 0])
                truths.append(batch['pose'].numpy()[:, 0])
                targets.append(batch['target'].numpy()[:, 0])
            return np.concatenate(preds), np.concatenate(truths), np.concatenate(targets)
        clean, true, _ = predict('clean')
        if truth is not None and not np.array_equal(truth, true):
            raise ValueError('Control TEST identities differ')
        truth = true
        rows, grid, errors, no_trigger, dces = [], [], [], [], []
        for dose in GRID:
            print(f'[evaluate] {label} trigger={method} dose={dose:g}', flush=True)
            pred, again, target = predict('trigger@dose', dose)
            if not np.array_equal(true, again) or (dose == 0 and not np.array_equal(pred, clean)):
                raise ValueError('Evaluation order/dose0 identity changed')
            measured, per = measure_pair(clean, pred, true, target)
            rows.append(dict(dose=dose, **measured))
            grid.append(pred)
            errors.append(per['target_error_mm'])
            no_trigger.append(per['no_trigger_target_error_mm'])
            dces.append(per['dce_mm'])
        metrics = dict(t_mpjpe_mm=rows[-1]['t_mpjpe_mm'], target_pck_pct=rows[-1]['target_pck_pct'],
            mean_positive_dce_mm=float(np.mean([r['dce_mm'] for r in rows[1:]])),
            clean_mpjpe_mm=float(M.mpjpe(clean, true).mean()*1000),
            clean_pampjpe_mm=float(M.pa_mpjpe(clean, true).mean()*1000))
        records[label] = dict(metrics=metrics, dose_rows=rows)
        for key, value in dict(clean_prediction=clean, trigger_predictions=np.stack(grid),
                target_error_mm=np.stack(errors), no_trigger_target_error_mm=np.stack(no_trigger),
                dce_mm=np.stack(dces)).items():
            arrays[label + '__' + key] = value
        del model, saved
        torch.cuda.empty_cache()
    arrays.update(true_pose=truth, pair_ids=np.asarray([r['name'] for r in data.items]), doses=np.asarray(GRID))
    save_npz(root / 'per_sample.npz', arrays)
    verify_inputs(plan, full_hash=True)
    row = dict(profile=plan['profile'], status='complete', method=method, epoch=49, frames=len(data),
        binding=cell_binding(plan, method, env), records=records,
        checkpoint_sha256=sha(root / 'checkpoint.pt'), per_sample_sha256=sha(root / 'per_sample.npz'),
        metric_contract=dict(target=[2, 3], pck_clean_gt_scale=[5, 12], threshold=.2,
            endpoint=1., DCE='mean five positive doses', units='mm / percent'))
    row['record_sha256'] = digest(row)
    write(root / 'evaluation.json', row)
    print('EVALUATION_COMPLETE ' + method + ' ' + json.dumps(records[method]['metrics']), flush=True)
    return row


def completed(plan, method, env):
    root = Path(plan['outdir']) / method
    if not (root / 'evaluation.json').exists():
        return None
    row = read(root / 'evaluation.json')
    unsigned = dict(row)
    claimed = unsigned.pop('record_sha256', None)
    if (claimed != digest(unsigned) or row.get('profile') != plan['profile'] or row.get('status') != 'complete'
            or row.get('method') != method or row.get('binding') != cell_binding(plan, method, env)
            or row.get('epoch') != 49 or row.get('frames') != COUNTS['test']
            or row.get('checkpoint_sha256') != sha(root / 'checkpoint.pt')
            or row.get('per_sample_sha256') != sha(root / 'per_sample.npz')):
        raise ValueError('Completed evaluation/checkpoint binding changed')
    load_cell(plan, method, env, final=True)
    load_trigger(plan, method, env)
    return row


def export(plan, env):
    rows = [completed(plan, method, env) for method in plan['methods']]
    if any(row is None for row in rows):
        raise ValueError('All selected final49 evaluations are required')
    keys = ('t_mpjpe_mm', 'target_pck_pct', 'mean_positive_dce_mm', 'clean_mpjpe_mm', 'clean_pampjpe_mm')
    clean = rows[0]['records']['clean_reference']['metrics']
    lines = ['# Exploratory HPE-Li / MM-Fi P3S3 / seed42', '',
        '| Method | T-MPJPE mm | Target-PCK % | Mean DCE mm | Clean MPJPE mm | Clean PA mm |',
        '|---|---:|---:|---:|---:|---:|',
        f"| Clean reference (new surrogate) | — | — | — | {clean['clean_mpjpe_mm']:.2f} | {clean['clean_pampjpe_mm']:.2f} |"]
    for row in rows:
        metrics = row['records'][row['method']]['metrics']
        lines.append('| ' + row['method'].upper() + ' adapted | ' + ' | '.join(f'{metrics[k]:.2f}' for k in keys) + ' |')
    lines += ['', 'Independent exploratory adaptations, not equal-noise or original-classification reproductions.',
        'Clean-reference+trigger and same-victim no-trigger controls are preserved in evaluation.json.',
        'Single seed; this does not replace or complete the historical paper table.']
    text = '\n'.join(lines) + '\n'
    path = Path(plan['outdir']) / 'comparison.md'
    reject_links(path)
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError('Existing export changed; no overwrite')
    if not path.exists():
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)
    write(Path(plan['outdir']) / 'results.json', dict(status='complete', plan_sha256=plan['plan_sha256'], records=rows))
    print(text, flush=True)


def launch(plan):
    import torch
    require_source_ready(plan)
    env = gate_check(plan)
    siba_cfg = plan['operators']['siba']
    siba_updates = siba_cfg['epochs'] * math.ceil(COUNTS['training'] / siba_cfg['batch_size'])
    print(f'REGISTERED_SEARCH_BUDGET: SIBA {siba_cfg["epochs"]} full TRAIN epochs / '
          f'{siba_updates} B{siba_cfg["batch_size"]} updates; '
          'additional to Clean reference50 and scratch victim50 per selected method. No ETA guarantee.', flush=True)
    if 'backtime' in plan['methods']:
        bt = plan['operators']['backtime']
        generator_updates = max(0, bt['fit_epochs'] - bt['warmup'] - 1) * plan['contract']['n_poison']
        print(f'BACKTIME_ADDITIONAL: {bt["fit_epochs"]} full TRAIN scratch-surrogate epochs '
              f'(clean warmup {bt["warmup"]+1}) + {generator_updates} per-recipient generator Adam steps; '
              'NOT a lightweight bank fit; victim still scratch50. Explicit policy=' + plan['backtime_recipient_policy'], flush=True)
    require_idle()
    with log_to(plan['outdir'], 'train.clean_surrogate'):
        train(plan, 'clean_surrogate', env)
    torch.cuda.empty_cache()
    for method in plan['methods']:
        require_idle()
        if completed(plan, method, env) is not None:
            print(f'COMPLETE_REUSED {method}', flush=True)
            continue
        with log_to(plan['outdir'], f'search.{method}'):
            fit_trigger(plan, method, env)
        torch.cuda.empty_cache()
        require_idle()
        with log_to(plan['outdir'], f'smoke.{method}'):
            learned_smoke(plan, method, env)
        torch.cuda.empty_cache()
        require_idle()
        with log_to(plan['outdir'], f'train.{method}'):
            train(plan, method, env)
        torch.cuda.empty_cache()
        require_idle()
        with log_to(plan['outdir'], f'evaluate.{method}'):
            evaluate(plan, method, env)
        torch.cuda.empty_cache()
    export(plan, env)


def status(args):
    """Do not construct models, initialize CUDA, or imply that a PID is progress."""
    from collections import deque
    root = args.outdir
    cells = ('clean_surrogate',) + selected_queue(args.backtime_policy, getattr(args, 'methods', None))
    for cell in cells:
        folder = root / cell
        print(json.dumps(dict(cell=cell, checkpoint=(folder/'checkpoint.pt').is_file(),
            frozen_trigger=(folder/'trigger.pt').is_file(), evaluation=(folder/'evaluation.json').is_file())))
    logs = sorted((root/'logs').glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in logs[:3]:
        print(f'LOG={path}')
        with path.open(encoding='utf-8', errors='replace') as stream:
            print(''.join(deque(stream, maxlen=12)))
    print('READ_ONLY; changing epoch/search logs establish progress, not file/PID existence.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path, default=Path.home()/'backdooranalog')
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--jobs-per-gpu', type=int, default=1)
    parser.add_argument('--backtime-policy', choices=BACKTIME_POLICIES,
                        help='Opt in to source-aligned adaptive BackTime; uniform is not an approved adaptation.')
    parser.add_argument('--methods', nargs='+', choices=('siba', 'backtime', 'ladder'),
                        help='Explicit queue; siba backtime runs only available sources, never silently skips LADDER.')
    parser.add_argument('--mode', choices=('inspect', 'preflight', 'prepare', 'check', 'launch', 'status'), required=True)
    args = parser.parse_args(argv)
    args.outdir = args.outdir or args.data_home/'runs'/(
        output_profile(args.backtime_policy, args.methods))
    for path in (args.data_home, args.outdir):
        reject_links(path)
    if args.device != 'cuda:0' or args.jobs_per_gpu != 1:
        parser.error('This profile requires one serial worker on cuda:0')
    if args.mode == 'status':
        status(args)
        return
    plan = build_plan(args.data_home, args.outdir, args.device, args.jobs_per_gpu,
                      args.backtime_policy, args.methods)
    if args.mode == 'inspect':
        print(json.dumps(plan, indent=2))
        print('INSPECT_ONLY; no data audit, CUDA initialization or training')
        return
    if args.mode in ('prepare', 'check', 'launch'):
        require_source_ready(plan)
    from run_mmfi_tables import _cell_lock
    root = Path(plan['outdir'])
    with _cell_lock(root/'.controller'):
        if shutil.disk_usage(plan['home']).free < 5*2**30:
            raise ValueError('At least 5 GiB free required for checkpoints and full predictions')
        with log_to(root, args.mode):
            if args.mode == 'preflight':
                audit(plan)
                print('CPU_PREFLIGHT_ONLY; no CUDA or training')
            else:
                require_source_ready(plan)
                require_idle()
                if args.mode == 'prepare':
                    prepare(plan)
                elif args.mode == 'check':
                    gate_check(plan)
                    print('VERIFIED_PRELAUNCH_GATE; no full training')
                else:
                    print('EXPLICIT_LAUNCH: Clean50 -> ' + ' -> '.join(plan['methods']) + '; each victim scratch50, serial1GPU', flush=True)
                    launch(plan)


if __name__ == '__main__':
    main()
