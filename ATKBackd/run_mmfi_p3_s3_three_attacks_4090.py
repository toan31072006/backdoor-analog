"""Exploratory MM-Fi P3S3 HPE-Li: Input-Aware, BadNets and SIG, seed42.

Independent scratch victims; fixed final49; one job/GPU. Prepare audits full
inputs and runs real B32 CUDA steps, but never starts full training. Historical
P3/defense/PiW runs are not writable outputs. Input-Aware is an explicitly
white-box pose/dose adaptation, not an unchanged classification reproduction.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
from importlib.metadata import version
import io
import json
import math
import ntpath
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid

os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  MKL_THREADING_LAYER='SEQUENTIAL', PYTHONUTF8='1')
HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_p3s3_hpeli_badnets_sig_inputaware_s42_4090_v1'
METHODS = ('input_aware', 'badnets', 'sig_source')
GRID = (0., .2, .4, .6, .8, 1.)
COUNTS = {'training': 240570, 'test': 80190}
ACTION_SHA = '2e8fc11192cc3ee7ee3f5544cb3c5e19f771f7f7265dde93af8969af41a2be3e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, *, immutable=False):
    path = Path(path)
    if path.exists() and immutable:
        if read(path) != value:
            raise ValueError(f'Existing immutable record differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.saving')
    if temporary.exists():
        raise ValueError(f'Unfinished write; inspect without deleting artifacts: {temporary}')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def sources():
    # Bind the entire code-only scientific snapshot, never tests/cache/runs.
    result = {}
    for path in HERE.rglob('*'):
        relative = path.relative_to(HERE)
        if any(p.startswith('.') or p in ('tests', '__pycache__', 'runs', 'envs')
               for p in relative.parts):
            continue
        if path.is_file() and (path.suffix in ('.py', '.yaml', '.yml', '.json')
                               or path.name.startswith(('LICENSE', 'NOTICE'))):
            result[relative.as_posix()] = sha(path)
    return result


def verify_bundle():
    path = HERE.parent / 'P3S3_BUNDLE_MANIFEST.json'
    if not path.exists():
        return  # Source checkout is also usable; plan binds its actual files.
    manifest = read(path)
    if manifest.get('profile') != PROFILE:
        raise ValueError('Wrong P3S3 code bundle')
    for name, expected in manifest['files'].items():
        candidate = (HERE.parent / name).resolve()
        if (not candidate.is_relative_to(HERE.parent) or candidate.is_symlink()
                or not candidate.is_file() or sha(candidate) != expected['sha256']
                or candidate.stat().st_size != expected['bytes']):
            raise ValueError(f'Missing/changed bundle file: {name}')


def configuration(home, method):
    import run_mmfi_p3_splits as p3
    from run_a1_native_comparison import _method_config
    from train_backdoor import _resolve_training_config, _validate_training_contract
    if method not in METHODS:
        raise ValueError('Only the three approved new P3S3 methods')
    reference = p3._configuration(Path(home), 'cuda:0', 's3', 'a1')
    if method == 'input_aware':
        from attack.input_aware_mmfi import source_config
        cfg = {k:v for k,v in reference.items()
               if k != 'confirmation_profile' and not k.startswith('method_')}
        cfg.update(trigger=method, input_aware=source_config(),
                   attacker_access='white_box_generator_and_victim_training',
                   clean_label_cover_ratio=.1)
    else:
        cfg = _method_config(Path(home), 'cuda:0', 0, method)
        cfg.update(mmfi_protocol='protocol3', mmfi_setting='s3',
                   mmfi_split_seed=0, mmfi_random_ratio=.8,
                   attacker_access='data_only')
        cfg = _resolve_training_config(cfg)
    cfg.update(p3s3_three_attack_profile=PROFILE, ckpt_every=1)
    _validate_training_contract(cfg)
    common = ('model', 'seed', 'experiment_name', 'mmfi_protocol', 'mmfi_setting',
              'mmfi_split_seed', 'mmfi_random_ratio', 'optimizer', 'lr', 'momentum',
              'weight_decay', 'batch_size', 'epochs', 'victim_epochs', 'num_workers',
              'pretrained', 'data_parallel', 'rho', 'pivot', 'eps', 'theta_max_deg',
              'payload_axis', 'target_joints', 'dose_mode', 'dose_min', 'dose_max',
              'dose_grid', 'dose_coupling', 'victim_loss', 'poison_select')
    changed = [k for k in common if cfg.get(k) != reference.get(k)]
    if changed or any(k.startswith('comparison_peak_') for k in cfg):
        raise ValueError(f'Common P3S3 A1 victim/target recipe changed: {changed}')
    return cfg


def build_plan(home, outdir, device='cuda:0', jobs=1):
    home, outdir = Path(home).resolve(), Path(outdir).resolve()
    runs = home / 'runs'
    if outdir == runs or not outdir.is_relative_to(runs):
        raise ValueError('Choose a dedicated new output beneath data-home/runs')
    protected = ('mmfi_p3_hpeli_s42_v1', 'mmfi_p3_hpeli_extra_s42_3090_v1',
                 'mmfi_three_tables_4090_win_fresh_v1')
    if any(outdir == runs/name or (runs/name) in outdir.parents for name in protected):
        raise ValueError('Do not overwrite historical results')
    if device != 'cuda:0' or jobs != 1:
        raise ValueError('This profile fixes serial one-job cuda:0 execution')
    plan = dict(profile=PROFILE, schema=1, home=str(home), outdir=str(outdir),
        device=device, jobs_per_gpu=1, methods=list(METHODS),
        cfg={m:configuration(home, m) for m in METHODS}, sources=sources(),
        runtime=dict(python=platform.python_version(), system=platform.system(),
                     packages={n:version(n) for n in ('torch','numpy','scipy','PyYAML')}),
        contract=dict(seed=42, protocol='protocol3', setting='s3',
            counts=COUNTS, scratch=True, victim_epochs=50, final_epoch=49,
            batch_size=32, drop_last=False, amp=False, test_checkpoint_selection=False,
            matching='same ordered TRAIN/TEST, poison identities and paired U(.2,1) doses',
            doses=list(GRID), target_joints=[2,3], pck_scale=[5,12], threshold=.2,
            status='EXPLORATORY_SINGLE_SEED_NEW_THREE_METHOD_EXTENSION',
            caveats=['Input-Aware has extra white-box generator/mask training.',
                'Source mask warmup25 adds TRAIN-only attacker computation, not victim epochs.',
                'Input-Aware retains 10% disjoint cross-trigger clean-label recipients.',
                'Fixed cross recipient IDs and paired doses adapt original per-batch attack.',
                'Native operators and learned masks do not have equal realized perturbation budgets.',
                'All classification methods are CSI/pose/dose adaptations; SIG changes label policy.',
                'No across-seed uncertainty or claim of unseen TEST tuning.']))
    plan['plan_sha256'] = digest(plan)
    return plan


def load_plan(plan):
    if read(Path(plan['outdir'])/'plan.json') != plan:
        raise ValueError('Plan/source/runtime differs; preserve old run and use a new output')


def gpu():
    import torch
    from run_piw_hpeli_clean import _setup_execution
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    info = torch.cuda.get_device_properties(0)
    if '4090' not in info.name or info.total_memory < 22*2**30:
        raise RuntimeError(f'Expected Van RTX4090 with24GiB, found {info.name}')
    return _setup_execution('cuda:0')


def active_compute(text, windows, own_pid):
    allowed = {'explorer.exe','dwm.exe','chrome.exe','msedge.exe','msedgewebview2.exe',
        'code.exe','lockapp.exe','searchhost.exe','startmenuexperiencehost.exe',
        'shellexperiencehost.exe','shellhost.exe','textinputhost.exe','systemsettings.exe',
        'applicationframehost.exe','windowsterminal.exe','taskmgr.exe','snippingtool.exe',
        'crossdeviceresume.exe','copilot.exe','copilotapp.exe','copilotapphost.exe',
        'phoneexperiencehost.exe','teams.exe','ms-teams.exe','teamviewer.exe','logonui.exe'}
    conflicts = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if len(row) != 2 or not row[0].strip().isdigit():
            raise ValueError(f'Cannot identify GPU compute process: {row}')
        pid, name = int(row[0]), row[1].strip()
        if pid == own_pid:
            continue
        if windows and ntpath.basename(name.replace('/', '\\')).lower() in allowed:
            continue
        conflicts.append((pid, name))
    return conflicts


def require_idle():
    report = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=30)
    conflicts = active_compute(report.stdout, os.name == 'nt', os.getpid())
    if conflicts:
        raise ValueError(f'Other/unknown GPU jobs active; no process stopped: {conflicts}')


class Deferred:
    requires_deferred_injection = True


def paired_data(base, trigger, cfg):
    from attack.poison import PoisonedDataset
    return PoisonedDataset(base, trigger, mode='train', rho=.1, dose_min=.2,
        dose_max=1., eps=cfg['eps'], pivot=1, theta_max_deg=40., dose_mode='linear',
        axis=cfg['payload_axis'], seed=42, select='uniform', dataset='mmfi',
        dose_coupling='paired', cover_ratio=0.)


def poison_summary(data, method):
    import numpy as np
    record = data.manifest()
    cross = []
    if method == 'input_aware':
        available = np.array([i for i in range(len(data)) if i not in data.poison_idx])
        rng = np.random.default_rng(np.random.SeedSequence([42,7044]))
        cross = sorted(map(int, rng.choice(available, math.floor(.1*len(data)), replace=False)))
    return dict(common_poison_sha256=digest(record), poison_plan_sha256=record['poison_plan_sha256'],
                n_total=len(data), n_poison=data.n_poison, n_cross=len(cross),
                cross_indices_sha256=digest(cross)), set(cross)


def input_stats(root, records):
    result = []
    for row in records:
        path = root/row['path']
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Audited dataset file missing/linked: {path}')
        st = path.stat()
        if st.st_size != row['bytes']:
            raise ValueError('Dataset file size changed since full audit')
        result.append([row['path'], st.st_size, st.st_mtime_ns])
    return digest(result)


def audit(plan):
    import numpy as np
    import run_mmfi_p3_splits as p3
    root = Path(plan['outdir'])
    if (root/'plan.json').exists():
        load_plan(plan)
        return verify_inputs(plan)
    if any(p.name != '.controller' for p in root.iterdir()):
        raise ValueError('Nonempty output without this plan; preserve it and choose a NEW output')
    action = Path(plan['cfg']['badnets']['action_npy'])
    if sha(action) != ACTION_SHA or not np.isfinite(np.load(action, allow_pickle=False)).all():
        raise ValueError('Authentic finite data_bend.npy is required')
    cfg = plan['cfg']['badnets']
    data_root = Path(cfg['dataset_root']).resolve()
    data = {s:p3._dataset(cfg,s) for s in COUNTS}
    splits = {s:p3._identity_digest(ds) for s,ds in data.items()}
    if any(len(data[s]) != n for s,n in COUNTS.items()):
        raise ValueError('Incomplete official P3S3 processed frames')
    a,b = splits['training']['sequence_keys'],splits['test']['sequence_keys']
    if len(a)!=810 or len(b)!=270 or a & b or {x[1] for x in a}&{x[1] for x in b}:
        raise ValueError('P3S3 sequence/subject split drift or overlap')
    for row in splits.values():
        row.pop('sequence_keys')
    selected = {}
    for ds in data.values():
        for item in ds.items:
            for key in ('csi','kpt'):
                path = Path(item[key]).resolve()
                if not path.is_relative_to(data_root) or path.is_symlink():
                    raise ValueError('Input file escapes dataset root')
                selected[path.relative_to(data_root).as_posix()] = (path,key)
    records = []
    for i,(name,(path,kind)) in enumerate(sorted(selected.items()),1):
        before = path.stat()
        values = np.load(path, allow_pickle=False, mmap_mode='r')
        expected = ((3,114,10) if kind=='csi' else None)
        if (not np.issubdtype(values.dtype,np.number) or np.iscomplexobj(values)
                or not np.isfinite(values).all() or (expected and values.shape!=expected)
                or (kind=='kpt' and (values.ndim!=3 or values.shape[1:]!=(17,3)))
                or (kind=='csi' and (values.min() < -1e-6 or values.max() > 1+1e-6))):
            raise ValueError(f'Bad full input shape/range/values: {path}')
        del values
        record = dict(path=name, bytes=before.st_size, sha256=sha(path))
        after = path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('Input changed during full audit')
        records.append(record)
        if i%10000==0:
            print(f'[preflight] hashed/validated {i}/{len(selected)} selected files',flush=True)
    poisons = {}
    for method in METHODS:
        pairs = paired_data(data['training'],Deferred(),plan['cfg'][method])
        poisons[method],_ = poison_summary(pairs,method)
    if len({v['common_poison_sha256'] for v in poisons.values()}) != 1:
        raise ValueError('Methods do not have identical poison IDs/doses/labels')
    receipt = dict(plan_sha256=plan['plan_sha256'], action_sha256=ACTION_SHA,
        dataset_root=str(data_root), splits=splits, files=records,
        full_content_sha256=digest(records), input_stat_fingerprint=input_stats(data_root,records),
        poison=poisons, no_test_selection=True)
    write(root/'plan.json',plan,immutable=True)
    write(root/'inputs.json',receipt,immutable=True)
    print('FULL_INPUT_AUDIT_PASS; no victim training',flush=True)
    return receipt


def verify_inputs(plan):
    load_plan(plan)
    row = read(Path(plan['outdir'])/'inputs.json')
    if (row['plan_sha256']!=plan['plan_sha256'] or row['action_sha256']!=ACTION_SHA
            or sha(plan['cfg']['badnets']['action_npy'])!=ACTION_SHA
            or row['full_content_sha256']!=digest(row['files'])
            or input_stats(Path(row['dataset_root']),row['files'])!=row['input_stat_fingerprint']):
        raise ValueError('Audited inputs/prior changed; no launch')
    return row


def seed():
    import numpy as np
    import torch
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def make_components(plan, method):
    import torch
    from models.factory import build_model
    from train_backdoor import _build_optimizer,build_trigger
    cfg = plan['cfg'][method]
    seed()
    model = build_model('hpeli',num_keypoints=17,subcarrier_num=114,
                        dataset='mmfi',pretrained=False).to('cuda:0')
    name,opt = _build_optimizer(model,cfg,'mmfi')
    if name!='sgd':
        raise ValueError('Expected common HPE-Li SGD victim optimizer')
    trigger = build_trigger(cfg) if method!='input_aware' else None
    opts, schedulers = {},{}
    if method=='input_aware':
        from attack.input_aware_mmfi import InputAwareCSI
        trigger = InputAwareCSI().to('cuda:0')
        opts = {'mask':trigger.make_mask_optimizer(), 'generator':trigger.make_generator_optimizer()}
        schedulers = {'mask':trigger.make_mask_scheduler(opts['mask']),
                      'generator':trigger.make_generator_scheduler(opts['generator'])}
    return model,opt,trigger,opts,schedulers


class TrainingRows:
    def __init__(self,pairs,cross):
        self.pairs,self.cross = pairs,cross
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self,index):
        row = self.pairs[index]
        row['cross'] = index in self.cross
        return row


def train_collate(rows):
    import torch
    from attack.poison import collate
    value = collate(rows)
    value['cross'] = torch.tensor([r['cross'] for r in rows],dtype=torch.bool)
    return value


def loader(data, *, train=True, custom=False):
    from torch.utils.data import DataLoader
    from attack.poison import collate
    return DataLoader(data,batch_size=32,shuffle=train,num_workers=0,
                      drop_last=False,collate_fn=train_collate if custom else collate,
                      pin_memory=train)


def step(method,model,opt,trigger,opts,batch,donor=None):
    from train_backdoor import _victim_update
    x,y = batch['csi'].to('cuda:0'),batch['pose'].to('cuda:0')
    if method!='input_aware':
        return {'pose_loss':_victim_update(model,opt,x,y)}
    return trigger.train_step(model,opt,opts['generator'],x,batch['clean_pose'].to('cuda:0'),
        donor['csi'].to('cuda:0'),batch['poisoned'].to('cuda:0').bool(),
        batch['cross'].to('cuda:0'),batch['dose'].to('cuda:0'),y)


def atomic_checkpoint(path,value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.saving')
    if temporary.exists():
        raise ValueError(f'Unfinished checkpoint: {temporary}')
    torch.save(value,temporary)
    os.replace(temporary,path)


def checkpoint_binding(plan,method,inputs,execution,poison):
    return dict(profile=PROFILE,plan_sha256=plan['plan_sha256'],method=method,
        config_sha256=digest(plan['cfg'][method]),inputs_sha256=sha(Path(plan['outdir'])/'inputs.json'),
        input_stat_fingerprint=inputs['input_stat_fingerprint'],execution=execution,poison=poison)


def validate_checkpoint(blob,binding,method):
    if blob.get('binding')!=binding or binding.get('method')!=method:
        raise ValueError('Resume source/data/runtime/GPU/poison binding mismatch')
    epoch,mask_epoch,phase=blob.get('epoch'),blob.get('mask_epoch'),blob.get('phase')
    if (type(epoch) is not int or type(mask_epoch) is not int or not -1<=epoch<=49
            or phase not in ('mask','victim')):
        raise ValueError('Unknown checkpoint phase/epoch')
    if method=='input_aware':
        if (not 0<=mask_epoch<=24 or (phase=='mask' and epoch!=-1)
                or (phase=='victim' and (mask_epoch!=24 or epoch<0))):
            raise ValueError('Input-Aware phase/mask warmup/victim epoch mismatch')
    elif phase!='victim' or epoch<0 or mask_epoch!=-1:
        raise ValueError('Native static method cannot have mask warmup state')
    if digest(blob.get('cfg',{}))!=binding.get('config_sha256'):
        raise ValueError('Checkpoint resolved configuration changed')
    for key in ('model','optimizer','rng'):
        if not blob.get(key):
            raise ValueError(f'Missing exact-resume state: {key}')
    if any(k not in blob['rng'] for k in ('python','numpy','torch','cuda')) or blob['rng']['cuda'] is None:
        raise ValueError('Missing exact CUDA/Python/NumPy/torch resume RNG')
    if method=='input_aware':
        if (not blob.get('trigger') or set(blob.get('trigger_optimizers',{}))!={'mask','generator'}
                or set(blob.get('schedulers',{}))!={'mask','generator'}):
            raise ValueError('Missing Input-Aware generator/mask/optimizer/scheduler state')
    elif blob.get('trigger') is not None or blob.get('trigger_optimizers') or blob.get('schedulers'):
        raise ValueError('Static method unexpectedly carries learned-trigger state')


def smoke(plan,method):
    import torch
    import run_mmfi_p3_splits as p3
    execution = gpu()
    model,opt,trigger,opts,schedulers = make_components(plan,method)
    train = p3._dataset(plan['cfg'][method],'training')
    pairs = paired_data(train,trigger,plan['cfg'][method])
    poison,cross = poison_summary(pairs,method)
    if poison!=read(Path(plan['outdir'])/'inputs.json')['poison'][method]:
        raise ValueError('Real smoke poison plan differs from full preflight')
    # Real physical B32, deliberately include poison/cross/clean examples.
    indices = sorted(pairs.poison_idx)[:4]
    indices += sorted(cross)[:4]
    indices += [i for i in range(len(pairs)) if i not in pairs.poison_idx and i not in cross][:32-len(indices)]
    batch = train_collate([TrainingRows(pairs,cross)[i] for i in indices])
    donor = next(iter(loader(train)))
    model.train()
    if method=='input_aware':
        trigger.mask_step(batch['csi'].to('cuda:0'),donor['csi'].to('cuda:0'),opts['mask'])
        trigger.freeze_mask()
    results=[]
    for _ in range(2):
        results.append(step(method,model,opt,trigger,opts,batch,donor))
    torch.cuda.synchronize()
    if any(not math.isfinite(float(x)) for r in results for x in r.values()):
        raise ValueError('Nonfinite smoke update')
    return dict(plan_sha256=plan['plan_sha256'],method=method,execution=execution,
                real_input=True,batch_size=32,steps=2,poison=poison,loss=results[-1])


def train_cell(plan,method):
    import torch
    import run_mmfi_p3_splits as p3
    from train_backdoor import _rng_state,_restore_rng_state
    from attack.poison import collate
    folder = Path(plan['outdir'])/method
    folder.mkdir(exist_ok=True)
    inputs = verify_inputs(plan); execution = gpu()
    model,opt,trigger,opts,schedulers = make_components(plan,method)
    train = p3._dataset(plan['cfg'][method],'training')
    pairs = paired_data(train,trigger,plan['cfg'][method])
    poison,cross = poison_summary(pairs,method)
    if poison!=inputs['poison'][method]:
        raise ValueError('Training poison/cross records changed')
    binding = checkpoint_binding(plan,method,inputs,execution,poison)
    checkpoint = folder/'checkpoint.pt'
    start,mask_start = 0,0
    if checkpoint.exists():
        old = torch.load(checkpoint,map_location='cpu',weights_only=False)
        validate_checkpoint(old,binding,method)
        model.load_state_dict(old['model'],strict=True); opt.load_state_dict(old['optimizer'])
        if method=='input_aware':
            trigger.load_state_dict(old['trigger'],strict=True)
            for k in opts: opts[k].load_state_dict(old['trigger_optimizers'][k])
            for k in schedulers: schedulers[k].load_state_dict(old['schedulers'][k])
        _restore_rng_state(old['rng'])
        start,mask_start = old['epoch']+1,old['mask_epoch']+1
        print(f'RESUME phase={old["phase"]} next_victim_epoch={start} next_mask_epoch={mask_start}',flush=True)
    def save(phase,epoch,mask_epoch):
        value = dict(binding=binding,phase=phase,epoch=epoch,mask_epoch=mask_epoch,
            cfg=plan['cfg'][method],model=model.state_dict(),optimizer=opt.state_dict(),rng=_rng_state(),
            trigger=trigger.state_dict() if method=='input_aware' else None,
            trigger_optimizers={k:o.state_dict() for k,o in opts.items()},
            schedulers={k:s.state_dict() for k,s in schedulers.items()})
        atomic_checkpoint(checkpoint,value)
    if method=='input_aware':
        for epoch in range(mask_start,25):
            began=time.monotonic(); losses=[]; trigger.train()
            for a,b in zip(loader(train),loader(train)):
                report=trigger.mask_step(a['csi'].to('cuda:0'),b['csi'].to('cuda:0'),opts['mask'])
                losses.append(float(report['total_loss']))
            schedulers['mask'].step(); save('mask',-1,epoch)
            print(f'mask_epoch {epoch}/24: loss={sum(losses)/len(losses):.8f} t={time.monotonic()-began:.1f}s',flush=True)
        trigger.freeze_mask()
    for epoch in range(start,50):
        began=time.monotonic(); losses=[]; model.train()
        if method=='input_aware': trigger.train(); trigger.freeze_mask()
        train_loader=loader(TrainingRows(pairs,cross),custom=True)
        donor_loader=loader(train) if method=='input_aware' else None
        batches=zip(train_loader,donor_loader) if donor_loader is not None else ((a,None) for a in train_loader)
        for batch,donor in batches:
            report=step(method,model,opt,trigger,opts,batch,donor)
            losses.append(float(report['pose_loss']))
            if not math.isfinite(losses[-1]): raise ValueError('Nonfinite training loss')
        if method=='input_aware': schedulers['generator'].step()
        save('victim',epoch,24 if method=='input_aware' else -1)
        print(f'epoch {epoch}/49: loss={sum(losses)/len(losses):.8f} lr={opt.param_groups[0]["lr"]:.2e} t={time.monotonic()-began:.1f}s',flush=True)
    final=torch.load(checkpoint,map_location='cpu',weights_only=False)
    validate_checkpoint(final,binding,method)
    if final['epoch']!=49: raise ValueError('Final49 checkpoint missing')
    del final,model,opt,trigger,opts,schedulers
    torch.cuda.empty_cache()
    evaluate(plan,method)


def evaluate(plan,method):
    import numpy as np
    import torch
    import run_mmfi_p3_splits as p3
    from models.factory import build_model
    from train_backdoor import build_trigger
    from extra_attack_eval import _loader,measure_pair
    from eval import metrics as M
    folder=Path(plan['outdir'])/method; cfg=plan['cfg'][method]
    checkpoint=folder/'checkpoint.pt'; old=torch.load(checkpoint,map_location='cpu',weights_only=False)
    inputs=verify_inputs(plan)
    expected=checkpoint_binding(plan,method,inputs,gpu(),inputs['poison'][method])
    validate_checkpoint(old,expected,method)
    if old['epoch']!=49 or old['phase']!='victim':
        raise ValueError('Evaluation requires bound fixed final49 checkpoint')
    model=build_model('hpeli',num_keypoints=17,subcarrier_num=114,dataset='mmfi',pretrained=False)
    model.load_state_dict(old['model'],strict=True); model=model.to('cuda:0').eval()
    if method=='input_aware':
        from attack.input_aware_mmfi import InputAwareCSI
        trigger=InputAwareCSI().to('cuda:0'); trigger.load_state_dict(old['trigger'],strict=True); trigger.eval()
    else: trigger=build_trigger(cfg)
    data=p3._dataset(cfg,'test')
    if len(data)!=COUNTS['test']: raise ValueError('Full official TEST count differs')
    ids=np.asarray([row['name'] for row in data.items],dtype=str)
    @torch.no_grad()
    def predict(mode,dose=0.):
        predictions=[]; truths=[]; targets=[]
        for batch in _loader(data,trigger,cfg,mode,dose):
            x=batch['csi'].to('cuda:0')
            if method=='input_aware' and mode=='trigger@dose': x=trigger.inject_tensor(x,dose)
            output=model(x)
            if isinstance(output,(tuple,list)): output=output[0]
            if tuple(output.shape[1:])!=(1,17,3): raise ValueError('HPE-Li output topology changed')
            predictions.append(output.cpu().numpy()[:,0]); truths.append(batch['pose'].numpy()[:,0])
            targets.append(batch['target'].numpy()[:,0])
        return np.concatenate(predictions),np.concatenate(truths),np.concatenate(targets)
    clean,true,_=predict('clean')
    rows=[]; errors=[]; no_trigger=[]; dces=[]; scales=None
    for dose in GRID:
        print(f'[evaluate] {method} dose={dose:g}',flush=True)
        pred,again,target=predict('trigger@dose',dose)
        if not np.array_equal(again,true) or (dose==0 and not np.array_equal(pred,clean)):
            raise ValueError('Evaluation order/dose-zero identity changed')
        metrics,arrays=measure_pair(clean,pred,true,target)
        rows.append(dict(dose=dose,**metrics)); errors.append(arrays['target_error_mm'])
        no_trigger.append(arrays['no_trigger_target_error_mm']); dces.append(arrays['dce_mm'])
        scales=arrays['clean_scale_mm'] if scales is None else scales
    metrics=dict(t_mpjpe_mm=rows[-1]['t_mpjpe_mm'],target_pck_pct=rows[-1]['target_pck_pct'],
        mean_positive_dce_mm=float(np.mean([r['dce_mm'] for r in rows[1:]])),
        clean_mpjpe_mm=float(M.mpjpe(clean,true).mean()*1000),
        clean_pampjpe_mm=float(M.pa_mpjpe(clean,true).mean()*1000))
    buffer=io.BytesIO()
    np.savez_compressed(buffer,pair_ids=ids,doses=np.asarray(GRID),
        target_error_mm=np.stack(errors),no_trigger_target_error_mm=np.stack(no_trigger),
        dce_mm=np.stack(dces),clean_scale_mm=scales)
    sample=folder/'per_sample.npz'; temporary=sample.with_suffix('.npz.saving')
    if temporary.exists(): raise ValueError('Unfinished per-sample export; inspect before retry')
    if sample.exists():
        with np.load(sample,allow_pickle=False) as before, np.load(io.BytesIO(buffer.getvalue()),allow_pickle=False) as after:
            if set(before.files)!=set(after.files) or any(not np.array_equal(before[k],after[k]) for k in after.files):
                raise ValueError('Existing partial per-sample result differs; preserve original')
    else:
        temporary.write_bytes(buffer.getvalue()); os.replace(temporary,sample)
    write(folder/'evaluation.json',dict(status='complete',profile=PROFILE,method=method,
        plan_sha256=plan['plan_sha256'],checkpoint_epoch=49,frames=len(data),metrics=metrics,
        dose_rows=rows,checkpoint_sha256=sha(checkpoint),per_sample_sha256=sha(sample),
        metric_contract=dict(target_joints=[2,3],pck_scale=[5,12],threshold=.2,
            comparison_dose=1.,DCE='mean of five positive doses',error_units='mm',pck_units='%')))
    print(f'COMPLETE {method}: {metrics}',flush=True)


def completed(plan,method):
    import torch
    folder=Path(plan['outdir'])/method
    if not (folder/'evaluation.json').exists(): return None
    row=read(folder/'evaluation.json')
    if (row.get('status')!='complete' or row.get('method')!=method
            or row.get('plan_sha256')!=plan['plan_sha256'] or row.get('checkpoint_epoch')!=49
            or row.get('frames')!=COUNTS['test'] or row.get('checkpoint_sha256')!=sha(folder/'checkpoint.pt')
            or row.get('per_sample_sha256')!=sha(folder/'per_sample.npz')):
        raise ValueError('Completed result/checkpoint/per-sample binding differs')
    inputs=read(Path(plan['outdir'])/'inputs.json')
    gate=read(Path(plan['outdir'])/'gpu_gate.json')
    expected=checkpoint_binding(plan,method,inputs,gate['execution'],inputs['poison'][method])
    saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    validate_checkpoint(saved,expected,method)
    if saved['epoch']!=49 or saved['phase']!='victim':
        raise ValueError('A completion report cannot replace a final49 checkpoint')
    return row


def gate_check(plan):
    inputs=verify_inputs(plan); gate=read(Path(plan['outdir'])/'gpu_gate.json')
    if (gate['plan_sha256']!=plan['plan_sha256'] or gate['inputs_sha256']!=sha(Path(plan['outdir'])/'inputs.json')
            or gate['execution']!=gpu() or gate['jobs_per_gpu']!=1):
        raise ValueError('Missing/foreign prelaunch gate')
    for method,path in gate['records'].items():
        record=Path(plan['outdir'])/path['path']; row=read(record)
        if (sha(record)!=path['sha256'] or row['batch_size']!=32 or row['steps']!=2
                or row['execution']!=gate['execution'] or not row['real_input']
                or row['method']!=method or row['plan_sha256']!=plan['plan_sha256']
                or row['poison']!=inputs['poison'][method]):
            raise ValueError('Real CUDA smoke evidence changed')
    if set(gate['records'])!=set(METHODS): raise ValueError('All three smokes required')
    return inputs


def command(plan,work,method):
    return [sys.executable,'-u',str(Path(__file__).resolve()),'--data-home',plan['home'],
            '--outdir',plan['outdir'],'--device','cuda:0','--jobs-per-gpu','1',
            '--mode','cell','--work',work,'--method',method]


def child(plan,work,method):
    root=Path(plan['outdir']); folder=root/method; folder.mkdir(exist_ok=True)
    log=folder/'console.log' if work=='train' else root/'smoke'/f'{method}.{uuid.uuid4().hex}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    print(f'[{work}] {method} log={log}',flush=True)
    with log.open('a' if work=='train' else 'x',encoding='utf-8') as stream:
        result=subprocess.run(command(plan,work,method),stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        tail=log.read_text(encoding='utf-8',errors='replace').splitlines()[-35:]
        print('\n'.join(tail),flush=True)
        raise RuntimeError(f'{method} {work} failed; no retry or next job. Log={log}')


def inspect(plan):
    import importlib.util
    import shutil
    cfg=plan['cfg']['badnets']
    print(json.dumps(dict(profile=PROFILE,python=sys.executable,runtime=plan['runtime'],
        dataset_root=cfg['dataset_root'],dataset_exists=Path(cfg['dataset_root']).is_dir(),
        action_exists=Path(cfg['action_npy']).is_file(),
        action_sha256=sha(cfg['action_npy']) if Path(cfg['action_npy']).is_file() else None,
        free_GiB=round(shutil.disk_usage(Path(plan['home'])).free/2**30,2),
        dependencies={m:importlib.util.find_spec(m) is not None for m in
                      ('torch','torchvision','numpy','scipy','yaml','tqdm')},
        methods=METHODS,jobs_per_gpu=1,full_training_started=False),indent=2))
    report=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
    print(report.stdout or report.stderr)


def status(plan):
    for method in METHODS:
        folder=Path(plan['outdir'])/method; log=folder/'console.log'
        print(method,'complete=',completed(plan,method) is not None,'checkpoint=',(folder/'checkpoint.pt').is_file())
        if log.exists(): print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-4:]))


def export(plan):
    rows=[completed(plan,m) for m in METHODS]
    if any(r is None for r in rows): raise ValueError('Three completed evaluations required')
    lines=['# Exploratory MM-Fi / HPE-Li / P3S3 / seed42','',
        '| Method | T-MPJPE mm | Target-PCK % | Mean DCE mm | Clean MPJPE mm | Clean PA mm |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        v=row['metrics']; keys=('t_mpjpe_mm','target_pck_pct','mean_positive_dce_mm','clean_mpjpe_mm','clean_pampjpe_mm')
        lines.append('| '+row['method']+' | '+' | '.join(f'{v[k]:.2f}' for k in keys)+' |')
    (Path(plan['outdir'])/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(Path(plan['outdir'])/'results.json',dict(status='complete',plan_sha256=plan['plan_sha256'],records=rows),immutable=True)


def main(argv=None):
    import shutil
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home',type=Path,default=Path.home()/'backdooranalog')
    parser.add_argument('--outdir',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--jobs-per-gpu',type=int,default=1)
    parser.add_argument('--mode',choices=('inspect','prepare','check','launch','status','cell'),required=True)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--work',choices=('smoke','train'))
    args=parser.parse_args(argv)
    if args.outdir is None: args.outdir=args.data_home/'runs/mmfi_p3s3_three_attacks_s42_4090_v1'
    verify_bundle()
    plan=build_plan(args.data_home,args.outdir,args.device,args.jobs_per_gpu)
    if args.mode=='inspect': inspect(plan); return
    if args.mode=='status': status(plan); return
    if args.mode=='cell':
        if not args.method or not args.work: parser.error('Internal worker requires method/work')
        load_plan(plan)
        if args.work=='smoke':
            verify_inputs(plan)
            write(Path(plan['outdir'])/'smoke'/f'{args.method}.json',smoke(plan,args.method),immutable=True)
        else:
            gate_check(plan); train_cell(plan,args.method)
        return
    from run_mmfi_tables import _cell_lock
    with _cell_lock(Path(plan['outdir'])/'.controller'):
        require_idle()
        if shutil.disk_usage(plan['home']).free<5*2**30: raise ValueError('Need at least5GiB free')
        root=Path(plan['outdir'])
        if args.mode=='prepare':
            audit(plan)
            if (root/'gpu_gate.json').exists(): gate_check(plan)
            else:
                records={}
                for method in METHODS:
                    require_idle(); child(plan,'smoke',method)
                    path=root/'smoke'/f'{method}.json'
                    records[method]=dict(path=path.relative_to(root).as_posix(),sha256=sha(path))
                write(root/'gpu_gate.json',dict(plan_sha256=plan['plan_sha256'],execution=gpu(),
                    inputs_sha256=sha(root/'inputs.json'),jobs_per_gpu=1,records=records),immutable=True)
                gate_check(plan)
            print('PREPARED_ONLY; all three real B32 smokes passed; no full training',flush=True)
        elif args.mode=='check':
            gate_check(plan); print('PRELAUNCH_OK; no full training',flush=True)
        else:
            gate_check(plan)
            for method in METHODS:
                if completed(plan,method) is None:
                    require_idle(); child(plan,'train',method)
                if completed(plan,method) is None: raise RuntimeError('Worker exited without final evaluated checkpoint')
            export(plan); print('COMPLETE3/3; all final49 checkpoints preserved',flush=True)


if __name__=='__main__':
    try: main()
    except Exception as error:
        import traceback
        traceback.print_exc()
        print(f'[P3S3 three attacks] ERROR: {error}',file=sys.stderr,flush=True)
        raise SystemExit(1)
