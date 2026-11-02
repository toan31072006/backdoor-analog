"""Exploratory Adap-Patch / MM-Fi P1S1 / HPE-Li / seed42 / one RTX4090.

Prepare reuses the completed P3S3 content audit without rereading dataset
contents, verifies subset identities/file stats, and runs two B32 CUDA updates.
Launch is a separate foreground invocation.
No historical output, independent Clean victim, or other process is modified.
Optional explicit sharing permits only the exact original P3S3 Adap-Patch peer.
This standalone sidecar never imports or modifies the running P3S3 runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid

os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  MKL_THREADING_LAYER='SEQUENTIAL', PYTHONUTF8='1')
HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_p1s1_hpeli_adap_patch_s42_4090_v1'
METHOD = 'adap_patch'
GRID = (0., .2, .4, .6, .8, 1.)
COUNTS = {'training': 133056, 'test': 33264}
DEFAULT_RUN = 'mmfi_p1s1_hpeli_adap_patch_s42_4090_v1'


def reject_links(path):
    """Reject existing symlink/junction/reparse ancestors before resolving writes."""
    candidate=Path(path).absolute()
    for current in (candidate,*candidate.parents):
        if current.is_symlink():
            raise ValueError(f'Output/input root must not traverse a symbolic link: {current}')
        try:
            info=current.lstat()
        except FileNotFoundError:
            continue
        if getattr(info,'st_file_attributes',0) & 0x400:
            raise ValueError(f'Output/input root must not traverse a reparse point: {current}')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, *, immutable=False):
    from run_mmfi_p3_s3_three_attacks_4090 import write as atomic_json
    atomic_json(path, value, immutable=immutable)


def configuration(home):
    from run_a1_three_tables import _configuration
    from attack.adap_patch_mmfi import source_config
    reference = _configuration(Path(home), 'cuda:0', 0, 42, 'a1')
    cfg = {k: v for k, v in reference.items()
           if k != 'confirmation_profile' and not k.startswith('method_')}
    cfg.update(trigger=METHOD, adap_patch=source_config(),
               confirmation_profile=PROFILE, ckpt_every=1,
               attacker_access='training_data_only_fixed_trigger',
               clean_label_cover_ratio=.2)
    common = ('model', 'seed', 'experiment_name', 'mmfi_protocol', 'mmfi_setting',
              'mmfi_split_seed', 'mmfi_random_ratio', 'optimizer', 'lr', 'momentum',
              'weight_decay', 'batch_size', 'epochs', 'victim_epochs', 'num_workers',
              'pretrained', 'data_parallel', 'rho', 'pivot', 'eps', 'theta_max_deg',
              'payload_axis', 'target_joints', 'dose_mode', 'dose_min', 'dose_max',
              'dose_grid', 'dose_coupling', 'victim_loss', 'poison_select')
    if any(cfg.get(k) != reference.get(k) for k in common):
        raise ValueError('Common P1S1 A1 training/target recipe changed')
    if any(k.startswith('comparison_peak_') for k in cfg):
        raise ValueError('No extra comparison cap is permitted')
    return cfg


def sources():
    # Existing frozen manifest is explicit, not a recursive walk over unrelated
    # untracked research runners that may not be published with this experiment.
    import run_mmfi_p3_splits as p3
    values = p3._sources()
    for name in ('run_mmfi_p1s1_adap_patch_4090.py', 'attack/adap_patch_mmfi.py',
                 'adap_patch_shared_gpu.py',
                 'run_mmfi_p3_s3_three_attacks_4090.py', 'run_mmfi_p1s1_trojannn.py',
                 'run_piw_hpeli_clean.py', 'run_mmfi_tables.py', 'extra_attack_eval.py'):
        values[name] = sha(HERE / name)
    return values


def build_plan(home, outdir, device='cuda:0', jobs=1, allow_concurrent_p3s3=False):
    from attack.adap_patch_mmfi import source_config
    if type(allow_concurrent_p3s3) is not bool:
        raise ValueError('Sharing permission must be boolean')
    reject_links(home); reject_links(outdir)
    home, outdir = Path(home).resolve(), Path(outdir).resolve()
    if device != 'cuda:0' or type(jobs) is not int or jobs != 1:
        raise ValueError('P1S1 fixes one worker on cuda:0; only the explicit exact-P3S3 policy may share')
    # Deliberately do not permit arbitrary existing historical run folders.
    if outdir != home / 'runs' / DEFAULT_RUN:
        raise ValueError(f'This profile requires its dedicated output: runs/{DEFAULT_RUN}')
    plan = dict(schema=1, profile=PROFILE, home=str(home), outdir=str(outdir),
        device=device, jobs_per_gpu=1, method=METHOD, cfg=configuration(home),
        execution_policy=dict(allow_concurrent_p3s3=allow_concurrent_p3s3,
            max_p1s1_workers=1, max_p3s3_workers=1 if allow_concurrent_p3s3 else 0,
            no_speedup_or_memory_guarantee=True),
        sources=sources(), operator=source_config(),
        runtime=dict(python=platform.python_version(), executable=sys.executable,
            system=platform.system(), machine=platform.machine(),
            packages={n: version(n) for n in ('torch', 'numpy', 'scipy', 'PyYAML', 'Pillow')},
            threading={k: os.environ[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MKL_THREADING_LAYER')}),
        contract=dict(table='T1 exploratory single-seed additional baseline',
            dataset='MM-Fi', protocol='protocol1', setting='s1', seed=42,
            counts=COUNTS, victim='hpeli', scratch=True, epochs=50, final_epoch=49,
            input_policy='reuse completed P3S3 content audit; subset identity/stat checks only; no CSI/pose rehash',
            physical_batch=32, drop_last=False, workers=0, amp=False,
            optimizer='SGD', lr=.001, momentum=.9, weight_decay=0.,
            poison_rate=.1, cover_rate=.2, total_modified_rate=.3,
            paired_poison_doses='common fixed U(.2,1)',
            cover_stream='independent SeedSequence([42,7044]); fixed U(.2,1)',
            train_trigger='one source trigger per sorted poison/cover identity group',
            test_trigger='native GTSRB preset: firefox then trojan, alpha1 each times dose',
            source_preset_reason='CIFAR configured test asset missing from pinned author tree; complete native GTSRB preset selected before TEST',
            doses=list(GRID), target='pivot1 Z40*d; joints[2,3]',
            pck_scale=[5,12], pck_threshold=.2, units='mm / percent',
            clean_reference='not trained/reused by this standalone baseline',
            clean_model_trigger_control=False, test_selection=False,
            caveats=['CSI/pose/dose adaptation, not unchanged image reproduction.',
                     '20% covers retain true poses; total modified TRAIN is 30%, not 10%.',
                     'Native trigger strengths are not equal-noise or equal-modified-data A1 budgets.',
                     'Single seed; no across-seed uncertainty or complete main-table claim.',
                     'Foreground worker needs this SSH session kept open; no persistence guarantee.']))
    plan['plan_sha256'] = digest(plan)
    return plan


def load_plan(plan):
    if read(Path(plan['outdir']) / 'plan.json') != plan:
        raise ValueError('Plan/source/runtime changed; preserve output, do not bypass binding')


class Deferred:
    requires_deferred_injection = True


def paired_data(base, cfg):
    from attack.poison import PoisonedDataset
    return PoisonedDataset(base, Deferred(), mode='train', rho=.1,
        dose_min=.2, dose_max=1., eps=cfg['eps'], pivot=1, theta_max_deg=40.,
        dose_mode='linear', axis=cfg['payload_axis'], seed=42, select='uniform',
        dataset='mmfi', dose_coupling='paired', cover_ratio=0.)


def assignments(pairs):
    import numpy as np
    from attack.adap_patch_mmfi import cover_group_plan
    common = pairs.manifest()
    cover_seed = int(np.random.SeedSequence([42,7044]).generate_state(1)[0])
    values = cover_group_plan(len(pairs),
        np.asarray([i for i,d in pairs.poison_plan], dtype=np.int64),
        np.asarray([d for i,d in pairs.poison_plan]), seed=cover_seed, cover_rate=.2)
    serial = {k: v.tolist() for k,v in values.items()}
    record = dict(common_poison_sha256=digest(common),
        poison_plan_sha256=common['poison_plan_sha256'],
        n_total=len(pairs), n_poison=pairs.n_poison,
        n_cover=len(values['cover_ids']), cover_seed=cover_seed,
        assignment_sha256=digest(serial), assignments=serial)
    return record, values


class TrainingRows:
    """Normalize exactly once, then apply static CSI marks; covers retain GT."""
    def __init__(self, pairs, operator):
        self.pairs, self.operator = pairs, operator
        self.record, values = assignments(pairs)
        self.poison = {int(i): (float(d), int(g)) for i,d,g in zip(
            values['poison_ids'], values['poison_doses'], values['poison_groups'])}
        self.cover = {int(i): (float(d), int(g)) for i,d,g in zip(
            values['cover_ids'], values['cover_doses'], values['cover_groups'])}
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, index):
        import torch
        row = self.pairs[index]
        spec = self.poison.get(index, self.cover.get(index))
        if spec is not None:
            dose, group = spec
            row['csi'] = self.operator.inject_train(
                torch.as_tensor(row['csi']).float(), dose, group).numpy()
        # Victim receives only ordinary supervised CSI/pose pairs, never metadata.
        return {'csi': row['csi'], 'pose': row['pose']}


def dataset(plan, split):
    from data_utils.feeder import MMFI
    cfg=plan['cfg']
    value = MMFI(split, cfg['dataset_root'], protocol='protocol1', setting='s1',
                 random_ratio=.8, random_seed=0)
    if len(value) != COUNTS[split]:
        raise ValueError(f'Expected full official {split}: {COUNTS[split]}, found {len(value)}')
    return value


def operator(plan, *, download=False):
    from attack.adap_patch_mmfi import AdapPatchCSI, prepare_assets, load_assets
    location = Path(plan['outdir']) / 'assets'
    reject_links(location)
    return AdapPatchCSI((prepare_assets if download else load_assets)(location))


def split_coverage(data, splits):
    a,b = splits['training']['sequence_keys'], splits['test']['sequence_keys']
    if len(a) != 448 or len(b) != 112 or a & b:
        raise ValueError('Official P1S1 sequence coverage or disjointness changed')
    # P1S1 draws subjects per action: global subject overlap is expected.
    ids=[{(item['name'],item['frame_idx']) for item in data[k].items} for k in COUNTS]
    if ids[0] & ids[1]:
        raise ValueError('P1S1 TRAIN/TEST identity overlap')


def parent_audit(plan):
    """Read original receipts + stat metadata only, never reread CSI/pose bytes."""
    from adap_patch_shared_gpu import P3_RUN, P3_ENTRY, P3_RUNNER_SHA256
    from run_mmfi_p3_s3_three_attacks_4090 import input_stats
    root=Path(plan['home'])/'runs'/P3_RUN
    saved=read(root/'plan.json'); inputs=read(root/'inputs.json'); gate=read(root/'gpu_gate.json')
    claimed=saved.get('plan_sha256')
    if (claimed!=digest({k:v for k,v in saved.items() if k!='plan_sha256'})
            or saved.get('profile')!=P3_RUN
            or saved.get('sources',{}).get(P3_ENTRY)!=P3_RUNNER_SHA256
            or inputs.get('plan_sha256')!=claimed or gate.get('plan_sha256')!=claimed
            or gate.get('inputs_sha256')!=sha(root/'inputs.json')
            or inputs.get('full_content_sha256')!=digest(inputs.get('files',[]))
            or Path(inputs['dataset_root']).resolve()!=Path(plan['cfg']['dataset_root']).resolve()
            or saved.get('contract',{}).get('counts')!={'training':240570,'test':80190}):
        raise ValueError('Missing/foreign P3S3 audit provenance; no automatic full re-audit')
    if input_stats(Path(inputs['dataset_root']),inputs['files'])!=inputs['input_stat_fingerprint']:
        raise ValueError('Files changed since the P3S3 audit; no silent bypass or full re-audit')
    return inputs,dict(mode='reused_original_P3S3_content_audit_plus_current_stat_check',
        path=str(root/'inputs.json'),sha256=sha(root/'inputs.json'),plan_sha256=claimed,
        dataset_content_reread=False)


def audit(plan):
    import run_mmfi_p3_splits as p3
    from run_mmfi_p3_s3_three_attacks_4090 import input_stats
    root = Path(plan['outdir'])
    if (root / 'plan.json').exists():
        return verify_inputs(plan)
    allowed = {'.controller', 'assets'}
    if any(path.name not in allowed for path in root.iterdir()):
        raise ValueError('Nonempty output without matching plan; no overwrite')
    parent,provenance=parent_audit(plan)
    data_root = Path(plan['cfg']['dataset_root']).resolve()
    data = {split: dataset(plan, split) for split in COUNTS}
    splits = {split: p3._identity_digest(ds) for split,ds in data.items()}
    split_coverage(data,splits)
    for value in splits.values():
        value.pop('sequence_keys')
    selected = {}
    for ds in data.values():
        for item in ds.items:
            for key in ('csi', 'kpt'):
                original = Path(item[key])
                path = original.resolve()
                if original.is_symlink() or not path.is_relative_to(data_root):
                    raise ValueError('Dataset file escapes root or is symlinked')
                selected[path.relative_to(data_root).as_posix()] = (path,key)
    audited={row['path']:row for row in parent['files']}
    missing=set(selected)-set(audited)
    if missing:
        raise ValueError(f'P1S1 has {len(missing)} files outside the completed P3S3 audit; no re-audit')
    files=[audited[name] for name in sorted(selected)]
    op = operator(plan)
    poison,_ = assignments(paired_data(data['training'],plan['cfg']))
    receipt = dict(plan_sha256=plan['plan_sha256'], dataset_root=str(data_root),
        splits=splits, files=files, full_content_sha256=digest(files),
        input_stat_fingerprint=input_stats(data_root,files),
        poison=poison, operator=op.state_summary(), no_test_selection=True,parent_audit=provenance)
    write(root/'plan.json',plan,immutable=True)
    write(root/'inputs.json',receipt,immutable=True)
    print('REUSED_P3S3_CONTENT_AUDIT: P1S1 subset identities/stat checks passed; no CSI/pose content rehash',flush=True)
    return receipt


def verify_inputs(plan):
    from run_mmfi_p3_s3_three_attacks_4090 import input_stats
    load_plan(plan)
    row = read(Path(plan['outdir'])/'inputs.json')
    if (row['plan_sha256']!=plan['plan_sha256']
            or row['full_content_sha256']!=digest(row['files'])
            or input_stats(Path(row['dataset_root']),row['files'])!=row['input_stat_fingerprint']
            or row['operator']!=operator(plan).state_summary()):
        raise ValueError('Audited input files/assets changed; no launch')
    provenance=row.get('parent_audit',{})
    if provenance.get('dataset_content_reread') is not False or sha(provenance['path'])!=provenance.get('sha256'):
        raise ValueError('Original reused P3S3 audit receipt changed')
    return row


def gpu():
    from run_mmfi_p3_s3_three_attacks_4090 import gpu as verify_gpu
    return verify_gpu()


def require_idle(plan):
    from adap_patch_shared_gpu import require_allowed_gpu
    return require_allowed_gpu(plan)


def components(plan, device='cuda:0'):
    from run_mmfi_p3_s3_three_attacks_4090 import seed
    from models.factory import build_model
    from train_backdoor import _build_optimizer
    seed()
    model = build_model('hpeli',num_keypoints=17,subcarrier_num=114,
                        dataset='mmfi',pretrained=False).to(device)
    name,opt = _build_optimizer(model,plan['cfg'],'mmfi')
    if name!='sgd':
        raise ValueError('Common victim recipe must use SGD')
    rows = TrainingRows(paired_data(dataset(plan,'training'),plan['cfg']),operator(plan))
    return model,opt,rows


def loader(rows, *, train=True):
    from torch.utils.data import DataLoader
    from attack.poison import collate
    return DataLoader(rows,batch_size=32,shuffle=train,num_workers=0,
                      drop_last=False,collate_fn=collate,pin_memory=train)


def ordinary_step(model,opt,batch,device='cuda:0'):
    import torch
    from train_backdoor import _mpjpe_loss
    model.train(); opt.zero_grad(set_to_none=True)
    output = model(batch['csi'].to(device))
    output = output[0] if isinstance(output,(tuple,list)) else output
    if not torch.isfinite(output).all():
        raise ValueError('Nonfinite victim prediction')
    loss = _mpjpe_loss(output,batch['pose'].to(device))
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite victim loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Nonfinite victim gradient')
    opt.step()
    return float(loss.detach())


def checkpoint_binding(plan,inputs,execution):
    return dict(profile=PROFILE, method=METHOD,plan_sha256=plan['plan_sha256'],
        config_sha256=digest(plan['cfg']),inputs_sha256=sha(Path(plan['outdir'])/'inputs.json'),
        input_stat_fingerprint=inputs['input_stat_fingerprint'],
        operator=inputs['operator'],poison_sha256=digest(inputs['poison']),execution=execution,
        steps_per_epoch=math.ceil(inputs['poison']['n_total']/32))


def validate_checkpoint(blob,binding,*,cuda=True):
    import torch
    if blob.get('binding')!=binding or digest(blob.get('cfg',{}))!=binding['config_sha256']:
        raise ValueError('Checkpoint source/config/data/asset/poison/runtime/GPU binding mismatch')
    epoch=blob.get('epoch')
    if type(epoch) is not int or not 0<=epoch<=49:
        raise ValueError('Checkpoint epoch must be fixed victim index 0..49')
    if (type(blob.get('optimizer_steps')) is not int
            or blob['optimizer_steps']!=(epoch+1)*binding['steps_per_epoch']):
        raise ValueError('Checkpoint optimizer-step count differs from epoch history')
    if not blob.get('model') or not blob.get('optimizer') or not blob.get('rng'):
        raise ValueError('Missing model/optimizer/RNG exact-resume state')
    rng=blob['rng']
    if any(k not in rng for k in ('python','numpy','torch','cuda')) or (cuda and not rng['cuda']):
        raise ValueError('Missing exact CUDA/Python/NumPy/torch RNG state')
    if any(not torch.isfinite(value).all() for value in blob['model'].values() if torch.is_tensor(value)):
        raise ValueError('Nonfinite checkpoint parameter or model buffer')
    groups=blob['optimizer'].get('param_groups',[])
    if len(groups)!=1 or any(groups[0].get(k)!=v for k,v in
        {'lr':.001,'momentum':.9,'weight_decay':0.,'dampening':0.,'nesterov':False}.items()):
        raise ValueError('Checkpoint SGD hyperparameters changed')
    state=blob['optimizer'].get('state',{})
    if not state or any('momentum_buffer' not in entry for entry in state.values()):
        raise ValueError('Missing exact SGD momentum state')
    if any(not torch.isfinite(entry['momentum_buffer']).all() for entry in state.values()):
        raise ValueError('Nonfinite SGD momentum state')


def smoke(plan):
    import torch
    from attack.poison import collate
    inputs=verify_inputs(plan); execution=gpu()
    model,opt,rows=components(plan)
    if rows.record!=inputs['poison']:
        raise ValueError('Smoke poison/cover identities/doses changed')
    # Exercise every native mark, not merely the first four sorted group-0 IDs.
    indices=[next(i for i,(_,g) in rows.poison.items() if g==group) for group in range(4)]
    indices += [next(i for i,(_,g) in rows.cover.items() if g==group) for group in range(4)]
    indices += [i for i in range(len(rows)) if i not in rows.poison and i not in rows.cover][:24]
    if len(indices)!=32 or len(set(indices))!=32:
        raise ValueError('Real B32 smoke needs 4 poison,4 cover,24 clean')
    batch=collate([rows[i] for i in indices])
    before=[p.detach().clone() for p in model.parameters()]
    losses=[ordinary_step(model,opt,batch) for _ in range(2)]
    torch.cuda.synchronize()
    changed=any(not torch.equal(a,b.detach()) for a,b in zip(before,model.parameters()))
    if not changed:
        raise ValueError('Smoke did not change model weights')
    return dict(profile=PROFILE,plan_sha256=plan['plan_sha256'],execution=execution,
        real_input=True,batch_size=32,steps=2,composition={'poison':4,'cover':4,'clean':24},
        losses=losses,weights_changed=changed,poison_sha256=digest(rows.record),
        operator=inputs['operator'],full_training_launched=False)


def gate_check(plan, *, execution=None):
    inputs=verify_inputs(plan); root=Path(plan['outdir'])
    gate=read(root/'gpu_gate.json'); record=read(root/'smoke.json')
    current=gpu() if execution is None else execution
    if (gate.get('plan_sha256')!=plan['plan_sha256'] or gate.get('execution')!=current
            or gate.get('inputs_sha256')!=sha(root/'inputs.json')
            or gate.get('smoke_sha256')!=sha(root/'smoke.json') or gate.get('jobs_per_gpu')!=1
            or record.get('plan_sha256')!=plan['plan_sha256'] or record.get('execution')!=current
            or record.get('real_input') is not True or record.get('batch_size')!=32
            or record.get('steps')!=2 or record.get('weights_changed') is not True
            or record.get('composition')!={'poison':4,'cover':4,'clean':24}
            or record.get('poison_sha256')!=digest(inputs['poison'])
            or record.get('operator')!=inputs['operator']
            or len(record.get('losses',[]))!=2
            or not all(math.isfinite(x) for x in record['losses'])):
        raise ValueError('Missing/foreign/nonfinite real CUDA B32 gate')
    return inputs


def execution_probe(plan):
    """Short-lived GPU check; foreground controller never owns a CUDA context.

    Calling torch.cuda.set_device in the controller would make its Python PID
    look like foreign GPU work to the subsequent worker's fail-closed guard.
    """
    result=subprocess.run(command(plan,'execution'),capture_output=True,text=True,check=True)
    prefix='EXECUTION_JSON='
    rows=[line[len(prefix):] for line in result.stdout.splitlines() if line.startswith(prefix)]
    if len(rows)!=1:
        raise ValueError('GPU execution probe did not produce exactly one receipt')
    return json.loads(rows[0])


def train_cell(plan):
    import torch
    from train_backdoor import _rng_state,_restore_rng_state
    from run_mmfi_p3_s3_three_attacks_4090 import atomic_checkpoint
    inputs=gate_check(plan); execution=gpu()
    folder=Path(plan['outdir'])/METHOD; folder.mkdir(exist_ok=True)
    model,opt,rows=components(plan)
    if rows.record!=inputs['poison']:
        raise ValueError('Training poison/cover identities changed')
    binding=checkpoint_binding(plan,inputs,execution)
    checkpoint=folder/'checkpoint.pt'; start=0; steps=0
    if checkpoint.exists():
        old=torch.load(checkpoint,map_location='cpu',weights_only=False)
        validate_checkpoint(old,binding)
        model.load_state_dict(old['model'],strict=True); opt.load_state_dict(old['optimizer'])
        _restore_rng_state(old['rng']); start=old['epoch']+1; steps=old['optimizer_steps']
        print(f'RESUME next_epoch={start}/49',flush=True)
    for epoch in range(start,50):
        began=time.monotonic(); total=0.; n=0
        for batch in loader(rows):
            loss=ordinary_step(model,opt,batch)
            size=len(batch['csi']); total+=loss*size; n+=size; steps+=1
        if n!=COUNTS['training']:
            raise ValueError('Training did not consume full official TRAIN')
        value=dict(binding=binding,epoch=epoch,cfg=plan['cfg'],model=model.state_dict(),
            optimizer=opt.state_dict(),optimizer_steps=steps,rng=_rng_state(),train_loss=total/n)
        validate_checkpoint(value,binding)
        atomic_checkpoint(checkpoint,value)
        write(folder/'progress.json',dict(profile=PROFILE,plan_sha256=plan['plan_sha256'],
            epoch=epoch,optimizer_steps=steps,status='TRAINING' if epoch<49 else 'TRAINED_NOT_EVALUATED',
            checkpoint_sha256=sha(checkpoint),utc=datetime.now(timezone.utc).isoformat()))
        print(f'epoch {epoch}/49: loss={total/n:.8f} lr={opt.param_groups[0]["lr"]:.2e} t={time.monotonic()-began:.1f}s',flush=True)
    del model,opt,rows
    torch.cuda.empty_cache()
    evaluate(plan)


def export_arrays(path,arrays):
    import numpy as np
    path=Path(path)
    if path.exists():
        with np.load(path,allow_pickle=False) as previous:
            if set(previous.files)!=set(arrays) or any(not np.array_equal(previous[k],v) for k,v in arrays.items()):
                raise ValueError('Existing per-sample result differs; preserve original')
        return
    buffer=io.BytesIO(); np.savez_compressed(buffer,**arrays)
    temporary=path.with_name(path.name+'.saving')
    with temporary.open('xb') as handle:
        handle.write(buffer.getvalue())
    os.replace(temporary,path)


def evaluation_metrics(clean,truth,predictions,targets):
    import numpy as np
    from extra_attack_eval import measure_pair
    from eval import metrics as M
    if len(predictions)!=6 or len(targets)!=6 or not np.array_equal(predictions[0],clean):
        raise ValueError('All six doses and exact dose-zero prediction identity required')
    rows=[]; arrays=[]
    for dose,pred,target in zip(GRID,predictions,targets):
        measured,per_sample=measure_pair(clean,pred,truth,target)
        rows.append(dict(dose=dose,**measured)); arrays.append(per_sample)
    main=dict(t_mpjpe_mm=rows[-1]['t_mpjpe_mm'],target_pck_pct=rows[-1]['target_pck_pct'],
        mean_positive_dce_mm=float(np.mean([r['dce_mm'] for r in rows[1:]])),
        clean_mpjpe_mm=float(M.mpjpe(clean,truth).mean()*1000),
        clean_pampjpe_mm=float(M.pa_mpjpe(clean,truth).mean()*1000))
    return main,rows,arrays


def evaluate(plan):
    import numpy as np
    import torch
    from models.factory import build_model
    from attack.poison import PoisonedDataset
    root=Path(plan['outdir']); folder=root/METHOD; checkpoint=folder/'checkpoint.pt'
    inputs=gate_check(plan); execution=gpu()
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    validate_checkpoint(blob,checkpoint_binding(plan,inputs,execution))
    if blob['epoch']!=49:
        raise ValueError('Evaluation requires fixed final49, never best TEST')
    model=build_model('hpeli',num_keypoints=17,subcarrier_num=114,dataset='mmfi',pretrained=False)
    model.load_state_dict(blob['model'],strict=True); model=model.to('cuda:0').eval()
    data=dataset(plan,'test'); op=operator(plan); cfg=plan['cfg']
    ids=np.asarray([row['name'] for row in data.items],dtype=str)
    @torch.no_grad()
    def predict(dose):
        view=PoisonedDataset(data,Deferred(),mode='trigger@dose',fixed_dose=dose,
            pivot=1,theta_max_deg=40.,dose_mode='linear',axis=cfg['payload_axis'],dataset='mmfi')
        predictions=[]; truths=[]; targets=[]
        for batch in loader(view,train=False):
            x=op.inject_test(batch['csi'].to('cuda:0'),dose)
            output=model(x); output=output[0] if isinstance(output,(tuple,list)) else output
            if tuple(output.shape[1:])!=(1,17,3) or not torch.isfinite(output).all():
                raise ValueError('Nonfinite/changed HPE-Li TEST topology')
            predictions.append(output.cpu().numpy()[:,0]); truths.append(batch['pose'].numpy()[:,0])
            targets.append(batch['target'].numpy()[:,0])
        return np.concatenate(predictions),np.concatenate(truths),np.concatenate(targets)
    predictions=[]; targets=[]; truth=None
    for dose in GRID:
        print(f'[evaluate] {METHOD} dose={dose:g}',flush=True)
        pred,actual,target=predict(dose)
        if truth is not None and not np.array_equal(actual,truth):
            raise ValueError('TEST truth/order changed across doses')
        truth=actual; predictions.append(pred); targets.append(target)
        write(folder/'evaluation_progress.json',dict(status='EVALUATING',dose=dose,
            plan_sha256=plan['plan_sha256'],utc=datetime.now(timezone.utc).isoformat()))
    clean=predictions[0]
    main,rows,per_sample=evaluation_metrics(clean,truth,predictions,targets)
    arrays=dict(names=ids,pair_ids=ids,doses=np.asarray(GRID),truth=truth,pred0=clean,
        predictions=np.stack(predictions),targets=np.stack(targets),
        target_error_mm=np.stack([a['target_error_mm'] for a in per_sample]),
        no_trigger_target_error_mm=np.stack([a['no_trigger_target_error_mm'] for a in per_sample]),
        dce_mm=np.stack([a['dce_mm'] for a in per_sample]),clean_scale_mm=per_sample[0]['clean_scale_mm'])
    sample=folder/'per_sample.npz'; export_arrays(sample,arrays)
    value=dict(status='complete',profile=PROFILE,method=METHOD,plan_sha256=plan['plan_sha256'],
        checkpoint_epoch=49,frames=len(data),metrics=main,dose_rows=rows,
        checkpoint_sha256=sha(checkpoint),per_sample_sha256=sha(sample),
        no_trigger_controls='same poisoned victim, evaluated against each attacker target',
        independent_clean_model_trigger_control=False,
        metric_contract=dict(target_joints=[2,3],pck_scale=[5,12],threshold=.2,
            comparison_dose=1.,DCE='mean five positive doses',units='mm / percent'))
    value['record_sha256']=digest(value)
    write(folder/'evaluation.json',value,immutable=True)
    lines=['# Exploratory MM-Fi / HPE-Li / P1S1 / seed42','','| Method | T-MPJPE mm | Target-PCK % | DCE mm | Clean MPJPE mm | Clean PA mm |',
        '|---|---:|---:|---:|---:|---:|', '| Adap-Patch adapted | '+' | '.join(f'{main[k]:.2f}' for k in
        ('t_mpjpe_mm','target_pck_pct','mean_positive_dce_mm','clean_mpjpe_mm','clean_pampjpe_mm'))+' |','',
        'Single-seed extension; no independent Clean reference trained/reused. 10% poison + 20% true-label covers.',
        'Not an equal-modified-data or equal-noise comparison. Full six-dose poses and same-victim no-trigger controls are retained.']
    report=root/'comparison.md'; text='\n'.join(lines)+'\n'
    if report.exists() and report.read_text(encoding='utf-8')!=text:
        raise ValueError('Existing comparison report differs; preserve original')
    if not report.exists():
        report.write_text(text,encoding='utf-8')
    print(f'EVALUATION_COMPLETE {METHOD}: {main}',flush=True)


def completed(plan):
    import torch
    folder=Path(plan['outdir'])/METHOD
    if not (folder/'evaluation.json').exists():
        return None
    row=read(folder/'evaluation.json')
    body={k:v for k,v in row.items() if k!='record_sha256'}
    if (row.get('status')!='complete' or row.get('profile')!=PROFILE
            or row.get('method')!=METHOD or row.get('plan_sha256')!=plan['plan_sha256']
            or row.get('checkpoint_epoch')!=49 or row.get('frames')!=COUNTS['test']
            or row.get('checkpoint_sha256')!=sha(folder/'checkpoint.pt')
            or row.get('per_sample_sha256')!=sha(folder/'per_sample.npz')
            or row.get('record_sha256')!=digest(body)):
        raise ValueError('Completed result/checkpoint/NPZ binding changed')
    validate_metrics(row)
    inputs=verify_inputs(plan); env=read(Path(plan['outdir'])/'gpu_gate.json')['execution']
    blob=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    validate_checkpoint(blob,checkpoint_binding(plan,inputs,env))
    if blob['epoch']!=49:
        raise ValueError('Complete result lacks final49 checkpoint')
    return row


def validate_metrics(row):
    keys=('t_mpjpe_mm','target_pck_pct','mean_positive_dce_mm','clean_mpjpe_mm','clean_pampjpe_mm')
    metrics=row.get('metrics',{}); doses=row.get('dose_rows',[])
    if (set(metrics)!=set(keys) or len(doses)!=6
            or [d.get('dose') for d in doses]!=list(GRID)
            or any(not isinstance(metrics[k],(float,int)) or isinstance(metrics[k],bool)
                   or not math.isfinite(metrics[k]) or metrics[k]<0 for k in keys)
            or metrics['target_pck_pct']>100):
        raise ValueError('Missing/nonfinite/full-six-dose completed metrics')
    dose_keys=('t_mpjpe_mm','target_pck_pct','no_trigger_t_mpjpe_mm','no_trigger_target_pck_pct','dce_mm')
    for value in doses:
        if any(not isinstance(value.get(k),(int,float)) or isinstance(value[k],bool)
               or not math.isfinite(value[k]) or value[k]<0 for k in dose_keys):
            raise ValueError('Nonfinite/missing per-dose metrics')
        if value['target_pck_pct']>100 or value['no_trigger_target_pck_pct']>100:
            raise ValueError('PCK must be a percentage in [0,100]')
    if (doses[0]['dce_mm']!=0
            or metrics['t_mpjpe_mm']!=doses[-1]['t_mpjpe_mm']
            or metrics['target_pck_pct']!=doses[-1]['target_pck_pct']
            or not math.isclose(metrics['mean_positive_dce_mm'],sum(d['dce_mm'] for d in doses[1:])/5,abs_tol=1e-9)):
        raise ValueError('Summary metrics do not replay from positive dose rows')


def command(plan,work):
    argv=[sys.executable,'-u',str(Path(__file__).resolve()),'--data-home',plan['home'],
        '--outdir',plan['outdir'],'--device','cuda:0','--jobs-per-gpu','1',
        '--mode','cell','--work',work]
    if plan['execution_policy']['allow_concurrent_p3s3']:
        argv.append('--allow-concurrent-p3s3')
    return argv


def child(plan,work):
    root=Path(plan['outdir']); folder=root/METHOD; folder.mkdir(exist_ok=True)
    log=folder/'console.log' if work=='train' else root/f'smoke.{uuid.uuid4().hex}.log'
    print(f'[{work}] LOG={log}',flush=True)
    with log.open('a' if work=='train' else 'x',encoding='utf-8') as handle:
        result=subprocess.run(command(plan,work),stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-35:]),flush=True)
        raise RuntimeError(f'{work} exited {result.returncode}; no retry, no next job. LOG={log}')


def inspect(plan):
    import importlib.util
    print(json.dumps(dict(profile=PROFILE,cfg=plan['cfg'],operator=plan['operator'],
        runtime=plan['runtime'],dataset_exists=Path(plan['cfg']['dataset_root']).is_dir(),
        dependencies={n:importlib.util.find_spec(n) is not None for n in ('torch','numpy','scipy','PIL','yaml')},
        no_network=True,no_gpu=True,full_training_launched=False),indent=2))


def status(outdir):
    # No build_plan/import torch/checkpoint load/CUDA/network; metadata only.
    root=Path(outdir); folder=root/METHOD
    result=dict(profile=PROFILE,output=str(root),worker='UNKNOWN (metadata is not liveness proof)',
                checkpoint_exists=(folder/'checkpoint.pt').is_file())
    for name in ('progress.json','evaluation_progress.json','evaluation.json'):
        path=folder/name
        if path.exists():
            try:
                result[name]=read(path)
            except (ValueError,OSError) as error:
                result[name]={'read_error':str(error)}
    print(json.dumps(result,indent=2))
    log=folder/'console.log'
    if log.exists():
        print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-10:]))
    print('READ_ONLY_STATUS; old log/PID/checkpoint existence does not prove live training',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home',type=Path,default=Path.home()/'backdooranalog')
    parser.add_argument('--outdir',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--jobs-per-gpu',type=int,default=1)
    parser.add_argument('--mode',choices=('inspect','dry-run','prepare','check','launch','status','cell'),required=True)
    parser.add_argument('--work',choices=('smoke','train','execution'))
    parser.add_argument('--allow-concurrent-p3s3',action='store_true',
                        help='Authorize only one exact original P3S3 Adap-Patch peer')
    args=parser.parse_args(argv)
    outdir=args.outdir or args.data_home/'runs'/DEFAULT_RUN
    if args.mode=='status':
        status(outdir); return
    plan=build_plan(args.data_home,outdir,args.device,args.jobs_per_gpu,args.allow_concurrent_p3s3)
    if args.mode=='inspect':
        inspect(plan); return
    if args.mode=='dry-run':
        print(json.dumps(plan,indent=2)); print('DRY_RUN_ONLY; no network/GPU/output writes'); return
    from run_mmfi_tables import _cell_lock
    if args.mode=='cell':
        if args.work is None:
            parser.error('Internal cell requires --work')
        load_plan(plan)
        if args.work=='execution':
            require_idle(plan)
            print('EXECUTION_JSON='+json.dumps(gpu(),sort_keys=True),flush=True)
            return
        with _cell_lock(Path(plan['outdir'])/METHOD/'.worker'):
            require_idle(plan)
            if args.work=='smoke':
                write(Path(plan['outdir'])/'smoke.json',smoke(plan),immutable=True)
            else:
                train_cell(plan)
        return
    with _cell_lock(Path(plan['outdir'])/'.controller'):
        require_idle(plan); root=Path(plan['outdir'])
        if args.mode in ('prepare','launch') and shutil.disk_usage(plan['home']).free<5*2**30:
            raise ValueError('Need at least 5 GiB free before Prepare/Launch; no old files removed')
        if args.mode=='prepare':
            operator(plan,download=True); audit(plan)
            if not (root/'gpu_gate.json').exists():
                child(plan,'smoke')
                write(root/'gpu_gate.json',dict(plan_sha256=plan['plan_sha256'],execution=read(root/'smoke.json')['execution'],
                    inputs_sha256=sha(root/'inputs.json'),smoke_sha256=sha(root/'smoke.json'),jobs_per_gpu=1),immutable=True)
            gate_check(plan,execution=execution_probe(plan))
            print('PREPARED_ONLY: reused P3S3 audit + subset/stat checks + CUDA B32 smoke; no full training',flush=True)
        elif args.mode=='check':
            gate_check(plan,execution=execution_probe(plan)); print('VERIFIED_PRELAUNCH_GATE; no full training',flush=True)
        elif args.mode=='launch':
            gate_check(plan,execution=execution_probe(plan))
            if completed(plan) is None:
                child(plan,'train')
            if completed(plan) is None:
                raise RuntimeError('Worker exited without bound final49/full TEST completion')
            print('COMPLETE Adap-Patch; final49/checkpoint/full six-dose predictions preserved',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        import traceback
        traceback.print_exc()
        print(f'[MM-Fi P1S1 Adap-Patch] ERROR: {error}',file=sys.stderr,flush=True)
        raise SystemExit(1)
