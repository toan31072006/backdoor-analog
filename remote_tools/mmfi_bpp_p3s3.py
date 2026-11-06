"""Frozen BppAttack-CSI row: HPE-Li/P3S3/seed42, native50/final49.

Reuses train_backdoor and the existing stochastic quantization/cover operator.
Private evaluator changes only split counts/identity enumeration, not metrics.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'ATKBackd'))
PROFILE = 'mmfi_bpp_hpeli_p3s3_s42_native_v1'
COUNTS = {'train':240570,'eval':80190}


def private(name):
    path=REPO/'ATKBackd'/name
    key='_bpp_private_'+uuid.uuid4().hex
    spec=importlib.util.spec_from_file_location(key,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[key]=module
    spec.loader.exec_module(module)
    return module


def cells(home,root):
    from run_a1_native_comparison import _method_config
    from train_backdoor import _resolve_training_config, _validate_training_contract
    cfg=_method_config(Path(home),'cuda:0',0,'bpp_source')
    cfg.update(mmfi_protocol='protocol3',mmfi_setting='s3',mmfi_split_seed=0,
        mmfi_random_ratio=.8,bpp_p3s3_profile=PROFILE,attacker_access='data_only',ckpt_every=1)
    cfg=_resolve_training_config(cfg)
    _validate_training_contract(cfg)
    folder=Path(root)/'bpp_source'
    return [dict(key='bpp_source',seed=42,method_key='p3s3_s42__bpp_source',origin='fresh',
        rho=.1,n_poison=24057,n_cover=24057,cfg=cfg,ckpt_dir=str(folder),run_dir=str(folder),
        eval_cache=str(folder/'eval_cache.json'),artifact_dir=str(Path(root)/'evaluations/bpp_source'))]


def dataset_record(cfg, *, return_test=False):
    from train_backdoor import _load_dataset,_config_fingerprint
    from run_peak_confirmation import _pair_ids
    from run_method_drafts import _sha_json
    import run_mmfi_p3_splits as p3
    from run_mmfi_p3_s3_three_attacks_4090 import sha
    record=dict(schema=1,profile=PROFILE,config_fingerprint=_config_fingerprint(cfg),
                action_file_sha256=sha(cfg['action_npy']))
    sets,identities,data={}, {}, {}
    for key,split in (('train','training'),('eval','test')):
        ds=_load_dataset(cfg,split)
        ids=_pair_ids(ds)
        if len(ds)!=COUNTS[key] or len(ids)!=len(ds) or len(set(map(tuple,ids)))!=len(ds):
            raise ValueError('Bpp requires complete, unique official P3S3 identities')
        geometry=p3._identity_digest(ds)
        sets[key]=geometry['sequence_keys']
        identities[key]=ids
        record[key]=dict(n=len(ids),pair_ids_sha256=_sha_json(ids))
        data[key]=ds
    if (sets['train'] & sets['eval']
            or {x[1] for x in sets['train']} & {x[1] for x in sets['eval']}
            or set(map(tuple,identities['train'])) & set(map(tuple,identities['eval']))):
        raise ValueError('P3S3 subject/sequence/frame leakage')
    if COUNTS=={'train':240570,'eval':80190}:
        if (len(sets['train'])!=810 or len(sets['eval'])!=270
                or {x[0] for x in sets['train']}!={'E01','E02','E03'}
                or {x[0] for x in sets['eval']}!={'E04'}):
            raise ValueError('P3S3 official held-out E04 split differs')
    if return_test:
        return data['eval'],record,identities['eval']
    return record


def audit_input(cell):
    from run_mmfi_p3_s3_three_attacks_4090 import ACTION_SHA
    record=dataset_record(cell['cfg'])
    if record['action_file_sha256']!=ACTION_SHA:
        raise ValueError('Common benchmark data_bend.npy changed')
    return record


def evaluator():
    module=private('a1_three_tables_eval.py')
    module.COUNTS=dict(COUNTS)
    module._dataset_record=lambda cfg:dataset_record(cfg,return_test=True)
    return module


def auditor():
    module=private('run_a1_three_tables.py')
    module.EXPECTED_COUNTS=dict(COUNTS)
    return module


def validate_resume(cell):
    import torch
    import mmfi_low_poison_shared as old
    old.validate_resume(cell)
    blob=torch.load(Path(cell['ckpt_dir'])/'checkpoint.pt',map_location='cpu',weights_only=False)
    trigger=blob.get('trigger') or {}
    if (trigger.get('schema')!=1 or not trigger.get('key_sha256')
            or type(trigger.get('cover_draws')) is not int or trigger['cover_draws']<0
            or not trigger.get('cover_rng')):
        raise ValueError('Bpp stochastic TRAIN donor/cover RNG is missing; no restart')


def execute(cell,plan_hash,directory):
    import torch
    from train_backdoor import train
    from run_mmfi_tables import _cell_lock
    from run_mmfi_p3_s3_three_attacks_4090 import read,write
    root=Path(cell['ckpt_dir'])
    with _cell_lock(root):
        record=audit_input(cell)
        path=root/'three_tables_inputs.json'
        if path.exists():
            if read(path)!=record:
                raise ValueError('Bpp input identities/config changed')
        elif any((root/n).exists() for n in ('checkpoint.pt','eval_cache.json')):
            raise ValueError('Unbound Bpp checkpoint/cache; no import')
        else:
            write(path,record,immutable=True)
        binding=dict(plan_sha256=plan_hash,cell=cell['key'],input=record)
        marker=root/'bpp_started.json'
        if marker.exists():
            if read(marker)!=binding or not (root/'checkpoint.pt').is_file():
                raise ValueError('Interrupted Bpp lacks its bound checkpoint; no epoch0 fallback')
            validate_resume(cell)
        else:
            write(marker,binding,immutable=True)
        model,unused=train(cell['cfg'],ckpt_dir=cell['ckpt_dir'])
        del model,unused
        torch.cuda.empty_cache()
        if audit_input(cell)!=record:
            raise ValueError('Bpp inputs changed during training')
        frozen=auditor()
        run=frozen._read_completed(cell)
        audit=frozen._audit_run(cell,run,record)
        result=evaluator().evaluate_cell(dict(cell,verified_inputs=record),cell['artifact_dir'],cell['cfg']['device'])
        write(root/'scientific_audit.json',audit,immutable=True)
        print('BPP_P3S3_COMPLETE; seed42/final49/six-dose/PCK; clean-reference not bound',flush=True)
        return result


def smoke(cell,barrier):
    import torch
    from train_backdoor import _load_dataset,build_trigger,_build_optimizer,_victim_update
    from attack.poison import PoisonedDataset,collate
    from models.factory import build_model
    cfg=cell['cfg']
    train=_load_dataset(cfg,'training')
    trigger=build_trigger(cfg)
    pairs=PoisonedDataset(train,trigger,mode='train',rho=.1,dose_min=.2,dose_max=1.,
        eps=cfg['eps'],pivot=1,theta_max_deg=40.,dose_mode='linear',axis=cfg['payload_axis'],
        seed=42,select='uniform',dataset='mmfi',dose_coupling='paired',cover_ratio=.1)
    selected=sorted(pairs.poison_idx)[:4]+sorted(pairs.cover_idx)[:4]
    selected += [i for i in range(len(pairs)) if i not in pairs.poison_idx and i not in pairs.cover_idx][:24]
    if len(selected)!=32:
        raise ValueError('Bpp smoke needs physical B32 with4 poison/4 covers/24 clean')
    batch=collate([pairs[i] for i in selected])
    torch.manual_seed(42)
    model=build_model('hpeli',num_keypoints=17,subcarrier_num=114,dataset='mmfi',pretrained=False).to('cuda:0').train()
    name,optimizer=_build_optimizer(model,cfg,'mmfi')
    if name!='sgd' or tuple(batch['csi'].shape)!=(32,3,114,10):
        raise ValueError('Native Bpp SGD/B32 smoke required')
    torch.cuda.reset_peak_memory_stats(0)
    before=next(model.parameters()).detach().clone()
    barrier('models_resident')
    losses=[_victim_update(model,optimizer,batch['csi'].to('cuda:0'),batch['pose'].to('cuda:0')) for _ in range(2)]
    if not all(math.isfinite(v) for v in losses) or torch.equal(before,next(model.parameters())):
        raise ValueError('Bpp CUDA smoke failed')
    barrier('B64_backward_resident')
    barrier('B32_complete')
    return dict(status='CAPACITY_ONLY_NOT_TRAINED',discarded=True,real_train_input=True,
        batch=32,steps=2,poison_examples=4,cover_examples=4,losses=losses,
        cover_draws=trigger.cover_draws,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
