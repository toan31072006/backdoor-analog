"""New seed42 MM-Fi defense sidecars; immutable A1 victims, full paired TEST.

FT/ANP/NAD/CLP plus mandatory no-defense baseline, never FT-SAM or FP.
Historical Windows manifests are not relocated or rewritten by this runner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import sys

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

HERE = Path(__file__).resolve().parent
PROFILE = 'vast_mmfi_seed42_cross_backbone_defense_v1'
MANIFEST = 'vast_seed42_defense.resolved.json'
DEFENSES = ('none','ft','anp','nad','clp')
BACKBONES = ('metafiplusplus','wiflow')
ALLOWED_BACKBONES = (*BACKBONES,'hpeli')
GRID = [0.,.2,.4,.6,.8,1.]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): result.update(block)
    return result.hexdigest()


def source_hashes():
    names=('vast_seed42_defense.py','vast_seed42_defense_adapter.py','vast_seed42_memory.py',
        'defend/a1_repair_ops.py','defend/a1_anp.py','defend/a1_anp_sources.json',
        'defend/a1_clp.py','defend/a1_clp_sources.json','defend/a1_defense_eval.py',
        'defend/a1_defense_metrics.py','defend/a1_defense_sources.json')
    return {name:file_sha(HERE/name) for name in names}


def runtime_versions():
    return dict(python=platform.python_version(),system=platform.system(),
        packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','scipy','PyYAML')})


def _backbones(values):
    values=list(values)
    if not values or len(set(values))!=len(values) or set(values)-set(ALLOWED_BACKBONES):
        raise ValueError('Choose unique hpeli/metafiplusplus/wiflow backbones')
    return values


def build_cells(data_home,outdir,attack_outdir,device='cuda:0',*,backbones=BACKBONES,activation_offload='none'):
    from vast_seed42_memory import validate_policy
    validate_policy(activation_offload)
    if not re.fullmatch(r'(?:cpu|cuda:(?:0|[1-9][0-9]*))',device):
        raise ValueError('Use cpu or an explicit logical cuda:N device')
    data_home,outdir,attack_outdir=map(lambda p:Path(p).resolve(),(data_home,outdir,attack_outdir))
    for root in (attack_outdir,HERE.parent,data_home/'datasets',data_home/'actions'):
        if outdir==root or outdir in root.parents or root in outdir.parents:
            raise ValueError('Defense output must be a separate sibling from attack/code/data/actions')
    if outdir==data_home or outdir in data_home.parents:
        raise ValueError('Defense output cannot be a broad data-home ancestor')
    cells = [dict(cell_id=f'{b}__s42__{d}',backbone=b,seed=42,defense=d,
        data_home=str(data_home),outdir=str(outdir),attack_outdir=str(attack_outdir),device=device,
        folder=str(outdir/b/'seed42'/d),
        dependencies=[f'{b}__s42__clean',f'{b}__s42__a1'])
        for b in _backbones(backbones) for d in DEFENSES]
    if activation_offload == 'cpu':
        for cell in cells: cell['activation_offload'] = 'cpu'
    return cells


def build_plan(data_home,outdir,attack_outdir,devices=('cuda:0',),*,backbones=BACKBONES,activation_offload='none'):
    devices=list(devices)
    if (not devices or len(set(devices))!=len(devices)
            or any(re.fullmatch(r'(?:cpu|cuda:(?:0|[1-9][0-9]*))',d) is None for d in devices)):
        raise ValueError('Distinct explicit cpu/cuda:N devices required')
    cells=build_cells(data_home,outdir,attack_outdir,devices[0],backbones=backbones,
                       activation_offload=activation_offload)
    for index,cell in enumerate(cells): cell['device']=devices[index%len(devices)]
    result=dict(schema=1,profile=PROFILE,cells=cells,sources=source_hashes(),runtime=runtime_versions(),
        contract=dict(seed=42,dataset='mmfi',protocol='protocol1-s1',train_count=133056,test_count=33264,
            defenses=list(DEFENSES),repair_methods=4,original_victim_training=False,
            trusted_clean_train_fraction=.05,selection_fraction_within_bank=.2,repair_batch_size=256,
            ft_epochs=100,nad_teacher_epochs=10,nad_student_epochs=10,anp_iterations=2000,clp_u=3.,
            doses=GRID,target_joints=[2,3],target_pck_threshold=.2,scale_joints=[5,12],
            main_metrics=['t_mpjpe_mm','target_pck_pct','clean_mpjpe_mm','triggered_gt_mpjpe_mm'],
            no_test_selection=True,uncertainty='one seed; no across-seed SD/CI',
            no_automatic_batch_shrink=True,memory_at_two_jobs_per_gpu_unverified=True))
    if activation_offload == 'cpu': result['activation_offload'] = 'cpu'
    result['plan_sha256']=digest(result)
    return result


def _cell_static(cell):
    value=copy.deepcopy(cell); value.pop('device',None)
    return dict(schema=1,profile=PROFILE,cell=value,sources=source_hashes(),runtime=runtime_versions())


def source_context(cell):
    """Authenticate fresh same-root Clean+A1; never use old HPE-Li loader on new models."""
    import vast_seed42_attack as attack
    import run_a1_three_tables as frozen
    from a1_three_tables_eval import _artifact_hashes,_checkpoint_metadata
    manifest=attack.load_manifest(Path(cell['attack_outdir'])/attack.MANIFEST)
    a1=attack.cell_for(manifest,f'{cell["backbone"]}__s42__a1')
    clean=attack.cell_for(manifest,f'{cell["backbone"]}__s42__clean')
    records=[attack.completed(c,manifest['sources']) for c in (clean,a1)]
    if any(r is None for r in records): raise ValueError('Both authenticated attack Clean and A1 are required')
    if Path(manifest['data_home']).resolve()!=Path(cell['data_home']).resolve():
        raise ValueError('Defense data home differs from its attack source')
    run=frozen._read_completed(a1)
    _checkpoint_metadata(run)
    inputs=attack.current_inputs(a1)
    audit=frozen._audit_run(a1,run,inputs)
    binding=dict(static=_cell_static(cell),attack_plan_sha256=manifest['plan_sha256'],
        source_cell_id=a1['method_key'],source_artifacts=_artifact_hashes(run),
        clean_record_sha256=digest(records[0]),a1_record_sha256=digest(records[1]),
        inputs=inputs,source_scientific_audit=audit)
    if cell['defense'] not in ('none','clp'):
        from train_backdoor import _load_dataset
        from run_peak_confirmation import _pair_ids
        from defend.a1_defense_eval import select_clean_bank
        train=_load_dataset(run['cfg'],'training')
        if len(train)!=133056: raise ValueError('Defense source requires full official TRAIN')
        bank=select_clean_bank(_pair_ids(train),42,.05,.2)
        binding['clean_bank_sha256']=digest(bank)
    return manifest,a1,run,binding


def _json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def completed(cell):
    folder=Path(cell['folder']); path=folder/'evaluation.json'
    if not path.is_file(): return None
    record=_json(path); checksum=record.pop('record_sha256',None)
    if checksum!=digest(record) or record.get('profile')!=PROFILE or record.get('status')!='complete':
        raise ValueError('Invalid defense record identity/checksum')
    _,_,_,binding=source_context(cell)
    if (record.get('binding')!=binding or record.get('cell_id')!=cell['cell_id']
            or record.get('schema')!=1 or record.get('backbone')!=cell['backbone']
            or record.get('seed')!=42 or record.get('method')!='a1'
            or record.get('defense')!=cell['defense']):
        raise ValueError('Defense source/runtime/config/data changed')
    plan=_json(folder/'cell.plan.json')
    if plan.get('binding')!=binding or plan.get('execution')!=record.get('execution'):
        raise ValueError('Defense execution plan changed')
    if record.get('execution',{}).get('device')!=cell['device']:
        raise ValueError('Defense logical device differs from its frozen placement')
    for name,sha in record.get('artifacts',{}).items():
        if name not in ('per_sample.npz','repair.pt','clean_bank.json') or file_sha(folder/name)!=sha:
            raise ValueError('Defense artifact changed')
    expected={'per_sample.npz'}|({'repair.pt'} if cell['defense']!='none' else set())
    if cell['defense'] not in ('none','clp'): expected.add('clean_bank.json')
    if set(record.get('artifacts',{}))!=expected: raise ValueError('Missing defense artifacts')
    if 'clean_bank.json' in expected and (record.get('bank_sha256')!=digest(_json(folder/'clean_bank.json'))
            or record['bank_sha256']!=binding.get('clean_bank_sha256')):
        raise ValueError('Clean TRAIN bank hash changed')
    if record.get('repair',{}).get('complete') is not True: raise ValueError('Incomplete repair metadata')
    import numpy as np
    from defend.a1_defense_metrics import enrich_evaluation
    with np.load(folder/'per_sample.npz',allow_pickle=False) as saved:
        arrays={name:saved[name] for name in saved.files}
    ids=arrays.get('pair_ids')
    if (ids is None or ids.shape!=(33264,3)
            or not np.array_equal(arrays.get('sample_indices'),np.arange(33264))):
        raise ValueError('Defense lacks full ordered paired TEST identities')
    identities=[[str(csi),str(pose),int(frame)] for csi,pose,frame in ids.tolist()]
    if digest(identities)!=binding['inputs']['eval']['pair_ids_sha256']:
        raise ValueError('Defense ordered TEST identities changed')
    replay=enrich_evaluation(dict(metrics=record['metrics'],dose_rows=record['dose_rows'],
        diagnostics=record['diagnostics'],arrays=arrays))
    if replay['metrics']!=record['metrics'] or replay['dose_rows']!=record['dose_rows']:
        raise ValueError('Defense metrics do not replay from paired sample arrays')
    if cell['defense']!='none':
        import torch
        blob=torch.load(folder/'repair.pt',map_location='cpu',weights_only=False)
        expected_hash=digest(dict(binding=binding,execution=record['execution']))
        if blob.get('binding_sha256')!=expected_hash or blob.get('complete') is not True:
            raise ValueError('Incomplete or differently bound repair checkpoint')
    record['record_sha256']=checksum
    return record


def _load_model(cell,run):
    import torch
    from train_backdoor import _config_fingerprint
    from vast_seed42_adapter import build_model
    cfg=run['cfg']; blob=torch.load(Path(run['folder'])/'checkpoint.pt',map_location='cpu',weights_only=False)
    if (blob.get('epoch')!=49 or blob.get('cfg_fingerprint')!=run['fingerprint']
            or _config_fingerprint(blob.get('cfg',{}))!=run['fingerprint']
            or cfg.get('model')!=cell['backbone'] or cfg.get('seed')!=42):
        raise ValueError('Original A1 checkpoint final epoch/backbone/seed/config changed')
    model=build_model(cell['backbone'],num_keypoints=17,num_coor=3,num_person=1,
                      subcarrier_num=114,dataset='mmfi',pretrained=False)
    model.load_state_dict(blob['model'],strict=True)
    return model,blob.get('trigger')


def execute_cell(cell):
    import numpy as np
    import torch
    import vast_seed42_defense_adapter as adapter
    from run_mmfi_tables import _cell_lock,_atomic_json
    from eval_a1_target_pck import _atomic_bytes,_restore_trigger
    from run_piw_hpeli_clean import _setup_execution
    from run_peak_confirmation import _pair_ids
    from train_backdoor import _load_dataset
    from defend.a1_defense_eval import select_clean_bank,clean_loaders,evaluate_defense
    from defend.a1_defense_metrics import enrich_evaluation
    folder=Path(cell['folder'])
    with _cell_lock(folder):
        cached=completed(cell)
        if cached is not None:
            print(f'[Vast defense] VERIFIED {cell["cell_id"]}; no repair/evaluation',flush=True)
            return cached
        _,a1,run,binding=source_context(cell)
        execution=_setup_execution(cell['device'])
        if cell.get('activation_offload') == 'cpu': execution['activation_offload'] = 'cpu'
        plan=dict(binding=binding,execution=execution)
        if (folder/'cell.plan.json').exists():
            if _json(folder/'cell.plan.json')!=plan:
                raise ValueError('Defense execution changed; do not override the saved checkpoint')
        else: _atomic_json(folder/'cell.plan.json',plan)
        cfg=copy.deepcopy(run['cfg']); cfg.update(device=cell['device'],num_workers=0)
        model,state=_load_model(cell,run)
        architecture=adapter.architecture_audit(model,cell['backbone'],cell['defense'])
        trigger=_restore_trigger(cfg,state)
        fit=selection=None; bank=None
        if cell['defense'] not in ('none','clp'):
            train=_load_dataset(cfg,'training')
            if len(train)!=133056: raise ValueError('Full audited TRAIN is required')
            bank=select_clean_bank(_pair_ids(train),42,.05,.2)
            if digest(bank)!=binding.get('clean_bank_sha256'):
                raise ValueError('Repair clean-bank identity differs from its scientific binding')
            path=folder/'clean_bank.json'
            if path.exists() and _json(path)!=bank: raise ValueError('Clean TRAIN bank changed')
            if not path.exists(): _atomic_json(path,bank)
            fit,selection=clean_loaders(train,bank,cfg,42,batch_size=256)
        repair_binding=digest(plan)
        model=model.to(cell['device'])
        model,metadata=adapter.repair(model,cell['backbone'],cell['defense'],fit,selection,
            cell['device'],folder/'repair.pt',repair_binding,42,
            **({'activation_offload':'cpu'} if cell.get('activation_offload') == 'cpu' else {}))
        if not metadata.get('complete'): raise ValueError('Repair did not finish')
        if cell['defense']=='clp':
            from defend.a1_repair_ops import _atomic_save,_state
            blob=dict(schema=PROFILE,defense='clp',binding_sha256=repair_binding,complete=True,
                selected_model_state=_state(model),metadata=metadata)
            if (folder/'repair.pt').exists():
                saved=torch.load(folder/'repair.pt',map_location='cpu',weights_only=False)
                if (saved.get('binding_sha256')!=repair_binding or saved.get('metadata')!=metadata
                        or not saved.get('complete') or set(saved['selected_model_state'])!=set(blob['selected_model_state'])
                        or any(not torch.equal(saved['selected_model_state'][n],v) for n,v in blob['selected_model_state'].items())):
                    raise ValueError('CLP cached state differs from exact fixed-u replay')
            else: _atomic_save(blob,folder/'repair.pt')
        test=_load_dataset(cfg,'validation')
        ids=_pair_ids(test)
        if len(ids)!=33264 or digest(ids)!=binding['inputs']['eval']['pair_ids_sha256']:
            raise ValueError('Defense full ordered TEST differs from source victim')
        measured=enrich_evaluation(evaluate_defense(model,test,trigger,cfg,cell['device']))
        measured['arrays']['pair_ids']=np.asarray(ids,dtype=str)
        if cell['defense']=='none':
            from eval_a1_target_pck import _assert_replay
            replay=dict(measured['metrics'],**measured['diagnostics'])
            replay['t1_mpjpe_mm']=measured['metrics']['t_mpjpe_mm']
            _assert_replay(run['res'],replay,.05,attack=True)
        if source_context(cell)[3]!=binding: raise ValueError('Immutable attack source changed during repair/evaluation')
        buffer=io.BytesIO(); np.savez(buffer,**measured.pop('arrays'))
        _atomic_bytes(folder/'per_sample.npz',buffer.getvalue())
        artifacts={'per_sample.npz':file_sha(folder/'per_sample.npz')}
        if cell['defense']!='none': artifacts['repair.pt']=file_sha(folder/'repair.pt')
        if bank is not None: artifacts['clean_bank.json']=file_sha(folder/'clean_bank.json')
        record=dict(schema=1,profile=PROFILE,status='complete',cell_id=cell['cell_id'],
            backbone=cell['backbone'],seed=42,method='a1',defense=cell['defense'],binding=binding,
            execution=execution,repair=metadata,architecture=architecture,
            bank_sha256=digest(bank) if bank else None,artifacts=artifacts,**measured,
            limitations=['HPE/CSI loss and architecture adaptations, not unchanged classification reproduction.',
                'One seed; no across-seed SD/CI. All TEST metrics at fixed protocol; no TEST selection.',
                'Two concurrent physical-batch256 repairs may not fit; no automatic shrink or fallback.',
                'Low Target-PCK alone is not recovery; report clean and triggered-to-true errors.',
                'CLP covers only declared direct Conv2d-BN2d pairs; BN1d/similarity/fusion exceptions disclosed.'])
        record['record_sha256']=digest(record)
        _atomic_json(folder/'evaluation.json',record)
        verified=completed(cell)
        print(f'[Vast defense] COMPLETE {cell["cell_id"]}; {folder/"evaluation.json"}',flush=True)
        return verified


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--attack-outdir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--backbones',nargs='+',default=list(BACKBONES))
    parser.add_argument('--activation-offload', choices=('none','cpu'), default='none')
    parser.add_argument('--cell')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--check-only',action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    plan=build_plan(args.data_home,args.outdir,args.attack_outdir,[args.device],backbones=args.backbones,
                    activation_offload=args.activation_offload)
    cells=[c for c in plan['cells'] if args.cell is None or c['cell_id']==args.cell]
    if not cells: raise ValueError('Unknown defense cell')
    if args.dry_run:
        print('[Vast defense] DRY RUN: four repairs + mandatory none per backbone; no data/checkpoints/GPU/output writes.')
        print(json.dumps(dict(plan_sha256=plan['plan_sha256'],cells=[c['cell_id'] for c in cells],contract=plan['contract']),indent=2))
        return 0
    if args.check_only:
        missing=[c['cell_id'] for c in cells if completed(c) is None]
        print(json.dumps(dict(complete=not missing,missing=missing)))
        return int(bool(missing))
    if args.cell is None: raise ValueError('Explicit --cell required; root controller owns multi-GPU scheduling')
    execute_cell(cells[0])
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,RuntimeError,OSError) as exc:
        print(f'[Vast defense] ERROR: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(1)
